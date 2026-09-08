#!/usr/bin/env python3
"""
Textual TUI for sot/sot_dc_characterization.py
=============================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-07

4-probe channel resistance on the Keithley 4200A alone. Edit the current
range / reversal / averaging / repeats without touching the dataclasses; the
sidebar shows the point count and estimated run time and flags a bad setup
as you type.

Two Stage-0 uses, same form:
  * baseline R_xx      — set min == max, small probe current, repeats ~10
  * safe-current ramp  — a real sweep, repeats 1, watch R for drift

Run:  python sot_dc_characterization_tui.py
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.validation import Number
from textual.widgets import (
    Button, Collapsible, DataTable, Footer, Header, Input, Label,
    ProgressBar, RichLog, Select, Static, Switch,
)

from sot.sot_dc_characterization import (
    AcquisitionConfig,
    CurrentPoint,
    Keithley4200AConfig,
    SMUChannelConfig,
    TemperatureControllerConfig,
    configure_smu,
    connect_4200a,
    connect_temperature_controller,
    run_measurement,
    set_source_level,
    shutdown_4200a,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import linear_sweep, safe_shutdown
from instruments.data_dir import DataDirPickerScreen, validate_directory
from instruments.data_naming import (
    TEST_SAMPLE,
    RunContext,
    allocate_run,
    ensure_sample,
    finalize_index_row,
    make_incremental_writer,
    preview_raw_filename,
    proc_path,
    write_record,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
    NewSampleScreen,
    StatusCommentScreen,
    sample_options,
)

log = logging.getLogger("sot_dc_characterization_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "sot_dc_characterization_tui_settings.json"

MEASUREMENT_TYPE = "RXX"

SOT_DCCHAR_DESCRIPTION = (
    "4-probe channel resistance R_xx on the Keithley 4200A alone: SMU1 forces "
    "the current, SMU2 (at 0 A) reads the voltage across the inner probes. "
    "Current-reversal (±I) averaging cancels the thermal-EMF offset. Set the "
    "sweep min == max for a fixed-current baseline (repeats ~10), or sweep a "
    "real range (repeats 1) and watch R and the channel voltage for the onset "
    "of heating/drift — that sets the I_max ceiling for the switching runs."
)

DEFAULTS: dict = {
    "k4200_visa_resource": "GPIB0::17::INSTR",
    "integration": "normal",
    "src_channel": "1",
    "sense_channel": "2",
    "compliance_voltage_V": "2.0",
    "source_limit_A": "0.005",
    "four_wire": True,
    "current_min_A": "-0.0001",
    "current_max_A": "0.0001",
    "step_A": "0.0001",
    "bidirectional_sweep": True,
    "reversal_enabled": True,
    "settling_time_s": "0.1",
    "n_averages": "5",
    "n_repeats": "10",
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
}

NUMERIC_FIELDS: dict = {
    "src_channel": int,
    "sense_channel": int,
    "compliance_voltage_V": float,
    "source_limit_A": float,
    "current_min_A": float,
    "current_max_A": float,
    "step_A": float,
    "settling_time_s": float,
    "n_averages": int,
    "n_repeats": int,
}
TEXT_FIELDS = ["k4200_visa_resource", "integration", "device", "cooldown",
               "temperature_visa_resource", "temperature_sensor_uids", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K"]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]


def parse_sensor_uids(raw: str) -> tuple:
    uids = [u.strip() for u in raw.split(",") if u.strip()]
    return tuple(uids[:2])


# ── formatting helpers (per-TUI copies, as elsewhere in the repo) ────────────

def format_si(value: float, unit: str) -> str:
    av = abs(value)
    if av == 0:
        return f"0 {unit}"
    for scale, prefix in ((1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, "")):
        if av < scale * 1000:
            return f"{value / scale:.3f} {prefix}{unit}"
    return f"{value:.3e} {unit}"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ── plan ────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    k4200_cfg: Keithley4200AConfig
    src_cfg: SMUChannelConfig
    sense_cfg: SMUChannelConfig
    acq_cfg: AcquisitionConfig
    currents_A: np.ndarray
    sample: str
    device: str
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    series: str
    temp_cfg: Optional[TemperatureControllerConfig] = None
    data_root: Path = _DEFAULT_DATA_DIR

    @property
    def total_points(self) -> int:
        return len(self.currents_A) * self.acq_cfg.n_repeats


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                        status: str, comment: str) -> dict:
    measured = [r["temperature_1_K"] for r in records if r.get("temperature_1_K") is not None]
    fields = {
        "run": ctx.run_number,
        "timestamp": ctx.timestamp.isoformat(timespec="seconds"),
        "sample": ctx.sample,
        "device": ctx.device,
        "type": MEASUREMENT_TYPE,
        "T_setpoint_K": plan.temperature_setpoint_K,
        "T_K": (sum(measured) / len(measured)) if measured else "",
        "cooldown": plan.cooldown,
        "status": status,
        "comment": comment,
        "series": plan.series,
    }
    fields.update(plan.header_extra)
    return fields


# ── widget helpers (per-TUI copies) ─────────────────────────────────────────

def field(field_id: str, label_text: str, default: str, *, kind: str = "number",
          hint: str = "", validators=None, valid_empty: bool = False) -> list:
    label = Label(label_text, classes="field-label")
    inp = Input(value=default, id=field_id, type=kind, validators=validators, valid_empty=valid_empty)
    widgets = [label, inp]
    if hint:
        widgets.append(Label(hint, classes="hint"))
    widgets[-1].styles.margin = (0, 0, 1, 0)
    return widgets


def switch_field(field_id: str, label_text: str, default: bool) -> Horizontal:
    row = Horizontal(Switch(value=default, id=field_id), Label(label_text, classes="switch-label"),
                     classes="switch-row")
    row.styles.margin = (0, 0, 1, 0)
    return row


def card(title: str, *groups, muted: bool = False) -> Vertical:
    children: list = [Static(title, classes="card-title")]
    for group in groups:
        children.extend(group) if isinstance(group, list) else children.append(group)
    return Vertical(*children, classes="stable-card" if muted else "param-card")


# ── summary ────────────────────────────────────────────────────────────────

def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    info: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    dir_warn, dir_err = validate_directory(state.get("data_dir", ""))
    if dir_err:
        errors.append(f"Data root: {dir_err}")
    elif dir_warn:
        warnings.append(f"Data root: {dir_warn}")
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL:
        errors.append("Choose a sample (or create a new one).")
    if not state.get("device"):
        errors.append("Device is required (e.g. HB3).")

    if state["src_channel"] == state["sense_channel"]:
        errors.append("Source SMU and sense SMU must be different channels (1 vs 2).")
    if state["src_channel"] not in (1, 2) or state["sense_channel"] not in (1, 2):
        errors.append("SMU channels are 1 or 2 on this system.")
    if state["compliance_voltage_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")
    if state["source_limit_A"] <= 0:
        errors.append("Source current software limit must be > 0 A.")

    if state["current_min_A"] > state["current_max_A"]:
        errors.append("Sweep current min must be ≤ max.")
    over = [c for c in (state["current_min_A"], state["current_max_A"])
            if abs(c) > state["source_limit_A"]]
    if over:
        errors.append(f"Sweep bound(s) {over} A exceed the source software limit "
                      f"±{state['source_limit_A']:g} A.")

    if state["step_A"] <= 0:
        errors.append("Sweep step size must be > 0 A.")
        n_one_way = 0
    elif state["current_min_A"] == state["current_max_A"]:
        n_one_way = 1
        info.append(f"Fixed current {format_si(state['current_min_A'], 'A')} "
                    f"(baseline R_xx, {state['n_repeats']} repeats)")
    else:
        n_one_way = max(2, round(abs(state["current_max_A"] - state["current_min_A"])
                                 / state["step_A"]) + 1)
    n_sweep = n_one_way if (not state["bidirectional_sweep"] or n_one_way <= 1) \
        else max(0, 2 * n_one_way - 1)
    total = n_sweep * max(1, state["n_repeats"])
    info.append(f"{n_sweep} current points × {state['n_repeats']} repeats = {total} rows")

    if not state["reversal_enabled"]:
        warnings.append("Current reversal is OFF — the thermal-EMF offset is not cancelled.")

    per_point_s = state["settling_time_s"] + state["n_averages"] * 0.05 * (2 if state["reversal_enabled"] else 1)
    info.append(f"Estimated run time ≈ {format_duration(total * per_point_s)}")

    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        info.append(f"Temperature logged via MercuryiTC ({', '.join(uids) or 'no UID set'})."
                    if uids else "Temperature on but no sensor UID — columns stay empty.")
    else:
        info.append("Temperature logging off.")

    return info, warnings, errors


def compute_filename_preview(state: dict) -> Optional[str]:
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL or not state.get("device"):
        return None
    preview = preview_raw_filename(
        state["sample"], state["device"], MEASUREMENT_TYPE,
        temperature_setpoint_K=state.get("temperature_setpoint_K"))
    return f"{preview}_<timestamp>.csv"


# ── live plot ──────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("SOT DC characterisation — live")
    except Exception:
        pass
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Resistance (Ω)")
    ax.set_title("Live — R_xx vs I")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    (line,) = ax.plot([], [], "o-", color="#2E3192")
    xs: list = []
    ys: list = []

    def _drain(_frame=None):
        changed = False
        while True:
            try:
                rec = queue.get_nowait()
            except Exception:
                break
            xs.append(rec["set_current_A"])
            ys.append(rec["resistance_ohm"])
            changed = True
        if changed:
            line.set_data(xs, ys)
            ax.relim()
            ax.autoscale_view()
        return (line,)

    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path) -> None:
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["set_current_A"] for r in records], [r["resistance_ohm"] for r in records],
            ".-", color="#2E3192")
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Resistance (Ω)")
    ax.set_title("R_xx vs I")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot to '%s'", png_path)


class _LogRelay(logging.Handler):
    def __init__(self, screen: "RunScreen") -> None:
        super().__init__()
        self.screen = screen
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        style = "bold red" if record.levelno >= logging.ERROR \
            else "bold yellow" if record.levelno >= logging.WARNING else ""
        try:
            self.screen.app.call_from_thread(self.screen.write_log, msg, style)
        except Exception:
            pass


# ── run screen ─────────────────────────────────────────────────────────────

class RunScreen(Screen):
    CSS = """
    #status_line { height: 1; padding: 0 1; text-style: bold; }
    #progress { margin: 1 2; }
    #results_table { height: 12; margin: 0 2 1 2; }
    #log { height: 1fr; margin: 0 2 1 2; border: solid $primary; }
    #runactionbar { height: 3; align: center middle; }
    """
    BINDINGS = [
        Binding("a", "abort", "Abort (safe ramp-down)", show=True),
        Binding("q", "back_or_abort", "Abort / Back", show=True),
    ]

    def __init__(self, plan: MeasurementPlan) -> None:
        super().__init__()
        self.plan = plan
        self._stop_event = threading.Event()
        self._measurement_running = True
        self._log_handler: Optional[_LogRelay] = None
        self._records: list[dict] = []
        self._plot_queue: Optional["mp.Queue"] = None
        self._plot_process: Optional[mp.Process] = None
        self._ctx: Optional[RunContext] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Starting …", id="status_line")
        yield ProgressBar(id="progress", total=self.plan.total_points, show_eta=False)
        yield DataTable(id="results_table", zebra_stripes=True, cursor_type="row")
        yield RichLog(id="log", max_lines=5000, markup=False, wrap=True)
        with Horizontal(id="runactionbar"):
            yield Button("Abort (safe ramp-down)", id="abort_btn", variant="error")
            yield Button("Back", id="back_btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#results_table", DataTable).add_columns(
            "rep", "#", "I (A)", "V (V)", "R (Ω)", "V_ch (V)", "T1 (K)")
        self._log_handler = _LogRelay(self)
        logging.getLogger().addHandler(self._log_handler)
        self._start_live_plot()
        self.do_run()

    def on_unmount(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
        if self._plot_process is not None and self._plot_process.is_alive():
            self._plot_process.terminate()

    def _start_live_plot(self) -> None:
        try:
            ctx = mp.get_context("spawn")
            self._plot_queue = ctx.Queue()
            self._plot_process = ctx.Process(target=_live_plot_worker,
                                             args=(self._plot_queue,), daemon=True)
            self._plot_process.start()
        except Exception:
            log.exception("Could not start live plot window")
            self._plot_queue = self._plot_process = None

    def write_log(self, msg: str, style: str) -> None:
        self.query_one("#log", RichLog).write(Text(msg, style=style))

    def _set_status(self, text: str) -> None:
        self.query_one("#status_line", Static).update(text)

    def _set_status_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_status, text)

    def _on_point(self, record: dict) -> None:
        self._records.append(record)
        if self._plot_queue is not None:
            try:
                self._plot_queue.put_nowait(record)
            except Exception:
                pass
        table = self.query_one("#results_table", DataTable)
        t1 = record.get("temperature_1_K")
        table.add_row(
            str(record["repeat_index"]),
            str(record["point_index"] + 1),
            f"{record['set_current_A']:.4e}",
            f"{record['voltage_V']:.4e}",
            f"{record['resistance_ohm']:.6g}",
            f"{record['channel_voltage_V']:.4g}",
            f"{t1:.3f}" if t1 is not None else "—",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(1)
        self._set_status(f"Point {len(self._records)} / {self.plan.total_points}.")

    def _make_on_point(self):
        def _cb(record: dict) -> None:
            self.app.call_from_thread(self._on_point, record)
        return _cb

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False
        self.query_one("#abort_btn", Button).disabled = True
        try:
            if self._ctx is not None:
                png_path = proc_path(self.plan.data_root, self.plan.sample,
                                     self._ctx.run_str, self.plan.device,
                                     MEASUREMENT_TYPE, "R_vs_I")
                _save_measurement_png(self._records, png_path)
        except Exception:
            log.exception("Could not save plot PNG")
        self.app.push_screen(StatusCommentScreen(), self._on_status_comment)

    def _on_status_comment(self, result: Optional[tuple[str, str]]) -> None:
        if result is None or self._ctx is None:
            return
        status, comment = result
        header_fields = build_header_fields(self.plan, self._ctx, self._records,
                                            status=status, comment=comment)
        try:
            if self._records or not self._ctx.raw_path.exists():
                write_record(self._ctx.raw_path, self._records, header_fields)
            finalize_index_row(self.plan.data_root, self._ctx.sample,
                               self._ctx.run_number, header_fields)
        except Exception:
            log.exception("Could not save final status/comment")

    def action_abort(self) -> None:
        if self._measurement_running and not self._stop_event.is_set():
            self._stop_event.set()
            self._set_status("Abort requested — finishing current point, then zeroing the SMUs …")

    def action_back_or_abort(self) -> None:
        if self._measurement_running:
            self.action_abort()
        else:
            self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "abort_btn":
            self.action_abort()
        elif event.button.id == "back_btn":
            self.app.pop_screen()

    @work(thread=True, exclusive=True)
    def do_run(self) -> None:
        plan = self.plan
        k4200 = None
        temp_ctrl = None
        try:
            self._set_status_threadsafe("Connecting to Keithley 4200A (KXCI) …")
            k4200 = connect_4200a(plan.k4200_cfg)
            configure_smu(k4200, plan.src_cfg)
            configure_smu(k4200, plan.sense_cfg)
            set_source_level(k4200, plan.sense_cfg, 0.0)   # park SMU2 as voltmeter

            if plan.temp_cfg is not None:
                self._set_status_threadsafe("Connecting to MercuryiTC …")
                temp_ctrl = connect_temperature_controller(plan.temp_cfg)

            ctx = allocate_run(plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                               temperature_setpoint_K=plan.temperature_setpoint_K,
                               series=plan.series)
            self._ctx = ctx
            plan.acq_cfg.output_file = str(ctx.raw_path)
            write_csv = make_incremental_writer(
                ctx.raw_path,
                lambda records: build_header_fields(plan, ctx, records,
                                                    status="in_progress", comment=""))

            points = [CurrentPoint(current_A=float(i)) for i in plan.currents_A]

            self._set_status_threadsafe("Running measurement …")
            iter_error: Optional[BaseException] = None
            try:
                run_measurement(k4200, plan.src_cfg, plan.sense_cfg, plan.acq_cfg, points,
                                stop_event=self._stop_event, on_point=self._make_on_point(),
                                temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg, write_csv=write_csv)
            except Exception as exc:
                iter_error = exc

            status = "error" if iter_error is not None \
                else ("aborted" if self._stop_event.is_set() else "completed")
            header_fields = build_header_fields(plan, ctx, self._records, status=status, comment="")
            write_record(ctx.raw_path, self._records, header_fields)
            finalize_index_row(plan.data_root, ctx.sample, ctx.run_number, header_fields)
            if iter_error is not None:
                raise iter_error

            final = "Measurement aborted." if self._stop_event.is_set() else "Measurement complete."
        except Exception as exc:
            log.exception("Measurement failed")
            final = f"ERROR: {exc}"
        finally:
            if k4200 is not None:
                safe_shutdown("4200A", lambda: shutdown_4200a(k4200))
            if temp_ctrl is not None:
                safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))
            self.app.call_from_thread(self._on_finished, final)


# ── app / form ─────────────────────────────────────────────────────────────

class SOTDCCharApp(App):
    TITLE = "SOT DC characterisation"
    SUB_TITLE = "Keithley 4200A · 4-probe R_xx · current reversal"

    data_root: Path = _DEFAULT_DATA_DIR

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 44; border-left: solid $primary; padding: 1 2; overflow-y: auto; }
    #identity_bar { height: auto; border: round $accent; padding: 1 2; margin-bottom: 1; }
    #filename_preview { text-style: bold; margin-bottom: 1; }
    #data_dir_row { height: 3; margin-bottom: 1; }
    #data_dir_row Input { width: 1fr; }
    #data_dir_row Button { margin-left: 1; }
    #identity_fields { layout: grid; grid-size: 4; grid-gutter: 0 2; height: auto; }
    #identity_fields > Vertical { height: auto; }
    .param-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; margin-bottom: 1; }
    .param-card { border: solid $primary; padding: 1 2; height: auto; }
    .stable-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }
    Collapsible { height: auto; margin: 1 0; }
    Collapsible > Contents { padding: 1 0 0 1; }
    CollapsibleTitle { text-style: bold; color: $text-muted; }
    .stable-card { border: round $panel-darken-1; padding: 1 2; height: auto; }
    .stable-card .card-title { color: $text-muted; text-style: none; }
    .stable-card .field-label { color: $text-muted; text-style: none; }
    .card-title { text-style: bold underline; margin-bottom: 1; }
    .field { margin-bottom: 1; }
    .field-label { text-style: bold; }
    .hint { text-style: italic; color: $text-muted; }
    .switch-row { height: 3; }
    .switch-row Label { margin-left: 1; content-align: left middle; height: 3; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; }
    .card-desc { color: $text-muted; margin-bottom: 1; }
    #actionbar { height: 3; align: center middle; }
    """

    BINDINGS = [
        Binding("f5", "start", "Start measurement", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                with Vertical(id="identity_bar"):
                    yield Static("", id="filename_preview")
                    with Horizontal(id="data_dir_row"):
                        yield Input(value=str(_DEFAULT_DATA_DIR), id="data_dir",
                                    placeholder="Absolute path to the data root")
                        yield Button("Browse…", id="browse_data_dir")
                    with Vertical(id="identity_fields"):
                        yield Vertical(
                            Label("Sample", classes="field-label"),
                            Select(sample_options(self.data_root), id="sample_select",
                                   allow_blank=False, value=TEST_SAMPLE),
                            classes="field",
                        )
                        yield Vertical(*field("device", "Device (e.g. HB3)",
                                              DEFAULTS["device"], kind="text"), classes="field")
                        yield Vertical(*field("cooldown", "Cooldown (optional)",
                                              DEFAULTS["cooldown"], kind="text"), classes="field")
                        yield Vertical(*field("temperature_setpoint_K", "Temp. setpoint (K, optional)",
                                              DEFAULTS["temperature_setpoint_K"], kind="number",
                                              valid_empty=True, hint="Filename T###K token only."),
                                       classes="field")

                with Vertical(classes="param-grid"):
                    yield card(
                        "Current sweep (4200A SMU1)",
                        field("current_min_A", "Sweep current min (A)", DEFAULTS["current_min_A"]),
                        field("current_max_A", "Sweep current max (A)", DEFAULTS["current_max_A"],
                              hint="min == max → fixed-current baseline R_xx."),
                        field("step_A", "Sweep step size (A)", DEFAULTS["step_A"],
                              validators=[Number(minimum=1e-12, failure_description="must be > 0")]),
                        switch_field("bidirectional_sweep", "Bidirectional (min → max → min)",
                                     DEFAULTS["bidirectional_sweep"]),
                    )
                    yield card(
                        "Acquisition",
                        switch_field("reversal_enabled", "Current reversal (±I)", DEFAULTS["reversal_enabled"]),
                        field("n_averages", "Reversal pairs / samples per point", DEFAULTS["n_averages"],
                              kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("n_repeats", "Repeats of the whole sweep", DEFAULTS["n_repeats"],
                              kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("settling_time_s", "Settling time per step (s)", DEFAULTS["settling_time_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                    )
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature", "Log temperature (MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                    )

                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Keithley 4200A (KXCI)",
                            field("k4200_visa_resource", "KXCI VISA resource",
                                  DEFAULTS["k4200_visa_resource"], kind="text",
                                  hint="GPIB0::17::INSTR  or  TCPIP0::<ip>::1225::SOCKET"),
                            field("integration", "Integration (fast/normal/quiet)",
                                  DEFAULTS["integration"], kind="text"),
                            field("src_channel", "Source SMU channel", DEFAULTS["src_channel"],
                                  kind="integer"),
                            field("sense_channel", "Sense (voltmeter) SMU channel",
                                  DEFAULTS["sense_channel"], kind="integer"),
                            muted=True,
                        )
                        yield card(
                            "SMU limits",
                            field("compliance_voltage_V", "Compliance voltage (V)",
                                  DEFAULTS["compliance_voltage_V"],
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("source_limit_A", "Source current software limit (A)",
                                  DEFAULTS["source_limit_A"],
                                  validators=[Number(minimum=1e-12, failure_description="must be > 0")]),
                            switch_field("four_wire", "4-wire (provenance only over KXCI)",
                                         DEFAULTS["four_wire"]),
                            muted=True,
                        )
                        yield card(
                            "Temperature (MercuryiTC)",
                            field("temperature_visa_resource", "MercuryiTC VISA resource",
                                  DEFAULTS["temperature_visa_resource"], kind="text"),
                            field("temperature_sensor_uids", "Sensor board UID(s)",
                                  DEFAULTS["temperature_sensor_uids"], kind="text",
                                  hint="1-2 UIDs, comma-separated."),
                            muted=True,
                        )

            with Vertical(id="sidebar"):
                yield Static("Description", classes="sidebar-title")
                yield Static(SOT_DCCHAR_DESCRIPTION, classes="card-desc")
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start measurement  (F5)", id="start", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        logging.getLogger().handlers.clear()
        self._load_settings()
        self._set_temperature_fields_enabled(self.query_one("#enable_temperature", Switch).value)
        self.refresh_summary()

    # sample picker
    def _refresh_sample_options(self, *, select_value: Optional[str] = None) -> None:
        select = self.query_one("#sample_select", Select)
        select.set_options(sample_options(self.data_root))
        if select_value is not None:
            select.value = select_value

    def _sync_data_root(self) -> None:
        path = Path(self.query_one("#data_dir", Input).value.strip()).expanduser()
        if not path.is_dir():
            return
        self.data_root = path.resolve()
        opts = [v for _, v in sample_options(self.data_root)]
        cur = self.query_one("#sample_select", Select).value
        self._refresh_sample_options(select_value=cur if cur in opts else TEST_SAMPLE)

    def _browse_data_dir(self) -> None:
        start = self.query_one("#data_dir", Input).value.strip() or str(_DEFAULT_DATA_DIR)
        self.push_screen(DataDirPickerScreen(start), self._on_data_dir_picked)

    def _on_data_dir_picked(self, picked: Optional[str]) -> None:
        if not picked:
            return
        self.query_one("#data_dir", Input).value = picked
        self._sync_data_root()
        self.refresh_summary()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "sample_select":
            return
        if event.value == NEW_SAMPLE_SENTINEL:
            self.push_screen(NewSampleScreen(self.data_root), self._on_new_sample_created)
            return
        self.refresh_summary()

    def _on_new_sample_created(self, result: Optional[str]) -> None:
        self._refresh_sample_options(select_value=result if result else TEST_SAMPLE)
        self.refresh_summary()

    # form I/O
    def _all_field_ids(self) -> list[str]:
        return list(NUMERIC_FIELDS) + TEXT_FIELDS + OPTIONAL_NUMERIC_FIELDS

    def collect_raw(self) -> dict:
        raw: dict = {fid: self.query_one(f"#{fid}", Input).value for fid in self._all_field_ids()}
        for sid in ("bidirectional_sweep", "reversal_enabled", "four_wire", "enable_temperature"):
            raw[sid] = self.query_one(f"#{sid}", Switch).value
        sample_value = self.query_one("#sample_select", Select).value
        if sample_value not in (None, Select.BLANK, NEW_SAMPLE_SENTINEL):
            raw["sample"] = sample_value
        return raw

    def _load_settings(self) -> None:
        try:
            saved = json.loads(SETTINGS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        for fid in self._all_field_ids():
            if fid in saved:
                try:
                    self.query_one(f"#{fid}", Input).value = str(saved[fid])
                except Exception:
                    pass
        for sid in ("bidirectional_sweep", "reversal_enabled", "four_wire", "enable_temperature"):
            if sid in saved:
                self.query_one(f"#{sid}", Switch).value = bool(saved[sid])
        self._sync_data_root()
        saved_sample = saved.get("sample")
        if saved_sample and saved_sample in [v for _, v in sample_options(self.data_root)]:
            self.query_one("#sample_select", Select).value = saved_sample

    def _save_settings(self, raw: dict) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(raw, indent=2))
        except OSError:
            pass

    def parse_state(self) -> tuple[dict, list[str]]:
        errors: list[str] = []
        state: dict = {}
        for fid, caster in NUMERIC_FIELDS.items():
            raw = self.query_one(f"#{fid}", Input).value.strip()
            try:
                state[fid] = caster(raw)
            except ValueError:
                errors.append(f"'{fid}' is not a valid number: {raw!r}")
                state[fid] = 0
        for fid in TEXT_FIELDS:
            state[fid] = self.query_one(f"#{fid}", Input).value.strip()
        for fid in OPTIONAL_NUMERIC_FIELDS:
            raw = self.query_one(f"#{fid}", Input).value.strip()
            if raw:
                try:
                    state[fid] = float(raw)
                except ValueError:
                    errors.append(f"'{fid}' is not a valid number: {raw!r}")
                    state[fid] = None
            else:
                state[fid] = None
        for sid in ("bidirectional_sweep", "reversal_enabled", "four_wire", "enable_temperature"):
            state[sid] = self.query_one(f"#{sid}", Switch).value
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""
        return state, errors

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "data_dir":
            self._sync_data_root()
        self.refresh_summary()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "enable_temperature":
            self._set_temperature_fields_enabled(event.value)
        self.refresh_summary()

    def _set_temperature_fields_enabled(self, enabled: bool) -> None:
        for fid in TEMPERATURE_FIELD_IDS:
            self.query_one(f"#{fid}", Input).disabled = not enabled

    def refresh_summary(self) -> None:
        state, parse_errors = self.parse_state()
        if parse_errors:
            info, warnings, errors, preview = [], [], parse_errors, None
        else:
            info, warnings, errors = build_summary(state)
            preview = compute_filename_preview(state)

        self.query_one("#filename_preview", Static).update(
            f"File:  [bold]{preview}[/bold]" if preview
            else "[dim]File:  (choose a sample and device to preview)[/dim]")

        lines: list[str] = []
        if errors:
            lines.append("[bold red]Blocking issues[/bold red]")
            lines += [f"  [red]✗ {e}[/red]" for e in errors]
        if warnings:
            lines.append("[bold yellow]Warnings[/bold yellow]")
            lines += [f"  [yellow]⚠ {w}[/yellow]" for w in warnings]
        lines.append("[bold]Derived values[/bold]")
        lines += [f"  [dim]•[/dim] {i}" for i in info if i]
        self.query_one("#summary", Static).update("\n".join(lines))
        self.query_one("#start", Button).disabled = bool(errors)

    def action_start(self) -> None:
        state, parse_errors = self.parse_state()
        if parse_errors:
            self.bell()
            return
        _, _, errors = build_summary(state)
        if errors:
            self.bell()
            return
        self.data_root = Path(state["data_dir"]).expanduser()
        ensure_sample(self.data_root, state["sample"], create=True)
        self._save_settings(self.collect_raw())
        self.push_screen(RunScreen(self._build_plan(state)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()
        elif event.button.id == "browse_data_dir":
            self._browse_data_dir()

    def _build_plan(self, state: dict) -> MeasurementPlan:
        k4200_cfg = Keithley4200AConfig(
            visa_resource=state["k4200_visa_resource"],
            integration=state["integration"] or "normal",
        )
        src_cfg = SMUChannelConfig(
            channel=state["src_channel"], source_function="current",
            compliance_voltage_V=state["compliance_voltage_V"],
            four_wire=state["four_wire"], source_limit_A=state["source_limit_A"],
        )
        sense_cfg = SMUChannelConfig(
            channel=state["sense_channel"], source_function="current",  # forces 0 A
            compliance_voltage_V=state["compliance_voltage_V"],
            four_wire=state["four_wire"], source_limit_A=1e-9,
        )
        acq_cfg = AcquisitionConfig(
            settling_time_s=state["settling_time_s"],
            reversal_enabled=state["reversal_enabled"],
            n_averages=state["n_averages"], n_repeats=state["n_repeats"],
            output_file="",
        )
        currents_A = linear_sweep(state["current_min_A"], state["current_max_A"],
                                  state["step_A"], bidirectional=state["bidirectional_sweep"])

        temp_cfg = None
        if state["enable_temperature"]:
            uids = parse_sensor_uids(state["temperature_sensor_uids"])
            if uids:
                temp_cfg = TemperatureControllerConfig(
                    visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

        header_extra = {
            "compliance_voltage_V": state["compliance_voltage_V"],
            "reversal_enabled": state["reversal_enabled"],
            "n_averages": state["n_averages"],
            "n_repeats": state["n_repeats"],
            "four_wire": state["four_wire"],
            "current_sweep_A": [state["current_min_A"], state["current_max_A"], state["step_A"]],
        }
        return MeasurementPlan(
            k4200_cfg=k4200_cfg, src_cfg=src_cfg, sense_cfg=sense_cfg, acq_cfg=acq_cfg,
            currents_A=currents_A, data_root=self.data_root,
            sample=state["sample"], device=state["device"],
            temperature_setpoint_K=state["temperature_setpoint_K"],
            cooldown=state["cooldown"], header_extra=header_extra, series="",
            temp_cfg=temp_cfg,
        )


def main() -> None:
    SOTDCCharApp().run()


if __name__ == "__main__":
    main()
