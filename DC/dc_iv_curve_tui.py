#!/usr/bin/env python3
"""
Textual TUI front-end for dc_iv_curve.py
==========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-31

Lets you edit the parameters that decide whether a DC I-V sweep is good or
bad — current range, compliance, voltmeter integration time, timing —
without touching the dataclasses in the script itself.

The sidebar recomputes derived values (estimated per-point acquisition
time, estimated sweep duration) and flags anything that risks a bad
measurement (source/voltmeter sharing a GPIB address, a degenerate
current range) as you type.

An optional gate voltage (Keithley 2400, off by default) can be held
fixed for the whole sweep, or given a comma-separated list of values —
one complete current sweep runs per value, each saved to its own file and
plotted together in the same window with a different color. The program
runs fine with no 2400 connected as long as the gate stays off.

Run with:
    python dc_iv_curve_tui.py

Requirements:
    pip install textual matplotlib  (in addition to dc_iv_curve.py's own deps)
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.validation import Number
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    Switch,
)

from dc.dc_iv_curve import (
    AcquisitionConfig,
    CurrentPoint,
    GateConfig,
    SourceConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    connect_gate,
    connect_source,
    connect_temperature_controller,
    connect_voltmeter,
    ramp_current_to_zero,
    run_measurement,
    set_gate_voltage,
    shutdown_gate,
    shutdown_source,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import linear_sweep, parse_value_list
from instruments.data_naming import (
    TEST_SAMPLE,
    RunContext,
    allocate_run,
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

log = logging.getLogger("dc_iv_curve_tui")

# Data/settings live outside "bridge" (a sibling of it). _DATA_DIR doubles
# as the data-convention "data root" -- see dc_hall_measurement_tui.py.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DATA_DIR / "dc_iv_curve_tui_settings.json"

# Locked type code (see instruments/data_naming.py) — never deviates.
MEASUREMENT_TYPE = "IV"

DC_IV_DESCRIPTION = (
    "Sweeps a DC current with a Keithley 6221 and records the DC voltage "
    "response with a Keithley 2182 at each point — a direct I-V curve, "
    "swept bidirectionally so hysteresis is visible. Far more informative "
    "than a single-point resistance for anything nonlinear (contacts, "
    "tunnel junctions, diodes, gated 2D systems). No magnet is involved; "
    "the current sweep is the whole measurement. An optional Keithley 2400 "
    "gate voltage (off by default) can be held fixed, or swept through a "
    "list of values — one complete I-V sweep per gate value, plotted "
    "together in different colors."
)


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors dc_iv_curve.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "source_visa_resource": "GPIB0::20::INSTR",
    "voltmeter_visa_resource": "GPIB0::7::INSTR",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "current_min_A": "-0.001",
    "current_max_A": "0.001",
    "nplc": "5",
    "auto_range": True,
    "settling_time_s": "0.2",
    "n_averages": "5",
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    "step_A": "0.00005",
    "bidirectional_sweep": True,
    "enable_gate": False,
    "gate_visa_resource": "GPIB0::25::INSTR",
    "gate_voltage_limit_V": "20.0",
    "gate_compliance_current_A": "1e-6",
    "gate_voltage_values": "0.0",
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
}

# id -> caster, for every free-text numeric field (Switch handled separately)
NUMERIC_FIELDS: dict = {
    "compliance_V": float,
    "source_delay_s": float,
    "current_min_A": float,
    "current_max_A": float,
    "nplc": float,
    "settling_time_s": float,
    "n_averages": int,
    "step_A": float,
    "gate_voltage_limit_V": float,
    "gate_compliance_current_A": float,
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "device",
               "cooldown", "gate_visa_resource", "gate_voltage_values",
               "temperature_visa_resource", "temperature_sensor_uids"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K"]
GATE_FIELD_IDS = ["gate_visa_resource", "gate_voltage_limit_V",
                   "gate_compliance_current_A", "gate_voltage_values"]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]


def parse_sensor_uids(raw: str) -> tuple:
    """Parse a comma-separated "MB1.T1, DB5.T1" field into a 1- or 2-tuple of UIDs."""
    uids = [u.strip() for u in raw.split(",") if u.strip()]
    return tuple(uids[:2])


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_si(value: float, unit: str) -> str:
    """Format a value with an SI prefix, e.g. 1.2e-8 -> '12.000 nA'."""
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


def _reading_duration_s(nplc: float) -> float:
    """Rough per-reading time for the 2182 — NPLC/line_frequency, worst case 50 Hz."""
    return max(1e-3, nplc / 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    acq_cfg: AcquisitionConfig
    currents_A: np.ndarray
    sample: str
    device: str
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    series: str
    gate_cfg: Optional[GateConfig] = None
    gate_voltages: Optional[List[float]] = None
    temp_cfg: Optional[TemperatureControllerConfig] = None

    @property
    def series_values(self) -> List[Optional[float]]:
        """[None] for a single (gate-less or fixed) run, else one entry per gate voltage."""
        return list(self.gate_voltages) if self.gate_voltages else [None]

    @property
    def total_points(self) -> int:
        return len(self.currents_A) * len(self.series_values)


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                         status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """Universal + measurement-specific header/index fields for ONE run
    within this (possibly multi-file, one-per-gate-voltage) session -- see
    dc_spin_valve_tui.py's build_header_fields for the full rationale."""
    measured = [r["temperature_1_K"] for r in records if r.get("temperature_1_K") is not None]
    T_K = (sum(measured) / len(measured)) if measured else ""
    fields = {
        "run": ctx.run_number,
        "timestamp": ctx.timestamp.isoformat(timespec="seconds"),
        "sample": ctx.sample,
        "device": ctx.device,
        "type": MEASUREMENT_TYPE,
        "T_setpoint_K": plan.temperature_setpoint_K,
        "T_K": T_K,
        "cooldown": plan.cooldown,
        "status": status,
        "comment": comment,
        "series": plan.series,
    }
    fields.update(plan.header_extra)
    if extra:
        fields.update(extra)
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# Small widget-building helpers (keep compose() readable)
# ─────────────────────────────────────────────────────────────────────────────

def field(field_id: str, label_text: str, default: str, *, kind: str = "number",
          hint: str = "", validators=None, valid_empty: bool = False) -> list:
    """A field's widgets, flat (not wrapped in a container). Grid cells
    (see card()) that contain a further nested auto-height Vertical break
    Textual's grid auto-row sizing -- GridLayout.arrange() computes an
    'auto' row's height by calling get_content_height() on each cell, and a
    doubly-nested Vertical makes that blow up to ~100 rows instead of the
    handful the content needs. One level of Vertical (the card itself) is
    fine; a Vertical inside that is not -- so fields stay flat and spacing
    is set directly on the last widget instead of via a wrapping container."""
    label = Label(label_text, classes="field-label")
    inp = Input(value=default, id=field_id, type=kind, validators=validators,
                valid_empty=valid_empty)
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
    """A bordered grid cell: a title plus its fields (each a flat list from
    field(), or a single widget like switch_field()'s Horizontal -- see
    field() for why fields must stay flat here). `muted` = stable/rarely
    -changed configuration, styled to recede rather than compete for attention."""
    children: list = [Static(title, classes="card-title")]
    for group in groups:
        children.extend(group) if isinstance(group, list) else children.append(group)
    return Vertical(*children, classes="stable-card" if muted else "param-card")


# ─────────────────────────────────────────────────────────────────────────────
# Live validation / derived-value summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (info, warnings, errors) for a fully-parsed state dict."""
    info: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    # ── Sample / run identity ───────────────────────────────────────────────
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL:
        errors.append("Choose a sample (or create a new one).")
    if not state.get("device"):
        errors.append("Device is required (e.g. HB3, SV2).")

    if state["source_visa_resource"] == state["voltmeter_visa_resource"]:
        errors.append("Source (6221) and voltmeter (2182) VISA resources must be different.")

    if state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")

    if state["current_min_A"] == state["current_max_A"]:
        warnings.append("current_min equals current_max — sweep will repeat a single point.")

    info.append(f"Current range: {format_si(state['current_min_A'], 'A')} → "
                f"{format_si(state['current_max_A'], 'A')}")

    read_s = _reading_duration_s(state["nplc"])
    info.append(f"Estimated 2182 reading time ≈ {read_s * 1000:.0f} ms (NPLC={state['nplc']:g})")

    per_point_s = state["settling_time_s"] + state["n_averages"] * read_s

    if state["step_A"] <= 0:
        errors.append("Sweep step size must be > 0 A.")
        n_one_way = 0
    else:
        n_one_way = max(2, round(abs(state["current_max_A"] - state["current_min_A"]) / state["step_A"]) + 1)
    n_sweep_points = n_one_way if not state["bidirectional_sweep"] else max(0, 2 * n_one_way - 1)
    direction = (f"{state['current_min_A']:g} A → {state['current_max_A']:g} A → {state['current_min_A']:g} A"
                 if state["bidirectional_sweep"]
                 else f"{state['current_min_A']:g} A → {state['current_max_A']:g} A")
    info.append(f"Sweep: {direction}, step={state['step_A']:g} A, {n_sweep_points} points")

    # ── Gate (optional) ─────────────────────────────────────────────────────
    if state["enable_gate"]:
        if state["gate_visa_resource"] in (state["source_visa_resource"], state["voltmeter_visa_resource"]):
            errors.append("Gate (2400) VISA resource must differ from the source/voltmeter resources.")
        if state["gate_voltage_limit_V"] <= 0:
            errors.append("Gate voltage limit must be > 0 V.")
        gate_list = state.get("gate_voltage_list", [])
        if state.get("gate_parse_error"):
            errors.append(f"Gate voltage list: {state['gate_parse_error']}")
        else:
            over_limit = [v for v in gate_list if abs(v) > state["gate_voltage_limit_V"]]
            if over_limit:
                errors.append(
                    f"Gate voltage(s) {over_limit} exceed the configured limit "
                    f"±{state['gate_voltage_limit_V']:g} V."
                )
            n_series = len(gate_list)
            if n_series > 1:
                info.append(f"Gate: {n_series} values {gate_list} — {n_series} complete sweeps, "
                            f"one file each, plotted together")
            else:
                info.append(f"Gate held fixed at {format_si(gate_list[0], 'V')}" if gate_list else "")
            total_points = n_sweep_points * max(1, n_series)
            info.append(f"Estimated total run time ≈ {format_duration(total_points * per_point_s)}")
    else:
        info.append(f"Estimated total run time ≈ {format_duration(n_sweep_points * per_point_s)}")

    # ── Temperature (MercuryiTC, optional) ──────────────────────────────────
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if not uids:
            warnings.append("Temperature logging is on but no sensor UID is set — "
                             "temperature columns will be empty.")
        else:
            info.append(f"Temperature logged via MercuryiTC ({', '.join(uids)}) — "
                         "if unreachable, columns are simply left empty.")
    else:
        info.append("Temperature logging off.")

    return info, warnings, errors


def compute_filename_preview(state: dict) -> Optional[str]:
    """Raw-file name the run will be saved as, or None until sample+device
    are both set -- drives the identity bar's #filename_preview."""
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL or not state.get("device"):
        return None
    preview = preview_raw_filename(
        state["sample"], state["device"], MEASUREMENT_TYPE,
        temperature_setpoint_K=state.get("temperature_setpoint_K"),
    )
    suffix = (" (one file per gate voltage)" if state.get("enable_gate") and
              len(state.get("gate_voltage_list", [])) > 1 else "")
    return f"{preview}_<timestamp>.csv{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────
# A GUI matplotlib backend and Textual's terminal control both want the main
# thread. Rather than fight that, the live preview gets its own process with
# its own main thread; new points are streamed to it over a
# multiprocessing.Queue, tagged with a series_index/series_label so a
# gate-voltage list shows up as one colored trace per value. The final
# combined overlay PNG is saved independently by the TUI process itself
# (see _save_measurement_png), so it doesn't depend on this window still
# being open when the run finishes.

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("DC I-V live measurement")
    except Exception:
        pass
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Live measurement — I-V curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    cmap = plt.get_cmap("tab10")
    lines: dict[int, "plt.Line2D"] = {}
    series_data: dict[int, tuple[list, list]] = {}

    def _drain(_frame=None):
        updated: set[int] = set()
        new_series = False
        while True:
            try:
                record = queue.get_nowait()
            except Exception:
                break
            idx = record.get("series_index", 0)
            if idx not in lines:
                label = record.get("series_label")
                (line,) = ax.plot([], [], "o-", color=cmap(idx % 10), label=label)
                lines[idx] = line
                series_data[idx] = ([], [])
                new_series = True
            xs, ys = series_data[idx]
            xs.append(record["current_A"])
            ys.append(record["voltage_V"])
            updated.add(idx)
        if updated:
            for idx in updated:
                xs, ys = series_data[idx]
                lines[idx].set_data(xs, ys)
            if new_series and any(l.get_label() and not l.get_label().startswith("_") for l in lines.values()):
                ax.legend(loc="best", fontsize=8)
            ax.relim()
            ax.autoscale_view()
        return tuple(lines.values())

    # Keep a reference so it isn't garbage-collected mid-run.
    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path) -> None:
    """Save a combined I-V overlay PNG (one color per gate-voltage series, if
    any) from whatever points were actually collected (including an
    aborted/partial run)."""
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")  # headless — must not touch the TUI's terminal
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5))

    series_ids = sorted({r.get("series_index", 0) for r in records})
    for idx in series_ids:
        rows = [r for r in records if r.get("series_index", 0) == idx]
        label = rows[0].get("series_label")
        ax.plot([r["current_A"] for r in rows], [r["voltage_V"] for r in rows],
                ".-", color=cmap(idx % 10), label=label)

    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Measurement result")
    ax.grid(alpha=0.3)
    if any(r.get("series_label") for r in records):
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot to '%s'", png_path)


# ─────────────────────────────────────────────────────────────────────────────
# Logging -> RichLog relay (keeps raw log lines from corrupting the alt screen)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Run screen  ── executes the plan in a worker thread, shows live progress
# ─────────────────────────────────────────────────────────────────────────────

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
        # Not named `_running` — that attribute already exists on Textual's
        # MessagePump base class and shadowing it silently breaks mounting.
        self._measurement_running = True
        self._log_handler: Optional[_LogRelay] = None
        self._records: list[dict] = []
        self._plot_queue: Optional["mp.Queue"] = None
        self._plot_process: Optional[mp.Process] = None
        # One RunContext per iteration of the gate-voltage series -- each
        # gets its own run number/file (see allocate_run() in do_run below).
        self._run_contexts: list[RunContext] = []

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
            "#", "Vg (V)", "I (A)", "V (V)", "R (Ω)", "T1 (K)", "T2 (K)"
        )
        self._log_handler = _LogRelay(self)
        root = logging.getLogger()
        root.addHandler(self._log_handler)
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
            self._plot_process = ctx.Process(
                target=_live_plot_worker,
                args=(self._plot_queue,),
                daemon=True,
            )
            self._plot_process.start()
        except Exception:
            log.exception("Could not start live plot window (is matplotlib installed?)")
            self._plot_queue = None
            self._plot_process = None

    def write_log(self, msg: str, style: str) -> None:
        self.query_one("#log", RichLog).write(Text(msg, style=style))

    def _set_status(self, text: str) -> None:
        self.query_one("#status_line", Static).update(text)

    def _on_point(self, record: dict) -> None:
        self._records.append(record)
        if self._plot_queue is not None:
            try:
                self._plot_queue.put_nowait(record)
            except Exception:
                pass
        table = self.query_one("#results_table", DataTable)
        gate_V = record.get("gate_voltage_V")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        table.add_row(
            str(record["point_index"] + 1),
            f"{gate_V:.4g}" if gate_V is not None else "—",
            f"{record['current_A']:.4e}",
            f"{record['voltage_V']:.4e}",
            f"{record['resistance_ohm']:.5g}",
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(1)
        self._set_status(f"Point {len(self._records)} / {self.plan.total_points} complete.")

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False
        self.query_one("#abort_btn", Button).disabled = True
        try:
            if self._run_contexts:
                first, last = self._run_contexts[0], self._run_contexts[-1]
                run_label = first.run_str if first is last else f"{first.run_str}-{last.run_str}"
                png_path = proc_path(_DATA_DIR, self.plan.sample, run_label, self.plan.device,
                                      MEASUREMENT_TYPE, "combined", combined=True)
                _save_measurement_png(self._records, png_path)
        except Exception:
            log.exception("Could not save measurement plot PNG")

        # One status/comment prompt for the whole session -- applied to
        # every file in the gate-voltage series.
        self.app.push_screen(StatusCommentScreen(), self._on_status_comment)

    def _on_status_comment(self, result: Optional[tuple[str, str]]) -> None:
        if result is None:
            return
        status, comment = result
        for series_idx, ctx in enumerate(self._run_contexts):
            iter_records = [r for r in self._records if r.get("series_index", 0) == series_idx]
            gate_V = iter_records[0].get("gate_voltage_V") if iter_records else None
            header_fields = build_header_fields(
                self.plan, ctx, iter_records, status=status, comment=comment,
                extra={"gate_voltage_V": gate_V} if gate_V is not None else None,
            )
            try:
                # Never truncate an already-written raw file to an empty stub —
                # only a run that never wrote a point gets a header-only write.
                if iter_records or not ctx.raw_path.exists():
                    write_record(ctx.raw_path, iter_records, header_fields)
                finalize_index_row(_DATA_DIR, ctx.sample, ctx.run_number, header_fields)
            except Exception:
                log.exception("Could not save final status/comment for run %d", ctx.run_number)

    def action_abort(self) -> None:
        if self._measurement_running and not self._stop_event.is_set():
            self._stop_event.set()
            self._set_status("Abort requested — finishing current point, then ramping current to zero …")

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

    def _make_on_point(self, series_index: int, series_label: Optional[str]):
        def _cb(record: dict) -> None:
            record["series_index"] = series_index
            record["series_label"] = series_label
            self.app.call_from_thread(self._on_point, record)
        return _cb

    @work(thread=True, exclusive=True)
    def do_run(self) -> None:
        plan = self.plan
        source = None
        voltmeter = None
        gate = None
        temp_ctrl = None
        try:
            self._set_status_threadsafe("Connecting to Keithley 6221 & 2182 …")
            source = connect_source(plan.src_cfg)
            voltmeter = connect_voltmeter(plan.volt_cfg)

            if plan.temp_cfg is not None:
                self._set_status_threadsafe("Connecting to MercuryiTC (temperature) …")
                temp_ctrl = connect_temperature_controller(plan.temp_cfg)

            if plan.gate_cfg is not None:
                self._set_status_threadsafe("Connecting gate (Keithley 2400) …")
                gate = connect_gate(plan.gate_cfg)

            for series_idx, gate_V in enumerate(plan.series_values):
                if self._stop_event.is_set():
                    break

                label = None
                key_axis = None
                if gate_V is not None:
                    label = f"Vg={gate_V:g}V"
                    key_axis = ("gate_V", gate_V)
                    self._set_status_threadsafe(f"Setting gate to {gate_V:g} V …")
                    set_gate_voltage(gate, plan.gate_cfg, gate_V)

                # A fresh RunContext (own run number, own file) EVERY
                # iteration -- never reuse one across the gate-voltage series.
                ctx = allocate_run(
                    _DATA_DIR, plan.sample, plan.device, MEASUREMENT_TYPE,
                    temperature_setpoint_K=plan.temperature_setpoint_K,
                    key_axis=key_axis, series=plan.series,
                )
                self._run_contexts.append(ctx)
                plan.acq_cfg.output_file = str(ctx.raw_path)
                write_csv = make_incremental_writer(
                    ctx.raw_path,
                    lambda records, _ctx=ctx, _gv=gate_V: build_header_fields(
                        plan, _ctx, records, status="in_progress", comment="",
                        extra={"gate_voltage_V": _gv} if _gv is not None else None,
                    ),
                )

                points = [CurrentPoint(current_A=float(i)) for i in plan.currents_A]

                status = "Running measurement …" if gate_V is None \
                    else f"Running measurement (Vg={gate_V:g} V) …"
                self._set_status_threadsafe(status)
                iter_error: Optional[BaseException] = None
                try:
                    run_measurement(
                        source, voltmeter, plan.src_cfg, plan.acq_cfg, points,
                        stop_event=self._stop_event,
                        on_point=self._make_on_point(series_idx, label),
                        gate_voltage_V=gate_V,
                        temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                        write_csv=write_csv,
                    )
                except Exception as exc:
                    iter_error = exc

                # Finalize THIS iteration's header/index row
                # UNCONDITIONALLY, right now.
                iter_status = "error" if iter_error is not None \
                    else ("aborted" if self._stop_event.is_set() else "completed")
                iter_records = [r for r in self._records if r.get("series_index", 0) == series_idx]
                header_fields = build_header_fields(
                    plan, ctx, iter_records, status=iter_status, comment="",
                    extra={"gate_voltage_V": gate_V} if gate_V is not None else None,
                )
                write_record(ctx.raw_path, iter_records, header_fields)
                finalize_index_row(_DATA_DIR, ctx.sample, ctx.run_number, header_fields)

                if iter_error is not None:
                    raise iter_error

            final = "Measurement aborted." if self._stop_event.is_set() else "Measurement complete."
        except Exception as exc:
            log.exception("Measurement failed")
            final = f"ERROR: {exc}"
        finally:
            if gate is not None:
                try:
                    shutdown_gate(gate)
                except Exception:
                    log.exception("Error while shutting down gate")
            if temp_ctrl is not None:
                try:
                    shutdown_temperature_controller(temp_ctrl)
                except Exception:
                    log.exception("Error while shutting down MercuryiTC")
            if source is not None:
                try:
                    ramp_current_to_zero(source)
                except Exception:
                    log.exception("Error while ramping current to zero")
                try:
                    shutdown_source(source)
                except Exception:
                    log.exception("Error while shutting down source")
            self.app.call_from_thread(self._on_finished, final)

    def _set_status_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_status, text)


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class DCIVCurveApp(App):
    TITLE = "DC I-V Curve"
    SUB_TITLE = "Keithley 6221 + 2182 · current sweep · optional gate"

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 44; border-left: solid $primary; padding: 1 2; overflow-y: auto; }

    #identity_bar { height: auto; border: round $accent; padding: 1 2; margin-bottom: 1; }
    #filename_preview { text-style: bold; margin-bottom: 1; }
    #identity_fields { layout: grid; grid-size: 4; grid-gutter: 0 2; height: auto; }

    .section-title { text-style: bold; color: $text-muted; margin: 1 0; }
    .param-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; margin-bottom: 1; }
    .param-card { border: solid $primary; padding: 1 2; height: auto; }

    .stable-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }
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
                # ── File & run identity ── changes every run, always on top ──
                with Vertical(id="identity_bar"):
                    yield Static("", id="filename_preview")
                    with Vertical(id="identity_fields"):
                        yield Vertical(
                            Label("Sample", classes="field-label"),
                            Select(sample_options(_DATA_DIR), id="sample_select",
                                   allow_blank=False, value=TEST_SAMPLE),
                            classes="field",
                        )
                        yield Vertical(*field("device", "Device (e.g. HB3, SV2)",
                                              DEFAULTS["device"], kind="text"), classes="field")
                        yield Vertical(*field("cooldown", "Cooldown (optional)",
                                              DEFAULTS["cooldown"], kind="text"), classes="field")
                        yield Vertical(*field("temperature_setpoint_K", "Temp. setpoint (K, optional)",
                                              DEFAULTS["temperature_setpoint_K"], kind="number",
                                              valid_empty=True, hint="Filename's T###K token only."),
                                       classes="field")

                # ── Parameters that decide the physics of this run ──────────
                with Vertical(classes="param-grid"):
                    yield card(
                        "Source (Keithley 6221) — current sweep",
                        field("current_min_A", "Sweep current min (A)", DEFAULTS["current_min_A"]),
                        field("current_max_A", "Sweep current max (A)", DEFAULTS["current_max_A"]),
                        field("step_A", "Sweep step size (A)", DEFAULTS["step_A"],
                              validators=[Number(minimum=1e-12, failure_description="must be > 0")]),
                        switch_field("bidirectional_sweep", "Bidirectional (min → max → min)",
                                     DEFAULTS["bidirectional_sweep"]),
                        field("compliance_V", "Compliance voltage (V)", DEFAULTS["compliance_V"],
                              hint="Set high enough to reach the expected voltage at "
                                   "current_max_A, or the sweep clips against compliance.",
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                    )
                    yield card(
                        "Voltmeter (Keithley 2182)",
                        field("nplc", "NPLC (integration time)", DEFAULTS["nplc"],
                              hint="Bigger = quieter but slower.",
                              validators=[Number(minimum=0.01, failure_description="must be > 0")]),
                        switch_field("auto_range", "Auto-range", DEFAULTS["auto_range"]),
                    )
                    yield card(
                        "Acquisition timing",
                        field("settling_time_s", "Settling time per current step (s)",
                              DEFAULTS["settling_time_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        field("n_averages", "Voltage samples averaged per point",
                              DEFAULTS["n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                    )
                    yield card(
                        "Gate voltage (Keithley 2400, optional)",
                        switch_field("enable_gate", "Enable gate", DEFAULTS["enable_gate"]),
                        field("gate_voltage_values", "Gate voltage (V)",
                              DEFAULTS["gate_voltage_values"], kind="text",
                              hint="Single value, or comma-separated list — one complete "
                                   "sweep runs per value, plotted together."),
                    )
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature", "Log temperature (MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                    )

                # ── Instrument wiring & timing constants ── rarely change ───
                yield Static("Instrument configuration", classes="section-title")
                with Vertical(classes="stable-grid"):
                    yield card(
                        "Instrument addresses",
                        field("source_visa_resource", "6221 (current source)",
                              DEFAULTS["source_visa_resource"], kind="text"),
                        field("voltmeter_visa_resource", "2182 (DUT voltage)",
                              DEFAULTS["voltmeter_visa_resource"], kind="text"),
                        field("gate_visa_resource", "2400 (gate)",
                              DEFAULTS["gate_visa_resource"], kind="text"),
                        field("temperature_visa_resource", "MercuryiTC",
                              DEFAULTS["temperature_visa_resource"], kind="text"),
                        muted=True,
                    )
                    yield card(
                        "Source & gate limits",
                        field("source_delay_s", "6221 source delay (s)", DEFAULTS["source_delay_s"]),
                        field("gate_voltage_limit_V", "Gate voltage software limit (V)",
                              DEFAULTS["gate_voltage_limit_V"],
                              hint="Hard safety ceiling."),
                        field("gate_compliance_current_A", "Gate leakage compliance (A)",
                              DEFAULTS["gate_compliance_current_A"]),
                        muted=True,
                    )
                    yield card(
                        "Temperature sensors",
                        field("temperature_sensor_uids", "MercuryiTC sensor board UID(s)",
                              DEFAULTS["temperature_sensor_uids"], kind="text",
                              hint="1-2 UIDs, comma-separated."),
                        muted=True,
                    )

            with Vertical(id="sidebar"):
                yield Static("Description", classes="sidebar-title")
                yield Static(DC_IV_DESCRIPTION, classes="card-desc")
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start measurement  (F5)", id="start", variant="success")
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        # basicConfig (in dc_iv_curve) put a StreamHandler on the root logger;
        # writing to stdout while Textual owns the alt-screen would corrupt
        # the display, so drop it. RunScreen attaches its own RichLog-backed
        # handler for the duration of a measurement.
        logging.getLogger().handlers.clear()
        self._load_settings()
        self._set_gate_fields_enabled(self.query_one("#enable_gate", Switch).value)
        self._set_temperature_fields_enabled(self.query_one("#enable_temperature", Switch).value)
        self.refresh_summary()

    # ── Sample picker ────────────────────────────────────────────────────────

    def _refresh_sample_options(self, *, select_value: Optional[str] = None) -> None:
        select = self.query_one("#sample_select", Select)
        select.set_options(sample_options(_DATA_DIR))
        if select_value is not None:
            select.value = select_value

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "sample_select":
            return
        if event.value == NEW_SAMPLE_SENTINEL:
            self.push_screen(NewSampleScreen(_DATA_DIR), self._on_new_sample_created)
            return
        self.refresh_summary()

    def _on_new_sample_created(self, result: Optional[str]) -> None:
        self._refresh_sample_options(select_value=result if result else TEST_SAMPLE)
        self.refresh_summary()

    # ── Form state I/O ───────────────────────────────────────────────────────

    def _all_field_ids(self) -> list[str]:
        return list(NUMERIC_FIELDS) + TEXT_FIELDS + OPTIONAL_NUMERIC_FIELDS

    def collect_raw(self) -> dict:
        raw: dict = {fid: self.query_one(f"#{fid}", Input).value for fid in self._all_field_ids()}
        raw["auto_range"] = self.query_one("#auto_range", Switch).value
        raw["bidirectional_sweep"] = self.query_one("#bidirectional_sweep", Switch).value
        raw["enable_gate"] = self.query_one("#enable_gate", Switch).value
        raw["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
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
        if "auto_range" in saved:
            self.query_one("#auto_range", Switch).value = bool(saved["auto_range"])
        if "bidirectional_sweep" in saved:
            self.query_one("#bidirectional_sweep", Switch).value = bool(saved["bidirectional_sweep"])
        if "enable_gate" in saved:
            self.query_one("#enable_gate", Switch).value = bool(saved["enable_gate"])
        if "enable_temperature" in saved:
            self.query_one("#enable_temperature", Switch).value = bool(saved["enable_temperature"])
        saved_sample = saved.get("sample")
        if saved_sample and saved_sample in [v for _, v in sample_options(_DATA_DIR)]:
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
        state["auto_range"] = self.query_one("#auto_range", Switch).value
        state["bidirectional_sweep"] = self.query_one("#bidirectional_sweep", Switch).value
        state["enable_gate"] = self.query_one("#enable_gate", Switch).value
        state["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""

        state["gate_voltage_list"] = []
        state["gate_parse_error"] = None
        if state["enable_gate"]:
            try:
                state["gate_voltage_list"] = parse_value_list(state["gate_voltage_values"])
            except ValueError as exc:
                state["gate_parse_error"] = str(exc)

        return state, errors

    # ── Reactivity ───────────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        self.refresh_summary()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "enable_gate":
            self._set_gate_fields_enabled(event.value)
        elif event.switch.id == "enable_temperature":
            self._set_temperature_fields_enabled(event.value)
        self.refresh_summary()

    def _set_gate_fields_enabled(self, enabled: bool) -> None:
        for fid in GATE_FIELD_IDS:
            self.query_one(f"#{fid}", Input).disabled = not enabled

    def _set_temperature_fields_enabled(self, enabled: bool) -> None:
        for fid in TEMPERATURE_FIELD_IDS:
            self.query_one(f"#{fid}", Input).disabled = not enabled

    def refresh_summary(self) -> None:
        state, parse_errors = self.parse_state()
        if parse_errors:
            info, warnings, errors = [], [], parse_errors
            preview = None
        else:
            info, warnings, errors = build_summary(state)
            preview = compute_filename_preview(state)

        self.query_one("#filename_preview", Static).update(
            f"File:  [bold]{preview}[/bold]" if preview
            else "[dim]File:  (choose a sample and device to preview the filename)[/dim]"
        )

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

    # ── Start ────────────────────────────────────────────────────────────────

    def action_start(self) -> None:
        state, parse_errors = self.parse_state()
        if parse_errors:
            self.bell()
            return
        _, _, errors = build_summary(state)
        if errors:
            self.bell()
            return

        self._save_settings(self.collect_raw())
        plan = self._build_plan(state)
        self.push_screen(RunScreen(plan))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()

    def _build_plan(self, state: dict) -> MeasurementPlan:
        src_cfg = SourceConfig(
            visa_resource=state["source_visa_resource"],
            compliance_V=state["compliance_V"],
            source_delay_s=state["source_delay_s"],
            current_min_A=state["current_min_A"],
            current_max_A=state["current_max_A"],
        )
        volt_cfg = VoltmeterConfig(
            visa_resource=state["voltmeter_visa_resource"],
            nplc=state["nplc"],
            auto_range=state["auto_range"],
        )
        acq_cfg = AcquisitionConfig(
            settling_time_s=state["settling_time_s"],
            n_averages=state["n_averages"],
            output_file="",  # overwritten per series iteration in RunScreen
        )

        currents_A = linear_sweep(
            start=state["current_min_A"], stop=state["current_max_A"], step=state["step_A"],
            bidirectional=state["bidirectional_sweep"],
        )

        gate_cfg = None
        gate_voltages = None
        if state["enable_gate"]:
            gate_cfg = GateConfig(
                visa_resource=state["gate_visa_resource"],
                gate_voltage_limit_V=state["gate_voltage_limit_V"],
                compliance_current_A=state["gate_compliance_current_A"],
            )
            gate_voltages = state["gate_voltage_list"]

        temp_cfg = None
        if state["enable_temperature"]:
            uids = parse_sensor_uids(state["temperature_sensor_uids"])
            if uids:
                temp_cfg = TemperatureControllerConfig(
                    visa_resource=state["temperature_visa_resource"],
                    sensor_uids=uids,
                )

        header_extra = {
            "compliance_V": state["compliance_V"],
            "n_averages": state["n_averages"],
            "settling_time_s": state["settling_time_s"],
            "current_sweep_A": [state["current_min_A"], state["current_max_A"], state["step_A"]],
        }
        series = ""
        if len(gate_voltages or []) > 1:
            series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                      f"{datetime.now():%Y%m%dT%H%M%S}")

        return MeasurementPlan(
            src_cfg=src_cfg, volt_cfg=volt_cfg, acq_cfg=acq_cfg, currents_A=currents_A,
            sample=state["sample"], device=state["device"],
            temperature_setpoint_K=state["temperature_setpoint_K"],
            cooldown=state["cooldown"], header_extra=header_extra, series=series,
            gate_cfg=gate_cfg, gate_voltages=gate_voltages,
            temp_cfg=temp_cfg,
        )


def main() -> None:
    DCIVCurveApp().run()


if __name__ == "__main__":
    main()
