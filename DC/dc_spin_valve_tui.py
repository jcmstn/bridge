#!/usr/bin/env python3
"""
Textual TUI front-end for dc_spin_valve.py
=============================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-31

Lets you edit the parameters that decide whether a spin-valve/field-sweep
measurement is good or bad — sense current, compliance, reversal
averaging, the magnet sweep, and the gate voltage — without touching the
dataclasses in the script itself.

The gate voltage (single value, or a comma-separated list) is held fixed
for each complete field sweep; a list runs one complete field sweep per
gate value, each saved to its own file and plotted together in the same
window with a different color.

Run with:
    python dc_spin_valve_tui.py

Requirements:
    pip install textual matplotlib  (in addition to dc_spin_valve.py's own deps)
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
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Static,
    Switch,
)

from dc.dc_spin_valve import (
    AcquisitionConfig,
    FieldPoint,
    GateConfig,
    GaussmeterConfig,
    MagnetConfig,
    SourceConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    connect_gate,
    connect_gaussmeter,
    connect_magnet,
    connect_source,
    connect_temperature_controller,
    connect_voltmeter,
    run_measurement,
    set_gate_voltage,
    set_magnet_current,
    shutdown_gate,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_source,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import build_output_path, linear_sweep, parse_value_list

log = logging.getLogger("dc_spin_valve_tui")

# Data/settings live outside "bridge" (a sibling of it).
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DATA_DIR / "dc_spin_valve_tui_settings.json"

DC_SPIN_VALVE_DESCRIPTION = (
    "Sources a fixed DC sense current with a Keithley 6221 and reads the "
    "longitudinal voltage with a Keithley 2182, reversing the current each "
    "rep to cancel thermal-EMF offsets by default — the same "
    "reversal-averaging technique as the Hall measurement, but for a "
    "longitudinal (spin-valve / magnetoresistance) read. Reversal can be "
    "switched off for bias-direction-dependent devices, where flipping the "
    "current destroys rather than cleans up the signal — the sense current "
    "is then just held fixed and plainly averaged instead. Sweeps a Kepco "
    "electromagnet's field (bidirectionally, for hysteresis) with the "
    "field measured live via a Lake Shore 475 Gaussmeter at every point. "
    "The gate voltage (Keithley 2400, optional — off by default needs no "
    "2400 connected) is held fixed for each field sweep — single value or "
    "a comma-separated list — one complete sweep per value, each saved to "
    "its own file and plotted together in different colors."
)


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors dc_spin_valve.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "source_visa_resource": "GPIB0::20::INSTR",
    "voltmeter_visa_resource": "GPIB0::7::INSTR",
    "gate_visa_resource": "GPIB0::25::INSTR",
    "sense_current_A": "0.001",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "nplc": "5",
    "auto_range": True,
    "settling_time_s": "1.0",
    "reversal_enabled": True,
    "n_averages": "5",
    "output_name": "dc_spin_valve",
    "output_subdir": "",
    "enable_gate": False,
    "gate_voltage_limit_V": "20.0",
    "gate_compliance_current_A": "0.000001",
    "gate_voltage_values": "0.0",
    "magnet_visa_resource": "GPIB0::6::INSTR",
    "current_limit_A": "35",
    "voltage_compliance_V": "15.0",
    "ramp_step_A": "0.1",
    "ramp_delay_s": "0.05",
    "i_min_A": "-20",
    "i_max_A": "20",
    "step_A": "2.0",
    "bidirectional_sweep": True,
    "gaussmeter_visa_resource": "GPIB0::12::INSTR",
    "gaussmeter_n_averages": "10",
    "gaussmeter_read_delay_s": "0.05",
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
}

NUMERIC_FIELDS: dict = {
    "sense_current_A": float,
    "compliance_V": float,
    "source_delay_s": float,
    "nplc": float,
    "settling_time_s": float,
    "n_averages": int,
    "gate_voltage_limit_V": float,
    "gate_compliance_current_A": float,
    "current_limit_A": float,
    "voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "i_min_A": float,
    "i_max_A": float,
    "step_A": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "gate_visa_resource",
               "output_name", "output_subdir", "magnet_visa_resource",
               "gaussmeter_visa_resource", "gate_voltage_values",
               "temperature_visa_resource", "temperature_sensor_uids"]
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
    return max(1e-3, nplc / 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    magnet_cfg: MagnetConfig
    gauss_cfg: GaussmeterConfig
    acq_cfg: AcquisitionConfig
    currents_A: np.ndarray
    output_subdir: str
    output_prefix: str
    gate_cfg: Optional[GateConfig] = None
    gate_voltages_V: Optional[List[float]] = None
    temp_cfg: Optional[TemperatureControllerConfig] = None

    @property
    def series_values(self) -> List[Optional[float]]:
        """[None] for a single gate-less run, else one entry per gate voltage."""
        return list(self.gate_voltages_V) if self.gate_voltages_V else [None]

    @property
    def total_points(self) -> int:
        return len(self.currents_A) * len(self.series_values)


# ─────────────────────────────────────────────────────────────────────────────
# Small widget-building helpers (keep compose() readable)
# ─────────────────────────────────────────────────────────────────────────────

def field(field_id: str, label_text: str, default: str, *, kind: str = "number",
          hint: str = "", validators=None) -> Vertical:
    children = [Label(label_text, classes="field-label"),
                Input(value=default, id=field_id, type=kind, validators=validators,
                      valid_empty=False)]
    if hint:
        children.append(Label(hint, classes="hint"))
    return Vertical(*children, classes="field")


def switch_field(field_id: str, label_text: str, default: bool) -> Vertical:
    return Vertical(
        Horizontal(Switch(value=default, id=field_id), Label(label_text, classes="switch-label"),
                   classes="switch-row"),
        classes="field",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Live validation / derived-value summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    info: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    resources = [state["source_visa_resource"], state["voltmeter_visa_resource"]]
    if state["enable_gate"]:
        resources.append(state["gate_visa_resource"])
    if len(set(resources)) < len(resources):
        errors.append(
            "Source (6221), voltmeter (2182)"
            + (", and gate (2400)" if state["enable_gate"] else "")
            + " VISA resources must all be different."
        )

    if state["sense_current_A"] == 0:
        errors.append("Sense current must be nonzero (resistance divides by it).")
    else:
        info.append(f"Sense current I = {format_si(state['sense_current_A'], 'A')}")

    if state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")

    if state["reversal_enabled"]:
        info.append("Sense current reversed +I/-I each rep to cancel thermal-EMF offsets.")
    else:
        info.append("Reversal off — sense current held fixed at +I "
                     "(use for bias-direction-dependent devices).")

    read_s = _reading_duration_s(state["nplc"])
    info.append(f"Estimated 2182 reading time ≈ {read_s * 1000:.0f} ms (NPLC={state['nplc']:g})")
    reps_per_point = 2 if state["reversal_enabled"] else 1
    per_point_s = state["settling_time_s"] + state["n_averages"] * reps_per_point * read_s

    # ── Gate (optional) ─────────────────────────────────────────────────────
    n_series = 1
    if state["enable_gate"]:
        if state["gate_voltage_limit_V"] <= 0:
            errors.append("Gate voltage limit must be > 0 V.")
        if state.get("gate_parse_error"):
            errors.append(f"Gate voltage list: {state['gate_parse_error']}")
            gate_list: list[float] = []
        else:
            gate_list = state.get("gate_voltage_list", [])
            over_limit = [v for v in gate_list if abs(v) > state["gate_voltage_limit_V"]]
            if over_limit:
                errors.append(
                    f"Gate voltage(s) {over_limit} exceed the configured limit "
                    f"±{state['gate_voltage_limit_V']:g} V."
                )
        n_series = len(gate_list)
        if n_series > 1:
            info.append(f"Gate: {n_series} values {gate_list} V — {n_series} complete field sweeps, "
                        f"one file each, plotted together")
        elif n_series == 1:
            info.append(f"Gate held fixed at {format_si(gate_list[0], 'V')}")
    else:
        info.append("Gate off — Keithley 2400 not used, single field sweep run.")

    # ── Field sweep ──────────────────────────────────────────────────────────
    max_abs_I = max(abs(state["i_min_A"]), abs(state["i_max_A"]))
    if max_abs_I > state["current_limit_A"]:
        errors.append(
            f"Sweep range (±{max_abs_I:g} A) exceeds the current limit "
            f"({state['current_limit_A']:g} A)."
        )
    if state["i_min_A"] == state["i_max_A"]:
        warnings.append("i_min equals i_max — sweep will repeat a single point.")

    n_one_way = 0
    if state["step_A"] <= 0:
        errors.append("Sweep step size must be > 0 A.")
    else:
        n_one_way = max(2, round(abs(state["i_max_A"] - state["i_min_A"]) / state["step_A"]) + 1)
    n_sweep_points = n_one_way if not state["bidirectional_sweep"] else max(0, 2 * n_one_way - 1)
    direction = (f"{state['i_min_A']:g} A → {state['i_max_A']:g} A → {state['i_min_A']:g} A"
                 if state["bidirectional_sweep"]
                 else f"{state['i_min_A']:g} A → {state['i_max_A']:g} A")
    info.append(f"Field sweep: {direction}, step={state['step_A']:g} A, {n_sweep_points} points")
    info.append("Field measured live at each point via Lake Shore 475 Gaussmeter "
                 f"({state['gaussmeter_visa_resource']})")

    total_points = n_sweep_points * max(1, n_series)
    info.append(f"Estimated total run time ≈ {format_duration(total_points * per_point_s)}")

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


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("DC Spin-Valve live measurement")
    except Exception:
        pass
    ax.set_xlabel("Magnetic field (mT)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Live measurement — field sweep")
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
            x = record.get("magnet_field_mT")
            xs.append(x if x is not None else record["point_index"])
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

    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path) -> None:
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5))

    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    series_ids = sorted({r.get("series_index", 0) for r in records})
    for idx in series_ids:
        rows = [r for r in records if r.get("series_index", 0) == idx]
        label = rows[0].get("series_label")
        xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in rows]
        ax.plot(xs, [r["voltage_V"] for r in rows], ".-", color=cmap(idx % 10), label=label)

    ax.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
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
# Logging -> RichLog relay
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
        self._measurement_running = True
        self._log_handler: Optional[_LogRelay] = None
        self._records: list[dict] = []
        self._plot_queue: Optional["mp.Queue"] = None
        self._plot_process: Optional[mp.Process] = None
        self._session_timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"

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
            "#", "Vg (V)", "I_magnet (A)", "B (mT)", "V (V)", "R (Ω)", "n_avg",
            "T1 (K)", "T2 (K)",
        )
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
            self._plot_process = ctx.Process(target=_live_plot_worker, args=(self._plot_queue,), daemon=True)
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
        I = record.get("magnet_current_A")
        B = record.get("magnet_field_mT")
        Vg = record.get("gate_voltage_V")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        table.add_row(
            str(record["point_index"] + 1),
            f"{Vg:.4g}" if Vg is not None else "—",
            f"{I:.4f}" if I is not None else "—",
            f"{B:.2f}" if B is not None else "—",
            f"{record['voltage_V']:.4e}",
            f"{record['resistance_ohm']:.5g}",
            str(record["n_averages"]),
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
            out_dir = _DATA_DIR / self.plan.output_subdir if self.plan.output_subdir else _DATA_DIR
            png_path = out_dir / f"{self.plan.output_prefix}_{self._session_timestamp}_combined.png"
            _save_measurement_png(self._records, png_path)
        except Exception:
            log.exception("Could not save measurement plot PNG")

    def action_abort(self) -> None:
        if self._measurement_running and not self._stop_event.is_set():
            self._stop_event.set()
            self._set_status("Abort requested — finishing current point, then ramping down safely …")

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
        magnet = None
        gaussmeter = None
        temp_ctrl = None
        try:
            self._set_status_threadsafe("Connecting to Keithley 6221 & 2182 …")
            source = connect_source(plan.src_cfg)
            voltmeter = connect_voltmeter(plan.volt_cfg)

            if plan.gate_cfg is not None:
                self._set_status_threadsafe("Connecting gate (Keithley 2400) …")
                gate = connect_gate(plan.gate_cfg)

            if plan.temp_cfg is not None:
                self._set_status_threadsafe("Connecting to MercuryiTC (temperature) …")
                temp_ctrl = connect_temperature_controller(plan.temp_cfg)

            self._set_status_threadsafe("Connecting magnet power supply …")
            magnet = connect_magnet(plan.magnet_cfg)
            self._set_status_threadsafe("Connecting gaussmeter …")
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)

            for series_idx, gate_V in enumerate(plan.series_values):
                if self._stop_event.is_set():
                    break

                label = None
                suffix = ""
                if gate_V is not None:
                    label = f"Vg={gate_V:g}V"
                    suffix = f"_Vg{gate_V:g}V"
                    self._set_status_threadsafe(f"Setting gate to {gate_V:g} V …")
                    set_gate_voltage(gate, plan.gate_cfg, gate_V)

                plan.acq_cfg.output_file = str(build_output_path(
                    _DATA_DIR, plan.output_subdir, plan.output_prefix,
                    self._session_timestamp, suffix,
                ))

                points = [
                    FieldPoint(
                        magnet_current_A=I,
                        set_action=lambda I=I: set_magnet_current(magnet, plan.magnet_cfg, I),
                    )
                    for I in plan.currents_A
                ]

                status = "Running field sweep …" if gate_V is None \
                    else f"Running field sweep (Vg={gate_V:g} V) …"
                self._set_status_threadsafe(status)
                run_measurement(
                    source, voltmeter, plan.src_cfg, plan.acq_cfg, points,
                    stop_event=self._stop_event,
                    on_point=self._make_on_point(series_idx, label),
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                    gate_voltage_V=gate_V,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                )

            final = "Measurement aborted." if self._stop_event.is_set() else "Measurement complete."
        except Exception as exc:
            log.exception("Measurement failed")
            final = f"ERROR: {exc}"
        finally:
            if magnet is not None:
                try:
                    shutdown_magnet(magnet, plan.magnet_cfg)
                except Exception:
                    log.exception("Error while shutting down magnet")
            if gaussmeter is not None:
                try:
                    shutdown_gaussmeter(gaussmeter)
                except Exception:
                    log.exception("Error while shutting down gaussmeter")
            if temp_ctrl is not None:
                try:
                    shutdown_temperature_controller(temp_ctrl)
                except Exception:
                    log.exception("Error while shutting down MercuryiTC")
            if gate is not None:
                try:
                    shutdown_gate(gate)
                except Exception:
                    log.exception("Error while shutting down gate")
            if source is not None:
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

class DCSpinValveApp(App):
    TITLE = "DC Spin-Valve / Field Sweep"
    SUB_TITLE = "Keithley 6221 + 2182 + 2400 · magnet field sweep"

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 48; border-left: solid $primary; padding: 1 2; overflow-y: auto; }
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
                with Collapsible(title="Instruments", collapsed=False):
                    yield field("source_visa_resource", "Keithley 6221 (current source)",
                                DEFAULTS["source_visa_resource"], kind="text")
                    yield field("voltmeter_visa_resource", "Keithley 2182 (voltage)",
                                DEFAULTS["voltmeter_visa_resource"], kind="text")
                    yield field("gate_visa_resource", "Keithley 2400 (gate)",
                                DEFAULTS["gate_visa_resource"], kind="text")

                with Collapsible(title="Sense current & compliance", collapsed=False):
                    yield field("sense_current_A", "Sense current (A)",
                                DEFAULTS["sense_current_A"],
                                hint="Reversed +I/-I each rep to cancel thermal-EMF offsets, "
                                     "unless reversal is switched off below.")
                    yield switch_field("reversal_enabled",
                                       "Reverse current each rep (+I/-I)",
                                       DEFAULTS["reversal_enabled"])
                    yield Label(
                        "Turn off for bias-direction-dependent devices (diodes, asymmetric "
                        "spin-orbit stacks, ...) where reversing the current destroys rather "
                        "than cleans up the signal — the sense current is then just held "
                        "fixed at +I and plainly averaged instead.",
                        classes="hint",
                    )
                    yield field("compliance_V", "Compliance voltage (V)",
                                DEFAULTS["compliance_V"],
                                validators=[Number(minimum=0.0, failure_description="must be ≥ 0")])
                    with Collapsible(title="Source timing (advanced)", collapsed=True):
                        yield field("source_delay_s", "6221 source delay (s)",
                                    DEFAULTS["source_delay_s"],
                                    hint="Also the settle time between a current reversal and "
                                         "reading the voltmeter, so the reversal has actually "
                                         "finished before it's read.")

                with Collapsible(title="Voltmeter (Keithley 2182)", collapsed=False):
                    yield field("nplc", "NPLC (integration time)", DEFAULTS["nplc"],
                                hint="Bigger = quieter but slower. 1 line cycle = 1/50 or 1/60 s.",
                                validators=[Number(minimum=0.01, failure_description="must be > 0")])
                    yield switch_field("auto_range", "Auto-range", DEFAULTS["auto_range"])

                with Collapsible(title="Acquisition timing", collapsed=False):
                    yield field("settling_time_s", "Settling time per point (s)",
                                DEFAULTS["settling_time_s"],
                                hint="Dead-time after a field change, before acquiring.",
                                validators=[Number(minimum=0.0, failure_description="must be ≥ 0")])
                    yield field("n_averages", "Voltage averages per point",
                                DEFAULTS["n_averages"], kind="integer",
                                hint="Reversal on: +I/-I reversal pairs — splits each point "
                                     "into (V(+I)-V(-I))/2 [reported R] and (V(+I)+V(-I))/2 "
                                     "[recorded, not discarded — real offsets land here, but "
                                     "so can genuine even-in-I physics]. Reversal off: plain "
                                     "voltage samples at the fixed sense current.",
                                validators=[Number(minimum=1, failure_description="must be ≥ 1")])
                    yield field("output_name", "Output file name (prefix)",
                                DEFAULTS["output_name"], kind="text")
                    yield field("output_subdir", "Data sub-directory (optional)",
                                DEFAULTS["output_subdir"], kind="text",
                                hint="Saved to data/<sub-directory>/<prefix>_<timestamp>.csv")

                with Collapsible(title="Gate voltage (Keithley 2400, optional)", collapsed=False):
                    yield switch_field("enable_gate", "Enable gate (Keithley 2400)",
                                       DEFAULTS["enable_gate"])
                    yield field("gate_voltage_limit_V", "Gate voltage software limit (V)",
                                DEFAULTS["gate_voltage_limit_V"],
                                hint="Hard safety ceiling — independent of the values below.")
                    yield field("gate_compliance_current_A", "Gate leakage compliance (A)",
                                DEFAULTS["gate_compliance_current_A"])
                    yield field("gate_voltage_values", "Gate voltage (V)",
                                DEFAULTS["gate_voltage_values"], kind="text",
                                hint="Single value, or comma-separated list — one complete "
                                     "field sweep runs per value, each saved to its own file "
                                     "and plotted together.")

                with Collapsible(title="Magnet & field sweep", collapsed=False):
                    yield field("magnet_visa_resource", "Magnet VISA resource",
                                DEFAULTS["magnet_visa_resource"], kind="text")
                    yield field("current_limit_A", "Software current limit (A)",
                                DEFAULTS["current_limit_A"],
                                hint="Hard safety ceiling — independent of the supply's own range.")
                    yield field("voltage_compliance_V", "Voltage compliance (V)",
                                DEFAULTS["voltage_compliance_V"])
                    with Collapsible(title="Ramp safety (advanced)", collapsed=True):
                        yield field("ramp_step_A", "Ramp step (A)", DEFAULTS["ramp_step_A"])
                        yield field("ramp_delay_s", "Ramp delay (s)", DEFAULTS["ramp_delay_s"])
                    yield field("i_min_A", "Sweep current min (A)", DEFAULTS["i_min_A"])
                    yield field("i_max_A", "Sweep current max (A)", DEFAULTS["i_max_A"])
                    yield field("step_A", "Sweep step size (A)", DEFAULTS["step_A"],
                                validators=[Number(minimum=1e-9, failure_description="must be > 0")])
                    yield switch_field("bidirectional_sweep",
                                       "Bidirectional sweep (min → max → min)",
                                       DEFAULTS["bidirectional_sweep"])
                    yield field("gaussmeter_visa_resource", "Gaussmeter VISA resource",
                                DEFAULTS["gaussmeter_visa_resource"], kind="text",
                                hint="Lake Shore 475 — measures the actual field at each point.")
                    with Collapsible(title="Gaussmeter averaging (advanced)", collapsed=True):
                        yield field("gaussmeter_n_averages", "Field readings averaged per point",
                                    DEFAULTS["gaussmeter_n_averages"], kind="integer",
                                    validators=[Number(minimum=1, failure_description="must be ≥ 1")])
                        yield field("gaussmeter_read_delay_s", "Delay between readings (s)",
                                    DEFAULTS["gaussmeter_read_delay_s"])

                with Collapsible(title="Temperature (MercuryiTC)", collapsed=False):
                    yield switch_field("enable_temperature",
                                        "Log temperature (Oxford Instruments MercuryiTC)",
                                        DEFAULTS["enable_temperature"])
                    yield field("temperature_visa_resource", "MercuryiTC VISA resource",
                                DEFAULTS["temperature_visa_resource"], kind="text",
                                hint="e.g. TCPIP0::<ip>::7020::SOCKET (Ethernet) or an ASRL resource.")
                    yield field("temperature_sensor_uids", "Sensor board UID(s)",
                                DEFAULTS["temperature_sensor_uids"], kind="text",
                                hint="1 or 2 board UIDs, comma-separated, e.g. 'MB1.T1, DB5.T1'. "
                                     "Not connected, or only one probe wired up? Fine either way — "
                                     "missing readings just leave the column empty.")

            with Vertical(id="sidebar"):
                yield Static("Description", classes="sidebar-title")
                yield Static(DC_SPIN_VALVE_DESCRIPTION, classes="card-desc")
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start measurement  (F5)", id="start", variant="success")
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        logging.getLogger().handlers.clear()
        self._load_settings()
        self._set_gate_fields_enabled(self.query_one("#enable_gate", Switch).value)
        self._set_temperature_fields_enabled(self.query_one("#enable_temperature", Switch).value)
        self.refresh_summary()

    # ── Form state I/O ───────────────────────────────────────────────────────

    def _all_field_ids(self) -> list[str]:
        return list(NUMERIC_FIELDS) + TEXT_FIELDS

    def collect_raw(self) -> dict:
        raw: dict = {fid: self.query_one(f"#{fid}", Input).value for fid in self._all_field_ids()}
        raw["auto_range"] = self.query_one("#auto_range", Switch).value
        raw["bidirectional_sweep"] = self.query_one("#bidirectional_sweep", Switch).value
        raw["reversal_enabled"] = self.query_one("#reversal_enabled", Switch).value
        raw["enable_gate"] = self.query_one("#enable_gate", Switch).value
        raw["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
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
        if "reversal_enabled" in saved:
            self.query_one("#reversal_enabled", Switch).value = bool(saved["reversal_enabled"])
        if "enable_gate" in saved:
            self.query_one("#enable_gate", Switch).value = bool(saved["enable_gate"])
        if "enable_temperature" in saved:
            self.query_one("#enable_temperature", Switch).value = bool(saved["enable_temperature"])

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
        state["auto_range"] = self.query_one("#auto_range", Switch).value
        state["bidirectional_sweep"] = self.query_one("#bidirectional_sweep", Switch).value
        state["reversal_enabled"] = self.query_one("#reversal_enabled", Switch).value
        state["enable_gate"] = self.query_one("#enable_gate", Switch).value
        state["enable_temperature"] = self.query_one("#enable_temperature", Switch).value

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
        else:
            info, warnings, errors = build_summary(state)

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
            sense_current_A=state["sense_current_A"],
            compliance_V=state["compliance_V"],
            source_delay_s=state["source_delay_s"],
        )
        volt_cfg = VoltmeterConfig(
            visa_resource=state["voltmeter_visa_resource"],
            nplc=state["nplc"],
            auto_range=state["auto_range"],
        )
        magnet_cfg = MagnetConfig(
            visa_resource=state["magnet_visa_resource"],
            current_limit_A=state["current_limit_A"],
            voltage_compliance_V=state["voltage_compliance_V"],
            ramp_step_A=state["ramp_step_A"],
            ramp_delay_s=state["ramp_delay_s"],
        )
        gauss_cfg = GaussmeterConfig(
            visa_resource=state["gaussmeter_visa_resource"],
            n_averages=state["gaussmeter_n_averages"],
            read_delay_s=state["gaussmeter_read_delay_s"],
        )
        acq_cfg = AcquisitionConfig(
            settling_time_s=state["settling_time_s"],
            reversal_enabled=state["reversal_enabled"],
            n_averages=state["n_averages"],
            output_file=str(_DATA_DIR / "dc_spin_valve.csv"),  # placeholder — overwritten per series
        )

        currents_A = linear_sweep(
            start=state["i_min_A"], stop=state["i_max_A"], step=state["step_A"],
            bidirectional=state["bidirectional_sweep"],
        )

        gate_cfg = None
        gate_voltages_V = None
        if state["enable_gate"]:
            gate_cfg = GateConfig(
                visa_resource=state["gate_visa_resource"],
                gate_voltage_limit_V=state["gate_voltage_limit_V"],
                compliance_current_A=state["gate_compliance_current_A"],
            )
            gate_voltages_V = state["gate_voltage_list"]

        temp_cfg = None
        if state["enable_temperature"]:
            uids = parse_sensor_uids(state["temperature_sensor_uids"])
            if uids:
                temp_cfg = TemperatureControllerConfig(
                    visa_resource=state["temperature_visa_resource"],
                    sensor_uids=uids,
                )

        return MeasurementPlan(
            src_cfg=src_cfg, volt_cfg=volt_cfg,
            magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, acq_cfg=acq_cfg,
            currents_A=currents_A,
            output_subdir=state["output_subdir"], output_prefix=state["output_name"],
            gate_cfg=gate_cfg, gate_voltages_V=gate_voltages_V,
            temp_cfg=temp_cfg,
        )


def main() -> None:
    DCSpinValveApp().run()


if __name__ == "__main__":
    main()
