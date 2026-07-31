#!/usr/bin/env python3
"""
Textual TUI front-end for mfli_dual_harmonic.py
================================================
Lets you edit the parameters that actually decide whether a dual-harmonic
lock-in measurement is good or bad — excitation, filters, timing, and the
magnet sweep — without having to touch the dataclasses in the script
itself. Parameters that rarely need changing (data-server host/port, ramp
step size) are tucked into collapsed "advanced" sub-sections rather than
hidden entirely.

The sidebar recomputes derived values (excitation current, filter
bandwidth, estimated sweep duration) and flags anything that risks a bad
measurement (mains-frequency pickup, under-settled filter, sweep exceeding
the magnet's software current limit) as you type.

Run with:
    python mfli_dual_harmonic_tui.py

Requirements:
    pip install textual  (in addition to mfli_dual_harmonic.py's own deps)
"""

from __future__ import annotations

import json
import logging
import math
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
    Button,
    Collapsible,
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

from mfli_dual_harmonic import (
    AcquisitionConfig,
    DemodConfig,
    FilterConfig,
    MagnetConfig,
    MeasurementPoint,
    OutputConfig,
    bidirectional_current_sweep,
    configure_demodulator,
    configure_output,
    connect,
    connect_device,
    connect_magnet,
    run_measurement,
    set_magnet_current,
    setup_mds,
    shutdown_magnet,
    shutdown_output,
    sync_follower_oscillator,
)

log = logging.getLogger("mfli_dual_harmonic_tui")

# Data/settings live outside "bridge" (a sibling of it), same convention as
# mfli_dual_harmonic.py, so nothing generated at runtime ends up in the
# git-tracked source tree.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DATA_DIR / "mfli_dual_harmonic_tui_settings.json"


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors mfli_dual_harmonic.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "leader_device": "dev7885",
    "follower_device": "dev7886",
    "daq_host": "localhost",
    "daq_port": "8004",
    "frequency_Hz": "17.777",
    "amplitude_V": "0.1",
    "series_R_ohm": "10000",
    "time_constant_s": "0.3",
    "order": "4",
    "sinc_filter": True,
    "input_range_1f_V": "1.0",
    "input_range_2f_V": "1.0",
    "sample_rate_Hz": "857.0",
    "settling_time_s": "15",
    "n_averages": "50",
    "output_name": "harmonic_hall",
    "enable_sweep": True,
    "visa_resource": "GPIB0::6::INSTR",
    "field_per_amp_mT": "10.0",
    "current_limit_A": "35",
    "voltage_compliance_V": "15.0",
    "ramp_step_A": "0.1",
    "ramp_delay_s": "0.05",
    "i_min_A": "-20",
    "i_max_A": "20",
    "n_points": "21",
}

# id -> caster, for every free-text numeric field (Select/Switch handled separately)
NUMERIC_FIELDS: dict = {
    "daq_port": int,
    "frequency_Hz": float,
    "amplitude_V": float,
    "series_R_ohm": float,
    "time_constant_s": float,
    "input_range_1f_V": float,
    "input_range_2f_V": float,
    "sample_rate_Hz": float,
    "settling_time_s": float,
    "n_averages": int,
    "field_per_amp_mT": float,
    "current_limit_A": float,
    "voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "i_min_A": float,
    "i_max_A": float,
    "n_points": int,
}
TEXT_FIELDS = ["leader_device", "follower_device", "daq_host", "output_name", "visa_resource"]
MAGNET_FIELD_IDS = [
    "visa_resource", "field_per_amp_mT", "current_limit_A", "voltage_compliance_V",
    "ramp_step_A", "ramp_delay_s", "i_min_A", "i_max_A", "n_points",
]


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


def _acquire_duration_s(n_averages: int, sample_rate_Hz: float) -> float:
    if sample_rate_Hz <= 0:
        return 0.0
    return max(0.1, (n_averages * 1.5) / sample_rate_Hz)


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    daq_host: str
    daq_port: int
    leader: str
    follower: str
    out_cfg: OutputConfig
    demod1_cfg: DemodConfig
    demod2_cfg: DemodConfig
    acq_cfg: AcquisitionConfig
    magnet_cfg: Optional[MagnetConfig]
    currents_A: Optional[np.ndarray]

    @property
    def total_points(self) -> int:
        return len(self.currents_A) if self.currents_A is not None else 1


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


def select_field(field_id: str, label_text: str, options: list[int], default: int) -> Vertical:
    return Vertical(
        Label(label_text, classes="field-label"),
        Select([(str(o), o) for o in options], id=field_id, value=default, allow_blank=False),
        classes="field",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Live validation / derived-value summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (info, warnings, errors) for a fully-parsed state dict."""
    info: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if state["leader_device"] == state["follower_device"]:
        errors.append("Leader and follower device IDs must be different.")

    # ── Excitation ──────────────────────────────────────────────────────────
    if state["series_R_ohm"] > 0:
        I = state["amplitude_V"] / state["series_R_ohm"]
        info.append(f"Excitation current I ≈ {format_si(I, 'A')}")
    else:
        errors.append("Series resistor must be > 0 Ω.")

    f = state["frequency_Hz"]
    for mains in (50, 60):
        nearest = round(f / mains) * mains
        if nearest > 0 and abs(f - nearest) < 0.5:
            warnings.append(
                f"{f:g} Hz is within 0.5 Hz of a {mains} Hz harmonic ({nearest} Hz) "
                "— mains pickup risk."
            )

    # ── Filter / timing ─────────────────────────────────────────────────────
    tc = state["time_constant_s"]
    if tc > 0:
        recommended_settle = 5 * tc
        if state["settling_time_s"] < recommended_settle:
            warnings.append(
                f"Settling time {state['settling_time_s']:g} s < 5×TC "
                f"({recommended_settle:g} s) — filter may not have settled."
            )
        else:
            info.append(f"Settling ≥ 5×TC ({recommended_settle:g} s) ✓")

        bw = 1.0 / (2 * math.pi * tc)
        min_rate = 4 * bw
        info.append(f"Filter noise bandwidth ≈ {bw:.3g} Hz")
        if state["sample_rate_Hz"] < min_rate:
            warnings.append(
                f"Sample rate {state['sample_rate_Hz']:g} Sa/s may be low for this TC "
                f"(want ≳ {min_rate:.1f} Sa/s)."
            )
    else:
        errors.append("Time constant must be > 0 s.")

    per_point_s = state["settling_time_s"] + _acquire_duration_s(
        state["n_averages"], state["sample_rate_Hz"]
    )

    # ── Sweep ────────────────────────────────────────────────────────────────
    if state["enable_sweep"]:
        if state["n_points"] < 2:
            errors.append("Points per sweep direction must be ≥ 2.")
        max_abs_I = max(abs(state["i_min_A"]), abs(state["i_max_A"]))
        if max_abs_I > state["current_limit_A"]:
            errors.append(
                f"Sweep range (±{max_abs_I:g} A) exceeds the current limit "
                f"({state['current_limit_A']:g} A)."
            )
        if state["i_min_A"] == state["i_max_A"]:
            warnings.append("i_min equals i_max — sweep will repeat a single point.")

        total_points = max(0, 2 * state["n_points"] - 1)
        info.append(
            f"Sweep: {state['i_min_A']:g} A → {state['i_max_A']:g} A → "
            f"{state['i_min_A']:g} A, {total_points} points"
        )
        info.append(
            f"Field range: {state['i_min_A'] * state['field_per_amp_mT']:.2f} → "
            f"{state['i_max_A'] * state['field_per_amp_mT']:.2f} mT"
        )
        info.append(f"Estimated total run time ≈ {format_duration(total_points * per_point_s)}")
    else:
        info.append("Single point — no field sweep, magnet untouched.")
        info.append(f"Estimated run time ≈ {format_duration(per_point_s)}")

    return info, warnings, errors


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
            "#", "I (A)", "B (mT)", "1f R (V)", "1f θ (°)", "2f R (V)", "2f θ (°)"
        )
        self._log_handler = _LogRelay(self)
        root = logging.getLogger()
        root.addHandler(self._log_handler)
        self.do_run()

    def on_unmount(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)

    def write_log(self, msg: str, style: str) -> None:
        self.query_one("#log", RichLog).write(Text(msg, style=style))

    def _set_status(self, text: str) -> None:
        self.query_one("#status_line", Static).update(text)

    def _on_point(self, record: dict) -> None:
        table = self.query_one("#results_table", DataTable)
        I = record.get("magnet_current_A")
        B = record.get("magnet_field_mT")
        table.add_row(
            str(record["point_index"] + 1),
            f"{I:.4f}" if I is not None else "—",
            f"{B:.2f}" if B is not None else "—",
            f"{record['1f_R_V']:.4e}",
            f"{record['1f_theta_deg']:.2f}",
            f"{record['2f_R_V']:.4e}",
            f"{record['2f_theta_deg']:.2f}",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(1)
        self._set_status(f"Point {record['point_index'] + 1} / {self.plan.total_points} complete.")

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False
        self.query_one("#abort_btn", Button).disabled = True

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

    @work(thread=True, exclusive=True)
    def do_run(self) -> None:
        plan = self.plan
        daq = None
        magnet = None
        try:
            self._set_status_threadsafe("Connecting to LabOne data server …")
            daq = connect(plan.daq_host, plan.daq_port)
            connect_device(daq, plan.leader, interface="1GbE")
            connect_device(daq, plan.follower, interface="1GbE")

            self._set_status_threadsafe("Synchronizing MDS …")
            setup_mds(daq, leader=plan.leader, follower=plan.follower)

            self._set_status_threadsafe("Configuring output & demodulators …")
            configure_output(daq, plan.out_cfg)
            sync_follower_oscillator(daq, plan.out_cfg, plan.follower)
            configure_demodulator(daq, plan.demod1_cfg)
            configure_demodulator(daq, plan.demod2_cfg)

            if plan.magnet_cfg is not None and plan.currents_A is not None:
                self._set_status_threadsafe("Connecting magnet power supply …")
                magnet = connect_magnet(plan.magnet_cfg)
                points = [
                    MeasurementPoint(
                        magnet_current_A=I,
                        magnet_field_mT=I * plan.magnet_cfg.field_per_amp_mT,
                        set_action=lambda daq, I=I: set_magnet_current(magnet, plan.magnet_cfg, I),
                    )
                    for I in plan.currents_A
                ]
            else:
                points = [MeasurementPoint()]

            self._set_status_threadsafe("Running measurement …")
            run_measurement(
                daq, plan.demod1_cfg, plan.demod2_cfg, plan.acq_cfg, points,
                stop_event=self._stop_event,
                on_point=lambda record: self.app.call_from_thread(self._on_point, record),
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
            if daq is not None:
                try:
                    shutdown_output(daq, plan.out_cfg)
                except Exception:
                    log.exception("Error while shutting down output")
            self.app.call_from_thread(self._on_finished, final)

    def _set_status_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_status, text)


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class MFLIDualHarmonicApp(App):
    TITLE = "MFLI Dual-Harmonic Measurement"
    SUB_TITLE = "1f / 2f lock-in · magnet field sweep"

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
                with Collapsible(title="Devices", collapsed=False):
                    yield field("leader_device", "Leader MFLI (current source + 1f)",
                                DEFAULTS["leader_device"], kind="text")
                    yield field("follower_device", "Follower MFLI (2f)",
                                DEFAULTS["follower_device"], kind="text")
                    with Collapsible(title="Connection (advanced)", collapsed=True):
                        yield field("daq_host", "LabOne data server host",
                                    DEFAULTS["daq_host"], kind="text")
                        yield field("daq_port", "LabOne data server port",
                                    DEFAULTS["daq_port"], kind="integer")

                with Collapsible(title="Excitation (current source)", collapsed=False):
                    yield field("frequency_Hz", "Excitation frequency (Hz)",
                                DEFAULTS["frequency_Hz"],
                                hint="Avoid exact multiples of 50/60 Hz (mains pickup).",
                                validators=[Number(minimum=1e-3, failure_description="must be > 0")])
                    yield field("amplitude_V", "Output amplitude (V, peak)",
                                DEFAULTS["amplitude_V"],
                                validators=[Number(minimum=0.0, failure_description="must be ≥ 0")])
                    yield field("series_R_ohm", "Series resistor (Ω)",
                                DEFAULTS["series_R_ohm"],
                                hint="Sets excitation current: I ≈ V / R.",
                                validators=[Number(minimum=1.0, failure_description="must be > 0")])

                with Collapsible(title="Lock-in filters & inputs", collapsed=False):
                    yield field("time_constant_s", "Filter time constant (s)",
                                DEFAULTS["time_constant_s"],
                                hint="Bigger = quieter but slower & longer settling.",
                                validators=[Number(minimum=1e-6, failure_description="must be > 0")])
                    yield select_field("order", "Filter order", list(range(1, 9)),
                                       int(DEFAULTS["order"]))
                    yield switch_field("sinc_filter", "Sinc filter (extra harmonic rejection)",
                                       DEFAULTS["sinc_filter"])
                    yield field("input_range_1f_V", "1f input range (V)",
                                DEFAULTS["input_range_1f_V"],
                                hint="Match expected 1f signal size — avoid clipping/poor resolution.",
                                validators=[Number(minimum=1e-6, failure_description="must be > 0")])
                    yield field("input_range_2f_V", "2f input range (V)",
                                DEFAULTS["input_range_2f_V"],
                                hint="2f is usually much smaller than 1f — set separately.",
                                validators=[Number(minimum=1e-6, failure_description="must be > 0")])
                    yield field("sample_rate_Hz", "Demodulator sample rate (Sa/s)",
                                DEFAULTS["sample_rate_Hz"],
                                validators=[Number(minimum=1e-3, failure_description="must be > 0")])

                with Collapsible(title="Acquisition timing", collapsed=False):
                    yield field("settling_time_s", "Settling time per point (s)",
                                DEFAULTS["settling_time_s"],
                                hint="Rule of thumb: ≥ 5 × time constant.",
                                validators=[Number(minimum=0.0, failure_description="must be ≥ 0")])
                    yield field("n_averages", "Samples to average per point",
                                DEFAULTS["n_averages"], kind="integer",
                                validators=[Number(minimum=1, failure_description="must be ≥ 1")])
                    yield field("output_name", "Output file name (prefix)",
                                DEFAULTS["output_name"], kind="text")

                with Collapsible(title="Magnet & field sweep", collapsed=False):
                    yield switch_field("enable_sweep", "Sweep magnetic field (Kepco magnet)",
                                       DEFAULTS["enable_sweep"])
                    yield field("visa_resource", "Magnet VISA resource",
                                DEFAULTS["visa_resource"], kind="text")
                    yield field("field_per_amp_mT", "Field calibration (mT / A)",
                                DEFAULTS["field_per_amp_mT"],
                                hint="Calibrate for your magnet/probe.")
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
                    yield field("n_points", "Points per sweep direction",
                                DEFAULTS["n_points"], kind="integer",
                                validators=[Number(minimum=2, failure_description="must be ≥ 2")])

            with Vertical(id="sidebar"):
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start measurement  (F5)", id="start", variant="success")
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        # basicConfig (in mfli_dual_harmonic) put a StreamHandler on the root
        # logger; writing to stdout while Textual owns the alt-screen would
        # corrupt the display, so drop it. RunScreen attaches its own
        # RichLog-backed handler for the duration of a measurement.
        logging.getLogger().handlers.clear()
        self._load_settings()
        self.refresh_summary()

    # ── Form state I/O ───────────────────────────────────────────────────────

    def _all_field_ids(self) -> list[str]:
        return list(NUMERIC_FIELDS) + TEXT_FIELDS

    def collect_raw(self) -> dict:
        raw: dict = {fid: self.query_one(f"#{fid}", Input).value for fid in self._all_field_ids()}
        raw["sinc_filter"] = self.query_one("#sinc_filter", Switch).value
        raw["enable_sweep"] = self.query_one("#enable_sweep", Switch).value
        raw["order"] = self.query_one("#order", Select).value
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
        if "sinc_filter" in saved:
            self.query_one("#sinc_filter", Switch).value = bool(saved["sinc_filter"])
        if "enable_sweep" in saved:
            self.query_one("#enable_sweep", Switch).value = bool(saved["enable_sweep"])
        if "order" in saved:
            try:
                self.query_one("#order", Select).value = int(saved["order"])
            except Exception:
                pass

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
        state["sinc_filter"] = self.query_one("#sinc_filter", Switch).value
        state["enable_sweep"] = self.query_one("#enable_sweep", Switch).value
        state["order"] = int(self.query_one("#order", Select).value)
        return state, errors

    # ── Reactivity ───────────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        self.refresh_summary()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "enable_sweep":
            self._set_magnet_fields_enabled(event.value)
        self.refresh_summary()

    def on_select_changed(self, event: Select.Changed) -> None:
        self.refresh_summary()

    def _set_magnet_fields_enabled(self, enabled: bool) -> None:
        for fid in MAGNET_FIELD_IDS:
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
        lines += [f"  [dim]•[/dim] {i}" for i in info]

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
        out_cfg = OutputConfig(
            device=state["leader_device"],
            frequency_Hz=state["frequency_Hz"],
            amplitude_V=state["amplitude_V"],
            series_R_ohm=state["series_R_ohm"],
        )
        filt = FilterConfig(
            time_constant_s=state["time_constant_s"],
            order=state["order"],
            sinc_filter=state["sinc_filter"],
        )
        demod1_cfg = DemodConfig(
            device=state["leader_device"], demod_index=0, harmonic=1,
            input_range_V=state["input_range_1f_V"],
            sample_rate_Hz=state["sample_rate_Hz"], filter=filt,
        )
        demod2_cfg = DemodConfig(
            device=state["follower_device"], demod_index=0, harmonic=2,
            input_range_V=state["input_range_2f_V"],
            sample_rate_Hz=state["sample_rate_Hz"], filter=filt,
        )
        acq_cfg = AcquisitionConfig(
            settling_time_s=state["settling_time_s"],
            n_averages=state["n_averages"],
            output_file=str(_DATA_DIR / f"{state['output_name']}_{datetime.now():%Y%m%d_%H%M%S}.csv"),
        )

        magnet_cfg = None
        currents_A = None
        if state["enable_sweep"]:
            magnet_cfg = MagnetConfig(
                visa_resource=state["visa_resource"],
                field_per_amp_mT=state["field_per_amp_mT"],
                current_limit_A=state["current_limit_A"],
                voltage_compliance_V=state["voltage_compliance_V"],
                ramp_step_A=state["ramp_step_A"],
                ramp_delay_s=state["ramp_delay_s"],
            )
            currents_A = bidirectional_current_sweep(
                i_min=state["i_min_A"], i_max=state["i_max_A"], n_points=state["n_points"],
            )

        return MeasurementPlan(
            daq_host=state["daq_host"], daq_port=state["daq_port"],
            leader=state["leader_device"], follower=state["follower_device"],
            out_cfg=out_cfg, demod1_cfg=demod1_cfg, demod2_cfg=demod2_cfg,
            acq_cfg=acq_cfg, magnet_cfg=magnet_cfg, currents_A=currents_A,
        )


def main() -> None:
    MFLIDualHarmonicApp().run()


if __name__ == "__main__":
    main()
