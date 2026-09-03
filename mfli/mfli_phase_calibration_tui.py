#!/usr/bin/env python3
"""
Textual TUI front-end for mfli_phase_calibration.py
=====================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-31

Lets you run the harmonic-Hall phase-calibration procedure — null the
leader's 1f Y quadrature against the sample's own resistive Hall response,
verify the null holds across a full field sweep, and empirically identify
which of X2f/Y2f carries the real signal — without touching the dataclasses
in the script itself. Mirrors the layout of mfli_dual_harmonic_tui.py.

Run with:
    python mfli_phase_calibration_tui.py

Requirements:
    pip install textual matplotlib  (in addition to mfli_phase_calibration.py's own deps)
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

from mfli.mfli_dual_harmonic import (
    DemodConfig,
    FilterConfig,
    GaussmeterConfig,
    MagnetConfig,
    MercuryITC,
    OutputConfig,
    TemperatureControllerConfig,
    configure_demodulator,
    configure_output,
    connect,
    connect_device,
    connect_gaussmeter,
    connect_magnet,
    connect_temperature_controller,
    setup_mds,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_output,
    shutdown_temperature_controller,
    sync_follower_oscillator,
)
from mfli.mfli_dual_harmonic_tui import (
    _LogRelay,
    _acquire_duration_s,
    field,
    format_duration,
    format_si,
    select_field,
    switch_field,
)
from mfli.mfli_phase_calibration import (
    AmplitudeCheckConfig,
    FrequencyCheckConfig,
    SweepConfig,
    format_report,
    run_phase_calibration,
)
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

log = logging.getLogger("mfli_phase_calibration_tui")

# Same convention as mfli_dual_harmonic_tui.py: data/settings live outside
# "bridge" so nothing generated at runtime ends up in the git-tracked source
# tree. _DEFAULT_DATA_DIR is the fallback data root; the real one is chosen
# per run in the identity bar's "Data root" field (mirrors the web app).
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "mfli_phase_calibration_tui_settings.json"

# Locked type code for this measurement (see instruments/data_naming.py) —
# never deviates.
MEASUREMENT_TYPE = "PHCAL"


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults
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
    "visa_resource": "GPIB0::6::INSTR",
    "current_limit_A": "35",
    "voltage_compliance_V": "15.0",
    "ramp_step_A": "0.1",
    "ramp_delay_s": "0.05",
    "gaussmeter_visa_resource": "GPIB0::12::INSTR",
    "gaussmeter_n_averages": "10",
    "gaussmeter_read_delay_s": "0.05",
    "calibration_current_A": "20",
    "i_min_A": "-20",
    "i_max_A": "20",
    "n_points": "11",
    "sweep_settling_time_s": "1.5",
    "field_settle_tolerance_mT": "0.02",
    "sweep_n_averages": "20",
    "hold_tol_ratio": "0.02",
    "null_n_averages": "20",
    "null_max_iterations": "5",
    "null_tol_deg": "0.02",
    "enable_amplitude_check": False,
    "amplitudes_V": "0.05, 0.1",
    "amp_n_averages": "20",
    "enable_frequency_check": False,
    "frequencies_Hz": "13.333, 17.777, 23.333",
    "freq_n_averages": "20",
    "freq_max_iterations": "5",
    "freq_tol_deg": "0.02",
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
}

# id -> caster, for every free-text numeric field (Select/Switch/list handled separately)
NUMERIC_FIELDS: dict = {
    "daq_port": int,
    "frequency_Hz": float,
    "amplitude_V": float,
    "series_R_ohm": float,
    "time_constant_s": float,
    "input_range_1f_V": float,
    "input_range_2f_V": float,
    "sample_rate_Hz": float,
    "current_limit_A": float,
    "voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
    "calibration_current_A": float,
    "i_min_A": float,
    "i_max_A": float,
    "n_points": int,
    "sweep_settling_time_s": float,
    "field_settle_tolerance_mT": float,
    "sweep_n_averages": int,
    "hold_tol_ratio": float,
    "null_n_averages": int,
    "null_max_iterations": int,
    "null_tol_deg": float,
    "amp_n_averages": int,
    "freq_n_averages": int,
    "freq_max_iterations": int,
    "freq_tol_deg": float,
}
TEXT_FIELDS = ["leader_device", "follower_device", "daq_host", "visa_resource",
               "gaussmeter_visa_resource", "device", "cooldown",
               "temperature_visa_resource", "temperature_sensor_uids", "data_dir"]
# Free-text, comma-separated floats — parsed to List[float] by hand in parse_state()
LIST_FIELDS = ["amplitudes_V", "frequencies_Hz"]
# Free-text, blank-allowed: parsed to Optional[float] by hand in parse_state()
# rather than going through NUMERIC_FIELDS' "blank is an error" casting.
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K"]


def parse_sensor_uids(raw: str) -> tuple:
    """Parse a comma-separated "MB1.T1, DB5.T1" field into a 1- or 2-tuple of UIDs."""
    uids = [u.strip() for u in raw.split(",") if u.strip()]
    return tuple(uids[:2])


# ─────────────────────────────────────────────────────────────────────────────
# Small widget-building helpers (keep compose() readable)  ── field()/
# switch_field()/select_field() themselves live in mfli_dual_harmonic_tui.py
# (imported above) and are already flat for the same grid-nesting reason
# card() documents below.
# ─────────────────────────────────────────────────────────────────────────────

def card(title: str, *groups, muted: bool = False) -> Vertical:
    """A bordered grid cell: a title plus its fields (each a flat list from
    field()/switch_field()/select_field(), or a single widget). `muted` =
    stable/rarely-changed configuration, styled to recede rather than
    compete for attention. Groups must stay flat (lists, not further
    Verticals) -- a grid cell containing a doubly-nested auto-height
    Vertical blows Textual 8.2.8's GridLayout.arrange() auto-row height up
    to ~100 rows instead of the handful the content needs; one level of
    Vertical (this card itself) is fine, a further nested one is not."""
    children: list = [Static(title, classes="card-title")]
    for group in groups:
        children.extend(group) if isinstance(group, list) else children.append(group)
    return Vertical(*children, classes="stable-card" if muted else "param-card")


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationPlan:
    daq_host: str
    daq_port: int
    leader: str
    follower: str
    out_cfg: OutputConfig
    demod1_cfg: DemodConfig
    demod2_cfg: DemodConfig
    magnet_cfg: MagnetConfig
    gauss_cfg: GaussmeterConfig
    sweep_cfg: SweepConfig
    calibration_current_A: float
    null_n_averages: int
    null_max_iterations: int
    null_tol_deg: float
    hold_tol_ratio: float
    amplitude_check_cfg: Optional[AmplitudeCheckConfig]
    frequency_check_cfg: Optional[FrequencyCheckConfig]
    output_csv: str
    run_ctx: RunContext
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    temp_cfg: Optional[TemperatureControllerConfig] = None
    series: str = ""
    data_root: Path = _DEFAULT_DATA_DIR

    @property
    def total_points(self) -> int:
        return max(0, 2 * self.sweep_cfg.n_points - 1)


def build_header_fields(plan: "CalibrationPlan", records: list[dict], *,
                         status: str, comment: str) -> dict:
    """
    Universal + measurement-specific header/index fields for one run. Called
    once with the complete sweep DataFrame's records (this suite writes in
    a single shot, not incrementally) and once more at end-of-run with the
    user's real good/open/short/noisy judgement -- see
    instruments/data_naming.py.

    T_setpoint_K is the nominal value used to build the filename's T###K
    token. T_K is the MEASURED mean (temperature_1_K) -- left blank (not
    backfilled with the setpoint) whenever the MercuryiTC is disconnected
    or hasn't produced a reading yet.
    """
    ctx = plan.run_ctx
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
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# Live validation / derived-value summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (info, warnings, errors) for a fully-parsed state dict."""
    info: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    # ── Sample / run identity ───────────────────────────────────────────────
    dir_warn, dir_err = validate_directory(state.get("data_dir", ""))
    if dir_err:
        errors.append(f"Data root: {dir_err}")
    elif dir_warn:
        warnings.append(f"Data root: {dir_warn}")
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL:
        errors.append("Choose a sample (or create a new one).")
    if not state.get("device"):
        errors.append("Device is required (e.g. HB3, SV2).")

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
        if state["sweep_settling_time_s"] < recommended_settle:
            warnings.append(
                f"Sweep settling time {state['sweep_settling_time_s']:g} s < 5×TC "
                f"({recommended_settle:g} s) — filter may not have settled."
            )
        else:
            info.append(f"Settling ≥ 5×TC ({recommended_settle:g} s) ✓")
    else:
        errors.append("Time constant must be > 0 s.")

    # ── Field sweep & calibration point ─────────────────────────────────────
    if state["n_points"] < 2:
        errors.append("Points per sweep direction must be ≥ 2.")
    max_abs_I = max(abs(state["i_min_A"]), abs(state["i_max_A"]))
    if max_abs_I > state["current_limit_A"]:
        errors.append(
            f"Sweep range (±{max_abs_I:g} A) exceeds the current limit "
            f"({state['current_limit_A']:g} A)."
        )
    if abs(state["calibration_current_A"]) > state["current_limit_A"]:
        errors.append(
            f"Calibration current ({state['calibration_current_A']:g} A) exceeds "
            f"the current limit ({state['current_limit_A']:g} A)."
        )
    elif abs(state["calibration_current_A"]) < max_abs_I:
        warnings.append(
            f"Calibration current ({state['calibration_current_A']:g} A) is smaller "
            f"than the sweep extremes (±{max_abs_I:g} A) — pick a point near "
            "saturation so the PHE/AHE 1f signal is large and well-behaved."
        )

    total_points = max(0, 2 * state["n_points"] - 1)
    per_point_s = state["sweep_settling_time_s"] + _acquire_duration_s(
        state["sweep_n_averages"], state["sample_rate_Hz"]
    )
    info.append(
        f"Sweep: {state['i_min_A']:g} A → {state['i_max_A']:g} A → "
        f"{state['i_min_A']:g} A, {total_points} points"
    )
    info.append("Field measured live at each point via Lake Shore 475 Gaussmeter "
                 f"({state['gaussmeter_visa_resource']})")
    info.append(f"Estimated sweep run time ≈ {format_duration(total_points * per_point_s)}")

    # ── Optional checks ─────────────────────────────────────────────────────
    if state["enable_amplitude_check"]:
        if len(state["amplitudes_V"]) < 2:
            errors.append("Amplitude check needs at least 2 amplitudes (comma-separated).")
        else:
            info.append(f"Amplitude check: {len(state['amplitudes_V'])} amplitude(s)")

    if state["enable_frequency_check"]:
        if len(state["frequencies_Hz"]) < 2:
            errors.append("Frequency check needs at least 2 frequencies (comma-separated).")
        else:
            info.append(f"Frequency check: {len(state['frequencies_Hz'])} frequencies")

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
    return f"{preview}_<timestamp>.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, same rationale as mfli_dual_harmonic_tui
# ─────────────────────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    try:
        fig.canvas.manager.set_window_title("MFLI phase calibration")
    except Exception:
        pass
    line_resid, = ax1.plot([], [], "o-", color="tab:red")
    line_x2f, = ax2.plot([], [], "o-", color="tab:blue", label="X2f")
    line_y2f, = ax2.plot([], [], "o-", color="tab:orange", label="Y2f")
    ax1.set_ylabel("1f  |Y| / R  (null residual)")
    ax1.set_yscale("log")
    ax2.set_ylabel("2f  (V)")
    ax2.set_xlabel("Magnetic field (mT)")
    ax2.legend(loc="best")
    ax1.set_title("Phase calibration — hold check & 2f channel ID")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    xs: list[float] = []
    resids: list[float] = []
    x2fs: list[float] = []
    y2fs: list[float] = []

    def _drain(_frame=None):
        updated = False
        while True:
            try:
                record = queue.get_nowait()
            except Exception:
                break
            x = record.get("magnet_field_mT")
            xs.append(x if x is not None else record["point_index"])
            resids.append(record["1f_residual_ratio"])
            x2fs.append(record["2f_X_V"])
            y2fs.append(record["2f_Y_V"])
            updated = True
        if updated:
            line_resid.set_data(xs, resids)
            line_x2f.set_data(xs, x2fs)
            line_y2f.set_data(xs, y2fs)
            for ax in (ax1, ax2):
                ax.relim()
                ax.autoscale_view()
        return line_resid, line_x2f, line_y2f

    # Keep a reference so it isn't garbage-collected mid-run.
    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_diagnostic_png(records: list[dict], png_path: Path) -> None:
    """Save a static hold-check / channel-ID PNG to proc/, from whatever
    sweep points were actually collected (including an aborted/partial run)."""
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")  # headless — must not touch the TUI's terminal
    import matplotlib.pyplot as plt

    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]
    resid = [r["1f_residual_ratio"] for r in records]
    x2f = [r["2f_X_V"] for r in records]
    y2f = [r["2f_Y_V"] for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    ax1.plot(xs, resid, "o-", color="tab:red")
    ax1.set_yscale("log")
    ax2.plot(xs, x2f, "o-", color="tab:blue", label="X2f")
    ax2.plot(xs, y2f, "o-", color="tab:orange", label="Y2f")
    ax1.set_ylabel("1f  |Y| / R  (null residual)")
    ax2.set_ylabel("2f  (V)")
    ax2.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax2.legend(loc="best")
    ax1.set_title("Phase calibration result")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot to '%s'", png_path)


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

    def __init__(self, plan: CalibrationPlan) -> None:
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
            "#", "I (A)", "B (mT)", "1f |Y|/R", "2f X (V)", "2f Y (V)", "T1 (K)", "T2 (K)"
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
                target=_live_plot_worker, args=(self._plot_queue,), daemon=True,
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
        B = record.get("magnet_field_mT")
        residual = record.get("1f_residual_ratio")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        table.add_row(
            str(record["point_index"] + 1),
            f"{record['magnet_current_A']:.4f}",
            f"{B:.2f}" if B is not None else "—",
            f"{residual:.2e}" if residual is not None else "—",
            f"{record['2f_X_V']:.4e}",
            f"{record['2f_Y_V']:.4e}",
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(1)
        self._set_status(f"Sweep point {record['point_index'] + 1} / {self.plan.total_points} complete.")

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False
        self.query_one("#abort_btn", Button).disabled = True

        # Finalize the header/index row UNCONDITIONALLY, right now — never
        # gated on the status/comment prompt below being answered, so a
        # closed session never leaves the record stuck at "in_progress".
        outcome_status = ("aborted" if self._stop_event.is_set()
                           else "error" if final_status.startswith("ERROR") else "completed")
        ctx = self.plan.run_ctx
        header_fields = build_header_fields(self.plan, self._records, status=outcome_status, comment="")
        try:
            write_record(ctx.raw_path, self._records, header_fields)
            finalize_index_row(self.plan.data_root, ctx.sample, ctx.run_number, header_fields)
        except Exception:
            log.exception("Could not finalize run record")

        try:
            png_path = proc_path(self.plan.data_root, ctx.sample, ctx.run_str, ctx.device,
                                  MEASUREMENT_TYPE, "plot")
            _save_diagnostic_png(self._records, png_path)
        except Exception:
            log.exception("Could not save diagnostic plot PNG")

        self.app.push_screen(StatusCommentScreen(), self._on_status_comment)

    def _on_status_comment(self, result: Optional[tuple[str, str]]) -> None:
        if result is None:
            return
        status, comment = result
        ctx = self.plan.run_ctx
        header_fields = build_header_fields(self.plan, self._records, status=status, comment=comment)
        try:
            # Never truncate an already-written raw file to an empty stub —
            # only a run that never wrote a point gets a header-only write.
            if self._records or not ctx.raw_path.exists():
                write_record(ctx.raw_path, self._records, header_fields)
            finalize_index_row(self.plan.data_root, ctx.sample, ctx.run_number, header_fields)
        except Exception:
            log.exception("Could not save final status/comment")

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
        gaussmeter = None
        temp_ctrl = None
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

            self._set_status_threadsafe("Connecting magnet power supply …")
            magnet = connect_magnet(plan.magnet_cfg)
            self._set_status_threadsafe("Connecting gaussmeter …")
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)

            if plan.temp_cfg is not None:
                self._set_status_threadsafe("Connecting to MercuryiTC (temperature) …")
                temp_ctrl = connect_temperature_controller(plan.temp_cfg)

            writer = make_incremental_writer(
                plan.run_ctx.raw_path,
                lambda records: build_header_fields(plan, records, status="in_progress", comment=""),
            )
            report = run_phase_calibration(
                daq, plan.leader, plan.follower, plan.out_cfg, plan.demod1_cfg, plan.demod2_cfg,
                plan.sweep_cfg, magnet, plan.magnet_cfg,
                calibration_current_A=plan.calibration_current_A,
                null_n_averages=plan.null_n_averages,
                null_max_iterations=plan.null_max_iterations,
                null_tol_deg=plan.null_tol_deg,
                output_csv=plan.output_csv,
                gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                hold_tol_ratio=plan.hold_tol_ratio,
                amplitude_check_cfg=plan.amplitude_check_cfg,
                frequency_check_cfg=plan.frequency_check_cfg,
                stop_event=self._stop_event,
                on_point=lambda record: self.app.call_from_thread(self._on_point, record),
                on_status=self._set_status_threadsafe,
                temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                write_csv=lambda df: writer(df.to_dict("records")),
            )
            for line in format_report(report).splitlines():
                log.info(line)
            final = "Calibration aborted." if self._stop_event.is_set() \
                else "Calibration complete — see log for the report."
        except Exception as exc:
            log.exception("Phase calibration failed")
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

class MFLIPhaseCalibrationApp(App):
    TITLE = "MFLI Phase Calibration"
    SUB_TITLE = "1f Y-null · hold check · 2f channel ID"

    # Session data root — fallback until _load_settings()/the identity bar's
    # "Data root" field replaces it. Read in compose(), so it must exist here.
    data_root: Path = _DEFAULT_DATA_DIR

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 48; border-left: solid $primary; padding: 1 2; overflow-y: auto; }
    .field-label { text-style: bold; }
    .hint { text-style: italic; color: $text-muted; }
    .switch-row { height: 3; }
    .switch-row Label { margin-left: 1; content-align: left middle; height: 3; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; }
    #actionbar { height: 3; align: center middle; }

    #identity_bar { border: round $accent; padding: 1 2; margin-bottom: 1; height: auto; }
    #filename_preview { margin-bottom: 1; }
    #data_dir_row { height: 3; margin-bottom: 1; }
    #data_dir_row Input { width: 1fr; }
    #data_dir_row Button { margin-left: 1; }
    #identity_fields { layout: grid; grid-size: 4; grid-gutter: 1 2; height: auto; }
    #identity_fields > Vertical { height: auto; }
    .section-title { text-style: bold underline; margin: 1 0; }
    .param-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }
    .param-card { border: round $primary; padding: 1 2; height: auto; }
    .stable-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }
    .stable-card { border: round $panel-darken-1; padding: 1 2; height: auto; }
    .stable-card .card-title { color: $text-muted; text-style: bold underline; }
    .stable-card .field-label { color: $text-muted; }
    .card-title { text-style: bold underline; margin-bottom: 1; }
    """

    BINDINGS = [
        Binding("f5", "start", "Start calibration", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                with Vertical(id="identity_bar"):
                    yield Static(id="filename_preview")
                    with Horizontal(id="data_dir_row"):
                        yield Input(value=str(_DEFAULT_DATA_DIR), id="data_dir",
                                    placeholder="Absolute path to the data root")
                        yield Button("Browse…", id="browse_data_dir")
                    with Vertical(id="identity_fields"):
                        yield Vertical(
                            Label("Sample", classes="field-label"),
                            Select(sample_options(self.data_root), id="sample_select",
                                   allow_blank=False, value=TEST_SAMPLE),
                        )
                        yield Vertical(*field("device", "Device (e.g. HB3, SV2)",
                                              DEFAULTS["device"], kind="text"))
                        yield Vertical(*field("cooldown", "Cooldown (optional)",
                                              DEFAULTS["cooldown"], kind="text"))
                        yield Vertical(*field(
                            "temperature_setpoint_K", "Temperature setpoint (K, optional)",
                            DEFAULTS["temperature_setpoint_K"], kind="number", valid_empty=True,
                            hint="Drives only the filename's T###K token — the header's "
                                 "T_K uses the measured temperature when available.",
                        ))

                with Vertical(classes="param-grid"):
                    yield card(
                        "Devices",
                        field("leader_device", "Leader MFLI (current source + 1f)",
                              DEFAULTS["leader_device"], kind="text"),
                        field("follower_device", "Follower MFLI (2f)",
                              DEFAULTS["follower_device"], kind="text"),
                    )
                    yield card(
                        "Excitation",
                        field("frequency_Hz", "Excitation frequency (Hz)",
                              DEFAULTS["frequency_Hz"],
                              hint="Avoid exact multiples of 50/60 Hz (mains pickup).",
                              validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                        field("amplitude_V", "Output amplitude (V, peak)",
                              DEFAULTS["amplitude_V"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        field("series_R_ohm", "Series resistor (Ω)",
                              DEFAULTS["series_R_ohm"],
                              hint="Sets excitation current: I ≈ V / R.",
                              validators=[Number(minimum=1.0, failure_description="must be > 0")]),
                    )
                    yield card(
                        "Field sweep & calibration point",
                        field("calibration_current_A", "Calibration magnet current (A)",
                              DEFAULTS["calibration_current_A"],
                              hint="Where the 1f Y-null is performed — pick a point near "
                                   "saturation (e.g. matching the sweep max) so the PHE/AHE "
                                   "1f signal is large and well-behaved."),
                        field("i_min_A", "Sweep current min (A)", DEFAULTS["i_min_A"]),
                        field("i_max_A", "Sweep current max (A)", DEFAULTS["i_max_A"]),
                        field("n_points", "Points per sweep direction",
                              DEFAULTS["n_points"], kind="integer",
                              validators=[Number(minimum=2, failure_description="must be ≥ 2")]),
                    )
                    yield card(
                        "Phase null",
                        field("null_n_averages", "Averages per phase read",
                              DEFAULTS["null_n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("null_max_iterations", "Max null iterations",
                              DEFAULTS["null_max_iterations"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("null_tol_deg", "Convergence tolerance (°)",
                              DEFAULTS["null_tol_deg"],
                              hint="Nulls the leader's 1f Y quadrature by adjusting its demod "
                                   "phaseshift node — the resistive PHE/AHE response at 1f must "
                                   "be exactly in phase with the drive current, so any measured "
                                   "Y there is pure instrumental delay."),
                    )
                    yield card(
                        "Amplitude check (optional)",
                        switch_field("enable_amplitude_check",
                                     "Run current-amplitude scaling check",
                                     DEFAULTS["enable_amplitude_check"]),
                        field("amplitudes_V", "Amplitudes to test (V, comma-separated)",
                              DEFAULTS["amplitudes_V"], kind="text",
                              hint="≥ 2 values. Checks whether the identified 2f signal "
                                   "channel scales linearly with drive current "
                                   "(resistive/SOT) rather than growing faster "
                                   "(Joule-heating/ANE)."),
                    )
                    yield card(
                        "Frequency check (optional)",
                        switch_field("enable_frequency_check",
                                     "Run frequency scaling check",
                                     DEFAULTS["enable_frequency_check"]),
                        field("frequencies_Hz", "Frequencies to test (Hz, comma-separated)",
                              DEFAULTS["frequencies_Hz"], kind="text",
                              hint="≥ 2 values. Checks whether the optimal 1f phase "
                                   "scales linearly with frequency, consistent with a "
                                   "fixed cable/electronics delay."),
                    )
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature",
                                     "Log temperature (Oxford Instruments MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                    )

                yield Static("Instrument configuration", classes="section-title")
                with Vertical(classes="stable-grid"):
                    yield card(
                        "Connection",
                        field("daq_host", "LabOne data server host",
                              DEFAULTS["daq_host"], kind="text"),
                        field("daq_port", "LabOne data server port",
                              DEFAULTS["daq_port"], kind="integer"),
                        muted=True,
                    )
                    yield card(
                        "Lock-in filter",
                        field("time_constant_s", "Filter time constant (s)",
                              DEFAULTS["time_constant_s"],
                              hint="Bigger = quieter but slower & longer settling.",
                              validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                        select_field("order", "Filter order", list(range(1, 9)),
                                     int(DEFAULTS["order"])),
                        switch_field("sinc_filter", "Sinc filter (extra harmonic rejection)",
                                     DEFAULTS["sinc_filter"]),
                        field("input_range_1f_V", "1f input range (V)",
                              DEFAULTS["input_range_1f_V"],
                              hint="Match expected 1f signal size — avoid clipping/poor resolution.",
                              validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                        field("input_range_2f_V", "2f input range (V)",
                              DEFAULTS["input_range_2f_V"],
                              hint="2f is usually much smaller than 1f — set separately.",
                              validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                        field("sample_rate_Hz", "Demodulator sample rate (Sa/s)",
                              DEFAULTS["sample_rate_Hz"],
                              validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                        muted=True,
                    )
                    yield card(
                        "Magnet & ramp safety",
                        field("visa_resource", "Magnet VISA resource",
                              DEFAULTS["visa_resource"], kind="text"),
                        field("current_limit_A", "Software current limit (A)",
                              DEFAULTS["current_limit_A"],
                              hint="Hard safety ceiling — independent of the supply's own range."),
                        field("voltage_compliance_V", "Voltage compliance (V)",
                              DEFAULTS["voltage_compliance_V"]),
                        field("ramp_step_A", "Ramp step (A)", DEFAULTS["ramp_step_A"]),
                        field("ramp_delay_s", "Ramp delay (s)", DEFAULTS["ramp_delay_s"]),
                        muted=True,
                    )
                    yield card(
                        "Gaussmeter",
                        field("gaussmeter_visa_resource", "Gaussmeter VISA resource",
                              DEFAULTS["gaussmeter_visa_resource"], kind="text",
                              hint="Lake Shore 475 — measures the actual field at each point."),
                        field("gaussmeter_n_averages", "Field readings averaged per point",
                              DEFAULTS["gaussmeter_n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("gaussmeter_read_delay_s", "Delay between readings (s)",
                              DEFAULTS["gaussmeter_read_delay_s"]),
                        muted=True,
                    )
                    yield card(
                        "Sweep timing & hold check",
                        field("sweep_settling_time_s", "Settling time per sweep point (s)",
                              DEFAULTS["sweep_settling_time_s"],
                              hint="Rule of thumb: ≥ 5 × time constant.",
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        field("field_settle_tolerance_mT", "Field-settle tolerance (mT)",
                              DEFAULTS["field_settle_tolerance_mT"],
                              hint="Advanced: after each magnet step, wait until a short window "
                                   "of gaussmeter readings spans less than this before the "
                                   "settling time above.",
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        field("sweep_n_averages", "Samples to average per sweep point",
                              DEFAULTS["sweep_n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("hold_tol_ratio",
                              "Max acceptable |Y|/R away from the calibration point",
                              DEFAULTS["hold_tol_ratio"],
                              hint="Flags drift if the null residual exceeds this "
                                   "anywhere in the sweep."),
                        muted=True,
                    )
                    yield card(
                        "Scaling-check advanced",
                        field("amp_n_averages", "Averages per amplitude point",
                              DEFAULTS["amp_n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("freq_n_averages", "Averages per phase read",
                              DEFAULTS["freq_n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("freq_max_iterations", "Max null iterations per frequency",
                              DEFAULTS["freq_max_iterations"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("freq_tol_deg", "Convergence tolerance per frequency (°)",
                              DEFAULTS["freq_tol_deg"]),
                        muted=True,
                    )
                    yield card(
                        "Temperature controller",
                        field("temperature_visa_resource", "MercuryiTC VISA resource",
                              DEFAULTS["temperature_visa_resource"], kind="text",
                              hint="e.g. TCPIP0::<ip>::7020::SOCKET (Ethernet) or an ASRL resource."),
                        field("temperature_sensor_uids", "Sensor board UID(s)",
                              DEFAULTS["temperature_sensor_uids"], kind="text",
                              hint="1 or 2 board UIDs, comma-separated, e.g. 'MB1.T1, DB5.T1'. "
                                   "Not connected, or only one probe wired up? Fine either way — "
                                   "missing readings just leave the column empty."),
                        muted=True,
                    )

            with Vertical(id="sidebar"):
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start calibration  (F5)", id="start", variant="success")
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        # basicConfig (in mfli_phase_calibration) put a StreamHandler on the
        # root logger; writing to stdout while Textual owns the alt-screen
        # would corrupt the display, so drop it. RunScreen attaches its own
        # RichLog-backed handler for the duration of a run.
        logging.getLogger().handlers.clear()
        self._load_settings()
        self.refresh_summary()

    # ── Sample picker ────────────────────────────────────────────────────────

    def _refresh_sample_options(self, *, select_value: Optional[str] = None) -> None:
        select = self.query_one("#sample_select", Select)
        options = sample_options(self.data_root)
        select.set_options(options)
        if select_value is not None:
            select.value = select_value

    def _sync_data_root(self) -> None:
        """Adopt the identity bar's "Data root" field when it names an
        existing directory, and re-list samples from there. Gated on
        is_dir() so a half-typed path doesn't scatter _test/ folders
        across the disk (sample_options() creates them)."""
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
        if event.select.id == "sample_select":
            if event.value == NEW_SAMPLE_SENTINEL:
                self.push_screen(NewSampleScreen(self.data_root), self._on_new_sample_created)
                return
        self.refresh_summary()

    def _on_new_sample_created(self, result: Optional[str]) -> None:
        self._refresh_sample_options(select_value=result if result else TEST_SAMPLE)
        self.refresh_summary()

    # ── Form state I/O ───────────────────────────────────────────────────────

    def _all_field_ids(self) -> list[str]:
        return list(NUMERIC_FIELDS) + TEXT_FIELDS + LIST_FIELDS + OPTIONAL_NUMERIC_FIELDS

    def collect_raw(self) -> dict:
        raw: dict = {fid: self.query_one(f"#{fid}", Input).value for fid in self._all_field_ids()}
        raw["sinc_filter"] = self.query_one("#sinc_filter", Switch).value
        raw["enable_amplitude_check"] = self.query_one("#enable_amplitude_check", Switch).value
        raw["enable_frequency_check"] = self.query_one("#enable_frequency_check", Switch).value
        raw["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
        raw["order"] = self.query_one("#order", Select).value
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
        if "sinc_filter" in saved:
            self.query_one("#sinc_filter", Switch).value = bool(saved["sinc_filter"])
        if "enable_amplitude_check" in saved:
            self.query_one("#enable_amplitude_check", Switch).value = bool(saved["enable_amplitude_check"])
        if "enable_frequency_check" in saved:
            self.query_one("#enable_frequency_check", Switch).value = bool(saved["enable_frequency_check"])
        if "enable_temperature" in saved:
            self.query_one("#enable_temperature", Switch).value = bool(saved["enable_temperature"])
        if "order" in saved:
            try:
                self.query_one("#order", Select).value = int(saved["order"])
            except Exception:
                pass
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
        for fid in LIST_FIELDS:
            raw = self.query_one(f"#{fid}", Input).value.strip()
            values: list[float] = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    values.append(float(part))
                except ValueError:
                    errors.append(f"'{fid}' contains a value that isn't a number: {part!r}")
            state[fid] = values
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
        state["sinc_filter"] = self.query_one("#sinc_filter", Switch).value
        state["enable_amplitude_check"] = self.query_one("#enable_amplitude_check", Switch).value
        state["enable_frequency_check"] = self.query_one("#enable_frequency_check", Switch).value
        state["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
        state["order"] = int(self.query_one("#order", Select).value)
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""
        return state, errors

    # ── Reactivity ───────────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "data_dir":
            self._sync_data_root()
        self.refresh_summary()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.refresh_summary()

    def on_select_changed(self, event: Select.Changed) -> None:
        self.refresh_summary()

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

        self.data_root = Path(state["data_dir"]).expanduser()
        # Typed a not-yet-existing root? bootstrap it now, so the run has
        # somewhere to write (allocate_run() itself stays strict).
        ensure_sample(self.data_root, state["sample"], create=True)
        self._save_settings(self.collect_raw())
        plan = self._build_plan(state)
        self.push_screen(RunScreen(plan))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()
        elif event.button.id == "browse_data_dir":
            self._browse_data_dir()

    def _build_plan(self, state: dict) -> CalibrationPlan:
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
        magnet_cfg = MagnetConfig(
            visa_resource=state["visa_resource"],
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
        sweep_cfg = SweepConfig(
            i_min_A=state["i_min_A"], i_max_A=state["i_max_A"], n_points=state["n_points"],
            settling_time_s=state["sweep_settling_time_s"], n_averages=state["sweep_n_averages"],
            field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
        )
        amplitude_check_cfg = AmplitudeCheckConfig(
            enabled=state["enable_amplitude_check"],
            amplitudes_V=state["amplitudes_V"] or [0.05, 0.1],
            n_averages=state["amp_n_averages"],
        )
        frequency_check_cfg = FrequencyCheckConfig(
            enabled=state["enable_frequency_check"],
            frequencies_Hz=state["frequencies_Hz"] or [13.333, 17.777, 23.333],
            n_averages=state["freq_n_averages"],
            max_iterations=state["freq_max_iterations"],
            tol_deg=state["freq_tol_deg"],
        )
        run_ctx = allocate_run(
            self.data_root, state["sample"], state["device"], MEASUREMENT_TYPE,
            temperature_setpoint_K=state["temperature_setpoint_K"],
        )
        output_csv = str(run_ctx.raw_path)

        temp_cfg = None
        if state["enable_temperature"]:
            uids = parse_sensor_uids(state["temperature_sensor_uids"])
            if uids:
                temp_cfg = TemperatureControllerConfig(
                    visa_resource=state["temperature_visa_resource"],
                    sensor_uids=uids,
                )

        header_extra = {
            "excitation_frequency_Hz": state["frequency_Hz"],
            "excitation_amplitude_V": state["amplitude_V"],
            "series_R_ohm": state["series_R_ohm"],
            "calibration_current_A": state["calibration_current_A"],
            "field_sweep_A": [state["i_min_A"], state["i_max_A"], state["n_points"]],
            "demod_time_constant_s": state["time_constant_s"],
            "demod_order": state["order"],
        }

        return CalibrationPlan(
            daq_host=state["daq_host"], daq_port=state["daq_port"],
            leader=state["leader_device"], follower=state["follower_device"],
            out_cfg=out_cfg, demod1_cfg=demod1_cfg, demod2_cfg=demod2_cfg,
            magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, sweep_cfg=sweep_cfg,
            calibration_current_A=state["calibration_current_A"],
            null_n_averages=state["null_n_averages"],
            null_max_iterations=state["null_max_iterations"],
            null_tol_deg=state["null_tol_deg"],
            hold_tol_ratio=state["hold_tol_ratio"],
            amplitude_check_cfg=amplitude_check_cfg,
            frequency_check_cfg=frequency_check_cfg,
            output_csv=output_csv,
            run_ctx=run_ctx, data_root=self.data_root, temperature_setpoint_K=state["temperature_setpoint_K"],
            cooldown=state["cooldown"], header_extra=header_extra,
            temp_cfg=temp_cfg,
        )


def main() -> None:
    MFLIPhaseCalibrationApp().run()


if __name__ == "__main__":
    main()
