#!/usr/bin/env python3
"""
Textual TUI front-end for mfli_dual_harmonic.py
================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-31

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
    pip install textual matplotlib  (in addition to mfli_dual_harmonic.py's own deps)
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
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
    AcquisitionConfig,
    DemodConfig,
    FilterConfig,
    GaussmeterConfig,
    MagnetConfig,
    MeasurementPoint,
    OutputConfig,
    SampleGeometryConfig,
    TemperatureControllerConfig,
    acquire_averaged,
    auto_null_phase,
    bidirectional_current_sweep,
    configure_demodulator,
    configure_output,
    connect,
    connect_device,
    connect_gaussmeter,
    connect_magnet,
    connect_temperature_controller,
    run_measurement,
    set_magnet_current,
    setup_mds,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_output,
    shutdown_temperature_controller,
    sync_follower_oscillator,
)
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

log = logging.getLogger("mfli_dual_harmonic_tui")

# Data/settings live outside "bridge" (a sibling of it), same convention as
# mfli_dual_harmonic.py, so nothing generated at runtime ends up in the
# git-tracked source tree. _DATA_DIR doubles as the data-convention "data
# root" (parent of every {sample}/ folder).
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DATA_DIR / "mfli_dual_harmonic_tui_settings.json"

# Locked type code for this measurement (see instruments/data_naming.py) —
# never deviates.
MEASUREMENT_TYPE = "HARM"


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors mfli_dual_harmonic.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "leader_device": "dev7885",
    "follower_device": "dev7886",
    "daq_host": "localhost",
    "daq_port": "8004",
    "frequency_Hz": "317.3",
    "amplitude_V": "0.1",
    "series_R_ohm": "10000",
    "time_constant_s": "0.3",
    "order": "4",
    "sinc_filter": True,
    "differential": True,
    "ac_coupling": True,
    "input_range_1f_V": "1.0",
    "input_range_2f_V": "1.0",
    "sample_rate_Hz": "857.0",
    "settling_time_s": "15",
    "n_averages": "50",
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    "enable_sweep": True,
    "visa_resource": "GPIB0::6::INSTR",
    "current_limit_A": "35",
    "voltage_compliance_V": "15.0",
    "ramp_step_A": "0.1",
    "ramp_delay_s": "0.05",
    "i_min_A": "-20",
    "i_max_A": "20",
    "n_points": "21",
    "gaussmeter_visa_resource": "GPIB0::12::INSTR",
    "gaussmeter_n_averages": "10",
    "gaussmeter_read_delay_s": "0.05",
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
    "enable_phase_cal": False,
    "phase_cal_current_A": "",
    "phase_cal_n_averages": "20",
    "phase_cal_max_iterations": "5",
    "hall_bar_length_um": "",
    "hall_bar_width_um": "",
    "hall_bar_thickness_nm": "",
    "field_angle_from_oop_deg": "",
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
    "current_limit_A": float,
    "voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "i_min_A": float,
    "i_max_A": float,
    "n_points": int,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
    "phase_cal_n_averages": int,
    "phase_cal_max_iterations": int,
}
TEXT_FIELDS = ["leader_device", "follower_device", "daq_host", "device", "cooldown", "visa_resource",
               "gaussmeter_visa_resource", "temperature_visa_resource", "temperature_sensor_uids"]
# Free-text, blank-allowed: parsed to Optional[float] by hand in parse_state()
# rather than going through NUMERIC_FIELDS' "blank is an error" casting.
OPTIONAL_NUMERIC_FIELDS = [
    "temperature_setpoint_K",
    "phase_cal_current_A",
    "hall_bar_length_um", "hall_bar_width_um", "hall_bar_thickness_nm",
    "field_angle_from_oop_deg",
]
MAGNET_FIELD_IDS = [
    "visa_resource", "current_limit_A", "voltage_compliance_V",
    "ramp_step_A", "ramp_delay_s", "i_min_A", "i_max_A", "n_points",
    "gaussmeter_visa_resource", "gaussmeter_n_averages", "gaussmeter_read_delay_s",
]
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
    gauss_cfg: Optional[GaussmeterConfig]
    currents_A: Optional[np.ndarray]
    temp_cfg: Optional[TemperatureControllerConfig]
    phase_cal_enabled: bool
    phase_cal_current_A: Optional[float]
    phase_cal_n_averages: int
    phase_cal_max_iterations: int
    geometry_cfg: SampleGeometryConfig
    run_ctx: RunContext
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    series: str = ""

    @property
    def total_points(self) -> int:
        return len(self.currents_A) if self.currents_A is not None else 1


def build_header_fields(plan: "MeasurementPlan", records: list[dict], *,
                         status: str, comment: str) -> dict:
    """
    Universal + measurement-specific header/index fields for one run. Called
    on every incremental write (status='in_progress', comment='') and once
    more at end-of-run (outcome status, then again with the user's real
    good/open/short/noisy judgement) -- see instruments/data_naming.py.

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


def select_field(field_id: str, label_text: str, options: list[int], default: int) -> list:
    label = Label(label_text, classes="field-label")
    sel = Select([(str(o), o) for o in options], id=field_id, value=default, allow_blank=False)
    sel.styles.margin = (0, 0, 1, 0)
    return [label, sel]


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

    if state["leader_device"] == state["follower_device"]:
        errors.append("Leader and follower device IDs must be different.")

    # ── Excitation ──────────────────────────────────────────────────────────
    if state["series_R_ohm"] > 0:
        I = state["amplitude_V"] / state["series_R_ohm"]
        info.append(f"Excitation current I ≈ {format_si(I, 'A')}")
    else:
        errors.append("Series resistor must be > 0 Ω.")

    f = state["frequency_Hz"]
    for label, check_f in (("1f", f), ("2f", 2 * f)):
        for mains in (50, 60):
            nearest = round(check_f / mains) * mains
            if nearest > 0 and abs(check_f - nearest) < 0.5:
                warnings.append(
                    f"{label} ({check_f:g} Hz) is within 0.5 Hz of a {mains} Hz "
                    f"harmonic ({nearest} Hz) — mains pickup risk."
                )

    # ── Filter / timing ─────────────────────────────────────────────────────
    tc = state["time_constant_s"]
    if tc > 0:
        # Rule of thumb: ≥5×TC for a 1st-order filter, ≥10×TC for 3rd/4th
        # order (settles more slowly per time constant at higher order).
        settle_multiple = 10 if state["order"] >= 3 else 5
        recommended_settle = settle_multiple * tc
        if state["settling_time_s"] < recommended_settle:
            warnings.append(
                f"Settling time {state['settling_time_s']:g} s < {settle_multiple}×TC "
                f"({recommended_settle:g} s, order {state['order']}) — filter may not have settled."
            )
        else:
            info.append(f"Settling ≥ {settle_multiple}×TC ({recommended_settle:g} s) ✓")

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
        info.append("Field measured live at each point via Lake Shore 475 Gaussmeter "
                     f"({state['gaussmeter_visa_resource']})")
        info.append(f"Estimated total run time ≈ {format_duration(total_points * per_point_s)}")
    else:
        info.append("Single point — no field sweep, magnet untouched.")
        info.append(f"Estimated run time ≈ {format_duration(per_point_s)}")

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

    # ── Phase calibration ───────────────────────────────────────────────────
    if state["enable_phase_cal"]:
        if state["phase_cal_current_A"] is not None:
            if not state["enable_sweep"]:
                warnings.append(
                    "Phase-cal current is set but field sweep is disabled — "
                    "it will be ignored; calibration runs at the present field."
                )
            else:
                max_abs_I = max(abs(state["i_min_A"]), abs(state["i_max_A"]))
                if abs(state["phase_cal_current_A"]) > state["current_limit_A"]:
                    errors.append(
                        f"Phase-cal current ({state['phase_cal_current_A']:g} A) exceeds "
                        f"the current limit ({state['current_limit_A']:g} A)."
                    )
                elif abs(state["phase_cal_current_A"]) < max_abs_I:
                    warnings.append(
                        f"Phase-cal current ({state['phase_cal_current_A']:g} A) is smaller "
                        f"than the sweep extremes (±{max_abs_I:g} A) — pick a point near "
                        "saturation for a clean, well-behaved PHE/AHE null."
                    )
                info.append(
                    f"Phase cal: ramp to {state['phase_cal_current_A']:g} A, null 1f Y "
                    "(leader demod phaseshift), then run the sweep."
                )
        else:
            info.append("Phase cal: null 1f Y at the present field (no magnet ramp).")

    # ── Sample geometry (optional — needed for quantitative analysis) ──────────
    geom_fields = {
        "Hall bar length": state["hall_bar_length_um"],
        "Hall bar width": state["hall_bar_width_um"],
        "Hall bar thickness": state["hall_bar_thickness_nm"],
        "Field angle from out-of-plane": state["field_angle_from_oop_deg"],
    }
    set_geom = {k: v for k, v in geom_fields.items() if v is not None}
    if not set_geom:
        warnings.append(
            "Sample geometry and field angle are unset — the run will still "
            "record raw 1f/2f voltages, but converting them to a resistivity "
            "or an absolute spin-Hall/damping-like field needs these (optional "
            "fields below)."
        )
    elif len(set_geom) < len(geom_fields):
        missing = ", ".join(k for k in geom_fields if geom_fields[k] is None)
        warnings.append(f"Sample geometry partially set — still missing: {missing}.")
        info.append("Sample geometry: " + ", ".join(f"{k}={v:g}" for k, v in set_geom.items()))
    else:
        info.append("Sample geometry: " + ", ".join(f"{k}={v:g}" for k, v in set_geom.items()))

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
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────
# A GUI matplotlib backend and Textual's terminal control both want the main
# thread (this matters especially on macOS, where Cocoa-backed GUI toolkits
# refuse to run off-main-thread). Rather than fight that, the live preview
# gets its own process with its own main thread; new points are streamed to
# it over a multiprocessing.Queue. The final PNG is saved independently by
# the TUI process itself (see _save_measurement_png), so it doesn't depend
# on this window still being open when the run finishes.

def _live_plot_worker(queue: "mp.Queue", has_field_sweep: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    try:
        fig.canvas.manager.set_window_title("MFLI live measurement")
    except Exception:
        pass
    line1, = ax1.plot([], [], "o-", color="tab:blue")
    line2, = ax2.plot([], [], "o-", color="tab:orange")
    ax1.set_ylabel("1f  R (V)")
    ax2.set_ylabel("2f  R (V)")
    ax2.set_xlabel("Magnetic field (mT)" if has_field_sweep else "Point #")
    ax1.set_title("Live measurement")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    xs: list[float] = []
    r1s: list[float] = []
    r2s: list[float] = []

    def _drain(_frame=None):
        updated = False
        while True:
            try:
                record = queue.get_nowait()
            except Exception:
                break
            x = record.get("magnet_field_mT") if has_field_sweep else None
            xs.append(x if x is not None else record["point_index"])
            r1s.append(record["1f_R_V"])
            r2s.append(record["2f_R_V"])
            updated = True
        if updated:
            line1.set_data(xs, r1s)
            line2.set_data(xs, r2s)
            for ax, ys in ((ax1, r1s), (ax2, r2s)):
                ax.relim()
                ax.autoscale_view()
        return line1, line2

    # Keep a reference so it isn't garbage-collected mid-run.
    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path) -> None:
    """Save a static 1f/2f R-vs-field PNG to proc/, from whatever points
    were actually collected (including an aborted/partial run)."""
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")  # headless — must not touch the TUI's terminal
    import matplotlib.pyplot as plt

    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]
    r1 = [r["1f_R_V"] for r in records]
    r2 = [r["2f_R_V"] for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    ax1.plot(xs, r1, "o-", color="tab:blue")
    ax2.plot(xs, r2, "o-", color="tab:orange")
    ax1.set_ylabel("1f  R (V)")
    ax2.set_ylabel("2f  R (V)")
    ax2.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax1.set_title("Measurement result")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
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
            "#", "I (A)", "B (mT)", "1f R (V)", "1f θ (°)", "2f R (V)", "2f θ (°)",
            "T1 (K)", "T2 (K)",
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
                args=(self._plot_queue, self.plan.magnet_cfg is not None),
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
        I = record.get("magnet_current_A")
        B = record.get("magnet_field_mT")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        table.add_row(
            str(record["point_index"] + 1),
            f"{I:.4f}" if I is not None else "—",
            f"{B:.2f}" if B is not None else "—",
            f"{record['1f_R_V']:.4e}",
            f"{record['1f_theta_deg']:.2f}",
            f"{record['2f_R_V']:.4e}",
            f"{record['2f_theta_deg']:.2f}",
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(1)
        self._set_status(f"Point {record['point_index'] + 1} / {self.plan.total_points} complete.")

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
            finalize_index_row(_DATA_DIR, ctx.sample, ctx.run_number, header_fields)
        except Exception:
            log.exception("Could not finalize run record")

        try:
            png_path = proc_path(_DATA_DIR, ctx.sample, ctx.run_str, ctx.device,
                                  MEASUREMENT_TYPE, "plot")
            _save_measurement_png(self._records, png_path)
        except Exception:
            log.exception("Could not save measurement plot PNG")

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
            finalize_index_row(_DATA_DIR, ctx.sample, ctx.run_number, header_fields)
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
            mds = setup_mds(daq, leader=plan.leader, follower=plan.follower)

            self._set_status_threadsafe("Configuring output & demodulators …")
            configure_output(daq, plan.out_cfg)
            sync_follower_oscillator(daq, plan.out_cfg, plan.follower)
            configure_demodulator(daq, plan.demod1_cfg)
            configure_demodulator(daq, plan.demod2_cfg)

            if plan.temp_cfg is not None:
                self._set_status_threadsafe("Connecting to MercuryiTC (temperature) …")
                temp_ctrl = connect_temperature_controller(plan.temp_cfg)

            if plan.magnet_cfg is not None and plan.currents_A is not None:
                self._set_status_threadsafe("Connecting magnet power supply …")
                magnet = connect_magnet(plan.magnet_cfg)
                self._set_status_threadsafe("Connecting gaussmeter …")
                gaussmeter = connect_gaussmeter(plan.gauss_cfg)
                points = [
                    MeasurementPoint(
                        magnet_current_A=I,
                        set_action=lambda daq, I=I: set_magnet_current(magnet, plan.magnet_cfg, I),
                    )
                    for I in plan.currents_A
                ]
            else:
                points = [MeasurementPoint()]

            if plan.phase_cal_enabled:
                self._set_status_threadsafe(
                    "Phase calibration: nulling 1f Y (leader demod phaseshift) …"
                )
                if magnet is not None and plan.phase_cal_current_A is not None:
                    log.info("Phase calibration: ramping magnet to %.4f A ...",
                             plan.phase_cal_current_A)
                    set_magnet_current(magnet, plan.magnet_cfg, plan.phase_cal_current_A)
                    time.sleep(plan.acq_cfg.settling_time_s)
                result = auto_null_phase(
                    daq, plan.demod1_cfg,
                    n_averages=plan.phase_cal_n_averages,
                    max_iterations=plan.phase_cal_max_iterations,
                )
                if not result.converged:
                    log.warning(
                        "Phase null did not fully converge after %d iteration(s) "
                        "(|Y|/R=%.2e) — check cabling/contacts before trusting the 2f data.",
                        result.iterations, result.residual_ratio,
                    )
                # 2f is measured on a different physical device (the follower) with its
                # own delay chain, so nulling the leader's 1f phase says nothing about
                # which 2f channel is physically correct — that must be verified
                # empirically. Both X2f/Y2f are already saved per point in the CSV;
                # this snapshot just gives an immediate look at the calibration point.
                d2 = acquire_averaged(daq, plan.demod2_cfg, plan.phase_cal_n_averages)
                log.info(
                    "2f snapshot at calibration point: X=%.4e V  Y=%.4e V  R=%.4e V — "
                    "don't assume this matches 1f's X/Y convention (V_2w ~ cos, not sin); "
                    "check which channel carries the structured field dependence in the "
                    "recorded sweep before trusting either one.",
                    d2["x_mean"], d2["y_mean"], d2["r_mean"],
                )

            self._set_status_threadsafe("Running measurement …")
            write_csv = make_incremental_writer(
                plan.run_ctx.raw_path,
                lambda records: build_header_fields(plan, records, status="in_progress", comment=""),
            )
            run_measurement(
                daq, plan.out_cfg, plan.demod1_cfg, plan.demod2_cfg, plan.acq_cfg, points,
                stop_event=self._stop_event,
                on_point=lambda record: self.app.call_from_thread(self._on_point, record),
                gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                geometry_cfg=plan.geometry_cfg, mds=mds,
                write_csv=write_csv,
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
    .field-label { text-style: bold; }
    .hint { text-style: italic; color: $text-muted; }
    .switch-row { height: 3; }
    .switch-row Label { margin-left: 1; content-align: left middle; height: 3; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; }
    #actionbar { height: 3; align: center middle; }

    #identity_bar { border: round $accent; padding: 1 2; height: auto; margin-bottom: 1; }
    #filename_preview { margin-bottom: 1; }
    #identity_fields { layout: grid; grid-size: 4; grid-gutter: 1 2; height: auto; }
    #identity_fields > Vertical { height: auto; }
    .field { margin-bottom: 1; }
    .section-title { text-style: bold underline; margin: 1 0; }
    .param-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }
    .param-card { border: round $primary; padding: 1 2; height: auto; }
    .stable-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }
    .stable-card { border: round $panel-darken-1; padding: 1 2; height: auto; }
    .stable-card .card-title { color: $text-muted; }
    .stable-card .field-label { color: $text-muted; }
    .card-title { text-style: bold underline; margin-bottom: 1; }
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
                    yield Static(id="filename_preview")
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
                                              valid_empty=True,
                                              hint="Filename's T###K token only."),
                                       classes="field")

                with Vertical(classes="param-grid"):
                    yield card(
                        "Devices",
                        field("leader_device", "Leader MFLI (current source + 1f)",
                              DEFAULTS["leader_device"], kind="text"),
                        field("follower_device", "Follower MFLI (2f)",
                              DEFAULTS["follower_device"], kind="text"),
                    )
                    yield card(
                        "Excitation (current source)",
                        field("frequency_Hz", "Excitation frequency (Hz)",
                              DEFAULTS["frequency_Hz"],
                              hint="Recommended ~300-1000 Hz — avoid exact multiples of 50/60 Hz "
                                   "(mains pickup).",
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
                        "Lock-in filter",
                        field("time_constant_s", "Filter time constant (s)",
                              DEFAULTS["time_constant_s"],
                              hint="Bigger = quieter but slower & longer settling.",
                              validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                        select_field("order", "Filter order", list(range(1, 9)),
                                     int(DEFAULTS["order"])),
                        switch_field("sinc_filter", "Sinc filter (extra harmonic rejection)",
                                     DEFAULTS["sinc_filter"]),
                    )
                    yield card(
                        "Input channels",
                        switch_field("differential", "Differential input (IN+/IN-)",
                                     DEFAULTS["differential"]),
                        switch_field("ac_coupling", "AC-couple the input",
                                     DEFAULTS["ac_coupling"]),
                        field("input_range_1f_V", "1f input range (V)",
                              DEFAULTS["input_range_1f_V"],
                              hint="Match expected 1f signal size.",
                              validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                        field("input_range_2f_V", "2f input range (V)",
                              DEFAULTS["input_range_2f_V"],
                              hint="2f is usually much smaller than 1f.",
                              validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                        field("sample_rate_Hz", "Demodulator sample rate (Sa/s)",
                              DEFAULTS["sample_rate_Hz"],
                              validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                    )
                    yield card(
                        "Acquisition timing",
                        field("settling_time_s", "Settling time per point (s)",
                              DEFAULTS["settling_time_s"],
                              hint="Rule of thumb: ≥ 5×TC (order 1), ≥ 10×TC (order 3-4, default).",
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        field("n_averages", "Samples to average per point",
                              DEFAULTS["n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                    )
                    yield card(
                        "Magnet & field sweep",
                        switch_field("enable_sweep", "Sweep magnetic field (Kepco magnet)",
                                     DEFAULTS["enable_sweep"]),
                        field("i_min_A", "Sweep current min (A)", DEFAULTS["i_min_A"]),
                        field("i_max_A", "Sweep current max (A)", DEFAULTS["i_max_A"]),
                        field("n_points", "Points per sweep direction",
                              DEFAULTS["n_points"], kind="integer",
                              validators=[Number(minimum=2, failure_description="must be ≥ 2")]),
                    )
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature",
                                     "Log temperature (Oxford Instruments MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                    )
                    yield card(
                        "Phase calibration",
                        switch_field(
                            "enable_phase_cal",
                            "Auto-null 1f phase before run (leader demod phaseshift)",
                            DEFAULTS["enable_phase_cal"],
                        ),
                        field(
                            "phase_cal_current_A", "Calibration magnet current (A)",
                            DEFAULTS["phase_cal_current_A"], kind="text", valid_empty=True,
                            hint="Blank = null at the present field. Otherwise pick a point near "
                                 "saturation (e.g. matching i_max). Only used if the field sweep "
                                 "above is enabled.",
                        ),
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
                        "Magnet & gaussmeter addresses",
                        field("visa_resource", "Magnet VISA resource",
                              DEFAULTS["visa_resource"], kind="text"),
                        field("current_limit_A", "Software current limit (A)",
                              DEFAULTS["current_limit_A"],
                              hint="Hard safety ceiling — independent of the supply's own range."),
                        field("voltage_compliance_V", "Voltage compliance (V)",
                              DEFAULTS["voltage_compliance_V"]),
                        field("ramp_step_A", "Ramp step (A)", DEFAULTS["ramp_step_A"]),
                        field("ramp_delay_s", "Ramp delay (s)", DEFAULTS["ramp_delay_s"]),
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
                        "Temperature controller",
                        field("temperature_visa_resource", "MercuryiTC VISA resource",
                              DEFAULTS["temperature_visa_resource"], kind="text",
                              hint="e.g. TCPIP0::<ip>::7020::SOCKET (Ethernet) or an ASRL resource."),
                        field("temperature_sensor_uids", "Sensor board UID(s)",
                              DEFAULTS["temperature_sensor_uids"], kind="text",
                              hint="1 or 2 board UIDs, comma-separated, e.g. 'MB1.T1, DB5.T1'. "
                                   "Missing readings just leave the column empty."),
                        muted=True,
                    )
                    yield card(
                        "Phase-cal advanced",
                        field("phase_cal_n_averages", "Averages per phase read",
                              DEFAULTS["phase_cal_n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("phase_cal_max_iterations", "Max null iterations",
                              DEFAULTS["phase_cal_max_iterations"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        Static(
                            "Nulls the leader's 1f Y quadrature by adjusting its demod "
                            "phaseshift node — the resistive PHE/AHE response at 1f must be "
                            "exactly in phase with the drive current, so any measured Y there is "
                            "pure instrumental delay. X and Y at 2f are both already recorded per "
                            "point in the CSV — check which one actually tracks field there before "
                            "trusting it (V₂ω ∝ cos, not sin, so X₁f being right says nothing "
                            "about X₂f).",
                            classes="hint",
                        ),
                        muted=True,
                    )
                    yield card(
                        "Sample geometry (optional)",
                        field(
                            "hall_bar_length_um", "Hall bar length (µm)",
                            DEFAULTS["hall_bar_length_um"], kind="text", valid_empty=True,
                            hint="Current-path length between voltage probes. Leave blank if "
                                 "unknown — doesn't block the run.",
                        ),
                        field(
                            "hall_bar_width_um", "Hall bar width (µm)",
                            DEFAULTS["hall_bar_width_um"], kind="text", valid_empty=True,
                        ),
                        field(
                            "hall_bar_thickness_nm", "Film/channel thickness (nm)",
                            DEFAULTS["hall_bar_thickness_nm"], kind="text", valid_empty=True,
                        ),
                        field(
                            "field_angle_from_oop_deg", "External field angle from out-of-plane (°)",
                            DEFAULTS["field_angle_from_oop_deg"], kind="text", valid_empty=True,
                            hint="0° = fully out-of-plane (film normal), 90° = in-plane.",
                        ),
                        muted=True,
                    )

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

    # ── Sample picker ────────────────────────────────────────────────────────

    def _refresh_sample_options(self, *, select_value: Optional[str] = None) -> None:
        select = self.query_one("#sample_select", Select)
        options = sample_options(_DATA_DIR)
        select.set_options(options)
        if select_value is not None:
            select.value = select_value

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sample_select":
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
        raw["sinc_filter"] = self.query_one("#sinc_filter", Switch).value
        raw["differential"] = self.query_one("#differential", Switch).value
        raw["ac_coupling"] = self.query_one("#ac_coupling", Switch).value
        raw["enable_sweep"] = self.query_one("#enable_sweep", Switch).value
        raw["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
        raw["enable_phase_cal"] = self.query_one("#enable_phase_cal", Switch).value
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
        if "differential" in saved:
            self.query_one("#differential", Switch).value = bool(saved["differential"])
        if "ac_coupling" in saved:
            self.query_one("#ac_coupling", Switch).value = bool(saved["ac_coupling"])
        if "enable_sweep" in saved:
            self.query_one("#enable_sweep", Switch).value = bool(saved["enable_sweep"])
        if "enable_temperature" in saved:
            self.query_one("#enable_temperature", Switch).value = bool(saved["enable_temperature"])
        if "enable_phase_cal" in saved:
            self.query_one("#enable_phase_cal", Switch).value = bool(saved["enable_phase_cal"])
        if "order" in saved:
            try:
                self.query_one("#order", Select).value = int(saved["order"])
            except Exception:
                pass
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
        state["sinc_filter"] = self.query_one("#sinc_filter", Switch).value
        state["differential"] = self.query_one("#differential", Switch).value
        state["ac_coupling"] = self.query_one("#ac_coupling", Switch).value
        state["enable_sweep"] = self.query_one("#enable_sweep", Switch).value
        state["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
        state["enable_phase_cal"] = self.query_one("#enable_phase_cal", Switch).value
        state["order"] = int(self.query_one("#order", Select).value)
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""
        return state, errors

    # ── Reactivity ───────────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        self.refresh_summary()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "enable_sweep":
            self._set_magnet_fields_enabled(event.value)
        elif event.switch.id == "enable_temperature":
            self._set_temperature_fields_enabled(event.value)
        self.refresh_summary()

    def on_select_changed(self, event: Select.Changed) -> None:
        self.refresh_summary()

    def _set_magnet_fields_enabled(self, enabled: bool) -> None:
        for fid in MAGNET_FIELD_IDS:
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
            differential=state["differential"], ac_coupling=state["ac_coupling"],
            input_range_V=state["input_range_1f_V"],
            sample_rate_Hz=state["sample_rate_Hz"], filter=filt,
        )
        demod2_cfg = DemodConfig(
            device=state["follower_device"], demod_index=0, harmonic=2,
            differential=state["differential"], ac_coupling=state["ac_coupling"],
            input_range_V=state["input_range_2f_V"],
            sample_rate_Hz=state["sample_rate_Hz"], filter=filt,
        )
        run_ctx = allocate_run(
            _DATA_DIR, state["sample"], state["device"], MEASUREMENT_TYPE,
            temperature_setpoint_K=state["temperature_setpoint_K"],
        )
        acq_cfg = AcquisitionConfig(
            settling_time_s=state["settling_time_s"],
            n_averages=state["n_averages"],
            output_file=str(run_ctx.raw_path),
        )

        magnet_cfg = None
        gauss_cfg = None
        currents_A = None
        if state["enable_sweep"]:
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
            currents_A = bidirectional_current_sweep(
                i_min=state["i_min_A"], i_max=state["i_max_A"], n_points=state["n_points"],
            )

        temp_cfg = None
        if state["enable_temperature"]:
            uids = parse_sensor_uids(state["temperature_sensor_uids"])
            if uids:
                temp_cfg = TemperatureControllerConfig(
                    visa_resource=state["temperature_visa_resource"],
                    sensor_uids=uids,
                )

        geometry_cfg = SampleGeometryConfig(
            hall_bar_length_um=state["hall_bar_length_um"],
            hall_bar_width_um=state["hall_bar_width_um"],
            hall_bar_thickness_nm=state["hall_bar_thickness_nm"],
            field_angle_from_oop_deg=state["field_angle_from_oop_deg"],
        )

        # Geometry/dimensions are recorded per-row in the CSV (via
        # build_run_metadata) and belong in sample.yaml long-term — they are
        # deliberately NOT duplicated into header_extra/index.csv here.
        header_extra = {
            "excitation_frequency_Hz": state["frequency_Hz"],
            "excitation_amplitude_V": state["amplitude_V"],
            "series_R_ohm": state["series_R_ohm"],
            "demod_time_constant_s": state["time_constant_s"],
            "demod_order": state["order"],
            "n_averages": state["n_averages"],
            "settling_time_s": state["settling_time_s"],
        }
        if state["enable_sweep"]:
            header_extra["field_sweep_A"] = [state["i_min_A"], state["i_max_A"], state["n_points"]]

        return MeasurementPlan(
            daq_host=state["daq_host"], daq_port=state["daq_port"],
            leader=state["leader_device"], follower=state["follower_device"],
            out_cfg=out_cfg, demod1_cfg=demod1_cfg, demod2_cfg=demod2_cfg,
            acq_cfg=acq_cfg, magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, currents_A=currents_A,
            temp_cfg=temp_cfg,
            phase_cal_enabled=state["enable_phase_cal"],
            phase_cal_current_A=state["phase_cal_current_A"],
            phase_cal_n_averages=state["phase_cal_n_averages"],
            phase_cal_max_iterations=state["phase_cal_max_iterations"],
            geometry_cfg=geometry_cfg,
            run_ctx=run_ctx,
            temperature_setpoint_K=state["temperature_setpoint_K"],
            cooldown=state["cooldown"], header_extra=header_extra,
        )


def main() -> None:
    MFLIDualHarmonicApp().run()


if __name__ == "__main__":
    main()
