#!/usr/bin/env python3
"""
Textual TUI for sot/sot_pulsed_switching.py
=========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-08

The switching curve: R_xy vs. pulse amplitude, one pulse per amplitude, at a
single static tilted field. The 4200A PMU delivers the write pulse; after a
fixed delay the 6221 forces ±I_read and the 2182 reads V_xy across the
transverse arms. One row per amplitude. For switching-probability statistics,
re-run the sweep N times.

Run:  python sot_pulsed_switching_tui.py
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

from sot.sot_pulsed_switching import (
    _READ_COMPLIANCE_CEILING_V,
    _READ_CURRENT_CEILING_A,
    _check_read_safety,
    AmplitudePoint,
    GaussmeterConfig,
    Keithley4200AConfig,
    MagnetConfig,
    PMUPulseConfig,
    ReadConfig,
    SourceConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    configure_pmu_pulse,
    connect_4200a,
    connect_gaussmeter,
    connect_magnet,
    connect_source,
    connect_temperature_controller,
    connect_voltmeter,
    list_user_libraries,
    ramp_current_to_zero,
    run_measurement,
    set_magnet_current,
    shutdown_4200a,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_source,
    shutdown_temperature_controller,
)
from sot.sot_pulsed_switching import _six221_output_off
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

log = logging.getLogger("sot_pulsed_switching_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "sot_pulsed_switching_tui_settings.json"

MEASUREMENT_TYPE = "SOTPS"

SOT_PULSED_DESCRIPTION = (
    "SOT switching curve: the 4200A PMU fires ONE write pulse per amplitude into "
    "the main channel through RPM1, then after a fixed delay the 6221 forces "
    "±I_read through the same path while the 2182 reads V_xy across the Hall arms "
    "(reversal-averaged, which cancels the thermal EMF and the 2182's offset). "
    "One row per amplitude, at one static field held slightly out of plane so "
    "the two in-plane remanent states read as different R_xy. Make the amplitude "
    "list a full loop (up then down) — the sweep sets each pulse's starting "
    "state, which is what gives the hysteresis. The pulse runs the KULT module "
    "instruments/kult/bridge_sot_pulse.c, which routes RPM1 back to the SMU on "
    "exit. Re-run the whole sweep for switching-probability statistics, or at "
    "the opposite field sign for the ±H_z control."
)

# Current MEASURE ceiling with a 4225-RPM on the PMU 10 V range. Above this the
# pulse current reads back overflowed rather than erroring (the KULT module sets
# KI_LIM_MODE=KI_VALUE), so build_summary() warns rather than blocks.
_RPM_10V_IMEAS_MAX_A = 0.01

DEFAULTS: dict = {
    # 4200A / PMU
    "k4200_visa_resource": "GPIB0::17::INSTR",
    "pmu_library": "bridge_sot",
    "pmu_module": "bridge_sot_pulse",
    "pmu_channel": "1",
    "pmu_id": "PMU1",
    "amplitude_start_V": "0.2",
    "amplitude_stop_V": "2.0",
    "amplitude_step_V": "0.2",
    "amplitude_bidirectional": True,
    "pulse_width_s": "1e-7",
    "pulse_rise_s": "2e-8",
    "pulse_fall_s": "2e-8",
    "pulse_period_s": "1e-3",
    "pulse_delay_s": "0",
    "n_pulses": "1",
    "pmu_sample_rate": "2e8",
    "pmu_meas_start_perc": "0.75",
    "pmu_meas_stop_perc": "0.90",
    "pmu_dut_res_ohm": "1000",
    "pmu_v_range_V": "10",
    "pmu_i_range_A": "0.01",
    "pmu_v_limit_V": "5.0",
    "pmu_return_names": ("pulse_voltage_measured_V, pulse_current_measured_A, "
                         "pulse_base_voltage_V, pulse_base_current_A"),
    # delayed R_xy read (6221 + 2182)
    "sense_current_A": "1e-4",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "nplc": "5",
    "auto_range": True,
    "n_reversals": "5",
    "settle_after_enable_s": "0.3",
    "delay_after_pulse_s": "1.0",
    # static field
    "magnet_current_A": "1.5",
    "field_angle_from_oop_deg": "85",
    "field_settle_tolerance_mT": "0.05",
    # identity
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    # instrument addresses
    "source_visa_resource": "GPIB0::20::INSTR",
    "voltmeter_visa_resource": "GPIB0::7::INSTR",
    "magnet_visa_resource": "GPIB0::6::INSTR",
    "current_limit_A": "35",
    "magnet_voltage_compliance_V": "15.0",
    "ramp_step_A": "0.1",
    "ramp_delay_s": "0.05",
    "gaussmeter_visa_resource": "GPIB0::12::INSTR",
    "gaussmeter_n_averages": "10",
    "gaussmeter_read_delay_s": "0.05",
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
}

NUMERIC_FIELDS: dict = {
    "pmu_channel": int,
    "amplitude_start_V": float,
    "amplitude_stop_V": float,
    "amplitude_step_V": float,
    "pulse_width_s": float,
    "pulse_rise_s": float,
    "pulse_fall_s": float,
    "pulse_period_s": float,
    "pulse_delay_s": float,
    "n_pulses": int,
    "pmu_sample_rate": float,
    "pmu_meas_start_perc": float,
    "pmu_meas_stop_perc": float,
    "pmu_dut_res_ohm": float,
    "pmu_v_range_V": float,
    "pmu_i_range_A": float,
    "pmu_v_limit_V": float,
    "sense_current_A": float,
    "compliance_V": float,
    "source_delay_s": float,
    "nplc": float,
    "n_reversals": int,
    "settle_after_enable_s": float,
    "delay_after_pulse_s": float,
    "magnet_current_A": float,
    "field_angle_from_oop_deg": float,
    "field_settle_tolerance_mT": float,
    "current_limit_A": float,
    "magnet_voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
}
TEXT_FIELDS = ["k4200_visa_resource", "pmu_library", "pmu_module",
               "pmu_id", "pmu_return_names", "device", "cooldown",
               "source_visa_resource", "voltmeter_visa_resource", "magnet_visa_resource",
               "gaussmeter_visa_resource", "temperature_visa_resource",
               "temperature_sensor_uids", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K"]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]

# Every Switch id on the form. Hardcoded in collect_raw / _load_settings /
# parse_state -- they must move together, and parse_state runs on every
# keystroke, so a stale entry here is an immediate crash.
SWITCH_FIELD_IDS = ("auto_range", "enable_temperature", "amplitude_bidirectional")


def parse_sensor_uids(raw: str) -> tuple:
    uids = [u.strip() for u in raw.split(",") if u.strip()]
    return tuple(uids[:2])


def parse_return_names(raw: str) -> tuple:
    return tuple(n.strip() for n in raw.split(",") if n.strip())


def _resolve_amplitudes(state: dict) -> tuple[list[float], Optional[str]]:
    """(list, None) or ([], error) — the amplitude sweep from start/stop/step
    (+ the bidirectional toggle). Shared by parse_state and the tests."""
    try:
        if state["amplitude_start_V"] == state["amplitude_stop_V"]:
            raise ValueError("Amplitude start and stop must differ.")
        return [float(v) for v in linear_sweep(
            state["amplitude_start_V"], state["amplitude_stop_V"],
            state["amplitude_step_V"],
            bidirectional=state["amplitude_bidirectional"])], None
    except ValueError as exc:
        return [], str(exc)


# ── formatting helpers (per-TUI copies) ─────────────────────────────────────

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
    pmu_cfg: PMUPulseConfig
    src_cfg: SourceConfig       # 6221 — forces ±I_read through the main channel
    volt_cfg: VoltmeterConfig   # 2182 — reads V_xy across the transverse arms
    read_cfg: ReadConfig
    magnet_cfg: MagnetConfig
    gauss_cfg: GaussmeterConfig
    amplitudes_V: List[float]
    magnet_current_A: float
    field_angle_from_oop_deg: Optional[float]
    field_settle_tolerance_mT: float
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
        return len(self.amplitudes_V)


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


# ── widget helpers (per-TUI copies) ────────────────────────────────────────

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

    # PMU
    if not state["pmu_module"].strip():
        errors.append("PMU module name is empty — run `UL` on the 4200A and set "
                      "the pulse library/module (Instrument config card).")
    if state["pmu_channel"] not in (1, 2):
        errors.append("PMU channel is 1 or 2.")
    if state["pmu_v_range_V"] not in (10.0, 40.0):
        errors.append("PMU voltage range must be 10 or 40 V.")
    elif state["pmu_v_range_V"] == 40.0:
        warnings.append("40 V PMU range — check these before running: (1) the DUT: the bare "
                        "PMU can source up to 0.8 A, so keep v_limit_V / your channel R where "
                        "the pulse current stays safe, and watch pulse_current_measured_A. "
                        "(2) The 4225-RPM is a 10 V device: a 40 V pulse through it may error "
                        "or bypass the RPM — fire one pulse and confirm EX returns 0. "
                        "(3) The 2182 and the standby 6221 tolerate the ~10-20 V pulse "
                        "transient on the shared bus (2182 CH1 limit 120 V, 6221 ±105 V "
                        "compliance rating) — keep the 2182 leads short and away from the "
                        "pulse path. See the module docstring's 'The 40 V range' section.")
    if state["pulse_width_s"] <= 0:
        errors.append("Pulse width must be > 0 s.")
    if (state["pulse_period_s"] < state["pulse_delay_s"] + state["pulse_width_s"]
            + state["pulse_rise_s"] + state["pulse_fall_s"]):
        errors.append("Pulse period must be ≥ delay + width + rise + fall.")
    edge_min = 100e-9 if state["pmu_v_range_V"] == 40.0 else 20e-9
    if state["pulse_width_s"] < 60e-9:
        errors.append("Pulse width must be ≥ 60 ns (4225-PMU minimum).")
    if min(state["pulse_rise_s"], state["pulse_fall_s"]) < edge_min:
        errors.append(f"Rise/fall must be ≥ {format_si(edge_min, 's')} on the "
                      f"{state['pmu_v_range_V']:g} V range.")
    top_s = state["pulse_width_s"] - 0.5 * (state["pulse_rise_s"] + state["pulse_fall_s"])
    if top_s <= 0:
        errors.append(f"No flat pulse top: width must exceed ½·(rise+fall) = "
                      f"{format_si(0.5 * (state['pulse_rise_s'] + state['pulse_fall_s']), 's')} "
                      "(PMU width is FWHM). Shorten the edges or widen the pulse — the bench "
                      "returns -826 otherwise.")
    if not 0.0 <= state["pmu_meas_start_perc"] < state["pmu_meas_stop_perc"] <= 1.0:
        errors.append("Need 0 ≤ measure-window start < stop ≤ 1.")
    if state["n_pulses"] < 1:
        errors.append("Pulses per cycle must be ≥ 1.")

    amps = state.get("amplitude_list", [])
    if state.get("amplitude_parse_error"):
        errors.append(f"Pulse amplitudes: {state['amplitude_parse_error']}")
    else:
        over = [a for a in amps if abs(a) > state["pmu_v_limit_V"]]
        if over:
            errors.append(f"Pulse amplitude(s) {over} V exceed the PMU voltage limit "
                          f"±{state['pmu_v_limit_V']:g} V.")
        over_range = [a for a in amps if abs(a) > state["pmu_v_range_V"]]
        if over_range:
            errors.append(f"Pulse amplitude(s) {over_range} V exceed the "
                          f"{state['pmu_v_range_V']:g} V PMU range.")
        loop = " loop" if state["amplitude_bidirectional"] else ""
        info.append(f"Amplitude sweep: {len(amps)} pulses "
                    f"{state['amplitude_start_V']:g} → {state['amplitude_stop_V']:g} V "
                    f"step {state['amplitude_step_V']:g}{loop}" if amps else "")
        if not state["amplitude_bidirectional"]:
            warnings.append("One-way sweep — turn on 'Sweep up then back down' for a "
                            "hysteresis loop; the sweep is what sets each pulse's starting "
                            "state.")

    # read (6221 + 2182) — the 6221 shares the main-channel pins with the PMU,
    # so a fat-fingered current/compliance lands on the 2182 and the disabled
    # PMU output. Block at the absolute ceilings, warn below them.
    if state["n_reversals"] < 1:
        errors.append("Reversal pairs per read must be ≥ 1.")
    if state["sense_current_A"] <= 0:
        errors.append("6221 sense current must be > 0 A.")
    elif state["sense_current_A"] > _READ_CURRENT_CEILING_A:
        errors.append(f"6221 sense current {format_si(state['sense_current_A'], 'A')} exceeds the "
                      f"{format_si(_READ_CURRENT_CEILING_A, 'A')} safety ceiling — the Hall read "
                      "needs µA–mA; check for a mistyped exponent.")
    elif state["sense_current_A"] > 1e-3:
        warnings.append(f"6221 sense current {format_si(state['sense_current_A'], 'A')} is large "
                        "for a read — it flows continuously through the channel; keep it well "
                        "below the switching current.")
    if state["compliance_V"] <= 0:
        errors.append("6221 compliance must be > 0 V.")
    elif state["compliance_V"] > _READ_COMPLIANCE_CEILING_V:
        errors.append(f"6221 compliance {state['compliance_V']:g} V exceeds the "
                      f"{_READ_COMPLIANCE_CEILING_V:g} V safety ceiling — on an open contact the "
                      "6221 rails to this across the shared bus, onto the 2182.")
    elif state["compliance_V"] > 5.0:
        warnings.append(f"6221 compliance {state['compliance_V']:g} V — the Hall read needs "
                        "< 1 V of headroom; a lower value limits what an open contact can put "
                        "on the shared bus.")

    # field
    if abs(state["magnet_current_A"]) > state["current_limit_A"]:
        errors.append(f"Static magnet current {state['magnet_current_A']:g} A exceeds the "
                      f"magnet limit ±{state['current_limit_A']:g} A.")

    n = max(1, len(amps))
    per_point_s = (state["delay_after_pulse_s"] + state["settle_after_enable_s"]
                   + state["n_reversals"] * 2 * max(state["source_delay_s"], state["nplc"] / 50.0)
                   + 0.2)
    info.append(f"{n} amplitudes, one pulse each")
    info.append(f"Estimated run time ≈ {format_duration(n * per_point_s)} "
                f"({format_si(state['delay_after_pulse_s'], 's')} post-pulse wait dominates)")
    info.append(f"For P(V) / I50 statistics, re-run this sweep several times.")
    info.append(f"PMU module: {state['pmu_library']}/{state['pmu_module'] or '<unset>'} "
                f"({state['pmu_id']} ch {state['pmu_channel']})")

    # Display-only current estimate off the load-line DUT resistance. It is a hint
    # for picking amplitudes; the honest pulse axis is the module's measured
    # pulse_current_measured_A, and pmu_dut_res_ohm is never written as data.
    r_ch = state.get("pmu_dut_res_ohm", 0.0)
    if r_ch > 0 and amps:
        i_lo, i_hi = min(amps) / r_ch, max(amps) / r_ch
        info.append(f"At DUT R ≈ {r_ch:g} Ω: pulses ≈ "
                    f"{format_si(i_lo, 'A')}…{format_si(i_hi, 'A')}; 6221 read current "
                    f"{format_si(state['sense_current_A'], 'A')} → "
                    f"≈ {format_si(state['sense_current_A'] * r_ch, 'V')} across the channel")
        if (state["pmu_v_range_V"] == 10.0
                and max(abs(i_lo), abs(i_hi)) > _RPM_10V_IMEAS_MAX_A
                and state["pmu_i_range_A"] <= _RPM_10V_IMEAS_MAX_A):
            warnings.append(
                f"Estimated pulse current exceeds the RPM's "
                f"{format_si(_RPM_10V_IMEAS_MAX_A, 'A')} measure ceiling on the 10 V range — "
                "pulse_current_measured_A will read overflowed, not error. The pulse itself "
                "still fires.")

    info.append(f"Static field via magnet current {state['magnet_current_A']:g} A "
                f"(measured live by the 475). Re-run at the opposite sign for ±H_z.")
    info.append(f"Field mount tilt recorded as field_angle_from_oop_deg = "
                f"{state['field_angle_from_oop_deg']:g}° (set to your real mount angle).")

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
    return f"{preview}_<I_mag A>_<timestamp>.csv"


# ── live plot ──────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("SOT pulsed switching — live")
    except Exception:
        pass
    ax.set_xlabel("Pulse amplitude (V)")
    ax.set_ylabel("R_xy (Ω)")
    ax.set_title("Live — R_xy vs pulse amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # line + markers, in acquisition order → the connecting line shows the
    # sweep direction (up-leg then down-leg for a bidirectional list).
    (pts,) = ax.plot([], [], "o-", ms=4, lw=1, alpha=0.6, color="#2E3192")
    xs: list = []
    ys: list = []

    def _drain(_frame=None):
        changed = False
        while True:
            try:
                rec = queue.get_nowait()
            except Exception:
                break
            xs.append(rec["pulse_amplitude_V"])
            ys.append(rec["hall_resistance_ohm"])
            changed = True
        if changed:
            pts.set_data(xs, ys)
            ax.relim()
            ax.autoscale_view()
        return (pts,)

    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path) -> None:
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    # records are in acquisition order → the line traces the sweep direction.
    ax.plot([r["pulse_amplitude_V"] for r in records],
            [r["hall_resistance_ohm"] for r in records],
            "o-", ms=4, lw=1, alpha=0.6, color="#2E3192")
    ax.set_xlabel("Pulse amplitude (V)")
    ax.set_ylabel("R_xy (Ω)")
    ax.set_title("R_xy vs pulse amplitude")
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
            "amp #", "V_pulse (V)", "I_pulse (A)", "V_xy (V)", "R_xy (Ω)", "T1 (K)")
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
        i_pulse = record.get("pulse_current_measured_A")
        t1 = record.get("temperature_1_K")
        table.add_row(
            str(record["amplitude_index"] + 1),
            f"{record['pulse_amplitude_V']:.4g}",
            f"{i_pulse:.4e}" if i_pulse is not None else "—",
            f"{record['hall_voltage_V']:.4e}",
            f"{record['hall_resistance_ohm']:.5g}",
            f"{t1:.3f}" if t1 is not None else "—",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(1)
        self._set_status(f"Amplitude {len(self._records)} / {self.plan.total_points}.")

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
                                     MEASUREMENT_TYPE, "Rxy_vs_amp")
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
            self._set_status("Abort requested — finishing this amplitude, then ramping the 6221 + magnet down …")

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
        k4200 = source = voltmeter = magnet = gaussmeter = temp_ctrl = None
        try:
            self._set_status_threadsafe("Connecting to Keithley 4200A (KXCI) …")
            k4200 = connect_4200a(plan.k4200_cfg)
            try:
                log.info("Installed user libraries (UL):\n%s", list_user_libraries(k4200))
            except Exception:
                log.warning("Could not read `UL` — set the PMU module name from the 4200A manually.")
            configure_pmu_pulse(k4200, plan.pmu_cfg)

            # connect_source() returns with the 6221 already sourcing — re-check
            # the read limits here too, not only in build_summary.
            _check_read_safety(plan.read_cfg)
            self._set_status_threadsafe("Connecting to Keithley 6221 + 2182 …")
            source = connect_source(plan.src_cfg)
            _six221_output_off(source)          # channel quiet before any pulse
            voltmeter = connect_voltmeter(plan.volt_cfg)

            self._set_status_threadsafe("Connecting to Kepco magnet + Lake Shore 475 …")
            magnet = connect_magnet(plan.magnet_cfg)
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)

            if plan.temp_cfg is not None:
                self._set_status_threadsafe("Connecting to MercuryiTC …")
                temp_ctrl = connect_temperature_controller(plan.temp_cfg)

            self._set_status_threadsafe(f"Ramping magnet to the static field ({plan.magnet_current_A:g} A) …")
            set_magnet_current(magnet, plan.magnet_cfg, plan.magnet_current_A,
                               gaussmeter, plan.gauss_cfg, plan.field_settle_tolerance_mT,
                               self._stop_event)

            ctx = allocate_run(plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                               temperature_setpoint_K=plan.temperature_setpoint_K,
                               key_axis=("current_A", plan.magnet_current_A), series=plan.series)
            self._ctx = ctx
            write_csv = make_incremental_writer(
                ctx.raw_path,
                lambda records: build_header_fields(plan, ctx, records,
                                                    status="in_progress", comment=""))

            points = [AmplitudePoint(amplitude_V=float(v)) for v in plan.amplitudes_V]

            self._set_status_threadsafe("Running the switching sweep …")
            iter_error: Optional[BaseException] = None
            try:
                run_measurement(
                    k4200, plan.pmu_cfg, source, voltmeter, plan.read_cfg, points,
                    stop_event=self._stop_event, on_point=self._make_on_point(),
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                    magnet_current_A=plan.magnet_current_A,
                    field_angle_from_oop_deg=plan.field_angle_from_oop_deg,
                    write_csv=write_csv, output_file=str(ctx.raw_path))
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
            # 6221 down first (it shares the channel pin), then the 4200A, then
            # the magnet — never ramp an inductive field while the DUT still
            # carries current.
            if source is not None:
                safe_shutdown("6221 (ramp)", lambda: ramp_current_to_zero(source))
                safe_shutdown("6221", lambda: shutdown_source(source))
            if k4200 is not None:
                # channels=() — this program never forces the 4200A SMUs.
                safe_shutdown("4200A", lambda: shutdown_4200a(k4200, channels=()))
            if magnet is not None:
                safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
            if gaussmeter is not None:
                safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
            if temp_ctrl is not None:
                safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))
            self.app.call_from_thread(self._on_finished, final)


# ── app / form ─────────────────────────────────────────────────────────────

class SOTPulsedSwitchingApp(App):
    TITLE = "SOT pulsed switching"
    SUB_TITLE = "4200A PMU pulse · delayed 6221/2182 R_xy · static tilted field"

    data_root: Path = _DEFAULT_DATA_DIR

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 46; border-left: solid $primary; padding: 1 2; overflow-y: auto; }
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
                        "Write pulse (4200A PMU)",
                        field("amplitude_start_V", "Amplitude start (V)",
                              DEFAULTS["amplitude_start_V"]),
                        field("amplitude_stop_V", "Amplitude stop (V)",
                              DEFAULTS["amplitude_stop_V"]),
                        field("amplitude_step_V", "Amplitude step (V)",
                              DEFAULTS["amplitude_step_V"],
                              validators=[Number(minimum=1e-12, failure_description="must be > 0")],
                              hint="One pulse per step."),
                        switch_field("amplitude_bidirectional",
                                     "Sweep up then back down (hysteresis loop)",
                                     DEFAULTS["amplitude_bidirectional"]),
                        field("pulse_width_s", "Pulse width (s)", DEFAULTS["pulse_width_s"]),
                        field("pulse_rise_s", "Rise time (s)", DEFAULTS["pulse_rise_s"]),
                        field("pulse_fall_s", "Fall time (s)", DEFAULTS["pulse_fall_s"]),
                        field("pulse_period_s", "Pulse period (s)", DEFAULTS["pulse_period_s"],
                              hint="≥ delay + width + rise + fall."),
                    )
                    yield card(
                        "Delayed R_xy read (6221 + 2182)",
                        field("delay_after_pulse_s", "Delay after pulse (s)",
                              DEFAULTS["delay_after_pulse_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")],
                              hint="Wait between pulse end and the read."),
                        field("sense_current_A", "6221 sense current (A)", DEFAULTS["sense_current_A"],
                              hint="Keep well below the switching current."),
                        field("n_reversals", "Reversal pairs per read", DEFAULTS["n_reversals"],
                              kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("settle_after_enable_s", "6221 settle after enable (s)",
                              DEFAULTS["settle_after_enable_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                    )
                    yield card(
                        "Static field (Kepco magnet)",
                        field("magnet_current_A", "Static magnet current (A)",
                              DEFAULTS["magnet_current_A"],
                              hint="One value. Re-run at the opposite sign for the ±H_z control."),
                        field("field_angle_from_oop_deg", "Field mount tilt from OOP (deg)",
                              DEFAULTS["field_angle_from_oop_deg"],
                              hint="0 = out-of-plane, 90 = in-plane. Recorded, not set."),
                        field("field_settle_tolerance_mT", "Field settle tolerance (mT)",
                              DEFAULTS["field_settle_tolerance_mT"]),
                    )
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature", "Log temperature (MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                    )

                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Keithley 4200A PMU (KXCI)",
                            field("k4200_visa_resource", "KXCI VISA resource",
                                  DEFAULTS["k4200_visa_resource"], kind="text",
                                  hint="GPIB0::17::INSTR  or  TCPIP0::<ip>::1225::SOCKET"),
                            field("pmu_library", "KULT pulse library",
                                  DEFAULTS["pmu_library"], kind="text",
                                  hint="Confirm against the `UL` output in the run log."),
                            field("pmu_module", "KULT pulse module name", DEFAULTS["pmu_module"],
                                  kind="text",
                                  hint="Default = instruments/kult/bridge_sot_pulse.c — compile "
                                       "it in KULT first (see that folder's README)."),
                            field("pmu_channel", "PMU channel", DEFAULTS["pmu_channel"], kind="integer"),
                            field("pmu_id", "PMU card name", DEFAULTS["pmu_id"], kind="text",
                                  hint="e.g. PMU1 (lowest-numbered slot)."),
                            field("pmu_return_names", "Module return params (comma-sep)",
                                  DEFAULTS["pmu_return_names"], kind="text",
                                  hint="Order must match the module's outputs. Blank = none, "
                                       "and the measured pulse columns stay empty."),
                            field("pmu_v_range_V", "PMU voltage range (V)",
                                  DEFAULTS["pmu_v_range_V"], hint="10 or 40."),
                            field("pmu_i_range_A", "PMU current measure range (A)",
                                  DEFAULTS["pmu_i_range_A"],
                                  hint="With an RPM on the 10 V range the ceiling is 0.01 A."),
                            field("pmu_v_limit_V", "Pulse amplitude software limit (V)",
                                  DEFAULTS["pmu_v_limit_V"]),
                            field("pulse_delay_s", "Pulse delay before rise (s)",
                                  DEFAULTS["pulse_delay_s"],
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")],
                                  hint="Dead time before the rise. Normally 0."),
                            field("n_pulses", "Pulses per point (burst-average)",
                                  DEFAULTS["n_pulses"], kind="integer",
                                  hint="PMU averages N identical pulses for the measured V/I "
                                       "readback only. Leave at 1 for switching — N means N "
                                       "switching attempts per amplitude."),
                            field("pmu_sample_rate", "PMU sample rate (S/s)",
                                  DEFAULTS["pmu_sample_rate"]),
                            field("pmu_meas_start_perc", "Spot-mean window start (0-1)",
                                  DEFAULTS["pmu_meas_start_perc"]),
                            field("pmu_meas_stop_perc", "Spot-mean window stop (0-1)",
                                  DEFAULTS["pmu_meas_stop_perc"]),
                            field("pmu_dut_res_ohm", "DUT resistance for load-line (Ω)",
                                  DEFAULTS["pmu_dut_res_ohm"],
                                  hint="Set near the real channel R (4-probe it first). "
                                       "Also drives the sidebar current estimate."),
                            muted=True,
                        )
                        yield card(
                            "6221 / 2182",
                            field("source_visa_resource", "6221 (current source)",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            field("voltmeter_visa_resource", "2182 (Hall voltage)",
                                  DEFAULTS["voltmeter_visa_resource"], kind="text"),
                            field("compliance_V", "6221 compliance (V)", DEFAULTS["compliance_V"],
                                  hint="Keep low — caps what an open contact can put on the "
                                       "shared bus. Read needs < 1 V."),
                            field("source_delay_s", "6221 source delay (s)", DEFAULTS["source_delay_s"]),
                            field("nplc", "2182 NPLC", DEFAULTS["nplc"]),
                            switch_field("auto_range", "2182 auto-range", DEFAULTS["auto_range"]),
                            muted=True,
                        )
                        yield card(
                            "Kepco magnet + Lake Shore 475",
                            field("magnet_visa_resource", "Kepco VISA resource",
                                  DEFAULTS["magnet_visa_resource"], kind="text"),
                            field("current_limit_A", "Magnet current limit (A)",
                                  DEFAULTS["current_limit_A"]),
                            field("magnet_voltage_compliance_V", "Magnet voltage compliance (V)",
                                  DEFAULTS["magnet_voltage_compliance_V"]),
                            field("ramp_step_A", "Magnet ramp step (A)", DEFAULTS["ramp_step_A"]),
                            field("ramp_delay_s", "Magnet ramp delay (s)", DEFAULTS["ramp_delay_s"]),
                            field("gaussmeter_visa_resource", "Lake Shore 475 VISA resource",
                                  DEFAULTS["gaussmeter_visa_resource"], kind="text"),
                            field("gaussmeter_n_averages", "475 readings averaged",
                                  DEFAULTS["gaussmeter_n_averages"], kind="integer"),
                            field("gaussmeter_read_delay_s", "475 read delay (s)",
                                  DEFAULTS["gaussmeter_read_delay_s"]),
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
                yield Static(SOT_PULSED_DESCRIPTION, classes="card-desc")
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
        for sid in SWITCH_FIELD_IDS:
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
        for sid in SWITCH_FIELD_IDS:
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
        for sid in SWITCH_FIELD_IDS:
            state[sid] = self.query_one(f"#{sid}", Switch).value
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""

        state["amplitude_list"], state["amplitude_parse_error"] = _resolve_amplitudes(state)
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
        k4200_cfg = Keithley4200AConfig(visa_resource=state["k4200_visa_resource"])
        pmu_cfg = PMUPulseConfig(
            library=state["pmu_library"] or "bridge_sot",
            module=state["pmu_module"],
            pmu_channel=state["pmu_channel"], pmu_id=state["pmu_id"] or "PMU1",
            width_s=state["pulse_width_s"], rise_s=state["pulse_rise_s"],
            fall_s=state["pulse_fall_s"], period_s=state["pulse_period_s"],
            delay_s=state["pulse_delay_s"], n_pulses=state["n_pulses"],
            sample_rate=state["pmu_sample_rate"],
            meas_start_perc=state["pmu_meas_start_perc"],
            meas_stop_perc=state["pmu_meas_stop_perc"],
            dut_res_ohm=state["pmu_dut_res_ohm"],
            v_range_V=state["pmu_v_range_V"],
            i_range_A=state["pmu_i_range_A"], v_limit_V=state["pmu_v_limit_V"],
            return_names=parse_return_names(state["pmu_return_names"]),
        )
        read_cfg = ReadConfig(
            sense_current_A=state["sense_current_A"], compliance_V=state["compliance_V"],
            source_delay_s=state["source_delay_s"], nplc=state["nplc"],
            auto_range=state["auto_range"], n_reversals=state["n_reversals"],
            settle_after_enable_s=state["settle_after_enable_s"],
            delay_after_pulse_s=state["delay_after_pulse_s"],
        )
        src_cfg = SourceConfig(
            visa_resource=state["source_visa_resource"], sense_current_A=state["sense_current_A"],
            compliance_V=state["compliance_V"], source_delay_s=state["source_delay_s"],
        )
        volt_cfg = VoltmeterConfig(
            visa_resource=state["voltmeter_visa_resource"], nplc=state["nplc"],
            auto_range=state["auto_range"],
        )
        magnet_cfg = MagnetConfig(
            visa_resource=state["magnet_visa_resource"], current_limit_A=state["current_limit_A"],
            voltage_compliance_V=state["magnet_voltage_compliance_V"],
            ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"],
        )
        gauss_cfg = GaussmeterConfig(
            visa_resource=state["gaussmeter_visa_resource"], unit="T",
            n_averages=state["gaussmeter_n_averages"], read_delay_s=state["gaussmeter_read_delay_s"],
        )

        temp_cfg = None
        if state["enable_temperature"]:
            uids = parse_sensor_uids(state["temperature_sensor_uids"])
            if uids:
                temp_cfg = TemperatureControllerConfig(
                    visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

        # pmu_dut_res_ohm is a real pulse parameter (PMU load-line correction),
        # so it is recorded; the sidebar's per-amplitude current estimate is
        # derived from it rather than from a separate display-only field.
        header_extra = {
            "pmu_library": pmu_cfg.library,
            "pmu_module": pmu_cfg.module,
            "pulse_width_s": state["pulse_width_s"],
            "pulse_period_s": state["pulse_period_s"],
            "pmu_v_range_V": state["pmu_v_range_V"],
            "pmu_i_range_A": state["pmu_i_range_A"],
            "pmu_dut_res_ohm": state["pmu_dut_res_ohm"],
            "n_pulses": state["n_pulses"],
            "delay_after_pulse_s": state["delay_after_pulse_s"],
            "sense_current_A": state["sense_current_A"],
            "n_reversals": state["n_reversals"],
            "field_angle_from_oop_deg": state["field_angle_from_oop_deg"],
            "amplitude_start_V": state["amplitude_start_V"],
            "amplitude_stop_V": state["amplitude_stop_V"],
            "amplitude_step_V": state["amplitude_step_V"],
            "amplitude_bidirectional": state["amplitude_bidirectional"],
            "amplitudes_V": state["amplitude_list"],
        }
        return MeasurementPlan(
            k4200_cfg=k4200_cfg, pmu_cfg=pmu_cfg, src_cfg=src_cfg, volt_cfg=volt_cfg,
            read_cfg=read_cfg, magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg,
            amplitudes_V=state["amplitude_list"], magnet_current_A=state["magnet_current_A"],
            field_angle_from_oop_deg=state["field_angle_from_oop_deg"],
            field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
            data_root=self.data_root,
            sample=state["sample"], device=state["device"],
            temperature_setpoint_K=state["temperature_setpoint_K"],
            cooldown=state["cooldown"], header_extra=header_extra, series="",
            temp_cfg=temp_cfg,
        )


def main() -> None:
    SOTPulsedSwitchingApp().run()


if __name__ == "__main__":
    main()
