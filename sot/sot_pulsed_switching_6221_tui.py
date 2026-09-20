#!/usr/bin/env python3
"""
Textual TUI for sot/sot_pulsed_switching_6221.py
=========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-14

Same switching-curve TUI shape as sot_pulsed_switching_2h_tui.py, with the
4200A removed entirely: a single 6221 fires a hardware-timed current pulse
(WAVE mode, square function, one cycle — verified against the 6220/6221
User's Manual, no 2182 needed), then reads a chosen harmonic (default 2f)
of V_xy via AC + a single externally-referenced MFLI. See sot/sot_pulsed_
switching_6221.py's module docstring — especially "How the write pulse
works" — before running this on a real device: Joule heating at a
switching-level current still scales badly with pulse width even though
the timing itself is now hardware-timed, not a software guess.

Run:  python sot_pulsed_switching_6221_tui.py
"""

from __future__ import annotations

import itertools
import json
import logging
import multiprocessing as mp
import textwrap
import threading
from dataclasses import dataclass
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

from sot.sot_pulsed_switching_6221 import (
    _READ_COMPLIANCE_CEILING_V,
    _READ_CURRENT_CEILING_A,
    _WRITE_CURRENT_HARD_MAX_A,
    _check_pulse_currents,
    _check_extref_demod_conflict,
    _check_read_safety,
    _check_write_safety,
    ACSourceConfig,
    DemodConfig,
    ExtRefConfig,
    FilterConfig,
    GaussmeterConfig,
    MagnetConfig,
    PulsePoint,
    ReadConfig,
    TemperatureControllerConfig,
    WritePulseConfig,
    configure_demodulator,
    configure_external_reference,
    connect,
    connect_ac_source,
    connect_device,
    connect_gaussmeter,
    connect_magnet,
    connect_temperature_controller,
    run_measurement,
    set_magnet_current,
    shutdown_ac_source,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_temperature_controller,
)
from sot.sot_pulsed_switching_6221 import _six221_ac_output_off
from dc.dc_sweep_utils import linear_sweep, parse_value_list, safe_shutdown
from instruments.data_dir import DataDirPickerScreen, validate_directory
from instruments.field_geometry import field_direction_summary_line, render_ascii_field_diagram
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

log = logging.getLogger("sot_pulsed_switching_6221_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "sot_pulsed_switching_6221_tui_settings.json"

MEASUREMENT_TYPE = "SOT1I"

SOT_PULSED_6221_DESCRIPTION = (
    "SOT switching curve, 6221-only (NO 4200A, no 2182): a single Keithley 6221 "
    "fires ONE hardware-timed current pulse per amplitude (WAVE mode, square "
    "function, one cycle — verified against the 6220/6221 User's Manual, not "
    "Pulse Delta), then after a fixed delay sources an AC current (phase marker "
    "on its Trigger Link) while a single Zurich MFLI, externally referenced to "
    "that marker via its Aux Input, locks in on a chosen harmonic of V_xy (2f by "
    "default — the standard harmonic-Hall SOT signal; 1f reads the resistive "
    "AHE/PHE signal instead). One row per amplitude, at a static field held "
    "slightly out of plane. The write pulse has no independent rise/fall control "
    "and its true floor is range/load-dependent (spec: ~1-5 µs best case) — its "
    "actual elapsed hold time is measured and recorded every row "
    "(pulse_width_measured_s); read the module docstring's 'How the write pulse "
    "works' section, especially the Joule-heating scaling, before pushing width "
    "or current up. Make the amplitude list a full loop (up then down) for the "
    "hysteresis. Enter one or more assist-field currents (comma-separated) to "
    "scan the assist condition — each gets its own file."
)

DEFAULTS: dict = {
    # write pulse (6221 DC) — the switching axis is now current, not voltage
    "pulse_current_start_A": "1e-3",
    "pulse_current_stop_A": "10e-3",
    "pulse_current_step_A": "1e-3",
    "amplitude_bidirectional": True,
    "pulse_width_s": "1e-3",
    "pulse_compliance_V": "5.0",
    # delayed harmonic read (6221 AC + MFLI)
    "sense_current_values": "1e-4",
    "compliance_V": "2.0",
    "frequency_Hz": "977.0",
    "phasemarker_line": "1",
    "harmonic": "2",
    "n_averages": "50",
    "settle_after_enable_s": "1.0",
    "lock_timeout_s": "5.0",
    "delay_after_pulse_s": "1.0",
    # static field
    "magnet_current_A": "1.5",
    "field_theta_deg": "85",
    "field_phi_deg": "",
    "field_settle_tolerance_mT": "0.05",
    # identity
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    # instrument addresses
    "source_visa_resource": "GPIB0::20::INSTR",
    "mfli_host": "localhost",
    "mfli_port": "8004",
    "mfli_device": "dev1234",
    "aux_input_ch": "0",
    "osc_index": "0",
    "extref_index": "0",
    "pll_demod_index": "0",
    "automode": "4",
    "demod_index": "1",
    "input_ch": "0",
    "input_range_V": "1.0",
    "sample_rate_Hz": "857.0",
    "filter_time_constant_s": "0.3",
    "filter_order": "4",
    "differential": True,
    "ac_coupling": True,
    "filter_sinc": True,
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

# extrefs/N/automode options — see ExtRefConfig.automode's docstring in
# sot_pulsed_switching_6221.py for the full rationale.
AUTOMODE_OPTIONS: list[tuple[str, int]] = [
    ("2 — low bandwidth", 2),
    ("3 — high bandwidth", 3),
    ("4 — dynamic (auto)", 4),
]
AUTOMODE_HINT = ("2=most forgiving acquisition (marginal/noisy signal), "
                 "3=fastest tracking once locked, 4=auto-adapts (default).")

NUMERIC_FIELDS: dict = {
    "pulse_current_start_A": float,
    "pulse_current_stop_A": float,
    "pulse_current_step_A": float,
    "pulse_width_s": float,
    "pulse_compliance_V": float,
    "compliance_V": float,
    "frequency_Hz": float,
    "phasemarker_line": int,
    "harmonic": int,
    "n_averages": int,
    "settle_after_enable_s": float,
    "lock_timeout_s": float,
    "delay_after_pulse_s": float,
    "field_theta_deg": float,
    "field_settle_tolerance_mT": float,
    "mfli_port": int,
    "aux_input_ch": int,
    "osc_index": int,
    "extref_index": int,
    "pll_demod_index": int,
    "demod_index": int,
    "input_ch": int,
    "input_range_V": float,
    "sample_rate_Hz": float,
    "filter_time_constant_s": float,
    "filter_order": int,
    "current_limit_A": float,
    "magnet_voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
}
TEXT_FIELDS = ["device", "cooldown",
               "source_visa_resource", "mfli_host", "mfli_device",
               "magnet_visa_resource",
               "gaussmeter_visa_resource", "temperature_visa_resource",
               "temperature_sensor_uids", "magnet_current_A", "sense_current_values", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K", "field_phi_deg"]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]

# Every Switch id on the form. Hardcoded in collect_raw / _load_settings /
# parse_state -- they must move together, and parse_state runs on every
# keystroke, so a stale entry here is an immediate crash.
SWITCH_FIELD_IDS = ("differential", "ac_coupling", "filter_sinc",
                    "enable_temperature", "amplitude_bidirectional")


def parse_sensor_uids(raw: str) -> tuple:
    uids = [u.strip() for u in raw.split(",") if u.strip()]
    return tuple(uids[:2])


def _resolve_pulse_currents(state: dict) -> tuple[list[float], Optional[str]]:
    """(list, None) or ([], error) — the pulse-current sweep from
    start/stop/step (+ the bidirectional toggle). Shared by parse_state and
    the tests."""
    try:
        if state["pulse_current_start_A"] == state["pulse_current_stop_A"]:
            raise ValueError("Pulse current start and stop must differ.")
        return [float(v) for v in linear_sweep(
            state["pulse_current_start_A"], state["pulse_current_stop_A"],
            state["pulse_current_step_A"],
            bidirectional=state["amplitude_bidirectional"])], None
    except ValueError as exc:
        return [], str(exc)


def _resolve_magnet_currents(state: dict) -> tuple[list[float], Optional[str]]:
    """(list, None) or ([], error) — one or more static assist-field currents
    (comma-separated); each gets its own complete amplitude sweep, its own
    file. A single value behaves exactly as before."""
    try:
        return parse_value_list(state["magnet_current_A"]), None
    except ValueError as exc:
        return [], str(exc)


def _resolve_sense_currents(state: dict) -> tuple[list[float], Optional[str]]:
    """(list, None) or ([], error) — one or more 6221 AC read-current
    amplitudes (comma-separated); nests with the assist-current list
    (sense outer, since changing it means a full 6221 AC re-arm; magnet
    inner), each pair its own complete pulse-current sweep, its own file."""
    try:
        return parse_value_list(state["sense_current_values"]), None
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
    ac_cfg: ACSourceConfig       # 6221 — AC current + phase marker (read phase)
    pulse_cfg: WritePulseConfig  # 6221 — DC write pulse
    extref_cfg: ExtRefConfig     # MFLI ExtRef PLL, locked to the phase marker
    demod_cfg: DemodConfig       # MFLI — the chosen harmonic
    mfli_host: str
    mfli_port: int
    read_cfg: ReadConfig
    magnet_cfg: MagnetConfig
    gauss_cfg: GaussmeterConfig
    pulse_currents_A: List[float]
    magnet_currents_A: List[float]
    sense_currents_A: List[float]
    field_theta_deg: Optional[float]
    field_phi_deg: Optional[float]
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
    def series_values(self) -> List[tuple[float, float]]:
        """Cross product of sense (6221 AC) currents x assist-field currents
        -- one complete pulse-current sweep per pair, each saved to its own
        file. Sense current is outer (changing it means a full 6221 AC
        re-arm) and magnet is inner (just a ramp, no reconnect) -- see
        dc_spin_valve_tui.py for the same nested-product pattern."""
        return list(itertools.product(self.sense_currents_A, self.magnet_currents_A))

    @property
    def total_points(self) -> int:
        return len(self.pulse_currents_A) * max(1, len(self.series_values))


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                        status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """`extra` carries this iteration's own magnet_current_A on top of the
    plan-wide header_extra — see instruments/data_naming.py's allocate_run(),
    called fresh per assist-field-current iteration for this suite."""
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
    if extra:
        fields.update(extra)
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


def select_field(field_id: str, label_text: str, options: list[tuple[str, int]] | list[int],
                  default: int, *, hint: str = "") -> list:
    label = Label(label_text, classes="field-label")
    opts = [(str(o), o) for o in options] if options and not isinstance(options[0], tuple) else options
    sel = Select(opts, id=field_id, value=default, allow_blank=False)
    widgets = [label, sel]
    if hint:
        widgets.append(Label(hint, classes="hint"))
    widgets[-1].styles.margin = (0, 0, 1, 0)
    return widgets


def card(title: str, *groups, muted: bool = False) -> Vertical:
    children: list = [Static(title, classes="card-title")]
    for group in groups:
        children.extend(group) if isinstance(group, list) else children.append(group)
    return Vertical(*children, classes="stable-card" if muted else "param-card")


# ── summary ────────────────────────────────────────────────────────────────

def _near_multiple(f: float, m: float) -> bool:
    return min(f % m, m - f % m) < 1.0


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

    # write pulse
    amps = state.get("pulse_current_list", [])
    if state.get("pulse_current_parse_error"):
        errors.append(f"Pulse currents: {state['pulse_current_parse_error']}")
    else:
        over = [a for a in amps if abs(a) > _WRITE_CURRENT_HARD_MAX_A]
        if over:
            errors.append(f"Pulse current(s) {over} A exceed the 6221's hardware range "
                          f"±{_WRITE_CURRENT_HARD_MAX_A:g} A.")
        loop = " loop" if state["amplitude_bidirectional"] else ""
        info.append(f"Pulse sweep: {len(amps)} pulses "
                    f"{format_si(state['pulse_current_start_A'], 'A')} → "
                    f"{format_si(state['pulse_current_stop_A'], 'A')} step "
                    f"{format_si(state['pulse_current_step_A'], 'A')}{loop}" if amps else "")
        if not state["amplitude_bidirectional"]:
            warnings.append("One-way sweep — turn on 'Sweep up then back down' for a "
                            "hysteresis loop; the sweep is what sets each pulse's starting "
                            "state.")
    if state["pulse_width_s"] <= 0:
        errors.append("Pulse width must be > 0 s.")
    if state["pulse_compliance_V"] <= 0:
        errors.append("Pulse compliance must be > 0 V.")
    info.append("Hardware-timed write pulse (WAVE square, one cycle) — no independent rise/fall "
               "control, and the true floor is range/load-dependent. Check "
               "pulse_width_measured_s on the first run, and read the module docstring's "
               "'How the write pulse works' section: Joule heating scales as I²R·t, so a slower "
               "pulse at switching current can be far hotter than the 4200A-PMU variants' ns "
               "pulse. Start well below the expected switching current.")

    # read (6221 AC + MFLI)
    sense_currents = state.get("sense_currents_A", [])
    if state.get("sense_currents_parse_error"):
        errors.append(f"6221 AC current amplitude(s): {state['sense_currents_parse_error']}")
    else:
        zero = [i for i in sense_currents if i <= 0]
        over = [i for i in sense_currents if i > _READ_CURRENT_CEILING_A]
        large = [i for i in sense_currents if 1e-3 < i <= _READ_CURRENT_CEILING_A]
        if zero:
            errors.append("6221 AC current amplitude must be > 0 A.")
        elif over:
            errors.append(f"6221 AC current amplitude(s) {over} exceed the "
                          f"{format_si(_READ_CURRENT_CEILING_A, 'A')} safety ceiling — the Hall "
                          "read needs µA–mA; check for a mistyped exponent.")
        elif large:
            warnings.append(f"6221 AC current amplitude(s) {large} are large for a read — keep "
                            "them well below the switching current.")
    if state["compliance_V"] <= 0:
        errors.append("6221 read compliance must be > 0 V.")
    elif state["compliance_V"] > _READ_COMPLIANCE_CEILING_V:
        errors.append(f"6221 read compliance {state['compliance_V']:g} V exceeds the "
                      f"{_READ_COMPLIANCE_CEILING_V:g} V safety ceiling.")
    elif state["compliance_V"] > 5.0:
        warnings.append(f"6221 read compliance {state['compliance_V']:g} V — the Hall read "
                        "needs < 1 V of headroom.")

    if not 1e-3 <= state["frequency_Hz"] <= 1e5:
        errors.append("6221 AC frequency must be in [1 mHz, 100 kHz] (WAVE mode range).")
    elif _near_multiple(state["frequency_Hz"], 50.0) or _near_multiple(state["frequency_Hz"], 60.0):
        warnings.append(f"{state['frequency_Hz']:g} Hz is close to a 50/60 Hz line harmonic — "
                        "pick an offset frequency to avoid mains pickup.")
    if not 1 <= state["phasemarker_line"] <= 6:
        errors.append("Trigger Link phase-marker line must be 1-6.")
    if state["harmonic"] < 1:
        errors.append("Harmonic must be ≥ 1.")
    elif state["harmonic"] > 3:
        warnings.append(f"Reading the {state['harmonic']}f harmonic — expect a much smaller "
                        "signal than 1f/2f; you may need more averaging or a larger sense "
                        "current.")
    if state["n_averages"] < 1:
        errors.append("MFLI samples averaged per read must be ≥ 1.")
    if state["lock_timeout_s"] < 0:
        errors.append("PLL lock timeout must be ≥ 0 s.")

    # field
    currents = state.get("magnet_currents_A", [])
    if state.get("magnet_currents_parse_error"):
        errors.append(f"Magnet current(s): {state['magnet_currents_parse_error']}")
    else:
        over = [i for i in currents if abs(i) > state["current_limit_A"]]
        if over:
            errors.append(f"Static magnet current(s) {over} A exceed the magnet limit "
                          f"±{state['current_limit_A']:g} A.")

    n = max(1, len(amps))
    n_currents = max(1, len(currents))
    n_sense = max(1, len(sense_currents))
    n_files = n_currents * n_sense
    tc = state["filter_time_constant_s"]
    acquire_window_s = max(0.1, 3.0 * tc,
                           state["n_averages"] * 1.5 / max(state["sample_rate_Hz"], 1.0))
    per_point_s = (state["pulse_width_s"] + state["delay_after_pulse_s"] + state["lock_timeout_s"]
                   + state["settle_after_enable_s"] + acquire_window_s + 0.2)
    info.append(f"{n} amplitudes, one pulse each"
                + (f", × {n_files} files ({n_currents} assist current(s) x {n_sense} sense "
                   f"current(s)) = {n * n_files} total points"
                   if n_files > 1 else ""))
    info.append(f"Estimated run time ≈ {format_duration(n * n_files * per_point_s)} "
                f"(worst case — assumes the full lock timeout every point)")
    info.append(f"For P(V) / I50 statistics, re-run this sweep several times.")
    if sense_currents:
        info.append(f"AC excitation: {format_si(sense_currents[0], 'A')} peak @ "
                    f"{state['frequency_Hz']:g} Hz, {state['harmonic']}f read, phase marker on "
                    f"Trigger Link pin {state['phasemarker_line']} → MFLI Aux In "
                    f"{state['aux_input_ch'] + 1}")

    if state["pll_demod_index"] == state["demod_index"]:
        errors.append(
            f"PLL phase-detector demod index and signal demod index are both "
            f"{state['demod_index']} — extrefs/N/adcselect is read-only, so the PLL "
            "phase-detector demod can't double as the signal demod. Pick distinct indices.")

    if len(currents) > 1:
        cur_str = ", ".join(f"{i:g}" for i in currents)
        info.append(f"Assist field: {len(currents)} magnet currents ({cur_str} A) — each gets "
                    "its own complete amplitude sweep and its own file (measured live by the "
                    "475). Include a negative value for the ±H_z control.")
    elif currents:
        info.append(f"Static field via magnet current {currents[0]:g} A "
                    "(measured live by the 475). Comma-separate more values to scan the "
                    "assist field, or add the opposite sign for the ±H_z control.")
    info.append(field_direction_summary_line(state["field_theta_deg"], state.get("field_phi_deg")))

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
    n_files = max(1, len(state.get("magnet_currents_A", []))) * max(1, len(state.get("sense_currents_A", [])))
    suffix = " (one file per run)" if n_files > 1 else ""
    return f"{preview}_<I_mag A>_<timestamp>.csv{suffix}"


# ── live plot ──────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue", harmonic: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    ylabel = f"V_{harmonic}f (V)"
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("SOT pulsed switching (6221-only) — live")
    except Exception:
        pass
    ax.set_xlabel("Pulse current (A)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Live — {ylabel} vs pulse current")
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
                rec = queue.get_nowait()
            except Exception:
                break
            idx = rec.get("series_index", 0)
            if idx not in lines:
                label = rec.get("series_label")
                (line,) = ax.plot([], [], "o-", ms=4, lw=1, alpha=0.6,
                                  color=cmap(idx % 10), label=label)
                lines[idx] = line
                series_data[idx] = ([], [])
                new_series = True
            xs, ys = series_data[idx]
            xs.append(rec["pulse_current_A"])
            ys.append(rec["demod_R_V"])
            updated.add(idx)
        if updated:
            for idx in updated:
                xs, ys = series_data[idx]
                lines[idx].set_data(xs, ys)
            if new_series and any(l.get_label() and not l.get_label().startswith("_")
                                  for l in lines.values()):
                ax.legend(loc="best", fontsize=8)
            ax.relim()
            ax.autoscale_view()
        return tuple(lines.values())

    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path, harmonic: int,
                           plan: Optional["MeasurementPlan"] = None, comment: str = "") -> None:
    """`records` is ONE run's points -- with several currents each run is
    saved (and plotted) on its own, exactly like a manual run.

    `plan`/`comment` add a small "at a glance" text annotation (the
    static assist-field direction, the fixed 6221 AC read current, the
    operator's comment) for context not already in the filename -- the
    swept assist-field magnitude is already this run's key_axis, and the
    harmonic is already in the axis label. Called once when the run ends
    (comment="") and again, to overwrite the PNG in place, once the
    operator's comment is known."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylabel = f"V_{harmonic}f (V)"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["pulse_current_A"] for r in records],
            [r["demod_R_V"] for r in records],
            "o-", ms=4, lw=1, alpha=0.6, color="tab:blue")
    ax.set_xlabel("Pulse current (A)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs pulse current")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lines: list[str] = []
    if plan is not None:
        if plan.field_theta_deg is not None:
            lines.append(field_direction_summary_line(plan.field_theta_deg, plan.field_phi_deg))
        sense_currents = sorted({r["excitation_current_A_peak"] for r in records
                                  if r.get("excitation_current_A_peak") is not None})
        if len(sense_currents) == 1:
            lines.append(f"6221 AC read current: {format_si(sense_currents[0], 'A')}")
    if comment:
        lines.append(f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}")
    if lines:
        fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color="0.4", va="bottom")
        fig.subplots_adjust(bottom=0.08 + 0.045 * len(lines))

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
    #progress_row { height: auto; margin: 1 2; align: left middle; }
    #run_label { width: auto; padding: 0 2 0 0; text-style: bold; }
    #progress { margin: 0; }
    #results_table { height: 12; margin: 0 2 1 2; }
    #log { height: 1fr; margin: 0 2 1 2; border: solid $primary; }
    #runactionbar { height: 3; align: center middle; }
    """
    BINDINGS = [
        Binding("a", "abort", "Abort (safe shutdown)", show=True),
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
        self._run_contexts: list[RunContext] = []
        # The LAST run's PNG, stashed by _save_run_png so _on_status_comment
        # can re-save it in place once the operator's comment is known.
        self._png_path: Optional[Path] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Starting …", id="status_line")
        with Horizontal(id="progress_row"):
            yield Static("", id="run_label")
            yield ProgressBar(id="progress", total=self.plan.total_points, show_eta=False)
        yield DataTable(id="results_table", zebra_stripes=True, cursor_type="row")
        yield RichLog(id="log", max_lines=5000, markup=False, wrap=True)
        with Horizontal(id="runactionbar"):
            yield Button("Abort (safe shutdown)", id="abort_btn", variant="error")
            yield Button("Back", id="back_btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        h = self.plan.read_cfg.harmonic
        self.query_one("#results_table", DataTable).add_columns(
            "amp #", "I_mag (A)", "I_pulse (A)", "width meas (s)", f"V_{h}f (V)",
            "locked", "T1 (K)")
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
                                             args=(self._plot_queue, self.plan.read_cfg.harmonic),
                                             daemon=True)
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

    def _set_run_label(self, text: str) -> None:
        self.query_one("#run_label", Static).update(text)

    def _set_run_label_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_run_label, text)

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
            str(record["amplitude_index"] + 1),
            f"{record['magnet_current_A']:g}" if record.get("magnet_current_A") is not None else "—",
            f"{record['pulse_current_A']:.4g}",
            f"{record['pulse_width_measured_s']:.4g}",
            f"{record['demod_R_V']:.4e}",
            "yes" if record.get("reference_locked") else "no",
            f"{t1:.3f}" if t1 is not None else "—",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(1)
        self._set_status(f"Point {len(self._records)} / {self.plan.total_points}.")

    def _make_on_point(self, series_index: int, series_label: Optional[str]):
        def _cb(record: dict) -> None:
            record["series_index"] = series_index
            record["series_label"] = series_label
            self.app.call_from_thread(self._on_point, record)
        return _cb

    def _save_run_png(self, ctx: RunContext, iter_records: list[dict]) -> None:
        """One PNG per run (own run number), as if each current had been
        started by hand -- no combined overlay."""
        try:
            png_path = proc_path(self.plan.data_root, ctx.sample, ctx.run_str, ctx.device,
                                 MEASUREMENT_TYPE, "Vnf_vs_pulse")
            self._png_path = png_path
            _save_measurement_png(iter_records, png_path, self.plan.read_cfg.harmonic, plan=self.plan)
        except Exception:
            log.exception("Could not save plot PNG")

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False
        self.query_one("#abort_btn", Button).disabled = True
        self.app.push_screen(StatusCommentScreen(), self._on_status_comment)

    def _on_status_comment(self, result: Optional[tuple[str, str]]) -> None:
        # With several currents the runs before the last were implicitly
        # "skipped" -- left at the outcome status do_run() wrote right after
        # each one, with no comment. Only the last run, the one the operator
        # is looking at, gets the status/comment they entered.
        if result is None or not self._run_contexts:
            return
        status, comment = result
        series_idx = len(self._run_contexts) - 1
        ctx = self._run_contexts[series_idx]
        iter_records = [r for r in self._records if r.get("series_index", 0) == series_idx]
        extra = None
        if iter_records:
            extra = {"magnet_current_A": iter_records[0].get("magnet_current_A"),
                      "sense_current_A": iter_records[0].get("excitation_current_A_peak")}
        header_fields = build_header_fields(
            self.plan, ctx, iter_records, status=status, comment=comment, extra=extra)
        try:
            if iter_records or not ctx.raw_path.exists():
                write_record(ctx.raw_path, iter_records, header_fields)
            finalize_index_row(self.plan.data_root, ctx.sample, ctx.run_number, header_fields)
        except Exception:
            log.exception("Could not save final status/comment for run %d", ctx.run_number)

        if comment and self._png_path is not None:
            try:
                _save_measurement_png(iter_records, self._png_path, self.plan.read_cfg.harmonic,
                                      plan=self.plan, comment=comment)
            except Exception:
                log.exception("Could not re-save measurement plot PNG with comment")

    def action_abort(self) -> None:
        if self._measurement_running and not self._stop_event.is_set():
            self._stop_event.set()
            self._set_status("Abort requested — finishing this amplitude, then shutting the "
                             "6221 + magnet down …")

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
        source = daq = magnet = gaussmeter = temp_ctrl = None
        try:
            points = [PulsePoint(pulse_current_A=float(v)) for v in plan.pulse_currents_A]
            _check_write_safety(plan.pulse_cfg)
            _check_pulse_currents(points)
            _check_extref_demod_conflict(plan.demod_cfg, plan.extref_cfg)

            self._set_status_threadsafe("Connecting to MFLI …")
            daq = connect(plan.mfli_host, plan.mfli_port)
            connect_device(daq, plan.extref_cfg.device, interface="1GbE")
            configure_external_reference(daq, plan.extref_cfg, plan.ac_cfg.frequency_Hz)
            configure_demodulator(daq, plan.demod_cfg)

            self._set_status_threadsafe("Connecting to Kepco magnet + Lake Shore 475 …")
            magnet = connect_magnet(plan.magnet_cfg)
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)

            if plan.temp_cfg is not None:
                self._set_status_threadsafe("Connecting to MercuryiTC …")
                temp_ctrl = connect_temperature_controller(plan.temp_cfg)

            multi_sense = len(plan.sense_currents_A) > 1
            for series_idx, (I_sense, I_mag) in enumerate(plan.series_values):
                if self._stop_event.is_set():
                    break

                plan.ac_cfg.amplitude_A = I_sense
                plan.read_cfg.sense_current_A = I_sense

                label_parts = []
                if multi_sense:
                    label_parts.append(f"I_sense={I_sense:g}A")
                if len(plan.magnet_currents_A) > 1:
                    label_parts.append(f"I_mag={I_mag:g}A")
                label = ", ".join(label_parts) or None

                # Checked here too, not just by build_summary(): connect_ac_source()
                # immediately arms and starts the 6221 at plan.ac_cfg.amplitude_A —
                # catch a mistyped exponent before that, not after. Amplitude
                # requires a full re-arm -- tear down the previous amplitude's
                # source first.
                _check_read_safety(plan.read_cfg)
                if source is not None:
                    safe_shutdown("6221 AC source", lambda _s=source: shutdown_ac_source(_s))
                    source = None
                self._set_status_threadsafe(
                    f"Starting 6221 AC current source{f' ({I_sense:g} A)' if multi_sense else ''} …"
                )
                source = connect_ac_source(plan.ac_cfg)
                _six221_ac_output_off(source)          # channel quiet before any pulse

                self._set_status_threadsafe(f"Ramping magnet to {I_mag:g} A …")
                set_magnet_current(magnet, plan.magnet_cfg, I_mag,
                                   gaussmeter, plan.gauss_cfg, plan.field_settle_tolerance_mT,
                                   self._stop_event)

                ctx = allocate_run(plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                                   temperature_setpoint_K=plan.temperature_setpoint_K,
                                   key_axis=("current_A", I_mag), series=plan.series)
                self._run_contexts.append(ctx)
                self._set_run_label_threadsafe(f"Run #{ctx.run_str}")
                write_csv = make_incremental_writer(
                    ctx.raw_path,
                    lambda records, _ctx=ctx, _I=I_mag, _s=I_sense: build_header_fields(
                        plan, _ctx, records, status="in_progress", comment="",
                        extra={"magnet_current_A": _I, "sense_current_A": _s}))

                status = "Running the switching sweep …" if not label_parts \
                    else f"Running the switching sweep ({', '.join(label_parts)}) …"
                self._set_status_threadsafe(status)
                iter_error: Optional[BaseException] = None
                try:
                    run_measurement(
                        source, daq, plan.demod_cfg, plan.extref_cfg, plan.pulse_cfg,
                        plan.read_cfg, points,
                        stop_event=self._stop_event, on_point=self._make_on_point(series_idx, label),
                        gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                        temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                        magnet_current_A=I_mag,
                        field_theta_deg=plan.field_theta_deg, field_phi_deg=plan.field_phi_deg,
                        write_csv=write_csv, output_file=str(ctx.raw_path))
                except Exception as exc:
                    iter_error = exc

                iter_status = "error" if iter_error is not None \
                    else ("aborted" if self._stop_event.is_set() else "completed")
                iter_records = [r for r in self._records if r.get("series_index", 0) == series_idx]
                header_fields = build_header_fields(
                    plan, ctx, iter_records, status=iter_status, comment="",
                    extra={"magnet_current_A": I_mag, "sense_current_A": I_sense})
                write_record(ctx.raw_path, iter_records, header_fields)
                finalize_index_row(plan.data_root, ctx.sample, ctx.run_number, header_fields)
                self._save_run_png(ctx, iter_records)
                if iter_error is not None:
                    raise iter_error

            final = "Measurement aborted." if self._stop_event.is_set() else "Measurement complete."
        except Exception as exc:
            log.exception("Measurement failed")
            final = f"ERROR: {exc}"
        finally:
            if source is not None:
                safe_shutdown("6221", lambda: shutdown_ac_source(source))
            if magnet is not None:
                safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
            if gaussmeter is not None:
                safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
            if temp_ctrl is not None:
                safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))
            self.app.call_from_thread(self._on_finished, final)


# ── app / form ─────────────────────────────────────────────────────────────

class SOTPulsedSwitching6221App(App):
    TITLE = "SOT pulsed switching (6221-only)"
    SUB_TITLE = "6221 DC pulse · delayed 6221 AC / MFLI harmonic · static tilted field"

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
    .param-grid { layout: grid; grid-size: 2; grid-gutter: 1 2; height: auto; margin-bottom: 1; }
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
    .plane-btn-row { height: 3; margin-bottom: 1; }
    .plane-btn-row Button { min-width: 5; margin-right: 1; }
    .field-diagram { color: $text-muted; margin-top: 1; }
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
                        "Write pulse (6221 WAVE, hardware-timed)",
                        field("pulse_current_start_A", "Pulse current start (A)",
                              DEFAULTS["pulse_current_start_A"]),
                        field("pulse_current_stop_A", "Pulse current stop (A)",
                              DEFAULTS["pulse_current_stop_A"]),
                        field("pulse_current_step_A", "Pulse current step (A)",
                              DEFAULTS["pulse_current_step_A"],
                              validators=[Number(minimum=1e-12, failure_description="must be > 0")],
                              hint="One pulse per step."),
                        switch_field("amplitude_bidirectional",
                                     "Sweep up then back down (hysteresis loop)",
                                     DEFAULTS["amplitude_bidirectional"]),
                        field("pulse_width_s", "Requested pulse width (s)",
                              DEFAULTS["pulse_width_s"],
                              hint="No rise/fall control; actual width is measured and logged "
                                   "as pulse_width_measured_s. See the module docstring."),
                        field("pulse_compliance_V", "Pulse voltage compliance (V)",
                              DEFAULTS["pulse_compliance_V"]),
                    )
                    yield card(
                        "Delayed harmonic read (6221 AC + MFLI)",
                        field("delay_after_pulse_s", "Delay after pulse (s)",
                              DEFAULTS["delay_after_pulse_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")],
                              hint="Wait between pulse end and the read."),
                        field("sense_current_values", "6221 AC current amplitude, peak (A)",
                              DEFAULTS["sense_current_values"], kind="text",
                              hint="Keep well below the switching current. Single value, or "
                                   "comma-separated list — one complete pulse-current sweep "
                                   "runs per value (own 6221 re-arm), each saved to its own "
                                   "file."),
                        field("compliance_V", "6221 read compliance (V)", DEFAULTS["compliance_V"],
                              hint="Keep low — the Hall read needs < 1 V of headroom."),
                        field("frequency_Hz", "AC excitation frequency (Hz)",
                              DEFAULTS["frequency_Hz"],
                              hint="Avoid exact multiples of 50/60 Hz."),
                        field("phasemarker_line", "Trigger Link phase-marker pin (1-6)",
                              DEFAULTS["phasemarker_line"], kind="integer",
                              hint="Wire this pin to the MFLI's Aux In. Confirm it isn't the "
                                   "6221's factory-default Trigger Link pin before assuming "
                                   "it's free."),
                        field("harmonic", "Harmonic to lock in on", DEFAULTS["harmonic"],
                              kind="integer",
                              hint="2 = standard harmonic-Hall SOT signal (default). "
                                   "1 = resistive AHE/PHE."),
                        field("n_averages", "MFLI samples averaged per read",
                              DEFAULTS["n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("settle_after_enable_s", "Settle after PLL lock (s)",
                              DEFAULTS["settle_after_enable_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        field("lock_timeout_s", "PLL lock timeout (s)",
                              DEFAULTS["lock_timeout_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")],
                              hint="A timeout is logged, not fatal — the row is tagged "
                                   "reference_locked=False."),
                    )
                    yield card(
                        "Static field (Kepco magnet)",
                        field("magnet_current_A", "Assist current(s) (A)",
                              DEFAULTS["magnet_current_A"], kind="text",
                              hint="One value, or comma-separated for several — each gets its "
                                   "own complete sweep and file. Add the opposite sign for ±H_z."),
                        field("field_theta_deg", "θ — mount tilt from OOP (°)",
                              DEFAULTS["field_theta_deg"],
                              validators=[Number(0, 180, failure_description="0-180°")],
                              hint="0° = out-of-plane, 90° = in-plane. Recorded, not set."),
                        field("field_phi_deg", "φ — azimuth from current axis (°)",
                              DEFAULTS["field_phi_deg"], kind="number", valid_empty=True,
                              validators=[Number(0, 360, failure_description="0-360°")],
                              hint="Optional. Meaningless when θ=0°."),
                        Horizontal(
                            Button("xy", id="plane_xy", classes="plane-btn"),
                            Button("zx", id="plane_zx", classes="plane-btn"),
                            Button("zy", id="plane_zy", classes="plane-btn"),
                            classes="plane-btn-row",
                        ),
                        Static(render_ascii_field_diagram(
                                   float(DEFAULTS["field_theta_deg"]) if DEFAULTS["field_theta_deg"] else None,
                                   None),
                               id="field_diagram", classes="field-diagram"),
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
                            "Keithley 6221",
                            field("source_visa_resource", "6221 VISA resource",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            muted=True,
                        )
                        yield card(
                            "Zurich Instruments MFLI",
                            field("mfli_host", "LabOne data server host",
                                  DEFAULTS["mfli_host"], kind="text"),
                            field("mfli_port", "LabOne data server port",
                                  DEFAULTS["mfli_port"], kind="integer"),
                            field("mfli_device", "MFLI device ID", DEFAULTS["mfli_device"],
                                  kind="text", hint="e.g. dev1234."),
                            field("aux_input_ch", "Aux Input carrying the marker (0-based)",
                                  DEFAULTS["aux_input_ch"], kind="integer",
                                  hint="0 = Aux In 1."),
                            field("osc_index", "Oscillator locked by the PLL", DEFAULTS["osc_index"],
                                  kind="integer"),
                            field("extref_index", "ExtRef/PLL module index", DEFAULTS["extref_index"],
                                  kind="integer"),
                            field("pll_demod_index", "PLL phase-detector demod index (≠ demod below)",
                                  DEFAULTS["pll_demod_index"], kind="integer",
                                  hint="extrefs/N/adcselect is read-only on real firmware — the PLL "
                                       "is steered via THIS dedicated demod's own adcselect/oscselect "
                                       "instead. Must differ from the demod index below."),
                            select_field("automode", "PLL bandwidth adaptation",
                                         AUTOMODE_OPTIONS, int(DEFAULTS["automode"]),
                                         hint=AUTOMODE_HINT),
                            field("demod_index", "Demodulator index", DEFAULTS["demod_index"],
                                  kind="integer",
                                  hint="Default skips index 0 — that's the PLL phase-detector demod "
                                       "above. See the module docstring's 'Bench-verify' section."),
                            field("input_ch", "Signal Input channel (0-based)",
                                  DEFAULTS["input_ch"], kind="integer"),
                            switch_field("differential", "Differential input (IN+ / IN−)",
                                        DEFAULTS["differential"]),
                            switch_field("ac_coupling", "AC-couple the input", DEFAULTS["ac_coupling"]),
                            field("input_range_V", "Signal Input range (V)",
                                  DEFAULTS["input_range_V"]),
                            field("sample_rate_Hz", "Demodulator output rate (Sa/s)",
                                  DEFAULTS["sample_rate_Hz"]),
                            field("filter_time_constant_s", "Filter time constant (s)",
                                  DEFAULTS["filter_time_constant_s"]),
                            field("filter_order", "Filter order (1-8)", DEFAULTS["filter_order"],
                                  kind="integer"),
                            switch_field("filter_sinc", "Sinc filter (extra harmonic rejection)",
                                        DEFAULTS["filter_sinc"]),
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
                yield Static(SOT_PULSED_6221_DESCRIPTION, classes="card-desc")
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
        raw["automode"] = self.query_one("#automode", Select).value
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
        if "automode" in saved:
            try:
                self.query_one("#automode", Select).value = int(saved["automode"])
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
        state["automode"] = int(self.query_one("#automode", Select).value)
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""

        state["pulse_current_list"], state["pulse_current_parse_error"] = \
            _resolve_pulse_currents(state)
        state["magnet_currents_A"], state["magnet_currents_parse_error"] = \
            _resolve_magnet_currents(state)
        state["sense_currents_A"], state["sense_currents_parse_error"] = \
            _resolve_sense_currents(state)
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

        theta = None if parse_errors else state.get("field_theta_deg")
        phi = None if parse_errors else state.get("field_phi_deg")
        self.query_one("#field_diagram", Static).update(render_ascii_field_diagram(theta, phi))

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
        elif event.button.id == "plane_xy":
            self.query_one("#field_theta_deg", Input).value = "90"
            self.refresh_summary()
        elif event.button.id == "plane_zx":
            self.query_one("#field_phi_deg", Input).value = "0"
            self.refresh_summary()
        elif event.button.id == "plane_zy":
            self.query_one("#field_phi_deg", Input).value = "90"
            self.refresh_summary()

    def _build_plan(self, state: dict) -> MeasurementPlan:
        pulse_cfg = WritePulseConfig(
            width_s=state["pulse_width_s"], compliance_V=state["pulse_compliance_V"],
        )
        read_cfg = ReadConfig(
            sense_current_A=state["sense_currents_A"][0], compliance_V=state["compliance_V"],
            frequency_Hz=state["frequency_Hz"], phasemarker_line=state["phasemarker_line"],
            harmonic=state["harmonic"], n_averages=state["n_averages"],
            settle_after_enable_s=state["settle_after_enable_s"],
            lock_timeout_s=state["lock_timeout_s"],
            delay_after_pulse_s=state["delay_after_pulse_s"],
        )
        ac_cfg = ACSourceConfig(
            visa_resource=state["source_visa_resource"], amplitude_A=state["sense_currents_A"][0],
            frequency_Hz=state["frequency_Hz"], compliance_V=state["compliance_V"],
            phasemarker_line=state["phasemarker_line"],
        )
        extref_cfg = ExtRefConfig(
            device=state["mfli_device"], extref_index=state["extref_index"],
            aux_input_ch=state["aux_input_ch"], osc_index=state["osc_index"],
            pll_demod_index=state["pll_demod_index"], automode=state["automode"],
        )
        shared_filter = FilterConfig(
            time_constant_s=state["filter_time_constant_s"], order=state["filter_order"],
            sinc_filter=state["filter_sinc"],
        )
        demod_cfg = DemodConfig(
            device=state["mfli_device"], demod_index=state["demod_index"],
            harmonic=state["harmonic"], osc_index=state["osc_index"],
            input_ch=state["input_ch"], differential=state["differential"],
            ac_coupling=state["ac_coupling"], input_range_V=state["input_range_V"],
            sample_rate_Hz=state["sample_rate_Hz"], filter=shared_filter,
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

        header_extra = {
            "pulse_width_s": state["pulse_width_s"],
            "pulse_compliance_V": state["pulse_compliance_V"],
            "delay_after_pulse_s": state["delay_after_pulse_s"],
            "sense_current_A": state["sense_currents_A"][0],
            "frequency_Hz": state["frequency_Hz"],
            "phasemarker_line": state["phasemarker_line"],
            "harmonic": state["harmonic"],
            "field_theta_deg": state["field_theta_deg"],
            "field_phi_deg": state["field_phi_deg"],
            "pulse_current_start_A": state["pulse_current_start_A"],
            "pulse_current_stop_A": state["pulse_current_stop_A"],
            "pulse_current_step_A": state["pulse_current_step_A"],
            "amplitude_bidirectional": state["amplitude_bidirectional"],
            "pulse_currents_A": state["pulse_current_list"],
        }
        return MeasurementPlan(
            ac_cfg=ac_cfg, pulse_cfg=pulse_cfg, extref_cfg=extref_cfg, demod_cfg=demod_cfg,
            mfli_host=state["mfli_host"], mfli_port=state["mfli_port"],
            read_cfg=read_cfg, magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg,
            pulse_currents_A=state["pulse_current_list"], magnet_currents_A=state["magnet_currents_A"],
            sense_currents_A=state["sense_currents_A"],
            field_theta_deg=state["field_theta_deg"], field_phi_deg=state["field_phi_deg"],
            field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
            data_root=self.data_root,
            sample=state["sample"], device=state["device"],
            temperature_setpoint_K=state["temperature_setpoint_K"],
            cooldown=state["cooldown"], header_extra=header_extra, series="",
            temp_cfg=temp_cfg,
        )


def main() -> None:
    SOTPulsedSwitching6221App().run()


if __name__ == "__main__":
    main()
