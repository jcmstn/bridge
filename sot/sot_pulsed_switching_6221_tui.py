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
import logging
import multiprocessing as mp
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.validation import Number
from textual.widgets import (
    Button, Collapsible, Footer, Header, Static,
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
from instruments.data_dir import validate_directory
from instruments.field_geometry import field_direction_summary_line, render_ascii_field_diagram
from instruments.data_naming import (
    RunContext,
    allocate_run,
    record_run,
    preview_raw_filename,
)
from instruments.keithley6221 import ac_source_restart_s, wave_pulse_s
from instruments.kepco_magnet import magnet_move_s
from instruments.lakeshore475 import read_field_s
from instruments.mfli_daq import acquire_s
from instruments.run_time import (
    ARM_S, GPIB_TXN_S, LOCK_TYP_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
    RunCost,
)
from instruments.tui_common import (
    MeasurementApp,
    MeasurementRunScreen,
    card,
    field,
    format_si,
    identity_bar,
    parse_sensor_uids,
    run_screen_bindings,
    select_field,
    switch_field,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
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

# Wiring schematic — shown on this program's card in bridge_tui.py.
SOT_PULSED_6221_SCHEMATIC = """\
  KEITHLEY 6221  (the ONLY source — no 4200A, no 2182)   HI ──▶ I+ pad ,  LO ──▶ I- pad
    WAVE square, ONE cycle = the write pulse (hardware-timed, per amplitude)
    then WAVE sine + phase marker = the read; wave OFF between the two.
    Trigger Link phase marker (pin 1) ──▶ ZURICH MFLI  AUX IN 1
  ZURICH MFLI    Signal Input (differential) ──▶ the transverse (Hall) arms
                 ExtRef-locked to the marker; one harmonic (2f default, 1f = AHE/PHE)

  KEPCO BOP-GL      ──GPIB──▶ electromagnet   (ONE static tilted field)
  LAKE SHORE 475    ──GPIB──▶ Gaussmeter probe at the sample

  Software-timed at the µs–ms scale: Joule heating is I²R·t, far above the
  4200A's ns pulses — start well below the switching current.
"""

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


def run_costs(state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (one complete pulse-current sweep per file: sense current outer x assist
    current inner, like MeasurementPlan.series_values). Also drives the run
    screen's progress bar, so estimate and live ETA cannot disagree."""
    pulses = state.get("pulse_current_list", [])
    series = list(itertools.product(state.get("sense_currents_A", []),
                                    state.get("magnet_currents_A", [])))
    rc = RunCost(len(pulses) * max(1, len(series)))
    magnet = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    gauss = GaussmeterConfig(n_averages=state["gaussmeter_n_averages"],
                             read_delay_s=state["gaussmeter_read_delay_s"])
    has_temp = state["enable_temperature"] and bool(parse_sensor_uids(state["temperature_sensor_uids"]))
    lock_typ = min(LOCK_TYP_S, state["lock_timeout_s"])   # the lock wait returns at the first lock
    # run_measurement(): AC off (2 writes) -> fire_wave_pulse (its own ARM) -> wait -> AC on
    # (compliance, enable, ARM, START = 4 writes + a second ARM) -> PLL lock -> settle -> one
    # acquire -> AC off (2) -> frequency read-back (1) -> temperature -> CSV rewrite
    rc.each("post-pulse wait", state["delay_after_pulse_s"])
    rc.each("6221 pulse", wave_pulse_s(state["pulse_width_s"]))
    rc.each("6221 re-arm", ARM_S)
    rc.each("PLL lock", lock_typ, worst_extra=max(0.0, state["lock_timeout_s"] - lock_typ))
    rc.each("settle", state["settle_after_enable_s"])
    rc.each("MFLI read", acquire_s(state["filter_time_constant_s"], state["n_averages"],
                                   max(state["sample_rate_Hz"], 1.0)))
    rc.each("overhead", 9 * GPIB_TXN_S + POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    prev = 0.0                                       # magnet starts at 0 A
    for k, (_I_sense, I_mag) in enumerate(series):
        first = k * len(pulses)                        # this file's first point
        rc.at("6221 rebuild", ac_source_restart_s(), first)    # do_run(): shutdown + connect_ac_source every file
        typ, worst = magnet_move_s(abs(I_mag - prev), magnet)  # set_magnet_current() every file, no parking guard
        rc.at("magnet", typ, first, worst_extra=worst - typ)
        prev = I_mag
        rc.at("field read", read_field_s(gauss), first)        # run_measurement() reads the field once
        rc.at("per-file", PER_FILE_S, first)
    rc.at("per-run", PER_RUN_S, 0)
    if series:                                       # shutdown_magnet() ramps back to 0 A
        rc.tail("ramps", magnet_move_s(abs(series[-1][1]), magnet, with_field=False)[0])
    return rc


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
    run_cost: Optional[RunCost] = None      # modelled seconds per point (progress bar + ETA)

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


# ── summary ────────────────────────────────────────────────────────────────

def _near_multiple(f: float, m: float) -> bool:
    return min(f % m, m - f % m) < 1.0


def resolve_state(state: dict) -> dict:
    """Add the derived keys build_summary() / build_plan() read — the parsed
    lists/sweeps, each with its parse error — to a state of raw field values.
    Pure: shared by the TUI's and the web page's parse_state()."""
    state["pulse_current_list"], state["pulse_current_parse_error"] = _resolve_pulse_currents(state)
    state["magnet_currents_A"], state["magnet_currents_parse_error"] = _resolve_magnet_currents(state)
    state["sense_currents_A"], state["sense_currents_parse_error"] = _resolve_sense_currents(state)
    return state


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
    info.append(f"{n} amplitudes, one pulse each"
                + (f", × {n_files} files ({n_currents} assist current(s) x {n_sense} sense "
                   f"current(s)) = {n * n_files} total points"
                   if n_files > 1 else ""))
    info.extend(run_costs(state).lines("Estimated run time"))
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


# ── plan + run (pure, shared by the TUI RunScreen and (no web page yet)) ──────────────

def build_plan(state: dict, data_root: Path) -> MeasurementPlan:
    """One parsed, validated run request from a state dict. Pure — shared by
    the TUI and the web page."""
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
        data_root=data_root,
        sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series="",
        temp_cfg=temp_cfg, run_cost=run_costs(state),
    )

def _ignore(*_args) -> None:
    pass


def run_plan(plan: MeasurementPlan, stop_event: threading.Event, *,
             on_status: Callable[[str], None] = _ignore,
             on_run_label: Callable[[str], None] = _ignore,
             on_point: Callable[[dict], None] = _ignore,
             on_run_finished: Optional[Callable[[RunContext, list], None]] = None,
             run_contexts: Optional[list] = None,
             run_extras: Optional[list] = None) -> None:
    """Check the pulse/read limits, connect the MFLI (ExtRef-locked to the 6221
    marker), magnet and gaussmeter, then per (read current, assist-field current)
    pair: re-arm the 6221, park the field, and record one pulse-current sweep
    (own run number, own file) through record_run(); always shut everything
    down. Pure — the TUI's RunScreen runs it with its own callbacks."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = daq = magnet = gaussmeter = temp_ctrl = None
    try:
        points = [PulsePoint(pulse_current_A=float(v)) for v in plan.pulse_currents_A]
        _check_write_safety(plan.pulse_cfg)
        _check_pulse_currents(points)
        _check_extref_demod_conflict(plan.demod_cfg, plan.extref_cfg)

        on_status("Connecting to MFLI …")
        daq = connect(plan.mfli_host, plan.mfli_port)
        connect_device(daq, plan.extref_cfg.device, interface="1GbE")
        configure_external_reference(daq, plan.extref_cfg, plan.ac_cfg.frequency_Hz)
        configure_demodulator(daq, plan.demod_cfg)

        on_status("Connecting to Kepco magnet + Lake Shore 475 …")
        magnet = connect_magnet(plan.magnet_cfg)
        gaussmeter = connect_gaussmeter(plan.gauss_cfg)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        multi_sense = len(plan.sense_currents_A) > 1
        for series_idx, (I_sense, I_mag) in enumerate(plan.series_values):
            if stop_event.is_set():
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
            # catch a mistyped exponent before that, not after. Amplitude needs a
            # full re-arm -- tear down the previous amplitude's source first.
            _check_read_safety(plan.read_cfg)
            if source is not None:
                safe_shutdown("6221 AC source", lambda _s=source: shutdown_ac_source(_s))
                source = None
            on_status(f"Starting 6221 AC current source{f' ({I_sense:g} A)' if multi_sense else ''} …")
            source = connect_ac_source(plan.ac_cfg)
            _six221_ac_output_off(source)          # channel quiet before any pulse

            on_status(f"Ramping magnet to {I_mag:g} A …")
            set_magnet_current(magnet, plan.magnet_cfg, I_mag,
                               gaussmeter, plan.gauss_cfg, plan.field_settle_tolerance_mT,
                               stop_event)

            ctx = allocate_run(plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                               temperature_setpoint_K=plan.temperature_setpoint_K,
                               key_axis=("current_A", I_mag), series=plan.series)
            extra = {"magnet_current_A": I_mag, "sense_current_A": I_sense}
            run_contexts.append(ctx)
            run_extras.append(extra)
            on_run_label(f"Run #{ctx.run_str}")

            on_status("Running the switching sweep …" if not label_parts
                      else f"Running the switching sweep ({', '.join(label_parts)}) …")
            record_run(
                plan.data_root, ctx,
                lambda records, status, _ctx=ctx, _x=extra: build_header_fields(
                    plan, _ctx, records, status=status, comment="", extra=_x),
                lambda point_cb, write_csv, _ctx=ctx, _I=I_mag, _src=source: run_measurement(
                    _src, daq, plan.demod_cfg, plan.extref_cfg, plan.pulse_cfg,
                    plan.read_cfg, points,
                    stop_event=stop_event, on_point=point_cb,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg, magnet_current_A=_I,
                    field_theta_deg=plan.field_theta_deg, field_phi_deg=plan.field_phi_deg,
                    write_csv=write_csv, output_file=str(_ctx.raw_path)),
                stop_event, on_point=on_point,
                tags={"series_index": series_idx, "series_label": label},
                on_finished=on_run_finished)
    finally:
        if source is not None:
            safe_shutdown("6221", lambda: shutdown_ac_source(source))
        if magnet is not None:
            safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
        if gaussmeter is not None:
            safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


PNG_SUFFIX = "Vnf_vs_pulse"


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (the RunScreen calls this)."""
    _save_measurement_png(records, png_path, plan.read_cfg.harmonic, plan=plan, comment=comment)


# ── run screen ─────────────────────────────────────────────────────────────

class RunScreen(MeasurementRunScreen):
    ABORT_LABEL = "Abort (safe shutdown)"
    BINDINGS = run_screen_bindings(ABORT_LABEL)
    ABORT_STATUS = "Abort requested — finishing this amplitude, then shutting the 6221 + magnet down …"
    POINT_STATUS = "Point {n} / {total}."
    MEASUREMENT_TYPE = MEASUREMENT_TYPE
    PNG_SUFFIX = PNG_SUFFIX

    def table_columns(self) -> tuple:
        h = self.plan.read_cfg.harmonic
        return ("amp #", "I_mag (A)", "I_pulse (A)", "width meas (s)", f"V_{h}f (V)", "locked", "T1 (K)")

    def live_plot_args(self):
        return (_live_plot_worker, self.plan.read_cfg.harmonic)

    def table_row(self, record: dict) -> tuple:
        t1 = record.get("temperature_1_K")
        return (
            str(record["amplitude_index"] + 1),
            f"{record['magnet_current_A']:g}" if record.get("magnet_current_A") is not None else "—",
            f"{record['pulse_current_A']:.4g}",
            f"{record['pulse_width_measured_s']:.4g}",
            f"{record['demod_R_V']:.4e}",
            "yes" if record.get("reference_locked") else "no",
            f"{t1:.3f}" if t1 is not None else "—",
        )


# ── app / form ─────────────────────────────────────────────────────────────

class SOTPulsedSwitching6221App(MeasurementApp):
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                yield identity_bar(DEFAULTS, _DEFAULT_DATA_DIR, self.data_root,
                                   device_label="Device (e.g. HB3)",
                                   temperature_hint="Filename T###K token only.")

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

    # form I/O

    def update_summary(self) -> None:
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

    def _build_plan(self, state: dict) -> MeasurementPlan:
        return build_plan(state, self.data_root)


def main() -> None:
    SOTPulsedSwitching6221App().run()


if __name__ == "__main__":
    main()
