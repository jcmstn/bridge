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

import itertools
import json
import logging
import multiprocessing as mp
import textwrap
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.validation import Number
from textual.widgets import (
    Button, Collapsible, Footer, Header, Select, Static,
)

import sot.sot_pulsed_switching_2h_tui as h2
import sot.sot_pulsed_switching_6221_tui as i1
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
from dc.dc_sweep_utils import linear_sweep, parse_value_list, safe_shutdown
from instruments.data_dir import validate_directory
from instruments.field_geometry import field_direction_summary_line, render_ascii_field_diagram
from instruments.data_naming import (
    RunContext,
    allocate_run,
    record_run,
    preview_raw_filename,
)
from instruments.keithley2182 import read_time_s
from instruments.keithley4200a import pulse_once_s
from instruments.keithley6221 import reversal_avg_s
from instruments.kepco_magnet import magnet_move_s
from instruments.lakeshore475 import read_field_s
from instruments.run_time import (
    GPIB_TXN_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
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
    select_field,
    switch_field,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
)

log = logging.getLogger("sot_pulsed_switching_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "sot_pulsed_switching_tui_settings.json"

MEASUREMENT_TYPE = "SOTPS"

SOT_PULSED_DESCRIPTION = (
    "SOT switching curve: ONE write pulse per amplitude into the main channel, then "
    "after a fixed delay a small read of the Hall arms — at a static field held "
    "slightly out of plane so the two in-plane remanent states read differently. "
    "Write pulse: the 4200A PMU (ns, through RPM1; KULT module "
    "instruments/kult/bridge_sot_pulse.c) or the 6221 WAVE (hardware-timed µs–ms). "
    "Read: DC R_xy (6221 ±I, 2182, reversal-averaged) or the lock-in harmonic "
    "(6221 AC + MFLI ExtRef-locked to its phase marker; 1f+2f with the PMU, one "
    "chosen harmonic with the 6221 pulse). Make the amplitude list a full loop "
    "(up then down) for the hysteresis; each assist-field / read current gets its "
    "own complete sweep and file. Re-run for switching-probability statistics."
)

# Wiring schematic — shown on this program's card in bridge_tui.py.
SOT_PULSED_SCHEMATIC = """\
  Write pulse = 4200A PMU   (SOTPS / SOT2H)
    PMU1-1 ──▶ RPM1 ──▶ I+ pad of the Hall-cross main channel (FORCE triax,
    centre = force, guard floating, outer = circuit COMMON). KULT module
    bridge_sot_pulse.c routes RPM1 to the pulse path for the burst.
    COMMON BUS ──▶ I- pad   (PMU outer shell + 6221 output LO land here)
  Write pulse = 6221 WAVE   (SOT1I — no 4200A)
    6221 HI ──▶ I+ pad , LO ──▶ I- pad ; ONE square cycle = the write pulse

  KEITHLEY 6221  HI ──▶ I+ pad , LO ──▶ common bus / I- pad — OFF while pulsing
  Read = DC R_xy     KEITHLEY 2182 ──▶ transverse (Hall) arms (V_xy, floating)
  Read = lock-in     6221 WAVE sine + Trigger Link phase marker (pin 1)
                       ──▶ ZURICH MFLI AUX IN 1 (ExtRef)
                     MFLI Signal Input (differential) ──▶ transverse (Hall) arms

  KEPCO BOP-GL      ──GPIB──▶ electromagnet   (ONE static tilted field)
  LAKE SHORE 475    ──GPIB──▶ Gaussmeter probe at the sample
"""

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
    "sense_current_values": "1e-4",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "nplc": "5",
    "auto_range": True,
    "n_reversals": "5",
    "settle_after_enable_s": "0.3",
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
    "compliance_V": float,
    "source_delay_s": float,
    "nplc": float,
    "n_reversals": int,
    "settle_after_enable_s": float,
    "delay_after_pulse_s": float,
    "field_theta_deg": float,
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
               "temperature_sensor_uids", "magnet_current_A", "sense_current_values", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K", "field_phi_deg"]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]

# Every Switch id on the form. Hardcoded in collect_raw / _load_settings /
# parse_state -- they must move together, and parse_state runs on every
# keystroke, so a stale entry here is an immediate crash.
SWITCH_FIELD_IDS = ("auto_range", "enable_temperature", "amplitude_bidirectional")


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


def _resolve_magnet_currents(state: dict) -> tuple[list[float], Optional[str]]:
    """(list, None) or ([], error) — one or more static assist-field currents
    (comma-separated); each gets its own complete amplitude sweep, its own
    file. A single value behaves exactly as before."""
    try:
        return parse_value_list(state["magnet_current_A"]), None
    except ValueError as exc:
        return [], str(exc)


def _resolve_sense_currents(state: dict) -> tuple[list[float], Optional[str]]:
    """(list, None) or ([], error) — one or more 6221 read currents
    (comma-separated); nests with the assist-current list (magnet outer,
    sense inner — magnet ramps, sense is an instant config mutation), each
    pair its own complete amplitude sweep, its own file."""
    try:
        return parse_value_list(state["sense_current_values"]), None
    except ValueError as exc:
        return [], str(exc)


# ── formatting helpers (per-TUI copies) ─────────────────────────────────────


def run_costs(state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (one complete amplitude sweep per file: assist current outer x sense
    current inner, like MeasurementPlan.series_values). Also drives the run
    screen's progress bar, so estimate and live ETA cannot disagree."""
    amps = state.get("amplitude_list", [])
    series = list(itertools.product(state.get("magnet_currents_A", []),
                                    state.get("sense_currents_A", [])))
    rc = RunCost(len(amps) * max(1, len(series)))
    magnet = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    gauss = GaussmeterConfig(n_averages=state["gaussmeter_n_averages"],
                             read_delay_s=state["gaussmeter_read_delay_s"])
    has_temp = state["enable_temperature"] and bool(parse_sensor_uids(state["temperature_sensor_uids"]))
    # run_measurement(): 6221 off (2 writes) -> PMU pulse -> wait -> 6221 on (1 write) + settle
    # -> reversal read (delay and read ADD) -> 6221 off (2 writes) -> temperature -> CSV rewrite
    rc.each("post-pulse wait", state["delay_after_pulse_s"])
    rc.each("settle", state["settle_after_enable_s"])
    rc.each("PMU pulse", pulse_once_s(state["n_pulses"], state["pulse_period_s"]))
    rc.each("reads", reversal_avg_s(state["n_reversals"], state["source_delay_s"],
                                    read_time_s(state["nplc"])))
    rc.each("overhead", 5 * GPIB_TXN_S + POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    parked = None                                    # do_run()'s _parked_magnet guard
    for k, (I_mag, _I_sense) in enumerate(series):
        first = k * len(amps)                        # this file's first point
        if I_mag != parked:
            typ, worst = magnet_move_s(abs(I_mag - (parked or 0.0)), magnet)   # magnet starts at 0 A
            rc.at("magnet", typ, first, worst_extra=worst - typ)
            parked = I_mag
        rc.at("field read", read_field_s(gauss), first)      # run_measurement() reads the field once
        rc.at("per-file", PER_FILE_S, first)
    rc.at("per-run", PER_RUN_S, 0)
    if series:                                       # shutdown_magnet() ramps back to 0 A
        rc.tail("ramps", magnet_move_s(abs(series[-1][0]), magnet, with_field=False)[0])
    return rc


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
        """Cross product of assist-field currents x sense currents -- one
        complete amplitude sweep per pair, each saved to its own file.
        Magnet is outer (a physical ramp), sense is inner (an instant
        config mutation) -- see dc_spin_valve_tui.py for the same
        nested-product pattern."""
        return list(itertools.product(self.magnet_currents_A, self.sense_currents_A))

    @property
    def total_points(self) -> int:
        return len(self.amplitudes_V) * max(1, len(self.series_values))


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

def _resolve_state_ps(state: dict) -> dict:
    """Add the derived keys build_summary() / build_plan() read — the parsed
    lists/sweeps, each with its parse error — to a state of raw field values.
    Pure: shared by the TUI's and the web page's parse_state()."""
    state["amplitude_list"], state["amplitude_parse_error"] = _resolve_amplitudes(state)
    state["magnet_currents_A"], state["magnet_currents_parse_error"] = _resolve_magnet_currents(state)
    state["sense_currents_A"], state["sense_currents_parse_error"] = _resolve_sense_currents(state)
    return state


def _build_summary_ps(state: dict) -> tuple[list[str], list[str], list[str]]:
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
    sense_currents = state.get("sense_currents_A", [])
    if state.get("sense_currents_parse_error"):
        errors.append(f"6221 sense current(s): {state['sense_currents_parse_error']}")
    else:
        zero = [i for i in sense_currents if i <= 0]
        over = [i for i in sense_currents if i > _READ_CURRENT_CEILING_A]
        large = [i for i in sense_currents if 1e-3 < i <= _READ_CURRENT_CEILING_A]
        if zero:
            errors.append("6221 sense current must be > 0 A.")
        elif over:
            errors.append(f"6221 sense current(s) {over} exceed the "
                          f"{format_si(_READ_CURRENT_CEILING_A, 'A')} safety ceiling — the Hall read "
                          "needs µA–mA; check for a mistyped exponent.")
        elif large:
            warnings.append(f"6221 sense current(s) {large} are large "
                            "for a read — they flow continuously through the channel; keep them "
                            "well below the switching current.")
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
    info.append(f"PMU module: {state['pmu_library']}/{state['pmu_module'] or '<unset>'} "
                f"({state['pmu_id']} ch {state['pmu_channel']})")

    # Display-only current estimate off the load-line DUT resistance. It is a hint
    # for picking amplitudes; the honest pulse axis is the module's measured
    # pulse_current_measured_A, and pmu_dut_res_ohm is never written as data.
    r_ch = state.get("pmu_dut_res_ohm", 0.0)
    if r_ch > 0 and amps and sense_currents:
        i_lo, i_hi = min(amps) / r_ch, max(amps) / r_ch
        i_sense0 = sense_currents[0]
        info.append(f"At DUT R ≈ {r_ch:g} Ω: pulses ≈ "
                    f"{format_si(i_lo, 'A')}…{format_si(i_hi, 'A')}; 6221 read current "
                    f"{format_si(i_sense0, 'A')} → "
                    f"≈ {format_si(i_sense0 * r_ch, 'V')} across the channel")
        if (state["pmu_v_range_V"] == 10.0
                and max(abs(i_lo), abs(i_hi)) > _RPM_10V_IMEAS_MAX_A
                and state["pmu_i_range_A"] <= _RPM_10V_IMEAS_MAX_A):
            warnings.append(
                f"Estimated pulse current exceeds the RPM's "
                f"{format_si(_RPM_10V_IMEAS_MAX_A, 'A')} measure ceiling on the 10 V range — "
                "pulse_current_measured_A will read overflowed, not error. The pulse itself "
                "still fires.")

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


def _preview_ps(state: dict) -> Optional[str]:
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL or not state.get("device"):
        return None
    preview = preview_raw_filename(
        state["sample"], state["device"], MEASUREMENT_TYPE,
        temperature_setpoint_K=state.get("temperature_setpoint_K"))
    n_files = max(1, len(state.get("magnet_currents_A", []))) * max(1, len(state.get("sense_currents_A", [])))
    suffix = " (one file per run)" if n_files > 1 else ""
    return f"{preview}_<I_mag A>_<timestamp>.csv{suffix}"


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

    # One line per assist-field current (series_index/series_label, set only
    # when more than one magnet_current_A is in play — see _make_on_point),
    # each in acquisition order so the connecting line shows the sweep
    # direction (up-leg then down-leg for a bidirectional amplitude list).
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
            xs.append(rec["pulse_amplitude_V"])
            ys.append(rec["hall_resistance_ohm"])
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


def _save_measurement_png(records: list[dict], png_path: Path,
                           plan: Optional["MeasurementPlan"] = None, comment: str = "") -> None:
    """`records` is ONE run's points -- with several currents each run is
    saved (and plotted) on its own, exactly like a manual run.

    `plan`/`comment` add a small "at a glance" text annotation (the
    static assist-field direction, the fixed 6221 read current, the
    operator's comment) for context not already in the filename -- the
    swept assist-field magnitude is already this run's key_axis. Called
    once when the run ends (comment="") and again, to overwrite the PNG
    in place, once the operator's comment is known."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    # In acquisition order so the line traces the sweep direction.
    ax.plot([r["pulse_amplitude_V"] for r in records],
            [r["hall_resistance_ohm"] for r in records],
            "o-", ms=4, lw=1, alpha=0.6, color="tab:blue")
    ax.set_xlabel("Pulse amplitude (V)")
    ax.set_ylabel("R_xy (Ω)")
    ax.set_title("R_xy vs pulse amplitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lines: list[str] = []
    if plan is not None:
        if plan.field_theta_deg is not None:
            lines.append(field_direction_summary_line(plan.field_theta_deg, plan.field_phi_deg))
        sense_currents = sorted({r["sense_current_A"] for r in records
                                  if r.get("sense_current_A") is not None})
        if len(sense_currents) == 1:
            lines.append(f"6221 read current: {format_si(sense_currents[0], 'A')}")
    if comment:
        lines.append(f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}")
    if lines:
        fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color="0.4", va="bottom")
        fig.subplots_adjust(bottom=0.08 + 0.045 * len(lines))

    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot to '%s'", png_path)


# ── plan + run (pure, shared by the TUI RunScreen and (no web page yet)) ──────────────

def _build_plan_ps(state: dict, data_root: Path) -> MeasurementPlan:
    """One parsed, validated run request from a state dict. Pure — shared by
    the TUI and the web page."""
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
        sense_current_A=state["sense_currents_A"][0], compliance_V=state["compliance_V"],
        source_delay_s=state["source_delay_s"], nplc=state["nplc"],
        auto_range=state["auto_range"], n_reversals=state["n_reversals"],
        settle_after_enable_s=state["settle_after_enable_s"],
        delay_after_pulse_s=state["delay_after_pulse_s"],
    )
    src_cfg = SourceConfig(
        visa_resource=state["source_visa_resource"], sense_current_A=state["sense_currents_A"][0],
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
        "sense_current_A": state["sense_currents_A"][0],
        "n_reversals": state["n_reversals"],
        "field_theta_deg": state["field_theta_deg"],
        "field_phi_deg": state["field_phi_deg"],
        "amplitude_start_V": state["amplitude_start_V"],
        "amplitude_stop_V": state["amplitude_stop_V"],
        "amplitude_step_V": state["amplitude_step_V"],
        "amplitude_bidirectional": state["amplitude_bidirectional"],
        "amplitudes_V": state["amplitude_list"],
    }
    return MeasurementPlan(
        k4200_cfg=k4200_cfg, pmu_cfg=pmu_cfg, src_cfg=src_cfg, volt_cfg=volt_cfg,
        read_cfg=read_cfg, magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg,
        amplitudes_V=state["amplitude_list"], magnet_currents_A=state["magnet_currents_A"],
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
    """Connect the 4200A PMU, 6221 + 2182, magnet and gaussmeter, then one
    amplitude sweep (own run number, own file) per (assist-field current, read
    current) pair, each recorded + finalized by record_run() before the next;
    always shut everything down (6221 first — it shares the channel pin).
    Pure — the TUI's RunScreen runs it with its own callbacks."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    k4200 = source = voltmeter = magnet = gaussmeter = temp_ctrl = None
    try:
        on_status("Connecting to Keithley 4200A (KXCI) …")
        k4200 = connect_4200a(plan.k4200_cfg)
        try:
            log.info("Installed user libraries (UL):\n%s", list_user_libraries(k4200))
        except Exception:
            log.warning("Could not read `UL` — set the PMU module name from the 4200A manually.")
        configure_pmu_pulse(k4200, plan.pmu_cfg)

        # connect_source() returns with the 6221 already sourcing — re-check
        # the read limits here too, not only in build_summary.
        _check_read_safety(plan.read_cfg)
        on_status("Connecting to Keithley 6221 + 2182 …")
        source = connect_source(plan.src_cfg)
        _six221_output_off(source)          # channel quiet before any pulse
        voltmeter = connect_voltmeter(plan.volt_cfg)

        on_status("Connecting to Kepco magnet + Lake Shore 475 …")
        magnet = connect_magnet(plan.magnet_cfg)
        gaussmeter = connect_gaussmeter(plan.gauss_cfg)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        points = [AmplitudePoint(amplitude_V=float(v)) for v in plan.amplitudes_V]

        _unset = object()
        parked_magnet = _unset
        for series_idx, (I_mag, I_sense) in enumerate(plan.series_values):
            if stop_event.is_set():
                break
            plan.src_cfg.sense_current_A = I_sense
            plan.read_cfg.sense_current_A = I_sense

            label_parts = []
            if len(plan.magnet_currents_A) > 1:
                label_parts.append(f"I_mag={I_mag:g}A")
            if len(plan.sense_currents_A) > 1:
                label_parts.append(f"I_sense={I_sense:g}A")
            label = ", ".join(label_parts) or None

            if I_mag != parked_magnet:
                on_status(f"Ramping magnet to {I_mag:g} A …")
                set_magnet_current(magnet, plan.magnet_cfg, I_mag,
                                   gaussmeter, plan.gauss_cfg, plan.field_settle_tolerance_mT,
                                   stop_event)
                parked_magnet = I_mag

            # A fresh RunContext (own run number, own file) EVERY iteration.
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
                lambda point_cb, write_csv, _ctx=ctx, _I=I_mag: run_measurement(
                    k4200, plan.pmu_cfg, source, voltmeter, plan.read_cfg, points,
                    stop_event=stop_event, on_point=point_cb,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg, magnet_current_A=_I,
                    field_theta_deg=plan.field_theta_deg, field_phi_deg=plan.field_phi_deg,
                    write_csv=write_csv, output_file=str(_ctx.raw_path)),
                stop_event, on_point=on_point,
                tags={"series_index": series_idx, "series_label": label},
                on_finished=on_run_finished)
    finally:
        # 6221 down first (it shares the channel pin), then the 4200A, then the
        # magnet — never ramp an inductive field while the DUT still carries current.
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


PNG_SUFFIX = "Rxy_vs_amp"


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (the RunScreen calls this)."""
    _save_measurement_png(records, png_path, plan=plan, comment=comment)


# ── pulse-source × read toggles (one program, three engines) ──────────────
# One form for the three pulsed-switching programs. "Write pulse" picks the
# 4200A PMU (ns pulses) or the 6221 WAVE (µs–ms pulses); "Read" picks the DC
# R_xy read (6221 ±I + 2182) or the lock-in harmonic read (6221 AC + MFLI).
# Each valid combination IS one of the former programs — its engine, type
# code, columns and header, unchanged:
#     4200A + DC        -> SOTPS  (this module's functions above)
#     4200A + harmonic  -> SOT2H  (sot_pulsed_switching_2h_tui)
#     6221  + harmonic  -> SOT1I  (sot_pulsed_switching_6221_tui)
# 6221 + DC is not a program here (a 6221 pulse + 2182A read is the nonlocal-
# switching program's). Two form fields mean different things per engine and
# get their own ids, renamed back to the engine's key before it sees the state:
# the 6221 pulse's width (ms, `wave_pulse_width_s`) vs the PMU's (ns,
# `pulse_width_s`), and the lock-in read's settle-after-PLL-lock
# (`lock_settle_s`) vs the DC read's settle-after-enable.

PULSE_SOURCES = [("Keithley 4200A PMU (ns pulses)", "pmu"), ("Keithley 6221 WAVE (µs–ms pulses)", "6221")]
READ_MODES = [("DC R_xy — 6221 ±I + 2182", "dc"), ("Lock-in harmonic — 6221 AC + MFLI", "harmonic")]
_PS = sys.modules[__name__]
_ENGINES = {("pmu", "dc"): _PS, ("pmu", "harmonic"): h2, ("6221", "harmonic"): i1}
_FORM_IDS = {i1: {"pulse_width_s": "wave_pulse_width_s", "settle_after_enable_s": "lock_settle_s"},
             h2: {"settle_after_enable_s": "lock_settle_s"}}    # engine key -> form id


def _form_keys(engine, keys) -> list:
    return [_FORM_IDS.get(engine, {}).get(k, k) for k in keys]


def _as_form(engine, d: dict) -> dict:
    return dict(zip(_form_keys(engine, d), d.values()))


_ENGINE_FIELDS = {eng: set(_form_keys(eng, eng.DEFAULTS)) for eng in (h2, i1)} | {_PS: set(DEFAULTS)}
DEFAULTS = {**_as_form(i1, i1.DEFAULTS), **_as_form(h2, h2.DEFAULTS), **DEFAULTS,
            "pulse_source": "pmu", "read_mode": "dc"}
NUMERIC_FIELDS = {**_as_form(i1, i1.NUMERIC_FIELDS), **_as_form(h2, h2.NUMERIC_FIELDS), **NUMERIC_FIELDS}
TEXT_FIELDS = list(dict.fromkeys(TEXT_FIELDS + h2.TEXT_FIELDS + i1.TEXT_FIELDS))
OPTIONAL_NUMERIC_FIELDS = list(dict.fromkeys(
    OPTIONAL_NUMERIC_FIELDS + h2.OPTIONAL_NUMERIC_FIELDS + i1.OPTIONAL_NUMERIC_FIELDS))


def mode(state: dict) -> tuple[str, str]:
    return state.get("pulse_source", "pmu"), state.get("read_mode", "dc")


def _engine_state(state: dict):
    """(engine, the state with that engine's own keys), or (None, state) for
    a combination that isn't a program."""
    engine = _ENGINES.get(mode(state))
    if engine is None:
        return None, state
    state = dict(state)
    for key, form_id in _FORM_IDS.get(engine, {}).items():
        if form_id in state:
            state[key] = state[form_id]
    return engine, state


def mode_errors(state: dict, errors: list[str]) -> list[str]:
    """Parse errors of the active engine's fields only."""
    engine = _ENGINES.get(mode(state))
    if engine is None:
        return errors
    hidden = set(DEFAULTS) - _ENGINE_FIELDS[engine]
    return [e for e in errors if not any(e.startswith(f"'{f}'") for f in hidden)]


def resolve_state(state: dict) -> dict:
    engine, state = _engine_state(state)
    if engine is None:
        return state
    return (_resolve_state_ps if engine is _PS else engine.resolve_state)(state)


def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    engine, state = _engine_state(state)
    if engine is None:
        return [], [], ["A 6221 write pulse with the DC 2182 read is not a program here — use the "
                        "lock-in read, or the nonlocal-switching program for 6221 pulses + a 2182A read."]
    return (_build_summary_ps if engine is _PS else engine.build_summary)(state)


def compute_filename_preview(state: dict) -> Optional[str]:
    engine, state = _engine_state(state)
    if engine is None:
        return None
    return (_preview_ps if engine is _PS else engine.compute_filename_preview)(state)


def build_plan(state: dict, data_root: Path):
    """The active combination's plan (its engine's own MeasurementPlan type)."""
    engine, state = _engine_state(state)
    if engine is None:
        raise ValueError("6221 write pulse + DC read is not a program")
    return (_build_plan_ps if engine is _PS else engine.build_plan)(state, data_root)


def engine(plan):
    """The module that runs `plan` — its run_plan / header / PNG / type code / RunScreen."""
    for eng in (h2, i1):
        if isinstance(plan, eng.MeasurementPlan):
            return eng
    return _PS


# ── run screen ─────────────────────────────────────────────────────────────

class RunScreen(MeasurementRunScreen):
    ABORT_STATUS = "Abort requested — finishing this amplitude, then ramping the 6221 + magnet down …"
    POINT_STATUS = "Point {n} / {total}."
    TABLE_COLUMNS = ("amp #", "I_mag (A)", "V_pulse (V)", "I_pulse (A)", "V_xy (V)", "R_xy (Ω)", "T1 (K)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE
    PNG_SUFFIX = PNG_SUFFIX

    def live_plot_args(self):
        return (_live_plot_worker,)

    def table_row(self, record: dict) -> tuple:
        i_pulse = record.get("pulse_current_measured_A")
        t1 = record.get("temperature_1_K")
        return (
            str(record["amplitude_index"] + 1),
            f"{record['magnet_current_A']:g}" if record.get("magnet_current_A") is not None else "—",
            f"{record['pulse_amplitude_V']:.4g}",
            f"{i_pulse:.4e}" if i_pulse is not None else "—",
            f"{record['hall_voltage_V']:.4e}",
            f"{record['hall_resistance_ohm']:.5g}",
            f"{t1:.3f}" if t1 is not None else "—",
        )


# ── app / form ─────────────────────────────────────────────────────────────

class SOTPulsedSwitchingApp(MeasurementApp):
    TITLE = "SOT pulsed switching"
    SUB_TITLE = "4200A PMU or 6221 write pulse · DC R_xy or lock-in harmonic read · static tilted field"

    # widget id -> shown for which (write pulse, read) combination
    MODE_WIDGETS = {
        "mode_pmu_pulse": lambda p, r: p == "pmu",
        "mode_pmu_config": lambda p, r: p == "pmu",
        "mode_6221_pulse": lambda p, r: p == "6221",
        "mode_dc_read": lambda p, r: r == "dc",
        "mode_dc_instruments": lambda p, r: r == "dc",
        "mode_lockin_read": lambda p, r: r == "harmonic",
        "mode_mfli": lambda p, r: r == "harmonic",
        "mode_sot2h_demods": lambda p, r: (p, r) == ("pmu", "harmonic"),
        "mode_sot1i_demod": lambda p, r: (p, r) == ("6221", "harmonic"),
        "mode_sot1i_harmonic": lambda p, r: (p, r) == ("6221", "harmonic"),
    }
    # a former program's settings file -> the (write pulse, read) it was
    LEGACY = ((h2, ("pmu", "harmonic")), (i1, ("6221", "harmonic")))

    def __init__(self, pulse_source: Optional[str] = None, read_mode: Optional[str] = None) -> None:
        super().__init__()
        self._forced = {"pulse_source": pulse_source, "read_mode": read_mode}

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
                        "Write pulse × read",
                        select_field("pulse_source", "Write pulse", PULSE_SOURCES, DEFAULTS["pulse_source"]),
                        select_field("read_mode", "Read", READ_MODES, DEFAULTS["read_mode"],
                                     hint="4200A+DC → SOTPS · 4200A+lock-in → SOT2H · 6221+lock-in → SOT1I"),
                    )
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
                        id="mode_pmu_pulse",
                    )
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
                        field("wave_pulse_width_s", "Requested pulse width (s)",
                              DEFAULTS["wave_pulse_width_s"],
                              hint="No rise/fall control; actual width is measured and logged "
                                   "as pulse_width_measured_s. See the module docstring."),
                        field("pulse_compliance_V", "Pulse voltage compliance (V)",
                              DEFAULTS["pulse_compliance_V"]),
                        id="mode_6221_pulse",
                    )
                    yield card(
                        "Delayed read",
                        field("delay_after_pulse_s", "Delay after pulse (s)",
                              DEFAULTS["delay_after_pulse_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")],
                              hint="Wait between pulse end and the read."),
                        field("sense_current_values", "6221 read current (A)",
                              DEFAULTS["sense_current_values"], kind="text",
                              hint="DC read: ±I sense current; lock-in read: AC amplitude, peak. "
                                   "Keep well below the switching current. Single value, or "
                                   "comma-separated list — one complete amplitude sweep runs "
                                   "per value, each saved to its own file."),
                    )
                    yield card(
                        "DC R_xy read (6221 ±I + 2182)",
                        field("n_reversals", "Reversal pairs per read", DEFAULTS["n_reversals"],
                              kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("settle_after_enable_s", "6221 settle after enable (s)",
                              DEFAULTS["settle_after_enable_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        id="mode_dc_read",
                    )
                    yield card(
                        "Lock-in read (6221 AC + MFLI)",
                        field("frequency_Hz", "AC excitation frequency (Hz)",
                              DEFAULTS["frequency_Hz"],
                              hint="Avoid exact multiples of 50/60 Hz."),
                        field("n_averages", "MFLI samples averaged per read",
                              DEFAULTS["n_averages"], kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        field("lock_settle_s", "Settle after PLL lock (s)",
                              DEFAULTS["lock_settle_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        field("lock_timeout_s", "PLL lock timeout (s)",
                              DEFAULTS["lock_timeout_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")],
                              hint="A timeout is logged, not fatal — the row is tagged "
                                   "reference_locked=False."),
                        id="mode_lockin_read",
                    )
                    yield card(
                        "Lock-in harmonic (6221 pulse)",
                        field("harmonic", "Harmonic to lock in on", DEFAULTS["harmonic"],
                              kind="integer",
                              hint="2 = standard harmonic-Hall SOT signal (default). "
                                   "1 = resistive AHE/PHE."),
                        id="mode_sot1i_harmonic",
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
                            id="mode_pmu_config",
                        )
                        yield card(
                            "Keithley 6221",
                            field("source_visa_resource", "6221 VISA resource",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            field("compliance_V", "6221 compliance (V)", DEFAULTS["compliance_V"],
                                  hint="Keep low — caps what an open contact can put on the "
                                       "shared bus. Read needs < 1 V."),
                            muted=True,
                        )
                        yield card(
                            "Keithley 2182 + DC read",
                            field("voltmeter_visa_resource", "2182 (Hall voltage)",
                                  DEFAULTS["voltmeter_visa_resource"], kind="text"),
                            field("source_delay_s", "6221 source delay (s)", DEFAULTS["source_delay_s"]),
                            field("nplc", "2182 NPLC", DEFAULTS["nplc"]),
                            switch_field("auto_range", "2182 auto-range", DEFAULTS["auto_range"]),
                            muted=True,
                            id="mode_dc_instruments",
                        )
                        yield card(
                            "Zurich Instruments MFLI + 6221 marker",
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
                            field("pll_demod_index", "PLL phase-detector demod index (≠ 1f/2f demods)",
                                  DEFAULTS["pll_demod_index"], kind="integer",
                                  hint="extrefs/N/adcselect is read-only on real firmware — the PLL "
                                       "is steered via THIS dedicated demod's own adcselect/oscselect "
                                       "instead. Must differ from both demod indices below."),
                            select_field("automode", "PLL bandwidth adaptation",
                                  h2.AUTOMODE_OPTIONS, int(DEFAULTS["automode"]),
                                  hint=h2.AUTOMODE_HINT),
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
                            field("phasemarker_line", "Trigger Link phase-marker pin (1-6)",
                                  DEFAULTS["phasemarker_line"], kind="integer",
                                  hint="Wire this pin to the MFLI's Aux In. Confirm it isn't the "
                                       "6221's factory-default Trigger Link pin before assuming "
                                       "it's free."),
                            muted=True,
                            id="mode_mfli",
                        )
                        yield card(
                            "MFLI demodulators (1f + 2f)",
                            field("demod1_index", "1f demodulator index", DEFAULTS["demod1_index"],
                                  kind="integer"),
                            field("demod2_index", "2f demodulator index", DEFAULTS["demod2_index"],
                                  kind="integer",
                                  hint="Defaults skip index 0 — that's the PLL phase-detector demod "
                                       "above. See the module docstring's 'Bench-verify' section."),
                            muted=True,
                            id="mode_sot2h_demods",
                        )
                        yield card(
                            "MFLI demodulator",
                            field("demod_index", "Demodulator index", DEFAULTS["demod_index"],
                                  kind="integer",
                                  hint="Default skips index 0 — that's the PLL phase-detector demod "
                                       "above. See the module docstring's 'Bench-verify' section."),
                            muted=True,
                            id="mode_sot1i_demod",
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

    def _build_plan(self, state: dict):
        return build_plan(state, self.data_root)

    # ── write-pulse × read toggles ───────────────────────────────────────────

    def _read_settings(self) -> dict:
        """This form's settings plus the former 2nd-harmonic and 6221-only
        programs' (their keys renamed to this form's ids), so no program's last
        values are lost by the merge. Until this form has been saved with the
        toggles, the files are merged oldest-first and the toggles follow the
        program used last; after that, this form's own values win."""
        files = [(SETTINGS_PATH, _PS, ("pmu", "dc"))] + [
            (eng.SETTINGS_PATH, eng, combo) for eng, combo in self.LEGACY]
        loaded = []
        for path, eng, combo in files:
            try:
                loaded.append((path.stat().st_mtime, _as_form(eng, json.loads(path.read_text())), combo))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass
        merged: dict = {}
        own = next((d for _, d, c in loaded if c == ("pmu", "dc")), {})
        if "pulse_source" in own:                       # saved since the merge
            for _, d, _ in loaded:
                merged.update(d)
            merged.update(own)
        elif loaded:
            loaded.sort(key=lambda item: item[0])
            for _, d, _ in loaded:
                merged.update(d)
            merged["pulse_source"], merged["read_mode"] = loaded[-1][2]
        merged.update({k: v for k, v in self._forced.items() if v})
        return merged

    def on_mount(self) -> None:
        super().on_mount()
        for key, value in self._forced.items():
            if value:
                self.query_one(f"#{key}", Select).value = value
        self._show_mode()

    def _show_mode(self) -> None:
        pulse = self.query_one("#pulse_source", Select).value
        read = self.query_one("#read_mode", Select).value
        for widget_id, shown in self.MODE_WIDGETS.items():
            self.query_one(f"#{widget_id}").display = shown(pulse, read)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("pulse_source", "read_mode"):
            self._show_mode()
        super().on_select_changed(event)

    def parse_state(self) -> tuple[dict, list[str]]:
        state, errors = super().parse_state()
        return state, mode_errors(state, errors)

    def run_screen(self, plan):
        return engine(plan).RunScreen(plan)


def main() -> None:
    SOTPulsedSwitchingApp().run()


if __name__ == "__main__":
    main()
