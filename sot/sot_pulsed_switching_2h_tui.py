#!/usr/bin/env python3
"""
4200A-pulse + lock-in-read engine of the SOT pulsed-switching form  (type SOT2H)
=================================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-11 (a mode of sot_pulsed_switching_tui.py since 2026-09-23)

The "Pulse = 4200A PMU, Read = lock-in harmonic" engine behind the switching
form in sot_pulsed_switching_tui.py: its parameter surface, summary, plan
builder, run loop, header, PNG and RunScreen. The delayed read is a 6221 AC
(phase marker on Trigger Link) with a single Zurich Instruments MFLI,
externally referenced to that marker via its Aux Input, reading V_xy at the
1st and 2nd harmonic. See sot/sot_pulsed_switching_2h.py's module docstring
for the wiring and the "Bench-verify" section on the MFLI ExtRef node paths.

Run:  python sot_pulsed_switching_2h_tui.py   (opens the form in this mode)
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



from sot.sot_pulsed_switching_2h import (
    _READ_COMPLIANCE_CEILING_V,
    _READ_CURRENT_CEILING_A,
    _check_extref_demod_conflict,
    _check_read_safety,
    AmplitudePoint,
    ACSourceConfig,
    DemodConfig,
    ExtRefConfig,
    FilterConfig,
    GaussmeterConfig,
    Keithley4200AConfig,
    MagnetConfig,
    PMUPulseConfig,
    ReadConfig,
    TemperatureControllerConfig,
    configure_demodulator,
    configure_external_reference,
    configure_pmu_pulse,
    connect,
    connect_4200a,
    connect_ac_source,
    connect_device,
    connect_gaussmeter,
    connect_magnet,
    connect_temperature_controller,
    list_user_libraries,
    run_measurement,
    set_magnet_current,
    shutdown_4200a,
    shutdown_ac_source,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_temperature_controller,
)
from sot.sot_pulsed_switching_2h import _six221_ac_output_off
from dc.dc_sweep_utils import linear_sweep, parse_value_list, safe_shutdown
from instruments.data_dir import validate_directory
from instruments.field_geometry import field_direction_summary_line
from instruments.data_naming import (
    RunContext,
    allocate_run,
    record_run,
    preview_raw_filename,
)
from instruments.keithley4200a import pulse_once_s
from instruments.keithley6221 import ac_source_restart_s
from instruments.kepco_magnet import magnet_move_s
from instruments.lakeshore475 import read_field_s
from instruments.mfli_daq import acquire_s
from instruments.run_time import (
    ARM_S, GPIB_TXN_S, LOCK_TYP_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
    RunCost,
)
from instruments.tui_common import (
    MeasurementRunScreen,
    format_si,
    parse_sensor_uids,
    run_screen_bindings,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
)

log = logging.getLogger("sot_pulsed_switching_2h_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "sot_pulsed_switching_2h_tui_settings.json"

MEASUREMENT_TYPE = "SOT2H"

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
    # delayed 1f/2f read (6221 AC + MFLI)
    "sense_current_values": "1e-4",
    "compliance_V": "2.0",
    "frequency_Hz": "977.0",
    "phasemarker_line": "1",
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
    "demod1_index": "1",
    "demod2_index": "2",
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
# sot_pulsed_switching_2h.py for the full rationale.
AUTOMODE_OPTIONS: list[tuple[str, int]] = [
    ("2 — low bandwidth", 2),
    ("3 — high bandwidth", 3),
    ("4 — dynamic (auto)", 4),
]
AUTOMODE_HINT = ("2=most forgiving acquisition (marginal/noisy signal), "
                 "3=fastest tracking once locked, 4=auto-adapts (default).")

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
    "frequency_Hz": float,
    "phasemarker_line": int,
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
    "demod1_index": int,
    "demod2_index": int,
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
TEXT_FIELDS = ["k4200_visa_resource", "pmu_library", "pmu_module",
               "pmu_id", "pmu_return_names", "device", "cooldown",
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
    """(list, None) or ([], error) — one or more 6221 AC read-current
    amplitudes (comma-separated); nests with the assist-current list
    (amplitude outer, since changing it means a full 6221 AC re-arm; magnet
    inner), each pair its own complete amplitude sweep, its own file."""
    try:
        return parse_value_list(state["sense_current_values"]), None
    except ValueError as exc:
        return [], str(exc)


# ── formatting helpers (per-TUI copies) ─────────────────────────────────────


def run_costs(state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (one complete amplitude sweep per file: sense current outer x assist
    current inner, like MeasurementPlan.series_values). Also drives the run
    screen's progress bar, so estimate and live ETA cannot disagree."""
    amps = state.get("amplitude_list", [])
    series = list(itertools.product(state.get("sense_currents_A", []),
                                    state.get("magnet_currents_A", [])))
    rc = RunCost(len(amps) * max(1, len(series)))
    magnet = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    gauss = GaussmeterConfig(n_averages=state["gaussmeter_n_averages"],
                             read_delay_s=state["gaussmeter_read_delay_s"])
    has_temp = state["enable_temperature"] and bool(parse_sensor_uids(state["temperature_sensor_uids"]))
    lock_typ = min(LOCK_TYP_S, state["lock_timeout_s"])   # the lock wait returns at the first lock
    # run_measurement(): 6221 AC off (2 writes) -> PMU pulse -> wait -> AC on (enable, ARM, START =
    # 3 writes) -> PLL lock -> settle -> 1f + 2f acquire, one shared window -> AC off (2) ->
    # frequency read-back (1) -> temperature -> CSV rewrite
    rc.each("post-pulse wait", state["delay_after_pulse_s"])
    rc.each("PMU pulse", pulse_once_s(state["n_pulses"], state["pulse_period_s"]))
    rc.each("6221 re-arm", ARM_S)
    rc.each("PLL lock", lock_typ, worst_extra=max(0.0, state["lock_timeout_s"] - lock_typ))
    rc.each("settle", state["settle_after_enable_s"])
    rc.each("MFLI reads", acquire_s(state["filter_time_constant_s"], state["n_averages"],
                                    max(state["sample_rate_Hz"], 1.0)))
    rc.each("overhead", 8 * GPIB_TXN_S + POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    prev = 0.0                                       # magnet starts at 0 A
    for k, (_I_sense, I_mag) in enumerate(series):
        first = k * len(amps)                        # this file's first point
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
    k4200_cfg: Keithley4200AConfig
    pmu_cfg: PMUPulseConfig
    ac_cfg: ACSourceConfig       # 6221 — AC current + phase marker
    extref_cfg: ExtRefConfig     # MFLI ExtRef PLL, locked to the phase marker
    demod1_cfg: DemodConfig      # MFLI 1f — resistive AHE/PHE anchor
    demod2_cfg: DemodConfig      # MFLI 2f — the switching signal
    mfli_host: str
    mfli_port: int
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
        """Cross product of sense (6221 AC) currents x assist-field currents
        -- one complete amplitude sweep per pair, each saved to its own
        file. Sense current is outer (changing it means a full 6221 AC
        re-arm) and magnet is inner (just a ramp, no reconnect) -- see
        dc_spin_valve_tui.py for the same nested-product pattern."""
        return list(itertools.product(self.sense_currents_A, self.magnet_currents_A))

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

def _near_multiple(f: float, m: float) -> bool:
    return min(f % m, m - f % m) < 1.0


def resolve_state(state: dict) -> dict:
    """Add the derived keys build_summary() / build_plan() read — the parsed
    lists/sweeps, each with its parse error — to a state of raw field values.
    Pure: shared by the TUI's and the web page's parse_state()."""
    state["amplitude_list"], state["amplitude_parse_error"] = _resolve_amplitudes(state)
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
                        "(3) The standby 6221 tolerates the pulse transient across its "
                        "(non-sourcing, high-Z) output stage the same as in DC mode — see "
                        "sot_pulsed_switching.py's '40 V range' section. Check the MFLI Signal "
                        "Input's absolute maximum rating (its datasheet, not this code) before "
                        "the first 40 V pulse — this code does not enforce it.")
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

    # read (6221 AC + MFLI) — the 6221 shares the main-channel pins with the
    # PMU, so a fat-fingered current/compliance lands on the MFLI input and the
    # disabled PMU output. Block at the absolute ceilings, warn below them.
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
            warnings.append(f"6221 AC current amplitude(s) {large} are large for a read — they "
                            "flow continuously through the channel while sourcing; keep them "
                            "well below the switching current.")
    if state["compliance_V"] <= 0:
        errors.append("6221 compliance must be > 0 V.")
    elif state["compliance_V"] > _READ_COMPLIANCE_CEILING_V:
        errors.append(f"6221 compliance {state['compliance_V']:g} V exceeds the "
                      f"{_READ_COMPLIANCE_CEILING_V:g} V safety ceiling — on an open contact the "
                      "6221 rails to this across the shared bus.")
    elif state["compliance_V"] > 5.0:
        warnings.append(f"6221 compliance {state['compliance_V']:g} V — the Hall read needs "
                        "< 1 V of headroom; a lower value limits what an open contact can put "
                        "on the shared bus.")
    if not 1e-3 <= state["frequency_Hz"] <= 1e5:
        errors.append("6221 AC frequency must be in [1 mHz, 100 kHz] (WAVE mode range).")
    elif _near_multiple(state["frequency_Hz"], 50.0) or _near_multiple(state["frequency_Hz"], 60.0):
        warnings.append(f"{state['frequency_Hz']:g} Hz is close to a 50/60 Hz line harmonic — "
                        "pick an offset frequency to avoid mains pickup.")
    if not 1 <= state["phasemarker_line"] <= 6:
        errors.append("Trigger Link phase-marker line must be 1-6.")
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
    info.append(f"PMU module: {state['pmu_library']}/{state['pmu_module'] or '<unset>'} "
                f"({state['pmu_id']} ch {state['pmu_channel']})")
    if sense_currents:
        info.append(f"AC excitation: {format_si(sense_currents[0], 'A')} peak @ "
                    f"{state['frequency_Hz']:g} Hz, phase marker on Trigger Link pin "
                    f"{state['phasemarker_line']} → MFLI Aux In {state['aux_input_ch'] + 1}")

    demod_indices = {
        "PLL phase-detector demod": state["pll_demod_index"],
        "1f demod": state["demod1_index"],
        "2f demod": state["demod2_index"],
    }
    seen: dict[int, str] = {}
    for label, idx in demod_indices.items():
        if idx in seen:
            errors.append(f"{label} and {seen[idx]} both use demod index {idx} — "
                           "extrefs/N/adcselect is read-only, so the PLL phase-detector "
                           "demod can't double as a signal demod. Pick distinct indices.")
        else:
            seen[idx] = label

    # Display-only current estimate off the load-line DUT resistance. It is a hint
    # for picking amplitudes; the honest pulse axis is the module's measured
    # pulse_current_measured_A, and pmu_dut_res_ohm is never written as data.
    r_ch = state.get("pmu_dut_res_ohm", 0.0)
    if r_ch > 0 and amps:
        i_lo, i_hi = min(amps) / r_ch, max(amps) / r_ch
        info.append(f"At DUT R ≈ {r_ch:g} Ω: pulses ≈ "
                    f"{format_si(i_lo, 'A')}…{format_si(i_hi, 'A')}")
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

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("SOT pulsed switching (2f) — live")
    except Exception:
        pass
    ax.set_xlabel("Pulse amplitude (V)")
    ax.set_ylabel("V_2f (V)")
    ax.set_title("Live — V_2f vs pulse amplitude")
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
            ys.append(rec["2f_R_V"])
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
    static assist-field direction, the fixed 6221 AC read current, the
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
            [r["2f_R_V"] for r in records],
            "o-", ms=4, lw=1, alpha=0.6, color="tab:blue")
    ax.set_xlabel("Pulse amplitude (V)")
    ax.set_ylabel("V_2f (V)")
    ax.set_title("V_2f vs pulse amplitude")
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
        n_averages=state["n_averages"],
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
    demod_common = dict(
        device=state["mfli_device"], osc_index=state["osc_index"],
        input_ch=state["input_ch"], differential=state["differential"],
        ac_coupling=state["ac_coupling"], input_range_V=state["input_range_V"],
        sample_rate_Hz=state["sample_rate_Hz"], filter=shared_filter,
    )
    demod1_cfg = DemodConfig(demod_index=state["demod1_index"], harmonic=1, **demod_common)
    demod2_cfg = DemodConfig(demod_index=state["demod2_index"], harmonic=2, **demod_common)
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
        "frequency_Hz": state["frequency_Hz"],
        "phasemarker_line": state["phasemarker_line"],
        "field_theta_deg": state["field_theta_deg"],
        "field_phi_deg": state["field_phi_deg"],
        "amplitude_start_V": state["amplitude_start_V"],
        "amplitude_stop_V": state["amplitude_stop_V"],
        "amplitude_step_V": state["amplitude_step_V"],
        "amplitude_bidirectional": state["amplitude_bidirectional"],
        "amplitudes_V": state["amplitude_list"],
    }
    return MeasurementPlan(
        k4200_cfg=k4200_cfg, pmu_cfg=pmu_cfg, ac_cfg=ac_cfg, extref_cfg=extref_cfg,
        demod1_cfg=demod1_cfg, demod2_cfg=demod2_cfg,
        mfli_host=state["mfli_host"], mfli_port=state["mfli_port"],
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
    """Connect the 4200A PMU, the MFLI (ExtRef-locked to the 6221 marker),
    magnet and gaussmeter, then per (read current, assist-field current) pair:
    re-arm the 6221 AC source, park the field, and record one amplitude sweep
    (own run number, own file) through record_run(); always shut everything
    down (6221 first). Pure — the TUI's RunScreen runs it with its own callbacks."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    k4200 = source = daq = magnet = gaussmeter = temp_ctrl = None
    try:
        on_status("Connecting to Keithley 4200A (KXCI) …")
        k4200 = connect_4200a(plan.k4200_cfg)
        try:
            log.info("Installed user libraries (UL):\n%s", list_user_libraries(k4200))
        except Exception:
            log.warning("Could not read `UL` — set the PMU module name from the 4200A manually.")
        configure_pmu_pulse(k4200, plan.pmu_cfg)

        _check_extref_demod_conflict(plan.demod1_cfg, plan.demod2_cfg, plan.extref_cfg)

        on_status("Connecting to MFLI …")
        daq = connect(plan.mfli_host, plan.mfli_port)
        connect_device(daq, plan.extref_cfg.device, interface="1GbE")
        configure_external_reference(daq, plan.extref_cfg, plan.ac_cfg.frequency_Hz)
        configure_demodulator(daq, plan.demod1_cfg)
        configure_demodulator(daq, plan.demod2_cfg)

        on_status("Connecting to Kepco magnet + Lake Shore 475 …")
        magnet = connect_magnet(plan.magnet_cfg)
        gaussmeter = connect_gaussmeter(plan.gauss_cfg)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        points = [AmplitudePoint(amplitude_V=float(v)) for v in plan.amplitudes_V]

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
                lambda point_cb, write_csv, _ctx=ctx, _I=I_mag, _src=source: run_measurement(
                    k4200, plan.pmu_cfg, _src, daq, plan.demod1_cfg, plan.demod2_cfg,
                    plan.extref_cfg, plan.read_cfg, points,
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
            safe_shutdown("6221", lambda: shutdown_ac_source(source))
        if k4200 is not None:
            # channels=() — this program never forces the 4200A SMUs.
            safe_shutdown("4200A", lambda: shutdown_4200a(k4200, channels=()))
        if magnet is not None:
            safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
        if gaussmeter is not None:
            safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


PNG_SUFFIX = "V2f_vs_amp"


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (the RunScreen calls this)."""
    _save_measurement_png(records, png_path, plan=plan, comment=comment)


# ── run screen ─────────────────────────────────────────────────────────────

class RunScreen(MeasurementRunScreen):
    ABORT_LABEL = "Abort (safe shutdown)"
    BINDINGS = run_screen_bindings(ABORT_LABEL)
    ABORT_STATUS = "Abort requested — finishing this amplitude, then shutting the 6221 + magnet down …"
    POINT_STATUS = "Point {n} / {total}."
    TABLE_COLUMNS = ("amp #", "I_mag (A)", "V_pulse (V)", "I_pulse (A)", "V_1f (V)", "V_2f (V)", "locked", "T1 (K)")
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
            f"{record['1f_R_V']:.4e}",
            f"{record['2f_R_V']:.4e}",
            "yes" if record.get("reference_locked") else "no",
            f"{t1:.3f}" if t1 is not None else "—",
        )


def main() -> None:
    """This program is a mode of the SOT pulsed-switching form now — open that
    form on 4200A PMU + lock-in read (SOT2H)."""
    from sot.sot_pulsed_switching_tui import SOTPulsedSwitchingApp
    SOTPulsedSwitchingApp(pulse_source="pmu", read_mode="harmonic").run()


if __name__ == "__main__":
    main()
