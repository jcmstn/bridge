#!/usr/bin/env python3
"""
Textual TUI for sot/sot_nonlocal_switching.py
=========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-21

Nonlocal spin-current switching with only a Keithley 6221 and 2182A: an
optional external-field initialization (Kepco magnet + Lake Shore 475; one
run per initial-state current), then a sweep of injector pulses — each ONE
lobe 0 → ±I → 0 (6221 WAVE square, one cycle, never a ± pair) — every one
followed by a DC nonlocal read on the 2182A (current-reversal averaged, or one fixed
polarity — switchable). Type NLSW. See
sot/sot_nonlocal_switching.py's module docstring — wiring, protocol, the
literature this is modelled on and the artifact checklist — before running
this on a real device.

Run:  uv run python sot/sot_nonlocal_switching_tui.py
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

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

from sot.sot_nonlocal_switching import (
    _READ_COMPLIANCE_CEILING_V,
    _READ_CURRENT_CEILING_A,
    _NO_PULSE_A,
    _WRITE_CURRENT_HARD_MAX_A,
    _check_pulse_currents,
    _check_read_safety,
    _check_write_safety,
    _output_off,
    GaussmeterConfig,
    MagnetConfig,
    PulsePoint,
    ReadConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    WritePulseConfig,
    connect,
    connect_gaussmeter,
    connect_magnet,
    connect_temperature_controller,
    connect_voltmeter,
    initialize_with_field,
    run_measurement,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_source,
    shutdown_temperature_controller,
    switch_currents_A,
)
from dc.dc_sweep_utils import linear_sweep, parse_value_list, safe_shutdown
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
from instruments.keithley2182 import read_time_s
from instruments.keithley6221 import reversal_avg_s, wave_pulse_s
from instruments.kepco_magnet import magnet_move_s
from instruments.lakeshore475 import read_field_s
from instruments.live_plot import start_live_plot
from instruments.run_time import (
    GPIB_TXN_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
    RunCost, progress_step, progress_total,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
    NewSampleScreen,
    StatusCommentScreen,
    sample_options,
)

log = logging.getLogger("sot_nonlocal_switching_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "sot_nonlocal_switching_tui_settings.json"

MEASUREMENT_TYPE = "NLSW"

NLSW_DESCRIPTION = (
    "Nonlocal spin-current switching, Keithley 6221 + 2182A only, modelled on the "
    "Kimura/Otani experiments with a DC read. Optionally initialize the magnet with an "
    "external field first (Kepco; one run per initial-state current). Then, per amplitude, "
    "the 6221 fires ONE hardware-timed pulse through the injector (WAVE square, one cycle: "
    "0 → +I → 0 or 0 → −I → 0 — a single lobe of either sign, never a ± pair), waits, and "
    "reads the nonlocal resistance across detector magnet / reference electrode with a small "
    "DC I_sense on the 2182A — current-reversal averaged by default, or (switch off) one fixed "
    "polarity so the read's own spin current never alternates. A step in R_NL that stays is a "
    "switching event (V_even tracks Joule heating / thermal EMF). Sweep one polarity upward "
    "from a field-initialized state — then only one initial state can switch, so run the "
    "opposite one as the control — or sweep −I → +I → −I for a hysteresis loop. "
    "Wiring: 6221 HI → injector, OUTPUT LOW (floating) → return electrode away from the "
    "detector, 2182A ch1 → detector magnet / reference electrode past it."
)

# Wiring schematic — shown on this program's card in bridge_tui.py.
NLSW_SCHEMATIC = """\
  KEITHLEY 6221  (the ONLY source — no 4200A, no MFLI)
    HI ──▶ injector electrode
    LO ──▶ return electrode, a bit further from the injector, on the side away
           from the detector (the current returns through it). Triax OUTPUT LOW
           FLOATING — the program sets it.
    WAVE square, ONE cycle: 0 → ±I → 0 (one lobe, never a ± pair) = the write pulse,
    then plain DC I_sense (±I current reversal by default, switchable) = the read.
  KEITHLEY 2182A  ch1 ──▶ detector magnet electrode / reference electrode past the
                          magnet (V_NL). ch2 unused (its LO is bonded to ch1 LO).

  KEPCO BOP-GL + LAKE SHORE 475  (optional) — external-field initialization of the
    magnet state before the sweep; one run per init current (e.g. +B and -B).
"""

DEFAULTS: dict = {
    # write pulse (6221 WAVE, one lobe 0 → ±I → 0)
    "pulse_current_start_A": "1e-3",
    "pulse_current_stop_A": "10e-3",
    "pulse_current_step_A": "0.5e-3",
    "amplitude_bidirectional": False,
    "pulse_width_s": "1e-3",
    "pulse_compliance_V": "5.0",
    # nonlocal read (6221 DC ± / 2182A)
    "sense_current_A": "1e-4",
    "compliance_V": "2.0",
    "reversal_enabled": True,
    "n_averages": "5",
    "source_delay_s": "0.1",
    "delay_after_pulse_s": "1.0",
    "nplc": "5",
    "switch_sigma": "5",
    "R_P_ohm": "",
    "R_AP_ohm": "",
    # field initialization (blank = magnet untouched)
    "init_magnet_currents": "",
    "sweep_magnet_current_A": "0",
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
    "pulse_current_start_A": float,
    "pulse_current_stop_A": float,
    "pulse_current_step_A": float,
    "pulse_width_s": float,
    "pulse_compliance_V": float,
    "sense_current_A": float,
    "compliance_V": float,
    "n_averages": int,
    "source_delay_s": float,
    "delay_after_pulse_s": float,
    "nplc": float,
    "switch_sigma": float,
    "sweep_magnet_current_A": float,
    "field_settle_tolerance_mT": float,
    "current_limit_A": float,
    "magnet_voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
}
TEXT_FIELDS = ["device", "cooldown",
               "source_visa_resource", "voltmeter_visa_resource", "magnet_visa_resource",
               "gaussmeter_visa_resource", "temperature_visa_resource",
               "temperature_sensor_uids", "init_magnet_currents", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K", "R_P_ohm", "R_AP_ohm"]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]

# Every Switch id on the form. Hardcoded in collect_raw / _load_settings /
# parse_state -- they must move together, and parse_state runs on every
# keystroke, so a stale entry here is an immediate crash.
SWITCH_FIELD_IDS = ("enable_temperature", "amplitude_bidirectional", "reversal_enabled")


def parse_sensor_uids(raw: str) -> tuple:
    uids = [u.strip() for u in raw.split(",") if u.strip()]
    return tuple(uids[:2])


def _resolve_pulse_currents(state: dict) -> tuple[list[float], Optional[str]]:
    """(list, None) or ([], error) — the pulse-current sweep from start/stop/
    step (+ the bidirectional toggle). Each amplitude is one lobe 0 → ±I → 0, so
    either sign is fine (a sweep through 0 gets one read-only 0 A point). Shared
    by parse_state and the tests."""
    try:
        if state["pulse_current_start_A"] == state["pulse_current_stop_A"]:
            raise ValueError("Pulse current start and stop must differ.")
        return [float(v) for v in linear_sweep(
            state["pulse_current_start_A"], state["pulse_current_stop_A"],
            state["pulse_current_step_A"],
            bidirectional=state["amplitude_bidirectional"])], None
    except ValueError as exc:
        return [], str(exc)


def _resolve_init_currents(state: dict) -> tuple[list[Optional[float]], Optional[str]]:
    """([None], None) when blank — no field initialization, magnet untouched.
    Otherwise (list, None) or ([], error): one or more comma-separated magnet
    currents, each its own complete sweep and file (its own initial state)."""
    if not state["init_magnet_currents"].strip():
        return [None], None
    try:
        return list(parse_value_list(state["init_magnet_currents"])), None
    except ValueError as exc:
        return [], str(exc)


def resolve_state(state: dict) -> dict:
    """Add the derived keys build_summary() reads — the pulse-current sweep and
    the init-current list, each with its parse error — to a state built from the
    raw field values. Shared by the TUI's and the web page's parse_state()."""
    state["pulse_current_list"], state["pulse_current_parse_error"] = _resolve_pulse_currents(state)
    state["init_currents_A"], state["init_currents_parse_error"] = _resolve_init_currents(state)
    return state


# ── formatting helpers (per-TUI copies) ─────────────────────────────────────

def format_si(value: float, unit: str) -> str:
    av = abs(value)
    if av == 0:
        return f"0 {unit}"
    for scale, prefix in ((1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, "")):
        if av < scale * 1000:
            return f"{value / scale:.3f} {prefix}{unit}"
    return f"{value:.3e} {unit}"


def run_costs(state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order: per
    file (one per initial-state current) a baseline read, then one point per
    pulse current. Also drives the run screen's progress bar, so estimate and
    live ETA cannot disagree."""
    pulses = state.get("pulse_current_list", [])
    inits = list(state.get("init_currents_A", [None]))       # == MeasurementPlan.series_values
    per_file = len(pulses) + 1
    rc = RunCost(per_file * max(1, len(inits)))
    magnet = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    gauss = GaussmeterConfig(n_averages=state["gaussmeter_n_averages"],
                             read_delay_s=state["gaussmeter_read_delay_s"])
    has_temp = state["enable_temperature"] and bool(parse_sensor_uids(state["temperature_sensor_uids"]))
    one_read_s = read_time_s(state["nplc"])
    if state["reversal_enabled"]:
        read_s = reversal_avg_s(state["n_averages"], state["source_delay_s"], one_read_s)
    else:                                                    # one source write, one delay, plain average
        read_s = GPIB_TXN_S + state["source_delay_s"] + state["n_averages"] * one_read_s
    # run_measurement(), per point: output off (3 writes) -> [WAVE pulse -> wait, only if fired]
    # -> DC read on (4 writes) -> read -> output off (3) -> temperature -> CSV rewrite.
    # The baseline (index 0) and any 0 A point fire nothing.
    for f in range(max(1, len(inits))):
        for j, I in enumerate([0.0, *pulses]):
            idx = f * per_file + j
            if abs(I) > _NO_PULSE_A:
                rc.at("pulses", wave_pulse_s(state["pulse_width_s"]), idx)
                rc.at("post-pulse wait", state["delay_after_pulse_s"], idx)
            rc.at("reads", read_s, idx)
            rc.at("overhead", 10 * GPIB_TXN_S + POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0), idx)
    cur = 0.0                                                # magnet starts at 0 A
    for f, init_A in enumerate(inits):
        if init_A is not None:                               # initialize_with_field(): init, read, hold
            (t1, w1), (t2, w2) = (magnet_move_s(abs(init_A - cur), magnet),
                                  magnet_move_s(abs(state["sweep_magnet_current_A"] - init_A), magnet))
            rc.at("magnet", t1 + t2, f * per_file, worst_extra=(w1 - t1) + (w2 - t2))
            rc.at("field read", 2 * read_field_s(gauss), f * per_file)   # once between, once in run_measurement
            cur = state["sweep_magnet_current_A"]
        rc.at("per-file", PER_FILE_S, f * per_file)
    rc.at("per-run", PER_RUN_S, 0)
    if any(i is not None for i in inits):                    # shutdown_magnet() ramps back to 0 A
        rc.tail("ramps", magnet_move_s(abs(cur), magnet, with_field=False)[0])
    return rc


# ── plan ────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    pulse_cfg: WritePulseConfig
    read_cfg: ReadConfig
    source_visa: str
    volt_cfg: VoltmeterConfig
    magnet_cfg: MagnetConfig
    gauss_cfg: GaussmeterConfig
    pulse_currents_A: List[float]
    init_currents_A: List[Optional[float]]   # [None] = no field initialization
    sweep_magnet_current_A: float            # magnet current held during the sweep
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
    def series_values(self) -> List[Optional[float]]:
        """One complete sweep per initial-state current, each saved to its own
        file, exactly like starting the run by hand for each."""
        return list(self.init_currents_A)

    @property
    def uses_magnet(self) -> bool:
        return any(i is not None for i in self.init_currents_A)

    @property
    def total_points(self) -> int:
        """+1 per file for the baseline read."""
        return (len(self.pulse_currents_A) + 1) * max(1, len(self.series_values))


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                        status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """`extra` carries this run's own init current / measured init field on top
    of the plan-wide header_extra — allocate_run() is called fresh per run."""
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
        "I_switch_A": ", ".join(f"{i:.6g}" for i in switch_currents_A(records)),
    }
    fields.update(plan.header_extra)
    if extra:
        fields.update(extra)
    return fields


def _run_extra(plan: "MeasurementPlan", init_A: Optional[float], init_info: dict) -> dict:
    return {"init_magnet_current_A": init_A,
            "sweep_magnet_current_A": plan.sweep_magnet_current_A if init_A is not None else None,
            **init_info}


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

    # write pulse
    amps = state.get("pulse_current_list", [])
    if state.get("pulse_current_parse_error"):
        errors.append(f"Pulse currents: {state['pulse_current_parse_error']}")
    else:
        over = [a for a in amps if abs(a) > _WRITE_CURRENT_HARD_MAX_A]
        if over:
            errors.append(f"Pulse current(s) {over} A exceed the 6221's hardware range "
                          f"±{_WRITE_CURRENT_HARD_MAX_A:g} A.")
        one_sign = all(a >= 0 for a in amps) or all(a <= 0 for a in amps)
        if not state["amplitude_bidirectional"]:
            back = ""
        elif one_sign:
            back = ", then back — same polarity, so no reset: R_NL must stay put"
        else:
            back = ", and back (hysteresis loop)"
        info.append(f"Pulse sweep: {len(amps)} single-lobe pulses "
                    f"{format_si(state['pulse_current_start_A'], 'A')} → "
                    f"{format_si(state['pulse_current_stop_A'], 'A')} step "
                    f"{format_si(state['pulse_current_step_A'], 'A')}{back}")
        n_zero = sum(abs(a) <= 1e-9 for a in amps)
        if n_zero:
            info.append(f"{n_zero} of the amplitudes is 0 A — a read-only point, no pulse fired.")
    if state["pulse_width_s"] <= 0:
        errors.append("Pulse width must be > 0 s.")
    if state["pulse_compliance_V"] <= 0:
        errors.append("Pulse compliance must be > 0 V.")
    info.append("Hardware-timed write pulse (WAVE square, one cycle, 0 → ±I → 0: one lobe, never a "
                "± pair; a negative one arrives one pulse width late) — the true floor "
                "is range/load-dependent. Joule heating scales as I²R·t: start well below the "
                "expected switching current, and keep compliance above I_max × R_injector or the "
                "6221 clips silently.")

    # nonlocal read
    sense = abs(state["sense_current_A"])
    if sense == 0:
        errors.append("Sense current must be non-zero.")
    elif sense > _READ_CURRENT_CEILING_A:
        errors.append(f"Sense current {state['sense_current_A']:g} A exceeds the "
                      f"{format_si(_READ_CURRENT_CEILING_A, 'A')} safety ceiling — the nonlocal "
                      "read needs µA–mA; check for a mistyped exponent.")
    nonzero = [abs(a) for a in amps if abs(a) > 1e-9]
    if nonzero and not state.get("pulse_current_parse_error") and sense > 0:
        smallest = min(nonzero)
        if sense >= smallest:
            errors.append(f"Sense current {sense:g} A must be below the smallest pulse "
                          f"({smallest:g} A) — the read itself would switch the magnet.")
        elif sense > 0.2 * smallest:
            warnings.append(f"Sense current is {100 * sense / smallest:.0f}% of the smallest pulse "
                            "— read disturb possible; keep it well below the switching current.")
    if state["compliance_V"] <= 0:
        errors.append("Read compliance must be > 0 V.")
    elif state["compliance_V"] > _READ_COMPLIANCE_CEILING_V:
        errors.append(f"Read compliance {state['compliance_V']:g} V exceeds the "
                      f"{_READ_COMPLIANCE_CEILING_V:g} V safety ceiling.")
    if state["n_averages"] < 2:
        errors.append("Averages per read must be ≥ 2 (± pairs with reversal, plain samples "
                      "without) — the SEM behind 'switched' needs two.")
    if not state["reversal_enabled"]:
        pol = "+" if state["sense_current_A"] > 0 else "−"
        warnings.append(f"Current reversal is OFF: one fixed read polarity ({pol}I_sense), plain "
                        "average. Thermal-EMF offsets are NOT cancelled (R_NL carries an offset "
                        "that can drift — steps in it still show) and V_even, the heating "
                        "proxy, is not recorded. Measure R_P / R_AP with the same setting.")
    if state["source_delay_s"] < 0:
        errors.append("Settle after polarity flip must be ≥ 0 s.")
    if state["delay_after_pulse_s"] < 0:
        errors.append("Delay after pulse must be ≥ 0 s.")
    if not 0.01 <= state["nplc"] <= 60:
        errors.append("2182A NPLC must be in [0.01, 60].")
    if state["switch_sigma"] <= 0:
        errors.append("'switched' threshold (σ) must be > 0.")
    r_p, r_ap = state.get("R_P_ohm"), state.get("R_AP_ohm")
    if (r_p is None) != (r_ap is None):
        errors.append("Give both R_P and R_AP (from a dc_spin_valve field sweep), or neither.")
    elif r_p is not None and r_p == r_ap:
        errors.append("R_P and R_AP must differ.")
    elif r_p is not None:
        info.append(f"Reference levels: R_P={r_p:g} Ω, R_AP={r_ap:g} Ω → state_AP_fraction per row; "
                    "'switched' also needs ≥ half the swing.")
    else:
        info.append("No reference levels — run dc_spin_valve first and enter R_P / R_AP to turn "
                    "R_NL into a state fraction.")

    # field initialization
    inits = state.get("init_currents_A", [None])
    real = [i for i in inits if i is not None]
    if state.get("init_currents_parse_error"):
        errors.append(f"Init magnet current(s): {state['init_currents_parse_error']}")
    else:
        over = [i for i in real if abs(i) > state["current_limit_A"]]
        if over:
            errors.append(f"Init magnet current(s) {over} A exceed the magnet limit "
                          f"±{state['current_limit_A']:g} A.")
        if abs(state["sweep_magnet_current_A"]) > state["current_limit_A"]:
            errors.append(f"Sweep magnet current {state['sweep_magnet_current_A']:g} A exceeds the "
                          f"magnet limit ±{state['current_limit_A']:g} A.")
        if not real:
            warnings.append("No field initialization: the magnet is not touched and the sweep "
                            "starts from whatever state it is in now (the baseline row shows "
                            "it). Initialize it externally first, or enter init current(s).")
        else:
            cur_str = ", ".join(f"{i:g}" for i in real)
            info.append(f"Initialization: {len(real)} initial state(s) via magnet current "
                        f"{cur_str} A, then {state['sweep_magnet_current_A']:g} A held for the "
                        "sweep (field measured by the 475). Each gets its own complete sweep "
                        "and file.")
            if len(real) == 1 and amps and (all(a >= 0 for a in amps) or all(a <= 0 for a in amps)):
                info.append("With one pulse polarity only one initial state can switch — add the "
                            "opposite sign (e.g. '5, -5') for the control run.")

    # size / time
    n = max(1, len(amps))
    n_files = max(1, len(real))
    info.append(f"{n} pulses + a baseline read"
                + (f", × {n_files} files = {(n + 1) * n_files} total points" if n_files > 1 else ""))
    info.extend(run_costs(state).lines("Estimated run time"))
    info.append("Wiring: 6221 HI → injector; OUTPUT LOW (set floating) → return electrode away from "
                "the detector; 2182A ch1 → detector magnet / reference electrode past it.")

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
    real = [i for i in state.get("init_currents_A", [None]) if i is not None]
    key = "_<I_init A>" if real else ""
    suffix = " (one file per initial state)" if len(real) > 1 else ""
    return f"{preview}{key}_<timestamp>.csv{suffix}"


# ── live plot ──────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue", reversal_enabled: bool = True) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax_r, ax_e) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    try:
        fig.canvas.manager.set_window_title("Nonlocal switching — live")
    except Exception:
        pass
    ax_r.set_ylabel("R_NL (mΩ)")
    ax_r.set_title("Live — nonlocal resistance vs pulse current")
    ax_e.set_ylabel("V_even (µV)" if reversal_enabled else "V_even (n/a — reversal off)")
    ax_e.set_xlabel("Pulse current (mA)")
    for ax in (ax_r, ax_e):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    cmap = plt.get_cmap("tab10")
    lines: dict[int, tuple] = {}
    data: dict[int, tuple[list, list, list]] = {}

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
                style = dict(ms=4, lw=1, alpha=0.6, color=cmap(idx % 10))
                (lr,) = ax_r.plot([], [], "o-", label=label, **style)
                (le,) = ax_e.plot([], [], "o-", **style)
                lines[idx] = (lr, le)
                data[idx] = ([], [], [])
                new_series = True
            xs, yr, ye = data[idx]
            xs.append(rec["pulse_current_A"] * 1e3)
            yr.append(rec["nl_resistance_ohm"] * 1e3)
            v_even = rec.get("voltage_even_V")
            ye.append(float("nan") if v_even is None else v_even * 1e6)
            updated.add(idx)
        if updated:
            for idx in updated:
                xs, yr, ye = data[idx]
                lines[idx][0].set_data(xs, yr)
                lines[idx][1].set_data(xs, ye)
            if new_series and any(lr.get_label() and not lr.get_label().startswith("_")
                                  for lr, _ in lines.values()):
                ax_r.legend(loc="best", fontsize=8)
            for ax in (ax_r, ax_e):
                ax.relim()
                ax.autoscale_view()
        return tuple(ln for pair in lines.values() for ln in pair)

    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path, comment: str = "") -> None:
    """`records` is ONE run's points -- with several initial states each run is
    saved (and plotted) on its own, exactly like a manual run.

    A small "at a glance" annotation (initial-state current, the run's own sense
    current, the switching current, the operator's comment) sits under the
    axes. Called once when the run ends (comment="") and again, to overwrite the
    PNG in place, once the operator's comment is known."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["pulse_current_A"] * 1e3 for r in records]
    has_even = any(r.get("voltage_even_V") is not None for r in records)   # blank with reversal off
    fig, axes = plt.subplots(2 if has_even else 1, 1, sharex=True, figsize=(7, 7 if has_even else 4.5))
    axes = [axes] if not has_even else list(axes)
    ax_r = axes[0]
    ax_r.plot(x, [r["nl_resistance_ohm"] * 1e3 for r in records],
              "o-", ms=4, lw=1, alpha=0.6, color="tab:blue")
    ax_r.set_ylabel("R_NL (mΩ)")
    ax_r.set_title("Nonlocal resistance vs pulse current")
    if has_even:
        axes[1].plot(x, [r["voltage_even_V"] * 1e6 for r in records],
                     "o-", ms=4, lw=1, alpha=0.6, color="tab:blue")
        axes[1].set_ylabel("V_even (µV)")
    axes[-1].set_xlabel("Pulse current (mA)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()

    lines: list[str] = []
    init = next((r["init_magnet_current_A"] for r in records
                 if r.get("init_magnet_current_A") is not None), None)
    if init is not None:
        lines.append(f"Initialized with magnet current {format_si(init, 'A')}")
    sense = sorted({r["sense_current_A"] for r in records if r.get("sense_current_A") is not None})
    if len(sense) == 1:
        lines.append(f"Sense current: {format_si(sense[0], 'A')}"
                     + ("" if records[0].get("reversal_enabled", True)
                        else " (fixed polarity, no current reversal)"))
    switched = switch_currents_A(records)
    lines.append("Switching current(s): " + ", ".join(format_si(i, "A") for i in switched)
                 if switched else "No switching detected")
    if comment:
        lines.append(f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}")
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


def build_plan(state: dict, data_root: Path) -> MeasurementPlan:
    """One parsed, validated run request from a state dict (see resolve_state()).
    Pure — shared by the TUI and the web page."""
    pulse_cfg = WritePulseConfig(
        width_s=state["pulse_width_s"], compliance_V=state["pulse_compliance_V"])
    read_cfg = ReadConfig(
        sense_current_A=state["sense_current_A"], compliance_V=state["compliance_V"],
        reversal_enabled=state["reversal_enabled"], n_averages=state["n_averages"],
        source_delay_s=state["source_delay_s"],
        delay_after_pulse_s=state["delay_after_pulse_s"], switch_sigma=state["switch_sigma"],
        R_P_ohm=state["R_P_ohm"], R_AP_ohm=state["R_AP_ohm"])
    volt_cfg = VoltmeterConfig(
        visa_resource=state["voltmeter_visa_resource"], nplc=state["nplc"], auto_range=True)
    magnet_cfg = MagnetConfig(
        visa_resource=state["magnet_visa_resource"], current_limit_A=state["current_limit_A"],
        voltage_compliance_V=state["magnet_voltage_compliance_V"],
        ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    gauss_cfg = GaussmeterConfig(
        visa_resource=state["gaussmeter_visa_resource"], unit="T",
        n_averages=state["gaussmeter_n_averages"], read_delay_s=state["gaussmeter_read_delay_s"])

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

    header_extra = {
        "pulse_current_start_A": state["pulse_current_start_A"],
        "pulse_current_stop_A": state["pulse_current_stop_A"],
        "pulse_current_step_A": state["pulse_current_step_A"],
        "amplitude_bidirectional": state["amplitude_bidirectional"],
        "pulse_currents_A": state["pulse_current_list"],
        "pulse_width_s": state["pulse_width_s"],
        "pulse_compliance_V": state["pulse_compliance_V"],
        "delay_after_pulse_s": state["delay_after_pulse_s"],
        "sense_current_A": state["sense_current_A"],
        "reversal_enabled": state["reversal_enabled"],
        "n_averages": state["n_averages"],
        "source_delay_s": state["source_delay_s"],
        "nplc": state["nplc"],
        "switch_sigma": state["switch_sigma"],
        "R_P_ohm": state["R_P_ohm"],
        "R_AP_ohm": state["R_AP_ohm"],
    }
    return MeasurementPlan(
        pulse_cfg=pulse_cfg, read_cfg=read_cfg, source_visa=state["source_visa_resource"],
        volt_cfg=volt_cfg, magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg,
        pulse_currents_A=state["pulse_current_list"], init_currents_A=state["init_currents_A"],
        sweep_magnet_current_A=state["sweep_magnet_current_A"],
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
    """Connect, then per initial state (or once, if none): initialize the magnet
    with the field, `allocate_run()` a fresh run, sweep, and finalize that run's
    raw file + index row UNCONDITIONALLY before moving on; always shut the
    instruments down. Shared by the TUI's RunScreen and the web page — each
    passes its own callbacks. `on_point` gets each record already tagged with
    series_index / series_label / init_magnet_current_A; `on_run_finished(ctx,
    records)` is where a caller saves its per-run PNG; `run_contexts` /
    `run_extras` (if given) collect each run's RunContext and header extras for
    the caller's end-of-session status/comment step. A failed run is finalized
    with status "error", then its exception is re-raised."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = magnet = gaussmeter = temp_ctrl = None
    try:
        points = [PulsePoint(pulse_current_A=float(v)) for v in plan.pulse_currents_A]
        # Checked before any hardware is touched (connect() leaves the 6221 live).
        _check_write_safety(plan.pulse_cfg)
        _check_pulse_currents(points)
        _check_read_safety(plan.read_cfg, points)

        on_status("Connecting to Keithley 6221 + 2182A …")
        source = connect(plan.source_visa, plan.read_cfg.compliance_V,
                         plan.read_cfg.source_delay_s)
        source.output_low_grounded = False     # the return goes through its own electrode
        _output_off(source)                    # channel quiet until the first pulse
        voltmeter = connect_voltmeter(plan.volt_cfg)

        if plan.uses_magnet:
            on_status("Connecting to Kepco magnet + Lake Shore 475 …")
            magnet = connect_magnet(plan.magnet_cfg)
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        for series_idx, init_A in enumerate(plan.series_values):
            if stop_event.is_set():
                break

            init_info: dict = {}
            if init_A is not None:
                on_status(f"Initializing: magnet → {init_A:g} A, then "
                          f"{plan.sweep_magnet_current_A:g} A …")
                init_info = initialize_with_field(
                    magnet, plan.magnet_cfg, gaussmeter, plan.gauss_cfg, init_A,
                    plan.sweep_magnet_current_A, plan.field_settle_tolerance_mT, stop_event)
                if stop_event.is_set():
                    break

            label = f"I_init={init_A:g}A" if len(plan.init_currents_A) > 1 else None
            # A fresh RunContext (own run number, own file) EVERY iteration.
            ctx = allocate_run(
                plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                temperature_setpoint_K=plan.temperature_setpoint_K,
                key_axis=("current_A", init_A) if init_A is not None else None,
                series=plan.series)
            extra = _run_extra(plan, init_A, init_info)
            run_contexts.append(ctx)
            run_extras.append(extra)
            on_run_label(f"Run #{ctx.run_str}")
            write_csv = make_incremental_writer(
                ctx.raw_path,
                lambda records, _ctx=ctx, _x=extra: build_header_fields(
                    plan, _ctx, records, status="in_progress", comment="", extra=_x))

            iter_records: list[dict] = []

            def tagged_on_point(record: dict, _idx=series_idx, _label=label, _init=init_A,
                                _iter=iter_records) -> None:
                record["series_index"] = _idx
                record["series_label"] = _label
                record["init_magnet_current_A"] = _init
                _iter.append(record)
                on_point(record)

            on_status("Running the switching sweep …" if label is None
                      else f"Running the switching sweep ({label}) …")
            iter_error: Optional[BaseException] = None
            try:
                run_measurement(
                    source, voltmeter, plan.pulse_cfg, plan.read_cfg, points,
                    stop_event=stop_event, on_point=tagged_on_point,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                    magnet_current_A=plan.sweep_magnet_current_A if init_A is not None else None,
                    write_csv=write_csv, output_file=str(ctx.raw_path))
            except Exception as exc:
                iter_error = exc

            # Finalize THIS run's header/index row right now — never gated on
            # the end-of-session status/comment prompt.
            iter_status = "error" if iter_error is not None \
                else ("aborted" if stop_event.is_set() else "completed")
            header_fields = build_header_fields(
                plan, ctx, iter_records, status=iter_status, comment="", extra=extra)
            write_record(ctx.raw_path, iter_records, header_fields)
            finalize_index_row(plan.data_root, ctx.sample, ctx.run_number, header_fields)
            if on_run_finished is not None:
                try:
                    on_run_finished(ctx, iter_records)
                except Exception:
                    log.exception("Could not save the per-run plot for run %s", ctx.run_str)
            if iter_error is not None:
                raise iter_error
    finally:
        if source is not None:
            safe_shutdown("6221", lambda: shutdown_source(source))
        if magnet is not None:
            safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
        if gaussmeter is not None:
            safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


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
        self._run_extras: list[dict] = []     # per-run header extras, parallel to _run_contexts
        # The LAST run's PNG, stashed by _save_run_png so _on_status_comment
        # can re-save it in place once the operator's comment is known.
        self._png_path: Optional[Path] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Starting …", id="status_line")
        with Horizontal(id="progress_row"):
            yield Static("", id="run_label")
            yield ProgressBar(id="progress", total=progress_total(self.plan.run_cost, self.plan.total_points),
                              show_eta=True)
        yield DataTable(id="results_table", zebra_stripes=True, cursor_type="row")
        yield RichLog(id="log", max_lines=5000, markup=False, wrap=True)
        with Horizontal(id="runactionbar"):
            yield Button("Abort (safe shutdown)", id="abort_btn", variant="error")
            yield Button("Back", id="back_btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#results_table", DataTable).add_columns(
            "pt #", "I_init (A)", "I_pulse (A)", "R_NL (mΩ)", "ΔR (mΩ)", "switched",
            "V_even (µV)", "T1 (K)")
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
            self._plot_queue, self._plot_process = start_live_plot(
                _live_plot_worker, self.plan.read_cfg.reversal_enabled)
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
        init = record.get("init_magnet_current_A")
        dr = record.get("delta_R_ohm")
        sw = record.get("switched")
        table.add_row(
            str(record["point_index"]),
            f"{init:g}" if init is not None else "—",
            f"{record['pulse_current_A']:.4g}",
            f"{record['nl_resistance_ohm'] * 1e3:.4f}",
            f"{dr * 1e3:+.4f}" if dr is not None else "—",
            "—" if sw is None else ("YES" if sw else "no"),
            f"{record['voltage_even_V'] * 1e6:.3f}" if record.get("voltage_even_V") is not None else "—",
            f"{t1:.3f}" if t1 is not None else "—",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(
            progress_step(self.plan.run_cost, len(self._records) - 1))
        self._set_status(f"Point {len(self._records)} / {self.plan.total_points}.")

    def _save_run_png(self, ctx: RunContext, iter_records: list[dict]) -> None:
        """One PNG per run (own run number), as if each initial state had been
        started by hand -- no combined overlay."""
        try:
            png_path = proc_path(self.plan.data_root, ctx.sample, ctx.run_str, ctx.device,
                                 MEASUREMENT_TYPE, "NL_vs_pulse")
            self._png_path = png_path
            _save_measurement_png(iter_records, png_path)
        except Exception:
            log.exception("Could not save plot PNG")

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False
        self.query_one("#abort_btn", Button).disabled = True
        self.app.push_screen(StatusCommentScreen(), self._on_status_comment)

    def _on_status_comment(self, result: Optional[tuple[str, str]]) -> None:
        # With several initial states the runs before the last were implicitly
        # "skipped" -- left at the outcome status do_run() wrote right after
        # each one, with no comment. Only the last run, the one the operator
        # is looking at, gets the status/comment they entered.
        if result is None or not self._run_contexts:
            return
        status, comment = result
        series_idx = len(self._run_contexts) - 1
        ctx = self._run_contexts[series_idx]
        iter_records = [r for r in self._records if r.get("series_index", 0) == series_idx]
        header_fields = build_header_fields(
            self.plan, ctx, iter_records, status=status, comment=comment,
            extra=self._run_extras[series_idx])
        try:
            if iter_records or not ctx.raw_path.exists():
                write_record(ctx.raw_path, iter_records, header_fields)
            finalize_index_row(self.plan.data_root, ctx.sample, ctx.run_number, header_fields)
        except Exception:
            log.exception("Could not save final status/comment for run %d", ctx.run_number)

        if comment and self._png_path is not None:
            try:
                _save_measurement_png(iter_records, self._png_path, comment=comment)
            except Exception:
                log.exception("Could not re-save measurement plot PNG with comment")

    def action_abort(self) -> None:
        if self._measurement_running and not self._stop_event.is_set():
            self._stop_event.set()
            self._set_status("Abort requested — finishing this point, then shutting the "
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
        try:
            run_plan(self.plan, self._stop_event,
                     on_status=self._set_status_threadsafe,
                     on_run_label=self._set_run_label_threadsafe,
                     on_point=lambda record: self.app.call_from_thread(self._on_point, record),
                     on_run_finished=self._save_run_png,
                     run_contexts=self._run_contexts, run_extras=self._run_extras)
            final = "Measurement aborted." if self._stop_event.is_set() else "Measurement complete."
        except Exception as exc:
            log.exception("Measurement failed")
            final = f"ERROR: {exc}"
        finally:
            self.app.call_from_thread(self._on_finished, final)


# ── app / form ─────────────────────────────────────────────────────────────

class NonlocalSwitchingApp(App):
    TITLE = "Nonlocal spin-current switching"
    SUB_TITLE = "6221 single-lobe pulse · DC nonlocal read (2182A) · field-initialized"

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
                        "Write pulse (6221 WAVE, one lobe 0 → ±I → 0)",
                        field("pulse_current_start_A", "Pulse current start (A)",
                              DEFAULTS["pulse_current_start_A"],
                              hint="Signed: negative = a −I lobe (never a ± pair)."),
                        field("pulse_current_stop_A", "Pulse current stop (A)",
                              DEFAULTS["pulse_current_stop_A"]),
                        field("pulse_current_step_A", "Pulse current step (A)",
                              DEFAULTS["pulse_current_step_A"],
                              validators=[Number(minimum=1e-12, failure_description="must be > 0")],
                              hint="One pulse per step. A sweep through 0 gets one read-only "
                                   "0 A point."),
                        switch_field("amplitude_bidirectional",
                                     "Then sweep back (loop / no-reset control)",
                                     DEFAULTS["amplitude_bidirectional"]),
                        field("pulse_width_s", "Requested pulse width (s)",
                              DEFAULTS["pulse_width_s"],
                              hint="Kimura/Otani: ~1 ms. Actual width is measured and logged as "
                                   "pulse_width_measured_s."),
                        field("pulse_compliance_V", "Pulse voltage compliance (V)",
                              DEFAULTS["pulse_compliance_V"],
                              hint="Above I_max × R_injector, or the pulse clips silently."),
                    )
                    yield card(
                        "Nonlocal read (6221 DC ± / 2182A)",
                        field("delay_after_pulse_s", "Delay after pulse (s)",
                              DEFAULTS["delay_after_pulse_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")],
                              hint="Wait between pulse end and the read."),
                        field("sense_current_A", "Sense current (A)", DEFAULTS["sense_current_A"],
                              hint="Kimura/Otani: 100 µA. Keep well below the switching current. "
                                   "Signed: with reversal off the sign is the fixed read polarity."),
                        field("compliance_V", "Read compliance (V)", DEFAULTS["compliance_V"]),
                        switch_field("reversal_enabled", "Reverse the sense current each read (+I/−I)",
                                     DEFAULTS["reversal_enabled"]),
                        Label("Off = one fixed polarity, plain average: the read's own spin current "
                              "never alternates in sign. Thermal-EMF offsets are then NOT cancelled "
                              "and V_even is not recorded.", classes="hint"),
                        field("n_averages", "Averages per read",
                              DEFAULTS["n_averages"], kind="integer",
                              hint="± pairs with reversal, plain samples without.",
                              validators=[Number(minimum=2, failure_description="must be ≥ 2")]),
                        field("source_delay_s", "Settle after polarity flip (s)",
                              DEFAULTS["source_delay_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        field("nplc", "2182A integration (NPLC)", DEFAULTS["nplc"]),
                        field("switch_sigma", "'switched' threshold (σ)", DEFAULTS["switch_sigma"],
                              hint="|ΔR| vs the previous row, in combined standard errors."),
                    )
                    yield card(
                        "Field initialization (Kepco magnet)",
                        field("init_magnet_currents", "Init magnet current(s) (A)",
                              DEFAULTS["init_magnet_currents"], kind="text",
                              hint="Blank = magnet untouched (initialize externally). One value, or "
                                   "comma-separated — each is its own initial state, sweep and "
                                   "file. With one pulse polarity use both signs, e.g. 5, -5."),
                        field("sweep_magnet_current_A", "Magnet current during the sweep (A)",
                              DEFAULTS["sweep_magnet_current_A"],
                              hint="0 = field off after initializing; the sweep starts from the "
                                   "remanent state."),
                        field("field_settle_tolerance_mT", "Field settle tolerance (mT)",
                              DEFAULTS["field_settle_tolerance_mT"]),
                    )
                    yield card(
                        "Reference levels (optional)",
                        field("R_P_ohm", "R_NL at the P level (Ω)", DEFAULTS["R_P_ohm"],
                              valid_empty=True,
                              hint="From a dc_spin_valve field sweep of the same nonlocal signal."),
                        field("R_AP_ohm", "R_NL at the AP level (Ω)", DEFAULTS["R_AP_ohm"],
                              valid_empty=True,
                              hint="Gives state_AP_fraction per row; 'switched' then also needs "
                                   "≥ half the swing."),
                    )
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature", "Log temperature (MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                    )

                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Keithley 6221 + 2182A",
                            field("source_visa_resource", "6221 VISA resource",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            field("voltmeter_visa_resource", "2182A VISA resource",
                                  DEFAULTS["voltmeter_visa_resource"], kind="text"),
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
                yield Static(NLSW_DESCRIPTION, classes="card-desc")
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

        return resolve_state(state), errors

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
        return build_plan(state, self.data_root)


def main() -> None:
    NonlocalSwitchingApp().run()


if __name__ == "__main__":
    main()
