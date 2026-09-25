#!/usr/bin/env python3
"""
Keithley-6221 AC source of the dual-harmonic program  (type HARM6)
==================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-14 (a mode of mfli_dual_harmonic_tui.py since 2026-09-23)

The "AC current source = Keithley 6221" engine behind the dual-harmonic form
in mfli_dual_harmonic_tui.py: its parameter surface, summary, plan builder,
run loop, header, PNG and RunScreen. The form there dispatches here when the
toggle is on the 6221 (see
mfli_dual_harmonic_6221.py's module docstring for why BOTH MFLIs must have
their Aux In 1 wired to the 6221's phase marker, not just the leader's).

Run with:
    python mfli_dual_harmonic_6221_tui.py

Requirements:
    pip install textual matplotlib  (in addition to mfli_dual_harmonic_6221.py's own deps)
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import textwrap
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np


from dc.dc_sweep_utils import build_segmented_sweep, parse_sweep_rows, safe_shutdown, try_parse
from mfli.mfli_dual_harmonic_6221 import (
    ACSourceConfig,
    AcquisitionConfig,
    DemodConfig,
    ExtRefConfig,
    FilterConfig,
    GaussmeterConfig,
    MagnetConfig,
    MeasurementPoint,
    SampleGeometryConfig,
    TemperatureControllerConfig,
    acquire_averaged,
    auto_null_phase,
    configure_demodulator,
    configure_external_reference,
    connect,
    connect_ac_source,
    connect_device,
    connect_gaussmeter,
    connect_magnet,
    connect_temperature_controller,
    disable_sigout,
    extref_lock_s,
    null_follower_reference_via_1f,
    run_measurement,
    set_magnet_current,
    setup_mds,
    shutdown_ac_source,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_temperature_controller,
    wait_for_reference_lock,
    _check_ac_safety,
    _AC_CURRENT_CEILING_A,
    _AC_COMPLIANCE_CEILING_V,
)
from mfli.mfli_dual_harmonic import phase_cal_s
from instruments.data_dir import validate_directory
from instruments.field_geometry import field_direction_summary_line
from instruments.data_naming import (
    RunContext,
    allocate_run,
    record_run,
    preview_raw_filename,
)
from instruments.kepco_magnet import magnet_move_s
from instruments.keithley6221 import ac_source_restart_s
from instruments.lakeshore475 import read_field_s
from instruments.mfli_daq import acquire_s, poll_window_s
from instruments.run_time import (
    GPIB_TXN_S, MDS_SYNC_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
    RunCost,
)
from instruments.tui_common import (
    MeasurementRunScreen,
    format_si,
    parse_sensor_uids,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
)

log = logging.getLogger("mfli_dual_harmonic_6221_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "mfli_dual_harmonic_6221_tui_settings.json"

# Locked type code for this measurement (see instruments/data_naming.py) —
# never deviates.
MEASUREMENT_TYPE = "HARM6"



# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors mfli_dual_harmonic_6221.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "leader_device": "dev7885",
    "follower_device": "dev7886",
    "daq_host": "localhost",
    "daq_port": "8004",
    "ac_visa_resource": "GPIB0::20::INSTR",
    "frequency_Hz": "317.3",
    "amplitude_values": "1e-7",
    "ac_compliance_V": "2.0",
    "phasemarker_line": "1",
    "leader_harmonic": "1",
    "leader_measure_rxx": False,
    "follower_harmonic": "2",
    "measure_rxx": False,
    "time_constant_1f_s": "0.3",
    "order_1f": "4",
    "sinc_filter_1f": True,
    "time_constant_2f_s": "0.3",
    "order_2f": "4",
    "sinc_filter_2f": True,
    "differential": True,
    "ac_coupling": True,
    "input_range_1f_V": "1.0",
    "input_range_2f_V": "1.0",
    "sample_rate_Hz": "857.0",
    "settling_time_s": "15",
    "field_settle_tolerance_mT": "0.02",
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
    "sweep_rows": "-20, 20, 21",
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
    "field_theta_deg": "",
    "field_phi_deg": "",
    "leader_extref_index": "0",
    "leader_aux_input_ch": "0",
    "leader_osc_index": "0",
    "leader_pll_demod_index": "1",
    "leader_automode": "4",
    "follower_extref_index": "0",
    "follower_aux_input_ch": "0",
    "follower_osc_index": "0",
    "follower_pll_demod_index": "1",
    "follower_automode": "4",
    "extref_lock_timeout_s": "5.0",
}

# extrefs/N/automode options — see ExtRefConfig.automode's docstring in
# mfli_dual_harmonic_6221.py for the full rationale.
AUTOMODE_OPTIONS: list[tuple[str, int]] = [
    ("2 — low bandwidth", 2),
    ("3 — high bandwidth", 3),
    ("4 — dynamic (auto)", 4),
]
AUTOMODE_HINT = "2 = most forgiving, 3 = fastest tracking, 4 = auto (default)."

# id -> caster, for every free-text numeric field (Select/Switch handled separately)
NUMERIC_FIELDS: dict = {
    "daq_port": int,
    "frequency_Hz": float,
    "ac_compliance_V": float,
    "phasemarker_line": int,
    "time_constant_1f_s": float,
    "time_constant_2f_s": float,
    "input_range_1f_V": float,
    "input_range_2f_V": float,
    "sample_rate_Hz": float,
    "settling_time_s": float,
    "field_settle_tolerance_mT": float,
    "n_averages": int,
    "current_limit_A": float,
    "voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
    "phase_cal_n_averages": int,
    "phase_cal_max_iterations": int,
    "leader_extref_index": int,
    "leader_aux_input_ch": int,
    "leader_osc_index": int,
    "leader_pll_demod_index": int,
    "follower_extref_index": int,
    "follower_aux_input_ch": int,
    "follower_osc_index": int,
    "follower_pll_demod_index": int,
    "extref_lock_timeout_s": float,
}
TEXT_FIELDS = ["leader_device", "follower_device", "daq_host", "ac_visa_resource",
               "amplitude_values", "device", "cooldown",
               "visa_resource", "gaussmeter_visa_resource", "temperature_visa_resource",
               "temperature_sensor_uids", "data_dir"]
# Free-text, blank-allowed: parsed to Optional[float] by hand in parse_state()
# rather than going through NUMERIC_FIELDS' "blank is an error" casting.
OPTIONAL_NUMERIC_FIELDS = [
    "temperature_setpoint_K",
    "phase_cal_current_A",
    "hall_bar_length_um", "hall_bar_width_um", "hall_bar_thickness_nm",
    "field_theta_deg", "field_phi_deg",
]
MAGNET_FIELD_IDS = [
    "visa_resource", "current_limit_A", "voltage_compliance_V",
    "ramp_step_A", "ramp_delay_s",
    "gaussmeter_visa_resource", "gaussmeter_n_averages", "gaussmeter_read_delay_s",
    "field_settle_tolerance_mT",
]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]


# The harmonic each MFLI's demod locks to (the "Leader/Follower MFLI lock-in"
# cards, shared with the MFLI-source form).
HARMONIC_OPTIONS: list[tuple[str, int]] = [(f"{h}f", h) for h in range(1, 9)]


def demod_naming(harmonic: int, rxx: bool) -> tuple[str, str]:
    """One MFLI demod's column prefix and display label: "<h>f", or
    "rxx_<h>f" with its R_xx toggle on (a pure naming switch — the harmonic
    is chosen separately). Leader 1f / follower 2f, and a follower at 1f with
    R_xx on, give exactly the names these programs always saved (1f_*, 2f_*,
    rxx_1f_*) — see mfli_dual_harmonic_6221.py's module docstring for why
    R_xx and R_xy's 2f can't both be live with just two physical MFLIs."""
    return (f"rxx_{harmonic}f", f"R_xx ({harmonic}f)") if rxx else (f"{harmonic}f", f"{harmonic}f")


def state_naming(state: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    """(leader, follower) demod_naming() of a form state."""
    return (demod_naming(state["leader_harmonic"], state["leader_measure_rxx"]),
            demod_naming(state["follower_harmonic"], state["measure_rxx"]))


def plan_naming(plan) -> tuple[tuple[str, str], tuple[str, str]]:
    """(leader, follower) demod_naming() of either source's MeasurementPlan."""
    return (demod_naming(plan.demod1_cfg.harmonic, plan.leader_measure_rxx),
            demod_naming(plan.demod2_cfg.harmonic, plan.measure_rxx))


def migrate_settings(saved: dict) -> dict:
    """A form saved before the harmonic selects existed: its R_xx mode meant
    the follower at 1f — keep that, rather than silently becoming rxx_2f."""
    if saved.get("measure_rxx") and "follower_harmonic" not in saved:
        saved["follower_harmonic"] = 1
    return saved


def harmonic_checks(state: dict, warnings: list[str], errors: list[str]) -> None:
    """Summary checks both AC sources share: the two MFLIs must save to
    different column prefixes, neither demod frequency may sit on a mains
    harmonic, and the phase-cal null is only physical with the leader at 1f."""
    (lead, lead_disp), (fol, fol_disp) = state_naming(state)
    if lead == fol:
        errors.append(f"Leader and follower would both save as {lead}_* — pick different "
                      "harmonics or turn R_xx on for one of them.")
    f = state["frequency_Hz"]
    for label, h in ((lead_disp, state["leader_harmonic"]), (fol_disp, state["follower_harmonic"])):
        check_f = h * f
        for mains in (50, 60):
            nearest = round(check_f / mains) * mains
            if nearest > 0 and abs(check_f - nearest) < 0.5:
                warnings.append(
                    f"{label} ({check_f:g} Hz) is within 0.5 Hz of a {mains} Hz "
                    f"harmonic ({nearest} Hz) — mains pickup risk."
                )
    if state["enable_phase_cal"] and state["leader_harmonic"] != 1:
        warnings.append(
            f"Phase cal nulls the leader's Y at its own harmonic ({lead_disp}) — "
            "that is only pure instrumental delay at 1f (resistive PHE/AHE). "
            "Check it is the calibration you want before trusting the result."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def run_costs(state: dict, currents_A=None) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order --
    amplitude-major: one full field sweep (`currents_A`, or a single point at
    the present field if None) per excitation current, each its own file.
    Also drives the run screen's progress bar, so the estimate and the live
    ETA cannot disagree. Every term mirrors a step of run_measurement() /
    RunScreen.do_run() -- see mfli_dual_harmonic_6221.py."""
    n_pts = len(currents_A) if currents_A is not None else 1
    n_amps = max(1, len(state.get("amplitude_list", [])))
    rc = RunCost(n_pts * n_amps)
    rate, n_avg = state["sample_rate_Hz"], state["n_averages"]
    tc1, tc2 = state["time_constant_1f_s"], state["time_constant_2f_s"]
    # acquire_averaged_pair(): leader and follower share ONE poll window -- the longer of the two.
    pair_s = max(acquire_s(tc1, n_avg, rate), acquire_s(tc2, n_avg, rate)) if rate > 0 else 0.0
    has_temp = bool(state["enable_temperature"] and parse_sensor_uids(state["temperature_sensor_uids"]))
    rc.each("settle", state["settling_time_s"])
    rc.each("acquire", pair_s)
    # per point: MDS check + 2 ExtRef lock checks + 3 LabOne reads in build_run_metadata
    # (frequency + 2 phase nodes) + CSV rewrite + temperature
    rc.each("overhead", 6 * GPIB_TXN_S + POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    rc.at("connect + MDS", PER_RUN_S + MDS_SYNC_S, 0)
    lock_typ, lock_worst = extref_lock_s(state["extref_lock_timeout_s"])
    magnet_cfg = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    if currents_A is not None:
        rc.each("field read", read_field_s(GaussmeterConfig(
            n_averages=state["gaussmeter_n_averages"], read_delay_s=state["gaussmeter_read_delay_s"])))
    phase_cal = (phase_cal_s(tc1, tc2, state["phase_cal_n_averages"],
                             state["phase_cal_max_iterations"], rate) if rate > 0 else 0.0)
    i_now = 0.0                                   # the magnet starts at 0 A
    for a in range(n_amps):
        first = a * n_pts
        # every amplitude re-arms the 6221 (waveform_arm) and re-locks both ExtRef PLLs, then saves its own file
        rc.at("6221 re-arm + ExtRef", ac_source_restart_s() + lock_typ, first, worst_extra=lock_worst - lock_typ)
        rc.at("per-file", PER_FILE_S, first)
        if state["enable_phase_cal"]:
            rc.at("phase cal", phase_cal, first)
            if currents_A is not None and state["phase_cal_current_A"] is not None:
                typ, worst = magnet_move_s(abs(state["phase_cal_current_A"] - i_now), magnet_cfg)
                rc.at("phase cal", typ + state["settling_time_s"], first, worst_extra=worst - typ)
                i_now = state["phase_cal_current_A"]
        if currents_A is not None:
            for j, current in enumerate(currents_A):
                typ, worst = magnet_move_s(abs(current - i_now), magnet_cfg)
                rc.at("magnet", typ, first + j, worst_extra=worst - typ)
                i_now = current
    if currents_A is not None:
        rc.tail("ramp-down", magnet_move_s(abs(i_now), magnet_cfg, with_field=False)[0])
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    daq_host: str
    daq_port: int
    leader: str
    follower: str
    ac_cfg: ACSourceConfig
    amplitudes_A: List[float]
    measure_rxx: bool                     # follower's R_xx naming toggle
    leader_measure_rxx: bool
    leader_extref_cfg: ExtRefConfig
    follower_extref_cfg: ExtRefConfig
    extref_lock_timeout_s: float
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
    sample: str
    device: str
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    series: str = ""
    data_root: Path = _DEFAULT_DATA_DIR
    run_cost: Optional[RunCost] = None      # modelled seconds per point (progress bar + ETA)

    @property
    def total_points(self) -> int:
        """Per-file point count (the inner field sweep) -- unrelated to how
        many amplitude values are in play."""
        return len(self.currents_A) if self.currents_A is not None else 1

    @property
    def total_files(self) -> int:
        return len(self.amplitudes_A)


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                         status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """Same shape/purpose as mfli_dual_harmonic_tui.py's build_header_fields() --
    `ctx` is THIS iteration's RunContext (one per amplitude value), not a
    single plan-wide one; `extra` carries this iteration's own amplitude on
    top of the plan-wide header_extra."""
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
    if extra:
        fields.update(extra)
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# Small widget-building helpers (keep compose() readable) — identical to
# mfli_dual_harmonic_tui.py's; see field()'s docstring for why fields must
# stay flat (a doubly-nested Vertical breaks Textual's grid auto-row sizing).
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Live validation / derived-value summary
# ─────────────────────────────────────────────────────────────────────────────

def resolve_state(state: dict) -> dict:
    """Add the derived keys build_summary() / build_plan() read — the parsed
    lists/sweeps, each with its parse error — to a state of raw field values.
    Pure: shared by the TUI's and the web page's parse_state()."""
    state["sweep_rows_parsed"], state["sweep_rows_parse_error"] = try_parse(state["sweep_rows"], parse_sweep_rows)
    state["amplitude_list"], state["amplitude_parse_error"] = try_parse(state["amplitude_values"])
    return state


def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (info, warnings, errors) for a fully-parsed state dict."""
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
        errors.append("Device is required (e.g. HB3, SV2).")

    if state["leader_device"] == state["follower_device"]:
        errors.append("Leader and follower device IDs must be different.")

    (_, leader_display), (_, follower_display) = state_naming(state)
    info.append(f"Lock-in: leader {leader_display} · follower {follower_display}")

    if state.get("amplitude_parse_error"):
        errors.append(f"Excitation current list: {state['amplitude_parse_error']}")
    else:
        amp_list = state.get("amplitude_list", [])
        over_limit = [i for i in amp_list if not 0 < i <= _AC_CURRENT_CEILING_A]
        if over_limit:
            errors.append(
                f"Excitation current(s) {over_limit} must be in "
                f"(0, {format_si(_AC_CURRENT_CEILING_A, 'A')}] — check for a mistyped exponent."
            )
        elif len(amp_list) > 1:
            info.append(f"Excitation currents: {', '.join(format_si(i, 'A') for i in amp_list)} "
                        f"peak — {len(amp_list)} sweeps, one file set each")
        elif amp_list:
            info.append(f"Excitation current: {format_si(amp_list[0], 'A')} peak")
    if not 0 < state["ac_compliance_V"] <= _AC_COMPLIANCE_CEILING_V:
        errors.append(
            f"6221 compliance must be in (0, {_AC_COMPLIANCE_CEILING_V:g}] V; "
            f"got {state['ac_compliance_V']:g} V."
        )

    harmonic_checks(state, warnings, errors)

    info.append(f"Phase marker: Trigger Link {state['phasemarker_line']} → Aux In "
                f"{state['leader_aux_input_ch'] + 1} on BOTH MFLIs — equal cables; an MDS-only "
                "follower silently loses 2f")

    # The real 1f/2f signal demod is fixed at index 0 (see _build_plan below)
    # — the PLL detector must be a different demod (extrefs/N/adcselect is
    # read-only on real firmware; see ExtRefConfig's docstring).
    if state["leader_pll_demod_index"] == 0:
        errors.append(f"Leader PLL phase-detector demod index must differ from 0 "
                       f"(demod 0 reads the real {leader_display} signal).")
    if state["follower_pll_demod_index"] == 0:
        errors.append(f"Follower PLL phase-detector demod index must differ from 0 "
                       f"(demod 0 reads the real {follower_display} signal).")

    acq_window_s = {"leader": 0.0, "follower": 0.0}
    for key, label, tc_key, order_key in (
        ("leader", leader_display, "time_constant_1f_s", "order_1f"),
        ("follower", follower_display, "time_constant_2f_s", "order_2f"),
    ):
        tc = state[tc_key]
        if tc > 0:
            order = state[order_key]
            settle_multiple = 10 if order >= 3 else 5
            recommended_settle = settle_multiple * tc
            if state["settling_time_s"] < recommended_settle:
                warnings.append(
                    f"{label} settling time {state['settling_time_s']:g} s < {settle_multiple}×TC "
                    f"({recommended_settle:g} s, order {order}) — filter may not have settled."
                )
            else:
                info.append(f"{label} settling: ✓ — ≥ {settle_multiple}×TC ({recommended_settle:g} s)")

            bw = 1.0 / (2 * math.pi * tc)
            min_rate = 4 * bw
            info.append(f"{label} noise bandwidth: ≈ {bw:.3g} Hz")
            if state["sample_rate_Hz"] < min_rate:
                warnings.append(
                    f"Sample rate {state['sample_rate_Hz']:g} Sa/s may be low for {label} TC "
                    f"(want ≳ {min_rate:.1f} Sa/s)."
                )

            acq_window_s[key] = (poll_window_s(tc, state["n_averages"], state["sample_rate_Hz"])
                                 if state["sample_rate_Hz"] > 0 else 0.0)
            n_indep = acq_window_s[key] / (math.pi * tc)
            if n_indep < 0.5 * state["n_averages"]:
                warnings.append(
                    f"{label} averaging window ≈ {acq_window_s[key]:g} s holds only "
                    f"~{max(1, round(n_indep))} independent filter outputs at TC={tc:g} s "
                    f"— far fewer than the {state['n_averages']} samples requested, so "
                    f"per-point noise averages down much less than √n and the reported "
                    f"R_sem understates it. Use a shorter time constant, or raise the "
                    f"sample count into the thousands."
                )
        else:
            errors.append(f"{label} time constant must be > 0 s.")

    total_points = 0
    resolved = None
    if state["enable_sweep"]:
        if state.get("sweep_rows_parse_error"):
            errors.append(f"Sweep rows: {state['sweep_rows_parse_error']}")
        else:
            rows = state.get("sweep_rows_parsed", [])
            max_abs_I = max((max(abs(s), abs(e)) for s, e, _ in rows), default=0.0)
            if max_abs_I > state["current_limit_A"]:
                errors.append(
                    f"Sweep range (±{max_abs_I:g} A) exceeds the current limit "
                    f"({state['current_limit_A']:g} A)."
                )
            for s, e, n in rows:
                if s == e and n > 1:
                    warnings.append(f"Row ({s:g}, {e:g}, {n}) repeats a single point {n} times.")
            resolved = build_segmented_sweep(rows, bidirectional=True)
            total_points = len(resolved)
            n_raw = sum(n for _, _, n in rows)
            n_merged = 2 * n_raw - total_points
            merged_note = f", {n_merged} shared boundary point(s) merged" if n_merged else ""
            info.append(f"Field sweep: {total_points} points — {len(rows)} row(s), bidirectional"
                         f"{merged_note}")
        info.append(f"Field read: Lake Shore 475 — {state['gaussmeter_visa_resource']}")
        tol_mT = state["field_settle_tolerance_mT"]
        if tol_mT <= 0:
            warnings.append("Field-settle tolerance is 0 — every magnet step will wait the "
                             "full settle timeout before acquiring.")
        elif tol_mT < 0.01:
            warnings.append(f"Field-settle tolerance {tol_mT:g} mT is below the 475's typical "
                             "reading noise — points may stall until the settle timeout.")
        if resolved is not None:
            info.extend(run_costs(state, resolved).lines())
    else:
        info.append("Field: none — single point, magnet untouched")
        info.extend(run_costs(state).lines())

    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if not uids:
            warnings.append("Temperature logging is on but no sensor UID is set — "
                             "temperature columns will be empty.")
        else:
            info.append(f"Temperature: MercuryiTC {', '.join(uids)} — empty if unreachable")
    else:
        info.append("Temperature: off")

    if state["enable_phase_cal"]:
        if state["measure_rxx"]:
            warnings.append(
                "Phase cal nulls the follower's phase against the leader's 1f "
                "reference — with R_xx on, that follower demod reads R_xx, "
                "not R_xy. Whether this is still the calibration you "
                "want for R_xx is a physics call this form doesn't make for "
                "you — check before trusting the result."
            )
        if state["phase_cal_current_A"] is not None:
            if not state["enable_sweep"]:
                warnings.append(
                    "Phase-cal current is set but field sweep is disabled — "
                    "it will be ignored; calibration runs at the present field."
                )
            else:
                rows = state.get("sweep_rows_parsed", [])
                max_abs_I = max((max(abs(s), abs(e)) for s, e, _ in rows), default=0.0)
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
                info.append(f"Phase cal: at {state['phase_cal_current_A']:g} A — null "
                            f"{leader_display} Y, then sweep")
        else:
            info.append(f"Phase cal: at present field — null {leader_display} Y")

    geom_fields = {
        "Hall bar length": state["hall_bar_length_um"],
        "Hall bar width": state["hall_bar_width_um"],
        "Hall bar thickness": state["hall_bar_thickness_nm"],
    }
    set_geom = {k: v for k, v in geom_fields.items() if v is not None}
    if not set_geom:
        warnings.append(
            "Sample geometry is unset — the run will still record raw 1f/2f "
            "voltages, but converting them to a resistivity or an absolute "
            "spin-Hall/damping-like field needs these (optional fields below)."
        )
    elif len(set_geom) < len(geom_fields):
        missing = ", ".join(k for k in geom_fields if geom_fields[k] is None)
        warnings.append(f"Sample geometry partially set — still missing: {missing}.")
        info.append("Sample geometry: " + ", ".join(f"{k}={v:g}" for k, v in set_geom.items()))
    else:
        info.append("Sample geometry: " + ", ".join(f"{k}={v:g}" for k, v in set_geom.items()))

    info.append(field_direction_summary_line(state["field_theta_deg"], state.get("field_phi_deg")))

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
    n_amps = len(state.get("amplitude_list", []))
    suffix = f" (one file per amplitude — {n_amps} files)" if n_amps > 1 else ""
    return f"{preview}_<timestamp>.csv{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────
# Same rationale as mfli_dual_harmonic_tui.py's — a GUI matplotlib backend
# and Textual's terminal control both want the main thread.

def _live_plot_worker(queue: "mp.Queue", has_field_sweep: bool,
                       naming: tuple = (("1f", "1f"), ("2f", "2f")),
                       multi: bool = False) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    try:
        fig.canvas.manager.set_window_title("MFLI live measurement (6221 AC source)")
    except Exception:
        pass
    (leader_prefix, leader_display), (follower_prefix, follower_display) = naming
    ax1.set_ylabel(f"{leader_display}  R (V)")
    ax2.set_ylabel(f"{follower_display}  R (V)")
    ax2.set_xlabel("Magnetic field (mT)" if has_field_sweep else "Point #")
    ax1.set_title("Live measurement")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    cmap = plt.get_cmap("tab10")
    lines1: dict[int, "plt.Line2D"] = {}
    lines2: dict[int, "plt.Line2D"] = {}
    series_data: dict[int, tuple[list, list, list]] = {}

    def _drain(_frame=None):
        updated: set[int] = set()
        new_series = False
        while True:
            try:
                record = queue.get_nowait()
            except Exception:
                break
            idx = record.get("series_index", 0)
            if idx not in lines1:
                label = record.get("series_label")
                # One current: leader and follower keep their own colors, as in
                # a manual run. Several: color by current so the overlaid
                # runs can be told apart.
                c1, c2 = (cmap(idx % 10),) * 2 if multi else ("tab:blue", "tab:orange")
                (l1,) = ax1.plot([], [], "o-", color=c1, label=label)
                (l2,) = ax2.plot([], [], "o-", color=c2, label=label)
                lines1[idx] = l1
                lines2[idx] = l2
                series_data[idx] = ([], [], [])
                new_series = True
            xs, r1s, r2s = series_data[idx]
            x = record.get("magnet_field_mT") if has_field_sweep else None
            xs.append(x if x is not None else record["point_index"])
            r1s.append(record[f"{leader_prefix}_R_V"])
            r2s.append(record[f"{follower_prefix}_R_V"])
            updated.add(idx)
        if updated:
            for idx in updated:
                xs, r1s, r2s = series_data[idx]
                lines1[idx].set_data(xs, r1s)
                lines2[idx].set_data(xs, r2s)
            if new_series and any(l.get_label() and not l.get_label().startswith("_")
                                   for l in lines1.values()):
                ax1.legend(loc="best", fontsize=8)
            for ax in (ax1, ax2):
                ax.relim()
                ax.autoscale_view()
        return tuple(lines1.values()) + tuple(lines2.values())

    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path,
                           plan: Optional["MeasurementPlan"] = None, comment: str = "") -> None:
    """Save a static 1f/2f R-vs-field PNG to proc/, from whatever points
    were actually collected (including an aborted/partial run).

    `records` is ONE run's points -- with several excitation currents each
    run is saved (and plotted) on its own, exactly like a manual run.

    `plan`/`comment` add a small "at a glance" text annotation (field
    direction, the AC excitation current, the 1f/2f filter TC/order, the
    operator's comment) for context not already in the filename. Called
    once when the run ends (comment="") and again, to overwrite the PNG
    in place, once the
    operator's comment is known."""
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (leader_prefix, leader_display), (follower_prefix, follower_display) = (
        plan_naming(plan) if plan else (("1f", "1f"), ("2f", "2f")))
    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    ax1.plot(xs, [r[f"{leader_prefix}_R_V"] for r in records], "o-", color="tab:blue")
    ax2.plot(xs, [r[f"{follower_prefix}_R_V"] for r in records], "o-", color="tab:orange")
    ax1.set_ylabel(f"{leader_display}  R (V)")
    ax2.set_ylabel(f"{follower_display}  R (V)")
    ax2.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax1.set_title("Measurement result")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    lines: list[str] = []
    if plan is not None:
        theta = plan.geometry_cfg.field_theta_deg
        if theta is not None:
            lines.append(field_direction_summary_line(theta, plan.geometry_cfg.field_phi_deg))
        freq_Hz = plan.header_extra.get("excitation_frequency_Hz")
        amps = sorted({r["excitation_current_A_peak"] for r in records
                       if r.get("excitation_current_A_peak") is not None})
        if freq_Hz is not None and len(amps) == 1:
            lines.append(f"AC excitation: {format_si(amps[0], 'A')} @ {format_si(freq_Hz, 'Hz')}")
        tc1, order1 = plan.demod1_cfg.filter.time_constant_s, plan.demod1_cfg.filter.order
        tc2, order2 = plan.demod2_cfg.filter.time_constant_s, plan.demod2_cfg.filter.order
        lines.append(f"Filter: {leader_display} TC={tc1:g} s order={order1}, "
                     f"{follower_display} TC={tc2:g} s order={order2}")
    if comment:
        lines.append(f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}")
    if lines:
        fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color="0.4", va="bottom")
        fig.subplots_adjust(bottom=0.08 + 0.045 * len(lines))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot to '%s'", png_path)


# ─────────────────────────────────────────────────────────────────────────────
# Logging -> RichLog relay
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Plan + run  ── pure, shared by the TUI RunScreen and web/mfli/dual_harmonic_6221.py
# ─────────────────────────────────────────────────────────────────────────────

def build_plan(state: dict, data_root: Path) -> MeasurementPlan:
    """One parsed, validated run request from a state dict. Pure — shared by
    the TUI and the web page."""
    ac_cfg = ACSourceConfig(
        visa_resource=state["ac_visa_resource"],
        amplitude_A=state["amplitude_list"][0],
        frequency_Hz=state["frequency_Hz"],
        compliance_V=state["ac_compliance_V"],
        phasemarker_line=state["phasemarker_line"],
    )
    leader_extref_cfg = ExtRefConfig(
        device=state["leader_device"], extref_index=state["leader_extref_index"],
        aux_input_ch=state["leader_aux_input_ch"], osc_index=state["leader_osc_index"],
        pll_demod_index=state["leader_pll_demod_index"], automode=state["leader_automode"],
    )
    follower_extref_cfg = ExtRefConfig(
        device=state["follower_device"], extref_index=state["follower_extref_index"],
        aux_input_ch=state["follower_aux_input_ch"], osc_index=state["follower_osc_index"],
        pll_demod_index=state["follower_pll_demod_index"], automode=state["follower_automode"],
    )
    filt_1f = FilterConfig(
        time_constant_s=state["time_constant_1f_s"],
        order=state["order_1f"],
        sinc_filter=state["sinc_filter_1f"],
    )
    filt_2f = FilterConfig(
        time_constant_s=state["time_constant_2f_s"],
        order=state["order_2f"],
        sinc_filter=state["sinc_filter_2f"],
    )
    demod1_cfg = DemodConfig(
        device=state["leader_device"], demod_index=0, harmonic=state["leader_harmonic"],
        osc_index=state["leader_osc_index"],
        differential=state["differential"], ac_coupling=state["ac_coupling"],
        input_range_V=state["input_range_1f_V"],
        sample_rate_Hz=state["sample_rate_Hz"], filter=filt_1f,
    )
    demod2_cfg = DemodConfig(
        device=state["follower_device"], demod_index=0,
        harmonic=state["follower_harmonic"],
        osc_index=state["follower_osc_index"],
        differential=state["differential"], ac_coupling=state["ac_coupling"],
        input_range_V=state["input_range_2f_V"],
        sample_rate_Hz=state["sample_rate_Hz"], filter=filt_2f,
    )
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"],
        field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
        n_averages=state["n_averages"],
        output_file="",  # overwritten per amplitude iteration in RunScreen.do_run()
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
        currents_A = build_segmented_sweep(state["sweep_rows_parsed"], bidirectional=True)

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
        field_theta_deg=state["field_theta_deg"],
        field_phi_deg=state["field_phi_deg"],
    )

    header_extra = {
        "excitation_frequency_Hz": state["frequency_Hz"],
        "excitation_amplitude_A": state["amplitude_list"][0],
        "measure_rxx": state["measure_rxx"],
        "demod1_time_constant_s": state["time_constant_1f_s"],
        "demod1_order": state["order_1f"],
        "demod2_time_constant_s": state["time_constant_2f_s"],
        "demod2_order": state["order_2f"],
        "n_averages": state["n_averages"],
        "settling_time_s": state["settling_time_s"],
    }
    if state["enable_sweep"]:
        header_extra["field_sweep_rows_A"] = state["sweep_rows_parsed"]

    series = ""
    if len(state["amplitude_list"]) > 1:
        series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                  f"{datetime.now():%Y%m%dT%H%M%S}")

    return MeasurementPlan(
        daq_host=state["daq_host"], daq_port=state["daq_port"],
        leader=state["leader_device"], follower=state["follower_device"],
        ac_cfg=ac_cfg, amplitudes_A=state["amplitude_list"],
        measure_rxx=state["measure_rxx"], leader_measure_rxx=state["leader_measure_rxx"],
        leader_extref_cfg=leader_extref_cfg,
        follower_extref_cfg=follower_extref_cfg,
        extref_lock_timeout_s=state["extref_lock_timeout_s"],
        demod1_cfg=demod1_cfg, demod2_cfg=demod2_cfg,
        acq_cfg=acq_cfg, magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, currents_A=currents_A,
        temp_cfg=temp_cfg,
        phase_cal_enabled=state["enable_phase_cal"],
        phase_cal_current_A=state["phase_cal_current_A"],
        phase_cal_n_averages=state["phase_cal_n_averages"],
        phase_cal_max_iterations=state["phase_cal_max_iterations"],
        geometry_cfg=geometry_cfg,
        sample=state["sample"], device=state["device"], data_root=data_root,
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series=series,
        run_cost=run_costs(state, currents_A),
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
    """Connect both MFLIs (MDS), then per excitation current: re-arm the 6221,
    lock both ExtRef PLLs, optionally phase-calibrate, and record one run (own
    run number, own file) through record_run() before the next; always shut
    the instruments down. Pure — the TUI's RunScreen and the web page each pass
    their own callbacks."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    daq = source = magnet = gaussmeter = temp_ctrl = None
    (leader_prefix, leader_display), (follower_prefix, follower_display) = plan_naming(plan)
    try:
        on_status("Connecting to LabOne data server …")
        daq = connect(plan.daq_host, plan.daq_port)
        connect_device(daq, plan.leader, interface="1GbE")
        connect_device(daq, plan.follower, interface="1GbE")

        on_status("Synchronizing MDS …")
        mds = setup_mds(daq, leader=plan.leader, follower=plan.follower)

        disable_sigout(daq, plan.leader)
        disable_sigout(daq, plan.follower)

        on_status("Configuring demodulators …")
        configure_demodulator(daq, plan.demod1_cfg)
        configure_demodulator(daq, plan.demod2_cfg)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC (temperature) …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        if plan.magnet_cfg is not None and plan.currents_A is not None:
            on_status("Connecting magnet power supply …")
            magnet = connect_magnet(plan.magnet_cfg)
            on_status("Connecting gaussmeter …")
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)
            # Amplitude-independent -- built once, reused for every amplitude.
            points = [
                MeasurementPoint(
                    magnet_current_A=I,
                    set_action=lambda daq, I=I: set_magnet_current(
                        magnet, plan.magnet_cfg, I, gaussmeter, plan.gauss_cfg,
                        plan.acq_cfg.field_settle_tolerance_mT, stop_event),
                )
                for I in plan.currents_A
            ]
        else:
            points = [MeasurementPoint()]

        multi = len(plan.amplitudes_A) > 1
        for series_idx, amp in enumerate(plan.amplitudes_A):
            if stop_event.is_set():
                break
            plan.ac_cfg.amplitude_A = amp
            label = f"I={amp:g}A" if multi else None

            # Checked here too, not just by build_summary(): connect_ac_source()
            # immediately arms and starts the 6221 at plan.ac_cfg.amplitude_A —
            # catch a mistyped exponent before that, not after.
            _check_ac_safety(plan.ac_cfg)
            if source is not None:
                # Amplitude requires a full re-arm (a property write after
                # waveform_arm() doesn't take effect until the next arm()) --
                # tear down the previous amplitude's source first, guarded so a
                # GPIB hiccup here never skips the remaining amplitudes.
                safe_shutdown("6221 AC source", lambda _s=source: shutdown_ac_source(_s))
                source = None
            on_status(f"Starting 6221 AC current source{f' ({amp:g} A)' if multi else ''} …")
            source = connect_ac_source(plan.ac_cfg)

            on_status("Locking MFLI oscillators to the 6221 marker (ExtRef) …")
            configure_external_reference(daq, plan.leader_extref_cfg, plan.ac_cfg.frequency_Hz)
            configure_external_reference(daq, plan.follower_extref_cfg, plan.ac_cfg.frequency_Hz)
            if not wait_for_reference_lock(daq, plan.leader_extref_cfg,
                                           plan.extref_lock_timeout_s, stop_event):
                log.warning("Leader ExtRef PLL did not report locked within %.2g s — "
                            "check the marker cabling before trusting any data.",
                            plan.extref_lock_timeout_s)
            if not wait_for_reference_lock(daq, plan.follower_extref_cfg,
                                           plan.extref_lock_timeout_s, stop_event):
                log.warning("Follower ExtRef PLL did not report locked within %.2g s — "
                            "check the marker fan-out cabling before trusting any data.",
                            plan.extref_lock_timeout_s)

            demod2_phase_null_1f_deg = None
            if plan.phase_cal_enabled:
                on_status(f"Phase calibration: nulling {leader_display} Y (leader demod phaseshift) …")
                if magnet is not None and plan.phase_cal_current_A is not None:
                    log.info("Phase calibration: ramping magnet to %.4f A ...",
                             plan.phase_cal_current_A)
                    set_magnet_current(magnet, plan.magnet_cfg, plan.phase_cal_current_A,
                                       gaussmeter, plan.gauss_cfg,
                                       plan.acq_cfg.field_settle_tolerance_mT, stop_event)
                    time.sleep(plan.acq_cfg.settling_time_s)
                result = auto_null_phase(
                    daq, plan.demod1_cfg,
                    n_averages=plan.phase_cal_n_averages,
                    max_iterations=plan.phase_cal_max_iterations,
                )
                if not result.converged:
                    log.warning(
                        "Phase null did not fully converge after %d iteration(s) "
                        "(|Y|/R=%.2e) — check cabling/contacts before trusting the %s data.",
                        result.iterations, result.residual_ratio, follower_display,
                    )
                d2 = acquire_averaged(daq, plan.demod2_cfg, plan.phase_cal_n_averages)
                log.info(
                    "%s snapshot at calibration point: X=%.4e V  Y=%.4e V  R=%.4e V — "
                    "don't assume this matches 1f's X/Y convention (V_2w ~ cos, not sin); "
                    "check which channel carries the structured field dependence in the "
                    "recorded sweep before trusting either one.",
                    follower_display, d2["x_mean"], d2["y_mean"], d2["r_mean"],
                )
                on_status(f"Phase calibration: anchoring follower {follower_display} reference (1f null) …")
                demod2_phase_null_1f_deg = null_follower_reference_via_1f(
                    daq, plan.demod2_cfg,
                    n_averages=plan.phase_cal_n_averages,
                    max_iterations=plan.phase_cal_max_iterations,
                )

            # A fresh RunContext (own run number, own file) EVERY amplitude
            # iteration -- never reuse one across the series.
            ctx = allocate_run(
                plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                temperature_setpoint_K=plan.temperature_setpoint_K,
                key_axis=None, series=plan.series,
            )
            extra = {"excitation_amplitude_A": amp}
            run_contexts.append(ctx)
            run_extras.append(extra)
            on_run_label(f"Run #{ctx.run_str}")
            plan.acq_cfg.output_file = str(ctx.raw_path)

            on_status("Running measurement …" if not multi else f"Running measurement ({label}) …")
            record_run(
                plan.data_root, ctx,
                lambda records, status, _ctx=ctx, _x=extra: build_header_fields(
                    plan, _ctx, records, status=status, comment="", extra=_x),
                lambda point_cb, write_csv, _null=demod2_phase_null_1f_deg: run_measurement(
                    daq, plan.ac_cfg, plan.leader_extref_cfg, plan.follower_extref_cfg,
                    plan.demod1_cfg, plan.demod2_cfg, plan.acq_cfg, points,
                    stop_event=stop_event, on_point=point_cb,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                    geometry_cfg=plan.geometry_cfg,
                    demod2_phase_null_1f_deg=_null, mds=mds,
                    write_csv=write_csv, demod2_label=follower_prefix,
                    demod1_label=leader_prefix),
                stop_event, on_point=on_point,
                tags={"series_index": series_idx, "series_label": label},
                on_finished=on_run_finished)
    finally:
        # 6221 output off first (immediate, no current into the DUT), so the
        # magnet can start its ramp-down right away rather than waiting behind it.
        if source is not None:
            safe_shutdown("6221 AC source", lambda: shutdown_ac_source(source))
        if magnet is not None:
            safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
        if gaussmeter is not None:
            safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (RunScreen and the web page both call this)."""
    _save_measurement_png(records, png_path, plan=plan, comment=comment)


# ─────────────────────────────────────────────────────────────────────────────
# Run screen  ── executes the plan in a worker thread, shows live progress
# ─────────────────────────────────────────────────────────────────────────────

class RunScreen(MeasurementRunScreen):
    MEASUREMENT_TYPE = MEASUREMENT_TYPE

    def __init__(self, plan: MeasurementPlan) -> None:
        super().__init__(plan)
        self._naming = plan_naming(plan)

    def table_columns(self) -> tuple:
        (_, l), (_, d) = self._naming
        return ("#", "I (A)", "B (mT)", f"{l} R (V)", f"{l} θ (°)", f"{d} R (V)", f"{d} θ (°)",
                "T1 (K)", "T2 (K)")

    def progress_points(self) -> int:
        return self.plan.total_points * self.plan.total_files

    def live_plot_args(self):
        return (_live_plot_worker, self.plan.magnet_cfg is not None,
                self._naming, self.plan.total_files > 1)

    def table_row(self, record: dict) -> tuple:
        I = record.get("magnet_current_A")
        B = record.get("magnet_field_mT")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        (lp, _), (fp, _) = self._naming
        return (
            str(record["point_index"] + 1),
            f"{I:.4f}" if I is not None else "—",
            f"{B:.2f}" if B is not None else "—",
            f"{record[f'{lp}_R_V']:.4e}",
            f"{record[f'{lp}_theta_deg']:.2f}",
            f"{record[f'{fp}_R_V']:.4e}",
            f"{record[f'{fp}_theta_deg']:.2f}",
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )


def main() -> None:
    """The 6221 source is a mode of the dual-harmonic program now — open that
    form with 'AC current source' set to the 6221."""
    from mfli.mfli_dual_harmonic_tui import MFLIDualHarmonicApp
    MFLIDualHarmonicApp(ac_source="6221").run()


if __name__ == "__main__":
    main()
