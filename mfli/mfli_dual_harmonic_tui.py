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
import textwrap
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.validation import Number
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Select,
    Static,
)

from dc.dc_sweep_utils import build_segmented_sweep, parse_sweep_rows, safe_shutdown, try_parse
import mfli.mfli_dual_harmonic_6221_tui as six
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
    configure_demodulator,
    configure_output,
    connect,
    connect_device,
    connect_gaussmeter,
    connect_magnet,
    connect_temperature_controller,
    null_follower_reference_via_1f,
    phase_cal_s,
    run_measurement,
    set_magnet_current,
    setup_mds,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_output,
    shutdown_temperature_controller,
    sync_follower_oscillator,
)
from instruments.data_dir import validate_directory
from instruments.field_geometry import field_direction_summary_line, render_ascii_field_diagram
from instruments.data_naming import (
    RunContext,
    allocate_run,
    record_run,
    preview_raw_filename,
)
from instruments.kepco_magnet import magnet_move_s
from instruments.lakeshore475 import read_field_s
from instruments.mfli_daq import acquire_s, poll_window_s
from instruments.run_time import (
    GPIB_TXN_S, MDS_SYNC_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
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
    sweep_rows_field,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
)

log = logging.getLogger("mfli_dual_harmonic_tui")

# Data/settings live outside "bridge" (a sibling of it), same convention as
# mfli_dual_harmonic.py, so nothing generated at runtime ends up in the
# git-tracked source tree. _DEFAULT_DATA_DIR is the fallback data-convention
# "data root"; the real root is chosen per run in the identity bar's "Data
# root" field (mirrors the web app) and persisted in the settings file.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "mfli_dual_harmonic_tui_settings.json"

# Locked type code for this measurement (see instruments/data_naming.py) —
# never deviates.
MEASUREMENT_TYPE = "HARM"

# One-paragraph blurb + wiring schematic — shown on this program's card in
# bridge_tui.py, and the description also on its web page.
MFLI_DUAL_HARMONIC_DESCRIPTION = (
    "Drives an AC current through the sample and reads the 1st-harmonic response "
    "on the leader while the follower reads the 2nd-harmonic response — the "
    "standard setup for e.g. a nonlinear/planar Hall measurement. The current "
    "comes from the leader MFLI's Signal Output through a series resistor, or "
    "— 'AC current source' toggle — from a Keithley 6221 ideal current source "
    "whose phase marker both MFLIs lock to (a list of currents and an R_xx mode "
    "too). Optionally sweeps a Kepco electromagnet's field (bidirectionally, for "
    "hysteresis) with the field measured live via a Lake Shore 475 Gaussmeter at "
    "every point."
)

MFLI_DUAL_HARMONIC_SCHEMATIC = """\
  AC current source = MFLI (type HARM)
    Leader Signal Output 1 ──[ R_series ]──▶ sample/DUT ── common ground
    Leader Signal Input 1  (differential)  ──▶ demod 1f   (V_Rseries → I)

  AC current source = Keithley 6221 (type HARM6)
    6221 HI/LO ──▶ sample/DUT ── common ground
    Trigger Link phase marker ──▶ split (BNC T, equal lengths) to Aux In 1
      on BOTH the leader AND the follower — each locks its oscillator to it

  LEADER MFLI    (1f)  Signal Input 1 (differential) ──▶ demod 1f
  FOLLOWER MFLI  (2f)  Signal Input 1 (differential) ──▶ demod 2f
                       (or R_xx's 1f in the 6221 source's R_xx mode)

  MDS cabling  (both units)
    Leader Ref Out      ───BNC───▶ Follower Ref In
    Leader Trigger Out 1 ──▶ fanned out to Trigger In 1 on BOTH units

  Magnet field sweep  (optional, "Sweep magnetic field" switch)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample
"""


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
}

# id -> caster, for every free-text numeric field (Select/Switch handled separately)
NUMERIC_FIELDS: dict = {
    "daq_port": int,
    "frequency_Hz": float,
    "amplitude_V": float,
    "series_R_ohm": float,
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
}
TEXT_FIELDS = ["leader_device", "follower_device", "daq_host", "device", "cooldown", "visa_resource",
               "gaussmeter_visa_resource", "temperature_visa_resource", "temperature_sensor_uids",
               "data_dir"]
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


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def run_costs(state: dict, currents_A=None) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (`currents_A` is the resolved field sweep, or None for a single point at
    the present field). Also drives the run screen's progress bar, so the
    estimate and the live ETA cannot disagree. Every term mirrors a step of
    run_measurement() / RunScreen.do_run() -- see mfli_dual_harmonic.py."""
    n = len(currents_A) if currents_A is not None else 1
    rc = RunCost(n)
    rate, n_avg = state["sample_rate_Hz"], state["n_averages"]
    tc1, tc2 = state["time_constant_1f_s"], state["time_constant_2f_s"]
    # acquire_averaged_pair(): 1f and 2f share ONE poll window -- the longer of the two.
    pair_s = max(acquire_s(tc1, n_avg, rate), acquire_s(tc2, n_avg, rate)) if rate > 0 else 0.0
    has_temp = bool(state["enable_temperature"] and parse_sensor_uids(state["temperature_sensor_uids"]))
    rc.each("settle", state["settling_time_s"])
    rc.each("acquire", pair_s)
    # per point: MDS status check + 2 phase-node reads (build_run_metadata) + CSV rewrite + temperature
    rc.each("overhead", 3 * GPIB_TXN_S + POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    rc.at("connect + MDS", PER_RUN_S + MDS_SYNC_S, 0)
    magnet_cfg = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    i_now = 0.0                                   # the magnet starts at 0 A
    if state["enable_phase_cal"]:
        rc.at("phase cal", phase_cal_s(tc1, tc2, state["phase_cal_n_averages"],
                                       state["phase_cal_max_iterations"], rate) if rate > 0 else 0.0, 0)
        if currents_A is not None and state["phase_cal_current_A"] is not None:
            typ, worst = magnet_move_s(abs(state["phase_cal_current_A"]), magnet_cfg)
            rc.at("phase cal", typ + state["settling_time_s"], 0, worst_extra=worst - typ)
            i_now = state["phase_cal_current_A"]
    if currents_A is not None:
        rc.each("field read", read_field_s(GaussmeterConfig(
            n_averages=state["gaussmeter_n_averages"], read_delay_s=state["gaussmeter_read_delay_s"])))
        for i, current in enumerate(currents_A):
            typ, worst = magnet_move_s(abs(current - i_now), magnet_cfg)
            rc.at("magnet", typ, i, worst_extra=worst - typ)
            i_now = current
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
    data_root: Path = _DEFAULT_DATA_DIR
    run_cost: Optional[RunCost] = None      # modelled seconds per point (progress bar + ETA)

    @property
    def total_points(self) -> int:
        return len(self.currents_A) if self.currents_A is not None else 1


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                        status: str, comment: str, extra: Optional[dict] = None) -> dict:
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
# Small widget-building helpers (keep compose() readable)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Live validation / derived-value summary
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_state_mfli(state: dict) -> dict:
    """Add the derived keys build_summary() / build_plan() read — the parsed
    lists/sweeps, each with its parse error — to a state of raw field values.
    Pure: shared by the TUI's and the web page's parse_state()."""
    state["sweep_rows_parsed"], state["sweep_rows_parse_error"] = try_parse(state["sweep_rows"], parse_sweep_rows)
    return state


def _build_summary_mfli(state: dict) -> tuple[list[str], list[str], list[str]]:
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

    acq_window_s = {"1f": 0.0, "2f": 0.0}
    for label, tc_key, order_key in (("1f", "time_constant_1f_s", "order_1f"),
                                      ("2f", "time_constant_2f_s", "order_2f")):
        tc = state[tc_key]
        if tc > 0:
            # Rule of thumb: ≥5×TC for a 1st-order filter, ≥10×TC for 3rd/4th
            # order (settles more slowly per time constant at higher order).
            order = state[order_key]
            settle_multiple = 10 if order >= 3 else 5
            recommended_settle = settle_multiple * tc
            if state["settling_time_s"] < recommended_settle:
                warnings.append(
                    f"{label} settling time {state['settling_time_s']:g} s < {settle_multiple}×TC "
                    f"({recommended_settle:g} s, order {order}) — filter may not have settled."
                )
            else:
                info.append(f"{label} settling ≥ {settle_multiple}×TC ({recommended_settle:g} s) ✓")

            bw = 1.0 / (2 * math.pi * tc)
            min_rate = 4 * bw
            info.append(f"{label} filter noise bandwidth ≈ {bw:.3g} Hz")
            if state["sample_rate_Hz"] < min_rate:
                warnings.append(
                    f"Sample rate {state['sample_rate_Hz']:g} Sa/s may be low for {label} TC "
                    f"(want ≳ {min_rate:.1f} Sa/s)."
                )

            # Independent-average check: acquire_averaged() polls for
            # max(0.1, 3xTC, n*1.5/rate) s, but consecutive demod outputs are
            # correlated over ~TC, so the window only holds ~window/(pi*TC)
            # independent samples. If that's well below n_averages, the mean
            # barely beats one reading and the reported 1f/2f R_sem is optimistic.
            acq_window_s[label] = (poll_window_s(tc, state["n_averages"], state["sample_rate_Hz"])
                                   if state["sample_rate_Hz"] > 0 else 0.0)
            n_indep = acq_window_s[label] / (math.pi * tc)
            if n_indep < 0.5 * state["n_averages"]:
                warnings.append(
                    f"{label} averaging window ≈ {acq_window_s[label]:g} s holds only "
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
            info.append(f"Sweep: {len(rows)} row(s), {total_points} points (bidirectional)"
                         f"{merged_note}")
        info.append("Field measured live at each point via Lake Shore 475 Gaussmeter "
                     f"({state['gaussmeter_visa_resource']})")
        tol_mT = state["field_settle_tolerance_mT"]
        if tol_mT <= 0:
            warnings.append("Field-settle tolerance is 0 — every magnet step will wait the "
                             "full settle timeout before acquiring.")
        elif tol_mT < 0.01:
            warnings.append(f"Field-settle tolerance {tol_mT:g} mT is below the 475's typical "
                             "reading noise — points may stall until the settle timeout.")
        if resolved is not None:
            info.extend(run_costs(state, resolved).lines("Estimated total run time"))
    else:
        info.append("Single point — no field sweep, magnet untouched.")
        info.extend(run_costs(state).lines("Estimated run time"))

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

    if state["enable_phase_cal"]:
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
                info.append(
                    f"Phase cal: ramp to {state['phase_cal_current_A']:g} A, null 1f Y "
                    "(leader demod phaseshift), then run the sweep."
                )
        else:
            info.append("Phase cal: null 1f Y at the present field (no magnet ramp).")

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


def _preview_mfli(state: dict) -> Optional[str]:
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


def _save_measurement_png(records: list[dict], png_path: Path,
                           plan: Optional["MeasurementPlan"] = None, comment: str = "") -> None:
    """Save a static 1f/2f R-vs-field PNG to proc/, from whatever points
    were actually collected (including an aborted/partial run).

    `plan`/`comment` add a small "at a glance" text annotation (field
    direction, the AC excitation, the 1f/2f filter TC/order, the
    operator's comment) for context not already in the filename. Called
    once when the run ends (comment="")
    and again, to overwrite the PNG in place, once the operator's comment
    is known."""
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

    lines: list[str] = []
    if plan is not None:
        theta = plan.geometry_cfg.field_theta_deg
        if theta is not None:
            lines.append(field_direction_summary_line(theta, plan.geometry_cfg.field_phi_deg))
        freq_Hz = plan.header_extra.get("excitation_frequency_Hz")
        amp_V = plan.header_extra.get("excitation_amplitude_V")
        if freq_Hz is not None and amp_V is not None:
            lines.append(f"AC excitation: {format_si(amp_V, 'V')} @ {format_si(freq_Hz, 'Hz')}")
        tc1, order1 = plan.demod1_cfg.filter.time_constant_s, plan.demod1_cfg.filter.order
        tc2, order2 = plan.demod2_cfg.filter.time_constant_s, plan.demod2_cfg.filter.order
        lines.append(f"Filter: 1f TC={tc1:g} s order={order1}, 2f TC={tc2:g} s order={order2}")
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
# Logging -> RichLog relay (keeps raw log lines from corrupting the alt screen)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Plan + run  ── pure, shared by the TUI RunScreen and web/mfli/dual_harmonic.py
# ─────────────────────────────────────────────────────────────────────────────

def _build_plan_mfli(state: dict, data_root: Path) -> MeasurementPlan:
    """One parsed, validated run request from a state dict. Pure — shared by
    the TUI and the web page."""
    out_cfg = OutputConfig(
        device=state["leader_device"],
        frequency_Hz=state["frequency_Hz"],
        amplitude_V=state["amplitude_V"],
        series_R_ohm=state["series_R_ohm"],
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
        device=state["leader_device"], demod_index=0, harmonic=1,
        differential=state["differential"], ac_coupling=state["ac_coupling"],
        input_range_V=state["input_range_1f_V"],
        sample_rate_Hz=state["sample_rate_Hz"], filter=filt_1f,
    )
    demod2_cfg = DemodConfig(
        device=state["follower_device"], demod_index=0, harmonic=2,
        differential=state["differential"], ac_coupling=state["ac_coupling"],
        input_range_V=state["input_range_2f_V"],
        sample_rate_Hz=state["sample_rate_Hz"], filter=filt_2f,
    )
    run_ctx = allocate_run(
        data_root, state["sample"], state["device"], MEASUREMENT_TYPE,
        temperature_setpoint_K=state["temperature_setpoint_K"],
    )
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"],
        field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
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

    # Geometry/dimensions are recorded per-row in the CSV (via
    # build_run_metadata) and belong in sample.yaml long-term — they are
    # deliberately NOT duplicated into header_extra/index.csv here.
    header_extra = {
        "excitation_frequency_Hz": state["frequency_Hz"],
        "excitation_amplitude_V": state["amplitude_V"],
        "series_R_ohm": state["series_R_ohm"],
        "demod1_time_constant_s": state["time_constant_1f_s"],
        "demod1_order": state["order_1f"],
        "demod2_time_constant_s": state["time_constant_2f_s"],
        "demod2_order": state["order_2f"],
        "n_averages": state["n_averages"],
        "settling_time_s": state["settling_time_s"],
    }
    if state["enable_sweep"]:
        header_extra["field_sweep_rows_A"] = state["sweep_rows_parsed"]

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
        run_ctx=run_ctx, data_root=data_root,
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra,
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
    """Record the plan's single run (plan.run_ctx, allocated at Start): connect
    both MFLIs (MDS), optionally phase-calibrate, sweep, finalize the run the
    instant it ends — even if connecting failed — then shut down, and only
    then save the PNG. Pure — the TUI's RunScreen and the web page each pass
    their own callbacks."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    ctx = plan.run_ctx
    run_contexts.append(ctx)
    run_extras.append(None)
    daq = magnet = gaussmeter = temp_ctrl = None
    recorded: list[dict] = []

    def measure(point_cb, write_csv) -> None:
        nonlocal daq, magnet, gaussmeter, temp_ctrl
        on_status("Connecting to LabOne data server …")
        daq = connect(plan.daq_host, plan.daq_port)
        connect_device(daq, plan.leader, interface="1GbE")
        connect_device(daq, plan.follower, interface="1GbE")

        on_status("Synchronizing MDS …")
        mds = setup_mds(daq, leader=plan.leader, follower=plan.follower)

        on_status("Configuring output & demodulators …")
        configure_output(daq, plan.out_cfg)
        sync_follower_oscillator(daq, plan.out_cfg, plan.follower)
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

        demod2_phase_null_1f_deg = None
        if plan.phase_cal_enabled:
            on_status("Phase calibration: nulling 1f Y (leader demod phaseshift) …")
            if magnet is not None and plan.phase_cal_current_A is not None:
                log.info("Phase calibration: ramping magnet to %.4f A ...", plan.phase_cal_current_A)
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
            # Anchor the follower's 2f reference to the current: null the
            # follower at 1f against the same (split) V_xy, record the delay
            # angle as demod2_phase_null_1f_deg so analysis can rotate the
            # recorded 2f X/Y into the current frame.
            on_status("Phase calibration: anchoring follower 2f reference (1f null) …")
            demod2_phase_null_1f_deg = null_follower_reference_via_1f(
                daq, plan.demod2_cfg,
                n_averages=plan.phase_cal_n_averages,
                max_iterations=plan.phase_cal_max_iterations,
            )

        on_status("Running measurement …")
        run_measurement(
            daq, plan.out_cfg, plan.demod1_cfg, plan.demod2_cfg, plan.acq_cfg, points,
            stop_event=stop_event, on_point=point_cb,
            gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
            temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
            geometry_cfg=plan.geometry_cfg,
            demod2_phase_null_1f_deg=demod2_phase_null_1f_deg, mds=mds,
            write_csv=write_csv,
        )

    try:
        record_run(plan.data_root, ctx,
                   lambda records, status: build_header_fields(plan, ctx, records, status=status, comment=""),
                   measure, stop_event,
                   on_point=lambda record: (recorded.append(record), on_point(record)))
    finally:
        # Excitation output off first (immediate, no current into the DUT), so
        # the magnet can start its ramp-down right away rather than waiting.
        if daq is not None:
            safe_shutdown("MFLI output", lambda: shutdown_output(daq, plan.out_cfg))
        if magnet is not None:
            safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
        if gaussmeter is not None:
            safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))
        if on_run_finished is not None:
            try:
                on_run_finished(ctx, recorded)
            except Exception:
                log.exception("Could not save the run's plot")


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """The run's PNG (RunScreen and the web page both call this)."""
    _save_measurement_png(records, png_path, plan=plan, comment=comment)


# ─────────────────────────────────────────────────────────────────────────────
# AC-source toggle  ── one program, two excitation-current sources
# ─────────────────────────────────────────────────────────────────────────────
# "mfli": the leader's Signal Output through a series resistor — type HARM, the
# functions above. "6221": a Keithley 6221 current source whose Trigger-Link
# phase marker both MFLIs ExtRef-lock to — type HARM6, mfli_dual_harmonic_6221_tui.
# Each source keeps its own type code, columns and header, so a run's files are
# exactly what that source's program always wrote. The form is the union of the
# two (44 shared fields; the source-only ones are hidden in the other mode).

AC_SOURCES = [("MFLI Signal Output (+ series resistor)", "mfli"),
              ("Keithley 6221 (phase marker → ExtRef)", "6221")]
_MFLI_ONLY_FIELDS = tuple(k for k in DEFAULTS if k not in six.DEFAULTS)
_6221_ONLY_FIELDS = tuple(k for k in six.DEFAULTS if k not in DEFAULTS)
DEFAULTS = {**six.DEFAULTS, **DEFAULTS, "ac_source": "mfli"}
NUMERIC_FIELDS = {**six.NUMERIC_FIELDS, **NUMERIC_FIELDS}
TEXT_FIELDS = TEXT_FIELDS + [f for f in six.TEXT_FIELDS if f not in TEXT_FIELDS]
OPTIONAL_NUMERIC_FIELDS = OPTIONAL_NUMERIC_FIELDS + [
    f for f in six.OPTIONAL_NUMERIC_FIELDS if f not in OPTIONAL_NUMERIC_FIELDS]


def _uses_6221(state: dict) -> bool:
    return state.get("ac_source") == "6221"


def mode_errors(state: dict, errors: list[str]) -> list[str]:
    """Parse errors of the ACTIVE source only — a hidden field of the other
    source never blocks a run."""
    hidden = _MFLI_ONLY_FIELDS if _uses_6221(state) else _6221_ONLY_FIELDS
    return [e for e in errors if not any(e.startswith(f"'{f}'") for f in hidden)]


def resolve_state(state: dict) -> dict:
    return six.resolve_state(state) if _uses_6221(state) else _resolve_state_mfli(state)


def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    return six.build_summary(state) if _uses_6221(state) else _build_summary_mfli(state)


def compute_filename_preview(state: dict) -> Optional[str]:
    return six.compute_filename_preview(state) if _uses_6221(state) else _preview_mfli(state)


def build_plan(state: dict, data_root: Path):
    """The active source's plan (its own MeasurementPlan type)."""
    return six.build_plan(state, data_root) if _uses_6221(state) else _build_plan_mfli(state, data_root)


def engine(plan):
    """The module that runs `plan` — its run_plan / header / PNG / type code
    (both front ends take them from here, never from the toggle)."""
    return six if isinstance(plan, six.MeasurementPlan) else sys.modules[__name__]



# ─────────────────────────────────────────────────────────────────────────────
# Run screen  ── executes the plan in a worker thread, shows live progress
# ─────────────────────────────────────────────────────────────────────────────

class RunScreen(MeasurementRunScreen):
    TABLE_COLUMNS = ("#", "I (A)", "B (mT)", "1f R (V)", "1f θ (°)", "2f R (V)", "2f θ (°)", "T1 (K)", "T2 (K)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE

    def live_plot_args(self):
        return (_live_plot_worker, self.plan.magnet_cfg is not None)

    def table_row(self, record: dict) -> tuple:
        I = record.get("magnet_current_A")
        B = record.get("magnet_field_mT")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        return (
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

    def progress_index(self, record: dict) -> int:
        return record["point_index"]


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class MFLIDualHarmonicApp(MeasurementApp):
    TITLE = "MFLI Dual-Harmonic Measurement"
    SUB_TITLE = "1f / 2f lock-in · MFLI or Keithley 6221 excitation · magnet field sweep"

    # widgets shown only for one AC source (see compose)
    SOURCE_WIDGETS = {"mfli": ("mode_mfli_excitation",),
                      "6221": ("mode_6221_excitation", "mode_6221_quantities", "mode_6221_extref")}

    def __init__(self, ac_source: Optional[str] = None) -> None:
        super().__init__()
        self._forced_source = ac_source      # e.g. "6221" from the old 6221 entry point

    # Session data root — fallback until _load_settings()/the identity bar's
    # "Data root" field replaces it. Read in compose(), so it must exist here.
    data_root: Path = _DEFAULT_DATA_DIR

    SWITCH_DEPENDENTS = {
        "enable_sweep": (*MAGNET_FIELD_IDS, "sweep_rows"),
        "enable_temperature": tuple(TEMPERATURE_FIELD_IDS),
    }

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 48; border-left: solid $primary; padding: 1 2; overflow-y: auto; }
    .field-label { text-style: bold; }
    .hint { text-style: italic; color: $text-muted; }
    .sweep-rows { height: 5; margin-bottom: 1; }
    .switch-row { height: 3; }
    .switch-row Label { margin-left: 1; content-align: left middle; height: 3; }
    .plane-btn-row { height: 3; margin-bottom: 1; }
    .plane-btn-row Button { min-width: 5; margin-right: 1; }
    .field-diagram { color: $text-muted; margin-top: 1; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; }
    #actionbar { height: 3; align: center middle; }

    #identity_bar { border: round $accent; padding: 1 2; height: auto; margin-bottom: 1; }
    #filename_preview { margin-bottom: 1; }
    #data_dir_row { height: 3; margin-bottom: 1; }
    #data_dir_row Input { width: 1fr; }
    #data_dir_row Button { margin-left: 1; }
    #identity_fields { layout: grid; grid-size: 4; grid-gutter: 1 2; height: auto; }
    #identity_fields > Vertical { height: auto; }
    .field { margin-bottom: 1; }
    .section-title { text-style: bold underline; margin: 1 0; }
    .param-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }
    .param-card { border: round $primary; padding: 1 2; height: auto; }
    .stable-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }

    /* Collapsible tiers -- precision knobs + instrument wiring, folded by default */
    Collapsible { height: auto; margin: 1 0; }
    Collapsible > Contents { padding: 1 0 0 1; }
    CollapsibleTitle { text-style: bold; color: $text-muted; }
    .stable-card { border: round $panel-darken-1; padding: 1 2; height: auto; }
    .stable-card .card-title { color: $text-muted; }
    .stable-card .field-label { color: $text-muted; }
    .card-title { text-style: bold underline; margin-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                yield identity_bar(DEFAULTS, _DEFAULT_DATA_DIR, self.data_root)

                # ── Tier 1: what defines this run — always visible ──────────
                with Vertical(classes="param-grid"):
                    yield card(
                        "Excitation",
                        select_field("ac_source", "AC current source", AC_SOURCES,
                                     DEFAULTS["ac_source"],
                                     hint="MFLI → saved as HARM · 6221 → HARM6"),
                        field("frequency_Hz", "Excitation frequency (Hz)",
                              DEFAULTS["frequency_Hz"],
                              hint="Recommended ~300-1000 Hz — avoid exact multiples of 50/60 Hz "
                                   "(mains pickup).",
                              validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                    )
                    yield card(
                        "MFLI Signal Output",
                        field("amplitude_V", "Output amplitude (V, peak)",
                              DEFAULTS["amplitude_V"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        field("series_R_ohm", "Series resistor (Ω)",
                              DEFAULTS["series_R_ohm"],
                              hint="Sets excitation current: I ≈ V / R.",
                              validators=[Number(minimum=1.0, failure_description="must be > 0")]),
                        id="mode_mfli_excitation",
                    )
                    yield card(
                        "Keithley 6221 AC current",
                        field("amplitude_values", "Excitation current (A, peak)",
                              DEFAULTS["amplitude_values"], kind="text",
                              hint="Ideal current source — no series resistor. Single value, "
                                   "or comma-separated list — one complete sweep runs per "
                                   "value (own 6221 re-arm), each saved to its own file."),
                        field("ac_compliance_V", "6221 voltage compliance (V)",
                              DEFAULTS["ac_compliance_V"],
                              validators=[Number(minimum=0.1, failure_description="must be > 0")]),
                        id="mode_6221_excitation",
                    )
                    yield card(
                        "Quantities",
                        switch_field(
                            "measure_rxx", "R_xx mode — follower reads R_xx's 1f "
                            "instead of R_xy's 2f",
                            DEFAULTS["measure_rxx"],
                        ),
                        Static(
                            "Only two physical MFLIs, so this trades 2f for R_xx — "
                            "move the follower's Signal Input cable by hand to match. "
                            "The '2f lock-in filter'/'2f input range' fields below "
                            "configure the follower either way.",
                            classes="hint",
                        ),
                        id="mode_6221_quantities",
                    )
                    yield card(
                        "Magnet & field sweep",
                        switch_field("enable_sweep", "Sweep magnetic field (Kepco magnet)",
                                     DEFAULTS["enable_sweep"]),
                        sweep_rows_field("sweep_rows", DEFAULTS["sweep_rows"]),
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
                    yield card(
                        "Sample geometry & field direction (optional)",
                        field("hall_bar_length_um", "Hall bar length (µm)",
                              DEFAULTS["hall_bar_length_um"], kind="text", valid_empty=True,
                              hint="Current-path length between voltage probes. Leave blank if "
                                   "unknown — doesn't block the run."),
                        field("hall_bar_width_um", "Hall bar width (µm)",
                              DEFAULTS["hall_bar_width_um"], kind="text", valid_empty=True),
                        field("hall_bar_thickness_nm", "Film/channel thickness (nm)",
                              DEFAULTS["hall_bar_thickness_nm"], kind="text", valid_empty=True),
                        field("field_theta_deg", "θ — tilt from out-of-plane (°)",
                              DEFAULTS["field_theta_deg"], kind="number", valid_empty=True,
                              validators=[Number(0, 180, failure_description="0-180°")],
                              hint="0° = fully out-of-plane (film normal), 90° = in-plane."),
                        field("field_phi_deg", "φ — azimuth from current axis (°)",
                              DEFAULTS["field_phi_deg"], kind="number", valid_empty=True,
                              validators=[Number(0, 360, failure_description="0-360°")],
                              hint="Meaningless when θ=0°."),
                        Horizontal(
                            Button("xy", id="plane_xy", classes="plane-btn"),
                            Button("zx", id="plane_zx", classes="plane-btn"),
                            Button("zy", id="plane_zy", classes="plane-btn"),
                            classes="plane-btn-row",
                        ),
                        Static(render_ascii_field_diagram(None, None),
                               id="field_diagram", classes="field-diagram"),
                    )

                # ── Tier 2: precision / speed knobs — collapsed ─────────────
                with Collapsible(title="Acquisition & filter settings", collapsed=True):
                    with Vertical(classes="param-grid"):
                        yield card(
                            "1f lock-in filter",
                            field("time_constant_1f_s", "Filter time constant (s)",
                                  DEFAULTS["time_constant_1f_s"],
                                  hint="Bigger = quieter but slower & longer settling.",
                                  validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                            select_field("order_1f", "Filter order", list(range(1, 9)),
                                         int(DEFAULTS["order_1f"])),
                            switch_field("sinc_filter_1f", "Sinc filter (extra harmonic rejection)",
                                         DEFAULTS["sinc_filter_1f"]),
                        )
                        yield card(
                            "2f lock-in filter",
                            field("time_constant_2f_s", "Filter time constant (s)",
                                  DEFAULTS["time_constant_2f_s"],
                                  hint="2f bleed-through from 1f is the usual reason "
                                       "this needs a longer TC / higher order than 1f.",
                                  validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                            select_field("order_2f", "Filter order", list(range(1, 9)),
                                         int(DEFAULTS["order_2f"])),
                            switch_field("sinc_filter_2f", "Sinc filter (extra harmonic rejection)",
                                         DEFAULTS["sinc_filter_2f"]),
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

                # ── Tier 3: instrument wiring — collapsed ───────────────────
                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Devices & connection",
                            field("leader_device", "Leader MFLI (1f; the source in MFLI mode)",
                                  DEFAULTS["leader_device"], kind="text"),
                            field("follower_device", "Follower MFLI (2f, or R_xx 1f in R_xx mode)",
                                  DEFAULTS["follower_device"], kind="text"),
                            field("daq_host", "LabOne data server host",
                                  DEFAULTS["daq_host"], kind="text"),
                            field("daq_port", "LabOne data server port",
                                  DEFAULTS["daq_port"], kind="integer"),
                            muted=True,
                        )
                        yield card(
                            "6221 & ExtRef (phase marker → both MFLIs' Aux In)",
                            field("ac_visa_resource", "6221 VISA resource",
                                  DEFAULTS["ac_visa_resource"], kind="text"),
                            field("phasemarker_line", "6221 Trigger Link phase-marker pin",
                                  DEFAULTS["phasemarker_line"], kind="integer",
                                  hint="Confirm your unit's factory default before assuming.",
                                  validators=[Number(minimum=1, maximum=6,
                                                     failure_description="must be 1-6")]),
                            field("extref_lock_timeout_s", "ExtRef PLL lock timeout (s)",
                                  DEFAULTS["extref_lock_timeout_s"]),
                            field("leader_extref_index", "Leader ExtRef module index",
                                  DEFAULTS["leader_extref_index"], kind="integer"),
                            field("leader_aux_input_ch", "Leader Aux In channel (0 = Aux In 1)",
                                  DEFAULTS["leader_aux_input_ch"], kind="integer"),
                            field("leader_osc_index", "Leader oscillator index",
                                  DEFAULTS["leader_osc_index"], kind="integer"),
                            field("leader_pll_demod_index", "Leader PLL phase-detector demod index",
                                  DEFAULTS["leader_pll_demod_index"], kind="integer",
                                  hint="Must differ from demod 0 (used for the real 1f signal) — "
                                       "extrefs/N/adcselect is read-only on real firmware, this "
                                       "demod's OWN adcselect is what actually selects Aux In.",
                                  validators=[Number(minimum=0, failure_description="must be ≥ 0")]),
                            select_field("leader_automode", "Leader PLL bandwidth adaptation",
                                         six.AUTOMODE_OPTIONS, int(DEFAULTS["leader_automode"]),
                                         hint=six.AUTOMODE_HINT),
                            field("follower_extref_index", "Follower ExtRef module index",
                                  DEFAULTS["follower_extref_index"], kind="integer"),
                            field("follower_aux_input_ch", "Follower Aux In channel (0 = Aux In 1)",
                                  DEFAULTS["follower_aux_input_ch"], kind="integer"),
                            field("follower_osc_index", "Follower oscillator index",
                                  DEFAULTS["follower_osc_index"], kind="integer"),
                            field("follower_pll_demod_index", "Follower PLL phase-detector demod index",
                                  DEFAULTS["follower_pll_demod_index"], kind="integer",
                                  hint="Must differ from demod 0 (used for the real 2f signal).",
                                  validators=[Number(minimum=0, failure_description="must be ≥ 0")]),
                            select_field("follower_automode", "Follower PLL bandwidth adaptation",
                                         six.AUTOMODE_OPTIONS, int(DEFAULTS["follower_automode"]),
                                         hint=six.AUTOMODE_HINT),
                            muted=True, id="mode_6221_extref",
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
                            field("field_settle_tolerance_mT", "Field-settle tolerance (mT)",
                                  DEFAULTS["field_settle_tolerance_mT"],
                                  hint="Advanced: after each magnet step, wait until a short "
                                       "window of gaussmeter readings spans less than this before "
                                       "the settling time above. Raise if points stall.",
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
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

            with Vertical(id="sidebar"):
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start measurement  (F5)", id="start", variant="success")
        yield Footer()

    # ── Form state I/O ───────────────────────────────────────────────────────

    # ── Reactivity ───────────────────────────────────────────────────────────

    def update_summary(self) -> None:
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

        theta = None if parse_errors else state.get("field_theta_deg")
        phi = None if parse_errors else state.get("field_phi_deg")
        self.query_one("#field_diagram", Static).update(render_ascii_field_diagram(theta, phi))

    # ── Start ────────────────────────────────────────────────────────────────

    def _build_plan(self, state: dict):
        return build_plan(state, self.data_root)

    # ── AC-source toggle ─────────────────────────────────────────────────────

    def _read_settings(self) -> dict:
        """This form's settings, falling back to the former 6221-program form's
        (key by key) — so neither source's last values are lost by the merge.
        A file saved before the toggle existed picks the source of whichever
        of the two forms was used last."""
        files = [(path, path.stat().st_mtime) for path in (six.SETTINGS_PATH, SETTINGS_PATH)
                 if path.is_file()]
        merged: dict = {}
        for path, _ in files:
            try:
                merged.update(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
        if merged and "ac_source" not in merged:
            newest = max(files, key=lambda f: f[1])[0]
            merged["ac_source"] = "6221" if newest == six.SETTINGS_PATH else "mfli"
        if self._forced_source:
            merged["ac_source"] = self._forced_source
        return merged

    def on_mount(self) -> None:
        super().on_mount()
        if self._forced_source:
            self.query_one("#ac_source", Select).value = self._forced_source
        self._show_source(self.query_one("#ac_source", Select).value)

    def _show_source(self, source) -> None:
        for mode, widget_ids in self.SOURCE_WIDGETS.items():
            for widget_id in widget_ids:
                self.query_one(f"#{widget_id}").display = mode == source

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "ac_source":
            self._show_source(event.value)
        super().on_select_changed(event)

    def parse_state(self) -> tuple[dict, list[str]]:
        state, errors = super().parse_state()
        return state, mode_errors(state, errors)

    def run_screen(self, plan):
        return engine(plan).RunScreen(plan)


def main() -> None:
    MFLIDualHarmonicApp().run()


if __name__ == "__main__":
    main()
