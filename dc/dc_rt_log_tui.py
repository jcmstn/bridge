#!/usr/bin/env python3
"""
Textual TUI front-end for dc_rt_log.py
======================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-25

Form for the resistance-vs-temperature log: a fixed ±I reversal sample
(6221 + 2182), taken back to back while the temperature drifts on its own and
the MercuryiTC is only read. The sidebar shows the modelled time per sample
(shorter = less temperature drift inside each point) and the run's upper
bound (the maximum duration). Stop ends the log normally, and the run is
saved as "completed".

At connect time the MercuryiTC sensors are probed verbosely (as in every
program: connect_temperature_controller -> probe_temperature_sensors logs the
identity, board catalog, and each sensor's exact query + raw reply). If two
sensors are configured but only one answers, the log continues with that one,
and the plots show one R-vs-T panel per sensor that is reading.

Optional fixed field ("Apply fixed field" switch, off by default): the Kepco
is ramped to one magnet current before the log starts, held there for the
whole log, and ramped to zero on shutdown (Stop, error, or normal end). The
Lake Shore 475 is read once per sample, after the R/T window, into
magnet_field_mT; the hand-entered field direction θ/φ goes into every row.

"Read" toggle (a merged form, docs/architecture.md §2): "6221 + 2182" is the
DC ±I log above (type RT, this module is its engine); "6221 AC + MFLI pair"
logs two voltage pairs at once — e.g. R_xy on the leader and R_xx on the
follower — with the 6221 sine ExtRef-locking both MFLIs as in HARM6 (type
RTAC, engine mfli/mfli_rt_log_tui.py). The time / stop / temperature / fixed
field part of the form is shared; each mode's own cards are hidden in the
other.

Run with:
    python dc_rt_log_tui.py
"""

from __future__ import annotations

import logging
import math
import sys
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.validation import Number
from textual.widgets import Button, Collapsible, Footer, Header, Select, Static

import mfli.mfli_dual_harmonic_6221_tui as six
import mfli.mfli_rt_log_tui as lockin_engine
from dc.dc_rt_log import (
    AcquisitionConfig,
    GaussmeterConfig,
    MagnetConfig,
    SourceConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    connect_gaussmeter,
    connect_magnet,
    connect_source,
    connect_temperature_controller,
    connect_voltmeter,
    live_plot_worker,
    plot_results,
    ramp_current_to_zero,
    run_measurement,
    set_magnet_current,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_source,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import check_sweep_size, safe_shutdown
from instruments.data_dir import validate_directory
from instruments.field_geometry import field_direction_summary_line, render_ascii_field_diagram
from instruments.data_naming import RunContext, allocate_run, preview_raw_filename, record_run
from instruments.keithley2182 import read_time_s
from instruments.keithley6221 import ACSourceConfig, ac_source_restart_s, reversal_avg_s
from instruments.kepco_magnet import magnet_move_s
from instruments.lakeshore475 import read_field_s
from instruments.mfli_daq import ExtRefConfig, acquire_s
from instruments.run_time import (
    GPIB_TXN_S, MDS_SYNC_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S, RunCost, format_duration,
)
from instruments.summary_lines import summary_markup
from instruments.mercury_itc import normalize_uid, set_temperature
from instruments.temperature_control import (
    TEMPERATURE_DEFAULTS, TEMPERATURE_FIELD_IDS, TEMPERATURE_TEXT_FIELDS,
    steps_from_state, temperature_cfg_from_state, temperature_summary,
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
    temperature_card,
    switch_field,
)
from mfli.mfli_dual_harmonic import DemodConfig, FilterConfig
from mfli.mfli_dual_harmonic_6221 import _AC_COMPLIANCE_CEILING_V, _AC_CURRENT_CEILING_A, extref_lock_s
from instruments.tui_sample_picker import NEW_SAMPLE_SENTINEL

log = logging.getLogger("dc_rt_log_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "dc_rt_log_tui_settings.json"

# Locked type code (see docs/data_convention.md) — never deviates.
MEASUREMENT_TYPE = "RT"
PNG_SUFFIX = "R_vs_T"

DC_RT_LOG_DESCRIPTION = (
    "Continuous resistance log while the temperature drifts freely: ±I reversal (6221 + "
    "2182, RT) or 6221 AC + MFLI pair for two probe pairs at once, e.g. R_xx + R_xy (RTAC). "
    "MercuryiTC read before and after every sample (drift saved per point). Optionally "
    "holds a fixed magnet current, B read every sample."
)

DC_RT_LOG_SCHEMATIC = """\
  KEITHLEY 6221  (current source: fixed ±I, or a continuous AC sine)
    Output ──▶ current path through the sample ── common ground

  KEITHLEY 2182  (nanovoltmeter)                              [Read: 6221 + 2182]
    Channel 1 (differential) ──▶ the voltage probe pair of interest

  6221 AC sine + MFLI pair                                    [Read: 6221 AC + MFLIs]
    6221 phase marker ──▶ Aux In 1 on BOTH MFLIs (equal cables)
    Leader   Signal Input 1 ──▶ e.g. R_xy pair  ·  Follower Signal Input 1 ──▶ R_xx pair
    MDS: Clock 10 MHz Out ──▶ follower Clock In; Trigger Out 1 ──▶ Trigger In 1 (both)

  MercuryiTC  (optional, read only — no temperature control)
    LAN ──▶ 1 or 2 temperature sensors

  Fixed field  (optional, "Apply fixed field" switch)
    Kepco BOP-GL  ──GPIB──▶ electromagnet coil (held at one current)
    Lake Shore 475 ──GPIB──▶ Hall probe at the sample (read after each sample)
"""

DEFAULTS: dict = {
    "source_visa_resource": "GPIB0::20::INSTR",
    "voltmeter_visa_resource": "GPIB0::7::INSTR",
    "sense_current_A": "0.0001",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "nplc": "1",
    "auto_range": True,
    "n_reversals": "5",
    "interval_s": "0",
    "max_duration_min": "240",
    "T_stop_K": "",
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "",
    **TEMPERATURE_DEFAULTS,
    "enable_field": False,
    "magnet_current_A": "0",
    "field_theta_deg": "",
    "field_phi_deg": "",
    "magnet_visa_resource": "GPIB0::6::INSTR",
    "current_limit_A": "35",
    "voltage_compliance_V": "15.0",
    "ramp_step_A": "0.1",
    "ramp_delay_s": "0.05",
    "field_settle_tolerance_mT": "0.02",
    "gaussmeter_visa_resource": "GPIB0::12::INSTR",
    "gaussmeter_n_averages": "1",
    "gaussmeter_read_delay_s": "0.05",
}

# ── Read toggle ── "dc": the ±I log (this module, RT). "lockin": 6221 AC +
# MFLI pair (mfli/mfli_rt_log_tui.py, RTAC) — its fields are HARM6's, minus
# the field sweep / geometry / multi-amplitude ones; one excitation current.
READ_MODES = [("6221 + 2182 (DC ±I) → RT", "dc"),
              ("6221 AC + MFLI pair (lock-in, 2 probe pairs) → RTAC", "lockin")]
LOCKIN_KEYS = (
    "leader_device", "follower_device", "daq_host", "daq_port",
    "frequency_Hz", "ac_compliance_V", "phasemarker_line",
    "leader_harmonic", "leader_measure_rxx", "follower_harmonic", "measure_rxx",
    "leader_phi_I_deg", "follower_phi_I_deg",
    "time_constant_1f_s", "order_1f", "sinc_filter_1f",
    "time_constant_2f_s", "order_2f", "sinc_filter_2f",
    "differential", "ac_coupling", "input_range_1f_V", "input_range_2f_V", "sample_rate_Hz",
    "settling_time_s", "n_averages",
    *(f"{role}_{k}" for role in ("leader", "follower")
      for k in ("extref_index", "aux_input_ch", "osc_index", "pll_demod_index", "automode")),
    "extref_lock_timeout_s",
)
LOCKIN_DEFAULTS = {
    **{k: six.DEFAULTS[k] for k in LOCKIN_KEYS},
    "ac_amplitude_A": "1e-5",
    "follower_harmonic": "1", "measure_rxx": True,     # leader 1f_* (R_xy), follower rxx_1f_*
}
DC_ONLY_FIELDS = ("voltmeter_visa_resource", "sense_current_A", "compliance_V",
                  "source_delay_s", "nplc", "auto_range", "n_reversals")
DEFAULTS.update(LOCKIN_DEFAULTS, read_mode="dc")

NUMERIC_FIELDS: dict = {
    "sense_current_A": float,
    "compliance_V": float,
    "source_delay_s": float,
    "nplc": float,
    "n_reversals": int,
    "interval_s": float,
    "max_duration_min": float,
    "magnet_current_A": float,
    "current_limit_A": float,
    "voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "field_settle_tolerance_mT": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
    "temperature_max_K": float,         # single target: no margin / hold fields
    "ac_amplitude_A": float,
    **{k: six.NUMERIC_FIELDS[k] for k in LOCKIN_KEYS if k in six.NUMERIC_FIELDS},
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "device", "cooldown",
               "leader_device", "follower_device", "daq_host",
               *TEMPERATURE_TEXT_FIELDS,
               "magnet_visa_resource", "gaussmeter_visa_resource", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K", "T_stop_K", "field_theta_deg", "field_phi_deg"]
MAGNET_FIELD_IDS = [
    "magnet_current_A", "field_theta_deg", "field_phi_deg",
    "magnet_visa_resource", "current_limit_A", "voltage_compliance_V",
    "ramp_step_A", "ramp_delay_s", "field_settle_tolerance_mT",
    "gaussmeter_visa_resource", "gaussmeter_n_averages", "gaussmeter_read_delay_s",
]


def _magnet_cfg(state: dict) -> MagnetConfig:
    return MagnetConfig(
        visa_resource=state["magnet_visa_resource"],
        current_limit_A=state["current_limit_A"],
        voltage_compliance_V=state["voltage_compliance_V"],
        ramp_step_A=state["ramp_step_A"],
        ramp_delay_s=state["ramp_delay_s"],
    )


def _gauss_cfg(state: dict) -> GaussmeterConfig:
    return GaussmeterConfig(
        visa_resource=state["gaussmeter_visa_resource"],
        n_averages=state["gaussmeter_n_averages"],
        read_delay_s=state["gaussmeter_read_delay_s"],
    )


def _has_temperature(state: dict) -> bool:
    return bool(state.get("enable_temperature")) and bool(parse_sensor_uids(state["temperature_sensor_uids"]))


def _is_lockin(state: dict) -> bool:
    return state.get("read_mode") == "lockin"


def mode_errors(state: dict, errors: list[str]) -> list[str]:
    """Parse errors of the ACTIVE read mode only — a hidden field of the
    other mode never blocks a run."""
    hidden = DC_ONLY_FIELDS if _is_lockin(state) else tuple(LOCKIN_DEFAULTS)
    return [e for e in errors if not any(e.startswith(f"'{f}'") for f in hidden)]


def sample_time_s(state: dict) -> float:
    """Modelled wall time of one sample: T read, the read itself, T read.
    DC: the ±I reversal. Lock-in: MDS + 2 PLL checks, then one shared poll
    window for both MFLIs (the longer of the two)."""
    if _is_lockin(state):
        rate, n = state["sample_rate_Hz"], state["n_averages"]
        t = 3 * GPIB_TXN_S + (max(acquire_s(state["time_constant_1f_s"], n, rate),
                                  acquire_s(state["time_constant_2f_s"], n, rate)) if rate > 0 else 0.0)
    else:
        t = reversal_avg_s(state["n_reversals"], state["source_delay_s"], read_time_s(state["nplc"]))
    return t + (2 * TEMP_READ_S if _has_temperature(state) else 0.0)


def sample_period_s(state: dict) -> float:
    """Start-to-start spacing: the sample, plus (fixed field) the field read
    that follows it outside the R/T window, or `interval_s` if longer."""
    t = sample_time_s(state) + POINT_OVERHEAD_S
    if state.get("enable_field"):
        t += read_field_s(_gauss_cfg(state))
    return max(state["interval_s"], t)


def max_samples(state: dict) -> int:
    """Samples the run takes if it runs to its maximum duration."""
    return max(1, math.ceil(state["max_duration_min"] * 60.0 / sample_period_s(state)))


def run_costs(state: dict) -> RunCost:
    """Upper-bound cost: the run at its maximum duration (Stop or T_stop end it sooner)."""
    n = max_samples(state)
    check_sweep_size(n)
    rc = RunCost(n)
    rc.each("samples", sample_period_s(state))
    rc.at("per-run", PER_RUN_S, 0)
    if _is_lockin(state):       # MDS, 6221 on, both PLLs locked, filters settled once
        rc.at("MDS + 6221 + ExtRef + settle", MDS_SYNC_S + ac_source_restart_s() + extref_lock_s()
              + state["settling_time_s"], 0)
    else:
        rc.tail("ramp", max(1, int(abs(state["sense_current_A"]) / 1e-4)) * 0.02)
    if state.get("enable_field"):       # park the magnet before sample 1, ramp it down at the end
        mcfg = _magnet_cfg(state)
        typ = magnet_move_s(state["magnet_current_A"], mcfg)
        rc.at("magnet", typ, 0)
        rc.tail("ramps", magnet_move_s(state["magnet_current_A"], mcfg, with_field=False))
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    acq_cfg: AcquisitionConfig
    sample: str
    device: str
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    temp_cfg: Optional[TemperatureControllerConfig] = None
    data_root: Path = _DEFAULT_DATA_DIR
    run_cost: Optional[RunCost] = None
    total_points: int = 1               # upper bound (max duration) — progress bar only
    series: str = ""
    magnet_cfg: Optional[MagnetConfig] = None       # None = no fixed field
    gauss_cfg: Optional[GaussmeterConfig] = None
    magnet_current_A: Optional[float] = None
    field_settle_tolerance_mT: Optional[float] = None
    field_theta_deg: Optional[float] = None
    field_phi_deg: Optional[float] = None
    temperature_target_K: Optional[float] = None    # sent to the iTC at start (no wait)


def build_header_fields(plan: MeasurementPlan, ctx: RunContext, records: list[dict], *,
                        status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """Universal header/index fields plus the log's own: the measured
    temperature range, and the sensor UIDs actually read (`extra`, set at
    connect time once the sensors have been probed)."""
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
    fields["T_min_K"] = min(measured) if measured else ""
    fields["T_max_K"] = max(measured) if measured else ""
    if extra:
        fields.update(extra)
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# Live validation / summary
# ─────────────────────────────────────────────────────────────────────────────

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

    if not _is_lockin(state) and state["source_visa_resource"] == state["voltmeter_visa_resource"]:
        errors.append("Source (6221) and voltmeter (2182) VISA resources must be different.")
    n_identity_errors = len(errors)
    if _is_lockin(state):
        _lockin_checks(state, info, warnings, errors)
    else:
        if state["sense_current_A"] == 0:
            errors.append("Sense current must be non-zero (R = V_odd / I).")
        if state["compliance_V"] <= 0:
            errors.append("Compliance voltage must be > 0 V.")
        if state["n_reversals"] < 1:
            errors.append("±I pairs per sample must be ≥ 1.")
        elif state["n_reversals"] == 1:
            warnings.append("One ±I pair per sample gives no error bar (voltage_sem_V is nan).")
        info.append(f"Sense current: ±{format_si(abs(state['sense_current_A']), 'A')}, "
                    f"{state['n_reversals']} ±I pairs per sample")
    if state["interval_s"] < 0:
        errors.append("Sample interval must be ≥ 0 s.")
    if state["max_duration_min"] <= 0:
        errors.append("Maximum duration must be > 0 min.")
    if state["enable_field"]:
        if state["ramp_step_A"] <= 0:
            errors.append("Magnet ramp step must be > 0 A.")
        if state["gaussmeter_n_averages"] < 1:
            errors.append("Field readings averaged must be ≥ 1.")
        if abs(state["magnet_current_A"]) > state["current_limit_A"]:
            errors.append(f"Magnet current ({state['magnet_current_A']:g} A) exceeds the "
                          f"current limit ({state['current_limit_A']:g} A).")

    if len(errors) == n_identity_errors:        # timing parameters valid -> model the run
        per = sample_time_s(state)
        info.append(f"Per sample: ≈ {per:.2f} s — the temperature drift inside each point "
                    "is saved as temperature_N_drift_K")
        info.append("Sampling: back to back" if state["interval_s"] <= per
                    else f"Sampling: every {state['interval_s']:g} s")
        try:
            rc = run_costs(state)
            info.append(f"Stops after ≤ {format_duration(state['max_duration_min'] * 60)} "
                        f"(≤ {len(rc.points):,} samples), or on Stop")
        except ValueError as exc:
            errors.append(str(exc))
    if state.get("T_stop_K") is not None:
        if state["T_stop_K"] <= 0:
            errors.append("Stop temperature must be > 0 K.")
        elif not _has_temperature(state):
            warnings.append("Stop temperature is set but temperature logging is off — it never triggers.")
        else:
            info.append(f"Also stops once sensor 1 crosses {state['T_stop_K']:g} K")

    temperature_summary(state, info, warnings, errors, single_target=True)
    if not _has_temperature(state):
        warnings.append("No temperature logging — the log runs against time only.")

    if state["enable_field"]:
        if state["magnet_current_A"] == 0:
            warnings.append("Fixed field is on but the magnet current is 0 A.")
        info.append(f"Fixed field: magnet held at {state['magnet_current_A']:g} A for the whole log, "
                    f"ramped to 0 at the end; B read after each sample "
                    f"(Lake Shore 475, {state['gaussmeter_n_averages']} avg)")
        info.append(field_direction_summary_line(state.get("field_theta_deg"),
                                                 state.get("field_phi_deg")))
    else:
        info.append("Field: none — magnet untouched")

    return info, warnings, errors


def _lockin_checks(state: dict, info: list[str], warnings: list[str], errors: list[str]) -> None:
    """The lock-in mode's own summary lines — the 6221 limits and the MFLI
    pair, with HARM6's harmonic/naming and φ_I checks."""
    (_, lead), (_, fol) = six.state_naming(state)
    info.append(f"Lock-in: leader {lead} · follower {fol} (one sample = both, same poll window)")
    if state["leader_device"] == state["follower_device"]:
        errors.append("Leader and follower device IDs must be different.")
    amp = state["ac_amplitude_A"]
    if not 0 < amp <= _AC_CURRENT_CEILING_A:
        errors.append(f"Excitation current must be in (0, {format_si(_AC_CURRENT_CEILING_A, 'A')}] "
                      "— check for a mistyped exponent.")
    else:
        info.append(f"Excitation: {format_si(amp, 'A')} peak ({format_si(amp / math.sqrt(2), 'A')} rms) "
                    f"@ {state['frequency_Hz']:g} Hz; saved <prefix>_X_ohm = X / I_rms")
    if not 0 < state["ac_compliance_V"] <= _AC_COMPLIANCE_CEILING_V:
        errors.append(f"6221 compliance must be in (0, {_AC_COMPLIANCE_CEILING_V:g}] V.")
    six.harmonic_checks(state, warnings, errors)
    # no Measure-φ_I mode here: φ_I comes from the dual-harmonic form (6221, same frequency)
    six.phase_summary({**state, "measure_phi_I": False,
                       "phi_I_context": six.phi_I_context(state, "6221")}, "6221", info, warnings)
    for role, label in (("leader", lead), ("follower", fol)):
        if state[f"{role}_pll_demod_index"] == 0:
            errors.append(f"{role.title()} PLL phase-detector demod index must differ from 0 "
                          f"(demod 0 reads the real {label} signal).")
    for label, tc, order in ((lead, state["time_constant_1f_s"], state["order_1f"]),
                             (fol, state["time_constant_2f_s"], state["order_2f"])):
        if tc <= 0:
            errors.append(f"{label} time constant must be > 0 s.")
            continue
        k = 10 if order >= 3 else 5
        if state["settling_time_s"] < k * tc:
            warnings.append(f"Settle before the first sample {state['settling_time_s']:g} s < "
                            f"{k}×TC ({k * tc:g} s) for {label} — its first samples may not have settled.")
        if state["sample_rate_Hz"] <= 0:
            errors.append("Demodulator sample rate must be > 0.")
    if state["n_averages"] < 1:
        errors.append("Samples averaged per point must be ≥ 1.")


def compute_filename_preview(state: dict) -> Optional[str]:
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL or not state.get("device"):
        return None
    preview = preview_raw_filename(state["sample"], state["device"],
                                   lockin_engine.MEASUREMENT_TYPE if _is_lockin(state) else MEASUREMENT_TYPE,
                                   temperature_setpoint_K=state.get("temperature_setpoint_K"))
    return f"{preview}_<timestamp>.csv"


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (RunScreen calls this)."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    note = f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}" if comment else ""
    plot_results(pd.DataFrame(records), png_path, note=note)


# ─────────────────────────────────────────────────────────────────────────────
# Plan + run  ── pure, used by the TUI RunScreen
# ─────────────────────────────────────────────────────────────────────────────

def _common_plan_kw(state: dict, data_root: Path) -> dict:
    """The plan fields both read modes share: when to sample / stop, the
    iTC, the optional target, the fixed field, identity, run cost."""
    acq_cfg = AcquisitionConfig(
        n_reversals=state.get("n_reversals", 1),
        interval_s=state["interval_s"],
        max_duration_s=state["max_duration_min"] * 60.0,
        T_stop_K=state.get("T_stop_K"),
        output_file="",
    )
    # "Drive iTC to a target": parse_state/form_state already made the target
    # the filename setpoint (with_first_setpoint); run_plan sends it, no wait.
    target_K = state.get("temperature_setpoint_K") if steps_from_state(state) else None
    header_extra = {
        "interval_s": state["interval_s"],
        "max_duration_min": state["max_duration_min"],
        "T_stop_K": state.get("T_stop_K"),
    }
    field_kw: dict = {}
    if state["enable_field"]:
        field_kw = dict(
            magnet_cfg=_magnet_cfg(state), gauss_cfg=_gauss_cfg(state),
            magnet_current_A=state["magnet_current_A"],
            field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
            field_theta_deg=state.get("field_theta_deg"), field_phi_deg=state.get("field_phi_deg"),
        )
        header_extra.update(magnet_current_A=state["magnet_current_A"],
                            field_theta_deg=state.get("field_theta_deg"),
                            field_phi_deg=state.get("field_phi_deg"))
    run_cost = run_costs(state)
    return dict(
        acq_cfg=acq_cfg, sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state.get("temperature_setpoint_K"),
        cooldown=state["cooldown"], header_extra=header_extra,
        temp_cfg=temperature_cfg_from_state(state), data_root=data_root,
        temperature_target_K=target_K,
        run_cost=run_cost, total_points=len(run_cost.points),
        **field_kw,
    )


def _build_dc_plan(state: dict, data_root: Path) -> MeasurementPlan:
    kw = _common_plan_kw(state, data_root)
    kw["header_extra"] = {
        "sense_current_A": state["sense_current_A"],
        "compliance_V": state["compliance_V"],
        "n_reversals": state["n_reversals"],
        "nplc": state["nplc"],
        "source_delay_s": state["source_delay_s"],
        **kw["header_extra"],
    }
    return MeasurementPlan(
        src_cfg=SourceConfig(
            visa_resource=state["source_visa_resource"],
            sense_current_A=state["sense_current_A"],
            compliance_V=state["compliance_V"],
            source_delay_s=state["source_delay_s"],
        ),
        volt_cfg=VoltmeterConfig(
            visa_resource=state["voltmeter_visa_resource"],
            nplc=state["nplc"],
            auto_range=state["auto_range"],
        ),
        **kw,
    )


def _build_lockin_plan(state: dict, data_root: Path) -> "lockin_engine.MeasurementPlan":
    """The lock-in mode's plan — HARM6's 6221 / ExtRef / demod configs, one
    excitation current, demod 0 on each MFLI, phase = harmonic × φ_I."""
    kw = _common_plan_kw(state, data_root)
    extref = {role: ExtRefConfig(
        device=state[f"{role}_device"], extref_index=state[f"{role}_extref_index"],
        aux_input_ch=state[f"{role}_aux_input_ch"], osc_index=state[f"{role}_osc_index"],
        pll_demod_index=state[f"{role}_pll_demod_index"], automode=state[f"{role}_automode"],
    ) for role in ("leader", "follower")}
    phases = six.demod_phases(state)
    demods = [DemodConfig(
        device=state[f"{role}_device"], demod_index=0, harmonic=state[f"{role}_harmonic"],
        osc_index=state[f"{role}_osc_index"],
        differential=state["differential"], ac_coupling=state["ac_coupling"],
        input_range_V=state[f"input_range_{tag}_V"], sample_rate_Hz=state["sample_rate_Hz"],
        filter=FilterConfig(time_constant_s=state[f"time_constant_{tag}_s"],
                            order=state[f"order_{tag}"], sinc_filter=state[f"sinc_filter_{tag}"]),
        phase_deg=phase,
    ) for role, tag, phase in (("leader", "1f", phases[0]), ("follower", "2f", phases[1]))]
    kw["header_extra"] = {
        "excitation_frequency_Hz": state["frequency_Hz"],
        "excitation_amplitude_A": state["ac_amplitude_A"],
        "measure_rxx": state["measure_rxx"],
        **six.phase_header(state),
        "demod1_time_constant_s": state["time_constant_1f_s"],
        "demod1_order": state["order_1f"],
        "demod2_time_constant_s": state["time_constant_2f_s"],
        "demod2_order": state["order_2f"],
        "n_averages": state["n_averages"],
        "settling_time_s": state["settling_time_s"],
        **kw["header_extra"],
    }
    return lockin_engine.MeasurementPlan(
        daq_host=state["daq_host"], daq_port=state["daq_port"],
        leader=state["leader_device"], follower=state["follower_device"],
        ac_cfg=ACSourceConfig(
            visa_resource=state["source_visa_resource"],
            amplitude_A=state["ac_amplitude_A"],
            frequency_Hz=state["frequency_Hz"],
            compliance_V=state["ac_compliance_V"],
            phasemarker_line=state["phasemarker_line"],
        ),
        leader_extref_cfg=extref["leader"], follower_extref_cfg=extref["follower"],
        extref_lock_timeout_s=state["extref_lock_timeout_s"],
        demod1_cfg=demods[0], demod2_cfg=demods[1],
        leader_measure_rxx=state["leader_measure_rxx"], measure_rxx=state["measure_rxx"],
        n_averages=state["n_averages"], settling_time_s=state["settling_time_s"],
        **kw,
    )


def build_plan(state: dict, data_root: Path):
    """The active read mode's plan (its own MeasurementPlan type)."""
    return (_build_lockin_plan if _is_lockin(state) else _build_dc_plan)(state, data_root)


def engine(plan):
    """The module that runs `plan` — its run_plan / header / PNG / type code."""
    return lockin_engine if isinstance(plan, lockin_engine.MeasurementPlan) else sys.modules[__name__]


def _ignore(*_args) -> None:
    pass


def run_plan(plan: MeasurementPlan, stop_event: threading.Event, *,
             on_status: Callable[[str], None] = _ignore,
             on_run_label: Callable[[str], None] = _ignore,
             on_point: Callable[[dict], None] = _ignore,
             on_run_finished: Optional[Callable[[RunContext, list], None]] = None,
             run_contexts: Optional[list] = None,
             run_extras: Optional[list] = None) -> None:
    """Connect, probe the iTC sensors, park the magnet (fixed field), log one
    run until Stop / max duration / T_stop, finalize it, and always ramp the
    sense current and the magnet down and shut down."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = temp_ctrl = magnet = gaussmeter = None
    try:
        on_status("Connecting to Keithley 6221 & 2182 …")
        source = connect_source(plan.src_cfg)
        voltmeter = connect_voltmeter(plan.volt_cfg)

        loop_uid = normalize_uid(plan.temp_cfg.sensor_uids[0]) if plan.temp_cfg is not None else None
        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC and probing its sensors (see log) …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)   # narrows sensor_uids
        temp_cfg = plan.temp_cfg if temp_ctrl is not None else None
        if plan.temperature_target_K is not None:
            # load-bearing once asked for: raises if the iTC is unreachable or
            # the loop's own sensor (the first UID) was dropped by the probe
            if temp_cfg is None or loop_uid not in temp_cfg.sensor_uids:
                raise RuntimeError(f"Cannot send the temperature target: MercuryiTC loop sensor "
                                   f"{loop_uid} is not answering.")
            set_temperature(temp_ctrl, temp_cfg, plan.temperature_target_K, uid=loop_uid)
            on_status(f"MercuryiTC target {plan.temperature_target_K:g} K sent — logging while it drifts there …")

        if plan.magnet_cfg is not None:
            on_status("Connecting magnet power supply & gaussmeter …")
            magnet = connect_magnet(plan.magnet_cfg)
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)
            on_status(f"Ramping magnet to {plan.magnet_current_A:g} A and waiting for the field to settle …")
            set_magnet_current(magnet, plan.magnet_cfg, plan.magnet_current_A, gaussmeter,
                               plan.gauss_cfg, plan.field_settle_tolerance_mT, stop_event)
            if stop_event.is_set():     # stopped during the ramp: no empty run file
                return

        ctx = allocate_run(plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                           temperature_setpoint_K=plan.temperature_setpoint_K)
        extra = {"temperature_sensor_uids_used": ",".join(temp_cfg.sensor_uids) if temp_cfg else ""}
        run_contexts.append(ctx)
        run_extras.append(extra)
        on_run_label(f"Run #{ctx.run_str}")
        plan.acq_cfg.output_file = str(ctx.raw_path)

        on_status("Logging … (Stop ends the log and saves it as completed)")
        # stop_event is NOT passed to record_run: Stop is how a log normally
        # ends, so it finalizes as "completed" (an exception still -> "error").
        record_run(
            plan.data_root, ctx,
            lambda records, status: build_header_fields(plan, ctx, records, status=status,
                                                        comment="", extra=extra),
            lambda point_cb, write_csv: run_measurement(
                source, voltmeter, plan.src_cfg, plan.acq_cfg, stop_event=stop_event,
                on_point=point_cb, temp_ctrl=temp_ctrl, temp_cfg=temp_cfg, write_csv=write_csv,
                gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                magnet_current_A=plan.magnet_current_A,
                field_theta_deg=plan.field_theta_deg, field_phi_deg=plan.field_phi_deg),
            None, on_point=on_point, on_finished=on_run_finished)
    finally:
        if source is not None:
            safe_shutdown("source (ramp)", lambda: ramp_current_to_zero(source))
            safe_shutdown("source", lambda: shutdown_source(source))
        if magnet is not None:
            safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
        if gaussmeter is not None:
            safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


# ─────────────────────────────────────────────────────────────────────────────
# Run screen
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value: Optional[float], spec: str) -> str:
    return format(value, spec) if value is not None else "—"


class RunScreen(MeasurementRunScreen):
    ABORT_LABEL = "Stop logging"
    ABORT_STATUS = "Stop requested — finishing the current sample, then ramping current (and magnet) to zero …"
    DONE_STATUS = "Log finished."
    STOP_IS_NORMAL_END = True      # Stop ends a log normally -> "completed" in the run history
    POINT_STATUS = "Sample {n} (at most {total})."
    TABLE_COLUMNS = ("#", "t (s)", "T1 (K)", "ΔT1 (mK)", "T2 (K)", "R (Ω)", "σR (Ω)", "B (mT)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE
    PNG_SUFFIX = PNG_SUFFIX

    def live_plot_args(self):
        return (live_plot_worker,)

    def table_row(self, record: dict) -> tuple:
        drift = record.get("temperature_1_drift_K")
        return (
            str(record["point_index"] + 1),
            f"{record['elapsed_s']:.1f}",
            _fmt(record.get("temperature_1_K"), ".3f"),
            _fmt(drift * 1e3 if drift is not None else None, ".1f"),
            _fmt(record.get("temperature_2_K"), ".3f"),
            f"{record['resistance_ohm']:.6g}",
            f"{record['resistance_sem_ohm']:.2g}",
            _fmt(record.get("magnet_field_mT"), ".2f"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class DCRTLogApp(MeasurementApp):
    TITLE = "R vs T log"
    SUB_TITLE = "6221 + 2182 (±I) or 6221 AC + MFLI pair · temperature read only"

    data_root: Path = _DEFAULT_DATA_DIR
    TEMPERATURE_STEPS = False       # one target, sent by run_plan, no settle wait

    # widget id -> the read mode it is shown for (see compose)
    MODE_WIDGETS = {
        "mode_dc_sense": "dc", "mode_dc_acq": "dc", "mode_dc_addr": "dc", "mode_dc_timing": "dc",
        "mode_lockin_excitation": "lockin", "mode_lockin_leader": "lockin",
        "mode_lockin_follower": "lockin", "mode_lockin_phase": "lockin",
        "mode_lockin_filter_leader": "lockin", "mode_lockin_filter_follower": "lockin",
        "mode_lockin_input": "lockin", "mode_lockin_timing": "lockin",
        "mode_lockin_devices": "lockin", "mode_lockin_marker": "lockin", "mode_lockin_extref": "lockin",
    }

    def __init__(self, read_mode: Optional[str] = None) -> None:
        super().__init__()
        self._forced_mode = read_mode        # "lockin" from mfli/mfli_rt_log_tui.py

    SWITCH_DEPENDENTS = {
        "enable_temperature": tuple(TEMPERATURE_FIELD_IDS),
        "enable_field": tuple(MAGNET_FIELD_IDS),
    }

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 44; border-left: solid $primary; padding: 1 2; overflow-y: auto; }

    #identity_bar { height: auto; border: round $accent; padding: 1 2; margin-bottom: 1; }
    #filename_preview { text-style: bold; margin-bottom: 1; }
    #data_dir_row { height: 3; margin-bottom: 1; }
    #data_dir_row Input { width: 1fr; }
    #data_dir_row Button { margin-left: 1; }
    #identity_fields { layout: grid; grid-size: 4; grid-gutter: 0 2; height: auto; }
    #identity_fields > Vertical { height: auto; }

    .section-title { text-style: bold; color: $text-muted; margin: 1 0; }
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
    .field-label { text-style: bold; width: 100%; }
    .hint { text-style: italic; color: $text-muted; width: 100%; }
    .switch-row { height: auto; }
    .switch-row Label { padding-left: 1; content-align: left middle; width: 1fr; height: auto; min-height: 3; }
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
                yield identity_bar(DEFAULTS, _DEFAULT_DATA_DIR, self.data_root)

                with Vertical(classes="param-grid"):
                    yield card(
                        "Read",
                        select_field("read_mode", "What reads the sample", READ_MODES, DEFAULTS["read_mode"],
                                     hint="Lock-in: two probe pairs at once (e.g. R_xx + R_xy)."),
                    )
                    yield card(
                        "Sense current (Keithley 6221)",
                        field("sense_current_A", "Sense current (A)", DEFAULTS["sense_current_A"],
                              hint="Reversed ±I every pair. Keep it small enough to avoid self-heating."),
                        field("n_reversals", "±I pairs per sample", DEFAULTS["n_reversals"],
                              kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")],
                              hint="More = quieter, but a longer sample (more drift inside it)."),
                        id="mode_dc_sense",
                    )
                    yield card(
                        "Keithley 6221 AC current",
                        field("ac_amplitude_A", "Excitation current (A, peak)", DEFAULTS["ac_amplitude_A"],
                              hint="Continuous sine for the whole log. Mind self-heating."),
                        field("frequency_Hz", "Excitation frequency (Hz)", DEFAULTS["frequency_Hz"],
                              hint="~300-1000 Hz; avoid multiples of 50/60 Hz.",
                              validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                        field("ac_compliance_V", "6221 voltage compliance (V)", DEFAULTS["ac_compliance_V"],
                              validators=[Number(minimum=0.1, failure_description="must be > 0")]),
                        id="mode_lockin_excitation",
                    )
                    for role, harmonic_id, rxx_id in (("Leader", "leader_harmonic", "leader_measure_rxx"),
                                                      ("Follower", "follower_harmonic", "measure_rxx")):
                        yield card(
                            f"{role} MFLI",
                            select_field(harmonic_id, "Harmonic", six.HARMONIC_OPTIONS,
                                         int(DEFAULTS[harmonic_id]), hint="Saved as <h>f_* columns."),
                            switch_field(rxx_id, "R_xx — save as rxx_<h>f_*", DEFAULTS[rxx_id]),
                            Static("R_xx only renames the columns — wire this MFLI's Signal Input "
                                   "to the matching probe pair.", classes="hint"),
                            id=f"mode_lockin_{role.lower()}",
                        )
                    yield card(
                        "Reference phase (φ_I)",
                        field("leader_phi_I_deg", "Leader φ_I — current phase at its input, 1f (°)",
                              DEFAULTS["leader_phi_I_deg"]),
                        field("follower_phi_I_deg", "Follower φ_I — current phase at its input, 1f (°)",
                              DEFAULTS["follower_phi_I_deg"]),
                        Static("Measure once with the dual-harmonic form's Measure φ_I (6221 source, "
                               "same frequency) and copy the values here. Demod phase = harmonic × φ_I; "
                               "the sign of X / I depends on it.", classes="hint"),
                        id="mode_lockin_phase",
                    )
                    yield card(
                        "When to sample / stop",
                        field("interval_s", "Sample interval (s)", DEFAULTS["interval_s"],
                              validators=[Number(minimum=0.0, failure_description="must be ≥ 0")],
                              hint="0 = back to back."),
                        field("max_duration_min", "Maximum duration (min)", DEFAULTS["max_duration_min"],
                              validators=[Number(minimum=0.0, failure_description="must be > 0")],
                              hint="Safety stop for an unattended run."),
                        field("T_stop_K", "Stop at temperature (K, optional)", DEFAULTS["T_stop_K"],
                              valid_empty=True,
                              hint="Stops once sensor 1 crosses it, in either direction."),
                    )
                    yield temperature_card(
                        DEFAULTS,
                        field("temperature_sensor_uids", "Sensor UID(s)",
                              DEFAULTS["temperature_sensor_uids"], kind="text",
                              hint="1 or 2, e.g. MB1.T1, DB5.T1 — a missing one is dropped."),
                        single_target=True)
                    yield card(
                        "Fixed field (Kepco magnet, optional)",
                        switch_field("enable_field", "Apply fixed field", DEFAULTS["enable_field"]),
                        field("magnet_current_A", "Magnet current (A)", DEFAULTS["magnet_current_A"],
                              hint="Held for the whole log, ramped to 0 at the end."),
                        field("field_theta_deg", "θ — tilt from out-of-plane (°)",
                              DEFAULTS["field_theta_deg"], kind="number", valid_empty=True,
                              validators=[Number(0, 180, failure_description="0-180°")],
                              hint="0° = out-of-plane, 90° = in-plane."),
                        field("field_phi_deg", "φ — azimuth from current axis (°)",
                              DEFAULTS["field_phi_deg"], kind="number", valid_empty=True,
                              validators=[Number(0, 360, failure_description="0-360°")],
                              hint="0° = along current. Ignored when θ=0°."),
                        Horizontal(
                            Button("xy", id="plane_xy", classes="plane-btn"),
                            Button("zx", id="plane_zx", classes="plane-btn"),
                            Button("zy", id="plane_zy", classes="plane-btn"),
                            classes="plane-btn-row",
                        ),
                        Static(render_ascii_field_diagram(None, None),
                               id="field_diagram", classes="field-diagram"),
                    )

                with Collapsible(title="Acquisition & filter settings", collapsed=True):
                    with Vertical(classes="param-grid"):
                        yield card(
                            "Source & voltmeter",
                            field("compliance_V", "Compliance voltage (V)", DEFAULTS["compliance_V"],
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("nplc", "NPLC (integration time)", DEFAULTS["nplc"],
                                  hint="Bigger = quieter but slower.",
                                  validators=[Number(minimum=0.01, failure_description="must be > 0")]),
                            switch_field("auto_range", "Auto-range", DEFAULTS["auto_range"]),
                            id="mode_dc_acq",
                        )
                        for role, tag in (("Leader", "1f"), ("Follower", "2f")):
                            yield card(
                                f"{role} MFLI filter",
                                field(f"time_constant_{tag}_s", "Filter time constant (s)",
                                      DEFAULTS[f"time_constant_{tag}_s"], hint="Bigger = quieter but slower.",
                                      validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                                select_field(f"order_{tag}", "Filter order", list(range(1, 9)),
                                             int(DEFAULTS[f"order_{tag}"])),
                                switch_field(f"sinc_filter_{tag}", "Sinc filter (extra harmonic rejection)",
                                             DEFAULTS[f"sinc_filter_{tag}"]),
                                id=f"mode_lockin_filter_{role.lower()}",
                            )
                        yield card(
                            "MFLI inputs",
                            switch_field("differential", "Differential input (IN+/IN-)", DEFAULTS["differential"]),
                            switch_field("ac_coupling", "AC-couple the input", DEFAULTS["ac_coupling"]),
                            field("input_range_1f_V", "Leader input range (V)", DEFAULTS["input_range_1f_V"],
                                  validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                            field("input_range_2f_V", "Follower input range (V)", DEFAULTS["input_range_2f_V"],
                                  validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                            field("sample_rate_Hz", "Demodulator sample rate (Sa/s)", DEFAULTS["sample_rate_Hz"],
                                  validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                            id="mode_lockin_input",
                        )
                        yield card(
                            "Lock-in timing",
                            field("settling_time_s", "Settle before the first sample (s)",
                                  DEFAULTS["settling_time_s"],
                                  hint="Once, after the PLLs lock. ≥ 5×TC (order 1), ≥ 10×TC (order 3-4).",
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("n_averages", "Samples averaged per sample", DEFAULTS["n_averages"],
                                  kind="integer",
                                  validators=[Number(minimum=1, failure_description="must be ≥ 1")],
                                  hint="More = quieter, but a longer sample (more drift inside it)."),
                            id="mode_lockin_timing",
                        )

                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "2182 address",
                            field("voltmeter_visa_resource", "2182 (voltage)",
                                  DEFAULTS["voltmeter_visa_resource"], kind="text"),
                            muted=True, id="mode_dc_addr",
                        )
                        yield card(
                            "MFLI devices & connection",
                            field("leader_device", "Leader MFLI", DEFAULTS["leader_device"], kind="text"),
                            field("follower_device", "Follower MFLI", DEFAULTS["follower_device"], kind="text"),
                            field("daq_host", "LabOne data server host", DEFAULTS["daq_host"], kind="text"),
                            field("daq_port", "LabOne data server port", DEFAULTS["daq_port"], kind="integer"),
                            muted=True, id="mode_lockin_devices",
                        )
                        yield card(
                            "6221 phase marker → both MFLIs",
                            field("phasemarker_line", "6221 Trigger Link phase-marker pin",
                                  DEFAULTS["phasemarker_line"], kind="integer",
                                  validators=[Number(minimum=1, maximum=6, failure_description="must be 1-6")]),
                            field("extref_lock_timeout_s", "Reference lock timeout (s)",
                                  DEFAULTS["extref_lock_timeout_s"]),
                            muted=True, id="mode_lockin_marker",
                        )
                        yield card(
                            "MFLI ExtRef (phase marker → both MFLIs' Aux In)",
                            *(group for role in ("leader", "follower") for group in (
                                field(f"{role}_extref_index", f"{role.title()} ExtRef module index",
                                      DEFAULTS[f"{role}_extref_index"], kind="integer"),
                                field(f"{role}_aux_input_ch", f"{role.title()} Aux In channel (0 = Aux In 1)",
                                      DEFAULTS[f"{role}_aux_input_ch"], kind="integer"),
                                field(f"{role}_osc_index", f"{role.title()} oscillator index",
                                      DEFAULTS[f"{role}_osc_index"], kind="integer"),
                                field(f"{role}_pll_demod_index", f"{role.title()} PLL phase-detector demod index",
                                      DEFAULTS[f"{role}_pll_demod_index"], kind="integer",
                                      hint="Not 0 (demod 0 reads the signal).",
                                      validators=[Number(minimum=0, failure_description="must be ≥ 0")]),
                                select_field(f"{role}_automode", f"{role.title()} PLL bandwidth adaptation",
                                             six.AUTOMODE_OPTIONS, int(DEFAULTS[f"{role}_automode"]),
                                             hint=six.AUTOMODE_HINT),
                            )),
                            muted=True, id="mode_lockin_extref",
                        )
                        yield card(
                            "Instrument addresses",
                            field("source_visa_resource", "6221 (current source, both modes)",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            field("temperature_visa_resource", "MercuryiTC",
                                  DEFAULTS["temperature_visa_resource"], kind="text"),
                            field("magnet_visa_resource", "Magnet (Kepco)",
                                  DEFAULTS["magnet_visa_resource"], kind="text"),
                            field("gaussmeter_visa_resource", "Gaussmeter (Lake Shore 475)",
                                  DEFAULTS["gaussmeter_visa_resource"], kind="text"),
                            muted=True,
                        )
                        yield card(
                            "Magnet & gaussmeter",
                            field("current_limit_A", "Magnet software current limit (A)",
                                  DEFAULTS["current_limit_A"], hint="Hard safety ceiling."),
                            field("voltage_compliance_V", "Magnet voltage compliance (V)",
                                  DEFAULTS["voltage_compliance_V"]),
                            field("ramp_step_A", "Magnet ramp step (A)", DEFAULTS["ramp_step_A"]),
                            field("ramp_delay_s", "Magnet ramp delay (s)", DEFAULTS["ramp_delay_s"]),
                            field("field_settle_tolerance_mT", "Field-settle tolerance (mT)",
                                  DEFAULTS["field_settle_tolerance_mT"],
                                  hint="Before logging starts: settled when readings span less than this.",
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("gaussmeter_n_averages", "Field readings averaged per sample",
                                  DEFAULTS["gaussmeter_n_averages"], kind="integer",
                                  validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                            field("gaussmeter_read_delay_s", "Delay between field readings (s)",
                                  DEFAULTS["gaussmeter_read_delay_s"]),
                            muted=True,
                        )
                        yield card(
                            "Source timing",
                            field("source_delay_s", "6221 source delay (s)", DEFAULTS["source_delay_s"],
                                  hint="Settle after each polarity flip."),
                            muted=True, id="mode_dc_timing",
                        )

            with Vertical(id="sidebar"):
                yield Static("Description", classes="sidebar-title")
                yield Static(DC_RT_LOG_DESCRIPTION, classes="card-desc")
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start log  (F5)", id="start", variant="success")
        yield Footer()

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
            else "[dim]File:  (choose a sample and device to preview the filename)[/dim]")
        self.query_one("#summary", Static).update(summary_markup(info, warnings, errors))
        self.query_one("#start", Button).disabled = bool(errors)
        theta = None if parse_errors else state.get("field_theta_deg")
        phi = None if parse_errors else state.get("field_phi_deg")
        self.query_one("#field_diagram", Static).update(render_ascii_field_diagram(theta, phi))

    def _build_plan(self, state: dict):
        return build_plan(state, self.data_root)

    # ── Read toggle ──────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        super().on_mount()
        if self._forced_mode:
            self.query_one("#read_mode", Select).value = self._forced_mode
        self._show_modes()

    def _show_modes(self) -> None:
        mode = self.query_one("#read_mode", Select).value
        for widget_id, shown_for in self.MODE_WIDGETS.items():
            self.query_one(f"#{widget_id}").display = mode == shown_for

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "read_mode":
            self._show_modes()
        super().on_select_changed(event)

    def parse_state(self) -> tuple[dict, list[str]]:
        state, errors = super().parse_state()
        return state, mode_errors(state, errors)

    def run_screen(self, plan):
        return engine(plan).RunScreen(plan)


def main() -> None:
    DCRTLogApp().run()


if __name__ == "__main__":
    main()
