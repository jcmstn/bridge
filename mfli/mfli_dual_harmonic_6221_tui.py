#!/usr/bin/env python3
"""
Textual TUI front-end for mfli_dual_harmonic_6221.py
======================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-14

Same TUI as mfli_dual_harmonic_tui.py — same filters/timing/magnet-sweep/
temperature/phase-cal/geometry parameter surface — with the excitation
section swapped for the 6221 AC source + dual ExtRef lock (see
mfli_dual_harmonic_6221.py's module docstring for why BOTH MFLIs must have
their Aux In 1 wired to the 6221's phase marker, not just the leader's).

Run with:
    python mfli_dual_harmonic_6221_tui.py

Requirements:
    pip install textual matplotlib  (in addition to mfli_dual_harmonic_6221.py's own deps)
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import textwrap
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    Collapsible,
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
    TextArea,
)

from dc.dc_sweep_utils import build_segmented_sweep, parse_sweep_rows, parse_value_list, safe_shutdown
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
from instruments.kepco_magnet import magnet_move_s
from instruments.keithley6221 import ac_source_restart_s
from instruments.lakeshore475 import read_field_s
from instruments.live_plot import start_live_plot
from instruments.mfli_daq import acquire_s, poll_window_s
from instruments.run_time import (
    GPIB_TXN_S, MDS_SYNC_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
    RunCost, progress_step, progress_total,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
    NewSampleScreen,
    StatusCommentScreen,
    sample_options,
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
AUTOMODE_HINT = ("2=most forgiving acquisition (marginal/noisy signal), "
                 "3=fastest tracking once locked, 4=auto-adapts (default).")

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


def parse_sensor_uids(raw: str) -> tuple:
    """Parse a comma-separated "MB1.T1, DB5.T1" field into a 1- or 2-tuple of UIDs."""
    uids = [u.strip() for u in raw.split(",") if u.strip()]
    return tuple(uids[:2])


def follower_naming(measure_rxx: bool) -> tuple:
    """The follower device's column prefix and display label: today's R_xy
    2f (default) when off, R_xx's 1f harmonic when on -- see
    mfli_dual_harmonic_6221.py's module docstring for why only one of the
    two can be live at a time with just two physical MFLIs."""
    return ("rxx_1f", "R_xx (1f)") if measure_rxx else ("2f", "2f")


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
    measure_rxx: bool
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

def field(field_id: str, label_text: str, default: str, *, kind: str = "number",
          hint: str = "", validators=None, valid_empty: bool = False) -> list:
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


def sweep_rows_field(field_id: str, default: str) -> list:
    """One row per line, "start, stop, points" -- see parse_sweep_rows()."""
    label = Label("Sweep rows: start, stop, points (one per line)", classes="field-label")
    area = TextArea(default, id=field_id, classes="sweep-rows")
    hint = Label("Adjacent rows sharing a boundary value are merged, not duplicated.",
                 classes="hint")
    return [label, area, hint]


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


# ─────────────────────────────────────────────────────────────────────────────
# Live validation / derived-value summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (info, warnings, errors) for a fully-parsed state dict."""
    info: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    # ── Sample / run identity ───────────────────────────────────────────────
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

    # ── R_xx toggle ──────────────────────────────────────────────────────
    follower_prefix, follower_display = follower_naming(state["measure_rxx"])
    if state["measure_rxx"]:
        info.append(
            "R_xx mode: the follower reads R_xx's 1f instead of R_xy's 2f — "
            "move its Signal Input cable by hand to the R_xx probe pair "
            "before this run. 2f is unavailable in this mode (only two "
            "physical MFLIs); use this program with R_xx off for 1f/2f."
        )

    # ── Excitation (6221) ───────────────────────────────────────────────────
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
            info.append(f"Excitation currents {amp_list} A peak — {len(amp_list)} complete "
                        "sweeps (one 6221 re-arm each), one file set each.")
        elif amp_list:
            info.append(f"Excitation current I = {format_si(amp_list[0], 'A')} peak "
                         "(ideal 6221 current source)")
    if not 0 < state["ac_compliance_V"] <= _AC_COMPLIANCE_CEILING_V:
        errors.append(
            f"6221 compliance must be in (0, {_AC_COMPLIANCE_CEILING_V:g}] V; "
            f"got {state['ac_compliance_V']:g} V."
        )

    f = state["frequency_Hz"]
    freq_checks = [("1f", f)] if state["measure_rxx"] else [("1f", f), ("2f", 2 * f)]
    for label, check_f in freq_checks:
        for mains in (50, 60):
            nearest = round(check_f / mains) * mains
            if nearest > 0 and abs(check_f - nearest) < 0.5:
                warnings.append(
                    f"{label} ({check_f:g} Hz) is within 0.5 Hz of a {mains} Hz "
                    f"harmonic ({nearest} Hz) — mains pickup risk."
                )

    info.append(
        "Wiring requirement: the 6221's Trigger Link phase marker "
        f"(line {state['phasemarker_line']}) must reach Aux In "
        f"{state['leader_aux_input_ch'] + 1} on BOTH the leader AND the "
        "follower (BNC T / power divider, equal cable lengths) — a "
        "follower fed only via MDS will silently collapse the 2f signal "
        "toward zero once the two clocks drift apart. See the module "
        "docstring."
    )

    # ── PLL phase-detector demod ────────────────────────────────────────────
    # The real 1f/2f signal demod is fixed at index 0 (see _build_plan below)
    # — the PLL detector must be a different demod (extrefs/N/adcselect is
    # read-only on real firmware; see ExtRefConfig's docstring).
    if state["leader_pll_demod_index"] == 0:
        errors.append("Leader PLL phase-detector demod index must differ from 0 "
                       "(demod 0 reads the real 1f signal).")
    if state["follower_pll_demod_index"] == 0:
        errors.append(f"Follower PLL phase-detector demod index must differ from 0 "
                       f"(demod 0 reads the real {follower_display} signal).")

    # ── Filter / timing (leader and follower each get their own filter) ────
    acq_window_s = {"leader": 0.0, "follower": 0.0}
    for key, label, tc_key, order_key in (
        ("leader", "1f", "time_constant_1f_s", "order_1f"),
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
                info.append(f"{label} settling ≥ {settle_multiple}×TC ({recommended_settle:g} s) ✓")

            bw = 1.0 / (2 * math.pi * tc)
            min_rate = 4 * bw
            info.append(f"{label} filter noise bandwidth ≈ {bw:.3g} Hz")
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

    # ── Sweep ────────────────────────────────────────────────────────────────
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
        if state["measure_rxx"]:
            warnings.append(
                "Phase cal nulls the follower's phase against the leader's 1f "
                "reference — with R_xx mode on, that follower demod is R_xx's "
                "1f, not R_xy's 2f. Whether this is still the calibration you "
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
                       follower_prefix: str = "2f", follower_display: str = "2f",
                       multi: bool = False) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    try:
        fig.canvas.manager.set_window_title("MFLI live measurement (6221 AC source)")
    except Exception:
        pass
    ax1.set_ylabel("1f  R (V)")
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
                # One current: 1f and the follower keep their own colors, as in
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
            r1s.append(record["1f_R_V"])
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

    follower_prefix, follower_display = follower_naming(plan.measure_rxx if plan else False)
    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    ax1.plot(xs, [r["1f_R_V"] for r in records], "o-", color="tab:blue")
    ax2.plot(xs, [r[f"{follower_prefix}_R_V"] for r in records], "o-", color="tab:orange")
    ax1.set_ylabel("1f  R (V)")
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
        lines.append(f"Filter: 1f TC={tc1:g} s order={order1}, "
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
    #progress_row { height: auto; margin: 1 2; align: left middle; }
    #run_label { width: auto; padding: 0 2 0 0; text-style: bold; }
    #progress { margin: 0; }
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
        # One RunContext per amplitude iteration -- each gets its own run
        # number/file (see allocate_run() in do_run below).
        self._run_contexts: list[RunContext] = []
        # The LAST run's PNG, stashed by _save_run_png so _on_status_comment
        # can re-save it in place once the operator's comment is known.
        self._png_path: Optional[Path] = None
        # Set for real in on_mount(); the fallback here just documents the
        # default shape before that runs.
        self._follower_prefix: str = "2f"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Starting …", id="status_line")
        with Horizontal(id="progress_row"):
            yield Static("", id="run_label")
            yield ProgressBar(id="progress",
                              total=progress_total(self.plan.run_cost,
                                                   self.plan.total_points * self.plan.total_files),
                              show_eta=True)
        yield DataTable(id="results_table", zebra_stripes=True, cursor_type="row")
        yield RichLog(id="log", max_lines=5000, markup=False, wrap=True)
        with Horizontal(id="runactionbar"):
            yield Button("Abort (safe ramp-down)", id="abort_btn", variant="error")
            yield Button("Back", id="back_btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._follower_prefix, follower_display = follower_naming(self.plan.measure_rxx)
        self.query_one("#results_table", DataTable).add_columns(
            "#", "I (A)", "B (mT)", "1f R (V)", "1f θ (°)",
            f"{follower_display} R (V)", f"{follower_display} θ (°)",
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
            follower_prefix, follower_display = follower_naming(self.plan.measure_rxx)
            self._plot_queue, self._plot_process = start_live_plot(
                _live_plot_worker, self.plan.magnet_cfg is not None,
                follower_prefix, follower_display, self.plan.total_files > 1)
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
        fp = self._follower_prefix
        table.add_row(
            str(record["point_index"] + 1),
            f"{I:.4f}" if I is not None else "—",
            f"{B:.2f}" if B is not None else "—",
            f"{record['1f_R_V']:.4e}",
            f"{record['1f_theta_deg']:.2f}",
            f"{record[f'{fp}_R_V']:.4e}",
            f"{record[f'{fp}_theta_deg']:.2f}",
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(
            progress_step(self.plan.run_cost, len(self._records) - 1))
        self._set_status(f"Point {len(self._records)} / "
                          f"{self.plan.total_points * self.plan.total_files} complete.")

    def _make_on_point(self, series_index: int, series_label: Optional[str]):
        def _cb(record: dict) -> None:
            record["series_index"] = series_index
            record["series_label"] = series_label
            self.app.call_from_thread(self._on_point, record)
        return _cb

    def _save_run_png(self, ctx: RunContext, iter_records: list[dict]) -> None:
        """One PNG per run (own run number), as if each excitation current
        had been started by hand -- no combined overlay."""
        try:
            png_path = proc_path(self.plan.data_root, ctx.sample, ctx.run_str, ctx.device,
                                  MEASUREMENT_TYPE, "plot")
            self._png_path = png_path
            _save_measurement_png(iter_records, png_path, plan=self.plan)
        except Exception:
            log.exception("Could not save measurement plot PNG")

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False
        self.query_one("#abort_btn", Button).disabled = True

        self.app.push_screen(StatusCommentScreen(), self._on_status_comment)

    def _on_status_comment(self, result: Optional[tuple[str, str]]) -> None:
        # With several excitation currents the runs before the last were
        # implicitly "skipped" -- left at the outcome status do_run() wrote
        # right after each one, with no comment. Only the last run, the one
        # the operator is looking at, gets the status/comment they entered.
        if result is None or not self._run_contexts:
            return
        status, comment = result
        series_idx = len(self._run_contexts) - 1
        ctx = self._run_contexts[series_idx]
        iter_records = [r for r in self._records if r.get("series_index", 0) == series_idx]
        amp = iter_records[0].get("excitation_current_A_peak") if iter_records else None
        extra = {"excitation_amplitude_A": amp} if amp is not None else None
        header_fields = build_header_fields(
            self.plan, ctx, iter_records, status=status, comment=comment, extra=extra,
        )
        try:
            if iter_records or not ctx.raw_path.exists():
                write_record(ctx.raw_path, iter_records, header_fields)
            finalize_index_row(self.plan.data_root, ctx.sample, ctx.run_number, header_fields)
        except Exception:
            log.exception("Could not save final status/comment for run %d", ctx.run_number)

        if comment and self._png_path is not None:
            try:
                _save_measurement_png(iter_records, self._png_path, plan=self.plan, comment=comment)
            except Exception:
                log.exception("Could not re-save measurement plot PNG with comment")

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
        source = None
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

            disable_sigout(daq, plan.leader)
            disable_sigout(daq, plan.follower)

            self._set_status_threadsafe("Configuring demodulators …")
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
                # Amplitude-independent -- built once, reused for every
                # amplitude iteration below (set_action closures capture
                # magnet/gaussmeter/plan by reference, unaffected by amplitude).
                points = [
                    MeasurementPoint(
                        magnet_current_A=I,
                        set_action=lambda daq, I=I: set_magnet_current(
                            magnet, plan.magnet_cfg, I, gaussmeter, plan.gauss_cfg,
                            plan.acq_cfg.field_settle_tolerance_mT, self._stop_event),
                    )
                    for I in plan.currents_A
                ]
            else:
                points = [MeasurementPoint()]

            multi = len(plan.amplitudes_A) > 1
            for series_idx, amp in enumerate(plan.amplitudes_A):
                if self._stop_event.is_set():
                    break

                plan.ac_cfg.amplitude_A = amp
                label = f"I={amp:g}A" if multi else None

                # Checked here too, not just by build_summary(): connect_ac_source()
                # immediately arms and starts the 6221 at plan.ac_cfg.amplitude_A —
                # catch a mistyped exponent before that, not after (same ordering
                # as mfli_dual_harmonic_6221.main() / sot_pulsed_switching_6221.main()).
                _check_ac_safety(plan.ac_cfg)
                if source is not None:
                    # Amplitude requires a full re-arm (connect_ac_source()'s
                    # own docstring: a property write after waveform_arm()
                    # doesn't take effect until the next arm()) -- tear down
                    # the previous amplitude's source first, via safe_shutdown
                    # so a flaky GPIB hiccup here never skips the remaining
                    # amplitudes (the 6221 may still be sourcing current).
                    safe_shutdown("6221 AC source", lambda _s=source: shutdown_ac_source(_s))
                    source = None
                self._set_status_threadsafe(
                    f"Starting 6221 AC current source{f' ({amp:g} A)' if multi else ''} …"
                )
                source = connect_ac_source(plan.ac_cfg)

                self._set_status_threadsafe(
                    "Locking MFLI oscillators to the 6221 marker (ExtRef) …"
                )
                configure_external_reference(daq, plan.leader_extref_cfg, plan.ac_cfg.frequency_Hz)
                configure_external_reference(daq, plan.follower_extref_cfg, plan.ac_cfg.frequency_Hz)
                if not wait_for_reference_lock(daq, plan.leader_extref_cfg,
                                               plan.extref_lock_timeout_s, self._stop_event):
                    log.warning("Leader ExtRef PLL did not report locked within %.2g s — "
                               "check the marker cabling before trusting any data.",
                               plan.extref_lock_timeout_s)
                if not wait_for_reference_lock(daq, plan.follower_extref_cfg,
                                               plan.extref_lock_timeout_s, self._stop_event):
                    log.warning("Follower ExtRef PLL did not report locked within %.2g s — "
                               "check the marker fan-out cabling before trusting any data.",
                               plan.extref_lock_timeout_s)

                demod2_phase_null_1f_deg = None
                if plan.phase_cal_enabled:
                    self._set_status_threadsafe(
                        "Phase calibration: nulling 1f Y (leader demod phaseshift) …"
                    )
                    if magnet is not None and plan.phase_cal_current_A is not None:
                        log.info("Phase calibration: ramping magnet to %.4f A ...",
                                 plan.phase_cal_current_A)
                        set_magnet_current(magnet, plan.magnet_cfg, plan.phase_cal_current_A,
                                           gaussmeter, plan.gauss_cfg,
                                           plan.acq_cfg.field_settle_tolerance_mT, self._stop_event)
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
                    d2 = acquire_averaged(daq, plan.demod2_cfg, plan.phase_cal_n_averages)
                    follower_display = follower_naming(plan.measure_rxx)[1]
                    log.info(
                        "%s snapshot at calibration point: X=%.4e V  Y=%.4e V  R=%.4e V — "
                        "don't assume this matches 1f's X/Y convention (V_2w ~ cos, not sin); "
                        "check which channel carries the structured field dependence in the "
                        "recorded sweep before trusting either one.",
                        follower_display, d2["x_mean"], d2["y_mean"], d2["r_mean"],
                    )

                    self._set_status_threadsafe(
                        f"Phase calibration: anchoring follower {follower_display} reference (1f null) …"
                    )
                    demod2_phase_null_1f_deg = null_follower_reference_via_1f(
                        daq, plan.demod2_cfg,
                        n_averages=plan.phase_cal_n_averages,
                        max_iterations=plan.phase_cal_max_iterations,
                    )

                # A fresh RunContext (own run number, own file) EVERY
                # amplitude iteration -- never reuse one across the series.
                ctx = allocate_run(
                    plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                    temperature_setpoint_K=plan.temperature_setpoint_K,
                    key_axis=None, series=plan.series,
                )
                self._run_contexts.append(ctx)
                self._set_run_label_threadsafe(f"Run #{ctx.run_str}")
                plan.acq_cfg.output_file = str(ctx.raw_path)
                write_csv = make_incremental_writer(
                    ctx.raw_path,
                    lambda records, _ctx=ctx, _a=amp: build_header_fields(
                        plan, _ctx, records, status="in_progress", comment="",
                        extra={"excitation_amplitude_A": _a},
                    ),
                )

                status = "Running measurement …" if not multi else f"Running measurement ({label}) …"
                self._set_status_threadsafe(status)
                iter_error: Optional[BaseException] = None
                try:
                    run_measurement(
                        daq, plan.ac_cfg, plan.leader_extref_cfg, plan.follower_extref_cfg,
                        plan.demod1_cfg, plan.demod2_cfg, plan.acq_cfg, points,
                        stop_event=self._stop_event,
                        on_point=self._make_on_point(series_idx, label),
                        gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                        temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                        geometry_cfg=plan.geometry_cfg,
                        demod2_phase_null_1f_deg=demod2_phase_null_1f_deg, mds=mds,
                        write_csv=write_csv,
                        demod2_label=follower_naming(plan.measure_rxx)[0],
                    )
                except Exception as exc:
                    iter_error = exc

                # Finalize THIS iteration's header/index row UNCONDITIONALLY,
                # right now -- never gated on the end-of-session status/
                # comment prompt, so an aborted/crashed session never leaves
                # a file stuck at "in_progress".
                iter_status = "error" if iter_error is not None \
                    else ("aborted" if self._stop_event.is_set() else "completed")
                iter_records = [r for r in self._records if r.get("series_index", 0) == series_idx]
                header_fields = build_header_fields(
                    plan, ctx, iter_records, status=iter_status, comment="",
                    extra={"excitation_amplitude_A": amp},
                )
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
            # 6221 output off first (immediate, no current into the DUT),
            # so the magnet can start its ramp-down right away rather than
            # waiting behind it.
            if source is not None:
                try:
                    shutdown_ac_source(source)
                except Exception:
                    log.exception("Error while shutting down 6221 AC source")
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
            self.app.call_from_thread(self._on_finished, final)

    def _set_status_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_status, text)

    def _set_run_label(self, text: str) -> None:
        self.query_one("#run_label", Static).update(text)

    def _set_run_label_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_run_label, text)


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class MFLIDualHarmonic6221App(App):
    TITLE = "MFLI Dual-Harmonic Measurement (6221 AC source)"
    SUB_TITLE = "1f / 2f lock-in · Keithley 6221 excitation · magnet field sweep"

    data_root: Path = _DEFAULT_DATA_DIR

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

    Collapsible { height: auto; margin: 1 0; }
    Collapsible > Contents { padding: 1 0 0 1; }
    CollapsibleTitle { text-style: bold; color: $text-muted; }
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
                        yield Vertical(*field("device", "Device (e.g. HB3, SV2)",
                                              DEFAULTS["device"], kind="text"), classes="field")
                        yield Vertical(*field("cooldown", "Cooldown (optional)",
                                              DEFAULTS["cooldown"], kind="text"), classes="field")
                        yield Vertical(*field("temperature_setpoint_K", "Temp. setpoint (K, optional)",
                                              DEFAULTS["temperature_setpoint_K"], kind="number",
                                              valid_empty=True,
                                              hint="Filename's T###K token only."),
                                       classes="field")

                # ── Tier 1: what defines this run — always visible ──────────
                with Vertical(classes="param-grid"):
                    yield card(
                        "Excitation (Keithley 6221 AC current source)",
                        field("frequency_Hz", "Excitation frequency (Hz)",
                              DEFAULTS["frequency_Hz"],
                              hint="Recommended ~300-1000 Hz — avoid exact multiples of 50/60 Hz "
                                   "(mains pickup).",
                              validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                        field("amplitude_values", "Excitation current (A, peak)",
                              DEFAULTS["amplitude_values"], kind="text",
                              hint="Ideal current source — no series resistor. Single value, "
                                   "or comma-separated list — one complete sweep runs per "
                                   "value (own 6221 re-arm), each saved to its own file."),
                        field("ac_compliance_V", "6221 voltage compliance (V)",
                              DEFAULTS["ac_compliance_V"],
                              validators=[Number(minimum=0.1, failure_description="must be > 0")]),
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
                            field("leader_device", "Leader MFLI (1f)",
                                  DEFAULTS["leader_device"], kind="text"),
                            field("follower_device", "Follower MFLI (2f, or R_xx 1f if R_xx mode is on)",
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
                                         AUTOMODE_OPTIONS, int(DEFAULTS["leader_automode"]),
                                         hint=AUTOMODE_HINT),
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
                                         AUTOMODE_OPTIONS, int(DEFAULTS["follower_automode"]),
                                         hint=AUTOMODE_HINT),
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

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        logging.getLogger().handlers.clear()
        self._load_settings()
        self.refresh_summary()

    # ── Sample picker ────────────────────────────────────────────────────────

    def _refresh_sample_options(self, *, select_value: Optional[str] = None) -> None:
        select = self.query_one("#sample_select", Select)
        options = sample_options(self.data_root)
        select.set_options(options)
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
        if event.select.id == "sample_select":
            if event.value == NEW_SAMPLE_SENTINEL:
                self.push_screen(NewSampleScreen(self.data_root), self._on_new_sample_created)
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
        raw["sweep_rows"] = self.query_one("#sweep_rows", TextArea).text
        raw["sinc_filter_1f"] = self.query_one("#sinc_filter_1f", Switch).value
        raw["sinc_filter_2f"] = self.query_one("#sinc_filter_2f", Switch).value
        raw["measure_rxx"] = self.query_one("#measure_rxx", Switch).value
        raw["differential"] = self.query_one("#differential", Switch).value
        raw["ac_coupling"] = self.query_one("#ac_coupling", Switch).value
        raw["enable_sweep"] = self.query_one("#enable_sweep", Switch).value
        raw["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
        raw["enable_phase_cal"] = self.query_one("#enable_phase_cal", Switch).value
        raw["order_1f"] = self.query_one("#order_1f", Select).value
        raw["order_2f"] = self.query_one("#order_2f", Select).value
        raw["leader_automode"] = self.query_one("#leader_automode", Select).value
        raw["follower_automode"] = self.query_one("#follower_automode", Select).value
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
        if "sweep_rows" in saved:
            self.query_one("#sweep_rows", TextArea).text = str(saved["sweep_rows"])
        if "sinc_filter_1f" in saved:
            self.query_one("#sinc_filter_1f", Switch).value = bool(saved["sinc_filter_1f"])
        if "sinc_filter_2f" in saved:
            self.query_one("#sinc_filter_2f", Switch).value = bool(saved["sinc_filter_2f"])
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
        if "measure_rxx" in saved:
            self.query_one("#measure_rxx", Switch).value = bool(saved["measure_rxx"])
        if "order_1f" in saved:
            try:
                self.query_one("#order_1f", Select).value = int(saved["order_1f"])
            except Exception:
                pass
        if "order_2f" in saved:
            try:
                self.query_one("#order_2f", Select).value = int(saved["order_2f"])
            except Exception:
                pass
        if "leader_automode" in saved:
            try:
                self.query_one("#leader_automode", Select).value = int(saved["leader_automode"])
            except Exception:
                pass
        if "follower_automode" in saved:
            try:
                self.query_one("#follower_automode", Select).value = int(saved["follower_automode"])
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
        state["sinc_filter_1f"] = self.query_one("#sinc_filter_1f", Switch).value
        state["sinc_filter_2f"] = self.query_one("#sinc_filter_2f", Switch).value
        state["measure_rxx"] = self.query_one("#measure_rxx", Switch).value
        state["differential"] = self.query_one("#differential", Switch).value
        state["ac_coupling"] = self.query_one("#ac_coupling", Switch).value
        state["enable_sweep"] = self.query_one("#enable_sweep", Switch).value
        state["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
        state["enable_phase_cal"] = self.query_one("#enable_phase_cal", Switch).value
        state["order_1f"] = int(self.query_one("#order_1f", Select).value)
        state["order_2f"] = int(self.query_one("#order_2f", Select).value)
        state["leader_automode"] = int(self.query_one("#leader_automode", Select).value)
        state["follower_automode"] = int(self.query_one("#follower_automode", Select).value)
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""

        state["sweep_rows"] = self.query_one("#sweep_rows", TextArea).text
        state["sweep_rows_parsed"] = []
        state["sweep_rows_parse_error"] = None
        try:
            state["sweep_rows_parsed"] = parse_sweep_rows(state["sweep_rows"])
        except ValueError as exc:
            state["sweep_rows_parse_error"] = str(exc)

        state["amplitude_list"] = []
        state["amplitude_parse_error"] = None
        try:
            state["amplitude_list"] = parse_value_list(state["amplitude_values"])
        except ValueError as exc:
            state["amplitude_parse_error"] = str(exc)

        return state, errors

    # ── Reactivity ───────────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "data_dir":
            self._sync_data_root()
        self.refresh_summary()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
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
        self.query_one("#sweep_rows", TextArea).disabled = not enabled

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

        theta = None if parse_errors else state.get("field_theta_deg")
        phi = None if parse_errors else state.get("field_phi_deg")
        self.query_one("#field_diagram", Static).update(render_ascii_field_diagram(theta, phi))

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

        self.data_root = Path(state["data_dir"]).expanduser()
        ensure_sample(self.data_root, state["sample"], create=True)
        self._save_settings(self.collect_raw())
        plan = self._build_plan(state)
        self.push_screen(RunScreen(plan))

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
            device=state["leader_device"], demod_index=0, harmonic=1,
            osc_index=state["leader_osc_index"],
            differential=state["differential"], ac_coupling=state["ac_coupling"],
            input_range_V=state["input_range_1f_V"],
            sample_rate_Hz=state["sample_rate_Hz"], filter=filt_1f,
        )
        demod2_cfg = DemodConfig(
            device=state["follower_device"], demod_index=0,
            harmonic=1 if state["measure_rxx"] else 2,
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
            measure_rxx=state["measure_rxx"],
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
            sample=state["sample"], device=state["device"], data_root=self.data_root,
            temperature_setpoint_K=state["temperature_setpoint_K"],
            cooldown=state["cooldown"], header_extra=header_extra, series=series,
            run_cost=run_costs(state, currents_A),
        )


def main() -> None:
    MFLIDualHarmonic6221App().run()


if __name__ == "__main__":
    main()
