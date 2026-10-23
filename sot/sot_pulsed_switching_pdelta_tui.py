#!/usr/bin/env python3
"""
6221 Pulse Delta + Delta-read engine of the SOT pulsed-switching form  (type SOTPD)
==================================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-10-23 (a mode of sot_pulsed_switching_tui.py)

The "Pulse = 6221, Read = DC" engine behind the switching form in
sot_pulsed_switching_tui.py: its parameter surface, summary, plan builder, run
loop, header, PNG and RunScreen. No 4200A: the 6221 fires one Pulse Delta
write pulse, then reads R_xy in Delta mode with the 2182A on its RS-232 port.
See sot/sot_pulsed_switching_pdelta.py for the mechanism and wiring.

Run:  python sot_pulsed_switching_pdelta_tui.py   (opens the form in this mode)
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

from sot.sot_pulsed_switching_6221 import (
    _READ_COMPLIANCE_CEILING_V,
    _READ_CURRENT_CEILING_A,
    _WRITE_CURRENT_HARD_MAX_A,
    _check_pulse_currents,
    _check_read_safety,
    PulsePoint,
)
from sot.sot_pulsed_switching_6221_tui import resolve_state  # same pulse/magnet/sense keys
from sot.sot_pulsed_switching_pdelta import (
    ReadConfig, _source_quiet, connect_pdelta_source, run_measurement,
)
from dc.dc_sweep_utils import safe_shutdown
from instruments.data_dir import validate_directory
from instruments.data_naming import RunContext, allocate_run, preview_raw_filename, record_run
from instruments.field_geometry import field_direction_summary_line
from instruments.keithley6221 import (
    DELTA_NPLC_RANGE, PDEL_WIDTH_RANGE_S, PulseDeltaConfig, delta_read_s, pdelta_pulse_s,
)
from instruments.kepco_magnet import (
    MagnetConfig, connect_magnet, magnet_move_s, set_magnet_current, shutdown_magnet,
)
from instruments.lakeshore475 import (
    GaussmeterConfig, connect_gaussmeter, read_field_s, shutdown_gaussmeter,
)
from instruments.mercury_itc import (
    TemperatureControllerConfig, connect_temperature_controller, shutdown_temperature_controller,
)
from instruments.run_time import (
    ARM_S, GPIB_TXN_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S, RunCost,
)
from instruments.temperature_control import (
    TEMPERATURE_DEFAULTS, TEMPERATURE_NUMERIC_FIELDS, TEMPERATURE_TEXT_FIELDS,
    temperature_cfg_from_state, temperature_summary,
)
from instruments.tui_common import (
    MeasurementRunScreen, format_si, parse_sensor_uids, run_screen_bindings,
)
from instruments.tui_sample_picker import NEW_SAMPLE_SENTINEL

log = logging.getLogger("sot_pulsed_switching_pdelta_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "sot_pulsed_switching_pdelta_tui_settings.json"

MEASUREMENT_TYPE = "SOTPD"

RESPONSE_OPTIONS = [("FAST (≈ 2-3 µs edge)", "FAST"), ("SLOW (≈ 100 µs edge)", "SLOW")]

DEFAULTS: dict = {
    # write pulse (6221 Pulse Delta)
    "pulse_current_start_A": "1e-3",
    "pulse_current_stop_A": "10e-3",
    "pulse_current_step_A": "1e-3",
    "amplitude_bidirectional": True,
    "pulse_width_s": "1e-3",
    "pulse_compliance_V": "5.0",
    "pulse_response": "FAST",
    "pulse_analog_filter": False,
    # delayed Delta read (6221 ±I + 2182A over RS-232)
    "sense_current_values": "1e-4",
    "compliance_V": "2.0",
    "n_delta_readings": "5",
    "delta_delay_s": "0.05",
    "delta_nplc": "5",
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
    "magnet_visa_resource": "GPIB0::6::INSTR",
    "current_limit_A": "35",
    "magnet_voltage_compliance_V": "15.0",
    "ramp_step_A": "0.1",
    "ramp_delay_s": "0.05",
    "gaussmeter_visa_resource": "GPIB0::12::INSTR",
    "gaussmeter_n_averages": "10",
    "gaussmeter_read_delay_s": "0.05",
    **TEMPERATURE_DEFAULTS,
}

NUMERIC_FIELDS: dict = {
    "pulse_current_start_A": float,
    "pulse_current_stop_A": float,
    "pulse_current_step_A": float,
    "pulse_width_s": float,
    "pulse_compliance_V": float,
    "compliance_V": float,
    "n_delta_readings": int,
    "delta_delay_s": float,
    "delta_nplc": int,
    "delay_after_pulse_s": float,
    "field_theta_deg": float,
    "field_settle_tolerance_mT": float,
    "current_limit_A": float,
    "magnet_voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
    **TEMPERATURE_NUMERIC_FIELDS,
}
TEXT_FIELDS = ["device", "cooldown", "source_visa_resource", "magnet_visa_resource",
               "gaussmeter_visa_resource", *TEMPERATURE_TEXT_FIELDS, "magnet_current_A",
               "sense_current_values", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K", "field_phi_deg"]


def _pulse_cfg(state: dict) -> PulseDeltaConfig:
    return PulseDeltaConfig(width_s=state["pulse_width_s"], compliance_V=state["pulse_compliance_V"],
                            response=state["pulse_response"],
                            analog_filter=bool(state["pulse_analog_filter"]))


def run_costs(state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (assist current outer x sense current inner, like MeasurementPlan.
    series_values). Also drives the run screen's progress bar."""
    pulses = state.get("pulse_current_list", [])
    series = list(itertools.product(state.get("magnet_currents_A", []),
                                    state.get("sense_currents_A", [])))
    rc = RunCost(len(pulses) * max(1, len(series)))
    magnet = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    gauss = GaussmeterConfig(n_averages=state["gaussmeter_n_averages"],
                             read_delay_s=state["gaussmeter_read_delay_s"])
    has_temp = state["enable_temperature"] and bool(parse_sensor_uids(state["temperature_sensor_uids"]))
    # run_measurement(): Pulse Delta pulse (own arm) -> wait -> compliance write -> Delta read
    # (11 writes/queries + arm, which configures the 2182A over RS-232) -> temperature -> CSV
    rc.each("6221 pulse", pdelta_pulse_s(_pulse_cfg(state)))
    rc.each("post-pulse wait", state["delay_after_pulse_s"])
    rc.each("Delta read", 12 * GPIB_TXN_S + ARM_S + delta_read_s(
        state["n_delta_readings"], state["delta_delay_s"], state["delta_nplc"]))
    rc.each("overhead", POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    parked = None
    for k, (I_mag, _I_sense) in enumerate(series):
        first = k * len(pulses)
        if I_mag != parked:
            rc.at("magnet", magnet_move_s(abs(I_mag - (parked or 0.0)), magnet), first)
            parked = I_mag
        rc.at("field read", read_field_s(gauss), first)
        rc.at("per-file", PER_FILE_S, first)
    rc.at("per-run", PER_RUN_S, 0)
    if series:
        rc.tail("ramps", magnet_move_s(abs(series[-1][0]), magnet, with_field=False))
    return rc


# ── plan ────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    source_visa_resource: str
    pulse_cfg: PulseDeltaConfig
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
    run_cost: Optional[RunCost] = None

    @property
    def series_values(self) -> List[tuple[float, float]]:
        """Assist-field currents x sense currents -- one pulse sweep + file
        per pair; magnet outer (a ramp), sense inner (a config value)."""
        return list(itertools.product(self.magnet_currents_A, self.sense_currents_A))

    @property
    def total_points(self) -> int:
        return len(self.pulse_currents_A) * max(1, len(self.series_values))


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                        status: str, comment: str, extra: Optional[dict] = None) -> dict:
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
        bad = [a for a in amps if not 0 < abs(a) <= _WRITE_CURRENT_HARD_MAX_A]
        if bad:
            errors.append(f"Pulse current(s) {bad} A are zero or exceed the 6221's range "
                          f"±{_WRITE_CURRENT_HARD_MAX_A:g} A.")
        loop = ", up and back" if state["amplitude_bidirectional"] else ""
        info.append(f"Pulse sweep: {format_si(state['pulse_current_start_A'], 'A')} → "
                    f"{format_si(state['pulse_current_stop_A'], 'A')} — step "
                    f"{format_si(state['pulse_current_step_A'], 'A')}, {len(amps)} pulses{loop}"
                    if amps else "")
        if not state["amplitude_bidirectional"]:
            warnings.append("One-way sweep — turn on 'Sweep up then back down' for a "
                            "hysteresis loop; the sweep is what sets each pulse's starting "
                            "state.")
    lo, hi = PDEL_WIDTH_RANGE_S
    if not lo <= state["pulse_width_s"] <= hi:
        errors.append(f"Pulse Delta width must be {format_si(lo, 's')} – {format_si(hi, 's')}.")
    if not 0.1 <= state["pulse_compliance_V"] <= 105.0:
        errors.append("Pulse compliance must be 0.1 – 105 V.")
    info.append(f"Write pulse: 6221 Pulse Delta, one 0 → I → 0 pulse, response "
                f"{state['pulse_response']}, analog filter "
                f"{'on' if state['pulse_analog_filter'] else 'off'}. Joule heating ∝ I²R·t: "
                "start well below the switching current")

    # read (Delta)
    sense_currents = state.get("sense_currents_A", [])
    if state.get("sense_currents_parse_error"):
        errors.append(f"6221 read current(s): {state['sense_currents_parse_error']}")
    else:
        bad = [i for i in sense_currents if not 0 < i <= _READ_CURRENT_CEILING_A]
        if bad:
            errors.append(f"6221 read current(s) {bad} must be in (0, "
                          f"{format_si(_READ_CURRENT_CEILING_A, 'A')}] — check for a mistyped "
                          "exponent.")
        elif any(i > 1e-3 for i in sense_currents):
            warnings.append("6221 read current > 1 mA — keep it well below the switching current.")
    if not 0 < state["compliance_V"] <= _READ_COMPLIANCE_CEILING_V:
        errors.append(f"6221 read compliance must be in (0, {_READ_COMPLIANCE_CEILING_V:g} V].")
    if state["n_delta_readings"] < 1:
        errors.append("Delta readings per read must be ≥ 1.")
    if state["delta_delay_s"] < 0:
        errors.append("Delta delay must be ≥ 0 s.")
    n_lo, n_hi = DELTA_NPLC_RANGE
    if not n_lo <= state["delta_nplc"] <= n_hi:
        errors.append(f"2182A NPLC must be an integer {n_lo} – {n_hi} for Delta mode.")
    if state["delay_after_pulse_s"] < 0:
        errors.append("Delay after pulse must be ≥ 0 s.")
    info.append("Read: 6221 Delta (±I, V_odd only) with the 2182A on the 6221's RS-232 — set the "
                "2182A range on its front panel")

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
    n_files = max(1, len(currents)) * max(1, len(sense_currents))
    info.append(f"Points: {n * n_files} — one pulse each"
                + (f"; {n} pulses × {n_files} files" if n_files > 1 else ""))
    info.extend(run_costs(state).lines())
    if len(currents) > 1:
        info.append(f"Magnet currents: {', '.join(f'{i:g}' for i in currents)} A — one sweep + "
                    "file each, field read by 475")
    elif currents:
        info.append(f"Magnet current: {currents[0]:g} A — static, field read by 475")
    info.append(field_direction_summary_line(state["field_theta_deg"], state.get("field_phi_deg")))

    temperature_summary(state, info, warnings, errors)
    return info, warnings, errors


def compute_filename_preview(state: dict) -> Optional[str]:
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL or not state.get("device"):
        return None
    preview = preview_raw_filename(
        state["sample"], state["device"], MEASUREMENT_TYPE,
        temperature_setpoint_K=state.get("temperature_setpoint_K"))
    n_files = max(1, len(state.get("magnet_currents_A", []))) * max(1, len(state.get("sense_currents_A", [])))
    return f"{preview}_<I_mag A>_<timestamp>.csv" + (" (one file per run)" if n_files > 1 else "")


# ── plots ──────────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("SOT pulsed switching (6221 Pulse Delta) — live")
    except Exception:
        pass
    ax.set_xlabel("Pulse current (A)")
    ax.set_ylabel("R_xy (Ω)")
    ax.set_title("Live — R_xy vs pulse current")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    cmap = plt.get_cmap("tab10")
    lines: dict = {}
    data: dict = {}

    def _drain(_frame=None):
        new_series = False
        while True:
            try:
                rec = queue.get_nowait()
            except Exception:
                break
            idx = rec.get("series_index", 0)
            if idx not in lines:
                (lines[idx],) = ax.plot([], [], "o-", ms=4, lw=1, alpha=0.6,
                                        color=cmap(idx % 10), label=rec.get("series_label"))
                data[idx] = ([], [])
                new_series = True
            data[idx][0].append(rec["pulse_current_A"])
            data[idx][1].append(rec["hall_resistance_ohm"])
            lines[idx].set_data(*data[idx])
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
    """One run's R_xy vs pulse current, in acquisition order, with the field
    direction / read current / comment as a small footer."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["pulse_current_A"] for r in records], [r["hall_resistance_ohm"] for r in records],
            "o-", ms=4, lw=1, alpha=0.6, color="tab:blue")
    ax.set_xlabel("Pulse current (A)")
    ax.set_ylabel("R_xy (Ω)")
    ax.set_title("R_xy vs pulse current (6221 Pulse Delta)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lines: list[str] = []
    if plan is not None and plan.field_theta_deg is not None:
        lines.append(field_direction_summary_line(plan.field_theta_deg, plan.field_phi_deg))
    lines.append(f"6221 read current: {format_si(records[0]['sense_current_A'], 'A')}")
    if comment:
        lines.append(f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}")
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color="0.4", va="bottom")
    fig.subplots_adjust(bottom=0.08 + 0.045 * len(lines))
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot to '%s'", png_path)


# ── plan + run (pure, used by the TUI RunScreen) ──────────────

def build_plan(state: dict, data_root: Path) -> MeasurementPlan:
    """One parsed, validated run request from a state dict. Pure."""
    read_cfg = ReadConfig(
        sense_current_A=state["sense_currents_A"][0], compliance_V=state["compliance_V"],
        n_delta_readings=state["n_delta_readings"], delta_delay_s=state["delta_delay_s"],
        nplc=state["delta_nplc"], delay_after_pulse_s=state["delay_after_pulse_s"],
    )
    pulse_cfg = _pulse_cfg(state)
    header_extra = {
        "pulse_width_s": pulse_cfg.width_s,
        "pulse_compliance_V": pulse_cfg.compliance_V,
        "output_response": pulse_cfg.response,
        "analog_filter": pulse_cfg.analog_filter,
        "delay_after_pulse_s": read_cfg.delay_after_pulse_s,
        "n_delta_readings": read_cfg.n_delta_readings,
        "delta_delay_s": read_cfg.delta_delay_s,
        "nplc": read_cfg.nplc,
        "field_theta_deg": state["field_theta_deg"],
        "field_phi_deg": state["field_phi_deg"],
        "pulse_current_start_A": state["pulse_current_start_A"],
        "pulse_current_stop_A": state["pulse_current_stop_A"],
        "pulse_current_step_A": state["pulse_current_step_A"],
        "amplitude_bidirectional": state["amplitude_bidirectional"],
        "pulse_currents_A": state["pulse_current_list"],
    }
    return MeasurementPlan(
        source_visa_resource=state["source_visa_resource"],
        pulse_cfg=pulse_cfg, read_cfg=read_cfg,
        magnet_cfg=MagnetConfig(
            visa_resource=state["magnet_visa_resource"], current_limit_A=state["current_limit_A"],
            voltage_compliance_V=state["magnet_voltage_compliance_V"],
            ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"]),
        gauss_cfg=GaussmeterConfig(
            visa_resource=state["gaussmeter_visa_resource"], unit="T",
            n_averages=state["gaussmeter_n_averages"],
            read_delay_s=state["gaussmeter_read_delay_s"]),
        pulse_currents_A=state["pulse_current_list"], magnet_currents_A=state["magnet_currents_A"],
        sense_currents_A=state["sense_currents_A"],
        field_theta_deg=state["field_theta_deg"], field_phi_deg=state["field_phi_deg"],
        field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
        data_root=data_root, sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series="",
        temp_cfg=temperature_cfg_from_state(state), run_cost=run_costs(state),
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
    """Check the limits, connect the 6221 (2182A on its RS-232), magnet and
    gaussmeter, then one pulse-current sweep (own run number, own file) per
    (assist-field current, read current) pair through record_run(); always
    shut everything down (6221 first, then the magnet). Pure."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = magnet = gaussmeter = temp_ctrl = None
    try:
        points = [PulsePoint(pulse_current_A=float(v)) for v in plan.pulse_currents_A]
        _check_pulse_currents(points)
        for I_sense in plan.sense_currents_A:
            plan.read_cfg.sense_current_A = I_sense
            _check_read_safety(plan.read_cfg)

        on_status("Connecting to Keithley 6221 (+ 2182A over RS-232) …")
        source = connect_pdelta_source(plan.source_visa_resource)
        on_status("Connecting to Kepco magnet + Lake Shore 475 …")
        magnet = connect_magnet(plan.magnet_cfg)
        gaussmeter = connect_gaussmeter(plan.gauss_cfg)
        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        parked = None
        for series_idx, (I_mag, I_sense) in enumerate(plan.series_values):
            if stop_event.is_set():
                break
            plan.read_cfg.sense_current_A = I_sense
            label_parts = []
            if len(plan.magnet_currents_A) > 1:
                label_parts.append(f"I_mag={I_mag:g}A")
            if len(plan.sense_currents_A) > 1:
                label_parts.append(f"I_sense={I_sense:g}A")
            label = ", ".join(label_parts) or None

            if I_mag != parked:
                on_status(f"Ramping magnet to {I_mag:g} A …")
                set_magnet_current(magnet, plan.magnet_cfg, I_mag, gaussmeter, plan.gauss_cfg,
                                   plan.field_settle_tolerance_mT, stop_event)
                parked = I_mag

            ctx = allocate_run(plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                               temperature_setpoint_K=plan.temperature_setpoint_K,
                               key_axis=("current_A", I_mag), series=plan.series)
            extra = {"magnet_current_A": I_mag, "sense_current_A": I_sense}
            run_contexts.append(ctx)
            run_extras.append(extra)
            on_run_label(f"Run #{ctx.run_str}")

            on_status("Running the switching sweep …" if not label_parts
                      else f"Running the switching sweep ({label}) …")
            record_run(
                plan.data_root, ctx,
                lambda records, status, _ctx=ctx, _x=extra: build_header_fields(
                    plan, _ctx, records, status=status, comment="", extra=_x),
                lambda point_cb, write_csv, _ctx=ctx, _I=I_mag: run_measurement(
                    source, plan.pulse_cfg, plan.read_cfg, points,
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
            safe_shutdown("6221", lambda: (_source_quiet(source), source.shutdown()))
        if magnet is not None:
            safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
        if gaussmeter is not None:
            safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


PNG_SUFFIX = "Rxy_vs_pulse"


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (the RunScreen calls this)."""
    _save_measurement_png(records, png_path, plan=plan, comment=comment)


# ── run screen ─────────────────────────────────────────────────────────────

class RunScreen(MeasurementRunScreen):
    ABORT_LABEL = "Abort (safe shutdown)"
    BINDINGS = run_screen_bindings(ABORT_LABEL)
    ABORT_STATUS = "Abort requested — finishing this pulse, then shutting the 6221 + magnet down …"
    POINT_STATUS = "Point {n} / {total}."
    TABLE_COLUMNS = ("amp #", "I_mag (A)", "I_pulse (A)", "pulse ok", "V_xy (V)", "R_xy (Ω)", "T1 (K)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE
    PNG_SUFFIX = PNG_SUFFIX

    def live_plot_args(self):
        return (_live_plot_worker,)

    def table_row(self, record: dict) -> tuple:
        t1 = record.get("temperature_1_K")
        return (
            str(record["amplitude_index"] + 1),
            f"{record['magnet_current_A']:g}" if record.get("magnet_current_A") is not None else "—",
            f"{record['pulse_current_A']:.4g}",
            "yes" if record["pulse_completed"] else "NO",
            f"{record['hall_voltage_V']:.4e}",
            f"{record['hall_resistance_ohm']:.5g}",
            f"{t1:.3f}" if t1 is not None else "—",
        )


def main() -> None:
    """Open the SOT pulsed-switching form on 6221 pulse + DC read (SOTPD)."""
    from sot.sot_pulsed_switching_tui import SOTPulsedSwitchingApp
    SOTPulsedSwitchingApp(pulse_source="6221", read_mode="dc").run()


if __name__ == "__main__":
    main()
