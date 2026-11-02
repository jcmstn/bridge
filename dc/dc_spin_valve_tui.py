#!/usr/bin/env python3
"""
Textual TUI front-end for dc_spin_valve.py
=============================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-31

Lets you edit the parameters that decide whether a spin-valve/field-sweep
measurement is good or bad — sense source (6221, or a 2450 sourcing
current or voltage), sense current, compliance, reversal
averaging, the magnet sweep, and the gate voltage — without touching the
dataclasses in the script itself.

The gate voltage (single value, or a comma-separated list) is held fixed
for each complete field sweep; a list runs one complete field sweep per
gate value, each saved to its own file and plotted together in the same
window with a different color.

Run with:
    python dc_spin_valve_tui.py

Requirements:
    pip install textual matplotlib  (in addition to dc_spin_valve.py's own deps)
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import textwrap
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.validation import Number
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Label,
    Static,
)

from dc.dc_spin_valve import (
    AcquisitionConfig,
    FieldPoint,
    GateConfig,
    GaussmeterConfig,
    MagnetConfig,
    SMUConfig,
    SourceConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    connect_gate,
    connect_gaussmeter,
    connect_magnet,
    connect_smu,
    connect_source,
    connect_temperature_controller,
    connect_voltmeter,
    run_measurement,
    set_gate_voltage,
    set_magnet_current,
    set_source_level,
    shutdown_gate,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_smu,
    shutdown_source,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import build_segmented_sweep, field_hops, parse_sweep_rows, safe_shutdown, try_parse
from instruments.data_dir import validate_directory
from instruments.data_naming import (
    RunContext,
    allocate_run,
    record_run,
    preview_raw_filename,
)
from instruments.keithley2182 import read_time_s
from instruments.keithley6221 import reversal_avg_s
from instruments.kepco_magnet import magnet_move_s
from instruments.lakeshore475 import read_field_s
from instruments.run_time import (
    GATE_RAMP_S, GPIB_TXN_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
    RunCost,
)
from instruments.summary_lines import summary_markup
from instruments.temperature_control import (
    TEMPERATURE_DEFAULTS, TEMPERATURE_FIELD_IDS, TEMPERATURE_NUMERIC_FIELDS, TEMPERATURE_TEXT_FIELDS,
    temperature_cfg_from_state, temperature_summary,
)
from instruments.tui_common import (
    MeasurementApp,
    MeasurementRunScreen,
    card,
    field,
    format_si,
    identity_bar,
    parse_sensor_uids,
    temperature_card,
    switch_field,
    sweep_rows_field,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
)

log = logging.getLogger("dc_spin_valve_tui")

# Data/settings live outside "bridge" (a sibling of it). _DEFAULT_DATA_DIR is
# the fallback data-convention "data root"; the real root is chosen per run in
# the identity bar's "Data root" field -- see dc_hall_measurement_tui.py.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "dc_spin_valve_tui_settings.json"

# Locked type code (see instruments/data_naming.py) — never deviates.
MEASUREMENT_TYPE = "BSWP"

DC_SPIN_VALVE_DESCRIPTION = (
    "6221 fixed current — or 2450 fixed I/V, 2-wire logged too — (±reversal, can be "
    "off) · 2182 voltage. Kepco field "
    "sweep, read by Lake Shore 475. Optional 2400 gate: fixed, or a list → one "
    "sweep per value."
)

# Wiring schematic — shown on this program's card in bridge_tui.py.
DC_SPIN_VALVE_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source)
    Output ──▶ sample ── common ground
    Current reversal (+I/-I) is a toggle — some devices are
    bias-direction dependent and reversal destroys the signal

  — or, "Use Keithley 2450" on —
  KEITHLEY 2450  (sources current OR voltage, 2-wire)
    Force HI/LO ──▶ sample current leads
    Its own 2-wire V (or I) is logged; plotting it is a toggle

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ longitudinal voltage leads

  KEITHLEY 2400  (gate source, optional — "Enable gate" switch)
    Output ──▶ gate electrode
    Fixed per sweep, single value or list

  Magnet field sweep  (the swept axis)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample
"""


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors dc_spin_valve.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "source_visa_resource": "GPIB0::20::INSTR",
    "voltmeter_visa_resource": "GPIB0::7::INSTR",
    "gate_visa_resource": "GPIB0::25::INSTR",
    "sense_current_values": "0.001",
    "use_2450": False,
    "smu_source_voltage": False,
    "plot_2wire": True,
    "smu_visa_resource": "GPIB0::18::INSTR",
    "source_voltage_values": "0.1",
    "smu_compliance_current_A": "0.001",
    "smu_voltage_limit_V": "21",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "nplc": "5",
    "auto_range": True,
    "settling_time_s": "1.0",
    "field_settle_tolerance_mT": "0.02",
    "reversal_enabled": True,
    "n_averages": "5",
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    "enable_gate": False,
    "gate_voltage_limit_V": "20.0",
    "gate_compliance_current_A": "0.000001",
    "gate_voltage_values": "0.0",
    "magnet_visa_resource": "GPIB0::6::INSTR",
    "current_limit_A": "35",
    "voltage_compliance_V": "15.0",
    "ramp_step_A": "0.1",
    "ramp_delay_s": "0.05",
    "sweep_rows": "-20, 20, 21",
    "bidirectional_sweep": True,
    "gaussmeter_visa_resource": "GPIB0::12::INSTR",
    "gaussmeter_n_averages": "10",
    "gaussmeter_read_delay_s": "0.05",
    **TEMPERATURE_DEFAULTS,
}

NUMERIC_FIELDS: dict = {
    "compliance_V": float,
    "source_delay_s": float,
    "nplc": float,
    "settling_time_s": float,
    "field_settle_tolerance_mT": float,
    "n_averages": int,
    "gate_voltage_limit_V": float,
    "gate_compliance_current_A": float,
    "current_limit_A": float,
    "voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
    "smu_compliance_current_A": float,
    "smu_voltage_limit_V": float,
    **TEMPERATURE_NUMERIC_FIELDS,
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "gate_visa_resource",
               "device", "cooldown", "magnet_visa_resource",
               "gaussmeter_visa_resource", "gate_voltage_values",
               *TEMPERATURE_TEXT_FIELDS,
               "sense_current_values", "smu_visa_resource", "source_voltage_values", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K"]
GATE_FIELD_IDS = ["gate_visa_resource", "gate_voltage_limit_V",
                   "gate_compliance_current_A", "gate_voltage_values"]
SMU_V_FIELD_IDS = ["source_voltage_values", "smu_compliance_current_A", "smu_voltage_limit_V"]


def smu_voltage_mode(state: dict) -> bool:
    return bool(state.get("use_2450") and state.get("smu_source_voltage"))


def level_list(state: dict) -> list[float]:
    """The per-file source levels of the active mode: 2450 source voltages
    (V) in voltage mode, else the sense currents (A, 6221 or 2450)."""
    key = "source_voltage_list" if smu_voltage_mode(state) else "sense_current_list"
    return state.get(key) or []


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def run_costs(currents_A, state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (series-major: one full field sweep per sense current x gate voltage).
    Also drives the run screen's progress bar, so estimate and live ETA
    cannot disagree."""
    n_gate = max(1, len(state.get("gate_voltage_list") or [])) if state["enable_gate"] else 1
    n_series = max(1, len(level_list(state))) * n_gate
    n_pts = len(currents_A)
    mcfg = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    gcfg = GaussmeterConfig(n_averages=state["gaussmeter_n_averages"],
                            read_delay_s=state["gaussmeter_read_delay_s"])
    has_temp = state["enable_temperature"] and bool(parse_sensor_uids(state["temperature_sensor_uids"]))
    read_s = read_time_s(state["nplc"])
    if state.get("use_2450"):
        # ponytail: 2450 read modelled with the 2182's NPLC timing -- unmeasured, like the other knobs
        read_s *= 2
    reads_s = (reversal_avg_s(state["n_averages"], state["source_delay_s"], read_s)
               if state["reversal_enabled"] else state["n_averages"] * read_s)

    rc = RunCost(n_pts * n_series)
    rc.each("settle", state["settling_time_s"])
    rc.each("field read", read_field_s(gcfg))
    rc.each("2182 + 2450 reads" if state.get("use_2450") else "2182 reads", reads_s)
    rc.each("overhead", POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    for i, hop in enumerate(field_hops(currents_A, n_series)):
        typ = magnet_move_s(hop, mcfg)
        rc.at("magnet", typ, i)
    for k in range(n_series):
        rc.at("per-file", PER_FILE_S, k * n_pts)
    rc.at("per-run", PER_RUN_S, 0)
    if n_pts:   # teardown: 6221 off, gate ramp-down, magnet ramp-down from the last field point
        rc.tail("ramps", 2 * GPIB_TXN_S + (GATE_RAMP_S if state["enable_gate"] else 0.0)
                + magnet_move_s(currents_A[-1], mcfg, with_field=False))
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    magnet_cfg: MagnetConfig
    gauss_cfg: GaussmeterConfig
    acq_cfg: AcquisitionConfig
    currents_A: np.ndarray
    sense_currents_A: List[float]
    sample: str
    device: str
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    series: str
    gate_cfg: Optional[GateConfig] = None
    gate_voltages_V: Optional[List[float]] = None
    temp_cfg: Optional[TemperatureControllerConfig] = None
    data_root: Path = _DEFAULT_DATA_DIR
    run_cost: Optional[RunCost] = None      # modelled seconds per point (progress bar + ETA)
    smu_cfg: Optional[SMUConfig] = None     # set -> 2450 is the source; sense_currents_A are its levels
    show_2wire: bool = False                # plot the 2450's 2-wire R alongside V

    @property
    def gate_series_values(self) -> List[Optional[float]]:
        """[None] for a single gate-less run, else one entry per gate voltage."""
        return list(self.gate_voltages_V) if self.gate_voltages_V else [None]

    @property
    def series_values(self) -> List[tuple[float, Optional[float]]]:
        """Cross product of sense currents x gate voltages -- one complete
        field sweep per (sense_current, gate_voltage) pair, each saved to
        its own file. A single sense current and a gate-less/single-gate
        run degenerates to today's plain gate-voltage series."""
        return list(itertools.product(self.sense_currents_A, self.gate_series_values))

    @property
    def total_points(self) -> int:
        return len(self.currents_A) * len(self.series_values)


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                         status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """
    Universal + measurement-specific header/index fields for ONE run within
    this (possibly multi-file, one-per-gate-voltage) session -- `ctx` is
    that particular iteration's RunContext, not a single plan-wide one (see
    instruments/data_naming.py's allocate_run() -- called fresh per
    iteration for this suite). `extra` carries this iteration's own values
    (gate_voltage_V) on top of the plan-wide header_extra.

    T_setpoint_K is the nominal value used to build the filename's T###K
    token; T_K is the MEASURED mean (temperature_1_K), left blank rather
    than backfilled with the setpoint when nothing was actually measured.
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

def resolve_state(state: dict) -> dict:
    """Add the derived keys build_summary() / build_plan() read — the parsed
    lists/sweeps, each with its parse error — to a state of raw field values.
    Pure: used by the TUI's parse_state()."""
    state["gate_voltage_list"], state["gate_parse_error"] = \
        try_parse(state["gate_voltage_values"]) if state["enable_gate"] else ([], None)
    state["sense_current_list"], state["sense_current_parse_error"] = try_parse(state["sense_current_values"])
    state["source_voltage_list"], state["source_voltage_parse_error"] = \
        try_parse(state["source_voltage_values"]) if smu_voltage_mode(state) else ([], None)
    state["sweep_rows_parsed"], state["sweep_rows_parse_error"] = try_parse(state["sweep_rows"], parse_sweep_rows)
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
        errors.append("Device is required (e.g. HB3, SV2).")

    src_name = "2450" if state.get("use_2450") else "6221"
    resources = [state["smu_visa_resource" if state.get("use_2450") else "source_visa_resource"],
                 state["voltmeter_visa_resource"]]
    if state["enable_gate"]:
        resources.append(state["gate_visa_resource"])
    if len(set(resources)) < len(resources):
        errors.append(
            f"Source ({src_name}), voltmeter (2182)"
            + (", and gate (2400)" if state["enable_gate"] else "")
            + " VISA resources must all be different."
        )

    if state.get("use_2450"):
        what = "2-wire I" if smu_voltage_mode(state) else "2-wire V"
        info.append(f"Source: 2450 sourcing {'V' if smu_voltage_mode(state) else 'I'} — "
                    f"{what} logged, plot {'on' if state.get('plot_2wire') else 'off'}")
    if smu_voltage_mode(state):
        if state.get("source_voltage_parse_error"):
            errors.append(f"Source voltage list: {state['source_voltage_parse_error']}")
        else:
            v_list = state.get("source_voltage_list", [])
            if any(v == 0 for v in v_list):
                errors.append("Source voltage must be nonzero (2-wire R divides by it).")
            over = [v for v in v_list if abs(v) > state["smu_voltage_limit_V"]]
            if over:
                errors.append(f"Source voltage(s) {over} exceed the 2450 limit "
                              f"±{state['smu_voltage_limit_V']:g} V.")
            if len(v_list) > 1:
                info.append(f"Source voltages: {', '.join(format_si(v, 'V') for v in v_list)} — "
                            f"{len(v_list)} field sweeps, one file each")
            elif v_list:
                info.append(f"Source voltage: {format_si(v_list[0], 'V')}")
        if state["smu_compliance_current_A"] <= 0:
            errors.append("2450 current compliance must be > 0 A.")
    elif state.get("sense_current_parse_error"):
        errors.append(f"Sense current list: {state['sense_current_parse_error']}")
    else:
        current_list = state.get("sense_current_list", [])
        if any(i == 0 for i in current_list):
            errors.append("Sense current must be nonzero (resistance divides by it).")
        if len(current_list) > 1:
            currents_str = ", ".join(format_si(i, "A") for i in current_list)
            info.append(f"Sense currents: {currents_str} — {len(current_list)} field sweeps, one file each")
        elif len(current_list) == 1:
            info.append(f"Sense current: {format_si(current_list[0], 'A')}")

    if not smu_voltage_mode(state) and state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")

    if state["reversal_enabled"]:
        info.append("Reversal: on — ±I each rep")
    else:
        info.append("Reversal: off — fixed +I")

    read_s = read_time_s(state["nplc"])
    info.append(f"2182 read: ≈ {read_s * 1000:.0f} ms — NPLC {state['nplc']:g}")

    n_gate_series = 1
    if state["enable_gate"]:
        if state["gate_voltage_limit_V"] <= 0:
            errors.append("Gate voltage limit must be > 0 V.")
        if state.get("gate_parse_error"):
            errors.append(f"Gate voltage list: {state['gate_parse_error']}")
            gate_list: list[float] = []
        else:
            gate_list = state.get("gate_voltage_list", [])
            over_limit = [v for v in gate_list if abs(v) > state["gate_voltage_limit_V"]]
            if over_limit:
                errors.append(
                    f"Gate voltage(s) {over_limit} exceed the configured limit "
                    f"±{state['gate_voltage_limit_V']:g} V."
                )
        n_gate_series = len(gate_list)
        if n_gate_series > 1:
            info.append(f"Gate: {', '.join(format_si(v, 'V') for v in gate_list)} — "
                        f"{n_gate_series} field sweeps per sense current, one file each")
        elif n_gate_series == 1:
            info.append(f"Gate: {format_si(gate_list[0], 'V')} — fixed")
    else:
        info.append("Gate: off")

    n_sweep_points = 0
    resolved: list = []
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
        resolved = build_segmented_sweep(rows, state["bidirectional_sweep"])
        n_sweep_points = len(resolved)
        n_raw = sum(n for _, _, n in rows)
        n_merged = (2 * n_raw if state["bidirectional_sweep"] else n_raw) - n_sweep_points
        merged_note = f", {n_merged} shared boundary point(s) merged" if n_merged else ""
        info.append(f"Field sweep: {n_sweep_points} points — {len(rows)} row(s)"
                     f"{', bidirectional' if state['bidirectional_sweep'] else ''}"
                     f"{merged_note}")
    info.append(f"Field read: Lake Shore 475 — {state['gaussmeter_visa_resource']}")
    tol_mT = state["field_settle_tolerance_mT"]
    if tol_mT <= 0:
        warnings.append("Field-settle tolerance is 0 — every magnet step will wait the "
                         "full settle timeout before acquiring.")
    elif tol_mT < 0.01:
        warnings.append(f"Field-settle tolerance {tol_mT:g} mT is below the 475's typical "
                         "reading noise — points may stall until the settle timeout.")

    info.extend(run_costs(resolved, state).lines())

    temperature_summary(state, info, warnings, errors)

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
    axes = []
    if len(level_list(state)) > 1:
        axes.append("source voltage" if smu_voltage_mode(state) else "sense current")
    if state.get("enable_gate") and len(state.get("gate_voltage_list", [])) > 1:
        axes.append("gate voltage")
    suffix = f" (one file per {' × '.join(axes)})" if axes else ""
    return f"{preview}_<timestamp>.csv{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue", show_2wire: bool = False) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if show_2wire:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        ax2.set_ylabel("2-wire R (Ω, 2450)")
        ax2.grid(True, alpha=0.3)
        bottom = ax2
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax2 = None
        bottom = ax
    try:
        fig.canvas.manager.set_window_title("DC Spin-Valve live measurement")
    except Exception:
        pass
    bottom.set_xlabel("Magnetic field (mT)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Live measurement — field sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    cmap = plt.get_cmap("tab10")
    lines: dict[int, "plt.Line2D"] = {}
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
            if idx not in lines:
                label = record.get("series_label")
                (line,) = ax.plot([], [], "o-", color=cmap(idx % 10), label=label)
                lines[idx] = line
                if ax2 is not None:
                    (lines2[idx],) = ax2.plot([], [], "o-", color=cmap(idx % 10))
                series_data[idx] = ([], [], [])
                new_series = True
            xs, ys, r2s = series_data[idx]
            x = record.get("magnet_field_mT")
            xs.append(x if x is not None else record["point_index"])
            ys.append(record["voltage_V"])
            r2s.append(record.get("resistance_2w_ohm", float("nan")))
            updated.add(idx)
        if updated:
            for idx in updated:
                xs, ys, r2s = series_data[idx]
                lines[idx].set_data(xs, ys)
                if idx in lines2:
                    lines2[idx].set_data(xs, r2s)
            if new_series and any(l.get_label() and not l.get_label().startswith("_") for l in lines.values()):
                ax.legend(loc="best", fontsize=8)
            for a in (ax, ax2):
                if a is not None:
                    a.relim()
                    a.autoscale_view()
        return (*lines.values(), *lines2.values())

    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path,
                           plan: Optional["MeasurementPlan"] = None, comment: str = "") -> None:
    """`records` is ONE run's points -- with several sense currents / gate
    voltages each run is saved (and plotted) on its own, exactly like a
    manual run.

    `plan`/`comment` add a small "at a glance" text annotation -- this
    run's sense current and gate voltage (read from its records, so they
    are right for every run of a series), plus the operator's comment.
    Called once when the run ends (comment="") and again, to overwrite the
    PNG in place, once the operator's comment is known."""
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show_2wire = (plan is not None and plan.show_2wire
                  and any("resistance_2w_ohm" in r for r in records))
    if show_2wire:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(7, 5))

    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]
    ax.plot(xs, [r["voltage_V"] for r in records], ".-", color="tab:blue")
    if show_2wire:
        ax2.plot(xs, [r.get("resistance_2w_ohm", float("nan")) for r in records], ".-", color="tab:orange")
        ax2.set_ylabel("2-wire R (Ω, 2450)")
        ax2.grid(alpha=0.3)

    (ax2 if show_2wire else ax).set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Measurement result")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lines: list[str] = []
    if plan is not None:
        sense_currents = sorted({r["sense_current_A"] for r in records
                                  if r.get("sense_current_A") is not None})
        if plan.smu_cfg is not None and plan.smu_cfg.source_function == "voltage":
            source_voltages = {r["smu_voltage_V"] for r in records if r.get("smu_voltage_V") is not None}
            if len(source_voltages) == 1:
                lines.append(f"Source voltage (2450): {format_si(source_voltages.pop(), 'V')}")
        elif len(sense_currents) == 1:
            lines.append(f"Sense current: {format_si(sense_currents[0], 'A')}")
        gate_voltages = sorted({r["gate_voltage_V"] for r in records
                                 if r.get("gate_voltage_V") is not None})
        if len(gate_voltages) == 1:
            lines.append(f"Gate voltage: {format_si(gate_voltages[0], 'V')}")
    if comment:
        lines.append(f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}")
    if lines:
        fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color="0.4", va="bottom")
        fig.subplots_adjust(bottom=0.08 + 0.045 * len(lines))

    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot to '%s'", png_path)


# ─────────────────────────────────────────────────────────────────────────────
# Logging -> RichLog relay
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Plan + run  ── pure, used by the TUI RunScreen
# ─────────────────────────────────────────────────────────────────────────────

def build_plan(state: dict, data_root: Path) -> MeasurementPlan:
    """One parsed, validated run request from a state dict. Pure."""
    levels = level_list(state)
    smu_cfg = None
    if state.get("use_2450"):
        voltage_mode = smu_voltage_mode(state)
        smu_cfg = SMUConfig(
            visa_resource=state["smu_visa_resource"],
            source_function="voltage" if voltage_mode else "current",
            compliance_current_A=state["smu_compliance_current_A"],
            compliance_voltage_V=state["compliance_V"],
            nplc=state["nplc"],
            four_wire=False,
            source_limit_V=state["smu_voltage_limit_V"],
            # the typed current list IS the intent (the 6221 path has no extra ceiling either)
            source_limit_A=SMUConfig.source_limit_A if voltage_mode else max(abs(i) for i in levels),
        )
    # In 2450 mode only source_delay_s (and the current-mode compliance) is used.
    src_cfg = SourceConfig(
        visa_resource=state["source_visa_resource"],
        sense_current_A=levels[0],
        compliance_V=state["compliance_V"],
        source_delay_s=state["source_delay_s"],
    )
    volt_cfg = VoltmeterConfig(
        visa_resource=state["voltmeter_visa_resource"],
        nplc=state["nplc"],
        auto_range=state["auto_range"],
    )
    magnet_cfg = MagnetConfig(
        visa_resource=state["magnet_visa_resource"],
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
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"],
        field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
        reversal_enabled=state["reversal_enabled"],
        n_averages=state["n_averages"],
        output_file=str(data_root / "dc_spin_valve.csv"),  # placeholder — overwritten per series
    )

    currents_A = build_segmented_sweep(
        state["sweep_rows_parsed"], state["bidirectional_sweep"],
    )

    gate_cfg = None
    gate_voltages_V = None
    if state["enable_gate"]:
        gate_cfg = GateConfig(
            visa_resource=state["gate_visa_resource"],
            gate_voltage_limit_V=state["gate_voltage_limit_V"],
            compliance_current_A=state["gate_compliance_current_A"],
        )
        gate_voltages_V = state["gate_voltage_list"]

    temp_cfg = temperature_cfg_from_state(state)

    header_extra = {
        "source_instrument": "2450" if smu_cfg is not None else "6221",
        "source_function": smu_cfg.source_function if smu_cfg is not None else "current",
        **({"smu_compliance_current_A": state["smu_compliance_current_A"]}
           if smu_cfg is not None and smu_cfg.source_function == "voltage"
           else {"compliance_V": state["compliance_V"]}),
        "reversal_enabled": state["reversal_enabled"],
        "n_averages": state["n_averages"],
        "settling_time_s": state["settling_time_s"],
        "field_sweep_rows_A": state["sweep_rows_parsed"],
    }
    # A "series" tag only means something for an actual family of runs
    # (>1 sense current and/or >1 gate voltage) -- a single-run session
    # gets no series tag.
    series = ""
    if len(levels) > 1 or len(gate_voltages_V or []) > 1:
        series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                  f"{datetime.now():%Y%m%dT%H%M%S}")

    return MeasurementPlan(
        src_cfg=src_cfg, volt_cfg=volt_cfg,
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, acq_cfg=acq_cfg,
        currents_A=currents_A, sense_currents_A=levels,
        data_root=data_root,
        sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series=series,
        gate_cfg=gate_cfg, gate_voltages_V=gate_voltages_V,
        temp_cfg=temp_cfg, run_cost=run_costs(currents_A, state),
        smu_cfg=smu_cfg, show_2wire=smu_cfg is not None and state.get("plot_2wire", True),
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
    """Connect, then one field sweep (own run number, own file) per (sense
    current, gate voltage) pair, each recorded + finalized by record_run()
    before the next; always shut the instruments down. Pure — the TUI's
    RunScreen passes its own callbacks."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = smu = voltmeter = gate = magnet = gaussmeter = temp_ctrl = None
    use_2450 = plan.smu_cfg is not None
    voltage_mode = use_2450 and plan.smu_cfg.source_function == "voltage"
    try:
        if use_2450:
            on_status("Connecting to Keithley 2450 & 2182 …")
            smu = connect_smu(plan.smu_cfg)
        else:
            on_status("Connecting to Keithley 6221 & 2182 …")
            source = connect_source(plan.src_cfg)
        voltmeter = connect_voltmeter(plan.volt_cfg)

        if plan.gate_cfg is not None:
            on_status("Connecting gate (Keithley 2400) …")
            gate = connect_gate(plan.gate_cfg)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC (temperature) …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        on_status("Connecting magnet power supply …")
        magnet = connect_magnet(plan.magnet_cfg)
        on_status("Connecting gaussmeter …")
        gaussmeter = connect_gaussmeter(plan.gauss_cfg)

        n_currents = len(plan.sense_currents_A)
        n_gates = len(plan.gate_series_values)
        for series_idx, (I_sense, gate_V) in enumerate(plan.series_values):
            if stop_event.is_set():
                break
            plan.src_cfg.sense_current_A = I_sense
            if use_2450:
                set_source_level(smu, plan.smu_cfg, I_sense)
            label_parts = []
            if n_currents > 1:
                label_parts.append(f"V={I_sense:g}V" if voltage_mode else f"I={I_sense:g}A")
            if gate_V is not None and n_gates > 1:
                label_parts.append(f"Vg={gate_V:g}V")
            label = ", ".join(label_parts) or None

            key_axis = None
            if n_gates > 1:
                key_axis = ("gate_V", gate_V)
            elif n_currents > 1 and not voltage_mode:   # no locked kind for a source voltage
                key_axis = ("current_A", I_sense)

            if gate_V is not None:
                on_status(f"Setting gate to {gate_V:g} V …")
                set_gate_voltage(gate, plan.gate_cfg, gate_V)

            extra = {"source_voltage_V" if voltage_mode else "sense_current_A": I_sense}
            if gate_V is not None:
                extra["gate_voltage_V"] = gate_V

            # A fresh RunContext (own run number, own file) EVERY iteration --
            # never reuse one across the series.
            ctx = allocate_run(
                plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                temperature_setpoint_K=plan.temperature_setpoint_K,
                key_axis=key_axis, series=plan.series,
            )
            run_contexts.append(ctx)
            run_extras.append(extra)
            on_run_label(f"Run #{ctx.run_str}")
            plan.acq_cfg.output_file = str(ctx.raw_path)
            points = [
                FieldPoint(
                    magnet_current_A=I,
                    set_action=lambda I=I: set_magnet_current(
                        magnet, plan.magnet_cfg, I, gaussmeter, plan.gauss_cfg,
                        plan.acq_cfg.field_settle_tolerance_mT, stop_event),
                )
                for I in plan.currents_A
            ]

            on_status("Running field sweep …" if not label_parts
                      else f"Running field sweep ({', '.join(label_parts)}) …")
            record_run(
                plan.data_root, ctx,
                lambda records, status, _ctx=ctx, _x=extra: build_header_fields(
                    plan, _ctx, records, status=status, comment="", extra=_x),
                lambda point_cb, write_csv, _points=points, _gv=gate_V, _level=I_sense: run_measurement(
                    source, voltmeter, plan.src_cfg, plan.acq_cfg, _points,
                    stop_event=stop_event, on_point=point_cb,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg, gate_voltage_V=_gv,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg, write_csv=write_csv,
                    smu=smu, smu_cfg=plan.smu_cfg, smu_level=_level),
                stop_event, on_point=on_point,
                tags={"series_index": series_idx, "series_label": label},
                on_finished=on_run_finished)
    finally:
        # 6221 output off first (immediate, no current into the DUT), so the
        # magnet can start its ramp-down right away rather than waiting behind it.
        if source is not None:
            safe_shutdown("source", lambda: shutdown_source(source))
        if smu is not None:
            safe_shutdown("2450", lambda: shutdown_smu(smu))
        if gate is not None:
            safe_shutdown("gate", lambda: shutdown_gate(gate))
        if magnet is not None:
            safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
        if gaussmeter is not None:
            safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (RunScreen calls this)."""
    _save_measurement_png(records, png_path, plan=plan, comment=comment)


# ─────────────────────────────────────────────────────────────────────────────
# Run screen  ── executes the plan in a worker thread, shows live progress
# ─────────────────────────────────────────────────────────────────────────────

class RunScreen(MeasurementRunScreen):
    TABLE_COLUMNS = ("#", "I_sense (A)", "Vg (V)", "I_magnet (A)", "B (mT)", "V (V)", "R (Ω)", "R_2w (Ω)", "n_avg", "T1 (K)", "T2 (K)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE

    def live_plot_args(self):
        return (_live_plot_worker, self.plan.show_2wire)

    def table_row(self, record: dict) -> tuple:
        Isense = record.get("sense_current_A")
        I = record.get("magnet_current_A")
        B = record.get("magnet_field_mT")
        Vg = record.get("gate_voltage_V")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        R2 = record.get("resistance_2w_ohm")
        return (
            str(record["point_index"] + 1),
            f"{Isense:.4g}" if Isense is not None else "—",
            f"{Vg:.4g}" if Vg is not None else "—",
            f"{I:.4f}" if I is not None else "—",
            f"{B:.2f}" if B is not None else "—",
            f"{record['voltage_V']:.4e}",
            f"{record['resistance_ohm']:.5g}",
            f"{R2:.5g}" if R2 is not None else "—",
            str(record["n_averages"]),
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class DCSpinValveApp(MeasurementApp):
    TITLE = "DC Spin-Valve / Field Sweep"
    SUB_TITLE = "Keithley 6221 (or 2450) + 2182 + 2400 · magnet field sweep"

    # Session data root — fallback until _load_settings()/the identity bar's
    # "Data root" field replaces it. Read in compose(), so it must exist here.
    data_root: Path = _DEFAULT_DATA_DIR

    SWITCH_DEPENDENTS = {
        "enable_gate": tuple(GATE_FIELD_IDS),
        "use_2450": ("smu_source_voltage", "plot_2wire", "smu_visa_resource"),
        "smu_source_voltage": tuple(SMU_V_FIELD_IDS),
        "enable_temperature": tuple(TEMPERATURE_FIELD_IDS),
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

    /* Collapsible tiers -- precision knobs + instrument wiring, folded by default */
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
    .sweep-rows { height: 5; margin-bottom: 1; }
    .switch-row { height: auto; }
    .switch-row Label { padding-left: 1; content-align: left middle; width: 1fr; height: auto; min-height: 3; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; }
    .card-desc { color: $text-muted; margin-bottom: 1; }
    #actionbar { height: 3; align: center middle; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                # ── File & run identity ── changes every run, always on top ──
                yield identity_bar(DEFAULTS, _DEFAULT_DATA_DIR, self.data_root)

                # ── Tier 1: what defines this run — always visible ──────────
                with Vertical(classes="param-grid"):
                    yield card(
                        "Field sweep (Kepco magnet)",
                        sweep_rows_field("sweep_rows", DEFAULTS["sweep_rows"]),
                        switch_field("bidirectional_sweep", "Bidirectional (retrace the merged rows)",
                                     DEFAULTS["bidirectional_sweep"]),
                    )
                    yield card(
                        "Sense source (Keithley 6221 / 2450)",
                        switch_field("use_2450", "Use 2450 instead of 6221",
                                     DEFAULTS["use_2450"]),
                        switch_field("smu_source_voltage", "2450 sources voltage (off = current)",
                                     DEFAULTS["smu_source_voltage"]),
                        switch_field("plot_2wire", "Plot 2450 2-wire R (always saved)",
                                     DEFAULTS["plot_2wire"]),
                        field("sense_current_values", "Sense current (A)",
                              DEFAULTS["sense_current_values"], kind="text",
                              hint="Comma-separate for one sweep + file per value."),
                        field("source_voltage_values", "2450 source voltage (V)",
                              DEFAULTS["source_voltage_values"], kind="text",
                              hint="Voltage mode only. Comma-separate for one file per value."),
                        switch_field("reversal_enabled", "Reverse current each rep (+I/-I)",
                                     DEFAULTS["reversal_enabled"]),
                        Label(
                            "Off for bias-direction-dependent devices: fixed +I, plain average.",
                            classes="hint",
                        ),
                    )
                    yield card(
                        "Gate voltage (Keithley 2400, optional)",
                        switch_field("enable_gate", "Enable gate (Keithley 2400)",
                                     DEFAULTS["enable_gate"]),
                        field("gate_voltage_values", "Gate voltage (V)",
                              DEFAULTS["gate_voltage_values"], kind="text",
                              hint="Comma-separate for one sweep + file per value."),
                    )
                    yield temperature_card(DEFAULTS)

                # ── Tier 2: precision / speed knobs — collapsed ─────────────
                with Collapsible(title="Acquisition & filter settings", collapsed=True):
                    with Vertical(classes="param-grid"):
                        yield card(
                            "Source & voltmeter",
                            field("compliance_V", "Compliance voltage (V)",
                                  DEFAULTS["compliance_V"],
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("nplc", "NPLC (integration time)", DEFAULTS["nplc"],
                                  hint="Bigger = quieter but slower.",
                                  validators=[Number(minimum=0.01, failure_description="must be > 0")]),
                            switch_field("auto_range", "Auto-range", DEFAULTS["auto_range"]),
                        )
                        yield card(
                            "Acquisition timing",
                            field("settling_time_s", "Settling time per point (s)",
                                  DEFAULTS["settling_time_s"],
                                  hint="Wait after a field change.",
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("n_averages", "Voltage averages per point",
                                  DEFAULTS["n_averages"], kind="integer",
                                  hint="± pairs with reversal, plain samples without.",
                                  validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        )

                # ── Tier 3: instrument wiring & timing constants — collapsed ─
                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Instrument addresses",
                            field("source_visa_resource", "6221 (current source)",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            field("smu_visa_resource", "2450 (source, optional)",
                                  DEFAULTS["smu_visa_resource"], kind="text"),
                            field("voltmeter_visa_resource", "2182 (voltage)",
                                  DEFAULTS["voltmeter_visa_resource"], kind="text"),
                            field("gate_visa_resource", "2400 (gate)",
                                  DEFAULTS["gate_visa_resource"], kind="text"),
                            field("magnet_visa_resource", "Magnet (Kepco)",
                                  DEFAULTS["magnet_visa_resource"], kind="text"),
                            field("gaussmeter_visa_resource", "Gaussmeter (Lake Shore 475)",
                                  DEFAULTS["gaussmeter_visa_resource"], kind="text"),
                            field("temperature_visa_resource", "MercuryiTC",
                                  DEFAULTS["temperature_visa_resource"], kind="text"),
                            muted=True,
                        )
                        yield card(
                            "Source & gate limits",
                            field("source_delay_s", "Source delay (s)", DEFAULTS["source_delay_s"],
                                  hint="Settle after each ±I (or ±V) reversal."),
                            field("smu_compliance_current_A", "2450 current compliance (A)",
                                  DEFAULTS["smu_compliance_current_A"],
                                  hint="Voltage mode only."),
                            field("smu_voltage_limit_V", "2450 source voltage software limit (V)",
                                  DEFAULTS["smu_voltage_limit_V"]),
                            field("gate_voltage_limit_V", "Gate voltage software limit (V)",
                                  DEFAULTS["gate_voltage_limit_V"],
                                  hint="Hard safety ceiling."),
                            field("gate_compliance_current_A", "Gate leakage compliance (A)",
                                  DEFAULTS["gate_compliance_current_A"]),
                            muted=True,
                        )
                        yield card(
                            "Magnet ramp safety",
                            field("current_limit_A", "Software current limit (A)",
                                  DEFAULTS["current_limit_A"],
                                  hint="Hard safety ceiling."),
                            field("voltage_compliance_V", "Voltage compliance (V)",
                                  DEFAULTS["voltage_compliance_V"]),
                            field("ramp_step_A", "Ramp step (A)", DEFAULTS["ramp_step_A"]),
                            field("ramp_delay_s", "Ramp delay (s)", DEFAULTS["ramp_delay_s"]),
                            muted=True,
                        )
                        yield card(
                            "Gaussmeter & temperature sensors",
                            field("gaussmeter_n_averages", "Field readings averaged per point",
                                  DEFAULTS["gaussmeter_n_averages"], kind="integer",
                                  validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                            field("gaussmeter_read_delay_s", "Delay between readings (s)",
                                  DEFAULTS["gaussmeter_read_delay_s"]),
                            field("field_settle_tolerance_mT", "Field-settle tolerance (mT)",
                                  DEFAULTS["field_settle_tolerance_mT"],
                                  hint="Field settled when readings span less than this. Raise if points stall.",
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("temperature_sensor_uids", "MercuryiTC sensor board UID(s)",
                                  DEFAULTS["temperature_sensor_uids"], kind="text",
                                  hint="1-2 UIDs, comma-separated."),
                            muted=True,
                        )

            with Vertical(id="sidebar"):
                yield Static("Description", classes="sidebar-title")
                yield Static(DC_SPIN_VALVE_DESCRIPTION, classes="card-desc")
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

        self.query_one("#summary", Static).update(summary_markup(info, warnings, errors))
        self.query_one("#start", Button).disabled = bool(errors)

    # ── Start ────────────────────────────────────────────────────────────────

    def _build_plan(self, state: dict) -> MeasurementPlan:
        return build_plan(state, self.data_root)


def main() -> None:
    DCSpinValveApp().run()


if __name__ == "__main__":
    main()
