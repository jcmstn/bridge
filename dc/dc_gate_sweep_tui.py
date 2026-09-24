#!/usr/bin/env python3
"""
Textual TUI front-end for dc_gate_sweep.py
=============================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-31

Lets you edit the parameters that decide whether a gate-voltage transfer
curve is good or bad — sense current, gate range/step, voltmeter
integration time, timing — without touching the dataclasses in the script
itself.

An optional magnet current (single value, or a comma-separated list) parks
the Kepco magnet once before each gate sweep; the actual field is measured
live via the Lake Shore 475 and logged on every row. A list runs one
complete gate sweep per value, each saved to its own file and plotted
together in the same window with a different color.

Run with:
    python dc_gate_sweep_tui.py

Requirements:
    pip install textual matplotlib  (in addition to dc_gate_sweep.py's own deps)
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import textwrap
import threading
import time
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
    Static,
)

from dc.dc_gate_sweep import (
    AcquisitionConfig,
    GateConfig,
    GatePoint,
    GaussmeterConfig,
    MagnetConfig,
    SourceConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    connect_gate,
    connect_gaussmeter,
    connect_magnet,
    connect_source,
    connect_temperature_controller,
    connect_voltmeter,
    read_field_mT,
    run_measurement,
    set_magnet_current,
    shutdown_gate,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_source,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import linear_sweep, safe_shutdown, sweep_point_count, try_parse
from instruments.data_dir import validate_directory
from instruments.data_naming import (
    RunContext,
    allocate_run,
    record_run,
    preview_raw_filename,
)
from instruments.keithley2182 import read_time_s
from instruments.kepco_magnet import magnet_move_s
from instruments.lakeshore475 import read_field_s
from instruments.run_time import (
    GATE_RAMP_S, GPIB_TXN_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
    RunCost,
)
from instruments.summary_lines import summary_markup
from instruments.tui_common import (
    MeasurementApp,
    MeasurementRunScreen,
    card,
    field,
    format_si,
    identity_bar,
    parse_sensor_uids,
    switch_field,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
)

log = logging.getLogger("dc_gate_sweep_tui")

# Data/settings live outside "bridge" (a sibling of it). _DEFAULT_DATA_DIR is
# the fallback data-convention "data root"; the real root is chosen per run in
# the identity bar's "Data root" field -- see dc_hall_measurement_tui.py.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "dc_gate_sweep_tui_settings.json"

# Locked type code (see instruments/data_naming.py) — never deviates.
MEASUREMENT_TYPE = "GSWP"

DC_GATE_SWEEP_DESCRIPTION = (
    "6221 fixed current · 2400 gate sweep · 2182 voltage. Optional Kepco field "
    "parked per value, read by Lake Shore 475."
)

# Wiring schematic — shown on this program's card in bridge_tui.py.
DC_GATE_SWEEP_SCHEMATIC = """\
  KEITHLEY 6221  (fixed DC sense current)
    Output ──▶ DUT ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ across the DUT

  KEITHLEY 2400  (gate source — the swept axis)
    Output ──▶ gate electrode

  Field  (optional, single value or list — parked, not swept)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample
"""


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors dc_gate_sweep.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "source_visa_resource": "GPIB0::20::INSTR",
    "voltmeter_visa_resource": "GPIB0::7::INSTR",
    "sense_current_values": "0.000001",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "nplc": "5",
    "auto_range": True,
    "settling_time_s": "0.2",
    "n_averages": "5",
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    "gate_visa_resource": "GPIB0::25::INSTR",
    "gate_voltage_limit_V": "20.0",
    "gate_compliance_current_A": "0.000001",
    "gate_min_V": "-10.0",
    "gate_max_V": "10.0",
    "step_V": "0.5",
    "bidirectional_sweep": True,
    "enable_field": False,
    "magnet_visa_resource": "GPIB0::6::INSTR",
    "current_limit_A": "35",
    "voltage_compliance_V": "15.0",
    "ramp_step_A": "0.1",
    "ramp_delay_s": "0.05",
    "field_settle_s": "1.0",
    "field_settle_tolerance_mT": "0.02",
    "gaussmeter_visa_resource": "GPIB0::12::INSTR",
    "gaussmeter_n_averages": "10",
    "gaussmeter_read_delay_s": "0.05",
    "field_current_values": "0.0",
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
}

NUMERIC_FIELDS: dict = {
    "compliance_V": float,
    "source_delay_s": float,
    "nplc": float,
    "settling_time_s": float,
    "n_averages": int,
    "gate_voltage_limit_V": float,
    "gate_compliance_current_A": float,
    "gate_min_V": float,
    "gate_max_V": float,
    "step_V": float,
    "current_limit_A": float,
    "voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "field_settle_s": float,
    "field_settle_tolerance_mT": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "gate_visa_resource",
               "device", "cooldown", "magnet_visa_resource",
               "gaussmeter_visa_resource", "field_current_values", "sense_current_values",
               "temperature_visa_resource", "temperature_sensor_uids", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K"]
FIELD_FIELD_IDS = [
    "magnet_visa_resource", "current_limit_A", "voltage_compliance_V",
    "ramp_step_A", "ramp_delay_s", "field_settle_s", "field_settle_tolerance_mT",
    "gaussmeter_visa_resource", "gaussmeter_n_averages", "gaussmeter_read_delay_s",
    "field_current_values",
]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def run_costs(n_sweep_points: int, state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (series-major: one full gate sweep per magnet park x sense current,
    field outer / sense inner). Also drives the run screen's progress bar,
    so estimate and live ETA cannot disagree."""
    n_sense = max(1, len(state.get("sense_current_list") or []))
    fields = (state.get("field_current_list") or [None]) if state["enable_field"] else [None]
    n_series = len(fields) * n_sense
    has_temp = state["enable_temperature"] and bool(parse_sensor_uids(state["temperature_sensor_uids"]))

    rc = RunCost(n_sweep_points * n_series)
    rc.each("settle", state["settling_time_s"])
    rc.each("2182 reads", state["n_averages"] * read_time_s(state["nplc"]))
    rc.each("overhead", GPIB_TXN_S + POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    for k in range(n_series):
        rc.at("per-file", PER_FILE_S, k * n_sweep_points)
    rc.at("per-run", PER_RUN_S, 0)
    teardown = 2 * GPIB_TXN_S + GATE_RAMP_S          # 6221 off + gate ramp-down
    if state["enable_field"]:
        mcfg = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
        gcfg = GaussmeterConfig(n_averages=state["gaussmeter_n_averages"],
                                read_delay_s=state["gaussmeter_read_delay_s"])
        parked = None                                # the loop only re-parks when the field current changes
        for k in range(n_series):
            current_A = fields[k // n_sense]
            if current_A is None or current_A == parked:
                continue
            typ, worst = magnet_move_s(abs(current_A - (parked or 0.0)), mcfg)
            i = k * n_sweep_points
            rc.at("magnet", typ, i, worst_extra=worst - typ)
            rc.at("field dwell", state["field_settle_s"], i)
            rc.at("field read", read_field_s(gcfg), i)
            parked = current_A
        if parked is not None:
            teardown += magnet_move_s(parked, mcfg, with_field=False)[0]
    rc.tail("ramps", teardown)
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    gate_cfg: GateConfig
    acq_cfg: AcquisitionConfig
    gate_voltages_V: np.ndarray
    sample: str
    device: str
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    series: str
    sense_currents_A: List[float]
    magnet_cfg: Optional[MagnetConfig] = None
    gauss_cfg: Optional[GaussmeterConfig] = None
    field_currents_A: Optional[List[float]] = None
    field_settle_s: float = 1.0
    field_settle_tolerance_mT: float = 0.02
    temp_cfg: Optional[TemperatureControllerConfig] = None
    data_root: Path = _DEFAULT_DATA_DIR
    run_cost: Optional[RunCost] = None      # modelled seconds per point (progress bar + ETA)

    @property
    def series_values(self) -> List[tuple[Optional[float], float]]:
        """Cross product of field (magnet-park) currents x sense currents --
        one complete gate sweep per pair, each saved to its own file. Field
        is outer (a physical ramp+settle) and sense is inner (an instant
        config mutation) -- see dc_spin_valve_tui.py for the same
        nested-product pattern."""
        field_values = list(self.field_currents_A) if self.field_currents_A else [None]
        return list(itertools.product(field_values, self.sense_currents_A))

    @property
    def total_points(self) -> int:
        return len(self.gate_voltages_V) * len(self.series_values)


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                         status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """Universal + measurement-specific header/index fields for ONE run
    within this (possibly multi-file, one-per-magnet-current) session --
    see dc_spin_valve_tui.py's build_header_fields for the full rationale."""
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
    Pure: shared by the TUI's and the web page's parse_state()."""
    state["field_current_list"], state["field_parse_error"] = \
        try_parse(state["field_current_values"]) if state["enable_field"] else ([], None)
    state["sense_current_list"], state["sense_current_parse_error"] = try_parse(state["sense_current_values"])
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

    resources = [state["source_visa_resource"], state["voltmeter_visa_resource"], state["gate_visa_resource"]]
    if len(set(resources)) < len(resources):
        errors.append("Source (6221), voltmeter (2182), and gate (2400) VISA resources must all be different.")

    sense_list: list[float] = []
    if state.get("sense_current_parse_error"):
        errors.append(f"Sense current list: {state['sense_current_parse_error']}")
    else:
        sense_list = state.get("sense_current_list", [])
        zero = [i for i in sense_list if i == 0]
        if zero:
            errors.append("Sense current must be nonzero (resistance divides by it).")
        elif len(sense_list) > 1:
            info.append(f"Sense currents: {', '.join(format_si(i, 'A') for i in sense_list)} — "
                        f"{len(sense_list)} sweeps per field value, one file each")
        elif sense_list:
            info.append(f"Sense current: {format_si(sense_list[0], 'A')}")

    if state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")

    read_s = read_time_s(state["nplc"])
    info.append(f"2182 read: ≈ {read_s * 1000:.0f} ms — NPLC {state['nplc']:g}")

    max_abs_Vg = max(abs(state["gate_min_V"]), abs(state["gate_max_V"]))
    if max_abs_Vg > state["gate_voltage_limit_V"]:
        errors.append(
            f"Gate sweep range (±{max_abs_Vg:g} V) exceeds the gate voltage limit "
            f"({state['gate_voltage_limit_V']:g} V)."
        )
    if state["gate_min_V"] == state["gate_max_V"]:
        warnings.append("gate_min equals gate_max — sweep will repeat a single point.")

    n_sweep_points = 0
    if state["step_V"] <= 0:
        errors.append("Gate sweep step size must be > 0 V.")
    else:
        try:
            n_sweep_points = sweep_point_count(state["gate_min_V"], state["gate_max_V"],
                                               state["step_V"], state["bidirectional_sweep"])
        except ValueError as exc:
            errors.append(str(exc))
    lo, hi = format_si(state["gate_min_V"], "V"), format_si(state["gate_max_V"], "V")
    direction = f"{lo} → {hi} → {lo}" if state["bidirectional_sweep"] else f"{lo} → {hi}"
    info.append(f"Gate sweep: {direction} — step {format_si(state['step_V'], 'V')}, "
                f"{n_sweep_points} points")

    if state["enable_field"]:
        if state.get("field_parse_error"):
            errors.append(f"Magnet current list: {state['field_parse_error']}")
            field_list: list[float] = []
        else:
            field_list = state.get("field_current_list", [])
            over_limit = [i for i in field_list if abs(i) > state["current_limit_A"]]
            if over_limit:
                errors.append(
                    f"Magnet current(s) {over_limit} exceed the configured limit "
                    f"±{state['current_limit_A']:g} A."
                )
        n_series = len(field_list)
        n_sense = max(1, len(sense_list))
        n_files = max(1, n_series) * n_sense
        if n_series > 1:
            info.append(f"Magnet currents: {', '.join(f'{i:g}' for i in field_list)} A — "
                        f"{n_series} gate sweeps, one file each")
        elif n_series == 1:
            info.append(f"Magnet current: {field_list[0]:g} A — parked, field read by 475")
        if n_files > max(1, n_series):
            info.append(f"Files: {n_files} — {n_sense} sense current(s) × "
                        f"{max(1, n_series)} field value(s)")
        tol_mT = state["field_settle_tolerance_mT"]
        if tol_mT <= 0:
            warnings.append("Field-settle tolerance is 0 — parking the magnet will wait the "
                             "full settle timeout every time.")
        elif tol_mT < 0.01:
            warnings.append(f"Field-settle tolerance {tol_mT:g} mT is below the 475's typical "
                             "reading noise — parking may stall until the settle timeout.")
        info.extend(run_costs(n_sweep_points, state).lines())
    else:
        n_sense = max(1, len(sense_list))
        info.append("Field: none — magnet untouched")
        if n_sense > 1:
            info.append(f"Files: {n_sense} — one per sense current")
        info.extend(run_costs(n_sweep_points, state).lines())

    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if not uids:
            warnings.append("Temperature logging is on but no sensor UID is set — "
                             "temperature columns will be empty.")
        else:
            info.append(f"Temperature: MercuryiTC {', '.join(uids)} — empty if unreachable")
    else:
        info.append("Temperature: off")

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
    n_field = len(state.get("field_current_list", [])) if state.get("enable_field") else 0
    n_sense = len(state.get("sense_current_list", []))
    n_files = max(1, n_field) * max(1, n_sense)
    suffix = f" (one file per run — {n_files} files)" if n_files > 1 else ""
    return f"{preview}_<timestamp>.csv{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("DC Gate Sweep live measurement")
    except Exception:
        pass
    ax.set_xlabel("Gate voltage (V)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Live measurement — gate transfer curve")
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
                record = queue.get_nowait()
            except Exception:
                break
            idx = record.get("series_index", 0)
            if idx not in lines:
                label = record.get("series_label")
                (line,) = ax.plot([], [], "o-", color=cmap(idx % 10), label=label)
                lines[idx] = line
                series_data[idx] = ([], [])
                new_series = True
            xs, ys = series_data[idx]
            xs.append(record["gate_voltage_V"])
            ys.append(record["voltage_V"])
            updated.add(idx)
        if updated:
            for idx in updated:
                xs, ys = series_data[idx]
                lines[idx].set_data(xs, ys)
            if new_series and any(l.get_label() and not l.get_label().startswith("_") for l in lines.values()):
                ax.legend(loc="best", fontsize=8)
            ax.relim()
            ax.autoscale_view()
        return tuple(lines.values())

    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path,
                           plan: Optional["MeasurementPlan"] = None, comment: str = "") -> None:
    """`records` is ONE run's points -- with several sense/field currents
    each run is saved (and plotted) on its own, exactly like a manual run.

    `plan`/`comment` add a small "at a glance" text annotation (the fixed
    sense current, the operator's comment) for context not already in the
    filename. Called once when the run ends (comment="") and again, to
    overwrite the PNG in place, once the operator's comment is known."""
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["gate_voltage_V"] for r in records], [r["voltage_V"] for r in records],
            ".-", color="tab:blue")

    ax.set_xlabel("Gate voltage (V)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Measurement result")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lines: list[str] = []
    sense_currents = sorted({r["sense_current_A"] for r in records if r.get("sense_current_A") is not None})
    if len(sense_currents) == 1:
        lines.append(f"Sense current: {format_si(sense_currents[0], 'A')}")
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
# Plan + run  ── pure, shared by the TUI RunScreen and web/dc/gate_sweep.py
# ─────────────────────────────────────────────────────────────────────────────

def build_plan(state: dict, data_root: Path) -> MeasurementPlan:
    """One parsed, validated run request from a state dict. Pure — shared by
    the TUI and the web page."""
    src_cfg = SourceConfig(
        visa_resource=state["source_visa_resource"],
        sense_current_A=state["sense_current_list"][0],
        compliance_V=state["compliance_V"],
        source_delay_s=state["source_delay_s"],
    )
    volt_cfg = VoltmeterConfig(
        visa_resource=state["voltmeter_visa_resource"],
        nplc=state["nplc"],
        auto_range=state["auto_range"],
    )
    gate_cfg = GateConfig(
        visa_resource=state["gate_visa_resource"],
        gate_voltage_limit_V=state["gate_voltage_limit_V"],
        compliance_current_A=state["gate_compliance_current_A"],
    )
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"],
        n_averages=state["n_averages"],
        output_file="",  # overwritten per series iteration in RunScreen
    )

    gate_voltages_V = linear_sweep(
        start=state["gate_min_V"], stop=state["gate_max_V"], step=state["step_V"],
        bidirectional=state["bidirectional_sweep"],
    )

    magnet_cfg = None
    gauss_cfg = None
    field_currents_A = None
    if state["enable_field"]:
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
        field_currents_A = state["field_current_list"]

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"],
                sensor_uids=uids,
            )

    header_extra = {
        "sense_current_A": state["sense_current_list"][0],
        "compliance_V": state["compliance_V"],
        "n_averages": state["n_averages"],
        "settling_time_s": state["settling_time_s"],
        "gate_sweep_V": [state["gate_min_V"], state["gate_max_V"], state["step_V"]],
    }
    series = ""
    if len(field_currents_A or [None]) * len(state["sense_current_list"]) > 1:
        series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                  f"{datetime.now():%Y%m%dT%H%M%S}")

    return MeasurementPlan(
        src_cfg=src_cfg, volt_cfg=volt_cfg, gate_cfg=gate_cfg, acq_cfg=acq_cfg,
        gate_voltages_V=gate_voltages_V, data_root=data_root,
        sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series=series,
        sense_currents_A=state["sense_current_list"],
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, field_currents_A=field_currents_A,
        field_settle_s=state["field_settle_s"],
        field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
        temp_cfg=temp_cfg, run_cost=run_costs(len(gate_voltages_V), state),
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
    """Connect, then one gate sweep (own run number, own file) per (parked
    magnet current, sense current) pair, each recorded + finalized by
    record_run() before the next; always shut the instruments down. Pure —
    the TUI's RunScreen and the web page each pass their own callbacks."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = voltmeter = gate = magnet = gaussmeter = temp_ctrl = None
    try:
        on_status("Connecting to Keithley 6221, 2182 & 2400 …")
        source = connect_source(plan.src_cfg)
        voltmeter = connect_voltmeter(plan.volt_cfg)
        gate = connect_gate(plan.gate_cfg)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC (temperature) …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        if plan.magnet_cfg is not None:
            on_status("Connecting magnet power supply …")
            magnet = connect_magnet(plan.magnet_cfg)
            on_status("Connecting gaussmeter …")
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)

        _unset = object()
        parked_field_A = _unset
        field_mT = None
        for series_idx, (field_current_A, sense_current_A) in enumerate(plan.series_values):
            if stop_event.is_set():
                break
            plan.src_cfg.sense_current_A = sense_current_A

            label_parts = []
            key_axis = None
            if field_current_A is not None:
                label_parts.append(f"I_mag={field_current_A:g}A")
                key_axis = ("current_A", field_current_A)
                if field_current_A != parked_field_A:
                    on_status(f"Parking magnet at {field_current_A:g} A …")
                    set_magnet_current(magnet, plan.magnet_cfg, field_current_A,
                                       gaussmeter, plan.gauss_cfg,
                                       plan.field_settle_tolerance_mT, stop_event)
                    time.sleep(plan.field_settle_s)
                    field_mT = read_field_mT(gaussmeter, plan.gauss_cfg)
                    log.info("Field parked: I_magnet=%.4f A  B=%.4f mT (measured)",
                             field_current_A, field_mT)
                    parked_field_A = field_current_A
            if len(plan.sense_currents_A) > 1:
                label_parts.append(f"I_sense={sense_current_A:g}A")
            label = ", ".join(label_parts) or None

            # A fresh RunContext (own run number, own file) EVERY iteration --
            # never reuse one across the series.
            ctx = allocate_run(
                plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                temperature_setpoint_K=plan.temperature_setpoint_K,
                key_axis=key_axis, series=plan.series,
            )
            extra = {"sense_current_A": sense_current_A, "magnet_current_A": field_current_A,
                     "magnet_field_mT": field_mT} if field_current_A is not None \
                else {"sense_current_A": sense_current_A}
            run_contexts.append(ctx)
            run_extras.append(extra)
            on_run_label(f"Run #{ctx.run_str}")
            plan.acq_cfg.output_file = str(ctx.raw_path)
            points = [GatePoint(gate_voltage_V=float(v)) for v in plan.gate_voltages_V]

            on_status("Running gate sweep …" if not label_parts
                      else f"Running gate sweep ({', '.join(label_parts)}) …")
            record_run(
                plan.data_root, ctx,
                lambda records, status, _ctx=ctx, _x=extra: build_header_fields(
                    plan, _ctx, records, status=status, comment="", extra=_x),
                lambda point_cb, write_csv, _points=points, _i=field_current_A, _b=field_mT: run_measurement(
                    source, voltmeter, gate, plan.src_cfg, plan.gate_cfg, plan.acq_cfg, _points,
                    stop_event=stop_event, on_point=point_cb,
                    magnet_current_A=_i, magnet_field_mT=_b,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg, write_csv=write_csv),
                stop_event, on_point=on_point,
                tags={"series_index": series_idx, "series_label": label},
                on_finished=on_run_finished)
    finally:
        # 6221 output off first (immediate, no current into the DUT), so the
        # magnet can start its ramp-down right away rather than waiting behind it.
        if source is not None:
            safe_shutdown("source", lambda: shutdown_source(source))
        if gate is not None:
            safe_shutdown("gate", lambda: shutdown_gate(gate))
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
    TABLE_COLUMNS = ("#", "I_mag (A)", "B (mT)", "Vg (V)", "V (V)", "R (Ω)", "T1 (K)", "T2 (K)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE

    def live_plot_args(self):
        return (_live_plot_worker,)

    def table_row(self, record: dict) -> tuple:
        I_mag = record.get("magnet_current_A")
        B = record.get("magnet_field_mT")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        return (
            str(record["point_index"] + 1),
            f"{I_mag:.4f}" if I_mag is not None else "—",
            f"{B:.2f}" if B is not None else "—",
            f"{record['gate_voltage_V']:.4g}",
            f"{record['voltage_V']:.4e}",
            f"{record['resistance_ohm']:.5g}",
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class DCGateSweepApp(MeasurementApp):
    TITLE = "DC Gate Sweep"
    SUB_TITLE = "Keithley 6221 + 2182 + 2400 · gate voltage sweep"

    # Session data root — fallback until _load_settings()/the identity bar's
    # "Data root" field replaces it. Read in compose(), so it must exist here.
    data_root: Path = _DEFAULT_DATA_DIR

    SWITCH_DEPENDENTS = {
        "enable_field": tuple(FIELD_FIELD_IDS),
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
                        "Gate voltage sweep (Keithley 2400)",
                        field("gate_min_V", "Sweep gate voltage min (V)", DEFAULTS["gate_min_V"]),
                        field("gate_max_V", "Sweep gate voltage max (V)", DEFAULTS["gate_max_V"]),
                        field("step_V", "Sweep step size (V)", DEFAULTS["step_V"],
                              validators=[Number(minimum=1e-9, failure_description="must be > 0")]),
                        switch_field("bidirectional_sweep", "Bidirectional (min → max → min)",
                                     DEFAULTS["bidirectional_sweep"]),
                    )
                    yield card(
                        "Sense current (Keithley 6221)",
                        field("sense_current_values", "Sense current (A)",
                              DEFAULTS["sense_current_values"], kind="text",
                              hint="Comma-separate for one sweep + file per value."),
                    )
                    yield card(
                        "Field (Kepco magnet, optional)",
                        switch_field("enable_field", "Park field (Kepco magnet)",
                                     DEFAULTS["enable_field"]),
                        field("field_current_values", "Magnet current (A)",
                              DEFAULTS["field_current_values"], kind="text",
                              hint="Comma-separate for one sweep + file per value."),
                    )
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature",
                                     "Log temperature (MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                    )

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
                            field("settling_time_s", "Settling time per gate step (s)",
                                  DEFAULTS["settling_time_s"],
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("n_averages", "Voltage samples averaged per point",
                                  DEFAULTS["n_averages"], kind="integer",
                                  validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        )

                # ── Tier 3: instrument wiring & timing constants — collapsed ─
                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Instrument addresses",
                            field("source_visa_resource", "6221 (sense current)",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            field("voltmeter_visa_resource", "2182 (DUT voltage)",
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
                            field("source_delay_s", "6221 source delay (s)", DEFAULTS["source_delay_s"]),
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
                            field("field_settle_s", "Settling time after parking field (s)",
                                  DEFAULTS["field_settle_s"]),
                            field("field_settle_tolerance_mT", "Field-settle tolerance (mT)",
                                  DEFAULTS["field_settle_tolerance_mT"],
                                  hint="Field settled when readings span less than this. Raise if parking stalls.",
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            muted=True,
                        )
                        yield card(
                            "Gaussmeter & temperature sensors",
                            field("gaussmeter_n_averages", "Field readings averaged",
                                  DEFAULTS["gaussmeter_n_averages"], kind="integer",
                                  validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                            field("gaussmeter_read_delay_s", "Delay between readings (s)",
                                  DEFAULTS["gaussmeter_read_delay_s"]),
                            field("temperature_sensor_uids", "MercuryiTC sensor board UID(s)",
                                  DEFAULTS["temperature_sensor_uids"], kind="text",
                                  hint="1-2 UIDs, comma-separated."),
                            muted=True,
                        )

            with Vertical(id="sidebar"):
                yield Static("Description", classes="sidebar-title")
                yield Static(DC_GATE_SWEEP_DESCRIPTION, classes="card-desc")
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
    DCGateSweepApp().run()


if __name__ == "__main__":
    main()
