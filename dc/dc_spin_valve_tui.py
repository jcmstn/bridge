#!/usr/bin/env python3
"""
Textual TUI front-end for dc_spin_valve.py
=============================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-31

Lets you edit the parameters that decide whether a spin-valve/field-sweep
measurement is good or bad — sense current, compliance, reversal
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
    SourceConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    connect_gate,
    connect_gaussmeter,
    connect_magnet,
    connect_source,
    connect_temperature_controller,
    connect_voltmeter,
    run_measurement,
    set_gate_voltage,
    set_magnet_current,
    shutdown_gate,
    shutdown_gaussmeter,
    shutdown_magnet,
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
from instruments.tui_common import (
    MeasurementApp,
    MeasurementRunScreen,
    card,
    field,
    format_si,
    identity_bar,
    parse_sensor_uids,
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
    "Sources a fixed DC sense current with a Keithley 6221 and reads the "
    "longitudinal voltage with a Keithley 2182, reversing the current each "
    "rep to cancel thermal-EMF offsets by default — the same "
    "reversal-averaging technique as the Hall measurement, but for a "
    "longitudinal (spin-valve / magnetoresistance) read. Reversal can be "
    "switched off for bias-direction-dependent devices, where flipping the "
    "current destroys rather than cleans up the signal — the sense current "
    "is then just held fixed and plainly averaged instead. Sweeps a Kepco "
    "electromagnet's field (bidirectionally, for hysteresis) with the "
    "field measured live via a Lake Shore 475 Gaussmeter at every point. "
    "The gate voltage (Keithley 2400, optional — off by default needs no "
    "2400 connected) is held fixed for each field sweep — single value or "
    "a comma-separated list — one complete sweep per value, each saved to "
    "its own file and plotted together in different colors."
)

# Wiring schematic — shown on this program's card in bridge_tui.py.
DC_SPIN_VALVE_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source)
    Output ──▶ sample ── common ground
    Current reversal (+I/-I) is a toggle — some devices are
    bias-direction dependent and reversal destroys the signal

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
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
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
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "gate_visa_resource",
               "device", "cooldown", "magnet_visa_resource",
               "gaussmeter_visa_resource", "gate_voltage_values",
               "temperature_visa_resource", "temperature_sensor_uids",
               "sense_current_values", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K"]
GATE_FIELD_IDS = ["gate_visa_resource", "gate_voltage_limit_V",
                   "gate_compliance_current_A", "gate_voltage_values"]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def run_costs(currents_A, state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (series-major: one full field sweep per sense current x gate voltage).
    Also drives the run screen's progress bar, so estimate and live ETA
    cannot disagree."""
    n_gate = max(1, len(state.get("gate_voltage_list") or [])) if state["enable_gate"] else 1
    n_series = max(1, len(state.get("sense_current_list") or [])) * n_gate
    n_pts = len(currents_A)
    mcfg = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
    gcfg = GaussmeterConfig(n_averages=state["gaussmeter_n_averages"],
                            read_delay_s=state["gaussmeter_read_delay_s"])
    has_temp = state["enable_temperature"] and bool(parse_sensor_uids(state["temperature_sensor_uids"]))
    read_s = read_time_s(state["nplc"])
    reads_s = (reversal_avg_s(state["n_averages"], state["source_delay_s"], read_s)
               if state["reversal_enabled"] else state["n_averages"] * read_s)

    rc = RunCost(n_pts * n_series)
    rc.each("settle", state["settling_time_s"])
    rc.each("field read", read_field_s(gcfg))
    rc.each("2182 reads", reads_s)
    rc.each("overhead", POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    for i, hop in enumerate(field_hops(currents_A, n_series)):
        typ, worst = magnet_move_s(hop, mcfg)
        rc.at("magnet", typ, i, worst_extra=worst - typ)
    for k in range(n_series):
        rc.at("per-file", PER_FILE_S, k * n_pts)
    rc.at("per-run", PER_RUN_S, 0)
    if n_pts:   # teardown: 6221 off, gate ramp-down, magnet ramp-down from the last field point
        rc.tail("ramps", 2 * GPIB_TXN_S + (GATE_RAMP_S if state["enable_gate"] else 0.0)
                + magnet_move_s(currents_A[-1], mcfg, with_field=False)[0])
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
    Pure: shared by the TUI's and the web page's parse_state()."""
    state["gate_voltage_list"], state["gate_parse_error"] = \
        try_parse(state["gate_voltage_values"]) if state["enable_gate"] else ([], None)
    state["sense_current_list"], state["sense_current_parse_error"] = try_parse(state["sense_current_values"])
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

    resources = [state["source_visa_resource"], state["voltmeter_visa_resource"]]
    if state["enable_gate"]:
        resources.append(state["gate_visa_resource"])
    if len(set(resources)) < len(resources):
        errors.append(
            "Source (6221), voltmeter (2182)"
            + (", and gate (2400)" if state["enable_gate"] else "")
            + " VISA resources must all be different."
        )

    n_current_series = 1
    if state.get("sense_current_parse_error"):
        errors.append(f"Sense current list: {state['sense_current_parse_error']}")
        current_list: list[float] = []
    else:
        current_list = state.get("sense_current_list", [])
        if any(i == 0 for i in current_list):
            errors.append("Sense current must be nonzero (resistance divides by it).")
    n_current_series = len(current_list)
    if n_current_series > 1:
        currents_str = ", ".join(format_si(i, "A") for i in current_list)
        info.append(f"Sense currents: {currents_str} — {n_current_series} complete field "
                     f"sweeps, one file each, plotted together")
    elif n_current_series == 1:
        info.append(f"Sense current I = {format_si(current_list[0], 'A')}")

    if state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")

    if state["reversal_enabled"]:
        info.append("Sense current reversed +I/-I each rep to cancel thermal-EMF offsets.")
    else:
        info.append("Reversal off — sense current held fixed at +I "
                     "(use for bias-direction-dependent devices).")

    read_s = read_time_s(state["nplc"])
    info.append(f"Estimated 2182 reading time ≈ {read_s * 1000:.0f} ms (NPLC={state['nplc']:g})")

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
            info.append(f"Gate: {n_gate_series} values {gate_list} V — {n_gate_series} complete field "
                        f"sweeps (per sense current), one file each, plotted together")
        elif n_gate_series == 1:
            info.append(f"Gate held fixed at {format_si(gate_list[0], 'V')}")
    else:
        info.append("Gate off — Keithley 2400 not used, single field sweep run.")

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
        info.append(f"Field sweep: {len(rows)} row(s), {n_sweep_points} points"
                     f"{' (bidirectional)' if state['bidirectional_sweep'] else ''}"
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

    info.extend(run_costs(resolved, state).lines("Estimated total run time"))

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
    if len(state.get("sense_current_list", [])) > 1:
        axes.append("sense current")
    if state.get("enable_gate") and len(state.get("gate_voltage_list", [])) > 1:
        axes.append("gate voltage")
    suffix = f" (one file per {' × '.join(axes)})" if axes else ""
    return f"{preview}_<timestamp>.csv{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("DC Spin-Valve live measurement")
    except Exception:
        pass
    ax.set_xlabel("Magnetic field (mT)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Live measurement — field sweep")
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
            x = record.get("magnet_field_mT")
            xs.append(x if x is not None else record["point_index"])
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

    fig, ax = plt.subplots(figsize=(7, 5))

    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]
    ax.plot(xs, [r["voltage_V"] for r in records], ".-", color="tab:blue")

    ax.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Measurement result")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lines: list[str] = []
    if plan is not None:
        sense_currents = sorted({r["sense_current_A"] for r in records
                                  if r.get("sense_current_A") is not None})
        if len(sense_currents) == 1:
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
# Plan + run  ── pure, shared by the TUI RunScreen and web/dc/spin_valve.py
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

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"],
                sensor_uids=uids,
            )

    header_extra = {
        "compliance_V": state["compliance_V"],
        "reversal_enabled": state["reversal_enabled"],
        "n_averages": state["n_averages"],
        "settling_time_s": state["settling_time_s"],
        "field_sweep_rows_A": state["sweep_rows_parsed"],
    }
    # A "series" tag only means something for an actual family of runs
    # (>1 sense current and/or >1 gate voltage) -- a single-run session
    # gets no series tag.
    series = ""
    if len(state["sense_current_list"]) > 1 or len(gate_voltages_V or []) > 1:
        series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                  f"{datetime.now():%Y%m%dT%H%M%S}")

    return MeasurementPlan(
        src_cfg=src_cfg, volt_cfg=volt_cfg,
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, acq_cfg=acq_cfg,
        currents_A=currents_A, sense_currents_A=state["sense_current_list"],
        data_root=data_root,
        sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series=series,
        gate_cfg=gate_cfg, gate_voltages_V=gate_voltages_V,
        temp_cfg=temp_cfg, run_cost=run_costs(currents_A, state),
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
    RunScreen and the web page each pass their own callbacks."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = voltmeter = gate = magnet = gaussmeter = temp_ctrl = None
    try:
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
            label_parts = []
            if n_currents > 1:
                label_parts.append(f"I={I_sense:g}A")
            if gate_V is not None and n_gates > 1:
                label_parts.append(f"Vg={gate_V:g}V")
            label = ", ".join(label_parts) or None

            key_axis = None
            if n_gates > 1:
                key_axis = ("gate_V", gate_V)
            elif n_currents > 1:
                key_axis = ("current_A", I_sense)

            if gate_V is not None:
                on_status(f"Setting gate to {gate_V:g} V …")
                set_gate_voltage(gate, plan.gate_cfg, gate_V)

            extra = {"sense_current_A": I_sense}
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
                lambda point_cb, write_csv, _points=points, _gv=gate_V: run_measurement(
                    source, voltmeter, plan.src_cfg, plan.acq_cfg, _points,
                    stop_event=stop_event, on_point=point_cb,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg, gate_voltage_V=_gv,
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
    TABLE_COLUMNS = ("#", "I_sense (A)", "Vg (V)", "I_magnet (A)", "B (mT)", "V (V)", "R (Ω)", "n_avg", "T1 (K)", "T2 (K)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE

    def live_plot_args(self):
        return (_live_plot_worker,)

    def table_row(self, record: dict) -> tuple:
        Isense = record.get("sense_current_A")
        I = record.get("magnet_current_A")
        B = record.get("magnet_field_mT")
        Vg = record.get("gate_voltage_V")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        return (
            str(record["point_index"] + 1),
            f"{Isense:.4g}" if Isense is not None else "—",
            f"{Vg:.4g}" if Vg is not None else "—",
            f"{I:.4f}" if I is not None else "—",
            f"{B:.2f}" if B is not None else "—",
            f"{record['voltage_V']:.4e}",
            f"{record['resistance_ohm']:.5g}",
            str(record["n_averages"]),
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class DCSpinValveApp(MeasurementApp):
    TITLE = "DC Spin-Valve / Field Sweep"
    SUB_TITLE = "Keithley 6221 + 2182 + 2400 · magnet field sweep"

    # Session data root — fallback until _load_settings()/the identity bar's
    # "Data root" field replaces it. Read in compose(), so it must exist here.
    data_root: Path = _DEFAULT_DATA_DIR

    SWITCH_DEPENDENTS = {
        "enable_gate": tuple(GATE_FIELD_IDS),
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
    .field-label { text-style: bold; }
    .hint { text-style: italic; color: $text-muted; }
    .sweep-rows { height: 5; margin-bottom: 1; }
    .switch-row { height: 3; }
    .switch-row Label { margin-left: 1; content-align: left middle; height: 3; }
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
                        "Sense current (Keithley 6221)",
                        field("sense_current_values", "Sense current (A)",
                              DEFAULTS["sense_current_values"], kind="text",
                              hint="Reversed +I/-I each rep to cancel thermal-EMF offsets, "
                                   "unless reversal is switched off below. Single value, or "
                                   "comma-separated list — one complete field sweep runs per "
                                   "value, each saved to its own file."),
                        switch_field("reversal_enabled", "Reverse current each rep (+I/-I)",
                                     DEFAULTS["reversal_enabled"]),
                        Label(
                            "Turn off for bias-direction-dependent devices (diodes, asymmetric "
                            "spin-orbit stacks, ...) where reversing the current destroys rather "
                            "than cleans up the signal — the sense current is then just held "
                            "fixed at +I and plainly averaged instead.",
                            classes="hint",
                        ),
                    )
                    yield card(
                        "Gate voltage (Keithley 2400, optional)",
                        switch_field("enable_gate", "Enable gate (Keithley 2400)",
                                     DEFAULTS["enable_gate"]),
                        field("gate_voltage_values", "Gate voltage (V)",
                              DEFAULTS["gate_voltage_values"], kind="text",
                              hint="Single value, or comma-separated list — one complete "
                                   "field sweep runs per value, each saved to its own file "
                                   "and plotted together."),
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
                            field("settling_time_s", "Settling time per point (s)",
                                  DEFAULTS["settling_time_s"],
                                  hint="Dead-time after a field change, before acquiring.",
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("n_averages", "Voltage averages per point",
                                  DEFAULTS["n_averages"], kind="integer",
                                  hint="Reversal on: (V(+I)-V(-I))/2 is the reported R. "
                                       "Reversal off: plain samples at the fixed sense current.",
                                  validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                        )

                # ── Tier 3: instrument wiring & timing constants — collapsed ─
                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Instrument addresses",
                            field("source_visa_resource", "6221 (current source)",
                                  DEFAULTS["source_visa_resource"], kind="text"),
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
                            field("source_delay_s", "6221 source delay (s)", DEFAULTS["source_delay_s"],
                                  hint="Also the settle time between a current reversal and "
                                       "reading the voltmeter."),
                            field("gate_voltage_limit_V", "Gate voltage software limit (V)",
                                  DEFAULTS["gate_voltage_limit_V"],
                                  hint="Hard safety ceiling — independent of the values above."),
                            field("gate_compliance_current_A", "Gate leakage compliance (A)",
                                  DEFAULTS["gate_compliance_current_A"]),
                            muted=True,
                        )
                        yield card(
                            "Magnet ramp safety",
                            field("current_limit_A", "Software current limit (A)",
                                  DEFAULTS["current_limit_A"],
                                  hint="Hard safety ceiling — independent of the supply's own range."),
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
                                  hint="Advanced: after each magnet step, the field counts as "
                                       "settled once a short window of gaussmeter readings spans "
                                       "less than this. Raise it if points stall waiting; lower "
                                       "for tighter field control before acquiring.",
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

    # ── Start ────────────────────────────────────────────────────────────────

    def _build_plan(self, state: dict) -> MeasurementPlan:
        return build_plan(state, self.data_root)


def main() -> None:
    DCSpinValveApp().run()


if __name__ == "__main__":
    main()
