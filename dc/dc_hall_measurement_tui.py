#!/usr/bin/env python3
"""
Textual TUI front-end for dc_hall_measurement.py
==================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-31

Lets you edit the parameters that decide whether a DC Hall measurement is
good or bad — sense current, compliance, reversal averaging, and the
magnet sweep — without touching the dataclasses in the script itself.

The sidebar recomputes derived values (estimated per-point acquisition
time, estimated sweep duration) and flags anything that risks a bad
measurement (source/voltmeter sharing a GPIB address, a sweep exceeding
the magnet's software current limit, zero sense current) as you type.

Run with:
    python dc_hall_measurement_tui.py

Requirements:
    pip install textual matplotlib  (in addition to
    dc_hall_measurement.py's own deps)
"""

from __future__ import annotations

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
    Static,
)

from dc.dc_hall_measurement import (
    AcquisitionConfig,
    FieldPoint,
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
    run_measurement,
    set_magnet_current,
    shutdown_gaussmeter,
    shutdown_magnet,
    shutdown_source,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import build_segmented_sweep, field_hops, parse_sweep_rows, safe_shutdown, try_parse
from instruments.data_dir import validate_directory
from instruments.field_geometry import field_direction_summary_line, render_ascii_field_diagram
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
    GPIB_TXN_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S,
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

DC_HALL_DESCRIPTION = (
    "Sources a fixed DC sense current with a Keithley 6221 and reads R_xy "
    "(transverse/Hall) and/or R_xx (longitudinal) voltage with a Keithley "
    "2182's two channels, reversing the current each rep to cancel "
    "thermal-EMF offsets. Optionally sweeps a Kepco electromagnet's field "
    "(bidirectionally, for hysteresis) with the field measured live via a "
    "Lake Shore 475 Gaussmeter at every point."
)

# Wiring schematic — shown on this program's card in bridge_tui.py.
DC_HALL_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source)
    Output ──▶ sample ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ transverse (Hall) voltage leads

  Magnet field sweep  (optional, "Sweep magnetic field" switch)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample
"""

log = logging.getLogger("dc_hall_measurement_tui")

# Data/settings live outside "bridge" (a sibling of it), same convention as
# dc_hall_measurement.py, so nothing generated at runtime ends up in the
# git-tracked source tree. _DEFAULT_DATA_DIR is the fallback data-convention
# "data root" (parent of every {sample}/ folder); the real root is chosen
# per run in the identity bar's "Data root" field (mirrors the web app —
# see web/directory_picker.py) and persisted in the settings file.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "dc_hall_measurement_tui_settings.json"

# Locked type code for this measurement (see instruments/data_naming.py) —
# never deviates.
MEASUREMENT_TYPE = "HALL"


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors dc_hall_measurement.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "source_visa_resource": "GPIB0::20::INSTR",
    "voltmeter_visa_resource": "GPIB0::7::INSTR",
    "sense_current_values": "0.001",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "nplc": "5",
    "auto_range": True,
    "measure_rxy": True,
    "measure_rxx": False,
    "channel_settle_s": "0.02",
    "settling_time_s": "1.0",
    "field_settle_tolerance_mT": "0.02",
    "n_reversals": "5",
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    "field_theta_deg": "",
    "field_phi_deg": "",
    "enable_sweep": True,
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

# id -> caster, for every free-text numeric field (Switch handled separately)
NUMERIC_FIELDS: dict = {
    "compliance_V": float,
    "source_delay_s": float,
    "nplc": float,
    "channel_settle_s": float,
    "settling_time_s": float,
    "field_settle_tolerance_mT": float,
    "n_reversals": int,
    "current_limit_A": float,
    "voltage_compliance_V": float,
    "ramp_step_A": float,
    "ramp_delay_s": float,
    "gaussmeter_n_averages": int,
    "gaussmeter_read_delay_s": float,
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "device",
               "cooldown", "magnet_visa_resource", "gaussmeter_visa_resource",
               "temperature_visa_resource", "temperature_sensor_uids",
               "sense_current_values", "data_dir"]
# Parsed separately from NUMERIC_FIELDS -- unlike every other numeric field,
# this one may be BLANK (valid_empty=True), which means "no temperature
# setpoint" -> the T### K filename token is simply omitted.
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K", "field_theta_deg", "field_phi_deg"]
MAGNET_FIELD_IDS = [
    "magnet_visa_resource", "current_limit_A", "voltage_compliance_V",
    "ramp_step_A", "ramp_delay_s",
    "gaussmeter_visa_resource", "gaussmeter_n_averages", "gaussmeter_read_delay_s",
]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]


def resolve_channel_map(measure_rxx: bool, measure_rxy: bool) -> dict:
    """Which 2182 channel each enabled quantity reads. Both on: R_xy keeps
    channel 1 (today's sole-channel meaning), R_xx gets channel 2. Only
    one on: that one uses channel 1, identical to today's wiring/behavior
    when the other is off. Neither on: empty dict (build_summary errors)."""
    if measure_rxx and measure_rxy:
        return {"rxy": 1, "rxx": 2}
    if measure_rxx:
        return {"rxx": 1}
    if measure_rxy:
        return {"rxy": 1}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def run_costs(currents_A, state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (series-major: one full field sweep, or single point when `currents_A`
    is None, per sense current). Also drives the run screen's progress bar,
    so estimate and live ETA cannot disagree."""
    n_series = max(1, len(state.get("sense_current_list") or []))
    n_pts = len(currents_A) if currents_A is not None else 1
    n_channels = max(1, len(resolve_channel_map(state["measure_rxx"], state["measure_rxy"])))
    has_temp = state["enable_temperature"] and bool(parse_sensor_uids(state["temperature_sensor_uids"]))

    rc = RunCost(n_pts * n_series)
    rc.each("settle", state["settling_time_s"])
    rc.each("2182 reads", reversal_avg_s(state["n_reversals"], state["source_delay_s"],
                                         read_time_s(state["nplc"]), n_channels,
                                         state["channel_settle_s"]))
    rc.each("overhead", POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    if currents_A is not None:      # field sweep: magnet + gaussmeter per point, ramp-down at the end
        mcfg = MagnetConfig(ramp_step_A=state["ramp_step_A"], ramp_delay_s=state["ramp_delay_s"])
        gcfg = GaussmeterConfig(n_averages=state["gaussmeter_n_averages"],
                                read_delay_s=state["gaussmeter_read_delay_s"])
        rc.each("field read", read_field_s(gcfg))
        for i, hop in enumerate(field_hops(currents_A, n_series)):
            typ, worst = magnet_move_s(hop, mcfg)
            rc.at("magnet", typ, i, worst_extra=worst - typ)
        if n_pts:
            rc.tail("ramps", magnet_move_s(currents_A[-1], mcfg, with_field=False)[0])
    for k in range(n_series):
        rc.at("per-file", PER_FILE_S, k * n_pts)
    rc.at("per-run", PER_RUN_S, 0)
    rc.tail("ramps", 2 * GPIB_TXN_S)    # 6221 off
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    acq_cfg: AcquisitionConfig
    channel_map: dict
    channel_settle_s: float
    magnet_cfg: Optional[MagnetConfig]
    gauss_cfg: Optional[GaussmeterConfig]
    currents_A: Optional[np.ndarray]
    sense_currents_A: List[float]
    temp_cfg: Optional[TemperatureControllerConfig]
    sample: str
    device: str
    temperature_setpoint_K: Optional[float]
    field_theta_deg: Optional[float]
    field_phi_deg: Optional[float]
    cooldown: str
    header_extra: dict
    series: str = ""
    data_root: Path = _DEFAULT_DATA_DIR
    run_cost: Optional[RunCost] = None      # modelled seconds per point (progress bar + ETA)

    @property
    def series_values(self) -> List[float]:
        """One entry per sense current -- one complete measurement (single
        point, or a full field sweep) per value, each saved to its own file."""
        return list(self.sense_currents_A)

    @property
    def total_points(self) -> int:
        base = len(self.currents_A) if self.currents_A is not None else 1
        return base * len(self.sense_currents_A)


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                         status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """
    Universal + measurement-specific header/index fields for ONE run within
    this (possibly multi-file, one-per-sense-current) session -- `ctx` is
    that particular iteration's RunContext, not a single plan-wide one (see
    instruments/data_naming.py's allocate_run() -- called fresh per
    iteration for this suite). `extra` carries this iteration's own value
    (sense_current_A) on top of the plan-wide header_extra.

    T_setpoint_K is the nominal value used to build the filename's T###K
    token. T_K is the MEASURED mean (temperature_1_K) -- left blank (not
    backfilled with the setpoint) whenever the MercuryiTC is disconnected
    or hasn't produced a reading yet, so a query like `T_K < 50` never
    silently trusts an unmeasured number.
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
    state["sense_current_list"], state["sense_current_parse_error"] = try_parse(state["sense_current_values"])
    state["sweep_rows_parsed"], state["sweep_rows_parse_error"] = try_parse(state["sweep_rows"], parse_sweep_rows)
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

    if state["source_visa_resource"] == state["voltmeter_visa_resource"]:
        errors.append("Source (6221) and voltmeter (2182) VISA resources must be different.")

    channel_map = resolve_channel_map(state["measure_rxx"], state["measure_rxy"])
    if not channel_map:
        errors.append("Enable at least one of R_xy or R_xx.")
    elif len(channel_map) == 2:
        info.append("R_xy on ch1, R_xx on ch2.")
        warnings.append("R_xy + R_xx: the 2182's ch1 LO and ch2 LO are one node "
                        "inside the instrument. Wire both LO leads to the SAME "
                        "sample contact. On two different contacts the 2182 "
                        "shorts them together, shunting part of the sample and "
                        "silently corrupting R_xx and R_xy.")
        warnings.append("Before the first run: unplug the leads from the 2182 "
                        "and meter between the two LO leads. Lead resistance "
                        "only = same contact, OK. Sample resistance = different "
                        "contacts, do not run both.")
        info.append("Both channels read per point — roughly doubles the "
                     "per-point acquisition time.")
    else:
        label = next(iter(channel_map))
        info.append(f"{label.upper()} only, on channel 1 (identical wiring/timing "
                     "to a single-channel run).")

    if state.get("sense_current_parse_error"):
        errors.append(f"Sense current list: {state['sense_current_parse_error']}")
        current_list: list[float] = []
    else:
        current_list = state.get("sense_current_list", [])
        if any(i == 0 for i in current_list):
            errors.append("Sense current must be nonzero (Hall resistance divides by it).")
    n_series = len(current_list)
    if n_series > 1:
        currents_str = ", ".join(format_si(i, "A") for i in current_list)
        info.append(f"Sense currents: {currents_str} — {n_series} complete measurements, "
                     f"one file each")
    elif n_series == 1:
        info.append(f"Sense current I = {format_si(current_list[0], 'A')}")

    if state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")

    read_s = read_time_s(state["nplc"])
    info.append(f"Estimated 2182 reading time ≈ {read_s * 1000:.0f} ms (NPLC={state['nplc']:g})")

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
            resolved = build_segmented_sweep(rows, state["bidirectional_sweep"])
            total_points = len(resolved)
            n_raw = sum(n for _, _, n in rows)
            n_merged = (2 * n_raw if state["bidirectional_sweep"] else n_raw) - total_points
            merged_note = f", {n_merged} shared boundary point(s) merged" if n_merged else ""
            info.append(f"Sweep: {len(rows)} row(s), {total_points} points"
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
        info.extend(run_costs(resolved if resolved is not None else [], state)
                    .lines("Estimated total run time"))
    else:
        info.append("Single point — no field sweep, magnet untouched.")
        info.extend(run_costs(None, state).lines("Estimated total run time"))

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

    info.append(field_direction_summary_line(
        state.get("field_theta_deg"), state.get("field_phi_deg")))

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
    suffix = (" (one file per sense current)"
              if len(state.get("sense_current_list", [])) > 1 else "")
    return f"{preview}_<timestamp>.csv{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────
# A GUI matplotlib backend and Textual's terminal control both want the main
# thread. Rather than fight that, the live preview gets its own process with
# its own main thread; new points are streamed to it over a
# multiprocessing.Queue. The final PNG is saved independently by the TUI
# process itself (see _save_measurement_png), so it doesn't depend on this
# window still being open when the run finishes.

QUANTITY_PLOT_LABELS = {"rxy": "R_xy", "rxx": "R_xx"}
QUANTITY_LINESTYLES = {"rxy": "-", "rxx": "--"}


def active_quantities(record: dict) -> list:
    """Which of rxy/rxx have a real (non-NaN) resistance in this record —
    see dc_hall_measurement.run_measurement's always-both-columns convention."""
    out = []
    for q in ("rxy", "rxx"):
        v = record.get(f"{q}_resistance_ohm")
        if v is not None and not np.isnan(v):
            out.append(q)
    return out


def _live_plot_worker(queue: "mp.Queue", has_field_sweep: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("DC Hall live measurement")
    except Exception:
        pass
    ax.set_ylabel("Resistance (Ω)")
    ax.set_xlabel("Magnetic field (mT)" if has_field_sweep else "Point #")
    ax.set_title("Live measurement")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    cmap = plt.get_cmap("tab10")
    lines: dict[tuple[int, str], "plt.Line2D"] = {}
    series_data: dict[tuple[int, str], tuple[list, list]] = {}

    def _drain(_frame=None):
        updated: set[tuple[int, str]] = set()
        new_series = False
        while True:
            try:
                record = queue.get_nowait()
            except Exception:
                break
            idx = record.get("series_index", 0)
            series_label = record.get("series_label")
            for q in active_quantities(record):
                key = (idx, q)
                if key not in lines:
                    label = f"{QUANTITY_PLOT_LABELS[q]} {series_label}" if series_label \
                        else QUANTITY_PLOT_LABELS[q]
                    (line,) = ax.plot([], [], "o", linestyle=QUANTITY_LINESTYLES[q],
                                       color=cmap(idx % 10), label=label)
                    lines[key] = line
                    series_data[key] = ([], [])
                    new_series = True
                xs, ys = series_data[key]
                x = record.get("magnet_field_mT") if has_field_sweep else None
                xs.append(x if x is not None else record["point_index"])
                ys.append(record[f"{q}_resistance_ohm"])
                updated.add(key)
        if updated:
            for key in updated:
                xs, ys = series_data[key]
                lines[key].set_data(xs, ys)
            if new_series:
                ax.legend(loc="best", fontsize=8)
            ax.relim()
            ax.autoscale_view()
        return tuple(lines.values())

    # Keep a reference so it isn't garbage-collected mid-run.
    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path,
                           plan: Optional["MeasurementPlan"] = None, comment: str = "") -> None:
    """Save a static resistance-vs-field PNG to proc/, from whatever
    points were actually collected (including an aborted/partial run) —
    solid for R_xy / dashed for R_xx when both were measured.

    `records` is ONE run's points -- with several sense currents each run
    is saved (and plotted) on its own, exactly like a manual run.

    `plan`/`comment` drive a small "at a glance" text annotation (field
    direction, a single fixed sense current, the operator's comment) for
    context that isn't already in the filename -- see _annotation_lines().
    Called once right when the run ends (comment="" -- not collected yet)
    and, if the operator later supplies a comment, again to overwrite the
    PNG in place with `comment` filled in (see _on_status_comment)."""
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")  # headless — must not touch the TUI's terminal
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]
    for q in ("rxy", "rxx"):
        ys = [r.get(f"{q}_resistance_ohm") for r in records]
        if all(y is None or np.isnan(y) for y in ys):
            continue
        ax.plot(xs, ys, marker=".", linestyle=QUANTITY_LINESTYLES[q],
                 color="tab:blue", label=QUANTITY_PLOT_LABELS[q])

    ax.set_ylabel("Resistance (Ω)")
    ax.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax.set_title("Measurement result")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    lines: list[str] = []
    if plan is not None:
        if plan.field_theta_deg is not None:
            lines.append(field_direction_summary_line(plan.field_theta_deg, plan.field_phi_deg))
        sense_currents = sorted({r["sense_current_A"] for r in records
                                  if r.get("sense_current_A") is not None})
        if len(sense_currents) == 1:
            lines.append(f"Sense current: {format_si(sense_currents[0], 'A')}")
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
# Plan + run  ── pure, shared by the TUI RunScreen and web/dc/hall.py
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
        channel=1,
    )
    channel_map = resolve_channel_map(state["measure_rxx"], state["measure_rxy"])
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"],
        field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
        n_reversals=state["n_reversals"],
        output_file="",  # overwritten per series iteration
    )

    magnet_cfg = None
    gauss_cfg = None
    currents_A = None
    if state["enable_sweep"]:
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
        currents_A = build_segmented_sweep(
            state["sweep_rows_parsed"], state["bidirectional_sweep"],
        )

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
        "n_reversals": state["n_reversals"],
        "settling_time_s": state["settling_time_s"],
        "measure_rxy": "rxy" in channel_map,
        "measure_rxx": "rxx" in channel_map,
    }
    if state["enable_sweep"]:
        header_extra["field_sweep_rows_A"] = state["sweep_rows_parsed"]

    # A "series" tag only means something for an actual family of runs
    # (>1 sense current) -- a single-current run gets no series tag.
    series = ""
    if len(state["sense_current_list"]) > 1:
        series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                  f"{datetime.now():%Y%m%dT%H%M%S}")

    return MeasurementPlan(
        src_cfg=src_cfg, volt_cfg=volt_cfg, acq_cfg=acq_cfg,
        channel_map=channel_map, channel_settle_s=state["channel_settle_s"],
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, currents_A=currents_A,
        sense_currents_A=state["sense_current_list"],
        temp_cfg=temp_cfg, data_root=data_root,
        sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state["temperature_setpoint_K"],
        field_theta_deg=state["field_theta_deg"],
        field_phi_deg=state["field_phi_deg"],
        cooldown=state["cooldown"], header_extra=header_extra, series=series,
        run_cost=run_costs(currents_A, state),
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
    """Connect, then one run (own run number, own file) per sense current,
    each recorded + finalized by record_run() before the next; always shut
    the instruments down. Pure — the TUI's RunScreen and the web page each
    pass their own callbacks (see sot_nonlocal_switching_tui.run_plan)."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = voltmeter = magnet = gaussmeter = temp_ctrl = None
    try:
        on_status("Connecting to Keithley 6221 & 2182 …")
        source = connect_source(plan.src_cfg)
        extra_channels = (2,) if len(plan.channel_map) == 2 else ()
        voltmeter = connect_voltmeter(plan.volt_cfg, extra_channels=extra_channels)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC (temperature) …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        if plan.magnet_cfg is not None and plan.currents_A is not None:
            on_status("Connecting magnet power supply …")
            magnet = connect_magnet(plan.magnet_cfg)
            on_status("Connecting gaussmeter …")
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)

        n_series = len(plan.series_values)
        for series_idx, I_sense in enumerate(plan.series_values):
            if stop_event.is_set():
                break
            label = f"I={I_sense:g}A" if n_series > 1 else None
            plan.src_cfg.sense_current_A = I_sense

            # A fresh RunContext (own run number, own file) EVERY iteration --
            # never reuse one across the sense-current series.
            ctx = allocate_run(
                plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                temperature_setpoint_K=plan.temperature_setpoint_K,
                key_axis=("current_A", I_sense) if n_series > 1 else None, series=plan.series,
            )
            extra = {"sense_current_A": I_sense}
            run_contexts.append(ctx)
            run_extras.append(extra)
            on_run_label(f"Run #{ctx.run_str}")
            plan.acq_cfg.output_file = str(ctx.raw_path)

            if magnet is not None and plan.currents_A is not None:
                points = [
                    FieldPoint(
                        magnet_current_A=I,
                        set_action=lambda I=I: set_magnet_current(
                            magnet, plan.magnet_cfg, I, gaussmeter, plan.gauss_cfg,
                            plan.acq_cfg.field_settle_tolerance_mT, stop_event),
                    )
                    for I in plan.currents_A
                ]
            else:
                points = [FieldPoint()]

            on_status("Running measurement …" if n_series == 1
                      else f"Running measurement (I={I_sense:g} A) …")
            record_run(
                plan.data_root, ctx,
                lambda records, status, _ctx=ctx, _x=extra: build_header_fields(
                    plan, _ctx, records, status=status, comment="", extra=_x),
                lambda point_cb, write_csv, _points=points: run_measurement(
                    source, voltmeter, plan.src_cfg, plan.acq_cfg, _points,
                    stop_event=stop_event, on_point=point_cb,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                    field_theta_deg=plan.field_theta_deg, field_phi_deg=plan.field_phi_deg,
                    write_csv=write_csv, channel_map=plan.channel_map,
                    channel_settle_s=plan.channel_settle_s),
                stop_event, on_point=on_point,
                tags={"series_index": series_idx, "series_label": label},
                on_finished=on_run_finished)
    finally:
        # 6221 output off first (immediate, no current into the DUT), so the
        # magnet can start its ramp-down right away rather than waiting behind it.
        if source is not None:
            safe_shutdown("source", lambda: shutdown_source(source))
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
    TABLE_COLUMNS = ("#", "I_sense (A)", "I_magnet (A)", "B (mT)", "R_xy (Ω)", "R_xx (Ω)", "n_rev", "T1 (K)", "T2 (K)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE

    def live_plot_args(self):
        return (_live_plot_worker, self.plan.magnet_cfg is not None)

    def table_row(self, record: dict) -> tuple:
        Isense = record.get("sense_current_A")
        I = record.get("magnet_current_A")
        B = record.get("magnet_field_mT")
        Rxy = record.get("rxy_resistance_ohm")
        Rxx = record.get("rxx_resistance_ohm")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        return (
            str(record["point_index"] + 1),
            f"{Isense:.4g}" if Isense is not None else "—",
            f"{I:.4f}" if I is not None else "—",
            f"{B:.2f}" if B is not None else "—",
            f"{Rxy:.5g}" if Rxy is not None and not np.isnan(Rxy) else "—",
            f"{Rxx:.5g}" if Rxx is not None and not np.isnan(Rxx) else "—",
            str(record["n_reversals"]),
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class DCHallMeasurementApp(MeasurementApp):
    TITLE = "DC Hall Measurement"
    SUB_TITLE = "Keithley 6221 + 2182 · magnet field sweep"

    # Data root for this session — the fallback until _load_settings() or the
    # identity bar's "Data root" field replaces it. Read in compose() (before
    # on_mount), so it must exist as a plain attribute here.
    data_root: Path = _DEFAULT_DATA_DIR

    SWITCH_DEPENDENTS = {
        "enable_sweep": (*MAGNET_FIELD_IDS, "sweep_rows", "bidirectional_sweep"),
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
                # ── File & run identity ── changes every run, always on top ──
                yield identity_bar(DEFAULTS, _DEFAULT_DATA_DIR, self.data_root)

                # ── Tier 1: what defines this run — always visible ──────────
                with Vertical(classes="param-grid"):
                    yield card(
                        "Source current (Keithley 6221)",
                        field("sense_current_values", "Sense current (A)",
                              DEFAULTS["sense_current_values"], kind="text",
                              hint="Reversed +I/-I each rep to cancel thermal-EMF offsets. "
                                   "Single value, or comma-separated list — one complete "
                                   "measurement runs per value, each saved to its own file."),
                    )
                    yield card(
                        "Quantities (Keithley 2182)",
                        switch_field("measure_rxy", "R_xy (transverse/Hall) — ch1",
                                     DEFAULTS["measure_rxy"]),
                        switch_field("measure_rxx", "R_xx (longitudinal) — ch2 if both on, "
                                     "else ch1", DEFAULTS["measure_rxx"]),
                    )
                    yield card(
                        "Field sweep (Kepco magnet)",
                        switch_field("enable_sweep", "Sweep magnetic field", DEFAULTS["enable_sweep"]),
                        sweep_rows_field("sweep_rows", DEFAULTS["sweep_rows"]),
                        switch_field("bidirectional_sweep", "Bidirectional (retrace the merged rows)",
                                     DEFAULTS["bidirectional_sweep"]),
                    )
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature",
                                     "Log temperature (MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                    )
                    yield card(
                        "Field direction",
                        field("field_theta_deg", "θ — tilt from out-of-plane (°)",
                              DEFAULTS["field_theta_deg"], kind="number", valid_empty=True,
                              validators=[Number(0, 180, failure_description="0-180°")],
                              hint="0° = fully out-of-plane (film normal), 90° = in-plane."),
                        field("field_phi_deg", "φ — azimuth from current axis (°)",
                              DEFAULTS["field_phi_deg"], kind="number", valid_empty=True,
                              validators=[Number(0, 360, failure_description="0-360°")],
                              hint="0° = along sense current, 90° = transverse in-plane. "
                                   "Meaningless when θ=0°."),
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
                            field("n_reversals", "+I/-I reversal pairs averaged",
                                  DEFAULTS["n_reversals"], kind="integer",
                                  hint="(V(+I)-V(-I))/2 is the reported R.",
                                  validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                            field("channel_settle_s", "2182 channel-mux settle (s)",
                                  DEFAULTS["channel_settle_s"],
                                  hint="Only used when both R_xy and R_xx are on — dead time "
                                       "after switching the 2182's active channel, before reading.",
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                        )

                # ── Tier 3: instrument wiring & timing constants — collapsed ─
                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Instrument addresses",
                            field("source_visa_resource", "6221 (current source)",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            field("voltmeter_visa_resource", "2182 (R_xy / R_xx voltage)",
                                  DEFAULTS["voltmeter_visa_resource"], kind="text"),
                            field("magnet_visa_resource", "Magnet (Kepco)",
                                  DEFAULTS["magnet_visa_resource"], kind="text"),
                            field("gaussmeter_visa_resource", "Gaussmeter (Lake Shore 475)",
                                  DEFAULTS["gaussmeter_visa_resource"], kind="text"),
                            field("temperature_visa_resource", "MercuryiTC",
                                  DEFAULTS["temperature_visa_resource"], kind="text"),
                            muted=True,
                        )
                        yield card(
                            "Source & ramp safety",
                            field("source_delay_s", "6221 source delay (s)", DEFAULTS["source_delay_s"],
                                  hint="Also the settle time between a current reversal and "
                                       "reading the voltmeter, so the reversal has actually "
                                       "finished before the 2182 integrates."),
                            field("current_limit_A", "Magnet software current limit (A)",
                                  DEFAULTS["current_limit_A"],
                                  hint="Hard safety ceiling."),
                            field("voltage_compliance_V", "Magnet voltage compliance (V)",
                                  DEFAULTS["voltage_compliance_V"]),
                            field("ramp_step_A", "Magnet ramp step (A)", DEFAULTS["ramp_step_A"]),
                            field("ramp_delay_s", "Magnet ramp delay (s)", DEFAULTS["ramp_delay_s"]),
                            muted=True,
                        )
                        yield card(
                            "Averaging & sensors",
                            field("gaussmeter_n_averages", "Field readings averaged per point",
                                  DEFAULTS["gaussmeter_n_averages"], kind="integer",
                                  validators=[Number(minimum=1, failure_description="must be ≥ 1")]),
                            field("gaussmeter_read_delay_s", "Delay between field readings (s)",
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
                yield Static(DC_HALL_DESCRIPTION, classes="card-desc")
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

    def _build_plan(self, state: dict) -> MeasurementPlan:
        return build_plan(state, self.data_root)


def main() -> None:
    DCHallMeasurementApp().run()


if __name__ == "__main__":
    main()
