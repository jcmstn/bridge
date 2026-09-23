#!/usr/bin/env python3
"""
Textual TUI front-end for dc_iv_curve.py
==========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-07-31

Lets you edit the parameters that decide whether a DC I-V sweep is good or
bad — current range, compliance, voltmeter integration time, timing —
without touching the dataclasses in the script itself.

The sidebar recomputes derived values (estimated per-point acquisition
time, estimated sweep duration) and flags anything that risks a bad
measurement (source/voltmeter sharing a GPIB address, a degenerate
current range) as you type.

An optional gate voltage (Keithley 2400, off by default) can be held
fixed for the whole sweep, or given a comma-separated list of values —
one complete current sweep runs per value, each saved to its own file and
plotted together in the same window with a different color. The program
runs fine with no 2400 connected as long as the gate stays off.

Run with:
    python dc_iv_curve_tui.py

Requirements:
    pip install textual matplotlib  (in addition to dc_iv_curve.py's own deps)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.validation import Number
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Select,
    Static,
    Switch,
)

from dc.dc_iv_curve import (
    AcquisitionConfig,
    CurrentPoint,
    GateConfig,
    SourceConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    connect_gate,
    connect_source,
    connect_temperature_controller,
    connect_voltmeter,
    ramp_current_to_zero,
    run_measurement,
    set_gate_voltage,
    shutdown_gate,
    shutdown_source,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import linear_sweep, parse_value_list, safe_shutdown
from instruments.data_dir import validate_directory
from instruments.data_naming import (
    RunContext,
    allocate_run,
    finalize_index_row,
    make_incremental_writer,
    preview_raw_filename,
    write_record,
)
from instruments.keithley2182 import read_time_s
from instruments.run_time import (
    GATE_RAMP_S, GPIB_TXN_S, POINT_OVERHEAD_S, PER_FILE_S, PER_RUN_S, TEMP_READ_S,
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
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
)

log = logging.getLogger("dc_iv_curve_tui")

# Data/settings live outside "bridge" (a sibling of it). _DEFAULT_DATA_DIR is
# the fallback data-convention "data root"; the real root is chosen per run in
# the identity bar's "Data root" field -- see dc_hall_measurement_tui.py.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "dc_iv_curve_tui_settings.json"

# Locked type code (see instruments/data_naming.py) — never deviates.
MEASUREMENT_TYPE = "IV"

DC_IV_DESCRIPTION = (
    "Sweeps a DC current with a Keithley 6221 and records the DC voltage "
    "response with a Keithley 2182 at each point — a direct I-V curve, "
    "swept bidirectionally so hysteresis is visible. Far more informative "
    "than a single-point resistance for anything nonlinear (contacts, "
    "tunnel junctions, diodes, gated 2D systems). No magnet is involved; "
    "the current sweep is the whole measurement. An optional Keithley 2400 "
    "gate voltage (off by default) can be held fixed, or swept through a "
    "list of values — one complete I-V sweep per gate value, plotted "
    "together in different colors."
)

# Wiring schematic — shown on this program's card in bridge_tui.py.
DC_IV_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source)
    Output ──▶ DUT ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ across the DUT itself (2-terminal), or
                                   across the inner voltage-sense leads
                                   (4-terminal / Kelvin)

  Gate voltage  (optional, "Enable gate" switch)
    KEITHLEY 2400 (gate source) ──▶ gate electrode
"""


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors dc_iv_curve.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "source_visa_resource": "GPIB0::20::INSTR",
    "voltmeter_visa_resource": "GPIB0::7::INSTR",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "current_min_A": "-0.001",
    "current_max_A": "0.001",
    "nplc": "5",
    "auto_range": True,
    "settling_time_s": "0.2",
    "n_averages": "5",
    "device": "",
    "cooldown": "",
    "temperature_setpoint_K": "300",
    "step_A": "0.00005",
    "bidirectional_sweep": True,
    "enable_gate": False,
    "gate_visa_resource": "GPIB0::25::INSTR",
    "gate_voltage_limit_V": "20.0",
    "gate_compliance_current_A": "1e-6",
    "gate_voltage_values": "0.0",
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
}

# id -> caster, for every free-text numeric field (Switch handled separately)
NUMERIC_FIELDS: dict = {
    "compliance_V": float,
    "source_delay_s": float,
    "current_min_A": float,
    "current_max_A": float,
    "nplc": float,
    "settling_time_s": float,
    "n_averages": int,
    "step_A": float,
    "gate_voltage_limit_V": float,
    "gate_compliance_current_A": float,
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "device",
               "cooldown", "gate_visa_resource", "gate_voltage_values",
               "temperature_visa_resource", "temperature_sensor_uids", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K"]
GATE_FIELD_IDS = ["gate_visa_resource", "gate_voltage_limit_V",
                   "gate_compliance_current_A", "gate_voltage_values"]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def run_costs(n_sweep_points: int, state: dict) -> RunCost:
    """Modelled cost of the whole run, one entry per point in loop order
    (series-major: one full sweep per gate voltage). Also drives the run
    screen's progress bar, so estimate and live ETA cannot disagree."""
    n_series = len(state.get("gate_voltage_list") or []) if state.get("enable_gate") else 1
    n_series = max(1, n_series)
    rc = RunCost(n_sweep_points * n_series)
    has_temp = state.get("enable_temperature") and bool(parse_sensor_uids(state["temperature_sensor_uids"]))
    rc.each("settle", state["settling_time_s"])
    rc.each("2182 reads", state["n_averages"] * read_time_s(state["nplc"]))
    rc.each("overhead", GPIB_TXN_S + POINT_OVERHEAD_S + (TEMP_READ_S if has_temp else 0.0))
    for k in range(n_series):
        rc.at("per-file", PER_FILE_S, k * n_sweep_points)
    rc.at("per-run", PER_RUN_S, 0)
    # teardown: ramp_current_to_zero (1e-4 A per 0.02 s) + gate ramp-down when a gate is used
    i_end = state["current_min_A"] if state["bidirectional_sweep"] else state["current_max_A"]
    rc.tail("ramps", max(1, int(abs(i_end) / 1e-4)) * 0.02 + (GATE_RAMP_S if state.get("enable_gate") else 0.0))
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    acq_cfg: AcquisitionConfig
    currents_A: np.ndarray
    sample: str
    device: str
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    series: str
    gate_cfg: Optional[GateConfig] = None
    gate_voltages: Optional[List[float]] = None
    temp_cfg: Optional[TemperatureControllerConfig] = None
    data_root: Path = _DEFAULT_DATA_DIR
    run_cost: Optional[RunCost] = None      # modelled seconds per point (progress bar + ETA)

    @property
    def series_values(self) -> List[Optional[float]]:
        """[None] for a single (gate-less or fixed) run, else one entry per gate voltage."""
        return list(self.gate_voltages) if self.gate_voltages else [None]

    @property
    def total_points(self) -> int:
        return len(self.currents_A) * len(self.series_values)


def build_header_fields(plan: "MeasurementPlan", ctx: RunContext, records: list[dict], *,
                         status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """Universal + measurement-specific header/index fields for ONE run
    within this (possibly multi-file, one-per-gate-voltage) session -- see
    dc_spin_valve_tui.py's build_header_fields for the full rationale."""
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

    if state["source_visa_resource"] == state["voltmeter_visa_resource"]:
        errors.append("Source (6221) and voltmeter (2182) VISA resources must be different.")

    if state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")

    if state["current_min_A"] > state["current_max_A"]:
        errors.append("Sweep current min must be ≤ max — the 6221 range guard rejects any "
                      "point outside [min, max], so a reversed range fails on the first step.")
    elif state["current_min_A"] == state["current_max_A"]:
        warnings.append("current_min equals current_max — sweep will repeat a single point.")

    info.append(f"Current range: {format_si(state['current_min_A'], 'A')} → "
                f"{format_si(state['current_max_A'], 'A')}")

    read_s = read_time_s(state["nplc"])
    info.append(f"Estimated 2182 reading time ≈ {read_s * 1000:.0f} ms (NPLC={state['nplc']:g})")

    if state["step_A"] <= 0:
        errors.append("Sweep step size must be > 0 A.")
        n_one_way = 0
    else:
        n_one_way = max(2, round(abs(state["current_max_A"] - state["current_min_A"]) / state["step_A"]) + 1)
    n_sweep_points = n_one_way if not state["bidirectional_sweep"] else max(0, 2 * n_one_way - 1)
    direction = (f"{state['current_min_A']:g} A → {state['current_max_A']:g} A → {state['current_min_A']:g} A"
                 if state["bidirectional_sweep"]
                 else f"{state['current_min_A']:g} A → {state['current_max_A']:g} A")
    info.append(f"Sweep: {direction}, step={state['step_A']:g} A, {n_sweep_points} points")

    # ── Gate (optional) ─────────────────────────────────────────────────────
    if state["enable_gate"]:
        if state["gate_visa_resource"] in (state["source_visa_resource"], state["voltmeter_visa_resource"]):
            errors.append("Gate (2400) VISA resource must differ from the source/voltmeter resources.")
        if state["gate_voltage_limit_V"] <= 0:
            errors.append("Gate voltage limit must be > 0 V.")
        gate_list = state.get("gate_voltage_list", [])
        if state.get("gate_parse_error"):
            errors.append(f"Gate voltage list: {state['gate_parse_error']}")
        else:
            over_limit = [v for v in gate_list if abs(v) > state["gate_voltage_limit_V"]]
            if over_limit:
                errors.append(
                    f"Gate voltage(s) {over_limit} exceed the configured limit "
                    f"±{state['gate_voltage_limit_V']:g} V."
                )
            n_series = len(gate_list)
            if n_series > 1:
                info.append(f"Gate: {n_series} values {gate_list} — {n_series} complete sweeps, "
                            f"one file each, plotted together")
            else:
                info.append(f"Gate held fixed at {format_si(gate_list[0], 'V')}" if gate_list else "")
            info.extend(run_costs(n_sweep_points, state).lines("Estimated total run time"))
    else:
        info.extend(run_costs(n_sweep_points, state).lines("Estimated total run time"))

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
    suffix = (" (one file per gate voltage)" if state.get("enable_gate") and
              len(state.get("gate_voltage_list", [])) > 1 else "")
    return f"{preview}_<timestamp>.csv{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────
# A GUI matplotlib backend and Textual's terminal control both want the main
# thread. Rather than fight that, the live preview gets its own process with
# its own main thread; new points are streamed to it over a
# multiprocessing.Queue, tagged with a series_index/series_label so a
# gate-voltage list shows up as one colored trace per value. The final
# PNGs (one per run) are saved independently by the TUI process itself
# (see _save_measurement_png), so they don't depend on this window still
# being open when the run finishes.

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("DC I-V live measurement")
    except Exception:
        pass
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Live measurement — I-V curve")
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
            xs.append(record["current_A"])
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

    # Keep a reference so it isn't garbage-collected mid-run.
    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], png_path: Path,
                           plan: Optional["MeasurementPlan"] = None, comment: str = "") -> None:
    """Save a static I-V PNG from whatever points were actually collected
    (including an aborted/partial run).

    `records` is ONE run's points -- with several gate voltages each run is
    saved (and plotted) on its own, exactly like a manual run.

    `plan`/`comment` add a small "at a glance" text annotation (compliance
    voltage, the operator's comment) for context not already in the
    filename. Called once when the run ends (comment="") and again, to
    overwrite the PNG in place, once the operator's comment is known."""
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")  # headless — must not touch the TUI's terminal
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["current_A"] for r in records], [r["voltage_V"] for r in records],
            ".-", color="tab:blue")

    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Measurement result")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    lines: list[str] = []
    if plan is not None:
        compliance_V = plan.header_extra.get("compliance_V")
        if compliance_V is not None:
            lines.append(f"Compliance: {format_si(compliance_V, 'V')}")
    if comment:
        lines.append(f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}")
    if lines:
        fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color="0.4", va="bottom")
        fig.subplots_adjust(bottom=0.08 + 0.045 * len(lines))

    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot to '%s'", png_path)


# ─────────────────────────────────────────────────────────────────────────────
# Logging -> RichLog relay (keeps raw log lines from corrupting the alt screen)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Run screen  ── executes the plan in a worker thread, shows live progress
# ─────────────────────────────────────────────────────────────────────────────

class RunScreen(MeasurementRunScreen):
    ABORT_STATUS = "Abort requested — finishing current point, then ramping current to zero …"
    TABLE_COLUMNS = ("#", "Vg (V)", "I (A)", "V (V)", "R (Ω)", "T1 (K)", "T2 (K)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE

    def save_png(self, records: list[dict], png_path: Path, comment: str = "") -> None:
        _save_measurement_png(records, png_path, plan=self.plan, comment=comment)

    def live_plot_args(self):
        return (_live_plot_worker,)

    def table_row(self, record: dict) -> tuple:
        gate_V = record.get("gate_voltage_V")
        T1 = record.get("temperature_1_K")
        T2 = record.get("temperature_2_K")
        return (
            str(record["point_index"] + 1),
            f"{gate_V:.4g}" if gate_V is not None else "—",
            f"{record['current_A']:.4e}",
            f"{record['voltage_V']:.4e}",
            f"{record['resistance_ohm']:.5g}",
            f"{T1:.3f}" if T1 is not None else "—",
            f"{T2:.3f}" if T2 is not None else "—",
        )

    def build_header(self, ctx: RunContext, records: list[dict], *, status: str, comment: str,
                     extra: Optional[dict]) -> dict:
        return build_header_fields(self.plan, ctx, records, status=status, comment=comment, extra=extra)



    @work(thread=True, exclusive=True)
    def do_run(self) -> None:
        plan = self.plan
        source = None
        voltmeter = None
        gate = None
        temp_ctrl = None
        try:
            self._set_status_threadsafe("Connecting to Keithley 6221 & 2182 …")
            source = connect_source(plan.src_cfg)
            voltmeter = connect_voltmeter(plan.volt_cfg)

            if plan.temp_cfg is not None:
                self._set_status_threadsafe("Connecting to MercuryiTC (temperature) …")
                temp_ctrl = connect_temperature_controller(plan.temp_cfg)

            if plan.gate_cfg is not None:
                self._set_status_threadsafe("Connecting gate (Keithley 2400) …")
                gate = connect_gate(plan.gate_cfg)

            for series_idx, gate_V in enumerate(plan.series_values):
                if self._stop_event.is_set():
                    break

                label = None
                key_axis = None
                if gate_V is not None:
                    label = f"Vg={gate_V:g}V"
                    key_axis = ("gate_V", gate_V)
                    self._set_status_threadsafe(f"Setting gate to {gate_V:g} V …")
                    set_gate_voltage(gate, plan.gate_cfg, gate_V)

                # A fresh RunContext (own run number, own file) EVERY
                # iteration -- never reuse one across the gate-voltage series.
                ctx = allocate_run(
                    plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                    temperature_setpoint_K=plan.temperature_setpoint_K,
                    key_axis=key_axis, series=plan.series,
                )
                self._run_contexts.append(ctx)
                self._run_extras.append({"gate_voltage_V": gate_V} if gate_V is not None else None)
                self._set_run_label_threadsafe(f"Run #{ctx.run_str}")
                plan.acq_cfg.output_file = str(ctx.raw_path)
                write_csv = make_incremental_writer(
                    ctx.raw_path,
                    lambda records, _ctx=ctx, _gv=gate_V: build_header_fields(
                        plan, _ctx, records, status="in_progress", comment="",
                        extra={"gate_voltage_V": _gv} if _gv is not None else None,
                    ),
                )

                points = [CurrentPoint(current_A=float(i)) for i in plan.currents_A]

                status = "Running measurement …" if gate_V is None \
                    else f"Running measurement (Vg={gate_V:g} V) …"
                self._set_status_threadsafe(status)
                iter_error: Optional[BaseException] = None
                try:
                    run_measurement(
                        source, voltmeter, plan.src_cfg, plan.acq_cfg, points,
                        stop_event=self._stop_event,
                        on_point=self._make_on_point(series_idx, label),
                        gate_voltage_V=gate_V,
                        temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                        write_csv=write_csv,
                    )
                except Exception as exc:
                    iter_error = exc

                # Finalize THIS iteration's header/index row
                # UNCONDITIONALLY, right now.
                iter_status = "error" if iter_error is not None \
                    else ("aborted" if self._stop_event.is_set() else "completed")
                iter_records = [r for r in self._records if r.get("series_index", 0) == series_idx]
                header_fields = build_header_fields(
                    plan, ctx, iter_records, status=iter_status, comment="",
                    extra={"gate_voltage_V": gate_V} if gate_V is not None else None,
                )
                write_record(ctx.raw_path, iter_records, header_fields)
                finalize_index_row(self.plan.data_root, ctx.sample, ctx.run_number, header_fields)
                self._save_run_png(ctx, iter_records)

                if iter_error is not None:
                    raise iter_error

            final = "Measurement aborted." if self._stop_event.is_set() else "Measurement complete."
        except Exception as exc:
            log.exception("Measurement failed")
            final = f"ERROR: {exc}"
        finally:
            # 6221 output off first (immediate, no current into the DUT).
            if source is not None:
                safe_shutdown("source (ramp)", lambda: ramp_current_to_zero(source))
                safe_shutdown("source", lambda: shutdown_source(source))
            if gate is not None:
                safe_shutdown("gate", lambda: shutdown_gate(gate))
            if temp_ctrl is not None:
                safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))
            self.app.call_from_thread(self._on_finished, final)


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class DCIVCurveApp(MeasurementApp):
    TITLE = "DC I-V Curve"
    SUB_TITLE = "Keithley 6221 + 2182 · current sweep · optional gate"

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
                        "Current sweep (Keithley 6221)",
                        field("current_min_A", "Sweep current min (A)", DEFAULTS["current_min_A"]),
                        field("current_max_A", "Sweep current max (A)", DEFAULTS["current_max_A"]),
                        field("step_A", "Sweep step size (A)", DEFAULTS["step_A"],
                              validators=[Number(minimum=1e-12, failure_description="must be > 0")]),
                        switch_field("bidirectional_sweep", "Bidirectional (min → max → min)",
                                     DEFAULTS["bidirectional_sweep"]),
                    )
                    yield card(
                        "Gate voltage (Keithley 2400, optional)",
                        switch_field("enable_gate", "Enable gate", DEFAULTS["enable_gate"]),
                        field("gate_voltage_values", "Gate voltage (V)",
                              DEFAULTS["gate_voltage_values"], kind="text",
                              hint="Single value, or comma-separated list — one complete "
                                   "sweep runs per value, plotted together."),
                    )
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature", "Log temperature (MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                    )

                # ── Tier 2: precision / speed knobs — collapsed ─────────────
                with Collapsible(title="Acquisition & filter settings", collapsed=True):
                    with Vertical(classes="param-grid"):
                        yield card(
                            "Source & voltmeter",
                            field("compliance_V", "Compliance voltage (V)", DEFAULTS["compliance_V"],
                                  hint="Set high enough to reach the expected voltage at "
                                       "current_max_A, or the sweep clips against compliance.",
                                  validators=[Number(minimum=0.0, failure_description="must be ≥ 0")]),
                            field("nplc", "NPLC (integration time)", DEFAULTS["nplc"],
                                  hint="Bigger = quieter but slower.",
                                  validators=[Number(minimum=0.01, failure_description="must be > 0")]),
                            switch_field("auto_range", "Auto-range", DEFAULTS["auto_range"]),
                        )
                        yield card(
                            "Acquisition timing",
                            field("settling_time_s", "Settling time per current step (s)",
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
                            field("source_visa_resource", "6221 (current source)",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            field("voltmeter_visa_resource", "2182 (DUT voltage)",
                                  DEFAULTS["voltmeter_visa_resource"], kind="text"),
                            field("gate_visa_resource", "2400 (gate)",
                                  DEFAULTS["gate_visa_resource"], kind="text"),
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
                            "Temperature sensors",
                            field("temperature_sensor_uids", "MercuryiTC sensor board UID(s)",
                                  DEFAULTS["temperature_sensor_uids"], kind="text",
                                  hint="1-2 UIDs, comma-separated."),
                            muted=True,
                        )

            with Vertical(id="sidebar"):
                yield Static("Description", classes="sidebar-title")
                yield Static(DC_IV_DESCRIPTION, classes="card-desc")
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start measurement  (F5)", id="start", variant="success")
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────────


    # ── Sample picker ────────────────────────────────────────────────────────


    # ── Form state I/O ───────────────────────────────────────────────────────

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
        state["auto_range"] = self.query_one("#auto_range", Switch).value
        state["bidirectional_sweep"] = self.query_one("#bidirectional_sweep", Switch).value
        state["enable_gate"] = self.query_one("#enable_gate", Switch).value
        state["enable_temperature"] = self.query_one("#enable_temperature", Switch).value
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""

        state["gate_voltage_list"] = []
        state["gate_parse_error"] = None
        if state["enable_gate"]:
            try:
                state["gate_voltage_list"] = parse_value_list(state["gate_voltage_values"])
            except ValueError as exc:
                state["gate_parse_error"] = str(exc)

        return state, errors

    # ── Reactivity ───────────────────────────────────────────────────────────

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
        lines += [f"  [dim]•[/dim] {i}" for i in info if i]

        self.query_one("#summary", Static).update("\n".join(lines))
        self.query_one("#start", Button).disabled = bool(errors)

    # ── Start ────────────────────────────────────────────────────────────────

    def _build_plan(self, state: dict) -> MeasurementPlan:
        src_cfg = SourceConfig(
            visa_resource=state["source_visa_resource"],
            compliance_V=state["compliance_V"],
            source_delay_s=state["source_delay_s"],
            current_min_A=state["current_min_A"],
            current_max_A=state["current_max_A"],
        )
        volt_cfg = VoltmeterConfig(
            visa_resource=state["voltmeter_visa_resource"],
            nplc=state["nplc"],
            auto_range=state["auto_range"],
        )
        acq_cfg = AcquisitionConfig(
            settling_time_s=state["settling_time_s"],
            n_averages=state["n_averages"],
            output_file="",  # overwritten per series iteration in RunScreen
        )

        currents_A = linear_sweep(
            start=state["current_min_A"], stop=state["current_max_A"], step=state["step_A"],
            bidirectional=state["bidirectional_sweep"],
        )

        gate_cfg = None
        gate_voltages = None
        if state["enable_gate"]:
            gate_cfg = GateConfig(
                visa_resource=state["gate_visa_resource"],
                gate_voltage_limit_V=state["gate_voltage_limit_V"],
                compliance_current_A=state["gate_compliance_current_A"],
            )
            gate_voltages = state["gate_voltage_list"]

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
            "n_averages": state["n_averages"],
            "settling_time_s": state["settling_time_s"],
            "current_sweep_A": [state["current_min_A"], state["current_max_A"], state["step_A"]],
        }
        series = ""
        if len(gate_voltages or []) > 1:
            series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                      f"{datetime.now():%Y%m%dT%H%M%S}")

        return MeasurementPlan(
            src_cfg=src_cfg, volt_cfg=volt_cfg, acq_cfg=acq_cfg, currents_A=currents_A,
            data_root=self.data_root,
            sample=state["sample"], device=state["device"],
            temperature_setpoint_K=state["temperature_setpoint_K"],
            cooldown=state["cooldown"], header_extra=header_extra, series=series,
            gate_cfg=gate_cfg, gate_voltages=gate_voltages,
            temp_cfg=temp_cfg, run_cost=run_costs(len(currents_A), state),
        )


def main() -> None:
    DCIVCurveApp().run()


if __name__ == "__main__":
    main()
