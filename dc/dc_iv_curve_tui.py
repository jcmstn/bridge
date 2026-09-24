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
from dc.dc_sweep_utils import linear_sweep, safe_shutdown, sweep_point_count, try_parse
from instruments.data_dir import validate_directory
from instruments.data_naming import (
    RunContext,
    allocate_run,
    record_run,
    preview_raw_filename,
)
from instruments.keithley2182 import read_time_s
from instruments.run_time import (
    GATE_RAMP_S, GPIB_TXN_S, POINT_OVERHEAD_S, PER_FILE_S, PER_RUN_S, TEMP_READ_S,
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

log = logging.getLogger("dc_iv_curve_tui")

# Data/settings live outside "bridge" (a sibling of it). _DEFAULT_DATA_DIR is
# the fallback data-convention "data root"; the real root is chosen per run in
# the identity bar's "Data root" field -- see dc_hall_measurement_tui.py.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "dc_iv_curve_tui_settings.json"

# Locked type code (see instruments/data_naming.py) — never deviates.
MEASUREMENT_TYPE = "IV"

DC_IV_DESCRIPTION = (
    "6221 current sweep · 2182 voltage. No magnet. Optional 2400 gate: fixed, or a "
    "list → one sweep per value."
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

def resolve_state(state: dict) -> dict:
    """Add the derived keys build_summary() / build_plan() read — the parsed
    lists/sweeps, each with its parse error — to a state of raw field values.
    Pure: shared by the TUI's and the web page's parse_state()."""
    state["gate_voltage_list"], state["gate_parse_error"] = \
        try_parse(state["gate_voltage_values"]) if state["enable_gate"] else ([], None)
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

    if state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")

    if state["current_min_A"] > state["current_max_A"]:
        errors.append("Sweep current min must be ≤ max — the 6221 range guard rejects any "
                      "point outside [min, max], so a reversed range fails on the first step.")
    elif state["current_min_A"] == state["current_max_A"]:
        warnings.append("current_min equals current_max — sweep will repeat a single point.")

    read_s = read_time_s(state["nplc"])
    info.append(f"2182 read: ≈ {read_s * 1000:.0f} ms — NPLC {state['nplc']:g}")

    n_sweep_points = 0
    if state["step_A"] <= 0:
        errors.append("Sweep step size must be > 0 A.")
    else:
        try:
            n_sweep_points = sweep_point_count(state["current_min_A"], state["current_max_A"],
                                               state["step_A"], state["bidirectional_sweep"])
        except ValueError as exc:
            errors.append(str(exc))
    lo, hi = format_si(state["current_min_A"], "A"), format_si(state["current_max_A"], "A")
    direction = f"{lo} → {hi} → {lo}" if state["bidirectional_sweep"] else f"{lo} → {hi}"
    info.append(f"Current sweep: {direction} — step {format_si(state['step_A'], 'A')}, "
                f"{n_sweep_points} points")

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
                info.append(f"Gate: {', '.join(format_si(v, 'V') for v in gate_list)} — "
                            f"{n_series} sweeps, one file each")
            else:
                info.append(f"Gate: {format_si(gate_list[0], 'V')} — fixed" if gate_list else "")
            info.extend(run_costs(n_sweep_points, state).lines())
    else:
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
    """I-V + dV/dI PNG of ONE run's points (one PNG per gate voltage, like a
    manual run).

    `plan`/`comment` add a small "at a glance" text annotation (compliance
    voltage, the operator's comment) -- see dc_iv_curve_tui.py's
    _save_measurement_png for the same logic. Called once when the run
    ends (comment="") and again, to overwrite the PNG in place, once the
    operator's comment is known."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    rows = sorted(records, key=lambda r: r["point_index"])
    I = np.array([r["current_A"] for r in rows])
    V = np.array([r["voltage_V"] for r in rows])
    ax1.plot(I, V, ".-", color="tab:blue")
    if len(I) > 1:
        ax2.plot(I, np.gradient(V, I), ".-", color="tab:blue")
    ax1.set_xlabel("Current (A)"); ax1.set_ylabel("Voltage (V)")
    ax1.set_title("I-V curve"); ax1.grid(alpha=0.4)
    ax2.set_xlabel("Current (A)"); ax2.set_ylabel("dV/dI (Ω)")
    ax2.set_title("Differential resistance (numerical dV/dI)"); ax2.grid(alpha=0.4)
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
# Plan + run  ── pure, shared by the TUI RunScreen and web/dc/iv_curve.py
# ─────────────────────────────────────────────────────────────────────────────

def build_plan(state: dict, data_root: Path) -> MeasurementPlan:
    """One parsed, validated run request from a state dict. Pure — shared by
    the TUI and the web page."""
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
        data_root=data_root,
        sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series=series,
        gate_cfg=gate_cfg, gate_voltages=gate_voltages,
        temp_cfg=temp_cfg, run_cost=run_costs(len(currents_A), state),
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
    """Connect, then one I-V sweep (own run number, own file) per gate
    voltage (or one, gate off), each recorded + finalized by record_run()
    before the next; always ramp the current down and shut down. Pure — the
    TUI's RunScreen and the web page each pass their own callbacks."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = voltmeter = gate = temp_ctrl = None
    try:
        on_status("Connecting to Keithley 6221 & 2182 …")
        source = connect_source(plan.src_cfg)
        voltmeter = connect_voltmeter(plan.volt_cfg)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC (temperature) …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)

        if plan.gate_cfg is not None:
            on_status("Connecting gate (Keithley 2400) …")
            gate = connect_gate(plan.gate_cfg)

        for series_idx, gate_V in enumerate(plan.series_values):
            if stop_event.is_set():
                break
            label = key_axis = None
            if gate_V is not None:
                label = f"Vg={gate_V:g}V"
                key_axis = ("gate_V", gate_V)
                on_status(f"Setting gate to {gate_V:g} V …")
                set_gate_voltage(gate, plan.gate_cfg, gate_V)

            # A fresh RunContext (own run number, own file) EVERY iteration --
            # never reuse one across the gate-voltage series.
            ctx = allocate_run(
                plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                temperature_setpoint_K=plan.temperature_setpoint_K,
                key_axis=key_axis, series=plan.series,
            )
            extra = {"gate_voltage_V": gate_V} if gate_V is not None else None
            run_contexts.append(ctx)
            run_extras.append(extra)
            on_run_label(f"Run #{ctx.run_str}")
            plan.acq_cfg.output_file = str(ctx.raw_path)
            points = [CurrentPoint(current_A=float(i)) for i in plan.currents_A]

            on_status("Running measurement …" if gate_V is None
                      else f"Running measurement (Vg={gate_V:g} V) …")
            record_run(
                plan.data_root, ctx,
                lambda records, status, _ctx=ctx, _x=extra: build_header_fields(
                    plan, _ctx, records, status=status, comment="", extra=_x),
                lambda point_cb, write_csv, _points=points, _gv=gate_V: run_measurement(
                    source, voltmeter, plan.src_cfg, plan.acq_cfg, _points,
                    stop_event=stop_event, on_point=point_cb, gate_voltage_V=_gv,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg, write_csv=write_csv),
                stop_event, on_point=on_point,
                tags={"series_index": series_idx, "series_label": label},
                on_finished=on_run_finished)
    finally:
        # 6221 output off first (immediate, no current into the DUT).
        if source is not None:
            safe_shutdown("source (ramp)", lambda: ramp_current_to_zero(source))
            safe_shutdown("source", lambda: shutdown_source(source))
        if gate is not None:
            safe_shutdown("gate", lambda: shutdown_gate(gate))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (RunScreen and the web page both call this)."""
    _save_measurement_png(records, png_path, plan=plan, comment=comment)


# ─────────────────────────────────────────────────────────────────────────────
# Run screen  ── executes the plan in a worker thread, shows live progress
# ─────────────────────────────────────────────────────────────────────────────

class RunScreen(MeasurementRunScreen):
    ABORT_STATUS = "Abort requested — finishing current point, then ramping current to zero …"
    TABLE_COLUMNS = ("#", "Vg (V)", "I (A)", "V (V)", "R (Ω)", "T1 (K)", "T2 (K)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE

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
                              hint="Comma-separate for one sweep + file per value."),
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
                                  hint="Must exceed V at current_max_A, or the sweep clips.",
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
    DCIVCurveApp().run()


if __name__ == "__main__":
    main()
