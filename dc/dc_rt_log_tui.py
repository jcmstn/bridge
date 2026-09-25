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

Run with:
    python dc_rt_log_tui.py
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.validation import Number
from textual.widgets import Button, Collapsible, Footer, Header, Static

from dc.dc_rt_log import (
    MARKER,
    SENSOR_COLORS,
    SENSOR_COLUMNS,
    TIME_COLOR,
    AcquisitionConfig,
    SourceConfig,
    TemperatureControllerConfig,
    VoltmeterConfig,
    connect_source,
    connect_temperature_controller,
    connect_voltmeter,
    plot_results,
    ramp_current_to_zero,
    run_measurement,
    shutdown_source,
    sensors_in,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import check_sweep_size, safe_shutdown
from instruments.data_dir import validate_directory
from instruments.data_naming import RunContext, allocate_run, preview_raw_filename, record_run
from instruments.keithley2182 import read_time_s
from instruments.keithley6221 import reversal_avg_s
from instruments.run_time import PER_RUN_S, POINT_OVERHEAD_S, TEMP_READ_S, RunCost, format_duration
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
from instruments.tui_sample_picker import NEW_SAMPLE_SENTINEL

log = logging.getLogger("dc_rt_log_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "dc_rt_log_tui_settings.json"

# Locked type code (see docs/data_convention.md) — never deviates.
MEASUREMENT_TYPE = "RT"
PNG_SUFFIX = "R_vs_T"

DC_RT_LOG_DESCRIPTION = (
    "Continuous ±I reversal resistance log (6221 + 2182) while the temperature drifts "
    "freely; MercuryiTC read only, before and after every sample (drift saved per point)."
)

DC_RT_LOG_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source, fixed ±I)
    Output ──▶ current path through the sample ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ the voltage probe pair of interest

  MercuryiTC  (optional, read only — no temperature control)
    LAN ──▶ 1 or 2 temperature sensors
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
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
}

NUMERIC_FIELDS: dict = {
    "sense_current_A": float,
    "compliance_V": float,
    "source_delay_s": float,
    "nplc": float,
    "n_reversals": int,
    "interval_s": float,
    "max_duration_min": float,
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "device", "cooldown",
               "temperature_visa_resource", "temperature_sensor_uids", "data_dir"]
OPTIONAL_NUMERIC_FIELDS = ["temperature_setpoint_K", "T_stop_K"]
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids"]


def _has_temperature(state: dict) -> bool:
    return bool(state.get("enable_temperature")) and bool(parse_sensor_uids(state["temperature_sensor_uids"]))


def sample_time_s(state: dict) -> float:
    """Modelled wall time of one sample: T read, ±I reversal read, T read."""
    t = reversal_avg_s(state["n_reversals"], state["source_delay_s"], read_time_s(state["nplc"]))
    return t + (2 * TEMP_READ_S if _has_temperature(state) else 0.0)


def max_samples(state: dict) -> int:
    """Samples the run takes if it runs to its maximum duration."""
    period = max(state["interval_s"], sample_time_s(state) + POINT_OVERHEAD_S)
    return max(1, math.ceil(state["max_duration_min"] * 60.0 / period))


def run_costs(state: dict) -> RunCost:
    """Upper-bound cost: the run at its maximum duration (Stop or T_stop end it sooner)."""
    n = max_samples(state)
    check_sweep_size(n)
    period = max(state["interval_s"], sample_time_s(state) + POINT_OVERHEAD_S)
    rc = RunCost(n)
    rc.each("samples", period)
    rc.at("per-run", PER_RUN_S, 0)
    rc.tail("ramp", max(1, int(abs(state["sense_current_A"]) / 1e-4)) * 0.02)
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

    if state["source_visa_resource"] == state["voltmeter_visa_resource"]:
        errors.append("Source (6221) and voltmeter (2182) VISA resources must be different.")
    n_identity_errors = len(errors)
    if state["sense_current_A"] == 0:
        errors.append("Sense current must be non-zero (R = V_odd / I).")
    if state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")
    if state["n_reversals"] < 1:
        errors.append("±I pairs per sample must be ≥ 1.")
    elif state["n_reversals"] == 1:
        warnings.append("One ±I pair per sample gives no error bar (voltage_sem_V is nan).")
    if state["interval_s"] < 0:
        errors.append("Sample interval must be ≥ 0 s.")
    if state["max_duration_min"] <= 0:
        errors.append("Maximum duration must be > 0 min.")

    info.append(f"Sense current: ±{format_si(abs(state['sense_current_A']), 'A')}, "
                f"{state['n_reversals']} ±I pairs per sample")
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

    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if not uids:
            warnings.append("Temperature logging is on but no sensor UID is set — "
                            "the log runs against time only.")
        else:
            info.append(f"Temperature: MercuryiTC {', '.join(uids)} — probed at start, "
                        "sensors that don't answer are dropped")
    else:
        warnings.append("Temperature logging is off — the log runs against time only.")

    return info, warnings, errors


def compute_filename_preview(state: dict) -> Optional[str]:
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL or not state.get("device"):
        return None
    preview = preview_raw_filename(state["sample"], state["device"], MEASUREMENT_TYPE,
                                   temperature_setpoint_K=state.get("temperature_setpoint_K"))
    return f"{preview}_<timestamp>.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── own OS process (see dc_iv_curve_tui.py for why)
# ─────────────────────────────────────────────────────────────────────────────

def _live_plot_worker(queue: "mp.Queue") -> None:
    """Same layout as the PNG (dc_rt_log.plot_results): one R-vs-T panel per
    sensor that is reading, then R vs time. The panels are built on the first
    record, since which sensors answered is only known after connect."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(7, 6))
    try:
        fig.canvas.manager.set_window_title("R vs T live log")
    except Exception:
        pass
    fig.text(0.5, 0.5, "Waiting for the first sample …", ha="center", va="center", color="0.5")
    panels: list = []          # (axes, line, x column, xs, ys)

    def _build(record: dict) -> None:
        fig.clear()
        sensors = sensors_in([record])
        axes = fig.subplots(len(sensors) + 1, 1, squeeze=False)[:, 0]
        for ax, k in zip(axes, sensors):
            (line,) = ax.plot([], [], color=SENSOR_COLORS[k - 1], **MARKER)
            ax.set_xlabel(f"Temperature, sensor {k} (K)")
            panels.append((ax, line, SENSOR_COLUMNS[k - 1], [], []))
        (line,) = axes[-1].plot([], [], color=TIME_COLOR, **MARKER)
        axes[-1].set_xlabel("Time (min)")
        panels.append((axes[-1], line, "elapsed_min", [], []))
        for ax in axes:
            ax.set_ylabel("R (Ω)")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()

    def _drain(_frame=None):
        got = False
        while True:
            try:
                record = queue.get_nowait()
            except Exception:
                break
            if not panels:
                _build(record)
            record = {**record, "elapsed_min": record["elapsed_s"] / 60.0}
            for _ax, _line, col, xs, ys in panels:
                if record.get(col) is not None:
                    xs.append(record[col])
                    ys.append(record["resistance_ohm"])
            got = True
        if got:
            for ax, line, _col, xs, ys in panels:
                line.set_data(xs, ys)
                ax.relim()
                ax.autoscale_view()
        return tuple(p[1] for p in panels)

    _ani = FuncAnimation(fig, _drain, interval=500, cache_frame_data=False)
    plt.show()


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (RunScreen and the web page both call this)."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    note = f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}" if comment else ""
    plot_results(pd.DataFrame(records), png_path, note=note)


# ─────────────────────────────────────────────────────────────────────────────
# Plan + run  ── pure, shared by the TUI RunScreen and web/dc/rt_log.py
# ─────────────────────────────────────────────────────────────────────────────

def build_plan(state: dict, data_root: Path) -> MeasurementPlan:
    src_cfg = SourceConfig(
        visa_resource=state["source_visa_resource"],
        sense_current_A=state["sense_current_A"],
        compliance_V=state["compliance_V"],
        source_delay_s=state["source_delay_s"],
    )
    volt_cfg = VoltmeterConfig(
        visa_resource=state["voltmeter_visa_resource"],
        nplc=state["nplc"],
        auto_range=state["auto_range"],
    )
    acq_cfg = AcquisitionConfig(
        n_reversals=state["n_reversals"],
        interval_s=state["interval_s"],
        max_duration_s=state["max_duration_min"] * 60.0,
        T_stop_K=state.get("T_stop_K"),
        output_file="",
    )
    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

    header_extra = {
        "sense_current_A": state["sense_current_A"],
        "compliance_V": state["compliance_V"],
        "n_reversals": state["n_reversals"],
        "nplc": state["nplc"],
        "source_delay_s": state["source_delay_s"],
        "interval_s": state["interval_s"],
        "max_duration_min": state["max_duration_min"],
        "T_stop_K": state.get("T_stop_K"),
    }
    run_cost = run_costs(state)
    return MeasurementPlan(
        src_cfg=src_cfg, volt_cfg=volt_cfg, acq_cfg=acq_cfg,
        sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state.get("temperature_setpoint_K"),
        cooldown=state["cooldown"], header_extra=header_extra,
        temp_cfg=temp_cfg, data_root=data_root,
        run_cost=run_cost, total_points=len(run_cost.points),
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
    """Connect, probe the iTC sensors, log one run until Stop / max duration
    / T_stop, finalize it, and always ramp the current down and shut down."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    source = temp_ctrl = None
    try:
        on_status("Connecting to Keithley 6221 & 2182 …")
        source = connect_source(plan.src_cfg)
        voltmeter = connect_voltmeter(plan.volt_cfg)

        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC and probing its sensors (see log) …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)   # narrows sensor_uids
        temp_cfg = plan.temp_cfg if temp_ctrl is not None else None

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
                on_point=point_cb, temp_ctrl=temp_ctrl, temp_cfg=temp_cfg, write_csv=write_csv),
            None, on_point=on_point, on_finished=on_run_finished)
    finally:
        if source is not None:
            safe_shutdown("source (ramp)", lambda: ramp_current_to_zero(source))
            safe_shutdown("source", lambda: shutdown_source(source))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


# ─────────────────────────────────────────────────────────────────────────────
# Run screen
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value: Optional[float], spec: str) -> str:
    return format(value, spec) if value is not None else "—"


class RunScreen(MeasurementRunScreen):
    ABORT_LABEL = "Stop logging"
    ABORT_STATUS = "Stop requested — finishing the current sample, then ramping current to zero …"
    DONE_STATUS = "Log finished."
    STOP_IS_NORMAL_END = True      # Stop ends a log normally -> "completed" in the run history
    POINT_STATUS = "Sample {n} (at most {total})."
    TABLE_COLUMNS = ("#", "t (s)", "T1 (K)", "ΔT1 (mK)", "T2 (K)", "R (Ω)", "σR (Ω)")
    MEASUREMENT_TYPE = MEASUREMENT_TYPE
    PNG_SUFFIX = PNG_SUFFIX

    def live_plot_args(self):
        return (_live_plot_worker,)

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
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class DCRTLogApp(MeasurementApp):
    TITLE = "DC R vs T log"
    SUB_TITLE = "Keithley 6221 + 2182 · fixed ±I · temperature read only"

    data_root: Path = _DEFAULT_DATA_DIR

    SWITCH_DEPENDENTS = {"enable_temperature": tuple(TEMPERATURE_FIELD_IDS)}

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
                        "Sense current (Keithley 6221)",
                        field("sense_current_A", "Sense current (A)", DEFAULTS["sense_current_A"],
                              hint="Reversed ±I every pair. Keep it small enough to avoid self-heating."),
                        field("n_reversals", "±I pairs per sample", DEFAULTS["n_reversals"],
                              kind="integer",
                              validators=[Number(minimum=1, failure_description="must be ≥ 1")],
                              hint="More = quieter, but a longer sample (more drift inside it)."),
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
                    yield card(
                        "Temperature logging",
                        switch_field("enable_temperature", "Log temperature (MercuryiTC)",
                                     DEFAULTS["enable_temperature"]),
                        field("temperature_sensor_uids", "Sensor UID(s)",
                              DEFAULTS["temperature_sensor_uids"], kind="text",
                              hint="1 or 2, e.g. MB1.T1, DB5.T1 — a missing one is dropped."),
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
                        )

                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Instrument addresses",
                            field("source_visa_resource", "6221 (current source)",
                                  DEFAULTS["source_visa_resource"], kind="text"),
                            field("voltmeter_visa_resource", "2182 (voltage)",
                                  DEFAULTS["voltmeter_visa_resource"], kind="text"),
                            field("temperature_visa_resource", "MercuryiTC",
                                  DEFAULTS["temperature_visa_resource"], kind="text"),
                            muted=True,
                        )
                        yield card(
                            "Source timing",
                            field("source_delay_s", "6221 source delay (s)", DEFAULTS["source_delay_s"],
                                  hint="Settle after each polarity flip."),
                            muted=True,
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

    def _build_plan(self, state: dict) -> MeasurementPlan:
        return build_plan(state, self.data_root)


def main() -> None:
    DCRTLogApp().run()


if __name__ == "__main__":
    main()
