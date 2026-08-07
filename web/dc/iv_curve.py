#!/usr/bin/env python3
"""
NiceGUI page for dc_iv_curve.py
====================================
Web equivalent of dc_iv_curve_tui.py. Reuses that TUI module's pure
DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/build_summary()/parse_sensor_uids() so
validation stays identical to the TUI.

Optional gate-voltage list means multiple complete current sweeps run per
Start click (plan.series_values), one CSV per value plus one combined
overlay PNG -- ported from RunScreen.do_run()'s per-series loop, just
targeting a user-chosen data_dir instead of the fixed _DATA_DIR.

Live view adds a second panel (dV/dI via np.gradient, recomputed on each
drain tick) that the TUI's live plot doesn't show -- the TUI only computes
that in the headless module's plot_results(), used by main(), not by the
TUI. Showing it live here matches an artifact that already exists
elsewhere, not a new invention.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui

_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "instruments"
_DC_DIR = Path(__file__).resolve().parent.parent.parent / "DC"
_WEB_DIR = Path(__file__).resolve().parent.parent
for _p in (_INSTRUMENTS_DIR, _DC_DIR, _WEB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dc_iv_curve import (  # noqa: E402
    AcquisitionConfig, CurrentPoint, GateConfig, SourceConfig,
    TemperatureControllerConfig, VoltmeterConfig,
    connect_gate, connect_source, connect_temperature_controller, connect_voltmeter,
    ramp_current_to_zero, run_measurement, set_gate_voltage, shutdown_gate,
    shutdown_source, shutdown_temperature_controller,
)
from dc_sweep_utils import build_output_path, linear_sweep, parse_value_list  # noqa: E402
from dc_iv_curve_tui import (  # noqa: E402
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, DC_IV_DESCRIPTION,
    build_summary, parse_sensor_uids,
)
from run_controller import (  # noqa: E402
    RunController, RunCallbacks, FinalStatus, num_field, text_field, bool_switch,
    render_summary, busy_banner, is_busy,
)
from directory_picker import directory_field, validate_directory  # noqa: E402

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "dc_iv_curve_web_settings.json"

PAGE_TITLE = "DC I-V Curve"
SUITE = "DC"


@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    acq_cfg: AcquisitionConfig
    currents_A: np.ndarray
    output_subdir: str
    output_prefix: str
    gate_cfg: Optional[GateConfig] = None
    gate_voltages: Optional[List[float]] = None
    temp_cfg: Optional[TemperatureControllerConfig] = None

    @property
    def series_values(self) -> List[Optional[float]]:
        return list(self.gate_voltages) if self.gate_voltages else [None]

    @property
    def total_points(self) -> int:
        return len(self.currents_A) * len(self.series_values)


def _load_settings() -> dict:
    try:
        return json.loads(_SETTINGS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_settings(raw: dict) -> None:
    try:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_PATH.write_text(json.dumps(raw, indent=2))
    except OSError:
        pass


def build_plan(state: dict) -> MeasurementPlan:
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
        n_averages=int(state["n_averages"]),
        output_file="",  # per-series, filled in by run_fn
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
                visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

    return MeasurementPlan(
        src_cfg=src_cfg, volt_cfg=volt_cfg, acq_cfg=acq_cfg, currents_A=currents_A,
        output_subdir=state["output_subdir"], output_prefix=state["output_name"],
        gate_cfg=gate_cfg, gate_voltages=gate_voltages, temp_cfg=temp_cfg,
    )


def series_label(gate_V: Optional[float]) -> Optional[str]:
    return f"Vg={gate_V:g}V" if gate_V is not None else None


def output_path_for_series(plan: MeasurementPlan, data_dir: str, session_ts: str,
                            gate_V: Optional[float]) -> str:
    suffix = f"_Vg{gate_V:g}V" if gate_V is not None else ""
    return str(build_output_path(Path(data_dir), plan.output_subdir, plan.output_prefix, session_ts, suffix))


def _save_combined_png(records: list[dict], png_path: Path) -> None:
    """Combined I-V + dV/dI overlay PNG, one color per gate-voltage series."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    series_ids = sorted({r.get("series_index", 0) for r in records})
    for idx in series_ids:
        rows = sorted((r for r in records if r.get("series_index", 0) == idx),
                      key=lambda r: r["point_index"])
        label = rows[0].get("series_label")
        I = np.array([r["current_A"] for r in rows])
        V = np.array([r["voltage_V"] for r in rows])
        ax1.plot(I, V, ".-", color=cmap(idx % 10), label=label)
        if len(I) > 1:
            ax2.plot(I, np.gradient(V, I), ".-", color=cmap(idx % 10), label=label)
    ax1.set_xlabel("Current (A)"); ax1.set_ylabel("Voltage (V)")
    ax1.set_title("I-V curve"); ax1.grid(alpha=0.4)
    ax2.set_xlabel("Current (A)"); ax2.set_ylabel("dV/dI (Ω)")
    ax2.set_title("Differential resistance (numerical dV/dI)"); ax2.grid(alpha=0.4)
    if any(r.get("series_label") for r in records):
        ax1.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def page() -> None:
    ui.page_title(PAGE_TITLE)
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(DC_IV_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = _load_settings()

    def d(key: str):
        if key in saved:
            return saved[key]
        return DEFAULTS.get(key, "")

    inputs: dict = {}
    switches: dict = {}
    controller: dict[str, Optional[RunController]] = {"c": None}

    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        with ui.column().classes("flex-grow gap-1 max-w-3xl"):
            with ui.expansion("Instruments", value=True, icon="cable").classes("w-full"):
                inputs["source_visa_resource"] = text_field("Keithley 6221 (current source)", d("source_visa_resource"))
                inputs["voltmeter_visa_resource"] = text_field("Keithley 2182 (DUT voltage)", d("voltmeter_visa_resource"))

            with ui.expansion("Current sweep & compliance", value=True, icon="bolt").classes("w-full"):
                inputs["current_min_A"] = num_field("Sweep current min (A)", float(d("current_min_A")))
                inputs["current_max_A"] = num_field("Sweep current max (A)", float(d("current_max_A")))
                inputs["compliance_V"] = num_field(
                    "Compliance voltage (V)", float(d("compliance_V")),
                    hint="Set high enough to reach the expected voltage at current_max_A.")
                with ui.expansion("Source timing (advanced)"):
                    inputs["source_delay_s"] = num_field("6221 source delay (s)", float(d("source_delay_s")))

            with ui.expansion("Voltmeter (Keithley 2182)", value=True, icon="speed").classes("w-full"):
                inputs["nplc"] = num_field("NPLC (integration time)", float(d("nplc")))
                switches["auto_range"] = bool_switch("Auto-range", d("auto_range"))

            with ui.expansion("Acquisition & output", value=True, icon="save").classes("w-full"):
                inputs["settling_time_s"] = num_field("Settling time per current step (s)", float(d("settling_time_s")))
                inputs["n_averages"] = num_field("Voltage samples averaged per point", float(d("n_averages")), integer=True)
                inputs["output_name"] = text_field("Output file name (prefix)", d("output_name"))
                inputs["output_subdir"] = text_field("Data sub-directory (optional)", d("output_subdir"))
                data_dir_input = directory_field("Save directory", saved.get("data_dir") or str(_DATA_DIR))

            with ui.expansion("Sweep resolution", value=True, icon="tune").classes("w-full"):
                inputs["step_A"] = num_field("Sweep step size (A)", float(d("step_A")))
                switches["bidirectional_sweep"] = bool_switch(
                    "Bidirectional sweep (min → max → min)", d("bidirectional_sweep"))

            with ui.expansion("Gate voltage (Keithley 2400, optional)", value=False, icon="tune").classes("w-full"):
                switches["enable_gate"] = bool_switch("Enable gate (Keithley 2400)", d("enable_gate"))
                inputs["gate_visa_resource"] = text_field("Keithley 2400 (gate) VISA resource", d("gate_visa_resource"))
                inputs["gate_voltage_limit_V"] = num_field("Gate voltage software limit (V)", float(d("gate_voltage_limit_V")))
                inputs["gate_compliance_current_A"] = num_field("Gate leakage compliance (A)", float(d("gate_compliance_current_A")))
                inputs["gate_voltage_values"] = text_field(
                    "Gate voltage (V)", d("gate_voltage_values"),
                    hint="Single value, or comma-separated list — one complete current sweep "
                         "runs per value, each saved to its own file and plotted together.")

            with ui.expansion("Temperature (MercuryiTC)", value=True, icon="thermostat").classes("w-full"):
                switches["enable_temperature"] = bool_switch(
                    "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))
                inputs["temperature_visa_resource"] = text_field("MercuryiTC VISA resource", d("temperature_visa_resource"))
                inputs["temperature_sensor_uids"] = text_field("Sensor board UID(s)", d("temperature_sensor_uids"))

        with ui.column().classes("w-96 gap-2"):
            ui.label("Summary").classes("text-lg font-bold")
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start measurement", color="primary").classes("w-full")

    ui.separator().classes("my-3")
    status_label = ui.label("Idle.").classes("text-sm font-bold")
    abort_btn = ui.button("Abort (safe ramp-down)", color="negative").props("outline")
    abort_btn.set_visibility(False)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("I-V curve", "Differential resistance (dV/dI)"))
    fig.update_xaxes(title_text="Current (A)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_xaxes(title_text="Current (A)", row=2, col=1)
    fig.update_yaxes(title_text="dV/dI (Ω)", row=2, col=1)
    fig.update_layout(margin=dict(l=60, r=20, t=40, b=50), height=650, showlegend=True)
    plot = ui.plotly(fig).classes("w-full")

    columns = [
        {"name": "n", "label": "#", "field": "n"},
        {"name": "Vg", "label": "Vg (V)", "field": "Vg"},
        {"name": "I", "label": "I (A)", "field": "I"},
        {"name": "V", "label": "V (V)", "field": "V"},
        {"name": "R", "label": "R (Ω)", "field": "R"},
        {"name": "T1", "label": "T1 (K)", "field": "T1"},
        {"name": "T2", "label": "T2 (K)", "field": "T2"},
    ]
    table = ui.table(columns=columns, rows=[], row_key="n").classes("w-full").props("dense")
    log_area = ui.log(max_lines=2000).classes("w-full h-48 font-mono text-xs")

    def parse_state() -> tuple[dict, list[str]]:
        errors: list[str] = []
        state: dict = {}
        for fid, caster in NUMERIC_FIELDS.items():
            raw = inputs[fid].value
            try:
                state[fid] = caster(raw)
            except (TypeError, ValueError):
                errors.append(f"'{fid}' is not a valid number.")
                state[fid] = 0
        for fid in TEXT_FIELDS:
            state[fid] = (inputs[fid].value or "").strip()
        for fid, sw in switches.items():
            state[fid] = sw.value
        state["gate_voltage_list"] = []
        state["gate_parse_error"] = None
        if state["enable_gate"]:
            try:
                state["gate_voltage_list"] = parse_value_list(state["gate_voltage_values"])
            except ValueError as exc:
                state["gate_parse_error"] = str(exc)
        return state, errors

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["data_dir"] = data_dir_input.value
        return raw

    @ui.refreshable
    def refresh_summary() -> None:
        state, parse_errors = parse_state()
        dir_warning, dir_error = validate_directory(data_dir_input.value or "")
        if parse_errors:
            info, warnings, errors = [], [], parse_errors
        else:
            info, warnings, errors = build_summary(state)
        if dir_warning:
            warnings = warnings + [dir_warning]
        if dir_error:
            errors = errors + [dir_error]
        with summary_box:
            summary_box.clear()
            render_summary([i for i in info if i], warnings, errors)
        start_btn.set_enabled(not errors and not is_busy())

    for inp in list(inputs.values()) + [data_dir_input]:
        inp.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    refresh_summary()
    ui.timer(2.0, refresh_summary.refresh)

    # ── Run wiring ───────────────────────────────────────────────────────

    series_state: dict[str, dict] = {}  # {"traces_iv": {idx: trace_index}, "traces_dvdi": {...}, "arrays": {idx: (I, V)}}

    def init_series(n_series: int, labels: list[Optional[str]]) -> None:
        fig.data = []
        series_state["traces_iv"] = {}
        series_state["traces_dvdi"] = {}
        series_state["arrays"] = {}
        cmap = ["#2E3192", "#e34948", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#ff7f0e", "#7f7f7f"]
        for i in range(n_series):
            color = cmap[i % len(cmap)]
            name = labels[i] or "I-V"
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name=name,
                                      legendgroup=str(i), line=dict(color=color)), row=1, col=1)
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name=name, showlegend=False,
                                      legendgroup=str(i), line=dict(color=color)), row=2, col=1)
            series_state["traces_iv"][i] = 2 * i
            series_state["traces_dvdi"][i] = 2 * i + 1
            series_state["arrays"][i] = ([], [])

    def on_record(record: dict) -> None:
        idx = record.get("series_index", 0)
        I_list, V_list = series_state["arrays"][idx]
        I_list.append(record["current_A"])
        V_list.append(record["voltage_V"])
        iv_i = series_state["traces_iv"][idx]
        fig.data[iv_i].x = tuple(I_list)
        fig.data[iv_i].y = tuple(V_list)
        if len(I_list) > 1:
            dvdi = np.gradient(np.array(V_list), np.array(I_list))
            dvdi_i = series_state["traces_dvdi"][idx]
            fig.data[dvdi_i].x = tuple(I_list)
            fig.data[dvdi_i].y = tuple(dvdi)
        plot.update()

        table.rows.append({
            "n": record["point_index"] + 1,
            "Vg": f"{record['gate_voltage_V']:.4g}" if record.get("gate_voltage_V") is not None else "—",
            "I": f"{record['current_A']:.4e}",
            "V": f"{record['voltage_V']:.4e}",
            "R": f"{record['resistance_ohm']:.5g}",
            "T1": f"{record['temperature_1_K']:.3f}" if record.get("temperature_1_K") is not None else "—",
            "T2": f"{record['temperature_2_K']:.3f}" if record.get("temperature_2_K") is not None else "—",
        })
        table.update()

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

    def on_finished(final: FinalStatus, result) -> None:
        label = {"completed": "Measurement complete.", "aborted": "Measurement aborted.",
                  "error": f"ERROR: {final.error}"}[final.status]
        status_label.set_text(label)
        abort_btn.set_visibility(False)
        start_btn.set_enabled(not is_busy())
        refresh_summary.refresh()

    def make_run_fn(plan: MeasurementPlan, data_dir: str, session_ts: str):
        def run_fn(stop_event, cb: RunCallbacks):
            source = voltmeter = gate = temp_ctrl = None
            try:
                cb.on_status("Connecting to Keithley 6221 & 2182 …")
                source = connect_source(plan.src_cfg)
                voltmeter = connect_voltmeter(plan.volt_cfg)

                if plan.temp_cfg is not None:
                    cb.on_status("Connecting to MercuryiTC (temperature) …")
                    temp_ctrl = connect_temperature_controller(plan.temp_cfg)

                if plan.gate_cfg is not None:
                    cb.on_status("Connecting gate (Keithley 2400) …")
                    gate = connect_gate(plan.gate_cfg)

                for series_idx, gate_V in enumerate(plan.series_values):
                    if stop_event.is_set():
                        break
                    label = series_label(gate_V)
                    if gate_V is not None:
                        cb.on_status(f"Setting gate to {gate_V:g} V …")
                        set_gate_voltage(gate, plan.gate_cfg, gate_V)

                    plan.acq_cfg.output_file = output_path_for_series(plan, data_dir, session_ts, gate_V)
                    points = [CurrentPoint(current_A=float(i)) for i in plan.currents_A]

                    def tagged_on_point(record: dict, _idx=series_idx, _label=label) -> None:
                        record["series_index"] = _idx
                        record["series_label"] = _label
                        cb.on_point(record)

                    status = "Running measurement …" if gate_V is None else f"Running measurement (Vg={gate_V:g} V) …"
                    cb.on_status(status)
                    run_measurement(
                        source, voltmeter, plan.src_cfg, plan.acq_cfg, points,
                        stop_event=stop_event, on_point=tagged_on_point,
                        gate_voltage_V=gate_V, temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                    )
                return None
            finally:
                if gate is not None:
                    shutdown_gate(gate)
                if temp_ctrl is not None:
                    shutdown_temperature_controller(temp_ctrl)
                if source is not None:
                    ramp_current_to_zero(source)
                    shutdown_source(source)
        return run_fn

    def on_start() -> None:
        state, parse_errors = parse_state()
        dir_warning, dir_error = validate_directory(data_dir_input.value or "")
        if parse_errors or dir_error:
            ui.notify("Fix the blocking issues before starting.", type="negative")
            return
        _, _, errors = build_summary(state)
        if errors:
            ui.notify("Fix the blocking issues before starting.", type="negative")
            return
        state["data_dir"] = data_dir_input.value.strip()

        _save_settings(collect_raw())

        plan = build_plan(state)
        Path(state["data_dir"]).mkdir(parents=True, exist_ok=True)
        session_ts = f"{datetime.now():%Y%m%d_%H%M%S}"
        labels = [series_label(v) for v in plan.series_values]
        output_paths = [output_path_for_series(plan, state["data_dir"], session_ts, v) for v in plan.series_values]

        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE,
            run_fn=make_run_fn(plan, state["data_dir"], session_ts),
            save_artifacts=lambda records, result: _finish_artifacts(records, output_paths, state["data_dir"],
                                                                       plan.output_prefix, session_ts),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=output_paths,
            on_record=on_record, on_status=on_status, on_log=on_log, on_finished=on_finished,
        )
        if not rc.try_start():
            ui.notify("Another measurement is already running — see the banner above.", type="warning")
            return
        controller["c"] = rc

        init_series(len(plan.series_values), labels)
        plot.update()
        table.rows.clear()
        table.update()
        log_area.clear()
        abort_btn.set_visibility(True)
        start_btn.set_enabled(False)

    def _finish_artifacts(records: list[dict], csv_paths: list[str], data_dir: str,
                           prefix: str, session_ts: str) -> list[str]:
        out_dir = Path(data_dir)
        png_path = out_dir / f"{prefix}_{session_ts}_combined.png"
        try:
            _save_combined_png(records, png_path)
            return csv_paths + [str(png_path)]
        except Exception:
            return csv_paths

    def on_abort() -> None:
        if controller["c"] is not None:
            controller["c"].abort()

    start_btn.on_click(on_start)
    abort_btn.on_click(on_abort)
