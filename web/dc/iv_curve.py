#!/usr/bin/env python3
"""
NiceGUI page for dc_iv_curve.py
====================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Web equivalent of dc_iv_curve_tui.py. Reuses that TUI module's pure
DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/build_summary()/parse_sensor_uids() so
validation stays identical to the TUI.

Optional gate-voltage list means multiple complete current sweeps run per
Start click, one CSV per value plus one combined overlay PNG.

Live view adds a second panel (dV/dI via np.gradient, recomputed on each
drain tick) alongside the raw I-V trace.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import background_tasks, ui

from dc.dc_iv_curve import (
    AcquisitionConfig, CurrentPoint, GateConfig, SourceConfig,
    TemperatureControllerConfig, VoltmeterConfig,
    connect_gate, connect_source, connect_temperature_controller, connect_voltmeter,
    ramp_current_to_zero, run_measurement, set_gate_voltage, shutdown_gate,
    shutdown_source, shutdown_temperature_controller,
)
from dc.dc_sweep_utils import linear_sweep, parse_value_list
from dc.dc_iv_curve_tui import (
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, OPTIONAL_NUMERIC_FIELDS, MEASUREMENT_TYPE,
    DC_IV_DESCRIPTION, MeasurementPlan, build_header_fields, build_summary,
    compute_filename_preview, parse_sensor_uids,
)
from instruments.data_naming import (
    TEST_SAMPLE, RunContext, allocate_run, finalize_index_row,
    make_incremental_writer, preview_raw_filename, proc_path, write_record,
)
from web.run_controller import (
    RunController, RunCallbacks, FinalStatus, num_field, optional_num_field, text_field,
    bool_switch, render_summary, busy_banner, is_busy,
    param_card, stable_card, param_grid, stable_grid, section_title,
)
from web.directory_picker import validate_directory
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, status_comment_dialog

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "dc_iv_curve_web_settings.json"

PAGE_TITLE = "DC I-V Curve"
SUITE = "DC"


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

    header_extra = {
        "compliance_V": state["compliance_V"],
        "n_averages": int(state["n_averages"]),
        "settling_time_s": state["settling_time_s"],
        "current_sweep_A": [state["current_min_A"], state["current_max_A"], state["step_A"]],
    }
    series = ""
    if len(gate_voltages or []) > 1:
        series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                  f"{datetime.now():%Y%m%dT%H%M%S}")

    return MeasurementPlan(
        src_cfg=src_cfg, volt_cfg=volt_cfg, acq_cfg=acq_cfg, currents_A=currents_A,
        sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series=series,
        gate_cfg=gate_cfg, gate_voltages=gate_voltages, temp_cfg=temp_cfg,
    )


def series_label(gate_V: Optional[float]) -> Optional[str]:
    return f"Vg={gate_V:g}V" if gate_V is not None else None


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

    _t_default = d("temperature_setpoint_K")
    identity = identity_bar(
        default_data_dir=saved.get("data_dir") or str(_DATA_DIR),
        default_sample=saved.get("sample") or TEST_SAMPLE,
        default_device=d("device"), default_cooldown=d("cooldown"),
        default_temperature_K=float(_t_default) if _t_default not in ("", None) else None,
    )

    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        with ui.column().classes("flex-grow gap-1"):
            with param_grid():
                with param_card("Source (6221) — current sweep"):
                    inputs["current_min_A"] = num_field("Sweep current min (A)", float(d("current_min_A")))
                    inputs["current_max_A"] = num_field("Sweep current max (A)", float(d("current_max_A")))
                    inputs["step_A"] = num_field("Sweep step size (A)", float(d("step_A")))
                    switches["bidirectional_sweep"] = bool_switch(
                        "Bidirectional sweep (min → max → min)", d("bidirectional_sweep"))
                    inputs["compliance_V"] = num_field(
                        "Compliance voltage (V)", float(d("compliance_V")),
                        hint="Set high enough to reach the expected voltage at current_max_A.")

                with param_card("Voltmeter (2182)"):
                    inputs["nplc"] = num_field("NPLC (integration time)", float(d("nplc")))
                    switches["auto_range"] = bool_switch("Auto-range", d("auto_range"))

                with param_card("Acquisition timing"):
                    inputs["settling_time_s"] = num_field("Settling time per current step (s)", float(d("settling_time_s")))
                    inputs["n_averages"] = num_field("Voltage samples averaged per point", float(d("n_averages")), integer=True)

                with param_card("Gate voltage (2400, optional)"):
                    switches["enable_gate"] = bool_switch("Enable gate (Keithley 2400)", d("enable_gate"))
                    inputs["gate_voltage_values"] = text_field(
                        "Gate voltage (V)", d("gate_voltage_values"),
                        hint="Single value, or comma-separated list — one complete current sweep "
                             "runs per value, each saved to its own file and plotted together.")

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))

            section_title("Instrument configuration")
            with stable_grid():
                with stable_card("Instrument addresses"):
                    inputs["source_visa_resource"] = text_field("Keithley 6221 (current source)", d("source_visa_resource"))
                    inputs["voltmeter_visa_resource"] = text_field("Keithley 2182 (DUT voltage)", d("voltmeter_visa_resource"))
                    inputs["gate_visa_resource"] = text_field("Keithley 2400 (gate) VISA resource", d("gate_visa_resource"))
                    inputs["temperature_visa_resource"] = text_field("MercuryiTC VISA resource", d("temperature_visa_resource"))

                with stable_card("Source & gate limits"):
                    inputs["source_delay_s"] = num_field("6221 source delay (s)", float(d("source_delay_s")))
                    inputs["gate_voltage_limit_V"] = num_field("Gate voltage software limit (V)", float(d("gate_voltage_limit_V")))
                    inputs["gate_compliance_current_A"] = num_field("Gate leakage compliance (A)", float(d("gate_compliance_current_A")))

                with stable_card("Temperature sensors"):
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
            if fid == "device":
                state[fid] = (identity.device_input.value or "").strip()
            elif fid == "cooldown":
                state[fid] = (identity.cooldown_input.value or "").strip()
            else:
                state[fid] = (inputs[fid].value or "").strip()
        for fid in OPTIONAL_NUMERIC_FIELDS:
            state[fid] = identity.temperature_input.value if fid == "temperature_setpoint_K" \
                else inputs[fid].value
        for fid, sw in switches.items():
            state[fid] = sw.value
        sample_value = identity.sample_dropdown.value
        state["sample"] = sample_value if sample_value not in (None, NEW_SAMPLE_SENTINEL) else ""
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
        raw["device"] = identity.device_input.value
        raw["cooldown"] = identity.cooldown_input.value
        raw["temperature_setpoint_K"] = identity.temperature_input.value
        raw["data_dir"] = identity.data_dir_input.value
        sample_value = identity.sample_dropdown.value
        if sample_value not in (None, NEW_SAMPLE_SENTINEL):
            raw["sample"] = sample_value
        return raw

    @ui.refreshable
    def refresh_summary() -> None:
        state, parse_errors = parse_state()
        dir_warning, dir_error = validate_directory(identity.data_dir_input.value or "")
        if parse_errors:
            info, warnings, errors = [], [], parse_errors
            preview = None
        else:
            info, warnings, errors = build_summary(state)
            preview = compute_filename_preview(state)
        identity.filename_label.set_text(
            f"File:  {preview}" if preview
            else "File:  (choose a sample and device to preview the filename)")
        if dir_warning:
            warnings = warnings + [dir_warning]
        if dir_error:
            errors = errors + [dir_error]
        with summary_box:
            summary_box.clear()
            render_summary([i for i in info if i], warnings, errors)
        start_btn.set_enabled(not errors and not is_busy())

    for inp in list(inputs.values()) + [
        identity.data_dir_input, identity.sample_dropdown, identity.device_input,
        identity.cooldown_input, identity.temperature_input,
    ]:
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

    async def _prompt_status_comment(plan: MeasurementPlan, run_contexts: list[RunContext],
                                      data_root: str, records: list[dict]) -> None:
        result = await status_comment_dialog()
        if result is None:
            return
        status, comment = result
        for series_idx, ctx in enumerate(run_contexts):
            iter_records = [r for r in records if r.get("series_index", 0) == series_idx]
            gate_V = iter_records[0].get("gate_voltage_V") if iter_records else None
            header_fields = build_header_fields(
                plan, ctx, iter_records, status=status, comment=comment,
                extra={"gate_voltage_V": gate_V} if gate_V is not None else None,
            )
            try:
                # Never truncate an already-written raw file to an empty stub —
                # only a run that never wrote a point gets a header-only write.
                if iter_records or not ctx.raw_path.exists():
                    write_record(ctx.raw_path, iter_records, header_fields)
                finalize_index_row(Path(data_root), ctx.sample, ctx.run_number, header_fields)
            except Exception:
                ui.notify("Could not save final status/comment.", type="negative")

    def make_on_finished(plan: MeasurementPlan, run_contexts: list[RunContext], data_root: str):
        def on_finished(final: FinalStatus, result) -> None:
            label = {"completed": "Measurement complete.", "aborted": "Measurement aborted.",
                      "error": f"ERROR: {final.error}"}[final.status]
            status_label.set_text(label)
            abort_btn.set_visibility(False)
            start_btn.set_enabled(not is_busy())
            refresh_summary.refresh()
            handle = controller["c"].handle if controller["c"] is not None else None
            records = list(handle.records) if handle is not None else []
            background_tasks.create(
                _prompt_status_comment(plan, run_contexts, data_root, records),
                name="status_comment_prompt",
            )
        return on_finished

    def make_run_fn(plan: MeasurementPlan, data_root: str, run_contexts: list[RunContext]):
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

                    key_axis = ("gate_V", gate_V) if gate_V is not None else None
                    ctx = allocate_run(
                        Path(data_root), plan.sample, plan.device, MEASUREMENT_TYPE,
                        temperature_setpoint_K=plan.temperature_setpoint_K,
                        key_axis=key_axis, series=plan.series,
                    )
                    run_contexts.append(ctx)
                    plan.acq_cfg.output_file = str(ctx.raw_path)
                    write_csv = make_incremental_writer(
                        ctx.raw_path,
                        lambda records, _ctx=ctx, _gv=gate_V: build_header_fields(
                            plan, _ctx, records, status="in_progress", comment="",
                            extra={"gate_voltage_V": _gv} if _gv is not None else None,
                        ),
                    )
                    points = [CurrentPoint(current_A=float(i)) for i in plan.currents_A]

                    iter_records: list[dict] = []

                    def tagged_on_point(record: dict, _idx=series_idx, _label=label,
                                         _iter=iter_records) -> None:
                        record["series_index"] = _idx
                        record["series_label"] = _label
                        _iter.append(record)
                        cb.on_point(record)

                    status = "Running measurement …" if gate_V is None else f"Running measurement (Vg={gate_V:g} V) …"
                    cb.on_status(status)
                    iter_error: Optional[BaseException] = None
                    try:
                        run_measurement(
                            source, voltmeter, plan.src_cfg, plan.acq_cfg, points,
                            stop_event=stop_event, on_point=tagged_on_point,
                            gate_voltage_V=gate_V, temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                            write_csv=write_csv,
                        )
                    except Exception as exc:
                        iter_error = exc

                    iter_status = "error" if iter_error is not None \
                        else ("aborted" if stop_event.is_set() else "completed")
                    header_fields = build_header_fields(
                        plan, ctx, iter_records, status=iter_status, comment="",
                        extra={"gate_voltage_V": gate_V} if gate_V is not None else None,
                    )
                    write_record(ctx.raw_path, iter_records, header_fields)
                    finalize_index_row(Path(data_root), ctx.sample, ctx.run_number, header_fields)

                    if iter_error is not None:
                        raise iter_error
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

    def _finish_artifacts(records: list[dict], run_contexts: list[RunContext], data_root: str) -> list[str]:
        output_paths = [str(c.raw_path) for c in run_contexts]
        if not run_contexts:
            return output_paths
        first, last = run_contexts[0], run_contexts[-1]
        run_label = first.run_str if first is last else f"{first.run_str}-{last.run_str}"
        png_path = proc_path(Path(data_root), first.sample, run_label, first.device,
                              MEASUREMENT_TYPE, "combined", combined=True)
        try:
            _save_combined_png(records, png_path)
            return output_paths + [str(png_path)]
        except Exception:
            return output_paths

    def on_start() -> None:
        state, parse_errors = parse_state()
        dir_warning, dir_error = validate_directory(identity.data_dir_input.value or "")
        if parse_errors or dir_error:
            ui.notify("Fix the blocking issues before starting.", type="negative")
            return
        _, _, errors = build_summary(state)
        if errors:
            ui.notify("Fix the blocking issues before starting.", type="negative")
            return
        state["data_dir"] = identity.data_dir_input.value.strip()

        _save_settings(collect_raw())

        plan = build_plan(state)
        Path(state["data_dir"]).mkdir(parents=True, exist_ok=True)
        labels = [series_label(v) for v in plan.series_values]
        run_contexts: list[RunContext] = []

        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE,
            run_fn=make_run_fn(plan, state["data_dir"], run_contexts),
            save_artifacts=lambda records, result, status: _finish_artifacts(
                records, run_contexts, state["data_dir"]),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[],
            on_record=on_record, on_status=on_status, on_log=on_log,
            on_finished=make_on_finished(plan, run_contexts, state["data_dir"]),
            sample=plan.sample, device=plan.device,
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

    def on_abort() -> None:
        if controller["c"] is not None:
            controller["c"].abort()

    start_btn.on_click(on_start)
    abort_btn.on_click(on_abort)
