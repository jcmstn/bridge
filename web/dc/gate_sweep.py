#!/usr/bin/env python3
"""
NiceGUI page for dc_gate_sweep.py
======================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Web equivalent of dc_gate_sweep_tui.py. Reuses that TUI module's pure
DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/build_summary()/parse_sensor_uids().
Optional magnet-current list means multiple complete gate sweeps run per
Start click (field parked once per value, not swept).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from nicegui import background_tasks, ui

from dc.dc_gate_sweep import (
    AcquisitionConfig, GateConfig, GatePoint, GaussmeterConfig, MagnetConfig,
    SourceConfig, TemperatureControllerConfig, VoltmeterConfig,
    connect_gate, connect_gaussmeter, connect_magnet, connect_source,
    connect_temperature_controller, connect_voltmeter, read_field_mT, run_measurement,
    set_magnet_current, shutdown_gate, shutdown_gaussmeter, shutdown_magnet,
    shutdown_source, shutdown_temperature_controller,
)
from dc.dc_sweep_utils import linear_sweep, parse_value_list
from dc.dc_gate_sweep_tui import (
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, OPTIONAL_NUMERIC_FIELDS, MEASUREMENT_TYPE,
    DC_GATE_SWEEP_DESCRIPTION, MeasurementPlan, build_header_fields, build_summary,
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
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "dc_gate_sweep_web_settings.json"

PAGE_TITLE = "DC Gate Sweep"
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
        visa_resource=state["source_visa_resource"], sense_current_A=state["sense_current_A"],
        compliance_V=state["compliance_V"], source_delay_s=state["source_delay_s"],
    )
    volt_cfg = VoltmeterConfig(
        visa_resource=state["voltmeter_visa_resource"], nplc=state["nplc"], auto_range=state["auto_range"])
    gate_cfg = GateConfig(
        visa_resource=state["gate_visa_resource"], gate_voltage_limit_V=state["gate_voltage_limit_V"],
        compliance_current_A=state["gate_compliance_current_A"],
    )
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"], n_averages=int(state["n_averages"]), output_file="")

    gate_voltages_V = linear_sweep(
        start=state["gate_min_V"], stop=state["gate_max_V"], step=state["step_V"],
        bidirectional=state["bidirectional_sweep"],
    )

    magnet_cfg = None
    gauss_cfg = None
    field_currents_A = None
    if state["enable_field"]:
        magnet_cfg = MagnetConfig(
            visa_resource=state["magnet_visa_resource"], current_limit_A=state["current_limit_A"],
            voltage_compliance_V=state["voltage_compliance_V"], ramp_step_A=state["ramp_step_A"],
            ramp_delay_s=state["ramp_delay_s"],
        )
        gauss_cfg = GaussmeterConfig(
            visa_resource=state["gaussmeter_visa_resource"], n_averages=int(state["gaussmeter_n_averages"]),
            read_delay_s=state["gaussmeter_read_delay_s"],
        )
        field_currents_A = state["field_current_list"]

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

    header_extra = {
        "sense_current_A": state["sense_current_A"],
        "compliance_V": state["compliance_V"],
        "n_averages": int(state["n_averages"]),
        "settling_time_s": state["settling_time_s"],
        "gate_sweep_V": [state["gate_min_V"], state["gate_max_V"], state["step_V"]],
    }
    series = ""
    if len(field_currents_A or []) > 1:
        series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                  f"{datetime.now():%Y%m%dT%H%M%S}")

    return MeasurementPlan(
        src_cfg=src_cfg, volt_cfg=volt_cfg, gate_cfg=gate_cfg, acq_cfg=acq_cfg,
        gate_voltages_V=gate_voltages_V,
        sample=state["sample"], device=state["device"],
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series=series,
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg,
        field_currents_A=field_currents_A, field_settle_s=state["field_settle_s"], temp_cfg=temp_cfg,
    )


def series_label(current_A: Optional[float]) -> Optional[str]:
    return f"I_mag={current_A:g}A" if current_A is not None else None


def _save_combined_png(records: list[dict], png_path: Path) -> None:
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5))
    series_ids = sorted({r.get("series_index", 0) for r in records})
    for idx in series_ids:
        rows = [r for r in records if r.get("series_index", 0) == idx]
        label = rows[0].get("series_label")
        ax.plot([r["gate_voltage_V"] for r in rows], [r["voltage_V"] for r in rows],
                ".-", color=cmap(idx % 10), label=label)
    ax.set_xlabel("Gate voltage (V)"); ax.set_ylabel("Voltage (V)")
    ax.set_title("Measurement result"); ax.grid(alpha=0.3)
    if any(r.get("series_label") for r in records):
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def page() -> None:
    ui.page_title(PAGE_TITLE)
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(DC_GATE_SWEEP_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = _load_settings()

    def d(key: str):
        if key in saved:
            return saved[key]
        return DEFAULTS.get(key, "")

    inputs: dict = {}
    switches: dict = {}
    controller: dict[str, Optional[RunController]] = {"c": None}

    identity = identity_bar(
        default_data_dir=saved.get("data_dir") or str(_DATA_DIR),
        default_sample=saved.get("sample") or TEST_SAMPLE,
        default_device=d("device"), default_cooldown=d("cooldown"),
        default_temperature_K=(lambda t: float(t) if t not in ("", None) else None)(
            d("temperature_setpoint_K")),
    )

    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        with ui.column().classes("flex-grow gap-1 max-w-4xl"):
            with param_grid():
                with param_card("Source (6221)"):
                    inputs["sense_current_A"] = num_field("Fixed sense current (A)", float(d("sense_current_A")))
                    inputs["compliance_V"] = num_field("Compliance voltage (V)", float(d("compliance_V")))

                with param_card("Voltmeter (2182)"):
                    inputs["nplc"] = num_field("NPLC (integration time)", float(d("nplc")))
                    switches["auto_range"] = bool_switch("Auto-range", d("auto_range"))

                with param_card("Acquisition timing"):
                    inputs["settling_time_s"] = num_field("Settling time per gate step (s)", float(d("settling_time_s")))
                    inputs["n_averages"] = num_field("Voltage samples averaged per point", float(d("n_averages")), integer=True)

                with param_card("Gate voltage sweep (2400)"):
                    inputs["gate_min_V"] = num_field("Sweep gate voltage min (V)", float(d("gate_min_V")))
                    inputs["gate_max_V"] = num_field("Sweep gate voltage max (V)", float(d("gate_max_V")))
                    inputs["step_V"] = num_field("Sweep step size (V)", float(d("step_V")))
                    switches["bidirectional_sweep"] = bool_switch(
                        "Bidirectional sweep (min → max → min)", d("bidirectional_sweep"))

                with param_card("Field (Kepco magnet, optional)"):
                    switches["enable_field"] = bool_switch("Park field (Kepco magnet)", d("enable_field"))
                    inputs["field_current_values"] = text_field(
                        "Magnet current (A)", d("field_current_values"),
                        hint="Single value, or comma-separated list — one complete gate sweep runs "
                             "per value, each saved to its own file and plotted together.")

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))

            section_title("Instrument configuration")
            with stable_grid():
                with stable_card("Instrument addresses"):
                    inputs["source_visa_resource"] = text_field("Keithley 6221 (sense current)", d("source_visa_resource"))
                    inputs["voltmeter_visa_resource"] = text_field("Keithley 2182 (DUT voltage)", d("voltmeter_visa_resource"))
                    inputs["gate_visa_resource"] = text_field("Keithley 2400 (gate)", d("gate_visa_resource"))
                    inputs["magnet_visa_resource"] = text_field("Magnet VISA resource", d("magnet_visa_resource"))
                    inputs["gaussmeter_visa_resource"] = text_field(
                        "Gaussmeter VISA resource", d("gaussmeter_visa_resource"),
                        hint="Lake Shore 475 — measures the actual field once parked.")
                    inputs["temperature_visa_resource"] = text_field("MercuryiTC VISA resource", d("temperature_visa_resource"))

                with stable_card("Source & gate limits"):
                    inputs["source_delay_s"] = num_field("6221 source delay (s)", float(d("source_delay_s")))
                    inputs["gate_voltage_limit_V"] = num_field(
                        "Gate voltage software limit (V)", float(d("gate_voltage_limit_V")),
                        hint="Hard safety ceiling — independent of the sweep range above.")
                    inputs["gate_compliance_current_A"] = num_field("Gate leakage compliance (A)", float(d("gate_compliance_current_A")))

                with stable_card("Magnet ramp safety"):
                    inputs["current_limit_A"] = num_field("Software current limit (A)", float(d("current_limit_A")))
                    inputs["voltage_compliance_V"] = num_field("Voltage compliance (V)", float(d("voltage_compliance_V")))
                    inputs["ramp_step_A"] = num_field("Ramp step (A)", float(d("ramp_step_A")))
                    inputs["ramp_delay_s"] = num_field("Ramp delay (s)", float(d("ramp_delay_s")))
                    inputs["field_settle_s"] = num_field("Settling time after parking field (s)", float(d("field_settle_s")))

                with stable_card("Gaussmeter & temperature sensors"):
                    inputs["gaussmeter_n_averages"] = num_field("Field readings averaged", float(d("gaussmeter_n_averages")), integer=True)
                    inputs["gaussmeter_read_delay_s"] = num_field("Delay between readings (s)", float(d("gaussmeter_read_delay_s")))
                    inputs["temperature_sensor_uids"] = text_field("Sensor board UID(s)", d("temperature_sensor_uids"))

        with ui.column().classes("w-96 gap-2"):
            ui.label("Summary").classes("text-lg font-bold")
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start measurement", color="primary").classes("w-full")

    ui.separator().classes("my-3")
    status_label = ui.label("Idle.").classes("text-sm font-bold")
    abort_btn = ui.button("Abort (safe ramp-down)", color="negative").props("outline")
    abort_btn.set_visibility(False)

    fig = go.Figure()
    fig.update_layout(xaxis_title="Gate voltage (V)", yaxis_title="Voltage (V)",
                       margin=dict(l=60, r=20, t=30, b=50), height=420, showlegend=True)
    plot = ui.plotly(fig).classes("w-full")

    columns = [
        {"name": "n", "label": "#", "field": "n"},
        {"name": "Imag", "label": "I_mag (A)", "field": "Imag"},
        {"name": "B", "label": "B (mT)", "field": "B"},
        {"name": "Vg", "label": "Vg (V)", "field": "Vg"},
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
            if fid in ("device", "cooldown"):
                continue
            state[fid] = (inputs[fid].value or "").strip()
        state["device"] = (identity.device_input.value or "").strip()
        state["cooldown"] = (identity.cooldown_input.value or "").strip()
        state["temperature_setpoint_K"] = identity.temperature_input.value
        for fid, sw in switches.items():
            state[fid] = sw.value
        sample_value = identity.sample_dropdown.value
        state["sample"] = sample_value if sample_value not in (None, NEW_SAMPLE_SENTINEL) else ""
        state["field_current_list"] = []
        state["field_parse_error"] = None
        if state["enable_field"]:
            try:
                state["field_current_list"] = parse_value_list(state["field_current_values"])
            except ValueError as exc:
                state["field_parse_error"] = str(exc)
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

    for inp in list(inputs.values()):
        inp.on_value_change(refresh_summary.refresh)
    for handle in (identity.data_dir_input, identity.sample_dropdown,
                   identity.device_input, identity.cooldown_input, identity.temperature_input):
        handle.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    refresh_summary()
    ui.timer(2.0, refresh_summary.refresh)

    series_state: dict = {}

    def init_series(n_series: int, labels: list[Optional[str]]) -> None:
        fig.data = []
        series_state["traces"] = {}
        cmap = ["#2E3192", "#e34948", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#ff7f0e", "#7f7f7f"]
        for i in range(n_series):
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name=labels[i] or "Vg sweep",
                                      line=dict(color=cmap[i % len(cmap)])))
            series_state["traces"][i] = i

    def on_record(record: dict) -> None:
        idx = record.get("series_index", 0)
        ti = series_state["traces"][idx]
        fig.data[ti].x = fig.data[ti].x + (record["gate_voltage_V"],)
        fig.data[ti].y = fig.data[ti].y + (record["voltage_V"],)
        plot.update()
        table.rows.append({
            "n": record["point_index"] + 1,
            "Imag": f"{record['magnet_current_A']:.4f}" if record.get("magnet_current_A") is not None else "—",
            "B": f"{record['magnet_field_mT']:.2f}" if record.get("magnet_field_mT") is not None else "—",
            "Vg": f"{record['gate_voltage_V']:.4g}",
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
            current_A = iter_records[0].get("magnet_current_A") if iter_records else None
            field_mT = iter_records[0].get("magnet_field_mT") if iter_records else None
            header_fields = build_header_fields(
                plan, ctx, iter_records, status=status, comment=comment,
                extra={"magnet_current_A": current_A, "magnet_field_mT": field_mT}
                if current_A is not None else None,
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
            source = voltmeter = gate = magnet = gaussmeter = temp_ctrl = None
            try:
                cb.on_status("Connecting to Keithley 6221, 2182 & 2400 …")
                source = connect_source(plan.src_cfg)
                voltmeter = connect_voltmeter(plan.volt_cfg)
                gate = connect_gate(plan.gate_cfg)

                if plan.temp_cfg is not None:
                    cb.on_status("Connecting to MercuryiTC (temperature) …")
                    temp_ctrl = connect_temperature_controller(plan.temp_cfg)

                if plan.magnet_cfg is not None:
                    cb.on_status("Connecting magnet power supply …")
                    magnet = connect_magnet(plan.magnet_cfg)
                    cb.on_status("Connecting gaussmeter …")
                    gaussmeter = connect_gaussmeter(plan.gauss_cfg)

                for series_idx, current_A in enumerate(plan.series_values):
                    if stop_event.is_set():
                        break
                    label = series_label(current_A)
                    field_mT = None
                    if current_A is not None:
                        cb.on_status(f"Parking magnet at {current_A:g} A …")
                        set_magnet_current(magnet, plan.magnet_cfg, current_A)
                        time.sleep(plan.field_settle_s)
                        field_mT = read_field_mT(gaussmeter, plan.gauss_cfg)

                    key_axis = ("current_A", current_A) if current_A is not None else None
                    ctx = allocate_run(
                        Path(data_root), plan.sample, plan.device, MEASUREMENT_TYPE,
                        temperature_setpoint_K=plan.temperature_setpoint_K,
                        key_axis=key_axis, series=plan.series,
                    )
                    run_contexts.append(ctx)
                    plan.acq_cfg.output_file = str(ctx.raw_path)
                    write_csv = make_incremental_writer(
                        ctx.raw_path,
                        lambda records, _ctx=ctx, _i=current_A, _b=field_mT: build_header_fields(
                            plan, _ctx, records, status="in_progress", comment="",
                            extra={"magnet_current_A": _i, "magnet_field_mT": _b} if _i is not None else None,
                        ),
                    )
                    points = [GatePoint(gate_voltage_V=float(v)) for v in plan.gate_voltages_V]

                    iter_records: list[dict] = []

                    def tagged_on_point(record: dict, _idx=series_idx, _label=label,
                                         _iter=iter_records) -> None:
                        record["series_index"] = _idx
                        record["series_label"] = _label
                        _iter.append(record)
                        cb.on_point(record)

                    status = "Running gate sweep …" if current_A is None else f"Running gate sweep (I_mag={current_A:g} A) …"
                    cb.on_status(status)
                    iter_error: Optional[BaseException] = None
                    try:
                        run_measurement(
                            source, voltmeter, gate, plan.src_cfg, plan.gate_cfg, plan.acq_cfg, points,
                            stop_event=stop_event, on_point=tagged_on_point,
                            magnet_current_A=current_A, magnet_field_mT=field_mT,
                            temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                            write_csv=write_csv,
                        )
                    except Exception as exc:
                        iter_error = exc

                    iter_status = "error" if iter_error is not None \
                        else ("aborted" if stop_event.is_set() else "completed")
                    header_fields = build_header_fields(
                        plan, ctx, iter_records, status=iter_status, comment="",
                        extra={"magnet_current_A": current_A, "magnet_field_mT": field_mT}
                        if current_A is not None else None,
                    )
                    write_record(ctx.raw_path, iter_records, header_fields)
                    finalize_index_row(Path(data_root), ctx.sample, ctx.run_number, header_fields)

                    if iter_error is not None:
                        raise iter_error
                return None
            finally:
                if magnet is not None:
                    shutdown_magnet(magnet, plan.magnet_cfg)
                if gaussmeter is not None:
                    shutdown_gaussmeter(gaussmeter)
                if temp_ctrl is not None:
                    shutdown_temperature_controller(temp_ctrl)
                if gate is not None:
                    shutdown_gate(gate)
                if source is not None:
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
