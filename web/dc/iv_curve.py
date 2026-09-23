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
Start click, one CSV and one PNG per value.

Live view adds a second panel (dV/dI via np.gradient, recomputed on each
drain tick) alongside the raw I-V trace.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import background_tasks, ui

from dc.dc_sweep_utils import parse_value_list
import dc.dc_iv_curve_tui as program
from dc.dc_iv_curve_tui import (
    build_plan,
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, OPTIONAL_NUMERIC_FIELDS, DC_IV_DESCRIPTION, MeasurementPlan, build_summary,
    compute_filename_preview,
)
from instruments.data_naming import (
    TEST_SAMPLE, RunContext,
)
from web.run_controller import (
    RunController, FinalStatus, num_field, text_field,
    bool_switch, render_summary, busy_banner, is_busy,
    param_card, stable_card, param_grid, stable_grid, advanced_section, measurement_layout,
    program_artifacts, program_run_fn, prompt_last_run,
    refresh_on_busy_change,
)
from web.directory_picker import validate_directory
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "dc_iv_curve_web_settings.json"

PAGE_TITLE = "DC I-V Curve"
SUITE = "DC"

log = logging.getLogger("web.dc.iv_curve")


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


def series_label(gate_V: Optional[float]) -> Optional[str]:
    return f"Vg={gate_V:g}V" if gate_V is not None else None


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client  # has slot context now; reused by the detached status/comment task
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

    with measurement_layout() as regions:
        with regions.identity:
            identity = identity_bar(
                default_data_dir=saved.get("data_dir") or str(_DATA_DIR),
                default_sample=saved.get("sample") or TEST_SAMPLE,
                default_device=d("device"), default_cooldown=d("cooldown"),
                default_temperature_K=float(_t_default) if _t_default not in ("", None) else None,
            )
        with regions.params:
            # ── Tier 1: what defines this run — always visible ───────────────
            with param_grid():
                with param_card("Current sweep (Keithley 6221)"):
                    inputs["current_min_A"] = num_field("Sweep current min (A)", float(d("current_min_A")))
                    inputs["current_max_A"] = num_field("Sweep current max (A)", float(d("current_max_A")))
                    inputs["step_A"] = num_field("Sweep step size (A)", float(d("step_A")))
                    switches["bidirectional_sweep"] = bool_switch(
                        "Bidirectional sweep (min → max → min)", d("bidirectional_sweep"))

                with param_card("Gate voltage (Keithley 2400, optional)"):
                    switches["enable_gate"] = bool_switch("Enable gate (Keithley 2400)", d("enable_gate"))
                    inputs["gate_voltage_values"] = text_field(
                        "Gate voltage (V)", d("gate_voltage_values"),
                        hint="Single value, or comma-separated list — one complete current sweep "
                             "runs per value, each saved to its own file and plotted together.")

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))

            # ── Tier 2: precision / speed knobs — collapsed ─────────────────
            with advanced_section("Acquisition & filter settings"):
                with stable_grid():
                    with param_card("Source & voltmeter"):
                        inputs["compliance_V"] = num_field(
                            "Compliance voltage (V)", float(d("compliance_V")),
                            hint="Set high enough to reach the expected voltage at current_max_A.")
                        inputs["nplc"] = num_field("NPLC (integration time)", float(d("nplc")))
                        switches["auto_range"] = bool_switch("Auto-range", d("auto_range"))

                    with param_card("Acquisition timing"):
                        inputs["settling_time_s"] = num_field("Settling time per current step (s)", float(d("settling_time_s")))
                        inputs["n_averages"] = num_field("Voltage samples averaged per point", float(d("n_averages")), integer=True)

            # ── Tier 3: instrument wiring & safety — collapsed ──────────────
            with advanced_section("Instrument configuration & addresses", icon="settings"):
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

        with regions.summary:
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start measurement", color="primary").classes("w-full")

        with regions.output:
            with ui.row().classes("w-full items-center gap-3"):
                run_label = ui.label("").classes("text-sm font-bold text-grey-6")
                status_label = ui.label("Idle.").classes("text-sm font-bold")
            abort_btn = ui.button("Abort (safe ramp-down)", color="negative").props("outline")
            abort_btn.set_visibility(False)

            fig = make_subplots(rows=2, cols=1, subplot_titles=("I-V curve", "Differential resistance (dV/dI)"))
            fig.update_xaxes(title_text="Current (A)", row=1, col=1)
            fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
            fig.update_xaxes(title_text="Current (A)", row=2, col=1)
            fig.update_yaxes(title_text="dV/dI (Ω)", row=2, col=1)
            fig.update_layout(margin=dict(l=60, r=20, t=40, b=50), showlegend=True)
            with ui.element("div").classes("w-full").style("aspect-ratio: 1 / 2; max-height: 90vh"):
                plot = ui.plotly(fig).classes("w-full h-full")

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
            elif fid == "data_dir":
                state[fid] = (identity.data_dir_input.value or "").strip()
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
    refresh_on_busy_change(refresh_summary.refresh)

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

        table.rows.append({
            "n": record["point_index"] + 1,
            "Vg": f"{record['gate_voltage_V']:.4g}" if record.get("gate_voltage_V") is not None else "—",
            "I": f"{record['current_A']:.4e}",
            "V": f"{record['voltage_V']:.4e}",
            "R": f"{record['resistance_ohm']:.5g}",
            "T1": f"{record['temperature_1_K']:.3f}" if record.get("temperature_1_K") is not None else "—",
            "T2": f"{record['temperature_2_K']:.3f}" if record.get("temperature_2_K") is not None else "—",
        })

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_run_label(text: str) -> None:
        run_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

    def make_on_finished(plan: MeasurementPlan, run_contexts: list[RunContext], run_extras: list):
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
                prompt_last_run(page_client, program, plan, run_contexts, run_extras, records),
                name="status_comment_prompt",
            )
        return on_finished

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
        state["data_dir"] = prepare_data_root(identity.data_dir_input.value, state["sample"])

        _save_settings(collect_raw())

        plan = build_plan(state, Path(state["data_dir"]))
        labels = [series_label(v) for v in plan.series_values]
        run_contexts: list[RunContext] = []
        run_extras: list[dict] = []

        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE,
            run_fn=program_run_fn(program, plan, run_contexts, run_extras),
            save_artifacts=lambda records, result, status: program_artifacts(program, plan, run_contexts),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[],
            on_tick=lambda: (plot.update(), table.update()),
            on_record=on_record, on_status=on_status, on_run_label=on_run_label, on_log=on_log,
            on_finished=make_on_finished(plan, run_contexts, run_extras),
            sample=plan.sample, device=plan.device, run_cost=plan.run_cost,
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
