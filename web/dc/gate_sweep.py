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

import logging
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from nicegui import ui

from dc.dc_sweep_utils import parse_value_list
import dc.dc_gate_sweep_tui as program
from dc.dc_gate_sweep_tui import (
    build_plan,
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, DC_GATE_SWEEP_DESCRIPTION, build_summary,
    compute_filename_preview,
)
from instruments.data_naming import (
    TEST_SAMPLE, RunContext,
)
from web.run_controller import (
    RunController, num_field, text_field,
    bool_switch, render_summary, busy_banner, is_busy,
    param_card, stable_card, param_grid, stable_grid, advanced_section, measurement_layout,
    program_artifacts, program_run_fn, refresh_on_busy_change,
    finished_handler, load_settings, save_settings,
)
from web.directory_picker import validate_directory
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "dc_gate_sweep_web_settings.json"

PAGE_TITLE = "DC Gate Sweep"
SUITE = "DC"

log = logging.getLogger("web.dc.gate_sweep")


def series_label(field_current_A: Optional[float], sense_current_A: float, n_sense: int) -> Optional[str]:
    parts = []
    if field_current_A is not None:
        parts.append(f"I_mag={field_current_A:g}A")
    if n_sense > 1:
        parts.append(f"I_sense={sense_current_A:g}A")
    return ", ".join(parts) or None


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client  # has slot context now; reused by the detached status/comment task
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(DC_GATE_SWEEP_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = load_settings(_SETTINGS_PATH)

    def d(key: str):
        if key in saved:
            return saved[key]
        return DEFAULTS.get(key, "")

    inputs: dict = {}
    switches: dict = {}
    controller: dict[str, Optional[RunController]] = {"c": None}

    with measurement_layout() as regions:
        with regions.identity:
            identity = identity_bar(
                default_data_dir=saved.get("data_dir") or str(_DATA_DIR),
                default_sample=saved.get("sample") or TEST_SAMPLE,
                default_device=d("device"), default_cooldown=d("cooldown"),
                default_temperature_K=(lambda t: float(t) if t not in ("", None) else None)(
                    d("temperature_setpoint_K")),
            )
        with regions.params:
            # ── Tier 1: what defines this run — always visible ───────────────
            with param_grid():
                with param_card("Gate voltage sweep (Keithley 2400)"):
                    inputs["gate_min_V"] = num_field("Sweep gate voltage min (V)", float(d("gate_min_V")))
                    inputs["gate_max_V"] = num_field("Sweep gate voltage max (V)", float(d("gate_max_V")))
                    inputs["step_V"] = num_field("Sweep step size (V)", float(d("step_V")))
                    switches["bidirectional_sweep"] = bool_switch(
                        "Bidirectional sweep (min → max → min)", d("bidirectional_sweep"))

                with param_card("Sense current (Keithley 6221)"):
                    inputs["sense_current_values"] = text_field(
                        "Sense current (A)", d("sense_current_values"),
                        hint="Single value, or comma-separated list — one complete gate sweep "
                             "runs per value, each saved to its own file and plotted together.")

                with param_card("Field (Kepco magnet, optional)"):
                    switches["enable_field"] = bool_switch("Park field (Kepco magnet)", d("enable_field"))
                    inputs["field_current_values"] = text_field(
                        "Magnet current (A)", d("field_current_values"),
                        hint="Single value, or comma-separated list — one complete gate sweep runs "
                             "per value, each saved to its own file and plotted together.")

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))

            # ── Tier 2: precision / speed knobs — collapsed ─────────────────
            with advanced_section("Acquisition & filter settings"):
                with stable_grid():
                    with param_card("Source & voltmeter"):
                        inputs["compliance_V"] = num_field("Compliance voltage (V)", float(d("compliance_V")))
                        inputs["nplc"] = num_field("NPLC (integration time)", float(d("nplc")))
                        switches["auto_range"] = bool_switch("Auto-range", d("auto_range"))

                    with param_card("Acquisition timing"):
                        inputs["settling_time_s"] = num_field("Settling time per gate step (s)", float(d("settling_time_s")))
                        inputs["n_averages"] = num_field("Voltage samples averaged per point", float(d("n_averages")), integer=True)

            # ── Tier 3: instrument wiring & safety — collapsed ──────────────
            with advanced_section("Instrument configuration & addresses", icon="settings"):
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
                        inputs["field_settle_tolerance_mT"] = num_field(
                            "Field-settle tolerance (mT)", float(d("field_settle_tolerance_mT")),
                            hint="Advanced: after parking the magnet, wait until a short window of "
                                 "gaussmeter readings spans less than this before the dwell above.")
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

            fig = go.Figure()
            fig.update_layout(xaxis_title="Gate voltage (V)", yaxis_title="Voltage (V)",
                               margin=dict(l=60, r=20, t=30, b=50), showlegend=True)
            with ui.element("div").classes("w-full").style("aspect-ratio: 1 / 1; min-height: 320px"):
                plot = ui.plotly(fig).classes("w-full h-full")

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
            if fid == "data_dir":
                state[fid] = (identity.data_dir_input.value or "").strip()
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
        state["sense_current_list"] = []
        state["sense_current_parse_error"] = None
        try:
            state["sense_current_list"] = parse_value_list(state["sense_current_values"])
        except ValueError as exc:
            state["sense_current_parse_error"] = str(exc)
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
    refresh_on_busy_change(refresh_summary.refresh)

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

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_run_label(text: str) -> None:
        run_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

    def make_on_finished(plan, run_contexts: list, run_extras: list):
        return finished_handler(
            page_client, controller, status_label, abort_btn, start_btn, refresh_summary.refresh,
            program, plan, run_contexts, run_extras,)

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

        save_settings(_SETTINGS_PATH, collect_raw())

        plan = build_plan(state, Path(state["data_dir"]))
        n_sense = len(plan.sense_currents_A)
        labels = [series_label(f, s, n_sense) for f, s in plan.series_values]
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
