#!/usr/bin/env python3
"""
NiceGUI page for dc_spin_valve.py
======================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Web equivalent of dc_spin_valve_tui.py. Reuses that TUI module's pure
DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/build_summary()/parse_sensor_uids().
Unlike I-V Curve / Gate Sweep, the magnet+gaussmeter are always required
(not an optional switch) and the gate-voltage list is always parsed (at
least one value, defaulting to "0.0") -- one complete field sweep runs per
gate value.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from nicegui import ui

import dc.dc_spin_valve_tui as program
from dc.dc_spin_valve_tui import (
    build_plan,
    DEFAULTS, DC_SPIN_VALVE_DESCRIPTION, build_summary,
    compute_filename_preview,
)
from instruments.data_naming import (
    TEST_SAMPLE, RunContext,
)
from web.identity_bar import identity_bar
from web.run_controller import (
    RunController, num_field, text_field, textarea_field,
    bool_switch, render_summary, busy_banner, is_busy,
    param_card, stable_card, param_grid, stable_grid, advanced_section, measurement_layout,
    program_artifacts, program_run_fn, refresh_on_busy_change,
    finished_handler, load_settings, save_settings,
    form_state,
)
from web.directory_picker import validate_directory
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "dc_spin_valve_web_settings.json"

PAGE_TITLE = "DC Spin-Valve / Field Sweep"
SUITE = "DC"

log = logging.getLogger("web.dc.spin_valve")


def series_label(I_sense: float, gate_V: Optional[float], n_currents: int, n_gates: int) -> Optional[str]:
    parts = []
    if n_currents > 1:
        parts.append(f"I={I_sense:g}A")
    if gate_V is not None and n_gates > 1:
        parts.append(f"Vg={gate_V:g}V")
    return ", ".join(parts) or None


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client  # has slot context now; reused by the detached status/comment task
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(DC_SPIN_VALVE_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

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
                default_temperature_K=(float(d("temperature_setpoint_K"))
                                        if d("temperature_setpoint_K") not in ("", None) else None),
            )
        with regions.params:
            # ── Tier 1: what defines this run — always visible ───────────────
            with param_grid():
                with param_card("Field sweep (Kepco magnet)"):
                    inputs["sweep_rows"] = textarea_field(
                        "Sweep rows: start, stop, points (one per line)",
                        d("sweep_rows"),
                        hint="Shared boundary points are merged.")
                    switches["bidirectional_sweep"] = bool_switch(
                        "Bidirectional (retrace the merged rows)", d("bidirectional_sweep"))

                with param_card("Sense current (Keithley 6221)"):
                    inputs["sense_current_values"] = text_field(
                        "Sense current (A)", d("sense_current_values"),
                        hint="Comma-separate for one sweep + file per value.")
                    switches["reversal_enabled"] = bool_switch(
                        "Reverse current each rep (+I/-I)", d("reversal_enabled"))
                    ui.label(
                        "Off for bias-direction-dependent devices: fixed +I, plain average."
                    ).classes("text-xs text-grey-6 -mt-1 mb-1")

                with param_card("Gate voltage (Keithley 2400, optional)"):
                    switches["enable_gate"] = bool_switch("Enable gate (Keithley 2400)", d("enable_gate"))
                    inputs["gate_voltage_values"] = text_field(
                        "Gate voltage (V)", d("gate_voltage_values"),
                        hint="Comma-separate for one sweep + file per value.")

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
                        inputs["settling_time_s"] = num_field(
                            "Settling time per point (s)", float(d("settling_time_s")),
                            hint="Wait after a field change.")
                        inputs["n_averages"] = num_field(
                            "Voltage averages per point", float(d("n_averages")), integer=True,
                            hint="± pairs with reversal, plain samples without.")

            # ── Tier 3: instrument wiring & safety — collapsed ──────────────
            with advanced_section("Instrument configuration & addresses", icon="settings"):
                with stable_grid():
                    with stable_card("Instrument addresses"):
                        inputs["source_visa_resource"] = text_field(
                            "Keithley 6221 (current source)", d("source_visa_resource"))
                        inputs["voltmeter_visa_resource"] = text_field(
                            "Keithley 2182 (voltage)", d("voltmeter_visa_resource"))
                        inputs["gate_visa_resource"] = text_field("Keithley 2400 (gate)", d("gate_visa_resource"))
                        inputs["magnet_visa_resource"] = text_field("Magnet VISA resource", d("magnet_visa_resource"))
                        inputs["gaussmeter_visa_resource"] = text_field(
                            "Gaussmeter VISA resource", d("gaussmeter_visa_resource"),
                            hint="Lake Shore 475.")
                        inputs["temperature_visa_resource"] = text_field(
                            "MercuryiTC VISA resource", d("temperature_visa_resource"))

                    with stable_card("Source & gate limits"):
                        inputs["source_delay_s"] = num_field(
                            "6221 source delay (s)", float(d("source_delay_s")),
                            hint="Also the settle after each ±I reversal.")
                        inputs["gate_voltage_limit_V"] = num_field(
                            "Gate voltage software limit (V)", float(d("gate_voltage_limit_V")))
                        inputs["gate_compliance_current_A"] = num_field(
                            "Gate leakage compliance (A)", float(d("gate_compliance_current_A")))

                    with stable_card("Magnet ramp safety"):
                        inputs["current_limit_A"] = num_field(
                            "Software current limit (A)", float(d("current_limit_A")),
                            hint="Hard safety ceiling.")
                        inputs["voltage_compliance_V"] = num_field(
                            "Voltage compliance (V)", float(d("voltage_compliance_V")))
                        inputs["ramp_step_A"] = num_field("Ramp step (A)", float(d("ramp_step_A")))
                        inputs["ramp_delay_s"] = num_field("Ramp delay (s)", float(d("ramp_delay_s")))

                    with stable_card("Gaussmeter & temperature sensors"):
                        inputs["gaussmeter_n_averages"] = num_field(
                            "Field readings averaged per point", float(d("gaussmeter_n_averages")), integer=True)
                        inputs["gaussmeter_read_delay_s"] = num_field(
                            "Delay between readings (s)", float(d("gaussmeter_read_delay_s")))
                        inputs["field_settle_tolerance_mT"] = num_field(
                            "Field-settle tolerance (mT)", float(d("field_settle_tolerance_mT")),
                            hint="Field settled when readings span less than this. Raise if points stall.")
                        inputs["temperature_sensor_uids"] = text_field(
                            "Sensor board UID(s)", d("temperature_sensor_uids"))

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
            fig.update_layout(xaxis_title="Magnetic field (mT)", yaxis_title="Voltage (V)",
                               margin=dict(l=60, r=20, t=30, b=50), showlegend=True)
            with ui.element("div").classes("w-full").style("aspect-ratio: 1 / 1; min-height: 320px"):
                plot = ui.plotly(fig).classes("w-full h-full")

            columns = [
                {"name": "n", "label": "#", "field": "n"},
                {"name": "Isense", "label": "I_sense (A)", "field": "Isense"},
                {"name": "Vg", "label": "Vg (V)", "field": "Vg"},
                {"name": "I", "label": "I_magnet (A)", "field": "I"},
                {"name": "B", "label": "B (mT)", "field": "B"},
                {"name": "V", "label": "V (V)", "field": "V"},
                {"name": "R", "label": "R (Ω)", "field": "R"},
                {"name": "T1", "label": "T1 (K)", "field": "T1"},
                {"name": "T2", "label": "T2 (K)", "field": "T2"},
            ]
            table = ui.table(columns=columns, rows=[], row_key="n").classes("w-full").props("dense")
            log_area = ui.log(max_lines=2000).classes("w-full h-48 font-mono text-xs")

    def parse_state() -> tuple[dict, list[str]]:
        return form_state(program, identity, inputs=inputs, switches=switches)

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
        if dir_warning:
            warnings = warnings + [dir_warning]
        if dir_error:
            errors = errors + [dir_error]
        identity.filename_label.set_text(
            f"File:  {preview}" if preview
            else "File:  (choose a sample and device to preview the filename)")
        with summary_box:
            summary_box.clear()
            render_summary([i for i in info if i], warnings, errors)
        start_btn.set_enabled(not errors and not is_busy())

    for inp in list(inputs.values()) + [
        identity.data_dir_input, identity.sample_dropdown,
        identity.device_input, identity.cooldown_input, identity.temperature_input,
    ]:
        inp.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    refresh_summary()
    refresh_on_busy_change(refresh_summary.refresh)

    series_state: dict = {}

    _SWEEP_UP_COLOR = "#2E3192"
    _SWEEP_DOWN_COLOR = "#e34948"
    _SERIES_DASH = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    def init_series(n_series: int, labels: list[str], bidirectional: bool) -> None:
        fig.data = []
        series_state["traces"] = {}
        series_state["last_I"] = {}
        series_state["direction"] = {}
        for i in range(n_series):
            dash = _SERIES_DASH[i % len(_SERIES_DASH)]
            up_name = "Sweep up" if n_series == 1 else f"{labels[i]} (up)"
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name=up_name,
                                      line=dict(color=_SWEEP_UP_COLOR, dash=dash),
                                      marker=dict(color=_SWEEP_UP_COLOR)))
            up_ti = len(fig.data) - 1
            down_ti = None
            if bidirectional:
                down_name = "Sweep down" if n_series == 1 else f"{labels[i]} (down)"
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name=down_name,
                                          line=dict(color=_SWEEP_DOWN_COLOR, dash=dash),
                                          marker=dict(color=_SWEEP_DOWN_COLOR)))
                down_ti = len(fig.data) - 1
            series_state["traces"][i] = (up_ti, down_ti)
            series_state["last_I"][i] = None
            series_state["direction"][i] = "up"

    def on_record(record: dict) -> None:
        idx = record.get("series_index", 0)
        up_ti, down_ti = series_state["traces"][idx]

        I = record.get("magnet_current_A")
        last_I = series_state["last_I"][idx]
        if I is not None and last_I is not None and I != last_I:
            series_state["direction"][idx] = "up" if I > last_I else "down"
        if I is not None:
            series_state["last_I"][idx] = I
        ti = up_ti if down_ti is None or series_state["direction"][idx] == "up" else down_ti

        has_field = record.get("magnet_field_mT") is not None
        x = record["magnet_field_mT"] if has_field else record["point_index"]
        fig.data[ti].x = fig.data[ti].x + (x,)
        fig.data[ti].y = fig.data[ti].y + (record["voltage_V"],)
        table.rows.append({
            "n": record["point_index"] + 1,
            "Isense": f"{record['sense_current_A']:.4g}" if record.get("sense_current_A") is not None else "—",
            "Vg": f"{record['gate_voltage_V']:.4g}" if record.get("gate_voltage_V") is not None else "—",
            "I": f"{record['magnet_current_A']:.4f}" if record.get("magnet_current_A") is not None else "—",
            "B": f"{record['magnet_field_mT']:.2f}" if record.get("magnet_field_mT") is not None else "—",
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
        n_currents = len(plan.sense_currents_A)
        n_gates = len(plan.gate_series_values)
        labels = [series_label(I, gv, n_currents, n_gates) for I, gv in plan.series_values]
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

        init_series(len(plan.series_values), labels, state["bidirectional_sweep"])
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
