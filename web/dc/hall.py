#!/usr/bin/env python3
"""
NiceGUI page for dc_hall_measurement.py
===========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Web equivalent of dc_hall_measurement_tui.py. Reuses that TUI module's pure
(Textual-independent) DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/build_summary() so
validation stays identical between the TUI and the web page — only the
layout (NiceGUI instead of Textual widgets) and plan-building (a free-choice
`data_dir` in place of the TUI's fixed _DATA_DIR) are reimplemented here.

The sense current (single value, or a comma-separated list) runs one
complete measurement per value -- a single point, or a full field sweep if
enabled -- each saved to its own file and plotted together in a different
color, mirroring web/dc/spin_valve.py's gate-voltage series.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from nicegui import background_tasks, ui

from dc.dc_sweep_utils import parse_sweep_rows, parse_value_list
import dc.dc_hall_measurement_tui as program
from dc.dc_hall_measurement_tui import (
    build_plan, DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, OPTIONAL_NUMERIC_FIELDS, DC_HALL_DESCRIPTION,
    MeasurementPlan, build_summary,
    compute_filename_preview, QUANTITY_PLOT_LABELS, active_quantities,
)
from instruments.data_naming import (
    TEST_SAMPLE, RunContext,
)
from web.run_controller import (
    RunController, FinalStatus, num_field, optional_num_field, text_field,
    textarea_field, bool_switch, render_summary, busy_banner, is_busy,
    param_card, stable_card, param_grid, stable_grid, advanced_section, measurement_layout,
    program_artifacts, program_run_fn, prompt_last_run,
    refresh_on_busy_change,
)
from web.directory_picker import validate_directory
from web.field_diagram import build_field_diagram_figure
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "dc_hall_web_settings.json"

PAGE_TITLE = "DC Hall Measurement"
SUITE = "DC"

log = logging.getLogger("web.dc.hall")


# ─────────────────────────────────────────────────────────────────────────────
# Settings persistence  ── own file, separate from the TUI's own settings JSON
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client  # has slot context now; reused by the detached status/comment task
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(DC_HALL_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = _load_settings()

    def d(key: str):
        # DEFAULTS comes straight from dc_hall_measurement_tui and has no
        # "data_dir" entry (that field is new here) — fall back to "" for
        # any key neither `saved` nor the TUI's DEFAULTS knows about.
        if key in saved:
            return saved[key]
        return DEFAULTS.get(key, "")

    inputs: dict[str, ui.number | ui.input | ui.textarea] = {}
    switches: dict[str, ui.switch] = {}
    controller: dict[str, Optional[RunController]] = {"c": None}

    _t_default = d("temperature_setpoint_K")
    _theta_default = d("field_theta_deg")
    _phi_default = d("field_phi_deg")

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
                with param_card("Sense current (Keithley 6221)"):
                    inputs["sense_current_values"] = text_field(
                        "Sense current (A)", d("sense_current_values"),
                        hint="Reversed +I/-I each rep to cancel thermal-EMF offsets. "
                             "Single value, or comma-separated list — one complete "
                             "measurement runs per value, each saved to its own file.")

                with param_card("Quantities (Keithley 2182)"):
                    switches["measure_rxy"] = bool_switch(
                        "R_xy (transverse/Hall) — ch1", d("measure_rxy"))
                    switches["measure_rxx"] = bool_switch(
                        "R_xx (longitudinal) — ch2 if both on, else ch1", d("measure_rxx"))

                with param_card("Field sweep (Kepco magnet)"):
                    switches["enable_sweep"] = bool_switch(
                        "Sweep magnetic field (Kepco magnet)", d("enable_sweep"))
                    inputs["sweep_rows"] = textarea_field(
                        "Sweep rows: start, stop, points (one per line)",
                        d("sweep_rows"),
                        hint="Adjacent rows sharing a boundary value are merged, not duplicated.")
                    switches["bidirectional_sweep"] = bool_switch(
                        "Bidirectional (retrace the merged rows)", d("bidirectional_sweep"))

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))

                with param_card("Field direction"):
                    inputs["field_theta_deg"] = optional_num_field(
                        "θ — tilt from out-of-plane (°)",
                        float(_theta_default) if str(_theta_default).strip() not in ("", "None") else None,
                        hint="0° = fully out-of-plane (film normal), 90° = in-plane.",
                        min=0, max=180)
                    inputs["field_phi_deg"] = optional_num_field(
                        "φ — azimuth from current axis (°)",
                        float(_phi_default) if str(_phi_default).strip() not in ("", "None") else None,
                        hint="0° = along sense current, 90° = transverse in-plane. "
                             "Meaningless when θ=0°.",
                        min=0, max=360)
                    with ui.row().classes("gap-2 mb-1"):
                        ui.button("xy", on_click=lambda: (inputs["field_theta_deg"].set_value(90),
                                                            refresh_summary.refresh())).props("dense outline")
                        ui.button("zx", on_click=lambda: (inputs["field_phi_deg"].set_value(0),
                                                            refresh_summary.refresh())).props("dense outline")
                        ui.button("zy", on_click=lambda: (inputs["field_phi_deg"].set_value(90),
                                                            refresh_summary.refresh())).props("dense outline")
                    field_diagram_plot = ui.plotly(build_field_diagram_figure(
                        _theta_default if str(_theta_default).strip() not in ("", "None") else None,
                        _phi_default if str(_phi_default).strip() not in ("", "None") else None,
                    )).classes("w-full").style("height: 220px")

            # ── Tier 2: precision / speed knobs — collapsed ─────────────────
            with advanced_section("Acquisition & filter settings"):
                with stable_grid():
                    with param_card("Source & voltmeter"):
                        inputs["compliance_V"] = num_field(
                            "Compliance voltage (V)", float(d("compliance_V")))
                        inputs["nplc"] = num_field(
                            "NPLC (integration time)", float(d("nplc")),
                            hint="Bigger = quieter but slower. 1 line cycle = 1/50 or 1/60 s.")
                        switches["auto_range"] = bool_switch("Auto-range", d("auto_range"))

                    with param_card("Acquisition timing"):
                        inputs["settling_time_s"] = num_field(
                            "Settling time per point (s)", float(d("settling_time_s")),
                            hint="Dead-time after a field change, before acquiring.")
                        inputs["n_reversals"] = num_field(
                            "+I/-I reversal pairs averaged per point", float(d("n_reversals")), integer=True,
                            hint="Splits each point into (V(+I)-V(-I))/2 [reported R] and "
                                 "(V(+I)+V(-I))/2 [recorded, not discarded].")
                        inputs["channel_settle_s"] = num_field(
                            "2182 channel-mux settle (s)", float(d("channel_settle_s")),
                            hint="Only used when both R_xy and R_xx are on — dead time after "
                                 "switching the 2182's active channel, before reading.")

            # ── Tier 3: instrument wiring & safety — collapsed ──────────────
            with advanced_section("Instrument configuration & addresses", icon="settings"):
                with stable_grid():
                    with stable_card("Instrument addresses"):
                        inputs["source_visa_resource"] = text_field(
                            "Keithley 6221 (current source)", d("source_visa_resource"))
                        inputs["voltmeter_visa_resource"] = text_field(
                            "Keithley 2182 (R_xy / R_xx voltage)", d("voltmeter_visa_resource"))
                        inputs["magnet_visa_resource"] = text_field(
                            "Magnet VISA resource", d("magnet_visa_resource"))
                        inputs["gaussmeter_visa_resource"] = text_field(
                            "Gaussmeter VISA resource", d("gaussmeter_visa_resource"),
                            hint="Lake Shore 475 — measures the actual field at each point.")
                        inputs["temperature_visa_resource"] = text_field(
                            "MercuryiTC VISA resource", d("temperature_visa_resource"),
                            hint="e.g. TCPIP0::<ip>::7020::SOCKET, or an ASRL resource.")

                    with stable_card("Source & ramp safety"):
                        inputs["source_delay_s"] = num_field(
                            "6221 source delay (s)", float(d("source_delay_s")),
                            hint="Also the settle time between a current reversal and reading "
                                 "the voltmeter, so the reversal has actually finished before "
                                 "the 2182 integrates.")
                        inputs["current_limit_A"] = num_field(
                            "Software current limit (A)", float(d("current_limit_A")),
                            hint="Hard safety ceiling — independent of the supply's own range.")
                        inputs["voltage_compliance_V"] = num_field(
                            "Voltage compliance (V)", float(d("voltage_compliance_V")))
                        inputs["ramp_step_A"] = num_field("Ramp step (A)", float(d("ramp_step_A")))
                        inputs["ramp_delay_s"] = num_field("Ramp delay (s)", float(d("ramp_delay_s")))

                    with stable_card("Averaging & sensors"):
                        inputs["gaussmeter_n_averages"] = num_field(
                            "Field readings averaged per point", float(d("gaussmeter_n_averages")), integer=True)
                        inputs["gaussmeter_read_delay_s"] = num_field(
                            "Delay between readings (s)", float(d("gaussmeter_read_delay_s")))
                        inputs["field_settle_tolerance_mT"] = num_field(
                            "Field-settle tolerance (mT)", float(d("field_settle_tolerance_mT")),
                            hint="Advanced: after each magnet step, the field counts as settled "
                                 "once a short window of gaussmeter readings spans less than this. "
                                 "Raise it if points stall; lower for tighter field control.")
                        inputs["temperature_sensor_uids"] = text_field(
                            "Sensor board UID(s)", d("temperature_sensor_uids"),
                            hint="1 or 2 board UIDs, comma-separated, e.g. 'MB1.T1, DB5.T1'.")

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
            fig.update_layout(
                xaxis_title="Magnetic field (mT)", yaxis_title="Resistance (Ω)",
                margin=dict(l=60, r=20, t=30, b=50), showlegend=True,
            )
            with ui.element("div").classes("w-full").style("aspect-ratio: 1 / 1; min-height: 320px"):
                plot = ui.plotly(fig).classes("w-full h-full")

            columns = [
                {"name": "n", "label": "#", "field": "n"},
                {"name": "Isense", "label": "I_sense (A)", "field": "Isense"},
                {"name": "I", "label": "I_magnet (A)", "field": "I"},
                {"name": "B", "label": "B (mT)", "field": "B"},
                {"name": "Rxy", "label": "R_xy (Ω)", "field": "Rxy"},
                {"name": "Rxx", "label": "R_xx (Ω)", "field": "Rxx"},
                {"name": "T1", "label": "T1 (K)", "field": "T1"},
                {"name": "T2", "label": "T2 (K)", "field": "T2"},
            ]
            table = ui.table(columns=columns, rows=[], row_key="n").classes("w-full").props("dense")
            log_area = ui.log(max_lines=2000).classes("w-full h-48 font-mono text-xs")

    # ── Validation / summary refresh ─────────────────────────────────────

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["data_dir"] = identity.data_dir_input.value
        raw["device"] = identity.device_input.value
        raw["cooldown"] = identity.cooldown_input.value
        raw["temperature_setpoint_K"] = identity.temperature_input.value
        sample_value = identity.sample_dropdown.value
        if sample_value not in (None, NEW_SAMPLE_SENTINEL):
            raw["sample"] = sample_value
        return raw

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

        state["sense_current_list"] = []
        state["sense_current_parse_error"] = None
        try:
            state["sense_current_list"] = parse_value_list(state["sense_current_values"])
        except ValueError as exc:
            state["sense_current_parse_error"] = str(exc)

        state["sweep_rows"] = inputs["sweep_rows"].value or ""
        state["sweep_rows_parsed"] = []
        state["sweep_rows_parse_error"] = None
        try:
            state["sweep_rows_parsed"] = parse_sweep_rows(state["sweep_rows"])
        except ValueError as exc:
            state["sweep_rows_parse_error"] = str(exc)

        return state, errors

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
            f"File:  {preview}" if preview else "File:  (choose a sample and device to preview the filename)")
        with summary_box:
            summary_box.clear()
            render_summary(info, warnings, errors)
        start_btn.set_enabled(not errors and not is_busy())

        theta = None if parse_errors else state.get("field_theta_deg")
        phi = None if parse_errors else state.get("field_phi_deg")
        field_diagram_plot.update_figure(build_field_diagram_figure(theta, phi))

    for inp in list(inputs.values()):
        inp.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    for identity_inp in (identity.data_dir_input, identity.sample_dropdown, identity.device_input,
                         identity.cooldown_input, identity.temperature_input):
        identity_inp.on_value_change(refresh_summary.refresh)
    refresh_summary()
    refresh_on_busy_change(refresh_summary.refresh)

    # ── Run wiring ────────────────────────────────────────────────────────

    trace_index: dict[tuple, int] = {}

    def on_record(record: dict) -> None:
        idx = record.get("series_index", 0)
        series_label = record.get("series_label")
        has_field = record.get("magnet_field_mT") is not None
        x = record["magnet_field_mT"] if has_field else record["point_index"]
        for q in active_quantities(record):
            key = (idx, q)
            if key not in trace_index:
                label = f"{QUANTITY_PLOT_LABELS[q]} {series_label}" if series_label \
                    else QUANTITY_PLOT_LABELS[q]
                dash = "solid" if q == "rxy" else "dash"
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name=label,
                                          line=dict(dash=dash)))
                trace_index[key] = len(fig.data) - 1
            ti = trace_index[key]
            fig.data[ti].x = fig.data[ti].x + (x,)
            fig.data[ti].y = fig.data[ti].y + (record[f"{q}_resistance_ohm"],)
        fig.update_layout(xaxis_title="Magnetic field (mT)" if has_field else "Point #")
        table.rows.append({
            "n": record["point_index"] + 1,
            "Isense": f"{record['sense_current_A']:.4g}" if record.get("sense_current_A") is not None else "—",
            "I": f"{record['magnet_current_A']:.4f}" if record.get("magnet_current_A") is not None else "—",
            "B": f"{record['magnet_field_mT']:.2f}" if record.get("magnet_field_mT") is not None else "—",
            "Rxy": f"{record['rxy_resistance_ohm']:.5g}" if "rxy" in active_quantities(record) else "—",
            "Rxx": f"{record['rxx_resistance_ohm']:.5g}" if "rxx" in active_quantities(record) else "—",
            "T1": f"{record['temperature_1_K']:.3f}" if record.get("temperature_1_K") is not None else "—",
            "T2": f"{record['temperature_2_K']:.3f}" if record.get("temperature_2_K") is not None else "—",
        })

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_run_label(text: str) -> None:
        run_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

    def make_on_finished(plan: MeasurementPlan, run_contexts: list[RunContext], run_extras: list[dict]):
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

        raw = collect_raw()
        _save_settings(raw)

        plan = build_plan(state, Path(state["data_dir"]))
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

        fig.data = []
        trace_index.clear()
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
