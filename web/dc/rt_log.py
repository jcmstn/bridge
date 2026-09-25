#!/usr/bin/env python3
"""
NiceGUI page for dc_rt_log.py
=============================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-25

Web equivalent of dc_rt_log_tui.py. Reuses that TUI module's pure
DEFAULTS / *_FIELDS / build_summary() / build_plan() / run_plan(), so
validation and the run itself stay identical to the TUI.

Live view: one R-vs-temperature panel per sensor that is reading (two
sensors -> two panels), then R vs time — the same layout as the PNG. The table keeps only the latest rows; the full log is in the
raw file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui

import dc.dc_rt_log_tui as program
from dc.dc_rt_log import MARKER, SENSOR_COLORS, SENSOR_COLUMNS, TIME_COLOR, sensors_in
from dc.dc_rt_log_tui import (
    DC_RT_LOG_DESCRIPTION,
    DEFAULTS,
    build_plan,
    build_summary,
    compute_filename_preview,
)
from instruments.data_naming import TEST_SAMPLE, RunContext
from web.run_controller import (
    RunController, num_field, optional_num_field, text_field,
    bool_switch, render_summary, busy_banner, is_busy,
    param_card, stable_card, param_grid, stable_grid, advanced_section, measurement_layout,
    program_artifacts, program_run_fn, refresh_on_busy_change,
    finished_handler, load_settings, save_settings,
    form_state,
)
from web.directory_picker import validate_directory
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "dc_rt_log_web_settings.json"

PAGE_TITLE = "DC R vs T log"
SUITE = "DC"
TABLE_ROWS = 200      # latest rows shown; a multi-hour log would bloat the page otherwise

log = logging.getLogger("web.dc.rt_log")


PANEL_PX = 340       # plot height per panel
_MARKER = dict(size=MARKER["ms"] * 1.5, opacity=MARKER["alpha"],
               line=dict(width=MARKER["mew"] * 2, color=MARKER["mec"]))


def make_figure(sensors: list[int]) -> go.Figure:
    """Same layout as the PNG (dc_rt_log.plot_results): one R-vs-T panel per
    sensor in `sensors`, then R vs time."""
    titles = [f"Resistance vs. temperature (sensor {k})" for k in sensors] + ["Resistance vs. time"]
    fig = make_subplots(rows=len(titles), cols=1, subplot_titles=titles)
    for row, k in enumerate(sensors, start=1):
        fig.add_trace(go.Scattergl(x=[], y=[], mode="markers",
                                   marker=dict(color=SENSOR_COLORS[k - 1], **_MARKER)), row=row, col=1)
        fig.update_xaxes(title_text=f"Temperature, sensor {k} (K)", row=row, col=1)
    row = len(titles)
    fig.add_trace(go.Scattergl(x=[], y=[], mode="markers",
                               marker=dict(color=TIME_COLOR, **_MARKER)), row=row, col=1)
    fig.update_xaxes(title_text="Time (min)", row=row, col=1)
    fig.update_yaxes(title_text="R = V_odd / I (Ω)")
    fig.update_layout(margin=dict(l=60, r=20, t=40, b=50), showlegend=False)
    return fig


def _opt(value) -> Optional[float]:
    return float(value) if value not in ("", None) else None


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(DC_RT_LOG_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = load_settings(_SETTINGS_PATH)

    def d(key: str):
        return saved[key] if key in saved else DEFAULTS.get(key, "")

    inputs: dict = {}
    switches: dict = {}
    controller: dict[str, Optional[RunController]] = {"c": None}

    with measurement_layout() as regions:
        with regions.identity:
            identity = identity_bar(
                default_data_dir=saved.get("data_dir") or str(_DATA_DIR),
                default_sample=saved.get("sample") or TEST_SAMPLE,
                default_device=d("device"), default_cooldown=d("cooldown"),
                default_temperature_K=_opt(d("temperature_setpoint_K")),
            )
        with regions.params:
            with param_grid():
                with param_card("Sense current (Keithley 6221)"):
                    inputs["sense_current_A"] = num_field(
                        "Sense current (A)", float(d("sense_current_A")),
                        hint="Reversed ±I every pair. Keep it small enough to avoid self-heating.")
                    inputs["n_reversals"] = num_field(
                        "±I pairs per sample", float(d("n_reversals")), integer=True,
                        hint="More = quieter, but a longer sample (more drift inside it).")

                with param_card("When to sample / stop"):
                    inputs["interval_s"] = num_field("Sample interval (s)", float(d("interval_s")),
                                                     hint="0 = back to back.")
                    inputs["max_duration_min"] = num_field(
                        "Maximum duration (min)", float(d("max_duration_min")),
                        hint="Safety stop for an unattended run.")
                    inputs["T_stop_K"] = optional_num_field(
                        "Stop at temperature (K, optional)", _opt(d("T_stop_K")),
                        hint="Stops once sensor 1 crosses it, in either direction.")

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))
                    inputs["temperature_sensor_uids"] = text_field(
                        "Sensor UID(s)", d("temperature_sensor_uids"),
                        hint="1 or 2, e.g. MB1.T1, DB5.T1 — a missing one is dropped.")

            with advanced_section("Acquisition & filter settings"):
                with stable_grid():
                    with param_card("Source & voltmeter"):
                        inputs["compliance_V"] = num_field("Compliance voltage (V)", float(d("compliance_V")))
                        inputs["nplc"] = num_field("NPLC (integration time)", float(d("nplc")),
                                                   hint="Bigger = quieter but slower.")
                        switches["auto_range"] = bool_switch("Auto-range", d("auto_range"))

            with advanced_section("Instrument configuration & addresses", icon="settings"):
                with stable_grid():
                    with stable_card("Instrument addresses"):
                        inputs["source_visa_resource"] = text_field(
                            "Keithley 6221 (current source)", d("source_visa_resource"))
                        inputs["voltmeter_visa_resource"] = text_field(
                            "Keithley 2182 (voltage)", d("voltmeter_visa_resource"))
                        inputs["temperature_visa_resource"] = text_field(
                            "MercuryiTC VISA resource", d("temperature_visa_resource"))
                    with stable_card("Source timing"):
                        inputs["source_delay_s"] = num_field(
                            "6221 source delay (s)", float(d("source_delay_s")),
                            hint="Settle after each polarity flip.")

        with regions.summary:
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start log", color="primary").classes("w-full")

        with regions.output:
            with ui.row().classes("w-full items-center gap-3"):
                run_label = ui.label("").classes("text-sm font-bold text-grey-6")
                status_label = ui.label("Idle.").classes("text-sm font-bold")
            abort_btn = ui.button("Stop logging", color="negative").props("outline")
            abort_btn.set_visibility(False)

            plot_box = ui.element("div").classes("w-full")
            with plot_box:
                plot = ui.plotly(make_figure([1])).classes("w-full h-full")
            plot_box.style(f"height: {PANEL_PX * 2}px")

            columns = [
                {"name": "n", "label": "#", "field": "n"},
                {"name": "t", "label": "t (s)", "field": "t"},
                {"name": "T1", "label": "T1 (K)", "field": "T1"},
                {"name": "dT1", "label": "ΔT1 (mK)", "field": "dT1"},
                {"name": "T2", "label": "T2 (K)", "field": "T2"},
                {"name": "R", "label": "R (Ω)", "field": "R"},
                {"name": "sR", "label": "σR (Ω)", "field": "sR"},
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

    # (trace index, record column) per panel; rebuilt on the first record,
    # once it is known which sensors answered at connect
    panels: list = []
    data: dict = {}

    def reset_plot() -> None:
        panels.clear()
        data.clear()
        plot.update_figure(make_figure([1]))      # placeholder until the first sample

    def build_panels(record: dict) -> None:
        sensors = sensors_in([record])
        plot_box.style(f"height: {PANEL_PX * (len(sensors) + 1)}px")
        plot.update_figure(make_figure(sensors))
        panels.extend((i, SENSOR_COLUMNS[k - 1]) for i, k in enumerate(sensors))
        panels.append((len(sensors), "elapsed_min"))
        for _, col in panels:
            data[col] = ([], [])

    def on_record(record: dict) -> None:
        if not panels:
            build_panels(record)
        record = {**record, "elapsed_min": record["elapsed_s"] / 60.0}
        for i, col in panels:
            if record.get(col) is not None:
                xs, ys = data[col]
                xs.append(record[col])
                ys.append(record["resistance_ohm"])
                plot.figure.data[i].x, plot.figure.data[i].y = tuple(xs), tuple(ys)

        drift = record.get("temperature_1_drift_K")
        table.rows.append({
            "n": record["point_index"] + 1,
            "t": f"{record['elapsed_s']:.1f}",
            "T1": f"{record['temperature_1_K']:.3f}" if record.get("temperature_1_K") is not None else "—",
            "dT1": f"{drift * 1e3:.1f}" if drift is not None else "—",
            "T2": f"{record['temperature_2_K']:.3f}" if record.get("temperature_2_K") is not None else "—",
            "R": f"{record['resistance_ohm']:.6g}",
            "sR": f"{record['resistance_sem_ohm']:.2g}",
        })
        del table.rows[:-TABLE_ROWS]

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_run_label(text: str) -> None:
        run_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

    def on_start() -> None:
        state, parse_errors = parse_state()
        _, dir_error = validate_directory(identity.data_dir_input.value or "")
        if parse_errors or dir_error or build_summary(state)[2]:
            ui.notify("Fix the blocking issues before starting.", type="negative")
            return
        state["data_dir"] = prepare_data_root(identity.data_dir_input.value, state["sample"])
        save_settings(_SETTINGS_PATH, collect_raw())

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
            on_finished=finished_handler(
                page_client, controller, status_label, abort_btn, start_btn, refresh_summary.refresh,
                program, plan, run_contexts, run_extras, done_text="Log finished."),
            sample=plan.sample, device=plan.device, run_cost=plan.run_cost,
            stop_is_normal_end=True,      # Stop ends a log normally -> "completed"
        )
        if not rc.try_start():
            ui.notify("Another measurement is already running — see the banner above.", type="warning")
            return
        controller["c"] = rc

        reset_plot()
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
