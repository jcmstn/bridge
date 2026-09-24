#!/usr/bin/env python3
"""
NiceGUI page for sot/sot_nonlocal_switching.py
==========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-21

Web equivalent of sot/sot_nonlocal_switching_tui.py. Reuses that TUI module's pure
DEFAULTS / field groups / resolve_state() / build_summary() / build_plan() /
build_header_fields(), and — because the connect → initialize → sweep → shut
down sequence is not UI-specific — its run_plan() and PNG writer too. This page
only supplies the form, the live two-panel Plotly figure (R_NL above V_even,
one trace pair per initial state) and the RunController callbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from nicegui import ui
from plotly.subplots import make_subplots

from instruments.data_naming import (
    TEST_SAMPLE, RunContext,
)
import sot.sot_nonlocal_switching_tui as program
from sot.sot_nonlocal_switching_tui import (
    DEFAULTS, NLSW_DESCRIPTION, build_plan, build_summary,
    compute_filename_preview,
)
from web.directory_picker import validate_directory
from web.identity_bar import identity_bar
from web.run_controller import (
    RunController, advanced_section, bool_switch, busy_banner,
    is_busy, measurement_layout, num_field, optional_num_field, param_card, param_grid,
    render_summary, stable_card, stable_grid, text_field,
    refresh_on_busy_change,
    finished_handler, load_settings, program_artifacts, program_run_fn, save_settings,
    form_state,
)
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "sot_nonlocal_switching_web_settings.json"

PAGE_TITLE = "Nonlocal Spin-Current Switching"
SUITE = "SOT"

log = logging.getLogger("web.sot.nonlocal_switching")

_COLORS = ["#2E3192", "#e34948", "#2E7D32", "#B26A00", "#7E57C2", "#0277BD"]


def _optional_float(value) -> Optional[float]:
    return None if value in ("", None) else float(value)


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client  # has slot context now; reused by the detached status/comment task
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(NLSW_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

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
                default_temperature_K=_optional_float(d("temperature_setpoint_K")),
            )
        with regions.params:
            # ── Tier 1: what defines this run — always visible ───────────────
            with param_grid():
                with param_card("Write pulse (6221 WAVE, one lobe 0 → ±I → 0)"):
                    inputs["pulse_current_start_A"] = num_field(
                        "Pulse current start (A)", float(d("pulse_current_start_A")),
                        hint="Signed: negative = a −I lobe (never a ± pair).")
                    inputs["pulse_current_stop_A"] = num_field(
                        "Pulse current stop (A)", float(d("pulse_current_stop_A")))
                    inputs["pulse_current_step_A"] = num_field(
                        "Pulse current step (A)", float(d("pulse_current_step_A")),
                        hint="One pulse per step; 0 A = read only.")
                    switches["amplitude_bidirectional"] = bool_switch(
                        "Then sweep back (loop / no-reset control)", d("amplitude_bidirectional"))
                    inputs["pulse_width_s"] = num_field(
                        "Requested pulse width (s)", float(d("pulse_width_s")),
                        hint="~1 ms typical. Measured width is logged.")
                    inputs["pulse_compliance_V"] = num_field(
                        "Pulse voltage compliance (V)", float(d("pulse_compliance_V")),
                        hint="Above I_max × R_injector, or the pulse clips silently.")

                with param_card("Nonlocal read (6221 DC / 2182A)"):
                    inputs["delay_after_pulse_s"] = num_field(
                        "Delay after pulse (s)", float(d("delay_after_pulse_s")),
                        hint="Wait between pulse end and the read.")
                    inputs["sense_current_A"] = num_field(
                        "Sense current (A)", float(d("sense_current_A")),
                        hint="Well below switching. Sign = read polarity if reversal off.")
                    inputs["compliance_V"] = num_field("Read compliance (V)", float(d("compliance_V")))
                    switches["reversal_enabled"] = bool_switch(
                        "Reverse the sense current each read (+I/−I)", d("reversal_enabled"))
                    ui.label(
                        "Off: fixed polarity — no EMF cancel, no V_even."
                    ).classes("text-xs text-grey-6 -mt-1 mb-1")
                    inputs["n_averages"] = num_field(
                        "Averages per read", float(d("n_averages")), integer=True,
                        hint="± pairs with reversal, plain samples without.")
                    inputs["source_delay_s"] = num_field(
                        "Settle after polarity flip (s)", float(d("source_delay_s")))
                    inputs["nplc"] = num_field("2182A integration (NPLC)", float(d("nplc")))
                    inputs["switch_sigma"] = num_field(
                        "'switched' threshold (σ)", float(d("switch_sigma")),
                        hint="|ΔR| vs the previous row, in combined standard errors.")

                with param_card("Field initialization (Kepco magnet)"):
                    inputs["init_magnet_currents"] = text_field(
                        "Init magnet current(s) (A)", d("init_magnet_currents"),
                        hint="Blank = magnet untouched. Comma-separate: one run per init state.")
                    inputs["sweep_magnet_current_A"] = num_field(
                        "Magnet current during the sweep (A)", float(d("sweep_magnet_current_A")),
                        hint="0 = field off (remanent state).")

                with param_card("Reference levels (optional)"):
                    inputs["R_P_ohm"] = optional_num_field(
                        "R_NL at the P level (Ω)", _optional_float(d("R_P_ohm")),
                        hint="From a dc_spin_valve sweep of this signal.")
                    inputs["R_AP_ohm"] = optional_num_field(
                        "R_NL at the AP level (Ω)", _optional_float(d("R_AP_ohm")),
                        hint="Enables state_AP_fraction.")

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))

            # ── Tier 2: instrument wiring & safety — collapsed ──────────────
            with advanced_section("Instrument configuration & addresses", icon="settings"):
                with stable_grid():
                    with stable_card("Keithley 6221 + 2182A"):
                        inputs["source_visa_resource"] = text_field(
                            "Keithley 6221 (pulse + read current)", d("source_visa_resource"))
                        inputs["voltmeter_visa_resource"] = text_field(
                            "Keithley 2182A (nonlocal voltage)", d("voltmeter_visa_resource"))

                    with stable_card("Kepco magnet + Lake Shore 475"):
                        inputs["magnet_visa_resource"] = text_field(
                            "Magnet VISA resource", d("magnet_visa_resource"))
                        inputs["current_limit_A"] = num_field(
                            "Software current limit (A)", float(d("current_limit_A")),
                            hint="Hard safety ceiling.")
                        inputs["magnet_voltage_compliance_V"] = num_field(
                            "Voltage compliance (V)", float(d("magnet_voltage_compliance_V")))
                        inputs["ramp_step_A"] = num_field("Ramp step (A)", float(d("ramp_step_A")))
                        inputs["ramp_delay_s"] = num_field("Ramp delay (s)", float(d("ramp_delay_s")))
                        inputs["gaussmeter_visa_resource"] = text_field(
                            "Gaussmeter VISA resource", d("gaussmeter_visa_resource"),
                            hint="Lake Shore 475.")
                        inputs["gaussmeter_n_averages"] = num_field(
                            "Field readings averaged", float(d("gaussmeter_n_averages")), integer=True)
                        inputs["gaussmeter_read_delay_s"] = num_field(
                            "Delay between readings (s)", float(d("gaussmeter_read_delay_s")))
                        inputs["field_settle_tolerance_mT"] = num_field(
                            "Field-settle tolerance (mT)", float(d("field_settle_tolerance_mT")))

                    with stable_card("Temperature (MercuryiTC)"):
                        inputs["temperature_visa_resource"] = text_field(
                            "MercuryiTC VISA resource", d("temperature_visa_resource"))
                        inputs["temperature_sensor_uids"] = text_field(
                            "Sensor board UID(s)", d("temperature_sensor_uids"))

        with regions.summary:
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start measurement", color="primary").classes("w-full")

        with regions.output:
            with ui.row().classes("w-full items-center gap-3"):
                run_label = ui.label("").classes("text-sm font-bold text-grey-6")
                status_label = ui.label("Idle.").classes("text-sm font-bold")
            abort_btn = ui.button("Abort (safe shutdown)", color="negative").props("outline")
            abort_btn.set_visibility(False)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
            fig.update_yaxes(title_text="R_NL (mΩ)", row=1, col=1)
            fig.update_yaxes(title_text="V_even (µV)", row=2, col=1)
            fig.update_xaxes(title_text="Pulse current (mA)", row=2, col=1)
            fig.update_layout(margin=dict(l=60, r=20, t=30, b=50), showlegend=True)
            with ui.element("div").classes("w-full").style("aspect-ratio: 1 / 2; min-height: 640px"):
                plot = ui.plotly(fig).classes("w-full h-full")

            columns = [
                {"name": "n", "label": "#", "field": "n"},
                {"name": "init", "label": "I_init (A)", "field": "init"},
                {"name": "Ip", "label": "I_pulse (A)", "field": "Ip"},
                {"name": "R", "label": "R_NL (mΩ)", "field": "R"},
                {"name": "dR", "label": "ΔR (mΩ)", "field": "dR"},
                {"name": "sw", "label": "switched", "field": "sw"},
                {"name": "Ve", "label": "V_even (µV)", "field": "Ve"},
                {"name": "T1", "label": "T1 (K)", "field": "T1"},
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

    def init_series(labels: list[Optional[str]]) -> None:
        """Two traces per initial state: R_NL on the top panel, V_even below,
        same colour, one legend entry."""
        fig.data = []
        for i, label in enumerate(labels):
            color = _COLORS[i % len(_COLORS)]
            style = dict(mode="lines+markers", line=dict(color=color), marker=dict(color=color),
                         legendgroup=str(i))
            fig.add_trace(go.Scatter(x=[], y=[], name=label or "R_NL", **style), row=1, col=1)
            fig.add_trace(go.Scatter(x=[], y=[], name=label or "V_even", showlegend=False, **style),
                          row=2, col=1)

    def on_record(record: dict) -> None:
        ri = 2 * record.get("series_index", 0)
        x = record["pulse_current_A"] * 1e3
        fig.data[ri].x = fig.data[ri].x + (x,)
        fig.data[ri].y = fig.data[ri].y + (record["nl_resistance_ohm"] * 1e3,)
        if record.get("voltage_even_V") is not None:        # blank with reversal off
            fig.data[ri + 1].x = fig.data[ri + 1].x + (x,)
            fig.data[ri + 1].y = fig.data[ri + 1].y + (record["voltage_even_V"] * 1e6,)
        init = record.get("init_magnet_current_A")
        dr = record.get("delta_R_ohm")
        sw = record.get("switched")
        t1 = record.get("temperature_1_K")
        table.rows.append({
            "n": len(table.rows) + 1,
            "init": f"{init:g}" if init is not None else "—",
            "Ip": f"{record['pulse_current_A']:.4g}",
            "R": f"{record['nl_resistance_ohm'] * 1e3:.4f}",
            "dR": f"{dr * 1e3:+.4f}" if dr is not None else "—",
            "sw": "—" if sw is None else ("YES" if sw else "no"),
            "Ve": f"{record['voltage_even_V'] * 1e6:.3f}" if record.get("voltage_even_V") is not None else "—",
            "T1": f"{t1:.3f}" if t1 is not None else "—",
        })

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_run_label(text: str) -> None:
        run_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

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

        data_root = Path(state["data_dir"])
        plan = build_plan(state, data_root)
        labels = [f"I_init={i:g}A" if i is not None and len(plan.series_values) > 1 else None
                  for i in plan.series_values]
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
                program, plan, run_contexts, run_extras),
            sample=plan.sample, device=plan.device, run_cost=plan.run_cost,
        )
        if not rc.try_start():
            ui.notify("Another measurement is already running — see the banner above.", type="warning")
            return
        controller["c"] = rc

        init_series(labels)
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
