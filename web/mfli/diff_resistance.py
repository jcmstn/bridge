#!/usr/bin/env python3
"""
NiceGUI page for mfli_diff_resistance_vs_bias.py
======================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Web equivalent of mfli_diff_resistance_tui.py. Reuses that TUI module's
pure DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/build_summary()/parse_sensor_uids().

No magnet/gaussmeter here (this measurement has no field axis — the DC
bias sweep is the whole measurement). Shutdown order matters:
ramp_bias_to_zero() before shutdown_output(), gentler on the DUT than a
hard jump to 0 V.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from plotly.subplots import make_subplots
from nicegui import ui

import mfli.mfli_diff_resistance_tui as program
from mfli.mfli_diff_resistance_tui import (
    build_plan,
    MFLI_DIFF_RESISTANCE_DESCRIPTION,
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS,
    build_summary,
    compute_filename_preview,
)
from instruments.data_naming import (
    TEST_SAMPLE,
)
from web import run_manager
from web.run_controller import (
    RunController, num_field, text_field, bool_switch,
    param_card, param_grid, advanced_section, stable_card, stable_grid, measurement_layout,
    render_summary, busy_banner, is_busy,
    program_artifacts, program_run_fn, refresh_on_busy_change,
    finished_handler, load_settings, save_settings,
)
from web.directory_picker import validate_directory
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "mfli_diff_resistance_web_settings.json"

PAGE_TITLE = "MFLI Differential Resistance vs. Bias"
SUITE = "MFLI"


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client  # has slot context now; reused by the detached status/comment task
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(MFLI_DIFF_RESISTANCE_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = load_settings(_SETTINGS_PATH)

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
                with param_card("Bias sweep"):
                    inputs["bias_min_V"] = num_field("DC bias sweep min (V)", float(d("bias_min_V")))
                    inputs["bias_max_V"] = num_field("DC bias sweep max (V)", float(d("bias_max_V")))
                    inputs["n_points"] = num_field(
                        "Points per sweep direction", float(d("n_points")), integer=True,
                        hint="Bidirectional: min → max → min (reveals hysteresis).")

                with param_card("Excitation"):
                    inputs["frequency_Hz"] = num_field(
                        "AC excitation frequency (Hz)", float(d("frequency_Hz")),
                        hint="Avoid exact multiples of 50/60 Hz (mains pickup).")
                    inputs["ac_amplitude_V"] = num_field(
                        "AC excitation amplitude (V, peak)", float(d("ac_amplitude_V")),
                        hint="Keep small vs. any bias step over which R_diff changes.")
                    inputs["series_R_ohm"] = num_field(
                        "Series resistor (Ω)", float(d("series_R_ohm")),
                        hint="Current-limiting/protection resistor — not used to compute I.")

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))

            # ── Tier 2: precision / speed knobs — collapsed ─────────────────
            with advanced_section("Acquisition & filter settings"):
                with stable_grid():
                    with param_card("Lock-in filter"):
                        inputs["time_constant_s"] = num_field(
                            "Filter time constant (s)", float(d("time_constant_s")),
                            hint="Bigger = quieter but slower & longer settling.")
                        order_select = ui.select(list(range(1, 9)), value=int(d("order")), label="Filter order").classes("w-full")
                        switches["sinc_filter"] = bool_switch("Sinc filter (extra harmonic rejection)", d("sinc_filter"))

                    with param_card("Input ranges"):
                        inputs["current_input_range_A"] = num_field(
                            "Current-sense input range (A)", float(d("current_input_range_A")),
                            hint="Leader's Current Input 1 — size to the actual DUT current.")
                        inputs["voltage_input_range_V"] = num_field(
                            "Voltage-sense input range (V)", float(d("voltage_input_range_V")),
                            hint="Follower input, across the DUT.")
                        inputs["sample_rate_Hz"] = num_field("Demodulator sample rate (Sa/s)", float(d("sample_rate_Hz")))

                    with param_card("Acquisition timing"):
                        inputs["settling_time_s"] = num_field(
                            "Settling time per bias point (s)", float(d("settling_time_s")),
                            hint="Rule of thumb: ≥ 5 × time constant.")
                        inputs["n_averages"] = num_field(
                            "Samples to average per point (each demod)", float(d("n_averages")), integer=True)

            # ── Tier 3: instrument wiring & safety — collapsed ──────────────
            with advanced_section("Instrument configuration & addresses", icon="settings"):
                with stable_grid():
                    with stable_card("Devices & connection"):
                        inputs["leader_device"] = text_field("Leader MFLI (bias + AC excitation, I-sense)", d("leader_device"))
                        inputs["follower_device"] = text_field("Follower MFLI (V-sense across DUT)", d("follower_device"))
                        inputs["daq_host"] = text_field("LabOne data server host", d("daq_host"))
                        inputs["daq_port"] = num_field("LabOne data server port", float(d("daq_port")), integer=True)

                    with stable_card("Temperature controller"):
                        inputs["temperature_visa_resource"] = text_field("MercuryiTC VISA resource", d("temperature_visa_resource"))
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

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
            fig.update_yaxes(title_text="R_diff (Ω)", row=1, col=1)
            fig.update_yaxes(title_text="Reactive (Ω)", row=2, col=1)
            fig.update_yaxes(title_text="Phase (deg)", row=3, col=1)
            fig.update_xaxes(title_text="DC bias (V)", row=3, col=1)
            fig.update_layout(margin=dict(l=60, r=20, t=20, b=50), showlegend=False)
            fig.add_scatter(x=[], y=[], mode="lines+markers", line=dict(color="#2E3192"), row=1, col=1)
            fig.add_scatter(x=[], y=[], mode="lines+markers", line=dict(color="#e34948"), row=2, col=1)
            fig.add_scatter(x=[], y=[], mode="lines+markers", line=dict(color="#00AEEF"), row=3, col=1)
            with ui.element("div").classes("w-full").style("aspect-ratio: 1 / 3; max-height: 90vh"):
                plot = ui.plotly(fig).classes("w-full h-full")

            columns = [
                {"name": "n", "label": "#", "field": "n"},
                {"name": "Vb", "label": "V_bias (V)", "field": "Vb"},
                {"name": "Iac", "label": "I_ac (A)", "field": "Iac"},
                {"name": "Vdut", "label": "V_dut (V)", "field": "Vdut"},
                {"name": "R", "label": "R_diff (Ω)", "field": "R"},
                {"name": "X", "label": "X_react (Ω)", "field": "X"},
                {"name": "Z", "label": "|Z| (Ω)", "field": "Z"},
                {"name": "phase", "label": "phase (°)", "field": "phase"},
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
        state["temperature_setpoint_K"] = identity.temperature_input.value
        for fid, sw in switches.items():
            state[fid] = sw.value
        state["order"] = int(order_select.value)
        state["device"] = (identity.device_input.value or "").strip()
        state["cooldown"] = (identity.cooldown_input.value or "").strip()
        sample_value = identity.sample_dropdown.value
        state["sample"] = sample_value if sample_value not in (None, NEW_SAMPLE_SENTINEL) else ""
        return state, errors

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        raw["temperature_setpoint_K"] = identity.temperature_input.value \
            if identity.temperature_input.value is not None else ""
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["order"] = order_select.value
        raw["data_dir"] = identity.data_dir_input.value
        raw["device"] = identity.device_input.value
        raw["cooldown"] = identity.cooldown_input.value
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
            render_summary(info, warnings, errors)
        start_btn.set_enabled(not errors and not is_busy())

    for inp in list(inputs.values()):
        inp.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    for inp in (identity.data_dir_input, identity.sample_dropdown, identity.device_input,
                identity.cooldown_input, identity.temperature_input):
        inp.on_value_change(refresh_summary.refresh)
    order_select.on_value_change(refresh_summary.refresh)
    refresh_summary()
    refresh_on_busy_change(refresh_summary.refresh)

    def on_record(record: dict) -> None:
        x = record["bias_V"]
        for i, key in enumerate(("R_diff_ohm", "X_reactive_ohm", "Z_phase_deg")):
            fig.data[i].x = fig.data[i].x + (x,)
            fig.data[i].y = fig.data[i].y + (record[key],)
        table.rows.append({
            "n": record["point_index"] + 1,
            "Vb": f"{record['bias_V']:.4f}",
            "Iac": f"{record['I_ac_A']:.4e}",
            "Vdut": f"{record['V_dut_ac_V']:.4e}",
            "R": f"{record['R_diff_ohm']:.5g}",
            "X": f"{record['X_reactive_ohm']:.3g}",
            "Z": f"{record['Z_mag_ohm']:.5g}",
            "phase": f"{record['Z_phase_deg']:.2f}",
            "T1": f"{record['temperature_1_K']:.3f}" if record.get("temperature_1_K") is not None else "—",
            "T2": f"{record['temperature_2_K']:.3f}" if record.get("temperature_2_K") is not None else "—",
        })

    def on_status(text: str) -> None:
        status_label.set_text(text)

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

        # build_plan() allocates the run number -- check the global lock
        # first, or a busy lock would leave an in_progress index row that is
        # never finalized. (Same event-loop tick as try_start() below, so no
        # other page can take the lock in between.)
        if run_manager.snapshot() is not None:
            ui.notify("Another measurement is already running — see the banner above.", type="warning")
            return
        plan = build_plan(state, Path(state["data_dir"]))
        run_contexts: list = []
        run_extras: list = []
        run_label.set_text(f"Run #{plan.run_ctx.run_str}")

        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE, run_fn=program_run_fn(program, plan, run_contexts, run_extras),
            save_artifacts=lambda records, result, status: program_artifacts(program, plan, run_contexts),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[plan.acq_cfg.output_file],
            on_tick=lambda: (plot.update(), table.update()),
            on_record=on_record, on_status=on_status, on_log=on_log,
            on_finished=make_on_finished(plan, run_contexts, run_extras),
            sample=plan.run_ctx.sample, device=plan.run_ctx.device,
            run_number=plan.run_ctx.run_number, run_cost=plan.run_cost,
        )
        if not rc.try_start():
            ui.notify("Another measurement is already running — see the banner above.", type="warning")
            return
        controller["c"] = rc

        for i in range(3):
            fig.data[i].x = (); fig.data[i].y = ()
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
