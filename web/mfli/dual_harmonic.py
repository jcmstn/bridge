#!/usr/bin/env python3
"""
NiceGUI page for mfli_dual_harmonic.py
===========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Web equivalent of mfli_dual_harmonic_tui.py. Reuses that TUI module's pure
DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/OPTIONAL_NUMERIC_FIELDS/build_summary()/
parse_sensor_uids() so validation stays identical to the TUI.

Supports optional pre-run phase calibration (auto_null_phase) and
sample-geometry metadata, both flowing straight through
run_measurement()'s existing on_point/stop_event hooks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from plotly.subplots import make_subplots
from nicegui import ui

import mfli.mfli_dual_harmonic_tui as program
from mfli.mfli_dual_harmonic_tui import (
    build_plan,
    MFLI_DUAL_HARMONIC_DESCRIPTION,
    DEFAULTS, build_summary,
    compute_filename_preview,
)
from instruments.data_naming import (
    TEST_SAMPLE,
)
from web import run_manager
from web.run_controller import (
    RunController, num_field, text_field, textarea_field, bool_switch,
    optional_num_field, render_summary, busy_banner, is_busy,
    param_card, param_grid, stable_card, stable_grid, advanced_section, measurement_layout,
    program_artifacts, program_run_fn, refresh_on_busy_change,
    finished_handler, load_settings, save_settings,
    form_state,
)
from web.directory_picker import validate_directory
from web.field_diagram import build_field_diagram_figure
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

log = logging.getLogger("web.mfli.dual_harmonic")

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "mfli_dual_harmonic_web_settings.json"

PAGE_TITLE = "MFLI Dual-Harmonic Measurement"
SUITE = "MFLI"


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client  # has slot context now; reused by the detached status/comment task
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(MFLI_DUAL_HARMONIC_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = load_settings(_SETTINGS_PATH)

    def d(key: str):
        if key in saved:
            return saved[key]
        return DEFAULTS.get(key, "")

    def opt(key: str) -> Optional[float]:
        v = d(key)
        return float(v) if v not in ("", None) else None

    inputs: dict = {}
    switches: dict = {}
    optional_inputs: dict = {}
    controller: dict[str, Optional[RunController]] = {"c": None}

    with measurement_layout() as regions:
        with regions.identity:
            identity = identity_bar(
                default_data_dir=saved.get("data_dir") or str(_DATA_DIR),
                default_sample=saved.get("sample") or TEST_SAMPLE,
                default_device=d("device"), default_cooldown=d("cooldown"),
                default_temperature_K=opt("temperature_setpoint_K"),
            )
        with regions.params:
            # ── Tier 1: what defines this run — always visible ───────────────
            with param_grid():
                with param_card("Excitation (current source)"):
                    inputs["frequency_Hz"] = num_field(
                        "Excitation frequency (Hz)", float(d("frequency_Hz")),
                        hint="Avoid exact multiples of 50/60 Hz (mains pickup).")
                    inputs["amplitude_V"] = num_field("Output amplitude (V, peak)", float(d("amplitude_V")))
                    inputs["series_R_ohm"] = num_field(
                        "Series resistor (Ω)", float(d("series_R_ohm")), hint="Sets excitation current: I ≈ V / R.")

                with param_card("Magnet & field sweep"):
                    switches["enable_sweep"] = bool_switch("Sweep magnetic field (Kepco magnet)", d("enable_sweep"))
                    inputs["sweep_rows"] = textarea_field(
                        "Sweep rows: start, stop, points (one per line)",
                        d("sweep_rows"),
                        hint="Adjacent rows sharing a boundary value are merged, not duplicated.")

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))

                with param_card("Phase calibration"):
                    switches["enable_phase_cal"] = bool_switch(
                        "Auto-null 1f phase before run (leader demod phaseshift)", d("enable_phase_cal"))
                    optional_inputs["phase_cal_current_A"] = optional_num_field(
                        "Calibration magnet current (A)", opt("phase_cal_current_A"),
                        hint="Blank = null at the present field (no ramp). Otherwise pick a point "
                             "near saturation — e.g. matching i_max. Only used if the field sweep "
                             "above is enabled.")

                with param_card("Sample geometry & field direction (optional)"):
                    optional_inputs["hall_bar_length_um"] = optional_num_field(
                        "Hall bar length (µm)", opt("hall_bar_length_um"),
                        hint="Current-path length between voltage probes. Leave blank if unknown.")
                    optional_inputs["hall_bar_width_um"] = optional_num_field(
                        "Hall bar width (µm)", opt("hall_bar_width_um"))
                    optional_inputs["hall_bar_thickness_nm"] = optional_num_field(
                        "Film/channel thickness (nm)", opt("hall_bar_thickness_nm"))
                    optional_inputs["field_theta_deg"] = optional_num_field(
                        "θ — tilt from out-of-plane (°)", opt("field_theta_deg"),
                        hint="0° = fully out-of-plane (film normal), 90° = in-plane.",
                        min=0, max=180)
                    optional_inputs["field_phi_deg"] = optional_num_field(
                        "φ — azimuth from current axis (°)", opt("field_phi_deg"),
                        hint="Meaningless when θ=0°.", min=0, max=360)
                    with ui.row().classes("gap-2 mb-1"):
                        ui.button("xy", on_click=lambda: (optional_inputs["field_theta_deg"].set_value(90),
                                                            refresh_summary.refresh())).props("dense outline")
                        ui.button("zx", on_click=lambda: (optional_inputs["field_phi_deg"].set_value(0),
                                                            refresh_summary.refresh())).props("dense outline")
                        ui.button("zy", on_click=lambda: (optional_inputs["field_phi_deg"].set_value(90),
                                                            refresh_summary.refresh())).props("dense outline")
                    field_diagram_plot = ui.plotly(build_field_diagram_figure(
                        opt("field_theta_deg"), opt("field_phi_deg"))).classes("w-full").style("height: 220px")

            # ── Tier 2: precision / speed knobs — collapsed ─────────────────
            with advanced_section("Acquisition & filter settings"):
                with stable_grid():
                    with param_card("1f lock-in filter"):
                        inputs["time_constant_1f_s"] = num_field(
                            "Filter time constant (s)", float(d("time_constant_1f_s")),
                            hint="Bigger = quieter but slower & longer settling.")
                        order_select_1f = ui.select(list(range(1, 9)), value=int(d("order_1f")), label="Filter order").classes("w-full")
                        switches["sinc_filter_1f"] = bool_switch("Sinc filter (extra harmonic rejection)", d("sinc_filter_1f"))

                    with param_card("2f lock-in filter"):
                        inputs["time_constant_2f_s"] = num_field(
                            "Filter time constant (s)", float(d("time_constant_2f_s")),
                            hint="1f bleed-through into the 2f channel is the usual reason "
                                 "this needs a longer TC / higher order than 1f.")
                        order_select_2f = ui.select(list(range(1, 9)), value=int(d("order_2f")), label="Filter order").classes("w-full")
                        switches["sinc_filter_2f"] = bool_switch("Sinc filter (extra harmonic rejection)", d("sinc_filter_2f"))

                    with param_card("Input channels"):
                        inputs["input_range_1f_V"] = num_field(
                            "1f input range (V)", float(d("input_range_1f_V")),
                            hint="Match expected 1f signal size — avoid clipping/poor resolution.")
                        inputs["input_range_2f_V"] = num_field(
                            "2f input range (V)", float(d("input_range_2f_V")),
                            hint="2f is usually much smaller than 1f — set separately.")
                        inputs["sample_rate_Hz"] = num_field("Demodulator sample rate (Sa/s)", float(d("sample_rate_Hz")))

                    with param_card("Acquisition timing"):
                        inputs["settling_time_s"] = num_field(
                            "Settling time per point (s)", float(d("settling_time_s")),
                            hint="Rule of thumb: ≥ 5 × time constant.")
                        inputs["n_averages"] = num_field("Samples to average per point", float(d("n_averages")), integer=True)

            # ── Tier 3: instrument wiring & safety — collapsed ──────────────
            with advanced_section("Instrument configuration & addresses", icon="settings"):
                with stable_grid():
                    with stable_card("Devices & connection"):
                        inputs["leader_device"] = text_field("Leader MFLI (current source + 1f)", d("leader_device"))
                        inputs["follower_device"] = text_field("Follower MFLI (2f)", d("follower_device"))
                        inputs["daq_host"] = text_field("LabOne data server host", d("daq_host"))
                        inputs["daq_port"] = num_field("LabOne data server port", float(d("daq_port")), integer=True)

                    with stable_card("Magnet & gaussmeter addresses"):
                        inputs["visa_resource"] = text_field("Magnet VISA resource", d("visa_resource"))
                        inputs["current_limit_A"] = num_field(
                            "Software current limit (A)", float(d("current_limit_A")),
                            hint="Hard safety ceiling — independent of the supply's own range.")
                        inputs["voltage_compliance_V"] = num_field("Voltage compliance (V)", float(d("voltage_compliance_V")))
                        inputs["ramp_step_A"] = num_field("Ramp step (A)", float(d("ramp_step_A")))
                        inputs["ramp_delay_s"] = num_field("Ramp delay (s)", float(d("ramp_delay_s")))
                        inputs["gaussmeter_visa_resource"] = text_field(
                            "Gaussmeter VISA resource", d("gaussmeter_visa_resource"),
                            hint="Lake Shore 475 — measures the actual field at each point.")
                        inputs["gaussmeter_n_averages"] = num_field(
                            "Field readings averaged per point", float(d("gaussmeter_n_averages")), integer=True)
                        inputs["gaussmeter_read_delay_s"] = num_field(
                            "Delay between readings (s)", float(d("gaussmeter_read_delay_s")))
                        inputs["field_settle_tolerance_mT"] = num_field(
                            "Field-settle tolerance (mT)", float(d("field_settle_tolerance_mT")),
                            hint="Advanced: after each magnet step, wait until a short window of "
                                 "gaussmeter readings spans less than this before the settling time.")

                    with stable_card("Temperature controller"):
                        inputs["temperature_visa_resource"] = text_field("MercuryiTC VISA resource", d("temperature_visa_resource"))
                        inputs["temperature_sensor_uids"] = text_field("Sensor board UID(s)", d("temperature_sensor_uids"))

                    with stable_card("Phase-cal advanced"):
                        inputs["phase_cal_n_averages"] = num_field("Averages per phase read", float(d("phase_cal_n_averages")), integer=True)
                        inputs["phase_cal_max_iterations"] = num_field("Max null iterations", float(d("phase_cal_max_iterations")), integer=True)
                        ui.label(
                            "Nulls the leader's 1f Y quadrature by adjusting its demod phaseshift node "
                            "(the same thing LabOne's \"Auto\" phase button does). X and Y at 2f are both "
                            "already recorded per point in the CSV — check which one actually tracks field "
                            "there before trusting it."
                        ).classes("text-xs text-grey-6")

        with regions.summary:
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start measurement", color="primary").classes("w-full")

        with regions.output:
            with ui.row().classes("w-full items-center gap-3"):
                run_label = ui.label("").classes("text-sm font-bold text-grey-6")
                status_label = ui.label("Idle.").classes("text-sm font-bold")
            abort_btn = ui.button("Abort (safe ramp-down)", color="negative").props("outline")
            abort_btn.set_visibility(False)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig.update_yaxes(title_text="1f  R (V)", row=1, col=1)
            fig.update_yaxes(title_text="2f  R (V)", row=2, col=1)
            fig.update_xaxes(title_text="Magnetic field (mT)", row=2, col=1)
            fig.update_layout(margin=dict(l=60, r=20, t=20, b=50), showlegend=False)
            fig.add_scatter(x=[], y=[], mode="lines+markers", name="1f R", line=dict(color="#1f77b4"), row=1, col=1)
            fig.add_scatter(x=[], y=[], mode="lines+markers", name="2f R", line=dict(color="#ff7f0e"), row=2, col=1)
            with ui.element("div").classes("w-full").style("aspect-ratio: 1 / 2; max-height: 90vh"):
                plot = ui.plotly(fig).classes("w-full h-full")

            columns = [
                {"name": "n", "label": "#", "field": "n"},
                {"name": "I", "label": "I (A)", "field": "I"},
                {"name": "B", "label": "B (mT)", "field": "B"},
                {"name": "R1", "label": "1f R (V)", "field": "R1"},
                {"name": "th1", "label": "1f θ (°)", "field": "th1"},
                {"name": "R2", "label": "2f R (V)", "field": "R2"},
                {"name": "th2", "label": "2f θ (°)", "field": "th2"},
                {"name": "T1", "label": "T1 (K)", "field": "T1"},
                {"name": "T2", "label": "T2 (K)", "field": "T2"},
            ]
            table = ui.table(columns=columns, rows=[], row_key="n").classes("w-full").props("dense")
            log_area = ui.log(max_lines=2000).classes("w-full h-48 font-mono text-xs")

    def parse_state() -> tuple[dict, list[str]]:
        return form_state(program, identity, inputs=inputs, switches=switches,
                          selects={"order_1f": order_select_1f, "order_2f": order_select_2f},
                          optional_inputs=optional_inputs)

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        for fid, inp in optional_inputs.items():
            raw[fid] = inp.value if inp.value is not None else ""
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["order_1f"] = order_select_1f.value
        raw["order_2f"] = order_select_2f.value
        raw["data_dir"] = identity.data_dir_input.value
        raw["device"] = identity.device_input.value
        raw["cooldown"] = identity.cooldown_input.value
        raw["temperature_setpoint_K"] = identity.temperature_input.value if identity.temperature_input.value is not None else ""
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
            f"File:  {preview}" if preview else "File:  (choose a sample and device to preview the filename)")
        with summary_box:
            summary_box.clear()
            render_summary(info, warnings, errors)
        start_btn.set_enabled(not errors and not is_busy())

        theta = None if parse_errors else state.get("field_theta_deg")
        phi = None if parse_errors else state.get("field_phi_deg")
        field_diagram_plot.update_figure(build_field_diagram_figure(theta, phi))

    for inp in list(inputs.values()) + list(optional_inputs.values()) + [
        identity.data_dir_input, identity.sample_dropdown, identity.device_input,
        identity.cooldown_input, identity.temperature_input,
    ]:
        inp.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    order_select_1f.on_value_change(refresh_summary.refresh)
    order_select_2f.on_value_change(refresh_summary.refresh)
    refresh_summary()
    refresh_on_busy_change(refresh_summary.refresh)

    # ── Run wiring ───────────────────────────────────────────────────────

    def on_record(record: dict) -> None:
        has_field = record.get("magnet_field_mT") is not None
        x = record["magnet_field_mT"] if has_field else record["point_index"]
        fig.data[0].x = fig.data[0].x + (x,)
        fig.data[0].y = fig.data[0].y + (record["1f_R_V"],)
        fig.data[1].x = fig.data[1].x + (x,)
        fig.data[1].y = fig.data[1].y + (record["2f_R_V"],)
        table.rows.append({
            "n": record["point_index"] + 1,
            "I": f"{record['magnet_current_A']:.4f}" if record.get("magnet_current_A") is not None else "—",
            "B": f"{record['magnet_field_mT']:.2f}" if record.get("magnet_field_mT") is not None else "—",
            "R1": f"{record['1f_R_V']:.4e}", "th1": f"{record['1f_theta_deg']:.2f}",
            "R2": f"{record['2f_R_V']:.4e}", "th2": f"{record['2f_theta_deg']:.2f}",
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

        fig.data[0].x = (); fig.data[0].y = ()
        fig.data[1].x = (); fig.data[1].y = ()
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
