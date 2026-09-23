#!/usr/bin/env python3
"""
NiceGUI page for mfli_phase_calibration.py
================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Web equivalent of mfli_phase_calibration_tui.py. Reuses that TUI module's
own DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/LIST_FIELDS/build_summary().

This is the one page that doesn't fit the plain run_measurement()->DataFrame
pattern every other page follows: the orchestrator is
run_phase_calibration() -> PhaseCalibrationReport, with a third callback
(on_status, alongside stop_event/on_point) already supported directly by
RunCallbacks. finalize (on_finished) renders format_report(report) into a
dedicated Report panel in addition to the log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from plotly.subplots import make_subplots
from nicegui import ui

from dc.dc_sweep_utils import parse_sweep_rows
from mfli.mfli_phase_calibration import (
    format_report,
)
import mfli.mfli_phase_calibration_tui as program
from mfli.mfli_phase_calibration_tui import (
    build_plan,
    MFLI_PHASE_CALIBRATION_DESCRIPTION,
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, LIST_FIELDS,
    build_summary, compute_filename_preview,
)
from instruments.data_naming import (
    TEST_SAMPLE,
)
from web import run_manager
from web.run_controller import (
    RunController, num_field, text_field, textarea_field, bool_switch,
    render_summary, busy_banner, is_busy,
    param_card, stable_card, param_grid, stable_grid, advanced_section, measurement_layout,
    program_artifacts, program_run_fn, refresh_on_busy_change,
    finished_handler, load_settings, save_settings,
)
from web.directory_picker import validate_directory
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "mfli_phase_calibration_web_settings.json"

PAGE_TITLE = "MFLI Phase Calibration"
SUITE = "MFLI"


def _parse_float_list(raw: str) -> tuple[list[float], list[str]]:
    """Same manual comma-split-and-skip-blanks parsing as the TUI's LIST_FIELDS
    handling (distinct from dc_sweep_utils.parse_value_list, which errors on
    an all-blank string instead of silently returning [])."""
    values: list[float] = []
    errors: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            errors.append(f"'{part}' is not a number")
    return values, errors


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client  # has slot context now; reused by the detached status/comment task
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(MFLI_PHASE_CALIBRATION_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

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
                with param_card("Field sweep & calibration point"):
                    inputs["calibration_current_A"] = num_field(
                        "Calibration magnet current (A)", float(d("calibration_current_A")),
                        hint="Where the 1f Y-null is performed — pick a point near saturation "
                             "(e.g. matching the sweep max).")
                    inputs["sweep_rows"] = textarea_field(
                        "Sweep rows: start, stop, points (one per line)",
                        d("sweep_rows"),
                        hint="Adjacent rows sharing a boundary value are merged, not duplicated.")

                with param_card("Excitation"):
                    inputs["frequency_Hz"] = num_field(
                        "Excitation frequency (Hz)", float(d("frequency_Hz")),
                        hint="Avoid exact multiples of 50/60 Hz (mains pickup).")
                    inputs["amplitude_V"] = num_field("Output amplitude (V, peak)", float(d("amplitude_V")))
                    inputs["series_R_ohm"] = num_field(
                        "Series resistor (Ω)", float(d("series_R_ohm")), hint="Sets excitation current: I ≈ V / R.")

                with param_card("Amplitude check (optional)"):
                    switches["enable_amplitude_check"] = bool_switch("Run current-amplitude scaling check", d("enable_amplitude_check"))
                    inputs["amplitudes_V"] = text_field(
                        "Amplitudes to test (V, comma-separated)", d("amplitudes_V"),
                        hint="≥ 2 values. Checks whether the 2f signal scales linearly with drive current.")

                with param_card("Frequency check (optional)"):
                    switches["enable_frequency_check"] = bool_switch("Run frequency scaling check", d("enable_frequency_check"))
                    inputs["frequencies_Hz"] = text_field(
                        "Frequencies to test (Hz, comma-separated)", d("frequencies_Hz"),
                        hint="≥ 2 values. Checks whether the optimal 1f phase scales linearly with frequency.")

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
                        inputs["input_range_1f_V"] = num_field("1f input range (V)", float(d("input_range_1f_V")))
                        inputs["input_range_2f_V"] = num_field("2f input range (V)", float(d("input_range_2f_V")))
                        inputs["sample_rate_Hz"] = num_field("Demodulator sample rate (Sa/s)", float(d("sample_rate_Hz")))

                    with param_card("Phase null"):
                        inputs["null_n_averages"] = num_field("Averages per phase read", float(d("null_n_averages")), integer=True)
                        inputs["null_max_iterations"] = num_field("Max null iterations", float(d("null_max_iterations")), integer=True)
                        inputs["null_tol_deg"] = num_field("Convergence tolerance (°)", float(d("null_tol_deg")))

                    with param_card("Sweep timing & hold check"):
                        inputs["sweep_settling_time_s"] = num_field(
                            "Settling time per sweep point (s)", float(d("sweep_settling_time_s")),
                            hint="Rule of thumb: ≥ 5 × time constant.")
                        inputs["sweep_n_averages"] = num_field(
                            "Samples to average per sweep point", float(d("sweep_n_averages")), integer=True)
                        inputs["hold_tol_ratio"] = num_field(
                            "Max acceptable |Y|/R away from the calibration point", float(d("hold_tol_ratio")),
                            hint="Flags drift if the null residual exceeds this anywhere in the sweep.")
                        ui.label(
                            "Nulls the leader's 1f Y quadrature by adjusting its demod phaseshift node "
                            "(the same thing LabOne's \"Auto\" phase button does)."
                        ).classes("text-xs text-grey-6")

                    with param_card("Scaling-check advanced"):
                        inputs["amp_n_averages"] = num_field("Averages per amplitude point", float(d("amp_n_averages")), integer=True)
                        inputs["freq_n_averages"] = num_field("Averages per phase read", float(d("freq_n_averages")), integer=True)
                        inputs["freq_max_iterations"] = num_field("Max null iterations per frequency", float(d("freq_max_iterations")), integer=True)
                        inputs["freq_tol_deg"] = num_field("Convergence tolerance per frequency (°)", float(d("freq_tol_deg")))

            # ── Tier 3: instrument wiring & safety — collapsed ──────────────
            with advanced_section("Instrument configuration & addresses", icon="settings"):
                with stable_grid():
                    with stable_card("Devices & connection"):
                        inputs["leader_device"] = text_field("Leader MFLI (current source + 1f)", d("leader_device"))
                        inputs["follower_device"] = text_field("Follower MFLI (2f)", d("follower_device"))
                        inputs["daq_host"] = text_field("LabOne data server host", d("daq_host"))
                        inputs["daq_port"] = num_field("LabOne data server port", float(d("daq_port")), integer=True)

                    with stable_card("Magnet & ramp safety"):
                        inputs["visa_resource"] = text_field("Magnet VISA resource", d("visa_resource"))
                        inputs["current_limit_A"] = num_field(
                            "Software current limit (A)", float(d("current_limit_A")),
                            hint="Hard safety ceiling — independent of the supply's own range.")
                        inputs["voltage_compliance_V"] = num_field("Voltage compliance (V)", float(d("voltage_compliance_V")))
                        inputs["ramp_step_A"] = num_field("Ramp step (A)", float(d("ramp_step_A")))
                        inputs["ramp_delay_s"] = num_field("Ramp delay (s)", float(d("ramp_delay_s")))

                    with stable_card("Gaussmeter"):
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

        with regions.summary:
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start calibration", color="primary").classes("w-full")

        with regions.output:
            with ui.row().classes("w-full items-center gap-3"):
                run_label = ui.label("").classes("text-sm font-bold text-grey-6")
                status_label = ui.label("Idle.").classes("text-sm font-bold")
            abort_btn = ui.button("Abort (safe ramp-down)", color="negative").props("outline")
            abort_btn.set_visibility(False)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig.update_yaxes(title_text="1f  |Y| / R  (null residual)", type="log", row=1, col=1)
            fig.update_yaxes(title_text="2f (V)", row=2, col=1)
            fig.update_xaxes(title_text="Magnetic field (mT)", row=2, col=1)
            fig.update_layout(margin=dict(l=60, r=20, t=20, b=50), showlegend=True)
            fig.add_scatter(x=[], y=[], mode="lines+markers", name="1f |Y|/R", line=dict(color="#d62728"), row=1, col=1)
            fig.add_scatter(x=[], y=[], mode="lines+markers", name="X2f", line=dict(color="#1f77b4"), row=2, col=1)
            fig.add_scatter(x=[], y=[], mode="lines+markers", name="Y2f", line=dict(color="#ff7f0e"), row=2, col=1)
            with ui.element("div").classes("w-full").style("aspect-ratio: 1 / 2; max-height: 90vh"):
                plot = ui.plotly(fig).classes("w-full h-full")

            columns = [
                {"name": "n", "label": "#", "field": "n"},
                {"name": "I", "label": "I (A)", "field": "I"},
                {"name": "B", "label": "B (mT)", "field": "B"},
                {"name": "resid", "label": "1f |Y|/R", "field": "resid"},
                {"name": "X2f", "label": "2f X (V)", "field": "X2f"},
                {"name": "Y2f", "label": "2f Y (V)", "field": "Y2f"},
                {"name": "T1", "label": "T1 (K)", "field": "T1"},
                {"name": "T2", "label": "T2 (K)", "field": "T2"},
            ]
            table = ui.table(columns=columns, rows=[], row_key="n").classes("w-full").props("dense")
            log_area = ui.log(max_lines=2000).classes("w-full h-48 font-mono text-xs")
            ui.label("Report").classes("text-lg font-bold mt-3")
            report_area = ui.label("No report yet — run a calibration to see one.").classes(
                "w-full font-mono text-xs whitespace-pre-wrap bg-grey-2 dark:bg-grey-9 rounded p-2")

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
        for fid in LIST_FIELDS:
            values, list_errors = _parse_float_list(inputs[fid].value or "")
            state[fid] = values
            errors += [f"'{fid}': {e}" for e in list_errors]
        state["temperature_setpoint_K"] = identity.temperature_input.value
        state["device"] = (identity.device_input.value or "").strip()
        state["cooldown"] = (identity.cooldown_input.value or "").strip()
        for fid, sw in switches.items():
            state[fid] = sw.value
        state["order"] = int(order_select.value)
        sample_value = identity.sample_dropdown.value
        state["sample"] = sample_value if sample_value not in (None, NEW_SAMPLE_SENTINEL) else ""

        state["sweep_rows"] = inputs["sweep_rows"].value or ""
        state["sweep_rows_parsed"] = []
        state["sweep_rows_parse_error"] = None
        try:
            state["sweep_rows_parsed"] = parse_sweep_rows(state["sweep_rows"])
        except ValueError as exc:
            state["sweep_rows_parse_error"] = str(exc)

        return state, errors

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        raw["temperature_setpoint_K"] = identity.temperature_input.value if identity.temperature_input.value is not None else ""
        raw["device"] = identity.device_input.value
        raw["cooldown"] = identity.cooldown_input.value
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["order"] = order_select.value
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
            f"File:  {preview}" if preview else "File:  (choose a sample and device to preview the filename)")
        if dir_warning:
            warnings = warnings + [dir_warning]
        if dir_error:
            errors = errors + [dir_error]
        with summary_box:
            summary_box.clear()
            render_summary(info, warnings, errors)
        start_btn.set_enabled(not errors and not is_busy())

    for inp in list(inputs.values()) + [identity.device_input, identity.cooldown_input,
                                         identity.temperature_input, identity.data_dir_input,
                                         identity.sample_dropdown]:
        inp.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    order_select.on_value_change(refresh_summary.refresh)
    refresh_summary()
    refresh_on_busy_change(refresh_summary.refresh)

    def on_record(record: dict) -> None:
        has_field = record.get("magnet_field_mT") is not None
        x = record["magnet_field_mT"] if has_field else record["point_index"]
        fig.data[0].x = fig.data[0].x + (x,)
        fig.data[0].y = fig.data[0].y + (record["1f_residual_ratio"],)
        fig.data[1].x = fig.data[1].x + (x,)
        fig.data[1].y = fig.data[1].y + (record["2f_X_V"],)
        fig.data[2].x = fig.data[2].x + (x,)
        fig.data[2].y = fig.data[2].y + (record["2f_Y_V"],)
        table.rows.append({
            "n": record["point_index"] + 1,
            "I": f"{record['magnet_current_A']:.4f}",
            "B": f"{record['magnet_field_mT']:.2f}" if record.get("magnet_field_mT") is not None else "—",
            "resid": f"{record['1f_residual_ratio']:.2e}" if record.get("1f_residual_ratio") is not None else "—",
            "X2f": f"{record['2f_X_V']:.4e}", "Y2f": f"{record['2f_Y_V']:.4e}",
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
            program, plan, run_contexts, run_extras,
            done_text="Calibration complete — see report below.",
            aborted_text="Calibration aborted.",
            on_result=lambda report: report_area.set_text(format_report(report)))

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
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[plan.output_csv],
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
        report_area.set_text("Running …")
        abort_btn.set_visibility(True)
        start_btn.set_enabled(False)

    def on_abort() -> None:
        if controller["c"] is not None:
            controller["c"].abort()

    start_btn.on_click(on_start)
    abort_btn.on_click(on_abort)
