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

import json
from pathlib import Path
from typing import Optional

from plotly.subplots import make_subplots
from nicegui import background_tasks, ui

from mfli.mfli_dual_harmonic import (
    DemodConfig, FilterConfig, GaussmeterConfig, MagnetConfig, OutputConfig,
    TemperatureControllerConfig, configure_demodulator, configure_output, connect,
    connect_device, connect_gaussmeter, connect_magnet, connect_temperature_controller,
    setup_mds, shutdown_gaussmeter, shutdown_magnet, shutdown_output,
    shutdown_temperature_controller, sync_follower_oscillator,
)
from mfli.mfli_phase_calibration import (
    AmplitudeCheckConfig, FrequencyCheckConfig, PhaseCalibrationReport, SweepConfig,
    format_report, run_phase_calibration,
)
from mfli.mfli_phase_calibration_tui import (
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, LIST_FIELDS,
    MEASUREMENT_TYPE, CalibrationPlan, build_header_fields, build_summary, parse_sensor_uids,
    compute_filename_preview,
)
from instruments.data_naming import (
    TEST_SAMPLE, allocate_run, finalize_index_row, make_incremental_writer,
    preview_raw_filename, proc_path, write_record,
)
from web.run_controller import (
    RunController, RunCallbacks, FinalStatus, num_field, text_field, bool_switch,
    render_summary, busy_banner, is_busy,
    param_card, stable_card, param_grid, stable_grid, section_title,
)
from web.directory_picker import validate_directory
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, status_comment_dialog

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "mfli_phase_calibration_web_settings.json"

PAGE_TITLE = "MFLI Phase Calibration"
SUITE = "MFLI"

MFLI_PHASE_CALIBRATION_DESCRIPTION = (
    "Calibrates the leader's 1f reference phase against the sample's own resistive "
    "Hall response (rather than a separate standard resistor), then verifies the "
    "result before you trust it: checks the null holds across a full field sweep, "
    "and empirically identifies which of X2f/Y2f carries the real signal. Optional "
    "current-amplitude and frequency scaling checks help separate a genuine "
    "resistive/SOT signal from Joule-heating/anomalous-Nernst contamination. Run "
    "this before Dual-Harmonic Measurement, using the same wiring."
)


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


def build_plan(state: dict) -> CalibrationPlan:
    out_cfg = OutputConfig(
        device=state["leader_device"], frequency_Hz=state["frequency_Hz"],
        amplitude_V=state["amplitude_V"], series_R_ohm=state["series_R_ohm"],
    )
    filt = FilterConfig(
        time_constant_s=state["time_constant_s"], order=int(state["order"]),
        sinc_filter=state["sinc_filter"],
    )
    demod1_cfg = DemodConfig(
        device=state["leader_device"], demod_index=0, harmonic=1,
        input_range_V=state["input_range_1f_V"], sample_rate_Hz=state["sample_rate_Hz"], filter=filt,
    )
    demod2_cfg = DemodConfig(
        device=state["follower_device"], demod_index=0, harmonic=2,
        input_range_V=state["input_range_2f_V"], sample_rate_Hz=state["sample_rate_Hz"], filter=filt,
    )
    magnet_cfg = MagnetConfig(
        visa_resource=state["visa_resource"], current_limit_A=state["current_limit_A"],
        voltage_compliance_V=state["voltage_compliance_V"], ramp_step_A=state["ramp_step_A"],
        ramp_delay_s=state["ramp_delay_s"],
    )
    gauss_cfg = GaussmeterConfig(
        visa_resource=state["gaussmeter_visa_resource"], n_averages=int(state["gaussmeter_n_averages"]),
        read_delay_s=state["gaussmeter_read_delay_s"],
    )
    sweep_cfg = SweepConfig(
        i_min_A=state["i_min_A"], i_max_A=state["i_max_A"], n_points=int(state["n_points"]),
        settling_time_s=state["sweep_settling_time_s"], n_averages=int(state["sweep_n_averages"]),
    )
    amplitude_check_cfg = AmplitudeCheckConfig(
        enabled=state["enable_amplitude_check"], amplitudes_V=state["amplitudes_V"] or [0.05, 0.1],
        n_averages=int(state["amp_n_averages"]),
    )
    frequency_check_cfg = FrequencyCheckConfig(
        enabled=state["enable_frequency_check"], frequencies_Hz=state["frequencies_Hz"] or [13.333, 17.777, 23.333],
        n_averages=int(state["freq_n_averages"]), max_iterations=int(state["freq_max_iterations"]),
        tol_deg=state["freq_tol_deg"],
    )
    run_ctx = allocate_run(
        Path(state["data_dir"]), state["sample"], state["device"], MEASUREMENT_TYPE,
        temperature_setpoint_K=state["temperature_setpoint_K"],
    )
    output_csv = str(run_ctx.raw_path)

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

    header_extra = {
        "excitation_frequency_Hz": state["frequency_Hz"],
        "excitation_amplitude_V": state["amplitude_V"],
        "series_R_ohm": state["series_R_ohm"],
        "calibration_current_A": state["calibration_current_A"],
        "field_sweep_A": [state["i_min_A"], state["i_max_A"], int(state["n_points"])],
        "demod_time_constant_s": state["time_constant_s"],
        "demod_order": int(state["order"]),
    }

    return CalibrationPlan(
        daq_host=state["daq_host"], daq_port=int(state["daq_port"]),
        leader=state["leader_device"], follower=state["follower_device"],
        out_cfg=out_cfg, demod1_cfg=demod1_cfg, demod2_cfg=demod2_cfg,
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, sweep_cfg=sweep_cfg,
        calibration_current_A=state["calibration_current_A"],
        null_n_averages=int(state["null_n_averages"]), null_max_iterations=int(state["null_max_iterations"]),
        null_tol_deg=state["null_tol_deg"], hold_tol_ratio=state["hold_tol_ratio"],
        amplitude_check_cfg=amplitude_check_cfg, frequency_check_cfg=frequency_check_cfg,
        output_csv=output_csv, run_ctx=run_ctx, temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, temp_cfg=temp_cfg,
    )


def _save_diagnostic_png(records: list[dict], png_path: Path) -> None:
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]
    resid = [r["1f_residual_ratio"] for r in records]
    x2f = [r["2f_X_V"] for r in records]
    y2f = [r["2f_Y_V"] for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    ax1.plot(xs, resid, "o-", color="tab:red")
    ax1.set_yscale("log")
    ax2.plot(xs, x2f, "o-", color="tab:blue", label="X2f")
    ax2.plot(xs, y2f, "o-", color="tab:orange", label="Y2f")
    ax1.set_ylabel("1f  |Y| / R  (null residual)")
    ax2.set_ylabel("2f  (V)")
    ax2.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax2.legend(loc="best")
    ax1.set_title("Phase calibration result")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


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
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(MFLI_PHASE_CALIBRATION_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = _load_settings()

    def d(key: str):
        if key in saved:
            return saved[key]
        return DEFAULTS.get(key, "")

    inputs: dict = {}
    switches: dict = {}
    controller: dict[str, Optional[RunController]] = {"c": None}

    _t_default = d("temperature_setpoint_K")
    identity = identity_bar(
        default_data_dir=saved.get("data_dir") or str(_DATA_DIR),
        default_sample=saved.get("sample") or TEST_SAMPLE,
        default_device=d("device"), default_cooldown=d("cooldown"),
        default_temperature_K=float(_t_default) if _t_default not in ("", None) else None,
    )

    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        with ui.column().classes("flex-grow gap-1 max-w-3xl"):
            with param_grid():
                with param_card("Devices"):
                    inputs["leader_device"] = text_field("Leader MFLI (current source + 1f)", d("leader_device"))
                    inputs["follower_device"] = text_field("Follower MFLI (2f)", d("follower_device"))

                with param_card("Excitation"):
                    inputs["frequency_Hz"] = num_field(
                        "Excitation frequency (Hz)", float(d("frequency_Hz")),
                        hint="Avoid exact multiples of 50/60 Hz (mains pickup).")
                    inputs["amplitude_V"] = num_field("Output amplitude (V, peak)", float(d("amplitude_V")))
                    inputs["series_R_ohm"] = num_field(
                        "Series resistor (Ω)", float(d("series_R_ohm")), hint="Sets excitation current: I ≈ V / R.")

                with param_card("Field sweep & calibration point"):
                    inputs["calibration_current_A"] = num_field(
                        "Calibration magnet current (A)", float(d("calibration_current_A")),
                        hint="Where the 1f Y-null is performed — pick a point near saturation "
                             "(e.g. matching the sweep max).")
                    inputs["i_min_A"] = num_field("Sweep current min (A)", float(d("i_min_A")))
                    inputs["i_max_A"] = num_field("Sweep current max (A)", float(d("i_max_A")))
                    inputs["n_points"] = num_field("Points per sweep direction", float(d("n_points")), integer=True)

                with param_card("Phase null"):
                    inputs["null_n_averages"] = num_field("Averages per phase read", float(d("null_n_averages")), integer=True)
                    inputs["null_max_iterations"] = num_field("Max null iterations", float(d("null_max_iterations")), integer=True)
                    inputs["null_tol_deg"] = num_field("Convergence tolerance (°)", float(d("null_tol_deg")))

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

            section_title("Instrument configuration")

            with stable_grid():
                with stable_card("Connection"):
                    inputs["daq_host"] = text_field("LabOne data server host", d("daq_host"))
                    inputs["daq_port"] = num_field("LabOne data server port", float(d("daq_port")), integer=True)

                with stable_card("Lock-in filter"):
                    inputs["time_constant_s"] = num_field(
                        "Filter time constant (s)", float(d("time_constant_s")),
                        hint="Bigger = quieter but slower & longer settling.")
                    order_select = ui.select(list(range(1, 9)), value=int(d("order")), label="Filter order").classes("w-full")
                    switches["sinc_filter"] = bool_switch("Sinc filter (extra harmonic rejection)", d("sinc_filter"))
                    inputs["input_range_1f_V"] = num_field("1f input range (V)", float(d("input_range_1f_V")))
                    inputs["input_range_2f_V"] = num_field("2f input range (V)", float(d("input_range_2f_V")))
                    inputs["sample_rate_Hz"] = num_field("Demodulator sample rate (Sa/s)", float(d("sample_rate_Hz")))

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

                with stable_card("Sweep timing & hold check"):
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

                with stable_card("Scaling-check advanced"):
                    inputs["amp_n_averages"] = num_field("Averages per amplitude point", float(d("amp_n_averages")), integer=True)
                    inputs["freq_n_averages"] = num_field("Averages per phase read", float(d("freq_n_averages")), integer=True)
                    inputs["freq_max_iterations"] = num_field("Max null iterations per frequency", float(d("freq_max_iterations")), integer=True)
                    inputs["freq_tol_deg"] = num_field("Convergence tolerance per frequency (°)", float(d("freq_tol_deg")))

                with stable_card("Temperature controller"):
                    inputs["temperature_visa_resource"] = text_field("MercuryiTC VISA resource", d("temperature_visa_resource"))
                    inputs["temperature_sensor_uids"] = text_field("Sensor board UID(s)", d("temperature_sensor_uids"))

        with ui.column().classes("w-96 gap-2"):
            ui.label("Summary").classes("text-lg font-bold")
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start calibration", color="primary").classes("w-full")

    ui.separator().classes("my-3")
    status_label = ui.label("Idle.").classes("text-sm font-bold")
    abort_btn = ui.button("Abort (safe ramp-down)", color="negative").props("outline")
    abort_btn.set_visibility(False)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.update_yaxes(title_text="1f  |Y| / R  (null residual)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="2f (V)", row=2, col=1)
    fig.update_xaxes(title_text="Magnetic field (mT)", row=2, col=1)
    fig.update_layout(margin=dict(l=60, r=20, t=20, b=50), height=600, showlegend=True)
    fig.add_scatter(x=[], y=[], mode="lines+markers", name="1f |Y|/R", line=dict(color="#d62728"), row=1, col=1)
    fig.add_scatter(x=[], y=[], mode="lines+markers", name="X2f", line=dict(color="#1f77b4"), row=2, col=1)
    fig.add_scatter(x=[], y=[], mode="lines+markers", name="Y2f", line=dict(color="#ff7f0e"), row=2, col=1)
    plot = ui.plotly(fig).classes("w-full")

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
    ui.timer(2.0, refresh_summary.refresh)

    def on_record(record: dict) -> None:
        has_field = record.get("magnet_field_mT") is not None
        x = record["magnet_field_mT"] if has_field else record["point_index"]
        fig.data[0].x = fig.data[0].x + (x,)
        fig.data[0].y = fig.data[0].y + (record["1f_residual_ratio"],)
        fig.data[1].x = fig.data[1].x + (x,)
        fig.data[1].y = fig.data[1].y + (record["2f_X_V"],)
        fig.data[2].x = fig.data[2].x + (x,)
        fig.data[2].y = fig.data[2].y + (record["2f_Y_V"],)
        plot.update()
        table.rows.append({
            "n": record["point_index"] + 1,
            "I": f"{record['magnet_current_A']:.4f}",
            "B": f"{record['magnet_field_mT']:.2f}" if record.get("magnet_field_mT") is not None else "—",
            "resid": f"{record['1f_residual_ratio']:.2e}" if record.get("1f_residual_ratio") is not None else "—",
            "X2f": f"{record['2f_X_V']:.4e}", "Y2f": f"{record['2f_Y_V']:.4e}",
            "T1": f"{record['temperature_1_K']:.3f}" if record.get("temperature_1_K") is not None else "—",
            "T2": f"{record['temperature_2_K']:.3f}" if record.get("temperature_2_K") is not None else "—",
        })
        table.update()

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

    async def _prompt_status_comment(plan: CalibrationPlan, records: list[dict]) -> None:
        result = await status_comment_dialog()
        if result is None:
            return
        status, comment = result
        ctx = plan.run_ctx
        header_fields = build_header_fields(plan, records, status=status, comment=comment)
        try:
            # Never truncate an already-written raw file to an empty stub —
            # only a run that never wrote a point gets a header-only write.
            if records or not ctx.raw_path.exists():
                write_record(ctx.raw_path, records, header_fields)
            finalize_index_row(ctx.sample_dir.parent, ctx.sample, ctx.run_number, header_fields)
        except Exception:
            ui.notify("Could not save final status/comment.", type="negative")

    def make_on_finished(plan: CalibrationPlan):
        def on_finished(final: FinalStatus, result: Optional[PhaseCalibrationReport]) -> None:
            label = {"completed": "Calibration complete — see report below.",
                      "aborted": "Calibration aborted.",
                      "error": f"ERROR: {final.error}"}[final.status]
            status_label.set_text(label)
            abort_btn.set_visibility(False)
            start_btn.set_enabled(not is_busy())
            if result is not None:
                report_area.set_text(format_report(result))
            refresh_summary.refresh()
            handle = controller["c"].handle if controller["c"] is not None else None
            records = list(handle.records) if handle is not None else []
            background_tasks.create(
                _prompt_status_comment(plan, records), name="status_comment_prompt")
        return on_finished

    def _finalize(plan: CalibrationPlan, records: list[dict], result, status: str) -> list[str]:
        """save_artifacts -- runs on the worker thread, right after run_fn
        returns/raises. Writes the outcome-derived header/index row
        UNCONDITIONALLY (never gated on the status/comment dialog above) and
        the final PNG into proc/."""
        ctx = plan.run_ctx
        data_root = ctx.sample_dir.parent
        header_fields = build_header_fields(plan, records, status=status, comment="")
        write_record(ctx.raw_path, records, header_fields)
        finalize_index_row(data_root, ctx.sample, ctx.run_number, header_fields)
        png_path = proc_path(data_root, ctx.sample, ctx.run_str, ctx.device, MEASUREMENT_TYPE, "plot")
        _save_diagnostic_png(records, png_path)
        return [str(ctx.raw_path), str(png_path)]

    def make_run_fn(plan: CalibrationPlan):
        def run_fn(stop_event, cb: RunCallbacks):
            daq = magnet = gaussmeter = temp_ctrl = None
            try:
                cb.on_status("Connecting to LabOne data server …")
                daq = connect(plan.daq_host, plan.daq_port)
                connect_device(daq, plan.leader, interface="1GbE")
                connect_device(daq, plan.follower, interface="1GbE")

                cb.on_status("Synchronizing MDS …")
                setup_mds(daq, leader=plan.leader, follower=plan.follower)

                cb.on_status("Configuring output & demodulators …")
                configure_output(daq, plan.out_cfg)
                sync_follower_oscillator(daq, plan.out_cfg, plan.follower)
                configure_demodulator(daq, plan.demod1_cfg)
                configure_demodulator(daq, plan.demod2_cfg)

                cb.on_status("Connecting magnet power supply …")
                magnet = connect_magnet(plan.magnet_cfg)
                cb.on_status("Connecting gaussmeter …")
                gaussmeter = connect_gaussmeter(plan.gauss_cfg)

                if plan.temp_cfg is not None:
                    cb.on_status("Connecting to MercuryiTC (temperature) …")
                    temp_ctrl = connect_temperature_controller(plan.temp_cfg)

                writer = make_incremental_writer(
                    plan.run_ctx.raw_path,
                    lambda records: build_header_fields(plan, records, status="in_progress", comment=""),
                )
                return run_phase_calibration(
                    daq, plan.leader, plan.follower, plan.out_cfg, plan.demod1_cfg, plan.demod2_cfg,
                    plan.sweep_cfg, magnet, plan.magnet_cfg,
                    calibration_current_A=plan.calibration_current_A,
                    null_n_averages=plan.null_n_averages, null_max_iterations=plan.null_max_iterations,
                    null_tol_deg=plan.null_tol_deg, output_csv=plan.output_csv,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg, hold_tol_ratio=plan.hold_tol_ratio,
                    amplitude_check_cfg=plan.amplitude_check_cfg, frequency_check_cfg=plan.frequency_check_cfg,
                    stop_event=stop_event, on_point=cb.on_point, on_status=cb.on_status,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                    write_csv=lambda df: writer(df.to_dict("records")),
                )
            finally:
                if magnet is not None:
                    shutdown_magnet(magnet, plan.magnet_cfg)
                if gaussmeter is not None:
                    shutdown_gaussmeter(gaussmeter)
                if temp_ctrl is not None:
                    shutdown_temperature_controller(temp_ctrl)
                if daq is not None:
                    shutdown_output(daq, plan.out_cfg)
        return run_fn

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

        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE, run_fn=make_run_fn(plan),
            save_artifacts=lambda records, result, status: _finalize(plan, records, result, status),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[plan.output_csv],
            on_record=on_record, on_status=on_status, on_log=on_log,
            on_finished=make_on_finished(plan),
            sample=plan.run_ctx.sample, device=plan.run_ctx.device,
            run_number=plan.run_ctx.run_number,
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
