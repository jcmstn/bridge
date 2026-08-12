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

import json
from pathlib import Path
from typing import Optional

from plotly.subplots import make_subplots
from nicegui import background_tasks, ui

from mfli.mfli_diff_resistance_vs_bias import (
    AcquisitionConfig, BiasPoint, DemodConfig, FilterConfig, OutputConfig,
    TemperatureControllerConfig, bidirectional_bias_sweep, configure_demodulator,
    configure_output, connect, connect_device, connect_temperature_controller,
    ramp_bias_to_zero, run_measurement, setup_mds, shutdown_output,
    shutdown_temperature_controller, sync_follower_oscillator,
)
from mfli.mfli_diff_resistance_tui import (
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, OPTIONAL_NUMERIC_FIELDS,
    MEASUREMENT_TYPE, MeasurementPlan, build_header_fields, build_summary, parse_sensor_uids,
)
from instruments.data_naming import (
    TEST_SAMPLE, allocate_run, finalize_index_row, make_incremental_writer,
    preview_raw_filename, proc_path, write_record,
)
from web.run_controller import (
    RunController, RunCallbacks, FinalStatus, num_field, text_field, bool_switch,
    optional_num_field, render_summary, busy_banner, is_busy,
)
from web.directory_picker import directory_field, validate_directory
from web.sample_picker import NEW_SAMPLE_SENTINEL, sample_select, status_comment_dialog

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "mfli_diff_resistance_web_settings.json"

PAGE_TITLE = "MFLI Differential Resistance vs. Bias"
SUITE = "MFLI"

MFLI_DIFF_RESISTANCE_DESCRIPTION = (
    "Superimposes a small AC excitation on top of a DC bias applied to the DUT, "
    "sweeps that DC bias, and records the complex ratio dV/dI at each point — an "
    "'I-V-curve-equivalent' characterization far more informative than a "
    "single-point resistance for anything nonlinear (contacts, tunnel junctions, "
    "diodes, gated 2D systems). No magnet is involved; the bias sweep is the "
    "whole measurement."
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


def build_plan(state: dict) -> MeasurementPlan:
    out_cfg = OutputConfig(
        device=state["leader_device"], frequency_Hz=state["frequency_Hz"],
        ac_amplitude_V=state["ac_amplitude_V"], series_R_ohm=state["series_R_ohm"],
        bias_min_V=state["bias_min_V"], bias_max_V=state["bias_max_V"],
    )
    filt = FilterConfig(
        time_constant_s=state["time_constant_s"], order=int(state["order"]),
        sinc_filter=state["sinc_filter"],
    )
    current_cfg = DemodConfig(
        device=state["leader_device"], label="I (Current Input 1)", demod_index=0, harmonic=1,
        input_ch=0, use_current_input=True, input_range=state["current_input_range_A"],
        sample_rate_Hz=state["sample_rate_Hz"], filter=filt,
    )
    voltage_cfg = DemodConfig(
        device=state["follower_device"], label="V (across DUT)", demod_index=0, harmonic=1,
        input_range=state["voltage_input_range_V"], sample_rate_Hz=state["sample_rate_Hz"], filter=filt,
    )
    run_ctx = allocate_run(
        Path(state["data_dir"]), state["sample"], state["device"], MEASUREMENT_TYPE,
        temperature_setpoint_K=state["temperature_setpoint_K"],
    )
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"], n_averages=int(state["n_averages"]),
        output_file=str(run_ctx.raw_path),
    )

    biases_V = bidirectional_bias_sweep(
        v_min=state["bias_min_V"], v_max=state["bias_max_V"], n_points=int(state["n_points"]))

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

    header_extra = {
        "excitation_frequency_Hz": state["frequency_Hz"],
        "ac_amplitude_V": state["ac_amplitude_V"],
        "series_R_ohm": state["series_R_ohm"],
        "bias_sweep_V": [state["bias_min_V"], state["bias_max_V"], int(state["n_points"])],
        "demod_time_constant_s": state["time_constant_s"],
        "demod_order": int(state["order"]),
        "n_averages": int(state["n_averages"]),
        "settling_time_s": state["settling_time_s"],
    }

    return MeasurementPlan(
        daq_host=state["daq_host"], daq_port=int(state["daq_port"]),
        leader=state["leader_device"], follower=state["follower_device"],
        out_cfg=out_cfg, current_cfg=current_cfg, voltage_cfg=voltage_cfg,
        acq_cfg=acq_cfg, biases_V=biases_V, temp_cfg=temp_cfg,
        run_ctx=run_ctx, temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra,
    )


def _save_measurement_png(records: list[dict], png_path: Path) -> None:
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["bias_V"] for r in records]
    r_diff = [r["R_diff_ohm"] for r in records]
    x_react = [r["X_reactive_ohm"] for r in records]
    phase = [r["Z_phase_deg"] for r in records]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7, 9))
    ax1.plot(xs, r_diff, "o-", color="#2E3192")
    ax2.plot(xs, x_react, "o-", color="#e34948")
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax3.plot(xs, phase, "o-", color="#00AEEF")
    ax1.set_ylabel("R_diff (Ω)"); ax2.set_ylabel("Reactive (Ω)"); ax3.set_ylabel("Phase (deg)")
    ax3.set_xlabel("DC bias (V)")
    ax1.set_title("Measurement result — dV/dI vs. bias")
    for ax in (ax1, ax2, ax3):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def page() -> None:
    ui.page_title(PAGE_TITLE)
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(MFLI_DIFF_RESISTANCE_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = _load_settings()

    def d(key: str):
        if key in saved:
            return saved[key]
        return DEFAULTS.get(key, "")

    inputs: dict = {}
    switches: dict = {}
    optional_inputs: dict = {}
    controller: dict[str, Optional[RunController]] = {"c": None}

    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        with ui.column().classes("flex-grow gap-1 max-w-3xl"):
            with ui.expansion("Devices", value=True, icon="cable").classes("w-full"):
                inputs["leader_device"] = text_field("Leader MFLI (bias + AC excitation, I-sense)", d("leader_device"))
                inputs["follower_device"] = text_field("Follower MFLI (V-sense across DUT)", d("follower_device"))
                with ui.expansion("Connection (advanced)"):
                    inputs["daq_host"] = text_field("LabOne data server host", d("daq_host"))
                    inputs["daq_port"] = num_field("LabOne data server port", float(d("daq_port")), integer=True)

            with ui.expansion("Excitation & bias", value=True, icon="bolt").classes("w-full"):
                inputs["frequency_Hz"] = num_field(
                    "AC excitation frequency (Hz)", float(d("frequency_Hz")),
                    hint="Avoid exact multiples of 50/60 Hz (mains pickup).")
                inputs["ac_amplitude_V"] = num_field(
                    "AC excitation amplitude (V, peak)", float(d("ac_amplitude_V")),
                    hint="Keep small vs. any bias step over which R_diff changes.")
                inputs["series_R_ohm"] = num_field(
                    "Series resistor (Ω)", float(d("series_R_ohm")),
                    hint="Current-limiting/protection resistor — not used to compute I.")
                inputs["bias_min_V"] = num_field("DC bias sweep min (V)", float(d("bias_min_V")))
                inputs["bias_max_V"] = num_field("DC bias sweep max (V)", float(d("bias_max_V")))

            with ui.expansion("Lock-in filters & inputs", value=True, icon="filter_alt").classes("w-full"):
                inputs["time_constant_s"] = num_field(
                    "Filter time constant (s)", float(d("time_constant_s")),
                    hint="Bigger = quieter but slower & longer settling.")
                order_select = ui.select(list(range(1, 9)), value=int(d("order")), label="Filter order").classes("w-full")
                switches["sinc_filter"] = bool_switch("Sinc filter (extra harmonic rejection)", d("sinc_filter"))
                inputs["current_input_range_A"] = num_field(
                    "Current-sense input range (A)", float(d("current_input_range_A")),
                    hint="Leader's Current Input 1 — size to the actual DUT current.")
                inputs["voltage_input_range_V"] = num_field(
                    "Voltage-sense input range (V)", float(d("voltage_input_range_V")),
                    hint="Follower input, across the DUT.")
                inputs["sample_rate_Hz"] = num_field("Demodulator sample rate (Sa/s)", float(d("sample_rate_Hz")))

            with ui.expansion("Acquisition & output", value=True, icon="save").classes("w-full"):
                inputs["settling_time_s"] = num_field(
                    "Settling time per bias point (s)", float(d("settling_time_s")),
                    hint="Rule of thumb: ≥ 5 × time constant.")
                inputs["n_averages"] = num_field(
                    "Samples to average per point (each demod)", float(d("n_averages")), integer=True)

            with ui.expansion("Sample & run identity", value=True, icon="science").classes("w-full"):
                data_dir_input = directory_field(
                    "Data root directory", saved.get("data_dir") or str(_DATA_DIR))
                sample_dropdown, refresh_sample_options = sample_select(
                    lambda: data_dir_input.value, default=saved.get("sample") or TEST_SAMPLE)
                data_dir_input.on_value_change(lambda: refresh_sample_options())
                inputs["device"] = text_field("Device (e.g. HB3, SV2)", d("device"))
                inputs["cooldown"] = text_field("Cooldown (optional)", d("cooldown"))
                _t_default = d("temperature_setpoint_K")
                optional_inputs["temperature_setpoint_K"] = optional_num_field(
                    "Temperature setpoint (K, optional)",
                    float(_t_default) if _t_default not in ("", None) else None,
                    hint="Drives only the filename's T###K token — the header's T_K uses the "
                         "measured temperature when available.")

            with ui.expansion("Bias sweep", value=True, icon="tune").classes("w-full"):
                inputs["n_points"] = num_field(
                    "Points per sweep direction", float(d("n_points")), integer=True,
                    hint="Bidirectional: min → max → min (reveals hysteresis).")

            with ui.expansion("Temperature (MercuryiTC)", value=True, icon="thermostat").classes("w-full"):
                switches["enable_temperature"] = bool_switch(
                    "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))
                inputs["temperature_visa_resource"] = text_field("MercuryiTC VISA resource", d("temperature_visa_resource"))
                inputs["temperature_sensor_uids"] = text_field("Sensor board UID(s)", d("temperature_sensor_uids"))

        with ui.column().classes("w-96 gap-2"):
            ui.label("Summary").classes("text-lg font-bold")
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start measurement", color="primary").classes("w-full")

    ui.separator().classes("my-3")
    status_label = ui.label("Idle.").classes("text-sm font-bold")
    abort_btn = ui.button("Abort (safe ramp-down)", color="negative").props("outline")
    abort_btn.set_visibility(False)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.update_yaxes(title_text="R_diff (Ω)", row=1, col=1)
    fig.update_yaxes(title_text="Reactive (Ω)", row=2, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=3, col=1)
    fig.update_xaxes(title_text="DC bias (V)", row=3, col=1)
    fig.update_layout(margin=dict(l=60, r=20, t=20, b=50), height=760, showlegend=False)
    fig.add_scatter(x=[], y=[], mode="lines+markers", line=dict(color="#2E3192"), row=1, col=1)
    fig.add_scatter(x=[], y=[], mode="lines+markers", line=dict(color="#e34948"), row=2, col=1)
    fig.add_scatter(x=[], y=[], mode="lines+markers", line=dict(color="#00AEEF"), row=3, col=1)
    plot = ui.plotly(fig).classes("w-full")

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
            state[fid] = (inputs[fid].value or "").strip()
        for fid in OPTIONAL_NUMERIC_FIELDS:
            v = optional_inputs[fid].value
            state[fid] = float(v) if v is not None else None
        for fid, sw in switches.items():
            state[fid] = sw.value
        state["order"] = int(order_select.value)
        sample_value = sample_dropdown.value
        state["sample"] = sample_value if sample_value not in (None, NEW_SAMPLE_SENTINEL) else ""
        return state, errors

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        for fid, inp in optional_inputs.items():
            raw[fid] = inp.value if inp.value is not None else ""
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["order"] = order_select.value
        raw["data_dir"] = data_dir_input.value
        sample_value = sample_dropdown.value
        if sample_value not in (None, NEW_SAMPLE_SENTINEL):
            raw["sample"] = sample_value
        return raw

    @ui.refreshable
    def refresh_summary() -> None:
        state, parse_errors = parse_state()
        dir_warning, dir_error = validate_directory(data_dir_input.value or "")
        if parse_errors:
            info, warnings, errors = [], [], parse_errors
        else:
            info, warnings, errors = build_summary(state)
        if dir_warning:
            warnings = warnings + [dir_warning]
        if dir_error:
            errors = errors + [dir_error]
        with summary_box:
            summary_box.clear()
            render_summary(info, warnings, errors)
        start_btn.set_enabled(not errors and not is_busy())

    for inp in list(inputs.values()) + list(optional_inputs.values()) + [data_dir_input, sample_dropdown]:
        inp.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    order_select.on_value_change(refresh_summary.refresh)
    refresh_summary()
    ui.timer(2.0, refresh_summary.refresh)

    def on_record(record: dict) -> None:
        x = record["bias_V"]
        for i, key in enumerate(("R_diff_ohm", "X_reactive_ohm", "Z_phase_deg")):
            fig.data[i].x = fig.data[i].x + (x,)
            fig.data[i].y = fig.data[i].y + (record[key],)
        plot.update()
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
        table.update()

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

    async def _prompt_status_comment(plan: MeasurementPlan, records: list[dict]) -> None:
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

    def make_on_finished(plan: MeasurementPlan):
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
                _prompt_status_comment(plan, records), name="status_comment_prompt")
        return on_finished

    def _finalize(plan: MeasurementPlan, records: list[dict], result, status: str) -> list[str]:
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
        _save_measurement_png(records, png_path)
        return [str(ctx.raw_path), str(png_path)]

    def make_run_fn(plan: MeasurementPlan):
        def run_fn(stop_event, cb: RunCallbacks):
            daq = None
            output_configured = False
            temp_ctrl = None
            try:
                cb.on_status("Connecting to LabOne data server …")
                daq = connect(plan.daq_host, plan.daq_port)
                connect_device(daq, plan.leader, interface="1GbE")
                connect_device(daq, plan.follower, interface="1GbE")

                if plan.temp_cfg is not None:
                    cb.on_status("Connecting to MercuryiTC (temperature) …")
                    temp_ctrl = connect_temperature_controller(plan.temp_cfg)

                cb.on_status("Synchronizing MDS …")
                setup_mds(daq, leader=plan.leader, follower=plan.follower)

                cb.on_status("Configuring output & demodulators …")
                configure_output(daq, plan.out_cfg)
                output_configured = True
                sync_follower_oscillator(daq, plan.out_cfg, plan.follower)
                configure_demodulator(daq, plan.current_cfg)
                configure_demodulator(daq, plan.voltage_cfg)

                points = [BiasPoint(bias_V=float(v)) for v in plan.biases_V]

                cb.on_status("Running measurement …")
                write_csv = make_incremental_writer(
                    plan.run_ctx.raw_path,
                    lambda records: build_header_fields(plan, records, status="in_progress", comment=""),
                )
                return run_measurement(
                    daq, plan.out_cfg, plan.current_cfg, plan.voltage_cfg, plan.acq_cfg, points,
                    stop_event=stop_event, on_point=cb.on_point,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                    write_csv=write_csv,
                )
            finally:
                if daq is not None and output_configured:
                    ramp_bias_to_zero(daq, plan.out_cfg)
                    shutdown_output(daq, plan.out_cfg)
                if temp_ctrl is not None:
                    shutdown_temperature_controller(temp_ctrl)
        return run_fn

    def on_start() -> None:
        state, parse_errors = parse_state()
        dir_warning, dir_error = validate_directory(data_dir_input.value or "")
        if parse_errors or dir_error:
            ui.notify("Fix the blocking issues before starting.", type="negative")
            return
        _, _, errors = build_summary(state)
        if errors:
            ui.notify("Fix the blocking issues before starting.", type="negative")
            return
        state["data_dir"] = data_dir_input.value.strip()

        _save_settings(collect_raw())

        plan = build_plan(state)
        Path(state["data_dir"]).mkdir(parents=True, exist_ok=True)

        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE, run_fn=make_run_fn(plan),
            save_artifacts=lambda records, result, status: _finalize(plan, records, result, status),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[plan.acq_cfg.output_file],
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
        abort_btn.set_visibility(True)
        start_btn.set_enabled(False)

    def on_abort() -> None:
        if controller["c"] is not None:
            controller["c"].abort()

    start_btn.on_click(on_start)
    abort_btn.on_click(on_abort)
