#!/usr/bin/env python3
"""
NiceGUI page for mfli_phase_calibration.py
================================================
Web equivalent of mfli_phase_calibration_tui.py. Reuses that TUI module's
own DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/LIST_FIELDS/build_summary() (which
itself references format_si/format_duration/_acquire_duration_s imported
from mfli_dual_harmonic_tui inside that module's own globals — reusing the
function object picks those up automatically, no need to re-import them
here).

This is the one page that doesn't fit the plain run_measurement()->DataFrame
pattern every other page follows: the orchestrator is
run_phase_calibration() -> PhaseCalibrationReport, with a third callback
(on_status, alongside stop_event/on_point) already supported directly by
RunCallbacks. finalize (on_finished) renders format_report(report) into a
dedicated Report panel in addition to the log.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from plotly.subplots import make_subplots
from nicegui import ui

_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "instruments"
_MFLI_DIR = Path(__file__).resolve().parent.parent.parent / "MFLI"
_WEB_DIR = Path(__file__).resolve().parent.parent
for _p in (_INSTRUMENTS_DIR, _MFLI_DIR, _WEB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mfli_dual_harmonic import (  # noqa: E402
    DemodConfig, FilterConfig, GaussmeterConfig, MagnetConfig, OutputConfig,
    TemperatureControllerConfig, configure_demodulator, configure_output, connect,
    connect_device, connect_gaussmeter, connect_magnet, connect_temperature_controller,
    setup_mds, shutdown_gaussmeter, shutdown_magnet, shutdown_output,
    shutdown_temperature_controller, sync_follower_oscillator,
)
from mfli_phase_calibration import (  # noqa: E402
    AmplitudeCheckConfig, FrequencyCheckConfig, PhaseCalibrationReport, SweepConfig,
    format_report, run_phase_calibration,
)
from output_paths import build_output_path  # noqa: E402
from mfli_phase_calibration_tui import (  # noqa: E402
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, LIST_FIELDS, build_summary, parse_sensor_uids,
)
from run_controller import (  # noqa: E402
    RunController, RunCallbacks, FinalStatus, num_field, text_field, bool_switch,
    render_summary, busy_banner, is_busy,
)
from directory_picker import directory_field, validate_directory  # noqa: E402

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


@dataclass
class CalibrationPlan:
    daq_host: str
    daq_port: int
    leader: str
    follower: str
    out_cfg: OutputConfig
    demod1_cfg: DemodConfig
    demod2_cfg: DemodConfig
    magnet_cfg: MagnetConfig
    gauss_cfg: GaussmeterConfig
    sweep_cfg: SweepConfig
    calibration_current_A: float
    null_n_averages: int
    null_max_iterations: int
    null_tol_deg: float
    hold_tol_ratio: float
    amplitude_check_cfg: Optional[AmplitudeCheckConfig]
    frequency_check_cfg: Optional[FrequencyCheckConfig]
    output_csv: str
    temp_cfg: Optional[TemperatureControllerConfig] = None

    @property
    def total_points(self) -> int:
        return max(0, 2 * self.sweep_cfg.n_points - 1)


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
    output_csv = str(build_output_path(
        Path(state["data_dir"]), "", state["output_name"], f"{datetime.now():%Y%m%d_%H%M%S}"))

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

    return CalibrationPlan(
        daq_host=state["daq_host"], daq_port=int(state["daq_port"]),
        leader=state["leader_device"], follower=state["follower_device"],
        out_cfg=out_cfg, demod1_cfg=demod1_cfg, demod2_cfg=demod2_cfg,
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, sweep_cfg=sweep_cfg,
        calibration_current_A=state["calibration_current_A"],
        null_n_averages=int(state["null_n_averages"]), null_max_iterations=int(state["null_max_iterations"]),
        null_tol_deg=state["null_tol_deg"], hold_tol_ratio=state["hold_tol_ratio"],
        amplitude_check_cfg=amplitude_check_cfg, frequency_check_cfg=frequency_check_cfg,
        output_csv=output_csv, temp_cfg=temp_cfg,
    )


def _save_diagnostic_png(records: list[dict], csv_path: str) -> list[str]:
    if not records:
        return [csv_path]
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

    png_path = Path(csv_path).with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return [csv_path, str(png_path)]


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

    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        with ui.column().classes("flex-grow gap-1 max-w-3xl"):
            with ui.expansion("Devices", value=True, icon="cable").classes("w-full"):
                inputs["leader_device"] = text_field("Leader MFLI (current source + 1f)", d("leader_device"))
                inputs["follower_device"] = text_field("Follower MFLI (2f)", d("follower_device"))
                with ui.expansion("Connection (advanced)"):
                    inputs["daq_host"] = text_field("LabOne data server host", d("daq_host"))
                    inputs["daq_port"] = num_field("LabOne data server port", float(d("daq_port")), integer=True)

            with ui.expansion("Excitation (current source)", value=True, icon="bolt").classes("w-full"):
                inputs["frequency_Hz"] = num_field(
                    "Excitation frequency (Hz)", float(d("frequency_Hz")),
                    hint="Avoid exact multiples of 50/60 Hz (mains pickup).")
                inputs["amplitude_V"] = num_field("Output amplitude (V, peak)", float(d("amplitude_V")))
                inputs["series_R_ohm"] = num_field(
                    "Series resistor (Ω)", float(d("series_R_ohm")), hint="Sets excitation current: I ≈ V / R.")

            with ui.expansion("Lock-in filters & inputs", value=True, icon="filter_alt").classes("w-full"):
                inputs["time_constant_s"] = num_field(
                    "Filter time constant (s)", float(d("time_constant_s")),
                    hint="Bigger = quieter but slower & longer settling.")
                order_select = ui.select(list(range(1, 9)), value=int(d("order")), label="Filter order").classes("w-full")
                switches["sinc_filter"] = bool_switch("Sinc filter (extra harmonic rejection)", d("sinc_filter"))
                inputs["input_range_1f_V"] = num_field("1f input range (V)", float(d("input_range_1f_V")))
                inputs["input_range_2f_V"] = num_field("2f input range (V)", float(d("input_range_2f_V")))
                inputs["sample_rate_Hz"] = num_field("Demodulator sample rate (Sa/s)", float(d("sample_rate_Hz")))

            with ui.expansion("Magnet", value=True, icon="explore").classes("w-full"):
                inputs["visa_resource"] = text_field("Magnet VISA resource", d("visa_resource"))
                inputs["current_limit_A"] = num_field(
                    "Software current limit (A)", float(d("current_limit_A")),
                    hint="Hard safety ceiling — independent of the supply's own range.")
                inputs["voltage_compliance_V"] = num_field("Voltage compliance (V)", float(d("voltage_compliance_V")))
                with ui.expansion("Ramp safety (advanced)"):
                    inputs["ramp_step_A"] = num_field("Ramp step (A)", float(d("ramp_step_A")))
                    inputs["ramp_delay_s"] = num_field("Ramp delay (s)", float(d("ramp_delay_s")))

            with ui.expansion("Gaussmeter", value=True, icon="speed").classes("w-full"):
                inputs["gaussmeter_visa_resource"] = text_field(
                    "Gaussmeter VISA resource", d("gaussmeter_visa_resource"),
                    hint="Lake Shore 475 — measures the actual field at each point.")
                with ui.expansion("Gaussmeter averaging (advanced)"):
                    inputs["gaussmeter_n_averages"] = num_field(
                        "Field readings averaged per point", float(d("gaussmeter_n_averages")), integer=True)
                    inputs["gaussmeter_read_delay_s"] = num_field(
                        "Delay between readings (s)", float(d("gaussmeter_read_delay_s")))

            with ui.expansion("Field sweep & calibration point", value=True, icon="tune").classes("w-full"):
                inputs["calibration_current_A"] = num_field(
                    "Calibration magnet current (A)", float(d("calibration_current_A")),
                    hint="Where the 1f Y-null is performed — pick a point near saturation "
                         "(e.g. matching the sweep max).")
                inputs["i_min_A"] = num_field("Sweep current min (A)", float(d("i_min_A")))
                inputs["i_max_A"] = num_field("Sweep current max (A)", float(d("i_max_A")))
                inputs["n_points"] = num_field("Points per sweep direction", float(d("n_points")), integer=True)
                inputs["sweep_settling_time_s"] = num_field(
                    "Settling time per sweep point (s)", float(d("sweep_settling_time_s")),
                    hint="Rule of thumb: ≥ 5 × time constant.")
                inputs["sweep_n_averages"] = num_field(
                    "Samples to average per sweep point", float(d("sweep_n_averages")), integer=True)
                with ui.expansion("Hold-check tolerance (advanced)"):
                    inputs["hold_tol_ratio"] = num_field(
                        "Max acceptable |Y|/R away from the calibration point", float(d("hold_tol_ratio")),
                        hint="Flags drift if the null residual exceeds this anywhere in the sweep.")

            with ui.expansion("Phase null", value=True, icon="rule").classes("w-full"):
                inputs["null_n_averages"] = num_field("Averages per phase read", float(d("null_n_averages")), integer=True)
                inputs["null_max_iterations"] = num_field("Max null iterations", float(d("null_max_iterations")), integer=True)
                inputs["null_tol_deg"] = num_field("Convergence tolerance (°)", float(d("null_tol_deg")))
                ui.label(
                    "Nulls the leader's 1f Y quadrature by adjusting its demod phaseshift node "
                    "(the same thing LabOne's \"Auto\" phase button does)."
                ).classes("text-xs text-grey-6")

            with ui.expansion("Optional: amplitude & frequency scaling checks", value=False, icon="science").classes("w-full"):
                switches["enable_amplitude_check"] = bool_switch("Run current-amplitude scaling check", d("enable_amplitude_check"))
                inputs["amplitudes_V"] = text_field(
                    "Amplitudes to test (V, comma-separated)", d("amplitudes_V"),
                    hint="≥ 2 values. Checks whether the 2f signal scales linearly with drive current.")
                inputs["amp_n_averages"] = num_field("Averages per amplitude point", float(d("amp_n_averages")), integer=True)

                switches["enable_frequency_check"] = bool_switch("Run frequency scaling check", d("enable_frequency_check"))
                inputs["frequencies_Hz"] = text_field(
                    "Frequencies to test (Hz, comma-separated)", d("frequencies_Hz"),
                    hint="≥ 2 values. Checks whether the optimal 1f phase scales linearly with frequency.")
                inputs["freq_n_averages"] = num_field("Averages per phase read", float(d("freq_n_averages")), integer=True)
                inputs["freq_max_iterations"] = num_field("Max null iterations per frequency", float(d("freq_max_iterations")), integer=True)
                inputs["freq_tol_deg"] = num_field("Convergence tolerance per frequency (°)", float(d("freq_tol_deg")))

            with ui.expansion("Temperature (MercuryiTC)", value=True, icon="thermostat").classes("w-full"):
                switches["enable_temperature"] = bool_switch(
                    "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))
                inputs["temperature_visa_resource"] = text_field("MercuryiTC VISA resource", d("temperature_visa_resource"))
                inputs["temperature_sensor_uids"] = text_field("Sensor board UID(s)", d("temperature_sensor_uids"))

            with ui.expansion("Output", value=True, icon="save").classes("w-full"):
                inputs["output_name"] = text_field("Output file name (prefix)", d("output_name"))
                data_dir_input = directory_field("Save directory", saved.get("data_dir") or str(_DATA_DIR))

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
            state[fid] = (inputs[fid].value or "").strip()
        for fid in LIST_FIELDS:
            values, list_errors = _parse_float_list(inputs[fid].value or "")
            state[fid] = values
            errors += [f"'{fid}': {e}" for e in list_errors]
        for fid, sw in switches.items():
            state[fid] = sw.value
        state["order"] = int(order_select.value)
        return state, errors

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["order"] = order_select.value
        raw["data_dir"] = data_dir_input.value
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

    for inp in list(inputs.values()) + [data_dir_input]:
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
            save_artifacts=lambda records, result: _save_diagnostic_png(records, plan.output_csv),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[plan.output_csv],
            on_record=on_record, on_status=on_status, on_log=on_log, on_finished=on_finished,
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
