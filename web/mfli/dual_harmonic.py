#!/usr/bin/env python3
"""
NiceGUI page for mfli_dual_harmonic.py
===========================================
Web equivalent of mfli_dual_harmonic_tui.py. Reuses that TUI module's pure
DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/OPTIONAL_NUMERIC_FIELDS/build_summary()/
parse_sensor_uids() so validation stays identical to the TUI.

Optional pre-run phase calibration (auto_null_phase) and sample-geometry
metadata are ported unchanged from RunScreen.do_run() / build_run_metadata()
-- both already flow straight through run_measurement()'s existing
on_point/stop_event hooks with no change needed in mfli_dual_harmonic.py
itself.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from plotly.subplots import make_subplots
from nicegui import ui

_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "instruments"
_MFLI_DIR = Path(__file__).resolve().parent.parent.parent / "MFLI"
_WEB_DIR = Path(__file__).resolve().parent.parent
for _p in (_INSTRUMENTS_DIR, _MFLI_DIR, _WEB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mfli_dual_harmonic import (  # noqa: E402
    AcquisitionConfig, DemodConfig, FilterConfig, GaussmeterConfig, MagnetConfig,
    MeasurementPoint, OutputConfig, SampleGeometryConfig, TemperatureControllerConfig,
    acquire_averaged, auto_null_phase, bidirectional_current_sweep, configure_demodulator,
    configure_output, connect, connect_device, connect_gaussmeter, connect_magnet,
    connect_temperature_controller, run_measurement, set_magnet_current, setup_mds,
    shutdown_gaussmeter, shutdown_magnet, shutdown_output, shutdown_temperature_controller,
    sync_follower_oscillator,
)
from output_paths import build_output_path  # noqa: E402
from mfli_dual_harmonic_tui import (  # noqa: E402
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, OPTIONAL_NUMERIC_FIELDS,
    build_summary, parse_sensor_uids,
)
from run_controller import (  # noqa: E402
    RunController, RunCallbacks, FinalStatus, num_field, text_field, bool_switch,
    optional_num_field, render_summary, busy_banner, is_busy,
)
from directory_picker import directory_field, validate_directory  # noqa: E402

log = logging.getLogger("web.mfli.dual_harmonic")

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "mfli_dual_harmonic_web_settings.json"

PAGE_TITLE = "MFLI Dual-Harmonic Measurement"
SUITE = "MFLI"

MFLI_DUAL_HARMONIC_DESCRIPTION = (
    "Drives an AC current through the sample and reads the 1st-harmonic response "
    "on the leader while the follower reads the 2nd-harmonic response — the "
    "standard setup for e.g. a nonlinear/planar Hall measurement. Optionally "
    "sweeps a Kepco electromagnet's field (bidirectionally, for hysteresis) with "
    "the field measured live via a Lake Shore 475 Gaussmeter at every point."
)


@dataclass
class MeasurementPlan:
    daq_host: str
    daq_port: int
    leader: str
    follower: str
    out_cfg: OutputConfig
    demod1_cfg: DemodConfig
    demod2_cfg: DemodConfig
    acq_cfg: AcquisitionConfig
    magnet_cfg: Optional[MagnetConfig]
    gauss_cfg: Optional[GaussmeterConfig]
    currents_A: Optional[np.ndarray]
    temp_cfg: Optional[TemperatureControllerConfig]
    phase_cal_enabled: bool
    phase_cal_current_A: Optional[float]
    phase_cal_n_averages: int
    phase_cal_max_iterations: int
    geometry_cfg: SampleGeometryConfig

    @property
    def total_points(self) -> int:
        return len(self.currents_A) if self.currents_A is not None else 1


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
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"], n_averages=int(state["n_averages"]),
        output_file=str(build_output_path(
            Path(state["data_dir"]), "", state["output_name"], f"{datetime.now():%Y%m%d_%H%M%S}")),
    )

    magnet_cfg = None
    gauss_cfg = None
    currents_A = None
    if state["enable_sweep"]:
        magnet_cfg = MagnetConfig(
            visa_resource=state["visa_resource"], current_limit_A=state["current_limit_A"],
            voltage_compliance_V=state["voltage_compliance_V"], ramp_step_A=state["ramp_step_A"],
            ramp_delay_s=state["ramp_delay_s"],
        )
        gauss_cfg = GaussmeterConfig(
            visa_resource=state["gaussmeter_visa_resource"], n_averages=int(state["gaussmeter_n_averages"]),
            read_delay_s=state["gaussmeter_read_delay_s"],
        )
        currents_A = bidirectional_current_sweep(
            i_min=state["i_min_A"], i_max=state["i_max_A"], n_points=int(state["n_points"]))

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

    geometry_cfg = SampleGeometryConfig(
        hall_bar_length_um=state["hall_bar_length_um"], hall_bar_width_um=state["hall_bar_width_um"],
        hall_bar_thickness_nm=state["hall_bar_thickness_nm"],
        field_angle_from_oop_deg=state["field_angle_from_oop_deg"],
    )

    return MeasurementPlan(
        daq_host=state["daq_host"], daq_port=int(state["daq_port"]),
        leader=state["leader_device"], follower=state["follower_device"],
        out_cfg=out_cfg, demod1_cfg=demod1_cfg, demod2_cfg=demod2_cfg, acq_cfg=acq_cfg,
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, currents_A=currents_A, temp_cfg=temp_cfg,
        phase_cal_enabled=state["enable_phase_cal"], phase_cal_current_A=state["phase_cal_current_A"],
        phase_cal_n_averages=int(state["phase_cal_n_averages"]),
        phase_cal_max_iterations=int(state["phase_cal_max_iterations"]), geometry_cfg=geometry_cfg,
    )


def _save_measurement_png(records: list[dict], csv_path: str) -> list[str]:
    if not records:
        return [csv_path]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]
    r1 = [r["1f_R_V"] for r in records]
    r2 = [r["2f_R_V"] for r in records]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    ax1.plot(xs, r1, "o-", color="tab:blue")
    ax2.plot(xs, r2, "o-", color="tab:orange")
    ax1.set_ylabel("1f  R (V)"); ax2.set_ylabel("2f  R (V)")
    ax2.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax1.set_title("Measurement result")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = Path(csv_path).with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return [csv_path, str(png_path)]


def page() -> None:
    ui.page_title(PAGE_TITLE)
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(MFLI_DUAL_HARMONIC_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = _load_settings()

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
                inputs["input_range_1f_V"] = num_field(
                    "1f input range (V)", float(d("input_range_1f_V")),
                    hint="Match expected 1f signal size — avoid clipping/poor resolution.")
                inputs["input_range_2f_V"] = num_field(
                    "2f input range (V)", float(d("input_range_2f_V")),
                    hint="2f is usually much smaller than 1f — set separately.")
                inputs["sample_rate_Hz"] = num_field("Demodulator sample rate (Sa/s)", float(d("sample_rate_Hz")))

            with ui.expansion("Acquisition & output", value=True, icon="save").classes("w-full"):
                inputs["settling_time_s"] = num_field(
                    "Settling time per point (s)", float(d("settling_time_s")),
                    hint="Rule of thumb: ≥ 5 × time constant.")
                inputs["n_averages"] = num_field("Samples to average per point", float(d("n_averages")), integer=True)
                inputs["output_name"] = text_field("Output file name (prefix)", d("output_name"))
                data_dir_input = directory_field("Save directory", saved.get("data_dir") or str(_DATA_DIR))

            with ui.expansion("Magnet & field sweep", value=True, icon="explore").classes("w-full"):
                switches["enable_sweep"] = bool_switch("Sweep magnetic field (Kepco magnet)", d("enable_sweep"))
                inputs["visa_resource"] = text_field("Magnet VISA resource", d("visa_resource"))
                inputs["current_limit_A"] = num_field(
                    "Software current limit (A)", float(d("current_limit_A")),
                    hint="Hard safety ceiling — independent of the supply's own range.")
                inputs["voltage_compliance_V"] = num_field("Voltage compliance (V)", float(d("voltage_compliance_V")))
                with ui.expansion("Ramp safety (advanced)"):
                    inputs["ramp_step_A"] = num_field("Ramp step (A)", float(d("ramp_step_A")))
                    inputs["ramp_delay_s"] = num_field("Ramp delay (s)", float(d("ramp_delay_s")))
                inputs["i_min_A"] = num_field("Sweep current min (A)", float(d("i_min_A")))
                inputs["i_max_A"] = num_field("Sweep current max (A)", float(d("i_max_A")))
                inputs["n_points"] = num_field("Points per sweep direction", float(d("n_points")), integer=True)
                inputs["gaussmeter_visa_resource"] = text_field(
                    "Gaussmeter VISA resource", d("gaussmeter_visa_resource"),
                    hint="Lake Shore 475 — measures the actual field at each point.")
                with ui.expansion("Gaussmeter averaging (advanced)"):
                    inputs["gaussmeter_n_averages"] = num_field(
                        "Field readings averaged per point", float(d("gaussmeter_n_averages")), integer=True)
                    inputs["gaussmeter_read_delay_s"] = num_field(
                        "Delay between readings (s)", float(d("gaussmeter_read_delay_s")))

            with ui.expansion("Temperature (MercuryiTC)", value=True, icon="thermostat").classes("w-full"):
                switches["enable_temperature"] = bool_switch(
                    "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))
                inputs["temperature_visa_resource"] = text_field("MercuryiTC VISA resource", d("temperature_visa_resource"))
                inputs["temperature_sensor_uids"] = text_field("Sensor board UID(s)", d("temperature_sensor_uids"))

            with ui.expansion("Phase calibration (Zurich lock-in null)", value=False, icon="tune").classes("w-full"):
                switches["enable_phase_cal"] = bool_switch(
                    "Auto-null 1f phase before run (leader demod phaseshift)", d("enable_phase_cal"))
                optional_inputs["phase_cal_current_A"] = optional_num_field(
                    "Calibration magnet current (A)", opt("phase_cal_current_A"),
                    hint="Blank = null at the present field (no ramp). Otherwise pick a point "
                         "near saturation — e.g. matching i_max. Only used if the field sweep "
                         "above is enabled.")
                inputs["phase_cal_n_averages"] = num_field("Averages per phase read", float(d("phase_cal_n_averages")), integer=True)
                inputs["phase_cal_max_iterations"] = num_field("Max null iterations", float(d("phase_cal_max_iterations")), integer=True)
                ui.label(
                    "Nulls the leader's 1f Y quadrature by adjusting its demod phaseshift node "
                    "(the same thing LabOne's \"Auto\" phase button does). X and Y at 2f are both "
                    "already recorded per point in the CSV — check which one actually tracks field "
                    "there before trusting it."
                ).classes("text-xs text-grey-6")

            with ui.expansion("Sample geometry (optional — for quantitative analysis)", value=False, icon="straighten").classes("w-full"):
                optional_inputs["hall_bar_length_um"] = optional_num_field(
                    "Hall bar length (µm)", opt("hall_bar_length_um"),
                    hint="Current-path length between voltage probes. Leave blank if unknown.")
                optional_inputs["hall_bar_width_um"] = optional_num_field("Hall bar width (µm)", opt("hall_bar_width_um"))
                optional_inputs["hall_bar_thickness_nm"] = optional_num_field("Film/channel thickness (nm)", opt("hall_bar_thickness_nm"))
                optional_inputs["field_angle_from_oop_deg"] = optional_num_field(
                    "External field angle from out-of-plane (°)", opt("field_angle_from_oop_deg"),
                    hint="0° = fully out-of-plane (film normal), 90° = in-plane.")

        with ui.column().classes("w-96 gap-2"):
            ui.label("Summary").classes("text-lg font-bold")
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start measurement", color="primary").classes("w-full")

    ui.separator().classes("my-3")
    status_label = ui.label("Idle.").classes("text-sm font-bold")
    abort_btn = ui.button("Abort (safe ramp-down)", color="negative").props("outline")
    abort_btn.set_visibility(False)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.update_yaxes(title_text="1f  R (V)", row=1, col=1)
    fig.update_yaxes(title_text="2f  R (V)", row=2, col=1)
    fig.update_xaxes(title_text="Magnetic field (mT)", row=2, col=1)
    fig.update_layout(margin=dict(l=60, r=20, t=20, b=50), height=600, showlegend=False)
    fig.add_scatter(x=[], y=[], mode="lines+markers", name="1f R", line=dict(color="#1f77b4"), row=1, col=1)
    fig.add_scatter(x=[], y=[], mode="lines+markers", name="2f R", line=dict(color="#ff7f0e"), row=2, col=1)
    plot = ui.plotly(fig).classes("w-full")

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
        return state, errors

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        for fid, inp in optional_inputs.items():
            raw[fid] = inp.value if inp.value is not None else ""
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

    for inp in list(inputs.values()) + list(optional_inputs.values()) + [data_dir_input]:
        inp.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    order_select.on_value_change(refresh_summary.refresh)
    refresh_summary()
    ui.timer(2.0, refresh_summary.refresh)

    # ── Run wiring ───────────────────────────────────────────────────────

    def on_record(record: dict) -> None:
        has_field = record.get("magnet_field_mT") is not None
        x = record["magnet_field_mT"] if has_field else record["point_index"]
        fig.data[0].x = fig.data[0].x + (x,)
        fig.data[0].y = fig.data[0].y + (record["1f_R_V"],)
        fig.data[1].x = fig.data[1].x + (x,)
        fig.data[1].y = fig.data[1].y + (record["2f_R_V"],)
        plot.update()
        table.rows.append({
            "n": record["point_index"] + 1,
            "I": f"{record['magnet_current_A']:.4f}" if record.get("magnet_current_A") is not None else "—",
            "B": f"{record['magnet_field_mT']:.2f}" if record.get("magnet_field_mT") is not None else "—",
            "R1": f"{record['1f_R_V']:.4e}", "th1": f"{record['1f_theta_deg']:.2f}",
            "R2": f"{record['2f_R_V']:.4e}", "th2": f"{record['2f_theta_deg']:.2f}",
            "T1": f"{record['temperature_1_K']:.3f}" if record.get("temperature_1_K") is not None else "—",
            "T2": f"{record['temperature_2_K']:.3f}" if record.get("temperature_2_K") is not None else "—",
        })
        table.update()

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

    def on_finished(final: FinalStatus, result) -> None:
        label = {"completed": "Measurement complete.", "aborted": "Measurement aborted.",
                  "error": f"ERROR: {final.error}"}[final.status]
        status_label.set_text(label)
        abort_btn.set_visibility(False)
        start_btn.set_enabled(not is_busy())
        refresh_summary.refresh()

    def make_run_fn(plan: MeasurementPlan):
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

                if plan.temp_cfg is not None:
                    cb.on_status("Connecting to MercuryiTC (temperature) …")
                    temp_ctrl = connect_temperature_controller(plan.temp_cfg)

                if plan.magnet_cfg is not None and plan.currents_A is not None:
                    cb.on_status("Connecting magnet power supply …")
                    magnet = connect_magnet(plan.magnet_cfg)
                    cb.on_status("Connecting gaussmeter …")
                    gaussmeter = connect_gaussmeter(plan.gauss_cfg)
                    points = [
                        MeasurementPoint(magnet_current_A=I,
                                         set_action=lambda daq, I=I: set_magnet_current(magnet, plan.magnet_cfg, I))
                        for I in plan.currents_A
                    ]
                else:
                    points = [MeasurementPoint()]

                if plan.phase_cal_enabled:
                    cb.on_status("Phase calibration: nulling 1f Y (leader demod phaseshift) …")
                    if magnet is not None and plan.phase_cal_current_A is not None:
                        log.info("Phase calibration: ramping magnet to %.4f A ...", plan.phase_cal_current_A)
                        set_magnet_current(magnet, plan.magnet_cfg, plan.phase_cal_current_A)
                        time.sleep(plan.acq_cfg.settling_time_s)
                    result = auto_null_phase(
                        daq, plan.demod1_cfg, n_averages=plan.phase_cal_n_averages,
                        max_iterations=plan.phase_cal_max_iterations,
                    )
                    if not result.converged:
                        log.warning(
                            "Phase null did not fully converge after %d iteration(s) "
                            "(|Y|/R=%.2e) — check cabling/contacts before trusting the 2f data.",
                            result.iterations, result.residual_ratio,
                        )
                    d2 = acquire_averaged(daq, plan.demod2_cfg, plan.phase_cal_n_averages)
                    log.info(
                        "2f snapshot at calibration point: X=%.4e V  Y=%.4e V  R=%.4e V — "
                        "check which channel carries the structured field dependence in the "
                        "recorded sweep before trusting either one.",
                        d2["x_mean"], d2["y_mean"], d2["r_mean"],
                    )

                cb.on_status("Running measurement …")
                return run_measurement(
                    daq, plan.out_cfg, plan.demod1_cfg, plan.demod2_cfg, plan.acq_cfg, points,
                    stop_event=stop_event, on_point=cb.on_point,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg, geometry_cfg=plan.geometry_cfg,
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
            save_artifacts=lambda records, result: _save_measurement_png(records, plan.acq_cfg.output_file),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[plan.acq_cfg.output_file],
            on_record=on_record, on_status=on_status, on_log=on_log, on_finished=on_finished,
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
