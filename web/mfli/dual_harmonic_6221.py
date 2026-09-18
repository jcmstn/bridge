#!/usr/bin/env python3
"""
NiceGUI page for mfli_dual_harmonic_6221.py
================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-14

Web equivalent of mfli_dual_harmonic_6221_tui.py. Reuses that TUI module's
pure DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/OPTIONAL_NUMERIC_FIELDS/
build_summary()/parse_sensor_uids() so validation stays identical to the TUI.
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import background_tasks, ui

from dc.dc_sweep_utils import build_segmented_sweep, parse_sweep_rows, parse_value_list, safe_shutdown
from mfli.mfli_dual_harmonic_6221 import (
    ACSourceConfig, AcquisitionConfig, DemodConfig, ExtRefConfig, FilterConfig,
    GaussmeterConfig, MagnetConfig, MeasurementPoint, SampleGeometryConfig,
    TemperatureControllerConfig,
    acquire_averaged, auto_null_phase,
    configure_demodulator, configure_external_reference, connect, connect_ac_source,
    connect_device, connect_gaussmeter, connect_magnet, connect_temperature_controller,
    disable_sigout, null_follower_reference_via_1f, run_measurement,
    set_magnet_current, setup_mds, shutdown_ac_source,
    shutdown_gaussmeter, shutdown_magnet, shutdown_temperature_controller,
    wait_for_reference_lock, _check_ac_safety,
)
from mfli.mfli_dual_harmonic_6221_tui import (
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, OPTIONAL_NUMERIC_FIELDS,
    MEASUREMENT_TYPE, MeasurementPlan, build_header_fields, build_summary,
    compute_filename_preview, format_si, parse_sensor_uids, follower_naming,
)
from instruments.data_naming import (
    TEST_SAMPLE, RunContext, allocate_run, finalize_index_row, make_incremental_writer,
    preview_raw_filename, proc_path, write_record,
)
from instruments.field_geometry import field_direction_summary_line
from web.run_controller import (
    RunController, RunCallbacks, FinalStatus, num_field, text_field, textarea_field, bool_switch,
    optional_num_field, render_summary, busy_banner, is_busy,
    param_card, param_grid, stable_card, stable_grid, advanced_section, measurement_layout,
)
from web.directory_picker import validate_directory
from web.field_diagram import build_field_diagram_figure
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, status_comment_dialog

log = logging.getLogger("web.mfli.dual_harmonic_6221")

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "mfli_dual_harmonic_6221_web_settings.json"

PAGE_TITLE = "MFLI Dual-Harmonic Measurement (6221 AC source)"
SUITE = "MFLI"

MFLI_DUAL_HARMONIC_6221_DESCRIPTION = (
    "Same 1f/2f dual-harmonic measurement as the pure-MFLI version, but the AC "
    "excitation current is sourced by a Keithley 6221 (an ideal current source) "
    "instead of an MFLI Signal Output — its Trigger Link phase marker drives "
    "BOTH MFLIs' Aux In 1, and each locks its own oscillator to it (ExtRef). "
    "Filters, magnet field sweep, temperature logging, phase calibration and "
    "sample geometry all match the pure-MFLI version."
)

# extrefs/N/automode options — see ExtRefConfig.automode's docstring in
# mfli_dual_harmonic_6221.py for the full rationale.
AUTOMODE_OPTIONS = {2: "2 — low bandwidth", 3: "3 — high bandwidth", 4: "4 — dynamic (auto)"}
AUTOMODE_HINT = ("2=most forgiving acquisition (marginal/noisy signal), "
                 "3=fastest tracking once locked, 4=auto-adapts (default).")


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
    ac_cfg = ACSourceConfig(
        visa_resource=state["ac_visa_resource"], amplitude_A=state["amplitude_list"][0],
        frequency_Hz=state["frequency_Hz"], compliance_V=state["ac_compliance_V"],
        phasemarker_line=int(state["phasemarker_line"]),
    )
    leader_extref_cfg = ExtRefConfig(
        device=state["leader_device"], extref_index=int(state["leader_extref_index"]),
        aux_input_ch=int(state["leader_aux_input_ch"]), osc_index=int(state["leader_osc_index"]),
        pll_demod_index=int(state["leader_pll_demod_index"]), automode=int(state["leader_automode"]),
    )
    follower_extref_cfg = ExtRefConfig(
        device=state["follower_device"], extref_index=int(state["follower_extref_index"]),
        aux_input_ch=int(state["follower_aux_input_ch"]), osc_index=int(state["follower_osc_index"]),
        pll_demod_index=int(state["follower_pll_demod_index"]), automode=int(state["follower_automode"]),
    )
    filt_1f = FilterConfig(
        time_constant_s=state["time_constant_1f_s"], order=int(state["order_1f"]),
        sinc_filter=state["sinc_filter_1f"],
    )
    filt_2f = FilterConfig(
        time_constant_s=state["time_constant_2f_s"], order=int(state["order_2f"]),
        sinc_filter=state["sinc_filter_2f"],
    )
    demod1_cfg = DemodConfig(
        device=state["leader_device"], demod_index=0, harmonic=1,
        osc_index=int(state["leader_osc_index"]),
        input_range_V=state["input_range_1f_V"], sample_rate_Hz=state["sample_rate_Hz"], filter=filt_1f,
    )
    demod2_cfg = DemodConfig(
        device=state["follower_device"], demod_index=0,
        harmonic=1 if state["measure_rxx"] else 2,
        osc_index=int(state["follower_osc_index"]),
        input_range_V=state["input_range_2f_V"], sample_rate_Hz=state["sample_rate_Hz"], filter=filt_2f,
    )
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"], n_averages=int(state["n_averages"]),
        field_settle_tolerance_mT=state["field_settle_tolerance_mT"],
        output_file="",  # overwritten per amplitude iteration in run_fn
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
        currents_A = build_segmented_sweep(state["sweep_rows_parsed"], bidirectional=True)

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"], sensor_uids=uids)

    geometry_cfg = SampleGeometryConfig(
        hall_bar_length_um=state["hall_bar_length_um"], hall_bar_width_um=state["hall_bar_width_um"],
        hall_bar_thickness_nm=state["hall_bar_thickness_nm"],
        field_theta_deg=state["field_theta_deg"], field_phi_deg=state["field_phi_deg"],
    )

    header_extra = {
        "excitation_frequency_Hz": state["frequency_Hz"],
        "excitation_amplitude_A": state["amplitude_list"][0],
        "measure_rxx": state["measure_rxx"],
        "demod1_time_constant_s": state["time_constant_1f_s"],
        "demod1_order": int(state["order_1f"]),
        "demod2_time_constant_s": state["time_constant_2f_s"],
        "demod2_order": int(state["order_2f"]),
        "n_averages": int(state["n_averages"]),
        "settling_time_s": state["settling_time_s"],
    }
    if state["enable_sweep"]:
        header_extra["field_sweep_rows_A"] = state["sweep_rows_parsed"]

    series = ""
    if len(state["amplitude_list"]) > 1:
        series = (f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_"
                  f"{datetime.now():%Y%m%dT%H%M%S}")

    return MeasurementPlan(
        daq_host=state["daq_host"], daq_port=int(state["daq_port"]),
        leader=state["leader_device"], follower=state["follower_device"],
        ac_cfg=ac_cfg, amplitudes_A=state["amplitude_list"],
        measure_rxx=state["measure_rxx"],
        leader_extref_cfg=leader_extref_cfg, follower_extref_cfg=follower_extref_cfg,
        extref_lock_timeout_s=state["extref_lock_timeout_s"],
        demod1_cfg=demod1_cfg, demod2_cfg=demod2_cfg, acq_cfg=acq_cfg,
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, currents_A=currents_A, temp_cfg=temp_cfg,
        phase_cal_enabled=state["enable_phase_cal"], phase_cal_current_A=state["phase_cal_current_A"],
        phase_cal_n_averages=int(state["phase_cal_n_averages"]),
        phase_cal_max_iterations=int(state["phase_cal_max_iterations"]), geometry_cfg=geometry_cfg,
        sample=state["sample"], device=state["device"], data_root=Path(state["data_dir"]),
        temperature_setpoint_K=state["temperature_setpoint_K"],
        cooldown=state["cooldown"], header_extra=header_extra, series=series,
    )


def _save_measurement_png(records: list[dict], png_path: Path,
                           plan: Optional[MeasurementPlan] = None, comment: str = "") -> None:
    """`plan`/`comment` add a small "at a glance" text annotation -- see
    mfli_dual_harmonic_6221_tui.py's _save_measurement_png for the same
    logic. Called once when the run ends (comment="") and again, to
    overwrite the PNG in place, once the operator's comment is known."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    follower_prefix, follower_display = follower_naming(plan.measure_rxx if plan else False)
    has_field = any(r.get("magnet_field_mT") is not None for r in records)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    cmap = plt.get_cmap("tab10")
    series_ids = sorted({r.get("series_index", 0) for r in records})
    for idx in series_ids:
        rows = [r for r in records if r.get("series_index", 0) == idx]
        label = rows[0].get("series_label")
        xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in rows]
        ax1.plot(xs, [r["1f_R_V"] for r in rows], "o-", color=cmap(idx % 10), label=label)
        ax2.plot(xs, [r[f"{follower_prefix}_R_V"] for r in rows], "o-", color=cmap(idx % 10), label=label)
    ax1.set_ylabel("1f  R (V)"); ax2.set_ylabel(f"{follower_display}  R (V)")
    ax2.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax1.set_title("Measurement result")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    if any(r.get("series_label") for r in records):
        ax1.legend(loc="best", fontsize=8)
    fig.tight_layout()

    lines: list[str] = []
    if plan is not None:
        theta = plan.geometry_cfg.field_theta_deg
        if theta is not None:
            lines.append(field_direction_summary_line(theta, plan.geometry_cfg.field_phi_deg))
        freq_Hz = plan.header_extra.get("excitation_frequency_Hz")
        amps = sorted({r["excitation_current_A_peak"] for r in records
                       if r.get("excitation_current_A_peak") is not None})
        if freq_Hz is not None and len(amps) == 1:
            lines.append(f"AC excitation: {format_si(amps[0], 'A')} @ {format_si(freq_Hz, 'Hz')}")
        tc1, order1 = plan.demod1_cfg.filter.time_constant_s, plan.demod1_cfg.filter.order
        tc2, order2 = plan.demod2_cfg.filter.time_constant_s, plan.demod2_cfg.filter.order
        lines.append(f"Filter: 1f TC={tc1:g} s order={order1}, "
                     f"{follower_display} TC={tc2:g} s order={order2}")
    if comment:
        lines.append(f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}")
    if lines:
        fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color="0.4", va="bottom")
        fig.subplots_adjust(bottom=0.08 + 0.045 * len(lines))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(MFLI_DUAL_HARMONIC_6221_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

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
                with param_card("Excitation (Keithley 6221 AC current source)"):
                    inputs["frequency_Hz"] = num_field(
                        "Excitation frequency (Hz)", float(d("frequency_Hz")),
                        hint="Avoid exact multiples of 50/60 Hz (mains pickup).")
                    inputs["amplitude_values"] = text_field(
                        "Excitation current (A, peak)", d("amplitude_values"),
                        hint="Ideal current source — no series resistor. Single value, or "
                             "comma-separated list — one complete sweep runs per value "
                             "(own 6221 re-arm), each saved to its own file.")
                    inputs["ac_compliance_V"] = num_field(
                        "6221 voltage compliance (V)", float(d("ac_compliance_V")))

                with param_card("Quantities"):
                    switches["measure_rxx"] = bool_switch(
                        "R_xx mode — follower reads R_xx's 1f instead of R_xy's 2f",
                        d("measure_rxx"))
                    ui.label(
                        "Only two physical MFLIs, so this trades 2f for R_xx — move the "
                        "follower's Signal Input cable by hand to match. The '2f lock-in "
                        "filter'/'2f input range' fields below configure the follower "
                        "either way."
                    ).classes("text-xs text-grey-6")

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
                        inputs["leader_device"] = text_field("Leader MFLI (1f)", d("leader_device"))
                        inputs["follower_device"] = text_field(
                            "Follower MFLI (2f, or R_xx 1f if R_xx mode is on)", d("follower_device"))
                        inputs["daq_host"] = text_field("LabOne data server host", d("daq_host"))
                        inputs["daq_port"] = num_field("LabOne data server port", float(d("daq_port")), integer=True)

                    with stable_card("6221 & ExtRef (phase marker → both MFLIs' Aux In)"):
                        inputs["ac_visa_resource"] = text_field("6221 VISA resource", d("ac_visa_resource"))
                        inputs["phasemarker_line"] = num_field(
                            "6221 Trigger Link phase-marker pin", float(d("phasemarker_line")), integer=True,
                            hint="Confirm your unit's factory default before assuming.")
                        inputs["extref_lock_timeout_s"] = num_field(
                            "ExtRef PLL lock timeout (s)", float(d("extref_lock_timeout_s")))
                        inputs["leader_extref_index"] = num_field(
                            "Leader ExtRef module index", float(d("leader_extref_index")), integer=True)
                        inputs["leader_aux_input_ch"] = num_field(
                            "Leader Aux In channel (0 = Aux In 1)", float(d("leader_aux_input_ch")), integer=True)
                        inputs["leader_osc_index"] = num_field(
                            "Leader oscillator index", float(d("leader_osc_index")), integer=True)
                        inputs["leader_pll_demod_index"] = num_field(
                            "Leader PLL phase-detector demod index", float(d("leader_pll_demod_index")),
                            integer=True,
                            hint="Must differ from demod 0 (used for the real 1f signal) — "
                                 "extrefs/N/adcselect is read-only on real firmware, this demod's "
                                 "OWN adcselect is what actually selects Aux In.")
                        leader_automode_select = ui.select(
                            AUTOMODE_OPTIONS, value=int(d("leader_automode")),
                            label="Leader PLL bandwidth adaptation").classes("w-full")
                        ui.label(AUTOMODE_HINT).classes("text-xs text-grey-6 -mt-2 mb-2")
                        inputs["follower_extref_index"] = num_field(
                            "Follower ExtRef module index", float(d("follower_extref_index")), integer=True)
                        inputs["follower_aux_input_ch"] = num_field(
                            "Follower Aux In channel (0 = Aux In 1)", float(d("follower_aux_input_ch")), integer=True)
                        inputs["follower_osc_index"] = num_field(
                            "Follower oscillator index", float(d("follower_osc_index")), integer=True)
                        inputs["follower_pll_demod_index"] = num_field(
                            "Follower PLL phase-detector demod index", float(d("follower_pll_demod_index")),
                            integer=True,
                            hint="Must differ from demod 0 (used for the real follower signal).")
                        follower_automode_select = ui.select(
                            AUTOMODE_OPTIONS, value=int(d("follower_automode")),
                            label="Follower PLL bandwidth adaptation").classes("w-full")
                        ui.label(AUTOMODE_HINT).classes("text-xs text-grey-6 -mt-2 mb-2")

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
            fig.update_layout(margin=dict(l=60, r=20, t=20, b=50), showlegend=True)
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
            v = identity.temperature_input.value if fid == "temperature_setpoint_K" else optional_inputs[fid].value
            state[fid] = float(v) if v is not None else None
        for fid, sw in switches.items():
            state[fid] = sw.value
        state["order_1f"] = int(order_select_1f.value)
        state["order_2f"] = int(order_select_2f.value)
        state["leader_automode"] = int(leader_automode_select.value)
        state["follower_automode"] = int(follower_automode_select.value)
        sample_value = identity.sample_dropdown.value
        state["sample"] = sample_value if sample_value not in (None, NEW_SAMPLE_SENTINEL) else ""

        state["sweep_rows"] = inputs["sweep_rows"].value or ""
        state["sweep_rows_parsed"] = []
        state["sweep_rows_parse_error"] = None
        try:
            state["sweep_rows_parsed"] = parse_sweep_rows(state["sweep_rows"])
        except ValueError as exc:
            state["sweep_rows_parse_error"] = str(exc)

        state["amplitude_list"] = []
        state["amplitude_parse_error"] = None
        try:
            state["amplitude_list"] = parse_value_list(state["amplitude_values"])
        except ValueError as exc:
            state["amplitude_parse_error"] = str(exc)

        return state, errors

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        for fid, inp in optional_inputs.items():
            raw[fid] = inp.value if inp.value is not None else ""
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["order_1f"] = order_select_1f.value
        raw["order_2f"] = order_select_2f.value
        raw["leader_automode"] = leader_automode_select.value
        raw["follower_automode"] = follower_automode_select.value
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
    leader_automode_select.on_value_change(refresh_summary.refresh)
    follower_automode_select.on_value_change(refresh_summary.refresh)
    refresh_summary()
    ui.timer(2.0, refresh_summary.refresh)

    # ── Run wiring ───────────────────────────────────────────────────────

    series_state: dict = {}

    def init_series(n_series: int, labels: list[Optional[str]], measure_rxx: bool = False) -> None:
        follower_prefix, follower_display = follower_naming(measure_rxx)
        series_state["follower_prefix"] = follower_prefix
        fig.data = []
        fig.update_yaxes(title_text=f"{follower_display}  R (V)", row=2, col=1)
        table.columns = [
            {"name": "n", "label": "#", "field": "n"},
            {"name": "I", "label": "I (A)", "field": "I"},
            {"name": "B", "label": "B (mT)", "field": "B"},
            {"name": "R1", "label": "1f R (V)", "field": "R1"},
            {"name": "th1", "label": "1f θ (°)", "field": "th1"},
            {"name": "R2", "label": f"{follower_display} R (V)", "field": "R2"},
            {"name": "th2", "label": f"{follower_display} θ (°)", "field": "th2"},
            {"name": "T1", "label": "T1 (K)", "field": "T1"},
            {"name": "T2", "label": "T2 (K)", "field": "T2"},
        ]
        series_state["traces"] = {}
        cmap = ["#2E3192", "#e34948", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#ff7f0e", "#7f7f7f"]
        for i in range(n_series):
            color = cmap[i % len(cmap)]
            name = labels[i]
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers",
                                      name=f"1f {name}" if name else "1f R",
                                      line=dict(color=color), legendgroup=f"s{i}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers",
                                      name=f"{follower_display} {name}" if name else f"{follower_display} R",
                                      line=dict(color=color, dash="dot"), legendgroup=f"s{i}"),
                          row=2, col=1)
            series_state["traces"][i] = (2 * i, 2 * i + 1)

    def on_record(record: dict) -> None:
        idx = record.get("series_index", 0)
        fp = series_state.get("follower_prefix", "2f")
        t1, t2 = series_state["traces"][idx]
        has_field = record.get("magnet_field_mT") is not None
        x = record["magnet_field_mT"] if has_field else record["point_index"]
        fig.data[t1].x = fig.data[t1].x + (x,)
        fig.data[t1].y = fig.data[t1].y + (record["1f_R_V"],)
        fig.data[t2].x = fig.data[t2].x + (x,)
        fig.data[t2].y = fig.data[t2].y + (record[f"{fp}_R_V"],)
        plot.update()
        table.rows.append({
            "n": record["point_index"] + 1,
            "I": f"{record['magnet_current_A']:.4f}" if record.get("magnet_current_A") is not None else "—",
            "B": f"{record['magnet_field_mT']:.2f}" if record.get("magnet_field_mT") is not None else "—",
            "R1": f"{record['1f_R_V']:.4e}", "th1": f"{record['1f_theta_deg']:.2f}",
            "R2": f"{record[f'{fp}_R_V']:.4e}", "th2": f"{record[f'{fp}_theta_deg']:.2f}",
            "T1": f"{record['temperature_1_K']:.3f}" if record.get("temperature_1_K") is not None else "—",
            "T2": f"{record['temperature_2_K']:.3f}" if record.get("temperature_2_K") is not None else "—",
        })
        table.update()

    def on_status(text: str) -> None:
        status_label.set_text(text)

    def on_run_label(text: str) -> None:
        run_label.set_text(text)

    def on_log(text: str, level: int) -> None:
        log_area.push(text)

    async def _prompt_status_comment(plan: MeasurementPlan, run_contexts: list[RunContext],
                                      records: list[dict]) -> None:
        result = await status_comment_dialog(page_client)
        if result is None:
            return
        status, comment = result
        for series_idx, ctx in enumerate(run_contexts):
            iter_records = [r for r in records if r.get("series_index", 0) == series_idx]
            amp = iter_records[0].get("excitation_current_A_peak") if iter_records else None
            extra = {"excitation_amplitude_A": amp} if amp is not None else None
            header_fields = build_header_fields(
                plan, ctx, iter_records, status=status, comment=comment, extra=extra,
            )
            try:
                if iter_records or not ctx.raw_path.exists():
                    write_record(ctx.raw_path, iter_records, header_fields)
                finalize_index_row(plan.data_root, ctx.sample, ctx.run_number, header_fields)
            except Exception:
                ui.notify("Could not save final status/comment.", type="negative")
        if comment and run_contexts:
            try:
                _save_measurement_png(records, _combined_png_path(run_contexts, plan),
                                       plan=plan, comment=comment)
            except Exception:
                pass

    def make_on_finished(plan: MeasurementPlan, run_contexts: list[RunContext]):
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
                _prompt_status_comment(plan, run_contexts, records), name="status_comment_prompt")
        return on_finished

    def _combined_png_path(run_contexts: list[RunContext], plan: MeasurementPlan) -> Path:
        first, last = run_contexts[0], run_contexts[-1]
        run_str_label = first.run_str if first is last else f"{first.run_str}-{last.run_str}"
        return proc_path(plan.data_root, first.sample, run_str_label, first.device,
                          MEASUREMENT_TYPE, "combined", combined=True)

    def _finish_artifacts(records: list[dict], run_contexts: list[RunContext],
                           plan: MeasurementPlan) -> list[str]:
        output_paths = [str(c.raw_path) for c in run_contexts]
        if not run_contexts:
            return output_paths
        png_path = _combined_png_path(run_contexts, plan)
        try:
            _save_measurement_png(records, png_path, plan=plan)
            return output_paths + [str(png_path)]
        except Exception:
            return output_paths

    def make_run_fn(plan: MeasurementPlan, run_contexts: list[RunContext]):
        def run_fn(stop_event, cb: RunCallbacks):
            daq = source = magnet = gaussmeter = temp_ctrl = None
            try:
                cb.on_status("Connecting to LabOne data server …")
                daq = connect(plan.daq_host, plan.daq_port)
                connect_device(daq, plan.leader, interface="1GbE")
                connect_device(daq, plan.follower, interface="1GbE")

                cb.on_status("Synchronizing MDS …")
                mds = setup_mds(daq, leader=plan.leader, follower=plan.follower)

                disable_sigout(daq, plan.leader)
                disable_sigout(daq, plan.follower)

                cb.on_status("Configuring demodulators …")
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
                    # Amplitude-independent -- built once, reused for every
                    # amplitude iteration below.
                    points = [
                        MeasurementPoint(magnet_current_A=I,
                                         set_action=lambda daq, I=I: set_magnet_current(
                            magnet, plan.magnet_cfg, I, gaussmeter, plan.gauss_cfg,
                            plan.acq_cfg.field_settle_tolerance_mT, stop_event))
                        for I in plan.currents_A
                    ]
                else:
                    points = [MeasurementPoint()]

                multi = len(plan.amplitudes_A) > 1
                for series_idx, amp in enumerate(plan.amplitudes_A):
                    if stop_event.is_set():
                        break

                    plan.ac_cfg.amplitude_A = amp
                    label = f"I={amp:g}A" if multi else None

                    # Checked here too, not just by build_summary(): connect_ac_source()
                    # immediately arms and starts the 6221 at plan.ac_cfg.amplitude_A —
                    # catch a mistyped exponent before that, not after.
                    _check_ac_safety(plan.ac_cfg)
                    if source is not None:
                        safe_shutdown("6221 AC source", lambda _s=source: shutdown_ac_source(_s))
                        source = None
                    cb.on_status(f"Starting 6221 AC current source{f' ({amp:g} A)' if multi else ''} …")
                    source = connect_ac_source(plan.ac_cfg)

                    cb.on_status("Locking MFLI oscillators to the 6221 marker (ExtRef) …")
                    configure_external_reference(daq, plan.leader_extref_cfg, plan.ac_cfg.frequency_Hz)
                    configure_external_reference(daq, plan.follower_extref_cfg, plan.ac_cfg.frequency_Hz)
                    if not wait_for_reference_lock(daq, plan.leader_extref_cfg,
                                                   plan.extref_lock_timeout_s, stop_event):
                        log.warning("Leader ExtRef PLL did not report locked — check the marker cabling.")
                    if not wait_for_reference_lock(daq, plan.follower_extref_cfg,
                                                   plan.extref_lock_timeout_s, stop_event):
                        log.warning("Follower ExtRef PLL did not report locked — check the marker fan-out cabling.")

                    demod2_phase_null_1f_deg = None
                    if plan.phase_cal_enabled:
                        cb.on_status("Phase calibration: nulling 1f Y (leader demod phaseshift) …")
                        if magnet is not None and plan.phase_cal_current_A is not None:
                            log.info("Phase calibration: ramping magnet to %.4f A ...",
                                     plan.phase_cal_current_A)
                            set_magnet_current(magnet, plan.magnet_cfg, plan.phase_cal_current_A,
                                               gaussmeter, plan.gauss_cfg,
                                               plan.acq_cfg.field_settle_tolerance_mT, stop_event)
                            time.sleep(plan.acq_cfg.settling_time_s)
                        result = auto_null_phase(
                            daq, plan.demod1_cfg, n_averages=plan.phase_cal_n_averages,
                            max_iterations=plan.phase_cal_max_iterations,
                        )
                        follower_display = follower_naming(plan.measure_rxx)[1]
                        if not result.converged:
                            log.warning(
                                "Phase null did not fully converge after %d iteration(s) "
                                "(|Y|/R=%.2e) — check cabling/contacts before trusting the %s data.",
                                result.iterations, result.residual_ratio, follower_display,
                            )
                        d2 = acquire_averaged(daq, plan.demod2_cfg, plan.phase_cal_n_averages)
                        log.info(
                            "%s snapshot at calibration point: X=%.4e V  Y=%.4e V  R=%.4e V — "
                            "check which channel carries the structured field dependence in the "
                            "recorded sweep before trusting either one.",
                            follower_display, d2["x_mean"], d2["y_mean"], d2["r_mean"],
                        )
                        cb.on_status(f"Phase calibration: anchoring follower {follower_display} reference (1f null) …")
                        demod2_phase_null_1f_deg = null_follower_reference_via_1f(
                            daq, plan.demod2_cfg,
                            n_averages=plan.phase_cal_n_averages,
                            max_iterations=plan.phase_cal_max_iterations,
                        )

                    # A fresh RunContext (own run number, own file) EVERY
                    # amplitude iteration -- never reuse one across the series.
                    ctx = allocate_run(
                        plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                        temperature_setpoint_K=plan.temperature_setpoint_K,
                        key_axis=None, series=plan.series,
                    )
                    run_contexts.append(ctx)
                    cb.on_run_label(f"Run #{ctx.run_str}")
                    plan.acq_cfg.output_file = str(ctx.raw_path)
                    write_csv = make_incremental_writer(
                        ctx.raw_path,
                        lambda records, _ctx=ctx, _a=amp: build_header_fields(
                            plan, _ctx, records, status="in_progress", comment="",
                            extra={"excitation_amplitude_A": _a},
                        ),
                    )

                    iter_records: list[dict] = []

                    def tagged_on_point(record: dict, _idx=series_idx, _label=label,
                                         _iter=iter_records) -> None:
                        record["series_index"] = _idx
                        record["series_label"] = _label
                        _iter.append(record)
                        cb.on_point(record)

                    cb.on_status("Running measurement …" if not multi
                                 else f"Running measurement ({label}) …")
                    iter_error: Optional[BaseException] = None
                    try:
                        run_measurement(
                            daq, plan.ac_cfg, plan.leader_extref_cfg, plan.follower_extref_cfg,
                            plan.demod1_cfg, plan.demod2_cfg, plan.acq_cfg, points,
                            stop_event=stop_event, on_point=tagged_on_point,
                            gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                            temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg, geometry_cfg=plan.geometry_cfg,
                            demod2_phase_null_1f_deg=demod2_phase_null_1f_deg, mds=mds,
                            write_csv=write_csv,
                            demod2_label=follower_naming(plan.measure_rxx)[0],
                        )
                    except Exception as exc:
                        iter_error = exc

                    iter_status = "error" if iter_error is not None \
                        else ("aborted" if stop_event.is_set() else "completed")
                    header_fields = build_header_fields(
                        plan, ctx, iter_records, status=iter_status, comment="",
                        extra={"excitation_amplitude_A": amp},
                    )
                    write_record(ctx.raw_path, iter_records, header_fields)
                    finalize_index_row(plan.data_root, ctx.sample, ctx.run_number, header_fields)

                    if iter_error is not None:
                        raise iter_error
                return None
            finally:
                # 6221 output off first (immediate, no current into the DUT),
                # so the magnet can start its ramp-down right away rather
                # than waiting behind it.
                if source is not None:
                    safe_shutdown("6221 AC source", lambda: shutdown_ac_source(source))
                if magnet is not None:
                    shutdown_magnet(magnet, plan.magnet_cfg)
                if gaussmeter is not None:
                    shutdown_gaussmeter(gaussmeter)
                if temp_ctrl is not None:
                    shutdown_temperature_controller(temp_ctrl)
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
        labels = [f"I={amp:g}A" if len(plan.amplitudes_A) > 1 else None for amp in plan.amplitudes_A]
        run_contexts: list[RunContext] = []

        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE, run_fn=make_run_fn(plan, run_contexts),
            save_artifacts=lambda records, result, status: _finish_artifacts(
                records, run_contexts, plan),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[],
            on_record=on_record, on_status=on_status, on_run_label=on_run_label, on_log=on_log,
            on_finished=make_on_finished(plan, run_contexts),
            sample=plan.sample, device=plan.device,
        )
        if not rc.try_start():
            ui.notify("Another measurement is already running — see the banner above.", type="warning")
            return
        controller["c"] = rc

        init_series(len(plan.amplitudes_A), labels, measure_rxx=plan.measure_rxx)
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
