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
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import background_tasks, ui

from dc.dc_sweep_utils import parse_sweep_rows, parse_value_list
import mfli.mfli_dual_harmonic_6221_tui as program
from mfli.mfli_dual_harmonic_6221_tui import (
    build_plan,
    MFLI_DUAL_HARMONIC_6221_DESCRIPTION,
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, OPTIONAL_NUMERIC_FIELDS,
    MeasurementPlan, build_summary,
    compute_filename_preview, follower_naming,
)
from instruments.data_naming import (
    TEST_SAMPLE, RunContext,
)
from web.run_controller import (
    RunController, FinalStatus, num_field, text_field, textarea_field, bool_switch,
    optional_num_field, render_summary, busy_banner, is_busy,
    param_card, param_grid, stable_card, stable_grid, advanced_section, measurement_layout,
    program_artifacts, program_run_fn, prompt_last_run,
)
from web.directory_picker import validate_directory
from web.field_diagram import build_field_diagram_figure
from web.identity_bar import identity_bar
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

log = logging.getLogger("web.mfli.dual_harmonic_6221")

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "mfli_dual_harmonic_6221_web_settings.json"

PAGE_TITLE = "MFLI Dual-Harmonic Measurement (6221 AC source)"
SUITE = "MFLI"


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
            # One current: 1f and the follower keep their own colors, as in a
            # manual run. Several: color by current, dotted follower.
            c1, c2 = (cmap[i % len(cmap)],) * 2 if n_series > 1 else ("#1f77b4", "#ff7f0e")
            name = labels[i]
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers",
                                      name=f"1f {name}" if name else "1f R",
                                      line=dict(color=c1), legendgroup=f"s{i}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers",
                                      name=f"{follower_display} {name}" if name else f"{follower_display} R",
                                      line=dict(color=c2, dash="dot" if n_series > 1 else "solid"),
                                      legendgroup=f"s{i}"),
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

    def make_on_finished(plan: MeasurementPlan, run_contexts: list[RunContext], run_extras: list):
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
                prompt_last_run(page_client, program, plan, run_contexts, run_extras, records),
                name="status_comment_prompt")
        return on_finished

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

        _save_settings(collect_raw())

        plan = build_plan(state, Path(state["data_dir"]))
        labels = [f"I={amp:g}A" if len(plan.amplitudes_A) > 1 else None for amp in plan.amplitudes_A]
        run_contexts: list[RunContext] = []
        run_extras: list[dict] = []

        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE, run_fn=program_run_fn(program, plan, run_contexts, run_extras),
            save_artifacts=lambda records, result, status: program_artifacts(program, plan, run_contexts),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[],
            on_record=on_record, on_status=on_status, on_run_label=on_run_label, on_log=on_log,
            on_finished=make_on_finished(plan, run_contexts, run_extras),
            sample=plan.sample, device=plan.device, run_cost=plan.run_cost,
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
