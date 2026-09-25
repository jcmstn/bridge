#!/usr/bin/env python3
"""
NiceGUI page for SOT pulsed switching (4200A or 6221 pulse · DC or lock-in read)
=================================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-23

Web equivalent of sot/sot_pulsed_switching_tui.py: the same form, summary,
plan and run — that module's pure DEFAULTS / field groups / resolve_state /
build_summary / build_plan, and the plan's engine's run_plan, header and PNG
(SOTPS, SOT2H or SOT1I, picked by the "Write pulse" × "Read" toggles; the other
modes' cards are hidden). This page only supplies the form, the live Plotly
switching curve, the results table and the RunController callbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from nicegui import ui

import sot.sot_pulsed_switching_tui as program
from instruments.data_naming import TEST_SAMPLE, RunContext
from sot.sot_pulsed_switching_tui import (
    DEFAULTS, PULSE_SOURCES, READ_MODES, SOT_PULSED_DESCRIPTION, SOTPulsedSwitchingApp,
    build_plan, build_summary, compute_filename_preview, engine, mode_errors,
)
from web.directory_picker import validate_directory
from web.field_diagram import build_field_diagram_figure
from web.identity_bar import identity_bar
from web.run_controller import (
    RunController, advanced_section, bool_switch, busy_banner, finished_handler, form_state,
    is_busy, load_settings, measurement_layout, num_field, optional_num_field, param_card,
    param_grid, program_artifacts, program_run_fn, refresh_on_busy_change, render_summary,
    save_settings, stable_card, stable_grid, text_field,
)
from web.sample_picker import NEW_SAMPLE_SENTINEL, prepare_data_root

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "sot_pulsed_switching_web_settings.json"

PAGE_TITLE = "SOT Pulsed Switching"
SUITE = "SOT"

log = logging.getLogger("web.sot.pulsed_switching")

_COLORS = ["#2E3192", "#e34948", "#2E7D32", "#B26A00", "#7E57C2", "#0277BD"]
_AUTOMODE_OPTIONS = {value: label for label, value in program.h2.AUTOMODE_OPTIONS}


def _fmt(value, spec: str) -> str:
    return "—" if value is None else format(value, spec)


# Per engine (type code): the live plot's x / y and the results table — the
# same quantities as that mode's TUI run screen.
def _plot_axes(plan) -> tuple[str, str, str, str, float]:
    """(x key, x title, y key, y title, x scale) for the plan's engine."""
    code = engine(plan).MEASUREMENT_TYPE
    if code == "SOT1I":
        return ("pulse_current_A", "Pulse current (mA)", "demod_R_V",
                f"V_{plan.read_cfg.harmonic}f (V)", 1e3)
    if code == "SOT2H":
        return "pulse_amplitude_V", "Pulse amplitude (V)", "2f_R_V", "V_2f (V)", 1.0
    return "pulse_amplitude_V", "Pulse amplitude (V)", "hall_resistance_ohm", "R_xy (Ω)", 1.0


def _table_spec(plan) -> tuple[list[tuple[str, str]], callable]:
    """([(key, label)], record -> {key: text}) for the plan's engine."""
    code = engine(plan).MEASUREMENT_TYPE
    common = [("n", "amp #"), ("Imag", "I_mag (A)")]
    locked = lambda r: "yes" if r.get("reference_locked") else "no"
    if code == "SOT1I":
        cols = common + [("Ip", "I_pulse (A)"), ("w", "width meas (s)"),
                         ("V", f"V_{plan.read_cfg.harmonic}f (V)"), ("lk", "locked"), ("T1", "T1 (K)")]
        row = lambda r: {"Ip": _fmt(r["pulse_current_A"], ".4g"),
                         "w": _fmt(r.get("pulse_width_measured_s"), ".4g"),
                         "V": _fmt(r["demod_R_V"], ".4e"), "lk": locked(r)}
    elif code == "SOT2H":
        cols = common + [("Vp", "V_pulse (V)"), ("Ip", "I_pulse (A)"), ("V1", "V_1f (V)"),
                         ("V2", "V_2f (V)"), ("lk", "locked"), ("T1", "T1 (K)")]
        row = lambda r: {"Vp": _fmt(r["pulse_amplitude_V"], ".4g"),
                         "Ip": _fmt(r.get("pulse_current_measured_A"), ".4e"),
                         "V1": _fmt(r["1f_R_V"], ".4e"), "V2": _fmt(r["2f_R_V"], ".4e"),
                         "lk": locked(r)}
    else:
        cols = common + [("Vp", "V_pulse (V)"), ("Ip", "I_pulse (A)"), ("Vxy", "V_xy (V)"),
                         ("R", "R_xy (Ω)"), ("T1", "T1 (K)")]
        row = lambda r: {"Vp": _fmt(r["pulse_amplitude_V"], ".4g"),
                         "Ip": _fmt(r.get("pulse_current_measured_A"), ".4e"),
                         "Vxy": _fmt(r["hall_voltage_V"], ".4e"),
                         "R": _fmt(r["hall_resistance_ohm"], ".5g")}

    def full_row(r: dict) -> dict:
        return {"n": str(r["amplitude_index"] + 1), "Imag": _fmt(r.get("magnet_current_A"), "g"),
                "T1": _fmt(r.get("temperature_1_K"), ".3f"), **row(r)}
    return cols, full_row


def page() -> None:
    ui.page_title(PAGE_TITLE)
    page_client = ui.context.client  # has slot context now; reused by the detached status/comment task
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(SOT_PULSED_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = load_settings(_SETTINGS_PATH)

    def d(key: str):
        return saved[key] if key in saved else DEFAULTS.get(key, "")

    def opt(key: str) -> Optional[float]:
        v = d(key)
        return float(v) if v not in ("", None) else None

    inputs: dict = {}
    optional_inputs: dict = {}
    switches: dict = {}
    mode_cards: dict = {}           # MODE_WIDGETS id -> card, shown per (pulse, read)
    controller: dict[str, Optional[RunController]] = {"c": None}

    def fld(fid: str, label: str, hint: str = "") -> None:
        """One form field, its widget picked by the program's field groups."""
        if fid in program.TEXT_FIELDS:
            inputs[fid] = text_field(label, d(fid), hint=hint)
        elif fid in program.OPTIONAL_NUMERIC_FIELDS:
            optional_inputs[fid] = optional_num_field(label, opt(fid), hint=hint)
        else:
            inputs[fid] = num_field(label, float(d(fid)), hint=hint,
                                    integer=program.NUMERIC_FIELDS[fid] is int)

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
                with param_card("Write pulse × read"):
                    pulse_select = ui.select({v: k for k, v in PULSE_SOURCES}, value=d("pulse_source"),
                                             label="Write pulse").classes("w-full")
                    read_select = ui.select({v: k for k, v in READ_MODES}, value=d("read_mode"),
                                            label="Read").classes("w-full")
                    ui.label("4200A+DC → SOTPS · 4200A+lock-in → SOT2H · 6221+lock-in → SOT1I").classes(
                        "text-xs text-grey-6 -mt-1 mb-1")
                    switches["amplitude_bidirectional"] = bool_switch(
                        "Sweep up then back down (hysteresis loop)", d("amplitude_bidirectional"))

                with param_card("Write pulse (4200A PMU)") as mode_cards["mode_pmu_pulse"]:
                    fld("amplitude_start_V", "Amplitude start (V)")
                    fld("amplitude_stop_V", "Amplitude stop (V)")
                    fld("amplitude_step_V", "Amplitude step (V)", "One pulse per step.")
                    fld("pulse_width_s", "Pulse width (s)")
                    fld("pulse_rise_s", "Rise time (s)")
                    fld("pulse_fall_s", "Fall time (s)")
                    fld("pulse_period_s", "Pulse period (s)", "≥ delay + width + rise + fall.")

                with param_card("Write pulse (6221 WAVE, hardware-timed)") as mode_cards["mode_6221_pulse"]:
                    fld("pulse_current_start_A", "Pulse current start (A)")
                    fld("pulse_current_stop_A", "Pulse current stop (A)")
                    fld("pulse_current_step_A", "Pulse current step (A)", "One pulse per step.")
                    fld("wave_pulse_width_s", "Requested pulse width (s)",
                        "No rise/fall control. Measured width is logged.")
                    fld("pulse_compliance_V", "Pulse voltage compliance (V)")

                with param_card("Delayed read"):
                    fld("delay_after_pulse_s", "Delay after pulse (s)", "Wait between pulse end and the read.")
                    fld("sense_current_values", "6221 read current (A)",
                        "DC: ±I; lock-in: AC peak. Comma-separate for one sweep + file per value.")

                with param_card("DC R_xy read (6221 ±I + 2182)") as mode_cards["mode_dc_read"]:
                    fld("n_reversals", "Reversal pairs per read")
                    fld("settle_after_enable_s", "6221 settle after enable (s)")

                with param_card("Lock-in read (6221 AC + MFLI)") as mode_cards["mode_lockin_read"]:
                    fld("frequency_Hz", "AC excitation frequency (Hz)", "Avoid multiples of 50/60 Hz.")
                    fld("n_averages", "MFLI samples averaged per read")
                    fld("lock_settle_s", "Settle after PLL lock (s)")
                    fld("lock_timeout_s", "PLL lock timeout (s)",
                        "Timeout is logged, not fatal.")

                with param_card("Lock-in harmonic (6221 pulse)") as mode_cards["mode_sot1i_harmonic"]:
                    fld("harmonic", "Harmonic to lock in on",
                        "2 = SOT harmonic Hall, 1 = AHE/PHE.")

                with param_card("Static field (Kepco magnet)"):
                    fld("magnet_current_A", "Assist current(s) (A)",
                        "Comma-separate for one sweep + file per value.")
                    fld("field_theta_deg", "θ — mount tilt from OOP (°)",
                        "0° = out-of-plane. Recorded, not set.")
                    fld("field_phi_deg", "φ — azimuth from current axis (°)", "Optional. Ignored when θ=0°.")
                    with ui.row().classes("gap-2 mb-1"):
                        ui.button("xy", on_click=lambda: inputs["field_theta_deg"].set_value(90)).props("dense outline")
                        ui.button("zx", on_click=lambda: optional_inputs["field_phi_deg"].set_value(0)).props("dense outline")
                        ui.button("zy", on_click=lambda: optional_inputs["field_phi_deg"].set_value(90)).props("dense outline")
                    field_diagram_plot = ui.plotly(build_field_diagram_figure(
                        opt("field_theta_deg"), opt("field_phi_deg"))).classes("w-full").style("height: 220px")
                    fld("field_settle_tolerance_mT", "Field settle tolerance (mT)")

                with param_card("Temperature logging"):
                    switches["enable_temperature"] = bool_switch(
                        "Log temperature (Oxford Instruments MercuryiTC)", d("enable_temperature"))

            # ── Tier 2: instrument wiring & safety — collapsed ──────────────
            with advanced_section("Instrument configuration & addresses", icon="settings"):
                with stable_grid():
                    with stable_card("Keithley 4200A PMU (KXCI)") as mode_cards["mode_pmu_config"]:
                        fld("k4200_visa_resource", "KXCI VISA resource",
                            "GPIB0::17::INSTR  or  TCPIP0::<ip>::1225::SOCKET")
                        fld("pmu_library", "KULT pulse library", "Confirm against the `UL` output in the run log.")
                        fld("pmu_module", "KULT pulse module name",
                            "instruments/kult/bridge_sot_pulse.c, compiled in KULT.")
                        fld("pmu_channel", "PMU channel")
                        fld("pmu_id", "PMU card name", "e.g. PMU1 (lowest-numbered slot).")
                        fld("pmu_return_names", "Module return params (comma-sep)",
                            "Must match the module's output order. Blank = none.")
                        fld("pmu_v_range_V", "PMU voltage range (V)", "10 or 40.")
                        fld("pmu_i_range_A", "PMU current measure range (A)",
                            "RPM on the 10 V range: max 0.01 A.")
                        fld("pmu_v_limit_V", "Pulse amplitude software limit (V)")
                        fld("pulse_delay_s", "Pulse delay before rise (s)", "Dead time before the rise. Normally 0.")
                        fld("n_pulses", "Pulses per point (burst-average)",
                            "Keep 1: N pulses = N switching attempts.")
                        fld("pmu_sample_rate", "PMU sample rate (S/s)")
                        fld("pmu_meas_start_perc", "Spot-mean window start (0-1)")
                        fld("pmu_meas_stop_perc", "Spot-mean window stop (0-1)")
                        fld("pmu_dut_res_ohm", "DUT resistance for load-line (Ω)",
                            "Real channel R (4-probe). Drives the current estimate.")

                    with stable_card("Keithley 6221"):
                        fld("source_visa_resource", "6221 VISA resource")
                        fld("compliance_V", "6221 compliance (V)",
                            "Keep low (read needs < 1 V) — protects the shared bus.")

                    with stable_card("Keithley 2182 + DC read") as mode_cards["mode_dc_instruments"]:
                        fld("voltmeter_visa_resource", "2182 (Hall voltage)")
                        fld("source_delay_s", "6221 source delay (s)")
                        fld("nplc", "2182 NPLC")
                        switches["auto_range"] = bool_switch("2182 auto-range", d("auto_range"))

                    with stable_card("Zurich Instruments MFLI + 6221 marker") as mode_cards["mode_mfli"]:
                        fld("mfli_host", "LabOne data server host")
                        fld("mfli_port", "LabOne data server port")
                        fld("mfli_device", "MFLI device ID", "e.g. dev1234.")
                        fld("aux_input_ch", "Aux Input carrying the marker (0-based)", "0 = Aux In 1.")
                        fld("osc_index", "Oscillator locked by the PLL")
                        fld("extref_index", "ExtRef/PLL module index")
                        fld("pll_demod_index", "PLL phase-detector demod index (≠ the read demods)",
                            "Dedicated PLL demod — not a read demod.")
                        automode_select = ui.select(_AUTOMODE_OPTIONS, value=int(d("automode")),
                                                    label="PLL bandwidth adaptation").classes("w-full")
                        ui.label(program.h2.AUTOMODE_HINT).classes("text-xs text-grey-6 -mt-2 mb-2")
                        fld("input_ch", "Signal Input channel (0-based)")
                        switches["differential"] = bool_switch("Differential input (IN+ / IN−)", d("differential"))
                        switches["ac_coupling"] = bool_switch("AC-couple the input", d("ac_coupling"))
                        fld("input_range_V", "Signal Input range (V)")
                        fld("sample_rate_Hz", "Demodulator output rate (Sa/s)")
                        fld("filter_time_constant_s", "Filter time constant (s)")
                        fld("filter_order", "Filter order (1-8)")
                        switches["filter_sinc"] = bool_switch("Sinc filter (extra harmonic rejection)",
                                                              d("filter_sinc"))
                        fld("phasemarker_line", "Trigger Link phase-marker pin (1-6)",
                            "Wire to MFLI Aux In; check it isn't a 6221 default pin.")

                    with stable_card("MFLI demodulators (1f + 2f)") as mode_cards["mode_sot2h_demods"]:
                        fld("demod1_index", "1f demodulator index")
                        fld("demod2_index", "2f demodulator index",
                            "Not the PLL demod index.")

                    with stable_card("MFLI demodulator") as mode_cards["mode_sot1i_demod"]:
                        fld("demod_index", "Demodulator index",
                            "Not the PLL demod index.")

                    with stable_card("Kepco magnet + Lake Shore 475"):
                        fld("magnet_visa_resource", "Kepco VISA resource")
                        fld("current_limit_A", "Magnet current limit (A)",
                            "Hard safety ceiling.")
                        fld("magnet_voltage_compliance_V", "Magnet voltage compliance (V)")
                        fld("ramp_step_A", "Magnet ramp step (A)")
                        fld("ramp_delay_s", "Magnet ramp delay (s)")
                        fld("gaussmeter_visa_resource", "Lake Shore 475 VISA resource")
                        fld("gaussmeter_n_averages", "475 readings averaged")
                        fld("gaussmeter_read_delay_s", "475 read delay (s)")

                    with stable_card("Temperature (MercuryiTC)"):
                        fld("temperature_visa_resource", "MercuryiTC VISA resource")
                        fld("temperature_sensor_uids", "Sensor board UID(s)", "1-2 UIDs, comma-separated.")

        with regions.summary:
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start measurement", color="primary").classes("w-full")

        with regions.output:
            with ui.row().classes("w-full items-center gap-3"):
                run_label = ui.label("").classes("text-sm font-bold text-grey-6")
                status_label = ui.label("Idle.").classes("text-sm font-bold")
            abort_btn = ui.button("Abort (safe shutdown)", color="negative").props("outline")
            abort_btn.set_visibility(False)

            fig = go.Figure()
            fig.update_layout(margin=dict(l=60, r=20, t=30, b=50), showlegend=True)
            with ui.element("div").classes("w-full").style("aspect-ratio: 1 / 1; max-height: 90vh"):
                plot = ui.plotly(fig).classes("w-full h-full")

            table = ui.table(columns=[], rows=[], row_key="i").classes("w-full").props("dense")
            log_area = ui.log(max_lines=2000).classes("w-full h-48 font-mono text-xs")

    def show_mode() -> None:
        shown = SOTPulsedSwitchingApp.MODE_WIDGETS
        for widget_id, card in mode_cards.items():
            card.set_visibility(shown[widget_id](pulse_select.value, read_select.value))

    def parse_state() -> tuple[dict, list[str]]:
        state, errors = form_state(
            program, identity, inputs=inputs, switches=switches, optional_inputs=optional_inputs,
            selects={"pulse_source": pulse_select, "read_mode": read_select,
                     "automode": automode_select})
        return state, mode_errors(state, errors)     # a hidden mode's field never blocks

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        for fid, inp in optional_inputs.items():
            raw[fid] = inp.value if inp.value is not None else ""
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["pulse_source"] = pulse_select.value
        raw["read_mode"] = read_select.value
        raw["automode"] = automode_select.value
        raw["data_dir"] = identity.data_dir_input.value
        raw["device"] = identity.device_input.value
        raw["cooldown"] = identity.cooldown_input.value
        raw["temperature_setpoint_K"] = (identity.temperature_input.value
                                         if identity.temperature_input.value is not None else "")
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
            render_summary([i for i in info if i], warnings, errors)
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
    automode_select.on_value_change(refresh_summary.refresh)
    for select in (pulse_select, read_select):
        select.on_value_change(lambda _e: (show_mode(), refresh_summary.refresh()))
    show_mode()
    refresh_summary()
    refresh_on_busy_change(refresh_summary.refresh)

    # ── Run wiring ───────────────────────────────────────────────────────

    run: dict = {}          # the running plan's plot axes, table row builder, trace per series

    def init_output(plan) -> None:
        x_key, x_title, y_key, y_title, x_scale = _plot_axes(plan)
        columns, row = _table_spec(plan)
        run.update(x_key=x_key, y_key=y_key, x_scale=x_scale, row=row, traces={})
        fig.data = []
        fig.update_xaxes(title_text=x_title)
        fig.update_yaxes(title_text=y_title)
        table.columns = [{"name": k, "label": label, "field": k} for k, label in columns]
        table.rows.clear()

    def on_record(record: dict) -> None:
        idx = record.get("series_index", 0)
        if idx not in run["traces"]:          # one line per read current / assist field
            color = _COLORS[len(run["traces"]) % len(_COLORS)]
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(color=color),
                                     name=record.get("series_label") or "run"))
            run["traces"][idx] = len(fig.data) - 1
        trace = fig.data[run["traces"][idx]]
        trace.x = trace.x + (record[run["x_key"]] * run["x_scale"],)
        trace.y = trace.y + (record[run["y_key"]],)
        table.rows.append(run["row"](record) | {"i": len(table.rows)})   # amp # repeats per series

    def on_start() -> None:
        state, parse_errors = parse_state()
        _, dir_error = validate_directory(identity.data_dir_input.value or "")
        if parse_errors or dir_error or build_summary(state)[2]:
            ui.notify("Fix the blocking issues before starting.", type="negative")
            return
        state["data_dir"] = prepare_data_root(identity.data_dir_input.value, state["sample"])
        save_settings(_SETTINGS_PATH, collect_raw())

        plan = build_plan(state, Path(state["data_dir"]))
        run_contexts: list[RunContext] = []
        run_extras: list[dict] = []
        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE,
            run_fn=program_run_fn(program, plan, run_contexts, run_extras),
            save_artifacts=lambda records, result, status: program_artifacts(program, plan, run_contexts),
            parameters=state, data_dir=state["data_dir"], planned_output_paths=[],
            on_tick=lambda: (plot.update(), table.update()),
            on_record=on_record, on_status=status_label.set_text, on_run_label=run_label.set_text,
            on_log=lambda text, level: log_area.push(text),
            on_finished=finished_handler(
                page_client, controller, status_label, abort_btn, start_btn, refresh_summary.refresh,
                program, plan, run_contexts, run_extras),
            sample=plan.sample, device=plan.device, run_cost=plan.run_cost, run_contexts=run_contexts,
        )
        if not rc.try_start():
            ui.notify("Another measurement is already running — see the banner above.", type="warning")
            return
        controller["c"] = rc

        init_output(plan)
        plot.update()
        table.update()
        log_area.clear()
        abort_btn.set_visibility(True)
        start_btn.set_enabled(False)

    def on_abort() -> None:
        if controller["c"] is not None:
            controller["c"].abort()

    start_btn.on_click(on_start)
    abort_btn.on_click(on_abort)
