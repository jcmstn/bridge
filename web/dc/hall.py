#!/usr/bin/env python3
"""
NiceGUI page for dc_hall_measurement.py
===========================================
Web equivalent of dc_hall_measurement_tui.py. Reuses that TUI module's pure
(Textual-independent) DEFAULTS/NUMERIC_FIELDS/TEXT_FIELDS/build_summary() so
validation stays identical between the TUI and the web page rather than
drifting apart as two hand-maintained copies -- the only things
reimplemented here are the layout (NiceGUI instead of Textual widgets) and
plan-building (adds a free-choice `data_dir` in place of the TUI's fixed
_DATA_DIR).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from nicegui import ui

_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "instruments"
_DC_DIR = Path(__file__).resolve().parent.parent.parent / "DC"
_WEB_DIR = Path(__file__).resolve().parent.parent
for _p in (_INSTRUMENTS_DIR, _DC_DIR, _WEB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dc_hall_measurement import (  # noqa: E402
    AcquisitionConfig, FieldPoint, GaussmeterConfig, MagnetConfig, SourceConfig,
    TemperatureControllerConfig, VoltmeterConfig,
    connect_gaussmeter, connect_magnet, connect_source, connect_temperature_controller,
    connect_voltmeter, run_measurement, set_magnet_current, shutdown_gaussmeter,
    shutdown_magnet, shutdown_source, shutdown_temperature_controller,
)
from dc_sweep_utils import build_output_path, linear_sweep  # noqa: E402
from dc_hall_measurement_tui import (  # noqa: E402
    DEFAULTS, NUMERIC_FIELDS, TEXT_FIELDS, DC_HALL_DESCRIPTION,
    build_summary, parse_sensor_uids,
)
from run_controller import (  # noqa: E402
    RunController, RunCallbacks, FinalStatus, num_field, text_field, bool_switch,
    render_summary, busy_banner, is_busy, format_duration,
)
from directory_picker import directory_field, validate_directory  # noqa: E402

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_SETTINGS_PATH = _DATA_DIR / "web_settings" / "dc_hall_web_settings.json"

PAGE_TITLE = "DC Hall Measurement"
SUITE = "DC"


@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    acq_cfg: AcquisitionConfig
    magnet_cfg: Optional[MagnetConfig]
    gauss_cfg: Optional[GaussmeterConfig]
    currents_A: Optional[np.ndarray]
    temp_cfg: Optional[TemperatureControllerConfig]

    @property
    def total_points(self) -> int:
        return len(self.currents_A) if self.currents_A is not None else 1


# ─────────────────────────────────────────────────────────────────────────────
# Settings persistence  ── own file, separate from the TUI's own settings JSON
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Plan building  ── same as DCHallMeasurementApp._build_plan(), except
# output_file is rooted at the user-chosen data_dir, not a fixed _DATA_DIR
# ─────────────────────────────────────────────────────────────────────────────

def build_plan(state: dict) -> MeasurementPlan:
    src_cfg = SourceConfig(
        visa_resource=state["source_visa_resource"],
        sense_current_A=state["sense_current_A"],
        compliance_V=state["compliance_V"],
        source_delay_s=state["source_delay_s"],
    )
    volt_cfg = VoltmeterConfig(
        visa_resource=state["voltmeter_visa_resource"],
        nplc=state["nplc"],
        auto_range=state["auto_range"],
    )
    acq_cfg = AcquisitionConfig(
        settling_time_s=state["settling_time_s"],
        n_reversals=int(state["n_reversals"]),
        output_file=str(build_output_path(
            Path(state["data_dir"]), state["output_subdir"], state["output_name"],
            f"{datetime.now():%Y%m%d_%H%M%S}",
        )),
    )

    magnet_cfg = None
    gauss_cfg = None
    currents_A = None
    if state["enable_sweep"]:
        magnet_cfg = MagnetConfig(
            visa_resource=state["magnet_visa_resource"],
            current_limit_A=state["current_limit_A"],
            voltage_compliance_V=state["voltage_compliance_V"],
            ramp_step_A=state["ramp_step_A"],
            ramp_delay_s=state["ramp_delay_s"],
        )
        gauss_cfg = GaussmeterConfig(
            visa_resource=state["gaussmeter_visa_resource"],
            n_averages=int(state["gaussmeter_n_averages"]),
            read_delay_s=state["gaussmeter_read_delay_s"],
        )
        currents_A = linear_sweep(
            start=state["i_min_A"], stop=state["i_max_A"], step=state["step_A"],
            bidirectional=state["bidirectional_sweep"],
        )

    temp_cfg = None
    if state["enable_temperature"]:
        uids = parse_sensor_uids(state["temperature_sensor_uids"])
        if uids:
            temp_cfg = TemperatureControllerConfig(
                visa_resource=state["temperature_visa_resource"],
                sensor_uids=uids,
            )

    return MeasurementPlan(
        src_cfg=src_cfg, volt_cfg=volt_cfg, acq_cfg=acq_cfg,
        magnet_cfg=magnet_cfg, gauss_cfg=gauss_cfg, currents_A=currents_A,
        temp_cfg=temp_cfg,
    )


def _save_measurement_png(records: list[dict], csv_path: str) -> list[str]:
    """Headless (Agg) final plot, same look as dc_hall_measurement_tui.py's
    _save_measurement_png -- called from the worker thread, no UI access."""
    if not records:
        return [csv_path]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_field = any(r.get("magnet_field_mT") is not None for r in records)
    xs = [r["magnet_field_mT"] if has_field else r["point_index"] for r in records]
    ys = [r["hall_voltage_V"] for r in records]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, "o-", color="tab:blue")
    ax.set_ylabel("Hall voltage (V)")
    ax.set_xlabel("Magnetic field (mT)" if has_field else "Point #")
    ax.set_title("Measurement result")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = Path(csv_path).with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return [csv_path, str(png_path)]


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

def page() -> None:
    ui.page_title(PAGE_TITLE)
    busy_banner()
    ui.link("← Back to measurement suite", "/").classes("text-sm")
    ui.label(PAGE_TITLE).classes("text-2xl font-bold mt-1")
    ui.label(DC_HALL_DESCRIPTION).classes("text-sm text-grey-7 mb-3")

    saved = _load_settings()

    def d(key: str):
        # DEFAULTS comes straight from dc_hall_measurement_tui and has no
        # "data_dir" entry (that field is new here) — fall back to "" for
        # any key neither `saved` nor the TUI's DEFAULTS knows about.
        if key in saved:
            return saved[key]
        return DEFAULTS.get(key, "")

    inputs: dict[str, ui.number | ui.input] = {}
    switches: dict[str, ui.switch] = {}
    controller: dict[str, Optional[RunController]] = {"c": None}

    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        with ui.column().classes("flex-grow gap-1 max-w-3xl"):
            with ui.expansion("Instruments", value=True, icon="cable").classes("w-full"):
                inputs["source_visa_resource"] = text_field(
                    "Keithley 6221 (current source)", d("source_visa_resource"))
                inputs["voltmeter_visa_resource"] = text_field(
                    "Keithley 2182 (Hall voltage)", d("voltmeter_visa_resource"))

            with ui.expansion("Sense current & compliance", value=True, icon="bolt").classes("w-full"):
                inputs["sense_current_A"] = num_field(
                    "Sense current (A)", float(d("sense_current_A")),
                    hint="Reversed +I/-I each rep to cancel thermal-EMF offsets.")
                inputs["compliance_V"] = num_field(
                    "Compliance voltage (V)", float(d("compliance_V")))
                with ui.expansion("Source timing (advanced)"):
                    inputs["source_delay_s"] = num_field(
                        "6221 source delay (s)", float(d("source_delay_s")))

            with ui.expansion("Voltmeter (Keithley 2182)", value=True, icon="speed").classes("w-full"):
                inputs["nplc"] = num_field(
                    "NPLC (integration time)", float(d("nplc")),
                    hint="Bigger = quieter but slower. 1 line cycle = 1/50 or 1/60 s.")
                switches["auto_range"] = bool_switch("Auto-range", d("auto_range"))

            with ui.expansion("Acquisition & output", value=True, icon="save").classes("w-full"):
                inputs["settling_time_s"] = num_field(
                    "Settling time per point (s)", float(d("settling_time_s")),
                    hint="Dead-time after a field change, before acquiring.")
                inputs["n_reversals"] = num_field(
                    "+I/-I reversal pairs averaged per point", float(d("n_reversals")), integer=True,
                    hint="Splits each point into (V(+I)-V(-I))/2 [reported R] and "
                         "(V(+I)+V(-I))/2 [recorded, not discarded].")
                inputs["output_name"] = text_field("Output file name (prefix)", d("output_name"))
                inputs["output_subdir"] = text_field(
                    "Data sub-directory (optional)", d("output_subdir"),
                    hint="Saved to <save directory>/<sub-directory>/<prefix>_<timestamp>.csv")
                data_dir_input = directory_field("Save directory", saved.get("data_dir") or str(_DATA_DIR))

            with ui.expansion("Magnet & field sweep", value=True, icon="explore").classes("w-full") as magnet_section:
                switches["enable_sweep"] = bool_switch(
                    "Sweep magnetic field (Kepco magnet)", d("enable_sweep"))
                inputs["magnet_visa_resource"] = text_field("Magnet VISA resource", d("magnet_visa_resource"))
                inputs["current_limit_A"] = num_field(
                    "Software current limit (A)", float(d("current_limit_A")),
                    hint="Hard safety ceiling — independent of the supply's own range.")
                inputs["voltage_compliance_V"] = num_field(
                    "Voltage compliance (V)", float(d("voltage_compliance_V")))
                with ui.expansion("Ramp safety (advanced)"):
                    inputs["ramp_step_A"] = num_field("Ramp step (A)", float(d("ramp_step_A")))
                    inputs["ramp_delay_s"] = num_field("Ramp delay (s)", float(d("ramp_delay_s")))
                inputs["i_min_A"] = num_field("Sweep current min (A)", float(d("i_min_A")))
                inputs["i_max_A"] = num_field("Sweep current max (A)", float(d("i_max_A")))
                inputs["step_A"] = num_field("Sweep step size (A)", float(d("step_A")))
                switches["bidirectional_sweep"] = bool_switch(
                    "Bidirectional sweep (min → max → min)", d("bidirectional_sweep"))
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
                inputs["temperature_visa_resource"] = text_field(
                    "MercuryiTC VISA resource", d("temperature_visa_resource"),
                    hint="e.g. TCPIP0::<ip>::7020::SOCKET, or an ASRL resource.")
                inputs["temperature_sensor_uids"] = text_field(
                    "Sensor board UID(s)", d("temperature_sensor_uids"),
                    hint="1 or 2 board UIDs, comma-separated, e.g. 'MB1.T1, DB5.T1'.")

        with ui.column().classes("w-96 gap-2"):
            ui.label("Summary").classes("text-lg font-bold")
            summary_box = ui.column().classes("w-full")
            start_btn = ui.button("▶  Start measurement", color="primary").classes("w-full")

    # ── Live run area (built once, populated on Start) ──────────────────
    ui.separator().classes("my-3")
    status_label = ui.label("Idle.").classes("text-sm font-bold")
    with ui.row().classes("w-full gap-2"):
        abort_btn = ui.button("Abort (safe ramp-down)", color="negative").props("outline")
        abort_btn.set_visibility(False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Hall voltage"))
    fig.update_layout(
        xaxis_title="Magnetic field (mT)", yaxis_title="Hall voltage (V)",
        margin=dict(l=60, r=20, t=30, b=50), height=420,
    )
    plot = ui.plotly(fig).classes("w-full")

    columns = [
        {"name": "n", "label": "#", "field": "n"},
        {"name": "I", "label": "I_magnet (A)", "field": "I"},
        {"name": "B", "label": "B (mT)", "field": "B"},
        {"name": "V", "label": "V_Hall (V)", "field": "V"},
        {"name": "R", "label": "R_Hall (Ω)", "field": "R"},
        {"name": "T1", "label": "T1 (K)", "field": "T1"},
        {"name": "T2", "label": "T2 (K)", "field": "T2"},
    ]
    table = ui.table(columns=columns, rows=[], row_key="n").classes("w-full").props("dense")
    log_area = ui.log(max_lines=2000).classes("w-full h-48 font-mono text-xs")

    # ── Validation / summary refresh ─────────────────────────────────────

    def collect_raw() -> dict:
        raw = {fid: inp.value for fid, inp in inputs.items()}
        for fid, sw in switches.items():
            raw[fid] = sw.value
        raw["data_dir"] = data_dir_input.value
        return raw

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
        for fid, sw in switches.items():
            state[fid] = sw.value
        return state, errors

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
        magnet_section.set_text(
            "Magnet & field sweep" + ("" if switches["enable_sweep"].value else " (disabled)"))

    for inp in list(inputs.values()) + [data_dir_input]:
        inp.on_value_change(refresh_summary.refresh)
    for sw in switches.values():
        sw.on_value_change(refresh_summary.refresh)
    refresh_summary()
    ui.timer(2.0, refresh_summary.refresh)  # also catch external busy/idle transitions

    # ── Run wiring ────────────────────────────────────────────────────────

    def on_record(record: dict) -> None:
        with plot:
            has_field = record.get("magnet_field_mT") is not None
            x = record["magnet_field_mT"] if has_field else record["point_index"]
            fig.data[0].x = fig.data[0].x + (x,)
            fig.data[0].y = fig.data[0].y + (record["hall_voltage_V"],)
            fig.update_layout(xaxis_title="Magnetic field (mT)" if has_field else "Point #")
        plot.update()
        table.rows.append({
            "n": record["point_index"] + 1,
            "I": f"{record['magnet_current_A']:.4f}" if record.get("magnet_current_A") is not None else "—",
            "B": f"{record['magnet_field_mT']:.2f}" if record.get("magnet_field_mT") is not None else "—",
            "V": f"{record['hall_voltage_V']:.4e}",
            "R": f"{record['hall_resistance_ohm']:.5g}",
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
            source = voltmeter = magnet = gaussmeter = temp_ctrl = None
            try:
                cb.on_status("Connecting to Keithley 6221 & 2182 …")
                source = connect_source(plan.src_cfg)
                voltmeter = connect_voltmeter(plan.volt_cfg)

                if plan.temp_cfg is not None:
                    cb.on_status("Connecting to MercuryiTC (temperature) …")
                    temp_ctrl = connect_temperature_controller(plan.temp_cfg)

                if plan.magnet_cfg is not None and plan.currents_A is not None:
                    cb.on_status("Connecting magnet power supply …")
                    magnet = connect_magnet(plan.magnet_cfg)
                    cb.on_status("Connecting gaussmeter …")
                    gaussmeter = connect_gaussmeter(plan.gauss_cfg)
                    points = [
                        FieldPoint(magnet_current_A=I,
                                   set_action=lambda I=I: set_magnet_current(magnet, plan.magnet_cfg, I))
                        for I in plan.currents_A
                    ]
                else:
                    points = [FieldPoint()]

                cb.on_status("Running measurement …")
                return run_measurement(
                    source, voltmeter, plan.src_cfg, plan.acq_cfg, points,
                    stop_event=stop_event, on_point=cb.on_point,
                    gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                    temp_ctrl=temp_ctrl, temp_cfg=plan.temp_cfg,
                )
            finally:
                if magnet is not None:
                    shutdown_magnet(magnet, plan.magnet_cfg)
                if gaussmeter is not None:
                    shutdown_gaussmeter(gaussmeter)
                if temp_ctrl is not None:
                    shutdown_temperature_controller(temp_ctrl)
                if source is not None:
                    shutdown_source(source)
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

        raw = collect_raw()
        _save_settings(raw)

        plan = build_plan(state)
        Path(state["data_dir"]).mkdir(parents=True, exist_ok=True)

        rc = RunController(
            suite=SUITE, measurement=PAGE_TITLE, run_fn=make_run_fn(plan),
            save_artifacts=lambda records, result: _save_measurement_png(records, plan.acq_cfg.output_file),
            parameters=state, data_dir=state["data_dir"],
            planned_output_paths=[plan.acq_cfg.output_file],
            on_record=on_record, on_status=on_status, on_log=on_log, on_finished=on_finished,
        )
        if not rc.try_start():
            ui.notify("Another measurement is already running — see the banner above.", type="warning")
            return
        controller["c"] = rc

        table.rows.clear()
        table.update()
        fig.data[0].x = ()
        fig.data[0].y = ()
        plot.update()
        log_area.clear()
        abort_btn.set_visibility(True)
        start_btn.set_enabled(False)

    def on_abort() -> None:
        if controller["c"] is not None:
            controller["c"].abort()

    start_btn.on_click(on_start)
    abort_btn.on_click(on_abort)
