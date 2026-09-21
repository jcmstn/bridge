#!/usr/bin/env python3
"""
Textual TUI front-end for mfli_noise_spectrum.py
===================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-17

Runs the 2-pass (Excitation ON / Excitation OFF) noise-floor estimate for
the mfli_dual_harmonic_6221 program without touching the dataclasses in the
script itself. Deliberately smaller than the other suite TUIs — this is a
quick nV/√Hz check, not a sweep: no live plot process, no per-point table,
no abort (each pass is a single ~30 s recording).

Run with:
    python mfli_noise_spectrum_tui.py

Requirements:
    pip install textual matplotlib  (in addition to mfli_noise_spectrum.py's own deps)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.validation import Number
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    Switch,
)

from dc.dc_sweep_utils import parse_value_list
from mfli.mfli_dual_harmonic_6221 import _AC_CURRENT_CEILING_A, _AC_COMPLIANCE_CEILING_V, extref_lock_s
from mfli.mfli_dual_harmonic_tui import _LogRelay, card, field, format_si, switch_field
from mfli.mfli_noise_spectrum import (
    ACSourceConfig,
    AcquisitionConfig,
    ExtRefConfig,
    NoiseDemodConfig,
    ReferenceConfig,
    configure_noise_demod,
    connect,
    connect_device,
    finalize_comment,
    measure_noise_floor,
    plot_results,
    proc_path,
    report_mains_peaks,
    save_results,
    setup_mds,
    thermal_noise_asd,
)
from instruments.data_dir import DataDirPickerScreen, validate_directory
from instruments.data_naming import TEST_SAMPLE, ensure_sample
from instruments.keithley6221 import ac_source_restart_s
from instruments.run_time import (
    ACQ_OVERHEAD_S, GPIB_TXN_S, MDS_SYNC_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S,
    RunCost, progress_step, progress_total,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
    NewSampleScreen,
    StatusCommentScreen,
    sample_options,
)

log = logging.getLogger("mfli_noise_spectrum_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "mfli_noise_spectrum_tui_settings.json"

MEASUREMENT_TYPE = "NOISE"


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "leader_device": "dev7885",
    "follower_device": "dev7886",
    "daq_host": "localhost",
    "daq_port": "8004",
    "ac_visa_resource": "GPIB0::20::INSTR",
    "frequency_Hz": "317.3",
    "amplitude_values": "1e-4",
    "ac_compliance_V": "2.0",
    "phasemarker_line": "1",
    "extref_lock_timeout_s": "5.0",
    "duration_s": "30",
    "also_measure_off": True,
    "thermal_R_ohm": "10000",
    "thermal_T_K": "293",
    "input_range_1f_V": "1.0",
    "input_range_2f_V": "1.0",
    "sample_rate_Hz": "13389.0",
    "time_constant_s": "3e-5",
    "leader_extref_index": "0",
    "leader_aux_input_ch": "0",
    "leader_osc_index": "0",
    "leader_pll_demod_index": "1",
    "leader_automode": "4",
    "follower_extref_index": "0",
    "follower_aux_input_ch": "0",
    "follower_osc_index": "0",
    "follower_pll_demod_index": "1",
    "follower_automode": "4",
    "device": "",
    "cooldown": "",
}

AUTOMODE_HINT = ("2=low bandwidth (most forgiving acquisition), 3=high bandwidth "
                  "(fastest tracking once locked), 4=dynamic/auto (default).")

NUMERIC_FIELDS: dict = {
    "daq_port": int,
    "frequency_Hz": float,
    "ac_compliance_V": float,
    "phasemarker_line": int,
    "extref_lock_timeout_s": float,
    "duration_s": float,
    "thermal_T_K": float,
    "input_range_1f_V": float,
    "input_range_2f_V": float,
    "sample_rate_Hz": float,
    "time_constant_s": float,
    "leader_extref_index": int,
    "leader_aux_input_ch": int,
    "leader_osc_index": int,
    "leader_pll_demod_index": int,
    "follower_extref_index": int,
    "follower_aux_input_ch": int,
    "follower_osc_index": int,
    "follower_pll_demod_index": int,
}
TEXT_FIELDS = ["leader_device", "follower_device", "daq_host", "ac_visa_resource",
               "amplitude_values", "device", "cooldown", "data_dir"]
# Free-text, blank-allowed: parsed to Optional[float] by hand in parse_state()
OPTIONAL_NUMERIC_FIELDS = ["thermal_R_ohm"]


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NoiseFloorPlan:
    daq_host: str
    daq_port: int
    leader: str
    follower: str
    ac_cfg: ACSourceConfig
    amplitudes_A: List[float]
    leader_extref_cfg: ExtRefConfig
    follower_extref_cfg: ExtRefConfig
    extref_lock_timeout_s: float
    demod_cfgs: List[NoiseDemodConfig]
    acq_cfg: AcquisitionConfig
    ref_cfg: ReferenceConfig
    also_measure_off: bool
    sample: str
    device: str
    cooldown: str
    series: str
    data_root: Path = _DEFAULT_DATA_DIR
    run_cost: Optional[RunCost] = None      # modelled seconds per step (progress bar + ETA)

    @property
    def steps_per_amplitude(self) -> int:
        return len(self.demod_cfgs) * (2 if self.also_measure_off else 1)

    @property
    def total_steps(self) -> int:
        return self.steps_per_amplitude * len(self.amplitudes_A)


_N_CHANNELS = 2   # leader 1f + follower 2f, as built in _build_plan()


def run_costs(state: dict) -> RunCost:
    """Modelled cost of the whole run; one entry per spectrum ("step"), in loop
    order: per amplitude, Excitation ON leader / follower, then (if enabled) OFF
    leader / follower. Also drives the run screen's progress bar."""
    n_amps = max(1, len(state.get("amplitude_list", [])))
    steps = _N_CHANNELS * (2 if state["also_measure_off"] else 1)
    rc = RunCost(steps * n_amps)
    duration = state["duration_s"]
    # acquire_time_series(): the recording is polled in poll_chunk_s chunks, each followed by an
    # overload read + MDS check; subscribe / sync / unsubscribe once; then two Welch estimates.
    n_chunks = math.ceil(duration / AcquisitionConfig().poll_chunk_s)
    rc.each("recording", duration)
    rc.each("overhead", 2 * n_chunks * GPIB_TXN_S + ACQ_OVERHEAD_S + POINT_OVERHEAD_S)
    rc.at("connect + MDS", PER_RUN_S + MDS_SYNC_S, 0)
    lock_typ, lock_worst = extref_lock_s(state["extref_lock_timeout_s"])
    for a in range(n_amps):
        # each amplitude arms the 6221 and locks both ExtRef PLLs before its first recording ...
        rc.at("6221 + ExtRef", ac_source_restart_s() + lock_typ, a * steps, worst_extra=lock_worst - lock_typ)
        if a:   # ... and the previous amplitude's save_results() (one file per spectrum + PNG) ran just before
            rc.at("save", steps * PER_FILE_S, a * steps)
    rc.tail("save", steps * PER_FILE_S)
    return rc


def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (info, warnings, errors) for a fully-parsed state dict."""
    info: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    dir_warn, dir_err = validate_directory(state.get("data_dir", ""))
    if dir_err:
        errors.append(f"Data root: {dir_err}")
    elif dir_warn:
        warnings.append(f"Data root: {dir_warn}")
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL:
        errors.append("Choose a sample (or create a new one).")
    if not state.get("device"):
        errors.append("Device is required (e.g. HB3, SV2).")
    if state["leader_device"] == state["follower_device"]:
        errors.append("Leader and follower device IDs must be different.")

    # ── Excitation (6221) ───────────────────────────────────────────────────
    if state.get("amplitude_parse_error"):
        errors.append(f"Excitation current list: {state['amplitude_parse_error']}")
    else:
        amp_list = state.get("amplitude_list", [])
        over_limit = [i for i in amp_list if not 0 < i <= _AC_CURRENT_CEILING_A]
        if over_limit:
            errors.append(
                f"Excitation current(s) {over_limit} must be in "
                f"(0, {format_si(_AC_CURRENT_CEILING_A, 'A')}] — check for a mistyped exponent."
            )
        elif len(amp_list) > 1:
            info.append(f"Excitation currents {amp_list} A peak — {len(amp_list)} complete "
                        "noise-floor sessions, one file set each.")
        elif amp_list:
            info.append(f"Excitation current I = {format_si(amp_list[0], 'A')} peak — match "
                         "your real HARM6 operating point for this estimate to be meaningful.")
    if not 0 < state["ac_compliance_V"] <= _AC_COMPLIANCE_CEILING_V:
        errors.append(
            f"6221 compliance must be in (0, {_AC_COMPLIANCE_CEILING_V:g}] V; "
            f"got {state['ac_compliance_V']:g} V."
        )

    f = state["frequency_Hz"]
    for label, check_f in (("1f", f), ("2f", 2 * f)):
        for mains in (50, 60):
            nearest = round(check_f / mains) * mains
            if nearest > 0 and abs(check_f - nearest) < 0.5:
                warnings.append(
                    f"{label} ({check_f:g} Hz) is within 0.5 Hz of a {mains} Hz "
                    f"harmonic ({nearest} Hz) — mains pickup risk."
                )

    if state["leader_pll_demod_index"] == 0:
        errors.append("Leader PLL phase-detector demod index must differ from 0 "
                       "(demod 0 is the noise-survey signal demod).")
    if state["follower_pll_demod_index"] == 0:
        errors.append("Follower PLL phase-detector demod index must differ from 0 "
                       "(demod 0 is the noise-survey signal demod).")

    # ── Reference / duration ─────────────────────────────────────────────────
    if state["thermal_R_ohm"] is not None:
        thermal = thermal_noise_asd(state["thermal_R_ohm"], state["thermal_T_K"])
        info.append(f"Johnson-noise reference @ {state['thermal_R_ohm']:g} Ω, "
                     f"{state['thermal_T_K']:g} K: {thermal:.3g} V/√Hz")
    else:
        info.append("No reference resistance set — plot will show the measured "
                     "floor with no Johnson-noise comparison line.")

    n_passes = 2 if state["also_measure_off"] else 1
    n_amps = max(1, len(state.get("amplitude_list", [])))
    amp_note = f" × {n_amps} excitation current(s)" if n_amps > 1 else ""
    info.append(f"{n_passes} pass(es) × {_N_CHANNELS} channels{amp_note}")
    info.extend(run_costs(state).lines("Estimated total run time"))
    if state["duration_s"] < 10:
        warnings.append(f"Duration {state['duration_s']:g} s is short — the lowest "
                         f"resolvable frequency is ~1/duration ≈ {1/state['duration_s']:.2g} Hz.")

    return info, warnings, errors


def _automode_select(field_id: str) -> list:
    """Small local Select builder -- the base select_field() in
    mfli_dual_harmonic_tui.py only takes plain int options, not the
    (label, value) pairs a descriptive automode dropdown needs."""
    label = Label("PID bandwidth mode", classes="field-label")
    sel = Select([("2 — low bandwidth", 2), ("3 — high bandwidth", 3), ("4 — dynamic (auto)", 4)],
                 id=field_id, value=int(DEFAULTS[field_id]), allow_blank=False)
    hint = Label(AUTOMODE_HINT, classes="hint")
    hint.styles.margin = (0, 0, 1, 0)
    return [label, sel, hint]


def compute_filename_preview(state: dict) -> Optional[str]:
    if not state.get("sample") or state["sample"] == NEW_SAMPLE_SENTINEL or not state.get("device"):
        return None
    n_amps = max(1, len(state.get("amplitude_list", [])))
    n_files = 2 * (2 if state["also_measure_off"] else 1) * n_amps
    return f"{state['sample']}_NNNN_{state['device']}_{MEASUREMENT_TYPE}_<timestamp>.csv  (×{n_files} files)"


# ─────────────────────────────────────────────────────────────────────────────
# Run screen  ── executes the plan in a worker thread, shows live progress
# ─────────────────────────────────────────────────────────────────────────────

class RunScreen(Screen):
    CSS = """
    #status_line { height: 1; padding: 0 1; text-style: bold; }
    #progress_row { height: auto; margin: 1 2; align: left middle; }
    #progress { margin: 0; }
    #log { height: 1fr; margin: 0 2 1 2; border: solid $primary; }
    #runactionbar { height: 3; align: center middle; }
    """
    BINDINGS = [Binding("q", "back", "Back", show=True)]

    def __init__(self, plan: NoiseFloorPlan) -> None:
        super().__init__()
        self.plan = plan
        self._measurement_running = True
        self._log_handler: Optional[_LogRelay] = None
        # Accumulated across every amplitude iteration -- ((cond, label), spec)
        # paired with its already-saved RunContext, so the end-of-session
        # status/comment prompt covers every file regardless of how many
        # amplitude values were run.
        self._context_pairs: list = []
        self._n_done = 0          # spectra finished so far -> index into plan.run_cost

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Starting …", id="status_line")
        with Horizontal(id="progress_row"):
            yield ProgressBar(id="progress", total=progress_total(self.plan.run_cost, self.plan.total_steps),
                              show_eta=True)
        yield RichLog(id="log", max_lines=5000, markup=False, wrap=True)
        with Horizontal(id="runactionbar"):
            yield Button("Back", id="back_btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._log_handler = _LogRelay(self)
        logging.getLogger().addHandler(self._log_handler)
        self.do_run()

    def on_unmount(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)

    def write_log(self, msg: str, style: str) -> None:
        self.query_one("#log", RichLog).write(Text(msg, style=style))

    def _set_status(self, text: str) -> None:
        self.query_one("#status_line", Static).update(text)

    def _on_result(self, cond: str, label: str, spec: dict) -> None:
        self.query_one("#progress", ProgressBar).advance(progress_step(self.plan.run_cost, self._n_done))
        self._n_done += 1
        thermal_note = ""
        if self.plan.ref_cfg.thermal_R_ohm:
            thermal = thermal_noise_asd(self.plan.ref_cfg.thermal_R_ohm, self.plan.ref_cfg.thermal_T_K)
            thermal_note = f"  (Johnson ref {thermal:.3g} V/√Hz)"
        self.write_log(
            f"{cond} — {label}: white floor {spec['white_floor_V_rthz']:.3e} V/√Hz{thermal_note}",
            "bold",
        )

    def _save_iteration(self, amp: float, iter_results: dict, iter_status: str) -> None:
        """Save ONE amplitude's results (own run(s), own file(s)) -- called
        right after that amplitude's measure_noise_floor() returns/raises,
        matching the unconditional-per-iteration-finalize convention used
        by every other multi-file suite in this codebase."""
        if not iter_results:
            return
        plan = self.plan
        key_axis = ("current_A", amp) if len(plan.amplitudes_A) > 1 else None
        contexts = save_results(
            iter_results, sample=plan.sample, device=plan.device,
            cooldown=plan.cooldown, series=plan.series, status=iter_status,
            data_root=plan.data_root, key_axis=key_axis,
        )
        # Zip in save_results()'s own iteration order (see its docstring) so
        # a later status/comment update can rebuild each run's full header
        # -- finalize_index_row() rewrites the whole index.csv row, so a
        # partial {"comment": ...} dict would blank out every other column.
        self._context_pairs.extend(zip(iter_results.items(), contexts))
        report_mains_peaks(iter_results, plan.ref_cfg)
        try:
            first, last = contexts[0], contexts[-1]
            run_label = first.run_str if first is last else f"{first.run_str}-{last.run_str}"
            png_path = proc_path(plan.data_root, plan.sample, run_label, plan.device,
                                  MEASUREMENT_TYPE, "combined", combined=True)
            import matplotlib
            matplotlib.use("Agg")
            plot_results(iter_results, plan.demod_cfgs, plan.ref_cfg, png_path)
            self.app.call_from_thread(self.write_log, f"Saved plot: {png_path}", "")
        except Exception:
            log.exception("Could not save noise-floor plot PNG")

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False

        if not self._context_pairs:
            log.warning("No results collected — nothing to save.")
            return

        self.app.push_screen(StatusCommentScreen(), self._on_status_comment)

    def _on_status_comment(self, result: Optional[tuple[str, str]]) -> None:
        if result is None:
            return
        status, comment = result
        plan = self.plan
        for (key, spec), ctx in self._context_pairs:
            cond, label = key
            try:
                finalize_comment(ctx, cond, label, spec, cooldown=plan.cooldown,
                                  series=plan.series, status=status, comment=comment,
                                  data_root=plan.data_root)
            except Exception:
                log.exception("Could not save status/comment for run %s", ctx.run_str)

    def action_back(self) -> None:
        if not self._measurement_running:
            self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()

    @work(thread=True, exclusive=True)
    def do_run(self) -> None:
        plan = self.plan
        daq = None
        try:
            self._set_status_threadsafe("Connecting to LabOne data server …")
            daq = connect(plan.daq_host, plan.daq_port)
            connect_device(daq, plan.leader, interface="1GbE")
            connect_device(daq, plan.follower, interface="1GbE")

            self._set_status_threadsafe("Synchronizing MDS …")
            mds = setup_mds(daq, leader=plan.leader, follower=plan.follower)

            self._set_status_threadsafe("Configuring noise-survey demodulators …")
            for cfg in plan.demod_cfgs:
                configure_noise_demod(daq, cfg)

            multi = len(plan.amplitudes_A) > 1
            for amp in plan.amplitudes_A:
                plan.ac_cfg.amplitude_A = amp
                if multi:
                    self._set_status_threadsafe(f"Excitation current {amp:g} A …")

                iter_results: dict = {}

                def on_result(cond, label, spec, _r=iter_results):
                    _r[(cond, label)] = spec
                    self.app.call_from_thread(self._on_result, cond, label, spec)

                iter_error: Optional[BaseException] = None
                try:
                    measure_noise_floor(
                        daq, plan.ac_cfg, plan.leader_extref_cfg, plan.follower_extref_cfg,
                        plan.demod_cfgs, plan.acq_cfg,
                        also_measure_off=plan.also_measure_off,
                        extref_lock_timeout_s=plan.extref_lock_timeout_s,
                        mds=mds,
                        on_status=self._set_status_threadsafe,
                        on_result=on_result,
                    )
                except Exception as exc:
                    iter_error = exc

                iter_status = "error" if iter_error is not None else "completed"
                self._save_iteration(amp, iter_results, iter_status)

                if iter_error is not None:
                    raise iter_error

            final = "Noise floor estimate complete."
        except Exception as exc:
            log.exception("Noise floor estimate failed")
            final = f"ERROR: {exc}"
        finally:
            self.app.call_from_thread(self._on_finished, final)

    def _set_status_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_status, text)


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class MFLINoiseSpectrumApp(App):
    TITLE = "MFLI Noise Floor Estimate (6221 AC source)"
    SUB_TITLE = "Quick nV/√Hz check for lock-in filter selection"

    data_root: Path = _DEFAULT_DATA_DIR

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 48; border-left: solid $primary; padding: 1 2; overflow-y: auto; }
    .field-label { text-style: bold; }
    .hint { text-style: italic; color: $text-muted; }
    .switch-row { height: 3; }
    .switch-row Label { margin-left: 1; content-align: left middle; height: 3; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; }
    #actionbar { height: 3; align: center middle; }

    #identity_bar { border: round $accent; padding: 1 2; margin-bottom: 1; height: auto; }
    #filename_preview { margin-bottom: 1; }
    #data_dir_row { height: 3; margin-bottom: 1; }
    #data_dir_row Input { width: 1fr; }
    #data_dir_row Button { margin-left: 1; }
    #identity_fields { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }
    #identity_fields > Vertical { height: auto; }
    .param-grid { layout: grid; grid-size: 2; grid-gutter: 1 2; height: auto; }
    .param-card { border: round $primary; padding: 1 2; height: auto; }
    .stable-grid { layout: grid; grid-size: 3; grid-gutter: 1 2; height: auto; }

    Collapsible { height: auto; margin: 1 0; }
    Collapsible > Contents { padding: 1 0 0 1; }
    CollapsibleTitle { text-style: bold; color: $text-muted; }
    .stable-card { border: round $panel-darken-1; padding: 1 2; height: auto; }
    .stable-card .card-title { color: $text-muted; text-style: bold underline; }
    .stable-card .field-label { color: $text-muted; }
    .card-title { text-style: bold underline; margin-bottom: 1; }
    """

    BINDINGS = [
        Binding("f5", "start", "Start estimate", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                with Vertical(id="identity_bar"):
                    yield Static(id="filename_preview")
                    with Horizontal(id="data_dir_row"):
                        yield Input(value=str(_DEFAULT_DATA_DIR), id="data_dir",
                                    placeholder="Absolute path to the data root")
                        yield Button("Browse…", id="browse_data_dir")
                    with Vertical(id="identity_fields"):
                        yield Vertical(
                            Label("Sample", classes="field-label"),
                            Select(sample_options(self.data_root), id="sample_select",
                                   allow_blank=False, value=TEST_SAMPLE),
                        )
                        yield Vertical(*field("device", "Device (e.g. HB3, SV2)",
                                              DEFAULTS["device"], kind="text"))
                        yield Vertical(*field("cooldown", "Cooldown (optional)",
                                              DEFAULTS["cooldown"], kind="text"))

                with Vertical(classes="param-grid"):
                    yield card(
                        "Excitation — match your real HARM6 operating point",
                        field("frequency_Hz", "Excitation frequency (Hz)",
                              DEFAULTS["frequency_Hz"],
                              validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                        field("amplitude_values", "Excitation current (A, peak)",
                              DEFAULTS["amplitude_values"], kind="text",
                              hint="Single value, or comma-separated list — one complete "
                                   "noise-floor session runs per value, each saved to its "
                                   "own file(s)."),
                    )
                    yield card(
                        "Reference & duration",
                        field("thermal_R_ohm", "DUT resistance (Ω, optional — Johnson-noise line)",
                              DEFAULTS["thermal_R_ohm"], kind="number", valid_empty=True,
                              hint="No physical resistor swap needed — just the DUT's approximate R."),
                        field("thermal_T_K", "Temperature for that comparison (K)",
                              DEFAULTS["thermal_T_K"]),
                        field("duration_s", "Recording duration per pass (s)",
                              DEFAULTS["duration_s"],
                              hint="Sets the lowest resolvable frequency (~1/duration).",
                              validators=[Number(minimum=1.0, failure_description="must be ≥ 1")]),
                        switch_field("also_measure_off", "Also record Excitation-OFF baseline",
                                     DEFAULTS["also_measure_off"]),
                    )

                with Collapsible(title="Noise-survey demodulator settings", collapsed=True):
                    with Vertical(classes="param-grid"):
                        yield card(
                            "Filter (deliberately wide-open)",
                            field("time_constant_s", "Time constant (s)",
                                  DEFAULTS["time_constant_s"],
                                  hint="Short TC = wide bandwidth for the noise survey — "
                                       "unrelated to the production filter setting.",
                                  validators=[Number(minimum=1e-9, failure_description="must be > 0")]),
                            field("sample_rate_Hz", "Demodulator sample rate (Sa/s)",
                                  DEFAULTS["sample_rate_Hz"],
                                  validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                        )
                        yield card(
                            "Input range",
                            field("input_range_1f_V", "1f channel input range (V)",
                                  DEFAULTS["input_range_1f_V"],
                                  validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                            field("input_range_2f_V", "2f channel input range (V)",
                                  DEFAULTS["input_range_2f_V"],
                                  validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                        )

                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Devices & connection",
                            field("leader_device", "Leader MFLI (1f)",
                                  DEFAULTS["leader_device"], kind="text"),
                            field("follower_device", "Follower MFLI (2f)",
                                  DEFAULTS["follower_device"], kind="text"),
                            field("daq_host", "LabOne data server host",
                                  DEFAULTS["daq_host"], kind="text"),
                            field("daq_port", "LabOne data server port",
                                  DEFAULTS["daq_port"], kind="integer"),
                            muted=True,
                        )
                        yield card(
                            "6221 & ExtRef (phase marker → both MFLIs' Aux In)",
                            field("ac_visa_resource", "6221 VISA resource",
                                  DEFAULTS["ac_visa_resource"], kind="text"),
                            field("ac_compliance_V", "6221 voltage compliance (V)",
                                  DEFAULTS["ac_compliance_V"]),
                            field("phasemarker_line", "6221 Trigger Link phase-marker pin",
                                  DEFAULTS["phasemarker_line"], kind="integer",
                                  validators=[Number(minimum=1, maximum=6,
                                                     failure_description="must be 1-6")]),
                            field("extref_lock_timeout_s", "ExtRef PLL lock timeout (s)",
                                  DEFAULTS["extref_lock_timeout_s"]),
                            muted=True,
                        )
                        yield card(
                            "Leader ExtRef",
                            field("leader_extref_index", "ExtRef module index",
                                  DEFAULTS["leader_extref_index"], kind="integer"),
                            field("leader_aux_input_ch", "Aux In channel (0 = Aux In 1)",
                                  DEFAULTS["leader_aux_input_ch"], kind="integer"),
                            field("leader_osc_index", "Oscillator index",
                                  DEFAULTS["leader_osc_index"], kind="integer"),
                            field("leader_pll_demod_index", "PLL phase-detector demod index",
                                  DEFAULTS["leader_pll_demod_index"], kind="integer",
                                  hint="Must differ from 0 (the noise-survey signal demod)."),
                            _automode_select("leader_automode"),
                            muted=True,
                        )
                        yield card(
                            "Follower ExtRef",
                            field("follower_extref_index", "ExtRef module index",
                                  DEFAULTS["follower_extref_index"], kind="integer"),
                            field("follower_aux_input_ch", "Aux In channel (0 = Aux In 1)",
                                  DEFAULTS["follower_aux_input_ch"], kind="integer"),
                            field("follower_osc_index", "Oscillator index",
                                  DEFAULTS["follower_osc_index"], kind="integer"),
                            field("follower_pll_demod_index", "PLL phase-detector demod index",
                                  DEFAULTS["follower_pll_demod_index"], kind="integer",
                                  hint="Must differ from 0 (the noise-survey signal demod)."),
                            _automode_select("follower_automode"),
                            muted=True,
                        )

            with Vertical(id="sidebar"):
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start estimate  (F5)", id="start", variant="success")
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        logging.getLogger().handlers.clear()
        self._load_settings()
        self.refresh_summary()

    # ── Sample picker ────────────────────────────────────────────────────────

    def _refresh_sample_options(self, *, select_value: Optional[str] = None) -> None:
        select = self.query_one("#sample_select", Select)
        options = sample_options(self.data_root)
        select.set_options(options)
        if select_value is not None:
            select.value = select_value

    def _sync_data_root(self) -> None:
        path = Path(self.query_one("#data_dir", Input).value.strip()).expanduser()
        if not path.is_dir():
            return
        self.data_root = path.resolve()
        opts = [v for _, v in sample_options(self.data_root)]
        cur = self.query_one("#sample_select", Select).value
        self._refresh_sample_options(select_value=cur if cur in opts else TEST_SAMPLE)

    def _browse_data_dir(self) -> None:
        start = self.query_one("#data_dir", Input).value.strip() or str(_DEFAULT_DATA_DIR)
        self.push_screen(DataDirPickerScreen(start), self._on_data_dir_picked)

    def _on_data_dir_picked(self, picked: Optional[str]) -> None:
        if not picked:
            return
        self.query_one("#data_dir", Input).value = picked
        self._sync_data_root()
        self.refresh_summary()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sample_select" and event.value == NEW_SAMPLE_SENTINEL:
            self.push_screen(NewSampleScreen(self.data_root), self._on_new_sample_created)
            return
        self.refresh_summary()

    def _on_new_sample_created(self, result: Optional[str]) -> None:
        self._refresh_sample_options(select_value=result if result else TEST_SAMPLE)
        self.refresh_summary()

    # ── Form state I/O ───────────────────────────────────────────────────────

    def _all_field_ids(self) -> list[str]:
        return list(NUMERIC_FIELDS) + TEXT_FIELDS + OPTIONAL_NUMERIC_FIELDS

    def collect_raw(self) -> dict:
        raw: dict = {fid: self.query_one(f"#{fid}", Input).value for fid in self._all_field_ids()}
        raw["also_measure_off"] = self.query_one("#also_measure_off", Switch).value
        raw["leader_automode"] = self.query_one("#leader_automode", Select).value
        raw["follower_automode"] = self.query_one("#follower_automode", Select).value
        sample_value = self.query_one("#sample_select", Select).value
        if sample_value not in (None, Select.BLANK, NEW_SAMPLE_SENTINEL):
            raw["sample"] = sample_value
        return raw

    def _load_settings(self) -> None:
        try:
            saved = json.loads(SETTINGS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        for fid in self._all_field_ids():
            if fid in saved:
                try:
                    self.query_one(f"#{fid}", Input).value = str(saved[fid])
                except Exception:
                    pass
        if "also_measure_off" in saved:
            self.query_one("#also_measure_off", Switch).value = bool(saved["also_measure_off"])
        for fid in ("leader_automode", "follower_automode"):
            if fid in saved:
                try:
                    self.query_one(f"#{fid}", Select).value = int(saved[fid])
                except Exception:
                    pass
        self._sync_data_root()
        saved_sample = saved.get("sample")
        if saved_sample and saved_sample in [v for _, v in sample_options(self.data_root)]:
            self.query_one("#sample_select", Select).value = saved_sample

    def _save_settings(self, raw: dict) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(raw, indent=2))
        except OSError:
            pass

    def parse_state(self) -> tuple[dict, list[str]]:
        errors: list[str] = []
        state: dict = {}
        for fid, caster in NUMERIC_FIELDS.items():
            raw = self.query_one(f"#{fid}", Input).value.strip()
            try:
                state[fid] = caster(raw)
            except ValueError:
                errors.append(f"'{fid}' is not a valid number: {raw!r}")
                state[fid] = 0
        for fid in TEXT_FIELDS:
            state[fid] = self.query_one(f"#{fid}", Input).value.strip()
        for fid in OPTIONAL_NUMERIC_FIELDS:
            raw = self.query_one(f"#{fid}", Input).value.strip()
            if raw:
                try:
                    state[fid] = float(raw)
                except ValueError:
                    errors.append(f"'{fid}' is not a valid number: {raw!r}")
                    state[fid] = None
            else:
                state[fid] = None
        state["also_measure_off"] = self.query_one("#also_measure_off", Switch).value
        state["leader_automode"] = int(self.query_one("#leader_automode", Select).value)
        state["follower_automode"] = int(self.query_one("#follower_automode", Select).value)
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""

        state["amplitude_list"] = []
        state["amplitude_parse_error"] = None
        try:
            state["amplitude_list"] = parse_value_list(state["amplitude_values"])
        except ValueError as exc:
            state["amplitude_parse_error"] = str(exc)

        return state, errors

    # ── Reactivity ───────────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "data_dir":
            self._sync_data_root()
        self.refresh_summary()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.refresh_summary()

    def refresh_summary(self) -> None:
        state, parse_errors = self.parse_state()
        if parse_errors:
            info, warnings, errors = [], [], parse_errors
            preview = None
        else:
            info, warnings, errors = build_summary(state)
            preview = compute_filename_preview(state)

        self.query_one("#filename_preview", Static).update(
            f"File:  [bold]{preview}[/bold]" if preview
            else "[dim]File:  (choose a sample and device to preview the filename)[/dim]"
        )
        lines: list[str] = []
        if errors:
            lines.append("[bold red]Blocking issues[/bold red]")
            lines += [f"  [red]✗ {e}[/red]" for e in errors]
        if warnings:
            lines.append("[bold yellow]Warnings[/bold yellow]")
            lines += [f"  [yellow]⚠ {w}[/yellow]" for w in warnings]
        lines.append("[bold]Derived values[/bold]")
        lines += [f"  [dim]•[/dim] {i}" for i in info]

        self.query_one("#summary", Static).update("\n".join(lines))
        self.query_one("#start", Button).disabled = bool(errors)

    # ── Start ────────────────────────────────────────────────────────────────

    def action_start(self) -> None:
        state, parse_errors = self.parse_state()
        if parse_errors:
            self.bell()
            return
        _, _, errors = build_summary(state)
        if errors:
            self.bell()
            return

        self.data_root = Path(state["data_dir"]).expanduser()
        ensure_sample(self.data_root, state["sample"], create=True)
        self._save_settings(self.collect_raw())
        plan = self._build_plan(state)
        self.push_screen(RunScreen(plan))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()
        elif event.button.id == "browse_data_dir":
            self._browse_data_dir()

    def _build_plan(self, state: dict) -> NoiseFloorPlan:
        ac_cfg = ACSourceConfig(
            visa_resource=state["ac_visa_resource"],
            amplitude_A=state["amplitude_list"][0],
            frequency_Hz=state["frequency_Hz"],
            compliance_V=state["ac_compliance_V"],
            phasemarker_line=state["phasemarker_line"],
        )
        leader_extref_cfg = ExtRefConfig(
            device=state["leader_device"], extref_index=state["leader_extref_index"],
            aux_input_ch=state["leader_aux_input_ch"], osc_index=state["leader_osc_index"],
            pll_demod_index=state["leader_pll_demod_index"], automode=state["leader_automode"],
        )
        follower_extref_cfg = ExtRefConfig(
            device=state["follower_device"], extref_index=state["follower_extref_index"],
            aux_input_ch=state["follower_aux_input_ch"], osc_index=state["follower_osc_index"],
            pll_demod_index=state["follower_pll_demod_index"], automode=state["follower_automode"],
        )
        demod_cfgs = [
            NoiseDemodConfig(
                device=state["leader_device"], label="MFLI-1 (1f channel)", harmonic=1,
                input_range_V=state["input_range_1f_V"], sample_rate_Hz=state["sample_rate_Hz"],
                time_constant_s=state["time_constant_s"],
            ),
            NoiseDemodConfig(
                device=state["follower_device"], label="MFLI-2 (2f channel)", harmonic=2,
                input_range_V=state["input_range_2f_V"], sample_rate_Hz=state["sample_rate_Hz"],
                time_constant_s=state["time_constant_s"],
            ),
        ]
        acq_cfg = AcquisitionConfig(duration_s=state["duration_s"])
        ref_cfg = ReferenceConfig(thermal_R_ohm=state["thermal_R_ohm"], thermal_T_K=state["thermal_T_K"])
        series = f"{state['sample']}_{state['device']}_{MEASUREMENT_TYPE}_{datetime.now():%Y%m%dT%H%M%S}"

        return NoiseFloorPlan(
            daq_host=state["daq_host"], daq_port=state["daq_port"],
            leader=state["leader_device"], follower=state["follower_device"],
            ac_cfg=ac_cfg, amplitudes_A=state["amplitude_list"],
            leader_extref_cfg=leader_extref_cfg, follower_extref_cfg=follower_extref_cfg,
            extref_lock_timeout_s=state["extref_lock_timeout_s"],
            demod_cfgs=demod_cfgs, acq_cfg=acq_cfg, ref_cfg=ref_cfg,
            also_measure_off=state["also_measure_off"],
            sample=state["sample"], device=state["device"], cooldown=state["cooldown"],
            series=series, data_root=self.data_root, run_cost=run_costs(state),
        )


def main() -> None:
    MFLINoiseSpectrumApp().run()


if __name__ == "__main__":
    main()
