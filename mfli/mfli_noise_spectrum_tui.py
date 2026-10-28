#!/usr/bin/env python3
"""
Textual TUI front-end for mfli_noise_spectrum.py
===================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-17

Runs the 2-pass (Excitation ON / Excitation OFF) noise-floor estimate for
the mfli_dual_harmonic_6221 program without touching the dataclasses in the
script itself. Each MFLI's harmonic, input range and coupling are set to
match the real run's wiring — "Copy from Dual-Harmonic" pulls them (plus
frequency, current, offset, devices, ExtRef) from that form's saved state. Deliberately smaller than the other suite TUIs — this is a
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


from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.validation import Number
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    Switch,
)

from dc.dc_sweep_utils import try_parse
from mfli.mfli_dual_harmonic_6221 import _AC_CURRENT_CEILING_A, _AC_COMPLIANCE_CEILING_V, extref_lock_s
from instruments.summary_lines import summary_markup
from instruments.tui_common import (
    LogRelay, MeasurementApp, MeasurementRunScreen, card, field, format_si, identity_bar,
    select_field, switch_field,
)
import mfli.mfli_dual_harmonic_6221_tui as six
from mfli.mfli_dual_harmonic_tui import SETTINGS_PATH as HARM_SETTINGS_PATH
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
from instruments.data_dir import validate_directory
from instruments.keithley6221 import ac_source_restart_s
from instruments.run_time import (
    ACQ_OVERHEAD_S, GPIB_TXN_S, MDS_SYNC_S, PER_FILE_S, PER_RUN_S, POINT_OVERHEAD_S,
    RunCost, progress_step, progress_total,
)
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL,
    StatusCommentScreen,
)

log = logging.getLogger("mfli_noise_spectrum_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DEFAULT_DATA_DIR / "mfli_noise_spectrum_tui_settings.json"

MEASUREMENT_TYPE = "NOISE"

# One-paragraph blurb + wiring schematic — shown on this program's card in
# bridge_tui.py.
MFLI_NOISE_SPECTRUM_DESCRIPTION = (
    "2 MFLI + 6221, wired as for the 6221 dual-harmonic run (harmonics/ranges copyable from "
    "it). Excitation-ON floor + optional OFF baseline, no rewiring."
)

MFLI_NOISE_SPECTRUM_SCHEMATIC = """\
  Same wiring as Dual-Harmonic (6221 AC source) — nothing
  separate to cable for this tool.

  Keithley 6221  (WAVE, sine — the excitation to toggle ON/OFF)
    HI/LO ──▶ sample/DUT ── common ground
    Trigger Link phase marker ──▶ Aux In 1 on BOTH the leader AND follower

  LEADER MFLI    (ExtRef-locked, noise-survey demod at its harmonic)
  FOLLOWER MFLI  (ExtRef-locked, noise-survey demod at its harmonic)
    Harmonic / input / coupling = whatever the real run uses.
"""


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
    "ac_offset_A": "0",
    "phasemarker_line": "1",
    "extref_lock_timeout_s": "5.0",
    "leader_harmonic": "1",
    "leader_measure_rxx": False,
    "follower_harmonic": "2",
    "measure_rxx": False,
    "differential": True,
    "ac_coupling": True,
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

AUTOMODE_HINT = "2 = low bandwidth, 3 = high bandwidth, 4 = auto (default)."

NUMERIC_FIELDS: dict = {
    "daq_port": int,
    "frequency_Hz": float,
    "ac_compliance_V": float,
    "ac_offset_A": float,
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

# The Dual-Harmonic (6221) form's keys this form shares (same ids, same
# meaning): the wiring and operating point. The noise-only filter / rate /
# duration / Johnson settings are never copied.
HARM6_COPY_KEYS = (
    "leader_device", "follower_device", "daq_host", "daq_port",
    "ac_visa_resource", "frequency_Hz", "amplitude_values", "ac_compliance_V", "ac_offset_A",
    "phasemarker_line", "extref_lock_timeout_s",
    "leader_harmonic", "leader_measure_rxx", "follower_harmonic", "measure_rxx",
    "differential", "ac_coupling", "input_range_1f_V", "input_range_2f_V",
    "leader_extref_index", "leader_aux_input_ch", "leader_osc_index",
    "leader_pll_demod_index", "leader_automode",
    "follower_extref_index", "follower_aux_input_ch", "follower_osc_index",
    "follower_pll_demod_index", "follower_automode",
    "device", "cooldown", "sample", "data_dir",
)


def harm6_copy(saved: dict) -> dict:
    """The subset of a saved Dual-Harmonic form this form takes over."""
    saved = six.migrate_settings(dict(saved))
    return {k: saved[k] for k in HARM6_COPY_KEYS if k in saved}


def channel_label(role: str, harmonic: int, rxx: bool) -> str:
    """e.g. "Leader 2f", "Follower R_xx (1f)" -- the plot title and the
    index.csv channel_label, named like the HARM6 columns it sizes."""
    return f"{role} {six.demod_naming(harmonic, rxx)[1]}"


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
    lock_typ = extref_lock_s()
    for a in range(n_amps):
        # each amplitude arms the 6221 and locks both ExtRef PLLs before its first recording ...
        rc.at("6221 + ExtRef", ac_source_restart_s() + lock_typ, a * steps)
        if a:   # ... and the previous amplitude's save_results() (one file per spectrum + PNG) ran just before
            rc.at("save", steps * PER_FILE_S, a * steps)
    rc.tail("save", steps * PER_FILE_S)
    return rc


def resolve_state(state: dict) -> dict:
    """Add the derived keys build_summary() / build_plan() read — the parsed
    lists/sweeps, each with its parse error — to a state of raw field values.
    Pure: used by the TUI's parse_state()."""
    state["amplitude_list"], state["amplitude_parse_error"] = try_parse(state["amplitude_values"])
    return state


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
        elif amp_list and max(amp_list) + abs(state["ac_offset_A"]) > _AC_CURRENT_CEILING_A:
            errors.append(f"Excitation current + |DC offset| exceeds "
                          f"{format_si(_AC_CURRENT_CEILING_A, 'A')}.")
        elif len(amp_list) > 1:
            info.append(f"Excitation currents: {', '.join(format_si(i, 'A') for i in amp_list)} "
                        f"peak — {len(amp_list)} sessions, one file set each")
        elif amp_list:
            info.append(f"Excitation current: {format_si(amp_list[0], 'A')} peak — "
                        "match the real HARM6 operating point")
    if not 0 < state["ac_compliance_V"] <= _AC_COMPLIANCE_CEILING_V:
        errors.append(
            f"6221 compliance must be in (0, {_AC_COMPLIANCE_CEILING_V:g}] V; "
            f"got {state['ac_compliance_V']:g} V."
        )

    if state["ac_offset_A"]:
        warnings.append(f"6221 DC offset {format_si(state['ac_offset_A'], 'A')} on — "
                        "only if the real run uses it too.")

    f = state["frequency_Hz"]
    channels = (("Leader", state["leader_device"], state["leader_harmonic"], state["leader_measure_rxx"]),
                ("Follower", state["follower_device"], state["follower_harmonic"], state["measure_rxx"]))
    info.append("Channels: " + ", ".join(f"{channel_label(role, h, rxx)} ({dev})"
                                         for role, dev, h, rxx in channels))
    for role, _, h, rxx in channels:
        check_f = h * f
        for mains in (50, 60):
            nearest = round(check_f / mains) * mains
            if nearest > 0 and abs(check_f - nearest) < 0.5:
                warnings.append(
                    f"{channel_label(role, h, rxx)} ({check_f:g} Hz) is within 0.5 Hz of a "
                    f"{mains} Hz harmonic ({nearest} Hz) — mains pickup risk."
                )

    if state["leader_pll_demod_index"] == 0:
        errors.append("Leader PLL phase-detector demod index must differ from 0 "
                       "(demod 0 is the noise-survey signal demod).")
    if state["follower_pll_demod_index"] == 0:
        errors.append("Follower PLL phase-detector demod index must differ from 0 "
                       "(demod 0 is the noise-survey signal demod).")

    if state["thermal_R_ohm"] is not None:
        thermal = thermal_noise_asd(state["thermal_R_ohm"], state["thermal_T_K"])
        info.append(f"Johnson reference: {thermal:.3g} V/√Hz — "
                    f"{state['thermal_R_ohm']:g} Ω at {state['thermal_T_K']:g} K")
    else:
        info.append("Johnson reference: none — no R set")

    n_passes = 2 if state["also_measure_off"] else 1
    n_amps = max(1, len(state.get("amplitude_list", [])))
    amp_note = f" × {n_amps} excitation current(s)" if n_amps > 1 else ""
    info.append(f"Passes: {n_passes} × {_N_CHANNELS} channels{amp_note}")
    if state["duration_s"] <= 0:
        errors.append("Recording duration must be > 0 s.")
        return info, warnings, errors
    info.extend(run_costs(state).lines())
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

class RunScreen(MeasurementRunScreen):
    """A log-only run screen (no per-point table, no abort — a noise record
    can't stop mid-spectrum): shares the base's log/status plumbing and
    run-history row, keeps its own layout and per-amplitude saving."""
    CSS = """
    #status_line { height: 1; padding: 0 1; text-style: bold; }
    #progress_row { height: auto; margin: 1 2; align: left middle; }
    #progress { margin: 0; }
    #log { height: 1fr; margin: 0 2 1 2; border: solid $primary; }
    #runactionbar { height: 3; align: center middle; }
    """
    BINDINGS = [Binding("q", "back", "Back", show=True)]

    def __init__(self, plan: NoiseFloorPlan) -> None:
        super().__init__(plan)
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

    def on_mount(self, event: events.Mount) -> None:
        # Textual also dispatches Mount to every base class's on_mount; the
        # base one needs #results_table (not composed here) and would start
        # history + do_run a second time, so this handler replaces it.
        event.prevent_default()
        self._log_handler = LogRelay(self)
        logging.getLogger().addHandler(self._log_handler)
        self._history_start()
        self.do_run()

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
            data_root=plan.data_root, key_axis=key_axis, thermal_R_ohm=plan.ref_cfg.thermal_R_ohm,
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
        self._run_contexts = [ctx for _, ctx in self._context_pairs]
        self._history_finish("error" if final_status.startswith("ERROR") else "completed",
                             final_status, point_count=self._n_done)

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
                                  data_root=plan.data_root, thermal_R_ohm=plan.ref_cfg.thermal_R_ohm)
            except Exception:
                log.exception("Could not save status/comment for run %s", ctx.run_str)

    def action_back(self) -> None:
        if not self._measurement_running:
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


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class MFLINoiseSpectrumApp(MeasurementApp):
    TITLE = "MFLI Noise Floor Estimate (6221 AC source)"
    SUB_TITLE = "Quick nV/√Hz check for lock-in filter selection"

    data_root: Path = _DEFAULT_DATA_DIR

    BINDINGS = [
        Binding("f5", "start", "Start estimate", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 48; border-left: solid $primary; padding: 1 2; overflow-y: auto; }
    .field-label { text-style: bold; width: 100%; }
    .hint { text-style: italic; color: $text-muted; width: 100%; }
    .switch-row { height: auto; }
    .switch-row Label { padding-left: 1; content-align: left middle; width: 1fr; height: auto; min-height: 3; }
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                yield identity_bar(DEFAULTS, _DEFAULT_DATA_DIR, self.data_root,
                                   temperature_label=None,
                                   cell_classes=None)

                with Vertical(classes="param-grid"):
                    yield card(
                        "Excitation — match your real HARM6 operating point",
                        field("frequency_Hz", "Excitation frequency (Hz)",
                              DEFAULTS["frequency_Hz"],
                              validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                        field("amplitude_values", "Excitation current (A, peak)",
                              DEFAULTS["amplitude_values"], kind="text",
                              hint="Comma-separate for one session + file set per value."),
                        field("ac_offset_A", "DC offset (A)", DEFAULTS["ac_offset_A"],
                              hint="0 unless the real run uses one."),
                        Button("Copy from Dual-Harmonic", id="copy_harm6"),
                        Static("Takes wiring + operating point from the Dual-Harmonic (6221) "
                               "form's last start; leaves filter/duration alone.", classes="hint"),
                    )
                    for role, harmonic_id, rxx_id, range_id in (
                            ("Leader", "leader_harmonic", "leader_measure_rxx", "input_range_1f_V"),
                            ("Follower", "follower_harmonic", "measure_rxx", "input_range_2f_V")):
                        yield card(
                            f"{role} lock-in — as wired for the real run",
                            select_field(harmonic_id, "Harmonic", six.HARMONIC_OPTIONS,
                                         int(DEFAULTS[harmonic_id])),
                            switch_field(rxx_id, "R_xx leads (label only)", DEFAULTS[rxx_id]),
                            field(range_id, "Input range (V)", DEFAULTS[range_id],
                                  validators=[Number(minimum=1e-6, failure_description="must be > 0")]),
                        )
                    yield card(
                        "Signal input (both MFLIs)",
                        switch_field("differential", "Differential input (IN+/IN-)",
                                     DEFAULTS["differential"]),
                        switch_field("ac_coupling", "AC coupling", DEFAULTS["ac_coupling"]),
                    )
                    yield card(
                        "Reference & duration",
                        field("thermal_R_ohm", "DUT resistance (Ω, optional — Johnson-noise line)",
                              DEFAULTS["thermal_R_ohm"], kind="number", valid_empty=True,
                              hint="DUT's approximate R, for the Johnson line."),
                        field("thermal_T_K", "Temperature for that comparison (K)",
                              DEFAULTS["thermal_T_K"]),
                        field("duration_s", "Recording duration per pass (s)",
                              DEFAULTS["duration_s"],
                              hint="Lowest frequency ≈ 1/duration.",
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
                                  hint="Short = wide survey bandwidth.",
                                  validators=[Number(minimum=1e-9, failure_description="must be > 0")]),
                            field("sample_rate_Hz", "Demodulator sample rate (Sa/s)",
                                  DEFAULTS["sample_rate_Hz"],
                                  validators=[Number(minimum=1e-3, failure_description="must be > 0")]),
                        )

                with Collapsible(title="Instrument configuration & addresses", collapsed=True):
                    with Vertical(classes="stable-grid"):
                        yield card(
                            "Devices & connection",
                            field("leader_device", "Leader MFLI",
                                  DEFAULTS["leader_device"], kind="text"),
                            field("follower_device", "Follower MFLI",
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
                                  hint="Not 0 (survey signal demod)."),
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
                                  hint="Not 0 (survey signal demod)."),
                            _automode_select("follower_automode"),
                            muted=True,
                        )

            with Vertical(id="sidebar"):
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start estimate  (F5)", id="start", variant="success")
        yield Footer()

    # ── Form state I/O ───────────────────────────────────────────────────────

    # ── Reactivity ───────────────────────────────────────────────────────────

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.refresh_summary()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy_harm6":
            self._copy_from_harm6()
        else:
            super().on_button_pressed(event)

    def _copy_from_harm6(self) -> None:
        try:
            saved = json.loads(HARM_SETTINGS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.notify("No saved Dual-Harmonic form — start that program once first.",
                        severity="warning")
            return
        self._load_settings(harm6_copy(saved))
        self.refresh_summary()
        if saved.get("ac_source", "6221") != "6221":
            self.notify("Dual-Harmonic form was last in MFLI-source mode — this tool uses the "
                        "6221; check current/frequency.", severity="warning")
        else:
            self.notify("Copied wiring + operating point from the Dual-Harmonic form.")

    def update_summary(self) -> None:
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
        self.query_one("#summary", Static).update(summary_markup(info, warnings, errors))
        self.query_one("#start", Button).disabled = bool(errors)

    # ── Start ────────────────────────────────────────────────────────────────

    def _build_plan(self, state: dict) -> NoiseFloorPlan:
        ac_cfg = ACSourceConfig(
            visa_resource=state["ac_visa_resource"],
            amplitude_A=state["amplitude_list"][0],
            frequency_Hz=state["frequency_Hz"],
            compliance_V=state["ac_compliance_V"],
            phasemarker_line=state["phasemarker_line"],
            offset_A=state["ac_offset_A"],
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
                device=state[f"{p}_device"], label=channel_label(role, state[f"{p}_harmonic"], state[rxx]),
                harmonic=state[f"{p}_harmonic"], osc_index=state[f"{p}_osc_index"],
                differential=state["differential"], ac_coupling=state["ac_coupling"],
                input_range_V=state[range_id], sample_rate_Hz=state["sample_rate_Hz"],
                time_constant_s=state["time_constant_s"],
            )
            for role, p, rxx, range_id in (("Leader", "leader", "leader_measure_rxx", "input_range_1f_V"),
                                           ("Follower", "follower", "measure_rxx", "input_range_2f_V"))
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
