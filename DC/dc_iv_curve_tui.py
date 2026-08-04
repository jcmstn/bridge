#!/usr/bin/env python3
"""
Textual TUI front-end for dc_iv_curve.py
==========================================
Same house style as dc_hall_measurement_tui.py / mfli_diff_resistance_tui.py:
lets you edit the parameters that decide whether a DC I-V sweep is good
or bad — current range, compliance, voltmeter integration time, timing —
without touching the dataclasses in the script itself.

The sidebar recomputes derived values (estimated per-point acquisition
time, estimated sweep duration) and flags anything that risks a bad
measurement (source/voltmeter sharing a GPIB address, a degenerate
current range) as you type.

Run with:
    python dc_iv_curve_tui.py

Requirements:
    pip install textual matplotlib  (in addition to dc_iv_curve.py's own deps)
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
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
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Static,
    Switch,
)

from dc_iv_curve import (
    AcquisitionConfig,
    CurrentPoint,
    SourceConfig,
    VoltmeterConfig,
    bidirectional_current_sweep,
    connect_source,
    connect_voltmeter,
    ramp_current_to_zero,
    run_measurement,
    shutdown_source,
)

log = logging.getLogger("dc_iv_curve_tui")

# Data/settings live outside "bridge" (a sibling of it), same convention as
# dc_iv_curve.py.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SETTINGS_PATH = _DATA_DIR / "dc_iv_curve_tui_settings.json"


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions & defaults  ── mirrors dc_iv_curve.main()'s example
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: dict = {
    "source_visa_resource": "GPIB0::12::INSTR",
    "voltmeter_visa_resource": "GPIB0::7::INSTR",
    "compliance_V": "2.0",
    "source_delay_s": "0.05",
    "current_min_A": "-0.001",
    "current_max_A": "0.001",
    "nplc": "5",
    "auto_range": True,
    "settling_time_s": "0.2",
    "n_averages": "5",
    "output_name": "dc_iv_curve",
    "n_points": "41",
}

# id -> caster, for every free-text numeric field (Switch handled separately)
NUMERIC_FIELDS: dict = {
    "compliance_V": float,
    "source_delay_s": float,
    "current_min_A": float,
    "current_max_A": float,
    "nplc": float,
    "settling_time_s": float,
    "n_averages": int,
    "n_points": int,
}
TEXT_FIELDS = ["source_visa_resource", "voltmeter_visa_resource", "output_name"]


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_si(value: float, unit: str) -> str:
    """Format a value with an SI prefix, e.g. 1.2e-8 -> '12.000 nA'."""
    av = abs(value)
    if av == 0:
        return f"0 {unit}"
    for scale, prefix in ((1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, "")):
        if av < scale * 1000:
            return f"{value / scale:.3f} {prefix}{unit}"
    return f"{value:.3e} {unit}"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _reading_duration_s(nplc: float) -> float:
    """Rough per-reading time for the 2182 — NPLC/line_frequency, worst case 50 Hz."""
    return max(1e-3, nplc / 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# Measurement plan  ── built from validated form state, executed by RunScreen
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementPlan:
    src_cfg: SourceConfig
    volt_cfg: VoltmeterConfig
    acq_cfg: AcquisitionConfig
    currents_A: np.ndarray

    @property
    def total_points(self) -> int:
        return len(self.currents_A)


# ─────────────────────────────────────────────────────────────────────────────
# Small widget-building helpers (keep compose() readable)
# ─────────────────────────────────────────────────────────────────────────────

def field(field_id: str, label_text: str, default: str, *, kind: str = "number",
          hint: str = "", validators=None) -> Vertical:
    children = [Label(label_text, classes="field-label"),
                Input(value=default, id=field_id, type=kind, validators=validators,
                      valid_empty=False)]
    if hint:
        children.append(Label(hint, classes="hint"))
    return Vertical(*children, classes="field")


def switch_field(field_id: str, label_text: str, default: bool) -> Vertical:
    return Vertical(
        Horizontal(Switch(value=default, id=field_id), Label(label_text, classes="switch-label"),
                   classes="switch-row"),
        classes="field",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Live validation / derived-value summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(state: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (info, warnings, errors) for a fully-parsed state dict."""
    info: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    if state["source_visa_resource"] == state["voltmeter_visa_resource"]:
        errors.append("Source (6221) and voltmeter (2182) VISA resources must be different.")

    if state["compliance_V"] <= 0:
        errors.append("Compliance voltage must be > 0 V.")

    if state["current_min_A"] == state["current_max_A"]:
        warnings.append("current_min equals current_max — sweep will repeat a single point.")

    info.append(f"Current range: {format_si(state['current_min_A'], 'A')} → "
                f"{format_si(state['current_max_A'], 'A')}")

    read_s = _reading_duration_s(state["nplc"])
    info.append(f"Estimated 2182 reading time ≈ {read_s * 1000:.0f} ms (NPLC={state['nplc']:g})")

    per_point_s = state["settling_time_s"] + state["n_averages"] * read_s

    if state["n_points"] < 2:
        errors.append("Points per sweep direction must be ≥ 2.")
    total_points = max(0, 2 * state["n_points"] - 1)
    info.append(
        f"Sweep: {state['current_min_A']:g} A → {state['current_max_A']:g} A → "
        f"{state['current_min_A']:g} A, {total_points} points (bidirectional — reveals hysteresis)"
    )
    info.append(f"Estimated total run time ≈ {format_duration(total_points * per_point_s)}")

    return info, warnings, errors


# ─────────────────────────────────────────────────────────────────────────────
# Live plot  ── runs in its own OS process, well away from the TUI
# ─────────────────────────────────────────────────────────────────────────────
# A GUI matplotlib backend and Textual's terminal control both want the main
# thread. Rather than fight that, the live preview gets its own process with
# its own main thread; new points are streamed to it over a
# multiprocessing.Queue. The final two-panel PNG (I-V + numerical dV/dI) is
# saved independently by dc_iv_curve.plot_results once the full sweep is in
# hand — see _save_measurement_png below.

def _live_plot_worker(queue: "mp.Queue") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        fig.canvas.manager.set_window_title("DC I-V live measurement")
    except Exception:
        pass
    line, = ax.plot([], [], "o-", color="#2E3192")
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Live measurement — I-V curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    xs: list[float] = []
    ys: list[float] = []

    def _drain(_frame=None):
        updated = False
        while True:
            try:
                record = queue.get_nowait()
            except Exception:
                break
            xs.append(record["current_A"])
            ys.append(record["voltage_V"])
            updated = True
        if updated:
            line.set_data(xs, ys)
            ax.relim()
            ax.autoscale_view()
        return (line,)

    # Keep a reference so it isn't garbage-collected mid-run.
    _ani = FuncAnimation(fig, _drain, interval=300, cache_frame_data=False)
    plt.show()


def _save_measurement_png(records: list[dict], csv_path: str) -> None:
    """Save the two-panel I-V / dV/dI summary PNG from whatever points were
    actually collected (including an aborted/partial run)."""
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")  # headless — must not touch the TUI's terminal
    import pandas as pd
    from dc_iv_curve import plot_results

    df = pd.DataFrame(records)
    png_path = Path(csv_path).with_suffix(".png")
    plot_results(df, png_path)


# ─────────────────────────────────────────────────────────────────────────────
# Logging -> RichLog relay (keeps raw log lines from corrupting the alt screen)
# ─────────────────────────────────────────────────────────────────────────────

class _LogRelay(logging.Handler):
    def __init__(self, screen: "RunScreen") -> None:
        super().__init__()
        self.screen = screen
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                             datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        style = "bold red" if record.levelno >= logging.ERROR \
            else "bold yellow" if record.levelno >= logging.WARNING else ""
        try:
            self.screen.app.call_from_thread(self.screen.write_log, msg, style)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Run screen  ── executes the plan in a worker thread, shows live progress
# ─────────────────────────────────────────────────────────────────────────────

class RunScreen(Screen):
    CSS = """
    #status_line { height: 1; padding: 0 1; text-style: bold; }
    #progress { margin: 1 2; }
    #results_table { height: 12; margin: 0 2 1 2; }
    #log { height: 1fr; margin: 0 2 1 2; border: solid $primary; }
    #runactionbar { height: 3; align: center middle; }
    """
    BINDINGS = [
        Binding("a", "abort", "Abort (safe ramp-down)", show=True),
        Binding("q", "back_or_abort", "Abort / Back", show=True),
    ]

    def __init__(self, plan: MeasurementPlan) -> None:
        super().__init__()
        self.plan = plan
        self._stop_event = threading.Event()
        # Not named `_running` — that attribute already exists on Textual's
        # MessagePump base class and shadowing it silently breaks mounting.
        self._measurement_running = True
        self._log_handler: Optional[_LogRelay] = None
        self._records: list[dict] = []
        self._plot_queue: Optional["mp.Queue"] = None
        self._plot_process: Optional[mp.Process] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Starting …", id="status_line")
        yield ProgressBar(id="progress", total=self.plan.total_points, show_eta=False)
        yield DataTable(id="results_table", zebra_stripes=True, cursor_type="row")
        yield RichLog(id="log", max_lines=5000, markup=False, wrap=True)
        with Horizontal(id="runactionbar"):
            yield Button("Abort (safe ramp-down)", id="abort_btn", variant="error")
            yield Button("Back", id="back_btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#results_table", DataTable).add_columns(
            "#", "I (A)", "V (V)", "R (Ω)"
        )
        self._log_handler = _LogRelay(self)
        root = logging.getLogger()
        root.addHandler(self._log_handler)
        self._start_live_plot()
        self.do_run()

    def on_unmount(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
        if self._plot_process is not None and self._plot_process.is_alive():
            self._plot_process.terminate()

    def _start_live_plot(self) -> None:
        try:
            ctx = mp.get_context("spawn")
            self._plot_queue = ctx.Queue()
            self._plot_process = ctx.Process(
                target=_live_plot_worker,
                args=(self._plot_queue,),
                daemon=True,
            )
            self._plot_process.start()
        except Exception:
            log.exception("Could not start live plot window (is matplotlib installed?)")
            self._plot_queue = None
            self._plot_process = None

    def write_log(self, msg: str, style: str) -> None:
        self.query_one("#log", RichLog).write(Text(msg, style=style))

    def _set_status(self, text: str) -> None:
        self.query_one("#status_line", Static).update(text)

    def _on_point(self, record: dict) -> None:
        self._records.append(record)
        if self._plot_queue is not None:
            try:
                self._plot_queue.put_nowait(record)
            except Exception:
                pass
        table = self.query_one("#results_table", DataTable)
        table.add_row(
            str(record["point_index"] + 1),
            f"{record['current_A']:.4e}",
            f"{record['voltage_V']:.4e}",
            f"{record['resistance_ohm']:.5g}",
        )
        table.move_cursor(row=table.row_count - 1, scroll=True)
        self.query_one("#progress", ProgressBar).advance(1)
        self._set_status(f"Point {record['point_index'] + 1} / {self.plan.total_points} complete.")

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False
        self.query_one("#abort_btn", Button).disabled = True
        try:
            _save_measurement_png(self._records, self.plan.acq_cfg.output_file)
        except Exception:
            log.exception("Could not save measurement plot PNG")

    def action_abort(self) -> None:
        if self._measurement_running and not self._stop_event.is_set():
            self._stop_event.set()
            self._set_status("Abort requested — finishing current point, then ramping current to zero …")

    def action_back_or_abort(self) -> None:
        if self._measurement_running:
            self.action_abort()
        else:
            self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "abort_btn":
            self.action_abort()
        elif event.button.id == "back_btn":
            self.app.pop_screen()

    @work(thread=True, exclusive=True)
    def do_run(self) -> None:
        plan = self.plan
        source = None
        voltmeter = None
        try:
            self._set_status_threadsafe("Connecting to Keithley 6221 & 2182 …")
            source = connect_source(plan.src_cfg)
            voltmeter = connect_voltmeter(plan.volt_cfg)

            points = [CurrentPoint(current_A=float(i)) for i in plan.currents_A]

            self._set_status_threadsafe("Running measurement …")
            run_measurement(
                source, voltmeter, plan.src_cfg, plan.acq_cfg, points,
                stop_event=self._stop_event,
                on_point=lambda record: self.app.call_from_thread(self._on_point, record),
            )
            final = "Measurement aborted." if self._stop_event.is_set() else "Measurement complete."
        except Exception as exc:
            log.exception("Measurement failed")
            final = f"ERROR: {exc}"
        finally:
            if source is not None:
                try:
                    ramp_current_to_zero(source)
                except Exception:
                    log.exception("Error while ramping current to zero")
                try:
                    shutdown_source(source)
                except Exception:
                    log.exception("Error while shutting down source")
            self.app.call_from_thread(self._on_finished, final)

    def _set_status_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_status, text)


# ─────────────────────────────────────────────────────────────────────────────
# Main app  ── the parameter form
# ─────────────────────────────────────────────────────────────────────────────

class DCIVCurveApp(App):
    TITLE = "DC I-V Curve"
    SUB_TITLE = "Keithley 6221 + 2182 · current sweep"

    CSS = """
    #body { height: 1fr; }
    #form { width: 1fr; padding: 1 2; }
    #sidebar { width: 48; border-left: solid $primary; padding: 1 2; overflow-y: auto; }
    .field { margin-bottom: 1; }
    .field-label { text-style: bold; }
    .hint { text-style: italic; color: $text-muted; }
    .switch-row { height: 3; }
    .switch-row Label { margin-left: 1; content-align: left middle; height: 3; }
    .sidebar-title { text-style: bold underline; margin-bottom: 1; }
    #actionbar { height: 3; align: center middle; }
    """

    BINDINGS = [
        Binding("f5", "start", "Start measurement", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                with Collapsible(title="Instruments", collapsed=False):
                    yield field("source_visa_resource", "Keithley 6221 (current source)",
                                DEFAULTS["source_visa_resource"], kind="text")
                    yield field("voltmeter_visa_resource", "Keithley 2182 (DUT voltage)",
                                DEFAULTS["voltmeter_visa_resource"], kind="text")

                with Collapsible(title="Current sweep & compliance", collapsed=False):
                    yield field("current_min_A", "Sweep current min (A)",
                                DEFAULTS["current_min_A"])
                    yield field("current_max_A", "Sweep current max (A)",
                                DEFAULTS["current_max_A"])
                    yield field("compliance_V", "Compliance voltage (V)",
                                DEFAULTS["compliance_V"],
                                hint="Set high enough to reach the expected voltage at "
                                     "current_max_A, or the sweep clips against compliance.",
                                validators=[Number(minimum=0.0, failure_description="must be ≥ 0")])
                    with Collapsible(title="Source timing (advanced)", collapsed=True):
                        yield field("source_delay_s", "6221 source delay (s)",
                                    DEFAULTS["source_delay_s"])

                with Collapsible(title="Voltmeter (Keithley 2182)", collapsed=False):
                    yield field("nplc", "NPLC (integration time)", DEFAULTS["nplc"],
                                hint="Bigger = quieter but slower. 1 line cycle = 1/50 or 1/60 s.",
                                validators=[Number(minimum=0.01, failure_description="must be > 0")])
                    yield switch_field("auto_range", "Auto-range", DEFAULTS["auto_range"])

                with Collapsible(title="Acquisition timing", collapsed=False):
                    yield field("settling_time_s", "Settling time per current step (s)",
                                DEFAULTS["settling_time_s"],
                                validators=[Number(minimum=0.0, failure_description="must be ≥ 0")])
                    yield field("n_averages", "Voltage samples averaged per point",
                                DEFAULTS["n_averages"], kind="integer",
                                validators=[Number(minimum=1, failure_description="must be ≥ 1")])
                    yield field("output_name", "Output file name (prefix)",
                                DEFAULTS["output_name"], kind="text")

                with Collapsible(title="Sweep resolution", collapsed=False):
                    yield field("n_points", "Points per sweep direction",
                                DEFAULTS["n_points"], kind="integer",
                                hint="Bidirectional: min → max → min (reveals hysteresis).",
                                validators=[Number(minimum=2, failure_description="must be ≥ 2")])

            with Vertical(id="sidebar"):
                yield Static("Summary", classes="sidebar-title")
                yield Static(id="summary")

        with Horizontal(id="actionbar"):
            yield Button("▶  Start measurement  (F5)", id="start", variant="success")
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        # basicConfig (in dc_iv_curve) put a StreamHandler on the root logger;
        # writing to stdout while Textual owns the alt-screen would corrupt
        # the display, so drop it. RunScreen attaches its own RichLog-backed
        # handler for the duration of a measurement.
        logging.getLogger().handlers.clear()
        self._load_settings()
        self.refresh_summary()

    # ── Form state I/O ───────────────────────────────────────────────────────

    def _all_field_ids(self) -> list[str]:
        return list(NUMERIC_FIELDS) + TEXT_FIELDS

    def collect_raw(self) -> dict:
        raw: dict = {fid: self.query_one(f"#{fid}", Input).value for fid in self._all_field_ids()}
        raw["auto_range"] = self.query_one("#auto_range", Switch).value
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
        if "auto_range" in saved:
            self.query_one("#auto_range", Switch).value = bool(saved["auto_range"])

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
        state["auto_range"] = self.query_one("#auto_range", Switch).value
        return state, errors

    # ── Reactivity ───────────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        self.refresh_summary()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.refresh_summary()

    def refresh_summary(self) -> None:
        state, parse_errors = self.parse_state()
        if parse_errors:
            info, warnings, errors = [], [], parse_errors
        else:
            info, warnings, errors = build_summary(state)

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

        self._save_settings(self.collect_raw())
        plan = self._build_plan(state)
        self.push_screen(RunScreen(plan))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()

    def _build_plan(self, state: dict) -> MeasurementPlan:
        src_cfg = SourceConfig(
            visa_resource=state["source_visa_resource"],
            compliance_V=state["compliance_V"],
            source_delay_s=state["source_delay_s"],
            current_min_A=state["current_min_A"],
            current_max_A=state["current_max_A"],
        )
        volt_cfg = VoltmeterConfig(
            visa_resource=state["voltmeter_visa_resource"],
            nplc=state["nplc"],
            auto_range=state["auto_range"],
        )
        acq_cfg = AcquisitionConfig(
            settling_time_s=state["settling_time_s"],
            n_averages=state["n_averages"],
            output_file=str(_DATA_DIR / f"{state['output_name']}_{datetime.now():%Y%m%d_%H%M%S}.csv"),
        )

        currents_A = bidirectional_current_sweep(
            i_min=state["current_min_A"], i_max=state["current_max_A"], n_points=state["n_points"],
        )

        return MeasurementPlan(
            src_cfg=src_cfg, volt_cfg=volt_cfg, acq_cfg=acq_cfg, currents_A=currents_A,
        )


def main() -> None:
    DCIVCurveApp().run()


if __name__ == "__main__":
    main()
