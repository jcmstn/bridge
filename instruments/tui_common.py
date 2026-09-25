"""
Shared Textual scaffolding for every measurement TUI
=====================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-23

What all 13 {suite}/*_tui.py programs used to carry as verbatim copies:

- form widgets: field / switch_field / select_field / sweep_rows_field / card,
  plus identity_bar() (data root + sample + device + cooldown + setpoint);
- format_si / parse_sensor_uids, and LogRelay (root logging -> RichLog);
- MeasurementRunScreen — status line, run label + progress bar, results
  table, log, Abort/Back, live-plot window, per-run PNG, the post-run
  status/comment rewrite of the LAST run, and the run's runs.db history row;
- MeasurementApp — data-root + sample picker, settings file save/load,
  switch -> dependent-field greying, and Start.

A program subclasses both and supplies only what is its own: the form
(compose), update_summary / _build_plan, and the run screen's display
hooks (table columns + row, live-plot args, status texts). Everything else
is the program MODULE's pure API, read from the subclass's own module at
call time — SETTINGS_PATH, _DEFAULT_DATA_DIR, the *_FIELDS groups,
resolve_state, build_summary, run_plan, build_header_fields, save_run_png,
PNG_SUFFIX, RunScreen — so each module stays their single source of truth
(the web pages run the same run_plan; tests import and monkeypatch there).

Textual-only (never NiceGUI), and imported only by *_tui.py modules and
bridge_tui.py — never by a measurement script (docs/architecture.md §2).
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, ProgressBar, RichLog, Select, Static,
    Switch, TextArea,
)

from dc.dc_sweep_utils import finite
from instruments import run_index
from instruments.data_dir import DataDirPickerScreen
from instruments.data_naming import (
    TEST_SAMPLE, RunContext, ensure_sample, finish_last_run, proc_path,
)
from instruments.live_plot import start_live_plot
from instruments.run_time import progress_step, progress_total
from instruments.tui_sample_picker import (
    NEW_SAMPLE_SENTINEL, NewSampleScreen, StatusCommentScreen, sample_options,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Small pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_si(value: float, unit: str) -> str:
    """Format a value with an SI prefix, 4 significant digits, e.g. 1.2e-8 -> '12 nA'."""
    av = abs(value)
    if av == 0:
        return f"0 {unit}"
    for scale, prefix in ((1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, "")):
        if float(f"{av / scale:.4g}") < 1000:      # 1e-6 is '1 µA', not '1000 nA'
            return f"{value / scale:.4g} {prefix}{unit}"
    return f"{value:.3e} {unit}"


def parse_sensor_uids(raw: str) -> tuple:
    """Parse a "MB1.T1, DB5.T1" field into a 1- or 2-tuple of UIDs. Commas,
    semicolons and whitespace all separate — "MB1.T1 DB5.T1" is two UIDs,
    not one bogus one the iTC would reject on every read."""
    uids = [u for u in re.split(r"[,;\s]+", raw) if u]
    return tuple(uids[:2])


# ─────────────────────────────────────────────────────────────────────────────
# Form widgets
# ─────────────────────────────────────────────────────────────────────────────

def field(field_id: str, label_text: str, default: str, *, kind: str = "number",
          hint: str = "", validators=None, valid_empty: bool = False) -> list:
    """A field's widgets, flat (not wrapped in a container). Grid cells
    (see card()) that contain a further nested auto-height Vertical break
    Textual's grid auto-row sizing -- GridLayout.arrange() computes an
    'auto' row's height by calling get_content_height() on each cell, and a
    doubly-nested Vertical makes that blow up to ~100 rows instead of the
    handful the content needs. One level of Vertical (the card itself) is
    fine; a Vertical inside that is not -- so fields stay flat and spacing
    is set directly on the last widget instead of via a wrapping container."""
    label = Label(label_text, classes="field-label")
    inp = Input(value=default, id=field_id, type=kind, validators=validators,
                valid_empty=valid_empty)
    widgets = [label, inp]
    if hint:
        widgets.append(Label(hint, classes="hint"))
    widgets[-1].styles.margin = (0, 0, 1, 0)
    return widgets


def switch_field(field_id: str, label_text: str, default: bool) -> Horizontal:
    row = Horizontal(Switch(value=default, id=field_id), Label(label_text, classes="switch-label"),
                      classes="switch-row")
    row.styles.margin = (0, 0, 1, 0)
    return row


def select_field(field_id: str, label_text: str, options: list[tuple[str, int]] | list[int],
                  default: int, *, hint: str = "") -> list:
    """A labelled Select; `options` are plain values or (label, value) pairs."""
    label = Label(label_text, classes="field-label")
    opts = [(str(o), o) for o in options] if options and not isinstance(options[0], tuple) else options
    sel = Select(opts, id=field_id, value=default, allow_blank=False)
    widgets = [label, sel]
    if hint:
        widgets.append(Label(hint, classes="hint"))
    widgets[-1].styles.margin = (0, 0, 1, 0)
    return widgets


def sweep_rows_field(field_id: str, default: str) -> list:
    """One row per line, "start, stop, points" -- see parse_sweep_rows()."""
    label = Label("Sweep rows: start, stop, points (one per line)", classes="field-label")
    area = TextArea(default, id=field_id, classes="sweep-rows")
    hint = Label("Shared boundary points are merged.",
                 classes="hint")
    return [label, area, hint]


def card(title: str, *groups, muted: bool = False, id: Optional[str] = None) -> Vertical:
    """A bordered grid cell: a title plus its fields (each a flat list from
    field(), or a single widget like switch_field()'s Horizontal -- see
    field() for why fields must stay flat here). `muted` = stable/rarely
    -changed configuration, styled to recede rather than compete for attention."""
    children: list = [Static(title, classes="card-title")]
    for group in groups:
        children.extend(group) if isinstance(group, list) else children.append(group)
    return Vertical(*children, classes="stable-card" if muted else "param-card", id=id)


def identity_bar(defaults: dict, data_dir: Path, data_root: Path, *,
                 device_label: str = "Device (e.g. HB3, SV2)",
                 temperature_label: Optional[str] = "Temp. setpoint (K, optional)",
                 temperature_hint: str = "Filename's T###K token only.",
                 cell_classes: Optional[str] = "field") -> Vertical:
    """The "file & run identity" bar every form starts with: filename
    preview, Data root (+ Browse…), then Sample / Device / Cooldown / optional
    temperature setpoint. `temperature_label=None` drops the setpoint."""
    def cell(*widgets) -> Vertical:
        return Vertical(*widgets, classes=cell_classes) if cell_classes else Vertical(*widgets)

    cells = [
        cell(Label("Sample", classes="field-label"),
             Select(sample_options(data_root), id="sample_select", allow_blank=False,
                    value=TEST_SAMPLE)),
        cell(*field("device", device_label, defaults["device"], kind="text")),
        cell(*field("cooldown", "Cooldown (optional)", defaults["cooldown"], kind="text")),
    ]
    if temperature_label is not None:
        cells.append(cell(*field("temperature_setpoint_K", temperature_label,
                                 defaults["temperature_setpoint_K"], kind="number",
                                 valid_empty=True, hint=temperature_hint)))
    return Vertical(
        Static("", id="filename_preview"),
        Horizontal(Input(value=str(data_dir), id="data_dir",
                         placeholder="Absolute path to the data root"),
                   Button("Browse…", id="browse_data_dir"), id="data_dir_row"),
        Vertical(*cells, id="identity_fields"),
        id="identity_bar",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run screen
# ─────────────────────────────────────────────────────────────────────────────

class LogRelay(logging.Handler):
    """Root-logger handler that mirrors every log line into a screen's RichLog
    (thread-safe: the worker thread logs, the UI thread writes)."""

    def __init__(self, screen) -> None:
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


RUN_SCREEN_CSS = """
    #status_line { height: 1; padding: 0 1; text-style: bold; }
    #progress_row { height: auto; margin: 1 2; align: left middle; }
    #run_label { width: auto; padding: 0 2 0 0; text-style: bold; }
    #progress { margin: 0; }
    #results_table { height: 12; margin: 0 2 1 2; }
    #log { height: 1fr; margin: 0 2 1 2; border: solid $primary; }
    #runactionbar { height: 3; align: center middle; }
    """


def run_screen_bindings(abort_label: str) -> list:
    return [Binding("a", "abort", abort_label, show=True),
            Binding("q", "back_or_abort", "Abort / Back", show=True)]


class MeasurementRunScreen(Screen):
    """Executes a MeasurementPlan in a worker thread and shows live progress.

    do_run() calls the program module's pure `run_plan(plan, stop_event, *,
    on_status, on_run_label, on_point, on_run_finished, run_contexts,
    run_extras)` — the same function the web page runs — which records every
    run through data_naming.record_run() (finalized the instant it ends) and
    appends each RunContext + its header extras here. The PNG and header
    hooks default to the module's `save_run_png(plan, records, png_path,
    comment=)` and `build_header_fields(plan, ctx, records, *, status,
    comment, extra)`."""

    CSS = RUN_SCREEN_CSS
    ABORT_LABEL = "Abort (safe ramp-down)"
    BINDINGS = run_screen_bindings(ABORT_LABEL)
    ABORT_STATUS = "Abort requested — finishing current point, then ramping down safely …"
    POINT_STATUS = "Point {n} / {total} complete."
    TABLE_COLUMNS: tuple = ()
    PNG_SUFFIX = "plot"
    MEASUREMENT_TYPE = ""          # the program's locked type code (PNG file names)

    def __init__(self, plan) -> None:
        super().__init__()
        self.plan = plan
        self._stop_event = threading.Event()
        self._measurement_running = True
        self._log_handler: Optional[LogRelay] = None
        self._records: list[dict] = []
        self._plot_queue: Optional["mp.Queue"] = None
        self._plot_process: Optional[mp.Process] = None
        self._run_contexts: list[RunContext] = []       # filled by run_plan()
        self._run_extras: list[Optional[dict]] = []     # parallel to _run_contexts
        # The LAST run's PNG, stashed by _save_run_png so _on_status_comment
        # can re-save it in place once the operator's comment is known.
        self._png_path: Optional[Path] = None
        self._history_id = -1
        self._started = time.monotonic()

    @property
    def program(self):
        """The subclass's own module — its pure run_plan / header / PNG API."""
        return sys.modules[type(self).__module__]

    # ── program hooks ─────────────────────────────────────────────────

    def table_columns(self) -> tuple:
        return self.TABLE_COLUMNS

    def table_row(self, record: dict) -> tuple:
        raise NotImplementedError

    def live_plot_args(self) -> Optional[tuple]:
        """(worker, *args) for start_live_plot(), or None for no plot window."""
        return None

    def save_png(self, records: list[dict], png_path: Path, comment: str = "") -> None:
        self.program.save_run_png(self.plan, records, png_path, comment=comment)

    def build_header(self, ctx: RunContext, records: list[dict], *, status: str, comment: str,
                     extra: Optional[dict]) -> dict:
        return self.program.build_header_fields(self.plan, ctx, records, status=status,
                                                comment=comment, extra=extra)

    def progress_points(self) -> int:
        return self.plan.total_points

    def progress_index(self, record: dict) -> int:
        """0-based index of the point that just arrived, for the progress bar."""
        return len(self._records) - 1

    def initial_run_label(self) -> str:
        ctx = getattr(self.plan, "run_ctx", None)
        return f"Run #{ctx.run_str}" if ctx is not None else ""

    # ── layout / lifecycle ────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Starting …", id="status_line")
        with Horizontal(id="progress_row"):
            yield Static(self.initial_run_label(), id="run_label")
            yield ProgressBar(id="progress", total=progress_total(self.plan.run_cost, self.progress_points()),
                              show_eta=True)
        yield DataTable(id="results_table", zebra_stripes=True, cursor_type="row")
        yield RichLog(id="log", max_lines=5000, markup=False, wrap=True)
        with Horizontal(id="runactionbar"):
            yield Button(self.ABORT_LABEL, id="abort_btn", variant="error")
            yield Button("Back", id="back_btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#results_table", DataTable).add_columns(*self.table_columns())
        self._log_handler = LogRelay(self)
        logging.getLogger().addHandler(self._log_handler)
        self._start_live_plot()
        self._history_start()
        self.do_run()

    def on_unmount(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
        if self._plot_process is not None and self._plot_process.is_alive():
            self._plot_process.terminate()

    DONE_STATUS = "Measurement complete."
    ABORTED_STATUS = "Measurement aborted."
    # True for a program whose Stop is the normal way to end (an open-ended
    # log): the run history then records a stopped run as "completed".
    STOP_IS_NORMAL_END = False

    @work(thread=True, exclusive=True)
    def do_run(self) -> None:
        try:
            self.program.run_plan(
                self.plan, self._stop_event,
                on_status=self._set_status_threadsafe,
                on_run_label=self._set_run_label_threadsafe,
                on_point=lambda record: self.app.call_from_thread(self._on_point, record),
                on_run_finished=self._save_run_png,
                run_contexts=self._run_contexts, run_extras=self._run_extras)
            final = self.ABORTED_STATUS if self._stop_event.is_set() and not self.STOP_IS_NORMAL_END \
                else self.DONE_STATUS
        except Exception as exc:
            log.exception("Measurement failed")
            final = f"ERROR: {exc}"
        finally:
            self.app.call_from_thread(self._on_finished, final)

    def _start_live_plot(self) -> None:
        args = self.live_plot_args()
        if args is None:
            return
        try:
            self._plot_queue, self._plot_process = start_live_plot(*args)
        except Exception:
            log.exception("Could not start live plot window (is matplotlib installed?)")
            self._plot_queue = self._plot_process = None

    # ── thread-safe UI updates ────────────────────────────────────────

    def write_log(self, msg: str, style: str) -> None:
        self.query_one("#log", RichLog).write(Text(msg, style=style))

    def _set_status(self, text: str) -> None:
        self.query_one("#status_line", Static).update(text)

    def _set_status_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_status, text)

    def _set_run_label(self, text: str) -> None:
        self.query_one("#run_label", Static).update(text)

    def _set_run_label_threadsafe(self, text: str) -> None:
        self.app.call_from_thread(self._set_run_label, text)

    def _make_on_point(self, series_index: int, series_label: Optional[str]):
        """run_measurement's on_point for one run of a series: tags each
        record with its series index/label, then hands it to the UI thread."""
        def _cb(record: dict) -> None:
            record["series_index"] = series_index
            record["series_label"] = series_label
            self.app.call_from_thread(self._on_point, record)
        return _cb

    def _on_point(self, record: dict) -> None:
        self._records.append(record)
        if self._plot_queue is not None:
            try:
                self._plot_queue.put_nowait(record)
            except Exception:
                pass
        table = self.query_one("#results_table", DataTable)
        table.add_row(*self.table_row(record))
        table.move_cursor(row=table.row_count - 1, scroll=True)
        idx = self.progress_index(record)
        self.query_one("#progress", ProgressBar).advance(progress_step(self.plan.run_cost, idx))
        self._set_status(self.POINT_STATUS.format(n=idx + 1, total=self.progress_points()))

    # ── per-run artefacts ─────────────────────────────────────────────

    def _save_run_png(self, ctx: RunContext, iter_records: list[dict]) -> None:
        """One PNG per run (own run number) — as if each run of a series had
        been started by hand; no combined overlay."""
        try:
            png_path = proc_path(self.plan.data_root, ctx.sample, ctx.run_str, ctx.device,
                                 self.MEASUREMENT_TYPE, self.PNG_SUFFIX)
            self._png_path = png_path
            self.save_png(iter_records, png_path)
        except Exception:
            log.exception("Could not save measurement plot PNG")

    def _on_finished(self, final_status: str) -> None:
        self._measurement_running = False
        self._set_status(final_status)
        self.query_one("#back_btn", Button).disabled = False
        self.query_one("#abort_btn", Button).disabled = True
        outcome = "aborted" if self._stop_event.is_set() and not self.STOP_IS_NORMAL_END \
            else ("error" if final_status.startswith("ERROR") else "completed")
        self._history_finish(outcome, final_status)
        if self._run_contexts:
            self.app.push_screen(StatusCommentScreen(), self._on_status_comment)

    def _on_status_comment(self, result: Optional[tuple[str, str]]) -> None:
        # With several runs in a series the ones before the last were
        # implicitly "skipped" -- left at the outcome status written right
        # after each one, with no comment. Only the last run, the one the
        # operator is looking at, gets the status/comment they entered.
        if result is None:
            return
        status, comment = result
        try:
            done = finish_last_run(self.plan.data_root, self._run_contexts, self._run_extras,
                                   self._records, status, comment, self.build_header)
        except Exception:
            log.exception("Could not save the final status/comment")
            done = None
        if done is not None and comment and self._png_path is not None:
            try:
                self.save_png(done[1], self._png_path, comment=comment)
            except Exception:
                log.exception("Could not re-save measurement plot PNG with comment")

    # ── run history (runs.db, shared with the web front end) ──────────

    def _history_start(self) -> None:
        ctx = getattr(self.plan, "run_ctx", None)
        try:
            self._history_id = run_index.start_run(
                type(self).__module__.split(".")[0].upper(), self.app.TITLE or type(self).__module__,
                dict(getattr(self.plan, "header_extra", None) or {}),
                str(self.plan.data_root), [],
                sample=ctx.sample if ctx is not None else getattr(self.plan, "sample", None),
                device=ctx.device if ctx is not None else getattr(self.plan, "device", None),
                run_number=ctx.run_number if ctx is not None else None)
        except Exception:
            log.exception("Could not record the run in the run history")

    def _history_finish(self, outcome: str, final_status: str,
                        point_count: Optional[int] = None) -> None:
        try:
            run_index.finish_run(
                self._history_id, status=outcome,
                point_count=len(self._records) if point_count is None else point_count,
                duration_s=time.monotonic() - self._started,
                error_message=final_status if outcome == "error" else None,
                output_paths=[str(c.raw_path) for c in self._run_contexts])
        except Exception:
            log.exception("Could not finish the run-history entry")

    # ── actions ───────────────────────────────────────────────────────

    def action_abort(self) -> None:
        if self._measurement_running and not self._stop_event.is_set():
            self._stop_event.set()
            self._set_status(self.ABORT_STATUS)

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


# ─────────────────────────────────────────────────────────────────────────────
# Form app
# ─────────────────────────────────────────────────────────────────────────────

class MeasurementApp(App):
    """The parameter form. Subclass supplies TITLE/SUB_TITLE/CSS, compose(),
    parse_state(), update_summary(), _build_plan() and SWITCH_DEPENDENTS;
    the program module supplies SETTINGS_PATH, _DEFAULT_DATA_DIR, DEFAULTS,
    NUMERIC_FIELDS / TEXT_FIELDS / OPTIONAL_NUMERIC_FIELDS (+ LIST_FIELDS),
    build_summary() and RunScreen."""

    BINDINGS = [
        Binding("f5", "start", "Start measurement", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    # switch id -> widget ids disabled while that switch is off
    SWITCH_DEPENDENTS: dict[str, tuple[str, ...]] = {
        "enable_temperature": ("temperature_visa_resource", "temperature_sensor_uids"),
    }
    # plane-shortcut button id -> (field id, value) (field-direction cards)
    PLANE_BUTTONS = {"plane_xy": ("field_theta_deg", "90"), "plane_zx": ("field_phi_deg", "0"),
                     "plane_zy": ("field_phi_deg", "90")}

    @property
    def program(self):
        """The subclass's own module — where its module-level names live."""
        return sys.modules[type(self).__module__]

    # ── lifecycle ─────────────────────────────────────────────────────

    def on_mount(self) -> None:
        # The measurement modules' logging.basicConfig() put a StreamHandler on
        # the root logger; writing to stdout while Textual owns the alt-screen
        # would corrupt the display, so drop it. RunScreen attaches its own
        # RichLog-backed handler for the duration of a measurement.
        logging.getLogger().handlers.clear()
        self._load_settings()
        for switch_id in self.SWITCH_DEPENDENTS:
            for switch in self.query(f"#{switch_id}").results(Switch):
                self._apply_switch_dependents(switch_id, switch.value)
        self.refresh_summary()

    def update_summary(self) -> None:
        """Re-parse the form and repaint the sidebar summary + filename preview."""
        raise NotImplementedError

    def refresh_summary(self) -> None:
        """update_summary(), but an unexpected error in it (a run-time model
        dividing by a just-typed 0, …) is shown in the sidebar and blocks
        Start instead of closing the whole TUI — an exception escaping a
        Textual event handler exits the app."""
        try:
            self.update_summary()
        except Exception as exc:
            log.exception("Could not evaluate the form")
            for summary in self.query("#summary").results(Static):
                summary.update(f"[bold red]Can't evaluate this form[/bold red]\n  [red]✗ {exc}[/red]")
            for start in self.query("#start").results(Button):
                start.disabled = True

    def parse_state(self) -> tuple[dict, list[str]]:
        """The form as a typed state dict: every field Input (_parse_fields),
        every Switch / Select / TextArea by widget id, and the sample — then the
        program module's pure `resolve_state(state)` (the derived lists/sweeps
        with their parse errors), the same function its web page calls."""
        state, errors = self._parse_fields()
        for switch in self.query(Switch):
            if switch.id:
                state[switch.id] = switch.value
        for select in self.query(Select):
            if select.id and select.id != "sample_select":
                state[select.id] = select.value
        for area in self.query(TextArea):
            if area.id:
                state[area.id] = area.text
        sample_value = self.query_one("#sample_select", Select).value
        state["sample"] = sample_value if sample_value not in (None, Select.BLANK) else ""
        resolve = getattr(self.program, "resolve_state", None)
        return (resolve(state) if resolve is not None else state), errors

    def _parse_fields(self) -> tuple[dict, list[str]]:
        """The shared head of every parse_state(): the NUMERIC_FIELDS (cast
        with their type), TEXT_FIELDS, LIST_FIELDS (comma lists) and
        OPTIONAL_NUMERIC_FIELDS Inputs. Only finite numbers pass — "1e999"
        parses to inf, which the run-time model / field diagram can't take."""
        p = self.program
        errors: list[str] = []
        state: dict = {}
        for fid, caster in p.NUMERIC_FIELDS.items():
            raw = self.query_one(f"#{fid}", Input).value.strip()
            try:
                state[fid] = finite(caster(raw))
            except ValueError:
                errors.append(f"'{fid}' is not a valid number: {raw!r}")
                state[fid] = 0
        for fid in p.TEXT_FIELDS:
            state[fid] = self.query_one(f"#{fid}", Input).value.strip()
        for fid in getattr(p, "LIST_FIELDS", []):
            raw = self.query_one(f"#{fid}", Input).value.strip()
            values: list[float] = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    values.append(finite(float(part)))
                except ValueError:
                    errors.append(f"'{fid}' contains a value that isn't a number: {part!r}")
            state[fid] = values
        for fid in p.OPTIONAL_NUMERIC_FIELDS:
            raw = self.query_one(f"#{fid}", Input).value.strip()
            if raw:
                try:
                    state[fid] = finite(float(raw))
                except ValueError:
                    errors.append(f"'{fid}' is not a valid number: {raw!r}")
                    state[fid] = None
            else:
                state[fid] = None
        return state, errors

    def _build_plan(self, state: dict):
        raise NotImplementedError

    # ── data root + sample picker ─────────────────────────────────────

    def _refresh_sample_options(self, *, select_value: Optional[str] = None) -> None:
        select = self.query_one("#sample_select", Select)
        select.set_options(sample_options(self.data_root))
        if select_value is not None:
            select.value = select_value

    def _sync_data_root(self) -> None:
        """Point self.data_root at the identity bar's "Data root" field when
        it names an existing directory, and re-list samples from there.
        Gated on is_dir() so a half-typed path doesn't scatter _test/
        folders across the disk (sample_options() creates them)."""
        path = Path(self.query_one("#data_dir", Input).value.strip()).expanduser()
        if not path.is_dir():
            return
        self.data_root = path.resolve()
        opts = [v for _, v in sample_options(self.data_root)]
        cur = self.query_one("#sample_select", Select).value
        self._refresh_sample_options(select_value=cur if cur in opts else TEST_SAMPLE)

    def _browse_data_dir(self) -> None:
        start = self.query_one("#data_dir", Input).value.strip() or str(self.program._DEFAULT_DATA_DIR)
        self.push_screen(DataDirPickerScreen(start), self._on_data_dir_picked)

    def _on_data_dir_picked(self, picked: Optional[str]) -> None:
        if not picked:
            return
        self.query_one("#data_dir", Input).value = picked
        self._sync_data_root()

    def _on_new_sample_created(self, result: Optional[str]) -> None:
        # Cancelled -> fall back to the quick-test sample, not the sentinel.
        self._refresh_sample_options(select_value=result or TEST_SAMPLE)
        self.refresh_summary()

    # ── settings file ─────────────────────────────────────────────────

    def _all_field_ids(self) -> list[str]:
        p = self.program
        return (list(p.NUMERIC_FIELDS) + list(p.TEXT_FIELDS) + list(getattr(p, "LIST_FIELDS", []))
                + list(p.OPTIONAL_NUMERIC_FIELDS))

    def collect_raw(self) -> dict:
        """The form's raw state: every field Input, then every Switch /
        Select / TextArea by widget id (the settings-file keys)."""
        raw: dict = {fid: self.query_one(f"#{fid}", Input).value for fid in self._all_field_ids()}
        for area in self.query(TextArea):
            if area.id:
                raw[area.id] = area.text
        for switch in self.query(Switch):
            if switch.id:
                raw[switch.id] = switch.value
        for select in self.query(Select):
            if select.id and select.id != "sample_select":
                raw[select.id] = select.value
        sample_value = self.query_one("#sample_select", Select).value
        if sample_value not in (None, Select.BLANK, NEW_SAMPLE_SENTINEL):
            raw["sample"] = sample_value
        return raw

    def _read_settings(self) -> dict:
        """The saved form ({} if none). Override to merge in older files."""
        try:
            return json.loads(self.program.SETTINGS_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _load_settings(self) -> None:
        saved = self._read_settings()
        if not saved:
            return
        for fid in self._all_field_ids():
            if fid in saved:
                try:
                    self.query_one(f"#{fid}", Input).value = str(saved[fid])
                except Exception:
                    pass
        for area in self.query(TextArea):
            if area.id in saved:
                area.text = str(saved[area.id])
        for switch in self.query(Switch):
            if switch.id in saved:
                switch.value = bool(saved[switch.id])
        for select in self.query(Select):
            if select.id in saved and select.id != "sample_select":
                try:
                    select.value = saved[select.id]
                except Exception:
                    pass
        self._sync_data_root()
        saved_sample = saved.get("sample")
        if saved_sample and saved_sample in [v for _, v in sample_options(self.data_root)]:
            self.query_one("#sample_select", Select).value = saved_sample

    def _save_settings(self, raw: dict) -> None:
        path = self.program.SETTINGS_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw, indent=2))
        except OSError:
            pass

    # ── events ────────────────────────────────────────────────────────

    def _apply_switch_dependents(self, switch_id: str, enabled: bool) -> None:
        for widget_id in self.SWITCH_DEPENDENTS.get(switch_id, ()):
            for widget in self.query(f"#{widget_id}"):
                widget.disabled = not enabled

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "data_dir":
            self._sync_data_root()
        self.refresh_summary()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self.refresh_summary()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self._apply_switch_dependents(event.switch.id, event.value)
        self.refresh_summary()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sample_select" and event.value == NEW_SAMPLE_SENTINEL:
            self.push_screen(NewSampleScreen(self.data_root), self._on_new_sample_created)
            return
        self.refresh_summary()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()
        elif event.button.id == "browse_data_dir":
            self._browse_data_dir()
        elif event.button.id in self.PLANE_BUTTONS:
            field_id, value = self.PLANE_BUTTONS[event.button.id]
            self.query_one(f"#{field_id}", Input).value = value
            self.refresh_summary()

    def action_start(self) -> None:
        try:
            state, parse_errors = self.parse_state()
            errors = parse_errors or self.program.build_summary(state)[2]
        except Exception:
            log.exception("Could not evaluate the form")
            errors = ["unevaluable form"]
        if errors:
            self.bell()
            return
        self.data_root = Path(state["data_dir"]).expanduser()
        ensure_sample(self.data_root, state["sample"], create=True)
        self._save_settings(self.collect_raw())
        self.push_screen(self.run_screen(self._build_plan(state)))

    def run_screen(self, plan):
        """The RunScreen for `plan` — the program module's, by default."""
        return self.program.RunScreen(plan)
