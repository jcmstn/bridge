"""
Shared page-execution engine for bridge/web
===============================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Every one of the 7 measurement pages follows the same shape: form -> parsed
state -> config dataclasses -> a background thread that connects
instruments, calls the module's existing run_measurement()-style function
with stop_event/on_point hooks it already supports, then shuts everything
down -> live updates streamed back to the page. This module implements that
shape once; each page supplies only what's genuinely page-specific (which
instruments to connect, which Plotly traces a record feeds, what a final
report looks like) via a handful of callables passed to RunController.

There is no multiprocessing/OS-process isolation for the live plot here — a
browser renders its own Plotly figure in its own process, so the web server
only needs to push data over the websocket it already has. One thread-safe
queue.Queue carries everything (point/status/log/finished), drained once
per ui.timer tick.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from nicegui import ui

from web import run_index, run_manager

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers  ── shared across pages (each *_tui.py redefines these
# identically today; consolidated here since they're pure and TUI-independent)
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


def num_field(label: str, default: float, *, hint: str = "", integer: bool = False) -> ui.number:
    """NiceGUI analog of the TUI's field() helper for a numeric Input."""
    inp = ui.number(label, value=(int(default) if integer else float(default)), step=(1 if integer else None)) \
        .classes("w-full").props("outlined dense")
    if hint:
        ui.label(hint).classes("text-xs text-grey-6 -mt-1 mb-1")
    return inp


def text_field(label: str, default: str, *, hint: str = "") -> ui.input:
    """NiceGUI analog of the TUI's field() helper for a text Input."""
    inp = ui.input(label, value=default).classes("w-full").props("outlined dense")
    if hint:
        ui.label(hint).classes("text-xs text-grey-6 -mt-1 mb-1")
    return inp


def bool_switch(label: str, default: bool) -> ui.switch:
    """NiceGUI analog of the TUI's switch_field() helper."""
    return ui.switch(label, value=bool(default))


def optional_num_field(label: str, default: Optional[float], *, hint: str = "") -> ui.number:
    """
    A numeric field that may be left blank -> None (ui.number natively
    supports a None value), matching the TUI's OPTIONAL_NUMERIC_FIELDS
    convention (phase-cal current, sample geometry, ...) without needing
    that pattern's manual string-parse-or-None dance -- .value is already
    Optional[float].
    """
    inp = ui.number(label, value=default).classes("w-full").props("outlined dense clearable")
    if hint:
        ui.label(hint).classes("text-xs text-grey-6 -mt-1 mb-1")
    return inp


@contextmanager
def param_card(title: str) -> Iterator[ui.card]:
    """NiceGUI analog of the TUI's card() -- a bordered card grouping a few
    related fields, used inside param_grid() in place of the old
    ui.expansion() sections. For parameters the user actually changes run
    to run (current, field, gate, ...)."""
    with ui.card().classes("w-full") as c:
        ui.label(title).classes("font-bold")
        yield c


@contextmanager
def stable_card(title: str) -> Iterator[ui.card]:
    """Muted variant of param_card, for wiring addresses/timing constants
    that rarely if ever change -- used inside stable_grid()."""
    with ui.card().classes("w-full bg-grey-1 dark:bg-grey-9") as c:
        ui.label(title).classes("font-bold text-grey-6")
        yield c


def param_grid() -> ui.grid:
    """3-column grid of param_card()s -- NiceGUI analog of the TUI's
    .param-grid. Use as `with param_grid(): ...`."""
    return ui.grid(columns=3).classes("w-full gap-4")


def stable_grid() -> ui.grid:
    """3-column grid of stable_card()s -- NiceGUI analog of the TUI's
    .stable-grid. Use as `with stable_grid(): ...`."""
    return ui.grid(columns=3).classes("w-full gap-4")


def section_title(text: str) -> None:
    """Divider label between param_grid() and stable_grid(), matching the
    TUI's "Instrument configuration" .section-title Static."""
    ui.label(text).classes("text-sm font-bold text-grey-6 mt-3 mb-1")


def render_summary(info: list[str], warnings: list[str], errors: list[str]) -> None:
    """
    Renders the info/warnings/errors sidebar every page's build_summary()
    produces -- the NiceGUI port of the TUI's Rich-markup summary Static.
    Call inside a container (e.g. a @ui.refreshable function) that's
    cleared and rebuilt on every form change.
    """
    if errors:
        ui.label("Blocking issues").classes("text-bold text-negative")
        for e in errors:
            ui.label(f"✗ {e}").classes("text-negative text-sm")
    if warnings:
        ui.label("Warnings").classes("text-bold text-warning mt-2")
        for w in warnings:
            ui.label(f"⚠ {w}").classes("text-warning text-sm")
    ui.label("Derived values").classes("text-bold mt-2")
    for i in info:
        ui.label(f"• {i}").classes("text-sm text-grey-7")


def busy_banner() -> None:
    """
    Persistent banner shown on every page whenever a run -- from any suite,
    possibly started on a different tab -- is in progress. The Abort action
    here works regardless of which page/tab started the run: a real
    hardware-safety property (any tab can stop misbehaving hardware, not
    just the one that started it), which the single-process TUI never had
    to solve.
    """

    @ui.refreshable
    def _content() -> None:
        handle = run_manager.snapshot()
        if handle is None:
            return
        with ui.row().classes("w-full items-center gap-3 bg-amber-100 dark:bg-amber-900 rounded p-2 mb-2"):
            ui.icon("hourglass_top")
            ui.label(f"Busy — {handle.suite} · {handle.measurement} — "
                      f"running {format_duration(handle.elapsed_s)} — {handle.status_text}") \
                .classes("flex-grow")
            ui.button("Abort", on_click=handle.stop_event.set, color="negative").props("outline dense")

    _content()
    ui.timer(1.0, _content.refresh)


def is_busy() -> bool:
    return run_manager.snapshot() is not None


# ─────────────────────────────────────────────────────────────────────────────
# RunController
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunCallbacks:
    """Passed into a page's run_fn(stop_event, cb) closure."""
    on_point: Callable[[dict], None]
    on_status: Callable[[str], None]


@dataclass
class FinalStatus:
    status: str                              # 'completed' | 'aborted' | 'error'
    error: Optional[BaseException] = None


class _QueueLogRelay(logging.Handler):
    """Relays root-logger records into the run's queue for the duration of
    one run -- attached in RunController.try_start(), removed once the
    'finished' item is drained. Mirrors the TUI's _LogRelay, targeting a
    queue instead of a RichLog widget directly."""

    def __init__(self, q: "queue.Queue[dict]") -> None:
        super().__init__()
        self._queue = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                             datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put_nowait({"kind": "log", "text": self.format(record), "level": record.levelno})
        except Exception:
            pass


class RunController:
    """
    One instance per run -- a page creates a fresh RunController each time
    Start is pressed. Owns the run lock, run-index lifecycle, worker
    thread, and queue -> ui.timer draining; delegates everything
    plot/table/report-shaped back to the page, since that differs per
    measurement.

    `run_fn(stop_event, cb) -> Any` must itself bracket the whole
    connect -> configure -> run_measurement(...) -> finally: shutdown
    sequence, exactly like every *_tui.py's do_run() already does -- that
    bracketing is page-specific (which instruments, in which order) and
    isn't further shared here.

    `save_artifacts(records, result, status) -> list[str]` runs on the
    WORKER thread (no UI access) immediately after run_fn returns/raises --
    this is where a page saves its final headless-matplotlib PNG (mirroring
    each *_tui.py's _save_measurement_png/plot_results) and, since the data
    convention migration, unconditionally writes the raw-file header +
    index.csv row with the outcome-derived `status`
    ("completed"/"aborted"/"error" -- not yet a physical good/open/short/
    noisy judgement, see instruments/data_naming.py).
    """

    def __init__(self, *, suite: str, measurement: str,
                 run_fn: Callable[[threading.Event, RunCallbacks], Any],
                 save_artifacts: Callable[[list[dict], Any, str], list[str]],
                 parameters: dict, data_dir: str, planned_output_paths: list[str],
                 on_record: Callable[[dict], None],
                 on_status: Callable[[str], None],
                 on_log: Callable[[str, int], None],
                 on_finished: Callable[["FinalStatus", Any], None],
                 sample: Optional[str] = None, device: Optional[str] = None,
                 run_number: Optional[int] = None) -> None:
        self.suite = suite
        self.measurement = measurement
        self.run_fn = run_fn
        self.save_artifacts = save_artifacts
        self.parameters = parameters
        self.data_dir = data_dir
        self.planned_output_paths = planned_output_paths
        self.sample = sample
        self.device = device
        self.run_number = run_number
        self.on_record = on_record
        self.on_status = on_status
        self.on_log = on_log
        self.on_finished = on_finished

        self.handle: Optional[run_manager.RunHandle] = None
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._worker_records: list[dict] = []
        self._timer: Optional[ui.timer] = None
        self._log_handler: Optional[_QueueLogRelay] = None
        self._start_time = 0.0
        self.result: Any = None
        self.final: Optional[FinalStatus] = None

    # ── Public API ──────────────────────────────────────────────────────

    def try_start(self) -> bool:
        """
        Returns False (and changes nothing) if another run is already in
        progress -- the page's busy_banner() is the feedback in that case,
        never a silently-disabled button alone.
        """
        handle = run_manager.try_acquire(self.suite, self.measurement)
        if handle is None:
            return False
        self.handle = handle
        handle.run_id = run_index.start_run(
            self.suite, self.measurement, self.parameters, self.data_dir,
            self.planned_output_paths,
            sample=self.sample, device=self.device, run_number=self.run_number,
        )
        self._start_time = time.monotonic()

        self._log_handler = _QueueLogRelay(self._queue)
        logging.getLogger().addHandler(self._log_handler)

        cb = RunCallbacks(on_point=self._worker_on_point, on_status=self._worker_on_status)
        threading.Thread(target=self._worker, args=(cb,), daemon=True).start()

        self._timer = ui.timer(0.3, self._drain)
        return True

    def abort(self) -> None:
        if self.handle is not None and not self.handle.stop_event.is_set():
            self.handle.stop_event.set()
            self.on_status("Abort requested — finishing current point, then shutting down safely …")

    # ── Worker thread (no UI access) ────────────────────────────────────

    def _worker_on_point(self, record: dict) -> None:
        self._worker_records.append(record)
        self._queue.put_nowait({"kind": "point", "record": record})

    def _worker_on_status(self, text: str) -> None:
        self._queue.put_nowait({"kind": "status", "text": text})

    def _worker(self, cb: RunCallbacks) -> None:
        assert self.handle is not None
        result: Any = None
        error: Optional[BaseException] = None
        try:
            result = self.run_fn(self.handle.stop_event, cb)
        except Exception as exc:                      # noqa: BLE001 -- must not crash the thread
            log.exception("Measurement failed")
            error = exc
        finally:
            status = "aborted" if self.handle.stop_event.is_set() else \
                     ("error" if error is not None else "completed")
            try:
                output_paths = self.save_artifacts(self._worker_records, result, status)
            except Exception:
                log.exception("Could not save final artifacts")
                output_paths = self.planned_output_paths
            duration_s = time.monotonic() - self._start_time
            run_index.finish_run(
                self.handle.run_id, status=status, point_count=len(self._worker_records),
                duration_s=duration_s, error_message=(str(error) if error else None),
                output_paths=output_paths,
            )
            run_manager.release(self.handle)
            self._queue.put_nowait({"kind": "finished", "status": status, "result": result, "error": error})

    # ── Event-loop thread (ui.timer) ────────────────────────────────────

    def _drain(self) -> None:
        finished_item: Optional[dict] = None
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            kind = item["kind"]
            if kind == "point":
                if self.handle is not None:
                    self.handle.records.append(item["record"])
                self.on_record(item["record"])
            elif kind == "status":
                if self.handle is not None:
                    self.handle.status_text = item["text"]
                self.on_status(item["text"])
            elif kind == "log":
                if self.handle is not None:
                    self.handle.log_lines.append(item["text"])
                    del self.handle.log_lines[:-500]
                self.on_log(item["text"], item["level"])
            elif kind == "finished":
                finished_item = item

        if finished_item is not None:
            if self._timer is not None:
                self._timer.cancel()
            if self._log_handler is not None:
                logging.getLogger().removeHandler(self._log_handler)
            self.result = finished_item["result"]
            self.final = FinalStatus(status=finished_item["status"], error=finished_item["error"])
            self.on_finished(self.final, self.result)
