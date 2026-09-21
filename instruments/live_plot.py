"""
Live-plot subprocess launcher  ── shared by every TUI RunScreen
================================================================
A GUI matplotlib backend and Textual's terminal control both want the main
thread, so each TUI's live preview runs in its own spawned process and is fed
records over a multiprocessing.Queue.

Why this is a helper and not three inline lines: while a Textual app is
running, `sys.stderr` is a capture object whose `fileno()` returns -1. The
first `mp.Queue()` in a process launches multiprocessing's resource tracker,
which passes `sys.stderr.fileno()` to `fork_exec` -> `ValueError: bad
value(s) in fds_to_keep`, and no plot window ever opens. Launching with the
real stderr restored (a no-op outside Textual) avoids that.

Usage (in a RunScreen):

    try:
        self._plot_queue, self._plot_process = start_live_plot(
            _live_plot_worker, self.plan.magnet_cfg is not None)
    except Exception:
        log.exception("Could not start live plot window")
        self._plot_queue = self._plot_process = None

`worker` is called in the child as `worker(queue, *args)`; it must be a
module-level function (spawn pickles it by name).
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import sys


def start_live_plot(worker, *args) -> "tuple[mp.Queue, mp.Process]":
    """Spawn `worker(queue, *args)` as a daemon process; return (queue, process)."""
    ctx = mp.get_context("spawn")
    with contextlib.redirect_stderr(sys.__stderr__):
        queue = ctx.Queue()
        process = ctx.Process(target=worker, args=(queue, *args), daemon=True)
        process.start()
    return queue, process
