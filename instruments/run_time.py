"""
Run-time model — the numbers behind every program's "Run time" estimate
=========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-21

Pure (no hardware, no Textual). Each program's ``build_summary()``
builds a `RunCost` — modelled seconds per point plus a labelled breakdown —
and shows ``RunCost.lines()``; the run screen feeds the *same* per-point
list to its progress bar (`progress_total` / `progress_step`), so the
estimate and the live ETA can never disagree.

Helpers that model one instrument live next to that instrument's code, so a
constant changed in the loop moves the estimate with it:
``kepco_magnet.magnet_move_s``, ``lakeshore475.read_field_s``,
``keithley2182.read_time_s``, ``keithley6221.reversal_avg_s``,
``mfli_daq.poll_window_s`` / ``acquire_s``.

The block of knobs below are BEST-GUESS hardware latencies that cannot be
derived from the code (nothing has been measured on the rig yet). If a run
takes much longer than the sidebar says, these are the numbers to tune — a
one-line edit each.

Usage example:
    from instruments.run_time import RunCost
    rc = RunCost(n_points=41)
    rc.each("settle", 1.0)                  # 1 s per point
    rc.each("reads", 0.5)
    rc.at("ramps", 10.0, 0)                 # one-off cost before point 0
    rc.tail("ramps", 10.0)                  # teardown after the last point (not on the bar)
    info.extend(rc.lines())                 # → "Run time: ≈ 1m 42s — …"
    ProgressBar(total=progress_total(rc, 41))
"""
from __future__ import annotations

from typing import Optional

# ── Best-guess hardware latencies — tune here after a bench measurement ─────
GPIB_TXN_S          = 0.02   # one SCPI write or query over GPIB/VISA (USB adapters are slow)   [s]
POINT_OVERHEAD_S    = 0.10   # per point: full-CSV rewrite + on_point/log hand-off to the UI      [s]
TEMP_READ_S         = 0.15   # MercuryiTC read_temperature() (2 queries, LAN)                     [s]
READ_2182_FACTOR    = 1.0    # 2182 real :READ? time / (NPLC / 50 Hz). Autozero or the digital
                             #   filter (unverified after *RST) would push this to 2 … 10        [-]
# The two field-settle knobs are FITTED to rig data (ChipL HARM6 runs 68-102, 2026-09: field
# sweeps, interior hops 0.29-1.4 A, each a single ramp step; settle 0.3 s, n_avg 200, gaussmeter
# 10 x 0.05 s): measured time per point = 12.4 s + 2.0 s/A x hop, and a 0 -> 42 A multi-step ramp
# does NOT pay 2 s/A -- so the drift scales with the size of the LAST current step, not the hop.
FIELD_SETTLE_EXTRA_S = 9.6   # per magnet move beyond the minimal settle window -- lumps the field
                             #   drift, slow gaussmeter queries and per-point latency the
                             #   timestamps can't split (fit residual after every modelled term)  [s]
FIELD_SETTLE_S_PER_A = 2.0   # extra field-settle time per A of the last applied current step     [s/A]
ACQ_OVERHEAD_S      = 0.20   # MFLI subscribe / sync / unsubscribe / overload read per acquire    [s]
LOCK_TYP_S          = 3.0    # typical MFLI ExtRef PLL lock time. The loop returns at the first lock,
                             #   the 5 s form default is only a timeout. Unmeasured; deliberately high
                             #   so the 2h / 6221-only estimates do not drop below the old (5 s timeout
                             #   per point) ones                                                     [s]
ARM_S               = 0.15   # 6221 waveform ARM / INIT after configuring it                       [s]
PMU_PULSE_S         = 1.0    # 4200A pulse_once: EX + fetch round trips + route-back               [s]
PER_FILE_S          = 1.5    # per output file: allocate_run + PNG + index.csv rewrite             [s]
PER_RUN_S           = 3.0    # once per run: connects, resets, enable-output sleeps                [s]
GATE_RAMP_S         = 1.2    # 2400 shutdown_gate(): ramp_to_voltage(0) + shutdown (~30 x 20 ms twice) [s]
MDS_SYNC_S          = 5.0    # setup_mds(): LabOne multi-device sync wait (polled every 0.2 s)      [s]
PHASE_NULL_ITER_TYP = 2      # typical auto_null_phase rounds used (the form only gives the max)    [-]


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


class RunCost:
    """Modelled seconds per point (index-aligned with the records the loop
    emits, in loop order across every file) and a labelled breakdown.

    One-off costs that PRECEDE a point (start ramp, per-file re-init) are charged
    to that point with `at()`, so the cumulative cost at point *i* is what the
    progress bar shows after *i* points. Costs that FOLLOW the last point
    (teardown ramps, final PNG) go in `tail()`: they count in the estimate and
    the ETA but not in the bar, which honestly reads 100 % once the last point
    is in and only shutdown is left."""

    def __init__(self, n_points: int):
        self.points: list[float] = [0.0] * max(0, int(n_points))
        self.parts: dict[str, float] = {}
        self.tail_s = 0.0                 # end-of-run cost after the last point (see class doc)

    def each(self, label: str, seconds: float) -> None:
        """Charge `seconds` to EVERY point."""
        for i in range(len(self.points)):
            self.points[i] += seconds
        self.parts[label] = self.parts.get(label, 0.0) + seconds * len(self.points)

    def at(self, label: str, seconds: float, index: int) -> None:
        """Charge `seconds` to one point (negative index counts from the end)."""
        if not self.points:
            return
        self.points[index] += seconds
        self.parts[label] = self.parts.get(label, 0.0) + seconds

    def tail(self, label: str, seconds: float) -> None:
        """Charge `seconds` to the shutdown after the last point (not on the bar)."""
        self.tail_s += seconds
        self.parts[label] = self.parts.get(label, 0.0) + seconds

    @property
    def total_s(self) -> float:
        return sum(self.points) + self.tail_s

    def lines(self) -> list[str]:
        """Sidebar line: total + the five biggest parts."""
        parts = sorted(((v, k) for k, v in self.parts.items() if v >= 0.5), reverse=True)
        detail = " · ".join(f"{k} {format_duration(v)}" for v, k in parts[:5])
        return [f"Run time: ≈ {format_duration(self.total_s)}" + (f" — {detail}" if detail else "")]


def progress_total(rc: Optional[RunCost], fallback_points: int) -> float:
    """Progress-bar total: modelled seconds of the points, or plain point count if no cost model."""
    return sum(rc.points) if rc is not None and sum(rc.points) > 0 else float(fallback_points)


def progress_step(rc: Optional[RunCost], index: int) -> float:
    """Progress-bar advance for the point that just finished (0-based `index`)."""
    if rc is None or sum(rc.points) <= 0 or not (0 <= index < len(rc.points)):
        return 1.0
    return rc.points[index]


