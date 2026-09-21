#!/usr/bin/env python3
"""
Nonlocal spin-current switching — 6221 write pulse + 2182A nonlocal read
(only the 6221 and the 2182A measure: no MFLI, no 4200A)
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-21

Does a pure spin current switch the detector magnet of a nonlocal spin
valve?  Physics is spin-transfer torque from a pure spin current (the charge
current never flows under the magnet), not SOT. Modelled on the Kimura/Otani
nonlocal switching experiments (single ~1 ms injector pulse, then read the
nonlocal resistance with a small sense current) with a DC reversal read in
place of their AC lock-in sense current.

Method
------
  0. INITIALIZE — an external magnetic field sets the magnet's starting
     state before the sweep (`initialize_with_field`: Kepco + Lake Shore 475;
     or do it by hand and skip it). One initial state per run.
  1. WRITE  — one hardware-timed 6221 WAVE-square pulse (`fire_wave_pulse`,
     one cycle) through the injector: 0 -> +I -> 0 or 0 -> -I -> 0 — ONE lobe
     of the chosen polarity, never an opposite-polarity lobe after it. Positive:
     offset = amplitude = I/2. Negative: offset = -|I|/2, so the wave swings
     between -|I| and 0 (it never goes positive) with the 0 A half first, i.e.
     the lobe arrives one `width_s` after the start. The output turns off after
     the cycle. The pulse sign is the sign of the injected spin accumulation:
     it pushes the magnet toward P or toward AP.
  2. WAIT   — `delay_after_pulse_s` (Joule heat and Peltier/thermal EMFs
     decay, the 2182A recovers).
  3. READ   — the SAME injector pair, small DC +/-I_sense, current-reversal
     averaged (`acquire_reversal_averaged_voltage`) on the 2182A across
     detector magnet / reference electrode past the magnet. V_odd/I_sense =
     R_NL, the nonlocal resistance: two levels (P / AP) for a two-state
     magnet. V_even (thermal EMF, Joule heating, rectification) is recorded
     next to it as the artifact proxy.
  A read-only baseline row (no pulse) is prepended: it shows the state the
  initialization left. Each row carries R_NL, delta_R vs the previous row and
  `switched` (|delta_R| above `switch_sigma` x combined SEM — and, if
  `R_P_ohm`/`R_AP_ohm` are given, at least half the P<->AP swing).
  `switch_currents_A(records)` lists the signed pulse currents that switched.

Wiring
------
    6221 HI      ──▶ injector electrode
    6221 LO      ──▶ return electrode, a bit further along the channel from the
                     injector on the side away from the detector: the pulse and
                     the sense current return through it, so no charge current
                     flows under the magnet.  Triax OUTPUT LOW must be FLOATING
                     (`output_low_grounded = False`, `:OUTP:LTE`; the TUI and
                     main() set it, run_measurement logs the read-back and warns
                     if grounded) — otherwise the return can leak through earth,
                     and the 2182A pair (which shares no contact with the 6221
                     pair) sees a common-mode set by that instead.
    2182A ch1    ──▶ detector magnet electrode / reference electrode past the magnet
    (2182A ch2 is not used: its LO is bonded to ch1 LO, wrong for a nonlocal
    geometry where the detector pair shares no contact with the injector pair.)

Protocols — `points` is a list of signed pulse currents (one lobe each)
------------------------------------------------------------------------
    linear_sweep(I0, I1, step, bidirectional=False)   the Kimura-style sweep:
        initialize, then ascending amplitudes; R_NL steps at the switching current
    ... run once per initial state (e.g. field +B and -B): with ONE pulse
        polarity only one initial state can switch — the other is the control
    bidirectional=True on a one-polarity sweep         back down again: no reset
        pulse exists, so R_NL must stay put (nonvolatile, not thermal)
    linear_sweep(-I, +I, step, bidirectional=True)     self-resetting hysteresis
        loop of R_NL vs pulse current (Kimura: different +/- switching currents)
    [+I, -I] * N                                       toggle test: R_NL must
        alternate between two levels following the pulse SIGN
    [+I, 0, 0, 0]                                      a 0 A point is a read-only
        point (no `delay_after_pulse_s`, back-to-back reads) -> relaxation after
        a pulse; the time axis is `elapsed_s` (end of each read, s since start)
  A pulse is never a +/- pair: every point is a single 0 -> +/-I -> 0 lobe.

How switching is verified in the literature
-------------------------------------------
  * Pure-spin-current switching of a nanomagnet (Kimura/Otani group, Nat.
    Phys. 4, 851 (2008); Sci. Rep. 2019, doi:10.1038/s41598-019-56082-x;
    graphene LSV: Nano Lett. 13, 5177 (2013)): a single ~1 ms current pulse
    into the injector, THEN the nonlocal resistance is read with a small
    sense current (+/-100 uA), R_NL vs pulse current. The P and AP levels are
    calibrated by a field sweep of the same nonlocal signal (`dc_spin_valve.py`);
    switching is accepted when the post-pulse levels match the field-sweep
    levels and the direction follows pulse polarity. Kimura also reads a DC
    nanovoltmeter voltage right after a ~200 us injector pulse (arXiv:1103.0852).
  * STT-MTJ: reset -> pulse -> read at low bias (<< switching bias), ~200
    repeats per amplitude for a switching probability.
  * SOT (Miron, Nature 476, 189 (2011); Liu, Science 336, 555 (2012)): pulse,
    then a low-current AHE resistance read; R_xy vs pulse amplitude.
  `R_P_ohm` / `R_AP_ohm` (from the field-swept reference loop) give
  `state_AP_fraction` per row: 0 = P level, 1 = AP level, in between =
  partial switching / multi-domain.

Artifact checklist — what the columns can and cannot rule out
-------------------------------------------------------------
  * Joule heating: even in current. Watch `voltage_even_V`; a real switch is
    a step that stays, not a bump that decays with `elapsed_s`.
  * Oersted field of the injector current is odd in current
    (mu0*I/2*pi*r ~ mT..tens of mT at mA / sub-um), so a polarity-dependent
    result alone does not isolate spin torque from it. Assist-field dependence,
    injector-to-magnet distance and a control device without the spin path do.
  * Read disturb: the sense current is itself a (weak) spin current. It must
    be << the smallest pulse; `_check_read_safety` refuses >= it, warns > 20%.
  * Compliance: if I_pulse x R_injector exceeds `compliance_V` the 6221 clips
    silently. Choose compliance_V above I_max x R_inj with margin.
  * Energy: pulse energy is I^2 R t. Start well below the expected switching
    current; ms pulses dump orders of magnitude more heat than ns PMU pulses.

Why not Delta / Pulse Delta / Differential Conductance (6221 manual, Ch. 5)
--------------------------------------------------------------------------
They make the 6221 the master of the 2182A over RS-232 (null-modem) + Trigger
Link, and Pulse Delta measures V DURING a 50 us-12 ms pulse (a heating-
avoidance measurement technique), not the magnet state after it. Both
instruments here stay independent on GPIB. Pulse Delta could later serve as a
transient diagnostic, not as the switching readout.

Bench-verify before trusting a run
----------------------------------
  * Plain :SOUR:CURR sourcing works right after WAVE abort (the manual's
    "+413 Not allowed with mode arm" is the failure to watch for).
  * Scope: every pulse is a single lobe 0 -> +/-I -> 0 with no opposite-polarity
    undershoot, and a negative pulse arrives one `width_s` after the start (0 A
    half first); requested vs delivered height for the first amplitudes. `pulse_width_measured_s` is only
    an order-of-magnitude check (it includes GPIB polling and the full
    2*width wave period).
  * Requires: pymeasure, pyvisa, numpy, pandas, matplotlib.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from pymeasure.instruments.keithley import Keithley2182, Keithley6221

from instruments.keithley2182 import VoltmeterConfig, connect_voltmeter
from instruments.keithley6221 import (
    PulseWaveConfig,
    acquire_reversal_averaged_voltage,
    connect,
    fire_wave_pulse,
    shutdown_source,
)
from instruments.kepco_magnet import (
    MagnetConfig,
    connect_magnet,
    set_magnet_current,
    shutdown_magnet,
)
from instruments.lakeshore475 import (
    GaussmeterConfig,
    LakeShore475,
    connect_gaussmeter,
    read_field_mT,
    shutdown_gaussmeter,
)
from instruments.mercury_itc import (
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import linear_sweep, safe_shutdown

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_WRITE_CURRENT_HARD_MAX_A = 0.105     # 6221 hardware range (pymeasure would silently clip)
_READ_CURRENT_CEILING_A = 10e-3       # fat-finger guard, same as the SOT programs
_READ_COMPLIANCE_CEILING_V = 21.0
_NO_PULSE_A = 1e-9                    # |I| below this is a read-only point (also absorbs linspace's ~1e-19 "zero")


@dataclass
class WritePulseConfig:
    """Timing/compliance of the write pulse (`fire_wave_pulse`); the signed
    amplitude is per point (`PulsePoint.pulse_current_A`)."""
    width_s: float      = 1e-3    # requested pulse width [s] — Kimura/Otani use ~1 ms
    compliance_V: float = 5.0     # pulse voltage compliance [V]


@dataclass
class ReadConfig:
    """The DC current-reversal nonlocal read, plus the wait before it."""
    sense_current_A: float     = 1e-4    # read current magnitude [A] — must stay << the smallest pulse
    compliance_V: float        = 2.0
    n_reversals: int           = 5       # +I/-I pairs averaged per read (>= 2: SEM needs two)
    source_delay_s: float      = 0.1     # settle after each polarity flip, before the 2182A read [s]
    delay_after_pulse_s: float = 1.0     # wait between pulse end and read [s]
    switch_sigma: float        = 5.0     # `switched` = |delta_R| > switch_sigma x combined SEM
    R_P_ohm: Optional[float]   = None    # nonlocal-R levels from a field-swept reference loop
    R_AP_ohm: Optional[float]  = None    # (dc_spin_valve.py); both or neither


@dataclass
class PulsePoint:
    pulse_current_A: float    # signed: one lobe 0 -> +/-I -> 0; 0 = read only, no pulse


def _check_write_safety(pulse_cfg: WritePulseConfig) -> None:
    if pulse_cfg.width_s <= 0:
        raise ValueError(f"WritePulseConfig.width_s must be > 0 s; got {pulse_cfg.width_s}.")
    if not 0.1 <= pulse_cfg.compliance_V <= 105.0:
        raise ValueError(
            f"WritePulseConfig.compliance_V must be in [0.1, 105] V (6221 hardware range); "
            f"got {pulse_cfg.compliance_V} V.")


def _check_pulse_currents(points: List[PulsePoint]) -> None:
    """Refuse a pulse beyond the 6221's range (or NaN) — pymeasure clips
    silently, which would hide a mistyped exponent. Either sign is fine (one
    lobe 0 -> +/-I -> 0); zero is allowed (read only)."""
    bad = [pt.pulse_current_A for pt in points
           if not abs(pt.pulse_current_A) <= _WRITE_CURRENT_HARD_MAX_A]
    if bad:
        raise ValueError(
            f"Pulse current(s) {bad} A exceed the 6221's hardware range "
            f"±{_WRITE_CURRENT_HARD_MAX_A} A (or are NaN).")


def _check_read_safety(read_cfg: ReadConfig, points: List[PulsePoint]) -> None:
    if not 0 < read_cfg.sense_current_A <= _READ_CURRENT_CEILING_A:
        raise ValueError(
            f"sense_current_A must be in (0, {_READ_CURRENT_CEILING_A} A]; got "
            f"{read_cfg.sense_current_A} A — check for a mistyped exponent.")
    if not 0 < read_cfg.compliance_V <= _READ_COMPLIANCE_CEILING_V:
        raise ValueError(
            f"compliance_V must be in (0, {_READ_COMPLIANCE_CEILING_V} V]; got "
            f"{read_cfg.compliance_V} V.")
    if read_cfg.n_reversals < 2:
        raise ValueError("n_reversals must be >= 2 (the SEM behind `switched` needs two pairs).")
    if read_cfg.delay_after_pulse_s < 0:
        raise ValueError("delay_after_pulse_s must be >= 0.")
    r_p, r_ap = read_cfg.R_P_ohm, read_cfg.R_AP_ohm
    if (r_p is None) != (r_ap is None) or (r_p is not None and r_p == r_ap):
        raise ValueError("Give both R_P_ohm and R_AP_ohm (different), or neither.")
    pulses = [abs(pt.pulse_current_A) for pt in points if abs(pt.pulse_current_A) > _NO_PULSE_A]
    if pulses and read_cfg.sense_current_A >= min(pulses):
        raise ValueError(
            f"sense_current_A ({read_cfg.sense_current_A} A) must be below the smallest "
            f"pulse ({min(pulses)} A) — the read itself would switch the magnet.")
    if pulses and read_cfg.sense_current_A > 0.2 * min(pulses):
        log.warning("Sense current is %.0f%% of the smallest pulse — read disturb possible.",
                    100 * read_cfg.sense_current_A / min(pulses))


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (unit-tested)
# ─────────────────────────────────────────────────────────────────────────────

def _switched(r: float, sem: float, r_prev: Optional[float], sem_prev: Optional[float],
              sigma: float, min_delta: float = 0.0) -> Optional[bool]:
    """Did R_NL change by more than `sigma` combined standard errors (and at
    least `min_delta`)? None when there is no previous row or no usable SEM."""
    if r_prev is None:
        return None
    err = math.hypot(sem, sem_prev)
    if not err > 0:      # NaN or zero
        return None
    return abs(r - r_prev) > max(sigma * err, min_delta)


def _ap_fraction(r: float, r_p: Optional[float], r_ap: Optional[float]) -> Optional[float]:
    """0 at the P level, 1 at the AP level of the field-swept reference loop."""
    return None if r_p is None else (r - r_p) / (r_ap - r_p)


def switch_currents_A(records: List[dict]) -> List[float]:
    """Signed pulse currents of the rows flagged `switched`, in run order — the
    switching current of an ascending sweep, or +Ic and -Ic of a loop. Empty if
    nothing switched."""
    return [r["pulse_current_A"] for r in records
            if r.get("switched") is True and r["pulse_current_A"] != 0]


# ─────────────────────────────────────────────────────────────────────────────
# Hardware helpers — one 6221 (WAVE for the pulse, plain DC for the read), field init
# ─────────────────────────────────────────────────────────────────────────────

def _interruptible_sleep(seconds: float, stop_event: Optional[threading.Event]) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if stop_event is not None and stop_event.is_set():
            return
        time.sleep(min(0.2, end - time.monotonic()))


def _output_off(source: Keithley6221) -> None:
    """Disarm any wave, zero the DC level, output off. Harmless when idle."""
    source.waveform_abort()
    source.source_current = 0.0
    source.disable_source()


def _dc_read_on(source: Keithley6221, compliance_V: float) -> None:
    """Back to plain DC after a WAVE pulse — re-assert what the pulse changed
    (compliance) and what WAVE ranging may have left (auto range)."""
    source.source_auto_range = True
    source.source_compliance = compliance_V
    source.source_current = 0.0
    source.enable_source()


def initialize_with_field(magnet, magnet_cfg: MagnetConfig, gaussmeter, gauss_cfg: GaussmeterConfig,
                          init_A: float, hold_A: float = 0.0,
                          tolerance_mT: Optional[float] = None,
                          stop_event: Optional[threading.Event] = None) -> dict:
    """Set the magnet's starting state with an external field: ramp the Kepco
    to `init_A` (waits for current AND field to settle), note the field, then
    ramp to `hold_A` (0 = field off, the sweep starts from the remanent state)
    and settle again. Returns {"init_field_measured_mT": ...} for the header."""
    set_magnet_current(magnet, magnet_cfg, init_A, gaussmeter, gauss_cfg, tolerance_mT, stop_event)
    init_mT = read_field_mT(gaussmeter, gauss_cfg)
    set_magnet_current(magnet, magnet_cfg, hold_A, gaussmeter, gauss_cfg, tolerance_mT, stop_event)
    log.info("Initialized with %.4g A (%.4f mT); holding %.4g A for the sweep.", init_A, init_mT, hold_A)
    return {"init_field_measured_mT": init_mT}


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop  (docs/architecture.md §3 contract)
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    source: Keithley6221,
    voltmeter: Keithley2182,
    pulse_cfg: WritePulseConfig,
    read_cfg: ReadConfig,
    points: List[PulsePoint],
    stop_event: Optional[threading.Event] = None,
    on_point: Optional[Callable[[dict], None]] = None,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg: Optional[GaussmeterConfig] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg: Optional[TemperatureControllerConfig] = None,
    magnet_current_A: Optional[float] = None,
    write_csv: Optional[Callable[[List[dict]], None]] = None,
    output_file: str = "nonlocal_switching.csv",
) -> pd.DataFrame:
    """Baseline read, then per point: (output off) -> WAVE pulse -> wait ->
    DC reversal read -> (output off). One row per point; CSV rewritten in
    full every row. The 6221 output is always left off. The caller has already
    initialized the magnet state (`initialize_with_field` or by hand).

    `gaussmeter`/`magnet_current_A`, if given, only RECORD the static field
    held during the sweep (read once); `temp_ctrl=None` is never an error.
    """
    _check_write_safety(pulse_cfg)
    _check_pulse_currents(points)
    _check_read_safety(read_cfg, points)

    field_mT = None
    if gaussmeter is not None and gauss_cfg is not None:
        field_mT = read_field_mT(gaussmeter, gauss_cfg)
        log.info("Static field: %.4f mT measured (magnet current %s A)", field_mT, magnet_current_A)

    if source.output_low_grounded:
        log.warning("6221 OUTPUT LOW is GROUNDED — the nonlocal 2182A pair expects it floating.")
    else:
        log.info("6221 OUTPUT LOW floating.")

    r_p, r_ap = read_cfg.R_P_ohm, read_cfg.R_AP_ohm
    min_delta = 0.5 * abs(r_ap - r_p) if r_p is not None else 0.0

    records: List[dict] = []
    plan = [PulsePoint(0.0), *points]        # index 0 = baseline read, no pulse
    r_prev = sem_prev = None
    t_start = time.monotonic()

    try:
        for idx, pt in enumerate(plan):
            if stop_event is not None and stop_event.is_set():
                log.info("Aborted after %d / %d points.", len(records), len(plan))
                break

            fired = abs(pt.pulse_current_A) > _NO_PULSE_A
            pinfo = None
            _output_off(source)
            if fired:
                pinfo = fire_wave_pulse(
                    source,
                    PulseWaveConfig(pulse_current_A=pt.pulse_current_A,
                                    width_s=pulse_cfg.width_s,
                                    compliance_V=pulse_cfg.compliance_V),
                    stop_event)
                _interruptible_sleep(read_cfg.delay_after_pulse_s, stop_event)
                if stop_event is not None and stop_event.is_set():
                    log.info("Aborted after the pulse of point %d, before its read.", idx)
                    break

            _dc_read_on(source, read_cfg.compliance_V)
            rv = acquire_reversal_averaged_voltage(
                source, voltmeter, read_cfg.sense_current_A, read_cfg.n_reversals,
                stop_event, source_delay_s=read_cfg.source_delay_s)
            _output_off(source)

            i_read = read_cfg.sense_current_A
            r = rv["mean"] / i_read
            r_sem = rv["sem"] / i_read
            t1_K, t2_K = read_temperature(temp_ctrl, temp_cfg) if temp_cfg is not None else (None, None)

            record = {
                "point_index":            idx,
                "timestamp":              time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_s":              time.monotonic() - t_start,
                "pulse_current_A":        pt.pulse_current_A if fired else 0.0,
                "pulse_width_s":          pulse_cfg.width_s if fired else None,
                "pulse_width_measured_s": pinfo["pulse_width_measured_s"] if fired else None,
                "sense_current_A":        i_read,
                "n_reversals":            rv["n_reversals"],
                "voltage_V":              rv["mean"],
                "voltage_sem_V":          rv["sem"],
                "voltage_even_V":         rv["even_mean"],
                "voltage_even_sem_V":     rv["even_sem"],
                "nl_resistance_ohm":      r,
                "nl_resistance_sem_ohm":  r_sem,
                "delta_R_ohm":            None if r_prev is None else r - r_prev,
                "switched":               _switched(r, r_sem, r_prev, sem_prev,
                                                    read_cfg.switch_sigma, min_delta),
                "state_AP_fraction":      _ap_fraction(r, r_p, r_ap),
                "magnet_current_A":       magnet_current_A,
                "assist_field_measured_mT": field_mT,
                "temperature_1_K":        t1_K,
                "temperature_2_K":        t2_K,
            }
            r_prev, sem_prev = r, r_sem
            records.append(record)
            if on_point is not None:
                on_point(record)

            if write_csv is not None:
                write_csv(records)
            else:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(records).to_csv(output_file, index=False)

            log.info("pt %d/%d  I_pulse=%.4g A  R_NL=%.5g ± %.2g Ω  dR=%s  switched=%s  V_even=%.3e V",
                     idx, len(plan) - 1, record["pulse_current_A"], r, r_sem,
                     record["delta_R_ohm"], record["switched"], rv["even_mean"])
    finally:
        _output_off(source)

    log.info("Done. %d rows → '%s'  (switching current(s): %s A)", len(records), output_file,
             switch_currents_A(records) or "none")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Plot + standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame) -> None:
    """R_NL vs pulse current (a step = switching) above V_even vs the same
    axis (a bump that follows the pulse = heating). Colour = order in time."""
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_e) = plt.subplots(2, 1, sharex=True, figsize=(6, 7))
    x = df["pulse_current_A"] * 1e3
    for ax, col, lab in ((ax_r, "nl_resistance_ohm", "R_NL (Ω)"),
                         (ax_e, "voltage_even_V", "V_even (V)")):
        ax.plot(x, df[col], "-", lw=0.5, color="gray")
        sc = ax.scatter(x, df[col], c=df["point_index"], cmap="viridis", zorder=3)
        ax.set_ylabel(lab)
    ax_e.set_xlabel("pulse current (mA)")
    fig.colorbar(sc, ax=[ax_r, ax_e], label="point #")
    plt.show()


def main() -> None:
    """Standalone run. For a one-polarity sweep the magnet state must already be
    initialized (external field, by hand) — the TUI (nonlocal_switching_tui.py) does it for you and
    saves into the data convention (type NLSW)."""
    pulse_cfg = WritePulseConfig(width_s=1e-3, compliance_V=5.0)
    read_cfg = ReadConfig(sense_current_A=1e-4, compliance_V=2.0, n_reversals=5,
                          delay_after_pulse_s=1.0)   # + R_P_ohm / R_AP_ohm from a dc_spin_valve loop

    # ── pick ONE protocol (see module docstring) ─────────────────────────────
    pulses_A = linear_sweep(1e-3, 10e-3, 0.5e-3, bidirectional=False)   # ascending, one polarity
    # pulses_A = linear_sweep(-5e-3, 5e-3, 0.5e-3, bidirectional=True)  # hysteresis loop
    # pulses_A = [+3e-3, -3e-3] * 10                                    # toggle test
    # pulses_A = [+3e-3, 0, 0, 0]                                       # relaxation after a pulse
    points = [PulsePoint(float(i)) for i in pulses_A]

    _check_write_safety(pulse_cfg)          # before connect() — it leaves the 6221 live
    _check_pulse_currents(points)
    _check_read_safety(read_cfg, points)

    output_file = str(_DATA_DIR / f"nonlocal_switching_{datetime.now():%Y%m%d_%H%M%S}.csv")
    source = connect("GPIB0::20::INSTR", read_cfg.compliance_V, read_cfg.source_delay_s)
    source.output_low_grounded = False      # floating: the return goes through its own electrode
    voltmeter = connect_voltmeter(VoltmeterConfig(visa_resource="GPIB0::7::INSTR", nplc=5))

    df = None
    try:
        df = run_measurement(source, voltmeter, pulse_cfg, read_cfg, points,
                             output_file=output_file)
        print("\n", df.to_string(index=False))
    finally:
        safe_shutdown("6221", lambda: shutdown_source(source))
    if df is not None and len(df):
        plot_results(df)


if __name__ == "__main__":
    main()
