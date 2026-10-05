"""
SRS SR830 DSP Lock-in Amplifier — connect/acquisition/reference helpers
=========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-10-02

The SR830 alternative to instruments/mfli_daq.py: same capabilities
(configure, averaged X/Y read, paired read of two units over one window,
external-reference lock + per-point lock re-check, overload flag, run-time
estimate) and the same result dict as mfli_daq.acquire_averaged(), so an
``*_sr830.py`` program can mirror its MFLI counterpart. Each one is done
the way the SR830 manual (thinksrs.com SR830m.pdf, chapter 5) supports it,
not by porting the MFLI mechanics:

  - averaging       -> internal data buffer (DDEF CH1=X / CH2=Y, SRAT, REST,
                       STRT, PAUS, SPTS?, TRCB?), not a loop of OUTP?/SNAP?
  - overload        -> LIAS? bits 0/1/2 (input/reserve, filter, output),
                       latched over the whole acquisition window
  - external ref    -> FMOD 0 + RSLP (sine / TTL rising / TTL falling)
  - lock status     -> LIAS? bit 3 (reference unlock, latched)
  - auto functions  -> AGAN / ARSV + Serial Poll bit 1 (IFC) wait; APHS + settle
  - settle time     -> the manual's 99% wait-time table (5/7/9/10 x TC)

The register queries use the per-bit form ``LIAS? i``: the manual states
that reading bit i clears only bit i, whereas reading the whole byte clears
all of it. That way acquire_averaged()'s overload check and
check_reference_locked() never steal each other's latched events.

Built on pymeasure's SR830 class for the discrete-value setters (TC,
sensitivity, slope etc. snap to the allowed table values); everything the
pymeasure class does not do the manual-supported way is written raw. In
particular ``OUTX 1`` is sent before any query (manual: required to route
responses to GPIB; pymeasure never sends it), and pymeasure's
start_buffer() is NOT used — it enables FAST mode, which streams every
sample over GPIB and must be read continuously.

ONE SR830 = ONE signal input + ONE demodulator + ONE reference. It cannot
read 1f and 2f, or Rxy and Rxx, at the same time — use two units, exactly
like the two MDS-synced MFLIs today. Two units need no clock sync: unit B
runs on an external reference taken from unit A's rear-panel TTL OUT
(manual: "active even when locked to an external reference").

Wiring (dual harmonic, SR830 sine out as the excitation):

    SR830 A  SINE OUT ──[R_series]──> sample I+          (internal ref, HARM 1)
             A/B in   <── V+ / V-                          1f
             TTL OUT (rear) ──> SR830 B REF IN            (B: external, TTL rising)
    SR830 B  A/B in   <── V+ / V-                          (HARM 2) 2f

    Or a 6221 as the source: its Trigger Link phase marker (TTL) into BOTH
    units' REF IN, both external, HARM 1 / HARM 2.

Usage example:
    from instruments.sr830 import (LockinConfig, connect, wait_for_reference_lock,
                                   acquire_averaged_pair, settle_time_s)

    cfg_1f = LockinConfig(visa_resource="GPIB0::8::INSTR", frequency_Hz=1333.0)
    cfg_2f = LockinConfig(visa_resource="GPIB0::9::INSTR", reference="external",
                          ext_slope="ttl_rising", frequency_Hz=1333.0, harmonic=2)
    la, lb = connect(cfg_1f), connect(cfg_2f)
    if not wait_for_reference_lock(lb, timeout_s=5.0):
        raise RuntimeError("2f unit not locked to the 1f unit's TTL OUT")
    time.sleep(settle_time_s(cfg_2f))
    d1, d2 = acquire_averaged_pair(la, cfg_1f, lb, cfg_2f, n_averages=200)
    # d1/d2: same keys as mfli_daq.acquire_averaged()

    # In a program: the "Lock-in: SR830" toggle (HARM / HARM6 / SOT2H / SOT1I)
    reader = SR830Read()
    reader.open([cfg_1f, cfg_2f])          # source unit first
    reader.lock(timeout_s=5.0)
    run_measurement(..., lockin=reader)    # the engine's MFLI calls swapped out
    reader.close()
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pymeasure.instruments.srs import SR830

from instruments.run_time import GPIB_TXN_S

log = logging.getLogger(__name__)

BUFFER_POINTS = 16383                       # manual: "holds 16383 samples"
MAX_DETECTION_HZ = 102e3                    # manual: HARM i limited by i*f <= 102 kHz
SAMPLE_RATES_HZ = [0.0625 * 2 ** k for k in range(14)]   # SRAT 0..13 (14 = trigger)
_RSLP = {"sine": 0, "ttl_rising": 1, "ttl_falling": 2}
_SETTLE_TC = {6: 5, 12: 7, 18: 9, 24: 10}   # manual: wait time to 99% per filter slope
_OVERLOAD_BITS = (0, 1, 2)                  # LIAS: RSRV/INPT, FILTR, OUTPT
_UNLOCK_BIT = 3                             # LIAS: UNLK
# Per acquire, per unit: 3 overload clears + REST + STRT + PAUS + SPTS? +
# 2 TRCB? + 3 overload reads.
_ACQ_TXNS = 12


@dataclass
class LockinConfig:
    """One SR830. String fields use pymeasure's SR830 value names
    (``SR830.INPUT_CONFIGS`` etc.). With ``reference="external"``,
    ``frequency_Hz`` is the EXPECTED reference frequency — it is not sent
    (the unit tracks REF IN) but is still used to validate HARM x f and for
    the synchronous-filter settle term."""
    visa_resource: str      = "GPIB0::8::INSTR"
    reference: str          = "internal"     # "internal" | "external"
    ext_slope: str          = "ttl_rising"   # "sine" | "ttl_rising" | "ttl_falling" (external only)
    frequency_Hz: float     = 1000.0
    sine_amplitude_V: float = 0.1            # SINE OUT [V rms], 0.004 … 5 (no "off" on an SR830)
    harmonic: int           = 1              # detection harmonic, 1 … 19999 with harmonic*f <= 102 kHz
    time_constant_s: float  = 0.1            # snapped UP to the next OFLT value
    filter_slope_dB: int    = 24             # 6 / 12 / 18 / 24 dB/oct
    sync_filter: bool       = False          # only acts below 200 Hz detection frequency
    sensitivity_V: float    = 1e-3           # full scale, snapped UP to the next SENS value
    input_config: str       = "A - B"        # "A" | "A - B" | "I (1 MOhm)" | "I (100 MOhm)"
    coupling: str           = "AC"           # "AC" | "DC"
    grounding: str          = "Float"        # "Float" | "Ground"
    notch: str              = "None"         # "None" | "Line" | "2 x Line" | "Both"
    reserve: str            = "Normal"       # "High Reserve" | "Normal" | "Low Noise"
    sample_rate_Hz: float   = 512.0          # buffer rate, snapped UP to 62.5 mHz x 2^k (max 512 Hz)


def validate(cfg: LockinConfig) -> None:
    """Reject settings the SR830 would otherwise silently clamp (HARM) or
    that the manual rules out — raised before any instrument I/O."""
    if cfg.reference not in ("internal", "external"):
        raise ValueError(f"reference must be 'internal' or 'external', got {cfg.reference!r}")
    if cfg.ext_slope not in _RSLP:
        raise ValueError(f"ext_slope must be one of {sorted(_RSLP)}, got {cfg.ext_slope!r}")
    if not 1 <= cfg.harmonic <= 19999:
        raise ValueError(f"harmonic must be 1 … 19999, got {cfg.harmonic}")
    if cfg.harmonic * cfg.frequency_Hz > MAX_DETECTION_HZ:
        raise ValueError(f"harmonic x frequency = {cfg.harmonic * cfg.frequency_Hz:.0f} Hz "
                         f"exceeds the SR830's {MAX_DETECTION_HZ:.0f} Hz detection limit "
                         "(the unit would silently lower the harmonic)")
    if cfg.reference == "external" and cfg.frequency_Hz < 1.0 and cfg.ext_slope == "sine":
        raise ValueError("below 1 Hz the SR830 needs a TTL reference (ext_slope='ttl_*')")
    if cfg.filter_slope_dB not in _SETTLE_TC:
        raise ValueError(f"filter_slope_dB must be 6/12/18/24 (filter order 1-4), "
                         f"got {cfg.filter_slope_dB}")
    if not 0.004 <= cfg.sine_amplitude_V <= 5.0:
        raise ValueError(f"SINE OUT must be 0.004 … 5 V rms, got {cfg.sine_amplitude_V:g} V")


def srat_index(sample_rate_Hz: float) -> int:
    """SRAT index of the smallest buffer rate >= sample_rate_Hz (512 Hz cap)."""
    for i, rate in enumerate(SAMPLE_RATES_HZ):
        if rate >= sample_rate_Hz:
            return i
    return len(SAMPLE_RATES_HZ) - 1


def _wait_idle(lockin: SR830, timeout_s: float = 60.0) -> bool:
    """Block until Serial Poll bit 1 (IFC, "no command execution in
    progress") is set — the manual's completion check for *RST, AGAN, ARSV."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if int(lockin.ask("*STB? 1")):
            return True
        time.sleep(0.05)
    log.warning("SR830 still busy after %.0f s (IFC bit never set)", timeout_s)
    return False


def connect(cfg: LockinConfig) -> SR830:
    """Open and fully configure one SR830 from `cfg`. The TC, sensitivity
    and buffer rate the unit actually applied (snapped to its discrete
    tables) are written back into `cfg`, so acquire_averaged()'s window and
    settle_time_s() use the real values."""
    validate(cfg)
    lockin = SR830(cfg.visa_resource)
    lockin.write("OUTX 1")      # manual: before any query, else replies go to RS232
    lockin.write("*RST")        # comms setup + status registers survive *RST
    _wait_idle(lockin)
    lockin.write("*CLS")

    if cfg.reference == "external":
        lockin.reference_source = "External"
        lockin.write(f"RSLP {_RSLP[cfg.ext_slope]}")
    else:
        lockin.reference_source = "Internal"
        lockin.frequency = cfg.frequency_Hz
    lockin.sine_voltage = cfg.sine_amplitude_V
    lockin.write(f"HARM {cfg.harmonic}")
    lockin.input_config = cfg.input_config
    lockin.input_coupling = cfg.coupling
    lockin.input_grounding = cfg.grounding
    lockin.input_notch_config = cfg.notch
    lockin.reserve = cfg.reserve
    lockin.sensitivity = cfg.sensitivity_V
    lockin.time_constant = cfg.time_constant_s
    lockin.filter_slope = cfg.filter_slope_dB
    lockin.filter_synchronous = cfg.sync_filter

    # Data buffer stores the CH1/CH2 *displays*: X and Y, no ratio, offset
    # and expand off (both would rescale the stored values). Loop mode keeps
    # the newest 16383 points in bins 0..16382 once it wraps, so the last
    # bins are always the freshest; TSTR 0 = STRT starts storage, not a TTL.
    srat = srat_index(cfg.sample_rate_Hz)
    for cmd in ("DDEF 1,0,0", "DDEF 2,0,0", "OEXP 1,0,0", "OEXP 2,0,0",
                "SEND 1", "TSTR 0", f"SRAT {srat}"):
        lockin.write(cmd)

    cfg.time_constant_s = lockin.time_constant
    cfg.sensitivity_V = lockin.sensitivity
    cfg.sample_rate_Hz = SAMPLE_RATES_HZ[srat]
    log.info("SR830 %s: %s ref, f=%.4f Hz (measured), HARM %d, TC=%g s, %d dB/oct, "
             "sens=%g V, %s, buffer %g Sa/s", cfg.visa_resource, cfg.reference,
             lockin.frequency, cfg.harmonic, cfg.time_constant_s,
             cfg.filter_slope_dB, cfg.sensitivity_V, cfg.input_config, cfg.sample_rate_Hz)
    return lockin


def shutdown(lockin: SR830) -> None:
    """Drop SINE OUT to its 4 mV minimum (an SR830 cannot switch it off),
    return the front panel to LOCAL and close the VISA session."""
    try:
        lockin.sine_voltage = 0.004
        lockin.write("LOCL 0")
    finally:
        lockin.adapter.close()


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition — internal data buffer
# ─────────────────────────────────────────────────────────────────────────────

def poll_window_s(time_constant_s: float, n_averages: int, sample_rate_Hz: float) -> float:
    """Length of the buffer-storage window -- the same floor as
    mfli_daq.poll_window_s(): at least 3 x TC so the samples are not all one
    correlated reading, and enough time to store n_averages (+50%)."""
    return max(0.1, 3.0 * time_constant_s, (n_averages * 1.5) / sample_rate_Hz)


def acquire_s(time_constant_s: float, n_averages: int, sample_rate_Hz: float) -> float:
    """Modelled wall time of one acquire_averaged() call (window + GPIB I/O)."""
    return poll_window_s(time_constant_s, n_averages, sample_rate_Hz) + _ACQ_TXNS * GPIB_TXN_S


def settle_time_s(cfg: LockinConfig) -> float:
    """Manual's wait time to reach 99% of a step: 5/7/9/10 x TC for
    6/12/18/24 dB/oct, plus one detection period with the synchronous filter."""
    t = _SETTLE_TC[cfg.filter_slope_dB] * cfg.time_constant_s
    if cfg.sync_filter:
        t += 1.0 / (cfg.harmonic * cfg.frequency_Hz)
    return t


def _clear_overload(lockin: SR830) -> None:
    for bit in _OVERLOAD_BITS:
        lockin.ask(f"LIAS? {bit}")


_overload_warned: set = set()


def _read_overload(lockin: SR830) -> Optional[bool]:
    """Any overload latched since _clear_overload(). None if unreadable
    (degrade the run, don't crash it), warned once per unit."""
    try:
        return any(int(lockin.ask(f"LIAS? {bit}")) for bit in _OVERLOAD_BITS)
    except Exception:
        key = id(lockin)
        if key not in _overload_warned:
            _overload_warned.add(key)
            log.warning("Could not read SR830 LIAS overload bits — overload "
                        "reported as unknown for this unit.")
        return None


def _read_trace(lockin: SR830, channel: int, start: int, count: int) -> np.ndarray:
    """TRCB? — 4-byte IEEE floats, no delimiter, EOI on the last byte; read
    as raw bytes (must not stop on a LF/CR byte inside the data)."""
    lockin.write(f"TRCB? {channel},{start},{count}")
    raw = lockin.read_bytes(4 * count)
    return np.frombuffer(raw, dtype="<f4").astype(float)


def _reduce_xy(x: np.ndarray, y: np.ndarray) -> dict:
    """Mean/SEM/std reduction, identical in meaning to
    mfli_daq._finish_average(): R/theta from the VECTOR-averaged phasor
    (mean X, mean Y), never the mean of per-sample magnitudes (rectifies
    noise, biased up near the noise floor); r_sem is x_sem/y_sem propagated
    onto r_mean; *_std are the raw per-sample scatter."""
    # ponytail: duplicates ~20 lines of mfli_daq._finish_average (sr830 must
    # not import zhinst); fold into a shared helper if a third lock-in appears.
    n = len(x)
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    r_mean = float(np.hypot(x_mean, y_mean))
    if n >= 2:
        x_sem = float(np.std(x, ddof=1) / np.sqrt(n))
        y_sem = float(np.std(y, ddof=1) / np.sqrt(n))
    else:
        x_sem = y_sem = float("nan")
    r_sem = (float(np.hypot(x_mean * x_sem, y_mean * y_sem) / r_mean)
             if n >= 2 and r_mean > 0 else float("nan"))
    return {
        "x_mean":     x_mean,
        "y_mean":     y_mean,
        "r_mean":     r_mean,
        "theta_mean": float(np.degrees(np.arctan2(y_mean, x_mean))),
        "x_sem":      x_sem,
        "y_sem":      y_sem,
        "r_sem":      r_sem,
        "r_std":      float(np.std(np.hypot(x, y))),
        "x_std":      float(np.std(x)),
        "y_std":      float(np.std(y)),
        "n_samples":  n,
    }


def _acquire(units: list, n_averages: int, duration_s: float,
             stop_event: Optional[threading.Event]) -> list:
    """Run one buffer window on every unit in `units`: clear the latched
    overload bits and the buffer, STRT all units back-to-back, wait, PAUS
    all, then read the freshest n_averages points of CH1 (X) / CH2 (Y)."""
    if not 1 <= n_averages <= BUFFER_POINTS:
        raise ValueError(f"n_averages must be 1 … {BUFFER_POINTS}, got {n_averages}")
    for lockin in units:
        _clear_overload(lockin)
        lockin.write("REST")
    for lockin in units:
        lockin.write("STRT")
    if stop_event is not None:
        stop_event.wait(duration_s)
    else:
        time.sleep(duration_s)
    for lockin in units:
        lockin.write("PAUS")

    results = []
    for lockin in units:
        stored = int(lockin.ask("SPTS?"))
        if stored == 0:
            raise RuntimeError("SR830 buffer is empty after the acquisition window. "
                               "Check SRAT / that storage started.")
        k = min(stored, n_averages)
        start = min(stored, BUFFER_POINTS) - k
        x = _read_trace(lockin, 1, start, k)
        y = _read_trace(lockin, 2, start, k)
        d = _reduce_xy(x, y)
        d["overload"] = _read_overload(lockin)
        results.append(d)
    return results


def acquire_averaged(lockin: SR830, cfg: LockinConfig, n_averages: int,
                     stop_event: Optional[threading.Event] = None) -> dict:
    """Store at least `n_averages` X/Y samples in the buffer over a window
    of poll_window_s() and reduce them. Returns the same dict as
    mfli_daq.acquire_averaged() (x/y/r/theta means, *_sem, *_std,
    n_samples, overload). `stop_event` cuts the window short; whatever was
    stored is still reduced."""
    duration_s = poll_window_s(cfg.time_constant_s, n_averages, cfg.sample_rate_Hz)
    return _acquire([lockin], n_averages, duration_s, stop_event)[0]


def acquire_averaged_pair(lockin_a: SR830, cfg_a: LockinConfig,
                          lockin_b: SR830, cfg_b: LockinConfig, n_averages: int,
                          stop_event: Optional[threading.Event] = None) -> tuple:
    """Same as two acquire_averaged() calls but over ONE shared window (the
    longer of the two). As with the MFLI pair read, both buffers cover the
    same wall-clock interval (STRT/PAUS a few ms apart), they are NOT
    sample-aligned."""
    duration_s = max(poll_window_s(cfg_a.time_constant_s, n_averages, cfg_a.sample_rate_Hz),
                     poll_window_s(cfg_b.time_constant_s, n_averages, cfg_b.sample_rate_Hz))
    return tuple(_acquire([lockin_a, lockin_b], n_averages, duration_s, stop_event))


# ─────────────────────────────────────────────────────────────────────────────
# External reference lock + auto functions
# ─────────────────────────────────────────────────────────────────────────────

def check_reference_locked(lockin: SR830) -> Optional[bool]:
    """False if a reference unlock was latched (LIAS bit 3) since the last
    check — i.e. the lock dropped at any moment in between, not just now.
    Reading the bit clears it. Never raises; None if unreadable. Always
    True on an internally referenced unit."""
    try:
        return not int(lockin.ask(f"LIAS? {_UNLOCK_BIT}"))
    except Exception:
        log.warning("Could not read SR830 LIAS unlock bit — reference lock "
                    "status unknown this point.")
        return None


def wait_for_reference_lock(lockin: SR830, timeout_s: float,
                            stop_event: Optional[threading.Event] = None,
                            dwell_s: float = 0.5) -> bool:
    """Wait until the unlock bit stays clear for a full `dwell_s` (the
    latched bit is cleared first, so setup-time unlocks don't count).
    Raise `dwell_s` for references below a few Hz, where one period alone
    exceeds 0.5 s. Never raises."""
    check_reference_locked(lockin)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if stop_event is not None and stop_event.is_set():
            return False
        time.sleep(dwell_s)
        locked = check_reference_locked(lockin)
        if locked is None:
            return False
        if locked:
            try:
                log.info("SR830 locked to external reference at %.4f Hz", lockin.frequency)
            except Exception:
                pass
            return True
    return False


def auto_gain(lockin: SR830, cfg: LockinConfig) -> None:
    """AGAN, wait for completion, write the chosen sensitivity back to cfg.
    Manual: AGAN does nothing when TC > 1 s."""
    if cfg.time_constant_s > 1.0:
        log.warning("SR830 AGAN skipped: it does nothing at TC = %g s (> 1 s)",
                    cfg.time_constant_s)
        return
    lockin.write("AGAN")
    _wait_idle(lockin)
    cfg.sensitivity_V = lockin.sensitivity
    log.info("SR830 auto gain -> sensitivity %g V", cfg.sensitivity_V)


def auto_reserve(lockin: SR830) -> None:
    """ARSV, wait for completion."""
    lockin.write("ARSV")
    _wait_idle(lockin)


def auto_phase(lockin: SR830, cfg: LockinConfig) -> float:
    """APHS, then wait the settle time (the manual gives no completion bit:
    "the outputs will take many time constants to reach their new values")
    and return the new reference phase [deg]. APHS silently does nothing on
    an unstable phase, so an unchanged phase is logged as a warning."""
    before = lockin.phase
    lockin.write("APHS")
    time.sleep(settle_time_s(cfg))
    after = lockin.phase
    if after == before:
        log.warning("SR830 APHS left the phase at %.2f deg (unstable phase?)", after)
    else:
        log.info("SR830 auto phase: %.2f -> %.2f deg", before, after)
    return after


# ─────────────────────────────────────────────────────────────────────────────
# Lock-in toggle — the SR830 side of the MFLI programs' "Lock-in" select
# (HARM / HARM6 / SOT2H / SOT1I). Their run_measurement() takes an SR830Read
# as ``lockin=`` in place of the MFLI daq calls; None keeps the MFLI path.
# ─────────────────────────────────────────────────────────────────────────────

DEMOD_OUTPUT_CONVENTION = (
    "RMS; SR830 X/Y outputs report the RMS amplitude of the input signal's "
    "component at the detection harmonic"
)


def config_from_form(visa_resource: str, *, harmonic: int, frequency_Hz: float,
                     time_constant_s: float, order: int, sinc_filter: bool,
                     differential: bool, ac_coupling: bool, sensitivity_V: float,
                     sample_rate_Hz: float, reference: str = "external",
                     sine_amplitude_V: float = 0.004) -> LockinConfig:
    """One SR830's config from the MFLI form's fields: filter order n ->
    6n dB/oct (validate() rejects order 5-8), the sinc switch -> the
    synchronous filter, differential -> "A - B" (else "A"), AC coupling ->
    coupling. External units take their reference from REF IN as TTL
    (an SR830's rear TTL OUT, or the 6221's Trigger Link phase marker).
    TC, sensitivity and buffer rate are snapped up to the unit's tables
    here already (connect() does the same on the unit), so a form's
    estimate and summary see the values that will actually apply."""
    return LockinConfig(
        visa_resource=visa_resource, reference=reference, ext_slope="ttl_rising",
        frequency_Hz=frequency_Hz, sine_amplitude_V=sine_amplitude_V, harmonic=harmonic,
        time_constant_s=snap_up(time_constant_s, SR830.TIME_CONSTANTS),
        filter_slope_dB=6 * order, sync_filter=sinc_filter,
        sensitivity_V=snap_up(sensitivity_V, SR830.SENSITIVITIES),
        input_config="A - B" if differential else "A",
        coupling="AC" if ac_coupling else "DC",
        sample_rate_Hz=SAMPLE_RATES_HZ[srat_index(sample_rate_Hz)],
    )


def snap_up(value: float, table) -> float:
    """The smallest table value >= value (the largest if none) — what the
    SR830 applies, same rule as pymeasure's truncated_discrete_set."""
    return next((v for v in sorted(table) if value <= v), max(table))


@dataclass
class SR830Read:
    """The 1 or 2 connected SR830s a run reads, in column order (unit 0 =
    the leader / 1f, unit 1 = the follower / 2f). Start it empty and open()
    it, so close() still reaches a unit that opened before a later one
    failed — unit A's SINE OUT is live the moment it is configured."""
    units: list = field(default_factory=list)      # [(SR830, LockinConfig), ...]

    def open(self, cfgs) -> None:
        """Connect every cfg in order (a reference source before the unit
        locked to it)."""
        for cfg in cfgs:
            self.units.append((connect(cfg), cfg))

    def lock(self, timeout_s: float, stop_event: Optional[threading.Event] = None) -> bool:
        """Wait for the external units to lock, then settle. Returns the
        lock result — logged, not fatal, like the MFLI ExtRef."""
        locked = self.wait_locked(timeout_s, stop_event)
        if not locked:
            log.warning("SR830 external reference not locked within %.2g s — "
                        "check the TTL into REF IN.", timeout_s)
        time.sleep(max(settle_time_s(cfg) for _, cfg in self.units))
        return locked

    def read(self, n_averages: int, stop_event: Optional[threading.Event] = None) -> list:
        """One acquire_averaged() dict per unit, over one shared window."""
        if len(self.units) == 1:
            (lk, cfg), = self.units
            return [acquire_averaged(lk, cfg, n_averages, stop_event)]
        (la, ca), (lb, cb) = self.units
        return list(acquire_averaged_pair(la, ca, lb, cb, n_averages, stop_event))

    def wait_locked(self, timeout_s: float,
                    stop_event: Optional[threading.Event] = None) -> bool:
        return all(wait_for_reference_lock(lk, timeout_s, stop_event)
                   for lk, cfg in self.units if cfg.reference == "external")

    def locked(self) -> list:
        """Per unit: False if its reference unlocked since the last call,
        None on an internally referenced unit (nothing to lock to)."""
        return [check_reference_locked(lk) if cfg.reference == "external" else None
                for lk, cfg in self.units]

    def frequency_Hz(self) -> float:
        """Unit 0's reference frequency — measured, on an external unit."""
        return self.units[0][0].frequency

    def phase_deg(self, i: int) -> float:
        return self.units[i][0].phase

    def filter_meta(self, i: int) -> tuple:
        """(applied time constant [s], equivalent filter order) of unit i."""
        cfg = self.units[i][1]
        return cfg.time_constant_s, cfg.filter_slope_dB // 6

    def close(self) -> None:
        """shutdown() every unit, even if an earlier one fails."""
        errors = []
        for lk, cfg in self.units:
            if cfg.reference == "internal":
                log.warning("SR830 %s SINE OUT left at its 4 mV rms minimum — it cannot "
                            "be switched off; unplug it if that matters.", cfg.visa_resource)
            try:
                shutdown(lk)
            except Exception as e:      # noqa: BLE001 — re-raised below
                errors.append(e)
        if errors:
            raise errors[0]
