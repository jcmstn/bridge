#!/usr/bin/env python3
"""
SOT pulsed switching, 6221-only — hardware-timed 6221 write pulse (WAVE
square, one cycle) + delayed AC / MFLI harmonic readout (NO 4200A)
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-14

Same switching-curve idea as sot_pulsed_switching.py / sot_pulsed_switching_2h.py
(one write pulse per amplitude, delayed read, amplitude swept up and down at a
fixed tilted field) — but the 4200A PMU is gone entirely. A single Keithley
6221 does BOTH jobs, sequenced in time on the same output, and BOTH phases
run through the same SCPI subsystem (WAVE), just a different function:

  1. WAVE MODE, square function, exactly ONE cycle: a single hardware-timed
     current pulse — the write pulse (see "How the write pulse works" below;
     this is ``instruments/keithley6221.py::fire_wave_pulse``).
  2. WAVE MODE, sine function, continuous: an AC sine with a Trigger Link
     phase marker, so a single Zurich MFLI (externally referenced via its
     Aux Input — same mechanism as sot_pulsed_switching_2h.py) locks in on
     ONE harmonic of V_xy (``ReadConfig.harmonic``, default 2 — the standard
     harmonic-Hall SOT technique; set 1 to read the resistive AHE/PHE signal
     instead).

Because there is only ever ONE source on the shared pins, there is no
shared-bus off/on safety dance the way sot_pulsed_switching.py needs between
the 4200A PMU and the 6221 — the 6221 simply can't pulse into itself. What
still matters is stopping whichever WAVE function is currently running
before arming the other one — see ``_six221_ac_output_off`` /
``_six221_ac_output_on`` below. The module never uses plain DC
``source_current`` sourcing at all now — pulse and read are both WAVE mode.

How the write pulse works (and what still needs bench-checking)
------------------------------------------------------------------
Verified against the Model 6220/6221 User's Manual (622x-900-01 Rev. C,
Section 7 "Wave Functions") — see ``instruments/keithley6221.py::
PulseWaveConfig`` for the full derivation. Short version: a square wave
swings between (offset - amplitude) and (offset + amplitude); setting
offset = amplitude = pulse_current_A / 2 gives a clean 0 → pulse_current_A →
0 unipolar pulse, one deliberate polarity — NOT Pulse Delta's alternating
triplet, and NOT anything that needs a 2182/2182A. Running it for exactly
one cycle (``waveform_duration_cycles = 1``) is genuinely hardware-timed:
the manual states plainly that "the output will turn off after the
currently set duration period has expired," and ``fire_wave_pulse`` polls
for that auto-off rather than trusting elapsed wall time or a software
sleep. ``pulse_width_measured_s`` on every row should now track the
requested ``pulse_width_s`` closely — a persistent large gap points at the
range/load-dependent duty-cycle floor (manual: "1 µs min. pulse duration
... limited by current range response and load impedance"; the datasheet's
own headline number is 5 µs), not GPIB/Python jitter.

The physically important consequence is still Joule heating, and hardware
timing doesn't change the physics: pulse energy scales as I²R·t. This
module's pulses are still realistically µs-to-ms scale, not the 4200A-PMU
variant's ~100 ns — call it a 10²-10⁴× increase in energy dumped into the
channel per pulse at the same current, smaller than the old software-timed
estimate but still real. A current that is a perfectly safe nanosecond
pulse on sot_pulsed_switching.py / sot_pulsed_switching_2h.py can still cook
this rig's device at the same amplitude here. Start well below your
expected switching current, watch ``pulse_width_measured_s`` (now for
confirming the true floor at your amplitude/range, not for catching GPIB
jitter), and increase cautiously. If you need a controlled rise/fall time
or a hardware-timed flat-top current READING synchronized to the pulse
(this module still only reads well after, via the separate AC/MFLI phase),
use sot_pulsed_switching_2h.py (4200A PMU) instead.

Wiring
------
    Keithley 6221 (current source)   HI ──▶ I+ pad ;  LO ──▶ I- pad
      Same channel for both phases — the write pulse and the AC read are
      the SAME two wires, sourced at different times, never simultaneously.

    Keithley 6221 TRIGGER LINK, phase marker on pin ``phasemarker_line``
    (default 1 — NOT the 6221's factory-default Trigger Link pin; confirm
    that default on your own unit) ──▶ Zurich Instruments MFLI AUX IN 1.
      One marker edge per AC excitation cycle, during the read phase only
      (the marker is only live while WAVE mode is armed/running — see
      sot_pulsed_switching_2h.py's docstring for the ExtRef/PLL mechanism
      this reuses verbatim).

    MFLI Signal Input (differential) ──▶ the transverse (Hall) voltage arms.

    Kepco BOP-GL ──GPIB──▶ electromagnet (ONE static tilted field, set once)
    Lake Shore 475 ──GPIB──▶ Gaussmeter probe at the sample

Bench-verify before trusting a run
-----------------------------------
Same MFLI ExtRef caveat as sot_pulsed_switching_2h.py: ``configure_external_
reference`` logs the live ``extrefs`` node tree on first connect — read it.
Also verify, once: that arming a fresh square-wave pulse cleanly overrides
the continuous sine wave left running from the previous read phase (this
module never assumes so — ``_six221_ac_output_off`` aborts + disables
before every pulse, and ``fire_wave_pulse`` sets every WAVE parameter
explicitly every cycle — but confirm the mode switch itself is clean on
your firmware). Below ~5-10 µs, check the oscilloscope-measured current
waveform shape too, not just ``pulse_width_measured_s`` — the 6221's
typical current-range rise time is ~2-3 µs (per its own settling-time spec),
so a pulse requested near the duty-cycle floor may not fully flat-top
before it ends.

Requirements: pymeasure, pyvisa, numpy, pandas, zhinst-core. A LabOne data
server reachable at ``mfli_host``/``mfli_port``. No 4200A / KXCI / KULT.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import zhinst.core as zi
from pymeasure.instruments.keithley import Keithley6221

from instruments.keithley6221 import (
    ACSourceConfig,
    PulseWaveConfig,
    connect_ac_source,
    fire_wave_pulse,
    shutdown_ac_source,
)
from instruments.mfli_daq import connect, connect_device, acquire_averaged
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

# The 6221's real hardware current range (pymeasure's source_current AND
# waveform_amplitude both validate to this) — the write pulse's hard bound.
# Not a "far more than legitimately needed" safety margin like the read
# ceiling below: switching currents genuinely may need to use most of it.
_WRITE_CURRENT_HARD_MAX_A = 0.105

# Same "far more than a Hall read legitimately needs" reasoning as
# sot_pulsed_switching_2h.py's read ceilings — still applies here even
# without a shared-bus PMU, since it's still a DUT-safety guard against a
# mistyped exponent, not only an inter-instrument one.
_READ_CURRENT_CEILING_A     = 10e-3
_READ_COMPLIANCE_CEILING_V  = 21.0


@dataclass
class WritePulseConfig:
    """Timing/compliance for a single hardware-timed write pulse (fired via
    ``instruments.keithley6221.fire_wave_pulse`` — WAVE mode, square
    function, one cycle) — see the module docstring's "How the write pulse
    works" section for the mechanism. The amplitude itself is NOT here —
    it's per-point (``PulsePoint.pulse_current_A``, the swept axis), checked
    against the hardware ceiling by ``_check_pulse_currents`` for the whole
    sweep at once, not duplicated onto this config."""
    width_s: float       = 1e-3    # requested hold time [s]; see pulse_width_measured_s
    compliance_V: float  = 5.0     # write-pulse voltage compliance [V]


@dataclass
class ReadConfig:
    """The 6221 AC + MFLI harmonic read, plus the wait before it."""
    sense_current_A: float        = 1e-4    # 6221 AC peak current amplitude for the read [A]
    compliance_V: float           = 2.0
    frequency_Hz: float           = 977.0   # AC excitation frequency [Hz] — avoid 50/60 Hz harmonics
    phasemarker_line: int         = 1       # 6221 Trigger Link pin -> MFLI Aux In
    harmonic: int                  = 2       # which harmonic the MFLI locks onto — 2f is the
                                              # standard harmonic-Hall SOT signal; 1 = resistive AHE/PHE
    n_averages: int                = 50      # independent MFLI demod samples averaged per read
    settle_after_enable_s: float  = 1.0      # dwell after the PLL reports locked, before reading [s]
    lock_timeout_s: float         = 5.0      # max wait for the MFLI ExtRef PLL to lock [s]
    delay_after_pulse_s: float    = 1.0      # wait between write-pulse end and the read [s]


def _check_write_safety(pulse_cfg: WritePulseConfig) -> None:
    """Sanity-check the pulse timing/compliance — the current ceiling is
    checked separately, per point, by _check_pulse_currents()."""
    if pulse_cfg.width_s <= 0:
        raise ValueError(f"WritePulseConfig.width_s must be > 0 s; got {pulse_cfg.width_s}.")
    if not 0.1 <= pulse_cfg.compliance_V <= 105.0:
        raise ValueError(
            f"WritePulseConfig.compliance_V must be in [0.1, 105] V (6221 hardware range); "
            f"got {pulse_cfg.compliance_V} V.")


def _check_pulse_currents(points: "List[PulsePoint]") -> None:
    """Refuse any pulse current outside the 6221's real hardware range —
    pymeasure's own validator silently CLIPS out-of-range writes rather than
    erroring, which would hide a mistyped exponent instead of catching it.
    Checked once for the whole sweep, before any hardware is touched."""
    bad = [pt.pulse_current_A for pt in points
           if not 0 < abs(pt.pulse_current_A) <= _WRITE_CURRENT_HARD_MAX_A]
    if bad:
        raise ValueError(
            f"Pulse current(s) {bad} A are zero or exceed the 6221's hardware range "
            f"±{_WRITE_CURRENT_HARD_MAX_A} A.")


def _check_read_safety(read_cfg: ReadConfig) -> None:
    """Same guard as sot_pulsed_switching_2h.py — refuse a read current /
    compliance that has no business driving the DUT continuously."""
    if not 0 < read_cfg.sense_current_A <= _READ_CURRENT_CEILING_A:
        raise ValueError(
            f"sense_current_A must be in (0, {_READ_CURRENT_CEILING_A} A]; got "
            f"{read_cfg.sense_current_A} A. The Hall read needs microamps-to-"
            "milliamps — check for a mistyped exponent.")
    if not 0 < read_cfg.compliance_V <= _READ_COMPLIANCE_CEILING_V:
        raise ValueError(
            f"compliance_V must be in (0, {_READ_COMPLIANCE_CEILING_V} V]; got "
            f"{read_cfg.compliance_V} V.")


@dataclass
class FilterConfig:
    """Lock-in filter parameters — same shape as mfli_dual_harmonic.py's."""
    time_constant_s: float = 0.3
    order: int             = 4
    sinc_filter: bool      = True


@dataclass
class ExtRefConfig:
    """The MFLI external-reference PLL, locked to the 6221's phase marker
    wired into an Aux Input. Same mechanism, same bench-verify caveat, as
    sot_pulsed_switching_2h.py::ExtRefConfig."""
    device: str        = "dev1234"
    extref_index: int  = 0     # which ExtRef/PLL module (0-based)
    aux_input_ch: int  = 0     # which Aux Input carries the marker (0-based; 0 = Aux In 1)
    osc_index: int     = 0     # oscillator the PLL locks — the demod references this


@dataclass
class DemodConfig:
    """The single demodulator on the externally-locked oscillator, at
    ``ReadConfig.harmonic``."""
    device: str
    demod_index: int
    harmonic: int
    osc_index: int       = 0
    input_ch: int        = 0
    differential: bool   = True
    ac_coupling: bool    = True
    input_range_V: float = 1.0
    sample_rate_Hz: float = 857.0
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class PulsePoint:
    pulse_current_A: float


# ─────────────────────────────────────────────────────────────────────────────
# MFLI setup helpers — identical mechanism to sot_pulsed_switching_2h.py;
# duplicated locally rather than imported, matching this suite's own
# convention (mfli/*.py programs each keep their own demod/output setup
# local even when nearly identical — see instruments/mfli_daq.py's
# docstring for why).
# ─────────────────────────────────────────────────────────────────────────────

def configure_external_reference(daq: "zi.ziDAQServer", cfg: ExtRefConfig,
                                  frequency_Hz: float) -> None:
    """See sot_pulsed_switching_2h.py::configure_external_reference — same
    node paths, same "bench-verify against the logged node tree" caveat."""
    d = cfg.device
    daq.setDouble(f"/{d}/oscs/{cfg.osc_index}/freq", frequency_Hz)
    try:
        nodes = daq.listNodesJSON(f"/{d}/extrefs/{cfg.extref_index}/*")
        log.info("MFLI %s extrefs/%d node tree (verify against this on first "
                 "bench run):\n%s", d, cfg.extref_index, nodes)
    except Exception:
        log.exception("Could not list /%s/extrefs/%d/* — node names below are "
                      "unverified for this device/firmware.", d, cfg.extref_index)
    daq.setInt(f"/{d}/extrefs/{cfg.extref_index}/adcselect", cfg.aux_input_ch)
    daq.setInt(f"/{d}/extrefs/{cfg.extref_index}/enable", 1)
    daq.sync()
    log.info("MFLI %s: oscillator %d locking to Aux In %d via extrefs/%d "
             "(target %.4f Hz)", d, cfg.osc_index, cfg.aux_input_ch + 1,
             cfg.extref_index, frequency_Hz)


def wait_for_reference_lock(daq: "zi.ziDAQServer", cfg: ExtRefConfig,
                             timeout_s: float,
                             stop_event: Optional[threading.Event] = None) -> bool:
    """See sot_pulsed_switching_2h.py::wait_for_reference_lock — never
    raises; an unreadable/unlocked PLL degrades to False, logged once."""
    path = f"/{cfg.device}/extrefs/{cfg.extref_index}/locked"
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if stop_event is not None and stop_event.is_set():
            return False
        try:
            if daq.getInt(path):
                return True
        except Exception:
            log.warning("Could not read ExtRef lock node %s — check the node "
                       "name against configure_external_reference()'s "
                       "listNodesJSON log.", path)
            return False
        time.sleep(0.05)
    return False


def configure_demodulator(daq: "zi.ziDAQServer", cfg: DemodConfig) -> None:
    """Identical node set to sot_pulsed_switching_2h.py::configure_demodulator."""
    d, di, flt = cfg.device, cfg.demod_index, cfg.filter

    daq.setInt(   f"/{d}/demods/{di}/oscselect",   cfg.osc_index)
    daq.setInt(   f"/{d}/demods/{di}/harmonic",    cfg.harmonic)
    daq.setDouble(f"/{d}/demods/{di}/timeconstant", flt.time_constant_s)
    daq.setInt(   f"/{d}/demods/{di}/order",         flt.order)
    daq.setInt(   f"/{d}/demods/{di}/sinc",          int(flt.sinc_filter))
    daq.setDouble(f"/{d}/demods/{di}/rate",          cfg.sample_rate_Hz)
    daq.setInt(   f"/{d}/demods/{di}/adcselect",     cfg.input_ch)
    daq.setInt(   f"/{d}/demods/{di}/enable",        1)

    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/diff",  int(cfg.differential))
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/ac",    int(cfg.ac_coupling))
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/imp50", 0)
    daq.setDouble(f"/{d}/sigins/{cfg.input_ch}/range", cfg.input_range_V)
    daq.setInt(   f"/{d}/sigins/{cfg.input_ch}/on",    1)
    daq.sync()
    log.info("Demod %s/demod%d  harmonic=%df  TC=%.3f s  order=%d  rate=%.1f Sa/s",
             d, di, cfg.harmonic, flt.time_constant_s, flt.order, cfg.sample_rate_Hz)


# ─────────────────────────────────────────────────────────────────────────────
# 6221 mode-switch helpers — the two things this module must get right on
# ONE instrument (no second source to isolate against).
# ─────────────────────────────────────────────────────────────────────────────

def _interruptible_sleep(seconds: float, stop_event: Optional[threading.Event]) -> None:
    if seconds <= 0:
        return
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if stop_event is not None and stop_event.is_set():
            return
        time.sleep(min(0.2, end - time.monotonic()))


def _six221_ac_output_off(source: Keithley6221) -> None:
    """Stop the AC wave (if running) and disable the output — the state
    needed before every write pulse and at rest between cycles. Harmless to
    call when nothing is armed."""
    source.waveform_abort()
    source.disable_source()


def _six221_ac_output_on(source: Keithley6221, compliance_V: float) -> None:
    """Resume the AC wave + phase marker for the read. Re-arms and restarts
    (rather than assuming a prior WAVE state survived the pulse phase in
    between) — also hands the MFLI's PLL a clean, unambiguous edge to
    re-lock to every time. WAVE parameters (function/amplitude/frequency/
    phase marker) were set once by connect_ac_source() at startup and
    persist across abort/re-arm — only compliance needs re-asserting here,
    since fire_wave_pulse() changes it (and the function/amplitude/offset/
    duty-cycle/duration) for the pulse phase."""
    source.source_compliance = compliance_V
    source.enable_source()
    source.waveform_arm()
    source.waveform_start()


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop  (docs/architecture.md §3 contract)
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    source: Keithley6221,
    daq: "zi.ziDAQServer",
    demod_cfg: DemodConfig,
    extref_cfg: ExtRefConfig,
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
    field_angle_from_oop_deg: Optional[float] = None,
    write_csv: Optional[Callable[[List[dict]], None]] = None,
    output_file: str = "sot_pulsed_switching_6221.csv",
) -> pd.DataFrame:
    """One write pulse + delayed harmonic read per amplitude in ``points``:
    (6221 wave off) → hardware-timed square-wave pulse → wait → (6221 wave
    on, wait for PLL lock, read ``read_cfg.harmonic``) → (6221 wave off).
    One row per amplitude; CSV rewritten in full every row.

    No 4200A, so there is no pulse-module return code to check and no
    consecutive-failure abort — a VISA-level failure raises naturally.

    Same static-field / stop_event / temp_ctrl=None-never-stops / reference_
    locked-is-logged-not-fatal semantics as sot_pulsed_switching_2h.py.
    """
    _check_write_safety(pulse_cfg)
    _check_pulse_currents(points)
    _check_read_safety(read_cfg)

    field_measured_mT = None
    if gaussmeter is not None and gauss_cfg is not None:
        field_measured_mT = read_field_mT(gaussmeter, gauss_cfg)
        log.info("Static field: %.4f mT measured (magnet current %s A)",
                 field_measured_mT, magnet_current_A)

    records: List[dict] = []

    for a_idx, pt in enumerate(points):
        if stop_event is not None and stop_event.is_set():
            log.info("Aborted after %d / %d amplitudes.", len(records), len(points))
            _six221_ac_output_off(source)
            return pd.DataFrame(records)

        # ── 1. ensure the wave is stopped ────────────────────────────
        _six221_ac_output_off(source)

        # ── 2. hardware-timed write pulse ──────────────────────────────
        pulse_wave_cfg = PulseWaveConfig(pulse_current_A=pt.pulse_current_A,
                                         width_s=pulse_cfg.width_s,
                                         compliance_V=pulse_cfg.compliance_V)
        pinfo = fire_wave_pulse(source, pulse_wave_cfg, stop_event)

        # ── 3. wait ─────────────────────────────────────────────────
        _interruptible_sleep(read_cfg.delay_after_pulse_s, stop_event)

        # ── 4. AC wave ON, wait for PLL lock, settle, read ────────────
        _six221_ac_output_on(source, read_cfg.compliance_V)
        locked = wait_for_reference_lock(daq, extref_cfg, read_cfg.lock_timeout_s,
                                         stop_event)
        if not locked:
            log.warning("MFLI reference PLL did not report locked within %.2g s "
                       "— reading anyway; this row is tagged reference_locked=False.",
                       read_cfg.lock_timeout_s)
        _interruptible_sleep(read_cfg.settle_after_enable_s, stop_event)

        d = acquire_averaged(daq, demod_cfg, read_cfg.n_averages)
        if d["overload"]:
            log.warning("Input overload at pulse %.4g A — this reading is not "
                       "trustworthy.", pt.pulse_current_A)

        # ── 5. wave OFF again ──────────────────────────────────────
        _six221_ac_output_off(source)

        t1_K, t2_K = read_temperature(temp_ctrl, temp_cfg) if temp_cfg is not None else (None, None)

        I_peak_A = read_cfg.sense_current_A
        record = {
            "amplitude_index":  a_idx,
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pulse_current_A":  pt.pulse_current_A,
            "pulse_width_s":    pulse_cfg.width_s,
            "pulse_width_measured_s": pinfo["pulse_width_measured_s"],
            "reference_locked": locked,
            "harmonic":          read_cfg.harmonic,
            "excitation_frequency_Hz": daq.getDouble(f"/{extref_cfg.device}/oscs/{extref_cfg.osc_index}/freq"),
            "excitation_current_A_peak": I_peak_A,
            "excitation_current_A_rms":  I_peak_A / 2 ** 0.5,
            "excitation_current_convention": "peak; 6221 waveform_amplitude is peak, not RMS",
            "demod_output_convention": (
                "RMS; ZI demodulator X/Y/R nodes report the RMS amplitude of "
                "the input signal's component at the reference frequency"),
            "demod_X_V":       d["x_mean"],
            "demod_Y_V":       d["y_mean"],
            "demod_R_V":       d["r_mean"],
            "demod_theta_deg": d["theta_mean"],
            "demod_R_std_V":   d["r_std"],
            "demod_overload":  d["overload"],
            "magnet_current_A":  magnet_current_A,
            "assist_field_measured_mT": field_measured_mT,
            "field_angle_from_oop_deg": field_angle_from_oop_deg,
            "temperature_1_K":   t1_K,
            "temperature_2_K":   t2_K,
        }
        records.append(record)
        if on_point is not None:
            on_point(record)

        if write_csv is not None:
            write_csv(records)
        else:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(output_file, index=False)

        log.info("amp %d/%d  I_pulse=%.4g A (measured width %.4g s)  V_%df=%.4e V",
                 a_idx + 1, len(points), pt.pulse_current_A,
                 pinfo["pulse_width_measured_s"], read_cfg.harmonic, d["r_mean"])

    log.info("Done. %d rows → '%s'", len(records), output_file)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    pulse_cfg = WritePulseConfig(width_s=1e-3, compliance_V=5.0)
    read_cfg = ReadConfig(sense_current_A=1e-4, harmonic=2, n_averages=50,
                          delay_after_pulse_s=1.0)
    PULSE_CURRENTS_A = list(linear_sweep(1e-3, 10e-3, 1e-3, bidirectional=True))
    points = [PulsePoint(pulse_current_A=float(i)) for i in PULSE_CURRENTS_A]

    _check_write_safety(pulse_cfg)
    _check_pulse_currents(points)
    _check_read_safety(read_cfg)   # before connect_ac_source — connect() leaves the 6221 live

    ac_cfg = ACSourceConfig(visa_resource="GPIB0::20::INSTR",
                            amplitude_A=read_cfg.sense_current_A,
                            frequency_Hz=read_cfg.frequency_Hz, compliance_V=read_cfg.compliance_V,
                            phasemarker_line=read_cfg.phasemarker_line)
    magnet_cfg = MagnetConfig(visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
                              voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05)
    gauss_cfg = GaussmeterConfig(visa_resource="GPIB0::12::INSTR", unit="T", n_averages=10)
    temp_cfg = TemperatureControllerConfig(
        visa_resource="TCPIP0::192.168.1.5::7020::SOCKET", sensor_uids=("MB1.T1",))

    MFLI_DEVICE = "dev1234"
    extref_cfg = ExtRefConfig(device=MFLI_DEVICE, aux_input_ch=0, osc_index=0)
    demod_cfg = DemodConfig(device=MFLI_DEVICE, demod_index=1, harmonic=read_cfg.harmonic,
                            osc_index=0, filter=FilterConfig(time_constant_s=0.3, order=4,
                                                             sinc_filter=True))

    FIELD_ANGLE_FROM_OOP_DEG = 85.0
    STATIC_MAGNET_CURRENT_A = 1.5
    OUTPUT_FILE = str(_DATA_DIR / f"sot_pulsed_6221_{datetime.now():%Y%m%d_%H%M%S}.csv")

    source = connect_ac_source(ac_cfg)
    _six221_ac_output_off(source)

    daq = connect("localhost", 8004)
    connect_device(daq, MFLI_DEVICE, interface="1GbE")
    configure_external_reference(daq, extref_cfg, ac_cfg.frequency_Hz)
    configure_demodulator(daq, demod_cfg)

    magnet = connect_magnet(magnet_cfg)
    gaussmeter = connect_gaussmeter(gauss_cfg)
    temp_ctrl = connect_temperature_controller(temp_cfg)

    set_magnet_current(magnet, magnet_cfg, STATIC_MAGNET_CURRENT_A, gaussmeter, gauss_cfg)

    try:
        df = run_measurement(source, daq, demod_cfg, extref_cfg, pulse_cfg, read_cfg, points,
                             gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
                             temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
                             magnet_current_A=STATIC_MAGNET_CURRENT_A,
                             field_angle_from_oop_deg=FIELD_ANGLE_FROM_OOP_DEG,
                             output_file=OUTPUT_FILE)
        print("\n", df.to_string(index=False))
    finally:
        safe_shutdown("6221", lambda: shutdown_ac_source(source))
        safe_shutdown("magnet", lambda: shutdown_magnet(magnet, magnet_cfg))
        safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        safe_shutdown("temperature controller", lambda: shutdown_temperature_controller(temp_ctrl))


if __name__ == "__main__":
    main()
