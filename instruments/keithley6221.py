"""
Keithley 6221 DC Current Source — shared connect/shutdown/acquisition helpers
===============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-06

The 6221 driver itself ships with pymeasure (pymeasure.instruments.keithley.
Keithley6221) — this module holds the connect/shutdown/acquisition wrapper
functions shared by the DC measurement programs.

Usage example (fixed sense-current case):
    from instruments.keithley6221 import SourceConfig, connect_source, shutdown_source

    src_cfg = SourceConfig(visa_resource="GPIB0::20::INSTR", sense_current_A=1e-3)
    source = connect_source(src_cfg)
    ...
    shutdown_source(source)

A program that instead sweeps the current itself (dc_iv_curve.py) keeps its
own SourceConfig shape (a sweep range, not a fixed sense current) and calls
the lower-level `connect()` directly with an explicit `initial_current_A`.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymeasure.instruments.keithley import Keithley2182, Keithley6221

log = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Keithley 6221 as a fixed DC sense-current source — the common case used
    by dc_hall_measurement.py, dc_gate_sweep.py and dc_spin_valve.py."""
    visa_resource: str     = "GPIB0::20::INSTR"
    sense_current_A: float = 1e-3    # Sense current magnitude [A]
    compliance_V: float    = 2.0     # Voltage compliance [V]
    source_delay_s: float  = 0.05    # Settle time after each current step [s]


@dataclass
class ACSourceConfig:
    """Keithley 6221 as an AC (sine) current source with a hardware phase
    marker on the Trigger Link, for external-reference lock-in detection
    (used by sot/sot_pulsed_switching_2h.py: an MFLI locks to this marker
    via its Aux Input, per the Zurich Instruments external-reference guide).

    `phasemarker_line` is the Trigger Link output line (1-6) the marker
    pulse (one edge per excitation cycle) appears on — set via SCPI
    (`:SOUR:WAVE:PMAR:OLIN`, confirmed against the Model 6220/6221
    Reference Manual, 622x-901-01 Rev. B) and read back by
    connect_ac_source() to confirm it took; no front-panel step needed. The
    Trigger Link's 8-pin DIN connector is a flat 1:1 map — DIN pin N =
    Trigger Link line N (pins 7/8 are ground) — so "line N" and "pin N" are
    the same thing. The 6221's own factory default is line 3 (same manual,
    p.7-10), and that's also what Zurich Instruments' own MFLI ↔ 6221
    external-reference guide wires (pin 3) — but default here is 1,
    matching this lab's actual cable (which only brings out pin 1 to the
    MFLI). Getting this wrong doesn't error — the 6221 happily outputs a
    clean marker on whichever line you ask for — it just means the real
    signal comes out on a different pin than whatever your cable taps, and
    what you see on the pin you're actually wired to is crosstalk, not the
    marker. Confirm against your own rig's cabling before trusting either
    default."""
    visa_resource: str        = "GPIB0::20::INSTR"
    amplitude_A: float        = 1e-4     # AC current amplitude, peak [A]
    frequency_Hz: float       = 977.0    # Excitation frequency [Hz] — avoid 50/60 Hz harmonics
    compliance_V: float       = 2.0      # Voltage compliance [V]
    ranging: str              = "best"   # "best" or "fixed"
    phasemarker_line: int     = 1        # Trigger Link line (1-6) the phase marker appears on — this lab's cable taps pin 1


def connect_ac_source(cfg: ACSourceConfig) -> Keithley6221:
    """Open and arm a Keithley 6221 as an AC current source with its phase
    marker enabled. Every WAVE parameter (function/amplitude/frequency/
    ranging/phase-marker state+line) must be set before `waveform_arm()` — a
    write after arming does not take effect until the next arm(), per the
    pymeasure driver's WAVE-mode example. `waveform_arm()` + `waveform_start()`
    is the official sequence for starting continuous sourcing (no separate
    `enable_source()` call needed — arming brings the output up itself); the
    duration is set to infinite so the wave runs until `waveform_abort()`.

    Reads the phase-marker line back after arming and raises if it didn't
    take — the usual cause is another Trigger Link function already owning
    that pin.
    """
    source = Keithley6221(cfg.visa_resource)
    source.reset()
    source.source_compliance = cfg.compliance_V
    source.waveform_function = "sine"
    source.waveform_amplitude = cfg.amplitude_A
    source.waveform_offset = 0.0
    source.waveform_frequency = cfg.frequency_Hz
    source.waveform_ranging = cfg.ranging
    source.waveform_use_phasemarker = True
    source.waveform_phasemarker_line = cfg.phasemarker_line
    source.waveform_duration_set_infinity()
    source.waveform_arm()
    readback = source.waveform_phasemarker_line
    if int(readback) != cfg.phasemarker_line:
        # Leave nothing armed behind a raise — the caller never gets a
        # handle back to shut down otherwise, and this program's whole
        # safety story is "the 6221 is quiet unless we say otherwise".
        source.waveform_abort()
        source.shutdown()
        raise RuntimeError(
            f"6221 phase-marker line rejected: asked for {cfg.phasemarker_line}, "
            f"device reports {readback}. Another Trigger Link function likely "
            "already owns that pin.")
    source.waveform_start()
    log.info("Keithley 6221 connected: %s  AC I=%.4g A peak  f=%.4g Hz  "
              "compliance=%.2f V  phase marker → Trigger Link pin %d",
              cfg.visa_resource, cfg.amplitude_A, cfg.frequency_Hz,
              cfg.compliance_V, cfg.phasemarker_line)
    return source


def shutdown_ac_source(source: Keithley6221) -> None:
    """Stop the AC wave and disable the 6221's output."""
    source.waveform_abort()
    source.shutdown()
    log.info("Keithley 6221 AC wave stopped, output disabled")


@dataclass
class PulseWaveConfig:
    """A single HARDWARE-TIMED current pulse via WAVE mode's square function,
    run for exactly one cycle — the 6221's real, verified, no-2182-required
    single-pulse mechanism. Confirmed against the Model 6220/6221 User's
    Manual (622x-900-01 Rev. C, Section 7 "Wave Functions"):

      * A square wave swings between (offset - amplitude) and
        (offset + amplitude) (manual Fig. 7-1). fire_wave_pulse() below sets
        offset = amplitude = pulse_current_A / 2, so the wave sits at 0 and
        rises to pulse_current_A for the duty-cycle "high" fraction of one
        period, then returns to 0 — a clean unipolar pulse, not the
        alternating-polarity triplet Pulse Delta mode uses.
      * "The output will turn off after the currently set duration period
        has expired" (manual, Section 7 "Duration") — with
        waveform_duration_cycles = 1, that is real hardware-timed
        auto-termination after exactly one pulse, not a software sleep
        guess. fire_wave_pulse() polls the OUTPUT state for this rather than
        trusting elapsed wall time.
      * This is DIFFERENT from Pulse Delta / Differential Conductance
        (`:SOUR:PDEL`, `:SOUR:DCON`), which require a 2182/2182A wired via
        the rear-panel Trigger Link cable and are architecturally built
        around alternating-polarity offset-cancelling measurement, not a
        single deliberate-polarity write pulse. Nothing here touches a 2182.

    Duty cycle is fixed at 50% internally so ``width_s`` alone determines
    the frequency this programs (frequency = 1 / (2 x width_s)) — pick
    ``width_s`` so that frequency stays inside the WAVE subsystem's 1 mHz to
    100 kHz range (width_s below ~5 µs pushes frequency past 100 kHz and the
    instrument will clip it).

    Spec floor: manual quotes "Settable to 1 µs min. pulse duration" for
    square-wave duty cycle, footnoted "minimum realizable duty cycle is
    limited by current range response and load impedance" — the datasheet's
    own headline number is 5 µs. Both are best-case; the true floor at your
    actual pulse amplitude/range is whatever fire_wave_pulse() measures and
    returns as ``pulse_width_measured_s`` — check that, not this docstring,
    before trusting a sub-10 µs pulse.
    """
    pulse_current_A: float = 5e-3    # peak pulse current [A] — pulse rises from 0 to this
    width_s: float          = 1e-3    # pulse width (duty-cycle "high" time) [s]
    compliance_V: float     = 5.0     # pulse voltage compliance [V]
    ranging: str             = "best"  # "best" or "fixed"


def fire_wave_pulse(source: Keithley6221, cfg: PulseWaveConfig,
                     stop_event: Optional[threading.Event] = None,
                     timeout_margin_s: float = 0.5) -> dict:
    """Fire ONE hardware-timed current pulse — see PulseWaveConfig's
    docstring for the mechanism and why it's the 6221's real supported
    single-pulse capability, not Pulse Delta and not a software sleep loop.

    Every WAVE parameter must be set before ``waveform_arm()`` (same
    pymeasure/SCPI ordering rule as connect_ac_source() — a write after
    arming doesn't take effect until the next arm). Disables the phase
    marker for the pulse (it's only meaningful during the continuous AC
    read elsewhere in this codebase) and always sets every parameter
    explicitly — never assumes a retained level from a prior AC-read or
    pulse cycle.

    Completion is detected by polling ``source_enabled`` (``OUTPUT?``)
    rather than trusting elapsed wall time or SRQ (SRQ needs GPIB; this
    driver's configs default to GPIB but nothing stops a TCPIP/LAN
    resource string) — bounded by the pulse's own period plus
    ``timeout_margin_s`` so a stuck poll can't hang a run forever; a
    timeout still returns rather than raising; the caller sees it via a
    ``pulse_width_measured_s`` that doesn't shrink back down. ``waveform_
    abort()`` + ``disable_source()`` run unconditionally afterward so the
    instrument is left in the same disarmed state the caller's next action
    (another pulse, or connect_ac_source()'s continuous read) expects.

    Returns ``{"pulse_width_measured_s": <elapsed to the instrument's own
    auto-off, or the timeout bound>}``.
    """
    half = cfg.pulse_current_A / 2.0
    period_s = 2.0 * cfg.width_s
    frequency_Hz = 1.0 / period_s

    source.source_compliance = cfg.compliance_V
    source.waveform_function = "square"
    source.waveform_amplitude = half
    source.waveform_offset = half
    source.waveform_dutycycle = 50.0
    source.waveform_frequency = frequency_Hz
    source.waveform_ranging = cfg.ranging
    source.waveform_use_phasemarker = False
    source.waveform_duration_cycles = 1
    source.waveform_arm()
    t0 = time.monotonic()
    source.waveform_start()

    timeout_s = period_s + timeout_margin_s
    poll_interval_s = min(0.001, timeout_s)
    while time.monotonic() - t0 < timeout_s:
        if stop_event is not None and stop_event.is_set():
            break
        if not source.source_enabled:
            break
        time.sleep(poll_interval_s)
    elapsed = time.monotonic() - t0

    source.waveform_abort()
    source.disable_source()
    return {"pulse_width_measured_s": elapsed}


def connect(visa_resource: str, compliance_V: float, source_delay_s: float,
            initial_current_A: float = 0.0) -> Keithley6221:
    """Open and configure a Keithley 6221 as a DC current source."""
    source = Keithley6221(visa_resource)
    source.reset()
    source.source_auto_range = True
    source.source_compliance = compliance_V
    source.source_delay = source_delay_s
    source.source_current = initial_current_A
    source.enable_source()
    log.info("Keithley 6221 connected: %s  I=%.4g A  compliance=%.2f V",
              visa_resource, initial_current_A, compliance_V)
    return source


def connect_source(cfg: SourceConfig) -> Keithley6221:
    """connect(), parameterized by a SourceConfig — the fixed sense-current case."""
    return connect(cfg.visa_resource, cfg.compliance_V, cfg.source_delay_s,
                    initial_current_A=cfg.sense_current_A)


def shutdown_source(source: Keithley6221, zero_first: bool = True) -> None:
    """
    Disable the 6221's output.

    `zero_first`, if True (the default), sets the current to 0 A immediately
    before disabling — pass False if the current was already ramped down
    gently (e.g. via ramp_current_to_zero()) beforehand, so it isn't jumped
    a second time.
    """
    if zero_first:
        source.source_current = 0.0
    source.shutdown()
    log.info("Keithley 6221 output disabled")


def ramp_current_to_zero(source: Keithley6221, step_A: float = 1e-4, delay_s: float = 0.02) -> None:
    """Step the sourced current back to 0 A gradually rather than jumping — gentler on the DUT."""
    current = source.source_current
    log.info("Ramping current from %.4g A to 0 A ...", current)
    n_steps = max(1, int(abs(current) / step_A))
    for i in np.linspace(current, 0.0, n_steps + 1)[1:]:
        source.source_current = float(i)
        time.sleep(delay_s)


def acquire_reversal_averaged_voltage(
    source: Keithley6221,
    voltmeter: Keithley2182,
    sense_current_A: float,
    n_reversals: int,
    stop_event: Optional[threading.Event] = None,
    source_delay_s: float = 0.0,
) -> dict:
    """
    Reverse the sense current n_reversals times and decompose the resulting
    voltage into its odd part (V_odd, the resistive signal) and even part
    (V_even, offset plus any genuinely even-in-current physics) — both are
    returned rather than just V_odd; see docs/current-reversal.md for why.

    `source_delay_s`, if given, is slept after each polarity flip before the
    voltmeter is read — a plain property write does not itself wait for the
    reversed current to settle.

    Leaves the source at +sense_current_A on return. If `stop_event` fires
    partway through, returns the mean/sem of whatever pairs were already
    collected. ``mean``/``even_mean`` are the odd/even components; ``sem``/
    ``even_sem`` are the standard error of each of those means (sample
    stdev over the reversal pairs / sqrt(n); ``nan`` if only one pair was
    collected). The raw pair-to-pair scatter is ``sem * sqrt(n_reversals)``.
    """
    samples_odd = np.empty(n_reversals)
    samples_even = np.empty(n_reversals)
    n_used = 0

    for i in range(n_reversals):
        source.source_current = sense_current_A
        if source_delay_s > 0:
            time.sleep(source_delay_s)
        v_plus = voltmeter.voltage

        source.source_current = -sense_current_A
        if source_delay_s > 0:
            time.sleep(source_delay_s)
        v_minus = voltmeter.voltage

        samples_odd[i] = (v_plus - v_minus) / 2.0
        samples_even[i] = (v_plus + v_minus) / 2.0
        n_used = i + 1

        if stop_event is not None and stop_event.is_set():
            break

    source.source_current = sense_current_A

    used_odd = samples_odd[:n_used]
    used_even = samples_even[:n_used]
    if n_used >= 2:
        sqrt_n = np.sqrt(n_used)
        sem_odd = float(np.std(used_odd, ddof=1) / sqrt_n)
        sem_even = float(np.std(used_even, ddof=1) / sqrt_n)
    else:
        sem_odd = sem_even = float("nan")
    return {
        "mean": float(np.mean(used_odd)),
        "sem": sem_odd,
        "even_mean": float(np.mean(used_even)),
        "even_sem": sem_even,
        "n_reversals": n_used,
    }
