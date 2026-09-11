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

    `phasemarker_line` is the Trigger Link output pin the marker square wave
    (one edge per excitation cycle) appears on — wire that BNC to the
    lock-in's Aux Input. Pass a line other than the 6221's factory default if
    that pin is already claimed by something else on the rig; confirm the
    factory default against your own 6221 (front panel CONFIG →
    Trigger Link) before assuming which one that is."""
    visa_resource: str        = "GPIB0::20::INSTR"
    amplitude_A: float        = 1e-4     # AC current amplitude, peak [A]
    frequency_Hz: float       = 977.0    # Excitation frequency [Hz] — avoid 50/60 Hz harmonics
    compliance_V: float       = 2.0      # Voltage compliance [V]
    ranging: str              = "best"   # "best" or "fixed"
    phasemarker_line: int     = 1        # Trigger Link pin (1-6) the phase marker appears on


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
    sem_odd = float(np.std(used_odd, ddof=1) / np.sqrt(n_used)) if n_used >= 2 else float("nan")
    sem_even = float(np.std(used_even, ddof=1) / np.sqrt(n_used)) if n_used >= 2 else float("nan")
    return {
        "mean": float(np.mean(used_odd)),
        "sem": sem_odd,
        "even_mean": float(np.mean(used_even)),
        "even_sem": sem_even,
        "n_reversals": n_used,
    }
