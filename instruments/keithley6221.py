"""
Keithley 6221 DC Current Source — shared connect/shutdown/acquisition helpers
===============================================================================
The 6221 driver itself ships with pymeasure (pymeasure.instruments.keithley.
Keithley6221) — this module holds the connect/shutdown/acquisition wrapper
functions that dc_hall_measurement.py, dc_gate_sweep.py, dc_spin_valve.py and
dc_iv_curve.py each used to define an identical (or near-identical) copy of,
the same way kepco_magnet.py and lakeshore475.py share their own
connect/read/shutdown helpers instead of leaving them duplicated per-script.

Usage example (fixed sense-current case — dc_hall_measurement.py,
dc_gate_sweep.py, dc_spin_valve.py):
    from keithley6221 import SourceConfig, connect_source, shutdown_source

    src_cfg = SourceConfig(visa_resource="GPIB0::12::INSTR", sense_current_A=1e-3)
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
    voltage into its odd and even parts in I:

        V_odd  = (V(+I) - V(-I)) / 2   — the resistive signal (R = V_odd/I);
                                          cancels any offset common to both
                                          polarities (thermal EMFs at the
                                          contacts, amplifier offset, etc.),
                                          since a true offset doesn't flip
                                          sign with the current but R does
        V_even = (V(+I) + V(-I)) / 2   — everything that DOES share the
                                          offset's sign symmetry: a real
                                          instrumental offset, but also any
                                          genuine even-in-I physics (e.g.
                                          unidirectional SMR, I^2 Joule
                                          heating) that V_odd alone would
                                          silently discard

    Both are returned (and both get logged to the CSV by the caller) so
    V_even can be checked after the fact instead of being thrown away.

    `source_delay_s`, if given, is slept after each polarity flip before the
    voltmeter is read — a plain property write (`source.source_current =
    ...`) does not itself wait for `source_delay_s` (that instrument
    register only governs the 6221's own triggered/delta-sweep timing, not
    a bare write), so without this the voltmeter can be read before the
    reversed current has actually settled, especially noticeable as
    n_reversals grows and many flips happen back-to-back.

    Leaves the source at +sense_current_A on return. If `stop_event` fires
    partway through, returns the mean/std of whatever pairs were already
    collected (at least one).
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
    return {
        "mean": float(np.mean(used_odd)),
        "std": float(np.std(used_odd)),
        "even_mean": float(np.mean(used_even)),
        "even_std": float(np.std(used_even)),
        "n_reversals": n_used,
    }
