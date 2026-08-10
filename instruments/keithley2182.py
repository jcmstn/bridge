"""
Keithley 2182 Nanovoltmeter — shared connect/acquisition helpers
===================================================================
The 2182 driver itself ships with pymeasure (pymeasure.instruments.keithley.
Keithley2182) — this module holds the connect/acquisition wrapper functions
that dc_hall_measurement.py, dc_gate_sweep.py, dc_iv_curve.py and
dc_spin_valve.py each used to define an identical copy of.

Usage example:
    from keithley2182 import VoltmeterConfig, connect_voltmeter, acquire_averaged_voltage

    volt_cfg = VoltmeterConfig(visa_resource="GPIB0::7::INSTR", nplc=5)
    voltmeter = connect_voltmeter(volt_cfg)
    v = acquire_averaged_voltage(voltmeter, n_averages=5)   # {"mean": ..., "std": ...}
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymeasure.instruments.keithley import Keithley2182

log = logging.getLogger(__name__)


@dataclass
class VoltmeterConfig:
    """Keithley 2182 — voltage readout across the DUT (channel 1, differential)."""
    visa_resource: str = "GPIB0::7::INSTR"
    nplc: float        = 5      # Integration time [power line cycles]
    auto_range: bool   = True


def connect_voltmeter(cfg: VoltmeterConfig) -> Keithley2182:
    """Open and configure the Keithley 2182 for a differential voltage readout."""
    voltmeter = Keithley2182(cfg.visa_resource)
    voltmeter.reset()
    voltmeter.ch_1.setup_voltage(auto_range=cfg.auto_range, nplc=cfg.nplc)
    log.info("Keithley 2182 connected: %s  NPLC=%.1f", cfg.visa_resource, cfg.nplc)
    return voltmeter


def acquire_averaged_voltage(
    voltmeter: Keithley2182,
    n_averages: int,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """
    Read `n_averages` voltage samples off the 2182 (current held fixed,
    unlike acquire_reversal_averaged_voltage) and return mean/std.

    `stop_event`, if given, is checked between samples — set it to break
    out early and return the mean/std of whatever was already collected
    (at least one sample).
    """
    samples = np.empty(n_averages)
    n_used = 0
    for i in range(n_averages):
        samples[i] = voltmeter.voltage
        n_used = i + 1
        if stop_event is not None and stop_event.is_set():
            break
    used = samples[:n_used]
    return {"mean": float(np.mean(used)), "std": float(np.std(used))}
