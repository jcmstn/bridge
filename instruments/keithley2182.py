"""
Keithley 2182 Nanovoltmeter — shared connect/acquisition helpers
===================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-06

The 2182 driver itself ships with pymeasure (pymeasure.instruments.keithley.
Keithley2182) — this module holds the connect/acquisition wrapper functions
shared by the DC measurement programs.

Usage example:
    from instruments.keithley2182 import VoltmeterConfig, connect_voltmeter, acquire_averaged_voltage

    volt_cfg = VoltmeterConfig(visa_resource="GPIB0::7::INSTR", nplc=5)
    voltmeter = connect_voltmeter(volt_cfg)
    v = acquire_averaged_voltage(voltmeter, n_averages=5)   # {"mean": ..., "sem": ...}
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
    """Keithley 2182 — differential voltage readout. Channel 1 is the DUT
    input every DC program uses; set ``channel=2`` for the second input
    (e.g. a reference or a thermocouple) in a custom script."""
    visa_resource: str = "GPIB0::7::INSTR"
    nplc: float        = 5      # Integration time [power line cycles]
    auto_range: bool   = True
    channel: int       = 1      # 2182 input channel (1 or 2)


def connect_voltmeter(cfg: VoltmeterConfig) -> Keithley2182:
    """Open and configure the Keithley 2182 for a differential voltage readout."""
    if cfg.channel not in (1, 2):
        raise ValueError(f"Keithley 2182 channel must be 1 or 2, got {cfg.channel}")
    voltmeter = Keithley2182(cfg.visa_resource)
    voltmeter.reset()
    getattr(voltmeter, f"ch_{cfg.channel}").setup_voltage(auto_range=cfg.auto_range, nplc=cfg.nplc)
    log.info("Keithley 2182 connected: %s  ch=%d  NPLC=%.1f",
             cfg.visa_resource, cfg.channel, cfg.nplc)
    return voltmeter


def acquire_averaged_voltage(
    voltmeter: Keithley2182,
    n_averages: int,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """
    Read `n_averages` voltage samples off the 2182 (current held fixed,
    unlike acquire_reversal_averaged_voltage) and return the mean and the
    standard error of that mean (``sem`` = sample stdev / sqrt(n); the raw
    sample scatter is ``sem * sqrt(n)`` if it's ever wanted).

    `stop_event`, if given, is checked between samples — set it to break
    out early and return the mean/sem of whatever was already collected.
    ``sem`` is ``nan`` if only a single sample was taken (no scatter to
    estimate an uncertainty from).
    """
    samples = np.empty(n_averages)
    n_used = 0
    for i in range(n_averages):
        samples[i] = voltmeter.voltage
        n_used = i + 1
        if stop_event is not None and stop_event.is_set():
            break
    used = samples[:n_used]
    sem = float(np.std(used, ddof=1) / np.sqrt(n_used)) if n_used >= 2 else float("nan")
    return {"mean": float(np.mean(used)), "sem": sem}
