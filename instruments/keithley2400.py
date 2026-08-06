"""
Keithley 2400 Gate Voltage Source — shared connect/set/shutdown helpers
==========================================================================
The 2400 driver itself ships with pymeasure (pymeasure.instruments.keithley.
Keithley2400) — this module holds the connect/set/shutdown wrapper functions
that dc_gate_sweep.py, dc_iv_curve.py and dc_spin_valve.py each used to
define an identical copy of.

Usage example:
    from keithley2400 import GateConfig, connect_gate, set_gate_voltage, shutdown_gate

    gate_cfg = GateConfig(visa_resource="GPIB0::24::INSTR", gate_voltage_limit_V=20.0)
    gate = connect_gate(gate_cfg)
    set_gate_voltage(gate, gate_cfg, 5.0)
    ...
    shutdown_gate(gate)
"""

import logging
from dataclasses import dataclass

from pymeasure.instruments.keithley import Keithley2400

log = logging.getLogger(__name__)


@dataclass
class GateConfig:
    """Keithley 2400 — gate voltage source, shared by every DC program with a gate."""
    visa_resource: str          = "GPIB0::25::INSTR"
    gate_voltage_limit_V: float = 20.0    # Software safety ceiling on |gate voltage| [V]
    compliance_current_A: float = 1e-6    # Gate leakage current compliance [A]
    source_delay_s: float       = 0.05


def connect_gate(cfg: GateConfig) -> Keithley2400:
    """Open and configure the Keithley 2400 as the gate voltage source."""
    gate = Keithley2400(cfg.visa_resource)
    gate.reset()
    gate.apply_voltage(compliance_current=cfg.compliance_current_A)
    gate.source_voltage = 0.0
    gate.enable_source()
    log.info(
        "Keithley 2400 gate connected: %s  V_limit=±%.2f V  I_compliance=%.3g A",
        cfg.visa_resource, cfg.gate_voltage_limit_V, cfg.compliance_current_A,
    )
    return gate


def set_gate_voltage(gate: Keithley2400, cfg: GateConfig, voltage_V: float) -> None:
    """Set the gate voltage, refusing to exceed the configured software limit."""
    if abs(voltage_V) > cfg.gate_voltage_limit_V:
        raise ValueError(
            f"Requested gate voltage {voltage_V:.3f} V exceeds configured "
            f"limit ±{cfg.gate_voltage_limit_V:.3f} V — refusing to set it."
        )
    gate.source_voltage = voltage_V


def shutdown_gate(gate: Keithley2400) -> None:
    """Ramp the gate voltage to 0 V and disable the 2400's output."""
    gate.ramp_to_voltage(0.0)
    gate.shutdown()
    log.info("Keithley 2400 gate output disabled")
