"""
Keithley 2400 SourceMeter — gate-source helpers + general SMU helpers
==========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-06

The 2400 driver ships with pymeasure (``pymeasure.instruments.keithley.
Keithley2400``). This module adds two wrapper surfaces on top of it:

* ``GateConfig`` + ``connect_gate`` / ``set_gate_voltage`` / ``shutdown_gate``
  — the narrow "2400 as a gate voltage source" case every DC program uses.
  Unchanged; import these for anything that already worked.

* ``SMUConfig`` + ``connect_smu`` / ``set_source_level`` / ``read_measurement``
  / ``acquire_measurement`` / ``measure_buffered`` / ``shutdown_smu`` — the
  general driver-contract layer (docs/architecture.md §4): source V or I,
  measure V/I/R, 2/4-wire, ranges, NPLC, front/rear terminals, buffered
  statistics. Parallel to ``instruments.keithley2450.SMUConfig`` — a custom
  script that needs the 2400 as a full SMU reaches for this.

Failure policy: sourcing into a device is **load-bearing** — ``connect_gate``
/ ``connect_smu`` / ``set_gate_voltage`` / ``set_source_level`` raise on any
failure.

Usage example (gate):
    from instruments.keithley2400 import GateConfig, connect_gate, set_gate_voltage, shutdown_gate

    gate_cfg = GateConfig(visa_resource="GPIB0::24::INSTR", gate_voltage_limit_V=20.0)
    gate = connect_gate(gate_cfg)
    set_gate_voltage(gate, gate_cfg, 5.0)
    ...
    shutdown_gate(gate)

Usage example (general SMU — source voltage, measure current, 4-wire):
    from instruments.keithley2400 import (
        SMUConfig, connect_smu, set_source_level, acquire_measurement, shutdown_smu)

    cfg = SMUConfig(visa_resource="GPIB0::24::INSTR", source_function="voltage",
                    compliance_current_A=1e-6, four_wire=True, source_limit_V=10.0)
    smu = connect_smu(cfg)
    try:
        set_source_level(smu, cfg, 1.0)
        out = acquire_measurement(smu, cfg, n=10)   # {"mean": A, "sem": A}
    finally:
        shutdown_smu(smu)
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
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


# ─────────────────────────────────────────────────────────────────────────────
# General SMU wrapper  ── the driver-contract layer (docs/architecture.md §4) ──
# ─────────────────────────────────────────────────────────────────────────────
# Mirrors instruments.keithley2450.SMUConfig so a custom script can swap the
# two SMUs with only an import change. The bodies differ where the SCPI does:
# the 2400's `wires` writes :SYSTEM:RSENSE (global 4-wire, correct here); the
# 2400 buffer is pymeasure's KeithleyBuffer, not the 2450's native one.

_SMU_FUNCS = ("voltage", "current")
_SENSE_FUNCS = ("voltage", "current", "resistance")
_OFF_STATES = {"himp": "HIMP", "normal": "NORM", "zero": "ZERO", "guard": "GUAR"}


@dataclass
class SMUConfig:
    """Keithley 2400 as a general-purpose SMU. Every field has a safe default;
    override only what your measurement needs.

    ``source_range`` / ``sense_range`` are in the unit of their function
    (V for a ``"voltage"`` function, A for ``"current"``, Ω for a
    ``"resistance"`` sense); ``None`` means autorange.
    """
    visa_resource: str        = "GPIB0::24::INSTR"
    source_function: str      = "voltage"     # "voltage" | "current" — what the SMU drives
    sense_function: Optional[str] = None      # "voltage"|"current"|"resistance"; None → the other of source_function
    compliance_current_A: float = 1e-3        # limit while sourcing voltage [A]
    compliance_voltage_V: float = 10.0        # limit while sourcing current [V]
    source_range: Optional[float] = None      # None → autorange the source
    sense_range: Optional[float]  = None      # None → autorange the measurement
    nplc: float               = 1.0           # integration time [power-line cycles], 0.01–10
    four_wire: bool           = False         # True → remote (4-wire) sense (:SYSTEM:RSENSE)
    terminals: str            = "front"       # "front" | "rear"
    source_delay_s: Optional[float] = None    # None → the 2400's own auto source delay
    auto_zero: bool           = True          # periodic internal auto-zero
    output_off_state: str     = "himp"        # state when the output is disabled; "himp" = relay open (safe)
    source_limit_V: float     = 21.0          # set_source_level() refuses |V| beyond this — raise per device
    source_limit_A: float     = 1e-3          # set_source_level() refuses |I| beyond this — raise per device


def _resolved_sense(cfg: SMUConfig) -> str:
    if cfg.sense_function is not None:
        return cfg.sense_function
    return "current" if cfg.source_function == "voltage" else "voltage"


def connect_smu(cfg: SMUConfig) -> Keithley2400:
    """Open the VISA session, reset, apply every field of ``cfg``, enable the
    source, and return the live handle. Raises on any failure (load-bearing)."""
    if cfg.source_function not in _SMU_FUNCS:
        raise ValueError(f"source_function must be one of {_SMU_FUNCS}, got {cfg.source_function!r}")
    sense = _resolved_sense(cfg)
    if sense not in _SENSE_FUNCS:
        raise ValueError(f"sense_function must be one of {_SENSE_FUNCS}, got {sense!r}")
    if cfg.output_off_state.lower() not in _OFF_STATES:
        raise ValueError(f"output_off_state must be one of {tuple(_OFF_STATES)}, got {cfg.output_off_state!r}")

    smu = Keithley2400(cfg.visa_resource)
    smu.reset()

    if cfg.source_function == "voltage":
        smu.apply_voltage(voltage_range=cfg.source_range,
                          compliance_current=cfg.compliance_current_A)
    else:
        smu.apply_current(current_range=cfg.source_range,
                          compliance_voltage=cfg.compliance_voltage_V)

    measure = getattr(smu, f"measure_{sense}")
    if cfg.sense_range is None:
        measure(nplc=cfg.nplc, auto_range=True)
    else:
        measure(nplc=cfg.nplc, auto_range=False, **{sense: cfg.sense_range})

    # On the 2400, `wires` writes :SYSTEM:RSENSE — the global remote-sense
    # enable, correct for any source/measure pairing (not resistance-only).
    smu.wires = 4 if cfg.four_wire else 2

    smu.use_rear_terminals() if cfg.terminals == "rear" else smu.use_front_terminals()

    if cfg.source_delay_s is not None:
        smu.source_delay = cfg.source_delay_s
    smu.auto_zero = cfg.auto_zero
    smu.output_off_state = _OFF_STATES[cfg.output_off_state.lower()]

    smu.enable_source()
    log.info(
        "Keithley 2400 SMU connected: %s  source=%s  sense=%s  %s  NPLC=%.3g",
        cfg.visa_resource, cfg.source_function, sense,
        "4-wire" if cfg.four_wire else "2-wire", cfg.nplc,
    )
    return smu


def set_source_level(smu: Keithley2400, cfg: SMUConfig, level: float) -> None:
    """Set the source setpoint (V or A per ``cfg.source_function``), refusing
    to exceed the configured software limit — mirrors ``set_gate_voltage``."""
    if cfg.source_function == "voltage":
        if abs(level) > cfg.source_limit_V:
            raise ValueError(
                f"Requested source voltage {level:.4g} V exceeds source_limit_V "
                f"±{cfg.source_limit_V:.4g} V — refusing to set it."
            )
        smu.source_voltage = level
    else:
        if abs(level) > cfg.source_limit_A:
            raise ValueError(
                f"Requested source current {level:.4g} A exceeds source_limit_A "
                f"±{cfg.source_limit_A:.4g} A — refusing to set it."
            )
        smu.source_current = level


def read_measurement(smu: Keithley2400, cfg: SMUConfig) -> float:
    """One fresh reading of the sense quantity, in canonical units (V/A/Ω)."""
    return float(getattr(smu, _resolved_sense(cfg)))


def acquire_measurement(
    smu: Keithley2400,
    cfg: SMUConfig,
    n: int,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Average ``n`` fresh readings of the sense quantity (slow Python loop,
    one ``:READ?`` per sample). Returns ``{"mean", "sem"}`` where ``sem`` is
    the sample stdev / sqrt(n) (``nan`` for n == 1) — same shape as
    ``keithley2182.acquire_averaged_voltage``. ``stop_event`` is checked
    between samples so a UI abort can cut a long average short."""
    sense = _resolved_sense(cfg)
    samples = np.empty(n)
    n_used = 0
    for i in range(n):
        samples[i] = float(getattr(smu, sense))
        n_used = i + 1
        if stop_event is not None and stop_event.is_set():
            break
    used = samples[:n_used]
    sem = float(np.std(used, ddof=1) / np.sqrt(n_used)) if n_used >= 2 else float("nan")
    return {"mean": float(np.mean(used)), "sem": sem}


def measure_buffered(smu: Keithley2400, n: int, timeout_s: float = 30.0) -> dict:
    """Take ``n`` samples into the instrument buffer and return
    ``{"mean", "std", "n"}`` computed from the stored points — faster than
    :func:`acquire_measurement` for large ``n``. ``n`` reflects the points
    actually stored.

    Restores single-shot triggering on the way out — ``config_buffer`` sets
    ``trigger_count = n``, which would otherwise make every later bare
    ``:READ?`` return an ``n``-point sweep."""
    try:
        smu.config_buffer(max(2, n))
        smu.start_buffer()
        smu.wait_for_buffer(timeout=timeout_s)
        data = smu.buffer_data
        std = float(np.std(data, ddof=1)) if data.size >= 2 else 0.0
        return {"mean": float(np.mean(data)) if data.size else float("nan"),
                "std": std, "n": int(data.size)}
    finally:
        smu.disable_buffer()
        smu.trigger_count = 1


def shutdown_smu(smu: Keithley2400) -> None:
    """Ramp the source to zero, disable the output, close the session."""
    smu.shutdown()
    log.info("Keithley 2400 SMU output disabled")
