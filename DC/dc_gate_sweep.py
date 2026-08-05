#!/usr/bin/env python3
"""
DC Gate Sweep — Keithley 2400 (gate) + 6221 (fixed sense current) + 2182
==========================================================================
Companion to dc_iv_curve.py / dc_hall_measurement.py — same house style
(dataclass configs, incremental CSV, stop_event/on_point hooks for a live
UI). Where dc_iv_curve.py sweeps current at a fixed gate, this program
sweeps the gate voltage at a fixed sense current — the standard
transfer-curve measurement for a gated device.

Wiring
------
    Keithley 6221 (current source)
      Output (current) ──▶ DUT ── common ground

    Keithley 2182 (nanovoltmeter)
      Channel 1 (differential) ──▶ across the DUT (2-terminal or
      4-terminal / Kelvin)

    Keithley 2400 (gate source)
      Output (voltage) ──▶ gate electrode

    Magnet field  (optional — see MagnetConfig/GaussmeterConfig)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample

Method
------
For each gate voltage point:
  1. Set the 2400's gate voltage.
  2. Settle (acq_cfg.settling_time_s — gated 2D systems can be slow to
     re-equilibrate after a gate step).
  3. Read `n_averages` voltage samples from the 2182 and average them.
  4. Chord resistance R = V / I_sense is recorded alongside V and Vg.

If a magnet current is given (optional — see GatePlan in the TUI), it is
parked once before the gate sweep starts (not swept) and the resulting
field is measured live via the Lake Shore 475 and recorded on every row
for reference — exactly the "field measured live, never computed from
current" convention used throughout bridge/DC.

Requirements:
    pip install pymeasure pyvisa numpy pandas
"""

import sys
import time
import logging
import threading
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

import pyvisa
from pymeasure.instruments.keithley import Keithley2182, Keithley2400, Keithley6221

# The Kepco magnet and Lake Shore 475 drivers live in the shared
# bridge/instruments folder — add it to sys.path directly (it's not
# installed as a normal package).
_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent / "instruments"
if str(_INSTRUMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTRUMENTS_DIR))
from kepco_magnet import KepkoBOPGL  # noqa: E402
from lakeshore475 import LakeShore475  # noqa: E402

from dc_sweep_utils import linear_sweep  # noqa: E402

# Data lives outside "bridge" (a sibling of it) so measurement output never
# ends up inside the git-tracked source tree.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses  ── change all your parameters here ──────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceConfig:
    """Keithley 6221 — fixed DC sense current (not swept in this program)."""
    visa_resource: str     = "GPIB0::12::INSTR"
    sense_current_A: float = 1e-6     # Fixed sense current [A]
    compliance_V: float    = 2.0      # Voltage compliance [V]
    source_delay_s: float  = 0.05     # Settle time after a current step [s]


@dataclass
class VoltmeterConfig:
    """Keithley 2182 — DUT voltage readout (channel 1, differential)."""
    visa_resource: str = "GPIB0::7::INSTR"
    nplc: float        = 5      # Integration time [power line cycles]
    auto_range: bool   = True


@dataclass
class GateConfig:
    """Keithley 2400 — gate voltage, the swept axis of this program."""
    visa_resource: str          = "GPIB0::24::INSTR"
    gate_voltage_limit_V: float = 20.0    # Software safety ceiling on |gate voltage| [V]
    compliance_current_A: float = 1e-6    # Gate leakage current compliance [A]
    source_delay_s: float       = 0.05


@dataclass
class MagnetConfig:
    """
    Kepco BOP-GL bipolar power supply, used as a current source for an
    electromagnet. Identical role/shape to MagnetConfig in
    dc_hall_measurement.py, but here the field is parked at a fixed value
    (or a list of values, each run separately) rather than swept.
    """
    visa_resource:        str   = "GPIB0::6::INSTR"
    current_limit_A:      float = 50.0    # Software current limit  [A]
    voltage_compliance_V: float = 20.0    # CC-mode compliance / OVP limit  [V]
    ramp_step_A:          float = 0.1     # Ramp step size  [A]
    ramp_delay_s:         float = 0.05    # Delay between ramp steps  [s]


@dataclass
class GaussmeterConfig:
    """Lake Shore 475 DSP Gaussmeter — measures the actual field once the
    magnet is parked."""
    visa_resource: str   = "GPIB0::12::INSTR"
    unit:          str   = "T"     # 'T' or 'G' — read_field_mT() only knows these two
    n_averages:    int   = 10      # Field readings averaged
    read_delay_s:  float = 0.05    # Delay between successive readings  [s]


@dataclass
class AcquisitionConfig:
    """Timing and averaging parameters."""
    settling_time_s: float = 0.2      # Dead-time after a gate voltage step [s]
    n_averages: int        = 5        # Voltage samples averaged per point
    output_file: str       = "dc_gate_sweep.csv"


@dataclass
class GatePoint:
    """One point in the gate voltage sweep."""
    gate_voltage_V: float
    settling_override_s: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Instrument setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def connect_source(cfg: SourceConfig) -> Keithley6221:
    """Open and configure the Keithley 6221 as a fixed DC current source."""
    source = Keithley6221(cfg.visa_resource)
    source.reset()
    source.source_auto_range = True
    source.source_compliance = cfg.compliance_V
    source.source_delay = cfg.source_delay_s
    source.source_current = cfg.sense_current_A
    source.enable_source()
    log.info("Keithley 6221 connected: %s  I_sense=%.4g A  compliance=%.2f V",
              cfg.visa_resource, cfg.sense_current_A, cfg.compliance_V)
    return source


def connect_voltmeter(cfg: VoltmeterConfig) -> Keithley2182:
    """Open and configure the Keithley 2182 for a DUT voltage readout."""
    voltmeter = Keithley2182(cfg.visa_resource)
    voltmeter.reset()
    voltmeter.ch_1.setup_voltage(auto_range=cfg.auto_range, nplc=cfg.nplc)
    log.info("Keithley 2182 connected: %s  NPLC=%.1f", cfg.visa_resource, cfg.nplc)
    return voltmeter


def shutdown_source(source: Keithley6221) -> None:
    """Disable the 6221's output. Call after ramping the sense current to 0."""
    source.source_current = 0.0
    source.shutdown()
    log.info("Keithley 6221 output disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Gate control (Keithley 2400)
# ─────────────────────────────────────────────────────────────────────────────

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
# Magnet control (Kepco BOP-GL current source, via VISA/PyVISA)
# ─────────────────────────────────────────────────────────────────────────────
# Identical to dc_hall_measurement.py's magnet helpers — duplicated here so
# this module stays self-contained, matching the existing house convention.

def connect_magnet(cfg: MagnetConfig) -> KepkoBOPGL:
    """Open a VISA session to the Kepco BOP-GL and arm it as a current source."""
    rm = pyvisa.ResourceManager()
    psu = KepkoBOPGL(rm.open_resource(cfg.visa_resource))

    psu.reset()
    psu.clear_status()
    psu.raise_range_limits_to_max()
    psu.mode = "current"
    psu.voltage_limit = cfg.voltage_compliance_V
    psu.current_limit = cfg.current_limit_A
    psu.current = 0.0
    psu.enable_output()

    log.info(
        "Magnet connected: %s  mode=CC  compliance=%.2f V  I_limit=±%.2f A",
        cfg.visa_resource, cfg.voltage_compliance_V, cfg.current_limit_A,
    )
    return psu


def set_magnet_current(psu: KepkoBOPGL, cfg: MagnetConfig, current_A: float) -> None:
    """Ramp the magnet current to `current_A`, enforcing the software limit in `cfg`."""
    if abs(current_A) > cfg.current_limit_A:
        raise ValueError(
            f"Requested current {current_A:.3f} A exceeds configured "
            f"limit ±{cfg.current_limit_A:.3f} A"
        )
    psu.ramp_current(current_A, step=cfg.ramp_step_A, delay=cfg.ramp_delay_s)


def shutdown_magnet(psu: KepkoBOPGL, cfg: MagnetConfig) -> None:
    """Ramp the magnet current safely to zero, disable the output, and close the VISA session."""
    log.info("Ramping magnet to zero and disabling output ...")
    psu.zero_output(ramp=True, step=cfg.ramp_step_A, delay=cfg.ramp_delay_s)
    psu.close()


# ─────────────────────────────────────────────────────────────────────────────
# Gaussmeter (Lake Shore 475, via pymeasure/VISA)
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_TO_MT = {"T": 1e3, "G": 1e-1}   # → mT, for GaussmeterConfig.unit


def connect_gaussmeter(cfg: GaussmeterConfig) -> LakeShore475:
    """Open a VISA session to the Lake Shore 475 and set its display unit."""
    if cfg.unit not in _FIELD_TO_MT:
        raise ValueError(f"Unsupported gaussmeter unit {cfg.unit!r}; use 'T' or 'G'.")
    gm = LakeShore475(cfg.visa_resource)
    gm.unit = cfg.unit
    log.info("Gaussmeter connected: %s  unit=%s  id=%s",
              cfg.visa_resource, cfg.unit, gm.identification)
    return gm


def read_field_mT(gm: LakeShore475, cfg: GaussmeterConfig) -> float:
    """Average `cfg.n_averages` field readings and return the result in mT."""
    mean, _std = gm.measure(cfg.n_averages, delay=cfg.read_delay_s)
    return mean * _FIELD_TO_MT[cfg.unit]


def shutdown_gaussmeter(gm: LakeShore475) -> None:
    """Close the VISA session to the gaussmeter."""
    gm.close()
    log.info("Gaussmeter connection closed")


# ─────────────────────────────────────────────────────────────────────────────
# Data acquisition
# ─────────────────────────────────────────────────────────────────────────────

def acquire_averaged_voltage(voltmeter: Keithley2182, n_averages: int) -> dict:
    """Read `n_averages` voltage samples off the 2182 and return mean/std."""
    samples = np.empty(n_averages)
    for i in range(n_averages):
        samples[i] = voltmeter.voltage
    return {"mean": float(np.mean(samples)), "std": float(np.std(samples))}


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    source:     Keithley6221,
    voltmeter:  Keithley2182,
    gate:       Keithley2400,
    src_cfg:    SourceConfig,
    gate_cfg:   GateConfig,
    acq_cfg:    AcquisitionConfig,
    points:     List[GatePoint],
    stop_event: Optional[threading.Event] = None,
    on_point:   Optional[Callable[[dict], None]] = None,
    magnet_current_A: Optional[float] = None,
    magnet_field_mT:  Optional[float] = None,
) -> pd.DataFrame:
    """
    Iterate over `points`, set each gate voltage, acquire the averaged
    voltage, log to CSV.

    `stop_event`, if given, is checked before each point — set it to break
    out of the sweep early while still returning the data collected so
    far, so callers can run their normal ramp-down/shutdown path instead
    of killing the process outright.

    `on_point`, if given, is called with each point's `record` dict right
    after it's appended — lets a caller (e.g. a live TUI) show progress
    without polling the output CSV.

    `magnet_current_A`/`magnet_field_mT`, if given, are recorded on every
    point — the magnet is parked once by the caller before the sweep
    starts, not swept per point.
    """
    records: List[dict] = []

    for idx, pt in enumerate(points):
        if stop_event is not None and stop_event.is_set():
            log.info("Measurement aborted after %d / %d points.", idx, len(points))
            break

        log.info("── Point %d / %d   Vg=%.4g V ──────────────────", idx + 1, len(points), pt.gate_voltage_V)

        # ── 1. Apply gate voltage ────────────────────────────────────────────
        set_gate_voltage(gate, gate_cfg, pt.gate_voltage_V)

        # ── 2. Settle ────────────────────────────────────────────────────────
        settle = pt.settling_override_s if pt.settling_override_s is not None \
                 else acq_cfg.settling_time_s
        if settle > 0:
            time.sleep(settle)

        # ── 3. Acquire voltage ───────────────────────────────────────────────
        v = acquire_averaged_voltage(voltmeter, acq_cfg.n_averages)
        r_chord = v["mean"] / src_cfg.sense_current_A if src_cfg.sense_current_A != 0 else float("nan")
        log.info("   V=%.4e V  σ=%.2e V  R=%.5g Ω", v["mean"], v["std"], r_chord)

        # ── 4. Build record ────────────────────────────────────────────────────
        record = {
            "point_index":     idx,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gate_voltage_V":  pt.gate_voltage_V,
            "sense_current_A": src_cfg.sense_current_A,
            "voltage_V":       v["mean"],
            "voltage_std_V":   v["std"],
            "resistance_ohm":  r_chord,
            "magnet_current_A": magnet_current_A,
            "magnet_field_mT":  magnet_field_mT,
        }
        records.append(record)
        if on_point is not None:
            on_point(record)

        # ── 5. Write incrementally (never lose data on a crash) ────────────
        Path(acq_cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(acq_cfg.output_file, index=False)

    log.info("Measurement complete. %d points saved to '%s'.", len(records), acq_cfg.output_file)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── configure your devices and sweep here ──────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Source & voltmeter ───────────────────────────────────────────────────
    src_cfg = SourceConfig(
        visa_resource   = "GPIB0::12::INSTR",
        sense_current_A = 1e-6,
        compliance_V    = 2.0,
        source_delay_s  = 0.05,
    )
    source = connect_source(src_cfg)

    volt_cfg = VoltmeterConfig(
        visa_resource = "GPIB0::7::INSTR",
        nplc          = 5,
        auto_range    = True,
    )
    voltmeter = connect_voltmeter(volt_cfg)

    # ── Gate (Keithley 2400) ──────────────────────────────────────────────────
    gate_cfg = GateConfig(
        visa_resource        = "GPIB0::24::INSTR",
        gate_voltage_limit_V = 20.0,
        compliance_current_A = 1e-6,
    )
    gate = connect_gate(gate_cfg)

    # ── Acquisition settings ─────────────────────────────────────────────────
    acq_cfg = AcquisitionConfig(
        settling_time_s = 0.2,
        n_averages      = 5,
        output_file     = str(_DATA_DIR / f"dc_gate_sweep_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    # ── Gate voltage sweep points ─────────────────────────────────────────────
    gate_voltages_V = linear_sweep(start=-10.0, stop=10.0, step=0.5, bidirectional=True)
    points = [GatePoint(gate_voltage_V=float(v)) for v in gate_voltages_V]

    # ── Run (no field parked in this example — see the TUI for the optional
    #    single-value-or-list magnet current) ─────────────────────────────────
    try:
        df = run_measurement(source, voltmeter, gate, src_cfg, gate_cfg, acq_cfg, points)
        print("\n", df.to_string(index=False))
    finally:
        shutdown_gate(gate)
        shutdown_source(source)


if __name__ == "__main__":
    main()
