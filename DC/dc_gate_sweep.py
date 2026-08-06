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
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

from pymeasure.instruments.keithley import Keithley2182, Keithley2400, Keithley6221

# The instrument connect/shutdown helpers live in the shared bridge/instruments
# folder — add it to sys.path directly (it's not installed as a normal package).
_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent / "instruments"
if str(_INSTRUMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTRUMENTS_DIR))
from keithley6221 import SourceConfig, connect_source, shutdown_source  # noqa: E402
from keithley2182 import (  # noqa: E402
    VoltmeterConfig,
    connect_voltmeter,
    acquire_averaged_voltage,
)
from keithley2400 import (  # noqa: E402
    GateConfig,
    connect_gate,
    set_gate_voltage,
    shutdown_gate,
)
from kepco_magnet import (  # noqa: E402
    MagnetConfig,
    connect_magnet,
    set_magnet_current,
    shutdown_magnet,
)
from lakeshore475 import (  # noqa: E402
    GaussmeterConfig,
    connect_gaussmeter,
    read_field_mT,
    shutdown_gaussmeter,
)
from mercury_itc import (  # noqa: E402
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    shutdown_temperature_controller,
)

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
# SourceConfig, VoltmeterConfig, GateConfig, MagnetConfig and GaussmeterConfig
# are the same shape as every other DC program's — they live in
# bridge/instruments/ (see the keithley6221 / keithley2182 / keithley2400 /
# kepco_magnet / lakeshore475 imports above) instead of being redefined here.
# Only what's specific to this sweep (the gate points and acquisition timing)
# is defined below.

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
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg:  Optional[TemperatureControllerConfig] = None,
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

    `temp_ctrl`/`temp_cfg`, if given, log the sample/probe temperature
    (temperature_1_K / temperature_2_K) at each point via the shared
    MercuryiTC controller (see mercury_itc.py). Unlike the magnet field,
    temperature is re-read every point (not just parked once) since it
    can drift over the course of a sweep. Passing `temp_ctrl=None` (e.g.
    because the MercuryiTC isn't connected) simply leaves those columns
    empty — it's never a reason to stop the measurement.
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

        # ── 3b. Read temperature (MercuryiTC, optional) ─────────────────────
        temp_1_K, temp_2_K = read_temperature(temp_ctrl, temp_cfg) \
            if temp_cfg is not None else (None, None)

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
            "temperature_1_K": temp_1_K,
            "temperature_2_K": temp_2_K,
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

    # ── Temperature (Oxford Instruments MercuryiTC, optional) ────────────────
    # Not every rig has one, and not every MercuryiTC has two probes wired up
    # — connect_temperature_controller() returns None rather than raising if
    # it can't be reached, and the measurement runs fine either way.
    temp_cfg = TemperatureControllerConfig(
        visa_resource = "TCPIP0::192.168.1.5::7020::SOCKET",  # ← set to your iTC's address
        sensor_uids   = ("DB6.T1",),   # ← 1 or 2 board UIDs, e.g. ("DB6.T1", "DB5.T1")
    )
    temp_ctrl = connect_temperature_controller(temp_cfg)

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
        df = run_measurement(source, voltmeter, gate, src_cfg, gate_cfg, acq_cfg, points,
                              temp_ctrl=temp_ctrl, temp_cfg=temp_cfg)
        print("\n", df.to_string(index=False))
    finally:
        shutdown_gate(gate)
        shutdown_source(source)
        shutdown_temperature_controller(temp_ctrl)


if __name__ == "__main__":
    main()
