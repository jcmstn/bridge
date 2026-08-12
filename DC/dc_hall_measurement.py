#!/usr/bin/env python3
"""
DC Hall Voltage Measurement — Keithley 6221 + 2182, with field sweep
=====================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-04

Wiring
------
    Keithley 6221 (current source)
      Output (current) ──▶ sample ── common ground

    Keithley 2182 (nanovoltmeter)
      Channel 1 (differential) ──▶ across the transverse (Hall) voltage
      leads of the sample

Method
------
Sources a fixed DC sense current with the 6221 and reads the transverse
(Hall) voltage with the 2182, reversing the current (+I / -I) at each field
point and decomposing the voltage into odd (the reported Hall voltage) and
even parts — see docs/current-reversal.md for why both are recorded
(columns hall_voltage_even_V / hall_voltage_even_std_V).

Magnetic field sweep
--------------------
A Kepco BOP-GL bipolar power supply (see kepco_magnet.KepkoBOPGL) drives
current through an electromagnet to provide the field axis:
bidirectional_current_sweep() builds an up-then-down magnet-current list so
hysteresis is visible, and the actual field at the sample is measured
directly with a Lake Shore 475 DSP Gaussmeter (see lakeshore475.LakeShore475)
at each point rather than inferred from the magnet current via a
calibration constant.

Extensibility
-------------
Add new sweep variables to FieldPoint and a corresponding set_action
callable — the run_measurement loop handles the rest.

Requirements:
    pip install pymeasure pyvisa numpy pandas
"""

import time
import logging
import threading
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

from pymeasure.instruments.keithley import Keithley2182, Keithley6221

from instruments.keithley6221 import (
    SourceConfig,
    connect_source,
    shutdown_source,
    acquire_reversal_averaged_voltage,
)
from instruments.keithley2182 import VoltmeterConfig, connect_voltmeter
from instruments.kepco_magnet import (
    MagnetConfig,
    connect_magnet,
    set_magnet_current,
    shutdown_magnet,
)
from instruments.lakeshore475 import (
    LakeShore475,
    GaussmeterConfig,
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

from dc.dc_sweep_utils import linear_sweep

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
# SourceConfig, VoltmeterConfig, MagnetConfig and GaussmeterConfig live in
# instruments/ (see the imports above). Only what's specific to this
# measurement (the field-sweep points and acquisition timing) is below.

@dataclass
class AcquisitionConfig:
    """Timing and averaging parameters."""
    settling_time_s: float = 1.0      # Dead-time after a field change  [s]
    n_reversals: int       = 5        # +I/-I reversal pairs averaged per point
    output_file: str       = "dc_hall.csv"


@dataclass
class FieldPoint:
    """
    One point in the measurement sequence.

    The magnetic field sweep (magnet_current_A below) is a worked example
    of the general pattern for sweeping any external parameter:

        1.  Add a plain field here, e.g. temperature_K: float = 300.0
        2.  Supply a set_action that applies it.
        3.  Add the field to the `record` dict inside run_measurement()
            so it is logged to the CSV.

    The set_action is called first at each point, then the script settles
    and acquires — no other changes are needed.

    Note there is no `magnet_field_mT` input field: the field isn't known
    ahead of the sweep, it's measured live by the Lake Shore 475 Gaussmeter
    inside run_measurement() and only appears in the output `record`.
    """
    magnet_current_A: Optional[float] = None
    settling_override_s: Optional[float] = None
    set_action: Optional[Callable[[], None]] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    source:     Keithley6221,
    voltmeter:  Keithley2182,
    src_cfg:    SourceConfig,
    acq_cfg:    AcquisitionConfig,
    points:     List[FieldPoint],
    stop_event: Optional[threading.Event] = None,
    on_point:   Optional[Callable[[dict], None]] = None,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg:  Optional[GaussmeterConfig] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg:  Optional[TemperatureControllerConfig] = None,
    write_csv: Optional[Callable[[List[dict]], None]] = None,
) -> pd.DataFrame:
    """
    Iterate over `points`, acquire the reversal-averaged Hall voltage at
    each, log to CSV.

    Returns a DataFrame of all recorded data. The CSV is written after
    every point so a crash never loses data.

    `stop_event`, if given, is checked before each point (and mid-reversal
    inside acquire_reversal_averaged_voltage) — set it to break out of the sweep early
    while still returning the data collected so far, so callers can run
    their normal shutdown/cleanup path instead of killing the process
    outright.

    `on_point`, if given, is called with each point's `record` dict right
    after it's appended — lets a caller (e.g. a live TUI) show progress
    without polling the output CSV.

    `gaussmeter`/`gauss_cfg`, if given, are used to measure the actual
    field at each point (after settling) instead of leaving
    `magnet_field_mT` unset.

    `temp_ctrl`/`temp_cfg`, if given, log the sample/probe temperature
    (temperature_1_K / temperature_2_K) at each point via the shared
    MercuryiTC controller (see mercury_itc.py). Passing `temp_ctrl=None`
    (e.g. because the MercuryiTC isn't connected) simply leaves those
    columns empty — it's never a reason to stop the measurement.

    `write_csv`, if given, replaces the plain
    `pd.DataFrame(records).to_csv(acq_cfg.output_file, index=False)` write
    with a caller-supplied writer (see instruments.data_naming) — used by
    the TUI/web layer to write the sample-convention header alongside the
    data. Omit it (the default) to keep writing a plain headerless CSV to
    acq_cfg.output_file, unchanged from before.
    """
    records: List[dict] = []

    for idx, pt in enumerate(points):
        if stop_event is not None and stop_event.is_set():
            log.info("Measurement aborted after %d / %d points.", idx, len(points))
            break

        if pt.magnet_current_A is not None:
            log.info("── Point %d / %d   I_magnet=%.4f A ──────────────────",
                      idx + 1, len(points), pt.magnet_current_A)
        else:
            log.info("── Point %d / %d ──────────────────────────────────", idx + 1, len(points))

        # ── 1. Apply external parameter ────────────────────────────────────
        if pt.set_action is not None:
            pt.set_action()

        # ── 2. Settle ──────────────────────────────────────────────────────
        settle = pt.settling_override_s if pt.settling_override_s is not None \
                 else acq_cfg.settling_time_s
        log.info("   Settling %.2f s ...", settle)
        time.sleep(settle)

        # ── 3. Measure field (Lake Shore 475 Gaussmeter) ────────────────────
        field_mT = None
        if gaussmeter is not None and gauss_cfg is not None:
            field_mT = read_field_mT(gaussmeter, gauss_cfg)
            log.info("   B=%.4f mT (measured)", field_mT)

        # ── 4. Acquire reversal-averaged Hall voltage ───────────────────────
        hv = acquire_reversal_averaged_voltage(
            source, voltmeter, src_cfg.sense_current_A, acq_cfg.n_reversals, stop_event)
        r_hall = hv["mean"] / src_cfg.sense_current_A
        log.info("   V_Hall=%.4e V  σ=%.2e V  R_Hall=%.5g Ω  V_even=%.4e V  (n=%d reversals)",
                  hv["mean"], hv["std"], r_hall, hv["even_mean"], hv["n_reversals"])

        # ── 4b. Read temperature (MercuryiTC, optional) ─────────────────────
        temp_1_K, temp_2_K = read_temperature(temp_ctrl, temp_cfg) \
            if temp_cfg is not None else (None, None)

        # ── 5. Build record ──────────────────────────────────────────────────
        record: dict = {
            "point_index":      idx,
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            "magnet_current_A": pt.magnet_current_A,
            "magnet_field_mT":  field_mT,
            "temperature_1_K":  temp_1_K,
            "temperature_2_K":  temp_2_K,
            "sense_current_A":  src_cfg.sense_current_A,
            "hall_voltage_V":   hv["mean"],
            "hall_voltage_std_V": hv["std"],
            "hall_voltage_even_V":     hv["even_mean"],
            "hall_voltage_even_std_V": hv["even_std"],
            "hall_resistance_ohm": r_hall,
            "n_reversals":      hv["n_reversals"],
        }
        records.append(record)

        if on_point is not None:
            on_point(record)

        # ── 6. Write incrementally (never lose data on a crash) ────────────
        if write_csv is not None:
            write_csv(records)
        else:
            Path(acq_cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(acq_cfg.output_file, index=False)

        if stop_event is not None and stop_event.is_set():
            log.info("User aborted measurement mid-point.")
            break

    log.info("Measurement complete. %d points saved to '%s'.", len(records), acq_cfg.output_file)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── configure your devices and sweep here ──────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Source & voltmeter ───────────────────────────────────────────────────
    src_cfg = SourceConfig(
        visa_resource   = "GPIB0::12::INSTR",
        sense_current_A = 1e-3,     # A
        compliance_V    = 2.0,      # V
        source_delay_s  = 0.05,     # s
    )
    source = connect_source(src_cfg)

    volt_cfg = VoltmeterConfig(
        visa_resource = "GPIB0::7::INSTR",
        nplc          = 5,
        auto_range    = True,
    )
    voltmeter = connect_voltmeter(volt_cfg)

    # ── Acquisition settings ─────────────────────────────────────────────────
    acq_cfg = AcquisitionConfig(
        settling_time_s = 1.0,      # wait for magnet to settle
        n_reversals     = 5,
        output_file     = str(_DATA_DIR / f"dc_hall_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    # ── Magnet (Kepco BOP-GL current source) ─────────────────────────────────
    magnet_cfg = MagnetConfig(
        visa_resource        = "GPIB0::6::INSTR",
        current_limit_A      = 35,    # ← safe continuous limit for your magnet
        voltage_compliance_V = 15.0,
        ramp_step_A          = 0.1,
        ramp_delay_s         = 0.05,
    )
    magnet = connect_magnet(magnet_cfg)

    # ── Gaussmeter (Lake Shore 475, measures the actual field) ────────────────
    gauss_cfg = GaussmeterConfig(
        visa_resource = "GPIB0::12::INSTR",   # ← set to your 475's GPIB address
        unit          = "T",
        n_averages    = 10,
        read_delay_s  = 0.05,
    )
    gaussmeter = connect_gaussmeter(gauss_cfg)

    # ── Temperature (Oxford Instruments MercuryiTC, optional) ────────────────
    # Not every rig has one, and not every MercuryiTC has two probes wired up
    # — connect_temperature_controller() returns None rather than raising if
    # it can't be reached, and the measurement runs fine either way.
    temp_cfg = TemperatureControllerConfig(
        visa_resource = "TCPIP0::192.168.1.5::7020::SOCKET",  # ← set to your iTC's address
        sensor_uids   = ("MB1.T1",),   # ← 1 or 2 board UIDs, e.g. ("MB1.T1", "DB5.T1")
    )
    temp_ctrl = connect_temperature_controller(temp_cfg)

    # ── Measurement points  — bidirectional magnet-current sweep ─────────────
    # The field itself (magnet_field_mT in the output) is measured live by
    # the gaussmeter at each point, not computed from the current.
    currents_A = linear_sweep(start=-20.0, stop=20.0, step=2.0, bidirectional=True)

    points = [
        FieldPoint(
            magnet_current_A = I,
            set_action = lambda I=I: set_magnet_current(magnet, magnet_cfg, I),
        )
        for I in currents_A
    ]

    # ── Run ──────────────────────────────────────────────────────────────────
    # The magnet drives an inductive load, so always ramp it back to zero and
    # disable the output — even if the measurement raises partway through.
    # Likewise always disable the 6221's current output.
    try:
        df = run_measurement(source, voltmeter, src_cfg, acq_cfg, points,
                              gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
                              temp_ctrl=temp_ctrl, temp_cfg=temp_cfg)
        print("\n", df.to_string(index=False))
    finally:
        shutdown_source(source)
        shutdown_magnet(magnet, magnet_cfg)
        shutdown_gaussmeter(gaussmeter)
        shutdown_temperature_controller(temp_ctrl)


if __name__ == "__main__":
    main()
