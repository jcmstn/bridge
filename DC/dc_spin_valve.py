#!/usr/bin/env python3
"""
DC Spin-Valve / Field Sweep — Keithley 6221/2182 + Kepco magnet + gate
==========================================================================
Companion to dc_hall_measurement.py / dc_gate_sweep.py — same house style
(dataclass configs, incremental CSV, stop_event/on_point hooks for a live
UI). Structurally this is dc_hall_measurement.py's field-sweep engine
(magnet current swept, field measured live via the Lake Shore 475,
+I/-I reversal averaging to cancel thermal-EMF offsets) generalized to a
longitudinal voltage read (spin-valve / magnetoresistance) rather than a
transverse Hall voltage, with an added fixed (or listed) gate voltage via
a Keithley 2400.

Wiring
------
    Keithley 6221 (current source)
      Output (current) ──▶ sample ── common ground

    Keithley 2182 (nanovoltmeter)
      Channel 1 (differential) ──▶ across the sample (longitudinal
      voltage leads, e.g. a spin-valve stack)

    Keithley 2400 (gate source)
      Output (voltage) ──▶ gate electrode

    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample

Method
------
Sources a fixed DC sense current with the 6221 and reads the longitudinal
voltage with the 2182. At each field point the sense current is reversed
(+I / -I) and the voltage decomposed into an odd and even part over
repeated +/- pairs:

    V_odd  = (V(+I) - V(-I)) / 2      <- reported as "the" voltage/R
    V_even = (V(+I) + V(-I)) / 2      <- recorded, not discarded

V_odd cancels any DC offset common to both polarities (thermal EMFs at
the contacts, amplifier offset, etc.) — this works for any resistive
element, not just an antisymmetric Hall response, since R itself is
unchanged by the current's sign. But "even in current" is not the same
thing as "boring instrumental offset": expanding V(I) = V_offset + R*I +
beta*I^2 + gamma*I^3 + ... shows that V_odd keeps only odd powers of I
and V_even keeps only even powers, including physics that genuinely
lives there for spin-valve-type stacks with strong spin-orbit coupling
(unidirectional spin Hall magnetoresistance, Joule-heating-driven
Delta-R(T), rectification-type effects) — an offset-cancelling average
that only ever reports V_odd would silently zero all of that out.
V_even is therefore recorded alongside V_odd on every point (columns
voltage_even_V / voltage_even_std_V) rather than thrown away: if it's
flat noise, nothing was lost; if it shows structure vs. field, that's a
real signal current-reversal averaging alone would otherwise hide.

The gate voltage is held fixed for the whole field sweep (or looped over
a list — one complete field sweep per value, each saved to its own file).

Requirements:
    pip install pymeasure pyvisa numpy pandas
"""

import sys
import time
import logging
import threading
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

from pymeasure.instruments.keithley import Keithley2182, Keithley2400, Keithley6221

# The instrument connect/shutdown helpers live in the shared bridge/instruments
# folder — add it to sys.path directly (it's not installed as a normal package).
_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent / "instruments"
if str(_INSTRUMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTRUMENTS_DIR))
from keithley6221 import (  # noqa: E402
    SourceConfig,
    connect_source,
    shutdown_source,
    acquire_reversal_averaged_voltage,
)
from keithley2182 import VoltmeterConfig, connect_voltmeter  # noqa: E402
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
    LakeShore475,
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
# Only what's specific to this measurement (the field-sweep points and
# acquisition timing) is defined below.

@dataclass
class AcquisitionConfig:
    """Timing and averaging parameters."""
    settling_time_s: float = 1.0      # Dead-time after a field change  [s]
    n_reversals: int       = 5        # +I/-I reversal pairs averaged per point
    output_file: str       = "dc_spin_valve.csv"


@dataclass
class FieldPoint:
    """One point in the magnet-current sweep — mirrors FieldPoint in
    dc_hall_measurement.py. There is no `magnet_field_mT` input field: the
    field isn't known ahead of the sweep, it's measured live by the Lake
    Shore 475 Gaussmeter inside run_measurement() and only appears in the
    output `record`."""
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
    gate_voltage_V: Optional[float] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg:  Optional[TemperatureControllerConfig] = None,
) -> pd.DataFrame:
    """
    Iterate over `points`, acquire the reversal-averaged voltage at each,
    log to CSV.

    Returns a DataFrame of all recorded data. The CSV is written after
    every point so a crash never loses data.

    `stop_event`, if given, is checked before each point (and mid-reversal
    inside acquire_reversal_averaged_voltage) — set it to break out of the
    sweep early while still returning the data collected so far.

    `on_point`, if given, is called with each point's `record` dict right
    after it's appended — lets a caller (e.g. a live TUI) show progress
    without polling the output CSV.

    `gaussmeter`/`gauss_cfg`, if given, are used to measure the actual
    field at each point (after settling) instead of leaving
    `magnet_field_mT` unset.

    `gate_voltage_V`, if given, is recorded on every point (the gate itself
    is set once by the caller before the sweep starts, not per point).

    `temp_ctrl`/`temp_cfg`, if given, log the sample/probe temperature
    (temperature_1_K / temperature_2_K) at each point via the shared
    MercuryiTC controller (see mercury_itc.py). Passing `temp_ctrl=None`
    (e.g. because the MercuryiTC isn't connected) simply leaves those
    columns empty — it's never a reason to stop the measurement.
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

        # ── 4. Acquire reversal-averaged voltage ────────────────────────────
        rv = acquire_reversal_averaged_voltage(
            source, voltmeter, src_cfg.sense_current_A, acq_cfg.n_reversals, stop_event)
        r = rv["mean"] / src_cfg.sense_current_A
        log.info("   V=%.4e V  σ=%.2e V  R=%.5g Ω  V_even=%.4e V  (n=%d reversals)",
                  rv["mean"], rv["std"], r, rv["even_mean"], rv["n_reversals"])

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
            "voltage_V":        rv["mean"],
            "voltage_std_V":    rv["std"],
            "voltage_even_V":     rv["even_mean"],
            "voltage_even_std_V": rv["even_std"],
            "resistance_ohm":   r,
            "n_reversals":      rv["n_reversals"],
            "gate_voltage_V":   gate_voltage_V,
        }
        records.append(record)

        if on_point is not None:
            on_point(record)

        # ── 6. Write incrementally (never lose data on a crash) ────────────
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
        sense_current_A = 1e-3,
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

    # ── Gate (Keithley 2400, held fixed for this example) ────────────────────
    gate_cfg = GateConfig(
        visa_resource        = "GPIB0::24::INSTR",
        gate_voltage_limit_V = 20.0,
        compliance_current_A = 1e-6,
    )
    gate = connect_gate(gate_cfg)
    set_gate_voltage(gate, gate_cfg, 0.0)

    # ── Acquisition settings ─────────────────────────────────────────────────
    acq_cfg = AcquisitionConfig(
        settling_time_s = 1.0,
        n_reversals     = 5,
        output_file     = str(_DATA_DIR / f"dc_spin_valve_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    # ── Magnet (Kepco BOP-GL current source) ─────────────────────────────────
    magnet_cfg = MagnetConfig(
        visa_resource        = "GPIB0::6::INSTR",
        current_limit_A      = 35,
        voltage_compliance_V = 15.0,
        ramp_step_A          = 0.1,
        ramp_delay_s         = 0.05,
    )
    magnet = connect_magnet(magnet_cfg)

    # ── Gaussmeter (Lake Shore 475, measures the actual field) ────────────────
    gauss_cfg = GaussmeterConfig(
        visa_resource = "GPIB0::12::INSTR",
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
    try:
        df = run_measurement(source, voltmeter, src_cfg, acq_cfg, points,
                              gaussmeter=gaussmeter, gauss_cfg=gauss_cfg, gate_voltage_V=0.0,
                              temp_ctrl=temp_ctrl, temp_cfg=temp_cfg)
        print("\n", df.to_string(index=False))
    finally:
        shutdown_source(source)
        shutdown_gate(gate)
        shutdown_magnet(magnet, magnet_cfg)
        shutdown_gaussmeter(gaussmeter)
        shutdown_temperature_controller(temp_ctrl)


if __name__ == "__main__":
    main()
