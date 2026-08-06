#!/usr/bin/env python3
"""
DC Hall Voltage Measurement — Keithley 6221 + 2182, with field sweep
=====================================================================
Companion to bridge/MFLI/mfli_dual_harmonic.py — same house style
(dataclass configs, incremental CSV, stop_event/on_point hooks for a
live UI), different (DC rather than lock-in) measurement technique.

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
(Hall) voltage with the 2182. At each field point the sense current is
reversed (+I / -I) and the voltage decomposed into an odd and even part
over repeated +/- pairs:

    V_odd  = (V(+I) - V(-I)) / 2   <- reported as "the" Hall voltage
    V_even = (V(+I) + V(-I)) / 2   <- recorded, not discarded

V_odd cancels any DC offset common to both polarities — thermal EMFs at
the contacts, amplifier offset, etc. — that a single-polarity reading
would fold straight into the Hall signal. But "even in current" is not
the same thing as "boring instrumental offset": expanding V(I) =
V_offset + R*I + beta*I^2 + gamma*I^3 + ... shows that V_odd keeps only
odd powers of I and V_even keeps only even powers, including physics
that genuinely lives there in spin-orbit-coupled stacks (unidirectional
spin Hall magnetoresistance, Joule-heating-driven Delta-R(T),
rectification-type effects) — an offset-cancelling average that only
ever reports V_odd would silently zero all of that out. V_even is
therefore recorded alongside V_odd on every point (columns
hall_voltage_even_V / hall_voltage_even_std_V) rather than thrown away:
if it's flat noise, nothing was lost; if it shows structure vs. field,
that's a real signal current-reversal averaging alone would otherwise
hide.

Magnetic field sweep
--------------------
A Kepco BOP-GL bipolar power supply (see kepco_magnet.KepkoBOPGL) drives
current through an electromagnet to provide the field axis, exactly as
in mfli_dual_harmonic.py: bidirectional_current_sweep() builds an
up-then-down magnet-current list so hysteresis is visible, and the
actual field at the sample is measured directly with a Lake Shore 475
DSP Gaussmeter (see lakeshore475.LakeShore475) at each point rather than
inferred from the magnet current via a calibration constant.

Extensibility
-------------
Add new sweep variables to FieldPoint and a corresponding set_action
callable — the run_measurement loop handles the rest.

Requirements:
    pip install pymeasure pyvisa numpy pandas
"""

import sys
import time
import logging
import threading
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

import pyvisa
from pymeasure.instruments.keithley import Keithley2182, Keithley6221

# The Kepco magnet and Lake Shore 475 drivers live in the shared
# bridge/instruments folder — add it to sys.path directly (it's not
# installed as a normal package).
_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent / "instruments"
if str(_INSTRUMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTRUMENTS_DIR))
from kepco_magnet import KepkoBOPGL  # noqa: E402
from lakeshore475 import LakeShore475  # noqa: E402
from mercury_itc import (  # noqa: E402
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    shutdown_temperature_controller,
)

from dc_sweep_utils import linear_sweep  # noqa: E402

# Data lives outside "bridge" (a sibling of it) so measurement output never
# ends up inside the git-tracked source tree. Same location as the MFLI
# programs' _DATA_DIR (bridge/DC/foo.py -> DC -> bridge -> repo root).
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
    """Keithley 6221 — DC sense current source."""
    visa_resource: str      = "GPIB0::20::INSTR"
    sense_current_A: float  = 1e-3    # Sense current magnitude [A]
    compliance_V: float     = 2.0     # Voltage compliance [V]
    source_delay_s: float   = 0.05    # Settle time after each current step [s]


@dataclass
class VoltmeterConfig:
    """Keithley 2182 — Hall voltage readout (channel 1, differential)."""
    visa_resource: str = "GPIB0::7::INSTR"
    nplc: float        = 5      # Integration time [power line cycles]
    auto_range: bool   = True


@dataclass
class MagnetConfig:
    """
    Kepco BOP-GL bipolar power supply, used as a current source for an
    electromagnet (see kepco_magnet.KepkoBOPGL). Identical role/shape to
    MagnetConfig in mfli_dual_harmonic.py — the field is measured live via
    the Gaussmeter (see GaussmeterConfig), never inferred from a
    current->field calibration constant.
    """
    visa_resource:        str   = "GPIB0::6::INSTR"
    current_limit_A:      float = 50.0    # Software current limit  [A]
    voltage_compliance_V: float = 20.0    # CC-mode compliance / OVP limit  [V]
    ramp_step_A:          float = 0.1     # Ramp step size  [A]
    ramp_delay_s:         float = 0.05    # Delay between ramp steps  [s]


@dataclass
class GaussmeterConfig:
    """Lake Shore 475 DSP Gaussmeter (see lakeshore475.LakeShore475)."""
    visa_resource: str   = "GPIB0::12::INSTR"
    unit:          str   = "T"     # 'T' or 'G' — read_field_mT() only knows these two
    n_averages:    int   = 10      # Field readings averaged per measurement point
    read_delay_s:  float = 0.05    # Delay between successive readings  [s]


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
    of the general pattern for sweeping any external parameter — mirrors
    MeasurementPoint in mfli_dual_harmonic.py:

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
# Instrument setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def connect_source(cfg: SourceConfig) -> Keithley6221:
    """Open and configure the Keithley 6221 as a DC current source."""
    source = Keithley6221(cfg.visa_resource)
    source.reset()
    source.source_auto_range = True
    source.source_compliance = cfg.compliance_V
    source.source_delay = cfg.source_delay_s
    source.source_current = 0.0
    source.enable_source()
    log.info("Keithley 6221 connected: %s  I_sense=%.4g A  compliance=%.2f V",
              cfg.visa_resource, cfg.sense_current_A, cfg.compliance_V)
    return source


def connect_voltmeter(cfg: VoltmeterConfig) -> Keithley2182:
    """Open and configure the Keithley 2182 for a Hall voltage readout."""
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
# Magnet control (Kepco BOP-GL current source, via VISA/PyVISA)
# ─────────────────────────────────────────────────────────────────────────────
# Identical to mfli_dual_harmonic.py's magnet helpers — duplicated here
# (rather than imported across the MFLI/DC boundary) so this module stays
# self-contained, matching the existing house convention.

def connect_magnet(cfg: MagnetConfig) -> KepkoBOPGL:
    """
    Open a VISA session to the Kepco BOP-GL and arm it as a current source.

    See mfli_dual_harmonic.connect_magnet for the full rationale — in
    short: clears stale CURR:LIM/VOLT:LIM setpoint ceilings back to the
    supply's full rating (independent of, and not restored by, *RST),
    then puts it in constant-current mode with the configured software
    compliance voltage and current limit.
    """
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

def acquire_hall_voltage(
    source: Keithley6221,
    voltmeter: Keithley2182,
    src_cfg: SourceConfig,
    n_reversals: int,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """
    Reverse the sense current n_reversals times and decompose the
    resulting Hall voltage into its odd and even parts in I:

        V_odd  = (V(+I) - V(-I)) / 2   — the Hall signal; cancels any
                                          offset common to both polarities
                                          (thermal EMFs, amplifier offset)
                                          since a true offset doesn't flip
                                          sign with the current
        V_even = (V(+I) + V(-I)) / 2   — everything that DOES share the
                                          offset's sign symmetry: a real
                                          instrumental offset, but also any
                                          genuine even-in-I physics (e.g.
                                          unidirectional SMR, I^2 Joule
                                          heating) that V_odd alone would
                                          silently discard

    Both are returned (and both get logged to the CSV by the caller) so
    V_even can be checked after the fact instead of being thrown away.

    Leaves the source at +sense_current_A on return. If `stop_event` fires
    partway through, returns the mean/std of whatever pairs were already
    collected (at least one).
    """
    samples_odd = np.empty(n_reversals)
    samples_even = np.empty(n_reversals)
    n_used = 0

    for i in range(n_reversals):
        source.source_current = src_cfg.sense_current_A
        v_plus = voltmeter.voltage

        source.source_current = -src_cfg.sense_current_A
        v_minus = voltmeter.voltage

        samples_odd[i] = (v_plus - v_minus) / 2.0
        samples_even[i] = (v_plus + v_minus) / 2.0
        n_used = i + 1

        if stop_event is not None and stop_event.is_set():
            break

    source.source_current = src_cfg.sense_current_A

    used_odd = samples_odd[:n_used]
    used_even = samples_even[:n_used]
    return {
        "mean": float(np.mean(used_odd)),
        "std": float(np.std(used_odd)),
        "even_mean": float(np.mean(used_even)),
        "even_std": float(np.std(used_even)),
        "n_reversals": n_used,
    }


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
) -> pd.DataFrame:
    """
    Iterate over `points`, acquire the reversal-averaged Hall voltage at
    each, log to CSV.

    Returns a DataFrame of all recorded data. The CSV is written after
    every point so a crash never loses data.

    `stop_event`, if given, is checked before each point (and mid-reversal
    inside acquire_hall_voltage) — set it to break out of the sweep early
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
        hv = acquire_hall_voltage(source, voltmeter, src_cfg, acq_cfg.n_reversals, stop_event)
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
        sensor_uids   = ("DB6.T1",),   # ← 1 or 2 board UIDs, e.g. ("DB6.T1", "DB5.T1")
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
