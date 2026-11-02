#!/usr/bin/env python3
"""
DC Spin-Valve / Field Sweep — Keithley 6221 (or 2450)/2182 + Kepco magnet + gate
==========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-04

Field-swept longitudinal voltage read (spin-valve / magnetoresistance),
with a fixed (or listed) gate voltage via a Keithley 2400.

Wiring
------
    Keithley 6221 (current source)
      Output (current) ──▶ sample ── common ground

      — or —

    Keithley 2450 (SMU, sources current OR voltage, 2-wire)
      Force HI/LO ──▶ sample current leads
      Also logs its own 2-wire reading (V when sourcing I, I when
      sourcing V) → smu_voltage_V / smu_current_A / resistance_2w_ohm

    Keithley 2182 (nanovoltmeter)
      Channel 1 (differential) ──▶ across the sample (longitudinal
      voltage leads, e.g. a spin-valve stack)

    Keithley 2400 (gate source)
      Output (voltage) ──▶ gate electrode

    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample

Method
------
Sources a fixed DC sense current with the 6221 (or a fixed current or
voltage with the 2450 — see acquire_smu_reversal()) and reads the longitudinal
voltage with the 2182, reversing the current (+I / -I) at each field point
and decomposing the voltage into odd (the reported voltage/R) and even
parts. See docs/current-reversal.md for why both are recorded (columns
voltage_even_V / voltage_even_sem_V) — for spin-valve-type stacks with
strong spin-orbit coupling, the even-in-current term can carry real
physics (unidirectional SMR, Joule heating, rectification), not just
instrumental offset.

The gate voltage is held fixed for the whole field sweep (or looped over
a list — one complete field sweep per value, each saved to its own file).

Requirements:
    pip install pymeasure pyvisa numpy pandas
"""

import time
import logging
import threading
import numpy as np
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
from instruments.keithley2182 import (
    VoltmeterConfig,
    connect_voltmeter,
    acquire_averaged_voltage,
)
from instruments.keithley2450 import (
    Keithley2450,
    SMUConfig,
    connect_smu,
    read_measurement,
    set_source_level,
    shutdown_smu,
)
from instruments.keithley2400 import (
    GateConfig,
    connect_gate,
    set_gate_voltage,
    shutdown_gate,
)
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

from dc.dc_sweep_utils import linear_sweep, safe_shutdown

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
# live in instruments/ (see the imports above). Only what's specific to this
# measurement (the field-sweep points and acquisition timing) is below.

@dataclass
class AcquisitionConfig:
    """Timing and averaging parameters."""
    settling_time_s: float  = 1.0      # Physics-equilibration dwell AFTER the field
                                        # has settled at its final value  [s]
    field_settle_tolerance_mT: float = 0.02  # Passed to set_magnet_current(): the field
                                        # counts as settled once a short window of
                                        # gaussmeter readings spans no more than this.
                                        # Exposed in the TUI "Advanced" section for
                                        # rigs where the default is too tight/loose.
    reversal_enabled: bool  = True     # Reverse the sense current each rep (+I/-I) to
                                        # cancel thermal-EMF offsets. Some devices are
                                        # bias-direction dependent (e.g. diodes,
                                        # asymmetric spin-orbit stacks) and reversing the
                                        # current there destroys rather than cleans up the
                                        # signal — turn this off for those.
    n_averages: int         = 5        # reversal_enabled=True:  +I/-I reversal pairs averaged per point
                                        # reversal_enabled=False: plain voltage samples averaged per point
    output_file: str        = "dc_spin_valve.csv"


@dataclass
class FieldPoint:
    """One point in the magnet-current sweep. There is no `magnet_field_mT`
    input field: the field isn't known ahead of the sweep, it's measured
    live by the Lake Shore 475 Gaussmeter inside run_measurement() and only
    appears in the output `record`."""
    magnet_current_A: Optional[float] = None
    settling_override_s: Optional[float] = None
    set_action: Optional[Callable[[], None]] = field(default=None, repr=False)


def acquire_smu_reversal(
    smu: Keithley2450,
    smu_cfg: SMUConfig,
    voltmeter: Keithley2182,
    level: float,
    n: int,
    reversal: bool = True,
    stop_event: Optional[threading.Event] = None,
    source_delay_s: float = 0.0,
) -> dict:
    """
    The 2450 counterpart of keithley6221.acquire_reversal_averaged_voltage():
    source `level` (A or V, per smu_cfg.source_function), and at each
    polarity read both the 2182 and the 2450's own (2-wire) sense quantity.

    `reversal=True`: n +level/-level pairs, decomposed into odd/even parts
    exactly like the 6221 path. `reversal=False`: n plain samples at +level,
    `even_mean`/`even_sem` are None. `smu_mean`/`smu_sem` are the odd part
    (or plain mean) of the 2450 reading. Leaves the source at +level.
    """
    v_odd, v_even, s_odd = np.empty(n), np.empty(n), np.empty(n)
    n_used = 0

    def _read(lvl: float) -> tuple[float, float]:
        set_source_level(smu, smu_cfg, lvl)
        if source_delay_s > 0:
            time.sleep(source_delay_s)
        return voltmeter.voltage, read_measurement(smu, smu_cfg)

    for i in range(n):
        v_plus, s_plus = _read(level)
        if reversal:
            v_minus, s_minus = _read(-level)
            v_odd[i], v_even[i] = (v_plus - v_minus) / 2.0, (v_plus + v_minus) / 2.0
            s_odd[i] = (s_plus - s_minus) / 2.0
        else:
            v_odd[i], s_odd[i] = v_plus, s_plus
        n_used = i + 1
        if stop_event is not None and stop_event.is_set():
            break
    if reversal:
        set_source_level(smu, smu_cfg, level)

    def _stats(a: np.ndarray) -> tuple[float, float]:
        a = a[:n_used]
        sem = float(np.std(a, ddof=1) / np.sqrt(n_used)) if n_used >= 2 else float("nan")
        return float(np.mean(a)), sem

    mean, sem = _stats(v_odd)
    even_mean, even_sem = _stats(v_even) if reversal else (None, None)
    smu_mean, smu_sem = _stats(s_odd)
    return {"mean": mean, "sem": sem, "even_mean": even_mean, "even_sem": even_sem,
            "smu_mean": smu_mean, "smu_sem": smu_sem, "n_reversals": n_used}


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
    write_csv: Optional[Callable[[List[dict]], None]] = None,
    smu: Optional[Keithley2450] = None,
    smu_cfg: Optional[SMUConfig] = None,
    smu_level: Optional[float] = None,
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

    `smu`/`smu_cfg`/`smu_level`, if given, replace the 6221 (pass
    `source=None`): the 2450 sources `smu_level` (A or V per
    smu_cfg.source_function) and its own 2-wire reading is logged too
    (smu_voltage_V, smu_current_A, resistance_2w_ohm). `src_cfg` then only
    supplies source_delay_s. In voltage mode sense_current_A is the
    MEASURED current, so resistance_ohm stays V_2182 / I.
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
        # set_action (for a field point, set_magnet_current) now blocks until
        # the magnet current and the measured field have both settled, and
        # returns a provenance dict folded into the record below.
        settle_info = pt.set_action() if pt.set_action is not None else None

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

        # ── 4. Acquire voltage ────────────────────────────────────────────
        # Reversal averaging (+I/-I, cancels thermal-EMF offsets) is the
        # default, but some devices are bias-direction dependent — reversing
        # the current there destroys the signal rather than cleaning it up
        # — so acq_cfg.reversal_enabled lets the sense current be held fixed
        # and just averaged plainly instead.
        smu_cols: dict = {}
        if smu is not None:
            rv = acquire_smu_reversal(
                smu, smu_cfg, voltmeter, smu_level, acq_cfg.n_averages,
                acq_cfg.reversal_enabled, stop_event, src_cfg.source_delay_s)
            v_mean, v_sem = rv["mean"], rv["sem"]
            v_even_mean, v_even_sem = rv["even_mean"], rv["even_sem"]
            n_used = rv["n_reversals"]
            if smu_cfg.source_function == "voltage":
                i_sense = rv["smu_mean"]
                smu_cols = {"smu_voltage_V": smu_level, "smu_current_A": i_sense,
                            "smu_current_sem_A": rv["smu_sem"],
                            "resistance_2w_ohm": smu_level / i_sense}
            else:
                i_sense = smu_level
                smu_cols = {"smu_voltage_V": rv["smu_mean"], "smu_voltage_sem_V": rv["smu_sem"],
                            "smu_current_A": smu_level,
                            "resistance_2w_ohm": rv["smu_mean"] / smu_level}
            r = v_mean / i_sense
            log.info("   V=%.4e V  SEM=%.2e V  R=%.5g Ω  R_2w=%.5g Ω  (n=%d, 2450 %s source)",
                      v_mean, v_sem, r, smu_cols["resistance_2w_ohm"], n_used,
                      smu_cfg.source_function)
        elif acq_cfg.reversal_enabled:
            i_sense = src_cfg.sense_current_A
            rv = acquire_reversal_averaged_voltage(
                source, voltmeter, src_cfg.sense_current_A, acq_cfg.n_averages,
                stop_event, source_delay_s=src_cfg.source_delay_s)
            v_mean, v_sem = rv["mean"], rv["sem"]
            v_even_mean, v_even_sem = rv["even_mean"], rv["even_sem"]
            n_used = rv["n_reversals"]
            r = v_mean / src_cfg.sense_current_A
            log.info("   V=%.4e V  SEM=%.2e V  R=%.5g Ω  V_even=%.4e V  (n=%d reversals)",
                      v_mean, v_sem, r, v_even_mean, n_used)
        else:
            i_sense = src_cfg.sense_current_A
            av = acquire_averaged_voltage(voltmeter, acq_cfg.n_averages, stop_event)
            v_mean, v_sem = av["mean"], av["sem"]
            v_even_mean, v_even_sem = None, None
            n_used = acq_cfg.n_averages
            r = v_mean / src_cfg.sense_current_A
            log.info("   V=%.4e V  SEM=%.2e V  R=%.5g Ω  (n=%d averages, reversal off)",
                      v_mean, v_sem, r, n_used)

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
            "sense_current_A":  i_sense,
            "reversal_enabled": acq_cfg.reversal_enabled,
            "voltage_V":        v_mean,
            "voltage_sem_V":    v_sem,
            "voltage_even_V":     v_even_mean,
            "voltage_even_sem_V": v_even_sem,
            "resistance_ohm":   r,
            "n_averages":       n_used,
            "gate_voltage_V":   gate_voltage_V,
            **smu_cols,
        }
        if isinstance(settle_info, dict):
            record.update({
                "current_settled":  settle_info.get("current_settled"),
                "field_settled":    settle_info.get("field_settled"),
                "settle_elapsed_s": settle_info.get("settle_elapsed_s"),
                "i_measured_A":     settle_info.get("i_measured_A"),
            })
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
        visa_resource   = "GPIB0::20::INSTR",
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
        visa_resource        = "GPIB0::25::INSTR",
        gate_voltage_limit_V = 20.0,
        compliance_current_A = 1e-6,
    )
    gate = connect_gate(gate_cfg)
    set_gate_voltage(gate, gate_cfg, 0.0)

    # ── Acquisition settings ─────────────────────────────────────────────────
    acq_cfg = AcquisitionConfig(
        settling_time_s  = 1.0,
        reversal_enabled = True,
        n_averages       = 5,
        output_file      = str(_DATA_DIR / f"dc_spin_valve_{datetime.now():%Y%m%d_%H%M%S}.csv"),
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
            set_action = lambda I=I: set_magnet_current(
                magnet, magnet_cfg, I, gaussmeter, gauss_cfg,
                acq_cfg.field_settle_tolerance_mT),
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
        safe_shutdown("6221", lambda: shutdown_source(source))
        safe_shutdown("gate (2400)", lambda: shutdown_gate(gate))
        safe_shutdown("magnet", lambda: shutdown_magnet(magnet, magnet_cfg))
        safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


if __name__ == "__main__":
    main()
