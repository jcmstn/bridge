#!/usr/bin/env python3
"""
SOT DC characterisation — 4-probe channel resistance, Keithley 4200A-SCS only
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-07

The "make measuring R_xx carefree" tool of the SOT plan (Stage 0). One
Keithley 4200A does the whole electrical chain — no 6221/2182.

Wiring
------
    4200A SMU1 (force current)
      Force HI/LO ──▶ outer current leads of the Hall bar

    4200A SMU2 (force 0 A → high-impedance voltmeter)
      Force HI/LO ──▶ inner longitudinal voltage leads (true 4-probe:
      no lead resistance, no reliance on a KXCI remote-sense toggle)

Method
------
For each current in a bidirectional sweep (a single point if min == max),
reverse the SMU1 current (+I / -I) and decompose the SMU2 voltage into odd
(the resistive drop, R = V_odd / I) and even parts — see
docs/current-reversal.md. Reversal cancels the thermal-EMF offset that
4-wire alone leaves in series with the channel. Turn reversal off
(``reversal_enabled=False``) for a bias-direction-dependent channel.

Two Stage-0 needs, one loop:
  * baseline R_xx     → min == max, small probe current, ``n_repeats`` ~10
  * safe-current ramp → a sweep of 30-50 steps, watch ``resistance_ohm``
    (and ``channel_voltage_V``) for the onset of drift/heating → sets I_max

Requirements: pymeasure, pyvisa, numpy, pandas. KXCI enabled on the 4200A
(see instruments/keithley4200a.py for transport + validation notes).
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from instruments.keithley4200a import (
    Keithley4200AConfig,
    SMUChannelConfig,
    acquire_measurement,
    acquire_reversal_averaged,
    configure_smu,
    connect_4200a,
    read_measurement,
    set_source_level,
    shutdown_4200a,
)
from instruments.mercury_itc import (
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    shutdown_temperature_controller,
)
from dc.dc_sweep_utils import linear_sweep, safe_shutdown

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Keithley4200AConfig, SMUChannelConfig, TemperatureControllerConfig come from
# instruments/ (imported above and re-exported for the TUI). Only the sweep +
# acquisition shape is local.

@dataclass
class SweepConfig:
    """The SMU1 current sweep. min == max collapses to a single fixed current
    (the baseline-R_xx case)."""
    current_min_A: float = -1e-4
    current_max_A: float =  1e-4
    step_A: float        =  1e-5
    bidirectional: bool  = True     # min → max → min, so heating hysteresis shows


@dataclass
class AcquisitionConfig:
    settling_time_s: float = 0.1      # dead-time after each current step [s]
    reversal_enabled: bool = True     # +I/-I decomposition per point
    n_averages: int        = 5        # reversal pairs (or plain V samples) per point
    n_repeats: int         = 1        # repeat the whole sweep; recorded as repeat_index
    output_file: str       = "sot_dc_characterization.csv"


@dataclass
class CurrentPoint:
    current_A: float
    settling_override_s: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop  (docs/architecture.md §3 contract)
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    k4200,
    smu_src_cfg: SMUChannelConfig,
    smu_sense_cfg: SMUChannelConfig,
    acq_cfg: AcquisitionConfig,
    points: List[CurrentPoint],
    stop_event: Optional[threading.Event] = None,
    on_point: Optional[Callable[[dict], None]] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg: Optional[TemperatureControllerConfig] = None,
    write_csv: Optional[Callable[[List[dict]], None]] = None,
) -> pd.DataFrame:
    """Run ``acq_cfg.n_repeats`` passes over ``points``; at each current
    acquire R = V_odd / I (SMU1 forces ±I, SMU2 reads V), append a row, write
    the CSV in full (crash-safe).

    ``stop_event`` is checked before each point and mid-reversal.
    ``on_point`` gets each record dict as it is appended.
    ``temp_ctrl``/``temp_cfg`` optionally log sample/probe temperature;
    ``None`` is never a reason to stop.
    """
    records: List[dict] = []
    total = acq_cfg.n_repeats * len(points)

    for repeat in range(acq_cfg.n_repeats):
        for idx, pt in enumerate(points):
            if stop_event is not None and stop_event.is_set():
                log.info("Aborted after %d / %d points.", len(records), total)
                return pd.DataFrame(records)

            settle = pt.settling_override_s if pt.settling_override_s is not None \
                else acq_cfg.settling_time_s

            if acq_cfg.reversal_enabled and pt.current_A != 0.0:
                rv = acquire_reversal_averaged(
                    k4200, smu_src_cfg, smu_sense_cfg, pt.current_A,
                    acq_cfg.n_averages, stop_event, source_delay_s=settle)
                v_mean, v_sem = rv["mean"], rv["sem"]
                v_even, v_even_sem = rv["even_mean"], rv["even_sem"]
                n_used = rv["n_reversals"]
            else:
                set_source_level(k4200, smu_src_cfg, pt.current_A)
                if settle > 0:
                    time.sleep(settle)
                av = acquire_measurement(k4200, smu_sense_cfg, acq_cfg.n_averages, stop_event)
                v_mean, v_sem = av["mean"], av["sem"]
                v_even, v_even_sem = None, None
                n_used = acq_cfg.n_averages

            r = v_mean / pt.current_A if pt.current_A != 0.0 else float("nan")
            channel_V = read_measurement(k4200, smu_src_cfg)   # SMU1 terminal V — heating/compliance monitor
            t1_K, t2_K = read_temperature(temp_ctrl, temp_cfg) if temp_cfg is not None else (None, None)

            record = {
                "point_index":       idx,
                "repeat_index":      repeat,
                "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                "set_current_A":     pt.current_A,
                "channel_voltage_V": channel_V,
                "voltage_V":         v_mean,
                "voltage_sem_V":     v_sem,
                "voltage_even_V":     v_even,
                "voltage_even_sem_V": v_even_sem,
                "resistance_ohm":    r,
                "n_averages":        n_used,
                "reversal_enabled":  acq_cfg.reversal_enabled,
                "temperature_1_K":   t1_K,
                "temperature_2_K":   t2_K,
            }
            records.append(record)
            if on_point is not None:
                on_point(record)

            if write_csv is not None:
                write_csv(records)
            else:
                Path(acq_cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(records).to_csv(acq_cfg.output_file, index=False)

            log.info("repeat %d  I=%.4g A  R=%.6g Ω  V_ch=%.4g V",
                     repeat, pt.current_A, r, channel_V)

    log.info("Done. %d rows → '%s'", len(records), acq_cfg.output_file)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    k_cfg = Keithley4200AConfig(visa_resource="GPIB0::17::INSTR", integration="normal")
    src_cfg = SMUChannelConfig(channel=1, source_function="current",
                               compliance_voltage_V=2.0, source_limit_A=5e-3)
    sense_cfg = SMUChannelConfig(channel=2, source_function="current",  # forces 0 A
                                 compliance_voltage_V=2.0, source_limit_A=1e-9)
    acq_cfg = AcquisitionConfig(
        n_averages=5, n_repeats=10,
        output_file=str(_DATA_DIR / f"sot_dcchar_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )
    sweep_cfg = SweepConfig(current_min_A=-1e-4, current_max_A=1e-4, step_A=1e-4)  # single point ±100 µA

    temp_cfg = TemperatureControllerConfig(
        visa_resource="TCPIP0::192.168.1.5::7020::SOCKET", sensor_uids=("MB1.T1",))
    temp_ctrl = connect_temperature_controller(temp_cfg)

    k4200 = connect_4200a(k_cfg)
    configure_smu(k4200, src_cfg)
    configure_smu(k4200, sense_cfg)
    set_source_level(k4200, sense_cfg, 0.0)   # park SMU2 as the voltmeter

    currents = linear_sweep(sweep_cfg.current_min_A, sweep_cfg.current_max_A,
                            sweep_cfg.step_A, bidirectional=sweep_cfg.bidirectional)
    points = [CurrentPoint(current_A=float(i)) for i in currents]

    try:
        df = run_measurement(k4200, src_cfg, sense_cfg, acq_cfg, points,
                             temp_ctrl=temp_ctrl, temp_cfg=temp_cfg)
        print("\n", df.to_string(index=False))
    finally:
        safe_shutdown("4200A", lambda: shutdown_4200a(k4200))
        safe_shutdown("temperature controller", lambda: shutdown_temperature_controller(temp_ctrl))


if __name__ == "__main__":
    main()
