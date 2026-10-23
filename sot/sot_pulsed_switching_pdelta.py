#!/usr/bin/env python3
"""
SOT pulsed switching, 6221 + 2182A only — Pulse Delta write pulse + delayed
Delta-mode R_xy read (NO 4200A, no MFLI)
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-10-23

Same switching curve as sot_pulsed_switching.py (one write pulse per
amplitude, delayed R_xy read, amplitude swept up and down at a fixed tilted
field) — but the 6221 does both jobs, in its two 2182A-linked modes:

  1. Pulse Delta, count 1, I-Lo = 0  →  ONE 0 → I_pulse → 0 write pulse
     (``instruments/keithley6221.py::fire_pdelta_pulse``). The 2182A reading
     Pulse Delta takes during it is never fetched.
  2. wait ``delay_after_pulse_s`` (on top of the mode switch below)
  3. Delta  →  the 6221 alternates ±I_read and the 2182A reads V_xy each
     side; the 6221 returns ``n_delta_readings`` delta voltages (V_odd —
     cancels thermal EMF and the 2182A offset) from its buffer
     (``acquire_delta_voltage``). R_xy = V_odd / I_read.

Switching between the modes is SOUR:SWE:ABOR → configure → ARM; arming makes
the 6221 configure the 2182A over RS-232, so there is a hardware wait (order
of a second, not measured yet) between the pulse and the read. Both helpers
poll the 6221 buffer, so the read is always complete before the next pulse.

The pulse is µs–ms (Pulse Delta: 50 µs – 12 ms), line-synchronised. Rise time
is not programmable: ``OUTP:RESP FAST`` (≈ 2-3 µs) or ``SLOW`` (≈ 100 µs),
plus the analog filter for a high-impedance load. Joule heating ∝ I²R·t — a
current that is safe as a 100 ns PMU pulse can cook the device at 1 ms.

Wiring
------
    Keithley 6221   HI ──▶ I+ pad ;  LO ──▶ I- pad   (pulse AND read current)
    Keithley 2182A  Ch 1 ──▶ transverse (Hall) arms
    2182A RS-232 ◀──null-modem──▶ 6221 RS-232 ;  8501 Trigger Link 2182A ↔ 6221
      2182A interface = RS-232 (this turns its GPIB off — only the 6221 is on
      the bus). Trigger Link must be the 2182A factory default (EXT TRIG =
      line 2, VMC = line 1). Set the 2182A voltage range on its front panel.
    Kepco BOP-GL   ──GPIB──▶ electromagnet (ONE static tilted field)
    Lake Shore 475 ──GPIB──▶ Gaussmeter probe at the sample

Bench-verify once: SOUR:PDEL:NVPR? = 1; scope one pulse (width, sign,
FAST/SLOW + filter edge); time the Pulse Delta → Delta switch.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from pymeasure.instruments.keithley import Keithley6221

from instruments.keithley6221 import (
    PulseDeltaConfig,
    acquire_delta_voltage,
    check_2182a_on_serial,
    fire_pdelta_pulse,
)
from instruments.lakeshore475 import GaussmeterConfig, LakeShore475, read_field_mT
from instruments.mercury_itc import MercuryITC, TemperatureControllerConfig, read_temperature
from sot.sot_pulsed_switching_6221 import (
    PulsePoint,
    _check_pulse_currents,
    _check_read_safety,
    _interruptible_sleep,
)

log = logging.getLogger(__name__)


@dataclass
class ReadConfig:
    """The Delta-mode R_xy read, plus the wait before it."""
    sense_current_A: float      = 1e-4   # ±I_read [A]
    compliance_V: float         = 2.0    # read compliance [V]
    n_delta_readings: int       = 5      # delta readings averaged per read
    delta_delay_s: float        = 0.05   # SOUR:DELT:DEL — settle after each flip [s]
    nplc: int                   = 5      # 2182A rate (integer PLC)
    delay_after_pulse_s: float  = 1.0    # extra wait after the pulse, on top of the mode switch [s]


def _source_quiet(source: Keithley6221) -> None:
    """Disarm whichever mode is running and turn the output off."""
    source.write("SOUR:SWE:ABOR")
    source.disable_source()


def connect_pdelta_source(visa_resource: str) -> Keithley6221:
    """Open + reset the 6221 and check it sees the 2182A on RS-232."""
    source = Keithley6221(visa_resource)
    source.reset()
    check_2182a_on_serial(source)
    log.info("Keithley 6221 connected: %s — 2182A on RS-232 (Pulse Delta / Delta)", visa_resource)
    return source


def run_measurement(
    source: Keithley6221,
    pulse_cfg: PulseDeltaConfig,
    read_cfg: ReadConfig,
    points: List[PulsePoint],
    stop_event: Optional[threading.Event] = None,
    on_point: Optional[Callable[[dict], None]] = None,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg: Optional[GaussmeterConfig] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg: Optional[TemperatureControllerConfig] = None,
    magnet_current_A: Optional[float] = None,
    field_theta_deg: Optional[float] = None,
    field_phi_deg: Optional[float] = None,
    write_csv: Optional[Callable[[List[dict]], None]] = None,
    output_file: str = "sot_pulsed_switching_pdelta.csv",
) -> pd.DataFrame:
    """One Pulse Delta write pulse + delayed Delta read per pulse current in
    ``points``. One row per pulse; CSV rewritten in full every row.
    ``temp_ctrl=None`` never stops the run; ``stop_event`` is checked before
    each pulse, in the post-pulse wait, and in both buffer polls."""
    _check_pulse_currents(points)
    _check_read_safety(read_cfg)

    field_measured_mT = None
    if gaussmeter is not None and gauss_cfg is not None:
        field_measured_mT = read_field_mT(gaussmeter, gauss_cfg)
        log.info("Static field: %.4f mT measured (magnet current %s A)",
                 field_measured_mT, magnet_current_A)

    source.write(f"CURR:COMP {read_cfg.compliance_V:g}")
    records: List[dict] = []
    try:
        for a_idx, pt in enumerate(points):
            if stop_event is not None and stop_event.is_set():
                log.info("Aborted after %d / %d pulses.", len(records), len(points))
                break

            pinfo = fire_pdelta_pulse(source, pulse_cfg, pt.pulse_current_A, stop_event)
            _interruptible_sleep(read_cfg.delay_after_pulse_s, stop_event)
            # the pulse set its own compliance — the read gets the read one back
            source.write(f"CURR:COMP {read_cfg.compliance_V:g}")
            rv = acquire_delta_voltage(source, read_cfg.sense_current_A,
                                       read_cfg.n_delta_readings, read_cfg.delta_delay_s,
                                       read_cfg.nplc, stop_event)
            r_xy = rv["mean"] / read_cfg.sense_current_A

            t1_K, t2_K = read_temperature(temp_ctrl, temp_cfg) if temp_cfg is not None else (None, None)
            record = {
                "amplitude_index":   a_idx,
                "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                "pulse_current_A":   pt.pulse_current_A,
                "pulse_width_s":     pulse_cfg.width_s,
                "output_response":   pulse_cfg.response,
                "analog_filter":     pulse_cfg.analog_filter,
                "pulse_completed":   pinfo["pulse_completed"],
                "sense_current_A":   read_cfg.sense_current_A,
                "hall_voltage_V":    rv["mean"],
                "hall_voltage_sem_V": rv["sem"],
                "hall_resistance_ohm": r_xy,
                "n_delta_readings":  rv["n_readings"],
                "magnet_current_A":  magnet_current_A,
                "assist_field_measured_mT": field_measured_mT,
                "field_theta_deg":   field_theta_deg,
                "field_phi_deg":     field_phi_deg,
                "field_frame":       "RH",  # right-handed (z=normal, x=current, y=z×x); see docs/data_convention.md
                "temperature_1_K":   t1_K,
                "temperature_2_K":   t2_K,
            }
            records.append(record)
            if on_point is not None:
                on_point(record)
            if write_csv is not None:
                write_csv(records)
            else:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(records).to_csv(output_file, index=False)
            log.info("pulse %d/%d  I_pulse=%.4g A  R_xy=%.6g Ω",
                     a_idx + 1, len(points), pt.pulse_current_A, r_xy)
    finally:
        _source_quiet(source)
    return pd.DataFrame(records)
