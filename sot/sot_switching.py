#!/usr/bin/env python3
"""
SOT switching — quasi-static DC current-staircase switching loop (Keithley 4200A)
================================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-07

Stage 3 (the "core measurement") of the SOT switching plan: at a fixed
in-plane assist field, sweep the channel current in a staircase from -I_max
to +I_max and back, reading the anomalous-Hall voltage V_xy after each step.
A jump in V_xy as the current crosses ±Ic is the switching event. Repeated
``n_repeats`` times per assist field for mean-Ic / loop-to-loop-spread
statistics.

The caller (TUI / main) loops the assist field over a list of values — one
output file per value — so a full Ic-vs-Hx phase diagram is one session.

Field geometry (single tilted electromagnet)
--------------------------------------------
One Kepco BOP-GL magnet at a fixed mount angle. The **magnet current** is the
knob (like every field program in ``dc/``) — the caller loops it over a list,
one output file per value; the true field is measured by the Lake Shore 475
and recorded as ``assist_field_measured_mT``. ``field_angle_from_oop_deg``
(0° = out-of-plane, 90° = in-plane) is the mount tilt — recorded on every
row, same convention as ``dc/dc_hall_measurement.py``. Flipping the magnet
current sign flips **both** the in-plane (Hx) and the small out-of-plane (Hz)
components together — they are not independently controllable with this
hardware.

Wiring
------
    4200A SMU1 (force current) ──▶ channel current leads of the Hall bar
    4200A SMU2 (force 0 A → voltmeter) ──▶ transverse (Hall) voltage leads
    Kepco BOP-GL  ──GPIB──▶ electromagnet coil (assist field, set per file)
    Lake Shore 475 ──GPIB──▶ Gaussmeter probe at the sample (measures the field)

    # ponytail: SMU2-as-voltmeter is high-impedance but far above 2182-grade
    # for offset/noise on a µV Hall signal. Fine for a first pass with the
    # ±field-sign contrast check below. Upgrade path: route V_xy to the 2182
    # or the MFLI lock-in and read that instead of SMU2.

Read modes
----------
  * ``at_write_current`` (default) — read V_xy at each staircase current.
    Current reversal is unavailable (the current *is* the axis); offset
    handling comes from the ±field-sign control instead.
  * ``write_then_read`` — after stepping to I_write and settling, drop SMU1
    to ``read_current_A`` (≪ Ic) and read V_xy there — a cleaner remanent-
    state readout. ``read_reversal`` then reverses ±read_current_A and takes
    the odd part (V_even also recorded).

Essential controls (run these, compare offline):
  * flip the field sign  → Hall contrast inverts, Ic(Hx) unchanged
  * a same-polarity-only current train → no spurious switching beyond noise

PMU-pulsed switching (Stage 4) is a planned follow-up — see
instruments/keithley4200a.py.

Requirements: pymeasure, pyvisa, numpy, pandas. KXCI enabled on the 4200A.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
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
from instruments.kepco_magnet import (
    MagnetConfig,
    connect_magnet,
    set_magnet_current,
    shutdown_magnet,
)
from instruments.lakeshore475 import (
    GaussmeterConfig,
    LakeShore475,
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

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

READ_MODES = ("at_write_current", "write_then_read")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StaircaseConfig:
    """The SMU1 channel-current staircase, one switching loop."""
    i_min_A: float = -20e-3
    i_max_A: float =  20e-3
    step_A: float  =  0.5e-3
    bidirectional: bool = True     # -I_max → +I_max → -I_max (a full loop)


@dataclass
class AcquisitionConfig:
    settling_time_s: float = 0.05     # dwell after each current step before reading V_xy [s]
    read_mode: str        = "at_write_current"
    read_current_A: float = 1e-3      # write_then_read only: the ≪Ic read current
    read_reversal: bool   = False     # write_then_read only: reverse ±read_current_A
    n_averages: int       = 5         # V_xy samples (or reversal pairs) per step
    n_repeats: int        = 5         # switching loops per assist field → repeat_index rows
    output_file: str      = "sot_switching.csv"


@dataclass
class CurrentPoint:
    current_A: float
    settling_override_s: Optional[float] = None
    set_action: Optional[Callable[[], None]] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop  (docs/architecture.md §3 contract)
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    k4200,
    smu_src_cfg: SMUChannelConfig,
    smu_hall_cfg: SMUChannelConfig,
    acq_cfg: AcquisitionConfig,
    points: List[CurrentPoint],
    stop_event: Optional[threading.Event] = None,
    on_point: Optional[Callable[[dict], None]] = None,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg: Optional[GaussmeterConfig] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg: Optional[TemperatureControllerConfig] = None,
    magnet_current_A: Optional[float] = None,
    field_angle_from_oop_deg: Optional[float] = None,
    write_csv: Optional[Callable[[List[dict]], None]] = None,
) -> pd.DataFrame:
    """Run ``acq_cfg.n_repeats`` staircase loops over ``points`` at a single,
    already-set assist field; read V_xy after each current step; write the CSV
    in full every point (crash-safe).

    The magnet current is set **once by the caller** before this is called
    (like ``gate_voltage_V`` in ``dc_iv_curve``). ``magnet_current_A`` is the
    nominal value recorded on every row; if a gaussmeter is passed the field
    is measured once here and recorded as ``assist_field_measured_mT``.

    ``stop_event`` checked before each point (+ mid-reversal). ``on_point``
    gets each record dict. ``temp_ctrl=None`` never stops the run.
    """
    if acq_cfg.read_mode not in READ_MODES:
        raise ValueError(f"read_mode must be one of {READ_MODES}, got {acq_cfg.read_mode!r}")

    field_measured_mT = None
    if gaussmeter is not None and gauss_cfg is not None:
        field_measured_mT = read_field_mT(gaussmeter, gauss_cfg)
        log.info("Assist field measured: %.4f mT (magnet current %.4f A)",
                 field_measured_mT, magnet_current_A if magnet_current_A is not None else float("nan"))

    records: List[dict] = []
    total = acq_cfg.n_repeats * len(points)

    for repeat in range(acq_cfg.n_repeats):
        for idx, pt in enumerate(points):
            if stop_event is not None and stop_event.is_set():
                log.info("Aborted after %d / %d points.", len(records), total)
                return pd.DataFrame(records)

            settle = pt.settling_override_s if pt.settling_override_s is not None \
                else acq_cfg.settling_time_s

            # ── write: step the channel current ────────────────────────────
            set_source_level(k4200, smu_src_cfg, pt.current_A)
            if settle > 0:
                time.sleep(settle)
            channel_V = read_measurement(k4200, smu_src_cfg)   # heating / compliance monitor

            # ── read: V_xy on SMU2 ────────────────────────────────────────
            v_even = v_even_sem = None
            if acq_cfg.read_mode == "write_then_read":
                if acq_cfg.read_reversal:
                    rv = acquire_reversal_averaged(
                        k4200, smu_src_cfg, smu_hall_cfg, acq_cfg.read_current_A,
                        acq_cfg.n_averages, stop_event, source_delay_s=settle)
                    v_xy, v_xy_sem = rv["mean"], rv["sem"]
                    v_even, v_even_sem = rv["even_mean"], rv["even_sem"]
                else:
                    set_source_level(k4200, smu_src_cfg, acq_cfg.read_current_A)
                    if settle > 0:
                        time.sleep(settle)
                    av = acquire_measurement(k4200, smu_hall_cfg, acq_cfg.n_averages, stop_event)
                    v_xy, v_xy_sem = av["mean"], av["sem"]
                i_for_r = acq_cfg.read_current_A
            else:   # at_write_current
                av = acquire_measurement(k4200, smu_hall_cfg, acq_cfg.n_averages, stop_event)
                v_xy, v_xy_sem = av["mean"], av["sem"]
                i_for_r = pt.current_A

            r_xy = v_xy / i_for_r if i_for_r not in (0.0, None) else float("nan")
            t1_K, t2_K = read_temperature(temp_ctrl, temp_cfg) if temp_cfg is not None else (None, None)

            record = {
                "point_index":       idx,
                "repeat_index":      repeat,
                "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                "set_current_A":     pt.current_A,
                "read_current_A":    i_for_r,
                "channel_voltage_V": channel_V,
                "hall_voltage_V":    v_xy,
                "hall_voltage_sem_V": v_xy_sem,
                "hall_voltage_even_V":     v_even,
                "hall_voltage_even_sem_V": v_even_sem,
                "hall_resistance_ohm": r_xy,
                "magnet_current_A":  magnet_current_A,
                "assist_field_measured_mT": field_measured_mT,
                "field_angle_from_oop_deg": field_angle_from_oop_deg,
                "read_mode":         acq_cfg.read_mode,
                "n_averages":        acq_cfg.n_averages,
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

    log.info("Done. %d rows → '%s'", len(records), acq_cfg.output_file)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point  — one full Ic-vs-Hx phase diagram
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    k_cfg = Keithley4200AConfig(visa_resource="GPIB0::17::INSTR", integration="fast")
    src_cfg = SMUChannelConfig(channel=1, source_function="current",
                               compliance_voltage_V=3.0, source_limit_A=30e-3)
    hall_cfg = SMUChannelConfig(channel=2, source_function="current",  # forces 0 A
                                compliance_voltage_V=3.0, source_limit_A=1e-9)

    stair_cfg = StaircaseConfig(i_min_A=-15e-3, i_max_A=15e-3, step_A=0.3e-3)
    acq_cfg = AcquisitionConfig(
        read_mode="at_write_current", n_averages=5, n_repeats=5,
        output_file="",   # set per assist-field file below
    )

    magnet_cfg = MagnetConfig(visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
                              voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05)
    gauss_cfg = GaussmeterConfig(visa_resource="GPIB0::12::INSTR", unit="T", n_averages=10)
    temp_cfg = TemperatureControllerConfig(
        visa_resource="TCPIP0::192.168.1.5::7020::SOCKET", sensor_uids=("MB1.T1",))

    FIELD_ANGLE_FROM_OOP_DEG = 85.0        # ← SET TO YOUR REAL MOUNT ANGLE (deg from film
                                          #    normal). Recorded verbatim, never measured;
                                          #    85 here is a placeholder, not a measurement.
    ASSIST_MAGNET_CURRENTS_A = [-3.0, -1.5, -0.75, 0.0, 0.75, 1.5, 3.0]   # the phase-diagram axis

    k4200 = connect_4200a(k_cfg)
    configure_smu(k4200, src_cfg)
    configure_smu(k4200, hall_cfg)
    set_source_level(k4200, hall_cfg, 0.0)      # park SMU2 as the voltmeter

    magnet = connect_magnet(magnet_cfg)
    gaussmeter = connect_gaussmeter(gauss_cfg)
    temp_ctrl = connect_temperature_controller(temp_cfg)

    currents = linear_sweep(stair_cfg.i_min_A, stair_cfg.i_max_A, stair_cfg.step_A,
                            bidirectional=stair_cfg.bidirectional)

    try:
        for i_mag in ASSIST_MAGNET_CURRENTS_A:
            set_magnet_current(magnet, magnet_cfg, i_mag, gaussmeter, gauss_cfg)
            acq_cfg.output_file = str(
                _DATA_DIR / f"sot_switching_{i_mag:+.2f}A_{datetime.now():%Y%m%d_%H%M%S}.csv")
            points = [CurrentPoint(current_A=float(i)) for i in currents]
            run_measurement(k4200, src_cfg, hall_cfg, acq_cfg, points,
                            gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
                            temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
                            magnet_current_A=float(i_mag),
                            field_angle_from_oop_deg=FIELD_ANGLE_FROM_OOP_DEG)
    finally:
        safe_shutdown("4200A", lambda: shutdown_4200a(k4200))
        safe_shutdown("magnet", lambda: shutdown_magnet(magnet, magnet_cfg))
        safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        safe_shutdown("temperature controller", lambda: shutdown_temperature_controller(temp_ctrl))


if __name__ == "__main__":
    main()
