#!/usr/bin/env python3
"""
Lock-in resistance vs. temperature log — Keithley 6221 AC + dual MFLI + MercuryiTC (read only)
=============================================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-10-29

The free-drift R(T) log of dc/dc_rt_log.py, read by the two MFLIs instead
of the 2182, so two voltage pairs are logged at once: typically R_xy on the
leader and R_xx on the follower (both at 1f; each MFLI's harmonic and R_xx
column naming are the caller's choice, exactly as in HARM6).

Wiring
------
    Keithley 6221 (WAVE, sine, continuous)   HI ──▶ I+ pad ;  LO ──▶ I- pad
      Trigger Link phase marker ──▶ split, equal cables ──▶ Aux In 1 on BOTH MFLIs

    Leader MFLI    Signal Input 1 (differential) ──▶ e.g. the R_xy (Hall) pair
    Follower MFLI  Signal Input 1 (differential) ──▶ e.g. the R_xx pair

    MDS: Leader Clock 10 MHz Out ──▶ Follower Clock In; Trigger Out 1 ──▶ Trigger In 1 on both

    MercuryiTC (optional, read only) ── 1 or 2 sensors, read before and after every sample
    Kepco BOP-GL + Lake Shore 475 (optional fixed field) ── parked, B read after every sample

Method
------
Both MFLIs are ExtRef-locked to the 6221 marker (see mfli_dual_harmonic_6221.py
for why BOTH must be). The demodulator filters are given `settle_s` once,
before the first sample; after that every sample is one paired poll window
(acquire_averaged_pair), bracketed by the two temperature reads of
dc_rt_log.run_log() — the shared loop that also records the drift, the field
and the field direction, and stops on Stop / max duration / T_stop.

Per sample, besides HARM6's demod columns (`<p>_X_V`, `_Y_V`, `_R_V`,
`_theta_deg`, `_R_sem_V`, `_n_samples`, `_overload`), `<p>_X_ohm` = X / I_rms
is the in-phase resistance (I_rms = 6221 peak / √2; the MFLI's X is RMS). Its
sign is only right with the demod phase set to harmonic × φ_I (lockin-phase
policy, mfli_dual_harmonic.py). The PLL / MDS state is re-checked every
sample and tagged on the row, never a reason to stop.
"""

import logging
import math
import threading
from typing import Callable, List, Optional

import pandas as pd
import zhinst.core as zi

from dc.dc_rt_log import AcquisitionConfig, _interruptible_sleep, run_log
from instruments.keithley6221 import ACSourceConfig
from instruments.lakeshore475 import GaussmeterConfig, LakeShore475
from instruments.mercury_itc import MercuryITC, TemperatureControllerConfig
from instruments.mfli_daq import ExtRefConfig, acquire_averaged_pair, check_mds_status, check_reference_locked
from mfli.mfli_dual_harmonic import DemodConfig
from mfli.mfli_dual_harmonic_6221 import _check_ac_safety

log = logging.getLogger(__name__)


def demod_columns(prefix: str, d: dict, I_rms_A: float) -> dict:
    """One demod's columns (HARM6's names) plus the in-phase resistance."""
    return {
        f"{prefix}_X_V":       d["x_mean"],
        f"{prefix}_Y_V":       d["y_mean"],
        f"{prefix}_R_V":       d["r_mean"],
        f"{prefix}_theta_deg": d["theta_mean"],
        f"{prefix}_R_sem_V":   d["r_sem"],
        f"{prefix}_n_samples": d["n_samples"],
        f"{prefix}_overload":  d["overload"],
        f"{prefix}_X_ohm":     d["x_mean"] / I_rms_A,
    }


def run_measurement(
    daq:        zi.ziDAQServer,
    ac_cfg:     ACSourceConfig,
    leader_extref_cfg:   ExtRefConfig,
    follower_extref_cfg: ExtRefConfig,
    demod1_cfg: DemodConfig,
    demod2_cfg: DemodConfig,
    acq_cfg:    AcquisitionConfig,
    n_averages: int,
    settle_s:   float,
    stop_event: Optional[threading.Event] = None,
    on_point:   Optional[Callable[[dict], None]] = None,
    temp_ctrl:  Optional[MercuryITC] = None,
    temp_cfg:   Optional[TemperatureControllerConfig] = None,
    write_csv:  Optional[Callable[[List[dict]], None]] = None,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg:  Optional[GaussmeterConfig] = None,
    magnet_current_A: Optional[float] = None,
    field_theta_deg:  Optional[float] = None,
    field_phi_deg:    Optional[float] = None,
    mds=None,
    demod1_label: str = "1f",
    demod2_label: str = "rxx_1f",
) -> pd.DataFrame:
    """Settle the lock-in filters once, then log paired MFLI samples until
    Stop / max duration / T_stop (dc_rt_log.run_log). The 6221 is already
    running (connect_ac_source) and both PLLs armed by the caller."""
    _check_ac_safety(ac_cfg)
    I_rms = ac_cfg.amplitude_A / math.sqrt(2.0)
    log.info("Settling the lock-in filters %.2f s before the first sample …", settle_s)
    _interruptible_sleep(settle_s, stop_event)

    def acquire() -> dict:
        mds_synced = check_mds_status(mds) if mds is not None else None
        if mds_synced is False:
            log.error("   MDS sync has dropped — check Ref/Trigger cabling.")
        leader_locked = check_reference_locked(daq, leader_extref_cfg)
        follower_locked = check_reference_locked(daq, follower_extref_cfg)
        for role, locked in (("Leader", leader_locked), ("Follower", follower_locked)):
            if locked is False:
                log.error("   %s ExtRef PLL has dropped lock — its data may be corrupted "
                          "until it relocks. Check the marker cabling.", role)
        d1, d2 = acquire_averaged_pair(daq, demod1_cfg, demod2_cfg, n_averages)
        for label, d in ((demod1_label, d1), (demod2_label, d2)):
            if d["overload"]:
                log.warning("   %s input is OVERLOADED — this reading is not trustworthy.", label)
        return {
            "mds_synced": mds_synced,
            "leader_reference_locked": leader_locked,
            "follower_reference_locked": follower_locked,
            "excitation_current_A_peak": ac_cfg.amplitude_A,
            **demod_columns(demod1_label, d1, I_rms),
            **demod_columns(demod2_label, d2, I_rms),
        }

    return run_log(acquire, acq_cfg, stop_event, on_point, temp_ctrl, temp_cfg, write_csv,
                   gaussmeter, gauss_cfg, magnet_current_A, field_theta_deg, field_phi_deg)
