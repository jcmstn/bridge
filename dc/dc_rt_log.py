#!/usr/bin/env python3
"""
DC resistance vs. temperature log — Keithley 6221 + Keithley 2182 + MercuryiTC (read only)
===========================================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-25

Logs a resistance continuously while the temperature drifts on its own. There
is no temperature control here: the MercuryiTC is only READ, and each sample
is stamped with the temperature at the moment it was taken. The cooling or
warming rate is set by hand on the cryostat; the program just follows.

Wiring
------
    Keithley 6221 (current source)
      Output (current) ──▶ current path through the sample ── common ground

    Keithley 2182 (nanovoltmeter)
      Channel 1 (differential) ──▶ the voltage probe pair of interest

    Oxford Instruments MercuryiTC (optional, read only)
      LAN ──▶ 1 or 2 temperature sensors, read before and after every sample

Method
------
The conventional R(T) technique for a slow, uncontrolled sweep: a FIXED small
sense current, reversed ±I over `n_reversals` pairs per sample (the 6221/2182
"delta" technique, instruments.keithley6221.acquire_reversal_averaged_voltage),
logged back to back for as long as the run lasts:

  1. Read T (every configured sensor)                  → T_before
  2. ±I reversal read: V_odd (the resistive signal, thermal EMFs cancelled)
     and V_even (offset + any even-in-current signal) — see
     docs/current-reversal.md
  3. Read T again                                      → T_after
  4. R = V_odd / I. The recorded temperature is (T_before + T_after) / 2, and
     `temperature_N_drift_K` = T_after − T_before is how much the temperature
     moved while that point was measured, over `sample_duration_s`, so the
     validity of each point can be judged afterwards (e.g. discard points
     whose drift exceeds the resolution needed).
  5. Wait out the rest of `interval_s` (0 = next sample immediately).

A sample is kept short on purpose: the shorter it is, the less the
temperature moves during it. The sample time is ≈ 2·n_reversals·(source delay
+ 2182 read) plus two temperature reads. A full I–V sweep per point was not
used, because it takes several times longer and so smears each point over a
wider temperature window. Check ohmicity separately with dc_iv_curve.py at a
few fixed temperatures if needed; V_even is recorded here on every sample.

The run ends on whichever comes first: Stop from the UI, `max_duration_s`, or
the first sensor's temperature crossing `T_stop_K` (in either direction,
relative to where the run started).

Requirements:
    pip install pymeasure pyvisa numpy pandas matplotlib
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from pymeasure.instruments.keithley import Keithley2182, Keithley6221

from instruments.keithley6221 import (
    SourceConfig,
    acquire_reversal_averaged_voltage,
    connect_source,
    ramp_current_to_zero,
    shutdown_source,
)
from instruments.keithley2182 import VoltmeterConfig, connect_voltmeter
from instruments.mercury_itc import (
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    probe_temperature_sensors,
    read_temperature,
    shutdown_temperature_controller,
)

from dc.dc_sweep_utils import safe_shutdown

# Data lives outside "bridge" (a sibling of it).
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  ── SourceConfig (fixed sense current) / VoltmeterConfig come
# from instruments/ and are re-exported from here for dc_rt_log_tui.py
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AcquisitionConfig:
    """What one sample is, and when the log stops."""
    n_reversals: int         = 5        # ±I pairs averaged per sample
    interval_s: float        = 0.0      # start-to-start sample spacing; 0 = back to back [s]
    max_duration_s: float    = 4 * 3600.0   # safety stop for an unattended run [s]
    T_stop_K: Optional[float] = None    # stop once sensor 1 crosses this (either direction) [K]
    save_every_s: float      = 10.0     # raw-file rewrite cadence while running [s]
    output_file: str         = "dc_rt_log.csv"


def _interruptible_sleep(seconds: float, stop_event: Optional[threading.Event]) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if stop_event is not None and stop_event.is_set():
            return
        time.sleep(min(0.2, end - time.monotonic()))


def _mean(a: Optional[float], b: Optional[float]) -> Optional[float]:
    vals = [v for v in (a, b) if v is not None]
    return sum(vals) / len(vals) if vals else None


def _drift(before: Optional[float], after: Optional[float]) -> Optional[float]:
    return after - before if before is not None and after is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    source:     Keithley6221,
    voltmeter:  Keithley2182,
    src_cfg:    SourceConfig,
    acq_cfg:    AcquisitionConfig,
    stop_event: Optional[threading.Event] = None,
    on_point:   Optional[Callable[[dict], None]] = None,
    temp_ctrl:  Optional[MercuryITC] = None,
    temp_cfg:   Optional[TemperatureControllerConfig] = None,
    write_csv:  Optional[Callable[[List[dict]], None]] = None,
) -> pd.DataFrame:
    """
    Take ±I reversal samples back to back (or every `acq_cfg.interval_s`)
    until `stop_event` is set, `acq_cfg.max_duration_s` has elapsed, or
    sensor 1 crosses `acq_cfg.T_stop_K`. Returns every sample taken.

    `temp_ctrl=None` (no MercuryiTC, or none of its sensors answered) is not
    an error: the temperature columns stay empty and the log runs against
    time only.

    `write_csv` (from data_naming.make_incremental_writer) is called at most
    every `acq_cfg.save_every_s` — the caller (record_run) writes the final
    file once the loop returns. Without it, a plain CSV is written to
    `acq_cfg.output_file` on the same cadence and at the end.
    """
    I = src_cfg.sense_current_A
    records: List[dict] = []
    t0 = time.monotonic()
    last_save = -math.inf
    start_above: Optional[bool] = None   # sensor 1 above T_stop_K at the first sample?

    def _save() -> None:
        if write_csv is not None:
            write_csv(records)
        else:
            Path(acq_cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(acq_cfg.output_file, index=False)

    def _read_T():
        return read_temperature(temp_ctrl, temp_cfg) if temp_cfg is not None else (None, None)

    while not (stop_event is not None and stop_event.is_set()):
        if time.monotonic() - t0 >= acq_cfg.max_duration_s:
            log.info("Maximum duration (%.0f s) reached — stopping.", acq_cfg.max_duration_s)
            break

        t_start = time.monotonic()
        t1_before, t2_before = _read_T()
        v = acquire_reversal_averaged_voltage(
            source, voltmeter, I, acq_cfg.n_reversals, stop_event,
            source_delay_s=src_cfg.source_delay_s)
        t1_after, t2_after = _read_T()
        t_end = time.monotonic()

        T1 = _mean(t1_before, t1_after)
        record = {
            "point_index":           len(records),
            "timestamp":             time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s":             (t_start + t_end) / 2 - t0,
            "sample_duration_s":     t_end - t_start,
            "temperature_1_K":       T1,
            "temperature_2_K":       _mean(t2_before, t2_after),
            "temperature_1_drift_K": _drift(t1_before, t1_after),
            "temperature_2_drift_K": _drift(t2_before, t2_after),
            "current_A":             I,
            "voltage_V":             v["mean"],
            "voltage_sem_V":         v["sem"],
            "voltage_even_V":        v["even_mean"],
            "voltage_even_sem_V":    v["even_sem"],
            "resistance_ohm":        v["mean"] / I,
            "resistance_sem_ohm":    v["sem"] / abs(I),
            "n_reversals":           v["n_reversals"],
        }
        records.append(record)
        log.info("#%d  T1=%s K  R=%.6g Ω  (%.2f s)", record["point_index"] + 1,
                 f"{T1:.3f}" if T1 is not None else "—", record["resistance_ohm"],
                 record["sample_duration_s"])
        if on_point is not None:
            on_point(record)

        # ponytail: whole-file rewrite every save_every_s (the shared writer has
        # no append mode); an append-mode writer if multi-day logs get slow.
        if t_end - last_save >= acq_cfg.save_every_s:
            _save()
            last_save = t_end

        if acq_cfg.T_stop_K is not None and T1 is not None:
            above = T1 > acq_cfg.T_stop_K
            if start_above is None:
                start_above = above
            elif above != start_above:
                log.info("Sensor 1 crossed %.3f K (now %.3f K) — stopping.", acq_cfg.T_stop_K, T1)
                break

        remaining = acq_cfg.interval_s - (time.monotonic() - t_start)
        if remaining > 0:
            _interruptible_sleep(remaining, stop_event)

    if write_csv is None and records:
        _save()
    log.info("Log finished: %d samples over %.0f s.", len(records), time.monotonic() - t0)
    return pd.DataFrame(records)


def plot_results(df: pd.DataFrame, out_path: Path, note: str = "") -> None:
    """R vs temperature (sensor 1; vs time when there is no temperature) and
    R vs time. `note` is printed small at the bottom (the operator comment)."""
    import matplotlib.pyplot as plt     # only the plot needs it — keeps imports light
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    minutes = df["elapsed_s"] / 60.0
    T = df["temperature_1_K"].astype(float) if "temperature_1_K" in df else None
    if T is not None and T.notna().any():
        ax1.plot(T, df["resistance_ohm"], ".", ms=3, color="#2E3192")
        ax1.set_xlabel("Temperature, sensor 1 (K)")
    else:
        ax1.plot(minutes, df["resistance_ohm"], ".", ms=3, color="#2E3192")
        ax1.set_xlabel("Time (min) — no temperature recorded")
    ax1.set_ylabel("R = V_odd / I (Ω)")
    ax1.set_title("Resistance vs. temperature")
    ax1.grid(alpha=0.4)

    ax2.plot(minutes, df["resistance_ohm"], ".", ms=3, color="#e34948")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("R (Ω)")
    ax2.set_title("Resistance vs. time")
    ax2.grid(alpha=0.4)
    fig.tight_layout()
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color="0.4", va="bottom")
        fig.subplots_adjust(bottom=0.1)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved plot: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── configure your devices here ─────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    src_cfg = SourceConfig(visa_resource="GPIB0::20::INSTR", sense_current_A=1e-4,
                           compliance_V=2.0, source_delay_s=0.05)
    volt_cfg = VoltmeterConfig(visa_resource="GPIB0::7::INSTR", nplc=1, auto_range=True)
    temp_cfg = TemperatureControllerConfig(
        visa_resource="TCPIP0::192.168.1.5::7020::SOCKET",   # ← your iTC's address
        sensor_uids=("MB1.T1",),                             # ← 1 or 2 sensor UIDs
    )
    acq_cfg = AcquisitionConfig(
        n_reversals=5, interval_s=0.0, max_duration_s=4 * 3600.0,
        output_file=str(_DATA_DIR / f"dc_rt_log_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    source = voltmeter = temp_ctrl = None
    try:
        source = connect_source(src_cfg)
        voltmeter = connect_voltmeter(volt_cfg)
        temp_ctrl = connect_temperature_controller(temp_cfg)
        temp_cfg = probe_temperature_sensors(temp_ctrl, temp_cfg)
        df = run_measurement(source, voltmeter, src_cfg, acq_cfg,
                             temp_ctrl=temp_ctrl if temp_cfg is not None else None,
                             temp_cfg=temp_cfg)
        if not df.empty:
            plot_results(df, Path(acq_cfg.output_file).with_suffix(".png"))
    finally:
        if source is not None:
            safe_shutdown("6221 (ramp)", lambda: ramp_current_to_zero(source))
            safe_shutdown("6221", lambda: shutdown_source(source))
        safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


if __name__ == "__main__":
    main()
