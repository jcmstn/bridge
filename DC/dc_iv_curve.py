#!/usr/bin/env python3
"""
DC I-V Curve — Keithley 6221 (current source) + Keithley 2182 (nanovoltmeter)
================================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-04

Sweeps a DC current with the 6221 and records the DC voltage response with
the 2182 at each point — the direct, single-frequency-free way to see
whether a device is ohmic (straight line through the origin) or not
(contacts, tunnel junctions, diodes, gated 2D systems, self-heating).

Wiring
------
    Keithley 6221 (current source)
      Output (current) ──▶ DUT ── common ground

    Keithley 2182 (nanovoltmeter)
      Channel 1 (differential) ──▶ across the DUT itself (2-terminal), or
      across the inner voltage-sense leads (4-terminal / Kelvin)

Method
------
For each current point:
  1. Set the 6221's source current.
  2. Settle (source_delay_s handles the 6221's own settling; add more via
     acq_cfg.settling_time_s for a DUT/wiring with its own slow response).
  3. Read `n_averages` voltage samples from the 2182 and average them.
  4. Chord resistance R = V / I is recorded alongside V and I.

The whole sweep is bidirectional (min → max → min) so hysteresis (charge
traps, self-heating, memristive contacts) shows up directly — the up and
down traces should sit on top of each other for a clean, purely resistive
DUT. A numerical differential resistance dV/dI (via np.gradient) is
computed once the full sweep is in hand — see plot_results() — since it
needs neighbouring points and can't be produced incrementally per-point
the way chord R can.

Practical guidance:
  - Keep compliance_V high enough to reach the expected voltage at
    current_max_A, or the sweep will silently clip against compliance
    well below the requested current.
  - This is a genuinely single-shot DC measurement (no lock-in rejection
    of pickup/drift) — for a small-signal AC alternative with much better
    noise rejection, see mfli_diff_resistance_vs_bias.py.

Requirements:
    pip install pymeasure pyvisa numpy pandas matplotlib
"""

import time
import logging
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

from pymeasure.instruments.keithley import Keithley2182, Keithley2400, Keithley6221

from instruments.keithley6221 import (
    connect as connect_6221,
    shutdown_source,
    ramp_current_to_zero,
)
from instruments.keithley2182 import (
    VoltmeterConfig,
    connect_voltmeter,
    acquire_averaged_voltage,
)
from instruments.keithley2400 import (
    GateConfig,
    connect_gate,
    set_gate_voltage,
    shutdown_gate,
)
from instruments.mercury_itc import (
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    shutdown_temperature_controller,
)

from dc.dc_sweep_utils import linear_sweep

# Data lives outside "bridge" (a sibling of it).
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
# VoltmeterConfig and GateConfig live in instruments/ (see the imports
# above). SourceConfig stays local: unlike the other DC programs (which
# source a fixed sense current), this one sweeps the current itself, so it
# needs sweep bounds instead of a single sense_current_A.

@dataclass
class SourceConfig:
    """Keithley 6221 — DC current source, sweeping current_min_A..current_max_A."""
    visa_resource: str     = "GPIB0::20::INSTR"
    compliance_V: float    = 2.0      # Voltage compliance [V]
    source_delay_s: float  = 0.05     # 6221's own settle time after each step [s]
    current_min_A: float   = -1e-3    # Sweep lower bound [A]
    current_max_A: float   =  1e-3    # Sweep upper bound [A]


@dataclass
class AcquisitionConfig:
    """Timing and averaging parameters."""
    settling_time_s: float = 0.2      # Extra dead-time after a current step [s]
    n_averages: int        = 5        # Voltage samples averaged per point
    output_file: str       = "dc_iv_curve.csv"


@dataclass
class CurrentPoint:
    """One point in the DC current sweep."""
    current_A: float
    settling_override_s: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Instrument setup helpers
# ─────────────────────────────────────────────────────────────────────────────
# connect_voltmeter, connect_gate, set_gate_voltage, shutdown_gate,
# shutdown_source and ramp_current_to_zero come from instruments/ unchanged.
# Only connect_source (sweep bounds, not the shared fixed-sense-current
# shape) and set_current (the sweep-range guard) are specific here.

def connect_source(cfg: SourceConfig) -> Keithley6221:
    """Open and configure the Keithley 6221 as a DC current source, starting at 0 A."""
    source = connect_6221(cfg.visa_resource, cfg.compliance_V, cfg.source_delay_s,
                           initial_current_A=0.0)
    log.info("Keithley 6221 sweep range: [%.4g, %.4g] A", cfg.current_min_A, cfg.current_max_A)
    return source


def set_current(source: Keithley6221, cfg: SourceConfig, current_A: float) -> None:
    """Set the sourced current, refusing to step outside the configured sweep range."""
    if not (cfg.current_min_A - 1e-15 <= current_A <= cfg.current_max_A + 1e-15):
        raise ValueError(
            f"Requested current {current_A:.4g} A is outside the configured sweep "
            f"range [{cfg.current_min_A}, {cfg.current_max_A}] A — refusing to set it."
        )
    source.source_current = current_A


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    source:     Keithley6221,
    voltmeter:  Keithley2182,
    src_cfg:    SourceConfig,
    acq_cfg:    AcquisitionConfig,
    points:     List[CurrentPoint],
    stop_event: Optional[threading.Event] = None,
    on_point:   Optional[Callable[[dict], None]] = None,
    gate_voltage_V: Optional[float] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg:  Optional[TemperatureControllerConfig] = None,
) -> pd.DataFrame:
    """
    Iterate over `points`, set each current, acquire the averaged voltage,
    log to CSV.

    `stop_event`, if given, is checked before each point — set it to break
    out of the sweep early (e.g. from a UI abort button) while still
    returning the data collected so far, so callers can run their normal
    ramp-down/shutdown path instead of killing the process outright.

    `on_point`, if given, is called with each point's `record` dict right
    after it's appended — lets a caller (e.g. a live TUI) show progress
    without polling the output CSV.

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

        log.info("── Point %d / %d   I=%.4g A ──────────────────", idx + 1, len(points), pt.current_A)

        # ── 1. Apply current ────────────────────────────────────────────────
        set_current(source, src_cfg, pt.current_A)

        # ── 2. Settle ────────────────────────────────────────────────────────
        settle = pt.settling_override_s if pt.settling_override_s is not None \
                 else acq_cfg.settling_time_s
        if settle > 0:
            time.sleep(settle)

        # ── 3. Acquire voltage ───────────────────────────────────────────────
        v = acquire_averaged_voltage(voltmeter, acq_cfg.n_averages)
        r_chord = v["mean"] / pt.current_A if pt.current_A != 0 else float("nan")
        log.info("   V=%.4e V  σ=%.2e V  R=%.5g Ω", v["mean"], v["std"], r_chord)

        # ── 3b. Read temperature (MercuryiTC, optional) ─────────────────────
        temp_1_K, temp_2_K = read_temperature(temp_ctrl, temp_cfg) \
            if temp_cfg is not None else (None, None)

        # ── 4. Build record ────────────────────────────────────────────────────
        record = {
            "point_index":     idx,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "current_A":       pt.current_A,
            "voltage_V":       v["mean"],
            "voltage_std_V":   v["std"],
            "resistance_ohm":  r_chord,
            "gate_voltage_V":  gate_voltage_V,
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


def plot_results(df: pd.DataFrame, out_path: Path) -> None:
    """
    Two-panel summary: the raw I-V curve, and the numerical differential
    resistance dV/dI (via np.gradient — needs neighbouring points, so it's
    computed here rather than incrementally during the sweep).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    ax1.plot(df["current_A"], df["voltage_V"], ".-", color="#2E3192")
    ax1.set_xlabel("Current (A)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("I-V curve")
    ax1.grid(alpha=0.4)

    r_diff = np.gradient(df["voltage_V"].to_numpy(), df["current_A"].to_numpy())
    ax2.plot(df["current_A"], r_diff, ".-", color="#e34948")
    ax2.set_xlabel("Current (A)")
    ax2.set_ylabel("dV/dI (Ω)")
    ax2.set_title("Differential resistance (numerical dV/dI)")
    ax2.grid(alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info("Saved plot: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── configure your devices and sweep here ──────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Source & voltmeter ───────────────────────────────────────────────────
    src_cfg = SourceConfig(
        visa_resource  = "GPIB0::12::INSTR",
        compliance_V   = 2.0,
        source_delay_s = 0.05,
        current_min_A  = -1e-3,
        current_max_A  =  1e-3,
    )
    source = connect_source(src_cfg)

    volt_cfg = VoltmeterConfig(
        visa_resource = "GPIB0::7::INSTR",
        nplc          = 5,
        auto_range    = True,
    )
    voltmeter = connect_voltmeter(volt_cfg)

    # ── Temperature (Oxford Instruments MercuryiTC, optional) ────────────────
    # Not every rig has one, and not every MercuryiTC has two probes wired up
    # — connect_temperature_controller() returns None rather than raising if
    # it can't be reached, and the measurement runs fine either way.
    temp_cfg = TemperatureControllerConfig(
        visa_resource = "TCPIP0::192.168.1.5::7020::SOCKET",  # ← set to your iTC's address
        sensor_uids   = ("MB1.T1",),   # ← 1 or 2 board UIDs, e.g. ("MB1.T1", "DB5.T1")
    )
    temp_ctrl = connect_temperature_controller(temp_cfg)

    # ── Acquisition settings ─────────────────────────────────────────────────
    acq_cfg = AcquisitionConfig(
        settling_time_s = 0.2,
        n_averages      = 5,
        output_file     = str(_DATA_DIR / f"dc_iv_curve_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    # ── Current sweep points ───────────────────────────────────────────────────
    # Bidirectional (up then down) so hysteresis is visible.
    currents_A = linear_sweep(src_cfg.current_min_A, src_cfg.current_max_A, step=5e-5, bidirectional=True)
    points = [CurrentPoint(current_A=float(i)) for i in currents_A]

    # ── Run ──────────────────────────────────────────────────────────────────
    # Always ramp the current back to zero and disable the output, even if
    # the measurement raises partway through.
    try:
        df = run_measurement(source, voltmeter, src_cfg, acq_cfg, points,
                              temp_ctrl=temp_ctrl, temp_cfg=temp_cfg)
        print("\n", df.to_string(index=False))
        plot_path = Path(acq_cfg.output_file).with_suffix(".png")
        plot_results(df, plot_path)
        plt.show()
    finally:
        ramp_current_to_zero(source)
        shutdown_source(source)
        shutdown_temperature_controller(temp_ctrl)


if __name__ == "__main__":
    main()
