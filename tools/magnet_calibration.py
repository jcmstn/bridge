#!/usr/bin/env python3
"""
magnet_calibration.py — one-off magnet current → field calibration
==================================================================
Steps a Kepco bipolar supply (BOP 20-50GL or BOP 36-28MG, the same two the
spin-valve program offers) through a list of currents, reads the Lake Shore
475 gaussmeter at each point, and writes one CSV row per point.

Deliberately NOT part of the data convention: no sample, no index.csv, just
a plain CSV. Edit the USER SETTINGS block below, then run from the repo root:

    uv run tools/magnet_calibration.py

Ctrl-C at any time: the rows measured so far are already on disk, and the
magnet is ramped back to 0 A before the script exits.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from instruments.kepco_magnet import (
    MAGNET_MODELS,
    MagnetConfig,
    connect_magnet,
    set_magnet_current,
    shutdown_magnet,
)
from instruments.lakeshore475 import (
    GaussmeterConfig,
    connect_gaussmeter,
    field_to_mT,
    shutdown_gaussmeter,
)

# ═════════════════════════════════════════════════════════════════════════════
#  USER SETTINGS — edit these
# ═════════════════════════════════════════════════════════════════════════════

# ── Instruments ──────────────────────────────────────────────────────────────
MAGNET_MODEL    = "BOP 20-50GL"        # "BOP 20-50GL" (±50 A, ±20 V) or "BOP 36-28MG" (±28 A, ±36 V)
MAGNET_GPIB     = "GPIB0::6::INSTR"
GAUSSMETER_GPIB = "GPIB0::12::INSTR"

# ── Safety ───────────────────────────────────────────────────────────────────
CURRENT_LIMIT_A      = 35.0    # no sweep point may exceed |I| > this  [A]
VOLTAGE_COMPLIANCE_V = 15.0    # supply compliance voltage  [V]
RAMP_STEP_A          = 0.1     # magnet ramp step  [A]  (smaller = gentler on the coil)
RAMP_DELAY_S         = 0.05    # pause between ramp steps  [s]

# ── Field reading ────────────────────────────────────────────────────────────
GAUSS_UNIT          = "T"      # "T" or "G" (CSV is always in mT)
N_AVERAGES          = 20       # gaussmeter readings averaged per point
READ_DELAY_S        = 0.05     # pause between those readings  [s]
SETTLE_TOLERANCE_MT = 0.02     # field counts as settled when it drifts less than this  [mT]
EXTRA_WAIT_S        = 1.0      # extra dwell after the field has settled, before reading  [s]

# ── Sweep points [A] — measured in this exact order ──────────────────────────
SWEEP_A = [0, 1, 2, 5, 10, 15, 20, 25, 30]

# Examples (uncomment one):
# SWEEP_A = list(np.arange(0, 30.01, 1.0))                     # 0 → 30 A in 1 A steps
# I_MAX, STEP = 30.0, 1.0                                      # full hysteresis loop
# SWEEP_A = list(np.concatenate([
#     np.arange(0, I_MAX, STEP),                               #  0    → +Imax
#     np.arange(I_MAX, -I_MAX, -STEP),                         # +Imax → −Imax
#     np.arange(-I_MAX, I_MAX + STEP / 2, STEP),               # −Imax → +Imax
# ]))

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "magnet_calibration"   # <repo>/../data/...
OUTPUT_CSV = OUTPUT_DIR / f"{MAGNET_MODEL.replace(' ', '_')}_{datetime.now():%Y%m%d-%H%M%S}.csv"

# ═════════════════════════════════════════════════════════════════════════════

COLUMNS = ["index", "time_s", "I_set_A", "I_meas_A", "V_meas_V",
           "B_mT", "B_std_mT", "current_settled", "field_settled"]


def preflight() -> None:
    """Refuse bad settings before any instrument is touched."""
    model = MAGNET_MODELS.get(MAGNET_MODEL)
    if model is None:
        raise SystemExit(f"Unknown MAGNET_MODEL {MAGNET_MODEL!r}; pick one of {list(MAGNET_MODELS)}")
    if CURRENT_LIMIT_A > model.MAX_CURRENT:
        raise SystemExit(f"CURRENT_LIMIT_A {CURRENT_LIMIT_A:g} A exceeds the {MAGNET_MODEL} rating ({model.MAX_CURRENT:g} A)")
    if VOLTAGE_COMPLIANCE_V > model.MAX_VOLTAGE:
        raise SystemExit(f"VOLTAGE_COMPLIANCE_V {VOLTAGE_COMPLIANCE_V:g} V exceeds the {MAGNET_MODEL} rating ({model.MAX_VOLTAGE:g} V)")
    if not SWEEP_A:
        raise SystemExit("SWEEP_A is empty")
    bad = [i for i in SWEEP_A if abs(i) > CURRENT_LIMIT_A]
    if bad:
        raise SystemExit(f"Sweep points beyond ±{CURRENT_LIMIT_A:g} A: {bad}")


def main() -> None:
    preflight()
    print(f"{MAGNET_MODEL} @ {MAGNET_GPIB}, Lake Shore 475 @ {GAUSSMETER_GPIB}")
    print(f"{len(SWEEP_A)} points, {min(SWEEP_A):g} … {max(SWEEP_A):g} A  (limit ±{CURRENT_LIMIT_A:g} A, "
          f"compliance {VOLTAGE_COMPLIANCE_V:g} V)")
    print(f"Saving to {OUTPUT_CSV}")
    input("Enter to start, Ctrl-C to abort … ")

    mcfg = MagnetConfig(visa_resource=MAGNET_GPIB, model=MAGNET_MODEL,
                        current_limit_A=CURRENT_LIMIT_A, voltage_compliance_V=VOLTAGE_COMPLIANCE_V,
                        ramp_step_A=RAMP_STEP_A, ramp_delay_s=RAMP_DELAY_S)
    gcfg = GaussmeterConfig(visa_resource=GAUSSMETER_GPIB, unit=GAUSS_UNIT,
                            n_averages=N_AVERAGES, read_delay_s=READ_DELAY_S)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    gm = connect_gaussmeter(gcfg)
    psu = None
    I_done, B_done = [], []

    # Live plot for visual inspection only -- not saved.
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], "o-", ms=4)
    ax.set_xlabel("Magnet current I_meas (A)")
    ax.set_ylabel("Field B (mT)")
    ax.set_title(f"{MAGNET_MODEL} calibration")
    ax.grid(True)
    plt.show()

    try:
        psu = connect_magnet(mcfg)
        t0 = time.monotonic()
        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for n, I in enumerate(SWEEP_A):
                I = float(I)
                st = set_magnet_current(psu, mcfg, I, gaussmeter=gm, gauss_cfg=gcfg,
                                        field_settle_tolerance_mT=SETTLE_TOLERANCE_MT)
                time.sleep(EXTRA_WAIT_S)
                mean, std = gm.measure(N_AVERAGES, delay=READ_DELAY_S)
                B, B_std = field_to_mT(mean, GAUSS_UNIT), field_to_mT(std, GAUSS_UNIT)
                I_meas, V_meas = float(psu.measure_current), float(psu.measure_voltage)
                w.writerow([n, round(time.monotonic() - t0, 2), I, I_meas, V_meas,
                            B, B_std, st["current_settled"], st["field_settled"]])
                f.flush()
                I_done.append(I_meas)
                B_done.append(B)
                line.set_data(I_done, B_done)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)
                flag = "" if st["field_settled"] else "   (field NOT settled)"
                print(f"[{n + 1:>3}/{len(SWEEP_A)}]  I_set={I:8.3f} A  I_meas={I_meas:8.3f} A  "
                      f"V={V_meas:7.3f} V  B={B:10.3f} ± {B_std:.3f} mT{flag}")
    except KeyboardInterrupt:
        print("\nAborted — ramping magnet to zero.")
    finally:
        if psu is not None:
            shutdown_magnet(psu, mcfg)
        shutdown_gaussmeter(gm)

    print(f"Saved {len(I_done)} points to {OUTPUT_CSV}")
    if len(I_done) >= 2:
        # ponytail: single straight line over all points -- ignores remanence/hysteresis and saturation
        a, b = np.polyfit(I_done, B_done, 1)
        print(f"Linear fit: B = {a:.4f} mT/A · I + {b:.4f} mT")

    print("Close the plot window to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
