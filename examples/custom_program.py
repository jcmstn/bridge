#!/usr/bin/env python3
"""
Custom standalone measurement — resistance vs. temperature (copy-me template)
============================================================================
A single script that runs a bespoke measurement loop with NO TUI and NO web
app, while still saving into the standard per-sample data convention
(instruments/data_naming.py + docs/data_convention.md). Meant to be copied
to a scratch file and edited for a one-off measurement (e.g. on a visit to
another lab that has the same instruments).

What it does
------------
Uses ONE Keithley 2450 SourceMeter (SMU) as the whole electrical chain:
sources a fixed DC current, measures the resulting voltage in 4-wire
(remote-sense) mode so the lead resistance drops out. At each of a list of
MercuryiTC temperature setpoints it averages N_AVERAGES readings → one CSV
row per temperature, one raw file for the whole sweep, one index.csv row.

This is the "general SMU" wrapper in instruments/keithley2450.py
(``SMUConfig`` + ``connect_smu`` / ``set_source_level`` /
``acquire_measurement`` / ``shutdown_smu``); instruments/keithley2400.py
exposes the same names for a 2400, so swapping the SMU is an import change.

4-wire kills the lead resistance but not a thermoelectric offset in series
with the DUT. If that matters, reverse the current +I/-I and take the odd
part — see instruments/keithley6221.acquire_reversal_averaged_voltage and
docs/current-reversal.md rather than hand-rolling it here.

There is no R-vs-T suite in dc/ — this is the "roll your own loop" case. If
your measurement instead matches an existing suite, don't rewrite the loop:
import that module's `run_measurement()` and pass
`write_csv=make_incremental_writer(ctx.raw_path, ...)` — see
docs/architecture.md §7a.

How to run
----------
    # from the bridge repo root:
    uv run examples/custom_program.py
    # from anywhere else (bridge is editable-installed by `uv sync`, so
    # instruments.* / dc.* import fine from any cwd in that env):
    uv run --project /path/to/bridge /path/to/my_custom_program.py

Adding a NEW instrument
-----------------------
Drop `instruments/my_instr.py` with a `MyInstrConfig` dataclass +
`connect_my_instr()` / `shutdown_my_instr()` (+ optional `set_*` / `read_*`
/ `acquire_*`), mirroring any existing instruments/*.py. Nothing else needs
to know it exists — no TUI, no web, no suite-picker registration. Import it
here directly. Full function contract + skeleton + the load-bearing vs
nice-to-have failure policy: docs/architecture.md §4 "Adding an instrument".
"""

import time
from pathlib import Path

from instruments.data_naming import (
    allocate_run,
    ensure_sample,
    finalize_index_row,
    make_incremental_writer,
    write_record,
)
from instruments.keithley2450 import (
    SMUConfig,
    connect_smu,
    set_source_level,
    acquire_measurement,
    shutdown_smu,
)
from instruments.mercury_itc import (
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    set_temperature,
    shutdown_temperature_controller,
    wait_for_temperature_stable,
)
from dc.dc_sweep_utils import safe_shutdown

# ─────────────────────────────────────────────────────────────────────────────
# EDIT HERE — everything that changes between runs / labs
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = Path.home() / "measurements"   # parent of every {sample}/ folder
SAMPLE = "_test"                            # "_test" = zero-setup smoke sample
DEVICE = "HB1"
COOLDOWN = ""                              # free text, e.g. "3"
COMMENT = ""

# Locked type code (docs/data_convention.md). Reuse the closest existing
# code, or add a row to that table first — `allocate_run()` does NOT
# validate this string, so an invented code silently "works" and pollutes
# index.csv. "RT" is registered in the table for exactly this template.
MEASUREMENT_TYPE = "RT"

SENSE_CURRENT_A = 100e-6
N_AVERAGES = 10                            # SMU readings averaged per temperature point
TEMPERATURES_K = [300.0, 250.0, 200.0, 150.0, 100.0, 50.0, 10.0]

SETTLE_TOLERANCE_K = 0.05    # "at" the setpoint once within this ...
SETTLE_HOLD_S = 30.0        # ... held continuously for this long
DWELL_S = 5.0              # extra dead-time after settle, before acquiring

SMU_CFG = SMUConfig(
    visa_resource="GPIB0::18::INSTR",
    source_function="current",     # source I, measure the complementary V
    compliance_voltage_V=2.0,      # stop before this if the DUT opens up
    four_wire=True,                # remote sense — lead resistance drops out
    nplc=5,                        # slow integration, quiet reading
    source_limit_A=1e-3,           # set_source_level() refuses beyond this
)
TEMP_CFG = TemperatureControllerConfig(
    visa_resource="TCPIP0::192.168.1.5::7020::SOCKET", sensor_uids=("MB1.T1",))
# ─────────────────────────────────────────────────────────────────────────────


def build_header_fields(ctx, records: list[dict], status: str) -> dict:
    """The `# key: value` CSV header + the index.csv row for this run.

    Covers data_naming.BASE_COLUMNS (universal) then any measurement-specific
    keys. Re-called on every incremental write with the records so far, so
    `T_K` (the *measured* mean) and `status` stay current. T_setpoint_K is
    blank here: temperature is the swept axis, not one fixed setpoint, so the
    filename carries no T###K token.
    """
    measured = [r["temperature_1_K"] for r in records if r.get("temperature_1_K") is not None]
    return {
        "run": ctx.run_number,
        "timestamp": ctx.timestamp.isoformat(timespec="seconds"),
        "sample": ctx.sample,
        "device": ctx.device,
        "type": MEASUREMENT_TYPE,
        "T_setpoint_K": "",
        "T_K": sum(measured) / len(measured) if measured else "",
        "cooldown": COOLDOWN,
        "status": status,
        "comment": COMMENT,
        "series": "",
        "sense_current_A": SENSE_CURRENT_A,
        "n_averages": N_AVERAGES,
        "four_wire": SMU_CFG.four_wire,
    }


def save_partial(ctx, records: list[dict], status: str) -> None:
    """End-of-run save: rewrite the raw file + finalize the index row with an
    outcome-derived status. Call UNCONDITIONALLY (in a `finally:`), so an
    aborted/crashed run never stays stuck at "in_progress". The guard keeps
    an already-written file from being truncated to a header-only stub."""
    fields = build_header_fields(ctx, records, status)
    if records or not ctx.raw_path.exists():
        write_record(ctx.raw_path, records, fields)
    finalize_index_row(ctx.sample_dir.parent, ctx.sample, ctx.run_number, fields)


def main() -> None:
    ensure_sample(DATA_ROOT, SAMPLE, create=True)
    ctx = allocate_run(DATA_ROOT, SAMPLE, DEVICE, MEASUREMENT_TYPE)
    write_csv = make_incremental_writer(
        ctx.raw_path, lambda recs: build_header_fields(ctx, recs, "in_progress"))
    print(f"→ {ctx.raw_path}")

    smu = temp_ctrl = None
    records: list[dict] = []
    final_status = "completed"
    try:
        smu = connect_smu(SMU_CFG)
        temp_ctrl = connect_temperature_controller(TEMP_CFG)  # load-bearing here

        set_source_level(smu, SMU_CFG, SENSE_CURRENT_A)   # park the source once

        for setpoint_K in TEMPERATURES_K:
            set_temperature(temp_ctrl, TEMP_CFG, setpoint_K)
            wait_for_temperature_stable(temp_ctrl, TEMP_CFG, setpoint_K,
                                        tolerance_K=SETTLE_TOLERANCE_K,
                                        hold_time_s=SETTLE_HOLD_S)
            time.sleep(DWELL_S)

            v = acquire_measurement(smu, SMU_CFG, N_AVERAGES)   # {"mean", "sem"}
            t1_K, t2_K = read_temperature(temp_ctrl, TEMP_CFG)

            records.append({
                "point_index": len(records),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "T_setpoint_step_K": setpoint_K,
                "temperature_1_K": t1_K,
                "temperature_2_K": t2_K,
                "sense_current_A": SENSE_CURRENT_A,
                "voltage_V": v["mean"],
                "voltage_sem_V": v["sem"],
                "resistance_ohm": v["mean"] / SENSE_CURRENT_A,
                "n_averages": N_AVERAGES,
            })
            write_csv(records)   # full rewrite every point → crash-safe
            print(f"{setpoint_K:8.3f} K  →  R = {records[-1]['resistance_ohm']:.6g} Ω")

    except KeyboardInterrupt:
        final_status = "aborted"
        print("\nInterrupted — saving partial data.")
    except Exception:
        final_status = "error"
        raise
    finally:
        save_partial(ctx, records, final_status)
        # One guard PER shutdown call — if shutdown_smu() raises (e.g. a VISA
        # timeout mid ramp-to-zero), the temperature controller must still be
        # closed. shutdown_smu() ramps the source to 0 and opens the output.
        safe_shutdown("SMU (2450)", lambda: shutdown_smu(smu))
        safe_shutdown("temperature controller", lambda: shutdown_temperature_controller(temp_ctrl))
    print(f"Done ({final_status}): {len(records)} points → {ctx.raw_path}")


if __name__ == "__main__":
    main()
