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
Sources a fixed DC sense current with a Keithley 6221, and at each of a list
of MercuryiTC temperature setpoints, reverses the current +I/-I and records
the odd (resistive) voltage from a Keithley 2182 → one CSV row per
temperature, one raw file for the whole sweep, one index.csv row.

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
from instruments.keithley6221 import (
    SourceConfig,
    connect_source,
    ramp_current_to_zero,
    shutdown_source,
    acquire_reversal_averaged_voltage,
)
from instruments.keithley2182 import VoltmeterConfig, connect_voltmeter
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
N_REVERSALS = 5
TEMPERATURES_K = [300.0, 250.0, 200.0, 150.0, 100.0, 50.0, 10.0]

SETTLE_TOLERANCE_K = 0.05    # "at" the setpoint once within this ...
SETTLE_HOLD_S = 30.0        # ... held continuously for this long
DWELL_S = 5.0              # extra dead-time after settle, before acquiring

SRC_CFG = SourceConfig(visa_resource="GPIB0::20::INSTR", sense_current_A=SENSE_CURRENT_A,
                       compliance_V=2.0, source_delay_s=0.05)
VOLT_CFG = VoltmeterConfig(visa_resource="GPIB0::7::INSTR", nplc=5, auto_range=True)
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
        "n_reversals": N_REVERSALS,
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

    source = voltmeter = temp_ctrl = None
    records: list[dict] = []
    final_status = "completed"
    try:
        source = connect_source(SRC_CFG)
        voltmeter = connect_voltmeter(VOLT_CFG)
        temp_ctrl = connect_temperature_controller(TEMP_CFG)  # load-bearing here

        for setpoint_K in TEMPERATURES_K:
            set_temperature(temp_ctrl, TEMP_CFG, setpoint_K)
            wait_for_temperature_stable(temp_ctrl, TEMP_CFG, setpoint_K,
                                        tolerance_K=SETTLE_TOLERANCE_K,
                                        hold_time_s=SETTLE_HOLD_S)
            time.sleep(DWELL_S)

            hv = acquire_reversal_averaged_voltage(
                source, voltmeter, SENSE_CURRENT_A, N_REVERSALS,
                source_delay_s=SRC_CFG.source_delay_s)
            t1_K, t2_K = read_temperature(temp_ctrl, TEMP_CFG)

            records.append({
                "point_index": len(records),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "T_setpoint_step_K": setpoint_K,
                "temperature_1_K": t1_K,
                "temperature_2_K": t2_K,
                "sense_current_A": SENSE_CURRENT_A,
                "voltage_odd_V": hv["mean"],
                "voltage_odd_sem_V": hv["sem"],
                "voltage_even_V": hv["even_mean"],
                "resistance_ohm": hv["mean"] / SENSE_CURRENT_A,
                "n_reversals": hv["n_reversals"],
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
        # One guard PER shutdown call — if the ramp raises (e.g. a VISA
        # timeout mid-ramp), shutdown_source() must still run, or the 6221
        # is left sourcing current into the DUT. shutdown_source()'s default
        # zero_first=True is the floor when the gentle ramp didn't finish.
        if source is not None:
            safe_shutdown("ramp 6221 to zero", lambda: ramp_current_to_zero(source))
            safe_shutdown("source (6221)", lambda: shutdown_source(source))
        safe_shutdown("temperature controller", lambda: shutdown_temperature_controller(temp_ctrl))
        # the 2182 needs no shutdown
    print(f"Done ({final_status}): {len(records)} points → {ctx.raw_path}")


if __name__ == "__main__":
    main()
