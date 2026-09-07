"""The examples/custom_program.py save path — no hardware, no VISA.

Drives the template's data-convention wiring (build_header_fields +
save_partial + the incremental writer) with stub records and checks the
three things a hand-rolled standalone script tends to get wrong.
"""

import importlib.util
import re
from pathlib import Path

import pandas as pd

from instruments.data_naming import (
    allocate_run,
    ensure_sample,
    make_incremental_writer,
    read_raw,
)

_EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "custom_program.py"
_spec = importlib.util.spec_from_file_location("custom_program_example", _EXAMPLE_PATH)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def _stub_records(n=3):
    return [
        {
            "point_index": i,
            "timestamp": "2026-09-07T12:00:00",
            "T_setpoint_step_K": 300.0 - 50 * i,
            "temperature_1_K": 299.5 - 50 * i,
            "temperature_2_K": None,
            "sense_current_A": example.SENSE_CURRENT_A,
            "voltage_odd_V": 1e-3 * (i + 1),
            "voltage_odd_sem_V": 1e-6,
            "voltage_even_V": 2e-5,
            "resistance_ohm": 1e-3 * (i + 1) / example.SENSE_CURRENT_A,
            "n_reversals": example.N_REVERSALS,
        }
        for i in range(n)
    ]


def test_save_path_round_trips(tmp_path):
    ensure_sample(tmp_path, "_test", create=True)
    ctx = allocate_run(tmp_path, "_test", "HB1", example.MEASUREMENT_TYPE)
    write_csv = make_incremental_writer(
        ctx.raw_path, lambda recs: example.build_header_fields(ctx, recs, "in_progress")
    )

    records = _stub_records()
    for i in range(len(records)):
        write_csv(records[: i + 1])          # incremental: full rewrite each point
    example.save_partial(ctx, records, "completed")

    # 1. two-row (name / units) header round-trips back to name_unit columns
    df = read_raw(ctx.raw_path)
    assert len(df) == 3
    assert {"resistance_ohm", "voltage_odd_sem_V", "T_setpoint_step_K"} <= set(df.columns)
    assert df["n_reversals"].iloc[0] == example.N_REVERSALS

    # 2. the finally: finalize fired — index row is "completed", not "in_progress"
    idx = pd.read_csv(ctx.sample_dir / "index.csv")
    row = idx.loc[idx["run"] == ctx.run_number].iloc[0]
    assert row["status"] == "completed"
    assert row["sense_current_A"] == example.SENSE_CURRENT_A

    # 3. allocate_run built the filename (no T token — temperature is the swept axis)
    assert re.fullmatch(
        rf"_test_\d{{4}}_HB1_{example.MEASUREMENT_TYPE}_\d{{8}}T\d{{6}}\.csv", ctx.raw_path.name
    )


def test_save_partial_does_not_truncate_existing_file(tmp_path):
    ensure_sample(tmp_path, "_test", create=True)
    ctx = allocate_run(tmp_path, "_test", "HB1", example.MEASUREMENT_TYPE)
    make_incremental_writer(
        ctx.raw_path, lambda recs: example.build_header_fields(ctx, recs, "in_progress")
    )(_stub_records())

    example.save_partial(ctx, [], "aborted")     # no records in hand at abort time

    assert len(read_raw(ctx.raw_path)) == 3      # kept, not stubbed to header-only
    idx = pd.read_csv(ctx.sample_dir / "index.csv")
    assert idx.loc[idx["run"] == ctx.run_number].iloc[0]["status"] == "aborted"
