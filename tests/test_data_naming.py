"""
Tests for instruments/data_naming.py
========================================
Hardware-free — this module never touches VISA/GPIB, so every test here
runs against a tmp_path data root with no instrument connections.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pandas as pd
import pytest

from instruments.data_naming import (
    TEST_SAMPLE,
    RunContext,
    SampleNotFoundError,
    allocate_run,
    ensure_sample,
    ensure_test_sample,
    finalize_index_row,
    format_axis_token,
    list_samples,
    preview_raw_filename,
    write_record,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sample bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def test_ensure_sample_creates_expected_tree(tmp_path: Path) -> None:
    sample_dir = ensure_sample(tmp_path, "A", create=True)
    assert sample_dir == tmp_path / "A"
    assert (sample_dir / "sample.yaml").is_file()
    assert (sample_dir / "notes.md").is_file()
    assert (sample_dir / "raw").is_dir()
    assert (sample_dir / "proc").is_dir()
    assert (sample_dir / "index.csv").is_file()

    index_text = (sample_dir / "index.csv").read_text()
    assert index_text.strip().startswith("run,timestamp,sample,device,type")


def test_ensure_sample_never_overwrites_existing_content(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    yaml_path = tmp_path / "A" / "sample.yaml"
    yaml_path.write_text("name: A\nstack: hand-edited content\n")

    ensure_sample(tmp_path, "A", create=True)  # called again, idempotent

    assert yaml_path.read_text() == "name: A\nstack: hand-edited content\n"


def test_ensure_sample_without_create_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(SampleNotFoundError):
        ensure_sample(tmp_path, "A", create=False)


def test_list_samples_excludes_test_sample(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    ensure_sample(tmp_path, "B", create=True)
    ensure_test_sample(tmp_path)

    assert list_samples(tmp_path) == ["A", "B"]


def test_list_samples_missing_data_root(tmp_path: Path) -> None:
    assert list_samples(tmp_path / "does_not_exist") == []


def test_ensure_test_sample_idempotent(tmp_path: Path) -> None:
    d1 = ensure_test_sample(tmp_path)
    d2 = ensure_test_sample(tmp_path)
    assert d1 == d2 == tmp_path / TEST_SAMPLE
    assert (d1 / "sample.yaml").is_file()


# ─────────────────────────────────────────────────────────────────────────────
# Filename convention
# ─────────────────────────────────────────────────────────────────────────────

def test_format_axis_token_examples() -> None:
    assert format_axis_token("angle_deg", 0) == "a000"
    assert format_axis_token("angle_deg", 90) == "a090"
    assert format_axis_token("angle_deg", -90) == "am090"
    assert format_axis_token("field_T", 2) == "B2T"


def test_allocate_run_filename_matches_convention_examples(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)

    ctx = allocate_run(
        tmp_path, "A", "HB3", "IV", temperature_setpoint_K=300,
    )
    assert ctx.raw_path.name.startswith("A_0001_HB3_IV_T300K_")
    assert ctx.raw_path.name.endswith(".csv")
    # No key-axis given -> no stray double underscore before the timestamp.
    assert "__" not in ctx.raw_path.name

    ctx2 = allocate_run(
        tmp_path, "A", "HB3", "BSWP", temperature_setpoint_K=10,
        key_axis=("angle_deg", 0),
    )
    assert ctx2.raw_path.name.startswith("A_0002_HB3_BSWP_T010K_a000_")

    ctx3 = allocate_run(
        tmp_path, "A", "SV2", "BSWP", temperature_setpoint_K=10,
        key_axis=("angle_deg", -90),
    )
    assert ctx3.raw_path.name.startswith("A_0003_SV2_BSWP_T010K_am090_")


def test_preview_raw_filename_is_pure_and_has_no_side_effects(tmp_path: Path) -> None:
    # No sample exists at all -- preview must not require one, and must not
    # touch the filesystem or allocate a run.
    name = preview_raw_filename(
        "A", "HB3", "BSWP", temperature_setpoint_K=10, key_axis=("angle_deg", 0),
    )
    assert name == "A_NNNN_HB3_BSWP_T010K_a000"
    assert not (tmp_path / "A").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Run-number allocation semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_allocate_run_increments_monotonically(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    numbers = [
        allocate_run(tmp_path, "A", "HB3", "IV", temperature_setpoint_K=300).run_number
        for _ in range(5)
    ]
    assert numbers == [1, 2, 3, 4, 5]


def test_allocate_run_missing_sample_raises(tmp_path: Path) -> None:
    with pytest.raises(SampleNotFoundError):
        allocate_run(tmp_path, "Nonexistent", "HB3", "IV", temperature_setpoint_K=300)


def test_crashed_run_permanently_consumes_its_number(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)

    ctx1 = allocate_run(tmp_path, "A", "HB3", "IV", temperature_setpoint_K=300)
    # Simulate a crash: no write_record()/finalize_index_row() ever happens
    # for ctx1.

    ctx2 = allocate_run(tmp_path, "A", "HB3", "IV", temperature_setpoint_K=300)
    assert ctx2.run_number == ctx1.run_number + 1

    df = pd.read_csv(tmp_path / "A" / "index.csv")
    row1 = df[df["run"] == ctx1.run_number].iloc[0]
    assert row1["status"] == "in_progress"


# ─────────────────────────────────────────────────────────────────────────────
# Header / CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def test_write_record_header_and_body_round_trip(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    ctx = allocate_run(tmp_path, "A", "HB3", "IV", temperature_setpoint_K=300)

    header_fields = {
        "run": ctx.run_number, "timestamp": ctx.timestamp.isoformat(),
        "sample": "A", "device": "HB3", "type": "IV",
        "T_K": 10.02, "cooldown": 3, "status": "good",
        "comment": "clear AHE hysteresis, coercive ~80 mT",
        "B_range_T": [-2, 2],
    }
    records = [
        {"B_T": -2.0, "Vxy_V": 1e-5, "Vxx_V": 2e-3, "T_K": 10.01},
        {"B_T": -1.0, "Vxy_V": 0.5e-5, "Vxx_V": 2e-3, "T_K": 10.02},
    ]
    write_record(ctx.raw_path, records, header_fields)

    raw_text = ctx.raw_path.read_text()
    lines = raw_text.splitlines()
    assert all(line.startswith("#") for line in lines if line.strip() and "," not in line.split(":")[0])
    assert f"# run: {ctx.run_number}" in lines
    assert "# B_range_T: [-2, 2]" in lines
    assert "# comment: clear AHE hysteresis, coercive ~80 mT" in lines

    df = pd.read_csv(ctx.raw_path, comment="#")
    assert list(df.columns) == ["B_T", "Vxy_V", "Vxx_V", "T_K"]
    assert len(df) == 2
    assert df.iloc[0]["B_T"] == -2.0


def test_write_record_can_be_called_repeatedly_incremental_style(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    ctx = allocate_run(tmp_path, "A", "HB3", "IV", temperature_setpoint_K=300)

    records: list[dict] = []
    for i in range(3):
        records.append({"x": i, "y": i * i})
        write_record(ctx.raw_path, records, {"run": ctx.run_number, "status": "in_progress"})

    df = pd.read_csv(ctx.raw_path, comment="#")
    assert len(df) == 3
    assert df["y"].tolist() == [0, 1, 4]


# ─────────────────────────────────────────────────────────────────────────────
# index.csv finalize / column union
# ─────────────────────────────────────────────────────────────────────────────

def test_finalize_index_row_unions_new_columns(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    ctx = allocate_run(tmp_path, "A", "HB3", "IV", temperature_setpoint_K=300)

    finalize_index_row(
        tmp_path, "A", ctx.run_number,
        {"status": "good", "comment": "looks clean", "contacts": "I=1,4 V=2,3"},
    )

    df = pd.read_csv(tmp_path / "A" / "index.csv")
    assert "contacts" in df.columns
    row = df[df["run"] == ctx.run_number].iloc[0]
    assert row["status"] == "good"
    assert row["contacts"] == "I=1,4 V=2,3"


def test_finalize_index_row_twice_final_write_wins(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    ctx = allocate_run(tmp_path, "A", "HB3", "IV", temperature_setpoint_K=300)

    finalize_index_row(tmp_path, "A", ctx.run_number, {"status": "completed"})
    finalize_index_row(tmp_path, "A", ctx.run_number, {"status": "good", "comment": "final"})

    df = pd.read_csv(tmp_path / "A" / "index.csv")
    row = df[df["run"] == ctx.run_number].iloc[0]
    assert row["status"] == "good"
    assert row["comment"] == "final"
    assert len(df) == 1  # updated in place, not appended twice


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency
# ─────────────────────────────────────────────────────────────────────────────

def _alloc_worker(data_root: str, sample: str, n: int) -> None:
    from instruments.data_naming import allocate_run  # re-import in child process
    for _ in range(n):
        allocate_run(Path(data_root), sample, "HB3", "IV", temperature_setpoint_K=300)


def test_allocate_run_is_unique_and_gap_free_under_concurrency(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)

    n_procs, n_per_proc = 4, 10
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_alloc_worker, args=(str(tmp_path), "A", n_per_proc))
        for _ in range(n_procs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    df = pd.read_csv(tmp_path / "A" / "index.csv")
    run_numbers = sorted(df["run"].tolist())
    expected = list(range(1, n_procs * n_per_proc + 1))
    assert run_numbers == expected
