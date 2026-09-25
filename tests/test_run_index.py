"""
instruments/run_index.py — the SQLite run history both front ends write:
a start/update/finish round trip, and connections are always closed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from instruments import run_index


def test_start_update_finish_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_index, "_DB_PATH", tmp_path / "sub" / "runs.db")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)   # an unclosed connection fails here
        run_id = run_index.start_run("DC", "DC Hall Measurement", {"a": 1}, str(tmp_path), [],
                                     sample="A", device="HB3", run_number=7)
        run_index.update_point_count(run_id, 3)
        assert run_index.recent_runs()[0]["point_count"] == 3
        run_index.finish_run(run_id, status="completed", point_count=5, duration_s=1.5,
                             output_paths=["x.csv"])
        (row,) = run_index.recent_runs()
    assert row["status"] == "completed" and row["point_count"] == 5
    assert row["sample"] == "A" and row["run_number"] == "7" and row["finished_at"]


def test_run_numbers_allocated_mid_session_land_at_finish(tmp_path: Path, monkeypatch) -> None:
    # Multi-value runs allocate their run numbers per iteration, after
    # start_run() -- so start_run() sees None and finish_run() fills it in.
    monkeypatch.setattr(run_index, "_DB_PATH", tmp_path / "runs.db")
    run_id = run_index.start_run("DC", "Hall", {}, str(tmp_path), [], sample="A", device="HB3")
    run_index.finish_run(run_id, status="completed", point_count=1, duration_s=1.0,
                         run_numbers=[12, 13, 14])
    assert run_index.recent_runs()[0]["run_number"] == "12-14"
    assert run_index.format_run_numbers([7]) == "7"
    assert run_index.format_run_numbers([9, 7]) == "7, 9"
