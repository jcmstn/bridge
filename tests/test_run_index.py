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
