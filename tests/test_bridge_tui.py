"""
bridge_tui.py — the unified menu: every program App is reachable, one launch
button per program, each exits the menu with its key, and the Recent-runs
table shows what both front ends wrote to runs.db.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import bridge_tui as menu
from instruments import run_index

_ALL_APPS = {
    ("dc.dc_hall_measurement_tui", "DCHallMeasurementApp"),
    ("dc.dc_iv_curve_tui", "DCIVCurveApp"),
    ("dc.dc_gate_sweep_tui", "DCGateSweepApp"),
    ("dc.dc_spin_valve_tui", "DCSpinValveApp"),
    ("dc.dc_rt_log_tui", "DCRTLogApp"),
    ("mfli.mfli_dual_harmonic_tui", "MFLIDualHarmonicApp"),
    ("mfli.mfli_diff_resistance_tui", "MFLIDiffResistanceApp"),
    ("mfli.mfli_phase_calibration_tui", "MFLIPhaseCalibrationApp"),
    ("mfli.mfli_noise_spectrum_tui", "MFLINoiseSpectrumApp"),
    ("sot.sot_pulsed_switching_tui", "SOTPulsedSwitchingApp"),
    ("sot.sot_nonlocal_switching_tui", "NonlocalSwitchingApp"),
}


def test_every_program_app_is_in_the_menu_once():
    programs = [p for ps in menu.PROGRAMS.values() for p in ps]
    assert list(menu.PROGRAMS) == ["DC Suite", "MFLI Suite", "SOT Suite"]
    assert {(p.app.__module__, p.app.__name__) for p in programs} == _ALL_APPS
    assert len({p.key for p in programs}) == len(programs) == len(_ALL_APPS)
    assert all(p.description and p.schematic for p in programs)
    # the classes really are what those modules export
    assert all(getattr(importlib.import_module(m), n) for m, n in _ALL_APPS)


def test_each_launch_button_exits_with_its_program_key(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(run_index, "_DB_PATH", tmp_path / "runs.db")
    keys = [p.key for ps in menu.PROGRAMS.values() for p in ps]

    async def go():
        seen = {}
        for key in keys:
            app = menu.LauncherApp()
            async with app.run_test(size=(240, 80)) as pilot:
                await pilot.pause()
                assert {b.id for b in app.query("Button")} == {f"launch_{k}" for k in keys}
                app.query_one(f"#launch_{key}").press()
                await pilot.pause()
            seen[key] = app.return_value
        return seen

    assert asyncio.run(go()) == {k: k for k in keys}


def test_recent_runs_table_shows_the_run_history(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(run_index, "_DB_PATH", tmp_path / "runs.db")

    async def go(expect_rows: int):
        app = menu.LauncherApp()
        async with app.run_test(size=(240, 80)) as pilot:
            await pilot.pause()
            table = app.query_one("#recent_runs")
            assert table.row_count == expect_rows
            assert app.query_one("#runs_empty").display is (expect_rows == 0)
            return [str(c) for c in table.get_row_at(0)] if expect_rows else None

    assert asyncio.run(go(0)) is None
    run_id = run_index.start_run("SOT", "Nonlocal spin-current switching", {}, str(tmp_path), [],
                                 sample="A", device="SV2", run_number=3)
    run_index.finish_run(run_id, status="completed", point_count=12, duration_s=4.25)
    row = asyncio.run(go(1))
    assert row[1:8] == ["A", "SV2", "3", "SOT", "Nonlocal spin-current switching", "completed", "12"]
    assert row[8] == "4.2" or row[8] == "4.3"
