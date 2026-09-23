"""
Every program's Textual App mounts (compose + on_mount + refresh_summary)
with no hardware, and its form round-trips through its settings file: edit
a field, save, remount, and the same raw form state comes back. The guard
for refactoring the shared TUI scaffolding (instruments/tui_common.py).
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest
from textual.widgets import Input, Select

APPS = [
    ("dc.dc_hall_measurement_tui", "DCHallMeasurementApp"),
    ("dc.dc_iv_curve_tui", "DCIVCurveApp"),
    ("dc.dc_gate_sweep_tui", "DCGateSweepApp"),
    ("dc.dc_spin_valve_tui", "DCSpinValveApp"),
    ("mfli.mfli_dual_harmonic_tui", "MFLIDualHarmonicApp"),
    ("mfli.mfli_dual_harmonic_6221_tui", "MFLIDualHarmonic6221App"),
    ("mfli.mfli_diff_resistance_tui", "MFLIDiffResistanceApp"),
    ("mfli.mfli_phase_calibration_tui", "MFLIPhaseCalibrationApp"),
    ("mfli.mfli_noise_spectrum_tui", "MFLINoiseSpectrumApp"),
    ("sot.sot_pulsed_switching_tui", "SOTPulsedSwitchingApp"),
    ("sot.sot_pulsed_switching_2h_tui", "SOTPulsedSwitching2HApp"),
    ("sot.sot_pulsed_switching_6221_tui", "SOTPulsedSwitching6221App"),
    ("sot.sot_nonlocal_switching_tui", "NonlocalSwitchingApp"),
]


def _isolate(mod, app_cls, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app_cls, "data_root", tmp_path)


@pytest.mark.parametrize("module,app_name", APPS, ids=[a for _, a in APPS])
def test_app_mounts_and_settings_round_trip(module, app_name, tmp_path, monkeypatch):
    mod = importlib.import_module(module)
    app_cls = getattr(mod, app_name)
    _isolate(mod, app_cls, tmp_path, monkeypatch)
    field_id = "device"      # identity-bar text field every TUI has; harmless to edit

    async def mount(edit: bool):
        app = app_cls()
        async with app.run_test(size=(220, 70)) as pilot:
            await pilot.pause()
            app.refresh_summary()
            assert app.query_one("#sample_select", Select).value == "_test"
            if edit:
                app.query_one(f"#{field_id}", Input).value = "HB9"
                await pilot.pause()
                app._save_settings(app.collect_raw())
            return app.collect_raw()

    saved = asyncio.run(mount(edit=True))
    assert saved[field_id] == "HB9"
    assert (tmp_path / "settings.json").is_file()
    assert asyncio.run(mount(edit=False)) == saved
