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
from textual.widgets import Input, Select, TextArea

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


# One sweep-defining field per program, set to a value that used to allocate
# billions of points on the next keystroke (frozen / OOM-killed form).
HUGE_SWEEP = {
    "DCHallMeasurementApp": ("sweep_rows", "-20, 20, 100000000"),
    "DCIVCurveApp": ("step_A", "1e-12"),
    "DCGateSweepApp": ("step_V", "1e-12"),
    "DCSpinValveApp": ("sweep_rows", "-20, 20, 100000000"),
    "MFLIDualHarmonicApp": ("sweep_rows", "-20, 20, 100000000"),
    "MFLIDualHarmonic6221App": ("sweep_rows", "-20, 20, 100000000"),
    "MFLIDiffResistanceApp": ("n_points", "1000000000"),
    "MFLIPhaseCalibrationApp": ("sweep_rows", "-20, 20, 100000000"),
    "SOTPulsedSwitchingApp": ("amplitude_step_V", "1e-12"),
    "SOTPulsedSwitching2HApp": ("amplitude_step_V", "1e-12"),
    "SOTPulsedSwitching6221App": ("pulse_current_step_A", "1e-12"),
    "NonlocalSwitchingApp": ("pulse_current_step_A", "1e-12"),
}


@pytest.mark.parametrize("module,app_name", [a for a in APPS if a[1] in HUGE_SWEEP],
                         ids=[a for _, a in APPS if a in HUGE_SWEEP])
def test_huge_sweep_is_a_form_error_not_a_freeze(module, app_name, tmp_path, monkeypatch):
    mod = importlib.import_module(module)
    app_cls = getattr(mod, app_name)
    _isolate(mod, app_cls, tmp_path, monkeypatch)
    field_id, value = HUGE_SWEEP[app_name]

    async def go():
        app = app_cls()
        async with app.run_test(size=(220, 70)) as pilot:
            await pilot.pause()
            widget = app.query_one(f"#{field_id}")
            if isinstance(widget, TextArea):
                widget.text = value
            else:
                widget.value = value
            await pilot.pause()              # the Changed event -> refresh_summary()
            state, parse_errors = app.parse_state()
            _, _, errors = mod.build_summary(state) if not parse_errors else ([], [], parse_errors)
            return errors

    errors = asyncio.run(go())
    assert any("limit" in e for e in errors), errors
