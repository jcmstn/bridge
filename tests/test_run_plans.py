"""
Every program's pure run_plan() — the one both the TUI RunScreen and the web
page run — end to end with the hardware faked: the form's default state
(parsed by the real App, headless) -> build_plan() -> run_plan() with every
connect/shutdown/instrument call stubbed and run_measurement() replaced by a
fake that emits two records. Checks the orchestration both front ends rely on:
one allocated + finalized run per series value, its header extras recorded,
records tagged with their series, the per-run PNG callback, and teardown.
"""

from __future__ import annotations

import asyncio
import importlib
import threading

from unittest.mock import MagicMock

import pandas as pd
import pytest

from instruments.data_naming import ensure_sample, read_raw

# (module, App, {field: value} overrides that make the form describe a TWO-run series)
MULTI_RUN = [
    ("dc.dc_hall_measurement_tui", "DCHallMeasurementApp", {"sense_current_values": "0.001, 0.002"}),
    ("dc.dc_iv_curve_tui", "DCIVCurveApp", {"enable_gate": True, "gate_voltage_values": "0, 1"}),
    ("dc.dc_gate_sweep_tui", "DCGateSweepApp", {"sense_current_values": "1e-6, 2e-6"}),
    ("dc.dc_spin_valve_tui", "DCSpinValveApp", {"sense_current_values": "0.001, 0.002"}),
    ("mfli.mfli_dual_harmonic_tui", "MFLIDualHarmonicApp", {"ac_source": "6221", "amplitude_values": "1e-6, 2e-6"}),
    ("sot.sot_pulsed_switching_tui", "SOTPulsedSwitchingApp", {"sense_current_values": "1e-4, 2e-4"}),
    ("sot.sot_pulsed_switching_tui", "SOTPulsedSwitchingApp",
     {"read_mode": "harmonic", "sense_current_values": "1e-4, 2e-4"}),
    ("sot.sot_pulsed_switching_tui", "SOTPulsedSwitchingApp",
     {"pulse_source": "6221", "read_mode": "harmonic", "sense_current_values": "1e-4, 2e-4"}),
]
# single-run programs: the run is allocated at Start (build_plan) as plan.run_ctx
SINGLE_RUN = [
    ("mfli.mfli_dual_harmonic_tui", "MFLIDualHarmonicApp", {}),
    ("mfli.mfli_diff_resistance_tui", "MFLIDiffResistanceApp", {}),
    ("mfli.mfli_phase_calibration_tui", "MFLIPhaseCalibrationApp", {}),
]
PROGRAMS = [(m, a, o, 2) for m, a, o in MULTI_RUN] + [(m, a, o, 1) for m, a, o in SINGLE_RUN]
_TOGGLES = ("ac_source", "pulse_source", "read_mode")        # a merged program's mode
_IDS = ["-".join([a, *(o[t] for t in _TOGGLES if t in o)]) for _, a, o, _ in PROGRAMS]

_STUB_PREFIXES = ("connect", "shutdown", "setup_", "configure_", "sync_", "ramp_", "set_",
                  "wait_", "null_", "auto_null", "acquire", "initialize_", "check_mds", "disable_")


def _stub_hardware(mod, monkeypatch) -> list[str]:
    shut: list[str] = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if not callable(obj) or isinstance(obj, type) or not name.startswith(_STUB_PREFIXES):
            continue
        if name.startswith("shutdown"):
            monkeypatch.setattr(mod, name, lambda *a, _n=name, **k: shut.append(_n))
        else:
            monkeypatch.setattr(mod, name, lambda *a, **k: MagicMock())
    return shut


def _fake_phase_calibration(*args, **kwargs):
    records = []
    for i in range(2):
        rec = {"point_index": i, "voltage_V": 1e-3 * (i + 1)}
        records.append(rec)
        kwargs["on_point"](rec)
        kwargs["write_csv"](pd.DataFrame(records))
    return MagicMock()


def _fake_acquisition(mod, monkeypatch, fake=None) -> None:
    if hasattr(mod, "run_phase_calibration"):
        monkeypatch.setattr(mod, "run_phase_calibration", fake or _fake_phase_calibration)
        monkeypatch.setattr(mod, "format_report", lambda report: "report")
    else:
        monkeypatch.setattr(mod, "run_measurement", fake or _fake_run_measurement)


def _fake_run_measurement(*args, **kwargs):
    records = []
    for i in range(2):
        rec = {"point_index": i, "voltage_V": 1e-3 * (i + 1)}
        records.append(rec)
        kwargs["on_point"](rec)
        kwargs["write_csv"](records)
    return pd.DataFrame(records)


def _default_state(mod, app_name, overrides, tmp_path, monkeypatch) -> dict:
    app_cls = getattr(mod, app_name)
    monkeypatch.setattr(mod, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app_cls, "data_root", tmp_path)

    async def go():
        app = app_cls()
        async with app.run_test(size=(220, 70)) as pilot:
            await pilot.pause()
            app.query_one("#data_dir").value = str(tmp_path)
            app.query_one("#device").value = "HB3"
            for fid, value in overrides.items():
                app.query_one(f"#{fid}").value = value
            await pilot.pause()
            state, errors = app.parse_state()
            assert not errors, errors
            assert not mod.build_summary(state)[2], mod.build_summary(state)[2]
            return state

    return asyncio.run(go())


@pytest.mark.parametrize("module,app_name,overrides,n", PROGRAMS, ids=_IDS)
def test_run_plan_records_one_finalized_run_per_series_value(module, app_name, overrides, n,
                                                              tmp_path, monkeypatch):
    mod = importlib.import_module(module)
    ensure_sample(tmp_path, "_test", create=True)
    state = _default_state(mod, app_name, overrides, tmp_path, monkeypatch)
    plan = mod.build_plan(state, tmp_path)
    mod = getattr(mod, "engine", lambda _plan: mod)(plan)     # a merged program runs the plan's engine
    shut = _stub_hardware(mod, monkeypatch)
    _fake_acquisition(mod, monkeypatch)

    points, finished, contexts, extras = [], [], [], []
    mod.run_plan(plan, threading.Event(), on_point=points.append,
                 on_run_finished=lambda ctx, recs: finished.append((ctx.run_number, len(recs))),
                 run_contexts=contexts, run_extras=extras)

    assert len(contexts) == len(extras) == n
    assert [r.get("series_index", 0) for r in points] == [i for i in range(n) for _ in range(2)]
    assert finished == [(c.run_number, 2) for c in contexts]
    index = pd.read_csv(tmp_path / "_test" / "index.csv")
    assert set(index["status"]) == {"completed"} and len(index) == n
    assert set(index["type"]) == {mod.MEASUREMENT_TYPE}
    for ctx in contexts:
        assert len(read_raw(ctx.raw_path)) == 2
    assert shut, "instruments were not shut down"


@pytest.mark.parametrize("module,app_name,overrides,n", PROGRAMS, ids=_IDS)
def test_run_plan_finalizes_a_failed_run_as_error_and_still_shuts_down(module, app_name, overrides, n,
                                                                       tmp_path, monkeypatch):
    mod = importlib.import_module(module)
    ensure_sample(tmp_path, "_test", create=True)
    state = _default_state(mod, app_name, overrides, tmp_path, monkeypatch)
    plan = mod.build_plan(state, tmp_path)
    mod = getattr(mod, "engine", lambda _plan: mod)(plan)
    shut = _stub_hardware(mod, monkeypatch)

    def boom(*args, **kwargs):
        kwargs["on_point"]({"point_index": 0, "voltage_V": 0.0})
        raise RuntimeError("VISA timeout")

    _fake_acquisition(mod, monkeypatch, boom)
    contexts: list = []
    with pytest.raises(RuntimeError, match="VISA"):
        mod.run_plan(plan, threading.Event(), run_contexts=contexts)
    index = pd.read_csv(tmp_path / "_test" / "index.csv")
    assert list(index["status"]) == ["error"] and len(contexts) == 1
    assert shut
