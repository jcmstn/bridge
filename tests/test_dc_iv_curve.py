"""Plan-purity tests for the I-V build_plan() the TUI and web/dc/iv_curve.py share."""

from __future__ import annotations

from pathlib import Path

import pytest

import dc.dc_iv_curve_tui as tui
from instruments.data_naming import allocate_run, ensure_sample
from dc.dc_iv_curve_tui import MEASUREMENT_TYPE, build_plan


def _tui_state(**overrides) -> dict:
    base = dict(
        source_visa_resource="GPIB0::20::INSTR", voltmeter_visa_resource="GPIB0::7::INSTR",
        compliance_V=2.0, source_delay_s=0.05, current_min_A=-1e-3, current_max_A=1e-3,
        nplc=5, auto_range=True, settling_time_s=0.2, n_averages=5,
        device="HB3", cooldown="", temperature_setpoint_K=300.0,
        step_A=5e-5, bidirectional_sweep=True, enable_gate=False,
        gate_visa_resource="GPIB0::25::INSTR", gate_voltage_limit_V=20.0,
        gate_compliance_current_A=1e-6, gate_voltage_values="0.0", gate_voltage_list=[],
        enable_temperature=False, temperature_visa_resource="", temperature_sensor_uids="",
        sample="A",
    )
    base.update(overrides)
    return base


def test_tui_build_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.DCIVCurveApp()
    app.data_root = tmp_path
    plan = app._build_plan(_tui_state())
    assert plan.series == ""


def test_build_plan_series_and_run_numbering(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    state = _tui_state(data_dir=str(tmp_path), gate_voltage_list=[0.0, 1.0], enable_gate=True)
    plan = build_plan(state, tmp_path)
    assert plan.series.startswith("A_HB3_IV_")

    contexts = [
        allocate_run(tmp_path, plan.sample, plan.device, MEASUREMENT_TYPE,
                     temperature_setpoint_K=plan.temperature_setpoint_K,
                     key_axis=("gate_V", gv), series=plan.series)
        for gv in plan.series_values
    ]
    assert [c.run_number for c in contexts] == [1, 2]


def test_iv_estimate_counts_more_than_settle_plus_reads(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    state = _tui_state(data_dir=str(tmp_path))
    n = 81                                                       # ±1 mA, 50 µA step, bidirectional
    info, _, errors = tui.build_summary(state)
    assert not errors
    line = next(i for i in info if i.startswith("Run time"))
    rc = tui.run_costs(n, state)
    assert len(rc.points) == n
    old_estimate = n * (0.2 + 5 * 5 / 50.0)                      # what the sidebar used to say: 57 s
    assert rc.total_s > old_estimate + 3.0 + 1.5                 # + per-run, per-file, GPIB/CSV per point
    assert rc.total_s == pytest.approx(n * (0.2 + 5 * (5 / 50.0 + 0.02) + 0.02 + 0.10) + 1.5 + 3.0 + 0.2)
    assert line.startswith("Run time: ≈ ")


def test_iv_estimate_multiplies_gate_series_and_plan_carries_cost(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    state = _tui_state(enable_gate=True, gate_voltage_list=[0.0, 1.0, 2.0])
    rc = tui.run_costs(81, state)
    assert len(rc.points) == 3 * 81
    assert rc.parts["per-file"] == pytest.approx(3 * 1.5)
    app = tui.DCIVCurveApp()
    app.data_root = tmp_path
    plan = app._build_plan(state)
    assert plan.run_cost is not None and len(plan.run_cost.points) == plan.total_points


def test_raw_file_rewrite_is_throttled(monkeypatch) -> None:
    """One write per save_every_s, not per point; every point still returned."""
    import dc.dc_iv_curve as iv
    from types import SimpleNamespace

    clock = iter(range(0, 1000))      # 1 s per monotonic() call
    monkeypatch.setattr(iv.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(iv.time, "sleep", lambda s: None)
    source = SimpleNamespace(source_current=0.0)
    voltmeter = SimpleNamespace(voltage=1e-3)
    src_cfg = iv.SourceConfig()
    points = [iv.CurrentPoint(current_A=i * 1e-4) for i in range(10)]
    writes = []
    df = iv.run_measurement(source, voltmeter, src_cfg, iv.AcquisitionConfig(save_every_s=5.0),
                            points, write_csv=lambda recs: writes.append(len(recs)))
    assert len(df) == 10
    assert writes == [1, 6]
