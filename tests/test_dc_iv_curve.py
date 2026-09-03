"""Plan-purity tests for dc_iv_curve_tui.py and web/dc/iv_curve.py."""

from __future__ import annotations

from pathlib import Path

import dc.dc_iv_curve_tui as tui
from instruments.data_naming import allocate_run, ensure_sample
from web.dc.iv_curve import MEASUREMENT_TYPE, build_plan as web_build_plan


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


def test_web_build_plan_and_series(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    state = _tui_state(data_dir=str(tmp_path), gate_voltage_list=[0.0, 1.0], enable_gate=True)
    plan = web_build_plan(state)
    assert plan.series.startswith("A_HB3_IV_")

    contexts = [
        allocate_run(tmp_path, plan.sample, plan.device, MEASUREMENT_TYPE,
                     temperature_setpoint_K=plan.temperature_setpoint_K,
                     key_axis=("gate_V", gv), series=plan.series)
        for gv in plan.series_values
    ]
    assert [c.run_number for c in contexts] == [1, 2]
