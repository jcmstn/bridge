"""Plan-purity tests for dc_gate_sweep_tui.py and web/dc/gate_sweep.py."""

from __future__ import annotations

from pathlib import Path

import dc.dc_gate_sweep_tui as tui
from instruments.data_naming import allocate_run, ensure_sample
from web.dc.gate_sweep import MEASUREMENT_TYPE, build_plan as web_build_plan


def _state(**overrides) -> dict:
    base = dict(
        source_visa_resource="GPIB0::20::INSTR", voltmeter_visa_resource="GPIB0::7::INSTR",
        gate_visa_resource="GPIB0::25::INSTR",
        sense_current_A=1e-6, compliance_V=2.0, source_delay_s=0.05, nplc=5,
        auto_range=True, settling_time_s=0.2, n_averages=5,
        device="HB3", cooldown="", temperature_setpoint_K=300.0,
        gate_voltage_limit_V=20.0, gate_compliance_current_A=1e-6,
        gate_min_V=-10.0, gate_max_V=10.0, step_V=0.5, bidirectional_sweep=True,
        enable_field=True, magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05, field_settle_s=1.0,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05, field_settle_tolerance_mT=0.02, field_current_values="0, 1", field_current_list=[0.0, 1.0],
        enable_temperature=False, temperature_visa_resource="", temperature_sensor_uids="",
        sample="A",
    )
    base.update(overrides)
    return base


def test_tui_build_plan_series(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.DCGateSweepApp()
    plan = app._build_plan(_state())
    assert plan.series.startswith("A_HB3_GSWP_")


def test_web_build_plan_and_per_iteration_allocation(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    state = _state(data_dir=str(tmp_path))
    plan = web_build_plan(state)
    assert plan.series.startswith("A_HB3_GSWP_")

    contexts = [
        allocate_run(tmp_path, plan.sample, plan.device, MEASUREMENT_TYPE,
                     temperature_setpoint_K=plan.temperature_setpoint_K,
                     key_axis=("current_A", i), series=plan.series)
        for i in plan.series_values
    ]
    assert [c.run_number for c in contexts] == [1, 2]
    assert contexts[0].raw_path.name.startswith("A_0001_HB3_GSWP_T300K_I0A_")
