"""Plan-purity test for web/dc/spin_valve.py's build_plan() and the
multi-file-per-session (one file per sense-current x gate-voltage
combination) run-numbering logic."""

from __future__ import annotations

from pathlib import Path

from instruments.data_naming import allocate_run, ensure_sample
from web.dc.spin_valve import MEASUREMENT_TYPE, build_plan


def _state(data_dir: Path, **overrides) -> dict:
    base = dict(
        source_visa_resource="GPIB0::20::INSTR",
        voltmeter_visa_resource="GPIB0::7::INSTR",
        gate_visa_resource="GPIB0::25::INSTR",
        sense_current_values="0.001", sense_current_list=[0.001],
        compliance_V=2.0, source_delay_s=0.05, nplc=5,
        auto_range=True, settling_time_s=1.0, n_averages=5,
        device="SV2", cooldown="3", temperature_setpoint_K=10.0,
        enable_gate=True, gate_voltage_limit_V=20.0, gate_compliance_current_A=1e-6,
        gate_voltage_values="0, 5", gate_voltage_list=[0.0, 5.0],
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        i_min_A=-20.0, i_max_A=20.0, step_A=2.0, bidirectional_sweep=True,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05, field_settle_tolerance_mT=0.02, enable_temperature=False,
        temperature_visa_resource="", temperature_sensor_uids="",
        reversal_enabled=True, sample="A", data_dir=str(data_dir),
    )
    base.update(overrides)
    return base


def test_build_plan_and_per_iteration_allocation(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    plan = build_plan(_state(tmp_path))
    assert plan.series.startswith("A_SV2_BSWP_")

    contexts = [
        allocate_run(tmp_path, plan.sample, plan.device, MEASUREMENT_TYPE,
                     temperature_setpoint_K=plan.temperature_setpoint_K,
                     key_axis=("gate_V", gv), series=plan.series)
        for _, gv in plan.series_values
    ]
    assert [c.run_number for c in contexts] == [1, 2]
    assert contexts[0].raw_path.name.startswith("A_0001_SV2_BSWP_T010K_Vg0V_")
    assert contexts[1].raw_path.name.startswith("A_0002_SV2_BSWP_T010K_Vg5V_")


def test_sense_current_series_cross_product(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    plan = build_plan(_state(
        tmp_path,
        sense_current_values="0.001, 0.002", sense_current_list=[0.001, 0.002],
        gate_voltage_values="0, 5", gate_voltage_list=[0.0, 5.0],
    ))
    assert len(plan.series_values) == 4

    contexts = [
        allocate_run(tmp_path, plan.sample, plan.device, MEASUREMENT_TYPE,
                     temperature_setpoint_K=plan.temperature_setpoint_K,
                     key_axis=("gate_V", gv), series=plan.series)
        for _, gv in plan.series_values
    ]
    assert [c.run_number for c in contexts] == [1, 2, 3, 4]
