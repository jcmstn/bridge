"""Plan-purity test for web/dc/gate_sweep.py's build_plan() and the
multi-file-per-session (one file per parked magnet current) run-numbering."""

from __future__ import annotations

from pathlib import Path

from instruments.data_naming import allocate_run, ensure_sample
from web.dc.gate_sweep import MEASUREMENT_TYPE, build_plan


def _state(data_dir: Path, **overrides) -> dict:
    base = dict(
        source_visa_resource="GPIB0::20::INSTR",
        voltmeter_visa_resource="GPIB0::7::INSTR",
        gate_visa_resource="GPIB0::25::INSTR",
        sense_current_A=1e-6, compliance_V=2.0, source_delay_s=0.05, nplc=5,
        auto_range=True, settling_time_s=0.2, n_averages=5,
        device="HB3", cooldown="", temperature_setpoint_K=10.0,
        gate_min_V=-10.0, gate_max_V=10.0, step_V=0.5, bidirectional_sweep=True,
        gate_voltage_limit_V=20.0, gate_compliance_current_A=1e-6,
        enable_field=True, field_current_values="0, 5", field_current_list=[0.0, 5.0],
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        field_settle_s=1.0, field_settle_tolerance_mT=0.02,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05,
        enable_temperature=False, temperature_visa_resource="", temperature_sensor_uids="",
        sample="A", data_dir=str(data_dir),
    )
    base.update(overrides)
    return base


def test_build_plan_and_per_magnet_current_allocation(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    plan = build_plan(_state(tmp_path))
    assert plan.series.startswith("A_HB3_GSWP_")
    assert plan.field_currents_A == [0.0, 5.0]

    contexts = [
        allocate_run(tmp_path, plan.sample, plan.device, MEASUREMENT_TYPE,
                     temperature_setpoint_K=plan.temperature_setpoint_K,
                     key_axis=("current_A", i), series=plan.series)
        for i in plan.field_currents_A
    ]
    assert [c.run_number for c in contexts] == [1, 2]


def test_single_magnet_current_gets_no_series_tag(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    plan = build_plan(_state(tmp_path, field_current_values="0",
                             field_current_list=[0.0]))
    assert plan.series == ""


def test_config_mapping_matches_tui(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    plan = build_plan(_state(tmp_path))
    assert plan.src_cfg.sense_current_A == 1e-6
    assert plan.gate_cfg.gate_voltage_limit_V == 20.0
    assert plan.magnet_cfg is not None and plan.gauss_cfg is not None
    assert plan.temp_cfg is None
