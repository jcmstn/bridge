"""
sot/sot_pulsed_switching_tui.py — build_summary validation + _build_plan
purity. Pure logic only, no Textual mount, no hardware.
"""

from __future__ import annotations

from pathlib import Path

import sot.sot_pulsed_switching_tui as tui


def _state(**overrides) -> dict:
    base = dict(
        k4200_visa_resource="GPIB0::17::INSTR",
        pmu_library="pmu-dut-examples", pmu_module="pulse_iv", pmu_channel=1,
        amplitudes_V="0.2, 0.6, 1.0, 1.4, 1.8", pmu_return_names="",
        pulse_width_s=1e-7, pulse_rise_s=2e-8, pulse_fall_s=2e-8, pulse_period_s=1e-3,
        n_pulses=1, pmu_i_range_A=0.2, pmu_v_limit_V=5.0, pmu_i_limit_A=0.2,
        sense_current_A=1e-4, compliance_V=2.0, source_delay_s=0.05, nplc=5.0,
        auto_range=True, n_reversals=5, settle_after_enable_s=0.3,
        delay_after_pulse_s=5.0, n_repeats=50,
        reset_enabled=True, reset_amplitude_V=-2.0, reset_delay_after_s=0.01,
        magnet_current_A=1.5, field_angle_from_oop_deg=85.0, field_settle_tolerance_mT=0.05,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        source_visa_resource="GPIB0::20::INSTR", voltmeter_visa_resource="GPIB0::7::INSTR",
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05,
        enable_temperature=False, temperature_visa_resource="", temperature_sensor_uids="",
        sample="A", data_dir="",
    )
    base.update(overrides)
    base["amplitude_list"] = [float(v) for v in str(base["amplitudes_V"]).split(",")]
    base["amplitude_parse_error"] = None
    return base


def test_summary_blocks_on_empty_pmu_module():
    _, _, errors = tui.build_summary(_state(pmu_module="  "))
    assert any("PMU module name is empty" in e for e in errors)


def test_summary_blocks_amplitude_over_v_limit():
    _, _, errors = tui.build_summary(_state(amplitudes_V="0.5, 8.0", pmu_v_limit_V=5.0))
    assert any("PMU voltage limit" in e for e in errors)


def test_summary_blocks_bad_pulse_timing():
    _, _, errors = tui.build_summary(_state(pulse_period_s=1e-8))  # < width+rise+fall
    assert any("period must be ≥ width" in e for e in errors)


def test_summary_warns_reset_off():
    _, warnings, _ = tui.build_summary(_state(reset_enabled=False))
    assert any("Reset pulse is OFF" in w for w in warnings)


def test_summary_warns_reset_same_sign_as_writes():
    _, warnings, _ = tui.build_summary(_state(amplitudes_V="0.5, 1.0", reset_amplitude_V=2.0))
    assert any("opposite polarity" in w for w in warnings)


def test_summary_blocks_magnet_current_over_limit():
    _, _, errors = tui.build_summary(_state(magnet_current_A=50.0, current_limit_A=35.0))
    assert any("magnet limit" in e for e in errors)


def test_build_plan_shapes(tmp_path: Path):
    app = tui.SOTPulsedSwitchingApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(amplitudes_V="0.2, 0.6, 1.0", n_repeats=10))

    assert plan.amplitudes_V == [0.2, 0.6, 1.0]
    assert plan.total_points == 3 * 10
    assert plan.series == ""                              # single file
    assert plan.pmu_cfg.module == "pulse_iv"
    assert plan.pmu_cfg.return_names == ()
    assert plan.read_cfg.n_reversals == 5
    assert plan.seq_cfg.reset_enabled is True
    assert plan.magnet_current_A == 1.5
    assert plan.field_angle_from_oop_deg == 85.0
    # 6221 sense current flows through to both the read cfg and the 6221 SourceConfig
    assert plan.src_cfg.sense_current_A == plan.read_cfg.sense_current_A


def test_build_plan_return_names_parsed(tmp_path: Path):
    app = tui.SOTPulsedSwitchingApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(
        pmu_return_names="pulse_voltage_measured_V, pulse_current_measured_A"))
    assert plan.pmu_cfg.return_names == ("pulse_voltage_measured_V", "pulse_current_measured_A")


def test_build_plan_temperature_cfg_gating(tmp_path: Path):
    app = tui.SOTPulsedSwitchingApp()
    app.data_root = tmp_path
    assert app._build_plan(_state(enable_temperature=False)).temp_cfg is None
    p = app._build_plan(_state(enable_temperature=True,
                               temperature_visa_resource="TCPIP0::x::7020::SOCKET",
                               temperature_sensor_uids="MB1.T1"))
    assert p.temp_cfg is not None and p.temp_cfg.sensor_uids == ("MB1.T1",)
