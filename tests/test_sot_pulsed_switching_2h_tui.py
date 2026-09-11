"""
sot/sot_pulsed_switching_2h_tui.py — build_summary validation + _build_plan
purity. Pure logic only, no Textual mount, no hardware.

Mirrors tests/test_sot_pulsed_switching_tui.py; only the read-side
(6221 AC + MFLI 1f/2f, no 2182/n_reversals) fields differ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sot.sot_pulsed_switching_2h_tui as tui


def _state(**overrides) -> dict:
    base = dict(
        k4200_visa_resource="GPIB0::17::INSTR",
        pmu_library="bridge_sot", pmu_module="bridge_sot_pulse", pmu_channel=1,
        pmu_id="PMU1",
        amplitude_start_V=0.2, amplitude_stop_V=2.0, amplitude_step_V=0.4,
        amplitude_bidirectional=True, pmu_return_names="",
        pulse_width_s=1e-7, pulse_rise_s=2e-8, pulse_fall_s=2e-8, pulse_period_s=1e-3,
        pulse_delay_s=0.0, n_pulses=1,
        pmu_sample_rate=2e8, pmu_meas_start_perc=0.75, pmu_meas_stop_perc=0.90,
        pmu_dut_res_ohm=1000.0, pmu_v_range_V=10.0, pmu_i_range_A=0.01, pmu_v_limit_V=5.0,
        sense_current_A=1e-4, compliance_V=2.0, frequency_Hz=977.0, phasemarker_line=1,
        n_averages=50, settle_after_enable_s=1.0, lock_timeout_s=5.0,
        delay_after_pulse_s=1.0,
        magnet_current_A="1.5", field_angle_from_oop_deg=85.0, field_settle_tolerance_mT=0.05,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        source_visa_resource="GPIB0::20::INSTR",
        mfli_host="localhost", mfli_port=8004, mfli_device="dev1234",
        aux_input_ch=0, osc_index=0, extref_index=0, demod1_index=1, demod2_index=2,
        input_ch=0, input_range_V=1.0, sample_rate_Hz=857.0,
        filter_time_constant_s=0.3, filter_order=4,
        differential=True, ac_coupling=True, filter_sinc=True,
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        magnet_voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05,
        enable_temperature=False, temperature_visa_resource="", temperature_sensor_uids="",
        sample="A", data_dir="",
    )
    base.update(overrides)
    base["amplitude_list"], base["amplitude_parse_error"] = tui._resolve_amplitudes(base)
    base["magnet_currents_A"], base["magnet_currents_parse_error"] = \
        tui._resolve_magnet_currents(base)
    return base


def test_summary_blocks_on_empty_pmu_module():
    _, _, errors = tui.build_summary(_state(pmu_module="  "))
    assert any("PMU module name is empty" in e for e in errors)


def test_summary_blocks_amplitude_over_v_limit():
    _, _, errors = tui.build_summary(_state(amplitude_stop_V=8.0, pmu_v_limit_V=5.0))
    assert any("PMU voltage limit" in e for e in errors)


def test_summary_blocks_zero_step():
    _, _, errors = tui.build_summary(_state(amplitude_step_V=0.0))
    assert any("Pulse amplitudes:" in e for e in errors)


def test_summary_blocks_bad_pulse_timing():
    _, _, errors = tui.build_summary(_state(pulse_period_s=1e-8))  # < delay+width+rise+fall
    assert any("period must be ≥ delay + width" in e for e in errors)


def test_summary_blocks_no_pulse_top():
    # width 100 ns, rise/fall 200 ns → settled top < 0 (bench -826)
    _, _, errors = tui.build_summary(_state(pulse_width_s=1e-7, pulse_rise_s=2e-7,
                                            pulse_fall_s=2e-7))
    assert any("No flat pulse top" in e for e in errors)


def test_summary_warns_bidirectional_off():
    _, warnings, _ = tui.build_summary(_state(amplitude_bidirectional=False))
    assert any("One-way sweep" in w for w in warnings)


def test_summary_no_warning_when_bidirectional():
    _, warnings, _ = tui.build_summary(_state(amplitude_bidirectional=True))
    assert not any("One-way sweep" in w for w in warnings)


def test_resolve_amplitudes_bidirectional_loop():
    amps, err = tui._resolve_amplitudes(_state(
        amplitude_start_V=0.0, amplitude_stop_V=1.0, amplitude_step_V=0.5,
        amplitude_bidirectional=True))
    assert err is None
    assert amps == [0.0, 0.5, 1.0, 0.5, 0.0]


def test_summary_blocks_magnet_current_over_limit():
    _, _, errors = tui.build_summary(_state(magnet_current_A="50.0", current_limit_A=35.0))
    assert any("magnet limit" in e for e in errors)


def test_summary_blocks_magnet_current_parse_error():
    _, _, errors = tui.build_summary(_state(magnet_current_A="not-a-number"))
    assert any("Magnet current(s)" in e for e in errors)


def test_resolve_magnet_currents_list():
    currents, err = tui._resolve_magnet_currents(_state(magnet_current_A="1, -1, 5"))
    assert err is None
    assert currents == [1.0, -1.0, 5.0]


def test_summary_blocks_zero_sense_current():
    _, _, errors = tui.build_summary(_state(sense_current_A=0.0))
    assert any("6221 AC current amplitude" in e for e in errors)


def test_summary_blocks_zero_compliance():
    _, _, errors = tui.build_summary(_state(compliance_V=0.0))
    assert any("6221 compliance" in e for e in errors)


def test_summary_blocks_read_current_over_safety_ceiling():
    _, _, errors = tui.build_summary(_state(sense_current_A=0.1))   # 100 mA
    assert any("safety ceiling" in e for e in errors)


def test_summary_warns_large_read_current_below_ceiling():
    _, warnings, errors = tui.build_summary(_state(sense_current_A=2e-3))
    assert not any("safety ceiling" in e for e in errors)
    assert any("large" in w for w in warnings)


def test_summary_blocks_compliance_over_safety_ceiling():
    _, _, errors = tui.build_summary(_state(compliance_V=100.0))
    assert any("safety ceiling" in e for e in errors)


def test_summary_warns_high_compliance_below_ceiling():
    _, warnings, errors = tui.build_summary(_state(compliance_V=10.0))
    assert not any("safety ceiling" in e for e in errors)
    assert any("headroom" in w for w in warnings)


def test_summary_warns_40v_pmu_range():
    _, warnings, _ = tui.build_summary(_state(pmu_v_range_V=40.0, pulse_width_s=5e-7,
                                              pulse_rise_s=1e-7, pulse_fall_s=1e-7))
    assert any("40 V PMU range" in w for w in warnings)


def test_summary_warns_pulse_current_over_rpm_measure_ceiling():
    # 2 V across 100 Ω = 20 mA, past the RPM's 10 mA measure range
    _, warnings, _ = tui.build_summary(_state(amplitude_start_V=0.5, amplitude_stop_V=2.0,
                                              amplitude_step_V=1.5, pmu_dut_res_ohm=100.0,
                                              pmu_v_range_V=10.0, pmu_i_range_A=0.01))
    assert any("measure ceiling" in w for w in warnings)


def test_summary_blocks_edge_below_range_minimum():
    # 40 V range needs rise/fall ≥ 100 ns; 20 ns is fine on the 10 V range
    _, _, errors = tui.build_summary(_state(pmu_v_range_V=40.0))
    assert any("Rise/fall must be" in e for e in errors)
    _, _, errors_10v = tui.build_summary(_state(pmu_v_range_V=10.0))
    assert not any("Rise/fall must be" in e for e in errors_10v)


def test_summary_blocks_frequency_out_of_wave_range():
    _, _, errors = tui.build_summary(_state(frequency_Hz=2e5))   # above 1e5 Hz WAVE ceiling
    assert any("6221 AC frequency" in e for e in errors)


def test_summary_warns_near_line_frequency_from_below_too():
    # 1049.5 Hz is 0.5 Hz below the 1050 Hz (21st) harmonic of 50 Hz — the
    # naive `f % 50 < 1` check misses this side; nearest-multiple must not.
    _, warnings, _ = tui.build_summary(_state(frequency_Hz=1049.5))
    assert any("line harmonic" in w for w in warnings)


def test_summary_no_warning_for_offset_frequency():
    _, warnings, _ = tui.build_summary(_state(frequency_Hz=977.0))
    assert not any("line harmonic" in w for w in warnings)


def test_summary_blocks_phasemarker_line_out_of_range():
    _, _, errors = tui.build_summary(_state(phasemarker_line=7))
    assert any("Trigger Link phase-marker line" in e for e in errors)


def test_summary_blocks_zero_n_averages():
    _, _, errors = tui.build_summary(_state(n_averages=0))
    assert any("samples averaged" in e for e in errors)


def test_summary_blocks_negative_lock_timeout():
    _, _, errors = tui.build_summary(_state(lock_timeout_s=-1.0))
    assert any("lock timeout" in e for e in errors)


def test_build_plan_shapes(tmp_path: Path):
    app = tui.SOTPulsedSwitching2HApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(amplitude_start_V=0.2, amplitude_stop_V=1.0,
                                  amplitude_step_V=0.4, amplitude_bidirectional=False))

    assert plan.amplitudes_V == pytest.approx([0.2, 0.6, 1.0])
    assert plan.total_points == 3                         # one pulse per amplitude
    assert plan.series == ""                              # single file
    assert plan.pmu_cfg.module == "bridge_sot_pulse"
    assert plan.pmu_cfg.return_names == ()
    assert plan.pmu_cfg.v_range_V == 10.0
    assert plan.read_cfg.sense_current_A == 1e-4
    assert plan.read_cfg.n_averages == 50
    assert plan.read_cfg.delay_after_pulse_s == 1.0
    assert plan.magnet_currents_A == [1.5]
    assert plan.field_angle_from_oop_deg == 85.0
    # 6221 sources the AC read current, phase marker → MFLI ExtRef
    assert plan.ac_cfg.visa_resource == "GPIB0::20::INSTR"
    assert plan.ac_cfg.amplitude_A == 1e-4 and plan.ac_cfg.compliance_V == 2.0
    assert plan.ac_cfg.frequency_Hz == 977.0 and plan.ac_cfg.phasemarker_line == 1
    assert plan.extref_cfg.device == "dev1234" and plan.extref_cfg.aux_input_ch == 0
    assert plan.demod1_cfg.harmonic == 1 and plan.demod1_cfg.demod_index == 1
    assert plan.demod2_cfg.harmonic == 2 and plan.demod2_cfg.demod_index == 2
    assert plan.demod1_cfg.device == plan.demod2_cfg.device == "dev1234"
    assert plan.mfli_host == "localhost" and plan.mfli_port == 8004


def test_build_plan_multiple_magnet_currents(tmp_path: Path):
    app = tui.SOTPulsedSwitching2HApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(amplitude_start_V=0.2, amplitude_stop_V=1.0,
                                  amplitude_step_V=0.4, amplitude_bidirectional=False,
                                  magnet_current_A="1.5, -1.5, 3"))

    assert plan.magnet_currents_A == [1.5, -1.5, 3.0]
    assert plan.series_values == [1.5, -1.5, 3.0]
    assert plan.total_points == 3 * 3                     # amplitudes × assist currents


def test_build_plan_channel_resistance_field_is_gone(tmp_path: Path):
    """The Phase-6 display-only channel_resistance_ohm field is removed; the
    load-line pmu_dut_res_ohm (a real pulse parameter) is what's recorded."""
    app = tui.SOTPulsedSwitching2HApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(pmu_dut_res_ohm=250.0))
    assert "channel_resistance_ohm" not in plan.header_extra
    assert plan.header_extra["pmu_dut_res_ohm"] == 250.0
    assert plan.header_extra["sense_current_A"] == 1e-4
    assert plan.header_extra["frequency_Hz"] == 977.0
    assert plan.header_extra["phasemarker_line"] == 1


def test_build_plan_return_names_parsed(tmp_path: Path):
    app = tui.SOTPulsedSwitching2HApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(
        pmu_return_names="pulse_voltage_measured_V, pulse_current_measured_A"))
    assert plan.pmu_cfg.return_names == ("pulse_voltage_measured_V", "pulse_current_measured_A")


def test_build_plan_temperature_cfg_gating(tmp_path: Path):
    app = tui.SOTPulsedSwitching2HApp()
    app.data_root = tmp_path
    assert app._build_plan(_state(enable_temperature=False)).temp_cfg is None
    p = app._build_plan(_state(enable_temperature=True,
                               temperature_visa_resource="TCPIP0::x::7020::SOCKET",
                               temperature_sensor_uids="MB1.T1"))
    assert p.temp_cfg is not None and p.temp_cfg.sensor_uids == ("MB1.T1",)
