"""
sot/sot_pulsed_switching_6221_tui.py — build_summary validation + _build_plan
purity. Pure logic only, no Textual mount, no hardware.

Mirrors tests/test_sot_pulsed_switching_2h_tui.py; no 4200A/PMU fields, the
sweep axis is pulse CURRENT not pulse voltage, and harmonic is a parameter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sot.sot_pulsed_switching_6221_tui as tui


def _state(**overrides) -> dict:
    base = dict(
        pulse_current_start_A=1e-3, pulse_current_stop_A=10e-3, pulse_current_step_A=3e-3,
        amplitude_bidirectional=True,
        pulse_width_s=1e-3, pulse_compliance_V=5.0,
        sense_current_A=1e-4, compliance_V=2.0, frequency_Hz=977.0, phasemarker_line=1,
        harmonic=2, n_averages=50, settle_after_enable_s=1.0, lock_timeout_s=5.0,
        delay_after_pulse_s=1.0,
        magnet_current_A="1.5", field_angle_from_oop_deg=85.0, field_settle_tolerance_mT=0.05,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        source_visa_resource="GPIB0::20::INSTR",
        mfli_host="localhost", mfli_port=8004, mfli_device="dev1234",
        aux_input_ch=0, osc_index=0, extref_index=0, pll_demod_index=0, automode=4, demod_index=1,
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
    base["pulse_current_list"], base["pulse_current_parse_error"] = \
        tui._resolve_pulse_currents(base)
    base["magnet_currents_A"], base["magnet_currents_parse_error"] = \
        tui._resolve_magnet_currents(base)
    return base


def test_summary_blocks_zero_step():
    _, _, errors = tui.build_summary(_state(pulse_current_step_A=0.0))
    assert any("Pulse currents:" in e for e in errors)


def test_summary_blocks_pulse_current_over_hardware_ceiling():
    _, _, errors = tui.build_summary(_state(pulse_current_stop_A=0.5))   # 500 mA
    assert any("hardware range" in e for e in errors)


def test_summary_blocks_pll_demod_collision():
    # extrefs/N/adcselect is read-only on real firmware — the PLL phase-detector
    # demod can't be the same index as the demod reading the real signal.
    _, _, errors = tui.build_summary(_state(pll_demod_index=1, demod_index=1))
    assert any("phase-detector demod" in e for e in errors)


def test_summary_warns_bidirectional_off():
    _, warnings, _ = tui.build_summary(_state(amplitude_bidirectional=False))
    assert any("One-way sweep" in w for w in warnings)


def test_summary_no_warning_when_bidirectional():
    _, warnings, _ = tui.build_summary(_state(amplitude_bidirectional=True))
    assert not any("One-way sweep" in w for w in warnings)


def test_summary_blocks_zero_pulse_width():
    _, _, errors = tui.build_summary(_state(pulse_width_s=0.0))
    assert any("Pulse width" in e for e in errors)


def test_summary_blocks_zero_pulse_compliance():
    _, _, errors = tui.build_summary(_state(pulse_compliance_V=0.0))
    assert any("Pulse compliance" in e for e in errors)


def test_summary_always_carries_joule_heating_reminder():
    info, _, _ = tui.build_summary(_state())
    assert any("Joule heating" in i for i in info)


def test_resolve_pulse_currents_bidirectional_loop():
    amps, err = tui._resolve_pulse_currents(_state(
        pulse_current_start_A=0.0, pulse_current_stop_A=2e-3, pulse_current_step_A=1e-3,
        amplitude_bidirectional=True))
    assert err is None
    assert amps == pytest.approx([0.0, 1e-3, 2e-3, 1e-3, 0.0])


def test_summary_blocks_magnet_current_over_limit():
    _, _, errors = tui.build_summary(_state(magnet_current_A="50.0", current_limit_A=35.0))
    assert any("magnet limit" in e for e in errors)


def test_resolve_magnet_currents_list():
    currents, err = tui._resolve_magnet_currents(_state(magnet_current_A="1, -1, 5"))
    assert err is None
    assert currents == [1.0, -1.0, 5.0]


def test_summary_blocks_zero_sense_current():
    _, _, errors = tui.build_summary(_state(sense_current_A=0.0))
    assert any("6221 AC current amplitude" in e for e in errors)


def test_summary_blocks_read_current_over_safety_ceiling():
    _, _, errors = tui.build_summary(_state(sense_current_A=0.1))   # 100 mA
    assert any("safety ceiling" in e for e in errors)


def test_summary_blocks_compliance_over_safety_ceiling():
    _, _, errors = tui.build_summary(_state(compliance_V=100.0))
    assert any("safety ceiling" in e for e in errors)


def test_summary_no_warning_for_offset_frequency():
    _, warnings, _ = tui.build_summary(_state(frequency_Hz=977.0))
    assert not any("line harmonic" in w for w in warnings)


def test_summary_warns_near_line_frequency_from_below_too():
    _, warnings, _ = tui.build_summary(_state(frequency_Hz=1049.5))
    assert any("line harmonic" in w for w in warnings)


def test_summary_blocks_phasemarker_line_out_of_range():
    _, _, errors = tui.build_summary(_state(phasemarker_line=7))
    assert any("Trigger Link phase-marker line" in e for e in errors)


def test_summary_defaults_to_2f():
    state = _state()
    assert state["harmonic"] == 2
    _, warnings, errors = tui.build_summary(state)
    assert not any("Harmonic must be" in e for e in errors)
    assert not any("expect a much smaller" in w for w in warnings)


def test_summary_blocks_harmonic_below_one():
    _, _, errors = tui.build_summary(_state(harmonic=0))
    assert any("Harmonic must be" in e for e in errors)


def test_summary_warns_high_harmonic():
    _, warnings, _ = tui.build_summary(_state(harmonic=4))
    assert any("expect a much smaller" in w for w in warnings)


def test_build_plan_shapes(tmp_path: Path):
    app = tui.SOTPulsedSwitching6221App()
    app.data_root = tmp_path
    plan = app._build_plan(_state(pulse_current_start_A=1e-3, pulse_current_stop_A=5e-3,
                                  pulse_current_step_A=2e-3, amplitude_bidirectional=False))

    assert plan.pulse_currents_A == pytest.approx([1e-3, 3e-3, 5e-3])
    assert plan.total_points == 3
    assert plan.series == ""
    assert plan.pulse_cfg.width_s == 1e-3 and plan.pulse_cfg.compliance_V == 5.0
    assert plan.read_cfg.harmonic == 2
    assert plan.read_cfg.sense_current_A == 1e-4
    assert plan.read_cfg.delay_after_pulse_s == 1.0
    assert plan.magnet_currents_A == [1.5]
    assert plan.field_angle_from_oop_deg == 85.0
    assert plan.ac_cfg.visa_resource == "GPIB0::20::INSTR"
    assert plan.ac_cfg.amplitude_A == 1e-4 and plan.ac_cfg.compliance_V == 2.0
    assert plan.ac_cfg.frequency_Hz == 977.0 and plan.ac_cfg.phasemarker_line == 1
    assert plan.extref_cfg.device == "dev1234" and plan.extref_cfg.aux_input_ch == 0
    assert plan.demod_cfg.harmonic == 2 and plan.demod_cfg.demod_index == 1
    assert plan.mfli_host == "localhost" and plan.mfli_port == 8004


def test_build_plan_respects_chosen_harmonic(tmp_path: Path):
    app = tui.SOTPulsedSwitching6221App()
    app.data_root = tmp_path
    plan = app._build_plan(_state(harmonic=1))
    assert plan.read_cfg.harmonic == 1
    assert plan.demod_cfg.harmonic == 1
    assert plan.header_extra["harmonic"] == 1


def test_build_plan_multiple_magnet_currents(tmp_path: Path):
    app = tui.SOTPulsedSwitching6221App()
    app.data_root = tmp_path
    plan = app._build_plan(_state(pulse_current_start_A=1e-3, pulse_current_stop_A=5e-3,
                                  pulse_current_step_A=2e-3, amplitude_bidirectional=False,
                                  magnet_current_A="1.5, -1.5, 3"))

    assert plan.magnet_currents_A == [1.5, -1.5, 3.0]
    assert plan.series_values == [1.5, -1.5, 3.0]
    assert plan.total_points == 3 * 3


def test_build_plan_temperature_cfg_gating(tmp_path: Path):
    app = tui.SOTPulsedSwitching6221App()
    app.data_root = tmp_path
    assert app._build_plan(_state(enable_temperature=False)).temp_cfg is None
    p = app._build_plan(_state(enable_temperature=True,
                               temperature_visa_resource="TCPIP0::x::7020::SOCKET",
                               temperature_sensor_uids="MB1.T1"))
    assert p.temp_cfg is not None and p.temp_cfg.sensor_uids == ("MB1.T1",)


def test_build_plan_no_pmu_fields_leak_into_header(tmp_path: Path):
    """The 4200A is gone — nothing PMU-shaped should show up in header_extra."""
    app = tui.SOTPulsedSwitching6221App()
    app.data_root = tmp_path
    plan = app._build_plan(_state())
    assert not any(k.startswith("pmu_") for k in plan.header_extra)
