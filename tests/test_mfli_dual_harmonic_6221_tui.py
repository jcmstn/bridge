"""
Plan-purity test for mfli/mfli_dual_harmonic_6221_tui.py's _build_plan(),
plus the excitation-ceiling error path in build_summary().

No hardware and no running Textual app loop needed -- _build_plan() only
touches the filesystem via allocate_run() (naming/index side effects),
so we exercise it directly against a tmp_path data root.
"""

from __future__ import annotations

from pathlib import Path

import mfli.mfli_dual_harmonic_6221_tui as tui
from instruments.data_naming import ensure_sample


def _state(**overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886",
        daq_host="localhost", daq_port=8004,
        ac_visa_resource="GPIB0::20::INSTR",
        frequency_Hz=317.3, amplitude_A=1e-7, ac_compliance_V=2.0, phasemarker_line=1,
        time_constant_1f_s=0.3, order_1f=4, sinc_filter_1f=True,
        time_constant_2f_s=0.3, order_2f=4, sinc_filter_2f=True,
        differential=True, ac_coupling=True,
        input_range_1f_V=1.0, input_range_2f_V=1.0, sample_rate_Hz=857.0,
        settling_time_s=15.0, n_averages=50,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        enable_sweep=False,
        visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        sweep_rows_parsed=[(-20.0, 20.0, 21)],
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05, field_settle_tolerance_mT=0.02, enable_temperature=False,
        temperature_visa_resource="", temperature_sensor_uids="",
        enable_phase_cal=False, phase_cal_current_A=None,
        phase_cal_n_averages=20, phase_cal_max_iterations=5,
        hall_bar_length_um=None, hall_bar_width_um=None,
        hall_bar_thickness_nm=None, field_theta_deg=None, field_phi_deg=None,
        leader_extref_index=0, leader_aux_input_ch=0, leader_osc_index=0, leader_pll_demod_index=1,
        leader_automode=4,
        follower_extref_index=0, follower_aux_input_ch=0, follower_osc_index=0,
        follower_pll_demod_index=1, follower_automode=4,
        extref_lock_timeout_s=5.0,
        sample="A",
    )
    base.update(overrides)
    return base


def test_build_plan_allocates_run_and_matches_filename_convention(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.MFLIDualHarmonic6221App()
    app.data_root = tmp_path

    plan1 = app._build_plan(_state(time_constant_1f_s=0.1, time_constant_2f_s=0.5))
    assert plan1.run_ctx.run_number == 1
    assert plan1.acq_cfg.output_file == str(plan1.run_ctx.raw_path)
    assert Path(plan1.acq_cfg.output_file).name.startswith("A_0001_HB3_HARM6_T300K_")
    assert plan1.ac_cfg.amplitude_A == 1e-7
    assert plan1.leader_extref_cfg.device == "dev7885"
    assert plan1.follower_extref_cfg.device == "dev7886"
    # 1f and 2f must get independent FilterConfig instances -- regresses if
    # someone re-collapses them into one shared object.
    assert plan1.demod1_cfg.filter is not plan1.demod2_cfg.filter
    assert plan1.demod1_cfg.filter.time_constant_s != plan1.demod2_cfg.filter.time_constant_s

    plan2 = app._build_plan(_state())
    assert plan2.run_ctx.run_number == 2


def test_build_summary_flags_excitation_current_ceiling(tmp_path) -> None:
    _, _, errors = tui.build_summary(_state(amplitude_A=50e-3, data_dir=str(tmp_path)))
    assert any("Excitation current" in e for e in errors)


def test_build_summary_ok_for_default_state(tmp_path) -> None:
    _, _, errors = tui.build_summary(_state(data_dir=str(tmp_path)))
    assert errors == []


def test_build_summary_flags_pll_demod_collision_with_signal_demod(tmp_path) -> None:
    # demod 0 reads the real 1f/2f signal (see _build_plan) — extrefs/N/adcselect
    # is read-only on real firmware, so the PLL detector can't reuse it.
    _, _, errors = tui.build_summary(_state(data_dir=str(tmp_path), leader_pll_demod_index=0))
    assert any("Leader PLL phase-detector demod" in e for e in errors)
    _, _, errors = tui.build_summary(_state(data_dir=str(tmp_path), follower_pll_demod_index=0))
    assert any("Follower PLL phase-detector demod" in e for e in errors)


def test_build_plan_multi_row_sweep(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.MFLIDualHarmonic6221App()
    app.data_root = tmp_path

    plan = app._build_plan(_state(
        enable_sweep=True,
        sweep_rows_parsed=[(-1.0, 1.0, 10), (1.0, 10.0, 10)],
    ))
    assert len(plan.currents_A) == 37
    assert plan.header_extra["field_sweep_rows_A"] == [(-1.0, 1.0, 10), (1.0, 10.0, 10)]
