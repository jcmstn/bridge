"""
Plan-purity test for mfli/mfli_dual_harmonic_6221_tui.py's _build_plan(),
plus the excitation-ceiling error path in build_summary().

No hardware and no running Textual app loop needed -- _build_plan() only
touches the filesystem via allocate_run() (naming/index side effects),
so we exercise it directly against a tmp_path data root.
"""

from __future__ import annotations


import mfli.mfli_dual_harmonic_6221_tui as tui
from instruments.data_naming import ensure_sample


def test_demod_naming_keeps_the_pre_harmonic_select_names():
    assert tui.demod_naming(1, False) == ("1f", "1f")              # leader, classic
    assert tui.demod_naming(2, False) == ("2f", "2f")              # follower, classic
    assert tui.demod_naming(1, True) == ("rxx_1f", "R_xx (1f)")    # the old R_xx mode


def test_build_plan_takes_each_harmonic_from_its_select(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)

    plan_off = tui.build_plan(_state(), tmp_path)
    assert (plan_off.demod1_cfg.harmonic, plan_off.demod2_cfg.harmonic) == (1, 2)
    assert plan_off.measure_rxx is False

    # the old R_xx mode = follower at 1f + R_xx on
    plan_on = tui.build_plan(_state(follower_harmonic=1, measure_rxx=True), tmp_path)
    assert plan_on.demod2_cfg.harmonic == 1
    assert plan_on.measure_rxx is True
    assert tui.plan_naming(plan_on) == (("1f", "1f"), ("rxx_1f", "R_xx (1f)"))

    # R_xx is naming only -- it no longer forces the follower to 1f
    plan_3f = tui.build_plan(_state(leader_harmonic=2, follower_harmonic=3, measure_rxx=True), tmp_path)
    assert (plan_3f.demod1_cfg.harmonic, plan_3f.demod2_cfg.harmonic) == (2, 3)
    assert tui.plan_naming(plan_3f) == (("2f", "2f"), ("rxx_3f", "R_xx (3f)"))


def _state(**overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886",
        daq_host="localhost", daq_port=8004,
        ac_visa_resource="GPIB0::20::INSTR",
        frequency_Hz=317.3, ac_compliance_V=2.0, phasemarker_line=1,
        amplitude_values="1e-7", amplitude_list=[1e-7], amplitude_parse_error=None,
        leader_harmonic=1, follower_harmonic=2, leader_measure_rxx=False, measure_rxx=False,
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


def test_build_plan_does_not_allocate_a_run_upfront(tmp_path, monkeypatch) -> None:
    # allocate_run() now happens once per amplitude, inside RunScreen.do_run()
    # -- _build_plan() itself must stay a pure dataclass-construction step
    # (no filesystem side effects), so a fresh run number is never burned
    # just from opening the run screen.
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)

    plan1 = tui.build_plan(_state(time_constant_1f_s=0.1, time_constant_2f_s=0.5), tmp_path)
    assert plan1.acq_cfg.output_file == ""
    assert plan1.ac_cfg.amplitude_A == 1e-7
    assert plan1.amplitudes_A == [1e-7]
    assert plan1.total_files == 1
    assert plan1.leader_extref_cfg.device == "dev7885"
    assert plan1.follower_extref_cfg.device == "dev7886"
    # 1f and 2f must get independent FilterConfig instances -- regresses if
    # someone re-collapses them into one shared object.
    assert plan1.demod1_cfg.filter is not plan1.demod2_cfg.filter
    assert plan1.demod1_cfg.filter.time_constant_s != plan1.demod2_cfg.filter.time_constant_s
    assert (tmp_path / "A" / "index.csv").read_text().count("\n") <= 1  # header only, no run rows


def test_build_plan_multiple_amplitudes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)

    plan = tui.build_plan(_state(amplitude_values="1e-7, 2e-7",
                                    amplitude_list=[1e-7, 2e-7]), tmp_path)
    assert plan.amplitudes_A == [1e-7, 2e-7]
    assert plan.ac_cfg.amplitude_A == 1e-7  # first value, mutated per iteration by do_run()
    assert plan.total_files == 2
    assert plan.series.startswith("A_HB3_HARM6_")


def test_build_summary_flags_excitation_current_ceiling(tmp_path) -> None:
    _, _, errors = tui.build_summary(_state(amplitude_values="50e-3", amplitude_list=[50e-3],
                                              data_dir=str(tmp_path)))
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

    plan = tui.build_plan(_state(
        enable_sweep=True,
        sweep_rows_parsed=[(-1.0, 1.0, 10), (1.0, 10.0, 10)],
    ), tmp_path)
    assert len(plan.currents_A) == 37
    assert plan.header_extra["field_sweep_rows_A"] == [(-1.0, 1.0, 10), (1.0, 10.0, 10)]


def _third_run_records() -> list[dict]:
    """One run out of a multi-current series (series_index=2) -- what
    RunScreen._save_run_png() hands to _save_measurement_png()."""
    return [
        {"point_index": i, "magnet_field_mT": None, "1f_R_V": 1e-3 * i, "2f_R_V": 2e-6 * i,
         "series_index": 2, "series_label": "I=1e-06A", "excitation_current_A_peak": 1e-6}
        for i in range(3)
    ]


def test_save_measurement_png_single_run_looks_like_a_manual_run(tmp_path) -> None:
    # A run's PNG must not depend on which position it had in a multi-current
    # series: 1f blue / follower orange, no legend, not the series-2 color.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    real_close = plt.close
    plt.close = lambda fig=None: (figs.append(fig), real_close(fig))
    try:
        png = tmp_path / "run.png"
        tui._save_measurement_png(_third_run_records(), png)
    finally:
        plt.close = real_close

    assert png.exists()
    ax1, ax2 = figs[0].axes
    assert [ln.get_color() for ln in ax1.lines] == ["tab:blue"]
    assert [ln.get_color() for ln in ax2.lines] == ["tab:orange"]
    assert ax1.get_legend() is None


def test_each_run_of_a_multi_current_series_gets_its_own_plot_png(tmp_path, monkeypatch) -> None:
    # Several excitation currents behave like several manual runs: one
    # <sample>_<run>_<device>_<type>_plot.png per run, never a combined one.

    from instruments.data_naming import allocate_run

    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    amps = [1e-7, 2e-7, 3e-7]
    plan = tui.build_plan(_state(amplitude_values="1e-7, 2e-7, 3e-7", amplitude_list=amps), tmp_path)

    screen = tui.RunScreen.__new__(tui.RunScreen)     # bare: no __init__/mount needed
    screen.plan, screen._png_path = plan, None
    run_strs = []
    for idx, amp in enumerate(amps):
        ctx = allocate_run(tmp_path, "A", "HB3", tui.MEASUREMENT_TYPE, series=plan.series)
        run_strs.append(ctx.run_str)
        records = [{"point_index": i, "magnet_field_mT": None, "1f_R_V": 1e-3 * i,
                    "2f_R_V": 2e-6 * i, "series_index": idx, "series_label": f"I={amp:g}A",
                    "excitation_current_A_peak": amp} for i in range(3)]
        tui.RunScreen._save_run_png(screen, ctx, records)

    names = sorted(p.name for p in (tmp_path / "A" / "proc").glob("*.png"))
    assert names == sorted(f"A_{r}_HB3_{tui.MEASUREMENT_TYPE}_plot.png" for r in run_strs)
    assert len(set(run_strs)) == 3
    assert not any("combined" in n for n in names)
    assert screen._png_path.name.startswith(f"A_{run_strs[-1]}_")
