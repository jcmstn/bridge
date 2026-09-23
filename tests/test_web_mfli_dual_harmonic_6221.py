"""
Plan-purity test for shared build_plan() (TUI + web).

No NiceGUI page render or hardware needed -- build_plan() is a plain
function of a state dict, side-effecting only via allocate_run() (naming/
index writes) against whatever data_dir is in `state`.
"""

from __future__ import annotations

from pathlib import Path

from instruments.data_naming import ensure_sample
from mfli.mfli_dual_harmonic_6221_tui import build_plan


def _state(data_dir: Path, **overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886",
        daq_host="localhost", daq_port=8004,
        ac_visa_resource="GPIB0::20::INSTR",
        frequency_Hz=317.3, ac_compliance_V=2.0, phasemarker_line=1,
        amplitude_values="1e-7", amplitude_list=[1e-7], amplitude_parse_error=None,
        measure_rxx=False,
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
        sample="A", data_dir=str(data_dir),
    )
    base.update(overrides)
    return base


def test_build_plan_does_not_allocate_a_run_upfront(tmp_path: Path) -> None:
    # allocate_run() now happens once per amplitude, inside run_fn() -- build_plan()
    # itself must stay a pure dataclass-construction step (no filesystem side
    # effects).
    ensure_sample(tmp_path, "A", create=True)

    plan1 = build_plan(_state(tmp_path), tmp_path)
    assert plan1.acq_cfg.output_file == ""
    assert plan1.ac_cfg.amplitude_A == 1e-7
    assert plan1.amplitudes_A == [1e-7]
    assert (tmp_path / "A" / "index.csv").read_text().count("\n") <= 1  # header only, no run rows


def test_build_plan_multiple_amplitudes(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    plan = build_plan(_state(tmp_path, amplitude_values="1e-7, 2e-7",
                              amplitude_list=[1e-7, 2e-7]), tmp_path)
    assert plan.amplitudes_A == [1e-7, 2e-7]
    assert plan.ac_cfg.amplitude_A == 1e-7
    assert plan.series.startswith("A_HB3_HARM6_")


def test_build_plan_multi_row_sweep(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    plan = build_plan(_state(
        tmp_path, enable_sweep=True,
        sweep_rows_parsed=[(-1.0, 1.0, 10), (1.0, 10.0, 10)],
    ), tmp_path)
    assert len(plan.currents_A) == 37
    assert plan.header_extra["field_sweep_rows_A"] == [(-1.0, 1.0, 10), (1.0, 10.0, 10)]


def test_save_measurement_png_single_run_looks_like_a_manual_run(tmp_path) -> None:
    # Same contract as the TUI's _save_measurement_png: one run's records,
    # colored by quantity (1f blue / follower orange), never by series index.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mfli.mfli_dual_harmonic_6221_tui import _save_measurement_png

    records = [
        {"point_index": i, "magnet_field_mT": None, "1f_R_V": 1e-3 * i, "2f_R_V": 2e-6 * i,
         "series_index": 2, "series_label": "I=1e-06A", "excitation_current_A_peak": 1e-6}
        for i in range(3)
    ]
    figs = []
    real_close = plt.close
    plt.close = lambda fig=None: (figs.append(fig), real_close(fig))
    try:
        png = tmp_path / "run.png"
        _save_measurement_png(records, png)
    finally:
        plt.close = real_close

    assert png.exists()
    ax1, ax2 = figs[0].axes
    assert [ln.get_color() for ln in ax1.lines] == ["tab:blue"]
    assert [ln.get_color() for ln in ax2.lines] == ["tab:orange"]
    assert ax1.get_legend() is None
