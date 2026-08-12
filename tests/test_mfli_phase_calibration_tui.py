"""
Plan-purity test for mfli/mfli_phase_calibration_tui.py's _build_plan().

No hardware and no running Textual app loop needed -- _build_plan() only
touches the filesystem via allocate_run() (naming/index side effects),
so we exercise it directly against a tmp_path data root.
"""

from __future__ import annotations

from pathlib import Path

import mfli.mfli_phase_calibration_tui as tui
from instruments.data_naming import ensure_sample


def _state(**overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886",
        daq_host="localhost", daq_port=8004,
        frequency_Hz=17.777, amplitude_V=0.1, series_R_ohm=10000.0,
        time_constant_s=0.3, order=4, sinc_filter=True,
        input_range_1f_V=1.0, input_range_2f_V=1.0, sample_rate_Hz=857.0,
        visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05,
        calibration_current_A=20.0, i_min_A=-20.0, i_max_A=20.0, n_points=11,
        sweep_settling_time_s=1.5, sweep_n_averages=20, hold_tol_ratio=0.02,
        null_n_averages=20, null_max_iterations=5, null_tol_deg=0.02,
        enable_amplitude_check=False, amplitudes_V=[], amp_n_averages=20,
        enable_frequency_check=False, frequencies_Hz=[], freq_n_averages=20,
        freq_max_iterations=5, freq_tol_deg=0.02,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        enable_temperature=False, temperature_visa_resource="", temperature_sensor_uids="",
        sample="A",
    )
    base.update(overrides)
    return base


def test_build_plan_allocates_run_and_matches_filename_convention(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.MFLIPhaseCalibrationApp()

    plan1 = app._build_plan(_state())
    assert plan1.run_ctx.run_number == 1
    assert plan1.output_csv == str(plan1.run_ctx.raw_path)
    assert Path(plan1.output_csv).name.startswith("A_0001_HB3_PHCAL_T300K_")

    plan2 = app._build_plan(_state())
    assert plan2.run_ctx.run_number == 2
