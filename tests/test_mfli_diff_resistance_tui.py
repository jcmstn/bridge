"""
Plan-purity test for mfli/mfli_diff_resistance_tui.py's _build_plan().

No hardware and no running Textual app loop needed -- _build_plan() only
touches the filesystem via allocate_run() (naming/index side effects),
so we exercise it directly against a tmp_path data root.
"""

from __future__ import annotations

from pathlib import Path

import mfli.mfli_diff_resistance_tui as tui
from instruments.data_naming import ensure_sample


def _state(**overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886",
        daq_host="localhost", daq_port=8004,
        frequency_Hz=137.0, ac_amplitude_V=0.005, series_R_ohm=100000.0,
        bias_min_V=-0.5, bias_max_V=0.5,
        time_constant_s=0.3, order=4, sinc_filter=True,
        current_input_range_A=1e-6, voltage_input_range_V=0.1, sample_rate_Hz=857.0,
        settling_time_s=1.5, n_averages=50,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        n_points=41, enable_temperature=False,
        temperature_visa_resource="", temperature_sensor_uids="",
        sample="A",
    )
    base.update(overrides)
    return base


def test_build_plan_allocates_run_and_matches_filename_convention(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.MFLIDiffResistanceApp()
    app.data_root = tmp_path

    plan1 = app._build_plan(_state())
    assert plan1.run_ctx.run_number == 1
    assert plan1.acq_cfg.output_file == str(plan1.run_ctx.raw_path)
    assert Path(plan1.acq_cfg.output_file).name.startswith("A_0001_HB3_DIFFR_T300K_")

    plan2 = app._build_plan(_state())
    assert plan2.run_ctx.run_number == 2
