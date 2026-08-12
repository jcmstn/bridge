"""
Plan-purity test for web/mfli/diff_resistance.py's build_plan().

No NiceGUI page render or hardware needed -- build_plan() is a plain
function of a state dict, side-effecting only via allocate_run() (naming/
index writes) against whatever data_dir is in `state`.
"""

from __future__ import annotations

from pathlib import Path

from instruments.data_naming import ensure_sample
from web.mfli.diff_resistance import build_plan


def _state(data_dir: Path, **overrides) -> dict:
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
        sample="A", data_dir=str(data_dir),
    )
    base.update(overrides)
    return base


def test_build_plan_allocates_run_and_matches_filename_convention(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)

    plan1 = build_plan(_state(tmp_path))
    assert plan1.run_ctx.run_number == 1
    assert plan1.acq_cfg.output_file == str(plan1.run_ctx.raw_path)
    assert Path(plan1.acq_cfg.output_file).name.startswith("A_0001_HB3_DIFFR_T300K_")

    plan2 = build_plan(_state(tmp_path))
    assert plan2.run_ctx.run_number == 2
