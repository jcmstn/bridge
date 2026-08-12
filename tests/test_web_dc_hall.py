"""
Plan-purity test for web/dc/hall.py's build_plan().

No NiceGUI page render or hardware needed -- build_plan() is a plain
function of a state dict, side-effecting only via allocate_run() (naming/
index writes) against whatever data_dir is in `state`.
"""

from __future__ import annotations

from pathlib import Path

from instruments.data_naming import ensure_sample
from web.dc.hall import build_plan


def _state(data_dir: Path, **overrides) -> dict:
    base = dict(
        source_visa_resource="GPIB0::20::INSTR",
        voltmeter_visa_resource="GPIB0::7::INSTR",
        sense_current_A=0.001, compliance_V=2.0, source_delay_s=0.05, nplc=5,
        auto_range=True, settling_time_s=1.0, n_reversals=5,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        enable_sweep=False,
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        i_min_A=-20.0, i_max_A=20.0, step_A=2.0, bidirectional_sweep=True,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05, enable_temperature=False,
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
    assert Path(plan1.acq_cfg.output_file).name.startswith("A_0001_HB3_HALL_T300K_")

    plan2 = build_plan(_state(tmp_path))
    assert plan2.run_ctx.run_number == 2
