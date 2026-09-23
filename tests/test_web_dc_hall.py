"""
Plan-purity test for the Hall build_plan() that web/dc/hall.py and the TUI share and the multi-file-per-
session (one file per sense current) run-numbering logic.

No NiceGUI page render or hardware needed -- build_plan() is a plain
function of a state dict; run allocation happens per series iteration
(mirrored here the same way make_run_fn()'s loop does it).
"""

from __future__ import annotations

from pathlib import Path

from instruments.data_naming import allocate_run, ensure_sample
from dc.dc_hall_measurement_tui import MEASUREMENT_TYPE, build_plan


def _state(data_dir: Path, **overrides) -> dict:
    base = dict(
        source_visa_resource="GPIB0::20::INSTR",
        voltmeter_visa_resource="GPIB0::7::INSTR",
        sense_current_values="0.001", sense_current_list=[0.001],
        compliance_V=2.0, source_delay_s=0.05, nplc=5,
        auto_range=True, measure_rxy=True, measure_rxx=False, channel_settle_s=0.02,
        settling_time_s=1.0, n_reversals=5,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        field_theta_deg=None, field_phi_deg=None,
        enable_sweep=False,
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        sweep_rows_parsed=[(-20.0, 20.0, 21)], bidirectional_sweep=True,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05, field_settle_tolerance_mT=0.02, enable_temperature=False,
        temperature_visa_resource="", temperature_sensor_uids="",
        sample="A", data_dir=str(data_dir),
    )
    base.update(overrides)
    return base


def test_build_plan_series_tag_only_set_for_a_real_family(tmp_path: Path) -> None:
    plan_single = build_plan(_state(tmp_path), tmp_path)
    assert plan_single.series == ""

    plan_multi = build_plan(_state(
        tmp_path, sense_current_values="0.001, 0.002",
        sense_current_list=[0.001, 0.002],
    ), tmp_path)
    assert plan_multi.series.startswith("A_HB3_HALL_")


def test_multi_file_session_allocates_one_run_per_sense_current(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    plan = build_plan(_state(
        tmp_path, sense_current_values="0.001, -0.002",
        sense_current_list=[0.001, -0.002],
    ), tmp_path)

    contexts = [
        allocate_run(tmp_path, plan.sample, plan.device, MEASUREMENT_TYPE,
                     temperature_setpoint_K=plan.temperature_setpoint_K,
                     key_axis=("current_A", I), series=plan.series)
        for I in plan.series_values
    ]
    assert [c.run_number for c in contexts] == [1, 2]
    assert contexts[0].raw_path.name.startswith("A_0001_HB3_HALL_T300K_I0p001A_")
    assert contexts[1].raw_path.name.startswith("A_0002_HB3_HALL_T300K_Im0p002A_")


def test_build_plan_multi_row_sweep(tmp_path: Path) -> None:
    plan = build_plan(_state(
        tmp_path, enable_sweep=True,
        sweep_rows_parsed=[(-1.0, 1.0, 10), (1.0, 10.0, 10)],
        bidirectional_sweep=True,
    ), tmp_path)
    assert len(plan.currents_A) == 37
    assert plan.header_extra["field_sweep_rows_A"] == [(-1.0, 1.0, 10), (1.0, 10.0, 10)]
