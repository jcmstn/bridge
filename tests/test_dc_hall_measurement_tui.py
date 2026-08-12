"""
Plan-purity test for dc/dc_hall_measurement_tui.py's _build_plan().

No hardware and no running Textual app loop needed -- _build_plan() only
touches the filesystem via allocate_run() (naming/index side effects),
so we exercise it directly against a tmp_path data root.
"""

from __future__ import annotations

from pathlib import Path

import dc.dc_hall_measurement_tui as tui
from instruments.data_naming import ensure_sample


def _state(**overrides) -> dict:
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
        sample="A",
    )
    base.update(overrides)
    return base


def test_build_plan_allocates_run_and_matches_filename_convention(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.DCHallMeasurementApp()

    plan1 = app._build_plan(_state())
    assert plan1.run_ctx.run_number == 1
    assert plan1.acq_cfg.output_file == str(plan1.run_ctx.raw_path)
    assert Path(plan1.acq_cfg.output_file).name.startswith("A_0001_HB3_HALL_T300K_")

    plan2 = app._build_plan(_state())
    assert plan2.run_ctx.run_number == 2
