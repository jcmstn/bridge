"""
Plan-purity test for dc/dc_hall_measurement_tui.py's _build_plan() and the
multi-file-per-session (one file per sense current) run-numbering logic.

No hardware and no running Textual app loop needed -- _build_plan() only
builds config dataclasses; run allocation happens per series iteration
(mirrored here the same way RunScreen.do_run() does it).
"""

from __future__ import annotations

from pathlib import Path

import dc.dc_hall_measurement_tui as tui
from instruments.data_naming import allocate_run, ensure_sample


def _state(**overrides) -> dict:
    base = dict(
        source_visa_resource="GPIB0::20::INSTR",
        voltmeter_visa_resource="GPIB0::7::INSTR",
        sense_current_values="0.001", compliance_V=2.0, source_delay_s=0.05, nplc=5,
        auto_range=True, settling_time_s=1.0, n_reversals=5,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        enable_sweep=False,
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        i_min_A=-20.0, i_max_A=20.0, step_A=2.0, bidirectional_sweep=True,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05, field_settle_tolerance_mT=0.02, enable_temperature=False,
        temperature_visa_resource="", temperature_sensor_uids="",
        sample="A",
    )
    base.update(overrides)
    # sense_current_list is normally computed by parse_state() from
    # sense_current_values -- fill it in directly here.
    base["sense_current_list"] = [float(v) for v in str(base["sense_current_values"]).split(",")]
    return base


def test_build_plan_series_tag_only_set_for_a_real_family(tmp_path) -> None:
    app = tui.DCHallMeasurementApp()

    plan_single = app._build_plan(_state())
    assert plan_single.series == ""

    plan_multi = app._build_plan(_state(sense_current_values="0.001, 0.002"))
    assert plan_multi.series != ""
    assert plan_multi.series.startswith("A_HB3_HALL_")


def test_multi_file_session_allocates_one_run_per_sense_current(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.DCHallMeasurementApp()
    plan = app._build_plan(_state(sense_current_values="0.001, -0.002"))

    # Mirror what RunScreen.do_run()'s loop does: allocate_run() fresh per
    # series value, each producing its own run number/file.
    contexts = [
        allocate_run(
            tmp_path, plan.sample, plan.device, tui.MEASUREMENT_TYPE,
            temperature_setpoint_K=plan.temperature_setpoint_K,
            key_axis=("current_A", I), series=plan.series,
        )
        for I in plan.series_values
    ]
    run_numbers = [c.run_number for c in contexts]
    assert run_numbers == [1, 2]
    names = [c.raw_path.name for c in contexts]
    assert names[0].startswith("A_0001_HB3_HALL_T300K_")
    assert any("Im0p002A" in n for n in names)
