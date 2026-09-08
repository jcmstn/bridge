"""
sot/sot_switching_tui.py — build_summary validation, _build_plan purity, and
the one-file-per-assist-magnet-current run-numbering (mirrors
test_dc_spin_valve_tui.py).
"""

from __future__ import annotations

from pathlib import Path

import sot.sot_switching_tui as tui
from instruments.data_naming import allocate_run, ensure_sample


def _state(**overrides) -> dict:
    base = dict(
        k4200_visa_resource="GPIB0::17::INSTR", integration="fast",
        src_channel=1, hall_channel=2,
        compliance_voltage_V=3.0, source_limit_A=0.03, four_wire=True,
        i_min_A=-0.015, i_max_A=0.015, step_A=0.003, bidirectional_sweep=True,
        read_mode="at_write_current", read_current_A=1e-3, read_reversal=False,
        settling_time_s=0.05, n_averages=5, n_repeats=5,
        assist_magnet_currents_A="-3, -1.5, 0, 1.5, 3",
        field_angle_from_oop_deg=85.0, field_settle_tolerance_mT=0.05,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05,
        enable_temperature=False, temperature_visa_resource="", temperature_sensor_uids="",
        sample="A", data_dir="",
    )
    base.update(overrides)
    base["assist_magnet_current_list"] = [float(v) for v in
                                          base["assist_magnet_currents_A"].split(",")]
    base["assist_parse_error"] = None
    return base


def test_summary_flags_same_smu_channel():
    _, _, errors = tui.build_summary(_state(src_channel=2, hall_channel=2))
    assert any("must be different" in e for e in errors)


def test_summary_flags_staircase_bound_over_limit():
    _, _, errors = tui.build_summary(_state(i_max_A=0.05, source_limit_A=0.03))
    assert any("software limit" in e for e in errors)


def test_summary_flags_magnet_current_over_magnet_limit():
    _, _, errors = tui.build_summary(_state(assist_magnet_currents_A="0, 50",
                                            current_limit_A=35.0))
    assert any("magnet limit" in e for e in errors)


def test_summary_warns_single_sign_field_list():
    _, warnings, _ = tui.build_summary(_state(assist_magnet_currents_A="1, 2, 3"))
    assert any("one sign" in w for w in warnings)


def test_summary_warns_read_current_not_small():
    _, warnings, _ = tui.build_summary(_state(read_mode="write_then_read",
                                              read_current_A=0.02,
                                              i_min_A=-0.015, i_max_A=0.015))
    assert any("not ≪" in w for w in warnings)


def test_build_plan_series_tag_only_for_a_real_family(tmp_path: Path):
    app = tui.SOTSwitchingApp()
    app.data_root = tmp_path

    plan_multi = app._build_plan(_state())
    assert plan_multi.series.startswith("A_HB3_SOTSW_")

    plan_single = app._build_plan(_state(assist_magnet_currents_A="0"))
    assert plan_single.series == ""
    assert plan_single.magnet_currents_A == [0.0]


def test_build_plan_staircase_and_smu_roles(tmp_path: Path):
    app = tui.SOTSwitchingApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(i_min_A=-0.015, i_max_A=0.015, step_A=0.003,
                                  bidirectional_sweep=True))
    # 11 up + 10 down
    assert len(plan.currents_A) == 21
    assert plan.total_points == 21 * 5 * 5
    assert plan.src_cfg.channel == 1
    assert plan.hall_cfg.channel == 2
    assert plan.hall_cfg.source_limit_A == 1e-9          # SMU2 is a 0-A voltmeter
    assert plan.field_angle_from_oop_deg == 85.0


def test_one_run_per_assist_magnet_current(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.SOTSwitchingApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(assist_magnet_currents_A="-1.5, 0, 1.5"))

    # Mirror RunScreen.do_run(): allocate_run() fresh per assist value,
    # key axis = magnet current (locked kind "current_A").
    contexts = [
        allocate_run(tmp_path, plan.sample, plan.device, tui.MEASUREMENT_TYPE,
                     temperature_setpoint_K=plan.temperature_setpoint_K,
                     key_axis=("current_A", i_mag), series=plan.series)
        for i_mag in plan.magnet_currents_A
    ]
    assert [c.run_number for c in contexts] == [1, 2, 3]
    names = [c.raw_path.name for c in contexts]
    assert names[0].startswith("A_0001_HB3_SOTSW_T300K_Im1p5A_")
    assert any("_I0A_" in n for n in names)
    assert any("_I1p5A_" in n for n in names)
