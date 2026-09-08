"""
sot/sot_dc_characterization_tui.py — build_summary validation + _build_plan
purity. Pure logic only, no Textual app mount, no hardware.
"""

from __future__ import annotations

from pathlib import Path

import sot.sot_dc_characterization_tui as tui


def _state(**overrides) -> dict:
    base = dict(
        k4200_visa_resource="GPIB0::17::INSTR", integration="normal",
        src_channel=1, sense_channel=2,
        compliance_voltage_V=2.0, source_limit_A=5e-3, four_wire=True,
        current_min_A=-1e-4, current_max_A=1e-4, step_A=1e-4,
        bidirectional_sweep=True, reversal_enabled=True,
        settling_time_s=0.1, n_averages=5, n_repeats=10,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        enable_temperature=False, temperature_visa_resource="", temperature_sensor_uids="",
        sample="A", data_dir="",
    )
    base.update(overrides)
    return base


def test_summary_flags_same_smu_channel():
    _, _, errors = tui.build_summary(_state(src_channel=1, sense_channel=1))
    assert any("different channels" in e for e in errors)


def test_summary_flags_bound_over_software_limit():
    _, _, errors = tui.build_summary(_state(current_max_A=1e-2, source_limit_A=5e-3))
    assert any("software limit" in e for e in errors)


def test_summary_warns_reversal_off():
    _, warnings, _ = tui.build_summary(_state(reversal_enabled=False))
    assert any("reversal is OFF" in w for w in warnings)


def test_summary_fixed_current_is_baseline_not_error():
    info, _, errors = tui.build_summary(_state(current_min_A=1e-4, current_max_A=1e-4))
    assert errors == [] or all("min must be ≤ max" not in e for e in errors)
    assert any("baseline R_xx" in i for i in info)


def test_build_plan_shapes(tmp_path: Path):
    app = tui.SOTDCCharApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(current_min_A=-1e-4, current_max_A=1e-4, step_A=5e-5,
                                  bidirectional_sweep=True))
    # 5 up + 4 down (turnaround not duplicated)
    assert len(plan.currents_A) == 9
    assert plan.total_points == 9 * 10
    assert plan.src_cfg.channel == 1 and plan.sense_cfg.channel == 2
    assert plan.src_cfg.source_function == "current"
    assert plan.sense_cfg.source_limit_A == 1e-9          # SMU2 is a 0-A voltmeter
    assert plan.acq_cfg.reversal_enabled is True
    assert plan.series == ""                              # single-file measurement


def test_build_plan_temperature_cfg_only_when_enabled_and_uid_set(tmp_path: Path):
    app = tui.SOTDCCharApp()
    app.data_root = tmp_path
    assert app._build_plan(_state(enable_temperature=False)).temp_cfg is None
    assert app._build_plan(_state(enable_temperature=True, temperature_sensor_uids="")).temp_cfg is None
    p = app._build_plan(_state(enable_temperature=True,
                               temperature_visa_resource="TCPIP0::x::7020::SOCKET",
                               temperature_sensor_uids="MB1.T1"))
    assert p.temp_cfg is not None and p.temp_cfg.sensor_uids == ("MB1.T1",)
