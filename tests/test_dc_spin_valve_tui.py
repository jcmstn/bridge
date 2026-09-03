"""
Plan-purity test for dc/dc_spin_valve_tui.py's _build_plan() and the
multi-file-per-session (one file per sense-current x gate-voltage
combination) run-numbering logic.
"""

from __future__ import annotations

from pathlib import Path

import dc.dc_spin_valve_tui as tui
from instruments.data_naming import allocate_run, ensure_sample


def _state(**overrides) -> dict:
    base = dict(
        source_visa_resource="GPIB0::20::INSTR",
        voltmeter_visa_resource="GPIB0::7::INSTR",
        gate_visa_resource="GPIB0::25::INSTR",
        sense_current_values="0.001", compliance_V=2.0, source_delay_s=0.05, nplc=5,
        auto_range=True, settling_time_s=1.0, n_averages=5,
        device="SV2", cooldown="3", temperature_setpoint_K=10.0,
        enable_gate=True, gate_voltage_limit_V=20.0, gate_compliance_current_A=1e-6,
        gate_voltage_values="0, 5, -5",
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        i_min_A=-20.0, i_max_A=20.0, step_A=2.0, bidirectional_sweep=True,
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05, field_settle_tolerance_mT=0.02, enable_temperature=False,
        temperature_visa_resource="", temperature_sensor_uids="",
        reversal_enabled=True, sample="A",
    )
    base.update(overrides)
    state, _ = base, None
    # gate_voltage_list / sense_current_list are normally computed by
    # parse_state() -- fill them in directly here.
    state["gate_voltage_list"] = [float(v) for v in state["gate_voltage_values"].split(",")]
    state["sense_current_list"] = [float(v) for v in str(state["sense_current_values"]).split(",")]
    return state


def test_build_plan_series_tag_only_set_for_a_real_family(tmp_path: Path) -> None:
    app = tui.DCSpinValveApp()
    app.data_root = tmp_path

    plan_multi = app._build_plan(_state())
    assert plan_multi.series != ""
    assert plan_multi.series.startswith("A_SV2_BSWP_")

    plan_single = app._build_plan(_state(enable_gate=False, gate_voltage_list=[]))
    assert plan_single.series == ""

    plan_current_multi = app._build_plan(_state(
        enable_gate=False, gate_voltage_list=[],
        sense_current_values="0.001, 0.002",
        sense_current_list=[0.001, 0.002],
    ))
    assert plan_current_multi.series != ""


def test_multi_file_session_allocates_one_run_per_gate_voltage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.DCSpinValveApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state())

    # Mirror what RunScreen.do_run()'s loop does: allocate_run() fresh per
    # series value, each producing its own run number/file.
    contexts = [
        allocate_run(
            tmp_path, plan.sample, plan.device, tui.MEASUREMENT_TYPE,
            temperature_setpoint_K=plan.temperature_setpoint_K,
            key_axis=("gate_V", gv), series=plan.series,
        )
        for _, gv in plan.series_values
    ]
    run_numbers = [c.run_number for c in contexts]
    assert run_numbers == [1, 2, 3]
    names = [c.raw_path.name for c in contexts]
    assert names[0].startswith("A_0001_SV2_BSWP_T010K_Vg0V_")
    assert any("Vgm5V" in n for n in names)


def test_multi_file_session_cross_product_of_current_and_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tui.DCSpinValveApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(
        sense_current_values="0.001, 0.002",
        sense_current_list=[0.001, 0.002],
        gate_voltage_values="0, 5",
        gate_voltage_list=[0.0, 5.0],
    ))

    # 2 sense currents x 2 gate voltages -> 4 runs, gate wins the key_axis
    # since it's the axis with more than one value alongside current too --
    # both vary here, so gate_V is picked per the do_run()/build_plan()
    # convention (see dc_spin_valve_tui.py's do_run()).
    assert len(plan.series_values) == 4
    contexts = [
        allocate_run(
            tmp_path, plan.sample, plan.device, tui.MEASUREMENT_TYPE,
            temperature_setpoint_K=plan.temperature_setpoint_K,
            key_axis=("gate_V", gv), series=plan.series,
        )
        for _, gv in plan.series_values
    ]
    assert [c.run_number for c in contexts] == [1, 2, 3, 4]
