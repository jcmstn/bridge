"""
Plan-purity test for dc/dc_spin_valve_tui.py's _build_plan() and the
multi-file-per-session (one file per sense-current x gate-voltage
combination) run-numbering logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
        sweep_rows_parsed=[(-20.0, 20.0, 21)], bidirectional_sweep=True,
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


def test_build_plan_multi_row_sweep(tmp_path: Path) -> None:
    app = tui.DCSpinValveApp()
    app.data_root = tmp_path

    plan = app._build_plan(_state(
        sweep_rows_parsed=[(-1.0, 1.0, 10), (1.0, 10.0, 10)],
        bidirectional_sweep=True,
    ))
    assert len(plan.currents_A) == 37
    assert plan.header_extra["field_sweep_rows_A"] == [(-1.0, 1.0, 10), (1.0, 10.0, 10)]


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


class _FakeSMU:
    """2450 stand-in: sample of R_2w = 150 Ohm with a +0.2 mV offset on its
    own reading; the 2182 sees R = 100 Ohm with a +1 mV thermal offset."""
    def __init__(self, voltage_mode: bool) -> None:
        self.voltage_mode = voltage_mode
        self.source_current = self.source_voltage = 0.0

    @property
    def i(self) -> float:
        return self.source_voltage / 150.0 if self.voltage_mode else self.source_current

    @property
    def voltage(self) -> float:
        return 150.0 * self.i + 2e-4

    @property
    def current(self) -> float:
        return self.i + 1e-7


class _FakeVoltmeter:
    def __init__(self, smu: _FakeSMU) -> None:
        self.smu = smu

    @property
    def voltage(self) -> float:
        return 100.0 * self.smu.i + 1e-3


def test_acquire_smu_reversal_current_and_voltage_mode() -> None:
    from dc.dc_spin_valve import acquire_smu_reversal, SMUConfig

    for voltage_mode, level in [(False, 1e-3), (True, 0.15)]:
        smu = _FakeSMU(voltage_mode)
        cfg = SMUConfig(source_function="voltage" if voltage_mode else "current",
                        source_limit_A=1e-3)
        vm = _FakeVoltmeter(smu)

        on = acquire_smu_reversal(smu, cfg, vm, level, n=3, reversal=True)
        assert on["mean"] == pytest.approx(0.1)          # odd: 100 Ohm x 1 mA, offset cancelled
        assert on["even_mean"] == pytest.approx(1e-3)    # even: the thermal offset
        assert on["smu_mean"] == pytest.approx(0.15 if not voltage_mode else 1e-3)
        assert on["n_reversals"] == 3
        assert (smu.source_voltage if voltage_mode else smu.source_current) == level

        off = acquire_smu_reversal(smu, cfg, vm, level, n=2, reversal=False)
        assert off["mean"] == pytest.approx(0.101) and off["even_mean"] is None


def test_build_plan_2450_modes(tmp_path: Path) -> None:
    app = tui.DCSpinValveApp()
    app.data_root = tmp_path
    smu = dict(use_2450=True, smu_visa_resource="GPIB0::18::INSTR", plot_2wire=False,
               smu_compliance_current_A=1e-3, smu_voltage_limit_V=21.0, enable_gate=False,
               gate_voltage_list=[])

    assert app._build_plan(_state()).smu_cfg is None
    assert app._build_plan(_state()).header_extra["source_instrument"] == "6221"

    cur = app._build_plan(_state(**smu, smu_source_voltage=False))
    assert cur.smu_cfg.source_function == "current"
    assert cur.smu_cfg.source_limit_A == pytest.approx(1e-3)
    assert cur.show_2wire is False

    volt = app._build_plan(_state(**smu, smu_source_voltage=True,
                                  source_voltage_list=[0.1, 0.2]))
    assert volt.smu_cfg.source_function == "voltage"
    assert volt.sense_currents_A == [0.1, 0.2]
    assert volt.header_extra["source_instrument"] == "2450"
    assert volt.series != ""


def test_magnet_model_and_gaussmeter_off(tmp_path: Path) -> None:
    from instruments.kepco_magnet import KepcoBOP3628MG

    app = tui.DCSpinValveApp()
    app.data_root = tmp_path

    plan = app._build_plan(_state())
    assert plan.magnet_cfg.model == "BOP 20-50GL" and plan.gauss_cfg is not None

    plan = app._build_plan(_state(magnet_model="BOP 36-28MG", enable_gaussmeter=False))
    assert plan.magnet_cfg.model == "BOP 36-28MG"
    assert plan.gauss_cfg is None
    assert plan.header_extra["gaussmeter_enabled"] is False
    assert KepcoBOP3628MG.current_values == [-28.0, 28.0]

    # the 20-50GL default limit (35 A) is above the 36-28MG rating -> blocks Start
    state = _state(magnet_model="BOP 36-28MG", enable_gaussmeter=False, data_dir=str(tmp_path),
                   sweep_rows="-20, 20, 21", gate_voltage_values="0", sense_current_values="0.001",
                   source_voltage_values="0.1", temperature_setpoint_K=None)
    state = tui.resolve_state(state)
    errors = tui.build_summary(state)[2]
    assert any("rating (28 A)" in e for e in errors)
    info = tui.build_summary({**state, "current_limit_A": 25.0})[0]
    assert any("Field read: off" in line for line in info)
    assert not any("field read" in line for line in info if line.startswith("Run time"))


def test_png_without_gaussmeter_plots_vs_magnet_current(tmp_path: Path) -> None:
    from types import SimpleNamespace
    recs = [dict(point_index=i, magnet_current_A=float(i), magnet_field_mT=None,
                 voltage_V=0.1, sense_current_A=1e-3) for i in range(3)]
    png = tmp_path / "sv.png"
    tui._save_measurement_png(recs, png, plan=SimpleNamespace(smu_cfg=None, show_2wire=False))
    assert png.exists()
