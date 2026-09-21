"""Run-time model for the field/gate DC programs: spin valve, Hall, gate sweep.

Pure logic: `run_costs()` is checked against the plan (one cost entry per
loop point), against the audit's floors, and (spin valve) against the real
loop + real Kepco ramp/settle code with `time.sleep` patched out.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import dc.dc_gate_sweep_tui as gate_tui
import dc.dc_hall_measurement_tui as hall_tui
import dc.dc_spin_valve_tui as sv_tui
from dc.dc_spin_valve import AcquisitionConfig, FieldPoint, SourceConfig, run_measurement
from dc.dc_sweep_utils import build_segmented_sweep, field_hops
from instruments import run_time as rt
from instruments.keithley2182 import read_time_s
from instruments.keithley6221 import reversal_avg_s
from instruments.kepco_magnet import KepkoBOPGL, MagnetConfig, magnet_move_s, set_magnet_current
from instruments.lakeshore475 import GaussmeterConfig, read_field_s

SWEEP = [(-20.0, 20.0, 21)]                      # ±20 A, bidirectional -> 41 points


def _sv_state(**kw) -> dict:
    s = dict(
        sense_current_list=[1e-3], enable_gate=False, gate_voltage_list=[],
        ramp_step_A=0.1, ramp_delay_s=0.05, gaussmeter_n_averages=10, gaussmeter_read_delay_s=0.05,
        enable_temperature=False, temperature_sensor_uids="", nplc=5, reversal_enabled=True,
        n_averages=5, source_delay_s=0.05, settling_time_s=1.0,
        magnet_visa_resource="GPIB0::6::INSTR", current_limit_A=35.0, voltage_compliance_V=15.0,
        sweep_rows_parsed=SWEEP, bidirectional_sweep=True,
    )
    s.update(kw)
    return s


def _hall_state(**kw) -> dict:
    s = _sv_state(enable_sweep=True, measure_rxy=True, measure_rxx=False, channel_settle_s=0.02,
                  n_reversals=5)
    s.update(kw)
    return s


def _gate_state(**kw) -> dict:
    s = dict(
        sense_current_list=[1e-6], enable_field=True, field_current_list=[0.0, 1.0],
        ramp_step_A=0.1, ramp_delay_s=0.05, field_settle_s=1.0,
        gaussmeter_n_averages=10, gaussmeter_read_delay_s=0.05,
        enable_temperature=False, temperature_sensor_uids="", nplc=5, n_averages=5,
        settling_time_s=0.2,
    )
    s.update(kw)
    return s


def _currents(state: dict):
    return build_segmented_sweep(state["sweep_rows_parsed"], state["bidirectional_sweep"])


# ── field_hops ───────────────────────────────────────────────────────────────

def test_field_hops_series_major_returns_to_start_between_series():
    assert field_hops([-2.0, 0.0, 2.0], 2) == [2.0, 2.0, 2.0, 4.0, 2.0, 2.0]   # 0->-2, ..., 2->-2 return
    assert field_hops([], 3) == []


# ── spin valve ───────────────────────────────────────────────────────────────

def test_spin_valve_cost_has_one_entry_per_loop_point_and_multiplies_series():
    s = _sv_state(enable_gate=True, gate_voltage_list=[0.0, 5.0], sense_current_list=[1e-3, 2e-3])
    rc = sv_tui.run_costs(_currents(s), s)
    assert len(rc.points) == 41 * 2 * 2
    assert rc.parts["per-file"] == pytest.approx(4 * rt.PER_FILE_S)


def test_spin_valve_plan_carries_cost_matching_total_points(tmp_path: Path):
    s = _sv_state(
        source_visa_resource="GPIB0::20::INSTR", voltmeter_visa_resource="GPIB0::7::INSTR",
        gate_visa_resource="GPIB0::25::INSTR", compliance_V=2.0, auto_range=True,
        field_settle_tolerance_mT=0.02, gate_voltage_limit_V=20.0, gate_compliance_current_A=1e-6,
        gaussmeter_visa_resource="GPIB0::12::INSTR", device="SV2", cooldown="", sample="A",
        temperature_setpoint_K=10.0, temperature_visa_resource="", enable_gate=True,
        gate_voltage_list=[0.0, 5.0, -5.0],
    )
    app = sv_tui.DCSpinValveApp()
    app.data_root = tmp_path
    plan = app._build_plan(s)
    assert plan.run_cost is not None and len(plan.run_cost.points) == plan.total_points == 41 * 3


def test_spin_valve_default_covers_the_audit_floor_not_just_settle_plus_reads():
    s = _sv_state()
    rc = sv_tui.run_costs(_currents(s), s)
    old_estimate = 41 * (1.0 + 5 * 2 * (5 / 50))            # what the sidebar used to say: 82 s
    assert rc.total_s >= 214.0 > 2.5 * old_estimate         # audit: >= 214 s with zero GPIB latency
    # the very first point carries the 0 -> -20 A ramp; later points only a 2 A hop
    assert rc.points[0] > rc.points[1] + 10.0
    assert rc.tail_s > 10.0                                  # 20 A magnet ramp-down at shutdown
    assert rc.worst_extra_s > 30 * 41 * 0.5                  # a settle timeout is ~30 s per point
    assert any("worst case" in line for line in rc.lines())


def test_spin_valve_unidirectional_series_pay_the_return_ramp():
    s = _sv_state(bidirectional_sweep=False, sense_current_list=[1e-3, 2e-3])
    rc = sv_tui.run_costs(_currents(s), s)
    n = 21
    assert rc.points[n] > rc.points[1] + 15.0                # 20 -> -20 A return hop before series 2


def test_spin_valve_reversal_off_uses_plain_reads():
    on = sv_tui.run_costs(_currents(_sv_state()), _sv_state())
    s_off = _sv_state(reversal_enabled=False)
    off = sv_tui.run_costs(_currents(s_off), s_off)
    assert off.parts["2182 reads"] == pytest.approx(41 * 5 * read_time_s(5))
    assert off.parts["2182 reads"] < on.parts["2182 reads"]


def test_spin_valve_summary_line_is_the_cost_model():
    info, _, _ = sv_tui.build_summary({**_sv_state(), "sample": "A", "device": "SV2", "data_dir": "",
                                       "source_visa_resource": "a", "voltmeter_visa_resource": "b",
                                       "gate_visa_resource": "c", "compliance_V": 2.0,
                                       "gate_voltage_limit_V": 20.0, "field_settle_tolerance_mT": 0.02,
                                       "gaussmeter_visa_resource": "g"})
    line = next(i for i in info if i.startswith("Estimated total run time"))
    rc = sv_tui.run_costs(_currents(_sv_state()), _sv_state())
    assert line == rc.lines("Estimated total run time")[0]


# ── modelled total >= every sleep the real loop + real Kepco code performs ───

def test_spin_valve_model_covers_every_sleep_of_the_real_loop(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    class FakePSU:                                   # real ramp_current(), fake I/O
        current = 0.0
        measure_current = property(lambda self: self.current)
        ramp_current = KepkoBOPGL.ramp_current

    psu = FakePSU()
    gm = SimpleNamespace(field=0.0, measure=lambda n, delay=0.0: (0.0, 0.0))
    voltmeter = SimpleNamespace(voltage=1e-3)
    source = SimpleNamespace(source_current=0.0)
    mcfg, gcfg = MagnetConfig(), GaussmeterConfig(unit="T")
    currents = [-2.0, 0.0, 2.0]
    points = [FieldPoint(magnet_current_A=i, set_action=lambda i=i: set_magnet_current(psu, mcfg, i, gm, gcfg))
              for i in currents]
    run_measurement(source, voltmeter, SourceConfig(sense_current_A=1e-3, source_delay_s=0.05),
                    AcquisitionConfig(settling_time_s=1.0, n_averages=5, reversal_enabled=True),
                    points, gaussmeter=gm, gauss_cfg=gcfg, write_csv=lambda records: None)
    assert sum(sleeps) > 3 * 1.0                               # sanity: it did sleep

    s = _sv_state(sweep_rows_parsed=[(-2.0, 2.0, 3)], bidirectional_sweep=False)
    rc = sv_tui.run_costs(currents, s)
    assert rc.total_s - rc.tail_s >= sum(sleeps)


# ── Hall ─────────────────────────────────────────────────────────────────────

def test_hall_cost_points_match_plan_totals_sweep_and_single_point():
    s = _hall_state(sense_current_list=[1e-3, 2e-3])
    assert len(hall_tui.run_costs(_currents(s), s).points) == 41 * 2
    single = hall_tui.run_costs(None, _hall_state(enable_sweep=False, sense_current_list=[1e-3, 2e-3, 3e-3]))
    assert len(single.points) == 3                            # one point per sense current, no magnet
    assert "magnet" not in single.parts and "field read" not in single.parts


def test_hall_default_sweep_covers_the_audit_floor():
    s = _hall_state()
    rc = hall_tui.run_costs(_currents(s), s)
    assert rc.total_s >= 214.0 > 2.5 * (41 * (1.0 + 5 * 2 * (5 / 50)))
    assert rc.points[0] > rc.points[1] + 10.0


def test_hall_two_channels_cost_more_per_point_than_one():
    one = hall_tui.run_costs(None, _hall_state(enable_sweep=False))
    two = hall_tui.run_costs(None, _hall_state(enable_sweep=False, measure_rxx=True))
    expected_two = reversal_avg_s(5, 0.05, read_time_s(5), 2, 0.02)
    assert two.parts["2182 reads"] == pytest.approx(expected_two)
    assert two.parts["2182 reads"] > one.parts["2182 reads"]


def test_hall_plan_carries_cost(tmp_path: Path):
    s = _hall_state(
        source_visa_resource="GPIB0::20::INSTR", voltmeter_visa_resource="GPIB0::7::INSTR",
        compliance_V=2.0, auto_range=True, field_settle_tolerance_mT=0.02,
        gaussmeter_visa_resource="GPIB0::12::INSTR", device="HB3", cooldown="", sample="A",
        temperature_setpoint_K=300.0, field_theta_deg=None, field_phi_deg=None,
        temperature_visa_resource="",
    )
    app = hall_tui.DCHallMeasurementApp()
    app.data_root = tmp_path
    plan = app._build_plan(s)
    assert plan.run_cost is not None and len(plan.run_cost.points) == plan.total_points == 41
    plan1 = app._build_plan({**s, "enable_sweep": False})
    assert len(plan1.run_cost.points) == plan1.total_points == 1


# ── gate sweep ───────────────────────────────────────────────────────────────

def test_gate_sweep_parks_only_when_the_field_changes_and_counts_the_ramp():
    s = _gate_state(sense_current_list=[1e-6, 2e-6])          # field outer, sense inner: 2 parks, 4 files
    n = 81
    rc = gate_tui.run_costs(n, s)
    assert len(rc.points) == n * 4
    assert rc.parts["per-file"] == pytest.approx(4 * rt.PER_FILE_S)
    first = [rc.points[k * n] for k in range(4)]                # cost of each series' first point
    assert first[1] == pytest.approx(first[3])                  # same field as the previous series: no park
    park_1A = magnet_move_s(1.0, MagnetConfig())[0] + 1.0 + read_field_s(GaussmeterConfig())
    assert first[2] - first[1] == pytest.approx(park_1A)        # series 2 parks 0 A -> 1 A (ramp+settle+dwell+read)
    assert first[0] > first[1]                                  # series 0: park + run start
    assert rc.parts["field dwell"] == pytest.approx(2 * 1.0)


def test_gate_sweep_default_is_more_than_the_old_dwell_only_estimate():
    s = _gate_state()
    n = 81
    old = n * 2 * (0.2 + 5 * (5 / 50)) + 2 * 1.0              # old: points x (settle + reads) + dwell per park
    rc = gate_tui.run_costs(n, s)
    typ, _ = magnet_move_s(0.0, MagnetConfig())
    assert rc.total_s > old + typ + read_field_s(GaussmeterConfig())
    assert rc.tail_s >= rt.GATE_RAMP_S                        # gate ramp-down at shutdown


def test_gate_sweep_without_field_never_touches_the_magnet():
    s = _gate_state(enable_field=False, sense_current_list=[1e-6, 2e-6])
    rc = gate_tui.run_costs(41, s)
    assert len(rc.points) == 82
    assert not {"magnet", "field dwell", "field read"} & set(rc.parts)


def test_gate_sweep_plan_carries_cost(tmp_path: Path):
    s = _gate_state(
        source_visa_resource="GPIB0::20::INSTR", voltmeter_visa_resource="GPIB0::7::INSTR",
        gate_visa_resource="GPIB0::25::INSTR", compliance_V=2.0, source_delay_s=0.05, auto_range=True,
        gate_voltage_limit_V=20.0, gate_compliance_current_A=1e-6, gate_min_V=-10.0, gate_max_V=10.0,
        step_V=0.5, bidirectional_sweep=True, magnet_visa_resource="GPIB0::6::INSTR",
        current_limit_A=35.0, voltage_compliance_V=15.0, gaussmeter_visa_resource="GPIB0::12::INSTR",
        field_settle_tolerance_mT=0.02, device="HB3", cooldown="", sample="A",
        temperature_setpoint_K=300.0, temperature_visa_resource="",
    )
    app = gate_tui.DCGateSweepApp()
    app.data_root = tmp_path
    plan = app._build_plan(s)
    assert plan.run_cost is not None and len(plan.run_cost.points) == plan.total_points == 81 * 2
