"""SOT programs' run-time model (`run_costs`): point count matches the plan, per-point
and per-file terms come from the instrument helpers, and the audited estimate bugs
stay fixed. Pure logic, no hardware; the expected numbers are built from the same
helpers/knobs, so tuning a knob in instruments/run_time.py never breaks these."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import sot.sot_nonlocal_switching as ns
import sot.sot_nonlocal_switching_tui as nlsw
import sot.sot_pulsed_switching as ps
import sot.sot_pulsed_switching_2h_tui as h2
import sot.sot_pulsed_switching_6221_tui as p6221
import sot.sot_pulsed_switching_tui as pulsed
from instruments import run_time as rt
from instruments.keithley2182 import read_time_s
from instruments.keithley4200a import pulse_once_s
from instruments.keithley6221 import ac_source_restart_s, reversal_avg_s, wave_pulse_s
from instruments.kepco_magnet import MagnetConfig, magnet_move_s
from instruments.lakeshore475 import GaussmeterConfig, read_field_s
from instruments.mfli_daq import acquire_s
from test_sot_nonlocal_switching import _FakeVoltmeter, _Fake6221 as _FakeNLSW6221
from test_sot_nonlocal_switching_tui import _state as nlsw_state
from test_sot_pulsed_switching_2h_tui import _state as h2_state
from test_sot_pulsed_switching_6221_tui import _state as p6221_state
from test_sot_pulsed_switching_tui import _state as pulsed_state
from test_sot_run_loops import _Fake2182, _Fake6221, _FakeKXCI

_MAG = MagnetConfig(ramp_step_A=0.1, ramp_delay_s=0.05)          # the test states' values
_GAUSS = GaussmeterConfig(n_averages=10, read_delay_s=0.05)
_NULL_WRITER = lambda recs: None


def _plans(tmp_path: Path) -> dict:
    """One multi-file plan per program (4 files each, 2 for NLSW)."""
    def app(cls):
        a = cls()
        a.data_root = tmp_path
        return a
    kw = dict(magnet_current_A="1.5, -1.5", sense_current_values="1e-4, 2e-4")
    return {
        "pulsed": app(pulsed.SOTPulsedSwitchingApp)._build_plan(pulsed_state(**kw)),
        "2h": app(h2.SOTPulsedSwitching2HApp)._build_plan(h2_state(**kw)),
        "6221": app(p6221.SOTPulsedSwitching6221App)._build_plan(p6221_state(**kw)),
        "nlsw": nlsw.build_plan(nlsw_state(tmp_path, init_magnet_currents="5, -5"), tmp_path),
    }


def test_every_plan_carries_a_cost_per_point_and_the_bar_ends_at_the_last_point(tmp_path):
    for name, plan in _plans(tmp_path).items():
        rc = plan.run_cost
        assert rc is not None and len(rc.points) == plan.total_points > 0, name
        assert rt.progress_total(rc, plan.total_points) == pytest.approx(sum(rc.points)), name
        assert sum(rt.progress_step(rc, i) for i in range(plan.total_points)) == \
            pytest.approx(rt.progress_total(rc, plan.total_points)), name
        assert rc.total_s == pytest.approx(sum(rc.points) + rc.tail_s) and rc.tail_s > 0, name


# ── sot_pulsed_switching (4200A) ────────────────────────────────────────────

def test_pulsed_point_is_wait_plus_pmu_plus_read_with_delay_and_read_added():
    state = pulsed_state()
    rc = pulsed.run_costs(state)
    read = reversal_avg_s(5, 0.05, read_time_s(5.0))
    expect = 1.0 + 0.3 + pulse_once_s(1, 1e-3) + read + 5 * rt.GPIB_TXN_S + rt.POINT_OVERHEAD_S
    assert rc.points[1] == pytest.approx(expect)
    assert read > 5 * 2 * max(0.05, 5.0 / 50)              # the old max() model undercounted the read
    typ, worst = magnet_move_s(1.5, _MAG)                  # first point: field, per-file, per-run one-offs
    assert rc.points[0] == pytest.approx(expect + typ + read_field_s(_GAUSS) + rt.PER_FILE_S + rt.PER_RUN_S)
    assert rc.tail_s == pytest.approx(magnet_move_s(1.5, _MAG, with_field=False)[0])
    assert rc.worst_extra_s == pytest.approx(worst - typ)
    with_temp = pulsed.run_costs(pulsed_state(enable_temperature=True, temperature_sensor_uids="MB1.T1"))
    assert with_temp.points[1] - rc.points[1] == pytest.approx(rt.TEMP_READ_S)


def test_pulsed_magnet_parking_guard_matches_the_loop():
    same = pulsed.run_costs(pulsed_state(sense_current_values="1e-4, 2e-4"))      # 2 files, one field
    assert same.parts["magnet"] == pytest.approx(magnet_move_s(1.5, _MAG)[0])     # 2nd file: no move
    assert same.parts["per-file"] == pytest.approx(2 * rt.PER_FILE_S)
    assert same.parts["field read"] == pytest.approx(2 * read_field_s(_GAUSS))
    flip = pulsed.run_costs(pulsed_state(magnet_current_A="1.5, -1.5"))
    assert flip.parts["magnet"] == pytest.approx(magnet_move_s(1.5, _MAG)[0] + magnet_move_s(3.0, _MAG)[0])


def test_summary_line_is_the_run_costs_total(tmp_path):
    state = pulsed_state(data_dir=str(tmp_path))
    info, _, _ = pulsed.build_summary(state)
    line = next(i for i in info if i.startswith("Estimated run time"))
    assert line.startswith(f"Estimated run time ≈ {rt.format_duration(pulsed.run_costs(state).total_s)}")


# ── sot_pulsed_switching_2h ─────────────────────────────────────────────────

def test_2h_uses_typical_lock_two_sequential_windows_and_rebuilds_the_6221_every_file():
    state = h2_state()
    rc = h2.run_costs(state)
    n = len(state["amplitude_list"])
    window = acquire_s(0.3, 50, 857.0)
    expect = (1.0 + pulse_once_s(1, 1e-3) + rt.ARM_S + rt.LOCK_TYP_S + 1.0 + 2 * window
              + 8 * rt.GPIB_TXN_S + rt.POINT_OVERHEAD_S)
    assert rc.points[1] == pytest.approx(expect)
    assert rc.parts["PLL lock"] == pytest.approx(n * rt.LOCK_TYP_S)             # not the 5 s timeout
    typ, worst = magnet_move_s(1.5, _MAG)
    assert rc.worst_extra_s == pytest.approx(n * (5.0 - rt.LOCK_TYP_S) + (worst - typ))
    assert rc.points[0] == pytest.approx(expect + ac_source_restart_s() + typ + read_field_s(_GAUSS)
                                         + rt.PER_FILE_S + rt.PER_RUN_S)


def test_2h_lock_timeout_below_typical_bounds_the_wait_and_adds_no_worst_case():
    state = h2_state(lock_timeout_s=0.2)
    rc = h2.run_costs(state)
    assert rc.parts["PLL lock"] == pytest.approx(len(state["amplitude_list"]) * 0.2)
    typ, worst = magnet_move_s(1.5, _MAG)
    assert rc.worst_extra_s == pytest.approx(worst - typ)


def test_2h_sense_outer_magnet_inner_and_no_parking_guard():
    rc = h2.run_costs(h2_state(sense_current_values="1e-4, 2e-4"))              # 2 files, same field
    assert rc.parts["6221 rebuild"] == pytest.approx(2 * ac_source_restart_s())
    # set_magnet_current() runs every file: the 2nd is a zero move (1 ramp step + settle floor)
    assert rc.parts["magnet"] == pytest.approx(magnet_move_s(1.5, _MAG)[0] + magnet_move_s(0.0, _MAG)[0])


# ── sot_pulsed_switching_6221 ───────────────────────────────────────────────

def test_6221_only_arms_twice_per_point_and_reads_one_demod():
    state = p6221_state()
    rc = p6221.run_costs(state)
    n = len(state["pulse_current_list"])
    expect = (1.0 + wave_pulse_s(1e-3) + rt.ARM_S + rt.LOCK_TYP_S + 1.0 + acquire_s(0.3, 50, 857.0)
              + 9 * rt.GPIB_TXN_S + rt.POINT_OVERHEAD_S)
    assert rc.points[1] == pytest.approx(expect)
    assert wave_pulse_s(1e-3) >= rt.ARM_S + 2e-3                                # fire_wave_pulse arms too
    assert rc.parts["6221 re-arm"] == pytest.approx(n * rt.ARM_S)
    assert rc.parts["PLL lock"] == pytest.approx(n * rt.LOCK_TYP_S)


def test_6221_only_file_order_is_sense_outer_magnet_inner():
    rc = p6221.run_costs(p6221_state(magnet_current_A="1.5, -1.5", sense_current_values="1e-4, 2e-4"))
    assert rc.parts["6221 rebuild"] == pytest.approx(4 * ac_source_restart_s())
    # 0 -> 1.5, 1.5 -> -1.5, -1.5 -> 1.5, 1.5 -> -1.5
    assert rc.parts["magnet"] == pytest.approx(magnet_move_s(1.5, _MAG)[0] + 3 * magnet_move_s(3.0, _MAG)[0])


# ── sot_nonlocal_switching ──────────────────────────────────────────────────

def test_nlsw_two_magnet_moves_per_init_current_and_zero_amp_points_fire_nothing(tmp_path):
    state = nlsw_state(tmp_path, init_magnet_currents="5, -5", pulse_current_start_A=-2e-3,
                       pulse_current_stop_A=2e-3, pulse_current_step_A=2e-3)      # -2 mA, 0, +2 mA
    rc = nlsw.run_costs(state)
    per_file = 3 + 1
    assert len(rc.points) == 2 * per_file
    t5 = magnet_move_s(5.0, _MAG)[0]
    assert rc.parts["magnet"] == pytest.approx(4 * t5)                            # init + hold, twice
    assert rc.parts["field read"] == pytest.approx(4 * read_field_s(_GAUSS))
    assert rc.worst_extra_s == pytest.approx(4 * (magnet_move_s(5.0, _MAG)[1] - t5))
    fired, zero = rc.points[3], rc.points[2]                                      # +2 mA, 0 A (file 0)
    assert fired - zero == pytest.approx(wave_pulse_s(1e-3) + state["delay_after_pulse_s"])
    assert rc.tail_s == pytest.approx(magnet_move_s(0.0, _MAG, with_field=False)[0])


def test_nlsw_no_init_leaves_the_magnet_out_of_the_estimate(tmp_path):
    state = nlsw_state(tmp_path)
    rc = nlsw.run_costs(state)
    assert "magnet" not in rc.parts and rc.tail_s == 0.0 and rc.worst_extra_s == 0.0
    assert rc.points[0] == pytest.approx(rc.points[-1] - state["delay_after_pulse_s"]
                                         - wave_pulse_s(1e-3) + rt.PER_FILE_S + rt.PER_RUN_S)
    info, _, _ = nlsw.build_summary(state)
    assert not any("excludes magnet ramps" in i for i in info)


def test_nlsw_reversal_read_costs_the_sum_of_delay_and_reads(tmp_path):
    rev = nlsw.run_costs(nlsw_state(tmp_path, reversal_enabled=True))
    plain = nlsw.run_costs(nlsw_state(tmp_path, reversal_enabled=False))
    n = len(rev.points)
    one = read_time_s(5.0)
    assert rev.parts["reads"] == pytest.approx(n * reversal_avg_s(5, 0.1, one))
    assert plain.parts["reads"] == pytest.approx(n * (rt.GPIB_TXN_S + 0.1 + 5 * one))
    assert rev.total_s > plain.total_s


# ── the model must cover every sleep the real loop performs ─────────────────

@pytest.fixture
def virtual_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    return clock


def test_pulsed_model_covers_the_loops_sleeps(virtual_clock, tmp_path):
    app = pulsed.SOTPulsedSwitchingApp()
    app.data_root = tmp_path
    state = pulsed_state()
    plan = app._build_plan(state)
    points = [ps.AmplitudePoint(amplitude_V=float(v)) for v in plan.amplitudes_V]
    ps.run_measurement(_FakeKXCI(), plan.pmu_cfg, _Fake6221(), _Fake2182(), plan.read_cfg, points,
                       write_csv=_NULL_WRITER)
    assert virtual_clock[0] > 0
    assert virtual_clock[0] <= plan.run_cost.total_s - rt.PER_RUN_S - rt.PER_FILE_S - plan.run_cost.tail_s


def test_nlsw_model_covers_the_loops_sleeps(virtual_clock, tmp_path):
    plan = nlsw.build_plan(nlsw_state(tmp_path), tmp_path)
    src = _FakeNLSW6221()
    ns.run_measurement(src, _FakeVoltmeter(src), plan.pulse_cfg, plan.read_cfg,
                       [ns.PulsePoint(float(v)) for v in plan.pulse_currents_A], write_csv=_NULL_WRITER)
    assert virtual_clock[0] > 0
    assert virtual_clock[0] <= plan.run_cost.total_s - rt.PER_RUN_S - rt.PER_FILE_S - plan.run_cost.tail_s
