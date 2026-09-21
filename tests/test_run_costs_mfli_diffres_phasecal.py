"""Run-time model for mfli_diff_resistance and mfli_phase_calibration: the estimate is the
sum of what the loop does (per-point cost list = progress-bar weights), not settle + one window."""

from __future__ import annotations

import pytest

import mfli.mfli_diff_resistance_tui as dr
import mfli.mfli_phase_calibration_tui as pc
from instruments import run_time as rt
from instruments.data_naming import ensure_sample
from instruments.kepco_magnet import MagnetConfig, magnet_move_s
from instruments.mfli_daq import acquire_s, poll_window_s
from mfli.mfli_phase_calibration import null_follower_s, null_phase_s


def _dr_state(**overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886", daq_host="localhost", daq_port=8004,
        frequency_Hz=137.0, ac_amplitude_V=0.005, series_R_ohm=100000.0,
        bias_min_V=-0.5, bias_max_V=0.5, time_constant_s=0.3, order=4, sinc_filter=True,
        current_input_range_A=1e-6, voltage_input_range_V=0.1, sample_rate_Hz=857.0,
        settling_time_s=1.5, n_averages=50, device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        n_points=41, enable_temperature=False, temperature_visa_resource="",
        temperature_sensor_uids="", sample="A",
    )
    base.update(overrides)
    return base


def _pc_state(**overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886", daq_host="localhost", daq_port=8004,
        frequency_Hz=17.777, amplitude_V=0.1, series_R_ohm=10000.0, time_constant_s=0.3, order=4,
        sinc_filter=True, input_range_1f_V=1.0, input_range_2f_V=1.0, sample_rate_Hz=857.0,
        visa_resource="GPIB0::6::INSTR", current_limit_A=35.0, voltage_compliance_V=15.0,
        ramp_step_A=0.1, ramp_delay_s=0.05, gaussmeter_visa_resource="GPIB0::12::INSTR",
        gaussmeter_n_averages=10, gaussmeter_read_delay_s=0.05, field_settle_tolerance_mT=0.02,
        calibration_current_A=20.0, sweep_rows_parsed=[(-20.0, 20.0, 11)],
        sweep_settling_time_s=1.5, sweep_n_averages=20, hold_tol_ratio=0.02,
        null_n_averages=20, null_max_iterations=5, null_tol_deg=0.02,
        enable_amplitude_check=False, amplitudes_V=[], amp_n_averages=20,
        enable_frequency_check=False, frequencies_Hz=[], freq_n_averages=20,
        freq_max_iterations=5, freq_tol_deg=0.02, device="HB3", cooldown="3",
        temperature_setpoint_K=300.0, enable_temperature=False, temperature_visa_resource="",
        temperature_sensor_uids="", sample="A",
    )
    base.update(overrides)
    return base


# ── diff_resistance ──────────────────────────────────────────────────────────

def test_diffres_two_windows_with_3tc_floor_about_twice_the_old_estimate() -> None:
    state = _dr_state()
    n = 2 * state["n_points"] - 1                                    # bidirectional, turn-around not repeated
    rc = dr.run_costs(n, state)
    old_estimate = n * (1.5 + 2 * max(0.1, 50 * 1.5 / 857.0))        # settle + 2 windows WITHOUT the 3·TC floor
    assert old_estimate == pytest.approx(137.7, abs=0.1)
    assert rc.total_s >= 2 * old_estimate
    # one steady-state point = settle + two SEQUENTIAL acquisitions (3·TC = 0.9 s window each) + overhead
    assert rc.points[1] == pytest.approx(
        1.5 + 2 * acquire_s(0.3, 50, 857.0) + 3 * rt.GPIB_TXN_S + rt.POINT_OVERHEAD_S)
    assert acquire_s(0.3, 50, 857.0) == pytest.approx(poll_window_s(0.3, 50, 857.0) + rt.ACQ_OVERHEAD_S)


def test_diffres_slow_filter_dominates_the_point() -> None:
    fast, slow = dr.run_costs(2, _dr_state()), dr.run_costs(2, _dr_state(time_constant_s=1.0))
    assert slow.points[1] - fast.points[1] == pytest.approx(2 * (3.0 - 0.9))   # 3·TC per window, two windows


def test_diffres_startup_on_first_point_teardown_off_the_bar() -> None:
    rc = dr.run_costs(81, _dr_state())
    assert rc.points[0] - rc.points[1] == pytest.approx(rt.PER_RUN_S + rt.MDS_SYNC_S)
    assert rc.tail_s > rt.PER_FILE_S                                # bias ramp-down + PNG/index after the last point
    assert rt.progress_total(rc, 81) == pytest.approx(sum(rc.points)) == pytest.approx(rc.total_s - rc.tail_s)


def test_diffres_temperature_read_charged_per_point() -> None:
    off = dr.run_costs(3, _dr_state())
    on = dr.run_costs(3, _dr_state(enable_temperature=True, temperature_sensor_uids="MB1.T1"))
    assert on.points[1] - off.points[1] == pytest.approx(rt.TEMP_READ_S)


def test_diffres_plan_and_sidebar_carry_the_cost(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dr, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = dr.MFLIDiffResistanceApp()
    app.data_root = tmp_path
    plan = app._build_plan(_dr_state())
    assert plan.total_points == 81 == len(plan.run_cost.points)
    info, _, _ = dr.build_summary(_dr_state(data_dir=str(tmp_path)))
    line = next(i for i in info if i.startswith("Estimated total run time"))
    assert dr.run_costs(81, _dr_state()).lines("Estimated total run time")[0] == line


# ── phase_calibration ────────────────────────────────────────────────────────

def test_phasecal_far_above_the_old_34s_and_counts_two_windows() -> None:
    state = _pc_state()
    rc = pc.run_costs(state)
    assert len(rc.points) == 21                                       # (-20, 20, 11) bidirectional
    old_estimate = 21 * (1.5 + max(0.1, 20 * 1.5 / 857.0))            # one window, no 3·TC floor: ~34 s
    assert old_estimate == pytest.approx(33.6, abs=0.1)
    assert rc.total_s >= 185                                          # the audit's floor (no GPIB latency)
    assert rc.parts["acquire"] == pytest.approx(21 * 2 * acquire_s(0.3, 20, 857.0))


def test_phasecal_hops_start_from_the_calibration_point_and_teardown_is_off_the_bar() -> None:
    state = _pc_state()
    rc = pc.run_costs(state)
    cfg = MagnetConfig(ramp_step_A=0.1, ramp_delay_s=0.05)
    assert rc.points[1] == pytest.approx(rc.points[2])                # steady state: 4 A hops
    # point 0 hops +20 A -> -20 A (40 A) and carries everything that precedes the sweep
    first_hop_extra = magnet_move_s(40.0, cfg)[0] - magnet_move_s(4.0, cfg)[0]
    pre = rt.PER_RUN_S + rt.MDS_SYNC_S + magnet_move_s(20.0, cfg)[0] + 1.5
    n1, _ = null_phase_s(0.3, 857.0, 20, 5)
    n2, _ = null_follower_s(0.3, 857.0, 20, 5)
    assert rc.points[0] - rc.points[1] == pytest.approx(first_hop_extra + pre + n1 + n2)
    assert rc.tail_s >= magnet_move_s(20.0, cfg, with_field=False)[0] + rt.PER_FILE_S
    assert rt.progress_total(rc, 21) == pytest.approx(sum(rc.points)) == pytest.approx(rc.total_s - rc.tail_s)


def test_phasecal_optional_checks_only_lengthen_the_tail() -> None:
    base = pc.run_costs(_pc_state())
    both = pc.run_costs(_pc_state(enable_amplitude_check=True, amplitudes_V=[0.05, 0.1],
                                  enable_frequency_check=True, frequencies_Hz=[263.3, 317.3, 383.3]))
    assert both.points == base.points                                 # bar weights untouched
    assert both.tail_s > base.tail_s + 3 * 1.5                        # >= a settle per extra step
    assert both.total_s > base.total_s


def test_phasecal_null_helpers_typical_vs_worst() -> None:
    typ, worst = null_phase_s(0.3, 857.0, 20, 5)
    acq = acquire_s(0.3, 20, 857.0)
    assert typ == pytest.approx(rt.GPIB_TXN_S + 2 * acq + (2 * rt.GPIB_TXN_S + 1.5))
    assert worst == pytest.approx(rt.GPIB_TXN_S + 5 * acq + 4 * (2 * rt.GPIB_TXN_S + 1.5))
    one = null_phase_s(0.3, 857.0, 20, 1)                             # max 1 round: no phase write, no sleep
    assert one == (pytest.approx(rt.GPIB_TXN_S + acq),) * 2
    ftyp, _ = null_follower_s(0.3, 857.0, 20, 5)
    assert ftyp > null_phase_s(0.3, 857.0, 20, 5, 1.5)[0]             # + harmonic switch / restore + 2 settles


def test_phasecal_worst_case_line_and_summary(tmp_path) -> None:
    rc = pc.run_costs(_pc_state())
    assert rc.worst_extra_s > 0 and any("worst case" in ln for ln in rc.lines())
    info, _, _ = pc.build_summary(_pc_state(data_dir=str(tmp_path)))
    assert any(i.startswith("Estimated run time ≈ ") for i in info)


def test_phasecal_plan_carries_cost_aligned_with_total_points(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = pc.MFLIPhaseCalibrationApp()
    app.data_root = tmp_path
    plan = app._build_plan(_pc_state(sweep_rows_parsed=[(-1.0, 1.0, 10), (1.0, 10.0, 10)]))
    assert plan.total_points == 37 == len(plan.run_cost.points)
