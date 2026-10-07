"""Run-time model for mfli_diff_resistance: the estimate is the
sum of what the loop does (per-point cost list = progress-bar weights), not settle + one window."""

from __future__ import annotations

import pytest

import mfli.mfli_diff_resistance_tui as dr
from instruments import run_time as rt
from instruments.data_naming import ensure_sample
from instruments.mfli_daq import acquire_s, poll_window_s


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


# ── diff_resistance ──────────────────────────────────────────────────────────

def test_diffres_one_shared_window_with_3tc_floor() -> None:
    state = _dr_state()
    n = 2 * state["n_points"] - 1                                    # bidirectional, turn-around not repeated
    rc = dr.run_costs(n, state)
    old_estimate = n * (1.5 + 2 * max(0.1, 50 * 1.5 / 857.0))        # settle + 2 windows WITHOUT the 3·TC floor
    assert old_estimate == pytest.approx(137.7, abs=0.1)
    assert rc.total_s > old_estimate
    # one steady-state point = settle + ONE shared I/V acquisition window
    # (acquire_averaged_pair; 3·TC = 0.9 s) + overhead
    assert rc.points[1] == pytest.approx(
        1.5 + acquire_s(0.3, 50, 857.0) + 3 * rt.GPIB_TXN_S + rt.POINT_OVERHEAD_S)
    assert acquire_s(0.3, 50, 857.0) == pytest.approx(poll_window_s(0.3, 50, 857.0) + rt.ACQ_OVERHEAD_S)


def test_diffres_slow_filter_dominates_the_point() -> None:
    fast, slow = dr.run_costs(2, _dr_state()), dr.run_costs(2, _dr_state(time_constant_s=1.0))
    assert slow.points[1] - fast.points[1] == pytest.approx(3.0 - 0.9)   # 3·TC, one shared window


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
    line = next(i for i in info if i.startswith("Run time"))
    assert dr.run_costs(81, _dr_state()).lines()[0] == line
