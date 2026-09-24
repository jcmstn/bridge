"""Run-time model: RunCost bookkeeping + the per-instrument helpers' arithmetic."""
import pytest

from instruments import run_time as rt
from instruments.keithley2182 import read_time_s
from instruments.keithley6221 import reversal_avg_s
from instruments.kepco_magnet import (
    FIELD_SETTLE_POLL_S, FIELD_SETTLE_TIMEOUT_S, FIELD_SETTLE_WINDOW_N, MagnetConfig, magnet_move_s,
)
from instruments.lakeshore475 import GaussmeterConfig, read_field_s
from instruments.mfli_daq import _poll_duration_s, acquire_s, poll_window_s


class _Filt:
    time_constant_s = 0.3


class _Cfg:
    filter = _Filt()
    sample_rate_Hz = 857.0


def test_runcost_each_and_at_and_lines():
    rc = rt.RunCost(4)
    rc.each("settle", 2.0)
    rc.at("ramps", 10.0, 0)
    rc.at("ramps", 6.0, -1)
    rc.tail("ramps", 4.0)
    assert rc.points == [12.0, 2.0, 2.0, 8.0]
    assert rc.total_s == 28.0                                   # points 24 + teardown tail 4
    assert rc.parts == {"settle": 8.0, "ramps": 20.0}
    (line,) = rc.lines()
    assert line.startswith("Run time: ≈ 28s") and "ramps 20s" in line and "settle 8s" in line


def test_runcost_worst_case_line_only_when_material():
    rc = rt.RunCost(2)
    rc.each("magnet", 3.0, worst_extra=20.0)
    lines = rc.lines()
    assert len(lines) == 2 and "Worst case: ≈ 46s" in lines[1]
    assert len(rt.RunCost(2).lines()) == 1


def test_progress_helpers_cost_weighted_and_fallback():
    rc = rt.RunCost(3)
    rc.each("x", 5.0)
    rc.at("start", 10.0, 0)
    rc.tail("teardown", 8.0)                                    # after the last point: not on the bar
    assert rt.progress_total(rc, 3) == 25.0
    assert [rt.progress_step(rc, i) for i in range(3)] == [15.0, 5.0, 5.0]
    assert rt.progress_total(None, 3) == 3.0 and rt.progress_step(None, 0) == 1.0
    assert rt.progress_step(rc, 99) == 1.0            # out-of-range index never raises


def test_empty_runcost_does_not_break():
    rc = rt.RunCost(0)
    rc.at("ramps", 5.0, 0)
    assert rc.total_s == 0.0 and rt.progress_total(rc, 0) == 0.0


def test_magnet_move_arithmetic():
    cfg = MagnetConfig(ramp_step_A=0.1, ramp_delay_s=0.05)
    typ, worst = magnet_move_s(2.0, cfg)                       # 20 steps
    base = rt.GPIB_TXN_S + 20 * (0.05 + 2 * rt.GPIB_TXN_S) + rt.GPIB_TXN_S
    floor = (FIELD_SETTLE_WINDOW_N - 1) * FIELD_SETTLE_POLL_S + FIELD_SETTLE_WINDOW_N * rt.GPIB_TXN_S
    assert typ == pytest.approx(base + floor + rt.FIELD_SETTLE_EXTRA_S)
    assert worst == pytest.approx(base + FIELD_SETTLE_TIMEOUT_S)
    assert magnet_move_s(2.0, cfg, with_field=False) == (pytest.approx(base), pytest.approx(base))
    # a zero move still costs one ramp step plus the settle floor
    typ0, _ = magnet_move_s(0.0, cfg)
    assert typ0 >= floor
    assert magnet_move_s(4.0, cfg)[0] > typ                    # bigger hop, longer


def test_read_field_s_matches_loop_shape():
    cfg = GaussmeterConfig(n_averages=10, read_delay_s=0.05)
    assert read_field_s(cfg) == pytest.approx(10 * (0.05 + rt.GPIB_TXN_S))


def test_2182_read_time_uses_50hz_and_a_txn():
    assert read_time_s(5) == pytest.approx(5 / 50.0 * rt.READ_2182_FACTOR + rt.GPIB_TXN_S)
    assert read_time_s(0.0001) >= 1e-3                          # floor


def test_reversal_avg_delay_and_read_add_not_max():
    # source_delay 0.05 s, read 0.1 s: sequential -> 0.15 s per half, not max(...)=0.1
    t = reversal_avg_s(5, 0.05, 0.1)
    assert t == pytest.approx(5 * 2 * (rt.GPIB_TXN_S + 0.05 + 0.1) + rt.GPIB_TXN_S)
    assert t > 5 * 2 * max(0.05, 0.1)
    # two channels: each read preceded by a mux write + settle
    t2 = reversal_avg_s(5, 0.05, 0.1, n_channels=2, channel_settle_s=0.02)
    assert t2 == pytest.approx(5 * 2 * (rt.GPIB_TXN_S + 0.05 + 2 * (0.1 + rt.GPIB_TXN_S + 0.02))
                               + rt.GPIB_TXN_S)


def test_mfli_window_is_the_loops_own_function():
    # the estimate and acquire_averaged() must share one formula (3*TC floor included)
    assert poll_window_s(0.3, 50, 857.0) == _poll_duration_s(_Cfg(), 50) == pytest.approx(0.9)
    assert poll_window_s(0.001, 200, 857.0) == pytest.approx(300 / 857.0)    # n-sample term wins
    assert acquire_s(0.3, 50, 857.0) == pytest.approx(0.9 + rt.ACQ_OVERHEAD_S)


def test_eta_scales_remaining_model_by_observed_pace():
    rc = rt.RunCost(4)
    rc.each("x", 10.0)
    assert rt.eta_s(None, 1, 5.0) is None
    assert rt.eta_s(rc, 0, 0.0) == 40.0                        # before point 1: the model itself
    assert rt.eta_s(rc, 1, 10.0) == pytest.approx(30.0)        # on model pace
    assert rt.eta_s(rc, 2, 40.0) == pytest.approx(40.0)        # running 2x slow -> remaining doubles
    assert rt.eta_s(rc, 4, 99.0) == 0.0
    rc.tail("teardown", 6.0)
    assert rt.eta_s(rc, 4, 99.0) == 6.0                         # only shutdown left, at face value
    assert rt.eta_s(rc, 1, 10.0) == pytest.approx(36.0)
