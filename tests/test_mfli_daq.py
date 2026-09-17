"""
acquire_averaged(): R and theta must be the polar form of the vector-averaged
phasor, not the noise-rectifying mean of per-sample magnitudes/angles. This
is the estimator a small 2f harmonic-Hall signal actually depends on.

Hardware-free — a fake DAQ that returns a scripted x/y poll payload.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from instruments.mfli_daq import acquire_averaged, acquire_averaged_pair


class _FakeCfg:
    device = "dev0000"
    demod_index = 0
    sample_rate_Hz = 1000.0
    # no input_ch -> overload reported as None, no getInt needed


class _FakeCfgTC(_FakeCfg):
    class filter:            # mimics FilterConfig — only time_constant_s is read
        time_constant_s = 0.3


class _FakeDAQ:
    def __init__(self, x, y):
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self.last_duration_s = None

    def subscribe(self, path): pass
    def unsubscribe(self, path): pass
    def sync(self): pass

    def poll(self, duration_s, timeout_ms, flat=True):
        self.last_duration_s = duration_s
        path = f"/{_FakeCfg.device}/demods/{_FakeCfg.demod_index}/sample"
        return {path: {"x": self._x, "y": self._y}}


def test_r_and_theta_are_polar_of_the_mean_phasor():
    # Clean phasor at 30 degrees, magnitude 2.
    ang = math.radians(30.0)
    x = np.full(64, 2.0 * math.cos(ang))
    y = np.full(64, 2.0 * math.sin(ang))
    out = acquire_averaged(_FakeDAQ(x, y), _FakeCfg(), n_averages=64)

    assert abs(out["x_mean"] - 2.0 * math.cos(ang)) < 1e-12
    assert abs(out["r_mean"] - 2.0) < 1e-12
    assert abs(out["theta_mean"] - 30.0) < 1e-9


def test_zero_mean_noise_gives_near_zero_R_not_a_rician_offset():
    # True signal is zero; only noise. The old per-sample mean(hypot(x,y))
    # returned a large positive R here (noise rectification). Vector-averaging
    # first must collapse it toward zero.
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0.0, 1.0, n)
    y = rng.normal(0.0, 1.0, n)
    out = acquire_averaged(_FakeDAQ(x, y), _FakeCfg(), n_averages=n)

    per_sample_mean_r = float(np.mean(np.hypot(x, y)))
    assert per_sample_mean_r > 1.0                      # the biased estimator
    assert out["r_mean"] < 0.1                           # vector-averaged: ~0
    assert out["r_mean"] == math.hypot(out["x_mean"], out["y_mean"])


def test_poll_window_is_floored_at_3x_time_constant():
    # 10 * 1.5 / 1000 Sa/s = 0.015 s of samples, but TC = 0.3 s, so a window
    # shorter than 3*TC = 0.9 s would return correlated (not independent)
    # samples — acquire_averaged must stretch the poll to 0.9 s.
    daq = _FakeDAQ(np.zeros(8), np.zeros(8))
    acquire_averaged(daq, _FakeCfgTC(), n_averages=10)
    assert daq.last_duration_s == pytest.approx(0.9)


def test_poll_window_without_a_filter_attr_uses_sample_count():
    # _FakeCfg has no .filter — the 3*TC floor is simply skipped.
    daq = _FakeDAQ(np.zeros(4000), np.zeros(4000))
    acquire_averaged(daq, _FakeCfg(), n_averages=2000)
    assert daq.last_duration_s == (2000 * 1.5) / 1000.0


def test_r_sem_is_propagated_from_xy_sem_not_the_std_of_magnitudes():
    # x=[1,2,3], y=[4,5,6]: sample stdev (ddof=1) of each is 1, so
    # x_sem = y_sem = 1/sqrt(3). r_sem must be that propagated onto
    # r_mean = hypot(x_mean, y_mean) via dR/dX=X/R, dR/dY=Y/R — a different
    # number from std(hypot(x_i, y_i)), which is what the old (wrong) column
    # reported.
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    out = acquire_averaged(_FakeDAQ(x, y), _FakeCfg(), n_averages=3)

    assert out["x_mean"] == pytest.approx(2.0)
    assert out["y_mean"] == pytest.approx(5.0)
    assert out["x_sem"] == pytest.approx(1.0 / math.sqrt(3))
    assert out["y_sem"] == pytest.approx(1.0 / math.sqrt(3))
    assert out["r_sem"] == pytest.approx(1.0 / math.sqrt(3))
    assert out["n_samples"] == 3
    # The old, wrong estimator this replaces — kept only as x_std/y_std/r_std
    # for mfli_phase_calibration.py's signal-to-noise diagnostic, never as
    # the uncertainty on r_mean.
    assert out["r_std"] != pytest.approx(out["r_sem"])


def test_sem_fields_are_nan_for_a_single_sample():
    out = acquire_averaged(_FakeDAQ(np.array([5.0]), np.array([5.0])), _FakeCfg(), n_averages=1)
    assert math.isnan(out["x_sem"])
    assert math.isnan(out["y_sem"])
    assert math.isnan(out["r_sem"])
    assert out["n_samples"] == 1


class _FakeCfgB:
    device = "dev1111"
    demod_index = 0
    sample_rate_Hz = 1000.0


class _FakeCfgBTC(_FakeCfgB):
    class filter:
        time_constant_s = 0.1


class _FakeDAQPair:
    """Subscribe/poll fake that serves multiple paths from ONE poll() call,
    like the real LabOne session-level poll() (see acquire_averaged_pair()'s
    docstring) -- as opposed to _FakeDAQ above, which only ever has one path
    subscribed at a time."""

    def __init__(self, data: dict):
        self._data = data  # {path: (x, y)}
        self.subscribed: list = []
        self.poll_calls = 0
        self.last_duration_s = None

    def subscribe(self, path):
        self.subscribed.append(path)

    def unsubscribe(self, path):
        self.subscribed.remove(path)

    def sync(self):
        pass

    def poll(self, duration_s, timeout_ms, flat=True):
        self.poll_calls += 1
        self.last_duration_s = duration_s
        return {
            path: {"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}
            for path, (x, y) in self._data.items()
            if path in self.subscribed
        }


def test_acquire_averaged_pair_uses_one_poll_for_both_demods():
    path_a = "/dev0000/demods/0/sample"
    path_b = "/dev1111/demods/0/sample"
    ang_a, ang_b = math.radians(30.0), math.radians(60.0)
    xa = np.full(16, 2.0 * math.cos(ang_a)); ya = np.full(16, 2.0 * math.sin(ang_a))
    xb = np.full(16, 3.0 * math.cos(ang_b)); yb = np.full(16, 3.0 * math.sin(ang_b))
    daq = _FakeDAQPair({path_a: (xa, ya), path_b: (xb, yb)})

    d_a, d_b = acquire_averaged_pair(daq, _FakeCfg(), _FakeCfgB(), n_averages=16)

    assert daq.poll_calls == 1
    assert d_a["r_mean"] == pytest.approx(2.0)
    assert d_b["r_mean"] == pytest.approx(3.0)
    assert d_a["theta_mean"] == pytest.approx(30.0)
    assert d_b["theta_mean"] == pytest.approx(60.0)


def test_acquire_averaged_pair_duration_is_max_not_sum():
    # cfg_a's TC=0.3s floors its own window at 0.9s; cfg_b's TC=0.1s floors
    # its own window at 0.3s. The shared poll must cover both in ONE call
    # at the MAX (0.9s), not the sum (1.2s) -- that's the whole point.
    path_a = f"/{_FakeCfgTC.device}/demods/{_FakeCfgTC.demod_index}/sample"
    path_b = f"/{_FakeCfgBTC.device}/demods/{_FakeCfgBTC.demod_index}/sample"
    daq = _FakeDAQPair({path_a: (np.zeros(8), np.zeros(8)),
                         path_b: (np.zeros(8), np.zeros(8))})

    acquire_averaged_pair(daq, _FakeCfgTC(), _FakeCfgBTC(), n_averages=10)

    assert daq.poll_calls == 1
    assert daq.last_duration_s == pytest.approx(0.9)


def test_acquire_averaged_pair_raises_naming_the_missing_path():
    # If one demod is disabled/misconfigured, poll() comes back missing
    # that path's key -- the error must name WHICH one, not just "no data".
    path_a = "/dev0000/demods/0/sample"
    path_b = "/dev1111/demods/0/sample"
    daq = _FakeDAQPair({path_a: (np.zeros(4), np.zeros(4))})

    with pytest.raises(RuntimeError) as exc:
        acquire_averaged_pair(daq, _FakeCfg(), _FakeCfgB(), n_averages=4)
    assert path_b in str(exc.value)
