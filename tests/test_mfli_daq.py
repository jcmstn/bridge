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

from instruments.mfli_daq import acquire_averaged


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
