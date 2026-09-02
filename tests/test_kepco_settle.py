"""
set_magnet_current()'s settle waits: the pure window detector and the
end-to-end current/field convergence with a hard timeout.
"""

from __future__ import annotations

import instruments.kepco_magnet as km
from instruments.kepco_magnet import MagnetConfig, _window_settled, set_magnet_current


# ─────────────────────────────────────────────────────────────────────────────
# _window_settled — the drift criterion
# ─────────────────────────────────────────────────────────────────────────────

def test_window_settled_needs_a_full_window():
    assert _window_settled([1.0, 1.0, 1.0], tol=0.1) is False  # < WINDOW_N samples


def test_window_settled_flat_within_tolerance():
    assert _window_settled([5.0, 50.0, 50.001, 49.999, 50.0], tol=0.02) is True


def test_window_settled_still_ramping():
    assert _window_settled([10.0, 20.0, 30.0, 40.0, 50.0], tol=0.02) is False


def test_window_settled_single_glitch_in_window_blocks():
    # last WINDOW_N span exceeds tol because of one outlier
    assert _window_settled([50.0, 50.0, 50.2, 50.0, 50.0], tol=0.02) is False


# ─────────────────────────────────────────────────────────────────────────────
# set_magnet_current — fakes for the supply and gaussmeter
# ─────────────────────────────────────────────────────────────────────────────

class _FakePSU:
    """measure_current walks toward the last ramp_current() target."""
    def __init__(self, per_read_step=1.0):
        self._i = 0.0
        self._target = 0.0
        self._step = per_read_step
        self.reads = 0

    def ramp_current(self, target, step=0.1, delay=0.05):
        self._target = target  # setpoint reached instantly; readback lags

    @property
    def measure_current(self):
        self.reads += 1
        if self._i < self._target:
            self._i = min(self._target, self._i + self._step)
        elif self._i > self._target:
            self._i = max(self._target, self._i - self._step)
        return self._i


class _FakeGauss:
    """field returns from a fixed list, repeating the last value forever."""
    def __init__(self, readings):
        self._readings = list(readings)
        self._k = 0

    @property
    def field(self):
        v = self._readings[min(self._k, len(self._readings) - 1)]
        self._k += 1
        return v


class _DriftingGauss:
    """field never stops climbing — used to exercise the settle timeout."""
    def __init__(self, step_T=0.01):
        self._step = step_T
        self._k = 0

    @property
    def field(self):
        self._k += 1
        return self._k * self._step


class _GaussCfg:
    unit = "T"


def _cfg():
    return MagnetConfig(current_limit_A=50.0, ramp_step_A=1.0, ramp_delay_s=0.0)


def test_current_and_field_both_settle(monkeypatch):
    monkeypatch.setattr(km.time, "sleep", lambda *_: None)
    psu = _FakePSU(per_read_step=2.0)
    # field in Tesla: drifts a couple of steps then holds at 0.05 T (= 50 mT)
    gm = _FakeGauss([0.030, 0.045, 0.0500, 0.0501, 0.0499, 0.0500, 0.0500, 0.0500])

    info = set_magnet_current(psu, _cfg(), 10.0, gm, _GaussCfg(),
                              field_settle_tolerance_mT=0.2)

    assert info["current_settled"] is True
    assert info["field_settled"] is True
    assert abs(info["i_measured_A"] - 10.0) <= _cfg().ramp_step_A


def test_current_loop_iterates_while_readback_lags(monkeypatch):
    # Readback starts at 0 and only advances 2 A per read; band for a 10 A
    # setpoint is max(0.5, 0.1) = 0.5 A, so the loop must poll several times.
    monkeypatch.setattr(km.time, "sleep", lambda *_: None)
    psu = _FakePSU(per_read_step=2.0)
    info = set_magnet_current(psu, _cfg(), 10.0)
    assert psu.reads >= 4                        # loop actually ran, not a no-op
    assert info["current_settled"] is True


def test_field_timeout_reports_not_settled(monkeypatch):
    # A zero timeout trips on the first poll — deterministic, no clock patching.
    monkeypatch.setattr(km.time, "sleep", lambda *_: None)
    monkeypatch.setattr(km, "FIELD_SETTLE_TIMEOUT_S", 0.0)
    psu = _FakePSU(per_read_step=100.0)          # current settles immediately
    gm = _DriftingGauss()                        # never stops drifting

    info = set_magnet_current(psu, _cfg(), 5.0, gm, _GaussCfg(),
                              field_settle_tolerance_mT=0.001)

    assert info["current_settled"] is True
    assert info["field_settled"] is False       # degraded, but returned — never hung
    assert info["settle_elapsed_s"] >= 0.0


def test_current_timeout_reports_not_settled(monkeypatch):
    monkeypatch.setattr(km.time, "sleep", lambda *_: None)
    monkeypatch.setattr(km, "CURRENT_SETTLE_TIMEOUT_S", 0.0)
    psu = _FakePSU(per_read_step=0.001)          # crawls, never reaches 20 A in time
    info = set_magnet_current(psu, _cfg(), 20.0)
    assert info["current_settled"] is False


def test_no_gaussmeter_skips_field_wait(monkeypatch):
    monkeypatch.setattr(km.time, "sleep", lambda *_: None)
    info = set_magnet_current(_FakePSU(per_read_step=100.0), _cfg(), 3.0)
    assert info["current_settled"] is True
    assert info["field_settled"] is None
