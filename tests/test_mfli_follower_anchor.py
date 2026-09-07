"""
null_follower_reference_via_1f() measures the follower's 1f delay angle by
briefly switching its demod to the 1st harmonic. The load-bearing property
is state restoration: a bug that leaves the follower on harmonic=1, or its
phaseshift moved, silently corrupts every 2f point in the run that follows.

Hardware-free — a fake DAQ that scripts a clean in-phase phasor (so
auto_null_phase converges on the first iteration) and records node writes.
"""

from __future__ import annotations

import numpy as np

from mfli.mfli_dual_harmonic import null_follower_reference_via_1f


class _Cfg:
    device = "devf"          # _poll_demod lowercases the path
    demod_index = 0
    harmonic = 2
    sample_rate_Hz = 1000.0

    class filter:             # only time_constant_s is read; 0 -> no real sleeps
        time_constant_s = 0.0


class _DAQ:
    def __init__(self) -> None:
        self.phaseshift = 3.0        # non-zero start — must come back to this
        self.harmonic = 2
        self.int_writes: list[tuple[str, int]] = []

    def getDouble(self, path):
        return self.phaseshift if path.endswith("/phaseshift") else 0.0

    def setDouble(self, path, value):
        if path.endswith("/phaseshift"):
            self.phaseshift = value

    def setInt(self, path, value):
        self.int_writes.append((path, value))
        if path.endswith("/harmonic"):
            self.harmonic = value

    def sync(self): pass
    def subscribe(self, path): pass
    def unsubscribe(self, path): pass

    def poll(self, duration_s, timeout_ms, flat=True):
        path = f"/{_Cfg.device}/demods/{_Cfg.demod_index}/sample"
        n = 64
        return {path: {"x": np.full(n, 1e-3), "y": np.zeros(n)}}


def test_anchor_switches_to_1f_then_restores_harmonic_and_phaseshift():
    daq = _DAQ()
    angle = null_follower_reference_via_1f(daq, _Cfg(), n_averages=8, max_iterations=3)

    assert isinstance(angle, float)
    # It went to the 1st harmonic to do the null ...
    assert ("/devf/demods/0/harmonic", 1) in daq.int_writes
    # ... and the LAST harmonic write puts it back to the cfg value (2f).
    assert daq.harmonic == 2
    assert daq.int_writes[-1] == ("/devf/demods/0/harmonic", 2)
    # Phaseshift ends exactly where it started — the function only measures.
    assert daq.phaseshift == 3.0
