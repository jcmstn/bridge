"""
Demod reference phase: measure_phi_I() (1f autophase on a resistive pair),
harmonic_phase_deg() (harmonic x phi_I) and configure_demodulator()'s write.

Hardware-free — a fake DAQ that scripts a clean in-phase phasor (so
auto_null_phase converges on the first iteration) and records node writes.
"""

from __future__ import annotations

import numpy as np



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


def test_measure_phi_I_nulls_at_1f_and_returns_the_phase():
    from mfli.mfli_dual_harmonic import measure_phi_I
    daq = _DAQ()
    (r,) = measure_phi_I(daq, [_Cfg()], n_averages=8, max_iterations=3)
    assert ("/devf/demods/0/harmonic", 1) in daq.int_writes
    assert r.converged and r.x_V > 0 and abs(r.phase_after_deg) < 1e-9


def test_harmonic_phase_deg_is_harmonic_times_phi_wrapped():
    from mfli.mfli_dual_harmonic import harmonic_phase_deg
    assert [harmonic_phase_deg(p, h) for p, h in ((10, 1), (10, 2), (100, 2), (-100, 2))] \
        == [10, 20, -160, 160]


def test_configure_demodulator_writes_phase_only_when_set():
    from mfli.mfli_dual_harmonic import DemodConfig, configure_demodulator
    daq = _DAQ()
    configure_demodulator(daq, DemodConfig(device="devf", demod_index=0, harmonic=2))
    assert daq.phaseshift == 3.0                 # None: the device's phase is left alone
    configure_demodulator(daq, DemodConfig(device="devf", demod_index=0, harmonic=2, phase_deg=-12.5))
    assert daq.phaseshift == -12.5
