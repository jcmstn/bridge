"""
The loops that read two demodulators per point do it in ONE shared poll
window (acquire_averaged_pair) and keep each channel's result on its own
columns: differential resistance (current-sense leader + voltage-sense
follower) and the phase-calibration field sweep (1f + 2f). The SOT 2nd-
harmonic loop is covered in test_sot_pulsed_switching_2h_run_loops.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import mfli.mfli_diff_resistance_vs_bias as dr
import mfli.mfli_phase_calibration as pc
from test_mfli_daq import _FakeDAQPair

_N = 8


def _const(x: float, y: float = 0.0):
    return np.full(_N, x), np.full(_N, y)


def test_diff_resistance_reads_current_and_voltage_in_one_window(monkeypatch):
    monkeypatch.setattr(dr, "set_bias", lambda *a, **k: None)
    current = dr.DemodConfig(device="lead", label="I", use_current_input=True)
    voltage = dr.DemodConfig(device="follow", label="V")
    daq = _FakeDAQPair({"/lead/demods/0/sample": _const(2e-6),        # 2 µA through the DUT
                        "/follow/demods/0/sample": _const(4e-3)})     # 4 mV across it
    acq = dr.AcquisitionConfig(settling_time_s=0.0, n_averages=_N)
    points = [dr.BiasPoint(bias_V=v) for v in (0.0, 0.1, 0.2)]

    df = dr.run_measurement(daq, dr.OutputConfig(), current, voltage, acq, points,
                            write_csv=lambda records: None)

    assert daq.poll_calls == len(points)                              # one window per point
    assert list(df["I_ac_A"]) == pytest.approx([2e-6] * 3)
    assert list(df["V_dut_ac_V"]) == pytest.approx([4e-3] * 3)
    assert list(df["R_diff_ohm"]) == pytest.approx([2000.0] * 3)      # V / I, not swapped
    assert list(df["I_n_samples"]) == list(df["V_n_samples"]) == [_N] * 3


def test_phase_calibration_sweep_reads_1f_and_2f_in_one_window(monkeypatch):
    monkeypatch.setattr(pc, "set_magnet_current", lambda *a, **k: None)
    d1 = pc.DemodConfig(device="lead", demod_index=0, harmonic=1)
    d2 = pc.DemodConfig(device="follow", demod_index=1, harmonic=2)
    daq = _FakeDAQPair({"/lead/demods/0/sample": _const(1e-3, 1e-5),
                        "/follow/demods/1/sample": _const(2e-6)})
    sweep = pc.SweepConfig(rows=[(-1.0, 1.0, 3)], settling_time_s=0.0, n_averages=_N)

    df = pc.run_field_sweep_diagnostic(daq, d1, d2, sweep, magnet=None, magnet_cfg=None)

    assert daq.poll_calls == len(df) > 0
    assert list(df["1f_X_V"]) == pytest.approx([1e-3] * len(df))
    assert list(df["1f_residual_ratio"]) == pytest.approx([1e-5 / np.hypot(1e-3, 1e-5)] * len(df))
    assert list(df["2f_X_V"]) == pytest.approx([2e-6] * len(df))
