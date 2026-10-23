import numpy as np
import pandas as pd
import pytest

from freq_check import branches, common_grid, judge, on_grid, signal_verdicts

FREQS = [113.0, 317.3, 731.0]


def _curves(k_per_Hz: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.tanh(np.linspace(-3, 3, 50))  # Hall-like loop, ptp ≈ 2
    return np.array([base + k_per_Hz * f + rng.normal(0, 1e-3, 50) for f in FREQS])


def test_flat_is_ok_and_trend_is_not():
    assert judge(FREQS, _curves(0.0), tol=0.02)["ok"]
    r = judge(FREQS, _curves(1e-3), tol=0.02)  # Δ ≈ 0.6 over 618 Hz, 30 % of scale
    assert not r["ok"] and r["rel"] > 0.2 and r["z"] > 3


def test_tiny_significant_trend_is_still_ok():
    r = judge(FREQS, _curves(1e-5), tol=0.02)  # well above noise, but < 1 % of scale
    assert r["z"] > 3 and r["ok"]


def test_up_down_sweep_splits_and_interpolates_per_branch():
    up = np.linspace(-1, 1, 5)
    down = up[::-1][1:]  # turning point measured once, belongs to the up branch
    a = pd.DataFrame({"magnet_current_A": np.r_[up, down],
                      "magnet_field_mT": np.r_[up, down] * 10,
                      "y": np.r_[up, down + 5]})  # down branch offset (hysteresis)
    assert [b.tolist() for b in branches(a, "magnet_field_mT")] == [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
    b = a.copy()
    b["magnet_field_mT"] += 1.0  # shifted field points, same physics
    b["y"] = np.r_[up + 0.1, down + 5.1]
    grids = common_grid([a, b], "magnet_field_mT")
    assert [g.tolist() for g in grids] == [[-5.0, 0.0, 5.0, 10.0], [5.0, 0.0, -5.0]]
    assert on_grid(b, "magnet_field_mT", "y", grids).tolist() == pytest.approx([-0.5, 0.0, 0.5, 1.0, 5.5, 5.0, 4.5])


def test_every_signal_must_pass():
    results = {"1f_X_ohm": {"ok": True}, "1f_Y_ohm": {"ok": True},
               "rxx_1f_X_ohm": {"ok": True}, "rxx_1f_Y_ohm": {"ok": False}}
    assert signal_verdicts(results) == {"1f": [], "rxx_1f": ["Y"]}
