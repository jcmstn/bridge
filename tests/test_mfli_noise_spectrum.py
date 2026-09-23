"""
Tests for mfli/mfli_noise_spectrum.py.

No hardware needed:
  - save_results() only touches the filesystem via allocate_run()/
    write_record()/finalize_index_row() (naming/index side effects), so we
    exercise it directly against a tmp_path data root with a synthetic
    `results` dict shaped like measure_noise_spectrum()'s real output.
  - compute_psd()/summarize_asd() are pure math — checked against a
    synthetic white-noise signal of known amplitude spectral density, to
    catch a Welch-normalization regression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mfli.mfli_noise_spectrum as noise
from instruments.data_naming import ensure_sample, read_raw


def _fake_spec(label: str) -> dict:
    freq = np.array([1.0, 10.0, 100.0])
    return {
        "freq_Hz": freq,
        "asd_x_V_rthz": np.array([1e-8, 2e-8, 3e-8]),
        "asd_y_V_rthz": np.array([1e-8, 2e-8, 3e-8]),
        "asd_avg_V_rthz": np.array([1e-8, 2e-8, 3e-8]),
        "nyquist_Hz": 100.0,
        "rms_V": 1e-7,
        "label": label,
        "overload_detected": False,
        "mds_synced": True,
        "leader_reference_locked": True,
        "follower_reference_locked": True,
        "white_floor_V_rthz": 2e-8,
        "corner_freq_Hz": 5.0,
    }


def test_save_results_allocates_one_run_per_pair(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(noise, "_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)

    results = {
        ("Excitation ON", "MFLI-1 (1f channel)"): _fake_spec("MFLI-1 (1f channel)"),
        ("Excitation OFF", "MFLI-1 (1f channel)"): _fake_spec("MFLI-1 (1f channel)"),
    }

    contexts = noise.save_results(
        results, sample="A", device="HB3", cooldown="", series="A_HB3_NOISE_20260101T000000",
    )

    assert [c.run_number for c in contexts] == [1, 2]
    assert contexts[0].raw_path.name.startswith("A_0001_HB3_NOISE_")
    assert contexts[0].raw_path.exists()

    df = read_raw(contexts[0].raw_path)
    assert list(df["frequency_Hz"]) == [1.0, 10.0, 100.0]

    index_path = tmp_path / "A" / "index.csv"
    index_df = pd.read_csv(index_path)
    assert set(index_df["run"]) == {1, 2}
    assert set(index_df["condition"]) == {"Excitation ON", "Excitation OFF"}


def test_save_results_key_axis_reaches_filename(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(noise, "_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)

    results = {("Excitation ON", "MFLI-1 (1f channel)"): _fake_spec("MFLI-1 (1f channel)")}
    contexts = noise.save_results(
        results, sample="A", device="HB3", cooldown="", series="A_HB3_NOISE_20260101T000000",
        key_axis=("current_A", 2e-4),
    )
    assert "I0p0002A" in contexts[0].raw_path.name


def test_summarize_asd_reproduces_known_white_noise_floor() -> None:
    """Synthetic white noise of known ASD sigma_v [V/√Hz] at known sample
    rate must round-trip through compute_psd()/summarize_asd() within a
    few percent -- catches a Welch scaling/normalization bug."""
    fs = 10_000.0
    duration_s = 20.0
    sigma_v = 5e-8  # target white-noise ASD, V/√Hz

    rng = np.random.default_rng(0)
    n = int(fs * duration_s)
    # White noise with ASD = sigma_v means variance = sigma_v**2 * (fs/2)
    # (one-sided PSD integrated over the full Nyquist band).
    x = rng.normal(0.0, sigma_v * np.sqrt(fs / 2), n)

    freq, psd = noise.compute_psd(x, fs, seg_s=2.0, overlap_frac=0.5)
    asd = np.sqrt(psd)
    stats = noise.summarize_asd(freq, asd)

    assert stats["white_floor_V_rthz"] == pytest.approx(sigma_v, rel=0.1)


class _FakeStreamDAQ:
    """poll(flat=True) the way zhinst returns it: {path: {field: array}}."""

    def __init__(self, samples_per_chunk: int = 4):
        self.n = samples_per_chunk
        self.calls = 0

    def getDouble(self, path): return 100.0
    def getInt(self, path): return 0
    def subscribe(self, path): self.path = path
    def unsubscribe(self, path): pass
    def sync(self): pass

    def poll(self, duration_s, timeout_ms, flat=True):
        self.calls += 1
        base = float(self.calls)
        return {self.path: {"timestamp": np.arange(self.n),
                            "x": np.full(self.n, base), "y": np.full(self.n, -base)}}


def test_acquire_time_series_concatenates_every_chunk() -> None:
    cfg = noise.NoiseDemodConfig(device="dev1234", demod_index=0, label="leader")
    daq = _FakeStreamDAQ(samples_per_chunk=4)
    out = noise.acquire_time_series(daq, cfg, duration_s=3.0, chunk_s=1.0)
    assert daq.calls == 3
    assert out["x"].tolist() == [1.0] * 4 + [2.0] * 4 + [3.0] * 4
    assert out["y"].tolist() == [-1.0] * 4 + [-2.0] * 4 + [-3.0] * 4
    assert out["fs"] == 100.0 and out["overload_detected"] is False
