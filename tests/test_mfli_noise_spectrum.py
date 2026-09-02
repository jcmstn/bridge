"""
Test for mfli/mfli_noise_spectrum.py's save_results().

No hardware needed -- save_results() only touches the filesystem via
allocate_run()/write_record()/finalize_index_row() (naming/index side
effects), so we exercise it directly against a tmp_path data root with a
synthetic `results` dict shaped like measure_noise_spectrum()'s real output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
        "temperature_1_K": 4.2,
        "temperature_2_K": None,
        "overload_detected": False,
        "mds_synced": True,
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
        results, sample="A", device="HB3", temperature_setpoint_K=293.0,
        cooldown="", series="A_HB3_NOISE_20260101T000000",
    )

    assert [c.run_number for c in contexts] == [1, 2]
    assert contexts[0].raw_path.name.startswith("A_0001_HB3_NOISE_T293K_")
    assert contexts[0].raw_path.exists()

    df = read_raw(contexts[0].raw_path)
    assert list(df["frequency_Hz"]) == [1.0, 10.0, 100.0]

    index_path = tmp_path / "A" / "index.csv"
    index_df = pd.read_csv(index_path)
    assert set(index_df["run"]) == {1, 2}
    assert set(index_df["condition"]) == {"Excitation ON", "Excitation OFF"}
