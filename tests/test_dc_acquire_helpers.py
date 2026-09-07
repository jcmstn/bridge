"""
The shared DC acquisition helpers: the per-point error column is the
standard error of the mean (not population sigma), and the Hall loop
forwards its inter-reversal settle the way the spin-valve loop does.

Hardware-free — a fake voltmeter/source with a scripted `.voltage` stream.
"""

from __future__ import annotations

import math

import numpy as np

import instruments.keithley6221 as k6221
from instruments.keithley2182 import acquire_averaged_voltage
from instruments.keithley6221 import acquire_reversal_averaged_voltage


class _FakeVoltmeter:
    def __init__(self, values):
        self._values = list(values)
        self._k = 0

    @property
    def voltage(self):
        v = self._values[self._k]
        self._k += 1
        return v


class _FakeSource:
    source_current = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# acquire_averaged_voltage — plain-average path (I-V, gate sweep, reversal-off)
# ─────────────────────────────────────────────────────────────────────────────

def test_averaged_voltage_sem_is_sample_stdev_over_sqrt_n():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = acquire_averaged_voltage(_FakeVoltmeter(samples), len(samples))
    assert out["mean"] == 3.0
    expected = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    assert abs(out["sem"] - expected) < 1e-12
    assert "std" not in out


def test_averaged_voltage_single_sample_sem_is_nan():
    out = acquire_averaged_voltage(_FakeVoltmeter([2.0]), 1)
    assert out["mean"] == 2.0
    assert math.isnan(out["sem"])


# ─────────────────────────────────────────────────────────────────────────────
# acquire_reversal_averaged_voltage — odd/even decomposition + SEM
# ─────────────────────────────────────────────────────────────────────────────

def _reversal_stream(odd_vals, even_vals):
    """v_plus/v_minus pairs that decompose to the given odd/even parts."""
    stream = []
    for o, e in zip(odd_vals, even_vals):
        stream += [e + o, e - o]          # (v+ - v-)/2 = o, (v+ + v-)/2 = e
    return stream


def test_reversal_sem_for_odd_and_even(monkeypatch):
    monkeypatch.setattr(k6221.time, "sleep", lambda *_: None)
    odd, even = [10.0, 20.0, 30.0], [1.0, 1.0, 1.0]
    out = acquire_reversal_averaged_voltage(
        _FakeSource(), _FakeVoltmeter(_reversal_stream(odd, even)),
        sense_current_A=1e-3, n_reversals=3)
    assert out["mean"] == 20.0
    assert out["even_mean"] == 1.0
    assert abs(out["sem"] - float(np.std(odd, ddof=1) / np.sqrt(3))) < 1e-12
    assert out["even_sem"] == 0.0
    assert {"std", "even_std"} .isdisjoint(out)


def test_reversal_single_pair_sem_is_nan(monkeypatch):
    monkeypatch.setattr(k6221.time, "sleep", lambda *_: None)
    out = acquire_reversal_averaged_voltage(
        _FakeSource(), _FakeVoltmeter([11.0, -9.0]),
        sense_current_A=1e-3, n_reversals=1)
    assert math.isnan(out["sem"])
    assert math.isnan(out["even_sem"])


def test_reversal_sleeps_source_delay_after_every_flip(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(k6221.time, "sleep", lambda s: slept.append(s))
    acquire_reversal_averaged_voltage(
        _FakeSource(), _FakeVoltmeter([1.0, -1.0, 1.0, -1.0]),
        sense_current_A=1e-3, n_reversals=2, source_delay_s=0.05)
    assert slept == [0.05, 0.05, 0.05, 0.05]     # both flips, both reversals


def test_reversal_no_sleep_when_source_delay_zero(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(k6221.time, "sleep", lambda s: slept.append(s))
    acquire_reversal_averaged_voltage(
        _FakeSource(), _FakeVoltmeter([1.0, -1.0]),
        sense_current_A=1e-3, n_reversals=1)          # default source_delay_s=0.0
    assert slept == []


# ─────────────────────────────────────────────────────────────────────────────
# dc_hall_measurement.run_measurement forwards src_cfg.source_delay_s
# (regression: it used to omit it, so Hall reversals took no settle at all)
# ─────────────────────────────────────────────────────────────────────────────

def test_hall_run_measurement_forwards_source_delay(tmp_path, monkeypatch):
    import dc.dc_hall_measurement as hall

    captured: dict = {}

    def fake_acquire(source, voltmeter, sense_current_A, n_reversals,
                     stop_event=None, source_delay_s=0.0):
        captured["source_delay_s"] = source_delay_s
        captured["n_reversals"] = n_reversals
        return {"mean": 1e-6, "sem": 1e-9,
                "even_mean": 0.0, "even_sem": 0.0, "n_reversals": n_reversals}

    monkeypatch.setattr(hall, "acquire_reversal_averaged_voltage", fake_acquire)
    monkeypatch.setattr(hall.time, "sleep", lambda *_: None)

    src_cfg = hall.SourceConfig(sense_current_A=1e-3, source_delay_s=0.077)
    acq_cfg = hall.AcquisitionConfig(settling_time_s=0.0, n_reversals=4,
                                     output_file=str(tmp_path / "h.csv"))
    df = hall.run_measurement(object(), object(), src_cfg, acq_cfg, [hall.FieldPoint()])

    assert captured["source_delay_s"] == 0.077
    assert captured["n_reversals"] == 4
    assert len(df) == 1


def test_hall_run_measurement_records_field_angle_from_oop(tmp_path, monkeypatch):
    import dc.dc_hall_measurement as hall

    monkeypatch.setattr(
        hall, "acquire_reversal_averaged_voltage",
        lambda *a, **k: {"mean": 1e-6, "sem": 1e-9, "even_mean": 0.0,
                          "even_sem": 0.0, "n_reversals": k.get("n_reversals", 1)},
    )
    monkeypatch.setattr(hall.time, "sleep", lambda *_: None)

    src_cfg = hall.SourceConfig(sense_current_A=1e-3)
    acq_cfg = hall.AcquisitionConfig(settling_time_s=0.0, output_file=str(tmp_path / "h.csv"))

    df = hall.run_measurement(object(), object(), src_cfg, acq_cfg, [hall.FieldPoint()],
                              field_angle_from_oop_deg=42.0)
    assert df["field_angle_from_oop_deg"].tolist() == [42.0]

    df_none = hall.run_measurement(object(), object(), src_cfg, acq_cfg, [hall.FieldPoint()])
    assert df_none["field_angle_from_oop_deg"].isna().all()
