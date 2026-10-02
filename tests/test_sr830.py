"""
instruments/sr830.py: the manual-supported command sequences (OUTX before
any query, buffer setup, external reference, per-bit LIAS reads, TRCB?
binary transfer) and the MFLI-shaped result dict.

Hardware-free — a fake SR830 that records writes and answers scripted
queries; sleeps are patched out.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from instruments import sr830
from instruments.sr830 import (LockinConfig, acquire_averaged, acquire_averaged_pair,
                               check_reference_locked, connect, settle_time_s,
                               srat_index, validate, wait_for_reference_lock)


class _FakeSR830:
    """Records every write and pymeasure-property set in `log`, in order."""

    _READBACK = ("time_constant", "sensitivity")   # the unit snaps these; sets don't stick

    def __init__(self, resource="GPIB0::8::INSTR", x=(), y=(), lias=None, lias_in_window=None):
        object.__setattr__(self, "log", [])
        object.__setattr__(self, "lias_in_window", dict(lias_in_window or {}))  # latched at STRT
        object.__setattr__(self, "x", np.asarray(x, dtype="<f4"))
        object.__setattr__(self, "y", np.asarray(y, dtype="<f4"))
        object.__setattr__(self, "lias", dict(lias or {}))   # bit -> 0/1, cleared on read
        object.__setattr__(self, "_pending", None)
        # values a real unit reads back after snapping
        object.__setattr__(self, "time_constant", 0.3)
        object.__setattr__(self, "sensitivity", 0.002)
        object.__setattr__(self, "frequency", 1333.0)
        object.__setattr__(self, "phase", 0.0)

    def __setattr__(self, name, value):
        self.log.append(f"{name}={value}")
        if name not in self._READBACK:
            object.__setattr__(self, name, value)

    def write(self, cmd):
        self.log.append(cmd)
        if cmd == "STRT":
            self.lias.update(self.lias_in_window)
        if cmd.startswith("TRCB?"):
            ch, start, count = (int(v) for v in cmd.split()[1].split(","))
            data = self.x if ch == 1 else self.y
            object.__setattr__(self, "_pending", data[start:start + count].tobytes())

    def read_bytes(self, count):
        raw, _ = self._pending, object.__setattr__(self, "_pending", None)
        assert len(raw) == count
        return raw

    def ask(self, cmd):
        self.log.append(cmd)
        if cmd.startswith("LIAS? "):
            return str(self.lias.pop(int(cmd.split()[1]), 0))
        if cmd == "SPTS?":
            return str(len(self.x))
        if cmd == "*STB? 1":
            return "1"
        raise AssertionError(f"unexpected query {cmd}")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sr830.time, "sleep", lambda s: None)


def test_connect_sends_outx_first_and_sets_up_buffer(monkeypatch):
    monkeypatch.setattr(sr830, "SR830", _FakeSR830)
    cfg = LockinConfig(sample_rate_Hz=100.0)
    lk = connect(cfg)
    assert lk.log[0] == "OUTX 1"
    assert lk.log.index("*RST") < lk.log.index("*STB? 1") < lk.log.index("*CLS")
    for cmd in ("DDEF 1,0,0", "DDEF 2,0,0", "OEXP 1,0,0", "OEXP 2,0,0",
                "SEND 1", "TSTR 0", "SRAT 11", "HARM 1"):
        assert cmd in lk.log
    assert "reference_source=Internal" in lk.log
    assert not any(c.startswith(("FAST", "STRD")) for c in lk.log)
    # applied (snapped) values written back
    assert cfg.time_constant_s == 0.3 and cfg.sensitivity_V == 0.002
    assert cfg.sample_rate_Hz == 128.0


def test_connect_external_reference(monkeypatch):
    monkeypatch.setattr(sr830, "SR830", _FakeSR830)
    lk = connect(LockinConfig(reference="external", ext_slope="ttl_falling", harmonic=2))
    assert "reference_source=External" in lk.log
    assert "RSLP 2" in lk.log and "HARM 2" in lk.log
    assert not any(c.startswith("frequency=") for c in lk.log)   # unit tracks REF IN


@pytest.mark.parametrize("kw", [
    {"harmonic": 2, "frequency_Hz": 60e3},                      # 120 kHz > 102 kHz
    {"harmonic": 0},
    {"reference": "external", "ext_slope": "sine", "frequency_Hz": 0.5},
    {"ext_slope": "rising"},
    {"filter_slope_dB": 30},
])
def test_validate_rejects(kw):
    with pytest.raises(ValueError):
        validate(LockinConfig(**kw))


def test_srat_index_snaps_up():
    assert srat_index(0.0625) == 0
    assert srat_index(100) == 11      # 128 Hz
    assert srat_index(512) == 13
    assert srat_index(10_000) == 13   # cap


def test_settle_time_matches_manual_table():
    for slope, k in ((6, 5), (12, 7), (18, 9), (24, 10)):
        assert settle_time_s(LockinConfig(filter_slope_dB=slope, time_constant_s=0.1)) \
            == pytest.approx(k * 0.1)


def test_acquire_reads_freshest_points_and_reduces_like_mfli():
    x = [9.0] * 5 + [1.0, 3.0] * 50      # stale points first, then 100 fresh ones
    y = [9.0] * 5 + [2.0] * 100
    lk = _FakeSR830(x=x, y=y, lias_in_window={1: 1})   # filter overload during window
    cfg = LockinConfig(time_constant_s=0.01, sample_rate_Hz=512.0)
    d = acquire_averaged(lk, cfg, n_averages=100, stop_event=threading.Event())

    seq = [c for c in lk.log if c in ("REST", "STRT", "PAUS", "SPTS?")]
    assert seq == ["REST", "STRT", "PAUS", "SPTS?"]
    assert "TRCB? 1,5,100" in lk.log and "TRCB? 2,5,100" in lk.log
    assert d["n_samples"] == 100
    assert d["x_mean"] == pytest.approx(2.0) and d["y_mean"] == pytest.approx(2.0)
    assert d["r_mean"] == pytest.approx(math.hypot(2.0, 2.0))
    assert d["theta_mean"] == pytest.approx(45.0)
    assert d["x_std"] == pytest.approx(1.0) and d["y_std"] == pytest.approx(0.0)
    assert d["overload"] is True
    assert set(d) == {"x_mean", "y_mean", "r_mean", "theta_mean", "x_sem", "y_sem",
                      "r_sem", "r_std", "x_std", "y_std", "n_samples", "overload"}


def test_overload_cleared_before_window():
    # an overload latched BEFORE the acquisition must not be reported
    lk = _FakeSR830(x=[1.0] * 10, y=[0.0] * 10, lias={0: 1})
    d = acquire_averaged(lk, LockinConfig(), n_averages=10, stop_event=threading.Event())
    assert d["overload"] is False


def test_pair_shares_one_window():
    a = _FakeSR830(x=[1.0] * 20, y=[0.0] * 20)
    b = _FakeSR830(x=[0.0] * 20, y=[1.0] * 20)
    order = []
    for lk, tag in ((a, "a"), (b, "b")):
        orig = lk.write
        object.__setattr__(lk, "write", lambda c, orig=orig, tag=tag: (order.append(tag + c), orig(c)))
    da, db = acquire_averaged_pair(a, LockinConfig(), b, LockinConfig(), n_averages=20,
                                   stop_event=threading.Event())
    starts_pauses = [c for c in order if c[1:] in ("STRT", "PAUS")]
    assert starts_pauses == ["aSTRT", "bSTRT", "aPAUS", "bPAUS"]
    assert da["theta_mean"] == pytest.approx(0.0) and db["theta_mean"] == pytest.approx(90.0)


def test_unlock_bit_is_latched_and_per_bit():
    lk = _FakeSR830(lias={3: 1, 0: 1})
    assert check_reference_locked(lk) is False
    assert check_reference_locked(lk) is True
    assert lk.lias == {0: 1}      # reading bit 3 left the overload bit alone


def test_wait_for_lock_ignores_setup_unlock():
    lk = _FakeSR830(lias={3: 1})  # unlock from while the reference was being set up
    assert wait_for_reference_lock(lk, timeout_s=5.0) is True
