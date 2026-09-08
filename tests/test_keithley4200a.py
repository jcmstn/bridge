"""
instruments/keithley4200a.py — KXCI reply parsing, the source-level software
guard, and the +I/-I reversal decomposition. Hardware-free: a fake KXCI
transport records every command and serves a scripted reading stream.
"""

from __future__ import annotations

import math

import pytest

import instruments.keithley4200a as k4200
from instruments.keithley4200a import SMUChannelConfig, _parse_reading


class _FakeKXCI:
    def __init__(self, replies):
        self.writes: list[str] = []
        self._replies = iter(replies)

    def command(self, cmd: str) -> None:
        self.writes.append(cmd)

    def query(self, cmd: str) -> str:
        self.writes.append(cmd)
        return next(self._replies)


# ── _parse_reading ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, want", [
    ("1.5", 1.5),
    ("  1.2345E-03 ", 1.2345e-3),
    ("N +1.2345E-03", 1.2345e-3),      # leading status letter, space-separated
    ("C -2.0E-6", -2.0e-6),
    ("NCV1.23E-3", 1.23e-3),           # status letters, no separator
])
def test_parse_reading(raw, want):
    assert _parse_reading(raw) == pytest.approx(want)


def test_parse_reading_empty_raises():
    with pytest.raises(ValueError):
        _parse_reading("   ")


# ── set_source_level: software limit guard + command shape ─────────────────

def test_set_source_level_current_command():
    dev = _FakeKXCI([])
    cfg = SMUChannelConfig(channel=1, source_function="current",
                           compliance_voltage_V=2.0, source_limit_A=5e-3)
    k4200.set_source_level(dev, cfg, 1e-3)
    assert dev.writes == ["DI1, 0, 1.000000E-03, 2.000000E+00"]


def test_set_source_level_voltage_command():
    dev = _FakeKXCI([])
    cfg = SMUChannelConfig(channel=2, source_function="voltage",
                           compliance_current_A=1e-4, source_limit_V=10.0)
    k4200.set_source_level(dev, cfg, -1.5)
    assert dev.writes == ["DV2, 0, -1.500000E+00, 1.000000E-04"]


def test_set_source_level_refuses_beyond_limit():
    dev = _FakeKXCI([])
    cfg = SMUChannelConfig(source_function="current", source_limit_A=1e-3)
    with pytest.raises(ValueError):
        k4200.set_source_level(dev, cfg, 2e-3)
    assert dev.writes == []          # nothing sent


# ── acquire_reversal_averaged: odd/even decomposition + contract ───────────

def test_acquire_reversal_averaged_decomposition():
    # TV replies, in order: pair0 (+I, -I), pair1 (+I, -I)
    dev = _FakeKXCI(["1.0", "-0.6", "1.2", "-0.4"])
    src = SMUChannelConfig(channel=1, source_function="current", source_limit_A=1.0)
    sense = SMUChannelConfig(channel=2, source_function="current", source_limit_A=1.0)

    out = k4200.acquire_reversal_averaged(dev, src, sense, level=0.1, n_reversals=2)

    # pair0: odd=(1.0-(-0.6))/2=0.8  even=(1.0+(-0.6))/2=0.2
    # pair1: odd=(1.2-(-0.4))/2=0.8  even=(1.2+(-0.4))/2=0.4
    assert out["mean"] == pytest.approx(0.8)
    assert out["sem"] == pytest.approx(0.0)
    assert out["even_mean"] == pytest.approx(0.3)
    assert out["even_sem"] == pytest.approx(0.1)          # std([0.2,0.4],ddof=1)/sqrt(2)
    assert out["n_reversals"] == 2
    assert set(out) == {"mean", "sem", "even_mean", "even_sem", "n_reversals"}

    # left forcing +level on the source channel
    assert dev.writes[-1] == "DI1, 0, 1.000000E-01, 1.000000E+01"
    # read the sense channel (SMU2), never the source, for the voltage
    assert dev.writes.count("TV2") == 4
    assert "TV1" not in dev.writes


def test_acquire_reversal_averaged_single_pair_sem_nan():
    dev = _FakeKXCI(["1.0", "-1.0"])
    src = SMUChannelConfig(channel=1, source_function="current", source_limit_A=1.0)
    sense = SMUChannelConfig(channel=2, source_function="current", source_limit_A=1.0)
    out = k4200.acquire_reversal_averaged(dev, src, sense, level=0.1, n_reversals=1)
    assert out["mean"] == pytest.approx(1.0)
    assert math.isnan(out["sem"])


def test_acquire_reversal_averaged_rejects_voltage_source():
    dev = _FakeKXCI([])
    src = SMUChannelConfig(source_function="voltage")
    sense = SMUChannelConfig(channel=2, source_function="current")
    with pytest.raises(ValueError):
        k4200.acquire_reversal_averaged(dev, src, sense, level=0.1, n_reversals=2)
