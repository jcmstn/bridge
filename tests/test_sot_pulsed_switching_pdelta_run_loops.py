"""
sot_pulsed_switching_pdelta.run_measurement — the Pulse Delta write → Delta
read sequence on one 6221 (2182A behind its RS-232). Hardware-free: a fake
6221 that logs SCPI and serves the buffer queries.
"""

from __future__ import annotations

import threading

import pytest

import sot.sot_pulsed_switching_pdelta as pdl
from instruments.keithley6221 import PulseDeltaConfig, check_2182a_on_serial
from sot.sot_pulsed_switching_6221 import PulsePoint

_NULL_WRITER = lambda recs: None


class _Fake6221:
    """Logs every write/ask. The buffer fills as soon as a mode is INIT'd; the
    Delta buffer holds `delta_v`, the Pulse Delta one a sentinel that must never
    be fetched."""

    def __init__(self, delta_v=(1e-3, 1.2e-3, 0.8e-3), nvpr="1"):
        self.log: list[str] = []
        self.delta_v = delta_v
        self.nvpr = nvpr
        self.mode = None
        self.armed = None
        self.points = 0

    def write(self, cmd):
        self.log.append(cmd)
        if cmd.endswith(":ARM") and cmd.startswith("SOUR:"):
            self.armed = cmd.split(":")[1]
        elif cmd == "INIT:IMM":
            self.mode, self.points = self.armed, 1 if self.armed == "PDEL" else len(self.delta_v)
        elif cmd in ("SOUR:SWE:ABOR", "TRAC:CLE"):
            self.armed, self.points = None, 0 if cmd == "TRAC:CLE" else self.points

    def ask(self, cmd):
        self.log.append(cmd)
        if cmd == "SOUR:PDEL:NVPR?":
            return self.nvpr
        if cmd.endswith(":ARM?"):
            return "1" if self.armed else "0"
        if cmd == "TRAC:POIN:ACT?":
            return str(self.points)
        if cmd == "TRAC:DATA?":
            assert self.mode == "DELT", "the Pulse Delta reading must never be fetched"
            return ",".join(f"{v:g}" for v in self.delta_v)
        raise AssertionError(cmd)

    def disable_source(self):
        self.log.append("OUTP OFF")


def _run(src, currents=(2e-3, -2e-3), **kw):
    read_cfg = pdl.ReadConfig(sense_current_A=1e-4, delay_after_pulse_s=0.0, n_delta_readings=3)
    return pdl.run_measurement(src, PulseDeltaConfig(width_s=1e-3), read_cfg,
                               [PulsePoint(c) for c in currents], write_csv=_NULL_WRITER, **kw)


def test_pulse_then_delta_read_per_point():
    src = _Fake6221()
    df = _run(src)
    assert list(df["pulse_current_A"]) == [2e-3, -2e-3]
    assert df["hall_resistance_ohm"].tolist() == pytest.approx([10.0, 10.0])   # mean 1 mV / 100 µA
    assert df["pulse_completed"].all()
    # per point: Pulse Delta setup → arm → INIT → abort, then Delta setup → arm → INIT → data
    inits = [i for i, c in enumerate(src.log) if c == "INIT:IMM"]
    assert len(inits) == 4
    for k, current in ((0, 2e-3), (2, -2e-3)):
        highs = [c for c in src.log[:inits[k]] if c.startswith("SOUR:PDEL:HIGH")]
        assert highs[-1] == f"SOUR:PDEL:HIGH {current:g}"          # signed write pulse
        read_block = src.log[inits[k]:inits[k + 1]]
        assert "SOUR:SWE:ABOR" in read_block                       # mode switch
        assert "SOUR:DELT:HIGH 0.0001" in read_block
    assert "SOUR:PDEL:LOW 0" in src.log and "SOUR:PDEL:COUN 1" in src.log
    assert src.log[-2:] == ["SOUR:SWE:ABOR", "OUTP OFF"]       # left quiet


def test_abort_before_first_pulse_leaves_source_quiet():
    src = _Fake6221()
    stop = threading.Event()
    stop.set()
    df = _run(src, stop_event=stop)
    assert df.empty
    assert not any(c.startswith("SOUR:PDEL:HIGH") for c in src.log)
    assert src.log[-2:] == ["SOUR:SWE:ABOR", "OUTP OFF"]


def test_missing_2182a_is_refused():
    with pytest.raises(RuntimeError, match="RS-232"):
        check_2182a_on_serial(_Fake6221(nvpr="0"))


def test_pulse_current_over_range_refused_before_hardware():
    src = _Fake6221()
    with pytest.raises(ValueError):
        _run(src, currents=(0.2,))
    assert src.log == []
