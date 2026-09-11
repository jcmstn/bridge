"""
sot_pulsed_switching_2h.run_measurement — row shape, the 6221-AC-off-before-
the-pulse ordering, the pulse-failure abort, and the read safety guards.
Hardware-free: a fake KXCI transport plus fake 6221 (AC/WAVE) + MFLI daq.

Mirrors tests/test_sot_run_loops.py; see that file for the ordering
invariant this is copied from — only the read side (DC reversal → AC/2f)
differs.
"""

from __future__ import annotations

import math
import itertools

import numpy as np

import sot.sot_pulsed_switching_2h as ps
from instruments.keithley4200a import PMUPulseConfig


class _FakeKXCI:
    """See tests/test_sot_run_loops.py::_FakeKXCI — identical behaviour."""

    def __init__(self, events=None):
        self.writes: list[str] = []
        self.events = events
        self._n = itertools.count(1)

    def command(self, cmd: str) -> None:
        self.writes.append(cmd)

    def query(self, cmd: str) -> str:
        self.writes.append(cmd)
        if cmd.startswith("EX"):
            if self.events is not None:
                self.events.append("EX")
            return "OK"
        return f"{next(self._n) * 1e-4:.6E}"


_NULL_WRITER = lambda recs: None   # keep run_measurement's fallback to_csv() off disk


# ── sot_pulsed_switching_2h.run_measurement ──────────────────────────────
# The 4200A only pulses; a 6221 (AC/WAVE) + MFLI pair does the delayed 1f/2f
# read. The check that matters: per cycle the 6221 wave is stopped while the
# PMU pulses (shared channel pin), and the MFLI read happens AFTER the wave
# resumes. The PMU and the 6221/MFLI are separate handles, so all three
# append to one shared `events` list and the ordering is asserted on that.

class _Fake6221AC:
    def __init__(self, events=None):
        self.events = events

    def _log(self, tag):
        if self.events is not None:
            self.events.append(tag)

    def waveform_abort(self):  self._log("6221.abort")
    def disable_source(self):  self._log("6221.disable")
    def enable_source(self):   self._log("6221.enable")
    def waveform_arm(self):    self._log("6221.arm")
    def waveform_start(self):  self._log("6221.start")


class _FakeDAQ:
    """Enough of zi.ziDAQServer for acquire_averaged() / wait_for_reference_lock()
    — always reports locked, and returns a small distinct (x, y) sample per
    poll() so demod1/demod2 reads land in a fresh CSV row without a NaN."""

    def __init__(self, events=None):
        self.events = events
        self._n = itertools.count(1)
        self._last_path = None

    def _log(self, tag):
        if self.events is not None:
            self.events.append(tag)

    def getDouble(self, path: str) -> float:
        return 977.0

    def getInt(self, path: str) -> int:
        return 1 if path.endswith("/locked") else 0

    def setDouble(self, path, value): pass
    def setInt(self, path, value): pass
    def sync(self): pass

    def subscribe(self, path: str) -> None:
        self._last_path = path

    def unsubscribe(self, path: str) -> None:
        pass

    def poll(self, duration_s, timeout_ms, flat=True):
        self._log("mfli.read")
        v = next(self._n) * 1e-4
        return {self._last_path: {"x": np.array([v]), "y": np.array([0.0])}}


def _pulsed_cfgs():
    read = ps.ReadConfig(sense_current_A=1e-4, n_averages=2,
                         settle_after_enable_s=0.0, lock_timeout_s=1.0,
                         delay_after_pulse_s=0.0)
    pmu = PMUPulseConfig(library="lib", module="m", return_names=())
    demod1 = ps.DemodConfig(device="dev1234", demod_index=0, harmonic=1)
    demod2 = ps.DemodConfig(device="dev1234", demod_index=1, harmonic=2)
    extref = ps.ExtRefConfig(device="dev1234")
    return pmu, read, demod1, demod2, extref


def _run_pulsed(dev, pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg,
                points, *, events=None, **kw):
    source = _Fake6221AC(events)
    daq = _FakeDAQ(events)
    return ps.run_measurement(dev, pmu_cfg, source, daq, demod1_cfg, demod2_cfg,
                              extref_cfg, read_cfg, points, write_csv=_NULL_WRITER, **kw)


def test_pulsed_ordering_6221_off_then_pulse_then_read():
    dev = _FakeKXCI(events=(events := []))
    pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg = _pulsed_cfgs()

    _run_pulsed(dev, pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg,
                [ps.AmplitudePoint(amplitude_V=0.5)], events=events)

    i_first_abort = next(i for i, e in enumerate(events) if e == "6221.abort")
    i_first_ex = next(i for i, e in enumerate(events) if e == "EX")
    i_first_start = next(i for i, e in enumerate(events) if e == "6221.start")
    i_first_read = next(i for i, e in enumerate(events) if e == "mfli.read")
    # 6221 wave stopped before the pulse; pulse fires before the wave resumes;
    # the MFLI read is after that.
    assert i_first_abort < i_first_ex < i_first_start < i_first_read

    # channel left quiet: a 6221.abort after the last MFLI read
    last_read = max(i for i, e in enumerate(events) if e == "mfli.read")
    assert any(e == "6221.abort" for e in events[last_read:])

    # one pulse per amplitude — a single EX, no reset
    ex_cmds = [c for c in dev.writes if c.startswith("EX ")]
    assert len(ex_cmds) == 1
    assert "5.000000E-01" in ex_cmds[0]


def test_pulsed_one_row_per_amplitude_blank_pulse_columns():
    dev = _FakeKXCI()
    pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg = _pulsed_cfgs()
    points = [ps.AmplitudePoint(amplitude_V=a) for a in (0.2, 0.4, 0.6, 0.4, 0.2)]
    seen: list[dict] = []

    df = _run_pulsed(dev, pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg,
                     points, on_point=seen.append,
                     magnet_current_A=1.5, field_angle_from_oop_deg=85.0)

    assert len(df) == 5
    assert [r["amplitude_index"] for r in seen] == [0, 1, 2, 3, 4]
    # return_names empty → every measured-pulse column stays blank
    assert all(r["pulse_voltage_measured_V"] is None for r in seen)
    assert all(r["pulse_current_measured_A"] is None for r in seen)
    assert all(r["pulse_2wire_resistance_ohm"] is None for r in seen)
    assert all(r["pulse_base_voltage_V"] is None for r in seen)
    assert all(r["magnet_current_A"] == 1.5 for r in seen)
    assert all(r["field_angle_from_oop_deg"] == 85.0 for r in seen)
    assert all(not math.isnan(r["1f_R_V"]) for r in seen)
    assert all(not math.isnan(r["2f_R_V"]) for r in seen)
    assert all(r["reference_locked"] is True for r in seen)
    assert all(r["excitation_current_A_peak"] == 1e-4 for r in seen)


def test_pulsed_derives_2wire_resistance_when_module_returns_values():
    """A module that reports spot means fills pulse_2wire_resistance_ohm."""
    class _MeasuringKXCI(_FakeKXCI):
        def query(self, cmd):
            self.writes.append(cmd)
            if cmd.startswith("EX"):
                return "OK"
            if cmd == "GP 17":     # first output param = measured pulse V
                return "2.0"
            if cmd == "GP 18":     # second = measured pulse I
                return "5.0E-3"
            return f"{next(self._n) * 1e-4:.6E}"

    dev = _MeasuringKXCI()
    pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg = _pulsed_cfgs()
    pmu_cfg.return_names = ("pulse_voltage_measured_V", "pulse_current_measured_A")
    seen: list[dict] = []

    _run_pulsed(dev, pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg,
                [ps.AmplitudePoint(amplitude_V=0.5)], on_point=seen.append)

    assert seen[0]["pulse_voltage_measured_V"] == 2.0
    assert seen[0]["pulse_current_measured_A"] == 5.0e-3
    assert seen[0]["pulse_2wire_resistance_ohm"] == 400.0


def test_pulsed_stop_event_breaks_early():
    dev = _FakeKXCI()
    pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg = _pulsed_cfgs()
    points = [ps.AmplitudePoint(amplitude_V=a) for a in (0.2, 0.4, 0.6, 0.8, 1.0)]

    class _Stop:
        def __init__(self): self.n = 0
        def is_set(self): self.n += 1; return self.n > 3

    df = _run_pulsed(dev, pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg,
                     points, stop_event=_Stop())
    assert 0 < len(df) < 5


def test_pulsed_aborts_after_repeated_pulse_failures():
    """A non-zero module return (here -826) three amplitudes running raises
    instead of filling the whole sweep with read-only noise."""
    import pytest

    class _FailingKXCI(_FakeKXCI):
        def query(self, cmd):
            self.writes.append(cmd)
            if cmd.startswith("EX"):
                return "-826"
            return f"{next(self._n) * 1e-4:.6E}"

    dev = _FailingKXCI()
    pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg = _pulsed_cfgs()
    points = [ps.AmplitudePoint(amplitude_V=a) for a in (0.2, 0.4, 0.6, 0.8, 1.0)]
    seen: list[dict] = []
    with pytest.raises(RuntimeError, match="consecutive pulse failures"):
        _run_pulsed(dev, pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg,
                   points, on_point=seen.append)
    assert len(seen) == ps._MAX_CONSECUTIVE_PULSE_FAILURES - 1   # 2 rows, 3rd amplitude raises


def test_pulsed_refuses_absurd_read_current_or_compliance():
    """The mistyped-exponent guard: run_measurement raises before touching
    hardware, so the standalone main() path is covered, not just the TUI."""
    import pytest
    dev = _FakeKXCI()
    pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg = _pulsed_cfgs()
    pt = [ps.AmplitudePoint(amplitude_V=0.5)]

    read_cfg.sense_current_A = 0.1          # 100 mA — 1e-4 with a slipped exponent
    with pytest.raises(ValueError):
        _run_pulsed(dev, pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg, pt)

    _, read_cfg, demod1_cfg, demod2_cfg, extref_cfg = _pulsed_cfgs()
    read_cfg.compliance_V = 100.0
    with pytest.raises(ValueError):
        _run_pulsed(dev, pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg, pt)


def test_pulsed_lock_timeout_tags_row_instead_of_aborting():
    """A PLL that never reports locked degrades the row (reference_locked=False),
    logged, not fatal — see wait_for_reference_lock()'s docstring."""
    class _NeverLockedDAQ(_FakeDAQ):
        def getInt(self, path: str) -> int:
            return 0   # never locked, regardless of node

    dev = _FakeKXCI()
    pmu_cfg, read_cfg, demod1_cfg, demod2_cfg, extref_cfg = _pulsed_cfgs()
    read_cfg.lock_timeout_s = 0.0   # don't actually block the test on a real timeout
    seen: list[dict] = []

    source = _Fake6221AC()
    daq = _NeverLockedDAQ()
    ps.run_measurement(dev, pmu_cfg, source, daq, demod1_cfg, demod2_cfg, extref_cfg,
                       read_cfg, [ps.AmplitudePoint(amplitude_V=0.5)],
                       on_point=seen.append, write_csv=_NULL_WRITER)

    assert seen[0]["reference_locked"] is False
    assert not math.isnan(seen[0]["2f_R_V"])   # still reads and records — just tagged
