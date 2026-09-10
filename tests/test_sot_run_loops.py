"""
The SOT run loops (sot_switching / sot_dc_characterization / sot_pulsed_switching
run_measurement) — row shape, repeat tagging, the read-mode branch, the
R = V/I guard at zero current, and (pulsed) the 6221-off-before-the-pulse
ordering. Hardware-free: a fake KXCI transport plus fake 6221/2182.
"""

from __future__ import annotations

import math
import itertools

import sot.sot_dc_characterization as dcc
import sot.sot_switching as sw
import sot.sot_pulsed_switching as ps
from instruments.keithley4200a import PMUPulseConfig, SMUChannelConfig


class _FakeKXCI:
    """One handle for the whole 4200A: ``EX`` (the KULT pulse module) answers
    "OK", every other query (TV/TI) returns the next value of an incrementing
    ramp so reversal pairs differ and odd parts are non-zero.

    ``.writes`` is the ordered log of every command AND query. ``events``, if
    given, is a shared timeline list the pulsed ordering test also feeds the
    fake 6221/2182 into — an ``"EX"`` marker is appended there on each pulse."""

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


def _src():
    return SMUChannelConfig(channel=1, source_function="current", source_limit_A=1.0)


def _sense():
    return SMUChannelConfig(channel=2, source_function="current", source_limit_A=1.0)


# ── sot_dc_characterization.run_measurement ────────────────────────────────

def test_dcchar_rows_repeats_and_zero_current_guard():
    dev = _FakeKXCI()
    acq = dcc.AcquisitionConfig(reversal_enabled=True, n_averages=3, n_repeats=2,
                                settling_time_s=0.0)
    points = [dcc.CurrentPoint(current_A=c) for c in (-1e-4, 0.0, 1e-4)]
    seen: list[dict] = []
    writes: list[int] = []

    df = dcc.run_measurement(dev, _src(), _sense(), acq, points,
                             on_point=seen.append,
                             write_csv=lambda recs: writes.append(len(recs)))

    assert len(df) == 6                                   # 3 points × 2 repeats
    assert [r["repeat_index"] for r in seen] == [0, 0, 0, 1, 1, 1]
    assert [r["point_index"] for r in seen] == [0, 1, 2, 0, 1, 2]
    assert len(seen) == 6 and writes == [1, 2, 3, 4, 5, 6]   # full rewrite each row
    zero_rows = [r for r in seen if r["set_current_A"] == 0.0]
    assert zero_rows and all(math.isnan(r["resistance_ohm"]) for r in zero_rows)
    assert all(not math.isnan(r["resistance_ohm"]) for r in seen if r["set_current_A"] != 0.0)


def test_dcchar_stop_event_breaks_early():
    dev = _FakeKXCI()
    acq = dcc.AcquisitionConfig(reversal_enabled=False, n_averages=1, n_repeats=3,
                                settling_time_s=0.0)
    points = [dcc.CurrentPoint(current_A=1e-4) for _ in range(4)]

    class _Stop:
        def __init__(self): self.n = 0
        def is_set(self): self.n += 1; return self.n > 5

    df = dcc.run_measurement(dev, _src(), _sense(), acq, points, stop_event=_Stop(),
                             write_csv=_NULL_WRITER)
    assert 0 < len(df) < 12


# ── sot_switching.run_measurement ─────────────────────────────────────────

def test_switching_at_write_current_zero_is_nan():
    dev = _FakeKXCI()
    acq = sw.AcquisitionConfig(read_mode="at_write_current", n_averages=2, n_repeats=2,
                               settling_time_s=0.0)
    points = [sw.CurrentPoint(current_A=c) for c in (-1e-3, 0.0, 1e-3)]
    seen: list[dict] = []

    df = sw.run_measurement(dev, _src(), _sense(), acq, points, on_point=seen.append,
                            magnet_current_A=1.5, field_angle_from_oop_deg=85.0,
                            write_csv=_NULL_WRITER)

    assert len(df) == 6
    assert [r["repeat_index"] for r in seen] == [0, 0, 0, 1, 1, 1]
    assert all(r["magnet_current_A"] == 1.5 for r in seen)
    assert all(r["field_angle_from_oop_deg"] == 85.0 for r in seen)
    assert all(r["read_current_A"] == r["set_current_A"] for r in seen)   # read AT write current
    zero = [r for r in seen if r["set_current_A"] == 0.0]
    assert zero and all(math.isnan(r["hall_resistance_ohm"]) for r in zero)


def test_switching_write_then_read_uses_read_current_at_zero():
    dev = _FakeKXCI()
    acq = sw.AcquisitionConfig(read_mode="write_then_read", read_current_A=5e-5,
                               read_reversal=False, n_averages=2, n_repeats=1,
                               settling_time_s=0.0)
    points = [sw.CurrentPoint(current_A=c) for c in (-1e-3, 0.0, 1e-3)]
    seen: list[dict] = []

    sw.run_measurement(dev, _src(), _sense(), acq, points, on_point=seen.append,
                       write_csv=_NULL_WRITER)

    assert all(r["read_current_A"] == 5e-5 for r in seen)
    # even the zero-write-current row has a finite R_xy (read current is non-zero)
    zero = [r for r in seen if r["set_current_A"] == 0.0]
    assert zero and all(not math.isnan(r["hall_resistance_ohm"]) for r in zero)


def test_switching_write_then_read_reversal_records_even_component():
    dev = _FakeKXCI()
    acq = sw.AcquisitionConfig(read_mode="write_then_read", read_current_A=5e-5,
                               read_reversal=True, n_averages=3, n_repeats=1,
                               settling_time_s=0.0)
    points = [sw.CurrentPoint(current_A=1e-3)]
    seen: list[dict] = []

    sw.run_measurement(dev, _src(), _sense(), acq, points, on_point=seen.append,
                       write_csv=_NULL_WRITER)

    assert seen[0]["hall_voltage_even_V"] is not None
    assert seen[0]["hall_voltage_even_sem_V"] is not None


def test_switching_rejects_bad_read_mode():
    import pytest
    dev = _FakeKXCI()
    acq = sw.AcquisitionConfig(read_mode="nonsense")
    with pytest.raises(ValueError):
        sw.run_measurement(dev, _src(), _sense(), acq, [sw.CurrentPoint(current_A=1e-3)])


# ── sot_pulsed_switching.run_measurement ─────────────────────────────────
# The 4200A only pulses; a 6221 + 2182 pair does the delayed R_xy read. The
# check that matters: per cycle the 6221 is OFF while the PMU pulses (shared
# channel pin), and the Hall read happens AFTER. The PMU and the 6221/2182
# are separate handles, so all three append to one shared `events` list and
# the ordering is asserted on that.

class _Fake6221:
    def __init__(self, events=None):
        self.events = events
        self.source_current = 0.0

    def _log(self, tag):
        if self.events is not None:
            self.events.append(tag)

    def enable_source(self):  self._log("6221.enable")
    def disable_source(self): self._log("6221.disable")
    def shutdown(self):       self._log("6221.shutdown")


class _Fake2182:
    def __init__(self, events=None):
        self.events = events
        self._n = itertools.count(1)

    @property
    def voltage(self) -> float:
        if self.events is not None:
            self.events.append("2182.read")
        return next(self._n) * 1e-4


def _pulsed_cfgs(**seq_overrides):
    read = ps.ReadConfig(sense_current_A=1e-4, n_reversals=2, source_delay_s=0.0,
                         settle_after_enable_s=0.0)
    seq = ps.PulseSequenceConfig(delay_after_pulse_s=0.0, n_repeats=1, output_file="",
                                 **seq_overrides)
    pmu = PMUPulseConfig(library="lib", module="m", return_names=())
    return pmu, read, seq


def _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg, points, *, events=None, **kw):
    source = _Fake6221(events)
    voltmeter = _Fake2182(events)
    return ps.run_measurement(dev, pmu_cfg, source, voltmeter, read_cfg, seq_cfg,
                              points, write_csv=_NULL_WRITER, **kw)


def test_pulsed_ordering_6221_off_then_pulse_then_read():
    dev = _FakeKXCI(events=(events := []))
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs(reset_enabled=True, reset_amplitude_V=-2.0)

    _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg,
                [ps.AmplitudePoint(amplitude_V=0.5)], events=events)

    i_first_disable = next(i for i, e in enumerate(events) if e == "6221.disable")
    i_first_ex = next(i for i, e in enumerate(events) if e == "EX")
    i_first_enable = next(i for i, e in enumerate(events) if e == "6221.enable")
    i_first_read = next(i for i, e in enumerate(events) if e == "2182.read")
    # 6221 disabled before the pulse; pulse fires before the 6221 is re-enabled;
    # the Hall read is after that.
    assert i_first_disable < i_first_ex < i_first_enable < i_first_read

    # channel left quiet: a 6221.disable after the last 2182 read
    last_read = max(i for i, e in enumerate(events) if e == "2182.read")
    assert any(e == "6221.disable" for e in events[last_read:])

    # reset + write = two EX queries, reset first and of opposite polarity
    ex_cmds = [c for c in dev.writes if c.startswith("EX ")]
    assert len(ex_cmds) == 2
    assert "-2.000000E+00" in ex_cmds[0] and "5.000000E-01" in ex_cmds[1]


def test_pulsed_rows_repeats_and_blank_pulse_columns():
    dev = _FakeKXCI()
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs()
    seq_cfg.n_repeats = 3
    points = [ps.AmplitudePoint(amplitude_V=a) for a in (0.4, 0.8)]
    seen: list[dict] = []

    df = _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg, points, on_point=seen.append,
                     magnet_current_A=1.5, field_angle_from_oop_deg=85.0)

    assert len(df) == 6
    assert [r["amplitude_index"] for r in seen] == [0, 0, 0, 1, 1, 1]
    assert [r["repeat_index"] for r in seen] == [0, 1, 2, 0, 1, 2]
    # return_names empty → every measured-pulse column stays blank, including
    # the derived 2-wire resistance and the base-level pair
    assert all(r["pulse_voltage_measured_V"] is None for r in seen)
    assert all(r["pulse_current_measured_A"] is None for r in seen)
    assert all(r["pulse_2wire_resistance_ohm"] is None for r in seen)
    assert all(r["pulse_base_voltage_V"] is None for r in seen)
    assert all(r["sense_current_A"] == 1e-4 for r in seen)
    assert all(r["magnet_current_A"] == 1.5 for r in seen)
    assert all(r["field_angle_from_oop_deg"] == 85.0 for r in seen)
    assert all(not math.isnan(r["hall_resistance_ohm"]) for r in seen)
    # the 6221/2182 read is always reversal-averaged → even columns populated
    assert all(r["hall_voltage_even_V"] is not None for r in seen)
    assert all(r["n_reversals"] == 2 for r in seen)


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
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs()
    pmu_cfg.return_names = ("pulse_voltage_measured_V", "pulse_current_measured_A")
    seen: list[dict] = []

    _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg,
                [ps.AmplitudePoint(amplitude_V=0.5)], on_point=seen.append)

    assert seen[0]["pulse_voltage_measured_V"] == 2.0
    assert seen[0]["pulse_current_measured_A"] == 5.0e-3
    assert seen[0]["pulse_2wire_resistance_ohm"] == 400.0


def test_pulsed_stop_event_breaks_early():
    dev = _FakeKXCI()
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs()
    seq_cfg.n_repeats = 5
    points = [ps.AmplitudePoint(amplitude_V=0.5)]

    class _Stop:
        def __init__(self): self.n = 0
        def is_set(self): self.n += 1; return self.n > 3

    df = _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg, points, stop_event=_Stop())
    assert 0 < len(df) < 5


def test_pulsed_aborts_after_repeated_pulse_failures():
    """A non-zero module return (here -826) three cycles running raises instead
    of filling the whole sweep with read-only noise."""
    import pytest

    class _FailingKXCI(_FakeKXCI):
        def query(self, cmd):
            self.writes.append(cmd)
            if cmd.startswith("EX"):
                return "-826"
            return f"{next(self._n) * 1e-4:.6E}"

    dev = _FailingKXCI()
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs()
    seq_cfg.n_repeats = 10
    seen: list[dict] = []
    with pytest.raises(RuntimeError, match="consecutive pulse failures"):
        _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg,
                    [ps.AmplitudePoint(amplitude_V=0.5)], on_point=seen.append)
    assert len(seen) == ps._MAX_CONSECUTIVE_PULSE_FAILURES - 1   # 2 rows, 3rd cycle raises


def test_pulsed_refuses_absurd_read_current_or_compliance():
    """The mistyped-exponent guard: run_measurement raises before touching
    hardware, so the standalone main() path is covered, not just the TUI."""
    import pytest
    dev = _FakeKXCI()
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs()
    pt = [ps.AmplitudePoint(amplitude_V=0.5)]

    read_cfg.sense_current_A = 0.1          # 100 mA — 1e-4 with a slipped exponent
    with pytest.raises(ValueError):
        _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg, pt)

    _, read_cfg, _ = _pulsed_cfgs()
    read_cfg.compliance_V = 100.0
    with pytest.raises(ValueError):
        _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg, pt)
