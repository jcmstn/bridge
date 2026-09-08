"""
The SOT run loops (sot_switching / sot_dc_characterization / sot_pulsed_switching
run_measurement) — row shape, repeat tagging, the read-mode branch, the
R = V/I guard at zero current, and (pulsed) the 6221-off-during-pulse
ordering. Hardware-free: fake transports serve incrementing reading streams.
"""

from __future__ import annotations

import math
import itertools

import sot.sot_dc_characterization as dcc
import sot.sot_switching as sw
import sot.sot_pulsed_switching as ps
from instruments.keithley4200a import PMUPulseConfig, SMUChannelConfig


class _FakeKXCI:
    """Every TV/TI query returns the next value of an incrementing ramp (so
    reversal pairs differ and odd parts are non-zero); force commands are
    recorded."""

    def __init__(self):
        self.writes: list[str] = []
        self._n = itertools.count(1)

    def command(self, cmd: str) -> None:
        self.writes.append(cmd)

    def query(self, cmd: str) -> str:
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
# The check that matters (advisor): per cycle the 6221 output is disabled
# BEFORE the PMU pulse and the read happens AFTER — asserted via a shared
# call log across three fake instruments.

class _FakePMU:
    def __init__(self, log): self.log = log
    def query(self, cmd):
        self.log.append(("pmu", cmd))
        return "OK" if cmd.startswith("EX") else "0"
    def command(self, cmd):
        self.log.append(("pmu", cmd))


class _Fake6221:
    def __init__(self, log):
        self.log = log
        self._i = 0.0
    @property
    def source_current(self): return self._i
    @source_current.setter
    def source_current(self, v):
        self._i = v
        self.log.append(("6221", f"current={v:g}"))
    def enable_source(self): self.log.append(("6221", "enable"))
    def disable_source(self): self.log.append(("6221", "disable"))


class _Fake2182:
    def __init__(self, log):
        self.log = log
        self._n = 0
    @property
    def voltage(self):
        self._n += 1
        self.log.append(("2182", "read"))
        return 1e-4 * self._n


def _pulsed_cfgs(**seq_overrides):
    read = ps.ReadConfig(sense_current_A=1e-4, n_reversals=2, source_delay_s=0.0,
                         settle_after_enable_s=0.0)
    seq = ps.PulseSequenceConfig(delay_after_pulse_s=0.0, n_repeats=1, output_file="",
                                 **seq_overrides)
    pmu = PMUPulseConfig(library="lib", module="m")
    return pmu, read, seq


def test_pulsed_ordering_6221_off_before_pulse_read_after():
    log: list = []
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs(reset_enabled=True, reset_amplitude_V=-2.0)
    points = [ps.AmplitudePoint(amplitude_V=0.5)]

    ps.run_measurement(_FakePMU(log), pmu_cfg, _Fake6221(log), _Fake2182(log),
                       read_cfg, seq_cfg, points, write_csv=lambda r: None)

    kinds = [f"{d}:{a}" for d, a in log]
    i_disable = next(i for i, x in enumerate(kinds) if x == "6221:disable")
    i_ex = next(i for i, (d, a) in enumerate(log) if d == "pmu" and a.startswith("EX"))
    i_enable = next(i for i, x in enumerate(kinds) if x == "6221:enable")
    i_read = next(i for i, x in enumerate(kinds) if x == "2182:read")
    assert i_disable < i_ex < i_enable < i_read
    # channel left quiet: a disable after the last read
    last_read = max(i for i, x in enumerate(kinds) if x == "2182:read")
    assert any(x == "6221:disable" for x in kinds[last_read:])
    # reset + write = two EX commands, reset first (opposite polarity)
    ex_cmds = [a for d, a in log if d == "pmu" and a.startswith("EX")]
    assert len(ex_cmds) == 2
    assert "-2.000000E+00" in ex_cmds[0] and "5.000000E-01" in ex_cmds[1]


def test_pulsed_rows_repeats_and_blank_pulse_columns():
    log: list = []
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs()
    seq_cfg.n_repeats = 3
    points = [ps.AmplitudePoint(amplitude_V=a) for a in (0.4, 0.8)]
    seen: list[dict] = []

    df = ps.run_measurement(_FakePMU(log), pmu_cfg, _Fake6221(log), _Fake2182(log),
                            read_cfg, seq_cfg, points, on_point=seen.append,
                            magnet_current_A=1.5, field_angle_from_oop_deg=85.0,
                            write_csv=lambda r: None)

    assert len(df) == 6
    assert [r["amplitude_index"] for r in seen] == [0, 0, 0, 1, 1, 1]
    assert [r["repeat_index"] for r in seen] == [0, 1, 2, 0, 1, 2]
    # return_names empty → measured pulse columns stay blank
    assert all(r["pulse_voltage_measured_V"] is None for r in seen)
    assert all(r["pulse_current_measured_A"] is None for r in seen)
    assert all(r["magnet_current_A"] == 1.5 for r in seen)
    assert all(not math.isnan(r["hall_resistance_ohm"]) for r in seen)


def test_pulsed_stop_event_breaks_early():
    log: list = []
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs()
    seq_cfg.n_repeats = 5
    points = [ps.AmplitudePoint(amplitude_V=0.5)]

    class _Stop:
        def __init__(self): self.n = 0
        def is_set(self): self.n += 1; return self.n > 8

    df = ps.run_measurement(_FakePMU(log), pmu_cfg, _Fake6221(log), _Fake2182(log),
                            read_cfg, seq_cfg, points, stop_event=_Stop(),
                            write_csv=lambda r: None)
    assert 0 < len(df) < 5
