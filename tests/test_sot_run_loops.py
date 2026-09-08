"""
The two SOT run loops (sot_switching / sot_dc_characterization
run_measurement) — row shape, repeat tagging, the read-mode branch, and the
R = V/I guard at zero current. Hardware-free: a fake KXCI transport serves an
incrementing reading stream.
"""

from __future__ import annotations

import math
import itertools

import sot.sot_dc_characterization as dcc
import sot.sot_switching as sw
from instruments.keithley4200a import SMUChannelConfig


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
