"""
The SOT run loops (sot_switching / sot_dc_characterization / sot_pulsed_switching
run_measurement) — row shape, repeat tagging, the read-mode branch, the
R = V/I guard at zero current, and (pulsed) the SMU1-parked-before-the-pulse
ordering. Hardware-free: one fake KXCI transport answers every command.
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

    ``.writes`` is the ordered log of every command AND query, which is what
    the pulsed ordering test asserts against."""

    def __init__(self):
        self.writes: list[str] = []
        self._n = itertools.count(1)

    def command(self, cmd: str) -> None:
        self.writes.append(cmd)

    def query(self, cmd: str) -> str:
        self.writes.append(cmd)
        if cmd.startswith("EX"):
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
# The check that matters: per cycle SMU1 is parked at 0 A BEFORE the PMU
# pulse and the Hall read happens AFTER. Everything now goes through one
# KXCI handle, so the ordering is asserted on the command strings.

def _pulsed_cfgs(**seq_overrides):
    read = ps.ReadConfig(read_current_A=1e-4, n_reversals=2, source_delay_s=0.0,
                         settle_before_read_s=0.0, reversal_enabled=True)
    seq = ps.PulseSequenceConfig(delay_after_pulse_s=0.0, n_repeats=1, output_file="",
                                 **seq_overrides)
    pmu = PMUPulseConfig(library="lib", module="m", return_names=())
    return pmu, read, seq


def _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg, points, **kw):
    return ps.run_measurement(dev, pmu_cfg, _src(), _sense(), read_cfg, seq_cfg,
                              points, write_csv=_NULL_WRITER, **kw)


def test_pulsed_ordering_smu_parked_then_pulse_then_read():
    dev = _FakeKXCI()
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs(reset_enabled=True, reset_amplitude_V=-2.0)

    _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg, [ps.AmplitudePoint(amplitude_V=0.5)])

    w = dev.writes
    is_park = lambda c: c.startswith("DI1,") and "0.000000E+00" in c
    i_park = next(i for i, c in enumerate(w) if is_park(c))
    i_ex = next(i for i, c in enumerate(w) if c.startswith("EX "))
    i_read = next(i for i, c in enumerate(w) if c.startswith("TV2"))
    assert i_park < i_ex < i_read

    # channel left quiet: SMU1 back to 0 A after the last Hall read
    last_read = max(i for i, c in enumerate(w) if c.startswith("TV2"))
    assert any(is_park(c) for c in w[last_read:])

    # reset + write = two EX commands, reset first and of opposite polarity
    ex_cmds = [c for c in w if c.startswith("EX ")]
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
    # the derived 2-wire resistance
    assert all(r["pulse_voltage_measured_V"] is None for r in seen)
    assert all(r["pulse_current_measured_A"] is None for r in seen)
    assert all(r["pulse_2wire_resistance_ohm"] is None for r in seen)
    assert all(r["read_current_A"] == 1e-4 for r in seen)
    assert all(not math.isnan(r["channel_voltage_V"]) for r in seen)
    assert all(r["magnet_current_A"] == 1.5 for r in seen)
    assert all(r["field_angle_from_oop_deg"] == 85.0 for r in seen)
    assert all(not math.isnan(r["hall_resistance_ohm"]) for r in seen)


def test_pulsed_derives_2wire_resistance_when_module_returns_values():
    """A module that reports spot means fills pulse_2wire_resistance_ohm."""
    class _MeasuringKXCI(_FakeKXCI):
        def query(self, cmd):
            self.writes.append(cmd)
            if cmd.startswith("EX"):
                return "OK"
            if cmd == "GN":
                # V then I, matching return_names order below
                return "2.0" if self.writes.count("GN") == 1 else "5.0E-3"
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


def test_pulsed_non_reversal_read_leaves_even_columns_blank():
    dev = _FakeKXCI()
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs()
    read_cfg.reversal_enabled = False
    seen: list[dict] = []

    _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg,
                [ps.AmplitudePoint(amplitude_V=0.5)], on_point=seen.append)

    assert seen[0]["hall_voltage_even_V"] is None
    assert seen[0]["hall_voltage_even_sem_V"] is None
    assert seen[0]["n_reversals"] == 0
    assert not math.isnan(seen[0]["hall_resistance_ohm"])


def test_pulsed_stop_event_breaks_early():
    dev = _FakeKXCI()
    pmu_cfg, read_cfg, seq_cfg = _pulsed_cfgs()
    seq_cfg.n_repeats = 5
    points = [ps.AmplitudePoint(amplitude_V=0.5)]

    class _Stop:
        def __init__(self): self.n = 0
        def is_set(self): self.n += 1; return self.n > 6

    df = _run_pulsed(dev, pmu_cfg, read_cfg, seq_cfg, points, stop_event=_Stop())
    assert 0 < len(df) < 5
