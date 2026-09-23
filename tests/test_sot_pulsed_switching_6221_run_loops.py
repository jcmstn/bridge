"""
sot_pulsed_switching_6221.run_measurement — row shape, the wave-off/pulse/
wave-on/read ordering on a SINGLE instrument (no 4200A), and the write/read
safety guards. Hardware-free: a fake 6221 (DC pulse + AC/WAVE) + MFLI daq.

Mirrors tests/test_sot_pulsed_switching_2h_run_loops.py; the KXCI/4200A fake
is gone entirely — the 6221 now does both the pulse and the read.
"""

from __future__ import annotations

import math
import itertools

import numpy as np
import pytest

import sot.sot_pulsed_switching_6221 as ps


_NULL_WRITER = lambda recs: None   # keep run_measurement's fallback to_csv() off disk


class _Fake6221AC:
    """A software model of the 6221 running entirely in WAVE mode: the
    single-cycle square pulse fire_wave_pulse() fires, and the continuous
    sine wave _six221_ac_output_on()/_off() drive for the read. `source_
    enabled` mimics real hardware's own auto-off after one pulse cycle —
    True right after waveform_start(), then False once fire_wave_pulse()'s
    poll loop has read it `polls_until_pulse_off` times. Logs enough events
    to assert the off/pulse/on/read ordering."""

    def __init__(self, events=None, polls_until_pulse_off=2):
        self.events = events
        self._polls_until_pulse_off = polls_until_pulse_off
        self._poll_count = 0
        self._enabled = False
        self.arms: list[dict] = []

    def _log(self, tag):
        if self.events is not None:
            self.events.append(tag)

    def enable_source(self):
        self._log("6221.enable")
        self._enabled = True

    def disable_source(self):
        self._log("6221.disable")
        self._enabled = False

    def waveform_abort(self): self._log("6221.abort")

    def waveform_arm(self):
        self._log("6221.arm")
        # WAVE state at each arm -- what the instrument would actually fire.
        self.arms.append({k: getattr(self, k, None) for k in (
            "waveform_function", "waveform_amplitude", "waveform_offset",
            "waveform_use_phasemarker", "waveform_duration_cycles", "_infinite")})

    def waveform_duration_set_infinity(self):
        self._infinite = True
        self.waveform_duration_cycles = None

    def __setattr__(self, name, value):
        if name == "waveform_duration_cycles" and value is not None:
            object.__setattr__(self, "_infinite", False)
        object.__setattr__(self, name, value)

    def waveform_start(self):
        self._log("6221.start")
        self._enabled = True
        self._poll_count = 0

    @property
    def source_enabled(self):
        # Only fire_wave_pulse() polls this (right after its own
        # waveform_start()) -- simulate the instrument's own auto-off after
        # one cycle, same as real hardware per the manual.
        if self._enabled:
            self._poll_count += 1
            if self._poll_count >= self._polls_until_pulse_off:
                self._enabled = False
        return self._enabled

    # WAVE-parameter writes (source_compliance, waveform_function,
    # waveform_amplitude, waveform_offset, waveform_dutycycle,
    # waveform_frequency, waveform_ranging, waveform_use_phasemarker,
    # waveform_duration_cycles) -- plain attribute sets, nothing to simulate.


class _FakeDAQ:
    """Enough of zi.ziDAQServer for acquire_averaged() / wait_for_reference_lock()
    — always reports locked, returns a small distinct (x, y) sample per poll()."""

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


def _cfgs(**read_overrides):
    pulse = ps.WritePulseConfig(width_s=1e-3, compliance_V=5.0)
    read = ps.ReadConfig(sense_current_A=1e-4, harmonic=2, n_averages=2,
                         settle_after_enable_s=0.0, lock_timeout_s=1.0,
                         delay_after_pulse_s=0.0)
    for k, v in read_overrides.items():
        setattr(read, k, v)
    demod = ps.DemodConfig(device="dev1234", demod_index=1, harmonic=read.harmonic)
    extref = ps.ExtRefConfig(device="dev1234", pll_demod_index=0)   # ≠ the signal demod 1
    return pulse, read, demod, extref


def _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, points, *, events=None, **kw):
    source = _Fake6221AC(events)
    daq = _FakeDAQ(events)
    return ps.run_measurement(source, daq, demod_cfg, extref_cfg, pulse_cfg, read_cfg,
                              points, write_csv=_NULL_WRITER, **kw)


def test_pulsed_ordering_wave_off_pulse_wave_on_read():
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    events: list[str] = []

    _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg,
        [ps.PulsePoint(pulse_current_A=5e-3)], events=events)

    # Exact sequence: wave stopped -> pulse fires and auto-completes
    # (fire_wave_pulse's own abort+disable) -> AC read resumes -> MFLI read
    # -> wave stopped again (channel left quiet).
    assert events == [
        "6221.abort", "6221.disable",
        "6221.arm", "6221.start",
        "6221.abort", "6221.disable",
        "6221.enable", "6221.arm", "6221.start",
        "mfli.read",
        "6221.abort", "6221.disable",
    ]


def test_pulsed_one_row_per_amplitude():
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    points = [ps.PulsePoint(pulse_current_A=a) for a in (1e-3, 2e-3, 3e-3, 2e-3, 1e-3)]
    seen: list[dict] = []

    df = _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, points, on_point=seen.append,
             magnet_current_A=1.5, field_theta_deg=85.0, field_phi_deg=10.0)

    assert len(df) == 5
    assert [r["amplitude_index"] for r in seen] == [0, 1, 2, 3, 4]
    assert [r["pulse_current_A"] for r in seen] == [1e-3, 2e-3, 3e-3, 2e-3, 1e-3]
    assert all(r["pulse_width_measured_s"] > 0 for r in seen)
    assert all(r["harmonic"] == 2 for r in seen)
    assert all(not math.isnan(r["demod_R_V"]) for r in seen)
    assert all(r["reference_locked"] is True for r in seen)
    assert all(r["magnet_current_A"] == 1.5 for r in seen)
    assert all(r["field_theta_deg"] == 85.0 for r in seen)
    assert all(r["field_phi_deg"] == 10.0 for r in seen)


def test_pulsed_stop_event_breaks_early():
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    points = [ps.PulsePoint(pulse_current_A=a) for a in (1e-3, 2e-3, 3e-3, 4e-3, 5e-3)]

    class _Stop:
        def __init__(self): self.n = 0
        def is_set(self): self.n += 1; return self.n > 3

    df = _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, points, stop_event=_Stop())
    assert 0 < len(df) < 5


def test_lock_timeout_tags_row_instead_of_aborting():
    class _NeverLockedDAQ(_FakeDAQ):
        def getInt(self, path: str) -> int:
            return 0

    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs(lock_timeout_s=0.0)
    seen: list[dict] = []

    source = _Fake6221AC()
    daq = _NeverLockedDAQ()
    ps.run_measurement(source, daq, demod_cfg, extref_cfg, pulse_cfg, read_cfg,
                       [ps.PulsePoint(pulse_current_A=5e-3)],
                       on_point=seen.append, write_csv=_NULL_WRITER)

    assert seen[0]["reference_locked"] is False
    assert not math.isnan(seen[0]["demod_R_V"])   # still reads and records — just tagged


def test_refuses_pulse_current_over_hardware_ceiling():
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    points = [ps.PulsePoint(pulse_current_A=0.5)]   # 500 mA — past the 105 mA hardware max
    with pytest.raises(ValueError):
        _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, points)


def test_refuses_zero_pulse_current():
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    points = [ps.PulsePoint(pulse_current_A=0.0)]
    with pytest.raises(ValueError):
        _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, points)


def test_refuses_absurd_read_current_or_compliance():
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    pt = [ps.PulsePoint(pulse_current_A=5e-3)]

    read_cfg.sense_current_A = 0.1          # 100 mA — 1e-4 with a slipped exponent
    with pytest.raises(ValueError):
        _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, pt)

    _, read_cfg, demod_cfg, extref_cfg = _cfgs()
    read_cfg.compliance_V = 100.0
    with pytest.raises(ValueError):
        _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, pt)


def test_refuses_bad_pulse_width_or_compliance():
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    pt = [ps.PulsePoint(pulse_current_A=5e-3)]

    pulse_cfg.width_s = 0.0
    with pytest.raises(ValueError):
        _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, pt)

    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    pulse_cfg.compliance_V = 200.0
    with pytest.raises(ValueError):
        _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, pt)


def test_refuses_extref_demod_collision():
    """extrefs/N/adcselect is read-only on real firmware (confirmed against
    a live device) — the PLL's dedicated phase-detector demod must differ
    from the signal demod, checked before any hardware call."""
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    extref_cfg.pll_demod_index = demod_cfg.demod_index
    pt = [ps.PulsePoint(pulse_current_A=5e-3)]
    with pytest.raises(ValueError):
        _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg, pt)


def test_harmonic_is_configurable_and_recorded():
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs(harmonic=1)
    demod_cfg.harmonic = 1
    seen: list[dict] = []

    _run(pulse_cfg, read_cfg, demod_cfg, extref_cfg,
        [ps.PulsePoint(pulse_current_A=5e-3)], on_point=seen.append)

    assert seen[0]["harmonic"] == 1


def test_read_rearms_the_sine_not_the_write_pulse():
    """fire_wave_pulse() leaves WAVE set to a one-cycle square at the pulse
    current with the marker off; the read must re-write the sine + marker
    before arming, or it fires the write pulse a second time."""
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = _cfgs()
    source = _Fake6221AC()
    ps.run_measurement(source, _FakeDAQ(), demod_cfg, extref_cfg, pulse_cfg, read_cfg,
                       [ps.PulsePoint(pulse_current_A=5e-3)], write_csv=_NULL_WRITER)

    pulse_arm, read_arm = source.arms
    assert pulse_arm["waveform_function"] == "square"
    assert pulse_arm["waveform_use_phasemarker"] is False
    assert read_arm["waveform_function"] == "sine"
    assert read_arm["waveform_amplitude"] == read_cfg.sense_current_A
    assert read_arm["waveform_offset"] == 0.0
    assert read_arm["waveform_use_phasemarker"] is True
    assert read_arm["_infinite"] is True
