"""
instruments/keithley6221.py::fire_wave_pulse — the WAVE-mode single-cycle
square-wave pulse (real 6221 hardware capability, no 2182). Hardware-free:
a fake pymeasure-shaped Keithley6221 handle.
"""

from __future__ import annotations

import instruments.keithley6221 as k


class _FakeWaveSource:
    """Tracks every WAVE/source property write and reports OUTPUT as ON
    from arm/start until `n_polls_until_off` source_enabled reads have
    happened — simulates the instrument's own auto-off after one cycle."""

    def __init__(self, n_polls_until_off: int = 2):
        self.writes: dict = {}
        self._n_polls_until_off = n_polls_until_off
        self._polls = 0
        self._enabled = False
        self.calls: list[str] = []

    def __setattr__(self, name, value):
        if name in ("writes", "_n_polls_until_off", "_polls", "_enabled", "calls"):
            super().__setattr__(name, value)
            return
        self.writes[name] = value

    @property
    def source_enabled(self):
        if self._enabled:
            self._polls += 1
            if self._polls >= self._n_polls_until_off:
                self._enabled = False
        return self._enabled

    def waveform_arm(self):  self.calls.append("arm")
    def waveform_start(self):
        self.calls.append("start")
        self._enabled = True
    def waveform_abort(self): self.calls.append("abort")
    def disable_source(self): self.calls.append("disable")


def test_fire_wave_pulse_unipolar_square_one_cycle():
    source = _FakeWaveSource()
    cfg = k.PulseWaveConfig(pulse_current_A=10e-3, width_s=1e-3, compliance_V=4.0)

    info = k.fire_wave_pulse(source, cfg)

    # unipolar 0 -> pulse_current_A: amplitude = offset = half the pulse height
    assert source.writes["waveform_amplitude"] == 5e-3
    assert source.writes["waveform_offset"] == 5e-3
    assert source.writes["waveform_function"] == "square"
    assert source.writes["waveform_dutycycle"] == 50.0
    assert source.writes["waveform_duration_cycles"] == 1
    assert source.writes["waveform_use_phasemarker"] is False
    assert source.writes["source_compliance"] == 4.0
    # 50% duty cycle => frequency = 1 / (2 * width_s)
    assert source.writes["waveform_frequency"] == 1.0 / (2 * 1e-3)

    assert source.calls == ["arm", "start", "abort", "disable"]
    assert info["pulse_width_measured_s"] >= 0


def test_fire_wave_pulse_negative_keeps_amplitude_positive():
    source = _FakeWaveSource()
    k.fire_wave_pulse(source, k.PulseWaveConfig(pulse_current_A=-10e-3, width_s=1e-3))

    # pymeasure clips a negative amplitude to ~0, so only the offset carries the sign
    assert source.writes["waveform_amplitude"] == 5e-3
    assert source.writes["waveform_offset"] == -5e-3


def test_fire_wave_pulse_stops_on_stop_event():
    source = _FakeWaveSource(n_polls_until_off=10_000)   # would "hang" without stop_event

    class _Stop:
        def is_set(self): return True

    info = k.fire_wave_pulse(source, k.PulseWaveConfig(width_s=1e-4), stop_event=_Stop())
    assert "abort" in source.calls and "disable" in source.calls
    assert info["pulse_width_measured_s"] < 1.0   # returned promptly, not after the full timeout


def test_fire_wave_pulse_never_exceeds_timeout_bound():
    source = _FakeWaveSource(n_polls_until_off=10_000_000)   # never reports off
    cfg = k.PulseWaveConfig(width_s=1e-3)   # period = 2ms

    info = k.fire_wave_pulse(source, cfg, timeout_margin_s=0.05)
    # bounded by period + timeout_margin_s, not left spinning forever
    assert info["pulse_width_measured_s"] < (2 * 1e-3 + 0.05) + 0.05
    assert "abort" in source.calls and "disable" in source.calls
