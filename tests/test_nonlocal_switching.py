"""
sot/nonlocal_switching.py — guards, the pure switched/AP-fraction helpers, the
field-init sequencing, and the pulse -> wait -> DC-read loop end to end against
a fake 6221 + 2182A whose magnet flips when a unipolar pulse reaches I_c.
Hardware-free.
"""

from __future__ import annotations

import itertools
import threading

import pytest

import sot.nonlocal_switching as ns

_NULL_WRITER = lambda recs: None   # keep run_measurement's fallback to_csv() off disk


class _Fake6221:
    """WAVE pulse flips `state` to +1 when its height reaches i_c. Records the
    (low, high) level of every pulse — the unipolar 0 -> +I -> 0 contract. Plain
    DC writes while a wave is armed raise, like the real instrument's +413."""

    def __init__(self, i_c=3e-3):
        self.i_c, self.state = i_c, -1
        self.armed = False
        self.enabled = False
        self.output_low_grounded = False
        self._i = 0.0
        self.waveform_offset = 0.0
        self.waveform_amplitude = 0.0
        self.pulse_levels: list[tuple[float, float]] = []

    @property
    def source_current(self):
        return self._i

    @source_current.setter
    def source_current(self, v):
        if self.armed:
            raise RuntimeError("+413 Not allowed with mode arm")
        self._i = v

    @property
    def source_enabled(self):      # the one-cycle wave is already over at the first poll
        return False

    def waveform_arm(self):   self.armed = True
    def waveform_abort(self): self.armed = False
    def enable_source(self):  self.enabled = True
    def disable_source(self): self.enabled = False

    def waveform_start(self):
        lo = self.waveform_offset - self.waveform_amplitude
        hi = self.waveform_offset + self.waveform_amplitude
        self.pulse_levels.append((lo, hi))
        if hi >= self.i_c:
            self.state = 1


class _FakeVoltmeter:
    """V = state * R0 * I + thermal offset + a little deterministic noise."""

    def __init__(self, src, r0=0.5, v_off=2e-6):
        self.src, self.r0, self.v_off = src, r0, v_off
        self._noise = itertools.cycle([1e-9, -2e-9, 3e-9, -1e-9])

    @property
    def voltage(self):
        return self.src.state * self.r0 * self.src.source_current + self.v_off + next(self._noise)


def _run(points, **read_kw):
    src = _Fake6221()
    read_cfg = ns.ReadConfig(sense_current_A=1e-4, n_reversals=3, source_delay_s=0,
                             delay_after_pulse_s=0, **read_kw)
    df = ns.run_measurement(src, _FakeVoltmeter(src), ns.WritePulseConfig(width_s=1e-4),
                            read_cfg, [ns.PulsePoint(i) for i in points], write_csv=_NULL_WRITER)
    return src, df


def test_unipolar_sweep_switches_once_and_never_goes_negative():
    src, df = _run([1e-3, 5e-3, 0.0, 8e-3], R_P_ohm=-0.5, R_AP_ohm=0.5)

    # baseline, sub-threshold, switch, read-only, above threshold (already switched)
    assert df["nl_resistance_ohm"].round(3).tolist() == [-0.5, -0.5, 0.5, 0.5, 0.5]
    assert df["switched"].tolist() == [None, False, True, False, False]
    assert df["state_AP_fraction"].round(3).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert df["pulse_current_A"].tolist() == [0.0, 1e-3, 5e-3, 0.0, 8e-3]
    assert df["pulse_width_s"].isna().tolist() == [True, False, False, True, False]
    assert ns.first_switch_current_A(df.to_dict("records")) == 5e-3
    # every pulse is exactly 0 -> +I: low level 0 A (no negative lobe), high level I
    assert src.pulse_levels == [(0.0, 1e-3), (0.0, 5e-3), (0.0, 8e-3)]
    # elapsed_s is the relaxation time axis: strictly increasing
    assert df["elapsed_s"].is_monotonic_increasing and df["elapsed_s"].iloc[0] > 0
    # V_even is the thermal-offset proxy, untouched by the switching
    assert df["voltage_even_V"].round(7).eq(2e-6).all()
    # output left off and zeroed
    assert src.enabled is False and src.source_current == 0.0 and src.armed is False


def test_reference_levels_require_half_the_swing_to_count_as_switched():
    # a step of 0.1 ohm is many sigma, but only 10% of the 1 ohm P<->AP swing
    assert ns._switched(0.1, 1e-4, 0.0, 1e-4, 5) is True
    assert ns._switched(0.1, 1e-4, 0.0, 1e-4, 5, min_delta=0.5) is False
    assert ns._switched(0.6, 1e-4, 0.0, 1e-4, 5, min_delta=0.5) is True


def test_stop_before_first_point_returns_empty_and_output_off():
    src = _Fake6221()
    stop = threading.Event()
    stop.set()
    df = ns.run_measurement(src, _FakeVoltmeter(src), ns.WritePulseConfig(),
                            ns.ReadConfig(delay_after_pulse_s=0), [ns.PulsePoint(5e-3)],
                            stop_event=stop, write_csv=_NULL_WRITER)
    assert len(df) == 0 and src.enabled is False


def test_switched_needs_history_and_a_usable_sem():
    assert ns._switched(1.0, 0.01, None, None, 5) is None
    assert ns._switched(1.0, float("nan"), 0.0, 0.01, 5) is None
    assert ns._switched(1.0, 0.01, 0.0, 0.01, 5) is True
    assert ns._switched(0.02, 0.01, 0.0, 0.01, 5) is False


def test_first_switch_current_none_when_nothing_switched():
    assert ns.first_switch_current_A([{"pulse_current_A": 1e-3, "switched": False},
                                      {"pulse_current_A": 0.0, "switched": None}]) is None


def test_ap_fraction():
    assert ns._ap_fraction(0.0, -1.0, 1.0) == 0.5
    assert ns._ap_fraction(0.0, None, None) is None


def test_initialize_with_field_ramps_to_init_then_hold(monkeypatch):
    calls = []
    monkeypatch.setattr(ns, "set_magnet_current",
                        lambda magnet, cfg, i, gm, gcfg, tol, stop: calls.append(("set", i, tol)))
    monkeypatch.setattr(ns, "read_field_mT", lambda gm, gcfg: calls.append(("read",)) or 12.5)

    info = ns.initialize_with_field(object(), None, object(), None, init_A=5.0, hold_A=0.0,
                                    tolerance_mT=0.05)

    assert calls == [("set", 5.0, 0.05), ("read",), ("set", 0.0, 0.05)]
    assert info == {"init_field_measured_mT": 12.5}


@pytest.mark.parametrize("points, read_kw", [
    ([ns.PulsePoint(0.2)], {}),                                  # beyond the 6221 range
    ([ns.PulsePoint(-5e-3)], {}),                                # unipolar only: no negative pulse
    ([ns.PulsePoint(float("nan"))], {}),
    ([ns.PulsePoint(5e-4)], {"sense_current_A": 5e-4}),          # read >= smallest pulse
    ([ns.PulsePoint(5e-3)], {"n_reversals": 1}),
    ([ns.PulsePoint(5e-3)], {"R_P_ohm": 1.0}),                   # only one reference level
    ([ns.PulsePoint(5e-3)], {"sense_current_A": 0.5}),           # over the read ceiling
])
def test_guards_refuse(points, read_kw):
    src = _Fake6221()
    with pytest.raises(ValueError):
        ns.run_measurement(src, _FakeVoltmeter(src), ns.WritePulseConfig(),
                           ns.ReadConfig(**read_kw), points, write_csv=_NULL_WRITER)
    assert src.enabled is False and src.pulse_levels == []   # refused before any hardware was touched
