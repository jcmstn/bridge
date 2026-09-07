"""
The general SMU wrapper (SMUConfig + connect_smu / set_source_level /
acquire_measurement) on instruments.keithley2400 and .keithley2450.

Hardware-free: the pymeasure / vendored driver class is monkeypatched with a
_FakeSMU that records every attribute write and serves a scripted reading
stream. Both modules expose the same function names, so every test runs
against both via `mod`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import instruments.keithley2400 as k2400
import instruments.keithley2450 as k2450

MODS = pytest.mark.parametrize("mod", [k2400, k2450], ids=["k2400", "k2450"])


class _FakeSMU:
    """Records attribute writes; `voltage`/`current`/`resistance` read from a
    scripted stream so acquire_measurement() sees changing samples."""

    def __init__(self, adapter, name="fake", **kw):
        object.__setattr__(self, "writes", {})
        object.__setattr__(self, "calls", [])
        object.__setattr__(self, "_stream", iter(()))
        object.__setattr__(self, "_buffer", np.array([1.0, 2.0, 3.0]))

    # -- attribute recording ------------------------------------------------
    def __setattr__(self, key, value):
        self.writes[key] = value

    def _call(name):
        def method(self, *a, **kw):
            self.calls.append((name, a, kw))
        return method

    reset = _call("reset")
    apply_voltage = _call("apply_voltage")
    apply_current = _call("apply_current")
    measure_voltage = _call("measure_voltage")
    measure_current = _call("measure_current")
    measure_resistance = _call("measure_resistance")
    use_front_terminals = _call("use_front_terminals")
    use_rear_terminals = _call("use_rear_terminals")
    enable_source = _call("enable_source")
    shutdown = _call("shutdown")
    config_buffer = _call("config_buffer")
    start_buffer = _call("start_buffer")
    wait_for_buffer = _call("wait_for_buffer")
    disable_buffer = _call("disable_buffer")
    del _call

    def ask(self, cmd):
        self.calls.append(("ask", (cmd,), {}))
        if "ACTual" in cmd:
            return "3"
        return "1.5"      # AVERage / STDDev

    @property
    def buffer_data(self):
        return self._buffer

    # -- scripted readings ------------------------------------------------
    def load_stream(self, values):
        object.__setattr__(self, "_stream", iter(values))

    def _next(self):
        return float(next(self._stream))

    voltage = property(_next)
    current = property(_next)
    resistance = property(_next)


@pytest.fixture
def patched(monkeypatch):
    """Both modules' driver class → _FakeSMU. Returns the class so a test can
    read what connect_smu() built."""
    monkeypatch.setattr(k2400, "Keithley2400", _FakeSMU)
    monkeypatch.setattr(k2450, "Keithley2450", _FakeSMU)
    return _FakeSMU


# ─────────────────────────────────────────────────────────────────────────────
# connect_smu — config → instrument state
# ─────────────────────────────────────────────────────────────────────────────

@MODS
def test_connect_source_current_measure_voltage_4wire(mod, patched):
    cfg = mod.SMUConfig(source_function="current", compliance_voltage_V=2.0,
                        four_wire=True, nplc=1.0, terminals="rear")
    smu = mod.connect_smu(cfg)

    call = dict((c[0], (c[1], c[2])) for c in smu.calls)
    assert "reset" in call
    # sourced current, with the voltage-compliance passed through
    assert call["apply_current"][1]["compliance_voltage"] == 2.0
    # sensed the complement (voltage), autoranged, at the requested NPLC
    assert call["measure_voltage"][1] == {"nplc": 1.0, "auto_range": True}
    assert ("use_rear_terminals", (), {}) in smu.calls
    assert ("enable_source", (), {}) in smu.calls
    # 4-wire target differs per model, but both must end up remote-sensing
    if mod is k2400:
        assert smu.writes["wires"] == 4
    else:
        assert smu.writes["sense_wire_mode"] == "4"


@MODS
def test_connect_two_wire_default(mod, patched):
    smu = mod.connect_smu(mod.SMUConfig())
    if mod is k2400:
        assert smu.writes["wires"] == 2
    else:
        assert smu.writes["sense_wire_mode"] == "2"


@MODS
def test_connect_explicit_sense_range_disables_autorange(mod, patched):
    cfg = mod.SMUConfig(source_function="voltage", sense_function="current",
                        sense_range=1e-3, nplc=2.0)
    smu = mod.connect_smu(cfg)
    call = dict((c[0], (c[1], c[2])) for c in smu.calls)
    assert call["measure_current"][1] == {"nplc": 2.0, "auto_range": False, "current": 1e-3}


@MODS
def test_connect_rejects_bad_source_function(mod, patched):
    with pytest.raises(ValueError):
        mod.connect_smu(mod.SMUConfig(source_function="power"))


@MODS
def test_output_off_state_maps_friendly_name_to_scpi_token(mod, patched):
    # "guard" must reach the instrument as GUAR, not GUARD (validator would reject)
    smu = mod.connect_smu(mod.SMUConfig(source_function="current", output_off_state="guard"))
    key = "current_output_off_state" if mod is k2450 else "output_off_state"
    assert smu.writes[key] == "GUAR"


@MODS
def test_connect_rejects_bad_output_off_state(mod, patched):
    with pytest.raises(ValueError):
        mod.connect_smu(mod.SMUConfig(output_off_state="open"))


# ─────────────────────────────────────────────────────────────────────────────
# set_source_level — software limit guard (mirrors set_gate_voltage)
# ─────────────────────────────────────────────────────────────────────────────

@MODS
def test_set_source_level_within_limit(mod, patched):
    cfg = mod.SMUConfig(source_function="current", source_limit_A=1e-3)
    smu = mod.connect_smu(cfg)
    mod.set_source_level(smu, cfg, 5e-4)
    assert smu.writes["source_current"] == 5e-4


@MODS
def test_set_source_level_past_limit_raises(mod, patched):
    cfg = mod.SMUConfig(source_function="current", source_limit_A=1e-3)
    smu = mod.connect_smu(cfg)
    with pytest.raises(ValueError):
        mod.set_source_level(smu, cfg, 2e-3)


@MODS
def test_set_source_level_voltage_limit_is_symmetric(mod, patched):
    cfg = mod.SMUConfig(source_function="voltage", source_limit_V=10.0)
    smu = mod.connect_smu(cfg)
    with pytest.raises(ValueError):
        mod.set_source_level(smu, cfg, -10.5)


# ─────────────────────────────────────────────────────────────────────────────
# acquire_measurement — same shape as keithley2182.acquire_averaged_voltage
# ─────────────────────────────────────────────────────────────────────────────

@MODS
def test_acquire_measurement_mean_and_sem(mod, patched):
    cfg = mod.SMUConfig(source_function="current")   # sense = voltage
    smu = mod.connect_smu(cfg)
    smu.load_stream([1.0, 2.0, 3.0, 4.0, 5.0])
    out = mod.acquire_measurement(smu, cfg, n=5)
    assert out["mean"] == 3.0
    assert abs(out["sem"] - float(np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))) < 1e-12
    assert "std" not in out


@MODS
def test_acquire_measurement_single_sample_sem_is_nan(mod, patched):
    cfg = mod.SMUConfig()
    smu = mod.connect_smu(cfg)
    smu.load_stream([7.0])
    out = mod.acquire_measurement(smu, cfg, n=1)
    assert out["mean"] == 7.0
    assert math.isnan(out["sem"])


@MODS
def test_acquire_measurement_stop_event_cuts_short(mod, patched):
    import threading

    cfg = mod.SMUConfig()
    smu = mod.connect_smu(cfg)
    smu.load_stream([1.0, 2.0, 3.0, 4.0, 5.0])
    ev = threading.Event()
    ev.set()
    out = mod.acquire_measurement(smu, cfg, n=5, stop_event=ev)
    assert out["mean"] == 1.0          # broke after the first sample
    assert math.isnan(out["sem"])


# ─────────────────────────────────────────────────────────────────────────────
# shutdown_smu
# ─────────────────────────────────────────────────────────────────────────────

@MODS
def test_measure_buffered_returns_stats_and_restores_single_shot(mod, patched):
    cfg = mod.SMUConfig(source_function="current")
    smu = mod.connect_smu(cfg)
    smu.calls.clear()
    out = mod.measure_buffered(smu, n=64)
    assert set(out) == {"mean", "std", "n"}
    names = [c[0] for c in smu.calls]
    assert "config_buffer" in names and "start_buffer" in names
    # state leak guard: single-shot restored so later :READ? is one sample
    assert "disable_buffer" in names
    if mod is k2400:
        assert smu.writes["trigger_count"] == 1


@MODS
def test_shutdown_smu_calls_driver_shutdown(mod, patched):
    smu = mod.connect_smu(mod.SMUConfig())
    smu.calls.clear()
    mod.shutdown_smu(smu)
    assert ("shutdown", (), {}) in smu.calls
