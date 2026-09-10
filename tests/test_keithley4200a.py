"""
instruments/keithley4200a.py — KXCI reply parsing, the source-level software
guard, and the +I/-I reversal decomposition. Hardware-free: a fake KXCI
transport records every command and serves a scripted reading stream.
"""

from __future__ import annotations

import math

import pytest

import instruments.keithley4200a as k4200
from instruments.keithley4200a import SMUChannelConfig, _parse_reading


class _FakeKXCI:
    def __init__(self, replies):
        self.writes: list[str] = []
        self._replies = iter(replies)

    def command(self, cmd: str) -> None:
        self.writes.append(cmd)

    def query(self, cmd: str) -> str:
        self.writes.append(cmd)
        return next(self._replies)


# ── _parse_reading ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, want", [
    ("1.5", 1.5),
    ("  1.2345E-03 ", 1.2345e-3),
    ("N +1.2345E-03", 1.2345e-3),      # leading status letter, space-separated
    ("C -2.0E-6", -2.0e-6),
    ("NCV1.23E-3", 1.23e-3),           # status letters, no separator
])
def test_parse_reading(raw, want):
    assert _parse_reading(raw) == pytest.approx(want)


def test_parse_reading_empty_raises():
    with pytest.raises(ValueError):
        _parse_reading("   ")


# ── set_source_level: software limit guard + command shape ─────────────────

def test_set_source_level_current_command():
    dev = _FakeKXCI([])
    cfg = SMUChannelConfig(channel=1, source_function="current",
                           compliance_voltage_V=2.0, source_limit_A=5e-3)
    k4200.set_source_level(dev, cfg, 1e-3)
    assert dev.writes == ["DI1, 0, 1.000000E-03, 2.000000E+00"]


def test_set_source_level_voltage_command():
    dev = _FakeKXCI([])
    cfg = SMUChannelConfig(channel=2, source_function="voltage",
                           compliance_current_A=1e-4, source_limit_V=10.0)
    k4200.set_source_level(dev, cfg, -1.5)
    assert dev.writes == ["DV2, 0, -1.500000E+00, 1.000000E-04"]


def test_set_source_level_at_zero_never_trips_a_tiny_limit():
    """sot_switching pins its voltmeter SMU with source_limit_A=1e-9; forcing it
    to 0 A must not trip that limit (abs(0) is not > 1e-9)."""
    dev = _FakeKXCI([])
    cfg = SMUChannelConfig(channel=2, source_function="current",
                           compliance_voltage_V=2.0, source_limit_A=1e-9)
    k4200.set_source_level(dev, cfg, 0.0)
    assert dev.writes == ["DI2, 0, 0.000000E+00, 2.000000E+00"]


def test_set_source_level_refuses_beyond_limit():
    dev = _FakeKXCI([])
    cfg = SMUChannelConfig(source_function="current", source_limit_A=1e-3)
    with pytest.raises(ValueError):
        k4200.set_source_level(dev, cfg, 2e-3)
    assert dev.writes == []          # nothing sent


# ── acquire_reversal_averaged: odd/even decomposition + contract ───────────

def test_acquire_reversal_averaged_decomposition():
    # TV replies, in order: pair0 (+I, -I), pair1 (+I, -I)
    dev = _FakeKXCI(["1.0", "-0.6", "1.2", "-0.4"])
    src = SMUChannelConfig(channel=1, source_function="current", source_limit_A=1.0)
    sense = SMUChannelConfig(channel=2, source_function="current", source_limit_A=1.0)

    out = k4200.acquire_reversal_averaged(dev, src, sense, level=0.1, n_reversals=2)

    # pair0: odd=(1.0-(-0.6))/2=0.8  even=(1.0+(-0.6))/2=0.2
    # pair1: odd=(1.2-(-0.4))/2=0.8  even=(1.2+(-0.4))/2=0.4
    assert out["mean"] == pytest.approx(0.8)
    assert out["sem"] == pytest.approx(0.0)
    assert out["even_mean"] == pytest.approx(0.3)
    assert out["even_sem"] == pytest.approx(0.1)          # std([0.2,0.4],ddof=1)/sqrt(2)
    assert out["n_reversals"] == 2
    assert set(out) == {"mean", "sem", "even_mean", "even_sem", "n_reversals"}

    # left forcing +level on the source channel
    assert dev.writes[-1] == "DI1, 0, 1.000000E-01, 1.000000E+01"
    # read the sense channel (SMU2), never the source, for the voltage
    assert dev.writes.count("TV2") == 4
    assert "TV1" not in dev.writes


def test_acquire_reversal_averaged_single_pair_sem_nan():
    dev = _FakeKXCI(["1.0", "-1.0"])
    src = SMUChannelConfig(channel=1, source_function="current", source_limit_A=1.0)
    sense = SMUChannelConfig(channel=2, source_function="current", source_limit_A=1.0)
    out = k4200.acquire_reversal_averaged(dev, src, sense, level=0.1, n_reversals=1)
    assert out["mean"] == pytest.approx(1.0)
    assert math.isnan(out["sem"])


def test_acquire_reversal_averaged_rejects_voltage_source():
    dev = _FakeKXCI([])
    src = SMUChannelConfig(source_function="voltage")
    sense = SMUChannelConfig(channel=2, source_function="current")
    with pytest.raises(ValueError):
        k4200.acquire_reversal_averaged(dev, src, sense, level=0.1, n_reversals=2)


# ── PMU: _parse_gn / configure_pmu_pulse / pulse_once ─────────────────────

from instruments.keithley4200a import PMUPulseConfig, _parse_gn


@pytest.mark.parametrize("raw, want", [
    ("1.5", 1.5),
    ("N 1.23E-3, 0", 1.23e-3),       # GA-style: value first, status last
    ("  -2.0e-6 ", -2.0e-6),
])
def test_parse_gn(raw, want):
    assert _parse_gn(raw) == pytest.approx(want)


def test_parse_gn_no_number_raises():
    with pytest.raises(ValueError):
        _parse_gn("ERROR")


def test_configure_pmu_pulse_rejects_empty_module():
    with pytest.raises(ValueError):
        k4200.configure_pmu_pulse(_FakeKXCI([]), PMUPulseConfig(module=""))


def test_configure_pmu_pulse_rejects_bad_timing():
    with pytest.raises(ValueError):   # period < width + rise + fall
        k4200.configure_pmu_pulse(_FakeKXCI([]),
                                  PMUPulseConfig(module="m", width_s=1e-6, rise_s=1e-6,
                                                 fall_s=1e-6, period_s=1e-6))


def test_configure_pmu_pulse_rejects_amplitude_over_limit():
    with pytest.raises(ValueError):
        k4200.configure_pmu_pulse(_FakeKXCI([]),
                                  PMUPulseConfig(module="m", amplitude_V=10.0, v_limit_V=5.0))


def test_pulse_once_builds_ex_command_and_substitutes_amplitude():
    """Pins the default arg_order against instruments/kult/bridge_sot_pulse.c's
    signature — KXCI passes arguments positionally, so a drift between the two
    pulses with the wrong numbers instead of erroring. The 4 trailing 0s are
    placeholders for the module's output params (KXCI EX wants every param)."""
    dev = _FakeKXCI(["OK"])
    cfg = PMUPulseConfig(return_names=())
    out = k4200.pulse_once(dev, cfg, amplitude_V=1.2)
    assert dev.writes == [
        "EX bridge_sot bridge_sot_pulse("
        "1.000000E-07, 2.000000E-08, 2.000000E-08, 1.000000E-03, 0.000000E+00, "
        "2.000000E+08, 7.500000E-01, 9.000000E-01, 1, 1.000000E+03, "
        "1.000000E+01, 1.000000E-02, 1.200000E+00, 0.000000E+00, 1, PMU1, "
        "0, 0, 0, 0)"
    ]
    assert out == {"module_return": "OK"}


def test_configure_pmu_pulse_rejects_bad_voltage_range():
    with pytest.raises(ValueError):
        k4200.configure_pmu_pulse(_FakeKXCI([]), PMUPulseConfig(v_range_V=20.0))


def test_configure_pmu_pulse_rejects_edge_below_range_minimum():
    # 40 V range needs rise/fall ≥ 100 ns; the 20 ns default is 10 V-only
    with pytest.raises(ValueError):
        k4200.configure_pmu_pulse(_FakeKXCI([]),
                                  PMUPulseConfig(v_range_V=40.0, period_s=1e-3))
    # ...and the same timings pass on the 10 V range
    k4200.configure_pmu_pulse(_FakeKXCI([]), PMUPulseConfig(v_range_V=10.0))


def test_configure_pmu_pulse_rejects_bad_measure_window():
    with pytest.raises(ValueError):   # start must be < stop
        k4200.configure_pmu_pulse(_FakeKXCI([]),
                                  PMUPulseConfig(meas_start_perc=0.9, meas_stop_perc=0.75))


def test_configure_pmu_pulse_rejects_no_flat_top():
    # width 100 ns, rise/fall 200 ns: settled top = 100 - 100 - 100 < 0 → bench -826
    with pytest.raises(ValueError, match="flat pulse top"):
        k4200.configure_pmu_pulse(_FakeKXCI([]),
                                  PMUPulseConfig(width_s=100e-9, rise_s=200e-9,
                                                 fall_s=200e-9, period_s=1e-3))


def test_pulse_once_reads_outputs_back_by_position_with_gp():
    dev = _FakeKXCI(["done", "0.48", "N 2.0E-3"])
    cfg = PMUPulseConfig(module="m",
                         return_names=("pulse_voltage_measured_V", "pulse_current_measured_A"))
    out = k4200.pulse_once(dev, cfg, amplitude_V=0.5)
    # outputs are params 17-18 (after the 16 inputs) → GP 17, GP 18
    assert [w for w in dev.writes if w.startswith("GP ")] == ["GP 17", "GP 18"]
    assert out["pulse_voltage_measured_V"] == pytest.approx(0.48)
    assert out["pulse_current_measured_A"] == pytest.approx(2.0e-3)


def test_pulse_once_unparseable_gp_becomes_none():
    dev = _FakeKXCI(["done", "GP error: junk"])
    cfg = PMUPulseConfig(module="m", return_names=("pulse_voltage_measured_V",))
    out = k4200.pulse_once(dev, cfg, amplitude_V=0.5)
    assert out["pulse_voltage_measured_V"] is None


def test_pulse_once_refuses_amplitude_over_limit():
    dev = _FakeKXCI([])
    cfg = PMUPulseConfig(module="m", v_limit_V=3.0)
    with pytest.raises(ValueError):
        k4200.pulse_once(dev, cfg, amplitude_V=5.0)
    assert dev.writes == []


# ── the Python↔C argument contract ────────────────────────────────────────
# KXCI passes EX arguments POSITIONALLY, so a drift between PMUPulseConfig's
# arg_order/return_names and bridge_sot_pulse.c's signature does not error --
# it pulses with the wrong numbers. Parse the C file and pin the two together.

def _kult_signature():
    import re
    from pathlib import Path
    c = Path(__file__).resolve().parent.parent / "instruments" / "kult" / "bridge_sot_pulse.c"
    sig = re.search(r"int bridge_sot_pulse\((.*?)\)\s*\n\{", c.read_text(), re.S).group(1)
    params = [p.strip().split()[-1].lstrip("*") for p in sig.split(",")]
    return params[:16], params[16:]


def test_arg_order_matches_the_kult_module_signature():
    c_inputs, c_outputs = _kult_signature()
    cfg = PMUPulseConfig()
    assert len(cfg.arg_order) == len(c_inputs), (
        f"arg_order has {len(cfg.arg_order)} names but bridge_sot_pulse.c takes "
        f"{len(c_inputs)} inputs")
    assert cfg.n_output_params == len(c_outputs), (
        f"n_output_params is {cfg.n_output_params} but bridge_sot_pulse.c has "
        f"{len(c_outputs)} output params — the EX call would be the wrong length")
    # Names differ by convention (C: PulseWidth, Python: width_s), so pin the
    # ORDER via the values pulse_once actually sends. 16 inputs then 4 output
    # placeholders = the module's 20 params.
    dev = _FakeKXCI(["OK"])
    k4200.pulse_once(dev, PMUPulseConfig(return_names=()), amplitude_V=1.2)
    sent = dev.writes[0].split("(", 1)[1].rstrip(")").split(", ")
    assert len(sent) == len(c_inputs) + cfg.n_output_params == 20
    assert sent[len(c_inputs):] == ["0"] * len(c_outputs)
    assert sent[c_inputs.index("AmplitudeV")] == "1.200000E+00"
    assert sent[c_inputs.index("Chan")] == "1"
    assert sent[c_inputs.index("PMU_ID")] == "PMU1"
    assert sent[c_inputs.index("VRange")] == "1.000000E+01"


def test_return_names_match_the_kult_module_outputs():
    _, c_outputs = _kult_signature()
    assert len(PMUPulseConfig().return_names) == len(c_outputs)
    # V before I, amplitude before base -- the order GN fetches them in
    assert c_outputs == ["V_Ampl", "I_Ampl", "V_Base", "I_Base"]
    assert PMUPulseConfig().return_names == (
        "pulse_voltage_measured_V", "pulse_current_measured_A",
        "pulse_base_voltage_V", "pulse_base_current_A")
