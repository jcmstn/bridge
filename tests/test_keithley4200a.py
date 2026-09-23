"""
instruments/keithley4200a.py — the PMU pulse path: KXCI reply parsing, the
configure_pmu_pulse guards and the EX / GP command shapes. Hardware-free: a
fake KXCI transport records every command and serves scripted replies.
"""

from __future__ import annotations


import pytest

import instruments.keithley4200a as k4200


class _FakeKXCI:
    def __init__(self, replies):
        self.writes: list[str] = []
        self._replies = iter(replies)

    def command(self, cmd: str) -> None:
        self.writes.append(cmd)

    def query(self, cmd: str) -> str:
        self.writes.append(cmd)
        return next(self._replies)


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
