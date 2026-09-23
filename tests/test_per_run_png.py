"""
Multi-current-value runs behave like N manual runs: every run gets its own
PNG (own run number, never a combined overlay), drawn exactly as a lone run
would be -- not colored/labelled by its position in the series.

Covers the TUIs of the Hall, gate-sweep, I-V, spin-valve and three SOT
measurements (RunScreen._save_run_png) and the Hall/gate-sweep/I-V/
spin-valve web pages' module-level _save_measurement_png. The MFLI 6221
pair has its own tests. No hardware, no Textual app loop: _save_run_png
only reads `.plan.data_root` (plus `.plan.read_cfg.harmonic` for the 6221
SOT and `.plan.header_extra` for I-V's annotation), so a SimpleNamespace
stands in for the screen.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dc.dc_gate_sweep_tui as gate_tui  # noqa: E402
import dc.dc_hall_measurement_tui as hall_tui  # noqa: E402
import dc.dc_iv_curve_tui as iv_tui  # noqa: E402
import dc.dc_spin_valve_tui as sv_tui  # noqa: E402
import sot.sot_nonlocal_switching_tui as nlsw_tui  # noqa: E402
import sot.sot_pulsed_switching_2h_tui as sot2h_tui  # noqa: E402
import sot.sot_pulsed_switching_6221_tui as sot6221_tui  # noqa: E402
import sot.sot_pulsed_switching_tui as sot_tui  # noqa: E402
import web.dc.gate_sweep as gate_web  # noqa: E402
import web.dc.iv_curve as iv_web  # noqa: E402
import web.dc.spin_valve as sv_web  # noqa: E402
from instruments.data_naming import allocate_run, ensure_sample  # noqa: E402


def _hall(i, idx):
    return {"point_index": i, "magnet_field_mT": None, "sense_current_A": 1e-3 * (idx + 1),
            "rxy_resistance_ohm": 1.0 + i, "rxx_resistance_ohm": np.nan}


def _gate(i, idx):
    return {"gate_voltage_V": float(i), "voltage_V": 1e-3 * i, "sense_current_A": 1e-6}


def _iv(i, idx):
    return {"point_index": i, "current_A": 1e-6 * i, "voltage_V": 1e-3 * i,
            "gate_voltage_V": float(idx)}


def _spin_valve(i, idx):
    return {"point_index": i, "magnet_field_mT": 10.0 * i, "voltage_V": 1e-3 * i,
            "sense_current_A": 1e-3 * (idx + 1), "gate_voltage_V": 2.0 * (idx + 1)}


def _sot(i, idx):
    return {"pulse_amplitude_V": float(i), "hall_resistance_ohm": 1.0 + i, "sense_current_A": 1e-4}


def _sot2h(i, idx):
    return {"pulse_amplitude_V": float(i), "2f_R_V": 1e-6 * i, "excitation_current_A_peak": 1e-4}


def _sot6221(i, idx):
    return {"pulse_current_A": 1e-3 * i, "demod_R_V": 1e-6 * i, "excitation_current_A_peak": 1e-4}


def _nlsw(i, idx):
    return {"pulse_current_A": 1e-3 * i, "nl_resistance_ohm": 0.1 * i, "voltage_even_V": 1e-6,
            "sense_current_A": 1e-4, "switched": None, "init_magnet_current_A": 5.0 * (idx + 1)}


# (module, suffix of the per-run PNG, record factory)
TUI_CASES = [
    pytest.param(hall_tui, "plot", _hall, id="hall"),
    pytest.param(gate_tui, "plot", _gate, id="gate_sweep"),
    pytest.param(iv_tui, "plot", _iv, id="iv_curve"),
    pytest.param(sv_tui, "plot", _spin_valve, id="spin_valve"),
    pytest.param(sot_tui, "Rxy_vs_amp", _sot, id="sot"),
    pytest.param(sot2h_tui, "V2f_vs_amp", _sot2h, id="sot_2h"),
    pytest.param(sot6221_tui, "Vnf_vs_pulse", _sot6221, id="sot_6221"),
    pytest.param(nlsw_tui, "NL_vs_pulse", _nlsw, id="nonlocal_switching"),
]


def _records(make, series_idx, n=3):
    """One run of a multi-current series: what RunScreen hands to
    _save_run_png -- tagged with its position in the series, as the
    on_point callback does."""
    out = []
    for i in range(n):
        r = make(i, series_idx)
        r.update(series_index=series_idx, series_label=f"I=run{series_idx}")
        out.append(r)
    return out


def _capture_figs(fn) -> list:
    figs: list = []
    real_close = plt.close
    plt.close = lambda fig=None: (figs.append(fig), real_close(fig))
    try:
        fn()
    finally:
        plt.close = real_close
    return figs


@pytest.mark.parametrize("mod, suffix, make", TUI_CASES)
def test_every_run_gets_its_own_png_and_no_combined(tmp_path, mod, suffix, make) -> None:
    ensure_sample(tmp_path, "A", create=True)
    # A bare RunScreen (no __init__/mount): _save_run_png only needs the plan
    # and the program's own PNG hook.
    screen = mod.RunScreen.__new__(mod.RunScreen)
    screen.plan = SimpleNamespace(data_root=tmp_path, field_theta_deg=None, field_phi_deg=None,
                                  header_extra={}, read_cfg=SimpleNamespace(harmonic=2))
    screen._png_path = None
    run_strs = []
    for idx in range(3):
        ctx = allocate_run(tmp_path, "A", "HB3", mod.MEASUREMENT_TYPE, series="")
        run_strs.append(ctx.run_str)
        mod.RunScreen._save_run_png(screen, ctx, _records(make, idx))

    names = sorted(p.name for p in (tmp_path / "A" / "proc").glob("*.png"))
    assert names == sorted(f"A_{r}_HB3_{mod.MEASUREMENT_TYPE}_{suffix}.png" for r in run_strs)
    assert len(set(run_strs)) == 3
    # The one the operator's comment is re-saved into is the LAST run's.
    assert screen._png_path.name == f"A_{run_strs[-1]}_HB3_{mod.MEASUREMENT_TYPE}_{suffix}.png"


def _save_fn(mod, make, path):
    recs = _records(make, 2)  # third run of a series
    if mod is sot6221_tui:
        return lambda: mod._save_measurement_png(recs, path, 2)
    return lambda: mod._save_measurement_png(recs, path)


@pytest.mark.parametrize("mod, make", [
    pytest.param(hall_tui, _hall, id="hall_tui"),
    pytest.param(gate_tui, _gate, id="gate_tui"),
    pytest.param(iv_tui, _iv, id="iv_tui"),
    pytest.param(sv_tui, _spin_valve, id="spin_valve_tui"),
    pytest.param(sot_tui, _sot, id="sot_tui"),
    pytest.param(sot2h_tui, _sot2h, id="sot2h_tui"),
    pytest.param(sot6221_tui, _sot6221, id="sot6221_tui"),
    pytest.param(nlsw_tui, _nlsw, id="nonlocal_switching_tui"),
    pytest.param(gate_web, _gate, id="gate_web"),
    pytest.param(iv_web, _iv, id="iv_web"),
    pytest.param(sv_web, _spin_valve, id="spin_valve_web"),
])
def test_png_of_a_lone_run_ignores_its_position_in_the_series(tmp_path, mod, make) -> None:
    # Run #3 of a series must look like a manual run: the default blue, no
    # "I=..." series label -- not tab10's third color with a legend entry.
    png = tmp_path / "run.png"
    (fig,) = _capture_figs(_save_fn(mod, make, png))
    assert png.exists()
    assert {ln.get_color() for ax in fig.axes for ln in ax.lines} == {"tab:blue"}
    for ax in fig.axes:
        legend = ax.get_legend()
        assert legend is None or not any("run" in t.get_text() for t in legend.get_texts())


def test_hall_png_annotates_the_runs_own_sense_current(tmp_path) -> None:
    # The annotation used to read plan.series_values, which is empty of
    # meaning once each run is plotted alone.
    plan = SimpleNamespace(field_theta_deg=None, field_phi_deg=None)
    for mod in (hall_tui,):
        figs = _capture_figs(lambda: mod._save_measurement_png(
            _records(_hall, 1), tmp_path / "h.png", plan=plan))
        text = "\n".join(t.get_text() for t in figs[0].texts)
        assert "Sense current: 2.000 mA" in text


def test_spin_valve_png_annotates_the_runs_own_sense_current_and_gate(tmp_path) -> None:
    # Was plan-derived (only when the plan had a single value), which is
    # meaningless once each run of a series is plotted alone.
    for mod in (sv_tui, sv_web):
        figs = _capture_figs(lambda: mod._save_measurement_png(
            _records(_spin_valve, 1), tmp_path / "sv.png", plan=SimpleNamespace()))
        text = "\n".join(t.get_text() for t in figs[0].texts)
        assert "Sense current: 2.000 mA" in text
        assert "Gate voltage: 4.000 V" in text
