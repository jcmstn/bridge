"""
web/sot/nonlocal_switching.py — the page is a thin layer over the TUI module (it
must reuse, not fork, build_plan / run_plan / the PNG writer), and no web page
may share a basename with a module of the same-named top-level suite package.
"""

from __future__ import annotations

import importlib

from pathlib import Path

import pytest

import sot.sot_nonlocal_switching_tui as tui
import web.sot.nonlocal_switching as page_mod

REPO = Path(__file__).resolve().parent.parent


def test_page_reuses_the_tui_modules_pure_helpers():
    assert callable(page_mod.page)
    assert page_mod.program is tui          # run_plan / headers / PNG come from the TUI module
    assert page_mod.build_plan is tui.build_plan
    assert page_mod.build_summary is tui.build_summary
    assert tui.MEASUREMENT_TYPE == "NLSW" and page_mod.SUITE == "SOT"


@pytest.mark.parametrize("page,program", [
    ("web.dc.hall", "dc.dc_hall_measurement_tui"),
    ("web.dc.iv_curve", "dc.dc_iv_curve_tui"),
    ("web.dc.gate_sweep", "dc.dc_gate_sweep_tui"),
    ("web.dc.spin_valve", "dc.dc_spin_valve_tui"),
    ("web.mfli.dual_harmonic", "mfli.mfli_dual_harmonic_tui"),
    ("web.mfli.diff_resistance", "mfli.mfli_diff_resistance_tui"),
    ("web.mfli.phase_calibration", "mfli.mfli_phase_calibration_tui"),
    ("web.sot.nonlocal_switching", "sot.sot_nonlocal_switching_tui"),
])
def test_every_page_runs_its_tui_modules_plan_and_run(page, program):
    """One plan builder + one run loop per program, shared by both front ends."""
    page_mod, prog = importlib.import_module(page), importlib.import_module(program)
    assert page_mod.program is prog and page_mod.build_plan is prog.build_plan
    for name in ("run_plan", "build_header_fields", "save_run_png", "MEASUREMENT_TYPE"):
        assert hasattr(prog, name), name


@pytest.mark.parametrize("suite", ["dc", "mfli", "sot"])
def test_no_web_page_shadows_a_measurement_module(suite):
    # `python web/app.py` puts web/ first on sys.path, so web/<suite>/x.py
    # would shadow <suite>/x.py and break the TUI-module import (circular).
    pages = {p.name for p in (REPO / "web" / suite).glob("*.py") if p.name != "__init__.py"}
    modules = {p.name for p in (REPO / suite).glob("*.py") if p.name != "__init__.py"}
    assert not pages & modules, f"web/{suite} shadows {suite}/: {sorted(pages & modules)}"


def test_web_build_plan_takes_the_data_root_from_the_page(tmp_path):
    state = {k: (tui.NUMERIC_FIELDS[k](v) if k in tui.NUMERIC_FIELDS
                 else (float(v) if v else None) if k in tui.OPTIONAL_NUMERIC_FIELDS else v)
             for k, v in tui.DEFAULTS.items()}
    state.update(device="HB3", sample="A", data_dir=str(tmp_path), enable_temperature=False,
                 init_magnet_currents="5, -5", pulse_current_start_A=-2e-3,
                 pulse_current_stop_A=2e-3, pulse_current_step_A=2e-3, reversal_enabled=False)
    plan = page_mod.build_plan(tui.resolve_state(state), tmp_path)

    assert plan.data_root == tmp_path
    assert plan.series_values == [5.0, -5.0]
    assert plan.pulse_currents_A == pytest.approx([-2e-3, 0.0, 2e-3], abs=1e-12)
    assert plan.total_points == (3 + 1) * 2
    assert plan.read_cfg.reversal_enabled is False


@pytest.mark.parametrize("program_name,app_name", [
    ("dc.dc_hall_measurement_tui", "DCHallMeasurementApp"),
    ("dc.dc_iv_curve_tui", "DCIVCurveApp"),
    ("dc.dc_gate_sweep_tui", "DCGateSweepApp"),
    ("dc.dc_spin_valve_tui", "DCSpinValveApp"),
    ("mfli.mfli_dual_harmonic_tui", "MFLIDualHarmonicApp"),
    ("mfli.mfli_diff_resistance_tui", "MFLIDiffResistanceApp"),
    ("mfli.mfli_phase_calibration_tui", "MFLIPhaseCalibrationApp"),
    ("sot.sot_nonlocal_switching_tui", "NonlocalSwitchingApp"),
])
def test_web_form_state_matches_the_tui_parse_state(program_name, app_name, tmp_path, monkeypatch):
    """The web page's form_state() and the TUI's parse_state() turn the same
    default form into the same state — so build_summary/build_plan see one thing."""
    import asyncio
    from types import SimpleNamespace as W

    from textual.widgets import Select, Switch, TextArea
    from web.run_controller import form_state

    prog = importlib.import_module(program_name)
    app_cls = getattr(prog, app_name)
    monkeypatch.setattr(prog, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(prog, "SETTINGS_PATH", tmp_path / "s.json")
    monkeypatch.setattr(app_cls, "data_root", tmp_path)

    async def tui_state():
        app = app_cls()
        async with app.run_test(size=(220, 70)) as pilot:
            await pilot.pause()
            state, _ = app.parse_state()
            switches = {s.id: s.value for s in app.query(Switch) if s.id}
            selects = {s.id: s.value for s in app.query(Select) if s.id and s.id != "sample_select"}
            areas = {a.id: a.text for a in app.query(TextArea) if a.id}
            return state, switches, selects, areas

    tui, switches, selects, areas = asyncio.run(tui_state())
    d = prog.DEFAULTS
    inputs = {fid: W(value=prog.NUMERIC_FIELDS[fid](d[fid])) for fid in prog.NUMERIC_FIELDS}
    inputs |= {fid: W(value=d[fid]) for fid in list(prog.TEXT_FIELDS) + list(getattr(prog, "LIST_FIELDS", []))
               if fid in d}
    inputs |= {fid: W(value=(float(d[fid]) if d.get(fid) not in ("", None) else None))
               for fid in prog.OPTIONAL_NUMERIC_FIELDS}
    inputs |= {fid: W(value=text) for fid, text in areas.items()}
    identity = W(device_input=W(value=d["device"]), cooldown_input=W(value=d["cooldown"]),
                 data_dir_input=W(value=tui["data_dir"]),
                 temperature_input=inputs.get("temperature_setpoint_K", W(value=None)),
                 sample_dropdown=W(value=tui["sample"]))
    web, errors = form_state(prog, identity, inputs=inputs,
                             switches={k: W(value=v) for k, v in switches.items()},
                             selects={k: W(value=v) for k, v in selects.items()})
    assert not errors
    assert web == tui
