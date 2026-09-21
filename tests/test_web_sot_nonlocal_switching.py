"""
web/sot/nonlocal_switching.py — the page is a thin layer over the TUI module (it
must reuse, not fork, build_plan / run_plan / the PNG writer), and no web page
may share a basename with a module of the same-named top-level suite package.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sot.sot_nonlocal_switching_tui as tui
import web.sot.nonlocal_switching as page_mod

REPO = Path(__file__).resolve().parent.parent


def test_page_reuses_the_tui_modules_pure_helpers():
    assert callable(page_mod.page)
    assert page_mod.build_plan is tui.build_plan
    assert page_mod.run_plan is tui.run_plan
    assert page_mod.resolve_state is tui.resolve_state
    assert page_mod.build_summary is tui.build_summary
    assert page_mod._save_measurement_png is tui._save_measurement_png
    assert page_mod.MEASUREMENT_TYPE == "NLSW" and page_mod.SUITE == "SOT"


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
    plan = page_mod.build_plan(page_mod.resolve_state(state), tmp_path)

    assert plan.data_root == tmp_path
    assert plan.series_values == [5.0, -5.0]
    assert plan.pulse_currents_A == pytest.approx([-2e-3, 0.0, 2e-3], abs=1e-12)
    assert plan.total_points == (3 + 1) * 2
    assert plan.read_cfg.reversal_enabled is False
