"""
The dual-harmonic form's reference phase: the Measure phi_I mode (no run
number, no magnet, autophase both lock-ins, result handed back on the plan),
the sidebar's phi_I warnings, and the MDS oscillator-phase reset.
Hardware-free.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import mfli.mfli_dual_harmonic_6221_tui as six
import mfli.mfli_dual_harmonic_tui as harm_tui
from instruments.data_naming import ensure_sample
from instruments.mfli_daq import sync_oscillator_phases
from test_sr830_lockin_toggle import _harm_state


def test_measure_mode_plans_record_nothing_and_touch_no_magnet(tmp_path):
    ensure_sample(tmp_path, "A", create=True)
    state = _harm_state(measure_phi_I=True, enable_sweep=True, enable_temperature=True)
    plan = harm_tui.build_plan(state, tmp_path)
    assert plan.measure_phi_I and plan.run_ctx is None             # no run number allocated
    assert plan.magnet_cfg is None and plan.currents_A is None and plan.temp_cfg is None
    assert len(plan.run_cost.points) == 1
    plan6 = harm_tui.build_plan({**state, "ac_source": "6221", "amplitude_list": [1e-4, 2e-4]},
                                tmp_path)
    assert plan6.measure_phi_I and plan6.amplitudes_A == [1e-4] and plan6.magnet_cfg is None


def test_measure_mode_run_autophases_and_records_nothing(tmp_path, monkeypatch):
    ensure_sample(tmp_path, "A", create=True)
    plan = harm_tui.build_plan(_harm_state(measure_phi_I=True), tmp_path)
    calls: list[str] = []
    for name in ("connect", "connect_device", "setup_mds", "disable_external_references",
                 "configure_output", "sync_follower_oscillator", "configure_demodulator",
                 "sync_oscillator_phases", "shutdown_output"):
        monkeypatch.setattr(harm_tui, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(harm_tui, "record_run", lambda *a, **k: calls.append("record_run"))
    monkeypatch.setattr(harm_tui, "connect", lambda *a, **k: calls.append("connect") or object())
    result = [SimpleNamespace(phase_after_deg=3.0, converged=True),
              SimpleNamespace(phase_after_deg=-1.0, converged=True)]
    monkeypatch.setattr(six, "measure_phi_I", lambda daq, cfgs, n: calls.append("measure") or result)
    contexts: list = []
    harm_tui.run_plan(plan, threading.Event(), run_contexts=contexts)
    assert plan.phi_I_result == result and contexts == []
    assert "record_run" not in calls and calls[-1] == "shutdown_output"
    # oscillator phases are reset AFTER every frequency write, before the null
    assert calls.index("sync_oscillator_phases") > calls.index("sync_follower_oscillator")
    assert calls.index("measure") > calls.index("sync_oscillator_phases")


def test_sidebar_warns_until_phi_I_is_measured_for_this_setup(tmp_path):
    def warnings(**kw):
        return harm_tui.build_summary(_harm_state(data_dir=str(tmp_path), **kw))[1]
    assert any("φ_I not measured" in w for w in warnings())
    measured = dict(leader_phi_I_deg=3.0, follower_phi_I_deg=-1.0)
    assert any("re-measure φ_I" in w for w in warnings(**measured, phi_I_context="6221 source"))
    ctx = six.phi_I_context(_harm_state(), "MFLI output")
    assert not any("φ_I" in w for w in warnings(**measured, phi_I_context=ctx))
    # another frequency invalidates it
    assert any("re-measure φ_I" in w for w in warnings(**measured, phi_I_context=ctx,
                                                        frequency_Hz=777.0))


def test_sync_oscillator_phases_sets_the_mds_phasesync_parameter():
    class _MDS:
        def __init__(self):
            self.value = None

        def set(self, key, value):
            assert key == "phasesync"
            self.value = value

        def getInt(self, key):
            return 0                       # cleared by the module

    mds, synced = _MDS(), []
    sync_oscillator_phases(mds, SimpleNamespace(sync=lambda: synced.append(True)))
    assert mds.value == 1 and synced == [True]


def test_tui_after_run_fills_the_phi_I_fields_and_turns_the_mode_off(monkeypatch, tmp_path):
    import asyncio
    import json
    from textual.widgets import Input, Switch
    monkeypatch.setattr(harm_tui, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(harm_tui, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(harm_tui.MFLIDualHarmonicApp, "data_root", tmp_path)
    plan = SimpleNamespace(phi_I_result=[
        SimpleNamespace(phase_after_deg=3.21, converged=True),
        SimpleNamespace(phase_after_deg=-1.05, converged=False)])     # unconverged: left alone

    async def go():
        app = harm_tui.MFLIDualHarmonicApp()
        async with app.run_test(size=(220, 70)) as pilot:
            app.query_one("#measure_phi_I", Switch).value = True
            app.query_one("#follower_phi_I_deg", Input).value = "7"
            await pilot.pause()
            app.after_run(plan)
            await pilot.pause()
            return ({i: app.query_one(f"#{i}", Input).value
                     for i in ("leader_phi_I_deg", "follower_phi_I_deg", "phi_I_context")},
                    app.query_one("#measure_phi_I", Switch).value)
    values, measuring = asyncio.run(go())
    assert values["leader_phi_I_deg"] == "3.210" and values["follower_phi_I_deg"] == "7"
    assert values["phi_I_context"].startswith("MFLI output source, MFLI lock-ins, ")
    assert measuring is False
    assert json.loads((tmp_path / "settings.json").read_text())["leader_phi_I_deg"] == "3.210"
