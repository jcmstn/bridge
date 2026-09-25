"""
Programs merged behind a mode toggle keep every mode's saved form values: the
merged form reads the former programs' settings files key by key, and a save
from before the toggle existed opens in whichever mode was used last.
"""

from __future__ import annotations

import asyncio
import json
import os


import mfli.mfli_dual_harmonic_6221_tui as six
import mfli.mfli_dual_harmonic_tui as harm
import sot.sot_pulsed_switching_tui as sot


def _mount_values(monkeypatch, tmp_path, ids, **app_kwargs):
    monkeypatch.setattr(harm, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(harm.MFLIDualHarmonicApp, "data_root", tmp_path)

    async def go():
        app = harm.MFLIDualHarmonicApp(**app_kwargs)
        async with app.run_test(size=(220, 70)) as pilot:
            await pilot.pause()
            return {i: app.query_one(f"#{i}").value for i in ids} | {
                "_6221_card_shown": app.query_one("#mode_6221_excitation").display,
                "_mfli_card_shown": app.query_one("#mode_mfli_excitation").display}
    return asyncio.run(go())


def test_dual_harmonic_merge_keeps_both_forms_settings(tmp_path, monkeypatch):
    harm_path, six_path = tmp_path / "harm.json", tmp_path / "six.json"
    monkeypatch.setattr(harm, "SETTINGS_PATH", harm_path)
    monkeypatch.setattr(six, "SETTINGS_PATH", six_path)
    harm_path.write_text(json.dumps({"amplitude_V": "0.25", "frequency_Hz": "311"}))
    six_path.write_text(json.dumps({"amplitude_values": "2e-6, 4e-6", "frequency_Hz": "977"}))
    os.utime(harm_path, (1, 1))                      # the 6221 form was used last

    v = _mount_values(monkeypatch, tmp_path, ["ac_source", "amplitude_V", "amplitude_values", "frequency_Hz"])
    assert v["ac_source"] == "6221" and v["_6221_card_shown"] and not v["_mfli_card_shown"]
    assert v["amplitude_V"] == "0.25" and v["amplitude_values"] == "2e-6, 4e-6"
    assert v["frequency_Hz"] == "311"                # the merged form's own file wins per key

    # once saved, the merged file carries the toggle itself
    harm_path.write_text(json.dumps({"ac_source": "mfli"}))
    assert _mount_values(monkeypatch, tmp_path, ["ac_source"])["ac_source"] == "mfli"
    # and the old 6221 entry point opens it on the 6221 source
    assert _mount_values(monkeypatch, tmp_path, ["ac_source"], ac_source="6221")["ac_source"] == "6221"


def _mount_sot(monkeypatch, tmp_path, ids, **app_kwargs):
    monkeypatch.setattr(sot, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(sot.SOTPulsedSwitchingApp, "data_root", tmp_path)

    async def go():
        app = sot.SOTPulsedSwitchingApp(**app_kwargs)
        async with app.run_test(size=(220, 70)) as pilot:
            await pilot.pause()
            state, errors = app.parse_state()
            return {i: app.query_one(f"#{i}").value for i in ids} | {
                "_shown": {w for w in app.MODE_WIDGETS if app.query_one(f"#{w}").display},
                "_plan": sot.build_plan(state, tmp_path), "_errors": errors}
    return asyncio.run(go())


def test_sot_pulsed_merge_keeps_all_three_forms_settings(tmp_path, monkeypatch):
    own, h2_path, i1_path = tmp_path / "ps.json", tmp_path / "h2.json", tmp_path / "i1.json"
    monkeypatch.setattr(sot, "SETTINGS_PATH", own)
    monkeypatch.setattr(sot.h2, "SETTINGS_PATH", h2_path)
    monkeypatch.setattr(sot.i1, "SETTINGS_PATH", i1_path)
    own.write_text(json.dumps({"pulse_width_s": "5e-8", "settle_after_enable_s": "0.2", "nplc": "2"}))
    h2_path.write_text(json.dumps({"demod2_index": "3", "frequency_Hz": "311"}))
    # the 6221-only form's pulse width / PLL settle land on this form's own ids
    i1_path.write_text(json.dumps({"pulse_width_s": "2e-3", "settle_after_enable_s": "1.5",
                                   "harmonic": "1", "frequency_Hz": "977"}))
    os.utime(own, (1, 1))
    os.utime(h2_path, (2, 2))                        # the 6221-only form was used last

    v = _mount_sot(monkeypatch, tmp_path, ["pulse_source", "read_mode", "pulse_width_s",
                                           "wave_pulse_width_s", "settle_after_enable_s",
                                           "lock_settle_s", "nplc", "demod2_index", "harmonic",
                                           "frequency_Hz"])
    assert (v["pulse_source"], v["read_mode"]) == ("6221", "harmonic")
    assert (v["pulse_width_s"], v["wave_pulse_width_s"]) == ("5e-8", "2e-3")
    assert (v["settle_after_enable_s"], v["lock_settle_s"]) == ("0.2", "1.5")
    assert (v["nplc"], v["demod2_index"], v["harmonic"], v["frequency_Hz"]) == ("2", "3", "1", "977")
    assert v["_shown"] == {"mode_6221_pulse", "mode_lockin_read", "mode_mfli",
                           "mode_sot1i_demod", "mode_sot1i_harmonic"}
    # ...and the plan is the 6221-only engine's, with its own keys
    plan = v["_plan"]
    assert sot.engine(plan) is sot.i1 and not v["_errors"]
    assert plan.pulse_cfg.width_s == 2e-3 and plan.read_cfg.harmonic == 1

    # once saved, this form's file carries the toggles and wins
    own.write_text(json.dumps({"pulse_source": "pmu", "read_mode": "dc"}))
    v = _mount_sot(monkeypatch, tmp_path, ["pulse_source", "read_mode", "wave_pulse_width_s"])
    assert (v["pulse_source"], v["read_mode"], v["wave_pulse_width_s"]) == ("pmu", "dc", "2e-3")
    assert sot.engine(v["_plan"]) is sot and "mode_dc_read" in v["_shown"]
    # the old 2nd-harmonic entry point opens it on PMU + lock-in
    v = _mount_sot(monkeypatch, tmp_path, ["pulse_source", "read_mode"],
                   pulse_source="pmu", read_mode="harmonic")
    assert (v["pulse_source"], v["read_mode"]) == ("pmu", "harmonic")
    assert sot.engine(v["_plan"]) is sot.h2


def test_web_form_keeps_a_numeric_looking_mode_choice_a_string(tmp_path):
    """The web form's "6221" AC-source choice must reach the program as the
    string "6221" — int()-casting it silently ran the MFLI-output mode."""
    from types import SimpleNamespace as NS
    from web.run_controller import form_state

    w = lambda v: NS(value=v)
    ident = NS(device_input=w("HB3"), cooldown_input=w(""), data_dir_input=w(str(tmp_path)),
               temperature_input=w(None), sample_dropdown=w("_test"))
    inputs = {k: w(v) for k, v in harm.DEFAULTS.items() if k not in harm.OPTIONAL_NUMERIC_FIELDS}
    optional = {k: w(None) for k in harm.OPTIONAL_NUMERIC_FIELDS}
    switches = {k: w(v) for k, v in harm.DEFAULTS.items() if isinstance(v, bool)}
    int_selects = {k: w(int(harm.DEFAULTS[k]))
                   for k in ("order_1f", "order_2f", "leader_automode", "follower_automode",
                             "leader_harmonic", "follower_harmonic")}
    state, errors = form_state(harm, ident, inputs=inputs, switches=switches, optional_inputs=optional,
                               selects={"ac_source": w("6221"), **int_selects})
    assert not errors and state["ac_source"] == "6221" and state["order_1f"] == int(harm.DEFAULTS["order_1f"])
    assert harm.engine(harm.build_plan(state, tmp_path)) is six


def test_sot_form_has_one_widget_per_field_id(tmp_path, monkeypatch):
    """Every form id appears once — a hidden twin sharing an id (the old
    per-pulse-card "sweep back down" switch) silently overrode the visible one."""
    monkeypatch.setattr(sot, "SETTINGS_PATH", tmp_path / "ps.json")
    monkeypatch.setattr(sot, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(sot.SOTPulsedSwitchingApp, "data_root", tmp_path)
    from collections import Counter
    from textual.widgets import Switch

    async def go():
        app = sot.SOTPulsedSwitchingApp()
        async with app.run_test(size=(220, 70)) as pilot:
            await pilot.pause()
            ids = Counter(w.id for w in app.query("*") if w.id in sot.DEFAULTS)
            app.query_one("#amplitude_bidirectional", Switch).value = False
            await pilot.pause()
            return ids, app.parse_state()[0]["amplitude_bidirectional"]

    ids, bidirectional = asyncio.run(go())
    assert [i for i, n in ids.items() if n > 1] == []
    assert bidirectional is False
