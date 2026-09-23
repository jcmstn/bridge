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
