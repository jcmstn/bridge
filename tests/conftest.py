"""Suite-wide isolation: no test ever writes the real <repo>/../data/runs.db,
or reads a real settings file of a program that is now a mode of a merged form
(the merged form falls back to those files)."""

from __future__ import annotations

import pytest

from instruments import run_index


@pytest.fixture(autouse=True)
def _isolated_run_history(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(run_index, "_DB_PATH", tmp_path_factory.mktemp("runs") / "runs.db")


@pytest.fixture(autouse=True)
def _isolated_legacy_settings(tmp_path_factory, monkeypatch):
    import mfli.mfli_dual_harmonic_6221_tui as harm6
    import sot.sot_pulsed_switching_2h_tui as sot2h
    import sot.sot_pulsed_switching_6221_tui as sot1i

    legacy = tmp_path_factory.mktemp("legacy_settings")
    for mod in (harm6, sot2h, sot1i):
        monkeypatch.setattr(mod, "SETTINGS_PATH", legacy / f"{mod.__name__}.json")
