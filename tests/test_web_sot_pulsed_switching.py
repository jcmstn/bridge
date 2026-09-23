"""
web/sot/pulsed_switching.py in NiceGUI's simulated browser: the Write pulse ×
Read toggles show exactly that mode's cards (the invalid 6221 + DC combination
is blocked), and Start runs the mode's own engine end to end — hardware faked —
saving a run with that mode's type code while the live table fills.
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest
from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

import sot.sot_pulsed_switching_tui as sot
import web.sot.pulsed_switching as page_mod
from test_run_plans import _stub_hardware
from web import run_manager

_CARDS = ["Write pulse (4200A PMU)", "Write pulse (6221 WAVE, hardware-timed)",
          "DC R_xy read (6221 ±I + 2182)", "Lock-in read (6221 AC + MFLI)",
          "Lock-in harmonic (6221 pulse)", "Keithley 4200A PMU (KXCI)", "Keithley 2182 + DC read",
          "Zurich Instruments MFLI + 6221 marker"]
MODES = [  # (pulse, read, type code, cards shown)
    ("pmu", "dc", "SOTPS", {0, 2, 5, 6}),
    ("pmu", "harmonic", "SOT2H", {0, 3, 5, 7}),
    ("6221", "harmonic", "SOT1I", {1, 3, 4, 7}),
]


def _visible(user, text: str) -> bool:
    try:
        return bool(user.find(text).elements)         # find() only sees visible elements
    except AssertionError:
        return False


def _fake_run_measurement(*args, **kwargs):
    """Two points carrying every column the page's plot / table read, any mode."""
    records = []
    for i in range(2):
        rec = {"amplitude_index": i, "magnet_current_A": 1.5, "temperature_1_K": None,
               "pulse_amplitude_V": 0.2 * (i + 1), "pulse_current_measured_A": 1e-3,
               "hall_voltage_V": 1e-4, "hall_resistance_ohm": 1.0 + i,
               "1f_R_V": 1e-4, "2f_R_V": 1e-6 * (i + 1), "reference_locked": True,
               "pulse_current_A": 1e-3 * (i + 1), "pulse_width_measured_s": 1e-3, "demod_R_V": 2e-6}
        records.append(rec)
        kwargs["on_point"](rec)
        kwargs["write_csv"](records)
    return pd.DataFrame(records)


@pytest.fixture
def page(tmp_path, monkeypatch):
    monkeypatch.setattr(page_mod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(page_mod, "_SETTINGS_PATH", tmp_path / "web.json")
    (tmp_path / "web.json").write_text(json.dumps({"device": "HB3", "data_dir": str(tmp_path)}))
    for eng in (sot, sot.h2, sot.i1):
        _stub_hardware(eng, monkeypatch)
        monkeypatch.setattr(eng, "run_measurement", _fake_run_measurement)
    return tmp_path


def _selects(user) -> tuple:
    by_label = {e.props.get("label"): e for e in user.find(ui.select).elements}
    return by_label["Write pulse"], by_label["Read"]


def test_toggles_show_exactly_the_modes_cards_and_block_6221_with_dc(page):
    async def go():
        async with user_simulation(root=page_mod.page) as user:
            await user.open("/")
            pulse, read = _selects(user)
            for p, r, _code, shown in MODES:
                pulse.value, read.value = p, r
                assert {i for i, c in enumerate(_CARDS) if _visible(user, c)} == shown, (p, r)
            pulse.value, read.value = "6221", "dc"
            await user.should_see("is not a program here")          # the summary blocks it

    asyncio.run(go())


@pytest.mark.parametrize("pulse,read,code", [m[:3] for m in MODES], ids=[m[2] for m in MODES])
def test_start_runs_the_modes_engine_and_saves_its_type_code(page, pulse, read, code, caplog):
    async def go():
        async with user_simulation(root=page_mod.page) as user:
            await user.open("/")
            p, r = _selects(user)
            p.value, r.value = pulse, read
            user.find("Start measurement").click()
            for _ in range(200):
                await asyncio.sleep(0.05)
                if run_manager.snapshot() is None and _visible(user, "Measurement complete."):
                    break
            await user.should_see("Measurement complete.")
            table = user.find(ui.table).elements.pop()
            return len(table.rows)

    assert asyncio.run(go()) == 2
    index = pd.read_csv(page / "_test" / "index.csv")
    assert list(index["type"]) == [code] and list(index["status"]) == ["completed"]
    assert list((page / "_test" / "proc").glob(f"*_{code}_*.png"))          # the per-run PNG
    assert [r.getMessage() for r in caplog.records if r.levelname == "ERROR"] == []
