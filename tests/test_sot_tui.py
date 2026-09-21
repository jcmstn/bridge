"""
sot/sot_tui.py — the SOT suite picker: one launch button per program, each
mapped to its own App, and pressing one exits the picker with that key.
"""

from __future__ import annotations

import asyncio

import sot.sot_tui as picker
from sot.sot_nonlocal_switching_tui import NonlocalSwitchingApp
from sot.sot_pulsed_switching_2h_tui import SOTPulsedSwitching2HApp
from sot.sot_pulsed_switching_6221_tui import SOTPulsedSwitching6221App
from sot.sot_pulsed_switching_tui import SOTPulsedSwitchingApp

_LAUNCH = {"launch_pulsed": "pulsed", "launch_pulsed_2h": "pulsed_2h",
           "launch_pulsed_6221": "pulsed_6221", "launch_nonlocal": "nonlocal"}


def test_every_program_is_reachable_from_the_picker():
    assert picker._PROGRAMS == {
        "pulsed": SOTPulsedSwitchingApp, "pulsed_2h": SOTPulsedSwitching2HApp,
        "pulsed_6221": SOTPulsedSwitching6221App, "nonlocal": NonlocalSwitchingApp}


def test_picker_has_one_launch_button_per_program_and_each_exits_with_its_key():
    async def go():
        seen = {}
        for button_id, key in _LAUNCH.items():
            app = picker.LauncherApp()
            async with app.run_test(size=(220, 80)) as pilot:
                await pilot.pause()
                assert {b.id for b in app.query("Button")} == set(_LAUNCH)
                app.query_one(f"#{button_id}").press()
                await pilot.pause()
            seen[button_id] = app.return_value
        return seen

    assert asyncio.run(go()) == _LAUNCH
