"""
instruments/tui_common.MeasurementRunScreen — the shared run lifecycle, on a
minimal single-run program: points reach the table, the run is finalized
with its outcome status the moment it ends (raw file + index row + PNG),
the operator's status/comment then rewrites that same run, and the session
lands in runs.db. No hardware, no plot window.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from textual import work
from textual.app import App

import instruments.tui_common as tc
from instruments import run_index
from instruments.data_naming import allocate_run, ensure_sample, read_raw


class _Screen(tc.MeasurementRunScreen):
    TABLE_COLUMNS = ("#", "V")
    MEASUREMENT_TYPE = "IV"
    pngs: list = []

    def table_row(self, record):
        return (str(record["point_index"] + 1), f"{record['voltage_V']:g}")

    def save_png(self, records, png_path, comment=""):
        type(self).pngs.append((png_path.name, len(records), comment))

    def build_header(self, ctx, records, *, status, comment, extra):
        return {"run": ctx.run_number, "sample": ctx.sample, "device": ctx.device,
                "type": "IV", "status": status, "comment": comment}

    @work(thread=True, exclusive=True)
    def do_run(self):
        for i in range(3):
            self.app.call_from_thread(self._on_point, {"point_index": i, "voltage_V": 0.1 * i})
        self.app.call_from_thread(self._on_finished, "Measurement complete.")


def test_single_run_lifecycle(tmp_path: Path, monkeypatch) -> None:
    ensure_sample(tmp_path, "A", create=True)
    ctx = allocate_run(tmp_path, "A", "HB3", "IV")
    plan = SimpleNamespace(run_ctx=ctx, data_root=tmp_path, run_cost=None, total_points=3,
                           header_extra={"sense_current_A": 1e-3})
    answers = []
    monkeypatch.setattr(tc, "StatusCommentScreen", lambda: answers.append("asked") or tc.Screen())
    _Screen.pngs = []
    screen = _Screen(plan)

    class Host(App):
        TITLE = "DC I-V Curve"

        def on_mount(self):
            self.push_screen(screen)

    async def go():
        async with Host().run_test(size=(120, 40)) as pilot:
            for _ in range(100):
                await pilot.pause(0.02)
                if not screen._measurement_running:
                    break
            assert screen.query_one("#results_table").row_count == 3
            assert str(screen.query_one("#run_label").render()) == f"Run #{ctx.run_str}"
            # the operator's answer rewrites the SAME (last) run
            screen._on_status_comment(("good", "clean curve"))

    asyncio.run(go())

    assert answers == ["asked"]
    assert len(read_raw(ctx.raw_path)) == 3
    row = pd.read_csv(tmp_path / "A" / "index.csv").iloc[0]
    assert (row["status"], row["comment"]) == ("good", "clean curve")
    assert [(n, k) for n, k, _ in _Screen.pngs] == [(f"A_{ctx.run_str}_HB3_IV_plot.png", 3)] * 2
    assert _Screen.pngs[-1][2] == "clean curve"          # re-saved with the comment
    (hist,) = run_index.recent_runs()
    # suite = the program's package ("dc" -> "DC"); here the test module's
    assert (hist["suite"], hist["measurement"], hist["status"], hist["point_count"]) == \
        (_Screen.__module__.split(".")[0].upper(), "DC I-V Curve", "completed", 3)
    assert hist["run_number"] == str(ctx.run_number)


def test_summary_errors_are_shown_not_fatal_and_infinity_is_a_parse_error(tmp_path, monkeypatch) -> None:
    """A summary that raises (e.g. a model dividing by a just-typed 0) must not
    close the TUI: the sidebar shows it and Start is disabled. And "1e999"
    (float -> inf) is a parse error, not a value."""
    import dc.dc_iv_curve_tui as iv
    from textual.widgets import Button, Input, Static

    monkeypatch.setattr(iv, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(iv, "SETTINGS_PATH", tmp_path / "s.json")
    monkeypatch.setattr(iv.DCIVCurveApp, "data_root", tmp_path)

    async def go():
        app = iv.DCIVCurveApp()
        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()
            app.query_one("#nplc", Input).value = "1e999"
            await pilot.pause()
            _, parse_errors = app.parse_state()
            assert any("nplc" in e for e in parse_errors)

            def boom():
                raise ZeroDivisionError("division by zero")
            app.update_summary = boom
            app.refresh_summary()
            assert "division by zero" in str(app.query_one("#summary", Static).render())
            assert app.query_one("#start", Button).disabled

    asyncio.run(go())
