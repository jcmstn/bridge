"""
sot/sot_nonlocal_switching_tui.py — build_summary validation, the signed pulse sweep
and init-current parsing, _build_plan purity, header fields and the filename
preview. Pure logic only, no Textual mount, no hardware.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pytest

matplotlib.use("Agg")

import sot.sot_nonlocal_switching_tui as tui  # noqa: E402


def _state(tmp_path: Path, **overrides) -> dict:
    """A parsed state built from DEFAULTS the way parse_state() does, so a new
    field can't drift out of the test."""
    base: dict = {}
    for k, v in tui.DEFAULTS.items():
        if k in tui.NUMERIC_FIELDS:
            base[k] = tui.NUMERIC_FIELDS[k](v)
        elif k in tui.OPTIONAL_NUMERIC_FIELDS:
            base[k] = float(v) if v else None
        else:
            base[k] = v
    base.update(device="HB3", sample="A", data_dir=str(tmp_path), temperature_setpoint_K=300.0,
                enable_temperature=False)
    base.update(overrides)
    return tui.resolve_state(base)


def test_default_state_has_no_blocking_errors(tmp_path):
    _, _, errors = tui.build_summary(_state(tmp_path))
    assert errors == []


def test_pulses_may_be_negative_and_a_sweep_through_zero_has_a_read_only_point(tmp_path):
    state = _state(tmp_path, pulse_current_start_A=-2e-3, pulse_current_stop_A=2e-3,
                   pulse_current_step_A=2e-3)
    assert state["pulse_current_parse_error"] is None
    assert state["pulse_current_list"] == pytest.approx([-2e-3, 0.0, 2e-3], abs=1e-12)
    info, _, errors = tui.build_summary(state)
    assert not errors and any("read-only" in i for i in info)
    # all-negative sweeps are fine too
    state = _state(tmp_path, pulse_current_start_A=-1e-3, pulse_current_stop_A=-3e-3,
                   pulse_current_step_A=1e-3)
    assert state["pulse_current_list"] == pytest.approx([-1e-3, -2e-3, -3e-3])
    assert tui.build_summary(state)[2] == []


def test_summary_blocks_zero_step_and_equal_start_stop(tmp_path):
    for overrides in (dict(pulse_current_step_A=0.0),
                      dict(pulse_current_start_A=2e-3, pulse_current_stop_A=2e-3)):
        _, _, errors = tui.build_summary(_state(tmp_path, **overrides))
        assert any("Pulse currents:" in e for e in errors)


def test_summary_blocks_pulse_over_hardware_range_either_sign(tmp_path):
    _, _, errors = tui.build_summary(_state(tmp_path, pulse_current_stop_A=0.5))
    assert any("hardware range" in e for e in errors)
    _, _, errors = tui.build_summary(_state(tmp_path, pulse_current_start_A=-0.5,
                                            pulse_current_stop_A=-0.4))
    assert any("hardware range" in e for e in errors)


def test_summary_blocks_bad_pulse_width_and_compliance(tmp_path):
    _, _, errors = tui.build_summary(_state(tmp_path, pulse_width_s=0.0, pulse_compliance_V=0.0))
    assert any("Pulse width" in e for e in errors)
    assert any("Pulse compliance" in e for e in errors)


def test_sense_current_must_stay_below_the_smallest_pulse(tmp_path):
    _, warnings, errors = tui.build_summary(_state(tmp_path, sense_current_A=1e-3))   # == first pulse
    assert any("smallest pulse" in e for e in errors)
    _, _, errors = tui.build_summary(_state(tmp_path, sense_current_A=1e-3,           # negative pulses too
                                            pulse_current_start_A=-1e-3, pulse_current_stop_A=-5e-3))
    assert any("smallest pulse" in e for e in errors)
    _, warnings, errors = tui.build_summary(_state(tmp_path, sense_current_A=3e-4))   # 30%
    assert not errors and any("read disturb" in w for w in warnings)
    _, warnings, _ = tui.build_summary(_state(tmp_path))                              # 10%
    assert not any("read disturb" in w for w in warnings)


def test_summary_blocks_single_average_and_bad_nplc(tmp_path):
    _, _, errors = tui.build_summary(_state(tmp_path, n_averages=1, nplc=100.0))
    assert any("Averages per read" in e for e in errors)
    assert any("NPLC" in e for e in errors)


def test_reversal_toggle_default_on_and_off_warns_about_the_uncancelled_offset(tmp_path):
    state = _state(tmp_path)
    assert state["reversal_enabled"] is True
    _, warnings, errors = tui.build_summary(state)
    assert not errors and not any("reversal is OFF" in w for w in warnings)

    _, warnings, errors = tui.build_summary(_state(tmp_path, reversal_enabled=False))
    assert not errors
    off = next(w for w in warnings if "reversal is OFF" in w)
    assert "+I_sense" in off and "NOT cancelled" in off and "V_even" in off
    _, warnings, _ = tui.build_summary(_state(tmp_path, reversal_enabled=False, sense_current_A=-1e-4))
    assert any("−I_sense" in w for w in warnings)


def test_sense_current_is_signed_but_never_zero_and_its_size_is_still_guarded(tmp_path):
    _, _, errors = tui.build_summary(_state(tmp_path, sense_current_A=-1e-4))
    assert errors == []                                          # a negative fixed read polarity is fine
    _, _, errors = tui.build_summary(_state(tmp_path, sense_current_A=0.0))
    assert any("non-zero" in e for e in errors)
    _, _, errors = tui.build_summary(_state(tmp_path, sense_current_A=-1e-3))   # |I| == first pulse
    assert any("smallest pulse" in e for e in errors)


def test_plain_read_is_estimated_faster_than_the_reversal_read(tmp_path):
    def eta(**kw):
        info, _, _ = tui.build_summary(_state(tmp_path, **kw))
        return next(i for i in info if i.startswith("Run time"))
    assert eta(reversal_enabled=False) != eta(reversal_enabled=True)


def test_reference_levels_both_or_neither_and_different(tmp_path):
    _, _, errors = tui.build_summary(_state(tmp_path, R_P_ohm=1.0))
    assert any("both R_P and R_AP" in e for e in errors)
    _, _, errors = tui.build_summary(_state(tmp_path, R_P_ohm=1.0, R_AP_ohm=1.0))
    assert any("must differ" in e for e in errors)
    info, _, errors = tui.build_summary(_state(tmp_path, R_P_ohm=-0.5, R_AP_ohm=0.5))
    assert not errors and any("state_AP_fraction" in i for i in info)


def test_no_init_current_leaves_the_magnet_alone_with_a_warning(tmp_path):
    state = _state(tmp_path)
    assert state["init_currents_A"] == [None]
    _, warnings, _ = tui.build_summary(state)
    assert any("No field initialization" in w for w in warnings)


def test_single_init_current_hints_at_the_control_run(tmp_path):
    info, warnings, errors = tui.build_summary(_state(tmp_path, init_magnet_currents="5"))
    assert not errors and not any("No field initialization" in w for w in warnings)
    assert any("opposite sign" in i for i in info)
    info, _, _ = tui.build_summary(_state(tmp_path, init_magnet_currents="5, -5"))
    assert not any("opposite sign" in i for i in info)
    # a sweep of both polarities can switch either initial state — no control-run hint
    info, _, _ = tui.build_summary(_state(tmp_path, init_magnet_currents="5",
                                          pulse_current_start_A=-5e-3, pulse_current_stop_A=5e-3))
    assert not any("opposite sign" in i for i in info)


def test_init_current_over_magnet_limit_is_blocked(tmp_path):
    _, _, errors = tui.build_summary(_state(tmp_path, init_magnet_currents="50"))
    assert any("magnet limit" in e for e in errors)
    _, _, errors = tui.build_summary(_state(tmp_path, init_magnet_currents="abc"))
    assert any("Init magnet current" in e for e in errors)


def test_summary_always_carries_joule_heating_reminder(tmp_path):
    info, _, _ = tui.build_summary(_state(tmp_path))
    assert any("Joule heating" in i for i in info)


def test_bidirectional_sweep_is_a_no_reset_control_or_a_loop(tmp_path):
    state = _state(tmp_path, amplitude_bidirectional=True, pulse_current_start_A=1e-3,
                   pulse_current_stop_A=3e-3, pulse_current_step_A=1e-3)
    assert state["pulse_current_list"] == pytest.approx([1e-3, 2e-3, 3e-3, 2e-3, 1e-3])
    info, _, _ = tui.build_summary(state)
    assert any("no reset" in i for i in info)
    state = _state(tmp_path, amplitude_bidirectional=True, pulse_current_start_A=-3e-3,
                   pulse_current_stop_A=3e-3, pulse_current_step_A=3e-3)
    assert state["pulse_current_list"] == pytest.approx([-3e-3, 0.0, 3e-3, 0.0, -3e-3], abs=1e-12)
    info, _, _ = tui.build_summary(state)
    assert any("hysteresis loop" in i for i in info)


def test_filename_preview_carries_the_init_key_axis_only_when_used(tmp_path):
    assert tui.compute_filename_preview(_state(tmp_path)) == "A_NNNN_HB3_NLSW_T300K_<timestamp>.csv"
    p = tui.compute_filename_preview(_state(tmp_path, init_magnet_currents="5, -5"))
    assert p == "A_NNNN_HB3_NLSW_T300K_<I_init A>_<timestamp>.csv (one file per initial state)"
    assert tui.compute_filename_preview(_state(tmp_path, device="")) is None


def test_build_plan_shapes(tmp_path):
    app = tui.NonlocalSwitchingApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(tmp_path, pulse_current_start_A=1e-3, pulse_current_stop_A=5e-3,
                                  pulse_current_step_A=2e-3))

    assert plan.pulse_currents_A == pytest.approx([1e-3, 3e-3, 5e-3])
    assert plan.series_values == [None] and not plan.uses_magnet
    assert plan.total_points == 3 + 1                       # + the baseline read
    assert plan.pulse_cfg.width_s == 1e-3 and plan.pulse_cfg.compliance_V == 5.0
    assert plan.read_cfg.sense_current_A == 1e-4 and plan.read_cfg.n_averages == 5
    assert plan.read_cfg.reversal_enabled is True
    assert plan.read_cfg.R_P_ohm is None and plan.read_cfg.R_AP_ohm is None
    assert plan.volt_cfg.visa_resource == "GPIB0::7::INSTR" and plan.volt_cfg.nplc == 5
    assert plan.source_visa == "GPIB0::20::INSTR"
    assert plan.temp_cfg is None
    assert plan.data_root == tmp_path


def test_build_plan_carries_the_reversal_toggle_into_the_read_and_the_header(tmp_path):
    app = tui.NonlocalSwitchingApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(tmp_path, reversal_enabled=False, n_averages=7))
    assert plan.read_cfg.reversal_enabled is False and plan.read_cfg.n_averages == 7
    assert plan.header_extra["reversal_enabled"] is False and plan.header_extra["n_averages"] == 7


def test_build_plan_one_run_per_initial_state(tmp_path):
    app = tui.NonlocalSwitchingApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(tmp_path, pulse_current_start_A=1e-3, pulse_current_stop_A=5e-3,
                                  pulse_current_step_A=2e-3, init_magnet_currents="5, -5",
                                  sweep_magnet_current_A=0.0))
    assert plan.series_values == [5.0, -5.0] and plan.uses_magnet
    assert plan.total_points == (3 + 1) * 2


def test_header_fields_carry_the_switching_current_and_init_state(tmp_path):
    app = tui.NonlocalSwitchingApp()
    app.data_root = tmp_path
    plan = app._build_plan(_state(tmp_path))
    ctx = SimpleNamespace(run_number=7, timestamp=__import__("datetime").datetime(2026, 9, 21, 12, 0),
                          sample="A", device="HB3")
    records = [{"pulse_current_A": 0.0, "switched": None, "temperature_1_K": None},
               {"pulse_current_A": 2e-3, "switched": False, "temperature_1_K": None},
               {"pulse_current_A": 3e-3, "switched": True, "temperature_1_K": None},
               {"pulse_current_A": -4e-3, "switched": True, "temperature_1_K": None}]
    extra = tui._run_extra(plan, 5.0, {"init_field_measured_mT": 12.5})

    h = tui.build_header_fields(plan, ctx, records, status="completed", comment="", extra=extra)

    assert h["type"] == "NLSW" and h["I_switch_A"] == "0.003, -0.004"
    assert h["init_magnet_current_A"] == 5.0 and h["init_field_measured_mT"] == 12.5
    assert h["sweep_magnet_current_A"] == 0.0 and h["pulse_width_s"] == 1e-3
    assert tui.build_header_fields(plan, ctx, records[:2], status="in_progress",
                                   comment="")["I_switch_A"] == ""
    assert tui._run_extra(plan, None, {})["sweep_magnet_current_A"] is None


def test_measurement_png_drops_the_v_even_panel_when_reversal_is_off(tmp_path):
    import matplotlib.pyplot as plt

    def panels(recs):
        figs, real_close = [], plt.close
        plt.close = lambda fig=None: (figs.append(fig), real_close(fig))
        try:
            tui._save_measurement_png(recs, tmp_path / "r.png")
        finally:
            plt.close = real_close
        return len(figs[0].axes)

    rec = {"pulse_current_A": 1e-3, "nl_resistance_ohm": 0.1, "sense_current_A": 1e-4,
           "switched": None, "init_magnet_current_A": None}
    assert panels([{**rec, "voltage_even_V": 2e-6, "reversal_enabled": True}]) == 2
    assert panels([{**rec, "voltage_even_V": None, "reversal_enabled": False}]) == 1


def test_measurement_png_annotates_the_runs_own_values(tmp_path):
    recs = [{"pulse_current_A": 0.0, "nl_resistance_ohm": -0.5, "voltage_even_V": 2e-6,
             "sense_current_A": 1e-4, "switched": None, "init_magnet_current_A": 5.0},
            {"pulse_current_A": 3e-3, "nl_resistance_ohm": 0.5, "voltage_even_V": 2e-6,
             "sense_current_A": 1e-4, "switched": True, "init_magnet_current_A": 5.0}]
    png = tmp_path / "run.png"
    tui._save_measurement_png(recs, png, comment="looks switched")
    assert png.exists() and png.stat().st_size > 0
    tui._save_measurement_png([], tmp_path / "empty.png")       # no records -> nothing written
    assert not (tmp_path / "empty.png").exists()


# ─────────────────────────────────────────────────────────────────────────────
# RunScreen.do_run end to end — fake 6221/2182A/magnet in a real Textual pilot
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reversal", [True, False])
def test_run_screen_saves_one_run_per_initial_state(tmp_path, monkeypatch, reversal):
    import asyncio

    import pandas as pd
    from textual.app import App
    from textual.screen import Screen
    from test_sot_nonlocal_switching import _Fake6221, _FakeVoltmeter

    import instruments.tui_common as tui_common
    from instruments import run_index
    import sot.sot_nonlocal_switching as ns
    from instruments.data_naming import ensure_sample, read_raw

    src = _Fake6221(i_c=3e-3)
    volt = _FakeVoltmeter(src)
    inits: list[float] = []

    def fake_init(magnet, cfg, gaussmeter, gcfg, init_A, hold_A, tol, stop):
        inits.append(init_A)
        src.state = -1 if init_A > 0 else 1          # +5: switchable state, -5: already at the top
        return {"init_field_measured_mT": 10.0 * init_A}

    def no_live_plot(*a, **k):
        raise RuntimeError("no plot window in tests")

    monkeypatch.setattr(tui, "connect", lambda visa, compliance, delay: src)
    monkeypatch.setattr(tui, "connect_voltmeter", lambda cfg: volt)
    monkeypatch.setattr(tui, "shutdown_source", lambda s: None)
    monkeypatch.setattr(tui, "connect_magnet", lambda cfg: "magnet")
    monkeypatch.setattr(tui, "connect_gaussmeter", lambda cfg: "gaussmeter")
    monkeypatch.setattr(tui, "shutdown_magnet", lambda m, cfg: None)
    monkeypatch.setattr(tui, "shutdown_gaussmeter", lambda g: None)
    monkeypatch.setattr(tui, "initialize_with_field", fake_init)
    monkeypatch.setattr(tui_common, "start_live_plot", no_live_plot)
    monkeypatch.setattr(tui_common, "StatusCommentScreen", Screen)   # the shared good/short/open dialog is not under test
    monkeypatch.setattr(ns, "read_field_mT", lambda gm, cfg: 0.5)

    ensure_sample(tmp_path, "A", create=True)
    app_for_plan = tui.NonlocalSwitchingApp()
    app_for_plan.data_root = tmp_path
    plan = app_for_plan._build_plan(_state(
        tmp_path, pulse_current_start_A=1e-3, pulse_current_stop_A=5e-3, pulse_current_step_A=2e-3,
        init_magnet_currents="5, -5", delay_after_pulse_s=0.0, source_delay_s=0.0, n_averages=3,
        reversal_enabled=reversal,
        pulse_width_s=1e-4, R_P_ohm=-0.5, R_AP_ohm=0.5))

    screen = tui.RunScreen(plan)

    class Host(App):
        def on_mount(self):
            self.push_screen(screen)

    async def go():
        async with Host().run_test(size=(200, 60)) as pilot:
            for _ in range(400):
                await pilot.pause(0.05)
                if not screen._measurement_running:
                    break
            assert not screen._measurement_running

    asyncio.run(go())

    assert inits == [5.0, -5.0]
    # the session is in the shared run history (runs.db), like a web run
    (hist,) = run_index.recent_runs()
    assert (hist["suite"], hist["status"], hist["point_count"]) == ("SOT", "completed", 8)
    assert hist["sample"] == "A" and hist["finished_at"]
    raws = sorted((tmp_path / "A" / "raw").glob("*.csv"))
    assert len(raws) == 2 and all("_NLSW_" in p.name for p in raws)
    first, second = (read_raw(p) for p in raws)
    off = 0.0 if reversal else 0.02          # plain read keeps the fake's 2 uV thermal offset (2e-6 / 1e-4 A)
    # init +5: baseline at P, switches at 3 mA-or-above (the 3 mA step)
    assert first["nl_resistance_ohm"].round(3).tolist() == [-0.5 + off, -0.5 + off, 0.5 + off, 0.5 + off]
    assert first["init_magnet_current_A"].tolist() == [5.0] * 4
    assert first["reversal_enabled"].astype(str).eq(str(reversal)).all()
    assert (first["voltage_even_V"].notna().all() if reversal else first["voltage_even_V"].isna().all())
    # init -5: already at the top state, nothing to switch
    assert second["nl_resistance_ohm"].round(3).tolist() == [0.5 + off] * 4
    assert (second["switched"] != True).all()  # noqa: E712 - object column of True/False/blank

    index = pd.read_csv(tmp_path / "A" / "index.csv")
    assert len(index) == 2 and set(index["type"]) == {"NLSW"}
    assert sorted(p.name for p in (tmp_path / "A" / "proc").glob("*_NL_vs_pulse.png")) \
        == sorted(f"A_{c.run_str}_HB3_NLSW_NL_vs_pulse.png" for c in screen._run_contexts)
    assert src.enabled is False
