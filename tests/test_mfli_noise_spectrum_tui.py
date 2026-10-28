"""
Plan-purity + summary tests for mfli/mfli_noise_spectrum_tui.py.

No hardware and no running Textual app loop needed -- _build_plan() only
builds dataclasses (no allocate_run() here; this suite saves all its files
in one shot at the end of the run, not incrementally -- see
mfli_noise_spectrum.save_results()), and build_summary()/
compute_filename_preview() are pure functions over a state dict.
"""

from __future__ import annotations

from pathlib import Path

import mfli.mfli_noise_spectrum_tui as tui
from instruments.data_naming import ensure_sample


def _state(data_dir: Path, **overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886",
        daq_host="localhost", daq_port=8004,
        ac_visa_resource="GPIB0::20::INSTR", frequency_Hz=317.3,
        amplitude_values="1e-4", amplitude_list=[1e-4], amplitude_parse_error=None,
        ac_compliance_V=2.0, ac_offset_A=0.0, phasemarker_line=1, extref_lock_timeout_s=5.0,
        leader_harmonic=1, leader_measure_rxx=False, follower_harmonic=2, measure_rxx=False,
        differential=True, ac_coupling=True,
        leader_extref_index=0, leader_aux_input_ch=0, leader_osc_index=0,
        leader_pll_demod_index=1, leader_automode=4,
        follower_extref_index=0, follower_aux_input_ch=0, follower_osc_index=0,
        follower_pll_demod_index=1, follower_automode=4,
        input_range_1f_V=1.0, input_range_2f_V=1.0, sample_rate_Hz=13389.0,
        time_constant_s=3e-5, duration_s=30.0, also_measure_off=True,
        thermal_R_ohm=10_000.0, thermal_T_K=293.0,
        device="HB3", cooldown="", sample="A", data_dir=str(data_dir),
    )
    base.update(overrides)
    return base


def test_build_plan_shapes_configs(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    app = tui.MFLINoiseSpectrumApp()
    app.data_root = tmp_path

    plan = app._build_plan(_state(tmp_path))
    assert plan.ac_cfg.amplitude_A == 1e-4
    assert plan.ac_cfg.frequency_Hz == 317.3
    assert len(plan.demod_cfgs) == 2
    assert plan.demod_cfgs[0].harmonic == 1
    assert plan.demod_cfgs[1].harmonic == 2
    assert plan.ref_cfg.thermal_R_ohm == 10_000.0
    assert plan.also_measure_off is True
    assert plan.total_steps == 4  # 2 channels x 2 passes (ON + OFF)

    plan_on_only = app._build_plan(_state(tmp_path, also_measure_off=False))
    assert plan_on_only.total_steps == 2


def test_build_plan_follows_chosen_harmonics(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    app = tui.MFLINoiseSpectrumApp()
    app.data_root = tmp_path

    plan = app._build_plan(_state(tmp_path, leader_harmonic=2, follower_harmonic=1,
                                    measure_rxx=True, differential=False, ac_offset_A=1e-5))
    lead, fol = plan.demod_cfgs
    assert (lead.harmonic, fol.harmonic) == (2, 1)
    assert (lead.label, fol.label) == ("Leader 2f", "Follower R_xx (1f)")
    assert lead.differential is False and fol.differential is False
    assert plan.ac_cfg.offset_A == 1e-5


def test_mains_warning_uses_chosen_harmonic(tmp_path: Path) -> None:
    # 25 Hz: 2f lands on 50 Hz mains, 1f doesn't
    _, warnings, _ = tui.build_summary(_state(tmp_path, frequency_Hz=25.0,
                                              leader_harmonic=1, follower_harmonic=1))
    assert not any("mains" in w.lower() for w in warnings)
    _, warnings, _ = tui.build_summary(_state(tmp_path, frequency_Hz=25.0, leader_harmonic=2))
    assert any("Leader 2f" in w for w in warnings)


def test_build_summary_flags_offset_past_ceiling(tmp_path: Path) -> None:
    _, _, errors = tui.build_summary(_state(tmp_path, ac_offset_A=0.02))
    assert any("DC offset" in e for e in errors)


def test_harm6_copy_takes_wiring_not_noise_settings() -> None:
    saved = {"leader_harmonic": 2, "follower_harmonic": 3, "input_range_1f_V": "0.1",
             "frequency_Hz": "123.4", "time_constant_1f_s": "0.3", "duration_s": "999",
             "sample_rate_Hz": "857.0", "ac_source": "6221"}
    copied = tui.harm6_copy(saved)
    assert copied == {"leader_harmonic": 2, "follower_harmonic": 3,
                      "input_range_1f_V": "0.1", "frequency_Hz": "123.4"}
    # legacy R_xx form (no harmonic selects) meant follower at 1f
    assert tui.harm6_copy({"measure_rxx": True})["follower_harmonic"] == 1


def test_build_plan_multiple_amplitudes_scales_total_steps(tmp_path: Path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    app = tui.MFLINoiseSpectrumApp()
    app.data_root = tmp_path

    plan = app._build_plan(_state(tmp_path, amplitude_values="1e-4, 2e-4",
                                    amplitude_list=[1e-4, 2e-4]))
    assert plan.amplitudes_A == [1e-4, 2e-4]
    assert plan.ac_cfg.amplitude_A == 1e-4
    assert plan.total_steps == 8  # 2 channels x 2 passes x 2 amplitudes


def test_build_summary_flags_mistyped_current(tmp_path: Path) -> None:
    state = _state(tmp_path, amplitude_values="5.0", amplitude_list=[5.0])  # past the safety ceiling
    _, _, errors = tui.build_summary(state)
    assert any("Excitation current" in e for e in errors)


def test_build_summary_flags_pll_demod_collision(tmp_path: Path) -> None:
    state = _state(tmp_path, leader_pll_demod_index=0)
    _, _, errors = tui.build_summary(state)
    assert any("Leader PLL" in e for e in errors)


def test_build_summary_flags_mains_adjacent_frequency(tmp_path: Path) -> None:
    state = _state(tmp_path, frequency_Hz=50.0)
    _, warnings, _ = tui.build_summary(state)
    assert any("mains" in w.lower() for w in warnings)


def test_compute_filename_preview(tmp_path: Path) -> None:
    state = _state(tmp_path)
    assert tui.compute_filename_preview({**state, "sample": ""}) is None

    preview = tui.compute_filename_preview(state)
    assert preview is not None and "×4 files" in preview

    preview_on_only = tui.compute_filename_preview({**state, "also_measure_off": False})
    assert "×2 files" in preview_on_only

    preview_multi_amp = tui.compute_filename_preview(
        {**state, "amplitude_list": [1e-4, 2e-4]})
    assert "×8 files" in preview_multi_amp


def test_run_screen_mounts_without_base_results_table(tmp_path: Path, monkeypatch) -> None:
    """Textual dispatches Mount to every class in the MRO; the base
    MeasurementRunScreen.on_mount queries #results_table, which this screen
    doesn't compose -- it must not run (it crashed the app on Start)."""
    import asyncio
    from types import SimpleNamespace

    from textual.app import App

    from instruments import run_index

    plan = SimpleNamespace(run_cost=None, total_steps=1, data_root=tmp_path,
                           sample="A", device="HB3", header_extra={})
    monkeypatch.setattr(tui.RunScreen, "do_run", lambda self: None)
    screen = tui.RunScreen(plan)

    class Host(App):
        def on_mount(self):
            self.push_screen(screen)

    async def go():
        async with Host().run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.05)
            assert screen.is_mounted

    asyncio.run(go())
    assert len(run_index.recent_runs()) == 1
