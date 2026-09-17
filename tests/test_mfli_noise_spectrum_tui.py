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
        ac_visa_resource="GPIB0::20::INSTR", frequency_Hz=317.3, amplitude_A=1e-4,
        ac_compliance_V=2.0, phasemarker_line=1, extref_lock_timeout_s=5.0,
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


def test_build_summary_flags_mistyped_current(tmp_path: Path) -> None:
    state = _state(tmp_path, amplitude_A=5.0)  # way past the safety ceiling
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
