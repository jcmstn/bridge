"""Run-cost model for the dual-harmonic, dual-harmonic-6221 and noise-spectrum TUIs.

Pure: builds states, calls run_costs() / build_summary() / _build_plan(); no hardware.
Expected values are composed from the run_time knobs and instrument helpers, not
hard-coded seconds, so tuning a knob never breaks these -- dropping a term does.
"""

from __future__ import annotations

import math

import pytest

import mfli.mfli_dual_harmonic_6221_tui as t6221
import mfli.mfli_dual_harmonic_tui as tharm
import mfli.mfli_noise_spectrum_tui as tnoise
from dc.dc_sweep_utils import build_segmented_sweep
from instruments import run_time as rt
from instruments.data_naming import ensure_sample
from instruments.kepco_magnet import MagnetConfig, magnet_move_s
from instruments.keithley6221 import ac_source_restart_s
from instruments.lakeshore475 import GaussmeterConfig, read_field_s
from instruments.mfli_daq import acquire_s
from mfli.mfli_dual_harmonic import phase_cal_s
from mfli.mfli_dual_harmonic_6221 import extref_lock_s
from mfli.mfli_noise_spectrum import AcquisitionConfig

_MAGNET = MagnetConfig(ramp_step_A=0.1, ramp_delay_s=0.05)
_GAUSS = GaussmeterConfig(n_averages=10, read_delay_s=0.05)


def _hstate(**overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886",
        daq_host="localhost", daq_port=8004,
        frequency_Hz=317.3, amplitude_V=0.1, series_R_ohm=10000.0,
        time_constant_1f_s=0.3, order_1f=4, sinc_filter_1f=True,
        time_constant_2f_s=0.3, order_2f=4, sinc_filter_2f=True,
        differential=True, ac_coupling=True,
        input_range_1f_V=1.0, input_range_2f_V=1.0, sample_rate_Hz=857.0,
        settling_time_s=15.0, n_averages=50,
        device="HB3", cooldown="3", temperature_setpoint_K=300.0,
        enable_sweep=False,
        visa_resource="GPIB0::6::INSTR", current_limit_A=35.0,
        voltage_compliance_V=15.0, ramp_step_A=0.1, ramp_delay_s=0.05,
        sweep_rows_parsed=[(-20.0, 20.0, 21)],
        gaussmeter_visa_resource="GPIB0::12::INSTR", gaussmeter_n_averages=10,
        gaussmeter_read_delay_s=0.05, field_settle_tolerance_mT=0.02, enable_temperature=False,
        temperature_visa_resource="", temperature_sensor_uids="",
        enable_phase_cal=False, phase_cal_current_A=None,
        phase_cal_n_averages=20, phase_cal_max_iterations=5,
        hall_bar_length_um=None, hall_bar_width_um=None,
        hall_bar_thickness_nm=None, field_theta_deg=None, field_phi_deg=None,
        sample="A",
    )
    base.update(overrides)
    return base


def _h6state(**overrides) -> dict:
    base = _hstate(
        ac_visa_resource="GPIB0::20::INSTR", ac_compliance_V=2.0, phasemarker_line=1,
        amplitude_values="1e-7", amplitude_list=[1e-7], amplitude_parse_error=None,
        measure_rxx=False,
        leader_extref_index=0, leader_aux_input_ch=0, leader_osc_index=0, leader_pll_demod_index=1,
        leader_automode=4,
        follower_extref_index=0, follower_aux_input_ch=0, follower_osc_index=0,
        follower_pll_demod_index=1, follower_automode=4,
        extref_lock_timeout_s=5.0,
    )
    base.update(overrides)
    return base


def _nstate(**overrides) -> dict:
    base = dict(
        leader_device="dev7885", follower_device="dev7886",
        daq_host="localhost", daq_port=8004,
        ac_visa_resource="GPIB0::20::INSTR", frequency_Hz=317.3,
        amplitude_values="1e-4", amplitude_list=[1e-4], amplitude_parse_error=None,
        ac_compliance_V=2.0, phasemarker_line=1, extref_lock_timeout_s=5.0,
        leader_extref_index=0, leader_aux_input_ch=0, leader_osc_index=0,
        leader_pll_demod_index=1, leader_automode=4,
        follower_extref_index=0, follower_aux_input_ch=0, follower_osc_index=0,
        follower_pll_demod_index=1, follower_automode=4,
        input_range_1f_V=1.0, input_range_2f_V=1.0, sample_rate_Hz=13389.0,
        time_constant_s=3e-5, duration_s=30.0, also_measure_off=True,
        thermal_R_ohm=10_000.0, thermal_T_K=293.0,
        device="HB3", cooldown="", sample="A",
    )
    base.update(overrides)
    return base


def _pair_s(state: dict) -> float:
    return max(acquire_s(state["time_constant_1f_s"], state["n_averages"], state["sample_rate_Hz"]),
               acquire_s(state["time_constant_2f_s"], state["n_averages"], state["sample_rate_Hz"]))


# ── mfli_dual_harmonic ───────────────────────────────────────────────────────

def test_harmonic_single_point_has_no_magnet_term() -> None:
    state = _hstate()
    rc = tharm.run_costs(state)
    assert len(rc.points) == 1 and "magnet" not in rc.parts and "field read" not in rc.parts
    assert rc.total_s == pytest.approx(
        15.0 + _pair_s(state) + 3 * rt.GPIB_TXN_S + rt.POINT_OVERHEAD_S + rt.PER_RUN_S + rt.MDS_SYNC_S)


def test_harmonic_field_sweep_counts_the_magnet_and_beats_the_old_estimate() -> None:
    state = _hstate(enable_sweep=True)
    currents = build_segmented_sweep(state["sweep_rows_parsed"], bidirectional=True)
    rc = tharm.run_costs(state, currents)
    assert len(rc.points) == len(currents) == 41
    old_estimate = 41 * (15.0 + 0.9)                     # settle + window: the sidebar's old number
    assert rc.total_s > 770.0 > old_estimate             # the audit's zero-latency floor
    # an interior 2 A hop: settle + pair window + overhead + gaussmeter + one magnet move
    hop_typ, _ = magnet_move_s(2.0, _MAGNET)
    assert rc.points[5] == pytest.approx(
        15.0 + _pair_s(state) + 3 * rt.GPIB_TXN_S + rt.POINT_OVERHEAD_S + read_field_s(_GAUSS) + hop_typ)
    # the first point carries the 0 -> -20 A ramp; the return to 0 A is teardown, not on the bar
    assert rc.points[0] > rc.points[5] + 10.0 and rc.tail_s > 10.0
    assert rt.progress_total(rc, 41) == pytest.approx(rc.total_s - rc.tail_s)
    assert rc.worst_extra_s > 41 * 25.0                   # every move can run into the 30 s settle timeout


def test_harmonic_multi_row_sweep_matches_plan_length(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tharm, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    app = tharm.MFLIDualHarmonicApp()
    app.data_root = tmp_path
    plan = app._build_plan(_hstate(enable_sweep=True,
                                   sweep_rows_parsed=[(-1.0, 1.0, 10), (1.0, 10.0, 10)]))
    assert plan.total_points == 37 == len(plan.run_cost.points)


def test_harmonic_phase_cal_adds_a_block_and_a_magnet_move() -> None:
    plain = tharm.run_costs(_hstate())
    with_cal = tharm.run_costs(_hstate(enable_phase_cal=True))
    assert with_cal.total_s - plain.total_s == pytest.approx(phase_cal_s(0.3, 0.3, 20, 5, 857.0))
    assert with_cal.parts["phase cal"] > 2 * 5 * 0.3 + 3 * acquire_s(0.3, 20, 857.0)  # sleeps + windows
    state = _hstate(enable_sweep=True, enable_phase_cal=True, phase_cal_current_A=20.0)
    currents = build_segmented_sweep(state["sweep_rows_parsed"], bidirectional=True)
    assert (tharm.run_costs(state, currents).parts["phase cal"]
            > tharm.run_costs(_hstate(enable_sweep=True, enable_phase_cal=True), currents).parts["phase cal"])


def test_harmonic_summary_and_bad_inputs_do_not_crash(tmp_path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    info, _, errors = tharm.build_summary(_hstate(data_dir=str(tmp_path)))
    assert not errors and any(i.startswith("Run time: ≈ ") for i in info)
    info, _, _ = tharm.build_summary(_hstate(data_dir=str(tmp_path), enable_sweep=True))
    assert any(i.startswith("Run time: ≈ ") for i in info)
    # a half-typed form (0 sample rate, 0 ramp step) must never take the live sidebar down
    tharm.build_summary(_hstate(data_dir=str(tmp_path), sample_rate_Hz=0.0, ramp_step_A=0.0, enable_sweep=True))


# ── mfli_dual_harmonic_6221 ──────────────────────────────────────────────────

def test_6221_points_span_every_amplitude_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(t6221, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "A", create=True)
    plan = t6221.build_plan(_h6state(enable_sweep=True, amplitude_values="1e-7, 2e-7, 3e-7",
                                     amplitude_list=[1e-7, 2e-7, 3e-7]), tmp_path)
    assert plan.total_points == 41 and plan.total_files == 3
    assert len(plan.run_cost.points) == plan.total_points * plan.total_files == 123


def test_6221_per_amplitude_costs_are_multiplied_in() -> None:
    state = _h6state(amplitude_list=[1e-7, 2e-7, 3e-7])
    rc = t6221.run_costs(state)                            # single point per amplitude
    lock_typ, lock_worst = extref_lock_s(5.0)
    assert len(rc.points) == 3
    assert rc.parts["6221 re-arm + ExtRef"] == pytest.approx(3 * (ac_source_restart_s() + lock_typ))
    assert rc.parts["per-file"] == pytest.approx(3 * rt.PER_FILE_S)
    assert rc.parts["settle"] == pytest.approx(3 * 15.0)
    assert rc.worst_extra_s == pytest.approx(3 * (lock_worst - lock_typ))   # ExtRef timeout on each
    # every later file re-arms the 6221 and re-locks too: it differs from file 1 only by the one-off connect + MDS
    assert rc.points[1] == pytest.approx(rc.points[0] - rt.PER_RUN_S - rt.MDS_SYNC_S)
    one = t6221.run_costs(_h6state())
    assert rc.total_s == pytest.approx(one.total_s + 2 * rc.points[1])   # 2 more full files


def test_6221_sweep_is_repeated_per_amplitude_with_magnet_terms() -> None:
    state = _h6state(enable_sweep=True, amplitude_list=[1e-7, 2e-7])
    currents = build_segmented_sweep(state["sweep_rows_parsed"], bidirectional=True)
    rc = t6221.run_costs(state, currents)
    assert len(rc.points) == 2 * 41
    assert rc.parts["settle"] == pytest.approx(2 * 41 * 15.0)
    # amplitude 2 starts its sweep from where amplitude 1 ended (-20 A), so its first hop is ~0, not 0 -> -20 A
    m20, m0 = magnet_move_s(20.0, _MAGNET)[0], magnet_move_s(0.0, _MAGNET)[0]
    assert rc.points[0] - rc.points[41] == pytest.approx(rt.PER_RUN_S + rt.MDS_SYNC_S + m20 - m0)
    assert "magnet" not in t6221.run_costs(_h6state(amplitude_list=[1e-7, 2e-7])).parts


def test_6221_summary_lines(tmp_path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    info, _, errors = t6221.build_summary(_h6state(data_dir=str(tmp_path)))
    assert not errors and any(i.startswith("Run time: ≈ ") for i in info)
    info, _, _ = t6221.build_summary(_h6state(data_dir=str(tmp_path), enable_sweep=True,
                                              amplitude_list=[1e-7, 2e-7]))
    assert any(i.startswith("Run time: ≈ ") for i in info)
    t6221.build_summary(_h6state(data_dir=str(tmp_path), sample_rate_Hz=0.0, ramp_step_A=0.0, enable_sweep=True))


def test_extref_and_phase_cal_helpers_arithmetic() -> None:
    typ, worst = extref_lock_s(5.0)
    assert typ == pytest.approx(2 * 12 * rt.GPIB_TXN_S + 2 * rt.LOCK_TYP_S)
    assert worst == pytest.approx(2 * 12 * rt.GPIB_TXN_S + 2 * 5.0)
    one_round = phase_cal_s(0.3, 0.3, 20, 1, 857.0)         # max_iterations=1 -> no settle sleeps in the null
    assert phase_cal_s(0.3, 0.3, 20, 5, 857.0) > one_round
    assert one_round > 3 * acquire_s(0.3, 20, 857.0)         # leader window + snapshot + follower window


# ── mfli_noise_spectrum ──────────────────────────────────────────────────────

def test_noise_steps_and_recording_time(tmp_path) -> None:
    app = tnoise.MFLINoiseSpectrumApp()
    app.data_root = tmp_path
    plan = app._build_plan(_nstate(data_dir=str(tmp_path)))
    rc = plan.run_cost
    assert len(rc.points) == plan.total_steps == 4          # ON/OFF x leader/follower
    assert rc.parts["recording"] == pytest.approx(4 * 30.0)
    old_estimate = 30.0 * 2 * 2                              # what the sidebar said: 120 s
    assert rc.total_s > old_estimate + rt.PER_RUN_S          # connect, MDS, lock, saves are not free
    assert len(tnoise.run_costs(_nstate(also_measure_off=False)).points) == 2


def test_noise_per_amplitude_costs_and_tail() -> None:
    state = _nstate(amplitude_list=[1e-4, 2e-4])
    rc = tnoise.run_costs(state)
    lock_typ, lock_worst = extref_lock_s(5.0)
    assert len(rc.points) == 8
    assert rc.parts["6221 + ExtRef"] == pytest.approx(2 * (ac_source_restart_s() + lock_typ))
    assert rc.parts["save"] == pytest.approx(2 * 4 * rt.PER_FILE_S)     # one file set per amplitude
    assert rc.tail_s == pytest.approx(4 * rt.PER_FILE_S)                 # the last save follows the last spectrum
    assert rc.worst_extra_s == pytest.approx(2 * (lock_worst - lock_typ))
    n_chunks = math.ceil(30.0 / AcquisitionConfig().poll_chunk_s)
    assert rc.parts["overhead"] == pytest.approx(
        8 * (2 * n_chunks * rt.GPIB_TXN_S + rt.ACQ_OVERHEAD_S + rt.POINT_OVERHEAD_S))


def test_noise_summary_shows_the_estimate(tmp_path) -> None:
    ensure_sample(tmp_path, "A", create=True)
    info, _, errors = tnoise.build_summary(_nstate(data_dir=str(tmp_path)))
    assert not errors and any(i.startswith("Run time: ≈ ") for i in info)
