"""
Per-MFLI harmonic + R_xx naming in the dual-harmonic program (HARM / HARM6).

The compat guard: leader 1f + follower 2f, and follower 1f with R_xx on, must
save exactly the columns they did before the harmonic selects existed (the
literal lists below). Hardware-free -- acquire_averaged_pair is stubbed and a
fake DAQ answers the few node reads run_measurement() makes.
"""

from __future__ import annotations

import mfli.mfli_dual_harmonic as harm_engine
import mfli.mfli_dual_harmonic_6221 as six_engine
import mfli.mfli_dual_harmonic_6221_tui as six_tui
import mfli.mfli_dual_harmonic_tui as harm_tui
from instruments.data_naming import ensure_sample
from instruments.keithley6221 import ACSourceConfig
from mfli.mfli_dual_harmonic import AcquisitionConfig, DemodConfig, MeasurementPoint, OutputConfig
from mfli.mfli_dual_harmonic_6221 import ExtRefConfig

_DEMOD = ["X_V", "Y_V", "R_V", "theta_deg", "R_sem_V", "n_samples", "overload"]
_GEOM = ["hall_bar_length_um", "hall_bar_width_um", "hall_bar_thickness_nm",
         "field_theta_deg", "field_phi_deg"]
_DEMOD_META = ["demod1_time_constant_s", "demod1_filter_order", "demod1_ref_phase_deg",
               "demod2_time_constant_s", "demod2_filter_order", "demod2_ref_phase_deg"]
_EXC = ["excitation_frequency_Hz", "excitation_current_A_peak", "excitation_current_A_rms",
        "excitation_current_convention", "demod_output_convention"]
_POINT = ["magnet_current_A", "magnet_field_mT", "temperature_1_K", "temperature_2_K"]

HARM_COLUMNS = (["point_index", "timestamp", "mds_synced", *_POINT]
                + [f"1f_{c}" for c in _DEMOD] + [f"2f_{c}" for c in _DEMOD]
                + ["demod2_phase_null_1f_deg", *_EXC, *_DEMOD_META, *_GEOM])


def _harm6_columns(follower: str) -> list[str]:
    return (["point_index", "timestamp", "mds_synced",
             "leader_reference_locked", "follower_reference_locked", *_POINT]
            + [f"1f_{c}" for c in _DEMOD] + [f"{follower}_{c}" for c in _DEMOD]
            + ["measure_rxx", "demod2_phase_null_1f_deg", *_EXC, *_DEMOD_META, *_GEOM])


class _DAQ:
    def getDouble(self, path):
        return 317.3 if path.endswith("/freq") else 0.0

    def getInt(self, path):
        return 1


def _reading(r: float) -> dict:
    return {"x_mean": r, "y_mean": 0.0, "r_mean": r, "theta_mean": 0.0,
            "r_sem": 0.0, "n_samples": 5, "overload": False}


def _demods():
    return (DemodConfig(device="dev1", demod_index=0, harmonic=1),
            DemodConfig(device="dev2", demod_index=0, harmonic=2))


def _run_harm(monkeypatch, **labels):
    monkeypatch.setattr(harm_engine, "acquire_averaged_pair", lambda *a: (_reading(1.0), _reading(2.0)))
    d1, d2 = _demods()
    return harm_engine.run_measurement(
        _DAQ(), OutputConfig(), d1, d2, AcquisitionConfig(settling_time_s=0.0),
        [MeasurementPoint()], write_csv=lambda records: None, **labels)


def _run_six(monkeypatch, **labels):
    monkeypatch.setattr(six_engine, "acquire_averaged_pair", lambda *a: (_reading(1.0), _reading(2.0)))
    d1, d2 = _demods()
    return six_engine.run_measurement(
        _DAQ(), ACSourceConfig(), ExtRefConfig(device="dev1"), ExtRefConfig(device="dev2"),
        d1, d2, AcquisitionConfig(settling_time_s=0.0),
        [MeasurementPoint()], write_csv=lambda records: None, **labels)


def test_harm_default_columns_unchanged(monkeypatch):
    assert list(_run_harm(monkeypatch).columns) == HARM_COLUMNS


def test_harm6_default_columns_unchanged(monkeypatch):
    df = _run_six(monkeypatch)
    assert list(df.columns) == _harm6_columns("2f")
    assert df["measure_rxx"].tolist() == [False]


def test_harm6_rxx_columns_unchanged(monkeypatch):
    df = _run_six(monkeypatch, demod2_label="rxx_1f")
    assert list(df.columns) == _harm6_columns("rxx_1f")
    assert df["measure_rxx"].tolist() == [True]


def test_harm6_non_rxx_follower_harmonic_is_not_flagged_rxx(monkeypatch):
    df = _run_six(monkeypatch, demod1_label="2f", demod2_label="3f")
    assert "2f_R_V" in df and "3f_R_V" in df and "1f_R_V" not in df
    assert df["measure_rxx"].tolist() == [False]


def test_harm_new_combo_columns_follow_the_prefix_rule(monkeypatch):
    df = _run_harm(monkeypatch, demod1_label="rxx_1f", demod2_label="3f")
    assert list(df.columns) == [c.replace("1f_", "rxx_1f_", 1) if c.startswith("1f_")
                                else c.replace("2f_", "3f_", 1) if c.startswith("2f_") else c
                                for c in HARM_COLUMNS]


def test_default_form_plans_keep_the_classic_prefixes(tmp_path, monkeypatch):
    from test_run_costs_mfli_harmonic import _h6state, _hstate
    monkeypatch.setattr(harm_tui, "_DEFAULT_DATA_DIR", tmp_path)
    ensure_sample(tmp_path, "S1", create=True)
    classic = (("1f", "1f"), ("2f", "2f"))
    old_rxx = (("1f", "1f"), ("rxx_1f", "R_xx (1f)"))
    for mode, state in (("mfli", _hstate), ("6221", _h6state)):
        plan = harm_tui.build_plan(state(ac_source=mode, sample="S1", device="HB3"), tmp_path)
        assert six_tui.plan_naming(plan) == classic
        plan = harm_tui.build_plan(state(ac_source=mode, sample="S1", device="HB3",
                                         follower_harmonic=1, measure_rxx=True), tmp_path)
        assert six_tui.plan_naming(plan) == old_rxx


def test_colliding_prefixes_block_the_run_in_both_modes(tmp_path):
    from test_run_costs_mfli_harmonic import _h6state, _hstate
    for mode, state in (("mfli", _hstate), ("6221", _h6state)):
        base = dict(ac_source=mode, data_dir=str(tmp_path), sample="S1", device="HB3")
        _, _, errors = harm_tui.build_summary(state(**base))
        assert not any("would both save" in e for e in errors)
        for clash in (dict(follower_harmonic=1), dict(leader_harmonic=2),
                      dict(follower_harmonic=1, measure_rxx=True, leader_measure_rxx=True)):
            _, _, errors = harm_tui.build_summary(state(**base, **clash))
            assert any("would both save" in e for e in errors), (mode, clash)


def test_old_rxx_settings_load_as_follower_1f():
    assert six_tui.migrate_settings({"measure_rxx": True}) == {"measure_rxx": True, "follower_harmonic": 1}
    assert six_tui.migrate_settings({"measure_rxx": True, "follower_harmonic": 3})["follower_harmonic"] == 3
    assert "follower_harmonic" not in six_tui.migrate_settings({"measure_rxx": False})
