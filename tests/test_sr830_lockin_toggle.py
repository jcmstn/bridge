"""
The "Lock-in: MFLI | SR830" toggle: instruments/sr830.py's form mapping and
SR830Read bundle, the engines' ``lockin=`` path (HARM / HARM6 / SOT2H /
SOT1I — same columns, no daq touched), and the forms' plan/summary for SR830
mode. The MFLI path is covered, unchanged, by the existing run-loop tests.

Hardware-free: fake SR830s (test_sr830._FakeSR830) and a duck-typed reader.
"""

from __future__ import annotations

import math
import threading

import pytest

import mfli.mfli_dual_harmonic as harm
import mfli.mfli_dual_harmonic_6221 as harm6
import mfli.mfli_dual_harmonic_tui as harm_tui
import sot.sot_pulsed_switching_2h as sot2h
import sot.sot_pulsed_switching_6221 as sot1i
import sot.sot_pulsed_switching_tui as sot_tui
from instruments import sr830
from instruments.data_naming import ensure_sample
from instruments.keithley4200a import PMUPulseConfig
from test_sr830 import _FakeSR830
from test_sot_pulsed_switching_2h_run_loops import _Fake6221AC, _FakeKXCI

_NULL_WRITER = lambda records: None      # noqa: E731


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    for mod in (sr830, harm, harm6):
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)


# ── instruments/sr830.py ─────────────────────────────────────────────────────

def _form_cfg(**kw):
    args = dict(harmonic=2, frequency_Hz=317.3, time_constant_s=0.25, order=3, sinc_filter=True,
                differential=False, ac_coupling=False, sensitivity_V=1.5e-3, sample_rate_Hz=857.0)
    args.update(kw)
    return sr830.config_from_form("GPIB0::8::INSTR", **args)


def test_config_from_form_maps_and_snaps_like_the_unit():
    cfg = _form_cfg()
    assert cfg.filter_slope_dB == 18 and cfg.sync_filter is True
    assert cfg.input_config == "A" and cfg.coupling == "DC"
    assert cfg.reference == "external" and cfg.ext_slope == "ttl_rising"
    assert cfg.time_constant_s == 0.3          # 0.25 snapped up to the next OFLT value
    assert cfg.sensitivity_V == 2e-3           # 1.5 mV snapped up to 2 mV
    assert cfg.sample_rate_Hz == 512.0         # buffer cap
    sr830.validate(cfg)


@pytest.mark.parametrize("kw", [{"order": 5}, {"harmonic": 2, "frequency_Hz": 60e3},
                                {"reference": "internal", "sine_amplitude_V": 6.0}])
def test_config_from_form_out_of_range_fails_validate(kw):
    with pytest.raises(ValueError):
        sr830.validate(_form_cfg(**kw))


def test_reader_pairs_one_window_and_reports_lock_per_unit():
    a = _FakeSR830(x=[1.0] * 10, y=[0.0] * 10)
    b = _FakeSR830(x=[0.0] * 10, y=[2.0] * 10, lias={3: 1})          # B unlocked once
    reader = sr830.SR830Read([(a, sr830.LockinConfig(reference="internal")),
                              (b, sr830.LockinConfig(reference="external", filter_slope_dB=12))])
    d1, d2 = reader.read(10, threading.Event())
    assert d1["x_mean"] == pytest.approx(1.0) and d2["y_mean"] == pytest.approx(2.0)
    assert reader.locked() == [None, False]                           # internal unit: nothing to lock
    assert reader.locked() == [None, True]                            # latched bit was cleared
    assert reader.filter_meta(1) == (0.1, 2)


def test_reader_close_shuts_every_unit_even_if_one_fails(monkeypatch):
    closed = []

    def fake_shutdown(lk):
        closed.append(lk)
        if len(closed) == 1:
            raise RuntimeError("GPIB hiccup")

    monkeypatch.setattr(sr830, "shutdown", fake_shutdown)
    units = [(_FakeSR830(), sr830.LockinConfig()), (_FakeSR830(), sr830.LockinConfig())]
    with pytest.raises(RuntimeError):
        sr830.SR830Read(units).close()
    assert len(closed) == 2


# ── engines: lockin= replaces every MFLI call (daq=None proves it) ───────────

class _FakeLockin:
    """Duck-typed SR830Read: distinct 1f/2f values, lock flags, phases."""

    def __init__(self, n_units=2, locked=True, events=None):
        self.n, self._locked, self.events = n_units, locked, events
        self.reads = 0

    def read(self, n_averages, stop_event=None):
        self.reads += 1
        if self.events is not None:
            self.events.append("lockin.read")
        return [{"x_mean": 1e-3 * (i + 1), "y_mean": 0.0, "r_mean": 1e-3 * (i + 1),
                 "theta_mean": 0.0, "r_sem": 1e-6, "n_samples": n_averages,
                 "overload": False} for i in range(self.n)]

    def wait_locked(self, timeout_s, stop_event=None):
        return self._locked

    def locked(self):
        return [None, self._locked][: self.n]

    def frequency_Hz(self):
        return 977.5

    def phase_deg(self, i):
        return 10.0 * (i + 1)

    def filter_meta(self, i):
        return 0.3, 4


def test_harm_run_measurement_with_sr830_keeps_columns():
    acq = harm.AcquisitionConfig(settling_time_s=0.0, n_averages=5)
    d1 = harm.DemodConfig(device="x", demod_index=0, harmonic=1)
    d2 = harm.DemodConfig(device="y", demod_index=0, harmonic=2)
    df = harm.run_measurement(None, harm.OutputConfig(amplitude_V=0.1, series_R_ohm=1e4), d1, d2,
                              acq, [harm.MeasurementPoint()] * 2, write_csv=_NULL_WRITER,
                              lockin=_FakeLockin(locked=False))
    row = df.iloc[0]
    assert row["1f_R_V"] == pytest.approx(1e-3) and row["2f_R_V"] == pytest.approx(2e-3)
    assert row["mds_synced"] is None or math.isnan(row["mds_synced"])
    assert bool(row["follower_reference_locked"]) is False
    assert row["demod1_ref_phase_deg"] == 10.0 and row["demod2_ref_phase_deg"] == 20.0
    assert row["demod_output_convention"] == sr830.DEMOD_OUTPUT_CONVENTION
    assert row["excitation_current_A_peak"] == pytest.approx(1e-5)   # peak / R, as for the MFLI


def test_harm6_run_measurement_with_sr830_reads_frequency_from_the_unit():
    acq = harm.AcquisitionConfig(settling_time_s=0.0, n_averages=5)
    d1 = harm.DemodConfig(device="x", demod_index=0, harmonic=1)
    d2 = harm.DemodConfig(device="y", demod_index=0, harmonic=2)
    ac = harm6.ACSourceConfig(amplitude_A=1e-4, frequency_Hz=977.0, compliance_V=2.0)
    df = harm6.run_measurement(None, ac, None, None, d1, d2, acq, [harm.MeasurementPoint()],
                               write_csv=_NULL_WRITER, lockin=_FakeLockin())
    row = df.iloc[0]
    assert row["excitation_frequency_Hz"] == 977.5
    assert bool(row["follower_reference_locked"]) is True
    assert row["2f_R_V"] == pytest.approx(2e-3)


def test_sot2h_with_sr830_keeps_pulse_ordering_and_columns():
    dev = _FakeKXCI(events=(events := []))
    read = sot2h.ReadConfig(sense_current_A=1e-4, n_averages=2, settle_after_enable_s=0.0,
                            lock_timeout_s=1.0, delay_after_pulse_s=0.0)
    pmu = PMUPulseConfig(library="lib", module="m", return_names=())
    # same demod index for 1f, 2f and the PLL: an MFLI conflict, irrelevant for SR830s
    d = sot2h.DemodConfig(device="x", demod_index=0, harmonic=1)
    df = sot2h.run_measurement(dev, pmu, _Fake6221AC(events), None, d, d,
                               sot2h.ExtRefConfig(device="x", pll_demod_index=0), read,
                               [sot2h.AmplitudePoint(amplitude_V=0.5)], write_csv=_NULL_WRITER,
                               lockin=_FakeLockin(events=events))
    i_abort, i_ex = events.index("6221.abort"), events.index("EX")
    assert i_abort < i_ex < events.index("6221.start") < events.index("lockin.read")
    row = df.iloc[0]
    assert row["1f_R_V"] == pytest.approx(1e-3) and row["2f_R_V"] == pytest.approx(2e-3)
    assert bool(row["reference_locked"]) is True and row["excitation_frequency_Hz"] == 977.5
    assert row["demod_output_convention"] == sr830.DEMOD_OUTPUT_CONVENTION


def test_sot1i_with_sr830_tags_an_unlocked_read():
    import test_sot_pulsed_switching_6221_run_loops as rl
    pulse_cfg, read_cfg, demod_cfg, extref_cfg = rl._cfgs()
    df = sot1i.run_measurement(rl._Fake6221AC(), None, demod_cfg, extref_cfg, pulse_cfg, read_cfg,
                               [sot1i.PulsePoint(pulse_current_A=1e-3)], write_csv=_NULL_WRITER,
                               lockin=_FakeLockin(n_units=1, locked=False))
    row = df.iloc[0]
    assert bool(row["reference_locked"]) is False
    assert row["demod_R_V"] == pytest.approx(1e-3) and row["excitation_frequency_Hz"] == 977.5


# ── forms: plan + summary in SR830 mode ──────────────────────────────────────

def _defaults(mod, **overrides) -> dict:
    """A parsed state straight from a form module's DEFAULTS."""
    state = {k: (mod.NUMERIC_FIELDS[k](v) if k in mod.NUMERIC_FIELDS else v)
             for k, v in mod.DEFAULTS.items()}
    for k in mod.OPTIONAL_NUMERIC_FIELDS:
        state[k] = float(state[k]) if state.get(k) not in ("", None) else None
    state.update(sample="A", device="HB3", cooldown="", enable_temperature=False)
    state.update(overrides)
    return mod.resolve_state(state)


def _harm_state(**kw) -> dict:
    return _defaults(harm_tui, **{**dict(leader_harmonic=1, follower_harmonic=2, order_1f=4,
                                         order_2f=4, leader_automode=4, follower_automode=4,
                                         enable_sweep=False), **kw})


def test_harm_mfli_mode_plan_is_untouched(tmp_path):
    ensure_sample(tmp_path, "A", create=True)
    plan = harm_tui.build_plan(_harm_state(), tmp_path)
    assert plan.sr830_cfgs is None and "lockin" not in plan.header_extra
    assert plan.demod1_cfg.sample_rate_Hz == 857.0


def test_harm_sr830_plan_lockin_source_drives_from_sine_out(tmp_path):
    ensure_sample(tmp_path, "A", create=True)
    plan = harm_tui.build_plan(_harm_state(lockin="sr830", amplitude_V=1.0), tmp_path)
    leader, follower = plan.sr830_cfgs
    assert leader.reference == "internal" and follower.reference == "external"
    assert leader.sine_amplitude_V == pytest.approx(1.0 / math.sqrt(2))   # peak form -> rms unit
    assert (leader.harmonic, follower.harmonic) == (1, 2)
    assert plan.header_extra["lockin"] == "SR830"
    assert plan.demod1_cfg.sample_rate_Hz == 512.0          # estimate + plan see the SR830 cap


def test_harm6_sr830_plan_has_both_units_on_the_marker(tmp_path):
    ensure_sample(tmp_path, "A", create=True)
    plan = harm_tui.build_plan(_harm_state(lockin="sr830", ac_source="6221"), tmp_path)
    assert [c.reference for c in plan.sr830_cfgs] == ["external", "external"]
    assert plan.header_extra["lockin"] == "SR830"


def test_harm_sr830_summary_flags_what_an_sr830_cannot_do(tmp_path):
    _, warnings, errors = harm_tui.build_summary(_harm_state(
        lockin="sr830", order_2f=6, amplitude_V=10.0, data_dir=str(tmp_path)))
    assert any("Follower SR830" in e and "filter_slope_dB" in e for e in errors)
    assert any("Leader SR830" in e and "SINE OUT" in e for e in errors)
    assert any("can't be switched off" in w for w in warnings)


def test_harm_sr830_mode_ignores_hidden_mfli_fields():
    errs = ["'leader_device' is not a valid number.", "'sensitivity_1f_V' is not a valid number."]
    assert harm_tui.mode_errors({"lockin": "sr830"}, errs) == errs[1:]
    assert harm_tui.mode_errors({"lockin": "mfli"}, errs) == errs[:1]


@pytest.mark.parametrize("pulse,harmonics", [("pmu", (1, 2)), ("6221", (2,))])
def test_sot_sr830_plan_one_unit_per_harmonic(tmp_path, pulse, harmonics):
    ensure_sample(tmp_path, "A", create=True)
    state = _defaults(sot_tui, pulse_source=pulse, read_mode="harmonic", lockin="sr830",
                      automode=4, data_dir=str(tmp_path))
    plan = sot_tui.build_plan(state, tmp_path)
    assert tuple(c.harmonic for c in plan.sr830_cfgs) == harmonics
    assert all(c.reference == "external" for c in plan.sr830_cfgs)
    assert plan.header_extra["lockin"] == "SR830"


def test_sr830_follower_anchor_measures_at_1f_then_restores(monkeypatch):
    lk = _FakeSR830(x=[1.0] * 10, y=[0.0] * 10)
    object.__setattr__(lk, "phase", 5.0)

    def fake_auto_phase(lockin, cfg):
        assert cfg.harmonic == 1                 # nulled at f, not at 2f
        lockin.phase = 33.0
        return 33.0

    monkeypatch.setattr(sr830, "auto_phase", fake_auto_phase)
    cfg = sr830.LockinConfig(harmonic=2, reference="external")
    assert harm.null_follower_reference_via_1f_sr830(lk, cfg, n_averages=10) == 33.0
    assert lk.phase == 5.0                       # measured only — data frame untouched
    assert [c for c in lk.log if c.startswith("HARM")] == ["HARM 1", "HARM 2"]


def _mount_and_toggle(monkeypatch, tmp_path, mod, app_cls, selects: dict, ids):
    import asyncio
    from textual.widgets import Select
    monkeypatch.setattr(mod, "_DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app_cls, "data_root", tmp_path)

    async def go():
        app = app_cls()
        async with app.run_test(size=(220, 70)) as pilot:
            for sel_id, value in selects.items():
                app.query_one(f"#{sel_id}", Select).value = value
                await pilot.pause()
            state, parse_errors = app.parse_state()
            return {i: app.query_one(f"#{i}").display for i in ids}, state, parse_errors
    return asyncio.run(go())


def test_harm_form_toggle_shows_the_sr830_cards(monkeypatch, tmp_path):
    shown, state, parse_errors = _mount_and_toggle(
        monkeypatch, tmp_path, harm_tui, harm_tui.MFLIDualHarmonicApp,
        {"ac_source": "6221", "lockin": "sr830"},
        ["lockin_sr830_addr", "lockin_sr830_sens", "lockin_mfli_devices", "mode_6221_extref",
         "mode_6221_source"])
    assert shown == {"lockin_sr830_addr": True, "lockin_sr830_sens": True,
                     "lockin_mfli_devices": False, "mode_6221_extref": False,
                     "mode_6221_source": True}
    assert state["lockin"] == "sr830" and not parse_errors


def test_sot_form_toggle_shows_the_sr830_card(monkeypatch, tmp_path):
    shown, state, _ = _mount_and_toggle(
        monkeypatch, tmp_path, sot_tui, sot_tui.SOTPulsedSwitchingApp,
        {"read_mode": "harmonic", "lockin": "sr830"},
        ["mode_sr830", "mode_mfli", "mode_lockin_filter", "mode_sot2h_demods"])
    assert shown == {"mode_sr830": True, "mode_mfli": False, "mode_lockin_filter": True,
                     "mode_sot2h_demods": False}
    assert state["lockin"] == "sr830"
