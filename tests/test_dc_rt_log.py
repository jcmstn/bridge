"""
dc/dc_rt_log.py + dc_rt_log_tui.py — the R-vs-T log loop with fake
instruments (R = V_odd / I, T = mean of before/after, drift recorded, the
three stop conditions, the save throttle, no iTC), and the MercuryiTC sensor
handling it relies on (UID normalisation, separators, the verbose 2 → 1 probe).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import dc.dc_rt_log as rt
import dc.dc_rt_log_tui as tui
from instruments import mercury_itc
from instruments.mercury_itc import (
    TemperatureControllerConfig, connect_temperature_controller, normalize_uid,
    parse_temperature_reply, probe_temperature_sensors,
)
from instruments.tui_common import parse_sensor_uids

R_TRUE, V_OFFSET = 50.0, 3e-6     # Ω, thermal-EMF offset the ±I reversal must cancel


class FakeSource:
    def __init__(self):
        self.source_current = 0.0


class FakeVoltmeter:
    def __init__(self, source):
        self.source = source

    @property
    def voltage(self):
        return R_TRUE * self.source.source_current + V_OFFSET


class FakeITC:
    """Temperature falls 0.1 K per read; answers only the UIDs in `uids`."""
    def __init__(self, uids=("MB1.T1",), T0=300.0):
        self.uids, self.T = set(uids), T0

    def ask(self, command: str) -> str:
        if command == "*IDN?":
            return "IDN:OXFORD INSTRUMENTS:MERCURY ITC:fake:1.0"
        if command == "READ:SYS:CAT":
            return "STAT:SYS:CAT:DEV:MB1.T1:TEMP:DEV:DB5.T1:TEMP"
        uid = command.split(":")[2]
        if uid not in self.uids:
            return f"STAT:DEV:{uid}:TEMP:SIG:TEMP:INVALID"
        self.T -= 0.1
        return f"STAT:DEV:{uid}:TEMP:SIG:TEMP:{self.T:.4f}K"

    def temperature(self, uid):
        return parse_temperature_reply(normalize_uid(uid), self.ask(mercury_itc.temperature_command(uid)))

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(rt.time, "sleep", lambda s: None)


def _run(acq_kw=None, itc=None, uids=("MB1.T1",), stop_after=None, writes=None, clock=None):
    source = FakeSource()
    src_cfg = rt.SourceConfig(sense_current_A=1e-4, source_delay_s=0.0)
    acq = rt.AcquisitionConfig(**{"n_reversals": 3, "max_duration_s": 1e9, **(acq_kw or {})})
    stop = threading.Event()
    got = []

    def on_point(rec):
        got.append(rec)
        if stop_after is not None and len(got) >= stop_after:
            stop.set()

    cfg = TemperatureControllerConfig(sensor_uids=uids) if itc is not None else None
    df = rt.run_measurement(source, FakeVoltmeter(source), src_cfg, acq, stop, on_point,
                            temp_ctrl=itc, temp_cfg=cfg,
                            write_csv=(lambda recs: writes.append(len(recs))) if writes is not None
                            else (lambda recs: None))
    return df, got


def test_resistance_is_v_odd_over_i_and_offset_goes_to_v_even():
    df, _ = _run(itc=FakeITC(), stop_after=3)
    assert len(df) == 3
    assert df["resistance_ohm"].tolist() == pytest.approx([R_TRUE] * 3)
    assert df["voltage_even_V"].tolist() == pytest.approx([V_OFFSET] * 3)


def test_temperature_is_before_after_mean_and_drift_is_recorded():
    df, _ = _run(itc=FakeITC(T0=300.0), stop_after=2)
    # reads: 299.9 (before), 299.8 (after) for sample 1
    assert df["temperature_1_K"][0] == pytest.approx(299.85)
    assert df["temperature_1_drift_K"][0] == pytest.approx(-0.1)
    assert (df["sample_duration_s"] >= 0).all()
    assert df["temperature_2_K"].isna().all()


def test_stops_when_sensor_1_crosses_T_stop():
    df, _ = _run({"T_stop_K": 299.0}, itc=FakeITC(T0=300.0))
    assert df["temperature_1_K"].iloc[-1] < 299.0 < df["temperature_1_K"].iloc[0]
    assert len(df) == 6       # 0.2 K per sample: 299.85 … 298.85


def test_stops_at_max_duration(monkeypatch):
    t = iter(range(0, 10_000, 10))     # every monotonic() call advances 10 s
    monkeypatch.setattr(rt.time, "monotonic", lambda: next(t))
    df, _ = _run({"max_duration_s": 100.0})
    assert 0 < len(df) < 10


def test_save_throttle_writes_at_most_every_save_every_s():
    writes: list = []
    _run({"save_every_s": 1e9}, stop_after=5, writes=writes)
    assert writes == [1]       # first sample only; record_run writes the final file


def test_no_itc_logs_against_time_only():
    df, _ = _run(stop_after=2)
    assert df["temperature_1_K"].isna().all() and df["temperature_1_drift_K"].isna().all()


# ── MercuryiTC sensor handling ───────────────────────────────────────────────

@pytest.mark.parametrize("raw,uid", [("MB1.T1", "MB1.T1"), (" DEV:MB1.T1:TEMP ", "MB1.T1"),
                                     ("DB5.T1:TEMP", "DB5.T1"), ("dev:MB1.T1", "MB1.T1")])
def test_normalize_uid(raw, uid):
    assert normalize_uid(raw) == uid
    assert mercury_itc.temperature_command(raw) == f"READ:DEV:{uid}:TEMP:SIG:TEMP"


@pytest.mark.parametrize("raw", ["MB1.T1, DB5.T1", "MB1.T1 DB5.T1", "MB1.T1;DB5.T1", " MB1.T1 ,  DB5.T1 "])
def test_parse_sensor_uids_accepts_any_separator(raw):
    assert parse_sensor_uids(raw) == ("MB1.T1", "DB5.T1")


def test_probe_keeps_both_sensors_when_both_answer():
    cfg = probe_temperature_sensors(FakeITC(uids=("MB1.T1", "DB5.T1")),
                                    TemperatureControllerConfig(sensor_uids=("MB1.T1", "DEV:DB5.T1:TEMP")))
    assert cfg.sensor_uids == ("MB1.T1", "DB5.T1")


def test_probe_falls_back_to_the_one_sensor_that_answers(caplog):
    caplog.set_level("INFO")
    cfg = probe_temperature_sensors(FakeITC(uids=("DB5.T1",)),
                                    TemperatureControllerConfig(sensor_uids=("MB1.T1", "DB5.T1")))
    assert cfg.sensor_uids == ("DB5.T1",)
    assert "INVALID" in caplog.text and "board catalog" in caplog.text and "using ['DB5.T1'] only" in caplog.text


def test_probe_returns_none_when_no_sensor_answers():
    assert probe_temperature_sensors(FakeITC(uids=()), TemperatureControllerConfig()) is None
    assert probe_temperature_sensors(None, TemperatureControllerConfig()) is None


def test_connect_probes_and_narrows_the_callers_cfg_in_place(monkeypatch):
    """Every program routes through connect_temperature_controller(), so the
    2 → 1 fallback reaches all of them via the cfg they already hold."""
    itc = FakeITC(uids=("DB5.T1",))
    monkeypatch.setattr(mercury_itc, "MercuryITC", lambda *a, **k: itc)
    cfg = TemperatureControllerConfig(sensor_uids=("MB1.T1", "DB5.T1"))
    assert connect_temperature_controller(cfg) is itc
    assert cfg.sensor_uids == ("DB5.T1",)


def test_connect_returns_none_and_closes_when_no_sensor_answers(monkeypatch):
    itc = FakeITC(uids=())
    monkeypatch.setattr(mercury_itc, "MercuryITC", lambda *a, **k: itc)
    assert connect_temperature_controller(TemperatureControllerConfig()) is None
    assert getattr(itc, "closed", False)


def test_probe_survives_a_query_that_raises():
    itc = SimpleNamespace(ask=lambda cmd: (_ for _ in ()).throw(TimeoutError("VI_ERROR_TMO")))
    assert probe_temperature_sensors(itc, TemperatureControllerConfig()) is None


# ── TUI pure helpers ─────────────────────────────────────────────────────────

def _state(**kw) -> dict:
    s = {k: (tui.NUMERIC_FIELDS[k](v) if k in tui.NUMERIC_FIELDS
             else (float(v) if v else None) if k in tui.OPTIONAL_NUMERIC_FIELDS else v)
         for k, v in tui.DEFAULTS.items()}
    s.update(sample="A", device="HB3", data_dir="", **kw)
    return s


def test_run_cost_is_the_max_duration_upper_bound():
    s = _state(max_duration_min=60.0, interval_s=10.0)
    assert tui.max_samples(s) == 360
    assert tui.run_costs(s).total_s == pytest.approx(360 * 10.0 + tui.PER_RUN_S + 0.02)


def test_summary_flags_zero_current_and_huge_duration():
    assert any("non-zero" in e for e in tui.build_summary(_state(sense_current_A=0.0))[2])
    assert any("limit" in e for e in tui.build_summary(_state(max_duration_min=1e9))[2])


def test_build_plan_maps_the_form(tmp_path):
    plan = tui.build_plan(_state(T_stop_K=80.0, temperature_sensor_uids="MB1.T1 DB5.T1"), tmp_path)
    assert plan.acq_cfg.T_stop_K == 80.0 and plan.acq_cfg.max_duration_s == 240 * 60
    assert plan.temp_cfg.sensor_uids == ("MB1.T1", "DB5.T1")
    assert plan.total_points == len(plan.run_cost.points)


def test_one_r_vs_t_panel_per_sensor_that_reads():
    import pandas as pd
    import web.dc.rt_log as page
    two = [{"temperature_1_K": 300.0, "temperature_2_K": 299.0}]
    one = [{"temperature_1_K": 300.0, "temperature_2_K": None}]
    assert rt.sensors_in(two) == [1, 2] and rt.sensors_in(one) == [1] and rt.sensors_in([{}]) == []
    assert rt.sensors_in(pd.DataFrame(one)) == [1]
    assert len(page.make_figure([1, 2]).data) == 3 and len(page.make_figure([]).data) == 1


@pytest.mark.parametrize("stop_is_normal_end,expected", [(False, "aborted"), (True, "completed")])
def test_web_run_history_records_a_stopped_log_as_completed(stop_is_normal_end, expected):
    """Stop is how a log ends: with stop_is_normal_end the runs.db row says
    "completed" (other programs keep "aborted")."""
    from instruments import run_index
    from web import run_manager
    from web.run_controller import RunCallbacks, RunController

    def run_fn(stop_event, cb):
        stop_event.set()          # the operator pressed Stop

    rc = RunController(suite="DC", measurement="RT test", run_fn=run_fn,
                       save_artifacts=lambda *a: [], parameters={}, data_dir="", planned_output_paths=[],
                       on_record=lambda r: None, on_status=lambda t: None, on_log=lambda t, l: None,
                       on_finished=lambda f, r: None, stop_is_normal_end=stop_is_normal_end)
    rc.handle = run_manager.try_acquire("DC", "RT test")
    rc.handle.run_id = run_index.start_run("DC", "RT test", {}, "", [])
    rc._start_time = 0.0
    rc._worker(RunCallbacks(on_point=lambda r: None, on_status=lambda t: None,
                            on_run_label=lambda t: None))
    assert run_index.recent_runs(1)[0]["status"] == expected
    assert rc._queue.get_nowait()["status"] == expected


# ── Fixed field ──────────────────────────────────────────────────────────────

class FakeGaussmeter:
    def __init__(self, B_T=0.1):
        self.B_T, self.reads = B_T, 0

    def measure(self, n, delay=0.0):
        self.reads += 1
        return self.B_T, 0.0


def test_fixed_field_is_read_after_the_timed_window_and_recorded(monkeypatch):
    """The B read happens after t_end, so sample_duration_s (the R/T window)
    does not include it; the hold current and θ/φ go into every row."""
    gm = FakeGaussmeter()
    clock = [0.0]

    def slow_measure(n, delay=0.0):
        clock[0] += 100.0              # a very slow field read
        return 0.1, 0.0

    gm.measure = slow_measure
    monkeypatch.setattr(rt.time, "monotonic", lambda: clock[0])
    source = FakeSource()
    stop = threading.Event()
    df = rt.run_measurement(
        source, FakeVoltmeter(source), rt.SourceConfig(sense_current_A=1e-4, source_delay_s=0.0),
        rt.AcquisitionConfig(n_reversals=2, max_duration_s=1e9), stop,
        lambda rec: stop.set(), write_csv=lambda recs: None,
        gaussmeter=gm, gauss_cfg=rt.GaussmeterConfig(unit="T"),
        magnet_current_A=2.0, field_theta_deg=90.0, field_phi_deg=0.0)
    row = df.iloc[0]
    assert row["magnet_field_mT"] == pytest.approx(100.0)
    assert row["sample_duration_s"] < 100.0
    assert (row["magnet_current_A"], row["field_theta_deg"], row["field_phi_deg"], row["field_frame"]) \
        == (2.0, 90.0, 0.0, "RH")


def test_no_field_leaves_field_columns_blank():
    df, _ = _run(stop_after=1)
    assert df["magnet_field_mT"].isna().all() and df["magnet_current_A"].isna().all()


def test_summary_and_cost_with_fixed_field():
    off = _state(max_duration_min=60.0, interval_s=0.0)
    on = _state(max_duration_min=60.0, interval_s=0.0, enable_field=True, magnet_current_A=5.0)
    assert tui.sample_time_s(on) == tui.sample_time_s(off)          # R/T window unchanged
    assert tui.sample_period_s(on) > tui.sample_period_s(off)       # only the period grows
    assert tui.build_summary(on)[2] == tui.build_summary(off)[2]       # no field-specific error
    assert any("exceeds" in e for e in tui.build_summary({**on, "magnet_current_A": 99.0})[2])
    plan = tui.build_plan(on, "/tmp")
    assert plan.magnet_current_A == 5.0 and plan.magnet_cfg is not None
    assert tui.build_plan(off, "/tmp").magnet_cfg is None


def test_run_plan_parks_the_magnet_and_always_ramps_it_down(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from instruments.data_naming import ensure_sample
    ensure_sample(tmp_path, "A", create=True)
    plan = tui.build_plan(_state(enable_field=True, magnet_current_A=3.0, enable_temperature=False),
                          tmp_path)
    calls: list = []
    for name in ("connect_source", "connect_voltmeter", "connect_magnet", "connect_gaussmeter",
                 "ramp_current_to_zero", "shutdown_source", "shutdown_gaussmeter"):
        monkeypatch.setattr(tui, name, lambda *a, **k: MagicMock())
    monkeypatch.setattr(tui, "set_magnet_current", lambda m, c, I, *a: calls.append(("set", I)))
    monkeypatch.setattr(tui, "shutdown_magnet", lambda *a: calls.append(("shutdown",)))

    def boom(*a, **k):
        assert k["magnet_current_A"] == 3.0 and k["gaussmeter"] is not None
        raise RuntimeError("2182 timeout")

    monkeypatch.setattr(tui, "run_measurement", boom)
    with pytest.raises(RuntimeError):
        tui.run_plan(plan, threading.Event())
    assert calls == [("set", 3.0), ("shutdown",)]
