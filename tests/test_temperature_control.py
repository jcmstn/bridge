"""MercuryiTC setpoint control: the settle wait, the software max-T cutoff,
the form/summary helpers, and the step runner (outer loop over a whole
program run). No hardware: a fake iTC and a fake clock."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

import instruments.mercury_itc as itc
import instruments.temperature_control as tc
from instruments.data_naming import allocate_run, ensure_sample, record_run


class FakeITC:
    def __init__(self, readings):
        self.readings = list(readings)
        self.calls: list = []

    def temperature(self, uid):
        self.calls.append(("read", uid))
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]

    def heater_off(self, uid):
        self.calls.append(("heater_off", uid))

    def set_temperature_setpoint(self, uid, T):
        self.calls.append(("tset", uid, T))

    def heater_auto(self, uid, on):
        self.calls.append(("auto", uid, on))


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 0.0}
    fake = SimpleNamespace(monotonic=lambda: now["t"],
                           sleep=lambda dt: now.__setitem__("t", now["t"] + dt))
    monkeypatch.setattr(itc, "time", fake)
    return now


CFG = itc.TemperatureControllerConfig(sensor_uids=("MB1.T1",), max_temperature_K=320.0)


def _wait(mitc, **kw):
    return itc.wait_for_temperature_stable(mitc, CFG, 10.0, tolerance_K=0.005, hold_time_s=3,
                                           timeout_s=None, poll_interval_s=1.0, **kw)


def test_hold_clock_restarts_on_any_excursion(clock):
    # in, in, OUT, in, in, in, in -> settles only 3 s after the excursion
    mitc = FakeITC([10.000, 10.001, 10.020, 10.000, 10.001, 10.002, 10.003, 10.0])
    assert _wait(mitc) == "settled"
    assert sum(c[0] == "read" for c in mitc.calls) == 7
    assert ("heater_off", "MB1.T1") not in mitc.calls


def test_skip_and_stop_end_the_wait(clock):
    skip, stop = threading.Event(), threading.Event()
    mitc = FakeITC([11.0])           # never in margin
    assert _wait(mitc, skip_event=skip,
                 on_progress=lambda T, s: skip.set() if clock["t"] >= 2 else None) == "skipped"
    assert _wait(FakeITC([11.0]), stop_event=stop,
                 on_progress=lambda T, s: stop.set()) == "stopped"


def test_reading_at_max_turns_heater_off_and_raises(clock):
    mitc = FakeITC([300.0, 320.0])
    with pytest.raises(itc.TemperatureLimitError):
        _wait(mitc)
    assert ("heater_off", "MB1.T1") in mitc.calls


def test_read_temperature_enforces_the_max():
    assert itc.read_temperature(FakeITC([300.0]), CFG) == (300.0, None)
    mitc = FakeITC([325.0])
    with pytest.raises(itc.TemperatureLimitError):
        itc.read_temperature(mitc, CFG)
    assert ("heater_off", "MB1.T1") in mitc.calls
    assert itc.read_temperature(None, CFG) == (None, None)


def _state(**kw):
    s = {**tc.TEMPERATURE_DEFAULTS, "temperature_max_K": 320.0, "temperature_margin_mK": 5.0,
         "temperature_hold_s": 120.0, "temperature_setpoint_K": 300.0}
    s.update(kw)
    return s


def _summary(state, **kw):
    info, warnings, errors = [], [], []
    tc.temperature_summary(state, info, warnings, errors, **kw)
    return info, warnings, errors


def test_summary_and_steps_from_state():
    assert tc.steps_from_state(_state()) is None                    # control off: today's behaviour
    assert not _summary(_state())[2]
    on = _state(temperature_control=True, temperature_setpoints_K="10, 20 50")
    assert not _summary(on)[2]
    steps = tc.steps_from_state(on)
    assert steps.setpoints_K == [10.0, 20.0, 50.0] and steps.margin_K == pytest.approx(0.005)
    state, steps = tc.with_first_setpoint(on)
    assert state["temperature_setpoint_K"] == 10.0 and steps is not None
    for bad in ("", "10, x", "10, 330", "-4"):
        assert _summary(_state(temperature_control=True, temperature_setpoints_K=bad))[2], bad
    assert _summary(_state(temperature_control=True, temperature_setpoints_K="10",
                           enable_temperature=False))[2]
    assert _summary(_state(temperature_control=True, temperature_setpoints_K="10, 20"),
                    single_target=True)[2]


def test_repeat_run_cost_charges_each_step():
    rc = tc.RunCost(3)
    rc.each("x", 1.0)
    steps = tc.TemperatureSteps([10.0, 20.0], 0.005, 60.0, CFG)
    out = tc.repeat_run_cost(rc, steps)
    assert out.points == [61.0, 1.0, 1.0, 61.0, 1.0, 1.0]
    assert tc.repeat_run_cost(rc, None) is rc


def test_go_to_temperature_sends_setpoint_waits_and_leaves_it_on(monkeypatch, clock):
    mitc = FakeITC([20.0])
    closed = []
    monkeypatch.setattr(tc, "connect_temperature_controller", lambda cfg: mitc)
    monkeypatch.setattr(tc, "shutdown_temperature_controller", lambda m: closed.append(m))
    steps = tc.TemperatureSteps([20.0], 0.005, 2.0, CFG)
    status = []
    assert tc.go_to_temperature(steps, 0, threading.Event(), status.append) == "settled"
    assert ("tset", "MB1.T1", 20.0) in mitc.calls and ("auto", "MB1.T1", True) in mitc.calls
    assert not any(c[0] == "heater_off" for c in mitc.calls)       # setpoint stays active
    assert closed == [mitc] and any("→ 20 K" in s for s in status)


def test_go_to_temperature_refuses_when_the_loop_sensor_is_dropped(monkeypatch):
    def connect(cfg):
        cfg.sensor_uids = ("DB5.T1",)          # the probe dropped MB1.T1
        return FakeITC([20.0])
    monkeypatch.setattr(tc, "connect_temperature_controller", connect)
    monkeypatch.setattr(tc, "shutdown_temperature_controller", lambda m: None)
    cfg = itc.TemperatureControllerConfig(sensor_uids=("MB1.T1", "DB5.T1"))
    with pytest.raises(RuntimeError, match="MB1.T1"):
        tc.go_to_temperature(tc.TemperatureSteps([20.0], 0.005, 2.0, cfg), 0, threading.Event(),
                             lambda s: None)


def test_run_temperature_steps_runs_the_program_once_per_setpoint(monkeypatch, tmp_path):
    ensure_sample(tmp_path, "_test", create=True)
    went = []
    monkeypatch.setattr(tc, "go_to_temperature",
                        lambda steps, k, stop, on_status: went.append(steps.setpoints_K[k]) or "settled")

    def run_plan(plan, stop_event, *, on_point, run_contexts, run_extras, **_kw):
        for i in range(2):                      # a two-run series, like two sense currents
            ctx = allocate_run(tmp_path, "_test", "HB3", "HALL",
                               temperature_setpoint_K=plan.temperature_setpoint_K)
            run_contexts.append(ctx)
            run_extras.append({})
            record_run(tmp_path, ctx, lambda recs, status: {"status": status},
                       lambda cb, write: cb({"point_index": 0}), stop_event, on_point=on_point,
                       tags={"series_index": i, "series_label": f"I={i}A"})

    program = SimpleNamespace(run_plan=run_plan)
    build = lambda state, root: SimpleNamespace(temperature_setpoint_K=state["temperature_setpoint_K"])
    steps = tc.TemperatureSteps([10.0, 20.0], 0.005, 0.0, CFG)
    state = {"temperature_setpoint_K": 10.0}
    points, plans, contexts = [], [], []
    tc.run_temperature_steps(steps, state, tmp_path, build(state, tmp_path), build,
                             lambda _p: program, threading.Event(), on_point=points.append,
                             run_contexts=contexts, run_extras=[], on_plan=plans.append)

    assert went == [10.0, 20.0]
    assert [p.temperature_setpoint_K for p in plans] == [10.0, 20.0]
    assert [r["series_index"] for r in points] == [0, 1, 2, 3]
    assert [r["series_label"] for r in points] == ["T=10K I=0A", "T=10K I=1A", "T=20K I=0A", "T=20K I=1A"]
    index = pd.read_csv(tmp_path / "_test" / "index.csv")
    assert list(index["T_setpoint_K"]) == [10.0, 10.0, 20.0, 20.0]
    assert ["T010K" in c.raw_path.name for c in contexts] == [True, True, False, False]


def test_stop_during_a_wait_skips_the_program(monkeypatch, tmp_path):
    monkeypatch.setattr(tc, "go_to_temperature", lambda *a: "stopped")
    program = SimpleNamespace(run_plan=lambda *a, **k: pytest.fail("ran after a stop"))
    tc.run_temperature_steps(tc.TemperatureSteps([10.0], 0.005, 0.0, CFG), {}, tmp_path, None,
                             None, lambda _p: program, threading.Event())


def test_hall_form_uses_the_first_setpoint_and_greys_the_manual_one(tmp_path, monkeypatch):
    import dc.dc_hall_measurement_tui as hall
    monkeypatch.setattr(hall, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(hall.DCHallMeasurementApp, "data_root", tmp_path)

    async def go():
        app = hall.DCHallMeasurementApp()
        async with app.run_test(size=(220, 70)) as pilot:
            await pilot.pause()
            app.query_one("#device").value = "HB3"
            app.query_one("#temperature_control").value = True
            app.query_one("#temperature_setpoints_K").value = "50, 100"
            await pilot.pause()
            state, errors = app.parse_state()
            assert not errors and not hall.build_summary(state)[2]
            assert state["temperature_setpoint_K"] == 50.0
            assert app.query_one("#temperature_setpoint_K").disabled
            assert not app.query_one("#temperature_hold_s").disabled
            assert any("Temperature steps: 2" in line for line in hall.build_summary(state)[0])

    asyncio.run(go())
