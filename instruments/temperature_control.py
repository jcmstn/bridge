"""
MercuryiTC temperature control — setpoints as the outer loop of a program
==========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-10-13

Pure (no Textual): the form fields, summary lines, config and the step
runner every series program shares, driven by the TUI (tui_common).

What a step is: the iTC is told to go to T (its own PID does the rest —
`set_temperature`: TSET + heater auto; PID values stay on the touch
screen), the reading must stay within ±margin for `hold_s` without a
break, then the WHOLE program runs once at T — `build_plan` with
`temperature_setpoint_K = T` (so the filename T###K token, header and index
carry the commanded setpoint) and the program's own `run_plan`, exactly as
if Start had been pressed at that temperature. Between steps the program's
own teardown runs (sources off, magnet to 0). The operator can skip a wait
(`TemperatureSteps.skip_event`) when the margin turns out unreachable.

After the last step — or an abort — the last setpoint stays active on the
iTC: nothing here ever turns the heater off, except the software maximum
(`mercury_itc.enforce_max_temperature`, checked on every reading).

    state -> steps_from_state(state) -> TemperatureSteps | None
    run_temperature_steps(steps, state, data_root, first_plan, build_plan,
                          engine, stop_event, on_status=..., ...)

The form fields (`TEMPERATURE_DEFAULTS` / `*_FIELDS`) are merged into each
program module's own groups; `temperature_summary()` is called from its
build_summary(); `temperature_cfg_from_state()` from its build_plan().
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

from instruments.mercury_itc import (
    TemperatureControllerConfig, connect_temperature_controller, normalize_uid,
    set_temperature, shutdown_temperature_controller, wait_for_temperature_stable,
)
from instruments.run_time import RunCost

log = logging.getLogger(__name__)

# ── Form fields (merged into every series program's groups) ─────────────────
TEMPERATURE_DEFAULTS = {
    "enable_temperature": True,
    "temperature_visa_resource": "TCPIP0::192.168.1.5::7020::SOCKET",
    "temperature_sensor_uids": "MB1.T1",
    "temperature_max_K": "320",
    "temperature_control": False,
    "temperature_setpoints_K": "",
    "temperature_margin_mK": "5",
    "temperature_hold_s": "120",
}
TEMPERATURE_NUMERIC_FIELDS = {
    "temperature_max_K": float,
    "temperature_margin_mK": float,
    "temperature_hold_s": float,
}
TEMPERATURE_TEXT_FIELDS = ["temperature_visa_resource", "temperature_sensor_uids",
                           "temperature_setpoints_K"]
# greyed out while "Log temperature" (enable_temperature) is off
TEMPERATURE_FIELD_IDS = ["temperature_visa_resource", "temperature_sensor_uids", "temperature_max_K"]
# greyed out while "Drive the iTC" (temperature_control) is off
TEMPERATURE_CONTROL_FIELD_IDS = ["temperature_setpoints_K", "temperature_margin_mK",
                                 "temperature_hold_s"]

POLL_INTERVAL_S = 1.0


def parse_sensor_uids(raw: str) -> tuple:
    """Parse a "MB1.T1, DB5.T1" field into a 1- or 2-tuple of UIDs. Commas,
    semicolons and whitespace all separate — "MB1.T1 DB5.T1" is two UIDs,
    not one bogus one the iTC would reject on every read."""
    uids = [u for u in re.split(r"[,;\s]+", raw) if u]
    return tuple(uids[:2])


def parse_setpoints(raw: str) -> tuple[list[float], Optional[str]]:
    """'10, 20 50' -> ([10.0, 20.0, 50.0], None); an error string instead
    of raising for anything that is not a finite positive number."""
    values: list[float] = []
    for part in re.split(r"[,;\s]+", raw or ""):
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            return values, f"Temperature setpoint {part!r} is not a number."
        if not (0 < v < float("inf")):
            return values, f"Temperature setpoint {part!r} must be a positive temperature (K)."
        values.append(v)
    return values, None


def temperature_cfg_from_state(state: dict) -> Optional[TemperatureControllerConfig]:
    """The iTC config for logging (None = logging off / no sensor UID)."""
    if not state.get("enable_temperature"):
        return None
    uids = parse_sensor_uids(state.get("temperature_sensor_uids", ""))
    if not uids:
        return None
    return TemperatureControllerConfig(
        visa_resource=state["temperature_visa_resource"], sensor_uids=uids,
        max_temperature_K=state.get("temperature_max_K", 320.0))


# ── Steps ───────────────────────────────────────────────────────────────────

@dataclass
class TemperatureSteps:
    setpoints_K: list[float]
    margin_K: float
    hold_s: float
    cfg: TemperatureControllerConfig
    # set by the run screen's "Skip T wait" button; cleared at each step
    skip_event: threading.Event = field(default_factory=threading.Event)


def steps_from_state(state: dict) -> Optional[TemperatureSteps]:
    """The setpoint steps of a validated state, or None when the program
    does not drive the iTC (log-only / manual setpoint, today's behaviour)."""
    if not state.get("temperature_control"):
        return None
    setpoints, err = parse_setpoints(state.get("temperature_setpoints_K", ""))
    cfg = temperature_cfg_from_state(state)
    if err or not setpoints or cfg is None:
        return None
    return TemperatureSteps(setpoints_K=setpoints,
                            margin_K=state.get("temperature_margin_mK", 5.0) * 1e-3,
                            hold_s=state.get("temperature_hold_s", 120.0), cfg=cfg)


def with_first_setpoint(state: dict) -> tuple[dict, Optional[TemperatureSteps]]:
    """(state, steps) for Start: with setpoint steps the first setpoint
    replaces the identity bar's hand-typed one, so the first plan (and a
    program's build_plan-time run allocation) is named after it."""
    steps = steps_from_state(state)
    if steps is None:
        return state, None
    return {**state, "temperature_setpoint_K": steps.setpoints_K[0]}, steps


def temperature_summary(state: dict, info: list, warnings: list, errors: list, *,
                        single_target: bool = False) -> None:
    """The sidebar's temperature lines (logging + control), appended in
    place. `single_target` = R vs T log: one setpoint, sent, never waited on."""
    logging_on = bool(state.get("enable_temperature"))
    uids = parse_sensor_uids(state.get("temperature_sensor_uids", "")) if logging_on else ()
    max_K = state.get("temperature_max_K", 320.0)
    if logging_on and not uids:
        warnings.append("Temperature logging is on but no sensor UID is set — "
                        "temperature columns will be empty.")
    elif logging_on:
        info.append(f"Temperature: MercuryiTC {', '.join(uids)} — empty if unreachable; "
                    f"heater forced off at ≥ {max_K:g} K")
    else:
        info.append("Temperature: off")
    if logging_on and max_K <= 0:
        errors.append("Software max temperature must be > 0 K.")

    if not state.get("temperature_control"):
        return
    if not uids:
        errors.append("Driving the iTC needs temperature logging on with a sensor UID "
                      "(the first UID's loop is the one driven).")
    setpoints, err = parse_setpoints(state.get("temperature_setpoints_K", ""))
    if err:
        errors.append(err)
    elif not setpoints:
        errors.append("Driving the iTC is on but no setpoint is given.")
    hot = [t for t in setpoints if t >= max_K]
    if hot:
        errors.append(f"Setpoint {hot[0]:g} K is at/above the software max {max_K:g} K.")
    if single_target:
        if len(setpoints) > 1:
            errors.append("R vs T takes ONE target setpoint (sent at start, logged while T drifts).")
        elif setpoints:
            info.append(f"Temperature target: {setpoints[0]:g} K — sent at start, logged "
                        "while it drifts there (no wait); filename T from it, identity "
                        "setpoint ignored")
        return
    margin_mK = state.get("temperature_margin_mK", 5.0)
    hold_s = state.get("temperature_hold_s", 120.0)
    if margin_mK <= 0:
        errors.append("Temperature margin must be > 0 mK.")
    if hold_s < 0:
        errors.append("Temperature hold time must be ≥ 0 s.")
    if setpoints:
        info.append(f"Temperature steps: {len(setpoints)} — {', '.join(f'{t:g}' for t in setpoints)} K "
                    f"(outer loop) — ±{margin_mK:g} mK held {hold_s:g} s each")
        info.append(f"Total time: {len(setpoints)} × the run time above + heating/settling "
                    f"(≥ {len(setpoints)} × {hold_s:g} s) — filename T from each setpoint, "
                    "identity setpoint ignored; last setpoint stays on the iTC")


def repeat_run_cost(rc: Optional[RunCost], steps: Optional[TemperatureSteps]) -> Optional[RunCost]:
    """The run's cost model repeated once per temperature step (each step is
    a full run), with the minimum settle hold charged before each step's
    first point — for the progress bar."""
    if rc is None or steps is None:
        return rc
    n = len(steps.setpoints_K)
    out = RunCost(0)
    out.points = list(rc.points) * n
    out.parts = {k: v * n for k, v in rc.parts.items()}
    out.tail_s = rc.tail_s * n
    for k in range(n):
        out.at("T settle", steps.hold_s, k * len(rc.points))
    return out


def go_to_temperature(steps: TemperatureSteps, index: int, stop_event: threading.Event,
                      on_status: Callable[[str], None]) -> str:
    """Send setpoint `index` and block until it has settled (or the
    operator skips / aborts). Own short iTC session, closed again before
    the program connects its own for logging. Returns the wait outcome."""
    T = steps.setpoints_K[index]
    n = len(steps.setpoints_K)
    steps.skip_event.clear()
    cfg = replace(steps.cfg)                       # connect narrows sensor_uids in place
    on_status(f"T step {index + 1}/{n}: connecting to the MercuryiTC …")
    mitc = connect_temperature_controller(cfg)
    if mitc is None:
        raise RuntimeError("Temperature control is on but the MercuryiTC is unreachable "
                           f"(or no sensor answered) at {cfg.visa_resource}.")
    try:
        loop_uid = normalize_uid(steps.cfg.sensor_uids[0])
        if cfg.sensor_uids[0] != loop_uid:
            raise RuntimeError(f"MercuryiTC loop sensor {loop_uid} does not answer — "
                               "refusing to drive a different sensor's loop.")
        set_temperature(mitc, cfg, T)

        def progress(current_K: float, in_margin_s: float) -> None:
            on_status(f"T step {index + 1}/{n} → {T:g} K · now {current_K:.4f} K "
                      f"(Δ {1e3 * (current_K - T):+.1f} mK, margin ±{1e3 * steps.margin_K:g} mK) · "
                      f"in margin {in_margin_s:.0f}/{steps.hold_s:g} s — Skip wait starts now")

        return wait_for_temperature_stable(
            mitc, cfg, T, tolerance_K=steps.margin_K, hold_time_s=steps.hold_s,
            timeout_s=None, poll_interval_s=POLL_INTERVAL_S,
            stop_event=stop_event, skip_event=steps.skip_event, on_progress=progress)
    finally:
        shutdown_temperature_controller(mitc)     # closes only: the setpoint stays active


def _ignore(*_args) -> None:
    pass


def run_temperature_steps(
    steps: TemperatureSteps, state: dict, data_root: Path, first_plan,
    build_plan: Callable[[dict, Path], object], engine: Callable[[object], object],
    stop_event: threading.Event, *,
    on_status: Callable[[str], None] = _ignore,
    on_run_label: Callable[[str], None] = _ignore,
    on_point: Callable[[dict], None] = _ignore,
    on_run_finished: Optional[Callable] = None,
    run_contexts: Optional[list] = None, run_extras: Optional[list] = None,
    on_plan: Callable[[object], None] = _ignore,
) -> None:
    """Run the whole program once per setpoint (outer loop). `first_plan`
    must already be built with `temperature_setpoint_K = setpoints_K[0]`;
    later steps rebuild the plan with their own T (a fresh run allocation
    for programs that allocate in build_plan). `engine(plan)` is the module
    whose run_plan runs it. `on_plan(plan)` fires before each step, so the
    front end's post-run status/comment uses the LAST step's plan.

    Records are re-tagged so series_index stays unique across the session
    (finish_last_run picks the last run by it) and series_label names T."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    for k, T in enumerate(steps.setpoints_K):
        if stop_event.is_set():
            break
        if go_to_temperature(steps, k, stop_event, on_status) == "stopped":
            break
        plan = first_plan if k == 0 else build_plan({**state, "temperature_setpoint_K": T}, data_root)
        on_plan(plan)
        offset = len(run_contexts)

        def tagged(record: dict, _T=T, _offset=offset) -> None:
            record["series_index"] = record.get("series_index") or 0
            record["series_index"] += _offset
            label = record.get("series_label")
            if not str(label or "").startswith(f"T={_T:g}K"):
                record["series_label"] = f"T={_T:g}K" + (f" {label}" if label else "")
            on_point(record)

        on_status(f"T step {k + 1}/{len(steps.setpoints_K)} at {T:g} K — running …")
        engine(plan).run_plan(plan, stop_event, on_status=on_status, on_run_label=on_run_label,
                              on_point=tagged, on_run_finished=on_run_finished,
                              run_contexts=run_contexts, run_extras=run_extras)
