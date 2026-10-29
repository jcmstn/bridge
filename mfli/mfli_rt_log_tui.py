#!/usr/bin/env python3
"""
Lock-in R vs T log — the "6221 AC + MFLI pair" read mode of the R vs T form
===========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-10-29

Engine module (no App of its own) behind the R vs T log form's
"Read" toggle (dc/dc_rt_log_tui.py), the way mfli_dual_harmonic_6221_tui.py
sits behind the dual-harmonic form: this module owns the lock-in mode's plan
type, run_plan, header, PNG and RunScreen, and saves as type `RTAC`. The form
builds the plan (build_lockin_plan in dc_rt_log_tui.py).

run_plan: LabOne + both MFLIs → MDS → Signal Outputs off → demodulators →
6221 AC sine on → both ExtRef PLLs locked to its marker → MercuryiTC (+ the
optional target, no wait) → fixed field parked + settled → one run, logged
by mfli_rt_log.run_measurement until Stop / max duration / T_stop. Stop is
the normal end (saved "completed"); the 6221, magnet, gaussmeter and iTC are
always shut down.

Run with:
    python mfli/mfli_rt_log_tui.py      (opens the R vs T form on this mode)
"""

from __future__ import annotations

import logging
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from dc.dc_rt_log import AcquisitionConfig, live_plot_worker, plot_results
from dc.dc_sweep_utils import safe_shutdown
from instruments.data_naming import RunContext, allocate_run, record_run
from instruments.keithley6221 import ACSourceConfig, connect_ac_source, shutdown_ac_source
from instruments.kepco_magnet import MagnetConfig, connect_magnet, set_magnet_current, shutdown_magnet
from instruments.lakeshore475 import GaussmeterConfig, connect_gaussmeter, shutdown_gaussmeter
from instruments.mercury_itc import (
    TemperatureControllerConfig, connect_temperature_controller, normalize_uid, set_temperature,
    shutdown_temperature_controller,
)
from instruments.mfli_daq import (
    ExtRefConfig, configure_external_reference, connect, connect_device, setup_mds,
    wait_for_reference_lock,
)
from instruments.run_time import RunCost
from instruments.tui_common import MeasurementRunScreen
from mfli.mfli_dual_harmonic import DemodConfig, SampleGeometryConfig, configure_demodulator
from mfli.mfli_dual_harmonic_6221 import _check_ac_safety, build_run_metadata, disable_sigout
from mfli.mfli_dual_harmonic_6221_tui import plan_naming
from mfli.mfli_rt_log import run_measurement

log = logging.getLogger("mfli_rt_log_tui")

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Locked type code (see docs/data_convention.md) — never deviates.
MEASUREMENT_TYPE = "RTAC"
PNG_SUFFIX = "R_vs_T"


@dataclass
class MeasurementPlan:
    daq_host: str
    daq_port: int
    leader: str
    follower: str
    ac_cfg: ACSourceConfig
    leader_extref_cfg: ExtRefConfig
    follower_extref_cfg: ExtRefConfig
    extref_lock_timeout_s: float
    demod1_cfg: DemodConfig
    demod2_cfg: DemodConfig
    leader_measure_rxx: bool
    measure_rxx: bool                   # follower's R_xx naming toggle
    n_averages: int
    settling_time_s: float
    acq_cfg: AcquisitionConfig
    sample: str
    device: str
    temperature_setpoint_K: Optional[float]
    cooldown: str
    header_extra: dict
    temp_cfg: Optional[TemperatureControllerConfig] = None
    data_root: Path = _DEFAULT_DATA_DIR
    run_cost: Optional[RunCost] = None
    total_points: int = 1               # upper bound (max duration) — progress bar only
    series: str = ""
    magnet_cfg: Optional[MagnetConfig] = None       # None = no fixed field
    gauss_cfg: Optional[GaussmeterConfig] = None
    magnet_current_A: Optional[float] = None
    field_settle_tolerance_mT: Optional[float] = None
    field_theta_deg: Optional[float] = None
    field_phi_deg: Optional[float] = None
    temperature_target_K: Optional[float] = None    # sent to the iTC at start (no wait)


def y_cols(plan: MeasurementPlan) -> tuple:
    """(column, label) of each MFLI's in-phase resistance — what gets plotted."""
    return tuple((f"{prefix}_X_ohm", f"{display} = X / I_rms (Ω)")
                 for prefix, display in plan_naming(plan))


def build_header_fields(plan: MeasurementPlan, ctx: RunContext, records: list[dict], *,
                        status: str, comment: str, extra: Optional[dict] = None) -> dict:
    """Universal header/index fields, the measured temperature range, and
    `extra` (sensor UIDs used + the lock-in metadata read once at start)."""
    measured = [r["temperature_1_K"] for r in records if r.get("temperature_1_K") is not None]
    fields = {
        "run": ctx.run_number,
        "timestamp": ctx.timestamp.isoformat(timespec="seconds"),
        "sample": ctx.sample,
        "device": ctx.device,
        "type": MEASUREMENT_TYPE,
        "T_setpoint_K": plan.temperature_setpoint_K,
        "T_K": (sum(measured) / len(measured)) if measured else "",
        "cooldown": plan.cooldown,
        "status": status,
        "comment": comment,
        "series": plan.series,
    }
    fields.update(plan.header_extra)
    fields["T_min_K"] = min(measured) if measured else ""
    fields["T_max_K"] = max(measured) if measured else ""
    if extra:
        fields.update(extra)
    return fields


def save_run_png(plan: MeasurementPlan, records: list[dict], png_path: Path, comment: str = "") -> None:
    """One run's PNG (RunScreen calls this)."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    note = f"Comment: {textwrap.shorten(comment, width=90, placeholder='…')}" if comment else ""
    plot_results(pd.DataFrame(records), png_path, note=note, y_cols=y_cols(plan))


def _ignore(*_args) -> None:
    pass


def run_plan(plan: MeasurementPlan, stop_event: threading.Event, *,
             on_status: Callable[[str], None] = _ignore,
             on_run_label: Callable[[str], None] = _ignore,
             on_point: Callable[[dict], None] = _ignore,
             on_run_finished: Optional[Callable[[RunContext, list], None]] = None,
             run_contexts: Optional[list] = None,
             run_extras: Optional[list] = None) -> None:
    """Connect + lock everything, park the magnet (fixed field), log one run
    until Stop / max duration / T_stop, finalize it, and always switch the
    6221 off and ramp the magnet down."""
    run_contexts = [] if run_contexts is None else run_contexts
    run_extras = [] if run_extras is None else run_extras
    (leader_prefix, _), (follower_prefix, _) = plan_naming(plan)
    source = temp_ctrl = magnet = gaussmeter = None
    try:
        on_status("Connecting to LabOne data server …")
        daq = connect(plan.daq_host, plan.daq_port)
        connect_device(daq, plan.leader, interface="1GbE")
        connect_device(daq, plan.follower, interface="1GbE")
        on_status("Synchronizing MDS …")
        mds = setup_mds(daq, leader=plan.leader, follower=plan.follower)
        disable_sigout(daq, plan.leader)
        disable_sigout(daq, plan.follower)
        on_status("Configuring demodulators …")
        configure_demodulator(daq, plan.demod1_cfg)
        configure_demodulator(daq, plan.demod2_cfg)

        _check_ac_safety(plan.ac_cfg)       # before connect_ac_source() turns the 6221 on
        on_status("Starting 6221 AC current source …")
        source = connect_ac_source(plan.ac_cfg)
        on_status("Locking MFLI oscillators to the 6221 marker (ExtRef) …")
        for role, cfg in (("Leader", plan.leader_extref_cfg), ("Follower", plan.follower_extref_cfg)):
            configure_external_reference(daq, cfg, plan.ac_cfg.frequency_Hz)
        for role, cfg in (("Leader", plan.leader_extref_cfg), ("Follower", plan.follower_extref_cfg)):
            if not wait_for_reference_lock(daq, cfg, plan.extref_lock_timeout_s, stop_event):
                log.warning("%s ExtRef PLL did not report locked within %.2g s — check the "
                            "marker cabling before trusting any data.", role, plan.extref_lock_timeout_s)

        loop_uid = normalize_uid(plan.temp_cfg.sensor_uids[0]) if plan.temp_cfg is not None else None
        if plan.temp_cfg is not None:
            on_status("Connecting to MercuryiTC and probing its sensors (see log) …")
            temp_ctrl = connect_temperature_controller(plan.temp_cfg)   # narrows sensor_uids
        temp_cfg = plan.temp_cfg if temp_ctrl is not None else None
        if plan.temperature_target_K is not None:
            # load-bearing once asked for (same rule as the DC log)
            if temp_cfg is None or loop_uid not in temp_cfg.sensor_uids:
                raise RuntimeError(f"Cannot send the temperature target: MercuryiTC loop sensor "
                                   f"{loop_uid} is not answering.")
            set_temperature(temp_ctrl, temp_cfg, plan.temperature_target_K, uid=loop_uid)
            on_status(f"MercuryiTC target {plan.temperature_target_K:g} K sent — logging while it drifts there …")

        if plan.magnet_cfg is not None:
            on_status("Connecting magnet power supply & gaussmeter …")
            magnet = connect_magnet(plan.magnet_cfg)
            gaussmeter = connect_gaussmeter(plan.gauss_cfg)
            on_status(f"Ramping magnet to {plan.magnet_current_A:g} A and waiting for the field to settle …")
            set_magnet_current(magnet, plan.magnet_cfg, plan.magnet_current_A, gaussmeter,
                               plan.gauss_cfg, plan.field_settle_tolerance_mT, stop_event)
            if stop_event.is_set():     # stopped during the ramp: no empty run file
                return

        ctx = allocate_run(plan.data_root, plan.sample, plan.device, MEASUREMENT_TYPE,
                           temperature_setpoint_K=plan.temperature_setpoint_K)
        # Frequency (the PLL's tracked value), phases, filters: read once, into the header.
        extra = {
            "temperature_sensor_uids_used": ",".join(temp_cfg.sensor_uids) if temp_cfg else "",
            **build_run_metadata(daq, plan.ac_cfg, plan.leader_extref_cfg, plan.demod1_cfg,
                                 plan.demod2_cfg,
                                 SampleGeometryConfig(field_theta_deg=plan.field_theta_deg,
                                                      field_phi_deg=plan.field_phi_deg),
                                 measure_rxx=plan.measure_rxx),
        }
        run_contexts.append(ctx)
        run_extras.append(extra)
        on_run_label(f"Run #{ctx.run_str}")
        plan.acq_cfg.output_file = str(ctx.raw_path)

        on_status("Logging … (Stop ends the log and saves it as completed)")
        # stop_event is NOT passed to record_run: Stop is how a log normally
        # ends, so it finalizes as "completed" (an exception still -> "error").
        record_run(
            plan.data_root, ctx,
            lambda records, status: build_header_fields(plan, ctx, records, status=status,
                                                        comment="", extra=extra),
            lambda point_cb, write_csv: run_measurement(
                daq, plan.ac_cfg, plan.leader_extref_cfg, plan.follower_extref_cfg,
                plan.demod1_cfg, plan.demod2_cfg, plan.acq_cfg, plan.n_averages,
                plan.settling_time_s, stop_event=stop_event, on_point=point_cb,
                temp_ctrl=temp_ctrl, temp_cfg=temp_cfg, write_csv=write_csv,
                gaussmeter=gaussmeter, gauss_cfg=plan.gauss_cfg,
                magnet_current_A=plan.magnet_current_A,
                field_theta_deg=plan.field_theta_deg, field_phi_deg=plan.field_phi_deg,
                mds=mds, demod1_label=leader_prefix, demod2_label=follower_prefix),
            None, on_point=on_point, on_finished=on_run_finished)
    finally:
        # 6221 output off first (immediate), then the magnet ramp.
        if source is not None:
            safe_shutdown("6221 AC source", lambda: shutdown_ac_source(source))
        if magnet is not None:
            safe_shutdown("magnet", lambda: shutdown_magnet(magnet, plan.magnet_cfg))
        if gaussmeter is not None:
            safe_shutdown("gaussmeter", lambda: shutdown_gaussmeter(gaussmeter))
        if temp_ctrl is not None:
            safe_shutdown("MercuryiTC", lambda: shutdown_temperature_controller(temp_ctrl))


def _fmt(value: Optional[float], spec: str) -> str:
    return format(value, spec) if value is not None else "—"


class RunScreen(MeasurementRunScreen):
    ABORT_LABEL = "Stop logging"
    ABORT_STATUS = "Stop requested — finishing the current sample, then 6221 off (and magnet to zero) …"
    DONE_STATUS = "Log finished."
    STOP_IS_NORMAL_END = True      # Stop ends a log normally -> "completed" in the run history
    POINT_STATUS = "Sample {n} (at most {total})."
    MEASUREMENT_TYPE = MEASUREMENT_TYPE
    PNG_SUFFIX = PNG_SUFFIX

    def __init__(self, plan: MeasurementPlan) -> None:
        super().__init__(plan)
        self._y = y_cols(plan)

    def table_columns(self) -> tuple:
        (_, lead), (_, fol) = plan_naming(self.plan)
        return ("#", "t (s)", "T1 (K)", "ΔT1 (mK)", "T2 (K)",
                f"{lead} X/I (Ω)", f"{fol} X/I (Ω)", "B (mT)")

    def live_plot_args(self):
        return (live_plot_worker, self._y)

    def table_row(self, record: dict) -> tuple:
        drift = record.get("temperature_1_drift_K")
        return (
            str(record["point_index"] + 1),
            f"{record['elapsed_s']:.1f}",
            _fmt(record.get("temperature_1_K"), ".3f"),
            _fmt(drift * 1e3 if drift is not None else None, ".1f"),
            _fmt(record.get("temperature_2_K"), ".3f"),
            *(_fmt(record.get(col), ".6g") for col, _ in self._y),
            _fmt(record.get("magnet_field_mT"), ".2f"),
        )


def main() -> None:
    """A mode of the R vs T form — open it with Read set to the lock-ins."""
    from dc.dc_rt_log_tui import DCRTLogApp
    DCRTLogApp(read_mode="lockin").run()


if __name__ == "__main__":
    main()
