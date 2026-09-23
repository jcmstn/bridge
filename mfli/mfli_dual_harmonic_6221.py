#!/usr/bin/env python3
"""
Dual MFLI Lock-in Harmonic Measurement, 6221-sourced AC current
=================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-14

Same measurement as mfli_dual_harmonic.py (leader reads 1f, follower reads
2f, MDS-synced, optional Kepco field sweep + Lake Shore 475 + MercuryiTC +
phase calibration + sample geometry) — the only thing that changes is WHO
sources the AC excitation current:

    mfli_dual_harmonic.py:       MFLI Signal Output → [R_series] → sample
    this module:                 Keithley 6221 WAVE (sine) → sample directly

The 6221 is an ideal AC current source (no series-resistor V/R assumption,
no series-resistor Johnson noise added to the signal path) — see
instruments/keithley6221.py::ACSourceConfig, the same mechanism
sot/sot_pulsed_switching_6221.py uses for its AC read phase.

Wiring
------
    Keithley 6221 (WAVE, sine, continuous)   HI ──▶ I+ pad ;  LO ──▶ I- pad

    Keithley 6221 TRIGGER LINK, phase marker on pin ``ACSourceConfig.
    phasemarker_line`` (default 1, matching this lab's cable — DIN pin N =
    Trigger Link line N, confirmed against the 622x Reference Manual; the
    6221's own factory default is line 3, same as Zurich Instruments' own
    MFLI/6221 external-reference guide, but that pin isn't what this rig's
    cable brings out — confirm against your own cabling before assuming
    either default) ──▶ split (BNC T or power divider, EQUAL cable lengths)
    to AUX IN 1 on **BOTH** MFLIs.

    Leader MFLI  Signal Input 1 (differential) ──▶ demod 1f (R_xy)
    Follower MFLI  Signal Input 1 (differential) ──▶ demod 2f (R_xy), OR,
      with `measure_rxx` on, the R_xx voltage leads instead ──▶ demod 1f
      (R_xx) — R_xx and R_xy's 2f can't be read at once with only two
      physical MFLIs (each needs its own Signal Input for an independent
      input range), so `measure_rxx` trades one for the other: move the
      follower's Signal Input BNC by hand between the R_xy and R_xx probe
      pairs depending on which mode you're running. The leader always
      reads R_xy 1f either way.

    MDS cabling (both units, same as mfli_dual_harmonic.py):
      Leader Ref Out       ───BNC───▶ Follower Ref In
      Leader Trigger Out 1  ──▶ fanned out to Trigger In 1 on BOTH units

    Kepco BOP-GL ──GPIB──▶ electromagnet ;  Lake Shore 475 ──GPIB──▶ Gaussmeter

Why the marker must reach BOTH MFLIs, not just the leader
-----------------------------------------------------------
It is tempting to wire the marker only into the leader's Aux In 1 and let
MDS carry the frequency to the follower the way mfli_dual_harmonic.py's
sync_follower_oscillator() does — copying a frequency *value* between two
devices whose oscillators are still independent NCOs. That works when both
oscillators are internally generated (both devices ultimately share the
same MDS-distributed sample clock, so a matched frequency *value* is also
phase-coherent). It stops working the instant the leader's oscillator is
instead phase-locked to an *external* reference (the 6221's crystal): the
leader now tracks the 6221's clock, while the follower's oscillator — even
set to the identical frequency value — still free-runs on the MFLI's own
clock. Any ppm-level offset between the two clocks then accumulates as a
slow rotation of the follower's demodulated phasor relative to the drive.

That is not a small phase error — acquire_averaged() (instruments/mfli_daq.py)
vector-averages X/Y before computing R (R = hypot(mean_X, mean_Y), the
statistically correct choice, see its docstring), so a phasor that rotates
across the averaging window collapses R2f toward *zero*, not toward a wrong
angle. At a few hundred Hz and a plausible clock offset this can happen
within a single sweep's timescale, and it looks exactly like "no signal"
rather than an obviously wrong number.

The fix used here: lock **both** MFLIs' oscillators to the same external
6221 marker (configure_external_reference() runs for leader and follower
alike). Both are then anchored to the actual drive, and the only remaining
leader/follower difference is a static cable/electronics delay — exactly
what null_follower_reference_via_1f() already measures and records as
demod2_phase_null_1f_deg. MDS is still configured and still checked
per-point (mds_synced) for a common sample clock and start instant; it is
simply no longer what keeps the two demodulators frequency-coherent.

Bench-verify before trusting a run
-----------------------------------
configure_external_reference() logs the live ``extrefs`` node tree on first
connect for each device — read it. Confirm both devices report ``locked``
before trusting any data (the per-point ``leader_reference_locked`` /
``follower_reference_locked`` columns tag this, but a run that starts
unlocked and stays unlocked is not useful data, just a documented failure).

``ExtRefConfig.pll_demod_index`` needs a real, free demodulator on the
device (confirmed against ``docs.zhinst.com/mfli_user_manual/nodedoc.html``
and a live device dump: ``extrefs/N/adcselect``/``oscselect`` are
READ-ONLY — the PLL is steered by pointing a dedicated demodulator's own
``adcselect``/``oscselect`` at the marker/oscillator and wiring it in via
``extrefs/N/demodselect``). Default is demod index 1, distinct from
demod1_cfg/demod2_cfg's index 0 — if your unit doesn't have that many
demods (no MF-MD/multi-demod option), pick a genuinely free index or this
raises the same read-only-node error one step later.

Requirements: zhinst-core, zhinst-utils, numpy, pandas, pyvisa, pymeasure.
"""

import logging
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import zhinst.core as zi

from instruments.keithley6221 import ACSourceConfig, connect_ac_source, shutdown_ac_source
from instruments.mfli_daq import (
    connect, connect_device, setup_mds, check_mds_status,
    acquire_averaged, acquire_averaged_pair,
    ExtRefConfig, configure_external_reference, wait_for_reference_lock,
    check_reference_locked,
)
from instruments.kepco_magnet import (
    MagnetConfig,
    connect_magnet,
    set_magnet_current,
    shutdown_magnet,
)
from instruments.lakeshore475 import (
    LakeShore475,
    GaussmeterConfig,
    connect_gaussmeter,
    read_field_mT,
    shutdown_gaussmeter,
)
from instruments.mercury_itc import (
    MercuryITC,
    TemperatureControllerConfig,
    connect_temperature_controller,
    read_temperature,
    shutdown_temperature_controller,
)
from mfli.mfli_dual_harmonic import (
    AcquisitionConfig,
    DemodConfig,
    FilterConfig,
    MeasurementPoint,
    SampleGeometryConfig,
    auto_null_phase,
    bidirectional_current_sweep,
    configure_demodulator,
    get_demod_phase_deg,
    null_follower_reference_via_1f,
)
from instruments.run_time import GPIB_TXN_S, LOCK_TYP_S

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Safety ceiling — same rationale as sot_pulsed_switching_6221.py's read phase:
# this isn't a margin against the 6221's own hardware range, it's a guard
# against a mistyped exponent driving continuous current into the DUT.
# ─────────────────────────────────────────────────────────────────────────────

_AC_CURRENT_CEILING_A    = 10e-3
_AC_COMPLIANCE_CEILING_V = 21.0


def _check_ac_safety(ac_cfg: ACSourceConfig) -> None:
    if not 0 < ac_cfg.amplitude_A <= _AC_CURRENT_CEILING_A:
        raise ValueError(
            f"amplitude_A must be in (0, {_AC_CURRENT_CEILING_A} A]; got "
            f"{ac_cfg.amplitude_A} A. A harmonic-Hall excitation needs "
            "microamps-to-milliamps — check for a mistyped exponent.")
    if not 0 < ac_cfg.compliance_V <= _AC_COMPLIANCE_CEILING_V:
        raise ValueError(
            f"compliance_V must be in (0, {_AC_COMPLIANCE_CEILING_V} V]; got "
            f"{ac_cfg.compliance_V} V.")


# ─────────────────────────────────────────────────────────────────────────────
# External-reference (ExtRef) setup — the PLL itself (ExtRefConfig,
# configure_external_reference, wait_for_reference_lock,
# check_reference_locked) is shared, in instruments/mfli_daq.py.
# ─────────────────────────────────────────────────────────────────────────────

                                    #   2 = low_bandwidth (most forgiving acquisition, best for a
                                    #       marginal/noisy signal), 3 = high_bandwidth (fastest
                                    #       tracking once locked, least noise tolerance), 4 = all/
                                    #       dynamic (auto-adapts — the default). Left at whatever the
                                    #       device last had if never set, which could be a bandwidth
                                    #       tuned for a different signal from a previous run.


def extref_lock_s(lock_timeout_s: float) -> tuple[float, float]:
    """Modelled (typical, worst-case) wall time of arming BOTH devices' ExtRef
    PLLs: per device, configure_external_reference() is 12 LabOne transactions
    and wait_for_reference_lock() returns at the first lock -- run_time.LOCK_TYP_S
    typically, the full `lock_timeout_s` if the marker never locks."""
    setup = 2 * 12 * GPIB_TXN_S
    return setup + 2 * LOCK_TYP_S, setup + 2 * lock_timeout_s


def disable_sigout(daq: "zi.ziDAQServer", device: str, out_ch: int = 0) -> None:
    """Force a device's own Signal Output off. The 6221 is the only source
    on the DUT pins in this module; a Signal Output left enabled from a
    previous mfli_dual_harmonic.py run on either device would otherwise
    fight it. Harmless to call when the output is already off."""
    daq.setInt(f"/{device}/sigouts/{out_ch}/on", 0)
    daq.sync()
    log.info("MFLI %s Signal Output %d forced off (6221 is the current source)",
             device, out_ch)


# ─────────────────────────────────────────────────────────────────────────────
# Run metadata
# ─────────────────────────────────────────────────────────────────────────────

def build_run_metadata(
    daq: "zi.ziDAQServer",
    ac_cfg: ACSourceConfig,
    leader_extref_cfg: ExtRefConfig,
    demod1_cfg: DemodConfig,
    demod2_cfg: DemodConfig,
    geometry_cfg: Optional[SampleGeometryConfig] = None,
    demod2_phase_null_1f_deg: Optional[float] = None,
    measure_rxx: bool = False,
) -> dict:
    """Same shape/purpose as mfli_dual_harmonic.build_run_metadata(), with
    the excitation terms sourced from the 6221 instead: amplitude_A is
    already the peak excitation current (an ideal current source — no
    series-resistor V/R assumption), and excitation_frequency_Hz is read
    live from the leader's ExtRef-locked oscillator (the PLL's true tracked
    frequency, not the 6221's commanded value — matches the convention
    already used by sot_pulsed_switching_6221.py for the same reason).

    `measure_rxx` is recorded verbatim so a downstream analysis script can
    tell what the follower's columns mean (R_xy 2f vs R_xx 1f) without
    re-deriving it from which column prefix happens to be present.
    """
    geometry_cfg = geometry_cfg or SampleGeometryConfig()
    I_peak_A = ac_cfg.amplitude_A
    excitation_frequency_Hz = daq.getDouble(
        f"/{leader_extref_cfg.device}/oscs/{leader_extref_cfg.osc_index}/freq")
    return {
        "measure_rxx": measure_rxx,
        "demod2_phase_null_1f_deg": demod2_phase_null_1f_deg,
        "excitation_frequency_Hz":       excitation_frequency_Hz,
        "excitation_current_A_peak":     I_peak_A,
        "excitation_current_A_rms":      I_peak_A / math.sqrt(2.0),
        "excitation_current_convention": (
            "peak; Keithley 6221 waveform_amplitude is peak, not RMS — an "
            "ideal current source, no series-resistor V/R assumption"
        ),
        "demod_output_convention": (
            "RMS; ZI demodulator X/Y/R nodes report the RMS amplitude of "
            "the input signal's component at the reference frequency"
        ),
        "demod1_time_constant_s":   demod1_cfg.filter.time_constant_s,
        "demod1_filter_order":      demod1_cfg.filter.order,
        "demod1_ref_phase_deg":     get_demod_phase_deg(daq, demod1_cfg),
        "demod2_time_constant_s":   demod2_cfg.filter.time_constant_s,
        "demod2_filter_order":      demod2_cfg.filter.order,
        "demod2_ref_phase_deg":     get_demod_phase_deg(daq, demod2_cfg),
        "hall_bar_length_um":       geometry_cfg.hall_bar_length_um,
        "hall_bar_width_um":        geometry_cfg.hall_bar_width_um,
        "hall_bar_thickness_nm":    geometry_cfg.hall_bar_thickness_nm,
        "field_theta_deg":          geometry_cfg.field_theta_deg,
        "field_phi_deg":            geometry_cfg.field_phi_deg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop  (docs/architecture.md §3 contract)
# ─────────────────────────────────────────────────────────────────────────────

def run_measurement(
    daq:        zi.ziDAQServer,
    ac_cfg:     ACSourceConfig,
    leader_extref_cfg:   ExtRefConfig,
    follower_extref_cfg: ExtRefConfig,
    demod1_cfg: DemodConfig,          # 1f channel (leader)
    demod2_cfg: DemodConfig,          # 2f channel (follower)
    acq_cfg:    AcquisitionConfig,
    points:     List[MeasurementPoint],
    stop_event: Optional[threading.Event] = None,
    on_point:   Optional[Callable[[dict], None]] = None,
    gaussmeter: Optional[LakeShore475] = None,
    gauss_cfg:  Optional[GaussmeterConfig] = None,
    temp_ctrl: Optional[MercuryITC] = None,
    temp_cfg:  Optional[TemperatureControllerConfig] = None,
    geometry_cfg: Optional[SampleGeometryConfig] = None,
    demod2_phase_null_1f_deg: Optional[float] = None,
    mds=None,
    write_csv: Optional[Callable[[List[dict]], None]] = None,
    demod2_label: str = "2f",
) -> pd.DataFrame:
    """Same loop shape as mfli_dual_harmonic.run_measurement(): iterate
    `points`, acquire 1f (leader) + demod2 (follower) at each, log to CSV,
    return a DataFrame. The 6221 (`source`, connected by the caller via
    connect_ac_source()) is not touched here — it is not re-armed per
    point, since the AC excitation runs continuously for the whole sweep
    (unlike sot_pulsed_switching_6221.py's pulsed write/read cycling).

    Every point re-checks BOTH ExtRef PLLs (leader_reference_locked /
    follower_reference_locked, via check_reference_locked()) in addition to
    the usual mds_synced check — a dropped marker cable is now a second,
    independent way this measurement can silently go bad (see the module
    docstring), so it gets the same "log + tag the row, don't abort" policy
    MDS already has.

    `demod2_label` names the follower's column prefix — `"2f"` (default,
    today's R_xy 2f) or e.g. `"rxx_1f"` when the caller has set
    `demod2_cfg.harmonic=1` and physically rewired the follower's Signal
    Input to the R_xx probe pair (see the module docstring). Whether the
    caller passed a non-"2f" label is what `measure_rxx` in the recorded
    run metadata reflects.
    """
    _check_ac_safety(ac_cfg)
    measure_rxx = demod2_label != "2f"
    records: List[dict] = []

    for idx, pt in enumerate(points):
        if stop_event is not None and stop_event.is_set():
            log.info("Measurement aborted after %d / %d points.", idx, len(points))
            break

        if pt.magnet_current_A is not None:
            log.info("── Point %d / %d   I_magnet=%.4f A ──────────────────",
                      idx + 1, len(points), pt.magnet_current_A)
        else:
            log.info("── Point %d / %d ──────────────────────────────────", idx + 1, len(points))

        # ── 1. Apply external parameter ────────────────────────────────────
        if pt.set_action is not None:
            pt.set_action(daq)

        # ── 1b. MDS + ExtRef sync re-check ───────────────────────────────────
        mds_synced = check_mds_status(mds) if mds is not None else None
        if mds_synced is False:
            log.error("   MDS sync has dropped — check Ref/Trigger cabling.")
        leader_locked = check_reference_locked(daq, leader_extref_cfg)
        if leader_locked is False:
            log.error("   Leader ExtRef PLL has dropped lock — 1f data from "
                       "this point on may be corrupted until it relocks.")
        follower_locked = check_reference_locked(daq, follower_extref_cfg)
        if follower_locked is False:
            log.error("   Follower ExtRef PLL has dropped lock — 2f data from "
                       "this point on may be corrupted (garbage/beating "
                       "phasor) until it relocks. Check the marker fan-out "
                       "cabling to the follower's Aux In.")

        # ── 2. Settle ──────────────────────────────────────────────────────
        settle = pt.settling_override_s if pt.settling_override_s is not None \
                 else acq_cfg.settling_time_s
        log.info("   Settling %.2f s ...", settle)
        time.sleep(settle)

        # ── 3. Acquire 1f + demod2 together (one poll window, not two) ──────
        d1, d2 = acquire_averaged_pair(daq, demod1_cfg, demod2_cfg, acq_cfg.n_averages)
        log.info("   1f  R=%.4e V  θ=%.2f°  SEM_R=%.2e V  (n=%d)",
                 d1["r_mean"], d1["theta_mean"], d1["r_sem"], d1["n_samples"])
        if d1["overload"]:
            log.warning("   1f input is OVERLOADED — this reading is not trustworthy.")
        log.info("   %s  R=%.4e V  θ=%.2f°  SEM_R=%.2e V  (n=%d)",
                 demod2_label, d2["r_mean"], d2["theta_mean"], d2["r_sem"], d2["n_samples"])
        if d2["overload"]:
            log.warning("   %s input is OVERLOADED — this reading is not trustworthy.", demod2_label)

        # ── 4b. Measure field (Lake Shore 475 Gaussmeter) ───────────────────
        field_mT = None
        if gaussmeter is not None and gauss_cfg is not None:
            field_mT = read_field_mT(gaussmeter, gauss_cfg)
            log.info("   B=%.4f mT (measured)", field_mT)

        # ── 4c. Read temperature (MercuryiTC, optional) ─────────────────────
        temp_1_K, temp_2_K = read_temperature(temp_ctrl, temp_cfg) \
            if temp_cfg is not None else (None, None)

        # ── 4d. Run metadata (excitation, filters, phases, geometry) ────────
        run_meta = build_run_metadata(daq, ac_cfg, leader_extref_cfg, demod1_cfg,
                                      demod2_cfg, geometry_cfg, demod2_phase_null_1f_deg,
                                      measure_rxx=measure_rxx)

        # ── 5. Build record ────────────────────────────────────────────────
        record: dict = {
            "point_index": idx,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mds_synced":  mds_synced,
            "leader_reference_locked":   leader_locked,
            "follower_reference_locked": follower_locked,
            # ── Magnet sweep ─────────────────────────────────────────────────
            "magnet_current_A": pt.magnet_current_A,
            "magnet_field_mT":  field_mT,
            # ── Temperature (MercuryiTC) ─────────────────────────────────────
            "temperature_1_K":  temp_1_K,
            "temperature_2_K":  temp_2_K,
            # ── 1f ─────────────────────────────────────────────────────────
            "1f_X_V":      d1["x_mean"],
            "1f_Y_V":      d1["y_mean"],
            "1f_R_V":      d1["r_mean"],
            "1f_theta_deg":d1["theta_mean"],
            "1f_R_sem_V":  d1["r_sem"],
            "1f_n_samples":d1["n_samples"],
            "1f_overload": d1["overload"],
            # ── demod2 (R_xy 2f, or R_xx 1f when measure_rxx is on) ─────────
            f"{demod2_label}_X_V":      d2["x_mean"],
            f"{demod2_label}_Y_V":      d2["y_mean"],
            f"{demod2_label}_R_V":      d2["r_mean"],
            f"{demod2_label}_theta_deg":d2["theta_mean"],
            f"{demod2_label}_R_sem_V":  d2["r_sem"],
            f"{demod2_label}_n_samples":d2["n_samples"],
            f"{demod2_label}_overload": d2["overload"],
            # ── Run metadata (excitation/demod/geometry — see build_run_metadata) ──
            **run_meta,
        }
        records.append(record)

        if on_point is not None:
            on_point(record)

        # ── 6. Write incrementally (never lose data on a crash) ────────────
        if write_csv is not None:
            write_csv(records)
        else:
            Path(acq_cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(acq_cfg.output_file, index=False)

    log.info("Measurement complete. %d points saved to '%s'.", len(records), acq_cfg.output_file)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── configure your devices and sweep here ──────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Device IDs ──────────────────────────────────────────────────────────
    LEADER   = "dev7885"    # 1f measurement, ExtRef-locked to the 6221 marker
    FOLLOWER = "dev7886"    # 2f measurement, ALSO ExtRef-locked to the same marker

    # ── AC excitation (Keithley 6221, ideal current source) ──────────────────
    ac_cfg = ACSourceConfig(
        visa_resource    = "GPIB0::20::INSTR",
        amplitude_A      = 100e-9,    # A peak — recommended micro-to-milliamp band
        frequency_Hz     = 317.3,     # Hz — recommended ~300-1000 Hz band, away
                                       #   from 1/f noise and 50/60 Hz harmonics
        compliance_V     = 2.0,
        phasemarker_line = 1,         # Trigger Link line -> BOTH MFLIs' Aux In 1 (matches this lab's cable)
    )
    _check_ac_safety(ac_cfg)

    # ── Connect ─────────────────────────────────────────────────────────────
    daq = connect("localhost", 8004)
    connect_device(daq, LEADER,   interface="1GbE")
    connect_device(daq, FOLLOWER, interface="1GbE")

    # ── MDS ─────────────────────────────────────────────────────────────────
    # Common sample clock / start instant — see the module docstring for why
    # this no longer also carries oscillator frequency between devices.
    mds = setup_mds(daq, leader=LEADER, follower=FOLLOWER)

    # ── 6221 AC source (must be running before the ExtRef PLLs can lock) ─────
    source = connect_ac_source(ac_cfg)

    # ── Make sure neither MFLI is still driving its own Signal Output ────────
    disable_sigout(daq, LEADER)
    disable_sigout(daq, FOLLOWER)

    # ── ExtRef: BOTH devices lock to the 6221 marker (see module docstring) ──
    leader_extref_cfg   = ExtRefConfig(device=LEADER,   aux_input_ch=0, osc_index=0)
    follower_extref_cfg = ExtRefConfig(device=FOLLOWER, aux_input_ch=0, osc_index=0)
    configure_external_reference(daq, leader_extref_cfg,   ac_cfg.frequency_Hz)
    configure_external_reference(daq, follower_extref_cfg, ac_cfg.frequency_Hz)
    if not wait_for_reference_lock(daq, leader_extref_cfg, timeout_s=5.0):
        log.warning("Leader ExtRef PLL did not report locked — check the "
                    "marker cabling before trusting any data.")
    if not wait_for_reference_lock(daq, follower_extref_cfg, timeout_s=5.0):
        log.warning("Follower ExtRef PLL did not report locked — check the "
                    "marker fan-out cabling before trusting any data.")

    # ── Filters ─────────────────────────────────────────────────────────────
    # Independent per harmonic: the follower's 2f channel needs real stopband
    # attenuation against 1f bleed-through (order/sinc) that the leader's 1f
    # channel doesn't, and shouldn't inherit 2f's settling-time cost.
    filter_1f = FilterConfig(time_constant_s=0.3, order=4, sinc_filter=True)
    filter_2f = FilterConfig(time_constant_s=0.3, order=4, sinc_filter=True)

    # ── 1f demodulator (leader) ───────────────────────────────────────────────
    demod1_cfg = DemodConfig(
        device=LEADER, demod_index=0, harmonic=1, osc_index=leader_extref_cfg.osc_index,
        input_range_V=1.0, sample_rate_Hz=857.0, filter=filter_1f,
    )
    configure_demodulator(daq, demod1_cfg)

    # ── 2f demodulator (follower) ─────────────────────────────────────────────
    demod2_cfg = DemodConfig(
        device=FOLLOWER, demod_index=0, harmonic=2, osc_index=follower_extref_cfg.osc_index,
        input_range_V=1.0, sample_rate_Hz=857.0, filter=filter_2f,
    )
    configure_demodulator(daq, demod2_cfg)

    # ── Acquisition settings ─────────────────────────────────────────────────
    acq_cfg = AcquisitionConfig(
        settling_time_s = 15,
        n_averages      = 50,
        output_file     = str(_DATA_DIR / f"harmonic_hall_6221_{datetime.now():%Y%m%d_%H%M%S}.csv"),
    )

    # ── Magnet (Kepco BOP-GL current source) ─────────────────────────────────
    magnet_cfg = MagnetConfig(
        visa_resource="GPIB0::6::INSTR", current_limit_A=35, voltage_compliance_V=15.0,
        ramp_step_A=0.1, ramp_delay_s=0.05,
    )
    magnet = connect_magnet(magnet_cfg)

    # ── Gaussmeter (Lake Shore 475) ────────────────────────────────────────────
    gauss_cfg = GaussmeterConfig(visa_resource="GPIB0::12::INSTR", unit="T",
                                 n_averages=10, read_delay_s=0.05)
    gaussmeter = connect_gaussmeter(gauss_cfg)

    # ── Temperature (Oxford Instruments MercuryiTC, optional) ────────────────
    temp_cfg = TemperatureControllerConfig(
        visa_resource="TCPIP0::192.168.1.5::7020::SOCKET", sensor_uids=("MB1.T1",))
    temp_ctrl = connect_temperature_controller(temp_cfg)

    # ── Sample geometry (optional) ────────────────────────────────────────────
    geometry_cfg = SampleGeometryConfig()

    # ── Measurement points — bidirectional field sweep ────────────────────────
    currents_A = bidirectional_current_sweep(i_min=-20.0, i_max=20.0, n_points=21)
    points = [
        MeasurementPoint(
            magnet_current_A=I,
            set_action=lambda daq, I=I: set_magnet_current(
                magnet, magnet_cfg, I, gaussmeter, gauss_cfg,
                acq_cfg.field_settle_tolerance_mT),
        )
        for I in currents_A
    ]

    try:
        df = run_measurement(daq, ac_cfg, leader_extref_cfg, follower_extref_cfg,
                              demod1_cfg, demod2_cfg, acq_cfg, points,
                              gaussmeter=gaussmeter, gauss_cfg=gauss_cfg,
                              temp_ctrl=temp_ctrl, temp_cfg=temp_cfg,
                              geometry_cfg=geometry_cfg, mds=mds)
        print("\n", df.to_string(index=False))
    finally:
        shutdown_ac_source(source)
        shutdown_magnet(magnet, magnet_cfg)
        shutdown_gaussmeter(gaussmeter)
        shutdown_temperature_controller(temp_ctrl)


if __name__ == "__main__":
    main()
