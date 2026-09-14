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

    Leader MFLI  Signal Input 1 (differential) ──▶ demod 1f
    Follower MFLI  Signal Input 1 (differential) ──▶ demod 2f

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import zhinst.core as zi

from instruments.keithley6221 import ACSourceConfig, connect_ac_source, shutdown_ac_source
from instruments.mfli_daq import connect, connect_device, setup_mds, check_mds_status, acquire_averaged
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
# External-reference (ExtRef) setup — local to this module, same mechanism
# and same "bench-verify against the logged node tree" caveat as
# sot/sot_pulsed_switching_6221.py::configure_external_reference /
# wait_for_reference_lock. Duplicated rather than imported: instruments/
# mfli_daq.py's own docstring says each program's Signal Input/output
# topology stays local even when nearly identical across programs.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtRefConfig:
    """One MFLI's oscillator, phase-locked to the 6221's Trigger Link
    marker via an Aux Input. One of these per device — see the module
    docstring for why BOTH leader and follower need their own.

    `pll_demod_index` is a demodulator DEDICATED to being the ExtRef PLL's
    phase detector — it must be different from the demod actually reading
    the Signal Input for 1f/2f (demod1_cfg/demod2_cfg's demod_index). Per
    Zurich's own node-tree reference (docs.zhinst.com/mfli_user_manual/
    nodedoc.html, confirmed against a real device's listNodesJSON dump),
    `extrefs/N/adcselect` and `extrefs/N/oscselect` are READ-ONLY — they
    only report whichever demodulator is wired in via `extrefs/N/
    demodselect`. There is no way to point the PLL at an Aux Input
    directly; you point a demodulator at that Aux Input (its own
    `adcselect`) and at the target oscillator (its own `oscselect`), then
    tell `extrefs/N/demodselect` to use that demodulator. Needs the target
    MFLI to have a free demod slot beyond the one used for the real signal
    (MF-MD / multi-demod option) — verify against the `demods/*`
    node count if this index doesn't exist on your unit."""
    device: str            = "dev1234"
    extref_index: int      = 0     # which ExtRef/PLL module (0-based)
    aux_input_ch: int      = 0     # which Aux Input carries the marker (0-based; 0 = Aux In 1)
    osc_index: int         = 0     # oscillator the PLL steers — demod1_cfg/demod2_cfg reference this
    pll_demod_index: int   = 1     # demod DEDICATED as the PLL's phase detector (≠ the signal demod)


# ZI demods/n/adcselect enum (docs.zhinst.com/mfli_user_manual/nodedoc.html):
# 8 = Aux In 1, 9 = Aux In 2 — NOT the same numbering as ExtRefConfig.aux_input_ch
# (0-based channel index), so the two must be added, not used interchangeably.
_ADCSELECT_AUX_IN_BASE = 8

# extrefs/N/automode enum (same doc): "all"/dynamic PID adaptation for the
# lock loop — left at whatever the device last had otherwise, which could be
# a bandwidth tuned for a different signal from a previous run.
_EXTREF_AUTOMODE_DYNAMIC = 4

# demods/n/rate is "number of samples sent to the host / LabOne Data
# Server per second" (node doc). MFLI's spec sheet lists 200 kSa/s as the
# "maximum transfer rate over 1 GbE (all demodulators)" — but that's an
# explicitly-labeled NETWORK/STORAGE limit, not the demodulator's native
# rate (docs.zhinst.com/mfli_user_manual/specifications.html); the Aux
# Input's own raw ADC is 16-bit/15 MSa/s with 5 MHz analog bandwidth (same
# page) — comfortably fast enough to resolve the 6221's ~1 µs marker pulse.
# Whether the on-device PLL's phase detection depends on this demod's own
# decimated rate at all isn't documented either way. Rather than guess a
# number, request something intentionally far above anything this device
# could really support and let the firmware clamp it — the node doc says a
# requested value "may be approximated to the nearest value supported by
# the instrument" — then read back and log what was actually applied.
_PLL_DETECTOR_RATE_REQUEST_HZ = 1e9


def configure_external_reference(daq: "zi.ziDAQServer", cfg: ExtRefConfig,
                                  frequency_Hz: float) -> None:
    """Arm cfg.device's ExtRef PLL to lock cfg.osc_index to the incoming
    marker, seeded with `frequency_Hz` as the search target (the 6221's
    commanded frequency — the PLL then tracks the marker's true frequency,
    which is the value actually worth trusting; see build_run_metadata()).

    See ExtRefConfig's docstring for why this goes through a dedicated
    `pll_demod_index` demodulator rather than writing extrefs/N/adcselect
    directly (that node is read-only on real firmware).
    """
    d = cfg.device
    daq.setDouble(f"/{d}/oscs/{cfg.osc_index}/freq", frequency_Hz)
    try:
        nodes = daq.listNodesJSON(f"/{d}/extrefs/{cfg.extref_index}/*")
        log.info("MFLI %s extrefs/%d node tree (verify against this on first "
                 "bench run):\n%s", d, cfg.extref_index, nodes)
    except Exception:
        log.exception("Could not list /%s/extrefs/%d/* — node names below are "
                      "unverified for this device/firmware.", d, cfg.extref_index)
    daq.setInt(f"/{d}/demods/{cfg.pll_demod_index}/adcselect",
              _ADCSELECT_AUX_IN_BASE + cfg.aux_input_ch)
    daq.setInt(f"/{d}/demods/{cfg.pll_demod_index}/oscselect", cfg.osc_index)
    # Phase detector must track the marker's FUNDAMENTAL, not whatever
    # harmonic this demod index was last left at (e.g. 2, from a previous
    # run's demod2_cfg reusing the same index) — a stale harmonic here has
    # the PLL searching the wrong frequency entirely and never locking.
    daq.setInt(f"/{d}/demods/{cfg.pll_demod_index}/harmonic", 1)
    daq.setDouble(f"/{d}/demods/{cfg.pll_demod_index}/rate", _PLL_DETECTOR_RATE_REQUEST_HZ)
    daq.setInt(f"/{d}/demods/{cfg.pll_demod_index}/enable", 1)
    daq.setInt(f"/{d}/extrefs/{cfg.extref_index}/demodselect", cfg.pll_demod_index)
    daq.setInt(f"/{d}/extrefs/{cfg.extref_index}/automode", _EXTREF_AUTOMODE_DYNAMIC)
    daq.setInt(f"/{d}/extrefs/{cfg.extref_index}/enable", 1)
    daq.sync()
    applied_rate = daq.getDouble(f"/{d}/demods/{cfg.pll_demod_index}/rate")
    log.info("MFLI %s: oscillator %d locking to Aux In %d via extrefs/%d "
             "(phase detector demod%d, target %.4f Hz, detector rate "
             "requested %.4g Sa/s -> device applied %.4g Sa/s)", d, cfg.osc_index,
             cfg.aux_input_ch + 1, cfg.extref_index, cfg.pll_demod_index, frequency_Hz,
             _PLL_DETECTOR_RATE_REQUEST_HZ, applied_rate)


def wait_for_reference_lock(daq: "zi.ziDAQServer", cfg: ExtRefConfig,
                             timeout_s: float,
                             stop_event: Optional[threading.Event] = None) -> bool:
    """Never raises; an unreadable/unlocked PLL degrades to False, logged
    once by the caller. See configure_external_reference()'s docstring."""
    path = f"/{cfg.device}/extrefs/{cfg.extref_index}/locked"
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if stop_event is not None and stop_event.is_set():
            return False
        try:
            if daq.getInt(path):
                return True
        except Exception:
            log.warning("Could not read ExtRef lock node %s — check the node "
                       "name against configure_external_reference()'s "
                       "listNodesJSON log.", path)
            return False
        time.sleep(0.05)
    return False


def check_reference_locked(daq: "zi.ziDAQServer", cfg: ExtRefConfig) -> Optional[bool]:
    """Cheap, non-blocking re-check that cfg's ExtRef PLL is still locked,
    mid-measurement — the ExtRef analogue of instruments/mfli_daq.py's
    check_mds_status(): never raises, no retry/wait (a transient unlocked
    read is exactly the signal a caller wants to flag and log per-point,
    not paper over)."""
    try:
        return bool(daq.getInt(f"/{cfg.device}/extrefs/{cfg.extref_index}/locked"))
    except Exception:
        log.warning("Could not read ExtRef lock node for %s — reference lock "
                    "status unknown this point.", cfg.device)
        return None


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
) -> dict:
    """Same shape/purpose as mfli_dual_harmonic.build_run_metadata(), with
    the excitation terms sourced from the 6221 instead: amplitude_A is
    already the peak excitation current (an ideal current source — no
    series-resistor V/R assumption), and excitation_frequency_Hz is read
    live from the leader's ExtRef-locked oscillator (the PLL's true tracked
    frequency, not the 6221's commanded value — matches the convention
    already used by sot_pulsed_switching_6221.py for the same reason).
    """
    geometry_cfg = geometry_cfg or SampleGeometryConfig()
    I_peak_A = ac_cfg.amplitude_A
    excitation_frequency_Hz = daq.getDouble(
        f"/{leader_extref_cfg.device}/oscs/{leader_extref_cfg.osc_index}/freq")
    return {
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
        "field_angle_from_oop_deg": geometry_cfg.field_angle_from_oop_deg,
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
) -> pd.DataFrame:
    """Same loop shape as mfli_dual_harmonic.run_measurement(): iterate
    `points`, acquire 1f (leader) + 2f (follower) at each, log to CSV,
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
    """
    _check_ac_safety(ac_cfg)
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

        # ── 3. Acquire 1f ──────────────────────────────────────────────────
        d1 = acquire_averaged(daq, demod1_cfg, acq_cfg.n_averages)
        log.info("   1f  R=%.4e V  θ=%.2f°  σ_R=%.2e V  (n=%d)",
                 d1["r_mean"], d1["theta_mean"], d1["r_std"], d1["n_samples"])
        if d1["overload"]:
            log.warning("   1f input is OVERLOADED — this reading is not trustworthy.")

        # ── 4. Acquire 2f ──────────────────────────────────────────────────
        d2 = acquire_averaged(daq, demod2_cfg, acq_cfg.n_averages)
        log.info("   2f  R=%.4e V  θ=%.2f°  σ_R=%.2e V  (n=%d)",
                 d2["r_mean"], d2["theta_mean"], d2["r_std"], d2["n_samples"])
        if d2["overload"]:
            log.warning("   2f input is OVERLOADED — this reading is not trustworthy.")

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
                                      demod2_cfg, geometry_cfg, demod2_phase_null_1f_deg)

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
            "1f_R_std_V":  d1["r_std"],
            "1f_overload": d1["overload"],
            # ── 2f ─────────────────────────────────────────────────────────
            "2f_X_V":      d2["x_mean"],
            "2f_Y_V":      d2["y_mean"],
            "2f_R_V":      d2["r_mean"],
            "2f_theta_deg":d2["theta_mean"],
            "2f_R_std_V":  d2["r_std"],
            "2f_overload": d2["overload"],
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
    shared_filter = FilterConfig(time_constant_s=0.3, order=4, sinc_filter=True)

    # ── 1f demodulator (leader) ───────────────────────────────────────────────
    demod1_cfg = DemodConfig(
        device=LEADER, demod_index=0, harmonic=1, osc_index=leader_extref_cfg.osc_index,
        input_range_V=1.0, sample_rate_Hz=857.0, filter=shared_filter,
    )
    configure_demodulator(daq, demod1_cfg)

    # ── 2f demodulator (follower) ─────────────────────────────────────────────
    demod2_cfg = DemodConfig(
        device=FOLLOWER, demod_index=0, harmonic=2, osc_index=follower_extref_cfg.osc_index,
        input_range_V=1.0, sample_rate_Hz=857.0, filter=shared_filter,
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
