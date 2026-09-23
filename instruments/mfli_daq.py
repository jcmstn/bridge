"""
Zhinst MFLI Lock-in DAQ Server — shared connect/MDS/acquisition helpers
==========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-06

This module holds the connect/MDS-sync/polling/averaging wrapper functions
shared by the MFLI measurement programs, built on the zhinst-core/zhinst-utils
APIs (zi.ziDAQServer, the MultiDeviceSync module, node-tree get/set).

Each program's own Signal Output topology (configure_output/OutputConfig)
and demodulator setup (configure_demodulator/DemodConfig) genuinely differ
— pure AC excitation vs. AC+DC bias vs. wide-bandwidth noise streaming — so
those stay local to each script. sync_follower_oscillator() and
acquire_averaged() below only need an object with the right attribute
(out_cfg.frequency_Hz / cfg.device + cfg.demod_index + cfg.sample_rate_Hz)
rather than a shared OutputConfig/DemodConfig type, so each script's own
differently-shaped config classes work with them unchanged.

The external-reference PLL (ExtRefConfig, configure_external_reference,
wait_for_reference_lock, check_reference_locked) IS the same everywhere —
every program that locks an MFLI to the Keithley 6221's Trigger Link phase
marker (dual-harmonic 6221 source, SOT 2nd-harmonic and 6221-only reads)
used byte-identical copies — so it lives here once.

Usage example:
    from instruments.mfli_daq import connect, connect_device, setup_mds, acquire_averaged

    daq = connect("localhost", 8004)
    connect_device(daq, "dev1234", interface="1GbE")
    connect_device(daq, "dev5678", interface="1GbE")
    setup_mds(daq, leader="dev1234", follower="dev5678")
    ...
    d = acquire_averaged(daq, demod_cfg, n_averages=50)

    # Two demods to read at the same point (e.g. 1f/2f)? Use
    # acquire_averaged_pair() -- one poll() window covering both instead
    # of two sequential ones:
    d1, d2 = acquire_averaged_pair(daq, demod1_cfg, demod2_cfg, n_averages=50)
"""

import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import zhinst.core as zi

from instruments.run_time import ACQ_OVERHEAD_S

log = logging.getLogger(__name__)


def connect(host: str = "localhost", port: int = 8004, api_level: int = 6) -> zi.ziDAQServer:
    """Open a session to the LabOne data server."""
    daq = zi.ziDAQServer(host, port, api_level)
    log.info("Connected to ZI data server at %s:%d", host, port)
    return daq


def connect_device(daq: zi.ziDAQServer, device: str, interface: str = "1GbE") -> None:
    """Connect a device to the data server (no-op if already connected)."""
    try:
        daq.connectDevice(device, interface)
        log.info("Connected device %s via %s", device, interface)
    except RuntimeError:
        log.info("Device %s already connected", device)


def setup_mds(daq: zi.ziDAQServer, leader: str, follower: str, timeout_s: float = 60.0):
    """
    Configure Multi-Device Synchronization between two MFLIs.

    Per the LabOne MultiDeviceSync module reference, there is no separate
    "leader" node — the role is inferred from *order* in the comma-separated
    `devices` list (first entry = leader) and must match the physical
    cabling. The MFLI requires BOTH of the following (ZSync is not an MFLI
    feature — that's UHFQA/SHF-family hardware):
      - Ref clock: BNC cable from the leader's Ref Out to the follower's
        Ref In.
      - Trigger: the leader's Trigger Out 1 fanned out (e.g. via a 1-to-N
        power divider, equal cable lengths) to Trigger In 1 on *both* the
        follower and the leader itself.

    NOTE: this synchronizes clocks and the measurement start instant — it
    does NOT copy oscillator frequency values between devices. See
    sync_follower_oscillator() below.

    Returns the MultiDeviceSync module handle so a caller can poll
    check_mds_status() on it later, mid-measurement, without re-running the
    sync handshake — MDS can silently drop out of sync (a loose Ref/Trigger
    cable) and there is otherwise no way to notice that partway through a
    long sweep.
    """
    mds = daq.multiDeviceSyncModule()

    mds.set("start", 0)
    mds.set("group", 0)
    mds.execute()   # starts the module's worker thread — without this, "start"
                     # is never actually processed and status sits at 0 forever
    mds.set("devices", f"{leader},{follower}")
    mds.set("start", 1)

    # Poll until synchronization is confirmed (status == 2 -> synced,
    # -1 -> failed, 0/1 -> idle/in progress)
    log.info("Waiting for MDS synchronization ...")
    t0 = time.monotonic()
    while True:
        status = mds.getInt("status")
        if status == 2:
            break
        if status == -1:
            raise RuntimeError(
                f"MDS synchronization failed (status=-1): {mds.getString('message')}. "
                "Check Ref clock cable, trigger fan-out cabling, and device order."
            )
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError(
                f"MDS sync timed out (status={status}): {mds.getString('message')}. "
                "Check Ref clock cable and trigger fan-out cabling."
            )
        time.sleep(0.2)
    log.info("MDS synchronized: leader=%s, follower=%s", leader, follower)
    return mds


def check_mds_status(mds) -> bool:
    """
    Re-check, mid-measurement, that MDS is still synced (status == 2) on the
    module handle setup_mds() returned. Cheap — just a getInt on an
    already-running module, safe to call every point/chunk of a long sweep
    or recording. Returns False (never raises) on any other status,
    including a module that failed outright, so callers can log/flag it and
    decide for themselves whether to keep going.
    """
    try:
        return mds.getInt("status") == 2
    except Exception:
        log.exception("Could not read MDS status")
        return False


def sync_follower_oscillator(daq: zi.ziDAQServer, out_cfg, follower: str,
                              follower_osc_index: int = 0) -> None:
    """
    Explicitly copy the leader's excitation frequency onto the follower's
    own local oscillator. Required because MDS (see setup_mds docstring)
    does not do this for you — each device's oscillator is independently
    set. Skipping this step is the single most common reason a two-MFLI
    lock-in measurement silently returns garbage (a slowly beating phasor
    instead of a stable one).

    `out_cfg` only needs a `.frequency_Hz` attribute — every program's own
    OutputConfig shape (pure AC, AC+bias, ...) already has one, so this
    works unchanged for any of them without a shared OutputConfig type.
    """
    daq.setDouble(f"/{follower}/oscs/{follower_osc_index}/freq", out_cfg.frequency_Hz)
    daq.sync()
    log.info("Follower %s oscillator %d frequency set to %.4f Hz (matches leader)",
             follower, follower_osc_index, out_cfg.frequency_Hz)


def _poll_demod_paths(daq: zi.ziDAQServer, paths: list,
                      duration_s: float, timeout_ms: int) -> dict:
    """
    Subscribe to every path in `paths`, issue ONE poll() spanning all of
    them over the same wall-clock window, then unsubscribe. LabOne's
    poll() is a session-level call -- it returns buffered data for every
    currently-subscribed path in one shot, regardless of which physical
    device each path lives on (a session already connects multiple
    devices; see docs.zhinst.com's "Subscribe and Poll" reference) -- so
    this is the supported way to read several demodulators without paying
    for N sequential poll windows. Returns {path: {x, y, r, theta_deg}}.
    """
    for path in paths:
        daq.subscribe(path)
    daq.sync()
    data = daq.poll(duration_s, timeout_ms, flat=True)
    for path in paths:
        daq.unsubscribe(path)

    result: dict = {}
    for path in paths:
        if path not in data or len(data[path]) == 0:
            raise RuntimeError(f"No data returned for {path}. "
                               "Check demodulator is enabled and sample rate > 0.")
        # With flat=True, data[path] is a single dict of field -> numpy
        # array (all samples from the poll window concatenated), not a
        # list of per-sample dicts.
        samples = data[path]
        x = np.atleast_1d(samples["x"])
        y = np.atleast_1d(samples["y"])
        result[path] = {"x": x, "y": y, "r": np.hypot(x, y),
                         "theta_deg": np.degrees(np.arctan2(y, x))}
    return result


def _poll_demod(daq: zi.ziDAQServer, path: str,
                duration_s: float, timeout_ms: int) -> dict:
    """Single-path case of _poll_demod_paths() -- see there for the shared
    subscribe/poll/unsubscribe mechanics."""
    return _poll_demod_paths(daq, [path], duration_s, timeout_ms)[path]


_overload_node_warned: set = set()


def _read_overload(daq: zi.ziDAQServer, device: str, input_ch: int) -> Optional[bool]:
    """
    Read the Signal Input overload flag — this is the single cheapest check
    against a silently clipped/garbage lock-in reading (front-end overload
    produces bad output regardless of how good the demodulation settings
    are). Returns None (rather than raising) if the node can't be read, so
    a firmware/node-name mismatch degrades a run instead of crashing it;
    logs that failure once per device/channel rather than once per point.
    """
    path = f"/{device}/sigins/{input_ch}/overload"
    try:
        return bool(daq.getInt(path))
    except Exception:
        key = (device, input_ch)
        if key not in _overload_node_warned:
            _overload_node_warned.add(key)
            log.warning("Could not read overload flag at %s — overload "
                        "will be reported as unknown for this channel.", path)
        return None


def acquire_averaged(daq: zi.ziDAQServer, cfg, n_averages: int) -> dict:
    """
    Collect at least `n_averages` samples from `cfg`'s demodulator and
    return their mean, TWO different uncertainty flavors, and the sample
    count behind them. Poll duration is chosen to guarantee enough samples
    at the configured rate AND to span at least 3x the demod time constant
    (below that the samples are correlated, so the mean barely improves on
    one reading and the reported spread is optimistic).

    Two uncertainty flavors, for two different jobs — don't swap them:
      - `x_std`/`y_std`/`r_std`: population stdev (ddof=0) of the raw
        per-sample X/Y/magnitude — the point-to-point scatter, useful as a
        signal-to-noise-style diagnostic (see
        mfli_phase_calibration.identify_2f_channel()). `r_std` in
        particular is the spread of the per-sample RECTIFIED magnitude
        `hypot(x_i, y_i)` — a different, positively-biased quantity from
        `r_mean` (see r_mean's own docstring note below) — so it must never
        be reported as "the" uncertainty on `r_mean`.
      - `x_sem`/`y_sem`/`r_sem`: standard error of the MEAN (sample stdev,
        ddof=1, / sqrt(n); `nan` if n<2, matching the SEM convention used
        everywhere else in this codebase, e.g. keithley2182.
        acquire_averaged_voltage). `r_sem` is `x_sem`/`y_sem` propagated
        onto `r_mean = hypot(x_mean, y_mean)` via first-order error
        propagation (dR/dX = X/R, dR/dY = Y/R) — this is the uncertainty
        that actually belongs on `r_mean`, and what a caller should save
        and plot as R's error bar.

    `cfg` only needs `.device`, `.demod_index` and `.sample_rate_Hz`
    attributes — every program's own DemodConfig shape already has these,
    so this works unchanged for any of them without a shared DemodConfig
    type. A `.filter.time_constant_s` attribute, if present, is used for the
    3x-TC poll-window floor above; without it that floor is simply skipped. If `cfg` also has an `.input_ch` attribute (i.e. it's reading a
    Signal Input, not a Current Input — checked via `.use_current_input`
    where that attribute exists, e.g. mfli_diff_resistance_vs_bias.py's
    current-sense channel), the returned dict includes an "overload" flag
    read right after the poll. Current Inputs (`currins/N`) have no
    `overload` node the same way `sigins/N` does, so that case reports
    `None` rather than reading the wrong (unused) Signal Input's flag.
    """
    path = f"/{cfg.device}/demods/{cfg.demod_index}/sample".lower()
    duration_s = _poll_duration_s(cfg, n_averages)
    timeout_ms = int(duration_s * 1000) + 2000

    raw = _poll_demod(daq, path, duration_s, timeout_ms)
    return _finish_average(daq, cfg, raw, n_averages)


def _poll_duration_s(cfg, n_averages: int) -> float:
    """Poll window long enough that the samples are actually independent.
    The demod low-pass has a correlation time on the order of its own time
    constant, so a window shorter than a few TC returns ~1 independent
    sample no matter how many rows come back -- the mean barely improves
    on a single reading and x_std/y_std/r_std understate the true
    uncertainty by ~sqrt(window / TC). Floor the window at 3x TC.
    `cfg.filter` is optional in this module's duck-typed contract (see the
    module docstring), so fall back to the plain sample-count window when
    it isn't present. 50% margin on the sample-count term so we comfortably
    exceed n_averages."""
    tc = getattr(getattr(cfg, "filter", None), "time_constant_s", 0.0)
    return poll_window_s(tc, n_averages, cfg.sample_rate_Hz)


def poll_window_s(time_constant_s: float, n_averages: int, sample_rate_Hz: float) -> float:
    """Length of the acquisition window ``acquire_averaged`` blocks for --
    the single source of truth, also used by every TUI's run-time estimate."""
    return max(0.1, 3.0 * time_constant_s, (n_averages * 1.5) / sample_rate_Hz)


def acquire_s(time_constant_s: float, n_averages: int, sample_rate_Hz: float) -> float:
    """Modelled wall time of one ``acquire_averaged`` / ``_pair`` call: the
    window plus run_time.ACQ_OVERHEAD_S (subscribe / sync / unsubscribe / overload read)."""
    return poll_window_s(time_constant_s, n_averages, sample_rate_Hz) + ACQ_OVERHEAD_S


def _finish_average(daq: zi.ziDAQServer, cfg, raw: dict, n_averages: int) -> dict:
    """Trim `raw` (as returned by _poll_demod/_poll_demod_paths) to the
    freshest n_averages samples and reduce to the mean/SEM/std/overload
    dict acquire_averaged() and acquire_averaged_pair() both return -- see
    acquire_averaged()'s docstring for the two uncertainty flavors and why
    R/theta come from the mean X/Y, not the mean of per-sample R/theta."""
    # Trim to last n_averages samples (freshest data after settling)
    for k in raw:
        raw[k] = raw[k][-n_averages:]

    input_ch = getattr(cfg, "input_ch", None)
    uses_current_input = getattr(cfg, "use_current_input", False)
    overload = (_read_overload(daq, cfg.device, input_ch)
                if input_ch is not None and not uses_current_input else None)

    x_mean = float(np.mean(raw["x"]))
    y_mean = float(np.mean(raw["y"]))

    # R and theta are the polar form of the VECTOR-averaged phasor
    # (mean X, mean Y) -- NOT the mean of per-sample hypot()/atan2().
    # Averaging per-sample magnitudes rectifies noise: E[|z+n|] > |z|, a
    # positive bias that dominates once the signal is near the noise floor
    # -- exactly the regime of a small 2f harmonic-Hall voltage. Averaging
    # per-sample angles is worse still (undefined mean across the +/-180deg
    # branch cut). X and Y average linearly and unbiasedly, so take R/theta
    # from their means.
    r_mean = float(np.hypot(x_mean, y_mean))
    theta_mean = float(np.degrees(np.arctan2(y_mean, x_mean)))

    n = len(raw["x"])
    if n >= 2:
        x_sem = float(np.std(raw["x"], ddof=1) / np.sqrt(n))
        y_sem = float(np.std(raw["y"], ddof=1) / np.sqrt(n))
    else:
        x_sem = y_sem = float("nan")
    # Propagate x_sem/y_sem onto r_mean = hypot(x_mean, y_mean), NOT the std
    # of the per-sample magnitudes (see docstring) -- first-order error
    # propagation: dR/dX = X/R, dR/dY = Y/R.
    if n >= 2 and r_mean > 0:
        r_sem = float(np.hypot(x_mean * x_sem, y_mean * y_sem) / r_mean)
    else:
        r_sem = float("nan")

    return {
        "x_mean":     x_mean,
        "y_mean":     y_mean,
        "r_mean":     r_mean,
        "theta_mean": theta_mean,
        "x_sem":      x_sem,
        "y_sem":      y_sem,
        "r_sem":      r_sem,
        "r_std":      float(np.std(raw["r"])),
        "x_std":      float(np.std(raw["x"])),
        "y_std":      float(np.std(raw["y"])),
        "n_samples":  n,
        "overload":   overload,
    }


def acquire_averaged_pair(daq: zi.ziDAQServer, cfg_a, cfg_b, n_averages: int) -> tuple:
    """
    Same result as calling acquire_averaged(cfg_a) then acquire_averaged(cfg_b),
    but in ONE poll() window instead of two back-to-back ones. Use this for
    a pair of demods that both need reading at the same measurement point
    (e.g. a 1f/2f harmonic pair on two MDS-synced MFLIs, or a current/
    voltage pair for differential resistance).

    LabOne's poll() operates at the session level: it returns buffered
    data for every currently-subscribed path in one call, regardless of
    which physical device each path lives on (a session already connects
    multiple devices -- see docs.zhinst.com's "Subscribe and Poll"
    reference). Subscribing both demods' paths and issuing a single
    poll(max(duration_a, duration_b)) is therefore the supported way to
    read them together, over the same wall-clock window, instead of
    duration_a + duration_b of sequential windows.

    Each cfg keeps its own poll-window floor -- the shared duration is the
    MAX of the two (see _poll_duration_s()), so neither channel's own
    3x-TC/n_averages floor is ever under-spanned; the channel with the
    shorter requirement just gets a few extra, harmless samples.

    A shared window means both channels cover the same WALL-CLOCK
    interval, NOT that sample i of cfg_a is simultaneous with sample i of
    cfg_b. MDS (setup_mds()) is orthogonal to this API call -- it's what
    makes the shared interval meaningful (both devices already share a
    sample clock and start instant), not what makes poll()ing them
    together possible.
    """
    path_a = f"/{cfg_a.device}/demods/{cfg_a.demod_index}/sample".lower()
    path_b = f"/{cfg_b.device}/demods/{cfg_b.demod_index}/sample".lower()

    duration_s = max(_poll_duration_s(cfg_a, n_averages),
                      _poll_duration_s(cfg_b, n_averages))
    timeout_ms = int(duration_s * 1000) + 2000

    raw = _poll_demod_paths(daq, [path_a, path_b], duration_s, timeout_ms)

    return (_finish_average(daq, cfg_a, raw[path_a], n_averages),
            _finish_average(daq, cfg_b, raw[path_b], n_averages))


# ─────────────────────────────────────────────────────────────────────────────
# External reference (ExtRef PLL) — lock an oscillator to the 6221's phase
# marker on an Aux Input. Bench-verify the node names against the
# listNodesJSON dump configure_external_reference() logs on first use.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtRefConfig:
    """One MFLI's oscillator, phase-locked to the 6221's Trigger Link
    marker via an Aux Input. One of these per device (with two MDS-synced
    MFLIs, BOTH need their own — see mfli/mfli_dual_harmonic_6221.py).

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
    automode: int          = 4     # extrefs/N/automode — PID bandwidth adaptation for the lock loop:

# ZI demods/n/adcselect enum (docs.zhinst.com/mfli_user_manual/nodedoc.html):
# 8 = Aux In 1, 9 = Aux In 2 — NOT the same numbering as ExtRefConfig.aux_input_ch
# (0-based channel index), so the two must be added, not used interchangeably.
_ADCSELECT_AUX_IN_BASE = 8

# demods/n/rate is "number of samples sent to the host / LabOne Data
# Server per second" (node doc). MFLI's spec sheet lists 200 kSa/s as the
# "maximum transfer rate over 1 GbE (all demodulators)" — but that's an
# explicitly-labeled NETWORK/STORAGE limit, not the demodulator's native
# rate (docs.zhinst.com/mfli_user_manual/specifications.html). Requested
# value here (15 MSa/s) matches the Aux Input's own raw ADC spec (16-bit,
# 15 MSa/s, 5 MHz analog bandwidth, same page) — the fastest this input
# could physically need resolving at, so asking for more wouldn't mean
# anything. Whether the on-device PLL's phase detection depends on this
# demod's own decimated rate at all still isn't documented either way, and
# demods/n/rate's own true max isn't documented independent of the network
# figure above — the node doc says a requested value "may be approximated
# to the nearest value supported by the instrument", so this may still get
# clamped down; read back and log what was actually applied rather than
# assume.
_PLL_DETECTOR_RATE_REQUEST_HZ = 15e6

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
    daq.setInt(f"/{d}/extrefs/{cfg.extref_index}/automode", cfg.automode)
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
