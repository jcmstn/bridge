"""
Zhinst MFLI Lock-in DAQ Server — shared connect/MDS/acquisition helpers
==========================================================================
The zhinst-core/zhinst-utils APIs (zi.ziDAQServer, the MultiDeviceSync
module, node-tree get/set) are used directly by every bridge/MFLI program —
this module holds the connect/MDS-sync/polling/averaging wrapper functions
that mfli_dual_harmonic.py, mfli_diff_resistance_vs_bias.py and
mfli_noise_spectrum.py each used to define an identical (or near-identical)
copy of, the same way keithley6221.py etc. now share the DC programs'
Keithley setup code.

Each program's own Signal Output topology (configure_output/OutputConfig)
and demodulator setup (configure_demodulator/DemodConfig) genuinely differ
— pure AC excitation vs. AC+DC bias vs. wide-bandwidth noise streaming — so
those stay local to each script, the same way dc_iv_curve.py keeps its own
sweep-range SourceConfig instead of using the shared fixed-sense-current
one. sync_follower_oscillator() and acquire_averaged() below only need an
object with the right attribute (out_cfg.frequency_Hz / cfg.device +
cfg.demod_index + cfg.sample_rate_Hz) rather than a shared OutputConfig/
DemodConfig type, so each script's own differently-shaped config classes
work with them unchanged.

Usage example:
    from mfli_daq import connect, connect_device, setup_mds, acquire_averaged

    daq = connect("localhost", 8004)
    connect_device(daq, "dev1234", interface="1GbE")
    connect_device(daq, "dev5678", interface="1GbE")
    setup_mds(daq, leader="dev1234", follower="dev5678")
    ...
    d = acquire_averaged(daq, demod_cfg, n_averages=50)
"""

import time
import logging
from typing import Optional

import numpy as np
import zhinst.core as zi

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


def _poll_demod(daq: zi.ziDAQServer, path: str,
                duration_s: float, timeout_ms: int) -> dict:
    """
    Subscribe, flush, poll for `duration_s`, unsubscribe.
    Returns a dict with arrays for x, y, r, theta_deg.
    """
    daq.subscribe(path)
    daq.sync()
    data = daq.poll(duration_s, timeout_ms, flat=True)
    daq.unsubscribe(path)

    if path not in data or len(data[path]) == 0:
        raise RuntimeError(f"No data returned for {path}. "
                           "Check demodulator is enabled and sample rate > 0.")

    # With flat=True, data[path] is a single dict of field -> numpy array
    # (all samples from the poll window concatenated), not a list of
    # per-sample dicts.
    samples = data[path]
    x = np.atleast_1d(samples["x"])
    y = np.atleast_1d(samples["y"])
    r = np.hypot(x, y)
    theta = np.degrees(np.arctan2(y, x))
    return {"x": x, "y": y, "r": r, "theta_deg": theta}


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
    return their mean +/- std. Poll duration is chosen to guarantee enough
    samples at the configured rate.

    `cfg` only needs `.device`, `.demod_index` and `.sample_rate_Hz`
    attributes — every program's own DemodConfig shape already has these,
    so this works unchanged for any of them without a shared DemodConfig
    type. If `cfg` also has an `.input_ch` attribute (i.e. it's reading a
    Signal Input, not a Current Input — checked via `.use_current_input`
    where that attribute exists, e.g. mfli_diff_resistance_vs_bias.py's
    current-sense channel), the returned dict includes an "overload" flag
    read right after the poll. Current Inputs (`currins/N`) have no
    `overload` node the same way `sigins/N` does, so that case reports
    `None` rather than reading the wrong (unused) Signal Input's flag.
    """
    path = f"/{cfg.device}/demods/{cfg.demod_index}/sample".lower()
    # Add a 50 % margin so we comfortably exceed n_averages
    duration_s  = max(0.1, (n_averages * 1.5) / cfg.sample_rate_Hz)
    timeout_ms  = int(duration_s * 1000) + 2000

    raw = _poll_demod(daq, path, duration_s, timeout_ms)

    # Trim to last n_averages samples (freshest data after settling)
    for k in raw:
        raw[k] = raw[k][-n_averages:]

    input_ch = getattr(cfg, "input_ch", None)
    uses_current_input = getattr(cfg, "use_current_input", False)
    overload = (_read_overload(daq, cfg.device, input_ch)
                if input_ch is not None and not uses_current_input else None)

    return {
        "x_mean":     float(np.mean(raw["x"])),
        "y_mean":     float(np.mean(raw["y"])),
        "r_mean":     float(np.mean(raw["r"])),
        "theta_mean": float(np.mean(raw["theta_deg"])),
        "r_std":      float(np.std(raw["r"])),
        "x_std":      float(np.std(raw["x"])),
        "y_std":      float(np.std(raw["y"])),
        "n_samples":  len(raw["r"]),
        "overload":   overload,
    }
