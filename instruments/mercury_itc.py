"""
Oxford Instruments MercuryiTC Temperature Controller Driver + Controller
=========================================================================
Built on pymeasure's Instrument architecture, the same way
lakeshore475.LakeShore475 wraps the Lake Shore 475 Gaussmeter. pymeasure
ships a driver for the older ITC 503 (pymeasure.instruments.oxfordinstruments)
but the MercuryiTC uses a completely different, newer ASCII command
language (colon-separated, hierarchical "device" addressing rather than
ITC 503's single-letter commands) — this fills that gap.

Like lakeshore475.py, this was originally hand-built from the published
Mercury Support command reference without hardware access. The
temperature-read path (MercuryITC.temperature()) has since been verified
against a real iTC and corrected to match its actual reply shape — see
that method's docstring. Everything else in this driver (identification,
close/context-manager handling) is still just the published reference,
unverified.

Interface: Ethernet (raw TCP socket, port 7020) or USB/RS-232 (same
command set, different pyvisa resource string).
Firmware command reference: Mercury Support Handbook, "REMOTE COMMUNICATION"
chapter — commands of the form READ:DEV:<uid>:TEMP:SIG:TEMP.

This instrument is shared by several lab programs (see kepco_magnet.py's
docstring for the same philosophy) — add this folder to sys.path rather
than duplicating the driver.

Unlike the Kepco magnet or Lake Shore 475 (which are load-bearing parts
of a specific measurement, so a connection failure is a real error), the
MercuryiTC is a "nice to have" — many rigs don't have one, and among
those that do, some only have one temperature probe wired up rather than
two. connect_temperature_controller() and read_temperature() below are
written so neither of those is ever an error: a measurement that doesn't
otherwise need the iTC should never be interrupted by it.

Usage example (low-level driver):
    from mercury_itc import MercuryITC

    mitc = MercuryITC("TCPIP0::192.168.1.5::7020::SOCKET")
    print(mitc.temperature("MB1.T1"))   # single reading, Kelvin
    mitc.close()

Usage example (shared controller, used by the DC/MFLI measurement scripts):
    from mercury_itc import (
        TemperatureControllerConfig, connect_temperature_controller,
        read_temperature, shutdown_temperature_controller,
    )

    temp_cfg = TemperatureControllerConfig(
        visa_resource="TCPIP0::192.168.1.5::7020::SOCKET",
        sensor_uids=("MB1.T1", "DB5.T1"),   # 1 or 2 probes — set to your rig
    )
    mitc = connect_temperature_controller(temp_cfg)   # None if unreachable

    ...
    t1_K, t2_K = read_temperature(mitc, temp_cfg)      # (None, None) if no iTC
    ...

    shutdown_temperature_controller(mitc)

Usage example (setpoint control, for a temperature-dependence sweep):
    from mercury_itc import set_temperature, wait_for_temperature_stable

    set_temperature(mitc, temp_cfg, 4.2)                        # never blocks
    wait_for_temperature_stable(mitc, temp_cfg, 4.2,             # optional —
                                 tolerance_K=0.02, hold_time_s=60)  # caller's choice
    ... take data at 4.2 K ...
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from pymeasure.instruments import Instrument

log = logging.getLogger(__name__)


class MercuryITC(Instrument):
    """
    Represents the Oxford Instruments MercuryiTC and provides a high-level
    interface for reading temperature signals from its temperature boards.

    Each temperature sensor probe is wired to a separate board inside the
    Mercury chassis, addressed by a UID such as "MB1.T1" (read off the
    iTC's front panel or via `catalog`). A given system may have one such
    board or several — `temperature()` reads whichever UID you ask for.
    A sensor board with an associated heater board (see `catalog`) also
    has a PID loop that can be driven to a setpoint — see
    `set_temperature_setpoint()` and `heater_auto()`.

    .. code-block:: python

        mitc = MercuryITC("TCPIP0::192.168.1.5::7020::SOCKET")
        print(mitc.temperature("MB1.T1"))   # Kelvin

        mitc.set_temperature_setpoint("MB1.T1", 4.2)
        mitc.heater_auto("MB1.T1", True)    # let the iTC's own PID drive it there
    """

    def __init__(self, adapter, name="Oxford Instruments MercuryiTC", **kwargs):
        super().__init__(
            adapter,
            name,
            includeSCPI=False,
            read_termination="\n",
            write_termination="\n",
            **kwargs,
        )

    identification = Instrument.measurement(
        "*IDN?",
        """ Get the instrument identification string (*IDN?). """,
    )

    @property
    def catalog(self) -> str:
        """
        Get the raw system catalog (`READ:SYS:CAT`) — lists every board
        installed in the chassis, temperature or otherwise, each with its
        UID. Useful for finding the UID of a probe from the front panel
        when you don't already know it; not parsed into a structured list
        since the exact catalog layout varies by chassis configuration.
        """
        return self.ask("READ:SYS:CAT").strip()

    def temperature(self, uid: str) -> float:
        """
        Return the temperature reported by the sensor board `uid` (e.g.
        "MB1.T1"), in Kelvin.

        Raises ValueError if the reply doesn't have the expected
        `STAT:DEV:<uid>:TEMP:SIG:TEMP:<value>K` shape — e.g. because
        `uid` doesn't exist on this chassis, or the board reports an
        error (no probe plugged in, open circuit, etc.) rather than a
        temperature.
        """
        command = f"READ:DEV:{uid}:TEMP:SIG:TEMP"
        reply = self.ask(command).strip()

        expected_prefix = f"STAT:DEV:{uid}:TEMP:SIG:TEMP:"
        if not reply.startswith(expected_prefix) or not reply.endswith("K"):
            raise ValueError(
                f"Unexpected reply to {command!r}: {reply!r} "
                f"(expected {expected_prefix}<value>K)"
            )

        value_str = reply[len(expected_prefix):-1]
        return float(value_str)

    def temperature_setpoint(self, uid: str) -> float:
        """
        Return the temperature setpoint (Kelvin) currently programmed
        into the PID loop associated with sensor board `uid`.

        Raises ValueError if the reply doesn't have the expected
        `STAT:DEV:<uid>:TEMP:LOOP:TSET:<value>K` shape. Assumed to follow
        the same reply shape as `temperature()` (no `VALUE:` segment) —
        UNVERIFIED against real hardware; check it the same way that one
        was checked (see module docstring) before trusting it.
        """
        command = f"READ:DEV:{uid}:TEMP:LOOP:TSET"
        reply = self.ask(command).strip()

        expected_prefix = f"STAT:DEV:{uid}:TEMP:LOOP:TSET:"
        if not reply.startswith(expected_prefix) or not reply.endswith("K"):
            raise ValueError(
                f"Unexpected reply to {command!r}: {reply!r} "
                f"(expected {expected_prefix}<value>K)"
            )
        return float(reply[len(expected_prefix):-1])

    def set_temperature_setpoint(self, uid: str, setpoint_K: float) -> None:
        """
        Program the PID loop associated with sensor board `uid` to
        regulate toward `setpoint_K`.

        This alone does not make the iTC actually drive its heater —
        call `heater_auto(uid, True)` as well (or first; order doesn't
        matter) so the loop is in automatic/PID mode rather than manual.
        Once both are set, the iTC's own PID loop chases the setpoint in
        the background for as long as the loop stays in auto mode —
        independent of whether this Python session stays connected.
        Does not itself wait for the temperature to arrive; see
        `wait_for_temperature_stable()` in the shared controller below
        if a caller wants to block on that.

        Raises RuntimeError if the instrument rejects the setpoint (e.g.
        out of range for this board) — UNVERIFIED reply shape, see
        `_check_set_reply()`.
        """
        command = f"SET:DEV:{uid}:TEMP:LOOP:TSET:{setpoint_K:.4f}"
        reply = self.ask(command).strip()
        self._check_set_reply(command, reply)

    def heater_auto(self, uid: str, enabled: bool) -> None:
        """
        Switch the PID loop associated with sensor board `uid` between
        automatic heater control (`enabled=True` — the iTC drives its
        paired heater board to chase `temperature_setpoint`) and manual
        control (`enabled=False` — heater output stays wherever it was
        last set, and the setpoint is no longer being chased).

        Call with `enabled=False` when a program is done controlling
        temperature, so a stale setpoint from this session can't keep
        heating unattended afterward.

        Raises RuntimeError if the instrument rejects the command —
        UNVERIFIED reply shape, see `_check_set_reply()`.
        """
        state = "ON" if enabled else "OFF"
        command = f"SET:DEV:{uid}:TEMP:LOOP:HTR:AUTO:{state}"
        reply = self.ask(command).strip()
        self._check_set_reply(command, reply)

    @staticmethod
    def _check_set_reply(command: str, reply: str) -> None:
        """
        Validate a Mercury `SET:...` acknowledgement, expected per the
        command reference to end in `:VALID` (accepted) or `:INVALID`
        (rejected). UNVERIFIED against real hardware — unlike
        `temperature()`'s reply shape, this one hasn't been checked
        against a live iTC yet, and the reference doc was already wrong
        once for a sibling command (see module docstring). Check a
        `set_temperature_setpoint()`/`heater_auto()` reply against your
        instrument's log before relying on the raise/no-raise behaviour.
        """
        if reply.endswith(":INVALID") or not reply.endswith(":VALID"):
            raise RuntimeError(f"MercuryiTC rejected {command!r}: {reply!r}")

    def close(self) -> None:
        """ Close the underlying VISA session. """
        self.adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Shared controller  ── used directly by the DC/MFLI measurement scripts ──────
# ─────────────────────────────────────────────────────────────────────────────
# read_temperature()'s "1 or 2 sensors, never let a missing/failed instrument
# stop the measurement" logic is shared here so every DC/MFLI script gets the
# same graceful-degradation behavior for free instead of re-implementing it —
# the same convention kepco_magnet.py and lakeshore475.py now follow for
# their own connect/set-or-read/shutdown helpers (used directly by every
# bridge/DC program; bridge/MFLI keeps its own copies for now).
#
# set_temperature()/wait_for_temperature_stable() below are the opposite of
# read_temperature() in one respect: once a program actually asks the iTC to
# drive to a setpoint, that setpoint is load-bearing for the measurement, so
# these DO raise on failure — same convention as kepco_magnet.set_magnet_current().

@dataclass
class TemperatureControllerConfig:
    """
    Oxford Instruments MercuryiTC, used to log sample/probe temperature
    alongside a measurement — shared across DC and MFLI programs the same
    way the Kepco supply and Lake Shore 475 gaussmeter are.

    sensor_uids lists 1 or 2 board UIDs, e.g. ("MB1.T1",) for a rig with a
    single probe or ("MB1.T1", "DB5.T1") for one with two. Find the UIDs
    for your own rig via the iTC front panel or MercuryITC(...).catalog.
    """
    visa_resource: str = "TCPIP0::192.168.1.5::7020::SOCKET"
    sensor_uids: Tuple[str, ...] = ("MB1.T1",)
    timeout_ms: int = 3000


def connect_temperature_controller(
    cfg: TemperatureControllerConfig,
) -> Optional[MercuryITC]:
    """
    Try to open a session to the MercuryiTC and return the handle.

    Returns None — never raises — if the instrument can't be reached.
    Temperature logging is a nice-to-have on every script that uses it;
    a rig without a MercuryiTC (or one that's powered off, unplugged, or
    just not on the network right now) should not stop a measurement
    that doesn't otherwise depend on it. The failure is logged once, here,
    at connect time — callers that get None back should pass it straight
    through to read_temperature()/shutdown_temperature_controller() (both
    accept None) rather than logging again themselves.
    """
    if not (1 <= len(cfg.sensor_uids) <= 2):
        raise ValueError(
            f"cfg.sensor_uids must list 1 or 2 board UIDs, got {cfg.sensor_uids!r}."
        )
    try:
        mitc = MercuryITC(cfg.visa_resource, timeout=cfg.timeout_ms)
        log.info("MercuryiTC connected: %s  id=%s", cfg.visa_resource, mitc.identification)
        return mitc
    except Exception as exc:
        log.warning(
            "MercuryiTC not connected (%s) — continuing without temperature logging.",
            exc,
        )
        return None


def read_temperature(
    mitc: Optional[MercuryITC],
    cfg: TemperatureControllerConfig,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Read 1 or 2 temperatures (Kelvin) from the MercuryiTC, per
    cfg.sensor_uids. Always returns a (T1, T2) pair — T2 is None if only
    one UID is configured. Never raises.

    If `mitc` is None (connect_temperature_controller() already failed,
    or there's simply no MercuryiTC on this rig), returns (None, None)
    silently — that state was already logged once at connect time, so
    logging it again on every point would just be noise.

    A failure reading the *first* sensor is logged as a warning: if the
    controller is connected at all, that reading is expected to work, so
    a failure there is a real problem worth flagging. A failure reading
    the *second* sensor is NOT logged: plenty of rigs only have one probe
    wired up, so a second-channel failure is the normal case for those
    setups, not something to warn about on every single point.
    """
    if mitc is None:
        return None, None

    t1: Optional[float] = None
    t2: Optional[float] = None

    try:
        t1 = mitc.temperature(cfg.sensor_uids[0])
    except Exception as exc:
        log.warning("MercuryiTC temperature reading failed (%s): %s", cfg.sensor_uids[0], exc)

    if len(cfg.sensor_uids) > 1:
        try:
            t2 = mitc.temperature(cfg.sensor_uids[1])
        except Exception:
            pass  # no warning — see docstring above

    return t1, t2


def set_temperature(
    mitc: Optional[MercuryITC],
    cfg: TemperatureControllerConfig,
    setpoint_K: float,
    uid: Optional[str] = None,
) -> None:
    """
    Program the MercuryiTC to regulate toward `setpoint_K` and switch its
    heater to automatic (PID) control, then return immediately — the
    instrument drives its heater toward the setpoint on its own from
    here on, the same way you'd type a setpoint in on the front panel.

    Unlike read_temperature() (a "nice to have" that never raises),
    driving to a setpoint is something a caller only does because a real
    measurement depends on it getting there, so this DOES raise — if
    `mitc` is None (no MercuryiTC connected) or the instrument rejects
    the setpoint — the same "load-bearing" convention
    kepco_magnet.set_magnet_current() already follows for the magnet.

    Deliberately does NOT wait for the temperature to settle. Different
    measurements want very different settling criteria — a quick survey
    sweep might accept +/-0.5 K after a minute, a measurement near a
    phase transition might want +/-0.01 K held for many minutes — so
    there's no single policy to bake in here. Call
    wait_for_temperature_stable() below afterward if a program wants to
    block on a specific tolerance/dwell, or poll read_temperature()
    directly for full control (e.g. to log temperature while it settles,
    or watch a live plot instead of blocking at all).

    Args:
        uid: Which sensor's loop to program. Defaults to
            cfg.sensor_uids[0] (the rig's primary probe — normally the
            one with an associated heater board, e.g. "MB1.T1" paired
            with heater "MB0.H1"). Pass this explicitly only if a rig's
            heater loop is instead tied to the second configured probe.
    """
    if mitc is None:
        raise RuntimeError("Cannot set a temperature setpoint: no MercuryiTC connected.")
    target_uid = uid if uid is not None else cfg.sensor_uids[0]
    mitc.set_temperature_setpoint(target_uid, setpoint_K)
    mitc.heater_auto(target_uid, True)
    log.info("MercuryiTC setpoint programmed: %s -> %.4f K (heater auto ON)", target_uid, setpoint_K)


def wait_for_temperature_stable(
    mitc: Optional[MercuryITC],
    cfg: TemperatureControllerConfig,
    target_K: float,
    uid: Optional[str] = None,
    tolerance_K: float = 0.05,
    hold_time_s: float = 30.0,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 1.0,
) -> None:
    """
    Block until the sensor reads within `tolerance_K` of `target_K`
    continuously for `hold_time_s`, then return. Raises TimeoutError if
    `timeout_s` elapses first.

    Optional — call this only if a program actually wants to block here.
    A program with its own settling logic (or one that wants to log
    points, check other instruments, or update a live plot while
    waiting) can skip this and poll read_temperature() itself instead;
    see set_temperature()'s docstring for why no single waiting policy
    is baked into the setpoint call.

    Any reading outside `tolerance_K` restarts the hold-time clock —
    deliberately stricter than "was in range at least once", since a
    reading merely passing through the target on a PID overshoot
    shouldn't count as "settled".

    Args:
        uid: Which sensor to poll. Defaults to cfg.sensor_uids[0], same
            as set_temperature().
        tolerance_K: How close counts as "at" the target. Default 0.05 K
            is a starting point, not a calibrated value for any
            particular rig/sample — tighten or loosen per measurement.
        hold_time_s: How long the reading must stay in tolerance before
            this returns.
        timeout_s: Give up and raise after this long regardless.
        poll_interval_s: Delay between temperature() reads while waiting.

    Raises:
        RuntimeError: `mitc` is None (no MercuryiTC connected).
        TimeoutError: settled reading not achieved within `timeout_s`.
    """
    if mitc is None:
        raise RuntimeError("Cannot wait for temperature: no MercuryiTC connected.")
    target_uid = uid if uid is not None else cfg.sensor_uids[0]

    deadline = time.monotonic() + timeout_s
    in_tolerance_since: Optional[float] = None

    while True:
        now = time.monotonic()
        current_K = mitc.temperature(target_uid)

        if abs(current_K - target_K) <= tolerance_K:
            if in_tolerance_since is None:
                in_tolerance_since = now
            elif now - in_tolerance_since >= hold_time_s:
                log.info(
                    "MercuryiTC settled: %s = %.4f K (target %.4f +/- %.4f K)",
                    target_uid, current_K, target_K, tolerance_K,
                )
                return
        else:
            in_tolerance_since = None

        if now >= deadline:
            raise TimeoutError(
                f"MercuryiTC did not settle to {target_K:.4f} +/- {tolerance_K:.4f} K "
                f"within {timeout_s:.0f} s (last reading: {current_K:.4f} K)"
            )
        time.sleep(poll_interval_s)


def shutdown_temperature_controller(mitc: Optional[MercuryITC]) -> None:
    """
    Close the MercuryiTC session, if one was opened. Safe to call with
    None (e.g. when connect_temperature_controller() returned None).
    """
    if mitc is None:
        return
    try:
        mitc.close()
    except Exception:
        log.exception("Error while closing MercuryiTC connection")
    else:
        log.info("MercuryiTC connection closed")
