"""
Oxford Instruments MercuryiTC Temperature Controller Driver + Controller
=========================================================================
Built on pymeasure's Instrument architecture, the same way
lakeshore475.LakeShore475 wraps the Lake Shore 475 Gaussmeter. pymeasure
ships a driver for the older ITC 503 (pymeasure.instruments.oxfordinstruments)
but the MercuryiTC uses a completely different, newer ASCII command
language (colon-separated, hierarchical "device" addressing rather than
ITC 503's single-letter commands) — this fills that gap.

Like lakeshore475.py, this is hand-built from the published Mercury
Support command reference and has not been verified against real
hardware yet — check the reply format against your own iTC before relying
on it (the low-level MercuryITC.temperature() method is the only place
that would need adjusting).

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
    print(mitc.temperature("DB6.T1"))   # single reading, Kelvin
    mitc.close()

Usage example (shared controller, used by the DC/MFLI measurement scripts):
    from mercury_itc import (
        TemperatureControllerConfig, connect_temperature_controller,
        read_temperature, shutdown_temperature_controller,
    )

    temp_cfg = TemperatureControllerConfig(
        visa_resource="TCPIP0::192.168.1.5::7020::SOCKET",
        sensor_uids=("DB6.T1", "DB5.T1"),   # 1 or 2 probes — set to your rig
    )
    mitc = connect_temperature_controller(temp_cfg)   # None if unreachable

    ...
    t1_K, t2_K = read_temperature(mitc, temp_cfg)      # (None, None) if no iTC
    ...

    shutdown_temperature_controller(mitc)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from pymeasure.instruments import Instrument

log = logging.getLogger(__name__)


class MercuryITC(Instrument):
    """
    Represents the Oxford Instruments MercuryiTC and provides a high-level
    interface for reading temperature signals from its temperature boards.

    Each temperature sensor probe is wired to a separate board inside the
    Mercury chassis, addressed by a UID such as "DB6.T1" (read off the
    iTC's front panel or via `catalog`). A given system may have one such
    board or several — `temperature()` reads whichever UID you ask for.

    .. code-block:: python

        mitc = MercuryITC("TCPIP0::192.168.1.5::7020::SOCKET")
        print(mitc.temperature("DB6.T1"))   # Kelvin
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
        "DB6.T1"), in Kelvin.

        Raises ValueError if the reply doesn't have the expected
        `STAT:DEV:<uid>:TEMP:SIG:TEMP:VALUE:<value>K` shape — e.g. because
        `uid` doesn't exist on this chassis, or the board reports an
        error (no probe plugged in, open circuit, etc.) rather than a
        temperature.
        """
        command = f"READ:DEV:{uid}:TEMP:SIG:TEMP"
        reply = self.ask(command).strip()

        expected_prefix = f"STAT:DEV:{uid}:TEMP:SIG:TEMP:VALUE:"
        if not reply.startswith(expected_prefix) or not reply.endswith("K"):
            raise ValueError(
                f"Unexpected reply to {command!r}: {reply!r} "
                f"(expected {expected_prefix}<value>K)"
            )

        value_str = reply[len(expected_prefix):-1]
        return float(value_str)

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
# Unlike the Kepco magnet / Lake Shore 475 wrapper functions (which are
# small enough that each measurement script keeps its own copy — see
# kepco_magnet.py's docstring), read_temperature()'s "1 or 2 sensors,
# never let a missing/failed instrument stop the measurement" logic is
# shared here so every DC/MFLI script gets the same graceful-degradation
# behavior for free instead of re-implementing it.

@dataclass
class TemperatureControllerConfig:
    """
    Oxford Instruments MercuryiTC, used to log sample/probe temperature
    alongside a measurement — shared across DC and MFLI programs the same
    way the Kepco supply and Lake Shore 475 gaussmeter are.

    sensor_uids lists 1 or 2 board UIDs, e.g. ("DB6.T1",) for a rig with a
    single probe or ("DB6.T1", "DB5.T1") for one with two. Find the UIDs
    for your own rig via the iTC front panel or MercuryITC(...).catalog.
    """
    visa_resource: str = "TCPIP0::192.168.1.5::7020::SOCKET"
    sensor_uids: Tuple[str, ...] = ("DB6.T1",)
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
