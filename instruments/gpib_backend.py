"""GPIB transport selection — Windows (NI-488.2 via gpib-ctypes, resource
string passed straight to pyvisa) vs macOS (Prologix GPIB-USB Controller
over serial, no vendor GPIB driver of any kind). NI never shipped a
current NI-488.2 driver for macOS/Apple Silicon, and gpib-ctypes has no
Darwin branch at all — so on macOS every "GPIB0::N::INSTR" resource is
rerouted through a Prologix controller instead.

The OS check happens once, at import time (`_IS_MACOS`), not per call.
Every `connect_*()` in `instruments/*.py` wraps its `cfg.visa_resource`
through `resolve_visa_resource()` before constructing its driver instance
— pymeasure's `Instrument`/`VISAAdapter` accepts either a resource string
or an already-built `Adapter` object interchangeably, so this is the only
change needed at each call site.

Usage (inside a connect_*() function)::

    from instruments.gpib_backend import resolve_visa_resource

    dev = Keithley2400(resolve_visa_resource(cfg.visa_resource))

macOS setup: set `BRIDGE_PROLOGIX_PORT` to the Prologix controller's
serial device path, e.g. `/dev/cu.usbserial-XXXXXXXX` (find it with
`ls /dev/cu.usbserial-*`). All instruments share one Prologix controller,
addressed by GPIB address via `PrologixAdapter.gpib()` — the same pattern
pymeasure documents for talking to multiple instruments over one adapter.
"""
import os
import platform
import re

from pymeasure.adapters import PrologixAdapter

_IS_MACOS = platform.system() == "Darwin"          # one-time check, not per-call
_GPIB_RE = re.compile(r"^GPIB\d*::(\d+)::INSTR$", re.IGNORECASE)

_controller: PrologixAdapter | None = None


def _prologix_controller() -> PrologixAdapter:
    global _controller
    if _controller is None:
        port = os.environ.get("BRIDGE_PROLOGIX_PORT")
        if not port:
            raise RuntimeError(
                "BRIDGE_PROLOGIX_PORT is not set -- point it at the Prologix "
                "controller's serial device, e.g. /dev/cu.usbserial-XXXXXXXX "
                "(see `ls /dev/cu.usbserial-*`).")
        _controller = PrologixAdapter(port)
    return _controller


def resolve_visa_resource(visa_resource: str):
    """What a connect_*() should hand its driver: `visa_resource` unchanged
    on Windows/Linux or for a non-GPIB resource (TCPIP0::..., ...); a
    PrologixAdapter for the same GPIB address on macOS."""
    if not _IS_MACOS:
        return visa_resource
    match = _GPIB_RE.match(visa_resource)
    if not match:
        return visa_resource          # e.g. TCPIP0::... -- untouched
    return _prologix_controller().gpib(int(match.group(1)))
