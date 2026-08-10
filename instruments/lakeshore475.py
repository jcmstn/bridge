"""
Lake Shore Model 475 DSP Gaussmeter Driver
===========================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-03

Built on pymeasure's Instrument architecture. pymeasure ships drivers for
the 421 and 425 Gaussmeters (pymeasure.instruments.lakeshore) but not the
475 — this fills that gap using the same proprietary command language
that whole DSP Gaussmeter family shares (RDGFIELD?, UNIT, RANGE, AUTO,
RDGMODE, ZPROBE), modeled closely on pymeasure's own LakeShore425 driver,
plus the IEEE-488.2 common commands (*IDN? etc.) the 475 adds courtesy of
its full GPIB implementation.

Interface: GPIB (IEEE 488.2) or RS-232C
Firmware command reference: Lake Shore 475 DSP Gaussmeter user's manual.

Usage example:
    from instruments.lakeshore475 import LakeShore475

    gm = LakeShore475("GPIB0::12::INSTR")
    gm.unit = "T"
    print(gm.field)             # single reading, Tesla
    mean, std = gm.measure(10)  # average of 10 readings
    gm.close()

Usage example (shared controller, used by the DC measurement scripts):
    from instruments.lakeshore475 import GaussmeterConfig, connect_gaussmeter, read_field_mT, shutdown_gaussmeter

    gauss_cfg = GaussmeterConfig(visa_resource="GPIB0::12::INSTR", unit="T")
    gm = connect_gaussmeter(gauss_cfg)
    field_mT = read_field_mT(gm, gauss_cfg)
    ...
    shutdown_gaussmeter(gm)
"""

import logging
from dataclasses import dataclass
from time import sleep

import numpy as np

from pymeasure.instruments import Instrument
from pymeasure.instruments.validators import strict_discrete_set, truncated_range

log = logging.getLogger(__name__)


class LakeShore475(Instrument):
    """
    Represents the Lake Shore Model 475 DSP Gaussmeter and provides a
    high-level interface for interacting with the instrument.

    .. code-block:: python

        gaussmeter = LakeShore475("GPIB0::12::INSTR")
        gaussmeter.unit = "T"           # Set units to Tesla
        gaussmeter.auto_range = True    # Turn on auto-range
        print(gaussmeter.field)
    """

    UNITS = {"G": 1, "T": 2, "Oe": 3, "A/m": 4}

    def __init__(self, adapter, name="Lake Shore 475 Gaussmeter", **kwargs):
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

    field = Instrument.measurement(
        "RDGFIELD?",
        """ Get the field reading in the currently configured unit. """,
        cast=float,
    )

    unit = Instrument.control(
        "UNIT?", "UNIT %d",
        """ Control the field unit used by the instrument. Valid values
        are 'G' (Gauss), 'T' (Tesla), 'Oe' (Oersted), or 'A/m'. (str) """,
        validator=strict_discrete_set,
        values=UNITS,
        map_values=True,
    )

    auto_range = Instrument.control(
        "AUTO?", "AUTO %d",
        """ Control the auto-range option of the meter. Valid values are
        True and False. Note that auto-range is relatively slow and might
        not suffice for rapid measurements. (bool) """,
        validator=strict_discrete_set,
        values={True: 1, False: 0},
        map_values=True,
    )

    field_range_raw = Instrument.control(
        "RANGE?", "RANGE %d",
        """ Control the field range as a raw integer index (1 = most
        sensitive). The number of available ranges and their full-scale
        values depend on the probe installed — consult the manual for
        your specific probe. (int) """,
        validator=truncated_range,
        values=[1, 9],
        cast=int,
    )

    @property
    def mode(self):
        """ Control the mode, filter, and bandwidth settings as a
        (mode, filter, band) tuple. See the manual for RDGMODE for the
        meaning of each value. """
        return tuple(self.values("RDGMODE?"))

    @mode.setter
    def mode(self, value):
        mode, filter_, band = value
        self.write(f"RDGMODE {mode:d},{filter_:d},{band:d}")

    def dc_mode(self, wideband: bool = True) -> None:
        """ Set up a steady-state (DC) measurement of the field. """
        self.mode = (1, 0, 1) if wideband else (1, 0, 2)

    def ac_mode(self, wideband: bool = True) -> None:
        """ Set up a measurement of an oscillating (AC) field. """
        self.mode = (2, 1, 1) if wideband else (2, 1, 2)

    def zero_probe(self) -> None:
        """ Initiate the zero-field sequence to calibrate the probe. Do
        this with the probe in a zero-gauss chamber, or in ambient field
        with the probe held still, before a measurement that needs
        absolute accuracy. """
        self.write("ZPROBE")

    def measure(self, points: int, delay: float = 1e-3) -> tuple[float, float]:
        """
        Return the mean and standard deviation of `points` field readings,
        blocking for `delay` seconds between each.
        """
        data = np.empty(points, dtype=np.float64)
        for i in range(points):
            data[i] = self.field
            sleep(delay)
        return float(data.mean()), float(data.std())

    def close(self) -> None:
        """ Close the underlying VISA session. """
        self.adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Shared controller  ── used directly by the DC measurement scripts ───────────
# ─────────────────────────────────────────────────────────────────────────────
# Every DC program that measures the field live (dc_hall_measurement.py,
# dc_gate_sweep.py, dc_spin_valve.py) opened/read/closed the 475 with an
# identical few lines — moved here once, the same way mercury_itc.py already
# shares its connect/read/shutdown helpers, so a new program that only needs
# the gaussmeter can import this instead of re-typing the setup sequence.

_FIELD_TO_MT = {"T": 1e3, "G": 1e-1}   # → mT, for GaussmeterConfig.unit


@dataclass
class GaussmeterConfig:
    """Lake Shore 475 DSP Gaussmeter — measures the actual field, shared by every
    DC program that logs it (never inferred from a magnet-current calibration)."""
    visa_resource: str   = "GPIB0::12::INSTR"
    unit:          str   = "T"     # 'T' or 'G' — read_field_mT() only knows these two
    n_averages:    int   = 10      # Field readings averaged per measurement point
    read_delay_s:  float = 0.05    # Delay between successive readings  [s]


def connect_gaussmeter(cfg: GaussmeterConfig) -> "LakeShore475":
    """Open a VISA session to the Lake Shore 475 and set its display unit."""
    if cfg.unit not in _FIELD_TO_MT:
        raise ValueError(f"Unsupported gaussmeter unit {cfg.unit!r}; use 'T' or 'G'.")
    gm = LakeShore475(cfg.visa_resource)
    gm.unit = cfg.unit
    log.info("Gaussmeter connected: %s  unit=%s  id=%s",
              cfg.visa_resource, cfg.unit, gm.identification)
    return gm


def read_field_mT(gm: "LakeShore475", cfg: GaussmeterConfig) -> float:
    """Average `cfg.n_averages` field readings and return the result in mT."""
    mean, _std = gm.measure(cfg.n_averages, delay=cfg.read_delay_s)
    return mean * _FIELD_TO_MT[cfg.unit]


def shutdown_gaussmeter(gm: "LakeShore475") -> None:
    """Close the VISA session to the gaussmeter."""
    gm.close()
    log.info("Gaussmeter connection closed")
