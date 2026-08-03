"""
Lake Shore Model 475 DSP Gaussmeter Driver
===========================================
Built on pymeasure's Instrument architecture, the same way KEPCO
magnet/kepco_magnet.py wraps the Kepco BOP-GL. pymeasure ships drivers for
the 421 and 425 Gaussmeters (pymeasure.instruments.lakeshore) but not the
475 — this fills that gap using the same proprietary command language
that whole DSP Gaussmeter family shares (RDGFIELD?, UNIT, RANGE, AUTO,
RDGMODE, ZPROBE), modeled closely on pymeasure's own LakeShore425 driver,
plus the IEEE-488.2 common commands (*IDN? etc.) the 475 adds courtesy of
its full GPIB implementation.

This instrument is shared by several lab programs (see kepco_magnet.py's
docstring for the same philosophy) — add this folder to sys.path rather
than duplicating the driver.

Interface: GPIB (IEEE 488.2) or RS-232C
Firmware command reference: Lake Shore 475 DSP Gaussmeter user's manual.

Usage example:
    from lakeshore475 import LakeShore475

    gm = LakeShore475("GPIB0::12::INSTR")
    gm.unit = "T"
    print(gm.field)             # single reading, Tesla
    mean, std = gm.measure(10)  # average of 10 readings
    gm.close()
"""

from time import sleep

import numpy as np

from pymeasure.instruments import Instrument
from pymeasure.instruments.validators import strict_discrete_set, truncated_range


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
