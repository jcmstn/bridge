import logging
from time import sleep, time

import numpy as np

from pymeasure.instruments import Instrument
from pymeasure.instruments.validators import truncated_range
from pymeasure.adapters import PrologixAdapter

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Keithley2450Buffer:
    """ Implements the buffering capability for Keithley 2450 using defbuffer1.
        Compatible with original API, adapted to 2450 SCPI commands.
    """

    buffer_points = Instrument.control(
        ":TRACe:POIN? \"defbuffer1\"", ":TRACe:POIN %d,\"defbuffer1\"",
        """ Control the number of buffer points in defbuffer1. """,
        validator=truncated_range,
        values=[1, 250000],  # 2450 defbuffer1 max ~250k points
        cast=int
    )

    def config_buffer(self, points=64, delay=0):
        """ Configure the measurement buffer for a number of points, to be
        taken with a specified delay.

        :param points: The number of points in the buffer.
        :param delay: The delay time in seconds.
        """
        self.write(":STAT:PRES;*CLS;")
        self.write(":TRACe:CLEar \"defbuffer1\";")
        self.buffer_points = points
        self.trigger_count = points
        self.trigger_delay = delay
        self.write(":TRACe:FEED SENS1;:TRACe:FEED:CONT NEXT,\"defbuffer1\";")
        self.check_errors()

    def is_buffer_full(self):
        """ Return True if the buffer is full of measurements (2450 uses MEAS event). """
        status_byte = int(self.ask("*STB?"))
        # MEAS (bit 6=64) + buffer full (bit 0=1, adapted from original logic)
        return (status_byte & 65) == 65

    def wait_for_buffer(self, should_stop=lambda: False,
                        timeout=60, interval=0.1):
        """ Block until buffer full, stats ready, or timeout/stop.
        """
        t = time()
        while not self.is_buffer_full():
            sleep(interval)
            if should_stop():
                return
            if (time() - t) > timeout:
                raise Exception("Timed out waiting for Keithley 2450 buffer to fill.")

    @property
    def buffer_data(self):
        """ Get numpy array from buffer. """
        self.write(":FORM:DATA ASCii")
        return np.array(self.values(":TRACe:DATA? \"defbuffer1\","), dtype=np.float64)

    def buffer_mean(self):
        """ Get the calculated mean of the buffer (2450 feature). """
        return float(self.values(":TRACe:STATistics:AVERage? \"defbuffer1\"," ))

    def buffer_std(self):
        """ Get the calculated standard deviation of the buffer (2450 feature). """
        return float(self.values(":TRACe:STATistics:STDev? \"defbuffer1\"," ))

    def start_buffer(self):
        """ Starts the buffer measurements. """
        self.write(":INIT")

    def reset_buffer(self):
        """ Reset the buffer and status. """
        self.write(":STAT:PRES;*CLS;:TRACe:CLEar \"defbuffer1\"; :TRACe:FEED:CONT NEXT,\"defbuffer1\";")

    def stop_buffer(self):
        """ Abort the buffering measurement. """
        if isinstance(self.adapter, PrologixAdapter):
            self.write("++clr")
        else:
            self.write(":ABORt")

    def disable_buffer(self):
        """ Disable buffer feeding. """
        self.write(":TRACe:FEED:CONT NEVer,\"defbuffer1\";")

    def enable_stats(self):
        """ Enable statistics calculation on defbuffer1 (for mean/std). """
        self.write(":TRACe:STATistics:COUNt %d,\"defbuffer1\";" % self.buffer_points)
        self.write(":TRACe:STATistics:STATe ON,\"defbuffer1\";")
