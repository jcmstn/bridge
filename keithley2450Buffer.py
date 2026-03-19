#
# This file is part of the PyMeasure package.
#
# Copyright (c) 2013-2026 PyMeasure Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OR CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import logging
from time import sleep, time

import numpy as np

from pymeasure.instruments import Instrument
from pymeasure.instruments.validators import truncated_range
from pymeasure.adapters import PrologixAdapter

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Keithley2450Buffer:
    """ Keithley 2450 compatible buffer class using defbuffer1.
        Full backward compatibility with original KeithleyBuffer API.
    """

    buffer_points = Instrument.control(
        ":TRACe:POIN? \"defbuffer1\"", ":TRACe:POIN %d,\"defbuffer1\"",
        """ Control the number of buffer points in defbuffer1.
        Does not represent actual points in buffer, but configuration value. """,
        validator=truncated_range,
        values=[1, 250000],  # 2450 defbuffer1 max capacity
        cast=int
    )

    def config_buffer(self, points=64, delay=0):
        """ Configure measurement buffer for specified points and delay.

        :param points: Number of points in buffer.
        :param delay: Delay time in seconds.
        """
        # Enable measurement status and buffer full bits
        self.write(":STAT:PRES;*CLS;*SRE 1;:STAT:MEAS:ENAB 512;")
        self.write(":TRACe:CLEar \"defbuffer1\";")
        self.buffer_points = points
        self.trigger_count = points
        self.trigger_delay = delay
        self.write(":TRACe:FEED SENS1;:TRACe:FEED:CONT NEXT,\"defbuffer1\";")
        self.check_errors()

    def is_buffer_full(self):
        """ Return True if buffer is full (MEAS event + status). """
        status_byte = int(self.ask("*STB?"))
        return (status_byte & 65) == 65

    def wait_for_buffer(self, should_stop=lambda: False,
                        timeout=60, interval=0.1):
        """ Wait for full buffer or timeout/stop condition.

        :param should_stop: Function returning True to exit early.
        :param timeout: Seconds before timeout.
        :param interval: Poll interval in seconds.
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
        """ Get numpy array of raw buffer values. """
        self.write(":FORM:DATA ASCii")
        return np.array(self.values(":TRACe:DATA? \"defbuffer1\","), dtype=np.float64)

    @property
    def mean_voltage(self):
        """ Get calculated mean voltage from buffer statistics. """
        return float(self.values(":TRACe:STATistics:AVERage? \"defbuffer1\"," ))

    @property
    def mean_current(self):
        """ Get calculated mean current from buffer statistics. """
        return float(self.values(":TRACe:STATistics:AVERage? \"defbuffer1\"," ))

    @property
    def mean_resistance(self):
        """ Get calculated mean resistance from buffer statistics. """
        return float(self.values(":TRACe:STATistics:AVERage? \"defbuffer1\"," ))

    @property
    def std_voltage(self):
        """ Get standard deviation of voltage from buffer statistics. """
        return float(self.values(":TRACe:STATistics:STDev? \"defbuffer1\"," ))

    @property
    def std_current(self):
        """ Get standard deviation of current from buffer statistics. """
        return float(self.values(":TRACe:STATistics:STDev? \"defbuffer1\"," ))

    @property
    def std_resistance(self):
        """ Get standard deviation of resistance from buffer statistics. """
        return float(self.values(":TRACe:STATistics:STDev? \"defbuffer1\"," ))

    def enable_statistics(self):
        """ Enable statistics calculation over entire buffer (call after config_buffer). """
        points = self.buffer_points
        self.write(f":TRACe:STATistics:COUNt {points},\"defbuffer1\";")
        self.write(":TRACe:STATistics:STATe ON,\"defbuffer1\";")

    def start_buffer(self):
        """ Start buffer measurements. """
        self.write(":INIT")

    def reset_buffer(self):
        """ Reset buffer and status. """
        self.write(":STAT:PRES;*CLS;:TRACe:CLEar \"defbuffer1\";:TRACe:FEED:CONT NEXT,\"defbuffer1\";")

    def stop_buffer(self):
        """ Abort buffering. Uses SDC for Prologix. """
        if isinstance(self.adapter, PrologixAdapter):
            self.write("++clr")
        else:
            self.write(":ABORt")

    def disable_buffer(self):
        """ Disable buffer feeding (measurements continue). """
        self.write(":TRACe:FEED:CONT NEVer,\"defbuffer1\";")
