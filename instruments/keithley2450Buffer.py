import logging
from time import sleep, time
import numpy as np

from pymeasure.instruments import Instrument
from pymeasure.instruments.validators import truncated_range

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class Keithley2450Buffer:
    """ Keithley 2450 compatible buffer class using native 2450 SCPI commands. """

    buffer_points = Instrument.control(
        ":TRACe:POINts? \"defbuffer1\"", ":TRACe:POINts %d, \"defbuffer1\"",
        """ Control the number of buffer points in defbuffer1. """,
        validator=truncated_range,
        values=[1, 250000],  # 2450 defbuffer1 max capacity
        cast=int
    )

    def config_buffer(self, points=64, delay=0):
        """ Configure measurement buffer for specified points. """
        self.write("*CLS")
        self.write(":TRACe:CLEar \"defbuffer1\"")
        self.buffer_points = points

        # Native 2450: Set measurement count to match buffer size
        self.write(f":SENSe:COUNt {points}")
        self.check_errors()

    def is_buffer_full(self):
        """ Return True if buffer has reached the requested number of points. """
        # Native 2450: TRACe:ACTual? does NOT take buffer name parameter
        actual = int(self.ask(":TRACe:ACTual?"))
        return actual >= self.buffer_points

    def wait_for_buffer(self, should_stop=lambda: False, timeout=60, interval=0.1):
        """ Wait for full buffer or timeout/stop condition. """
        t = time()
        while not self.is_buffer_full():
            sleep(interval)
            if should_stop():
                return
            if (time() - t) > timeout:
                # Print buffer status for debugging
                actual = int(self.ask(":TRACe:ACTual?"))
                log.error(f"Buffer timeout: {actual}/{self.buffer_points} points")
                raise Exception("Timed out waiting for Keithley 2450 buffer to fill.")

    @property
    def buffer_data(self):
        """ Get numpy array of raw buffer values. """
        self.write(":FORMat:DATA ASCii")
        points = self.buffer_points
        # Native 2450: TRACe:DATA? requires start, end, buffername, dataelement
        data = self.values(f':TRACe:DATA? 1, {points}, "defbuffer1", READ')
        return np.array(data, dtype=np.float64)

    # Native 2450: Statistics are queried without trailing commas
    @property
    def mean_voltage(self): return float(self.ask(":TRACe:STATistics:AVERage? \"defbuffer1\""))
    @property
    def mean_current(self): return float(self.ask(":TRACe:STATistics:AVERage? \"defbuffer1\""))
    @property
    def mean_resistance(self): return float(self.ask(":TRACe:STATistics:AVERage? \"defbuffer1\""))

    @property
    def std_voltage(self): return float(self.ask(":TRACe:STATistics:STDDev? \"defbuffer1\""))
    @property
    def std_current(self): return float(self.ask(":TRACe:STATistics:STDDev? \"defbuffer1\""))
    @property
    def std_resistance(self): return float(self.ask(":TRACe:STATistics:STDDev? \"defbuffer1\""))

    @property
    def max_voltage(self): return float(self.ask(":TRACe:STATistics:MAXimum? \"defbuffer1\""))
    @property
    def max_current(self): return float(self.ask(":TRACe:STATistics:MAXimum? \"defbuffer1\""))
    @property
    def min_voltage(self): return float(self.ask(":TRACe:STATistics:MINimum? \"defbuffer1\""))
    @property
    def min_current(self): return float(self.ask(":TRACe:STATistics:MINimum? \"defbuffer1\""))

    def enable_statistics(self):
        """ The 2450 computes statistics automatically. Kept for backwards compatibility. """
        pass

    def start_buffer(self):
        """ Start buffer measurements natively on 2450. """
        self.write(":TRACe:TRIGger \"defbuffer1\"")

    def reset_buffer(self):
        """ Reset buffer and status. """
        self.write("*CLS")
        self.write(":TRACe:CLEar \"defbuffer1\"")

    def stop_buffer(self):
        """ Abort buffering. """
        self.write(":ABORt")

    def disable_buffer(self):
        """ Restore instrument to single-shot measurement mode. """
        self.write(":SENSe:COUNt 1")
