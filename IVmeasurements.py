import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from pymeasure.instruments.keithley import Keithley2450
from pymeasure.experiment import Procedure, Results, Worker
from pymeasure.experiment import FloatParameter, IntegerParameter
from time import sleep
import numpy as np


class IV2450Procedure(Procedure):

    data_points = IntegerParameter('Data points', default=20)
    averages = IntegerParameter('Averages', default=8)
    max_current = FloatParameter('Maximum current', units='A', default=1e-6)
    min_current = FloatParameter('Minimum current', units='A', default=-1e-6)

    DATA_COLUMNS = ['Current (A)', 'Voltage (V)', 'Voltage Std (V)']

    def startup(self):
        log.info("Connecting and configuring the instrument")
        self.sourcemeter = Keithley2450("GPIB::16")
        self.sourcemeter.reset()
        self.sourcemeter.use_front_terminals()
        self.sourcemeter.apply_current(current_range=1e-3, compliance_voltage=20.0)
        self.sourcemeter.measure_voltage(1,21)
        sleep(0.1)
        self.sourcemeter.stop_buffer()
        self.sourcemeter.disable_buffer()


    def execute(self):
        currents = np.linspace(
            self.min_current,
            self.max_current,
            num=self.data_points
        )
        self.sourcemeter.enable_source()

        # Loop through each current point and measure the voltage
        log.info("Starting IV measurement using Keithley 2450")
        for current in currents:
            self.sourcemeter.config_buffer(self.averages.value)
            log.info(f"Setting the current to {current} A")
            self.sourcemeter.source_current = current
            self.sourcemeter.start_buffer()
            log.info("Waiting for buffer to fill with measurements")
            self.sourcemeter.wait_for_buffer()
            data = {
                'Current (A)': current,
                'Voltage (V)': self.sourcemeter.mean_voltage,
                'Voltage Std (V)': self.sourcemeter.std_voltage
            }
            self.emit('results', data)
            sleep(0.01)
            if self.should_stop():
                log.info("User aborted the procedure")
                break


    def shutdown(self):
        self.sourcemeter.shutdown()
        log.info("Finished measuring")


if __name__ == '__main__':
    pass
