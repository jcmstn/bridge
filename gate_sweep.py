import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from pymeasure.instruments.keithley import Keithley2400, Keithley2182, Keithley6221
from keithley2450 import Keithley2450
from pymeasure.instruments.kepco import KepcoBOP3612
from pymeasure.experiment import Procedure, Results, Worker
from pymeasure.experiment import FloatParameter, IntegerParameter, Metadata, BooleanParameter
from time import sleep
import numpy as np
from datetime import datetime
import pyvisa

class GateSweep(Procedure):
    '''
    Gate sweep program with ONE Keithley 2450 and ONE Keithley 2400 or TWO Keithley 2450.
    Sweeps the gate with Keithley 2400 or second 2450.
    First or only 2450 sends a bias voltage (or current) and measures the current in the same channel.
    '''

    GPIB_2450_1 = "GPIB0::16::INSTR"
    GPIB_2450_2 = "GPIB0::18::INSTR"
    GPIB_2400   = "GPIB0::25::INSTR"

    buffer_averages = 10

    use_2400 = BooleanParameter("Use 2400 for gate?", default=True)

    start_vg = FloatParameter("Start Vg", default=0, units="V")
    end_vg = FloatParameter("End Vg", default=1, units="V")

    voltage_bias = BooleanParameter("Use voltage bias?", default=True)
    bias = FloatParameter("Bias", default=1e-6)

    data_points = IntegerParameter("Data points", default=20)

    DATA_COLUMNS = ["Current (A)", "Resistance", "Gate voltage (V)", "Current Std (A)", "Resistance Std"]

    def startup(self):
        # if self.use_2400:

        #     log.info("Connecting to 2450 and 2400")
        #     try:
        #         self.gate = Keithley2400(self.GPIB_2400)
        #         try:
        #             self.channel = Keithley2450(self.GPIB_2450_1)
        #         except Exception:
        #             self.channel = Keithley2450(self.GPIB_2450_2)
        #     except Exception as e:
        #        log.warning(f"Failed connection to instrument: {e}")

        # else:
        #     log.info("Connecting to two 2450")
        #     try:
        #         self.gate = Keithley2450(self.GPIB_2450_1)
        #         self.channel = Keithley2450(self.GPIB_2450_2)
        #     except Exception as e:
        #         log.warning(f"Failed connection to instrument: {e}")
        self.gate = Keithley2400(self.GPIB_2400)
        self.channel = Keithley2450(self.GPIB_2450_2)

        # Configure instruments
        self.gate.reset()
        self.gate.use_front_terminals()
        self.gate.apply_voltage(compliance_current=10e-6)
        sleep(0.1)
        self.gate.stop_buffer()
        self.gate.disable_buffer()
        log.info("Gate sweep configured")

        self.channel.reset()
        self.channel.use_front_terminals()

        if self.voltage_bias: self.channel.apply_voltage(compliance_current=1e-3)
        else: self.channel.apply_current(compliance_voltage=1)

        self.channel.measure_current(current=1e-3, auto_range=True, nplc=1)
        sleep(0.1)
        self.channel.sense_wire_mode = "2"
        self.channel.stop_buffer()
        self.channel.disable_buffer()

        log.info("Channel bias and measure configured")


    def execute(self):
        gate_voltages = np.linspace(
            self.start_vg,
            self.end_vg,
            num=self.data_points
        )

        for voltage in gate_voltages:
            #self.channel.config_buffer(self.buffer_averages)

            log.info(f"Gate voltage set to {voltage} V")
            self.gate.source_voltage = voltage

            if self.voltage_bias: self.channel.source_voltage = self.bias
            else: self.channel.source_current = self.bias

            self.channel.start_buffer()
            self.channel.wait_for_buffer()
            self.channel.disable_buffer()

            data = {
                'Gate voltage (V)': voltage,
                'Current (A)': self.channel.mean_current,
                'Resistance': self.channel.bias / self.channel.mean_current,
                'Current Std (A)': self.channel.std_current,
                'Resistance Std': np.abs(self.channel.bias/(self.channel.mean_current**2)) * self.channel.std_current
            }

            self.emit("results", data)
            sleep(0.1)
            if self.should_stop():
                log.info("User aborted measurement")
                break

    def shutdown(self):
        self.gate.shutdown()
        self.channel.shutdown()
        log.info("Measurement exited.")
