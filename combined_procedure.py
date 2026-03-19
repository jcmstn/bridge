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


class CombinedProcedure(Procedure):

    rm = pyvisa.ResourceManager('@py')
    all_resources = rm.list_resources()
    adresses = [r for r in all_resources if 'GPIB' in r.upper()]
    instruments = {
        "keithley2450": [],
        "keithley2400": [],
        "keithley2182": [],
        "keithley6221": [],
        "kepcoPSU": []
    }

    # Parameters
    data_points = IntegerParameter("data_points", default=20)

    # IV
    is_IV = BooleanParameter("IV", default=True)
    use_2400 = BooleanParameter("Use 2400", default=False)
    min_current = FloatParameter("Min current", group_by={"is_IV": True, "sweep_H": False}, default=-1e-6, units="A")
    max_current = FloatParameter("Max current", group_by={"is_IV": True, "sweep_H": False}, default=1e-6, units="A")

    # Field sweep
    sweep_H = BooleanParameter("Sweep H", default=False)
    start_field = FloatParameter("Start field", group_by={"sweep_H": True, "is_IV": False}, default=0, units="T")
    end_field = FloatParameter("End field", group_by={"sweep_H": True, "is_IV": False}, default=1, units="T")
    bias_current = FloatParameter("Bias current", group_by={"sweep_H": True, "is_IV": False}, default=1e-6, units="A")

    # Gate
    use_gate = BooleanParameter("Gate", default=False)
    sweep_gate = BooleanParameter("Sweep gate", group_by={"use_gate": True}, default=True)
    use_current = BooleanParameter("Use current", group_by={"use_gate": True}, default=False)
    sweep_channel = BooleanParameter("Sweep channel", group_by={"use_gate": True, "use_current": True}, default=True)
    min_vg = FloatParameter("Min Vg", group_by={"use_gate": True, "use_current": False, "sweep_gate": True}, default=-10, units="V")
    max_vg = FloatParameter("Max Vg", group_by={"use_gate": True, "use_current": False, "sweep_gate": True}, default=10, units="V")
    gate_current = FloatParameter("Gate current", group_by={"use_gate": True, "use_current": True, "sweep_gate": False}, default=1e-6, units="A")
    min_vch = FloatParameter("Min Vch", group_by={"use_gate": True, "use_current": True, "sweep_channel": True, "sweep_gate": False}, default=-0.1, units="V")
    max_vch = FloatParameter("Max Vch", group_by={"use_gate": True, "use_current": True, "sweep_channel": True, "sweep_gate": False}, default=0.1, units="V")

    DATA_COLUMNS = ["Voltage (V)", "Current (V)", "Std Voltage (V)", "Std Current (A)"]

    def startup(self):
        log.info("Connecting and configuring the instruments")

        for adress in self.adresses:
            try:
                inst = self.rm.open_resource(adress)
                idn = inst.query('*IDN?').lower()

                if "keithley" in idn:
                    if "2450" in idn:
                        self.instruments["keithley2450"].append(Keithley2450(adress))
                    elif "2400" in idn:
                        self.instruments["keithley2400"].append(Keithley2400(adress))
                    elif "2182" in idn:
                        self.instruments["keithley2182"].append(Keithley2182(adress))
                    elif "6221" in idn:
                        self.instruments["keithley6221"].append(Keithley6221(adress))
                    else:
                        log.warning(f"Instrument {adress} found, but *IDN? device {idn} is currently not supported.")
                elif "kepco" in idn:
                    self.instruments["kepcoPSU"].append(KepcoBOP3612(adress))
                # elif "zurich" TODO
                else:
                    log.warning(f"Instrument {adress} found, but *IDN? device {idn} is currently not supported.")

            except Exception as e:
                log.warning(f"{adress} returns error - {e}")


    def execute(self):

        if self.is_IV:
            # IV measurement. Check instrument availability
            if self.use_2400:
                self.sourcemeter = self.instruments["keithley2400"][0]
                self.sourcemeter.enable_source()
