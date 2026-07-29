import os
import logging

# === LOGGING SETUP (overwrites hall_measurement.log each run) ===
LOGFILE = "hall_measurement.log"

if os.path.exists(LOGFILE):
    os.remove(LOGFILE)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)-20s: %(message)s",
    datefmt="%H:%M:%S"
)

file_handler = logging.FileHandler(LOGFILE, mode='w')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

log = logging.getLogger(__name__)
log.info("=== LOGGING INITIALIZED ===")

import sys
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows import ManagedWindow
from hall_measurement import HallMeasurement
import platform


if platform.system() == "Darwin":
    path_to_lib = '/Library/Frameworks/NI4882.framework/NI4882'
    log.info(f"MacOS environment detected. Looking for NI4882 library in {path_to_lib}")
    log.warning("Only x86 archetecture supported. Make sure you are not running on ARM.")
    import gpib_ctypes
    gpib_ctypes.gpib.gpib._load_lib(path_to_lib)  # On MacOS, to find the NI library


class MainWindow(ManagedWindow):

    def __init__(self):
        super().__init__(
            procedure_class=HallMeasurement,
            inputs=[
                'sense_current',
                'compliance_voltage',
                'start_field',
                'end_field',
                'field_points',
                'field_coefficient',
                'averages',
            ],
            displays=[
                'sense_current',
                'compliance_voltage',
                'start_field',
                'end_field',
                'field_points',
                'field_coefficient',
                'averages',
            ],
            x_axis='Magnetic Field (T)',
            y_axis='Hall Voltage (V)',
        )
        self.setWindowTitle('Hall Measurement Program')

        self.filename = r'HallMeasurement'
        self.store_measurement = True
        self.file_input.extensions = ['txt', 'csv']  # First is default
        self.file_input.filename_fixed = False


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
