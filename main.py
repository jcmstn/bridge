import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import sys
from pymeasure.display.Qt import QtWidgets
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import IntegerParameter, FloatParameter, Parameter
from IVmeasurements import IV2450Procedure
from IVgmeasurements import SVMC18_SV16_Procedure, SVMC18_SC16_Procedure
import platform

print(platform.system())
if platform.system() == "Darwin":
    path_to_lib = '/Library/Frameworks/NI4882.framework/NI4882'
    log.info(f"MacOS environment detected. Looking for NI4882 library in {path_to_lib}")
    import gpib_ctypes
    gpib_ctypes.gpib.gpib._load_lib(path_to_lib)  # On MacOS, to find the NI library


class MainWindow(ManagedWindow):

    def __init__(self):
        super().__init__(
            procedure_class=SVMC18_SC16_Procedure,
            inputs=['data_points', 'averages', 'max_vg', 'min_vg', 'current_bias'],
            displays=['data_points', 'averages', 'max_vg', 'min_vg', 'current_bias'],
            x_axis='Gate Voltage (V)',
            y_axis='Current (A)',
            sequencer=True,
            # sequence_file='path/to/file'
        )
        self.setWindowTitle('GUI Example')

        self.filename = r'I_vs_Vg'
        self.store_measurement = True
        self.file_input.extensions = ['txt', 'csv']  # First is default
        self.file_input.filename_fixed = False


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
