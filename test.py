import gpib_ctypes
gpib_ctypes.gpib.gpib._load_lib('/Library/Frameworks/NI4882.framework/NI4882')

import pyvisa
rm = pyvisa.ResourceManager('@py')
print(rm.list_resources())
