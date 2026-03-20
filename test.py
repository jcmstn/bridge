import logging
import pyvisa
from pymeasure.instruments.keithley import Keithley2400
from keithley2450 import Keithley2450

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# GPIB addresses (update to match your NI‑MAX / Keysight‑IO)
GP2450_1 = "GPIB0::16::INSTR"  # 2450 for channel
GP2450_2 = "GPIB0::18::INSTR"  # 2450 for gate (if not using 2400)
GP2400   = "GPIB0::25::INSTR"  # 2400 for gate

rm = pyvisa.ResourceManager()

def test_instrument(name, address, instrument_class):
    try:
        instr = instrument_class(address)
        idn = instr.adapter.connection.query("*IDN?")
        log.info(f"🟢 {name} @ {address}: OK, IDN = {idn.strip()}")
        return instr
    except Exception as e:
        log.error(f"🔴 {name} @ {address}: FAILED, error = {e!r}")
        return None

print("📡 Keithley connection self‑test\n" + "-" * 50)

# Try 2450‑1 (channel)
keithley_2450_1 = test_instrument("Keithley 2450 (1)", GP2450_1, Keithley2450)

# Try 2450‑2 (if present)
keithley_2450_2 = test_instrument("Keithley 2450 (2)", GP2450_2, Keithley2450)

# Try 2400 (gate source)
keithley_2400   = test_instrument("Keithley 2400", GP2400, Keithley2400)

print("\n🔧 If any instrument failed, check:")
print("   - correct GPIB address in NI‑MAX / Keysight‑IO")
print("   - no other program holding the resource open")
print("   - VISA backend (32/64‑bit) matching your Python install")
