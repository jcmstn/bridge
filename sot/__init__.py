"""SOT (spin-orbit torque) measurement programs — Keithley 4200A-SCS.

Three programs, all 4200A-only on the electrical side:
  * sot_dc_characterization — 4-probe channel resistance R_xx (type RXX)
  * sot_switching           — quasi-static (DC staircase) SOT switching (type SOTSW)
  * sot_pulsed_switching    — PMU write pulse + delayed dual-SMU R_xy read (type SOTPS)

Field (assist / read field) still comes from the Kepco magnet + Lake Shore 475,
exactly as in dc/.

The pulsed program runs a KULT user module on the 4200A — source in
instruments/kult/, which also documents the RPM-pathway trap that silently
breaks the other two programs after a vendor pulse test.
"""
