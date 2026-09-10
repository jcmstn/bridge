"""SOT (spin-orbit torque) pulsed-switching measurement — Keithley 4200A-SCS.

One program:
  * sot_pulsed_switching — 4200A PMU write pulse + delayed 6221/2182 R_xy read
    (type SOTPS). The 4200A only pulses; a 6221 sources the DC read current and
    a 2182 reads V_xy across the Hall arms. Field (static tilted assist/read
    field) comes from the Kepco magnet + Lake Shore 475, as in dc/.

The pulse runs a KULT user module on the 4200A — source in instruments/kult/,
which also documents the RPM-pathway trap (a Clarius pulse test that does not
route the RPM back leaves an SMU on that RPM reading an open circuit).

Entry point: ``uv run python sot/sot_pulsed_switching_tui.py``.
"""
