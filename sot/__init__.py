"""Pulsed-switching measurements: SOT via the Keithley 4200A-SCS PMU, and nonlocal
spin-current switching with only a 6221 + 2182A (spin-transfer torque, not SOT).

Programs:
  * sot_pulsed_switching — 4200A PMU write pulse + delayed 6221/2182 R_xy read
    (type SOTPS). The 4200A only pulses; a 6221 sources the DC read current and
    a 2182 reads V_xy across the Hall arms. Field (static tilted assist/read
    field) comes from the Kepco magnet + Lake Shore 475, as in dc/.

  * sot_nonlocal_switching — no 4200A: 6221 pulse + 2182A nonlocal read for pure-spin-current
    switching of a detector magnet (type NLSW, entry point
    ``uv run python sot/sot_nonlocal_switching_tui.py``).

The pulse runs a KULT user module on the 4200A — source in instruments/kult/,
which also documents the RPM-pathway trap (a Clarius pulse test that does not
route the RPM back leaves an SMU on that RPM reading an open circuit).

Entry point: ``uv run python sot/sot_pulsed_switching_tui.py``.
"""
