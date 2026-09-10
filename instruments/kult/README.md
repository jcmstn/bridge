# KULT user modules for the Keithley 4200A-SCS

C source for the user modules `bridge` calls on the 4200A. These are **not** compiled by
this repo — they are compiled on the 4200A itself in KULT (Keithley User Library Tool)
and then invoked over KXCI with `EX <library> <module>(<args>)`.

| File | Module | Called from |
|------|--------|-------------|
| `bridge_sot_pulse.c` | `bridge_sot_pulse` | `instruments/keithley4200a.py` `pulse_once()`, via `sot/sot_pulsed_switching.py` |

The vendor example this was derived from, `PMU_1Chan_Sweep_Example.c`, is kept in
`instruments/` locally but is deliberately **not** tracked (see `.gitignore`).


## Installing `bridge_sot_pulse`

1. On the 4200A, open **KULT**.
2. **File → New Library**, name it `bridge_sot`.
   (Any name works — it just has to match `PMUPulseConfig.library` on the Python side.)
3. **File → New Module**, name it `bridge_sot_pulse`.
4. Paste the body of `bridge_sot_pulse.c` and fill in the parameter list to match the
   `USRLIB MODULE INFORMATION` block at the top of the file: 16 inputs, then 4 outputs
   (`V_Ampl`, `I_Ampl`, `V_Base`, `I_Base`, all `double`, direction **Output**).
5. Set the includes to just `#include "keithley.h"` plus the
   `BOOL LPTIsInCurrentConfiguration(char* hrid);` declaration. The module deliberately
   does **not** need `PMU_examples_ulib_internal.h`.
6. **Build Library**. Fix any compile error before going further — see the two
   verification points below, which are the likely ones.
7. Confirm it is visible to KXCI: from `bridge`,
   ```python
   from instruments.keithley4200a import Keithley4200AConfig, connect_4200a, list_user_libraries
   dev = connect_4200a(Keithley4200AConfig(visa_resource="GPIB0::17::INSTR"))
   print(list_user_libraries(dev))    # KXCI `UL`
   ```


## Build fails: redefinition of every parameter

KULT generates the function signature from the **Parameters** grid. If you also paste the
`int bridge_sot_pulse(...)` signature (or the whole `.c` file) into the **Code** editor,
every parameter is declared twice. Split the file across KULT's three areas:

* **Parameters** grid — the 20 rows from the `USRLIB MODULE INFORMATION` block. Outputs
  (`V_Ampl`, `I_Ampl`, `V_Base`, `I_Base`) must be typed `double *` with direction Output.
* **Includes** box — `#include "keithley.h"`, the `BOOL LPTIsInCurrentConfiguration(...)`
  forward declaration, and the two `#define BRIDGE_ERR_*` lines. Nothing else.
* **Code** editor — only the body, from `/* USRLIB MODULE CODE */` to
  `/* USRLIB MODULE END */`. Not the comment blocks, not the `#include` lines, not the
  signature, not the outermost `{` `}`.


## Two things to verify against your local `keithley.h`

Both are flagged because the vendor example does not exercise them, so they are the
places where this module could be wrong on your firmware.

**0. If the build still fails, read the message-console output and match it below.**
`Sleep()` / `printf()` were removed on purpose (they need `<windows.h>` / `<stdio.h>`,
which the vendor examples get from `PMU_examples_ulib_internal.h` — this module is
`keithley.h`-only). The poll loop uses LPT `delay(1)` instead. If your `keithley.h` spells
`delay` differently, that is the one to fix.

**1. `KI_RPM_SMU`** — the value that routes the RPM back to the SMU pathway.
`PMU_1Chan_Sweep_Example.c` only ever uses `KI_RPM_PULSE`, so the route-back constant is
taken from the documented `rpm_config` pathway set rather than copied from working code.
If the build fails on that symbol, grep `keithley.h` for `KI_RPM_` and substitute the
SMU-pathway name.

**2. No `pulse_sweep_linear`.** This module sets a single amplitude with `pulse_vhigh` /
`pulse_vlow` instead of configuring a one-point sweep. If `pulse_exec` returns no data
(`pulse_fetch` errors, or the spot means come back as zeros with status 0), fall back to
a degenerate sweep — replace the `pulse_vhigh` call with:

```c
    status = pulse_sweep_linear(InstId, Chan, PULSE_AMPLITUDE_SP, AmplitudeV, AmplitudeV, 0);
    if ( status )
        goto cleanup;
```

and keep everything else. (`pulse_vlow` still sets the base in that arrangement.)


## Argument order is a contract

`PMUPulseConfig.arg_order` in `instruments/keithley4200a.py` lists the parameter names in
the order `EX` sends them, and it must match this module's signature exactly — KXCI
passes arguments positionally, so a mismatch silently pulses with the wrong numbers
rather than erroring. If you edit the module's parameter list, edit `arg_order` in the
same commit.

Likewise `PMUPulseConfig.return_names` lists the output parameters in module order, since
they are fetched one at a time with `GN`.

If `PMU_ID` comes back rejected, it is the string quoting: `_fmt_arg` passes `str` values
through verbatim, so set `pmu_id` to `"PMU1"` or `'"PMU1"'` depending on what your KXCI
build wants.


## RPM sense mode

The rig wires only the RPM **FORCE** triax (2-wire), so the RPM channel must be in
**LOCAL (2-wire) sense** — that is the 4225-PMU/RPM power-up default, and no vendor
example (nor this module) changes it, so normally there is nothing to do. If the channel
was ever switched to remote sense in **KCON**, switch it back: a remote-sense channel with
the SENSE triax capped will not source correctly. `bridge_sot_pulse` does not assert the
mode over LPT; if you want it to, `pulse_remote_sense()` is the call — confirm the
local-sense constant in your `keithley.h` first.


## 6221 front-panel checks (shared-bus wiring)

The 6221 output sits on the same I+/I- pads as the PMU pulse. `sot_pulsed_switching.py`
opens the 6221 output before every pulse, so the 6221 only sees the pulse across open
terminals — but two 6221 settings the code cannot read back must be right for that to
hold:

* **Output-off state = NORMAL** (factory default — the output relay physically opens).
  `ZERO` keeps the relay closed and the 6221 output stage absorbs every pulse transient.
* **OUTPUT LOW = floating** (not earthed). The common bus already ties I- to the 4200A
  common; a second internal earth is a ground loop through that bus.

The program refuses a read `sense_current_A` > 10 mA or `compliance_V` > 21 V (mistyped
exponent guard) and the TUI warns well below those.


## The RPM pathway trap

`bridge_sot_pulse` routes the RPM back to the SMU pathway on **every** exit, including
error paths. The vendor example does not: it leaves the RPM on the pulse pathway with the
PMU output still enabled.

That matters beyond the pulsed program. `sot/sot_switching.py` and
`sot/sot_dc_characterization.py` also force current from SMU1 through RPM1 — so after
running any Clarius pulse test (or the vendor example) those two programs will read an
**open circuit with no error** until something puts the pathway back. Running
`bridge_sot_pulse` once is the quickest way to recover it.
