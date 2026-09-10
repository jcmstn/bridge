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


## Two things to verify against your local `keithley.h`

Both are flagged because the vendor example does not exercise them, so they are the
places where this module could be wrong on your firmware.

**1. `KI_RPM_SMU`** — the value that routes the RPM back to the SMU pathway.
`PMU_1Chan_Sweep_Example.c` only ever uses `KI_RPM_PULSE`, so the route-back constant is
taken from the documented `rpm_config` pathway set rather than copied from working code.
If the build fails on that symbol, grep `keithley.h` for `KI_RPM_` and substitute the
SMU-pathway name.

**2. `pulse_sweep_linear` sets the amplitude, not `pulse_vhigh`.** `pulse_exec` needs a
sweep point defined; a bare `pulse_vhigh` returns **-826** (seen on the bench). The
module uses the documented single-pulse form — `pulse_sweep_linear(InstId, Chan,
PULSE_AMPLITUDE_SP, AmplitudeV, AmplitudeV, 0)` — with `pulse_vlow` still setting the
base. If your firmware rejects `Step = 0`, try a tiny non-zero step (e.g. `1e-6`); the
single point is `Start`.


## Argument order is a contract

`PMUPulseConfig.arg_order` in `instruments/keithley4200a.py` lists the parameter names in
the order `EX` sends them, and it must match this module's signature exactly — KXCI
passes arguments positionally, so a mismatch silently pulses with the wrong numbers
rather than erroring. If you edit the module's parameter list, edit `arg_order` in the
same commit.

KXCI's `EX` wants a value for every output parameter in the call itself (as a placeholder
`0`), so `PMUPulseConfig.n_output_params` must equal the module's output count — a short
`EX` call comes back `EX ERROR: invalid number of UTM parameters`.

The output values are then read back one at a time. **`GN` needs a parameter name**
(`GN V_Ampl`) — a bare `GN` is `GN error: Invalid command syntax`. `pulse_once` uses
`GP <n>` instead (query by 1-based position): the four outputs are params 17-20, right
after the 16 inputs, so it reads `GP 17` … `GP 20`. `PMUPulseConfig.return_names` names
those four columns in that order.

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


## 6221 on the shared-bus wiring

The 6221 output sits on the same I+/I- pads as the PMU pulse. `sot_pulsed_switching.py`
puts the 6221 in standby (`OUTPUT OFF`) before every pulse, so it only sees the pulse
across a non-sourcing high-Z output stage rated to its ±105 V compliance — nothing to
configure (the 6221 has no 2400-style output-off-state selector).

* **OUTPUT LOW = float** (`OUTPut:LTEarth OFF`; front panel CONFIG→OUTPUT). The common
  bus already ties I- / 6221-LO to the 4200A common; `:LTEarth ON` adds a second internal
  earth = a ground loop. Triax inner shield `OUTPut:ISHield` = OLOW (default) is fine.

The program refuses a read `sense_current_A` > 10 mA or `compliance_V` > 21 V (mistyped
exponent guard) and the TUI warns well below those.


## The RPM pathway trap

`bridge_sot_pulse` routes the RPM back to the SMU pathway on **every** exit, including
error paths. The vendor example does not: it leaves the RPM on the pulse pathway with the
PMU output still enabled.

That matters for any SMU work on that RPM channel outside `bridge`: after a Clarius pulse
test (or the vendor example) built on a module that does not route back, an SMU forcing
through RPM1 reads an **open circuit with no error** until something restores the pathway.
Running `bridge_sot_pulse` once is the quickest way to recover it.
