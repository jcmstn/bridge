# Current-reversal averaging: V_odd and V_even

Used by `instruments.keithley6221.acquire_reversal_averaged_voltage`, called
from every DC measurement program (`dc_hall_measurement.py`,
`dc_gate_sweep.py`, `dc_iv_curve.py`, `dc_spin_valve.py`).

## The general algorithm

At each measurement point the sense current is reversed (+I / -I) over
`n_reversals` pairs, and the resulting voltage is decomposed into an odd and
even part in the current:

```
V_odd  = (V(+I) - V(-I)) / 2      <- reported as "the" voltage/R
V_even = (V(+I) + V(-I)) / 2      <- recorded, not discarded
```

`V_odd` cancels any DC offset common to both polarities (thermal EMFs at the
contacts, amplifier offset, etc.) — this works for any resistive element,
not just an antisymmetric Hall response, since R itself is unchanged by the
current's sign.

But "even in current" is not the same thing as "boring instrumental
offset": expanding `V(I) = V_offset + R*I + beta*I^2 + gamma*I^3 + ...`
shows that `V_odd` keeps only odd powers of I and `V_even` keeps only even
powers — including physics that genuinely lives there, not just the offset.
An offset-cancelling average that only ever reports `V_odd` would silently
zero all of that out. `V_even` is therefore recorded alongside `V_odd` on
every point rather than thrown away: if it's flat noise, nothing was lost;
if it shows structure vs. the swept parameter, that's a real signal
current-reversal averaging alone would otherwise hide.

## Spin-valve-specific interpretation (`dc_spin_valve.py`)

For spin-valve-type stacks with strong spin-orbit coupling, the even-in-I
term can carry real physics rather than just offset: unidirectional spin
Hall magnetoresistance, Joule-heating-driven `Delta-R(T)`, and
rectification-type effects are all even in the current. `dc_spin_valve.py`
records `V_even` in its output columns (`voltage_even_V` /
`voltage_even_std_V`) specifically so this isn't discarded before it can be
checked.
