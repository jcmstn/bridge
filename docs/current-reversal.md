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

### What the uncertainty column holds

`mean` / `even_mean` are the pair-averaged `V_odd` / `V_even`. The paired
`*_sem_V` column (`hall_voltage_sem_V`, `voltage_sem_V`,
`hall_voltage_even_sem_V`, `voltage_even_sem_V`) is the **standard error of
that mean** — the sample standard deviation over the `n_reversals` pairs
divided by `sqrt(n_reversals)`, i.e. the error bar you would put on the
reported `V` / `R`. The raw pair-to-pair scatter, if you want it instead, is
`sem * sqrt(n_reversals)`; `n_reversals` (or `n_averages` in the reversal-off
path) is on every row. `sem` is `nan` when only one pair/sample was collected
(an aborted point) — one sample gives no scatter to estimate from.

The plain-average path (`acquire_averaged_voltage`, used by the I–V and
gate-sweep programs and by `dc_spin_valve.py` when reversal is switched off)
follows the same convention: `voltage_sem_V` is sample-stdev / `sqrt(n)` over
the `n_averages` readings.

Older raw files (written before this changeover) carry a `*_std_V` column
instead — that one was the *population* standard deviation (`ddof=0`) of the
same samples, `sqrt((n-1)/n)` smaller than the sample stdev and `sqrt(n)`
larger than the SEM. Branch on which column is present when reading old data.

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
`voltage_even_sem_V`) specifically so this isn't discarded before it can be
checked.
