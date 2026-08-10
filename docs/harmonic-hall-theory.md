# Harmonic Hall analysis theory (`analysis/harmonic_hall.py`)

Harmonic Hall analysis for an in-plane magnetised film measured with an
out-of-plane external field.

## Geometry assumed

(Change the sign conventions in `harmonic_hall.py` if your setup differs.)

```
     z  (film normal, external field H applied along +z)
     ^
     |        NM / Ni bilayer, Ni easy plane = film plane
     |
     +-----> x   (charge current I along +x)
    /
   y            (spin polarisation of the SHE spin current, sigma = +y for
                 a positive spin Hall angle with I along +x)
```

## Physics this script assumes

```
R_xy = R_AHE * m_z  +  R_PHE * m_x * m_y
```

With H along z and an easy-plane magnet, m tilts out of plane in the x-z
plane:

```
m_z = H_z / H_K_eff        for |H_z| < H_K_eff
m_z = sign(H_z)            for |H_z| > H_K_eff       (hard-axis loop)
```

The damping-like (DL) effective field is `H_DL_vec ~ H_DL * (m x sigma)`.
For `m = (cos(t), 0, sin(t))` and `sigma = y`: `m x y = (-sin(t), 0, cos(t))`,
so the z-component of the DL field is `H_DL * sqrt(1 - m_z^2)`.

- The DL torque modulates `m_z` → shows up in the AHE → 2f signal.
- It **vanishes** at out-of-plane saturation (m ‖ z), which is a useful
  built-in null: whatever 2f signal survives above `H_K_eff` is background
  (thermal / ANE / offset / pickup), not torque.

The field-like (FL) effective field and the Oersted field are both ‖ y. To
first order they tilt m *in-plane* and therefore appear in the PHE term, not
the AHE term. So this measurement geometry is DL-selective — a feature, but
it also means you cannot get `H_FL` from this dataset alone.

## Second-harmonic amplitude

```
R_2f = 0.5 * H_DL * (dR_1f/dH_z) * sqrt(1 - m_z^2)   +   R_background
```

So a plot of `R_2f` against `0.5 * (dR_1f/dH_z) * sqrt(1 - m_z^2)` is a
straight line whose **slope** is `H_DL` and whose **intercept** is the
non-torque background.

## Second-harmonic fit derivation

`I(t) = I0 sin(wt)` generates `H_DL(t) = H_DL sin(wt)` along `(m x sigma)`.
Its z-component is `H_DL sin(wt) sqrt(1 - m_z^2)`, so

```
dm_z(t) = (dm_z/dH_z) H_DL sqrt(1-m_z^2) sin(wt)
dR_xy   = R_AHE dm_z
V_xy(t) = I0 sin(wt) [ R_1f + R_AHE (H_DL/H_K) sqrt(1-m_z^2) sin(wt) ]
```

and `sin^2(wt) = (1 - cos 2wt)/2`, hence

```
R_2f = 0.5 * H_DL * (dR_1f/dH_z) * sqrt(1 - m_z^2)  +  background.
```

`fit_second_harmonic()` in `harmonic_hall.py` does a least-squares fit of
`R_2f` against this basis, giving `H_DL` as the slope and the non-torque
background as the intercept — see that function's docstring/body for the
two runtime sanity checks this derivation implies (the saturation null, and
why `H_FL` cannot be extracted from this geometry).

## Philosophy

Nothing here is trusted until it has been checked. Every quantity that gets
reported is accompanied by a sanity check that could have failed. The
script will happily tell you that your data is not analysable — that is the
point. A fit that converges is not evidence that the model is right.
