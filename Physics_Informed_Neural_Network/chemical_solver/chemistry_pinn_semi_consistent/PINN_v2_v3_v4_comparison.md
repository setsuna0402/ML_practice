# ChemistryPINN PDE Formulations: v2, v3, and v4

## Overview

The three trainer versions differ mainly in **where the chemistry residual is evaluated** and **how the time derivative is obtained from the network output**.

The network predicts normalized quantities. For each species fraction \(X\),

\[
f = A\log_{10}X + C,
\]

where, for the current normalization,

\[
A = 0.4,\qquad C = 2.
\]

The chemistry equations themselves are naturally written in terms of the physical fractions \(X\).

---

## v2 — Physical-space derivative through denormalisation

### Method

v2 first converts the normalized network prediction back to physical fractions:

```python
prediction_norm = model(x)
prediction_phys = denormalise_outputs(prediction_norm)
```

It then asks autograd directly for

\[
\frac{dX}{dt}.
\]

The physical chemistry RHS is evaluated using the predicted physical fractions and frozen Grackle coefficients \(k_1,\ldots,k_6\).

For example,

\[
\frac{d\,{\rm HeII}}{dt}
=
(k_3 n_e+\Gamma_{\rm HeI}){\rm HeI}
-
[(k_4+k_5)n_e+\Gamma_{\rm HeII}]{\rm HeII}
+
k_6n_e{\rm HeIII}.
\]

### Numerical issue

The physical fractions span many orders of magnitude. In addition, the derivative graph passes through the exponential inverse-normalization,

\[
X = 10^{(f-C)/A}.
\]

This produced unstable PDE training in practice. The original implementation of `denormalise_outputs()` also contained in-place slice assignments that conflicted with higher-order autograd.

### Residual scaling

v2 used species-dependent fixed reference rates:

\[
R_X =
\frac{dX/dt - RHS_X}{PDE\_X\_RATE\_REF}.
\]

This was intended as characteristic-rate scaling, but a fixed reference rate does not necessarily represent the actual chemistry timescale of every trajectory.

---

## v3 — Entire PDE rewritten in normalized log-fraction space

### Motivation

v3 was introduced to avoid taking derivatives of very small physical fractions.

Instead of differentiating \(X\), autograd directly differentiates the normalized network output:

\[
\frac{df}{dt}.
\]

Because

\[
f=A\log_{10}X+C,
\]

we have

\[
\frac{df}{dt}
=
\frac{A}{X\ln 10}\frac{dX}{dt}.
\]

The full chemistry equation was therefore rewritten in terms of \(f\).

For HI:

\[
\frac{df_{\rm HI}}{dt}
=
\frac{A}{\ln10}
\left[
-(k_1+k_2)n_e
-\Gamma_{\rm HI}
+k_2n_e\,10^{(C-f_{\rm HI})/A}
\right].
\]

For HeI:

\[
\frac{df_{\rm HeI}}{dt}
=
\frac{A}{\ln10}
\left[
-k_3n_e-\Gamma_{\rm HeI}
+k_4n_e\,10^{(f_{\rm HeII}-f_{\rm HeI})/A}
\right].
\]

### HeII problem

The normalized HeII equation contains ratios such as

\[
\frac{{\rm HeI}}{{\rm HeII}}
=
10^{(f_{\rm HeI}-f_{\rm HeII})/A}.
\]

The source term

\[
T_1
=
\frac{A}{\ln10}
(k_3n_e+\Gamma_{\rm HeI})
\frac{{\rm HeI}}{{\rm HeII}}
\]

becomes extremely large when HeII is initially tiny while HeI and \(\Gamma_{\rm HeI}\) are not small.

This is not merely an implementation bug. It follows mathematically from

\[
\frac{df_{\rm HeII}}{dt}
\propto
\frac{1}{{\rm HeII}}
\frac{d\,{\rm HeII}}{dt}.
\]

Therefore a perfectly finite physical HeII production rate can correspond to an enormous log-fraction derivative near \( {\rm HeII}\rightarrow 0\).

### Other v3 changes

v3 also introduced gradient-norm clipping to reduce large optimizer updates.

---

## v4 — Differentiate in normalized space, compare in physical space

### Main idea

v4 keeps the numerically useful part of v3:

- autograd differentiates the **normalized network output**;

but avoids rewriting the chemistry equation in log-fraction space.

Starting from

\[
f=A\log_{10}X+C,
\]

the chain rule gives

\[
\frac{df}{dt}
=
\frac{A}{X\ln10}\frac{dX}{dt},
\]

hence

\[
\boxed{
\frac{dX}{dt}
=
\frac{\ln10}{A}X\frac{df}{dt}
}
\]

v4 therefore performs the following steps:

1. Network predicts normalized \(f_{\rm HI},f_{\rm HeI},f_{\rm HeII}\).
2. Autograd computes \(df/dt\).
3. Fractions are reconstructed algebraically:
   \[
   X=10^{(f-C)/A}.
   \]
4. The chain rule converts \(df/dt\) into physical \(dX/dt\).
5. The ordinary physical chemistry RHS is evaluated.
6. The residual is
   \[
   R_X = \frac{dX}{dt}-RHS_X.
   \]

### HeII example

The derivative side is

\[
\frac{d\,{\rm HeII}}{dt}
=
\frac{\ln10}{A}
{\rm HeII}
\frac{df_{\rm HeII}}{dt}.
\]

The RHS remains

\[
(k_3n_e+\Gamma_{\rm HeI}){\rm HeI}
-
[(k_4+k_5)n_e+\Gamma_{\rm HeII}]{\rm HeII}
+
k_6n_e{\rm HeIII}.
\]

There is therefore **no explicit \(1/{\rm HeII}\)** term in the residual.

This keeps the physical chemistry equation well behaved when HeII is initially extremely small.

---

## Electron-density treatment

In the semi-consistent PINN versions, the frozen quantities are the Grackle reaction coefficients \(k_1,\ldots,k_6\).

Electron density is reconstructed from the NN-predicted state:

\[
\rho_e
=
\rho_H\,{\rm HII}
+
\frac14\rho_{\rm He}\,{\rm HeII}
+
\frac12\rho_{\rm He}\,{\rm HeIII}.
\]

Therefore the chemistry residual is still state-dependent even though the \(k_i\) coefficients are frozen to the reference trajectory.

---

## Training stabilization in v4

v4 retains two stabilization mechanisms.

### Delayed PDE activation

Before `PDE_START_EPOCH`, the PDE term is not evaluated.

After activation, its weight is ramped linearly over `PDE_WEIGHT_WARMUP_EPOCHS`.

This allows the supervised value loss to first learn a reasonable trajectory shape before the PDE residual is introduced.

### Gradient clipping

After

```python
loss.backward()
```

v4 applies

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    MAX_GRAD_NORM,
)
```

before `optimizer.step()`.

The current default is

```python
MAX_GRAD_NORM = 1.0
```

This limits a single large PDE batch from producing an excessively large parameter update.

---

## Summary

| Version | Autograd derivative | Residual space | Main issue / motivation |
|---|---|---|---|
| v2 | Physical fraction \(dX/dt\) through denormalisation | Physical fraction | Unstable derivative through exponential inverse normalization; original in-place autograd issue |
| v3 | Normalized \(df/dt\) | Normalized log-fraction | Avoids physical derivative scaling, but introduces \(1/X\)-type terms; HeII RHS can explode |
| v4 | Normalized \(df/dt\), converted by chain rule to \(dX/dt\) | Physical fraction | Avoids both direct derivative through denormalisation and the normalized HeII \(1/{\rm HeII}\) singularity |

The key v4 principle is:

> **Differentiate the network in normalized space; enforce the chemistry equation in physical fraction space.**
