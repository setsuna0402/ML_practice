"""Normalization utilities for the primordial chemistry PINN.

The dimensional inputs are scaled by a reference value and transformed with
log10::

    q_norm = log10(q / q_ref)

Ion fractions are already O(1) and are left unchanged. Time is also left
unchanged for now; the existing model ``time_scale`` continues to control the
hard initial-condition factor.

Input column order must match pinn_model.INPUT_NAMES:
    time, density, HI0, HeI0, HeII0, u0,
    Gamma_HI, Gamma_HeI, Gamma_HeII,
    pi_HI, pi_HeI, pi_HeII

Output column order must match pinn_model.OUTPUT_NAMES:
    HI, HeI, HeII, u
"""

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Reference values
# -----------------------------------------------------------------------------
# density is already expressed relative to the chosen Grackle density unit.
DENSITY_REF = 1.0

# Specific internal energy after conversion by (code_length / code_time)^2.
U_REF = 3.10289869e12

# RT ionisation rates in 1 / Myr (Grackle code-time units).
GAMMA_HI_REF = 25.8376836
GAMMA_HEI_REF = 14.9876104
GAMMA_HEII_REF = 0.11397924

# Species-specific photoheating rates, in the same units used by the current
# grackle_data.py inputs (eV / s / absorbing atom).
PI_HI_REF = 3.2556039206009867e-12
PI_HEI_REF = 3.387724795493442e-12
PI_HEII_REF = 7.789881261416638e-14


INPUT_REFS = {
    1: DENSITY_REF,
    5: U_REF,
    6: GAMMA_HI_REF,
    7: GAMMA_HEI_REF,
    8: GAMMA_HEII_REF,
    9: PI_HI_REF,
    10: PI_HEI_REF,
    11: PI_HEII_REF,
}

OUTPUT_REFS = {
    3: U_REF,
}


# -----------------------------------------------------------------------------
# Backend helpers: support both NumPy arrays and torch tensors.
# -----------------------------------------------------------------------------
def _copy(x):
    if torch.is_tensor(x):
        return x.clone()
    return np.array(x, copy=True)


def _log10(x):
    if torch.is_tensor(x):
        return torch.log10(x)
    return np.log10(x)


def _pow10(x):
    if torch.is_tensor(x):
        return torch.pow(torch.as_tensor(10.0, dtype=x.dtype, device=x.device), x)
    return np.power(10.0, x)


def _check_positive(x, column, name):
    values = x[..., column]
    if torch.is_tensor(values):
        bad = bool(torch.any(values <= 0).item())
    else:
        bad = bool(np.any(values <= 0))
    if bad:
        raise ValueError(f"{name} must be > 0 before log10 normalization.")


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
def normalise_inputs(x):
    """Return normalized PINN inputs.

    Dimensional positive quantities use log10(q / q_ref).
    Time and ion fractions are unchanged.
    """
    y = _copy(x)

    names = {
        1: "density",
        5: "u0",
        6: "Gamma_HI",
        7: "Gamma_HeI",
        8: "Gamma_HeII",
        9: "pi_HI",
        10: "pi_HeI",
        11: "pi_HeII",
    }

    for column, ref in INPUT_REFS.items():
        _check_positive(y, column, names[column])
        y[..., column] = _log10(y[..., column] / ref)

    return y


def denormalise_inputs(x):
    """Invert :func:`normalise_inputs`."""
    y = _copy(x)

    for column, ref in INPUT_REFS.items():
        y[..., column] = ref * _pow10(y[..., column])

    return y


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
def normalise_outputs(y):
    """Return normalized model targets/outputs.

    HI, HeI and HeII fractions are unchanged. Specific internal energy uses
    log10(u / U_REF).
    """
    z = _copy(y)
    _check_positive(z, 3, "u")
    z[..., 3] = _log10(z[..., 3] / U_REF)
    return z


def denormalise_outputs(y):
    """Invert :func:`normalise_outputs`."""
    z = _copy(y)
    z[..., 3] = U_REF * _pow10(z[..., 3])
    return z
