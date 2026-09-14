"""Normalisation utilities shared by training and inference."""
import numpy as np
import torch
from config import (
    DENSITY_REF, U_REF,
    GAMMA_HI_REF, GAMMA_HEI_REF, GAMMA_HEII_REF,
    PI_HI_REF, PI_HEI_REF, PI_HEII_REF,
    FRACTION_FLOOR, FRACTION_NORM_MIN, FRACTION_NORM_MAX,
)

# Input column order:
# time, density, HI0, HeI0, HeII0, u0,
# Gamma_HI, Gamma_HeI, Gamma_HeII, pi_HI, pi_HeI, pi_HeII
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
INPUT_FRACTION_COLUMNS = (2, 3, 4)

# Output column order: HI, HeI, HeII, u
OUTPUT_FRACTION_COLUMNS = (0, 1, 2)


def _copy(x):
    return x.clone() if torch.is_tensor(x) else np.array(x, copy=True)


def _log10(x):
    return torch.log10(x) if torch.is_tensor(x) else np.log10(x)


def _pow10(x):
    if torch.is_tensor(x):
        return torch.pow(torch.as_tensor(10.0, dtype=x.dtype, device=x.device), x)
    return np.power(10.0, x)


def _maximum(x, value):
    if torch.is_tensor(x):
        floor = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        return torch.maximum(x, floor)
    return np.maximum(x, value)


def _check_positive(x, column, name):
    values = x[..., column]
    bad = bool(torch.any(values <= 0).item()) if torch.is_tensor(values) else bool(np.any(values <= 0))
    if bad:
        raise ValueError(f"{name} must be > 0 before log10 normalisation.")


def _normalise_fraction(x):
    """Map physical fraction to [FRACTION_NORM_MIN, FRACTION_NORM_MAX].

    Values below FRACTION_FLOOR are treated as effectively zero.
    The mapping is logarithmic in physical fraction and affine in log-space:

        FRACTION_FLOOR -> FRACTION_NORM_MIN
        1.0            -> FRACTION_NORM_MAX
    """
    x = _maximum(x, FRACTION_FLOOR)
    log_x = _log10(x)
    log_floor = np.log10(FRACTION_FLOOR)

    scale = (FRACTION_NORM_MAX - FRACTION_NORM_MIN) / (0.0 - log_floor)
    return FRACTION_NORM_MIN + (log_x - log_floor) * scale


def _denormalise_fraction(x):
    """Invert :func:`_normalise_fraction` without clipping model outputs."""
    log_floor = np.log10(FRACTION_FLOOR)
    scale = (0.0 - log_floor) / (FRACTION_NORM_MAX - FRACTION_NORM_MIN)
    log_x = log_floor + (x - FRACTION_NORM_MIN) * scale
    return _pow10(log_x)


def normalise_inputs(x):
    """Normalise model inputs.

    - time is unchanged;
    - HI0, HeI0 and HeII0 use log+affine fraction scaling;
    - positive dimensional quantities use log10(q / q_ref).
    """
    y = _copy(x)

    for column in INPUT_FRACTION_COLUMNS:
        y[..., column] = _normalise_fraction(y[..., column])

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

    for column in INPUT_FRACTION_COLUMNS:
        y[..., column] = _denormalise_fraction(y[..., column])

    for column, ref in INPUT_REFS.items():
        y[..., column] = ref * _pow10(y[..., column])

    return y


def normalise_outputs(y):
    """Normalise [HI, HeI, HeII, u] model targets/outputs."""
    z = _copy(y)

    for column in OUTPUT_FRACTION_COLUMNS:
        z[..., column] = _normalise_fraction(z[..., column])

    _check_positive(z, 3, "u")
    z[..., 3] = _log10(z[..., 3] / U_REF)

    return z


def denormalise_outputs(y):
    """Invert :func:`normalise_outputs`."""
    z = _copy(y)

    for column in OUTPUT_FRACTION_COLUMNS:
        z[..., column] = _denormalise_fraction(z[..., column])

    z[..., 3] = U_REF * _pow10(z[..., 3])
    return z
