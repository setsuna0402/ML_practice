"""
Diagnose ChemistryPINN PDE terms on the precomputed training PDE grid.

Manually set:
    CHECKPOINT_FILE
    TRAINING_DATA_FILE

Reports:
- NaN / Inf in normalized derivatives, RHS, residuals
- global min / max / max-abs for HI, HeI, HeII
- global min / max / max-abs for each HeII RHS term

Nothing is saved.
"""

from pathlib import Path

import numpy as np
import torch

from config import (
    ALLOW_DEVICE,
    DENSITY_REF,
    HYDROGEN_FRACTION_BY_MASS,
    GAMMA_HI_REF,
    GAMMA_HEI_REF,
    GAMMA_HEII_REF,
    FRACTION_FLOOR,
    FRACTION_NORM_MIN,
    FRACTION_NORM_MAX,
    INTERPOLATION_KWARGS,
)
from grackle_data import load_precomputed_dataset
from normalisation import normalise_inputs
from pinn_model import ChemistryPINN


CHECKPOINT_FILE = Path("./chemistry_pinn_best_epoch_500.pt")
TRAINING_DATA_FILE = Path("./grackle_relaxed_training_data_merged_10000.h5")

HIDDEN_DIM = 64
N_HIDDEN_LAYERS = 4
TIME_SCALE = 1.0

SPECIES_NAMES = ("HI", "HeI", "HeII")
HEII_TERM_NAMES = (
    "term1: HeI -> HeII",
    "term2: HeII collisional loss",
    "term3: HeII photoionisation",
    "term4: HeIII -> HeII recombination",
)

log_floor = np.log10(FRACTION_FLOOR)
A = (FRACTION_NORM_MAX - FRACTION_NORM_MIN) / (0.0 - log_floor)
C = FRACTION_NORM_MIN - A * log_floor
S = 1.0 / A
FACTOR = A / np.log(10.0)


def select_device():
    if ALLOW_DEVICE and torch.cuda.is_available():
        return torch.device("cuda")
    if ALLOW_DEVICE and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device, dtype):
    model = ChemistryPINN(
        hidden_dim=HIDDEN_DIM,
        n_hidden_layers=N_HIDDEN_LAYERS,
        time_scale=TIME_SCALE,
    ).to(device=device, dtype=dtype)

    state = torch.load(
        CHECKPOINT_FILE,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def calculate_terms(model, X_pde_norm, PDE_aux, device, dtype):
    x = torch.as_tensor(
        X_pde_norm, dtype=dtype, device=device
    ).detach().clone().requires_grad_(True)

    aux = torch.as_tensor(
        PDE_aux, dtype=dtype, device=device
    )

    pred = model(x)

    f_HI = pred[:, 0]
    f_HeI = pred[:, 1]
    f_HeII = pred[:, 2]

    grad_HI = torch.autograd.grad(
        f_HI.sum(), x, retain_graph=True
    )[0][:, 0]

    grad_HeI = torch.autograd.grad(
        f_HeI.sum(), x, retain_graph=True
    )[0][:, 0]

    grad_HeII = torch.autograd.grad(
        f_HeII.sum(), x
    )[0][:, 0]

    k1, k2, k3, k4, k5, k6 = [aux[:, i] for i in range(6)]

    ten = torch.tensor(10.0, dtype=dtype, device=device)
    factor = torch.tensor(FACTOR, dtype=dtype, device=device)

    HI = torch.pow(ten, (f_HI - C) * S)
    HeI = torch.pow(ten, (f_HeI - C) * S)
    HeII = torch.pow(ten, (f_HeII - C) * S)

    HII = 1.0 - HI
    HeIII = 1.0 - HeI - HeII

    density = DENSITY_REF * torch.pow(ten, x[:, 1])
    rho_H = HYDROGEN_FRACTION_BY_MASS * density
    rho_He = (1.0 - HYDROGEN_FRACTION_BY_MASS) * density

    e_nn = (
        rho_H * HII
        + 0.25 * rho_He * HeII
        + 0.50 * rho_He * HeIII
    )

    Gamma_HI = GAMMA_HI_REF * torch.pow(ten, x[:, 6])
    Gamma_HeI = GAMMA_HEI_REF * torch.pow(ten, x[:, 7])
    Gamma_HeII = GAMMA_HEII_REF * torch.pow(ten, x[:, 8])

    rhs_HI = factor * (
        -(k1 + k2) * e_nn
        - Gamma_HI
        + k2 * e_nn * torch.pow(ten, (C - f_HI) * S)
    )

    ratio_HeII_HeI = torch.pow(ten, (f_HeII - f_HeI) * S)

    rhs_HeI = factor * (
        -k3 * e_nn
        - Gamma_HeI
        + k4 * e_nn * ratio_HeII_HeI
    )

    ratio_HeI_HeII = torch.pow(ten, (f_HeI - f_HeII) * S)
    inv_HeII = torch.pow(ten, (C - f_HeII) * S)
    ratio_HeIII_HeII = inv_HeII - ratio_HeI_HeII - 1.0

    term1 = factor * ((k3 * e_nn + Gamma_HeI) * ratio_HeI_HeII)
    term2 = factor * (-(k4 + k5) * e_nn)
    term3 = factor * (-Gamma_HeII)
    term4 = factor * (k6 * e_nn * ratio_HeIII_HeII)

    rhs_HeII = term1 + term2 + term3 + term4

    grad = torch.stack((grad_HI, grad_HeI, grad_HeII), dim=1)
    rhs = torch.stack((rhs_HI, rhs_HeI, rhs_HeII), dim=1)
    residual = grad - rhs
    heii_terms = torch.stack((term1, term2, term3, term4), dim=1)

    return (
        grad.detach().cpu().numpy(),
        rhs.detach().cpu().numpy(),
        residual.detach().cpu().numpy(),
        heii_terms.detach().cpu().numpy(),
    )


def report(name, arr, times, labels):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    for j, label in enumerate(labels):
        values = arr[:, :, j]
        finite = np.isfinite(values)
        bad = np.argwhere(~finite)

        print(f"\n{label}")

        if len(bad):
            print(f"  NON-FINITE count = {len(bad)}")
            for traj, point in bad:
                print(
                    f"    trajectory={traj}, point={point}, "
                    f"t={times[traj, point]:.6e}, "
                    f"value={values[traj, point]}"
                )
        else:
            print("  No NaN / Inf")

        if not np.any(finite):
            print("  No finite values")
            continue

        safe = np.where(finite, values, np.nan)

        imin = np.nanargmin(safe)
        imax = np.nanargmax(safe)
        iabs = np.nanargmax(np.abs(safe))

        min_traj, min_point = np.unravel_index(imin, values.shape)
        max_traj, max_point = np.unravel_index(imax, values.shape)
        abs_traj, abs_point = np.unravel_index(iabs, values.shape)

        print(
            f"  min = {values[min_traj, min_point]: .6e} "
            f"(trajectory={min_traj}, point={min_point}, "
            f"t={times[min_traj, min_point]:.6e})"
        )
        print(
            f"  max = {values[max_traj, max_point]: .6e} "
            f"(trajectory={max_traj}, point={max_point}, "
            f"t={times[max_traj, max_point]:.6e})"
        )
        print(
            f"  max|.| = {abs(values[abs_traj, abs_point]): .6e} "
            f"(value={values[abs_traj, abs_point]: .6e}, "
            f"trajectory={abs_traj}, point={abs_point}, "
            f"t={times[abs_traj, abs_point]:.6e})"
        )


def main():
    dtype = torch.float32
    device = select_device()

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(CHECKPOINT_FILE)
    if not TRAINING_DATA_FILE.exists():
        raise FileNotFoundError(TRAINING_DATA_FILE)

    X_value, _, X_pde, PDE_aux = load_precomputed_dataset(
        TRAINING_DATA_FILE,
        include_pde=True,
    )

    n_value_times = int(INTERPOLATION_KWARGS["n_times"])
    n_traj = X_value.shape[0] // n_value_times
    n_pde = X_pde.shape[0] // n_traj

    times = X_pde[:, 0].reshape(n_traj, n_pde)
    X_pde_norm = normalise_inputs(X_pde)

    model = load_model(device, dtype)

    grad, rhs, residual, heii_terms = calculate_terms(
        model,
        X_pde_norm,
        PDE_aux,
        device,
        dtype,
    )

    grad = grad.reshape(n_traj, n_pde, 3)
    rhs = rhs.reshape(n_traj, n_pde, 3)
    residual = residual.reshape(n_traj, n_pde, 3)
    heii_terms = heii_terms.reshape(n_traj, n_pde, 4)

    print(f"Device: {device}")
    print(f"Trajectories: {n_traj}")
    print(f"PDE points per trajectory: {n_pde}")
    print(f"A={A}, C={C}, S={S}, A/ln10={FACTOR}")

    report("NORMALIZED DERIVATIVE df/dt", grad, times, SPECIES_NAMES)
    report("NORMALIZED RHS", rhs, times, SPECIES_NAMES)
    report("RAW RESIDUAL df/dt - RHS", residual, times, SPECIES_NAMES)
    report("HeII RHS INDIVIDUAL TERMS", heii_terms, times, HEII_TERM_NAMES)


if __name__ == "__main__":
    main()
