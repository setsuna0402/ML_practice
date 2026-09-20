"""
Diagnose ChemistryPINN PDE terms on the PRECOMPUTED TRAINING PDE grid
using physical-fraction residuals with normalized-output derivatives.

Method:
    Network output:
        f = A * log10(x) + C

    Autograd:
        df/dt

    Chain rule:
        dx/dt = (ln(10) / A) * x * df/dt

    Residual:
        physical derivative - physical chemistry RHS

Reports:
- NaN / Inf in derivative, RHS, residual
- global min / max / max-abs for HI, HeI, HeII

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


# ---------------------------------------------------------------------
# Manually set files here
# ---------------------------------------------------------------------
CHECKPOINT_FILE = Path("./chemistry_pinn_best_epoch_500.pt")
TRAINING_DATA_FILE = Path("./grackle_relaxed_training_data_merged_10000.h5")


# Must match training.
HIDDEN_DIM = 64
N_HIDDEN_LAYERS = 4
TIME_SCALE = 1.0

SPECIES_NAMES = ("HI", "HeI", "HeII")


# ---------------------------------------------------------------------
# Fraction normalization
#
# f = A log10(x) + C
# x = 10^((f-C)/A)
# ---------------------------------------------------------------------
LOG_FLOOR = np.log10(FRACTION_FLOOR)

A = (
    (FRACTION_NORM_MAX - FRACTION_NORM_MIN)
    / (0.0 - LOG_FLOOR)
)

C = (
    FRACTION_NORM_MIN
    - A * LOG_FLOOR
)

CHAIN_FACTOR = np.log(10.0) / A


def select_device():
    if ALLOW_DEVICE and torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")

    if ALLOW_DEVICE and torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")

    print("Using CPU")
    return torch.device("cpu")


def load_model(device, dtype):
    model = ChemistryPINN(
        hidden_dim=HIDDEN_DIM,
        n_hidden_layers=N_HIDDEN_LAYERS,
        time_scale=TIME_SCALE,
    ).to(device=device, dtype=dtype)

    state_dict = torch.load(
        CHECKPOINT_FILE,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def calculate_terms(model, X_pde_norm, PDE_aux, device, dtype):
    x = torch.as_tensor(
        X_pde_norm,
        dtype=dtype,
        device=device,
    ).detach().clone().requires_grad_(True)

    aux = torch.as_tensor(
        PDE_aux,
        dtype=dtype,
        device=device,
    )

    prediction = model(x)

    f_HI = prediction[:, 0]
    f_HeI = prediction[:, 1]
    f_HeII = prediction[:, 2]

    # -------------------------------------------------------------
    # Normalized derivatives df/dt
    # -------------------------------------------------------------
    grad_f_HI = torch.autograd.grad(
        f_HI.sum(),
        x,
        retain_graph=True,
    )[0][:, 0]

    grad_f_HeI = torch.autograd.grad(
        f_HeI.sum(),
        x,
        retain_graph=True,
    )[0][:, 0]

    grad_f_HeII = torch.autograd.grad(
        f_HeII.sum(),
        x,
    )[0][:, 0]

    ten = torch.tensor(
        10.0,
        dtype=dtype,
        device=device,
    )

    # -------------------------------------------------------------
    # Convert normalized outputs to physical fractions
    # -------------------------------------------------------------
    HI = torch.pow(
        ten,
        (f_HI - C) / A,
    )

    HeI = torch.pow(
        ten,
        (f_HeI - C) / A,
    )

    HeII = torch.pow(
        ten,
        (f_HeII - C) / A,
    )

    HII = 1.0 - HI
    HeIII = 1.0 - HeI - HeII

    # -------------------------------------------------------------
    # Chain rule:
    #
    # dx/dt = ln(10)/A * x * df/dt
    # -------------------------------------------------------------
    chain_factor = torch.tensor(
        CHAIN_FACTOR,
        dtype=dtype,
        device=device,
    )

    grad_HI = chain_factor * HI * grad_f_HI
    grad_HeI = chain_factor * HeI * grad_f_HeI
    grad_HeII = chain_factor * HeII * grad_f_HeII

    # -------------------------------------------------------------
    # Frozen reaction coefficients
    # -------------------------------------------------------------
    k1, k2, k3, k4, k5, k6 = [
        aux[:, i] for i in range(6)
    ]

    # -------------------------------------------------------------
    # Reconstruct Grackle electron density
    # -------------------------------------------------------------
    density = DENSITY_REF * torch.pow(
        ten,
        x[:, 1],
    )

    rho_H = HYDROGEN_FRACTION_BY_MASS * density
    rho_He = (1.0 - HYDROGEN_FRACTION_BY_MASS) * density

    e_nn = (
        rho_H * HII
        + 0.25 * rho_He * HeII
        + 0.50 * rho_He * HeIII
    )

    # -------------------------------------------------------------
    # Recover physical photoionisation rates
    # -------------------------------------------------------------
    Gamma_HI = GAMMA_HI_REF * torch.pow(
        ten,
        x[:, 6],
    )

    Gamma_HeI = GAMMA_HEI_REF * torch.pow(
        ten,
        x[:, 7],
    )

    Gamma_HeII = GAMMA_HEII_REF * torch.pow(
        ten,
        x[:, 8],
    )

    # -------------------------------------------------------------
    # Physical fraction RHS
    # -------------------------------------------------------------
    rhs_HI = (
        -(k1 * e_nn + Gamma_HI) * HI
        + k2 * e_nn * HII
    )

    rhs_HeI = (
        -(k3 * e_nn + Gamma_HeI) * HeI
        + k4 * e_nn * HeII
    )

    rhs_HeII = (
        (k3 * e_nn + Gamma_HeI) * HeI
        - ((k4 + k5) * e_nn + Gamma_HeII) * HeII
        + k6 * e_nn * HeIII
    )

    derivative = torch.stack(
        (grad_HI, grad_HeI, grad_HeII),
        dim=1,
    )

    rhs = torch.stack(
        (rhs_HI, rhs_HeI, rhs_HeII),
        dim=1,
    )

    residual = derivative - rhs

    return (
        derivative.detach().cpu().numpy(),
        rhs.detach().cpu().numpy(),
        residual.detach().cpu().numpy(),
    )


def report(name, array, times):
    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    for j, species in enumerate(SPECIES_NAMES):
        values = array[:, :, j]
        finite = np.isfinite(values)

        print(f"\n{species}")

        bad = np.argwhere(~finite)

        if len(bad) > 0:
            print(f"  NON-FINITE count = {len(bad)}")

            for traj, point in bad:
                print(
                    f"    trajectory={traj}, "
                    f"point={point}, "
                    f"t={times[traj, point]:.6e}, "
                    f"value={values[traj, point]}"
                )
        else:
            print("  No NaN / Inf")

        if not np.any(finite):
            print("  No finite values")
            continue

        safe = np.where(
            finite,
            values,
            np.nan,
        )

        imin = np.nanargmin(safe)
        imax = np.nanargmax(safe)
        iabs = np.nanargmax(np.abs(safe))

        min_traj, min_point = np.unravel_index(
            imin,
            values.shape,
        )

        max_traj, max_point = np.unravel_index(
            imax,
            values.shape,
        )

        abs_traj, abs_point = np.unravel_index(
            iabs,
            values.shape,
        )

        print(
            f"  min = {values[min_traj, min_point]: .6e} "
            f"(trajectory={min_traj}, "
            f"point={min_point}, "
            f"t={times[min_traj, min_point]:.6e})"
        )

        print(
            f"  max = {values[max_traj, max_point]: .6e} "
            f"(trajectory={max_traj}, "
            f"point={max_point}, "
            f"t={times[max_traj, max_point]:.6e})"
        )

        print(
            f"  max|.| = {abs(values[abs_traj, abs_point]): .6e} "
            f"(value={values[abs_traj, abs_point]: .6e}, "
            f"trajectory={abs_traj}, "
            f"point={abs_point}, "
            f"t={times[abs_traj, abs_point]:.6e})"
        )


def main():
    dtype = torch.float32
    device = select_device()

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}"
        )

    if not TRAINING_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAINING_DATA_FILE}"
        )

    X_value, _, X_pde, PDE_aux = load_precomputed_dataset(
        TRAINING_DATA_FILE,
        include_pde=True,
    )

    n_value_times = int(
        INTERPOLATION_KWARGS["n_times"]
    )

    if X_value.shape[0] % n_value_times != 0:
        raise ValueError(
            "Cannot infer number of trajectories from value grid."
        )

    n_trajectories = (
        X_value.shape[0] // n_value_times
    )

    if X_pde.shape[0] % n_trajectories != 0:
        raise ValueError(
            "Cannot infer number of PDE points per trajectory."
        )

    n_pde_points = (
        X_pde.shape[0] // n_trajectories
    )

    times = X_pde[:, 0].reshape(
        n_trajectories,
        n_pde_points,
    )

    X_pde_norm = normalise_inputs(X_pde)

    model = load_model(
        device,
        dtype,
    )

    derivative, rhs, residual = calculate_terms(
        model,
        X_pde_norm,
        PDE_aux,
        device,
        dtype,
    )

    derivative = derivative.reshape(
        n_trajectories,
        n_pde_points,
        3,
    )

    rhs = rhs.reshape(
        n_trajectories,
        n_pde_points,
        3,
    )

    residual = residual.reshape(
        n_trajectories,
        n_pde_points,
        3,
    )

    print()
    print(f"Device                : {device}")
    print(f"Training trajectories : {n_trajectories}")
    print(f"PDE points/trajectory : {n_pde_points}")
    print(f"A                     : {A}")
    print(f"C                     : {C}")
    print(f"ln(10)/A              : {CHAIN_FACTOR}")

    report(
        "PHYSICAL FRACTION DERIVATIVE dX/dt",
        derivative,
        times,
    )

    report(
        "PHYSICAL FRACTION RHS",
        rhs,
        times,
    )

    report(
        "PHYSICAL FRACTION RESIDUAL dX/dt - RHS",
        residual,
        times,
    )


if __name__ == "__main__":
    main()
