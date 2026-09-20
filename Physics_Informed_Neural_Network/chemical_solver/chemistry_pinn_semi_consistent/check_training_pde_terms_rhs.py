"""
Diagnose ChemistryPINN PDE terms on the PRECOMPUTED TRAINING PDE grid.

Manually set:
    CHECKPOINT_FILE
    TRAINING_DATA_FILE

For every PDE midpoint point, this script computes in normalized fraction space:

    grad_f = d f / dt
    rhs_f
    residual = grad_f - rhs_f

for HI, HeI and HeII.

It reports:
- every NaN / Inf found in gradient, RHS, or residual;
- global min/max of gradient, RHS, and residual;
- trajectory index, PDE-point index, and time for each global extreme.

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
# Fraction-normalisation constants
#
# f = A * log10(x) + C
# x = 10 ** ((f - C) / A)
# ---------------------------------------------------------------------
_LOG_FLOOR = np.log10(FRACTION_FLOOR)

FRACTION_A = (
    (FRACTION_NORM_MAX - FRACTION_NORM_MIN)
    / (0.0 - _LOG_FLOOR)
)

FRACTION_C = (
    FRACTION_NORM_MIN
    - FRACTION_A * _LOG_FLOOR
)

FRACTION_S = 1.0 / FRACTION_A

DF_DLOG_FACTOR = FRACTION_A / np.log(10.0)


def select_device():
    if ALLOW_DEVICE and torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")

    if ALLOW_DEVICE and torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")

    print("Using CPU")
    return torch.device("cpu")


def load_model(checkpoint_file, device, dtype):
    model = ChemistryPINN(
        hidden_dim=HIDDEN_DIM,
        n_hidden_layers=N_HIDDEN_LAYERS,
        time_scale=TIME_SCALE,
    ).to(device=device, dtype=dtype)

    state_dict = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def calculate_pde_terms(model, X_pde_norm, PDE_aux, device, dtype):
    """
    Return:
        grad_f     shape (N, 3)
        rhs_f      shape (N, 3)
        residual   shape (N, 3)

    All three are in normalized-fraction / code-time units.
    """
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
    # d(normalized fraction)/dt
    # -------------------------------------------------------------
    grad_f_HI = torch.autograd.grad(
        f_HI.sum(),
        x,
        create_graph=False,
        retain_graph=True,
    )[0][:, 0]

    grad_f_HeI = torch.autograd.grad(
        f_HeI.sum(),
        x,
        create_graph=False,
        retain_graph=True,
    )[0][:, 0]

    grad_f_HeII = torch.autograd.grad(
        f_HeII.sum(),
        x,
        create_graph=False,
        retain_graph=False,
    )[0][:, 0]

    # -------------------------------------------------------------
    # Frozen Grackle coefficients
    # -------------------------------------------------------------
    k1, k2, k3, k4, k5, k6 = [
        aux[:, i] for i in range(6)
    ]

    ten = torch.as_tensor(
        10.0,
        dtype=x.dtype,
        device=x.device,
    )

    # -------------------------------------------------------------
    # Recover physical species fractions ONLY for e_density.
    #
    # x_species = 10^((f - C) / A)
    # -------------------------------------------------------------
    HI = torch.pow(
        ten,
        (f_HI - FRACTION_C) * FRACTION_S,
    )

    HeI = torch.pow(
        ten,
        (f_HeI - FRACTION_C) * FRACTION_S,
    )

    HeII = torch.pow(
        ten,
        (f_HeII - FRACTION_C) * FRACTION_S,
    )

    HII = 1.0 - HI
    HeIII = 1.0 - HeI - HeII

    # -------------------------------------------------------------
    # Reconstruct Grackle e_density.
    #
    # Input column 1 is normalized density:
    # density = DENSITY_REF * 10**x[:,1]
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
    # Recover physical photoionisation rates from normalized inputs.
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

    factor = torch.as_tensor(
        DF_DLOG_FACTOR,
        dtype=x.dtype,
        device=x.device,
    )

    # -------------------------------------------------------------
    # Normalized-space chemistry RHS.
    #
    # HI:
    # df_HI/dt = A/ln(10) [
    #   -(k1+k2) ne - Gamma_HI
    #   + k2 ne 10^((C-f_HI)/A)
    # ]
    # -------------------------------------------------------------
    rhs_f_HI = factor * (
        -(k1 + k2) * e_nn
        - Gamma_HI
        + k2 * e_nn * torch.pow(
            ten,
            (FRACTION_C - f_HI) * FRACTION_S,
        )
    )

    # -------------------------------------------------------------
    # HeI:
    # df_HeI/dt = A/ln(10) [
    #   -k3 ne - Gamma_HeI
    #   + k4 ne 10^((f_HeII-f_HeI)/A)
    # ]
    # -------------------------------------------------------------
    ratio_HeII_HeI = torch.pow(
        ten,
        (f_HeII - f_HeI) * FRACTION_S,
    )

    rhs_f_HeI = factor * (
        -k3 * e_nn
        - Gamma_HeI
        + k4 * e_nn * ratio_HeII_HeI
    )

    # -------------------------------------------------------------
    # HeII:
    #
    # HeI/HeII = 10^((f_HeI-f_HeII)/A)
    #
    # HeIII/HeII =
    #   10^((C-f_HeII)/A)
    #   - 10^((f_HeI-f_HeII)/A)
    #   - 1
    # -------------------------------------------------------------
    ratio_HeI_HeII = torch.pow(
        ten,
        (f_HeI - f_HeII) * FRACTION_S,
    )

    inv_HeII = torch.pow(
        ten,
        (FRACTION_C - f_HeII) * FRACTION_S,
    )

    ratio_HeIII_HeII = (
        inv_HeII
        - ratio_HeI_HeII
        - 1.0
    )

    rhs_f_HeII = factor * (
        (k3 * e_nn + Gamma_HeI) * ratio_HeI_HeII
        - (k4 + k5) * e_nn
        - Gamma_HeII
        + k6 * e_nn * ratio_HeIII_HeII
    )

    grad_f = torch.stack(
        (grad_f_HI, grad_f_HeI, grad_f_HeII),
        dim=1,
    )

    rhs_f = torch.stack(
        (rhs_f_HI, rhs_f_HeI, rhs_f_HeII),
        dim=1,
    )

    residual = grad_f - rhs_f

    return (
        grad_f.detach().cpu().numpy(),
        rhs_f.detach().cpu().numpy(),
        residual.detach().cpu().numpy(),
    )


def report_quantity(name, array, times):
    """
    array shape:
        (n_trajectories, n_pde_points, 3)
    """
    n_trajectories, _, n_species = array.shape

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    bad_records = []

    global_min = np.full(n_species, np.inf)
    global_max = np.full(n_species, -np.inf)

    min_location = [None] * n_species
    max_location = [None] * n_species

    for traj in range(n_trajectories):
        for s, species in enumerate(SPECIES_NAMES):
            values = array[traj, :, s]

            finite = np.isfinite(values)

            for point_index in np.where(~finite)[0]:
                bad_records.append(
                    (
                        traj,
                        species,
                        int(point_index),
                        float(times[traj, point_index]),
                        values[point_index],
                    )
                )

            finite_indices = np.where(finite)[0]

            if finite_indices.size == 0:
                continue

            finite_values = values[finite_indices]

            local_min_pos = int(np.argmin(finite_values))
            local_max_pos = int(np.argmax(finite_values))

            imin = int(finite_indices[local_min_pos])
            imax = int(finite_indices[local_max_pos])

            vmin = float(values[imin])
            vmax = float(values[imax])

            if vmin < global_min[s]:
                global_min[s] = vmin
                min_location[s] = (
                    traj,
                    imin,
                    float(times[traj, imin]),
                )

            if vmax > global_max[s]:
                global_max[s] = vmax
                max_location[s] = (
                    traj,
                    imax,
                    float(times[traj, imax]),
                )

    if bad_records:
        print("\nNaN / Inf found:")

        for traj, species, index, time_value, value in bad_records:
            print(
                f"  trajectory={traj:6d}, "
                f"species={species:4s}, "
                f"point={index:3d}, "
                f"t={time_value:.6e}, "
                f"value={value}"
            )

        affected = sorted(
            set(record[0] for record in bad_records)
        )

        print()
        print(
            f"Total non-finite values: {len(bad_records)}"
        )
        print(
            f"Affected trajectories: {len(affected)}"
        )
        print(affected)

    else:
        print("\nNo NaN / Inf found.")

    print("\nGlobal finite range:")

    for s, species in enumerate(SPECIES_NAMES):
        if min_location[s] is None:
            print(
                f"  {species}: no finite values"
            )
            continue

        min_traj, min_index, min_time = min_location[s]
        max_traj, max_index, max_time = max_location[s]

        print(
            f"\n  {species}:"
            f"\n    min = {global_min[s]: .6e}"
            f"  (trajectory={min_traj}, point={min_index}, t={min_time:.6e})"
            f"\n    max = {global_max[s]: .6e}"
            f"  (trajectory={max_traj}, point={max_index}, t={max_time:.6e})"
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

    # Need both the value grid and the PDE midpoint grid so that the
    # number of trajectories can be reconstructed robustly.
    X_value, _, X_pde, PDE_aux = load_precomputed_dataset(
        TRAINING_DATA_FILE,
        include_pde=True,
    )

    n_value_times = int(
        INTERPOLATION_KWARGS["n_times"]
    )

    if X_value.shape[0] % n_value_times != 0:
        raise ValueError(
            f"Value-grid rows ({X_value.shape[0]}) are not divisible "
            f"by n_times={n_value_times}."
        )

    n_trajectories = (
        X_value.shape[0] // n_value_times
    )

    if X_pde.shape[0] % n_trajectories != 0:
        raise ValueError(
            f"PDE rows ({X_pde.shape[0]}) are not divisible by "
            f"n_trajectories={n_trajectories}."
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
        CHECKPOINT_FILE,
        device,
        dtype,
    )

    grad_f, rhs_f, residual = calculate_pde_terms(
        model,
        X_pde_norm,
        PDE_aux,
        device,
        dtype,
    )

    grad_f = grad_f.reshape(
        n_trajectories,
        n_pde_points,
        3,
    )

    rhs_f = rhs_f.reshape(
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
    print(f"Training trajectories : {n_trajectories}")
    print(f"PDE points/trajectory : {n_pde_points}")
    print(f"Total PDE points       : {X_pde.shape[0]}")

    print()
    print("Fraction normalization:")
    print(f"  A = {FRACTION_A:.8g}")
    print(f"  C = {FRACTION_C:.8g}")
    print(f"  S = {FRACTION_S:.8g}")
    print(f"  A/ln(10) = {DF_DLOG_FACTOR:.8g}")

    report_quantity(
        "NORMALIZED DERIVATIVE  df/dt",
        grad_f,
        times,
    )

    report_quantity(
        "NORMALIZED RHS",
        rhs_f,
        times,
    )

    report_quantity(
        "RAW RESIDUAL  df/dt - RHS",
        residual,
        times,
    )


if __name__ == "__main__":
    main()
