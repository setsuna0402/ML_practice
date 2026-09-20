#!/usr/bin/env python3
"""
Calculate both value residuals and v4 PDE residuals for a precomputed HDF5 dataset.

Value residual:
    normalized PINN prediction - normalized Grackle target

PDE residual (v4):
    physical-fraction derivative - physical chemistry RHS

The PDE derivative is computed exactly as in train_pinn_v4.py:
1. autograd calculates df_norm/dt;
2. chain rule converts it to dX/dt;
3. the physical chemistry RHS is evaluated with frozen k1..k6.

This intentionally uses the current three torch.autograd.grad calls rather than JVP,
so it diagnoses the same residual formulation used by v4 training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    VALIDATION_DATA_FILE,
    DENSITY_REF,
    HYDROGEN_FRACTION_BY_MASS,
    GAMMA_HI_REF,
    GAMMA_HEI_REF,
    GAMMA_HEII_REF,
    FRACTION_FLOOR,
    FRACTION_NORM_MIN,
    FRACTION_NORM_MAX,
)

from grackle_data import load_precomputed_dataset
from normalisation import normalise_inputs, normalise_outputs
from pinn_model import ChemistryPINN


MODEL_HIDDEN_DIM = 64
MODEL_N_HIDDEN_LAYERS = 4
MODEL_TIME_SCALE = 1.0

VALUE_NAMES = ("HI", "HeI", "HeII", "u")
PDE_NAMES = ("HI", "HeI", "HeII")


# ---------------------------------------------------------------------
# Fraction normalization
#
# f = A log10(X) + C
# X = 10^((f-C)/A)
# dX/dt = ln(10)/A * X * df/dt
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

CHAIN_FACTOR = np.log(10.0) / FRACTION_A


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate normalized value residuals and v4 physical PDE residuals "
            "from a precomputed ChemistryPINN HDF5 dataset."
        )
    )

    parser.add_argument(
        "-I",
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained .pt/.pth checkpoint.",
    )

    parser.add_argument(
        "-D",
        "--data",
        type=Path,
        default=VALIDATION_DATA_FILE,
        help="Precomputed HDF5 dataset.",
    )

    parser.add_argument(
        "-O",
        "--output-dir",
        type=Path,
        default=Path("./trained_results_v4_a/residual_results_epoch_9960"),
        help="Output directory.",
    )

    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")

    print("Using CPU")
    return torch.device("cpu")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> ChemistryPINN:
    model = ChemistryPINN(
        hidden_dim=MODEL_HIDDEN_DIM,
        n_hidden_layers=MODEL_N_HIDDEN_LAYERS,
        time_scale=MODEL_TIME_SCALE,
    ).to(device)

    state = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    model.load_state_dict(state_dict)
    model.eval()

    return model


def calculate_statistics(
    residuals: np.ndarray,
) -> dict[str, np.ndarray]:
    abs_residuals = np.abs(residuals)

    return {
        "mean": np.nanmean(residuals, axis=0),
        "std": np.nanstd(residuals, axis=0),
        "max": np.nanmax(residuals, axis=0),
        "min": np.nanmin(residuals, axis=0),
        "mean_abs": np.nanmean(abs_residuals, axis=0),
        "std_abs": np.nanstd(abs_residuals, axis=0),
        "max_abs": np.nanmax(abs_residuals, axis=0),
        "min_abs": np.nanmin(abs_residuals, axis=0),
    }


def print_statistics(
    title: str,
    residuals: np.ndarray,
    names: tuple[str, ...],
) -> None:
    stats = calculate_statistics(residuals)

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    for i, name in enumerate(names):
        values = residuals[:, i]
        finite = np.isfinite(values)

        print(f"\n{name}")
        print(f"  finite:                      {np.count_nonzero(finite)}/{len(values)}")
        print(f"  NaN:                         {np.count_nonzero(np.isnan(values))}")
        print(f"  Inf:                         {np.count_nonzero(np.isinf(values))}")
        print(f"  mean residual:               {stats['mean'][i]:.8e}")
        print(f"  standard deviation:          {stats['std'][i]:.8e}")
        print(f"  max residual:                {stats['max'][i]:.8e}")
        print(f"  min residual:                {stats['min'][i]:.8e}")
        print(f"  mean absolute residual:      {stats['mean_abs'][i]:.8e}")
        print(f"  std absolute residual:       {stats['std_abs'][i]:.8e}")
        print(f"  max absolute residual:       {stats['max_abs'][i]:.8e}")
        print(f"  min absolute residual:       {stats['min_abs'][i]:.8e}")


def save_statistics_text(
    output_file: Path,
    checkpoint_label: str,
    value_residuals: np.ndarray,
    pde_residuals: np.ndarray,
) -> None:
    value_stats = calculate_statistics(value_residuals)
    pde_stats = calculate_statistics(pde_residuals)

    with output_file.open("w") as f:
        f.write(f"Model name: {checkpoint_label}\n\n")

        f.write("VALUE RESIDUAL\n")
        f.write("Definition: normalized PINN - normalized Grackle\n\n")

        for i, name in enumerate(VALUE_NAMES):
            values = value_residuals[:, i]
            f.write(f"{name}\n")
            f.write(f"  NaN count:                    {np.count_nonzero(np.isnan(values))}\n")
            f.write(f"  Inf count:                    {np.count_nonzero(np.isinf(values))}\n")
            f.write(f"  Mean residual:                {value_stats['mean'][i]:.8e}\n")
            f.write(f"  Standard deviation:           {value_stats['std'][i]:.8e}\n")
            f.write(f"  Max residual:                 {value_stats['max'][i]:.8e}\n")
            f.write(f"  Min residual:                 {value_stats['min'][i]:.8e}\n")
            f.write(f"  Mean absolute residual:       {value_stats['mean_abs'][i]:.8e}\n")
            f.write(f"  Max absolute residual:        {value_stats['max_abs'][i]:.8e}\n\n")

        f.write("\nPDE RESIDUAL\n")
        f.write("Definition: physical dX/dt - physical chemistry RHS\n")
        f.write("Derivative: df_norm/dt converted by chain rule\n\n")

        for i, name in enumerate(PDE_NAMES):
            values = pde_residuals[:, i]
            f.write(f"{name}\n")
            f.write(f"  NaN count:                    {np.count_nonzero(np.isnan(values))}\n")
            f.write(f"  Inf count:                    {np.count_nonzero(np.isinf(values))}\n")
            f.write(f"  Mean residual:                {pde_stats['mean'][i]:.8e}\n")
            f.write(f"  Standard deviation:           {pde_stats['std'][i]:.8e}\n")
            f.write(f"  Max residual:                 {pde_stats['max'][i]:.8e}\n")
            f.write(f"  Min residual:                 {pde_stats['min'][i]:.8e}\n")
            f.write(f"  Mean absolute residual:       {pde_stats['mean_abs'][i]:.8e}\n")
            f.write(f"  Max absolute residual:        {pde_stats['max_abs'][i]:.8e}\n\n")


def save_histograms(
    output_dir: Path,
    residuals: np.ndarray,
    names: tuple[str, ...],
    prefix: str,
) -> None:
    for i, name in enumerate(names):
        values = np.abs(residuals[:, i])
        values = values[np.isfinite(values)]

        fig, ax = plt.subplots(figsize=(9, 5.5))

        if len(values) > 0:
            ax.hist(values, bins=60, density=True)

        ax.set_xlabel(f"Absolute residual ({name})")
        ax.set_ylabel("Probability density")
        ax.set_title(f"{prefix}: {name}")
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            output_dir / f"{prefix}_{name}.png",
            dpi=180,
        )
        plt.close(fig)


def calculate_value_residuals(
    model: ChemistryPINN,
    X_value: np.ndarray,
    Y_value: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    X_norm = normalise_inputs(X_value)
    Y_norm = normalise_outputs(Y_value)

    X_tensor = torch.as_tensor(
        X_norm,
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        prediction_norm = model(X_tensor).cpu().numpy()

    return prediction_norm - Y_norm


def calculate_pde_residuals(
    model: ChemistryPINN,
    X_pde: np.ndarray,
    PDE_aux: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        derivative_phys : (N, 3)
        rhs_phys        : (N, 3)
        residual_phys   : (N, 3)
    """
    X_pde_norm = normalise_inputs(X_pde)

    x = torch.as_tensor(
        X_pde_norm,
        dtype=torch.float32,
        device=device,
    ).detach().clone().requires_grad_(True)

    aux = torch.as_tensor(
        PDE_aux,
        dtype=torch.float32,
        device=device,
    )

    prediction = model(x)

    f_HI = prediction[:, 0]
    f_HeI = prediction[:, 1]
    f_HeII = prediction[:, 2]

    # Exactly the current v4 derivative method.
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

    ten = torch.as_tensor(
        10.0,
        dtype=x.dtype,
        device=x.device,
    )

    HI = torch.pow(
        ten,
        (f_HI - FRACTION_C) / FRACTION_A,
    )

    HeI = torch.pow(
        ten,
        (f_HeI - FRACTION_C) / FRACTION_A,
    )

    HeII = torch.pow(
        ten,
        (f_HeII - FRACTION_C) / FRACTION_A,
    )

    HII = 1.0 - HI
    HeIII = 1.0 - HeI - HeII

    chain_factor = torch.as_tensor(
        CHAIN_FACTOR,
        dtype=x.dtype,
        device=x.device,
    )

    grad_HI = chain_factor * HI * grad_f_HI
    grad_HeI = chain_factor * HeI * grad_f_HeI
    grad_HeII = chain_factor * HeII * grad_f_HeII

    k1, k2, k3, k4, k5, k6 = [
        aux[:, i] for i in range(6)
    ]

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



def save_trajectory_residual_plots(
    output_dir: Path,
    X_value: np.ndarray,
    X_pde: np.ndarray,
    value_residuals: np.ndarray,
    pde_residuals: np.ndarray,
) -> None:
    """Save one 2x4 residual figure for every trajectory.

    Top row:
        HI, HeI, HeII, u value residuals.

    Bottom row:
        HI, HeI, HeII PDE residuals.
        The final panel is intentionally unused because no energy PDE
        residual is included at this stage.
    """
    n_value_total = X_value.shape[0]
    n_pde_total = X_pde.shape[0]

    candidates = []

    for n_traj in range(1, min(n_value_total, n_pde_total) + 1):
        if n_value_total % n_traj != 0:
            continue
        if n_pde_total % n_traj != 0:
            continue

        n_value_per_traj = n_value_total // n_traj
        n_pde_per_traj = n_pde_total // n_traj

        if n_value_per_traj == n_pde_per_traj + 1:
            candidates.append(
                (n_traj, n_value_per_traj, n_pde_per_traj)
            )

    if not candidates:
        raise ValueError(
            "Could not infer trajectory structure from value/PDE array sizes."
        )

    n_trajectories, n_value_times, n_pde_times = candidates[-1]

    value_time = X_value[:, 0].reshape(
        n_trajectories,
        n_value_times,
    )

    pde_time = X_pde[:, 0].reshape(
        n_trajectories,
        n_pde_times,
    )

    value_res = value_residuals.reshape(
        n_trajectories,
        n_value_times,
        len(VALUE_NAMES),
    )

    pde_res = pde_residuals.reshape(
        n_trajectories,
        n_pde_times,
        len(PDE_NAMES),
    )

    trajectory_dir = output_dir / "trajectory_residuals"
    trajectory_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Saving {n_trajectories} trajectory residual plots "
        f"to {trajectory_dir}"
    )

    for traj in range(n_trajectories):
        fig, axes = plt.subplots(
            2,
            4,
            figsize=(18, 8),
            sharex=False,
            squeeze=False,
        )

        # Top row: value residuals
        for feature_index, feature_name in enumerate(VALUE_NAMES):
            ax = axes[0, feature_index]

            ax.plot(
                value_time[traj],
                value_res[traj, :, feature_index],
                linewidth=1.2,
            )

            ax.axhline(
                0.0,
                linewidth=0.8,
                linestyle="--",
            )

            ax.set_title(
                f"{feature_name}: value residual"
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("PINN - Grackle")
            ax.grid(True, alpha=0.25)

        # Bottom row: species PDE residuals
        for species_index, species_name in enumerate(PDE_NAMES):
            ax = axes[1, species_index]

            ax.plot(
                pde_time[traj],
                pde_res[traj, :, species_index],
                linewidth=1.2,
            )

            ax.axhline(
                0.0,
                linewidth=0.8,
                linestyle="--",
            )

            ax.set_title(
                f"{species_name}: PDE residual"
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("dX/dt - RHS")
            ax.grid(True, alpha=0.25)

        # No internal-energy PDE residual yet.
        axes[1, 3].axis("off")

        fig.suptitle(
            f"Residuals: trajectory {traj}"
        )

        fig.tight_layout(
            rect=(0.0, 0.0, 1.0, 0.96)
        )

        fig.savefig(
            trajectory_dir / f"trajectory_{traj:05d}.png",
            dpi=160,
        )

        plt.close(fig)

def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint
    data_path = args.data
    output_dir = args.output_dir

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"HDF5 dataset not found: {data_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = select_device()

    model = load_model(
        checkpoint_path,
        device,
    )

    X_value, Y_value, X_pde, PDE_aux = load_precomputed_dataset(
        data_path,
        include_pde=True,
    )

    print(f"Checkpoint:       {checkpoint_path}")
    print(f"HDF5 dataset:     {data_path}")
    print(f"Value points:     {len(X_value)}")
    print(f"PDE points:       {len(X_pde)}")
    print(f"Fraction A:       {FRACTION_A:.8g}")
    print(f"Fraction C:       {FRACTION_C:.8g}")
    print(f"ln(10)/A:         {CHAIN_FACTOR:.8g}")

    # -------------------------------------------------------------
    # Value residual
    # -------------------------------------------------------------
    value_residuals = calculate_value_residuals(
        model,
        X_value,
        Y_value,
        device,
    )

    # -------------------------------------------------------------
    # PDE residual
    # -------------------------------------------------------------
    derivative_phys, rhs_phys, pde_residuals = calculate_pde_residuals(
        model,
        X_pde,
        PDE_aux,
        device,
    )

    print_statistics(
        "VALUE RESIDUAL: normalized PINN - normalized Grackle",
        value_residuals,
        VALUE_NAMES,
    )

    print_statistics(
        "PDE DERIVATIVE: physical dX/dt (Not residual!)",
        derivative_phys,
        PDE_NAMES,
    )

    print_statistics(
        "PDE RHS: physical chemistry RHS (Not residual!)",
        rhs_phys,
        PDE_NAMES,
    )

    print_statistics(
        "PDE RESIDUAL: physical dX/dt - physical RHS",
        pde_residuals,
        PDE_NAMES,
    )

    save_trajectory_residual_plots(
        output_dir,
        X_value,
        X_pde,
        value_residuals,
        pde_residuals,
    )

    # Save raw diagnostic arrays.
    np.save(
        output_dir / "value_residuals.npy",
        value_residuals,
    )

    np.save(
        output_dir / "pde_derivative.npy",
        derivative_phys,
    )

    np.save(
        output_dir / "pde_rhs.npy",
        rhs_phys,
    )

    np.save(
        output_dir / "pde_residuals.npy",
        pde_residuals,
    )

    save_statistics_text(
        output_dir / "residual_statistics.txt",
        checkpoint_path.stem,
        value_residuals,
        pde_residuals,
    )

    save_histograms(
        output_dir,
        value_residuals,
        VALUE_NAMES,
        "value_residual",
    )

    save_histograms(
        output_dir,
        pde_residuals,
        PDE_NAMES,
        "pde_residual",
    )

    print()
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
