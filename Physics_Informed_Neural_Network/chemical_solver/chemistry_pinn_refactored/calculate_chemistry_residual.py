#!/usr/bin/env python3
"""
Load a trained ChemistryPINN checkpoint and calculate residuals on
the precomputed validation dataset.

Residual definition:
    residual = PINN - Grackle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import VALIDATION_DATA_FILE, INTERPOLATION_KWARGS
from grackle_data import load_precomputed_dataset
from normalisation import normalise_inputs, normalise_outputs, denormalise_outputs
from pinn_model import ChemistryPINN


# Must match train_pinn.py
MODEL_HIDDEN_DIM = 64
MODEL_N_HIDDEN_LAYERS = 4
MODEL_TIME_SCALE = 1.0

FEATURE_NAMES = ("HI", "HeI", "HeII", "u")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained ChemistryPINN checkpoint and calculate validation residuals."
    )
    parser.add_argument(
        "-I",
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained .pt/.pth checkpoint.",
    )
    parser.add_argument(
        "-O",
        "--output-dir",
        type=Path,
        default=Path("trained_results_v1_relaxed_n_10000/residual_results"),
        help="Directory where residual statistics and plots are saved.",
    )
    parser.add_argument(
        "-D",
        "--validation-data",
        type=Path,
        default=VALIDATION_DATA_FILE,
        help="Validation .npz file. Defaults to VALIDATION_DATA_FILE from config.py.",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=1,
        help="Trajectory index used for the detailed residual plot.",
    )
    parser.add_argument(
        "--denormalise",
        action="store_true",
        help=(
            "Calculate residuals after denormalising predictions to physical/code "
            "space. By default, residuals are calculated in normalized space."
        ),
    )
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> ChemistryPINN:
    model = ChemistryPINN(
        hidden_dim=MODEL_HIDDEN_DIM,
        n_hidden_layers=MODEL_N_HIDDEN_LAYERS,
        time_scale=MODEL_TIME_SCALE,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    model.load_state_dict(state_dict)
    model.eval()
    return model


def calculate_statistics(residuals: np.ndarray) -> dict[str, np.ndarray]:
    abs_residuals = np.abs(residuals)

    return {
        "mean": np.mean(residuals, axis=0),
        "std": np.std(residuals, axis=0),
        "max": np.max(residuals, axis=0),
        "min": np.min(residuals, axis=0),
        "mean_abs": np.mean(abs_residuals, axis=0),
        "std_abs": np.std(abs_residuals, axis=0),
        "max_abs": np.max(abs_residuals, axis=0),
        "min_abs": np.min(abs_residuals, axis=0),
    }


def print_statistics(checkpoint_label: str, stats: dict[str, np.ndarray]) -> None:
    print(f"\nModel name: {checkpoint_label}")

    for i, name in enumerate(FEATURE_NAMES):
        print(f"\n{name}")
        print(f"  Mean residual:               {stats['mean'][i]:.8e}")
        print(f"  Standard deviation:          {stats['std'][i]:.8e}")
        print(f"  Max residual:                {stats['max'][i]:.8e}")
        print(f"  Min residual:                {stats['min'][i]:.8e}")
        print(f"  Mean absolute residual:      {stats['mean_abs'][i]:.8e}")
        print(f"  Std absolute residual:       {stats['std_abs'][i]:.8e}")
        print(f"  Max absolute residual:       {stats['max_abs'][i]:.8e}")
        print(f"  Min absolute residual:       {stats['min_abs'][i]:.8e}")


def save_statistics_text(
    output_dir: Path,
    checkpoint_label: str,
    stats: dict[str, np.ndarray],
    residual_space: str,
) -> None:
    output_file = output_dir / "residual_statistics.txt"

    with output_file.open("w") as f:
        f.write(f"Model name: {checkpoint_label}\n")
        f.write("Residual definition: PINN - Grackle\n")
        f.write(f"Residual space: {residual_space}\n\n")

        for i, name in enumerate(FEATURE_NAMES):
            f.write(f"{name}\n")
            f.write(f"  Mean residual:               {stats['mean'][i]:.8e}\n")
            f.write(f"  Standard deviation:          {stats['std'][i]:.8e}\n")
            f.write(f"  Max residual:                {stats['max'][i]:.8e}\n")
            f.write(f"  Min residual:                {stats['min'][i]:.8e}\n")
            f.write(f"  Mean absolute residual:      {stats['mean_abs'][i]:.8e}\n")
            f.write(f"  Std absolute residual:       {stats['std_abs'][i]:.8e}\n")
            f.write(f"  Max absolute residual:       {stats['max_abs'][i]:.8e}\n")
            f.write(f"  Min absolute residual:       {stats['min_abs'][i]:.8e}\n\n")

    print(f"Saved statistics: {output_file}")


def save_histograms(output_dir: Path, residuals: np.ndarray) -> None:
    abs_residuals = np.abs(residuals)

    for i, name in enumerate(FEATURE_NAMES):
        values = abs_residuals[:, i]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(values, bins=50, density=True)
        ax.set_xlabel(f"Absolute residual ({name})")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"Histogram of absolute residuals: {name}")
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            output_dir / f"histogram_absolute_residuals_{name}.png",
            dpi=200,
        )
        plt.close(fig)


def save_one_trajectory_plot(
    output_dir: Path,
    trajectory_index: int,
    X_raw: np.ndarray,
    Y_grackle: np.ndarray,
    Y_pinn: np.ndarray,
    normalized_space: bool,
) -> None:
    t = X_raw[:, 0]
    residuals = Y_pinn - Y_grackle

    density = X_raw[0, 1]
    HI0 = X_raw[0, 2]
    HeI0 = X_raw[0, 3]
    HeII0 = X_raw[0, 4]
    u0 = X_raw[0, 5]
    Gamma_HI = X_raw[0, 6]
    Gamma_HeI = X_raw[0, 7]
    Gamma_HeII = X_raw[0, 8]
    pi_HI = X_raw[0, 9]
    pi_HeI = X_raw[0, 10]
    pi_HeII = X_raw[0, 11]

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(13, 14),
        sharex="col",
    )

    for i, name in enumerate(FEATURE_NAMES):
        axes[i, 0].plot(t, residuals[:, i])
        axes[i, 0].set_ylabel("Residual")
        axes[i, 0].set_title(f"{name}: PINN - Grackle")
        axes[i, 0].grid(True, alpha=0.25)

        if normalized_space:
            axes[i, 1].plot(t, Y_grackle[:, i], "-", label="Grackle")
            axes[i, 1].plot(t, Y_pinn[:, i], "--", label="PINN")
        else:
            axes[i, 1].semilogy(t, Y_grackle[:, i], "-", label="Grackle")
            axes[i, 1].semilogy(t, Y_pinn[:, i], "--", label="PINN")
        axes[i, 1].set_title(f"{name}: Grackle vs PINN")
        axes[i, 1].grid(True, which="both", alpha=0.25)
        axes[i, 1].legend(loc="best")

    axes[-1, 0].set_xlabel("Time [Myr]")
    axes[-1, 1].set_xlabel("Time [Myr]")

    parameter_text = (
        f"density = {density:.4e}\n"
        f"HI0 = {HI0:.4e}, HeI0 = {HeI0:.4e}, "
        f"HeII0 = {HeII0:.4e}, u0 = {u0:.4e}\n"
        f"Gamma_HI = {Gamma_HI:.4e} /Myr, "
        f"Gamma_HeI = {Gamma_HeI:.4e} /Myr, "
        f"Gamma_HeII = {Gamma_HeII:.4e} /Myr\n"
        f"pi_HI = {pi_HI:.4e}, "
        f"pi_HeI = {pi_HeI:.4e}, "
        f"pi_HeII = {pi_HeII:.4e}"
    )

    space_label = "normalized space" if normalized_space else "physical/code space"
    fig.suptitle(
        f"Residual example: trajectory {trajectory_index} ({space_label})",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        parameter_text,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.97])

    output_file = output_dir / f"residual_trajectory_{trajectory_index:04d}.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    print(f"Saved trajectory residual plot: {output_file}")


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    validation_data_path = args.validation_data

    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not validation_data_path.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {validation_data_path}"
        )

    checkpoint_label = checkpoint_path.stem

    device = select_device()
    model = load_model(checkpoint_path, device)

    X_raw, Y_grackle = load_precomputed_dataset(validation_data_path)

    print(f"Using checkpoint:  {checkpoint_path}")
    print(f"Validation data:   {validation_data_path}")
    print(f"Number of samples: {len(X_raw)}")

    X_norm = normalise_inputs(X_raw)
    X_tensor = torch.as_tensor(
        X_norm,
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        Y_pinn_norm = model(X_tensor).cpu()

    if args.denormalise:
        # Optional: compare in physical/code space.
        Y_pinn = denormalise_outputs(Y_pinn_norm).numpy()
        Y_grackle_compare = Y_grackle
        residual_space = "physical/code"
    else:
        # Default: match the training loss representation.
        # The model output is normalized, so normalize the Grackle target too.
        Y_pinn = Y_pinn_norm.numpy()
        Y_grackle_compare = normalise_outputs(Y_grackle)
        residual_space = "normalized"

    residuals = Y_pinn - Y_grackle_compare

    print(f"Residual space:    {residual_space}")

    np.save(output_dir / "residuals.npy", residuals)
    np.save(output_dir / "pinn_predictions.npy", Y_pinn)
    np.save(output_dir / "grackle_targets.npy", Y_grackle_compare)

    stats = calculate_statistics(residuals)

    print_statistics(checkpoint_label, stats)
    save_statistics_text(output_dir, checkpoint_label, stats, residual_space)
    save_histograms(output_dir, residuals)

    n_times = INTERPOLATION_KWARGS["n_times"]

    if len(X_raw) % n_times != 0:
        raise ValueError(
            f"Validation rows ({len(X_raw)}) are not divisible by "
            f"n_times ({n_times})."
        )

    n_trajectories = len(X_raw) // n_times
    trajectory_index = args.trajectory_index

    if trajectory_index < 0 or trajectory_index >= n_trajectories:
        raise IndexError(
            f"trajectory-index={trajectory_index} is outside "
            f"[0, {n_trajectories - 1}]"
        )

    start = trajectory_index * n_times
    end = start + n_times

    save_one_trajectory_plot(
        output_dir,
        trajectory_index,
        X_raw[start:end],
        Y_grackle_compare[start:end],
        Y_pinn[start:end],
        not args.denormalise,
    )

    print(f"\nSaved residual outputs to: {output_dir}")


if __name__ == "__main__":
    main()
