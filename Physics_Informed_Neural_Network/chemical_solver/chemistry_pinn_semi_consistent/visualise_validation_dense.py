#!/usr/bin/env python3
"""Visualise a trained ChemistryPINN on the precomputed validation dataset.

Each validation trajectory is saved as one figure with four panels:
HI, HeI, HeII, and specific internal energy.

Number of time samples used for plotting the trained PINN is set by N_PINN_PLOT_TIMES.

Solid line  : Grackle
Dashed line : PINN
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
from normalisation import normalise_inputs, denormalise_outputs, normalise_outputs
from pinn_model import ChemistryPINN


# Must match train_pinn.py
MODEL_HIDDEN_DIM = 64
MODEL_N_HIDDEN_LAYERS = 4
MODEL_TIME_SCALE = 1.0

FEATURE_NAMES = ("HI", "HeI", "HeII", "u")

# Number of time samples used only for plotting the trained PINN.
# The Grackle reference trajectory still uses the original HDF5 time grid.
N_PINN_PLOT_TIMES = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained ChemistryPINN checkpoint and plot validation trajectories."
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
        default=Path("trained_results_v4_a/validation_model_plots_epoch_9960"),
        help="Directory where validation figures are saved.",
    )

    parser.add_argument(
        "-D",
        "--validation-data",
        type=Path,
        default=VALIDATION_DATA_FILE,
        help="Validation HDF5 file. Defaults to VALIDATION_DATA_FILE from config.py.",
    )

    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to plot. Default: plot all trajectories.",
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

    # Support both plain state_dict and wrapped checkpoint dictionaries.
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    model.load_state_dict(state_dict)
    model.eval()

    return model


def plot_trajectory(
    trajectory_index: int,
    X_raw: np.ndarray,
    Y_grackle: np.ndarray,
    t_pinn: np.ndarray,
    Y_pinn: np.ndarray,
    output_dir: Path,
) -> None:

    t = X_raw[:, 0]

    # Every row in one trajectory contains the same conditioning parameters.
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
        2,
        2,
        figsize=(12, 9),
        sharex=True,
    )
    axes = axes.ravel()

    ylabels = (
        "HI fraction",
        "HeI fraction",
        "HeII fraction",
        "Specific internal energy",
    )

    for i, (ax, feature_name, ylabel) in enumerate(
        zip(axes, FEATURE_NAMES, ylabels)
    ):
        ax.plot(
            t,
            Y_grackle[:, i],
            "-",
            label="Grackle",
        )

        ax.plot(
            t_pinn,
            Y_pinn[:, i],
            "--",
            linewidth=1.2,
            label="PINN",
        )

        ax.set_title(feature_name)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")

    axes[2].set_xlabel("Time [Myr]")
    axes[3].set_xlabel("Time [Myr]")

    parameter_text = (
        f"density = {density:.4e}\n"
        f"HI0 = {HI0:.4e}, "
        f"HeI0 = {HeI0:.4e}, "
        f"HeII0 = {HeII0:.4e}, "
        f"u0 = {u0:.4e}\n"
        f"Gamma_HI = {Gamma_HI:.4e} /Myr, "
        f"Gamma_HeI = {Gamma_HeI:.4e} /Myr, "
        f"Gamma_HeII = {Gamma_HeII:.4e} /Myr\n"
        f"pi_HI = {pi_HI:.4e}, "
        f"pi_HeI = {pi_HeI:.4e}, "
        f"pi_HeII = {pi_HeII:.4e}"
    )

    fig.suptitle(
        f"Validation trajectory {trajectory_index}",
        fontsize=14,
    )

    fig.text(
        0.5,
        0.015,
        parameter_text,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0.0, 0.13, 1.0, 0.95])

    output_file = output_dir / f"trajectory_{trajectory_index:04d}.png"
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    validation_data_path = args.validation_data

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not validation_data_path.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {validation_data_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    model = load_model(checkpoint_path, device)

    X_raw, Y_raw = load_precomputed_dataset(validation_data_path)

    n_times = INTERPOLATION_KWARGS["n_times"]

    if len(X_raw) % n_times != 0:
        raise ValueError(
            f"Validation rows ({len(X_raw)}) are not divisible by "
            f"n_times ({n_times})."
        )

    n_trajectories = len(X_raw) // n_times

    if args.max_trajectories is not None:
        n_trajectories = min(n_trajectories, args.max_trajectories)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Validation data:   {validation_data_path}")
    print(f"Trajectories:      {n_trajectories}")
    print(f"Output directory:  {output_dir}")

    Y_norm = normalise_outputs(Y_raw)

    for trajectory_index in range(n_trajectories):
        start = trajectory_index * n_times
        end = start + n_times

        X_traj = X_raw[start:end]
        Y_traj = Y_norm[start:end]

        # Grackle keeps the original HDF5 time samples.
        # PINN is evaluated on a denser, arbitrary time grid for smooth plotting.
        t_pinn = np.linspace(
            X_traj[0, 0],
            X_traj[-1, 0],
            N_PINN_PLOT_TIMES,
        )

        X_pinn = np.repeat(
            X_traj[0:1],
            N_PINN_PLOT_TIMES,
            axis=0,
        )
        X_pinn[:, 0] = t_pinn

        X_pinn_norm = normalise_inputs(X_pinn)

        X_pinn_tensor = torch.as_tensor(
            X_pinn_norm,
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            Y_pinn = model(X_pinn_tensor).cpu().numpy()

        plot_trajectory(
            trajectory_index,
            X_traj,
            Y_traj,
            t_pinn,
            Y_pinn,
            output_dir,
        )

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
