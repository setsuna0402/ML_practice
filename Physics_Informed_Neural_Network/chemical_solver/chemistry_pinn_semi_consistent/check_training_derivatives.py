"""
Inspect time derivatives of a trained ChemistryPINN on every training trajectory.

Manually set CHECKPOINT_FILE and TRAINING_DATA_FILE below.

The derivative is computed in normalized output space:
    d(HI_norm)/dt, d(HeI_norm)/dt, d(HeII_norm)/dt, d(u_norm)/dt

The script does NOT save any output file.

It reports:
- every trajectory / feature / time point containing NaN or Inf;
- global minimum and maximum derivative for each predicted feature.
"""

from pathlib import Path

import numpy as np
import torch

from config import ALLOW_DEVICE, INTERPOLATION_KWARGS
from grackle_data import load_precomputed_dataset
from normalisation import normalise_inputs
from pinn_model import ChemistryPINN


# ---------------------------------------------------------------------
# Manually set files here
# ---------------------------------------------------------------------
CHECKPOINT_FILE = Path("./chemistry_pinn_best_epoch_500.pt")
TRAINING_DATA_FILE = Path("./grackle_relaxed_training_data_merged_10000.h5")


# Must match the training model.
HIDDEN_DIM = 64
N_HIDDEN_LAYERS = 4
TIME_SCALE = 1.0

FEATURE_NAMES = ("HI", "HeI", "HeII", "u")


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


def calculate_time_derivatives(model, X_norm, device, dtype):
    x = torch.as_tensor(
        X_norm,
        dtype=dtype,
        device=device,
    ).detach().clone().requires_grad_(True)

    prediction = model(x)
    derivatives = []

    for i in range(prediction.shape[1]):
        grad = torch.autograd.grad(
            prediction[:, i].sum(),
            x,
            create_graph=False,
            retain_graph=(i < prediction.shape[1] - 1),
        )[0]

        derivatives.append(grad[:, 0])

    return torch.stack(derivatives, dim=1).detach().cpu().numpy()


def report_derivatives(times, derivatives):
    n_trajectories, _, n_features = derivatives.shape

    bad_records = []

    global_min = np.full(n_features, np.inf)
    global_max = np.full(n_features, -np.inf)

    global_min_location = [None] * n_features
    global_max_location = [None] * n_features

    for traj in range(n_trajectories):
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            values = derivatives[traj, :, feature_index]

            finite_mask = np.isfinite(values)

            bad_indices = np.where(~finite_mask)[0]

            for time_index in bad_indices:
                bad_records.append(
                    (
                        traj,
                        feature_name,
                        int(time_index),
                        float(times[traj, time_index]),
                        values[time_index],
                    )
                )

            finite_indices = np.where(finite_mask)[0]

            if finite_indices.size == 0:
                continue

            finite_values = values[finite_indices]

            local_min_pos = int(np.argmin(finite_values))
            local_max_pos = int(np.argmax(finite_values))

            min_time_index = int(finite_indices[local_min_pos])
            max_time_index = int(finite_indices[local_max_pos])

            local_min = float(values[min_time_index])
            local_max = float(values[max_time_index])

            if local_min < global_min[feature_index]:
                global_min[feature_index] = local_min
                global_min_location[feature_index] = (
                    traj,
                    min_time_index,
                    float(times[traj, min_time_index]),
                )

            if local_max > global_max[feature_index]:
                global_max[feature_index] = local_max
                global_max_location[feature_index] = (
                    traj,
                    max_time_index,
                    float(times[traj, max_time_index]),
                )

    print()
    print("=" * 90)
    print("NON-FINITE DERIVATIVES")
    print("=" * 90)

    if not bad_records:
        print("No NaN or Inf derivatives found.")
    else:
        for traj, feature, time_index, time_value, value in bad_records:
            print(
                f"trajectory={traj:5d}, "
                f"feature={feature:4s}, "
                f"time_index={time_index:3d}, "
                f"t={time_value:.6e}, "
                f"value={value}"
            )

        bad_trajectories = sorted(set(record[0] for record in bad_records))

        print()
        print(
            f"Found {len(bad_records)} non-finite derivative values "
            f"in {len(bad_trajectories)} trajectories."
        )
        print("Affected trajectories:")
        print(bad_trajectories)

    print()
    print("=" * 90)
    print("GLOBAL DERIVATIVE RANGE")
    print("=" * 90)

    for i, feature_name in enumerate(FEATURE_NAMES):
        min_location = global_min_location[i]
        max_location = global_max_location[i]

        if min_location is None:
            print(f"d{feature_name}/dt: no finite values")
            continue

        min_traj, min_index, min_time = min_location
        max_traj, max_index, max_time = max_location

        print(
            f"d{feature_name}/dt:\n"
            f"  min = {global_min[i]: .6e} "
            f"(trajectory={min_traj}, index={min_index}, t={min_time:.6e})\n"
            f"  max = {global_max[i]: .6e} "
            f"(trajectory={max_traj}, index={max_index}, t={max_time:.6e})"
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
            f"Training data file not found: {TRAINING_DATA_FILE}"
        )

    X_phys, _ = load_precomputed_dataset(
        TRAINING_DATA_FILE,
        include_pde=False,
    )

    n_times = int(INTERPOLATION_KWARGS["n_times"])

    if X_phys.shape[0] % n_times != 0:
        raise ValueError(
            f"{X_phys.shape[0]} training rows cannot be divided "
            f"into trajectories with n_times={n_times}."
        )

    n_trajectories = X_phys.shape[0] // n_times

    times = X_phys[:, 0].reshape(
        n_trajectories,
        n_times,
    )

    X_norm = normalise_inputs(X_phys)

    model = load_model(
        CHECKPOINT_FILE,
        device,
        dtype,
    )

    derivatives = calculate_time_derivatives(
        model,
        X_norm,
        device,
        dtype,
    )

    derivatives = derivatives.reshape(
        n_trajectories,
        n_times,
        len(FEATURE_NAMES),
    )

    print(f"Training trajectories: {n_trajectories}")
    print(f"Time points per trajectory: {n_times}")

    report_derivatives(times, derivatives)


if __name__ == "__main__":
    main()
