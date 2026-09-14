"""
Minimal training draft for ChemistryPINN.

This is intentionally only a baseline:
- one Grackle trajectory
- no final parameter sampling strategy
- no final normalization
- supervised trajectory loss only

The model and data generator are already separated so that the next step
can add PINN residual terms and a proper parameter sampler without
rewriting the basic project structure.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

from pinn_model import ChemistryPINN
from normalisation import (
    normalise_inputs,
    denormalise_inputs,
    normalise_outputs,
    denormalise_outputs,
)
from grackle_data import (
    generate_grackle_trajectory,
    trajectory_to_pinn_arrays,
    generate_random_dataset,
    load_precomputed_dataset,
)

# Metal flag for ML
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.set_default_dtype(torch.float32)

run_in_background = True # make tqdm to be silent mode
show_model_summary = False # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
allow_device = True  # Set to False if you want to force using CPU.
use_pin_memory = True  # Set to True if you use GPU. False for CPU and MPS.
n_epochs = 1500 # The number of training epochs for classifier.
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 200
validation_interval = 20
output_dir = Path("./trained_results_v1_a")
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
alpha_phys = 1e-3  # Weight for the physics loss term.

# ---------------------------------------------------------------------
# Training-data mode
# ---------------------------------------------------------------------
# "precomputed": load one fixed .npz dataset and reshuffle mini-batches each epoch.
# "random_epoch": generate fresh random Grackle trajectories at the start of every epoch.
DATA_MODE = "precomputed"
PRECOMPUTED_DATA_FILE = Path("./grackle_training_data.npz")
N_RANDOM_TRAJECTORIES_PER_EPOCH = 8

# POC parameter ranges used by random_epoch.  These are deliberately kept
# in one dictionary so they can be changed without touching the data code.
RANDOM_SAMPLER_KWARGS = {
    "density_range": (1.0, 1.0e4),
    "temperature_range": (1.0e2, 1.0e5),
    "Gamma_HI_range": (1.0e-15, 1.0e-11),
    "Gamma_HeI_range": (1.0e-16, 1.0e-11),
    "Gamma_HeII_range": (1.0e-18, 1.0e-12),
    "final_time": 1.0,
}


# ---------------------------------------------------------------------
# Example physical parameters: copied from the tested wrapper.
# Replace these later with the final parameter-sampling scheme.
# ---------------------------------------------------------------------

Z_GRACKLE = 3.017

N_BARYON_Z0 = 2.5e-7               # cm^-3
M_H = 1.6735575e-24                 # g
DENSITY_Z0 = N_BARYON_Z0 * M_H

DENSITY = DENSITY_Z0 * (1.0 + Z_GRACKLE) ** 3
TEMPERATURE = 1.0e3

GAMMA_HI = 8.187467866905527e-13
GAMMA_HEI = 4.749287143327514e-13
GAMMA_HEII = 3.611784136088691e-15

PI_HI = 3.2556039206009867e-12
PI_HEI = 3.387724795493442e-12
PI_HEII = 7.789881261416638e-14

FEATURE_NAMES = ("HI", "HeI", "HeII", "u")
loss_weights = {
    "HI": 1.0,
    "HeI": 1.0,
    "HeII": 1.0,
    "u": 1.0,
}
validation_plot_feature = "HI"


def make_parameter_set(density=1.0, temperature=TEMPERATURE,
                       Gamma_HI=GAMMA_HI, Gamma_HeI=GAMMA_HEI,
                       Gamma_HeII=GAMMA_HEII, pi_HI=PI_HI,
                       pi_HeI=PI_HEI, pi_HeII=PI_HEII):
    return {
        "density": density,
        "temperature": temperature,
        "Gamma_HI": Gamma_HI,
        "Gamma_HeI": Gamma_HeI,
        "Gamma_HeII": Gamma_HeII,
        "pi_HI": pi_HI,
        "pi_HeI": pi_HeI,
        "pi_HeII": pi_HeII,
        "code_density": DENSITY_Z0,
        "final_time": 1.0,
    }


training_parameter_sets = [make_parameter_set()]
validation_parameter_sets = [
    make_parameter_set(temperature=2.0e3),
    make_parameter_set(temperature=5.0e3),
]


def select_device():
    """Select the fastest supported device and matching loader settings."""
    global num_workers, use_pin_memory

    if allow_device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        print("Using GPU")
        return torch.device("cuda")

    if allow_device and torch.backends.mps.is_available():
        use_pin_memory = False
        num_workers = 0
        print("Using MPS")
        return torch.device("mps")

    use_pin_memory = False
    num_workers = 0
    print("Using CPU")
    return torch.device("cpu")


def calculate_feature_losses(prediction, target, criterion):
    """Return one loss tensor for each predicted feature."""
    return {
        name: criterion(prediction[:, index], target[:, index])
        for index, name in enumerate(FEATURE_NAMES)
    }


def combine_feature_losses(feature_losses):
    """Combine independently calculated feature losses using configured weights."""
    return sum(loss_weights[name] * feature_losses[name] for name in FEATURE_NAMES)


def generate_parameter_dataset(parameter_sets):
    """Generate and concatenate supervised rows for several parameter sets."""
    input_arrays = []
    target_arrays = []
    for parameters in parameter_sets:
        trajectory = generate_grackle_trajectory(**parameters)
        inputs, targets = trajectory_to_pinn_arrays(trajectory)
        input_arrays.append(inputs)
        target_arrays.append(targets)

    if not input_arrays:
        raise ValueError("At least one parameter set is required.")

    return np.concatenate(input_arrays), np.concatenate(target_arrays)


def generate_validation_datasets(parameter_sets, dtype):
    """Generate one ordered TensorDataset per independent validation case."""
    validation_data = []
    for parameters in parameter_sets:
        trajectory = generate_grackle_trajectory(**parameters)
        inputs, targets = trajectory_to_pinn_arrays(trajectory)
        inputs = normalise_inputs(inputs)
        targets = normalise_outputs(targets)
        validation_data.append(
            (
                parameters,
                torch.utils.data.TensorDataset(
                    torch.as_tensor(inputs, dtype=dtype),
                    torch.as_tensor(targets, dtype=dtype),
                ),
            )
        )
    return validation_data


def make_tensor_dataloader(X_np, Y_np, dtype, *, shuffle=True):
    """Build a normalized TensorDataset/DataLoader pair from NumPy arrays."""
    X_np = normalise_inputs(X_np)
    Y_np = normalise_outputs(Y_np)

    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X_np, dtype=dtype),
        torch.as_tensor(Y_np, dtype=dtype),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    return dataset, dataloader


def build_epoch_training_data(dtype, rng):
    """Return the training DataLoader for the selected DATA_MODE."""
    if DATA_MODE == "precomputed":
        X_np, Y_np = load_precomputed_dataset(PRECOMPUTED_DATA_FILE)
    elif DATA_MODE == "random_epoch":
        X_np, Y_np, _ = generate_random_dataset(
            N_RANDOM_TRAJECTORIES_PER_EPOCH,
            code_density=DENSITY_Z0,
            rng=rng,
            sampler_kwargs=RANDOM_SAMPLER_KWARGS,
        )
    else:
        raise ValueError(
            f"Unknown DATA_MODE={DATA_MODE!r}. "
            "Use 'precomputed' or 'random_epoch'."
        )

    return make_tensor_dataloader(X_np, Y_np, dtype, shuffle=True)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Run one shuffled mini-batch training epoch and return feature losses."""
    model.train()
    total_losses = {name: 0.0 for name in FEATURE_NAMES}
    sample_count = 0

    for input_batch, target_batch in dataloader:
        input_batch = input_batch.to(device, non_blocking=use_pin_memory)
        target_batch = target_batch.to(device, non_blocking=use_pin_memory)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(input_batch)
        feature_losses = calculate_feature_losses(prediction, target_batch, criterion)
        loss = combine_feature_losses(feature_losses)
        loss.backward()
        optimizer.step()

        batch_size_actual = input_batch.size(0)
        for name in FEATURE_NAMES:
            total_losses[name] += feature_losses[name].item() * batch_size_actual
        sample_count += batch_size_actual

    return {name: value / sample_count for name, value in total_losses.items()}


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate one ordered validation trajectory."""
    model.eval()
    total_losses = {name: 0.0 for name in FEATURE_NAMES}
    sample_count = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for input_batch, target_batch in dataloader:
            input_batch = input_batch.to(device, non_blocking=use_pin_memory)
            target_batch = target_batch.to(device, non_blocking=use_pin_memory)
            prediction = model(input_batch)
            feature_losses = calculate_feature_losses(prediction, target_batch, criterion)

            batch_size_actual = input_batch.size(0)
            for name in FEATURE_NAMES:
                total_losses[name] += feature_losses[name].item() * batch_size_actual
            sample_count += batch_size_actual
            predictions.append(prediction.cpu())
            targets.append(target_batch.cpu())

    losses = {name: value / sample_count for name, value in total_losses.items()}

    # Losses are intentionally measured in normalized space.  Convert the
    # stored arrays back to physical/code units for validation plots/output.
    predictions = denormalise_outputs(torch.cat(predictions))
    targets = denormalise_outputs(torch.cat(targets))
    return losses, predictions, targets


def save_validation_plot(epoch, validation_results):
    """Save one selected feature versus time for each parameter case."""
    feature_index = FEATURE_NAMES.index(validation_plot_feature)
    figure, axes = plt.subplots(
        len(validation_results), 1,
        figsize=(9, 3.5 * len(validation_results)),
        sharex=False,
        squeeze=False,
    )

    for axis, (parameters, inputs, predictions, targets) in zip(axes[:, 0], validation_results):
        time_values = inputs[:, 0].numpy()
        axis.plot(time_values, targets[:, feature_index].numpy(), label="Grackle")
        axis.plot(time_values, predictions[:, feature_index].numpy(), "--", label="PINN")
        axis.set_ylabel(validation_plot_feature)
        axis.set_xlabel("Time [code units]")
        axis.set_title(
            f"density={parameters['density']:.3g}, T={parameters['temperature']:.3g}\n"
            f"Gamma=({parameters['Gamma_HI']:.3g}, {parameters['Gamma_HeI']:.3g}, "
            f"{parameters['Gamma_HeII']:.3g}), "
            f"pi=({parameters['pi_HI']:.3g}, {parameters['pi_HeI']:.3g}, "
            f"{parameters['pi_HeII']:.3g})"
        )
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    figure.suptitle(f"Validation: {validation_plot_feature}, epoch {epoch}")
    figure.tight_layout()
    figure.savefig(output_dir / f"validation_epoch_{epoch:05d}.png")
    plt.close(figure)


def main():
    # -------------------------------------------------------------
    # Generate one reference trajectory.
    # -------------------------------------------------------------
    dtype = torch.float32
    device = select_device()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if validation_plot_feature not in FEATURE_NAMES:
        raise ValueError(f"Unknown validation feature: {validation_plot_feature}")
    if not validation_parameter_sets:
        raise ValueError("At least one validation parameter set is required.")

    validation_datasets = generate_validation_datasets(validation_parameter_sets, dtype)

    # Keep validation fixed.  Only the training set changes in random_epoch mode.
    data_rng = np.random.default_rng(random_seed)

    if DATA_MODE == "precomputed":
        if not PRECOMPUTED_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Missing {PRECOMPUTED_DATA_FILE}. "
                "Run precompute_grackle_data.py first."
            )
        train_dataset, train_dataloader = build_epoch_training_data(dtype, data_rng)
    elif DATA_MODE == "random_epoch":
        train_dataset = None
        train_dataloader = None
    else:
        raise ValueError(
            f"Unknown DATA_MODE={DATA_MODE!r}. "
            "Use 'precomputed' or 'random_epoch'."
        )

    # time_scale [code_time_units]
    model = ChemistryPINN(hidden_dim=64, n_hidden_layers=4, time_scale=1.0,   ).to(device=device, dtype=dtype)

    if show_model_summary:
        print(model)
        if train_dataloader is not None:
            print(f"Training batches per epoch: {len(train_dataloader)}")
        else:
            print("Training batches per epoch: generated dynamically")

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    criterion = torch.nn.MSELoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in tqdm(range(n_epochs), desc="Training", disable=run_in_background):
        if DATA_MODE == "random_epoch":
            train_dataset, train_dataloader = build_epoch_training_data(dtype, data_rng)

        train_losses = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        if epoch % 250 == 0 or epoch == n_epochs - 1:
            train_total = sum(loss_weights[name] * train_losses[name] for name in FEATURE_NAMES)
            tqdm.write(
                f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_total:.6e}, "
                + ", ".join(f"{name}={train_losses[name]:.3e}" for name in FEATURE_NAMES)
            )

        if (epoch + 1) % validation_interval == 0 or epoch == n_epochs - 1:
            all_validation_losses = []
            validation_results = []
            for parameters, validation_dataset in validation_datasets:
                validation_dataloader = torch.utils.data.DataLoader(
                    validation_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=use_pin_memory,
                )
                validation_losses, predictions, targets = evaluate_model(
                    model, validation_dataloader, criterion, device
                )
                all_validation_losses.append(validation_losses)
                validation_results.append(
                    (
                        parameters,
                        validation_dataset.tensors[0],
                        predictions,
                        targets,
                    )
                )

            mean_validation_losses = {
                name: sum(losses[name] for losses in all_validation_losses)
                / len(all_validation_losses)
                for name in FEATURE_NAMES
            }
            validation_total = sum(
                loss_weights[name] * mean_validation_losses[name]
                for name in FEATURE_NAMES
            )
            tqdm.write(
                f"Epoch {epoch + 1}/{n_epochs}, Validation Loss: {validation_total:.6e}, "
                + ", ".join(
                    f"{name}={mean_validation_losses[name]:.3e}"
                    for name in FEATURE_NAMES
                )
            )
            save_validation_plot(epoch + 1, validation_results)
            if validation_total < best_validation_loss:
                best_validation_loss = validation_total
                torch.save(model.state_dict(), output_dir / f"chemistry_pinn_best_epoch_{epoch + 1}.pt")

    torch.save(
        model.state_dict(),
        output_dir / f"chemistry_pinn_final_epoch_{epoch + 1}.pt",
    )


if __name__ == "__main__":
    main()
