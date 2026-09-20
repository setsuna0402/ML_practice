"""
Minimal training draft for ChemistryPINN.

This is intentionally only a baseline:
- precomputed or random-per-epoch Grackle datasets
- centralized parameter ranges/configuration
- log10 input/output normalization
- supervised trajectory loss only
- v2, recover the physical quantities and differentiate them to compute the PDE residual.

The model and data generator are already separated so that the next step
can add PINN residual terms and a proper parameter sampler without
rewriting the basic project structure.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from pinn_model import ChemistryPINN
from normalisation import normalise_inputs, normalise_outputs, denormalise_outputs
from grackle_data import (
    generate_random_dataset,
    load_precomputed_dataset,
)
from config import (
    RANDOM_SEED, N_EPOCHS, PDE_WEIGHT_WARMUP_EPOCHS,LEARNING_RATE, COSINE_ETA_MIN,
    BATCH_SIZE, VALIDATION_INTERVAL, OUTPUT_DIR,
    NUM_WORKERS, USE_PIN_MEMORY, ALLOW_DEVICE,
    DATA_MODE, PRECOMPUTED_DATA_FILE, VALIDATION_DATA_FILE,
    N_RANDOM_TRAJECTORIES_PER_EPOCH,
    DENSITY_UNIT, DENSITY_REF, HYDROGEN_FRACTION_BY_MASS,
    INTERPOLATION_KWARGS,
    VALIDATION_PLOT_N_SAMPLES, VALIDATION_PLOT_LOGY,
    PDE_ENABLED, PDE_LOSS_WEIGHT,
    PDE_START_EPOCH, PDE_HI_RATE_REF, PDE_HEI_RATE_REF, PDE_HEII_RATE_REF,
)

random_seed = RANDOM_SEED
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.set_default_dtype(torch.float32)

run_in_background = True
show_model_summary = False
allow_device = ALLOW_DEVICE
use_pin_memory = USE_PIN_MEMORY
n_epochs = N_EPOCHS
batch_size = BATCH_SIZE
validation_interval = VALIDATION_INTERVAL
output_dir = OUTPUT_DIR
num_workers = NUM_WORKERS

FEATURE_NAMES = ("HI", "HeI", "HeII", "u")
loss_weights = {
    "HI": 1.0,
    "HeI": 1.0,
    "HeII": 1.0,
    "u": 1.0,
}

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


def load_validation_datasets(path, dtype):
    """Load one precomputed dataset for validation."""
    X_np, Y_np = load_precomputed_dataset(path)
    dataset, _ = make_tensor_dataloader(X_np, Y_np, dtype, shuffle=False)
    return [(None, dataset)]


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


def make_pde_dataloader(X_pde_np, PDE_aux_np, dtype, *, shuffle=True):
    """Build a DataLoader for frozen-coefficient H/He midpoint physics points.

    X_pde uses the normal model-input normalisation. PDE_aux stays in Grackle
    code units with columns [k1, k2, k3, k4, k5, k6].
    """
    X_pde_np = normalise_inputs(X_pde_np)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X_pde_np, dtype=dtype),
        torch.as_tensor(PDE_aux_np, dtype=dtype),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    return dataset, dataloader


def frozen_coefficient_species_loss(model, input_batch, aux_batch, criterion):
    """Semi-consistent H/He residual using frozen Grackle k1..k6.

    The network supplies HI, HeI, HeII and their time derivatives. Reaction
    coefficients are frozen to the reference Grackle trajectory at the staggered
    midpoint collocation times, while electron density is reconstructed from the
    NN-predicted ion fractions and the input gas density.

    Provisional rate mapping:
      k1/k2: HI <-> HII
      k3/k4: HeI <-> HeII
      k5/k6: HeII <-> HeIII
    """
    x = input_batch.detach().clone().requires_grad_(True)
    prediction_norm = model(x)
    prediction_phys = denormalise_outputs(prediction_norm)

    HI = prediction_phys[:, 0]
    HeI = prediction_phys[:, 1]
    HeII = prediction_phys[:, 2]
    HII = 1.0 - HI
    HeIII = 1.0 - HeI - HeII

    grad_HI = torch.autograd.grad(HI.sum(), x, create_graph=True, retain_graph=True)[0][:, 0]
    grad_HeI = torch.autograd.grad(HeI.sum(), x, create_graph=True, retain_graph=True)[0][:, 0]
    grad_HeII = torch.autograd.grad(HeII.sum(), x, create_graph=True, retain_graph=True)[0][:, 0]

    k1, k2, k3, k4, k5, k6 = [aux_batch[:, i] for i in range(6)]

    # Reconstruct Grackle e_density from the NN state. The model input density
    # is rho_H + rho_He in code-density units. Grackle's six-species network
    # uses e_density = rho_HII + 0.25 * rho_HeII + 0.5 * rho_HeIII.
    density = DENSITY_REF * torch.pow(
        torch.as_tensor(10.0, dtype=x.dtype, device=x.device), x[:, 1]
    )
    rho_H = HYDROGEN_FRACTION_BY_MASS * density
    rho_He = (1.0 - HYDROGEN_FRACTION_BY_MASS) * density
    e_nn = rho_H * HII + 0.25 * rho_He * HeII + 0.5 * rho_He * HeIII
    ten = torch.as_tensor(10.0, dtype=x.dtype, device=x.device)
    Gamma_HI = PDE_HI_RATE_REF * torch.pow(ten, x[:, 6])
    Gamma_HeI = PDE_HEI_RATE_REF * torch.pow(ten, x[:, 7])
    Gamma_HeII = PDE_HEII_RATE_REF * torch.pow(ten, x[:, 8])

    # Fraction equations. The total H or He mass density cancels because each
    # elemental fraction is normalized by its conserved elemental total.
    rhs_HI = -(k1 * e_nn + Gamma_HI) * HI + k2 * e_nn * HII

    rhs_HeI = -(k3 * e_nn + Gamma_HeI) * HeI + k4 * e_nn * HeII

    rhs_HeII = (k3 * e_nn + Gamma_HeI) * HeI \
        - (k4 * e_nn + k5 * e_nn + Gamma_HeII) * HeII \
        + k6 * e_nn * HeIII

    residual_HI = (grad_HI - rhs_HI) / PDE_HI_RATE_REF
    residual_HeI = (grad_HeI - rhs_HeI) / PDE_HEI_RATE_REF
    residual_HeII = (grad_HeII - rhs_HeII) / PDE_HEII_RATE_REF

    zero_H = torch.zeros_like(residual_HI)
    zero_HeI = torch.zeros_like(residual_HeI)
    zero_HeII = torch.zeros_like(residual_HeII)
    return (
        criterion(residual_HI, zero_H)
        + criterion(residual_HeI, zero_HeI)
        + criterion(residual_HeII, zero_HeII)
    )


def build_epoch_training_data(dtype, rng):
    """Return value-loss and optional PDE DataLoaders for the selected mode."""
    pde_dataset = None
    pde_dataloader = None

    if DATA_MODE == "precomputed":
        if PDE_ENABLED:
            X_np, Y_np, X_pde_np, PDE_aux_np = load_precomputed_dataset(
                PRECOMPUTED_DATA_FILE, include_pde=True
            )
            pde_dataset, pde_dataloader = make_pde_dataloader(
                X_pde_np, PDE_aux_np, dtype, shuffle=True
            )
        else:
            X_np, Y_np = load_precomputed_dataset(PRECOMPUTED_DATA_FILE)

    elif DATA_MODE == "random_epoch":
        # Keep the first frozen-coefficient prototype focused on the fixed
        # precomputed dataset. This avoids regenerating Grackle PDE auxiliaries
        # inside every training epoch.
        if PDE_ENABLED:
            raise NotImplementedError(
                "PDE_ENABLED=True currently requires DATA_MODE='precomputed'."
            )
        X_np, Y_np, _ = generate_random_dataset(
            N_RANDOM_TRAJECTORIES_PER_EPOCH,
            code_density=DENSITY_UNIT,
            rng=rng,
            interpolation_kwargs=INTERPOLATION_KWARGS,
        )
    else:
        raise ValueError(
            f"Unknown DATA_MODE={DATA_MODE!r}. Use 'precomputed' or 'random_epoch'."
        )

    value_dataset, value_dataloader = make_tensor_dataloader(
        X_np, Y_np, dtype, shuffle=True
    )
    return value_dataset, value_dataloader, pde_dataset, pde_dataloader


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    *,
    pde_dataloader=None,
    pde_weight=0.0,
):
    """Run one epoch with supervised value loss plus optional frozen H/He PDE loss."""
    model.train()
    total_losses = {name: 0.0 for name in FEATURE_NAMES}
    sample_count = 0
    pde_loss_sum = 0.0
    pde_batch_count = 0

    pde_iterator = iter(pde_dataloader) if pde_dataloader is not None else None

    for input_batch, target_batch in dataloader:
        input_batch = input_batch.to(device, non_blocking=use_pin_memory)
        target_batch = target_batch.to(device, non_blocking=use_pin_memory)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(input_batch)
        feature_losses = calculate_feature_losses(prediction, target_batch, criterion)
        value_loss = combine_feature_losses(feature_losses)
        loss = value_loss

        if pde_iterator is not None and pde_weight > 0.0:
            try:
                pde_input_batch, pde_aux_batch = next(pde_iterator)
            except StopIteration:
                pde_iterator = iter(pde_dataloader)
                pde_input_batch, pde_aux_batch = next(pde_iterator)

            pde_input_batch = pde_input_batch.to(device, non_blocking=use_pin_memory)
            pde_aux_batch = pde_aux_batch.to(device, non_blocking=use_pin_memory)

            pde_loss = frozen_coefficient_species_loss(
                model, pde_input_batch, pde_aux_batch, criterion
            )
            loss = loss + pde_weight * pde_loss
            pde_loss_sum += pde_loss.item()
            pde_batch_count += 1

        loss.backward()
        optimizer.step()

        batch_size_actual = input_batch.size(0)
        for name in FEATURE_NAMES:
            total_losses[name] += feature_losses[name].item() * batch_size_actual
        sample_count += batch_size_actual

    feature_mean = {
        name: value / sample_count for name, value in total_losses.items()
    }
    mean_pde_loss = pde_loss_sum / pde_batch_count if pde_batch_count else 0.0
    return feature_mean, mean_pde_loss


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
    # predictions = denormalise_outputs(torch.cat(predictions))
    # targets = denormalise_outputs(torch.cat(targets))
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    return losses, predictions, targets


def _validation_trajectories_for_plot(validation_results):
    """Return up to N complete trajectories for plotting only.

    Validation loss is computed before this function is called and always uses
    every validation row.  Here we only split the already-evaluated arrays into
    fixed-length trajectories and retain a small deterministic subset for the
    figure.
    """
    n_times = int(INTERPOLATION_KWARGS["n_times"])
    trajectories = []

    for _, inputs, predictions, targets in validation_results:
        n_rows = inputs.shape[0]
        if n_rows % n_times != 0:
            raise ValueError(
                f"Validation rows ({n_rows}) are not divisible by n_times={n_times}. "
                "The plotting code assumes fixed-length interpolated trajectories."
            )

        for start in range(0, n_rows, n_times):
            stop = start + n_times
            trajectories.append((
                inputs[start:stop],
                predictions[start:stop],
                targets[start:stop],
            ))

            if len(trajectories) >= VALIDATION_PLOT_N_SAMPLES:
                return trajectories

    return trajectories


def save_validation_plot(epoch, validation_results):
    """Plot up to ten validation trajectories in four quantity panels.

    Each trajectory uses one colour.  Grackle is solid and PINN is dashed.
    The plotting subset does not affect validation-loss calculation.
    """
    trajectories = _validation_trajectories_for_plot(validation_results)
    if not trajectories:
        return

    figure, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for feature_index, (axis, feature_name) in enumerate(zip(axes, FEATURE_NAMES)):
        for sample_index, (inputs, predictions, targets) in enumerate(trajectories):
            time_values = inputs[:, 0].numpy()
            target_values = targets[:, feature_index].numpy()
            prediction_values = predictions[:, feature_index].numpy()

            # Use the target line to obtain one colour, then reuse it for the
            # matching PINN trajectory.  Non-positive predictions are left
            # unmodified; matplotlib simply cannot draw them on a log y-axis.
            if VALIDATION_PLOT_LOGY:
                target_line, = axis.semilogy(
                    time_values, target_values,
                    linewidth=1.2, alpha=0.85,
                    label="Grackle" if sample_index == 0 else None,
                )
                axis.semilogy(
                    time_values, prediction_values, "o",
                    color=target_line.get_color(),
                    linewidth=1.2, alpha=0.85,
                    markersize=1,
                    label="PINN" if sample_index == 0 else None,
                )
            else:
                target_line, = axis.plot(
                    time_values, target_values,
                    linewidth=1.2, alpha=0.85,
                    label="Grackle" if sample_index == 0 else None,
                )
                axis.plot(
                    time_values, prediction_values, "o",
                    color=target_line.get_color(),
                    linewidth=1.2, alpha=0.85,
                    markersize=1,
                    label="PINN" if sample_index == 0 else None,
                )

        axis.set_title(feature_name)
        axis.set_ylabel(feature_name)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(loc="best")

    axes[2].set_xlabel("Time [code units]")
    axes[3].set_xlabel("Time [code units]")

    scale_name = "log y-scale" if VALIDATION_PLOT_LOGY else "linear y-scale"
    figure.suptitle(
        f"Validation trajectories, epoch {epoch} "
        f"({len(trajectories)} samples, {scale_name})"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_dir / f"validation_epoch_{epoch:05d}.png", dpi=150)
    plt.close(figure)


def main():
    # -------------------------------------------------------------
    # Generate one reference trajectory.
    # -------------------------------------------------------------
    dtype = torch.float32
    device = select_device()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if VALIDATION_DATA_FILE is None:
        raise ValueError(
            "VALIDATION_DATA_FILE must be set to a precomputed validation HDF5 file. "
            "On-the-fly validation generation is not supported."
        )
    if not VALIDATION_DATA_FILE.exists():
        raise FileNotFoundError(f"Missing validation dataset: {VALIDATION_DATA_FILE}")
    validation_datasets = load_validation_datasets(VALIDATION_DATA_FILE, dtype)

    # Keep validation fixed.  Only the training set changes in random_epoch mode.
    data_rng = np.random.default_rng(random_seed)

    if DATA_MODE == "precomputed":
        if not PRECOMPUTED_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Missing {PRECOMPUTED_DATA_FILE}. "
                "Run precompute_grackle_data.py first."
            )
        train_dataset, train_dataloader, pde_dataset, pde_dataloader = build_epoch_training_data(dtype, data_rng)
    elif DATA_MODE == "random_epoch":
        train_dataset = None
        train_dataloader = None
        pde_dataset = None
        pde_dataloader = None
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

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=COSINE_ETA_MIN,
    )
    criterion = torch.nn.MSELoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    
    time_start = time.time()
    for epoch in tqdm(range(n_epochs), desc="Training", disable=run_in_background):
        if DATA_MODE == "random_epoch":
            train_dataset, train_dataloader, pde_dataset, pde_dataloader = build_epoch_training_data(dtype, data_rng)

        current_pde_weight = (
            PDE_LOSS_WEIGHT
            if PDE_ENABLED and epoch >= PDE_START_EPOCH
            else 0.0
        )
        # slowly increse pde loss's weight
        if epoch < PDE_WEIGHT_WARMUP_EPOCHS:
            current_pde_weight = current_pde_weight * epoch / PDE_WEIGHT_WARMUP_EPOCHS

        train_losses, train_pde_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device,
            pde_dataloader=pde_dataloader,
            pde_weight=current_pde_weight,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        # if epoch % 250 == 0 or epoch == n_epochs - 1:
        train_total = sum(loss_weights[name] * train_losses[name] for name in FEATURE_NAMES)
        total_with_pde = train_total + current_pde_weight * train_pde_loss
        tqdm.write(
            f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_with_pde:.6e}, "
            f"Value: {train_total:.6e}, Species_PDE: {train_pde_loss:.3e}, "
            f"lambda_PDE: {current_pde_weight:.3e}, LR: {current_lr:.3e}, "
            + ", ".join(f"{name}={train_losses[name]:.3e}" for name in FEATURE_NAMES)
        )

        # Update the learning rate for the next epoch.
        scheduler.step()

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
        # save model state every 100 epochs and at the last epoch
        if (epoch + 1) % 100 == 0 or epoch == n_epochs - 1:
            torch.save(
                model.state_dict(),
                output_dir / f"chemistry_pinn_epoch_{epoch + 1}.pt",
            )

    torch.save(
        model.state_dict(),
        output_dir / f"chemistry_pinn_final_epoch_{epoch + 1}.pt",
    )
    time_end = time.time()
    elapsed_time = time_end - time_start
    
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time /60:.2f} minutes).")

if __name__ == "__main__":
    main()
