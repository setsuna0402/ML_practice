'''
This is the simplest example of a Physics-Informed Neural Network (PINN) for solving the 1D Radiative Transfer Equation (RTE).
Here, absorption and emission are considered, but scattering is neglected. 
The values of them follow the analytic functions of the spatial variable x, 
and the PINN does not see them. The PINN is trained to learn the solution of the RTE.

Version two: In additional to position x, the PINN also takes I_o as an input, 
so that the PINN can learn the solution of the RTE for different initial conditions.

Version three: The absorption and emission functions are not fixed. 
The absorption center x_c and emission frequency are added as conditioning inputs.

Version four: use L-BFGS optimizer instead of Adam optimizer. 
The default L-BFGS optimizer in PyTorch doesn't mini-batch, 
so we implement a mini-batch version.
This version also read the trained model from version three, and continue training with L-BFGS optimizer. 
The loss term of PDE is also increased.
'''

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt
# This is for the progress bar.
from tqdm.auto import tqdm
import time
from scipy.interpolate import CubicSpline
from pathlib import Path
import argparse
from analytic_solve import RT_1D_solver
from NN_classes import FCN_four


random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.set_default_dtype(torch.float32)

run_in_background = True # make tqdm to be silent mode
show_model_summary = False # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
allow_device = True  # Set to False if you want to force using CPU.
use_pin_memory = True  # Set to True if you use GPU. False for CPU and MPS.
n_epochs = 1000 # The number of training epochs for classifier. 
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 100  # Mini-batch size for supervised and physics updates.
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
alpha_phys = 5e-4  # Weight for the physics loss term. 

# code_to_physics_length * boxsize = physical length of the problem domain. This is used to scale the spatial variable x to the physical length.
code_to_physics_length = 10.0 
Boxsize_data = 1.0   # Code length
N_cells_data = 4096
d_x = Boxsize_data / N_cells_data
Boxsize_training = 0.25
n_points_pde_loss = 50  # Number of data points for the pde loss term
n_points_value_loss = 50  # Number of data points for the value loss term
n_points_I_o = 20  # Number of data points for the initial intensity I_o
n_points_x_c = 5  # Number of data points for the absorption center x_c
n_points_frequency = 10  # Number of data points for the emission frequency

parser = argparse.ArgumentParser(description="Load a trained PINN v3 checkpoint.")
parser.add_argument("-I", "--checkpoint", type=Path, default="./pinn_v3_best_model_epoch_1600.pth", help="Path to a .pth checkpoint.")
# parser.add_argument("--checkpoint-dir", type=Path, default=Path("."), help="Directory to search for pinn_v3_best_model_epoch_*.pth when --checkpoint is not provided.")
parser.add_argument("-O", "--output-dir", type=Path, default=Path("./"), help="Directory where checkpoints are saved.")
parser.add_argument("-P", "--plot-dir", type=Path, default=Path("./trained_result_plots_v4"), help="Directory where plots are saved.")
args = parser.parse_args()


def numerical_intensity_cubic_spline(I_o: float | np.float64, kappa: np.ndarray, j_emit: np.ndarray,
                                 delta_x: float | np.float64 | np.ndarray) -> CubicSpline:
    '''
    # This func returns the numerical value of the intensity at locattion x.
    # The input absorption and emission are discretized in the spatial domain, and the output is the intensity at location x.
    # Since we don't know the analytic solution of the RTE, we use numerical integration to get the solution at fixed x.
    # Then we use cubic spline interpolation to get the solution at any x.
    Parameters:
    I_o : float | np.float64
        Initial intensity.
    kappa : ndarray
        Absorption coefficient array of shape (N,).
    j_emit : ndarray
        Emission coefficient array of shape (N,).
    delta_x : float | np.float64 | np.ndarray
        Spatial step size array of shape (N,).
        If delta_x is a single float, it will be used as the uniform spatial step size for all cells.
    Returns:
    CubicSpline
        Callable object that can be used to evaluate the intensity at any location x.
    '''
    discrete_I = RT_1D_solver(I_o, kappa, j_emit, delta_x, return_last_I=False)
    if isinstance(delta_x, float) or isinstance(delta_x, np.float64):
        # If delta_x is a single float, we can use cubic spline interpolation to get the intensity at location x.
        # location of the left boundary of each cell
        x_boundaries = np.arange(len(discrete_I)) * delta_x  # Assuming uniform delta_x for simplicity
        intensity_cubic_spline = CubicSpline(x_boundaries, discrete_I)
        # I_x = intensity_cubic_spline(x)
    else:
        # If delta_x is an array, we need to calculate the cumulative sum of delta_x to get the location of the left boundary of each cell.
        x_boundaries = np.cumsum(delta_x) - delta_x  # The left boundary 
        intensity_cubic_spline = CubicSpline(x_boundaries, discrete_I)
        # I_x = intensity_cubic_spline(x)

    return intensity_cubic_spline

def kappa_func(x: float | np.float64 | torch.Tensor, x_c: float | np.float64 | torch.Tensor=1.0):
    '''
    # This func returns the value of the absorption coefficient at location x.
    # The absorption coefficient is a function of the spatial variable x.
    # propotional to 1 + exp(-x^2/10), which is a simple example of a spatially varying absorption coefficient.
    Parameters:
    x : float | np.float64 
        Spatial variable.
    x_c : float | np.float64 | torch.Tensor
        Center of the gaussian part.
    Returns:
    float | torch.float32
        Absorption coefficient at location x.
    '''
    if isinstance(x, torch.Tensor):
        x_c = torch.as_tensor(x_c, device=x.device, dtype=x.dtype)
        return code_to_physics_length * 0.8 * (1.0 + 5.0 * torch.exp(-((code_to_physics_length * (x - x_c)) / 3.16228) ** 2.0))
    return code_to_physics_length * 0.8 * (1.0 + 5.0 * np.exp(-((code_to_physics_length * (x - x_c)) / 3.16228) ** 2.0))  # Example function, can be changed

def j_emit_func(x: float | np.float64 | torch.Tensor, frequency: float | np.float64 | torch.Tensor=1.0):
    '''
    # This func returns the value of the emission coefficient at location x.
    # The emission coefficient is a function of the spatial variable x.
    # propotional to 1 + cos(2*pi*x), which is a simple example of a spatially varying emission coefficient.
    Parameters:
    x : float | np.float64
        Spatial variable.
    frequency : float | np.float64 | torch.Tensor
        Frequency of the emission.
    Returns:
    float
        Emission coefficient at location x.
    '''
    # shift the frequency since the normalised input x is in the range of [0.05, 1.05], and the frequency is in the range of [0.5, 1.5]
    frequency = frequency + 0.45 
    if isinstance(x, torch.Tensor):
        frequency = torch.as_tensor(frequency, device=x.device, dtype=x.dtype)
        return code_to_physics_length * 0.5 * (1 + torch.cos(2 * torch.pi * frequency * code_to_physics_length * x))
    return code_to_physics_length * 0.5 * (1 + np.cos(2 * np.pi * frequency * code_to_physics_length * x))  # Example function, can be changed


# Construct the absorption and emission coefficient arrays for the entire spatial domain
I_o = 1.0
x_location = np.arange(N_cells_data) * d_x + 0.5 * d_x  # location of the center of each cell
kappa_array = kappa_func(x_location, x_c=0.2)
j_emit_array = j_emit_func(x_location, frequency=1.0)

def _to_scalar(value: float | np.float64 | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def build_condition_inputs(x_values: torch.Tensor, I_o_value: torch.Tensor,
                           x_c_value: torch.Tensor, freq_value: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        x_values,
        torch.full_like(x_values, _to_scalar(I_o_value)),
        torch.full_like(x_values, _to_scalar(x_c_value)),
        torch.full_like(x_values, _to_scalar(freq_value)),
    ], dim=1)


def reference_intensity_from_conditions(x_values: torch.Tensor, I_o_value: torch.Tensor,
                                        x_c_value: torch.Tensor, freq_value: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the reference intensity for one set of conditioning values.
    This keeps the target-generation logic separate from the model-facing helper.
    """
    I_o_scalar = _to_scalar(I_o_value)
    x_c_scalar = _to_scalar(x_c_value)
    freq_scalar = _to_scalar(freq_value)
    kappa_array = kappa_func(x_location, x_c_scalar)
    j_emit_array = j_emit_func(x_location, freq_scalar)
    spline = numerical_intensity_cubic_spline(I_o_scalar, kappa_array, j_emit_array, d_x)
    x_np = x_values.detach().cpu().numpy().reshape(-1)
    y_np = spline(x_np)
    return torch.from_numpy(np.asarray(y_np)).to(device=x_values.device, dtype=torch.float32).reshape_as(x_values)


def build_conditioned_dataset(x_values: torch.Tensor, I_o_values: torch.Tensor,
                              x_c_values: torch.Tensor, freq_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a Cartesian-product dataset for the supervised loss.
    Each input row is [x, I_o, x_c, frequency].

    Expected shapes:
    x_values: [N, 1]
    I_o_values: [M, 1]
    x_c_values: [L, 1]
    freq_values: [P, 1]
    input_rows: [N, M, L, P, 4]
    target_rows: [N, M, L, P, 1]
    Returned inputs: [N, M, L, P, 4]
    Returned targets: [N, M, L, P, 1]
    """
    x_block = x_values[:, None, None, None, :]
    I_o_block = I_o_values[None, :, None, None, :]
    x_c_block = x_c_values[None, None, :, None, :]
    freq_block = freq_values[None, None, None, :, :]

    input_rows = torch.cat(
        [
            x_block.expand(-1, I_o_values.shape[0], x_c_values.shape[0], freq_values.shape[0], -1),
            I_o_block.expand(x_values.shape[0], -1, x_c_values.shape[0], freq_values.shape[0], -1),
            x_c_block.expand(x_values.shape[0], I_o_values.shape[0], -1, freq_values.shape[0], -1),
            freq_block.expand(x_values.shape[0], I_o_values.shape[0], x_c_values.shape[0], -1, -1),
        ], dim=-1,
    )

    target_rows = torch.empty(
        x_values.shape[0], I_o_values.shape[0], x_c_values.shape[0], freq_values.shape[0], 1,
        dtype=torch.float32,
        device=x_values.device,
    )
    for freq_index, freq_value in enumerate(freq_values.view(-1)):
        for x_c_index, x_c_value in enumerate(x_c_values.view(-1)):
            for I_o_index, I_o_value in enumerate(I_o_values.view(-1)):
                target_rows[:, I_o_index, x_c_index, freq_index, :] = reference_intensity_from_conditions(
                    x_values, I_o_value, x_c_value, freq_value
                ).view(-1, 1)

    return input_rows, target_rows


x_location_truth = torch.linspace(0, Boxsize_data, N_cells_data).view(-1, 1)  # Position points for numerical solution
# Create I_o, x_c, and frequency samples for the training data.
I_o_training = torch.linspace(0.1, 1.0, n_points_I_o).view(-1, 1)
x_c_training = torch.linspace(0.0, 0.9, n_points_x_c).view(-1, 1)
frequency_training = torch.linspace(0.05, 1.05, n_points_frequency).view(-1, 1)
x_location_training = torch.linspace(0, Boxsize_training, n_points_value_loss).view(-1, 1)  # Position points for training

# train_inputs: (n_points_value_loss, n_points_I_o, n_points_x_c, n_points_frequency, 4)
# I_training: (n_points_value_loss, n_points_I_o, n_points_x_c, n_points_frequency, 1)
train_inputs, I_training = build_conditioned_dataset(
    x_location_training, I_o_training, x_c_training, frequency_training
)

x_location_pde = torch.linspace(0, 2 * Boxsize_training, n_points_pde_loss).view(-1, 1)  # sample locations over the problem domain
# physics_inputs: (n_points_pde_loss, n_points_I_o, n_points_x_c, n_points_frequency, 4)
physics_inputs, _ = build_conditioned_dataset(
    x_location_pde, I_o_training, x_c_training, frequency_training
)

validate_I_o = torch.tensor([[1.0]], dtype=torch.float32)
validate_x_c = torch.tensor([[0.2]], dtype=torch.float32)
validate_frequencies = torch.linspace(0.05, 1.05, 5).view(-1, 1)
x_location_validate = x_location_truth[0:2048:64]  # Position points for validation
# validate_inputs_list: 5 tensors of shape (len(x_location_validate), 4)
# I_validate_list: 5 tensors of shape (len(x_location_validate), 1)
validate_inputs_list = [
    build_condition_inputs(x_location_validate, validate_I_o, validate_x_c, freq_value)
    for freq_value in validate_frequencies
]
I_validate_list = [
    reference_intensity_from_conditions(x_location_validate, validate_I_o, validate_x_c, freq_value).view(-1, 1)
    for freq_value in validate_frequencies
]

I_validate_numerical_list = [
    reference_intensity_from_conditions(x_location_truth, validate_I_o, validate_x_c, freq_value).view(-1, 1)
    for freq_value in validate_frequencies
]

# check array shapes
print(f"train_inputs shape: {train_inputs.shape}, I_training shape: {I_training.shape}")
print(f"physics_inputs shape: {physics_inputs.shape}")
print(f"validate_inputs count: {len(validate_inputs_list)}, validation x shape: {validate_inputs_list[0].shape}")

# Automatically choose the device to use.
if allow_device:
# Move the network to the appropriate device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        device = torch.device("cuda")
        print("Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_pin_memory = False  # MPS does not support pin_memory
        num_workers = 0  # MPS does not support multi-process data loading
        print("Using MPS")
    else:
        device = torch.device("cpu")
        use_pin_memory = False # pin_memory is not useful for CPU
        num_workers = 0  # CPU does not support multi-process data loading
        print("Using CPU")
else:
    device = torch.device("cpu")
    use_pin_memory = False # pin_memory is not useful for CPU
    num_workers = 0  # CPU does not support multi-process data loading
    print("Using CPU")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


checkpoint = args.checkpoint

if checkpoint is not None and checkpoint.exists():
    model = FCN_four(N_hidden=64, dropout_rate=0.0)
    if show_model_summary:
        print(model)
        summary(model, input_size=(batch_size, 4))
        print("Code execution stopped here. Please set 'show_model_summary' to False to continue.")
        exit()
    model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"Loaded checkpoint: {checkpoint}")
else:
    print("No checkpoint found. Terminating.")
    exit()

output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
plot_dir = args.plot_dir
plot_dir.mkdir(parents=True, exist_ok=True)

train_dataset = torch.utils.data.TensorDataset(train_inputs.reshape(-1, 4), I_training.reshape(-1, 1))
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=use_pin_memory,
)

physics_dataset = torch.utils.data.TensorDataset(physics_inputs.reshape(-1, 4))
physics_dataloader = torch.utils.data.DataLoader(
    physics_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=use_pin_memory,
)


criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=25,
    line_search_fn="strong_wolfe",
)
best_val_loss = float('inf')  # Initialize best validation loss to infinity
# Training loop
model.train()
train_losses = []
val_losses = []
val_epochs = []
best_epoch = 0

start_time = time.time()

for epoch in tqdm(range(n_epochs), desc="Training", disable=run_in_background):
    epoch_loss = 0.0
    num_samples = 0
    physics_batches = iter(physics_dataloader)
    for input_batch, target_batch in train_dataloader:
        try:
            (physics_batch,) = next(physics_batches)
        except StopIteration:
            physics_batches = iter(physics_dataloader)
            (physics_batch,) = next(physics_batches)

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        physics_batch = physics_batch.to(device)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            I_pred = model(input_batch)
            loss_data = criterion(I_pred, target_batch)
            alpha = min(1.0, epoch / 250) * alpha_phys # slowly increase the weight of the physics loss term
            physics_batch_local = physics_batch.detach().requires_grad_(True)
            I_val = model(physics_batch_local)
            dx  = torch.autograd.grad(I_val, physics_batch_local, torch.ones_like(I_val), create_graph=True)[0][:, :1] # computes dI/dx
            kappa_val = kappa_func(
                physics_batch_local[:, :1],
                physics_batch_local[:, 2:3],
            )
            j_emit_val = j_emit_func(
                physics_batch_local[:, :1],
                physics_batch_local[:, 3:4],
            )
            pde = dx + kappa_val * I_val - j_emit_val # computes the residual of dI/dx + kappa(x) I - j_emit(x) = 0
            loss_phys = alpha * torch.mean(pde ** 2)

            loss = loss_data + loss_phys
            loss.backward()
            return loss

        loss_value = optimizer.step(closure)
        batch_loss = float(loss_value.item()) if torch.is_tensor(loss_value) else float(loss_value)
        batch_size_actual = input_batch.size(0)
        epoch_loss += batch_loss * batch_size_actual
        num_samples += batch_size_actual

    epoch_loss /= num_samples
    train_losses.append(epoch_loss)
    tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")
    # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

    # validation every 200 epochs, also make plot
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            validation_curves = []
            for freq_value, validate_inputs, I_validate in zip(validate_frequencies, validate_inputs_list, I_validate_list):
                validate_inputs_device = validate_inputs.to(device)
                I_validate_device = I_validate.to(device)
                I_pred_validate = model(validate_inputs_device)
                freq_loss = criterion(I_pred_validate, I_validate_device).item()
                val_loss_sum += freq_loss
                validation_curves.append((freq_value, I_validate, I_pred_validate.cpu()))

            val_loss = val_loss_sum / len(validate_inputs_list)

            val_losses.append(val_loss)
            val_epochs.append(epoch + 1)
            if val_loss < best_val_loss and epoch > 50:
                best_val_loss = val_loss
                print(f"New best model found at epoch {epoch+1} with validation loss {val_loss:.6f}. Saving the model.")
                # save the best model
                torch.save(model.state_dict(), output_dir / f'pinn_v4_best_model_epoch_{epoch+1}.pth')
                best_epoch = epoch + 1
            
            # I_numerical = reference_intensity_from_conditions(x_location_truth, validate_I_o, validate_x_c, validate_frequencies[0]).view(-1, 1)
            # plot_truth_inputs = build_condition_inputs(x_location_truth, validate_I_o, validate_x_c, validate_frequencies[0])
            # I_model = model(plot_truth_inputs.to(device)).cpu()  # Model prediction for the numerical position points
            # plot_train_inputs = build_condition_inputs(x_location_training, validate_I_o, validate_x_c, validate_frequencies[0])
            # I_training_pred = model(plot_train_inputs.to(device)).cpu()  # Model prediction for the training position points

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")
        fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
        for axis, (freq_value, I_numerical_freq, I_model_freq), I_validate_numerical in zip(axes, validation_curves, I_validate_numerical_list):
            axis.plot(x_location_truth.cpu(), I_validate_numerical.cpu(), '-b', label='Numerical Solution')
            axis.plot(x_location_validate.cpu(), I_model_freq, '--or', label='Model Prediction')
            # axis.plot(x_location_training.cpu(), I_training_pred.cpu(), 'go', label='Training Data')
            axis.set_xlim(0, Boxsize_data)
            axis.set_ylabel('Intensity')
            axis.set_title(f'Frequency = {_to_scalar(freq_value + 0.45):.2f}')
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel('Path')
        axes[0].legend(loc='best')
        fig.suptitle(f'Epoch {epoch+1} | validation frequency sweep')
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(plot_dir / f'pinn_v4_epoch_{epoch+1}.png')
        plt.close(fig)
        model.train()
endtime = time.time()
print(f"Training completed in {endtime - start_time:.2f} seconds.")
