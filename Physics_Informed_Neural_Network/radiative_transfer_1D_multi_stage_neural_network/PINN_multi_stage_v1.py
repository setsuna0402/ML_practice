'''
This is the simplest example of a Physics-Informed Neural Network (PINN) for solving the 1D Radiative Transfer Equation (RTE).
Here, absorption and emission are considered, but scattering is neglected. 
The values of them follow the analytic functions of the spatial variable x, 
and the PINN does not see them. The PINN is trained to learn the solution of the RTE.

Multi-stage Neural Networks: 
version 7 of the PINN method achieves a good result: the mean absolute error is 2e-5 and  
the maximum absolute error is 5e-3 for the training data.
Now, we want to train a network to learn the residual. 
This is a multi-stage approach, where the first stage is to train a network to learn the solution of the RTE,
and the second stage is to train a network to learn the residual of the RTE.

Here, we will use the trained model from the version 7 of the PINN method to generate the training data for the second stage.
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
from NN_classes import FCN_hard_bc_four, FCN_tiny


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
batch_size = 100
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
alpha_phys = 1e-3  # Weight for the physics loss term. 

# code_to_physics_length * boxsize = physical length of the problem domain. This is used to scale the spatial variable x to the physical length.
code_to_physics_length = 10.0 
Boxsize_data = 1.0   # Code length
N_cells_data = 4096
d_x = Boxsize_data / N_cells_data
Boxsize_training = 0.5
# n_points_pde_loss = 50  # Number of data points for the pde loss term
n_points_value_loss = 50  # Number of data points for the value loss term
n_points_I_o = 20  # Number of data points for the initial intensity I_o
n_points_x_c = 5  # Number of data points for the absorption center x_c
n_points_frequency = 10  # Number of data points for the emission frequency
residual_scaling_factor = 750.0  # Scaling factor for the residual. This is used to scale the residual to a reasonable range for training.
# initial_scaling_factor (kappa) initialise the first layer weights of the second stage network.
# This is used to scale the input to a reasonable range for training.
initial_scaling_factor = 15.0  # kappa >= pi * f_d * sqrt(var), where var = 2 / (dim_input + N_hidden)

parser = argparse.ArgumentParser(description="Optput location.")
parser = argparse.ArgumentParser(description="The irst stage checkpoint and output location.")
parser.add_argument("-I", "--checkpoint", type=Path, default=("./pinn_v7_best_model_epoch_1440.pth"), help="Path to a .pth checkpoint.")
parser.add_argument("-O", "--output-dir", type=Path, default=Path("./trained_result_plots_v1"), help="Directory where checkpoints and plots are saved.")
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

'''
x_location_pde = x_location_training  # Use the same spatial points as the training data.

# physics_inputs: same spatial grid as the training data.
physics_inputs, _ = build_conditioned_dataset(
    x_location_pde, I_o_training, x_c_training, frequency_training
)
'''
validate_I_o = torch.tensor([[0.25], [0.55], [1.0]], dtype=torch.float32)
validate_x_c = torch.tensor([[0.0], [0.25], [0.5]], dtype=torch.float32)
validate_frequencies = torch.linspace(0.05, 1.05, 5).view(-1, 1)
x_location_validate = x_location_truth[0:2048:64]  # Position points for validation
validation_cases = [
    (I_o_value.view(1, 1), x_c_value.view(1, 1), freq_value)
    for I_o_value in validate_I_o
    for x_c_value in validate_x_c
    for freq_value in validate_frequencies
]
# validate_inputs_list: 45 tensors of shape (len(x_location_validate), 4)
# I_validate_list: 45 tensors of shape (len(x_location_validate), 1)
validate_inputs_list = [
    build_condition_inputs(x_location_validate, I_o_value, x_c_value, freq_value)
    for I_o_value, x_c_value, freq_value in validation_cases
]
I_validate_list = [
    reference_intensity_from_conditions(x_location_validate, I_o_value, x_c_value, freq_value).view(-1, 1)
    for I_o_value, x_c_value, freq_value in validation_cases
]

I_validate_numerical_list = [
    reference_intensity_from_conditions(x_location_truth, I_o_value, x_c_value, freq_value).view(-1, 1)
    for I_o_value, x_c_value, freq_value in validation_cases
]

# check array shapes
print(f"train_inputs shape: {train_inputs.shape}, I_training shape: {I_training.shape}")
# print(f"physics_inputs shape: {physics_inputs.shape}")
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

# build the training dataset and dataloader
train_dataset = torch.utils.data.TensorDataset(train_inputs.reshape(-1, 4), I_training.reshape(-1, 1))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)

#load the trained model from the checkpoint
checkpoint = Path(args.checkpoint)
stage_a_model = FCN_hard_bc_four(N_hidden=64)
state = torch.load(checkpoint, map_location=torch.device('cpu')) # load to cpu first, then move to gpu if available
stage_a_model.load_state_dict(state)
stage_a_model.to(device)
stage_a_model.eval()  # Set the model to evaluation mode


stage_b_model = FCN_tiny(N_hidden=32, kappa=initial_scaling_factor).to(device)
if show_model_summary:
    print(stage_b_model)
    summary(stage_b_model, input_size=(batch_size, 4))
    print("Code execution stopped here. Please set 'show_model_summary' to False to continue.")
    exit()

output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(stage_b_model.parameters(), lr=1e-3)
'''
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100, threshold_mode='rel', 
    threshold=0.005, min_lr=1e-7)
'''
# Cosine Annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
best_val_loss = float('inf')  # Initialize best validation loss to infinity
# Training loop
stage_b_model.train()
train_losses = []
val_losses = []
val_epochs = []
best_epoch = 0
# Reuse one tensor instance for PDE residual derivatives.
start_time = time.time()

for epoch in tqdm(range(n_epochs), desc="Training", disable=run_in_background):
    stage_a_model.eval()  # Set the first stage model to evaluation mode
    stage_b_model.train()  # Set the second stage model to training mode
    epoch_loss = 0.0
    num_sample = 0
    num_batches = len(train_dataloader)
    count = 0
    for input_batch, I_batch in train_dataloader:
        count += 1
        # print("Total batches: ", num_batches, "Current batch: ", count)
        input_batch, I_batch = input_batch.to(device), I_batch.to(device)
        physics_batch = input_batch.detach().clone().requires_grad_(True)
        I_pred_stage_a = stage_a_model(input_batch)
        # compute the residual of the RTE using the output of the first stage model
        residual_stage_a = (I_batch - I_pred_stage_a) * residual_scaling_factor  # Scale the residual to a reasonable range for training.
        optimizer.zero_grad()
        # compute the "data loss"
        residual_pred = stage_b_model(input_batch)
        loss_data = criterion(residual_pred, residual_stage_a)

        loss = loss_data
        loss.backward()
        optimizer.step()
        num_sample += input_batch.size(0)
        epoch_loss += loss.item() * input_batch.size(0)  # Accumulate the loss for the epoch
    epoch_loss /= num_sample  # Average loss for the epoch
    # scheduler.step(epoch_loss)  # Update the learning rate scheduler based on the epoch loss
    scheduler.step()  # Update the learning rate scheduler based on the epoch loss
    train_losses.append(epoch_loss)
    tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}, Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6e}")
    # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

    # validation every 200 epochs, also make plot
    if (epoch + 1) % 20 == 0:
        stage_b_model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            validation_curves = []
            for (I_o_value, x_c_value, freq_value), validate_inputs, I_validate in zip(validation_cases, validate_inputs_list, I_validate_list):
                validate_inputs_device = validate_inputs.to(device)
                I_validate_device = I_validate.to(device)
                I_pred_validate = stage_a_model(validate_inputs_device)
                residual_pred_validate = stage_b_model(validate_inputs_device)
                residual_stage_a_validate = (I_validate_device - I_pred_validate) * residual_scaling_factor
                # compute loss for the validation set
                freq_loss = criterion(residual_pred_validate, residual_stage_a_validate).item()
                val_loss_sum += freq_loss
                validation_curves.append((I_o_value, x_c_value, freq_value, residual_stage_a_validate.cpu(), residual_pred_validate.cpu()))

            val_loss = val_loss_sum / len(validate_inputs_list)

            val_losses.append(val_loss)
            val_epochs.append(epoch + 1)
            if (val_loss < best_val_loss and epoch > 50):  # Save the model if it has the best validation loss or every 100 epochs
                best_val_loss = val_loss
                save_dict = {
                    'model_state_dict': stage_b_model.state_dict(),
                    'residual_scaling_factor': residual_scaling_factor,
                    'stage': 2,          # indicate the stage of the model
                    'epoch': epoch + 1,  # epoch
                }
                print(f"New best model found at epoch {epoch+1} with validation loss {val_loss:.6f}. Saving the model.")
                # save the best model
                torch.save(save_dict, output_dir / f'multi_stage_v1_best_model_epoch_{epoch+1}.pth')
                best_epoch = epoch + 1
            if (epoch + 1) % 100 == 0:
                save_dict = {
                    'model_state_dict': stage_b_model.state_dict(),
                    'residual_scaling_factor': residual_scaling_factor,
                    'stage': 2,          # indicate the stage of the model
                    'epoch': epoch + 1,  # epoch
                }
                # save the model every 100 epochs
                torch.save(save_dict, output_dir / f'multi_stage_v1_model_epoch_{epoch+1}.pth')
                print(f"Model saved at epoch {epoch+1} with epoch loss {epoch_loss:.6f}.")

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")
        plot_cases = validation_curves[: len(validate_I_o) * len(validate_x_c)]
        fig, axes = plt.subplots(len(plot_cases), 1, figsize=(10, 3.5 * len(plot_cases)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, (I_o_value, x_c_value, freq_value, residual_truth, residual_model) in zip(axes, plot_cases):
            axis.plot(x_location_validate.cpu(), residual_truth, '-b', label='Stage a Residual')
            axis.plot(x_location_validate.cpu(), residual_model, '--or', label='Residual Prediction')
            axis.set_xlim(0, Boxsize_data)
            axis.set_ylabel('Residual')
            axis.set_title(
                f'I_o = {_to_scalar(I_o_value):.2f}, x_c = {_to_scalar(x_c_value):.2f}, '
                f'Frequency = {_to_scalar(freq_value + 0.45):.2f}'
            )
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel('Path')
        axes[0].legend(loc='best')
        fig.suptitle(f'Epoch {epoch+1} | validation residual sweep')
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(output_dir / f'multi_stage_v1_residual_epoch_{epoch+1}.png')
        plt.close(fig)
        stage_b_model.train()
endtime = time.time()
print(f"Training completed in {endtime - start_time:.2f} seconds.")
