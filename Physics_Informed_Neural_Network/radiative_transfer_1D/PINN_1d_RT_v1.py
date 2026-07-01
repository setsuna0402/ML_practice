'''
This is the simplest example of a Physics-Informed Neural Network (PINN) for solving the 1D Radiative Transfer Equation (RTE).
Here, absorption and emission are considered, but scattering is neglected. 
The values of them follow the analytic functions of the spatial variable x, 
and the PINN does not see them. The PINN is trained to learn the solution of the RTE.

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
from analytic_solve import RT_1D_solver
from NN_classes import FCN


random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.set_default_dtype(torch.float32)

run_in_background = False # make tqdm to be silent mode
show_model_summary = False # Whether to print the model summary. You may set it to False if you don't want to see the model summary.
allow_device = False  # Set to False if you want to force using CPU.
use_pin_memory = True  # Set to True if you use GPU. False for CPU and MPS.
n_epochs = 15000 # The number of training epochs for classifier. 
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 1
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
alpha_phys = 1e-4  # Weight for the physics loss term. 

# code_to_physics_length * boxsize = physical length of the problem domain. This is used to scale the spatial variable x to the physical length.
code_to_physics_length = 10.0 
Boxsize_data = 1.0   # Code length
N_cells_data = 4096
d_x = Boxsize_data / N_cells_data
Boxsize_training = 0.25
n_points_pde_loss = 50  # Number of data points for the pde loss term
n_points_value_loss = 50  # Number of data points for the value loss term


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

def kappa_func(x: float | np.float64 | torch.Tensor):
    '''
    # This func returns the value of the absorption coefficient at location x.
    # The absorption coefficient is a function of the spatial variable x.
    # propotional to 1 + exp(-x^2/10), which is a simple example of a spatially varying absorption coefficient.
    Parameters:
    x : float | np.float64
        Spatial variable.
    Returns:
    float | torch.float32
        Absorption coefficient at location x.
    '''
    if isinstance(x, torch.Tensor):
        return code_to_physics_length * 0.8 * (1 + torch.exp(-(code_to_physics_length * x) ** 2.0 / 10.0))
    return code_to_physics_length * 0.8 * (1 + np.exp(-(code_to_physics_length * x) ** 2.0 / 10.0))  # Example function, can be changed

def j_emit_func(x: float | np.float64 | torch.Tensor):
    '''
    # This func returns the value of the emission coefficient at location x.
    # The emission coefficient is a function of the spatial variable x.
    # propotional to 1 + cos(2*pi*x), which is a simple example of a spatially varying emission coefficient.
    Parameters:
    x : float | np.float64
        Spatial variable.
    Returns:
    float
        Emission coefficient at location x.
    '''
    if isinstance(x, torch.Tensor):
        return code_to_physics_length * 0.5 * (1 + torch.cos(2 * torch.pi * code_to_physics_length * x))
    return code_to_physics_length * 0.5 * (1 + np.cos(2 * np.pi * code_to_physics_length * x))  # Example function, can be changed


# Construct the absorption and emission coefficient arrays for the entire spatial domain
I_o = 1.0
x_location = np.arange(N_cells_data) * d_x + 0.5 * d_x  # location of the center of each cell
kappa_array = kappa_func(x_location)
j_emit_array = j_emit_func(x_location)
I_test= RT_1D_solver(I_o, kappa_array, j_emit_array, d_x, return_last_I=False)
I_spline = numerical_intensity_cubic_spline(I_o, kappa_array, j_emit_array, d_x)
'''
test_x = np.linspace(0, Boxsize_data, 50)
I_x = I_spline(np.arange(N_cells_data)[::10] * d_x + 0.5 * d_x)  # Evaluate the cubic spline at the center of each cell

plt.plot(np.arange(N_cells_data) * d_x, I_test, label='Numerical Solution', color='blue')
plt.plot(np.arange(N_cells_data)[::10] * d_x + 0.5 * d_x, I_x, 'r--', label='Cubic Spline')
plt.xlabel('x')
plt.ylabel('Intensity I(x)')
plt.title('Numerical Solution of the 1D Radiative Transfer Equation')
plt.legend()
plt.grid()
plt.show()
exit()
# '''

# Make the I_spline callable object to be used for torch tensor input.
# This is necessary because the PINN will take torch tensors as input, and we need to evaluate the intensity at those locations.
def I_spline_callable(x: torch.Tensor) -> torch.Tensor:
    '''
    # This func returns the value of the intensity at location x.
    # The input x is a torch tensor, and the output is also a torch tensor.
    # The input x can be a single value or a batch of values.
    Parameters:
    x : torch.Tensor
        Spatial variable. Can be a single value or a batch of values.
    Returns:
    torch.Tensor
        Intensity at location x. Same shape as input x.
    '''
    x_np = x.detach().cpu().numpy()  # Convert to numpy array
    I_x_np = I_spline(x_np)  # Evaluate the cubic spline at location x
    I_x = torch.from_numpy(I_x_np).to(device=x.device, dtype=torch.float32)  # Convert back to torch tensor in float32
    return I_x

x_location_truth = torch.linspace(0, Boxsize_data, N_cells_data).view(-1, 1)  # Position points for numerical solution
I_truth = I_spline_callable(x_location_truth).view(-1, 1)  # Numerical solution at the specified position points
x_location_training = torch.linspace(0, Boxsize_training, n_points_value_loss).view(-1, 1)  # Position points for training
I_training = I_spline_callable(x_location_training).view(-1, 1)  # Numerical solution at the training position points
x_location_pde = torch.linspace(0, 2*Boxsize_training, n_points_pde_loss).view(-1, 1).requires_grad_(True)  # sample locations over the problem domain

x_location_validate = x_location_truth[0:1024:64]  # Position points for validation
I_validate = I_truth[0:1024:64]  # numerical solution at the validation position points


# check array shapes
print(f"x_location_truth shape: {x_location_truth.shape}, I_truth shape: {I_truth.shape}")
print(f"x_location_training shape: {x_location_training.shape}, I_training shape: {I_training.shape}")
print(f"x_location_pde shape: {x_location_pde.shape}")
print(f"x_location_validate shape: {x_location_validate.shape}, I_validate shape: {I_validate.shape}")

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
train_dataset = torch.utils.data.TensorDataset(x_location_training, I_training)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)

model = FCN(N_hidden=64, dropout_rate=0.0).to(device)
if show_model_summary:
    print(model)
    summary(model, input_size=(batch_size, 1))
    print("Code execution stopped here. Please set 'show_model_summary' to False to continue.")
    exit()


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val_loss = float('inf')  # Initialize best validation loss to infinity
# Training loop
model.train()
train_losses = []
val_losses = []
val_epochs = []
best_epoch = 0
# Reuse one tensor instance for PDE residual derivatives.
x_location_pde_device = x_location_pde.to(device)
start_time = time.time()

for epoch in tqdm(range(n_epochs), desc="Training", disable=run_in_background):
    epoch_loss = 0.0
    num_sample = 0
    for x_batch, I_batch in train_dataloader:
        x_batch, I_batch = x_batch.to(device), I_batch.to(device)
        optimizer.zero_grad()
        # compute the "data loss"
        I_pred = model(x_batch)
        loss_data = criterion(I_pred, I_batch)

        # compute the "pde loss"    
        alpha = min(1.0, epoch / 2500) * alpha_phys # slowly increase the weight of the physics loss term
        I_val = model(x_location_pde_device)
        dx  = torch.autograd.grad(I_val, x_location_pde_device, torch.ones_like(I_val), create_graph=True)[0] # computes dI/dx
        kappa_val = kappa_func(x_location_pde_device)
        j_emit_val = j_emit_func(x_location_pde_device)
        pde = dx + kappa_val * I_val - j_emit_val # computes the residual of dI/dx + kappa(x) I - j_emit(x) = 0
        loss_phys = alpha * torch.mean(pde ** 2)

        loss = loss_data + loss_phys
        loss.backward()
        optimizer.step()
        num_sample += x_batch.size(0)
        epoch_loss += loss.item() * x_batch.size(0)  # Accumulate the loss for the epoch
    epoch_loss /= num_sample  # Average loss for the epoch
    train_losses.append(epoch_loss)
    tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")
    # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

    # validation every 200 epochs, also make plot
    if (epoch + 1) % 200 == 0:
        model.eval()
        with torch.no_grad():
            x_validate_device, I_validate = x_location_validate.to(device), I_validate.to(device)
            I_pred_validate = model(x_validate_device)
            val_loss = criterion(I_pred_validate, I_validate).item()


            val_losses.append(val_loss)
            val_epochs.append(epoch + 1)
            if val_loss < best_val_loss and epoch > 1000:
                best_val_loss = val_loss
                print(f"New best model found at epoch {epoch+1} with validation loss {val_loss:.6f}. Saving the model.")
                # save the best model
                torch.save(model.state_dict(), f'pinn_v1_best_model_epoch_{epoch+1}.pth')
                best_epoch = epoch + 1
            I_model = model(x_location_truth.to(device)).cpu()  # Model prediction for the numerical position points
            I_training_pred = model(x_location_training.to(device)).cpu()  # Model prediction for the training position points

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")
        plt.figure(figsize=(10, 6))
        plt.plot(x_location_truth.cpu(), I_truth, '-b', label='Numerical Solution')
        plt.plot(x_location_truth.cpu(), I_model, '--r', label='Model Prediction')
        plt.plot(x_location_training.cpu(), I_training_pred, 'go', label='Training Data')
        plt.xlim(0, Boxsize_data)
        # plt.ylim(-1.2, 1.2)
        plt.xlabel('Path')
        plt.ylabel('Intensity')
        plt.legend()
        plt.title(f'Epoch {epoch+1}')
        plt.savefig(f'trained_result_plots_v1/pinn_v1_epoch_{epoch+1}.png')
        plt.close()
        model.train()
endtime = time.time()
print(f"Training completed in {endtime - start_time:.2f} seconds.")