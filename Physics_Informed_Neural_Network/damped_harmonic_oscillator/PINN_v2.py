'''
Purpose: This code implements a Physics-Informed Neural Network (PINN) to solve a partial differential equation (PDE). 
The PDE we are solving is the underdamped harmonic oscillator, given by the equation:
m * d^2u/dt^2 + c * du/dt + k * u = 0
where m is the mass, c is the damping coefficient, k is the spring constant, and u is the displacement. 
For damped harmonic oscillator, coefficients need to satisfy the condition c^2 < 4*m*k for underdamped case, c^2 = 4*m*k for critically damped case, and c^2 > 4*m*k for overdamped case.
In this case, the general solution to the underdamped harmonic oscillator can be expressed as:
u(t) = A * exp(-c*t/(2*m)) * cos(w*t + phi)
where A is the amplitude, w is the damped natural frequency given by w = sqrt(4*m*k - c^2) / (2*m),
and phi is the phase angle. 
Here, we choose u(t=0) = 1 and du/dt(t=0) = 0 as the initial conditions, which leads to 
A = 1/cos(phi) and phi = arctan(-delta/w) where delta = c/(2*m).
We define w_o^2 = k/m as the natural frequency squared, and delta = c/(2*m) as the damping ratio.
So, w = sqrt(w_o^2 - delta^2) is the oscillation frequency.

In version 2, we introduce the PDE loss term in the training process.
The time range of the PDE is 2 seconds, which is the double of the time range for training data.

Note, the data points used to train the network are fixed.
So, model sees the same data points in each epoch. 
Therefore, we assume batch_size is one and no parallel strategy is needed for training.

Version: 2
Author: Dr. Ka Hou Leong
Date: 22/4/2026
'''

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt
# This is for the progress bar.
from tqdm.auto import tqdm
import time


random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

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

delta = 2.0  # Damping ratio
w_o = 20.0  # Natural frequency
mu, k = 2*delta, w_o**2

duration_analytical = 5.0  # Duration of the time series for the analytical solution
n_points_analytical = 2500  # Number of data points for the analytical solution
n_points_physics = 50  # Number of data points for the physics loss term
duration_training = 1.0  # Duration of the time series used for training
n_points_training = 50 * batch_size  # Number of data points used for training (N times of batch size)

def oscillator_solution(delta, w_o, t):
    # The analytical solution for the underdamped harmonic oscillator
    # we use torch functions to ensure compatibility with PyTorch tensors

    # check if the system is underdamped
    assert delta < w_o, "The system is not underdamped. Please ensure that delta < w_o."

    w = np.sqrt(w_o**2 - delta**2)  # Damped natural frequency
    phi = np.arctan(-delta / w)  # Phase angle
    amplitude = 1 / np.cos(phi)  # Amplitude based on initial conditions
    exp_term = torch.exp(-delta * t)  # Exponential decay term
    cos_term = torch.cos(w * t + phi)  # Oscillatory term
    y = amplitude * exp_term * cos_term  # displacement as a function of time
    return y


class FCN(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.1):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(1, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        

    def forward(self, x):
        return self.network(x)
    
t_analytical = torch.linspace(0, duration_analytical, n_points_analytical).view(-1, 1)  # Time points for analytical solution
u_analytical = oscillator_solution(delta, w_o, t_analytical).view(-1, 1)  # Analytical solution at the specified time points
t_training = torch.linspace(0, duration_training, n_points_training).view(-1, 1)  # Time points for training
u_training = oscillator_solution(delta, w_o, t_training).view(-1, 1)  # Analytical solution at the training time points
t_physics = torch.linspace(0, 2*duration_training, n_points_physics).view(-1, 1).requires_grad_(True)  # sample locations over the problem domain

t_validate = t_analytical[0:500:20]  # Time points for validation (every 20th point from the first 1000 points of the analytical solution)
u_validate = u_analytical[0:500:20]  # Analytical solution at the validation time points

print(t_analytical.shape, u_analytical.shape)
print(t_training.shape, u_training.shape)
print(t_validate.shape, u_validate.shape)

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
train_dataset = torch.utils.data.TensorDataset(t_training, u_training)
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
t_physics_device = t_physics.to(device)
start_time = time.time()
for epoch in tqdm(range(n_epochs), desc="Training", disable=run_in_background):
    epoch_loss = 0.0
    num_sample = 0
    for t_batch, u_batch in train_dataloader:
        t_batch, u_batch = t_batch.to(device), u_batch.to(device)
        optimizer.zero_grad()
        u_pred = model(t_batch)
        loss_data = criterion(u_pred, u_batch)
        alpha = min(1.0, epoch / 2500) * alpha_phys # slowly increase the weight of the physics loss term
        # compute the "physics loss"    
        u_phy = model(t_physics_device)
        dx  = torch.autograd.grad(u_phy, t_physics_device, torch.ones_like(u_phy), create_graph=True)[0] # computes dy/dx
        dx2 = torch.autograd.grad(dx,  t_physics_device, torch.ones_like(dx),  create_graph=True)[0] # computes d^2y/dx^2
        physics = dx2 + mu * dx + k * u_phy # computes the residual of the 1D harmonic oscillator differential equation
        loss_phys = alpha * torch.mean(physics ** 2)

        loss = loss_data + loss_phys
        loss.backward()
        optimizer.step()
        num_sample += t_batch.size(0)
        epoch_loss += loss.item() * t_batch.size(0)  # Accumulate the loss for the epoch
    epoch_loss /= num_sample  # Average loss for the epoch
    train_losses.append(epoch_loss)
    tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")
    # print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

    # validation every 200 epochs, also make plot
    if (epoch + 1) % 200 == 0:
        model.eval()
        with torch.no_grad():
            t_validate, u_validate = t_validate.to(device), u_validate.to(device)
            u_pred_validate = model(t_validate)
            val_loss = criterion(u_pred_validate, u_validate).item()

            # compute the physics loss for validation
            ''' 
            # ignore PDE loss
            u_phy = model(t_physics)
            dx  = torch.autograd.grad(u_phy, t_physics, torch.ones_like(u_phy), create_graph=True)[0] # computes dy/dx
            dx2 = torch.autograd.grad(dx,  t_physics, torch.ones_like(dx),  create_graph=True)[0] # computes d^2y/dx^2
            physics = dx2 + mu * dx + k * u_phy # computes the residual of the 1D harmonic oscillator differential equation
            loss_phys = alpha_phys * torch.mean(physics ** 2)

            val_loss = val_loss_data + loss_phys.item()
            '''

            val_losses.append(val_loss)
            val_epochs.append(epoch + 1)
            if val_loss < best_val_loss and epoch > 1000:
                best_val_loss = val_loss
                print(f"New best model found at epoch {epoch+1} with validation loss {val_loss:.6f}. Saving the model.")
                # save the best model
                torch.save(model.state_dict(), f'pinn_v2_best_model_epoch_{epoch+1}.pth')
                best_epoch = epoch + 1
            u_model = model(t_analytical.to(device)).cpu()  # Model prediction for the analytical time points
            u_training_pred = model(t_training.to(device)).cpu()  # Model prediction for the training time points

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")
        plt.figure(figsize=(10, 6))
        plt.plot(t_analytical.cpu(), u_analytical, '-b', label='Analytical Solution')
        plt.plot(t_analytical.cpu(), u_model, '--r', label='Model Prediction')
        plt.plot(t_training.cpu(), u_training_pred, 'go', label='Training Data')
        plt.xlim(0, duration_analytical)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('Time')
        plt.ylabel('Displacement')
        plt.legend()
        plt.title(f'Epoch {epoch+1}')
        plt.savefig(f'trained_result_plots_v2/pinn_v2_epoch_{epoch+1}.png')
        plt.close()
        model.train()

#plot the training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), train_losses, 'r-',label='Training Loss')
plt.plot(val_epochs, val_losses, 'bo', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.savefig('training_validation_loss_curves_v2.png')
# save the trained model
torch.save(model.state_dict(), f'pinn_v2_trained_model_epoch_{n_epochs}.pth')
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# load the best model and make prediction
if best_epoch > 0:
    best_model = FCN(N_hidden=64, dropout_rate=0.1).to(device)
    best_model.load_state_dict(torch.load(f'pinn_v2_best_model_epoch_{best_epoch}.pth'))
    best_model.eval()
    with torch.no_grad():
        u_model_best = best_model(t_analytical.to(device)).cpu()  # Model prediction for the analytical time points
        error_best = torch.abs(u_model_best - u_analytical)
    plt.figure(figsize=(10, 6))
    plt.plot(t_analytical.cpu(), error_best, '-m', label='Absolute Error of Best Model')
    plt.xlim(0, duration_analytical)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.title(f'Absolute Error of Best Model at Epoch {best_epoch}')
    plt.savefig(f'absolute_error_best_model_epoch_{best_epoch}.png')

    # PDE residual for the best model (requires grad-enabled input tensor)
    t_physics_eval = t_physics.to(device).detach().requires_grad_(True)
    u_phy_best = best_model(t_physics_eval)
    dx_best  = torch.autograd.grad(u_phy_best, t_physics_eval, torch.ones_like(u_phy_best), create_graph=True)[0] # computes dy/dx
    dx2_best = torch.autograd.grad(dx_best,  t_physics_eval, torch.ones_like(dx_best),  create_graph=True)[0] # computes d^2y/dx^2
    physics_residual_best = dx2_best + mu * dx_best + k * u_phy_best # computes the residual of the 1D harmonic oscillator differential equation
    plt.figure(figsize=(10, 6))
    plt.plot(t_physics.cpu(), physics_residual_best.cpu().detach(), '-c', label='PDE Residual of Best Model')
    plt.xlim(0, duration_analytical)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('PDE Residual')
    plt.title(f'PDE Residual of Best Model at Epoch {best_epoch}')
    plt.savefig(f'pde_residual_best_model_epoch_{best_epoch}.png')
