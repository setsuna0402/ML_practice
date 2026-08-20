#!/usr/bin/env python3
"""
Load a trained v6 PINN checkpoint and calculate residuals.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analytic_solve import RT_1D_solver
from NN_classes import FCN_tiny_sine_hard_constraint, FCN_hard_bc_four

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.set_default_dtype(torch.float32)
n_sample = 5

code_to_physics_length = 10.0
Boxsize_data = 1.0
N_cells_data = 4096
d_x = Boxsize_data / N_cells_data
Boxsize_training = 0.5
n_points_value_loss = 50  # Number of data points for the value loss term
n_points_I_o = 20  # Number of data points for the initial intensity I_o
n_points_x_c = 10  # Number of data points for the absorption center x_c
n_points_frequency = 20  # Number of data points for the emission frequency
omega_first=30.0 # For the first sine layer of the second stage network.
omega_hidden=1.0 # For the hidden sine layers of the second stage network.


def numerical_intensity_cubic_spline(I_o: float, kappa: np.ndarray, j_emit: np.ndarray, delta_x: float):
	discrete_I = RT_1D_solver(I_o, kappa, j_emit, delta_x, return_last_I=False)
	x_boundaries = np.arange(len(discrete_I)) * delta_x
	from scipy.interpolate import CubicSpline

	return CubicSpline(x_boundaries, discrete_I)


def kappa_func(x: float | np.ndarray | torch.Tensor, x_c: float | np.ndarray | torch.Tensor = 1.0):
	if isinstance(x, torch.Tensor):
		x_c = torch.as_tensor(x_c, device=x.device, dtype=x.dtype)
		return code_to_physics_length * 0.8 * (1.0 + 5.0 * torch.exp(-((code_to_physics_length * (x - x_c)) / 3.16228) ** 2.0))
	return code_to_physics_length * 0.8 * (1.0 + 5.0 * np.exp(-((code_to_physics_length * (x - x_c)) / 3.16228) ** 2.0))


def j_emit_func(x: float | np.ndarray | torch.Tensor, frequency: float | np.ndarray | torch.Tensor = 1.0):
	frequency = frequency + 0.45
	if isinstance(x, torch.Tensor):
		frequency = torch.as_tensor(frequency, device=x.device, dtype=x.dtype)
		return code_to_physics_length * 0.5 * (1 + torch.cos(2 * torch.pi * frequency * code_to_physics_length * x))
	return code_to_physics_length * 0.5 * (1 + np.cos(2 * np.pi * frequency * code_to_physics_length * x))


def _to_scalar(value: float | np.ndarray | torch.Tensor) -> float:
	if isinstance(value, torch.Tensor):
		return float(value.detach().cpu().reshape(-1)[0].item())
	return float(value)


def build_condition_inputs(x_values: torch.Tensor, I_o_value: torch.Tensor, x_c_value: torch.Tensor, freq_value: torch.Tensor) -> torch.Tensor:
	return torch.cat(
		[
			x_values,
			torch.full_like(x_values, _to_scalar(I_o_value)),
			torch.full_like(x_values, _to_scalar(x_c_value)),
			torch.full_like(x_values, _to_scalar(freq_value)),
		],
		dim=1,
	)


def reference_intensity_from_conditions(x_values: torch.Tensor, I_o_value: torch.Tensor, x_c_value: torch.Tensor, freq_value: torch.Tensor) -> torch.Tensor:
	I_o_scalar = _to_scalar(I_o_value)
	x_c_scalar = _to_scalar(x_c_value)
	freq_scalar = _to_scalar(freq_value)
	x_location = np.arange(N_cells_data) * d_x + 0.5 * d_x
	kappa_array = kappa_func(x_location, x_c_scalar)
	j_emit_array = j_emit_func(x_location, freq_scalar)
	spline = numerical_intensity_cubic_spline(I_o_scalar, kappa_array, j_emit_array, d_x)
	x_np = x_values.detach().cpu().numpy().reshape(-1)
	y_np = spline(x_np)
	return torch.from_numpy(np.asarray(y_np)).to(device=x_values.device, dtype=torch.float32).reshape_as(x_values)


def build_model_prediction(x_values: torch.Tensor, I_o_value: torch.Tensor, x_c_value: torch.Tensor, freq_value: torch.Tensor, model, device: torch.device) -> torch.Tensor:
	inputs = build_condition_inputs(x_values, I_o_value, x_c_value, freq_value)
	with torch.inference_mode():
		return model(inputs.to(device)).cpu()


def load_model(checkpoint_path: Path, stage: int, device: torch.device):
	state = torch.load(checkpoint_path, map_location="cpu")
	if not isinstance(state, dict):
		raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")
	state_dict = state.get("model_state_dict", state.get("state_dict", state))
	if not any(key.startswith("network.") for key in state_dict):
		raise ValueError(f"Checkpoint does not contain a model state dict: {checkpoint_path}")

	if stage == 1:
		model = FCN_hard_bc_four(N_hidden=64, dropout_rate=0.0).to(torch.device("cpu"))
	elif stage == 2:
		model = FCN_tiny_sine_hard_constraint(
			N_hidden=64,
			omega_first=state.get("omega_first", omega_first),
			omega_hidden=state.get("omega_hidden", omega_hidden),
		).to(torch.device("cpu"))
	else:
		raise ValueError(f"Unsupported model stage: {stage}")

	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()
	return model, state.get("residual_scaling_factor", 500.0) if stage == 2 else None


def save_curve_plot(output_dir: Path, x_values: torch.Tensor, reference: torch.Tensor, prediction: torch.Tensor, title: str, filename: str) -> None:
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(x_values.cpu(), reference.cpu(), "-b", label="Numerical Solution")
	ax.plot(x_values.cpu(), prediction.cpu(), "--r", label="Model Prediction")
	ax.axvspan(0.0, Boxsize_training, color="tab:green", alpha=0.08, label="Training (value & PDE) x-range")
	# ax.axvspan(0.0, 2.0 * Boxsize_training, color="tab:blue", alpha=0.08, label="Training (PDE) x-range")
	ax.set_xlabel("Path")
	ax.set_ylabel("Intensity")
	ax.set_title(title)
	ax.legend(loc="best")
	ax.grid(True, alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_dir / filename, dpi=200)
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Calculate the corrected residual from both PINN stages.")
	parser.add_argument("-I", "--checkpoint_a", type=Path, default=Path("./pinn_v7_best_model_epoch_960.pth"), 
					    help="Path to a .pth checkpoint.")
	parser.add_argument("-i", "--checkpoint_b", type=Path, default=Path("./multi_stage_v4_best_model_epoch_220.pth"), 
					    help="Path to a .pth checkpoint.")
	# parser.add_argument("--checkpoint-dir", type=Path, default=Path("."), help="Directory to search for pinn_v3_best_model_epoch_*.pth when --checkpoint is not provided.")
	parser.add_argument("-O", "--output-dir", type=Path, default=Path("./trained_result_plots_v4_d"), help="Directory where plots are saved.")
	# parser.add_argument("--show", action="store_true", help="Display the figures interactively after saving them.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	# Read the test parameters from a text file
	param_file = Path(output_dir / "parameter_random_samples.txt")

	checkpoint_path_a = args.checkpoint_a
	checkpoint_path_b = args.checkpoint_b
	print(f"Using checkpoint: {checkpoint_path_a}")
	print(f"Using stage-two checkpoint: {checkpoint_path_b}")
	if not checkpoint_path_a.is_file():
		raise FileNotFoundError(f"Stage-one checkpoint not found: {checkpoint_path_a}")
	if not checkpoint_path_b.is_file():
		raise FileNotFoundError(f"Stage-two checkpoint not found: {checkpoint_path_b}")
	checkpoint_label = checkpoint_path_a.stem

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(random_seed)
		device = torch.device("cuda")
		print("Using GPU")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
		print("Using MPS")
	else:
		device = torch.device("cpu")
		print("Using CPU")

	stage_a_model, _ = load_model(checkpoint_path_a, stage=1, device=device)
	stage_b_model, residual_scaling_factor = load_model(checkpoint_path_b, stage=2, device=device)


	x_values = torch.linspace(0, Boxsize_training, n_points_value_loss).view(-1, 1)
	x_location_truth = np.linspace(0, Boxsize_data, N_cells_data) # Position points for numerical solution

	'''
	I_o = torch.linspace(0.1, 1.0, n_points_I_o).view(-1, 1)
	x_c = torch.linspace(0.0, 0.9, n_points_x_c).view(-1, 1)
	freq = torch.linspace(0.05, 1.05, n_points_frequency).view(-1, 1)
	'''
	# switch to random sampling for I_o, x_c, freq
	I_o = torch.rand(n_points_I_o, 1) * 0.9 + 0.1
	x_c = torch.rand(n_points_x_c, 1) * 0.9
	freq = torch.rand(n_points_frequency, 1) * 1.0 + 0.05

	# For Saving the interpolated data
	I_save_pred = np.zeros((n_points_I_o, n_points_x_c, n_points_frequency, n_points_value_loss))
	I_save_num = np.zeros((n_points_I_o, n_points_x_c, n_points_frequency, n_points_value_loss))
	'''
	# Read the test parameters from a text file
	test_params = np.loadtxt(param_file, delimiter=",", skiprows=1)
	I_o_rand = test_params[:, 0]
	x_c_rand = test_params[:, 1]
	freq_rand = test_params[:, 2] - 0.45  # Subtract 0.45 for normalization
	
	# For Saving the interpolated data
	I_save_pred = np.zeros((len(I_o_rand), len(x_c_rand), len(freq_rand), len(x_location_truth)))
	I_save_num = np.zeros((len(I_o_rand), len(x_c_rand), len(freq_rand), len(x_location_truth)))

	I_o = torch.tensor(I_o_rand, dtype=torch.float32).view(-1, 1)
	x_c = torch.tensor(x_c_rand, dtype=torch.float32).view(-1, 1)
	freq = torch.tensor(freq_rand, dtype=torch.float32).view(-1, 1)
	'''
	sweep_output_dir = output_dir / "condition_sweeps"
	sweep_output_dir.mkdir(parents=True, exist_ok=True)
	for i, I_o_value in enumerate(I_o):
		for j, x_c_value in enumerate(x_c):
			for k, freq_value in enumerate(freq):
				reference = reference_intensity_from_conditions(x_values, I_o_value, x_c_value, freq_value).view(-1, 1)
				stage_a_prediction = build_model_prediction(x_values, I_o_value, x_c_value, freq_value, stage_a_model, device)
				stage_b_prediction = build_model_prediction(x_values, I_o_value, x_c_value, freq_value, stage_b_model, device)
				prediction = stage_a_prediction + stage_b_prediction / residual_scaling_factor
				# Save the numerical and predicted intensity data to .npy files
				I_save_pred[i, j, k] = prediction.detach().cpu().numpy().reshape(-1)
				I_save_num[i, j, k] = reference.detach().cpu().numpy().reshape(-1)
				'''
				title = ("I_o = {:.2f}, x_c = {:.2f}, frequency = {:.2f}".format(_to_scalar(I_o_value), _to_scalar(x_c_value), _to_scalar(freq_value + 0.45)))
				filename = f"pinn_v3_Io_{_to_scalar(I_o_value):.2f}_xc_{_to_scalar(x_c_value):.2f}_freq_{_to_scalar(freq_value + 0.45):.2f}.png"
				save_curve_plot(sweep_output_dir, x_values, reference, prediction, title, filename)
				'''
	print(f"Loaded stage-one checkpoint: {checkpoint_path_a}")
	print(f"Loaded stage-two checkpoint: {checkpoint_path_b}")
	print(f"Stage-two residual scaling factor: {residual_scaling_factor}")

	# calculate the residuals 
	residuals = I_save_pred - I_save_num
	# calculate the mean and standard deviation of the residuals, max and min of the residuals
	mean_residuals = np.mean(residuals)
	std_residuals = np.std(residuals)
	max_residuals = np.max(residuals)
	min_residuals = np.min(residuals)

	# print model name:
	print(f"Model name: {checkpoint_label}")
	print(f"Mean of residuals: {mean_residuals}")
	print(f"Standard deviation of residuals: {std_residuals}")
	print(f"Max of residuals: {max_residuals}")
	print(f"Min of residuals: {min_residuals}")

	# absolute residuals
	abs_residuals = np.abs(residuals)
	# calculate the mean and standard deviation of the absolute residuals, max and min of the absolute residuals
	mean_abs_residuals = np.mean(abs_residuals)
	std_abs_residuals = np.std(abs_residuals)
	max_abs_residuals = np.max(abs_residuals)
	min_abs_residuals = np.min(abs_residuals)

	# print model name:
	print(f"Model name: {checkpoint_label}")
	print(f"Mean of absolute residuals: {mean_abs_residuals}")
	print(f"Standard deviation of absolute residuals: {std_abs_residuals}")
	print(f"Max of absolute residuals: {max_abs_residuals}")
	print(f"Min of absolute residuals: {min_abs_residuals}")

	# plot histogram of the absolute residuals
	# hist_range = (0.0, max_abs_residuals)
	hist_range = (0.0, 0.008)
	plt.figure(figsize=(10, 6))
	plt.hist(abs_residuals.flatten(), bins=50, range=hist_range, density=True, color="blue", alpha=0.7)
	plt.xlabel("Absolute Residuals")
	plt.ylabel("Probability Density")
	plt.title("Histogram of Absolute Residuals")
	plt.savefig(output_dir / "histogram_absolute_residuals_epoch_960_220.png")

	# plot one residual curve and the corresponding numerical solution and model prediction
	# subplot the residuals, numerical solution and model prediction for I_o=0.1, x_c=0.0, freq=0.05
	fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
	case_label = "I_o={}, x_c={}, freq={}".format(_to_scalar(I_o[0]), _to_scalar(x_c[0]), _to_scalar(freq[0] + 0.45))

	axes[0].plot(x_values, residuals[0, 0, 0], "-r")
	axes[0].set_ylabel("Residual")
	axes[0].set_title(f"Residual ({case_label})")
	axes[0].grid(True, alpha=0.25)

	axes[1].plot(x_values, I_save_num[0, 0, 0], "-b", label="Numerical Solution")
	axes[1].plot(x_values, I_save_pred[0, 0, 0], "-g", label="Model Prediction")
	axes[1].set_ylabel("Intensity")
	axes[1].set_title("Numerical vs model")
	axes[1].legend(loc="best")
	axes[1].grid(True, alpha=0.25)

	fig.savefig(output_dir / "residuals_one_case_epoch_960_220.png", dpi=200)
	plt.close(fig)

if __name__ == "__main__":
	main()
