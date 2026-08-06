#!/usr/bin/env python3
"""Load a trained v2 PINN checkpoint and visualize its predictions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline

from analytic_solve import RT_1D_solver
from NN_classes import FCN_two

allow_device = True
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.set_default_dtype(torch.float32)

code_to_physics_length = 10.0
Boxsize_data = 1.0
N_cells_data = 4096
d_x = Boxsize_data / N_cells_data
Boxsize_training = 0.25


def numerical_intensity_cubic_spline(
	I_o: float | np.float64,
	kappa: np.ndarray,
	j_emit: np.ndarray,
	delta_x: float | np.float64 | np.ndarray,
) -> CubicSpline:
	"""Build a spline for the discrete radiative transfer solution."""
	discrete_I = RT_1D_solver(I_o, kappa, j_emit, delta_x, return_last_I=False)
	if isinstance(delta_x, float) or isinstance(delta_x, np.float64):
		x_boundaries = np.arange(len(discrete_I)) * delta_x
	else:
		x_boundaries = np.cumsum(delta_x) - delta_x
	return CubicSpline(x_boundaries, discrete_I)


def kappa_func(x: float | np.ndarray | torch.Tensor):
	if isinstance(x, torch.Tensor):
		return code_to_physics_length * 0.8 * (1 + torch.exp(-(code_to_physics_length * x) ** 2.0 / 10.0))
	return code_to_physics_length * 0.8 * (1 + np.exp(-(code_to_physics_length * x) ** 2.0 / 10.0))


def j_emit_func(x: float | np.ndarray | torch.Tensor):
	if isinstance(x, torch.Tensor):
		return code_to_physics_length * 0.5 * (1 + torch.cos(2 * torch.pi * code_to_physics_length * x))
	return code_to_physics_length * 0.5 * (1 + np.cos(2 * np.pi * code_to_physics_length * x))


def get_checkpoint_path(checkpoint_dir: Path) -> Path:
	candidates = list(checkpoint_dir.glob("pinn_v2_best_model_epoch_*.pth"))
	if not candidates:
		raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

	def epoch_number(path: Path) -> int:
		match = re.search(r"epoch_(\d+)\.pth$", path.name)
		return int(match.group(1)) if match else -1

	return max(candidates, key=epoch_number)


def load_model(checkpoint_path: Path, device: torch.device) -> FCN_two:
	model = FCN_two(N_hidden=64, dropout_rate=0.0).to(device)
	state = torch.load(checkpoint_path, map_location=device)
	if isinstance(state, dict) and any(key.startswith("network.") for key in state):
		model.load_state_dict(state)
	elif isinstance(state, dict) and "state_dict" in state:
		model.load_state_dict(state["state_dict"])
	elif isinstance(state, dict) and "model_state_dict" in state:
		model.load_state_dict(state["model_state_dict"])
	else:
		raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")
	model.eval()
	return model


def build_reference_curve(I_o: float, x_values: np.ndarray) -> np.ndarray:
	x_location = np.arange(N_cells_data) * d_x + 0.5 * d_x
	kappa_array = kappa_func(x_location)
	j_emit_array = j_emit_func(x_location)
	spline = numerical_intensity_cubic_spline(I_o, kappa_array, j_emit_array, d_x)
	return np.asarray(spline(x_values), dtype=np.float64)


def build_model_curve(model: FCN_two, x_values: np.ndarray, I_o: float, device: torch.device) -> np.ndarray:
	x_tensor = torch.from_numpy(x_values.astype(np.float32)).view(-1, 1)
	i_tensor = torch.full_like(x_tensor, float(I_o))
	model_input = torch.cat([x_tensor, i_tensor], dim=1).to(device)
	with torch.inference_mode():
		y_pred = model(model_input).detach().cpu().numpy().reshape(-1)
	return y_pred


def save_line_plot(
	output_dir: Path,
	x_values: np.ndarray,
	reference: np.ndarray,
	prediction: np.ndarray,
	I_o: float,
	checkpoint_label: str,
) -> None:
	fig, (ax_curve, ax_error) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)

	ax_curve.plot(x_values, reference, label="Reference", color="tab:blue", linewidth=2)
	ax_curve.plot(x_values, prediction, label="Model", color="tab:red", linewidth=2, linestyle="--")
	ax_curve.axvspan(0.0, Boxsize_training, color="tab:green", alpha=0.08, label="Training (Value) x-range")
	ax_curve.axvspan(0.0, 2.0 * Boxsize_training, color="tab:blue", alpha=0.08, label="Training (ODE) x-range")
	ax_curve.set_ylabel("Intensity")
	ax_curve.set_title(f"PINN v2 prediction for I_o = {I_o:.2f} ({checkpoint_label})")
	ax_curve.legend(loc="best")
	ax_curve.grid(True, alpha=0.25)

	abs_error = np.abs(prediction - reference) / reference
	ax_error.semilogy(x_values, abs_error, color="tab:purple", linewidth=1.5)
	ax_error.set_xlabel("x")
	ax_error.set_ylabel("|pred - numerical| / numerical")
	ax_error.grid(True, alpha=0.25)

	fig.savefig(output_dir / f"pinn_v2_lineplot_Io_{I_o:.2f}.png", dpi=200)
	plt.close(fig)


def save_surface_plots(
	output_dir: Path,
	x_values: np.ndarray,
	I_o_values: np.ndarray,
	prediction_surface: np.ndarray,
	reference_surface: np.ndarray,
	checkpoint_label: str,
) -> None:
	abs_error = np.abs(prediction_surface - reference_surface)

	fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
	plots = [
		(prediction_surface, "Model prediction", "viridis"),
		(reference_surface, "Reference solution", "viridis"),
		(abs_error, "Absolute error", "magma"),
	]

	mesh_x, mesh_Io = np.meshgrid(x_values, I_o_values)
	for ax, (data, title, cmap) in zip(axes, plots, strict=True):
		pcm = ax.pcolormesh(mesh_x, mesh_Io, data, shading="auto", cmap=cmap)
		ax.set_title(title)
		ax.set_xlabel("x")
		ax.set_ylabel("I_o")
		fig.colorbar(pcm, ax=ax)

	fig.suptitle(f"PINN v2 field visualization ({checkpoint_label})")
	fig.savefig(output_dir / "pinn_v2_surface_summary.png", dpi=200)
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Load a trained PINN v2 checkpoint and create plots.")
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=None,
		help="Path to a .pth checkpoint. If None, will try to find the latest *pth in the output dirctory.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("./"),
		help="Directory where plots will be saved.",
	)
	parser.add_argument(
		"--initial-conditions",
		type=float,
		nargs="+",
		default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
		help="Initial intensity values to visualize.",
	)
	parser.add_argument(
		"--surface-samples",
		type=int,
		default=48,
		help="Number of I_o samples used for the 2D surface summary.",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display the figures interactively after saving them.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	checkpoint_path = args.checkpoint or get_checkpoint_path(output_dir)
	checkpoint_label = checkpoint_path.stem
	
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
	model = load_model(checkpoint_path, device)

	x_values = np.arange(N_cells_data) * d_x
	
    #I_o_list = args.initial_conditions
	I_o_list = np.arange(1, 21) * 0.05

	for I_o in I_o_list:
		reference = build_reference_curve(I_o, x_values)
		prediction = build_model_curve(model, x_values, I_o, device)
		save_line_plot(output_dir, x_values, reference, prediction, I_o, checkpoint_label)

	surface_Io_values = np.linspace(min(I_o_list), max(I_o_list), args.surface_samples)
	reference_surface = np.stack([build_reference_curve(I_o, x_values) for I_o in surface_Io_values], axis=0)
	prediction_surface = np.stack([build_model_curve(model, x_values, I_o, device) for I_o in surface_Io_values], axis=0)
	save_surface_plots(output_dir, x_values, surface_Io_values, prediction_surface, reference_surface, checkpoint_label)

	print(f"Loaded checkpoint: {checkpoint_path}")
	print(f"Saved plots to: {output_dir}")

	if args.show:
		plt.show()


if __name__ == "__main__":
	main()
