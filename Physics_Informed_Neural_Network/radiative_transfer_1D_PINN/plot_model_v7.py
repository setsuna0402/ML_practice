#!/usr/bin/env python3
"""Load a trained v3 PINN checkpoint and visualize conditioned predictions."""

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
from NN_classes import FCN_four, FCN_hard_bc_four

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


def get_checkpoint_path(checkpoint_dir: Path) -> Path:
	candidates = list(checkpoint_dir.glob("pinn_v3_best_model_epoch_*.pth"))
	if not candidates:
		raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

	def epoch_number(path: Path) -> int:
		match = re.search(r"epoch_(\d+)\.pth$", path.name)
		return int(match.group(1)) if match else -1

	return max(candidates, key=epoch_number)


def load_model(checkpoint_path: Path, device: torch.device):
	model = FCN_hard_bc_four(N_hidden=64).to(device)
	# model = FCN_four_short(N_hidden=64, dropout_rate=0.0).to(device)
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
	parser = argparse.ArgumentParser(description="Load a trained PINN v3 checkpoint and create visualization plots.")
	parser.add_argument("-I", "--checkpoint", type=Path, default=None, help="Path to a .pth checkpoint. If omitted, the latest pinn_v3_best_model_epoch_*.pth in --output-dir is used.")
	# parser.add_argument("--checkpoint-dir", type=Path, default=Path("."), help="Directory to search for pinn_v3_best_model_epoch_*.pth when --checkpoint is not provided.")
	parser.add_argument("-O", "--output-dir", type=Path, default=Path("trained_result_plots_v7_epoch_5000"), help="Directory where plots are saved.")
	# parser.add_argument("--show", action="store_true", help="Display the figures interactively after saving them.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	checkpoint_path = args.checkpoint
	if checkpoint_path is None:
		print("You need to provide a checkpoint path using the --checkpoint argument.")
		print("Exiting.")
		return
	checkpoint_label = checkpoint_path.stem

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

	model = load_model(checkpoint_path, device)

	x_values = torch.linspace(0, Boxsize_data, N_cells_data).view(-1, 1)

	I_o = torch.rand(n_sample, 1) * 0.85 + 0.15
	x_c = torch.rand(n_sample, 1) * 0.85 + 0.05
	freq = torch.rand(n_sample, 1) * 0.96 + 0.09
	sweep_output_dir = output_dir / "condition_sweeps"
	sweep_output_dir.mkdir(parents=True, exist_ok=True)
	for I_o_value in I_o:
		for x_c_value in x_c:
			for freq_value in freq:
				reference = reference_intensity_from_conditions(x_values, I_o_value, x_c_value, freq_value).view(-1, 1)
				prediction = build_model_prediction(x_values, I_o_value, x_c_value, freq_value, model, device)
				title = ("I_o = {:.2f}, x_c = {:.2f}, frequency = {:.2f}".format(_to_scalar(I_o_value), _to_scalar(x_c_value), _to_scalar(freq_value + 0.45)))
				filename = f"pinn_v7_Io_{_to_scalar(I_o_value):.2f}_xc_{_to_scalar(x_c_value):.2f}_freq_{_to_scalar(freq_value + 0.45):.2f}.png"
				save_curve_plot(sweep_output_dir, x_values, reference, prediction, title, filename)
	print(f"Loaded checkpoint: {checkpoint_path}")
	print(f"Saved plots to: {sweep_output_dir}")



if __name__ == "__main__":
	main()
