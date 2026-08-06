"""Test linear interpolation"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
# import torch

from analytic_solve import RT_1D_solver

random_seed = 42
np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# torch.set_default_dtype(torch.float32)
n_sample = 5

code_to_physics_length = 10.0
Boxsize_data = 1.0
N_cells_data = 4096
d_x = Boxsize_data / N_cells_data
Boxsize_training = 0.5
n_points_value_loss = 50  # Number of data points for the value loss term
n_points_I_o = 20  # Number of data points for the initial intensity I_o
n_points_x_c = 5  # Number of data points for the absorption center x_c
n_points_frequency = 10  # Number of data points for the emission frequency

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Linear interpolation of numerical intensity data.")
	parser.add_argument("-O", "--output-dir", type=Path, default=Path("cubic_interpolation_test_plots"), help="Directory where plots are saved.")
	return parser.parse_args()

def numerical_intensity_cubic_spline(I_o: float, kappa: np.ndarray, j_emit: np.ndarray, delta_x: float):
	discrete_I = RT_1D_solver(I_o, kappa, j_emit, delta_x, return_last_I=False)
	x_boundaries = np.arange(len(discrete_I)) * delta_x
	from scipy.interpolate import CubicSpline

	return CubicSpline(x_boundaries, discrete_I)


def kappa_func(x: float | np.ndarray, x_c: float | np.ndarray = 1.0):
	return code_to_physics_length * 0.8 * (1.0 + 5.0 * np.exp(-((code_to_physics_length * (x - x_c)) / 3.16228) ** 2.0))


def j_emit_func(x: float | np.ndarray, frequency: float | np.ndarray = 1.0):
	frequency = frequency + 0.45
	return code_to_physics_length * 0.5 * (1 + np.cos(2 * np.pi * frequency * code_to_physics_length * x))



def save_curve_plot(output_dir: Path, x_values: np.ndarray, reference: np.ndarray, prediction: np.ndarray, title: str, filename: str) -> None:
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(x_values, reference, "-b", label="Numerical Solution")
	ax.plot(x_values, prediction, "--r", label="Model Prediction")
	ax.axvspan(0.0, Boxsize_training, color="tab:green", alpha=0.08, label="Training (value & PDE) x-range")
	# ax.axvspan(0.0, 2.0 * Boxsize_training, color="tab:blue", alpha=0.08, label="Training (PDE) x-range")
	ax.set_xlabel("Path")
	ax.set_ylabel("Intensity")
	ax.set_ylim(0.0, 1.5)
	ax.set_title(title)
	ax.legend(loc="best")
	ax.grid(True, alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_dir / filename, dpi=200)
	plt.close(fig)


args = parse_args()
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

x_location_truth = np.linspace(0, Boxsize_data, N_cells_data)# Position points for numerical solution
# Create I_o, x_c, and frequency samples for the training data.
I_o_training = np.linspace(0.1, 1.0, n_points_I_o)
x_c_training = np.linspace(0.0, 0.9, n_points_x_c)
frequency_training = np.linspace(0.05, 1.05, n_points_frequency)
x_location_training = np.linspace(0, Boxsize_training, n_points_value_loss)  # Position points for training


I_num = np.zeros((len(I_o_training), len(x_c_training), len(frequency_training), len(x_location_training)))


# Loop over all combinations of I_o, x_c, and frequency to compute the numerical intensity using cubic spline interpolation
for i, I_o in enumerate(I_o_training):
	for j, x_c in enumerate(x_c_training):
		for k, frequency in enumerate(frequency_training):
			kappa = kappa_func(x_location_truth, x_c)
			j_emit = j_emit_func(x_location_truth, frequency)
			cubic_spline = numerical_intensity_cubic_spline(I_o, kappa, j_emit, d_x)
			I_num[i, j, k] = cubic_spline(x_location_training)
'''
I_o_rand = np.random.rand(n_sample) * 0.9 + 0.1
x_c_rand = np.random.rand(n_sample) * 0.9
freq_rand = np.random.rand(n_sample) * 1.0 + 0.05
'''
I_o_rand = I_o_training[::4]
x_c_rand = x_c_training[::1]
freq_rand = frequency_training[::2]
# Linear interpolation of the numerical intensity for the random samples
interpolator = RegularGridInterpolator(
	(I_o_training, x_c_training, frequency_training, x_location_training),
	I_num,
	method="cubic",
	bounds_error=False,
	fill_value=None,
)

output_dir = output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Saving parameters
temp_save = np.vstack((I_o_rand, x_c_rand))
temp_save = np.vstack((temp_save, freq_rand + 0.45))  # Add 0.45 to the frequency for saving
temp_save = np.transpose(temp_save)
temp_save = np.sort(temp_save, axis=0)  # Sort the samples for better visualization
# Header for the saved file
header = "I_o, x_c, frequency"
np.savetxt(output_dir / "parameter_random_samples.txt", temp_save, header=header, fmt="%.6f", delimiter=",")
# Load the parameters back for garnteeing the order is identical in different runs
temp_load = np.loadtxt(output_dir / "parameter_random_samples.txt", delimiter=",")
I_o_rand = temp_load[:, 0]
x_c_rand = temp_load[:, 1]
freq_rand = temp_load[:, 2] - 0.45  # Subtract 0.45, following the original definition of frequency in the code

# For Saving the interpolated data
I_save_pred = np.zeros((len(I_o_rand), len(x_c_rand), len(freq_rand), len(x_location_truth)))
I_save_num = np.zeros((len(I_o_rand), len(x_c_rand), len(freq_rand), len(x_location_truth)))

# Loop over the random samples and compute the interpolated intensity
for i, I_o in enumerate(I_o_rand):
	for j, x_c in enumerate(x_c_rand):
		for k, freq in enumerate(freq_rand):
			# Interpolate the intensity at the random sample points
			I_interp = interpolator((I_o, x_c, freq, x_location_truth))
			# calculate the truth intensity using the cubic spline for comparison
			kappa = kappa_func(x_location_truth, x_c)
			j_emit = j_emit_func(x_location_truth, freq)
			cubic_spline = numerical_intensity_cubic_spline(I_o, kappa, j_emit, d_x)
			I_ref = cubic_spline(x_location_truth)
			# print(I_interp.shape, I_ref.shape)
			I_save_pred[i, j, k] = I_interp
			I_save_num[i, j, k] = I_ref
			
			# You can now use I_interp for further analysis or plotting
			title = ("I_o = {:.3f}, x_c = {:.3f}, frequency = {:.2f}".format(I_o, x_c, freq + 0.45))
			filename = f"cubic_interp_Io_{I_o:.2f}_xc_{x_c:.2f}_freq_{freq + 0.45:.2f}.png"
			save_curve_plot(output_dir, x_location_truth, I_ref, I_interp, title, filename)
			
print("Output saves in ", output_dir)
np.save(output_dir / "I_random_samples_pred.npy", I_save_pred)
np.save(output_dir / "I_random_samples_num.npy", I_save_num)