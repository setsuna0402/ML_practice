'''
Make plots comparing NN method and interpolation methods for random samples of I_o, x_c, and frequency.
'''

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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


def save_curve_plot(output_dir: Path, x_values: np.ndarray,
					linear_interp: np.ndarray, cubic_spline: np.ndarray,
					nn_pred: np.ndarray, numerical: np.ndarray, title: str, filename: str) -> None:
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(x_values, numerical, "-b", label="Numerical")
	ax.plot(x_values, nn_pred, "--r", label="Nerual Network")
	ax.plot(x_values, linear_interp, "--g", label="Linear Interpolation")
	ax.plot(x_values, cubic_spline, "--k", label="Cubic Spline")
	ax.axvspan(0.0, Boxsize_training, color="tab:green", alpha=0.08, label="Training (value & PDE) x-range")
	# ax.axvspan(0.0, 2.0 * Boxsize_training, color="tab:blue", alpha=0.08, label="Training (PDE) x-range")
	ax.set_xlabel("Path")
	ax.set_ylabel("Intensity")
	ax.set_ylim(0.0, 1.1)
	ax.set_title(title)
	ax.legend(loc="best")
	ax.grid(True, alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_dir / filename, dpi=200)
	plt.close(fig)

para_file = Path("linear_interpolation_random_plots/parameter_random_samples.txt")
I_o_rand, x_c_rand, freq_rand = np.loadtxt(para_file, delimiter=",", unpack=True)
freq_rand = freq_rand - 0.45  # Adjust frequency to match the original definition
I_linear_interp = np.load("linear_interpolation_random_plots/I_random_samples_pred.npy")
I_cubic_spline = np.load("cubic_interpolation_random_plots/I_random_samples_pred.npy")
I_nn_pred = np.load("trained_result_plots_v6_b/I_random_samples_pred.npy")
I_ref = np.load("trained_result_plots_v6_b/I_random_samples_num.npy")
output_dir = Path("compare_ML_and_interpolation_random_plots")
output_dir.mkdir(parents=True, exist_ok=True)

x_location_truth = np.linspace(0, Boxsize_data, N_cells_data)# Position points for numerical solution
# Create I_o, x_c, and frequency samples for the training data.
I_o_training = np.linspace(0.1, 1.0, n_points_I_o)
x_c_training = np.linspace(0.0, 0.9, n_points_x_c)
frequency_training = np.linspace(0.05, 1.05, n_points_frequency)
x_location_training = np.linspace(0, Boxsize_training, n_points_value_loss)  # Position points for training





# Loop over the random samples and compute the interpolated intensity
for i, I_o in enumerate(I_o_rand):
	for j, x_c in enumerate(x_c_rand):
		for k, freq in enumerate(freq_rand):
			I_linear_1D = I_linear_interp[i, j, k]
			I_cubic_1D = I_cubic_spline[i, j, k]
			I_nn_1D = I_nn_pred[i, j, k]
			I_ref_1D = I_ref[i, j, k]
			# Save the plots comparing the neural network prediction, linear interpolation, cubic spline interpolation, and numerical solution
			title = ("I_o = {:.3f}, x_c = {:.3f}, frequency = {:.2f}".format(I_o, x_c, freq + 0.45))
			filename = f"three_models_Io_{I_o:.2f}_xc_{x_c:.2f}_freq_{freq + 0.45:.2f}.png"
			save_curve_plot(output_dir, x_location_truth, I_linear_1D, I_cubic_1D, I_nn_1D, I_ref_1D, title, filename)

