"""Precompute a fixed Grackle training dataset for train_pinn.py."""

from gracklepy.utilities.physical_constants import mass_hydrogen_cgs
from grackle_data import generate_precomputed_dataset

OUTPUT_FILE = "grackle_training_data.npz"
N_TRAJECTORIES = 1000
SEED = 1234

# Same density unit convention as train_pinn.py.
N_BARYON_Z0 = 2.5e-7       # cm^-3
DENSITY_Z0 = N_BARYON_Z0 * mass_hydrogen_cgs

# Easy-to-edit POC parameter ranges.
SAMPLER_KWARGS = {
    "density_range": (1.0, 1.0e4),
    "temperature_range": (1.0e2, 1.0e5),
    "Gamma_HI_range": (1.0e-15, 1.0e-11),
    "Gamma_HeI_range": (1.0e-16, 1.0e-11),
    "Gamma_HeII_range": (1.0e-18, 1.0e-12),
    "final_time": 1.0,
}


def main():
    X, Y = generate_precomputed_dataset(
        OUTPUT_FILE,
        N_TRAJECTORIES,
        code_density=DENSITY_Z0,
        seed=SEED,
        sampler_kwargs=SAMPLER_KWARGS,
    )

    print("Saved:", OUTPUT_FILE)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)


if __name__ == "__main__":
    main()
