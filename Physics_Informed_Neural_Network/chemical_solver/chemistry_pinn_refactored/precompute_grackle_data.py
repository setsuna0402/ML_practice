"""Generate the fixed precomputed Grackle dataset."""
from config import (
    PRECOMPUTED_DATA_FILE,
    PRECOMPUTE_N_TRAJECTORIES,
    PRECOMPUTE_SEED,
    DENSITY_UNIT,
    VALIDATION_DATA_FILE,
    INTERPOLATION_KWARGS,
)
from grackle_data import generate_precomputed_dataset


def main():
    X, Y = generate_precomputed_dataset(
        PRECOMPUTED_DATA_FILE,
        # VALIDATION_DATA_FILE,
        PRECOMPUTE_N_TRAJECTORIES,
        code_density=DENSITY_UNIT,
        seed=PRECOMPUTE_SEED,
        interpolation_kwargs=INTERPOLATION_KWARGS,
    )
    print("Saved:", PRECOMPUTED_DATA_FILE)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)


if __name__ == "__main__":
    main()
