"""Generate the fixed precomputed Grackle HDF5 dataset."""
from config import (
    PRECOMPUTED_DATA_FILE,
    PRECOMPUTE_N_TRAJECTORIES,
    PRECOMPUTE_SEED,
    DENSITY_UNIT,
    VALIDATION_DATA_FILE,
    INTERPOLATION_KWARGS,
    PDE_ENABLED,
)
from grackle_data import generate_precomputed_dataset


def main():
    result = generate_precomputed_dataset(
        # PRECOMPUTED_DATA_FILE,
        VALIDATION_DATA_FILE,
        PRECOMPUTE_N_TRAJECTORIES,
        code_density=DENSITY_UNIT,
        seed=PRECOMPUTE_SEED,
        interpolation_kwargs=INTERPOLATION_KWARGS,
        include_pde=PDE_ENABLED,
    )

    print("Saved:", PRECOMPUTED_DATA_FILE)
    if PDE_ENABLED:
        X, Y, X_pde, PDE_aux = result
        print("X shape:", X.shape)
        print("Y shape:", Y.shape)
        print("X_pde shape:", X_pde.shape)
        print("PDE_aux shape:", PDE_aux.shape, "[k1..k6]")
        print("HDF5 groups: /initial_condition/*, /value/* and /pde/*")
        print("PDE sampling: value-grid midpoints; N_pde = N_value - 1 per trajectory")
    else:
        X, Y = result
        print("X shape:", X.shape)
        print("Y shape:", Y.shape)


if __name__ == "__main__":
    main()
