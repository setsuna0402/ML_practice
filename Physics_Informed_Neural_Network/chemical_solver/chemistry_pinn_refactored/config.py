"""Single source of truth for the chemistry-PINN proof-of-concept."""
from pathlib import Path
from gracklepy.utilities.physical_constants import mass_hydrogen_cgs

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path("./")
GRACKLE_DATA_FILE = PROJECT_DIR / "CloudyData_UVB=HM2012.h5"
PRECOMPUTED_DATA_FILE = PROJECT_DIR / "grackle_relaxed_training_data_10000.npz"
VALIDATION_DATA_FILE = PROJECT_DIR / "grackle_relaxed_validate_data.npz"  # Set to None to generate validation trajectories on the fly.
OUTPUT_DIR = PROJECT_DIR / "trained_results_v1_relaxed_n_10000"

# -----------------------------------------------------------------------------
# Grackle code units / fixed physics
# -----------------------------------------------------------------------------
GAMMA = 5.0 / 3.0
N_BARYON_Z0 = 2.5e-7  # cm^-3
DENSITY_UNIT = N_BARYON_Z0 * mass_hydrogen_cgs  # g cm^-3 represented by code density=1

# -----------------------------------------------------------------------------
# Sampling ranges (latest POC ranges)
# gamma_* are physical photoionisation rates [s^-1 / absorbing atom].
# pi_* are species-specific photoheating rates [eV s^-1 / absorbing atom].
# density is Grackle code density.
# -----------------------------------------------------------------------------
SAMPLER_RANGES = {
    "density_range": (1.0e-2, 1.0e2),
    # "temperature_range": (1.0e2, 1.0e4),
    "temperature_range": (1.0e2, 5.0e3),  # For relaxation, gas is no longer neutral at high temperatures, so we restrict to T < 5e3 K.
    "Gamma_HI_range": (1.0e-15, 1.0e-11),
    "Gamma_HeI_range": (1.0e-15, 1.0e-11),
    "Gamma_HeII_range": (1.0e-17, 1.0e-12),
    "pi_HI_range": (1.0e-14, 1.0e-10),
    "pi_HeI_range": (1.0e-14, 1.0e-10),
    "pi_HeII_range": (1.0e-14, 1.0e-12),
}

TRAJECTORY_KWARGS = {
    "final_time": 1.0,
    "max_change_HII": 1.0e-3,
    "cooling_fraction": 1.0e-3,
    "max_timestep": 1.0e-2,
    "initial_max_change_HII": 1.0e-3,
    "relaxation": True,
    "relax_fraction": 0.01,
    "tolerance": 0.01,
    "max_iterations": 10000,
    "flag_debug": False,
}

INTERPOLATION_KWARGS = {
    "n_times": 128,
    "spacing": "uniform",
    "method": "pchip",
}

# -----------------------------------------------------------------------------
# Normalisation reference values
# -----------------------------------------------------------------------------
DENSITY_REF = 1.0
U_REF = 3.10289869e12  # erg/g-equivalent specific energy after velocity_unit^2 conversion
GAMMA_HI_REF = 25.8376836       # 1/Myr (dataset stores Grackle code-time rate)
GAMMA_HEI_REF = 14.9876104
GAMMA_HEII_REF = 0.11397924
PI_HI_REF = 3.2556039206009867e-12   # eV/s/absorbing atom
PI_HEI_REF = 3.387724795493442e-12
PI_HEII_REF = 7.789881261416638e-14

# Fractions are first floored, then log10-transformed and affinely mapped.
# FRACTION_FLOOR maps to FRACTION_NORM_MIN; fraction=1 maps to FRACTION_NORM_MAX.
FRACTION_FLOOR = 1.0e-10
FRACTION_NORM_MIN = -2.0
FRACTION_NORM_MAX = 2.0

# -----------------------------------------------------------------------------
# Dataset / training settings
# -----------------------------------------------------------------------------
PRECOMPUTE_N_TRAJECTORIES = 5000
# PRECOMPUTE_SEED = 75369
# PRECOMPUTE_SEED = 5642
PRECOMPUTE_SEED = 1234
# PRECOMPUTE_SEED = 8925

DATA_MODE = "precomputed"  # "precomputed" or "random_epoch"
N_RANDOM_TRAJECTORIES_PER_EPOCH = 8
RANDOM_SEED = 42
N_EPOCHS = 10000
LEARNING_RATE = 1.0e-3
COSINE_ETA_MIN = 1.0e-6
BATCH_SIZE = 2048
VALIDATION_INTERVAL = 20
NUM_WORKERS = 8
USE_PIN_MEMORY = True
ALLOW_DEVICE = True

# Validation plotting only; validation loss still uses the full validation set.
VALIDATION_PLOT_N_SAMPLES = 10
VALIDATION_PLOT_LOGY = False

# Reference validation radiation field (physical Gamma in s^-1; pi in eV/s).
VALIDATION_BASE = {
    "density": 1.0,
    "temperature": 1.0e3,
    "Gamma_HI": 8.187467866905527e-13,
    "Gamma_HeI": 4.749287143327514e-13,
    "Gamma_HeII": 3.611784136088691e-15,
    "pi_HI": PI_HI_REF,
    "pi_HeI": PI_HEI_REF,
    "pi_HeII": PI_HEII_REF,
}
