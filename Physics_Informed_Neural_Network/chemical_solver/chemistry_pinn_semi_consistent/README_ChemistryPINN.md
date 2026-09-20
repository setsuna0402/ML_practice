# ChemistryPINN: Grackle-Based Primordial Chemistry Surrogate

This repository contains a proof-of-concept neural surrogate for the Grackle primordial six-species chemistry network. The current version combines supervised trajectory fitting with a staggered, frozen-coefficient physics loss.

The model predicts four independent quantities:

- `HI_fraction`
- `HeI_fraction`
- `HeII_fraction`
- `internal_energy`

The remaining ion fractions are reconstructed from conservation:

```text
HII_fraction   = 1 - HI_fraction
HeIII_fraction = 1 - HeI_fraction - HeII_fraction
```

The current PINN prototype uses Grackle reaction coefficients `k1`--`k6` evaluated on the reference Grackle trajectory, while the species state is supplied by the neural network. This is therefore a **frozen-coefficient / semi-consistent PINN** rather than a fully self-consistent PINN.

---

## 1. Main files

```text
config.py
    Central configuration: paths, sampling ranges, normalization constants,
    training hyperparameters, and PDE-loss settings.


grackle_data.py
    Main data/physics backend. It builds Grackle trajectories, interpolates the
    supervised value grid, constructs midpoint PDE collocation points, and
    reads/writes the HDF5 training dataset.

grackle_timestep.py
    Adaptive HII timestep calculation and interpolation of Grackle reaction
    rates k1--k6.

initialise_grackle_fluid.py
    Creates the initial FluidContainer and optionally relaxes the chemistry at
    fixed initial temperature before the radiation field is switched on.

normalisation.py
    Input/output normalization and inverse normalization.

pinn_model.py
    ChemistryPINN neural-network definition and hard initial-condition handling.

precompute_grackle_data.py
    Generates the fixed HDF5 training dataset.

train_pinn_v2.py
    Trains the neural network using supervised value loss plus the optional
    frozen-coefficient species PDE loss.
```

> Note: `initialise_grackle_fluid.py` is imported by `grackle_data.py`, so it must be present even though it is not part of the most recent generated archive.

---

## 2. Python requirements

The code requires a working Python environment with at least:

```text
numpy
scipy
h5py
torch
matplotlib
tqdm
gracklepy
```

A typical environment should therefore be able to run:

```bash
python -c "import numpy, scipy, h5py, torch, matplotlib, tqdm, gracklepy"
```

The project also expects the Grackle Cloudy data file configured in `config.py`, currently:

```python
GRACKLE_DATA_FILE = PROJECT_DIR / "CloudyData_UVB=HM2012.h5"
```

Make sure this file exists at the configured path before generating trajectories.

---

## 3. Recommended project layout

A simple layout is:

```text
ChemistryPINN/
├── config.py
├── grackle_data.py
├── grackle_timestep.py
├── initialise_grackle_fluid.py
├── normalisation.py
├── pinn_model.py
├── precompute_grackle_data.py
├── train_pinn_v2.py
└── CloudyData_UVB=HM2012.h5
```

Run all commands from this directory unless `PROJECT_DIR` is changed.

---

## 4. Configure the experiment

Most settings are controlled from `config.py`.

### Sampling ranges

The current proof-of-concept ranges include:

```python
SAMPLER_RANGES = {
    "density_range": (1.0e-2, 1.0e2),
    "temperature_range": (1.0e2, 5.0e3),
    "Gamma_HI_range": (...),
    "Gamma_HeI_range": (...),
    "Gamma_HeII_range": (...),
    "pi_HI_range": (...),
    "pi_HeI_range": (...),
    "pi_HeII_range": (...),
}
```

Positive dimensional parameters are sampled log-uniformly.

The upper initial-temperature limit is currently `5e3 K` so that the relaxed initial state remains mostly neutral and does not begin in the strongly collisionally ionized regime.

### Trajectory integration

Important settings are in:

```python
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
```

With `relaxation=True`, the initialization sequence is:

```text
near-neutral initial guess
        -> chemistry relaxation at fixed temperature
        -> relaxed t=0 species state
        -> set external Gamma/pi radiation fields
        -> time-dependent trajectory
```

Therefore the `HI_fraction`, `HeI_fraction`, `HeII_fraction`, and `internal_energy` stored under `/initial_condition` are the **post-relaxation t=0 values** used to condition the neural network.

### Value grid

The current supervised grid is:

```python
INTERPOLATION_KWARGS = {
    "n_times": 128,
    "spacing": "uniform",
    "method": "pchip",
}
```

Each adaptive Grackle trajectory is interpolated to 128 fixed value points.

### PDE settings

The current frozen-coefficient physics loss is controlled by:

```python
PDE_ENABLED = True
PDE_LOSS_WEIGHT = 1.0e-3
PDE_START_EPOCH = 0
```

When enabled, the PDE collocation grid contains one midpoint between every pair of value points:

```text
128 value points -> 127 PDE midpoint points
```

For value-grid times `t[i]`, the PDE time is

```text
t_pde[i] = 0.5 * (t[i] + t[i+1])
```

The reaction coefficients `k1`--`k6` are interpolated directly from the original adaptive Grackle trajectory to these midpoint times. No Grackle time derivative is stored or interpolated.

---

## 5. Generate the precomputed HDF5 dataset

For the current PINN mode, use a fixed precomputed dataset.

Check in `config.py`:

```python
PRECOMPUTED_DATA_FILE = PROJECT_DIR / "grackle_training_data_10000.h5"
PRECOMPUTE_N_TRAJECTORIES = 10000
PRECOMPUTE_SEED = 5642
PDE_ENABLED = True
```

Then run:

```bash
python precompute_grackle_data.py
```

For each sampled parameter set, the script will:

1. Build the Grackle chemistry object.
2. Create and optionally relax the initial fluid state.
3. Switch on the external photoionization/photoheating fields.
4. Integrate one adaptive Grackle trajectory.
5. Interpolate the trajectory to the 128-point supervised value grid.
6. Construct 127 midpoint PDE collocation points.
7. Interpolate `k1`--`k6` from the adaptive trajectory to the PDE midpoints.
8. Save the result in semantic HDF5 groups.

For `N` trajectories and the default 128-point value grid, the in-memory flattened arrays reconstructed by the loader have shapes approximately:

```text
X        : (N * 128, 12)
Y        : (N * 128, 4)
X_pde    : (N * 127, 12)
PDE_aux  : (N * 127, 6)
```

The HDF5 file itself stores trajectory-structured arrays rather than these flattened matrices.

---

## 6. HDF5 structure

The file is divided into three semantic groups plus metadata:

```text
/initial_condition
/value
/pde
/meta
```

### `/initial_condition`

One scalar per trajectory. These quantities are constant for the whole trajectory:

```text
/initial_condition/total_density
/initial_condition/HI_fraction
/initial_condition/HeI_fraction
/initial_condition/HeII_fraction
/initial_condition/internal_energy
/initial_condition/Gamma_HI
/initial_condition/Gamma_HeI
/initial_condition/Gamma_HeII
/initial_condition/pi_HI
/initial_condition/pi_HeI
/initial_condition/pi_HeII
```

The species fractions and internal energy are the relaxed `t=0` conditioning state.

### `/value`

Shape:

```text
(n_trajectories, n_value_points)
```

Datasets:

```text
/value/time
/value/HI_fraction
/value/HeI_fraction
/value/HeII_fraction
/value/internal_energy
```

These are the supervised targets used by the value loss.

### `/pde`

Shape:

```text
(n_trajectories, n_pde_points)
```

Datasets:

```text
/pde/time
/pde/k1
/pde/k2
/pde/k3
/pde/k4
/pde/k5
/pde/k6
```

Temperature is intentionally not stored in the current HDF5 format.

The present reaction mapping is:

```text
k1 / k2 : HI   <-> HII
k3 / k4 : HeI  <-> HeII
k5 / k6 : HeII <-> HeIII
```

### `/meta`

Contains attributes such as:

```text
seed
n_trajectories
n_value_points_per_trajectory
n_pde_points_per_trajectory
value_spacing
interpolation_method
pde_sampling
```

### Quick inspection example

```python
import h5py

with h5py.File("grackle_training_data_10000.h5", "r") as f:
    print(list(f.keys()))
    print(list(f["initial_condition"].keys()))
    print(list(f["value"].keys()))
    print(list(f["pde"].keys()))
    print(dict(f["meta"].attrs))

    print(f["value/HI_fraction"].shape)
    print(f["pde/k1"].shape)
```

---

## 7. Validation setup

The current workflow expects a **precomputed validation HDF5 file**. Validation is not treated as a random-per-epoch dataset.

Configure, for example:

```python
VALIDATION_DATA_FILE = PROJECT_DIR / "grackle_validate_data.h5"
```

The file must already exist before training starts. If `VALIDATION_DATA_FILE` is `None`, or if the configured file does not exist, the trainer should stop with an error instead of silently switching to a different validation mode.

This keeps validation deterministic and separate from `DATA_MODE = "random_epoch"`, which only controls how the **training** data are produced.

If a new validation set is required, generate it explicitly with the same HDF5 schema used for the training data, but with its own seed and parameter samples.

---

## 8. Train the model

With the training HDF5 file already generated, run:

```bash
python train_pinn_v2.py
```

The default training mode is:

```python
DATA_MODE = "precomputed"
```

This is the required mode when:

```python
PDE_ENABLED = True
```

The current prototype deliberately does not support:

```text
DATA_MODE = "random_epoch"
PDE_ENABLED = True
```

because that would regenerate Grackle PDE auxiliary data every epoch.

### Device selection

If `ALLOW_DEVICE=True`, the trainer selects in this order:

1. CUDA GPU
2. Apple MPS
3. CPU

### Optimizer

The current training setup uses:

```text
Adam
+ CosineAnnealingLR
```

with hyperparameters from `config.py`.

---

## 9. Training loss

The total loss is conceptually

```text
L_total = L_value + lambda_PDE * L_species_PDE
```

### Supervised value loss

The network predicts:

```text
HI_fraction
HeI_fraction
HeII_fraction
internal_energy
```

at the 128 supervised time points.

The value loss is the weighted sum of the four normalized-space MSE terms.

### Frozen-coefficient PDE loss

At the 127 midpoint collocation points:

- `HI`, `HeI`, and `HeII` come from the neural network.
- `HII = 1 - HI`.
- `HeIII = 1 - HeI - HeII`.
- Time derivatives are obtained with PyTorch autograd.
- `k1`--`k6` come from the reference Grackle trajectory at the midpoint.
- The species equations are evaluated in Grackle code units.

This is intentionally a staged approximation. The present model does **not** derive `k1`--`k6` from a neural-network temperature.

---

## 10. Normalization

The trainer normalizes the 12 model inputs and four outputs before training.

Fractions are floored and transformed logarithmically. The current mapping uses:

```python
FRACTION_FLOOR = 1.0e-10
FRACTION_NORM_MIN = -2.0
FRACTION_NORM_MAX = 2.0
```

Therefore all fractions below `1e-10` are treated equivalently by the normalized representation.

Positive dimensional inputs use log-normalization relative to the corresponding reference values defined in `config.py`.

The model's hard initial condition operates in this normalized representation.

---

## 11. Training outputs

The output directory is configured by:

```python
OUTPUT_DIR = PROJECT_DIR / "trained_results_v1_n_10000"
```

During training, the code writes:

```text
chemistry_pinn_best_epoch_XXXXX.pt
chemistry_pinn_epoch_XXXXX.pt
chemistry_pinn_final_epoch_XXXXX.pt
validation_epoch_XXXXX.png
```

The "best" model is selected by the supervised validation loss.

Validation loss is evaluated using all validation rows. `VALIDATION_PLOT_N_SAMPLES` limits only how many complete trajectories are displayed in the plot.

---

## 12. Minimal run sequence

For a clean first test:

### Step 1: check paths

In `config.py`, verify:

```python
GRACKLE_DATA_FILE
PRECOMPUTED_DATA_FILE
OUTPUT_DIR
```

For the easiest validation setup:

```python
VALIDATION_DATA_FILE = None
```

### Step 2: reduce the dataset for a smoke test

Before generating 10,000 trajectories, it is useful to test with something small:

```python
PRECOMPUTE_N_TRAJECTORIES = 10
N_EPOCHS = 20
```

### Step 3: generate data

```bash
python precompute_grackle_data.py
```

### Step 4: inspect the HDF5 file

Verify that the groups exist:

```text
/initial_condition
/value
/pde
/meta
```

and that the value/PDE dimensions are 128/127 per trajectory.

### Step 5: train

```bash
python train_pinn_v2.py
```

### Step 6: full run

Once the smoke test works, restore the intended settings, for example:

```python
PRECOMPUTE_N_TRAJECTORIES = 10000
N_EPOCHS = 10000
```

regenerate the HDF5 file, and start the full training run.

---

## 13. Important implementation notes

### The PDE grid is independent of the value grid

The value and PDE losses are intentionally evaluated at different times:

```text
value:  x-----x-----x-----x
PDE:       o-----o-----o
```

The PDE midpoint data are interpolated directly from the original adaptive Grackle trajectory, not from the already interpolated value trajectory.

### `midpoint_pde_collocation_points()` restores the true t=0 conditioning state

`trajectory_to_pinn_arrays()` normally takes the first row of its input as the initial state. For midpoint data, however, the first row is already at `t > 0`.

Therefore `midpoint_pde_collocation_points()` overwrites the conditioning columns with the true `t=0` state from the original adaptive trajectory before returning `X_pde`.

### Constant quantities may still pass through interpolation

Some trajectory arrays such as total density and fixed radiation parameters are constant but are represented as trajectory-length arrays. The generic interpolation utilities may therefore interpolate them as well. Interpolation of an exact constant leaves the value unchanged.

### Temperature is used internally but is not stored in the final HDF5 file

Grackle temperature is still required while generating the trajectory because the reaction rates `k1`--`k6` depend on temperature. Only the final HDF5 storage omits temperature.

---

## 14. Physics convention to verify before final production runs

Grackle species fields use mass-density conventions. The reaction coefficients `k1`--`k6` are expressed in matching Grackle code units.

The HII evolution used by the adaptive timestep has already been checked against Grackle and follows the Grackle mass-density form.

The current PINN species residual reconstructs electron density from the network-predicted ion fractions. Before treating the PDE loss as final physics, verify that `train_pinn_v2.py` uses the exact Grackle `e_density` convention consistently, especially for helium mass-density factors.

For a Grackle convention in which

```text
e_density = n_e * m_H
```

and the helium species fields are helium mass densities, the helium contribution requires the corresponding mass-to-number conversion. This should be checked directly against the Grackle convention used by the installed version.

---

## 15. Suggested development workflow

Because the project is changing quickly, use version control before modifying the physics residual or dataset format:

```bash
git status
git add .
git commit -m "Working frozen-coefficient midpoint PINN baseline"
```

A useful staged development path is:

```text
1. Supervised value loss only
2. Frozen k1--k6 + NN species state
3. NN-derived electron density with verified Grackle mass-density convention
4. Add/verify helium residuals
5. Add energy/cooling physics loss
6. Replace frozen rates with k_i(T_NN) for a fully self-consistent model
```

---

## 16. Quick troubleshooting

### `FileNotFoundError: Missing grackle_training_data_....h5`

Run:

```bash
python precompute_grackle_data.py
```

first.

### Missing validation HDF5 file

Either generate the validation HDF5 file separately or set:

```python
VALIDATION_DATA_FILE = None
```

### `PDE_ENABLED=True currently requires DATA_MODE='precomputed'`

Use:

```python
DATA_MODE = "precomputed"
```

or disable the PDE loss.

### HDF5 dataset-name error after renaming a field

The HDF5 writer and loader must use identical names. If a dataset is renamed under `/initial_condition`, `/value`, or `/pde`, update both the write and load mappings in `grackle_data.py`.

### Training becomes too slow for a smoke test

Temporarily reduce:

```python
PRECOMPUTE_N_TRAJECTORIES
N_EPOCHS
BATCH_SIZE
```

and verify the full pipeline before launching the production-sized run.
