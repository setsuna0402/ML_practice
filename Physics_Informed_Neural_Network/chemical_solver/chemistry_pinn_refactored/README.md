# Primordial Chemistry PINN — proof of concept

## Structure

- `config.py` — single source of truth for parameter ranges, units, interpolation, dataset/training settings, normalisation references.
- `initialise_grackle_fluid.py` — initialise one neutral primordial H/He Grackle cell.
- `grackle_timestep.py` — adaptive HII timestep helper.
- `grackle_data.py` — Grackle setup, trajectory generation, interpolation, random sampling, dataset save/load.
- `normalisation.py` — log10 normalisation/denormalisation for model inputs and outputs.
- `pinn_model.py` — current PINN architecture (unchanged).
- `precompute_grackle_data.py` — generate a fixed `.npz` training dataset.
- `train_pinn.py` — train using either the fixed dataset or fresh trajectories each epoch.
- `CloudyData_UVB=HM2012.h5` — Grackle rate/cooling data (place here).

## Dataset conventions

Input columns:
`time, density, HI0, HeI0, HeII0, u0, Gamma_HI, Gamma_HeI, Gamma_HeII, pi_HI, pi_HeI, pi_HeII`

Targets:
`HI, HeI, HeII, u`

- `density`: Grackle code density; physical density unit is set by `DENSITY_UNIT`.
- `Gamma_*`: dataset stores Grackle code-time rates (1/Myr), though sampling is specified in physical s^-1.
- `pi_*`: eV/s per absorbing atom.
- `u`: exported from Grackle after multiplying by `velocity_units**2`, so it is on the same scale as `U_REF ~ 3.1e12`.
- Fractions use a floor at `1e-10`, followed by `log10` and affine scaling to `[-2, 2]`; positive dimensional variables use `log10(q/q_ref)`.

## Run

1. Put `CloudyData_UVB=HM2012.h5` in this directory.
2. Edit `config.py` only if you want to change ranges/settings.
3. For a fixed dataset:

```bash
python precompute_grackle_data.py
python train_pinn.py
```

4. For fresh data every epoch, set `DATA_MODE = "random_epoch"` in `config.py`, then run:

```bash
python train_pinn.py
```

To validate against a separate precomputed dataset, set `VALIDATION_DATA_FILE`
in `config.py` to its `.npz` path, for example:

```python
VALIDATION_DATA_FILE = PROJECT_DIR / "grackle_validation_data.npz"
```

The file must contain the same `X` and `Y` arrays produced by
`precompute_grackle_data.py`. Leave it as `None` to use the built-in reference
trajectories.

Training outputs are written to `trained_results_v1_a/`.


## Visualise a trained model

Use `visualise_validation.py` to compare a trained ChemistryPINN checkpoint
against the precomputed validation dataset.

Each validation trajectory is saved as one figure with four subplots:

- `HI`
- `HeI`
- `HeII`
- `u`

The Grackle result is shown with a solid line and the PINN prediction with a
dashed line. The figure also displays the trajectory conditioning parameters:
density, initial species fractions, initial internal energy, `Gamma_*`, and
`pi_*`.

Basic usage:

```bash
python visualise_validation.py \
    -I trained_results_v1_a/chemistry_pinn_best_epoch_200.pt \
    -O trained_results_v1_a/validation_plots
```

`-I` / `--checkpoint` specifies the trained `.pt` / `.pth` model checkpoint.

`-O` / `--output-dir` specifies the directory where the trajectory figures are
saved.

By default, the validation dataset is taken from `VALIDATION_DATA_FILE` in
`config.py`. A different validation dataset can be supplied with:

```bash
python visualise_validation.py \
    -I trained_results_v1_a/chemistry_pinn_best_epoch_200.pt \
    -O trained_results_v1_a/validation_plots \
    -D grackle_validate_data.npz
```

For a quick visual check, limit the number of plotted trajectories:

```bash
python visualise_validation.py \
    -I trained_results_v1_a/chemistry_pinn_best_epoch_200.pt \
    -O test_plots \
    --max-trajectories 10
```

The validation `.npz` file stores flattened trajectories. The plotting script
uses `INTERPOLATION_KWARGS["n_times"]` from `config.py` to split the dataset
back into individual trajectories. The corresponding Grackle result is read
directly from the stored `Y` array; Grackle is not rerun during plotting.
