"""
Grackle data-generation utilities for the primordial 6-species network.

This module intentionally does NOT decide the training parameter ranges.
It accepts one physical parameter set and returns a trajectory.

Based on the user's tested gracklepy wrapper:
- primordial_chemistry = 1
- Gamma = 5/3
- external HI / HeI / HeII photoionization rates
- species-specific photoheating inputs from HM2012-like tables
- adaptive HII timestep from grackle_timestep.get_HII_timestep
- cooling-time timestep limiter

Units exposed by this module
----------------------------
density           : mass density, code units (~ g / cm^3)
temperature       : K
Gamma_*           : 1 / s / absorbing atom
pi_*              : eV / s / absorbing atom
time              : code units (~ Myr)
specific energy u : code units (decided by gracklepy, code_length^2 / code_time^2, (non-cosmological))
"""

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from gracklepy import chemistry_data, setup_fluid_container
from gracklepy.utilities.physical_constants import (
    cm_per_mpc,
    sec_per_Myr,
)

from grackle_timestep import get_HII_timestep
from initialise_grackle_fluid import initialise_fluid_container_neutral


EV_TO_ERG = 1.602176634e-12


def total_heating_rate_per_HI(
    rho_HI, rho_HeI, rho_HeII,
    pi_HI, pi_HeI, pi_HeII,
):
    """
    Convert species-specific heating rates to Grackle RT_heating_rate.

    rho_* are Grackle species mass densities.
    The 0.25 factors convert helium mass density to a hydrogen-mass
    equivalent number density because a He atom has ~4 times the mass.

    pi_* are in eV / s / absorbing atom.

    Returns
    -------
    heating : ndarray
        erg / s / HI atom, as expected by RT_heating_rate.
    """
    H_total = rho_HI * pi_HI + 0.25 * rho_HeI * pi_HeI + 0.25 * rho_HeII * pi_HeII

    return (H_total / rho_HI) * EV_TO_ERG


def build_chemistry(
    density_unit,
    grackle_data_file="./CloudyData_UVB=HM2012.h5",
    gamma=5.0 / 3.0,
):
    """
    Build the chemistry_data object.

    density_unit is used as the Grackle density code unit.
    """
    chem = chemistry_data()

    chem.use_grackle = 1
    chem.with_radiative_cooling = 1    # enable radiative cooling and heating
    chem.use_radiative_transfer = 1    # enable radiative transfer (RT) fields
    chem.primordial_chemistry = 1      # 6-species network (HI, HII, HeI, HeII, HeIII, e-)
    chem.metal_cooling = 0
    chem.UVbackground = 0              # no UV background, we provide our own RT fields
    chem.Gamma = gamma
    chem.grackle_data_file = grackle_data_file

    chem.comoving_coordinates = 0      # non-cosmological
    chem.self_shielding_method = 0     # no self-shielding
    chem.a_units = 1.0
    chem.a_value = 1.0

    chem.density_units = density_unit  # code_density
    chem.length_units = cm_per_mpc     # code_length
    chem.time_units = sec_per_Myr      # code_time
    chem.set_velocity_units()          

    return chem


def _set_radiation_fields(
    fc, chem, 
    Gamma_HI, Gamma_HeI, Gamma_HeII,
    pi_HI, pi_HeI, pi_HeII,
):
    """
    Set ionization and heating fields on the FluidContainer.
    """
    shape = fc["RT_HI_ionization_rate"].shape
    ones = np.ones(shape)

    # Grackle RT ionization rates use 1 / code_time.
    fc["RT_HI_ionization_rate"] = ones * Gamma_HI * chem.time_units
    fc["RT_HeI_ionization_rate"] = ones * Gamma_HeI * chem.time_units
    fc["RT_HeII_ionization_rate"] = ones * Gamma_HeII * chem.time_units

    fc["RT_heating_rate"] = total_heating_rate_per_HI(
        fc["HI_density"], fc["HeI_density"], fc["HeII_density"],
        pi_HI, pi_HeI, pi_HeII,
    )


def _update_heating_field(fc, pi_HI, pi_HeI, pi_HeII,):
    """
    Update Grackle's HI-normalized heating field after species change.
    """
    fc["RT_heating_rate"] = total_heating_rate_per_HI(
        fc["HI_density"], fc["HeI_density"], fc["HeII_density"],
        pi_HI, pi_HeI, pi_HeII,
    )


def _current_state(fc, chem):
    """
    Return the current physical state as scalar values.

    This draft assumes a one-cell FluidContainer.
    """
    fc.calculate_temperature()

    rho_H = fc["HI_density"][0] + fc["HII_density"][0]
    rho_He = fc["HeI_density"][0] + fc["HeII_density"][0] + fc["HeIII_density"][0]
    rho_total = rho_H + rho_He

    HI_frac = fc["HI_density"][0] / rho_H
    HII_frac = fc["HII_density"][0] / rho_H

    HeI_frac = fc["HeI_density"][0] / rho_He
    HeII_frac = fc["HeII_density"][0] / rho_He
    HeIII_frac = fc["HeIII_density"][0] / rho_He

    Gamma_HI = fc["RT_HI_ionization_rate"][0]
    Gamma_HeI = fc["RT_HeI_ionization_rate"][0]
    Gamma_HeII = fc["RT_HeII_ionization_rate"][0]
    pi_total = fc["RT_heating_rate"][0]

    return {
        "temperature": float(fc["temperature"][0]),
        "internal_energy": float(fc["internal_energy"][0]),
        "density": float(rho_total),   
        "HI_frac": float(HI_frac),
        "HII_frac": float(HII_frac),
        "HeI_frac": float(HeI_frac),
        "HeII_frac": float(HeII_frac),
        "HeIII_frac": float(HeIII_frac),
        "Gamma_HI": float(Gamma_HI),
        "Gamma_HeI": float(Gamma_HeI),
        "Gamma_HeII": float(Gamma_HeII),
        "Heating_rate": float(pi_total) # erg/sec per HI atom (all species contributions)
    }


def generate_grackle_trajectory(
    density,
    temperature,
    Gamma_HI,
    Gamma_HeI,
    Gamma_HeII,
    pi_HI,
    pi_HeI,
    pi_HeII,
    code_density=None,
    final_time=1.0,
    max_change_HII=0.001,
    cooling_fraction=0.001,
    max_timestep=1.0e-2,
    initial_max_change_HII=1.0e-3,
    grackle_data_file="./CloudyData_UVB=HM2012.h5",
    # state="neutral",
    # max_iterations=10000,
):
    """
    Generate one Grackle trajectory.

    Parameters
    ----------
    density : float
        Physical mass density [g / cm^3].
    temperature : float
        Initial temperature [K].
    Gamma_* : float
        Photoionization rates [1 / s / absorbing atom].
    pi_* : float
        Photoheating rates [eV / s / absorbing atom].
    final_time : float
        End time [Myr].

    Returns
    -------
    data : dict[str, np.ndarray]
        One-dimensional arrays containing the trajectory.

    Notes
    -----
    The current draft keeps setup_fluid_container(..., state=state).
    Arbitrary initial ion fractions can be added later once the preferred
    sampling strategy has been decided.
    """
    if code_density is None:
        raise RuntimeError("Missing code density unit.")
    chem = build_chemistry(density_unit=code_density, grackle_data_file=grackle_data_file)

    # density code unit, tem
    fc = initialise_fluid_container_neutral(chem, density, temperature)

    _set_radiation_fields(fc, chem, Gamma_HI, Gamma_HeI, Gamma_HeII, pi_HI, pi_HeI, pi_HeII)

    history = {
        "time": [],
        "dt": [],
        "density": [],
        "temperature": [],
        "internal_energy": [],
        "HI_frac": [],
        "HII_frac": [],
        "HeI_frac": [],
        "HeII_frac": [],
        "HeIII_frac": [],
        "Gamma_HI": [],
        "Gamma_HeI": [],
        "Gamma_HeII": [],
        "Heating_rate": []
    }

    def record(t, dt):
        s = _current_state(fc, chem)

        history["time"].append(float(t))
        history["dt"].append(float(dt))
        field_name_list = ["temperature", "internal_energy", "density", 
                           "HI_frac", "HII_frac", "HeI_frac", "HeII_frac", "HeIII_frac",
                           "Gamma_HI", "Gamma_HeI", "Gamma_HeII", "Heating_rate"]
        for key in field_name_list:
            history[key].append(s[key])

    t_total = 0.0

    dt = get_HII_timestep(fc, max_change=initial_max_change_HII)

    dt = min(dt, max_timestep, final_time)
    record(t_total, dt)  # initial time and initial timestep

    while t_total < final_time:
        fc.solve_chemistry(dt)
        t_total += dt

        _update_heating_field(fc, pi_HI, pi_HeI, pi_HeII)

        if t_total >= final_time:
            record(t_total, 0.0)
            break

        fc.calculate_temperature()
        fc.calculate_cooling_time()

        dt_HII = get_HII_timestep(fc, max_change=max_change_HII)

        dt_cooling = (cooling_fraction * abs(float(fc["cooling_time"][0])))

        dt = min(dt_HII, dt_cooling, max_timestep, final_time - t_total)

        record(t_total, dt)

    for key in history:
        history[key] = np.asarray(history[key], dtype=np.float64)

    # Store the constant external parameters as trajectory-length arrays.
    n = history["time"].size
    '''
    history["Gamma_HI"] = np.full(n, Gamma_HI)
    history["Gamma_HeI"] = np.full(n, Gamma_HeI)
    history["Gamma_HeII"] = np.full(n, Gamma_HeII)
    '''
    # In principle, these could be time-dependent, but for now we keep them constant.
    # Also store the total heating rate, which is time-dependent because the species densities change.
    history["pi_HI"] = np.full(n, pi_HI)
    history["pi_HeI"] = np.full(n, pi_HeI)
    history["pi_HeII"] = np.full(n, pi_HeII)

    return history


def trajectory_to_pinn_arrays(data):
    """
    Convert one trajectory to simple supervised/PINN arrays.

    Every row uses the trajectory's t=0 state as the initial condition.

    X columns
    ---------
    time, density,
    HI0, HeI0, HeII0, u0,
    Gamma_HI, Gamma_HeI, Gamma_HeII,
    pi_HI, pi_HeI, pi_HeII

    Y columns
    ---------
    HI, HeI, HeII, u
    """
    n = data["time"].size

    HI0 = data["HI_frac"][0]
    HeI0 = data["HeI_frac"][0]
    HeII0 = data["HeII_frac"][0]
    u0 = data["internal_energy"][0]

    X = np.column_stack(
        (
            data["time"],
            data["density"],
            np.full(n, HI0),
            np.full(n, HeI0),
            np.full(n, HeII0),
            np.full(n, u0),
            data["Gamma_HI"],
            data["Gamma_HeI"],
            data["Gamma_HeII"],
            data["pi_HI"],
            data["pi_HeI"],
            data["pi_HeII"],
        )
    )

    Y = np.column_stack(
        (
            data["HI_frac"],
            data["HeI_frac"],
            data["HeII_frac"],
            data["internal_energy"],
        )
    )

    return X.astype(np.float64), Y.astype(np.float64)


def interpolate_trajectory(
    data,
    *,
    n_times=128,
    spacing="uniform",
    method="pchip",
    log_start_time=None,
):
    """Interpolate one adaptive trajectory onto a time grid.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        One trajectory returned by :func:`generate_grackle_trajectory`.
    n_times : int
        Number of output times, including both endpoints.
    spacing : {"uniform", "log-uniform"}
        Time-grid spacing. A log-uniform grid includes ``t=0`` explicitly,
        followed by logarithmically spaced positive times. By default, the
        first positive adaptive sample is used as the lower log-grid limit.
    method : {"linear", "cubic", "pchip"}
        Interpolation method used for every trajectory field.
    log_start_time : float or None
        Positive lower limit for the log-uniform grid. If ``None``, use the
        first positive trajectory time, or ``final_time / 10`` for a
        two-sample trajectory.

    Returns
    -------
    dict[str, np.ndarray]
        A copy of ``data`` with every trajectory-length field interpolated
        onto the new time grid. Scalar metadata, if present, is copied.
    """
    if not isinstance(n_times, (int, np.integer)) or n_times < 2:
        raise ValueError("n_times must be an integer >= 2")
    if spacing not in {"uniform", "log-uniform"}:
        raise ValueError("spacing must be 'uniform' or 'log-uniform'")
    if method not in {"linear", "cubic", "pchip"}:
        raise ValueError("method must be 'linear', 'cubic', or 'pchip'")

    old_time = np.asarray(data["time"], dtype=np.float64)
    if old_time.ndim != 1 or old_time.size < 2:
        raise ValueError("trajectory time must contain at least two values")
    if np.any(np.diff(old_time) <= 0.0):
        raise ValueError("trajectory time must be strictly increasing")

    start_time = old_time[0]
    final_time = old_time[-1]
    if spacing == "uniform":
        new_time = np.linspace(start_time, final_time, n_times)
    else:
        if start_time < 0.0 or final_time <= 0.0:
            raise ValueError("log-uniform time requires t >= 0 and final time > 0")
        if log_start_time is None:
            positive_start = old_time[1] if start_time == 0.0 else start_time
            if positive_start >= final_time:
                positive_start = final_time / 10.0
        else:
            positive_start = float(log_start_time)
        if positive_start <= 0.0 or positive_start > final_time:
            raise ValueError("log_start_time must be positive and <= final time")
        positive_grid = np.geomspace(positive_start, final_time, n_times - 1)
        new_time = np.concatenate(([start_time], positive_grid))

    if method == "linear":
        interpolate = lambda values: np.interp(new_time, old_time, values)
    else:
        if method == "cubic":
            if old_time.size < 4:
                raise ValueError("cubic interpolation requires at least four samples")
            interpolator_type = CubicSpline
        else:
            interpolator_type = PchipInterpolator

        def interpolate(values):
            return interpolator_type(old_time, values)(new_time)

    result = {}
    trajectory_length = old_time.size
    for key, values in data.items():
        values = np.asarray(values)
        if values.ndim == 1 and values.size == trajectory_length:
            result[key] = np.asarray(interpolate(values), dtype=np.float64)
        else:
            result[key] = values.copy() if isinstance(values, np.ndarray) else values

    result["time"] = new_time.astype(np.float64)
    return result




# ---------------------------------------------------------------------
# Dataset sampling utilities
# ---------------------------------------------------------------------

def _log_uniform(rng, low, high):
    """Sample uniformly in log10-space between two positive limits."""
    if low <= 0.0 or high <= 0.0:
        raise ValueError("log-uniform limits must be positive")
    return 10.0 ** rng.uniform(np.log10(low), np.log10(high))


def sample_parameter_set(
    rng,
    *,
    code_density,
    density_range=(0.01, 100.0),
    temperature_range=(1.0e2, 1.0e4),
    Gamma_HI_range=(1.0e-15, 1.0e-11),
    Gamma_HeI_range=(1.0e-15, 1.0e-11),
    Gamma_HeII_range=(1.0e-17, 1.0e-12),
    excess_energy_HI_range=(1.0e-14, 1.0e-10),
    excess_energy_HeI_range=(1.0e-14, 1.0e-10),
    excess_energy_HeII_range=(1.0e-14, 1.0e-12),
    final_time=1.0,
    max_change_HII=0.001,
    cooling_fraction=0.001,
    max_timestep=1.0e-2,
    initial_max_change_HII=1.0e-3,
    grackle_data_file="./CloudyData_UVB=HM2012.h5",
):
    """
    Draw one random Grackle trajectory parameter set.

    Notes
    -----
    ``density`` follows the convention already used by this module: it is the
    FluidContainer density in Grackle code-density units. ``code_density`` is
    the physical density represented by density=1.

    The three photoionization rates are sampled log-uniformly.  Instead of
    sampling photoheating rates independently, an excess energy per ionization
    is sampled and ``pi = Gamma * E_excess``.  This keeps the random radiation
    fields reasonably well behaved for the proof-of-concept while still
    allowing spectral hardness to vary.
    """
    density = _log_uniform(rng, *density_range)
    temperature = _log_uniform(rng, *temperature_range)

    Gamma_HI = _log_uniform(rng, *Gamma_HI_range)
    Gamma_HeI = _log_uniform(rng, *Gamma_HeI_range)
    Gamma_HeII = _log_uniform(rng, *Gamma_HeII_range)

    e_HI = rng.uniform(*excess_energy_HI_range)
    e_HeI = rng.uniform(*excess_energy_HeI_range)
    e_HeII = rng.uniform(*excess_energy_HeII_range)

    return {
        "density": density,
        "temperature": temperature,
        "Gamma_HI": Gamma_HI,
        "Gamma_HeI": Gamma_HeI,
        "Gamma_HeII": Gamma_HeII,
        "pi_HI": Gamma_HI * e_HI,
        "pi_HeI": Gamma_HeI * e_HeI,
        "pi_HeII": Gamma_HeII * e_HeII,
        "code_density": code_density,
        "final_time": final_time,
        "max_change_HII": max_change_HII,
        "cooling_fraction": cooling_fraction,
        "max_timestep": max_timestep,
        "initial_max_change_HII": initial_max_change_HII,
        "grackle_data_file": grackle_data_file,
    }


def generate_random_dataset(
    n_trajectories,
    *,
    code_density,
    rng=None,
    sampler_kwargs=None,
    interpolate=True,
    interpolation_kwargs=None,
):
    """Generate and concatenate randomly sampled Grackle trajectories.

    By default, each adaptive trajectory is interpolated independently before
    it is converted to flattened PINN arrays. Set ``interpolate=False`` to
    retain the raw adaptive output.
    """
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be >= 1")

    rng = np.random.default_rng() if rng is None else rng
    sampler_kwargs = {} if sampler_kwargs is None else dict(sampler_kwargs)
    interpolation_kwargs = (
        {} if interpolation_kwargs is None else dict(interpolation_kwargs)
    )

    input_arrays = []
    target_arrays = []
    parameter_sets = []

    for _ in range(n_trajectories):
        parameters = sample_parameter_set(
            rng,
            code_density=code_density,
            **sampler_kwargs,
        )
        trajectory = generate_grackle_trajectory(**parameters)
        if interpolate:
            trajectory = interpolate_trajectory(trajectory, **interpolation_kwargs)
        X, Y = trajectory_to_pinn_arrays(trajectory)
        input_arrays.append(X)
        target_arrays.append(Y)
        parameter_sets.append(parameters)

    return (
        np.concatenate(input_arrays, axis=0),
        np.concatenate(target_arrays, axis=0),
        parameter_sets,
    )


def generate_precomputed_dataset(
    output_file,
    n_trajectories,
    *,
    code_density,
    seed=1234,
    sampler_kwargs=None,
    interpolate=True,
    interpolation_kwargs=None,
):
    """Generate and save a fixed random dataset as a compressed NPZ file."""
    rng = np.random.default_rng(seed)
    X, Y, _ = generate_random_dataset(
        n_trajectories,
        code_density=code_density,
        rng=rng,
        sampler_kwargs=sampler_kwargs,
        interpolate=interpolate,
        interpolation_kwargs=interpolation_kwargs,
    )

    np.savez_compressed(
        output_file,
        X=X,
        Y=Y,
        seed=np.int64(seed),
        n_trajectories=np.int64(n_trajectories),
    )
    return X, Y


def load_precomputed_dataset(path):
    """Load ``X`` and ``Y`` from a dataset produced above."""
    with np.load(path) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        Y = np.asarray(data["Y"], dtype=np.float64)
    return X, Y
