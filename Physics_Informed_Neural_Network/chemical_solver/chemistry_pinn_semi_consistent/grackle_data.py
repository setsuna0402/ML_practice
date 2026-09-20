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
import h5py
from scipy.interpolate import CubicSpline, PchipInterpolator

from gracklepy import chemistry_data
from gracklepy.utilities.physical_constants import (
    cm_per_mpc,
    sec_per_Myr,
)

from grackle_timestep import get_HII_timestep, get_primordial_reaction_rates
from initialise_grackle_fluid import initialise_fluid_container_neutral
from config import GAMMA, SAMPLER_RANGES, TRAJECTORY_KWARGS, GRACKLE_DATA_FILE


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
    grackle_data_file=None,
    gamma=GAMMA,
):
    """
    Build the chemistry_data object.

    density_unit is used as the Grackle density code unit.
    """
    if grackle_data_file is None:
        grackle_data_file = str(GRACKLE_DATA_FILE)

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

    # Frozen-coefficient PINN support. k1..k6 are evaluated at the true
    # Grackle reference temperature. Electron density is reconstructed later
    # from the NN-predicted ion fractions, so it is not stored as PDE auxiliary data.
    k1, k2, k3, k4, k5, k6 = get_primordial_reaction_rates(fc)

    return {
        "temperature": float(fc["temperature"][0]),
        # Expose specific internal energy in physical/code-converted units.
        "internal_energy": float(fc["internal_energy"][0] * chem.velocity_units**2),
        "density": float(rho_total),   
        "HI_frac": float(HI_frac),
        "HII_frac": float(HII_frac),
        "HeI_frac": float(HeI_frac),
        "HeII_frac": float(HeII_frac),
        "HeIII_frac": float(HeIII_frac),
        "Gamma_HI": float(Gamma_HI),
        "Gamma_HeI": float(Gamma_HeI),
        "Gamma_HeII": float(Gamma_HeII),
        "Heating_rate": float(pi_total), # erg/sec per HI atom (all species contributions)
        "k1": float(np.asarray(k1).reshape(-1)[0]),
        "k2": float(np.asarray(k2).reshape(-1)[0]),
        "k3": float(np.asarray(k3).reshape(-1)[0]),
        "k4": float(np.asarray(k4).reshape(-1)[0]),
        "k5": float(np.asarray(k5).reshape(-1)[0]),
        "k6": float(np.asarray(k6).reshape(-1)[0]),
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
    final_time=None,
    max_change_HII=0.001,
    cooling_fraction=0.001,
    max_timestep=1.0e-2,
    initial_max_change_HII=1.0e-3,
    grackle_data_file=None,
    relaxation=False , # Whether to do relaxation based on cooling time.
    relax_fraction=0.01, # Fraction of cooling time to use for relaxation timestep.
    tolerance=0.01, # Tolerance for convergence check during relaxation.
    max_iterations=10000, # Maximum number of iterations for relaxation.
    flag_debug=False
    # state="neutral",
    # max_iterations=10000,
):
    """
    Generate one Grackle trajectory.

    Parameters
    ----------
    density : float
        Gas density in Grackle code-density units.
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
    if final_time is None:
        raise RuntimeError("Missing final time.")
    chem = build_chemistry(density_unit=code_density, grackle_data_file=grackle_data_file)

    # density code unit, tem
    fc = initialise_fluid_container_neutral(chem, density, temperature, 
                                            relaxation=relaxation, 
                                            relax_fraction=relax_fraction, 
                                            tolerance=tolerance, 
                                            max_iterations=max_iterations, 
                                            flag_debug=flag_debug)

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
        "Heating_rate": [],
        "k1": [],
        "k2": [],
        "k3": [],
        "k4": [],
        "k5": [],
        "k6": [],
    }

    def record(t, dt):
        s = _current_state(fc, chem)

        history["time"].append(float(t))
        history["dt"].append(float(dt))
        field_name_list = ["temperature", "internal_energy", "density", 
                           "HI_frac", "HII_frac", "HeI_frac", "HeII_frac", "HeIII_frac",
                           "Gamma_HI", "Gamma_HeI", "Gamma_HeII", "Heating_rate",
                           "k1", "k2", "k3", "k4", "k5", "k6"]
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
    density_range=None,
    temperature_range=None,
    Gamma_HI_range=None,
    Gamma_HeI_range=None,
    Gamma_HeII_range=None,
    pi_HI_range=None,
    pi_HeI_range=None,
    pi_HeII_range=None,
    final_time=None,
    max_change_HII=None,
    cooling_fraction=None,
    max_timestep=None,
    initial_max_change_HII=None,
    grackle_data_file=None,
    relaxation=None,
    relax_fraction=None,
    tolerance=None,
    max_iterations=None,
    flag_debug=None,
):
    """Draw one random Grackle trajectory parameter set.

    Positive dimensional parameters are sampled log-uniformly. ``density`` is
    in Grackle code-density units. Gamma values are physical rates [s^-1],
    while pi values are species-specific photoheating rates [eV/s].
    """
    ranges = SAMPLER_RANGES
    density_range = ranges["density_range"] if density_range is None else density_range
    temperature_range = ranges["temperature_range"] if temperature_range is None else temperature_range
    Gamma_HI_range = ranges["Gamma_HI_range"] if Gamma_HI_range is None else Gamma_HI_range
    Gamma_HeI_range = ranges["Gamma_HeI_range"] if Gamma_HeI_range is None else Gamma_HeI_range
    Gamma_HeII_range = ranges["Gamma_HeII_range"] if Gamma_HeII_range is None else Gamma_HeII_range
    pi_HI_range = ranges["pi_HI_range"] if pi_HI_range is None else pi_HI_range
    pi_HeI_range = ranges["pi_HeI_range"] if pi_HeI_range is None else pi_HeI_range
    pi_HeII_range = ranges["pi_HeII_range"] if pi_HeII_range is None else pi_HeII_range

    traj = TRAJECTORY_KWARGS
    final_time = traj["final_time"] if final_time is None else final_time
    max_change_HII = traj["max_change_HII"] if max_change_HII is None else max_change_HII
    cooling_fraction = traj["cooling_fraction"] if cooling_fraction is None else cooling_fraction
    max_timestep = traj["max_timestep"] if max_timestep is None else max_timestep
    initial_max_change_HII = traj["initial_max_change_HII"] if initial_max_change_HII is None else initial_max_change_HII
    relaxation = traj["relaxation"] if relaxation is None else relaxation
    relax_fraction = traj["relax_fraction"] if relax_fraction is None else relax_fraction
    tolerance = traj["tolerance"] if tolerance is None else tolerance
    max_iterations = traj["max_iterations"] if max_iterations is None else max_iterations
    flag_debug = traj["flag_debug"] if flag_debug is None else flag_debug

    return {
        "density": _log_uniform(rng, *density_range),
        "temperature": _log_uniform(rng, *temperature_range),
        "Gamma_HI": _log_uniform(rng, *Gamma_HI_range),
        "Gamma_HeI": _log_uniform(rng, *Gamma_HeI_range),
        "Gamma_HeII": _log_uniform(rng, *Gamma_HeII_range),
        "pi_HI": _log_uniform(rng, *pi_HI_range),
        "pi_HeI": _log_uniform(rng, *pi_HeI_range),
        "pi_HeII": _log_uniform(rng, *pi_HeII_range),
        "code_density": code_density,
        "final_time": final_time,
        "max_change_HII": max_change_HII,
        "cooling_fraction": cooling_fraction,
        "max_timestep": max_timestep,
        "initial_max_change_HII": initial_max_change_HII,
        "grackle_data_file": grackle_data_file,
        "relaxation": relaxation,
        "relax_fraction": relax_fraction,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "flag_debug": flag_debug,
    }


def interpolate_trajectory_at_times(data, new_time, method="pchip"):
    """Interpolate one adaptive Grackle trajectory onto explicit times.

    This is used for frozen-coefficient PDE collocation. We interpolate smooth
    state/rate quantities only; no time derivative is ever interpolated.
    """
    old_time = np.asarray(data["time"], dtype=np.float64)
    new_time = np.asarray(new_time, dtype=np.float64)
    if np.any(new_time < old_time[0]) or np.any(new_time > old_time[-1]):
        raise ValueError("Requested interpolation time lies outside trajectory range.")

    if method == "linear":
        interpolate = lambda values: np.interp(new_time, old_time, values)
    elif method == "cubic":
        if old_time.size < 4:
            raise ValueError("cubic interpolation requires at least four samples")
        interpolate = lambda values: CubicSpline(old_time, values)(new_time)
    elif method == "pchip":
        interpolate = lambda values: PchipInterpolator(old_time, values)(new_time)
    else:
        raise ValueError(f"Unknown interpolation method: {method!r}")

    result = {}
    n_old = old_time.size
    for key, values in data.items():
        values = np.asarray(values)
        if values.ndim == 1 and values.size == n_old:
            result[key] = np.asarray(interpolate(values), dtype=np.float64)
        else:
            result[key] = values.copy() if isinstance(values, np.ndarray) else values
    result["time"] = new_time
    return result


def midpoint_pde_collocation_points(raw_data, value_times, method="pchip"):
    """Create one frozen-coefficient PDE point between every value point.

    For N value times, return N-1 staggered midpoint collocation points:

        t_pde[i] = 0.5 * (t_value[i] + t_value[i+1])

    The model inputs and frozen auxiliary quantities are interpolated from the
    original adaptive Grackle trajectory to these midpoint times. Derivatives
    are not stored or interpolated.

    PDE_aux columns:
        [k1, k2, k3, k4, k5, k6]

    Electron density is not stored. It is reconstructed self-consistently
    from the NN-predicted H/He ion fractions and the input gas density.
    """
    value_times = np.asarray(value_times, dtype=np.float64)
    if value_times.size < 2:
        raise ValueError("Need at least two value times to define midpoint PDE points.")

    midpoint_times = 0.5 * (value_times[:-1] + value_times[1:])
    pde_data = interpolate_trajectory_at_times(
        raw_data, midpoint_times, method=method
    )
    X_pde, _ = trajectory_to_pinn_arrays(pde_data)

    # trajectory_to_pinn_arrays normally takes the first row of its input as
    # the initial condition. Here the first row is already a midpoint, so put
    # back the true t=0 conditioning state from the original trajectory.
    X_pde[:, 2] = raw_data["HI_frac"][0]
    X_pde[:, 3] = raw_data["HeI_frac"][0]
    X_pde[:, 4] = raw_data["HeII_frac"][0]
    X_pde[:, 5] = raw_data["internal_energy"][0]

    aux_pde = np.column_stack([
        pde_data[f"k{i}"] for i in range(1, 7)
    ]).astype(np.float64)
    return X_pde.astype(np.float64), aux_pde


def generate_random_dataset_with_pde(
    n_trajectories,
    *,
    code_density,
    rng=None,
    sampler_kwargs=None,
    interpolate=True,
    interpolation_kwargs=None,
):
    """Generate value targets plus staggered midpoint frozen-coefficient PDE points."""
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be >= 1")
    if not interpolate:
        raise ValueError("Midpoint PDE sampling requires the fixed value-time grid.")

    rng = np.random.default_rng() if rng is None else rng
    sampler_kwargs = {} if sampler_kwargs is None else dict(sampler_kwargs)
    interpolation_kwargs = {} if interpolation_kwargs is None else dict(interpolation_kwargs)
    method = interpolation_kwargs.get("method", "pchip")

    input_arrays = []
    target_arrays = []
    pde_input_arrays = []
    pde_aux_arrays = []
    parameter_sets = []

    for counter in range(n_trajectories):
        print("trajectory {}/{}".format(counter + 1, n_trajectories))
        parameters = sample_parameter_set(rng, code_density=code_density, **sampler_kwargs)
        raw_trajectory = generate_grackle_trajectory(**parameters)

        value_trajectory = interpolate_trajectory(raw_trajectory, **interpolation_kwargs)
        X, Y = trajectory_to_pinn_arrays(value_trajectory)
        X_pde, aux_pde = midpoint_pde_collocation_points(
            raw_trajectory, value_trajectory["time"], method=method
        )

        input_arrays.append(X)
        target_arrays.append(Y)
        pde_input_arrays.append(X_pde)
        pde_aux_arrays.append(aux_pde)
        parameter_sets.append(parameters)

    return (
        np.concatenate(input_arrays, axis=0),
        np.concatenate(target_arrays, axis=0),
        np.concatenate(pde_input_arrays, axis=0),
        np.concatenate(pde_aux_arrays, axis=0),
        parameter_sets,
    )


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

    counter = 0
    for _ in range(n_trajectories):
        print("trajectory {}/{}".format(counter + 1, n_trajectories))
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
        counter += 1
    return (
        np.concatenate(input_arrays, axis=0),
        np.concatenate(target_arrays, axis=0),
        parameter_sets,
    )


MODEL_INPUT_NAMES = (
    "time", "density", "HI_fraction", "HeI_fraction", "HeII_fraction", "internal_energy",
    "Gamma_HI", "Gamma_HeI", "Gamma_HeII", "pi_HI", "pi_HeI", "pi_HeII",
)
MODEL_TARGET_NAMES = ("HI_fraction", "HeI_fraction", "HeII_fraction", "internal_energy")

# HDF5 names are deliberately explicit.  Initial conditions are stored once
# per trajectory; /value and /pde contain only quantities that vary along their
# respective time grids.
INITIAL_CONDITION_NAMES = (
    "total_density",
    "HI_fraction",
    "HeI_fraction",
    "HeII_fraction",
    "internal_energy",
    "Gamma_HI",
    "Gamma_HeI",
    "Gamma_HeII",
    "pi_HI",
    "pi_HeI",
    "pi_HeII",
)
VALUE_NAMES = (
    "time",
    "HI_fraction",
    "HeI_fraction",
    "HeII_fraction",
    "internal_energy",
)
PDE_NAMES = ("time", "k1", "k2", "k3", "k4", "k5", "k6")


def _write_dataset(group, name, data, *, compression="gzip"):
    """Write one named HDF5 dataset using float64 storage."""
    group.create_dataset(
        name,
        data=np.asarray(data, dtype=np.float64),
        compression=compression,
        shuffle=True,
    )


def _require_datasets(group, names):
    missing = [name for name in names if name not in group]
    if missing:
        raise ValueError(f"Missing HDF5 datasets in {group.name}: {missing}")


def _trajectory_shapes(n_trajectories, X, X_pde=None):
    """Infer fixed value/PDE points per trajectory and validate divisibility."""
    if X.shape[0] % n_trajectories != 0:
        raise ValueError(
            f"Value rows ({X.shape[0]}) are not divisible by "
            f"n_trajectories={n_trajectories}."
        )
    n_value = X.shape[0] // n_trajectories

    n_pde = None
    if X_pde is not None:
        if X_pde.shape[0] % n_trajectories != 0:
            raise ValueError(
                f"PDE rows ({X_pde.shape[0]}) are not divisible by "
                f"n_trajectories={n_trajectories}."
            )
        n_pde = X_pde.shape[0] // n_trajectories
    return n_value, n_pde


def _write_precomputed_hdf5(
    h5,
    X,
    Y,
    n_trajectories,
    *,
    X_pde=None,
    PDE_aux=None,
):
    """Write the semantic /initial_condition, /value and /pde layout."""
    n_value, n_pde = _trajectory_shapes(n_trajectories, X, X_pde)

    X3 = np.asarray(X, dtype=np.float64).reshape(n_trajectories, n_value, -1)
    Y3 = np.asarray(Y, dtype=np.float64).reshape(n_trajectories, n_value, -1)

    initial = h5.create_group("initial_condition")
    value = h5.create_group("value")

    # All entries below are constant along one trajectory, so store only once.
    initial_columns = {
        "total_density": 1,
        "HI_fraction": 2,
        "HeI_fraction": 3,
        "HeII_fraction": 4,
        "internal_energy": 5,
        "Gamma_HI": 6,
        "Gamma_HeI": 7,
        "Gamma_HeII": 8,
        "pi_HI": 9,
        "pi_HeI": 10,
        "pi_HeII": 11,
    }
    for name, column in initial_columns.items():
        _write_dataset(initial, name, X3[:, 0, column])

    _write_dataset(value, "time", X3[:, :, 0])
    _write_dataset(value, "HI_fraction", Y3[:, :, 0])
    _write_dataset(value, "HeI_fraction", Y3[:, :, 1])
    _write_dataset(value, "HeII_fraction", Y3[:, :, 2])
    _write_dataset(value, "internal_energy", Y3[:, :, 3])

    if X_pde is not None:
        Xp3 = np.asarray(X_pde, dtype=np.float64).reshape(n_trajectories, n_pde, -1)
        aux3 = np.asarray(PDE_aux, dtype=np.float64).reshape(n_trajectories, n_pde, -1)
        pde = h5.create_group("pde")
        _write_dataset(pde, "time", Xp3[:, :, 0])
        for index, name in enumerate(("k1", "k2", "k3", "k4", "k5", "k6")):
            _write_dataset(pde, name, aux3[:, :, index])

    return n_value, n_pde


def generate_precomputed_dataset(
    output_file,
    n_trajectories,
    *,
    code_density,
    seed=1234,
    sampler_kwargs=None,
    interpolate=True,
    interpolation_kwargs=None,
    include_pde=False,
):
    """Generate and save a fixed random dataset as a semantic HDF5 file.

    Layout
    ------
    /initial_condition/*
        One scalar per trajectory.  This includes the relaxed t=0 species
        fractions, u0, total density, and the fixed radiation parameters.

    /value/*
        Shape (n_trajectories, n_value_points).  Contains only the supervised
        value grid and targets.

    /pde/*
        Shape (n_trajectories, n_pde_points).  Contains only midpoint PDE times
        and frozen k1..k6 coefficients.  Temperature is intentionally not saved.

    The public return values remain X/Y(/X_pde/PDE_aux) so the current trainer
    API does not change.
    """
    rng = np.random.default_rng(seed)
    print("Generating precomputed dataset with {} trajectories".format(n_trajectories))

    output_file = str(output_file)
    if include_pde:
        X, Y, X_pde, PDE_aux, _ = generate_random_dataset_with_pde(
            n_trajectories,
            code_density=code_density,
            rng=rng,
            sampler_kwargs=sampler_kwargs,
            interpolate=interpolate,
            interpolation_kwargs=interpolation_kwargs,
        )
    else:
        X, Y, _ = generate_random_dataset(
            n_trajectories,
            code_density=code_density,
            rng=rng,
            sampler_kwargs=sampler_kwargs,
            interpolate=interpolate,
            interpolation_kwargs=interpolation_kwargs,
        )
        X_pde = PDE_aux = None

    with h5py.File(output_file, "w") as h5:
        n_value, n_pde = _write_precomputed_hdf5(
            h5,
            X,
            Y,
            n_trajectories,
            X_pde=X_pde,
            PDE_aux=PDE_aux,
        )

        meta = h5.create_group("meta")
        meta.attrs["seed"] = int(seed)
        meta.attrs["n_trajectories"] = int(n_trajectories)
        meta.attrs["n_value_points_per_trajectory"] = int(n_value)
        meta.attrs["initial_condition_names"] = np.asarray(INITIAL_CONDITION_NAMES, dtype="S")
        meta.attrs["value_names"] = np.asarray(VALUE_NAMES, dtype="S")

        if interpolation_kwargs is not None:
            if "spacing" in interpolation_kwargs:
                meta.attrs["value_spacing"] = str(interpolation_kwargs["spacing"])
            if "method" in interpolation_kwargs:
                meta.attrs["interpolation_method"] = str(interpolation_kwargs["method"])

        if include_pde:
            meta.attrs["pde_sampling"] = "value_midpoints"
            meta.attrs["n_pde_points_per_trajectory"] = int(n_pde)
            meta.attrs["pde_names"] = np.asarray(PDE_NAMES, dtype="S")

    if include_pde:
        return X, Y, X_pde, PDE_aux
    return X, Y


def _load_initial_condition(h5):
    if "initial_condition" not in h5:
        raise ValueError("HDF5 file does not contain /initial_condition group.")
    group = h5["initial_condition"]
    _require_datasets(group, INITIAL_CONDITION_NAMES)
    return {name: np.asarray(group[name], dtype=np.float64) for name in INITIAL_CONDITION_NAMES}


def _repeat_initial_columns(initial, n_points):
    """Return trajectory-level conditioning values repeated on one time grid."""
    return {
        name: np.repeat(values[:, None], n_points, axis=1)
        for name, values in initial.items()
    }


def load_precomputed_dataset(path, *, include_pde=False):
    """Load semantic HDF5 data and reconstruct the current trainer matrices."""
    with h5py.File(path, "r") as h5:
        initial = _load_initial_condition(h5)

        if "value" not in h5:
            raise ValueError(f"{path} does not contain /value group.")
        value = h5["value"]
        _require_datasets(value, VALUE_NAMES)

        value_time = np.asarray(value["time"], dtype=np.float64)
        if value_time.ndim != 2:
            raise ValueError(f"/value/time must be 2-D, got {value_time.shape}.")
        n_trajectories, n_value = value_time.shape
        if initial["total_density"].shape != (n_trajectories,):
            raise ValueError("/initial_condition length does not match /value trajectories.")

        init_value = _repeat_initial_columns(initial, n_value)
        X = np.stack([
            value_time,
            init_value["total_density"],
            init_value["HI_fraction"],
            init_value["HeI_fraction"],
            init_value["HeII_fraction"],
            init_value["internal_energy"],
            init_value["Gamma_HI"],
            init_value["Gamma_HeI"],
            init_value["Gamma_HeII"],
            init_value["pi_HI"],
            init_value["pi_HeI"],
            init_value["pi_HeII"],
        ], axis=-1).reshape(-1, len(MODEL_INPUT_NAMES))

        Y = np.stack([
            np.asarray(value["HI_fraction"], dtype=np.float64),
            np.asarray(value["HeI_fraction"], dtype=np.float64),
            np.asarray(value["HeII_fraction"], dtype=np.float64),
            np.asarray(value["internal_energy"], dtype=np.float64),
        ], axis=-1).reshape(-1, len(MODEL_TARGET_NAMES))

        if not include_pde:
            return X, Y

        if "pde" not in h5:
            raise ValueError(
                f"{path} does not contain /pde frozen-coefficient data. "
                "Regenerate it with include_pde=True."
            )
        pde = h5["pde"]
        _require_datasets(pde, PDE_NAMES)
        pde_time = np.asarray(pde["time"], dtype=np.float64)
        if pde_time.ndim != 2 or pde_time.shape[0] != n_trajectories:
            raise ValueError(
                f"/pde/time must have shape (n_trajectories, n_pde), got {pde_time.shape}."
            )
        n_pde = pde_time.shape[1]
        init_pde = _repeat_initial_columns(initial, n_pde)
        X_pde = np.stack([
            pde_time,
            init_pde["total_density"],
            init_pde["HI_fraction"],
            init_pde["HeI_fraction"],
            init_pde["HeII_fraction"],
            init_pde["internal_energy"],
            init_pde["Gamma_HI"],
            init_pde["Gamma_HeI"],
            init_pde["Gamma_HeII"],
            init_pde["pi_HI"],
            init_pde["pi_HeI"],
            init_pde["pi_HeII"],
        ], axis=-1).reshape(-1, len(MODEL_INPUT_NAMES))

        PDE_aux = np.stack([
            np.asarray(pde[name], dtype=np.float64)
            for name in ("k1", "k2", "k3", "k4", "k5", "k6")
        ], axis=-1).reshape(-1, 6)

    return X, Y, X_pde, PDE_aux
