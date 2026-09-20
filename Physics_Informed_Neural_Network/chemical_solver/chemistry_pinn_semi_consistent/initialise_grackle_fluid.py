import numpy as np
import sys

from gracklepy import chemistry_data
from gracklepy.utilities.physical_constants import (
    cm_per_mpc,
    sec_per_Myr,
)
from gracklepy.fluid_container import FluidContainer



def check_convergence(fc1, fc2, fields=None, tol=0.01, flag_debug=False):
    "Copy from gracklepy.fluid_container.check_convergence"
    "This function check the convergence of two fluid containers by comparing the maximum relative difference on density fields"

    if fields is None:
        fields = fc1.density_fields
    max_field = None
    max_val = 0.0 
    for field in fields:
        if field not in fc2:
            continue
        convergence = np.max(np.abs((fc1[field] - fc2[field]) / fc1[field]))
        if convergence > max_val:
            max_val = convergence
            max_field = field
    if np.any(max_val > tol):
        if flag_debug:
            print("max change - %5s: %.10e." % (max_field, max_val))
        return False
    return True

def initialise_fluid_container_neutral(
        my_chemistry, 
        density     : float,  # density in code units
        temperature : float,  # Kelvin
        relaxation  : bool = False , # Whether to do relaxation based on cooling time.
        relax_fraction : float = 0.01, # Fraction of cooling time to use for relaxation timestep.
        tolerance   : float = 0.01, # Tolerance for convergence check during relaxation.
        max_iterations : int = 10000, # Maximum number of iterations for relaxation.
        flag_debug : bool = False
):
    """
    Initialize a fluid container using settings from a chemistry_data object.

    By default, initialize with a constant density and a constant temperature. 
    The state of the gas is assumed to be neutral. 
    All ionized species are set to effectively zero. 
    Molecular fractions are always initialized to effectively zero.
    Metal mass fraction and dust-to-gas ratio are set to effectively zero.
    If relaxation is enabled, the fluid container will be iteratively relaxed to equilibrium at the given temperature.

    Parameters
    ----------
    my_chemistry : chemistry_data
        Struct of Grackle runtime parameters.
    density : float
        The total mass density in code units for all elements in the fluid container.
    temperature : float 
        Temperature values in K.
    relaxation : bool
        Whether to do relaxation at a fixed temperature. The relaxation is based on the cooling time. Default is False.
        Turns a fluid container with the same density and temperature, but with the ionization fractions relaxed to equilibrium.
    max_iterations : int
        Maximum number of iterations for relaxation. Default is 10000.
    Returns
    -------
    fc : FluidContainer
        A fully initialized FluidContainer object.
    """

    rval = my_chemistry.initialize()
    if rval == 0:
        raise RuntimeError("Failed to initialize chemistry_data.")

    tiny_number = 1e-10
    metal_mass_fraction = tiny_number
    dust_to_gas_ratio = tiny_number
    temperature = np.array([temperature])
    n_points = temperature.size

    fc = FluidContainer(my_chemistry, n_points)
    fh = my_chemistry.HydrogenFractionByMass
    d2h = my_chemistry.DeuteriumToHydrogenRatio

    metal_free = 1 - metal_mass_fraction
    H_total = fh * metal_free
    He_total = (1 - fh) * metal_free
    # someday, maybe we'll include D in the total
    D_total = H_total * d2h

    fc_density = density
    tiny_density = tiny_number * fc_density

    state_vals = {
        "density": fc_density,
        "metal_density": metal_mass_fraction * fc_density,
        "dust_density": dust_to_gas_ratio * fc_density
    }
    state_vals["HI_density"] = H_total * fc_density * (1.0 - tiny_number)
    state_vals["HII_density"] = H_total * fc_density * tiny_number
    state_vals["HeI_density"] = He_total * fc_density * (1.0 - 2.0 * tiny_number)
    state_vals["HeII_density"] = He_total * fc_density * tiny_number
    state_vals["HeIII_density"] = He_total * fc_density * tiny_number
    state_vals["DI_density"] = D_total * fc_density * (1.0 - tiny_number)
    state_vals["DII_density"] = D_total * fc_density * tiny_number
    state_vals["e_density"] = state_vals["HII_density"] + \
        state_vals["HeII_density"] + 2.0 * state_vals["HeIII_density"]


    for field in fc.density_fields:
        fc[field][:] = state_vals.get(field, tiny_density)

    fc.calculate_mean_molecular_weight()
    fc["internal_energy"] = temperature / \
        fc.chemistry_data.temperature_units / \
        fc["mean_molecular_weight"] / (my_chemistry.Gamma - 1.0)
    fc["x_velocity"][:] = 0.0
    fc["y_velocity"][:] = 0.0
    fc["z_velocity"][:] = 0.0

    if relaxation:
        fc_last = fc.copy()
        # disable cooling to iterate to equilibrium
        val = fc.chemistry_data.with_radiative_cooling
        fc.chemistry_data.with_radiative_cooling = 0

        my_time = 0.0
        i = 0
        while i < max_iterations:
            fc.calculate_cooling_time()
            dt = relax_fraction * np.abs(fc["cooling_time"]).min()
            if flag_debug:
                print("t: %.3f Myr, dt: %.3e Myr, " % \
                                ((my_time * my_chemistry.time_units / sec_per_Myr),
                                (dt * my_chemistry.time_units / sec_per_Myr)))
            for field in fc.density_fields:
                fc_last[field] = np.copy(fc[field])
            fc.solve_chemistry(dt)
            fc.calculate_mean_molecular_weight()
            fc["internal_energy"] = temperature / \
                fc.chemistry_data.temperature_units / fc["mean_molecular_weight"] / \
                (my_chemistry.Gamma - 1.0)
            if flag_debug:
                print("internal energy: %.10e." % fc["internal_energy"].min())
            converged = check_convergence(fc, fc_last, flag_debug=flag_debug, tol=tolerance)
            if converged:
                if flag_debug:
                    print("Converged after %d iterations." % i)
                    # print("\n")
                break
            my_time += dt
            i += 1

        fc.chemistry_data.with_radiative_cooling = val
        if i >= max_iterations:
            print("Warning: Reached maximum number of iterations (%d) without convergence.\n" % max_iterations)
            print("It is equal to {} times the cooling time of the gas.\n" % (relax_fraction * max_iterations))
            print("The unconverged state of the fluid container is returned if it is not in debug mode.\n")
            if flag_debug:
                raise RuntimeError("Reached maximum number of iterations without convergence.")
    return fc
