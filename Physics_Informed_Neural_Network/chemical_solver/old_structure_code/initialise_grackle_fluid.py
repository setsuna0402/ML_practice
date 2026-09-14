import numpy as np

from gracklepy import chemistry_data
from gracklepy.utilities.physical_constants import (
    cm_per_mpc,
    sec_per_Myr,
)
from gracklepy.fluid_container import FluidContainer

def initialise_fluid_container_neutral(
        my_chemistry, 
        density     : float,  # density in code units
        temperature : float,  # Kelvin
):
    """
    Initialize a fluid container using settings from a chemistry_data object.

    By default, initialize with a constant density and a constant temperature. 
    The state of the gas is assumed to be neutral. 
    All ionized species are set to effectively zero. 
    Molecular fractions are always initialized to effectively zero.
    Metal mass fraction and dust-to-gas ratio are set to effectively zero.

    Parameters
    ----------
    my_chemistry : chemistry_data
        Struct of Grackle runtime parameters.
    density : float
        The total mass density in code units for all elements in the fluid container.
    temperature : float 
        Temperature values in K.

    Returns
    -------
    fc : FluidContainer
        A fully initialized FluidContainer object.
    """

    rval = my_chemistry.initialize()
    if rval == 0:
        raise RuntimeError("Failed to initialize chemistry_data.")

    tiny_number = 1e-20
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

    return fc
