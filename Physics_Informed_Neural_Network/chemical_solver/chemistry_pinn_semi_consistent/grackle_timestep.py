import numpy as np

MAX_CHANGE = 0.1


def _get_chemistry_data(fc):
    """
    Get the chemistry_data object attached to a Grackle FluidContainer.

    Different gracklepy versions may expose it under slightly different
    attribute names, so keep the lookup here rather than in user code.
    """
    for name in ("chemistry_data", "_chemistry_data", "chemistry"):
        if hasattr(fc, name):
            return getattr(fc, name)

    raise AttributeError(
        "Cannot find chemistry_data on the FluidContainer. "
        "Expected one of: fc.chemistry_data, fc._chemistry_data, fc.chemistry."
    )


def _interpolate_grackle_rate(chem, rate_table, temperature):
    """
    Interpolate a Grackle reaction-rate table linearly in ln(T).

    This follows the same interpolation logic as the supplied C++
    ComputeTimeStepHII implementation.
    """
    logT0 = np.log(chem.TemperatureStart)
    logT1 = np.log(chem.TemperatureEnd)
    nbins = int(chem.NumberOfTemperatureBins)

    dlogT = (logT1 - logT0) / float(nbins - 1)

    T = np.asarray(temperature)
    logT = np.clip(np.log(T), logT0, logT1)

    # Upper point of the interpolation interval, same convention as
    # tem_id in the supplied C++ code.
    tem_id = ((logT - logT0) / dlogT).astype(np.int64) + 1
    tem_id = np.clip(tem_id, 1, nbins - 1)

    logT_left = logT0 + (tem_id - 1) * dlogT
    frac = (logT - logT_left) / dlogT

    rate_table = np.asarray(rate_table)

    return (
        rate_table[tem_id - 1]
        + frac * (rate_table[tem_id] - rate_table[tem_id - 1])
    )



def get_primordial_reaction_rates(fc):
    """Return provisional primordial H/He reaction coefficients k1..k6.

    Mapping used in this prototype:
      k1: HI  + e -> HII  + 2e
      k2: HII + e -> HI   + photon
      k3: HeI + e -> HeII + 2e
      k4: HeII+ e -> HeI  + photon
      k5: HeII+ e -> HeIII+ 2e
      k6: HeIII+e -> HeII + photon

    The mapping should be checked against the installed Grackle version before
    treating the He residuals as final. Coefficients are interpolated from the
    Grackle tables at the current *reference* temperature.
    """
    chem = _get_chemistry_data(fc)
    fc.calculate_temperature()
    T = np.asarray(fc["temperature"])
    rates = [
        _interpolate_grackle_rate(chem, getattr(chem, f"k{i}"), T)
        for i in range(1, 7)
    ]
    return tuple(np.asarray(rate) for rate in rates)


def get_H_reaction_rates(fc):
    """Backward-compatible helper returning only k1 and k2."""
    k1, k2, *_ = get_primordial_reaction_rates(fc)
    return k1, k2

def get_HII_timestep(fc, max_change=MAX_CHANGE):
    """
    Return the adaptive chemistry/RT timestep in Grackle code-time units.

    Only the FluidContainer is required:

        dt = get_HII_timestep(fc)

    The timestep limits the fractional HII change to `max_change`:

        dt = max_change * |rho_HII / (d rho_HII / dt)|

    with

        d rho_HII / dt
          = a^-3 [k1 rho_HI rho_e - k2 rho_HII rho_e]
            + Gamma_HI rho_HI

    matching the supplied C++ implementation.

    Notes
    -----
    * k1 and k2 are taken from the chemistry_data rate tables attached
      to the FluidContainer and interpolated in ln(T).
    * RT_HI_ionization_rate is assumed to already be in 1/code_time.
    * The returned value is in the same code-time units used by
      fc.solve_chemistry(dt).
    """
    chem = _get_chemistry_data(fc)

    # Refresh temperature from the current internal energy/species state.
    fc.calculate_temperature()

    T = np.asarray(fc["temperature"])
    rho_HI = np.asarray(fc["HI_density"])
    rho_HII = np.asarray(fc["HII_density"])
    rho_e = np.asarray(fc["e_density"])
    Gamma_HI = np.asarray(fc["RT_HI_ionization_rate"])

    k1 = _interpolate_grackle_rate(chem, chem.k1, T)
    k2 = _interpolate_grackle_rate(chem, chem.k2, T)

    # Preserve the comoving-density convention used in the supplied C++ code.
    a = chem.a_value * chem.a_units
    inv_a3 = 1 / (a ** 3.0)

    dHII_dt = (
        inv_a3 * (k1 * rho_HI * rho_e - k2 * rho_HII * rho_e)
        + rho_HI * Gamma_HI
    )
    local_dt = max_change * np.abs(rho_HII / dHII_dt)
    return float(np.min(local_dt))
    '''
    # If HII is exactly zero, the literal fractional-change criterion gives
    # dt = 0. Keep that behaviour explicit instead of hiding it with a floor.
    with np.errstate(divide="ignore", invalid="ignore"):
        local_dt = max_change * np.abs(rho_HII / dHII_dt)

    # Cells with exactly zero net change should not constrain the timestep.
    local_dt = np.where(np.abs(dHII_dt) == 0.0, np.inf, local_dt)

    return float(np.min(local_dt))
    '''
