import numpy as np

def RT_1D_solver(I: float | np.float64, kappa: np.ndarray, j: np.ndarray,
                  delta_x: float | np.float64 | np.ndarray, return_last_I=False) -> np.ndarray:
    """
    Solves the 1D radiative transfer equation using the discrete ordinates method.

    Parameters:
    I : float | np.float64
        Initial intensity of the radiation field. 
    kappa : ndarray
        Absorption coefficient array of shape (N,).
    j : ndarray
        Emission coefficient array of shape (N,).
    delta_x : float | ndarray
        Spatial step size array of shape (N,).
    return_last_I : bool, optional
        If True, the intensity array includes the last cell right boundary (I_{N+1}).
    Returns:
    ndarray
        Updated intensity array of shape (N,) or (N+1,) depending on the return_last_I flag.
        The I_i locates at the left boundary of the i-th cell.
    """
    N_size = len(kappa)
    if N_size != len(j):
        raise ValueError("Input arrays kappa and j must have the same length.")
    if N_size == 0:
        raise ValueError("Input arrays must not be empty.")
    # check if delta_x is a single float or an array
    if isinstance(delta_x, float) or isinstance(delta_x, np.float64):
        delta_x = np.full(N_size, delta_x, dtype=np.float64)
    else:
        if N_size != len(delta_x):
            raise ValueError("Input array delta_x must have the same length as kappa and j.")

    I_new = np.zeros(N_size + 1, dtype=np.float64)
    I_new[0] = I  # Set the initial intensity at the left boundary

    for i in range(N_size):
        I_i = I_new[i]
        kappa_i = kappa[i]
        j_i = j[i]
        delta_x_i = delta_x[i]
        tau_i = kappa_i * delta_x_i
        emission_term_i = j_i * delta_x_i
        if tau_i < 10**-10:  # Avoid division by zero for very small kappa
            # Euler's method for small tau
            I_new[i + 1] = I_i + (- tau_i * I_i + emission_term_i)
        else:
            # Use the analytical solution for tau > 10^-10
            # np.expm1(x) = exp(x) - 1, more accurate for small x
            I_new[i + 1] = I_i * np.exp(-tau_i) + (emission_term_i / tau_i) * (-np.expm1(-tau_i))  


    if return_last_I == False:
        return I_new[:-1]
    else:
        return I_new