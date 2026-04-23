'''
Purpose:  
The PDE we are solving is the underdamped harmonic oscillator, given by the equation:
m * d^2u/dt^2 + c * du/dt + k * u = 0
where m is the mass, c is the damping coefficient, k is the spring constant, and u is the displacement. 
For damped harmonic oscillator, coefficients need to satisfy the condition c^2 < 4*m*k for underdamped case, c^2 = 4*m*k for critically damped case, and c^2 > 4*m*k for overdamped case.
In this case, the general solution to the underdamped harmonic oscillator can be expressed as:
u(t) = A * exp(-c*t/(2*m)) * cos(w*t + phi)
where A is the amplitude, w is the damped natural frequency given by w = sqrt(4*m*k - c^2) / (2*m),
and phi is the phase angle. 
Here, we choose u(t=0) = 1 and du/dt(t=0) = 0 as the initial conditions, which leads to 
A = 1/cos(phi) and phi = arctan(-delta/w) where delta = c/(2*m).
We define w_o^2 = k/m as the natural frequency squared, and delta = c/(2*m) as the damping ratio.
So, w = sqrt(w_o^2 - delta^2) is the oscillation frequency.

We use Velocity Verlet method to solve the equation of motion.


Author: Dr. Ka Hou Leong
Date: 22/4/2026
'''

import numpy as np
import matplotlib.pyplot as plt
# This is for the progress bar.
from tqdm.auto import tqdm
import time


delta = 2.0  # Damping ratio
w_o = 20.0  # Natural frequency
dt = 0.0001  # Time step for numerical solution
u_int = 1.0  # Initial displacement
v_int = 0.0  # Initial velocity

duration_analytical = 5.0  # Duration of the time series for the analytical solution
n_points_analytical = 2500  # Number of data points for the analytical solution
n_points_numerical = int(duration_analytical / dt)  # Number of data points for the numerical solution


def oscillator_solution(delta: float, w_o: float, t: np.ndarray) -> np.ndarray:
    # The analytical solution for the underdamped harmonic oscillator
    # we use torch functions to ensure compatibility with PyTorch tensors

    # check if the system is underdamped
    assert delta < w_o, "The system is not underdamped. Please ensure that delta < w_o."

    w = np.sqrt(w_o**2 - delta**2)  # Damped natural frequency
    phi = np.arctan(-delta / w)  # Phase angle
    amplitude = 1 / np.cos(phi)  # Amplitude based on initial conditions
    exp_term = np.exp(-delta * t)  # Exponential decay term
    cos_term = np.cos(w * t + phi)  # Oscillatory term
    y = amplitude * exp_term * cos_term  # displacement as a function of time
    return y

def velocity_verlet_forward_wrong(displacement: float, velocity: float, delta: float, w_o: float, dt: float, n_points: int):
    # Update one step forward using the Velocity Verlet method
    # The acceleration is given by the equation of motion: a = -2 * delta * v - w_o^2 * u
    # where u is the displacement and v is the velocity.
    # This function is incorrect because the updated acceleration is calculated based on the new position and half-step velocity.
    acc_current = -2 * delta * velocity - w_o**2 * displacement  # Current acceleration (t_i)
    velocity_half = velocity + 0.5 * acc_current * dt  # Update velocity to the half step (t_i+0.5)
    displacement_new = displacement + velocity_half * dt  # Update displacement to  the new position (t_i+1)
    # This is not correct, we need to use the velocity at t_i+1 to calculate the acceleration at t_i+1.
    acc_new = -2 * delta * velocity_half - w_o**2 * displacement_new  # Calculate new acceleration based on the new position and half-step velocity
    velocity_new = velocity + 0.5 * acc_new * dt  # Update velocity to the new position (t_i+1)

    return displacement_new, velocity_new



def velocity_verlet_forward(displacement: float, velocity: float, delta: float, w_o: float, dt: float, n_points: int):
    # Update one step forward using the Velocity Verlet method
    # The acceleration is given by the equation of motion: a = -2 * delta * v - w_o^2 * u
    # where u is the displacement and v is the velocity.
    # velocity is updated by the exact method, which is more accurate than the half-step method.
    acc_current = -2 * delta * velocity - w_o**2 * displacement  # Current acceleration (t_i)
    velocity_half = velocity + 0.5 * acc_current * dt  # Update velocity to the half step (t_i+0.5)
    displacement_new = displacement + velocity_half * dt  # Update displacement to  the new position (t_i+1)
    velocity_new = (velocity_half - 0.5 * w_o**2 * displacement_new * dt) / (1 + delta * dt)  # Update velocity to the new position (t_i+1) using the exact method

    return displacement_new, velocity_new

t_analytical = np.linspace(0, duration_analytical, n_points_analytical)  # Time points for analytical solution
u_analytical = oscillator_solution(delta, w_o, t_analytical)  # Analytical solution at the specified time points
t_sim = np.arange(0, n_points_numerical) * dt  # Time points for numerical solution

u_sim = [u_int]  # List to store the numerical solution for displacement
v_sim = [v_int]  # List to store the numerical solution for velocity

# Time-stepping loop for the numerical solution using Velocity Verlet method
for i in range(len(t_sim)-1):
    u_new, v_new = velocity_verlet_forward(u_sim[-1], v_sim[-1], delta, w_o, dt, n_points_numerical)
    u_sim.append(u_new)  # Append the new displacement to the list
    v_sim.append(v_new)  # Append the new velocity to the list

u_sim = np.array(u_sim)  # Convert the list of displacements to a NumPy array
v_sim = np.array(v_sim)  # Convert the list of velocities to a NumPy array

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(t_analytical, u_analytical, label='Analytical Solution', color='blue')
plt.plot(t_sim, u_sim, label=f'Numerical Solution, dt={dt}', color='red', linestyle='--')
plt.title('Damped Harmonic Oscillator: Analytical vs Numerical Solution')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (u)')
plt.legend()
plt.grid()
# plt.show()
plt.savefig('damped_harmonic_oscillator_numerical_solution.png', dpi=300)  # Save the plot as a PNG file with high resolution
plt.close()  # Close the plot to free up memory

# plotting the absolute error between the analytical and numerical solutions
u_analytical_plot = oscillator_solution(delta, w_o, t_sim)  # Analytical solution at the same time points as the numerical solution
plt.figure(figsize=(12, 6))
plt.semilogy(t_sim, np.abs(u_analytical_plot - u_sim), label='Absolute Error', color='green')
plt.title('Error in Numerical Solution')
plt.xlabel('Time (s)')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid()
# plt.show()
plt.savefig('damped_harmonic_oscillator_numerical_solution_error.png', dpi=300)  # Save the plot as a PNG file with high resolution
plt.close()  # Close the plot to free up memory