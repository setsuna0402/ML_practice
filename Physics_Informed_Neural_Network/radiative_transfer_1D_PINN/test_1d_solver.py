import numpy as np
from analytic_solve import RT_1D_solver
from matplotlib import pyplot as plt

# random seed 
np.random.seed(42)

N = 128
L = 10.0
delta_x = L / N
# kappa = np.ones(N) * 1.0
# emission = np.ones(N) * 1.1

kappa = np.random.rand(N) * 2.0
emission = np.random.rand(N) * 2.0

I_numerical = RT_1D_solver(I=1.0, kappa=kappa, j=emission, delta_x=delta_x)
'''
I_analytical = np.exp(-kappa * delta_x * np.arange(N)) + emission * (1 - np.exp(-kappa * delta_x * np.arange(N))) / kappa

plt.plot(np.arange(N) * delta_x, I_numerical, label='Numerical Solution')
plt.plot(np.arange(N) * delta_x, I_analytical, label='Analytical Solution', linestyle='dashed')
plt.xlabel('Grid Point')
plt.ylabel('Intensity')
plt.title('1D Radiative Transfer Solver Comparison')
plt.legend()
plt.show()

loss = np.abs((I_numerical - I_analytical)/ I_analytical)
plt.semilogy(np.arange(N) * delta_x, loss)
plt.xlabel('Grid Point')
plt.ylabel('Relative Error')
plt.title('Relative Error between Numerical and Analytical Solutions')
plt.show()
'''

plt.plot(np.arange(N) * delta_x, I_numerical, label='Numerical Solution')
plt.xlabel('Grid Point')
plt.ylabel('Intensity')
plt.title('1D Radiative Transfer Solver Numerical Solution')
plt.legend()
plt.show()
