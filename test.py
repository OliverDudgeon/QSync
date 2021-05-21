import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from sync_calcs import signal_hamiltonian

# detuning = 5 * 3 * 0.1
# signal_strength = 5 * 2 * 0.1

# H = signal_hamiltonian(detuning, signal_strength)
# t = np.linspace(0, 18, 17)
# result = mesolve(H, ket2dm(basis(2, 0)), t * 0.2)

b = Bloch3d()

v1 = np.array([0.5, 1, 1]) / (3/2)
v2 = np.array([-0.5, 0, 0])

b.add_vectors([v1, v2])

b.show()
