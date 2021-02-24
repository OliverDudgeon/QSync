import numpy as np
from qutip import Qobj, expect, jmat, spin_state
from scipy import integrate
from scipy.linalg import expm

from utils import profile


def spin_S_measure(theta, Q):
    # Calculate synchronisation measure from Q representation
    # theta parameter and theta 'axis' of Q should be the same.

    return integrate.simps(np.sin(theta) * Q, theta) - 1 / (2 * np.pi)


def my_spin_coherent_dm(j, theta, phi):
    Qsize = [phi.size, theta.size]
    Sp = np.ones(Qsize, dtype=Qobj) * jmat(j, "+")
    Sm = np.ones(Qsize, dtype=Qobj) * jmat(j, "-")

    v_expm = np.vectorize(lambda v: Qobj(expm(v.full())), otypes=[Qobj])

    psi = v_expm(0.5 * theta * np.exp(1j * phi) * Sm - 0.5 * theta * np.exp(-1j * phi) * Sp) * spin_state(j, j)

    return psi


# @profile
def my_spin_q_func(J, density_op, phi, theta):
    Qsize = [phi.size, theta.size]
    cs = my_spin_coherent_dm(J, theta, phi)

    Q = (2 * J + 1) / (4 * np.pi) * expect(density_op, cs.flatten())
    return Q.reshape(Qsize)
