import numpy as np
from qutip import Qobj, expect, jmat, spin_state, steadystate
from qutip.superoperator import lindblad_dissipator
from scipy import integrate
from scipy.linalg import expm

from utils import profile


def spin_S_measure(theta, Q):
    # Calculate synchronisation measure from Q representation
    # theta parameter and theta 'axis' of Q should be the same.

    if len(theta.shape) > 1:
        # To ensure params are passed the right way around
        raise ValueError("theta must be either a row of column vector")

    return integrate.trapz(Q * np.sin(theta), theta) - 1 / (2 * np.pi)


def my_spin_coherent_dm(j, theta, phi):
    Qsize = [1, 1]
    if isinstance(theta, np.ndarray):
        Qsize[1] = theta.size
    if isinstance(phi, np.ndarray):
        Qsize[0] = phi.size

    Sp = np.ones(Qsize, dtype=Qobj) * jmat(j, "+")
    Sm = np.ones(Qsize, dtype=Qobj) * jmat(j, "-")

    v_expm = np.vectorize(lambda v: Qobj(expm(v.full())), otypes=[Qobj])

    psi = v_expm(0.5 * theta * np.exp(1j * phi) * Sm - 0.5 * theta * np.exp(-1j * phi) * Sp) * spin_state(j, j)

    return psi


# @profile
def my_spin_q_func(density_op, theta, phi):
    J = (density_op.shape[0] - 1) / 2

    Qsize = [phi.size, theta.size]
    cs = my_spin_coherent_dm(J, theta, phi)

    Q = (2 * J + 1) / (4 * np.pi) * expect(density_op, cs.flatten())
    return Q.reshape(Qsize)


## Spin-1/2 Systems


def signal_hamiltonian(detuning, signal_strength, J=0.5):
    return detuning * jmat(J, "z") + signal_strength * jmat(J, "y")


def get_disipators(gain_amp, loss_amp, J=0.5):
    return (0.5 * gain_amp * lindblad_dissipator(jmat(J, "+")), 0.5 * loss_amp * lindblad_dissipator(jmat(J, "-")))


# Stationary solutions for the Bloch vector components for the driven case
# For spin 1/2 only
# From spin-1/2 sync paper


def bloch_vector_comps(gain_amp, loss_amp, detuning, signal_strength):
    """From Parra-López & Bergli"""
    m_x = (
        4
        * signal_strength
        * (gain_amp - loss_amp)
        / ((gain_amp + loss_amp) ** 2 + 8 * (signal_strength ** 2 + 2 * detuning * 2))
    )

    m_y = m_x * 4 * detuning / (gain_amp + loss_amp)

    m_z = (
        (loss_amp - gain_amp)
        * ((loss_amp + gain_amp) ** 2 + 16 * detuning ** 2)
        / (loss_amp + gain_amp)
        / ((loss_amp + gain_amp) ** 2 + 8 * (signal_strength ** 2 + 2 * detuning * 2))
    )
    return m_x, m_y, m_z


def calculate_steady_state(*, gain_amp, loss_amp, signal_strength, detuning):
    # Handle cases where a function is passed to calculate strength and detuning
    # This is skipped if plane numbers are passed in
    if callable(signal_strength):
        signal_strength = signal_strength(gain_amp, loss_amp)
    if callable(detuning):
        detuning = detuning(gain_amp, loss_amp)

    gain, loss = get_disipators(gain_amp, loss_amp)

    H = signal_hamiltonian(detuning, signal_strength)
    return steadystate(H, [gain, loss])


def analytic_Q_function(mx, my, mz, phi, theta):
    return (
        1
        + mx * np.cos(phi) * np.sin(theta)
        + my * np.sin(phi) * np.sin(theta)
        + mz * np.ones_like(phi) * np.cos(theta)
    ) / (4 * np.pi)
