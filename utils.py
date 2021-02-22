from functools import partial
from typing import Literal, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
from qutip import Qobj, expect, jmat, qfunc, spin_coherent, spin_q_function, wigner, spin_state
from scipy import integrate

xlabel = r"$\rm{Re}(\alpha)$"
ylabel = r"$\rm{Im}(\alpha)$"


def pi_formatter(val, _):
    return "{:.0g}$\pi$".format(val / np.pi) if val != 0 else "0"


def plot_representation(
    rep_name: Literal["wigner", "qfunc", "spin_qfunc"],
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    cmap: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None
) -> Tuple[np.ndarray, Tuple[plt.Axes, mpl.image.AxesImage]]:
    if bool(fig) ^ bool(ax):
        raise TypeError("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots()

    rep = None
    if rep_name == "wigner":
        rep = wigner(state, x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    elif rep_name == "qfunc":
        rep = qfunc(state, x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if rep is None:
        raise Exception("Called with an invalid rep_name")

    img = ax.imshow(rep, extent=[min(x), max(x), min(y), max(y)], origin="lower", aspect="auto", cmap=cmap)
    (fig or plt).colorbar(img, ax=ax)

    return rep, (ax, img)


plot_wigner = partial(plot_representation, "wigner")
plot_qfunc = partial(plot_representation, "qfunc")


def my_spin_coherent_dm(j, theta, phi):
    Qsize = [phi.size, theta.size]
    Sp = np.ones(Qsize, dtype=Qobj) * jmat(j, "+")
    Sm = np.ones(Qsize, dtype=Qobj) * jmat(j, "-")

    expm = np.vectorize(lambda qobj: qobj.expm(), otypes=[Qobj])

    psi = expm(0.5 * theta * np.exp(1j * phi) * Sm - 0.5 * theta * np.exp(-1j * phi) * Sp) * spin_state(j, j)

    return psi


def my_spin_q_func(density_op, phi, theta):
    Qsize = [phi.size, theta.size]
    cs = my_spin_coherent_dm(0.5, theta, phi)

    Q = expect(density_op, cs.flatten()) / (2 * np.pi)
    return Q.reshape(Qsize)


def plot_spin_qfunc(
    state: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    cmap: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None
):
    if bool(fig) ^ bool(ax):
        raise TypeError("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots()

    Q, THETA, PHI = spin_q_function(state, theta, phi)

    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\theta$")
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

    ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))

    img = ax.pcolormesh(PHI, THETA, Q, cmap=cmap, shading="nearest")
    fig.colorbar(img, ax=ax)

    return Q, THETA, PHI, (ax, img)


def spin_S_measure(theta, Q):
    # Calculate synchronisation measure from Q representation
    # theta parameter and theta 'axis' of Q should be the same.
    return integrate.simps(np.sin(theta) * Q, theta) - 1 / (2 * np.pi)


def plot_S_measure(
    Q: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None
):
    if bool(fig) ^ bool(ax):
        raise TypeError("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots()

    S = spin_S_measure(theta, Q)

    (line,) = ax.plot(phi, S)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$S(\varphi|\hat\rho)$")

    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

    return line
