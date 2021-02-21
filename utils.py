from typing import Tuple, Literal, Optional
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib as mpl
from scipy import integrate

from qutip import wigner, qfunc, spin_q_function

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
        raise Exception("A fig and ax must be both passed or not at all")
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
        raise Exception("A fig and ax must be both passed or not at all")
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


def plot_S_measure(
    Q: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None
):
    if bool(fig) ^ bool(ax):
        raise Exception("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots()

    # Calculate synchronisation measure from Q representation
    S = integrate.simps(np.sin(theta) * Q, theta) - 1 / (2 * np.pi)

    (line,) = ax.plot(phi, S)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$S(\varphi|\hat\rho)$")

    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

    return line
