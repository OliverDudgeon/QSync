from functools import partial
from typing import Literal, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
from qutip import qfunc, wigner


coherent_xlabel = r"$\rm{Re}(\alpha)$"
coherent_ylabel = r"$\rm{Im}(\alpha)$"


def pi_formatter(val, _):
    return "{:.0g}$\pi$".format(val / np.pi) if val != 0 else "0"


def angle_xaxis(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))


def angle_yaxis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))


def plot_representation(
    rep_name: Literal["wigner", "qfunc", "spin_qfunc"], state, x, y, *, cmap=None, ax=None, fig=None, **kwargs
) -> Tuple[np.ndarray, Tuple[plt.Axes, mpl.image.AxesImage]]:
    if bool(fig) ^ bool(ax):
        raise TypeError("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots()

    rep = None
    if rep_name == "wigner":
        rep = wigner(state, x, y)
        ax.set_xlabel(coherent_xlabel)
        ax.set_ylabel(coherent_ylabel)
    elif rep_name == "qfunc":
        rep = qfunc(state, x, y)
        ax.set_xlabel(coherent_xlabel)
        ax.set_ylabel(coherent_ylabel)

    if rep is None:
        raise Exception("Called with an invalid rep_name")

    img = ax.imshow(rep, extent=[min(x), max(x), min(y), max(y)], origin="lower", aspect="auto", cmap=cmap, **kwargs)
    (fig or plt).colorbar(img, ax=ax)

    return rep, (ax, img)


plot_wigner = partial(plot_representation, "wigner")
plot_qfunc = partial(plot_representation, "qfunc")


def plot_spin_qfunc(Q, theta, phi, *, cmap=None, ax=None, fig=None, **kwargs):
    if bool(fig) ^ bool(ax):
        raise TypeError("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\theta$")
    angle_xaxis(ax)
    angle_yaxis(ax)

    THETA, PHI = np.meshgrid(theta, phi)  # need a meshgrid for pcolor-type plots

    img = ax.pcolormesh(THETA, PHI, Q, cmap=cmap, shading="nearest", **kwargs)
    fig.colorbar(img, ax=ax)

    return ax, img


def plot_S_measure(S, phi, *, ax=None, fig=None, **kwargs):
    if bool(fig) ^ bool(ax):
        raise TypeError("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots()

    (line,) = ax.plot(phi, S, **kwargs)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$S(\varphi|\hat\rho)$")

    angle_xaxis(ax)

    return line, (ax, fig)
