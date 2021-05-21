from functools import partial
from typing import Literal, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator, FixedFormatter
from qutip import qfunc, wigner

from sync_calcs import spin_husimi_qfunc, spin_S_measure

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}
plt.rcParams.update(tex_fonts)

DOCUMENT_WIDTH = 483.69687  # pt


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


coherent_xlabel = r"$\rm{Re}(\alpha)$"
coherent_ylabel = r"$\rm{Im}(\alpha)$"


def pi_formatter(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * np.abs(value) / np.pi))
    sgn = "-" if np.sign(value) == -1 else ""
    if N == 0:
        return "0"
    elif N == 1:
        return r"${0}\pi/2$".format(sgn)
    elif N == 2:
        return r"${0}\pi$".format(sgn)
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)


def angle_xaxis(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 2))


def angle_yaxis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.yaxis.set_major_locator(MultipleLocator(base=np.pi / 2))


def plot_representation(
    rep_name: Literal["wigner", "qfunc", "spin_qfunc"], state, x, y, *, cmap=None, ax=None, fig=None, **kwargs
) -> Tuple[np.ndarray, Tuple[plt.Axes, mpl.image.AxesImage]]:
    if bool(fig) ^ bool(ax):
        raise TypeError("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots(figsize=set_size(DOCUMENT_WIDTH))

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


def plot_spin_qfunc(Q, theta, phi, *, cbar_step=0.05, cmap=None, ax=None, fig=None, **kwargs):
    if bool(fig) ^ bool(ax):
        raise TypeError("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots(figsize=set_size(DOCUMENT_WIDTH))

    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\theta$")
    angle_xaxis(ax)
    angle_yaxis(ax)

    THETA, PHI = np.meshgrid(theta, phi)  # need a meshgrid for pcolor-type plots

    # img = ax.pcolormesh(THETA, PHI, Q, cmap=cmap, shading="nearest", **kwargs)
    img = ax.contourf(THETA, PHI, Q, 100)
    for c in img.collections:
        c.set_edgecolor("face")
    cbar = fig.colorbar(img, ax=ax)

    cbar.set_label(r"$Q(\theta,\varphi\,|\,\hat\rho)$", rotation=270, labelpad=20)

    cbar.ax.yaxis.set_major_locator(MultipleLocator(base=cbar_step))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))

    return img, ax


def plot_S_measure(S, phi, *, yax_step=0.05, ax=None, fig=None, **kwargs):
    if bool(fig) ^ bool(ax):
        raise TypeError("A fig and ax must be both passed or not at all")
    if ax is None:
        fig, ax = plt.subplots(figsize=set_size(DOCUMENT_WIDTH))

    (line,) = ax.plot(phi, S, lw=2, **kwargs)
    ax.set_xlim(np.min(phi), np.max(phi))
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$S(\varphi\,|\,\hat\rho)$")

    ax.yaxis.set_major_locator(MultipleLocator(base=yax_step))

    angle_xaxis(ax)

    return line, (ax, fig)


def plot_Q_and_S(theta, phi, Q, S, *, cmap=None, axs=None, fig=None, **kwargs):
    if bool(fig) ^ bool(axs):
        raise TypeError("A fig and ax must be both passed or not at all")
    if axs is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=set_size(DOCUMENT_WIDTH, 2, [1, 2]))

    img, *_ = plot_spin_qfunc(Q, phi, theta, cmap=cmap, ax=ax1, fig=fig)
    line, *_ = plot_S_measure(S, phi, ax=ax2, fig=fig)

    return img, line, fig, (ax1, ax2)


def calc_and_plot_Q_and_S(state, n=50, theta=None, phi=None, method="qutip"):
    if theta is None:
        theta = np.linspace(0, np.pi, n)

    if phi is None:
        phi = np.linspace(-np.pi, np.pi, 2 * n).reshape(-1, 1)  # 1D vector -> 2D column vector

    Q = spin_husimi_qfunc(state, theta, phi, method=method)
    S = spin_S_measure(theta, Q)

    *_, fig, axs = plot_Q_and_S(theta, phi, Q.T, S)

    return fig, axs
