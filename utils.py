from typing import Union, Tuple, Literal
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib as mpl

from qutip import wigner, qfunc, spin_q_function

xlabel = r"$\rm{Re}(\alpha)$"
ylabel = r"$\rm{Im}(\alpha)$"


def plot_representation(
    rep_name: Literal["wigner", "qfunc", "spin_qfunc"],
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    cmap: Union[None, str] = None,
    ax: Union[None, plt.Axes] = None
) -> Union[Tuple[plt.Axes, mpl.image.AxesImage], None]:
    fig = None
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
    elif rep_name == "spin_qfunc":
        rep, *_ = spin_q_function(state, x, y)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\varphi$")
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda val, _: "{:.0g}$\pi$".format(val / np.pi) if val != 0 else "0")
        )
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, _: "{:.0g}$\pi$".format(val / np.pi) if val != 0 else "0")
        )
        ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))

    if rep is None:
        raise Exception("Called with an invalid rep_name")

    img = ax.imshow(rep, extent=[min(x), max(x), min(y), max(y)], origin="lower", cmap=cmap)
    (fig or plt).colorbar(img, ax=ax)

    return ax, img


plot_wigner = partial(plot_representation, "wigner")
plot_qfunc = partial(plot_representation, "qfunc")
plot_spin_qfunc = partial(plot_representation, "spin_qfunc")
