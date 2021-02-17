from typing import Union, Tuple, Literal
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from qutip import wigner, qfunc


def plot_representation(
    rep_name: Literal["wigner", "qfunc"],
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    cmap: Union[None, str] = None,
    ax: Union[None, plt.Axes] = None
) -> Union[Tuple[plt.Axes, mpl.image.AxesImage], None]:
    rep = None
    if rep_name == "wigner":
        rep = wigner(state, x, y)
    elif rep_name == "qfunc":
        rep = qfunc(state, x, y)

    if rep is None:
        raise Exception("Called with an invalid rep_name")

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    img = ax.imshow(rep, extent=[min(x), max(x), min(y), max(y)], origin="lower", cmap=cmap)
    (fig or plt).colorbar(img, ax=ax)

    ax.set_xlabel(r"$\rm{Re}(\alpha)$")
    ax.set_ylabel(r"$\rm{Im}(\alpha)$")

    return ax, img


plot_wigner = partial(plot_representation, "wigner")
plot_qfunc = partial(plot_representation, "qfunc")
