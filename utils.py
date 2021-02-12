from typing import Union, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
from qutip import wigner
import numpy as np


def plot_wigner(
    state: np.ndarray, x: np.ndarray, y: np.ndarray, *, cmap: Union[None, str] = None, ax: Union[None, plt.Axes] = None
) -> Union[Tuple[plt.Axes, mpl.image.AxesImage], None]:
    wigner_rep = wigner(state, x, y)

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    img = ax.imshow(wigner_rep, extent=[min(x), max(x), min(y), max(y)], origin="lower", cmap=cmap)
    (fig or plt).colorbar(img, ax=ax)

    return ax, img
