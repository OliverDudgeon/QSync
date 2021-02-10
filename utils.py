from typing import Union, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
from qutip import wigner
import numpy as np


def plot_wigner(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    cmap: Union[None, str] = None,
    ax: Union[None, plt.Axes] = None
) -> Union[Tuple[plt.Axes, mpl.image.AxesImage], None]:
    wigner_rep = wigner(state, x, y)

    args = [wigner_rep]
    if cmap is not None:
        args.append(cmap)

    if ax is None:
        plt.imshow(*args)
        plt.colorbar()
        return None
    else:
        _, new_ax = plt.subplots()
        img = new_ax.imshow(*args)
        return new_ax, img
