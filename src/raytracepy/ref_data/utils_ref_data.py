from typing import Tuple

import numpy as np
from scipy.integrate import cumtrapz


def generate_cdf(func, npts: int = 11, x_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cumulative distribution function (cdf).

    Parameters
    ----------
    func: Callable
        probability distribution function
    npts: int
        number of points for cdf
    x_range: Tuple[float, float]
        range for which the func, will be evaluated over.

    Returns
    -------
    x: array[npt]
        x position of cdf
    cdf: array[npt]
        cdf
    """
    x = np.linspace(x_range[0], x_range[1], npts)
    y = func(x)

    y_norm = y / np.trapz(y, x)
    cdf = cumtrapz(y_norm, x)
    cdf = np.insert(cdf, 0, 0)

    # deal with numerical round off errors
    cdf, index_ = np.unique(cdf, return_index=True)
    x = x[index_]
    x[-1] = x_range[1]
    return x, cdf


def print_cdf(x, y):
    for x_, y_ in zip(x, y):
        print(f"[{x_}, {y_}],")
