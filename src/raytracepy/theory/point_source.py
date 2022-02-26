from typing import Union

import numpy as np
from numba import njit

numerical = (int, float, np.ndarray)


def intensity_spherical(r: numerical) -> numerical:
    """
    calculates the intensity of light from a point source in spherical coordinates

    Parameters
    ----------
    r: int, float, np.ndarray
        radius

    Returns
    -------
    output: int, float, np.ndarray
        intensity at radius r

    """
    return 1 / (r**2)


def intensity_cartesian(x: numerical, y: numerical, z: numerical) -> numerical:
    """
    calculates the intensity of light from a point source in cartesian coordinates

    Parameters
    ----------
    x: int, float, np.ndarray
        x position
    y: int, float, np.ndarray
        y position
    z: int, float, np.ndarray
        z position

    Returns
    -------
    output: int, float, np.ndarray
        intensity at position (x,y,z)

    """
    return 1 / (x**2 + y**2 + z**2)


def intensity_on_flat_surface(x: numerical, y: numerical, h: numerical) -> numerical:
    """
    calculates the intensity of light from a point source over a flat surface in cartesian coordinates

    Parameters
    ----------
    x: int, float, np.ndarray
        x position
    y: int, float, np.ndarray
        y position
    h: int, float, np.ndarray
        height of light

    Returns
    -------
    output: int, float, np.ndarray
        intensity at position (x,y,z)

    """
    return 1 / (x**2 + y**2 + h**2) * (h / np.sqrt(x**2 + y**2 + h**2))


def hits_on_flat_surface(n: int, x_dim: Union[int, float] = 10, y_dim: Union[int, float] = 10,
                         h: Union[int, float] = 5, **kwargs) -> np.ndarray:
    """

    Parameters
    ----------
    n: int
        number of points or hits
    x_dim: int, float
        x dimension of plane
    y_dim: int, float
        y dimension of plane
    h: int, float
        height of light

    Returns
    -------

    """
    @njit
    def pdf2D(z: np.ndarray):
        """ probability density function """
        if abs(z[0]) > x_dim or abs(z[1]) > y_dim:
            return float(0)
        return 1 / (z[0] ** 2 + z[1] ** 2 + h ** 2) * (h / np.sqrt(z[0] ** 2 + z[1] ** 2 + h ** 2))

    return metropolis_hastings(pdf2D, size=n, **kwargs)


@njit
def sample(mean, L, n: int = 1):
    dim = mean.size
    rnd = np.random.normal(loc=0, scale=1, size=dim * n).reshape(dim, n)
    rnd = mean.reshape(2, 1) + np.dot(L, rnd)
    return rnd.reshape(dim)


@njit
def calc_L(dim=2, epsilon=1.0001):
    k = epsilon * np.identity(dim)
    return np.linalg.cholesky(k)


@njit
def metropolis_hastings_loop(target_density, xt, samples):
    L = calc_L()
    for i in range(samples.shape[0]):
        xt_candidate = sample(xt, L)
        accept_prob = target_density(xt_candidate) / target_density(xt)
        if np.random.uniform(0, 1) < accept_prob:
            xt = xt_candidate
        samples[i, :] = xt
    return samples


def metropolis_hastings(target_density, init=np.array([0, 0], dtype="float64"), size=50000, burnin_size=1000):
    size += burnin_size
    samples = np.empty((size, 2), dtype="float64")
    samples = metropolis_hastings_loop(target_density, init, samples)
    return samples[burnin_size:, :]
