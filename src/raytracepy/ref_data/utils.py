import numpy as np


def generate_cdf(fun):
    """
    Generate the cumulative distribution function for relative emitted light intensity.
    :param fun: function of relative emitted light intensity vs angle.
    """
    x, cdf = theta_fun(fun)
    for num in x:
        print(str(num) + ",")
    for num in cdf:
        print(str(num) + ",")


def theta_fun(fun):
    x = np.linspace(-90, 90, 180)
    y = fun(x)
    x, cdf = generate_cdf_distribution(x, y)
    return x, cdf


def generate_cdf_distribution(x: np.ndarray, y: np.ndarray):
    """
    Given a distribution x and y; return n points random chosen from the distribution
    :param x: value
    :param y: population
    :return: cdf: cumulative distribution function and same size x (some cropping may be needed)
    """
    y_norm = y / np.trapz(y, x)
    cdf = np.cumsum(y_norm)  # generate
    cdf = cdf / np.max(cdf)  # normalize it
    index = np.argmin(np.abs(cdf - 1))
    if cdf[index] > 0.999:
        cdf = cdf[0:index]
        x = x[0:index]
    return x, cdf