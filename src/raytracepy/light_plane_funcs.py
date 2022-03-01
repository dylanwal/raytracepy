"""
This python file contains the data related to lights, lens, and planes.

"""

import numpy as np

from . import dtype, njit


@njit  # ("f8[:](f8[:])")
def diff_trans_fun(x):
    """
    Data for transmittance through a typical ground glass diffuser. Data obtained from:
    Ching-Cherng Sun, Wei-Ting Chien, Ivan Moreno, Chih-To Hsieh, Mo-Cha Lin, Shu-Li Hsiao, and Xuan-Hao Lee,
    "Calculating model of light transmission efficiency of diffusers attached to a lighting cavity,"
    Opt. Express 18, 6137-6148 (2010) DOI: 10.1364/OE.18.006137.
    https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-18-6-6137&id=196561
    :param x: Angle of incidence in radians
    :return: Interpolated value of relative transmittance [0, 1]
    """
    x_len = np.asarray([
        -3.14159265358979,
        -1.5707963267949,
        -1.32551332944082,
        -1.1613471806925,
        -0.998016063071913,
        -0.834796606590507,
        -0.664121341653082,
        -0.501065735756336,
        -0.330845611331853,
        -0.167669848339691,
        0,
        0.167669848339691,
        0.330845611331853,
        0.501065735756336,
        0.664121341653082,
        0.834796606590507,
        0.998016063071913,
        1.1613471806925,
        1.32551332944082,
        1.5707963267949,
        3.14159265358979,

    ], dtype=dtype)
    y_len = np.asarray([
        0,
        0,
        0.30869419312023,
        0.427777402942152,
        0.469646101095011,
        0.501189602920031,
        0.537564420418095,
        0.553956818935958,
        0.548245238358588,
        0.575748445968366,
        0.603700575157615,
        0.575748445968366,
        0.548245238358588,
        0.553956818935958,
        0.537564420418095,
        0.501189602920031,
        0.469646101095011,
        0.427777402942152,
        0.30869419312023,
        0,
        0,
    ], dtype=dtype)
    return np.interp(x, x_len, y_len)


@njit
def probability_func_selector(fun_id, x: float = 0):
    """
    probability of transmission or reflection function selector

    Parameters
    ----------
    fun_id: int
        function id
    x: float
        light angle of incidence (radians)

    Returns
    -------
    output: float
        probability of light transmitting or reflecting

    """
    if fun_id == 3:
        """ probability of transmission through diffuser.  """
        return diff_trans_fun(x)
    elif fun_id == 4:
        """ probability of reflection on mirror. """
        return 0.85
    elif fun_id == 5:
        """ probability of reflection on ground. """
        return 0.05
    elif fun_id == 6:
        """ probability of reflection on white pcb. """
        return 0.02
    else:
        """ Error/invalid id; can't throw real error in numba codeblock so cause crazy bad data to signify error"""
        return -1000000


@njit
def uniform_cdf():
    """ Cumulative distribution function for uniform point source. """
    x = np.array([  # radian
        0,
        np.pi/2
    ],
        dtype=dtype)
    cdf = np.array([
        0,
        1
    ],
        dtype=dtype)

    return x, cdf


@njit
def LED_theta_cdf():
    """ cumulative distribution function for LED. """
    x = np.array([  # radian
        -3.141592653589793,
        -2.0943951023931957,
        -1.8849555921538759,
        -1.6755160819145565,
        -1.4660765716752369,
        -1.2566370614359172,
        -1.0471975511965979,
        -0.8377580409572785,
        -0.6283185307179586,
        -0.41887902047863923,
        -0.20943951023931984,
        0.0,
        0.2094395102393194,
        0.4188790204786388,
        0.6283185307179586,
        0.837758040957278,
        1.0471975511965974,
        1.2566370614359172,
        1.4660765716752362,
        1.675516081914556,
        1.8849555921538759,
        2.094395102393195,
        3.141592653589793,
    ], dtype=dtype)
    cdf = np.array([
        0.0,
        0.00032840107445633907,
        0.0018631861015409447,
        0.01315343826570248,
        0.042783578865501576,
        0.08832701490563799,
        0.14452438681527968,
        0.20810218294993083,
        0.2771099172910312,
        0.3496252593586689,
        0.4239598295146397,
        0.4988881327931622,
        0.5737986151332136,
        0.648167590090538,
        0.7208273409030261,
        0.7900772848911926,
        0.8539775475814667,
        0.9105650107163438,
        0.9565437639076648,
        0.9866011575847736,
        0.9981198480706429,
        0.9996725415536005,
        0.9999999999999998,
    ], dtype=dtype)
    return x, cdf


@njit
def glass_diffuser_theta_cdf():
    """ Cumulative distribution function for light through ground glass diffuser. """
    x = np.array([  # radian
        -3.141,
        -0.942,
        -0.7853981633974483,
        -0.6283185307179586,
        -0.47123889803846897,
        -0.3141592653589793,
        -0.15707963267948966,
        0.0,
        0.15707963267948966,
        0.3141592653589793,
        0.47123889803846897,
        0.6283185307179586,
        0.7853981633974483,
        0.942,
        3.141,
    ], dtype=dtype)
    cdf = np.array([
        0.0,
        0.0005,
        0.003,
        0.006,
        0.012,
        0.04,
        0.16,
        0.5,
        0.84,
        0.96,
        0.98,
        0.994,
        0.997,
        0.9995,
        1,
    ], dtype=dtype)
    return x, cdf


@njit  # ("f8[:](f8[:],f8[:],i4)")
def get_theta(x: np.ndarray, cdf: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Given a distribution x and y; return n points random chosen from the distribution

    Parameters
    ----------
    x: np.ndarray
        angle for cdf (radians)
    cdf: np.ndarray
        cumulative distribution of refraction
    n: int
        number of random numbers you want

    Returns
    -------
    output: np.ndarray

    """
    rnd = np.random.random((n,))
    rnd = sphere_correction(rnd)
    return np.interp(rnd, cdf, x)


@njit
def sphere_correction(x: np.ndarray) -> np.ndarray:
    """ Turns [0, 1] to [0, 1] re-distributed to account for sphere."""
    return np.arccos(x)/(np.pi/2)


@njit  # ("f8[:](f8,f8[:],i4)")
def theta_func_selector(fun_id: int, n: int = 1):
    """
    Theta function selector.

    Parameters
    ----------
    fun_id: int
        function id
    n: int
        number of theta values desired

    Returns
    -------
    output: np.ndarray
        values of theta distributed appropriately in spherical coordinates

    """
    if fun_id == 0:
        """ theta function for uniform. """
        x_cdf, cdf = uniform_cdf()
        return get_theta(x_cdf, cdf, n)
    elif fun_id == 1:
        """ theta function for LED. """
        x_cdf, cdf = LED_theta_cdf()
        return get_theta(x_cdf, cdf, n)
    elif fun_id == 2:
        """ theta function for light scattered through diffuser. """
        x_cdf, cdf = glass_diffuser_theta_cdf()
        return get_theta(x_cdf, cdf, n)
    else:
        """ Error/invalid id; can't throw real error in numba codeblock so cause crazy bad data to signify error"""
        return np.ones(n, dtype=dtype) * -1000000
