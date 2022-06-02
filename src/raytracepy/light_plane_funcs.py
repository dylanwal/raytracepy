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
        np.pi / 2
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
    data = np.array([  # radian
        [0.0, 0.0],
        [0.10471975511965977, 0.07482143130987519],
        [0.20943951023931953, 0.14959556775093982],
        [0.3141592653589793, 0.22413268708375259],
        [0.41887902047863906, 0.2981765110977009],
        [0.5235987755982988, 0.3714353685215445],
        [0.6283185307179586, 0.4434811921712258],
        [0.7330382858376183, 0.5138399337192116],
        [0.8377580409572781, 0.5820076776839055],
        [0.9424777960769379, 0.6474797674802365],
        [1.0471975511965976, 0.7098022347518462],
        [1.1519173063162573, 0.768524369126068],
        [1.2566370614359172, 0.8230957127523906],
        [1.361356816555577, 0.8725785784817236],
        [1.4660765716752366, 0.915621493849396],
        [np.pi/2, 1]
    ]
        , dtype=dtype)
    return data[:, 0], data[:, 1]


@njit
def glass_diffuser_theta_cdf():
    """ Cumulative distribution function for light through ground glass diffuser. """
    data = np.array([  # radian
        [0.0, 0.0],
        [0.05235987755982988, 0.3045337204253261],
        [0.10471975511965977, 0.5612189822214677],
        [0.15707963267948966, 0.7365212311451111],
        [0.20943951023931953, 0.8415117167400985],
        [0.2617993877991494, 0.9019323496412862],
        [0.3141592653589793, 0.935947278984782],
        [0.36651914291880916, 0.9557185963302133],
        [0.41887902047863906, 0.9679309210138324],
        [0.47123889803846897, 0.9757018662748441],
        [0.5235987755982988, 0.9806444656605455],
        [0.5759586531581287, 0.9843311494371655],
        [0.6283185307179586, 0.9876196736293653],
        [0.6806784082777885, 0.9905213126224829],
        [0.7330382858376183, 0.9930360664165181],
        [0.7853981633974483, 0.9951639350114709],
        [0.8377580409572781, 0.9969049184073414],
        [0.890117918517108, 0.9982590166041296],
        [0.9424777960769379, 0.9992262296018355],
        [0.9948376736367678, 0.999806557400459],
        [1.5707963267948966, 1.0000000000000002]],
        dtype=dtype)
    return data[:, 0], data[:, 1]


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
    """
    Turns [0, 1] to [0, 1] re-distributed to account for sphere.
    * only can be re-mapped to [0, np.pi/2]
    """
    return np.arccos(x) / (np.pi / 2)


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
