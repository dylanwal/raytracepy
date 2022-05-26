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
        [np.pi/2, 0.1]
    ]
        , dtype=dtype)
    return data[:, 0], data[:, 1]


@njit
def glass_diffuser_theta_cdf():
    """ Cumulative distribution function for light through ground glass diffuser. """
    data = np.array([  # radian
        [-1.5707963267948966, 0.0],
        [-0.9424777960769379, 0.00034648415919064644],
        [-0.8377580409572782, 0.0013859366367625836],
        [-0.7330382858376184, 0.003118357432715814],
        [-0.6283185307179586, 0.005543746547050338],
        [-0.5235987755982989, 0.008662103979766151],
        [-0.41887902047863923, 0.01490234463762245],
        [-0.3141592653589793, 0.031810643629989026],
        [-0.20943951023931962, 0.08371483192072962],
        [-0.10471975511965992, 0.23292123723345268],
        [0.0, 0.5000392614256586],
        [0.1047197551196597, 0.7668639868190834],
        [0.2094395102393194, 0.9151396181001689],
        [0.3141592653589793, 0.9663794796929772],
        [0.418879020478639, 0.9835428652779825],
        [0.5235987755982987, 0.9902847269913296],
        [0.6283185307179586, 0.9937894646280477],
        [0.7330382858376181, 0.996506573853277],
        [0.837758040957278, 0.9984473661570122],
        [0.9424777960769379, 0.9996118415392533],
        [1.5707963267948966, 1.000000000000000]],
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
