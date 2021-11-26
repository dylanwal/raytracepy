from numba import njit
import numpy as np
dtype = "float64"

from functools import wraps
from time import time


def time_it(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        print(f'func:{func.__name__}  took: {te-ts} s to run.')
        return result
    return wrap


@njit  # ("f8[:](i4)")
def diff_trans_fun(x):
    """
    Data for transmittance through a typical ground glass diffuser. Data obtained from:
    Ching-Cherng Sun, Wei-Ting Chien, Ivan Moreno, Chih-To Hsieh, Mo-Cha Lin, Shu-Li Hsiao, and Xuan-Hao Lee,
    "Calculating model of light transmission efficiency of diffusers attached to a lighting cavity,"
    Opt. Express 18, 6137-6148 (2010) DOI: 10.1364/OE.18.006137.
    https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-18-6-6137&id=196561
    :param x: Angle of incidence in degrees
    :return: Interpolated value of relative transmittance [0, 1]
    """
    x_len = np.asarray([
        -180,
        -90,
        -75.946319465293,
        -66.540292003097,
        -57.1821083002828,
        -47.830322309479,
        -38.0513499612874,
        -28.7089519174554,
        -18.9560571997408,
        -9.60677466146286,
        -0.257213961793098,
        9.05785472529903,
        20.2399426738508,
        28.133328487683,
        37.8798254933872,
        47.189609114036,
        56.5027306713859,
        66.2337897198476,
        75.0675001185077,
        90,
        180
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
        0.575986428492423,
        0.54919392323858,
        0.554681583895587,
        0.538644806990377,
        0.50240115031523,
        0.471544552593739,
        0.430592628027873,
        0.310619688087601,
        0,
        0
    ], dtype=dtype)
    return np.interp(x, x_len, y_len)


@time_it
def main():
    for i in range(100_000):
        get_phi()


if __name__ == '__main__':
    result = diff_trans_fun(10)
    print(result)
    print(type(result))

