import re

import numpy as np
from numba import njit


@njit("f8[:](f8[:],f8[:],f8)", cache=True)
def sample_a_distribution(x: np.ndarray, cdf: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Given a distribution x and y; return n points random chosen from the distribution
    :param x: value
    :param cdf: population
    :param n: number of random numbers you want
    :return: np.array of random numbers
    """
    rnd = np.random.random((n,))  # generate random numbers between [0,1]
    return np.interp(rnd, cdf, x)


class CDFFuncs:
    def __init__(self, x_cdf, y_cdf):
        self.x_cdf = x_cdf
        self.y_cdf = y_cdf

    def __call__(self):
        @njit(cache=True)
        def func(n):
            """
            :param n: number of random numbers you want
            :return
            """
            return sample_a_distribution(self.x_cdf, self.y_cdf, n) / 360 * (2 * np.pi)

        return func


class InterpFuncs:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self):
        @njit(cache=True)
        def func(new_x):
            return np.interp(new_x, self.x, self.y)

        return func


class ConstantFuncs:
    def __init__(self, const):
        self.const = const

    def __call__(self):
        @njit(cache=True)
        def func(x):
            return np.ones_like(x, dtype="float64") * self.const
        return func


class ThetaFactory:
    _funcs = {}

    def __call__(self, func_name: str):
        if func_name in self._funcs:
            return self._funcs[func_name]

        if func_name == "led":
            from .raw_data.led import led_cdf_x, led_cdf_y
            func = CDFFuncs(led_cdf_x, led_cdf_y)()

        else:
            raise ValueError(f"Invalid theta function. Invalid: {func_name}. See theta_funcs.py for list.")

        self._funcs[func_name] = func
        return func


theta_factory = ThetaFactory()


class PlaneFuncFactory:
    _funcs = {}

    def __call__(self, func_name: str):
        if func_name in self._funcs:
            return self._funcs[func_name]

        if func_name == "ground_glass_diffuser":
            from .raw_data.ground_glass_diffuser import cdf_x, cdf_y
            func = CDFFuncs(cdf_x, cdf_y)()
        elif func_name == "ground_glass_transmit":
            from .raw_data.ground_glass_transmit import x, y
            func = InterpFuncs(x, y)()
        elif bool(re.match("(^mirror)([0-9]{0,3}$)", func_name)):
            constant = re.split("(^mirror)([0-9]{0,3}$)", func_name)[2]
            func = ConstantFuncs(constant)()
        else:
            raise ValueError(f"Invalid theta function. Invalid: {func_name}. See theta_funcs.py for list.")

        self._funcs[func_name] = func
        return func


plane_func_factory = PlaneFuncFactory()
