from numba.pycc import CC
from numba import njit

cc = CC('math_custom')
cc.verbose = True


@njit
@cc.export('calc', 'f8(f8, f8)')
def calc(a, b):
    "calc 1 help string"
    return a + b


@njit
@cc.export('calc2', 'f8(f8, f8)')
def calc2(a, b):
    return a * calc(a, b)


if __name__ == "__main__":
    cc.compile()