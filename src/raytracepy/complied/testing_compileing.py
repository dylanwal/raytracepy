from numba import jit


@jit('f8(f8, f8)')
def calc(a, b):
    return a + b


@jit('f8(f8, f8)')
def calc2(a, b):
    return a * calc(a, b)


if __name__ == "__main__":
    #cc.compile()
    print(calc2(3.2, 4.4))
