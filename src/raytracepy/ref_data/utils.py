import numpy as np


def generate_cdf(func, npt: int = 11, x_range=(0, 1)):
    """
    Given a distribution x and y; return n points random chosen from the distribution
    """
    x = np.linspace(x_range[0], x_range[1], npt)
    y = func(x)

    y_norm = y / np.trapz(y, x)
    cdf = np.cumsum(y_norm)  # generate
    cdf = cdf / np.max(cdf)  # normalize it
    index = np.argmin(np.abs(cdf - 1))
    if cdf[index] > 1:
        cdf = cdf[0:index]
        x = x[0:index]

    x = np.insert(x, 0, 0)
    cdf = np.insert(x, 0, 0)
    return x, cdf


def func_(x):
    return np.ones_like(x)  # -np.cos(x*np.pi/2)


def main():
    x, y = generate_cdf(func_)
    import plotly.graph_objs as go

    # x = np.linspace(0, 1, 100)
    # y = func(x)

    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines"))
    fig.write_html("temp.html", auto_open=True)

    for num in x:
        print(str(num) + ",")
    print("")
    print("")
    print("")
    for num in y:
        print(str(num) + ",")


if __name__ == "__main__":
    main()
