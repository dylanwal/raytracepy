from typing import Tuple
import numpy as np
from scipy.integrate import cumtrapz
import plotly.graph_objs as go


def generate_cdf(func, npt: int = 11, x_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cumulative distribution function (cdf).

    Parameters
    ----------
    func: Callable
        probability distribution function
    npt: int
        number of points for cdf
    x_range: Tuple[float, float]
        range for which the func, will be evaluated over.

    Returns
    -------
    x: array[npt]
        x position of cdf
    cdf: array[npt]
        cdf
    """
    x = np.linspace(x_range[0], x_range[1], npt)
    y = func(x)

    y_norm = y / np.trapz(y, x)
    cdf = cumtrapz(y_norm, x)
    cdf = np.insert(cdf, 0, 0)

    # deal with numerical round off errors
    cdf, index_ = np.unique(cdf, return_index=True)
    x = x[index_]
    x[-1] = x_range[1]
    return x, cdf


def plot(func, x, cdf):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=func(x), mode="lines", yaxis="y1"))
    fig.add_trace(go.Scatter(x=x, y=cdf, mode="lines", yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying='y', side='right'))
    fig.write_html("temp.html", auto_open=True)


def print_cdf(x, y):
    for num in x:
        print(str(num) + ",")
    print("")
    print("")
    print("")
    for num in y:
        print(str(num) + ",")


def main():
    """ Use this function to make new cdf for theta functions."""
    from raytracepy.ref_data.ground_glass_diffuser import x, y
    x = x/180 * np.pi
    # x, y = ffunc()

    def func(x_):
        return np.interp(x_, x, y)

    x_cdf, cdf = generate_cdf(func, npt=41, x_range=(0, 1.08))
    plot(func, x_cdf, cdf)
    print_cdf(x_cdf, cdf)


def ffunc():
    x = np.array([0.02829390189315851, 0.08361977424831626, 0.13894564660347403, 0.19427151895863176,
                 0.2495973913137895, 0.3049232636689473, 0.360249136024105, 0.41557500837926276, 0.4709008807344205,
                 0.5262267530895782, 0.581552625444736, 0.6368784977998938, 0.6922043701550515, 0.7475302425102093,
                 0.802856114865367, 0.8581819872205247, 0.9135078595756825, 0.9688337319308402, 1.024159604285998,
                 1.0794854766411557])
    y = np.array([3523, 10377, 16963, 24165, 31286, 38830, 45868, 53229, 62175, 71459, 80690, 91214, 102689,
                  113749, 128264, 142346, 161349, 182497, 204182, 232036])
    return x, y

if __name__ == "__main__":
    main()
