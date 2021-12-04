import numpy as np
import plotly.graph_objs as go
from raytracepy.utils.analysis_func import sphere_distribution, rdf, adf, hits_along_line

from numba import njit, config

config.DISABLE_JIT = False


def main():
    h = 1

    def _pdf(_x):
        return 1 / (_x ** 2 + h ** 2)

    n = 5_000_000
    theta = np.random.uniform(low=0, high=2*np.pi, size=(n,))
    phi = np.arccos(2*np.random.uniform(low=0, high=0.5, size=(n,)) - 1) - np.pi/2

    x = np.cos(theta) * np.cos(phi)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(phi)

    # fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers"))
    # fig.write_html("temp.html", auto_open=True)

    x_, hist = sphere_distribution(xyz=np.column_stack((x,y,z)))
    fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    fig.write_html("temp2.html", auto_open=True)

    x_, hist = adf(xy=np.column_stack((x,y)))
    fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    fig.write_html("temp3.html", auto_open=True)

    x_, hist = adf(xy=np.column_stack((y,z)))
    fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    fig.write_html("temp3.html", auto_open=True)

    t = h/z
    hit_x = x*t
    hit_y = y*t

    mask = np.abs(hit_x) < 10
    hit_x = hit_x[mask]
    hit_y = hit_y[mask]
    mask = np.abs(hit_y) < 10
    hit_x = hit_x[mask]
    hit_y = hit_y[mask]

    # traces = np.zeros((hit_y.size, 2, 3))
    # for i in range(hit_x.size):
    #     if np.abs(hit_x[i]) > 10:
    #         continue
    #     if np.abs(hit_y[i]) > 10:
    #         continue
    #     traces[i, 1, :] = hit_x[i], hit_y[i], h
    #
    # fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers"))
    # fig.add_trace(go.Scatter3d(x=hit_x, y=hit_y, z=np.ones_like(hit_x)*5, mode="markers"))
    # for ray in traces:
    #     fig.add_trace(go.Scatter3d(x=ray[:, 0], y=ray[:, 1], z=ray[:, 2], mode="lines"))
    # fig.write_html("temp.html", auto_open=True)

    # fig = go.Figure(go.Histogram2d(x=hit_x, y=hit_y, nbinsx=20, nbinsy=20))
    # fig.write_html("temp.html", auto_open=True)

    # x_, hist = adf(np.column_stack((hit_x, hit_y)), bins=40, normalize=True)
    # fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    # fig.write_html("temp2.html", auto_open=True)

    # x_, hist = hits_along_line(np.column_stack((hit_x, hit_y)), bins=40, normalize=True, line_angle=np.pi / 4)
    # fig.add_trace(go.Scatter(x=x_, y=hist, mode="lines"))
    # x_, hist = rdf(np.column_stack((hit_x, hit_y)), bins=20, normalize=True)
    # print(",".join([str(i) for i in x_]))
    # print(",".join([str(i) for i in hist]))
    # fig.add_trace(go.Scatter(x=x_, y=hist, mode="lines"))
    # fig.add_trace(go.Scatter(x=np.linspace(0, 10, 50), y=_pdf(np.linspace(0, 10, 50)) / (1 / h ** 2), mode="lines"))
    # fig.write_html("temp3.html", auto_open=True)


dtype = "float64"


@njit
def pdf2D(z):
    if np.abs(z[0]) > 10 or np.abs(z[1]) > 10:
        return float(0)
    return 1 / (z[0]**2 + z[1]**2 + 5**2)


@njit
def sample(mean, L, n: int = 1):
    dim = mean.size
    rnd = np.random.normal(loc=0, scale=1, size=dim * n).reshape(dim, n)
    rnd = mean.reshape(2, 1) + np.dot(L, rnd)
    return rnd.reshape(dim)


@njit
def calc_L(dim=2, epsilon=1.0001):
    k = epsilon * np.identity(dim)
    return np.linalg.cholesky(k)


@njit
def metropolis_hastings_loop(target_density, xt, samples):
    L = calc_L()
    for i in range(samples.shape[0]):
        xt_candidate = sample(xt, L)
        accept_prob = target_density(xt_candidate) / target_density(xt)
        if np.random.uniform(0, 1) < accept_prob:
            xt = xt_candidate
        samples[i, :] = xt
    return samples


def metropolis_hastings(target_density, init=np.array([0, 0], dtype=dtype), size=50000, burnin_size=1000):
    size += burnin_size
    samples = np.empty((size, 2), dtype=dtype)
    samples = metropolis_hastings_loop(target_density, init, samples)
    return samples[burnin_size:, :]


def main_theory():
    """ Using the answer check analysis tools. """
    h = 5

    def _pdf(_x):
        return 1 / (_x ** 2 + h ** 2)

    n = 5_000_000
    samples = metropolis_hastings(pdf2D, size=n)
    hit_x = samples[:, 0]
    hit_y = samples[:, 1]
    print(hit_x.size)

    fig = go.Figure(go.Histogram2d(x=hit_x, y=hit_y, nbinsx=20, nbinsy=20))
    fig.write_html("temp.html", auto_open=True)

    x_, hist = adf(np.column_stack((hit_x, hit_y)), bins=20, normalize=True)
    fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    fig.write_html("temp2.html", auto_open=True)

    #
    x_, hist = hits_along_line(np.column_stack((hit_x, hit_y)), bins=20, normalize=True, line_angle=np.pi/8)
    fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    x_, hist = rdf(np.column_stack((hit_x, hit_y)), bins=20, normalize=True)
    fig.add_trace(go.Scatter(x=x_, y=hist, mode="lines"))
    fig.add_trace(go.Scatter(x=np.linspace(0, 10, 50), y=_pdf(np.linspace(0, 10, 50)) / (1 / h ** 2), mode="lines"))
    fig.write_html("temp3.html", auto_open=True)

    print("hi")


@njit
def normalise(vector: np.ndarray) -> np.ndarray:
    """
    Object is guaranteed to be a unit quaternion after calling this
    operation UNLESS the object is equivalent to Quaternion(0)
    """
    n = np.sqrt(np.dot(vector, vector))
    if n > 0:
        return vector / n
    else:
        return vector


def main_theory2():
    """ Using the answer check analysis tools. """

    n = 1000_0000
    samples = metropolis_hastings(pdf2D, size=n)
    hit_x = samples[:, 0]
    hit_y = samples[:, 1]
    r_sq = hit_y**2 + hit_x**2
    mask = r_sq < 100
    hit_x = hit_x[mask]
    hit_y = hit_y[mask]

    print(hit_x.size)

    fig = go.Figure(go.Histogram2d(x=hit_x, y=hit_y, nbinsx=30, nbinsy=30))
    layout = {
        "autosize": False,
        "width": 900,
        "height": 790,
        "showlegend": False,
        "font": dict(family="Arial", size=18, color="black"),
        "plot_bgcolor": "white"
    }
    fig.update_layout(layout)
    fig.update_xaxes({"title": "<b>X<b>"})
    fig.update_yaxes({"title": "<b>Y<b>"})
    fig.write_html("temp.html", auto_open=True)


    m = 800
    mm = int(hit_x.size/m)
    m = int(hit_x.size/mm)
    xyz = np.empty((m, 3))
    for i in range(m):
        ii = i*mm
        xyz[i, :] = normalise(np.array((hit_x[ii], hit_y[ii], -5)))

    fig = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers"))
    layout = {
        "autosize": False,
        "width": 900,
        "height": 790,
        "showlegend": False,
        "font": dict(family="Arial", size=18, color="black"),
        "plot_bgcolor": "white"
    }
    fig.update_layout(layout)
    fig.update_xaxes({"title": "<b>X<b>"})
    fig.update_yaxes({"title": "<b>Y<b>"})
    fig.write_html("temp3D.html", auto_open=True)
    #
    #
    # xyz = np.empty((hit_x.size, 3))
    # for i in range(hit_x.size):
    #     xyz[i, :] = normalise(np.array((hit_x[i], hit_y[i], 5)))
    #
    #
    # hist, x_hist = np.histogram(xyz[:, 2], bins=40)
    # fig = go.Figure(go.Scatter(x=x_hist[:-1], y=hist, mode="lines"))
    # fig.write_html("temp2.html", auto_open=True)
    #
    # def func(x):
    #     return np.pi * (np.sqrt(1-x**2) * x + np.arcsin(x))
    #
    # x = np.empty_like(hist, dtype="float64")
    # for i in range(hist.size):
    #     a1 = func(x_hist[i])
    #     a2 = func(x_hist[i+1])
    #     hist[i] = hist[i] / (a2-a1)
    #     x[i] = (x_hist[i] + x_hist[i + 1]) / 2
    # fig = go.Figure(go.Scatter(x=x_hist[:-1], y=hist, mode="lines"))
    # fig.write_html("temp22.html", auto_open=True)

    # x_, hist = sphere_distribution(xyz=xyz)
    # print(",".join([str(i) for i in x_]))
    # print(",".join([str(i) for i in hist]))
    # fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    # fig.write_html("temp2.html", auto_open=True)
    #
    # x_, hist = adf(xy=xyz[:, :-1])
    # fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    # fig.write_html("temp3.html", auto_open=True)


    theta = np.arctan(np.sqrt(hit_x**2 + hit_y**2) / 5)
    hist, x_hist = np.histogram(theta, bins=100)
    print(",".join([str(i) for i in x_hist]))
    print(",".join([str(i) for i in hist]))
    for i in range(hist.size):
        hist[i] = hist[i] #/ np.tan((x_hist[i]+x_hist[i+1])/2) # (2 * np.pi * (np.cos(x_hist[i]) - np.cos(x_hist[i +
        # 1])))
    fig = go.Figure(go.Scatter(x=x_hist[:-1], y=hist, mode="lines"))
    fig.write_html("temp22.html", auto_open=True)



    # x_, hist = hits_along_line(np.column_stack((hit_x, hit_y)), bins=20, normalize=True, line_angle=np.pi/8)
    # fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    # x_, hist = rdf(np.column_stack((hit_x, hit_y)), bins=20, normalize=True)
    # fig.add_trace(go.Scatter(x=x_, y=hist, mode="lines"))
    # def _pdf(_x):
    #     return 1 / (_x ** 2 + 5 ** 2)
    # fig.add_trace(go.Scatter(x=np.linspace(0, 10, 50), y=_pdf(np.linspace(0, 10, 50)) / (1 / 5 ** 2), mode="lines"))
    # fig.write_html("temp3.html", auto_open=True)

    print("hi")


class DynamicArray:
    def __init__(self, size=(100, 2), dtype: str = "float64"):
        self.data = np.empty(size, dtype)
        self.capacity = size[0]
        self.fill_level = 0

    def update(self, row):
        for r in row:
            self.add(r)

    def add(self, x):
        if self.fill_level == self.capacity:
            self.capacity *= 4
            newdata = np.zeros((self.capacity,))
            newdata[:self.fill_level] = self.data
            self.data = newdata

        self.data[self.fill_level] = x
        self.fill_level += 1

    def reduce(self):
        data = self.data[:self.fill_level]
        return np.reshape(data, newshape=(len(data)/5, 5))


if __name__ == "__main__":
    main_theory2()
    # main()
