import numpy as np
import plotly.graph_objs as go
from scipy.integrate import cumtrapz
from raytracepy.utils.distributions import sphere_distribution

from numba import njit, config

config.DISABLE_JIT = False


def main():
    h = 5

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

    # x_, hist = sphere_distribution(xyz=np.column_stack((x,y,z)))
    # fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    # fig.write_html("temp2.html", auto_open=True)

    # x_, hist = adf(xy_hits=np.column_stack((x,y)))
    # fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    # fig.write_html("temp3.html", auto_open=True)

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

    fig = go.Figure(go.Histogram2d(x=hit_x, y=hit_y, nbinsx=20, nbinsy=20))
    fig.write_html("temp.html", auto_open=True)

    # x_, hist = adf(np.column_stack((hit_x, hit_y)), bins=40, normalize=True)
    # fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    # fig.write_html("temp2.html", auto_open=True)

    x_, hist = hits_along_axis(np.column_stack((hit_x, hit_y)), delta=0.1, normalize=True)
    fig = go.Figure(go.Scatter(x=x_, y=hist, mode="lines"))
    x_, hist = hits_along_line(np.column_stack((hit_x, hit_y)), bins=40, normalize=True, line_angle=np.pi / 4)
    fig.add_trace(go.Scatter(x=x_, y=hist, mode="lines"))
    x_, hist = rdf(np.column_stack((hit_x, hit_y)), bins=40, normalize=True)
    fig.add_trace(go.Scatter(x=x_, y=hist, mode="lines"))
    fig.add_trace(go.Scatter(x=np.linspace(0, 10, 50), y=_pdf(np.linspace(0, 10, 50)) / (1 / h ** 2), mode="lines"))
    fig.write_html("temp3.html", auto_open=True)


def generate_cdf(func, npt: int = 11, x_range=(0, 1)):
    """
    Given a distribution x and y; return n points random chosen from the distribution
    """
    x = np.linspace(x_range[0], x_range[1], npt)
    y = func(x)

    y_norm = y / np.trapz(y, x)
    cdf = cumtrapz(y_norm, x)
    cdf = np.insert(cdf, 0, 0)

    # index = np.argmin(np.abs(cdf - 1))
    # if cdf[index] > 1:
    #     cdf = cdf[0:index]
    #     x = x[0:index]

    return x, cdf


def rvs(n, x, cdf):
    _rnd = np.random.random(n)
    return np.interp(_rnd, cdf, x)


def hits_along_axis(xy_hits, delta: float = 0.05, bins: int = 20, normalize: bool = False, axis: int = 0):
    if axis:  # y-axis
        mask = np.abs(xy_hits[:, 0]) < delta
        distance = xy_hits[mask, 1]
    else:  # x-axis
        mask = np.abs(xy_hits[:, 1]) < delta
        distance = xy_hits[mask, 0]

    hist, bin_edges = np.histogram(distance, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

    if normalize:
        hist = hist / np.max(hist)

    return x, hist


def hits_along_line(xy_hits, line_point=np.array((0, 0)), line_angle: float = np.pi/4,
                    delta: float = 0.05, bins: int = 20, normalize: bool = False):

    distance_from_line = np.abs(np.cos(line_angle)*(line_point[1]-xy_hits[:, 1]) -
                                np.sin(line_angle)*(line_point[0]-xy_hits[:, 0]))
    mask = np.abs(distance_from_line) < delta
    xy_hits = xy_hits[mask, :]

    distance = np.linalg.norm(xy_hits, axis=1)

    hist, bin_edges = np.histogram(distance, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

    if normalize:
        hist = hist / np.max(hist)

    return x, hist


def rdf(xy_hits, bins: int = 20, _range=(0, 10), normalize: bool = False):
    """ Calculates radial averaged density. """
    distance = np.linalg.norm(xy_hits, axis=1)
    mask = distance > _range[0]
    distance = distance[mask]
    mask = distance < _range[1]
    distance = distance[mask]

    hist, bin_edges = np.histogram(distance, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        hist[i] = hist[i] / (np.pi * (bin_edges[i + 1]**2 - bin_edges[i]**2))
        x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

    if normalize:
        hist = hist / np.max(hist)

    return x, hist


def adf(xy_hits, bins: int = 20, _range=(0, 10), normalize: bool = False):
    """ Calculates radial averaged density. """
    distance = np.linalg.norm(xy_hits, axis=1)
    mask = distance > _range[0]
    xy_hits = xy_hits[mask, :]
    mask = distance[mask] < _range[1]
    xy_hits = xy_hits[mask, :]

    angle = np.arctan2(xy_hits[:, 0], xy_hits[:, 1])

    hist, bin_edges = np.histogram(angle, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

    if normalize:
        hist = hist / np.max(hist)

    return x, hist


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
    K = epsilon * np.identity(dim)
    return np.linalg.cholesky(K)


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


def main3():
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
    x__, hist_ = hits_along_axis(np.column_stack((hit_x, hit_y)), delta=0.1, bins=20, normalize=True)
    fig.add_trace(go.Scatter(x=x__, y=hist_, mode="lines"))
    fig.add_trace(go.Scatter(x=np.linspace(0, 10, 50), y=_pdf(np.linspace(0, 10, 50)) / (1 / h ** 2), mode="lines"))
    fig.write_html("temp3.html", auto_open=True)

    print("hi")


def main4():
    h = 5

    def _pdf(_x):
        return 1 / (_x ** 2 + h ** 2)

    n = 10_000_000
    x, cdf = generate_cdf(_pdf, npt=101, x_range=[-10, 10])
    rnd = rvs(n, x, cdf)
    _n = int(rnd.size/2)
    hit_x = rnd[:_n]
    hit_y = rnd[_n:2*_n]
    x__, hist_ = hits_along_axis(np.column_stack((hit_x, hit_y)), delta=0.1, bins=20, normalize=True)
    fig = go.Figure(go.Scatter(x=x__, y=hist_, mode="lines"))

    theta = np.random.uniform(low=0, high=2*np.pi, size=(n,))
    phi = np.arccos(2*np.random.uniform(low=0, high=1, size=(n,)) - 1) - np.pi/2
    x = np.cos(theta) * np.cos(phi)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(phi)
    t = h/z
    hit_x = x*t
    hit_y = y*t
    mask = np.abs(hit_x) < 10
    hit_x = hit_x[mask]
    hit_y = hit_y[mask]
    mask = np.abs(hit_y) < 10
    hit_x = hit_x[mask]
    hit_y = hit_y[mask]
    x__, hist_ = hits_along_axis(np.column_stack((hit_x, hit_y)), delta=0.1, bins=20, normalize=True)
    fig.add_trace(go.Scatter(x=x__, y=hist_, mode="lines"))

    fig.add_trace(go.Scatter(x=np.linspace(0, 10, 50), y=_pdf(np.linspace(0, 10, 50)) / (1 / h ** 2), mode="lines"))
    fig.write_html("temp1.html", auto_open=True)

    print("hi")


if __name__ == "__main__":
    main()
