"""
https://juanitorduz.github.io/multivariate_normal/
"""
import numpy as np
from numba import int32, float32, config, njit
from numba.experimental import jitclass

config.DISABLE_JIT = False

spec = [
    ('dim', int32),
    ('mean', float32[:,:]),
    ("covar", float32[:, :]),
    ("epsilon", float32),
]


@jitclass(spec)
class MultivariateNormal:
    def __init__(self,
                 dim: int = 2,
                 mean: np.ndarray = None,
                 covar: np.ndarray = None,
                 epsilon: float = 0.0001
                 ):
        """

        :param dim: Define dimension
        """
        self.dim = dim
        self.mean = mean
        self.covar = covar
        self.epsilon = epsilon

    @property
    def L(self):
        K = self.covar + self.epsilon * np.identity(self.dim)
        return np.linalg.cholesky(K)

    def sample(self, n: int = 1):
        rnd = np.random.normal(loc=0, scale=1, size=self.dim * n).reshape(self.dim, n)
        return self.mean + np.dot(self.L, rnd)

# @njit
# def sample(mean, L, n: int = 1):
#     dim = mean.size
#     rnd = np.random.normal(loc=0, scale=1, size=dim * n).reshape(dim, n)
#     return mean + np.dot(L, rnd)
#
#
# def calc_L(dim=2, epsilon=0.0001):
#     K = epsilon * np.identity(dim)
#     return np.linalg.cholesky(K)


@njit
def sample(mean, L, n: int = 1):
    dim = mean.size
    rnd = np.random.normal(loc=0, scale=1, size=dim * n).reshape(dim, n)
    return mean + np.dot(L, rnd)


def calc_L(dim=2, epsilon=1.0001):
    K = epsilon * np.identity(dim)
    return np.linalg.cholesky(K)


if __name__ == "__main__":
    import plotly.graph_objs as go

    mean = np.array([0, 0], dtype="float32").reshape(2, 1)
    # covar = np.eye(2, dtype="float32")
    # mv = MultivariateNormal(2, mean, covar)
    # points = mv.sample(1_000_000)

    L = calc_L()
    points = sample(mean, L, n=1_000_000)
    fig = go.Figure(go.Histogram2d(x=points[0, :], y=points[1, :]))

    # points = np.random.multivariate_normal(mean, np.eye(2), size=1_000_000)
    # fig = go.Figure(go.Histogram2d(x=points[:, 0], y=points[:, 1]))

    fig.write_html("temp.html", auto_open=True)
