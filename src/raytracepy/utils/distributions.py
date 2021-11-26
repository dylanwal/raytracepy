import numpy as np
from scipy.integrate import cumtrapz


def generate_cdf(func, npt: int = 11, x_range=(0, 1)):
    """
    Given a function; return n cumulative distribution function.
    """
    x = np.linspace(x_range[0], x_range[1], npt)
    y = func(x)

    y_norm = y / np.trapz(y, x)
    cdf = cumtrapz(y_norm, x)
    cdf = np.insert(cdf, 0, 0)

    # deal with numerical round off errors
    y, index_ = np.unique(y, return_index=True)
    x = x[index_]

    return x, cdf


def sample_from_cdf(n: int, x: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """ Given a cumulative distribution function generate n random numbers. """
    _rnd = np.random.random(n)
    return np.interp(_rnd, cdf, x)


def hits_along_axis(xy: np.ndarray, delta: float = 0.05, bins: int = 20, normalize: bool = False, axis: int = 0):
    """
    Calculate histogram of hits along x or y axis.
    """
    if axis:  # y-axis
        mask = np.abs(xy[:, 0]) < delta
        distance = xy[mask, 1]
    else:  # x-axis
        mask = np.abs(xy[:, 1]) < delta
        distance = xy[mask, 0]

    hist, bin_edges = np.histogram(distance, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

    if normalize:
        hist = hist / np.max(hist)

    return x, hist


def hits_along_line(xy, line_point=np.array((0, 0)), line_angle: float = np.pi / 4,
                    delta: float = 0.05, bins: int = 20, normalize: bool = False):
    """ Calculate histogram of hits along a line. """
    distance_from_line = np.abs(np.cos(line_angle) * (line_point[1] - xy[:, 1]) -
                                np.sin(line_angle) * (line_point[0] - xy[:, 0]))
    mask = np.abs(distance_from_line) < delta
    xy = xy[mask, :]

    distance = np.linalg.norm(xy, axis=1)

    hist, bin_edges = np.histogram(distance, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

    if normalize:
        hist = hist / np.max(hist)

    return x, hist


def rdf(xy, bins: int = 20, _range=(0, 10), normalize: bool = False):
    """ Calculates radial averaged density. 0 is the x-y plane, then pi/2 is vertical (z-axis)."""
    distance = np.linalg.norm(xy, axis=1)
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


def adf(xy: np.ndarray, bins: int = 20, _range=(0, 10), normalize: bool = False):
    """ Calculates angle averaged density. around the x-y plane. 0 is y=0 then it goes over range [-pi,pi]. """
    distance = np.linalg.norm(xy, axis=1)
    mask = distance > _range[0]
    xy = xy[mask, :]
    mask = distance[mask] < _range[1]
    xy = xy[mask, :]

    angle = np.arctan2(xy[:, 0], xy[:, 1])

    hist, bin_edges = np.histogram(angle, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

    if normalize:
        hist = hist / np.max(hist)

    return x, hist


def sphere_distribution(xyz: np.ndarray, bins: int = 20, norm: bool = True):
    """
    Create a histogram of rays. angle off of x,y plane.

    Parameters
    ----------
    xyz: array[:,3]
        x,y,z positions of points.
    bins: int
        Number of bins for histogram
    norm: bool
        Normalize histogram of

    Returns
    -------
    x: array[bins]
        x position of histogram
    hist: array[bins]
        counts, or normalized counts of histogram

    """
    angle_off_plane = np.arctan(xyz[:, 2] / np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2))
    hist, bin_edges = np.histogram(angle_off_plane, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        if norm:
            hist[i] = hist[i] / (2 * np.pi * (
                        (1 - np.cos(np.pi / 2 - bin_edges[i])) - (1 - np.cos(np.pi / 2 - bin_edges[i + 1]))))
        x[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

    return x, hist
