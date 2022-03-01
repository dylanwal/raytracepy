"""
A collection of functions that are useful for analyzing the hits of rays on a plane as histograms.
"""
from typing import Tuple

import numpy as np


def hits_along_line(xy, line_point=np.array((0, 0)), line_angle: float = np.pi / 2,
                    delta: float = 0.05, bins: int = 20, normalize: bool = False):
    """
    Calculate histogram of hits along a line.

    Parameters
    ----------
    xy: array[:,2]
        x,y positions of points.
    line_point: array[1,1]
        point that line goes through
    line_angle: float
        angle that line goes along, in radian
    delta: float
        width of line
    bins: int
        Number of bins for histogram
    normalize: bool
        Normalize histogram of

    Returns
    -------
    x: array[bins]
        x position of histogram
    hist: array[bins]
        counts, or normalized counts of histogram

    """
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
        if max_ := np.max(hist) != 0:
            hist = hist / np.max(max_)

    return x, hist


def rdf(xy, bins: int = 20, range_: Tuple[float, float] = (0, 10), normalize: bool = False):
    """
    Calculates radial averaged density.

    Parameters
    ----------
    xy: array[:,2]
        x,y positions of points.
    bins: int
        Number of bins for histogram
    range_: Tuple[float, float]
        Limits of the radial averaging
    normalize: bool
        Normalize histogram of

    Returns
    -------
    x: array[bins]
        x position of histogram (0 is the x-y plane)
    hist: array[bins]
        counts, or normalized counts of histogram
    """
    distance = np.linalg.norm(xy, axis=1)
    mask = distance > range_[0]
    distance = distance[mask]
    mask = distance < range_[1]
    distance = distance[mask]

    hist, bin_edges = np.histogram(distance, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        hist[i] = hist[i] / (np.pi * (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2))
        x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

    if normalize:
        if max_ := np.max(hist) != 0:
            hist = hist / np.max(max_)

    return x, hist


def adf(xy: np.ndarray, bins: int = 20, range_: Tuple[float, float] = (0, 10), normalize: bool = False):
    """
    Calculates angle averaged density. around the x-y plane.

    Parameters
    ----------
    xy: array[:,2]
        x,y positions of points.
    bins: int
        Number of bins for histogram
    range_: Tuple[float, float]
        Limits of the radial averaging
    normalize: bool
        Normalize histogram of

    Returns
    -------
    x: array[bins]
        x position of histogram (0 is y=0 then it goes over range [-pi,pi])
    hist: array[bins]
        counts, or normalized counts of histogram
    """
    distance = np.linalg.norm(xy, axis=1)
    mask = distance > range_[0]
    xy = xy[mask, :]
    mask = distance[mask] < range_[1]
    xy = xy[mask, :]

    angle = np.arctan2(xy[:, 0], xy[:, 1])

    hist, bin_edges = np.histogram(angle, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

    if normalize:
        if max_ := np.max(hist) != 0:
            hist = hist / np.max(max_)

    return x, hist


def sphere_distribution(xyz: np.ndarray, bins: int = 20, normalize: bool = False):
    """
    Create a histogram of rays. angle off of x,y plane.

    Parameters
    ----------
    xyz: array[:,3]
        x,y,z positions of points.
    bins: int
        Number of bins for histogram
    normalize: bool
        Normalize histogram of

    Returns
    -------
    x: array[bins]
        x position of histogram (pi/2 or -pi/2 is z-axis, as it goes to zero is z,y axis)
    hist: array[bins]
        counts, or normalized counts of histogram

    """
    angle_off_plane = np.arctan(np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2) / xyz[:, 2])
    hist, bin_edges = np.histogram(angle_off_plane, bins=bins)

    x = np.empty_like(hist, dtype="float64")
    for i in range(hist.size):
        hist[i] = hist[i] / (2 * np.pi * (np.cos(bin_edges[i]) - np.cos(bin_edges[i + 1])))
        x[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

    if normalize:
        if max_ := np.max(hist) != 0:
            hist = hist / np.max(max_)

    return x, hist
