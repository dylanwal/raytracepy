from typing import Tuple, List
from enum import Enum
import inspect

import numpy as np
import plotly.graph_objs as go

from . import get_object_uid, default_plot_layout, dtype
from .utils.sig_figs import sig_figs
from .core import normalise


class TransmissionTypes(Enum):
    absorb = 0
    transmit = 1
    transmit_reflect = 2
    reflect = 3


class Plane:

    def __init__(self,
                 name: str = None,
                 position: np.ndarray = np.array([0, 0, 0], dtype=dtype),
                 normal: np.ndarray = np.array([0, 0, 1], dtype=dtype),
                 length: float = 10,
                 width: float = 10,
                 trans_type: str = "absorb",
                 transmit_func: int = 3,  # see numba_func_selector
                 scatter_func: int = 2,  # see numba_func_selector
                 reflect_func: int = 5,  # see numba_func_selector
                 bins: Tuple = (21, 21)):
        """
        This class is used to define ground planes, diffraction planes and mirrors.

        :param name: user description of plane (default id number)
        :param trans_type: ["absorb", "transmit", "tran_refl", "reflect"]
        :param transmit_func: id for function for angle dependence of transmit/absorbance
        :param scatter_func: if ray transits, id for function defines the amount of scattering
        :param position: center position of plane [x, y, z]
        :param normal: normal vector for plane [x, y, z]
        :param length: length in y direction or z direction
        :param width: length in x direction or z direction
        """
        self.uid = get_object_uid()

        if name is None:
            self.name = "plane_" + str(self.uid)
        else:
            self.name = name

        self.position = position
        self.normal = normalise(normal)

        self.length = length
        self.width = width

        self.corners = None
        self._calc_corner()

        self.transmit_type = TransmissionTypes[trans_type]
        self.transmit_func = transmit_func
        self.scatter_func = scatter_func
        self.reflect_func = reflect_func

        # data
        self.hits = None
        self.bins = bins
        self.hist = None

    def __str__(self):
        return f"Plane {self.uid}|| pos: {self.position}; norm: {self.normal}"

    def __repr__(self):
        return self.print_stats()

    def print_stats(self) -> str:
        text = "\n"
        text += f"Plane: {self.name} (uid: {self.uid})"
        text += f"\n\t pos: {self.position}, norm: {self.normal}"
        text += f"\n\t length: {self.length}, width: {self.width}"
        text += f"\n\t hits: {len(self.hits)}"
        return text

    def _calc_corner(self):
        if self.normal[0] == 0 and self.normal[1] == 0:
            self.corners = np.array([self.position[0] - self.width / 2, self.position[0] + self.width / 2,
                                     self.position[1] - self.length / 2, self.position[1] + self.length / 2,
                                     self.position[2], self.position[2]], dtype="float64")
        elif self.normal[0] == 0 and self.normal[2] == 0:
            self.corners = np.array([self.position[0] - self.width / 2, self.position[0] + self.width / 2,
                                     self.position[1], self.position[1],
                                     self.position[2] - self.width / 2, self.position[2] + self.width / 2],
                                    dtype="float64")
        elif self.normal[1] == 0 and self.normal[2] == 0:
            self.corners = np.array([self.position[0], self.position[0],
                                     self.position[1] - self.length / 2, self.position[1] + self.length / 2,
                                     self.position[2] - self.width / 2, self.position[2] + self.width / 2],
                                    dtype="float64")
        else:
            raise ValueError("non-vertical or -horizontal planes is currently not supported.")

    def generate_plane_array(self):
        """
        This function creates the grouped ref_data array for a plane. This array will be passed into the ray trace
        code in which numba requires all ref_data to be float64 to provide all ref_data necessary for calculations.

        grouped_data:
        0 = trans_type_id
        1 = trans_fun_id
        2 = trans_scatter_fun_id
        3 = reflect_fun_id
        4,5,6 = position
        7,8,9 = normal
        10,11,12,13,14,15 = corners [-x,+x,-y,+y,-z,+z]
        16 = uid
        """
        grouped_data = np.array([
            self.transmit_type.value,
            self.transmit_func,
            self.scatter_func,
            self.reflect_func], dtype=dtype)
        grouped_data = np.append(grouped_data, self.position)
        grouped_data = np.append(grouped_data, self.normal)

        for corner in self.corners:
            grouped_data = np.append(grouped_data, corner)

        return np.append(grouped_data, self.uid)

    def plane_corner_xyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Return x,y,z for the 4 corners in a 4x4 grid. """
        if self.normal[2] != 0:
            d = - np.dot(self.position, self.normal)
            xx, yy = np.meshgrid([self.corners[0], self.corners[1]], [self.corners[2], self.corners[3]])
            zz = (-self.normal[0] * xx - self.normal[1] * yy - d) * 1. / self.normal[2]
        elif self.normal[0] == 0:
            xx = np.array([[self.corners[0], self.corners[0]], [self.corners[1], self.corners[1]]])
            yy = np.array([[self.corners[2], self.corners[3]], [self.corners[2], self.corners[3]]])
            zz = np.array([[self.corners[4], self.corners[5]], [self.corners[4], self.corners[5]]])
        elif self.normal[1] == 0:
            xx = np.array([[self.corners[0], self.corners[0]], [self.corners[1], self.corners[1]]])
            yy = np.array([[self.corners[2], self.corners[3]], [self.corners[2], self.corners[3]]])
            zz = np.array([[self.corners[4], self.corners[4]], [self.corners[5], self.corners[5]]])
        else:
            raise ValueError(f"Plane normal needs to be 1 of 3 main directions.")

        return xx, yy, zz

    def create_histogram(self, kwargs: dict = None):
        kkwargs = {
            "bins": self.bins,
        }

        if kwargs is not None:
            kkwargs = kkwargs | kwargs

        hist, xedges, yedges = np.histogram2d(x=self.hits[:, 0], y=self.hits[:, 1], **kkwargs)
        if self.hist is None:
            self.hist = hist
        return hist, xedges, yedges

    def hit_stats(self, normalized: bool = False):
        if self.hist is None:
            self.create_histogram()

        headers = ["min", "1", "5", "10", "mean", "90", "95", "99", "max"]
        his_array = np.reshape(self.hist, (self.hist.shape[0] * self.hist.shape[1],))
        mean_ = float(np.mean(his_array))
        if normalized:
            data = [sig_figs(np.min(his_array) / mean_),
                    sig_figs(np.percentile(his_array, 1) / mean_),
                    sig_figs(np.percentile(his_array, 5) / mean_),
                    sig_figs(np.percentile(his_array, 10) / mean_),
                    sig_figs(mean_ / mean_),
                    sig_figs(np.percentile(his_array, 90) / mean_),
                    sig_figs(np.percentile(his_array, 95) / mean_),
                    sig_figs(np.percentile(his_array, 99) / mean_),
                    sig_figs(np.max(his_array) / mean_)]
        else:
            data = [sig_figs(np.min(his_array)),
                    sig_figs(np.percentile(his_array, 1)),
                    sig_figs(np.percentile(his_array, 5)),
                    sig_figs(np.percentile(his_array, 10)),
                    sig_figs(mean_),
                    sig_figs(np.percentile(his_array, 90)),
                    sig_figs(np.percentile(his_array, 95)),
                    sig_figs(np.percentile(his_array, 99)),
                    sig_figs(np.max(his_array))]

        return [headers, data]

    def print_hit_stats(self, normalized=False):
        data = self.hit_stats(normalized)
        s = [[str(e) for e in row] for row in data]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print("")
        print(f"Plane ({self.uid}) hit stats (norm: {normalized})")
        print('\n'.join(table))

    def plot_heat_map(self, **kwargs):
        kkwargs = {
            "nbinsx": self.bins[0],
            "nbinsy": self.bins[1]
        }

        kkwargs = kkwargs | kwargs
        heat_map = go.Histogram2d(x=self.hits[:, 0], y=self.hits[:, 1], **kkwargs)
        fig = go.Figure(heat_map)
        default_plot_layout(fig)
        return fig

    def plot_sensor(self, xy: np.ndarray, r: float, normalize: bool = False, **kwargs):
        z = self.shape_grid(xy, r)
        kkwargs = {
            "marker": dict(showscale=True, size=25, colorbar=dict(title="<b>counts</b>"))
        }

        if normalize:
            kkwargs["marker"] = dict(showscale=True, size=25, colorbar=dict(title="<b>normalized<br>irradiance</b>"))
            z = z / np.max(z)

        kkwargs = kkwargs | kwargs
        scatter = go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='markers', marker_color=z, **kkwargs)
        fig = go.Figure(scatter)
        default_plot_layout(fig)
        return fig

    def plot_surface(self, kwargs: dict = None):
        kkwargs = {}
        if kwargs is not None:
            kkwargs = kkwargs | kwargs

        surf = go.Surface(z=self.hist, **kkwargs)
        fig = go.Figure(surf)
        default_plot_layout(fig)
        return fig

    def shape_grid(self, xy: np.ndarray, r: float = 1) -> np.ndarray:
        out = np.empty(xy.shape[0])
        for i, point in enumerate(xy):
            out[i] = self.hits_in_circle(point, r)

        return out

    def hits_in_circle(self, point, r) -> int:
        """
        Counts how many hits are within r of point.
        :param point [x,y,z]
        :param r radius
        :return number of hits in circle
        """
        distance = np.linalg.norm(point - self.hits[:, 0:2], axis=1)
        return int(np.sum(distance <= r, axis=0))

    def plot_rdf(self, **kwargs):
        fig = go.Figure()
        self.plot_add_rdf(fig, **kwargs)
        default_plot_layout(fig)

        return fig

    def plot_add_rdf(self, fig, **kwargs):
        _args = [k for k, v in inspect.signature(self.rdf).parameters.items()]
        _dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in _args}
        x, hist = self.rdf(**_dict)

        line = go.Scatter(x=x, y=hist, mode="lines", **kwargs)
        fig.add_trace(line)

    def rdf(self, bins: int = 20, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculates radial density. """
        distance = np.linalg.norm(self.hits[:, 0:2], axis=1)

        flag = True
        default = self.width / 2
        while flag:
            hist, bin_edges = np.histogram(distance, bins=bins,
                                           range=(0, min([self.width / 2, self.length / 2, default])))
            if np.count_nonzero(hist) < int(hist.size / 2):
                default = default / 2  # automatic adjust window to make sure its not mostly zeros
            else:
                flag = False

        x = np.empty_like(hist, dtype="float64")
        for i in range(hist.size):
            hist[i] = hist[i] / (np.pi * (bin_edges[i + 1] ** 2 - bin_edges[i] ** 2))
            x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

        if normalize:
            hist = hist / np.max(hist)

        return x, hist

    def plot_hits_x(self, **kwargs):
        fig = go.Figure()
        self.plot_add_hits_x(fig, **kwargs)
        default_plot_layout(fig)

        return fig

    def plot_add_hits_x(self, fig, **kwargs):
        _args = [k for k, v in inspect.signature(self.hits_along_x).parameters.items()]
        _dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in _args}
        x, hist = self.hits_along_x(**_dict)

        line = go.Scatter(x=x, y=hist, mode="lines", **kwargs)
        fig.add_trace(line)

    def hits_along_x(self, delta_y: float = 0.05, bins: int = 20, normalize: bool = False) \
            -> Tuple[np.ndarray, np.ndarray]:

        mask = np.abs(self.hits[:, 1]) < delta_y
        x_dist = self.hits[mask, 0]

        hist, bin_edges = np.histogram(x_dist, bins=bins)

        x = np.empty_like(hist, dtype="float64")
        for i in range(hist.size):
            x[i] = (bin_edges[i + 1] + bin_edges[i]) / 2

        if normalize:
            hist = hist / np.max(hist)

        return x, hist
