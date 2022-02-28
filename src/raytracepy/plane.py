from typing import Tuple
from enum import Enum
from functools import wraps
import inspect

import numpy as np
import plotly.graph_objs as go

from . import get_object_uid, default_plot_layout, dtype
from .utils.sig_figs import sig_figs
from .core import normalise
from .utils.analysis_func import hits_along_line, rdf


class TransmissionTypes(Enum):
    absorb = 0
    transmit = 1
    transmit_reflect = 2
    reflect = 3


class Plane:
    """ Plane

        A plane can be a mirror, ground, wall, diffusor, ect.

        Attributes
        ----------
        name: str
            user description of plane (default id number)
        position: np.ndarray
            center position of plane [x, y, z]
        normal: np.ndarray
            normal vector for plane [x, y, z]
        length: float
            length in y direction or z direction
        width: float
            length in x direction or z direction
        transmit_type: str
            type of transmission
            options = ["absorb", "transmit", "tran_refl", "reflect"]
        transmit_func: int
            id for function for angle dependence of transmit/absorbance
            light_plane_funcs
        scatter_func: int
            id for function for scattering
            light_plane_funcs
        reflect_func: int
            selector for reflection function (what % gets reflected)
            light_plane_funcs
        bins: tuple[int]
            bins for heatmap
        hits: np.ndarray
            [x,y,z] coordinates of every ray that hit the surface (and was absorbed)
        hist: np.ndarray
            2d histogram of hits

    """

    def __init__(self,
                 name: str = None,
                 position: np.ndarray = np.array([0, 0, 0], dtype=dtype),
                 normal: np.ndarray = np.array([0, 0, 1], dtype=dtype),
                 length: float = 10,
                 width: float = 10,
                 transmit_type: str = "absorb",
                 transmit_func: int = 3,  # see light_plane_funcs
                 scatter_func: int = 2,  # see light_plane_funcs
                 reflect_func: int = 5,  # see light_plane_funcs
                 bins: Tuple = (21, 21)):
        """
        Parameters
        ----------
        name: str
            user description of plane (default id number)
        position: np.ndarray
            center position of plane [x, y, z]
        normal: np.ndarray
            normal vector for plane [x, y, z]
        length: float
            length in y direction or z direction
        width: float
            length in x direction or z direction
        transmit_type: str
            type of transmission
            options = ["absorb", "transmit", "tran_refl", "reflect"]
        transmit_func: int
            id for function for angle dependence of transmit/absorbance
            light_plane_funcs
        scatter_func: int
            id for function for scattering
            light_plane_funcs
        reflect_func: int
            selector for reflection function (what % gets reflected)
            light_plane_funcs
        bins: tuple[int]
            bins for heatmap
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

        self.transmit_type = TransmissionTypes[transmit_type]
        self.transmit_func = transmit_func
        self.scatter_func = scatter_func
        self.reflect_func = reflect_func

        # data
        self.hits = None
        self._bins = None
        self.bins = bins
        self._histogram = None
        self._hist_update = True

    def __repr__(self):
        return f"Plane {self.name}; {self.uid}|| pos: {self.position}; norm: {self.normal}"

    @property
    def histogram(self):
        # Calculate histogram
        if self._histogram is None or self._hist_update:
            hist, xedges, yedges = np.histogram2d(x=self.hits[:, 0], y=self.hits[:, 1], bins=self.bins)
            self._hist_update = False
            self._histogram = hist
        return self._histogram

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, bins: tuple[int]):
        self._hist_update = True
        self._bins = bins

    def _calc_corner(self):
        """ Calculates the corners of the plane. """
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

    # Stats ############################################################################################################
    def stats(self) -> str:
        """ general stats about plane"""
        text = "\n"
        text += f"Plane: {self.name} (uid: {self.uid})"
        text += f"\n\t pos: {self.position}, norm: {self.normal}"
        text += f"\n\t length: {self.length}, width: {self.width}"
        text += f"\n\t hits: {len(self.hits)}"
        print(text)
        return text

    def hit_stats(self, normalized: bool = False) -> str:
        """ stats about the hits on the plane, (can be normalized to mean)."""
        data = self._hit_stats(normalized)
        s = [[str(e) for e in row] for row in data]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        text = '\n'.join(table)
        print(text)
        return text

    def _hit_stats(self, normalized: bool = False):
        """ calculates hit stats. """
        headers = ["min", "1", "5", "10", "mean", "90", "95", "99", "max"]
        his_array = np.reshape(self.histogram, (self.histogram.shape[0] * self.histogram.shape[1],))
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

    # Plotting #########################################################################################################
    def plot_heat_map(self):
        """ Creates heatmap/2D histogram. """
        fig = go.Figure(go.Heatmap(z=self.histogram))
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
        """

        Parameters
        ----------
        xy: np.ndarray
        r: float
            radius

        Returns
        -------

        """
        out = np.empty(xy.shape[0])
        for i, point in enumerate(xy):
            out[i] = self.hits_in_circle(point, r)

        return out

    def hits_in_circle(self, point: np.ndarray, r: float) -> int:
        """
        Counts how many hits are within r of point.

        Parameters
        ----------
        point: np.ndarray
            [x,y,z] of the point of interest
        r: float
            radius

        Returns
        -------
        output: int
            hits within the radius of point
        """
        distance = np.linalg.norm(point - self.hits[:, 0:2], axis=1)
        return int(np.sum(distance <= r, axis=0))

    @wraps(rdf)
    def plot_rdf(self, **kwargs):
        fig = go.Figure()
        self.plot_add_rdf(fig, **kwargs)
        default_plot_layout(fig)

        return fig

    @wraps(rdf)
    def plot_add_rdf(self, fig, **kwargs):
        _args = [k for k, v in inspect.signature(self.rdf).parameters.items()]
        _dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in _args}
        x, hist = self.rdf(**_dict)

        line = go.Scatter(x=x, y=hist, mode="lines", **kwargs)
        fig.add_trace(line)

    @wraps(rdf)
    def rdf(self, **kwargs):
        return rdf(self.hits, **kwargs)

    @wraps(hits_along_line)
    def plot_hits_line(self, **kwargs):
        fig = go.Figure()
        self.plot_add_hits_line(fig, **kwargs)
        default_plot_layout(fig)

        return fig

    @wraps(hits_along_line)
    def plot_add_hits_line(self, fig, **kwargs):
        _args = [k for k, v in inspect.signature(self.hits_along_line).parameters.items()]
        _dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in _args}
        x, hist = self.hits_along_line(**_dict)

        line = go.Scatter(x=x, y=hist, mode="lines", **kwargs)
        fig.add_trace(line)

    @wraps(hits_along_line)
    def hits_along_line(self, **kwargs):
        return hits_along_line(self.hits[:, :2], **kwargs)
