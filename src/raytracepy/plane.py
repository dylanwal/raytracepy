from dataclasses import dataclass
from typing import Tuple
from enum import Enum
from functools import wraps
import inspect

import numpy as np
import plotly.graph_objs as go
import pandas as pd

from . import get_object_uid, default_plot_layout, dtype, merge_html_figs
from .utils.sig_figs import sig_figs
from .core import normalise
from .utils.analysis_func import hits_along_line, rdf


class TransmissionTypes(Enum):
    absorb = 0
    transmit = 1  # with scattering
    transmit_reflect = 2
    reflect = 3


class OrientationTypes(Enum):
    horizontal = 0
    vertical_x = 1  # x doesn't change
    vertical_y = 2  # y doesn't change


@dataclass
class Histogram:
    values: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray


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
        self.orientation = None
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
    def histogram(self) -> Histogram:
        # Calculate histogram
        if self._histogram is None or self._hist_update:
            if self.orientation == OrientationTypes.vertical_y:
                x = self.hits[:, 0]
                y = self.hits[:, 2]
                xedges = np.linspace(self.corners[0], self.corners[1], self.bins[0])
                yedges = np.linspace(self.corners[4], self.corners[5], self.bins[1])

            elif self.orientation == OrientationTypes.vertical_x:
                x = self.hits[:, 1]
                y = self.hits[:, 2]
                xedges = np.linspace(self.corners[2], self.corners[3], self.bins[0])
                yedges = np.linspace(self.corners[4], self.corners[5], self.bins[1])

            else:  # self.orientation == OrientationTypes.horizontal:
                x = self.hits[:, 0]
                y = self.hits[:, 1]
                xedges = np.linspace(self.corners[0], self.corners[1], self.bins[0])
                yedges = np.linspace(self.corners[2], self.corners[3], self.bins[1])

            hist, xedges, yedges = np.histogram2d(x=x, y=y, bins=[xedges, yedges])
            self._histogram = Histogram(hist, xedges, yedges)
            self._hist_update = False

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
            self.orientation = OrientationTypes.horizontal
        elif self.normal[0] == 0 and self.normal[2] == 0:
            self.corners = np.array([self.position[0] - self.length / 2, self.position[0] + self.length / 2,
                                     self.position[1], self.position[1],
                                     self.position[2] - self.width / 2, self.position[2] + self.width / 2],
                                    dtype="float64")
            self.orientation = OrientationTypes.vertical_y
        elif self.normal[1] == 0 and self.normal[2] == 0:
            self.corners = np.array([self.position[0], self.position[0],
                                     self.position[1] - self.length / 2, self.position[1] + self.length / 2,
                                     self.position[2] - self.width / 2, self.position[2] + self.width / 2],
                                    dtype="float64")
            self.orientation = OrientationTypes.vertical_x
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
    def stats(self, print_: bool = True):
        """ general stats about plane"""
        text = "\n"
        text += f"Plane: {self.name} (uid: {self.uid})"
        text += f"\n\t pos: {self.position}, norm: {self.normal}"
        text += f"\n\t length: {self.length}, width: {self.width}"
        text += f"\n\t hits: {len(self.hits)}"
        text += f"\n\t type: {self.transmit_type}"
        if print_:
            print(text)
        else:
            return text

    def hit_stats(self, normalized: bool = False, print_: bool = True):
        """ stats about the hits on the plane, (can be normalized to mean)."""
        data = self._hit_stats(normalized)
        s = [[str(e) for e in row] for row in data]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        text = '\n'.join(table)
        if print_:
            print(text)
        else:
            return text

    def _hit_stats(self, normalized: bool = False):
        """ calculates hit stats. """
        headers = ["min", "1", "5", "10", "mean", "std", "90", "95", "99", "max"]
        his_array = np.reshape(self.histogram.values,
                               (self.histogram.values.shape[0] * self.histogram.values.shape[1],))
        mean_ = float(np.mean(his_array))
        if mean_ == 0:
            data = [0] * 9
        elif normalized:
            data = [sig_figs(np.min(his_array) / mean_),
                    sig_figs(np.percentile(his_array, 1) / mean_),
                    sig_figs(np.percentile(his_array, 5) / mean_),
                    sig_figs(np.percentile(his_array, 10) / mean_),
                    sig_figs(mean_ / mean_),
                    sig_figs(np.std(his_array) / mean_),
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
                    sig_figs(np.std(his_array)),
                    sig_figs(np.percentile(his_array, 90)),
                    sig_figs(np.percentile(his_array, 95)),
                    sig_figs(np.percentile(his_array, 99)),
                    sig_figs(np.max(his_array))]

        return [headers, data]

    def hit_stats_series(self) -> pd.Series:
        headers, data = self._hit_stats()
        return pd.Series(data, headers)

    # Plotting #########################################################################################################
    def plot_heat_map(self, save_open: bool = True, colorbar_range: tuple = None):
        """ Creates heatmap/2D histogram. """
        # edges to center points
        dx = abs(self.histogram.xedges[0] - self.histogram.xedges[1])
        x = self.histogram.xedges[:-1] + dx / 2
        dy = abs(self.histogram.yedges[0] - self.histogram.yedges[1])
        y = self.histogram.yedges[:-1] + dy / 2

        fig = go.Figure(go.Heatmap(z=self.histogram.values, x=x, y=y))

        if colorbar_range is not None:
            his_array = np.reshape(self.histogram.values,
                                   (self.histogram.values.shape[0] * self.histogram.values.shape[1],))
            fig.data[0].update(
                zmin=np.percentile(his_array, colorbar_range[0]),
                zmax=np.percentile(his_array, colorbar_range[1]))

        default_plot_layout(fig, save_open)

        return fig

    def plot_sensor(self, xy: np.ndarray, r: float, normalize: bool = False, save_open: bool = True, **kwargs):
        z = self.shape_grid(xy, r)
        kkwargs = {
            "marker": dict(showscale=True, size=18, colorbar=dict(title="<b>counts</b>"))
        }

        if normalize:
            kkwargs["marker"] = dict(showscale=True, size=18,
                                     colorbar=dict(
                                         title={
                                             "text": "<b>normalized<br>irradiance</b>",
                                             "font": {
                                                 "size": 18}})
                                     )
            z = z / np.max(z)
            # mask = z < 0.35
            # z[mask] = 0.35

        kkwargs = kkwargs | kwargs
        scatter = go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='markers', marker_color=z, **kkwargs)
        fig = go.Figure(scatter)
        default_plot_layout(fig, save_open=False)
        if save_open:
            fig.write_html("radio.html", auto_open=True)
        return fig

    def plot_surface(self, save_open: bool = True, kwargs: dict = None):
        kkwargs = {}
        if kwargs is not None:
            kkwargs = kkwargs | kwargs

        surf = go.Surface(z=self.histogram.values, **kkwargs)
        fig = go.Figure(surf)
        default_plot_layout(fig, save_open)
        return fig

    def plot_histogram(self, save_open: bool = True, bins: int = 100, kwargs: dict = None):
        kkwargs = {}
        if kwargs is not None:
            kkwargs = kkwargs | kwargs

        his_array = np.reshape(self.histogram.values,
                               (self.histogram.values.shape[0] * self.histogram.values.shape[1],))

        d1_histogram = np.histogram(his_array, bins=bins)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d1_histogram[1], y=d1_histogram[0], mode="lines"))
        default_plot_layout(fig, save_open)
        fig.update_yaxes(range=[0, np.max(d1_histogram[0])])
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
    def plot_rdf(self, save_open: bool = True, **kwargs):
        fig = go.Figure()
        self.plot_add_rdf(fig, **kwargs)
        default_plot_layout(fig, save_open)

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
        if self.orientation == OrientationTypes.vertical_y:
            x = self.hits[:, 0]
            y = self.hits[:, 2]

        elif self.orientation == OrientationTypes.vertical_x:
            x = self.hits[:, 1]
            y = self.hits[:, 2]

        else:  # self.orientation == OrientationTypes.horizontal:
            x = self.hits[:, 0]
            y = self.hits[:, 1]

        return rdf(np.column_stack((x, y)), **kwargs)

    @wraps(hits_along_line)
    def plot_hits_line(self, save_open: bool = True, **kwargs):
        fig = go.Figure()
        self.plot_add_hits_line(fig, **kwargs)
        default_plot_layout(fig, save_open)

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
        if self.orientation == OrientationTypes.vertical_y:
            x = self.hits[:, 0]
            y = self.hits[:, 2]

        elif self.orientation == OrientationTypes.vertical_x:
            x = self.hits[:, 1]
            y = self.hits[:, 2]

        else:  # self.orientation == OrientationTypes.horizontal:
            x = self.hits[:, 0]
            y = self.hits[:, 1]

        return hits_along_line(np.column_stack((x, y)), **kwargs)

    def plot_report(self, file_name: str = "report.html", auto_open: bool = True, write: bool = True,
                    plot_rdf: bool = False):
        """ Generate html report. """
        figs = [self.html_stats(), self.plot_heat_map(save_open=False)]
        if plot_rdf:
            figs.append(self.plot_rdf(bins=40, normalize=True, save_open=False))
        figs.append(self.html_hit_stats())
        figs.append(self.html_hit_stats(normalized=True))
        figs.append(self.plot_histogram(save_open=False))

        if write:
            merge_html_figs(figs, file_name, auto_open=auto_open)
        else:
            return figs

    def html_hit_stats(self, normalized: bool = False) -> str:
        """ Print hit tables in html"""
        text = self.hit_stats(normalized, print_=False)
        text = text.replace("\n", "<br>")
        text = text.replace("\t", "    ")

        if normalized:
            out = "<h2>Normalized</h2>"
        else:
            out = "<h2>Not Normalized</h2>"

        return out + "<p><pre>" + text + "</pre><p/>"

    def html_stats(self) -> str:
        text = self.stats(print_=False)
        text = text.replace("\n", "<br>")
        text = text.replace("\t", "    ")

        return f"<h2> Stats about {self.name}</h2>" + "<p><pre>" + text + "</pre><p/>"
