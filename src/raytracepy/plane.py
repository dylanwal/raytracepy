from typing import Tuple

import numpy as np
import plotly.graph_objs as go

from . import get_object_uid, default_plot_layout, number_type
from .core_functions import normalise
from .ref_data.light_lens_mirror_funcs import plane_func_factory
from .utils.sig_figs import sig_figs

class Plane:

    def __init__(self,
                 name: str = None,
                 position: np.ndarray = np.array([0, 0, 0], dtype=number_type),
                 normal: np.ndarray = np.array([0, 0, 1], dtype=number_type),
                 length: float = 10,
                 width: float = 10,
                 trans_type: str = "absorb",
                 transmit_func: str = "ground_glass_transmit",
                 scatter_func: str = "ground_glass_diffuser",
                 reflect_func: str = "mirror95",
                 bins: Tuple = (20, 20)):
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
        self.range = None
        self._calc_corner()

        self.transmit_type = trans_type
        if trans_type == "absorb":
            self.trans_type_id = 0
        elif trans_type == "transmit":
            self.trans_type_id = 1
        elif trans_type == "tran_refl":
            self.trans_type_id = 2
        elif trans_type == "reflect":
            self.trans_type_id = 3
        self.transmit_func = plane_func_factory(transmit_func)
        self.scatter_func = plane_func_factory(scatter_func)
        self.reflect_func = plane_func_factory(reflect_func)

        # data
        self.hits = None
        self.bins = bins
        self.hist = None

        # # Plane ref_data is grouped in this way for efficiency with numba
        # self.grouped_data = None
        # self.create_grouped_data()

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
            self.range = self.corners[:4]
        elif self.normal[0] == 0 and self.normal[2] == 0:
            self.corners = np.array([self.position[0] - self.width / 2, self.position[0] + self.width / 2,
                                     self.position[1], self.position[1],
                                     self.position[2] - self.width / 2, self.position[2] + self.width / 2],
                                    dtype="float64")
            self.range = np.hstack((self.corners[:2], self.corners[4:]))
        elif self.normal[1] == 0 and self.normal[2] == 0:
            self.corners = np.array([self.position[0], self.position[0],
                                     self.position[1] - self.length / 2, self.position[1] + self.length / 2,
                                     self.position[2] - self.width / 2, self.position[2] + self.width / 2],
                                    dtype="float64")
            self.range = self.corners[2:]
        else:
            raise ValueError("non-vertical or -horizontal planes is currently not supported.")

    # def create_grouped_data(self):
    #     """
    #     This function creates the grouped ref_data array for a plane. This array will be passed into the ray trace
    #     code in which numba requires all ref_data to be float64 to provide all ref_data necessary for calculations.
    #
    #     grouped_data:
    #     0 = trans_type_id
    #     1 = trans_fun_id
    #     2 = trans_scatter_fun_id
    #     3 = reflect_fun_id
    #     4,5,6 = position
    #     7,8,9 = normal
    #     10,11,12,13,14,15 = corners [-x,+x,-y,+y,-z,+z]
    #     """
    #     self.grouped_data = np.array(self.trans_type_id, dtype="float64")
    #     self.grouped_data = np.append(self.grouped_data, self.transmit_func)
    #     self.grouped_data = np.append(self.grouped_data, self.scatter_func)
    #     self.grouped_data = np.append(self.grouped_data, self.reflect_func)
    #     self.grouped_data = np.append(self.grouped_data, self.position)
    #     self.grouped_data = np.append(self.grouped_data, self.normal)
    #     for corner in self.corners:
    #         self.grouped_data = np.append(self.grouped_data, corner)

    def create_histogram(self, **kwargs):
        kkwargs = {
            "bins": self.bins,
        }

        kkwargs = kkwargs | kwargs
        hist, xedges, yedges = np.histogram2d(x=self.hits[:, 0], y=self.hits[:, 1], **kkwargs)
        if self.hist is None:
            self.hist = hist
        return hist, xedges, yedges

    def hit_stats(self, normalized=False):
        if self.hist is None:
            self.create_histogram()

        headers = ["min", "1", "5", "10", "mean", "90", "95", "99", "max"]
        his_array = np.reshape(self.hist, (self.hist.shape[0] * self.hist.shape[1],))
        mean_ = float(np.mean(his_array))
        if normalized:
            data = [sig_figs(np.min(his_array)/mean_),
                    sig_figs(np.percentile(his_array, 1)/mean_),
                    sig_figs(np.percentile(his_array, 5)/mean_),
                    sig_figs(np.percentile(his_array, 10)/mean_),
                    sig_figs(mean_/mean_),
                    sig_figs(np.percentile(his_array, 90)/mean_),
                    sig_figs(np.percentile(his_array, 95)/mean_),
                    sig_figs(np.percentile(his_array, 99)/mean_),
                    sig_figs(np.max(his_array)/mean_)]
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
            # "nbinsx": self.bins[0],
            # "nbinsy": self.bins[1]
        }

        kkwargs = kkwargs | kwargs
        heat_map = go.Histogram2d(x=self.hits[:, 0], y=self.hits[:, 1], **kkwargs)
        fig = go.Figure(heat_map)
        layout_kwargs = {"width": 1200}
        default_plot_layout(fig, layout_kwargs)
        return fig

    def plot_sensor(self, xy: np.ndarray, r: float, normalize: bool = False, **kwargs):
        z = self.shape_grid(xy, r)
        kkwargs = {
            "marker": dict(showscale=True, size=25, colorbar=dict(title="<b>counts</b>"))
        }

        if normalize:
            kkwargs["marker"] = dict(showscale=True, size=25, colorbar=dict(title="<b>normalized<br>irradiance</b>"))
            z = z/np.max(z)

        kkwargs = kkwargs | kwargs
        scatter = go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='markers', marker_color=z, **kkwargs)
        fig = go.Figure(scatter)
        layout_kwargs = {"width": 800 * (np.max(xy[:, 0])-np.min(xy[:, 0])) / (np.max(xy[:, 1])-np.min(xy[:, 1])) + 150}
        default_plot_layout(fig, layout_kwargs)
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
        dis = np.linalg.norm(point-self.hits[:, 0:2], axis=1)
        return int(np.sum(dis <= r, axis=0))


