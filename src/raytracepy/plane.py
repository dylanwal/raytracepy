

import numpy as np
import plotly.graph_objs as go

from . import get_object_uid, default_plot_layout, number_type
from .core_functions import normalise
from .ref_data.light_lens_mirror_funcs import plane_func_factory


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
                 reflect_func: str = "mirror95"):
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

        # # Plane ref_data is grouped in this way for efficiency with numba
        # self.grouped_data = None
        # self.create_grouped_data()

    def __str__(self):
        return f"Plane|| id: {self.uid}; type: { self.transmit_type}; dim: {self.width} x {self.length}; pos:" \
               f" {self.position}"

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

    def plot_heat_map(self):
        # kwargs = {"z": grid}
        #
        # if z_range is not None:
        #     kwargs["zmin"] = z_range[0]
        #     kwargs["zmax"] = z_range[1]
        # if xy_range is not None:
        #     dx = (16 - 4.5) / 10
        #     dy = (25 - 8) / 10
        #     width = 600
        #     height = 600
        # else:
        #     width = 600
        #     height = 600

        heat_map = go.Histogram2d(x=self.hits[:, 0], y=self.hits[:, 1])
        fig = go.Figure(heat_map)
        default_plot_layout(fig)
        return fig
