

import numpy as np

from .core_functions import normalise


class Plane:
    _counter = 0

    def __init__(self,
                 trans_type: str = "absorb",
                 trans_fun_id: int = 3,
                 trans_scatter_fun_id: int = 2,
                 reflect_fun_id: int = 5,
                 position=np.array([0, 0, 0], dtype='float64'),
                 normal=np.array([0, 0, 1], dtype='float64'),
                 length: float = 10, width: float = 10):
        """
        This class is used to define ground planes, diffraction planes and mirrors.
        :param trans_type: ["absorb", "transmit", "tran_refl", "reflect"]
        :param trans_fun_id: id for function for angle dependence of transmit/absorbance
        :param trans_scatter_fun_id: if ray transits, id for function defines the amount of scattering
        :param position: center position of plane [x, y, z]
        :param normal: normal vector for plane [x, y, z]
        :param length: length in y direction or z direction
        :param width: length in x direction or z direction
        """
        self.id = Plane._counter
        Plane._counter += 1

        self.trans_type = trans_type
        if trans_type == "absorb":
            self.trans_type_id = 0
        elif trans_type == "transmit":
            self.trans_type_id = 1
        elif trans_type == "tran_refl":
            self.trans_type_id = 2
        elif trans_type == "reflect":
            self.trans_type_id = 3
        self.trans_fun_id = trans_fun_id
        self.trans_scatter_fun_id = trans_scatter_fun_id
        self.reflect_fun_id = reflect_fun_id

        self.position = position
        self.normal = normalise(normal)

        self.length = length
        self.width = width
        if self.normal[0] == 0 and self.normal[1] == 0:
            self.corners = np.array([position[0] - width / 2, position[0] + width / 2,
                                     position[1] - length / 2, position[1] + length / 2,
                                     position[2], position[2]], dtype="float64")
            self.range = self.corners[:4]
        elif self.normal[0] == 0 and self.normal[2] == 0:
            self.corners = np.array([position[0] - width / 2, position[0] + width / 2,
                                     position[1], position[1],
                                     position[2] - width / 2, position[2] + width / 2], dtype="float64")
            self.range = np.hstack((self.corners[:2], self.corners[4:]))
        elif self.normal[1] == 0 and self.normal[2] == 0:
            self.corners = np.array([position[0], position[0],
                                     position[1] - length / 2, position[1] + length / 2,
                                     position[2] - width / 2, position[2] + width / 2], dtype="float64")
            self.range = self.corners[2:]
        else:
            exit("non-vertical or -horizontal planes is currently not supported.")

        # Plane data is grouped in this way for efficiency with numba
        self.grouped_data = None
        self.create_grouped_data()

    def create_grouped_data(self):
        """
        This function creates the grouped data array for a plane. This array will be passed into the ray trace code in
        which numba requires all data to be float64 to provide all data necessary for calculations.
        grouped_data:
        0 = trans_type_id
        1 = trans_fun_id
        2 = trans_scatter_fun_id
        3 = reflect_fun_id
        4,5,6 = position
        7,8,9 = normal
        10,11,12,13,14,15 = corners [-x,+x,-y,+y,-z,+z]
        """
        self.grouped_data = np.array(self.trans_type_id, dtype="float64")
        self.grouped_data = np.append(self.grouped_data, self.trans_fun_id)
        self.grouped_data = np.append(self.grouped_data, self.trans_scatter_fun_id)
        self.grouped_data = np.append(self.grouped_data, self.reflect_fun_id)
        self.grouped_data = np.append(self.grouped_data, self.position)
        self.grouped_data = np.append(self.grouped_data, self.normal)
        for corner in self.corners:
            self.grouped_data = np.append(self.grouped_data, corner)
