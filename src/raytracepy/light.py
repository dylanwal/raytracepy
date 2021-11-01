import numpy as np

from . import get_object_uid
from .core_functions import normalise
from .ref_data.light_lens_mirror_funcs import theta_factory


class Light:

    def __init__(self,
                 name: str = None,
                 position: np.ndarray = np.array([0, 0, 10], dtype='float64'),
                 direction: np.ndarray = np.array([0, 0, -1], dtype='float64'),
                 phi: np.ndarray = np.array([0, 359.999], dtype="float64"),
                 theta_func: str = "led",
                 power: float = 1,
                 num_rays: int = None
                 ):
        """
        This class defines light sources.

        :param name: user description of light (default id number)
        :param position: x, y, z position of light
        :param direction: vector that the light points in
        :param phi: angles that the light is admitted at in spherical coordinates
        :param theta_func: theta function in spherical coordinates
        :param power: Intensity of light [0, 1] with respect to number of rays generated.
        """
        self.uid = get_object_uid()

        if name is None:
            self.name = "light_" + str(self.uid)
        else:
            self.name = name

        self.position = position
        self.direction = normalise(direction)
        self.phi_deg = phi
        self.phi_rad = self.phi_deg / 360 * (2 * np.pi)
        self.theta_func = theta_factory(theta_func)
        self.power = power
        self.num_rays = num_rays

    def __str__(self):
        return f"Light (uid: {self.uid})|| pos: {self.position}; dir: {self.direction}"

    def __repr__(self):
        return self.print_stats()

    def print_stats(self) -> str:
        text = "\n"
        text += f"Light: {self.name} ({self.uid})"
        text += f"\n\t pos: {self.position}, dir: {self.direction}"
        text += f"\n\t power: {self.power}, num_rays: {self.num_rays}"
        return text
