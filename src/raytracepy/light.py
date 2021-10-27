import numpy as np


class Light:
    def __init__(self,
                 position: np.ndarray = np.array([0, 0, 10], dtype='float64'),
                 direction: np.ndarray = np.array([0, 0, -1], dtype='float64'),
                 phi: np.ndarray = np.array([0, 359.999], dtype="float64"),
                 emit_light_fun_id: int = 1,
                 power: float = 1):
        """
        This class defines light sources.

        :param position: x, y, z position of light
        :param direction: vector that the light points in
        :param phi: angles that the light is admitted at in spherical coordinates
        :param emit_light_fun_id: theta in spherical coordinates
        :param power: Intensity of light [0, 1] with respect to number of rays generated.
        """
        self.position = position
        self.direction = normalise(direction)
        self.phi = phi
        self.emit_light_fun_id = emit_light_fun_id
        self.power = power
