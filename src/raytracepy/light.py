
import numpy as np


class Light:
    def __init__(self, position=np.array([0, 0, 10], dtype='float64'),
                 direction=np.array([0, 0, -1], dtype='float64'),
                 phi=np.array([0, 359.999], dtype="float64"),
                 emit_light_fun_id: int = 1,
                 power: float = 1):
        """

        :param position:
        :param direction:
        :param phi:
        :param emit_light_fun_id: theta in spherical coordinates
        :param power:
        """
        self.position = position
        self.direction = normalise(direction)
        self.phi = phi
        self.emit_light_fun_id = emit_light_fun_id
        self.power = power
