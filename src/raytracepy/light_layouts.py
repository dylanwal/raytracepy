"""
This python file contains functions that generate various lighting patterns.
They input various parameters, then output a numpy array of (x,y) positions for the lights.


"""

from abc import ABC, abstractmethod
import warnings

import numpy as np
import plotly.graph_objs as go

from . import number_type, default_plot_layout


class PointPattern(ABC):
    number_type = number_type

    @property
    @abstractmethod
    def xy_points(self): # pragma: no cover
        raise NotImplementedError

    def plot_add(self, fig, **kwargs):  # pragma: no cover
        """ Add points to an existing plot."""
        styling = {
            "mode": 'markers',
            "marker": dict(color='rgb(0,0,0)', size=10, symbol="x")
        }
        if kwargs:
            styling = styling | kwargs

        fig.add_trace(go.Scatter(x=self.xy_points[:, 0], y=self.xy_points[:, 1], **styling))

    def plot_create(self): # pragma: no cover
        """ Create a plot of setup. """
        fig = go.Figure()
        self.plot_add(fig)
        default_plot_layout(fig)

        fig.write_html('tmp.html', auto_open=True)
        #fig.show()
        return fig


class CirclePattern(PointPattern):

    def __init__(self,
                 center: np.ndarray = np.array([0, 0]),
                 outer_radius: float = 10,
                 num_points: int = 10,
                 layers: int = 1,
                 op_center: bool = True):
        """

        :param center: Center of circle
        :param outer_radius: Outer most ring size
        :param num_points: number of points
        :param layers:  number of rings or layers
        :param op_center: Option to put a point on center
        """
        self.center = center
        self.outer_radius = outer_radius
        self.num_points = num_points
        self.layers = layers
        self.op_center = op_center

        self._xy_points: np.ndarray = np.empty((num_points, 2), dtype=CirclePattern.number_type)
        self._radii: np.ndarray = np.empty(layers, dtype=CirclePattern.number_type)
        self._points_per_radii = np.empty(layers, dtype=CirclePattern.number_type)

        self._get_points()

    @property
    def xy_points(self):
        return self._xy_points

    @property
    def radii(self):
        return self._radii

    @property
    def points_per_radii(self):
        return self._points_per_radii

    def _get_points(self):
        """Main Function"""
        self._add_center_point()
        self._get_radii_of_rings()
        self._get_points_per_ring()
        self._add_ring_points()

    def _add_center_point(self):
        """Add center point to xy_points"""
        if self.op_center:
            self.xy_points[0, :] = self.center

    def _get_radii_of_rings(self):
        """ Determine radiusof each layer."""
        for i in range(self.layers):
            self._radii[i] = self.outer_radius - i * (self.outer_radius / self.layers)

    def _get_points_per_ring(self):
        """Figure out number of points per layer"""
        if self.op_center:
            _num_points = self.num_points - 1
        else:
            _num_points = self.num_points

        sum_radii = np.sum(self._radii)
        point_per_radii = _num_points / sum_radii
        self._points_per_radii = np.round_(point_per_radii * self._radii)

        # Check to make sure all lights have been added to a layer (rounding of number may lose one)
        points_accounted_for = np.sum(self._points_per_radii)
        if _num_points != points_accounted_for:
            self._points_per_radii[-1] += _num_points - points_accounted_for

    def _add_ring_points(self):
        """find xy_points around each ring"""
        if self.op_center:
            k = 1
        else:
            k = 0

        for radius, points in zip(self._radii, self._points_per_radii):
            angle_slice = 2 * np.pi / points
            for i in range(int(points)):
                angle = angle_slice * i
                self.xy_points[k, :] = [
                    self.center[0] + radius * np.cos(angle),  # x
                    self.center[1] + radius * np.sin(angle)  # y
                ]
                k += 1


class GridPattern(PointPattern):

    def __init__(self,
                 center: np.ndarray = None,
                 corner: np.ndarray = None,
                 x_length: float = 10,
                 y_length: float = None,
                 x_count: int = None,
                 y_count: int = None,
                 num_points: int = 25):
        """
        Generates x,y locations for a grid pattern
        :param center: center of pattern (defualt is [0, 0]) (only give one center or corner)
        :param corner: corner of pattern (only give one center or corner)

        :param x_length: x span of the pattern
        :param y_length: y span of the pattern (if not given, it will make square)


        (do not provide if num_points are provided)
        :param x_count: number of rows (x-direction) in pattern; * if not provide squt(num_points) will be used
        :param y_count: number of columns (y-direction) in pattern; * if not provide squt(num_points) will be used
        (If offset_op=True, some layers will be have "col-1" columns)

        (do not provide if col and row are provided)
        :param num_points: number of points

        :return: np.array([x_positions, y_positions])
        """
        self.x_length = x_length
        if y_length is None:
            self.y_length = x_length  # make a square
        else:
            self.y_length = y_length

        if y_count is None and x_count is not None:
            self.y_length = 0
        elif y_count is not None and x_count is None:
            self.x_length = 0

        self.x_count = None
        self.y_count = None
        self.num_points = None
        self._set_counts(x_count, y_count, num_points)

        self.corner = None
        self.center = None
        self._set_center_corner(center, corner)

        self._xy_points: np.ndarray = np.empty((self.num_points, 2), dtype=GridPattern.number_type)
        self._get_points()

    @property
    def xy_points(self):
        return self._xy_points

    def _set_counts(self, x_count, y_count, num_points):
        if x_count is None and y_count is None:
            self.x_count = int(np.sqrt(num_points))  # make square with equal number points in grid
            self.y_count = self.x_count
            if (self.x_count * self.y_count) != num_points:
                warnings.warn(
                    f"'num_points' changed! The given num_points ({num_points}) was adjusted/set to "
                    f"{self.num_points} to get a complete pattern.")
        elif x_count is not None and y_count is not None:
            self.x_count = x_count
            self.y_count = y_count
        elif x_count is not None and y_count is None:
            self.x_count = x_count
            self.y_count = 1
        else:  # x_count is None and y_count is not None
            self.x_count = 1
            self.y_count = y_count

        self.num_points = self.x_count * self.y_count

    def _set_center_corner(self, center, corner):
        if corner is not None:
            self.corner = corner
            self.center = self._corner_to_center(self.corner, self.x_length, self.y_length)
        elif center is not None:
            self.center = center
            self.corner = self._center_to_corner(self.center, self.x_length, self.y_length)
        else:
            self.center = np.array([0, 0], dtype=GridPattern.number_type)
            self.corner = self._center_to_corner(self.center, self.x_length, self.y_length)

    @staticmethod
    def _center_to_corner(center: np.ndarray, x_length: float, y_length: float) -> np.ndarray:
        """ Calculate the corner x,y position given the center. """
        return np.array([center[0] - x_length / 2, center[1] - y_length / 2], dtype=GridPattern.number_type)

    @staticmethod
    def _corner_to_center(corner: np.ndarray, x_length: float, y_length: float) -> np.ndarray:
        """ Calculate the corner x,y position given the center. """
        return np.array([corner[0] + x_length / 2, corner[1] + y_length / 2], dtype=GridPattern.number_type)

    def _get_points(self):
        """ Main function that calculates grid xy positions. """
        if self.y_count == 1:
            dy = 0
        else:
            dy = self.y_length / (self.y_count - 1)
        if self.x_count == 1:
            dx = 0
        else:
            dx = self.x_length / (self.x_count - 1)

        k = 0
        for i in range(self.y_count):  # loop over row
            for ii in range(self.x_count):  # loop over column
                self.xy_points[k, :] = [
                    self.corner[0] + dx * ii,
                    self.corner[1] + dy * i
                ]
                k += 1


class OffsetGridPattern(PointPattern):

    def __init__(self,
                 center: np.ndarray = None,
                 corner: np.ndarray = None,
                 x_length: float = 10,
                 y_length: float = None,
                 x_count: int = None,
                 y_count: int = None,
                 num_points: int = 25):
        """
        Generates x,y locations for a grid pattern
        :param center: center of pattern (defualt is [0, 0]) (only give one center or corner)
        :param corner: corner of pattern (only give one center or corner)

        :param x_length: x span of the pattern
        :param y_length: y span of the pattern (if not given, it will make square)


        (do not provide if num_points are provided)
        :param x_count: number of rows (x-direction) in pattern; * if not provide squt(num_points) will be used
        :param y_count: number of columns (y-direction) in pattern; * if not provide squt(num_points) will be used
        (If offset_op=True, some layers will be have "col-1" columns)

        (do not provide if col and row are provided)
        :param num_points: number of points

        :return: np.array([x_positions, y_positions])
        """
        self.x_length = x_length
        if y_length is None:
            self.y_length = x_length  # make a square
        else:
            self.y_length = y_length

        if y_count is None and x_count is not None:
            self.y_length = 0
        elif y_count is not None and x_count is None:
            self.x_length = 0

        self.x_count = None
        self.y_count = None
        self.num_points = None
        self._set_counts(x_count, y_count, num_points)

        self.corner = None
        self.center = None
        self._set_center_corner(center, corner)

        self._xy_points: np.ndarray = np.empty((self.num_points, 2), dtype=GridPattern.number_type)
        self._get_points()

    @property
    def xy_points(self):
        return self._xy_points

    def _set_counts(self, x_count, y_count, num_points):
        if x_count is None and y_count is None:
            # make square with equal number points in grid
            self.x_count = int(np.sqrt(num_points)) + 1
            self.y_count = self.x_count
            self.num_points = self._calc_num_points_offset_grid(self.x_count)
            if self.num_points >= num_points:
                self.x_count = int(np.sqrt(num_points))
                self.y_count = self.x_count
                self.num_points = self._calc_num_points_offset_grid(self.x_count)
        elif x_count is not None and y_count is not None:
            self.x_count = x_count
            self.y_count = y_count
            self.num_points = x_count * y_count
        elif x_count is not None and y_count is None:
            self.x_count = x_count
            self.y_count = 1
            self.num_points = x_count
        else:  # x_count is None and y_count is not None:
            self.x_count = 1
            self.y_count = y_count
            self.num_points = y_count

        if self.num_points != num_points:
            warnings.warn(
                f"'num_points' changed! The given num_points ({num_points}) was adjusted/set to "
                f"{self.num_points} to get a complete pattern.")

    @staticmethod
    def _calc_num_points_offset_grid(x_count):
        return int(
                    np.ceil(x_count / 2) * x_count +  # rows with higher point count * number of points
                    np.floor(x_count / 2) * (x_count - 1)  # rows with lower point count * number of points
                )

    def _set_center_corner(self, center, corner):
        if corner is not None:
            self.corner = corner
            self.center = self._corner_to_center(self.corner, self.x_length, self.y_length)
        elif center is not None:
            self.center = center
            self.corner = self._center_to_corner(self.center, self.x_length, self.y_length)
        else:
            self.center = np.array([0, 0], dtype=GridPattern.number_type)
            self.corner = self._center_to_corner(self.center, self.x_length, self.y_length)

    @staticmethod
    def _center_to_corner(center: np.ndarray, x_length: float, y_length: float) -> np.ndarray:
        """ Calculate the corner x,y position given the center. """
        return np.array([center[0] - x_length / 2, center[1] - y_length / 2], dtype=GridPattern.number_type)

    @staticmethod
    def _corner_to_center(corner: np.ndarray, x_length: float, y_length: float) -> np.ndarray:
        """ Calculate the corner x,y position given the center. """
        return np.array([corner[0] + x_length / 2, corner[1] + y_length / 2], dtype=GridPattern.number_type)

    def _get_points(self):
        """ Main function that calculates grid xy positions. """
        if self.y_count == 1:
            dy = 0
        else:
            dy = self.y_length / (self.y_count - 1)

        k = 0
        for ii in range(self.y_count):
            # offset rows
            if ii % 2 == 1:  # odd lines
                if self.x_count == 1:
                    dx = 0
                else:
                    dx = self.x_length / (2 * (self.x_count - 1))

                for i in range(self.x_count - 1):
                    self.xy_points[k, :] = [
                        self.corner[0] + dx * 2 * i + dx,
                        self.corner[1] + dy * ii
                    ]
                    k += 1
                continue

            # normal rows
            if self.x_count == 1:
                dx = 0
            else:
                dx = self.x_length / (self.x_count - 1)

            for i in range(self.x_count):
                self.xy_points[k, :] = [
                    self.corner[0] + dx * i,
                    self.corner[1] + dy * ii
                ]
                k += 1
