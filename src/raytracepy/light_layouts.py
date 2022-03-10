"""
This python file contains functions that generate various lighting patterns.
They input various parameters, then output a numpy array of (x,y) positions for the lights.


"""

from abc import ABC, abstractmethod
import warnings

import numpy as np
import plotly.graph_objs as go

from . import dtype, default_plot_layout


class PointPattern(ABC):
    """

    Attributes
    ----------
    xy_points: np.array([x_positions, y_positions])
        x,y position of all points in the pattern

    """
    number_type = dtype

    @property
    @abstractmethod
    def xy_points(self):  # pragma: no cover
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

    def plot_create(self):  # pragma: no cover
        """ Create a plot of setup. """
        fig = go.Figure()
        self.plot_add(fig)
        default_plot_layout(fig)

        return fig


class CirclePattern(PointPattern):
    """
    See PointPattern for attributes
    """
    def __init__(self,
                 center: np.ndarray = np.array([0, 0]),
                 outer_radius: (float, int) = 10,
                 num_points: int = 10,
                 layers: int = 1,
                 op_center: bool = True):
        """

        Parameters
        ----------
        center: np.ndarray[2]
            Center of circle
        outer_radius: int, float
            Outer most ring size
        num_points: int
            number of points
        layers: int
            number of rings or layers
        op_center: bool
            Option to put a point on center
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
    """
    See PointPattern for attributes
    """
    def __init__(self,
                 center: np.ndarray = None,
                 corner: np.ndarray = None,
                 x_length: (int, float) = 10,
                 y_length: (int, float) = None,
                 x_count: int = None,
                 y_count: int = None,
                 num_points: int = 25):
        """
        Generates x,y locations for a grid pattern

        Parameters
        ----------
        center: np.ndarray[2]
            center of pattern
            default is [0, 0]
            * only give one center or corner
        corner: np.ndarray[2]
            corner of pattern [x, y]
            * only give one center or corner
        x_length: int, float
            x span of the pattern
        y_length: int, float
            y span of the pattern
            * if not given, it will make square
        x_count: int
             number of rows (x-direction) in pattern
             * if not provide sqrt(num_points) will be used
        y_count: int
            number of columns (y-direction) in pattern
            * if not provide sqrt(num_points) will be used
            * If offset_op=True, some layers will have "col-1" columns
        num_points: int
            number of points
            * do not provide if col and row are provided

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
    def _center_to_corner(center: np.ndarray, x_length: (int, float), y_length: (int, float)) -> np.ndarray:
        """ Calculate the corner x,y position given the center. """
        return np.array([center[0] - x_length / 2, center[1] - y_length / 2], dtype=GridPattern.number_type)

    @staticmethod
    def _corner_to_center(corner: np.ndarray, x_length: (int, float), y_length: (int, float)) -> np.ndarray:
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
    """
    See PointPattern for attributes
    """
    def __init__(self,
                 center: np.ndarray = None,
                 corner: np.ndarray = None,
                 x_length: (int, float) = 10,
                 y_length: (int, float) = None,
                 x_count: int = None,
                 y_count: int = None,
                 num_points: int = 25):
        """
        Generates x,y locations for a offset grid pattern

        Parameters
        ----------
        center: np.ndarray[2]
            center of pattern
            default is [0, 0]
            * only give one center or corner
        corner: np.ndarray[2]
            corner of pattern [x, y]
            * only give one center or corner
        x_length: int, float
            x span of the pattern
        y_length: int, float
            y span of the pattern
            * if not given, it will make square
        x_count: int
             number of rows (x-direction) in pattern
             * if not provide sqrt(num_points) will be used
        y_count: int
            number of columns (y-direction) in pattern
            * if not provide sqrt(num_points) will be used
            * If offset_op=True, some layers will have "col-1" columns
        num_points: int
            number of points
            * do not provide if col and row are provided

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
    def _center_to_corner(center: np.ndarray, x_length: (int, float), y_length: (int, float)) -> np.ndarray:
        """ Calculate the corner x,y position given the center. """
        return np.array([center[0] - x_length / 2, center[1] - y_length / 2], dtype=GridPattern.number_type)

    @staticmethod
    def _corner_to_center(corner: np.ndarray, x_length: (int, float), y_length: (int, float)) -> np.ndarray:
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


class SpiralPattern(PointPattern):
    """
    See PointPattern for attributes
    """
    def __init__(self,
                 center: np.ndarray = None,
                 radius: (int, float) = 10,
                 radius_start: (int, float) = 0,
                 velocity: (int, float) = 1,
                 a_velocity: (int, float) = 1,
                 num_points: int = 25):
        """
        Generates x,y locations for a grid pattern

        Parameters
        ----------
        center: np.ndarray[2]
            center of pattern
            default is [0, 0]
        radius: int, float
            radius of the pattern
        radius_start: int, float, np.ndarray
            time starting point
        velocity: int, float
            velocity of spiral
            * velocity + a_velocity control spacing
        a_velocity: int, float
            angular velocity of spiral
            * velocity + a_velocity control spacing
        num_points: int
            number of points
            * do not provide if col and row are provided

        """

        self.num_points = num_points
        if center is None:
            self.center = np.zeros(2, dtype=SpiralPattern.number_type)
        else:
            self.center = center
        self.radius = radius
        self.radius_start = radius_start
        self.velocity = velocity
        self.a_velocity = a_velocity

        if radius_start == 0:
            self.t_start = 0
        else:
            self.t_start = self._get_t_end(self.radius_start)

        self.t_end: (int, float) = self._get_t_end(self.radius)

        self._xy_points: np.ndarray = np.empty((self.num_points, 2), dtype=SpiralPattern.number_type)
        self._get_points()

    @property
    def xy_points(self):
        return self._xy_points

    @staticmethod
    def _spiral(t: (int, float, np.ndarray), v: float, w: float, c: np.ndarray) -> np.ndarray:
        """
        Cartesian Equation of spiral

        Parameters
        ----------
        t: int, float, np.ndarray
            time
        v: int, float
            velocity of spiral
        w: int, float
            angular velocity of spiral
        c: np.ndarray[2]
            [x,y] center

        Returns
        -------
        output: np.ndarray
            x,y position of spiral
        """
        x = (v * t + c[0]) * np.cos(w * t)
        y = (v * t + c[1]) * np.sin(w * t)
        return np.column_stack((x, y))

    @staticmethod
    def _spiral_length(t1: (int, float), t2: (int, float), v: (int, float), w: (int, float)) -> float:
        """
         Arc length of spiral between two points

        Parameters
        ----------
        t1: int, float
            start time
        t2: int, float
            end time
        v: int, float
            velocity of spiral
        w: int, float
            angular velocity of spiral

        Returns
        -------
        length: float
            arc length

        """
        def func(th: (int, float)):
            return th * np.sqrt(1+th**2) + np.log(th + np.sqrt(1+th**2))

        b = v/w
        th1 = w * t1
        th2 = w * t2
        return b/2*(func(th2) - func(th1))

    def _get_t_end(self, radius: (int, float)) -> (int, float):
        """ Finds end time for spiral based on provided radius. """
        from scipy.optimize import fsolve

        # get t end point
        def func(x):  # find were spiral equals radius
            xy = self._spiral(x, self.velocity, self.a_velocity, self.center)
            return radius - np.sqrt(xy[0, 0] ** 2 + xy[0, 1] ** 2)

        result = fsolve(func, x0=np.array([1]))

        if not np.isclose(func(result[0]), 0):
            raise ValueError("Unable to find t_end of spiral.")

        return result[0]

    def _get_points(self):
        """ Main function that calculates xy positions. """
        from scipy.optimize import fsolve
        # calc spiral length
        length = self._spiral_length(self.t_start, self.t_end, self.velocity, self.a_velocity)

        # distribute lights
        span = length / (self.num_points - 1)
        self._xy_points[0, :] = self._spiral(self.t_start, self.velocity, self.a_velocity, self.center)

        t_0 = self.t_start
        for i in range(1, self.num_points):
            def func(x):  # find where span length equals span
                span_length = self._spiral_length(t_0, x, self.velocity, self.a_velocity)
                return span - span_length

            result = fsolve(func, x0=t_0 + 0.01)
            t_0 = result[0]
            self._xy_points[i, :] = self._spiral(result[0], self.velocity, self.a_velocity, self.center)