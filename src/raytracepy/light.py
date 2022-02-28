import inspect
from functools import wraps

import numpy as np
import plotly.graph_objs as go

from . import dtype, get_object_uid, default_plot_layout
from .core import normalise
from .utils.analysis_func import sphere_distribution


class Light:
    """
    This class defines light sources.

    Attributes
    ----------
    name: str
        user description of light (default is id number)
    position:
        x, y, z position of light
    direction:
        vector that the light points in
    phi: np.ndarray
        range that the light is emitted at in spherical coordinates (degrees)
    theta_func: int
        theta function in spherical coordinates that the light is emitted at (degrees)
        see light_plane_funcs (default is uniform)
    power: float
        Intensity of light [0, 1] with respect to number of rays generated.
    num_rays: int
         number of rays the light generates.
         Typically, this is not set, and total rays in Raytrace is set -> this will autofill
    num_traces: int
        number of traces that will be captured for 3d plotting

    """
    def __init__(self,
                 name: str = None,
                 position: np.ndarray = np.array([0, 0, 10], dtype=dtype),
                 direction: np.ndarray = np.array([0, 0, -1], dtype=dtype),
                 phi: np.ndarray = np.array([0, 359.999], dtype=dtype),
                 theta_func: int = 0,  # see light_plane_funcs
                 power: float = 1,
                 num_rays: int = None,
                 num_traces: int = 10
                 ):
        """
        Parameters
        ----------
        name: str
            user description of light (default is id number)
        position:
            x, y, z position of light
        direction:
            vector that the light points in
        phi: np.ndarray
            range that the light is emitted at in spherical coordinates (degrees)
        theta_func: int
            theta function in spherical coordinates that the light is emitted at (degrees)
            see light_plane_funcs (default is uniform)
        power: float
            Intensity of light [0, 1] with respect to number of rays generated.
        num_rays: int
             number of rays the light generates.
             Typically, this is not set, and total rays in Raytrace is set -> this will autofill
        num_traces: int
            number of traces that will be captured for 3d plotting
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
        self.theta_func = theta_func
        self.power = power
        self.num_rays = num_rays
        self.rays = None

        self.num_traces = num_traces
        self.traces = None

    def __repr__(self):
        return f"Light (uid: {self.uid})|| pos: {self.position}; dir: {self.direction}"

    # Stats ############################################################################################################
    def stats(self) -> str:
        text = "\n"
        text += f"Light: {self.name} ({self.uid})"
        text += f"\n\t pos: {self.position}, dir: {self.direction}"
        text += f"\n\t power: {self.power}, num_rays: {self.num_rays}"
        print(text)
        return text

    # Plotting #########################################################################################################
    @wraps(sphere_distribution)
    def plot_sphere_dist(self, **kwargs):
        fig = go.Figure()
        self.plot_add_sphere_dist(fig, **kwargs)
        default_plot_layout(fig)

        return fig

    @wraps(sphere_distribution)
    def plot_add_sphere_dist(self, fig, **kwargs):
        _args = [k for k, v in inspect.signature(sphere_distribution).parameters.items()]
        _dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in _args}
        x, hist = self.sphere_distribution(**_dict)

        line = go.Scatter(x=x, y=hist, mode="lines", **kwargs)
        fig.add_trace(line)

    @wraps(sphere_distribution)
    def sphere_distribution(self, **kwargs):
        return sphere_distribution(self.rays, **kwargs)

    def plot_rays(self):
        fig = go.Figure()
        self.plot_add_rays(fig)
        self.plot_add_light(fig)
        default_plot_layout(fig)

        return fig

    def plot_add_rays(self, fig, **kwargs):
        points = go.Scatter3d(x=self.rays[:, 0] + self.position[0],
                              y=self.rays[:, 1] + self.position[1],
                              z=self.rays[:, 2] + self.position[2],
                              mode="markers", **kwargs)
        fig.add_trace(points)

    def plot_add_light(self, fig, **kwargs):
        points = go.Scatter3d(x=[self.position[0]],
                              y=[self.position[1]],
                              z=[self.position[2]],
                              mode="markers",
                              marker={"color": 'red', "size": 16},
                              **kwargs)
        fig.add_trace(points)
