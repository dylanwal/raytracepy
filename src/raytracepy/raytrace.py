from typing import List, Union
import pickle
import datetime

import numpy as np
import plotly.graph_objs as go

from . import dtype, Plane, Light, time_it, default_plot_layout
import raytracepy.core as core


class BaseList:
    def __init__(self, objs: Union[List[Plane], Plane, List[Light], Light]):
        if not isinstance(objs, list):
            objs = [objs]

        self._objs = objs

    def __repr__(self):
        return repr([obj.name for obj in self._objs])

    def __call__(self):
        return self._objs

    def __getitem__(self, item: Union[str, int]):
        """

        :param item:
            int:
            string: name of object
        :return:
        """
        if isinstance(item, int):
            return self._objs[item]
        elif isinstance(item, str):
            obj = [obj for obj in self._objs if obj.name == item]
            if len(obj) != 1:
                raise ValueError("Item not found.")
            return obj[0]
        else:
            raise TypeError("Invalid Type. string or int only.")

    def __len__(self):
        return len(self._objs)

    def __iter__(self):
        for obj in self._objs:
            yield obj


class RayTrace:
    def __init__(self,
                 planes: Union[Plane, List[Plane]],
                 lights: Union[Light, List[Light]],
                 total_num_rays: int = 10_000,
                 bounce_max: int = 0):
        """

        :param planes:
        :param lights:
        :param total_num_rays:
        :param bounce_max:
        """
        self._run: bool = False
        self.planes = BaseList(planes)
        self.lights = BaseList(lights)

        self.total_num_rays = total_num_rays
        self._set_rays_per_light()
        self.bounce_max = bounce_max
        self.plane_matrix = None

    def __str__(self):
        return f"Simulation || run:{self._run} num_lights: {len(self.lights)}; num_planes: {len(self.planes)}"

    def __repr__(self):
        return self.print_stats()

    # @property
    # def bounce_total(self):
    #     return np.sum(self.traces[:, 0])
    #
    # @property
    # def bounce_avg(self):
    #     return np.mean(self.traces[:, 0])

    def save_data(self, _dir: str = None, file_name: str = "data"):
        """ Save class in pickle format. """
        _date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        file_name = f"{file_name}_{_date}"
        if _dir is not None:
            file_name = _dir + r"//" + file_name

        with open(file_name + '.pickle', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_data(file_name: str):
        """ Load Pickled class. """
        with open(file_name, 'rb') as file:
            return pickle.load(file)

    def _set_rays_per_light(self):
        """ Give total_num_rays; distribute across all light by power."""
        if not self._check_light_power_ray_num():
            self.total_num_rays = sum([light.num_rays for light in self.lights])
            return

        rays_per_power = self.total_num_rays / sum([light.power for light in self.lights])
        for light in self.lights:
            light.num_rays = int(light.power * rays_per_power)

        self.total_num_rays = sum([light.num_rays for light in self.lights])

    def _check_light_power_ray_num(self) -> bool:
        """ Return True if all lights have power attribute, or false if num_defined individually."""
        out = True
        for light in self.lights:
            if light.num_rays is not None and out:
                out = False
            elif light.num_rays is None and not out:
                raise ValueError(f"If one ray_num defined, they all need it. (light:{light.name} is missing num_ray.)")

        return out

    @time_it
    def run(self):
        """ Main Loop: Loop through each light and ray trace. """
        self._generate_plane_matrix()

        for i, light in enumerate(self.lights):
            light.traces = np.ones((light.num_traces + 1, 1 + 3 + 3 + 3 * self.bounce_max), dtype=dtype) * -1
            # (the +1 will be cut as it may have bad data, xyz_bounce[start]+ xyz_bounce[end] + xyz_max + bounce
            # counter, num_traces)

            rays_dir = core.create_rays(light.theta_func, light.direction, light.phi_rad,
                                        light.num_rays)
            light.rays = rays_dir
            rays_dir = np.append(rays_dir, np.ones(light.num_rays).reshape((light.num_rays, 1)) * -1, axis=1)
            # last row is for plane id, as rays_dir gets turned into hits matrix

            hits, light.traces = core.trace_rays(light.position, rays_dir,
                                                 self.plane_matrix, self.bounce_max, light.traces)

            light.traces = light.traces[:-1, :]  # the last row could have bad data if it didn't hit a plane
            self._unpack_hits(hits)
            print(f"Calculations for {i + 1}/{len(self.lights)} complete.")

        self._run = True

    def _generate_plane_matrix(self):
        """ Create matrix of plane data for efficient use in numba. """
        for i, plane in enumerate(self.planes):
            if self.plane_matrix is None:
                self.plane_matrix = plane.generate_plane_array()
            else:
                self.plane_matrix = np.vstack((self.plane_matrix, plane.generate_plane_array()))

        if len(self.plane_matrix.shape) == 1:
            self.plane_matrix = self.plane_matrix.reshape((1, self.plane_matrix.shape[0]))

    def _unpack_hits(self, hits: np.ndarray):
        """
        Unpack hit matrix from ray trace by placing hits by assigning hits to correct plane.
        Plane_id of -1 is a ray that hit no plane.
        """
        for plane in self.planes:
            index_ = np.where(hits[:, -1] == plane.uid)
            if plane.hits is None:
                plane.hits = hits[index_][:, :-1]
            else:
                plane.hits = np.vstack((plane.hits, hits[index_][:, :-1]))

    def stats(self):
        """ Prints stats about simulation. """
        text = "\n"
        text += f"Ray Trace Simulation Results (run: {self._run})"
        text += "\n--------------------------------"
        text += f"\nrays generated: {self.total_num_rays}"
        text += f"\nmax bounces: {self.bounce_max}"
        text += f"\nnumber of lights: {len(self.lights)}"
        text += f"\nnumber of planes: {len(self.planes)}"
        for light in self.lights:
            text += light.print_stats()
        for plane in self.planes:
            text += plane.print_stats()

        return text

    def print_stats(self):
        print(self.stats())

    def plot_traces(self):
        """ Create 3d plot of light ray traces. """

        fig = go.Figure()
        self._add_planes(fig)
        self._add_lights_3D(fig)
        self._add_ray_traces(fig)
        self._add_hits(fig, num=self.lights[0].num_traces)

        # default_plot_layout(fig)
        fig.write_html('temp.html', auto_open=True)
        return fig

    @staticmethod
    def _get_trace_plot_data(light: Light) -> np.ndarray:
        """"""
        _out = np.empty((light.traces.shape[0]*(light.traces.shape[1]-1), 3))
        _out[:] = np.NaN

        _fill_level = 0
        for trace in light.traces:
            _bounce_count = int(trace[-1]) + 2  # 2 for start and end points
            _out[_fill_level:_fill_level + _bounce_count, 0] = trace[:-1][0::3]   # x; get every 3rd one, don't want
            # last point as its the bounce count
            _out[_fill_level:_fill_level + _bounce_count, 1] = trace[1::3]  # y; get every 3rd one
            _out[_fill_level:_fill_level + _bounce_count, 2] = trace[2::3]  # z; get every 3rd one
            _fill_level += _bounce_count + 1  # the plus one is to leave a NaN in between

        return _out

    def _add_ray_traces(self, fig, **kwargs):
        """ Add traces of rays to 3d plot. """
        kkwargs = {
            "connectgaps": False,
            "line": dict(color='rgb(100,100,100)', width=2)
        }
        if kwargs:
            kkwargs = kkwargs | kwargs

        for light in self.lights:
            xyz = self._get_trace_plot_data(light)
            line = go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='lines', **kkwargs)
            fig.add_trace(line)

    def _add_planes(self, fig, **kwargs):
        """ Add planes to 3d plot. """
        kkwargs = {
            "showscale": False,
            # "surfacecolor": [-1, -1, -1],
            "colorscale": 'Gray'
        }
        if kwargs:
            kkwargs = kkwargs | kwargs

        for plane in self.planes:
            #         alpha = 0.6
            #         color = (0.5, 0.5, 0.5)
            #         if plane.trans_type == 0:
            #             alpha = 1
            #             color = (0, 0, 1)
            #         elif plane.trans_type == 1:
            #             alpha = 0.2
            #             color = (0, 0, 0)
            x, y, z = plane.plane_corner_xyz()
            surf = go.Surface(x=x, y=y, z=z, name=plane.name, **kkwargs)
            fig.add_trace(surf)

    def _add_hits(self, fig, num: int):
        """ Add hits on surface for traces. """
        for plane in self.planes:
            hits = go.Scatter3d(x=plane.hits[:num, 0], y=plane.hits[:num, 1], z=plane.hits[:num, 2], mode="markers")
            fig.add_trace(hits)

    def _add_lights_3D(self, fig, **kwargs):
        """ Add lights to 3d plot. """
        kkwargs = {
            "opacity": 0.3,
            "showscale": False,
            "anchor": "tail",
            "sizeref": 1,
            "colorscale": "Hot"
        }
        if kwargs:
            kkwargs = kkwargs | kwargs

        for light in self.lights:
            cone = go.Cone(
                x=[float(light.position[0])],
                y=[float(light.position[1])],
                z=[float(light.position[2])],
                u=[float(light.direction[0])],
                v=[float(light.direction[1])],
                w=[float(light.direction[2])],
                name=light.name, **kkwargs)
            fig.add_trace(cone)
            light.plot_add_rays(fig)

    def _add_lights_2D(self, fig, **kwargs):
        """ Add lights to 2d plot. """
        kkwargs = {
            "marker": dict(color='rgb(0,0,0)', size=10, symbol="x")
        }
        if kwargs:
            kkwargs = kkwargs | kwargs

        x = np.empty(len(self.lights))
        y = np.empty_like(x)
        for i, light in enumerate(self.lights):
            x[i] = light.position[0]
            y[i] = light.position[1]

        points = go.Scatter(x=x, y=y, mode='markers', **kkwargs)
        fig.add_trace(points)

    def plot_light_positions(self, mode: str = "2d"):
        """
        :param mode:
        :return: plot
        """
        fig = go.Figure()
        if mode == "2d":
            self._add_lights_2D(fig)
        elif mode == "3d":
            self._add_lights_3D(fig)
        else:
            raise ValueError("'2d' or '3d' only for mode.")

        default_plot_layout(fig)
        return fig

    @staticmethod
    def _plot_rays_cone(rays):
        fig = go.Figure()
        kwargs = {
            "opacity": 1,
            "showscale": False,
            "anchor": "tail",
        }

        for rays in rays:
            cone = go.Cone(
                x=[0], y=[0], z=[0],
                u=[float(rays[0])], v=[float(rays[1])], w=[float(rays[2])],
                **kwargs)
            fig.add_trace(cone)
        default_plot_layout(fig)
        return fig
