from typing import Protocol, List, Union
import pickle
import datetime

import numpy as np

from . import Plane, Light, time_it
from .core_functions import trace_rays, get_phi, create_ray, trace_rays_lock


class GeometryObject(Protocol):
    uid: str
    name: str


class BaseList:
    def __init__(self, objs: Union[List[GeometryObject], GeometryObject]):
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
            int: uid
            string: name of object
        :return:
        """
        if isinstance(item, int):
            obj = [obj for obj in self._objs if obj.uid == item]
            if len(obj) != 1:
                raise ValueError("Item not found.")
            return obj[0]
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
                 max_bounces: int = 0,
                 num_traces: int = 100):
        """

        :param planes:
        :param lights:
        :param total_num_rays:
        :param max_bounces:
        :param num_traces:
        """
        self._run: bool = False
        self.planes = BaseList(planes)
        self.lights = BaseList(lights)

        self.total_num_rays = total_num_rays
        self._set_rays_per_light()
        self.max_bounces = max_bounces

        self.traces_per_light = int(num_traces/len(self.lights))
        self.num_traces = self.traces_per_light * len(self.lights)

    def __str__(self):
        return f"Simulation || run:{self._run} num_lights: {len(self.lights)}; num_planes: {len(self.planes)}"

    def __repr__(self):
        return self.print_stats()

    def save_data(self, file_name: str = None):
        """ Save class in pickle format. """
        if file_name is None:
            _date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            file_name = f"data_{_date}"
        with open(file_name + '.pickle', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_data(file_name: str):
        """ Load Pickled class. """
        with open(file_name, 'rb') as file:
            return pickle.load(file)

    def _set_rays_per_light(self):
        """ Give tootal_num_rays; distribute across all light by power."""
        if not self._check_light_power_ray_num():
            self.total_num_rays = sum([light.num_rays for light in self.lights])
            return

        rays_per_power = self.total_num_rays/sum([light.power for light in self.lights])
        for light in self.lights:
            light.num_rays = int(light.power * rays_per_power)

    def _check_light_power_ray_num(self) -> bool:
        """ Return True if all lights have power, or false if num_defined individually."""
        out = True
        for light in self.lights:
            if light.num_rays is not None and out:
                out = False
            elif light.num_rays is None and not out:
                raise ValueError(f"If one ray_num defined, they all need it. (light:{light.name} is missing num_ray.)")

        return out

    @time_it
    def run(self, multiproc: bool = False):
        """ Main Loop: Loop through each light and ray trace. """
        if multiproc:
            from multiprocessing import Process, Lock

            lock = Lock()
            for i, light in enumerate(self.lights):
                Process(target=self._raytrace_single_light_lock, args=(lock, light)).start()
                print(f"Calculations for {i+1}/{len(self.lights)} complete.")

        else:
            for i, light in enumerate(self.lights):
                self._raytrace_single_light(light)
                print(f"Calculations for {i+1}/{len(self.lights)} complete.")

        self._run = True

    def _raytrace_single_light(self, light: Light):
        ray_direction = self._create_ray(light)
        trace_rays(light.position,  ray_direction, self.planes, self.max_bounces)

    def _raytrace_single_light_lock(self, lock,  light: Light):
        ray_direction = self._create_ray(light)
        trace_rays_lock(lock, light.position,  ray_direction, self.planes, self.max_bounces)

    @staticmethod
    def _create_ray(light: Light) -> np.ndarray:
        """

        :param light:
        :return: direction vector of ray [[x1,y1,z1], [x2,y2,z2]]
        """
        theta = light.theta_func(light.num_rays)
        phi = get_phi(light.phi_rad, light.num_rays)

        return create_ray(theta, phi, light.direction)

    def stats(self):
        text = "\n"
        text += f"Ray Trace Simulation Results (run: {self._run})"
        text += "\n--------------------------------"
        text += f"\nrays generated: {self.total_num_rays}"
        text += f"\nmax bounces: {self.max_bounces}"
        text += f"\nnumber of lights: {len(self.lights)}"
        text += f"\nnumber of planes: {len(self.planes)}"
        for light in self.lights:
            text += light.print_stats()
        for plane in self.planes:
            text += plane.print_stats()

        return text

    def print_stats(self):
        print(self.stats())

    # # unpack ref_data for planes of interest and save
    # for plane, index in zip(data.planes, data_plane_indexing):
    #     # get the final hits for specific plane
    #     hits = out[out[:, 0] == index, 1:]  # [x, y, z] for every hit
    #     # create histogram
    #     data.histogram(plane, hits)
    #
    #     # Plane ref_data is grouped in this way for efficiency with numba
    #     if type(planes) != list:
    #         grouped_plane_data = planes.grouped_data
    #     elif type(planes) == list:
    #         grouped_plane_data = np.vstack([plane.grouped_data for plane in planes])
    #     else:
    #         exit("TypeError in planes entry to main simulation loop.")
    #

    # def plot_traces(self):
    #     plotting.plot_traces(self.planes, self.lights, self.traces)

    # def plotting_light_positions(positions):
    #     """
    #     Takes x,y and generates a simple plot of light locations.
    #     :param positions:
    #     :return: plot
    #     """
    #     x, y = np.split(positions, 2, axis=1)
    #     plt.plot(x, y, 'ko')
    #     plt.show()
    #
    # def plotting_lens_angles(x, y):
    #     """
    #     Takes x,y and generates a simple plot
    #     :param x:
    #     :param y:
    #     :return:
    #     """
    #     plt.plot(x, y, 'k-')
    #     plt.show()
    #
    # def plane_corner_xyz(plane):
    #     if plane.normal[2] != 0:
    #         d = - np.dot(plane.position, plane.normal)
    #         xx, yy = np.meshgrid([plane.corners[0], plane.corners[1]], [plane.corners[2], plane.corners[3]])
    #         zz = (-plane.normal[0] * xx - plane.normal[1] * yy - d) * 1. / plane.normal[2]
    #     else:
    #         if plane.normal[0] == 0:
    #             xx = np.array([[plane.corners[0], plane.corners[0]], [plane.corners[1], plane.corners[1]]])
    #             yy = np.array([[plane.corners[2], plane.corners[3]], [plane.corners[2], plane.corners[3]]])
    #             zz = np.array([[plane.corners[4], plane.corners[5]], [plane.corners[4], plane.corners[5]]])
    #         if plane.normal[1] == 0:
    #             xx = np.array([[plane.corners[0], plane.corners[0]], [plane.corners[1], plane.corners[1]]])
    #             yy = np.array([[plane.corners[2], plane.corners[3]], [plane.corners[2], plane.corners[3]]])
    #             zz = np.array([[plane.corners[4], plane.corners[4]], [plane.corners[5], plane.corners[5]]])
    #
    #     return xx, yy, zz
    #
    # def plot_traces(planes: list, lights: list, traces: list):
    #     ax = plt.axes(projection='3d')
    #
    #     # Planes
    #     for plane in planes:
    #         alpha = 0.6
    #         color = (0.5, 0.5, 0.5)
    #         if plane.trans_type == 0:
    #             alpha = 1
    #             color = (0, 0, 1)
    #         elif plane.trans_type == 1:
    #             alpha = 0.2
    #             color = (0, 0, 0)
    #
    #         xx, yy, zz = plane_corner_xyz(plane)
    #         ax.plot_surface(xx, yy, zz, color=color, alpha=alpha)
    #
    #     # Lights
    #     for light in lights:
    #         ax.scatter3D(light.position[0], light.position[1], light.position[2], s=40, color=(0, 0.8, 0.8))
    #         ax.quiver(light.position[0], light.position[1], light.position[2],
    #                   light.direction[0], light.direction[1], light.direction[2],
    #                   length=4, color=(0.1, 0.2, 0.5))
    #
    #     # Traces
    #     if not traces == []:
    #         for trace in traces:
    #             x = trace[:, 0]
    #             y = trace[:, 1]
    #             z = trace[:, 2]
    #             ax.plot(x, y, z)
    #             ax.scatter3D(x[-1], y[-1], z[-1], color=(1, 0, 0), alpha=1, s=4)
    #
    #     # ax.set_xlim3d(-10, 1)
    #     # ax.set_ylim3d(-1, 1)
    #     # ax.set_zlim3d(-1, 1)
    #     plt.show()
    #
    # def plot_hist(his, plane):
    #     plt.imshow(his, origin='lower', aspect='auto', extent=plane.range)
    #     cb = plt.colorbar()
    #     cb.set_label("density")
    #     plt.clim(0, np.percentile(np.reshape(his, (his.shape[0] * his.shape[1],)), 95))
    #     plt.show()
