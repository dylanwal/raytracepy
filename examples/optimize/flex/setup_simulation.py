
import numpy as np

import raytracepy as rpy


def define_planes(light_height: float = 5, mirrors: bool = False, length: float = 10, width: float = 10, **kwargs) \
        -> list[rpy.Plane]:
    ground = rpy.Plane(
            name="ground",
            position=np.array([0, 0, 0], dtype='float64'),
            normal=np.array([0, 0, 1], dtype='float64'),
            length=length,
            width=width,
            transmit_type="absorb",
            bins=(100, 100)
        )

    if mirrors:
        if 'light_width' in kwargs and "box_offset" in kwargs:
            # mirror box can't be smaller than the area of interest
            box_dim = max([width, kwargs["light_width"] + kwargs["box_offset"]])
        else:
            box_dim = width

        box_height = light_height + 0.25
        mirror_left = rpy.Plane(
            name="mirror_left",
            position=np.array([-box_dim / 2, 0, box_height / 2], dtype='float64'),
            normal=np.array([1, 0, 0], dtype='float64'),
            length=box_dim,
            width=box_height,
            transmit_type="reflect",
            reflect_func=4,
        )
        mirror_right = rpy.Plane(
            name="mirror_right",
            position=np.array([box_dim / 2, 0, box_height / 2], dtype='float64'),
            normal=np.array([-1, 0, 0], dtype='float64'),
            length=box_dim,
            width=box_height,
            transmit_type="reflect",
            reflect_func=4,
        )
        mirror_front = rpy.Plane(
            name="mirror_front",
            position=np.array([0, -box_dim / 2, box_height / 2], dtype='float64'),
            normal=np.array([0, 1, 0], dtype='float64'),
            length=box_dim,
            width=box_height,
            transmit_type="reflect",
            reflect_func=4,
        )
        mirror_back = rpy.Plane(
            name="mirror_back",
            position=np.array([0, box_dim / 2, box_height / 2], dtype='float64'),
            normal=np.array([0, -1, 0], dtype='float64'),
            length=box_dim,
            width=box_height,
            transmit_type="reflect",
            reflect_func=4,
        )
        top = rpy.Plane(
            name="top",
            position=np.array([0, 0, box_height], dtype='float64'),
            normal=np.array([0, 0, -1], dtype='float64'),
            length=box_dim,
            width=box_dim,
            transmit_type="reflect",
            reflect_func=6,
        )
        planes = [top, mirror_left, mirror_right, mirror_front, mirror_back, ground]

    else:
        planes = [ground]

    return planes


def define_lights(grid_type: str = "ogrid", light_height: float = 5, light_width: float = 10, number_lights: int = 25,
                  num_rays: int = 10_000, **kwargs) -> list[rpy.Light]:
    if grid_type == "circle":
        grid = rpy.CirclePattern(
            center=np.array([0, 0]),
            outer_radius=light_width / 2,
            layers=3,
            num_points=number_lights)
    elif grid_type == "ogrid":
        grid = rpy.OffsetGridPattern(
            center=np.array([0, 0]),
            x_length=light_width,
            y_length=light_width,
            num_points=number_lights)
    elif grid_type == "grid":
        grid = rpy.GridPattern(
            center=np.array([0, 0]),
            x_length=light_width,
            y_length=light_width,
            num_points=number_lights)
    elif grid_type == "spiral":
        grid = rpy.SpiralPattern(
            center=np.array([0, 0]),
            radius=light_width / 2,
            radius_start=.5,
            velocity=0.2,
            a_velocity=1,
            num_points=number_lights)
    else:
        raise ValueError(f"{grid_type} invalid choice.")

    # define lights
    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.insert(xy_pos, 2, light_height),
                direction=np.array([0, 0, -1], dtype='float64'),
                num_traces=5,
                theta_func=1,
                num_rays=num_rays,
            ))
    return lights


def simulation(params: dict = None, **kwargs):
    # print("simulation running")
    params = {k: v for k, v in zip(kwargs["param_names"], params)}
    if params is None:
        params = kwargs
    else:
        params = {**params, **kwargs}
    # run simulation
    lights = define_lights(**params)
    planes = define_planes(**params)
    sim = rpy.RayTrace(
        planes=planes,
        lights=lights,
        bounce_max=10,
    )
    sim.run()
    # calculate dependent parameters
    histogram = sim.planes["ground"].histogram
    his_array = np.reshape(histogram.values,
                           (histogram.values.shape[0] * histogram.values.shape[1],))

    mean_ = np.mean(his_array)  # /(sim.total_num_rays / sim.planes["ground"].bins[0] ** 2)
    std = np.std(his_array)  # 100-np.std(his_array)
    p10 = np.percentile(his_array, 10)
    p90 = np.percentile(his_array, 90)
    return mean_, std, p10, p90
