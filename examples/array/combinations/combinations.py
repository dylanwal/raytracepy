"""
Generates huge combinational dataset across a range of properties.

* Warning: highly computational expensive to run
"""
import itertools

import numpy as np
import pandas as pd

import raytracepy as rpy


def run_single(height: float, number_lights: int, width: float, grid_type: str = "ogrid",
               mirrors: bool = False, mirror_offset: float = 0.5,
               diffuser: bool = False, diffuser_height: float = 0.5):
    surface = _define_surface(width, height, mirrors, mirror_offset, diffuser, diffuser_height)
    grid = _define_grid(grid_type, number_lights, width)
    lights = _define_lights(grid, height)

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=surface,
        lights=lights,
        bounce_max=10,
    )
    sim.run()
    return sim


def _define_surface(width: float, height: float, mirrors: bool, mirror_offset: float,
                    diffuser: bool, diffuser_height: float) -> list[rpy.Plane]:
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=10,
        width=10,
        transmit_type="absorb",
        bins=(100, 100)
    )
    if not mirrors and not diffuser:
        return [ground]

    if diffuser:
        diffuser = rpy.Plane(
            name="diffuser",
            position=np.array([0, 0, height * diffuser_height], dtype='float64'),
            normal=np.array([0, 0, 1], dtype='float64'),
            length=30,
            width=30,
            transmit_type="transmit",
            transmit_func=3,
            scatter_func=2
        )

    if not mirrors:
        return [ground, diffuser]

    box_dim = width + mirror_offset
    box_height = height + 0.25
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

    return [top, mirror_left, mirror_right, mirror_front, mirror_back, ground]


def _define_lights(grid: rpy.light_layouts.PointPattern, height: float) -> list[rpy.Light]:
    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.insert(xy_pos, 2, height),
                direction=np.array([0, 0, -1], dtype='float64'),
                num_traces=5,
                theta_func=1,
                num_rays=100_000,
            ))
    return lights


def _define_grid(grid_type: str, number_lights: int, width: float) -> rpy.light_layouts.PointPattern:
    if grid_type == "circle":
        return rpy.CirclePattern(
            center=np.array([0, 0]),
            outer_radius=width / 2,
            layers=3,
            num_points=number_lights)
    elif grid_type == "ogrid":
        return rpy.OffsetGridPattern(
            center=np.array([0, 0]),
            x_length=width,
            y_length=width,
            num_points=number_lights)
    elif grid_type == "grid":
        return rpy.GridPattern(
            center=np.array([0, 0]),
            x_length=width,
            y_length=width,
            num_points=number_lights)
    elif grid_type == "spiral":
        return rpy.SpiralPattern(
            center=np.array([0, 0]),
            radius=width / 2,
            radius_start=.5,
            velocity=0.2,
            a_velocity=1,
            num_points=number_lights)


def simulation(*args, **kwargs):
    args = args[0]
    grid_type = args[0]
    number_lights = args[1]
    height = args[2]
    width = args[3]
    mirrors = False  ##################################################################################
    if mirrors:
        mirror_offset = args[4]
    else:
        mirror_offset = 0
    diffuser = True #################################################################################
    if diffuser:
        diffuser_height = args[4]
    else:
        diffuser_height = 0

    sim = run_single(height, number_lights, width, grid_type, mirrors, mirror_offset, diffuser, diffuser_height)
    if not mirrors and not diffuser:
        sim.plot_report(f"{grid_type}_n{number_lights}_h{height}_w{width}", auto_open=False)
    elif diffuser:
        sim.plot_report(f"diffuse_{grid_type}_n{number_lights}_h{height}_w{width}_off{diffuser_height}", auto_open=False)
    else:
        sim.plot_report(f"mirror_{grid_type}_n{number_lights}_h{height}_w{width}_off{mirror_offset}", auto_open=False)
    histogram = sim.planes["ground"].histogram
    his_array = np.reshape(histogram.values, (histogram.values.shape[0] * histogram.values.shape[1],))

    return [np.mean(his_array), np.std(his_array), np.min(his_array), np.percentile(his_array, 1),
            np.percentile(his_array, 5), np.percentile(his_array, 10), np.percentile(his_array, 90),
            np.percentile(his_array, 95), np.percentile(his_array, 99), np.max(his_array)]


def main():
    # set parameters
    sim_args = ["grid_type", "number_lights", "height", "width", "diffuser_height"]  ################## "diffuser_height", "mirror_offset" ** change bool in simulation
    height = [1, 2, 3, 5, 10, 15]
    number_lights = [4, 16, 36, 49, 81]
    width = [7.5, 10, 12.5, 15]
    grid_type = ['ogrid']  # ['circle', 'ogrid', 'grid', 'spiral']
    mirror_offset = [0.1, 1, 2.5, 5, 10]
    diffuser_height = [0.05, 0.33, 0.66, 0.95]

    # build all combinations
    all_combinations = list(itertools.product(grid_type, number_lights, height, width, diffuser_height))  ########## diffuser_height, mirror_offset ** change bool in simulation
    df = pd.DataFrame(all_combinations, columns=sim_args)

    # run all combinations
    from multiprocessing import Pool
    with Pool(14) as p:
        output = p.map(simulation, all_combinations)

    # save output
    df_output = pd.DataFrame(output, columns=["mean", "std", "min", "1", "5", "10", "90", "95", "99", "max"])
    df = pd.concat([df, df_output], axis=1)
    df.to_csv("combinations.csv")


if __name__ == "__main__":
    main()
