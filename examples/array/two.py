"""
Array of Lights over a horizontal plane

"""

import numpy as np

import raytracepy as rpy


def run_single(h: float, grid):
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=10,
        width=10,
        transmit_type="absorb",
        bins=(100, 100)
    )

    # define lights
    height = h
    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.insert(xy_pos, 2, height),
                direction=np.array([0, 0, -1], dtype='float64'),
                num_traces=5,
                theta_func=1,
            ))

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=ground,
        lights=lights,
        total_num_rays=10_000_000,
    )
    sim.run()
    return sim


def main_multi():
    height = 5  # [1, 2.5, 5, 7.5, 10, 15]   #  [1, 3, 5, 8, 10]
    num_lights = 49  # [4, 16, 36, 49, 81]
    width = [7.5, 10, 12.5, 15]

    for w in width:
        # grid = rpy.CirclePattern(
        #     center=np.array([0, 0]),
        #     outer_radius=w/2,
        #     layers=3,
        #     num_points=num_lights)

        grid = rpy.OffsetGridPattern(
            center=np.array([0, 0]),
            x_length=w,
            y_length=w,
            num_points=num_lights)

        # grid = rpy.GridPattern(
        #     center=np.array([0, 0]),
        #     x_length=w,
        #     y_length=w,
        #     num_points=num_lights)
        #
        # grid = rpy.SpiralPattern(
        #     center=np.array([0, 0]),
        #     radius=w/2,
        #     radius_start=.5,
        #     velocity=0.2,
        #     a_velocity=1,
        #     num_points=num_lights)

        sim = run_single(h=height, grid=grid)
        file_name = f"ogrid_w_{w}"
        sim.plot_report(file_name)
        print(f"n_{w} done")

    # for height in heights:
    #     sim = run_single(h=height, grid=grid)
    #     file_name = f"array_led_{height}cm"
    #     sim.plot_report(file_name)
    #     print(f"h={height} done")


def main():
    grid = rpy.OffsetGridPattern(
        center=np.array([0, 0]),
        x_length=12.5,
        y_length=12.5,
        num_points=46)

    sim = run_single(h=1, grid=grid)
    file_name = "array_uniform"
    # sim.save_data(file_name)

    # print stats
    sim.stats()
    sim.planes["ground"].hit_stats()
    sim.planes["ground"].hit_stats(True)

    # plotting
    sim.plot_report(file_name)


if __name__ == "__main__":
    # main()
    main_multi()
