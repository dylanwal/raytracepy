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
        length=15,
        width=15,
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
    heights = [1, 2.5, 5, 7.5, 10]   #  [1, 3, 5, 8, 10]
    num_lights = 49  # [4, 16, 36, 49, 81]
    width = 12.5  # [7.5, 10, 12.5, 15]

    for h in heights:
        grid = rpy.OffsetGridPattern(
            center=np.array([0, 0]),
            x_length=width,
            y_length=width,
            num_points=num_lights)

        sim = run_single(h=h, grid=grid)
        file_name = f"no_mirror_h_{h}"
        sim.plot_report(file_name)
        sim.planes["ground"].hit_stats()
        sim.planes["ground"].hit_stats(True)
        print(f"h_{h} done")


def main():
    grid = rpy.SpiralPattern(
        center=np.array([0, 0]),
        radius=6,
        radius_start=.5,
        velocity=0.2,
        a_velocity=1,
        num_points=50)

    sim = run_single(h=5, grid=grid)
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
