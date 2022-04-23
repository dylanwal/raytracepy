"""
Array of Lights over a flat surface
"""

import numpy as np

import raytracepy as rpy


def main():
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
    height = 5
    grid = rpy.OffsetGridPattern(
        center=np.array([0, 0]),
        x_length=12.5,
        y_length=12.5,
        num_points=50)
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

    # print stats
    sim.stats()
    sim.planes["ground"].hit_stats()
    sim.planes["ground"].hit_stats(True)

    # plotting
    sim.plot_report("array_uniform")


if __name__ == "__main__":
    main()
