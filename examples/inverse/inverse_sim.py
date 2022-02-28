"""
Single Light over a plane at different heights.
Simulating the inverse intensity principal.

The code runs the simulation and saves the data as pickle files, so it can be processed in a separate file.

Time it takes to compute: ~ 2 min on laptop from 2014
"""
import raytracepy as rpy

import numpy as np


def single_run(height: float):
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=20,
        width=20)

    # define lights
    light = rpy.Light(
        position=np.array([0, 0, height], dtype='float64'),
        direction=np.array([0, 0, -1], dtype='float64'),
        theta_func=0
    )

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=ground,
        lights=light,
        total_num_rays=3_000_000
    )

    # pass setup/ref_data object to the simulation; then run the simulation
    sim.run()
    # ground.print_hit_stats()

    # Analyze/plot output
    return sim


def main():
    n = 5
    height = np.linspace(1, 10, n)
    for i in range(len(height)):
        sim = single_run(height[i])
        sim.save_data()


if __name__ == "__main__":
    main()
