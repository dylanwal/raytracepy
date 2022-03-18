"""
Single Light over a horizontal plane

The code runs the simulation and saves the data as pickle files, so it can be processed in a separate file.

Time it takes to compute: ~ 30 sec on laptop from 2014
"""

import numpy as np

import raytracepy as rpy


def main():
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=20,
        width=20,
        bins=(100, 100)
    )

    # define lights
    light = rpy.Light(
        position=np.array([0, 0, 5], dtype='float64'),
        direction=np.array([0, 0, -1], dtype='float64'),
        num_traces=100,
        theta_func=1
    )

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=ground,
        lights=light,
        total_num_rays=10_000_000
    )
    sim.run()

    file_name = "single_led"
    # sim.save_data(file_name)

    # print stats
    sim.stats()
    ground.hit_stats()
    ground.hit_stats(True)

    # plotting
    sim.plot_report(file_name)
    print("hi")


if __name__ == "__main__":
    main()
