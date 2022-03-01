"""
Single Light over a horizontal plane with a diffuser in between
"""
import raytracepy as rpy

import numpy as np


def main():
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=20,
        width=20)
    diffuser = rpy.Plane(
        name="diffuser",
        position=np.array([0, 0, 4], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=30,
        width=30,
        transmit_type="transmit",
        transmit_func=3,
        scatter_func=2
    )

    # Important note** ground should be last in list! RayTrace simulation evaluates ray hits in order of plane list
    planes = [diffuser, ground]

    # define lights
    light = rpy.Light(
        position=np.array([0, 0, 5], dtype='float64'),
        direction=np.array([0, 0, -1], dtype='float64'),
        num_traces=100,
        theta_func=0
    )

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=planes,
        lights=light,
        total_num_rays=5_000_000,
        bounce_max=1  # needs at least one bounce to make through the diffuser
    )
    sim.run()

    file_name = "diffuser"
    sim.save_data(file_name)

    # print stats
    sim.stats()
    ground.hit_stats()
    ground.hit_stats(True)

    # plotting
    sim.plot_report(file_name)


if __name__ == "__main__":
    main()
