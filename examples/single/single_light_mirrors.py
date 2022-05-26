"""
Single Light over a horizontal plane
"""
import raytracepy as rpy

import numpy as np


def main():
    # define lights
    height = 10
    light = rpy.Light(
        position=np.array([0, 0, height], dtype='float64'),
        direction=np.array([0, 0, -1], dtype='float64'),
        num_traces=100,
        theta_func=0
    )

    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=10,
        width=10,
        transmit_type="absorb",
    )

    box_dim = 10
    box_height = height
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

    # Important note** ground should be last in list! RayTrace simulation evaluates ray hits in order of plane list
    planes = [top, mirror_left, mirror_right, mirror_front, mirror_back, ground]

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=planes,
        lights=light,
        total_num_rays=5_000_000,
        bounce_max=20
    )
    sim.run()

    file_name = "mirror_wide"
    # sim.save_data(file_name)

    # print stats
    sim.stats()
    ground.hit_stats()
    ground.hit_stats(True)

    # plotting
    sim.plot_report(file_name)


if __name__ == "__main__":
    main()
