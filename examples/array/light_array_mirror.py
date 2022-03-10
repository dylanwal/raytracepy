"""
Single Light over a horizontal plane
"""
import raytracepy as rpy

import numpy as np


def run_single(h: float):
    # define lights
    grid = rpy.OffsetGridPattern(
        center=np.array([0, 0]),
        x_length=12.5,
        y_length=12.5,
        num_points=50)

    height = h
    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.insert(xy_pos, 2, height),
                direction=np.array([0, 0, -1], dtype='float64'),
                num_traces=5,
                theta_func=1
            ))

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

    box_dim = 12.75
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

    # Important note** ground should be last in list! RayTrace simulation evaluates ray hits in order of plane list
    planes = [top, mirror_left, mirror_right, mirror_front, mirror_back, ground]

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=planes,
        lights=lights,
        total_num_rays=10_000_000,
        bounce_max=20
    )
    sim.run()
    return sim


def main_multi():
    heights = [1, 2.33, 3.66, 5, 7.5, 10, 12.5, 15]
    for height in heights:
        sim = run_single(h=height)
        file_name = f"mirror_led_{height}cm_90"
        sim.plot_report(file_name)
        print(f"h={height} done")


def main():
    sim = run_single(h=5)
    file_name = "mirror_led"
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
