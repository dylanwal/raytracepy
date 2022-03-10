"""
Single Light over a horizontal plane
"""
import raytracepy as rpy

import numpy as np


def run_single(h, h_diff):
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
    diffuser = rpy.Plane(
        name="diffuser",
        position=np.array([0, 0, h_diff], dtype='float64'),
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

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=planes,
        lights=lights,
        total_num_rays=5_000_000,
        bounce_max=2
    )
    sim.run()
    return sim


def main_multi():
    heights = np.linspace(0.5, 4.8, 4)
    for height in heights:
        sim = run_single(h=5, h_diff=height)
        file_name = f"diffuser_led_{height}cm_90"
        sim.plot_report(file_name)
        print(f"h={height} done")


def main():
    sim = run_single(h=5, h_diff=4)
    file_name = "diffuser_led"
    #sim.save_data(file_name)

    # print stats
    sim.stats()
    sim.planes["ground"].hit_stats()
    sim.planes["ground"].hit_stats(True)

    # plotting
    sim.plot_report(file_name)


if __name__ == "__main__":
    # main()
    main_multi()
