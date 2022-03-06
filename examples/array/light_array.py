"""
Array of Lights over a horizontal plane

"""

import numpy as np

import raytracepy as rpy


def run_single(h: float):
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=10,
        width=10,
        bins=(100, 100)
    )

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

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=ground,
        lights=lights,
        total_num_rays=5_000_000
    )
    sim.run()
    return sim


def main_multi():
    heights = np.linspace(1, 11, 6)
    for height in heights:
        sim = run_single(h=height)
        file_name = f"array_led_{height}cm"
        sim.plot_report(file_name)
        print(f"h={height} done")


def main():
    sim = run_single(h=2)
    file_name = "array_uniform"
    sim.save_data(file_name)

    # print stats
    sim.stats()
    sim.planes["ground"].hit_stats()
    sim.planes["ground"].hit_stats(True)

    # plotting
    sim.plot_report(file_name)


if __name__ == "__main__":
    # main()
    main_multi()
