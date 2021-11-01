"""
Single Light over a horizontal plane
"""
import raytracepy as rpy

import numpy as np


def main():
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=10,
        width=10
    )

    # define lights
    light = rpy.Light(
        position=np.array([0, 0, 5], dtype='float64'),
        direction=np.array([0, 0, -1], dtype='float64')
    )

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=ground,
        lights=light,
        total_num_rays=10_000
    )
    sim.run()

    # Analyze/plot output
    ground.plot_heat_map()
    sim.print_stats()
    ground.print_hit_stats()
    ground.print_hit_stats(True)
    print("hi")


if __name__ == "__main__":
    main()
