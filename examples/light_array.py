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
        length=20,
        width=20)

    # define lights
    grid = rpy.OffsetGridPattern(
        center=np.array([0, 0]),
        x_length=12.5,
        y_length=12.5,
        num_points=50)

    height = 10
    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.insert(xy_pos, 2, height),
                direction=np.array([0, 0, -1], dtype='float64'),
            ))

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=ground,
        lights=lights,
        total_num_rays=1_500_000
    )
    sim.run()

    # Analyze/plot output
    sim.save_data()
    ground.plot_heat_map()
    # sim.print_stats()
    ground.print_hit_stats()
    ground.print_hit_stats(True)
    print("hi")


if __name__ == "__main__":
    main()
