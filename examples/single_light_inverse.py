"""
Single Light over a horizontal plane going from close to far away.
Simulating the inverse intensity principal.
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
    n = 10
    height = np.linspace(1, 10, n)
    for i in range(len(height)):
        sim = single_run(height[i])
        sim.save_data(r"C:\Users\nicep\Desktop\Reseach_Post\Case_studies\raytracepy\examples\inverse")


if __name__ == "__main__":
    main()
