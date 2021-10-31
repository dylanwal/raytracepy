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
        length=10,
        width=10)

    # define lights
    light = rpy.Light(
        position=np.array([0, 0, height], dtype='float64'),
        direction=np.array([0, 0, -1], dtype='float64'),
        emit_light_fun_id=1
    )

    # Create ref_data class
    data = rpy.RayTraceData(
        planes=ground,
        lights=light
    )

    # pass setup/ref_data object to the simulation; then run the simulation
    sim = rpy.RayTrace(data)
    sim.run()

    # Analyze/plot output
    return data.plane["ground"].intensity


def main():
    n = 10
    height = np.linspace(0.2, 10, n)
    data = np.empty((2, n))
    for i in range(n):
        data[i][0] = height
        data[i][1] = single_run(height[i])
        print(f"{i}: {data[i]}")


if __name__ == "__main__":
    main()
