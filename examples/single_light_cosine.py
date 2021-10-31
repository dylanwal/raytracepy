"""
Single Light over a horizontal plane rotating from right above to the side.
This simulates the cosine effect.
"""
import raytracepy as rpy

import numpy as np


def single_run(x: float, z: float):
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=10,
        width=10)

    # define lights
    light = rpy.Light(
        position=np.array([x, 0, z], dtype='float64'),
        direction=np.array([-x, 0, -z], dtype='float64'),  # so its always points back to [0, 0]
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
    x = np.linspace(0, np.pi/2, n)
    z = np.sqrt(5-x)
    data = np.empty((3, n))
    for i in range(n):
        data[i][0] = x
        data[i][1] = z
        data[i][2] = single_run(x[i], z[i])
        print(f"{i}: {data[i]}")


if __name__ == "__main__":
    main()
