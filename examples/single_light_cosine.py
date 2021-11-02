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
    )

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=ground,
        lights=light,
        total_num_rays=50_000
    )
    sim.run()

    # Analyze/plot output
    return float(ground.shape_grid(xy=np.array([[0, 0]]), r=2))


def main():
    n = 10
    x = np.linspace(0, 5, n)
    z = np.sqrt(25-x**2)
    a = np.pi/2 - np.arctan(z/x)
    data = np.empty((n, 4))
    for i in range(n):
        data[i, 0] = x[i]
        data[i, 1] = z[i]
        data[i, 2] = a[i]
        data[i, 3] = single_run(x[i], z[i])

    print(data)


if __name__ == "__main__":
    main()
