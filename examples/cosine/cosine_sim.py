"""
Single Light over a horizontal plane rotating from right above to the side.
This simulates the cosine law.

The code runs the simulation and saves the data as pickle files, so it can be processed in a separate file.

Time it takes to compute: ~ 2 min on laptop from 2014
"""
import raytracepy as rpy

import numpy as np


def single_run(x: float, z: float):
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=20,
        width=20)

    # define lights
    light = rpy.Light(
        position=np.array([x, 0, z], dtype='float64'),
        direction=np.array([-x, 0, -z], dtype='float64'),  # so its always points back to [0, 0]
        theta_func=0
    )

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=ground,
        lights=light,
        total_num_rays=3_000_000,
    )
    sim.run()

    # Analyze/plot output
    return sim


def main():
    n = 10
    r = 5
    x = np.linspace(0, r, n)
    z = np.sqrt(r**2-x**2)
    a = np.pi/2 - np.arctan(z/x)
    data = np.empty((len(x), 4))
    for i in range(len(x)):
        data[i, 0] = x[i]
        data[i, 1] = z[i]
        data[i, 2] = a[i]
        sim = single_run(x[i], z[i])
        sim.save_data()
        data[i, 3] = float(sim.planes[0].shape_grid(xy=np.array([[0, 0]]), r=0.2))

    for i in range(len(x)):
        print(f"{data[i,2]},{data[i,3]}")


if __name__ == "__main__":
    main()
