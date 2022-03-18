
import numpy as np

import raytracepy as rpy
from scipy import optimize


def run_single(lights):
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

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=ground,
        lights=lights,
        total_num_rays=50_000,
    )
    sim.run()
    return sim


def define_lights(height: (int, float), xy: np.ndarray) -> list[rpy.Light]:
    # define lights
    lights = []
    for i in range(int(len(xy)/2)):
        lights.append(
            rpy.Light(
                position=np.array((xy[2*i], xy[2*i+1], height)),
                direction=np.array([0, 0, -1], dtype='float64'),
                num_traces=3,
                theta_func=1,
            ))
    return lights

count = 0
def sim(params: np.ndarray):
    global count
    print(count)
    count += 1
    lights = define_lights(height=params[0], xy=params[1:])
    sim = run_single(lights)
    histogram = sim.planes["ground"].histogram
    his_array = np.reshape(histogram.values,
                           (histogram.values.shape[0] * histogram.values.shape[1],))
    mean_ = float(np.mean(his_array))
    std = np.std(his_array)

    return std


def main():
    n = 9

    height_bound = (0.5, 15)
    x_bound = (-10, 10)
    y_bound = (-10, 10)
    bounds = [height_bound] + [x_bound, y_bound] * n

    # Initialize
    grid = rpy.SpiralPattern(
        center=np.array([0, 0]),
        radius=10,
        radius_start=.5,
        velocity=0.2,
        a_velocity=1,
        num_points=n)
    guess = np.zeros(n*2+1)
    guess[0] = 5
    guess[1::2] = grid.xy_points[:, 0]
    guess[2::2] = grid.xy_points[:, 1]

    results = optimize.brute(sim, bounds, Ns=3, full_output=False, disp=True)
    print(results)
    print("done")


if __name__ == "__main__":
    main()
