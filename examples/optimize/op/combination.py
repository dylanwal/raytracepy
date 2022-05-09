import itertools

import numpy as np
import pandas as pd

from problem import simulation

import raytracepy as rpy


def linspace_from_bounds(bound_lower, bound_upper, num_samples: int = 1, seed: int = None):
    return np.linspace(bound_lower, bound_upper, num_samples)


def metric(sim: rpy.RayTrace):
    # calculate dependent parameters
    histogram = sim.planes["ground"].histogram
    his_array = np.reshape(histogram.values, (histogram.values.shape[0] * histogram.values.shape[1],))

    mean_ = np.mean(his_array)
    std = np.std(his_array)
    return mean_, std


def evaluate(param, **kwargs):
    sim = simulation({**param, **kwargs})
    return metric(sim)


def main():
    n = 5

    # build all combinations
    params = ["light_width", "light_height", "mirror_offset"]
    light_height = linspace_from_bounds(1, 15, n)
    light_width = linspace_from_bounds(5, 15, n)
    mirror_offset = linspace_from_bounds(0.1, 10, n)
    args = {
        "num_rays": 100_000,
        "number_lights": 16,
        "grid_type": "ogrid",
        "mirrors": False
    }
    all_combinations = list(itertools.product(light_width, light_height, mirror_offset))

    df = pd.DataFrame(all_combinations, columns=params)
    combinations = df.values.tolist()
    combinations = [{k: v for k, v in zip(params, comb)} for comb in combinations]

    # run all combinations
    import os
    from multiprocessing import Pool
    from functools import partial
    with Pool(os.cpu_count() - 1) as p:
        output = p.map(partial(evaluate, **args), combinations)

    # save output
    df_output = pd.DataFrame(output, columns=["mean", "std"])
    df = pd.concat([df, df_output], axis=1)
    df.to_csv("combination_no_mirror.csv")


if __name__ == "__main__":
    main()
