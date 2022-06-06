import multiprocessing
from functools import partial

import pandas as pd
import raytracepy as rpy
from setup_simulation import simulation

rpy.config.single_warning = True


def metric(args):
    mean_ = args[0]
    std = args[1]
    print("metric")
    return std/mean_


def main():
    param = [8, 10]
    kwargs=dict(
        num_rays=100_000,
        number_lights=16,
        mirrors=False,
        grid_type='ogrid',
        param_names=["light_height", "light_width"])
    sim = partial(simulation, **kwargs)

    param_list = [param] * 10

    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        data = pool.map(sim, param_list)

    df = pd.DataFrame(data)
    df.columns=["mean", "std", "p10", "p90"]
    df["std/mean"] = df["std"]/df["mean"]

    print(df["std/mean"].std())


if __name__ == '__main__':
    main()
