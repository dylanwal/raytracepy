
import numpy as np
import pandas as pd

import raytracepy as rpy


def run(reflect_func: int, height: float):
    # define lights
    grid = rpy.OffsetGridPattern(
        center=np.array([0, 0]),
        x_length=12.5,
        y_length=12.5,
        num_points=50)

    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.insert(xy_pos, 2, height),
                direction=np.array([0, 0, -1], dtype='float64'),
                num_traces=5,
                theta_func=1
            ))

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

    box_dim = 12.75
    box_height = height + 0.25
    mirror_left = rpy.Plane(
        name="mirror_left",
        position=np.array([-box_dim / 2, 0, box_height / 2], dtype='float64'),
        normal=np.array([1, 0, 0], dtype='float64'),
        length=box_dim,
        width=box_height,
        transmit_type="reflect",
        reflect_func=reflect_func,
    )
    mirror_right = rpy.Plane(
        name="mirror_right",
        position=np.array([box_dim / 2, 0, box_height / 2], dtype='float64'),
        normal=np.array([-1, 0, 0], dtype='float64'),
        length=box_dim,
        width=box_height,
        transmit_type="reflect",
        reflect_func=reflect_func,
    )
    mirror_front = rpy.Plane(
        name="mirror_front",
        position=np.array([0, -box_dim / 2, box_height / 2], dtype='float64'),
        normal=np.array([0, 1, 0], dtype='float64'),
        length=box_dim,
        width=box_height,
        transmit_type="reflect",
        reflect_func=reflect_func,
    )
    mirror_back = rpy.Plane(
        name="mirror_back",
        position=np.array([0, box_dim / 2, box_height / 2], dtype='float64'),
        normal=np.array([0, -1, 0], dtype='float64'),
        length=box_dim,
        width=box_height,
        transmit_type="reflect",
        reflect_func=reflect_func,
    )

    # Important note** ground should be last in list! RayTrace simulation evaluates ray hits in order of plane list
    planes = [mirror_left, mirror_right, mirror_front, mirror_back, ground]

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=planes,
        lights=lights,
        total_num_rays=5_000_000,
        bounce_max=20
    )
    sim.run()

    return sim


def pd_append(df: pd.DataFrame, series: pd.Series) -> pd.DataFrame | pd.Series:
    if df is None:
        return series
    else:
        return pd.concat([df, series], axis=1)


def main():
    reflect_func = [4, 7]
    heights = [1, 2, 3, 5, 10, 15]
    df = None
    for h in heights:
        for reflect in reflect_func:
            sim = run(reflect, h)
            sim.save_data(f"reflect{reflect}_h{h}")
            series = sim.planes["ground"].hit_stats_series()
            series["height"] = h
            series["reflect"] = reflect
            df = pd_append(df, series)

    df = df.T
    df.to_csv("mirror_data.csv")


if __name__ == "__main__":
    main()
