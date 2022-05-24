import glob

import pandas as pd

import flex_optimization as fo


def main():
    pattern = "*.csv"
    files = glob.glob(pattern)

    data = pd.read_csv(files[0], index_col=0)
    data_set = [data]
    for file in files[1:]:
        data_temp = pd.read_csv(file, index_col=0)
        data_set.append(data_temp)
        data = pd.concat((data, data_temp))

    print(data)

    problem = fo.Problem(
        func=lambda x: 0,
        variables=[
            fo.ContinuousVariable(name="light_height", min_=1, max_=15),
            fo.ContinuousVariable(name="light_width", min_=5, max_=15),
            fo.ContinuousVariable(name="mirror_offset", min_=0.1, max_=10),
            # fo.DiscreteVariable(name='grid_type', items=['circle', 'ogrid', 'grid', 'spiral'])
        ],
        kwargs=dict(
            num_rays=100_000,
            number_lights=16,
            mirrors=True,
            grid_type='ogrid',
            param_names=["light_height", "light_width", "mirror_offset"]
        ),
    )

    vis = fo.OptimizationVis(problem, data)
    fig = vis.plot_4d_vis()
    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()
