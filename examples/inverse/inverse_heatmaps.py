"""
Heatmaps for Inverse Square Law

"""
import glob

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import datashader as ds
import raytracepy as rpy

import examples.fig_to_grid


def main():
    # load pickle files
    _dir = r".\*.pickle"
    sims = [rpy.RayTrace.load_data(file) for file in sorted(glob.glob(_dir))]
    print("data loaded")

    # heatmap plot
    res = 300
    plot_titles = ["h = 1 cm", "h = 4 cm", "h = 7 cm", "h = 10 cm", "h = 13 cm", "h = 16 cm"]
    x_range = (-10, 10)
    y_range = (-10, 10)
    title = "Heatmaps for Inverse Square Law"
    file_name: str = "inverse_heatmap.html"

    data = [sim.planes[0].hits[:, :2] for sim in sims]

    # heatmap_array(data, plot_titles, x_range, y_range, res, title, file_name)

    x = np.linspace(x_range[0], x_range[1], res)
    y = np.linspace(y_range[0], y_range[1], res)
    # create heatmap
    figs = []
    for datum in data:
        df = pd.DataFrame(datum, columns=["x", "y"])
        canvas = ds.Canvas(plot_width=res, plot_height=res)
        agg = canvas.points(df, 'x', 'y')
        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=x, y=y, z=agg, name="count",
                                 colorbar=dict(title="<b>count</b>", tickfont=dict())))

        # formatting
        fig.update_layout(autosize=False, width=600, height=580, font=dict(family="Arial", size=18, color="black"),
                          plot_bgcolor="white")
        fig.update_xaxes(title="<b>x (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                         linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                         gridwidth=1, gridcolor="lightgray", range=[-10, 10])
        fig.update_yaxes(title="<b>y (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                         linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                         gridwidth=1, gridcolor="lightgray", scaleanchor="x")

        figs.append(fig)
        # fig.show()

    examples.fig_to_grid.figs_to_grid_png(figs, shape=(2, 3))

    print("done")


if __name__ == '__main__':
    main()
