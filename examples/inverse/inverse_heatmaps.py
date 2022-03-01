"""
Heatmaps for Inverse Square Law

"""

import glob

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd
import datashader as ds

import raytracepy as rpy


def colorbar_positions(fig) -> list[list[float]]:
    layout = fig.layout

    pos_plot = []
    x_span = None
    y_span = None
    # get all x,y of each axis in figure
    for i in range(1, 100):  # limit is 100 subplots
        attr = f"axis{i}"
        if i == 1:
            attr = f"axis"
        if hasattr(layout, f"x{attr}"):
            xaxis = getattr(layout, f"x{attr}")
            yaxis = getattr(layout, f"y{attr}")
            pos_plot.append([xaxis.domain[1], yaxis.domain[1]])
            if i == 1:
                x_span = xaxis.domain[1] - xaxis.domain[0]
                y_span = yaxis.domain[1] - yaxis.domain[0]
        else:
            break

    # create colorbar positions from axis positions
    pos = []
    for xy in pos_plot:
        pos.append([xy[0], xy[1] - y_span / 2])

    return pos


def main():
    # load pickle files
    _dir = r".\*.pickle"
    sims = [rpy.RayTrace.load_data(file) for file in sorted(glob.glob(_dir))]
    print("data loaded")

    # heatmap plot
    rows: int = 3
    cols: int = 2
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=("h = 1 cm", "h = 3.25 cm", "h = 5.5 cm", "h = 7.75 cm", "h = 10 cm"),
        horizontal_spacing=0.25,
        vertical_spacing=0.1
    )
    colorbar_pos = colorbar_positions(fig)

    for i, sim in enumerate(sims):
        row_index = int(i / cols) + 1
        cols_index = i % cols + 1

        df = pd.DataFrame(sim.planes[0].hits[:, :2], columns=["x", "y"])
        canvas = ds.Canvas(plot_width=300, plot_height=300)
        agg = canvas.points(df, 'x', 'y')

        fig.add_trace(go.Heatmap(z=agg, colorbar=dict(len=1 / rows, x=colorbar_pos[i][0], y=colorbar_pos[i][1])),
                      row=row_index, col=cols_index)

        print(f"plot {i}/{len(sims) - 1} done")

    fig.update_layout(height=400 * rows, width=600 * cols, title_text="Heatmaps for Inverse Square Law")
    fig.write_html("rdf_heatmap.html", auto_open=True, include_plotlyjs='cdn')
    print("done")


if __name__ == '__main__':
    main()
