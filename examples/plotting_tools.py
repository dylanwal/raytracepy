import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd
import datashader as ds


def colorbar_positions(fig) -> list[list[float]]:
    """ Gets colorbar positions for subplots"""
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


def heatmap_array(data: list[np.ndarray], plot_titles: list[str],
                  x_range: tuple[float] = None,
                  y_range: tuple[float] = None,
                  res: int = 300,
                  title: str = "heatmap",
                  file_name: str = "heatmap.html"):
    """

    Parameters
    ----------
    data: list[np.ndarray[:, 2]]
    plot_titles
    x_range
    y_range
    res
    title
    file_name

    Returns
    -------

    """
    # guard statements
    if len(data) != len(plot_titles):
        raise ValueError(f"len(sims) must equal len(plot_titles). sims: {len(data)}; plot_titles: {len(plot_titles)}")

    cols = 2
    rows = round(len(data) / 2 + 0.1)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"<b>{title}</b>" for title in plot_titles],
        horizontal_spacing=0.25,
        vertical_spacing=0.1 * 3/rows
    )
    colorbar_pos = colorbar_positions(fig)

    # set axis values
    if x_range is None:
        x = np.linspace(0, res-1, res)
    else:
        x = np.linspace(x_range[0], x_range[1], res)
    if y_range is None:
        y = np.linspace(0, res-1, res)
    else:
        y = np.linspace(y_range[0], y_range[1], res)

    # main loop
    for i, datum in enumerate(data):
        # get index for current subplot
        row_index = int(i / cols) + 1
        cols_index = i % cols + 1

        # create heatmap
        df = pd.DataFrame(datum, columns=["x", "y"])
        canvas = ds.Canvas(plot_width=res, plot_height=res)
        agg = canvas.points(df, 'x', 'y')
        fig.add_trace(go.Heatmap(x=x, y=y, z=agg,
                                 colorbar=dict(len=1 / rows, x=colorbar_pos[i][0], y=colorbar_pos[i][1],
                                               title="count")),
                      row=row_index, col=cols_index)

        # set axis values
        fig.update_xaxes(title="<b>x (cm)</b>", row=row_index, col=cols_index, range=[-10, 10])
        fig.update_yaxes(title="<b>y (cm)</b>", row=row_index, col=cols_index, scaleanchor=f"x{i+1}")

    # final formatting and save
    fig.update_layout(height=400 * rows, width=600 * cols, title_text=title, plot_bgcolor="white",
                      font=dict(family="Arial", color="black"))
    if not file_name.endswith(".html"):
        file_name += ".html"

    fig.write_html(file_name, auto_open=True, include_plotlyjs='cdn')

    return fig
