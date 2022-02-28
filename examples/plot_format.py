
import re

import numpy as np
import plotly.graph_objs as go

layout_figure = {
    "autosize": False,
    "width": 1000,
    "height": 600,
    "font": dict(family="Arial", size=24, color="black"),
    "plot_bgcolor": "white",
    "legend": {"x": 0.8, "y": 0.98}
}

layout_xaxis = {
    # "title": "<b>X<b>",
    "tickprefix": "<b>",
    "ticksuffix": "</b>",
    "showline": True,
    "linewidth": 5,
    "mirror": True,
    "linecolor": 'black',
    "ticks": "outside",
    "tickwidth": 4,
    "showgrid": False,
    "gridwidth": 1,
    "gridcolor": 'lightgray'
}

layout_yaxis = {
    # "title": "<b>Y<b>",
    "tickprefix": "<b>",
    "ticksuffix": "</b>",
    "showline": True,
    "linewidth": 5,
    "mirror": True,
    "linecolor": 'black',
    "ticks": "outside",
    "tickwidth": 4,
    "showgrid": False,
    "gridwidth": 1,
    "gridcolor": 'lightgray'
}

hex_options = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']


def get_plot_color(num_colors: int = 1) -> list[str]:
    """Gets color for 2D plots."""
    color_list = [
        'rgb(10,36,204)',  # blue
        'rgb(172,24,25)',  # red
        'rgb(6,127,16)',  # green
        'rgb(251,118,35)',  # orange
        'rgb(145,0,184)',  # purple
        'rgb(255,192,0)'  # yellow
    ]
    if num_colors <= 1:
        return [color_list[0]]
    if num_colors <= len(color_list):
        return color_list[:num_colors]
    else:
        num_colors_extra = num_colors - len(color_list)
        for i in range(num_colors_extra):
            color = ["#" + ''.join([np.random.choice(hex_options) for _ in range(6)])]
            color = [col.lstrip('#') for col in color]
            color = ["rgb" + str(tuple(int(col[i:i + 2], 16) for i in (0, 2, 4))) for col in color]
            color_list = color_list + color

        return color_list


def get_similar_color(color_in: str, num_colors: int, mode: str = "dark") -> list[str]:
    rgb = re.findall("[0-9]{1,3}", color_in)
    rgb = [int(i) for i in rgb]
    if mode == "dark":
        change_rgb = [i > 120 for i in rgb]
        jump_amount = [-int((i - 80) / num_colors) for i in rgb]
        jump_amount = [v if i else 0 for i, v in zip(change_rgb, jump_amount)]

    elif mode == "light":
        jump_amount = [int(100 / num_colors) if i < 100 else int((245 - i) / num_colors) for i in rgb]

    else:
        raise ValueError(f"Invalid 'mode'; only 'light' or 'dark'. (mode: {mode})")

    colors = []
    for i in range(num_colors):
        r = rgb[0] + jump_amount[0] * (i + 1)
        g = rgb[1] + jump_amount[1] * (i + 1)
        b = rgb[2] + jump_amount[2] * (i + 1)
        colors.append(f"rgb({r},{g},{b})")

    return colors


def add_plot_format(fig: go.Figure, x_axis: str, y_axis: str, layout_kwargs: dict = None,
                    x_kwargs: dict = None, y_kwargs: dict = None):
    """ Add default format to plot_add_on. """
    if layout_kwargs:
        layout_format = {**layout_figure, **layout_kwargs}
    else:
        layout_format = layout_figure
    fig.update_layout(layout_format)

    # x-axis
    x_axis_format = {"title": x_axis}
    if x_kwargs:
        x_axis_format = {**x_axis_format, **layout_xaxis, **x_kwargs}
    else:
        x_axis_format = {**x_axis_format, **layout_xaxis}
    fig.update_xaxes(x_axis_format)

    # y-axis
    y_axis_format = {"title": y_axis}
    if y_kwargs:
        y_axis_format = {**y_axis_format, **layout_yaxis, **y_kwargs}
    else:
        y_axis_format = {**y_axis_format, **layout_yaxis}
    fig.update_yaxes(y_axis_format)


def get_multi_y_axis(colors: list[str], fig: go.Figure, spread: float = 0.2) -> dict:
    y_axis_labels = {data.yaxis for data in fig.data}
    y_axis_labels = [label for label in y_axis_labels if label is not None]
    num_y_axis = len(y_axis_labels)

    gap = spread / (num_y_axis - 1)
    # first trace
    axis_format = {
        "xaxis": {"domain": [spread, 1]},  # 'tickformat': '%I:%M %p \n %b %d'
        # https://github.com/d3/d3-time-format/tree/v2.2.3#locale_format
        "yaxis1": {
            "title": {"text": f"{y_axis_labels[0]}", "standoff": 0},
            # "tickformat": ".4f",
            # "titlefont": {"color": colors[0]},
            # "tickfont": {"color": colors[0]},
            "tickangle": -45
        }
    }
    # the rest of the traces
    for i in range(1, num_y_axis):
        axis_format[f"yaxis{i + 1}"] = {
            "title": {"text": f"{y_axis_labels[i]}", "standoff": 0},
            # "tickformat": ".4f",
            # "titlefont": {"color": colors[i]},
            # "tickfont": {"color": colors[i]},
            "anchor": "free",
            "overlaying": "y",
            "side": "left",
            "position": spread - gap * i,
            "tickangle": -45
        }

    return axis_format


