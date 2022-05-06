"""

"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import scipy.spatial
from plotly.subplots import make_subplots

from examples import results_html, plot_format

colors_ = plot_format.get_plot_color(4)
grid_colors = {
    "circle": colors_[0],
    "ogrid": colors_[1],
    "grid": colors_[2],
    "spiral": colors_[3]
}


def _convex_hull(fig, df, grid):
    points = df.loc[df["grid_type"] == grid][["mean", "std"]].to_numpy()
    grid_hull = scipy.spatial.ConvexHull(points)
    hull_xy = np.array([points[grid_hull.vertices, 0], points[grid_hull.vertices, 1]])
    hull_xy = np.concatenate((hull_xy, hull_xy[:, 0].reshape((2, 1))), axis=1)
    color = plot_format.rgb_add_opacity(grid_colors[grid], 0.1)  # add opacity
    fig.add_trace(go.Scatter(x=hull_xy[0, :], y=hull_xy[1, :], mode="lines",
                             fill="toself", name=f"<b>{grid}</b>", fillcolor=color))


def plot_grid_pareto_fronts(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    grids = ['circle', 'ogrid', 'grid', 'spiral']
    for grid in grids:
        _convex_hull(fig, df, grid)

    # add plot formatting
    fig.update_layout(autosize=False, width=780, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.05, y=.95))
    fig.update_xaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>std irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")

    return fig


def _convex_hull_norm(fig, df, grid):
    df = df.loc[df["grid_type"] == grid]
    if grid == "ogrid":
        adjust = [[4, 3], [16, 14], [36, 33], [49, 46], [81, 77]]
        for i in adjust:
            df["number_lights"] = df["number_lights"].replace([i[0]], i[1])
            # row = df.loc[df["number_lights"] == i[0]]
            # df[row]["number_lights"] = i[1]

    df["mean"] = df["mean"] / df["number_lights"]
    points = df[["mean", "std"]].to_numpy()

    grid_hull = scipy.spatial.ConvexHull(points)
    hull_xy = np.array([points[grid_hull.vertices, 0], points[grid_hull.vertices, 1]])
    hull_xy = np.concatenate((hull_xy, hull_xy[:, 0].reshape((2, 1))), axis=1)
    color = plot_format.rgb_add_opacity(grid_colors[grid], 0.1)  # add opacity
    fig.add_trace(go.Scatter(x=hull_xy[0, :], y=hull_xy[1, :], mode="lines",
                             fill="toself", name=f"<b>{grid}</b>", fillcolor=color))


def plot_grid_pareto_fronts_norm(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    grids = ['circle', 'ogrid', 'grid', 'spiral']
    for grid in grids:
        _convex_hull_norm(fig, df, grid)

    # add plot formatting
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.05, y=.95))
    fig.update_xaxes(title="<b>normalized mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>std irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    return fig


def plot_num_lights(df: pd.DataFrame) -> go.Figure:
    # select data
    df = df.loc[(df["height"] == 5) & (df["width"] == 12.5) & (df["grid_type"] == "ogrid")]
    df["mean"] = df["mean"]

    # plot data
    colors = plot_format.get_plot_color(3)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["number_lights"], y=df["mean"], mode="lines+markers", name="<b>mean</b>",
                             marker=dict(color=colors[0], size=10), line=dict(color=colors[0])))
    fig.add_trace(go.Scatter(x=df["number_lights"], y=df["std"], mode="lines+markers", name="<b>std</b>",
                             marker=dict(color=colors[1], size=10), line=dict(color=colors[1]), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["number_lights"], y=df["std"]/df["mean"], mode="lines+markers", name="<b>std/mean</b>",
                             marker=dict(color=colors[2], size=10), line=dict(color=colors[2]), yaxis="y3"))

    # add plot formatting
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.4, y=.95))
    fig.update_xaxes(title="<b>number of lights</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", domain=[0.03, 0.88])
    fig.update_yaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_layout(yaxis2=dict(title="<b>std irradiance<b>", range=[0, 40], anchor="x",
                                 overlaying="y", side="right"))
    fig.update_layout(yaxis3=dict(title="<b>std/mean<b>", tickprefix="<b>", ticksuffix="</b>", range=[0, 0.25],
                                  anchor="free", overlaying="y", side="right", position=1,
                                  linecolor="black", linewidth=5
                                  ))

    return fig


def plot_height(df: pd.DataFrame) -> go.Figure:
    # select data
    df = df.loc[(df["number_lights"] == 49) & (df["width"] == 12.5) & (df["grid_type"] == "ogrid")]

    # plot data
    colors = plot_format.get_plot_color(3)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["height"], y=df["mean"], mode="lines+markers", name="<b>mean</b>",
                             marker=dict(color=colors[0], size=10), line=dict(color=colors[0])))
    fig.add_trace(go.Scatter(x=df["height"], y=df["std"], mode="lines+markers", name="<b>std</b>",
                             marker=dict(color=colors[1], size=10), line=dict(color=colors[1]), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["height"], y=df["std"]/df["mean"], mode="lines+markers", name="<b>std/mean</b>",
                             marker=dict(color=colors[2], size=10), line=dict(color=colors[2]), yaxis="y3"))

    # add plot formatting
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.3, y=.95))
    fig.update_xaxes(title="<b>height of lights (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", domain=[0.03, 0.88], range=[0, 16])
    fig.update_yaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_layout(yaxis2=dict(title="<b>std irradiance<b>", range=[0, 40], anchor="x",
                                 overlaying="y", side="right"))
    fig.update_layout(yaxis3=dict(title="<b>std/mean<b>", tickprefix="<b>", ticksuffix="</b>", range=[0, 0.25],
                                  anchor="free", overlaying="y", side="right", position=1,
                                  linecolor="black", linewidth=5
                                  ))

    return fig


def plot_width(df: pd.DataFrame) -> go.Figure:
    # select data
    df = df.loc[(df["number_lights"] == 49) & (df["height"] == 5) & (df["grid_type"] == "ogrid")]

    # plot data
    colors = plot_format.get_plot_color(3)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["width"], y=df["mean"], mode="lines+markers", name="<b>mean</b>",
                             marker=dict(color=colors[0], size=10), line=dict(color=colors[0])),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df["width"], y=df["std"], mode="lines+markers", name="<b>std</b>",
                             marker=dict(color=colors[1], size=10), line=dict(color=colors[1]), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["width"], y=df["std"]/df["mean"], mode="lines+markers", name="<b>std/mean</b>",
                         marker=dict(color=colors[2], size=10), line=dict(color=colors[2]), yaxis="y3"))

    # add plot formatting
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.5, y=.95))
    fig.update_xaxes(title="<b>width of light pattern (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", domain=[0.03, 0.88])
    fig.update_yaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_layout(yaxis2=dict(title="<b>std irradiance<b>", range=[0, 40], anchor="x",
                                 overlaying="y", side="right"))
    fig.update_layout(yaxis3=dict(title="<b>std/mean<b>", tickprefix="<b>", ticksuffix="</b>", range=[0, 0.25],
                                  anchor="free", overlaying="y", side="right", position=1,
                                  linecolor="black", linewidth=5
                                  ))

    return fig


def main():
    # load data
    df = pd.read_csv("no_mirror/combinations.csv", index_col=0)

    # add color column
    new_col = []
    for grid_type in df["grid_type"]:
        new_col.append(grid_colors[grid_type])
    df['colors'] = pd.Series(new_col, index=df.index)

    # figures
    fig_grid = ff.create_scatterplotmatrix(df, diag='box', index='colors',
                                           colormap_type='cat',
                                           height=3000, width=3000
                                           )

    fig_grid_type = plot_grid_pareto_fronts(df)
    fig_grid_type_norm = plot_grid_pareto_fronts_norm(df)
    fig_num = plot_num_lights(df)
    fig_height = plot_height(df)
    fig_width = plot_width(df)

    results_html.merge_html_figs([fig_grid_type, fig_grid_type_norm, fig_num, fig_height, fig_width], "results.html",
                                 auto_open=True)


if __name__ == "__main__":
    main()
