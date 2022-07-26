"""
For plotting convex hull of no_mirror, mirror, and diffusor
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import scipy.spatial

from examples import plot_format


def plot_pareto_fronts(df: pd.DataFrame, fig: go.Figure = None, color: str = None, name: str = None) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    # calculate convexhull
    points = df[["mean", "std"]].to_numpy()
    grid_hull = scipy.spatial.ConvexHull(points)
    hull_xy = np.array([points[grid_hull.vertices, 0], points[grid_hull.vertices, 1]])
    hull_xy = np.concatenate((hull_xy, hull_xy[:, 0].reshape((2, 1))), axis=1)

    kwargs = {"fill": "toself"}
    if color is not None:
        kwargs["fillcolor"] = plot_format.rgb_add_opacity(color, 0.1)
        kwargs["line"] = {"color": color}
    if name is not None:
        kwargs["name"] = f"<b>{name}</b>"
    fig.add_trace(go.Scatter(x=hull_xy[0, :], y=hull_xy[1, :], mode="lines", **kwargs))

    return fig


def main():
    # load data
    df = pd.read_csv(r"no_mirror\combinations.csv", index_col=0)
    df_mirror = pd.read_csv(r"mirror\combinations.csv", index_col=0)
    df_diffuser = pd.read_csv(r"diffuser\combinations.csv", index_col=0)

    colors = plot_format.get_plot_color(3)
    fig = plot_pareto_fronts(df, color=colors.pop(), name="no mirror")
    fig = plot_pareto_fronts(df_diffuser, fig, color=colors.pop(), name="diffuser")
    fig = plot_pareto_fronts(df_mirror, fig, color=colors.pop(), name="mirror")

    fig.update_layout(autosize=False, width=int(800*.7), height=int(600*.7), font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", legend={"x": 0.68, "y": 0.02, "bgcolor":"rgba(0,0,0,0)"})
    fig.update_xaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True, linewidth=5,
                     mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False, gridwidth=1,
                     gridcolor="lightgray", range=[0, 650])
    fig.update_yaxes(title="<b>std irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True, linewidth=5,
                     mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False, gridwidth=1,
                     gridcolor="lightgray", range=[0, 50])

    fig.write_html("temp.html", auto_open=True)


def main2():
    # load data
    df = pd.read_csv(r"mirror\combinations.csv", index_col=0)

    # select data
    df = df.loc[(df["height"] == 5) & (df["width"] == 12.5) & (df["number_lights"] == 49) & (df["grid_type"] == "ogrid")]
    df["mean"] = df["mean"]

    # plot data
    colors = plot_format.get_plot_color(3)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["mirror_offset"], y=df["mean"], mode="lines+markers", name="<b>mean</b>",
                             marker=dict(color=colors[0], size=10), line=dict(color=colors[0])))
    fig.add_trace(go.Scatter(x=df["mirror_offset"], y=df["std"], mode="lines+markers", name="<b>std</b>",
                             marker=dict(color=colors[1], size=10), line=dict(color=colors[1]), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["mirror_offset"], y=df["std"]/df["mean"], mode="lines+markers", name="<b>std/mean</b>",
                             marker=dict(color=colors[2], size=10), line=dict(color=colors[2]), yaxis="y3"))

    # add plot formatting
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.4, y=.95))
    fig.update_xaxes(title="<b>mirror offset (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
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
    fig.show()


def main3():
    # load data
    df = pd.read_csv(r"diffuser\combinations.csv", index_col=0)

    # select data
    df = df.loc[(df["height"] == 5) & (df["width"] == 12.5) & (df["number_lights"] == 49) & (df["grid_type"] == "ogrid")]
    df["mean"] = df["mean"]

    # plot data
    colors = plot_format.get_plot_color(3)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["diffuser_height"], y=df["mean"], mode="lines+markers", name="<b>mean</b>",
                             marker=dict(color=colors[0], size=10), line=dict(color=colors[0])))
    fig.add_trace(go.Scatter(x=df["diffuser_height"], y=df["std"], mode="lines+markers", name="<b>std</b>",
                             marker=dict(color=colors[1], size=10), line=dict(color=colors[1]), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["diffuser_height"], y=df["std"]/df["mean"], mode="lines+markers", name="<b>std/mean</b>",
                             marker=dict(color=colors[2], size=10), line=dict(color=colors[2]), yaxis="y3"))

    # add plot formatting
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.4, y=.95))
    fig.update_xaxes(title="<b>diffuser height</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
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
    fig.show()


def main4():
    # load data
    df = pd.read_csv(r"mirror\combinations.csv", index_col=0)

    # select data
    df = df.loc[(df["mirror_offset"] == 1) & (df["width"] == 12.5) & (df["number_lights"] == 49) & (df["grid_type"] == "ogrid")]
    df["mean"] = df["mean"]

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
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.4, y=.95))
    fig.update_xaxes(title="<b>height (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
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
    fig.show()


if __name__ == "__main__":
    main()
