"""
For plotting convex hull of no_mirror, mirror, and diffusor
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
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

    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", legend={"x": 0.05, "y": 0.98})
    fig.update_xaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True, linewidth=5,
                     mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False, gridwidth=1,
                     gridcolor="lightgray")
    fig.update_yaxes(title="<b>std irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True, linewidth=5,
                     mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False, gridwidth=1,
                     gridcolor="lightgray", range=[0, 50])

    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()
