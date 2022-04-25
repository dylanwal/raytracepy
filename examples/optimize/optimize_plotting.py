import os
import pickle
import itertools
from argparse import Namespace

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.spatial
from plotly.subplots import make_subplots

from examples import results_html, plot_format


def add_pareto_front_to_df(raw_results: Namespace, df: pd.DataFrame) -> pd.DataFrame:
    """ Adds a 'pareto' True/False row to dataframe. """
    pareto_points = [list(itertools.chain(*sub)) for sub in raw_results.curr_true_pareto_points[-1]]
    pareto_vals = raw_results.curr_true_pareto_vals[-1]
    pareto = [points+list(vals) for points, vals in zip(pareto_points, pareto_vals)]

    pareto_list = []
    for row in df.iterrows():
        row = row[1].to_list()
        if row in pareto:
            pareto_list.append(True)
            pareto.remove(row)
        else:
            pareto_list.append(False)

    df["pareto"] = pareto_list
    return df


def add_initial_df(raw_results: Namespace, df: pd.DataFrame) -> pd.DataFrame:
    """ Adds a 'initial' True/False row to dataframe. """
    initial_list = [True] * raw_results.init_expts
    initial_list += [False] * (len(df) - raw_results.init_expts)

    df["initial"] = initial_list
    return df


def extract_results(raw_results: Namespace) -> pd.DataFrame:
    all_points = [list(itertools.chain(*sub)) for sub in raw_results.query_points]
    all_vals = raw_results.query_vals

    df = pd.DataFrame([points+list(vals) for points, vals in zip(all_points, all_vals)],
                      columns=raw_results.domain_ordering+raw_results.value_ordering)
    df.index.name = "expt"

    df = add_pareto_front_to_df(raw_results, df)
    df = add_initial_df(raw_results, df)

    return df


def plot_optimization_expt(df: pd.DataFrame) -> go.Figure:
    # plot data
    colors = plot_format.get_plot_color(2)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["mean"], mode="lines+markers", name="<b>mean</b>",
                             marker=dict(color=colors[0], size=10), line=dict(color=colors[0])),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["std"], mode="lines+markers", name="<b>std</b>",
                             marker=dict(color=colors[1], size=10), line=dict(color=colors[1])),
                  secondary_y=True)

    # add plot formatting
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.75, y=.95))
    fig.update_xaxes(title="<b>width of light pattern (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>std irradiance<b>", secondary_y=True)

    return fig


def plot_convex_hull(df: pd.DataFrame, df_compare: pd.DataFrame = None) -> go.Figure:
    colors = plot_format.get_plot_color(3)
    fig = go.Figure()

    # calculate convexhull for comparison
    if df_compare is not None:
        hull_color = "rgb(148,148,148)"
        points = df_compare.loc[df_compare["number_lights"] == 49][["mean", "std"]].to_numpy()
        grid_hull = scipy.spatial.ConvexHull(points)
        hull_xy = np.array([points[grid_hull.vertices, 0], points[grid_hull.vertices, 1]])
        hull_xy = np.concatenate((hull_xy, hull_xy[:, 0].reshape((2, 1))), axis=1)
        kwargs = {"fill": "toself", "fillcolor": plot_format.rgb_add_opacity(hull_color, 0.1),
                  "line": {"color": hull_color}, "name": f"<b>combination convex hull</b>"}
        fig.add_trace(go.Scatter(x=hull_xy[0, :], y=hull_xy[1, :], mode="lines", **kwargs))

    fig.add_trace(
        go.Scatter(x=df["mean"], y=100 - df["std"], mode="markers+text", marker=dict(color=colors[0]),
                   name="<b>expts</b>", text=[f"<b>{i}</b>" for i in df.index], textfont=dict(size=12, family="Arial"),
                    textposition="top center")
    )
    df_initial = df.loc[df["initial"] == True]
    fig.add_trace(
        go.Scatter(x=df_initial["mean"], y=100 - df_initial["std"], mode="markers", marker=dict(color=colors[1]),
                   name="<b>initial expts<b>")
    )

    df = df.loc[df["pareto"] == True]
    df = df.sort_values(by=["mean"], inplace=False)
    fig.add_trace(
        go.Scatter(x=df["mean"], y=100 - df["std"], mode="lines", line=dict(color=colors[2]),
                   name="<b>pareto front</b>")
    )

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


def main():
    # Load data
    with open("results_2022_04_24-09_32_17_PM.pickle", 'rb') as file:
        raw_results = pickle.load(file)
    df = extract_results(raw_results)
    dir_ = os.getcwd()
    df_mirror = pd.read_csv("\\".join(dir_.split("\\")[:-1]) + "\\array\\combinations\\mirror\\combinations.csv", index_col=0)

    figs = [plot_optimization_expt(df), plot_convex_hull(df, df_mirror)]
    results_html.merge_html_figs(figs, "results.html", auto_open=True)

    print("hi")


if __name__ == "__main__":
    main()
