
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from examples.plot_format import get_plot_color, get_similar_color


def plot_df(fig: go.Figure, df, x: str, y: str, color: str):
    values = df[color].unique()
    colors = get_plot_color(len(values))
    for val, col in zip(values, colors):
        df_plot = df.loc[df[color] == val].sort_values(x)
        fig.add_trace(go.Scatter(
            x=df_plot[x], y=df_plot[y], mode="markers+lines", marker=dict(color=col), line=dict(color=col, width=1)
        ))


def plot_progress(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["mean"], mode="markers+lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["std"], mode="markers+lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=-(df["std"]-df["mean"]), mode="markers+lines", line=dict(width=1)))
    fig.show()


def main():
    df = pd.read_csv("scipy_data.csv")

    df_combinations = pd.read_csv("combination.csv")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_combinations["mean"], y=df_combinations["std"], name="combinatorial",
                             mode="markers"))
    fig.add_trace(go.Scatter(x=df["mean"], y=df["std"], name="final", mode="markers+lines", line=dict(width=1)))

    plot_progress(df)

    fig.show()
    print(df)


if __name__ == "__main__":
    main()
