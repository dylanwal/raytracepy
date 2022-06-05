
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


def main():
    df = pd.read_csv("nsga2_data.csv")
    df["mean"] = -1 * df["mean"]

    df2 = pd.read_csv("nsga2_results.csv")
    df2["mean"] = -1 * df2["mean"]
    df2 = df2.sort_values("mean")

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=df["mean"], y=df["std"], z=df["cost"], name="final", mode="markers+lines",
                             line=dict(width=1)))

    # plot_df(fig, df, x="mean", y="std", color="step")

    fig.write_html("temp.html", auto_open=True)
    print(df)


if __name__ == "__main__":
    main()