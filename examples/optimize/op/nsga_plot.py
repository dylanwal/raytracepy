
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

    df_rand = pd.read_csv("random2.csv")
    df_rand = df_rand.sort_values("mean")

    df_dragon = pd.read_csv("dragon3.csv")
    df_dragon["mean"] = -1 * df_dragon["mean"]
    df_dragon = df_dragon.sort_values("mean")

    # df_combinations = pd.read_csv(r"C:\Users\nicep\Desktop\pyth_proj\raytracepy\examples\array\combinations\mirror\combinations.csv")
    df_combinations = pd.read_csv("combination.csv")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_combinations["mean"], y=df_combinations["std"], name="combinatorial",
                             mode="markers"))
    fig.add_trace(go.Scatter(x=df2["mean"], y=df2["std"], name="final", mode="markers+lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df_rand["mean"], y=df_rand["std"], name="random", mode="markers+lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df_dragon["mean"], y=df_dragon["std"], name="dragon", mode="markers+lines", line=dict(width=1)))

    plot_df(fig, df, x="mean", y="std", color="step")

    fig.show()
    print(df)


if __name__ == "__main__":
    main()
