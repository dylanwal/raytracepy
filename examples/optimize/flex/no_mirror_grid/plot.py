import pandas as pd
import plotly.graph_objs as go
import numpy as np

from raytracepy import merge_html_figs


def responce_surface(df, grid_type: str):
    df = df[df["grid_type"]==grid_type]

    x = df["light_height"].unique()
    y = df["light_width"].unique()
    x = np.sort(x)
    y = np.sort(y)
    z = np.empty((len(x), len(y)))
    for i in range(len(x)):
        for ii in range(len(x)):
            z[ii, i] = df["metric"].loc[(df["light_height"]==x[i]) & (df["light_width"]==y[ii])]

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z))
    best = df.loc[df["metric"].idxmin()]
    fig.add_trace(go.Scatter3d(x=[best["light_height"]], y=[best["light_width"]], z=[best["metric"] + 0.006],
                               mode="markers",
                               marker=dict(color="green"), name="global min"))
    print(f"{grid_type}: {best}")

    return fig



def main():
    df = pd.read_csv("Factorial_no_mirror_gird.csv", index_col=0)

    grids = df["grid_type"].unique()
    figs = [responce_surface(df, grid) for grid in grids]

    merge_html_figs(figs)

    #
    # fig.update_layout(autosize=False, width=1000, height=1000, font=dict(family="Arial", size=14, color="black"),
    #                   plot_bgcolor="white", showlegend=True, legend=dict(x=.1, y=.95),
    #                   scene=dict(xaxis_title="<b>light height (cm)</b>", yaxis_title="<b>light width (cm)</b>",
    #                              zaxis_title="<b>std/mean</b>"))
    #
    # fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()
