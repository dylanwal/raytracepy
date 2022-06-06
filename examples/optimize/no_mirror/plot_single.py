import pandas as pd
import plotly.graph_objs as go
import numpy as np


def responce_surface():
    df = pd.read_csv("Factorial_no_mirror.csv", index_col=0)

    x = df["light_height"].unique()
    y = df["light_width"].unique()
    x = np.sort(x)
    y = np.sort(y)
    z = np.empty((len(x), len(y)))
    for i in range(len(x)):
        for ii in range(len(x)):
            z[ii, i] = df["metric"].loc[(df["light_height"]==x[i]) & (df["light_width"]==y[ii])]

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorbar=dict(title="std/mean")))


    return fig, df



def main():
    df = pd.read_csv("nelder_mead/MethodNelderMead_no_mirror_0.csv", index_col=0)

    offset = 0.005
    fig, df_full = responce_surface()
    fig.add_trace(go.Scatter3d(x=df["light_height"], y=df["light_width"], z=df["metric"] + offset, name="optimization path",
                               mode="lines+markers", marker=dict(color="yellow", size=4), line=dict(color="yellow",
                                                                                                    width=2)))
    start = df.iloc[0]
    fig.add_trace(go.Scatter3d(x=[start["light_height"]], y=[start["light_width"]], z=[start["metric"] + offset], mode="markers",
                               marker=dict(color="cyan", size=6), name="start"))
    best = df.loc[df["metric"].idxmin()]
    fig.add_trace(go.Scatter3d(x=[best["light_height"]], y=[best["light_width"]], z=[best["metric"] + offset], mode="markers",
                               marker=dict(color="red", size=6), name="best"))

    best = df_full.loc[df_full["metric"].idxmin()]
    fig.add_trace(go.Scatter3d(x=[best["light_height"]], y=[best["light_width"]], z=[best["metric"]+0.006],
                               mode="markers",
                               marker=dict(color="green"), name="global min"))

    fig.update_layout(autosize=False, width=1000, height=1000, font=dict(family="Arial", size=14, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.1, y=.95),
                      scene=dict(xaxis_title="<b>light height (cm)</b>", yaxis_title="<b>light width (cm)</b>",
                                 zaxis_title="<b>std/mean</b>"))

    camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=-0.2),
    eye=dict(x=1.25, y=1.25, z=1.25)
    )
    fig.update_layout(scene_camera=camera)

    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()
