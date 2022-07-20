import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def matrix_plot(df):
    dimensions = [dict(label=col, values=df[col]) for col in ["light_height","light_width","mirror_offset","inter_0","inter_1", "metric"]]
    fig = go.Figure(data=go.Splom(
                dimensions=dimensions,
                showupperhalf=False, # remove plots on diagonal
                diagonal=dict(visible=False),
                hoverinfo="all"
                # text=self.df['class'],
                # marker=dict(color=index_vals,
                #             showscale=False, # colors encode categorical variables
                #             line_color='white', line_width=0.5)
                ))
    fig.show()


def rotate_z(x: float, y: float, z: float, theta: float):
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z


def add_optimization_trace(fig, df, name:str, color):
    fig.add_trace(go.Scatter3d(x=df["light_height"], y=df["light_width"], z=df["mirror_offset"], name=name,
                               mode="markers+lines", marker=dict(color=color, size=7), line=dict(color=color)))
    fig.add_trace(go.Scatter3d(x=[df["light_height"][0]], y=[df["light_width"][0]], z=[df["mirror_offset"][0]],
                               mode="markers", marker=dict(color="red"), showlegend=False))
    best = df[df["metric"] == df["metric"].min()]
    fig.add_trace(go.Scatter3d(x=best["light_height"], y=best["light_width"], z=best["mirror_offset"],
                               mode="markers", marker=dict(color="green"), showlegend=False))


def main():
    df = pd.read_csv("Factorial_mirror.csv", index_col=0)
    cols = ["light_height","light_width","mirror_offset","inter_0","inter_1","inter_2", "inter_3", "metric"]

    df2 = pd.read_csv("nelder_mead_mirrors.csv", index_col=0)
    df3 = pd.read_csv("nelder_mead_mirrors2.csv", index_col=0)
    df4 = pd.read_csv("nelder_mead_mirrors3.csv", index_col=0)


    # df = df[df["metric"] < 0.101]
    df["metric"] = trunc(df["metric"].to_numpy(), 3)
    fig = px.scatter_3d(df, x=cols[0], y=cols[1], z=cols[2], color=cols[-1], hover_data=df.columns, ) # range_color=[0.1, 0.15]
    fig.update_traces(marker=dict(size=8, opacity=0.2))  #opacity=0.5

    df_best = df[df["metric"] < 0.098]
    fig.add_trace(go.Scatter3d(x=df_best["light_height"], y=df_best["light_width"], z=df_best["mirror_offset"],
                               mode="markers", marker=dict(color="black"), name="global min"))

    add_optimization_trace(fig, df2, "optimization 1", "lightseagreen")
    add_optimization_trace(fig, df3,"optimization 2", "cadetblue")
    add_optimization_trace(fig, df4, "optimization 3", "cyan")

    fig.update_layout(autosize=False, width=1000, height=800, font=dict(family="Arial", size=14, color="black"),
                  plot_bgcolor="white", showlegend=True, legend=dict(x=.1, y=.99),
                  scene=dict(xaxis_title="<b>light height (cm)</b>", yaxis_title="<b>light width (cm)</b>",
                             zaxis_title="<b>mirror offset (cm)</b>"), coloraxis_colorbar=dict(title="<b>std/mean</b>"))

    x, y, z = rotate_z(*(-1.25, 2, 1), np.pi/2)
    r = 0.8
    x= r*x
    y=r*y
    z=r*z
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.2),
        eye=dict(x=x, y=y, z=z)
    )
    fig.update_layout(scene_camera=camera)

    fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()
