import numpy as np
import plotly.graph_objs as go

import examples.plot_format


def get_point_source(distance: np.ndarray) -> np.ndarray:
    out = np.zeros_like(distance)
    for i in range(out.size):
        out[i] = 1 / (distance[i] ** 2)
    return out


def get_circle_source(distance: np.ndarray, r: float = 1) -> np.ndarray:
    out = np.zeros_like(distance)
    for i in range(out.size):
        out[i] = r**2 / (r**2 + distance[i] ** 2)
    return out


def add_shade_box(fig: go.Figure, xy0: (list, tuple) = (0, 0), xy1: (list, tuple) = (1, 1), color: str = "gray"):
    x = [xy0[0], xy1[0], xy1[0], xy0[0], xy0[0]]
    y = [xy0[1], xy0[1], xy1[1], xy1[1], xy0[1]]
    fig.add_trace(go.Scatter(x=x, y=y, fill="toself", showlegend=False, line=dict(dash="dot", color=color)))


def main():
    r = 6
    hit_max = 216  # found by fitting exponential to hits vs distance and this is the value at zero distance
    sim_distance = np.array([1, 2, 3, 5, 10, 15])
    sim_hits = np.array([195, 169, 146, 108, 55, 31])

    n = 100
    distance = np.logspace(-1.5, 1.5, n)
    plane = np.ones_like(distance)
    point = get_point_source(distance)
    circle = get_circle_source(distance)

    # plotting
    colors = examples.plot_format.get_plot_color(4)
    fig = go.Figure()
    add_shade_box(fig, xy0=[0.01, 0.001], xy1=[0.1, 10], color=colors[0])
    add_shade_box(fig, xy0=[10, 0.001], xy1=[100, 10], color=colors[1])
    fig.add_trace(go.Scatter(x=distance, y=plane, mode="lines", line=dict(color="black"), showlegend=False))
    fig.add_trace(go.Scatter(x=distance, y=point, mode="lines", line=dict(color="black"), showlegend=False))
    fig.add_trace(go.Scatter(x=distance, y=circle, mode="lines", line=dict(color="gray"), showlegend=False))

    fig.add_trace(go.Scatter(x=sim_distance/r, y=sim_hits/hit_max, mode="markers", line=dict(color=colors[3]),
                  name="<b>simulation<b>"))

    fig.add_annotation(x=-1.06, y=0, text="<b>area light source<b>", yshift=10, showarrow=False)
    fig.add_annotation(x=0.78, y=-1.5, text="<b>point light source<b>", yshift=10, textangle=58, showarrow=False)
    fig.add_annotation(x=np.log10(distance[int(distance.size/2.2)]), y=np.log10(circle[int(circle.size/2.2)]),
                       ax=-.2, ay=-.7, axref="x", ayref="y", arrowwidth=3, arrowcolor="black",
                       text="<b>circle light<br>source<b>",  showarrow=True, arrowhead=2)
    fig.add_annotation(x=1, y=-0.8, text="<b>d/r > 10<br>inverse square law error < 1 %<b>", showarrow=False,
                       textangle=90, font=dict(size=12))
    fig.add_annotation(x=-1, y=-0.8, text="<b>d/r < 0.1<br>area source error < 1 %<b>", showarrow=False,
                   textangle=90, font=dict(size=12))


    # formatting
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.55, y=.95))
    fig.update_xaxes(title="<b>distance from light / (radius of light source)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", type="log", range=[-1.5, 1.5])
    fig.update_yaxes(title="<b>mean irradiance / (max mean irradiance)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", type="log", range=[-2, 0.5])

    fig.show()


if __name__ == "__main__":
    main()
