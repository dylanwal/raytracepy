import numpy as np
import plotly.graph_objs as go

import examples.plot_format as plot_format


def main():
    sim_data = np.array(
        [[0.0, 2351.0],
         [0.11134101434096388, 2301.0],
         [0.22409309230137087, 2314.0],
         [0.3398369094541218, 2253.0],
         [0.4605539916813224, 2183.0],
         [0.5890309702162738, 1961.0],
         [0.7297276562269663, 1847.0],
         [0.8911225078866528, 1546.0],
         [1.09491407713448, 1140.0],
         [1.5707963267948966, 0.0]]
    )
    expt_data = np.array(
        [[0, 2.372],
         [0.1428, 2.291],
         [0.2856, 2.29],
         [0.4284, 2.141],
         [0.5712, 1.938],
         [0.714, 1.652],
         [0.8568, 1.564],
         [0.9996, 1.166],
         [1.142, 0.9869],
         [1.285, 0.5511],
         [1.428, 0.2764],
         [1.571, 0.1156]
         ]
    )

    x = np.linspace(0, np.pi/2, 50)
    y = np.cos(x)

    colors = plot_format.get_plot_color(2)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", width=2, dash='dash'), name="<b>theory</b>"))
    fig.add_trace(
        go.Scatter(x=sim_data[:, 0], y=sim_data[:, 1]/np.max(sim_data[:, 1]), mode="markers",
                   marker=dict(color=colors[0], size=10), name="<b>simulation</b>"))
    fig.add_trace(
        go.Scatter(x=expt_data[:, 0], y=expt_data[:, 1]/np.max(expt_data[:, 1]), mode="markers",
                   marker=dict(color=colors[1], size=10), name="<b>experiment</b>"))

    plot_format.add_plot_format(fig, x_axis="<b>angle (rad)</b>", y_axis="<b>normalized light flux</b>")
    fig.update_layout(legend=dict(x=.75, y=.95))
    fig.write_html("cosine.html", auto_open=True, include_plotlyjs='cdn')


if __name__ == '__main__':
    main()
