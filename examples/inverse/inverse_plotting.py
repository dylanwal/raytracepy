"""
This generates figures after 'inverse_sim.py' and 'inverse_sim_processing.py' have been run.

"""

import numpy as np
import scipy.stats
import plotly.graph_objs as go

import raytracepy.theory as raypy_t
from raytracepy.utils.sig_figs import sig_figs

import examples.plot_format as plot_format


def lin_reg(fig: go.Figure, x: np.ndarray, y: np.ndarray, min_extend: float = 0, max_extend: float = 0, **kwargs):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    print(f"Linear regression results:\n\tslope: {slope}\n\tintercept: {intercept}\n\tR**2: {r_value**2}")

    min_ = np.min(x)
    max_ = np.max(x)
    span = max_ - min_
    x_plot = np.array([min_ - min_extend * span, max_ + max_extend * span])
    y_plot = slope * x_plot + intercept

    fig.add_trace(
        go.Scatter(
            x=x_plot, y=y_plot, mode="lines", line=dict(color="black", width=2, dash='dash'), showlegend=False, **kwargs
        )
    )
    fig.add_annotation(xref="x domain", yref="y domain", x=0.5, y=0.8,
                       text=f"y = {sig_figs(slope)} * x + {sig_figs(intercept)}<br>r<sup>2</sup>:"
                            f" {sig_figs(r_value**2, 4)}")


def main():
    # plot ln-ln
    # counts at center data printed out from 'inverse_sim_processing.py'
    sim = np.array([
        [1.0, 58201],  # height of light, counts at center
        [3.25, 5697],
        [5.5, 1996],
        [7.75, 967],
        [10.0, 630],
    ])

    ln_sim = np.log(sim)

    colors = plot_format.get_plot_color(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ln_sim[:, 0], y=ln_sim[:, 1], mode="markers", marker=dict(color=colors[0], size=10),
                  name=f"simulation"))
    lin_reg(fig, ln_sim[:, 0], ln_sim[:, 1])

    plot_format.add_plot_format(fig, x_axis="ln(distance)", y_axis="ln(intensity)")
    fig.write_html("ln_dependence.html", auto_open=True)

    ###################################################################################################################
    # rdf plots
    with open("rdf_inverse.csv", "r") as file:
        rdf = np.loadtxt(file, delimiter=",")
    colors = plot_format.get_plot_color(rdf.shape[1]-1)
    x_theory = np.linspace(0, rdf[-1, 0], 50)

    # plotting rdf
    max_ = np.max(rdf[:, 1:])
    max_theory = None
    fig = go.Figure()
    for i, (color, height) in enumerate(zip(colors, sim[:, 0])):
        y_theory = raypy_t.intensity_on_flat_surface(x_theory, 0, height)
        if max_theory is None:
            max_theory = np.max(y_theory)
        fig.add_trace(
            go.Scatter(x=x_theory, y=y_theory/max_theory, mode="lines", line=dict(color=color, width=3),
                       name=f"{height} cm ")  # (theory)
        )
        fig.add_trace(
            go.Scatter(
                x=rdf[:, 0], y=rdf[:, i+1]/max_,
                mode="lines", line=dict(color=color, width=3, dash='dash'), name=f"{height} cm (sim)", showlegend=False
                       )
        )

    plot_format.add_plot_format(fig, x_axis="distance (cm)", y_axis="relative light flux")

    fig.write_html("rdf.html", auto_open=True)

    # plotting rdf normalized
    fig = go.Figure()
    for i, (color, height) in enumerate(zip(colors, sim[:, 0])):
        y_theory = raypy_t.intensity_on_flat_surface(x_theory, 0, height)
        fig.add_trace(
            go.Scatter(x=x_theory, y=y_theory/np.max(y_theory), mode="lines", line=dict(color=color, width=3),
                       name=f"{height} cm ")  # (theory)
        )
        fig.add_trace(
            go.Scatter(
                x=rdf[:, 0], y=rdf[:, i+1]/np.max(rdf[:, i+1]),
                mode="lines", line=dict(color=color, width=3, dash='dash'), name=f"{height} cm (sim)", showlegend=False
                       )
        )

    plot_format.add_plot_format(fig, x_axis="distance (cm)", y_axis="normalized light flux")


if __name__ == '__main__':
    main()
