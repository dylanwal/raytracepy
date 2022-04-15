"""
This generates figures after 'inverse_sim.py' and 'inverse_sim_processing.py' have been run.

"""

import numpy as np
import scipy.stats
import plotly.graph_objs as go

import raytracepy.theory as raypy_t
from raytracepy.utils.sig_figs import sig_figs

import examples.plot_format as plot_format


def lin_reg(fig: go.Figure, x: np.ndarray, y: np.ndarray, min_extend: float = 0, max_extend: float = 0,
         color: str = "black", x_eq: float = 0.5, y_eq: float = 0.8, **kwargs):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    print(f"Linear regression results:\n\tslope: {slope}\n\tintercept: {intercept}\n\tR**2: {r_value ** 2}")

    min_ = np.min(x)
    max_ = np.max(x)
    span = max_ - min_
    x_plot = np.array([min_ - min_extend * span, max_ + max_extend * span])
    y_plot = slope * x_plot + intercept

    fig.add_trace(
        go.Scatter(
            x=x_plot, y=y_plot, mode="lines", line=dict(color=color, width=2, dash='dash'), showlegend=False, **kwargs
        )
    )
    fig.add_annotation(xref="x domain", yref="y domain", x=x_eq, y=y_eq,
                       text=f"<b>y = {sig_figs(slope)} * x + {sig_figs(intercept)}<br>r<sup>2</sup>:"
                            f" {sig_figs(r_value ** 2, 4)}</b>",
                       font=dict(
                           color=color,
                           size=22,
                       )
                       )


def main():
    # plot ln-ln
    # counts at center data printed out from 'inverse_sim_processing.py'
    sim = np.array([
        [1.0, 24631],  # height of light (cm), counts at center radius of 0.1 cm (bigger cause errors, small too few
        # counts)
        [4.0, 1524],
        [7.0, 522],
        [10.0, 250],
        [13.0, 152],
        [16.0, 91],
    ])

    expt = np.array([
        [3.5, 20.7],  # height of light (cm), watts/m^2
        [4, 13.5],
        [4.5, 11],
        [5, 9.2],
        [5.5, 7.86],
        [6, 6.4],
        [7, 4.85],
        [8, 3.6],
        [10, 2.3],
        [12, 1.63]
    ])

    # normalize to point in common (10 cm)
    sim[:, 1] = sim[:, 1] / sim[3, 1]
    expt[:, 1] = expt[:, 1] / expt[-2, 1]

    ln_sim = np.log(sim)
    ln_expt = np.log(expt)

    colors = plot_format.get_plot_color(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ln_sim[:, 0], y=ln_sim[:, 1], mode="markers", marker=dict(color=colors[0], size=10),
                             name=f"<b>simulation</b>"))
    lin_reg(fig, ln_sim[:, 0], ln_sim[:, 1], color=colors[0])
    fig.add_trace(go.Scatter(x=ln_expt[:, 0], y=ln_expt[:, 1], mode="markers", marker=dict(color=colors[1], size=10),
                             name=f"<b>experiment</b>"))
    lin_reg(fig, ln_expt[:, 0], ln_expt[:, 1], x_eq=0.6, y_eq=0.60, color=colors[1])

    fig.add_annotation(x=2.1, y=0.2, text="\u031A ", font={"size": 100})
    fig.add_annotation(x=2.4, y=.7, text="<b> -2 slope <br>(theory) </b>", font={"size": 24})

    plot_format.add_plot_format(fig, x_axis="<b>ln(distance (cm))</b>", y_axis="<b>ln(normalized intensity)</b>")
    fig.update_layout(legend=dict(x=.75, y=.95))
    fig.write_html("ln_dependence.html", auto_open=True, include_plotlyjs='cdn')

    ###################################################################################################################
    # rdf plots
    with open("rdf_inverse.csv", "r") as file:
        rdf = np.loadtxt(file, delimiter=",")
    colors = plot_format.get_plot_color(rdf.shape[1] - 1)
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
            go.Scatter(x=x_theory, y=y_theory / max_theory, mode="lines", line=dict(color=color, width=3),
                       name=f"{height} cm ")  # (theory)
        )
        fig.add_trace(
            go.Scatter(
                x=rdf[:, 0], y=rdf[:, i + 1] / max_,
                mode="lines", line=dict(color=color, width=3, dash='dash'), name=f"{height} cm (sim)", showlegend=False
            )
        )

    plot_format.add_plot_format(fig, x_axis="distance (cm)", y_axis="relative light flux")
    fig.write_html("rdf.html", auto_open=True, include_plotlyjs='cdn')

    # plotting rdf normalized
    fig = go.Figure()
    for i, (color, height) in enumerate(zip(colors, sim[:, 0])):
        y_theory = raypy_t.intensity_on_flat_surface(x_theory, 0, height)
        fig.add_trace(
            go.Scatter(x=x_theory, y=y_theory / np.max(y_theory), mode="lines", line=dict(color=color, width=3),
                       name=f"{height} cm ")  # (theory)
        )
        fig.add_trace(
            go.Scatter(
                x=rdf[:, 0], y=rdf[:, i + 1] / np.max(rdf[:, i + 1]),
                mode="lines", line=dict(color=color, width=3, dash='dash'), name=f"<b>{height} cm (sim)</b>", showlegend=False
            )
        )

    plot_format.add_plot_format(fig, x_axis="<b>distance (cm)</b>", y_axis="<b>normalized light flux</b>")
    fig.write_html("rdf_normalized.html", auto_open=True, include_plotlyjs='cdn')


if __name__ == '__main__':
    main()
