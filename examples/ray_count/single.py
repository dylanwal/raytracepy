"""
Single Light over a horizontal plane

The code runs the simulation and saves the data as pickle files, so it can be processed in a separate file.

Time it takes to compute: ~ 30 sec on laptop from 2014
"""

import numpy as np
import plotly.graph_objs as go

import raytracepy as rpy
import raytracepy.theory as raypy_t
import raytracepy.utils.analysis_func as raypy_a
import examples.plot_format as plot_format


def run(num_rays):
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=20,
        width=20
    )

    # define lights
    light = rpy.Light(
        position=np.array([0, 0, 4], dtype='float64'),
        direction=np.array([0, 0, -1], dtype='float64'),
        theta_func=0
    )

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=ground,
        lights=light,
        total_num_rays=num_rays
    )
    sim.run()

    return sim


def get_similar_color(color_in: str, num_colors: int, mode: str = "dark") -> list[str]:
    import re
    rgb = re.findall("[0-9]{1,3}", color_in)
    rgb = [int(i) for i in rgb]
    if mode == "dark":
        change_rgb = [i > 120 for i in rgb]
        jump_amount = [-int((i - 80) / num_colors) for i in rgb]
        jump_amount = [v if i else 0 for i, v in zip(change_rgb, jump_amount)]

    elif mode == "light":
        jump_amount = [int(100 / num_colors) if i < 100 else int((245-i)/num_colors) for i in rgb]

    else:
        raise ValueError(f"Invalid 'mode'; only 'light' or 'dark'. (mode: {mode})")

    colors = []
    for i in range(num_colors):
        r = rgb[0] + jump_amount[0] * (i+1)
        g = rgb[1] + jump_amount[1] * (i+1)
        b = rgb[2] + jump_amount[2] * (i+1)
        colors.append(f"rgb({r},{g},{b})")

    return colors


def main():
    num_rays = [10_000, 50_000, 100_000, 300_000, 600_000, 2_000_000, 5_000_000]
    fig = go.Figure()
    colors = ['rgb(26,161,199)', 'rgb(144,215,248)', 'rgb(130,227,212)', 'rgb(108,187,150)', 'rgb(85,176,109)',
              'rgb(61,164,68)', 'rgb(29,98,0)']
    for num, color in zip(num_rays, colors):
        sim = run(num)
        xy = sim.planes["ground"].rdf(bins=30)
        y_theory = raypy_t.intensity_on_flat_surface(xy[0], 0, 4)
        y_theory /= np.max(y_theory)
        fig.add_trace(go.Scatter(x=xy[0], y=xy[1] / np.max(xy[1]), mode="lines", name=str(num), line=dict(
            color=color)))
        print(num, np.sqrt(np.mean((y_theory-xy[1] / np.max(xy[1]))**2)))


    fig.add_trace(go.Scatter(x=xy[0], y=y_theory, mode="lines", name="theory",
                             line=dict(color='rgb(172,24,25)')))

    plot_format.add_plot_format(fig, x_axis="distance (cm)", y_axis="normalized intensity")
    fig.write_html("radial_dist.html", auto_open=True, include_plotlyjs='cdn')


if __name__ == "__main__":
    main()
