"""
Array of Lights over a horizontal plane

"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go

import raytracepy as rpy


def rectangle_points(corner: list[float] = (0, 0), center: list[float] = None, x_length: float = 10,
                     y_length: float = 10):
    """ Creates the points of a rectangle in an order appropate for plotting."""
    # set position
    if center is not None:
        corner = [center[0] - x_length / 2, center[1] - y_length / 2]

    out = np.array([
        corner,
        [corner[0] + x_length, corner[1]],
        [corner[0] + x_length, corner[1] + y_length],
        [corner[0], corner[1] + y_length]]
    )
    return out


def rectangle_points_for_plotting(corner: list[float] = (0, 0), center: list[float] = None, x_length: float = 10,
                                  y_length: float = 10):
    """ Creates the points of a rectangle in an order appropate for plotting and adds 5 point to close the drawing."""
    _points = rectangle_points(corner, center, x_length, y_length)
    return np.insert(_points, 4, _points[0], axis=0)


def run_single(h: float, grid):
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, -1], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=20,
        width=20,
        transmit_type="absorb",
        bins=(100, 100)
    )

    box_dim = 15
    box_height = h + 0.25
    mirror_left = rpy.Plane(
        name="mirror_left",
        position=np.array([-box_dim / 2, 0-0.5, box_height / 2], dtype='float64'),
        normal=np.array([1, 0, 0], dtype='float64'),
        length=box_dim,
        width=box_height,
        transmit_type="reflect",
        reflect_func=4,
    )
    mirror_right = rpy.Plane(
        name="mirror_right",
        position=np.array([box_dim / 2, -0.5, box_height / 2], dtype='float64'),
        normal=np.array([-1, 0, 0], dtype='float64'),
        length=box_dim,
        width=box_height,
        transmit_type="reflect",
        reflect_func=4,
    )
    mirror_front = rpy.Plane(
        name="mirror_front",
        position=np.array([0, -box_dim / 2-.5, box_height / 2], dtype='float64'),
        normal=np.array([0, 1, 0], dtype='float64'),
        length=box_dim,
        width=box_height,
        transmit_type="reflect",
        reflect_func=4,
    )
    mirror_back = rpy.Plane(
        name="mirror_back",
        position=np.array([0, box_dim / 2-0.5, box_height / 2], dtype='float64'),
        normal=np.array([0, -1, 0], dtype='float64'),
        length=box_dim,
        width=box_height,
        transmit_type="reflect",
        reflect_func=4,
    )
    top = rpy.Plane(
        name="top",
        position=np.array([0, -.5, box_height], dtype='float64'),
        normal=np.array([0, 0, -1], dtype='float64'),
        length=box_dim,
        width=box_dim,
        transmit_type="reflect",
        reflect_func=6,
    )

    # Important note** ground should be last in list! RayTrace simulation evaluates ray hits in order of plane list
    planes = [top, mirror_left, mirror_right, mirror_front, mirror_back, ground]

    # define lights
    height = h
    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.insert(xy_pos, 2, height),
                direction=np.array([0, 0, -1], dtype='float64'),
                num_traces=5,
                theta_func=1,
            ))

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=planes,
        lights=lights,
        total_num_rays=10_000_000,
        bounce_max=20,
    )
    sim.run()
    return sim


def main():
    grid = rpy.OffsetGridPattern(
        center=np.array([0, -0.5]),
        x_length=12.5,
        y_length=12.5,
        num_points=50)

    sim = run_single(h=8, grid=grid)
    sim.plot_report()

    # print stats
    sim.stats()
    sim.planes["ground"].hit_stats()
    sim.planes["ground"].hit_stats(True)

    # getting sensor data
    xy = rpy.GridPattern(center=np.array([0, 0]), x_length=18, y_length=18, num_points=18*18).xy_points
    z = sim.planes["ground"].shape_grid(xy=xy, r=0.4)
    df = pd.DataFrame(np.column_stack((xy, z)), columns=["x", "y", "integ_watts"])
    # df.to_csv("mirror.csv")

    # plotting
    fig = sim.planes["ground"].plot_sensor(xy=xy, r=0.4, normalize=True, save_open=False)
    sim._add_lights_2D(fig)
    box = rectangle_points_for_plotting(center=[0, 0], x_length=11, y_length=11)
    fig.add_trace(
        go.Scatter(
            x=box[:, 0],
            y=box[:, 1],
            mode='lines', connectgaps=True, line=dict(color='rgb(100,100,100)', width=3)
        ))

    fig.data[0].marker.colorbar.title.text = "<b>Irradiance (W/m<sup>2</sup>)</b>"
    fig.data[0].marker.colorbar.tickprefix = "<b>"
    fig.data[0].marker.colorbar.ticksuffix = "</b>"
    fig.update_layout(autosize=False, width=1000, height=790, showlegend=False,
                      font=dict(family="Arial", size=18, color="black"), plot_bgcolor="white",
                      margin=dict(
                          l=50,
                          r=40,
                          b=50,
                          t=62,
                          pad=4
                      ),
                      )
    fig.update_xaxes(title="<b>X</b>", range=[-10, 10], tickprefix="<b>", ticksuffix="</b>", showline=True, linewidth=5,
                     linecolor='black', showgrid=False, mirror=True)
    fig.update_yaxes(title="<b>Y</b>", range=[-10, 10], tickprefix="<b>", ticksuffix="</b>", showline=True, linewidth=5,
                     linecolor='black', showgrid=False, mirror=True)
    # fig.write_html("temp.html", auto_open=True)


if __name__ == "__main__":
    main()

