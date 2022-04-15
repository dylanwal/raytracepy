"""
Array of Lights over a horizontal plane

"""

import numpy as np
import plotly.graph_objs as go

import raytracepy as rpy


def rectangle_points(corner: list[float] = [0, 0], center: list[float] = None, x_length: float = 10,
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


def rectangle_points_for_plotting(corner: list[float] = [0, 0], center: list[float] = None, x_length: float = 10,
                                  y_length: float = 10):
    """ Creates the points of a rectangle in an order appropate for plotting and adds 5 point to close the drawing."""
    _points = rectangle_points(corner, center, x_length, y_length)
    return np.insert(_points, 4, _points[0], axis=0)


def run_single(h: float, grid):
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=20,
        width=20,
        transmit_type="absorb",
        bins=(100, 100)
    )

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
        planes=ground,
        lights=lights,
        total_num_rays=5_000_000,
    )
    sim.run()
    return sim


def main():
    grid = rpy.OffsetGridPattern(
        center=np.array([0, 0]),
        x_length=12.5,
        y_length=12.5,
        num_points=50)

    sim = run_single(h=9, grid=grid)
    file_name = "array_led_radio"
    # sim.save_data(file_name)

    # print stats
    sim.stats()
    sim.planes["ground"].hit_stats()
    sim.planes["ground"].hit_stats(True)

    # plotting
    # sim.plot_report(file_name)

    xy = rpy.GridPattern(center=np.array([0, 0]), x_length=18, y_length=18, num_points=18*18).xy_points
    # x = xy[:, 0].reshape((10, 10)).T.reshape(100)
    # y = xy[:, 1].reshape((10, 10)).T.reshape(100)
    # xy = np.column_stack((x, y))
    # xy = np.insert(xy, 2, no_mirror_integral[2:-1], axis=1)

    fig = sim.planes["ground"].plot_sensor(xy=xy, r=0.85, normalize=True, save_open=False)
    sim._add_lights_2D(fig)
    box = rectangle_points_for_plotting(center=[0, 0], x_length=10, y_length=10)
    fig.add_trace(
        go.Scatter(
            x=box[:, 0],
            y=box[:, 1],
            mode='lines', connectgaps=True, line=dict(color='rgb(100,100,100)', width=3)
        ))

    fig.update_layout(width=1080, height=790)
    fig.write_html("radio.html", auto_open=True)


if __name__ == "__main__":
    main()

