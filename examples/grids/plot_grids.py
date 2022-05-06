""" Plots all the grid patterns. """

import raytracepy as rpy

from examples.results_html import merge_html_figs


def main():
    num_points = 49
    circle = rpy.CirclePattern(num_points=num_points, layers=3, outer_radius=5)
    spiral = rpy.SpiralPattern(num_points=num_points, radius_start=.5, velocity=0.2, a_velocity=1, radius=5)
    grid = rpy.GridPattern(num_points=num_points)
    ogrid = rpy.OffsetGridPattern(num_points=num_points)

    merge_html_figs(
        [
            circle.plot_create(save_open=False),
            spiral.plot_create(save_open=False),
            grid.plot_create(save_open=False),
            ogrid.plot_create(save_open=False)
        ]
    )


if __name__ == "__main__":
    main()
