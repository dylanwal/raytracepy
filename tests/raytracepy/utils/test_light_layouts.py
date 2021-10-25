import pytest
import numpy as np

import raytracepy as rpy


def test_circle():
    circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=50, layers=3, op_center=True)
    assert circle.xy_points[0, 0] == 0


def test_circle_no_center():
    circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=50, layers=3, op_center=False)
    assert circle.xy_points[0, 0] != 0


def test_square():
    square = rpy.GridPattern(center=np.array([0, 0]), x_length=10, num_points=23)
    assert all(square.xy_points[0] == np.array([-5, -5]))


def test_square_not_offset():
    square = rpy.GridPattern(center=np.array([0, 0]), x_count=10, op_offset=False)
    assert square.xy_points[5][1] == 0


if __name__ == '__main__':
    # circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=50, layers=3, op_center=True)
    # circle.create_plot()
    # circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=40, layers=2, op_center=False)
    # circle.create_plot()
    square = rpy.GridPattern(center=np.array([0, 0]), x_count=10, op_offset=False)
    fig = square.create_plot()
