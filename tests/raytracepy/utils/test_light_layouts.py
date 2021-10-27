import pytest
import numpy as np

import raytracepy as rpy


def test_circle():
    circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=5, layers=2, op_center=True)
    assert np.allclose(circle.xy_points, np.array([[0., 0.],
                                                   [10., 0.],
                                                   [-5., 8.66025404],
                                                   [-5., -8.66025404],
                                                   [5., 0.]]), atol=0.1)


def test_circle_radii():
    circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=50, layers=3, op_center=True)
    assert np.allclose(circle.radii, np.array([10, 6.66, 3.33]), atol=.1)


def test_circle_points():
    circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=5, layers=2, op_center=True)
    assert all(circle.points_per_radii == np.array([3, 1]))


def test_circle_no_center():
    circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=50, layers=3, op_center=False)
    assert circle.xy_points[0, 0] != 0


def test_grid():
    grid = rpy.GridPattern(center=np.array([0, 0]), x_length=10, num_points=23)
    assert all(grid.xy_points[0] == np.array([-5, -5]))


def test_grid2():
    grid = rpy.GridPattern(corner=np.array([0, 0]), x_count=10)
    assert grid.xy_points[0][0] == 0


def test_grid3():
    grid = rpy.GridPattern(corner=np.array([0, 0]), y_count=10)
    assert grid.xy_points[0][0] == 0


def test_grid4():
    grid = rpy.GridPattern(corner=np.array([0, 0]), x_count=5, y_count=10)
    assert grid.xy_points[0][0] == 0


def test_grid5():
    grid = rpy.GridPattern(x_length=5, y_length=10)
    assert all(grid.xy_points[0] == [-2.5, -5])


def test_offsetgrid():
    grid = rpy.OffsetGridPattern()
    assert all(grid.xy_points[0] == np.array([-5, -5]))


def test_offsetgrid2():
    grid = rpy.OffsetGridPattern(x_length=5, y_length=10)
    assert all(grid.xy_points[0] == [-2.5, -5])


def test_offsetgrid3():
    grid = rpy.OffsetGridPattern(corner=np.array([0, 0]), x_count=10)
    assert grid.xy_points[0][0] == 0


def test_offsetgrid4():
    grid = rpy.OffsetGridPattern(corner=np.array([0, 0]), y_count=10)
    assert grid.xy_points[0][0] == 0


def test_offsetgrid5():
    grid = rpy.OffsetGridPattern(center=np.array([0, 0]), x_count=10, y_count=10)
    assert all(grid.xy_points[0] == [-5, -5])


if __name__ == '__main__':
    # circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=5, layers=2, op_center=True)
    # print(circle.points_per_radii)
    # circle = rpy.CirclePattern(center=np.array([0, 0]), outer_radius=10, num_points=40, layers=2, op_center=False)
    # circle.create_plot()
    grid = rpy.OffsetGridPattern()
    fig = grid.plot_create()
    print("hi")
