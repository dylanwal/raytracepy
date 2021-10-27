
import pytest
import numpy as np

import raytracepy.core_functions as rpyc

target_normalise = [
    [np.array([0,0,0]), np.array([0,0,0])],
    [np.array([0,0,1]), np.array([0,0,1])],
    [np.array([0,1,0]), np.array([0,1,0])],
    [np.array([1,0,0]), np.array([1,0,0])],
    [np.array([-1,0,0]), np.array([-1,0,0])],
    [np.array([1,1,1]), np.array([0.57735027,0.57735027,0.57735027])],
]


@pytest.mark.parametrize("input_, output_", target_normalise)
def test_normalise(input_, output_):
    result = rpyc.normalise(input_)
    assert np.allclose(result, output_)


target_spherical_to_cartesian = [
    [np.array([0,0,0]), np.array([0,0,0])],
    [np.array([0,0,1]), np.array([0,0,1])],
    [np.array([np.pi/2,0,1]), np.array([1,0,0])],
    [np.array([0,np.pi/2,1]), np.array([0,0,1])],
    [np.array([np.pi/2,np.pi/2,1]), np.array([0,1,0])],
    [np.array([-np.pi/2,-np.pi/2,1]), np.array([0,1,0])],
]


@pytest.mark.parametrize("input_, output_", target_spherical_to_cartesian)
def test_normalise(input_, output_):
    result = rpyc.spherical_to_cartesian(input_)
    assert np.allclose(result, output_)


target_refection_vector = [
    [np.array([0,0,0]), np.array([0,0,1]), np.array([0,0,0])],
    [np.array([0,0,1]), np.array([0,0,1]), np.array([0,0,-1])],
    [np.array([0,0,-1]), np.array([0.57735027, 0.57735027, 0.57735027]), np.array([0.66666667,  0.66666667,
                                                                             -0.33333333])],
]


@pytest.mark.parametrize("input_, plane_, output_", target_refection_vector)
def test_normalise(input_, plane_, output_):
    result = rpyc.refection_vector(input_, plane_)
    assert np.allclose(result, output_)


target_check_in_plane_range = [
    [np.array([0,0,0]), np.array([-5,5,-0,0,0,0]), True],
    [np.array([0,0,1]), np.array([-5,5,-0,0,0,0]), False],

]


@pytest.mark.parametrize("input_, corner_, output_", target_check_in_plane_range)
def test_normalise(input_, corner_, output_):
    result = rpyc.check_in_plane_range(input_, corner_)
    assert result == output_


target_rotate_vec = [
    [np.array([0,0,1]), np.array([0,0,-1]), np.array([0,0,-1])],

]


@pytest.mark.parametrize("input_, dir_, output_", target_rotate_vec)
def test_normalise(input_, dir_, output_):
    result = rpyc.rotate_vec(input_, dir_)
    assert result == output_
