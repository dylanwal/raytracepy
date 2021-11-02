"""
This python file preforms the math for the calculations.
numba is used to speed up the calculations.
    * This makes the code take ~15 sec for numba compiling for low ray counts (<100,000)
    * But is x10 to x30 faster for large ray counts (>10,000,000)

Notes:
    numba is very particular on data type, thus numpy arrays with dtype of dtype is solely used. If you are
    modifying or using the code, try to keep all variables in this format.
"""

import numpy as np
from numba import njit, config

from . import dtype
from ref_data.numba_funcs import plane_func_selector

config.DISABLE_JIT = True


@njit
def trace_rays(ray_pos: np.ndarray, ray_dir: np.ndarray, plane_matrix: np.ndarray,
               bounce_max, traces: np.ndarray):
    """

    :param ray_pos:
    :param ray_dir:
    :param plane_matrix:
    :param bounce_max:
    :param traces:
    :return:
    """
    for i in range(ray_dir.shape[0]):  # Loop through each ray until it reaches max bounces or absorbs into surface.
        ray = ray_dir[i, :]
        ray_position_now = ray_pos
        bounces_count = 0
        skip_plane = -1
        for _ in range(bounce_max + 1):
            bounces_count += 1

            # calculate plane intersections
            for plane in plane_matrix:
                if plane[-1] != skip_plane:
                    # check to see if ray will hit infinite plane
                    intersect_cord = plane_ray_intersection(rays_dir=ray[:-1],
                                                            rays_pos=ray_position_now,
                                                            plane_dir=plane[7:10],
                                                            plane_pos=plane[4:7])
                    if intersect_cord is not None:
                        # check to see if the hit is within the bounds of the plane
                        if check_in_plane_range(point=intersect_cord, plane_corners=plane[10:]):
                            skip_plane = plane[-1]
                            ray_dir[i, -1] = plane[-1]
                            ray_dir[i, :-1] = intersect_cord
                            break

            if bounces_count <= bounce_max:  # skip if no bounces left
                # calculate angle light hit plane
                angle = np.arcsin(np.dot(ray[:-1], plane[7:10]))
                # transmitted light
                if 0 < plane[0] <= 2:
                    # calculate probably of light transmitting given the angle.
                    prob = plane_func_selector(plane[1], np.array([angle], dtype=dtype))
                    if np.random.random() < prob:
                        # if transmitted, calculate diffraction ray and continue tracing the ray
                        ray[:-1] = create_ray(theta_fun_id=plane[2], direction=ray[:-1])
                        #ray[:-1] = np.reshape(ray[:-1], 3)
                        ray_position_now = intersect_cord
                        continue

                # reflected light
                if plane[0] >= 2:
                    # calculate probably of light transmitting given the angle.
                    prob = plane_func_selector(plane[3], np.array([angle], dtype=dtype))
                    if np.random.random() < prob:
                        # if reflected, calculate reflection ray and continue tracing the ray
                        ray[:-1] = normalise(refection_vector(ray[:-1], plane[7:10]))
                        ray_position_now = intersect_cord
                        continue

            # absorbed
            break

    return ray_dir, traces


@njit
def create_ray(theta_func_id, direction):
    phi = np.array([0, 359.999], dtype=dtype)
    theta = plane_func_selector(theta_func_id)
    rays_dir = spherical_to_cartesian(theta, phi)
    return rotate_vec(rays_dir, direction)


@njit
def refection_vector(vector, plane_normal):
    return -2 * np.dot(vector, plane_normal) * plane_normal + vector


@njit
def check_in_plane_range(point, plane_corners):
    """
    Checks if point is within the bounds of plane
    :param point:
    :param plane_corners:
    :return:
    """
    # x
    if plane_corners[0] != plane_corners[1]:
        if plane_corners[0] > point[0] or point[0] > plane_corners[1]:
            return False
    # y
    if plane_corners[2] != plane_corners[3]:
        if plane_corners[2] > point[1] or point[1] > plane_corners[3]:
            return False
    # z
    if plane_corners[4] != plane_corners[5]:
        if plane_corners[4] > point[2] or point[2] > plane_corners[5]:
            return False

    return True


@njit
def rotate_vec(rays, direction):
    prim_dir = np.array([0.00010000999999979996, 0.00010000999999979996, 0.9999999899979999], dtype='float64')
    axis_of_rotate = np.cross(prim_dir, direction)
    angle_of_rotate = np.arccos(np.dot(prim_dir, direction))
    q_vector = into_quaternion_from_axis_angle(axis=axis_of_rotate, angle=angle_of_rotate)

    for i in range(rays.shape[0]):
        rays[i] = rotate_quaternion(q_vector, rays[i])[1:]
    return rays


@njit
def into_quaternion_from_axis_angle(axis, angle):
    """Initialise from axis and angle representation

    Create a Quaternion by specifying the 3-vector rotation axis and rotation
    angle (in radians) from which the quaternion's rotation should be created.

    Params:
        axis: a valid numpy 3-vector
        angle: a real valued angle in radians
    """
    mag_sq = np.dot(axis, axis)
    if mag_sq == 0.0:
        raise ZeroDivisionError("Provided rotation axis has no length")
    # Ensure axis is in unit vector form
    if abs(1.0 - mag_sq) > 1e-12:
        axis = axis / np.sqrt(mag_sq)
    theta = angle / 2.0
    r = np.cos(theta)
    i = axis * np.sin(theta)

    return [r, i[0], i[1], i[2]]


@njit
def rotate_quaternion(q, vector):
    """Rotate a quaternion vector using the stored rotation.

    Params:
        vec: The vector to be rotated, in quaternion form (0 + xi + yj + kz)

    Returns:
        A Quaternion object representing the rotated vector in quaternion from (0 + xi + yj + kz)
    """
    q_vector = np.array([0, vector[0], vector[1], vector[2]], dtype=dtype)  # turn into q
    q_vector = normalise(q_vector)
    return np.dot(q_matrix(np.dot(q_matrix(q), q_vector)), q_conjugate(q))


@njit
def q_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=dtype)


@njit
def q_matrix(q):
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]]], dtype=dtype)


@njit
def plane_ray_intersection(rays_dir: np.ndarray, rays_pos: np.ndarray, plane_dir: np.ndarray,
                           plane_pos: np.ndarray):
    """
    Given a rays position and direction, and given a plane position and normal direction; calculate the intersection
    point and angle.
    :param rays_dir:
    :param rays_pos:
    :param plane_dir:
    :param plane_pos:
    :return:
    """
    ndotu = np.dot(plane_dir, rays_dir)
    if -1 * ndotu < 0.1:
        return None
        # print("No intersection; vector point away from surface.")

    w = rays_pos - plane_pos
    si = -np.dot(plane_dir, w) / ndotu
    if si < 0:
        return None
        # print("No intersection; Vector is located on the back of the plane pointing away.")

    intersection = w + si * rays_dir + plane_pos
    return intersection


@njit
def get_phi(phi_rad: np.ndarray = np.array([0, 359.999], dtype=dtype), num_rays: int = 1) -> np.ndarray:
    """generate rays angles in spherical coordinates"""
    return (phi_rad[1] - phi_rad[0]) * np.random.random_sample(num_rays) + phi_rad[0]


@njit
def spherical_to_cartesian(theta, phi, r=1):
    """
    Converts spherical coordinates (theta, phi, r) into cartesian coordinates [x, y, z]
    :param theta
    :param phi
    :param r
    :return np.array([x,y,z])
    """
    mat = np.empty((theta.size, 3), dtype=dtype)
    cp = np.cos(phi)
    sp = np.sin(phi)
    st = np.sin(theta)
    ct = np.cos(theta)
    x = r * cp * st
    y = r * sp * st
    z = r * ct
    mat[:, 0] = np.transpose(x)
    mat[:, 1] = np.transpose(y)
    mat[:, 2] = np.transpose(z)
    return mat


@njit
def normalise(vector: np.ndarray) -> np.ndarray:
    """
    Object is guaranteed to be a unit quaternion after calling this
    operation UNLESS the object is equivalent to Quaternion(0)
    """
    n = np.sqrt(np.dot(vector, vector))
    if n > 0:
        return vector / n
    else:
        return vector
