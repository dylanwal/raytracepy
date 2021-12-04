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
from numba import njit
from numba.pycc import CC


dtype = "float64"


cc = CC('core')
cc.verbose = True


@njit
@cc.export('refection_vector', 'f8[:](f8[:], f8[:])')
def refection_vector(vector, plane_normal):
    return -2 * np.dot(vector, plane_normal) * plane_normal + vector


@njit
@cc.export('check_in_plane_range', 'b1(f8[:], f8[:])')
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
@cc.export('rotate_vec', 'f8[:,:](f8[:,:], f8[:])')
def rotate_vec(rays, direction):
    prim_dir = np.array([0.0000000000000016, 0.000000000000296, 0.99999999999988], dtype='float64')
    axis_of_rotate = np.cross(prim_dir, direction)
    angle_of_rotate = np.arccos(np.dot(prim_dir, direction))
    q_vector = into_quaternion_from_axis_angle(axis=axis_of_rotate, angle=angle_of_rotate)

    for i in range(rays.shape[0]):
        rays[i] = rotate_quaternion(q_vector, rays[i])[1:]
    return rays


@njit
@cc.export('into_quaternion_from_axis_angle', 'f8[:](f8[:], f8)')
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

    return np.append(r, i)


@njit
@cc.export('rotate_quaternion', 'f8[:](f8[:], f8[:])')
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
@cc.export('q_conjugate', 'f8[:](f8[:])')
def q_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=dtype)


@njit
@cc.export('q_matrix', 'f8[:,:](f8[:])')
def q_matrix(q):
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]]], dtype=dtype)


# @njit
# @cc.export('plane_ray_intersection', 'f8[:](f8[:], f8[:], f8[:], f8[:])')
# def plane_ray_intersection(rays_dir: np.ndarray, rays_pos: np.ndarray, plane_dir: np.ndarray,
#                            plane_pos: np.ndarray):
#     """
#     Given a rays position and direction, and given a plane position and normal direction; calculate the intersection
#     point and angle.
#     :param rays_dir:
#     :param rays_pos:
#     :param plane_dir:
#     :param plane_pos:
#     :return:
#     """
#     ndotu = np.dot(plane_dir, rays_dir)
#     if -1 * ndotu < 0.1:
#         return None
#         # print("No intersection; vector point away from surface.")
#
#     w = rays_pos - plane_pos
#     si = -np.dot(plane_dir, w) / ndotu
#     if si < 0:
#         return None
#         # print("No intersection; Vector is located on the back of the plane pointing away.")
#
#     intersection = w + si * rays_dir + plane_pos
#     return intersection


@njit
@cc.export('get_phi', "f8[:](f8[:],i4)")
def get_phi(phi_rad: np.ndarray = np.array([0, 2 * np.pi], dtype=dtype),
            num_rays: int = 1) -> np.ndarray:
    """generate rays angles in spherical coordinates"""
    return (phi_rad[1] - phi_rad[0]) * np.random.random(num_rays) + phi_rad[0]


@njit
@cc.export('spherical_to_cartesian', 'f8[:,:](f8[:], f8[:], i8)')
def spherical_to_cartesian(theta, phi, r=1):
    """
    Converts spherical coordinates (theta, phi, r) into cartesian coordinates [x, y, z]
    :param theta (radian)
    :param phi (radian)
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
@cc.export('normalise', "f8[:](f8[:])")
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


if __name__ == "__main__":
    cc.compile()
