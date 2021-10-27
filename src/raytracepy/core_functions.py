
import numpy as np
from numba import jit

from . import _jit

def toggle_jit(func):
    if not _jit:
        return func

    return jit(func, nopython=True)



@toggle_jit
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


@toggle_jit
def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    """
    Converts spherical coordinates (theta, phi, r) into cartesian coordinates [x, y, z]
    Spherical coordinates physics (ISO 80000-2:2019) convention
    :param spherical = [theta, phi, r]
        theta is z,x plane    [0, 360] or  [0, 2*pi]
        phi is x,y plane (positive only) [0, 180] or [0, pi]
        r along z axis
    :return np.array([x,y,z])
    """
    theta = spherical[0]
    phi = spherical[1]
    r = spherical[2]
    mat = np.empty((theta.size, 3), dtype="float64")
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


@toggle_jit
def refection_vector(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Calculates the reflection of incoming ray.
    :param vector: Incoming normalized vector
    :param plane_normal: normal vector of plane
    """
    return -2 * np.dot(vector, plane_normal) * plane_normal + vector


@toggle_jit
def check_in_plane_range(point: np.ndarray, plane_corners: np.ndarray) -> bool:
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
    else:
        if plane_corners[0] != point[0]:
            return False

    # y
    if plane_corners[2] != plane_corners[3]:
        if plane_corners[2] > point[1] or point[0] > plane_corners[3]:
            return False
    else:
        if plane_corners[2] != point[1]:
            return False

    # z
    if plane_corners[4] != plane_corners[5]:
        if plane_corners[4] > point[2] or point[0] > plane_corners[5]:
            return False
    else:
        if plane_corners[4] != point[2]:
            return False

    # raise ValueError
    return True


@toggle_jit
def rotate_vec(rays: np.ndarray, direction: np.ndarray, prim_dir = np.array([1e-10, 1e-10, 1-1e-10], dtype='float64')) \
        -> np.ndarray:
    """
    Given a ray with a referance to [0,0,1], rotate it to a new referance direction
    :param rays: rays to be rotated
    :param prim_dir: referance direction
    :param direction: new reference direction
    :return: rotated rays
    """
    axis_of_rotate = np.cross(prim_dir, direction)
    angle_of_rotate = np.arccos(np.dot(prim_dir, direction))
    q_vector = into_quaternion_from_axis_angle(axis=axis_of_rotate, angle=angle_of_rotate)

    for i in range(rays.shape[0]):
        rays[i] = rotate_quaternion(q_vector, rays[i])[1:]
    return rays


@toggle_jit
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


@toggle_jit
def rotate_quaternion(q, vector):
    """Rotate a quaternion vector using the stored rotation.

    Params:
        vec: The vector to be rotated, in quaternion form (0 + xi + yj + kz)

    Returns:
        A Quaternion object representing the rotated vector in quaternion from (0 + xi + yj + kz)
    """
    q_vector = np.array([0, vector[0], vector[1], vector[2]], dtype="float64")  # turn into q
    q_vector = normalise(q_vector)
    return np.dot(q_matrix(np.dot(q_matrix(q), q_vector)), q_conjugate(q))


@toggle_jit
def q_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype="float64")


@toggle_jit
def q_matrix(q):
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]]], dtype="float64")


@toggle_jit
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