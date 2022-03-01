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

from . import dtype, njit
from .light_plane_funcs import theta_func_selector, probability_func_selector


@njit(cache=True)   # ("f8[::](f8,f8[:],f8[:],i4)")
def create_rays(theta_func_id, direction,
                phi: np.ndarray = np.array([0, 2 * np.pi], dtype=dtype),
                num_rays: int = 1) -> np.ndarray:
    """
    Create the direction vectors for rays.

    :param theta_func_id:
    :param direction:
    :param phi:
    :param num_rays:
    :return num_rays by 3 matrix [x,y,z] direction vector
     """
    theta = theta_func_selector(theta_func_id, num_rays)
    phi = get_phi(phi, num_rays)
    rays_dir = spherical_to_cartesian(theta, phi)
    rays_dir = rotate_vec(rays_dir, direction)
    return rays_dir


@njit(cache=True)
def trace_rays(ray_pos: np.ndarray, ray_dir: np.ndarray, plane_matrix: np.ndarray,
               bounce_max: int, traces: np.ndarray):
    """

    :param ray_pos:
    :param ray_dir:
    :param plane_matrix:
    :param bounce_max:
    :param traces:
    :return:
    """
    traces_counter = 0
    for i in range(ray_dir.shape[0]):  # Loop through each ray until it reaches max bounces or absorbs into surface.
        ray = ray_dir[i, :-1]
        plane_hit = -1
        traces[traces_counter, :3] = ray_pos
        for bounce in range(bounce_max + 1):
            for plane in plane_matrix:  # calculate plane intersections
                if plane[-1] != plane_hit:
                    # check to see if ray will hit infinite plane
                    intersect_cord = plane_ray_intersection(rays_dir=ray,
                                                            rays_pos=traces[traces_counter, bounce*3: 3+bounce*3],
                                                            plane_dir=plane[7:10], plane_pos=plane[4:7])
                    # check to see if the hit is within the bounds of the plane
                    if intersect_cord is not None and check_in_plane_range(point=intersect_cord,
                                                                           plane_corners=plane[10:]):
                        plane_hit = plane[-1]
                        traces[traces_counter, 3+bounce*3: 6+bounce*3] = intersect_cord
                        traces[traces_counter, -1] = bounce
                        break
            else:
                break  # does not hit any planes

            if bounce < bounce_max:  # skip if no bounces left
                angle = np.sin(np.sqrt(ray[0]**2+ray[1]**2))  # calculate angle light hit plane

                if 0 < plane[0] <= 2:  # if plane type allows transmitted light
                    # calculate probably of light transmitting given the angle.
                    prob = probability_func_selector(plane[1], angle)
                    if np.random.random() < prob:
                        # if transmitted, calculate diffraction ray new direction
                        ray = create_rays(plane[2], direction=ray).reshape(3)
                        continue  # continue tracing the ray

                if plane[0] >= 2:  # reflected light
                    # calculate probably of light reflecting given the angle.
                    prob = probability_func_selector(plane[3], angle)
                    if np.random.random() < prob:
                        # if reflected, calculate reflection ray
                        ray = normalise(refection_vector(ray, plane[7:10]))
                        continue  # continue tracing the ray

            break  # absorbed ray (mirror or diffuser)

        # record final location of ray
        ray_dir[i, -1] = plane_hit
        ray_dir[i, :-1] = traces[traces_counter, 3+bounce*3: 6+bounce*3]

        if traces[traces_counter, -1] != -1:  # if the trace doesn't hit anything, continue with same data row
            if traces_counter < traces.shape[0]-1:
                traces_counter += 1

    return ray_dir, traces


@njit(cache=True)
def refection_vector(vector, plane_normal):
    return -2 * np.dot(vector, plane_normal) * plane_normal + vector


@njit(cache=True)
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


@njit(cache=True)
def rotate_vec(rays, direction):
    prim_dir = np.array([0.0000000000000016, 0.000000000000296, 0.99999999999988], dtype='float64')
    axis_of_rotate = np.cross(prim_dir, direction)
    angle_of_rotate = np.arccos(np.dot(prim_dir, direction))
    q_vector = into_quaternion_from_axis_angle(axis=axis_of_rotate, angle=angle_of_rotate)

    for i in range(rays.shape[0]):
        rays[i] = rotate_quaternion(q_vector, rays[i])[1:]
    return rays


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def q_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=dtype)


@njit(cache=True)
def q_matrix(q):
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]]], dtype=dtype)


@njit(cache=True)
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


@njit(cache=True)   # ("f8[:](f8[:],i4)")
def get_phi(phi_rad: np.ndarray = np.array([0, 2 * np.pi], dtype=dtype),
            num_rays: int = 1) -> np.ndarray:
    """generate rays angles in spherical coordinates"""
    return (phi_rad[1] - phi_rad[0]) * np.random.random(num_rays) + phi_rad[0]


@njit(cache=True)
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


@njit(cache=True)  # ("f8[:](f8[:])")
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
