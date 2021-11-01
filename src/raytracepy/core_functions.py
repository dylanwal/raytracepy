from typing import Union

import numpy as np
from numba import njit, config

config.DISABLE_JIT = True


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


@njit
def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    """
    Converts spherical coordinates (theta, phi, r) into cartesian coordinates [x, y, z]
    Spherical coordinates physics (ISO 80000-2:2019) convention
    :param spherical = [theta, phi, r]
        theta is z,x plane    [0, 360] or  [0, 2*pi]
        phi is x,y plane (positive only) [0, 180] or [0, pi]
        r along z axis
    :return np.ndarray([x,y,z])
    """
    if len(spherical.shape) == 1:
        spherical = spherical.reshape(1, 3)

    theta = spherical[:, 0]
    phi = spherical[:, 1]
    r = spherical[:, 2]
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


@njit
def refection_vector(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Calculates the reflection of incoming ray.
    :param vector: Incoming normalized vector
    :param plane_normal: normal vector of plane
    """
    return -2 * np.dot(vector, plane_normal) * plane_normal + vector


@njit
def check_in_plane_range(point: np.ndarray, plane_corners: np.ndarray) -> bool:
    """
    Checks if point is within the bounds of plane
    :param point:
    :param plane_corners: [-x, x, -y, y, -z, z]
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
        if plane_corners[2] > point[1] or point[1] > plane_corners[3]:
            return False
    else:
        if plane_corners[2] != point[1]:
            return False

    # z
    if plane_corners[4] != plane_corners[5]:
        if plane_corners[4] > point[2] or point[2] > plane_corners[5]:
            return False
    else:
        if plane_corners[4] != point[2]:
            return False

    return True


@njit
def rotate_vec(rays: np.ndarray,
               direction: np.ndarray,
               prim_dir: np.ndarray = normalise(np.array([0, 0, 1], dtype='float64'))) -> np.ndarray:
    """
    Given a ray with a reference to [0,0,1], rotate it to a new reference direction
    :param rays: rays to be rotated (must be normalized)
    :param prim_dir: reference direction (must be normalized)
    :param direction: new reference direction (must be normalized)
    :return: rotated rays
    """

    if not np.any(prim_dir) or not np.any(direction):
        return rays  # No rotation

    axis_of_rotate = normalise(np.cross(prim_dir, direction))
    if np.all(axis_of_rotate == np.zeros(3)):
        if np.all(prim_dir == direction):
            return rays  # No rotation
        else:
            neg_index = np.argmax(np.abs(prim_dir, direction))
            flip_array = np.ones(3)
            flip_array[neg_index] = -1
            return rays * flip_array

    angle_of_rotate = np.arccos(np.dot(prim_dir, direction))
    q = quaternion_from_axis_angle(axis_of_rotate, angle_of_rotate)

    return rotate_quaternion(q, rays)


@njit
def quaternion_from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Create a Quaternion

    :param axis: a valid numpy 3-vector
    :param angle: a real valued angle in radians
    :return
    """
    theta = angle / 2.0
    r = np.cos(theta)
    q = axis * np.sin(theta)

    return np.append(q, r)


@njit("f8[:](f8[:],f8[:])")
def rotate_quaternion(q: np.ndarray, rays: np.ndarray):
    """
    Rotate a quaternion vector using the stored rotation.
    :param q: quaternion form (0 + xi + yj + kz)
    :param rays: The vector to be rotated, in
    :return rotated vectors
    """
    q_ray = np.append(np.zeros(rays.shape[0]).reshape(rays.shape[0], 1), rays, axis=1)
    q1 = np.dot(q_matrix(q), q_ray)
    q2 = q_matrix(q1)
    q3 = q_conjugate(q)
    q_rotated = np.dot(q2, q3)

    return q_rotated[1:, :]


@njit
def q_conjugate(q):
    """w, x, y, z = q  ->   [w, -x, -y, -z]"""
    return q*np.array([1, -1, -1, -1])


@njit("f8[:,:](f8[:])")
def q_matrix(q: np.ndarray) -> np.ndarray:
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]]], dtype="float64")


@njit
def plane_ray_intersection(rays_dir: np.ndarray,
                           rays_pos: np.ndarray,
                           plane_dir: np.ndarray,
                           plane_pos: np.ndarray) -> Union[None, np.ndarray]:
    """
    Given a rays position and direction, and given a plane position and normal direction; calculate the intersection
    point and angle.

    :param rays_dir: direction of ray
    :param rays_pos: x,y,z position of ray
    :param plane_dir: normal vector of plane
    :param plane_pos: x,y,z position of plane
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
def get_phi(phi_rad: np.ndarray = np.array([0, 359.999], dtype="float64"), num_rays: int = 1) -> np.ndarray:
    """generate rays angles in spherical coordinates"""
    return (phi_rad[1] - phi_rad[0]) * np.random.random_sample(num_rays) + phi_rad[0]


def create_ray(theta: np.ndarray, phi: np.ndarray, source_dir: np.ndarray) -> np.ndarray:
    """
    :return: direction vector of ray [[x1,y1,z1], [x2,y2,z2]]
    """
    # convert from spherical to cartesian coordinates
    rays = spherical_to_cartesian(np.column_stack((theta, phi, np.ones_like(theta))))

    # rotate rays to correct direction
    rays = rotate_vec(rays, source_dir)

    return rays


def trace_rays(ray_position: np.ndarray, ray_direction: np.ndarray, planes, max_bounces):
    for ray_dir in ray_direction:
        ray = np.hstack((ray_dir, ray_position))
        trace_ray(ray, planes, max_bounces)


def trace_ray(ray: np.ndarray, planes, max_bounces: int):
    for bounce in range(max_bounces + 1):
        plane_uid, hit = check_ray_plane_hit(ray, planes)

        if bounce <= max_bounces and plane_uid is not None:
            ray = check_bounce_transmit(planes[plane_uid], ray, hit)
            if ray is None:
                break  # ray absorbed

    if plane_uid is not None:
        try:
            planes[plane_uid].hits = np.vstack((planes[plane_uid].hits, hit))
        except ValueError:
            planes[plane_uid].hits = hit


def trace_rays_lock(lock, ray_position: np.ndarray, ray_direction: np.ndarray, planes, max_bounces):
    for ray_dir in ray_direction:
        ray = np.hstack((ray_dir, ray_position))
        trace_ray_lock(lock, ray, planes, max_bounces)


def trace_ray_lock(lock, ray: np.ndarray, planes, max_bounces: int):
    for bounce in range(max_bounces + 1):
        plane_uid, hit = check_ray_plane_hit(ray, planes)

        if bounce <= max_bounces and plane_uid is not None:
            ray = check_bounce_transmit(planes[plane_uid], ray, hit)
            if ray is None:
                break  # ray absorbed

    if plane_uid is not None:
        lock.acquire()
        try:
            try:
                planes[plane_uid].hits = np.vstack((planes[plane_uid].hits, hit))
            except ValueError:
                planes[plane_uid].hits = hit
        finally:
            lock.release()


def check_bounce_transmit(plane, ray: np.ndarray, hit: np.ndarray):
    angle = np.arcsin(np.dot(ray[0:3], plane.normal))
    if 0 < plane.trans_type_id <= 2:  # transmitted light
        prob = plane.transmit_func(angle)  # probably of light transmitting given the angle.
        if np.random.random() < prob:
            # if transmitted, calculate diffraction ray and continue tracing the ray
            new_dir = create_ray(plane.scatter_func(), get_phi(), ray[0:3])
            return np.hstack((new_dir, hit))

    if plane.trans_type_id >= 2:  # reflected light
        prob = plane.reflect_func(angle)  # probably of light reflected given the angle
        if np.random.random() < prob:
            # if reflected, calculate reflection ray and continue tracing the ray
            new_dir = normalise(refection_vector(ray, plane.normal))
            return np.hstack((new_dir, hit))

    return None


def check_ray_plane_hit(ray, planes):
    for plane in planes:
        intersect_cord = plane_ray_intersection(rays_dir=ray[0:3], rays_pos=ray[3:],
                                                plane_dir=plane.normal, plane_pos=plane.position)
        if intersect_cord is not None:
            if check_in_plane_range(point=intersect_cord, plane_corners=plane.corners):
                return plane.uid, intersect_cord

    return None, None
