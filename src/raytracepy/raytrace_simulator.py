"""
This python file preforms the math for the calculations.
numba is used to speed up the calculations.
    * This makes the code take ~15 sec for numba compiling for low ray counts (<100,000)
    * But is x10 to x30 faster for large ray counts (>10,000,000)

Notes:
    numba is very particular on data type, thus numpy arrays with dtype of "float64" is solely used. If you are
    modifying or using the code, try to keep all variables in this format.
"""

import numpy as np
from numba import jit
from tabulate import tabulate
import pickle

import lens_details, plotting
from utils.sig_figs import sig_figs






def main_simulation_loop(planes, lights, data, num_rays: int = 100_000, bounces: int = 0):
    # input checking
    if isinstance(planes, Plane):
        planes = [planes]
    elif type(planes) == list:
        pass
    else:
        exit("Invalid format for planes to be passed into main_simulation_loop.")

    if isinstance(lights, Light):
        lights = [lights]
    elif type(planes) == list:
        pass
    else:
        exit("Invalid format for ligths to be passed into main_simulation_loop.")

    # Saving simulation settings to data class
    data.num_rays = num_rays
    data.calc_rays_per_light(len(lights))
    data.calc_traces_per_light(len(lights))
    data.max_bounces = bounces
    data.planes = planes
    data.lights = lights

    # determine which planes ids you want data for
    data_plane_indexing = []

    for data_plane in data.planes:
        k = 1
        for i, plane2 in enumerate(planes):
            if plane2.id == data_plane.id:
                data_plane_indexing.append([i])
                k = 0
                break
        if k == 1:
            exit("A plane given for data analysis was not given to main_simulation_loop.")

    # Plane data is grouped in this way for efficiency with numba
    if type(planes) != list:
        grouped_plane_data = planes.grouped_data
    elif type(planes) == list:
        grouped_plane_data = np.vstack([plane.grouped_data for plane in planes])
    else:
        exit("TypeError in planes entry to main simulation loop.")

    # Main Loop: Loop through each light and ray trace
    for light in lights:
        # create rays for one light
        rays = create_rays(theta_fun_id=light.emit_light_fun_id,
                           phi_range=light.phi,
                           num_rays=data.rays_per_light,
                           direction=light.direction,
                           )
        ray_positions = light.position

        # do raytracing
        out = np.ones([rays.shape[0], 4], dtype="float64") * -1
        out = set_ray_trace(ray_positions, rays, grouped_plane_data, bounces, out)

        # unpack data for planes of interest and save
        for plane, index in zip(data.planes, data_plane_indexing):
            # get the final hits for specific plane
            hits = out[out[:, 0] == index, 1:]  # [x, y, z] for every hit
            # create histogram
            data.histogram(plane, hits)


@jit(nopython=True)
def create_rays(theta_fun_id: int,
                phi_range=np.array([0, 359.999], dtype="float64"),
                num_rays: int = 100_000,
                direction: np.ndarray = np.array([0, 0, -1], dtype="float64")):
    """

    :param theta_fun_id:
    :param num_rays:
    :param direction:
    :param phi_range: y,z angle (typically between [-180,180])
    :return:
    """
    # generate rays angles in spherical coordinates
    phi = (phi_range[1] / 360 * (2 * np.pi) - phi_range[0] / 360 * (2 * np.pi)) * np.random.random_sample((num_rays,)) \
        + phi_range[0] / 360 * (2 * np.pi)
    theta = lens_details.function_selector(theta_fun_id, np.array([phi.size], dtype="float64"))

    # convert from spherical to cartesian coordinates
    rays = spherical_to_cartesian(theta, phi)

    # rotate rays to correct direction
    rays = rotate_vec(rays, direction)
    return rays


@jit(nopython=True)
def spherical_to_cartesian(theta, phi, r=1):
    """
    Converts spherical coordinates (theta, phi, r) into cartesian coordinates [x, y, z]
    :param theta
    :param phi
    :param r
    :return np.array([x,y,z])
    """
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


@jit(nopython=True)
def set_ray_trace(ray_position, rays, planes, bounces, out):
    """

    :param ray_position:
    :param rays:
    :param planes: group plane data
    :param bounces:
    :param out:
    :return: [plane_id, x, y, z] x,y,z point of intersection. if plane_id = -1 then no intersection.
    """

    # Loop through each ray until it reaches max bounces or absorbs into surface.
    for i in range(rays.shape[0]):
        ray = rays[i, :]
        ray_position_now = ray_position
        bounces_count = 0
        skip_plane = -1
        for _ in range(bounces + 1):
            bounces_count += 1

            # calculate plane intersections
            for plane_id, plane in enumerate(planes):
                if plane_id != skip_plane:
                    # check to see if ray will hit infinite plane
                    intersect_cord = plane_ray_intersection(rays_dir=ray,
                                                            rays_pos=ray_position_now,
                                                            plane_dir=plane[7:10],
                                                            plane_pos=plane[4:7])
                    if intersect_cord is not None:
                        # check to see if the hit is within the bounds of the plane
                        if check_in_plane_range(point=intersect_cord, plane_corners=plane[10:]):
                            skip_plane = plane_id
                            out[i, 0] = plane_id
                            out[i, 1:] = intersect_cord
                            break

            if bounces_count <= bounces:  # skip if no bounces left
                # calculate angle light hit plane
                angle = np.arcsin(np.dot(ray, plane[7:10]))
                # transmitted light
                if 0 < plane[0] <= 2:
                    # calculate probably of light transmitting given the angle.
                    prob = lens_details.function_selector(plane[1], np.array([angle], dtype="float64"))
                    if np.random.random() < prob:
                        # if transmitted, calculate diffraction ray and continue tracing the ray
                        ray = create_rays(theta_fun_id=plane[2], num_rays=1, direction=ray)
                        ray = np.reshape(ray, 3)
                        ray_position_now = out[i, 1:]
                        continue

                # reflected light
                if plane[0] >= 2:
                    # calculate probably of light transmitting given the angle.
                    prob = lens_details.function_selector(plane[3], np.array([angle], dtype="float64"))
                    if np.random.random() < prob:
                        # if reflected, calculate reflection ray and continue tracing the ray
                        ray = normalise(refection_vector(ray, plane[7:10]))
                        ray_position_now = out[i, 1:]
                        continue

            # absorbed
            break

    return out


@jit(nopython=True)
def refection_vector(vector, plane_normal):
    return -2 * np.dot(vector, plane_normal) * plane_normal + vector


@jit(nopython=True)
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
        if plane_corners[2] > point[1] or point[0] > plane_corners[3]:
            return False
    # z
    if plane_corners[4] != plane_corners[5]:
        if plane_corners[4] > point[2] or point[0] > plane_corners[5]:
            return False

    return True


@jit(nopython=True)
def rotate_vec(rays, direction):
    prim_dir = np.array([0.00010000999999979996, 0.00010000999999979996, 0.9999999899979999], dtype='float64')
    axis_of_rotate = np.cross(prim_dir, direction)
    angle_of_rotate = np.arccos(np.dot(prim_dir, direction))
    q_vector = into_quaternion_from_axis_angle(axis=axis_of_rotate, angle=angle_of_rotate)

    for i in range(rays.shape[0]):
        rays[i] = rotate_quaternion(q_vector, rays[i])[1:]
    return rays


@jit(nopython=True)
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


@jit(nopython=True)
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


@jit(nopython=True)
def q_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype="float64")


@jit(nopython=True)
def q_matrix(q):
    return np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], q[0], -q[3], q[2]],
        [q[2], q[3], q[0], -q[1]],
        [q[3], -q[2], q[1], q[0]]], dtype="float64")




@jit(nopython=True)
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
