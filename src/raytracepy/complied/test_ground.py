import numpy as np
from numba import njit, config

config.DISABLE_JIT = False


@njit("f8[:](f8[:])")
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


@njit("f8[:](f8[:],f8[:])")
def rotate_quaternion(q: np.ndarray, rays: np.ndarray):
    """Rotate a quaternion vector using the stored rotation.

    Params:
        vec: The vector to be rotated, in quaternion form (0 + xi + yj + kz)

    Returns:
        A Quaternion object representing the rotated vector in quaternion from (0 + xi + yj + kz)
    """
    q_ray = np.append(0, rays)
    q1 = q_matrix(q)
    q2 = np.dot(q1, q_ray)
    q3 = q_matrix(q2)
    q4 = q_conjugate(q)
    q5 = np.dot(q3, q4)
    q6 = np.delete(q5, 0)
    return q6


if __name__ == "__main__":
    a = np.array([0, 0, -1, 0], dtype="float64")
    c = np.array([0, 0, 1], dtype="float64")
    b = rotate_quaternion(a, c)
    print(b)
