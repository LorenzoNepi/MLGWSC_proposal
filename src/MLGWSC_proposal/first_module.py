import math

import numpy as np
from numpy.typing import NDArray


def rot_axis(
    vector: NDArray[np.float64], angle: float, axis: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    """Rotate a vector [x, y, z] around a given axis by a certain angle.

    Parameters
    ----------
    vector : np.ndarray
        Vector to rotate
    angle : float
        Angle in radians of the rotation
    axis : np.ndarray
        Axis around which the rotation is made

    Returns
    -------
    np.ndarray
        Rotated vector
    """
    if np.allclose(vector, 0) or np.allclose(axis, 0):
        raise ValueError("Cannot compute rotations with zero vector or axis")

    axis = axis / np.linalg.norm(axis)
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    return cos_theta * vector + sin_theta * np.cross(axis, vector) + ((1 - cos_theta) *
                                                                      np.dot(axis, vector) * axis)

def rot_angle(vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> float:
    """Compute the angle between two vectors [x, y, z].

    Parameters
    ----------
    vec1 : np.ndarray
        First vector
    vec2 : np.ndarray
        Second vector

    Returns
    -------
    float
        Angle between the vectors in radians
    """
    if np.allclose(vec1, 0) or np.allclose(vec2, 0):
        raise ValueError("Cannot compute rotations with zero vectors")

    return np.arccos(np.clip(np.dot(vec1, vec2) /
                    (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

def find_perp(vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Find a vector perpendicular to two given vectors [x, y, z] (right hand rule).

    Parameters
    ----------
    vec1 : np.ndarray
        First vector
    vec2 : np.ndarray
        Second vector

    Returns
    -------
    np.ndarray
        Perpendicular vector
    """
    if np.allclose(vec1, 0) or np.allclose(vec2, 0):
        raise ValueError("Cannot find the perpendicular with zero vectors")

    perp=np.cross(vec1, vec2)
    return perp/np.linalg.norm(perp)
