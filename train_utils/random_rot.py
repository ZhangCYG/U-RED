import numpy as np
from math import cos, sin, radians

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 3D transformations
# ------------------------------------------------------------------------------


def offset_matrix_3d(offset) -> np.ndarray:
    return np.array(
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            offset[0], offset[1], offset[2], 1,
        ],
        dtype=float
    ).reshape(4, 4)


def scale_matrix_3d(scale) -> np.ndarray:
    return np.array(
        [
            scale[0], 0, 0, 0,
            0, scale[1], 0, 0,
            0, 0, scale[2], 0,
            0, 0, 0, 1,
        ],
        dtype=float
    ).reshape(4, 4)


def x_rotation_matrix_3d(angle: float) -> np.ndarray:
    '''Creates a 3D rotation matrix over the X axis.'''
    return np.array(
        [
            1, 0, 0, 0,
            0, cos(angle), sin(angle), 0,
            0, -sin(angle), cos(angle), 0,
            0, 0, 0, 1,
        ],
        dtype=float
    ).reshape(4, 4)


def y_rotation_matrix_3d(angle: float) -> np.ndarray:
    '''Creates a 3D rotation matrix over the Y axis.'''
    return np.array(
        [
            cos(angle), 0, -sin(angle), 0,
            0, 1, 0, 0,
            sin(angle), 0, cos(angle), 0,
            0, 0, 0, 1,
        ],
        dtype=float
    ).reshape(4, 4)


def z_rotation_matrix_3d(angle: float) -> np.ndarray:
    '''Creates a 3D rotation matrix over the Z axis.'''
    return np.array(
        [
            cos(angle), sin(angle), 0, 0,
            -sin(angle), cos(angle), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ],
        dtype=float
    ).reshape(4, 4)


def rotation_matrix_3d(
    angle_x: float,
    angle_y: float,
    angle_z: float,
) -> np.ndarray:
    '''Creates a 3D rotation matrix in X -> Y -> Z order.'''
    angle_x, angle_y, angle_z = (
        radians(angle) for angle in (angle_x, angle_y, angle_z)
    )

    return (
        x_rotation_matrix_3d(angle_x) @
        y_rotation_matrix_3d(angle_y) @
        z_rotation_matrix_3d(angle_z)
    )


if __name__ == '__main__':
    R = rotation_matrix_3d(90.0, 0.0, 0.0)
    print (R)