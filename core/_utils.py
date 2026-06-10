import warp as wp

import numpy as np

def rodriguesMatrix(rodriguesParams) -> np.ndarray:
    rodriguesParams = np.asarray(rodriguesParams, dtype=np.float32).reshape(3)

    I = np.eye(3, dtype=np.float32)

    theta = np.linalg.norm(rodriguesParams)
    if theta < 1.e-8:
        return I

    k = rodriguesParams / theta
    K = np.array([
        [   0.0, -k[2],  k[1]],
        [  k[2],   0.0, -k[0]],
        [ -k[1],  k[0],  0.0],
    ], dtype=np.float32)

    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

@wp.func
def _rodriguesMatrix(rodriguesParams: wp.vec3f) -> wp.mat33f:

    I = wp.identity(3, wp.float32)

    theta = wp.length(rodriguesParams)
    if theta < wp.float32(1.e-8):
        return I

    k = rodriguesParams / theta
    K = wp.mat33f(
         0.0, -k.z,  k.y,
         k.z,  0.0, -k.x,
        -k.y,  k.x,  0.0
    )

    return I + wp.sin(theta) * K + (1.0 - wp.cos(theta)) * (K @ K)