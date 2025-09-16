import numpy as np

__all__ = [
    "PI", "TAU", "HALF_PI",
    "bfloat", "bint", "bbool",
    "ZERO", "EPS", "EPSILON",
]

PI = np.pi
TAU = np.pi*2
HALF_PI = np.pi/2

bfloat = np.float32
bint = np.int32
bbool = np.bool_

ZERO = 1e-6
EPS = ZERO
EPSILON = ZERO

