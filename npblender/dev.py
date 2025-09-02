import numpy as np
from maths import Rotation

tg = np.zeros((10, 3), float)
tg[:, 2] = 1

print(Rotation.look_at((0, 0, 1), tg, up=(1, 0, 0), normalized=True)._mat)