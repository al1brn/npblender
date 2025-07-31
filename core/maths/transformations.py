"""
This file is part of the geonodes distribution (https://github.com/al1brn/npblender).
Copyright (c) 2025 Alain Bernard.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------
numpy for Blender
-----------------------------------------------------


Spatial transformations and geometry utilities
-----------------------------------------------------


Created on Wed Mar  9 13:33:50 2022
Updated : 2025/07/18

"""

import numpy as np

#from npblender import Rotation

# Zero
ZERO = 1e-10
PI   = np.pi
TAU  = 2*np.pi

# =============================================================================================================================
# Spatial operations
# =============================================================================================================================

# -----------------------------------------------------------------------------------------------------------------------------
# Normalize vectors
# -----------------------------------------------------------------------------------------------------------------------------

def normalize(v, null=[0., 0., 1.]):
    """ Normalize a vector or an array of vectors.

    The vector can be specified as a string naming an axis : 'x', '-z', ...

    Arguments
    ---------
        - v (vector or array of vectors or str) : the vector to normalize
        - null (vector=(0, 0, 1)) : value to set to null vectors

    Returns
    -------
        - normalized vector(s), vector norm(s)
    """

    # ---------------------------------------------------------------------------
    # An axis string specification
    # ---------------------------------------------------------------------------

    if isinstance(v, str):

        upper = v.upper()

        if upper in ['X', '+X', 'POS_X', 'I', '+I']:
            return np.array((1., 0., 0.)), np.array(1.)
        elif upper in ['Y', '+Y', 'POS_Y', 'J', '+J']:
            return np.array((0., 1., 0.)), np.array(1.)
        elif upper in ['Z', '+Z', 'POS_Z', 'K', '+K']:
            return np.array((0., 0., 1.)), np.array(1.)

        elif upper in ['-X', 'NEG_X', '-I']:
            return np.array((-1., 0., 0.)), np.array(1.)
        elif upper in ['-Y', 'NEG_Y', '-J']:
            return np.array((0., -1., 0.)), np.array(1.)
        elif upper in ['-Z', 'NEG_Z', '-K']:
            return np.array((0., 0., -1.)), np.array(1.)
        else:
            raise RuntimeError(f"Unknwon axis spec: '{v}'")
        
    # ---------------------------------------------------------------------------
    # An array of vectors
    # ---------------------------------------------------------------------------
        
    vectors = np.asarray(v)
    VNull = np.asarray(null)

    # Check input shapes
    if vectors.shape[-1] != 3 or VNull.shape != (3,):
        raise ValueError("vectors must have shape (..., 3) and null must have shape (3,)")

    # Compute the Euclidean norm along the last axis (vector dimension)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)

    # Identify zero-norm vectors to avoid division by zero
    is_zero = norms[..., 0] < ZERO

    # To avoid warning message with 0
    norms[is_zero] = 1

    # Normalize non-zero vectors, replace zero vectors with replace_nulls    
    normalized = np.where(
        is_zero[..., None],  # Expand dims for broadcasting
        VNull,               # Use fallback vector
        vectors / norms      # Normalize normally
    )

    return normalized, norms.reshape(norms.shape[:-1])

# -----------------------------------------------------------------------------------------------------------------------------
# Reflect
# -----------------------------------------------------------------------------------------------------------------------------

def reflect(v, normal=[0., 0., 1.], factor=1.):
    """ Reflection one a plane

    Arguments
    ---------
    - v (vector) : incident vector
    - normal (vector = (0, 0, 1)) : normal vector of length = 1
    - factor (float = 1) : percentage of normal component to return

    Returns
    -------
    - vector : reflected vector
    """
import numpy as np

def reflect(v, normal, factor=1.0):
    """
    Reflect vector(s) `v` against a plane with unit normal(s) `normal`.

    Parameters
    ----------
    v : array_like (..., 3)
        Incident vector(s)
    normal : array_like (..., 3)
        Unit normal vector(s), must be broadcastable to v
    factor : float
        Reflection strength (1.0 = full reflection)

    Returns
    -------
    reflected : ndarray (..., 3)
        Reflected vector(s), same shape as v
    """
    v = np.asarray(v)
    normal = np.asarray(normal, dtype=v.dtype)

    if v.shape[-1] != 3 or normal.shape[-1] != 3:
        raise ValueError("v and normal must end with dimension 3")

    v, normal = np.broadcast_arrays(v, normal)

    dot = np.einsum('...i,...i->...', v, normal)
    correction = (1 + factor) * dot[..., None] * normal
    return v - correction


# -----------------------------------------------------------------------------------------------------------------------------
# Perpendicular plane
# -----------------------------------------------------------------------------------------------------------------------------

def get_plane(v):
    """
    Given a vector or array of vectors v (..., 3), returns two orthonormal vectors
    in the plane perpendicular to each v.

    Arguments
    ---------
    - v (vector or array of vectors) : the vectors perpendicular to the planes

    Returns
    -------
    - couple of vectors or couple of arrays of vectors
    """

    v = np.asarray(v)
    original_shape = v.shape

    # Ensure shape (..., 3)
    v = np.reshape(v, (-1, 3)) if v.ndim == 1 else v

    u, n = normalize(v, null=np.array([0.0, 0.0, 1.0]))

    base = np.array([0.0, 1.0, 0.0])
    i_raw = np.cross(base, u)
    i, i_n = normalize(i_raw)

    # Replace degenerate cases
    fallback = np.array([1.0, 0.0, 0.0])
    i[i_n < ZERO] = fallback

    j = np.cross(u, i)

    # If input was a single vector, return (3,) arrays
    if original_shape == (3,):
        return i[0], j[0]
    else:
        return i, j

# -----------------------------------------------------------------------------------------------------------------------------
# One perpendicular vector
# -----------------------------------------------------------------------------------------------------------------------------

def one_perp_vector(u):
    """ Returns one vector perpendicular to the argument.

    Arguments
    ---------
        - u (vector or arraay of vectors) : the vectors

    Returns
    -------
        - vector or array of vectors : vectors perpendicular to the argument
    """

    return get_plane(u)[0]

# -----------------------------------------------------------------------------------------------------------------------------
# Rotation from vectors to other ones

def rotation_to(v0, v1, perp_axis=None):
import numpy as np
from scipy.spatial.transform import Rotation

ZERO = 1e-10

def normalize(v, null=(0, 0, 1)):
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    is_zero = norm[..., 0] < ZERO
    v_normed = np.where(is_zero[..., None], null, v / norm)
    return v_normed, norm.squeeze(-1)

def one_perp_vector(v):
    """
    Return a vector perpendicular to `v`. v must be (..., 3).
    The logic is simple: pick the smallest component and swap axes.
    """
    v = np.asarray(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    use_x = np.abs(x) < np.abs(y)
    use_x &= np.abs(x) < np.abs(z)
    # Construct perp vector
    perp = np.where(use_x[..., None],
                    np.cross(v, [1, 0, 0]),
                    np.cross(v, [0, 1, 0]))
    return normalize(perp)[0]

def rotation_to(v0, v1, perp_axis=None):
    """
    Compute the rotation(s) that rotates `v0` to `v1`.

    Parameters
    ----------
    v0 : array_like (..., 3)
        Initial vector(s)
    v1 : array_like (..., 3)
        Target vector(s)
    perp_axis : None or array_like (..., 3) or (3,)
        Axis to use when vectors are opposite (angle = pi)

    Returns
    -------
    scipy.spatial.transform.Rotation
        Rotation object(s)
    """
    v0 = np.asarray(v0)
    v1 = np.asarray(v1)

    u0, _ = normalize(v0)
    u1, _ = normalize(v1)

    # Cross and sin of angle
    cross = np.cross(u0, u1)
    perp, sn = normalize(cross)

    # Cosine of angle
    cs = np.einsum('...i,...i', u0, u1)

    # Detect vectors aligned or anti-aligned (sin(angle) ≈ 0)
    colinear = np.abs(sn) < ZERO

    # For colinear and cs < 0 (opposite), use π rotation around a given axis
    opposite = colinear & (cs < 0)

    if np.any(opposite):
        if perp_axis is None:
            # Auto-perpendicular for each opposite pair
            perp[opposite] = one_perp_vector(u0[opposite])
        else:
            perp_axis = np.asarray(perp_axis)
            if perp_axis.shape == (3,):
                perp[opposite] = perp_axis
            else:
                perp[opposite] = perp_axis[opposite]

    # For aligned vectors (cs ≈ 1), use identity (rotvec = 0)
    angles = np.arctan2(sn, cs)
    rotvecs = perp * angles[..., None]
    return Rotation.from_rotvec(rotvecs)



    """ Get the rotation turning one vector to another one.

    Arguments
    ---------
        - v0 (vector or array of vectors) : first vector
        - v1 (vector or array of vectors) : second vectors
        - perp_axis (vector or array of vectors = None) : an axis to use in case the two vectors are aligned

    Returns
    -------
        - Transformations : transformations rotating v0 to v1
    """

    u0, _ = normalize(v0)
    u1, _ = normalize(v1)

    perp, sn = normalize(np.cross(u0, u1))

    # ----- Single vector

    if np.shape(sn) == ():
        cs = np.dot(u0, u1)
        if abs(sn) < ZERO:
            if cs > 0 or abs(cs) < ZERO:
                return Rotation.identity()
            else:
                if perp_axis is None:
                    perp_axis = one_perp_vector(u0)
                return Rotation.from_rotvec(perp_axis * np.pi)

        ag = np.arctan2(sn, cs)
        return Rotation.from_rotvec(perp*ag)

    # ----- Multiple vectors

    sn_0 = np.abs(sn) < ZERO
    if np.sum(sn_0):
        u = u0 if np.size(u0) > np.size(u1) else u1
        if perp_axis is None:
            perp[sn_0] = one_perp_vector(u[sn_0])
        elif np.shape(perp_axis) == (3,):
            perp[sn_0] = perp_axis
        else:
            perp[sn_0] = perp_axis[sn_0]

    cs = np.einsum('...i, ...i', u0, u1)
    ag = np.arctan2(sn, cs)

    return Rotation.from_rotvec(perp*ag[:, None])

# -----------------------------------------------------------------------------------------------------------------------------
# A tracker orients axis towards a target direction.
# Another contraint is to have the up axis oriented towards the sky
# The sky direction is the normally the Z
#
# - direction  : The target direction for the axis
# - track_axis : The axis to rotate into the given direction
# - up_axis    : The up direction wich must remain oriented vertically
# - vertical   : The up axis must be rotated into the plane (target, vertical)

def tracker(direction, track_axis='+Y', up_axis=None, vertical='Z'):
    """Compute a rotation which rotates a track axis into a direction.

    The rotation is computed using a complementary axis named 'up' which
    must be oriented vertically.
    The vertical direction is Z by default and can be overriden by the argument 'vertical'.

    After rotation:
        - 'track axis' is oriented in the 'direction'.
        - 'up axis' is in the plane (direction, vertical)

    Arguments
    ---------
        - direction (vector) : the target direction for the track axis.
        - track_axis (vector='+Y') : the axis to orient along direction
        - up_axis (vector=None) : up axis to keep in the plane (direction, vertical)
        - vertical (vector='Z') : the vertical direction

    Returns
    -------
        - Transformations
    """

    # ----- Rotate track axis toward direction

    u_track = axis_vector(track_axis)
    u_dir   = axis_vector(direction)
    rot     = rotation_to(u_track, u_dir)

    # ----- No constraint on up, it is all what we need

    if up_axis is None:
        return Transformations(rotation=rot)

    # ----- We must rotate the up axis around direction in the plane containing vertical and direction

    rotated_up = rot.apply(axis_vector(up_axis))

    # ----- Vertical vector

    Z = axis_vector(vertical)

    # ----- Perpendicular to the target plane

    perp = np.cross(Z, u_dir)

    # ----- Back to the plane

    target, t_n = normalize(np.cross(u_dir, perp))

    # ----- Single vector

    if np.shape(target) == (3,):
        if t_n < ZERO:#
            return Transformations(rotation=rot)
        else:
            return Transformations(rotation=rotation_to(rotated_up, target, perp_axis=u_dir) * rot)

    # ----- Multiple vector

    if np.shape(rotated_up) == (3,):
        target[t_n < ZERO] = rotated_up
    else:
        target[t_n < ZERO] = rotated_up[t_n < ZERO]

    return Transformations(rotation=rotation_to(rotated_up, target, perp_axis=u_dir) * rot)

# -----------------------------------------------------------------------------------------------------------------------------
# Rotate points located in XY plane to a plane perpendicular to the given vector

def rotate_xy_into_plane(points, plane=None, origin=(0, 0, 0)):
    if plane is None:
        return points

    i, j = get_plane(plane)
    if np.shape(points) == (3,):
        return i*points[0] + j*points[1] + origin
    else:
        return i*points[..., 0, None] + j*points[..., 1, None] + origin

# -----------------------------------------------------------------------------------------------------------------------------
# Index of an axis

def axis_index(vector, signed=False):
    """ Return the axis index of a vector.

    If the argument 'signed' is False, the returned index is in [0, 1, 2], otherwise it is in [-3, -2, -1, 1, 2, 3].

    Arguments
    ---------
        - vector (vector or str) : an axis vector
        - signed (bool=False) : take the orientation into account

    Returns
    -------
        - Axis index
    """

    if isinstance(vector, str):

        upper = vector.upper()

        if upper in ['X', '+X', 'POS_X', 'I', '+I']:
            index = 1
        elif upper in ['Y', '+Y', 'POS_Y', 'J', '+J']:
            index = 2
        elif upper in ['Z', '+Z', 'POS_Z', 'K', '+K']:
            index = 3

        elif upper in ['-X', 'NEG_X', '-I']:
            index = -1
        elif upper in ['-Y', 'NEG_Y', '-J']:
            index = -2
        elif upper in ['-Z', 'NEG_Z', '-K']:
            index = -3
        else:
            raise RuntimeError(f"Unknwon axis spec: '{vectors}'")

    else:
        vector = np.reshape(vector, (3,))
        v = axis_vector(vector)

        index = np.argmax(v)
        if v[index] < 0:
            index = - (index + 1)
        else:
            index += 1

    if signed:
        return index
    else:
        return abs(index) - 1

# ---------------------------------------------------------------------------
# Angle with another vector

def angle_with(v0, v1):
    """ Angle between to vectors.

    Arguments
    ---------
        - v0 (vector or array of vectors) : first vector
        - v1 (vector or array of vectors) : second vector

    Returns
    -------
        - angle : the angle (mod pi) between the two vectors
    """

    return np.arccos(np.clip(np.einsum('...i, ...i', axis_vector(v0), axis_vector(v1)), -1, 1))

# =============================================================================================================================
# Transformation

class Transformations:
    def __init__(self, position=None, scale=None, rotation=None, order='xyz'):
        """ Transformations manages the 3 transformation : position, scale and rotation.

        The rotation can be either an array of eulers angles or a, instance of scipy Rotation

        Arguments
        ---------
            - position (vector or array of vectors = None) : the translation part
            - scale (vector or array of vectors = None) : the scale part
            - rotation (vector or array of vectors or Rotation = None) = the rotation part
        """

        # scale can be a float

        if scale is not None and np.shape(scale) == ():
            scale = (scale, scale, scale)

        # rotation can be array of eulers or Rotation

        if rotation is None:
            rot_shape = ()
        else:
            if isinstance(rotation, Rotation):
                if rotation.single:
                    rot_shape = (3,)
                else:
                    rot_shape = (len(rotation), 3)

            else:
                rot_shape = np.shape(rotation)
                rotation = Rotation.from_euler(order.lower(), np.reshape(rotation, (np.size(rotation)//3, 3)))

        # Global shape

        self._shape = np.broadcast_shapes(np.shape(position), np.shape(scale), rot_shape)
        self._size  = np.prod(self._shape, dtype=int)

        # We are still alove, let's init

        self.position = None if position is None else np.reshape(position, (np.size(position)//3, 3))
        self.scale    = None if scale    is None else np.reshape(scale,    (np.size(scale)   //3, 3))
        self.rotation = rotation

    def __str__(self):
        return f"<Transformations {self.shape}: position: {np.shape(self.position)[:-1]}, scale: {np.shape(self.scale)[:-1]}, rotation: ({'' if self.rotation is None else self.rotation})>"

    # =============================================================================================================================
    # Shape

    @property
    def shape(self):
        """ The shape of the Transformations.

        Returns
        -------
            - tuple : shape
        """

        return self._shape[:-1]

    @property
    def size(self):
        """ The size of the Transformations.

        Returns
        -------
            - int : Transformations size
        """

        return self._size // 3

    @property
    def single(self):
        """ A single transformation (no shape, no len).

        Returns
        -------
            - bool : True if single, False otherwise
        """

        return self._shape == (3,)

    def reshape(self, *new_shape):
        """ Change the shape of the Transformations.

        Both the following syntaxes are accepted:
            - ``` transfos.reshape(2, 3, 4) ````
            - ``` transfos.reshape((2, 3, 4)) ```

        Raises an error if the size of the shape doesn't correspond to the current size.

        Arguments
        ---------
            - ints or tuple of ints : the new shape
        """

        if len(new_shape) == 1:
            target = new_shape[0]
        else:
            target = new_shape

        if isinstance(target, list):
            target = tuple(target)
        elif not isinstance(target, tuple):
            target = (target,)

        if np.prod(target, dtype=int) != self.size:
            raise RuntimeError(f"Impossible to resize Transformations of shape {self.shape} to shape {target}")

        self._shape = target + (3,)

    # =============================================================================================================================
    # As a list of transformation

    def __len__(self):
        if self.single:
            raise TypeError(f"Single Transformations has no len()")
        else:
            return self._shape[0]

    def __getitem__(self, index):

        if self.single:
            raise TypeError(f"Single Transformations is not subscriptable")

        indices      = np.reshape(range(self.size), self.shape)[index]
        target_shape = np.shape(indices)
        indices      = indices.flatten()

        count = len(indices)
        if count == 0:
            return None

        if self.position is None:
            position = None
        elif np.size(self.position) == 3:
            position = np.reshape(self.position, (3,))
        else:
            position = self.position[indices]

        if self.scale is None:
            scale = None
        elif np.size(self.scale) == 3:
            scale = np.reshape(self.scale, (3,))
        else:
            scale = self.scale[indices]

        if self.rotation is None:
            rotation = None
        elif self.rotation.single:
            rotation = self.rotation
        else:
            rotation = self.rotation[indices]

        transfos = Transformations(position=position, scale=scale, rotation=rotation)
        transfos.reshape(target_shape)

        return transfos

    # =============================================================================================================================
    # Transformation matrices

    @property
    def scaled_matrices(self):
        size = 1 if self.shape == () else self.size

        if self.rotation is None:
            mats = Rotation.identity(size).as_matrix()

        else:
            mats = np.resize(self.rotation.as_matrix(), (self.size, 3, 3))

        if self.scale is not None:
            mats[:, :3, 0] *= self.scale[:, 0, None]
            mats[:, :3, 1] *= self.scale[:, 1, None]
            mats[:, :3, 2] *= self.scale[:, 2, None]

        return np.reshape(mats, self.shape + (3, 3))

    @scaled_matrices.setter
    def scaled_matrices(self, mats):

        size = 1 if self.shape == () else self.size
        mats = np.reshape(mats, (size, 3, 3))

        scale = np.linalg.norm(mats, axis=-1)
        self.scale = np.array(scale)

        scale[scales < ZERO] = 1
        self.rotation = Rotation.from_matrix(mats / scale[..., None])


    @property
    def tmatrices(self):

        size = 1 if self.shape == () else self.size

        tmats = np.ones((size, 4, 4), float)
        tmats[:,  3, 3]  = 0
        if self.position is None:
            tmats[:, :3, 3]  = 0
        else:
            tmats[:, :3, 3]  = self.position
        tmats[:, :3, :3] = np.reshape(self.scaled_matrices, (size, 3, 3))

        return np.reshape(tmats, self.shape + (4, 4))

    @tmatrices.setter
    def tmatrices(self, tmats):

        size = 1 if self.shape == () else self.size
        mats = np.reshape(tmats, (size, 4, 4))

        self.position        = tmats[..., :3, 3]
        self.scaled_matrices = tmats[..., :3, :3]

    # =============================================================================================================================
    # Transformation

    def __matmul__(self, other):
        return self.transform(other)

    def transform(self, v):
        """ Transform an array of vectors.

        The shape of the argument 'v' must be broadcastable with the shape of the transformations.

        Arguments
        ----------
            - v (vector or array of vectors) : the vectors to transform

        Returns
        -------
            - array of vectors :  the transformed vectors
        """

        # ----- Simple : both are linear

        if len(np.shape(v)) <= 2 and len(self._shape) <= 2:
            if self.scale is not None:
                v = v * self.scale

            if self.rotation is not None:
                v = self.rotation.apply(v)

            if self.position is not None:
                v = v + self.position

            return v

        # ----- Things are shapped, we must use a shapped transformation matrix

        v = np.append(v, np.ones(np.shape(v)[:-1] + (1,), float), axis=-1)

        v = np.einsum('...ij,...j', self.tmatrices, v)

        return np.delete(v, 3, axis=-1)
