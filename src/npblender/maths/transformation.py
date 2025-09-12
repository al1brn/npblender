# MIT License
#
# Copyright (c) 2025 Alain Bernard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the \"Software\"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module Name: transformation
Author: Alain Bernard
Version: 0.1.0
Created: 2022-11-11
Last updated: 2025-09-02

Batch of 4×4 homogeneous transformation matrices, including translation, rotation and scale.
Inherits from `ItemsArray` with `_item_shape = (4, 4)`. Provides:

- Vectorized operations on transformation matrices
- Decomposition into rotation (as `Rotation`), scale, and translation
- Composition (`@` operator), inversion (`~`), and interpolation
- Component accessors (position, rotation, scale)
- Construction from components via `combine(...)`

Example:
    >>> T = Transformation.combine(translation=[1, 2, 3], rotation=R, scale=[2, 2, 2])
    >>> v_transformed = T @ vectors
"""

__all__ = ['Transformation']

import numpy as np

from .constants import ZERO

from .itemsarray import ItemsArray
from .rotation import Rotation
from .quaternion import Quaternion

class Transformation(ItemsArray):

    _item_shape = (4, 4) # Array of 4x4 matrices

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def is_identity(self, eps: float = 1e-5) -> np.ndarray:
        """Return a boolean array indicating which matrices are identity.

        Parameters
        ----------
        eps : float, optional
            Tolerance for element-wise comparison. Default is global ZERO.

        Returns
        -------
        mask : ndarray of bool, shape == self.shape
            True where transformation is (approximately) the identity.
        """
        # Reference identity matrix
        ident = np.eye(4, dtype=self.dtype)

        # Compute absolute difference and reduce over last two dims
        diff = np.abs(self.as_array() - ident)
        return np.all(diff < eps, axis=(-2, -1))
        
    # ------------------------------------------------------------------
    # Comoose several transformations together
    # ------------------------------------------------------------------

    @classmethod
    def compose(cls, *transforms: "Transformation") -> "Transformation":
        """Compose multiple transformations together (right to left).

        Parameters
        ----------
        *transforms : Transformation
            Any number of Transformation objects to compose. Composition is
            performed as `T1 @ T2 @ T3`, meaning `T3` is applied first.

        Returns
        -------
        Transformation
            The composed transformation.
        """
        if not transforms:
            raise ValueError("At least one Transformation must be provided")

        result = transforms[0]
        for tf in transforms[1:]:
            result = result @ tf
        return result

    # ------------------------------------------------------------------
    # Constructor helper: combine T * R * S
    # ------------------------------------------------------------------

    @classmethod
    def from_components(cls, translation=None, rotation=None, scale=None) -> "Transformation":
        """Build a *Transformation* from translation, rotation, and scale.

        Parameters
        ----------
        translation : array_like (..., 3), optional
            Translation vector(s).  Default is ``[0, 0, 0]``.
        rotation : None or array_like (..., 3, 3) or Rotation, optional
            Rotation matrix/matrices.  If *None*, uses the identity.
        scale : array_like (..., 3), optional
            Per‑axis scale vector(s).  Default is ``[1, 1, 1]``.

        Notes
        -----
        *All three inputs are *broadcasted* together*.  The returned batch
        shape is the broadcasted shape of ``translation[...,0]``,
        ``scale[...,0]``, and ``rotation[...,0,0]``.
        """
        # Convert to arrays with the global dtype
        if translation is None:
            t = np.array([0, 0, 0], dtype=cls.FLOAT)
        else:
            t = np.asarray(translation, dtype=cls.FLOAT)

        if scale is None:
            s = np.array([1, 1, 1], dtype=cls.FLOAT)
        else:
            s = np.asarray(scale, dtype=cls.FLOAT)

        if t.shape[-1] != 3 or s.shape[-1] != 3:
            raise ValueError("translation and scale must end with 3 components")

        if rotation is None:
            r = np.eye(3, dtype=cls.FLOAT)
        else:
            if isinstance(rotation, Quaternion):
                rotation = rotation.as_matrix()
            r = np.asarray(rotation, dtype=cls.FLOAT)
            if r.shape[-2:] != (3, 3):
                raise ValueError("rotation must end with (3, 3)")

        # Determine broadcast batch shape (NumPy ≥1.20)
        batch_shape = np.broadcast_shapes(t.shape[:-1], s.shape[:-1], r.shape[:-2])

        # Broadcast each component to the batch shape
        t_b = np.broadcast_to(t, batch_shape + (3,))
        s_b = np.broadcast_to(s, batch_shape + (3,))
        r_b = np.broadcast_to(r, batch_shape + (3, 3))

        # Build the 4×4 matrix : R * S then add translation
        rs = r_b * s_b[..., None, :]  # scale columns

        mat = np.zeros(batch_shape + (4, 4), dtype=cls.FLOAT)
        mat[..., :3, :3] = rs
        mat[..., :3, 3] = t_b
        mat[..., 3, 3] = 1.0

        return cls(mat, copy=False)

    # ------------------------------------------------------------------
    # Decomposition: rotation, scale, translation
    # ------------------------------------------------------------------

    def decompose(self):
        """Return rotation, scale, translation from each 4x4 transform.

        Returns
        -------
        rot : Rotation
            Rotation (orthonormal, no scale)
        scale : ndarray (..., 3)
            Scale vector along each axis
        trans : ndarray (..., 3)
            Translation vector
        """
        mat = self.as_array()
        rot_scale = mat[..., :3, :3]  # (..., 3, 3)
        trans = mat[..., :3, 3]       # (..., 3)

        # Extract scale as norm of each column vector
        scale = np.linalg.norm(rot_scale, axis=-2)  # (..., 3)

        # Normalize columns to get pure rotation
        rot = rot_scale / np.maximum(scale[..., None, :], ZERO)

        return Rotation(rot, copy=False), scale, trans
    
    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Access the translation component (view, not copy)."""
        return self.as_array()[..., :3, 3]

    @position.setter
    def position(self, value: np.ndarray):
        """Set the translation component in-place."""
        value = np.asarray(value, dtype=self.FLOAT)
        if value.shape != self.as_array()[..., :3, 3].shape:
            raise ValueError(f"Position shape {value.shape} does not match {self._mat[..., :3, 3].shape}")
        self.as_array()[..., :3, 3] = value

    @property
    def rotation(self) -> np.ndarray:
        """Return the pure rotation part of the matrix."""
        rot_scale = self._mat[..., :3, :3]
        scale = np.linalg.norm(rot_scale, axis=-2, keepdims=True)
        return Rotation(rot_scale / np.maximum(scale, ZERO), copy=False)

    @rotation.setter
    def rotation(self, value: np.ndarray):
        """Set the rotation component (keeping scale)."""
        mat = self.as_array()
        value = np.asarray(value, dtype=self.FLOAT)
        if value.shape != mat[..., :3, :3].shape:
            raise ValueError(f"Rotation shape {value.shape} does not match {mat[..., :3, :3].shape}")
        scale = self.scale[..., None, :]  # shape (..., 1, 3)
        mat[..., :3, :3] = value * scale

    @property
    def scale(self) -> np.ndarray:
        """Extract scale from transformation matrix (per axis)."""
        return np.linalg.norm(self.as_array()[..., :3, :3], axis=-2)

    @scale.setter
    def scale(self, value: np.ndarray):
        """Set the scale component (keeping rotation)."""
        mat = self.as_array()
        value = np.asarray(value, dtype=self.FLOAT)
        if value.shape != mat[..., :3, 3].shape:
            raise ValueError(f"Scale shape {value.shape} does not match {(mat[..., :3, 3].shape)}")
        rot = self.rotation  # pure rotation, shape (..., 3, 3)
        mat[..., :3, :3] = rot * value[..., None, :]
    
    # ------------------------------------------------------------------
    # Inversion
    # ------------------------------------------------------------------

    def inverse(self) -> "Transformation":
        """Return the inverse of the transformation(s).

        Returns
        -------
        Transformation
            A new Transformation representing the inverse.
        """
        mat = self.as_array()
        rot = mat[..., :3, :3]       # (..., 3, 3)
        trans = mat[..., :3, 3]      # (..., 3)

        rot_inv = np.swapaxes(rot, -1, -2)  # transpose
        trans_inv = -np.einsum("...ij,...j->...i", rot_inv, trans)

        inv = np.empty_like(mat)
        inv[..., :3, :3] = rot_inv
        inv[..., :3, 3] = trans_inv
        inv[..., 3, :3] = 0.0
        inv[..., 3, 3] = 1.0

        return Transformation(inv, copy=False)
    
    # ------------------------------------------------------------------
    # Normalize rotation
    # ------------------------------------------------------------------

    def normalize_rotation(self, in_place: bool = False) -> "Transformation":
        """Orthonormalize the rotation part of each matrix via SVD.

        Parameters
        ----------
        in_place : bool, default False
            If True, modifies the current instance. Otherwise, returns a new one.

        Returns
        -------
        Transformation
            A Transformation with orthonormal rotation matrices.
        """
        mat = self.as_array() if in_place else self.as_array().copy()
        rs = mat[..., :3, :3]  # extract rotation-scale block

        # Perform SVD on each 3×3 matrix
        u, _, vt = np.linalg.svd(rs)
        r_ortho = np.matmul(u, vt)

        # Restore orthonormal block
        mat[..., :3, :3] = r_ortho
        return self if in_place else Transformation(mat, copy=False)

    # ------------------------------------------------------------------
    # Transform vectors using the transformation matrix
    # ------------------------------------------------------------------

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the transformation(s) to 3D or 4D vectors.

        Parameters
        ----------
        vectors : array_like (..., 3) or (..., 4)
            Input vectors. If shape[-1] == 3, a 1 is appended to enable
            homogeneous transformation. If already 4D, used as-is.

        Returns
        -------
        transformed : ndarray (..., 3 or 4)
            Transformed vectors. Shape is the broadcast of self.shape and vectors.shape[:-1].
        """
        vectors = np.asarray(vectors, dtype=self.FLOAT)
        if vectors.shape[-1] not in (3, 4):
            raise ValueError("Input vectors must have shape (..., 3) or (..., 4)")

        was_3d = vectors.shape[-1] == 3

        if was_3d:
            ones = np.ones(vectors.shape[:-1] + (1,), dtype=self.FLOAT)
            vectors = np.concatenate([vectors, ones], axis=-1)  # (..., 4)

        # Broadcast matrix and vectors together
        mat = self.as_array()
        result = np.einsum('...ij,...j->...i', mat, vectors)  # (..., 4)

        if was_3d:
            return result[..., :3]  # drop w component if input was 3D
        else:
            return result
        
    # ------------------------------------------------------------------
    # Overloaded operators / dunder helpers
    # ------------------------------------------------------------------

    def __matmul__(self, other):
        """Overload the **@** operator.

        * **Transformation @ Transformation**  → composition (self ∘ other)
          where the right operand is applied first, then *self* (column‑major
          convention).
        * **Transformation @ vectors**  → alias for :py:meth:`transform`.
        """
        # Compose two transformations (broadcast‑aware)
        if isinstance(other, Transformation):
            composed = np.matmul(self.as_array(), other.as_array())  # (..., 4, 4)
            return Transformation(composed, copy=False)

        # Otherwise, assume "other" is an array‑like of vectors
        return self.apply(other)
    
    def __invert__(self) -> "Transformation":
        """Overload the ~ operator to return the inverse transformation."""
        return self.inverse()

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(self, other: "Transformation", t: float | np.ndarray) -> "Transformation":
        """Interpolate between two transformations A and B using factor t.

        Parameters
        ----------
        other : Transformation
            The other transformation to interpolate with.
        t : float or array_like (...,)
            Interpolation weight(s) in [0, 1]. Shape must be broadcastable.

        Returns
        -------
        Transformation
            Interpolated transformation(s).
        """
        t = np.asarray(t, dtype=self.FLOAT)

        # Decompose A and B
        Ra, Sa, Ta = self.decompose()
        Rb, Sb, Tb = other.decompose()

        # Interpolate translation and scale
        T_interp = Ta + (Tb - Ta) * t[..., None]
        S_interp = Sa + (Sb - Sa) * t[..., None]

        # Interpolate rotation via SLERP (quaternions)
        qa = Ra.as_quaternion().as_array()
        qb = Rb.as_quaternion().as_array()

        # Ensure same hemisphere (avoid long path)
        dot = np.sum(qa * qb, axis=-1, keepdims=True)
        qb = np.where(dot < 0, -qb, qb)

        theta = np.arccos(np.clip(np.sum(qa * qb, axis=-1), -1.0, 1.0))  # (...,)
        sin_theta = np.sin(theta)

        # Avoid division by zero
        mask = sin_theta > ZERO

        w1 = np.where(mask, np.sin((1 - t) * theta) / sin_theta, 1.0 - t)
        w2 = np.where(mask, np.sin(t * theta) / sin_theta, t)

        q_interp = (qa * w1[..., None]) + (qb * w2[..., None])
        q_interp = q_interp / np.linalg.norm(q_interp, axis=-1, keepdims=True)

        # Recompose
        return Transformation.from_components(Quaternion(q_interp, copy=False), translation=T_interp, scale=S_interp)
    
