import numpy as np

ZERO = 1e-6
FLOAT = np.float32

class Transformation:
    """Efficient wrapper around an array of 4×4 homogeneous transformation
    matrices.  The underlying storage is a contiguous **numpy.ndarray** of
    shape ``(..., 4, 4)``.  Only light Python logic lives here—​all heavy
    numerics stay inside NumPy, so the class scales to **millions of
    matrices/vectors** without overhead.

    Notes
    -----
    * We assume right‑handed column‑major transforms (OpenGL/PyTorch).
      If you use row‑major matrices, flip the "apply" helpers later.
    * No automatic normalisation / re‑scaling is done; callers must
      provide valid transforms.
    """

    __slots__ = ("_mat",)

    def __init__(self, mat: np.ndarray | list | tuple, *, copy: bool = True):
        """Parameters
        ----------
        mat : array_like (..., 4, 4)
            Input transformation matrix or batch of matrices.
        copy : bool, default True
            If *True*, the input is copied; otherwise the array is taken
            by reference (no copy, zero‑cost).
        """
        mat = np.asarray(mat, dtype=FLOAT)
        if mat.shape[-2:] != (4, 4):
            raise ValueError("Input must have shape (..., 4, 4)")
        self._mat = mat.copy() if copy else mat

    # ------------------------------------------------------------------
    # Dunder helpers for convenience (optional but nice to have)
    # ------------------------------------------------------------------

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"Transformation(shape={self.shape}, dtype={self._mat.dtype})"

    # Allow NumPy to treat us like an array when needed.
    def __array__(self, dtype=None):
        return np.asarray(self._mat, dtype=dtype)

    # ------------------------------------------------------------------
    # Shaping helpers
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """**View** on the internal array (no copy)."""
        return self._mat

    @property
    def shape(self) -> tuple:
        """Batch shape (everything *except* the final 4×4)."""
        return self._mat.shape[:-2]

    @property
    def size(self) -> int:
        """Number of individual transformation matrices."""
        # E.g. shape (A, B, 4, 4) → A×B matrices.
        return int(np.prod(self.shape))

    def reshape(self, *new_shape: int) -> "Transformation":
        """Return a **new** *view* with a different batch shape.

        Examples
        --------
        >>> tf = Transformation(np.eye(4).repeat(8).reshape(2, 4, 4))
        >>> tf2 = tf.reshape(2, 2)
        >>> tf2.shape
        (2, 2)
        """
        new_mat = self._mat.reshape(*new_shape, 4, 4)
        return Transformation(new_mat, copy=False)

    def resize(self, *new_shape: int, fill_identity: bool = True) -> "Transformation":
        """In‑place *resize* (like ``ndarray.resize``).

        *If the total number of matrices increases*, new slots are filled
        with **identity matrices** when *fill_identity* is *True*; else
        with zeros.
        """
        total_new = int(np.prod(new_shape))
        total_old = self.size

        if total_new == total_old:
            # Simple reshape view.
            self._mat = self._mat.reshape(*new_shape, 4, 4)
            return self

        # Allocate new array.
        new_arr = np.empty((*new_shape, 4, 4), dtype=self._mat.dtype)
        if fill_identity:
            new_arr[...] = np.eye(4, dtype=self._mat.dtype)
        else:
            new_arr.fill(0.0)

        # Copy existing data as much as fits.
        flat_new = new_arr.reshape(-1, 4, 4)
        flat_old = self._mat.reshape(-1, 4, 4)
        flat_new[: min(total_old, total_new)] = flat_old[: min(total_old, total_new)]

        self._mat = new_arr
        return self

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def is_identity(self, eps: float = ZERO) -> np.ndarray:
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
        ident = np.eye(4, dtype=self._mat.dtype)

        # Compute absolute difference and reduce over last two dims
        diff = np.abs(self._mat - ident)
        return np.all(diff < eps, axis=(-2, -1))

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter(self, mask: np.ndarray, in_place: bool = False) -> "Transformation":
        """Filter transformations using a boolean mask.

        Parameters
        ----------
        mask : array_like (...,)
            Boolean mask matching the shape of the batch.
        in_place : bool, default False
            If True, modifies the current instance. Otherwise, returns a new one.

        Returns
        -------
        Transformation
            Filtered transformations (or self if in_place is True).
        """
        mask = np.asarray(mask)
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match batch shape {self.shape}")

        filtered = self._mat[mask]
        if in_place:
            self._mat = filtered
            return self
        else:
            return Transformation(filtered, copy=False)
        
    # ------------------------------------------------------------------
    # Comoose several transformations together
    # ------------------------------------------------------------------

    @classmethod
    def Compose(cls, *transforms: "Transformation") -> "Transformation":
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
    def Combine(cls,
                translation=(0.0, 0.0, 0.0),
                rotation=None,
                scale=(1.0, 1.0, 1.0)) -> "Transformation":
        """Build a *Transformation* from translation, rotation, and scale.

        Parameters
        ----------
        translation : array_like (..., 3), optional
            Translation vector(s).  Default is ``[0, 0, 0]``.
        rotation : None or array_like (..., 3, 3), optional
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
        t = np.asarray(translation, dtype=FLOAT)
        s = np.asarray(scale, dtype=FLOAT)

        if t.shape[-1] != 3 or s.shape[-1] != 3:
            raise ValueError("translation and scale must end with 3 components")

        if rotation is None:
            r = np.eye(3, dtype=FLOAT)
        else:
            r = np.asarray(rotation, dtype=FLOAT)
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

        mat = np.zeros(batch_shape + (4, 4), dtype=FLOAT)
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
        rot : ndarray (..., 3, 3)
            Pure rotation matrices (orthonormal, no scale)
        scale : ndarray (..., 3)
            Scale vector along each axis
        trans : ndarray (..., 3)
            Translation vector
        """
        mat = self._mat
        rot_scale = mat[..., :3, :3]  # (..., 3, 3)
        trans = mat[..., :3, 3]       # (..., 3)

        # Extract scale as norm of each column vector
        scale = np.linalg.norm(rot_scale, axis=-2)  # (..., 3)

        # Normalize columns to get pure rotation
        rot = rot_scale / np.maximum(scale[..., None, :], ZERO)

        return rot, scale, trans
    
    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Access the translation component (view, not copy)."""
        return self._mat[..., :3, 3]

    @position.setter
    def position(self, value: np.ndarray):
        """Set the translation component in-place."""
        value = np.asarray(value, dtype=FLOAT)
        if value.shape != self._mat[..., :3, 3].shape:
            raise ValueError(f"Position shape {value.shape} does not match {self._mat[..., :3, 3].shape}")
        self._mat[..., :3, 3] = value

    @property
    def rotation(self) -> np.ndarray:
        """Return the pure rotation part of the matrix."""
        rot_scale = self._mat[..., :3, :3]
        scale = np.linalg.norm(rot_scale, axis=-2, keepdims=True)
        return rot_scale / np.maximum(scale, ZERO)

    @rotation.setter
    def rotation(self, value: np.ndarray):
        """Set the rotation component (keeping scale)."""
        value = np.asarray(value, dtype=FLOAT)
        if value.shape != self._mat[..., :3, :3].shape:
            raise ValueError(f"Rotation shape {value.shape} does not match {self._mat[..., :3, :3].shape}")
        scale = self.scale[..., None, :]  # shape (..., 1, 3)
        self._mat[..., :3, :3] = value * scale

    @property
    def scale(self) -> np.ndarray:
        """Extract scale from transformation matrix (per axis)."""
        return np.linalg.norm(self._mat[..., :3, :3], axis=-2)

    @scale.setter
    def scale(self, value: np.ndarray):
        """Set the scale component (keeping rotation)."""
        value = np.asarray(value, dtype=FLOAT)
        if value.shape != self._mat[..., :3, 3].shape:
            raise ValueError(f"Scale shape {value.shape} does not match {(self._mat[..., :3, 3].shape)}")
        rot = self.rotation  # pure rotation, shape (..., 3, 3)
        self._mat[..., :3, :3] = rot * value[..., None, :]
    
    # ------------------------------------------------------------------
    # Inversion
    # ------------------------------------------------------------------

    def inverted(self) -> "Transformation":
        """Return the inverse of the transformation(s).

        Returns
        -------
        Transformation
            A new Transformation representing the inverse.
        """
        mat = self._mat
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

    def __invert__(self) -> "Transformation":
        """Overload the ~ operator to return the inverse transformation."""
        return self.inverted()
    
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
        mat = self._mat if in_place else self._mat.copy()
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

    def transform(self, vectors: np.ndarray) -> np.ndarray:
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
        vectors = np.asarray(vectors, dtype=FLOAT)
        if vectors.shape[-1] not in (3, 4):
            raise ValueError("Input vectors must have shape (..., 3) or (..., 4)")

        was_3d = vectors.shape[-1] == 3

        if was_3d:
            ones = np.ones(vectors.shape[:-1] + (1,), dtype=FLOAT)
            vectors = np.concatenate([vectors, ones], axis=-1)  # (..., 4)

        # Broadcast matrix and vectors together
        mat = self._mat
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
            composed = np.matmul(self._mat, other._mat)  # (..., 4, 4)
            return Transformation(composed, copy=False)

        # Otherwise, assume "other" is an array‑like of vectors
        return self.transform(other)

    # ------------------------------------------------------------------
    # Quaternion conversion
    # ------------------------------------------------------------------

    def to_quaternion(self) -> np.ndarray:
        """Extract unit quaternions from rotation matrices.

        Returns
        -------
        quat : ndarray (..., 4)
            Quaternion as (x, y, z, w), following scalar-last convention.
        """
        r, _, _ = self.decompose()
        m = r.astype(np.float64)  # improve numerical stability for sqrt ops

        trace = np.trace(m, axis1=-2, axis2=-1)
        q = np.empty(m.shape[:-2] + (4,), dtype=FLOAT)

        cond1 = trace > 0
        s = np.sqrt(trace[cond1] + 1.0) * 2.0
        q[cond1, 3] = 0.25 * s
        q[cond1, 0] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / s
        q[cond1, 1] = (m[cond1, 0, 2] - m[cond1, 2, 0]) / s
        q[cond1, 2] = (m[cond1, 1, 0] - m[cond1, 0, 1]) / s

        cond2 = (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2]) & ~cond1
        s = np.sqrt(1.0 + m[cond2, 0, 0] - m[cond2, 1, 1] - m[cond2, 2, 2]) * 2.0
        q[cond2, 3] = (m[cond2, 2, 1] - m[cond2, 1, 2]) / s
        q[cond2, 0] = 0.25 * s
        q[cond2, 1] = (m[cond2, 0, 1] + m[cond2, 1, 0]) / s
        q[cond2, 2] = (m[cond2, 0, 2] + m[cond2, 2, 0]) / s

        cond3 = (m[..., 1, 1] > m[..., 2, 2]) & ~cond1 & ~cond2
        s = np.sqrt(1.0 + m[cond3, 1, 1] - m[cond3, 0, 0] - m[cond3, 2, 2]) * 2.0
        q[cond3, 3] = (m[cond3, 0, 2] - m[cond3, 2, 0]) / s
        q[cond3, 0] = (m[cond3, 0, 1] + m[cond3, 1, 0]) / s
        q[cond3, 1] = 0.25 * s
        q[cond3, 2] = (m[cond3, 1, 2] + m[cond3, 2, 1]) / s

        cond4 = ~cond1 & ~cond2 & ~cond3
        s = np.sqrt(1.0 + m[cond4, 2, 2] - m[cond4, 0, 0] - m[cond4, 1, 1]) * 2.0
        q[cond4, 3] = (m[cond4, 1, 0] - m[cond4, 0, 1]) / s
        q[cond4, 0] = (m[cond4, 0, 2] + m[cond4, 2, 0]) / s
        q[cond4, 1] = (m[cond4, 1, 2] + m[cond4, 2, 1]) / s
        q[cond4, 2] = 0.25 * s

        return q.astype(FLOAT)
    
    # ------------------------------------------------------------------
    # axis-angle conversion
    # ------------------------------------------------------------------

    def to_axis_angle(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract axis and angle from rotation matrix.

        Returns
        -------
        axis : ndarray (..., 3)
            Normalized axis of rotation.
        angle : ndarray (..., 1)
            Rotation angle in radians.
        """
        r, _, _ = self.decompose()
        m = r.astype(np.float64)

        trace = np.trace(m, axis1=-2, axis2=-1)
        angle = np.arccos(np.clip((trace - 1) * 0.5, -1.0, 1.0))  # (...,)
        axis = np.stack([
            m[..., 2, 1] - m[..., 1, 2],
            m[..., 0, 2] - m[..., 2, 0],
            m[..., 1, 0] - m[..., 0, 1]
        ], axis=-1)

        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = axis / np.maximum(norm, ZERO)
        return axis.astype(FLOAT), angle[..., None].astype(FLOAT)
    
    # ------------------------------------------------------------------
    # Constructors from quaternion and axis-angle
    # ------------------------------------------------------------------

    @classmethod
    def FromQuaternion(cls, quat: np.ndarray, translation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)) -> "Transformation":
        """Create transformation(s) from quaternion(s).

        Parameters
        ----------
        quat : array_like (..., 4)
            Quaternion(s) in (x, y, z, w) format.
        translation : array_like (..., 3), optional
            Translation vector(s).
        scale : array_like (..., 3), optional
            Scale vector(s).

        Returns
        -------
        Transformation
        """
        quat = np.asarray(quat, dtype=FLOAT)
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        rot = np.empty(quat.shape[:-1] + (3, 3), dtype=FLOAT)
        rot[..., 0, 0] = 1 - 2 * (yy + zz)
        rot[..., 0, 1] = 2 * (xy - wz)
        rot[..., 0, 2] = 2 * (xz + wy)
        rot[..., 1, 0] = 2 * (xy + wz)
        rot[..., 1, 1] = 1 - 2 * (xx + zz)
        rot[..., 1, 2] = 2 * (yz - wx)
        rot[..., 2, 0] = 2 * (xz - wy)
        rot[..., 2, 1] = 2 * (yz + wx)
        rot[..., 2, 2] = 1 - 2 * (xx + yy)

        return cls.Combine(translation=translation, rotation=rot, scale=scale)

    @classmethod
    def FromAxisAngle(cls, axis: np.ndarray, angle: np.ndarray, translation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)) -> "Transformation":
        """Create transformation(s) from axis-angle representation.

        Parameters
        ----------
        axis : array_like (..., 3)
            Normalized axis of rotation.
        angle : array_like (..., 1) or (...,)
            Angle in radians.
        translation : array_like (..., 3), optional
            Translation vector(s).
        scale : array_like (..., 3), optional
            Scale vector(s).

        Returns
        -------
        Transformation
        """
        axis = np.asarray(axis, dtype=FLOAT)
        angle = np.asarray(angle, dtype=FLOAT)[..., None]  # ensure (..., 1)

        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        c = np.cos(angle[..., 0])
        s = np.sin(angle[..., 0])
        t = 1 - c

        rot = np.empty(axis.shape[:-1] + (3, 3), dtype=FLOAT)
        rot[..., 0, 0] = t * x * x + c
        rot[..., 0, 1] = t * x * y - s * z
        rot[..., 0, 2] = t * x * z + s * y
        rot[..., 1, 0] = t * x * y + s * z
        rot[..., 1, 1] = t * y * y + c
        rot[..., 1, 2] = t * y * z - s * x
        rot[..., 2, 0] = t * x * z - s * y
        rot[..., 2, 1] = t * y * z + s * x
        rot[..., 2, 2] = t * z * z + c

        return cls.Combine(translation=translation, rotation=rot, scale=scale)
    
    # ------------------------------------------------------------------
    # Constructor Euler angles
    # ------------------------------------------------------------------

    @classmethod
    def FromEuler(cls, angles: np.ndarray, order: str = 'XYZ', translation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), degrees: bool = False) -> "Transformation":
        """Create transformation(s) from Euler angles.

        Parameters
        ----------
        angles : array_like (..., 3)
            Euler angles in the specified order.
        order : str, default 'XYZ'
            Order of axis rotations (e.g. 'XYZ', 'ZYX', ...)
        translation : array_like (..., 3), optional
            Translation vector(s).
        scale : array_like (..., 3), optional
            Scale vector(s).
        degrees : bool, default False
            If True, input angles are interpreted in degrees.

        Returns
        -------
        Transformation
        """
        angles = np.asarray(angles, dtype=FLOAT)
        if degrees:
            angles = np.radians(angles)
        if angles.shape[-1] != 3:
            raise ValueError("Euler angles must have shape (..., 3)")

        def single_axis_matrix(axis, theta):
            c = np.cos(theta)
            s = np.sin(theta)
            eye = np.eye(3, dtype=FLOAT)
            mat = np.broadcast_to(eye, theta.shape + (3, 3)).copy()
            i = {'X': 0, 'Y': 1, 'Z': 2}[axis.upper()]
            j, k = (i + 1) % 3, (i + 2) % 3
            mat[..., i, i] = 1
            mat[..., j, j] = c
            mat[..., j, k] = -s
            mat[..., k, j] = s
            mat[..., k, k] = c
            return mat

        r = single_axis_matrix(order[0], angles[..., 0])
        for ax, th in zip(order[1:], angles[..., 1:].transpose(-1, *range(angles.ndim - 1))):
            r_next = single_axis_matrix(ax, th)
            r = np.matmul(r, r_next)

        return cls.Combine(translation=translation, rotation=r, scale=scale)

    def to_euler(self, order: str = 'XYZ', degrees: bool = False) -> np.ndarray:
        """Extract Euler angles from rotation matrix.

        Parameters
        ----------
        order : str, default 'XYZ'
            Rotation axis order.
        degrees : bool, default False
            If True, return angles in degrees.

        Returns
        -------
        angles : ndarray (..., 3)
            Euler angles.
        """
        r, _, _ = self.decompose()
        if order.upper() != 'XYZ':
            # Fallback to SciPy
            from scipy.spatial.transform import Rotation as R
            flat_r = r.reshape(-1, 3, 3)
            rot = R.from_matrix(flat_r.astype(np.float64))
            angles = rot.as_euler(order, degrees=degrees)
            return angles.reshape(r.shape[:-2] + (3,)).astype(FLOAT)

        # Optimized 'XYZ' extraction
        r00, r01, r02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
        r10, r11, r12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
        r20, r21, r22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]

        sy = np.clip(-r20, -1.0, 1.0)
        pitch = np.arcsin(sy)
        cos_pitch = np.cos(pitch)

        # Avoid division by zero near gimbal lock
        near_zero = np.abs(cos_pitch) < ZERO

        yaw = np.where(near_zero, np.arctan2(-r01, r11), np.arctan2(r10, r00))
        roll = np.where(near_zero, 0.0, np.arctan2(r21, r22))

        angles = np.stack([yaw, pitch, roll], axis=-1)  # (..., 3)
        if degrees:
            angles = np.degrees(angles)

        return angles.astype(FLOAT)
    
    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    @classmethod
    def Interpolate(cls, A: "Transformation", B: "Transformation", t: float | np.ndarray) -> "Transformation":
        """Interpolate between two transformations A and B using factor t.

        Parameters
        ----------
        A, B : Transformation
            Input transformations.
        t : float or array_like (...,)
            Interpolation weight(s) in [0, 1]. Shape must be broadcastable.

        Returns
        -------
        Transformation
            Interpolated transformation(s).
        """
        t = np.asarray(t, dtype=FLOAT)

        # Decompose A and B
        Ra, Sa, Ta = A.decompose()
        Rb, Sb, Tb = B.decompose()

        # Interpolate translation and scale
        T_interp = Ta + (Tb - Ta) * t[..., None]
        S_interp = Sa + (Sb - Sa) * t[..., None]

        # Interpolate rotation via SLERP (quaternions)
        qa = A.to_quaternion()
        qb = B.to_quaternion()

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
        return cls.FromQuaternion(q_interp, translation=T_interp, scale=S_interp)
    
    # ------------------------------------------------------------------
    # Align vectors
    # ------------------------------------------------------------------

    @staticmethod
    def align_vectors(
        source: np.ndarray,
        target: np.ndarray,
        *,
        up_axis: str = "z",
        upwards: np.ndarray = (0.0, 0.0, 1.0)
    ) -> np.ndarray:
        """
        Return rotation matrices that align each `source` vector to the corresponding `target` vector,
        keeping the given axis (e.g. 'z') aligned as close as possible to `upwards`.

        Parameters
        ----------
        source : array_like (..., 3)
            Source direction vectors.
        target : array_like (..., 3)
            Target direction vectors to align to.
        up_axis : str, default 'z'
            Axis that should stay pointing upwards. Can be one of: 'x', 'y', 'z', '-x', '-y', '-z'.
        upwards : array_like (..., 3), default (0, 0, 1)
            Desired global upwards direction. Will be used to constrain roll.

        Returns
        -------
        rot : ndarray (..., 3, 3)
            Rotation matrices aligning source to target and preserving orientation w.r.t. upwards.
        """
        source = np.asarray(source, dtype=FLOAT)
        target = np.asarray(target, dtype=FLOAT)
        upwards = np.asarray(upwards, dtype=FLOAT)

        # Normalize source and target
        source_norm = source / np.maximum(np.linalg.norm(source, axis=-1, keepdims=True), ZERO)
        target_norm = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), ZERO)
        upwards_norm = upwards / np.maximum(np.linalg.norm(upwards, axis=-1, keepdims=True), ZERO)

        # Map up_axis to index and sign
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax_str = up_axis.lower().lstrip('-')
        sign = -1 if up_axis.startswith('-') else 1
        if ax_str not in axis_map:
            raise ValueError("Invalid up_axis. Must be one of: 'x', 'y', 'z', '-x', '-y', '-z'")
        ax_idx = axis_map[ax_str]

        def make_frame(forward, up, axis_idx, sign):
            # force up orthogonal to forward
            side = np.cross(forward, up)
            side /= np.maximum(np.linalg.norm(side, axis=-1, keepdims=True), ZERO)
            new_up = np.cross(side, forward)
            frame = np.empty(forward.shape[:-1] + (3, 3), dtype=FLOAT)
            for i in range(3):
                frame[..., i, :] = 0.0
            frame[..., axis_idx, :] = sign * new_up
            remain = [i for i in range(3) if i != axis_idx]
            frame[..., remain[0], :] = side
            frame[..., remain[1], :] = forward
            return frame

        # Create source and target frames
        R_source = make_frame(source_norm, upwards_norm, ax_idx, sign)
        R_target = make_frame(target_norm, upwards_norm, ax_idx, sign)

        # Final rotation: target frame × transpose of source frame
        R = np.matmul(R_target, np.swapaxes(R_source, -1, -2))
        return R
    
    # ------------------------------------------------------------------
    # Track to
    # ------------------------------------------------------------------

    @classmethod
    def TrackTo(
        cls,
        points: "Transformation",
        target: np.ndarray,
        *,
        track_axis: str = "z",
        up_axis: str = "y",
        upwards: np.ndarray = (0.0, 0.0, 1.0)
    ) -> "Transformation":
        """
        Return a rotation transformation that re-orients each object in `points` to face `target`.

        Parameters
        ----------
        points : Transformation
            Current transformation(s), used to extract position and current axes.
        target : array_like (..., 3)
            Point(s) to look at.
        track_axis : str, default 'z'
            Local axis (e.g. 'x', 'y', 'z', '-x', ...) that must face towards the target.
        up_axis : str, default 'y'
            Local axis to align with the global 'upwards' direction.
        upwards : array_like (..., 3), default (0, 0, 1)
            Desired global "up" vector.

        Returns
        -------
        Transformation
            A new transformation with updated rotation, same position.
        """
        target = np.asarray(target, dtype=FLOAT)
        upwards = np.asarray(upwards, dtype=FLOAT)

        # Current object position and rotation
        r, _, t = points.decompose()  # r: (..., 3, 3), t: (..., 3)

        # Vector from each point to its target
        direction = target - t  # (..., 3)

        # Normalize direction
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), ZERO)

        # Determine source vectors in world coordinates
        def axis_vector(name: str) -> np.ndarray:
            name = name.lower()
            sign = -1.0 if name.startswith("-") else 1.0
            base = name[-1]
            idx = {"x": 0, "y": 1, "z": 2}[base]
            vec = np.zeros(r.shape[:-2] + (3,), dtype=FLOAT)
            vec[..., idx] = sign
            return vec

        local_track = axis_vector(track_axis)   # (..., 3)
        local_up = axis_vector(up_axis)         # (..., 3)

        # Convert these to world space
        world_track = np.einsum("...ij,...j->...i", r, local_track)  # (..., 3)
        world_up = np.einsum("...ij,...j->...i", r, local_up)        # (..., 3)

        # Compute rotation to align (world_track → direction), keeping world_up ≈ upwards
        rot = cls.AlignVectors(world_track, direction, up_axis=up_axis, upwards=upwards)

        # Final transformation: new rotation + same translation
        return cls.Combine(rotation=rot, translation=t)
    
    # ---------------------------------------------------------------------------
    # To Euler angles
    # ---------------------------------------------------------------------------
    
    def test_eulers(self, order='XYZ'):

        # Matrices as a linear array of matrices
        ms = self.rotation.reshape(self.size, 3, 3)

        TRANSPOSE = True
        if TRANSPOSE:
            ms = np.transpose(ms, axes=(0, 2, 1))

            
        # ---------------------------------------------------------------------------
        # Indices in the array to compute the angles
    
        if order == 'XYZ':
    
            # cz.cy              | cz.sy.sx - sz.cx   | cz.sy.cx + sz.sx
            # sz.cy              | sz.sy.sx + cz.cx   | sz.sy.cx - cz.sx
            # -sy                | cy.sx              | cy.cx
    
            xyz = [1, 0, 2]
    
            ls0, cs0, sgn = (2, 0, -1)          # sy
            ls1, cs1, lc1, cc1 = (2, 1, 2, 2)   # cy.sx cy.cx
            ls2, cs2, lc2, cc2 = (1, 0, 0, 0)   # cy.sz cy.cz
    
            ls3, cs3, lc3, cc3 = (0, 1, 1, 1)   
    
        elif order == 'XZY':
    
            # cy.cz              | -cy.sz.cx + sy.sx  | cy.sz.sx + sy.cx
            # sz                 | cz.cx              | -cz.sx
            # -sy.cz             | sy.sz.cx + cy.sx   | -sy.sz.sx + cy.cx
    
            xyz = [2, 0, 1]
    
            ls0, cs0, sgn = (1, 0, +1)
            ls1, cs1, lc1, cc1 = (1, 2, 1, 1)
            ls2, cs2, lc2, cc2 = (2, 0, 0, 0)
    
            ls3, cs3, lc3, cc3 = (0, 2, 2, 2)
    
        elif order == 'YXZ':
    
            # cz.cy - sz.sx.sy   | -sz.cx             | cz.sy + sz.sx.cy
            # sz.cy + cz.sx.sy   | cz.cx              | sz.sy - cz.sx.cy
            # -cx.sy             | sx                 | cx.cy
    
            xyz = [0, 1, 2]
    
            ls0, cs0, sgn = (2, 1, +1)
            ls1, cs1, lc1, cc1 = (2, 0, 2, 2)
            ls2, cs2, lc2, cc2 = (0, 1, 1, 1)
    
            ls3, cs3, lc3, cc3 = (1, 0, 0, 0)
    
        elif order == 'YZX':
    
            # cz.cy              | -sz                | cz.sy
            # cx.sz.cy + sx.sy   | cx.cz              | cx.sz.sy - sx.cy
            # sx.sz.cy - cx.sy   | sx.cz              | sx.sz.sy + cx.cy
    
            xyz = [2, 1, 0]
            
    
            ls0, cs0, sgn = (0, 1, -1)
            ls1, cs1, lc1, cc1 = (0, 2, 0, 0)
            ls2, cs2, lc2, cc2 = (2, 1, 1, 1)
    
            ls3, cs3, lc3, cc3 = (1, 2, 2, 2)
    
        elif order == 'ZXY':
    
            # cy.cz + sy.sx.sz   | -cy.sz + sy.sx.cz  | sy.cx
            # cx.sz              | cx.cz              | -sx
            # -sy.cz + cy.sx.sz  | sy.sz + cy.sx.cz   | cy.cx
    
            xyz = [0, 2, 1]
    
            ls0, cs0, sgn = (1, 2, -1)
            ls1, cs1, lc1, cc1 = (1, 0, 1, 1)
            ls2, cs2, lc2, cc2 = (0, 2, 2, 2)
    
            ls3, cs3, lc3, cc3 = (2, 0, 0, 0)
    
        elif order == 'ZYX':
    
            # cy.cz              | -cy.sz             | sy
            # cx.sz + sx.sy.cz   | cx.cz - sx.sy.sz   | -sx.cy
            # sx.sz - cx.sy.cz   | sx.cz + cx.sy.sz   | cx.cy
    
            xyz = [1, 2, 0]
    
            ls0, cs0, sgn = (0, 2, +1)
            ls1, cs1, lc1, cc1 = (0, 1, 0, 0)
            ls2, cs2, lc2, cc2 = (1, 2, 2, 2)
    
            ls3, cs3, lc3, cc3 = (2, 1, 1, 1)
    
        else:
            raise self.error(f"Conversion to eulers error: '{order}' is not a valid euler order")
            
        # ---------------------------------------------------------------------------
        # Compute the euler angles
    
        angles = np.zeros((len(ms), 3), FLOAT)   # Place holder for the angles in the order of their computation
        
        # Computation depends upon sin(angle 0) == ±1
    
        neg_1  = np.where(np.abs(ms[:, cs0, ls0] + 1) < ZERO)[0] # sin(angle 0) = -1
        pos_1  = np.where(np.abs(ms[:, cs0, ls0] - 1) < ZERO)[0] # sin(angle 0) = +1
        rem    = np.delete(np.arange(len(ms)), np.concatenate((neg_1, pos_1)))
        
        if len(neg_1) > 0:
            angles[neg_1, xyz[0]] = -np.pi/2 * sgn
            angles[neg_1, xyz[1]] = 0
            angles[neg_1, xyz[2]] = np.arctan2(sgn * ms[neg_1, cs3, ls3], ms[neg_1, cc3, lc3])
    
        if len(pos_1) > 0:
            angles[pos_1, xyz[0]] = np.pi/2 * sgn
            angles[pos_1, xyz[1]] = 0
            angles[pos_1, xyz[2]] = np.arctan2(sgn * ms[pos_1, cs3, ls3], ms[pos_1, cc3, lc3])
    
        if len(rem) > 0:
            angles[rem, xyz[0]] = sgn * np.clip(np.arcsin(ms[rem, cs0, ls0]), -1, 1)
            angles[rem, xyz[1]] = np.arctan2(-sgn * ms[rem, cs1, ls1], ms[rem, cc1, lc1])
            angles[rem, xyz[2]] = np.arctan2(-sgn * ms[rem, cs2, ls2], ms[rem, cc2, lc2])
            
        # ---------------------------------------------------------------------------
        # At this stage, the result could be two 180 angles and a value ag
        # This is equivalent to two 0 values and 180-ag
        # Let's correct this
        
        # -180° --> 180°
            
        angles[abs(angles + np.pi) < ZERO] = np.pi
        
        # Let's change where we have two 180 angles
        
        idx = np.where(np.logical_and(abs(angles[:, 0]-np.pi) < ZERO, abs(angles[:, 1]-np.pi) < ZERO))[0]
        angles[idx, 0] = 0
        angles[idx, 1] = 0
        angles[idx, 2] = np.pi - angles[idx, 2]
        
        idx = np.where(np.logical_and(abs(angles[:, 0]-np.pi) < ZERO, abs(angles[:, 2]-np.pi) < ZERO))[0]
        angles[idx, 0] = 0
        angles[idx, 2] = 0
        angles[idx, 1] = np.pi - angles[idx, 1]
        
        idx = np.where(np.logical_and(abs(angles[:, 1]-np.pi) < ZERO, abs(angles[:, 2]-np.pi) < ZERO))[0]
        angles[idx, 1] = 0
        angles[idx, 2] = 0
        angles[idx, 0] = np.pi - angles[idx, 0]
        
        # ---------------------------------------------------------------------------
        # Returns the result
        
        return np.reshape(angles, self.shape + (3,)) 








import itertools
import time
import matplotlib.pyplot as plt

import numpy as np
import time

N = 100_000  # Nombre de rotations testées

# -------------------------------
# Génération d'angles Euler (grille des combinaisons typiques)
# -------------------------------
angles = np.deg2rad([0, 90, 180, 270])
combinations = np.array(np.meshgrid(angles, angles, angles)).T.reshape(-1, 3).astype(FLOAT)
print(f"Nombre de combinaisons testées : {len(combinations)}")

# -------------------------------
# Boucle sur les ordres Euler
# -------------------------------
for order in ['XYZ', 'ZYX']:
    print(f"\n=== Test {order} ===")

    # Création de la transformation
    tf0 = Transformation.FromEuler(combinations, order=order)

    # Reconstruction des angles via test_eulers
    start = time.perf_counter()
    euler_recon = tf0.test_eulers(order=order)
    print(f"[test_eulers] {time.perf_counter() - start:.4f} s")

    # Reconstruire la transformation à partir des angles retrouvés
    tf1 = Transformation.FromEuler(euler_recon, order=order)

    # Comparaison sur des vecteurs 3D aléatoires (fixes pour reproductibilité)
    np.random.seed(42)
    vecs = np.random.randn(10, 3).astype(FLOAT)
    vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
    vecs_b = np.broadcast_to(vecs, (len(combinations), 10, 3))

    # Appliquer les transformations
    out0 = tf0.reshape(len(combinations), 1).transform(vecs_b)
    out1 = tf1.reshape(len(combinations), 1).transform(vecs_b)

    # Erreur angulaire par norme de différence
    errors = np.linalg.norm(out0 - out1, axis=-1)  # shape (N, 10)
    max_error = np.max(errors)
    mean_error = np.mean(errors)

    # Histogramme des erreurs si besoin
    hist, bins = np.histogram(errors, bins=20, range=(0, 2))
    print("Erreur angulaire sur vecteurs :")
    print(f"  Max error :  {max_error:.6f}")
    print(f"  Mean error:  {mean_error:.6f}")
    print(f"  Distribution (histogramme):\n{hist}")









