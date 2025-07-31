# npblender/maths/rotation.py
# MIT License
# Created on 2022-11-11
# Last update: 2025-07-21
# Author: Alain Bernard

"""
Rotation
========

Rotation class based on 3×3 orthogonal matrices (SO(3)) with batch support.

This class wraps a batch of rotation matrices and provides:
- Construction from multiple representations: matrix, quaternion, Euler, axis–angle
- Conversion to other representations: matrix, quaternion, Euler, axis–angle
- Composition, inversion, and application to vectors
- Full support for NumPy broadcasting and vectorized operations

All instances are backed by a NumPy array of shape (..., 3, 3), with strict
control over validity (orthogonality, determinant) and optional batch processing.

Example:

    >>> R = Rotation.from_axis_angle([0, 1, 0], np.pi / 4)
    >>> v = np.array([1, 0, 0])
    >>> R @ v
    array([0.7071, 0.0, -0.7071])
"""


import numpy as np

if __name__ == '__main__':
    from itemsarray import ItemsArray
else:
    from . itemsarray import ItemsArray

# for as_euler and from_euler
TRANSPOSE = True


# ====================================================================================================
# Rotation
# ====================================================================================================

class Rotation(ItemsArray):
    """
    Rotation represented as 3×3 matrices.

    This class wraps batches of rotation matrices and provides
    methods for applying, composing and inverting them.

    All rotations must have shape (..., 3, 3) and be valid rotation matrices.
    """

    _item_shape = (3, 3)

    # ====================================================================================================
    # Constructors
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Identity
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def identity(cls, shape=(), dtype=None):
        """Create an identity rotation for the given batch shape."""
        dtype = dtype or cls.FLOAT
        mat = np.broadcast_to(np.eye(3, dtype=dtype), shape + (3, 3)).copy()
        return cls(mat, copy=False)

    # ----------------------------------------------------------------------------------------------------
    # From matrix
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_matrix(cls, mat, *, validate: bool = True, tol: float = 1e-5) -> "Rotation":
        """
        Construct a Rotation from raw 3×3 matrices.

        Parameters
        ----------
        mat : array_like (..., 3, 3)
            One or more candidate rotation matrices.

        validate : bool, default True
            If True, checks that matrices are orthogonal and have determinant close to 1.

        tol : float, default 1e-5
            Tolerance used for validation.

        Returns
        -------
        Rotation
            New instance wrapping the validated (or unvalidated) matrices.

        Raises
        ------
        ValueError
            If validation is enabled and any matrix is not a proper rotation.
        """
        mat = np.asarray(mat, dtype=cls.FLOAT)

        if validate:
            if mat.shape[-2:] != (3, 3):
                raise ValueError("Expected matrices of shape (..., 3, 3)")

            # Orthogonality: R.T @ R ≈ I
            should_be_identity = mat @ np.swapaxes(mat, -1, -2)
            identity = np.eye(3, dtype=cls.FLOAT)
            error = np.abs(should_be_identity - identity).max(axis=(-2, -1))
            if np.any(error > tol):
                raise ValueError("Rotation matrix is not orthogonal within tolerance.")

            # Determinant ≈ +1
            det = np.linalg.det(mat)
            if np.any(np.abs(det - 1.0) > tol):
                raise ValueError("Rotation matrix determinant not close to 1.")

        return cls(mat, copy=True)
    
    # ----------------------------------------------------------------------------------------------------
    # From quaternion
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_quaternion(cls, quat, *, normalize: bool = True, tol: float = 1e-5) -> "Rotation":
        """
        Construct a Rotation from unit quaternions (x, y, z, w convention).

        Parameters
        ----------
        quat : array_like (..., 4)
            Input quaternions in (x, y, z, w) order. Must be broadcastable.

        normalize : bool, default True
            If True, normalizes the input quaternions to unit norm.

        tol : float, default 1e-5
            Tolerance used to check if quaternions are close to unit length (if normalize=False).

        Returns
        -------
        Rotation
            Rotation instance representing the quaternions.
        """
        quat = np.asarray(quat, dtype=cls.FLOAT)

        if quat.shape[-1] != 4:
            raise ValueError("Quaternions must have shape (..., 4)")

        if normalize:
            quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
        else:
            norms = np.linalg.norm(quat, axis=-1)
            if np.any(np.abs(norms - 1.0) > tol):
                raise ValueError("Quaternion norms not within tolerance of 1.0")

        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        # Rotation matrix formula for (x, y, z, w)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        xw, yw, zw = x * w, y * w, z * w

        mat = np.empty(quat.shape[:-1] + (3, 3), dtype=cls.FLOAT)

        mat[..., 0, 0] = 1 - 2 * (yy + zz)
        mat[..., 0, 1] = 2 * (xy - zw)
        mat[..., 0, 2] = 2 * (xz + yw)

        mat[..., 1, 0] = 2 * (xy + zw)
        mat[..., 1, 1] = 1 - 2 * (xx + zz)
        mat[..., 1, 2] = 2 * (yz - xw)

        mat[..., 2, 0] = 2 * (xz - yw)
        mat[..., 2, 1] = 2 * (yz + xw)
        mat[..., 2, 2] = 1 - 2 * (xx + yy)

        return cls(mat, copy=False)



    @classmethod
    def from_quaternion_MAUVAISE_CONVENTION(cls, quat, *, normalize: bool = True, tol: float = 1e-5) -> "Rotation":
        """
        Construct a Rotation from unit quaternions (xyzw convention).

        Parameters
        ----------
        quat : array_like (..., 4)
            Input quaternions in (x, y, z, w) order. Must be broadcastable.

        normalize : bool, default True
            If True, normalizes the input quaternions to unit norm.

        tol : float, default 1e-5
            Tolerance used to check if quaternions are close to unit length (if normalize=False).

        Returns
        -------
        Rotation
            Rotation instance representing the quaternions.

        Raises
        ------
        ValueError
            If input shape is invalid or quaternions are not unit (when normalize=False).
        """
        quat = np.asarray(quat, dtype=cls.FLOAT)

        if quat.shape[-1] != 4:
            raise ValueError("Quaternions must have shape (..., 4)")

        if normalize:
            quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
        else:
            norms = np.linalg.norm(quat, axis=-1)
            if np.any(np.abs(norms - 1.0) > tol):
                raise ValueError("Quaternion norms not within tolerance of 1.0")

        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        xx = x * x
        yy = y * y
        zz = z * z
        ww = w * w

        xy = x * y
        xz = x * z
        yz = y * z
        xw = x * w
        yw = y * w
        zw = z * w

        # Rotation matrix formula (xyzw convention)
        mat = np.empty(quat.shape[:-1] + (3, 3), dtype=cls.FLOAT)
        mat[..., 0, 0] = 1 - 2 * (yy + zz)
        mat[..., 0, 1] = 2 * (xy - zw)
        mat[..., 0, 2] = 2 * (xz + yw)

        mat[..., 1, 0] = 2 * (xy + zw)
        mat[..., 1, 1] = 1 - 2 * (xx + zz)
        mat[..., 1, 2] = 2 * (yz - xw)

        mat[..., 2, 0] = 2 * (xz - yw)
        mat[..., 2, 1] = 2 * (yz + xw)
        mat[..., 2, 2] = 1 - 2 * (xx + yy)

        return cls(mat, copy=False)
    
    # --------------------------------------------------------------------------------------
    #  From axis angle
    # --------------------------------------------------------------------------------------

    @classmethod
    def from_axis_angle(
        cls,
        axis,
        angle,
        *,
        normalize_axis: bool = True,
        degrees: bool = False,
    ) -> "Rotation":
        """
        Construct a rotation from an axis–angle pair.

        Parameters
        ----------
        axis : array_like (..., 3)
            Rotation axis. Does **not** need to be unit length if `normalize_axis=True`.
        angle : array_like (...,)
            Rotation angle, broadcastable to the batch shape of `axis`.
        normalize_axis : bool, default True
            Normalise the axis to unit length before constructing the matrix.
        degrees : bool, default False
            If True, `angle` is interpreted in degrees instead of radians.

        Returns
        -------
        Rotation
            Rotation instance representing the axis–angle pair.

        Notes
        -----
        • Uses the right‑hand rule.  
        • Supports arbitrary batch shapes; `axis` and `angle` are broadcast together.
        """
        axis = np.asarray(axis, dtype=cls.FLOAT)
        angle = np.asarray(angle, dtype=cls.FLOAT)

        if axis.shape[-1] != 3:
            raise ValueError("Axis must have shape (..., 3)")

        if degrees:
            angle = np.deg2rad(angle)

        if normalize_axis:
            axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)

        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1.0 - c

        mat = np.empty(axis.shape[:-1] + (3, 3), dtype=cls.FLOAT)
        mat[..., 0, 0] = t * x * x + c
        mat[..., 0, 1] = t * x * y - s * z
        mat[..., 0, 2] = t * x * z + s * y

        mat[..., 1, 0] = t * x * y + s * z
        mat[..., 1, 1] = t * y * y + c
        mat[..., 1, 2] = t * y * z - s * x

        mat[..., 2, 0] = t * x * z - s * y
        mat[..., 2, 1] = t * y * z + s * x
        mat[..., 2, 2] = t * z * z + c

        return cls(mat, copy=False)

    # ----------------------------------------------------------------------------------------------------
    # From euler
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_euler(
        cls,
        euler,
        *,
        order: str = "XYZ",
        degrees: bool = False,
    ) -> "Rotation":
        
        euler = np.asarray(euler, dtype=cls.FLOAT)
        if euler.shape[-1] != 3:
            raise ValueError("Euler angle array must have shape (..., 3)")

        if degrees:
            euler = np.deg2rad(euler)

        shape = euler.shape[:-1]
        count = int(np.prod(shape))
        es = euler.reshape(count, 3) # Short
        
        # ----- Let's go
    
        m = np.zeros((count, 3, 3), float)
    
        cx = np.cos(es[:, 0])
        sx = np.sin(es[:, 0])
        cy = np.cos(es[:, 1])
        sy = np.sin(es[:, 1])
        cz = np.cos(es[:, 2])
        sz = np.sin(es[:, 2])
        
        if order == 'XYZ':
            m[:, 0, 0] = cz*cy
            m[:, 1, 0] = cz*sy*sx - sz*cx
            m[:, 2, 0] = cz*sy*cx + sz*sx
            m[:, 0, 1] = sz*cy
            m[:, 1, 1] = sz*sy*sx + cz*cx
            m[:, 2, 1] = sz*sy*cx - cz*sx
            m[:, 0, 2] = -sy
            m[:, 1, 2] = cy*sx
            m[:, 2, 2] = cy*cx
    
        elif order == 'XZY':
            m[:, 0, 0] = cy*cz
            m[:, 1, 0] = -cy*sz*cx + sy*sx
            m[:, 2, 0] = cy*sz*sx + sy*cx
            m[:, 0, 1] = sz
            m[:, 1, 1] = cz*cx
            m[:, 2, 1] = -cz*sx
            m[:, 0, 2] = -sy*cz
            m[:, 1, 2] = sy*sz*cx + cy*sx
            m[:, 2, 2] = -sy*sz*sx + cy*cx
    
        elif order == 'YXZ':
            m[:, 0, 0] = cz*cy - sz*sx*sy
            m[:, 1, 0] = -sz*cx
            m[:, 2, 0] = cz*sy + sz*sx*cy
            m[:, 0, 1] = sz*cy + cz*sx*sy
            m[:, 1, 1] = cz*cx
            m[:, 2, 1] = sz*sy - cz*sx*cy
            m[:, 0, 2] = -cx*sy
            m[:, 1, 2] = sx
            m[:, 2, 2] = cx*cy
    
        elif order == 'YZX':
            m[:, 0, 0] = cz*cy
            m[:, 1, 0] = -sz
            m[:, 2, 0] = cz*sy
            m[:, 0, 1] = cx*sz*cy + sx*sy
            m[:, 1, 1] = cx*cz
            m[:, 2, 1] = cx*sz*sy - sx*cy
            m[:, 0, 2] = sx*sz*cy - cx*sy
            m[:, 1, 2] = sx*cz
            m[:, 2, 2] = sx*sz*sy + cx*cy
    
        elif order == 'ZXY':
            m[:, 0, 0] = cy*cz + sy*sx*sz
            m[:, 1, 0] = -cy*sz + sy*sx*cz
            m[:, 2, 0] = sy*cx
            m[:, 0, 1] = cx*sz
            m[:, 1, 1] = cx*cz
            m[:, 2, 1] = -sx
            m[:, 0, 2] = -sy*cz + cy*sx*sz
            m[:, 1, 2] = sy*sz + cy*sx*cz
            m[:, 2, 2] = cy*cx
    
        elif order == 'ZYX':
            m[:, 0, 0] = cy*cz
            m[:, 1, 0] = -cy*sz
            m[:, 2, 0] = sy
            m[:, 0, 1] = cx*sz + sx*sy*cz
            m[:, 1, 1] = cx*cz - sx*sy*sz
            m[:, 2, 1] = -sx*cy
            m[:, 0, 2] = sx*sz - cx*sy*cz
            m[:, 1, 2] = sx*cz + cx*sy*sz
            m[:, 2, 2] = cx*cy

        if TRANSPOSE:
            return cls(m.transpose((0, 2, 1)).reshape(shape + (3, 3)), copy=False)
        else:
            return cls(m.reshape(shape + (3, 3)), copy=False)    

    @classmethod
    def from_euler_OLD2(
        cls,
        euler,
        *,
        order: str = "XYZ",
        degrees: bool = False,
    ) -> "Rotation":
        """
        Construct a rotation from Euler angles using intrinsic XYZ order.
        """
        order = order.upper()
        if order != "XYZ":
            raise NotImplementedError("Only 'XYZ' intrinsic order is currently implemented.")

        euler = np.asarray(euler, dtype=cls.FLOAT)
        if euler.shape[-1] != 3:
            raise ValueError("Euler angle array must have shape (..., 3)")

        if degrees:
            euler = np.deg2rad(euler)

        x, y, z = np.moveaxis(euler, -1, 0)

        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        cz, sz = np.cos(z), np.sin(z)

        # R = Rz @ Ry @ Rx
        R = np.empty(euler.shape[:-1] + (3, 3), dtype=cls.FLOAT)
        R[..., 0, 0] = cy * cz
        R[..., 0, 1] = -cy * sz
        R[..., 0, 2] = sy
        R[..., 1, 0] = cx * sz + cz * sx * sy
        R[..., 1, 1] = cx * cz - sx * sy * sz
        R[..., 1, 2] = -cy * sx
        R[..., 2, 0] = sx * sz - cx * cz * sy
        R[..., 2, 1] = cz * sx + cx * sy * sz
        R[..., 2, 2] = cx * cy

        return cls(R, copy=False)




    @classmethod
    def from_euler_OLD(
        cls,
        euler,
        *,
        order: str = "XYZ",
        degrees: bool = False,
    ) -> "Rotation":
        """
        Construct a rotation from Euler/Tait‑Bryan angles.

        Parameters
        ----------
        euler : array_like (..., 3)
            Angles for the three axes, in the order specified by `order`.
        order : {'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'}, default 'XYZ'
            Axis order (uppercase = intrinsic / body‑fixed convention).
        degrees : bool, default False
            Interpret `euler` in degrees instead of radians.

        Returns
        -------
        Rotation
            Rotation instance representing the Euler angles.

        Raises
        ------
        ValueError
            If `order` is not one of the supported axis permutations.

        Notes
        -----
        • Uses the right‑hand rule for each elemental rotation.  
        • Broadcasting works: `euler` can have any leading batch shape.
        """

        if order == "XYZ":
            # R = Rz @ Ry @ Rx       (intrinsèque XYZ)
            y = -np.arcsin(np.clip(r[..., 2, 0], -1.0, 1.0))
            cos_y = np.cos(y)

            # Gimbal‑lock : |cos y| ≈ 0
            near_lock = cos_y < 1e-6

            x = np.empty_like(y)
            z = np.empty_like(y)

            # général
            x[~near_lock] = np.arctan2(r[..., 2, 1][~near_lock],
                                    r[..., 2, 2][~near_lock])
            z[~near_lock] = np.arctan2(r[..., 1, 0][~near_lock],
                                    r[..., 0, 0][~near_lock])

            # gimbal‑lock : z arbitraire, on fixe z = 0
            x[near_lock] = np.arctan2(-r[..., 0, 1][near_lock],
                                    r[..., 1, 1][near_lock])
            z[near_lock] = 0.0

            mat = np.stack([x, y, z], axis=-1)
            return cls(mat, copy=False)
        
        if order != "XYZ":
            raise NotImplementedError("Only XYZ order is currently implemented.")


        axes = {"X": 0, "Y": 1, "Z": 2}
        order = order.upper()
        if order not in {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"}:
            raise ValueError("Unsupported Euler order; choose from 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'.")

        euler = np.asarray(euler, dtype=cls.FLOAT)
        if euler.shape[-1] != 3:
            raise ValueError("Euler angle array must have shape (..., 3)")

        if degrees:
            euler = np.deg2rad(euler)

        batch_shape = euler.shape[:-1]
        mat = np.broadcast_to(np.eye(3, dtype=cls.FLOAT), batch_shape + (3, 3)).copy()

        # Helper to build single‑axis rotation matrices (vectorised)
        def _axis_rot(angle, axis_id):
            c = np.cos(angle)
            s = np.sin(angle)
            R = np.zeros(angle.shape + (3, 3), dtype=cls.FLOAT)
            ax = axis_id
            R[..., ax, ax] = 1
            R[..., (ax + 1) % 3, (ax + 1) % 3] = c
            R[..., (ax + 2) % 3, (ax + 2) % 3] = c
            R[..., (ax + 1) % 3, (ax + 2) % 3] = -s
            R[..., (ax + 2) % 3, (ax + 1) % 3] = s
            return R

        # Apply rotations in the specified order (intrinsic / body‑fixed)
        for angle, ax_letter in zip(euler.swapaxes(-1, 0), order):
            R_axis = _axis_rot(angle, axes[ax_letter])
            mat = R_axis @ mat

        return cls(mat, copy=False)
    
    # ====================================================================================================
    # As other types
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # As matrix
    # ----------------------------------------------------------------------------------------------------
    
    def as_matrix(self) -> "Rotation":
        """Return the underlying rotation matrices (view, no copy)."""
        return self

    # ----------------------------------------------------------------------------------------------------
    # As quaternion
    # ----------------------------------------------------------------------------------------------------

    def as_quaternion(self) -> "Quaternion":
        """
        Convert each rotation matrix to a quaternion (xyzw convention).
        
        Returns
        -------
        Quaternion
            Quaternions of shape (..., 4), in (x, y, z, w) order.
        """
        from . quaternion import Quaternion

        R = self._mat
        m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
        m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
        m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

        trace = m00 + m11 + m22
        q = np.empty(R.shape[:-2] + (4,), dtype=self.FLOAT)
        t = np.empty_like(trace)
        eps = 1e-8  # sécurité numérique

        # Case 1: trace > 0
        cond = trace > 0.0
        val = np.maximum(1.0 + trace[cond], 0.0)
        t[cond] = np.sqrt(val) * 2
        q[..., 3][cond] = 0.25 * t[cond]
        q[..., 0][cond] = (m21[cond] - m12[cond]) / (t[cond] + eps)
        q[..., 1][cond] = (m02[cond] - m20[cond]) / (t[cond] + eps)
        q[..., 2][cond] = (m10[cond] - m01[cond]) / (t[cond] + eps)

        # Case 2: m00 is largest
        cond2 = (m00 >= m11) & (m00 >= m22) & (~cond)
        val = np.maximum(1.0 + m00[cond2] - m11[cond2] - m22[cond2], 0.0)
        t[cond2] = np.sqrt(val) * 2
        q[..., 0][cond2] = 0.25 * t[cond2]
        q[..., 1][cond2] = (m01[cond2] + m10[cond2]) / (t[cond2] + eps)
        q[..., 2][cond2] = (m02[cond2] + m20[cond2]) / (t[cond2] + eps)
        q[..., 3][cond2] = (m21[cond2] - m12[cond2]) / (t[cond2] + eps)

        # Case 3: m11 is largest
        cond3 = (m11 > m00) & (m11 >= m22) & (~cond) & (~cond2)
        val = np.maximum(1.0 + m11[cond3] - m00[cond3] - m22[cond3], 0.0)
        t[cond3] = np.sqrt(val) * 2
        q[..., 0][cond3] = (m01[cond3] + m10[cond3]) / (t[cond3] + eps)
        q[..., 1][cond3] = 0.25 * t[cond3]
        q[..., 2][cond3] = (m12[cond3] + m21[cond3]) / (t[cond3] + eps)
        q[..., 3][cond3] = (m02[cond3] - m20[cond3]) / (t[cond3] + eps)

        # Case 4: m22 is largest
        cond4 = ~cond & ~cond2 & ~cond3
        val = np.maximum(1.0 + m22[cond4] - m00[cond4] - m11[cond4], 0.0)
        t[cond4] = np.sqrt(val) * 2
        q[..., 0][cond4] = (m02[cond4] + m20[cond4]) / (t[cond4] + eps)
        q[..., 1][cond4] = (m12[cond4] + m21[cond4]) / (t[cond4] + eps)
        q[..., 2][cond4] = 0.25 * t[cond4]
        q[..., 3][cond4] = (m10[cond4] - m01[cond4]) / (t[cond4] + eps)

        # Normalize
        q /= np.linalg.norm(q, axis=-1, keepdims=True)

        # Optionnel : trace les NaNs pour debug
        # if np.isnan(q).any():
        #     print("⚠️ NaNs in as_quaternion")

        return Quaternion(q, copy=False)
    
    # ----------------------------------------------------------------------------------------------------
    # As euler
    # ----------------------------------------------------------------------------------------------------

    def as_euler(self, order: str = "XYZ", degrees: bool = False) -> np.ndarray:
        """
        Convert each rotation matrix to Euler angles (XYZ intrinsic order).
        """
        count = len(self)
        ms = self._non_scalar_mat
        zero = 1e-6
        pi = np.pi
        
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
    
            xyz = [1, 0, 2]
            xyz = [2, 0, 1]
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
    
        angles = np.zeros((len(ms), 3), float)   # Place holder for the angles in the order of their computation
        
        # Computation depends upon sin(angle 0) == ±1
    
        neg_1  = np.where(np.abs(ms[:, cs0, ls0] + 1) < zero)[0] # sin(angle 0) = -1
        pos_1  = np.where(np.abs(ms[:, cs0, ls0] - 1) < zero)[0] # sin(angle 0) = +1
        rem    = np.delete(np.arange(len(ms)), np.concatenate((neg_1, pos_1)))
        
        if len(neg_1) > 0:
            angles[neg_1, xyz[0]] = -pi/2 * sgn
            angles[neg_1, xyz[1]] = 0
            angles[neg_1, xyz[2]] = np.arctan2(sgn * ms[neg_1, cs3, ls3], ms[neg_1, cc3, lc3])
    
        if len(pos_1) > 0:
            angles[pos_1, xyz[0]] = pi/2 * sgn
            angles[pos_1, xyz[1]] = 0
            angles[pos_1, xyz[2]] = np.arctan2(sgn * ms[pos_1, cs3, ls3], ms[pos_1, cc3, lc3])
    
        if len(rem) > 0:
            angles[rem, xyz[0]] = sgn * np.arcsin(ms[rem, cs0, ls0])
            angles[rem, xyz[1]] = np.arctan2(-sgn * ms[rem, cs1, ls1], ms[rem, cc1, lc1])
            angles[rem, xyz[2]] = np.arctan2(-sgn * ms[rem, cs2, ls2], ms[rem, cc2, lc2])
            
        # ---------------------------------------------------------------------------
        # At this stage, the result could be two 180 angles and a value ag
        # This is equivalent to two 0 values and 180-ag
        # Let's correct this
        
        # -180° --> 180°
            
        angles[abs(angles+np.pi) < zero] = np.pi
        
        # Let's change where we have two 180 angles
        
        idx = np.where(np.logical_and(abs(angles[:, 0]-np.pi) < zero, abs(angles[:, 1]-np.pi) < zero))[0]
        angles[idx, 0] = 0
        angles[idx, 1] = 0
        angles[idx, 2] = np.pi - angles[idx, 2]
        
        idx = np.where(np.logical_and(abs(angles[:, 0]-np.pi) < zero, abs(angles[:, 2]-np.pi) < zero))[0]
        angles[idx, 0] = 0
        angles[idx, 2] = 0
        angles[idx, 1] = np.pi - angles[idx, 1]
        
        idx = np.where(np.logical_and(abs(angles[:, 1]-np.pi) < zero, abs(angles[:, 2]-np.pi) < zero))[0]
        angles[idx, 1] = 0
        angles[idx, 2] = 0
        angles[idx, 0] = np.pi - angles[idx, 0]
        
        # ---------------------------------------------------------------------------
        # Returns the result

        if degrees:
            angles = np.rad2deg(angles)

        return np.reshape(angles, self.shape + (3,))
    
    # ----------------------------------------------------------------------------------------------------
    # As axis angle
    # ----------------------------------------------------------------------------------------------------

    def as_axis_angle(self, tol: float = 1e-6, degrees: bool = False):
        """
        Convert each rotation matrix to an axis–angle pair.

        Parameters
        ----------
        tol : float, default 1e-6
            Numerical tolerance to treat very small angles as zero.
        degrees : bool, default False
            If True, the returned angles are in degrees instead of radians.

        Returns
        -------
        axis : np.ndarray (..., 3)
            Unit vectors representing the rotation axes.
        angle : np.ndarray (...,)
            Rotation angles (same batch shape as `axis` without the last dimension).

        Notes
        -----
        • Uses the `xyzw` quaternion sign convention internally (but does *not* return quaternions).  
        • When the angle is below `tol`, the axis is arbitrarily set to `[1, 0, 0]`.
        • The function is batch‑compatible and produces a *separate* array for the angles
        so that both outputs can be used independently:

            axis, theta = R.as_axis_angle()
        """
        R = self._mat
        dtype = self.FLOAT

        # Angle from trace: trace = 1 + 2*cos(theta)
        trace = np.clip(np.trace(R, axis1=-2, axis2=-1), -1.0, 3.0)
        angle = np.arccos((trace - 1.0) / 2.0).astype(dtype)

        # Axis extraction
        # For angles near zero we can pick any axis; choose +X.
        near_zero = angle < tol
        axis = np.empty(R.shape[:-2] + (3,), dtype=dtype)

        # Normal case (sin(theta) != 0)
        sin_theta = np.sin(angle)
        mask = ~near_zero
        denom = 2.0 * sin_theta[mask][..., None]  # broadcast for 3 components

        axis[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / denom[..., 0]
        axis[mask, 1] = (R[mask, 0, 2] - R[mask, 2, 0]) / denom[..., 0]
        axis[mask, 2] = (R[mask, 1, 0] - R[mask, 0, 1]) / denom[..., 0]

        # Near‑zero angle: pick X‑axis by convention
        if np.any(near_zero):
            axis[near_zero] = np.array([1.0, 0.0, 0.0], dtype=dtype)

        # Normalise axis (safe even for near‑zero case)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)

        if degrees:
            angle = np.rad2deg(angle)

        return axis, angle
    
    # ====================================================================================================
    # Operations
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Apply to an array of vectors
    # ----------------------------------------------------------------------------------------------------

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply the rotation to a set of 3D vectors.

        Parameters
        ----------
        vectors : array_like (..., 3)
            Vectors to rotate. Must be broadcastable with the rotation batch shape.

        Returns
        -------
        np.ndarray
            Rotated vectors, same shape as input.
        """
        vectors = np.asarray(vectors, dtype=self._mat.dtype)
        if vectors.shape[-1] != 3:
            raise ValueError("Input must have shape (..., 3)")
        return np.einsum('...ij,...j->...i', self._mat, vectors)
    
    # ----------------------------------------------------------------------------------------------------
    # Compose with another Rotation
    # ----------------------------------------------------------------------------------------------------

    def compose(self, other: "Rotation") -> "Rotation":
        """
        Compose this rotation with another Rotation instance.

        The resulting rotation applies `self` first, then `other`, i.e.:

            composed = self ∘ other  ⇔  composed @ v == other @ (self @ v)

        Parameters
        ----------
        other : Rotation
            The second rotation to compose. Must be a Rotation instance.

        Returns
        -------
        Rotation
            The composed rotation. A new object is always returned.

        Raises
        ------
        TypeError
            If `other` is not a Rotation instance.

        Notes
        -----
        • Fully supports broadcasting over batch dimensions.
        • This method is functionally equivalent to: `self @ other`.
        """
        if not isinstance(other, Rotation):
            raise TypeError(f"Expected a Rotation, got {type(other).__name__}")
        
        composed = self._mat @ (other.as_matrix()._mat)
        return type(self)(composed, copy=False)
    
    # ----------------------------------------------------------------------------------------------------
    # Inverse
    # ----------------------------------------------------------------------------------------------------

    def inverse(self) -> "Rotation":
        """
        Return the inverse rotation (i.e., transpose of the matrix).

        Returns
        -------
        Rotation
            Inverse of each rotation in the batch.
        """
        return type(self)(np.swapaxes(self._mat, -1, -2), copy=False)
    
    # ----------------------------------------------------------------------------------------------------
    # Identity check
    # ----------------------------------------------------------------------------------------------------

    def is_identity(self, tol: float = 1e-6):
        """
        Test whether each rotation in the batch is (numerically) the identity.

        Parameters
        ----------
        tol : float, default 1e-6
            Maximum absolute deviation allowed from the exact identity matrix.

        Returns
        -------
        bool or np.ndarray
            • If the rotation is scalar (batch shape == ()), returns a single bool.  
            • Otherwise, returns a boolean array with the same batch shape,
            where each element indicates whether the corresponding matrix
            is close to identity within `tol`.
        """
        eye = np.eye(3, dtype=self.FLOAT)
        # ‖R - I‖_∞  < tol   (max abs error per matrix)
        err = np.abs(self._mat - eye).max(axis=(-2, -1))
        result = err < tol
        # Return a Python bool when there is no batch dimension
        return bool(result) if result.shape == () else result
    
    # ----------------------------------------------------------------------------------------------------
    # Angle to
    # ----------------------------------------------------------------------------------------------------

    def angle_to(self, other: "Rotation", degrees: bool = False) -> np.ndarray:
        """
        Compute the angular distance between two rotations.

        Parameters
        ----------
        other : Rotation
            The other rotation(s) to compare with.
        degrees : bool, default False
            If True, return angle in degrees instead of radians.

        Returns
        -------
        np.ndarray
            Angle(s) of shape `(...)`, same batch shape as self and other (broadcasted),
            representing the rotation needed to go from `self` to `other`.

        Raises
        ------
        TypeError
            If `other` is not a Rotation.
        """
        if not isinstance(other, Rotation):
            raise TypeError(f"Expected Rotation, got {type(other).__name__}")

        delta = other @ ~self                      # difference: what rotates self into other
        trace = np.clip(np.trace(delta._mat, axis1=-2, axis2=-1), -1.0, 3.0)
        theta = np.arccos((trace - 1.0) / 2.0)     # from trace = 1 + 2 cos(θ)

        if degrees:
            theta = np.rad2deg(theta)
        return theta


    # ----------------------------------------------------------------------------------------------------
    # Operators
    # ----------------------------------------------------------------------------------------------------

    def __matmul__(self, other):
        """
        Overload ``@`` for rotations.

        * ``R @ S`` where both are ``Rotation`` → composition (apply *R*, puis *S*).
        * ``R @ v`` where *v* has shape ``(..., 3)`` → applique la rotation.

        Parameters
        ----------
        other : Rotation or array_like (..., 3)

        Returns
        -------
        Rotation or np.ndarray
        """
        if isinstance(other, Rotation):
            return self.compose(other)
        elif isinstance(other, np.ndarray):
            return self.apply(other)
        else:
            raise TypeError(
                f"{type(self).__name__}: unsupported operand type for @: "
                f"{type(other).__name__}"
            )
        
    def __invert__(self) -> "Rotation":
        """
        Operator overload: return the inverse of the rotation (~R).

        Equivalent to `R.inverse()`. Subclasses can override `inverse()` to
        customize this behavior.
        """
        return self.inverse()
    
    # ====================================================================================================
    # Utilities
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Interpolation with another rotation
    # ----------------------------------------------------------------------------------------------------


    def interpolate(self, other, t):
        """
        Performs spherical linear interpolation (SLERP) between two rotations.

        Parameters
        ----------
        other : Rotation
            The target rotation to interpolate to.
        t : float or ndarray
            Interpolation parameter(s), where 0 gives self and 1 gives other.
            Supports broadcasting over batch dimensions.

        Returns
        -------
        Quaternion
            Interpolated rotations as quaternions.
        """

        from . quaternion import Quaternion

        q1 = self.as_quaternion().as_array()
        q2 = other.as_quaternion().as_array()

        # Ensure shortest path by flipping q2 if necessary
        dot = np.sum(q1 * q2, axis=-1, keepdims=True)
        q2 = np.where(dot < 0, -q2, q2)
        dot = np.abs(dot)
        dot = np.clip(dot, -1.0, 1.0)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        # Broadcast t to match q shape
        t = np.asarray(t)[..., None]

        # Handle small angles with linear interpolation
        small = sin_theta < 1e-6
        w1 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / sin_theta)
        w2 = np.where(small, t, np.sin(t * theta) / sin_theta)

        q_interp = w1 * q1 + w2 * q2
        return Quaternion(q_interp, copy=False)
    
    # ----------------------------------------------------------------------------------------------------
    # Rotation from vectors to others
    # ----------------------------------------------------------------------------------------------------

    @staticmethod
    def from_vectors(v_src, v_dst):
        """
        Constructs a rotation (as a quaternion) that rotates v_src onto v_dst.
        
        Parameters
        ----------
        v_src : array_like, shape (..., 3)
            Source vectors.
        v_dst : array_like, shape (..., 3)
            Target vectors.
        
        Returns
        -------
        Rotation
            Rotation that aligns v_src with v_dst.
        """
        from .quaternion import Quaternion
        from .rotation import Rotation

        v_src = np.asarray(v_src, dtype=np.float32)
        v_dst = np.asarray(v_dst, dtype=np.float32)

        v_src, v_dst = np.broadcast_arrays(v_src, v_dst)

        # Normalize vectors
        v_src = v_src / np.linalg.norm(v_src, axis=-1, keepdims=True)
        v_dst = v_dst / np.linalg.norm(v_dst, axis=-1, keepdims=True)

        dot = np.einsum('...i,...i->...', v_src, v_dst)  # shape (...)

        # Case: vectors are (almost) identical → identity quaternion
        mask_same = dot > 0.999999

        if np.all(mask_same):
            q = np.zeros(v_src.shape[:-1] + (4,), dtype=v_src.dtype)
            q[..., 3] = 1.0
            return Quaternion(q, copy=False)

        # Case: vectors are opposite → 180° rotation
        mask_opp = dot < -0.999999
        if np.all(mask_opp):
            # Find a vector orthogonal to v_src
            ortho = np.zeros_like(v_src)
            ortho[..., 0] = 1.0
            axis = np.cross(v_src, ortho)
            axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
            # If colinear, use Y axis instead
            fallback = np.zeros_like(v_src)
            fallback[..., 1] = 1.0
            axis = np.where(axis_norm < 1e-6, np.cross(v_src, fallback), axis)
            axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
            q = np.zeros(v_src.shape[:-1] + (4,), dtype=v_src.dtype)
            q[..., :3] = axis  # 180° rotation → w = 0
            return Quaternion(q, copy=False)

        # General case
        axis = np.cross(v_src, v_dst)
        w = 1.0 + dot
        #q = np.concatenate([w[..., None], axis], axis=-1)
        #q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        q = np.concatenate([axis, w[..., None]], axis=-1)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)

        return Quaternion(q, copy=False)

    # ----------------------------------------------------------------------------------------------------
    # Look at
    # ----------------------------------------------------------------------------------------------------

    @staticmethod
    def look_at_NEW_BOF(v_src, v_dst, up=None, upward=(0, 0, 1)):
        """
        Build a Rotation that sends (v_src, up) → (v_dst, upward)
        while preserving a right-handed orthonormal frame.

        Parameters
        ----------
        v_src : array_like, shape (..., 3)
            Source direction vector(s)
        v_dst : array_like, shape (..., 3)
            Target direction vector(s)
        up : array_like or None, shape (..., 3), optional
            Source up vector(s). If None, roll is ignored.
        upward : array_like, shape (..., 3), optional
            Target up vector(s). Default is (0, 0, 1).

        Returns
        -------
        Rotation
            A rotation that maps the (v_src, up) frame onto the (v_dst, upward) frame.
        """
        # If roll doesn't matter: use direct rotation
        if up is None:
            return Rotation.from_vectors(v_src, v_dst)

        v_src   = np.asarray(v_src, dtype=np.float32)
        v_dst   = np.asarray(v_dst, dtype=np.float32)
        up      = np.asarray(up, dtype=np.float32)
        upward  = np.asarray(upward, dtype=np.float32)

        # Broadcast to common shape
        v_src, v_dst, up, upward = np.broadcast_arrays(v_src, v_dst, up, upward)

        # Detect scalar input
        is_scalar = v_src.ndim == 1
        if is_scalar:
            v_src   = v_src[None, :]
            v_dst   = v_dst[None, :]
            up      = up[None, :]
            upward  = upward[None, :]

        def make_basis(forward, up_hint):
            forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
            right = np.cross(forward, up_hint)
            norm = np.linalg.norm(right, axis=-1, keepdims=True)

            # Detect degenerate up vectors
            bad = (norm < 1e-6)[..., 0]
            if np.any(bad):
                alt_up = np.broadcast_to([1.0, 0.0, 0.0], up_hint.shape)
                up_hint = np.where(bad[:, None], alt_up, up_hint)
                right = np.cross(forward, up_hint)
                norm = np.linalg.norm(right, axis=-1, keepdims=True)

            right = right / norm
            up = np.cross(right, forward)
            return np.stack((right, up, forward), axis=-1)  # shape (..., 3, 3)

        B_src = make_basis(v_src, up)
        B_dst = make_basis(v_dst, upward)

        R = B_dst @ np.swapaxes(B_src, -1, -2)

        if is_scalar:
            R = R[0]

        return Rotation(R, copy=False)




    @staticmethod
    def look_at(v_src, v_dst, up=None, upward=(0, 0, 1)):
        """
        Build a Rotation that sends (v_src, up) → (v_dst, upward)
        while keeping a right‑handed frame.
        Returns a Rotation (matrix 3×3).
        """

        # If roll doesn't matter: use direct rotation
        if up is None:
            return Rotation.from_vectors(v_src, v_dst)

        v_src   = np.asarray(v_src,   dtype=np.float32)
        v_dst   = np.asarray(v_dst,   dtype=np.float32)
        up      = np.asarray(up,      dtype=np.flooat32)
        upward  = np.asarray(upward,  dtype=np.flooat32)

        # Same shape for everybody
        v_src, v_dst, up, upward = np.broadcast_arrays(v_src, v_dst, up, upward)

        is_scalar = v_src.shape == (3,)
        if is_scalar:
            v_src = v_src[None, :]
            v_dst = v_dst[None, :]
            up    = up[None, :]
            upward= upward[None, :]

        def make_basis(forward, up_hint):
            forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
            right   = np.cross(forward, up_hint)
            right   = right / np.linalg.norm(right, axis=-1, keepdims=True)
            up      = np.cross(right, forward)
            return np.stack([right, up, forward], axis=-1)       # (..., 3, 3)
        
        def make_basis_NEW_BOF(forward, up_hint):
            forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)

            # Si forward ⊥ up_hint → produit vectoriel nul ⇒ choisir un autre up_hint
            right = np.cross(forward, up_hint)
            norm_right = np.linalg.norm(right, axis=-1, keepdims=True)

            # Cas dégénéré : norm_right ≈ 0 → choisir un up_hint alternatif
            mask = norm_right.squeeze(-1) < 1e-6
            print("ERREUR", norm_right.squeeze(-1), mask)
            if np.any(mask):
                # Un vecteur arbitrairement non-colinéaire à forward
                alt_up = np.array([1.0, 0.0, 0.0], dtype=forward.dtype)
                alt_up = np.broadcast_to(alt_up, up_hint.shape)
                up_hint = np.where(mask[:, None], alt_up, up_hint)
                right = np.cross(forward, up_hint)
                norm_right = np.linalg.norm(right, axis=-1, keepdims=True)

            right = right / norm_right
            up = np.cross(right, forward)

            return np.stack([right, up, forward], axis=-1)

        B_src = make_basis(v_src, up)
        B_dst = make_basis(v_dst, upward)

        # R = B_dst · B_srcᵀ   (car B_src est orthonormale ⇒ inverse = transpose)
        R = B_dst @ np.swapaxes(B_src, -2, -1)

        if is_scalar:
            R = R[0]

        return Rotation(R, copy=False)


    @staticmethod
    def look_at_OLD(v_src, v_dst, up=None, upward=(0, 0, 1)):
        """
        Returns quaternions aligning v_src to v_dst while rotating up towards upward.

        Parameters
        ----------
        v_src : array_like, shape (..., 3)
            Input direction(s) to be aligned.
        v_dst : array_like, shape (..., 3)
            Target direction(s).
        up : array_like, shape (..., 3), optional
            Up vector(s) defining orientation. Defaults to (0, 1, 0).
        upward : array_like, shape (..., 3), optional
            Target upward vector(s). Defaults to (0, 0, 1).

        Returns
        -------
        Quaternion
            Quaternion(s) rotating each (v_src, up) to align with (v_dst, upward).
        """
        from . quaternion import Quaternion

        v_src = np.asarray(v_src)
        v_dst = np.asarray(v_dst)

        if up is None:
            up = np.array([0, 1, 0], dtype=v_src.dtype)
        if upward is None:
            upward = np.array([0, 0, 1], dtype=v_src.dtype)

        up = np.asarray(up)
        upward = np.asarray(upward)

        # Broadcast all inputs to common shape
        v_src, v_dst, up, upward = np.broadcast_arrays(v_src, v_dst, up, upward)

        def make_basis(fwd, up_hint):
            fwd = fwd / np.linalg.norm(fwd, axis=-1, keepdims=True)
            right = np.cross(up_hint, fwd)
            right = right / np.linalg.norm(right, axis=-1, keepdims=True)
            up = np.cross(fwd, right)
            return np.stack([right, up, fwd], axis=-2)  # (..., 3, 3)

        basis_src = make_basis(v_src, up)
        basis_dst = make_basis(v_dst, upward)

        # Rotation matrix R that maps basis_src → basis_dst
        R = basis_dst @ np.swapaxes(basis_src, -2, -1)  # (..., 3, 3)

        return Rotation(R)


