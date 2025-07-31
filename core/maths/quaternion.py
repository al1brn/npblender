# npblender/maths/quaternion.py
# MIT License
# Created on 2022-11-11
# Last update: 2025-07-21
# Author: Alain Bernard

"""
Quaternion
==========

Subclass of Rotation storing unit quaternions (xyzw convention).
Backed by an ItemsArray with shape (..., 4), allowing full interoperability
with the Rotation interface (composition, inversion, conversion, etc.).

Features:
- Efficient storage of batched unit quaternions
- Composition via Hamilton product
- Direct application to 3D vectors (vectorised)
- Interconversion with matrices, Euler angles, and axis–angle
- Full compatibility with the Rotation protocol and NumPy API

All quaternion operations follow the right-hand rule and assume normalized inputs.
"""

import numpy as np

from . rotation import Rotation

class Quaternion(Rotation):
    """
    Quaternion representation of rotations (unit, xyzw convention).

    This subclass of Rotation stores each rotation as a 4D unit quaternion,
    and implements all core operations (compose, inverse, apply) via
    matrix conversion when needed.

    Internal shape: (..., 4)
    """

    _item_shape = (4,)

    # ----------------------------------------------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def identity(cls, shape=()) -> "Quaternion":
        """Return the identity quaternion (no rotation)."""
        dtype = cls.FLOAT
        q = np.zeros(shape + (4,), dtype=dtype)
        q[..., 3] = 1.0
        return cls(q, copy=False)

    @classmethod
    def from_matrix(cls, mat, *, normalize: bool = True) -> "Quaternion":
        """
        Construct a Quaternion from rotation matrices.

        Parameters
        ----------
        mat : array_like (..., 3, 3)
            Input rotation matrices.
        normalize : bool, default True
            If True, output quaternions are normalized.

        Returns
        -------
        Quaternion
        """
        rot = Rotation.from_matrix(mat)
        q = rot.as_quaternion()
        if normalize:
            q /= np.linalg.norm(q, axis=-1, keepdims=True)
        return cls(q, copy=False)
    
    # ----------------------------------------------------------------------------------------------------
    # From quaternion
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_quaternion(
        cls,
        quat,
        *,
        normalize: bool = True,
        tol: float = 1e-5,
    ) -> "Quaternion":
        """
        Construct a Quaternion from raw (x y z w) values.

        Parameters
        ----------
        quat : array_like (..., 4)
            Input quaternions in (x, y, z, w) order. Must be broadcastable.
        normalize : bool, default True
            If True, the input is renormalised to unit length.
        tol : float, default 1e-5
            Tolerance for the norm check when ``normalize=False``.

        Returns
        -------
        Quaternion
            Instance storing the (possibly normalised) quaternions.

        Raises
        ------
        ValueError
            If shape is wrong or quaternions are not unit (when normalize=False).
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

        return cls(quat, copy=False)
    
    @classmethod
    def from_euler(
        cls,
        euler,
        *,
        order: str = "XYZ",
        degrees: bool = False,
    ) -> "Quaternion":
        """
        Construct a Quaternion from Euler angles.

        Parameters
        ----------
        euler : array_like (..., 3)
            Angles for the three axes, in the order specified by `order`.
        order : {'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'}, default 'XYZ'
            Axis order (uppercase = intrinsic / body‑fixed convention).
        degrees : bool, default False
            If True, `euler` is interpreted in degrees instead of radians.

        Returns
        -------
        Quaternion
            Quaternion representing the composed rotation.
        """
        euler = np.asarray(euler, dtype=cls.FLOAT)
        if euler.shape[-1] != 3:
            raise ValueError("Euler angles must have shape (..., 3)")

        if degrees:
            euler = np.deg2rad(euler)

        # Axes convention
        order = order.upper()
        if order not in {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"}:
            raise ValueError(f"Unsupported Euler order '{order}'")

        def axis_quat(axis, angle):
            """Quaternion for a rotation around a single axis."""
            q = np.zeros(angle.shape + (4,), dtype=cls.FLOAT)
            q[..., axis] = np.sin(angle / 2)
            q[..., 3] = np.cos(angle / 2)
            return q

        # Compose quaternions in the specified order
        i0, i1, i2 = map("XYZ".index, order)
        q0 = axis_quat(i0, euler[..., 0])
        q1 = axis_quat(i1, euler[..., 1])
        q2 = axis_quat(i2, euler[..., 2])

        # Quaternion multiplication: q2 * q1 * q0 (right to left, intrinsic)
        def quat_mul(q1, q2):
            """
            Hamilton product  q = q1 ⊗ q2
            """
            x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
            x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

            return np.stack([
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ], axis=-1)

        q = quat_mul(q2, quat_mul(q1, q0))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        return cls(q, copy=False)
    
    
    # ----------------------------------------------------------------------------------------------------
    # From axis–angle
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_axis_angle(
        cls,
        axis,
        angle,
        *,
        normalize_axis: bool = True,
        degrees: bool = False,
    ) -> "Quaternion":
        """
        Construct a Quaternion from an axis–angle pair.

        Parameters
        ----------
        axis : array_like (..., 3)
            Rotation axis (x, y, z). Does **not** need to be unit length if
            ``normalize_axis=True``.
        angle : array_like (...,)
            Rotation angle (radians by default), broadcastable with ``axis``.
        normalize_axis : bool, default True
            Normalise the axis to unit length before conversion.
        degrees : bool, default False
            Interpret ``angle`` in degrees instead of radians.

        Returns
        -------
        Quaternion
            Quaternion representing the given axis–angle rotation.
        """
        axis = np.asarray(axis, dtype=cls.FLOAT)
        angle = np.asarray(angle, dtype=cls.FLOAT)

        if axis.shape[-1] != 3:
            raise ValueError("Axis must have shape (..., 3)")

        if degrees:
            angle = np.deg2rad(angle)

        if normalize_axis:
            axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)

        half = 0.5 * angle
        s = np.sin(half)

        q = np.empty(axis.shape[:-1] + (4,), dtype=cls.FLOAT)
        q[..., :3] = s[..., None] * axis          # (x, y, z)
        q[..., 3]  = np.cos(half)                # w

        return cls(q, copy=False)    

    # ----------------------------------------------------------------------------------------------------
    # Conversion
    # ----------------------------------------------------------------------------------------------------

    def as_matrix(self) -> np.ndarray:
        """Convert each quaternion to a 3×3 rotation matrix."""
        return Rotation.from_quaternion(self.as_array())

    def as_quaternion(self) -> np.ndarray:
        """Return the raw quaternion array (xyzw, shape (..., 4))."""
        return self
    
    def as_euler(self, order: str = "XYZ", degrees: bool = False) -> np.ndarray:
        """
        Convert each quaternion to Euler/Tait–Bryan angles.

        Parameters
        ----------
        order : {'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'}, default 'XYZ'
            Axis order used for the decomposition (uppercase = intrinsic/body-fixed).
        degrees : bool, default False
            If True, angles are returned in degrees instead of radians.

        Returns
        -------
        np.ndarray
            Euler angles of shape (..., 3), in the specified order.
        """
        # Convert to matrix, then delegate to Rotation
        R = self.as_matrix()             # -> Rotation
        return R.as_euler(order=order, degrees=degrees)
    
    def as_axis_angle(self, tol: float = 1e-6, degrees: bool = False):
        """
        Convert each quaternion to an axis–angle pair.

        Parameters
        ----------
        tol : float, default 1e-6
            Numerical tolerance: angles below this are treated as zero.
        degrees : bool, default False
            If True, angles are returned in degrees instead of radians.

        Returns
        -------
        axis : np.ndarray (..., 3)
            Unit rotation axes.
        angle : np.ndarray (...,)
            Rotation angles.

        Notes
        -----
        - Quaternions must be normalized.
        - For angles close to zero, axis is set arbitrarily to [1, 0, 0].
        """
        q = self._mat  # (..., 4) → (x, y, z, w)
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        norm_xyz = np.linalg.norm(q[..., :3], axis=-1)
        angle = 2.0 * np.arctan2(norm_xyz, w)

        axis = np.zeros_like(q[..., :3])
        small = norm_xyz < tol
        large = ~small

        if np.any(large):
            axis[large] = q[large, :3] / norm_xyz[large, None]

        if np.any(small):
            axis[small] = np.array([1.0, 0.0, 0.0], dtype=self.FLOAT)

        if degrees:
            angle = np.rad2deg(angle)

        return axis, angle

    # ----------------------------------------------------------------------------------------------------
    # Overloads
    # ----------------------------------------------------------------------------------------------------

    def inverse(self) -> "Quaternion":
        """Return the inverse (conjugate) of the unit quaternion."""
        q = self._mat.copy()
        q[..., :3] *= -1
        return type(self)(q, copy=False)

    def compose(self, other: "Rotation") -> "Quaternion":
        """
        Compose this quaternion with another rotation (self ∘ other).

        Internally performs a Hamilton product: self @ other.as_quaternion()

        Parameters
        ----------
        other : Rotation
            The second rotation to apply (any type: matrix, quaternion, ...).

        Returns
        -------
        Quaternion
            Composed rotation, of the same type as `self`.
        """
        if not isinstance(other, Rotation):
            raise TypeError(f"{type(self).__name__}: expected Rotation, got {type(other).__name__}")

        q1 = self._mat
        q2 = other.as_quaternion()._mat

        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        q = np.stack([x, y, z, w], axis=-1)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)

        return type(self)(q, copy=False)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply the quaternion rotation to 3D vectors.

        Parameters
        ----------
        vectors : array_like (..., 3)
            Vectors to rotate. Must be broadcastable with the rotation batch shape.

        Returns
        -------
        np.ndarray
            Rotated vectors, same shape as input.
        
        Notes
        -----
        Uses the formula:  
            v_rot = q ⋅ v ⋅ q⁻¹  
        where v is a pure quaternion with w=0.

        Broadcasting is supported across batch dimensions.
        """
        vectors = np.asarray(vectors, dtype=self._mat.dtype)
        if vectors.shape[-1] != 3:
            raise ValueError("Vectors must have shape (..., 3)")

        # Extract quaternion components
        q = self._mat
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        # Compute cross products (vectorised)
        uv = np.cross(q[..., :3], vectors)                          # u = q_vec × v
        uuv = np.cross(q[..., :3], uv)                              # u = q_vec × (q_vec × v)

        # Compute rotated vector: v + 2 * (w * u + uu)
        rotated = vectors + 2 * (w[..., None] * uv + uuv)
        return rotated
    
    def is_identity(self, tol: float = 1e-6):
        """
        Test whether each quaternion is (numerically) the identity rotation.

        Parameters
        ----------
        tol : float, default 1e-6
            Tolerance on the angular deviation from identity.

        Returns
        -------
        bool or np.ndarray
            • If the quaternion is scalar (no batch), returns a single bool.  
            • Otherwise, returns a boolean array with the same batch shape.
        """
        q = self._mat
        # Identity rotation ⇔ angle ≈ 0 ⇔ w ≈ ±1
        # Use the quaternion angle = 2 * arccos(|w|) — robust even if flipped
        angle = 2 * np.arccos(np.clip(np.abs(q[..., 3]), 0.0, 1.0))
        result = angle < tol
        return bool(result) if result.shape == () else result
    
    def angle_to(self, other: "Rotation", degrees: bool = False) -> np.ndarray:
        """
        Compute the angular distance between this quaternion and another rotation.

        Parameters
        ----------
        other : Rotation
            Another rotation (can be any subclass of Rotation).
        degrees : bool, default False
            If True, returns the angle in degrees.

        Returns
        -------
        np.ndarray
            Rotation angle(s) needed to go from self to other, shape (...,).
        """
        if not isinstance(other, Rotation):
            raise TypeError(f"{type(self).__name__}> expected Rotation, got {type(other).__name__}")

        q1 = self._mat
        q2 = other.as_quaternion()._mat

        # Angle from quaternion dot product
        dot = np.sum(q1 * q2, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)  # numerical safety
        angle = 2 * np.arccos(np.abs(dot))  # abs ensures shortest arc

        if degrees:
            angle = np.rad2deg(angle)
        return angle
    

if __name__ == '__main__':

    import itertools
    import time

    def generate_base_eulers():
        """
        Generate a test set of Euler angles formed by all combinations of
        0, 90, 180, and -90 degrees on each axis (X, Y, Z).

        Returns
        -------
        np.ndarray of shape (64, 3)
            Euler angles in radians (XYZ convention).
        """
        degrees = [0, 90, 180, -90]
        combos = list(itertools.product(degrees, repeat=3))  # 4 × 4 × 4 = 64
        eulers_deg = np.array(combos, dtype=np.float32)      # shape (64, 3)
        return np.deg2rad(eulers_deg)                        # convert to radians

    def test_rotations_consistency(euler_angles: np.ndarray,
                                vectors:      np.ndarray,
                                *,
                                atol: float = 1e-6):
        """
        Consistency (and micro‑benchmark) for Rotation & Quaternion.

        Parameters
        ----------
        euler_angles : (n, 3) radians
        vectors      : (m, 3)
        atol         : tolerance for np.allclose
        """
        print(f"\nTests with {len(euler_angles)} Euler angles × {len(vectors)} vectors")

        n = euler_angles.shape[0]

        # ------------------------------------------------------------------
        # Référence : matrice simple
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        ref_rot     = Rotation.from_euler(euler_angles).reshape(n, 1)
        rotated_ref = ref_rot @ vectors
        t1 = time.perf_counter()
        print(f"Reference (Rotation.from_euler  +  @vectors) : {(t1-t0)*1e3:.2f} ms")

        # ------------------------------------------------------------------
        # Boucle sur les deux classes
        # ------------------------------------------------------------------
        for Rot in (Rotation, Quaternion):
            print(f"\n┌── {Rot.__name__} ─────────────────────────────────────")

            rot = Rot.from_euler(euler_angles)            # shape (n, …)

            # Helper: mesure et assertion
            def _check(name, func):
                t0 = time.perf_counter()
                out = func()
                t1 = time.perf_counter()

                diff = np.abs(out - rotated_ref)
                err_mask = ~np.all(diff <= atol, axis=-1)  # shape (n, m)
                num_errors = np.count_nonzero(err_mask)

                if num_errors == 0:
                    print(f" ✅ {name:<34s}: {(t1 - t0) * 1e3:7.2f} ms")
                else:
                    print(f"❌ {Rot.__name__} {name} : {num_errors} erreurs sur {n * len(vectors)} combinaisons")

                    # Trouve les indices (i, j) des erreurs
                    i_err, j_err = np.where(err_mask)

                    for k in range(min(3, num_errors)):
                        i, j = i_err[k], j_err[k]
                        expected = rotated_ref[i, j]
                        actual = out[i, j]
                        delta = actual - expected
                        print(f"   Rotation #{i}, Vecteur #{j}")
                        print(f"     attendu : {expected}")
                        print(f"     obtenu  : {actual}")
                        print(f"     Δ       : {delta}")

                    print(f"   Max erreur absolue : {diff.max():.2e}")

                    # Tu peux lever une AssertionError ici si tu veux rendre le test bloquant :
                    raise AssertionError(f"{Rot.__name__} {name} failed")

            # --- Test 1 : via as_matrix ---
            _check("Test 1  as_matrix", lambda: rot.as_matrix().reshape(n, 1) @ vectors)

            # --- Test 2 : via as_quaternion ---
            _check("Test 2  as_quaternion", lambda: rot.as_quaternion().reshape(n, 1) @ vectors)

            # --- Test 3 : matrix → quaternion ---
            _check("Test 3  matrix→quat", lambda: rot.as_matrix()
                                                .as_quaternion()
                                                .reshape(n, 1) @ vectors)

            # --- Test 4 : quaternion → matrix ---
            _check("Test 4  quat→matrix", lambda: rot.as_quaternion()
                                                .as_matrix()
                                                .reshape(n, 1) @ vectors)

            # --- Test 6 : round‑trip via Euler ---
            t0 = time.perf_counter()
            rot6  = Rot.from_euler(rot.as_euler())
            delta = rot.inverse() @ rot6
            err_mask = ~delta.is_identity(tol=atol)
            n_fail = np.count_nonzero(err_mask)
            t1 = time.perf_counter()

            if n_fail == 0:
                print(f" Test 6  via Euler round‑trip       : {(t1 - t0)*1e3:7.2f} ms")
            else:
                # Pour caractériser l’erreur, on regarde l’angle résiduel (angle de rotation de delta)
                angles = delta.angle_to(Rotation.identity())
                err_angles = angles[err_mask]
                mean_err = np.mean(err_angles)
                max_err  = np.max(err_angles)
                print(f"❌ Test 6  via Euler round‑trip       : {(t1 - t0)*1e3:7.2f} ms"
                    f" — {n_fail} échecs, angle moyen = {mean_err:.2e}, max = {max_err:.2e}")

            # --- Test 7 : round‑trip via axis–angle ---
            t0 = time.perf_counter()
            axis, angle = rot.as_axis_angle()
            rot7 = Rot.from_axis_angle(axis, angle)
            out7 = rot7.reshape(n, 1) @ vectors
            t1 = time.perf_counter()

            err = np.abs(out7 - rotated_ref).max(axis=-1)  # erreur max sur chaque couple (n, m)
            err_mask = np.any(err > atol, axis=-1)
            n_fail = np.count_nonzero(err_mask)

            if n_fail == 0:
                print(f" Test 7  via axis‑angle round‑trip  : {(t1 - t0)*1e3:7.2f} ms")
            else:
                delta = rot.inverse() @ rot7
                angles = delta.angle_to(Rotation.identity())
                mean_err = np.mean(angles[err_mask])
                max_err = np.max(angles[err_mask])
                print(f"❌ Test 7  via axis‑angle round‑trip  : {(t1 - t0)*1e3:7.2f} ms"
                    f" — {n_fail} échecs, angle moyen = {mean_err:.2e}, max = {max_err:.2e}")

            # --- Test 8 : rot ∘ rot⁻¹ == I ---
            t0 = time.perf_counter()
            identity = rot @ rot.inverse()
            ok = identity.is_identity(tol=atol).all()
            t1 = time.perf_counter()
            assert ok, f"{Rot.__name__} Test 8 failed"
            print(f" Test 8  rot @ inv == I             : {(t1-t0)*1e3:7.2f} ms")

        print("\n✅  All rotation consistency tests passed.")

    def test_rotation_quaternion_composition(euler_angles: np.ndarray, atol: float = 1e-6):
        """
        Teste la composition de Rotations et de Quaternions (et leur inverse).

        Paramètres
        ----------
        euler_angles : (n, 3)
            Angles d'Euler (en radians)
        atol : float
            Tolérance pour l'identité
        """
        import time

        n = len(euler_angles)
        print(f"\n▶ Test de composition avec {n} rotations/quaternions")

        # Étape 1 : Génération des rotations
        t0 = time.perf_counter()
        R = Rotation.from_euler(euler_angles)      # (n,)
        t1 = time.perf_counter()
        print(f" Génération des rotations        : {(t1 - t0) * 1e3:.2f} ms")

        # Étape 2 : Génération des quaternions
        t0 = time.perf_counter()
        Q = Quaternion.from_euler(euler_angles)    # (n,)
        t1 = time.perf_counter()
        print(f" Génération des quaternions      : {(t1 - t0) * 1e3:.2f} ms")

        # Étape 3 : Combinaisons n × n
        t0 = time.perf_counter()
        R2 = R[:, None] @ R[None, :]               # (n, n)
        t1 = time.perf_counter()
        print(f" Combinaison des rotations       : {(t1 - t0) * 1e3:.2f} ms {R2.shape}")

        t0 = time.perf_counter()
        Q2 = Q[:, None] @ Q[None, :]               # (n, n)
        t1 = time.perf_counter()
        print(f" Combinaison des quaternions     : {(t1 - t0) * 1e3:.2f} ms {Q2.shape}")

        print(f"Shapes: {R2.shape=}, {R2.inverse().shape=}, {Q2.shape=}, {Q2.inverse().shape=}")

        # Test 1 : Q2⁻¹ @ R2 → I
        t0 = time.perf_counter()
        identity1 = Q2.inverse() @ R2
        angles1 = identity1.angle_to(Rotation.identity())
        fail1 = angles1 > atol
        n_fail1 = np.count_nonzero(fail1)
        t1 = time.perf_counter()
        if n_fail1 == 0:
            print(f" ✅ Test 1 (Q2⁻¹ @ R == I)         : {(t1 - t0) * 1e3:.2f} ms — OK")
        else:
            print(f" ❌ Test 1 (Q2⁻¹ @ R == I)         : {(t1 - t0) * 1e3:.2f} ms — {n_fail1} erreurs, angle ∈ [{angles1.min():.2e}, {angles1.max():.2e}]")

        # Test 2 : R2⁻¹ @ Q2 → I
        t0 = time.perf_counter()
        identity2 = R2.inverse() @ Q2
        angles2 = identity2.angle_to(Rotation.identity())
        fail2 = angles2 > atol
        n_fail2 = np.count_nonzero(fail2)
        t1 = time.perf_counter()
        if n_fail2 == 0:
            print(f" ✅ Test 2 (R2⁻¹ @ Q2 == I)         : {(t1 - t0) * 1e3:.2f} ms — OK")
        else:
            print(f" ❌ Test 2 (R2⁻¹ @ Q2 == I)         : {(t1 - t0) * 1e3:.2f} ms — {n_fail2} erreurs, angle ∈ [{angles2.min():.2e}, {angles2.max():.2e}]")


        
    # ----------------------------------------------------------------------------------------------------
    # rot @ vectors tests
    # ----------------------------------------------------------------------------------------------------

    rng = np.random.default_rng(0)

    def test_rotate():
        n, m = 100, 100
        for test_type in range(5):
            print("\n>>> Test type", test_type)
            if test_type == 0:
                eulers = generate_base_eulers()
                vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            elif test_type == 1:
                eulers = generate_base_eulers()
                vecs = rng.normal(size=(m, 3))
            elif test_type == 2:
                eulers = rng.uniform(-np.pi, np.pi, size=(n, 3))
                vecs = rng.normal(size=(m, 3))
            elif test_type == 3:
                eulers = rng.uniform(-np.pi, np.pi, size=(1_000_000, 3))
                vecs = rng.normal(size=(100, 3))
            elif test_type == 4:
                eulers = rng.uniform(-np.pi, np.pi, size=(100, 3))
                vecs = rng.normal(size=(1_000_000, 3))

            test_rotations_consistency(eulers, vecs, atol=1e-4)

    # ----------------------------------------------------------------------------------------------------
    # rot @ rot tests
    # ----------------------------------------------------------------------------------------------------

    def test_compose():
        n = 100
        eulers = generate_base_eulers()

        for test_type in range(4):
            print("\n>>> Test type", test_type)
            if test_type == 0:
                pass
                
            elif test_type == 1:
                eulers = np.append(eulers, rng.normal(size=(n, 3)), axis=0)

            elif test_type == 2:
                eulers = np.append(eulers, rng.normal(size=(3000, 3)), axis=0)

            elif test_type == 3:
                eulers = rng.normal(size=(10000, 3))

            test_rotation_quaternion_composition(eulers, atol= 1e-5)



    test_compose()


