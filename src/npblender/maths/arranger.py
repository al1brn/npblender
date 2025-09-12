__all__ = ["Transfo2d", "Zone"]

import numpy as np
import math

# ====================================================================================================
# Transfo2D
# ====================================================================================================

class Transfo2d:
    """2D affine transform in homogeneous coordinates (3x3).
    
    Column-vector convention: p' = M @ p_h, with p_h = [x, y, 1]^T.
    Linear part is M[:2,:2], translation is M[:2,2].
    """

    def __init__(self, matrix=None):
        if matrix is None:
            self._matrix = np.eye(3, dtype=float)
        else:
            M = np.array(matrix, dtype=float, copy=True)
            if M.shape != (3, 3):
                raise ValueError(f"Matrix must be shaped (3, 3), not {M.shape}")
            self._matrix = M
        self._touched = False

    def __str__(self):
        sx, sy = self.scale
        tx, ty = self.translation
        angle_deg = math.degrees(self.rotation)
        return (f"Transfo2d(\n"
                f"  scale=({sx:.3f}, {sy:.3f}),\n"
                f"  rotation={angle_deg:.2f}°,\n"
                f"  translation=({tx:.3f}, {ty:.3f})\n"
                f")")
    
    @property
    def touched(self):
        return self._touched
    
    def touch(self, touch=True):
        self._touched = touch

    # -------------------------------------
    # Basic accessors
    # -------------------------------------
    def __array__(self):
        # Allow np.asarray(tf) to view the matrix
        return self._matrix

    @property
    def matrix(self):
        # Expose the underlying matrix (view, not copy)
        return self._matrix

    @matrix.setter
    def matrix(self, M):
        M = np.asarray(M, dtype=float)
        if M.shape != (3, 3):
            raise ValueError(f"Matrix must be shaped (3, 3), not {M.shape}")
        self._matrix[...] = M
        self.touch()

    @property
    def translation(self):
        # Returns a view (2,)
        return self._matrix[:2, 2]

    @translation.setter
    def translation(self, value):
        v = np.asarray(value, dtype=float).reshape(2)
        self._matrix[:2, 2] = v
        self.touch()

    # -------------------------------------
    # Scale / Rotation decomposition (left polar: A = P @ R)
    # -------------------------------------
    @property
    def _scale_rot(self):
        """Return (scale, angle, R) from the 2x2 block using left polar decomposition.
        
        A = P @ R with:
          - P = U @ diag(S) @ U^T (symmetric positive-definite "scale/shear")
          - R = P^{-1} @ A (proper orthogonal if det(A) > 0)
        
        We return:
          - scale: np.array([sx, sy]) taken from diag(P) (ignores off-diagonal shear)
          - angle: rotation angle extracted from R
          - R: the 2x2 rotation matrix
        """
        A = self._matrix[:2, :2]
        # SVD-based polar decomposition (no SciPy required)
        U, S, Vt = np.linalg.svd(A)        # A = U @ diag(S) @ Vt
        P = U @ np.diag(S) @ U.T           # symmetric positive definite
        # Guard against numerical issues if P is near-singular
        # Use solve instead of explicit inverse where possible
        R = np.linalg.solve(P, A)          # R = P^{-1} @ A

        # Project R back to the closest rotation if slight numerical drift
        # via polar of R itself: R = R @ (R^T R)^{-1/2} ~ using SVD
        Ur, Sr, Vtr = np.linalg.svd(R)
        R = Ur @ Vtr
        if np.linalg.det(R) < 0:
            # Ensure a proper rotation (det=+1)
            Vtr[-1, :] *= -1
            R = Ur @ Vtr

        scale = np.diag(P).astype(float)
        angle = math.atan2(R[1, 0], R[0, 0])
        return scale, angle, R

    @_scale_rot.setter
    def _scale_rot(self, value):
        """Setter expects (scale, angle). Rebuild A = P @ R with P=diag(scale)."""
        scale, angle = value
        sx, sy = np.broadcast_to(np.asarray(scale, dtype=float).reshape(2), (2,))
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s],
                      [s,  c]], dtype=float)
        # Here we ignore shear and build P as a diagonal (pure scale)
        P = np.array([[sx, 0.0],
                      [0.0, sy]], dtype=float)
        self._matrix[:2, :2] = P @ R
        self.touch()

    @property
    def scale(self):
        scale, _, _ = self._scale_rot
        return scale

    @scale.setter
    def scale(self, value):
        _, angle, _ = self._scale_rot
        self._scale_rot = (value, angle)

    @property
    def rotation(self):
        _, angle, _ = self._scale_rot
        return angle

    @rotation.setter
    def rotation(self, angle):
        scale, _, _ = self._scale_rot
        # Normalize angle into (-pi, pi] for stability (optional)
        a = float(angle)
        a = (a + math.pi) % (2 * math.pi) - math.pi
        self._scale_rot = (scale, a)

    def __repr__(self):
        ang_deg = math.degrees(self.rotation)
        sx, sy = self.scale
        tx, ty = self.translation
        return (f"Transfo2d(scale=({sx:.6g}, {sy:.6g}), "
                f"rotation={ang_deg:.3f}°, translation=({tx:.6g}, {ty:.6g}))")

    # -------------------------------------
    # Constructors
    # -------------------------------------
    @classmethod
    def new(cls, translation=None, scale=None, rotation=None):
        """Create a transform from (translation, scale, rotation)."""
        tf = cls()
        # Set linear part
        if rotation is None and scale is not None:
            sx, sy = np.resize(scale, 2)
            tf._matrix[:2, :2] = np.array([[sx, 0.0], [0.0, sy]], dtype=float)
        else:
            if scale is None:
                scale = (1.0, 1.0)
            tf._scale_rot = (scale, 0.0 if rotation is None else float(rotation))

        # Set translation
        if translation is not None:
            tf.translation = translation
        return tf

    @classmethod
    def from_components(cls, tx=0.0, ty=0.0, sx=1.0, sy=1.0, angle=0.0):
        """Explicit alternative constructor."""
        tf = cls()
        tf._scale_rot = (np.array([sx, sy], dtype=float), float(angle))
        tf.translation = (tx, ty)
        return tf

    # -------------------------------------
    # Composition and application
    # -------------------------------------
    def matmul(self, other):
        """Compose with another transform or apply to a matrix.
        
        - If `other` is Transfo2d or a (3,3) array: returns a new Transfo2d with self.matrix @ other.
          (Apply `other` first, then `self`, under column-vector convention.)
        - If `other` looks like points (...,2) or (...,3): applies transform to the points.
        """
        if isinstance(other, Transfo2d):
            other_matrix = other.matrix
        else:
            other_matrix = np.asarray(other)
            if other_matrix.shape == (3, 3):
                pass
            else:
                # Not a 3x3: treat as points
                if other_matrix.shape[-1] in (2, 3):
                    return self.transform(other_matrix)
                raise ValueError(
                    f"Shape {other_matrix.shape} is not valid. "
                    "Argument must be either a (3,3) matrix or an array of 2D/3D vectors."
                )
        return Transfo2d(matrix=self.matrix @ other_matrix)

    def __matmul__(self, other):
        return self.matmul(other)

    # -------------------------------------
    # Points transform
    # -------------------------------------
    def transform(self, vectors):
        """Transform an array of 2D or 3D vectors, shape (...,2) or (...,3)."""
        vectors = np.asarray(vectors, dtype=float)
        last_dim = vectors.shape[-1]

        if last_dim == 2:
            ones = np.ones(vectors.shape[:-1] + (1,), dtype=float)
            vh = np.concatenate([vectors, ones], axis=-1)     # (..., 3)
            # (...,3) <- (3,3) @ (...,3)
            out = np.einsum('ij,...j->...i', self._matrix, vh)
            return out[..., :2]

        if last_dim == 3:
            # Extend to 4x4 so that z passes through unchanged and translation only affects x,y
            M4 = np.eye(4, dtype=float)
            M4[:2, :2] = self._matrix[:2, :2]
            M4[:2, 3]  = self._matrix[:2, 2]
            ones = np.ones(vectors.shape[:-1] + (1,), dtype=float)
            vh = np.concatenate([vectors, ones], axis=-1)     # (..., 4)
            out = np.einsum('ij,...j->...i', M4, vh)
            return out[..., :3]

        raise ValueError(f"Vectors must be 2D or 3D; got last_dim={last_dim} with shape {vectors.shape}")

    # -------------------------------------
    # Utility
    # -------------------------------------
    def inverse(self):
        """Return the inverse transform."""
        A = self._matrix[:2, :2]
        t = self._matrix[:2, 2]
        Ai = np.linalg.inv(A)
        Mi = np.eye(3, dtype=float)
        Mi[:2, :2] = Ai
        Mi[:2, 2]  = -Ai @ t
        return Transfo2d(matrix=Mi)
    
    # -------------------------------------
    # Operations
    # -------------------------------------

    def apply_transfo(self, other):
        self.matrix[...] = other.matrix @ self.matrix
        return self

    def translate(self, *t, forward=True):
        """Apply a translation AFTER the current transform (column-vector convention)."""
        T = type(self).new(translation=t).matrix
        if not forward:
            T[:2, 2] *= -1  # invert the just-built translation
        self.matrix[...] = T @ self.matrix
        self.touch()
        return self


    def rotate(self, angle, pivot=None):
        """Apply a rotation AFTER the current transform (column-vector convention)."""
        if pivot is not None:
            self.translate(pivot, False)

        R = type(self).new(rotation=angle).matrix
        self.matrix[...] = R @ self.matrix

        if pivot is not None:
            self.translate(pivot, True)

        self.touch()
        return self

    def apply_scale(self, *scale, pivot=None):
        """Apply a scaling AFTER the current transform (column-vector convention)."""

        if pivot is not None:
            self.translate(pivot, False)

        S = type(self).new(scale=scale).matrix
        self.matrix[...] = S @ self.matrix

        if pivot is not None:
            self.translate(pivot, True)

        self.touch()
        return self
    
# ====================================================================================================
# Item : a rectangular zone
# ====================================================================================================

class Zone:
    def __init__(self, x0, y0, x1, y1):
        self._x0, self._y0, self._x1, self._y1 = float(x0), float(y0), float(x1), float(y1)
        self._bbox = (self._x0, self._y0, self._x1, self._y1)

        self._transfo = Transfo2d()

    def __str__(self):
        st = str(self.transfo).replace("\n", " ")
        return f"<Zone [{self.x0:.2f}, {self.y0:.2f}, {self.x1:.2f}, {self.y1:.2f}] {st}>"
    
    @classmethod
    def from_points(cls, points):
        vmin, vmax = np.min(points, axis=0), np.max(points, axis=0)
        return cls(vmin[0], vmin[1], vmax[0], vmax[1])

    @property
    def transfo(self):
        return self._transfo
    
    @property
    def bbox(self):
        if self.transfo.touched:
            self._bbox = self.transformed().bbox
            self.transfo.touch(False)

        return self._bbox
    
    @classmethod
    def as_zone(cls, other):
        if isinstance(other, Zone):
            return other
        try:
            return cls(*other)
        except Exception as e:
            msg = str(e)

        raise ValueError(f"Cannot convert {other} to a 4-tuple: {msg}")
    
    # ====================================================================================================
    # Transformation
    # ====================================================================================================

    @property
    def points(self):
        return np.array([
            [self._x0, self._y0],
            [self._x1, self._y0],
            [self._x1, self._y1],
            [self._x0, self._y1],
            ], dtype=float)
    
    def transformed(self):
        points = self._transfo @ self.points
        vmin, vmax = np.min(points, axis=0), np.max(points, axis=0)
        return Zone(vmin[0], vmin[1], vmax[0], vmax[1])

    # ====================================================================================================
    # Zone dimensions
    # ====================================================================================================

    @property
    def x0(self):
        return self.bbox[0]
        
    @property
    def y0(self):
        return self.bbox[1]
    
    @property
    def x1(self):
        return self.bbox[2]
    
    @property
    def y1(self):
        return self.bbox[3]
    
    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0
    
    @property
    def center(self):
        return (self.x0 + self.x1)/2, (self.y0 + self.y1)/2
    
    # ====================================================================================================
    # Merge
    # ====================================================================================================

    def merge(self, *others):
        x0, y0, x1, y1 = self.bbox
        for other in others:
            x0 = min(x0, other.x0)
            y0 = min(y0, other.y0)
            x1 = max(x1, other.x1)
            y1 = max(y1, other.y1)

        return Zone(x0, y0, x1, y1)

    # ====================================================================================================
    # Operations
    # ====================================================================================================

    def x_align(self, x, *others, align='left', margin=0.0):

        # ----- Align together with other zones
        if others:
            zone = self.merge(*others)
            zone.x_align(x, align=align, margin=margin)
            self.transfo.apply_transfo(zone.transfo)
            for other in others:
                other.transfo.apply_transfo(zone.transfo)
            return self
        
        # ----- Only self
        align = align.lower()

        if align == 'left':
            x_ref = self.x0
        elif align == 'center':
            x_ref = self.x0 + self.width / 2
        elif align == 'right':
            x_ref = self.x1
        else:
            raise ValueError(f"Unknown x-align: {align}")

        delta = x - x_ref
        self.transfo.translate(delta + margin, 0.0)
        return self

    def y_align(self, y, *others, align='bottom', margin=0.0):

        # ----- Align together with other zones
        if others:
            zone = self.merge(*others)
            zone.y_align(y, align=align, margin=margin)
            self.transfo.apply_transfo(zone.transfo)
            for other in others:
                other.transfo.apply_transfo(zone.transfo)
            return self

        # ----- Only self
        align = align.lower()

        if align in ['bottom', 'bot']:
            y_ref = self.y0
        elif align in ['center', 'middle']:
            y_ref = self.y0 + self.height / 2
        elif align in ['top']:
            y_ref = self.y1
        else:
            raise ValueError(f"Unknown y-align: {align}")

        delta = y - y_ref
        self.transfo.translate(0.0, delta + margin)
        return self
    
    # ====================================================================================================
    # Relative placement
    # ====================================================================================================

    def after(self, other, *, margin=0.0):
        self.x_align(other.x1 + margin, align='left')
        return self

    def before(self, other, *, margin=0.0):
        self.x_align(other.x0 -  margin, align='right')
        return self

    def above(self, other, align='center', *, margin=0.0):
        self.y_align(other.y1 + margin, align='bottom')
        self.x_align(other.center[0], align=align)
        return self
    
    def below(self, other, align='center', *, margin=0.0):
        self.y_align(other.y0 - margin, align='top')
        self.x_align(other.center[0], align=align)
        return self
    
    def superscript(self, other, scale, y, *, margin=0.0):
        self.transfo.apply_scale(scale)
        self.after(other, margin=margin)
        self.y_align(y, align='bottom')
        return self
        
    def subscript(self, other, scale, y, *, margin=0.0):
        self.transfo.apply_scale(scale)
        self.after(other, margin=margin)
        self.y_align(y, align='top')
        return self

    # ====================================================================================================
    # Debug
    # ====================================================================================================

    def _plot(self, plt, color='k', alpha=1.0, **kwargs):
        """Plot the zone on a matplotlib figure."""
        points = self.transfo @ self.points
        x, y = list(points[:, 0]), list(points[:, 1])
        x.append(x[0])
        y.append(y[0])

        # Plot the bounding box
        plt.plot(x, y, color=color, alpha=alpha, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    margin = .1

    zone = Zone(-1, -1, 1, 1)

    exp = Zone(-1, -1, 1, 1)
    exp.superscript(zone, scale=.8, y=.5, margin=margin)

    sub = Zone(-1, -1, 1, 1)
    sub.subscript(zone, scale=.8, y=-.1, margin=margin)

    bef = Zone(-1, -1, 1, 1)
    bef.transfo.apply_scale(.5, 1.5)
    bef.before(zone, margin=margin)

    exp.x_align(0, zone, bef, sub, align='left')

    zone._plot(plt, color='k')
    exp._plot(plt, color='r')
    sub._plot(plt, color='blue')
    bef._plot(plt, color='pink')


    plt.axis("equal")
    plt.show()
    
