__all__ = ["Transfo2d", "BBox", "Formula", "FormulaBox", "FormulaGeom"]

import numpy as np
import math

if __name__ == '__main__':
    from constants import TAU, PI, ZERO
else:
    from .constants import TAU, PI, ZERO

INT_SYMBS = {
    'int': '∫',
    'iint': '∬',
    'iiint': '∭',
    'oint': '∮',
    'oiint': '∯',
    'oiiint': '∰',        
}

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

    def __str__(self):
        sx, sy = self.scale
        tx, ty = self.translation
        angle_deg = math.degrees(self.rotation)
        return (f"<T: "
                f"tr: ({tx:.2f}, {ty:.2f}), "
                f"sc: ({sx:.2f}, {sy:.2f}), "
                f"ag: {angle_deg:.1f}°"
                f">")

    def reset(self):
        self._matrix[...] = np.eye(3, dtype=float)

    def clone(self):
        return Transfo2d(np.array(self._matrix))
    
    def copy(self, other):
        self._matrix[:] = other._matrix

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

    @property
    def translation(self):
        # Returns a view (2,)
        return self._matrix[:2, 2]

    @translation.setter
    def translation(self, value):
        v = np.asarray(value, dtype=float).reshape(2)
        self._matrix[:2, 2] = v

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
        null_scale = False
        try:
            R = np.linalg.solve(P, A)          # R = P^{-1} @ A
        except:
            null_scale = True

        if null_scale:
            return np.zeros(2, dtype=float), 0., np.eye(2)

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

    @property
    def transformation3d(self):
        if __name__ == '__main__':
            from constants import bfloat
            from transformation import Transformation
        else:
            from .constants import bfloat
            from .transformation import Transformation

        M4 = np.eye(4, dtype=bfloat)
        M4[:2, :2] = self._matrix[:2, :2]
        M4[:2, 3]  = self._matrix[:2, 2]

        return Transformation(M4, copy=True)

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
        return self


    def rotate(self, angle, pivot=None):
        """Apply a rotation AFTER the current transform (column-vector convention)."""
        if pivot is not None:
            self.translate(pivot, False)

        R = type(self).new(rotation=angle).matrix
        self.matrix[...] = R @ self.matrix

        if pivot is not None:
            self.translate(pivot, True)

        return self

    def apply_scale(self, *scale, pivot=None):
        """Apply a scaling AFTER the current transform (column-vector convention)."""

        if pivot is not None:
            self.translate(pivot, False)

        S = type(self).new(scale=scale).matrix
        self.matrix[...] = S @ self.matrix

        if pivot is not None:
            self.translate(pivot, True)

        return self
    
    # -------------------------------------
    # Interpolation
    # -------------------------------------

    def interpolate(self, other, factor=.5, ymax=0., turns=0, smooth='LINEAR'):
        """Interpolate with another transformation.
        """
        from .easings import maprange

        if not isinstance(other, Transfo2d):
            raise ValueError(f"Argument 'other' must be a 'Transfo2d', not '{type(other).__name__}'")
        
        # Read the components
        tr0 = self.translation
        sc0, rot0, _ = self._scale_rot

        tr1 = other.translation
        sc1, rot1, _ = other._scale_rot

        # Rotation angle with optional turns
        angle = ((rot1 - rot0 + PI)%TAU) - PI
        angle += int(turns)*TAU

        # Interpolation
        tr = maprange(factor, 0., 1., tr0, tr1, mode=smooth)
        sc = maprange(factor, 0., 1., sc0, sc1, mode=smooth)
        rot = maprange(factor, 0., 1., rot0, rot0 + angle, mode=smooth)

        f = maprange(factor, mode="SMOOTH")
        dy = maprange(abs(f - 0.5), 0.5, 0.0, 0., ymax, mode='CIRCULAR.OUT')

        # Return the intermediate transformation
        return Transfo2d.from_components(tx=tr[0], ty=tr[1] + dy, sx=sc[0], sy=sc[1], angle=rot)

    
# ====================================================================================================
# A Bounding Box
# ====================================================================================================

class BBox:
    def __init__(self, *bounds):
        """Non mutable bound box.
        """
        if len(bounds) == 1:
            self.set_bbox(bounds[0])
        else:
            self.set_bbox(bounds)

    def __str__(self):
        return f"<BBox {self.x0:.2f}, {self.y0:.2f}, {self.x1:.2f}, {self.y1:.2f}>"
    
    @classmethod
    def from_points(cls, points):
        try:
            vmin, vmax = np.min(points, axis=0), np.max(points, axis=0)
        except:
            return BBox()
        return cls(vmin[:2], vmax[:2])

    def as_tuple(self):
        return self._bbox
    
    def set_bbox(self, bounds):

        if isinstance(bounds, BBox):
            self._bbox = bounds.as_tuple()
            return
        
        if hasattr(bounds, '__len__'):
            if len(bounds) == 0:
                self._bbox = (0., 0., 0., 0.)
                return

            elif len(bounds) in (1, 2):
                bounds = np.asarray(bounds, dtype=float)
                if bounds.size == 4:
                    self._bbox = tuple(bounds.ravel())
                return
                
            elif len(bounds) == 4:
                self._bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
                return

        raise ValueError(f"BBox must be initialized with 4 floats or 2 vectors")        
    
    @classmethod
    def as_bbox(cls, other):
        if isinstance(other, BBox):
            return other
        try:
            return cls(other)
        except Exception as e:
            msg = str(e)

        raise ValueError(f"Cannot convert {other} to a 4-tuple: {msg}")
    
    # ====================================================================================================
    # Transformation
    # ====================================================================================================

    @property
    def corners(self):
        return np.array([
            [self.x0, self.y0],
            [self.x1, self.y0],
            [self.x1, self.y1],
            [self.x0, self.y1],
            ], dtype=float)
    
    def transform(self, transfo):
        return BBox.from_points(transfo @ self.corners)
    
    # ====================================================================================================
    # Box dimensions
    # ====================================================================================================

    @property
    def x0(self):
        return self._bbox[0]
        
    @property
    def y0(self):
        return self._bbox[1]
    
    @property
    def x1(self):
        return self._bbox[2]
    
    @property
    def y1(self):
        return self._bbox[3]
    
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
    # Join
    # ====================================================================================================

    def join(self, *others):
        x0, y0, x1, y1 = self._bbox
        for other in others:
            x0 = min(x0, BBox.as_bbox(other).x0)
            y0 = min(y0, BBox.as_bbox(other).y0)
            x1 = max(x1, BBox.as_bbox(other).x1)
            y1 = max(y1, BBox.as_bbox(other).y1)

        return BBox(x0, y0, x1, y1)
    
    def __add__(self, other):
        return self.join(other)
    
    def __radd__(self, other):
        return self.join(other)
    
    def __iadd__(self, other):
        self.set_bbox(self.join(other))
        return self
    
    # ====================================================================================================
    # Utilities
    # ====================================================================================================

    def translated(self, tx=0.0, ty=0.0):
        x0, y0, x1, y1 = self.as_tuple()
        return BBox(x0 + tx, y0 + ty, x1 + tx, y1 + ty)

    def scaled(self, *scale, align='center'):
        if len(scale) == 0:
            return BBox(self)
        elif len(scale) == 1:
            if hasattr(scale[0], '__len__'):
                sx, sy = scale[0]
            else:
                sx, sy = scale[0], scale[0]
        else:
            sx, sy = scale

        if align.lower() == 'left':
            cx, cy = self.x0, self.y0
        elif align.lower() == 'right':
            cx, cy = self.x1, self.y1
        else:
            cx, cy = self.center
        
        x0, y0, x1, y1 = self.x0 - cx, self.y0 - cy, self.x1 - cx, self.y1 - cy 
        return BBox(cx + sx*x0, cy + sy*y0, cx + sx*x1, cy + sy*y1)
    
    def bordered(self, border):
        return BBox(self.x0 - border, self.y0 - border, self.x1 + border, self.y1 + border)
    
    def interpolate(self, other, factor):
        factor = np.clip(factor, 0, 1)
        a, b = 1.0 - factor, factor
        return BBox(
            self.x0*a + other.x0*b, 
            self.y0*a + other.y0*b, 
            self.x1*a + other.x1*b, 
            self.y1*a + other.y1*b,
            )

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


# ====================================================================================================
# Animation
# ====================================================================================================

class Animation:
    def __init__(self):
        """FormulaBox animation

        Control size, location and rotation of FormulaBoxes
        """
        self._tx, self._ty = 0.0, 0.0
        self._sx, self._sy = 1.0, 1.0
        self._rot = 0.0
        self._z = 0.0

    @property
    def tx(self):
        """Translation along x
        """
        return self._tx
    
    @tx.setter
    def tx(self, value):
        self._tx = value

    @property
    def ty(self):
        """Translation along y
        """
        return self._ty
    
    @ty.setter
    def ty(self, value):
        self._ty = value

    @property
    def sx(self):
        """Scale along x
        """
        return self._sx
    
    @sx.setter
    def sx(self, value):
        self._sx = value

    @property
    def sy(self):
        """Scale along y
        """
        return self._sy
    
    @sy.setter
    def sy(self, value):
        self._sy = value

    @property
    def rot(self):
        """Rotation around z
        """
        return self._rot
    
    @rot.setter
    def rot(self, value):
        self._rot = value

    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        self._z = value

    @property
    def translation(self):
        """Translation (tx, ty).
        """
        return self.tx, self.ty

    @property
    def scale(self):
        """Scale (sx, sy)
        """
        return self.sx, self.sy
    
    @property
    def transfo(self):
        """The Transfo2d buit with properties
        """
        return Transfo2d.from_components(tx=self.tx, ty=self.ty, sx=self.sx, sy=self.sy, angle=self.rot)

    @property
    def scale_rot_transfo(self):
        return Transfo2d.from_components(angle=self.rot) @ Transfo2d.from_components(sx=self.sx, sy=self.sy)

    @property
    def translation_transfo(self):
        return Transfo2d.from_components(tx=self.tx, ty=self.ty)

# ====================================================================================================
# An area
# ====================================================================================================

class FormulaBox:

    def __init__(self, name="FormulaBox"):
        """The base class for box in a formula.

        A FormulaBox is made of a boundingbox and a 2D transformation.
        The 2D transformation is used by the owner to place the box at the right place.
        """

        self.name      = name
        self._transfo  = Transfo2d()
        self._bbox     = None
        self.anim      = Animation()

        self._adjust_size = None 

        # ----- Tree structure

        self.owner        = None # Owner
        self.child_key    = None # Key in owner
        self.child_index  = None
        self.children     = []   # List of children

    # ====================================================================================================
    # Tree management
    # ====================================================================================================

    def detach(self):
        if self.owner is not None and self in self.owner.children:
            self.owner.children.remove(self)

        self.child_key   = None
        self.child_index = None
        return self

    def attach_to(self, owner, child_key, **kwargs):

        self.detach()

        self.owner       = owner
        self.child_key   = child_key
        if owner is not None:
            self.owner.children.append(self)

        for k, v in kwargs.items():
            setattr(self, k, v)

        return self
    
    def all_children(self, depth=False):

        child_depth = 1
        for child in self.children:
            if depth:
                yield child, child_depth
            else:
                yield child

            for c, d in child.all_children(True):
                if depth:
                    yield c, child_depth + d
                else:
                    yield c

    # ====================================================================================================
    # Hierarchical attributes read
    # ====================================================================================================

    def __getattr__(self, name):
        if self.owner is None:
            raise AttributeError(f"'{name}' is not a valid attribute for '{type(self).__name__}'.")
        
        return getattr(self.owner, name)

    # ====================================================================================================
    # Dump
    # ====================================================================================================

    def __str__(self):
        return f"<{type(self).name} '{self.name}' {self.transfo}>"
    
    def __repr__(self):
        lines = [str(self)]
        for c, d in self.all_children(True):
            lines.append(f"{'   '*d}" + str(c))
        return "\n".join(lines)
    
    # ====================================================================================================
    # Horizontal separation after
    # ====================================================================================================

    @property
    def sepa_after(self):
        return True

    # ====================================================================================================
    # Transformations
    # ====================================================================================================

    def reset_transfo(self):
        self._transfo = self.anim.scale_rot_transfo
    
    @property
    def transfo(self):
        return self._transfo
    
    def get_transfo(self, owner_transfo=None):
        if owner_transfo is None:
            return self.transfo
        else:
            return owner_transfo @ self.transfo
        
    @property
    def absolute_transfo(self):
        if self.owner is None:
            return self.transfo
        else:
            return self.get_transfo(self.owner.absolute_transfo)
        
    @property
    def transfo3d(self):

        T = self.transfo.transformation3d
        T.z = self.anim.z

        if self.owner is None:
            return T
        else:
            return self.owner.transfo3d @ T
    
    # ====================================================================================================
    # BBox
    # ====================================================================================================

    def get_bbox(self):
        raise NotImplementedError(f"get_bbox must be implemented")

    def reset_bbox(self):
        for c in self.children:
            c.reset_bbox()

        self._bbox = self.get_bbox()

    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = self.get_bbox()

        if self._adjust_size is None:
            return self._bbox
        
        width, height = self._adjust_size
        x0, y0, x1, y1 = self._bbox.as_tuple()
        return BBox(x0, y0, max(x1, x0 + width), max(y1, y0 + height))
    
    @property
    def transformed_bbox(self):
        try:
            return self.bbox.transform(self.transfo)
        except Exception as e:
            print(str(e))
            raise e
    
    # ====================================================================================================
    # Operations
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # X alignment
    # ----------------------------------------------------------------------------------------------------

    def x_align(self, x, align='left', margin=0.0):
        """Update the transformation to align the bbox horizontally.

        Note : the margin is multiplied by the animation scale

        Parameters
        ----------
        x : float
            Position to align the bbox to
        align : str in ('left', 'center', 'right'), optional
            alignment mode
            default is 'left'
        margin : float, optional
            additional margin
            default is 0.0

        Returns
        -------
        self
        """

        align = align.lower()
        bbox = self.transformed_bbox

        if align == 'left':
            x_ref = bbox.x0
        elif align in ['center', 'middle']:
            x_ref = bbox.x0 + bbox.width / 2
        elif align == 'right':
            x_ref = bbox.x1
        else:
            raise ValueError(f"Unknown x-align: {align}")

        delta = x - x_ref
        self.transfo.translate(delta + margin*self.anim.sx, 0.0)

        return self

    # ----------------------------------------------------------------------------------------------------
    # Y alignment
    # ----------------------------------------------------------------------------------------------------

    def y_align(self, y, align='bottom', margin=0.0):
        """Update the transformation to align the bbox vertically.

        Note : the margin is multiplied by the animation scale

        Parameters
        ----------
        x : float
            Position to align the bbox to
        align : str in ('bottom', 'middle', 'top'), optional
            alignment mode
            default is 'bottom'
        margin : float, optional
            additional margin
            default is 0.0

        Returns
        -------
        self
        """

        align = align.lower()
        bbox = self.transformed_bbox

        if align in ['bottom', 'bot']:
            y_ref = bbox.y0
        elif align in ['center', 'middle']:
            y_ref = bbox.y0 + bbox.height / 2
        elif align in ['top']:
            y_ref = bbox.y1
        else:
            raise ValueError(f"Unknown y-align: {align}")

        delta = y - y_ref
        self.transfo.translate(0.0, delta + margin*self.anim.sy)
        return self
    
    # ----------------------------------------------------------------------------------------------------
    # Adjust size
    # ----------------------------------------------------------------------------------------------------
    
    def adjust_size(self, width=0.0, height=0.0):
        self._adjust_size = width, height
    
    # ====================================================================================================
    # DEBUG
    # ====================================================================================================

    def _plot(self, plt, color=None, full=False, **kwargs):

        def plot_bbox(transfo, bbox, col, name=None, **attrs):
            points = bbox.corners
            if transfo is not None:
                points = transfo @ points

            x, y = list(points[:, 0]), list(points[:, 1])
            x.append(x[0])
            y.append(y[0])

            plt.plot(x, y, color=col, **attrs)

            # Plot the name
            if name is not None:
                xs, ys = (x[0] + x[1])/2, (y[1] + y[2])/2
                plt.text(xs, ys, name, ha="center", va="center", color=col, fontsize=12)

        if full:
            bbox = self.bbox
            plot_bbox(None, bbox, 'gray', **kwargs)

            transfo = self.transfo
            plot_bbox(transfo, bbox, 'black', **kwargs)

        if color is None:
            color = np.random.uniform(0, 1, 3)

        plot_bbox(None, self.bbox.transform(self.absolute_transfo), color, name=self.name, **kwargs)

    def _plot_global(self, plt, color=None, **kwargs):
        
        bbox = self.bbox.transform(self.absolute_transfo).bordered(.1)
        points = bbox.corners

        x, y = list(points[:, 0]), list(points[:, 1])
        x.append(x[0])
        y.append(y[0])

        if color is None:
            color = np.random.uniform(0, 1, 3)

        plt.plot(x, y, color=color, **kwargs)

        # Plot the name
        xs, ys = (x[0] + x[1])/2, (y[1] + y[2])/2
        plt.text(xs, ys, self.name, ha="center", va="center", color=color, fontsize=14)


# ====================================================================================================
# The content of a term
# ====================================================================================================

class FormulaGeom(FormulaBox):

    def __init__(self, term, content, symbol=True, name=None):
        """The base class for actual geometry box.

        This class is necessarily initialized with an owner term to benefit
        from attribute reading.
        """

        if name is None:
            name = str(content)

        super().__init__(name=name)

        # Owning formula term
        self.term   = term
        self.symbol = symbol

        # Create the content
        self.set_content(content)

    def __str__(self):
        return f"<{type(self).__name__}: {self.name}>"
    
    @property
    def font(self):
        if self.symbol:
            font = self.term.math_font
            if font is not None:
                return font
            
        return self.term.font

    # ---------------------------------------------------------------------------
    # Interface to be implemented
    # ---------------------------------------------------------------------------

    def set_content(self, content):
        assert False, "'set_content' not implemented"

    def get_bbox(self):
        assert False, "'get_bbox' not implemented"

# ====================================================================================================
# A place Holder
# ====================================================================================================

class PlaceHolder(FormulaBox):
    def __init__(self, width=0., height=0., name="PlaceHolder", **attrs):
        """PlaceHolder is treated as a term but get its size from external info.

        A PlaceHolder can be:
        - fixed size box (default)
        - get size from an external formula which doesn't belong to the PlaceHolder
        - anim effect between two formulas which belong to the PlaceHolder
        """

        super().__init__(name=name)

        self.name  = name
        self.fix(width, height)
        
        for k, v in attrs.items():
            setattr(self, k, v)

    # ====================================================================================================
    # Initializers
    # ====================================================================================================

    def fix(self, width, height):
        for c in self.children:
            c.detach()

        self.mode = 'FIX'
        self.width = width
        self.height = height

    def placeholder(self, formula):
        if formula is None:
            raise ValueError(f"PlaceHolder in mode 'PLACEHOLDER' needs a not None formula.")

        for c in self.children:
            c.detach()

        self.mode = 'PLACEHOLDER'
        self.formula = formula

        return self

    def switch(self, from_formula, to_formula, factor=0.5):
        if from_formula is None or to_formula is None:
            raise ValueError(f"PlaceHolder in mode 'SWITCH' needs two not None formulas to switch between.")
        
        for c in self.children:
            c.detach()

        self.mode = 'SWITCH'
        from_formula.attach_to(self, 'from_formula')
        to_formula.attach_to(self, 'to_formula')
        self.factor = factor

        return self
    
    def jump(self, formula, to_placeholder, factor=0.5):

        if formula is None:
            raise ValueError(f"PlaceHolder in mode 'JUMP' needs a not None Formula.")
        if to_placeholder is None:
            raise ValueError(f"PlaceHolder in mode 'JUMP' needs a not None target PlaceHolder to jump to.")
        
        for c in self.children:
            c.detach()

        self.mode = 'JUMP'
        self.formula = formula
        self.factor  = factor

        self.to_ph = to_placeholder
        self.to_ph.placeholder(self.formula)

        # Compute the two jump transformations
        self.jump_init()

        return self
    
    # ====================================================================================================
    # str
    # ====================================================================================================

    def __str__(self):
        if self.mode == 'FIX':
            return f"<PlaceHolder[{self.mode}] '{self.name}' width: {self.width:.2f}, height: {self.height:.2f}>"
        else:
            return f"<PlaceHolder[{self.mode}] '{self.name}'>"

    # ====================================================================================================
    # Sepa after
    # ====================================================================================================

    @property
    def sepa_after(self):
        if self.mode == 'FIX':
            return False
        elif self.mode == 'PLACEHOLDER':
            return self.formula.sepa_after
        elif self.mode == 'SWITCH':
            return self.children[0].sepa_after
        elif self.mode == 'JUMP':
            return self.formula.sepa_after

    # ====================================================================================================
    # Expose the bbox of the encapsulated formula
    # ====================================================================================================

    def reset_bbox(self):
        if self.mode == 'PLACEHOLDER':
            self.formula.reset_bbox()

        elif self.mode == 'JUMP':
            # Set other PH scale (will reset formula bbox)
            self.to_ph.anim.sx = self.factor

            self.anim.sx = 1.0 - self.factor

        super().reset_bbox()

    def get_bbox(self):

        if self.mode == 'FIX':
            return BBox(0, 0, self.width, self.height)
        
        if self.mode == 'PLACEHOLDER':
            return self.formula.bbox
        
        if self.mode == 'SWITCH':

            f = self.factor

            frm0, frm1 = self.children
            frm0.reset_transfo()
            frm1.reset_transfo()

            frm0.transfo.apply_scale(1 - f, 1.)
            frm1.transfo.apply_scale(f, 1.)
            frm1.transfo.translate(frm0.transformed_bbox.width, 0.)

            return frm0.transformed_bbox + frm1.transformed_bbox
        
        if self.mode == 'JUMP':
            self.anim.sx = 1 - self.factor
            self.to_ph.anim.sx = self.factor
            return self.formula.bbox
        
        assert False, "Shouldn't occur"

    # ====================================================================================================
    # Specific to jump
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Initial and final absolute transfo
    # ----------------------------------------------------------------------------------------------------

    def jump_init(self):

        self.transfo0 = None
        self.transfo1 = None

        if self.mode != 'JUMP':
            return
        
        # ----- Get the top formula
        top = self
        if top is None:
            return
        
        while top.owner is not None:
            top = top.owner

        # ----- Compute the formula for the two cases

        # Store current factor
        factor = self.factor

        # Factor = 0
        self.factor = 0.0
        top.reset_bbox()
        self.transfo0 = self.absolute_transfo

        # Factor = 1
        self.factor = 1.0
        top.reset_bbox()
        self.transfo1 = self.to_ph.absolute_transfo

        # Restore current factor
        self.factor = factor

    # ----------------------------------------------------------------------------------------------------
    # Intermediary transfos
    # ----------------------------------------------------------------------------------------------------

    def get_jump_transfo(self, ymax):
        if self.mode != 'JUMP':
            return Transfo2d()

        return self.transfo0.interpolate(self.transfo1, self.factor, ymax=ymax)


# ====================================================================================================
# A Formula
# ====================================================================================================

class Formula(FormulaBox):

    ATTRIBUTES = {
        'geom_cls'      : FormulaGeom,
        'font'          : None,
        'math_font'     : None,

        'x_sepa'        : 0.1,  # Space between successive terms
        'y_sepa'        : 0.1,  # Space between arranged one over the oher one
        'x_sepa_factor' : 1.0,  # Factor to apply to x_sepa
        'em'            : 0.6,  # em unit (around M width)
        'script_scale'  : 0.5,  # Scripts scale
        'oper_scale'    : 1.2,  # Over scale for integral, sum,...
        'dy_super'      : 0.1,  # Location of superscript: y1 - y_super
        'y_super_min'   : 0.35, # Minimum y location for superscript
        'dy_sub'        : 0.15, # Location of subscript y0 + y_sub
        'y_sub_max'     : 0.15, # Max y location of subscript
        'thickness'     : 0.07, # Bar thickness : arrow, bar, borders...
        'oversize'      : 0.04, # Delta oversize for decorators
        'elon_mode'     : 'CENTER', # Elongation type 
        'elon_margin'   : 0.03, # Margin for decorator elongation
        'elon_smooth'   : 'SMOOTH', # For decorator elongation

        'int_shear'     : 0.0, # ∫ symbol shear
        'sub_offset'    : 0.0,  # Subscript offset (when sheared)
        'int_sub_ofs'   : -0.2, # Subscript offset for integral

        'y_fraction'    : 0.22, # Vertical position of fraction bar
        'dy_fraction'   : 0.1,  # Separation between bar and operands
        'dx_fraction'   : 0.1, # Horizontal over space for fraction

    }

    # Params for future use
    DECORATORS = {
        'subscript'     : {}, 
        'superscript'   : {},
        'overscript'    : {},
        'underscript'   : {},

        'fix_over'      : {},
        'over'          : {},
        'fix_under'     : {},
        'under'         : {},

        'fix_right'     : {},
        'right'         : {},
        'fix_left'      : {},
        'left'          : {},

        'operation'     : {},
        'fix_operation' : {},

        'denominator'   : {},
        'fraction_bar'  : {},

        'sqrt'          : {},
        'sqrt_option'   : {},
    }

    # ====================================================================================================
    # Initialize
    # ====================================================================================================

    def __init__(self, owner=None, body=None, name="Formula", **attrs):
        """Formula

        The body of the formula is a list of Formulas.
        The formula also belongs decorators.

        The bbox is computed:
        - by putting the terms in the body one right to the other one
        - then by adding decorators
        """

        super().__init__(name)

        # ----------------------------------------
        # Initialization with no ownership
        # ----------------------------------------

        self.owner     = owner
        self.child_key = None

        # ----------------------------------------
        # Decorators
        # ----------------------------------------

        self._decos = {}

        # ----------------------------------------
        # Set the attributes which are not decorators
        # ----------------------------------------

        has_decorator = False
        for k, v in attrs.items():
            if k in Formula.DECORATORS:
                has_decorator = True
            else:
                setattr(self, k, v)

        # ----------------------------------------
        # The list of areas forming the content
        # ----------------------------------------

        self.body = []

        if body is None:
            pass

        elif isinstance(body, FormulaBox):
            self.body.append(body.attach_to(self, 'term', child_index=0))

        elif isinstance(body, list):
            for i, item in enumerate(body):
                self.body.append(self._to_term(item).attach_to(self, 'term', child_index=i))

        else:
            self.body.append(self._to_term(body).attach_to(self, 'term', child_index=0))

        # ----------------------------------------
        # Set decorators
        # ----------------------------------------

        # Decorators are set in priority order
        for k in Formula.DECORATORS:
            if k not in attrs:
                continue
            self.set_decorator(k, attrs[k])

    # ====================================================================================================
    # Str / repr
    # ====================================================================================================

    def __str__(self):

        if self.owner is None:
            skey = '(ROOT)'
        else:
            skey = f"({str(self.child_key)})"

        slen = f"[{len(self.body)}]"

        sname = f"'{self.name}'"
        s = f"<Term {slen} {sname:10s}: {skey:15s}"

        return f"{s}, {str(self.transfo)[1:-1]}>"
    
    def __repr__(self):

        lines = [str(self)]
        #for depth, term in self.depths():
        for term, depth in self.all_children(True):
            if term.child_key == 'term':
                lines.append(f"{'   '*(depth+1)}{term.child_index:2d}: {str(term)}")
            else:
                lines.append(f"{'   '*(depth+1)}    {term.child_key}: {str(term)}")


        return "\n".join(lines)
    
    # ====================================================================================================
    # Special
    # ====================================================================================================

    def _to_term(self, body):

        if body is None:
            return None

        elif isinstance(body, FormulaBox):
            return body
        
        geom_cls = self.geom_cls
        if geom_cls is None:
            raise RuntimeError(
                "The class 'geom_cls' is not defined in the formula. "
                "Use geom_cls=class_name argument when create the Formula.")

        if isinstance(body, dict):
            if 'type' in body:
                return self.parse_dict(body)
            else:
                return geom_cls(self, **body)
            
        else:
            return geom_cls(self, body)


        #elif isinstance(body, str):
        #    return geom_cls(self, body)
        
        #else:
        #    return geom_cls(self, str(body))

    @staticmethod
    def to_underover(attrs):
        new_attrs = {}
        for k, v in attrs.items():
            if k == 'subscript':
                new_attrs['underscript'] = v
            elif k == 'superscript':
                new_attrs['overscript'] = v
            else:
                new_attrs[k] = v
        return new_attrs
    
    @staticmethod
    def symbol_string(s, func_style):

        if func_style is not None:
            s = func_style.same_style(s)
        return {'type': 'SYMBOL', 'string': s}

    def parse_dict(self, fdict):

        if fdict['type'] in ['STRING', 'SYMBOL']:

            attrs = {k: v for k, v in fdict.items() if k not in ['type', 'string']}
            name = str(fdict['string'])
            # Space ~ 
            if fdict['type'] == 'SYMBOL' and name == " ":
                name = "Space"
                content = self.placeholder(name=name)

                # Only width, no sepa
                content.width=fdict.get('width', .434)*self.em
                content.x_sepa_factor=0.0

            # Other
            else:
                content = self.geom_cls(self, fdict['string'], symbol=fdict['type']=='SYMBOL')

            if len(attrs):
                return Formula(self, content, name=name, **attrs)
            else:
                return content

        elif fdict['type'] == 'BLOCK':
            attrs = {k: v for k, v in fdict.items() if k not in ['type', 'content']}
            terms = [self.parse_dict(d) for d in fdict['content']]

            if len(terms) == 1:
                return Formula(self, terms[0], **attrs)
            else:
                return Formula(self, terms, **attrs)
            
        elif fdict['type'] == 'FUNCTION':

            name    = fdict['name']
            option  = fdict.get('option')
            args    = fdict.get('args')
            attrs   = {k: v for k, v in fdict.items() if k not in ['type', 'name', 'options', 'args']}
            fstyle  = attrs.get('func_style')

            if name == 'sum':
                return self.operator(self.symbol_string('Σ', fstyle), args[0], **self.to_underover(attrs))

            elif name == 'prod':
                return self.operator(self.symbol_string('Π', fstyle), args[0], **self.to_underover(attrs))
            
            elif name in INT_SYMBS:
                return self.operator(self.symbol_string(INT_SYMBS[name], fstyle), args[0], sub_offset=self.int_sub_ofs, **attrs)
            
            elif name in ['lim', 'limsup', 'liminf']:
                return self.fix_operator(self.symbol_string(name, fstyle), args[0], **self.to_underover(attrs))
            
            elif name in ['frac', 'tfrac', 'dfrac']:
                return self.fraction(args[0], args[1], **attrs)
            
            elif name == 'sqrt':
                return self.sqrt(args[0], option=option, **attrs)
            
            elif name == 'binom':
                return self.binom(args[0], args[1])
            
            elif name == 'term':
                name = "Term" if option is None else str(option)
                return Formula(self, args[0], name=name)
            
            elif name == 'ph':
                if True:
                    name = "PlaceHolder" if option is None else str(option)
                    ph = self.placeholder(name=name)
                    return Formula(self, ph, name=f"{name} term", **attrs)
                    return self.append(ph)
                else:
                    print("DEBUG")
                    print(attrs)
                    name = "PlaceHolder" if option is None else str(option)
                    return self.placeholder(name=name)
            



            
            else:
                frm = Formula(self, name, **attrs)
                for a in args:
                    frm.append(a)
                return frm
            
        raise ValueError(f"Unknown type '{fdict['type']}' in dict.\n{fdict}")
    
    @staticmethod
    def _same_style(model, s):
        if model is None:
            return s
        else:
            return model.same_style(s)

    def operator(self, symbol, body, **attrs):
        frm = Formula(self, body)
        if len(attrs):
            frm.operation = Formula(frm, symbol, **attrs)
        else:
            frm.operation = symbol
        return frm

    def fix_operator(self, symbol, body, **attrs):
        frm = Formula(self, body)
        esymb = self._same_style(attrs.get('func_style'), symbol)
        if len(attrs):
            frm.fix_operation = Formula(frm, esymb, **attrs)
        else:
            frm.fix_operation = esymb
        return frm
    
    def fraction(self, numerator, denominator, **attrs):
        if len(attrs):
            numerator = Formula(self, numerator, **attrs)
        frm = Formula(self, numerator)
        frm.denominator = Formula(frm, denominator)

        frm.fraction_bar = Formula(frm, self._same_style(attrs.get('func_style'), "_"))

        return frm
    
    def binom(self, n, k, **attrs):
        if len(attrs):
            n = Formula(self, n, **attrs)
        body = Formula(self, n)
        body.denominator = Formula(body, k)
        eleft = self._same_style(attrs.get('func_style'), "(")
        eright = self._same_style(attrs.get('func_style'), ")")
        return Formula(self, body, left=self.symbol_string(eleft), right=self.symbol_string(eright))
        #return Formula(self, body, left=self.symbol_string("("), right=self.symbol_string(")"))
    
    def sqrt(self, body, option=None, **attrs):

        if len(attrs):
            body = Formula(self, body, **attrs)

        frm = Formula(self, body)
        esqrt = self._same_style(attrs.get('func_style'), r"\sqrt")

        frm.sqrt = self.geom_cls(frm, esqrt)
        #frm.sqrt = self.geom_cls(frm, r"\sqrt")

        return frm

    # ====================================================================================================
    # Owner / Children
    # ====================================================================================================

    def _add_layer(self):
        # Keep the attributes at this level
        # Put content and decorators in another child

        new_term = Formula(self, self.body, name=self.name, **self._decos)
        self._decos = {}
        self.body.clear()
        self.body.append(new_term.attach_to(self, 'term', child_index=0))

    def set_decorator(self, key, child, **kwargs):

        if child is None:
            return None
        else:
            child = self._to_term(child)

        if key in self._decos:
            self._add_layer()

        # We can safely add the deco
        self._decos[key] = child.attach_to(self, key, **kwargs)

        return child
    
    def get_decorator(self, key):
        return self._decos.get(key)
    
    def append(self, term):

        term = self._to_term(term)

        add_layer = isinstance(term, Formula) and len(term._decos) and len(self.body)
        if add_layer or len(self._decos):
            self._add_layer()

        self.body.append(term.attach_to(self, 'term', child_index=len(self.body)))

        return self.body[-1]

    # ----------------------------------------------------------------------------------------------------
    # Iterators
    # ----------------------------------------------------------------------------------------------------

    def terms_OLD(self):

        # Bodies
        for term in self.body:
            yield term

        # Decorators
        for term in self._decos.values():
            yield term
    
    def depths_OLD(self):

        def recur(depth, term):
            if not isinstance(term, Formula):
                return
            
            for child in term.terms():
                yield depth, child

                for ic in recur(depth + 1, child):
                    yield ic

        for ic in recur(0, self):
            yield ic

    def by_name(self, name):
        if self.name == name:
            return self
        
        #for _, frm in self.depths():
        for frm in self.all_children(False):
            if frm.name == name:
                return frm
            
        return None

    # ====================================================================================================
    # Attributes
    # ====================================================================================================

    def __getattr__(self, name):

        if name in Formula.DECORATORS:
            return self._decos.get(name)
        
        elif name in Formula.ATTRIBUTES:
            if self.owner is None:
                return Formula.ATTRIBUTES[name]
            else:
                return getattr(self.owner, name)
        else:
            raise AttributeError(
                f"'{name}' is not a valid attribute for '{type(self).__name__}'. Valid are\n"
                f"{list(Formula.DECORATORS.keys())}"
                f"{list(Formula.ATTRIBUTES.keys())}"
                )
        
    def __setattr__(self, name, value):

        if name in Formula.DECORATORS:
            self.set_decorator(name, value)
        
        else:
            super().__setattr__(name, value)

    # ====================================================================================================
    # Create a place holder
    # ====================================================================================================

    def placeholder(self, name):
        return PlaceHolder(name=name)

    # ====================================================================================================
    # Bounding boxes
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Sepa after
    # ----------------------------------------------------------------------------------------------------

    @property
    def sepa_after(self):
        if len(self.body):
            return self.body[-1].sepa_after
        else:
            return False

    # ----------------------------------------------------------------------------------------------------
    # Body bbox
    # ----------------------------------------------------------------------------------------------------

    def get_body_bbox(self):
        """Surrounding BBox for all the formulas in terms list.
        """
        bbox = BBox()
        use_sepa = False # Sepa from previous terme
        for frm in self.body:

            frm.reset_transfo()

            x = bbox.x1
            margin = self.x_sepa*frm.x_sepa_factor if use_sepa else 0.

            frm.x_align(x, align='left', margin=margin)
            frm.transfo.translate(frm.anim.translation)

            bbox += frm.transformed_bbox

            use_sepa = frm.sepa_after

        return bbox
    
    # ----------------------------------------------------------------------------------------------------
    # Translate the body
    # ----------------------------------------------------------------------------------------------------

    def body_translate(self, tx, ty):
        for frm in self.body:
            frm.transfo.translate(tx, ty)    

    # ----------------------------------------------------------------------------------------------------
    # Adjust size
    # ----------------------------------------------------------------------------------------------------
    
    def adjust_size(self, width=0.0, height=0.0):
        for term in self.body:
            term.adjust_size(width, height)

    # ----------------------------------------------------------------------------------------------------
    # Compute the bbox
    # ----------------------------------------------------------------------------------------------------

    def get_bbox(self):
        """Formula bbox is the body bbox plus decorators
        """
        # ---------------------------------------------------------------------------
        # Starting from the body bbox
        # ---------------------------------------------------------------------------

        bbox = self.get_body_bbox()

        if abs(bbox.x0) > ZERO:
            dx = -bbox.x0
            self.body_translate(dx, 0.)
            bbox = bbox.translated(dx, 0.0)

        # x to place the scripts
        x_script = bbox.x1

        # ---------------------------------------------------------------------------
        # Some decorators need to move the body and the decorators already placed
        # ---------------------------------------------------------------------------

        dones = []
        def _translate_dones(dx=0., dy=0.):
            self.body_translate(dx, dy)
            for term in dones:
                term.transfo.translate(dx, dy)

        # A term over or under can be wider than the body
        def _left_adjust(term):
            width = term.transformed_bbox.width         
            w = (width - bbox.width)/2
            if w > 0:
                term.transfo.translate(w, 0.)
                _translate_dones(w, 0.)
            return width

        # ---------------------------------------------------------------------------
        # Loop on the decorators with respect to their priority
        # ---------------------------------------------------------------------------

        # Make sure left and right symbols make the same size
        left_right_bbox = None

        for key in Formula.DECORATORS:

            if not key in self._decos:
                continue

            # Keys which are treated by their owner
            if key in ['fraction_bar', 'sqrt_option']:
                continue

            term = self._decos[key]
            term.reset_transfo()

            # --------------------------------------------------
            # Term bbox
            # --------------------------------------------------

            # Adjust size if necessary
            if key in ['over', 'under']:
                term.adjust_size(bbox.width + 2*self.x_sepa, 0.0)

            elif key in ['left', 'right']:
                if left_right_bbox is None:
                    left_right_bbox = bbox
                term.adjust_size(0.0, left_right_bbox.height + 2*self.y_sepa)

            elif key in ['operation']:
                term.adjust_size(0.0, bbox.height*self.oper_scale + 2*self.y_sepa)

            # --------------------------------------------------
            # Scripts
            # --------------------------------------------------

            if key in ['subscript', 'superscript', 'underscript', 'overscript']:

                term.transfo.apply_scale(self.script_scale)

                if key in ['subscript', 'superscript']:
                    space = self.script_scale*self.x_sepa
                    if key == 'subscript':
                        space += self.sub_offset
                    term.x_align(x_script, align='left', margin=space)

                    if key == 'superscript':
                        y = max(bbox.y1 - self.dy_super, self.y_super_min)
                        term.y_align(y, align='bottom')
                    else:
                        y = min(bbox.y0 + self.dy_sub, self.y_sub_max)
                        term.y_align(y, align='top')

                    #bbox = bbox + term.transformed_bbox

                elif key in ['underscript', 'overscript']:
                    term.x_align(bbox.center[0], align='middle')
                    if key == 'overscript':
                        term.y_align(bbox.y1 + self.y_sepa, align='bottom')
                    else:
                        term.y_align(bbox.y0 - self.y_sepa, align='top')

                    # Term can be wider
                    width = _left_adjust(term)

                    # Resulting bbox
                    #_, ty0, _, ty1 = term.transformed_bbox.as_tuple()
                    #bbox = bbox + (0, ty0, width, ty1)

            # --------------------------------------------------
            # Over / under
            # --------------------------------------------------

            elif key in ['over', 'under', 'fix_over', 'fix_under']:

                t_bbox = term.transformed_bbox

                term.x_align(bbox.center[0], align='center')

                if key in ['over', 'fix_over']:
                    term.y_align(bbox.y1 + self.y_sepa, align='bottom')
                else:
                    term.y_align(bbox.y0 - self.y_sepa, align='top')

                # Term can be wider
                _left_adjust(term)

                #bbox = bbox + term.transformed_bbox

            # --------------------------------------------------
            # Left / right
            # --------------------------------------------------

            elif key in ['left', 'right', 'fix_left', 'fix_right', 'operation', 'fix_operation']:

                t_bbox = term.transformed_bbox

                if key in ['left', 'fix_left', 'operation', 'fix_operation']:

                    term.x_align(0., align='left')
                    term.y_align(bbox.center[1], align='middle')

                    w = t_bbox.width + self.x_sepa*term.anim.sx*term.x_sepa_factor

                    # Left shift the body and done terms
                    _translate_dones(w, 0)

                    # Resulting bbox
                    #_, ty0, _, ty1 = term.transformed_bbox.as_tuple()
                    #bbox = bbox + (0, ty0, bbox.x1 + w, ty1)
                    bbox = bbox + (0, 0, bbox.x1 + w, 0)

                elif key in ['right', 'fix_right']:
                    x = bbox.x1 + self.x_sepa*self.x_sepa_factor
                    term.x_align(x, align='left')
                    term.y_align(bbox.center[1], align='middle')

                    #bbox = bbox + term.transformed_bbox

            # --------------------------------------------------
            # Fraction
            # --------------------------------------------------

            elif key == 'denominator':

                # fraction bar is treated with denominator
                #if key == 'fraction_bar':
                #    continue

                # Dimensions

                t_bbox = term.bbox

                w_num, h_num = bbox.width, bbox.height
                w_den, h_den = t_bbox.width, t_bbox.height

                # Max width
                width = max(w_num, w_den) + 2*self.dx_fraction
                # x center
                xc = width/2

                # y positions
                y_frac = self.y_fraction
                dy = self.dy_fraction

                y_num = y_frac + dy
                y_den = y_frac - dy

                # Do we have a fraction bar
                bar = self._decos.get('fraction_bar')
                if bar is not None:
                    bar.reset_transfo()
                    bar.adjust_size(width, 0.0)

                    h2 = bar.bbox.height/2
                    y_num += h2
                    y_den -= h2

                    bar.x_align(xc, align='center')
                    bar.y_align(y_frac, align='middle')

                # Let's translate the numerator
                tx = xc - bbox.center[0]
                ty = y_num - bbox.y0
                _translate_dones(tx, ty)

                # Let's align the denominator term
                term.x_align(xc, align='center')
                term.y_align(y_den, align='top')

                # New box
                new_box = BBox(0, y_den - h_den, width, y_num + h_num)
                #bbox = bbox.interpolate(new_box, term.owner_factor)
                bbox = new_box

                # Just in case other decorators
                # Should be useless
                # CAUTION: must be done after call to _translate_dones
                if bar is not None:
                    dones.append(bar)

            # --------------------------------------------------
            # Sqrt
            # --------------------------------------------------

            elif key == 'sqrt':

                # Do we have an option ?
                # NOT YET IMPLEMENTED
                opt = self._decos.get('sqrt_option')

                # The symbol is around the bbox size
                term.adjust_size(bbox.width, bbox.height)
                t_bbox = term.transformed_bbox
                tx = -t_bbox.x0

                # Move body to right
                _translate_dones(tx, 0.0)

                # Adjust term location
                term.transfo.translate(tx, bbox.y0)

                # New bbox
                bbox = term.transformed_bbox

            # --------------------------------------------------
            # Fall Back
            # --------------------------------------------------

            else:
                # Should never occur
                assert(False)

            # --------------------------------------------------
            # Update the bbox
            # --------------------------------------------------

            term.transfo.translate(term.anim.translation)
            bbox = bbox + term.transformed_bbox

            dones.append(term)
            
        # ----- We've got our bounding box

        return bbox

    # ====================================================================================================
    # Some useful animations
    # ====================================================================================================

    def move_to(self, transfo0, transfo1, factor=1.0, ymax=0.0, turns=0, smooth="LINEAR"):

        tr = transfo0.interpolate(transfo1, factor=factor, ymax=ymax, turns=turns, smooth=smooth)
        self._transfo = tr

    # ====================================================================================================
    # Debug
    # ====================================================================================================

    def _plot(self, plt, color='black', border=.2, **kwargs):

        if isinstance(self, Formula):
            n = 0
            #for _ in self.terms():
            for _ in self.children:
                n += 1
            if False and n > 1:
                super()._plot_global(plt, color='red')

        if False:
            print("PLOT", self)

        #for term in self.terms():
        #    term._plot(plt)
        for term in self.children:
            term._plot(plt)



# ====================================================================================================
# Main
# ====================================================================================================

if __name__ == "__main__":

    from pprint import pprint
    import matplotlib.pyplot as plt

    class Fake(FormulaBox):

        def __init__(self, owner, name, bbox=(-1, -1, 1, 1), **kwargs):
            super().__init__(name)
            self.owner = owner
            self.the_bbox = BBox(bbox)

        def __str__(self):
            return f"<Fake {self.name} ({self.child_key}), {self.transfo}>"

        def get_bbox(self):
            return self.the_bbox
        
        def adjust_size(self, width=0, height=0):
            super().adjust_size(width, height)

    np.random.seed(0)
    ok_plot = True
    frm = Formula(None, "x", geom_cls=Fake)

    # ---------------------------------------------------------------------------
    # FormulaBox
    # ---------------------------------------------------------------------------

    if False:
        class FakeArea(FormulaBox):
            def get_bbox(self):
                return BBox(-1, -1, 1, 1)
            
        area = FakeArea("Test")
        area.anim.sx = .5
        area.anim.rot = np.radians(30)
        area.anim.ty = 1.
        area.reset_transfo()
        area.x_align(0, margin=1)
        
        area._transfo.translate(area.anim.translation)

        area._plot(plt)
        plt.axis("equal")
        plt.show()

    # ---------------------------------------------------------------------------
    # Decorators
    # ---------------------------------------------------------------------------

    if False:

        frm = Formula(None, ["x", "y"], geom_cls = Fake,
                    superscript = "2", 
                    subscript   = "i",
                    overscript  = Fake(None, "oscript", (0, 0, 3, .6)),
                    underscript = Fake(None, "uscript", (0, 0, 3, .6)),

                    fix_over  = Fake(None, "over", (0, 0, 2, .6)),
                    fix_under = Fake(None, "under", (0, 0, 2, .6)),
                    over      = Fake(None, "<< over >>", (0, 0, 1, .6)),
                    under     = Fake(None, "<< under >>", (0, 0, 1, .6)),

                    fix_left  = Fake(None, "(", (0, 0, .3, 2)),
                    fix_right = Fake(None, ")", (0, 0, .3, 2)),
                    left      = Fake(None, "[", (0, 0, .3, 1)),
                    right     = Fake(None, "]", (0, 0, .3, 1)),
                    )
        
        if False:
            frm = Formula(None, [frm, "z"], geom_cls = Fake,
                    superscript = "2", 
                    subscript   = "i",
                    overscript  = Fake(None, "oscript", (0, 0, 3, .6)),
                    underscript = Fake(None, "uscript", (0, 0, 3, .6)),

                    fix_over  = Fake(None, "over", (0, 0, 2, .6)),
                    fix_under = Fake(None, "under", (0, 0, 2, .6)),
                    over      = Fake(None, "<< over >>", (0, 0, 1, .6)),
                    under     = Fake(None, "<< under >>", (0, 0, 1, .6)),

                    fix_left  = Fake(None, "(", (0, 0, .3, 2)),
                    fix_right = Fake(None, ")", (0, 0, .3, 2)),
                    left      = Fake(None, "[", (0, 0, .3, 1)),
                    right     = Fake(None, "]", (0, 0, .3, 1)),
                    )
            
    # ---------------------------------------------------------------------------
    # Append
    # ---------------------------------------------------------------------------

    if False:
            frm = Formula(geom_cls=Fake)
            frm.append("x")
            frm.superscript="2"
            frm.superscript="2"
            frm.append("y")
            frm.append("z")
            frm.subscript="sub"
            frm.superscript="sup"
            frm.superscript="sup"
            frm.append("After")
            
    # ---------------------------------------------------------------------------
    # Integral
    # ---------------------------------------------------------------------------

    if False:
        frm = Formula(geom_cls=Fake, name="Main")

        itg1 = Fake(None, "/", (0, 0, .2, .6))
        itg2 = Fake(None, "/", (0, 0, .2, .6))
        frm.append(frm.operator( itg1, "F1", subscript="0", superscript="1") )
        frm.append(frm.fix_operator( itg2, "F2") )
        frm.append(("x"))

    # ---------------------------------------------------------------------------
    # from dict
    # ---------------------------------------------------------------------------

    if False:
        d = {'type': 'BLOCK', 'content': [
                {'type': 'STRING', 'string': 'I='},
                {'type': 'FUNCTION', 'name': 'sum', 'args': [
                    {'type': 'BLOCK', 'content': [
                        {'type': 'STRING', 'string': 'x', 'superscript': "2"},
                        {'type': 'STRING', 'string': '+'},
                        {'type': 'STRING', 'string': 'y', 'superscript': "2"},
                        {'type': 'STRING', 'string': 'dx'},                        
                        ]},
                    ],
                    'subscript': 0,
                    'superscript': 1,
                },
        ]}

        d = {'content': [{'string': '∀', 'type': 'SYMBOL'},
             {'string': 'n', 'type': 'STRING'},
             {'string': '∈', 'type': 'SYMBOL'},
             {'string': 'ℕ', 'type': 'SYMBOL'},
             {'string': ',', 'type': 'SYMBOL'},
             {'string': '∃', 'type': 'SYMBOL'},
             {'string': 'x', 'type': 'STRING'},
             {'string': '∈', 'type': 'SYMBOL'},
             {'string': 'ℝ', 'type': 'SYMBOL'},
             {'content': [{'content': [{'string': ' tel que ', 'type': 'STRING'}],
                           'type': 'BLOCK'}],
              'type': 'BLOCK'},
             {'string': 'sin', 'type': 'STRING'},
             {'string': '(', 'type': 'SYMBOL'},
             {'string': 'x', 'type': 'STRING'},
             {'string': ')', 'type': 'SYMBOL'},
             {'string': '+', 'type': 'SYMBOL'},
             {'string': 'cos', 'type': 'STRING'},
             {'string': '(', 'type': 'SYMBOL'},
             {'string': 'x', 'type': 'STRING'},
             {'string': ')', 'type': 'SYMBOL'},
             {'string': '+', 'type': 'SYMBOL'},
             {'string': 'tan', 'type': 'STRING'},
             {'string': '(', 'type': 'SYMBOL'},
             {'string': 'x', 'type': 'STRING'},
             {'string': ')', 'type': 'SYMBOL'},
             {'string': '+', 'type': 'SYMBOL'},
             {'string': 'arcsin', 'type': 'STRING'},
             {'content': [{'args': [{'content': [{'string': '1',
                                                  'type': 'STRING'}],
                                     'type': 'BLOCK'},
                                    {'content': [{'string': '2',
                                                  'type': 'STRING'}],
                                     'type': 'BLOCK'}],
                           'name': 'tfrac',
                           'type': 'FUNCTION'}],
              'type': 'BLOCK'},
             {'string': '+', 'type': 'SYMBOL'},
             {'string': 'ln', 'type': 'STRING'},
             {'string': '(', 'type': 'SYMBOL'},
             {'content': [{'string': 'e', 'type': 'STRING'}],
              'superscript': {'content': [{'string': 'x', 'type': 'STRING'}],
                              'type': 'BLOCK'},
              'type': 'BLOCK'},
             {'string': ')', 'type': 'SYMBOL'},
             {'string': '+', 'type': 'SYMBOL'},
             {'content': [{'string': 'log', 'type': 'STRING'}],
              'subscript': {'content': [{'string': '10', 'type': 'STRING'}],
                            'type': 'BLOCK'},
              'type': 'BLOCK'},
             {'string': '(', 'type': 'SYMBOL'},
             {'string': '100', 'type': 'STRING'},
             {'string': ')', 'type': 'SYMBOL'},
             {'string': '+', 'type': 'SYMBOL'},
             {'string': 'exp', 'type': 'STRING'},
             {'string': '(', 'type': 'SYMBOL'},
             {'string': '0', 'type': 'STRING'},
             {'string': ')', 'type': 'SYMBOL'},
             {'string': '=', 'type': 'SYMBOL'},
             {'string': '1', 'type': 'STRING'},
             {'string': ',', 'type': 'SYMBOL'},
             {'args': [{'args': [{'content': [{'string': 'n', 'type': 'STRING'}],
                                  'type': 'BLOCK'},
                                 {'content': [{'string': 'k', 'type': 'STRING'}],
                                  'type': 'BLOCK'}],
                        'name': 'binom',
                        'type': 'FUNCTION'}],
              'name': 'sum',
              'subscript': {'content': [{'string': 'k', 'type': 'STRING'},
                                        {'string': '=', 'type': 'SYMBOL'},
                                        {'string': '0', 'type': 'STRING'}],
                            'type': 'BLOCK'},
              'superscript': {'content': [{'string': 'n', 'type': 'STRING'}],
                              'type': 'BLOCK'},
              'type': 'FUNCTION'},
             {'string': '=', 'type': 'SYMBOL'},
             {'content': [{'string': '2', 'type': 'STRING'}],
              'superscript': {'content': [{'string': 'n', 'type': 'STRING'}],
                              'type': 'BLOCK'},
              'type': 'BLOCK'},
             {'string': ',', 'type': 'SYMBOL'},
             {'args': [{'string': 'k', 'type': 'STRING'}],
              'name': 'prod',
              'subscript': {'content': [{'string': 'k', 'type': 'STRING'},
                                        {'string': '=', 'type': 'SYMBOL'},
                                        {'string': '1', 'type': 'STRING'}],
                            'type': 'BLOCK'},
              'superscript': {'content': [{'string': 'n', 'type': 'STRING'}],
                              'type': 'BLOCK'},
              'type': 'FUNCTION'},
             {'string': '=', 'type': 'SYMBOL'},
             {'string': 'n', 'type': 'STRING'},
             {'string': '!', 'type': 'SYMBOL'},
             {'string': ',', 'type': 'SYMBOL'},
             {'args': [{'content': [{'content': [{'string': 'x',
                                                  'type': 'STRING'}],
                                     'superscript': {'content': [{'string': '2',
                                                                  'type': 'STRING'}],
                                                     'type': 'BLOCK'},
                                     'type': 'BLOCK'}],
                        'type': 'BLOCK'}],
              'name': 'int',
              'subscript': {'content': [{'string': '0', 'type': 'STRING'}],
                            'type': 'BLOCK'},
              'superscript': {'content': [{'string': '1', 'type': 'STRING'}],
                              'type': 'BLOCK'},
              'type': 'FUNCTION'},
             {'string': 'dx', 'type': 'STRING'},
             {'string': '=', 'type': 'SYMBOL'},
             {'args': [{'content': [{'string': '1', 'type': 'STRING'}],
                        'type': 'BLOCK'},
                       {'content': [{'string': '3', 'type': 'STRING'}],
                        'type': 'BLOCK'}],
              'name': 'tfrac',
              'type': 'FUNCTION'},
             {'string': ',', 'type': 'SYMBOL'},
             {'args': [{'args': [{'content': [{'string': '1', 'type': 'STRING'}],
                                  'type': 'BLOCK'},
                                 {'content': [{'string': 'n', 'type': 'STRING'}],
                                  'type': 'BLOCK'}],
                        'name': 'tfrac',
                        'type': 'FUNCTION'}],
              'name': 'lim',
              'subscript': {'content': [{'string': 'n', 'type': 'STRING'},
                                        {'string': '→', 'type': 'SYMBOL'},
                                        {'string': '∞', 'type': 'SYMBOL'}],
                            'type': 'BLOCK'},
              'type': 'FUNCTION'},
            ],
            'type': 'BLOCK'}

        frm = Formula(None, d, geom_cls=Fake)

    # ---------------------------------------------------------------------------
    # Fraction
    # ---------------------------------------------------------------------------

    if False:
        frm = Formula(geom_cls=Fake)
        frm.append(frm.fraction(1, "x"))
        print(repr(frm))

    # ---------------------------------------------------------------------------
    # Sqrt
    # ---------------------------------------------------------------------------

    if False:
        frm = Formula(geom_cls=Fake)
        frm.append(frm.sqrt("x"))
        print(repr(frm))

    # ---------------------------------------------------------------------------
    # Binom
    # ---------------------------------------------------------------------------

    if False:
        frm = Formula(geom_cls=Fake)
        frm.append(frm.binom("n", "k"))
        print(repr(frm))

    # ---------------------------------------------------------------------------
    # PlaceHolder
    # ---------------------------------------------------------------------------

    if False:
        term = Formula(geom_cls=Fake)
        term.append("x")

        frm = Formula(geom_cls=Fake)
        frm.append("a+")
        frm.append(Formula.placeholder(name="test", width=.5, height=0.))

        frm.append(term)
        frm.append("=")
        ph = frm.append(term.placeholder(name="frm"))
        frm.append("+19")

        ph.anim.sx = 2

    # ---------------------------------------------------------------------------
    # Algo
    # ---------------------------------------------------------------------------

    #term = Formula("{{{x}}}", geom_cls=Fake)

    # ---------------------------------------------------------------------------
    # Draw
    # ---------------------------------------------------------------------------

    if False:
        print("----- MAIN repr")
        print(repr(frm))
        print()

    if ok_plot:

        frm.reset_bbox()
        print(repr(frm))
        frm._plot(plt, color='red')

        plt.axis("equal")
        plt.show()
    
