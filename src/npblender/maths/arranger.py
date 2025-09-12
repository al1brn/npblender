__all__ = ["Transfo2d", "Zone"]

import numpy as np
import math

#from .constants import ZERO
ZERO = 1e-6

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
        return (f"<Transfo2d "
                f"tr: ({tx:.2f}, {ty:.2f}), "
                f"sc: ({sx:.2f}, {sy:.2f}), "
                f"ag: {angle_deg:.1f}°"
                f">")
    
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

    @property
    def transformation3d(self):
        from .constants import bfloat
        from .transformation import Transformation

        M4 = np.eye(4, dtype=bfloat)
        M4[:2, :2] = self._matrix[:2, :2]
        M4[:2, 3]  = self._matrix[:2, 2]

        return Transformation(M4, copy=False)

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
# A Bounding Box
# ====================================================================================================

class BBox:
    def __init__(self, *bounds):
        if len(bounds) == 1:
            self.bbox = bounds[0]
        else:
            self.bbox = bounds

    def __str__(self):
        return f"<BBox ({self.x0:.2f}, {self.y0:.2f}, {self.x1:.2f}, {self.y1:.2f}])"
    
    @classmethod
    def from_points(cls, points):
        vmin, vmax = np.min(points, axis=0), np.max(points, axis=0)
        return cls(vmin[:2], vmax[:2])

    @property
    def bbox(self):
        return self._bbox
    
    @bbox.setter
    def bbox(self, bounds):
        error = False

        print("???", bounds, type(bounds).__name__, isinstance(bounds, BBox))

        if isinstance(bounds, BBox):
            self._bbox = bounds.bbox
            return

        if len(bounds) == 0:
            self._bbox = (0., 0., 0., 0.)

        elif len(bounds) in (1, 2):
            bounds = np.asarray(bounds, dtype=float)
            if bounds.size != 4:
                error = True
            else:
                self._bbox = tuple(bounds.ravel())
            
        elif len(bounds) == 4:
            self._bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

        else:
            error = True

        if error:
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
    # Join
    # ====================================================================================================

    def join(self, *others):
        x0, y0, x1, y1 = self.bbox
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
        self.bbox = self.join(other)
        return self
    
    # ====================================================================================================
    # Utilities
    # ====================================================================================================

    def scaled(self, scale):
        cx, cy = self.center
        x0, y0, x1, y1 = self.x0 - cx, self.y0 - cy, self.x1 - cx, self.y1 - cy 
        return BBox(cx + scale*x0, cy + scale*y0, cx + scale*x1, cy + scale*y1)
    
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
# Zoen Parameters
# ====================================================================================================

class ZoneParams:
    def __init__(self, zone, link='child', **parameters):

        self.zone = zone
        self.link = link.lower()

        self.x_space_factor = 1.

        for k, v in parameters.items():
            setattr(self, k, v)

# ====================================================================================================
# Root Zone
# ====================================================================================================

class Zone:
    def __init__(self):
        """A Zone is made of a bbox and a transformation
        """
        self._transfo = Transfo2d()
        self._bbox = BBox(-1, -1, 1, 1)

    @property
    def ref_bbox(self):
        """Returns the zone bbox before transformation
        """
        return self._bbox

    @property
    def bbox(self):
        """Returns the zone bbox after transformation
        """
        if self._bbox is None:
            self.update()
        return self._bbox.transform(self.transfo)

    @property
    def transfo(self):
        return self._transfo
    
    def update(self):
        pass

    def update_for(self, bbox):
        pass

    # ====================================================================================================
    # Operations
    # ====================================================================================================

    def x_align(self, x, *others, align='left', margin=0.0):

        bbox = self.bbox

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
            x_ref = bbox.x0
        elif align in ['center', 'middle']:
            x_ref = bbox.x0 + bbox.width / 2
        elif align == 'right':
            x_ref = bbox.x1
        else:
            raise ValueError(f"Unknown x-align: {align}")

        delta = x - x_ref
        self.transfo.translate(delta + margin, 0.0)
        return self

    def y_align(self, y, *others, align='bottom', margin=0.0):

        bbox = self.bbox

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
            y_ref = bbox.y0
        elif align in ['center', 'middle']:
            y_ref = bbox.y0 + bbox.height / 2
        elif align in ['top']:
            y_ref = bbox.y1
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

    def _plot(self, plt, transfo=None, color='k', alpha=1.0, **kwargs):

        trf = self.transfo if transfo is None else transfo @ self.transfo

        print("ZONE", trf)

        points = trf @ self.ref_bbox.corners

        print("CORNERS\n", self.bbox.corners)
        print("POINTS\n", self.bbox.corners)

        x, y = list(points[:, 0]), list(points[:, 1])
        x.append(x[0])
        y.append(y[0])

        # Plot the bounding box
        plt.plot(x, y, color=color, alpha=alpha, **kwargs)

# ====================================================================================================
# A decorator
# ====================================================================================================

class Decorator(Zone):
    """For tests !
    """

    def update_for(self, bbox):
        self._bbox = BBox(0, 0, bbox.width + 2, .5)


# ====================================================================================================
# A List of zones
# ====================================================================================================

class Formula(Zone):
    def __init__(self):
        """A series of zones arranged in a succession:

        Succession code of one item after the previous one are:
        - After
        - Above
        - Below
        - Superscript
        - Subscript

        Each succession is driven with the following parameters
        - Space : space between the zones
        - Separator : zone to insert
        - Prev_Offset to apply to the previous block

        Each item can be decorated by five decorations:
        - Bot
        - Top
        - Left
        - Right
        - Around

        Animation is driven by the following parameters:
        - x_space_factor : control the space occupied by the item in the flow
        - prev_offset_factor : control the offet of the previous item
        - in addition, the item can be freely transformed to create the desirate effect

        Example
        - To animate from 'a = b + c' to 'a - c = b':
          argm = [a, C, eq, b, C] with C = [op, c]]
          - first C space from 0 to 1
          - second C space from 1 to 0
          - op is a special mesh transformable from + to -
          - C transformation start from second C transfo to first C transfo
            with whatever intermediary positions
        """
        # ----- All the zones used for form the whole zone
        # Content zones and decorator

        self.children = []

        self._bbox = None
        self._transfo = Transfo2d()

        # Parameters
        self.x_child_space = 0.1 # Space between two successors
        self.x_border = 0.1 # decorator x space
        self.y_border = 0.1 # decorator y space
        self.script_scale = .9 # Scale to apply to zones put as scripts
        self.y_superscript = .2 # vertical offset of superscript
        self.y_subscript = .2 # vertical offset of subscript

    # ====================================================================================================
    # Transformation
    # ====================================================================================================

    @property
    def transfo(self):
        return self._transfo
    
    @property
    def ref_bbox(self):
        if self._bbox is None:
            self.update()
        return self._bbox

    # ====================================================================================================
    # Update
    # ====================================================================================================

    def update(self):

        # ---------------------------------------------------------------------------
        # Loop on the child zones forming the content
        # ---------------------------------------------------------------------------

        bbox = BBox()
        x = 0.
        for zp in self.children:

            # Only children
            if zp.link != 'child':
                continue

            # Make sure bbox is up to date
            zp.zone.update()

            # Locate after current cursor
            space = 0. if x < ZERO else self.x_child_space
            zp.zone.x_align(x + space, align='left')

            # Animated actual space
            x += (space + zp.zone.bbox.width)*zp.x_space_factor

            # Update bbox
            bbox += zp.zone.bbox

        # True bbox depends upon animation
        x0, y0, _, y1 = bbox.bbox
        bbox = BBox(x0, y0, x, y1)

        # ---------------------------------------------------------------------------
        # Loop on the child decorative zones 
        # ---------------------------------------------------------------------------

        whole_bbox = BBox(bbox)
        for zp in self.children:

            # Children already done
            if zp.link == 'child':
                continue

            # Make sure bbox is up to date
            zp.zone.update_for(bbox)

            if zp.link == 'above':
                zp.zone.x_align(bbox.center[0], align='middle')
                zp.zone.y_align(bbox.y1 + self.y_border, align='bottom')

            elif zp.link == 'below':
                zp.zone.x_align(bbox.center[0], align='middle')
                zp.zone.y_align(bbox.y0 - self.y_border, align='top')

            elif zp.link in ['superscript', 'subscript']:
                zp.zone.transfo.apply_scale(self.script_scale)
                space = self.script_scale*self.x_child_space
                zp.zone.x_align(bbox.x1 + space, align='left')

                if zp.link == 'superscript':
                    zp.zone.y_align(bbox.y1 - self.y_superscript, align='bottom')
                else:
                    zp.zone.y_align(bbox.y0 + self.y_subscript, align='top')

            else:
                raise ValueError(f"Unknown zone link: '{zp.link}'")
            
            # Animation factor
            full_bbox = bbox.interpolate(bbox + zp.zone.bbox, zp.x_space_factor)
            whole_bbox += full_bbox

        self._bbox = whole_bbox

    # ====================================================================================================
    # Debug
    # ====================================================================================================

    def _plot(self, plt, transfo=None, color='red', alpha=1.0, **kwargs):

        trf = self.transfo if transfo is None else transfo @ self.transfo

        for zp in self.children:
            zp.zone._plot(plt, transfo=transfo, color=np.random.uniform(0, 1, 3), **kwargs)

        # Self with a border

        bbox = self._bbox
        self._bbox = bbox.bordered(.1)
        super()._plot(plt, transfo=transfo, color=color, alpha=alpha, **kwargs)
        self._bbox = bbox



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ok_plot = True

    zone = Zone()

    frm = Formula()
    frm.children.extend([ZoneParams(Zone()), ZoneParams(Zone()), ZoneParams(Zone())])

    zone = Decorator()
    frm.children.append(ZoneParams(zone, link='below'))

    frm.children.append(ZoneParams(Zone(), link='subscript'))
    frm.children.append(ZoneParams(Zone(), link='superscript'))

    frm.children[-1].x_space_factor = .1
    frm.children[-2].x_space_factor = .1


    frm.update()

    if ok_plot:

        frm._plot(plt, color='red')

        plt.axis("equal")
        plt.show()
    
