__all__ = ["Transfo2d", "BBox", "TermItem", "Term"]

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

    def reset(self):
        self._matrix[...] = np.eye(3, dtype=float)
        self.touch(self)

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
    
    # -------------------------------------
    # Interpolation
    # -------------------------------------

    def interpolate(self, other, factor=.5, ymax=0., turns=0, smooth='LINEAR'):
        """Interpolate with another transformation.
        """

        from .constants import TAU, PI
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
        dy = maprange(abs(factor - 0.5), 0.5, 0.0, 0., ymax, mode='CIRCULAR')

        # Return the intermediate transformation
        return Transfo2d.from_components(tx=tr[0], ty=tr[1] + dy, sx=sc[0], sy=sc[1], angle=rot)

    
# ====================================================================================================
# A Bounding Box
# ====================================================================================================

class BBox:
    def __init__(self, *bounds):
        if len(bounds) == 1:
            self.set_bbox(bounds[0])
        else:
            self.set_bbox(bounds)

    def __str__(self):
        return f"<BBox ({self.x0:.2f}, {self.y0:.2f}, {self.x1:.2f}, {self.y1:.2f}])"
    
    @classmethod
    def from_points(cls, points):
        vmin, vmax = np.min(points, axis=0), np.max(points, axis=0)
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
# Template for box instance
# Used for dev only
# User must implement this interface
# ====================================================================================================

class TermContent:

    def __init__(self, *bounds):
        self._bbox = BBox(*bounds)
        if self._bbox.width < .001 and self._bbox.height < .001:
            self._bbox = BBox(-1, -1, 1, 1)

    def attach_to_formula(self, term):
        self.formula_term = term

    @property
    def bbox(self):
        return BBox(self._bbox)
    
    def adjust_size(self, width=None, height=None):
        if width is None:
            width = self.bbox.width
        if height is None:
            height = self.bbox.height
        self._bbox = BBox(0, 0, width, height)

# ====================================================================================================
# Root for term
# ====================================================================================================

class Term:

    ATTRIBUTES = {
        'font'          : None,

        'x_sepa'        : 0.1,  # Space between successive terms
        'y_sepa'        : 0.1,  # Space between arranged one over the oher one
        'script_scale'  : 0.5,  # Scripts scale
        'dy_super'      : 0.1,  # Location of superscript: y1 - y_super
        'y_super_min'   : 0.35, # Minimum y location for superscript
        'dy_sub'        : 0.15, # Location of subscript y0 + y_sub
        'y_sub_max'     : 0.15, # Max y location of subscript
        'thickness'     : 0.07, # Bar thickness : arrow, bar, borders...
        'oversize'      : 0.04, # Delta oversize for decorators
        'elon_mode'     : 'CENTER', # Elongation type 
        'elon_margin'   : 0.03, # Margin for decorator elongation
        'elon_smooth'   : 'SMOOTH', # For decorator elongation
    }

    def __init__(self, **attrs):
        # Owner
        self.owner = None
        self.child_key = None
        self.owner_factor = 1.

        # Transformation
        self._transfo = Transfo2d()

        # Attributes
        self._attrs = {}

        for k, v in attrs.items():
            setattr(self, k, v)


    def __str__(self):
        return f"<Term[{self.child_key}]: {self.bbox}, transfo: {self.transfo}, {self.transformed_bbox}>"
    
    def __repr__(self):

        lines = [str(self)]
        for depth, term in self.depths():
            lines.append(f"{' '*depth}{str(term)}")

        return "\n".join(lines)

    # ====================================================================================================
    # Attach to an owner
    # ====================================================================================================

    def attach_to(self, owner, child_key, **kwargs):

        self.owner = owner
        self.child_key = child_key

        for k, v in kwargs.items():
            setattr(self, k, v)

        return self

    # ====================================================================================================
    # Transformation
    # ====================================================================================================

    def reset_transfo(self):
        self._transfo.reset()

    @property
    def transfo(self):
        return self._transfo
    
    def get_transfo(self, owner_transfo=None):
        if owner_transfo is None:
            return self._transfo
        else:
            return owner_transfo @ self._transfo

    # ====================================================================================================
    # Bounding boxes
    # ====================================================================================================

    @property
    def bbox(self):
        assert(False)
    
    @property
    def transformed_bbox(self):
        """Returns the transformed bounding box.
        """
        return self.bbox.transform(self.transfo)
        
    def adjust_size(self, width=None, height=None):
        assert(False)

    # ====================================================================================================
    # Attributes
    # ====================================================================================================

    def __getattr__(self, name):

        if name in Term.ATTRIBUTES:
            if name in self._attrs:
                return self._attrs[name]
            elif self.owner is None:
                return Term.ATTRIBUTES[name]
            else:
                return getattr(self.owner, name)
        else:
            raise AttributeError(
                f"'{name}' is not a valid attribute. Valid are\n"
                f"{list(Term.ATTRIBUTES.keys())}"
                )
        
    def __setattr__(self, name, value):

        if name in Term.ATTRIBUTES:
            self._attrs[name] = value

        else:
            super().__setattr__(name, value)

    # ====================================================================================================
    # Iterators
    # ====================================================================================================

    def terms(self):
        return None
    
    def depths(self):

        def recur(term, depth):
            for child in term.terms:
                yield depth, t

                for i in recur(child, depth + 1):
                    yield i

        for i in recur(self, 0):
            yield i

    # ====================================================================================================
    # Alignments
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # X alignment
    # ----------------------------------------------------------------------------------------------------

    def x_align(self, x, *others, align='left', margin=0.0):

        bbox = self.transformed_bbox

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

    # ----------------------------------------------------------------------------------------------------
    # Y alignment
    # ----------------------------------------------------------------------------------------------------

    def y_align(self, y, *others, align='bottom', margin=0.0):

        bbox = self.transformed_bbox

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
    # Debug
    # ====================================================================================================

    def _plot(self, plt, owner_transfo=None, color='k', alpha=1.0, **kwargs):

        points = self.get_transfo(owner_transfo) @ self.bbox.corners

        x, y = list(points[:, 0]), list(points[:, 1])
        x.append(x[0])
        y.append(y[0])

        # Plot the bounding box
        plt.plot(x, y, color=color, alpha=alpha, **kwargs)

# ====================================================================================================
# An item containing actual geometry
# ====================================================================================================

class ContentItem(Term):

    def __init__(self, content, **attrs):

        assert (
            hasattr(content, 'bbox') and 
            hasattr(content, 'attach_to_formula') and 
            hasattr(content, 'adjust_size')
            )
        
        super().__init__(**attrs)

        # content
        self._content = content

    def __str__(self):
        return f"<ContainerItem: {type(self._content).__name__}, transfo: {self.transfo}, {self.transformed_bbox}>"

    # ====================================================================================================
    # Bounding boxes
    # ====================================================================================================

    @property
    def bbox(self):
        """Returns the bounding box before transformation.

        Simply call get_bbox
        """
        return BBox(self._content.bbox)
        
    def adjust_size(self, width=None, height=None):
        self._content.adjust_size(width=width, height=height)

# ====================================================================================================
# A list of terms
# ====================================================================================================

class Terms(Term):

    def __init__(self, *terms, **attrs):
        """Term is a list concatenated terms.
        """
        super().__init__(**attrs)

        self._terms = []
        for i, term in enumerate(terms):
            self._terms.append(term.attach_to(self, 'term', child_index=i))

    def __str__(self):
        return f"<Terms: {len(self)} terms, transfo: {self.transfo}>"

    def __len__(self):
        return len(self._terms)
    
    def __getitem__(self, index):
        return self._terms[index]

    # ====================================================================================================
    # Iterators
    # ====================================================================================================

    def terms(self):
        for term in self._terms:
            yield term

    # ====================================================================================================
    # Compute the bbox
    # ====================================================================================================

    @property
    def bbox(self):

        bbox = BBox()
        x = 0.
        for term in self:

            # Term bbox
            # Calling get_box() will update the term first
            term.reset_transfo()
            t_bbox = term.bbox

            # Locate after current cursor
            space = 0. if x < ZERO else self.x_sepa
            term.x_align(x + space, align='left')

            x += (space + t_bbox.width)*term.owner_factor

            # Update bbox
            bbox += term.transformed_bbox

        # True bbox depends upon animation
        x0, y0, _, y1 = bbox.as_tuple()
        bbox = BBox(x0, y0, x, y1)

        return bbox

    # ====================================================================================================
    # Debug
    # ====================================================================================================

    def _plot(self, plt, owner_transfo=None, color='k', alpha=1.0, **kwargs):

        bbox = self.bbox

        # Children
        trf = self.get_transfo(owner_transfo)

        child_col = np.random.uniform(0, 1, 3)
        for term in self:
            term._plot(plt, owner_transfo=trf, color=child_col)

        # Surrounding box
        bbox = bbox.bordered(.1)

        points = trf @ bbox.corners

        x, y = list(points[:, 0]), list(points[:, 1])
        x.append(x[0])
        y.append(y[0])

        # Plot the bounding box
        plt.plot(x, y, color=color, alpha=alpha, **kwargs)   

# ====================================================================================================
# A Container (brackets, integral, fraction...
# ====================================================================================================

class ContainerTerm(Terms):
    def __init__(self, *terms, **attrs):
        super().__init__(*terms, **attrs)


    




# ====================================================================================================
# A decorated item
# ====================================================================================================

class DecoratedTerm:

    DECORATORS = {
        'subscript'     : {'bbox': 'core', }, 
        'superscript'   : {'bbox': 'core', },
        'overscript'    : {'bbox': 'core', },
        'underscript'   : {'bbox': 'core', },
        'fix_over'      : {'bbox': 'full', },
        'over'          : {'bbox': 'full', },
        'fix_under'     : {'bbox': 'full', },
        'under'         : {'bbox': 'full', },
        'fix_right'     : {'bbox': 'full', },
        'right'         : {'bbox': 'full', },
        'fix_left'      : {'bbox': 'full', },
        'left'          : {'bbox': 'full', },
        'around'        : {'bbox': 'full', }
    }

    # ----------------------------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------------------------

    def __init__(self, body, **kwargs):

        # Automatically computed when None
        self._transfo = None      
        
        # Content provides the actual bbox (can be a Term)
        if isinstance(body, Term):
            body = [body]

        if isinstance(body, list):
            self._body = []
            for i, term in enumerate(body):
                assert isinstance(term, Term)
                self._body.append(term.attach_to(self, 'body', child_index=i))
        else:
            assert body is not None
            assert hasattr(body, 'bbox')
            self._body = body

        # Children and atributes dicts
        self._children = {}
        self._attrs = {}

        # Relationship with the owner
        self.owner = None
        self.child_key = None
        self.owner_factor = 1.

        # We are ready to set the attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ====================================================================================================
    # Body
    # ====================================================================================================

    @property
    def body_is_terms(self):
        return isinstance(self._body, list)

    @property
    def body_is_content(self):
        return not self.body_is_terms

    @property
    def body_terms(self):
        if isinstance(self._body, list):
            return self._body
        else:
            return []
        
    @property
    def body_content(self):
        if isinstance(self._body, list):
            return None
        else:
            return self._body
        
    # ----------------------------------------------------------------------------------------------------
    # Iterators on all child terms
    # ----------------------------------------------------------------------------------------------------

    def terms(self, include_body=False):

        if include_body:
            for child in self.body_terms:
                yield child

        for child in self._children.values():
            yield child

    def depths(self, include_body=False):

        def _iter(term, depth):
            for child in term.terms(include_body):
                yield depth, child
                for dc in _iter(child, depth + 1):
                    yield dc

        for depth_child in _iter(self):
            return depth_child
        
    # ----------------------------------------------------------------------------------------------------
    # Str and Repr
    # ----------------------------------------------------------------------------------------------------

    def __str__(self):
        return f"<Term {self._body} {list(self._children.keys())}>"
    
    def __repr__(self):
        lines = [str(self)]
        for depth, term in self.depths(True):
            lines.append(f"{' '*depth}{str(term)}")
                         
        return "\n".join(lines)

    # ====================================================================================================
    # The bbox from the actual content
    # ====================================================================================================

    @property
    def body_bbox(self):
        """Body bbox (without decorators)
        """
        if self.body_is_content:
            return self._body.bbox()
        
        x = 0.
        bbox = BBox()
        sepa = BBox(self.x_sepa, 0, 0, 0)
        for term in self.body_terms:
            bx = term.bbox + sepa
            bbox += bx
            x += bx.width * term.owner_factor

        x0, y0, _, y1 = bbox.as_tuple()
        return BBox(x0, y0, x, y1)

    @property
    def bbox(self):
        """Full bbox (including decorators)
        """
        bbox = self.body_bbox
        for term in self.terms(False):
            bbox += term.bbox
        return bbox


    @property
    def transfo(self):

        # Already computed
        if self._transfo is not None:
            return self._transfo
        
        # Initialize the transfo
        self._transfo = Transfo2d()

        # No owner = nothing to do
        if self.owner is None or self.child_key == 'body':
            return self._transfo
        
        # The transformation depends upon the relationship with the owner
        if Term.RELATIONS[self.child_key]['bbox'] == 'full':
            ref_bbox = self.owner.bbox
        else:
            ref_bbox = self.owner.core_bbox

        bbox = self.bbox

        if self.child_key == 'after':
            dx = ref_bbox.x1 + self.x_sepa
            self._transfo.translate(dx, 0.)

        elif self.child_key in ['subscript', 'superscript']:
            pass




        """
        'subscript'     : {'bbox': 'core', }, 
        'superscript'   : {'bbox': 'core', },
        'overscript'    : {'bbox': 'core', },
        'underscript'   : {'bbox': 'core', },
        'fix_over'      : {'bbox': 'full', },
        'over'          : {'bbox': 'full', },
        'fix_under'     : {'bbox': 'full', },
        'under'         : {'bbox': 'full', },
        'fix_right'     : {'bbox': 'full', },
        'right'         : {'bbox': 'full', },
        'fix_left'      : {'bbox': 'full', },
        'left'          : {'bbox': 'full', },
        'around'        : {'bbox': 'full', }
        """


        
        
        
        
        

        
        
    
    @property
    def core_bbox(self):
        if self._content is None:
            return BBox(self._content.bbox)
        else:
            return BBox() if self.body is None else self.body.bbox

    @property
    def bbox(self):
        if self._content is None:
            return BBox()
        else:
            return BBox(self._content.bbox)
        
    
        
    @property
    def transformed_bbox(self):
        """Returns the transformed bounding box.
        """
        return self.bbox.transform(self.transfo)

    # ====================================================================================================
    # Owner / child
    # ====================================================================================================

    def _add_layer(self):
        # Keep the attributes at this level
        # Put content and childre in another child
        child = Term(self._content, **self._children)

        # Remove the children
        self._children.clear()

        # The new child becomes the body
        self.body = child

    def attach_to(self, owner, child_role, **kwargs):

        self.owner = owner
        self.child_role = child_role

        for k, v in kwargs.items():
            setattr(self, k, v)

        return self
    
    def link_child(self, key, child, **kwargs):

        # Key already exists, we create an additional layer
        if key in self._children:
            self._add_layer()

        child.attach_to(self, key, **kwargs)
        self._children[key] = child

        return child

    # ====================================================================================================
    # Attributes
    # ====================================================================================================

    def __getattr__(self, name):

        if name in Term.RELATIONS:
            return self._children.get(name)
        
        elif name in Term.ATTRIBUTES:
            if name in self._attrs:
                return self._attrs[name]
            elif self.owner is None:
                return Term.ATTRIBUTES[name]
            else:
                return getattr(self.owner, name)
        else:
            raise AttributeError(
                f"'{name}' is not a valid attribute. Valid are\n"
                f"{list(Term.RELATIONS.keys())}"
                f"{list(Term.ATTRIBUTES.keys())}"
                )
        
    def __setattr__(self, name, value):

        if name in Term.RELATIONS:
            self.link_child(name, value)
        
        elif name in Term.ATTRIBUTES:
            self._attrs[name] = value

        else:
            super().__setattr__(name, value)

    

        

t = Term(TermContent(), subscript=Term(TermContent()))
t.superscript = Term(TermContent())

print("BODY", t.body)

print(repr(t))
t.superscript = Term(TermContent())

print(repr(t))
























# ====================================================================================================
# A decorated term
# ====================================================================================================

class TermOLD(TermItem):

    # Deco names are given in their priority order
    DECO_NAMES = [
        'subscript', 'superscript', 'overscript', 'underscript',
        'fix_over', 'over','fix_under', 'under', 
        'fix_right', 'right', 'fix_left', 'left', 
        'around',
    ]
    DECO_SCRIPTS = DECO_NAMES[:4] # Not dynamic

    def __init__(self, body, **kwargs):
        """A decorated term.
        """
        super().__init__(None)

        self._body = body.attach_to(self, 'body')

        self._decos = {}
        for key, value in kwargs.items():
            if key in Term.DECO_NAMES:
                self.set_decorator(key, value)
            else:
                setattr(self, key, value)

    def __str__(self):
        return f"<Decorated: decorators: {list(self._decos.keys())}, transfo: {self.transfo}>"
    
    def __repr__(self):

        s = str(self)[1:-1]

        lines = repr(self._body).split("\n")
        s += f"\n   {i}: " + lines[0]
        for line in lines[1:]:
            s += f"\n      {line}"

        for key, term in self._decos.items():
            lines = repr(term).split("\n")
            s += f"\n   {key:}: " + lines[0]
            for line in lines[1:]:
                s += f"\n       {line}"

        return s
    
    def __iter__(self):
        
        yield self._body

        for deco in self._decos.values():
            yield deco

    # ====================================================================================================
    # Decorators
    # ====================================================================================================

    def set_decorator(self, key, decorator):

        if key not in Term.DECO_NAMES:
            raise ValueError(f"Invalid decorator key: '{key}', valid keys are {Term.DECO_NAMES}")
        
        self._decos[key] = decorator.attach_to(self, 'decorator', child_key= key)
        return decorator

    def get_decorator(self, key):
        return self._decos.get(key)
    
    def decorators(self, keys=None):
        if keys is None:
            keys = Term.DECO_NAMES

        for key in keys:
            if key in self._decos:
                yield (key, self._decos[key])   
    
    # ----- Expose keys as shortcuts
    
    def __getattr__(self, name):
        if name in Term.DECO_NAMES and '_decos' in self.__dict__:
            return self.__dict__['_decos'].get(name)
        else:
            return super().__getattr__(name)
        
    def __setattr__(self, name, value):
        if name in Term.DECO_NAMES:
            self.set_decorator(name, value)
        else:
            super().__setattr__(name, value)

    # ====================================================================================================
    # Compute the bbox
    # ====================================================================================================

    def get_bbox(self):
        """Compute the bounding box.

        The bbox is computed by:
        - chaining the terms in the content list one after the other
        - adding decorators
        """

        # ---------------------------------------------------------------------------
        # Body bbox
        # ---------------------------------------------------------------------------

        bbox = self._body.bbox

        # ---------------------------------------------------------------------------
        # Loop on the decorators
        # ---------------------------------------------------------------------------

        # bbox will change, let's store script locations
        x_script = bbox.x1

        # Keep the list of the treated decorators
        done = []

        # To take the keys with respect to their priority order
        # decorators is an iterator which gaurantees the proper order
        for key, term in self.decorators():

            # Term bbox
            # Calling get_box() will update the term first
            term.reset_transfo()
            t_bbox = term.get_bbox()

            # --------------------------------------------------
            # Scripts
            # --------------------------------------------------

            if key in Term.DECO_SCRIPTS:

                term.transfo.apply_scale(self.script_scale)

                if key in ['superscript', 'subscript']:
                    space = self.script_scale*self.x_sepa
                    term.x_align(x_script + space, align='left')

                    if key == 'superscript':
                        y = max(bbox.y1 - self.dy_super, self.y_super_min)
                        term.y_align(y, align='bottom')
                    else:
                        y = min(bbox.y0 + self.dy_sub, self.y_sub_max)
                        term.y_align(y, align='top')

                    bbox = bbox.interpolate(bbox + term.transformed_bbox, term.x_space_factor)

                elif key in ['overscript', 'underscript']:
                    term.x_align(bbox.center[0], align='middle')
                    if key == 'overscript':
                        term.y_align(bbox.y1 + self.y_sepa, align='bottom')
                    else:
                        term.y_align(bbox.y0 - self.y_sepa, align='top')

                    width = term.transformed_bbox.width*term.x_space_factor
                    w = (width - bbox.width)/2
                    if w > 0:
                        self._body.transfotranslate(w, 0)

                        for k, t in self.decorators(done):
                            t.transfo.translate(w, 0)
                        term.transfo.translate(w, 0)

                    # Resulting bbox
                    _, ty0, _, ty1 = term.transformed_bbox.as_tuple()
                    bbox = bbox + (0, ty0, width, ty1)

            # --------------------------------------------------
            # Dynamic
            # --------------------------------------------------

            else:
                if not key.startswith('fix_'):
                    term.adjust_size(None, bbox.height + 2*self.y_sepa)
                t_bbox = term.transformed_bbox

                if key in ['over', 'under', 'fix_over', 'fix_under']:

                    term.x_align(bbox.center[0], align='center')

                    if key in ['over', 'fix_over']:
                        term.y_align(bbox.y1 + self.y_sepa, align='bottom')
                    else:
                        term.y_align(bbox.y0 - self.y_sepa, align='top')

                    bbox = bbox.interpolate(bbox + term.transformed_bbox, term.x_space_factor)

                elif key in ['left', 'fix_left']:

                    term.x_align(0., align='left')
                    term.y_align(bbox.center[1], align='middle')

                    w = (t_bbox.width + self.x_sepa)*term.x_space_factor

                    # Shift left all what must be shifted left
                    self._body.transfo.translate(w, 0)

                    for (k, t) in self.decorators(done):
                        t.transfo.translate(w, 0)

                    # Resulting bbox
                    _, ty0, _, ty1 = term.transformed_bbox.as_tuple()
                    bbox = bbox + (0, ty0, bbox.x1 + w, ty1)

                elif key in ['right', 'fix_right']:
                    x = bbox.x1 + self.x_sepa
                    term.x_align(x, align='left')
                    term.y_align(bbox.center[1], align='middle')

                    bbox = bbox.interpolate(bbox + term.transformed_bbox, term.x_space_factor)

                elif key == 'around':
                    pass

                else:
                    # Should never occur
                    assert(False)

            # Done
            done.append(key)
            
        # ----- We've got our bounding box

        return bbox
    
    # ====================================================================================================
    # Debug
    # ====================================================================================================

    def _plot(self, plt, owner_transfo=None, color='k', alpha=1.0, **kwargs):

        bbox = self.bbox

        # Children
        trf = self.get_transfo(owner_transfo)

        for term in self:
            term._plot(plt, owner_transfo=trf, color=np.random.uniform(0, 1, 3))

        # Surrounding box
        bbox = bbox.bordered(.1)

        points = trf @ bbox.corners

        x, y = list(points[:, 0]), list(points[:, 1])
        x.append(x[0])
        y.append(y[0])

        # Plot the bounding box
        plt.plot(x, y, color=color, alpha=alpha, **kwargs)


if __name__ == "__main__OLD":

    from pprint import pprint
    import matplotlib.pyplot as plt

    ok_plot = True

    # ----- Implement a fixed size term

    terms = Terms(
        TermItem(TermContent()),
        TermItem(TermContent()),
        TermItem(TermContent()),
    )
    print(terms)

    x = TermItem(TermContent())
    p = TermItem(TermContent())

    frm = Term(terms, left=p)


    if ok_plot:

        frm._plot(plt, color='red')

        plt.axis("equal")
        plt.show()
    
