__all__ = ["Transfo2d", "BBox", "Formula", "FormulaGeom"]

import numpy as np
import math

if __name__ == '__main__':
    from constants import TAU, PI, ZERO
else:
    from .constants import TAU, PI, ZERO

BBOX_CACHE = False
DEF_ADJUSTABLE = True

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
        self._touched = False

    def __str__(self):
        sx, sy = self.scale
        tx, ty = self.translation
        angle_deg = math.degrees(self.rotation)
        return (f"<T: "
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
        if __name__ == '__main__':
            from constants import bfloat
            from transformation import Transformation
        else:
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

    def scaled(self, *scale):
        if len(scale) == 0:
            return BBox(self)
        elif len(scale) == 1:
            if hasattr(scale[0], '__len__'):
                sx, sy = scale[0]
            else:
                sx, sy = scale[0], scale[0]
        else:
            sx, sy = scale

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
# The content of a term
# ====================================================================================================

class FormulaGeom:

    def __init__(self, term, content, symbol=True, adjustable=DEF_ADJUSTABLE):

        # Owning formula term
        self.term = term
        self.symbol = symbol

        # Default name
        self.name = type(content).__name__

        # BBox
        self.adjustable = adjustable
        self._bbox = None
        self.adjust_dims = (0.0, 0.0)

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

    @property
    def transfo3d(self):
        
        from .transformation import Transformation

        if self.term is None:
            return Transformation()
        else:
            transfo = self.term.absolute_transfo
            return transfo.transformation3d

    def adjust_size(self, width=0.0, height=0.0):
        self.adjust_dims = (width, height)
        if self.adjustable:
            self._bbox = None

    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = self.get_bbox()
        return self._bbox
        
    # ---------------------------------------------------------------------------
    # Interface to be implemented
    # ---------------------------------------------------------------------------

    def set_content(self, content):
        assert False, "'set_content' not implemented"

    def get_bbox(self):
        assert False, "'get_bbox' not implemented"

# ====================================================================================================
# A Formula
# ====================================================================================================

class Formula:

    ATTRIBUTES = {
        'geom_cls'      : FormulaGeom,
        'font'          : None,
        'math_font'     : None,

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

    def __init__(self, owner=None, body=None, **attrs):

        # ----------------------------------------
        # Initialization with no ownership
        # ----------------------------------------

        self.owner = owner
        self.child_key = None
        self.owner_factor = 1.

        if owner is None and 'geom_cls' not in attrs:
            raise ValueError(f"A Formula must be initialized with a 'geom_cls' argument when there is no owner.")

        # ----------------------------------------
        # Transformation
        # ----------------------------------------

        self._transfo = Transfo2d()
        self._bbox    = None

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
        # No body : we stop here (certainly root)
        # ----------------------------------------

        if body is None:
            self.body_type = 'LIST'
            return

        # ----------------------------------------
        # Body
        # - TERM : accepts decorators
        # - LIST : doesn't accept decorators
        # - GEOM : doesn't accept decorators
        # ----------------------------------------

        # To manage decorators the content must be a Formula
        if has_decorator and not isinstance(body, Formula):
            body = Formula(self, body)

        # body is a Formula
        if isinstance(body, Formula):
            self.body_type = 'TERM'
            self._term = body.attach_to(self, 'body')

        # body is a list of Formulas
        elif isinstance(body, list):
            self.body_type = 'LIST'

            self._list = []
            for i, item in enumerate(body):
                if not isinstance(item, Formula):
                    item = Formula(self, item)
                self._list.append(item.attach_to(self, 'term', child_index=i))

        # body is a dict to parse
        elif isinstance(body, dict):
            if self._is_formula_dict(body):
                self.body_type = 'TERM'
                self._term = self.parse_dict(body)
            else:
                self.body_type = 'GEOM'
                self._geom = self.geom_cls(self, body)

        # body is a FormulaGeom
        elif isinstance(body, FormulaGeom):
            self.body_type = 'GEOM'
            body.term = self
            self._geom = body

        # body is to build a FormulaGeom
        else:
            self.body_type = 'GEOM'
            self._geom = self.geom_cls(self, body)

        # ----------------------------------------
        # Set decorators
        # ----------------------------------------

        # Decorators are set in priority order
        for k in Formula.DECORATORS:
            if k not in attrs:
                continue
            self._set_decorator(k, attrs[k])

        # ----------------------------------------
        # Let's compute the bbox
        # ----------------------------------------

        _ = self.get_bbox()

    # ====================================================================================================
    # Str / repr
    # ====================================================================================================

    @property
    def name(self):
        if self.body_is_geom:
            if hasattr(self._geom, 'name'):
                return self._geom.name
            else:
                return 'content'
            
        elif self.owner is None:
            return "ROOT"
        
        else:
            return self.child_key

    def __str__(self):

        if self.owner is None:
            skey = '(ROOT)'
        else:
            skey = f"({str(self.child_key)})"

        if self.body_type == 'ROOT':
            stype = "[ROOT]"
        elif self.body_type == 'GEOM':
            stype = f"'{self._geom.name}'"
        elif self.body_type == 'LIST':
            stype = f"list[len(self._list)]"
        else:
            stype = f"term"

        s = f"<Term {skey:15s} {stype:10s} "

        return f"{s}, {str(self.transfo)[1:-1]}>"
    
    def __repr__(self):

        lines = [str(self)]
        for depth, term in self.depths():
            lines.append(f"{'   '*(depth+1)}{str(term)}")

        return "\n".join(lines)
    
    # ====================================================================================================
    # Special
    # ====================================================================================================

    def ensure_formula(self, term):
        if isinstance(term, Formula):
            return term
        else:
            return Formula(self, term)
        
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
    
    def _is_formula_dict(self, fdict):
        return 'type' in fdict
    
    @staticmethod
    def symbol_string(s):
        return {'type': 'SYMBOL', 'string': s}

    def parse_dict(self, fdict):

        if fdict['type'] in ['STRING', 'SYMBOL']:

            attrs = {k: v for k, v in fdict.items() if k not in ['type', 'string']}
            content = self.geom_cls(self, fdict['string'], symbol=fdict['type']=='SYMBOL')

            return Formula(self, content, **attrs)

        elif fdict['type'] == 'BLOCK':
            attrs = {k: v for k, v in fdict.items() if k not in ['type', 'content']}
            terms = [self.parse_dict(d) for d in fdict['content']]

            if len(terms) == 1:
                return Formula(self, terms[0], **attrs)
            else:
                return Formula(self, terms, **attrs)
            
        elif fdict['type'] == 'FUNCTION':
            name    = fdict['name']
            option  = fdict.get('options')
            args    = fdict.get('args')
            attrs   = {k: v for k, v in fdict.items() if k not in ['type', 'name', 'options', 'args']}

            if name == 'sum':
                return self.operator(self.symbol_string('Σ'), args[0], adjustable=True, **self.to_underover(attrs))

            elif name == 'prod':
                return self.operator(self.symbol_string('Π'), args[0], adjustable=True, **self.to_underover(attrs))
            
            elif name in INT_SYMBS:
                return self.operator(self.symbol_string(INT_SYMBS[name]), args[0], adjustable=True, sub_offset=self.int_sub_ofs, **attrs)
            
            elif name in ['lim', 'limsup', 'liminf']:
                return self.fix_operator(self.symbol_string(name), args[0], adjustable=False, **self.to_underover(attrs))
            
            elif name in ['frac', 'tfrac', 'dfrac']:
                return self.fraction(args[0], args[1], **attrs)
            
            elif name == 'sqrt':
                return self.sqrt(args[0], option=option, **attrs)
            
            elif name == 'binom':
                return self.binom(args[0], args[1])
            
            else:
                frm = Formula(self, name, **attrs)
                for a in args:
                    frm.append(a)
                return frm
            
        raise ValueError(f"Unknown type '{fdict['type']}' in dict.\n{fdict}")

    def operator(self, symbol, body, **attrs):
        frm = Formula(self, body)
        if len(attrs):
            frm.operation = Formula(frm, symbol, **attrs)
        else:
            frm.operation = symbol
        return frm


    def fix_operator(self, symbol, body, **attrs):
        frm = Formula(self, body)
        if len(attrs):
            frm.fix_operation = Formula(frm, symbol, **attrs)
        else:
            frm.fix_operation = symbol
        return frm
    
    def fraction(self, numerator, denominator, **attrs):
        if len(attrs):
            numerator = Formula(self, numerator, **attrs)
        frm = Formula(self, numerator)
        frm.denominator = Formula(frm, denominator)
        frm.fraction_bar = Formula(frm, "_")
        return frm
    
    def binom(self, n, k, **attrs):
        if len(attrs):
            n = Formula(self, n, **attrs)
        body = Formula(self, n)
        body.denominator = Formula(body, k)
        return Formula(self, body, left=self.symbol_string("("), right=self.symbol_string(")"))
    
    def sqrt(self, body, option=None, **attrs):

        if len(attrs):
            body = Formula(self, body, **attrs)

        frm = Formula(self, body)
        frm.sqrt = Formula(frm, r"\sqrt")

        return frm

    # ====================================================================================================
    # Body
    # ====================================================================================================

    @property
    def body_type(self):
        return self._body_type
    
    @body_type.setter
    def body_type(self, value):
        """Set the body_type.
        """

        if not value in ['TERM', 'LIST', 'GEOM']:
            raise ValueError(f"Unknown bodytype: '{value}'")

        self._body_type = value

        self._term = None
        self._list = []
        self._geom = None


    @property
    def body_is_term(self):
        return self.body_type == 'TERM'
    
    @property
    def body_is_list(self):
        return self.body_type == 'LIST'

    @property
    def body_is_geom(self):
        return self.body_type == 'GEOM'
    
    @property
    def body(self):
        if self.body_type == 'TERM':
            return self._term
        elif self.body_type == 'LIST':
            return self._list
        elif self.body_type == 'GEOM':
            return self._geom

    # ----------------------------------------------------------------------------------------------------
    # Body bbox
    # ----------------------------------------------------------------------------------------------------

    @property
    def body_bbox(self):

        # --------------------------------------------------
        # Nothing to do
        # --------------------------------------------------

        if self.body_type == 'TERM':
            return self._term.bbox
        
        elif self.body_type == 'GEOM':
            try:
                return BBox(self._geom.bbox)
            except Exception as e:
                raise RuntimeError(str(e))

        # --------------------------------------------------
        # List of formulas : one after the other
        # --------------------------------------------------

        bbox = BBox()
        x = 0.
        for term in self._list:

            # Term bbox
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
    # Owner / Children
    # ====================================================================================================

    def _add_layer(self, as_list=False):

        # Keep the attributes at this level
        # Put content and decorators in another child
        if self.body_is_geom:
            new_body = Formula(self, self._geom)
        elif self.body_is_list:
            new_body = Formula(self, self._list)
        else:
            new_body = Formula(self, self._term, **self._decos).attach_to(self, 'body')

        self._decos = {}
        if as_list:
            self.body_type = 'LIST'
            self._list.append(new_body.attach_to(self, 'term', child_index=0))
        else:
            self.body_type = 'TERM'
            self._term = new_body.attach_to(self, 'body')

        self._bbox = None

    def attach_to(self, owner, child_key, **kwargs):

        self.owner = owner
        self.child_key = child_key

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._bbox = None

        return self
    
    def _set_decorator(self, key, child, **kwargs):

        if False:
            print("SET DECO:", self.body_type, child, key)

        # Make sure child is a Formula
        if not isinstance(child, Formula):
            child = Formula(self, child)

        # We can safely add the deco
        child.attach_to(self, key, **kwargs)
        self._decos[key] = child

        self._bbox = None

        return child
    
    def set_decorator(self, key, child, **kwargs):

        # Need to encapsulate the content ?
        if (not self.body_is_term) or (key in self._decos):
            self._add_layer()

        # Now the decorator can be added safely
        child = self._set_decorator(key, child, **kwargs)

        # Update the bbox
        self.get_bbox()

        return child
    
    def get_decorator(self, key):
        return self._decos.get(key)
    
    def append(self, term):
        if self.body_type != 'LIST':
            self._add_layer(as_list=True)

        if not isinstance(term, Formula):
            term = Formula(self, term)

        self._list.append(term.attach_to(self, 'term', child_index=len(self._list)))

        self._bbox = None
        _ = self.get_bbox()

        return term
    
    # ----------------------------------------------------------------------------------------------------
    # Iterators
    # ----------------------------------------------------------------------------------------------------

    def terms(self):

        # Body
        if self.body_type == 'TERM':
            yield self._term

        elif self.body_type == 'LIST':
            for term in self._list:
                yield term

        # Decorators
        for term in self._decos.values():
            yield term
    
    def depths(self):

        def recur(depth, term):
            for child in term.terms():
                yield depth, child

                for ic in recur(depth + 1, child):
                    yield ic

        for ic in recur(0, self):
            yield ic

    # ====================================================================================================
    # Attributes
    # ====================================================================================================

    def __getattr__(self, name):

        if name in Formula.DECORATORS:
            return self._decos.get(name)
        
        elif name in Formula.ATTRIBUTES:
            #if name in self._attrs:
            #    return self._attrs[name]
            if self.owner is None:
                return Formula.ATTRIBUTES[name]
            else:
                return getattr(self.owner, name)
        else:
            raise AttributeError(
                f"'{name}' is not a valid attribute. Valid are\n"
                f"{list(Formula.DECORATORS.keys())}"
                f"{list(Formula.ATTRIBUTES.keys())}"
                )
        
    def __setattr__(self, name, value):

        if name in Formula.DECORATORS:
            self.set_decorator(name, value)
        
        #elif name in Formula.ATTRIBUTES:
        #    self._attrs[name] = value

        else:
            super().__setattr__(name, value)

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
        
    @property
    def absolute_transfo(self):
        if self.owner is None:
            return self.transfo
        else:
            return self.get_transfo(self.owner.absolute_transfo)

    # ----------------------------------------------------------------------------------------------------
    # X alignment
    # ----------------------------------------------------------------------------------------------------

    def x_align(self, x, align='left', margin=0.0):

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
        self.transfo.translate(delta + margin, 0.0)
        return self

    # ----------------------------------------------------------------------------------------------------
    # Y alignment
    # ----------------------------------------------------------------------------------------------------

    def y_align(self, y, align='bottom', margin=0.0):

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
        self.transfo.translate(0.0, delta + margin)
        return self

    # ====================================================================================================
    # Bounding boxes
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Cached
    # ----------------------------------------------------------------------------------------------------

    @property
    def bbox(self):

        if not BBOX_CACHE:
            return self.get_bbox()

        # We have a cache
        if self._bbox is not None:
            return self._bbox

        msg = None
        try:
            return self.get_bbox()
        except Exception as e:
            raise RuntimeError(f"Error when getting bbox.", msg)

    # ----------------------------------------------------------------------------------------------------
    # Compute and cache bbox
    # ----------------------------------------------------------------------------------------------------

    def get_bbox(self):
        """Formula bbox is the body bbox plus decorators
        """

        # Reset the children transformations
        for term in self.terms():
            term.reset_transfo()

        # Starting from the body bbox
        bbox = self.body_bbox

        # Terms and content : no decorator to take into account
        if self.body_type != 'TERM':
            assert len(self._decos) == 0, f"Body_type {self.body_type} doesn't accept decorators"
            return bbox

        # Align the body to left
        if abs(bbox.x0) > ZERO:
            dx = -bbox.x0
            self.body.x_align(0, 'left')
            bbox = bbox.translated(dx, 0.0)

        # x to place the scripts
        x_script = bbox.x1

        # ---------------------------------------------------------------------------
        # Some decorators need to move the body and the decorators already placed
        # ---------------------------------------------------------------------------

        dones = [self.body]
        def _translate_dones(dx=0., dy=0.):
            tr = Transfo2d.from_components(tx=dx, ty=dy)
            for term in dones:
                term.transfo.apply_transfo(tr)

        # A term over or under can be wider than the body
        def _left_adjust(term):
            width = term.transformed_bbox.width*term.owner_factor            
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
                term.adjust_size(0.0, bbox.height + 2*self.y_sepa)

            # Recompute
            _ = term.get_bbox()

            # --------------------------------------------------
            # Scripts
            # --------------------------------------------------

            if key in ['subscript', 'superscript', 'underscript', 'overscript']:

                term.transfo.apply_scale(self.script_scale)

                if key in ['subscript', 'superscript']:
                    space = self.script_scale*self.x_sepa
                    if key == 'subscript':
                        space += self.sub_offset
                    term.x_align(x_script + space, align='left')

                    if key == 'superscript':
                        y = max(bbox.y1 - self.dy_super, self.y_super_min)
                        term.y_align(y, align='bottom')
                    else:
                        y = min(bbox.y0 + self.dy_sub, self.y_sub_max)
                        term.y_align(y, align='top')

                    bbox = bbox.interpolate(bbox + term.transformed_bbox, term.owner_factor)

                elif key in ['underscript', 'overscript']:
                    term.x_align(bbox.center[0], align='middle')
                    if key == 'overscript':
                        term.y_align(bbox.y1 + self.y_sepa, align='bottom')
                    else:
                        term.y_align(bbox.y0 - self.y_sepa, align='top')

                    # Term can be wider
                    width = _left_adjust(term)

                    # Resulting bbox
                    _, ty0, _, ty1 = term.transformed_bbox.as_tuple()
                    bbox = bbox + (0, ty0, width, ty1)

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

                bbox = bbox.interpolate(bbox + term.transformed_bbox, term.owner_factor)

            # --------------------------------------------------
            # Left / right
            # --------------------------------------------------

            elif key in ['left', 'right', 'fix_left', 'fix_right', 'operation', 'fix_operation']:

                t_bbox = term.transformed_bbox

                if key in ['left', 'fix_left', 'operation', 'fix_operation']:

                    term.x_align(0., align='left')
                    term.y_align(bbox.center[1], align='middle')

                    w = (t_bbox.width + self.x_sepa)*term.owner_factor

                    # Left shift the body and done terms
                    _translate_dones(w, 0)

                    # Resulting bbox
                    _, ty0, _, ty1 = term.transformed_bbox.as_tuple()
                    bbox = bbox + (0, ty0, bbox.x1 + w, ty1)

                elif key in ['right', 'fix_right']:
                    x = bbox.x1 + self.x_sepa
                    term.x_align(x, align='left')
                    term.y_align(bbox.center[1], align='middle')

                    bbox = bbox.interpolate(bbox + term.transformed_bbox, term.owner_factor)

            # --------------------------------------------------
            # Fraction
            # --------------------------------------------------

            elif key == 'denominator':

                # fraction bar is treated with denominator
                if key == 'fraction_bar':
                    continue

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
                bbox = bbox.interpolate(new_box, term.owner_factor)

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
                t_bbox = term.bbox
                tx = -t_bbox.x0

                # Move body to right
                _translate_dones(tx, 0.0)

                # Adjust term location
                term.transfo.translate(tx, bbox.y0)

                # New bbox
                bbox = bbox.interpolate(term.transformed_bbox, term.owner_factor)

            # --------------------------------------------------
            # Fall Back
            # --------------------------------------------------

            else:
                # Should never occur
                assert(False)

            # Done
            dones.append(term)
            
        # ----- We've got our bounding box

        self._bbox = bbox

        return self._bbox
        
    @property
    def transformed_bbox(self):
        """Returns the transformed bounding box.
        """
        return self.bbox.transform(self.transfo)
        
    def adjust_size(self, width=0.0, height=0.0):
        
        if self.body_is_geom:
            self._geom.adjust_size(width, height)

        elif self.body_is_term:
            self._term.adjust_size(width, height)

        else:
            for t in self._list:
                t.adjust_size(width, height)

        self._bbox = None

    # ====================================================================================================
    # Debug
    # ====================================================================================================

    def _plot(self, plt, owner_transfo=None, color='black', border=.2, **kwargs):

        if False:
            print("PLOT", self)

        transfo = self.get_transfo(owner_transfo)

        def plot_bbox(bbox, col, content=False):

            points = transfo @ bbox.corners
            x, y = list(points[:, 0]), list(points[:, 1])
            x.append(x[0])
            y.append(y[0])

            plt.plot(x, y, color=col, **kwargs)

            if content:
                if True:
                    xs, ys = (x[0] + x[1])/2, (y[1] + y[2])/2
                    plt.text(xs, ys, self.name, ha="center", va="center", fontsize=12)
                else:                    
                    plt.plot([x[0], x[2]], [y[0], y[2]], color=col, **kwargs)
                    plt.plot([x[1], x[3]], [y[1], y[3]], color=col, **kwargs)

        # ----- Body bbox
        if self.body_is_geom:
            bbox = self.bbox
            plot_bbox(bbox, col=color, content=self.body_is_geom)

        # ----- Decorators
        has_children = False
        for term in self.terms():

            term_col = color
            
            if term.child_key == 'body':
                term_col = np.random.uniform(0, 1, 3)
            elif term.child_key == 'term':
                term_col = np.random.uniform(0, 1, 3)
            
            term._plot(plt, owner_transfo=transfo, color=term_col, border=border/2, **kwargs)
            has_children = True

        # ----- Surrounding box
        if has_children:
            bbox = self.bbox.bordered(border)
            plot_bbox(bbox, col=color)

# ====================================================================================================
# Main
# ====================================================================================================

if __name__ == "__main__":

    class Fake(FormulaGeom):

        def set_content(self, content):

            #print("SET CONTENT", content)

            self.adjustable = True

            if isinstance(content, (str, int, float)):
                self.name = str(content)
                if self.name == '_':
                    bbox = BBox(0, 0, .1, .1)
                else:
                    bbox = BBox(-1, -1, 1, 1)
            elif isinstance(content, dict):
                self.name = content['text']
                bbox = BBox(-1, -1, 1, 1)
            else:
                self.name = content[0]
                bbox = BBox(*content[1])

            self._bbox = bbox
            self.ref_bbox = bbox

        def get_bbox(self):

            if self._bbox is None:
                w, h = self.adjust_dims

                sx = 1. if w <= self.ref_bbox.width else w/self.ref_bbox.width
                sy = 1. if h <= self.ref_bbox.height else h/self.ref_bbox.height

                return self.ref_bbox.scaled(sx, sy)
            else:
                return self._bbox
        

    from pprint import pprint
    import matplotlib.pyplot as plt

    ok_plot = True
    frm = Formula(None, "x", geom_cls=Fake)

    # ---------------------------------------------------------------------------
    # Layer
    # ---------------------------------------------------------------------------

    if False:
        frm = Formula(None, "x", geom_cls=Fake)

        # Type content
        print("----- x alone")
        print(repr(frm))
        # Add a decorator -> add a layer
        frm.fix_under = "U"
        print("----- x _")
        print(repr(frm))

    # ---------------------------------------------------------------------------
    # Decorators
    # ---------------------------------------------------------------------------

    if False:

        frm = Formula(None, ["x", "y"], geom_cls = Fake,
                    superscript = "2", 
                    subscript   = "i",
                    overscript  = ("oscript", (0, 0, 3, .6)),
                    underscript = ("uscript", (0, 0, 3, .6)),

                    fix_over  = ("over", (0, 0, 2, .6)),
                    fix_under = ("under", (0, 0, 2, .6)),
                    over      = ("<< over >>", (0, 0, 1, .6)),
                    under     = ("<< under >>", (0, 0, 1, .6)),

                    fix_left  = ("(", (0, 0, .3, 2)),
                    fix_right = (")", (0, 0, .3, 2)),
                    left      = ("[", (0, 0, .3, 1)),
                    right     = ("]", (0, 0, .3, 1)),
                    )
        
        if True:
            frm = Formula(None, [frm, "z"], geom_cls = Fake,
                    superscript = "2", 
                    subscript   = "i",
                    overscript  = ("oscript", (0, 0, 3, .6)),
                    underscript = ("uscript", (0, 0, 3, .6)),

                    fix_over  = ("over", (0, 0, 2, .6)),
                    fix_under = ("under", (0, 0, 2, .6)),
                    over      = ("<< over >>", (0, 0, 1, .6)),
                    under     = ("<< under >>", (0, 0, 1, .6)),

                    fix_left  = ("(", (0, 0, .3, 2)),
                    fix_right = (")", (0, 0, .3, 2)),
                    left      = ("[", (0, 0, .3, 1)),
                    right     = ("]", (0, 0, .3, 1)),
                        )
            
    # ---------------------------------------------------------------------------
    # Append
    # ---------------------------------------------------------------------------

    if False:
            frm = Formula(geom_cls=Fake)
            frm.append("x")
            frm.superscript="2"
            frm.append("y")
            frm.append("z")
            frm.subscript="sub"
            frm.superscript="sup"
            frm.append("After")
            
    # ---------------------------------------------------------------------------
    # Integral
    # ---------------------------------------------------------------------------

    if False:
        frm = Formula(geom_cls=Fake)
        frm.append(frm.operator( ("/", (0, 0, .2, .6)), "F1", subscript="0", superscript="1") )
        frm.append(frm.fix_operator( ("/", (0, 0, .2, .6)), "F2") )

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

    if True:
        frm = Formula(geom_cls=Fake)
        frm.append(frm.binom("n", "k"))
        print(repr(frm))



    # ---------------------------------------------------------------------------
    # Draw
    # ---------------------------------------------------------------------------

    if False:
        print("----- MAIN repr")
        print(repr(frm))
        print()

    if ok_plot:

        frm._plot(plt, color='red')

        plt.axis("equal")
        plt.show()
    
