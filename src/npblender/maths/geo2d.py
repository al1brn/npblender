import numpy as np

__all__ = [
    "MITER_NONE", "MITER_FLAT", "MITER_ROUND", "MITER_CUT",
    "Line", "Poly", 
    "set_spline2d_thickness"]

from .constants import bfloat, bint, EPS

D_MIN = 1e-4

MITER_NONE = 0
MITER_FLAT = 1
MITER_ROUND = 2
MITER_CUT = 3

# ====================================================================================================
# Line or Lines
# ====================================================================================================

class Line:
    def __init__(self, p0=None, p1=None):
        if p0 is None or p1 is None:
            self.p = np.zeros(2, dtype=bfloat)
            self.d = np.array([1, 0], dtype=bfloat)
        else:
            self.p = np.asarray(p0, dtype=bfloat)
            self.d = np.asarray(p1, dtype=bfloat) - self.p

            n = np.linalg.norm(self.d, axis=-1, keepdims=True)
            if np.any(n < 1e-6):
                raise ValueError("Zero-length direction in Line.__init__")
            self.d /= n

    @classmethod
    def point_dir(cls, p, d):
        line = cls()
        line.p = np.array(p)
        line.d = np.array(d)
        return line

    def clone(self):
        line = Line()
        line.p = np.array(self.p)
        line.d = np.array(self.d)
        return line
    
    def __str__(self):
        if self.is_scalar:
            return f"<Line({self.p}, {self.d})>"
        else:
            return f"<Line: {self.p.shape[0]} lines>"
        
    def __len__(self):
        if self.is_scalar:
            raise Exception(f"Line {self} is scalar: it has no len")
        else:
            return self.p.shape[0]
        
    def __getitem__(self, index):
        line = Line()
        line.p = np.array(self.p[index])
        line.d = np.array(self.d[index])
        return line

    def __setitem__(self, index, value):
        self.p[index] = value.p
        self.d[index] = value.d

    def append(self, line):
        if self.is_scalar:
            self.p = self.p[None]
            self.d = self.d[None]

        if line.is_scalar:
            self.p = np.append(self.p, line.p[None], axis=0)
            self.d = np.append(self.d, line.d[None], axis=0)
        else:
            self.p = np.append(self.p, line.p, axis=0)
            self.d = np.append(self.d, line.d, axis=0)

        return self
    
    @property
    def is_scalar(self):
        return self.p.ndim == 1

    @staticmethod
    def dot(a, b):
        return a[..., 0]*b[..., 0] + a[..., 1]*b[..., 1]
        
    @staticmethod
    def cross(a, b):
        return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

    def evaluate(self, t):
        return self.p + t*self.d
    
    def translate(self, delta):
        self.p += delta

        return self

    def rotate(self, angle):
        cag, sag = np.cos(angle), np.sin(angle)
        new_d = np.empty_like(self.d)
        new_d[..., 0] = cag*self.d[..., 0] - sag*self.d[..., 1]
        new_d[..., 1] = sag*self.d[..., 0] + cag*self.d[..., 1]
        self.d = new_d

        return self
    
    @property
    def angle(self):
        return np.arctan2(self.d[..., 1], self.d[..., 0])
    
    @property
    def perpendicular(self):
        return np.stack((-self.d[..., 1], self.d[..., 0]), axis=-1)
    
    def angle_with(self, other):
        return np.arctan2(self.cross(self.d, other.d), self.dot(self.d, other.d))
    
    def parallel_to(self, other):
        return np.abs(self.cross(self.d, other.d)) < 1e-6
    
    def intersect(self, other):
        num = self.cross(other.p - self.p, other.d)
        den = self.cross(self.d, other.d)

        if self.is_scalar:
            if abs(den) < EPS:
                return np.full(2, np.nan, dtype=bfloat)
            return self.p + (num/den)*self.d
        else:
            mask = (np.abs(den) < EPS)[:, None]
            return np.where(mask, np.nan, self.p + ((num/den)[..., None])*self.d)
        
    def intersections(self, cyclic=False):
        if self.is_scalar or len(self) <= 1:
            return np.zeros((0, 2), dtype=bfloat)
        
        if cyclic:
            lines = self.clone().append(self[0])
        else:
            lines = self

        line0 = lines[:-1]
        line1 = lines[1:]

        return line0.intersect(line1)
    
        
# ====================================================================================================
# Poly line
# ====================================================================================================

class Poly:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=bfloat)
        assert(len(self.points) > 2)
        self._lines = None

    # ====================================================================================================
    # Lines
    # ====================================================================================================

    @property
    def lines(self):
        if self._lines is None:
            self._lines = Line(self.points[:-1], self.points[1:])
        return self._lines

    # ====================================================================================================
    # Perpendicular translation
    # ====================================================================================================

    def perp_translated(self, offset, cyclic=False):

        # Working segments
        lines = self.lines.clone()
        if cyclic:
            closing_line = Line(self.points[-1], self.points[0])
            lines = lines.append(closing_line)

        # Perp vector per segment
        perp = lines.perpendicular

        # Simple case
        if lines.is_scalar or len(lines) == 1:
            return lines.p + perp*offset
        
        # Check offset
        # If n+1 is the number of points, offset can be (), (n,) or (n+1,)
        # (n,) is valid only if not cyclic

        n = len(self.lines)
        offset = np.asarray(offset, dtype=bfloat)
        if offset.shape == ():
            offset = np.ones((len(lines), 1))*offset

        elif offset.shape == (n,):
            if cyclic:
                raise ValueError("Offset length is {n} for a cylic {n+1}-poly")
            offset = offset[:, None]

        elif offset.shape == (n+1,):
            if cyclic:
                offset = offset[:, None]
            else:
                offset = offset[:-1, None]

        else:
            raise ValueError("Offset shape is {offset.shape} which is not valid for a {n+1}-poly")
        
        # Translate segments perpendiculary to the central line
        lines.translate(perp*offset)

        # Intersections

        inters = lines.intersections(cyclic=cyclic)

        if cyclic:
            pts = np.roll(inters, 1, axis=0)

        else:
            pts = np.empty_like(self.points)

            pts[0]  = self.points[0]  + perp[0]*offset[0]
            pts[-1] = self.points[-1] + perp[-1]*offset[-1]
            pts[1:-1] = inters

        # Replace nan points by mid points

        inan = np.argwhere(np.isnan(pts[:, 0]))
        if len(inan):
            inan = inan[0]
            pts[inan] = (pts[inan - 1] + pts[inan + 1])/2

        return Poly(pts)
    
    # ====================================================================================================
    # Miter corners 
    # ====================================================================================================
    
    def miter_corners(self,
            mode = MITER_NONE,
            factor = 1.,
            radius = .1, 
            resolution = 12,
            cuts = (.1, np.nan), 
            cyclic = False):

        n = len(self.points)

        # ---------------------------------------------------------------------------
        # Prepare
        # ---------------------------------------------------------------------------

        i_next = lambda i: (i+1) % n if cyclic else i+1
        i_prev = lambda i: (i-1) % n if cyclic else i-1

        mode = np.broadcast_to(mode, (n,))
        factor = np.broadcast_to(factor, (n,))
        radius = np.broadcast_to(radius, (n,))*factor
        cuts = np.broadcast_to(cuts, (n, 2)).copy()
        cuts[:, 0] *= factor

        points = np.zeros((resolution*(n+2), 2), dtype=bfloat)
        index = 0
        if not cyclic:
             points[0] = self.points[0]
             index += 1

        # ---------------------------------------------------------------------------
        # Loop on the corners
        # ---------------------------------------------------------------------------

        it = range(n) if cyclic else range(1, n-1)
        for i_corner in it:

            P0, P1, P2 = self.points[i_prev(i_corner)], self.points[i_corner], self.points[i_next(i_corner)]

            # ---------------------------------------------------------------------------
            # No miter : so easy :-)
            # ---------------------------------------------------------------------------

            if (mode[i_corner] == MITER_NONE) or (factor[i_corner] <= EPS):
                points[index] = P1
                index += 1
                continue

            # ----- Compute the two segments

            v0 = P1 - P0
            v1 = P2 - P1
            l0 = np.linalg.norm(v0)
            l1 = np.linalg.norm(v1)
            if l0 < D_MIN or l1 < D_MIN:
                points[index] = P1
                index += 1
                continue

            d0 = v0 / l0
            d1 = v1 / l1

            # ---------------------------------------------------------------------------
            # A cut
            # Parameter : distance (along P0-P1 or P1_P2) and direction angle (radians)
            # ---------------------------------------------------------------------------

            if mode[i_corner] == MITER_CUT:
                cut_ofs, cut_ag = cuts[i_corner]

                if abs(cut_ofs) <= D_MIN:
                    points[index] = P1
                    index += 1
                    continue

                if np.isnan(cut_ag):
                    cut_dir = -v0[1], v0[0]
                else:
                    cut_dir = np.cos(cut_ag), np.sin(cut_ag)

                # Cutting line
                # Offset is along first or second segment depending on the sign
                if cut_ofs < 0:
                    cut_point = P1 + cut_ofs*d0
                else:
                    cut_point = P1 + cut_ofs*d1

                cut_line = Line.point_dir(cut_point, cut_dir)

                E0 = cut_line.intersect(self.lines[i_prev(i_corner)])
                E1 = cut_line.intersect(self.lines[i_corner])

                if np.isnan(E0[0]) or np.isnan(E1[0]):
                    points[index] = P1
                    index += 1
                else:
                    points[index] = E0
                    points[index+1] = E1
                    index += 2
                continue

            # ---------------------------------------------------------------------------
            # Compute E0, E1
            # ---------------------------------------------------------------------------

            # ----- Effective radius

            r_eff = radius[i_corner]

            # ----- Angle intérieur θ entre u=-d0 (entrant) et v=d1 (sortant)

            u = -d0
            v =  d1
            cth = float(np.clip(np.dot(u, v), -1.0, 1.0))            # cos θ
            sth = abs(d0[0]*d1[1] - d0[1]*d1[0])                     # = |sin θ| (car |cross(u,v)| = |cross(d0,d1)|)

            # ----- Points E0, E1

            # a = radius * cot(θ/2) = thickness * (1 + cos θ) / sin θ
            if sth < EPS or (1.0 + cth) <= 0.0:
                a = 0.0
            else:
                a = r_eff * (1.0 + cth) / sth

            a = min(a, l0/2, l1/2)

            E0 = P1 - d0 * a               # sur [P0 -> P1]
            E1 = P1 + d1 * a               # sur [P1 -> P2]

            # ---------------------------------------------------------------------------
            # Flat : we are done
            # ---------------------------------------------------------------------------

            if mode[i_corner] == MITER_FLAT:
                points[index] = E0
                points[index+1] = E1
                index += 2
                continue

            # ---------------------------------------------------------------------------
            # Arc between E0 and E1
            # ---------------------------------------------------------------------------

            chord = E1 - E0
            c = np.linalg.norm(chord)
            if c < D_MIN:
                points[index] = P1
                index += 1
                continue

            # rayon effectif : au moins c/2 pour que l'arc existe
            r_arc = max(r_eff, 0.5*c + EPS)

            mid = 0.5*(E0 + E1)
            # normale à la corde
            nhat = np.array([-chord[1], chord[0]], dtype=bfloat)
            nrm = np.linalg.norm(nhat)
            if nrm < D_MIN:
                points[index] = P1
                index += 1
                continue

            nhat /= nrm
            # oriente vers l'intérieur (du côté de P1)
            if np.dot(nhat, P1 - mid) > 0:
                nhat = -nhat

            # distance du centre à la corde
            h = np.sqrt(max(r_arc*r_arc - (0.5*c)*(0.5*c), 0.0))
            C = mid + nhat * h

            # angles depuis C vers E0 et E1
            a0 = np.arctan2(E0[1]-C[1], E0[0]-C[0])
            a1 = np.arctan2(E1[1]-C[1], E1[0]-C[0])
            # minor arc (dans (-pi, pi])
            da = (a1 - a0 + np.pi) % (2*np.pi) - np.pi

            m = max(2, int(resolution))
            thetas = a0 + np.linspace(0.0, da, m)
            arc = C + np.stack([np.cos(thetas), np.sin(thetas)], axis=-1) * r_arc

            points[index:index + m] = arc
            index += m

        # Not cylic: we append the last point

        if not cyclic:
             points[index] = self.points[-1]
             index += 1

        return points[:index].copy()
    
    # ====================================================================================================
    # Give a thickness to the poly
    # ====================================================================================================

    def set_thickness(self,
            thickness = .1,
            mode = 0,      
            factor = 1.0,
            cuts = (.1, np.nan),
            inner_mode = None,
            inner_factor = None,
            inner_cuts = None,
            resolution = 12,
            offset = 0.0,
            cyclic = False,
            start_thickness = 1,
            end_thickness = 1,
            ):
        
        # ----------------------------------------------------------------------------------------------------
        # Check argument
        # ----------------------------------------------------------------------------------------------------

        def check_param(param, corner_modes=None, mode=MITER_NONE, shape=(), label="Parameter"):

            if param is None:
                return None

            npoints = len(self.points)
            target_shape = (npoints,) + shape

            # Check the shape
            param = np.asarray(param)
            if param.shape == target_shape:
                return param
            
            if param.shape in [shape, (1,) + shape]:
                return np.resize(param, target_shape)

            # We accept (nmodes,...)            
            if corner_modes is not None:
                mask = corner_modes == mode
                nparams = np.sum(mask)

                if nparams != 0 and param.shape == (nparams,) + shape:
                    new_param = np.zeros(target_shape, dtype=param.dtype)
                    new_param[mask] = param
                    return new_param
            
            # No alternative            
            if corner_modes is None:
                raise ValueError(f"Parameter '{label}' must have a shape {target_shape}, not {param.shape}")
            
        # ----------------------------------------------------------------------------------------------------
        # Build the two poly lines
        # ----------------------------------------------------------------------------------------------------

        thickness = check_param(thickness, label="thickness")
        offset = check_param(offset, label="offset")
        # -1 -> -th   - 0
        #  0 -> -th/2 - th/2
        #  1 ->  0    - th
        offset = np.clip(offset, -1, 1)/2 - 0.5

        poly0 = self.perp_translated(thickness*offset, cyclic=cyclic)
        poly1 = self.perp_translated(thickness*(offset + 1), cyclic=cyclic)

        # ----------------------------------------------------------------------------------------------------
        # Miter the corners
        # ----------------------------------------------------------------------------------------------------

        mode = check_param(mode, label="mode")
        factor = check_param(factor, label="factor")
        cuts = check_param(cuts, corner_modes=mode, mode=MITER_CUT, shape=(2,), label="cuts")

        inner_mode = check_param(inner_mode, label="inner_mode")
        inner_factor = check_param(inner_factor, label="inner_factor")
        inner_cuts = check_param(inner_cuts, corner_modes=mode if inner_mode is None else inner_mode, mode=MITER_CUT, shape=(2,), label="inner_cuts")

        pts0 = poly0.miter_corners(
            mode = mode if inner_mode is None else inner_mode,
            factor = factor if inner_factor is None else inner_factor, 
            radius = thickness, 
            resolution = resolution, 
            cuts = cuts if inner_cuts is None else inner_cuts,
            cyclic = cyclic,
            )
        
        pts1 = poly1.miter_corners(
            mode = mode, 
            factor = factor, 
            radius = thickness, 
            resolution = resolution, 
            cuts = cuts, 
            cyclic = cyclic,
            )
        
        # ----------------------------------------------------------------------------------------------------
        # Finish the extremities
        # ----------------------------------------------------------------------------------------------------

        if not cyclic:
            if start_thickness != 1:
                P0 = self.points[0]
                if start_thickness < EPS:
                    pts0[0] = P0
                    pts1 = pts1[1:]
                else:
                    pts0[0] = P0 + start_thickness*(pts0[0] - P0)
                    pts1[0] = P0 + start_thickness*(pts1[0] - P0)
            
            if end_thickness != 1:
                P0 = self.points[-1]
                if end_thickness < EPS:
                    pts0[-1] = P0
                    pts1 = pts1[:-1]
                else:
                    pts0[-1] = P0 + end_thickness*(pts0[-1] - P0)
                    pts1[-1] = P0 + end_thickness*(pts1[-1] - P0)

        pts1 = np.flip(pts1, axis=0)
        return np.append(pts0, pts1, axis=0)
    
    # ====================================================================================================
    # Transvere thickness
    # ====================================================================================================

    def transverse_thickness(self, delta, invert=False):
        pts0 = np.array(self.points, copy=True)

        if invert:
            pts1 = pts0*(-1, -1) + delta
        else:
            pts1 = np.flip(pts0 + delta, axis=0)

        P0, P1, P = pts0[-1], pts1[0], None
        dx = P1[0] - P0[0]
        if dx > D_MIN:
            P = P0 + (dx, 0)
        elif dx < -D_MIN:
            P = P1 - (dx, 0)
        if P is not None:
            pts0 = np.append(pts0, [P], axis=0)

        P0, P1, P = pts1[-1], pts0[0], None
        dx = P1[0] - P0[0]
        if dx > D_MIN:
            P = P1 - (dx, 0)
        elif dx < -D_MIN:
            P = P0 + (dx, 0)
        if P is not None:
            pts1 = np.append(pts1, [P], axis=0)

        return np.append(pts0, pts1, axis=0)
    
    # ====================================================================================================
    # DEBUG
    # ====================================================================================================
    
    def _plot(self, ax, col, cyclic=False):
        if cyclic:
            pts = np.append(self.points, [self.points[0]], axis=0)
        else:
            pts = self.points

        ax.plot(pts[..., 0], pts[..., 1], col)

# ======================================================================================
# Main 
# ======================================================================================

def set_spline2d_thickness(points,
            thickness = .1,
            mode = 0,      
            factor = 1.0,
            cuts = (.1, np.nan),
            inner_mode = None,
            inner_factor = None,
            inner_cuts = None,
            resolution = 12,
            offset = 0.0,
            cyclic = False,
            start_thickness = 1,
            end_thickness = 1,
        ):
    """
    Build a 2D stroke outline from the polyline with per-corner miters/fillets.

    This method constructs a single polygonal outline from the current polyline by:
    (1) offsetting a copy to the "inner" side and resolving its corners,
    (2) offsetting a copy to the "outer" side and resolving its corners,
    then (3) reversing the outer side and concatenating both sides into one path.
    Corner geometry is produced by [`miter_corners`](npblender.maths.geo2d.Poly.miter_corners);
    the effective corner radius is computed per vertex as `radius = thickness * factor`.

    Parameters
    ----------
    thickness : float or array-like of shape (n,), default=0.1
        Target stroke width per vertex. Scalars are broadcast to all vertices.
        The distance between the two resulting offset sides is `thickness[i]` at vertex *i*.
    mode : int or array-like of shape (n,), default=0
        Corner style for the **outer** side:
        
        - `0` → *nothing* corner is not changed
        - `1` → *flat*  (bevel: straight segment between the two trimmed points)
        - `2` → *round* (fillet: arc sampled between the two trimmed points)
        - `3` → *cut* (cut the corner at distance and given angle) 
    factor : float or array-like of shape (n,), default=1.0
        Per-corner effect factor.
    cuts : couple of floats or array-like of couples of floats, default=(.1, np.nan)
        First value is the distance of the cut measured along the first segment.
        Second value is the angle (in radians) of the cut line. If second value is
        `np.nan`, the cut is perpendicular to the first segment.
    inner_mode : {None, int or array-like of shape (n,)}, default=None
        Corner style for the **inner** side. If `None`, falls back to `mode`.
    inner_factor : float or array-like of shape (n,), default=None
        Same as `factor` for **inner** side.
    inner_cuts : float or couple of floats or array-like of such values, default=None
        Same as `cuts` for inner line. If `None`, cuts is taken.
    resolution : int, default=12
        Number of samples used for each rounded corner (when the corresponding mode is `0`).
        Must be ≥ 2 to produce a visible arc.
    offset : float, default=0.0
        Centerline bias in the range `[-1, 1]` that determines how the stroke is split
        between the two sides. Internally mapped to side offsets as:
        
        - `-1` → `[-thickness, 0]`  (all thickness on the inner/left side)
        - ` 0` → `[-thickness/2, +thickness/2]` (centered stroke)
        - `+1` → `[0, +thickness]`  (all thickness on the outer/right side)

        Values are clipped to `[-1, 1]`.
    cyclic : bool, default=False
        If `True`, treat the polyline as closed; no end caps are applied and the outline
        is generated in closed form.
    start_thickness : float, default=1
        Start-cap scaling when `cyclic=False`. A value of `0` collapses the start cap
        onto the first vertex; values `> 0` scale the first outline points radially
        around the first vertex.
    end_thickness : float, default=1
        End-cap scaling when `cyclic=False`. A value of `0` collapses the end cap
        onto the last vertex; values `> 0` scale the last outline points radially
        around the last vertex.

    Returns
    -------
    pts : ndarray of shape (m, 2)
        The outline polygon: inner side (resolved with `inner_mode`) followed by the
        **reversed** outer side (resolved with `mode`). For `cyclic=True`, the outline is
        generated in closed form; otherwise, the path is open and not explicitly closed.

    Notes
    -----
    - Side offsets are computed via
    [`perp_translated`](npblender.maths.geo2d.Poly.perp_translated) using the two
    per-vertex offset amounts derived from `thickness` and `offset`.
    - Corners on each side are replaced by either a *bevel* (`mode==1`) or a *fillet arc*
    (`mode==0`) computed by
    [`miter_corners`](npblender.maths.geo2d.Poly.miter_corners) with
    `radius = thickness * factor`.
    - When `cyclic=False`, start/end caps are optionally tapered using `start_thickness`
    and `end_thickness`.

    > ***Caution:*** If a local segment is too short for the requested trim/fillet,
    internal clamping ensures the trim does not exceed half the adjacent segment length.

    Examples
    --------
    Centered round stroke:

    ```python
    outline = poly.set_thickness(
        thickness=0.12,
        offset=0.0,            # centered
        mode=0,                # round outer
        inner_mode=0,          # round inner
        resolution=16,
        cyclic=False
    )
    ```

    Asymmetric stroke with flat outer corners and rounded inner corners:

    ```python
    outline = poly.set_thickness(
        thickness=0.2,
        offset=+1,             # all thickness on outer/right side
        mode=1,                # flat outer
        inner_mode=0,          # round inner
        factor=0.8,
        resolution=12
    )
    ```

    Tapered open stroke (fade-in/out caps):

    ```python
    outline = poly.set_thickness(
        thickness=np.linspace(0.0, 0.15, len(poly.points)),
        mode=0,
        start_thickness=0.0,   # collapse start cap
        end_thickness=0.4      # shrink end cap
    )
    ```
    """
    
    points = np.asarray(points)
    poly = Poly(points[..., :2])

    pts = poly.set_thickness(
        thickness = thickness, 
        mode = mode, 
        factor = factor, 
        cuts = cuts,
        inner_mode = inner_mode, 
        inner_factor = inner_factor, 
        inner_cuts = inner_cuts,
        resolution = resolution, 
        offset = offset, 
        cyclic = cyclic, 
        start_thickness = start_thickness, 
        end_thickness = end_thickness,
        )
    
    if points.shape[-1] == 3:
        return np.hstack([pts, np.zeros((pts.shape[0], 1))])
    else:
        return pts

# ======================================================================================
# Démo / tests
# ======================================================================================
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    if True:
        pts =  np.array([
            [-0.15 ,      -0.48615   ],
            [-0.12370607, -0.48123962],
            [-0.10089844, -0.46731153],
            [-0.08131348, -0.44557023],
            [-0.06468751, -0.41722032],
            [-0.05075684, -0.3834663 ],
            [-0.03925781, -0.3455127 ],
            [-0.02992676, -0.3045641 ],
            [-0.0225   ,  -0.261825  ],
            [-0.01671387, -0.21849997],
            [-0.01230469, -0.17579356],
            [-0.00900879, -0.13491029],
            [-0.0065625,  -0.09705468],
            [-0.00470215, -0.06343132],
            [-0.00316406, -0.03524473],
            [-0.00168457, -0.01369944],
            [ 0.         , 0.        ],
            [ 0.00168457 , 0.01369944],
            [ 0.00316406 , 0.03524473],
            [ 0.00470215 , 0.06343132],
            [ 0.0065625  , 0.09705468],
            [ 0.00900879 , 0.13491029],
            [ 0.01230469 , 0.17579356],
            [ 0.01671387 , 0.21849997],
            [ 0.0225     , 0.261825  ],
            [ 0.02992676 , 0.3045641 ],
            [ 0.03925781 , 0.3455127 ],
            [ 0.05075684 , 0.3834663 ],
            [ 0.06468751 , 0.4172203 ],
            [ 0.08131348 , 0.44557023],
            [ 0.10089844 , 0.46731153],
            [ 0.12370607 , 0.48123962],
            [ 0.14999995 , 0.48615   ],])
        
        poly = Poly(pts)
        target = poly.set_thickness(
            thickness=.05, 
            )
        p0 = poly.perp_translated(.05)
        p1 = poly.perp_translated(-.05)
        
        poly._plot(ax, "black")
        #p0._plot(ax, "green")
        #p1._plot(ax, "blue")

        #ax.plot(target[:, 0], target[:, 1], "red")

        test = poly.transverse_thickness(delta=(.05, -.05))
        p = Poly(test)
        p._plot(ax, "blue")

        plt.show()
    

    if False:

        # Polyligne d'exemple
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.2],
            [2.0, 0.0],
            [3.0, 1.0],
            [1.5, 2.0],
        ], dtype=bfloat)

        dx, dy = .3, .15
        pts = np.array([[0, 0,], [dx, dy], [dx + 3*dy, dy - 3*dx], [dx + 5*dy, 2], [5, 2], [5, 1.8]], dtype=bfloat)
        pts *= 2

        w = .2
        th = [.8*w, 1.8*w, .9*w, w, w, w]

        cyclic = False

        poly = Poly(pts)

        p0 = poly.set_thickness(thickness=th, 
                                cyclic=cyclic, 
                                mode=0, 
                                inner_mode=[0, 0, MITER_CUT, 0, 0, 0],
                                inner_cuts= (.7, 0),
                                factor=1.,
                                end_thickness=0)
        #p0 = Poly(p0)


        #poly._plot(ax, "black", cyclic=cyclic)
        ax.plot(p0[:, 0], p0[:, 1], "green")
        #p0._plot(ax, "blue", cyclic=True)
        #p1._plot(ax, "green", cyclic=cyclic)

        plt.show()

