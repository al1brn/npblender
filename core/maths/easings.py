# npblender/maths/easings.py
# MIT License
# Created on 2025-08-11
# Last update: 2025-08-11
# Author: Alain Bernard

"""
Easing & Interpolation
======================

Vectorized easing functions and interpolation utilities with NumPy, plus a small
framework to chain segments (including Bézier F-Curves) in a Blender-friendly way.

This module provides:
- Normalized easing functions on u ∈ [0, 1] (vectorized with NumPy)
- Time/value interpolation: map [t0, t1] → [v0, v1] with a chosen easing
- A chain API via `EasingKey` and `Easings`:
  * Blender-like modes: LINEAR, CONSTANT, QUAD, CUBIC, QUART, QUINT,
    EXPO, SINE, CIRC, BACK, BOUNCE, ELASTIC, BEZIER
  * Extra S-curves: SMOOTH (smoothstep), SMOOTHER (smootherstep)
  * Bézier segments with absolute (t, v) handles
    - handle_type='AUTO' : recompute smooth C¹ handles on insertion
    - handle_type='FREE' : fill handles only if None, then preserve
  * Extrapolation per end key: 'CONSTANT' (hold) or 'LINEAR' (endpoint tangent)

Behavior & assumptions
----------------------
- Inputs are not clamped to [0, 1]; behavior outside the interval follows each easing.
- Division-by-zero guards are intentionally omitted for performance (ensure t1 ≠ t0).
- For non-Bézier segments, the endpoint tangent used by 'LINEAR' extrapolation is
  derived from the easing’s endpoint derivative; for Bézier, from the stored handles.
- Fully vectorized: functions accept scalars or arrays and broadcast as expected.

Public API (most useful entry points)
-------------------------------------
- `ease(mode, easing, u, ...)`      → easing value(s) on normalized `u`
- `maprange(t, mode, easing, t0, t1, v0, v1, ...)` → time-to-value interpolation
- `EasingKey(t0, v0, ...)`          → a single key with mode/easing/extrapolation
- `Easings(*keys)`                  → chain of segments
- `Easings.maprange(...)`           → convenience 2-key constructor
- `Easings.bezier((t, v), ...)`     → build a pure Bézier chain with auto handles
- Blender I/O:
  * `EasingKey.from_key_frame(kf)` / `EasingKey.to_key_frame(kf)`
  * `Easings.from_fcurve(fcurve)` / `Easings.to_fcurve(fcurve)`

Examples
--------
Basic sine in/out over [0, 1]:
    >>> t = np.linspace(0.0, 1.0, 256)
    >>> y = maprange(t, mode='SINE', easing='IN_OUT')

Using Easings.maprange with any intervals:
    >>> curve = Easings.maprange(-1.0, 4.0, -1.0, 1.0, 'SINE', easing='IN_OUT')
    >>> t = np.linspace(-1.0, 4.0, 256)
    >>> y = curve(t)

Bézier chain with automatic handles:
    >>> curve = Easings.bezier((0, 0), (1, 3), (2, 1))
    >>> t = np.linspace(0.0, 2.0, 200)
    >>> y = curve(t)

Dependencies
------------
- NumPy
"""




import numpy as np

# ====================================================================================================
# Constants
# ====================================================================================================

ZERO    = 1e-6
PI      = np.pi
TAU     = np.pi*2
HALF_PI = np.pi/2

LINEAR, CONSTANT = 0, 1
QUAD, CUBIC, QUART, QUINT, EXPO, SINE, CIRC = 2, 3, 4, 5, 6, 7, 8
BACK, BOUNCE, ELASTIC = 9, 10, 11
BEZIER = 12
SMOOTH, SMOOTHER = 13, 14

EASINGS = {
    'LINEAR' : LINEAR,
    'CONSTANT' : CONSTANT,
    'QUAD' : QUAD,
    'CUBIC' : CUBIC,
    'QUART' : QUART,
    'QUINT' : QUINT,
    'EXPO' : EXPO,
    'SINE' : SINE,
    'CIRC' : CIRC,
    'BACK' : BACK,
    'BOUNCE' : BOUNCE,
    'ELASTIC' : ELASTIC,
    'BEZIER' : BEZIER,
    'SMOOTH' : SMOOTH,
    'SMOOTHER' : SMOOTHER,
}
NON_BLENDER = [SMOOTH, SMOOTHER]

AUTO_EASING = {
    'CONSTANT'  : 'EASE_IN',
    'LINEAR'    : 'EASE_IN',
    'BEZIER'    : 'EASE_IN',
    'SINE'      : 'EASE_IN',
    'QUAD'      : 'EASE_IN',
    'CUBIC'     : 'EASE_IN',
    'QUART'     : 'EASE_IN',
    'QUINT'     : 'EASE_IN',
    'EXPO'      : 'EASE_IN',
    'CIRC'      : 'EASE_IN',
    
    'BACK'      : 'EASE_OUT',
    'BOUNCE'    : 'EASE_OUT',
    'ELASTIC'   : 'EASE_OUT',
}


def easing_code(mode):
    if isinstance(mode, str):
        try:
            return EASINGS[str(mode).upper()]
        except:
            raise ValueError(f"Invalid interpolation mode: '{mode}' not in {list(EASINGS.keys())}")
    else:
        return mode

# ====================================================================================================
# Core easing function: [0, 1] -> [0, 1]
# ====================================================================================================

def ease_linear(u):      return u
def ease_constant(u):    return (u >= 1.).astype(float)  # hold left, jump at 1
def ease_quad_in(u):     return u * u
def ease_cubic_in(u):    return u * u * u
def ease_quartic_in(u):  u2 = u * u; return u2 * u2
def ease_quintic_in(u):  u2 = u * u; return u2 * u2 * u
def ease_sine_in(u):     return 1.0 - np.cos(u * HALF_PI)
def ease_circ_in(u):     return 1.0 - np.sqrt(1.0 - u * u)
def ease_back_in(u, s):  return u * u * ((s + 1.0) * u - s)
def ease_exp_in(u, f):   return (np.power(2.0, f * u) - 1.0) / (np.power(2.0, f) - 1.0)

def bounce_out(u):
    n1, d1 = 7.5625, 2.75
    r = np.empty_like(u, dtype=float)
    c1 = u < 1.0/d1
    c2 = (u >= 1.0/d1) & (u < 2.0/d1)
    c3 = (u >= 2.0/d1) & (u < 2.5/d1)
    r[c1] = n1 * u[c1] * u[c1]
    r[c2] = n1 * (u[c2] - 1.5/d1)**2 + 0.75
    r[c3] = n1 * (u[c3] - 2.25/d1)**2 + 0.9375
    r[~(c1 | c2 | c3)] = n1 * (u[~(c1 | c2 | c3)] - 2.625/d1)**2 + 0.984375
    return r

def elastic_out(u, A=1.0, p=0.3):
    s = np.where(A < 1.0, p/4.0, (p/TAU) * np.arcsin(1.0/A))
    res = A * np.power(2.0, -10.0 * u) * np.sin((u - s) * (TAU / p)) + 1.0
    res = np.where(u == 0.0, 0.0, res)
    res = np.where(u == 1.0, 1.0, res)
    return res

# --- core S-curves on u ∈ [0,1]
def smoothstep(u):
    """C^1 S-curve: 3u^2 - 2u^3"""
    return (u * u) * (3.0 - 2.0 * u)

def smootherstep(u):
    """C^2 S-curve: 6u^5 - 15u^4 + 10u^3"""
    u2 = u * u
    u3 = u2 * u
    return u3 * (u * (6.0 * u - 15.0) + 10.0)


# ====================================================================================================
# in / out and in_out
# ====================================================================================================

def ease_in(f, u, *args):      return f(u, *args) if args else f(u)
def ease_out(f, u, *args):     return 1.0 - ease_in(f, 1.0 - u, *args)
def ease_in_out(f, u, *args):
    left = u < 0.5
    out  = np.empty_like(u, dtype=float)
    u2   = u * 2.0
    out[left]  = 0.5 * ease_in(f, u2[left], *args)
    out[~left] = 1.0 - 0.5 * ease_in(f, 2.0 - u2[~left], *args)
    return out

# ====================================================================================================
# Dispatcher
# ====================================================================================================

def ease(mode='LINEAR', easing='IN', u=0., factor=1., back=1.70158, amplitude=1., period=.3):
    mode = easing_code(mode)                         
    easing = str(easing).upper()
    u = np.asarray(u)

    # ----- Ignore easing parameter
    if mode == CONSTANT:
        return ease_constant(u)
    elif mode == SMOOTH:
        return smoothstep(u)
    elif mode == SMOOTHER:
        return smootherstep(u)
    
    # ----- Easing parameter matters
    if mode == LINEAR:
        base = ease_linear; args = ()
    elif mode == QUAD:
        base = ease_quad_in;  args = ()
    elif mode == CUBIC:
        base = ease_cubic_in; args = ()
    elif mode == QUART:
        base = ease_quartic_in; args = ()
    elif mode == QUINT:
        base = ease_quintic_in; args = ()
    elif mode == SINE:
        base = ease_sine_in;  args = ()
    elif mode == CIRC:
        base = ease_circ_in;  args = ()
    elif mode == BACK:
        base = ease_back_in;  args = (float(back),)
    elif mode == EXPO:
        base = ease_exp_in;   args = (float(factor),)
    elif mode == BOUNCE:
        # base is OUT by definition
        if easing == 'IN':     return 1.0 - bounce_out(1.0 - u)
        if easing == 'OUT':    return bounce_out(u)
        if easing == 'IN_OUT':
            left = u < 0.5
            out = np.empty_like(u, dtype=float)
            out[left]  = 0.5 * (1.0 - bounce_out(1.0 - 2.0*u[left]))
            out[~left] = 0.5 *  bounce_out(2.0*u[~left] - 1.0) + 0.5
            return out
        raise ValueError(easing)
    elif mode == ELASTIC:
        # base is OUT by definition
        A, p = float(amplitude), float(period)
        if easing == 'IN':     return 1.0 - elastic_out(1.0 - u, A, p)
        if easing == 'OUT':    return elastic_out(u, A, p)
        if easing == 'IN_OUT':
            u2 = 2.0 * u
            left = u < 0.5
            out = np.empty_like(u, dtype=float)
            out[left]  = 0.5 * (1.0 - elastic_out(1.0 - u2[left], A, p))
            out[~left] = 0.5 *  elastic_out(u2[~left] - 1.0, A, p) + 0.5
            return out
        raise ValueError(easing)
    else:
        raise ValueError(mode)

    if easing == 'IN':     return ease_in(base, u, *args)
    if easing == 'OUT':    return ease_out(base, u, *args)
    if easing == 'IN_OUT': return ease_in_out(base, u, *args)

    raise ValueError(easing)

# ====================================================================================================
# Interpolation
# ====================================================================================================

def maprange(t, mode='LINEAR', easing='IN',
           t0=0., t1=1., v0=0., v1=1.,
           factor=1., back=1.70158, amplitude=1., period=.3,
           normalized=False):
    """
    If normalized=True, `t` is already u∈[0,1] and we only remap to [v0,v1].
    Otherwise we compute u = (t - t0) / (t1 - t0) first.
    """
    if normalized:
        u = np.asarray(t)
    else:
        t, t0, t1 = map(np.asarray, (t, t0, t1))
        u = (t - t0) / (t1 - t0)

    eased = ease(mode, easing, u, factor, back, amplitude, period)
    v0, v1 = map(np.asarray, (v0, v1))
    return v0 + (v1 - v0) * eased


# ====================================================================================================
# Derivatives for extrapolation
# ====================================================================================================

def _deriv_mult_base_in(mode, at, factor=1.0, back=1.70158):
    """
    Return f'(0) or f'(1) for the base IN easing (u in [0,1]).
    For modes with undefined/ill-conditioned endpoint slopes, return np.nan to trigger fallback.
    """
    mode = easing_code(mode)

    if mode == LINEAR:
        return 1.0  # both ends

    if mode in [CONSTANT, SMOOTH, SMOOTHER]:
        return 0.0  # flat except jump

    if mode == QUAD:
        return 0.0 if at == 'start' else 2.0

    if mode == CUBIC:
        return 0.0 if at == 'start' else 3.0

    if mode == QUART:
        return 0.0 if at == 'start' else 4.0

    if mode == QUINT:
        return 0.0 if at == 'start' else 5.0

    if mode == SINE:
        # f(u) = 1 - cos(pi/2 * u), f'(u) = (pi/2) * sin(pi/2 * u)
        return 0.0 if at == 'start' else (np.pi / 2.0)

    if mode == CIRC:
        # f'(u) = u / sqrt(1 - u^2) → f'(0)=0, f'(1)=∞
        return 0.0 if at == 'start' else np.nan  # force fallback on right end

    if mode == BACK:
        # f(u) = (s+1)u^3 - s u^2 → f'(u) = 3(s+1)u^2 - 2 s u
        s = float(back)
        return 0.0 if at == 'start' else (s + 3.0)

    if mode == EXPO:
        # normalized: f(u) = (2^{k u} - 1)/(2^k - 1)
        # f'(u) = ln(2)*k*2^{k u} / (2^k - 1)
        k = float(factor)
        denom = (2.0 ** k) - 1.0
        if denom == 0.0:
            return 1.0  # k→0 behaves like linear
        if at == 'start':
            return np.log(2.0) * k / denom
        else:
            return np.log(2.0) * k * (2.0 ** k) / denom

    if mode in (BOUNCE, ELASTIC):
        # endpoint derivative is not stable or not well-defined for extrapolation
        return np.nan

    # Unknown mode
    return np.nan

def _deriv_mult(mode, easing, end, factor=1.0, back=1.70158):
    """
    Derivative multiplier at a segment endpoint in the chosen easing variant.
    end ∈ {'left','right'} for u=0 or u=1 respectively.
    For OUT and IN_OUT, map to base IN derivatives:
      OUT:  f_out(u) = 1 - f_in(1-u) → f_out'(0) = f_in'(1),  f_out'(1) = f_in'(0)
      IN_OUT:
        near u=0:   0.5 * f_in(2u)         → slope = f_in'(0)
        near u=1:   1 - 0.5 * f_in(2-2u)   → slope = f_in'(0)
    """
    at = 'start' if end == 'left' else 'end'
    if easing == 'IN':
        return _deriv_mult_base_in(mode, at, factor, back)
    if easing == 'OUT':
        # swap ends relative to base IN
        at_swapped = 'end' if at == 'start' else 'start'
        return _deriv_mult_base_in(mode, at_swapped, factor, back)
    if easing == 'IN_OUT':
        # both ends behave like base-IN at start (scaled, but multiplier preserved)
        return _deriv_mult_base_in(mode, 'start', factor, back)
    return np.nan

def _finite_or(value, fallback):
    """Return value if finite, else fallback."""
    return value if np.isfinite(value) else fallback

# ====================================================================================================
# Easing
# ====================================================================================================

class EasingKey:
    def __init__(self, t0, v0, mode='LINEAR', easing='IN',
                 extrapolation='CONSTANT', left=None, right=None,
                 handle_type='AUTO', **kwargs):
        """
        left  : (tx, vy) used when the *previous* point is BEZIER
        right : (tx, vy) used when *this* point is BEZIER
        handle_type ∈ {'AUTO','FREE'}:
          - 'AUTO' : handles are recomputed on each insertion via Easings.compute_bezier_handles()
          - 'FREE' : handles are computed only if None, then kept as-is
        """
        self.t0 = t0
        self.v0 = v0

        if mode not in EASINGS:
            raise ValueError(f"Easing mode '{mode}' must be in {list(EASINGS.keys())}.")
        if easing not in ('IN', 'OUT', 'IN_OUT'):
            raise ValueError(f"Easing '{easing}' must be in ('IN','OUT','IN_OUT').")
        if extrapolation not in ('CONSTANT', 'LINEAR'):
            raise ValueError(f"Extrapolation '{extrapolation}' must be in ('CONSTANT','LINEAR').")
        if handle_type not in ('AUTO', 'FREE'):
            raise ValueError(f"handle_type '{handle_type}' must be 'AUTO' or 'FREE'.")

        self.mode           = mode
        self.easing         = easing
        self.extrapolation  = extrapolation
        self.handle_type    = handle_type

        self.factor    = kwargs.get('factor', 1.0)
        self.back      = kwargs.get('back', 1.70158)
        self.period    = kwargs.get('period', 0.3)
        self.amplitude = kwargs.get('amplitude', 1.0)

        # Bezier handles (absolute (t,v) coords); default None
        self.left  = left
        self.right = right

        self.kwargs = kwargs  # keep other params (factor, back, amplitude, period...)

    # ====================================================================================================
    # Interface with Blender keyframes
    # ====================================================================================================

    @classmethod
    def from_key_frame(cls, kf):

        easing = kf.easing
        if easing == 'AUTO':
            easing = AUTO_EASING[kf.interpolation]
        easing = easing[5:] 

        return cls(
            t0 = kf.co.x, 
            v0 = kf.co.y, 
            mode = kf.interpolation,
            easing=easing,
            back = kf.back,
            period = kf.period,
            amplitude = kf.amplitude,
            left = np.asarray(kf.handle_left),
            right = np.asarray(kf.handle_right),
            handle_type = 'FREE',
            )
    
    def to_key_frame(self, kf):
        
        from mathutils import Vector
        
        kf.type          = 'KEYFRAME'
        kf.interpolation = self.mode
        kf.easing        = 'EASE_' + self.easing
        #kf.factor        = self.factor
        kf.back          = self.back
        kf.period        = self.period
        kf.amplitude     = self.amplitude
        kf.co            = Vector((self.t0, self.v0))

        kf.handle_left_type  = 'FREE'
        kf.handle_right_type = 'FREE'
        kf.handle_left   = Vector(self.left)
        kf.handle_right  = Vector(self.right)

        return kf        
        
# ====================================================================================================
# Easing
# ====================================================================================================

class Easings:
    """
    A chain of Easing segments. Each Easing defines a key at (t0, v0) and
    interpolation mode/easing/extrapolation *to the NEXT key*.
    """
    def __init__(self, *items):
        self.items = []
        for it in items:
            self.add(it)

    def _check_sorted_unique(self):
        t0s = [e.t0 for e in self.items]
        if any(t0s[i] >= t0s[i+1] for i in range(len(t0s)-1)):
            raise ValueError("Easing keys must have strictly increasing t0.")

    def add(self, easing: "EasingKey"):

        """Insert keeping t0 strictly increasing (no duplicates)."""
        if not self.items:
            self.items.append(easing)

        else:
            # binary insert
            t0 = easing.t0
            lo, hi = 0, len(self.items)
            while lo < hi:
                mid = (lo + hi) // 2
                if self.items[mid].t0 < t0:
                    lo = mid + 1
                else:
                    hi = mid

            if lo < len(self.items) and np.isclose(self.items[lo].t0, t0):
                raise ValueError(f"Duplicate key time t0={t0}.")

            self.items.insert(lo, easing)
            self._check_sorted_unique()

        # ===== Recompute Bezier handles

        self.compute_bezier_handles()

    def __len__(self):
        return len(self.items)

    def times(self):
        return np.array([e.t0 for e in self.items])

    def values(self):
        return [e.v0 for e in self.items]
    
    # ====================================================================================================
    # Bezier utilities
    # ====================================================================================================

    @staticmethod
    def _secant(t0, v0, t1, v1):
        return (v1 - v0) / (t1 - t0)
    
    def _endpoint_slope(self, i):
        """
        Estimate dv/dt at key i using surrounding keys.
        Monotone-clamped average of adjacent secants; endpoint -> single secant.
        """
        items = self.items

        n = len(items)
        ti, vi = items[i].t0, items[i].v0
        if i == 0:
            t1, v1 = items[1].t0, items[1].v0
            return self._secant(ti, vi, t1, v1)
        if i == n - 1:
            t0, v0 = items[n-2].t0, items[n-2].v0
            return self._secant(t0, v0, ti, vi)

        t0, v0 = items[i-1].t0, items[i-1].v0
        t1, v1 = items[i+1].t0, items[i+1].v0
        s_left  = self._secant(t0, v0, ti, vi)
        s_right = self._secant(ti, vi, t1, v1)

        # Monotone clamp (Fritsch–Carlson style): if slopes flip sign, use 0
        if s_left == 0.0 or s_right == 0.0 or (s_left > 0 and s_right < 0) or (s_left < 0 and s_right > 0):
            return 0.0
        # Average; you can weight by time spans if desired
        return 0.5 * (s_left + s_right)

    def _default_handles_for_segment(self, i_left, frac=1.0/3.0):
        """
        Compute default (right, left) handles for segment [i_left -> i_left+1].
        Returns P1 (right of left key) and P2 (left of right key) as (tx, vy) pairs.
        """

        items = self.items

        e0 = items[i_left]
        e1 = items[i_left + 1]
        t0, v0 = e0.t0, e0.v0
        t1, v1 = e1.t0, e1.v0
        dt = (t1 - t0)

        m0 = self._endpoint_slope(i_left)
        m1 = self._endpoint_slope(i_left + 1)

        # Place handles a fraction of dt along time axis, project using slopes
        hx1 = t0 + frac * dt
        hy1 = v0 + m0  * (hx1 - t0)

        hx2 = t1 - frac * dt
        hy2 = v1 - m1  * (t1 - hx2)

        return (hx1, hy1), (hx2, hy2)
    
    @staticmethod
    def _bezier_eval_segment(t_query, P0, P1, P2, P3, max_iter=10, tol=1e-6):
        """
        Evaluate cubic Bezier in (time,value) space.
        Pk = (tx, vy). Solve Bx(s)=t for s∈[0,1], then return By(s).
        """
        tq = np.asarray(t_query)
        orig_shape = tq.shape
        t_query = tq.ravel()

        x0, y0 = P0; x1, y1 = P1; x2, y2 = P2; x3, y3 = P3

        def bx(s):
            inv = 1.0 - s
            return (inv*inv*inv)*x0 + 3.0*(inv*inv)*s*x1 + 3.0*inv*(s*s)*x2 + (s*s*s)*x3

        def by(s):
            inv = 1.0 - s
            return (inv*inv*inv)*y0 + 3.0*(inv*inv)*s*y1 + 3.0*inv*(s*s)*y2 + (s*s*s)*y3

        def bx_prime(s):
            inv = 1.0 - s
            return 3.0*(inv*inv)*(x1 - x0) + 6.0*inv*s*(x2 - x1) + 3.0*(s*s)*(x3 - x2)

        out = np.empty_like(t_query, dtype=float)
        # initial guess: linear in time
        u0 = np.where(x3 != x0, (t_query - x0) / (x3 - x0), 0.0)
        u0 = np.clip(u0, 0.0, 1.0)

        for i, tq in enumerate(t_query):
            s = u0[i]; lo, hi = 0.0, 1.0
            xs = bx(s)
            for _ in range(max_iter):
                dx = xs - tq
                if abs(dx) <= tol:
                    break
                d = bx_prime(s)
                if d != 0.0 and np.isfinite(d):
                    s_new = s - dx / d
                    if s_new < lo or s_new > hi:
                        s_new = 0.5 * (lo + hi)  # bisection
                    else:
                        if dx > 0: hi = s
                        else:      lo = s
                    s = np.clip(s_new, lo, hi)
                else:
                    s = 0.5 * (lo + hi)
                xs = bx(s)
                if xs > tq: hi = s
                else:       lo = s
            out[i] = by(s)

        return out.reshape(orig_shape)
    
    def compute_bezier_handles(self, frac=1.0/3.0):
        """
        Recompute default Bezier handles:
        - For a BEZIER key with handle_type 'AUTO' → always reset its handle(s).
        - For a BEZIER key with handle_type 'FREE' → set only if currently None.
        Right handle belongs to the left key of the segment; left handle to the right key.
        """
        if len(self.items) < 2:
            return

        for i in range(len(self.items) - 1):
            e0, e1 = self.items[i], self.items[i + 1]
            # default handles for segment i -> i+1
            def_right, def_left = self._default_handles_for_segment(i, frac=frac)

            if e0.mode == 'BEZIER':
                # Right handle belongs to e0 (used when e0.mode == 'BEZIER')
                if e0.handle_type == 'AUTO' or e0.right is None:
                    e0.right = def_right

                # Left handle belongs to e1 (used when previous is BEZIER)
                if e1.handle_type == 'AUTO' or e1.left is None:
                    e1.left = def_left
    
    # ====================================================================================================
    # First or last segment tangent for extrapolation
    # ====================================================================================================

    def _segment_tangent_value(self, i_left, t, end='right'):
        """
        Linear extrapolation using the tangent at the endpoint of segment [i_left -> i_left+1].
        end='right' uses slope at u=1; end='left' uses slope at u=0.
        Falls back to secant if slope is non-finite.
        """
        e0 = self.items[i_left]
        e1 = self.items[i_left + 1]
        t0, t1 = e0.t0, e1.t0
        v0, v1 = np.asarray(e0.v0), np.asarray(e1.v0)

        # If the segment is BEZIER, use handle slopes at the end
        if e0.mode == 'BEZIER':
            right = e0.right
            left  = e1.left
            if right is None or left is None:
                def_right, def_left = self._default_handles_for_segment(i_left, frac=1.0/3.0)
                if right is None: right = def_right
                if left  is None: left  = def_left

            if end == 'left':
                slope = (right[1] - v0) / (right[0] - t0)
                t_anchor, v_anchor = t0, v0
            else:
                slope = (v1 - left[1]) / (t1 - left[0])
                t_anchor, v_anchor = t1, v1
            return v_anchor + slope * (np.asarray(t) - t_anchor)

        # Fallback: easing-based tangent multiplier
        dt = (t1 - t0)
        dv = (v1 - v0)
        m = _deriv_mult(e0.mode, e0.easing, end=end,
                        factor=e0.kwargs.get('factor', 1.0),
                        back=e0.kwargs.get('back', 1.70158))
        secant = dv / dt
        slope = _finite_or(m * secant, secant)
        if end == 'right':
            t_anchor, v_anchor = t1, v1
        else:
            t_anchor, v_anchor = t0, v0
        return v_anchor + slope * (np.asarray(t) - t_anchor)
    
    # ====================================================================================================
    # Evaluation
    # ====================================================================================================

    def evaluate(self, t):
        """
        Vectorized evaluation over t (scalar or array). For each interval [t_i, t_{i+1}),
        call the segment's interpolation; for the last key, use its extrapolation rule.
        """
        if not self.items:
            raise ValueError("No keys in Easings.")
        if len(self.items) == 1:
            # Single key: constant (linear fallback to constant)
            t = np.asarray(t)
            return np.broadcast_to(np.asarray(self.items[0].v0), t.shape)

        # Gather key times and values
        key_t = self.times()
        t = np.asarray(t)

        # For each t, find the left key index:
        # idx = searchsorted(..., 'right') - 1 gives i such that key_t[i] <= t < key_t[i+1]
        idx = np.searchsorted(key_t, t, side='right') - 1

        # Masks for left extrapolation, right extrapolation, and interior
        left_mask  = idx < 0
        right_mask = idx >= (len(key_t) - 1)
        mid_mask   = (~left_mask) & (~right_mask)

        out = None

        # Interior segments
        if np.any(mid_mask):
            # For each interior t, evaluate its segment i -> i+1
            out_mid = np.empty_like(t, dtype=float)

            # If v0 are vectors/arrays, we'll accumulate in object then broadcast at the end
            # Better: evaluate per unique segment to minimize calls
            unique_left_idx = np.unique(idx[mid_mask])
            # For each distinct segment, evaluate all its t's at once
            for i_left in unique_left_idx:
                seg_mask = mid_mask & (idx == i_left)
                e0 = self.items[i_left]
                e1 = self.items[i_left + 1]

                if e0.mode == 'BEZIER':
                    # use stored handles; if missing for any reason, compute defaults on the fly (no mutation)
                    right = e0.right
                    left  = e1.left
                    if right is None or left is None:
                        def_right, def_left = self._default_handles_for_segment(i_left, frac=1.0/3.0)
                        if right is None: right = def_right
                        if left  is None: left  = def_left

                    y = self._bezier_eval_segment(
                        t[seg_mask],
                        (e0.t0, e0.v0), right, left, (e1.t0, e1.v0)
                    )
                else:
                    y = maprange(
                        t[seg_mask],
                        mode=e0.mode,
                        easing=e0.easing,
                        t0=e0.t0, t1=e1.t0,
                        v0=e0.v0, v1=e1.v0,
                        **e0.kwargs
                    )

                out_mid[seg_mask] = y

            out = out_mid

        # Left extrapolation
        if np.any(left_mask):
            e0 = self.items[0]
            if e0.extrapolation == 'CONSTANT' or len(self.items) == 1:
                left_vals = np.broadcast_to(np.asarray(e0.v0), t[left_mask].shape)
            elif e0.extrapolation == 'LINEAR':
                # use tangent at left end of first segment
                left_vals = self._segment_tangent_value(0, t[left_mask], end='left')
            else:
                raise ValueError(f"Unknown extrapolation '{e0.extrapolation}'")

            if out is None:
                out = np.empty_like(t, dtype=float)
            out[left_mask] = left_vals

        # Right extrapolation
        if np.any(right_mask):
            e_last = self.items[-1]
            if e_last.extrapolation == 'CONSTANT' or len(self.items) == 1:
                right_vals = np.broadcast_to(np.asarray(e_last.v0), t[right_mask].shape)
            elif e_last.extrapolation == 'LINEAR':
                # use tangent at right end of last segment
                right_vals = self._segment_tangent_value(len(self.items) - 2, t[right_mask], end='right')
            else:
                raise ValueError(f"Unknown extrapolation '{e_last.extrapolation}'")

            if out is None:
                out = np.empty_like(t, dtype=float)
            out[right_mask] = right_vals

        return out

    def __call__(self, t):
        return self.evaluate(t)
    
    # ====================================================================================================
    # Interface with Blender FCurve
    # ====================================================================================================

    @classmethod
    def from_fcurve(cls, fcurve):
        return cls(*[EasingKey.from_key_frame(kf) for kf in fcurve.keyframe_points])
    
    def to_fcurve(self, fcurve):
        
        kfs = fcurve.keyframe_points
        
        kfs.clear()
        kfs.add(len(self))
        
        for ek, kf in zip(self.items, kfs):
            ek.to_key_frame(kf)

        return fcurve
    
    # ====================================================================================================
    # Constructors
    # ====================================================================================================

    @classmethod
    def maprange(cls, t0=0., t1=1., v0=0., v1=1., mode='LINEAR', easing='IN', extrapolation='CONSTANT', **kwargs):
        return cls(
            EasingKey(t0, v0, mode=mode, easing=easing, extrapolation=extrapolation, **kwargs),
            EasingKey(t1, v1, mode=mode, easing=easing, extrapolation=extrapolation, **kwargs),
        )
    
    @classmethod
    def bezier(cls, *points, extrapolation='CONSTANT', handle_type='AUTO'):
        """
        Build a BEZIER easing chain from (t, v) points.
        - points: either (t, v), (t, v), ... or a single iterable of (t, v)
        - handle_type: 'AUTO' (recompute each time) or 'FREE' (fill None once)
        - extrapolation: 'CONSTANT' or 'LINEAR'
        Handles are computed via compute_bezier_handles() for a smooth curve.
        """
        # Accept both signatures: (p0, p1, ...) or ([p0, p1, ...],)
        if len(points) == 1 and hasattr(points[0], '__iter__') and points[0] and hasattr(points[0][0], '__iter__'):
            pts = list(points[0])
        else:
            pts = list(points)

        if not pts:
            raise ValueError("At least one (t, v) point is required.")

        # Sort by time and create keys
        pts = sorted(pts, key=lambda p: p[0])
        chain = cls()
        for t, v in pts:
            chain.add(EasingKey(t, v,
                                mode='BEZIER',
                                easing='IN',                # ignored for BEZIER
                                extrapolation=extrapolation,
                                handle_type=handle_type))

        # add() already calls compute_bezier_handles(), but call once more to be explicit
        chain.compute_bezier_handles()
        return chain
