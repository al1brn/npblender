from pprint import pprint
import numpy as np

# Spline (abstract)
# ├── SplineFunction            # function derivative
# └── PointsSpline (abstract)   # interpolation
#     ├── Poly
#     └── Bezier
#     └── Nurbs (not implemented yet)

# ====================================================================================================
# Spline
# ====================================================================================================

class Spline:

    def __init__(self, cyclic=True):
        self.cyclic = cyclic

    # ----------------------------------------------------------------------------------------------------
    # To be overloaded
    # ----------------------------------------------------------------------------------------------------

    @property
    def is_scalar(self):
        raise Exception(f"Spline.is_scalar must be overloaded")
    
    def __len__(self):
        raise Exception(f"Spline.__len__ must be overloaded")

    # resample
    def resample(self, count):
        pass

    # resample
    def evaluate(self, t):
        raise Exception(f"Spline.evaluate must be overloaded")
    
    # ----------------------------------------------------------------------------------------------------
    # Clip t
    # ----------------------------------------------------------------------------------------------------
    
    def _clip_t(self, t):
        """
        Normalize input t ∈ [0, 1] depending on cyclicity.

        If cyclic, wrap values using modulo.
        If non-cyclic, clip values to [0, 1 - ε].

        Parameters
        ----------
        t : float or np.ndarray

        Returns
        -------
        t_processed : same type and shape as input
        """
        t = np.asarray(t)
        if self.cyclic:
            return np.mod(t, 1.0)
        else:
            return np.clip(t, 0.0, 1.0 - 1e-8)

    # ----------------------------------------------------------------------------------------------------
    # Default length
    # ----------------------------------------------------------------------------------------------------

    def length(self):
        t = np.linspace(0, 1, 100)
        v = self.evaluate(t)
        if self.is_scalar:
            ds = np.linalg.norm(np.diff(v, axis=0), axis=1)
            return np.sum(ds)
        else:
            ds = np.linalg.norm(np.diff(v, axis=1), axis=2)  # shape (B, 99)
            return np.sum(ds, axis=1)  # shape (B,)
        
    # ----------------------------------------------------------------------------------------------------
    # Default tangent
    # ----------------------------------------------------------------------------------------------------

    def tangent(self, t):
        dt = 1/1000
        v0 = self.evaluate(t - dt)
        v1 = self.evaluate(t + dt)
        return (v1 - v0)/(2*dt)

    # ----------------------------------------------------------------------------------------------------
    # Default attribute
    # ----------------------------------------------------------------------------------------------------

    def sample_attribute(self, t, attribute):
        """
        Sample a given attribute along the spline using linear interpolation.

        Parameters
        ----------
        t : float or np.ndarray
            Sampling locations ∈ [0, 1], shape (), (T,), or (T, B).
        attribute : np.ndarray
            Array of shape (n,) or (n, D) with values to interpolate between.

        Returns
        -------
        interpolated : np.ndarray
            Interpolated values at locations t, shape (T, D), (T, B, D), or similar.
        """
        attr = np.asarray(attribute)
        n = attr.shape[0]
        D = 1 if attr.ndim == 1 else attr.shape[1:]

        t_clipped = self._clip_t(t)
        s = t_clipped * (n - 1)
        i = np.floor(s).astype(int)
        u = s - i  # fractional part

        # Gather values
        a0 = attr[i]             # shape of t + D
        a1 = attr[i + 1]         # shape of t + D

        result = (1 - u)[..., None] * a0 + u[..., None] * a1 if attr.ndim == 2 else (1 - u) * a0 + u * a1
        return result
    
    # ----------------------------------------------------------------------------------------------------
    # Conversion to poly
    # ----------------------------------------------------------------------------------------------------

    def to_poly(self, count=100):
        """
        Approximate this spline by a Poly spline using sampled points.

        Parameters
        ----------
        count : int
            Number of control points in the Poly approximation.

        Returns
        -------
        Poly : a new Poly spline instance
        """
        t = np.linspace(0, 1, count, endpoint=not self.cyclic)
        return Poly(self.evaluate(t), cyclic=self.cyclic, already_closed=False)
    
    # ----------------------------------------------------------------------------------------------------
    # Conversion to bezier
    # ----------------------------------------------------------------------------------------------------

    def to_bezier(self, count=20, handle_scale=0.3):
        """
        Approximate this spline by a Bézier spline using sampled points and estimated tangents.

        Parameters
        ----------
        count : int
            Number of control points in the Bézier approximation.
        handle_scale : float
            Scale factor for the distance from anchors to handles along the tangent.

        Returns
        -------
        Bezier : a new Bézier spline instance
        """
        count = count + 1 if self.cyclic else count

        # Sample points and tangents along the spline
        t_values = np.linspace(0, 1, count)
        anchors = self.evaluate(t_values)          # shape (count, 3)
        tangents = self.tangent(t_values)          # shape (count, 3)

        # Estimate average spacing between anchors to scale handles
        d = np.linalg.norm(np.diff(anchors, axis=0), axis=1)  # (count - 1,)
        avg_dist = np.concatenate([
            d[:1],
            (d[:-1] + d[1:]) / 2,
            d[-1:]
        ])[..., None]                                           # shape (count, 1)

        # Offset handles along tangents
        offset = handle_scale * avg_dist * tangents
        left_handles  = anchors - offset
        right_handles = anchors + offset

        return Bezier(
            anchors,
            left_handles=left_handles,
            right_handles=right_handles,
            cyclic=self.cyclic,
            already_closed=True  # anchors already closed if needed
        )
    
# ====================================================================================================
# Spline Function
# ====================================================================================================

class SplineFunction(Spline):

    def __init__(self, func, derivative=None, cyclic=True):
        super().__init__(cyclic)
        self._func = func
        self._derivative = derivative

    def __str__(self):
        return f"<SplineFunction {self._func.__name__}, cyclic: {self.cyclic}>"

    @property
    def is_scalar(self):
        return True
    
    def __len__(self):
        raise Exception(f"SplineFunction is scalar.")

    def evaluate(self, t):
        return self._func(self._clip_t(t))
    
    def tangent(self, t):
        if self._derivative is None:
            return super().tangent(t)
        else:
            return self._derivative(self._clip_t(t))
        
    # ====================================================================================================
    # Helpers
    # ====================================================================================================

    @staticmethod
    def _circle(t, radius=1., derivative=False):

        w = 2*np.pi
        ag = w*t

        rcag = radius*np.cos(ag) # -> -w.sin
        rsag = radius*np.sin(ag) # -> w.cos

        if derivative:
            return np.stack(((-w)*rsag, w*rcag, np.zeros(rcag.shape)), axis=-1)    
        else:
            return np.stack((rcag, rsag, np.zeros(rcag.shape)), axis=-1)    

    @staticmethod
    def _spiral(t, r0=1, r1=1, angle0=0, angle1=2*np.pi, z0=0, z1=0, derivative=False):

        r_slope  = r1 - r0
        ag_slope = angle1 - angle0
        z_slope  = z1 - z0

        r = r0 + r_slope*t
        ag = angle0 + ag_slope*t

        cag = np.cos(ag)
        sag = np.sin(ag)

        if derivative:
            x = r_slope*cag - ag_slope*(r*sag)
            y = r_slope*sag + ag_slope*(r*cag)
            z = np.ones(x.shape)*z_slope
        else:
            x = r*cag
            y = r*sag
            z =  z0 + z_slope*t

        return np.stack((x, y, z), axis=-1)

    @classmethod
    def circle(cls, radius=1.):
        func = lambda t: cls._circle(t, radius=radius, derivative=False)
        der  = lambda t: cls._circle(t, radius=radius, derivative=True)
        return cls(func, derivative=der, cyclic=True)

    @classmethod
    def spiral(
        cls,
        radius0 = 1.,
        radius1 = 1.,
        angle0 = 0.,
        angle1 = 6*np.pi,
        z0 = 0.,
        z1 = 1.,
        ):
        func = lambda t: cls._spiral(t, radius0, radius1, angle0, angle1, z0, z1, derivative=False)
        der  = lambda t: cls._spiral(t, radius0, radius1, angle0, angle1, z0, z1, derivative=True)
        return cls(func, derivative=der, cyclic=False)
    
    # ====================================================================================================
    # For tests : plot
    # ====================================================================================================
    
    def _plot(self, resolution=100, display_points='NO', label=None, ax=None, **kwargs):
        """
        Plot the Poly spline using matplotlib.

        Parameters:
            resolution (int): Number of points to evaluate the curve.
            display_points (str): 'NO', 'POINTS', 'HANDLES', or 'ALL'
                                (only 'POINTS' and 'ALL' are used for Poly).
            label (str): Legend label for the spline.
            ax (matplotlib.axes.Axes): Optional matplotlib axis to draw on.
            **kwargs: Additional keyword arguments for `plot()` and `scatter()`.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        curve = self.evaluate(np.linspace(0, 1, resolution)) 

        ax.plot(curve[:, 0], curve[:, 1], label=label or "Function", **kwargs)


        ax.axis('equal')    

# ====================================================================================================
# Points interpolation Spline
# ====================================================================================================

class PointsSpline(Spline):
    """
    Represents a batch of interpolation splines.
    Compatible with single spline input (N, 3) or batch (B, N, 3).
    An additional fourth ccoordinate w is managed for compatibility with Nurbs
    """
    def __init__(self, points, cyclic=False, already_closed=False):
        """
        Initialize the spline(s) from control points.

        Parameters
        ----------
        points : ndarray of shape (N, 3) or (B, N, 3)
            Control points of the spline(s).
        cyclic : bool
            Whether the spline is cyclic (looped).
        already_closed : bool
            If True, the input points are already closed (i.e., last point = first point),
            and no additional point will be added.
        """
        self.cyclic = cyclic
        points = np.asarray(points)

        if points.ndim == 2:
            if cyclic and not already_closed:
                points = np.concatenate([points, points[:1]], axis=0)  # (N+1, 3)
            self._points = points

        elif points.ndim == 3:
            if cyclic and not already_closed:
                points = np.concatenate([points, points[:, :1]], axis=1)  # (B, N+1, 3)
            self._points = points

        else:
            raise ValueError("Expected shape (N, 3) or (B, N, 3)")

    def __str__(self):
        if self.is_scalar:
            return f"<{type(self).__name__} of {self.num_points} points>"
        else:
            return f"<{type(self).__name__} {len(self)} splines of {self.num_points} points>"

    def __getitem__(self, index):
        raise NotImplementedError(f"{type(self).__name__}.__getitem__ is not implemented")
        
    def __array__(self, dtype=None):
        if dtype is None:
            return self._points
        else:
            return self._points.astype(dtype)
        
    # ====================================================================================================
    # Properties
    # ====================================================================================================

    @property
    def shape(self):
        return self._points.shape[:-1]
    
    @property
    def is_scalar(self):
        return self._points.shape[:-2] == ()
    
    def __len__(self):
        if self.is_scalar:
            raise TypeError(f"<{type(self).__name__}> is scalar and has no length")
        return self._points.shape[0]
        
    @property
    def num_points(self):
        return self._points.shape[-2]

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = np.asarray(value)
        if len(self._points.shape) < 2 or self._points.shape[-1] != 3:
            raise AttributeError("The points must be an array of vectors 3D")

    @property
    def w(self):
        if not hasattr(self, '_w') or self._w is None:
            return np.ones_like(self._points[..., 0])
        else:
            return self._w

    @w.setter
    def w(self, value):
        if value is None:
            self._w = None
        else:
            self._w = np.asarray(value)
            if self._w.shape != self._points.shape[:-1]:
                self._w = np.broadcast_to(self._w, self._points.shape[:-1])

    # ----------------------------------------------------------------------------------------------------
    # Get arrays as array of arrays
    # ----------------------------------------------------------------------------------------------------

    @property
    def _array_of_points(self):
        if self.is_scalar:
            return self._points[None]
        else:
            return self._points

    def _array_of_w(self):
        if self.is_scalar:
            return self.w[None]
        else:
            return self.w
        
    # ====================================================================================================
    # Broadcast the evaluation parameter
    # BT convention
    # ====================================================================================================

    def _broadcast_t(self, t):
        """
        Broadcast parameter t according to (B, T) convention.

        If self is_scalar:
            - t.shape == ()         → result shape: ()
            - t.shape == (T,)       → result shape: (T,)
        
        If self is a batch of B splines:
            - t.shape == ()         → result shape: (B,)
            - t.shape == (T,)       → result shape: (B, T)
            - t.shape == (1, T)     → result shape: (B, T)
            - t.shape == (B, T)     → result shape: (B, T)

        Returns
        -------
        - t: broadcasted array of shape (B, T)
        - res_shape: tuple to reshape result (e.g., (B, T, 3))
        """
        if self.is_scalar:
            B = 1
        else:
            B = len(self)

        t = self._clip_t(t)
        if t.shape == ():
            t = np.broadcast_to(t, (B, 1))
            res_shape = () if self.is_scalar else (B,)
            
        else:
            try:
                t = np.broadcast_to(t, (B, t.shape[-1]))

            except ValueError:
                if self.is_scalar:
                    msg = (f"The shape of t {t.shape} is not compatible with a single spline. "
                          f"Authorized shapes are (1, n) or (n,)")
                else:
                    msg = (f"The shape of t {t.shape} is not compatible with {len(self)} spline. "
                          f"Authorized shapes are (n,) (1, n), or ({len(self)}, n)")

                    raise ValueError(msg)
                
            if self.is_scalar:
                res_shape = (t.shape[-1],)
            else:
                res_shape = t.shape

        return t, res_shape + (3,)
              
# ====================================================================================================
# Poly Spline
# ====================================================================================================

class Poly(PointsSpline):
    """
    Represents a batch of POLY splines.
    Compatible with single spline input (N, 3) or batch (B, N, 3).
    An additional fourth ccoordinate w is managed for compatibility with Nurbs
    """

    CACHE_TANGENTS = False

    def __getitem__(self, index):
        if len(self):
            return Poly(self._points[index], cyclic=self.cyclic, already_closed=True)
        
    # ====================================================================================================
    # Evaluation
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------------------------------------

    def evaluate(self, t):
        """
        Evaluate the spline(s) at normalized parameter(s) t ∈ [0, 1].

        Parameters:
            t: scalar, (T,), (B,), (1,T), or (B,T) depending on convention

        Returns:
            Array of shape:
                - (3,)                  if scalar spline and scalar t
                - (T, 3)                if scalar spline and t.shape == (T,)
                - (B, 3)                if B splines and scalar t
                - (B, 3)                if t.shape == (B,)
                - (B, T, 3)             if t.shape == (B,T) or (1,T)
        """
        coords = self._array_of_points  # (B, N, 3)
        B, N, _ = coords.shape

        t, res_shape = self._broadcast_t(t)  # <- returns t of shape (B, T)
        T = t.shape[1]

        s = np.clip(t * (N - 1), 0, N - 1 - 1e-6)     # (B, T)
        i = np.floor(s).astype(int)                  # (B, T)
        u = s - i                                     # (B, T)

        # Indexing: p0 and p1 shapes: (B, T, 3)
        p0 = coords[np.arange(B)[:, None], i]
        p1 = coords[np.arange(B)[:, None], i + 1]

        res = (1 - u)[..., None] * p0 + u[..., None] * p1  # (B, T, 3)
        return res.reshape(res_shape)

    # ----------------------------------------------------------------------------------------------------
    # Tangent
    # ----------------------------------------------------------------------------------------------------

    def _control_point_tangents(self):
        """
        Compute (or retrieve) the tangent vectors at control points.
        These are the average of incoming and outgoing segments.
        The result is stored in `self._points_tangents`.

        Returns
        -------
        tangents : ndarray of shape (B, N, 3)
            Tangent vectors at each control point.
        """
        if hasattr(self, '_points_tangents'):

            return self._points_tangents

        points = self._array_of_points  # shape (B, N, 3)
        B, N, _ = points.shape
        tangents = np.zeros_like(points)

        # Interior points
        tangents[:, 1:-1] = points[:, 2:] - points[:, :-2]

        if self.cyclic:
            # First point: avg of [P1 - P0] and [Pn-1 - P0]
            tangents[:, 0] = points[:, 1] - points[:, -2]
            # Last point (duplicate of P0): same tangent
            tangents[:, -1] = tangents[:, 0]
        else:
            # Start and end: single-sided
            tangents[:, 0] = points[:, 1] - points[:, 0]
            tangents[:, -1] = points[:, -1] - points[:, -2]

        if self.CACHE_TANGENTS:
            self._points_tangents = tangents

        return tangents


    def tangent(self, t):
        """
        Evaluate the tangent vector(s) at parametric coordinate(s) t by
        interpolating the tangents at control points.

        Parameters
        ----------
        t : float or array-like
            Parametric coordinate(s) in [0, 1].

        Returns
        -------
        tangents : ndarray of shape (B, T, 3)
            Normalized tangent vectors at the given parameter(s).
        """
        tangents_cp = self._control_point_tangents()   # shape (B, N, 3)
        B, N, _ = tangents_cp.shape

        t, res_shape = self._broadcast_t(t)            # t: shape (B, T)
        B_check, T = t.shape
        assert B_check == B

        num_segments = N - 1
        s = t * num_segments                          # shape (B, T)
        s = np.clip(s, 0, num_segments - 1e-6)
        i = np.floor(s).astype(int)                   # (B, T)
        u = s - i                                     # (B, T)

        # Fancy indexing for tangents_cp[range(B), i]
        rows = np.arange(B)[:, None]                  # shape (B, 1)
        Ti = tangents_cp[rows, i]                     # (B, T, 3)
        Ti1 = tangents_cp[rows, i + 1]                # (B, T, 3)

        T_interp = (1 - u[..., None]) * Ti + u[..., None] * Ti1  # (B, T, 3)

        # Normalize
        norm = np.linalg.norm(T_interp, axis=-1, keepdims=True)
        norm[norm < 1e-8] = 1.0
        T_interp /= norm

        return T_interp.reshape(res_shape)

    # ----------------------------------------------------------------------------------------------------
    # Resample
    # ----------------------------------------------------------------------------------------------------

    def resample(self, count):
        """
        Sample evenly spaced points along a smoothed version of the spline(s),
        using a temporary Bezier interpolation.

        Parameters:
            count: int -- number of samples

        Returns:
            self (modified in-place)
        """
        count = count + 1 if self.cyclic else count

        if self._points.shape[-2] != count:
            bezier = Bezier(self._points, cyclic=False)
            t = np.linspace(0, 1, count)
            self._points = bezier.evaluate(t)
        return self

    # ----------------------------------------------------------------------------------------------------
    # Length
    # ----------------------------------------------------------------------------------------------------

    def length(self):
        """
        Compute total arc length of each spline.

        Returns:
            Array of shape (B,)
        """
        coords = self._array_of_points  # (B, N, 3)
        seg = coords[:, 1:] - coords[:, :-1]
        d = np.linalg.norm(seg, axis=-1)  # (B, N-1)
        res = np.sum(d, axis=-1)

        if self.is_scalar:
            return res[0]
        else:
            return res
        
    # ----------------------------------------------------------------------------------------------------
    # Sample an attribute along the splines
    # ----------------------------------------------------------------------------------------------------

    def sample_attribute(self, t, attribute):
        """
        Interpolate attribute values along the spline(s) at parameter t ∈ [0, 1].

        Parameters:
            t : scalar, (T,), (B,), (1, T), or (B, T)
            attribute : array of shape (N,), (B,N), (N,D), or (B,N,D)

        Returns:
            Array of shape:
                - (T,) or (T,D)       if scalar spline
                - (B,) or (B,D)       if t is scalar
                - (B,T) or (B,T,D)    otherwise
        """
        attr = np.asarray(attribute)
        coords = self._array_of_points  # (B, N, 3)
        B, N, _ = coords.shape

        # Format attribute to shape (B, N, D)
        if attr.ndim == 1:
            attr = attr[None, :, None]  # (1, N, 1)
        elif attr.ndim == 2:
            if attr.shape[0] == N:
                attr = attr[None]  # (1, N, D)
            elif attr.shape[1] == N:
                attr = attr[..., None]  # (B, N, 1)
            else:
                raise ValueError(f"Invalid attribute shape {attr.shape}")
        elif attr.ndim == 3:
            if attr.shape[0] != B or attr.shape[1] != N:
                raise ValueError(f"Attribute shape {attr.shape} must be (B,N,D)")
        else:
            raise ValueError("Attribute must be 1D, 2D, or 3D")

        # Add cyclic attribute value
        if self.cyclic:
            attr = np.concatenate([attr, attr[:, :1]], axis=1)  # (B, N+1, D)

        t, res_shape = self._broadcast_t(t)  # t: (B, T), res_shape: (B, T, 3) or (T, 3), etc.
        T = t.shape[-1]
        s = np.clip(t * (attr.shape[1] - 1), 0, attr.shape[1] - 1 - 1e-6)
        i = np.floor(s).astype(int)
        u = s - i

        a0 = attr[np.arange(B)[:, None], i]       # (B, T, D)
        a1 = attr[np.arange(B)[:, None], i + 1]   # (B, T, D)

        res = (1 - u)[..., None] * a0 + u[..., None] * a1  # (B, T, D)
        
        # Adapt output shape to match scalar or batch
        if self.is_scalar:
            return res[0].reshape(res_shape[:-1] + res.shape[2:])
        else:
            return res.reshape(res_shape[:-1] + res.shape[2:])
        
    # ----------------------------------------------------------------------------------------------------
    # Conversion to poly
    # ----------------------------------------------------------------------------------------------------

    def to_poly(self, count=100):
        """
        Approximate this spline by a Poly spline using sampled points.

        Parameters
        ----------
        count : int
            Number of control points in the Poly approximation.

        Returns
        -------
        Poly : a new Poly spline instance
        """
        new_poly = Poly(self._points.copy(), cyclic=self.cyclic, already_closed=True)
        return new_poly.resample(count)
        
    # ----------------------------------------------------------------------------------------------------
    # Conversion to Bezier
    # ----------------------------------------------------------------------------------------------------

    def to_bezier(self, count=10, handle_scale=0.3):
        """
        Approximate a Poly spline with a Bézier spline by sampling `count` points and estimating tangents.

        Parameters
        ----------
        count : int
            Number of Bézier control points to use (≥ 2).
        handle_scale : float
            Scale factor for the distance to handles.

        Returns
        -------
        Bezier : new Bézier spline instance
        """
        count = count + 1 if self.cyclic else count

        coords = self._array_of_points           # shape (B, N, 3)
        B, N, _ = coords.shape
        M = max(2, min(count, N))                # clamp between 2 and N

        indices = np.linspace(0, N - 1, M).round().astype(int)  # shape (M,)
        control_points = coords[:, indices]                   # (B, M, 3)

        # Clamp indices for tangent estimation
        prev_idx = np.clip(indices - 1, 0, N - 1)
        next_idx = np.clip(indices + 1, 0, N - 1)
        prev_points = coords[:, prev_idx]         # (B, M, 3)
        next_points = coords[:, next_idx]         # (B, M, 3)

        tangents = next_points - prev_points
        norm = np.linalg.norm(tangents, axis=-1, keepdims=True)
        tangents /= np.where(norm < 1e-8, 1.0, norm)

        # Handle distance (same for all intermediate points)
        d = np.linalg.norm(np.diff(control_points, axis=1), axis=-1)  # (B, M-1)
        avg_dist = np.concatenate([
            d[:, :1],
            (d[:, :-1] + d[:, 1:]) / 2,
            d[:, -1:]
        ], axis=1)[..., None]  # (B, M, 1)

        offset = handle_scale * avg_dist * tangents  # (B, M, 3)

        right_handles = control_points + offset
        left_handles = control_points - offset

        if self.is_scalar:
            control_points = control_points[0]
            left_handles = left_handles[0]
            right_handles = right_handles[0]

        return Bezier(
            control_points,
            left_handles=left_handles,
            right_handles=right_handles,
            cyclic=self.cyclic,
            already_closed=True
        )

    # ====================================================================================================
    # For tests : plot
    # ====================================================================================================
    
    def _plot(self, resolution=100, display_points='NO', label=None, ax=None, **kwargs):
        """
        Plot the Poly spline using matplotlib.

        Parameters:
            resolution (int): Number of points to evaluate the curve.
            display_points (str): 'NO', 'POINTS', 'HANDLES', or 'ALL'
                                (only 'POINTS' and 'ALL' are used for Poly).
            label (str): Legend label for the spline.
            ax (matplotlib.axes.Axes): Optional matplotlib axis to draw on.
            **kwargs: Additional keyword arguments for `plot()` and `scatter()`.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        curve = self.evaluate(np.linspace(0, 1, resolution))  # (T, 3) or (B, T, 3)

        if curve.ndim == 2:  # Scalar spline
            ax.plot(curve[:, 0], curve[:, 1], label=label or "Poly", **kwargs)
            if display_points in {'POINTS', 'ALL'}:
                pts = self._array_of_points  # (N, 3)
                ax.scatter(pts[:, 0], pts[:, 1], color='red', s=40, label='Control Points', **kwargs)

        else:  # Batched splines
            for b in range(curve.shape[0]):
                pts = self._array_of_points[b]  # (N, 3)
                lbl = f"{label} {b}" if label else f"Poly {b}"
                ax.plot(curve[b, :, 0], curve[b, :, 1], label=lbl, **kwargs)
                if display_points in {'POINTS', 'ALL'}:
                    ax.scatter(pts[:, 0], pts[:, 1], s=30, color='red', **kwargs)

        ax.axis('equal')

# ====================================================================================================
# Bezier splines
# ====================================================================================================

class Bezier(PointsSpline):
    """
    Represents a batch of BEZIER splines.
    """
    def __init__(self, points, left_handles=None, right_handles=None, cyclic=False, already_closed=False):
        """
        Initialize the Bezier spline(s).

        Parameters:
            points: ndarray of shape (N, 3) or (B, N, 3)
            left_handles: optional ndarray of same shape as points
            right_handles: optional ndarray of same shape as points
            cyclic: whether the spline is cyclic (looped)
            already_closed: if True, data is assumed to already include the cyclic closure
        """
        super().__init__(points, cyclic=cyclic, already_closed=already_closed)

        # Helper: extend array with first point if needed
        def _close_array(arr):
            if arr.ndim == 2:
                return np.concatenate([arr, arr[:1]], axis=0)
            elif arr.ndim == 3:
                return np.concatenate([arr, arr[:, :1]], axis=1)
            return arr

        # Prepare handles
        left = np.asarray(left_handles) if left_handles is not None else None
        right = np.asarray(right_handles) if right_handles is not None else None

        if cyclic and not already_closed:
            if left is not None:
                left = _close_array(left)
            if right is not None:
                right = _close_array(right)

        if left is None or right is None:
            # Auto-compute handles if missing
            left_h, right_h = self._compute_bezier_handles(self._points, cyclic=cyclic)
            self._left = left_h if left is None else left
            self._right = right_h if right is None else right
        else:
            self._left = left
            self._right = right

        if self._left.shape != self._points.shape:
            raise ValueError("left_handles must match shape of points")
        if self._right.shape != self._points.shape:
            raise ValueError("right_handles must match shape of points")
        
    def __getitem__(self, index):
        if len(self):
            return Bezier(
                self._points[index], 
                left_handles = self._left[index],
                right_handles = self._right[index],
                cyclic = self.cyclic, 
                already_closed = True,
                )

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def handles(self):
        return self._left, self._right

    @property
    def _array_of_handles(self):
        if self.is_scalar:
            return self._left[None], self._right[None]
        else:
            return self._left, self._right
        
    def copy(self):
        return Bezier(
            self._points.copy(),
            left_handles=self._left.copy(),
            right_handles=self._right.copy(),
            cyclic=self.cyclic,
            already_closed=True
        )

    @staticmethod
    def _compute_bezier_handles(points, cyclic=False):
        """
        Automatically compute left/right handles for smooth Bezier curves.

        Parameters:
            points: ndarray of shape (B, N, 3) or (N, 3)
            cyclic: bool
        Returns:
            left_handles, right_handles
        """
        points = np.asarray(points)
        single = points.ndim == 2
        if single:
            points = points[None]  # shape (1, N, 3)

        B, N, _ = points.shape

        if cyclic:
            ext_points = np.concatenate([points[:, -1:], points, points[:, :1]], axis=1)  # (B, N+2, 3)
        else:
            ext_points = np.pad(points, ((0, 0), (1, 1), (0, 0)), mode='edge')  # duplicate ends

        # Tangents
        der = ext_points[:, 2:] - ext_points[:, :-2]  # shape (B, N, 3)

        # Normalize
        norm = np.linalg.norm(der, axis=-1, keepdims=True)
        norm[norm < 1e-8] = 1
        der /= norm

        # Distances
        dists = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1, keepdims=True)

        # Left handles
        lefts = np.array(points)
        lefts[:, 1:] -= der[:, 1:] * dists / 3
        lefts[:, 0] -= der[:, 0] * dists[:, 0] / 3

        # Right handles
        rights = np.array(points)
        rights[:, :-1] += der[:, :-1] * dists / 3
        rights[:, -1] += der[:, -1] * dists[:, -1] / 3

        if single:
            return lefts[0], rights[0]
        return lefts, rights
    
    # ----------------------------------------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------------------------------------
    
    def evaluate(self, t):
        """
        Evaluate the Bezier spline(s) at normalized parameter(s) t ∈ [0, 1].

        Parameters:
            t: scalar, (T,), (B,), (1,T), or (B,T)

        Returns:
            Array of shape:
                - (3,)                if scalar spline and scalar t
                - (T, 3)              if scalar spline and t.shape == (T,)
                - (B, 3)              if B splines and scalar t
                - (B, 3)              if t.shape == (B,)
                - (B, T, 3)           if t.shape == (1,T) or (B,T)
        """
        points = self._array_of_points        # (B, N, 3)
        left, right = self._array_of_handles  # (B, N, 3), (B, N, 3)
        B, N, _ = points.shape

        t, res_shape = self._broadcast_t(t)   # (B, T)
        T = t.shape[-1]

        s = np.clip(t * (N - 1), 0, N - 1 - 1e-6)   # (B, T)
        i = np.floor(s).astype(int)                # (B, T)
        u = s - i                                   # (B, T)

        # Batch indexing
        batch_idx = np.arange(B)[:, None]          # (B, 1)

        P0 = points[batch_idx, i]                  # (B, T, 3)
        R0 = right[batch_idx, i]                   # (B, T, 3)
        L1 = left[batch_idx, i + 1]                # (B, T, 3)
        P1 = points[batch_idx, i + 1]              # (B, T, 3)

        u2 = u * u
        u3 = u2 * u
        omu = 1 - u
        omu2 = omu * omu
        omu3 = omu2 * omu

        res = (
            omu3[..., None] * P0 +
            3 * omu2[..., None] * u[..., None] * R0 +
            3 * omu[..., None] * u2[..., None] * L1 +
            u3[..., None] * P1
        )

        return res.reshape(res_shape)

    # ----------------------------------------------------------------------------------------------------
    # Tangent
    # ----------------------------------------------------------------------------------------------------

    def tangent(self, t):
        """
        Evaluate unit tangents of the Bezier spline(s) at normalized parameter(s) t ∈ [0, 1].

        Parameters:
            t: scalar, (T,), (1, T), or (B, T)

        Returns:
            Array of shape:
                - (3,)           if scalar spline and scalar t
                - (T, 3)         if scalar spline and t.shape == (T,)
                - (B, 3)         if B splines and scalar t
                - (B, T, 3)      if t.shape == (B, T)
        """
        points = self._array_of_points       # (B, N, 3)
        left, right = self._array_of_handles # (B, N, 3), (B, N, 3)
        B, N, _ = points.shape

        t, res_shape = self._broadcast_t(t)  # → t.shape == (B, T)
        T = t.shape[1]

        s = np.clip(t * (N - 1), 0, N - 1 - 1e-6)      # (B, T)
        i = np.floor(s).astype(int)                   # (B, T)
        u = s - i                                     # (B, T)

        # Batched indexing
        P0 = points[np.arange(B)[:, None], i]         # (B, T, 3)
        R0 = right[np.arange(B)[:, None], i]          # (B, T, 3)
        L1 = left[np.arange(B)[:, None], i + 1]       # (B, T, 3)
        P1 = points[np.arange(B)[:, None], i + 1]     # (B, T, 3)

        u2 = u * u
        omu = 1 - u
        omu2 = omu * omu

        dP = (
            3 * omu2[..., None] * (R0 - P0) +
            6 * omu[..., None] * u[..., None] * (L1 - R0) +
            3 * u2[..., None] * (P1 - L1)
        )  # (B, T, 3)

        norm = np.linalg.norm(dP, axis=-1, keepdims=True)  # (B, T, 1)
        norm[norm < 1e-8] = 1.0

        res = dP / norm  # (B, T, 3)
        return res.reshape(res_shape)
    
    # ----------------------------------------------------------------------------------------------------
    # Resample
    # ----------------------------------------------------------------------------------------------------
    
    def resample(self, count):
        """
        Resample evenly spaced points along the Bezier spline(s).

        Parameters:
            count: int -- number of samples along [0, 1]

        Returns:
            self
        """
        count = count + 1 if self.cyclic else count

        if self._points.shape[-2] != count:
            t = np.linspace(0, 1, count)
            points = self.evaluate(t)
            left, right = self._compute_bezier_handles(points)
            self._points = points
            self._left = left
            self._right = right

        return self
    
    # ----------------------------------------------------------------------------------------------------
    # Length
    # ----------------------------------------------------------------------------------------------------

    def length(self, resolution=100):
        """
        Approximate the arc length of each Bezier spline.

        Parameters:
            resolution: int -- number of samples used to approximate the length

        Returns:
            Array of shape (B,) or a scalar if the spline is scalar
        """
        coords = self.resample(resolution)._array_of_points  # (B, R, 3)
        segs = coords[:, 1:] - coords[:, :-1]  # (B, R-1, 3)
        dists = np.linalg.norm(segs, axis=-1)  # (B, R-1)
        lengths = np.sum(dists, axis=-1)       # (B,)

        if self.is_scalar:
            return lengths[0]
        else:
            return lengths

    # ----------------------------------------------------------------------------------------------------
    # Sample an attribute along the splines
    # ----------------------------------------------------------------------------------------------------

    def sample_attribute(self, t, attribute):
        """
        Interpolate an attribute along the Bezier spline(s) using cubic interpolation.

        Parameters:
            t : scalar, (T,), (B,), (1, T), or (B, T)
            attribute : ndarray of shape (B, N) or (B, N, D)

        Returns:
            Interpolated values:
                - (T,) or (T, D)     if scalar spline
                - (B,) or (B, D)     if scalar t
                - (B, T) or (B, T, D) if t.shape == (B, T)
        """
        attr = np.asarray(attribute)
        points = self._array_of_points  # (B, N, 3)
        B, N, _ = points.shape

        if attr.ndim == 2:
            attr = attr[..., None]  # (B, N, 1)
        elif attr.ndim != 3:
            raise ValueError("Attribute must have shape (B, N) or (B, N, D)")

        if attr.shape[:2] != (B, N):
            raise ValueError(f"Attribute shape {attr.shape} does not match spline shape {(B, N)}")

        if self.cyclic:
            attr = np.concatenate([attr, attr[:, :1]], axis=1)  # (B, N+1, D)

        t, res_shape = self._broadcast_t(t)  # (B, T)
        T = t.shape[1]

        s = np.clip(t * (attr.shape[1] - 1), 0, attr.shape[1] - 1 - 1e-6)
        i = np.floor(s).astype(int)
        u = s - i

        a0 = attr[np.arange(B)[:, None], i]       # (B, T, D)
        h0 = attr[np.arange(B)[:, None], i]       # fake handle
        h1 = attr[np.arange(B)[:, None], i + 1]   # fake handle
        a1 = attr[np.arange(B)[:, None], i + 1]   # (B, T, D)

        u2 = u ** 2
        u3 = u2 * u
        omu = 1 - u
        omu2 = omu ** 2
        omu3 = omu2 * omu

        result = (
            omu3[..., None] * a0 +
            3 * omu2[..., None] * u[..., None] * h0 +
            3 * omu[..., None] * u2[..., None] * h1 +
            u3[..., None] * a1
        )  # (B, T, D)

        if result.shape[-1] == 1:
            result = result[..., 0]  # → (B, T)

        return result.reshape(res_shape[:-1] + result.shape[-1:]) if result.ndim > 2 else result.reshape(res_shape)

    # ----------------------------------------------------------------------------------------------------
    # Conversion to bezier
    # ----------------------------------------------------------------------------------------------------

    def to_bezier(self, count=20, handle_scale=0.3):
        new_bezier = Bezier(
            self._points.copy(),
            left_handles=self._left.copy(),
            right_handles=self._right.copy(),
            cyclic=self.cyclic,
            already_closed=True
        )
        return new_bezier.resample(count)
    
    # ====================================================================================================
    # For tests : plot
    # ====================================================================================================
    
    def _plot(self, resolution=100, display_points='NO', label=None, ax=None, **kwargs):
        """
        Plot the Bezier spline using matplotlib.

        Parameters:
            resolution (int): Number of points to evaluate the curve.
            display_points (str): One of 'NO', 'POINTS', 'HANDLES', 'ALL'
            label (str): Legend label for the spline(s).
            ax (matplotlib.axes.Axes): Optional matplotlib axis to draw on.
            **kwargs: Additional keyword arguments for `plot()`.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        curve = self.evaluate(np.linspace(0, 1, resolution))  # (T, 3) or (B, T, 3)
        anchors = self._array_of_points       # (B, N, 3)
        lefts, rights = self._array_of_handles  # each (B, N, 3)

        if curve.ndim == 2:  # scalar spline
            ax.plot(curve[:, 0], curve[:, 1], label=label or "Bezier", **kwargs)

            if display_points in {'POINTS', 'ALL'}:
                ax.scatter(anchors[0, :, 0], anchors[0, :, 1], c='red', s=40, label='Anchors')

            if display_points in {'HANDLES', 'ALL'}:
                ax.scatter(lefts[0, :, 0], lefts[0, :, 1], c='blue', marker='v', label='Left Handles')
                ax.scatter(rights[0, :, 0], rights[0, :, 1], c='green', marker='^', label='Right Handles')
                for i in range(len(anchors[0]) - 1):
                    ax.plot([anchors[0, i, 0], rights[0, i, 0]], [anchors[0, i, 1], rights[0, i, 1]], 'k--', lw=0.5)
                    ax.plot([anchors[0, i+1, 0], lefts[0, i+1, 0]], [anchors[0, i+1, 1], lefts[0, i+1, 1]], 'k--', lw=0.5)

        else:  # batched splines
            B = curve.shape[0]
            for b in range(B):
                lbl = f"{label} {b}" if label else f"Bezier {b}"
                ax.plot(curve[b, :, 0], curve[b, :, 1], label=lbl, **kwargs)

                if display_points in {'POINTS', 'ALL'}:
                    ax.scatter(anchors[b, :, 0], anchors[b, :, 1], c='red', s=40)

                if display_points in {'HANDLES', 'ALL'}:
                    ax.scatter(lefts[b, :, 0], lefts[b, :, 1], c='blue', marker='v')
                    ax.scatter(rights[b, :, 0], rights[b, :, 1], c='green', marker='^')
                    for i in range(len(anchors[b]) - 1):
                        ax.plot([anchors[b, i, 0], rights[b, i, 0]], [anchors[b, i, 1], rights[b, i, 1]], 'k--', lw=0.5)
                        ax.plot([anchors[b, i+1, 0], lefts[b, i+1, 0]], [anchors[b, i+1, 1], lefts[b, i+1, 1]], 'k--', lw=0.5)

        ax.axis('equal')









# ====================================================================================================
# Tests
# ====================================================================================================



def test_evaluate_spline_shapes():

    print("\n--- Running test_evaluate_spline_shapes ---")

    # Cas scalaire : un seul spline avec 5 points
    points_scalar = np.random.rand(5, 3)
    S_scalar = Poly(points_scalar)
    print(S_scalar)

    # Cas batch : 4 splines avec 5 points chacun
    points_batch = np.random.rand(4, 5, 3)
    S_batch = Poly(points_batch)
    print(S_batch)

    # --- Tests scalaires ---
    out = S_scalar.evaluate(0.5)
    print("scalar t:", out.shape)
    assert out.shape == (3,)

    t = np.linspace(0, 1, 10)
    out = S_scalar.evaluate(t)
    print("t.shape == (T,):", out.shape)
    assert out.shape == (10, 3)

    t = np.linspace(0, 1, 10)
    out = S_scalar.evaluate(t)
    print("t.shape == (T,1):", out.shape)
    assert out.shape == (10, 3)

    # --- Tests batch ---
    out = S_batch.evaluate(0.5)
    print("scalar t with batch:", out.shape)
    assert out.shape == (4, 3)

    t = np.linspace(0, 1, 10)
    try:
        S_batch.evaluate(t)  # devrait échouer car ambigu (shape = (10,) sans batch info)
    except ValueError as e:
        print("Expected failure for shape (T,) with batch:", e)

    t = np.linspace(0, 1, 10)
    out = S_batch.evaluate(t)
    print("t.shape == (T,1):", out.shape)
    assert out.shape == (4, 10, 3)

    t = np.tile(np.linspace(0, 1, 10), (4, 1))  # (T, B)
    out = S_batch.evaluate(t)
    print("t.shape == (T,B):", out.shape)
    assert out.shape == (4, 10, 3)

    print("✅ All shape assertions passed.")


def test_spline_evaluate_plot():
    
    import matplotlib.pyplot as plt

    # Poly scalaire : une sinusoïde simple
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.sin(x)
    points_scalar = np.stack([x, y, np.zeros_like(x)], axis=-1)
    spline_scalar = Poly(points_scalar)
    spline_scalar.resample(50)

    # Poly batchée : 4 sinusoïdes avec décalages de phase
    B = 4
    offsets = np.linspace(0, np.pi, B, endpoint=False)
    points_batch = np.stack([
        np.stack([
            x,
            np.sin(x + offset),
            np.full_like(x, offset)
        ], axis=-1)
        for offset in offsets
    ], axis=0)
    spline_batch = Poly(points_batch)
    spline_batch.resample(50)

    # Évaluation des splines
    t = np.linspace(0, 1, 100)
    eval_scalar = spline_scalar.evaluate(t)  # (100, 3)
    eval_batch = spline_batch.evaluate(t)    # (4, 100, 3)

    # Tracé
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(eval_scalar[:, 0], eval_scalar[:, 1], label='Scalar Poly')
    axs[0].set_title("Poly scalaire")
    axs[0].axis("equal")
    axs[0].legend()

    for i in range(B):
        axs[1].plot(eval_batch[i, :, 0], eval_batch[i, :, 1], label=f"Poly {i}")
    axs[1].set_title("Batch de 4 splines")
    axs[1].axis("equal")
    axs[1].legend()

    plt.show()

def test_spline_all_t_shapes():

    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 10)
    T = 50
    B = 3
    t_values = np.linspace(0, 1, T)

    # 1. Scalar spline
    y = np.sin(x)
    points_scalar = np.stack([x, y, np.zeros_like(x)], axis=-1)
    spline_scalar = Poly(points_scalar)
    spline_scalar.resample(100)  # Smooth using temporary Bezier

    t_cases_scalar = {
        "scalar": 0.5,
        "(T,)": t_values,
        "(1, T)": t_values[None, :]
    }

    # 2. Batched splines
    offsets = np.linspace(0, np.pi, B, endpoint=False)
    points_batch = np.stack([
        np.stack([
            x,
            np.sin(x + offset),
            np.full_like(x, offset)
        ], axis=-1)
        for offset in offsets
    ], axis=0)
    spline_batch = Poly(points_batch)
    spline_batch.resample(100)

    t_cases_batch = {
        "scalar": 0.5,
        "(T,)": t_values,
        "(1, T)": t_values[None, :],
        "(B, T)": np.tile(t_values, (B, 1))
    }

    cols_scalar = len(t_cases_scalar)
    cols_batch = len(t_cases_batch)
    num_cols = max(cols_scalar, cols_batch)

    fig, axs = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10))

    # --- Scalar spline plots ---
    for j, (label, t) in enumerate(t_cases_scalar.items()):
        points = spline_scalar.evaluate(t)     # (T, 3) or (3,)
        tangents = spline_scalar.tangent(t)    # same shape

        ax = axs[0, j]
        if points.ndim == 1:  # (3,)
            x, y = points[0], points[1]
            dx, dy = tangents[0], tangents[1]
            ax.scatter(x, y, label="Poly")
            ax.quiver(x, y, dx, dy, angles='xy', scale=20)
        else:  # (T, 3)
            x, y = points[:, 0], points[:, 1]
            dx, dy = tangents[:, 0], tangents[:, 1]
            ax.plot(x, y, label="Poly")
            ax.quiver(x, y, dx, dy, angles='xy', scale=20)

        ax.set_title(f"Scalar spline, t = {label}")
        ax.axis('equal')
        ax.legend()

    # --- Batch spline plots ---
    for j, (label, t) in enumerate(t_cases_batch.items()):
        points = spline_batch.evaluate(t)      # (B, T, 3) or (B, 3)
        tangents = spline_batch.tangent(t)     # same

        ax = axs[1, j]
        if points.ndim == 2:  # (B, 3)
            for b in range(B):
                x, y = points[b, 0], points[b, 1]
                dx, dy = tangents[b, 0], tangents[b, 1]
                ax.scatter(x, y, label=f"Poly {b}")
                ax.quiver(x, y, dx, dy, angles='xy', scale=20)
        elif points.ndim == 3:  # (B, T, 3)
            for b in range(B):
                x, y = points[b, :, 0], points[b, :, 1]
                dx, dy = tangents[b, :, 0], tangents[b, :, 1]
                ax.plot(x, y, label=f"Poly {b}")
                ax.quiver(x, y, dx, dy, angles='xy', scale=20)
        else:
            raise RuntimeError(f"Unexpected shape: {points.shape}")

        ax.set_title(f"Batch spline, t = {label}")
        ax.axis('equal')
        ax.legend()

    plt.tight_layout()
    plt.show()

def test_to_bezier_fit():
    import matplotlib.pyplot as plt

    # Create a sinuous Poly spline
    x = np.linspace(0, 5 * np.pi, 60)
    y = np.sin(x)
    points = np.stack([x, y, np.zeros_like(x)], axis=-1)
    poly = Poly(points)

    # Convert to Bezier using sample-based approximation
    bezier = poly.to_bezier(reduction=0.3, handle_scale=0.3)

    # Prepare plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw both splines

    poly._plot(resolution=200, display_points='NO', ax=ax, label='Poly', color='blue')
    bezier._plot(resolution=200, display_points='POINTS', ax=ax, label='Bezier', linestyle='--', color='orange')

    # Draw dashed handle lines
    anchors = bezier._array_of_points       # (1, N, 3)
    lefts, rights = bezier._array_of_handles  # (1, N, 3), (1, N, 3)

    a = anchors[0]
    l = lefts[0]
    r = rights[0]

    for i in range(len(a)):
        ax.plot([a[i, 0], l[i, 0]], [a[i, 1], l[i, 1]], 'k--', lw=0.5)
        ax.plot([a[i, 0], r[i, 0]], [a[i, 1], r[i, 1]], 'k--', lw=0.5)

    ax.set_title("Conversion Poly → Bézier (sampled)")
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

def test_to_bezier_fit_closed():
    import matplotlib.pyplot as plt
    import numpy as np

    # Crée un cercle avec 60 points également espacés (fermé)
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    points = np.stack([x, y, np.zeros_like(x)], axis=-1)
    poly = Poly(points, cyclic=True)

    # Approximation en Bézier par échantillonnage
    bezier = poly.to_bezier(reduction=0.25, handle_scale=0.4)

    # Préparer le tracé
    fig, ax = plt.subplots(figsize=(6, 6))

    # Tracer la courbe Poly et la Bézier
    poly._plot(resolution=300, display_points='NO', ax=ax, label='Poly', color='blue')
    bezier._plot(resolution=300, display_points='POINTS', ax=ax, label='Bezier', linestyle='--', color='orange')

    # Récupérer les points et poignées
    anchors = bezier._array_of_points[0]
    lefts, rights = bezier._array_of_handles
    l = lefts[0]
    r = rights[0]

    # Tracer les poignées
    for i in range(len(anchors)):
        ax.plot([anchors[i, 0], l[i, 0]], [anchors[i, 1], l[i, 1]], 'k--', lw=0.5)
        ax.plot([anchors[i, 0], r[i, 0]], [anchors[i, 1], r[i, 1]], 'k--', lw=0.5)

    ax.set_title("Approximation Bézier d’un cercle fermé")
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

def test_circle_splinefunction():
    import matplotlib.pyplot as plt

    # Create a circle spline
    circle = SplineFunction.circle(radius=1.5)

    # Plot the curve
    fig, ax = plt.subplots(figsize=(6, 6))
    circle._plot(resolution=300, ax=ax, label="Circle", color='blue')

    # Plot tangents
    t = np.linspace(0, 1, 12)
    pts = circle.evaluate(t)
    tan = circle.tangent(t)
    ax.quiver(pts[:, 0], pts[:, 1], tan[:, 0], tan[:, 1], angles='xy', scale=20, color='orange', label='Tangents')

    ax.set_title("Circle from SplineFunction")
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

def test_spiral_splinefunction_3d():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
    import numpy as np

    # Create a 3D spiral spline
    spiral = SplineFunction.spiral(
        radius0=0.5, radius1=2.0,
        angle0=0, angle1=6 * np.pi,
        z0=0.0, z1=5.0
    )

    # Evaluation
    t = np.linspace(0, 1, 200)
    curve = spiral.evaluate(t)
    tangents = spiral.tangent(np.linspace(0, 1, 20))
    points = spiral.evaluate(np.linspace(0, 1, 20))

    # Plot 3D spiral and tangents
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label='Spiral', color='green', linewidth=2)

    ax.quiver(
        points[:, 0], points[:, 1], points[:, 2],
        tangents[:, 0], tangents[:, 1], tangents[:, 2],
        length=0.3, normalize=True, color='red', label='Tangents'
    )

    ax.set_title("3D Spiral from SplineFunction")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def plot_tangents(spline, count=10, scale=0.1, ax=None, title="Spline Tangents (2D)"):
    """
    Plot 2D spline(s) with their normalized tangents.

    Parameters
    ----------
    spline : Spline
        Any spline object with .evaluate(t) and .tangent(t)
    count : int
        Number of evaluation points per spline
    scale : float
        Scale of tangent arrows
    ax : matplotlib.axes.Axes or None
        Axis to draw on. If None, a new figure is created.
    """

    import matplotlib.pyplot as plt

    t = np.linspace(0, 1, count)
    points = spline.evaluate(t)      # (..., 3)
    tangents = spline.tangent(t)     # (..., 3)

    # Drop z component if present
    points = np.asarray(points)[..., :2]
    tangents = np.asarray(tangents)[..., :2]

    # Reshape to (B, T, 2)
    if points.ndim == 2:
        points = points[None]    # (1, T, 2)
        tangents = tangents[None]
    elif points.ndim != 3:
        raise ValueError("Unexpected shape for evaluated points")

    B, T, _ = points.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for b in range(B):
        P = points[b]
        Tn = tangents[b]

        ax.plot(P[:, 0], P[:, 1], 'k--')

        for p, t in zip(P, Tn):
            ax.arrow(
                p[0], p[1], scale * t[0], scale * t[1],
                head_width=scale * 0.2, head_length=scale * 0.3, fc='r', ec='r'
            )

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True)
    plt.show()



def test_conversion_pipeline():

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

    bezier_count = 4
    poly_count = 100

    # Étape 1 : créer un cercle comme SplineFunction
    sf = SplineFunction.circle(radius=1.0)

    def func(t):
        x = t*np.pi*2
        y = np.sin(x)
        z = np.zeros_like(x)
        return np.stack((x, y, z), axis=-1)
    
    #sf = SplineFunction(func=func, cyclic=False)

    # Étapes de conversion
    bez1 = sf.to_bezier(count=bezier_count, handle_scale=0.05)
    poly1 = sf.to_poly(count=poly_count)
    poly2 = bez1.to_poly(count=poly_count)
    bez2 = poly1.to_bezier(count=bezier_count)

    # Setup plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    def draw_bezier_handles(bezier, ax, color_anchor='red', color_lh='blue', color_rh='green'):
        anchors = bezier._array_of_points
        lefts, rights = bezier._array_of_handles
        a, l, r = anchors[0], lefts[0], rights[0]

        ax.scatter(a[:, 0], a[:, 1], c=color_anchor, s=20, label='Anchors')
        ax.scatter(l[:, 0], l[:, 1], c=color_lh, marker='v', label='Left handles')
        ax.scatter(r[:, 0], r[:, 1], c=color_rh, marker='^', label='Right handles')

        for i in range(len(a)):
            ax.plot([a[i, 0], l[i, 0]], [a[i, 1], l[i, 1]], 'k--', lw=0.5)
            ax.plot([a[i, 0], r[i, 0]], [a[i, 1], r[i, 1]], 'k--', lw=0.5)

    # Plot 1: SplineFunction
    sf._plot(ax=axes[0], resolution=300, label="SplineFunction", color='black', linestyle='--')
    axes[0].set_title("SplineFunction")

    # Plot 2: Bézier from function
    bez1._plot(ax=axes[1], resolution=300, label="Bezier (from Function)", color='orange')
    draw_bezier_handles(bez1, axes[1])
    axes[1].set_title("Bezier (from Function)")

    # Plot 3: Poly from function
    poly1._plot(ax=axes[2], resolution=300, label="Poly (from Function)", color='blue')
    axes[2].scatter(poly1._points[:, 0], poly1._points[:, 1], c='black', s=10, label='Control Points')
    axes[2].set_title("Poly (from Function)")

    # Plot 4: Poly from bezier
    poly2._plot(ax=axes[3], resolution=300, label="Poly (from Bezier)", color='green')
    axes[3].scatter(poly2._points[:, 0], poly2._points[:, 1], c='black', s=10, label='Control Points')
    axes[3].set_title(f"Poly (from Bezier) {poly2.shape=}")

    # Plot 5: Bezier from poly
    bez2._plot(ax=axes[4], resolution=300, label="Bezier (from Poly)", color='red')
    draw_bezier_handles(bez2, axes[4])
    axes[4].set_title("Bezier (from Poly)")

    # Final cleanup
    for ax in axes:
        ax.axis('equal')
        ax.legend()

    fig.suptitle("Conversion Pipeline: SplineFunction ↔ Bézier ↔ Poly", fontsize=16)
    plt.tight_layout()
    plt.show()










# Run all
if __name__ == "__main__":
    #test_evaluate_spline_shapes()
    #test_spline_evaluate_plot()
    #test_spline_all_t_shapes()
    #test_to_bezier_fit()
    #test_to_bezier_fit_closed()
    #test_circle_splinefunction()
    #test_spiral_splinefunction_3d()
    test_conversion_pipeline()







