# ==========================================================================================
# npblender.geometry.spline
# ==========================================================================================
# Part of the npblender package — https://github.com/...
# MIT License — Created on 11/11/2022 by Alain Bernard — Last updated on 02/08/2025
#
# This module defines a generic spline system with support for:
# - Scalar or batched curves
# - Multiple types (POLY, Bézier, Function-based)
# - Cyclic curves and curve conversion
# - Tangent/length/attribute sampling and plotting
#
# Core classes:
#   - Spline (abstract base class)
#   - FunctionSpline (function-defined splines, no control points)
#   - PointsSpline (base for splines with control points)
#   - PolySpline (piecewise linear splines)
#   - BezierSpline (cubic Bézier splines with handle management)
#
# Curve evaluation and sampling follow a consistent interface:
#   evaluate(t), tangent(t), length(), resample(n), sample_attribute(t, attr), to_poly(), to_bezier()
#
# All spline classes support both single curves (shape (N, 3)) and batches (shape (B, N, 3)),
# with optional cyclic closure.
#
# Dependencies:
#   - numpy
#   - matplotlib (optional, for ._plot())
#
# ==========================================================================================

# Spline (abstract)
# ├── FunctionSpline            # function & erivative
# └── PointsSpline (abstract)   # interpolation
#     ├── PolySpline
#     └── BezierSpline
#     └── Nurbs (not implemented yet)

# Public API of the spline module
__all__ = ['FunctionSpline', 'PolySpline', 'BezierSpline']

import numpy as np

# ====================================================================================================
# Spline
# ====================================================================================================

class Spline:
    """
    Abstract base class for spline curves (scalar or batched).

    This class defines the core interface and common behavior for any type of spline,
    including polygonal, Bézier, NURBS, or procedural (function-defined) curves.

    Subclasses must implement the following methods:
        - evaluate(t): compute points on the curve at parameter t ∈ [0, 1]
        - tangent(t): compute tangent vectors at parameter t
        - length(): return total arc length of each spline
        - sample_attribute(t, attribute): interpolate an attribute along the curve

    Shapes and batching conventions:
        - Scalar spline: points have shape (N, 3), evaluation returns (T, 3) or (3,)
        - Batched splines: points have shape (B, N, 3), evaluation returns (B, T, 3), etc.
        - Parameter t can be scalar, 1D, or 2D, and is broadcasted to shape (B, T)

    Key attributes:
        - cyclic: whether the curve is looped (i.e., P[-1] == P[0])
        - is_scalar: True if the instance represents a single spline

    Subclasses may extend:
        - Conversion methods like to_poly(), to_bezier(), etc.
        - Plotting utilities for visualization and debugging
        - Caching of derived data like tangents, resampled points, etc.

    This class is not meant to be instantiated directly.
    """

    curve_type = None

    def __init__(self, cyclic=True):
        self.cyclic = cyclic
        self.is_scalar = None

    # ----------------------------------------------------------------------------------------------------
    # To be overloaded
    # ----------------------------------------------------------------------------------------------------

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

    def tangent(self, t, normalize=False):
        """
        Approximate the tangent vector(s) of the spline at parameter t using central difference.

        This is the default fallback implementation, valid for any subclass that
        defines `evaluate()`. It estimates γ′(t) ≈ (γ(t + δ) - γ(t - δ)) / (2δ).

        Parameters:
            t : scalar, (T,), (1, T), or (B, T)
                Parameter(s) at which to evaluate the tangent(s).
            normalize : bool, default False
                If True, return unit tangent vectors.

        Returns:
            Tangent vector(s) of shape:
                - (T, 3)       for scalar spline
                - (B, T, 3)    for batch of splines
        """
        dt = 1e-3
        t, res_shape = self._broadcast_t(t)  # (B, T)

        v0 = self.evaluate(t - dt)           # (B, T, 3)
        v1 = self.evaluate(t + dt)           # (B, T, 3)
        d = (v1 - v0) / (2 * dt)             # (B, T, 3)

        if normalize:
            norm = np.linalg.norm(d, axis=-1, keepdims=True)
            d = np.divide(d, norm, where=(norm > 1e-12), out=np.zeros_like(d))

        return d.reshape(res_shape)
    
    # ----------------------------------------------------------------------------------------------------
    # Sample attribute
    # ----------------------------------------------------------------------------------------------------

    def sample_attribute(self, t, attribute):
        raise NotImplementedError("Sample attribute not implemented.")

    # ----------------------------------------------------------------------------------------------------
    # Conversion to poly
    # ----------------------------------------------------------------------------------------------------

    def to_poly(self, count=100):
        """
        Approximate this spline by a PolySpline spline using sampled points.

        Parameters
        ----------
        count : int
            Number of control points in the PolySpline approximation.

        Returns
        -------
        PolySpline : a new PolySpline spline instance
        """
        t = np.linspace(0, 1, count, endpoint=not self.cyclic)
        return PolySpline(self.evaluate(t), cyclic=self.cyclic, already_closed=False)
    
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
        BezierSpline : a new Bézier spline instance
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

        return BezierSpline(
            anchors,
            left_handles=left_handles,
            right_handles=right_handles,
            cyclic=self.cyclic,
            already_closed=True  # anchors already closed if needed
        )
    
# ====================================================================================================
# Spline Function
# ====================================================================================================

class FunctionSpline(Spline):
    """
    Represents a spline curve defined by a parametric function.

    Unlike control-point-based splines (PolySpline, BezierSpline, Nurbs), a FunctionSpline defines its geometry
    through analytical expressions or callable functions for position and optionally for tangent,
    arc length, or other derivatives.

    This class supports scalar and batched curves:
        - Scalar: a single function defines a single spline
        - Batched: multiple functions can be evaluated in parallel (vectorized over batch dimension)

    Expected interface for subclasses or function providers:
        - f(t): evaluates the point at parameter t ∈ [0, 1]
        - df(t): (optional) evaluates the derivative (tangent vector) at t
        - length(): (optional) returns the total arc length
        - sample_attribute(t, attribute): (optional) interpolates additional attributes

    Parameters
    ----------
    - t : float, array-like
        Evaluation parameter(s) in [0, 1]. Can be scalar, 1D or 2D and will be broadcasted
        according to (B, T) convention.

    Returns
    -------
    Depending on the method:
        - evaluate(t) → points of shape (3,), (T, 3), (B, 3), or (B, T, 3)
        - tangent(t) → unit vectors of same shape
        - length() → float or (B,) array
        - sample_attribute(...) → interpolated values

    Notes
    -----
    This class is ideal for mathematical or procedural curves, such as:
        - Circles, spirals, helices
        - Noise-based or parametric shapes
        - Custom splines not tied to discrete control points

    Subclasses or users must override `evaluate()` at minimum.
    """

    curve_type = 'FUNCTION'

    def __init__(self, func, derivative=None, cyclic=True):
        super().__init__(cyclic)
        self.is_scalar = True

        self._func = func
        self._derivative = derivative

    def __str__(self):
        return f"<FunctionSpline {self._func.__name__}, cyclic: {self.cyclic}>"
    
    def copy(self):
        return FunctionSpline(
            self._func,
            self._derivative,
            cyclic=self.cyclic,
        )
    
    def __len__(self):
        raise Exception(f"FunctionSpline is scalar.")

    def evaluate(self, t):
        return self._func(self._clip_t(t))
    
    def tangent(self, t, normalize=False):
        """
        Compute the tangent vector γ′(t) of the scalar FunctionSpline.

        If an analytical derivative is defined, it is used.
        Otherwise, fallback to the default central difference approximation.

        Parameters:
            t : scalar or array-like of shape (T,)
                Parameter(s) in [0, 1] at which to evaluate the tangent(s).
            normalize : bool, default False
                If True, return unit tangent vectors.

        Returns:
            Tangent vector(s) of shape (T, 3) or (3,) if t is scalar.
        """
        if self._derivative is None:
            return super().tangent(t, normalize=normalize)

        t = self._clip_t(t)
        d = self._derivative(t)  # (T, 3) or (3,)

        if normalize:
            d = np.asarray(d)
            norm = np.linalg.norm(d, axis=-1, keepdims=True)
            d = np.divide(d, norm, where=(norm > 1e-12), out=np.zeros_like(d))

        return d

        
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
        Plot the PolySpline spline using matplotlib.

        Parameters:
            resolution (int): Number of points to evaluate the curve.
            display_points (str): 'NO', 'POINTS', 'HANDLES', or 'ALL'
                                (only 'POINTS' and 'ALL' are used for PolySpline).
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
    Abstract base class for splines defined by control points.

    This class provides common infrastructure for spline types such as PolySpline, BezierSpline, or Nurbs,
    which rely on a set of control points (and optionally other geometry like handles or weights).
    It handles batching, cyclic closure, and unified access to the `_points` array.

    Features
    --------
    - Supports scalar splines (shape: (N, 3)) and batched splines (shape: (B, N, 3))
    - Supports optional fourth coordinate (w) for compatibility with rational splines (e.g., NURBS)
    - Manages cyclic splines by optionally appending the first point at the end
    - Provides helper methods for shape normalization, access, slicing, and serialization

    Parameters
    ----------
    points : ndarray
        Array of shape (N, 3/4) or (B, N, 3/4) defining the control points.
    cyclic : bool
        If True, the spline is considered cyclic (looped).
    already_closed : bool
        If True, the first point is assumed to already be duplicated at the end for cyclic curves.

    Attributes
    ----------
    _points : ndarray
        Internal representation of the control points (with optional closure).
    is_scalar : bool
        True if the spline is a scalar curve (not batched).
    cyclic : bool
        Whether the curve is cyclic (closed).
    dimension : int
        Dimensionality of points (3 or 4).

    Expected to Implement
    ---------------------
    Subclasses must implement:
        - evaluate(t)
        - tangent(t)
        - length()
        - sample_attribute(t, attribute)
    
    Examples
    --------
    >>> class MySpline(PointsSpline):
    ...     def evaluate(self, t):
    ...         return self._points[0]  # Dummy implementation

    >>> spline = MySpline(np.random.rand(10, 3), cyclic=True)
    >>> spline._array_of_points.shape
    (1, 11, 3)  # Includes closure
    """

    def __init__(self, points, cyclic=False, already_closed=False, **attributes):
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
        attributes (dict): struct array or array of floats
            Attributes along the splines
        """
        super().__init__(cyclic)
        points = np.asarray(points)

        # ----------------------------------------------------------------------------------------------------
        # points
        # ----------------------------------------------------------------------------------------------------

        if points.ndim == 2:
            self.is_scalar = True
            if cyclic and not already_closed:
                points = np.concatenate([points, points[:1]], axis=0)  # (N+1, 3)
            self._points = points[None]

        elif points.ndim == 3:
            self.is_scalar = False
            if cyclic and not already_closed:
                points = np.concatenate([points, points[:, :1]], axis=1)  # (B, N+1, 3)
            self._points = points

        else:
            raise ValueError("Expected shape (N, 3) or (B, N, 3)")
        
        # ----------------------------------------------------------------------------------------------------
        # attributes
        # ----------------------------------------------------------------------------------------------------

        self._attributes = None
        if attributes:

            B, N, _ = self._points.shape
            fields = []
            arrays = []

            for name, value in attributes.items():
                arr = np.asarray(value, dtype=np.float32)

                msg = None

                # One dim, must be (N,)
                if arr.ndim == 1:
                    try:
                        arr = np.broadcast_to(arr, (1, N, 1))
                    except Exception as e:
                        msg = str(e)
                
                # Two dims, (N, ?) if scalar, (B, N) otherwise (no ambiguity is accepted :-)
                elif arr.ndim == 2:
                    # scalar -> need N vectors
                    if self.is_scalar:
                        try:
                            arr = np.broadcast_to(arr, (1, N) + (arr.shape[-1],))
                        except Exception as e:
                            msg = str(e)

                    # not scalar -> (B, N) scalars
                    else:
                        try:
                            arr = np.broadcast_to(arr, (B, N, 1))
                        except:
                            msg = str(e)
                
                # Three dims (B, N, D)
                elif arr.ndim == 3:
                    try:
                        arr = np.broadcast_to(arr, (B, N, arr.shape[-1]))
                    except:
                        msg = str(e)

                else:
                    msg = f"Attribute shape {arr.shape} is not valid"
                    


                arr = np.broadcast_to(arr, ())

                # Accept (N,) → (1, N, 1), (B, N) → (B, N, 1), etc.
                if arr.ndim == 1 and arr.shape[0] == N:
                    arr = arr[None, :, None]
                elif arr.ndim == 2:
                    if arr.shape == (B, N):
                        arr = arr[..., None]
                    elif arr.shape == (N, 1):
                        arr = arr[None, ...]
                    else:
                        raise ValueError(f"Attribute '{name}' must have shape (B,N) or (B,N,D)")
                elif arr.ndim == 3:
                    if arr.shape[:2] != (B, N):
                        raise ValueError(f"Attribute '{name}' shape mismatch with points: {arr.shape} vs {(B, N)}")
                else:
                    raise ValueError(f"Attribute '{name}' must be of shape (B,N) or (B,N,D)")

                D = () if arr.shape[-1] == 1 else (arr.shape[-1],)
                fields.append((name, 'f4', D))
                arrays.append(arr if D else arr[..., 0])  # collapse last dim if scalar

            dtype = np.dtype(fields)
            result = np.empty((B, N), dtype=dtype)
            for i, name in enumerate(dtype.names):
                result[name] = arrays[i]        

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
    # Get back the geometry
    # ====================================================================================================

    def get_geometry(self):
        """
        Return the raw geometry data of the spline, with shape and structure
        adapted for user-level access (e.g. saving, exporting, etc).

        - Points and handles are returned as (N, D) or (B, N, D)
        - Attributes are returned as structured arrays with named fields
        - Cyclic closing point is removed if applicable

        Returns
        -------
        dict : {
            'points': ndarray,
            'left_handles': ndarray (if present),
            'right_handles': ndarray (if present),
            'attributes': structured ndarray (if present)
        }
        """
        def strip(arr):
            return arr[..., :-1, :] if self.cyclic else arr

        geom = {}

        # Points
        pts = strip(self._points)
        geom["points"] = pts[0] if self.is_scalar else pts

        # Optional handles
        if hasattr(self, '_left_handles'):
            L = strip(self._left_handles)
            geom["left_handles"] = L[0] if self.is_scalar else L
        if hasattr(self, '_right_handles'):
            R = strip(self._right_handles)
            geom["right_handles"] = R[0] if self.is_scalar else R

        # Attributes (structured array)
        if self._attributes is not None:
            A = self._attributes[:, :-1] if self.cyclic else self._attributes
            geom["attributes"] = A[0] if self.is_scalar else A

        return geom

        
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
    def control_points(self):
        if self.is_cyclic:
            return self._points[..., :-1, :]
        else:
            return self._points

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
    # Ensure attribute to sample has the proper shape
    # ====================================================================================================
    
    def _adjust_attributes(self, attribute):
        """
        Ensure attribute has correct length, adding cyclic closing point if needed.

        Parameters
        ----------
        attribute : ndarray
            Array of shape (B, N), (B, N, D), (N,), (N, D), or structured array of shape (B, N) or (N,)

        Returns
        -------
        attr : ndarray
            Possibly extended array with duplicated first point (if cyclic and not already closed),
            and shape compatible with internal usage.
        """
        attr = np.asarray(attribute)
        n_pts = self._points.shape[0 if self.is_scalar else 1]
        n_attr = attr.shape[0 if self.is_scalar else 1]

        # Check and adjust
        if self.cyclic:
            if n_attr == n_pts:
                return attr  # already closed
            elif n_attr == n_pts - 1:
                if attr.dtype.fields is None:
                    # non-structured → repeat first entry
                    if self.is_scalar:
                        return np.concatenate([attr, attr[:1]], axis=0)
                    else:
                        return np.concatenate([attr, attr[:, :1]], axis=1)
                else:
                    # structured → same
                    first = attr[:1] if self.is_scalar else attr[:, :1]
                    return np.concatenate([attr, first], axis=0 if self.is_scalar else 1)
            else:
                raise ValueError(
                    f"Cyclic spline: attribute.shape[1] = {n_attr}, expected {n_pts-1} or {n_pts}"
                    if not self.is_scalar else
                    f"Cyclic spline: attribute.shape[0] = {n_attr}, expected {n_pts-1} or {n_pts}"
                )
        else:
            if n_attr != n_pts:
                raise ValueError(
                    f"Non-cyclic spline: attribute.shape mismatch with points: got {n_attr}, expected {n_pts}"
                )
            return attr
        
    # ====================================================================================================
    # Sample attribute
    # ====================================================================================================

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
        attr = self._adjust_attributes(attribute)
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

              
# ====================================================================================================
# PolySpline Spline
# ====================================================================================================

class PolySpline(PointsSpline):
    """
    Represents a batch of piecewise-linear (POLY) splines.

    A PolySpline spline is a simple curve defined by linear interpolation between control points.
    It is commonly used for rough approximations, debugging, or for converting more complex
    curves into a piecewise-linear form. This class supports both scalar (single) and batched
    splines, as well as optional cyclic closure.

    Features
    --------
    - Supports 3D or 4D points (the 4th coordinate can store weights for NURBS compatibility)
    - Supports cyclic splines with automatic closure (duplicating the first point at the end)
    - Efficient evaluation by linear interpolation between control points
    - Tangents are computed per control point by averaging surrounding segments
    - Can convert to and from Bezier splines for smooth approximations
    - Attribute sampling along the curve is supported (e.g., radius, color, weights)

    Parameters
    ----------
    points : ndarray
        Control points of shape (N, 3/4) or (B, N, 3/4)
    cyclic : bool
        If True, the curve is treated as cyclic (looped).
    already_closed : bool
        If True, the curve is assumed to already include a duplicate of the first point at the end.

    Methods
    -------
    evaluate(t)
        Evaluate the point(s) on the curve at normalized parameter(s) t ∈ [0, 1].
    tangent(t)
        Evaluate the unit tangent vector(s) at parameter(s) t.
    length()
        Compute the total arc length of each curve.
    resample(count)
        Resample the spline to contain `count` evenly spaced control points.
    sample_attribute(t, attribute)
        Interpolate an attribute array along the curve using linear interpolation.
    to_bezier(count, handle_scale)
        Convert the poly spline to a Bezier spline with `count` control points.
    to_poly(count)
        Return a resampled copy of the curve with `count` control points.
    _plot(...)
        Visualize the curve using matplotlib (for debugging/testing).

    Examples
    --------
    >>> pts = np.random.rand(10, 3)
    >>> curve = PolySpline(pts)
    >>> p = curve.evaluate(0.5)       # Evaluate midpoint
    >>> t = curve.tangent(0.3)        # Tangent vector at t = 0.3
    >>> b = curve.to_bezier(6)        # Smooth Bezier approximation

    Notes
    -----
    - When cyclic=True, the first point is appended at the end to enable continuity.
    - Tangents at control points are cached if `CACHE_TANGENTS = True`.
    - The fourth coordinate (w) is preserved for compatibility with NURBS but unused in evaluation.
    """

    curve_type = 'POLY'

    CACHE_TANGENTS = False

    def __getitem__(self, index):
        if len(self):
            return PolySpline(self._points[index], cyclic=self.cyclic, already_closed=True)
        
    def copy(self):
        return PolySpline(
            self._points.copy(),
            cyclic=self.cyclic,
            already_closed=True
        )
        
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

    def tangent(self, t, normalize=False):
        """
        Compute the (piecewise constant) tangent vector(s) of a PolySpline at parameter t.

        For each segment, the tangent is simply the vector from one control point
        to the next. The tangent is constant within each segment.

        Parameters:
            t : scalar, (T,), (1, T), or (B, T)
                Parameter(s) at which to evaluate the tangent(s).
            normalize : bool, default False
                Whether to return unit vectors.

        Returns:
            Tangent vector(s) of shape:
                - (T, 3)       for scalar spline
                - (B, T, 3)    for batch of splines
        """
        points = self._array_of_points  # (B, N, 3)
        B, N, _ = points.shape

        # Compute segment vectors with modular wrapping
        indices = (np.arange(N) + 1) % N
        segments = points[:, indices, :] - points  # (B, N, 3)

        # Fix last segment if not cyclic
        if not self.cyclic:
            segments[:, -1] = points[:, -1] - points[:, -2]  # repeat last segment

        # Parametric coordinate to segment index
        t, res_shape = self._broadcast_t(t)  # (B, T)
        S = N - 1  # number of segments
        s = np.clip(t * S, 0, S - 1e-6)
        i = np.floor(s).astype(int)  # (B, T)

        # Lookup segment tangents
        rows = np.arange(B)[:, None]
        tangents = segments[rows, i]  # (B, T, 3)

        if normalize:
            norm = np.linalg.norm(tangents, axis=-1, keepdims=True)
            tangents = np.divide(tangents, norm, where=(norm > 1e-12), out=np.zeros_like(tangents))

        return tangents.reshape(res_shape)

    # ----------------------------------------------------------------------------------------------------
    # Resample
    # ----------------------------------------------------------------------------------------------------

    def resample(self, count):
        """
        Sample evenly spaced points along a smoothed version of the spline(s),
        using a temporary BezierSpline interpolation.

        Parameters:
            count: int -- number of samples

        Returns:
            self (modified in-place)
        """
        count = count + 1 if self.cyclic else count

        if self._points.shape[-2] != count:
            bezier = BezierSpline(self._points, cyclic=False)
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
            t : scalar, (T,), (1, T), or (B, T)
            attribute : array of shape (N,), (B,N), (N,D), or (B,N,D)

        Returns:
            Array of shape:
                - (T,) or (T,D)       if scalar spline
                - (B,) or (B,D)       if t is scalar
                - (B,T) or (B,T,D)    otherwise
        """
        attr = self._adjust_attributes(attribute)
        coords = self._array_of_points  # (B, N, 3)
        B, N, _ = coords.shape

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
        Approximate this spline by a PolySpline spline using sampled points.

        Parameters
        ----------
        count : int
            Number of control points in the PolySpline approximation.

        Returns
        -------
        PolySpline : a new PolySpline spline instance
        """
        new_poly = PolySpline(self._points.copy(), cyclic=self.cyclic, already_closed=True)
        return new_poly.resample(count)
        
    # ----------------------------------------------------------------------------------------------------
    # Conversion to BezierSpline
    # ----------------------------------------------------------------------------------------------------

    def to_bezier(self, count=10, handle_scale=0.3):
        """
        Approximate a PolySpline spline with a Bézier spline by sampling `count` points and estimating tangents.

        Parameters
        ----------
        count : int
            Number of Bézier control points to use (≥ 2).
        handle_scale : float
            Scale factor for the distance to handles.

        Returns
        -------
        BezierSpline : new Bézier spline instance
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

        return BezierSpline(
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
        Plot the PolySpline spline using matplotlib.

        Parameters:
            resolution (int): Number of points to evaluate the curve.
            display_points (str): 'NO', 'POINTS', 'HANDLES', or 'ALL'
                                (only 'POINTS' and 'ALL' are used for PolySpline).
            label (str): Legend label for the spline.
            ax (matplotlib.axes.Axes): Optional matplotlib axis to draw on.
            **kwargs: Additional keyword arguments for `plot()` and `scatter()`.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        curve = self.evaluate(np.linspace(0, 1, resolution))  # (T, 3) or (B, T, 3)

        if curve.ndim == 2:  # Scalar spline
            ax.plot(curve[:, 0], curve[:, 1], label=label or "PolySpline", **kwargs)
            if display_points in {'POINTS', 'ALL'}:
                pts = self._array_of_points  # (N, 3)
                ax.scatter(pts[:, 0], pts[:, 1], color='red', s=40, label='Control Points', **kwargs)

        else:  # Batched splines
            for b in range(curve.shape[0]):
                pts = self._array_of_points[b]  # (N, 3)
                lbl = f"{label} {b}" if label else f"PolySpline {b}"
                ax.plot(curve[b, :, 0], curve[b, :, 1], label=lbl, **kwargs)
                if display_points in {'POINTS', 'ALL'}:
                    ax.scatter(pts[:, 0], pts[:, 1], s=30, color='red', **kwargs)

        ax.axis('equal')

# ====================================================================================================
# Bezier splines
# ====================================================================================================

class BezierSpline(PointsSpline):
    """
    Represents a batch of cubic Bézier splines.

    Each segment of the spline is defined by four control points:
    - An anchor point (P₀)
    - A right handle from the anchor (R₀)
    - A left handle toward the next anchor (L₁)
    - The next anchor point (P₁)

    This structure enables smooth C1-continuous curves, widely used in 2D/3D modeling,
    animation paths, and shape interpolation. The class supports scalar and batched splines,
    cyclic curves (with automatic handle closure), and operations like evaluation, tangent
    computation, resampling, and conversion.

    Features
    --------
    - Supports both scalar (N, 3) and batched (B, N, 3) Bézier splines
    - Handles can be explicitly provided or auto-computed for smoothness
    - Cyclic curves are supported (with closed-loop geometry and handles)
    - Evaluation via cubic Bézier interpolation per segment
    - Exact tangent evaluation using analytical derivatives
    - Attribute sampling using Bézier-style interpolation
    - Conversion to `PolySpline` or `BezierSpline` with fixed resolution

    Parameters
    ----------
    points : ndarray
        Anchor points of shape (N, 3) or (B, N, 3)
    left_handles : ndarray, optional
        Left handle positions, same shape as `points`
    right_handles : ndarray, optional
        Right handle positions, same shape as `points`
    cyclic : bool, default=False
        Whether the curve is cyclic (closed loop)
    already_closed : bool, default=False
        If True, assumes the last point is a duplicate of the first and skips closure logic

    Methods
    -------
    evaluate(t)
        Evaluate the point(s) on the curve at normalized parameter(s) t ∈ [0, 1]
    tangent(t)
        Compute the unit tangent(s) at parameter(s) t using Bézier derivatives
    length(resolution)
        Estimate arc length of each curve by sampling
    resample(count)
        Resample the curve into `count` evenly spaced segments
    sample_attribute(t, attribute)
        Sample a scalar or vector attribute along the curve using cubic interpolation
    to_bezier(count, handle_scale)
        Return a copy of the spline with `count` segments and optionally recomputed handles
    _plot(...)
        Visualize the curve using matplotlib (with anchors, handles, etc.)

    Examples
    --------
    >>> pts = np.random.rand(6, 3)
    >>> bez = BezierSpline(pts)
    >>> p = bez.evaluate(0.25)        # Evaluate position at 25%
    >>> t = bez.tangent(0.8)          # Get tangent vector at 80%
    >>> bez2 = bez.resample(20)       # Increase control points for more precision

    Notes
    -----
    - Handles can be left unspecified; they will be auto-generated for smooth interpolation.
    - Auto-generated handles are based on tangent estimation using surrounding points.
    - Evaluation is segment-based: `t=0` is at the first anchor, `t=1` at the last.
    - When cyclic=True, an extra segment joins the last and first anchor.
    - Bézier interpolation and tangent formulas follow the standard cubic basis.
    """

    curve_type = 'BEZIER'

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
            return BezierSpline(
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
    def control_handles(self):
        if self.cyclic:
            return self._left[..., :-1, :], self._right[...,:-1, :]
        else:
            return self._left, self._right

    @property
    def _array_of_handles(self):
        if self.is_scalar:
            return self._left[None], self._right[None]
        else:
            return self._left, self._right
        
    def copy(self):
        return BezierSpline(
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

    def tangent(self, t, normalize=False):
        """
        Evaluate the tangent vector(s) of the Bezier spline(s) at parameter(s) t ∈ [0, 1].

        The tangent corresponds to the derivative γ′(t) of the Bézier curve.
        If `normalize=True`, the tangent is returned as a unit vector.

        Parameters:
            t : scalar, (T,), (1, T), or (B, T)
                Evaluation parameter(s).
            normalize : bool, default False
                Whether to normalize the tangent(s) to unit length.

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
        rows = np.arange(B)[:, None]
        P0 = points[rows, i]      # (B, T, 3)
        R0 = right[rows, i]       # (B, T, 3)
        L1 = left[rows, i + 1]    # (B, T, 3)
        P1 = points[rows, i + 1]  # (B, T, 3)

        u2 = u * u
        omu = 1 - u
        omu2 = omu * omu

        dP = (
            3 * omu2[..., None] * (R0 - P0) +
            6 * omu[..., None] * u[..., None] * (L1 - R0) +
            3 * u2[..., None] * (P1 - L1)
        )  # (B, T, 3)

        if normalize:
            norm = np.linalg.norm(dP, axis=-1, keepdims=True)
            dP = np.divide(dP, norm, where=(norm > 1e-12), out=np.zeros_like(dP))

        return dP.reshape(res_shape)

    
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
        attr = self._adjust_attributes(attribute)
        points = self._array_of_points  # (B, N, 3)
        B, N, _ = points.shape

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
        new_bezier = BezierSpline(
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
            ax.plot(curve[:, 0], curve[:, 1], label=label or "BezierSpline", **kwargs)

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
                lbl = f"{label} {b}" if label else f"BezierSpline {b}"
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
    S_scalar = PolySpline(points_scalar)
    print(S_scalar)

    # Cas batch : 4 splines avec 5 points chacun
    points_batch = np.random.rand(4, 5, 3)
    S_batch = PolySpline(points_batch)
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

    # PolySpline scalaire : une sinusoïde simple
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.sin(x)
    points_scalar = np.stack([x, y, np.zeros_like(x)], axis=-1)
    spline_scalar = PolySpline(points_scalar)
    spline_scalar.resample(50)

    # PolySpline batchée : 4 sinusoïdes avec décalages de phase
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
    spline_batch = PolySpline(points_batch)
    spline_batch.resample(50)

    # Évaluation des splines
    t = np.linspace(0, 1, 100)
    eval_scalar = spline_scalar.evaluate(t)  # (100, 3)
    eval_batch = spline_batch.evaluate(t)    # (4, 100, 3)

    # Tracé
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(eval_scalar[:, 0], eval_scalar[:, 1], label='Scalar PolySpline')
    axs[0].set_title("PolySpline scalaire")
    axs[0].axis("equal")
    axs[0].legend()

    for i in range(B):
        axs[1].plot(eval_batch[i, :, 0], eval_batch[i, :, 1], label=f"PolySpline {i}")
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
    spline_scalar = PolySpline(points_scalar)
    spline_scalar.resample(100)  # Smooth using temporary BezierSpline

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
    spline_batch = PolySpline(points_batch)
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
            ax.scatter(x, y, label="PolySpline")
            ax.quiver(x, y, dx, dy, angles='xy', scale=20)
        else:  # (T, 3)
            x, y = points[:, 0], points[:, 1]
            dx, dy = tangents[:, 0], tangents[:, 1]
            ax.plot(x, y, label="PolySpline")
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
                ax.scatter(x, y, label=f"PolySpline {b}")
                ax.quiver(x, y, dx, dy, angles='xy', scale=20)
        elif points.ndim == 3:  # (B, T, 3)
            for b in range(B):
                x, y = points[b, :, 0], points[b, :, 1]
                dx, dy = tangents[b, :, 0], tangents[b, :, 1]
                ax.plot(x, y, label=f"PolySpline {b}")
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

    # Create a sinuous PolySpline spline
    x = np.linspace(0, 5 * np.pi, 60)
    y = np.sin(x)
    points = np.stack([x, y, np.zeros_like(x)], axis=-1)
    poly = PolySpline(points)

    # Convert to BezierSpline using sample-based approximation
    bezier = poly.to_bezier(reduction=0.3, handle_scale=0.3)

    # Prepare plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw both splines

    poly._plot(resolution=200, display_points='NO', ax=ax, label='PolySpline', color='blue')
    bezier._plot(resolution=200, display_points='POINTS', ax=ax, label='BezierSpline', linestyle='--', color='orange')

    # Draw dashed handle lines
    anchors = bezier._array_of_points       # (1, N, 3)
    lefts, rights = bezier._array_of_handles  # (1, N, 3), (1, N, 3)

    a = anchors[0]
    l = lefts[0]
    r = rights[0]

    for i in range(len(a)):
        ax.plot([a[i, 0], l[i, 0]], [a[i, 1], l[i, 1]], 'k--', lw=0.5)
        ax.plot([a[i, 0], r[i, 0]], [a[i, 1], r[i, 1]], 'k--', lw=0.5)

    ax.set_title("Conversion PolySpline → Bézier (sampled)")
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
    poly = PolySpline(points, cyclic=True)

    # Approximation en Bézier par échantillonnage
    bezier = poly.to_bezier(reduction=0.25, handle_scale=0.4)

    # Préparer le tracé
    fig, ax = plt.subplots(figsize=(6, 6))

    # Tracer la courbe PolySpline et la Bézier
    poly._plot(resolution=300, display_points='NO', ax=ax, label='PolySpline', color='blue')
    bezier._plot(resolution=300, display_points='POINTS', ax=ax, label='BezierSpline', linestyle='--', color='orange')

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
    circle = FunctionSpline.circle(radius=1.5)

    # Plot the curve
    fig, ax = plt.subplots(figsize=(6, 6))
    circle._plot(resolution=300, ax=ax, label="Circle", color='blue')

    # Plot tangents
    t = np.linspace(0, 1, 12)
    pts = circle.evaluate(t)
    tan = circle.tangent(t)
    ax.quiver(pts[:, 0], pts[:, 1], tan[:, 0], tan[:, 1], angles='xy', scale=20, color='orange', label='Tangents')

    ax.set_title("Circle from FunctionSpline")
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

def test_spiral_splinefunction_3d():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
    import numpy as np

    # Create a 3D spiral spline
    spiral = FunctionSpline.spiral(
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

    ax.set_title("3D Spiral from FunctionSpline")
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

    # Étape 1 : créer un cercle comme FunctionSpline
    sf = FunctionSpline.circle(radius=1.0)

    def func(t):
        x = t*np.pi*2
        y = np.sin(x)
        z = np.zeros_like(x)
        return np.stack((x, y, z), axis=-1)
    
    #sf = FunctionSpline(func=func, cyclic=False)

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

    # Plot 1: FunctionSpline
    sf._plot(ax=axes[0], resolution=300, label="FunctionSpline", color='black', linestyle='--')
    axes[0].set_title("FunctionSpline")

    # Plot 2: Bézier from function
    bez1._plot(ax=axes[1], resolution=300, label="BezierSpline (from Function)", color='orange')
    draw_bezier_handles(bez1, axes[1])
    axes[1].set_title("BezierSpline (from Function)")

    # Plot 3: PolySpline from function
    poly1._plot(ax=axes[2], resolution=300, label="PolySpline (from Function)", color='blue')
    axes[2].scatter(poly1._points[:, 0], poly1._points[:, 1], c='black', s=10, label='Control Points')
    axes[2].set_title("PolySpline (from Function)")

    # Plot 4: PolySpline from bezier
    poly2._plot(ax=axes[3], resolution=300, label="PolySpline (from BezierSpline)", color='green')
    axes[3].scatter(poly2._points[:, 0], poly2._points[:, 1], c='black', s=10, label='Control Points')
    axes[3].set_title(f"PolySpline (from BezierSpline) {poly2.shape=}")

    # Plot 5: BezierSpline from poly
    bez2._plot(ax=axes[4], resolution=300, label="BezierSpline (from PolySpline)", color='red')
    draw_bezier_handles(bez2, axes[4])
    axes[4].set_title("BezierSpline (from PolySpline)")

    # Final cleanup
    for ax in axes:
        ax.axis('equal')
        ax.legend()

    fig.suptitle("Conversion Pipeline: FunctionSpline ↔ Bézier ↔ PolySpline", fontsize=16)
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







