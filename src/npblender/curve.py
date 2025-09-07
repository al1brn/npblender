# MIT License
#
# Copyright (c) 2025 Alain Bernard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the \"Software\"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module Name: curve
Author: Alain Bernard
Version: 0.1.0
Created: 2023-11-10
Last updated: 2025-08-29

Summary:
    Curve Geeomety.

Usage example:
    >>> from curve import Curve

Notes:
    - Curve is made of splines.
    - Splines math relies upon the splinemaths module.
"""

__all__ = ["Curve", "MITER_NONE", "MITER_FLAT", "MITER_ROUND", "MITER_CUT"]

from contextlib import contextmanager
import numpy as np

import bpy

from .maths import MITER_NONE, MITER_FLAT, MITER_ROUND, MITER_CUT


from . constants import SPLINE_TYPES, BEZIER, POLY, NURBS
from . constants import bfloat, bint, bbool
from . import blender
from . maths import Rotation
from . maths import splinemaths

from . geometry import Geometry
from . domain import ControlPoint, Spline

DATA_TEMP_NAME = "npblender_TEMP"

# ====================================================================================================
# Curve, made of one or more splies
# ====================================================================================================

class Curve(Geometry):

    domain_names = ["points", "splines"]

    def __init__(self, points=None, splines=None, curve_type=POLY, materials=None, attr_from=None, **attrs):
        """
        Construct a curve geometry.

        Initializes empty domains by default, or builds a curve from provided
        control points and spline sizes/types. If both `points` and `splines` are
        already instances of [`ControlPoint`][npblender.ControlPoint] and
        [`Spline`][npblender.Spline], the curve is created as a **view**
        on these domains (no data copy). Otherwise, domains are allocated and
        populated via [`add_splines`][npblender.Curve.add_splines].

        Parameters
        ----------
        points : array-like or [`ControlPoint`][npblender.ControlPoint], optional
            Control-point positions (and optional per-point attributes) used to
            populate the curve. If a `ControlPoint` domain is provided together
            with a `Spline` domain, the curve becomes a view on them.
        splines : array-like or [`Spline`][npblender.Spline], optional
            Per-spline control-point counts (when building), or a ready-made
            `Spline` domain (to create a view).
        curve_type : int, default=POLY
            Default spline type for construction. One of
            [`BEZIER`][npblender.BEZIER],
            [`POLY`][npblender.POLY],
            [`NURBS`][npblender.NURBS].
            Ignored when `points`/`splines` are full domains.
        materials : str or sequence of str, optional
            Material names used by the curve. A single string is accepted and
            promoted to a one-element list.
        attr_from : object, optional
            Source whose transferable attributes are joined into this geometry
            (see [`join_attributes`][npblender.Geometry.join_attributes]).
        **attrs
            Additional geometry attributes to forward to
            [`add_splines`][npblender.Curve.add_splines] during construction.

        Notes
        -----
        - When `points` and `splines` are provided as domains, the instance is a
        **view**: `self.is_view` is `True` if the sum of `splines.loop_total`
        differs from `len(points)`. Use
        [`no_view`][npblender.Curve.no_view] to materialize a
        standalone copy. :contentReference[oaicite:0]{index=0}
        - Otherwise, the constructor allocates empty
        [`ControlPoint`](npblende.ControlPoint) and
        [`Spline`][npblender.Spline] domains, joins attributes from
        `attr_from` if any, then calls
        [`add_splines`][npblender.Curve.add_splines]. :contentReference[oaicite:1]{index=1}

        Raises
        ------
        ValueError
            Propagated from [`add_splines`][npblender.Curve.add_splines]
            when `curve_type` is not a single value (e.g., a list). :contentReference[oaicite:2]{index=2}
        """

        # ----- The materials
        if materials is None:
            self.materials = []
        elif isinstance(materials, str):
            self.materials = [materials]
        else:
            self.materials = [mat for mat in materials]

        # ----- The two domains are already built
        # can be a view on a larger curve
        if ( points is not None and isinstance(points, ControlPoint) and
             splines is not None and isinstance(splines, Spline) ):
            self.points = points
            self.splines = splines
            self.is_view = np.sum(self.splines.loop_total) != len(self.points)
            return
        
        self.is_view = False

        # ----- Initialize empty domains

        self.points  = ControlPoint()
        self.splines = Spline()

        self.join_attributes(attr_from)

        # ----- Add geometry
        curve_type = self.get_curve_type(curve_type)
        self.add_splines(points, splines, curve_type=curve_type, **attrs)

    # ----------------------------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------------------------

    @staticmethod
    def get_curve_type(curve_type):
        if isinstance(curve_type, str):
            if curve_type.upper() == 'POLY':
                return POLY
            elif curve_type.upper() == 'BEZIER':
                return BEZIER
            elif curve_type.upper() == 'NURBS':
                return NURBS
            else:
                raise ValueError(f"Unknown curve type: '{curve_type}'")
        else:
            curve_type = int(curve_type)
            if curve_type not in [BEZIER, POLY, NURBS]:
                raise ValueError(f"Curve type must be in [0, 1, 2], not {curve_type}")
            return curve_type

    def get_points_selection(self):
        """
        Indices/slice selecting the points actually referenced by the current splines.

        If the curve is a *view* (splines refer to a subset of points), returns a
        boolean mask selecting those rows in `points`. Otherwise returns `slice(None)`.

        Returns
        -------
        slice or ndarray of bool
            Selection usable to index `self.points`.
        """

        if not self.is_view:
            return slice(None)
        
        all_indices = np.arange(len(self.points))
        sel = np.zeros(len(self.points), dtype=bool)

        totals, rev_index = np.unique(self.splines.loop_total, return_inverse=True)
        for index, total in enumerate(totals):
            indices = self.splines.loop_total[rev_index == index][None, :] + np.arange(total)
            sel[indices.flatten()] = True

        return sel

    def no_view(self):
        """
        Materialize a view curve into an owned, self-consistent curve.

        Deep-copies the `splines` and gathers the referenced rows of `points`
        so that `sum(splines.loop_total) == len(points)` holds true.

        Returns
        -------
        Curve
            Self (for chaining).
        """
        if not self.is_view:
            return self

        import numpy as np

        # 1) Copy splines domain (owning copy)
        splines = Spline(self.splines, mode='COPY')

        # 2) Build a single flat index of point rows, bucketed by loop_total (N)
        lt = splines.loop_total
        ls = splines.loop_start

        uniq_N = np.unique(lt)
        idx_blocks = []
        for N in uniq_N:
            N = int(N)
            if N <= 0:
                continue
            sel = (lt == N)
            if not np.any(sel):
                continue
            starts = ls[sel]                                   # (B,)
            idx = starts[:, None] + np.arange(N, dtype=np.int64)  # (B, N)
            idx_blocks.append(idx.reshape(-1))                 # (B*N,)

        if idx_blocks:
            idx_all = np.concatenate(idx_blocks, axis=0)
            # 3) Slice the point FieldArray to keep all fields aligned
            points = ControlPoint(self.points, mode='COPY', selector=idx_all)  # owning copy
        else:
            points = ControlPoint()                       # empty

        # 4) Assign back and normalize bookkeeping
        self.points = points
        self.splines = splines.update_loop_start()
        self.is_view = False
        return self

    def check(self, title="Mesh Check", halt=True):
        """
        Validate internal spline bookkeeping.

        Delegates to [`Spline.check`][npblender.Spline.check]. When invalid
        and `halt=True`, raises; otherwise prints a message.

        Parameters
        ----------
        title : str, default="Mesh Check"
            Label used in error messages.
        halt : bool, default=True
            Whether to raise on failure.

        Returns
        -------
        bool
            True if valid.

        Raises
        ------
        Exception
            If the check fails and `halt=True`.
        """
        ok = self.splines.check(halt=False)
        if ok:
            return True
        elif halt:
            raise Exception(f"{title} check failed")
        else:
            print(f"{title} check failed")

    def __str__(self):
        scount = f"{len(self.splines)} spline{'s' if len(self.splines) > 1 else ''}"
        sview = f"view {np.sum(self.splines.loop_total)}/{len(self.points)}" if self.is_view else f"{len(self.points)}"
        return f"<Curve: {scount}, {sview} points.>"

    def __repr__(self):

        max_n = 10

        scount = f"{len(self.splines)} spline{'s' if len(self.splines) > 1 else ''}"
        sview = f"view {np.sum(self.splines.loop_total)}/{len(self.points)}" if self.is_view else f"{len(self.points)}"

        s = f"<Curve: {scount}, {sview} points:\n"
        if self.splines.is_scalar:
            s += (f"{'BEZIER' if self.splines.curve_type == 0 else 'POLY  '}, "
                f"{'Cyclic' if self.splines.cyclic else 'Open  '}, "
                f"{self.splines.loop_total:3d} points, "
                "f[{self.splines.loop_start:3d}:]\n")
        else:
            for i in range(min(max_n, len(self.splines))):
                s += (f" {i}: {'BEZIER' if self.splines.curve_type[i] == 0 else 'POLY  '}, "
                    f"{'Cyclic' if self.splines.cyclic[i] else 'Open  '}, "
                    f"{self.splines.loop_total[i]:3d} points, "
                    f"[{self.splines.loop_start[i]:3d}:]\n")
            if len(self.splines) > max_n:
                s += " ..."
        return s + ">"


    # ====================================================================================================
    # Serialization
    # ====================================================================================================

    def to_dict(self):
        """
        Serialize the curve to a plain Python dict.

        Returns
        -------
        dict
            Keys: ``geometry``, ``materials``, ``points``, ``splines``.
        """
        self.no_view()
        return {
            'geometry':     'Curve',
            'materials' :   self.materials,
            'points':       self.points.to_dict(),
            'splines':      self.splines.to_dict(),
            }

    @classmethod
    def from_dict(cls, d):
        """
        Deserialize a curve from a dict produced by `to_dict`.

        Parameters
        ----------
        d : dict
            Serialized curve payload.

        Returns
        -------
        Curve
        """
        curve = cls()
        curve.materials  = d['materials']
        curve.points     = ControlPoint.from_dict(d['points'])
        curve.splines    = Spline.from_dict(d['splines'])
        return curve
    
    # ====================================================================================================
    # Clear the geometry
    # ====================================================================================================

    def clear(self):
        """
        Remove all points and splines (attributes kept, values cleared).

        Returns
        -------
        None
        """
        self.no_view()

        self.points.clear()
        self.splines.clear()

    # ====================================================================================================
    # From another Curve
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Copy
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_curve(cls, other, points=None, splines=None):
        """
        Copy a curve (optionally subsetting points and/or splines).

        Parameters
        ----------
        other : Curve
            Source curve.
        points : selection or None, optional
            Points to **exclude** when copying (interpreted as mask if array of bool;
            see code for details).
        splines : selection or None, optional
            Splines to **exclude** when copying.

        Returns
        -------
        Curve
        """
        curve = cls(materials=other.materials)
        curve.points  = ControlPoint(other.points,  mode='COPY')
        curve.splines = Spline(other.splines, mode='COPY')

        if points is None:
            points_mask = None
        else:
            points_mask = np.ones(len(curve.points), dtype=bool)
            points_mask[points] = False

        if splines is None:
            splines_mask = None
        else:
            splines_mask = np.ones(len(curve.splines), dtype=bool)
            splines_mask[splines] = False

        curve.delete_points(points=points_mask, splines=splines_mask)

        return curve

    # ----------------------------------------------------------------------------------------------------
    # Capture another Curve
    # ----------------------------------------------------------------------------------------------------

    def capture(self, other):
        """
        Capture another curve’s buffers (no copy).

        Parameters
        ----------
        other : Curve
            Source whose internal buffers are adopted by this instance.

        Returns
        -------
        Curve
            Self.
        """
        self.materials = other.materials

        self.is_view = other.is_view
        self.points  = other.points
        self.splines = other.splines

    # ====================================================================================================
    # Interface with Blender
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Initialize from Blender Curve data
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_curve_data(cls, data):
        """
        Build a curve from a Blender `Curve` datablock.

        Parameters
        ----------
        data : bpy.types.Curve
            Blender curve data.

        Returns
        -------
        Curve
        """

        def read_point_attr(coll, attr, dtype, shape):
            n = len(coll)
            a = np.empty((n,) + shape, dtype=dtype)
            coll.foreach_get(attr, a.ravel())
            return a
        
        bl_splines = data.splines
        nsplines = len(bl_splines)
        curve = cls()
        curve.splines.resize(nsplines)

        # ----- Read the spline attributes

        a = np.empty(nsplines, int)
        for name, key in [('material_index', 'material_index'), ('resolution_u', 'resolution')]:
            bl_splines.foreach_get(name, a)
            setattr(curve.splines, key, a)

        a = np.empty(nsplines, bool)
        for name, key in [('use_cyclic_u', 'cyclic')]:
            bl_splines.foreach_get(name, a)
            setattr(curve.splines, key, a)

        # ----- Read the points

        for i_spline, bl_spline in enumerate(data.splines):

            if bl_spline.type == 'POLY':
                stype = 1
            elif bl_spline.type == 'NURBS':
                stype = 2
            else:
                stype = 0

            curve.splines[i_spline].curve_type = stype

            if stype == 0:
                coll = bl_spline.bezier_points
                attrs = {
                    'position'          : read_point_attr(coll, 'co',                bfloat, (3,)),
                    'handle_left'       : read_point_attr(coll, 'handle_left',       bfloat, (3,)),
                    'handle_right'      : read_point_attr(coll, 'handle_right',      bfloat, (3,)),
                    'handle_type_left'  : read_point_attr(coll, 'handle_left_type',  bint,   ()),
                    'handle_type_right' : read_point_attr(coll, 'handle_right_type', bint,   ()),
                    'radius'            : read_point_attr(coll, 'radius',            bfloat, ()),
                    'tilt'              : read_point_attr(coll, 'tilt',              bfloat, ()),
                }
                curve.splines[i_spline].loop_start = len(curve.points)
                curve.splines[i_spline].loop_total = len(coll)
                curve.points.append(**attrs)

            else:
                coll = bl_spline.points
                points4 = read_point_attr(coll, 'co', bfloat, (4,))
                attrs = {
                    'radius'  : read_point_attr(coll, 'radius',            bfloat, ()),
                    'tilt'    : read_point_attr(coll, 'tilt',              bfloat, ()),
                }
                curve.splines[i_spline].loop_start = len(curve.points)
                curve.splines[i_spline].loop_total = len(coll)
                curve.points.append(position=points4[:, :3], w=points4[:, 3], **attrs)

        return curve
    
    # ----------------------------------------------------------------------------------------------------
    # Write curve data
    # ----------------------------------------------------------------------------------------------------

    def to_curve_data(self, data):
        """
        Write this curve into an existing Blender `Curve` datablock.

        Parameters
        ----------
        data : bpy.types.Curve
            Target Blender curve data (cleared and repopulated).
        """
        self.no_view()

        bl_splines = data.splines
        bl_splines.clear()

        for i_spline, (curve_type, loop_start, loop_total) in enumerate(zip(
            self.splines.curve_type,
            self.splines.loop_start,
            self.splines.loop_total,
        )):
            
            if curve_type == 0:
                bl_spline = bl_splines.new('BEZIER')
                coll = bl_spline.bezier_points

                coll.add(loop_total - len(coll))

                a = self.points.position[loop_start:loop_start+loop_total]
                coll.foreach_set('co', a.ravel())

                attrs = [
                    'handle_left', 
                    'handle_right', 
                    ('handle_type_left', 'handle_left_type'),
                    ('handle_type_right', 'handle_right_type'),
                    'radius', 
                    'tilt',
                    ]

            else:
                if curve_type == 1:
                    bl_spline = bl_splines.new('POLY')
                else:            
                    bl_spline = bl_splines.new('NURBS')

                coll = bl_spline.points
                coll.add(loop_total - len(coll))

                p4 = np.empty((loop_total, 4), dtype=bfloat)
                p4[:, :3] = self.points.position[loop_start:loop_start+loop_total]
                p4[:, 3] = self.points.w[loop_start:loop_start+loop_total]

                coll.foreach_set('co', p4.ravel())

                attrs = ['radius', 'tilt', 'weight']


            for attr in attrs:

                if isinstance(attr, str):
                    field_name = attr
                    attr_name = attr
                else:
                    field_name = attr[0]
                    attr_name = attr[1]

                a = getattr(self.points, field_name)[loop_start:loop_start+loop_total]
                coll.foreach_set(attr_name, a.ravel())

        # Spline attrixbxutes

        a = self.splines.material_index
        bl_splines.foreach_set('material_index', a)

        a = self.splines.resolution
        bl_splines.foreach_set('resolution_u', a)

        a = self.splines.cyclic
        bl_splines.foreach_set('use_cyclic_u', a)

    # ----------------------------------------------------------------------------------------------------
    # From object
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_object(cls, obj, evaluated=False):
        """
        Build a curve from a Blender object holding curve data.

        Parameters
        ----------
        obj : str or bpy.types.Object
            Object or object name.
        evaluated : bool, default=False
            If True, read the evaluated (modifier-applied) data via depsgraph.

        Returns
        -------
        Curve
        """

        if evaluated:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            object_eval = blender.get_object(obj).evaluated_get(depsgraph)
            return cls.from_curve_data(object_eval.data)

        else:
            return cls.from_curve_data(blender.get_object(obj).data)

    # ----------------------------------------------------------------------------------------------------
    # To object
    # ----------------------------------------------------------------------------------------------------

    def to_object(self, obj, collection=None):
        """
        Create or update a Blender curve object from this geometry.

        Parameters
        ----------
        obj : str or bpy.types.Object
            Target object or name (created if it doesn't exist).
        collection : bpy.types.Collection or None, optional
            Collection to link a newly created object into.

        Returns
        -------
        bpy.types.Object
            The Blender curve object.

        Notes
        -----
        The object type must be *Curve*. Type conversion is not performed.
        """
        curve = blender.create_curve_object(obj, collection=collection)
        self.to_curve_data(curve.data)

        return curve
    
    # ----------------------------------------------------------------------------------------------------
    # Blender data context manager
    # ----------------------------------------------------------------------------------------------------
    
    @contextmanager
    def blender_data(self, readonly=False):
        """
        Temporary access to a transient Blender `Curve` datablock.

        Yields a throwaway curve data populated from this instance; upon exit,
        reads back into `self` unless `readonly=True`.

        Parameters
        ----------
        readonly : bool, default=False
            If True, do not read back any change made to the temporary data.

        Yields
        ------
        bpy.types.Curve
            The temporary curve data.

        Examples
        --------
        ``` python
        curve = Curve.Spiral()
        with curve.blender_data() as bcurve:
            print("Number of points", len(bcurve.splines[0].points))
        ```
        """

        data = bpy.data.curves.get(DATA_TEMP_NAME)
        if data is None:
            data = bpy.data.curves.new(DATA_TEMP_NAME, type='CURVE')

        self.to_curve_data(data)

        yield data

        # ----- Back

        if not readonly:
            self.capture(Curve.FromCurveData(data))   

    # ====================================================================================================
    # Add splines
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Extract attributes per domain
    # ----------------------------------------------------------------------------------------------------

    def _attributes_per_domain(self, **attrs):

        dispatched = {
            'points'  : {},
            'splines' : {},
            }

        for k, v in attrs.items():
            count = 0

            if k in self.points.all_names:
                dispatched['points'][k] = v
                count += 1
                
            if k in self.splines.all_names:
                dispatched['corners'][k] = v
                count += 1

            if count == 0:
                raise AttributeError(f"Unknown curve attribute '{k}'."
                                     f"- points:  {self.points.all_names}\n"
                                     f"- splines: {self.splines.all_names}\n"
                                     )

            if count > 1:
                raise AttributeError(f"Curve attribute '{k}' is ambigous, it belongs to more than one domain (count)."
                                     f"- points:  {self.points.all_names}\n"
                                     f"- splines: {self.splines.all_names}\n"
                                     )
        return dispatched
    
    # ----------------------------------------------------------------------------------------------------
    # Check consistency between points shape and spline sizes
    # ----------------------------------------------------------------------------------------------------

    def _check_points_splines_args(self, points, splines):
        
        points = np.asarray(points)

        if len(points.shape) == 2:
            if splines is None:
                splines = [len(points)]

            elif isinstance(splines, (int, np.int32, np.int64)):
                if len(points) % splines != 0:
                    raise ValueError(f"splines length ({splines}) is not a divider of the number of points ({len(points)}).")

                nsplines = len(points) // splines
                #splines = [splines]*nsplines
                points = points.reshape(nsplines, splines, points.shape[-1])

            elif hasattr(splines, '__len__'):
                if np.sum(splines) != len(points):
                    raise ValueError(f"The sum of spline lengths {np.sum(splines)} doesn't match the number of points {len(points)}.")

        elif len(points.shape) == 3:
            if splines is not None and splines != points.shape[1]:
                raise ValueError(
                    f"Points arguments is an array of {len(points)} splines of {points.shape[1]} points each, "
                    f"splines argument must be None or {points.shape[1]}, not {splines}")
            
            splines = points.shape[1]

        else:
            raise ValueError(f"Points argument must be an array of points, shape {points.shape} is not valid.")

        return points, splines
    
    # ----------------------------------------------------------------------------------------------------
    # Compute Bezier handles
    # ----------------------------------------------------------------------------------------------------

    @staticmethod
    def compute_bezier_handles(points, cyclic=False, eps=1e-6):
        """
        Compute cubic Bezier handles (left/right) from anchors using Catmull-Rom style tangents.

        Parameters
        ----------
        points : (N,3) or (B,N,3) float32/float64
            OPEN storage (no duplicated first point).
        cyclic : bool
            If True, wrap neighbors; else use one-sided differences at ends.
        eps : float
            Small epsilon to guard against zero-length tangents.

        Returns
        -------
        left, right : same shape as `points`
            left[i]  = P[i] - T[i] * (len_in[i]  / 3)
            right[i] = P[i] + T[i] * (len_out[i] / 3)
            with len_in = ||P[i] - P[i-1]|| and len_out = ||P[i+1] - P[i]|| (wrapped if cyclic).
        """
        P = np.asarray(points)
        single = (P.ndim == 2)
        if single:
            P = P[None, ...]  # (1,N,3)

        B, N, C = P.shape
        if N == 0:
            L = np.empty_like(P)
            R = np.empty_like(P)
            return (L[0], R[0]) if single else (L, R)
        if N == 1:
            # With a single point, handles equal the point
            L = P.copy()
            R = P.copy()
            return (L[0], R[0]) if single else (L, R)

        # Neighbors (OPEN). Use roll for cyclic, clamp edges for non-cyclic.
        if cyclic:
            P_prev = np.roll(P, 1, axis=1)    # P[i-1]
            P_next = np.roll(P, -1, axis=1)   # P[i+1]
        else:
            P_prev = np.concatenate([P[:, :1, :],  P[:, :-1, :]], axis=1)  # [P0, P0..P_{N-2}]
            P_next = np.concatenate([P[:, 1:, :],  P[:, -1:, :]], axis=1)  # [P1..P_{N-1}, P_{N-1}]

        # Catmull-Rom style tangent (centered diff), then normalize
        der = 0.5 * (P_next - P_prev)                         # (B,N,3)
        n = np.linalg.norm(der, axis=-1, keepdims=True)       # (B,N,1)
        n = np.maximum(n, eps)
        T = der / n                                           # unit tangents

        # Segment lengths in/out
        d_out = np.linalg.norm(P_next - P, axis=-1, keepdims=True)  # |P[i+1]-P[i]|
        d_in  = np.linalg.norm(P - P_prev, axis=-1, keepdims=True)  # |P[i]-P[i-1]|

        # Handles: linear-preserving scale (len/3)
        L = P - T * (d_in  / 3.0)
        R = P + T * (d_out / 3.0)

        if single:
            return L[0], R[0]
        return L, R
    
    # ----------------------------------------------------------------------------------------------------
    # Add Bezier
    # ----------------------------------------------------------------------------------------------------

    def add_bezier(self, points, splines=None, handle_left=None, handle_right=None, cyclic=False, **attrs):
        """
        Append Bezier spline(s).

        The arguments `splines` gives the length(s) of the bezier spline(s). If None:
        - the number of points is taken (one spline is added)
        - points.shape[1] is taken if the shape of points is (m, , 3)

        handle_left and handle_right must have the same shape as points if provided.
        If they aren't provided, they are computed.


        Parameters
        ----------
        points : ndarray, shape (N, 3) or (B, N, 3)
            Anchor positions (single spline or a batch).
        splines : int or 1D array of int or None, optional
            Per-spline point counts. If `None`, inferred from `points`.
        handle_left : ndarray or None, optional
            Left handles (same shape as `points`). Computed if omitted.
        handle_right : ndarray or None, optional
            Right handles (same shape as `points`). Computed if omitted.
        cyclic : bool or 1D array of bool, default=False
            Whether each spline is closed (broadcastable to number of splines).
        **attrs
            Additional attributes dispatched to points/splines.

        Returns
        -------
        dict
            Indices of appended points and splines.

        Raises
        ------
        ValueError
            If `points` last dimension is not 3, or if handle shapes don’t match.
        """
        self.no_view()

        if points is None:
            return {'points': [], 'splines': []}
        
        # ----- handle_left and handle_right shapes must match points shape

        if handle_left is not None and np.shape(handle_left) != np.shape(points):
            raise ValueError("add_bezier> handle_left and points shapes must match.")

        if handle_right is not None and np.shape(handle_right) != np.shape(points):
            raise ValueError("add_bezier> handle_right and points shapes must match.")

        # ----- Make sure args are consistent

        points, splines = self._check_points_splines_args(points, splines)
        if points.shape[-1] != 3:
            raise ValueError("add_bezier> points must be an array of 3D vectors.")

        # ----- Dispatch attributes

        disp_attrs = self._attributes_per_domain(**attrs)
        added = {}

        # ----- A stack of splines

        if len(points.shape) == 3:

            if handle_left is None or handle_right is None:
                h_left, h_right = self.compute_bezier_handles(points, cyclic=disp_attrs.get('cyclic', False))
                if handle_left is None:
                    handle_left = h_left
                if handle_right is None:
                    handle_right = h_right
            
            if np.shape(handle_left) != points.shape or np.shape(handle_right) != points.shape:
                raise ValueError(
                    f"add_bezier> handle_left ({np.shape(handle_left)}) and handle_right "
                    f"({np.shape(handle_right)}) must have the same shape as points  ({np.shape(points)}).")
            
            shape = points.shape[:-1]
            n = int(np.prod(shape))
            p_attrs = {}
            for name, value in disp_attrs['points'].items():
                sh = self.points._infos[name]['shape']
                if sh == ():
                    p_attrs[name] = np.broadcast_to(value, shape).ravel()
                else:
                    p_attrs[name] = np.broadcast_to(value, shape + sh).reshape((n,) + sh )

            added['points'] = self.points.append(
                position = points.reshape(-1, 3),
                handle_left = handle_left.reshape(-1, 3),
                handle_right = handle_right.reshape(-1, 3),
                **p_attrs,
                #**{name: [value] for name, value in disp_attrs['points'].items()},
                )
            
            # Add the splines (splines argument is an int)
            added['splines'] = self.splines.append_sizes([splines]*len(points), curve_type=BEZIER, cyclic=cyclic, **disp_attrs['splines'])
            
        # ----- A series of splines

        else:
            cur_points = len(self.points)

            offset = 0
            h_left, h_right = None, None
            for size, cycl in zip(splines, np.broadcast_to(cyclic, len(splines))):
                
                pts = points[offset:offset + size]

                # Compute handles
                if handle_left is None or handle_right is None:
                    h_left, h_right = self.compute_bezier_handles(pts, cyclic=cycl)
                hl = h_left if handle_left is None else handle_left[offset:offset + size]
                hr = h_right if handle_right is None else handle_right[offset:offset + size]

                # Append points
                new_points = self.points.append(position=pts, handle_left=hl, handle_right=hr, **disp_attrs['points'])

                # Next
                offset += size
            
            added['points']= np.arange(cur_points, len(self.points))

            # Add the splines (splines argument is an array of ints)
            added['splines'] = self.splines.append_sizes(splines, curve_type=BEZIER, cyclic=cyclic, **disp_attrs['splines'])

        return added
    
    # ----------------------------------------------------------------------------------------------------
    # Add Bezier
    # ----------------------------------------------------------------------------------------------------

    def add_poly(self, points, splines=None, curve_type=POLY, cyclic=False, w=1., **attrs):
        """
        Append Poly or NURBS spline(s).

        The arguments 'splines' gives the length(s) of the spline(s). If None:
        - the number of points is taken (one spline is added)
        - points.shape[1] is taken if the shape of points is (m, , 3)

        Non bezeier splines use 4D points. If the provided vectors are 4D, the argument w
        is ignored.

        Parameters
        ----------
        points : ndarray
            Either `(N, 3|4)` or `(B, N, 3|4)`. If 4D, the 4th component is used as w.
        splines : int or 1D array of int or None, optional
            Per-spline point counts. If `None`, inferred from `points`.
        curve_type : int, default=POLY
            [`POLY`][npblender.POLY] or [`NURBS`][npblender.NURBS].
        cyclic : bool or 1D array of bool, default=False
            Whether each spline is closed.
        w : float, default=1.0
            Default weight if `points` are 3D.
        **attrs
            Additional attributes dispatched to points/splines.

        Returns
        -------
        dict
            Indices of appended points and splines.

        Raises
        ------
        ValueError
            If `points` are neither 3D nor 4D vectors.
        """

        self.no_view()

        if points is None:
            return {'points': [], 'splines': []}

        # ----- Make sure args are consistent

        points, splines = self._check_points_splines_args(points, splines)
        if points.shape[-1] not in (3, 4):
            raise ValueError("add_poly> points must be an array of 3D or 4D vectors.")

        # ----- Dispatch attributes

        disp_attrs = self._attributes_per_domain(**attrs)
        added = {}

        # ----- A stack of splines

        if len(points.shape) == 3:

            position = points[..., :3]
            if points.shape[-1] == 4:
                w = points[..., 3].ravel()

            shape = points.shape[:-1]
            n = int(np.prod(shape))
            p_attrs = {}
            for name, value in disp_attrs['points'].items():
                sh = self.points._infos[name]['shape']
                if sh == ():
                    p_attrs[name] = np.broadcast_to(value, shape).ravel()
                else:
                    p_attrs[name] = np.broadcast_to(value, shape + sh).reshape((n,) + sh )

            # Add the points
            added['points'] = self.points.append(
                position = position.reshape(-1, 3),
                w = w,
                #**{name: [value] for name, value in disp_attrs['points'].items()},
                **p_attrs,
                )
            
            # Add the splines (splines argument is an int)
            curve_type = self.get_curve_type(curve_type)
            added['splines'] = self.splines.append_sizes([splines]*len(points), curve_type=curve_type, cyclic=cyclic, **disp_attrs['splines'])
            
        # ----- A series of splines

        else:
            # Add the points
            position = points[:, :3]
            if points.shape[-1] == 4:
                w = points[:, 3]
            else:
                w = np.broadcast_to(w, len(points))

            added['points'] = self.points.append(
                position = position,
                w = w,
                **disp_attrs['points'],
            )

            # Add the splines (splines argument is an array of ints)
            added['splines'] = self.splines.append_sizes(splines, curve_type=curve_type, cyclic=cyclic, **disp_attrs['splines'])

        return added    

    # ----------------------------------------------------------------------------------------------------
    # Add splines
    # ----------------------------------------------------------------------------------------------------

    def add_splines(self, points=None, splines=None, curve_type=POLY, **attrs):
        """
        Append new splines (Bezier, Poly, or NURBS) with their control points.

        Parameters
        ----------
        points : ndarray or None, optional
            Control-point coordinates. Shape depends on `splines` (see
            [`add_bezier`][npblender.Curve.add_bezier] and
            [`add_poly`][npblender.Curve.add_poly]).
        splines : int or 1D array of int or None, optional
            Per-spline sizes. If `None`, inferred from `points` shape.
        curve_type : int, default=POLY
            One of [`BEZIER`][npblender.BEZIER], [`POLY`][npblender.POLY],
            [`NURBS`][npblender.NURBS].
        **attrs
            Mixed per-point and per-spline attributes to broadcast and assign.

        Returns
        -------
        dict
            Keys ``'points'`` and ``'splines'`` with the indices of appended rows.

        Raises
        ------
        ValueError
            If `curve_type` is not a single scalar value.
        """
        self.no_view()
        curve_type = self.get_curve_type(curve_type)

        if hasattr(curve_type, '__len__'):
            raise ValueError("Curve add_splines> curve_type must be a single value in (BEZIER, POLY, NURBS), not a list.\ncurve_type: {curve_type}")
        
        if curve_type == BEZIER:
            return self.add_bezier(points, splines=splines, **attrs)
        else:
            return self.add_poly(points, splines=splines, curve_type=curve_type, **attrs)
        
    # ====================================================================================================
    # Delete
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Delete points
    # ----------------------------------------------------------------------------------------------------

    def delete_points(self, points=None, splines=None):
        """
        Delete points (and prune splines when emptied).

        Parameters
        ----------
        points : selection or None, optional
            Points to delete.
        splines : selection or None, optional
            Splines whose **all** points should be deleted.

        Returns
        -------
        Curve
            Self (for chaining).
        """

        self.no_view()

        del_points = np.zeros(len(self.points), dtype=bool)
        if points is not None:
            del_points[points] = True

        del_splines = np.zeros(len(self.splines), dtype=bool)
        if splines is not None:
            del_splines[splines] = True

        new_total = self.splines.loop_total
        for i_spline, (loop_start, loop_total) in enumerate(zip(self.splines.loop_start, self.splines.loop_total)):

            if del_splines[i_spline]:
                del_points[loop_start:loop_start+loop_total] = True

            else:
                n = np.sum(del_points[loop_start:loop_start+loop_total])
                if n == loop_total:
                    del_splines[i_spline] = True
                else:
                    new_total[i_spline] = loop_total - n

        self.points.delete(del_points)
        self.splines.delete(del_splines)
        self.splines.update_loop_start()

        return self
    
    # ----------------------------------------------------------------------------------------------------
    # Delete splines
    # ----------------------------------------------------------------------------------------------------

    def delete_splines(self, splines=None):
        """
        Delete splines (and their points).

        Parameters
        ----------
        splines : selection or None, optional
            Splines to delete.

        Returns
        -------
        Curve
            Self (for chaining).
        """

        self.no_view()

        return self.delete_points(splines=splines)
        
    # ====================================================================================================
    # Combining
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Join other curves
    # ----------------------------------------------------------------------------------------------------

    def join(self, *others):
        """
        Append other curves to this one (points, splines, and material mapping).

        Parameters
        ----------
        *others : Curve
            Curves to concatenate.

        Returns
        -------
        Curve
            Self (for chaining).
        """

        self.no_view()

        for other in others:
            other.no_view()

            # Splines

            pts_offset = len(self.points)
            self.points.extend(other.points)

            spl_offset = len(self.splines)
            self.splines.extend(other.splines)
            self.splines[spl_offset:].loop_start += pts_offset

            # Materials

            remap = np.array([self.get_material_index(mat_name) for mat_name in other.materials])
            if len(remap):
                self.splines.material_index[spl_offset:] = remap[other.splines.material_index]

        return self
    
    # ----------------------------------------------------------------------------------------------------
    # Multiply
    # ----------------------------------------------------------------------------------------------------

    def multiply(self, count, in_place=True):
        """
        Duplicate the whole curve `count` times (instancing-like expansion).

        Multiplying is a way to efficiently duplicate the geometry a great number of times.

        Parameters
        ----------
        count : int
            Number of copies to create.
        in_place : bool, default=True
            If True, expand this instance; otherwise return a new expanded curve.

        Returns
        -------
        Curve or None
            Self (in place) or a new curve; `None` if `count == 0`.

        Raises
        ------
        Exception
            If `count` is not an integer.
        """

        self.no_view()

        # ----------------------------------------------------------------------------------------------------
        # Checks and simple cases

        if not isinstance(count, (int, np.int32, np.int64)):
            raise Exception(f"A Mesh can be multiplied only by an int, not '{count}'")

        if count == 0:
            return None
        
        if count == 1:
            if in_place:
                return self
            else:
                return type(self).from_curve(self)
            
        if not in_place:
            return type(self).from_curve(self).multiply(count, in_place=True)
        
        # ----------------------------------------------------------------------------------------------------
        # Points

        nverts = len(self.points)
        self.points.multiply(count)

        # ----------------------------------------------------------------------------------------------------
        # Splines

        self.splines.multiply(count)
        self.splines.update_loop_start()

        return self
    
    # ----------------------------------------------------------------------------------------------------
    # Operators
    # ----------------------------------------------------------------------------------------------------

    def __mul__(self, count):
        return self.multiply(count, in_place=False)

    def __imul__(self, count):
        return self.multiply(count, in_place=True)

    # ====================================================================================================
    # Evaluation
    # ====================================================================================================

    @property
    def length(self):
        """
        Lengths of splines.

        Returns
        -------
        array of floats
        """
        return splinemaths.length_curve(self)

    def evaluate(self, t):
        """
        Evaluate positions along each spline at parameter(s) `t`.

        Parameters
        ----------
        t : float or ndarray
            Parametric coordinate(s) in `[0, 1]` per spline.

        Returns
        -------
        ndarray, shape (..., 3)
            Evaluated positions.
        """
        return splinemaths.evaluate_curve(self, t)

    def tangent(self, t, normalize=True):
        """
        Evaluate tangents along each spline at parameter(s) `t`.

        Parameters
        ----------
        t : float or ndarray
            Parametric coordinate(s) in `[0, 1]` per spline.
        normalize : bool, default=True
            If True, return unit tangents.

        Returns
        -------
        ndarray, shape (..., 3)
            Tangent vectors.
        """
        return splinemaths.tangent_curve(self, t, normalize=normalize)

    def sample_attributes(self, t, names=None, cubic=False):
        """
        Sample point-domain attributes along the curve(s) at `t`.

        Parameters
        ----------
        t : float or ndarray
            Parametric coordinate(s) in `[0, 1]` per spline.
        names : sequence of str or None, optional
            Attribute names to sample; if `None`, samples transferable point fields.
        cubic : bool, default=False
            Use cubic interpolation (typically for Bezier).

        Returns
        -------
        dict[str, ndarray]
            Sampled arrays keyed by attribute name.
        """
        return splinemaths.sample_attributes_curve(self, t, names=names, cubic=cubic)
    
    # ====================================================================================================
    # Operations on splines
    # ====================================================================================================

    def __len__(self):
        return len(self.splines)
    
    def __getitem__(self, index):
        splines = self.splines[index]
        if splines is None:
            raise IndexError(f"Curve index is not valid on curve {self}, index:\n{index}")
        return Curve(points=self.points, splines=splines.no_scalar())
    
    def select(self, indices):
        """
        Convenience alias for subsetting splines: `curve[indices]`.

        Parameters
        ----------
        indices : selection
            Boolean mask, slice, or integer array indexing splines.

        Returns
        -------
        Curve
            A view on the selected splines (potentially a *view* curve).
        """
        return self[indices]   
    
    # ====================================================================================================
    # Loop on buckets
    # ====================================================================================================

    def for_each_bucket(self, func):
        """
        Iterate homogeneous spline buckets and apply `func`.

        Spline buckets share the same `(curve_type, N, cyclic, resolution)` signature.
        `func` is called with `(curve, curve_type, N, cyclic, resolution)` and must
        return a value; the generator yields `(bucket_indices, value)` pairs.

        Parameters
        ----------
        func : callable
            Callback applied once per bucket.

        Yields
        ------
        tuple
            `(bucket_indices, value)` for each bucket.
        """

        spl = self.splines
        if len(spl) == 0:
            return

        # Build buckets keyed by (type, N, cyclic, resolution_or_None)
        buckets = {}
        for index, (ctype, N, cyc) in enumerate(zip(spl.curve_type, spl.loop_total, spl.cyclic)):
            if ctype == BEZIER:
                resol = spl.resolution[index]
                key = (ctype, N, cyc, resol)
            else:
                key = (ctype, N, cyc, None)
            buckets.setdefault(key, []).append(index)

        # Iterate buckets and call user function once per bucket
        for (ctype, N, cyc, resol), idxs in buckets.items():
            result = func(self[idxs], ctype, N, cyc, resol)
            yield idxs, result
    
    # ====================================================================================================
    # Conversions
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # To poly
    # ----------------------------------------------------------------------------------------------------

    def to_poly(self, resolution=None):
        """
        Convert all splines to Poly.

        Parameters
        ----------
        resolution : int or None, optional
            If `None`, Bezier splines are split using their per-spline resolution and
            poly splines are left unchanged. If an integer, resample *all* splines
            to that resolution (cyclic splines have no duplicate endpoint).

        Returns
        -------
        Curve
            A new curve with `curve_type = POLY`.
        """

        if resolution is not None:
            resolution = max(2, resolution)

        attr_names = self.points.transdom_names
        attr_names.extend(['w', 'tilt'])
        
        def _to_poly(curve, ctype, loop_total, is_cyclic, resol=None):

            if resolution is None:
                if ctype == POLY:
                    return curve.no_view()
                
                r = max(1, int(resol))
                
                if is_cyclic:
                    t = np.linspace(0, 1, loop_total*r, endpoint=False, dtype=bfloat)
                else:
                    t = np.linspace(0, 1, (loop_total - 1)*r + 1, endpoint=True, dtype=bfloat)
            else:
                if ctype == POLY and resolution == loop_total:
                    return curve.no_view()
                
                t = np.linspace(0, 1, resolution, endpoint=not is_cyclic, dtype=bfloat)

            pos = curve.evaluate(t).reshape(-1, 3)
            attrs = {}
            for k, v in curve.sample_attributes(t, names=attr_names, cubic=ctype==BEZIER).items():
                field_shape = curve.points.get_field_shape(k)
                if field_shape == ():
                    attrs[k] = v.flatten()
                else:
                    attrs[k] = v.reshape((-1,) + field_shape)

            points = ControlPoint(position=pos, attr_from=self.points, **attrs)
            splines = Spline(curve.splines)
            splines.curve_type = POLY
            splines.loop_total = len(t)
            splines.update_loop_start()

            new_curve = Curve(points=points, splines=splines)
            new_curve.is_view=False
            return new_curve
        
        new_curve = Curve(materials=self.materials)
        for _, c in self.for_each_bucket(_to_poly):
            new_curve.splines.extend(c.splines)
            new_curve.points.extend(c.points)

        new_curve.splines.update_loop_start()
        return new_curve
    
    # ----------------------------------------------------------------------------------------------------
    # To Bezier
    # ----------------------------------------------------------------------------------------------------

    def to_bezier(self, control_count=None, resolution=16):
        """
        Convert all splines to Bezier.

        Parameters
        ----------
        control_count : int or None, optional
            Number of anchors per spline. If `None`, keep the current `loop_total`.
            For cyclic splines, anchors are sampled on `[0, 1)` (no duplicate).
        resolution : int, default=16
            Per-segment resolution written to `splines.resolution` (>= 1).

        Returns
        -------
        Curve
            A new Bezier curve.
        """

        resolution = max(1, int(resolution))

        attr_names = self.points.transdom_names
        attr_names.extend(['tilt'])

        def _to_bezier(curve, ctype, loop_total, is_cyclic, _resol=None):

            if ctype == BEZIER and control_count is None:
                return curve.no_view()

            # Decide target anchor count N for this bucket
            N = int(control_count) if control_count is not None else int(loop_total)
            if N <= 0:
                # build empty bucket preserving schema
                pts = ControlPoint()
                spl = Spline(curve.splines)
                spl.curve_type = BEZIER
                spl.loop_total = 0
                spl.resolution = resolution
                spl.update_loop_start()
                out = Curve(points=pts, splines=spl)
                out.is_view = False
                return out

            # Parameter grid for anchors
            t = np.linspace(0.0, 1.0, N, endpoint=not is_cyclic, dtype=bfloat)

            # Evaluate anchors
            P = curve.evaluate(t)                  # (S, N, 3) or (S,3) if N==1
            if P.ndim == 2:
                P = P[:, None, :]                  # (S,1,3) uniformize

            # Compute handles from anchors (Catmull-Rom style tangents, linear-preserving scale)
            L, R = Curve.compute_bezier_handles(P, cyclic=is_cyclic)  # (S,N,3) each

            # Flatten (S,N,3) → (S*N,3)
            P_flat = P.reshape(-1, 3)
            L_flat = L.reshape(-1, 3)
            R_flat = R.reshape(-1, 3)

            # Sample additional point attributes at anchors
            attrs = {}
            for k, v in curve.sample_attributes(t, name=attr_names, cubic=(ctype == BEZIER)).items():
                field_shape = curve.points.get_field_shape(k)  # e.g. (), (D,), (H,W), ...
                if field_shape == ():
                    attrs[k] = v.reshape(-1)
                else:
                    attrs[k] = v.reshape((-1,) + field_shape)

            # Build new domains for this bucket
            points = ControlPoint(position=P_flat,
                                    handle_left=L_flat,
                                    handle_right=R_flat,
                                    attr_from = self.points,
                                    **attrs)

            splines = Spline(curve.splines)  # copy bucket rows
            splines.curve_type = BEZIER
            splines.loop_total = N
            splines.resolution = resolution
            splines.update_loop_start()

            out = Curve(points=points, splines=splines)
            out.is_view = False
            return out

        # Aggregate all buckets into a fresh curve (materials kept)
        new_curve = Curve(materials=self.materials)
        for _, c in self.for_each_bucket(_to_bezier):
            new_curve.splines.extend(c.splines, join_fields=False)
            new_curve.points.extend(c.points,   join_fields=False)

        new_curve.splines.update_loop_start()
        new_curve.splines.resolution = resolution
        return new_curve

    # ----------------------------------------------------------------------------------------------------
    # To mesh
    # ----------------------------------------------------------------------------------------------------

    def to_mesh(self, profile=None, caps=True, use_radius=True, camera_culling=False):
        """
        Convert the curve to a mesh (optionally sweeping a profile).

        Parameters
        ----------
        profile : Curve or None, optional
            Profile curve to sweep along each spline. If `None`, outputs edges only.
        caps : bool, default=True
            Close ends when the profile is cyclic and the path is open.
        use_radius : bool, default=True
            Use per-point `radius` to scale the profile (when applicable).
        camera_culling : bool or Camera, default=False
            If truthy, drop splines that would be sub-pixel using a camera model.

        Returns
        -------
        [Mesh][npblender.Mesh]
            The generated mesh.
        """

        from . camera import Camera
        from . mesh import Mesh
        from .maths.topology import grid_corners, grid_uv_map, disk_uv_map

        if camera_culling != False:
            camera = Camera(camera_culling)
            camera_culling = True

        # Circle as default profile
        if profile is None:
            profile_size = .001
        else:
            full_prof_mesh = profile[0].to_mesh()
            full_prof_pts = full_prof_mesh.points.position
            full_nprof = len(full_prof_pts)
            prof_closed = profile.splines.cyclic[0]
            profile_size = profile.max_size

        # ---------------------------------------------------------------------------
        # Add a set of splines of the same type, number of points and cyclic
        # ---------------------------------------------------------------------------

        def _to_mesh(curve, ctype, N, cyclic, resol, profile_scale=None):

            if ctype == BEZIER:
                curve = curve.to_poly()
                N = curve.splines.loop_total[0]
            else:
                curve.no_view()

            mesh = Mesh()
            
            # Number of splines and points
            nsplines = len(curve.splines)
            npoints = len(curve.points)

            # ---------------------------------------------------------------------------
            # No profile : we simply build edges
            # ---------------------------------------------------------------------------

            if profile is None:

                # Create mesh points
                mesh.add_points(curve.points.position)

                # Transfer trans domain attributes
                mesh.points.transfer_attributes(curve.points)

                # Edges
                inds = np.arange(npoints).reshape(nsplines, N)
                if cyclic:
                    edges = np.empty((nsplines, N, 2), dtype=bint)
                    edges[:, :-1, 0] = inds[:, :-1]
                    edges[:, :-1, 1] = inds[:, 1:]
                    edges[:, -1, 0] = inds[:, -1]
                    edges[:, -1, 1] = inds[:, 0]
                else:
                    edges = np.empty((nsplines, N-1, 2), dtype=bint)
                    edges[..., 0] = inds[:, :-1]
                    edges[..., 1] = inds[:, 1:]

                edges = edges.reshape(-1, 2)
                mesh.edges.append(vertex0=edges[..., 0], vertex1=edges[..., 1])

                return mesh
        
            # ---------------------------------------------------------------------------
            # We have a profile
            # ---------------------------------------------------------------------------

            ok_caps = caps and not cyclic and prof_closed

            # Let's adapt to the scale
            if profile_scale is None:
                nprof = full_nprof
                prof_mesh = full_prof_mesh
            else:
                nprof = max(4, int(profile_scale*full_nprof))
                prof_mesh = profile[0].to_poly(nprof).to_mesh()
                ok_caps = ok_caps and (profile_scale > .8)

            # One profile per center
            all_points = np.empty((nsplines, N, nprof, 3), dtype=bfloat)
            all_points[...] = prof_mesh.points.position

            # Radius
            if use_radius:
                all_points *= curve.points.radius.reshape(nsplines, N, 1, 1)

            # Orientation along the tangent
            t = np.linspace(0, 1, N, endpoint=not cyclic)
            tangent = curve.tangent(t, normalize=True) # (nsplines, N, 3)

            rot = Rotation.look_at((0, 0, 1), tangent, up=(0, 1, 0), normalized=True) # (nsplines, N, 3)

            # Tilt
            euler = np.zeros((npoints, 3), dtype=bfloat)
            euler[:, 2] = curve.points.tilt
            tilt = Rotation.from_euler(euler.reshape(nsplines, N, 3))
            rot = rot @ tilt

            # Rotation
            all_points = rot[:, :, None] @ all_points

            # Translation
            all_points += curve.points.position.reshape(nsplines, N, 1, 3)

            # Add to mesh
            all_points = all_points.reshape(-1, 3)
            mesh.add_points(all_points)


            # Transfer trans domain attributes
            mesh.points.transfer_attributes(curve.points, shape=(nsplines, N, nprof), other_shape=(nsplines, N, 1))
            mesh.points.transfer_attributes(profile.points, shape=(nsplines, N, nprof), other_shape=(1, 1, nprof))

            # Grid corners
            corners = grid_corners(N, nprof, row_first=False, close_x=cyclic, close_y=prof_closed, clockwise=True).flatten()
            ncorners = len(corners)
            corners = np.tile(corners, nsplines) + np.repeat(np.arange(nsplines, dtype=bint)*(nprof*N), ncorners)

            bbox = [0, 0, 1, .499] if ok_caps else [0, 0, 1, 1]
            uvmap = grid_uv_map(N, nprof, close_x=cyclic, close_y=prof_closed, clockwise=False, bbox=bbox).reshape(-1, 2)
            uvmap = np.tile(uvmap, (nsplines, 1))

            mesh.add_geometry(corners=corners.flatten(), faces=4, UVMap=uvmap)

            #Caps
            if ok_caps:
                inds = np.arange(len(all_points)).reshape(nsplines, N, nprof)

                cap_inds = np.flip(inds[:, 0], axis=-1)
                uvmap = disk_uv_map(nprof, mode='NGON', clockwise=True, bbox=[0, .5, .5, 1]).reshape(-1, 2)
                uvmap = np.tile(uvmap, (nsplines, 1))
                mesh.add_geometry(corners=cap_inds.flatten(), faces=nprof, UVMap=uvmap)

                cap_inds = inds[:, -1]
                uvmap = disk_uv_map(nprof, mode='NGON', clockwise=False, bbox=[.5, .5, 1, 1]).reshape(-1, 2)
                uvmap = np.tile(uvmap, (nsplines, 1))
                mesh.add_geometry(corners=cap_inds.flatten(), faces=nprof, UVMap=uvmap)

            return mesh
        
        # ---------------------------------------------------------------------------
        # Add a set of splines of the same type, number of points and cyclic
        # ---------------------------------------------------------------------------

        def _cam_culling(curve, ctype, N, cyclic, resol):

            if ctype == BEZIER:
                curve = curve.to_poly()
                N = curve.splines.loop_total[0]
            else:
                curve.no_view()

            # No camera culling, this is simple
            if not camera_culling:
                return _to_mesh(curve, ctype, N, cyclic, resol)
            
            mesh = Mesh()
            
            # Visible points
            vis, dist = camera.visible_points(curve.points.position, radius=profile_size)
            vis = vis[:, camera.VISIBLE]

            # Spline is visible if any point is visible
            sel = np.any(vis.reshape(-1, N), axis=1)
            curve = curve[sel].no_view()
            if len(curve) == 0:
                return mesh
            
            # One p_size per spline
            p_size = dist[:, camera.SIZE].reshape(-1, N)[sel][:, 0]

            # On distance per spline
            dist = dist[:, camera.DISTANCE].reshape(-1, N)[sel][:, 0]

            # Curve length
            length = curve.length

            # profile_size is seen as p_size, ratio = p_size/profile_size
            #ratio = p_size/profile_size
            #app_length = ratio*length
            app_length = length*camera.pixels_per_meter(dist)

            # around 5 pixel per segment
            npix = np.clip(app_length/5, 4, N).astype(int)

            # We loop per npix
            new_Ns, rev_index = np.unique(npix, return_inverse=True)
            for i_new, new_N in enumerate(new_Ns):
                c = curve[rev_index == i_new].no_view().to_poly(new_N)
                mesh.join(_to_mesh(c, POLY, new_N, cyclic, resol, profile_scale=new_N/N))

            # Done !!!
            return mesh
        
        # ---------------------------------------------------------------------------
        # Main
        # ---------------------------------------------------------------------------
        
        mesh = Mesh()
        for _, m in self.for_each_bucket(_cam_culling):
            mesh.join(m)

        return mesh    
    
    # ====================================================================================================
    # To flat mesh
    # ====================================================================================================

    def solidify_flat(self, width=.1):

        from .mesh import Mesh

        mesh = Mesh()
        for loop_start, loop_total in zip(self.splines.loop_start, self.splines.loop_total):
            verts = self.points.position[loop_start:loop_start+loop_total]

            segms = verts[1:] - verts[:-1] # (n-1, 3)
            ags = np.unwrap(np.arctan2(segms[:, 1], segms[:, 0])) # (n-1,)
            diffs = ags[1:] - ags[:-1] # (n-2,)

            perps = np.empty(loop_total) # (n,)

            perps[0] = ags[0] + np.pi/2
            perps[1:-1] = ags[:-1]  + (np.pi + diffs)/2
            perps[-1] = ags[-1] + np.pi/2

            bot = np.array(verts)
            top = np.array(verts)

            cag = np.cos(perps)
            sag = np.sin(perps)

            scale = np.ones(loop_total)
            scale[1:-1] = np.minimum(10, 1/np.abs(np.cos(diffs/2)))

            w = width*scale
            cag *= w
            sag *= w

            bot[:, 0] += cag
            top[:, 0] -= cag
            bot[:, 1] += sag
            top[:, 1] -= sag

            pts = np.append(bot, np.flip(top, axis=0), axis=0)

            mesh.join(Mesh(points=pts, corners=np.arange(len(pts))))

        return mesh
        
    def set_spline2d_thickness(self,
            thickness = .1,
            mode = 0,      
            factor = 1.0,
            cuts = (.1, np.nan),
            inner_mode = None,
            inner_factor = None,
            inner_cuts = None,
            resolution = 12,
            offset = 0.0,
            start_thickness = 1,
            end_thickness = 1,
        ):
        """
        Transform the splines into flat 2D mesh..

        This method constructs a single polygonal outline from the current polyline by:
        (1) offsetting a copy to the "inner" side and resolving its corners,
        (2) offsetting a copy to the "outer" side and resolving its corners,
        then (3) reversing the outer side and concatenating both sides into one path.

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
        Mesh : splines transformed into faces
        """        

        from .maths import set_spline2d_thickness
        from .mesh import Mesh

        # ----------------------------------------------------------------------------------------------------
        # Normalize arguments
        # All arguments but cuts are broacasted
        # ----------------------------------------------------------------------------------------------------

        thickness = self.points.get(thickness, broadcast_shape=())

        mode = self.points.get(mode, broadcast_shape=())
        factor = self.points.get(factor, broadcast_shape=())
        cuts = self.points.get(cuts, broadcast_shape=None)

        inner_mode = mode if inner_mode is None else self.points.get(inner_mode, broadcast_shape=())
        inner_factor = factor if inner_factor is None else self.points.get(inner_factor, broadcast_shape=())
        inner_cuts = cuts if inner_cuts is None else self.points.get(inner_cuts, broadcast_shape=None)

        start_thickness = self.splines.get(start_thickness, broadcast_shape=())
        end_thickness = self.splines.get(end_thickness, broadcast_shape=())

        # ----------------------------------------------------------------------------------------------------
        # Loop on the splines
        # ----------------------------------------------------------------------------------------------------

        mesh = Mesh()
        for i_spline, (loop_start, loop_total) in enumerate(zip(self.splines.loop_start, self.splines.loop_total)):

            slc = slice(loop_start, loop_start + loop_total)

            points = self.points.position[slc]

            pts = set_spline2d_thickness(points,
                thickness = thickness[slc], 

                mode = mode[slc], 
                factor = factor[slc], 
                cuts = cuts,

                inner_mode = inner_mode[slc], 
                inner_factor = inner_factor[slc], 
                inner_cuts = inner_cuts,

                resolution = resolution,
                offset = offset, 
                cyclic = self.splines.cyclic[i_spline], 
                start_thickness = start_thickness[i_spline], 
                end_thickness = end_thickness[i_spline],
                )
            
            mesh.join(Mesh(points=pts, corners=np.arange(len(pts))))

        return mesh

    # ====================================================================================================
    # Primitives
    # ====================================================================================================

    @classmethod
    def bezier_circle(cls):
        """
        Unit circle approximated by 4 cubic Bezier arcs (cyclic).

        Returns
        -------
        Curve
            Bezier circle with pre-set handles.
        """
        return cls(
            points = [[-1.,  0.,  0.], [ 0.,  1.,  0.], [ 1.,  0.,  0.], [ 0., -1.,  0.]],
            curve_type = BEZIER, 
            cyclic = True,
            handle_left = [[-1., -0.55212522,  0.], [-0.55212522,  1.,  0.], [ 1.,  0.55212522,  0.], [ 0.55212522, -1.,  0.]],
            handle_right = [[-1.,  0.55212522,  0.], [ 0.55212522,  1.,  0. ],[ 1., -0.55212522,  0.], [-0.55212522, -1.,  0.]],
        )

    @classmethod
    def circle(cls, resolution=32, radius=1.):
        """
        Regular polygonal approximation of a circle (Poly, cyclic).

        Parameters
        ----------
        resolution : int, default=32
            Number of points.
        radius : float, default=1.0
            Circle radius.

        Returns
        -------
        Curve
        """
        ags = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        return cls(
            points = np.stack((radius*np.cos(ags), radius*np.sin(ags), np.zeros(resolution, float)), axis=-1),
            curve_type = POLY, 
            cyclic = True,
        )

    @classmethod
    def arc(
        cls,
        resolution: int = 16,
        radius: float = 1.0,
        start_angle: float = 0.0,
        sweep_angle: float = 7 * np.pi / 4,
        connect_center: bool = False,
        invert_arc: bool = False,
    ):
        """
        Build a polyline arc in the *XY* plane.

        The arc is sampled uniformly with `resolution` points between
        `start_angle` and `start_angle + sweep_angle`. If `invert_arc` is True,
        the parameterization is reversed (clockwise), producing the same locus
        but with swapped start/end angles. When `connect_center` is True, the
        center `(0, 0, 0)` is appended, and the spline is marked as cyclic to
        form a pie slice.

        Parameters
        ----------
        resolution : int, default=16
            Number of samples along the arc (min 2).
        radius : float, default=1.0
            Arc radius.
        start_angle : float, default=0.0
            Start angle in radians.
        sweep_angle : float, default=7π/4
            Signed sweep angle in radians.
        connect_center : bool, default=False
            If True, append the origin and mark the spline cyclic (pie slice).
        invert_arc : bool, default=False
            If True, reverse the arc direction (clockwise).

        Returns
        -------
        Curve
            A curve with one **POLY** spline sampled in the XY plane.

        See Also
        --------
        [Curve.circle][npblender.Curve.circle],
        [Curve.bezier_circle][npblender.Curve.bezier_circle]
        """

        ag0, ag1 = (start_angle + sweep_angle - 2*np.pi, start_angle) if invert_arc else (start_angle, start_angle + sweep_angle)
        ags = np.linspace(ag0, ag1, resolution)
        points = np.stack((radius*np.cos(ags), radius*np.sin(ags), np.zeros(resolution, float)), axis=-1)
        if connect_center:
            points = np.append(points, [(0, 0, 0)], axis=0)

        return cls(
            points = points, 
            curve_type = POLY,
            cyclic = connect_center,
            )

    @classmethod
    def bezier_segment(
        cls,
        resolution: int = 16,
        start: tuple = (-1, 0, 0),
        start_handle: tuple = (-0.5, 0.5, 0),
        end_handle: tuple = (0, 0, 0),
        end: tuple = (1, 0, 0),
    ):
        """
        Create a single two-point **Bezier** segment.

        Produces a Bezier spline with two control points located at `start`
        and `end`. The left/right handles are set from `start_handle` and
        `end_handle` (mirrored appropriately). The `resolution` argument is
        accepted for API symmetry but not used at construction; sampling and
        evaluation rely on the per-spline `resolution` attribute.

        Parameters
        ----------
        resolution : int, default=16
            Kept for API symmetry; not used during construction.
        start : (3,) float, default=(-1, 0, 0)
            Start control point.
        start_handle : (3,) float, default=(-0.5, 0.5, 0)
            Handle associated to the start point (as *right* handle).
        end_handle : (3,) float, default=(0, 0, 0)
            Handle associated to the end point (as *left* handle).
        end : (3,) float, default=(1, 0, 0)
            End control point.

        Returns
        -------
        Curve
            A curve with one **BEZIER** spline (open).

        Notes
        -----
        Handles are applied as:
        - `handle_right[0] = start_handle`
        - `handle_left[1]  = end_handle`
        The opposite handles are mirrored so that each handle is expressed
        in absolute coordinates.
        """

        points = np.array([start, end])
        return cls(
            points = points,
            curve_type = BEZIER,
            handle_left  = [2*points[0] - start_handle, end_handle],
            handle_right = [start_handle, 2*points[1] - end_handle],
        )

    @classmethod
    def line(
        cls,
        start: tuple = (0, 0, 0),
        end: tuple = (0, 0, 1),
        resolution: int = 2,
    ):
        """
        Build a straight polyline between two points.

        Generates `resolution` evenly spaced points from `start` to `end`
        (inclusive) and returns an open **POLY** spline.

        Parameters
        ----------
        start : (3,) float, default=(0, 0, 0)
            Line start point.
        end : (3,) float, default=(0, 0, 1)
            Line end point.
        resolution : int, default=2
            Number of samples along the line (min 2).

        Returns
        -------
        Curve
            A curve with one **POLY** spline containing `resolution` points.

        Examples
        --------
        ``` python
        c = Curve.line(start=(0, 0, 0), end=(1, 0, 0), resolution=5)
        print(len(c.points))
        ```
        """

        resolution = max(2, resolution)
        start = np.asarray(start)
        end = np.asarray(end)
        points = np.linspace(start, end, resolution)
        if len(points.shape) == 3:
            points = points.transpose(1, 0, 2)
        
        return cls(
            points = points.reshape(-1, 3),
            splines = resolution,
            curve_type = POLY,
        )

    @classmethod
    def spiral(
        cls,
        resolution: int = 32,
        rotations: float = 2.0,
        start_radius: float = 1.0,
        end_radius: float = 2.0,
        height: float = 2.0,
        reverse: bool = False,
    ):
        """
        Create a 3D polyline spiral in the XY plane with linear Z elevation.

        Samples a spiral with `1 + int(rotations * resolution)` points. The radius
        linearly interpolates from `start_radius` to `end_radius`. The angle evolves
        by `2π * rotations` (clockwise unless `reverse=True`). The Z coordinate is
        linearly distributed from `0` to `height`.

        Parameters
        ----------
        resolution : int, default=32
            Number of samples per full rotation.
        rotations : float, default=2.0
            Number of turns (can be fractional).
        start_radius : float, default=1.0
            Radius at the beginning of the spiral.
        end_radius : float, default=2.0
            Radius at the end of the spiral.
        height : float, default=2.0
            Final Z value (start is Z = 0).
        reverse : bool, default=False
            If True, swap the trigonometric direction of the spiral.

        Returns
        -------
        Curve
            A curve with a single **POLY** spline.

        Examples
        --------
        ``` python
        c = Curve.spiral(resolution=64, rotations=3.5, start_radius=0.5, end_radius=3.0, height=5.0)
        ```
        """
        count = 1 + int(rotations*resolution)
        # Reverse is strangely trigonometric!
        ags = np.linspace(0, 2*np.pi*rotations, count) * (1 if reverse else -1)
        rs  = np.linspace(start_radius, end_radius, count)
        return cls(
            points = np.stack((rs*np.cos(ags), rs*np.sin(ags), np.linspace(0, height, count)), axis=-1),
            curve_type = POLY,
        )

    @classmethod
    def quadratic_bezier(
        cls,
        resolution: int = 16,
        start: tuple = (-1, 0, 0),
        middle: tuple = (0, 2, 0),
        end: tuple = (1, 0, 0),
    ):
        """
        Quadratic Bézier segment (not implemented).

        Intended to create a single quadratic Bézier curve defined by the control
        points `start`, `middle`, and `end`. Currently **not implemented**.

        Parameters
        ----------
        resolution : int, default=16
            Suggested sampling resolution (unused in the current implementation).
        start : (3,) float, default=(-1, 0, 0)
            Start control point.
        middle : (3,) float, default=(0, 2, 0)
            Middle (control) point.
        end : (3,) float, default=(1, 0, 0)
            End control point.

        Raises
        ------
        Exception
            Always raised with the message "Not implemented yet".

        Examples
        --------
        ``` python
        try:
            c = Curve.quadratic_bezier()
        except Exception as e:
            print(e)  # "Not implemented yet"
        ```
        """
        raise Exception(f"Not implemented yet")

    @classmethod
    def quadrilateral(cls, width: float = 2.0, height: float = 2.0):
        """
        Axis-aligned rectangle in the XY plane (closed polyline).

        Builds a cyclic **POLY** spline with four vertices:
        `(-w/2, -h/2) → (w/2, -h/2) → (w/2, h/2) → (-w/2, h/2)`.

        Parameters
        ----------
        width : float, default=2.0
            Rectangle width along X.
        height : float, default=2.0
            Rectangle height along Y.

        Returns
        -------
        Curve
            A curve with one cyclic **POLY** spline.

        Examples
        --------
        ``` python
        c = Curve.quadrilateral(width=1.0, height=0.5)
        ```
        """

        return cls(
            points = [(-width/2, -height/2, 0), (width/2, -height/2, 0), (width/2, height/2, 0), (-width/2, height/2, 0)],
            curve_type = POLY,
            cyclic = True,
        )

    @classmethod
    def star(
        cls,
        points: int = 8,
        inner_radius: float = 1.0,
        outer_radius: float = 2.0,
        twist: float = 0.0,
    ):
        """
        Create a star polygon (alternating outer/inner vertices) in the *XY* plane.

        Builds a cyclic **POLY** spline with `2 * points` vertices alternating
        between `outer_radius` and `inner_radius`. The inner vertices are rotated
        by `π / points + twist` to control the star lobes' alignment.

        Parameters
        ----------
        points : int, default=8
            Number of star tips (minimum 3).
        inner_radius : float, default=1.0
            Radius of inner vertices.
        outer_radius : float, default=2.0
            Radius of outer vertices (tips).
        twist : float, default=0.0
            Additional rotation (radians) applied to inner vertices.

        Returns
        -------
        Curve
            A curve with one cyclic **POLY** spline forming a star.

        See Also
        --------
        [Curve.circle][npblender.Curve.circle],
        [Curve.arc][npblender.Curve.arc]
        """

        points = max(3, points)
        ag = np.linspace(0, 2*np.pi, points, endpoint=False)

        vs = np.zeros((points, 2, 3), float)
        vs[:, 0, 0] = np.cos(ag)
        vs[:, 0, 1] = np.sin(ag)
        vs[:, 1, :2] = inner_radius * vs[:, 0, :2]
        vs[:, 0, :2] *= outer_radius

        rot = np.pi/points + twist
        M = np.zeros((2, 2), float)
        M[0, 0] = np.cos(rot)
        M[1, 1] = M[0, 0]
        M[1, 0] = np.sin(rot)
        M[0, 1] = -M[1, 0]

        vs[:, 1, :2] = np.einsum('...ij, ...j', M[None], vs[:, 1, :2])

        return cls(
            points = np.reshape(vs, (2*points, 3)),
            curve_type = POLY,
            cyclic = True,
        )
    
    @classmethod
    def xyfunction(
        cls,
        func,
        x0: float = 0.0,
        x1: float = 1.0,
        resolution: int = 100,
        materials=None,
    ):
        """
        Sample a 2D function y = f(x) as a polyline in the XY plane.

        Evaluates `y = func(x)` for `x` uniformly spaced in `[x0, x1]` and creates a
        **POLY** spline with points `(x, y, 0)`.

        Parameters
        ----------
        func : callable
            A function mapping an array of X values to Y values (vectorized).
        x0 : float, default=0.0
            Start of the X interval.
        x1 : float, default=1.0
            End of the X interval.
        resolution : int, default=100
            Number of samples in `[x0, x1]`.
        materials : str or sequence of str or None, optional
            Optional material(s) to attach to the curve.

        Returns
        -------
        Curve
            A curve with one **POLY** spline sampled from `func`.

        Examples
        --------
        ``` python
        import numpy as np

        def f(x):
            return np.sin(2 * np.pi * x)

        c = Curve.xyfunction(f, x0=0.0, x1=1.0, resolution=200)
        ```
        """

        x = np.linspace(x0, x1, resolution, dtype=bfloat)
        y = func(x)
        return cls(points=np.stack((x, y, np.zeros_like(x)), axis=-1), materials=materials)

