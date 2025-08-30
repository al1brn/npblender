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

from contextlib import contextmanager
import numpy as np

import bpy

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
        """ Curve Geometry.

        Arguments
        ---------
            - points (array of vectors = None) : the vertices
            - splines (array of ints = None) : sizes
            - curve_type (array onf ints) : curve types
            - materials (str or list of strs = None) : list of materials used in the geometry
            - **attrs (dict) : other geometry attributes
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
        self.add_splines(points, splines, curve_type=curve_type, **attrs)

    # ----------------------------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------------------------

    def get_points_selection(self):
        """ Get selection on points associated to the splines.

        If the Curve is not a view on splines, if return [:], otherwise
        it returns a selection on the points.

        Returns
        -------
            - sel (slice) : indices of splines points
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
        Materialize a view-curve into a standalone curve (deep copy of splines and points),
        preserving all point-domain fields (handles, custom attrs, ...).
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
        self.no_view()
        return {
            'geometry':     'Curve',
            'materials' :   self.materials,
            'points':       self.points.to_dict(),
            'splines':      self.splines.to_dict(),
            }

    @classmethod
    def from_dict(cls, d):
        curve = cls()
        curve.materials  = d['materials']
        curve.points     = ControlPoint.from_dict(d['points'])
        curve.splines    = Spline.from_dict(d['splines'])
        return curve
    
    # ====================================================================================================
    # Clear the geometry
    # ====================================================================================================

    def clear(self):
        """ Clear the geometry
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
        """ Create a Curve from another curve.

        Arguments
        ---------
            - other (Mesh) : the mesh to copy
            - points (selector = None) : points selection
            - splines (selector = None) : splines selection

        Returns
        -------
            - Curve
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
        """ Capture the data of another Curve.

        Arguments
        ---------
            - other (Curve) : the mesh to capture

        Returns
        -------
            - self
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
        """ Initialize the geometry from a Blender Curve

        Arguments
        ---------
            - data (Blender Curve instance) : the curve to load
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
        """ Initialize the geometry from a Blender Curve

        Arguments
        ---------
            - data (Blender Curve instance) : the curve to load
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
        """ Create a Curve from an existing curve.

        Arguments
        ---------
            - obj (str or Blender object) : the object to initialize from
            - evaluated (bool = False) : object modified by the modifiers if True, raw vertices otherwise

        Returns
        -------
            - Curve
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
        """ Create or update a blender object.

        The method 'to_object' creates the whole geometry. It creates a new object if it doesn't already exist.
        If the object exists, it must be a curve, there is no object type conversion.

        Once the object is created, use the method 'update_object' to change the vertices.

        Arguments
        ---------
            - obj (str or Blender object) : the object the create

        Returns
        -------
            - Blender curve object
        """
        curve = blender.create_curve_object(obj, collection=collection)
        self.to_curve_data(curve.data)

        return curve
    
    # ----------------------------------------------------------------------------------------------------
    # Blender data context manager
    # ----------------------------------------------------------------------------------------------------
    
    @contextmanager
    def blender_data(self, readonly=False):
        """ Acces to Blender Curve API.

        Transfer the geometry to a temporay Blender Curve.
        The example below use a blender Mesh to get the normals.

        ``` python
        curve = Curve.Spiral()

        with curve.blender_data() as bcurve:
            print("Number of points", len(bcurve.splines[0].points))

        # > Number of points 65
        ```

        Arguments
        ---------
            - readonly (bool=False) : don't read back the geometry if not modified

        Returns
        -------
            - Blender Mesh
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
                                     "- points:  {self.points.all_names}\n"
                                     "- splines: {self.splines.all_names}\n"
                                     )

            if count > 1:
                raise AttributeError(f"Curve attribute '{k}' is ambigous, it belongs to more than one domain (count)."
                                     "- points:  {self.points.all_names}\n"
                                     "- splines: {self.splines.all_names}\n"
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


    def _compute_bezier_handles_OLD(cls, points, cyclic=False):

        single = len(points.shape) == 2
        if single:
            points = points[None]

        # ----- Add the first point at the end if cyclic

        count = points.shape[1]
        if cyclic:
            count += 1
            points = np.append(points, [points[:,0]], axis=1)

        # ----- Compute the control points

        der = np.empty_like(points)
        der[:, 1:-1] = points[:, 2:]  - points[:, :-2]
        der[:,  0]   = (points[:,  1] - points[:, 0])/2
        der[:, -1]   = (points[:, -1] - points[:, -2])/2

        nrm = np.linalg.norm(der, axis=-1)
        nrm[abs(nrm) < 0.001] = 1.
        der /= nrm[:, :, None]

        dists = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1)[:, :, None]

        # Left handles
        lefts = points.copy()
        lefts[:, 1:] -= der[:, 1:]*dists/3
        lefts[:, 0]  -= der[:, 0]*dists[:, 0]/3

        # Right handles
        rights = points.copy()
        rights[:, :-1] += der[:, :-1]*dists/3
        rights[:, -1]  += der[:, -1]*dists[:, -1]/3

        # ----- Returns the result

        if cyclic:
            if single:
                return np.array(lefts[0, :-1]), np.array(rights[0, :-1])
            else:
                return np.array(lefts[:, :-1]), np.array(rights[:, :-1])
        else:
            if single:
                return lefts[0], rights[0]
            else:
                return lefts, rights
    
    # ----------------------------------------------------------------------------------------------------
    # Add Bezier
    # ----------------------------------------------------------------------------------------------------

    def add_bezier(self,
                   points, 
                   splines = None, 
                   handle_left = None, 
                   handle_right = None, 
                   cyclic = False,
                   **attrs):
        """ Add Bezier splines

        The arguments 'splines' gives the length(s) of the bezier spline(s). If None:
        - the number of points is taken (one spline is added)
        - points.shape[1] is taken if the shape of points is (m, , 3)

        handle_left and handle_right must have the same shape as points if provided.
        If they aren't provided, they are computed.
        
        Arguments
        ---------
        - points (array (n, 3) or (m, n, 3) of vectors) : the points of the curves
        - splines (int or array of ints = None) : spline or splines length
        - handle_left (same shape as points = None) : left handles
        - handle_right (same shape as points = None) : right handles
        - cyclic (bool = False) : whether the curve is cyclic or not
        - attrs (dict) : spline and/or points attributes

        Returns
        -------
        - dict ('points': added points indices, 'splines': added splines indices)
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

    def add_poly(self,
                   points, 
                   splines = None,
                   curve_type = POLY,
                   cyclic = False,
                   w = 1.,
                   **attrs):
        """ Add Poly or Nurbs splines

        The arguments 'splines' gives the length(s) of the spline(s). If None:
        - the number of points is taken (one spline is added)
        - points.shape[1] is taken if the shape of points is (m, , 3)

        Non bezeier splines use 4D points. If the provided vectors are 4D, the argument w
        is ignored.

        Arguments
        ---------
        - points (array (n, 3 or 4) or (m, n, 3 or 4) of vectors) : the points of the curves
        - splines (int or array of ints = None) : spline or splines length
        - cyclic (bool = False) : whether the curve is cyclic or not
        - w (float) : w value, ignored if points are 4D
        - attrs (dict) : spline and/or points attributes

        Returns
        -------
        - dict ('points': added points indices, 'splines': added splines indices)
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
        """ Add splines
        """
        self.no_view()

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
        """Delete points.

        Arguments
        ---------
            points : points selection, optional
                Points indices to delete directly.
            splines : splines selection, optional
                Splines owning vertices to delete.
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
        """Delete splines.

        Arguments
        ---------
            splines : splines selection, optional
                Splines owning vertices to delete.
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
        """ Join other Curves.

        Arguments
        ---------
            - others (Curve) : the curves to append
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
        """ Duplicate the geometry.

        Multiplying is a way to efficiently duplicate the geometry a great number of times.
        Once duplicated, the points can be reshapped to address each instance individually.

        ``` python
        count = 16

        cube = Mesh.Cube() * count

        # Shape the points as 16 blocks of 8 vertices
        points = np.reshape(cube.points.position, (16, 8, 3))

        # Place the cubes in a circle
        ags = np.linspace(0, 2*np.pi, count, endpoint=False)
        points[..., 0] += 6 * np.cos(ags)[:, None]
        points[..., 1] += 6 * np.sin(ags)[:, None]

        cube.to_object("Cubes")
        ```

        Arguments
        ---------
            - count (int=10) : number of instances
            - attributes (name=value) : value for named attributes

        Returns
        -------
            - Mesh
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
        return splinemaths.length_curve(self)

    def evaluate(self, t):
        return splinemaths.evaluate_curve(self, t)

    def tangent(self, t, normalize=True):
        return splinemaths.tangent_curve(self, t, normalize=normalize)

    def sample_attributes(self, t, names=None, cubic=False):
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
        """Explicit alias for subsetting splines; same semantics as curve[indices]."""
        return self[indices]   
    
    # ====================================================================================================
    # Loop on buckets
    # ====================================================================================================

    def for_each_bucket(self, func):
        """
        Iterate over homogeneous spline buckets and call `func` once per bucket.

        Bucketing key:
        (curve_type, N, cyclic, resolution)    # resolution only relevant for BEZIER

        Parameters
        ----------
        func : callable
            Signature:
                func(curve, curve_type, loop_total, is_cyclic, resolution=None) -> any
            - `curve` is a Curve view on self .
            - `curve_type` is the spline type.
            - `loop_total` is the number of points in the spline.
            - `is_cyclic` is True if the spline is cyclic.
            - `resolution` is the resolution of the loop, or None if the spline is not Bezier.

        Yields
        ------
        (indices, result)
            indices : list[int]
                Spline indices for this bucket, in stacking order.
            result : any
                Whatever `func` returned for this bucket.
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
        """ Convert the splines to Poly splines.

        If resolution is None:
        - `splines.resolution` is used to split Bezier segments
        - poly lines are left unchanged

        If resolution is not None, all splines are resampled using this value
        
        Arguments:
        ---------
            resolution (int):          Poly resolution, use spline resolution for Bezier if None.
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
        """Convert splines to Bezier splines (OPEN storage).

        Parameters
        ----------
        control_count : int or None
            Number of anchor points to generate per spline. If None, keep current loop_total.
            For cyclic splines, anchors are sampled on [0,1) (no duplicate). For non-cyclic, on [0,1].
        resolution : int
            Per-segment resolution to write into `splines.resolution` (>=1).
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
        """ > Transform curve to mesh

        If profile is None, the mesh contains only edges
        otherwise they are ignored

        Arguments
        ---------
        - profile (Curve = None) : profile curve
        - caps (bool = True) : use caps
        - use_radius (bool = True) : use the radius as a scale for profile

        Returns
        -------
        - Mesh
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
    # Primitives
    # ====================================================================================================

    @classmethod
    def bezier_circle(cls):
        return cls(
            points = [[-1.,  0.,  0.], [ 0.,  1.,  0.], [ 1.,  0.,  0.], [ 0., -1.,  0.]],
            curve_type = BEZIER, 
            cyclic = True,
            handle_left = [[-1., -0.55212522,  0.], [-0.55212522,  1.,  0.], [ 1.,  0.55212522,  0.], [ 0.55212522, -1.,  0.]],
            handle_right = [[-1.,  0.55212522,  0.], [ 0.55212522,  1.,  0. ],[ 1., -0.55212522,  0.], [-0.55212522, -1.,  0.]],
        )

    @classmethod
    def circle(cls, resolution=32, radius=1.):
        ags = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        return cls(
            points = np.stack((radius*np.cos(ags), radius*np.sin(ags), np.zeros(resolution, float)), axis=-1),
            curve_type = POLY, 
            cyclic = True,
        )

    @classmethod
    def arc(cls, resolution=16, radius=1., start_angle=0., sweep_angle=7*np.pi/4, connect_center=False, invert_arc=False):
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
    def bezier_segment(cls, resolution=16, start=(-1, 0, 0), start_handle=(-.5, .5, 0), end_handle=(0, 0, 0), end=(1, 0, 0)):
        points = np.array([start, end])
        return cls(
            points = points,
            curve_type = BEZIER,
            handle_left  = [2*points[0] - start_handle, end_handle],
            handle_right = [start_handle, 2*points[1] - end_handle],
        )

    @classmethod
    def line(cls, start=(0, 0, 0), end=(0, 0, 1), resolution=2):
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
    def spiral(cls, resolution=32, rotations=2., start_radius=1., end_radius=2., height=2., reverse=False):
        count = 1 + int(rotations*resolution)
        # Reverse is strangely trigonometric!
        ags = np.linspace(0, 2*np.pi*rotations, count) * (1 if reverse else -1)
        rs  = np.linspace(start_radius, end_radius, count)
        return cls(
            points = np.stack((rs*np.cos(ags), rs*np.sin(ags), np.linspace(0, height, count)), axis=-1),
            curve_type = POLY,
        )

    @classmethod
    def quadratic_bezier(cls, resolution=16, start=(-1, 0, 0), middle=(0, 2, 0), end=(1, 0, 0)):
        raise Exception(f"Not implemented yet")

    @classmethod
    def quadrilateral(cls, width=2., height=2.):
        return cls(
            points = [(-width/2, -height/2, 0), (width/2, -height/2, 0), (width/2, height/2, 0), (-width/2, height/2, 0)],
            curve_type = POLY,
            cyclic = True,
        )

    @classmethod
    def star(cls, points=8, inner_radius=1, outer_radius=2, twist=0.):

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
    def xyfunction(cls, func, x0=0., x1=1., resolution=100, materials=None):
        x = np.linspace(x0, x1, resolution, dtype=bfloat)
        y = func(x)
        return cls(points=np.stack((x, y, np.zeros_like(x)), axis=-1), materials=materials)

