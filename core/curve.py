#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module

Created on Fri Nov 10 11:50:13 2023

@author: alain.bernard
@email: alain@ligloo.net

-----

Curve geometry.

"""

from contextlib import contextmanager
import numpy as np

from . constants import SPLINE_TYPES, BEZIER, POLY, NURBS
from . constants import bfloat, bint, bbool
from . constants import FillCap
from . import blender
from . maths import BSplines, Bezier, Poly, Nurbs
from . maths import Transformation, Quaternion, Rotation
from . maths.topology import grid_corners, grid_uv_map, fans_corners, disk_uv_map
from . maths.topology import border_edges, edges_between, row_edges, col_edges

from . geometry import Geometry
from . domain import SplinePointDomain, SplineDomain


import bpy


DATA_TEMP_NAME = "npblender_TEMP"

# ====================================================================================================
# Curve, made of one or more splies
# ====================================================================================================

class Curve(Geometry):

    def __init__(self, points=None, splines=None, curve_type=POLY, materials=None, **attrs):
        """ Curve Geometry.

        Arguments
        ---------
            - points (array of vectors = None) : the vertices
            - splines (array of ints = None) : sizes
            - curve_type (array onf ints) : curve types
            - materials (str or list of strs = None) : list of materials used in the geometry
            - **attrs (dict) : other geometry attributes
        """

        # ----- Initialize an empty geometry

        self.points  = SplinePointDomain()
        self.splines = SplineDomain()

        # ----- The materials

        if materials is None:
            self.materials = []
        elif isinstance(materials, str):
            self.materials = [materials]
        else:
            self.materials = [mat for mat in materials]

        # ----- Add geometry

        self.add_splines(points, splines, curve_type=curve_type, **attrs)

    # ----------------------------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------------------------

    def check(self, halt=True):
        return self.splines.check(halt=halt)

    def __str__(self):
        return f"<Curve: points {self.points.shape}, splines {self.splines.shape}>"

    def __repr__(self):
        s = "Curve:\n   " + "\n   ".join([str(self.points), str(self.splines)])
        return s

    # ====================================================================================================
    # Serialization
    # ====================================================================================================

    def to_dict(self):
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
        curve.points     = SplinePointDomain.from_dict(d['points'])
        curve.splines    = SplineDomain.from_dict(d['splines'])
        return curve
    
    # ====================================================================================================
    # Clear the geometry
    # ====================================================================================================

    def clear(self):
        """ Clear the geometry
        """

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
        curve.points  = SplinePointDomain(other.points,  mode='COPY')
        curve.splines = SplineDomain(other.splines, mode='COPY')

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
                splines = [splines]*nsplines
                points = points.reshape(nsplines, splines, points.shape[-1])

            elif hasattr(splines, '__len__'):
                if np.sum(splines) != len(points):
                    raise ValueError(f"The sum of spline lengths {np.sum(splines)} doesn't match the number of points {len(points)}.")

        elif len(points.shape) == 3:
            if splines is not None and splines != points.shape[1]:
                raise ValueError(
                    f"Points arguments is an array of {len(points)} splines of {points.shape[1]} points each, "
                    "splines argument must be None or {points.shape[1]}, not {splines}")
            
            splines = points.shape[1]

        else:
            raise ValueError(f"Points argument must be an array of points, shape {points.shape} is not valid.")

        return points, splines
    
    # ----------------------------------------------------------------------------------------------------
    # Compute Bezier handles
    # ----------------------------------------------------------------------------------------------------

    def _compute_bezier_handles(cls, points, cyclic=False):

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
                h_left, h_right = self._compute_bezier_handles(points, cyclic=disp_attrs.get('cyclic', False))
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
                    h_left, h_right = self._compute_bezier_handles(pts, cyclic=cycl)
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

        for other in others:

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
        return cls(
            points = np.linspace(start, end, resolution),
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
    def quadrilaterail(cls, width=2., height=2.):
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

    # ====================================================================================================
    # Field of vectors
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Field line
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def field_line(cls, field_func, start_point, max_len=10., prec=.01, sub_steps=10):

        pts = [start_point]
        rs  = [np.linalg.norm(field_func(start_point))]
        p   = np.array(start_point)
        l   = 0.
        for _ in range(10000):
            for _ in range(sub_steps):

                # ----- Vector at current location
                v0 = field_func(p)

                # ----- Precision along this vector
                norm  = np.sqrt(np.dot(v0, v0))
                factor = prec/norm
                v0 *= factor

                # ----- Average with target vector for more accurracy
                v1 = field_func(p + v0)*factor
                v = (v0 + v1)/2

                # ----- Next point
                p += v

            # ----- Segment length

            v = p - pts[-1]
            l += np.sqrt(np.dot(v, v))

            # ----- Add a new point

            pts.append(np.array(p))
            rs.append(norm)

            # ----- Done if loop or max_len is reached

            v = p - start_point
            cyclic = np.sqrt(np.dot(v, v)) < prec*(sub_steps-1)
            if cyclic or l >= max_len:
                pts.append(np.array(start_point))
                break

        if cyclic:
            pts.pop()

        return cls(pts, curve_type=POLY, cyclic=cyclic, radius=rs)
    
    # ----------------------------------------------------------------------------------------------------
    # Field lines
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def field_lines_OLD(cls, field_func, start_points,
        backwards=False, max_length=None, length_scale=None, end_points=None, zero=1e-6, max_points=1000,
        precision=.1, sub_steps=10, seed=0, **kwargs):

        """ Build splines showing lines of field

        Arguments :
        -----------
            - field_func (function of template (array of vectors, **kwargs) -> array of vectors) : the field function
            - start_points (array of vectors) : lines starting points
            - backwards (bool = False) : build lines backwards
            - max_length (float = None) : max line lengths
            - length_scale (float = None) : line length scale if random length scale around central value
            - end_points (array of vectors) : points where lines must end
            - zero (float = 1e-6) : value below which the field is null
            - max_points (int = 1000) : max number of points per spline
            - precision (float = 0.1) : step length
            - sub_steps (int = 10) : number of sub steps
        """

        splines = field.field_lines(field_func, start_points,
            backwards       = backwards,
            max_length      = max_length,
            length_scale    = length_scale,
            end_points      = end_points,
            zero            = zero,
            max_points      = max_points,
            precision       = precision,
            sub_steps       = sub_steps,
            seed            = seed,
            **kwargs)

        return cls(**splines)

        curves = cls()
        for avects, cyclic in lines:
            if len(avects) <= 1:
                continue
            curves.add(avects.co, curve_type='POLY', cyclic=cyclic, radius=avects.radius, tilt=avects.color)

        return curves
    
    # ----------------------------------------------------------------------------------------------------
    # Lines of electric field
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def electric_field_lines_OLD(cls, charge_locations, charges=1., field_color=True,
                           count=100, start_points=None, plane=None, plane_center=(0, 0, 0),
                           frag_length=None, frag_scale=None, max_points=1000,
                           precision=.1, sub_steps=10, seed=None):

        """ Create lines of field for a vector field generated by charges, typically an electric field.

        Arguments:
        ----------
            - charge_locations (array (n, 3) of vectors) : where the charges are located
            - charges (float or array (n) of floats = 1) : the charges
            - field_color (bool = True) : manage the field_color attribute
            - count (int = 100) : number of lines to create. Overriden by len(start_points) if not None
            - start_points (array (s, 3) of vectors = None) : the starting points to compute the lines from
            - plane (vector = None) = restrict start points to a plane defined by its perpendicular
            - plane_center (vector = (0, 0, 0)) : center of the plane
            - frag_length (float=None) : length of fragments, None for full lines
            - frag_scale (float=None) : length distribution scale
            - precision (float = .1) : step precision
            - sub_steps (int=10) : number of steps between two sucessive points of the lines
        """

        # ----------------------------------------------------------------------------------------------------
        # Field function

        poles = AttrVectors(charge_locations, charge=charges)
        field_func = lambda points: field.electric_field(points,
                            locations=poles.co, charges=poles.charge)

        # ----------------------------------------------------------------------------------------------------
        # Starting points

        rng = np.random.default_rng(seed=seed)
        n_charges = len(poles)

        if start_points is None:
            backwards = rng.choice([True, False], count)
            if frag_length is None:
                if plane is None:
                    start_points, _ = distribs.sphere_dist(radius=precision, count=count, seed=rng.integers(1<<63))
                else:
                    start_points, _ = distribs.circle_dist(radius=precision, count=count, seed=rng.integers(1<<63))
                    start_points = rotate_xy_into_plane(start_points, plane=plane, origin=plane_center)

                inds = rng.integers(0, n_charges, count)
                start_points += poles.co[inds]
                backwards[:] = poles.charge[inds] < 0

            else:
                center = np.average(poles.co, axis=0)
                bbox0, bbox1 = np.min(poles.co, axis=0), np.max(poles.co, axis=0)
                radius = 1.3*max(np.linalg.norm(bbox1 - center), np.linalg.norm(bbox0 - center))

                if plane is None:
                    start_points, _ = distribs.ball_dist(radius=radius, count=count, seed=rng.integers(1<<63))
                    start_points += center
                else:
                    start_points, _ = distribs.disk_dist(radius=radius, count=count, seed=rng.integers(1<<63))
                    start_points = rotate_xy_into_plane(start_points, plane=plane, origin=plane_center)

        else:
            if len(np.shape(start_points)) == 1:
                count = 1
            else:
                count = len(start_points)
            backwards = rng.choice([True, False], count)

        # ----------------------------------------------------------------------------------------------------
        # Full lines if frag_length is None

        full_lines = frag_length is None
        if full_lines:
            backwards[:] = False

        # ----------------------------------------------------------------------------------------------------
        # Field lines

        lines = field.field_lines(field_func,
            start_points    = start_points,
            backwards       = backwards,
            max_length      = frag_length,
            length_scale    = frag_scale,
            end_points      = charge_locations,
            max_points      = max_points,
            precision       = precision,
            sub_steps       = sub_steps,
            seed            = rng.integers(1 << 63),
            )

        # ----------------------------------------------------------------------------------------------------
        # Twice il full lines

        if full_lines:

            # ----- Exclude cyclic lines which are done

            open_lines = np.logical_not(lines['cyclic'] )

            # ----- Backwards lines

            backwards[:] = True
            back_lines = field.field_lines(field_func,
                start_points    = start_points[open_lines],
                backwards       = backwards[open_lines],
                max_length      = frag_length,
                length_scale    = frag_scale,
                end_points      = charge_locations,
                max_points      = max_points,
                precision       = precision,
                sub_steps       = sub_steps,
                seed            = rng.integers(1 << 63),
                )

            # ----- Merge the two dictionnaries

            all_lines = {'types':   list(lines['types']) + list(back_lines['types']),
                         'cyclic':  list(lines['cyclic']) + list(back_lines['cyclic']),
                         'splines': lines['splines'] + back_lines['splines'],
                        }
            lines = all_lines

        return cls(**lines)

    # ----------------------------------------------------------------------------------------------------
    # Lines of magnetic field
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def magnetic_field_lines_OLD(cls, magnet_locations, moments=(1, 0, 0), field_color=True,
                           count=100, start_points=None, min_width=.3, plane=None, plane_center=(0, 0, 0),
                           frag_length=None, frag_scale=None, max_points=1000,
                           precision=.1, sub_steps=10, seed=None):

        """ Create lines of field for a vector field generated by bipoles, typically an magnetic field.

        Arguments:
        ----------
            - magnet_locations (array (n, 3) of vectors) : where the bipoles are located
            - moments (vector or array (n) of vectors = (1, 0, 0)) : the moments of the magnets
            - field_color (bool = True) : manage the field_color attribute
            - count (int = 100) : number of lines to create. Overriden by len(start_points) if not None
            - start_points (array (s, 3) of vectors = None) : the starting points to compute the lines from
            - min_width (float = .3) : min width for volume generation when magnet locations are in a plane
            - plane (vector = None) = restrict start points to a plane defined by its perpendicular
            - plane_center (vector = (0, 0, 0)) : center of the plane
            - frag_length (float=None) : length of fragments, None for full lines
            - frag_scale (float=None) : length distribution scale
            - precision (float = .1) : step precision
            - sub_steps (int=10) : number of steps between two sucessive points of the lines
        """

        # ----------------------------------------------------------------------------------------------------
        # Field function

        magnets = AttrVectors(magnet_locations, moment=moments)
        field_func = lambda points: field.magnetic_field(points,
                            locations=magnets.co, moments=magnets.moment)

        # ----------------------------------------------------------------------------------------------------
        # Starting points

        rng = np.random.default_rng(seed=seed)
        n_magnets = len(magnets)

        backwards = rng.choice([True, False], count)
        if start_points is None:
            if frag_length is None:
                if plane is None:
                    start_points, _ = distribs.sphere_dist(radius=precision*10, count=count, seed=rng.integers(1<<63))
                else:
                    start_points, _ = distribs.circle_dist(radius=precision*10, count=count, seed=rng.integers(1<<63))
                    start_points = rotate_xy_into_plane(start_points, plane=plane, origin=plane_center)

                inds = rng.integers(0, n_magnets, count)
                mag_locs = magnets.co[inds]
                backwards[:] = np.einsum('...i, ...i', start_points, magnets.moment[inds]) < 0
                start_points += mag_locs

            else:
                center = np.average(magnets.co, axis=0)
                bbox0, bbox1 = np.min(magnets.co, axis=0), np.max(magnets.co, axis=0)
                radius = 1.3*max(1., max(np.linalg.norm(bbox1 - center), np.linalg.norm(bbox0 - center)))

                if plane is None:
                    dims = np.maximum(bbox1 - bbox0, (min_width, min_width, min_width))
                    center = (bbox0 + bbox1)/2
                    bbox0, bbox1 = center - 1.3*dims, center + 1.3*dims

                    start_points, _ = distribs.cube_dist(corner0=bbox0, corner1=bbox1, count=count, seed=rng.integers(1<<63))
                else:
                    start_points, _ = distribs.disk_dist(radius=radius, count=count, seed=rng.integers(1<<63))
                    start_points = rotate_xy_into_plane(start_points, plane=plane, origin=plane_center)

        else:
            if len(np.shape(start_points)) == 1:
                count = 1
            else:
                count = len(start_points)

        # ----------------------------------------------------------------------------------------------------
        # Field lines

        lines = field.field_lines(field_func,
            start_points    = start_points,
            backwards       = backwards,
            max_length      = frag_length,
            length_scale    = frag_scale,
            end_points      = magnet_locations,
            max_points      = max_points,
            precision       = precision,
            sub_steps       = sub_steps,
            seed            = rng.integers(1 << 63),
            )

        return cls(**lines)
    
    # ====================================================================================================
    # To mesh
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # with profile
    # ----------------------------------------------------------------------------------------------------

    def to_mesh(self, radius=.1, resol=12, caps=True, profile=None, use_radius=True):
        """ > Transform curve to mesh

        if profile is None, radius and resol are used to build a round profile
        otherwise they are ignored

        Arguments
        ---------
        - radius (float = .1)) : circle profile radius
        - resol (int = 12) : circle profile resolution
        - caps (bool = True) : use caps
        - profile (Curve = None) : curve profile
        - use_radius (bool = True) : use the radius as a scale for profile

        Returns
        -------
        - Mesh
        """

        from mesh import Mesh

        # ----------------------------------------------------------------------------------------------------
        # First, let's build the vertices of the profile
        
        if profile is None:
            verts = Curve.circle(resolution=resol, radius=radius).points.position

        elif isinstance(profile, Geometry):
            verts = profile.points.position

        else:
            verts = np.asarray(profile)
            if len(verts.shape) != 2 and verts.shape[-1] != 0:
                raise ValueError(f"Invalid profile shape {verts.shape}")

        # ----------------------------------------------------------------------------------------------------
        # Let's loop on the splines

        mesh = Mesh(materials=self.materials)
        for i_spline, func in enumerate(self.splines.functions):

            cyclic = self.splines.cyclic[i_spline]

            # Curve back bone points

            bbone = func.sample_points
            if cyclic:
                bbone = bbone[:-1]

            # ----- No profile : points and edges

            if profile is None:
                meshed = Mesh(verts=bbone, materials=self.materials)

                # Create edges
                _ = meshed.edges
                n = len(meshed.points)
                meshed.edges.add(n - 1, vertex0=np.arange(0, n - 1), vertex1=np.arange(1, n))
                if cyclic:
                    meshed.edges.add(1, vertex0=n-1, vertex1=0)

            # ----- Profile exists
            else:

                # Locate, scale and rotate the rings

                radius = func.sample_value(self.splines[i_spline].get_points().radius)(np.linspace(0, 1, len(bbone)))

                transfos = tracker(func.tangent(np.linspace(0, 1, len(bbone), endpoint=not cyclic)), 'Z', 'X')
                transfos.position = bbone
                transfos.scale    = np.ones((len(bbone), 3))*radius[:, None]

                cyl_verts = transfos[:, None] @ np.resize(verts, (len(bbone),) + np.shape(verts))

                # Cylinder / torus topology
                if cyclic:
                    cyl = topology.torus(x_count=len(verts), y_count=len(bbone), verts=cyl_verts)
                else:
                    cyl = topology.cylinder(x_count=len(verts), y_count=1, z_count=len(bbone), caps='NGON' if caps else 'NONE', verts=cyl_verts)

                # Create the mesh
                meshed = Mesh(materials=self.materials, **cyl)

            # ----- Transfer the attributes

            for name in self.splines.attributes.names:
                if name in ['loop_start', 'loop_total', 'curve_type', 'cyclic', 'resolution']:
                    continue

                meshed.points.attributes.copy_attribute(self.splines.attributes, name)
                setattr(meshed.points, name, getattr(self.splines, name))

            for name in self.points.attributes.names:
                if name in ['position']: #, 'w', 'radius']:
                    continue

                vals = func.sample_value(getattr(self.splines[i_spline].get_points(), name))(np.linspace(0, 1, len(bbone)))

                meshed.points.attributes.copy_attribute(self.points.attributes, name)
                setattr(meshed.points, name, vals)

            # Join the new meshed spline
            mesh.join(meshed)

        return mesh

    
