#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module

Created on Fri Nov 10 11:13:13 2023

@author: alain.bernard
@email: alain@ligloo.net

-----

Root class for geometries
"""

from contextlib import contextmanager
import numpy as np

import bpy

from . constants import SPLINE_TYPES, BEZIER, POLY, NURBS

from . import blender

# ----------------------------------------------------------------------------------------------------
# Root class for geometries
# ----------------------------------------------------------------------------------------------------

class Geometry:

    # ====================================================================================================
    # Copy attributes definition
    # ====================================================================================================

    def join_attributes(self, other, **kwargs):
        """ Capture the attributes from another geometry.

        Other can be a different geometry, in that case, only domains with the same name are captured.
        kwargs allows to filter the domains to capture:

        ```python
        mesh.join_attributes(other_mesh, faces=False)
        mesh.join_attributes(curve)
        ```

        Returns:
            - self
        """
        if other is None:
            return self

        for name in self.domain_names:
            if name in other.domain_names and kwargs.get(name, True):
                getattr(self, name).join_fields(getattr(other, name))

        return self


    # ====================================================================================================
    # Check geometry consistency
    # ====================================================================================================

    def check(self, title="Geometry Check", halt=True):
        return True

    # ====================================================================================================
    # Load a Mesh or a Curve
    # ====================================================================================================

    @staticmethod
    def load_object(name):
        """ Load a Blender object and returns either a Mesh or a Curve.

        Arguments
        ---------
            - name (str or bpy.types.Object) : the object to load

        Returns
        -------
            - Mesh or Curve
        """

        from .mesh import Mesh
        from .curve import Curve

        obj = blender.getobject(name)
        if obj is None:
            return None

        if isinstance(obj.data, bpy.types.Mesh):
            return Mesh.from_object(obj)

        elif isinstance(obj.data, bpy.types.Curve):
            return Curve.from_object(obj)

        else:
            raise Exception(f"Geometry.load_object error: impossible to load the objet '{obj.name}' of type '{type(obj.data).__name__}'")

    # ====================================================================================================
    # Load a Mesh or a Curve
    # ====================================================================================================

    @staticmethod
    def load_models(*specs):
        """ Load a geometry or geometries from specifications.

        The specs can be:
            - a Blender collection
            - a Blender object
            - a Geometry

        Arguments
        ---------
            - specs (list of objects / collections) : the models to load

        Returns
        -------
            - list of geometries
        """

        from .mesh import Mesh
        from .curve import Curve

        models = []

        for spec in specs:
            # A list
            if isinstance(spec, (list, tuple)):
                models.extend(Geometry.load_models(*spec))
                continue

            # A collection
            coll = blender.get_collection(spec, halt=False)
            if coll is not None:
                for obj in coll.objects:
                    geo = Geometry.load_object(obj)
                    if isinstance(geo, (Mesh, Curve)):
                        models.append(geo)
                continue

            # An object
            obj = blender.get_object(spec, halt=False)
            if obj is not None:
                geo = Geometry.load_object(obj)
                if isinstance(geo, (Mesh, Curve)):
                    models.append(geo)
                continue

            # A valide geometry
            if isinstance(spec, (Mesh, Curve)):
                models.append(spec)
                continue

            raise ValueError(f"Unknown model (type '{type(spec).__name__}'): {spec}")
        
        return models

    # ====================================================================================================
    # Object edition
    # ====================================================================================================

    @contextmanager
    def object(self, index=0, readonly=True):

        import bpy
        from . import blender

        temp_name = index if isinstance(index, str) else f"BPBL Temp {index}"

        ctx = bpy.context

        old_sel = [obj.name for obj in bpy.data.objects if obj.select_get()]
        old_active = ctx.view_layer.objects.active
        if old_active is None:
            old_active_name = None
        else:
            old_active_name = old_active.name

        bpy.ops.object.select_all(action='DESELECT')        

        obj = self.to_object(temp_name)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj        

        yield obj

        # ===== Returns from context

        if not readonly:
            self.capture(type(self).from_object(obj))

        blender.delete_object(obj)

        bpy.ops.object.select_all(action='DESELECT')        
        for name in old_sel:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.select_set(True)

        if old_active_name is not None:
            bpy.context.view_layer.objects.active = bpy.data.objects.get(old_active_name)

    # ====================================================================================================
    # Material
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Index from material name
    # ----------------------------------------------------------------------------------------------------

    def get_material_index(self, mat_name):
        """ Return the index of a material name.

        If the material doesn't exist, it is created

        Arguments
        ---------
            - mat_name (str) : material name

        Returns
        -------
            - int : index of the material name in the materials list
        """

        if mat_name in self.materials:
            return self.materials.index(mat_name)
        else:
            self.materials.append(mat_name)
            return len(self.materials)-1
        
    # ----------------------------------------------------------------------------------------------------
    # Add a material or a list of materials
    # ----------------------------------------------------------------------------------------------------

    def add_materials(self, materials):
        """ Add a materials list to the existing one.

        If a material already exist, it is not added another time.

        Arguments
        ---------
            - materials (list of strs) : the list of materials to append.
        """
        if isinstance(materials, str):
            self.materials.append(materials)
        else:
            self.materials.extend(materials)

    # ====================================================================================================
    # Transformation
    # ====================================================================================================

    def get_points_selection(self):
        return slice(None)
    
    def _check_transformation_shape(self, t_shape, npoints, label="Transformation"):
        if t_shape == ():
            return (npoints, 3)

        elif len(t_shape) == 1:
            if t_shape[0] not in [1, npoints]:
                return (npoints, 3)

        else:
            n = int(np.prod(t_shape))
            if npoints % n == 0:
                return t_shape[:-1] + (-1, 3)

        raise AttributeError(f"{label} shape {t_shape} is not valid to transform {npoints} points.")
    
    # ----------------------------------------------------------------------------------------------------
    # Apply one or more basic transformation
    # ----------------------------------------------------------------------------------------------------
    
    def transformation(self, rotation=None, scale=None, translation=None, pivot=None):

        # Curve splines can be a subset of the points 
        pts_sel = self.get_points_selection()
        pos = self.points.position[pts_sel]
        npoints = len(pos)
        all_vecs = [pos]

        has_handles = "handle_left" in self.points.actual_names
        if has_handles:
            left = self.points.handle_left[pts_sel]
            right = self.points.handle_right[pts_sel]
            all_vecs.extend([left, right])

        if False:
            # First pivot
            if pivot is not None:
                pivot = np.asarray(pivot)
                for v in all_vecs:
                    v -= pivot

            # Scale
            if scale is not None:
                scale = np.asarray(scale)
                for v in all_vecs:
                    v *= scale
                    
            # Rotation
            if rotation is not None:
                for v in all_vecs:
                    v = rotation @ v

            # Pivot back
            if pivot is not None:
                for v in all_vecs:
                    v += pivot

            # Translation
            if translation is not None:
                translation = np.asarray(translation)
                for v in all_vecs:
                    v += translation

            # Back
            self.points[pts_sel].position = pos
            if has_handles:
                self.points[pts_sel].handle_left = all_vecs[1]
                self.points[pts_sel].handle_right = all_vecs[2]

        else:
            # First pivot
            if pivot is not None:
                pivot = np.asarray(pivot)
                pivot_shape = self._check_transformation_shape(pivot.shape[:-1], npoints, label="Pivot")
                for v in all_vecs:
                    v.reshape(pivot_shape)[:] -= pivot

            # Scale
            if scale is not None:
                scale = np.asarray(scale)
                scale_shape = self._check_transformation_shape(scale.shape[:-1], npoints, label="Scale")
                for v in all_vecs:
                    v.reshape(scale_shape)[:] *= scale
                    
            # Rotation
            if rotation is not None:
                rot_shape = self._check_transformation_shape(rotation.shape, npoints, label="Rotation")
                for v in all_vecs:
                    v.reshape(rot_shape)[:] = rotation @ v.reshape(rot_shape)

            # Pivot back
            if pivot is not None:
                for v in all_vecs:
                    v.reshape(pivot_shape)[:] += pivot

            # Translation
            if translation is not None:
                translation = np.asarray(translation)
                tr_shape = self._check_transformation_shape(translation.shape[:-1], npoints, label="Pivot")
                for v in all_vecs:
                    v.reshape(tr_shape)[:] += translation

            # Back
            self.points[pts_sel].position = pos
            if has_handles:
                self.points[pts_sel].handle_left = all_vecs[1]
                self.points[pts_sel].handle_right = all_vecs[2]

        return self

    def translate(self, translation):
        return self.transformation(translation=translation)

    def apply_scale(self, scale, pivot=None):
        return self.transformation(scale=scale, pivot=pivot)
    
    def rotate(self, rotation, pivot=None):
        return self.transformation(rotation=rotation, pivot=pivot)
    
    def transform(self, transformation):
        return self.transformation(rotation=transformation)
    
    # ====================================================================================================
    # Envelop
    # ====================================================================================================

    @property
    def bounding_box(self):
        pos = self.points.position
        if len(pos):
            return np.min(pos, axis=0), np.max(pos, axis=0)
        else:
            return np.zeros(3, float), np.zeros(3, float)
        
    @property
    def bounding_box_dims(self):
        v0, v1 = self.bounding_box
        return v1 - v0
        
    @property
    def max_size(self):
        return max(self.bounding_box_dims)
    
    def get_cubic_envelop(self):
        from . mesh import Mesh

        size = self.bounding_box_dims
        return Mesh.cube(size=size, materials=getattr(self, "materials", None))
        







