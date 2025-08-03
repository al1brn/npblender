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

from . constants import SPLINE_TYPES, BEZIER, POLY, NURBS
from . maths.splinesmaths import BSplines, Bezier, Poly, Nurbs



# =============================================================================================================================
# Root class for geometries

class Geometry:

    # ====================================================================================================
    # Check geometry consistency
    # ====================================================================================================

    def check(self, halt=True):
        return True

    @staticmethod
    def LoadObject(name):
        """ Load a Blender object and returns either a Mesh or a Curve.

        Arguments
        ---------
            - name (str or bpy.types.Object) : the object to load

        Returns
        -------
            - Mesh or Curve
        """

        from . import Mesh
        from . import Curve

        obj = blender.getobject(name)
        if isinstance(obj.data, bpy.types.Mesh):
            return Mesh.FromObject(oj)

        elif isinstance(obj.data, bpy.types.Curve):
            return Curve.FromObject(obj)

        else:
            raise Exception(f"Geometry.LoadObject error: impossible to load the objet '{obj.name}' of type '{type(obj.data).__name__}'")


    @staticmethod
    def LoadModel(model):
        """ Load a geometry or geometries from specification.

        The model can be:
            - a string : the name of a Blender object
            - a Blender object : the object to load
            - a Blender collection : the object to load
            - a list of these items : the list of objects to load

        Note that if model is a list or a collection, the method return Instances with models
        initialized with this list.

        Arguments
        ---------
            - model (any)

        Returns
        -------
            - Mesh or Curve or arrays of Meshes and Curves
        """

        from . import Mesh
        from . import Curve
        from . import Cloud
        from . import Instances

        if isinstance(model, (Mesh, Curve, Cloud, Instances)):
            return model

        elif isinstance(model, bpy.types.Collection):
            subs = [Geometry.FromModel(obj) for obj in model.objects]
            insts = Instances(np.zeros((len(subs), 3), float), subs)
            return insts

        elif isinstance(model, list):
            subs = [Geometry.FromModel(obj) for obj in model]
            insts = Instances(np.zeros((len(subs), 3), float), subs)
            return insts

        obj = blender.get_object(model)
        if isinstance(obj.data, bpy.types.Mesh):
            return Mesh.FromObject(obj)

        elif isinstance(obj.data, bpy.types.Curve):
            return Curve.FromObject(obj)

        else:
            raise Exception(f"Geometry.FromObject error: impossile to load the objet '{obj.name}' of type '{type(obj.data).__name__}'")

    # =============================================================================================================================
    # Save / restore configuration

    @property
    def domains(self):
        return {}

    def save(self):
        return {'domains': {domain_name: None if domain is None else domain.as_dict() for domain_name, domain in self.domains.items()}}

    def restore(self, data):
        domains = self.domains
        for domain_name, data in data['domains'].items():
            if data is not None:
                domains[domain_name].from_dict(data)

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

    # =============================================================================================================================
    # Transformation
    # =============================================================================================================================

    def transform(self, transfo):
        self.points.position = transfo @ self.points.position
        return self
    
    def translate(self, translation):
        self.points.position += translation
        return self

    def scale(self, scale, pivot = None):
        if pivot is not None:
            self.points.position -= pivot
        self.points.position *= scale
        if pivot is not None:
            self.points.position += pivot

        return self




