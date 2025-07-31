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


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

else:
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
