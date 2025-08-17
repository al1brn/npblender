#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module

Created on Fri Nov 10 11:50:13 2023

@author: alain

Mesh geometry
"""

import numpy as np

from . constants import bfloat, bint, bbool
from . import blender
from . maths import Transformation, Quaternion, Rotation
from . maths import splinemaths

from . geometry import Geometry
from . domain import InstanceDomain

DATA_TEMP_NAME = "npblender_TEMP"

class Instances(Geometry):

    def __init__(self, points=None, models=None, model_index=0, attr_from=None, **attributes):
        """ Create new instances.

        Arguments
        ---------
            - points (array of vectors = None) : instances locations
            - models (geometry or list of geometries = None) : the geometries to instantiate
            - model_index (int = 0) : model index of instances
            - **attributes (dict) : other geometry attributes
        """
        self.domain_names = ['points']
        self.points  = InstanceDomain()

        self.join_attributes(attr_from)

        if models is None:
            self.models = []
        else:
            self.models = self.load_models(models)
        self.low_resols = []
        
        self.points.append(position=points, model_index=model_index, **attributes)

    def check(self, title="Instances Check", halt=True):
        n = np.max(self.points.model_index)
        if n >= len(self.models):
            print(title)
            print(f"Model index {n} greater than the number {len(self.models)} of models.")
            if halt:
                assert(False)
            return False
        return True

    def __str__(self):
        return f"<Instances: {len(self)}, models: {len(self.models)}>"
    
    def __len__(self):
        return len(self.points)
    
    # =============================================================================================================================
    # Serialization
    # =============================================================================================================================

    def to_dict(self):
        return {
            'geometry':   'Instances',
            'points':     self.points.to_dict(),
            'models':     [model.to_dict() for model in self.models],
            'low_resols': [[lr.to_dict() for lr in low_resol] for low_resol in self.low_resols]
            }

    @classmethod
    def from_dict(cls, d):
        insts = cls()
        insts.points     = InstanceDomain.from_dict(d['points'])
        insts.models     = [Geometry.from_dict(model) for model in d['models']]
        insts.low_resols = [[Geometry.from_dict(lrd) for lrd in low_resol_dict] for low_resol_dict in d['low_resols']]

        return insts    
    
    # ====================================================================================================
    # Low scale models
    # ====================================================================================================

    def add_low_resol(self, dist, models):

        ls_models = self.load_models(models)
        if len(ls_models) != len(self.models):
            raise ValueError(
                f"Instances.add_low_resol: the number of low scale models ({len(ls_models)}) "
                f"is not equal to the number of models ({len(self.models)})")
        
        self.low_resols.append({"dist": dist, "models": ls_models})

    def compute_low_resols(self, start_scale=.1, scale=.8, detail=1.):

        from .camera import Camera
        from .mesh import Mesh
        from .curve import Curve

        cam = Camera()

        scale = np.clip(scale, .01, .99)

        # Start scale
        cur_scale = min(start_scale, 1.)

        for _ in range(10):
            max_count = None
            min_dist = None
            ls_models = []
            for model in self.models:
                d, ps = cam.distance_for_scale(model.max_size, scale=cur_scale)
                if min_dist is None:
                    min_dist = d
                else:
                    min_dist = min(d, min_dist)

                if isinstance(model, Mesh):
                    ls_models.append(model.simplified(cur_scale, ps/detail))
                    if max_count is None:
                        max_count = len(ls_models[-1].points)
                    else:
                        max_count = max(max_count, len(ls_models[-1].points))
                else:
                    ls_models.append(Curve.from_curve(model))

            self.add_low_resol(min_dist, ls_models)
            cur_scale *= scale

            if max_count is None or max_count <= 8:
                break


    def _sort_low_resols(self):

        dist = np.array([item['dist'] for item in self.low_resols])
        s = np.flip(np.argsort(dist))

        return dist[s], [self.low_resols[i]['models'] for i in s]
    
    # ====================================================================================================
    # To Blender objects
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Realize as Mesh and Curve
    # ----------------------------------------------------------------------------------------------------

    def realize(self, camera_culling=False):

        from . mesh import Mesh
        from . curve import Curve
        from . camera import Camera

        # ---------------------------------------------------------------------------
        # Camera
        # ---------------------------------------------------------------------------

        if camera_culling == False:
            camera = None
        else:
            camera = Camera(camera_culling)
            camera_culling = True

        if camera_culling:
            mdl_radius = np.zeros(len(self.models), bfloat)
            for i, model in enumerate(self.models):
                vmin = np.min(model.points.position, axis=0)
                vmax = np.max(model.points.position, axis=0)
                mdl_radius[i] = max(vmax[0] - vmin[0], vmax[1] - vmin[1], vmax[2] - vmin[2])

            vis, dist = camera.visible_points(self.points.position, radius=mdl_radius[self.points.model_index])
            
            vis = vis[:, camera.VISIBLE]
            dist = dist[:, camera.DISTANCE]

            if len(self.low_resols):
                ls_dist, ls_all_models = self._sort_low_resols()

        mesh  = Mesh()
        curve = Curve()
        insts = self.points

        # ---------------------------------------------------------------------------
        # Add a model on a selection
        # ---------------------------------------------------------------------------

        def _add_model(sel, model):

            n = np.sum(sel)
            if n == 0:
                return

            # --------------------------------------------------
            # Instantiate the model
            # --------------------------------------------------

            geo = model*n

            # --------------------------------------------------
            # Transformation
            # --------------------------------------------------

            pts = insts[sel]

            if "scale" in self.points.actual_names:
                scale = pts.scale
            else:
                scale = None

            if self.points.has_rotation:
                rot = pts.rotation
            else:
                rot = None

            geo.transformation(rotation=rot[:, None], scale=scale[:, None], translation=pts.position[:, None])

            # --------------------------------------------------
            # Add to realized geometry
            # --------------------------------------------------

            if isinstance(model, Mesh):
                mesh.join(geo)

            elif isinstance(model, Curve):
                curve.join(geo)

            else:
                raise TypeError(f"Instances.realize> Unsupported model type: '{type(model).__name__}'")

        # ---------------------------------------------------------------------------
        # Loop on models
        # ---------------------------------------------------------------------------

        for model_index, model in enumerate(self.models):

            # Selection on instances of the current model
            sel = insts.model_index == model_index

            # Visible instances
            if camera_culling:
                sel = np.logical_and(sel, vis)

            # Low scale
            if camera_culling and len(self.low_resols):
                for d, ls_models in zip(ls_dist, ls_all_models):
                    sel_dist = dist > d
                    ls_sel = np.logical_and(sel, sel_dist)
                    _add_model(ls_sel, ls_models[model_index])
                    sel = np.logical_and(sel, np.logical_not(sel_dist))

            # Normal scale
            _add_model(sel, model)
            
        return {
            'mesh': mesh if len(mesh.points) else None,
            'curve': curve if len(curve.points) else None,
            }
    
    # ----------------------------------------------------------------------------------------------------
    # To object
    # ----------------------------------------------------------------------------------------------------
    
    def to_object(self, name, profile=None, caps=True, use_radius=True, shade_smooth=True, camera_culling=False):

        # Realize to mesh and curve
        geos = self.realize(camera_culling=camera_culling)

        # Curve to mesh
        if geos['curve'] is not None and ((profile is not None) or camera_culling):
            meshed = geos['curve'].to_mesh(profile=profile, caps=caps, use_radius=use_radius, camera_culling=camera_culling)
            if geos['mesh'] is None:
                geos['mesh'] = meshed
            else:
                geos['mesh'].join(meshed)
            geos['curve'] = None

        # Geometries to objects
        both = (geos['mesh'] is not None) and (geos['curve'] is not None)

        objects = {}

        if geos['mesh'] is not None:
            suffix = " - (M)" if both else ""
            objects['mesh'] = geos['mesh'].to_object(f"{name}{suffix}", shade_smooth=shade_smooth)

        if geos['curve'] is not None:
            if both:
                suffix = " - (C)" if both else ""
            objects['curve'] = geos['curve'].to_object(f"{name}{suffix}")

        return objects
    
    # ====================================================================================================
    # Dump 
    # ====================================================================================================

    def models_to_object(self, name="Models"):
        from .mesh import Mesh
        from .curve import Curve

        mesh = Mesh()

        if len(self.low_resols):
            _, lr_all_models = self._sort_low_resols()

        x = 0.
        for i_model, model in enumerate(self.models):
            bbox = model.bounding_box_dims

            if isinstance(model, Mesh):
                m = Mesh.from_mesh(model)
            else:
                m = model.to_mesh()

            m.translate((x, 0, 0))
            mesh.join(m)

            # ----- Low resols

            if not len(self.low_resols):
                continue

            z = bbox[2] + 1
            for i in reversed(range(len(lr_all_models))):
                mdl = lr_all_models[i][i_model]

                if isinstance(mdl, Mesh):
                    m = Mesh.from_mesh(mdl)
                else:
                    m = mdl.to_mesh()

                m.translate((x, 0, z))
                mesh.join(m)
                z += 1 + bbox[2]

            x += 1 + bbox[0]


        return mesh.to_object(name, shade_smooth=False)

    # ====================================================================================================
    # Joining instances
    # ====================================================================================================

    def join(self, *others):

        for other in others:
            insts_count = len(self.points)
            models_count = len(self.models)

            self.models.extend(other.models)
            self.points.extend(other.points, join_fields=True)
            self.points[insts_count:] += models_count

        return self
    
    # ====================================================================================================
    # Multiply
    # ====================================================================================================

    def multiply(self, count, in_place=True):

        count = int(count)

        if count == 0:
            return None
        
        if count == 1:
            if in_place:
                return self
            else:
                return type(self)().join(self)
            
        if not in_place:
            return type(self)().join(self).multiply(count, in_place=True)
        
        self.points.mmultiply(count)

        return self

    def __mul__(self, count):
        return self.multiply(count, in_place=False)

    def __imul__(self, count):
        return self.multiply(count, in_place=True)
    
    # ====================================================================================================
    # Operations
    # ====================================================================================================

    def clear_geometry(self):
        self.points.clear()

