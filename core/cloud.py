#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module

Created on Fri Nov 10 11:50:13 2023

@author: alain.bernard
@email: alain@ligloo.net

-----

Mesh geometry.

"""

from contextlib import contextmanager
import numpy as np

from . constants import SPLINE_TYPES, BEZIER, POLY, NURBS
from . constants import bfloat, bint, bbool
from . constants import PI, TAU
from . import blender
#from . maths import BSplines, Bezier, Poly, Nurbs
from . maths import Transformation, Quaternion, Rotation
from . maths.topology import grid_corners, grid_uv_map, fans_corners, disk_uv_map
from . maths.topology import border_edges, edges_between, row_edges, col_edges
from . maths import distribs

from . geometry import Geometry
from . domain import CloudPointDomain

DATA_TEMP_NAME = "NPBL_TEMP"


# ====================================================================================================
# Cloud of points
# ====================================================================================================

class Cloud(Geometry):

    def __init__(self, points=None, attr_from=None, **attrs):
        """ Clod Geometry.

        Arguments
        ---------
            - points (array of vectors = None) : the vertices
            - attrs (dict) : other geometry attributes
        """
        # ----- Initialize an empty geometry

        self.domain_names = ['points']
        self.points = CloudPointDomain()

        self.join_attributes(attr_from)

        # ----- Add geometry
        if points is not None:
            self.points.append(position=points, **attrs)

    # ====================================================================================================
    # Dump
    # ====================================================================================================

    def __str__(self):
        return f"<Mesh: points {len(self.points)}, corners {len(self.corners)}, faces {len(self.faces)}, edges {len(self.edges)}>"

    def __repr__(self):
        s = "Mesh:\n   " + "\n   ".join([str(self.points), str(self.corners), str(self.faces)])
        return s

    # ====================================================================================================
    # Serialization
    # ====================================================================================================

    def to_dict(self):
        return {
            'geometry': 'Cloud',
            'points':    self.points.to_dict(),
            }

    @classmethod
    def from_dict(cls, d):
        cloud = cls()
        cloud.points = CloudPointDomain.from_dict(d['points'])
        return cloud
    
    # ====================================================================================================
    # Clear the geometry
    # ====================================================================================================

    def clear_geometry(self):
        """ Clear the geometry.

        Delete all the content.
        """
        self.points.clear()

    # ====================================================================================================
    # From another geometry with points
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Copy
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_geometry(cls, other, selection=None):
        """ Create a Cloud from another gemetry with points domain.

        Arguments
        ---------
            - other (Geometry) : the geometry to copy
            - selection (selection) : a valid selection on points

        Returns
        -------
            - Cloud
        """

        points = getattr(other, 'points')
        if points is None:
            raise ValueError(f"from_geometry> {type(other)} has no points.")

        cloud = cls()
        cloud.points = CloudPointDomain(points,  mode='COPY')

        if selection is not None:
            points_mask = np.ones(len(cloud.points), dtype=bool)
            points_mask[selection] = False

            cloud.points.delete(points_mask)

        return cloud
    
    # ----------------------------------------------------------------------------------------------------
    # Synonym
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_cloud(cls, other, selection=None):

        return cls.from_geometry(other, selection)
 

    # ----------------------------------------------------------------------------------------------------
    # Capture another Mesh
    # ----------------------------------------------------------------------------------------------------

    def capture(self, other):
        """ Capture the data of another Mesh.

        Arguments
        ---------
            - other (Cloud) : the mesh to capture

        Returns
        -------
            - self
        """
        self.points  = other.points

    # ====================================================================================================
    # Blender Interface
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # From Mesh data
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_data(cls, data):
        """ Initialize the geometry from a Blender data (Mesh or PointCloud)

        Arguments
        ---------
            - data (Blender Mesh or PointCloud) : the data to load
        """

        import bpy
        from . import blender

        data = blender.get_data(data)

        cloud = cls()

        if isinstance(data, bpy.types.Mesh):

            n = len(data.vertices)
            if n != 0:
                cloud.points.resize(n)

            cloud.points.from_bl_attributes(data.attributes)

        elif isinstance(data, bpy.types.PointCloud):

            n = len(data.points)
            if n != 0:
                cloud.points.add(n)

            cloud.points.from_bl_attributes(data.attributes)

        else:
            raise ValueError(f"Cloud.from_data> data type {type(date).__name__} not supported yet")

        return cloud
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # To data
    # -----------------------------------------------------------------------------------------------------------------------------

    def to_data(self, data):
        """ Write the geometry into a Blender Mesh

        > [!CAUTION:
        > to_data creates a blender Mesh, not PointCloud since the pyton API doesn't allow to dynamically
        > change the number of points

        Arguments
        ---------
            - mesh (Blender Mesh instance) : the mesh to write
        """

        from . import blender

        bl_mesh = blender.get_data(data)
        bl_mesh.clear_geometry()

        bl_mesh.materials.clear()

        # ----------------------------------------------------------------------------------------------------
        # Vertices

        if len(self.points):
            bl_mesh.vertices.add(len(self.points))

        # ----------------------------------------------------------------------------------------------------
        # Attributes

        attributes = data.attributes

        self.points.to_bl_attributes(attributes, update=False)

        # ----------------------------------------------------------------------------------------------------
        # Update

        bl_mesh.update()

    # ----------------------------------------------------------------------------------------------------
    # Initialize from an object
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def from_object(cls, obj, evaluated=False):
        """ Create a Mesh from an existing object.

        Arguments
        ---------
            - obj (str or Blender object) : the object to initialize from
            - evaluated (bool = False) : object modified by the modifiers if True, raw vertices otherwise

        Returns
        -------
            - Mesh
        """

        from . import blender

        if evaluated:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            object_eval = blender.get_object(obj).evaluated_get(depsgraph)
            return cls.from_data(object_eval.data)

        else:
            return cls.from_data(blender.get_object(obj).data)
        
    # ----------------------------------------------------------------------------------------------------
    # To blender object
    # ----------------------------------------------------------------------------------------------------

    def to_object(self, obj, point_cloud=False, collection=None):
        """ Create or update a blender object.

        By default, a mesh object is created. If as_point_cloud is True, the object is the converted
        to a PointCloud object.

        Arguments
        ---------
            - obj (str or Blender object) : the object the create
            - point_cloud (bool = False) : the object is a PointCloud object if True
            - collection (str or Blender collection) : the collection to add the object to

        Returns
        -------
            - Blender Mesh or PointCloud object
        """

        from . import blender

        obj = blender.create_mesh_object(obj, collection=collection, halt=False)
        self.to_data(obj.data)

        if point_cloud:
            blender.mesh_to_point_cloud(obj)

        return obj

    # ====================================================================================================
    # Object edition
    # ====================================================================================================

    @contextmanager
    def object(self, index=0, as_point_cloud=True, readonly=True):

        temp_name = index if isinstance(index, str) else f"BPBL Temp Mesh {index}"

        ctx = bpy.context

        old_sel = [obj.name for obj in bpy.data.objects if obj.select_get()]
        old_active = ctx.view_layer.objects.active
        if old_active is None:
            old_active_name = None
        else:
            old_active_name = old_active.name

        bpy.ops.object.select_all(action='DESELECT')        

        obj = self.to_object(temp_name, as_point_cloud=as_point_cloud)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj        

        yield obj

        if not readonly:
            self.capture(Mesh.from_object(obj))

        blender.delete_object(obj)

        bpy.ops.object.select_all(action='DESELECT')        
        for name in old_sel:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.select_set(True)

        if old_active_name is not None:
            bpy.context.view_layer.objects.active = bpy.data.objects.get(old_active_name)

    # ====================================================================================================
    # Combining
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Join other clouds
    # ----------------------------------------------------------------------------------------------------

    def join(self, *others):
        """ Join other clouds.

        Arguments
        ---------
            - other (Mesh) : the Meshes to append
        """
        for other in others:
            self.points.extend(other.points)

        return self

    # =============================================================================================================================
    # Transformation
    # =============================================================================================================================

    def transform(self, transfo):
        self.points.position = transfo @ self.points.position
        return self
    
    def translate(self, translation):
        self.points.position += translation
        return self

    def scale(self, scale, pivot=None):
        if pivot is not None:
            self.points.position -= pivot
        self.points.position *= scale
        if pivot is not None:
            self.points.position += pivot

        return self

    # ====================================================================================================
    # Distributions
    # ====================================================================================================

    @classmethod
    def line_dist(cls, point0=(-1, -1, -1), point1=(1, 1, 1), count=10, density=None, seed=None):
        cloud = cls()
        cloud.points.new_vector('tangent')
        d = distribs.line_dist(point0, point1, count, density, seed)
        cloud.points.append(position=d['points'], tangent=d['tangents'])
        return cloud

    @classmethod
    def arc_dist(
        cls,
        radius=1.0,
        scale=None,
        center=(0, 0, 0),
        arc_center=0.0,
        arc_angle=PI/2,
        use_vonmises=False,
        count=10,
        density=None,
        seed=None,
    ):
        cloud = cls()
        cloud.points.new_vector('tangent')
        cloud.points.new_float('angle')
        d = distribs.arc_dist(
            radius, scale, center, arc_center, arc_angle, use_vonmises, count, density, seed
        )
        cloud.points.append(position=d['points'], tangent=d['tangents'], angle=d['angles'])
        return cloud

    @classmethod
    def circle_dist(
        cls,
        radius=1.0,
        scale=None,
        center=(0, 0, 0),
        count=10,
        density=None,
        seed=None,
    ):
        cloud = cls()
        cloud.points.new_vector('tangent')
        cloud.points.new_float('angle')
        d = distribs.circle_dist(radius, scale, center, count, density, seed)
        cloud.points.append(position=d['points'], tangent=d['tangents'], angle=d['angles'])
        return cloud

    @classmethod
    def rect_dist(cls, a=1, b=1, center=(0, 0, 0), count=10, density=None, seed=None):
        cloud = cls()
        d = distribs.rect_dist(a, b, center, count, density, seed)
        cloud.points.append(position=d['points'])
        return cloud


    @classmethod
    def pie_dist(
        cls,
        radius=1,
        outer_radius=None,
        center=(0, 0, 0),
        pie_center=0.,
        pie_angle=PI/2,
        use_vonmises=False,
        count=10,
        density=None,
        seed=None
    ):
        cloud = cls()
        cloud.points.new_vector('tangent')
        cloud.points.new_float('angle')
        d = distribs.pie_dist(
            radius, outer_radius, center, pie_center, pie_angle, use_vonmises, count, density, seed
        )
        cloud.points.append(position=d['points'], tangent=d['tangents'], angle=d['angles'])
        return cloud

    @classmethod
    def disk_dist(cls, radius=1, outer_radius=None, center=(0, 0, 0), count=10, density=None, seed=None):
        cloud = cls()
        cloud.points.new_vector('tangent')
        cloud.points.new_float('angle')
        d = distribs.disk_dist(
            radius, outer_radius, center, count, density, seed
        )
        cloud.points.append(position=d['points'], tangent=d['tangents'], angle=d['angles'])
        return cloud

    @classmethod
    def cylinder_dist(
        cls,
        radius=1.0,
        scale=None,
        height=1.0,
        center=(0, 0, 0),
        arc_center=0.0,
        arc_angle=TAU,
        use_vonmises=False,
        count=10,
        density=None,
        seed=None,
    ):
        cloud = cls()
        cloud.points.new_vector('normal')
        cloud.points.new_vector('tangent')
        cloud.points.new_float('angle')
        d = distribs.cylinder_dist(
            radius, scale, height, center, arc_center, arc_angle, use_vonmises, count, density, seed
        )
        cloud.points.append(position=d['points'], normal=d['normals'], tangent=d['tangents'], angle=d['angles'])
        return cloud


    @classmethod
    def sphere_dist(
        cls,
        radius=1.0,
        scale=None,
        center=(0, 0, 0),
        count=10,
        density=None,
        seed=None
    ):
        cloud = cls()
        cloud.points.new_vector('normal')
        cloud.points.new_float('theta')
        cloud.points.new_float('phi')
        d = distribs.sphere_dist(radius, scale, center, count, density, seed)
        cloud.points.append(position=d['points'], normal=d['normals'], theta=d['thetas'], phi=d['phis'])
        return cloud

    @classmethod
    def dome_dist(
        cls,
        radius=1.0,
        scale=None,
        axis=(0, 0, 1),
        angle=np.pi / 2,
        use_vonmises=False,
        center=(0, 0, 0),
        count=10,
        density=None,
        seed=None
    ):
        cloud = cls()
        cloud.points.new_vector('normal')
        d = distribs.dome_dist(
            radius, scale, axis, angle, use_vonmises, center, count, density, seed
        )
        cloud.points.append(position=d['points'], normal=d['normals'])
        return cloud


    @classmethod
    def cube_dist(cls, size=1, center=(0, 0, 0), count=10, density=None, seed=None):
        return cls(distribs.cube_dist(size, center, count, density, seed)['points'])

    @classmethod
    def ball_dist(
        cls,
        radius=1.0,
        axis=(0, 0, 1),
        angle=np.pi,
        use_vonmises=False,
        center=(0, 0, 0),
        count=10,
        density=None,
        scale=None,
        seed=None,
        **kwargs
    ):
        cloud = cls()
        cloud.points.new_vector('normal')        
        d = distribs.ball_dist(
            radius, axis, angle, use_vonmises, center, count, density, scale, seed, **kwargs)
        cloud.points.append(position=d['points'], normal=d['normals'])
        return cloud
    
    # ====================================================================================================
    # Tests
    # ====================================================================================================

    def _test_distribs():

        rng = np.random.default_rng(0)

        cloud = Cloud()

        for i, dist_name in enumerate([
            "line_dist", "arc_dist", "circle_dist", "rect_dist", "pie_dist", "disk_dist", 
            "cylinder_dist", "sphere_dist", "dome_dist", "cube_dist", "ball_dist"]):
            
            f = getattr(Cloud, dist_name)
            cl = f(density=100, seed=rng)
            cl.points.x += i*3
            cloud.join(cl)

        cloud.to_object("Distributions")
            

