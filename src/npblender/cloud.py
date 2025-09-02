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
Module Name: cloud
Author: Alain Bernard
Version: 0.1.0
Created: 2023-11-10
Last updated: 2025-08-29

Summary:
    Cloud of points geometry.

Usage example:
    >>> from cloud import Cloud

"""

__all__ = ["Cloud"]

from contextlib import contextmanager
import numpy as np

from . constants import PI, TAU
from . import blender
from . maths import distribs

from . geometry import Geometry
from . domain import Point

DATA_TEMP_NAME = "NPBL_TEMP"


# ====================================================================================================
# Cloud of points
# ====================================================================================================

class Cloud(Geometry):
    """
    Point-cloud geometry container.

    `Cloud` stores a set of points and their attributes, with helpers to
    import/export from Blender data (Mesh or PointCloud), join other clouds,
    apply basic transforms, and generate common point distributions.

    Attributes
    ----------
    points : [Point][npblender.domain.Point]
        Point domain storing per-point attributes (e.g., `position`, `normal`, ...).

    Notes
    -----
    - This class focuses on **point-only** data. For topological data
      (faces/edges), use [`Mesh`](npblender.mesh.Mesh).
    - Blender interoperability accepts both `bpy.types.Mesh` and
      `bpy.types.PointCloud` when reading, but writing currently targets a
      Mesh data block (see [`to_data`](npblender.cloud.Cloud.to_data)).
    """

    def __init__(self, points=None, attr_from=None, **attrs):
        """
        Initialize an empty cloud, optionally with points and attributes.

        Parameters
        ----------
        points : array-like of shape (N, 3) or (N, D), optional
            Coordinates to append as `points.position`. If extra keys are present,
            pass them via `**attrs`.
        attr_from : object, optional
            Source whose attribute schemas should be merged into this geometry,
            see [`join_attributes`](npblender.geometry.Geometry.join_attributes).
        **attrs
            Additional per-point attributes to append alongside `points`.
        """

        self.points = Point()
        self.join_attributes(attr_from)

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
        """
        Serialize the cloud to a plain Python dictionary.

        Returns
        -------
        dict
            A dictionary with keys: ``"geometry"`` (``"Cloud"``) and ``"points"``.
        """
        return {
            'geometry': 'Cloud',
            'points':    self.points.to_dict(),
            }

    @classmethod
    def from_dict(cls, d):
        """
        Deserialize a cloud from a dictionary produced by `to_dict`.

        Parameters
        ----------
        d : dict
            Serialized payload with at least the ``"points"`` key.

        Returns
        -------
        Cloud
            New instance with points loaded from `d`.
        """
        cloud = cls()
        cloud.points = Point.from_dict(d['points'])
        return cloud
    
    # ====================================================================================================
    # Clear the geometry
    # ====================================================================================================

    def clear_geometry(self):
        """
        Clear all point data (schemas kept, values cleared).

        Returns
        -------
        None
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
        """
        Build a cloud from another geometry that has a point domain.

        Parameters
        ----------
        other : [Geometry][npblender.geometry.Geometry]
            Source geometry (must have a `points` domain).
        selection : selection or None, optional
            Selection on points **to keep**; if provided, the complement is deleted
            after copying.

        Returns
        -------
        Cloud

        Raises
        ------
        ValueError
            If `other` has no `points` domain.
        """

        points = getattr(other, 'points')
        if points is None:
            raise ValueError(f"from_geometry> {type(other)} has no points.")

        cloud = cls()
        cloud.points = Point(points,  mode='COPY')

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
        """
        Synonym of [`from_geometry`](npblender.cloud.Cloud.from_geometry).

        Parameters
        ----------
        other : Cloud
        selection : selection or None, optional

        Returns
        -------
        Cloud
        """
        return cls.from_geometry(other, selection)
 

    # ----------------------------------------------------------------------------------------------------
    # Capture another Mesh
    # ----------------------------------------------------------------------------------------------------

    def capture(self, other):
        """
        Capture another cloud's buffers (no copy).

        Parameters
        ----------
        other : Cloud
            Source cloud whose `points` buffer is adopted.

        Returns
        -------
        Cloud
            `self`, for chaining.
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
        """
        Initialize the cloud from Blender data (Mesh or PointCloud).

        Parameters
        ----------
        data : bpy.types.Mesh or bpy.types.PointCloud or str
            A Blender data-block or a resolvable identifier accepted by
            [`blender.get_data`](npblender.blender.get_data).

        Returns
        -------
        Cloud

        Raises
        ------
        ValueError
            If the data-block type is not supported.
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
        """
        Write this cloud into a Blender **Mesh** data-block.

        Parameters
        ----------
        data : bpy.types.Mesh or str
            Target mesh data (or identifier resolvable by
            [`blender.get_data`](npblender.blender.get_data)). The geometry is
            cleared and repopulated.

        Returns
        -------
        None

        Notes
        -----
        - Vertices are created to match the point count; per-point attributes are
        written to `data.attributes`.

        > ***Caution:*** This writes to a **Mesh** data-block (not PointCloud)
        > because Blender’s Python API does not allow changing the point count of
        > a `PointCloud` at runtime.
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
        """
        Create a cloud from an existing Blender object.

        Parameters
        ----------
        obj : str or bpy.types.Object
            Object or name resolvable by [`blender.get_object`](npblender.blender.get_object).
        evaluated : bool, default=False
            If True, read from the evaluated object (modifiers applied).

        Returns
        -------
        Cloud
        """

        import bpy
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
        """
        Create or update a Blender object from this cloud.

        Parameters
        ----------
        obj : str or bpy.types.Object
            Object or name. If it does not exist, it is created.
        point_cloud : bool, default=False
            If True, convert the created mesh object to a `PointCloud` object.
        collection : str or bpy.types.Collection or None, optional
            Collection to link a newly created object into.

        Returns
        -------
        bpy.types.Object
            The created/updated Blender object.
        """

        from . import blender

        obj = blender.create_mesh_object(obj, collection=collection, halt=False)
        self.to_data(obj.data)

        if point_cloud:
            blender.mesh_to_point_cloud(obj)

        return obj

    # ====================================================================================================
    # Combining
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Join other clouds
    # ----------------------------------------------------------------------------------------------------

    def join(self, *others):
        """
        Append other clouds' points to this cloud.

        Parameters
        ----------
        *others : Geometry
            One or more Geometries to concatenate.

        Returns
        -------
        Cloud
            `self`, for chaining.
        """
        for other in others:
            self.points.extend(other.points)

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
            

