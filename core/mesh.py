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
from . constants import FillCap
from . import blender
#from . maths import BSplines, Bezier, Poly, Nurbs
from . maths import Transformation, Quaternion, Rotation
from . maths.topology import grid_corners, grid_uv_map, fans_corners, disk_uv_map
from . maths.topology import border_edges, edges_between, row_edges, col_edges

from . geometry import Geometry
from . domain import PointDomain, CornerDomain, FaceDomain, EdgeDomain


import bpy
import bmesh
from mathutils.bvhtree import BVHTree



DATA_TEMP_NAME = "npblender_TEMP"


# =============================================================================================================================
# Mesh Gemetry

class Mesh(Geometry):

    def __init__(self, points=None, corners=None, faces=None, edges=None, materials=None, attr_from=None, **attrs):
        """ Mesh Geometry.

        Arguments
        ---------
            - points (array of vectors = None) : the vertices
            - corners (array of ints = None) : corners, i.e. indices on the array of points
            - faces (array of ints = None) : size of the faces, the sum of this array must be equal to the length of the corners array
            - edges (array of couples of ints = None) : list of edges defined by two vertex indices
            - materials (str or list of strs = None) : list of materials used in the geometry
            - attr_from (Geometry) : domain attributes to copy from
            - **attrs (dict) : other geometry attributes
        """

        # ----- Initialize an empty geometry

        self.domain_names = ['points', 'corners', 'faces', 'edges']

        self.points  = PointDomain()
        self.corners = CornerDomain()
        self.faces   = FaceDomain()
        self.edges   = EdgeDomain()

        self.join_attributes(attr_from)

        # ----- The materials

        if materials is None:
            self.materials = []
        elif isinstance(materials, str):
            self.materials = [materials]
        else:
            self.materials = [mat for mat in materials]

        # ----- Add geometry

        self.add_geometry(points, corners, faces, edges, **attrs)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Utilities

    @classmethod
    def dummy(cls, points_count=0, corners_count=0, faces=None, edges=None, seed=0, **attrs):
        rng = np.random.default_rng(seed)
        verts = rng.uniform(-1, 1, (points_count, 3)) if points_count else None
        corners = rng.uniform(0, points_count, corners_count) if corners_count else None
        loops = None

        if faces is None:
            n = corners_count if corners_count else points_count
            if n > 0:
                loops = [4] * (n //4)
                loops[0] = n - np.sum(loops[1:])
            else:
                loops = None
        else:
            loops = faces

        return cls(points=verts, corners=corners, faces=loops, edges=edges,**attrs)

    def check(self, title="Mesh Check", halt=True):
        ok = self.corners.check(len(self.points), halt=False) and \
               self.faces.check(len(self.corners), halt=False) and \
               self.edges.check(len(self.points), halt=False)
        if ok:
            return True
        elif halt:
            raise Exception(f"{title} check failed")
        else:
            print(f"{title} check failed")
        

    def __str__(self):
        return f"<Mesh: points {len(self.points)}, corners {len(self.corners)}, faces {len(self.faces)}, edges {len(self.edges)}>"

    def __repr__(self):
        s = "Mesh:\n   " + "\n   ".join([str(self.points), str(self.corners), str(self.faces)])
        return s

    # =============================================================================================================================
    # Serialization
    # =============================================================================================================================

    def to_dict(self):
        return {
            'geometry':     'Mesh',
            'materials' :   self.materials,
            'points':       self.points.to_dict(),
            'corners':      self.corners.to_dict(),
            'faces':        self.faces.to_dict(),
            'edges':        self.edges.to_dict(),
            }

    @classmethod
    def from_dict(cls, d):
        mesh = cls()
        mesh.materials  = d['materials']
        mesh.points     = PointDomain.from_dict(d['points'])
        mesh.corners    = CornerDomain.from_dict(d['corners'])
        mesh.faces      = FaceDomain.from_dict(d['faces'])
        mesh.edges      = EdgeDomain.from_dict(d['edges'])
        return mesh
    
    # =============================================================================================================================
    # Clear the geometry
    # =============================================================================================================================

    def clear_geometry(self):
        """ Clear the geometry.

        Delete all the content.
        """

        self.points.clear()
        self.corners.clear()
        self.faces.clear()
        self.edges.clear()

    # =============================================================================================================================
    # From another Mesh
    # =============================================================================================================================

    # -----------------------------------------------------------------------------------------------------------------------------
    # Copy
    # -----------------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_mesh(cls, other, points=None, faces=None, edges=None):
        """ Create a Mesh from another mesh.

        Arguments
        ---------
            - other (Mesh) : the mesh to copy

        Returns
        -------
            - Mesh
        """
        mesh = cls(materials=other.materials)
        mesh.points  = PointDomain(other.points,  mode='COPY')
        mesh.corners = CornerDomain(other.corners, mode='COPY')
        mesh.faces   = FaceDomain(other.faces,   mode='COPY')
        mesh.edges   = EdgeDomain(other.edges,    mode='COPY')

        if points is None:
            points_mask = None
        else:
            points_mask = np.ones(len(mesh.points), dtype=bool)
            points_mask[points] = False

        if faces is None:
            faces_mask = None
        else:
            faces_mask = np.ones(len(mesh.faces), dtype=bool)
            faces_mask[faces] = False

        if edges is None:
            edges_mask = None
        else:
            edges_mask = np.ones(len(mesh.edges), dtype=bool)
            edges_mask[edges] = False

        mesh.delete_vertices(points=points_mask, faces=faces_mask, edges=edges_mask)

        return mesh

    # -----------------------------------------------------------------------------------------------------------------------------
    # Capture another Mesh
    # -----------------------------------------------------------------------------------------------------------------------------

    def capture(self, other):
        """ Capture the data of another Mesh.

        Arguments
        ---------
            - other (Mesh) : the mesh to capture

        Returns
        -------
            - self
        """

        self.materials = other.materials

        self.points  = other.points
        self.corners = other.corners
        self.faces   = other.faces
        self.edges   = other.edges

    # =============================================================================================================================
    # Blender Interface
    # =============================================================================================================================

    # -----------------------------------------------------------------------------------------------------------------------------
    # From Mesh data
    # -----------------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_mesh_data(cls, data):
        """ Initialize the geometry from a Blender Mesh

        Arguments
        ---------
            - mesh (Blender Mesh instance) : the mesh to load
        """

        from . import blender

        bl_mesh = blender.get_mesh(data)

        mesh = cls()

        # ----- Materials

        mesh.materials = [None if mat is None else mat.name for mat in bl_mesh.materials]

        # ----- Vertices

        n = len(bl_mesh.vertices)
        if n != 0:
            # Positions will be read as position attribute
            mesh.points.resize(n)

        # ----- Corners

        n = len(bl_mesh.polygons)
        if n != 0:
            a = np.empty(len(bl_mesh.loops), bint)
            bl_mesh.loops.foreach_get("vertex_index", a)
            mesh.corners.append(vertex_index=a)

            a = np.empty(len(bl_mesh.polygons), bint)
            bl_mesh.polygons.foreach_get("loop_total", a)
            mesh.faces.append_sizes(a)

            del a

        # ----- Edges

        if len(bl_mesh.edges):
            n = len(bl_mesh.edges)
            a = np.empty(2*n, bint)
            bl_mesh.edges.foreach_get("vertices", a)
            a = np.reshape(a, (n, 2))
            mesh.edges.append(vertex0=a[:, 0], vertex1=a[:, 1])

            del a

            mesh.edges.remove_face_edges(mesh.faces.get_edges(mesh.corners))

        # ----- Attributes

        mesh.points.from_bl_attributes(bl_mesh.attributes)
        mesh.corners.from_bl_attributes(bl_mesh.attributes)
        mesh.faces.from_bl_attributes(bl_mesh.attributes)
        mesh.edges.from_bl_attributes(bl_mesh.attributes)

        return mesh
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # To Mesh data
    # -----------------------------------------------------------------------------------------------------------------------------

    def to_mesh_data(self, data):
        """ Write the geometry into a Blender Mesh

        Arguments
        ---------
            - mesh (Blender Mesh instance) : the mesh to write
        """

        from npblender import blender

        bl_mesh = blender.get_data(data)
        bl_mesh.clear_geometry()

        # ----------------------------------------------------------------------------------------------------
        # Materials

        bl_mesh.materials.clear()
        for mat_name in self.materials:
            if mat_name is not None:
                bl_mesh.materials.append(bpy.data.materials.get(mat_name))

        # ----------------------------------------------------------------------------------------------------
        # Vertices

        points = self.points.ravel()
        if len(points):
            bl_mesh.vertices.add(len(points))

        # ----------------------------------------------------------------------------------------------------
        # Corners

        corners = self.corners.ravel()
        if len(corners):
            bl_mesh.loops.add(len(corners))
            bl_mesh.loops.foreach_set("vertex_index", blender.int_array(corners.vertex_index))

        # ----------------------------------------------------------------------------------------------------
        # Faces

        faces = self.faces.ravel()
        if len(faces):
            bl_mesh.polygons.add(len(faces))
            bl_mesh.polygons.foreach_set("loop_start", blender.int_array(faces.loop_start))
            bl_mesh.polygons.foreach_set("loop_total", blender.int_array(faces.loop_total))

        # ----------------------------------------------------------------------------------------------------
        # Edges

        edges = self.edges.ravel()
        if len(self.edges):
            # edges to add
            add_edges = self.edges.vertices

            # edges have been created by faces
            if len(faces):
                bl_mesh.update()

                cur_n = len(bl_mesh.edges)
                if cur_n > 0:
                    a = np.empty((cur_n, 2), dtype=bint)
                    bl_mesh.edges.foreach_get('vertices', a.ravel())

                add_edges = np.append(a, add_edges, axis=0)

            # add the edges
            n = len(edges)
            bl_mesh.edges.add(n)

            bl_mesh.edges.foreach_set('vertices', add_edges.ravel())

        # ----------------------------------------------------------------------------------------------------
        # Attributes

        attributes = data.attributes

        points.to_bl_attributes(attributes, update=False)
        corners.to_bl_attributes(attributes, update=False)
        faces.to_bl_attributes(attributes, update=False)
        edges.to_bl_attributes(attributes, update=False)

        # ----------------------------------------------------------------------------------------------------
        # Update

        bl_mesh.update()

    # -----------------------------------------------------------------------------------------------------------------------------
    # Initialize from an object
    # -----------------------------------------------------------------------------------------------------------------------------

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
            return cls.FromMeshData(object_eval.data)

        else:
            return cls.from_mesh_data(blender.get_object(obj).data)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # To blender object
    # -----------------------------------------------------------------------------------------------------------------------------

    def to_object(self, obj, shade_smooth=None, shapekeys=None, collection=None):
        """ Create or update a blender object.

        The method 'to_object' creates the whole geometry. It creates a new object if it doesn't already exist.
        If the object exists, it must be a mesh, there is no object type conversion.

        Once the object is created, use the method 'update_object' to change the vertices.

        Arguments
        ---------
            - obj (str or Blender object) : the object the create

        Returns
        -------
            - Blender mesh object
        """

        from npblender import blender

        res = blender.create_mesh_object(obj, collection=collection)
        self.to_mesh_data(res.data)

        if shade_smooth is not None:
            res.data.polygons.foreach_set('use_smooth', [shade_smooth]*len(res.data.polygons))

        if shapekeys is not None:
            if shapekeys is not None:
                if isinstance(shapekeys, ShapeKeys):
                    shapekeys.to_mesh_object(obj)
                else:
                    for sks in shapekeys:
                        sks.to_mesh_object(obj)

        return res
    
    # ====================================================================================================
    # bmesh edition
    # ====================================================================================================

    # -----------------------------------------------------------------------------------------------------------------------------
    # bmesh utility to pass from internal geometry to bmesh
    # -----------------------------------------------------------------------------------------------------------------------------

    def _bm_verts(self, bm):
        nverts = len(self.points.ravel())
        verts  = np.zeros(nverts, dtype=object)
        for vert in bm.verts:
            verts[vert.index] = vert
        return verts

    def _bm_edges(self, bm):
        nedges = len(bm.edges)
        edges  = np.empty(nedges, dtype=object)
        verts  = np.empty((nedges, 2), dtype=bint)

        for i_edge, edge in enumerate(bm.edges):
            edges[i_edge] = edge

            i0, i1 = edge.verts[0].index, edge.verts[1].index
            verts[i_edge] = (i0, i1) if i0 < i1 else (i1, i0)

        return edges, verts

    # -----------------------------------------------------------------------------------------------------------------------------
    # bmesh context
    # -----------------------------------------------------------------------------------------------------------------------------

    @contextmanager
    def bmesh(self, readonly=False):
        """ Acces to bmesh api.

        The example below use bmesh to offset the vertices of +1 in the x axis.

        ``` python
        mesh = Mesh.Cube()

        # Move the vertices with bmesh
        with mesh.bmesh() as bm:
            for v in bm.verts:
                v.co.x += 1.0

        # Move the vertices directy in numpy array
        mesh.points.position[:, 1] += 1

        # Cube moved along x and y
        mesh.to_object("Cube")
        ```

        Arguments
        ---------
            - readonly (bool=False) : avoid to read back the bmesh if not modications were done
        """

        data = bpy.data.meshes.get(DATA_TEMP_NAME)
        if data is None:
            data = bpy.data.meshes.new(DATA_TEMP_NAME)
        self.to_mesh_data(data)

        bm = bmesh.new()   # create an empty BMesh
        bm.from_mesh(data) # fill it in from a Mesh

        yield bm

        # ----- Back

        if not readonly:
            bm.to_mesh(data)
            self.capture(Mesh.from_mesh_data(data))

        bm.free()

    # ====================================================================================================
    # Mesh data edition
    # ====================================================================================================

    @contextmanager
    def blender_data(self, readonly=False):
        """ Acces to Blender Mesh API.

        Transfer the geometry to a temporay Blender Mesh.
        The example below use a blender Mesh to get the normals.

        ``` python
        mesh = Mesh.Cube()

        with mesh.blender_data() as data:
            normals = np.array([poly.normal for poly in data.polygons])

        print(normals)

        # > [[-1. -0.  0.]
        #    [ 0.  1.  0.]
        #    [ 1. -0.  0.]
        #    [ 0. -1.  0.]
        #    [ 0.  0. -1.]
        #    [ 0. -0.  1.]]
        ```

        Arguments
        ---------
            - readonly (bool=False) : don't read back the geometry if not modified

        Returns
        -------
            - Blender Mesh
        """

        data = bpy.data.meshes.get(DATA_TEMP_NAME)
        if data is None:
            data = bpy.data.meshes.new(DATA_TEMP_NAME)

        self.to_mesh_data(data)

        yield data

        # ----- Back

        if not readonly:
            self.capture(Mesh.FromMeshData(data))

    # ====================================================================================================
    # From something
    # ====================================================================================================

    @classmethod
    def from_model(cls, model, materials=None):

        if isinstance(model, (str, bpy.types.Object)):
            mesh = cls.from_object(model, evaluated=True)

        elif isinstance(model, dict):
            mesh = cls.from_dict(model)

        elif isinstance(model, Mesh):
            mesh = cls.from_mesh(model)

        elif isinstance(model, bpy.types.Mesh):
            mesh = cls.from_mesh_data(model)

        else:
            raise Exception(f"Mesh.from_model: 'model' type is not valid: {type(model)}")

        return mesh

    # =============================================================================================================================
    # Utility
    # =============================================================================================================================

    def get_faces_position(self, selection=None):
        if selection is None:
            return self.faces[selection].get_position(self.corners, self.points)
        else:
            return self.faces[selection].get_position(self.corners, self.points)

    # =============================================================================================================================
    # Combining
    # =============================================================================================================================

    # -----------------------------------------------------------------------------------------------------------------------------
    # Join other meshes
    # -----------------------------------------------------------------------------------------------------------------------------

    def join(self, *others):
        """ Join other Meshes.

        Arguments
        ---------
            - others (Mesh) : the Mesh to append
        """
        for other in others:

            # ----------------------------------------------------------------------------------------------------
            # Vertices

            v_ofs = len(self.points)
            self.points.extend(other.points)

            # ----------------------------------------------------------------------------------------------------
            # Corners

            c_ofs = len(self.corners)
            self.corners.extend(other.corners)
            if len(self.corners):
                self.corners.vertex_index[c_ofs:] += v_ofs

            # ----------------------------------------------------------------------------------------------------
            # Faces

            f_ofs = len(self.faces)
            self.faces.extend(other.faces)
            if len(self.faces):
                self.faces.loop_start[f_ofs:] += c_ofs

            # ----------------------------------------------------------------------------------------------------
            # Edges

            e_ofs = len(self.edges)
            self.edges.extend(other.edges)
            if len(self.edges):
                self.edges.vertex0[e_ofs:] += v_ofs
                self.edges.vertex1[e_ofs:] += v_ofs

            # ----- Materials

            remap = np.array([self.get_material_index(mat_name) for mat_name in other.materials])
            if len(remap)>0:
                self.faces.material_index[f_ofs:] = remap[other.faces.material_index]

        return self
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Multiply
    # -----------------------------------------------------------------------------------------------------------------------------

    def multiply(self, count, in_place=True):
        """ Duplicate the geometry.

        Multiplying is a way to efficiently duplicate the geometry a great number of times.
        Once duplicated, the vertices can be reshapped to address each instance individually.

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
                return type(self).from_mesh(self)
            
        if not in_place:
            return type(self).from_mesh(self).multiply(count, in_place=True)
        
        # ----------------------------------------------------------------------------------------------------
        # Vertices

        nverts = len(self.points)
        self.points.multiply(count)

        # ----------------------------------------------------------------------------------------------------
        # Corners

        ncorners = len(self.corners)
        self.corners.multiply(count)
        self.corners.vertex_index += np.repeat(np.arange(count)*nverts, ncorners)

        # ----------------------------------------------------------------------------------------------------
        # Faces

        self.faces.multiply(count)
        self.faces.update_loop_start()

        # ----------------------------------------------------------------------------------------------------
        # Edges

        nedges = len(self.edges)
        self.edges.multiply(count)
        ofs = np.repeat(np.arange(count)*nverts, nedges)
        self.edges.vertex0 += ofs
        self.edges.vertex1 += ofs

        return self
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Operators
    # -----------------------------------------------------------------------------------------------------------------------------

    def __mul__(self, count):
        return self.multiply(count, in_place=False)

    def __imul__(self, count):
        return self.multiply(count, in_place=True)

    # =============================================================================================================================
    # Editing
    # =============================================================================================================================

    # -----------------------------------------------------------------------------------------------------------------------------
    # Extract attributes per domain
    # -----------------------------------------------------------------------------------------------------------------------------

    def _attributes_per_domain(self, **attrs):

        dispatched = {
            'points'  : {},
            'corners' : {},
            'faces'   : {},
            'edges'   : {},
            }

        for k, v in attrs.items():
            count = 0

            if k in self.points.all_names:
                dispatched['points'][k] = v
                count += 1
                
            if k in self.corners.all_names:
                dispatched['corners'][k] = v
                count += 1
                
            if k in self.faces.all_names:
                dispatched['faces'][k] = v
                count += 1
                
            if k in self.edges.all_names:
                dispatched['edges'][k] = v
                count += 1

            if count == 0:
                raise AttributeError(f"Unknown mesh attribute '{k}'.\n"
                                     f"- points:  {self.points.all_names}\n"
                                     f"- corners: {self.corners.all_names}\n"
                                     f"- faces:   {self.faces.all_names}\n"
                                     f"- edges:   {self.edges.all_names}\n"
                                     )

            if count > 1:
                raise AttributeError(f"Mesh add_geometry> attribute '{k}' is ambigous, it belongs to more than one domain (count).\n"
                                     f"- points:  {self.points.all_names}\n"
                                     f"- corners: {self.corners.all_names}\n"
                                     f"- faces:   {self.faces.all_names}\n"
                                     f"- edges:   {self.edges.all_names}\n"
                                     )
        return dispatched

    # -----------------------------------------------------------------------------------------------------------------------------
    # Add geometry
    # -----------------------------------------------------------------------------------------------------------------------------

    def add_geometry(self, points=None, corners=None, faces=None, edges=None, safe_mode=False, **attrs):
        """ Add geometry

        Note that the added geometry can refer to existing vertices. It is appended as is, whithout shifting
        indices.

        To add indenpendant geometry, use_join geometry.

        ``` python
        cube = Mesh.cube()
        # add a triangle on existing vertices
        # corners argument refers to cube vertices
        cube.add_geometry(corners=[0, 1, 2], faces=3)

        # add a triangle with additional vertices
        # corners argument refers to the new vertices
        cube.join_geometry(points=[[0, 0, 0], [0, 1, 0], [1, 0, 0]], corners=[0, 1, 2], faces=3)
        ```

        Arguments
        ---------
            - points (array of vectors = None) : the vertices
            - corners (array of ints = None) : corners, i.e. indices on the array of points
            - sizes (array of ints = None) : size of the faces, the sum of this array must be equal to the length of the corners array
            - materials (str or list of strs = None) : list of materials used in the geometry
            - **attrs (dict) : other geometry attributes

        Returns
        -------
            - dict : {'points', 'corners', 'faces', 'edges': added geometry indices}
        """

        disp_attrs = self._attributes_per_domain(**attrs)
        added = {'points': [], 'corners': [], 'faces': [], 'edges': []}

        # ----------------------------------------------------------------------------------------------------
        # Add vertices
        # ----------------------------------------------------------------------------------------------------

        if points is not None:
            added['points'] = self.points.append(position=points, **disp_attrs['points'])

        # ----------------------------------------------------------------------------------------------------
        # Edges
        # ----------------------------------------------------------------------------------------------------

        if edges is not None:
            if np.shape(edges) == (2,):
                added['edges'] = self.edges.append(vertex0=edges[0], vertex1=edges[1], **disp_attrs['edges'])
            else:
                added['edges'] = self.edges.append(vertex0=edges[:, 0], vertex1=edges[:, 1], **disp_attrs['edges'])

        # ----------------------------------------------------------------------------------------------------
        # Corners and Faces
        # ----------------------------------------------------------------------------------------------------

        ok = False
        if faces is None:
            if corners is None:
                return
            else:
                ok = False
        else:
            ok = corners is not None

        if not ok:
            raise AttributeError("Mesh add_geometry> corners and sizes must be both None or both not None.")
        
        if np.shape(faces) == ():
            nfaces = len(corners) // faces
            if len(corners) % faces != 0:
                raise ValueError(f"Mesh add_geometry> when faces is a single number {faces}, it must divide the number of corners ({len(corners)}).")
            faces = np.ones(nfaces, dtype=bint)*faces
        
        added['corners'] = self.corners.append(vertex_index=corners, **disp_attrs['corners'])
        added['faces'] = self.faces.append_sizes(faces, **disp_attrs['faces'])

        if safe_mode:
            self.check()

        return added
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Join geometry
    # -----------------------------------------------------------------------------------------------------------------------------

    def join_geometry(self, points=None, corners=None, faces=None, edges=None, safe_mode=False, **attrs):
        """ Join geometry defined by components.

        The geometry passed in argument is consistent and doesn't refer to existing vertices. It is used
        to build an independant mesh which is then joined.
        See 'add_geometry' which, on the contrary, can refer to existing vertices.

        Returns
        -------
            - self
        """
        mesh = Mesh().join_attributes(self).add_geometry(
            points=points, 
            corners=corners, 
            faces=faces, 
            edges=edges,
            **attrs)
        self.join_mesh(mesh)
        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Add Vertices
    # -----------------------------------------------------------------------------------------------------------------------------

    def add_points(self, points,  **attributes):
        """ Add vertices.

        Arguments
        ---------
            - verts (array of vectors) : the vertices to add
            - attributes (name=value) : value for named attributes

        Returns
        -------
            - array of ints : indices of the added vertices
        """
        npoints = len(self.points)
        return self.points.append(position=points, **attributes)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Add edges
    # -----------------------------------------------------------------------------------------------------------------------------

    def add_edges(self, vertices, **attributes):
        nedges = len(self.edges)
        return self.edges.append(vertex0=vertices[..., 0], vertex1=vertices[..., 1])

    # -----------------------------------------------------------------------------------------------------------------------------
    # Add faces
    # -----------------------------------------------------------------------------------------------------------------------------

    def add_faces(self, corners, faces, use_offset=True, **attributes):
        """ Add faces.

        Faces can be added in different formats:
            - a list or 1D-array of ints : a single face
            - a list of list of ints : a list of faces
            - a 2D-array of ints : a list of faces of the same size
            - a tuple (array of ints, array of ints) : sizes and corners

        The offset_index is the value to add to the vertex indices. It allows to add geometry faces defined from
        index 0 to a geometry with existing vertices.

        ``` python
        # ----- One face at a time

        mesh = Mesh()

        # Create 8 vertices form a cube
        mesh.add_verts(Mesh.Cube().points.position)

        # list of ints
        mesh.add_faces([0, 1, 3, 2])

        # tuple of ints
        mesh.add_faces((2, 3, 7, 6))

        # array of ints
        mesh.add_faces(np.array((6, 7, 5, 4)))

        obj = mesh.to_object("Single faces")
        obj.location.x = -3

        # ----- Several faces

        mesh = Mesh()

        # Create 8 vertices form a cube
        mesh.add_verts(Mesh.Cube().points.position)

        # list of lists
        mesh.add_faces( [[0, 1, 3, 2], [2, 3, 7, 6], [6, 7, 5, 4]])

        # Structured array
        faces = np.reshape([4, 5, 1, 0, 2, 6, 4, 0, 7, 3, 1, 5], (3, 4))
        mesh.add_faces(faces)

        mesh.to_object("List of faces")

        # ----- Corners and sizes

        mesh = Mesh()

        # Create 8 vertices form a cube
        mesh.add_verts(Mesh.Cube().points.position)

        mesh.add_faces((
            [0, 1, 3, 2, 2, 3, 7, 6, 6, 7, 5, 4, 4, 5, 1, 0, 2, 6, 4, 0, 7, 3, 1, 5],
            [4, 4, 4, 4, 4, 4],
            ))

        obj = mesh.to_object("Corners, faces")
        obj.location.x = 3
        ```

        Arguments
        ---------
            - faces (various) : the faces to add
            - mat (int or array of ints = 0) : material indices of the faces
            - offset (int) : the offset to add to the vertex indices.
            - attributes (name=value) : value for named attributes

        Returns
        -------
            - array of ints : indices of the created faces
        """
        nfaces = len(self.faces)
        return self.add_geometry(faces=faces, corners=corners, **attributes)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Join geometry
    # -----------------------------------------------------------------------------------------------------------------------------

    def join_geometry(self, points=None, corners=None, faces=None, edges=None, safe_mode=False, **attrs):
        """ Join geometry defined by components.

        The geometry passed in argument is consistent and doesn't refer to existing vertices. It is used
        to build an independant mesh which is then joined.
        On the contrary 'add_geometry' can use existing vertices.

        Returns
        -------
            - self
        """
        mesh = Mesh().join_fields(self)
        mesh.add_geometry(
            points=points, 
            corners=corners, 
            faces=faces, 
            edges=edges,
            **attrs)
        self.join(mesh)
        return self
    
    # =============================================================================================================================
    # Split edges
    # =============================================================================================================================

    def split_edges(self, loop0, loop1, cuts=1):

        n0 = 1 if np.shape(loop0) == () else len(loop0)
        n1 = 1 if np.shape(loop1) == () else len(loop1)
        n = max(n0, n1)
        if n0 == n:
            loop0 = np.atleast_1d(loop0, dtype=bint)
        else:
            loop0 = np.ones(n, dtype=bint)*loop0

        if n1 == n:
            loop1 = np.atleast_1d(loop1, dtype=bint)
        else:
            loop1 = np.ones(n, dtype=bint)*loop1

        to_select = np.stack([np.minimum(loop0, loop1), np.maximum(loop0, loop1)], axis=-1)  # shape (p, 2)
        with self.bmesh() as bm:

            edges, verts_indices = self._bm_edges(bm)

            # select edges in to_select
            dtype = np.dtype([('a', bint), ('b', bint)])
            edges_view = verts_indices.view(dtype)
            to_select_view = to_select.view(dtype)

            # selection mask
            mask = np.isin(edges_view.ravel(), to_select_view.ravel())

            edges_to_cut = list(edges[mask])

            if len(edges_to_cut) == 0:
                return

            # Edges subidivision
            bmesh.ops.subdivide_edges(
                bm,
                edges=edges_to_cut,
                cuts=cuts,
                use_grid_fill=False
            )

    # =============================================================================================================================
    # Bridge loops
    # =============================================================================================================================

    def bridge_loops(self, loop0, loop1, close=False, segments=1, **attributes):
        """ Create a grid linking two loops of the same size.

        Arguments
        ---------
        - loop0 (array of ints) : the first loop
        - loop1 (array of ints) : the second loop
        - segments (int = 1) : the number of segments to join the loops
        - attributes (dict) : attributes to add to the mesh
        """
        segments = max(1, segments)

        if close:
            loop0 = np.append(loop0, loop0[0]).astype(bint)
            loop1 = np.append(loop1, loop1[0]).astype(bint)
        else:
            loop0 = np.asarray(loop0, dtype=bint)
            loop1 = np.asarray(loop1, dtype=bint)

        sel0 = np.stack((loop0[:-1], loop0[1:]), axis=-1)
        sel1 = np.stack((loop1[:-1], loop1[1:]), axis=-1)

        sel0 = np.sort(sel0, axis=-1)
        sel1 = np.sort(sel1, axis=-1)

        with self.bmesh() as bm:

            edges, verts_indices = self._bm_edges(bm)

            # Prepare edges view
            dtype = np.dtype([('a', bint), ('b', bint)])
            edges_view = verts_indices.view(dtype)

            # First loop
            sel0_view = sel0.view(dtype)
            mask = np.isin(edges_view.ravel(), sel0_view.ravel())

            edges_to_bridge = list(edges[mask])

            # Second loop
            sel1_view = sel1.view(dtype)
            mask = np.isin(edges_view.ravel(), sel1_view.ravel())

            edges_to_bridge.extend(list(edges[mask]))

            if len(edges_to_bridge) == 0:
                return

            # Bridge
            res = bmesh.ops.bridge_loops(
                bm, 
                edges=edges_to_bridge,
                #use_cyclic = close,
                #segments=segments,
                )
            
            # Grid
            if segments > 1:
                bmesh.ops.subdivide_edges(
                    bm,
                    edges=res['edges'],
                    cuts=segments - 1,
                    use_grid_fill=False
                )

    # =============================================================================================================================
    # Fill cap
    # =============================================================================================================================

    def fill_cap(self, loop, mode='NGON', center=None, segments=1, clockwise=False, **attributes):
        """ Fill a cap between an existing loop

        For NGON mode, center is not required.
        For FANS mode, a center point is required:
        - if center is None, it is computed as the center of the loop
        - if center is an int, it is the index of the point to use
        - otherwise, the center is considered as the point to use

        Arguments
        ---------
        - loop (array of ints) : the circle loop
        - mode (FillCap = 'NGON') : NGON or FANS
        - center (point index or point or None) : center of the cap
        - clockwise (bool = False) : order of the corners in the loop
        - indexing (str = 'ij') : indexing of the points
        - attributes (dict) : attributes to add to the mesh (see Geometry.add_geometry(
        """
        if mode == 'NONE':
            # To have en empty dict
            return self.add_geometry()

        elif mode == 'NGON':
            return self.add_geometry(corners=loop, faces=len(loop), UVMap=self.disk_uv_map(len(loop), mode='NGON', clockwise=clockwise), **attributes)
        
        elif mode == 'FANS':
            
            if center is None:
                verts = self.points.position[loop]
                center = np.average(verts, axis=0)
                center_index = len(self.points)
                cind = self.add_points(center)

            elif isinstance(center, (int, np.int32, np.int64)):
                center_index = center
                cind = [center_index]

            else:
                center_index = len(self.points)
                cind = self.add_points(center)

            indices = np.append(loop, center_index)

            topo = fans_corners(len(loop), close=True, clockwise=clockwise).ravel()

            corners = indices[topo]
            uv_map = disk_uv_map(len(loop), mode='FANS', clockwise=clockwise).reshape(-1, 2)

            added = self.add_geometry(corners=corners, faces=3, UVMap=uv_map, **attributes)
            added['points'] = np.asarray(cind)

            # More than one segments

            if segments > 1:
                self.split_edges(loop, center_index, cuts=segments - 1)

            return added
        
        assert(False)

    # ====================================================================================================
    # Deletion
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Delete faces
    # ----------------------------------------------------------------------------------------------------

    def delete_faces(self, selection):
        """ Delete only faces.
        """
        self.faces.delete_loops(selection, self.corners)

    # ----------------------------------------------------------------------------------------------------
    # Delete vertices
    # ----------------------------------------------------------------------------------------------------

    def delete_vertices(self, points=None, faces=None, edges=None):
        """Delete vertices.

        Arguments
        ---------
            points : array selection, optional
                Vertex indices to delete directly.
            faces : array selection, optional
                Faces owning vertices to delete.
            edges : array selection, optional
                Edges owning vertices to delete.
        """
        go = False
        if points is not None:
            p_sel = set(np.arange(len(self.points))[points])
            go = True
        if faces is not None:
            f_sel = set(np.arange(len(self.faces))[faces])
            go = True
        if edges is not None:
            e_sel = set(np.arange(len(self.edges))[edges])
            go = True

        if not go:
            return 

        with self.bmesh() as bm:
            verts_to_delete = set()

            for vert in bm.verts:
                if points is not None and vert.index in p_sel:
                    verts_to_delete.add(vert)
                    continue

                if faces is not None:
                    if any(f.index in f_sel for f in vert.link_faces):
                        verts_to_delete.add(vert)
                        continue

                if edges is not None:
                    if any(e.index in e_sel for e in vert.link_edges):
                        verts_to_delete.add(vert)
                        continue

            bmesh.ops.delete(bm, geom=list(verts_to_delete), context='VERTS')

    # ====================================================================================================    
    # Blender primitives
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Grid
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def bl_grid(cls, x_segments=1, y_segments=1, size=2, materials=None):
        """ Create a Grid.

        Blender constructor for a Grid.

        Arguments
        ---------
            - x_segments (int=1) : number of segments along x axis
            - y_segments (int=1) : number of segments along y axis
            - size (float or tuple of floats = 1. : size of the grid
            - materials (list of strs = None) : materials list

        Returns
        -------
            - Mesh
        """

        mesh = cls(materials=materials)
        with mesh.bmesh() as bm:
            bmesh.ops.create_grid(bm, x_segments=x_segments, y_segments=y_segments, size=size, calc_uvs=True)

        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # Circle
    # ----------------------------------------------------------------------------------------------------
    
    @classmethod
    def bl_circle(cls, radius=1, segments=16, fill_tris=False, materials=None):
        """ Create a Circle.

        Blender constructor for a Circle.

        Arguments
        ---------
            - radius (float=1.) : circle radius
            - segments (int=16) : number of segments
            - fill_tris (bool = False) : fill with triangle fans
            - materials (list of strs = None) : materials list
            - transformation (Transformations = None) : the transformation to apply

        Returns
        -------
            - Mesh
        """

        mesh = cls(materials=materials)
        with mesh.bmesh() as bm:
            bmesh.ops.create_circle(bm, cap_ends=True, cap_tris=fill_tris, segments=segments, radius=radius, calc_uvs=True)

        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # Cone
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def bl_cone(cls, radius1=1, radius2=0, depth=2, segments=16, side_segments=1, cap_ends=True, cap_tris=False, materials=None):
        """ Create a Cone.

        Blender constructor for a Cone.

        Arguments
        ---------
            - radius1 (float=1.) : base radius
            - radius2 (float=0.) : top radius
            - depth (float=2.) : cone height
            - segments (int=16) : number of segments
            - cap_ends (bool=True) : fill cap faces
            - cap_tris (bool = False) : fill with triangle fans
            - materials (list of strs = None) : materials list

        Returns
        -------
            - Mesh
        """

        mesh = cls(materials=materials)
        with mesh.bmesh() as bm:
            res = bmesh.ops.create_cone(bm, cap_ends=cap_ends, cap_tris=cap_tris, segments=segments, radius1=radius1, radius2=radius2, depth=depth, calc_uvs=True)
            if side_segments > 1:

                edges_to_cut = [
                    e for e in bm.edges
                    if (e.verts[0].co.z > 0 and e.verts[1].co.z < 0)
                    or (e.verts[0].co.z < 0 and e.verts[1].co.z > 0)
                ]

                # Edges subidivision
                bmesh.ops.subdivide_edges(
                    bm,
                    edges         = edges_to_cut,
                    cuts          = side_segments - 1,
                    use_grid_fill = False
                )

        return mesh

    # ====================================================================================================    
    # Primitives
    # ====================================================================================================    


    # ----------------------------------------------------------------------------------------------------
    # Points
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def points(cls, points=None, materials=None):
        """ Create a mesh with points at the given positions.

        Arguments
        ---------
            - position (shaped array (?, 3)=(0, 0, 0) : position of the points

        Returns
        -------
            - mesh (Mesh) : the mesh
        """
        return cls(points=points, materials=materials)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Line
    # -----------------------------------------------------------------------------------------------------------------------------

    @classmethod
    def line(cls, start=(0, 0, 0), end=(0, 0, 1), segments=1, materials=None):
        """ Create a mesh with lines between the given positions.

        Arguments
        ---------
            - start (vertex =(0, 0, 0) : position of the start of the lines
            - end (vertex=(0, 0, 1) : position of the end of the lines
            - count (int=2) : number of points in the line

        Returns
        -------
            - mesh (Mesh) : the mesh
        """
        segments = max(1, segments)

        start = np.asarray(start, dtype=bfloat)
        end   = np.asarray(end, dtype=bfloat)

        pos = np.linspace(start, end, segments + 1) # shape (segments, n points, 3)

        if len(pos.shape) == 2:
            edges = border_edges(np.arange(segments))
        else:
            shape = pos.shape[:-1]
            n = int(np.prod(shape))
            edges = col_edges(np.arange(n).reshape(shape))

        return cls(points=pos.reshape(-1, 3), edges=edges.reshape(-1, 2), materials=materials)

    # ----------------------------------------------------------------------------------------------------
    # Grid
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def grid(cls, size_x=1, size_y=1, vertices_x=3, vertices_y=3, materials=None):
        """ Create a Grid.

        > [!IMPORTANT]
        > Grid is created with indexing = 'ij': shape = (resolution x, resolution y)

        Arguments
        ---------
            - size_x (float=1) : size along x
            _ size_y (float=1) : size along y
            - vertices_x (int=3) : number of vertices along x
            - vertices_y (int=3) : number of vertices along y
            - materials (list of strs = None) : materials list

        Returns
        -------
            - Mesh
        """

        nx = max(2, vertices_x)
        ny = max(2, vertices_y)

        topo = grid_corners(nx, ny).ravel()

        # ----------------------------------------------------------------------------------------------------
        # Base grid for points and uvs

        x, y = np.meshgrid(
            np.linspace(-size_x/2, size_x/2, nx, dtype=bfloat), 
            np.linspace(-size_y/2, size_y/2, ny, dtype=bfloat), 
            indexing='ij')
        x, y = x.ravel(), y.ravel()

        # ----------------------------------------------------------------------------------------------------
        # Let 's build the grid

        points = np.stack((x, y, np.zeros_like(x)), axis=-1)
        corners = np.arange(len(points))[topo]
        faces = 4
        uvmap = grid_uv_map(nx, ny).reshape(-1, 2)

        return cls(points, corners, faces, materials=materials, UVMap=uvmap)

    # ----------------------------------------------------------------------------------------------------
    # Cube
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def cube(cls, size=2, materials=None):
        """ Create a Cube.

        Arguments
        ---------
            - size (float=1.) : size of the cube
            - materials (list of strs = None) : materials list

        Returns
        -------
            - Mesh
        """

        verts = [[-1., -1., -1.], [-1., -1.,  1.], [-1.,  1., -1.], [-1.,  1.,  1.], [ 1., -1., -1.], [ 1., -1.,  1.], [ 1.,  1., -1.], [ 1.,  1.,  1.],]
        corners = [0, 1, 3, 2,  2, 3, 7, 6,  6, 7, 5, 4,  4, 5, 1, 0,  2, 6, 4, 0,  7, 3, 1, 5]
        faces = [4]*6
        uvs   = [[0.375, 0.000], [0.625, 0.000], [0.625, 0.250], [0.375, 0.250], [0.375, 0.250], [0.625, 0.250], [0.625, 0.500], [0.375, 0.500],
                 [0.375, 0.500], [0.625, 0.500], [0.625, 0.750], [0.375, 0.750], [0.375, 0.750], [0.625, 0.750], [0.625, 1.000], [0.375, 1.000],
                 [0.125, 0.500], [0.375, 0.500], [0.375, 0.750], [0.125, 0.750], [0.625, 0.500], [0.875, 0.500], [0.875, 0.750], [0.625, 0.750], ]


        mesh = cls(points=verts, corners=corners, faces=faces, materials=materials, UVMap=uvs)

        size = np.asarray(size)
        mesh.points.position *= size/2

        return mesh

    # ----------------------------------------------------------------------------------------------------
    # Circle
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def circle(cls, radius=1, segments=16, fill_segments=0, cap='NONE', materials=None):
        """ Create a Circle.

        'fill_segments' argument gives the number of internal circles to create.
        If zero, the circle if filled with a polygon.
        If positive, the circle is filled with triangle fans.

        Arguments
        ---------
            - radius (float=1.) : circle radius
            - segments (int=16) : number of segments
            - fill_segments (int = 0) : number of internal segments, polygon is None
            - materials (list of strs = None) : materials list

        Returns
        -------
            - Mesh
        """
        ag = np.linspace(0, 2*np.pi, segments, endpoint=False)
        x = radius*np.cos(ag)
        y = radius*np.sin(ag)
        points = np.stack((x, y, np.zeros_like(x)), axis=-1)

        if cap == 'NONE':
            i = np.arange(segments)
            edges = np.stack((i, np.roll(i, shift=-1)), axis=-1)
            return cls(points=points, edges=edges)
        
        elif cap == 'NGON':
            return cls(
                points      = points, 
                corners     = np.arange(segments), 
                faces       = segments,
                UVMap       = disk_uv_map(segments, mode='NGON').reshape(-1, 2),
                materials   = materials,
            )
        
        elif cap == 'FANS':
            points = np.append(points, [[0, 0, 0]], axis=0)
            topo = fans_corners(segments)

            return cls(
                points      = points, 
                corners     = topo.ravel(), 
                faces       = 3,
                UVMap       = disk_uv_map(segments, mode='FANS').reshape(-1, 2),
                materials   = materials,
            )

        #mesh = cls(points=points)
        #mesh.fill_cap(np.arange(segments), mode=cap, center=segments, clockwise=True)

        assert(False)
    
    # ----------------------------------------------------------------------------------------------------
    # Disk
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def disk(cls, radius=1, segments=16, fill_segments=0, cap='NGON', materials=None):
        """ Create a disk.

        Same as circle but default cap is NGON
        """    
        return cls.circle(radius=radius, segments=segments, fill_segments=fill_segments, cap=cap, materials=materials)
    
    # ----------------------------------------------------------------------------------------------------
    # Cone
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def cone(cls, vertices=32, side_segments=1, fill_segments=1, radius_top=0, radius_bottom=1, depth=2, fill_type='NGON', materials=None):
        """ Create a Cone.

        Arguments
        ---------
            - vertices (int=16) : number of segments
            - side_segments (int = 1) : number of vertical segments
            - fill_segments (int = 1) : number of internal circles on the caps
            - radius_top (float=0) : top radius
            - radius_bottom (float=1) : bottom radius
            - depth (float=2.) : cylinder height
            - fill_type (str or couple of strs ='NGON' in 'NGON', 'FANS', 'NONE') : cap filling
            - materials (list of strs = None) : materials list

        Returns
        -------
            - Mesh
        """
        # Empty geometry if fhe two radius are null
        if radius_top == 0 and radius_bottom == 0:
            return cls()
        
        mesh = cls.bl_cone(
            radius1       = radius_bottom,
            radius2       = radius_top,
            segments      = vertices,
            side_segments = side_segments,
            depth         = depth,
            cap_ends      = fill_type in ['NGON', 'FANS'],
            cap_tris      = fill_type == 'FANS',
            materials     = materials,
        )

        return mesh

    # ----------------------------------------------------------------------------------------------------
    # Cylinder
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def cylinder(cls, vertices=32, side_segments=1, radius=1, depth=2, fill_type='NGON', materials=None):
        """ Create a Cylinder.

        Arguments
        ---------
            - vertices (int=16) : number of segments
            - side_segments (int = 1) : number of vertical segments
            - radius (float=1.) : radius
            - depth (float=2.) : cylinder height
            - fill_type (str or couple of strs ='NGON' in 'NGON', 'TRIANGLE_FAN', 'NONE') : cap filling
            - materials (list of strs = None) : materials list

        Returns
        -------
            - Mesh
        """
        mesh = cls.bl_cone(
            radius1       = radius,
            radius2       = radius,
            segments      = vertices,
            side_segments = side_segments,
            depth         = depth,
            cap_ends      = fill_type in ['NGON', 'FANS'],
            cap_tris      = fill_type == 'FANS',
            materials     = materials,
        )

        return mesh

    # ----------------------------------------------------------------------------------------------------
    # Pyramid
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def pyramid(cls, size=1, materials=None):
        """ Create a Pyramid.

        Arguments
        ---------
            - size (float=1.) : size

        Returns
        -------
            - Mesh
        """
        return cls.cone(vertices=3, radius_bottom=size*0.8660254037844386, depth=size, materials=materials)

    # ----------------------------------------------------------------------------------------------------
    # UV Sphere
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def uvsphere(cls, segments=32, rings=16, radius=1, materials=None):
        """ Create a uv sphere.

        Arguments
        ---------
            - segments (int=32) : number of segments
            - rings (int=16) : number of rings
            - radius (float=1.) : radius
            - materials (list of strs = None) : materials list
            - transformation (Transformations = None) : the transformation to apply

        Returns
        -------
            - Mesh
        """
        mesh = cls(materials=materials)
        with mesh.bmesh() as bm:
            bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius, calc_uvs=True)

        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # IcoSphere
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def icosphere(cls, radius=1, subdivisions=2, materials=None):
        """ Create a IcoSphere.

        Blender constructor for a IcoSphere.

        Arguments
        ---------
            - radius (float=1.) : radius
            - subdivisions (int=2) : number subdivisions
            - materials (list of strs = None) : materials list
            - transformation (Transformations = None) : the transformation to apply

        Returns
        -------
            - Mesh
        """
        subdivisions = min(10, subdivisions)

        mesh = cls(materials=materials)
        with mesh.bmesh() as bm:
            bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=radius, calc_uvs=True)

        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # Torus
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def torus(cls, major_segments=48, minor_segments=12, major_radius=1., minor_radius=0.25, materials=None):
        """ Create a Torus.

        Arguments
        ---------
            - major_segments (int=48) : number of segments for the major radius
            - minor_segments (int=12) : number of segments for the minor radius
            - major_radius (float=1.) : major radius
            - minor_radius (float=.25) : minor radius
            - materials (list of strs = None) : materials list
            - transformation (Transformations = None) : the transformation to apply

        Returns
        -------
            - Mesh
        """
        # Major backbone
        maj_ag = np.linspace(0, 2*np.pi, major_segments, endpoint=False, dtype=bfloat) + np.pi # + pi to match blender uv

        x = major_radius*np.cos(maj_ag)
        y = major_radius*np.sin(maj_ag)
        zeros = np.zeros_like(x)

        transfos = Transformation.from_components(
            translation=np.stack((x, y, zeros), axis=-1), 
            rotation=Rotation.from_euler(np.stack((zeros, zeros, maj_ag), axis=-1)),
        )

        # Minor section
        min_ag = np.linspace(0, 2*np.pi, minor_segments, endpoint=False, dtype=bfloat) + np.pi # + pi to match blender uv
        x = minor_radius*np.cos(min_ag)
        z = minor_radius*np.sin(min_ag)
        y = np.zeros_like(x)

        circle = np.stack((x, y, z), axis=-1)

        # Transform the circle
        points = transfos[:, None] @ circle

        # Generate the faces
        corners = grid_corners(major_segments, minor_segments, close_x=True, close_y=True)

        # Get the uv map
        uvmap = grid_uv_map(major_segments + 1, minor_segments + 1)

        return cls(points=points.reshape(-1, 3), corners=corners.ravel(), faces=4, UVMap=uvmap.reshape(-1, 2), materials=materials)
    
    # ----------------------------------------------------------------------------------------------------
    # Monkey
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def monkey(cls, materials=None):
        """ Create a Monkey.

        Arguments
        ---------
            - materials (list of strs = None) : materials list

        Returns
        -------
            - MeshBuilder
        """
        mesh = cls(materials=materials)
        with mesh.bmesh() as bm:
            bmesh.ops.create_monkey(bm)

        return mesh

    # ----------------------------------------------------------------------------------------------------
    # Arrow
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def arrow(cls, vector=(0, 0, 1), radius=.05, angle=24., segments=8, adjust_norm=None, materials=None):

        height = np.linalg.norm(vector)
        if type(adjust_norm).__name__ == 'function':
            height = adjust_norm(height)
        elif adjust_norm is not None:
            height = min(adjust_norm, height)

        head_radius = 3*radius
        head_height = head_radius/np.tan(np.radians(angle))

        cyl_height = height - head_height*.8

        #cyl  = cls.cylinder(vertices=segments, side_segments=2, radius=radius, depth=cyl_height, transformation=Transformations(position=(0, 0, cyl_height/2)), materials=materials)
        cyl = cls.cylinder(vertices=segments, side_segments=2, radius=radius, depth=cyl_height, materials=materials)
        cyl.points.z += cyl_height/2
        cyl.points[[segments + i for i in range(segments)]].position -= (0, 0, cyl_height/2 - .01)

        cone = cls.cone(vertices=segments, side_segments=2, fill_segments=1, radius_top=0, radius_bottom=head_radius, depth=head_height, fill_type='FANS', materials=materials)
        cone.points[-1].position += (0, 0, head_height/10)
        cone.points.position += (0, 0, height - head_height/2)

        arrow = cyl.join(cone)
        #arrow.points.position = tracker(vector, track_axis='Z') @ arrow.points.position
        arrow.points.position = Rotation.look_at((0, 0, 1), vector) @ arrow.points.position

        return arrow
    
    # ----------------------------------------------------------------------------------------------------
    # Field of vectors
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def vectors_field(cls, locations, vectors, radius=.05, scale_length=1., angle=24., segments=8, head=None, adjust_norm=None, materials=None):
        """ Create an arrow at each location corresponding to the vectors.

        The arrow length is equal to the corresponding vector lengths.
        The arrow radius is constant and equal to the value passe in argument for lengths greater that
        the argument scale_length. When the length is less than this value, the arrow is scaled down.

        Use the adjust_norm argument to transform the vector lengths to arrows lengths.

        Arguments
        ---------
            - locations (array of 3-vectors) : vectors locations
            - vectors (array of 3 vectors) : vectors to visualize
            - radius (float = .05) : arrow radius
            - angle (float = 24) : head radius in degrees
            - segments (int = 8) : number of segments for the section
            - head (mesh = None) : mesh model for the head. Create a cone if None
            - adjust_norm (max length or function = None) : max arrow length or function transforming
                the vector length into arrow length
            - scale_length (float = 1.) : arrow length below which the arrow radius is scaled

        Returns
        -------
            - Mesh Object
        """
        locations = np.atleast_2d(locations)
        vectors = np.atleast_2d(vectors)

        # ---------------------------------------------------------------------------
        # Vector lengths
        # ---------------------------------------------------------------------------

        n = len(locations)

        lengths = np.linalg.norm(vectors, axis=-1)
        is_null = lengths < .00001
        lengths[is_null] = 1
        v_dir = vectors / lengths[:, None]
        if type(adjust_norm).__name__ == 'function':
            lengths = adjust_norm(lengths)
        elif adjust_norm is not None:
            lengths = np.minimum(adjust_norm, lengths)
        lengths[is_null] = 0
        vectors = v_dir*lengths[:, None]

        # ---------------------------------------------------------------------------
        # Arrow head
        # ---------------------------------------------------------------------------

        head_radius = 3*radius
        head_height = head_radius/np.tan(np.radians(angle))

        if head is None:
            cone = cls.cone(
                vertices = segments, 
                side_segments = 1, 
                fill_segments = 1, 
                radius_top = 0, 
                radius_bottom = head_radius, 
                depth = head_height, 
                fill_type = 'FANS', 
                materials = materials)
            cone.points[-1].z += head_height/10
        else:
            cone = head
            head_height = cone.bounding_box_dims[2]

        # Head top point is z=0
        v0, v1 = cone.bounding_box
        cone.points.z -= v1[2]

        # ---------------------------------------------------------------------------
        # Small arrows: Vectors whose length < min_length
        # ---------------------------------------------------------------------------

        # Minimum length
        # Below this length, the arrow is scaled

        min_length = 2*head_height

        # Small and long arrows if any
        small_arrows = cls()
        long_arrows = cls()

        small = lengths < min_length
        nsmalls = np.sum(small)

        if nsmalls:
            # shaft height = head height
            cyl_height = min_length/2 + .01
            arrow = cls.cylinder(vertices=segments, side_segments=1, radius=radius, depth=cyl_height, materials=materials)
            arrow.points.z += cyl_height/2

            # Join the head
            c = cls.from_mesh(cone)
            c.points.z += min_length

            arrow.join(c)

            # Duplicate the geometry            
            small_arrows = arrow*nsmalls
            small_arrows.points.reshape(nsmalls, len(arrow.points))

            # Rotate, scale and translate
            lg = lengths[small]
            scale = np.stack((np.ones_like(lg), np.ones_like(lg), lg), axis=-1)
            small_arrows.transformation(
                rotation = Rotation.look_at((0, 0, 1), vectors[small])[:, None], 
                scale = scale[:, None], 
                translation = locations[small, None],
                )
            small_arrows.points.reshape(-1)

            
        # ---------------------------------------------------------------------------
        # Long arrows
        # ---------------------------------------------------------------------------

        long = np.logical_not(small)
        nlongs = len(locations) - nsmalls
        if nlongs:

            # Shaft model with a normalized height = 1
            shaft = cls.cylinder(vertices=segments, side_segments=1, radius=radius, depth=1, materials=materials)
            shaft.points.z += .5

            # We duplicate and transform the shafts with a scale long z
            long_arrows = shaft*nlongs
            long_arrows.points.reshape(nlongs, len(shaft.points))

            lg = lengths[long] - head_height + .01
            scale = np.stack((np.ones_like(lg), np.ones_like(lg), lg), axis=-1)
            long_arrows.transformation(
                rotation = Rotation.look_at((0, 0, 1), vectors[long])[:, None], 
                scale = scale[:, None], 
                translation = locations[long, None],
                )
            long_arrows.points.reshape(-1)
            
            # We duplicate and tranform the heads with no scale
            heads = cone*nlongs
            heads.points.reshape(nlongs, len(cone.points))
            heads.transformation(
                rotation = Rotation.look_at((0, 0, 1), vectors[long])[:, None], 
                translation = locations[long, None] + vectors[long, None],
                )
            heads.points.reshape(-1)
    
            long_arrows.join(heads)

        # Let's join the result
        arrows = cls()
        arrows.join(small_arrows, long_arrows)

        return arrows

    # ----------------------------------------------------------------------------------------------------
    # Chain Link
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def chain_link(cls, major_segments=48, minor_segments=12, radius=1., section=0.5, length=4., materials=None):
        """ Create a chain link.

        ``` python
        # ----- Some maths

        # Chain follows a catenary curve
        def catenary(t):
            return np.stack((t, np.zeros_like(t), np.cosh(t)), axis=-1)

        # Orientation is given by the derivative
        def derivative(t):
            dt = 1/10000
            return (catenary(t + dt) - catenary(t - dt))/(2*dt)

        # Catenary length
        pts = catenary(np.linspace(-1, 1, 1000))
        cat_s = np.cumsum(np.linalg.norm(pts[1:] - pts[:-1], axis=-1))
        cat_len = cat_s[-1]

        # Catenary inverse : t from length
        def cat_inverse(l):
            return 2*np.argmin(np.abs(cat_s - l))/1000 - 1

        # ----- Let's build the geometry

        # One chain link
        section = .02
        length  = .15
        link = MeshBuilder.ChainLink(radius=.04, section=section, length=length)

        # Link length taking into account the section
        l = length - 2*section

        # Number of links
        count = round(cat_len / l)

        # The chain
        chain = link*count

        # Rotate pi/2 one on two
        eulers = Eulers(shape=count)
        eulers[[i % 2 == 1 for i in range(count)]] = (0, np.pi/2, 0)

        chain.rotate(eulers)
        chain.rotate_z(np.pi/2)

        # Location of each link
        t = np.array([cat_inverse(l*i) for i in range(count)])

        chain.toward(derivative(t), track_axis='X', up_axis='Z')
        chain.translate(catenary(t))

        # To object
        chain.to_object("Catenary")
        ```

        Arguments
        ---------
            - major_segments (int=48) : number of segments for the link
            - minor_segments (int=12) : number of segments for the section
            - radius (float=1.) : radius of the link
            - section (float=.5) : section (diameter)
            - length (float=4.) : total length of the link
            - materials (list of strs = None) : materials list

        Returns
        -------
            - MeshBuilder
        """

        delta = length - 2*radius

        # Starting from a torus
        clink = cls.torus(
            major_radius    = radius, 
            minor_radius    = section/2, 
            major_segments  = major_segments, 
            minor_segments  = minor_segments, 
            materials       = materials
        )
        # Nothing else
        if delta < radius/10:
            return clink
        
        epsilon = radius/major_segments

        # Delete half of the points
        clink.delete_vertices(points=clink.points.y < -epsilon)
        npoints = len(clink.points)
        borders = clink.points.y < epsilon
        loop0 = np.arange(npoints)[np.logical_and(borders, clink.points.x < 0)]
        loop1 = np.arange(npoints)[np.logical_and(borders, clink.points.x > 0)]
        clink.points.y += delta/2

        # Duplicate and inverse
        half = Mesh.from_mesh(clink)
        half.points.position[:, :2] *= -1

        # Join
        clink.join(half)

        # Bridge
        clink.bridge_loops(loop0, loop1 + npoints, close=True)
        clink.bridge_loops(loop1, loop0 + npoints, close=True)

        # UVMap
        nu, nv = major_segments + 3, minor_segments + 1
        uvmap = grid_uv_map(nu, nv).reshape(nu - 1, nv - 1, 4, 2)

        ratio = (length - radius)/(length - radius + np.pi*radius)
        dx_side = ratio/2

        uvmap[:-2,..., 0] *= (1 - ratio)/(1 - 2/nu)
        uvmap[-2,:, [0, 3], 0] = 1 - ratio
        uvmap[-2,:, [1, 2], 0] = 1 - ratio/2
        uvmap[-1,:, [0, 3], 0] = 1 - ratio/2

        clink.corners.UVMap = uvmap.reshape(-1, 2)

        return clink
    
    # ====================================================================================================
    # Extrusion
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Extrude vertices
    # ----------------------------------------------------------------------------------------------------

    def extrude_vertices(self, selection, offset, **attributes):
        """ Extrude individual vertices

        Arguments
        ---------
            - selection (selector) : points selection, all if None
            - offset (vector or array of vectors) : extrusion vector
            - attributes (dict) : points attributes

        Returns
        -------
            - dictionnary of the created geometry : vertex indices, face indices
        """
        inds = np.arange(len(self.points))
        if selection is not None:
            inds = inds[selection]

        pos = self.points.position[inds] + offset
        new_inds = np.arange(len(pos)) + len(self.points)
        edges = edges_between(inds, new_inds)

        return self.add_geometry(points=pos, edges=edges.reshape(-1, 2), **attributes)

    # ----------------------------------------------------------------------------------------------------
    # Extrude a loop of points
    # ----------------------------------------------------------------------------------------------------

    def extrude_loop(self, loop, offset, close=False, clockwise=False, **attributes):
        """ Extrude a loop of vertices.

        Arguments
        ---------
            - loop (array of ints) : vertex indices
            - offset (float = 1) : multiplicator of the direction vector
            - direction (vector = None) : extrusion direction, normal if None
            - clockwise (bool=False) : faces orientation
            - attributes : attribute for the new geometry

        Returns
        -------
            - dictionnary of the created geometry : vertex indices, face indices
        """
        loop = np.atleast_1d(loop)
        if len(loop) < 2:
            return None
        
        verts = self.points.position[loop]
        try:
            new_verts = verts + offset
        except Exception as e:
            raise AttributeError(
                f"Extrude> Offset argument must be a single vector or an array of {len(verts)} vectors, not {np.shape(offset)}."
                )

        indices = np.append(loop, len(self.points) + np.arange(len(new_verts)))
        gc = grid_corners(len(verts), 2, row_first=True, close_x=close, clockwise=clockwise)
        corners = indices[gc.ravel()]
        uvmap = grid_uv_map(len(verts), 2, close_x=close, clockwise=clockwise).reshape(-1, 2)

        return self.add_geometry(points=new_verts, corners=corners, faces=4, UVMap=uvmap)
    
    # ----------------------------------------------------------------------------------------------------
    # Extrude individual faces
    # ----------------------------------------------------------------------------------------------------

    def extrude_faces(self, selection, offset=None, scale=1.):
        """ Extrude individual faces.

        If offset is None, faces are extruder along their normal

        Arguments
        ---------
            - selection : indices of the faces to extrude
            - offset (vector or array of vectors) : the extrusion vector
            - scale (float = 1) : scale factor for offsets
            - dissolve (bool = True) : remove starting faces

        Returns
        -------
            - dictionnary : 'top' : extruded faces, 'side' : extruded side faces
        """

        # Selected faces indices
        faces_sel = np.arange(len(self.faces))
        if selection is not None:
            faces_sel = faces_sel[selection]

        # bmesh edition
        with self.bmesh() as bm:
            #bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            # select faces
            start_faces = [bm.faces[i] for i in faces_sel]

            res = {'top': [], 'side': []}

            # bmesh.ops
            d = bmesh.ops.extrude_discrete_faces(bm, faces=start_faces)
            new_faces = d["faces"]

            

            # One direction per face
            if offset is not None:
                shape = np.broadcast_shapes(np.shape(offset), (len(new_faces), 3))
                offsets = np.broadcast_to(offset, shape)*scale

            # Loop on created faces
            for i_face, face in enumerate(new_faces):
                verts = face.verts
                if offset is None:
                    ofs = face.normal*scale
                else:
                    ofs = offsets[i_face]

                bmesh.ops.translate(bm, verts=list(verts), vec=ofs)

                res['top'].append(face.index)
                for e in face.edges:
                    assert(len(e.link_faces) == 2)
                    if e.link_faces[0].index == face.index:
                        res['side'].append(e.link_faces[1].index)
                    else:
                        res['side'].append(e.link_faces[0].index)

        return res
    
    # ----------------------------------------------------------------------------------------------------
    # Extrude region
    # ----------------------------------------------------------------------------------------------------

    def extrude_region(self, selection, offset=(0, 0, 1), dissolve=False):
        """ Extrude individual faces.

        Arguments
        ---------
            - selection : indices of the faces to extrude
            - offset (vector = (0, 0, 1)) : the extrusion vector
            - dissolve (bool = True) : remove starting faces

        Returns
        -------
            - dictionnary : 'top' : extruded faces, 'side' : extruded side faces
        """

        # Selected faces indices
        faces_sel = np.arange(len(self.faces))
        if selection is not None:
            faces_sel = faces_sel[selection]

        # bmesh edition
        with self.bmesh() as bm:
            #bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            # select faces
            start_faces = [bm.faces[i] for i in faces_sel]

            res = {'top': [], 'side': []}


            # Extrusion
            d = bmesh.ops.extrude_face_region(
                bm, 
                geom                      = start_faces,
                #edges_exclude            = set(),
                #use_keep_orig            = False,
                #use_normal_flip          = False,
                #use_normal_from_adjacent = False,
                #use_dissolve_ortho_edges = False,
                #use_select_history       = False,
                )
                
            # Move new vectors
            extruded_geom = d["geom"]
            new_verts = [ele for ele in extruded_geom if isinstance(ele, bmesh.types.BMVert)]
            new_faces = [ele for ele in extruded_geom if isinstance(ele, bmesh.types.BMFace)]

            bmesh.ops.translate(bm, verts=new_verts, vec=list(offset))

            # ===== Result
            for face in new_faces:
                res['top'].append(face.index)
                for e in face.edges:
                    assert(len(e.link_faces) == 2)
                    if e.link_faces[0].index == face.index:
                        res['side'].append(e.link_faces[1].index)
                    else:
                        res['side'].append(e.link_faces[0].index)

            # ===== Dissolve extruded faces
            if dissolve:
                bm.faces.ensure_lookup_table()
                del_faces = [bm.faces[i] for i in faces_sel]

                bmesh.ops.delete(bm, geom=del_faces, context='FACES')                   

        return res

    # ----------------------------------------------------------------------------------------------------
    # inset faces
    # ----------------------------------------------------------------------------------------------------

    def inset_faces(self, selection, thickness=0.1, depth=0.0, use_even_offset=True, use_relative_offset=False):
        """ Extrude individual faces.

        If offset is None, faces are extruder along their normal

        Arguments
        ---------
        selection : selection on faces
            indices of the faces to inset
        thickness : float
            Inset thickness.
        depth : float
            Inset depth (extrusion).
        use_even_offset : bool
            Keep consistent thickness.
        use_relative_offset : bool
            Offset relative to face size.

        Returns
        -------
        dict with:
            'faces' : list of new inset faces
        """

        # Selected faces indices
        faces_sel = np.arange(len(self.faces))
        if selection is not None:
            faces_sel = faces_sel[selection]

        # bmesh edition
        with self.bmesh() as bm:
            bm.faces.ensure_lookup_table()

            # select faces
            start_faces = [bm.faces[i] for i in faces_sel]

            res = {'top': [], 'side': []}

            # bmesh.ops
            d = bmesh.ops.inset_individual(
                bm,
                faces               = start_faces,
                thickness           = thickness,
                depth               = depth,
                use_even_offset     = use_even_offset,
                use_relative_offset = use_relative_offset,
            )

            res = {
                'top': faces_sel,
                'side': [face.index for face in d["faces"]]
            }

        return res
    
    # ----------------------------------------------------------------------------------------------------
    # Solidify socle
    # ----------------------------------------------------------------------------------------------------

    def solidify_socle(self, shape, z=0, bottom_material_index=0):
        """ The mesh is supposed to be a grid.

        The socle is buit by extruding the external edges to the value z.

        Arguments
        ---------
        - shape (tuple of ints) : the grid shade
        - z (float) : socle base z
        - bottom_material_index (int = 0) : base face material index
        - attributes (dict) : attributes to faces

        Returns
        -------
        - bottom face index (int) : the index of bottom face
        """

        n = shape[0]*shape[1]
        inds = np.arange(n).reshape(shape)

        loop1 = np.append(inds[0, :-1], inds[:-1, -1])
        loop1 = np.append(loop1, np.flip(inds[-1, 1:]))
        loop1 = np.append(loop1, np.flip(inds[1:, 0]))

        points = np.array(self.points.position[loop1])
        points[:, 2] = z

        loop0 = self.add_points(points)
        res = self.add_faces(corners=loop0, faces=len(loop0))

        self.bridge_loops(loop0, loop1, close=True)

        self.faces._ensure_optional_field("material_index")
        self.faces[res['faces']].material_index = bottom_material_index

    # ----------------------------------------------------------------------------------------------------
    # Boolean
    # ----------------------------------------------------------------------------------------------------

    def boolean(self, other, operation='DIFFERENCE'):
        """
        Apply a boolean operation with another object.

        Parameters
        ----------
        other : Mesh
            The object to use as boolean operand.
        operation : str
            Boolean operation: 'INTERSECT', 'UNION', or 'DIFFERENCE'.
        """

        with other.object(0, readonly=True) as other_obj:

            with self.object(1) as obj:
                mod = obj.modifiers.new(name="Boolean", type='BOOLEAN')
                mod.object = other_obj
                mod.operation = operation

                # Apply modifier
                bpy.ops.object.modifier_apply(modifier=mod.name)

                mesh = Mesh.from_object(obj)

        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # Solidify
    # ----------------------------------------------------------------------------------------------------

    def solidify(self, thickness=.01, offset=-1):
        """ Boolean difference with another MeshBuilder.

        The methods uses the Solidify Modifier

        ``` python
        glass = Mesh.Circle(segments=128)
        glass.extrude_faces(0, -.01)
        glass.extrude_faces(0, -2)
        glass.extrude_faces(0, -.01)

        glass.points.translate((0, 0, 2))

        glass = glass.solidify(thickness=.1)

        glass.to_object("Solidify", shade_smooth=True)
        ```

        Arguments
        ---------
            - thickness (float=.01) : thickness
            - offset (float=-1) : offset

        Returns
        -------
            - MeshBuilder : the result of the solidify operation
        """

        with self.object() as obj:
            mod = obj.modifiers.new("Solidify", 'SOLIDIFY')

            mod.thickness       = thickness
            mod.use_even_offset = True
            mod.offset          = offset

            # Apply modifier
            bpy.ops.object.modifier_apply(modifier=mod.name)

            mesh = Mesh.from_object(obj)

        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # Remove doubles
    # ----------------------------------------------------------------------------------------------------

    def remove_doubles(self, dist=.001):
        """ Remove doubles.

        Arguments:
            - dist (float=0.001) : maximum distance between vertices to merge.
        """
        with self.bmesh() as bm:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)

        return self

    # ----------------------------------------------------------------------------------------------------
    # Triangulate
    # ----------------------------------------------------------------------------------------------------

    def triangulate(self, selection=None):

        faces_sel = np.arange(len(self.faces))
        if selection is not None:
            faces_sel = faces_sel[selection]

        if len(faces_sel) == 0:
            return
        
        copy = Mesh.from_mesh(self)

        with copy.bmesh(readonly = False) as bm:
            bm.faces.ensure_lookup_table()
            faces = [bm.faces[i] for i in faces_sel]
            
            bmesh.ops.triangulate(bm, faces=faces)

        return copy
    
    # ----------------------------------------------------------------------------------------------------
    # Simplify
    # ----------------------------------------------------------------------------------------------------

    def simplified(self, scale, dist=.001):

        copy = Mesh.from_mesh(self)
        copy.remove_doubles(dist=dist/scale)

        if len(copy.points) < 8:
            copy = self.get_cubic_envelop()

        return copy
    
    # ----------------------------------------------------------------------------------------------------
    # Faces to islands
    # ----------------------------------------------------------------------------------------------------

    def faces_to_islands(self, groups=None):
        """ Split faces into isolated islands

        Arguments
        ---------
        - groups (list of ints):
            group ids of faces
        """

        mesh = Mesh(materials=self.materials)
        attr_names = [name for name in self.faces.actual_names if name not in ['loop_total', 'loop_start']]

        # No group: each face becomes an island
        if groups is None:
            attrs = {name: self.faces[name] for name in attr_names}
            return Mesh().join_attributes(self).join_geometry(
                points = self.points.position[self.corners.vertex_index],
                corners = np.arange(len(self.corners)),
                faces = self.faces.loop_total,
                **attrs,
            )
        
        groups = np.asarray(groups)
        if groups.shape != (len(self.faces),)   :
            raise ValueError(f"The 'groups' argument must be a index per face with a length of {len(self.faces)}.")

        ugroups, rev_index = np.unique(groups, return_inverse=True)
        for group in ugroups:
            faces = self.faces[rev_index == group]
            attrs = {name: faces[name] for name in attr_names}

            corners = self.corners[faces.get_corner_indices()]
            uniques, new_corners = np.unique(corners.vertex_index, return_inverse=True)
            mesh.join(Mesh().join_attributes(self).join_geometry(
                points = self.points.position[uniques],
                corners = new_corners,
                faces = faces.loop_total,
                **attrs,
            ))
        
        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # Dual mesh
    # ----------------------------------------------------------------------------------------------------

    def dual(self, center="median"):

        verts = np.empty((len(self.faces), 3), np.float32)
        corners = []
        faces = []

        with self.bmesh() as bm:
            
            bm.faces.ensure_lookup_table()
            bm.verts.ensure_lookup_table()    
            bm.edges.ensure_lookup_table()    
        
            # ---------------------------------------------------------------------------
            # Faces become points
            # ---------------------------------------------------------------------------

            if center.lower() == 'median':
                for f in bm.faces:
                    c = f.calc_center_median()
                    verts[f.index] = (c.x, c.y, c.z)

            elif center.lower() == 'bounds':
                for f in bm.faces:
                    c = f.calc_center_bounds()
                    verts[f.index] = (c.x, c.y, c.z)

            elif center.lower() == 'weighted':
                for f in bm.faces:
                    c = f.calc_center_median_weighted()
                    verts[f.index] = (c.x, c.y, c.z)

            else:
                raise ValueError(f"Center must be in ('median', 'bounds','weighted').")
                
            # ---------------------------------------------------------------------------
            # Vertices becom faces
            # ---------------------------------------------------------------------------

            for v in bm.verts:
                # Faces need to be ordered (link_faces is not good)
                # Get the edges as couples (face.index, face.index)
                # then chain the edges

                couples = []
                first = True
                for edge in v.link_edges:

                    if len(edge.link_faces) != 2:
                        couples = []
                        break
                    
                    # First is used for the order between:
                    # - face0 then face1
                    # - face1 then face0
                    if first:
                        first = False
                        
                        # The edge links two vertices
                        other_index = edge.verts[0].index if edge.verts[1].index == v.index else edge.verts[1].index
                        
                        # We select as first face the one where loop is v -> other                
                        face0 = edge.link_faces[0]
                        for i_loop, lp in enumerate(face0.loops):
                            if lp.vert.index == other_index:
                                j = (i_loop + 1)%len(face0.loops)
                                take0 = face0.loops[j].vert.index == v.index
                                break
                            
                        if take0:
                            loop = [edge.link_faces[0].index, edge.link_faces[1].index]
                        else:
                            loop = [edge.link_faces[1].index, edge.link_faces[0].index]
                    
                    else:
                        couples.append((edge.link_faces[0].index, edge.link_faces[1].index))
                    
                if len(couples) < 2:
                    continue
                
                # Build the other faces by chaing the edges 
                # First loop to consume the edge couples      
                for _ in range(len(couples)):
                    found = False
                    # Second loop to find the following edge
                    for i, e in enumerate(couples):
                        if e[0] == loop[-1]:
                            loop.append(e[1])
                            found = True
                        elif e[1] == loop[-1]:
                            loop.append(e[0])
                            found = True
                        else:
                            continue
                        break
                    
                    if found:
                        del couples[i]
                    else:
                        loop = []
                        break
                    
                if len(loop) < 3:
                    continue

                faces.append(len(loop))
                corners.extend(loop)

        # We can build the dual mesh
        return Mesh(points=verts, corners=corners, faces=faces, materials=self.materials)

    # ----------------------------------------------------------------------------------------------------
    # Faces neighbors
    # ----------------------------------------------------------------------------------------------------

    def faces_neighbors(self):

        offset = 0
        neighbors = []

        with self.bmesh() as bm:           
            bm.faces.ensure_lookup_table()

            for face in bm.faces:
                ns = set()
                for edge in face.edges:
                    ns = ns.union([edge.link_faces[0].index, edge.link_faces[1].index])
                ns.remove(face.index)
                neighbors.append(list(ns))

        return neighbors

    
    # ====================================================================================================
    # BVHTree
    # ====================================================================================================

    def bvh_tree(self, count=None):
        if count is None:
            return BVHTree.FromPolygons(self.points.position, self.faces.sequences(), all_triangles=False, epsilon=0.0)

        else:
            pos    = self.points.position
            pos    = pos.reshape(count, -1, 3)

            nfaces = len(self.faces)//count
            inds   = list(self.corners.vertex_index)
            faces  = [inds[lstart:lstart+ltotal] for (lstart, ltotal) in zip(self.faces.loop_start[:nfaces], self.faces.loop_total[:nfaces])]

            return [BVHTree.FromPolygons(pos[i], faces, all_triangles=False, epsilon=0.0) for i in range(count)]




    # ====================================================================================================
    # Tests
    # ====================================================================================================
    
    # ----------------------------------------------------------------------------------------------------
    # Test the primitives
    # ----------------------------------------------------------------------------------------------------

    @staticmethod
    def _all_primitives():

        rng = np.random.default_rng(0)

        def toobj(mesh, name, x):
            mesh.materials = ["Material"]   
            mesh.points.x += x - np.min(mesh.points.x)
            x = np.max(mesh.points.x) + 1

            mesh.to_object(name, shade_smooth=False)

            return x
        
        x = 0

        # ----- Points
        mesh = Mesh.points(rng.uniform(-1, 1, size=(1000, 3)))
        x = toobj(mesh, "points", x)

        # ----- Lines
        mesh = Mesh.line([0, 0, 0], [[0, -2, 1], [0, -1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1]])
        mesh.join(Mesh.line([1, 0, 0], [[1, -2, 1], [1, -1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]], segments=3))
        mesh.join(Mesh.line([-1, 0, 0], [-1, 0, 1], segments=10))
        x = toobj(mesh, "line", x)

        # ----- Grid
        mesh = Mesh.grid().translate((0, 0, 1)).join(Mesh.grid(5, 3, 50, 30))
        x = toobj(mesh, "grid", x)

        # ----- Cicle
        mesh = Mesh.circle().join(Mesh.circle(cap='NGON').translate((0, 0, 1)), Mesh.circle(cap='FANS').translate((0, 0, 2)))      
        x = toobj(mesh, "circle", x)

        # ----- Cone
        mesh = Mesh.cone(fill_type='NONE').join(
            Mesh.cone(side_segments=10).translate((0, 0, 3)),
            Mesh.cone(fill_type='NGON').translate((0, 0, 6)),
            Mesh.cone(fill_type='FANS').translate((0, 0, 9)),
        )
        x = toobj(mesh, "cone", x)

        # ----- Cylinder
        mesh = Mesh.cylinder(fill_type='NONE').join(
            Mesh.cylinder(side_segments=10).translate((0, 0, 3)),
            Mesh.cylinder(fill_type='NGON').translate((0, 0, 6)),
            Mesh.cylinder(fill_type='FANS').translate((0, 0, 9)),
        )
        x = toobj(mesh, "cylinder", x)

        # ----- Arrow
        mesh = Mesh().join(
            Mesh.arrow((-1, 0, 0)),
            Mesh.arrow((+1, 0, 0)),
            Mesh.arrow((0, -1, 0)),
            Mesh.arrow((0, +1, 0)),
            Mesh.arrow((0, 0, -1)),
            Mesh.arrow((0, 0, +1)),
        )
        x = toobj(mesh, "arrow", x)

        # ----- Vectors field
        n = 50
        pos = rng.uniform(-1, 1, size=(n, 3))
        ori = rng.uniform(-1, 1, size=(n, 3))
        mesh = Mesh.vectors_field(
            pos,
            ori,
            radius = .02,
        )
        x = toobj(mesh, "vectors_field", x)

        # ----- Simple
        for name in ["cube", "pyramid", "uvsphere", "icosphere", "torus", "chain_link", "monkey"]:
            f = getattr(Mesh, name)
            mesh = f()
            x = toobj(mesh, name, x)

    # ----------------------------------------------------------------------------------------------------
    # Test edition
    # ----------------------------------------------------------------------------------------------------

    def _test_edition():
        rng = np.random.default_rng(0)

        # extrude vertices
        mesh = Mesh.icosphere()
        sel = rng.uniform(0, 1, len(mesh.points)) < .5
        mesh.extrude_vertices(sel, mesh.points.position[sel])
        mesh.to_object("extrude_vertices", shade_smooth=False)

        # extrude loop of vertices
        mesh = Mesh.circle(materials="Material")
        mesh.extrude_loop(np.arange(len(mesh.points)), offset=(0, 0, 1), close=True)
        mesh.points.x += 2
        mesh.to_object("extrude_loop", shade_smooth=False)

        # extrude individualfaces
        mesh = Mesh.icosphere()
        sel = rng.uniform(0, 1, len(mesh.faces)) < .5
        inds = np.arange(len(mesh.faces))[sel]
        res = mesh.extrude_faces(sel, scale=.3)
        res = mesh.inset_faces(res['top'])
        res = mesh.extrude_faces(res['top'], scale=-.1)
        mesh.points.x += 4
        mesh.to_object("extrude_faces", shade_smooth=False)

        # extrude region
        mesh = Mesh.grid(1, 1, 11, 11)
        sel = [23, 24, 25, 26, 27, 34, 35, 36, 45]
        res = mesh.extrude_region(sel, (0, 0, 1), dissolve=False)
        mesh.extrude_region(res['top'], (0, 0, .5), dissolve=True)
        mesh.points.x += 7
        mesh.to_object("extrude_region", shade_smooth=False)

        # solidify socle
        mesh = Mesh.grid(2, 1.6/3, 80, 30)
        mesh.points.z = .5 + rng.uniform(-.05, .05, len(mesh.points))
        mesh.points.x += 10
        mesh.solidify_socle(shape=(80, 30), z=0)
        mesh.to_object("solidify_socle", shade_smooth=False)

        # boolean
        cube = Mesh.cube()
        cyl = Mesh.cylinder(radius=.5, depth = 3)
        mesh = cube.boolean(cyl)
        mesh.points.x += 13
        mesh.to_object("boolean difference", shade_smooth=False)

        # solidify
        mesh = Mesh.grid().solidify(thickness=.2)
        mesh.points.x += 16
        mesh.to_object("solidify", shade_smooth=False)

        # triangulate
        mesh = Mesh.cube().triangulate()
        mesh.points.x += 19
        mesh.to_object("triangulate", shade_smooth=False)

















    

    # =============================================================================================================================
    # To curve
    # =============================================================================================================================

    def to_curve_REVIEW(self):
        """ > Convert mesh to curve

        Simple conversion when edges domain is defined
        """

        from npblender import Curve

        if self._edges is None:
            return None

        splines = []
        for edge in self.edges:
            v0, v1 = edge.vertex0, edge.vertex1

            ok = False
            for spline in splines:
                if spline[0] == spline[-1]:
                    continue

                if v0 == spline[0]:
                    spline.insert(0, v1)
                    ok = True
                elif v0 == spline[-1]:
                    spline.append(v1)
                    ok = True

                elif v1 == spline[0]:
                    spline.insert(0, v0)
                    ok = True
                elif v1 == spline[-1]:
                    spline.append(v0)
                    ok = True

            if not ok:
                splines.append([v0, v1])

        curve = Curve()
        for spline in splines:
            cyclic = spline[0] == spline[-1]
            if cyclic:
                spline = spline[:-1]

            curve.add(self.points.position[spline], curve_type='POLY', cyclic=cyclic)

        return curve
    








class OLD:









    # ====================================================================================================
    # Build a pipe around a line

    @classmethod
    def CurveToMesh(cls, points, profile=8, radius=1., torus=False, caps=True, mat=0, **attributes):
        """ Add a cylinder topology by extruding a profile along a line.

        The 'points' argument provides the line to extrude along.
        The profile argument either gives the resolution of a round profile or gives the
        vertices of a profile. The profile is given in the plane (x, y).

        ``` python
        count = 100
        z = np.linspace(0, 2, count)
        ag = np.linspace(0, 4*np.pi, count)
        verts = np.stack((np.cos(ag), np.sin(ag), z), axis=-1)

        helix = MeshBuilder().add_pipe(verts, profile=16, radius=.1, caps=True)

        helix.to_object("Helix")

        helix = MeshBuilder().add_pipe(verts, profile=16, radius=np.linspace(.1, .5, count), caps=True)

        obj = helix.to_object("Helix var raidus")
        obj.location.y = 3
        ```

        Arguments
        ---------
            - points (array[n, 3] of floats) : the line
            - profile (array[m, 3] of floats or int = 3) : the resolution of a round profile if int else the profile
            - radius (float or array of floats=1.) : the scale to apply to each instance of profile
            - torus (bool=False) : the cylinder as the topology of a torus
            - caps (bool=True) : add faces to the ends (meaningless if torus)
            - mat (int=0) : material index
            - attributes (name=value) : value for named attributes
        """

        # ----- Round profile

        if isinstance(profile, (int, np.int32, np.int64)):
            n = profile
            ags = np.linspace(0, np.pi*2, n, endpoint=False)
            profile = np.zeros((n, 3), float)
            profile[:, 0] = np.cos(ags)
            profile[:, 1] = np.sin(ags)

        # ----- Base shape : profiles multiplied by the number of vertices
        # Add_cylinder will create the faces

        cyl = MeshBuilder()
        cyl.add_cylinder(np.resize(profile, (len(verts), len(profile), 3)), torus=torus, caps=caps, mat=mat, **attributes)

        # ----- Profiles orientation

        dirs = np.empty((len(verts), 3), float)
        dirs[1:-1] = verts[2:] - verts[:-2]
        if torus:
            dirs[0]    = verts[1] - verts[-1]
            dirs[-1]   = verts[0] - verts[-2]
        else:
            dirs[0]    = verts[1] - verts[0]
            dirs[-1]   = verts[-1] - verts[-2]

        cyl.toward(dirs, track_axis='Z', up_axis='Y')

        # ----- Profiles scale and locations
        cyl.scale(np.reshape(radius, np.shape(radius) + (1,)))
        cyl.translate(verts)

        self.append(cyl)

        return self

    # =============================================================================================================================
    # Operations

    @staticmethod
    def from_geom(geom):
        inds = {'vert_inds': [], 'face_inds': [], 'edge_inds': [], 'verts': [], 'faces': [], 'edges': []}
        for item in geom['geom']:
            if isinstance(item, bmesh.types.BMVert):
                inds['vert_inds'].append(item.index)
                inds['verts'].append(item)
            elif isinstance(item, bmesh.types.BMFace):
                inds['face_inds'].append(item.index)
                inds['faces'].append(item)
            elif isinstance(item, bmesh.types.BMEdge):
                inds['edge_inds'].append(item.index)
                inds['edges'].append(item)
            else:
                print(type(item))
                assert(False)
        return inds



    # ----------------------------------------------------------------------------------------------------
    # Distribute points

    def distribute_points(self, density=10, seed=0):

        import geonodes as gn
        from npblender import Cloud

        with gn.GeoNodes("npblender Distribute Points"):

            pts = gn.Mesh(Geometry()).distribute_points_on_faces(density=density, seed=seed)
            normal, rotation = pts.normal_, pts.rotation_
            pts.points.store("Normal", normal)
            pts.points.store("Rotation", rotation)

            pts.to_vertices().out()

        obj = self.to_object("npblender Temp", collection=blender.get_temp_collection())
        nodes.add_gn_modifier(obj, "npblender Distribute Points")

        cloud = Cloud.FromObject(obj, evaluated=True)

        blender.delete_object(f"npblender Temp")

        return cloud

    def distribute_poisson(self, distance_min=0., density_max=10., density_factor=1., seed=0):

        import geonodes as gn
        from npblender import Cloud

        with gn.GeoNodes("npblender Distribute Points"):

            pts = gn.Mesh(Geometry()).distribute_points_on_faces(
                distance_min=distance_min, density_max=density_max, density_factor=density_factor,
                seed=seed)
            normal, rotation = pts.normal_, pts.rotation_
            pts.points.store("Normal", normal)
            pts.points.store("Rotation", rotation)

            pts.to_vertices().out()

        obj = self.to_object("npblender Temp", collection=blender.get_temp_collection())
        nodes.add_gn_modifier(obj, "npblender Distribute Points")

        cloud = Cloud.FromObject(obj, evaluated=True)

        blender.delete_object(f"npblender Temp")

        return cloud

    # =============================================================================================================================

        # =============================================================================================================================
        # =============================================================================================================================
        # Shape Keys

        # =============================================================================================================================
        # Build shape keys

        def shapekeys(self, count):
            sks = ShapeKeys.FromGeometry(self, count=count)
            sks.key_name = key_name
            sks.clear    = clear
            return sks

    # =============================================================================================================================
    # Field

    def field_value(self, x, y, values=None, dx=0, dy=0, bivariate=True, h=None):
        """ > Field value

        A surface is considered as a field.

        > [!NOTE]
        > Field is computed using scipy module

        > [!NOTE]
        > Grid must be built with indexing='ij': shape=(nx, ny)

        Arguments
        ---------
        - xy (array of (x, y)) : coordinates where to evaluate the field
        - name (str="z") : attribute name to evaluate
        - dx (int = 0) : derivative along x (values greater than 0 require bivariate=True)
        - dy (int = 0) : derivative along y (values greater than 0 require bivariate=True)
        - bivariate (bool = True) : use **RectBivariateSpline** rather than **interpn** from scipy module
        - h (float=.1) : small value used to computed derivative if bivariate is False

        Returns
        -------
        - Evaluated value at coordinates xy
        """

        from scipy.interpolate import interpn, RectBivariateSpline

        if self._shape is None:
            raise Exception("Mesh error: impossible to compute a field because '_shape' is not defined")

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if values is None:
            values = np.reshape(self.points.position[:, 2], self._shape)

        else:
            value_size = len(self.points) // np.size(x)
            try:
                if value_size == 1:
                    values = np.reshape(values, self._shape)
                else:
                    values = np.reshape(values, self._shape + (value_size,))
            except:
                raise Exception(f"Mesh.field_value error: values is shaped {np.shape(values)} which is not compatible with grid shape {self._shape}")

        # ----------------------------------------------------------------------------------------------------
        # Grid

        vertices = np.reshape(self.points.position, self._shape + (3,))
        grid_x = vertices[:, 0, 0]
        grid_y = vertices[0, :, 1]

        if bivariate:
            bv = RectBivariateSpline(grid_x, grid_y, z=values)

            # ----- Derivative directly computed by bivariate spline

            if h is None:
                return bv.ev(x, y, dx=dx, dy=dy)

        # ----------------------------------------------------------------------------------------------------
        # Compute the value

        def f(delta_x=0, delta_y=0):
            if bivariate:
                return bv.ev(x + delta_x, y + delta_y)
            else:
                return interpn((grid_x, grid_y), values, np.stack((x + delta_x, y + delta_y), axis=-1), bounds_error=False, fill_value=0)

        # ----------------------------------------------------------------------------------------------------
        # No derivative

        if dx == 0 and dy == 0:
            return f()

        # ----------------------------------------------------------------------------------------------------
        # Manually compute derivatives

        if h is None:
            h = .1

        if dx == 0:
            if dy == 1:
                return (f(delta_y=h/2) - f(delta_y=-h/2))/h
            else:
                return (f(delta_y=-h/2) + f(delta_y=h/2) - 2*f())/(h**2)
        else:
            if dx == 1:
                return (f(delta_x=h/2) - f(delta_x=-h/2))/h
            else:
                return (f(delta_x=-h/2) + f(delta_x=h/2) - 2*f())/(h**2)
            

    # =============================================================================================================================
    # Topology

    def topology(self, vertices=True, edges=False, corners=False, faces=False, size=.1):

        from . import text

        topo = Mesh()
        v0, v1 = np.min(self.points.position, axis=0), np.max(self.points.position, axis=0)
        center = (v0 + v1)/2

        def add(i, v, normal=None):
            if normal is None:
                rel = v - center
                d = np.linalg.norm(rel)
                pos = v + rel*(size/d)
            else:
                pos = v + size*normal

            txt = text.Text(text=str(i), size=size, align_x='CENTER', align_y='CENTER').to_mesh()
            txt.points.position += pos
            topo.join(txt)

        # ----- Vertices

        if vertices:
            for i, v in enumerate(self.points.position):
                add(i, v)

        # ----- Edges

        if edges:
            for i in range(len(self.edges)):
                e0 = self.points.position[self.edges.vertex0[i]]
                e1 = self.points.position[self.edges.vertex1[i]]
                add(i, (e0 + e1)/2)

        # ----- Faces and corner

        if corners or faces:
            for i, (position, normal) in enumerate(zip(self.faces.position, self.faces.normal)):

                print("Face", i, "/", len(self.faces))
                print(self.corners.vertex_index)

                if corners:
                    c0 = self.faces.loop_start[i]
                    ct = self.faces.loop_total[i]
                    for c in range(c0, c0+ct):
                        print(c0, ct, "-->", c , "/", len(self.points))
                        p = self.points.position[self.corners.vertex_index[c]]
                        add(c, p, normal=normal)

                if faces:
                    add(i, position, normal=normal)


        return topo
    
