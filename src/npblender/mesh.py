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
Module Name: mesh
Author: Alain Bernard
Version: 0.1.0
Created: 2023-10-23
Last updated: 2025-08-31

Summary:
    Mesh class.

Usage example:
    ```python
    cube = Mesh.cube()
    ```
"""

from contextlib import contextmanager
import numpy as np

import bpy
import bmesh
from mathutils.bvhtree import BVHTree

from . import blender

from .constants import bfloat, bint, bbool
from .maths import Transformation, Quaternion, Rotation
from .maths.topology import grid_corners, grid_uv_map, fans_corners, disk_uv_map
from .maths.topology import border_edges, edges_between, row_edges, col_edges

from .geometry import Geometry
from .domain import Vertex, Corner, Face, Edge

DATA_TEMP_NAME = "npblender_TEMP"

# =============================================================================================================================
# Mesh Gemetry

class Mesh(Geometry):

    domain_names = ['points', 'corners', 'faces', 'edges']

    def __init__(self, points=None, corners=None, faces=None, edges=None, materials=None, attr_from=None, **attrs):
        """
        Initialize a Mesh Geometry object.

        Parameters
        ----------
        points : array_like, optional
            The vertices of the mesh (default is None).
        corners : array_like of int, optional
            Corners, i.e., indices on the array of points (default is None).
        faces : array_like of int, optional
            Sizes of the faces; the sum of this array must be equal to the length of the corners array (default is None).
        edges : array_like of tuple of int, optional
            List of edges defined by pairs of vertex indices (default is None).
        materials : str or list of str, optional
            List of materials used in the geometry. If a single string is provided, it is converted to a list containing that string (default is None).
        attr_from : Geometry, optional
            Domain attributes to copy from another Geometry object (default is None).
        **attrs : dict, optional
            Additional geometry attributes.

        Attributes
        ----------
        points : Vertex
            The vertices of the mesh.
        corners : Corner
            The corners of the mesh.
        faces : Face
            The faces of the mesh.
        edges : Edge
            The edges of the mesh.
        materials : list of str
            The list of materials used in the geometry.
        """

        # ----- Initialize an empty geometry

        self.points  = Vertex()
        self.corners = Corner()
        self.faces   = Face()
        self.edges   = Edge()

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
        """
        Check if mesh domains (corners, faces, edges) are consistent.

        This method verifies the consistency of the mesh domains by checking the validity
        of corners, faces, and edges relative to the number of points. In development mode,
        it raises an exception to prevent Blender from crashing if inconsistencies are found.

        Parameters
        ----------
        title : str, optional
            Title prefix for error messages (default is "Mesh Check").
        halt : bool, optional
            If True, raise an exception on failure; otherwise, print a warning (default is True).

        Returns
        -------
        bool
            True if all checks pass; otherwise, raises an exception or prints an error.

        Raises
        ------
        Exception
            If the check fails and halt is True.
        """
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
        """
        Serialize the Mesh object to a dictionary representation.

        Returns
        -------
        dict
            A dictionary containing the serialized data of the mesh, including:
            - 'geometry': The type of geometry (always 'Mesh').
            - 'materials': List of material names.
            - 'points': Serialized points data.
            - 'corners': Serialized corners data.
            - 'faces': Serialized faces data.
            - 'edges': Serialized edges data.
        """
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
        """
        Create a Mesh instance from a dictionary representation.

        Parameters
        ----------
        d : dict
            Dictionary containing mesh data with keys 'materials', 'points', 'corners', 'faces', and 'edges'.

        Returns
        -------
        Mesh
            A new Mesh instance initialized with the data from the dictionary.
        """
        mesh = cls()
        mesh.materials  = d['materials']
        mesh.points     = Vertex.from_dict(d['points'])
        mesh.corners    = Corner.from_dict(d['corners'])
        mesh.faces      = Face.from_dict(d['faces'])
        mesh.edges      = Edge.from_dict(d['edges'])
        return mesh
    
    # =============================================================================================================================
    # Clear the geometry
    # =============================================================================================================================

    def clear_geometry(self):
        """
        Clear the geometry by deleting all geometric content.

        This method clears the points, corners, faces, and edges collections,
        effectively removing all geometric data from the mesh.

        Note:
            The materials list associated with the mesh remains unchanged.
        """
        self.points.clear()
        self.corners.clear()
        self.faces.clear()
        self.edges.clear()

    # =============================================================================================================================
    # From another Mesh
    # =============================================================================================================================

    def clone(self):
        return Mesh.from_mesh(self)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Copy
    # -----------------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_mesh(cls, other, points=None, faces=None, edges=None):
        """
        Create a copy of a Mesh object, optionally excluding specified points, faces, or edges.

        Parameters
        ----------
        other : Mesh
            The source Mesh object to copy.
        points : array-like of int, optional
            Indices of points to exclude from the copy.
        faces : array-like of int, optional
            Indices of faces to exclude from the copy.
        edges : array-like of int, optional
            Indices of edges to exclude from the copy.

        Returns
        -------
        Mesh
            A new Mesh instance copied from the source, with specified elements excluded.
        """
        mesh = cls(materials=other.materials)
        mesh.points  = Vertex(other.points,  mode='COPY')
        mesh.corners = Corner(other.corners, mode='COPY')
        mesh.faces   = Face(other.faces,   mode='COPY')
        mesh.edges   = Edge(other.edges,    mode='COPY')

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
        """Initialize the geometry from a Blender Mesh data.

        This method creates and returns an instance of the mesh class
        initialized with vertices, edges, faces, corners, materials,
        and attributes extracted from the provided Blender mesh data.

        Args:
            data: Blender mesh data or object that can be processed
                  by the blender.get_mesh function to obtain a Blender Mesh instance.

        Returns:
            An instance of the mesh class initialized with the geometry
            and attributes from the Blender Mesh.

        Raises:
            ImportError: If the local blender module cannot be imported.
            Any exceptions raised by blender.get_mesh if the data is invalid.
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
        """
        Write the geometry data from this mesh into a Blender Mesh instance.

        This method transfers the mesh's vertices, edges, faces, corners, materials,
        and custom attributes into the provided Blender Mesh data structure.

        Parameters
        ----------
        data : Blender Mesh instance
            The Blender Mesh object to which the geometry will be written.

        Returns
        -------
        None

        Side Effects
        ------------
        Modifies the provided Blender Mesh instance by clearing its current geometry
        and populating it with the data from this mesh.
        Updates the mesh to reflect the changes.
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
        """
        Create a Mesh instance from an existing Blender object.

        This method initializes a mesh from a Blender object, optionally using the evaluated
        version of the object (i.e., after applying modifiers).

        Parameters
        ----------
        obj : str or bpy.types.Object
            The Blender object or its name from which to create the mesh.
        evaluated : bool, optional
            If True, use the evaluated object with modifiers applied. If False, use the raw mesh data.
            Default is False.

        Returns
        -------
        Mesh
            A new Mesh instance created from the specified Blender object.

        Raises
        ------
        ImportError
            If the local blender module cannot be imported.
        """
        if evaluated:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            object_eval = blender.get_object(obj).evaluated_get(depsgraph)
            return cls.from_mesh_data(object_eval.data)

        else:
            return cls.from_mesh_data(blender.get_object(obj).data)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # To blender object
    # -----------------------------------------------------------------------------------------------------------------------------

    def to_object(self, obj, shade_smooth=None, shapekeys=None, collection=None):
        """
        Create or update a Blender mesh object from this mesh data.

        This method creates a new Blender mesh object if it does not already exist,
        or updates the existing object's mesh data. It does not perform object type conversion;
        the existing object must be a mesh.

        After the object is created or updated, use 'update_object' to modify vertices.

        Parameters
        ----------
        obj : str or bpy.types.Object
            The Blender object or its name to create or update.
        shade_smooth : bool or None, optional
            If specified, sets the shading mode of the mesh polygons to smooth or flat.
        shapekeys : ShapeKeys or iterable of ShapeKeys, optional
            Shape keys to apply to the mesh object.
        collection : bpy.types.Collection or None, optional
            The collection to which the object should be linked.

        Returns
        -------
        bpy.types.Object
            The created or updated Blender mesh object.
        """
        from .shapekeys import ShapeKeys
        
        res = blender.create_mesh_object(obj, collection=collection)
        self.to_mesh_data(res.data)

        if shade_smooth is not None:
            res.data.polygons.foreach_set('use_smooth', [shade_smooth]*len(res.data.polygons))

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
        """
        Context manager to access and manipulate the mesh using Blender's BMesh API.

        This method creates a temporary BMesh from the mesh data, yields it for modification,
        and then writes back the changes to the mesh data unless in readonly mode.

        Example usage:
            ```python
            mesh = Mesh.Cube()

            # Move the vertices with bmesh
            with mesh.bmesh() as bm:
                for v in bm.verts:
                    v.co.x += 1.0

            # Move the vertices directly in numpy array
            mesh.points.position[:, 1] += 1

            # Cube moved along x and y
            mesh.to_object("Cube")
            ```

        Parameters
        ----------
        readonly : bool, optional
            If True, changes made to the BMesh are not written back to the mesh data (default is False).

        Yields
        ------
        bmesh.types.BMesh
            A BMesh object representing the mesh data, which can be modified within the context.
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
        """
        Context manager to access the Blender Mesh API with a temporary mesh.

        This method transfers the current mesh geometry to a temporary Blender Mesh data block,
        yields it for reading or modification, and optionally captures the changes back into the mesh.

        Example usage:
            ```python
            mesh = Mesh.Cube()

            with mesh.blender_data() as data:
                normals = np.array([poly.normal for poly in data.polygons])

            print(normals)
            # Output:
            # [[-1. -0.  0.]
            #  [ 0.  1.  0.]
            #  [ 1. -0.  0.]
            #  [ 0. -1.  0.]
            #  [ 0.  0. -1.]
            #  [ 0. -0.  1.]]
            ```

        Parameters
        ----------
        readonly : bool, optional
            If True, the geometry is not read back from the Blender Mesh after modification.
            Default is False.

        Yields
        ------
        bpy.types.Mesh
            A temporary Blender Mesh data block representing the mesh geometry.
        """

        data = bpy.data.meshes.get(DATA_TEMP_NAME)
        if data is None:
            data = bpy.data.meshes.new(DATA_TEMP_NAME)

        self.to_mesh_data(data)

        yield data

        # Read back changes unless readonly
        if not readonly:
            self.capture(Mesh.from_mesh_data(data))

    # ====================================================================================================
    # From something
    # ====================================================================================================

    @classmethod
    def from_model(cls, model, materials=None):
        """
        Create a Mesh instance from various types of input models.

        Parameters
        ----------
        model : str, bpy.types.Object, dict, Mesh, or bpy.types.Mesh
            The input model to create the Mesh from. It can be:
            - A string or Blender object to be evaluated and converted.
            - A dictionary representing the mesh data.
            - An existing Mesh instance.
            - A Blender Mesh data block.
        materials : list or None, optional
            Materials to associate with the mesh (currently unused in this method).

        Returns
        -------
        Mesh
            The created Mesh instance based on the input model.

        Raises
        ------
        Exception
            If the type of the model is not supported.
        """
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
        """
        Join other Mesh instances into this mesh.

        This method appends the geometry and materials of the given meshes to the current mesh,
        updating indices to maintain consistency.

        Parameters
        ----------
        *others : Mesh
            One or more Mesh instances to be joined with the current mesh.

        Returns
        -------
        self : Mesh
            The updated mesh instance with joined geometry.
        """
        for other in others:

            # Vertices
            v_ofs = len(self.points)
            self.points.extend(other.points)

            # Corners
            c_ofs = len(self.corners)
            self.corners.extend(other.corners)
            if len(self.corners):
                self.corners.vertex_index[c_ofs:] += v_ofs

            # Faces
            f_ofs = len(self.faces)
            self.faces.extend(other.faces)
            if len(self.faces):
                self.faces.loop_start[f_ofs:] += c_ofs

            # Edges
            e_ofs = len(self.edges)
            self.edges.extend(other.edges)
            if len(self.edges):
                self.edges.vertex0[e_ofs:] += v_ofs
                self.edges.vertex1[e_ofs:] += v_ofs

            # Materials
            remap = np.array([self.get_material_index(mat_name) for mat_name in other.materials])
            if len(remap) > 0:
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
    # Extracting from faces
    # =============================================================================================================================

    def extract_from_faces(self, selection=True):
        """
        Extract a Mesh from a selection of faces.

        Parameters
        ----------
        selection : array_like, optional
            A valide selection on face (default is True).

        Returns
        -------
        Mesh
            The extracted mesh.
        """
        # Faces and corner indices
        faces = self.faces[selection]
        cinds = faces.loop_index

        # Corners and vertices to extract
        corners = self.corners.vertex_index[cinds]
        iverts, new_corners = np.unique(corners, return_inverse=True)

        # Create an empty mesh
        mesh = Mesh()

        # Points
        mesh.points.extend(self.points[iverts])

        # Corners
        mesh.corners.extend(self.corners[cinds])
        mesh.corners.vertex_index = new_corners

        # Faces
        mesh.faces.extend(faces)
        mesh.faces.update_loop_start()

        return mesh

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
        """
        Add geometry components (vertices, corners, faces, edges) to the mesh.

        This method appends the specified geometry to the mesh without altering existing indices.
        It supports referencing existing vertices through corners or adding new vertices.

        > ***Note:***
        > To add independent geometry with new vertices, use [`Mesh.join_geometry`][npblender.Mesh.joint_geometry] instead.

        Examples
        --------
            ``` python
            cube = Mesh.cube()
            # Add a triangle on existing vertices
            # corners argument refers to cube vertices
            cube.add_geometry(corners=[0, 1, 2], faces=3)

            # Add a triangle with additional vertices
            # corners argument refers to the new vertices, passed values [0, 1, 2]
            # will be shifted to actual values [8, 9, 10]
            cube.join_geometry(points=[[0, 0, 0], [0, 1, 0], [1, 0, 0]], corners=[0, 1, 2], faces=3)
            ```

        Parameters
        ----------
        points : array-like of vectors, optional
            Vertices to add to the mesh.
        corners : array-like of int, optional
            Indices referring to vertices in the points array.
        faces : int, array-like of int, or list of lists, optional
            Defines the faces topology:
            - If `corners` is provided:
                - None: Single face made of all corners.
                - int: All faces have the same size (must divide the number of corners).
                - array-like: Face sizes; sum must equal the number of corners.
            - If `corners` is None:
                - Must be a list of lists, each sublist is a list of corners.
        edges : array-like of pairs of int, optional
            Edges defined by pairs of vertex indices.
        safe_mode : bool, optional
            If True, perform a mesh integrity check after adding geometry.
        **attrs : dict
            Additional geometry attributes to apply.

        Returns
        -------
        dict
            Dictionary with keys {'points', 'corners', 'faces', 'edges'} mapping to lists of added geometry indices.

        Raises
        ------
        ValueError
            If faces and corners lengths are inconsistent or invalid.
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

        # If corners is provided, faces can be:
        # - None -> single face made of all corners
        # - int -> faces are all the same size (len(corners) must be a multiplief of faces)
        # - array like -> face sizes (len(corners) == np.sum(faces))
        #
        # If faces is provided, corners can be:
        # - None -> faces must be an array of arrays, each array being of list of corners
        # - not None -> see above

        ok_faces = True
        if corners is None:
            if faces is None:
                ok_faces = False
            else:
                corners = []
                sizes = []
                ok = hasattr(faces, '__len__')
                if ok:
                    for face in faces:
                        ok = hasattr(faces, '__len__') and len(face) > 2
                        if not ok:
                            break
                        corners.extend(face)
                        sizes.append(len(face))
                if not ok:
                    raise ValueError(f"Mesh add_geometry> when corners is None, faces must be None or an array of arrays, each array being of list of corners.")
                faces = sizes

        else:
            corners = np.asarray(corners)
            ncorners = len(corners)
            if faces is None:
                faces = [ncorners]

            else:
                faces = np.asarray(faces)
                if faces.shape == ():
                    size = int(faces)
                    if ncorners % size != 0:
                        raise ValueError(f"Mesh add_geometry> when faces is a single number {size}, it must divide the number of corners ({ncorners}).")
                    faces = [size]*(ncorners // size)
                else:
                    if np.sum(faces) != ncorners:
                        raise ValueError(f"Mesh add_geometry> the sum of faces ({np.sum(faces)}) must be equal to the number of corners ({ncorners}).")
                    
        if ok_faces:
            added['corners'] = self.corners.append(vertex_index=corners, **disp_attrs['corners'])
            added['faces'] = self.faces.append_sizes(faces, **disp_attrs['faces'])

        if safe_mode:
            self.check()

        return added

    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Join geometry
    # -----------------------------------------------------------------------------------------------------------------------------

    def join_geometry(self, points=None, corners=None, faces=None, edges=None, safe_mode=False, **attrs):
        """
        Join geometry defined by components into the current mesh.

        This method creates a new independent mesh from the provided geometry components
        (points, corners, faces, edges) which do not refer to existing vertices. The new mesh
        is then joined to the current mesh instance.

        To add geometry using existing vertices, see [`Mesh.add_geometry`][npblender.Mesh.add_geometry].

        Parameters
        ----------
        points : iterable, optional
            Iterable of points (vertices) to add to the mesh.
        corners : iterable, optional
            Iterable of corner indices defining the mesh topology.
        faces : iterable, optional
            Iterable of faces defined by indices of corners.
        edges : iterable, optional
            Iterable of edges defined by indices of vertices.
        safe_mode : bool, optional
            Flag to enable safe mode operations (currently unused).
        **attrs : dict
            Additional attributes to be passed to the geometry addition.

        Returns
        -------
        self : Mesh
            The current mesh instance with the new geometry joined.
        """
        mesh = Mesh(attr_from=self)
        mesh.add_geometry(
            points=points, 
            corners=corners, 
            faces=faces, 
            edges=edges,
            **attrs)
        self.join(mesh)
        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Add Vertices
    # -----------------------------------------------------------------------------------------------------------------------------

    def add_points(self, points,  **attributes):
        """ Add vertices.

        Arguments
        ---------
            - points (array of vectors) : the vertices to add
            - attributes (name=value) : value for named attributes

        Returns
        -------
            - array of ints : indices of the added vertices
        """
        npoints = len(self.points)
        return self.points.append(position=points, **attributes)
    
    # =============================================================================================================================
    # Split edges
    # =============================================================================================================================

    def split_edges(self, loop0, loop1, cuts=1):
        """
        Subdivide in place the edges whose endpoints match the pairs
        (loop0[i], loop1[i]) regardless of order (edges are treated as undirected).

        The inputs `loop0` and `loop1` can be:
        - scalars (a single vertex index),
        - sequences of the same length,
        - or a mix of both (a scalar is broadcast to match the length of the other).

        The vertex pairs are normalized by sorting (min, max) so that order does not
        matter, and then compared against the BMesh edge list to determine which
        edges should be subdivided.

        Parameters
        ----------
        loop0 : int or array-like of int
            First vertex (or list of vertices) of the edges to be selected.
            If scalar, it will be broadcast to the length of `loop1` if needed.
        loop1 : int or array-like of int
            Second vertex (or list of vertices) of the edges to be selected.
            If scalar, it will be broadcast to the length of `loop0` if needed.
        cuts : int, optional
            Number of cuts per selected edge, as defined by
            `bmesh.ops.subdivide_edges`. Default is `1`. Must be >= 1.

        Returns
        -------
        None
            Modifies the geometry **in place**. Returns `None`.
            If no edge matches the given pairs, the function returns immediately
            without modifying the mesh.

        Notes
        -----
        - Edge selection is performed by constructing an array of sorted vertex pairs
        `(min(v0, v1), max(v0, v1))` and checking membership (via `np.isin` on a
        structured dtype view) against the BMesh edge list.
        - Subdivision is executed with `bmesh.ops.subdivide_edges` and
        `use_grid_fill=False`.

        Examples
        --------
        Subdivide a single edge (vertices 12 and 34) with 2 cuts:

        ```python
        obj = ...  # wrapper object providing .bmesh() and ._bm_edges(...)
        obj.split_edges(12, 34, cuts=2)
        ```

        Subdivide multiple edges defined by pairs of vertices:

        ```python
        v0 = [1, 5, 9]
        v1 = [2, 6, 10]
        obj.split_edges(v0, v1, cuts=1)
        ```

        Use a scalar broadcast against a vector:

        ```python
        # All edges (7, x) for x in [8, 9, 10]
        obj.split_edges(7, [8, 9, 10], cuts=1)
        ```

        See Also
        --------
        bmesh.ops.subdivide_edges : The underlying BMesh operator used for subdivision.

        > ***Warning:*** This operation modifies the mesh **in place** and may
        > create new vertices/edges/faces. Handle undo/history in Blender if needed.

        > ***Caution:*** `use_grid_fill=False` prevents automatic grid filling.
        > Depending on topology, additional n-gons or triangles may be introduced.

        > ***Note:*** Edges are considered undirected: (a, b) and (b, a) are
        > equivalent when matching edges.
        """
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
        """
        Create a grid connecting two vertex loops of equal size.

        The operation selects the edges forming each loop and bridges them
        using `bmesh.ops.bridge_loops`. If `segments > 1`, the newly created
        edges are subdivided to form a denser grid between the loops.

        Parameters
        ----------
        loop0 : array-like of int
            The first loop of vertex indices.
        loop1 : array-like of int
            The second loop of vertex indices. Must have the same length as `loop0`.
        close : bool, optional
            If True, the loops are treated as closed and the first vertex is
            appended at the end to close the cycle. Default is False.
        segments : int, optional
            Number of segments to subdivide between the loops. Must be >= 1.
            Default is 1 (no subdivision).
        **attributes : dict, optional
            Additional attributes to set on the mesh after bridging (passed as keyword arguments).

        Returns
        -------
        None
            Modifies the mesh **in place**. Returns `None`.

        Notes
        -----
        - Edges belonging to each loop are identified by sorting endpoint pairs
        and matching them against the current BMesh edge list via a structured
        dtype view and `np.isin`.
        - Bridging is performed with `bmesh.ops.bridge_loops`.
        - When `segments > 1`, subdivision of the bridge edges is performed with
        `bmesh.ops.subdivide_edges` using `cuts=segments - 1` and
        `use_grid_fill=False`.

        Examples
        --------
        Bridge two loops with no subdivision:

        ```python
        obj.bridge_loops(loop0, loop1, segments=1)
        ```

        Bridge two closed loops with 3 subdivisions:

        ```python
        obj.bridge_loops(loop0, loop1, close=True, segments=3)
        ```

        See Also
        --------
        bmesh.ops.bridge_loops : BMesh operator for bridging edge loops.
        bmesh.ops.subdivide_edges : BMesh operator for subdividing edges.

        > ***Warning:*** This function modifies the mesh **in place** and may create
        > new vertices/edges/faces. Handle undo/history in Blender if needed.

        > ***Caution:*** Both loops must have the same number of vertices for correct
        > bridging.

        > ***Note:*** When `close=True`, the first vertex of each loop is duplicated at
        > the end to ensure cyclic connectivity.
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
        """
        Fill a cap between vertices forming a loop.

        Supports two modes:
        - **NGON**: creates a single n-gon face from the loop. No center point is required.
        - **FANS**: creates a fan of triangles around a center point. The center can be:
        * `None`: automatically computed as the centroid of the loop.
        * `int`: the index of an existing vertex to use as center.
        * `array-like`: explicit coordinates of the center, which will be added as a new vertex.

        Parameters
        ----------
        loop : array-like of int
            The vertex indices defining the loop.
        mode : {'NGON', 'FANS'}, optional
            Fill mode to use. Default is 'NGON'.
        center : int or array-like or None, optional
            Center of the cap (used only in 'FANS' mode).
            - `None`: computed centroid.
            - `int`: index of an existing vertex.
            - array-like: coordinates of a new vertex.
        segments : int, optional
            Number of radial subdivisions for FANS mode. Must be >= 1.
            Default is 1 (no subdivision).
        clockwise : bool, optional
            Whether the loop is ordered clockwise. Default is False.
        **attributes : dict, optional
            Additional attributes to add to the mesh (passed to `add_geometry`).

        Returns
        -------
        dict
            A dictionary of the newly added geometry, as returned by
            [`add_geometry`][npblender.Mesh.add_geometry]. Includes at least keys for
            'faces' and 'corners'. In FANS mode, also includes the added 'points' if a new
            center is created.

        Notes
        -----
        - In 'NGON' mode, a UV map is generated using [`disk_uv_map`][npblender.Mesh.disk_uv_map].
        - In 'FANS' mode, the fan topology is created with [`fans_corners`][npblender.Mesh.fans_corners]
        and UVs are generated with [`disk_uv_map`][npblender.Mesh.disk_uv_map].
        - If `segments > 1` in FANS mode, radial edges are subdivided using
        [`split_edges`][npblender.Mesh.split_edges].

        Examples
        --------
        Fill a loop with an n-gon:

        ```python
        obj.fill_cap(loop, mode='NGON')
        ```

        Fill a loop with a triangle fan around an automatically computed center:

        ```python
        obj.fill_cap(loop, mode='FANS')
        ```

        Fill a loop with a fan using an existing vertex as the center and add 3 subdivisions:

        ```python
        obj.fill_cap(loop, mode='FANS', center=42, segments=3)
        ```

        See Also
        --------
        [`add_geometry`][npblender.Mesh.add_geometry] :
            Method used to add the created geometry to the mesh.
        [`split_edges`][npblender.Mesh.split_edges] :
            Used to subdivide radial edges in FANS mode.
        [`disk_uv_map`][npblender.Mesh.disk_uv_map] :
            Generates UV coordinates for circular caps.
        [`fans_corners`][npblender.Mesh.fans_corners] :
            Generates corner topology for FANS mode.

        > ***Warning:*** This function modifies the mesh **in place** and may
        > create new vertices, faces, and edges.

        > ***Caution:*** In FANS mode, if `center=None`, a new vertex is added at
        > the centroid of the loop.

        > ***Note:*** The `segments` parameter only applies to FANS mode; NGON
        > mode always produces a single polygon face.
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
        """
        Delete only the selected faces from the mesh.

        Parameters
        ----------
        selection : array-like of int or bool
            Indices or boolean mask specifying which faces to delete.

        Returns
        -------
        None
            Modifies the mesh **in place**. Returns `None`.

        See Also
        --------
        [`delete_loops`][npblender.Faces.delete_loops] :
            Method used internally to remove the corners and faces.
        [`corners`][npblender.Mesh.corners] :
            Corner array of the mesh, used to identify face connectivity.

        > ***Warning:*** This function permanently deletes faces and their
        > associated corners. Handle undo/history in Blender if needed.

        > ***Note:*** Only faces are removed. Edges and vertices remain in
        > the mesh unless explicitly deleted by other operations.
        """
        self.faces.delete_loops(selection, self.corners)

    # ----------------------------------------------------------------------------------------------------
    # Delete vertices
    # ----------------------------------------------------------------------------------------------------

    def delete_vertices(self, points=None, faces=None, edges=None):
        """
        Delete vertices from the mesh, with optional selection by points, faces, or edges.

        A vertex is deleted if it is explicitly listed in `points`, or if it belongs
        to any of the given `faces` or `edges`.

        Parameters
        ----------
        points : array-like of int or bool, optional
            Vertex indices (or boolean mask) specifying which vertices to delete directly.
        faces : array-like of int or bool, optional
            Face indices (or boolean mask). Any vertex belonging to these faces will be deleted.
        edges : array-like of int or bool, optional
            Edge indices (or boolean mask). Any vertex belonging to these edges will be deleted.

        Returns
        -------
        None
            Modifies the mesh **in place**. Returns `None`.

        Notes
        -----
        - At least one of `points`, `faces`, or `edges` must be provided, otherwise the function does nothing.
        - The deletion is executed using `bmesh.ops.delete` with `context='VERTS'`.

        Examples
        --------
        Delete specific vertices:

        ```python
        obj.delete_vertices(points=[0, 1, 2])
        ```

        Delete all vertices belonging to certain faces:

        ```python
        obj.delete_vertices(faces=[10, 11])
        ```

        Delete all vertices belonging to certain edges:

        ```python
        obj.delete_vertices(edges=[5, 6, 7])
        ```

        See Also
        --------
        [`bmesh.ops.delete`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.delete) :
            Blender BMesh operator used for deleting geometry.

        > ***Warning:*** This function permanently removes vertices and any connected
        > geometry (edges, faces). Handle undo/history in Blender if needed.

        > ***Note:*** If multiple selectors (`points`, `faces`, `edges`) are provided,
        > the union of all matched vertices will be deleted.
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
        """
        Create a grid mesh.

        Blender constructor for generating a grid primitive using
        `bmesh.ops.create_grid`.

        Parameters
        ----------
        x_segments : int, optional
            Number of segments along the X axis. Default is 1.
        y_segments : int, optional
            Number of segments along the Y axis. Default is 1.
        size : float or tuple of float, optional
            Size of the grid. If a single float is given, the grid is square.
            If a tuple is given, defines the grid dimensions along X and Y.
            Default is 2.
        materials : list of str, optional
            List of material names to assign to the grid. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the grid.

        Notes
        -----
        - The grid is created using `bmesh.ops.create_grid` with `calc_uvs=True`
        so UV coordinates are automatically generated.

        Examples
        --------
        Create a 10x10 grid of size 5:

        ```python
        grid = Mesh.bl_grid(x_segments=10, y_segments=10, size=5)
        ```

        Create a rectangular grid 4x8 of size (2, 5):

        ```python
        grid = Mesh.bl_grid(x_segments=4, y_segments=8, size=(2, 5))
        ```

        See Also
        --------
        [`bmesh.ops.create_grid`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_grid) :
            BMesh operator used for creating a grid primitive.

        > ***Note:*** UVs are automatically calculated when the grid is created.
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
        """
        Create a circle mesh.

        Blender constructor for generating a circle primitive using
        `bmesh.ops.create_circle`.

        Parameters
        ----------
        radius : float, optional
            Radius of the circle. Default is 1.
        segments : int, optional
            Number of segments (vertices) forming the circle. Default is 16.
        fill_tris : bool, optional
            If True, fills the circle with a triangle fan. Default is False.
        materials : list of str, optional
            List of material names to assign to the circle. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the circle.

        Notes
        -----
        - The circle is created using `bmesh.ops.create_circle` with `calc_uvs=True`
        so UV coordinates are automatically generated.
        - By default (`fill_tris=False`), the circle is an open ring.
        With `fill_tris=True`, the circle is filled with triangles (fan topology).

        Examples
        --------
        Create an empty circle of radius 2 with 32 segments:

        ```python
        circle = Mesh.bl_circle(radius=2, segments=32)
        ```

        Create a filled circle (disk) of radius 1 with 24 segments:

        ```python
        circle = Mesh.bl_circle(radius=1, segments=24, fill_tris=True)
        ```

        See Also
        --------
        [`bmesh.ops.create_circle`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_circle) :
            BMesh operator used for creating a circle primitive.

        > ***Note:*** UVs are automatically calculated when the circle is created.
        """

        mesh = cls(materials=materials)
        with mesh.bmesh() as bm:
            bmesh.ops.create_circle(bm, cap_ends=True, cap_tris=fill_tris, segments=segments, radius=radius, calc_uvs=True)

        return mesh
    
    # ----------------------------------------------------------------------------------------------------
    # Cone
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def bl_cone(cls, radius1=1, radius2=0, depth=2, segments=16, side_segments=1,
                cap_ends=True, cap_tris=False, materials=None):
        """
        Create a cone mesh.

        Blender constructor for generating a cone (or cylinder) primitive using
        `bmesh.ops.create_cone`.

        Parameters
        ----------
        radius1 : float, optional
            Base radius of the cone. Default is 1.
        radius2 : float, optional
            Top radius of the cone. If set to 0, produces a true cone; if equal to
            `radius1`, produces a cylinder. Default is 0.
        depth : float, optional
            Height of the cone along the Z axis. Default is 2.
        segments : int, optional
            Number of segments around the circumference. Default is 16.
        side_segments : int, optional
            Number of subdivisions along the vertical side edges.
            Default is 1 (no subdivision).
        cap_ends : bool, optional
            If True, fill the top and bottom caps. Default is True.
        cap_tris : bool, optional
            If True, fill the caps using triangle fans instead of n-gons.
            Default is False.
        materials : list of str, optional
            List of material names to assign to the cone. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the cone.

        Notes
        -----
        - The cone is created using `bmesh.ops.create_cone` with `calc_uvs=True`
        so UV coordinates are automatically generated.
        - When `side_segments > 1`, vertical edges crossing the top and bottom are
        subdivided using `bmesh.ops.subdivide_edges`.

        Examples
        --------
        Create a simple cone with radius 1 and height 2:

        ```python
        cone = Mesh.bl_cone(radius1=1, radius2=0, depth=2, segments=16)
        ```

        Create a cylinder with 32 segments and subdivided sides:

        ```python
        cylinder = Mesh.bl_cone(radius1=1, radius2=1, depth=3,
                                segments=32, side_segments=4)
        ```

        Create a cone with filled caps using triangle fans:

        ```python
        cone = Mesh.bl_cone(radius1=1, radius2=0, depth=2,
                            cap_ends=True, cap_tris=True)
        ```

        See Also
        --------
        [`bmesh.ops.create_cone`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_cone) :
            BMesh operator used for creating cone and cylinder primitives.
        [`bmesh.ops.subdivide_edges`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.subdivide_edges) :
            BMesh operator used for subdividing vertical edges when `side_segments > 1`.

        > ***Note:*** UVs are automatically calculated when the cone is created.
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
    def points_cloud(cls, points=None, materials=None):
        """
        Create a mesh containing only points at the given positions.

        Parameters
        ----------
        points : array-like of shape (N, 3), optional
            Coordinates of the points. If None, an empty mesh is created.
        materials : list of str, optional
            List of material names to assign to the mesh. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the given points.

        Notes
        -----
        - This method does not create any edges or faces, only isolated points.

        Examples
        --------
        Create a point cloud with three points:

        ```python
        pts = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])
        cloud = Mesh.points_cloud(points=pts)
        ```

        See Also
        --------
        [`Mesh`][npblender.Mesh] :
            The mesh class used to construct and manage geometry.

        > ***Note:*** This constructor is useful for importing raw point data or
        > initializing a mesh before adding edges and faces.
        """
        return cls(points=points, materials=materials)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Line
    # -----------------------------------------------------------------------------------------------------------------------------

    @classmethod
    def line(cls, start=(0, 0, 0), end=(0, 0, 1), segments=1, materials=None):
        """
        Create a mesh representing a straight line (or multiple lines) subdivided into segments.

        Parameters
        ----------
        start : array-like of float, shape (..., 3), optional
            Coordinates of the start point(s). Can be a single 3D vector or an array of
            multiple vectors. Default is (0, 0, 0).
        end : array-like of float, shape (..., 3), optional
            Coordinates of the end point(s). Can be a single 3D vector or an array of
            multiple vectors with the same shape as `start`. Default is (0, 0, 1).
        segments : int, optional
            Number of line segments (subdivisions) between each pair of start and end points.
            Must be >= 1. Default is 1.
        materials : list of str, optional
            List of material names to assign to the line mesh. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the subdivided line(s).

        Notes
        -----
        - The function interpolates `segments + 1` points between `start` and `end`
        using `np.linspace`.
        - If `start` and `end` are arrays of shape `(N, 3)`, the method generates
        `N` independent polylines, each subdivided into `segments`.
        - For higher-dimensional batched input, the function reshapes the grid and
        constructs edges using [`col_edges`][npblender.col_edges].

        Examples
        --------
        Create a simple line with 5 segments between (0, 0, 0) and (0, 0, 1):

        ```python
        line = Mesh.line(start=(0, 0, 0), end=(0, 0, 1), segments=5)
        ```

        Create three parallel lines defined by arrays of start and end points:

        ```python
        starts = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [2, 0, 0]])
        ends = np.array([[0, 0, 1],
                        [1, 0, 1],
                        [2, 0, 1]])
        lines = Mesh.line(start=starts, end=ends, segments=4)
        ```

        See Also
        --------
        [`border_edges`][npblender.border_edges] :
            Helper for constructing consecutive edges in a single polyline.
        [`col_edges`][npblender.col_edges] :
            Helper for constructing edges in multi-dimensional point grids.

        > ***Note:*** The line mesh consists only of vertices and edges,
        > no faces are created.
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
        """
        Create a rectangular grid mesh.

        The grid is constructed in the XY plane with indexing set to `'ij'`,
        meaning the generated arrays have shape `(vertices_x, vertices_y)`.

        Parameters
        ----------
        size_x : float, optional
            Size of the grid along the X axis. Default is 1.
        size_y : float, optional
            Size of the grid along the Y axis. Default is 1.
        vertices_x : int, optional
            Number of vertices along the X axis. Must be >= 2. Default is 3.
        vertices_y : int, optional
            Number of vertices along the Y axis. Must be >= 2. Default is 3.
        materials : list of str, optional
            List of material names to assign to the grid. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the rectangular grid.

        Notes
        -----
        - The grid is created with `'ij'` indexing, so coordinates follow
        NumPy's `meshgrid(..., indexing='ij')` convention.
        - UV coordinates are generated using [`grid_uv_map`][npblender.grid_uv_map].
        - The grid topology is built using [`grid_corners`][npblender.grid_corners].

        Examples
        --------
        Create a 2x2 grid with 10 vertices along X and 5 along Y:

        ```python
        grid = Mesh.grid(size_x=2, size_y=2, vertices_x=10, vertices_y=5)
        ```

        Create a square grid of size 5 with default vertex count:

        ```python
        grid = Mesh.grid(size_x=5, size_y=5)
        ```

        See Also
        --------
        [`grid_corners`][npblender.grid_corners] :
            Helper for constructing the corner topology of the grid.
        [`grid_uv_map`][npblender.grid_uv_map] :
            Generates UV coordinates for a regular grid.

        > ***Important:*** The grid is always created with `'ij'` indexing
        > (shape = `(vertices_x, vertices_y)`).
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
        """
        Create a cube mesh.

        Parameters
        ----------
        size : float or array-like of shape (3,), optional
            Size of the cube. If a single float is given, the cube is uniform in all
            dimensions. If an array of three floats is given, it specifies the size
            along the X, Y, and Z axes. Default is 2.
        materials : list of str, optional
            List of material names to assign to the cube. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the cube.

        Notes
        -----
        - The cube is created centered at the origin with side length `size`.
        - UV coordinates are assigned so that all six faces are unwrapped into
        a cross-like layout.

        Examples
        --------
        Create a default cube of size 2:

        ```python
        cube = Mesh.cube()
        ```

        Create a cube of size 5:

        ```python
        cube = Mesh.cube(size=5)
        ```

        Create a rectangular box of dimensions (2, 3, 4):

        ```python
        box = Mesh.cube(size=(2, 3, 4))
        ```

        See Also
        --------
        [`Mesh`][npblender.Mesh] :
            The mesh class used to construct and manage geometry.

        > ***Note:*** The cube is centered at the origin and scaled by `size/2`
        > after construction.
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
        """
        Create a circle mesh.

        The circle can be created as:
        - An open ring (`cap='NONE'`).
        - A filled n-gon (`cap='NGON'`).
        - A triangle fan (`cap='FANS'`).

        The argument `fill_segments` controls how the interior of the circle is filled:
        - If `fill_segments == 0` and `cap='NGON'`, the circle is filled with a single polygon.
        - If `fill_segments > 0`, the circle is filled with concentric rings and
        triangle fans (not yet implemented in this method, but the behavior
        corresponds to `cap='FANS'`).
        
        > ***Note:*** The [`disk`][npblender.Mesh.disk] method provides the same
        > functionality with `cap='NGON'` as its default mode.

        Parameters
        ----------
        radius : float, optional
            Radius of the circle. Default is 1.
        segments : int, optional
            Number of segments (vertices) around the circle. Default is 16.
        fill_segments : int, optional
            Number of internal subdivisions (concentric circles). If 0, the circle is
            filled with a single polygon when `cap='NGON'`. Default is 0.
        cap : {'NONE', 'NGON', 'FANS'}, optional
            How to fill the interior of the circle. Default is 'NONE'.
        materials : list of str, optional
            List of material names to assign to the circle. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the circle.

        Notes
        -----
        - `cap='NONE'`: returns only the ring of edges.
        - `cap='NGON'`: fills the circle with a polygon face.
        - `cap='FANS'`: fills the circle with a fan of triangles around a central point.
        - UV coordinates are generated with [`disk_uv_map`][npblender.disk_uv_map].
        - Fan topology is generated with [`fans_corners`][npblender.fans_corners].

        Examples
        --------
        Create an open circle with 32 segments:

        ```python
        circle = Mesh.circle(radius=1, segments=32, cap='NONE')
        ```

        Create a filled disk using an n-gon:

        ```python
        circle = Mesh.circle(radius=2, segments=24, cap='NGON')
        ```

        Create a filled disk with triangle fans:

        ```python
        circle = Mesh.circle(radius=1, segments=16, cap='FANS')
        ```

        See Also
        --------
        [`disk`][npblender.Mesh.disk] :
            Equivalent method for creating disks (default `cap='NGON'`).
        [`disk_uv_map`][npblender.disk_uv_map] :
            Generates UV coordinates for circular caps.
        [`fans_corners`][npblender.fans_corners] :
            Generates corner topology for triangle fans.

        > ***Caution:*** When using `cap='FANS'`, a new center vertex is added.
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

        assert(False)
    
    # ----------------------------------------------------------------------------------------------------
    # Disk
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def disk(cls, radius=1, segments=16, fill_segments=0, cap='NGON', materials=None):
        """
        Create a disk mesh.

        This is equivalent to [`circle`][npblender.Mesh.circle], but with
        `cap='NGON'` as the default filling mode.

        Parameters
        ----------
        radius : float, optional
            Radius of the disk. Default is 1.
        segments : int, optional
            Number of segments (vertices) around the disk. Default is 16.
        fill_segments : int, optional
            Number of internal subdivisions (concentric circles).
            Default is 0 (single n-gon when `cap='NGON'`).
        cap : {'NONE', 'NGON', 'FANS'}, optional
            How to fill the interior of the disk. Default is 'NGON'.
        materials : list of str, optional
            List of material names to assign to the disk. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the disk.

        Examples
        --------
        Create a default disk of radius 2 with 32 segments:

        ```python
        disk = Mesh.disk(radius=2, segments=32)
        ```

        Create a disk filled with triangle fans:

        ```python
        disk = Mesh.disk(radius=1, segments=16, cap='FANS')
        ```

        See Also
        --------
        [`circle`][npblender.Mesh.circle] :
            General method for circle/disk creation with customizable cap.
        [`disk_uv_map`][npblender.disk_uv_map] :
            Generates UV coordinates for circular caps.
        [`fans_corners`][npblender.fans_corners] :
            Generates corner topology for triangle fans.

        > ***Note:*** This method is a shorthand for `circle(..., cap='NGON')`.
        """
        return cls.circle(radius=radius, segments=segments, fill_segments=fill_segments, cap=cap, materials=materials)
    
    # ----------------------------------------------------------------------------------------------------
    # Cone
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def cone(cls, vertices=32, side_segments=1, fill_segments=1, radius_top=0,
            radius_bottom=1, depth=2, fill_type='NGON', materials=None):
        """
        Create a cone (or cylinder) mesh.

        Parameters
        ----------
        vertices : int, optional
            Number of vertices around the circumference. Default is 32.
        side_segments : int, optional
            Number of subdivisions along the vertical side edges.
            Default is 1 (no subdivision).
        fill_segments : int, optional
            Number of concentric circles added to the caps. Currently unused.
            Default is 1.
        radius_top : float, optional
            Radius of the top face. Default is 0 (cone).
        radius_bottom : float, optional
            Radius of the bottom face. Default is 1.
        depth : float, optional
            Height of the cone along the Z axis. Default is 2.
        fill_type : {'NGON', 'FANS', 'NONE'}, optional
            Type of filling for the top and bottom caps:
            - `'NGON'`: fill with n-gons.
            - `'FANS'`: fill with triangle fans.
            - `'NONE'`: no cap filling.
            Default is `'NGON'`.
        materials : list of str, optional
            List of material names to assign to the mesh. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the cone.

        Notes
        -----
        - If both `radius_top` and `radius_bottom` are zero, the result is an empty mesh.
        - Internally calls [`bl_cone`][npblender.Mesh.bl_cone] with
        `cap_ends` and `cap_tris` derived from `fill_type`.
        - UVs are generated automatically by Blender's cone operator.

        Examples
        --------
        Create a simple cone of height 2 and base radius 1:

        ```python
        cone = Mesh.cone(vertices=32, radius_top=0, radius_bottom=1, depth=2)
        ```

        Create a cylinder with 16 vertices and subdivided sides:

        ```python
        cylinder = Mesh.cone(vertices=16, radius_top=1, radius_bottom=1,
                            depth=3, side_segments=3)
        ```

        Create a cone with triangle fan caps:

        ```python
        cone = Mesh.cone(vertices=24, radius_top=0, radius_bottom=2,
                        depth=4, fill_type='FANS')
        ```

        See Also
        --------
        [`bl_cone`][npblender.Mesh.bl_cone] :
            Low-level constructor for cones and cylinders.
        [`bl_circle`][npblender.Mesh.bl_circle] :
            For creating circle primitives with optional triangle fan filling.

        > ***Note:*** Use `fill_type='NONE'` to create an open-ended cone or cylinder.
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
    def cylinder(cls, vertices=32, side_segments=1, radius=1, depth=2,
                fill_type='NGON', materials=None):
        """
        Create a cylinder mesh.

        Parameters
        ----------
        vertices : int, optional
            Number of vertices around the circumference. Default is 32.
        side_segments : int, optional
            Number of subdivisions along the vertical side edges.
            Default is 1 (no subdivision).
        radius : float, optional
            Radius of both the top and bottom faces. Default is 1.
        depth : float, optional
            Height of the cylinder along the Z axis. Default is 2.
        fill_type : {'NGON', 'FANS', 'NONE'}, optional
            Type of filling for the top and bottom caps:
            - `'NGON'`: fill with n-gons.
            - `'FANS'`: fill with triangle fans.
            - `'NONE'`: no cap filling.
            Default is `'NGON'`.
        materials : list of str, optional
            List of material names to assign to the mesh. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the cylinder.

        Notes
        -----
        - Internally calls [`bl_cone`][npblender.Mesh.bl_cone] with `radius1 = radius2 = radius`.
        - UVs are generated automatically by Blender's cone operator.

        Examples
        --------
        Create a default cylinder of radius 1 and height 2:

        ```python
        cyl = Mesh.cylinder()
        ```

        Create a cylinder with 64 vertices and 4 vertical subdivisions:

        ```python
        cyl = Mesh.cylinder(vertices=64, side_segments=4, radius=2, depth=5)
        ```

        Create an open cylinder without caps:

        ```python
        cyl = Mesh.cylinder(radius=1, depth=3, fill_type='NONE')
        ```

        See Also
        --------
        [`bl_cone`][npblender.Mesh.bl_cone] :
            Low-level constructor for cones and cylinders.
        [`cone`][npblender.Mesh.cone] :
            Generalized method for cones and cylinders.

        > ***Note:*** This method is a convenience wrapper for `bl_cone` with
        > equal top and bottom radii.
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
        """
        Create a pyramid mesh.

        The pyramid is generated as a cone with 3 vertices at the base (a triangle)
        and an apex at the top.

        Parameters
        ----------
        size : float, optional
            Size of the pyramid. Determines both the base dimensions and the height.
            Default is 1.
        materials : list of str, optional
            List of material names to assign to the mesh. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the pyramid.

        Notes
        -----
        - The base radius is scaled by `size * sqrt(3)/2` so that the pyramid has
        approximately unit proportions when `size=1`.
        - Internally calls [`cone`][npblender.Mesh.cone] with `vertices=3`.

        Examples
        --------
        Create a default pyramid of size 1:

        ```python
        pyramid = Mesh.pyramid()
        ```

        Create a larger pyramid of size 5:

        ```python
        pyramid = Mesh.pyramid(size=5)
        ```

        See Also
        --------
        [`cone`][npblender.Mesh.cone] :
            Generalized method for cones and pyramids.
        [`bl_cone`][npblender.Mesh.bl_cone] :
            Low-level constructor for cone-based primitives.

        > ***Note:*** This method is equivalent to creating a triangular-based cone.
        """
        return cls.cone(vertices=3, radius_bottom=size*0.8660254037844386, depth=size, materials=materials)

    # ----------------------------------------------------------------------------------------------------
    # UV Sphere
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def uvsphere(cls, segments=32, rings=16, radius=1, materials=None):
        """
        Create a UV sphere mesh.

        Parameters
        ----------
        segments : int, optional
            Number of longitudinal segments (meridians). Default is 32.
        rings : int, optional
            Number of latitudinal rings (parallels). Default is 16.
        radius : float, optional
            Radius of the sphere. Default is 1.
        materials : list of str, optional
            List of material names to assign to the sphere. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the UV sphere.

        Notes
        -----
        - The sphere is created using [`bmesh.ops.create_uvsphere`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_uvsphere)
        with `calc_uvs=True` so UV coordinates are automatically generated.
        - The geometry is distributed evenly in the UV parameterization, which means
        denser vertices near the poles.

        Examples
        --------
        Create a default UV sphere of radius 1:

        ```python
        sphere = Mesh.uvsphere()
        ```

        Create a high-resolution sphere:

        ```python
        sphere = Mesh.uvsphere(segments=64, rings=32, radius=2)
        ```

        See Also
        --------
        [`icosphere`][npblender.Mesh.icosphere] :
            Alternative sphere primitive with more uniform vertex distribution.
        [`bmesh.ops.create_uvsphere`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_uvsphere) :
            BMesh operator used for creating UV spheres.

        > ***Note:*** Use [`icosphere`][npblender.Mesh.icosphere] if you need
        > a more uniform tessellation without poles.
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
        """
        Create an icosphere mesh.

        Parameters
        ----------
        radius : float, optional
            Radius of the icosphere. Default is 1.
        subdivisions : int, optional
            Number of recursive subdivisions applied to the base icosahedron.
            Higher values yield smoother spheres. Clamped to a maximum of 10.
            Default is 2.
        materials : list of str, optional
            List of material names to assign to the icosphere. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the icosphere.

        Notes
        -----
        - The icosphere is created using
        [`bmesh.ops.create_icosphere`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_icosphere)
        with `calc_uvs=True` so UV coordinates are automatically generated.
        - Unlike a UV sphere, an icosphere has more uniform vertex distribution,
        making it suitable for certain simulation and subdivision tasks.
        - Subdivisions are internally capped at 10 for performance reasons.

        Examples
        --------
        Create a default icosphere of radius 1 with 2 subdivisions:

        ```python
        ico = Mesh.icosphere()
        ```

        Create a larger icosphere with 4 subdivisions:

        ```python
        ico = Mesh.icosphere(radius=3, subdivisions=4)
        ```

        See Also
        --------
        [`uvsphere`][npblender.Mesh.uvsphere] :
            Sphere primitive based on UV parameterization.
        [`bmesh.ops.create_icosphere`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_icosphere) :
            BMesh operator used for creating icospheres.

        > ***Note:*** Use [`uvsphere`][npblender.Mesh.uvsphere] when you
        > require consistent UV mapping, and `icosphere` for uniform tessellation.
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
        """
        Create a torus mesh.

        Parameters
        ----------
        major_segments : int, optional
            Number of segments around the major (outer) radius. Default is 48.
        minor_segments : int, optional
            Number of segments around the minor (inner) radius (the cross-section). Default is 12.
        major_radius : float, optional
            The distance from the center of the torus to the center of the tube. Default is 1.
        minor_radius : float, optional
            The radius of the tube itself. Default is 0.25.
        materials : list of str, optional
            List of material names to assign to the torus. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the torus.

        Notes
        -----
        - The torus is constructed by sweeping a circle of radius `minor_radius`
        around a larger circle of radius `major_radius`.
        - The transformation of the cross-section is handled by
        [`Transformation`][npblender.transform.Transformation] and
        [`Rotation`][npblender.rotation.Rotation].
        - UV coordinates are generated using [`grid_uv_map`][npblender.grid_uv_map],
        resulting in a square parameterization.
        - Topology is constructed with [`grid_corners`][npblender.grid_corners]
        with both axes closed.

        Examples
        --------
        Create a standard torus:

        ```python
        torus = Mesh.torus()
        ```

        Create a torus with a larger tube and finer resolution:

        ```python
        torus = Mesh.torus(major_segments=64, minor_segments=32,
                        major_radius=2, minor_radius=0.5)
        ```

        See Also
        --------
        [`grid_corners`][npblender.grid_corners] :
            Helper for constructing the corner topology of the torus grid.
        [`grid_uv_map`][npblender.grid_uv_map] :
            Generates UV coordinates for grid-like surfaces.
        [`Transformation`][npblender.transform.Transformation] :
            Used to position and orient the swept circle.
        [`Rotation`][npblender.rotation.Rotation] :
            Used to orient the minor circle along the sweep path.

        > ***Note:*** UV coordinates are generated with an offset of π to match
        > Blender's default torus orientation.
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
        """
        Create the famous Blender "Suzanne" monkey mesh.

        Parameters
        ----------
        materials : list of str, optional
            List of material names to assign to the mesh. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the Suzanne primitive.

        Notes
        -----
        - The monkey head is created using
        [`bmesh.ops.create_monkey`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_monkey).
        - Suzanne is often used as a test model and is considered Blender’s mascot.

        Examples
        --------
        Create a Suzanne mesh:

        ```python
        monkey = Mesh.monkey()
        ```

        See Also
        --------
        [`bmesh.ops.create_monkey`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_monkey) :
            BMesh operator used to generate the Suzanne primitive.

        > ***Note:*** Suzanne is widely used as a benchmark and test object in Blender.
        """
        mesh = cls(materials=materials)
        with mesh.bmesh() as bm:
            bmesh.ops.create_monkey(bm)

        return mesh

    # ----------------------------------------------------------------------------------------------------
    # Arrow
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def arrow(cls, vector=(0, 0, 1), radius=.05, angle=24., segments=8,
            adjust_norm=None, materials=None):
        """
        Create an arrow mesh oriented along a given vector.

        The arrow is composed of a cylindrical shaft and a conical head,
        proportionally scaled to the length of the input vector.

        Parameters
        ----------
        vector : array-like of float, shape (3,), optional
            Direction and length of the arrow. The norm of the vector defines the
            arrow length. Default is (0, 0, 1).
        radius : float, optional
            Radius of the cylindrical shaft. Default is 0.05.
        angle : float, optional
            Opening angle of the conical head in degrees. Default is 24.
        segments : int, optional
            Number of segments around the circumference. Default is 8.
        adjust_norm : {callable, float, None}, optional
            - If callable: a function applied to the vector norm to adjust the
            arrow length.
            - If float: the arrow length is clamped to this maximum.
            - If None: use the norm of `vector` directly. Default is None.
        materials : list of str, optional
            List of material names to assign to the arrow. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the arrow.

        Notes
        -----
        - The shaft is created with [`cylinder`][npblender.Mesh.cylinder].
        - The head is created with [`cone`][npblender.Mesh.cone] using
        `fill_type='FANS'` for proper triangulation.
        - The arrow is aligned to `vector` using
        [`Rotation.look_at`][npblender.rotation.Rotation.look_at].
        - A small correction is applied to avoid overlap between shaft and head.

        Examples
        --------
        Create a default arrow of length 1 along Z:

        ```python
        arrow = Mesh.arrow()
        ```

        Create an arrow along vector (1, 2, 0.5) with custom shaft radius:

        ```python
        arrow = Mesh.arrow(vector=(1, 2, 0.5), radius=0.1)
        ```

        Create an arrow clamped to maximum length 2:

        ```python
        arrow = Mesh.arrow(vector=(0, 0, 5), adjust_norm=2)
        ```

        See Also
        --------
        [`cylinder`][npblender.Mesh.cylinder] :
            Used to create the arrow shaft.
        [`cone`][npblender.Mesh.cone] :
            Used to create the arrow head.
        [`Rotation.look_at`][npblender.rotation.Rotation.look_at] :
            Utility to orient the arrow along a target vector.

        > ***Caution:*** If `vector` has zero length, the arrow cannot be
        > constructed properly.

        > ***Note:*** The conical head radius is set to `3 * radius` by default,
        > and its height is determined by the opening `angle`.
        """
        height = np.linalg.norm(vector)
        if type(adjust_norm).__name__ == 'function':
            height = adjust_norm(height)
        elif adjust_norm is not None:
            height = min(adjust_norm, height)

        head_radius = 3*radius
        head_height = head_radius/np.tan(np.radians(angle))

        cyl_height = height - head_height*.8

        cyl = cls.cylinder(vertices=segments, side_segments=2, radius=radius, depth=cyl_height, materials=materials)
        cyl.points.z += cyl_height/2
        cyl.points[[segments + i for i in range(segments)]].position -= (0, 0, cyl_height/2 - .01)

        cone = cls.cone(vertices=segments, side_segments=2, fill_segments=1, radius_top=0, radius_bottom=head_radius, depth=head_height, fill_type='FANS', materials=materials)
        cone.points[-1].position += (0, 0, head_height/10)
        cone.points.position += (0, 0, height - head_height/2)

        arrow = cyl.join(cone)
        arrow.points.position = Rotation.look_at((0, 0, 1), vector) @ arrow.points.position

        return arrow
    
    # ----------------------------------------------------------------------------------------------------
    # Field of vectors
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def vectors_field(cls, locations, vectors, radius=.05, scale_length=1., angle=24.,
                    segments=8, head=None, adjust_norm=None, materials=None):
        """
        Create an arrow at each `location` oriented and scaled by the corresponding `vector`.

        Each arrow consists of a cylindrical shaft and a conical head aligned with the
        vector direction. Arrow length is derived from the vector norm, optionally
        transformed by `adjust_norm`. For very short vectors, the arrow is scaled
        down to preserve proportions.

        Parameters
        ----------
        locations : array-like of shape (N, 3)
            Positions where arrows are placed.
        vectors : array-like of shape (N, 3)
            Direction (and base length) of each arrow. Must match the length of `locations`.
        radius : float, optional
            Shaft radius for the arrows. Default is 0.05.
        scale_length : float, optional
            Length threshold below which arrows are uniformly scaled down (radius and
            shaft) while keeping proportions. Default is 1.0.
        angle : float, optional
            Opening angle (in degrees) of the conical head. Default is 24.
        segments : int, optional
            Number of radial segments for both shaft and head. Default is 8.
        head : Mesh or None, optional
            Optional mesh to use as the arrow head. If `None`, a cone is created.
            When provided, its Z size defines the head height.
        adjust_norm : {callable, float, None}, optional
            Controls how vector norms are mapped to arrow lengths:
            - callable: applied to the array of norms.
            - float: acts as a maximum length (clamp).
            - None: use the raw norms. Default is None.
        materials : list of str, optional
            Material names to assign to created geometry. Default is None.

        Returns
        -------
        Mesh
            A mesh containing all arrows.

        Notes
        -----
        - The head radius is `3 * radius`; its height is derived from `angle` via
        `head_height = head_radius / tan(angle)`.
        - Arrows with very small vectors are handled specially to avoid degenerate
        geometry (a minimum total length of approximately `2 * head_height` is enforced).
        - Alignment is achieved with [`Rotation.look_at`][npblender.rotation.Rotation.look_at].
        - The shaft is built from [`cylinder`][npblender.Mesh.cylinder], and the
        head from [`cone`][npblender.Mesh.cone] (when `head is None`).

        Examples
        --------
        Create a field of unit arrows from a grid:

        ```python
        P = np.stack(np.meshgrid(np.linspace(-1, 1, 5),
                                np.linspace(-1, 1, 5),
                                [0.0], indexing='ij'), axis=-1).reshape(-1, 3)
        V = np.tile(np.array([0, 0, 1.0]), (len(P), 1))
        field = Mesh.vectors_field(P, V, radius=0.03, segments=12)
        ```

        Clamp arrow lengths to 2 units:

        ```python
        field = Mesh.vectors_field(P, V * 5.0, adjust_norm=2.0)
        ```

        Map norms nonlinearly (e.g., sqrt):

        ```python
        field = Mesh.vectors_field(P, V, adjust_norm=np.sqrt)
        ```

        See Also
        --------
        [`arrow`][npblender.Mesh.arrow] :
            Convenience method to create a single arrow.
        [`cylinder`][npblender.Mesh.cylinder] :
            Used to create the arrow shafts.
        [`cone`][npblender.Mesh.cone] :
            Used to create the arrow heads (when `head is None`).
        [`Rotation.look_at`][npblender.rotation.Rotation.look_at] :
            Used to orient arrows along their vectors.

        > ***Caution:*** `locations` and `vectors` must have the same length (N).
        > Mismatched inputs will lead to incorrect alignment or runtime errors.

        > ***Note:*** Zero-length vectors are handled safely; corresponding arrows
        > collapse to length 0 and are effectively omitted.
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
    def chain_link(cls, major_segments=48, minor_segments=12, radius=1., section=0.5,
                length=4., materials=None):
        """
        Create a single chain link (oval torus with straightened sides).

        The link is built from a torus of major radius `radius` and tube radius
        `section / 2`. If `length > 2 * radius`, the torus is split in half,
        translated to open a gap of size `delta = length - 2 * radius`, mirrored,
        then the opposite borders are bridged to form the elongated link. UVs are
        adjusted to keep a clean seam layout.

        Parameters
        ----------
        major_segments : int, optional
            Number of segments around the major loop. Default is 48.
        minor_segments : int, optional
            Number of segments around the tube section. Default is 12.
        radius : float, optional
            Major radius of the link (half the distance between opposite sides on
            the long axis before elongation). Default is 1.0.
        section : float, optional
            Diameter of the link cross-section (tube thickness). Default is 0.5.
        length : float, optional
            Target overall length of the link along its long axis. If close to
            `2 * radius`, the result is essentially a pure torus. Default is 4.0.
        materials : list of str, optional
            Material names to assign to the link. Default is None.

        Returns
        -------
        Mesh
            A new mesh instance containing the chain link.

        Notes
        -----
        - Construction steps:
        1) Create a torus with [`torus`][npblender.Mesh.torus].
        2) Delete approximately half the vertices on the negative Y side with
            [`delete_vertices`][npblender.Mesh.delete_vertices].
        3) Duplicate and mirror the remaining half to the other side.
        4) Bridge the facing border loops with
            [`bridge_loops`][npblender.Mesh.bridge_loops] (twice, crossing).
        5) Recompute and assign UVs using [`grid_uv_map`][npblender.grid_uv_map]
            to distribute the texture coordinates and minimize stretching.
        - When `length - 2 * radius` is smaller than ~`radius / 10`, the method
        returns the original torus since elongation would be negligible.

        Examples
        --------
        Create a standard chain link:

        ```python
        link = Mesh.chain_link(major_segments=64, minor_segments=16,
                            radius=0.5, section=0.12, length=1.6)
        ```

        Create a thicker, longer link:

        ```python
        link = Mesh.chain_link(radius=1.0, section=0.25, length=3.0)
        ```

        See Also
        --------
        [`torus`][npblender.Mesh.torus] :
            Base primitive used to start the link.
        [`delete_vertices`][npblender.Mesh.delete_vertices] :
            Used to remove half of the torus before mirroring.
        [`bridge_loops`][npblender.Mesh.bridge_loops] :
            Used to reconnect mirrored borders.
        [`grid_uv_map`][npblender.grid_uv_map] :
            Generates UVs for the final link surface.
        [`from_mesh`][npblender.Mesh.from_mesh] :
            Utility for duplicating mesh halves before joining.

        > ***Caution:*** Very small `section` relative to `major_segments` can
        > create skinny triangles near the bridged areas. Increase segment counts
        > or `section` for cleaner topology.

        > ***Note:*** If `length <= 2 * radius`, no elongation is performed and
        > the result is (nearly) identical to a torus of the given parameters.
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
        """
        Extrude individual vertices by creating new points displaced by `offset`
        and connecting each original vertex to its duplicate with an edge.

        Parameters
        ----------
        selection : array-like of int or bool, or None
            Vertex indices or boolean mask selecting the vertices to extrude.
            If `None`, all vertices are extruded.
        offset : array-like, shape (3,) or (N, 3)
            Extrusion vector(s). Can be a single 3D vector applied to every
            selected vertex, or an array of vectors with one per selected vertex.
        **attributes : dict, optional
            Optional attributes to attach to the created geometry (forwarded to
            [`add_geometry`][npblender.Mesh.add_geometry]).

        Returns
        -------
        dict
            Dictionary describing the created geometry as returned by
            [`add_geometry`][npblender.Mesh.add_geometry]. Contains at least:
            - `'points'`: indices of newly added vertices.
            - `'edges'`: indices of newly added edges.

        Raises
        ------
        AttributeError
            If `offset` is neither a single `(3,)` vector nor an array of shape
            `(len(loop), 3)`.

        Notes
        -----
        - New vertices are positioned at `points[selection] + offset`.
        - One edge is created between each original vertex and its newly created
        counterpart using [`edges_between`][npblender.edges_between].

        Examples
        --------
        Extrude all vertices by (0, 0, 1):

        ```python
        added = Mesh.extrude_vertices(selection=None, offset=(0, 0, 1))
        ```

        Extrude a subset with per-vertex offsets:

        ```python
        sel = np.array([0, 2, 5, 7])
        offs = np.random.randn(len(sel), 3) * 0.1
        added = Mesh.extrude_vertices(selection=sel, offset=offs)
        ```

        See Also
        --------
        [`add_geometry`][npblender.Mesh.add_geometry] :
            Adds new points/edges/faces and returns their indices.
        [`edges_between`][npblender.edges_between] :
            Builds edge pairs between two index arrays of equal length.

        > ***Caution:*** When `offset` is an array, its length must match the
        > number of selected vertices.

        > ***Note:*** This operation creates only points and edges. Faces are not
        > generated automatically.
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
        """
        Extrude a loop of vertices by duplicating the loop, offsetting it, and
        creating a quad strip between the original and the offset loop.

        Parameters
        ----------
        loop : array-like of int
            Vertex indices defining the loop to extrude. Must contain at least 2 vertices.
        offset : array-like, shape (3,) or (N, 3)
            Extrusion vector(s). A single 3D vector is broadcast to all vertices in
            `loop`, or provide one vector per vertex (N == len(loop)).
        close : bool, optional
            If True, treats the input as a closed loop and connects the last vertex
            back to the first when building side quads. Default is False.
        clockwise : bool, optional
            Controls the orientation (winding) of the generated faces and the UV
            layout. Default is False.
        **attributes : dict, optional
            Extra attributes intended for the new geometry (see ***Caution***).

        Returns
        -------
        dict
            Dictionary describing the created geometry as returned by
            [`add_geometry`][npblender.Mesh.add_geometry]. Contains at least:
            - `'points'`: indices of the duplicated (offset) vertices.
            - `'corners'`: indices of the generated quad strip corners.
            - `'faces'`: face arity (4 for quads).

        Raises
        ------
        AttributeError
            If `offset` is neither a single `(3,)` vector nor an array of shape
            `(len(loop), 3)`.

        Notes
        -----
        - New vertices are computed as `points[loop] + offset` (with broadcasting if
        `offset` is a single vector).
        - Side faces are constructed using the topology from
        [`grid_corners`][npblender.grid_corners] with two rows (original and
        offset loop).
        - UVs for the side strip are generated by
        [`grid_uv_map`][npblender.grid_uv_map] with matching parameters.

        Examples
        --------
        Extrude an open loop along a single vector:

        ```python
        new = Mesh.extrude_loop(loop, offset=(0, 0, 1), close=False)
        ```

        Extrude a closed loop with per-vertex offsets and flipped winding:

        ```python
        offs = np.random.randn(len(loop), 3) * 0.02
        new = Mesh.extrude_loop(loop, offset=offs, close=True, clockwise=True)
        ```

        See Also
        --------
        [`extrude_vertices`][npblender.Mesh.extrude_vertices] :
            Extrude isolated vertices with edges to their duplicates.
        [`add_geometry`][npblender.Mesh.add_geometry] :
            Adds the new points/corners/faces and returns their indices.
        [`grid_corners`][npblender.grid_corners] :
            Builds the quad topology of the side strip.
        [`grid_uv_map`][npblender.grid_uv_map] :
            Generates UVs for the side strip.

        > ***Caution:*** `offset` must be either a single `(3,)` vector or an array
        > of shape `(len(loop), 3)`. Any other shape will raise an error.

        > ***Caution:*** The `attributes` kwargs are currently **not forwarded** to
        > `add_geometry` in this implementation. If you need them applied, pass
        > them through explicitly in the call to `add_geometry`.
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
        """
        Extrude individual faces by duplicating them, optionally displacing
        them by `offset`, and connecting side faces.

        Parameters
        ----------
        selection : array-like of int or bool, or None
            Indices (or mask) of faces to extrude. If `None`, all faces are extruded.
        offset : array-like, shape (3,) or (N, 3), optional
            Extrusion vector(s). A single vector is broadcast to all faces.
            If `None`, each face is extruded along its own normal.
        scale : float, optional
            Scale factor applied to `offset` (or to the face normal if `offset=None`).
            Default is 1.0.
        dissolve : bool, optional
            Not implemented in this version. Placeholder for removing the starting
            faces after extrusion.

        Returns
        -------
        dict
            Dictionary with two keys:
            - `'top'`: indices of the extruded (displaced) faces.
            - `'side'`: indices of the side faces connecting the original and new faces.

        Raises
        ------
        ValueError
            If `offset` cannot be broadcast to shape `(len(faces), 3)`.
        AssertionError
            If a side edge of an extruded face does not have exactly two linked faces.

        Notes
        -----
        - Uses [`bmesh.ops.extrude_discrete_faces`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.extrude_discrete_faces)
        to duplicate each selected face independently.
        - If `offset` is `None`, displacement is along each face's local normal.
        - Side faces are identified by checking edges linked to the extruded faces.

        Examples
        --------
        Extrude all faces along their normals:

        ```python
        res = Mesh.extrude_faces(selection=None, scale=0.2)
        ```

        Extrude a subset of faces by a fixed offset:

        ```python
        res = Mesh.extrude_faces(selection=[0, 2, 5], offset=(0, 0, 1))
        ```

        Extrude faces with per-face offsets:

        ```python
        offs = np.random.randn(len(sel), 3) * 0.1
        res = Mesh.extrude_faces(selection=sel, offset=offs)
        ```

        See Also
        --------
        [`extrude_vertices`][npblender.Mesh.extrude_vertices] :
            Extrude isolated vertices.
        [`extrude_loop`][npblender.Mesh.extrude_loop] :
            Extrude a vertex loop into a quad strip.

        > ***Caution:*** If `offset` is given per-face, its length must match the
        > number of extruded faces or broadcasting will fail.

        > ***Note:*** Side face indices may be repeated if multiple extrusions
        > share edges.
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
        """
        Extrude a connected face region, translate the new geometry by `offset`,
        and optionally dissolve the original faces.

        Parameters
        ----------
        selection : array-like of int or bool, or None
            Indices (or mask) of faces to extrude. If `None`, all faces are used.
        offset : array-like of float, shape (3,), optional
            Translation vector applied to the newly created vertices of the region.
            Default is (0, 0, 1).
        dissolve : bool, optional
            If True, delete the original (pre-extrusion) faces after the region
            has been extruded. Default is False.

        Returns
        -------
        dict
            Dictionary with two keys:
            - `'top'`: indices of the newly extruded faces (translated region).
            - `'side'`: indices of the side faces that connect original and new faces.

        Raises
        ------
        AssertionError
            If a side edge of an extruded face does not have exactly two linked faces
            (non-manifold condition).

        Notes
        -----
        - Region extrusion is performed via
        [`bmesh.ops.extrude_face_region`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.extrude_face_region),
        then the new vertices are moved using `bmesh.ops.translate`.
        - Side faces are discovered by scanning the edges of the extruded faces and
        collecting the adjacent face opposite to each extruded face.

        Examples
        --------
        Extrude a region upward and keep the original faces:

        ```python
        res = Mesh.extrude_region(selection=[0, 1, 2], offset=(0, 0, 0.2), dissolve=False)
        ```

        Extrude a region and dissolve the starting faces:

        ```python
        res = Mesh.extrude_region(selection=mask, offset=(0.1, 0, 0), dissolve=True)
        ```

        See Also
        --------
        [`extrude_faces`][npblender.Mesh.extrude_faces] :
            Extrude faces individually (discrete), not as a connected region.
        [`extrude_loop`][npblender.Mesh.extrude_loop] :
            Create a quad strip by offsetting a vertex loop.
        [`extrude_vertices`][npblender.Mesh.extrude_vertices] :
            Duplicate and connect selected vertices.

        > ***Caution:*** `offset` must be a 3D vector. Non-3D inputs may cause the
        > translation operator to fail.

        > ***Note:*** With `dissolve=True`, the original faces are removed, leaving
        > only the extruded shell.
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

    def inset_faces(self, selection, thickness=0.1, depth=0.0,
                    use_even_offset=True, use_relative_offset=False):
        """
        Inset selected faces individually, optionally adding depth (local extrusion).

        Parameters
        ----------
        selection : array-like of int or bool, or None
            Indices (or mask) of faces to inset. If `None`, all faces are used.
        thickness : float, optional
            Inset thickness applied per face. Default is 0.1.
        depth : float, optional
            Local extrusion depth along each face normal. Default is 0.0.
        use_even_offset : bool, optional
            Keep thickness consistent across faces (even offset). Default is True.
        use_relative_offset : bool, optional
            Scale thickness relative to face size. Default is False.

        Returns
        -------
        dict
            Dictionary with:
            - `'top'`: indices of the original (selected) faces.
            - `'side'`: indices of the new faces created by the inset operation
            (typically the rim/side faces around each inset).

        Notes
        -----
        - Implementation uses
        [`bmesh.ops.inset_individual`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.inset_individual).
        - The `'top'` entry mirrors the input selection; `'side'` comes from
        `d["faces"]` returned by the BMesh operator.

        Examples
        --------
        Inset a set of faces with even offset:

        ```python
        res = Mesh.inset_faces(selection=[0, 2, 5], thickness=0.05, depth=0.0)
        ```

        Inset all faces with relative offset and a small depth:

        ```python
        res = Mesh.inset_faces(selection=None, thickness=0.02,
                            depth=0.01, use_relative_offset=True)
        ```

        See Also
        --------
        [`extrude_faces`][npblender.Mesh.extrude_faces] :
            Extrude faces discretely instead of insetting.
        [`extrude_region`][npblender.Mesh.extrude_region] :
            Extrude connected face regions.
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
        """
        Build a solid base (“socle”) by extruding the outer boundary of a grid mesh
        down (or up) to a given Z level, then bridging the side wall.

        Parameters
        ----------
        shape : tuple of int
            Grid shape `(nx, ny)` of the mesh topology (using `'ij'` indexing).
        z : float, optional
            Target Z coordinate for the base ring (the new bottom boundary).
            Default is 0.
        bottom_material_index : int, optional
            Material index assigned to the bottom face created by the operation.
            Default is 0.

        Returns
        -------
        None
            Modifies the mesh **in place**. Returns `None`.
            The bottom face indices produced by [`add_geometry`][npblender.Mesh.add_geometry]
            are stored internally and their `material_index` is set to `bottom_material_index`.

        Notes
        -----
        - The outer boundary loop is derived from the provided grid `shape` assuming
        a regular lattice of `nx * ny` points laid out with NumPy’s `'ij'` indexing.
        - A new ring of points is created at Z = `z` via [`add_points`][npblender.Mesh.add_points].
        - The vertical side wall is created by bridging loops with
        [`bridge_loops`][npblender.Mesh.bridge_loops] using `close=True`.
        - The bottom face material is assigned by ensuring and editing the optional
        `material_index` field on `self.faces`.

        Examples
        --------
        Solidify a 20×30 grid down to Z = -0.1 with material index 2:

        ```python
        Mesh.solidify_socle(shape=(20, 30), z=-0.1, bottom_material_index=2)
        ```

        See Also
        --------
        [`add_points`][npblender.Mesh.add_points] :
            Adds the new base ring vertices at Z = `z`.
        [`add_geometry`][npblender.Mesh.add_geometry] :
            Creates the bottom polygon from the added ring.
        [`bridge_loops`][npblender.Mesh.bridge_loops] :
            Connects the side wall between original and new boundary loops.

        > ***Caution:*** This method assumes the mesh vertices correspond to a regular
        > `(nx, ny)` grid ordered consistently with `'ij'` indexing; inconsistent
        > layouts will produce incorrect boundaries.

        > ***Note:*** The function does not return the created face indices; it sets
        > their `material_index` internally based on `bottom_material_index`.
        """
        n = shape[0]*shape[1]
        inds = np.arange(n).reshape(shape)

        loop1 = np.append(inds[0, :-1], inds[:-1, -1])
        loop1 = np.append(loop1, np.flip(inds[-1, 1:]))
        loop1 = np.append(loop1, np.flip(inds[1:, 0]))

        points = np.array(self.points.position[loop1])
        points[:, 2] = z

        loop0 = self.add_points(points)
        res = self.add_geometry(corners=loop0) #, faces=len(loop0))

        self.bridge_loops(loop0, loop1, close=True)

        self.faces._ensure_optional_field("material_index")
        self.faces[res['faces']].material_index = bottom_material_index

    # ----------------------------------------------------------------------------------------------------
    # Boolean
    # ----------------------------------------------------------------------------------------------------

    def boolean(self, other, operation='DIFFERENCE'):
        """
        Apply a boolean CSG operation with another mesh object and return the result.

        Parameters
        ----------
        other : Mesh
            The mesh used as the boolean operand.
        operation : {'INTERSECT', 'UNION', 'DIFFERENCE'}, optional
            Type of boolean operation to perform. Default is 'DIFFERENCE'.

        Returns
        -------
        Mesh
            A new mesh instance created from the object after applying the Boolean
            modifier.

        Notes
        -----
        - Internally, a Blender **Boolean** modifier is added to `self`, pointing to
        `other`, and then applied via `bpy.ops.object.modifier_apply`.
        - The result is read back as a new mesh using
        [`Mesh.from_object`][npblender.Mesh.from_object].
        - Context managers [`object`][npblender.Mesh.object] are used to obtain
        temporary Blender objects for both meshes.

        Examples
        --------
        Subtract `B` from `A`:

        ```python
        result = A.boolean(B, operation='DIFFERENCE')
        ```

        Compute the union:

        ```python
        result = A.boolean(B, operation='UNION')
        ```

        Keep only the intersection:

        ```python
        result = A.boolean(B, operation='INTERSECT')
        ```

        See Also
        --------
        [`Mesh.from_object`][npblender.Mesh.from_object] :
            Converts a Blender object back into a mesh wrapper.
        [`object`][npblender.Mesh.object] :
            Context manager yielding a temporary Blender object.

        > ***Warning:*** Applying the modifier is **destructive** to the underlying
        > Blender object for `self` (its mesh data is changed). The method returns
        > a new mesh instance representing the modified result.

        > ***Caution:*** Ensure `operation` is one of {'INTERSECT', 'UNION',
        > 'DIFFERENCE'}; other values are invalid for Blender's Boolean modifier.
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
        """
        Apply a **Solidify** modifier to give thickness to a surface mesh.

        Parameters
        ----------
        thickness : float, optional
            Thickness of the shell to generate. Positive values expand outward,
            negative values inward. Default is 0.01.
        offset : float, optional
            Offset factor determining the solidification direction relative to
            the original surface:
            -1 → inward, 0 → centered, +1 → outward.
            Default is -1.

        Returns
        -------
        Mesh
            A new mesh instance resulting from the solidify operation.

        Notes
        -----
        - Internally creates a Blender **Solidify** modifier with
        `use_even_offset=True` for consistent thickness.
        - The modifier is applied destructively via
        `bpy.ops.object.modifier_apply`, and the resulting mesh is retrieved with
        [`Mesh.from_object`][npblender.Mesh.from_object].
        - Works best on manifold surfaces (open meshes may produce artifacts).

        Examples
        --------
        Solidify a circular glass mesh:

        ```python
        glass = Mesh.circle(segments=128)
        glass.extrude_faces(0, -0.01)
        glass.extrude_faces(0, -2)
        glass.extrude_faces(0, -0.01)
        glass.points.translate((0, 0, 2))

        glass = glass.solidify(thickness=0.1)
        glass.to_object("Solidify", shade_smooth=True)
        ```

        See Also
        --------
        [`Mesh.from_object`][npblender.Mesh.from_object] :
            Converts a Blender object back into a mesh wrapper.
        [`object`][npblender.Mesh.object] :
            Context manager yielding a temporary Blender object.

        > ***Caution:*** Applying the modifier is **destructive** to the underlying
        > Blender object; the method returns a new mesh instance of the result.
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

    def remove_doubles(self, dist=0.001):
        """
        Merge duplicate vertices within a distance threshold.

        Parameters
        ----------
        dist : float, optional
            Maximum distance between vertices to be merged. Default is 0.001.

        Returns
        -------
        Mesh
            The current mesh instance (`self`) with duplicate vertices removed.

        Notes
        -----
        - Internally uses
        [`bmesh.ops.remove_doubles`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.remove_doubles).
        - All vertices in the mesh are considered for merging.
        - Useful for cleaning geometry after operations that may generate
        coincident vertices (e.g., mirroring, joining, or extrusion).

        Examples
        --------
        Remove doubles with default threshold:

        ```python
        mesh.remove_doubles()
        ```

        Remove doubles with a larger threshold:

        ```python
        mesh.remove_doubles(dist=0.01)
        ```

        > ***Note:*** This method modifies the mesh **in place** and returns `self`
        > for chaining.
        """
        with self.bmesh() as bm:
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)

        return self

    # ----------------------------------------------------------------------------------------------------
    # Triangulate
    # ----------------------------------------------------------------------------------------------------

    def triangulate(self, selection=None):
        """
        Triangulate selected faces (or all faces) and return a new mesh.

        Parameters
        ----------
        selection : array-like of int or bool, or None, optional
            Indices (or mask) of faces to triangulate. If `None`, all faces are
            triangulated. Default is None.

        Returns
        -------
        Mesh or None
            A new mesh instance with the selected faces triangulated.
            Returns `None` if no faces were selected.

        Notes
        -----
        - Creates a copy of the current mesh with [`Mesh.from_mesh`][npblender.Mesh.from_mesh].
        - Triangulation is applied in-place on the copy via
        [`bmesh.ops.triangulate`](https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.triangulate).
        - The original mesh is left unchanged.

        Examples
        --------
        Triangulate all faces:

        ```python
        tri_mesh = mesh.triangulate()
        ```

        Triangulate only a subset:

        ```python
        tri_mesh = mesh.triangulate(selection=[0, 5, 7])
        ```

        See Also
        --------
        [`Mesh.from_mesh`][npblender.Mesh.from_mesh] :
            Utility to duplicate the mesh before applying triangulation.

        > ***Note:*** If `selection` is empty, the method returns `None`.
        """
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

    def simplified(self, scale, dist=0.001):
        """
        Return a simplified copy of the mesh by merging close vertices,
        with a fallback to a cubic envelope if the result is too small.

        Parameters
        ----------
        scale : float
            Scale factor applied to the distance threshold.
        dist : float, optional
            Base merge distance for vertices. The effective threshold is
            `dist / scale`. Default is 0.001.

        Returns
        -------
        Mesh
            A simplified copy of the mesh. If the simplification produces
            fewer than 8 vertices, returns a cubic envelope instead.

        Notes
        -----
        - The copy is created with [`Mesh.from_mesh`][npblender.Mesh.from_mesh].
        - Duplicate vertices are merged with [`remove_doubles`][npblender.Mesh.remove_doubles].
        - If too few vertices remain, a fallback is generated using
        [`get_cubic_envelop`][npblender.Mesh.get_cubic_envelop].

        Examples
        --------
        Simplify a mesh with scale factor 10:

        ```python
        simp = mesh.simplified(scale=10, dist=0.002)
        ```

        See Also
        --------
        [`remove_doubles`][npblender.Mesh.remove_doubles] :
            Merges vertices within a distance threshold.
        [`get_cubic_envelop`][npblender.Mesh.get_cubic_envelop] :
            Provides a fallback cubic mesh when simplification collapses geometry.
        """
        copy = Mesh.from_mesh(self)
        copy.remove_doubles(dist=dist/scale)

        if len(copy.points) < 8:
            copy = self.get_cubic_envelop()

        return copy
    
    # ----------------------------------------------------------------------------------------------------
    # Faces to islands
    # ----------------------------------------------------------------------------------------------------

    def separate_faces(self, groups=None):
        """
        Split faces into isolated islands, either one per face or grouped by
        provided IDs.

        Parameters
        ----------
        groups : array-like of int, shape (n_faces,), optional
            Group IDs for each face. If `None`, each face is isolated as its own
            island. If provided, must be the same length as the number of faces.

        Returns
        -------
        Mesh
            A new mesh where faces are separated into independent islands with
            duplicated vertices.

        Raises
        ------
        ValueError
            If `groups` is provided but its shape does not match `(n_faces,)`.

        Notes
        -----
        - When `groups` is `None`, the output mesh has one disconnected island per face.
        - When grouping, faces sharing the same group ID are kept together in the
        same island, with vertices duplicated so that each group is independent.
        - Face attributes are preserved except for `'loop_total'` and `'loop_start'`.

        Examples
        --------
        Separate every face:

        ```python
        islands = mesh.separate_faces()
        ```

        Separate faces into two groups:

        ```python
        groups = np.array([0, 0, 1, 1, 1, 0])  # one group ID per face
        split = mesh.separate_faces(groups=groups)
        ```

        See Also
        --------
        [`join_geometry`][npblender.Mesh.join_geometry] :
            Utility to assemble new meshes from points, corners, faces, and attributes.
        [`join`][npblender.Mesh.join] :
            Used internally to accumulate separated islands.

        > ***Caution:*** The number of groups must equal the number of faces in
        > the mesh, otherwise a `ValueError` is raised.
        """
        mesh = Mesh(materials=self.materials)
        attr_names = [name for name in self.faces.actual_names if name not in ['loop_total', 'loop_start']]

        # ---------------------------------------------------------------------------
        # No group: each face becomes an island
        # ---------------------------------------------------------------------------

        if groups is None:
            attrs = {name: self.faces[name] for name in attr_names}
            return Mesh(attr_from=self).join_geometry(
                points = self.points.position[self.corners.vertex_index],
                corners = np.arange(len(self.corners)),
                faces = self.faces.loop_total,
                **attrs,
            )

        # ---------------------------------------------------------------------------
        # Faces are grouped with groupds IDs
        # ---------------------------------------------------------------------------

        groups = np.asarray(groups)
        if groups.shape != (len(self.faces),)   :
            raise ValueError(f"The 'groups' argument must be a index per face with a length of {len(self.faces)}.")

        ugroups, rev_index = np.unique(groups, return_inverse=True)
        for group in ugroups:
            faces = self.faces[ugroups[rev_index] == group]
            attrs = {name: faces[name] for name in attr_names}

            corners = self.corners[faces.loop_index]
            uniques, new_corners = np.unique(corners.vertex_index, return_inverse=True)
            mesh.join(Mesh(attr_from=self).join_geometry(
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
        """
        Construct the **dual mesh**: one vertex per original face, and one face per
        original vertex (linking adjacent face-centers around that vertex).

        Parameters
        ----------
        center : {'median', 'bounds', 'weighted'}, optional
            Method to compute the position of each dual vertex (i.e., the center
            of the corresponding original face):
            - `'median'`: face median center (`BMFace.calc_center_median`).
            - `'bounds'`: face bounds center (`BMFace.calc_center_bounds`).
            - `'weighted'`: area-weighted center (`BMFace.calc_center_median_weighted`).
            Default is `'median'`.

        Returns
        -------
        Mesh
            The dual mesh, where:
            - points = centers of original faces,
            - faces  = polygons formed by chaining the adjacent original faces
            around each original vertex.

        Raises
        ------
        ValueError
            If `center` is not one of `{'median', 'bounds', 'weighted'}`.

        Notes
        -----
        - For each original face `f`, a dual vertex is computed using the chosen
        center method and stored at index `f.index`.
        - For each original vertex `v`, its incident faces are **ordered** by
        walking across `v`’s incident edges (each with exactly two linked faces)
        to form a cyclic sequence of face indices; this ordered loop becomes a
        polygon in the dual.
        - Non-manifold or boundary configurations (edges with a number of linked
        faces different from 2) are skipped for that vertex; no dual face is
        created in such cases.

        Examples
        --------
        Build the dual using area-weighted face centers:

        ```python
        d = mesh.dual(center='weighted')
        ```

        Build the dual with bounds centers:

        ```python
        d = mesh.dual(center='bounds')
        ```

        See Also
        --------
        [`triangulate`][npblender.Mesh.triangulate] :
            Triangulation can improve robustness before dualization.
        [`remove_doubles`][npblender.Mesh.remove_doubles] :
            Helpful for cleaning geometry prior to constructing the dual.

        > ***Caution:*** On meshes with boundaries or non-manifold edges, some
        > vertices may not yield a valid cyclic ordering of adjacent faces; those
        > dual faces are omitted.

        > ***Note:*** Dualization does not, in general, invert perfectly (i.e.,
        > the dual of the dual is not guaranteed to reproduce the original mesh),
        > especially in the presence of boundaries or irregular valences.
        """

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
    # Symatric
    # ----------------------------------------------------------------------------------------------------

    def symmetrical(self, x=-1., y=1., z=1., flip=True):
        """
        Construct the symmetrical mesh, flip the faces if requested.

        Parameters
        ----------
        x : float, optional
            x multiplicator. 
            Default is -1.0.
        y : float, optional
            y multiplicator. 
            Default is 1.0.
        z : float, optional
            z multiplicator. 
            Default is 1.0.

        flip : bool, optional
            flip the faces (invert the normals).
            Default is `'median'`.

        Returns
        -------
        Mesh
            The symmetrical mesh.
        """ 
        mesh = Mesh.from_mesh(self)
        center = np.mean(self.points.position, axis=0)
        mesh.points.position -= center
        mesh.points.position *= (x, y, z)
        mesh.points.position += center

        if flip:
            mesh.faces.flip(mesh.corners)
            
        return mesh

    # ----------------------------------------------------------------------------------------------------
    # Faces neighbors
    # ----------------------------------------------------------------------------------------------------

    def faces_neighbors(self):
        """
        Compute the neighboring faces for each face, defined as faces sharing
        at least one edge.

        Returns
        -------
        list of list of int
            For each face (by index), a list of indices of adjacent faces.

        Notes
        -----
        - Each face’s neighbors are determined by scanning its incident edges
        and collecting the two faces linked to each edge.
        - The current face index is excluded from its own neighbor list.
        - Non-manifold edges (with more or fewer than two linked faces) are not
        expected; if present, results may be incomplete or inconsistent.

        Examples
        --------
        Get adjacency information for all faces:

        ```python
        neighbors = mesh.faces_neighbors()
        for i, ns in enumerate(neighbors):
            print(f"Face {i} neighbors: {ns}")
        ```

        > ***Note:*** The output is a Python list of lists (not a NumPy array).
        """

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
    
    # ----------------------------------------------------------------------------------------------------
    # Get islands
    # ----------------------------------------------------------------------------------------------------

    def get_islands(self):
        """
        Compute connected components of faces (islands) and assign an island ID
        to each face.

        Returns
        -------
        ndarray of int, shape (n_faces,)
            Array of island IDs, one per face. Faces in the same connected
            component share the same integer ID. Empty mesh returns an empty list.

        Notes
        -----
        - Islands are defined as groups of faces connected through shared edges.
        - A breadth-first search (BFS) is used to traverse each connected component.
        - IDs are assigned sequentially starting from 0.

        Examples
        --------
        Get island IDs for all faces:

        ```python
        ids = mesh.get_islands()
        print("Unique islands:", np.unique(ids))
        ```

        Map faces by island:

        ```python
        ids = mesh.get_islands()
        for island_id in np.unique(ids):
            faces = np.where(ids == island_id)[0]
            print(f"Island {island_id}: faces {faces}")
        ```

        > ***Note:*** Non-manifold meshes are still handled, but faces that share
        > only a vertex (not an edge) are considered separate islands.
        """

        from collections import deque

        nfaces = len(self.faces)
        if not nfaces:
            return []
        
        islands = np.full(nfaces, -1, dtype=np.int32)
        cur_island = -1

        passed = np.zeros(nfaces, dtype=bool)

        with self.bmesh() as bm:
            bm.faces.ensure_lookup_table()
            bm.edges.ensure_lookup_table()

            for f in bm.faces:
                # Already visited
                if islands[f.index] >= 0:
                    continue

                # New island index
                cur_island += 1

                q = deque([f])
                while q:
                    cur = q.popleft()

                    # Part of the current island
                    islands[cur.index] = cur_island

                    # No infinite loop
                    passed[cur.index] = True

                    # Loop on the edges
                    for e in cur.edges:                                               
                        # Loop on the edge faces
                        for nb in e.link_faces:
                            if passed[nb.index]:
                                continue
                            passed[nb.index] = True

                            if islands[nb.index] < 0:
                                q.append(nb)

        return islands
    
    def elongation(self, axis='X', size=0.0, mode='CENTER', margin=0., smooth='SMOOTH'):
        """
        Elongate the mesh in the plane xy.

        The elongation is performed only if the mesh is smaller than the
        size passed in argument.

        The elongation consists in shifting the points close to the border
        while keeping the center unchanged.

        This is used, for instance, to make taller a character such as '{'.

        Parameters
        ----------
        axis : str, optional
            Axis to elongate along. 
            Default is 'X'.
        size : float, optional
            Target size. 
            Default is 0.0.
        mode : str, optional
            Elongation mode in ('CENTER', 'LEFT', 'RIGHT', 'BOT', 'TOP', 'SCALE').
            Default is 'CENTER'.
        margin : float in [0.0, 0.5], optional
            Border margin vertices are just shifted without scale.
            The margin is expressed as a ratio.
            Default is 0.0
        smooth : str, optional
            Easing mode for marrange.
            Default is 'SMOOTH'.
        """

        from .maths import maprange

        margin = np.clip(margin, 0.0, 0.5)
        axis = axis.upper()
        mode = mode.upper()

        index = 'XYZ'.find(axis)
        if index < 0:
            return
        
        x = self.points.position[..., index]

        x0, x1 = np.min(x), np.max(x)
        cur_size = x1 - x0

        # Only if current size is not enough
        if cur_size >= size:
            return
        
        # Dimensions
        border = cur_size*margin
        ds = size - cur_size
        
        # Depending on the algo
        if mode == 'CENTER':
            cx = (x0 + x1)/2            
            rel_x = x - cx

            dx = maprange(np.abs(rel_x), border, cur_size/2 - border, 0.0, ds/2, mode=smooth) * np.sign(rel_x)

            self.points.position[..., index] += dx

        elif mode in ['BOT', 'BOTTOM', 'LEFT']:
            dx = np.zeros_like(x)
            dx[x <= x0 + border] = -ds

            self.points.position[..., index] += dx

        elif mode in ['TOP', 'RIGHT']:
            dx = np.zeros_like(x)
            dx[x >= x1 - border] = ds

            self.points.position[..., index] += dx

        elif mode == 'SCALE':

            sc = x0 + (x - x0)*(size/cur_size)

            self.points.position[..., index] =  x0 + x*sc

        else:
            raise ValueError("Unknown elongation mode: '{mode}'")




    
    # ====================================================================================================
    # BVHTree
    # ====================================================================================================

    def bvh_tree(self, count=None):
        """
        Build a Blender BVH tree for fast spatial queries (ray casting, overlap,
        nearest point, etc.).

        Parameters
        ----------
        count : int, optional
            If `None`, build a single BVH tree for the whole mesh.
            If an integer `count` is provided, the mesh is assumed to represent
            a batch of `count` sub-meshes laid out in a structured array, and a
            list of BVH trees (one per sub-mesh) is returned.

        Returns
        -------
        BVHTree or list of BVHTree
            - If `count` is `None`, a single `BVHTree` instance built from the
            current mesh.
            - If `count` is an integer, a list of `BVHTree` objects, one for each
            sub-mesh.

        Notes
        -----
        - Internally uses
        [`mathutils.bvhtree.BVHTree.FromPolygons`](https://docs.blender.org/api/current/mathutils.bvhtree.html#mathutils.bvhtree.BVHTree.FromPolygons).
        - When `count` is given, vertices are reshaped to `(count, n, 3)` and faces
        are assumed to be identical across all sub-meshes.
        - `epsilon=0.0` is used for exact geometry.

        Examples
        --------
        Build a single BVH tree:

        ```python
        tree = mesh.bvh_tree()
        loc, normal, index, dist = tree.ray_cast((0, 0, 10), (0, 0, -1))
        ```

        Build multiple BVH trees for a batch of 5 sub-meshes:

        ```python
        trees = mesh.bvh_tree(count=5)
        for t in trees:
            print(t.find_nearest((0, 0, 0)))
        ```

        > ***Caution:*** When `count` is provided, the mesh must be structured
        > consistently: faces are taken from the first sub-mesh and reused for all.
        """

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
    
