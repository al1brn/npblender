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
Module Name: geometry
Author: Alain Bernard
Version: 0.1.0
Created: 2023-11-10
Last updated: 2025-08-31

Summary:
    Root class for actual Geometries.

Usage example:
    >>> geo = Geometry.from_dict(d)
"""

from contextlib import contextmanager
import numpy as np

from . import blender

# ----------------------------------------------------------------------------------------------------
# Root class for geometries
# ----------------------------------------------------------------------------------------------------

class Geometry:
    """
    Base class for concrete geometries.

    `Geometry` defines common behaviors shared by actual geometries such as [Mesh][npblender.mesh.Mesh]
    or [Curve][npblender.curve.Curve] : attribute propagation across domains, Blender I/O helpers, simple
    transforms, and material bookkeeping. Concrete subclasses override
    `domain_names` and implement domain-specific logic.

    Attributes
    ----------
    domain_names : list[str]
        Names of available domains in the concrete geometry. Overridden by
        subclasses (e.g., `["points", "corners", "faces", "edges"]` for Mesh).

    Notes
    -----
    - Subclasses are expected to provide the domains listed in `domain_names`
      as attributes (e.g., `self.points`, `self.faces`, ...).
    - Blender interoperability helpers (`load_object`, `load_models`, and
      context managers) rely on the presence of `to_object` / `from_object`
      implemented by subclasses.
    """

    # Overriden by Mesh and Curve
    domain_names = ["points"]

    # ====================================================================================================
    # From a dict
    # ====================================================================================================

    @classmethod
    def from_dict(cls, d):
        """
        Construct a geometry from a serialized payload.

        Dispatches to the appropriate subclass based on `d["geometry"]`
        (supported: `"Mesh"`, `"Curve"`, `"Cloud"`, `"Instances"`, `"Meshes"`).

        Parameters
        ----------
        d : dict
            Serialized geometry dictionary as produced by `to_dict()` of the
            corresponding subclass.

        Returns
        -------
        Geometry
            An instance of the appropriate subclass.

        Raises
        ------
        ValueError
            If `d["geometry"]` is unknown.
        """

        from .mesh import Mesh
        from .curve import Curve
        from .cloud import Cloud
        from .instances import Instances, Meshes

        if d['geometry'] == 'Mesh':
            return Mesh.from_dict(d)
        elif d['geometry'] == 'Curve':
            return Curve.from_dict(d)
        elif d['geometry'] == 'Cloud':
            return Cloud.from_dict(d)
        elif d['geometry'] == 'Instances':
            return Instances.from_dict(d)
        elif d['geometry'] == 'Meshes':
            return Meshes.from_dict(d)
        else:
            raise ValueError(f"Unknown geometry {d['geometry']}")

    # ====================================================================================================
    # Copy attributes definition
    # ====================================================================================================

    def join_attributes(self, other, **kwargs):
        """
        Merge attribute schemas from another geometry.

        For each domain listed in `self.domain_names` and also present in `other`,
        copies (joins) the attribute *definitions* (names, dtypes, metadata) from
        `other` into this geometry's domains. Use keyword flags to include/exclude
        domains by name (e.g., `faces=False`).

        Parameters
        ----------
        other : Geometry or None
            Source geometry. If `None`, does nothing and returns `self`.
        **kwargs
            Per-domain boolean switches to filter which domains to join.

        Returns
        -------
        Geometry
            `self`, for chaining.

        Examples
        --------
        ``` python
        mesh.join_attributes(other_mesh, faces=False)
        curve.join_attributes(mesh)  # only common domains are merged
        ```
        """
        if other is None:
            return self

        for name in self.domain_names:
            if name in other.domain_names and kwargs.get(name, True):
                getattr(self, name).join_fields(getattr(other, name))

        return self
    
    # ====================================================================================================
    # Compute an attribute of a domain to another domain
    # ====================================================================================================

    def compute_attribute_on_domain(self, domain_from, attr, domain_to):
        """
        Transfer an attribute from one domain to another.

        Performs a domain mapping (e.g., points → faces) using the appropriate
        domain operator, and returns the computed array on the target domain.

        Parameters
        ----------
        domain_from : str
            Source domain name (e.g., `"points"`, `"faces"`, `"edges"`, `"corners"`, `"splines"`).
        attr : str or numpy.ndarray
            Source attribute to transfer. If a string, it is looked up on the
            source domain; if an array, it must match the source domain length.
        domain_to : str
            Target domain name.

        Returns
        -------
        numpy.ndarray
            Attribute values on the target domain. If `domain_from == domain_to`,
            returns `attr` unchanged.

        Raises
        ------
        AttributeError
            If either `domain_from` or `domain_to` is not a valid domain of this geometry.
        Exception
            If the requested mapping is not implemented.

        Notes
        -----
        Implemented mappings include:
        - points → faces: [`Point.compute_attribute_on_faces`](npblender.domain.Point.compute_attribute_on_faces)
        - points → edges: [`Point.compute_attribute_on_edges`](npblender.domain.Point.compute_attribute_on_edges)
        - points → corners: [`Point.compute_attribute_on_corners`](npblender.domain.Point.compute_attribute_on_corners)
        - points → splines: [`Point.compute_attribute_on_splines`](npblender.domain.Point.compute_attribute_on_splines)
        - faces → points: [`Face.compute_attribute_on_points`](npblender.domain.Face.compute_attribute_on_points)
        - edges → points: [`Edge.compute_attribute_on_points`](npblender.domain.Edge.compute_attribute_on_points)
        - corners → points: [`Corner.compute_attribute_on_points`](npblender.domain.Corner.compute_attribute_on_points)
        """

        if domain_from == domain_to:
            return attr
        
        if (domain_from not in self.domain_names) or (domain_to not in self.domain_names):
            raise AttributeError(
                f"The geometry '{type(self).__name__}' doesn't have domains '{domain_from}' and '{domain_to}'.\n"
                f"Valid domains are: {self.domain_names}."
                )
        
        if domain_from == 'points':
            if domain_to == 'faces':
                return self.points.compute_attribute_on_faces(attr, self.corners, self.faces)
            elif domain_to == 'edges':
                return self.points.compute_attribute_on_edges(attr, self.edges)
            elif domain_to == 'corners':
                return self.points.compute_attribute_on_corners(attr, self.corners)
            elif domain_to == 'splines':
                return self.points.compute_attribute_on_splines(attr, self.splines)
            
        elif domain_from == 'faces':
            if domain_to == 'points':
                return self.faces.compute_attribute_on_points(attr, self.corners, self.points)
            elif domain_to == 'edges':
                pass
            elif domain_to == 'corners':
                pass

        elif domain_from == 'edges':
            if domain_to == 'points':
                return self.edges.compute_attribute_on_points(attr, self.points)
            elif domain_to == 'faces':
                pass
            elif domain_to == 'corners':
                pass
            
        elif domain_from == 'corners':
            if domain_to == 'points':
                return self.corners.compute_attribute_on_points(attr, self.points)
            elif domain_to == 'faces':
                pass
            elif domain_to == 'edges':
                pass
            
        elif domain_from == 'splines':
            if domain_to == 'points':
                return self.splines.compute_attribute_on_points(attr, self.points)

        raise Exception(
            f"Sorry, computing attribue from '{domain_from}' to '{domain_to}' is not implemented yet.")

    # ====================================================================================================
    # Check geometry consistency
    # ====================================================================================================

    def check(self, title="Geometry Check", halt=True):
        """
        Validate the geometry consistency.

        Placeholder in the base class: returns `True`. Subclasses may override
        to perform domain-level checks.

        Parameters
        ----------
        title : str, default="Geometry Check"
            Label for messages or errors.
        halt : bool, default=True
            Whether to raise on failure (in subclasses that implement checks).

        Returns
        -------
        bool
            Always `True` in the base class.
        """
        return True

    # ====================================================================================================
    # Load a Mesh or a Curve
    # ====================================================================================================

    @staticmethod
    def load_object(name):
        """
        Load a Blender object and return a `Mesh` or a `Curve`.

        Resolves `name` to a Blender object, inspects its data type, and returns a
        matching geometry by calling the subclass' `from_object`.

        Parameters
        ----------
        name : str or bpy.types.Object
            Object name or object instance.

        Returns
        -------
        Mesh or Curve or None
            A [`Mesh`](npblender.mesh.Mesh) or a [`Curve`](npblender.geometry.curve.Curve),
            or `None` if the object is not found.

        Raises
        ------
        Exception
            If the object exists but is neither a `bpy.types.Mesh` nor `bpy.types.Curve`.

        Examples
        --------
        ``` python
        geo = Geometry.load_object("MyObject")
        if geo is not None:
            print(type(geo).__name__)
        ```
        """

        import bpy

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
        """
        Load multiple geometries from collections, objects, or instances.

        Accepts mixed inputs such as Blender collections, Blender objects, lists/
        tuples of either, or already-instantiated `Mesh`/`Curve`. Returns a flat
        list of geometries discovered or constructed.

        This method is mainly intended to be used by [`Instances`][npblender.instances.Instances]
        to load its models.

        Parameters
        ----------
        *specs
            Collections, objects, lists/tuples, or `Mesh`/`Curve` instances.

        Returns
        -------
        list
            List of geometries (`Mesh`/`Curve`).

        Raises
        ------
        ValueError
            If a spec cannot be resolved to a geometry.
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

            # A valid geometry
            if type(spec).__name__ in ['Mesh', 'Curve']:
                models.append(spec)
                continue

            raise ValueError(f"Unknown model (type '{type(spec)}'): {spec}")
        
        return models

    # ====================================================================================================
    # Object edition
    # ====================================================================================================

    @contextmanager
    def object(self, index=0, readonly=True, **kwargs):
        """
        Temporary access to a Blender Object built from this geometry.

        Creates a transient object (named `"BPBL Temp {index}"` unless `index` is a
        string), selects and activates it, yields it for editing, then cleans up.
        If `readonly=True`, the edited object is captured back into `self`.

        This method can be used to set and apply a modifier (see exemple below).

        Parameters
        ----------
        index : int or str, default=0
            Index or name used to label the temporary object.
        readonly : bool, default=True
            If `False`, re-capture the possibly edited object back into this geometry.
        kwargs : dict, optional
            Keyword arguments passed to `self.to_object`.

        Yields
        ------
        bpy.types.Object
            The temporary Blender object built from `self`.

        Examples
        --------
        ``` python
        plane = Mesh.Grid()
        with plane.object(readonly=False) as obj:
            mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
            mod.thickness = .1
            bpy.ops.object.modifier_apply(modifier=mod.name)

        # plane is now solidifed
        ```
        """

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

        obj = self.to_object(temp_name, **kwargs)
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
        """
        Return the index of a material name, creating it if needed.

        Parameters
        ----------
        mat_name : str
            Material name to look up or append.

        Returns
        -------
        int
            Index of `mat_name` in `self.materials`.

        Notes
        -----
        If `mat_name` is not present, it is appended to `self.materials` and the
        new index is returned.
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
        """
        Append material name(s) to the geometry.

        Parameters
        ----------
        materials : str or sequence of str
            One name or a sequence of names to append.

        Returns
        -------
        None

        Notes
        -----
        This method does not deduplicate names; duplicates may be appended.
        """
        if isinstance(materials, str):
            self.materials.append(materials)
        else:
            self.materials.extend(materials)

    # ====================================================================================================
    # Transformation
    # ====================================================================================================

    def get_points_selection(self):
        """
        Selection of points relevant to operations.

        Returns `slice(None)` in the base class (all points). Subclasses (e.g.,
        curves) may override to select only referenced points.

        Returns
        -------
        slice
            `slice(None)` by default.
        """
        return slice(None)
    
    # ----------------------------------------------------------------------------------------------------
    # Apply one or more basic transformation
    # ----------------------------------------------------------------------------------------------------
    
    def transformation(self, rotation=None, scale=None, translation=None, pivot=None):
        """
        Apply rotation/scale/translation (with optional per-packet broadcasting).

        Operates in-place on `points.position` and, when present, Bezier handles
        (`points.handle_left`, `points.handle_right`). Shapes can represent packets
        of points: broadcasting rules are handled by
        [`Point._get_shape_for_operation`](npblender.domain.Point._get_shape_for_operation).

        Parameters
        ----------
        rotation : ndarray or Rotation-like, optional
            Rotation matrix/matrices applied as `R @ v`. Shape may broadcast over
            points (see notes).
        scale : ndarray, optional
            Per-axis scaling. Shape may broadcast over points.
        translation : ndarray, optional
            Per-point translation. Shape may broadcast over points.
        pivot : ndarray, optional
            Pivot(s) subtracted before, and added after, the rotation/scale; same
            broadcasting rules as `scale`/`translation`.

        Returns
        -------
        Geometry
            `self`, for chaining.

        Notes
        -----
        - If handles exist, they are transformed consistently with positions.

        Examples
        --------
        ``` python
        # 12 cubes laid out randomly with per-instance transforms
        cubes = Mesh.cube(size=1).multiply(12)
        T = np.random.uniform(-1, 1, (12, 3))
        S = np.random.uniform(0.5, 2.0, (12, 3))
        R = Rotation.from_euler(np.random.uniform(0, 2*np.pi, (12, 3)))
        cubes.transformation(rotation=R, scale=S, translation=T)
        ```
        """

        # Curve splines can be a subset of the points 
        pts_sel = self.get_points_selection()
        pos = self.points.position[pts_sel]
        npoints = len(pos)

        # ---------------------------------------------------------------------------
        # The list of all attributes to transform
        # ---------------------------------------------------------------------------

        all_vecs = [pos]

        has_handles = "handle_left" in self.points.actual_names
        if has_handles:
            left = self.points.handle_left[pts_sel]
            right = self.points.handle_right[pts_sel]
            all_vecs.extend([left, right])

        # ---------------------------------------------------------------------------
        # Initial pivot
        # ---------------------------------------------------------------------------

        if pivot is not None:
            pivot = np.asarray(pivot)
            if True:
                pv_shape0, pv_shape1 = self.points._get_shape_for_operation(pivot.shape[:-1], title="Pivot")
                pv = np.reshape(pivot, pv_shape1 + (3,))
                for v in all_vecs:
                    v.reshape(pv_shape0)[:] -= pv
            else:
                pivot_shape = self._check_transformation_shape(pivot.shape[:-1], npoints, label="Pivot")
                for v in all_vecs:
                    v.reshape(pivot_shape)[:] -= pivot

        # ---------------------------------------------------------------------------
        # Scale and rotation
        # ---------------------------------------------------------------------------

        # Scale
        if scale is not None:
            scale = np.asarray(scale)
            if True:
                shape, op_shape = self.points._get_shape_for_operation(scale.shape[:-1], title="Scale")
                sc = np.reshape(scale, op_shape + (3,))
                for v in all_vecs:
                    v.reshape(shape)[:] *= sc
            else:
                scale_shape = self._check_transformation_shape(scale.shape[:-1], npoints, label="Scale")
                for v in all_vecs:
                    v.reshape(scale_shape)[:] *= scale
                
        # Rotation
        if rotation is not None:
            if True:
                shape, op_shape = self.points._get_shape_for_operation(rotation.shape, title="Rotation")
                rot = rotation.reshape(op_shape)
                for v in all_vecs:
                    v.reshape(shape)[:] = rot @ v.reshape(shape)
            else:
                rot_shape = self._check_transformation_shape(rotation.shape, npoints, label="Rotation")
                for v in all_vecs:
                    v.reshape(rot_shape)[:] = rotation @ v.reshape(rot_shape)

        # ---------------------------------------------------------------------------
        # Pivot back
        # ---------------------------------------------------------------------------

        if pivot is not None:
            if True:
                for v in all_vecs:
                    v.reshape(pv_shape0)[:] += pv
            else:
                for v in all_vecs:
                    v.reshape(pivot_shape)[:] += pivot

        # ---------------------------------------------------------------------------
        # Translation
        # ---------------------------------------------------------------------------

        if translation is not None:
            translation = np.asarray(translation)
            if True:
                shape, op_shape = self.points._get_shape_for_operation(translation.shape[:-1], title="Scale")
                tr = np.reshape(translation, op_shape + (3,))
                for v in all_vecs:
                    v.reshape(shape)[:] += tr
            else:
                tr_shape = self._check_transformation_shape(translation.shape[:-1], npoints, label="Pivot")
                for v in all_vecs:
                    v.reshape(tr_shape)[:] += translation

        # ---------------------------------------------------------------------------
        # Set the points with the result
        # ---------------------------------------------------------------------------

        self.points[pts_sel].position = pos
        if has_handles:
            self.points[pts_sel].handle_left = all_vecs[1]
            self.points[pts_sel].handle_right = all_vecs[2]

        return self

    def translate(self, translation):
        """
        Translate points (convenience wrapper).

        ***See:*** [`transformation`][npblender.geometry.Geometry.transformation]

        Parameters
        ----------
        translation : ndarray
            Per-point or broadcastable translation vectors.

        Returns
        -------
        Geometry
            `self`.
        """
        return self.transformation(translation=translation)

    def apply_scale(self, scale, pivot=None):
        """
        Scale points (convenience wrapper).

        ***See:*** [`transformation`][npblender.geometry.Geometry.transformation]

        Parameters
        ----------
        scale : ndarray
            Per-point or broadcastable scales.
        pivot : ndarray, optional
            Optional pivot(s) for scaling.

        Returns
        -------
        Geometry
            `self`.
        """
        return self.transformation(scale=scale, pivot=pivot)
    
    def rotate(self, rotation, pivot=None):
        """
        Rotate points (convenience wrapper).

        ***See:*** [`transformation`][npblender.geometry.Geometry.transformation]

        Parameters
        ----------
        rotation : ndarray or Rotation-like
            Rotation(s) to apply as `R @ v`.
        pivot : ndarray, optional
            Optional pivot(s) for rotation.

        Returns
        -------
        Geometry
            `self`.
        """
        return self.transformation(rotation=rotation, pivot=pivot)
    
    def transform(self, transformation):
        """
        Apply a rotation matrix or batch of matrices.

        ***See:*** [`transformation`][npblender.geometry.Geometry.transformation]

        Parameters
        ----------
        transformation : ndarray
            Rotation matrix or batch of rotation matrices.

        Returns
        -------
        Geometry
            `self`.
        """
        return self.transformation(rotation=transformation)
    
    # ====================================================================================================
    # Envelop
    # ====================================================================================================

    @property
    def bounding_box(self):
        """
        Axis-aligned bounding box of the point positions.

        Returns
        -------
        tuple of numpy.ndarray
            `(min_xyz, max_xyz)`. If empty, returns two zero vectors.
        """
        pos = self.points.position
        if len(pos):
            return np.min(pos, axis=0), np.max(pos, axis=0)
        else:
            return np.zeros(3, float), np.zeros(3, float)
        
    @property
    def bounding_box_dims(self):
        """
        Extents of the axis-aligned bounding box.

        Returns
        -------
        numpy.ndarray of shape (3,)
            `max_xyz - min_xyz`.
        """

        v0, v1 = self.bounding_box
        return v1 - v0
        
    @property
    def max_size(self):
        """
        Maximum dimension of the bounding box.

        Returns
        -------
        float
            `max(bounding_box_dims)`.
        """
        return max(self.bounding_box_dims)
    
    def get_cubic_envelop(self):
        """
        Return a cube mesh that encloses the geometry’s bounding box.

        Uses the bounding box dimensions to build a cube via
        [`Mesh.cube`](npblender.mesh.Mesh.cube), forwarding this geometry’s
        `materials` if present.

        Returns
        -------
        Mesh
            A cube mesh sized to the bounding box.
        """

        from .mesh import Mesh

        size = self.bounding_box_dims
        return Mesh.cube(size=size, materials=getattr(self, "materials", None))
        







