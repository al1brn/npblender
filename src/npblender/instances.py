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
from . domain import Point

DATA_TEMP_NAME = "npblender_TEMP"

# ====================================================================================================
# Instances
# ====================================================================================================

class Instances(Geometry):
    """
    Instance container for duplicating geometries.

    `Instances` manages a set of instance transforms (positions, optional scale
    and rotation) and a list of source models (meshes and/or curves). It can
    realize instances into concrete geometries or directly create Blender
    objects.

    Attributes
    ----------
    points : [Point][npblender.Point]
        Per-instance attributes (e.g., `position`, optional `rotation`, `scale`,
        `model_index`, plus any user fields).
    models : list[[Geometry][npblender.Geometry]]
        List of source models to instance (e.g., [Mesh][npblender.Mesh],
        [Curve][npblender.geometry.curve.Curve]).
    low_resols : list[dict]
        Optional Level-of-Detail (LOD) entries, each as
        `{"dist": float, "models": list[Geometry]}`.

    Notes
    -----
    - Each instance chooses its model via `points.model_index`.
    """

    def __init__(self, points=None, models=None, model_index=None, attr_from=None, **attributes):
        """
        Create a new instance set.

        Initializes the per-instance point domain and captures model geometries.

        Parameters
        ----------
        points : array-like of shape (N, 3) or None, optional
            Instance locations appended as `points.position`.
        models : Geometry or sequence of Geometry or None, optional
            Model(s) to instance. If `None`, starts empty.
        model_index : int or array-like of int or None, optional
            Model index per instance (broadcast rules apply).
        attr_from : Geometry or None, optional
            Source geometry from which to join attribute schemas.
        **attributes
            Additional per-instance attributes to append (e.g., `rotation`, `scale`).

        Notes
        -----
        - Models are loaded via [`Geometry.load_models`][npblender.Geometry.load_models].
        - `low_resols` starts empty and can be populated with
        [`add_low_resol`](npblender.Instances.add_low_resol).
        """

        self.points  = Point()
        self.join_attributes(attr_from)

        if models is None:
            self.models = []
        else:
            self.models = self.load_models(models)
        self.low_resols = []

        self.points.append(position=points, model_index=model_index, **attributes)

    def check(self, title="Instances Check", halt=True):
        """
        Validate model indices against the models list.

        Ensures that `max(points.model_index) < len(models)`.

        Parameters
        ----------
        title : str, default="Instances Check"
            Prefix for diagnostic messages.
        halt : bool, default=True
            If True, raise on failure; otherwise print and return `False`.

        Returns
        -------
        bool
            `True` if valid (or no instances), `False` when invalid and `halt=False`.

        Raises
        ------
        AssertionError
            If a model index is out of range and `halt=True`.
        """

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
        """
        Serialize the instances to a plain dictionary.

        Returns
        -------
        dict
            Keys:
            - ``"geometry"`` = ``"Instances"``
            - ``"points"``   : point-domain payload
            - ``"models"``   : list of serialized models
            - ``"low_resols"`` : list of serialized LOD levels
        """
        return {
            'geometry':   'Instances',
            'points':     self.points.to_dict(),
            'models':     [model.to_dict() for model in self.models],
            'low_resols': [[lr.to_dict() for lr in low_resol] for low_resol in self.low_resols]
            }

    @classmethod
    def from_dict(cls, d):
        """
        Deserialize an `Instances` object produced by `to_dict`.

        Parameters
        ----------
        d : dict
            Serialized payload with keys ``"points"``, ``"models"``, ``"low_resols"``.

        Returns
        -------
        Instances
            New instance with points, models and LODs reconstructed.
        """

        insts = cls()
        insts.points     = Point.from_dict(d['points'])
        insts.models     = [Geometry.from_dict(model) for model in d['models']]
        insts.low_resols = [[Geometry.from_dict(lrd) for lrd in low_resol_dict] for low_resol_dict in d['low_resols']]

        return insts    
    
    # ====================================================================================================
    # Low scale models
    # ====================================================================================================

    def add_low_resol(self, dist, models):
        """
        Register a Level-of-Detail (LOD) set.

        Associates a view distance threshold with a list of low-res models
        (same length and order as `self.models`).

        Parameters
        ----------
        dist : float
            Distance threshold at which this LOD should be used.
        models : Geometry or sequence of Geometry
            One low-res model per source model.

        Raises
        ------
        ValueError
            If the number of LOD models does not match `len(self.models)`.

        Returns
        -------
        None
        """
        ls_models = self.load_models(models)
        if len(ls_models) != len(self.models):
            raise ValueError(
                f"Instances.add_low_resol: the number of low scale models ({len(ls_models)}) "
                f"is not equal to the number of models ({len(self.models)})")
        
        self.low_resols.append({"dist": dist, "models": ls_models})

    def compute_low_resols(self, start_scale=.1, scale=.8, detail=1.):
        """
        Auto-compute a LOD pyramid from current `models`.

        Uses a camera model to estimate view distances for a target on-screen scale.
        For meshes, generates simplified copies; for curves, converts to curve views.

        Parameters
        ----------
        start_scale : float, default=0.1
            Initial relative on-screen scale.
        scale : float, default=0.8
            Multiplicative factor between consecutive LOD levels (clipped to 0.01–0.99).
        detail : float, default=1.0
            Detail factor forwarded to simplification (mesh-dependent).

        Returns
        -------
        None

        Notes
        -----
        - Up to 10 levels are created, stopping when the max vertex count drops to ≤ 8.
        - Each level is recorded via [`add_low_resol`](npblender.Instances.add_low_resol).
        """

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
        """
        Realize instances into concrete geometries.

        Duplicates each model for its selected instances, applies per-instance
        transform (translation, optional rotation/scale), and accumulates results
        into a [Mesh][npblender.Mesh] and/or a
        [Curve][npblender.geometry.curve.Curve]. With `camera_culling=True`,
        hidden instances are skipped and LODs may be used.

        Parameters
        ----------
        camera_culling : bool or object, default=False
            If truthy, perform visibility tests and LOD selection using a camera.

        Returns
        -------
        dict
            A dictionary with keys:
            - ``"mesh"`` : a [Mesh][npblender.Mesh] or `None`
            - ``"curve"``: a [Curve][npblender.geometry.curve.Curve] or `None`

        Raises
        ------
        TypeError
            If a model type is not supported (neither Mesh nor Curve).

        Examples
        --------
        ``` python
        insts = Instances(points=np.random.randn(100, 3), models=[Mesh.cube(), Curve.circle()])
        geos = insts.realize(camera_culling=True)
        if geos["mesh"] is not None:
            geos["mesh"].to_object("InstancedMesh")
        ```
        """

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
                scale = pts.scale[:, None]
            else:
                scale = None

            if self.points.has_rotation:
                rot = pts.rotation[:, None]
            else:
                rot = None

            geo.transformation(rotation=rot, scale=scale, translation=pts.position[:, None])

            # --------------------------------------------------
            # Add to realized geometry
            # --------------------------------------------------

            if type(model).__name__ == 'Mesh':
                mesh.join(geo)

            elif type(model).__name__ == 'Curve':
                curve.join(geo)

            else:
                raise TypeError(f"Instances.realize> Unsupported model type: '{type(model)}'")

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
        """
        Create Blender object(s) from realized instances.

        Realizes instances, converts curves to mesh when a profile is provided (or
        when culling requires meshing), and creates one or two Blender objects.

        Parameters
        ----------
        name : str
            Base name for created objects.
        profile : Curve or None, optional
            Profile to sweep along curve outputs (see
            [`Curve.to_mesh`][npblender.geometry.curve.Curve.to_mesh]).
        caps : bool, default=True
            Close ends when sweeping.
        use_radius : bool, default=True
            Use per-point radius when sweeping.
        shade_smooth : bool, default=True
            Smooth shading for the mesh object.
        camera_culling : bool, default=False
            If True, perform visibility tests and LOD selection.

        Returns
        -------
        dict
            Possibly contains:
            - ``"mesh"`` : the created mesh object (with optional “ - (M)” suffix)
            - ``"curve"``: the created curve object (with “ - (C)” suffix if both exist)
        """

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
        """
        Dump all models (and their LODs) into a single Blender mesh object.

        Places each source model along +X with its LOD stack above it along +Z,
        then creates a single object with flat shading.

        Parameters
        ----------
        name : str, default="Models"
            Object name.

        Returns
        -------
        bpy.types.Object
            The created object.

        Examples
        --------
        ``` python
        obj = insts.models_to_object("AllModelsPreview")
        ```
        """

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
        """
        Concatenate other instance sets into this one.

        Appends models and per-instance points; newly appended `model_index` values
        are offset by the previous model count.

        Parameters
        ----------
        *others : Instances
            Other `Instances` objects to append.

        Returns
        -------
        Instances
            `self`, for chaining.

        Raises
        ------
        AttributeError
            If any argument is not an `Instances`.
        """

        for other in others:

            if not isinstance(other, Instances):
                raise AttributeError(f"Instances can be joined with Instances only, not {type(other).__name__}.")
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
        """
        Duplicate the instance set `count` times.

        Parameters
        ----------
        count : int
            Number of copies to create.
        in_place : bool, default=True
            If True, expand this instance; otherwise return a new expanded copy.

        Returns
        -------
        Instances or None
            `self` (in place) or a new `Instances`; `None` if `count == 0`.

        Raises
        ------
        TypeError, ValueError
            If `count` cannot be converted to `int`.
        """

        count = int(count)

        if count == 0:
            return None
        
        if count == 1:
            if in_place:
                return self
            else:
                return type(self)(attr_from=self).join(self)
            
        if not in_place:
            return type(self)(attr_from=self).join(self).multiply(count, in_place=True)
        
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
        """
        Clear all instances (keeps attribute schemas and models).

        Returns
        -------
        None
        """
        self.points.clear()

# ====================================================================================================
# Meshes
# ====================================================================================================

class Meshes(Geometry):
    """
    Bucketed mesh instances.

    `Meshes` looks like `instances`, the difference being the each instance is different:
    - `Instances` : each point instance points on a model with `model_index`
    - `Meshes` : each point refers to vertices within `Mesh.mesh` attributes

    To allow handling the individual pieces in with vectorization, the pieces are
    groupded in buckets of the same size (same number of vertices).

    It is possible to iterate on pieces sharing the same size:

    ```python
    for bucket, offset in meshes:
        # bucket is an array (npieces, nverts) of vertex indices in `meshes.mesh.points`
        # offset is the first point index in `meshes.points`
        # meshes.points[offset:offset + bucket.shape[0]] represent all the bucket pieces
        pass
    ```

    Attributes
    ----------
    mesh : [Mesh][npblender.Mesh]
        The source mesh containing all vertices (possibly many concatenated copies).
    points : [Point][npblender.Point]
        Per-piece attributes. At least `position` is present (piece centroid).
    buckets : list[numpy.ndarray]
        List of arrays with shape `(n_pieces, piece_vertex_count)` indexing rows
        into `mesh.points` for each bucket size.

    Notes
    -----
    Buckets are created by grouping vertices per piece size (same vertex count),
    then stacking each contiguous group into a 2D array; `points.position`
    holds the centroid of each row and the mesh is recentered so that row-local
    vertices are around the origin. :contentReference[oaicite:1]{index=1}
    """

    def __init__(self, mesh=None, mesh_id=None, attr_from=None, **attributes):
        """
        Initialize bucketed mesh instances.

        Parameters
        ----------
        mesh : [Mesh][npblender.Mesh] or None, optional
            Mesh to be bucketized. If `None`, creates an empty container.
        mesh_id : array-like or None, optional
            Per-vertex group id used to split the mesh into pieces (passed to
            `mesh.points.make_buckets`).
        attr_from : [Geometry][npblender.Geometry] or None, optional
            Geometry to copy attribute schemas from (matching domain names).
        **attributes
            Extra per-piece attributes appended to `points`.
        """
        self.points  = Point()
        self.join_attributes(attr_from)

        self._init_buckets(mesh, mesh_id)

    # ----------------------------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------------------------

    def __str__(self):
        return f"<Meshes: {len(self)}, total vertices: {0 if self.mesh is None else len(self.mesh.points)}>"
    
    def __len__(self):
        return len(self.points)
    
    # ----------------------------------------------------------------------------------------------------
    # Loop on the buckets
    # ----------------------------------------------------------------------------------------------------

    def __iter__(self):
        offset = 0
        for bucket in self.buckets:
            yield bucket, offset
            offset += len(bucket)

    # ----------------------------------------------------------------------------------------------------
    # Init buckets
    # ----------------------------------------------------------------------------------------------------

    def _init_buckets(self, mesh, mesh_id):

        from .mesh import Mesh

        self.points.clear()
        
        if mesh is None:
            self.mesh = Mesh()
            self.buckets = []

        else:
            self.mesh = mesh

            self.buckets = self.mesh.points.make_buckets(mesh_id)
            npoints = sum([len(b) for b in self.buckets])

            # Position of each mesh
            position = np.empty((npoints, 3), dtype=bfloat)
            for bucket, offset in self:
                pos = np.average(self.mesh.points.position[bucket], axis=1)
                position[offset:offset + len(bucket)] = pos

                # Put the mesh at center (position is captured by points.position)
                self.mesh.points.position[bucket] -= pos[:, None]

            self.points.append(position=position)

    # ====================================================================================================
    # Check
    # ====================================================================================================

    def check(self, title="Meshes Check", halt=True):
        """
        Validate internal consistency between buckets and points.

        Verifies that the total number of rows across all buckets equals
        `len(points)`. When invalid, prints diagnostics and either raises
        (if `halt=True`) or returns `False`.

        Parameters
        ----------
        title : str, default="Meshes Check"
            Message prefix for diagnostics.
        halt : bool, default=True
            If True, raise via assertion on failure.

        Returns
        -------
        bool
            True when consistent; False when inconsistent and `halt=False`.

        Raises
        ------
        AssertionError
            If inconsistent and `halt=True`. :contentReference[oaicite:6]{index=6}
        """
        npoints = sum([len(b) for b in self.buckets])
        if npoints != len(self.points):
            print(title)
            print(f"The numbers of points {len(self.mesh.points)} in the mesh is different from the buckets length {npoints}")
            if halt:
                assert(False)
            return False
        return True

    # ====================================================================================================
    # Get the mesh ids
    # ====================================================================================================

    @property
    def mesh_id(self):
        """
        Per-piece identifiers aligned with the flattened bucket rows.

        Returns
        -------
        numpy.ndarray of dtype int
            Vector of ids assigning each row (piece) to its bucket index. The
            length matches the concatenation of bucket rows (see iteration order). :contentReference[oaicite:7]{index=7}
        """
        mesh_id = np.empty(len(self.mesh.points), dtype=bint)
        for index, (bucket, offset) in enumerate(self):
            mesh_id[offset:offset + len(bucket)] = index
        return mesh_id

    # ====================================================================================================
    # From meshes
    # ====================================================================================================

    @classmethod
    def from_meshes(cls, meshes):
        """
        Copy-construct a `Meshes` from another `Meshes`.

        Duplicates the source mesh, copies the bucket lists, and clones the
        per-piece point domain.

        Parameters
        ----------
        meshes : Meshes
            Source `Meshes` to copy from.

        Returns
        -------
        Meshes

        Raises
        ------
        AttributeError
            If `meshes` is not a `Meshes` instance. :contentReference[oaicite:8]{index=8}
        """

        from .mesh import Mesh

        if not isinstance(meshes, Meshes):
            raise AttributeError(f"meshes argument is not Meshes '{type(meshes).__name__}'.")

        m = cls(attr_from=meshes)
        m.mesh = Mesh.from_mesh(meshes.mesh)
        m.buckets = [np.array(b) for b in meshes.buckets]
        m.points = Point(meshes.points, mode='COPY')

        return m

    # ====================================================================================================
    # From islands
    # ====================================================================================================

    @classmethod
    def from_mesh_islands(cls, mesh):
        """
        Build `Meshes` by splitting a mesh into face islands.

        Computes face islands on `mesh`, transfers the island id to the point
        domain, then uses it as `mesh_id` to create buckets.

        Parameters
        ----------
        mesh : [Mesh][npblender.Mesh]
            Source mesh.

        Returns
        -------
        Meshes
            New instance with one piece per island. :contentReference[oaicite:9]{index=9}
        """
        face_islands = mesh.get_islands()
        islands = mesh.compute_attribute_on_domain("faces", face_islands, "points")
        return cls(mesh, mesh_id=islands)

    # ====================================================================================================
    # Serialization
    # ====================================================================================================

    def to_dict(self):
        """
        Serialize to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``"geometry" = "Meshes"``, ``"points"``, ``"mesh"``, ``"buckets"``. :contentReference[oaicite:10]{index=10}
        """
        return {
            'geometry':   'Meshes',
            'points':     self.points.to_dict(),
            'mesh':       self.mesh.to_dict(),
            'buckets':    self.buckets,
            }

    @classmethod
    def from_dict(cls, d):
        """
        Deserialize a `Meshes` produced by `to_dict`.

        Parameters
        ----------
        d : dict
            Serialized payload.

        Returns
        -------
        Meshes
            Reconstructed instance with mesh, points and buckets restored. :contentReference[oaicite:11]{index=11}
        """
        from .mesh import Mesh

        meshes = cls()
        meshes.points     = Point.from_dict(d['points'])
        meshes.mesh       = Mesh.from_dict(d['mesh'])
        meshes.buckets    = d['buckets']

        return meshes

    # ====================================================================================================
    # To mesh
    # ====================================================================================================

    def realize(self):
        """
        Realize the bucketed pieces into a concrete mesh.

        Copies the source mesh, joins per-piece point attributes (other than
        `position`) additively into the mesh per-vertex fields for each row,
        and applies per-piece transforms (rotation/scale/translation) to the
        vertices addressed by each bucket.

        Returns
        -------
        [Mesh][npblender.Mesh]
            A mesh with all pieces transformed into world placement. :contentReference[oaicite:12]{index=12}
        """

        from .mesh import Mesh

        mesh = Mesh.from_mesh(self.mesh)
        mesh.points.join_fields(self.points)

        attr_names = [name for name in self.points.actual_names if name != 'position']
        for bucket, offset in self:

            sl = slice(offset, offset + len(bucket))
            # Transfer the attrbutes
            for name in attr_names:
                mesh.points[name][bucket] += self.points[name][offset:offset + len(bucket), None]

            # Transformation
            if self.points.has_rotation:
                rot = self.points.rotation[sl]
            else:
                rot = None

            scale = self.points.get("scale")
            if scale is not None:
                print("DEBUG", self.points.shape, scale.shape, self.points.scale.shape)
                scale = scale[sl]

            transfo = Transformation.from_components(
                rotation = rot,
                scale = scale,
                translation = self.points.position[sl],
            )
            mesh.points.position[bucket] = transfo[:, None] @ mesh.points.position[bucket]

        return mesh
    
    # ====================================================================================================
    # Joining meshes
    # ====================================================================================================

    def join(self, *others):
        """
        Concatenate other `Meshes` into this one.

        Appends the other mesh data, then merges their buckets: rows with the
        same piece vertex count are concatenated; new sizes create new entries.

        Parameters
        ----------
        *others : Meshes
            Other `Meshes` objects to append.

        Returns
        -------
        Meshes
            `self`, for chaining.

        Raises
        ------
        AttributeError
            If any input is not a `Meshes`. :contentReference[oaicite:13]{index=13}
        """
        
        from .mesh import Mesh

        for other in others:

            if not isinstance(other, Meshes):
                raise AttributeError(f"Meshes can be joined with Meshes only, not {type(other).__name__}.")
            
            self.mesh.join(other.mesh)
            for ob in other.buckets:
                ok = False
                for i, sb in enumerate(self.buckets):
                    if ob.shape[-1] == sb.shape[-1]:
                        self.buckets[i] = np.append(sb, ob, axis=0)
                        ok = True
                if not ok:
                    self.buckets.append(np.array(ob))

        return self
    
    # ====================================================================================================
    # Multiply
    # ====================================================================================================

    def multiply(self, count, in_place=True):
        """
        Duplicate all pieces `count` times.

        Buckets are offset by the source mesh vertex count to index the newly
        appended mesh copies; both the `points` domain and the `mesh` are
        multiplied accordingly.

        Parameters
        ----------
        count : int
            Number of copies to create.
        in_place : bool, default=True
            If True, expand this instance; otherwise return a new expanded copy.

        Returns
        -------
        Meshes or None
            `self` (in place) or a new `Meshes`; `None` if `count == 0`. :contentReference[oaicite:14]{index=14}
        """

        count = int(count)

        if count == 0:
            return None
        
        if count == 1:
            if in_place:
                return self
            else:
                return type(self)(attr_from=self).join(self)
            
        if not in_place:
            return type(self)(attr_from=self).join(self).multiply(count, in_place=True)
        
        # Multiply the buckets
        npoints = len(self.mesh.points)
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            n, length = bucket.shape
            bucket = bucket[None] + (np.arange(count)*npoints)[:, None, None]
            self.buckets[i] = bucket.reshape(n*count, length)
            
        # Multiply the points and the mesh        
        self.points.multiply(count)
        self.mesh.multiply(count)
        
        return self

    def __mul__(self, count):
        return self.multiply(count, in_place=False)

    def __imul__(self, count):
        return self.multiply(count, in_place=True)
    
    # ====================================================================================================
    # Operations
    # ====================================================================================================

    def clear_geometry(self):
        """
        Reset to an empty state.

        Clears the mesh, empties `points`, and removes all buckets.

        Returns
        -------
        None :contentReference[oaicite:17]{index=17}
        """
        from .mesh import Mesh

        self.mesh = Mesh()  
        self.points.clear()  
        self.buckets = []

    # ----------------------------------------------------------------------------------------------------
    # Set an attribute on mesh points
    # ----------------------------------------------------------------------------------------------------

    def set_mesh_points_attribute(self, domain_name, name, value):
        """
        Broadcast a per-piece value to a per-vertex mesh attribute.

        Looks up `mesh.points[name]`, broadcasts `value` to `(len(self), *attr_shape)`,
        then writes the piece value to all vertices of each row in the buckets.

        Parameters
        ----------
        domain_name : str
            Unused placeholder (kept for API symmetry).
        name : str
            Name of the mesh **point** attribute to write.
        value : array-like
            Value(s) per piece, broadcastable to the attribute shape.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the mesh point attribute `name` does not exist. :contentReference[oaicite:18]{index=18}
        """

        # Will raise en error if attribute doesn't exist
        attr = self.mesh.points[name]

        # _infos can be accessed securely
        value = np.broadcast_to(value, (len(self),) + self.mesh.points._infos[name]['shape'])

        # Loop on the buckets
        offset = 0
        for bucket in self.buckets:
            attr[bucket] = value[offset:offset + len(bucket), None]
            offset += len(bucket)

        return self


