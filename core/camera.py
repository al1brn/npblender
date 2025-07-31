#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 07:38:55 2022

@author: alain
"""

import numpy as np

import bpy
from mathutils import Vector

from npblender.core import engine

from numba import njit, prange
from time import time

# =============================================================================================================================
# Just in time optimization

from numba import njit, prange
import numpy as np

USE_JIT = True

@njit(parallel=True)
def camera_projection_jit(M, cam_z, cam_x0, cam_x1, cam_y0, cam_y1, verts, radius=0., cam_loc=(0., 0., 0.), normals=None):
    """ Just in time optimization

    Arguments
    ---------
    - M (camera matrix world) : np.array(camera.matrix_world.inverted())
    - cam_z, cam_x0, cam_x1, cam_y0, cam_y1 : camera data
    - verts (array of vectors) : vertex locations
    - radius (array of floats or float = 0.) : size at the locations
    - margin (float, default=0.) : margin factor around the camera
    - normals (array of vectors = None) : normal pointing outwards are not visible

    Returns
    -------
    - visibility : array(n, 7) of bools
    - distance : array(n, 2) of floats
    - pts : array(n, 3) of vectors
    """

    # ----------------------------------------------------------------------------------------------------
    # Resulting arrays

    count = verts.shape[0]
    visibility = np.empty((count, 7), dtype=np.bool_)
    dist_size  = np.empty((count, 2), dtype=np.float32)
    projected  = np.empty((count, 2), dtype=np.float32)

    for i in prange(count):

        r = radius[i]

        # Transform
        vx, vy, vz = verts[i]
        vec = np.empty(4, dtype=np.float32)
        vec[0], vec[1], vec[2], vec[3] = vx, vy, vz, 1.0

        rot = M @ vec
        x, y, z = rot[0], rot[1], rot[2]

        # Distance and size
        dist = np.sqrt(x*x + y*y + z*z)
        size = (-r * cam_z) / dist if dist != 0 else 0.0

        # Clipping
        behind = (z - r > 0.01)
        f = cam_z / z if abs(z) > 1e-6 else 0.0
        x_proj = x * f
        y_proj = y * f

        # Projection limits
        left  = x_proj + size < cam_x0
        right = x_proj - size > cam_x1
        below = y_proj + size < cam_y0
        above = y_proj - size > cam_y1

        # Back-facing
        back_face = False
        if normals is not None:
            nx, ny, nz = normals[i]
            px, py, pz = verts[i] - cam_loc
            dot = nx*px + ny*py + nz*pz
            back_face = False
            back_face = dot > 0.0

        visible = not (behind or left or right or below or above)

        # Write results directly
        visibility[i, 0] = visible
        visibility[i, 1] = back_face

        visibility[i, 2] = behind
        visibility[i, 3] = left
        visibility[i, 4] = right
        visibility[i, 5] = below
        visibility[i, 6] = above

        dist_size[i, 0]  = dist
        dist_size[i, 1]  = size

        projected[i, 0] = x_proj
        projected[i, 1] = y_proj

    return visibility, dist_size, projected

# =============================================================================================================================
# Numpy version

def camera_projection(M, cam_z, cam_x0, cam_x1, cam_y0, cam_y1, verts, radius=0., cam_loc=(0., 0., 0.), normals=None):
    """ Camera projection, numpy legacy version 

    Arguments
    ---------
    - M (camera matrix world) : np.array(camera.matrix_world.inverted())
    - cam_z, cam_x0, cam_x1, cam_y0, cam_y1 : camera data
    - verts (array of vectors) : vertex locations
    - radius (array of floats or float = 0.) : size at the locations
    - margin (float, default=0.) : margin factor around the camera
    - normals (array of vectors = None) : normal pointing outwards are not visible

    Returns
    -------
    - visibility : array(n, 7) of bools
    - distance : array(n, 2) of floats
    - pts : array(n, 3) of vectors
    """

    VISIBLE   = 0
    BACK_FACE = 1

    BEHIND    = 2
    LEFT      = 3
    RIGHT     = 4
    BELOW     = 5
    ABOVE     = 6
    LAST      = 6

    DISTANCE = 0
    SIZE     = 1

    # ----------------------------------------------------------------------------------------------------
    # Resulting arrays

    count = np.size(verts)//3
    vis  = np.zeros((count, 7), dtype=np.bool_)
    dist = np.zeros((count, 2), dtype=np.float32)

    # ----------------------------------------------------------------------------------------------------
    # Rotate the points in the camera frame

    #M = np.array(camera.matrix_world.inverted())
    pts = np.array(np.einsum('...jk, ...k', M, np.insert(verts, 3, 1, axis=-1))[..., :3])

    # ----------------------------------------------------------------------------------------------------
    # Compute the distances

    dist[:, DISTANCE] = np.linalg.norm(pts, axis=-1)

    # ----------------------------------------------------------------------------------------------------
    # Apparent size

    dist[:, SIZE] = (radius*(-cam_z))/dist[:, DISTANCE]
    r = dist[:, SIZE]

    # ----------------------------------------------------------------------------------------------------
    # Points must be in front of the camera

    vis[:, BEHIND] = pts[..., 2] - radius > 0.01 #clip_start

    # ---------------------------------------------------------------------------
    # Project the points on the plane z = cam_z

    pts = cam_z*(pts/np.expand_dims(pts[..., 2], axis=-1))[..., :2]

    # ---------------------------------------------------------------------------
    # Must be projected into the rectangle
    
    vis[:, LEFT]  = (pts[..., 0] + r < cam_x0)

    vis[:, RIGHT] = (pts[..., 0] - r > cam_x1)
    vis[:, ABOVE] = (pts[..., 1] - r > cam_y1)
    vis[:, BELOW] = (pts[..., 1] + r < cam_y0)

    vis[:, VISIBLE] = ~np.any(vis[:, 2:(LAST+1)], axis=1)

    # ----------------------------------------------------------------------------------------------------
    # And when properly oriented

    if normals is not None:
        vs = verts - cam_loc
        vis[:, BACK_FACE] = np.einsum('...i, ...i', vs, normals) > 0

    # ----------------------------------------------------------------------------------------------------
    # Return visibility and distances

    return vis, dist, pts

# =============================================================================================================================
# Camera

class Camera:

    VISIBLE   = 0
    BACK_FACE = 1

    BEHIND    = 2
    LEFT      = 3
    RIGHT     = 4
    BELOW     = 5
    ABOVE     = 6
    LAST      = 6

    DISTANCE = 0
    SIZE     = 1

    def __init__(self, camera=None):

        if camera is None:
            self._bcamera = None

        elif isinstance(camera, str):
            self._bcamera = bpy.data.objects[camera]

        else:
            assert(isinstance(camera.data, bpy.types.Camera))
            self._bcamera = camera


    def __str__(self):
        return f"<Camera '{self.bcamera.name}': focal: {np.degrees(self.focal_angle):.1f}°, resolution: ({self.resolution_x}, {self.resolution_y})>"

    # ----------------------------------------------------------------------------------------------------
    # The Blender camera

    @property
    def bcamera(self):
        if self._bcamera is None:
            return bpy.context.scene.camera
        else:
            return self._bcamera

    # ----------------------------------------------------------------------------------------------------
    # Camera world location

    @property
    def location(self):
        return self.bcamera.matrix_world.translation

    # ----------------------------------------------------------------------------------------------------
    # Focal angle of the camera

    @property
    def focal_angle(self):
        return self.bcamera.data.angle
    
    # ----------------------------------------------------------------------------------------------------
    # Clip

    @property
    def clip_start(self):
        return self.bcamera.data.clip_start

    @property
    def clip_end(self):
        return self.bcamera.data.clip_end

    # ----------------------------------------------------------------------------------------------------
    # Normalized vector representing the direction of the camera

    @property
    def direction(self):
        return (self.bcamera.matrix_world @ Vector((0, 0, -1, 0))).resized(3).normalized()

    # ----------------------------------------------------------------------------------------------------
    # Scene resolution

    @property
    def resolution_x(self):
        render = bpy.context.scene.render
        return render.resolution_x * render.pixel_aspect_x

    @property
    def resolution_y(self):
        render = bpy.context.scene.render
        return render.resolution_y * render.pixel_aspect_y

    @property
    def resolution(self):
        if self.bcamera.data.sensor_fit == 'VERTICAL':
            return self.resolution_y
        else:
            return self.resolution_x
    
    # ----------------------------------------------------------------------------------------------------
    # Number of pixels per meter function of the distance
    # ----------------------------------------------------------------------------------------------------
    
    def pixels_per_meter(self, distances):
        """ Returns the number of pixels per meter given the distances

        Arguments
        ---------
        - distances (array of floats) : the distances

        Returns
        -------
        - array of floats : the size of a meter in pixels
        """
        cam_data = self.bcamera.data

        # Focal and sensor in mm
        focal_mm = cam_data.lens
        sensor_fit = cam_data.sensor_fit

        if sensor_fit == 'VERTICAL':
            sensor_mm = cam_data.sensor_height
            res_px = self.resolution_y
        else:  # 'HORIZONTAL' or 'AUTO'
            sensor_mm = cam_data.sensor_width
            res_px = self.resolution_x

        # Focal in pixels
        focal_px = (focal_mm / sensor_mm) * res_px

        # Meter size
        size_in_pixels = focal_px / distances

        return size_in_pixels    

    # ----------------------------------------------------------------------------------------------------
    # Distance of a location to the camera
    # ----------------------------------------------------------------------------------------------------

    def distance(self, location):

        if isinstance(location, Vector) or np.shape(location) == (3,):
            return (self.location - Vector(location)).length

        else:
            return np.linalg.norm(location - self.location, axis=-1)
        
    # =============================================================================================================================
    # Compute a set of vertices

    def visible_points(self, verts, radius=0., margin=0., normals=None, return_proj=False):
        """ Compute the visibility of vertices.

        For each vertex, the following values are computed:
        - visible : vertex is visible (all bools below are False)
        - behind : vertex is behind the visible rectangle
        - left : vertex is left to the visible rectangle
        - right : vertex if right to the visible rectangle
        - above : vertex is above the visible rectangle
        - below : vertex is below the visible rectangle
        - back : normal points outards
        - distance : distance to the camera
        - size : apparent size (based on radius)

        Arguments
        ---------
            - verts (array of vectors) : vertex locations
            - radius (array of floats or float = 0.) : size at the locations
            - margin (float, default=0.) : margin factor around the camera
            - normals (array of vectors = None) : normal pointing outwards are not visible

        Returns
        -------
            - couple of arrays : array[n, 7] of bools, array[n, 2] of floats
        """

        # ----------------------------------------------------------------------------------------------------
        # Resulting arrays

        #count = np.size(verts)//3
        #vis  = np.zeros((count, 7), dtype=bool)
        #dist = np.zeros((count, 2), dtype=float)

        # ----------------------------------------------------------------------------------------------------
        # Current camera object

        camera = self.bcamera

        # ----------------------------------------------------------------------------------------------------
        # The projection rectangle
        # The plane is horizontal. All z<0 are identical
        # View frame corners are returned in the order:
        # - top-right
        # - bottom-right
        # - bottom-left
        # - top-left        

        c0, c1, c2, c3 = camera.data.view_frame(scene=bpy.context.scene)

        cam_x0, cam_y0 = c2.x - margin, c2.y - margin
        cam_x1, cam_y1 = c0.x + margin, c0.y + margin
        cam_z = c0.z

        if False:
            print("-"*80)
            print("view_frame", c0, c1, c2, c3)
            print(f"CAMERA SHAPE: plane z = {cam_z:.1f}")
            print(f"   x: {cam_x0:5.1f} {cam_x1:5.1f}")
            print(f"   y: {cam_y0:5.1f} {cam_y1:5.1f}")
            print()

        # ----------------------------------------------------------------------------------------------------
        # Camera rotation matrix

        M = np.array(camera.matrix_world.inverted(), dtype=np.float32)

        # ----------------------------------------------------------------------------------------------------
        # Prepare arguments

        count  = np.size(verts)//3
        verts  = np.reshape(verts, (count, 3)).astype(np.float32)
        if not isinstance(radius, np.ndarray):
            radius = np.full(count, radius, dtype=np.float32)
        if normals is not None:
            normals = np.resize(normals, (count, 3)).astype(np.float32)
        cam_loc = np.array(self.location, dtype=np.float32)

        if USE_JIT:
            if True:
                vis, dist, pts = camera_projection_jit(M, cam_z, cam_x0, cam_x1, cam_y0, cam_y1, verts, radius, cam_loc, normals)

            else:
                t0 = time()
                vis_, dist_, pts_ = camera_projection(M, cam_z, cam_x0, cam_x1, cam_y0, cam_y1, verts, radius, normals)
                t1 = time() - t0

                t0 = time()
                vis, dist, pts = camera_projection_jit(M, cam_z, cam_x0, cam_x1, cam_y0, cam_y1, verts, radius, normals)
                t2 = time() - t0

                print(f"JIT Perfs ({count}): numpy: {t1:.2f} s, jit={t2:.2f} s, {np.sum(np.logical_xor(vis, vis_))}, {np.max(np.abs(dist-dist_)):.3f}, {np.max(np.abs(pts-pts_)):.3f}, ")
        else:
            vis, dist, pts = camera_projection(M, cam_z, cam_x0, cam_x1, cam_y0, cam_y1, verts, radius, cam_loc, normals)

        # ----------------------------------------------------------------------------------------------------
        # Return visibility and distances

        if return_proj:
            return vis, dist, pts
        else:
            return vis, dist
    
    # =============================================================================================================================
    # Compute the visibility of edges

    def visible_edges(self, mesh, radius=0., margin=0.):
        """ Mesh edges visibility

        The visibility of each point is computed with the given radius.
        An edge is considered invisible if both points are hidden for the same reason:
        both are left to the camera, or right, or behind...

        In the other case, the edge is considered as visible.

        Arguments
        ---------
        mesh : Mesh
            The mesh with points and edges
        radius : float
            The radius of the camera
        margin : float
            The margin of the camera

        Returns
        -------
        vis : array (n) of bools
            The visibility of each edge
        size : array (n) of floats
            The size of the projected edges
        """

        v_vis, v_dist, proj = self.visible_points(mesh.points.position, radius=radius, margin=margin, return_proj=True)
        edge0 = mesh.edges.vertex0
        edge1 = mesh.edges.vertex1
        
        diff_side = np.logical_and(v_vis[edge0, :], v_vis[edge1, :])
        vis = ~np.any(diff_side[:, 2:7], axis=1)

        size = np.linalg.norm(proj[edge1] - proj[edge0], axis=1)

        return vis, size
    
    # =============================================================================================================================
    # Compute the visibility of edges

    def visible_faces(self, mesh, margin=0., back_face_culling=False):
        """ Mesh faces visibility

        A face is considered invisible if all its points are hidden for the same reason:
        all are left to the camera, or right, or behind...

        Arguments
        ---------
        mesh : Mesh
            The mesh with points and faces
        margin : float
            The margin of the camera
        back_face_culling : bool
            If True, the back face is not considered as visible

        Returns
        -------
        vis : array (n) of bools
            The visibility of each face
        size : array (n) of floats
            The size of the projected faces
        proj : array (n, 3) of floats
            The projected position of each face
        """

        # ----- Number of faces to analyze

        count = len(mesh.faces)
        if count == 0:
            return np.zeros(0, bool), np.zeros(0, float), np.zeros(0, float)
        
        # ----- Project the points

        v_vis, v_dist, proj = self.visible_points(mesh.points.position, margin=margin, return_proj=True)

        # ----- Back face culling : normal centered on the face position

        if back_face_culling:
            n_vis, _ = self.visible_points(mesh.faces.position, margin=margin, normals=mesh.faces.normal)

        # ----------------------------------------------------------------------------------------------------
        # Loop on the faces

        vis_faces = np.zeros(count, bool)
        size = np.zeros(count, float)
        for i_face, face in enumerate(mesh.faces):

            # ----- Nothing to do if back face

            if back_face_culling:
                if not n_vis[i_face, self.BACK]:
                    continue

            # ----- Visibility of corners : (n, 7)

            index = face.loop_start
            n = face.loop_total

            c_vis = v_vis[mesh.corners.vertex_index[index:index + n]]
            diff_side = np.all(c_vis, axis=0)

            vis_faces[i_face] = ~np.any(diff_side[1:6])

            if False:
                print("Camera c_vis\n", c_vis)
                print("Camera diff_side\n", diff_side)
                print("Camera vis_faces\n", vis_faces)

            # ----- Face Size

            f_proj = proj[mesh.corners.vertex_index[index:index + n]]
            xmin, xmax = np.min(f_proj, axis=0), np.max(f_proj, axis=0)
            size[i_face] = np.linalg.norm(xmax - xmin)

        return vis_faces, size
    
    # =============================================================================================================================
    # Compute the visibility of edges

    def visible_islands(self, mesh, islands, attribute="Island", margin=0.):
        """ Mesh islands visibility

        Islands are defined by an integer.

        Visibility is computed with the position and size of the islands

        Arguments
        ---------
        mesh : Mesh
            The mesh with points and faces
        islands: array of ints
            One identifier per island
        attribute: attribute name
            The attribute name to use for the islands
        margin : float
            The margin of the camera

        Returns
        -------
            - couple of arrays : array[n, 7] of bools, array[n, 2] of floats
        """

        # ----- Number of islands to analyze

        count = len(islands)
        if count == 0:
            return np.zeros(0, bool), np.zeros(0, float), np.zeros(0, float)
        
        # ----------------------------------------------------------------------------------------------------
        # Loop on the islands

        positions = np.zeros((count, 3), float)
        radius = np.zeros(count, float)
        size = np.zeros(count)
        for i_island, island in enumerate(islands):

            sel = mesh.points.attributes[attribute] == island
            pts = mesh.points[sel].position

            vmin, vmax = np.min(pts, axis=0), np.max(pts, axis=0)
            pos = (vmin + vmax) / 2
            r = np.linalg.norm(vmax - vmin) / 2

            positions[i_island] = pos
            radius[i_island] = r

        # ----- Let' go

        return self.visible_points(positions, radius, margin=margin)
    

        # ----------------------------------------------------------------------------------------------------
        # Alternate algorithm, same as faces
        
        # ----- Project the points

        v_vis, v_dist, proj = self.visible_points(mesh.points.position, margin=margin, return_proj=True)

        # ----------------------------------------------------------------------------------------------------
        # Loop on the islands

        vis_islands = np.zeros(count, bool)
        size = np.zeros(count, float)
        for i_island, island in enumerate(islands):

            # ----- Points belonging to the island

            sel = mesh.points.attributes[attribute] == island

            # ----- Visibility of islands points : (sel, 7)

            c_vis = v_vis[sel]
            diff_side = np.all(c_vis, axis=0)

            vis_islands[i_island] = ~np.any(diff_side[1:6])

            if False:
                print("Camera c_vis\n", c_vis)
                print("Camera diff_side\n", diff_side)
                print("Camera vis_faces\n", vis_faces)

            # ----- Island Size

            f_proj = proj[sel]
            xmin, xmax = np.min(f_proj, axis=0), np.max(f_proj, axis=0)
            size[i_island] = np.linalg.norm(xmax - xmin)

        return vis_islands, size


    # =============================================================================================================================
    # Compute the visibility of vertices

    def visible_verts_OLD(self, verts, radius=None, close_distance=None, max_distance=None, margin=1., normals=None):
        """ Compute the visibility of vertices.

        Arguments
        ---------
            - verts (array of vectors) : vertex locations
            - radius (float or array of floats = None) : size at the locations
            - close_distance (float = None) : vertices closer than this distance are visible
            - max_distance (float = None) : vertices father that this distance are not visibles
            - margin (float, default=1.) : margin factor around the camera
            - normals (array of vectors = None) : normal pointing outwards are not visible

        Returns
        -------
            - visibles (array of bools)   : visible vertices
            - distances (array of floats) : The distance to the camera
        """

        camera = self.bcamera

        # ----------------------------------------------------------------------------------------------------
        # The projection rectangle
        # The plane is horizontal. All z<0 are identical

        c0, c1, c2, c3 = camera.data.view_frame(scene=bpy.context.scene)

        cam_x0 = min(c0.x, c1.x, c2.x, c3.x)*margin
        cam_x1 = max(c0.x, c1.x, c2.x, c3.x)*margin
        cam_y0 = min(c0.y, c1.y, c2.y, c3.y)*margin
        cam_y1 = max(c0.y, c1.y, c2.y, c3.y)*margin

        cam_z = c0.z

        if False:
            print("-"*80)
            print(f"CAMERA SHAPE: plane z = {cam_z:.1f}")
            print(f"   x: {cam_x0:5.1f} {cam_x1:5.1f}")
            print(f"   y: {cam_y0:5.1f} {cam_y1:5.1f}")
            print()

        # ----------------------------------------------------------------------------------------------------
        # Rotate the points in the camera frame

        M = np.array(camera.matrix_world.inverted())
        pts = np.array(np.einsum('...jk, ...k', M, np.insert(verts, 3, 1, axis=-1))[..., :3])

        # ----------------------------------------------------------------------------------------------------
        # Compute the distances

        distances = np.linalg.norm(pts, axis=-1)

        # ----------------------------------------------------------------------------------------------------
        # Projected radius

        if radius is None:
            r = 0
        else:
            r = (radius*cam_z)/distances

        # ----------------------------------------------------------------------------------------------------
        # Points must be in front of the camera

        visibles = pts[..., 2] <= cam_z

        # ---------------------------------------------------------------------------
        # Project the points on the plane z = cam_z

        pts = cam_z*(pts/np.expand_dims(pts[..., 2], axis=-1))[..., :2]

        # ---------------------------------------------------------------------------
        # Must be projected into the rectangle

        visibles &= (
            (pts[..., 0] - r >= cam_x0)  &
            (pts[..., 0] + r <= cam_x1)  &
            (pts[..., 1] - r >= cam_y0)  &
            (pts[..., 1] + r <= cam_y1)
            )

        # ---------------------------------------------------------------------------
        # Not too far

        if max_distance is not None:
            visibles &= distances < max_distance

        # ---------------------------------------------------------------------------
        # Visible when close to the camera

        if close_distance is not None:
            visibles |= distances <= close_distance

        # ----------------------------------------------------------------------------------------------------
        # And when properly oriented

        if normals is not None:
            vs = verts - self.location
            visibles &= np.einsum('...i, ...i', vs, normals) < 0

        # ----------------------------------------------------------------------------------------------------
        # Return visibility and distances

        return visibles, distances

    # =============================================================================================================================
    # Demonstration

    @staticmethod
    def demo(count=100000, size=1000, seed=0):

        from npblender.core.instances import Instances
        from npblender.core.mesh import Mesh

        rng = np.random.default_rng(seed)

        verts = rng.uniform(-size, size, (count, 3))
        verts[..., 2] = 0

        radius = rng.uniform(.1, 3, count)

        def update(eng):
            visibles, _ = Camera().visible_verts(verts, radius, close_distance=None, max_distance=500)
            scale = np.empty_like(verts[visibles])
            scale[:] = radius[visibles, None]
            insts = Instances(verts[visibles], models=Mesh.IcoSphere(), Scale=scale)
            insts.to_object("Camera Culling")

        engine.go(update)




