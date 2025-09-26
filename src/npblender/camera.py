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
Module Name: camera
Author: Alain Bernard
Version: 0.1.0
Created: 2022-11-11
Last updated: 2025-08-29

Summary:

This module offers:
- `camera_projection_jit(...)`: Numba-accelerated per-point projection and visibility.
- `camera_projection(...)`: NumPy reference implementation of the same API.
- `Camera`: convenience class wrapping Blender camera data and providing:
    * pixel density vs distance (`pixels_per_meter`)
    * distance computations
    * visibility of points, edges, faces, and islands

Conventions:
- Coordinates are transformed into the camera frame using `camera.matrix_world.inverted()`.
- Projection is onto the plane `z = cam_z` defined by Blender's `view_frame`.
- Visibility flags are returned as a (N, 7) boolean array with indices:
    0: VISIBLE, 1: BACK_FACE, 2: BEHIND, 3: LEFT, 4: RIGHT, 5: BELOW, 6: ABOVE
- Distances/size are returned as a (N, 2) float array with indices:
    0: DISTANCE (Euclidean), 1: SIZE (apparent size from `radius`)

Notes:
- Prefer float32 contiguous arrays for best JIT performance.
- Back-face test uses dot(normal, point - cam_loc) > 0 as "facing away".

Usage example:
    >>> from camera import Camera
    >>> cam = Camera()
    >>> vis, dist = cam.visible_points(...)

Notes:
    - Add any relevant implementation details or dependencies.
    - Update version and dates when modifying this file.
"""

__all__ = ['Camera']


# ====================================================================================================
# npblender — Camera visibility & projection utilities
# Part of the npblender package
# License: MIT
# Created: 11/11/2022
# Last Updated: 12/08/2025
# Author: Alain Bernard
#
# Description:
#     High-performance camera-space projection and visibility tests for Blender meshes.
#     Provides a Numba-accelerated path and a NumPy fallback, plus a convenience Camera class.
#
# Dependencies:
#     - numpy, numba
#     - bpy, mathutils (Blender Python API)
#
# File:
#     camera.py
# ====================================================================================================

"""
Camera projection & visibility helpers for Blender.

This module offers:
- `camera_projection_jit(...)`: Numba-accelerated per-point projection and visibility.
- `camera_projection(...)`: NumPy reference implementation of the same API.
- `Camera`: convenience class wrapping Blender camera data and providing:
    * pixel density vs distance (`pixels_per_meter`)
    * distance computations
    * visibility of points, edges, faces, and islands

Conventions:
- Coordinates are transformed into the camera frame using `camera.matrix_world.inverted()`.
- Projection is onto the plane `z = cam_z` defined by Blender's `view_frame`.
- Visibility flags are returned as a (N, 7) boolean array with indices:
    0: VISIBLE, 1: BACK_FACE, 2: BEHIND, 3: LEFT, 4: RIGHT, 5: BELOW, 6: ABOVE
- Distances/size are returned as a (N, 2) float array with indices:
    0: DISTANCE (Euclidean), 1: SIZE (apparent size from `radius`)

Notes:
- Prefer float32 contiguous arrays for best JIT performance.
- Back-face test uses dot(normal, point - cam_loc) > 0 as "facing away".
"""

import numpy as np

import bpy
from mathutils import Vector

from numba import njit, prange
from time import time

# =============================================================================================================================
# Just in time optimization

#from numba import njit, prange
from .numbawrapper import njit, prange, NUMBA_AVAILABLE
import numpy as np

USE_JIT = NUMBA_AVAILABLE

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
    - pts : array(n, 2) of vectors
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
    - pts : array(n, 2) of vectors
    """

    VISIBLE   = 0
    BACK_FACE = 1

    BEHIND    = 2
    LEFT      = 3
    RIGHT     = 4
    BELOW     = 5
    ABOVE     = 6

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

# ====================================================================================================
# Camera
# ====================================================================================================

class Camera:

    VISIBLE   = 0
    BACK_FACE = 1

    BEHIND    = 2
    LEFT      = 3
    RIGHT     = 4
    BELOW     = 5
    ABOVE     = 6
    SLICE_END = 7

    DISTANCE = 0
    SIZE     = 1

    def __init__(self, camera=None):

        if camera is None or camera == True:
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
    # ----------------------------------------------------------------------------------------------------

    @property
    def bcamera(self):
        if self._bcamera is None:
            return bpy.context.scene.camera
        else:
            return self._bcamera

    # ----------------------------------------------------------------------------------------------------
    # Camera world location
    # ----------------------------------------------------------------------------------------------------

    @property
    def location(self):
        return self.bcamera.matrix_world.translation

    # ----------------------------------------------------------------------------------------------------
    # Focal angle of the camera
    # ----------------------------------------------------------------------------------------------------

    @property
    def focal_angle(self):
        return self.bcamera.data.angle
    
    # ----------------------------------------------------------------------------------------------------
    # Clip
    # ----------------------------------------------------------------------------------------------------

    @property
    def clip_start(self):
        return self.bcamera.data.clip_start

    @property
    def clip_end(self):
        return self.bcamera.data.clip_end

    # ----------------------------------------------------------------------------------------------------
    # Normalized vector representing the direction of the camera
    # ----------------------------------------------------------------------------------------------------

    @property
    def direction(self):
        return (self.bcamera.matrix_world @ Vector((0, 0, -1, 0))).resized(3).normalized()

    # ----------------------------------------------------------------------------------------------------
    # Scene resolution
    # ----------------------------------------------------------------------------------------------------

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
        
    # ====================================================================================================
    # Compute visibility, distance and apparent size
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Visible points
    # ----------------------------------------------------------------------------------------------------

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
            vis, dist, pts = camera_projection_jit(M, cam_z, cam_x0, cam_x1, cam_y0, cam_y1, verts, radius, cam_loc, normals)
        else:
            vis, dist, pts = camera_projection(M, cam_z, cam_x0, cam_x1, cam_y0, cam_y1, verts, radius, cam_loc, normals)

        # ----------------------------------------------------------------------------------------------------
        # Return visibility and distances

        if return_proj:
            return vis, dist, pts
        else:
            return vis, dist
    
    # ----------------------------------------------------------------------------------------------------
    # Visible edges
    # ----------------------------------------------------------------------------------------------------

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
    
    # ----------------------------------------------------------------------------------------------------
    # Visible faces
    # ----------------------------------------------------------------------------------------------------

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
                if not n_vis[i_face, self.BACK_FACE]:
                    continue

            # ----- Visibility of corners : (n, 7)

            index = face.loop_start
            n = face.loop_total

            c_vis = v_vis[mesh.corners.vertex_index[index:index + n]]
            diff_side = np.all(c_vis, axis=0)

            vis_faces[i_face] = ~np.any(diff_side[1:self.SLICE_END])

            if False:
                print("Camera c_vis\n", c_vis)
                print("Camera diff_side\n", diff_side)
                print("Camera vis_faces\n", vis_faces)

            # ----- Face Size

            f_proj = proj[mesh.corners.vertex_index[index:index + n]]
            xmin, xmax = np.min(f_proj, axis=0), np.max(f_proj, axis=0)
            size[i_face] = np.linalg.norm(xmax - xmin)

        return vis_faces, size, proj
    
    # ----------------------------------------------------------------------------------------------------
    # Visible islands
    # ----------------------------------------------------------------------------------------------------

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
    
    # ====================================================================================================
    # Scaling
    # ====================================================================================================

    def distance_for_scale(self, size_max, scale=1.0, margin=0.0, fit_axis="auto"):
        """
        Compute:
        - d0: distance at which an unscaled object of real size `size_max` exactly fits
                the camera frame along `fit_axis` (largest dimension touches the borders).
        - d : distance such that the UNscaled object at distance d has the same apparent size
                as the SCALED object (size_max * scale) at distance d0. (i.e. d = d0 / scale)
        - meters_per_pixel: world-space length at distance d that projects to exactly 1 pixel
                (so two vertices closer than this fall onto the same pixel).

        Parameters
        ----------
        size_max : float
            Largest real dimension of the mesh (Blender units).
        scale : float
            Geometry scale (> 0) applied at distance d0.
        margin : float, optional
            Extra margin around the frame (same convention as elsewhere).
        fit_axis : {"auto", "width", "height"}
            Which frame span to use to define "exactly fits".
            - "auto": min(width, height)
            - "width": frame width
            - "height": frame height

        Returns
        -------
        d : float
            Distance where the UNscaled object matches the apparent size
            of the SCALED one at d0 (d = d0 / scale).
        meters_per_pixel : float
            World-space size corresponding to 1 pixel at distance d.

        Notes
        -----
        - Uses Blender's view_frame on plane z = cam_z.
        width  = cam_x1 - cam_x0
        height = cam_y1 - cam_y0
        - Perspective: apparent_size ∝ size / distance
        equality ⇒ size_max / d = (size_max * scale) / d0 ⇒ d = d0 / scale.
        """
        if scale <= 0:
            raise ValueError("`scale` must be > 0.")

        camera = self.bcamera
        c0, c1, c2, c3 = camera.data.view_frame(scene=bpy.context.scene)

        cam_x0, cam_y0 = c2.x - margin, c2.y - margin
        cam_x1, cam_y1 = c0.x + margin, c0.y + margin
        cam_z = c0.z  # typically negative in Blender

        frame_w = (cam_x1 - cam_x0)
        frame_h = (cam_y1 - cam_y0)

        if fit_axis == "width":
            frame_span = frame_w
        elif fit_axis == "height":
            frame_span = frame_h
        else:  # "auto"
            frame_span = frame_w if frame_w < frame_h else frame_h

        if frame_span <= 0:
            raise RuntimeError("Invalid camera frame span (width/height <= 0).")

        # d0: unchanged logic — object (unscaled) exactly fits the frame
        d0 = (-cam_z) * float(size_max) / float(frame_span)

        # Distance so that UNscaled@d matches SCALED@d0
        d = d0 / float(scale)

        # Pixel size at distance d:
        # pixels_per_meter(d) -> px/m, so 1 px corresponds to 1 / (px/m) meters
        px_per_m = float(self.pixels_per_meter(d))
        meters_per_pixel = 1.0 / px_per_m if px_per_m > 0.0 else float("inf")

        return d, meters_per_pixel





        


