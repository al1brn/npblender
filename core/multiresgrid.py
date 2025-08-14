#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module

Created on Tue Jul 25 2025

@author: alain.bernard
@email: alain@ligloo.net

-----
A multires grid is a grid with a variable resolution, adapted to the camera.

"""

import numpy as np

from .constants import bfloat, bint, bbool
from .blender import merge_by_distance, merge_by_distance_2D
from .mesh import Mesh
from .fieldarray import FieldArray

# ====================================================================================================
# Multi resolution grid

class MultiResGrid(Mesh):
    """ Multi resolution grid

    A surface defined by a function of two parameters u and v.
    The function can return either a vector or an altitude, u and being interpretated as (x, y).

    The grid resolution is defined by a depth when calling the method update_grid.
    The method accepts a camera to adapt its the resolution only to the visible parts.

    Arguments
    ---------
    - shape (2-tuple of ints) : initial division
    - u_space (2-tuple of floats) : u min and max values
    - v_space (2-tuple of floats) : v min and max values
    - is_altitude (bool = False) : the function returns z only vector is built with (u, v) as (x, y)
    """

    def __init__(self, func, shape=(3, 3), u_space=(0, 1), v_space=(0, 1), is_altitude=False, normal_func=None):
        super().__init__()

        if func is None:
            self.func = lambda U, V: np.stack((U, V, np.zeros_like(U)), axis=-1)
        else:
            self.func = func
        self.is_altitude = is_altitude

        if normal_func is None:
            self.use_normals = False
        else:
            self.use_normals = True
            self.normal_func = normal_func

        self.uv_shape = shape
        self.u0, self.ufac = u_space[0], u_space[1] - u_space[0]
        self.v0, self.vfac = v_space[0], v_space[1] - v_space[0]

        self.update_grid()

    # ====================================================================================================
    # Call the surface function

    def call_func(self, u, v):
        """ Call the surface function

        Arguments
        ---------
        - u (array of floats) : u value
        - v (array of floats) : v value

        Returns
        -------
        - array of vectors
        """

        shape = np.shape(u)
        x = self.u0 + u.flatten()*self.ufac
        y = self.v0 + v.flatten()*self.vfac

        if self.is_altitude:
            return np.stack((x, y, self.func(x, y)), axis=1).reshape(shape + (3,))
        else:
            return self.func(x, y).reshape(shape + (3,))
        
    # ====================================================================================================
    # Get the normals to the surface

    def get_normals(self, u, v):
        """ Call the surface function

        Arguments
        ---------
        - u (array of floats) : u value
        - v (array of floats) : v value

        Returns
        -------
        - array of vectors
        """
        if self.use_normals:

            shape = np.shape(u)
            x = self.u0 + u.flatten()*self.ufac
            y = self.v0 + v.flatten()*self.vfac

            return self.normal_func(x, y).reshape(shape + (3,))
        else:
            return None
        
    # ====================================================================================================
    # Compute the normals

    def compute_normals(self, u, v, du, dv):
        """ Compute the face normals
        """

        shape = np.shape(u)

        if hasattr(self, 'normal_func'):
            x = self.u0 + u.flatten()*self.ufac
            y = self.v0 + v.flatten()*self.vfac
            return self.normal_func(x, y).reshape(shape + (3,))
        
        #dx, dy = du*self.ufac, dv*self.vfac

        O = self.call_func(u - du, v - dv)
        I = self.call_func(u + du, v) - O
        J = self.call_func(u, v + dv) - O

        vecs = np.cross(I, J)
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]

        return vecs.reshape(shape + (3,))
        
    # ====================================================================================================
    # Update the grid with the given depth

    def update_grid(self, depth=0, camera=None, max_vertices=10_000_000, max_size=.05):
        """ Update the grid resolution with the given depth

        Arguments
        ---------
        - depth (int) : maximum depth
        - camera (Camera) : camera to compute visible faces
        - max_vertices (int = 10_000_000) : maximum number of vertices
        - max_size (float = .05) : maximum size of faces
        """

        normals_done = False

        # ----------------------------------------------------------------------------------------------------
        # Prepare

        du = 1 / self.uv_shape[0]
        u = (np.arange(self.uv_shape[0]) + 0.5)*du

        dv = 1 / self.uv_shape[1]
        v = (np.arange(self.uv_shape[1]) + 0.5)*dv

        U, V = np.meshgrid(u, v)
        vecs = self.call_func(U, V)

        verts = FieldArray()
        verts.new_field("position", bfloat, 3)
        verts.new_field("size",     bfloat)
        verts.new_field("u",        bfloat)
        verts.new_field("v",        bfloat)
        verts.new_field("du",       bfloat)
        verts.new_field("dv",       bfloat)

        n = self.uv_shape[0]*self.uv_shape[1]
        size = np.sqrt((self.ufac/(self.uv_shape[0] - 1))**2 + (self.vfac/(self.uv_shape[1] - 1))**2)
        du /= 2
        dv /= 2
        verts.append(
            position = vecs.reshape(n, 3),
            size     = size,
            u        = U.reshape(n),
            v        = V.reshape(n),
            du       = du,
            dv       = dv)
        
        # ----------------------------------------------------------------------------------------------------
        # Loop on depth

        #final = verts.clone(empty=True)
        final = FieldArray().join_fields(verts)
        for idepth in range(depth):

            normals = None
            if len(final) > 1000 and not normals_done:
                normals = self.compute_normals(verts.u, verts.v, du, dv)
                normals_done = True

            if camera is not None:
                p_vis, p_size = camera.visible_points(verts.position, radius=verts.size, normals=normals)
                vis = p_vis[:, camera.VISIBLE]
                if normals is not None:
                    vis &= np.logical_not(p_vis[:, camera.BACK_FACE])
                vis &= p_size[:, camera.SIZE] >= max_size
                del p_vis, p_size

                final.extend(verts[np.logical_not(vis)])
                #verts = verts.extract(vis)
                verts = FieldArray(verts, selector=vis)

            count = len(verts)
            if not count:
                break

            size /= 2
            du /= 2
            dv /= 2

            verts.du, verts.dv = du, dv

            #new_verts = verts.clone(empty=True)
            new_verts = FieldArray().join_fields(verts)
            new_verts.set_buffer_size(count*4)

            new_verts.append(
                size = size,
                u = verts.u - du,
                v = verts.v - dv,
                du = du,
                dv = dv
            )
            new_verts.append(
                size = size,
                u = verts.u + du,
                v = verts.v - dv,
                du = du,
                dv = dv
            )
            new_verts.append(
                size = size,
                u = verts.u + du,
                v = verts.v + dv,
                du = du,
                dv = dv
            )
            new_verts.append(
                size = size,
                u = verts.u - du,
                v = verts.v + dv,
                du = du,
                dv = dv
            )

            new_verts.position = self.call_func(new_verts.u, new_verts.v)

            del verts
            verts = new_verts

            if len(final) + len(verts) > max_vertices:
                break

        # ----------------------------------------------------------------------------------------------------
        # Compute the vertices in the quads centered on the points

        final.extend(verts)
        del verts
        nfinal = len(final)

        Ul = final.u - final.du
        Ur = final.u + final.du
        U4 = np.stack((Ul, Ur, Ur, Ul), axis=1)
        del Ul, Ur

        Vb = final.v - final.dv
        Vt = final.v + final.dv
        V4 = np.stack((Vb, Vb, Vt, Vt), axis=1)
        del Vb, Vt

        corners = np.arange(nfinal*4)

        UV, inv = merge_by_distance_2D(np.stack((U4.flatten(), V4.flatten()), axis=1), precision=(du/2, dv/2))
        vertices = self.call_func(UV[:, 0], UV[:, 1])
        corners = inv[corners]

        # ----- To mesh

        self.clear_geometry()

        self.add_geometry(
            points = vertices,
            corners = corners,
            faces = 4,
            UVMap = np.stack((U4.flatten(), V4.flatten()), axis=1),
        )

        #self.points.add_points(vertices)
        #self.corners.add_corners(corners, UVMap=np.stack((U4.flatten(), V4.flatten()), axis=1)),
        #self.faces.add_faces([4]*nfinal)

    # ====================================================================================================
    # Demo function

    @classmethod
    def sphere_demo(cls):

        def sphere(theta, phi):
            ct = np.cos(theta)
            st = np.sin(theta)
            cp = np.cos(phi)
            sp = np.sin(phi)

            return np.stack((cp*ct, cp*st, sp), axis=-1)

        return cls(func=sphere, shape=(16, 8), u_space=(-np.pi, np.pi), v_space=(-np.pi/2, np.pi/2))
    
    @classmethod
    def demo_sphere(cls, depth=6):

        from .camera import Camera
        from .engine import engine

        sphere = cls.sphere_demo()

        def update():
            sphere.update_grid(
                depth=depth,
                camera=Camera(), 
                max_vertices=10_000_000, 
                max_size=.05)
            
            sphere.to_object("MR Sphere", shade_smooth=False)
            
        engine.go(update)

    @classmethod
    def demo_terrain(cls, depth=8):

        from .camera import Camera
        from .engine import engine
        from .maths.perlin import noise

        def altitude(x, y):
            coords = np.stack((x, y), axis=-1)
            return noise(coords, scale=3, octaves=8, lacunarity=4, algo='hetero')
        
        surface = cls(altitude, shape=(10, 10), u_space=(-10, 10), v_space=(-10, 10), is_altitude=True)

        def update():
            surface.update_grid(
                depth=depth,
                camera=Camera(), 
                max_vertices=10_000_000, 
                max_size=.01)
            
            surface.to_object("Terrain", shade_smooth=False)
            
        engine.go(update)

        


            




