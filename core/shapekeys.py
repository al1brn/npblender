#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module

Created on Sat Oct  1 17:29:17 2022
Modified 2024 June 19

@author: alain.bernard
@email: alain@ligloo.net

-----

Curve and Mesh shape keys
"""

def maprange(t, *args, **kwargs):
    return t

# ---------------------------------------------------------------------------
# Blender shape keys are organized
#
# object
#      shape_keys (Key)
#           key_blocks (Prop Collection of ShapeKey)
#                data (Prop Collection of
#                     ShapeKeyPoint
#                     ShapeKeyBezierPoint
#                     ShapeKeyCurvePoint
#
# cube.data.shape_keys.key_blocks[].data[].co
#
# A key is either a string (name of the shape) or an int (index in the array)

import numpy as np
import bpy

from npblender import CubicSpline, BSpline
from npblender.core import blender

# ====================================================================================================
# Sub array

class SubArray:
    def __init__(self, shapekeys, data_slice, item_size):
        self.shapekeys  = shapekeys
        self.data_slice = data_slice
        self.item_size  = item_size

    def __len__(self):
        return len(self.shapekeys)

    def __getitem__(self, index):
        a = self.shapekeys._data[index, :, self.data_slice]
        if self.item_size == 1:
            return np.reshape(a, len(a))
        else:
            return a

    def __setitem__(self, index, value):
        if self.item_size == 1:
            self.shapekeys._data[index, :, self.data_slice] = np.reshape(value, np.shape(value) + (1,))
        else:
            self.shapekeys._data[index, :, self.data_slice] = value

# ====================================================================================================
# Float attributes shape keys

class ShapeKeys:
    def __init__(self, points, count=1, key_name="Key", clear=False, **kwargs):

        self.key_name = key_name
        self.clear    = False

        self._attributes = {'position': np.shape(points)[-1]}
        self._size       = self._attributes['position']
        self._slices     = {'position': slice(self._size)}
        self.position    = SubArray(self, self._slices['position'], self._size)

        for name, v in kwargs.items():
            if len(np.shape(v)) <= 1:
                size = 1
            else:
                size = np.shape(v)[-1]
            self._attributes[name] = size
            self._slices[name] = slice(self._size, self._size + size)
            self._size += size
            setattr(self, name, SubArray(self, self._slices[name], size))

        self._data = np.zeros((count, len(points), self._size), float)

        self._data[..., self._slices['position']] = points[None]
        for k, v in kwargs.items():
            self.set_value(slice(count), k, v)
            #self._data[..., self._slices[k]] = v[None]

    def __str__(self):
        return f"<Shapekeys: {len(self)} shapes of {self.points_count} points, attributes: {list(self._attributes.keys())}>"

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    @property
    def points_count(self):
        return self._data.shape[1]

    # =============================================================================================================================
    # New shape

    def new(self):
        self._data = np.resize(self._data, (len(self)+1,) + np.shape(self._data)[1:])
        self._data[-1] = 0

    # =============================================================================================================================
    # Attributes

    def get_value(self, index, name):
        a = self._data[index, :, self._slices[name]]
        if self._attributes[name] == 1:
            return np.reshape(a, len(a))
        else:
            return a

    def set_value(self, index, name, value):
        if self._attributes[name] == 1:
            self._data[index, :, self._slices[name]] = np.reshape(value, np.shape(value) + (1,))
        else:
            self._data[index, :, self._slices[name]] = value

    def get_interpolated_value(self, data, name):

        a = data[..., self._slices[name]]
        if self._attributes[name] == 1:
            return np.reshape(a, np.shape(a)[:-1])
        else:
            return a

    # =============================================================================================================================
    # From Geometry

    @classmethod
    def FromGeometry(cls, geometry, count=1):

        if type(geometry).__name__ == 'Curve':
            kwargs = {}
            if geometry.has_bezier:
                kwargs['handle_left']  = geometry.points.handle_left
                kwargs['handle_right'] = geometry.points.handle_right
            return cls(geometry.points.position, count=count, radius=geometry.points.radius, tilt=geometry.points.tilt, **kwargs)

        else:
            return cls(geometry.points.position, count=count)

    # =============================================================================================================================
    # From / to Blender Object

    @classmethod
    def FromObject(cls, spec):

        obj = blender.get_object(spec)
        if obj is None:
            return None

        elif isinstance(obj.data, bpy.types.Mesh):
            return cls.FromMeshObject(obj)

        elif isinstance(obj.data, bpy.types.Curve):
            return cls.FromCurveObject(obj)

        else:
            raise RuntimeError(f"Impossible to read shape keys from object of type '{type(obj.data).__name__}'")

    def to_object(self, spec):

        obj = blender.get_object(spec)
        if obj is None:
            return None

        elif isinstance(obj.data, bpy.types.Mesh):
            return self.to_mesh_object(obj)

        elif isinstance(obj.data, bpy.types.Curve):
            return self.to_curve_object(obj)

        else:
            raise RuntimeError(f"Impossible to write shape keys to object of type '{type(obj.data).__name__}'")

    # =============================================================================================================================
    # Mesh

    # -----------------------------------------------------------------------------------------------------------------------------
    # Read the shapes defined in an object

    @classmethod
    def FromMeshObject(cls, spec):

        obj   = blender.get_object(spec)
        count = blender.shape_keys_count(obj)

        if count == 0:
            return None

        sks = cls(blender.get_mesh_vertices(obj), count=count)

        a = np.empty(sks.points_count*3, float)
        for index in range(count):
            kb = blender.get_key_block(obj, index)
            kb.data.foreach_get('co', a)

            sks[index] = np.reshape(a, (sks.points_count, 3))

        return sks

    # -----------------------------------------------------------------------------------------------------------------------------
    # Write the shapes in an existing mesh object

    def to_mesh_object(self, spec):

        obj = blender.get_object(spec)
        assert(len(obj.data.vertices) == self.points_count)

        if self.clear:
            blender.shape_keys_clear(obj)

        a = np.empty(self.points_count*3, float)

        for index in range(0, len(self)):
            kb = blender.get_key_block(obj, index, create=True, name=self.key_name)
            a[:] = np.reshape(self[index], a.shape)
            kb.data.foreach_set('co', a)

        return obj

    # =============================================================================================================================
    # Curve

    # -----------------------------------------------------------------------------------------------------------------------------
    # Read the shapes defined in a curve object

    @classmethod
    def FromCurveObject(cls, spec):

        from npblender.core.curve import Curve

        obj = blender.get_object(spec)

        count = blender.shape_keys_count(obj)

        # ----------------------------------------------------------------------------------------------------
        # No shape key

        if count == 0:
            return None

        # ----------------------------------------------------------------------------------------------------
        # Read the splines

        curve = Curve.FromObject(obj)

        sks = cls.FromGeometry(curve, count=count)

        is_mix     = curve.has_mix_types
        has_bezier = curve.has_bezier
        nverts     = sks.points_count

        if not is_mix:
            v_array = np.empty(nverts*3, float)
            f_array = np.empty(nverts, float)

        # ----------------------------------------------------------------------------------------------------
        # Loop on the shape keys

        for index in range(count):

            key_data = blender.get_key_block(obj, index).data

            # ----- Mix types : we must loop on the splines

            if is_mix:
                for curve_type, loop_start, loop_total in zip(curve.splines.curve_type, curve.splines.loop_start, curve.splines.loop_total):

                    # ----- Bezier

                    if curve_type == blender.BEZIER:

                        for i in range(loop_total):
                            sks._data[index, loop_start + i, sks._slices['position']] = key_data[loop_start + i].co
                            sks._data[index, loop_start + i, sks._slices['handle_left']] = key_data[loop_start + i].handle_left
                            sks._data[index, loop_start + i, sks._slices['handle_right']] = key_data[loop_start + i].handle_right

                            sks._data[index, loop_start + i, sks._slices['radius']] = key_data[loop_start + i].radius
                            sks._data[index, loop_start + i, sks._slices['tilt']]   = key_data[loop_start + i].tilt

                    # ----- Non Bezier

                    else:
                        for i in range(loop_total):
                            sks._data[index, loop_start + i, sks._slices['position']] = key_data[loop_start + i].co
                            try:
                                radius, tilt = key_data[loop_start + i].radius, key_data[loop_start + i].tilt
                            except:
                                radius, tilt = 1., 0.

                            sks._data[index, loop_start + i, sks._slices['radius']] = radius
                            sks._data[index, loop_start + i, sks._slices['tilt']]   = tilt

            # ----- Only BEZIER or non Bezier

            else:
                key_data.foreach_get('co', v_array)
                sks.position[index] = np.reshape(v_array, (nverts, 3))

                key_data.foreach_get('radius', f_array)
                sks.radius[index] = f_array

                key_data.foreach_get('tilt', f_array)
                sks.tilt[index] = f_array

                if has_bezier:
                    key_data.foreach_get('handle_left', v_array)
                    sks.handle_left[index] = np.reshape(v_array, (nverts, 3))

                    key_data.foreach_get('handle_right', v_array)
                    sks.handle_right[index] = np.reshape(v_array, (nverts, 3))

        return sks

    # -----------------------------------------------------------------------------------------------------------------------------
    # Write the shapes in an existing curve object

    def to_curve_object(self, spec):

        from npblender.core.curve import Curve

        obj = blender.get_object(spec)
        curve = Curve.FromObject(obj)

        if self.clear:
            blender.shape_keys_clear(obj)

        is_mix     = curve.has_mix_types
        has_bezier = curve.has_bezier
        nverts     = self.points_count

        if not is_mix:
            v_array = np.empty(nverts*3, float)
            f_array = np.empty(nverts, float)

        # ----------------------------------------------------------------------------------------------------
        # Loop on the shapekeys

        for index in range(len(self)):

            key_data = blender.get_key_block(obj, index, create=True, name=self.key_name).data

            # ----- Mix types : we must loop on the splines

            if is_mix:
                for curve_type, loop_start, loop_total in zip(curve.splines.curve_type, curve.splines.loop_start, curve.splines.loop_total):

                    # ----- Bezier

                    if curve_type == blender.BEZIER:

                        for i in range(loop_total):
                            key_data[loop_start + i].co = self._data[index, loop_start + i, self._slices['position']]
                            key_data[loop_start + i].handle_left = self._data[index, loop_start + i, self._slices['handle_left']]
                            key_data[loop_start + i].handle_right = self._data[index, loop_start + i, self._slices['handle_right']]

                            key_data[loop_start + i].radius = self._data[index, loop_start + i, self._slices['radius']]
                            key_data[loop_start + i].tilt   = self._data[index, loop_start + i, self._slices['tilt']]

                    # ----- Non Bezier

                    else:

                        for i in range(loop_total):
                            key_data[loop_start + i].co = self._data[index, loop_start + i, self._slices['position']]
                            if hasattr(key_data[loop_start + i], 'radius'):
                                key_data[loop_start + i].radius = self._data[index, loop_start + i, self._slices['radius']]
                                key_data[loop_start + i].tilt = self._data[index, loop_start + i, self._slices['tilt']]

            # ----- Only BEZIER or non Bezier

            else:
                np.reshape(v_array, (nverts, 3))[:] = self.position[index]
                key_data.foreach_set('co', v_array)

                f_array[:] = self.radius[index]
                key_data.foreach_set('radius', f_array)

                f_array[:] = self.tilt[index]
                key_data.foreach_set('tilt', f_array)

                if has_bezier:
                    np.reshape(v_array, (nverts, 3))[:] = self.handle_left[index]
                    key_data.foreach_set('handle_left', v_array)

                    np.reshape(v_array, (nverts, 3))[:] = self.handle_right[index]
                    key_data.foreach_set('handle_right', v_array)

        return obj

    # ====================================================================================================
    # Interpolation (default is absolute)

    def interpolate(self, t, relative=False, extrapolation='CLIP', smooth=1):
        """ Interpolation.

        Use rel_interpolate for relative interpolation.

        Smooth define the interpolation on each interval:
            - integer : Degree of BSpline interpolation
            - str     : Name of an interpolation function
            - function : function to use

        Arguments
        ---------
            - t (float or array of floats) : interpolation factor in [0, 1]
            - extrapolation (str in 'CLIP', 'CYCLIC', 'BACK' = 'CLIP') : extrapolation mode
            - smooth (function = None) : smooth function

        Returns
        -------
            - array of floats
        """

        # ---------------------------------------------------------------------------
        # Only one shape

        if len(self) == 1:
            return np.resize(self._data, np.shape(t) + np.shape(self._data)[1:])

        # ---------------------------------------------------------------------------
        # Compute factors between 0 and 1 depending upon the mode

        t_shape = np.shape(t)
        single = t_shape == ()
        if single:
            t = np.reshape(t, (1,))

        if extrapolation == 'CLIP':
            factors = np.clip(t, 0, 1).astype(float)

        elif extrapolation == 'CYCLIC':
            factors = np.array(t) % 1.

        elif extrapolation == 'BACK':
            factors = (2*np.array(t)) % 2.
            i_back = factors > 1
            factors[i_back] = 2 - factors[i_back]

        else:
            raise AttributeError(f"ShapeKeys.interpolate error: invalid extrapolation '{extrapolation}'.")

        # ---------------------------------------------------------------------------
        # Smooth

        degree = None
        use_cubic = False
        if smooth is None:
            degree = 1

        elif isinstance(smooth, (int, np.int64, np.int32)):
            degree = smooth

        elif hasattr(smooth, '__call__'):
            factors = smooth(factors)

        elif isinstance(smooth, str):
            if smooth == 'CUBIC':
                use_cubic = True
            else:
                factors = maprange(factors, easing=smooth)

        else:
            raise AttributeError(f"ShapeKeys.interpolate error: smooth parameter must be a function or a string, not {type(smooth).__name__}.")

        # ---------------------------------------------------------------------------
        # Interpolation

        if degree is None:

            fs = factors*(len(self) - 1)
            inds = np.floor(fs).astype(int)
            inds[inds == len(self) - 1] = len(self) - 2
            p = np.reshape(fs - inds, np.shape(fs) + (1, 1))

            verts = self._data[inds]*(1 - p) + self._data[inds + 1]*p

        elif use_cubic:
            verts = CubicSpline(np.linspace(0, 1, self.points_count), self._data, extrapolate=False)(factors)

        else:

            n = len(self)
            k = smooth

            dx = 1/(n - 1)
            t = np.linspace(-k*dx, 1 + k*dx, n + k + 1)

            verts = BSpline(t, self._data, k=k, extrapolate=False)(factors)

        # ---------------------------------------------------------------------------
        # Done

        if single:
            return verts[0]
        else:
            return verts

    # ====================================================================================================
    # Relative interpolation

    def rel_interpolate(self, weights, smooth=1):
        """ Relative interpolation.

        Use interpolate for absolute interpolation.

        The number of weights must be equal to the number of shapes minus 1.
        The interpolation mixes the differences of each shape with the first one.

        Smooth define the interpolation on each interval:
            - integer : Degree of BSpline interpolation
            - str     : Name of an interpolation function
            - function : function to use

        Arguments
        ---------
            - weights (array of floats or array of arrays of flaotsd) : interpolation factor in [0, 1]
            - extrapolation (str in 'CLIP', 'CYCLIC', 'BACK' = 'CLIP') : extrapolation mode
            - smooth (function = None) : smooth function

        Returns
        -------
            - array of floats
        """

        # ---------------------------------------------------------------------------
        # Check the validity of the array of weights

        ok     = True
        single = False
        if np.shape(weights) == ():
            if len(self) != 2:
                ok = False
            single = True
            weights = np.reshape(weight, (1, 1))
        else:
            if np.shape(weights)[-1] != len(self) - 1:
                ok = False

        if not ok:
            raise RuntimeError(f"ShapeKeys.rel_interpolate error: the shape of the array of weights is not valid {np.shape(weights)}, expected (n, {len(self)-1})")

        weights = np.clip(weights, 0, 1)

        # ---------------------------------------------------------------------------
        # Smooth

        degree = None
        use_cubic = False
        if smooth is None:
            degree = 1

        elif isinstance(smooth, (int, np.int64, np.int32)):
            degree = smooth

        elif hasattr(smooth, '__call__'):
            weights = smooth(weights)

        elif isinstance(smooth, str):
            if smooth == 'CUBIC':
                use_cubic = True
            else:
                weights = maprange(weights, easing=smooth)

        else:
            raise AttributeError(f"ShapeKeys.rel_interpolate error: smooth parameter must be a function or a string, not {type(smooth).__name__}.")

        # ---------------------------------------------------------------------------
        # Relative interpolation:
        # basis shape + weights * differences

        verts = self._data[0] + np.sum(
                    (self._data[1:] - self._data[0])*np.reshape(weights, np.shape(weights) + (1, 1)),
                    axis = - 3)

        # ---------------------------------------------------------------------------
        # Done

        if single:
            return verts[0]
        else:
            return verts

    # ====================================================================================================
    # Instantiate with absolute interpolation weights

    def instantiate(self, geometry, t, extrapolation='CLIP', smooth=1):

        verts = self.interpolate(t, extrapolation=extrapolation, smooth=smooth)

        count = 1 if len(np.shape(verts)) == 2 else len(verts)
        n = self.points_count

        geos = geometry*count

        #print(f"SHAPEKEYS: {count = }, {n = }, {geos = }, {np.shape(self.get_interpolated_value(verts, 'position')) = }")
        for name in self._attributes.keys():
            setattr(geos.points, name, self.get_interpolated_value(verts, name))

        return geos

    # ====================================================================================================
    # Instantiate with relative interpolation weights

    def rel_instantiate(self, geometry, weights, smooth=1):

        verts = self.rel_interpolate(weights, smooth=smooth)

        count = 1 if len(np.shape(verts)) == 2 else len(verts)
        n = self.points_count

        geos = geometry*count
        for name in self._attributes.keys():
            setattr(geos.points, name, self.get_interpolated_value(verts, name))

        return geos

# ====================================================================================================
# Demo shape keys

def demo(seed=0):

    from npblender.core.mesh import Mesh
    from npblender.core.curve import Curve


    rng = np.random.default_rng(seed)

    # ====================================================================================================
    # Instantiate

    def instantiate(geo, sks, plural_name, location=(0, 0, 0)):

        # ----- Instantiate with absolute interpolation

        count = 10
        t = rng.uniform(0, 1, count)
        locs = rng.uniform(-10, 10, (count, 3))

        geos = sks.instantiate(geo, t, smooth=1)
        geos.points.translate(locs)
        obj = geos.to_object(plural_name)
        obj.location = location

        # ----- Instantiate with relative interpolation

        count = 10
        w = rng.uniform(0, 1, (count, len(sks)-1))
        locs = rng.uniform(-10, 10, (count, 3))

        geos = sks.rel_instantiate(geo, w, smooth=1)
        geos.points.translate(locs)
        obj = geos.to_object(f"{plural_name} Rel")
        obj.location = location
        obj.location.x += 30

    # ====================================================================================================
    # Mesh

    cube = Mesh.Cube()
    sks = ShapeKeys.FromGeometry(cube, 4)

    a = sks.get_value(1, 'position')
    a[a[:, 2] > .5] *= (.1, .1, 1)

    a = sks.get_value(2, 'position')
    a[a[:, 1] < -.5] *= (2, 2, 1)

    a = sks.get_value(3, 'position')
    a[a[:, 0] > .5] += (2, 0, 0)

    obj = cube.to_object("SKS Cube", shade_smooth=False)
    sks.to_object(obj)

    instantiate(cube, sks, "SKS Cubes", location=(0, 0, 0))

    # ====================================================================================================
    # Curve

    curve = Curve.BezierSegment()
    curve.points.position[:, 2] -= 1

    curve.join(Curve.Spiral())

    sks = ShapeKeys.FromGeometry(curve, 4)

    a = sks.get_value(1, 'position')
    a[a[:, 2] > .5] *= (.1, .1, 1)

    a = sks.get_value(2, 'position')
    a[a[:, 1] < -.5] *= (2, 2, 1)

    a = sks.get_value(3, 'position')
    a[a[:, 0] > .5] += (2, 0, 0)

    obj = curve.to_object("SKS Curve")
    sks.to_object(obj)

    instantiate(curve, sks, "SKS Curves", location=(0, 30, 0))


# =============================================================================================================================
# Interpolation

if __name__ == '__main__':

    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.interpolate import BSpline
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    count = 10

    x = np.linspace(0, 1, count)
    y = rng.uniform(0, 1, count)

    t = np.linspace(-.5, 1.5, 100)

    # BSpline

    for k in range(4):

        if True:
            dx = 1/(count-1)
            x1 = np.linspace(-dx*k, 1+k*dx, count + k + 1)

            y1 = y

        else:
            dx = 1/(count-1)
            x1 = np.linspace(-dx*k, 1+dx*k, count+2*k)

            y1 = np.resize(y, count+k)
            y1[0] = y[0]
            y1[-1] = y[-1]
            if k == 1:
                y1[:-k] = y
            elif k == 2:
                y1[1:-1] = y
            elif k == 3:
                y1[1:-2] = y
            else:
                y1[:len(y)] = y


        bs = BSpline(x1, y1, k=k, extrapolate=False)

        fig, ax = plt.subplots()

        ax.plot(t, bs(t))
        ax.plot(x, y, 'xr')

        plt.title(f"Degree {k}")

        plt.show()

    # CubicSpline

    cs = CubicSpline(x, y, extrapolate=False)

    fig, ax = plt.subplots()

    ax.plot(t, cs(t))
    ax.plot(x, y, 'xr')

    plt.title(f"CubicSpline")

    plt.show()
