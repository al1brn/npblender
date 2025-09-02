#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module

Created on Sat Dec  9 09:24:40 2023

@author: alain.bernard
@email: alain@ligloo.net

-----

Custom properties used as parameters.

"""
import numpy as np
import bpy
import idprop
from npblender.core import blender

PROP_SUBTYPES = ['NONE', 'FILE_PATH', 'DIR_PATH', 'FILE_NAME', 'BYTE_STRING', 'PASSWORD', 'PIXEL',
                 'UNSIGNED', 'PERCENTAGE', 'FACTOR', 'ANGLE', 'TIME', 'TIME_ABSOLUTE', 'DISTANCE',
                 'DISTANCE_CAMERA', 'POWER', 'TEMPERATURE', 'COLOR', 'TRANSLATION', 'DIRECTION',
                 'VELOCITY', 'ACCELERATION', 'MATRIX', 'EULER', 'QUATERNION', 'AXISANGLE', 'XYZ',
                 'XYZ_LENGTH', 'COLOR_GAMMA', 'COORDINATES', 'LAYER', 'LAYER_MEMBER']

# ====================================================================================================
# Parameters

class Parameters(dict):

    def new(self, name, value, tip="", **attrs):
        """ Add a custom property definition to the dict

        Arguments
        ---------
        - name (str) : property name
        - value (any) : property value
        - tip (str = "") : pop over help text
        - **attrs : keywords valid in layout.prop method
        """
        self[name] = {'default': value, 'description': tip}
        for k, v in attrs.items():
            if v is not None:
                self[name][k] = v

        if self[name].get('subtype', None) is None:
            self[name]['subtype'] = 'NONE'

        #self.name        = name
        #self.value       = value
        #self.tip = tip
        #self.attrs       = attrs

    # ====================================================================================================
    # Basic types

    # ----------------------------------------------------------------------------------------------------
    # Integer

    def new_int(self, name, value=0, min=None, max=None, tip="", length=1, subtype=None):

        if min is None:
            min = -10000
        if max is None:
            max = 10000

        if hasattr(value, '__len__'):
            length = len(value)
            value = [int(v) for v in value][:32]
        else:
            value = int(value)
            if length > 1:
                value = ([value]*length)[:32]

        return self.new(name, value=value, tip=tip, min=min, max=max, subtype=subtype)

    # ----------------------------------------------------------------------------------------------------
    # Float

    def new_float(self, name, value=0, min=None, max=None, tip="", length=1, subtype=None):

        if min is None:
            min = -10000
        if max is None:
            max = 10000

        if hasattr(value, '__len__'):
            length = len(value)
            value = [float(v) for v in value][:32]
        else:
            value = float(value)
            if length > 1:
                value = ([value]*length)[:32]

        return self.new(name, value=value, tip=tip, min=min, max=max, subtype=subtype)

    # ----------------------------------------------------------------------------------------------------
    # Boolean

    def new_bool(self, name, value=False, tip="", length=1):

        if hasattr(value, '__len__'):
            length = len(value)
            value = [bool(v) for v in value][:32]
        else:
            value = bool(value)
            if length > 1:
                value = ([value]*length)[:32]

        return self.new(name, value=value, tip=tip)

    # ----------------------------------------------------------------------------------------------------
    # Str

    def new_str(self, name, value="", tip=""):
        return self.new(name, value=str(value), tip=tip)

    # ====================================================================================================
    # Advances types

    def new_factor(self, name, value=0, min=0, max=1, tip="", length=1):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=length, subtype='FACTOR')

    def new_percentage(self, name, value=0, min=0, max=100, tip="", length=1):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=length, subtype='PERCENTAGE')

    def new_angle(self, name, value=0, min=None, max=None, tip="", length=1):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=length, subtype='ANGLE')

    def new_time(self, name, value=0, min=None, max=None, tip="", length=1):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=length, subtype='TIME')

    def new_distance(self, name, value=0, min=0, max=10000, tip="", length=1):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=length, subtype='DISTANCE')

    def new_color(self, name, value=0, min=0, max=1, tip=""):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=3, subtype='COLOR')

    def new_coordinates(self, name, value=0, min=None, max=None, tip=""):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=3, subtype='COORDINATES')

    def new_translation(self, name, value=0, min=None, max=None, tip=""):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=3, subtype='TRANSLATION')

    def new_direction(self, name, value=(0, 0, 1), tip=""):
        return self.new_float(name, value=value, tip=tip, length=3, subtype='DIRECTION')

    def new_velocity(self, name, value=0, min=None, max=None, tip=""):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=3, subtype='VELOCITY')

    def new_acceleration(self, name, value=0, min=None, max=None, tip=""):
        return self.new_float(name, value=value, min=min, max=max, tip=tip, length=3, subtype='ACCELERATION')

    def new_matrix(self, name, value=[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.], tip=""):
        return self.new_float(name, value=value, tip=tip, length=16, subtype='MATRIX')

    def new_euler(self, name, value=0, tip=""):
        return self.new_float(name, value=value, tip=tip, length=3, subtype='EULER')

    def new_quaternion(self, name, value=[1, 0, 0, 0], tip=""):
        return self.new_float(name, value=value, tip=tip, length=4, subtype='QUATERNION')

    def new_axisangle(self, name, value=[0, 0, 0, 1], tip=""):
        return self.new_float(name, value=value, tip=tip, length=4, subtype='AXISANGLE')

    # ====================================================================================================
    # Add to a block accepting custom properties

    def to_block(self, block):
        """ Create the custom properties in an object.

        Arguments
        ---------
        - block (data block): Blender block accepting custom properties

        Returns
        -------
        - Data block : the data block with the custome properties
        """

        for name, attrs in self.items():
            cur = block.get(name)

            if cur is None:
                block[name] = attrs['default']
            else:
                cur_attrs = block.id_properties_ui(name).as_dict()
                if cur_attrs["default"] != attrs['default'] or cur_attrs['subtype'] != attrs['subtype']:
                    block[name] = attrs['default']

            id_prop = block.id_properties_ui(name)
            id_prop.update(**attrs)

        return block

    def to_object(self, spec):
        """ Create the custom properties in an object.

        Arguments
        ---------
        - spec (str of obj) : the object

        Returns
        -------
        - object : the object with created custom property
        """
        return self.to_block(blender.get_object(spec))

    def to_scene(self):
        """ Create the custom properties in the scene.

        Arguments
        ---------
        - spec (str of obj) : the object

        Returns
        -------
        - Scene : the current scene
        """
        return self.to_block(bpy.context.scene)


    @classmethod
    def All(cls):
        params = cls()

        params.new_int("int", 50, min=0, max=100, tip="Base int")
        params.new_int("int2", (20, 80), min=0, max=100, tip="Two ints")

        params.new_float("float", 50, min=0, max=100, tip="Base float")
        params.new_float("float2", (20, 80), min=0, max=100, tip="Two floats")

        params.new_bool("bool", False, tip="Base bool")
        params.new_bool("bool2", (False, True), tip="Two bools")

        params.new_str("str", "One string", tip="Str parameter")

        params.new_factor("factor")
        params.new_percentage("percentage")
        params.new_angle("angle")
        params.new_time("time")
        params.new_distance("distance")
        params.new_color("color")
        params.new_coordinates("coordinates")
        params.new_translation("translation")
        params.new_direction("direction")
        params.new_velocity("velocity")
        params.new_acceleration("acceleration")
        params.new_matrix("matrix")
        params.new_euler("euler")
        params.new_quaternion("quaternion")
        params.new_axisangle("axisangle")

        return params
