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
from . import blender

__all__ = ["Custom"]

PROP_SUBTYPES = ['NONE', 'FILE_PATH', 'DIR_PATH', 'FILE_NAME', 'BYTE_STRING', 'PASSWORD', 'PIXEL',
                 'UNSIGNED', 'PERCENTAGE', 'FACTOR', 'ANGLE', 'TIME', 'TIME_ABSOLUTE', 'DISTANCE',
                 'DISTANCE_CAMERA', 'POWER', 'TEMPERATURE', 'COLOR', 'TRANSLATION', 'DIRECTION',
                 'VELOCITY', 'ACCELERATION', 'MATRIX', 'EULER', 'QUATERNION', 'AXISANGLE', 'XYZ',
                 'XYZ_LENGTH', 'COLOR_GAMMA', 'COORDINATES', 'LAYER', 'LAYER_MEMBER']

class Custom:
    __array_priority__ = 10_000
    
    def __init__(self, name, value, target='SCENE', tip="", **attrs):
        
        if name is None or value is None:
            return
        
        attributes = {**attrs}
        if 'description' not in attributes:
            attributes['description'] = tip
        if 'subtype' not in attributes:
            attributes['subtypes'] = 'NONE'
        
        self._create(name, value, target, attributes)
    
    # ====================================================================================================
    # Create the entry in a data block
    # ====================================================================================================
    
    def _create(self, name, default, target, attrs):
        
        from npblender import blender
        
        # ----- Target data
        
        data = None
        
        if isinstance(target, str):            
            if target == 'SCENE':
                data = bpy.context.scene
                
        if data is None:
            data = blender.get_object(target, halt=True)
            
        # ----- Create / update
        
        entry = data.get(name)
        if entry is None:
            data[name] = default
        else:
            cur_attrs = data.id_properties_ui(name).as_dict()
            if cur_attrs["default"] != default or cur_attrs['subtype'] != attrs['subtype']:
                data[name] = default

        id_prop = data.id_properties_ui(name)        
        id_prop.update(**{k: v for k, v in attrs.items() if v is not None})
        
        # ----- Store
        
        self._data = data
        self._name = name
        self._default = default
        
    # ====================================================================================================
    # Basic types
    # ====================================================================================================
        
        
    # ----------------------------------------------------------------------------------------------------
    # Integer
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def new_int(cls, name, value=0, target='SCENE', min=None, max=None, tip="", length=1, subtype=None):

        if hasattr(value, '__len__'):
            length = len(value)
            value = [int(v) for v in value][:32]
        else:
            value = int(value)
            if length > 1:
                value = ([value]*length)[:32]

        return cls(name, value=value, target=target, tip=tip, min=min, max=max, subtype=subtype)

    # ----------------------------------------------------------------------------------------------------
    # Float
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def new_float(cls, name, value=0.0, target='SCENE', min=None, max=None, tip="", length=1, subtype=None):

        if hasattr(value, '__len__'):
            length = len(value)
            value = [float(v) for v in value][:32]
        else:
            value = float(value)
            if length > 1:
                value = ([value]*length)[:32]

        return cls(name, value=value, target=target, tip=tip, min=min, max=max, subtype=subtype)

    # ----------------------------------------------------------------------------------------------------
    # Boolean
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def new_bool(cls, name, value=False, target='SCENE', tip="", length=1):

        if hasattr(value, '__len__'):
            length = len(value)
            value = [bool(v) for v in value][:32]
        else:
            value = bool(value)
            if length > 1:
                value = ([value]*length)[:32]

        return cls(name, value=value, target=target, tip=tip)

    # ----------------------------------------------------------------------------------------------------
    # Str
    # ----------------------------------------------------------------------------------------------------

    @classmethod
    def new_str(cls, name, value="", target='SCENE', tip=""):
        return self.new(name, value=str(value), target=target, tip=tip)

    # ====================================================================================================
    # Advances types
    # ====================================================================================================

    @classmethod
    def new_factor(cls, name, value=0, target='SCENE', min=0, max=1, tip="", length=1):
        return cls.new_float(name, value=value, target=target, min=min, max=max, tip=tip, length=length, subtype='FACTOR')

    @classmethod
    def new_percentage(cls, name, value=0, target='SCENE', min=0, max=100, tip="", length=1):
        return cls.new_float(name, value=value, target=target, min=min, max=max, tip=tip, length=length, subtype='PERCENTAGE')

    @classmethod
    def new_angle(cls, name, value=0, target='SCENE', min=None, max=None, tip="", length=1):
        return cls.new_float(name, value=value, target=target, min=min, max=max, tip=tip, length=length, subtype='ANGLE')

    @classmethod
    def new_time(cls, name, value=0, target='SCENE', min=None, max=None, tip="", length=1):
        return cls.new_float(name, value=value, target=target, min=min, max=max, tip=tip, length=length, subtype='TIME')

    @classmethod
    def new_distance(cls, name, value=0, target='SCENE', min=0, max=10000, tip="", length=1):
        return cls.new_float(name, value=value, target=target, min=min, max=max, tip=tip, length=length, subtype='DISTANCE')

    @classmethod
    def new_color(cls, name, value=0, target='SCENE', min=0, max=1, tip=""):
        return cls.new_float(name, value=value, min=min, max=max, tip=tip, length=3, subtype='COLOR')

    @classmethod
    def new_coordinates(cls, name, value=0, target='SCENE', min=None, max=None, tip=""):
        return cls.new_float(name, value=value, target=target, min=min, max=max, tip=tip, length=3, subtype='COORDINATES')

    @classmethod
    def new_translation(cls, name, value=0, target='SCENE', min=None, max=None, tip=""):
        return cls.new_float(name, value=value, target=target, min=min, max=max, tip=tip, length=3, subtype='TRANSLATION')

    @classmethod
    def new_direction(cls, name, value=(0, 0, 1), target='SCENE', tip=""):
        return cls.new_float(name, value=value, target=target, tip=tip, length=3, subtype='DIRECTION')

    @classmethod
    def new_velocity(cls, name, value=0, target='SCENE', min=None, max=None, tip=""):
        return cls.new_float(name, value=value, target=target, min=min, max=max, tip=tip, length=3, subtype='VELOCITY')

    @classmethod
    def new_acceleration(cls, name, value=0, target='SCENE', min=None, max=None, tip=""):
        return cls.new_float(name, value=value, target=target, min=min, max=max, tip=tip, length=3, subtype='ACCELERATION')

    @classmethod
    def new_matrix(cls, name, value=[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.], target='SCENE', tip=""):
        return cls.new_float(name, value=value, target=target, tip=tip, length=16, subtype='MATRIX')

    @classmethod
    def new_euler(cls, name, value=0, target='SCENE', tip=""):
        return cls.new_float(name, value=value, target=target, tip=tip, length=3, subtype='EULER')

    @classmethod
    def new_quaternion(cls, name, value=[1, 0, 0, 0], target='SCENE', tip=""):
        return cls.new_float(name, value=value, target=target, tip=tip, length=4, subtype='QUATERNION')

    @classmethod
    def new_axisangle(cls, name, value=[0, 0, 0, 1], target='SCENE', tip=""):
        return cls.new_float(name, value=value, target=target, tip=tip, length=4, subtype='AXISANGLE')

    # ====================================================================================================
    # Function as alternative
    # ====================================================================================================
        
    @classmethod
    def function(cls, func):
        c = cls(None, None)
        c._func = func
        return c
    
    @property
    def is_function(self):
        return '_func' in self.__dict__
    
    # ====================================================================================================
    # Access to attributes
    # ====================================================================================================
        
    def __getattr__(self, name):
        return getattr(self.value, name)
    
    def __repr__(self):
        if self.is_function:
            return f"Custom( function {self._func.__name__})"
        else:
            data_name = getattr(self._data, 'name', type(self._data).__name__)    
            return f"Custom( {data_name}[{self._name}] = {self.value!r})"

    # ====================================================================================================
    # As a value
    # ====================================================================================================
        
    # ----- Value
    @property
    def value(self):
        if self.is_function:
            return self._func(bpy.context.scene.frame_current)
        else:
            return self._data[self._name]
    
    @value.setter
    def value(self, val):
        if self.is_function:
            raise AttributeError(f"Custom value {repr(self)} is read only")
        self._data[self._name] = val
        
    @staticmethod
    def _val(x):
        return x.value if isinstance(x, Custom) else x
    
    @staticmethod
    def set(param, value):
        if isinstance(param, Custom):
            param.value = value
        return param
    
    # --- If array
    def __len__(self): 
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __contains__(self, item):
        return item in self.value

    def __getitem__(self, i):
        return self.value[i]

    def __setitem__(self, i, v):
        val = self.value
        try:
            # tente la modif in-place si possible
            val[i] = v
            # pour certaines ID props, réassigner force la mise à jour
            self.value = val
        except TypeError:
            # valeur immuable : reconstruire si tu veux le supporter
            if isinstance(val, tuple):
                lst = list(val)
                lst[i] = v
                self.value = tuple(lst)
            else:
                raise
                
    # ====================================================================================================
    # Dunder
    # ====================================================================================================
                
         
    # --- conversions ---
    def __str__(self): return str(self.value)
    def __int__(self):  return int(self.value)
    def __float__(self): return float(self.value)    
    def __bool__(self): return bool(self.value)
    def __index__(self): return int(self.value)
    
    # --- comparaisons ---
    def __eq__(self, o): return self.value == self._val(o)
    def __ne__(self, o): return self.value != self._val(o)
    def __lt__(self, o): return self.value <  self._val(o)
    def __le__(self, o): return self.value <= self._val(o)
    def __gt__(self, o): return self.value >  self._val(o)
    def __ge__(self, o): return self.value >= self._val(o)

    # --- unaires & utilitaires ---
    def __neg__(self):   return -self.value
    def __pos__(self):   return +self.value
    def __abs__(self):   return abs(self.value)
    def __round__(self, ndigits=None):
        return round(self.value, ndigits) if ndigits is not None else round(self.value)

    # --- addition ---
    def __add__(self, o):      return self.value + self._val(o)
    def __radd__(self, o):     return self._val(o) + self.value
    def __iadd__(self, o):
        self.value = self.value + self._val(o); return self

    # --- soustraction ---
    def __sub__(self, o):      return self.value - self._val(o)
    def __rsub__(self, o):     return self._val(o) - self.value
    def __isub__(self, o):
        self.value = self.value - self._val(o); return self

    # --- multiplication ---
    def __mul__(self, o):      return self.value * self._val(o)
    def __rmul__(self, o):     return self._val(o) * self.value
    def __imul__(self, o):
        self.value = self.value * self._val(o); return self

    # --- division réelle ---
    def __truediv__(self, o):  return self.value / self._val(o)
    def __rtruediv__(self, o): return self._val(o) / self.value
    def __itruediv__(self, o):
        self.value = self.value / self._val(o); return self

    # --- division entière ---
    def __floordiv__(self, o):  return self.value // self._val(o)
    def __rfloordiv__(self, o): return self._val(o) // self.value
    def __ifloordiv__(self, o):
        self.value = self.value // self._val(o); return self

    # --- modulo ---
    def __mod__(self, o):    return self.value % self._val(o)
    def __rmod__(self, o):   return self._val(o) % self.value
    def __imod__(self, o):
        self.value = self.value % self._val(o); return self

    # --- puissance ---
    def __pow__(self, o):    return self.value ** self._val(o)
    def __rpow__(self, o):   return self._val(o) ** self.value
    def __ipow__(self, o):
        self.value = self.value ** self._val(o); return self

    # --- matmul (utile si la valeur supporte @ : matrices/vecteurs) ---
    def __matmul__(self, o):    return self.value @ self._val(o)
    def __rmatmul__(self, o):   return self._val(o) @ self.value
    def __imatmul__(self, o):
        self.value = self.value @ self._val(o); return self

    # --- bitwise (pour int/bool) ---
    def __and__(self, o):    return self.value & self._val(o)
    def __rand__(self, o):   return self._val(o) & self.value
    def __iand__(self, o):
        self.value = self.value & self._val(o); return self

    def __or__(self, o):     return self.value | self._val(o)
    def __ror__(self, o):    return self._val(o) | self.value
    def __ior__(self, o):
        self.value = self.value | self._val(o); return self

    def __xor__(self, o):    return self.value ^ self._val(o)
    def __rxor__(self, o):   return self._val(o) ^ self.value
    def __ixor__(self, o):
        self.value = self.value ^ self._val(o); return self

    def __lshift__(self, o):   return self.value << self._val(o)
    def __rlshift__(self, o):  return self._val(o) << self.value
    def __ilshift__(self, o):
        self.value = self.value << self._val(o); return self

    def __rshift__(self, o):   return self.value >> self._val(o)
    def __rrshift__(self, o):  return self._val(o) >> self.value
    def __irshift__(self, o):
        self.value = self.value >> self._val(o); return self
    
    # --- conversions complémentaires ---
    def __complex__(self):
        return complex(self.value)

    def __trunc__(self):
        import math
        return math.trunc(self.value)

    def __floor__(self):
        import math
        return math.floor(self.value)

    def __ceil__(self):
        import math
        return math.ceil(self.value)

    def keyframe_insert(self, **kw):
        dp = f'["{self.name}"]'
        self.data.keyframe_insert(data_path=dp, **kw)
        
    
    
       


# ====================================================================================================
# Parameters

class Parameters_OLD(dict):

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

        return self[name]

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
