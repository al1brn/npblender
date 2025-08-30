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
Module Name: constants
Author: Alain Bernard
Version: 0.1.0
Created: 2025-07-21
Last updated: 2025-08-29

Summary:
This module defines core constants used throughout the `npblender` package for consistent geometry
attribute management in Blender and NumPy.

It includes:
  - Supported geometry domains used in Blender (e.g., POINT, EDGE, FACE)
  - Supported attribute types with metadata (NumPy dtype, shape, default Blender name)
  - Mapping between Blender's `bl_type` and internal type keys

These constants are designed to ensure compatibility between Geometry Nodes attributes in Blender
and NumPy structured arrays used in the `FieldArray` and `Attributes` classes.

Usage example:
    >>> from constants import bfloat, bint

"""

__all__ = [
    'PI', 'TAU',
    'SPLINE_TYPES', 'BEZIER', 'POLY', 'NURBS', 
    'bfloat', 'bint', 'bbool',
    'DOMAINS', 'TYPES', 'BL_TYPES',    
    ]

import numpy as np
from enum import Enum

# =============================================================================================================================
# Helpers
# =============================================================================================================================

PI = np.pi
TAU = 2*np.pi

# =============================================================================================================================
# Splines 
# =============================================================================================================================

SPLINE_TYPES = ['BEZIER', 'POLY', 'NURBS']

BEZIER = 0
POLY   = 1
NURBS  = 2

# =============================================================================================================================
# np.ndarray dtypes
# =============================================================================================================================

bfloat = np.float32
bint   = np.int32
bbool  = np.bool_

# =============================================================================================================================
# Enums
# =============================================================================================================================

class CodeLabelEnum(Enum):

    def __init__(self, code, label):
        self.code = code
        self.label = label

    def __str__(self):
        return self.label

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, cls):
            return value
        
        if isinstance(value, str):
            for member in cls:
                if member.label == value.upper():
                    return member
        elif isinstance(value, (int, np.int32, np.int64)):
            for member in cls:
                if member.code == value:
                    return member
                
        raise ValueError(
            f"Invalid value for {cls.__name__!r} : {value!r}. "
            f"Authorized values are : {[m.label for m in cls]}"
            )

class FillCap(CodeLabelEnum):
    NONE = (0, "NONE")
    NGON = (1, "NGON")
    CAPS = (2, "FANS")

class DomainName(CodeLabelEnum):
    UNDEFINED = (0, 'UNDEFINED')
    POINT     = (1, 'POINT')
    EDGE      = (2, 'EDGE')
    CORNER    = (3, 'CORNER')
    FACE      = (4, 'FACE')
    CURVE     = (5, 'CURVE')
    INSTANCE  = (6, 'INSTANCE')

class SplineType(CodeLabelEnum):
    BEZIER = (0, "BEZIER")
    CURVE  = (1, "CURVE")
    NURBS  = (2, "NURBS")


# =============================================================================================================================
# Domains
# =============================================================================================================================

DOMAINS = ('POINT', 'EDGE', 'FACE', 'CORNER', 'CURVE', 'INSTANCE')


TYPES = {
    'FLOAT'      : {'dtype': bfloat,  'size':  1, 'shape': (),     'name': 'value',      'bl_type': 'FLOAT'},
    'INT'        : {'dtype': bint,    'size':  1, 'shape': (),     'name': 'value',      'bl_type': 'INT'},
    'VECTOR'     : {'dtype': bfloat,  'size':  3, 'shape': (3,),   'name': 'vector',     'bl_type': 'FLOAT_VECTOR'},
    'COLOR'      : {'dtype': bfloat,  'size':  4, 'shape': (4,),   'name': 'color',      'bl_type': 'FLOAT_COLOR'},
    'BYTE_COLOR' : {'dtype': 'u1',    'size':  4, 'shape': (4,),   'name': 'color_srgb', 'bl_type': 'BYTE_COLOR'},
    'STRING'     : {'dtype': 'U128',  'size':  0, 'shape': (),     'name': 'value',      'bl_type': 'STRING'},
    'BOOLEAN'    : {'dtype': bbool,   'size':  1, 'shape': (),     'name': 'value',      'bl_type': 'BOOLEAN'},
    'FLOAT2'     : {'dtype': bfloat,  'size':  2, 'shape': (2,),   'name': 'vector',     'bl_type': 'FLOAT2'},
    'INT8'       : {'dtype': np.int8, 'size':  1, 'shape': (),     'name': 'value',      'bl_type': 'INT8'},
    'INT32_2D'   : {'dtype': bint,    'size':  2, 'shape': (2,),   'name': 'value',      'bl_type': 'INT32_2D'},
    'QUATERNION' : {'dtype': bfloat,  'size':  4, 'shape': (4,),   'name': 'value',      'bl_type': 'QUATERNION'},
    'MATRIX'     : {'dtype': bfloat,  'size': 16, 'shape': (4, 4), 'name': 'value',      'bl_type': 'FLOAT4X4'},
    }

BL_TYPES = {item['bl_type']: key for key, item in TYPES.items()}

# bl_type can be used when set by geometry nodes
for bl_key, key in BL_TYPES.items():
    if bl_key not in TYPES:
        TYPES[bl_key] = TYPES[key]

