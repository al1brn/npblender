# addon/__init__.py

bl_info = {
    "name": "npblender",
    "author": "Alain Bernard",
    "version": (3, 0, 0),
    "blender": (4, 5, 0),   # version minimale de Blender ciblée
    "location": "Anywhere (Python API)",
    "description": "NumPy-based geometry and Blender utilities",
    "category": "Object",
}

from . import npblender

def register():
    npblender.engine.register()

def unregister():
    npblender.engine.unregister()

