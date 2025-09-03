# __init__.py (à la racine du dossier zippé: npblender/__init__.py)

bl_info = {
    "name": "npblender",
    "author": "Alain Bernard",
    "version": (3, 0, 0),
    "blender": (4, 5, 0),
    "location": "Properties > Scene > npblender Bake Panel",
    "category": "Object",
    "description": "NumPy-based geometry tools with a Blender UI for baking and simulation control.",
}

from . import npblender as _pkg

# Import public API
from .npblender import *

__all__ = list(getattr(_pkg, "__all__", [])) + ["register", "unregister"]

# Keep access to sub-packages
npblender = _pkg

# Register

def register():
    _pkg.enginemod.register()

def unregister():
    _pkg.enginemod.unregister()
