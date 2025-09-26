from . import npblender as _npb

# Declare global npblender

import sys as _sys

_sys.modules["npblender"] = _npb

def register():
    if hasattr(_npb, "enginemod") and hasattr(_npb.enginemod, "register"):
        _npb.enginemod.register()

def unregister():
    if hasattr(_npb, "enginemod") and hasattr(_npb.enginemod, "unregister"):
        _npb.enginemod.unregister()

