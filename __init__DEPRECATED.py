# ---------------------------------------------------------------------------
# Expose sub folder npblender
# ---------------------------------------------------------------------------

from . import npblender as _inner
from .npblender import *

# Build __all__
__all__ = getattr(_inner, "__all__", [n for n in dir() if not n.startswith("_")])

# ---------------------------------------------------------------------------
# Expose sub packages
# ---------------------------------------------------------------------------

import importlib as _il
import sys as _sys

# maths
_maths = _il.import_module('.npblender.maths', __name__)
maths = _maths
_sys.modules[__name__ + '.maths'] = _maths

# blender
_blender = _il.import_module('.npblender.blender', __name__)
blender = _blender
_sys.modules[__name__ + '.blender'] = _blender

__all__.extend(["maths", "blender"])
