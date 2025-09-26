
from .camera import Camera

from .enginemod import engine, Animation, Engine
from .fieldarray import FieldArray
from .domain import Domain, Point, Vertex, ControlPoint, Corner, Face, Edge, Spline
from .shapekeys import ShapeKeys
from .parameters import Custom

from .geometry import Geometry
from .cloud import Cloud
from .mesh import Mesh
from .curve import Curve
from .instances import Instances, Meshes
from .multiresgrid import MultiResGrid

from .simulation import Simulation

from . import constants
from . import blender
from . import textutils

from .maths import Rotation, Quaternion, Transformation

VERSION         = (3, 0, 0)
BLENDER_VERSION = (4, 5, 0)

__version__ = ".".join(map(str, VERSION))
__blender_version__ = ".".join(map(str, BLENDER_VERSION))

__all__ = [
    "VERSION", "BLENDER_VERSION",
    "Camera",
    "engine", "Animation", "Engine",
    "FieldArray",
    "Domain", "Point", "Vertex", "ControlPoint", "Corner", "Face", "Edge", "Spline",
    "ShapeKeys",
    "Custom",
    "Geometry", "Mesh", "Curve", "Cloud",
    "Instances", "Meshes",
    "MultiResGrid",
    "Simulation",
    "constants",
    "blender",
    "textutils",
    "Rotation", "Quaternion", "Transformation",
]

# ---------------------------------------------------------------------------
# Add from all

from . import constants
from .constants import *    
__all__.extend(list(constants.__all__))

from . import deps
from .deps import *    
__all__.extend(list(deps.__all__))

from . import text
from .text import *
__all__.extend(list(text.__all__))


# TEMP
enginemod.register()





