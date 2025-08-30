from .camera import Camera

from .engine import engine, Animation
from .fieldarray import FieldArray
from .domain import Point, Vertex, ControlPoint, Corner, Face, Edge, Spline
from .shapekeys import ShapeKeys

from .geometry import Geometry
from .cloud import Cloud
from .mesh import Mesh
from .curve import Curve
from .instances import Instances, Meshes
from .multiresgrid import MultiResGrid

from .simulation import Simulation

from . import constants
from . import blender
from . import maths

VERSION         = (3, 0, 0)
BLENDER_VERSION = (4, 5, 0)

__version__ = ".".join(map(str, VERSION))
__blender_version__ = ".".join(map(str, BLENDER_VERSION))

__all__ = [
    "VERSION",
    "BLENDER_VERSION",
    "Camera",
    "engine", "Animation",
    "FieldArray",
    "Point", "Vertex", "ControlPoint", "Corner", "Face", "Edge", "Spline",
    "ShapeKeys",
    "Geometry", "Cloud", "Mesh", "Curve",
    "Instances", "Meshes",
    "MultiResGrid",
    "Simulation",
    "constants",
    "blender",
    "maths",
]


