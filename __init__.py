"""
This file is part of the geonodes distribution (https://github.com/al1brn/npblender).
Copyright (c) 2025 Alain Bernard.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------
numpy for Blender
-----------------------------------------------------


Created : 2022/06/29
Updated : 2024/10/15
Updated : 2025/07/18
"""

__author__ = "Alain Bernard"
__email__  = "lesideesfroides@gmail.com"
__copyright__ = "Copyright (c) 2025, Alain Bernard"
__license__ = "GNU GPL V3"
__version__ = "2.0.0"

from . core import blender


"""

# -----------------------------------------------------------------------------------------------------------------------------
# Constants

pi   = 3.141592653589793
tau  = 6.283185307179586
half_pi = pi/2

from .core import blender

version         = (2, 1, 0)
blender_version = (4, 1, 0)

# -----------------------------------------------------------------------------------------------------------------------------
# Dynamic install

import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade", "--user"])

# -----------------------------------------------------------------------------------------------------------------------------
# Import / install modules

try:
    import scipy
except:
    install_package("scipy")
print(f"scipy version {scipy.__version__}, version lower than 1.13.0 could crash.")

try:
    import numba
except:
    install_package("numba")

try:
    import PIL
except:
    install_package("Pillow")

try:
    from scipy.spatial.transform import RotationSpline
    from scipy.spatial.transform import Rotation
    from scipy.spatial import KDTree
    from scipy.interpolate import CubicSpline, BSpline, make_interp_spline, splder

    OK_SCIPY = True

except:
    print("CAUTION: Module 'scipy' not installed in Blender. Some features will crash...")
    OK_SCIPY = False

    class ScipyMissing:
        def __init__(self, *args, **kwargs):
            raise Exception("Module scipy is not installed in Blender")

    RotationSpline     = ScipyMissing
    Rotation           = ScipyMissing
    KDTree             = ScipyMissing
    CubicSpline        = ScipyMissing
    BSpline            = ScipyMissing
    make_interp_spline = ScipyMissing
    splder             = ScipyMissing

if True: # RELOAD

    # ----- Reload

    from importlib import reload

    from .core import engine

    from .maths import splinesmaths
    from .maths import distribs
    from .maths import functions
    from .maths import noise
    from .maths import primes
    from .maths import transformations

    reload(blender)
    reload(engine)

    reload(splinesmaths)
    reload(distribs)
    reload(functions)
    reload(noise)
    reload(primes)
    reload(transformations)

# =============================================================================================================================
# Python geometry

# ----- Maths

# Maths modules

from .maths import splinesmaths, distribs

# Maths classes

from .maths.functions import Easing, Function, keyed
from .maths.noise import Noise
from .maths.transformations import normalize, axis_vector, get_plane, rotation_to, tracker, axis_index, angle_with, Transformations
from .maths import field

# ----- Core

if True:
    from .core import camera

    from .core import domain
    from .core import geometry
    from .core import cloud
    from .core import mesh
    from .core import curve
    #from .core import volume
    from .core import text
    from .core import instances
    from .core import shapekeys
    from .core import parameters

    from .core import bingrid
    from .core import simulation
    from .core import particles

    reload(camera)

    reload(domain)
    reload(geometry)
    reload(cloud)
    reload(mesh)
    reload(curve)
    #reload(volume)
    reload(text)
    reload(instances)
    reload(shapekeys)
    reload(parameters)

    reload(bingrid)
    reload(simulation)
    reload(particles)


from .core.camera import Camera

from .core.attributes import AttrVectors, DynamicRecArray

from .core.geometry import Geometry
from .core.cloud import Cloud
from .core.mesh import Mesh
from .core.curve import Curve
#from .core.volume import Volume
from .core.text import Text
from .core.instances import Instances
from .core.shapekeys import ShapeKeys
from .core.parameters import Parameters
from .core.overlay import Overlay

from .core.bingrid import BinGrid
from .core.engine import Animation
from .core.simulation import Simulation
from .core.particles import Particles


engine.register()


"""