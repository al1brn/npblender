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

# ----------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------

version         = (3, 0, 0)
blender_version = (4, 5, 0)

PI   = 3.141592653589793
TAU  = 6.283185307179586
HALF_PI = PI/2

# ----------------------------------------------------------------------------------------------------
# Dynamic install
# ----------------------------------------------------------------------------------------------------

import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade", "--user"])

try:
    import scipy
except:
    install_package("scipy")

try:
    import numba
except:
    install_package("numba")

try:
    import PIL
except:
    install_package("Pillow")    

# ----------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------

from .core import *


