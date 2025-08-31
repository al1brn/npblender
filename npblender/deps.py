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
Module Name: deps
Author: Alain Bernard
Version: 0.1.0
Created: 2025-08-30
Last updated: 2025-08-30

Summary:
Dependency loader for Blender addons.
Tries in order:
  1. Import existing module
  2. Load from local "libs/" folder (pre-bundled wheels)
  3. Install via pip in Blender's Python

Usage example:
    >>> from deps import ensure_package
    >>> scipy = ensure_package("scipy")
"""

__all__ = ["ensure_package"]

import sys
import os
import subprocess
import importlib

def ensure_package(pkg: str, pip_name: str = None):
    """
    Ensure that a package is available inside Blender's Python.
    
    Args:
        pkg (str): Module name to import (e.g. "scipy").
        pip_name (str): Name to use with pip (defaults to pkg).
    
    Returns:
        The imported module.
    """
    pip_name = pip_name or pkg

    # 1. Try to import directly
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[npblender] Missing dependency: {pkg}")

    # 2. Try local libs/ directory (bundled wheels)
    addon_dir = os.path.dirname(__file__)
    libs_dir = os.path.join(addon_dir, "libs")
    if os.path.isdir(libs_dir):
        sys.path.append(libs_dir)
        try:
            return importlib.import_module(pkg)
        except ImportError:
            # try installing wheel manually if found
            wheels = [f for f in os.listdir(libs_dir) if pip_name.lower() in f.lower() and f.endswith(".whl")]
            if wheels:
                whl_path = os.path.join(libs_dir, wheels[0])
                print(f"[npblender] Installing {pip_name} from local wheel {whl_path}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", whl_path])
                return importlib.import_module(pkg)

    # 3. Try installing from PyPI
    print(f"[npblender] Installing {pip_name} from PyPI...")
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pip_name])

    return importlib.import_module(pkg)

def uninstall_package(pkg: str, pip_name: str = None):
    """
    Uninstall a package inside Blender's Python.

    Args:
        pkg (str): Module name to remove (e.g. "scipy").
        pip_name (str): Name to use with pip (defaults to pkg).
    """
    pip_name = pip_name or pkg

    # 1. Try to uninstall via pip
    print(f"[npblender] Uninstalling {pip_name} via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", pip_name])

    # 2. Remove from sys.modules if already imported
    if pkg in sys.modules:
        del sys.modules[pkg]
        importlib.invalidate_caches()
        print(f"[npblender] Removed {pkg} from sys.modules")
