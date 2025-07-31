import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade", "--user"])

"""
try:
    import elevation
except:
    install_package("elevation")
    import elevation
"""
try:
    import requests
except:
    install_package("requests")

try:
    import pyproj
except:
    install_package("pyproj")

try:
    import PIL
except:
    install_package("Pillow")

try:
    import aiohttp
except:
    install_package("aiohttp")

try:
    import lmdb
except:
    install_package("lmdb")

# Progress bar
try:
    import tqdm
except:
    install_package("tqdm")

try:
    import rasterio
except:
    install_package("rasterio")

try:
    import geopandas
except:
    install_package("geopandas")

