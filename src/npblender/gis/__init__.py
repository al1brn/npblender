import subprocess
import sys

from npblender import ensure_package

requests = ensure_package("requests")
pyproj = ensure_package("pyproj")
PIL = ensure_package("PIL", "Pillow")
aiohttp = ensure_package("aiohttp")
lmdb = ensure_package("lmdb")
tqdm = ensure_package("tqdm")
rasterio = ensure_package("rasterio")
geopandas = ensure_package("geopandas")

