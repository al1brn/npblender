import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade", "--user"])

try:
    import rasterio
except:
    install_package("rasterio")

from pprint import pprint


import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Remplace par le chemin de ton fichier .asc
fichier_asc = "/Users/alain/cache/vosges 1M/RGEALTI_2-0_1M_ASC_LAMB93-IGN69_D088_2021-09-24/RGEALTI/1_DONNEES_LIVRAISON_2021-11-00178/RGEALTI_MNT_1M_ASC_LAMB93_IGN69_D088_20211123/RGEALTI_FXX_0984_6808_MNT_LAMB93_IGN69.asc"

with rasterio.open(fichier_asc) as src:
    data = src.read(1)           # Lire la 1ère bande (2D array)
    profile = src.profile        # Métadonnées (CRS, transform, etc.)

# Infos utiles
print("Dimensions :", data.shape, type(data), np.shape(data))
print("CRS :", profile["crs"])
print("Transform :")
print(profile["transform"])
print("Valeurs min/max :", np.nanmin(data), np.nanmax(data))

"""
# Visualisation
plt.imshow(data, cmap="terrain")
plt.colorbar(label="Altitude (m)")
plt.title("MNT - RGE ALTI (fichier .asc)")
plt.show()
"""
