import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade", "--user"])

try:
    import geopandas
except:
    install_package("geopandas")

try:
    import matplotlib
except:
    install_package("matplotlib")


import geopandas as gpd
import matplotlib.pyplot as plt


# Remplace ce chemin par le chemin de ton fichier .shp
chemin_fichier = "/Users/alain/Downloads/RGEALTI_2-0_TA-1M_SHP_WGS84G_WLD_2023-10-19/TA_RGEALTI_FR_LAMB93.shp"

# Chargement du shapefile
gdf = gpd.read_file(chemin_fichier)

# Affichage d'un aperçu des données attributaires
print("HEAD")
print(gdf.head())
print()

print("COLUMNS")
print(gdf.columns)
print()

print("INFO")
print(gdf.info())
print()

# Affichage des types de géométrie
print("TYPES")
print(gdf.geom_type.value_counts())
print()

# Tracer la carte
#gdf.plot()
#plt.show()

for i in range(10):
    print('-----', i)
    print(gdf.geometry.iloc[i])    