import numpy as np

from npblender.core.blender import pil_to_bl_image
from npblender.gis.gisservices  import Area, BlenderArea, IgnImages, IgnAltitudes #, RgeAlti
from npblender import Mesh, Curve

# ====================================================================================================
# Load an area with satellite image and build a Mesh socle
# ====================================================================================================

def mesh_from_web(lon, lat, size=1000, name="Socle", z_scale=1., socle=False):

    if True:
        # Lambert 93 Area
        x, y = Area.gps_to_lamb93(lon, lat)
        area = Area.Center(x, y, size, projection="LAMB93")
    else:
        # GPS Area
        area = Area.Center(lon, lat, size, projection="GPS")

    # ---------------------------------------------------------------------------
    # Image and image shader
    # ---------------------------------------------------------------------------
    
    mat_name = f"{name} Satellite"
    
    img = IgnImages(projection="LAMB93").load_image(*area.rect_lamb93, image_size=4096)    
    bl_img = pil_to_bl_image(img)
    
    import geonodes as gn
    
    with gn.ShaderNodes(mat_name):
        
        ped = gn.Shader.Principled(
            base_color =  gn.snd.image_texture(image=bl_img),
            roughness  = 1.,
        )
        
        ped.out()
    
    # ---------------------------------------------------------------------------
    # Load the altitudes from the Web
    # ---------------------------------------------------------------------------
    
    alts = IgnAltitudes()
    grid = alts.load_area(area, shape=100)
    nx, ny, _ = grid.shape
    
    # ---------------------------------------------------------------------------
    # Build he mesh
    # ---------------------------------------------------------------------------
    
    zmin, zmax = np.min(grid[..., 2]), np.max(grid[..., 2])
    barea = BlenderArea(area, alt_offset=zmin - 1)
    barea.alt_scale = z_scale/(zmax - zmin)
    
    xygrid = barea.grid_to_mesh(grid)
    
    mesh = Mesh.grid(1, 1, nx, ny, materials=mat_name)
    mesh.points.position = xygrid.reshape(-1, 3)

    if socle:
        mesh.solidify_socle((nx, ny), z=0)

    mesh.to_object(name, shade_smooth=False)

def city_mesh(name, size=1000, socle=False):

    places = Area.get_gps_from_place_name(name, country=None, limit=1)
    lon, lat = places[0]["lon"], places[0]["lat"]
    mesh_from_web(lon, lat, size, name, socle=socle)




def demo(name="Paris"):
    if name.lower() == "paris":
        mesh_from_web(2.33, 48.86, 15000, "Paris", socle=False)
    elif name.lower() == "france":
        mesh_from_web(1.83, 46.5, 1000000, "France", socle=False) 
    elif name.lower() == "lyon":
        city_mesh("Lyon", 8000)
    else:
        raise AttributeError(f"Demo '{name}' not implemented.")          


        
        