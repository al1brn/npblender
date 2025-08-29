import numpy as np
from pyproj import Geod, Transformer
import requests


# ====================================================================================================
# Geographical area

class Area:

    GRID_INDEXING = 'ij'

    def __init__(self, x0, y0, x1, y1, projection="GPS"):
        """ Manages an area

        Arguments
        ---------
        - x0, y0, x1, y1 : area rectangle
        - projection (str in ('GPS', 'LAMB93', 'BLENDER')) : projection
        """

        self.projection = projection
        if projection == 'GPS':
            self.unit_scale = 1_000_000
        elif projection == 'LAMB93':
            self.unit_scale = 1
        elif projection == 'BLENDER':
            self.unit_scale = 100_000
        else:
            raise ValueError(f"Area: Unkwnown projection '{projection}'")
        
        self._x0, self._y0, self._x1, self._y1 = round(x0*self.unit_scale), round(y0*self.unit_scale), round(x1*self.unit_scale), round(y1*self.unit_scale)

    @classmethod
    def Center(cls, x, y, size, projection="GPS"):

        if not hasattr(size, '__len__'):
            size = (size, size)

        if projection == 'GPS':
            x0, y0 = cls.move_gps_point(x, y, -size[0]/2, -size[1]/2)
            x1, y1 = cls.move_gps_point(x, y, size[0]/2, size[1]/2)

            return cls(x0, y0, x1, y1, projection=projection)
        
        else:
            return cls(x - size[0]/2, y - size[1]/2, x + size[0]/2, y + size[1]/2, projection=projection)
    
    def __str__(self):
        return f"<Area: ({self.x0}, {self.y0}, {self.x1}, {self.y1}), projection: {self.projection}>"
    
    @property
    def x0(self):
        return self._x0/self.unit_scale

    @property
    def y0(self):
        return self._y0/self.unit_scale

    @property
    def x1(self):
        return self._x1/self.unit_scale

    @property
    def y1(self):
        return self._y1/self.unit_scale
    
    @property
    def rect(self):
        return (self.x0, self.y0, self.x1, self.y1)
    
    @property
    def rect_lamb93(self):
        (x0, x1), (y0, y1) = self.to_lamb93((self.x0, self.x1), (self.y0, self.y1))
        return (x0, y0, x1, y1)

    @property
    def rect_gps(self):
        (x0, x1), (y0, y1) = self.to_gps((self.x0, self.x1), (self.y0, self.y1))
        return (x0, y0, x1, y1)

    @property
    def width(self):
        return self.x1 - self.x0
    
    @property
    def height(self):
        return self.y1 - self.y0

    def grid_lamb93(self, shape=100, extend=False):
        if not hasattr(shape, '__len__'):
            shape = (shape, shape)
        s0, s1 = (shape[0] + 1, shape[1] + 1) if extend else shape
        x0, y0, x1, y1 = self.rect_lamb93

        return np.meshgrid(np.linspace(x0, x1, s0), np.linspace(y0, y1, s1), indexing=self.GRID_INDEXING)

    def grid_gps(self, shape=100, extend=False):
        if not hasattr(shape, '__len__'):
            shape = (shape, shape)
        s0, s1 = (shape[0] + 1, shape[1] + 1) if extend else shape
        x0, y0, x1, y1 = self.rect_gos

        return np.meshgrid(np.linspace(x0, x1, s0), np.linspace(y0, y1, s1), indexing=self.GRID_INDEXING)

    @property
    def m_rect_size(self):

        if self.projection == 'LAMB93':
            return self.width, self.height
        
        if self.projection == 'GPS':
            geod = Geod(ellps="WGS84")

            lon0, lat0, lon1, lat1 = self.rect

            # Distance est-ouest (lon1 → lon2 à latitude constante)
            az0, az1, width_m = geod.inv(lon0, lat0, lon1, lat0)

            # Distance nord-sud (lat1 → lat2 à longitude constante)
            az0, az1, height_m = geod.inv(lon0, lat0, lon0, lat1)

            return width_m, height_m

    @property
    def m_width(self):
        return self.m_rect_size[0]
    
    @property
    def m_height(self):
        return self.m_rect_size[1]
    
    def move_gps_point(lon, lat, dx_m, dy_m, projection="EPSG:3857"):
        """
        Déplace un point GPS d'un vecteur (dx, dy) en mètres dans une projection métrique.

        Paramètres
        ----------
        - lon, lat : coordonnées GPS de départ (WGS84)
        - dx_m, dy_m : décalage en mètres (vers l'est, vers le nord)
        - projection : projection métrique à utiliser (par défaut EPSG:3857)

        Retour
        ------
        - (lon2, lat2) : coordonnées GPS du point déplacé
        """
        # WGS84 ↔ Projection métrique
        to_proj = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        to_gps  = Transformer.from_crs(projection, "EPSG:4326", always_xy=True)

        x, y = to_proj.transform(lon, lat)
        x2, y2 = x + dx_m, y + dy_m
        lon2, lat2 = to_gps.transform(x2, y2)
        return lon2, lat2
    
    # ====================================================================================================
    # Get the GPS coordinates of a name

    @staticmethod
    def get_gps_from_place_name(place_name, country=None, limit=1):
        """
        Get GPS coordinates (latitude, longitude) from a place name using OpenStreetMap Nominatim API.

        Parameters
        ----------
        place_name : str
            The name of the place (e.g., "Tour Eiffel", "Corciux").
        country : str, optional
            Limit search to a specific country (e.g., "France").
        limit : int
            Maximum number of results to return.

        Returns
        -------
        list of dict
            Each dict contains 'lat', 'lon', and 'display_name'.
        """

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": place_name,
            "format": "json",
            "addressdetails": 1,
            "limit": limit,
        }
        if country:
            params["countrycodes"] = country.lower()

        headers = {
            "User-Agent": "GeoApp/1.0 (contact@example.com)"  # Change this to your contact info
        }

        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            return [
                {
                    "lat": float(result["lat"]),
                    "lon": float(result["lon"]),
                    "display_name": result["display_name"]
                }
                for result in response.json()
            ]
        else:
            print("Error:", response.status_code)
            return []    

    # ====================================================================================================
    # Static coordinates conversion

    @staticmethod
    def gps_to_lamb93(lon, lat):
        trf = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True).transform
        x, y = trf(lon, lat)
        return np.round(x).astype(int), np.round(y).astype(int)

    @staticmethod
    def lamb93_to_gps(x, y):
        trf = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True).transform
        return trf(x, y)
    
    # ====================================================================================================
    # Conversion from projection to target

    def to_lamb93(self, x, y):
        if self.projection == 'GPS':
            return self.gps_to_lamb93(x, y)
        else:
            return x, y

    def to_gps(self, x, y):
        if self.projection == 'GPS':
            return x, y
        else:
            return self.lamb93_to_gps(x, y)

    # ====================================================================================================
    # Cut

    def select_inside(self, vecs):

        sel = np.ones(vecs.shape[:-1], bool)

        x, y = vecs[..., 0], vecs[..., 1]
        sel[x < self.x0] = False
        sel[x > self.x1] = False
        sel[y < self.y0] = False
        sel[y > self.y1] = False

        return sel

    def cut_curve(self, vecs):

        def is_outside(v):
            x, y = v[0], v[1]
            return x < self.x0 or x > self.x1 or y < self.y0 or y > self.y1

        def cut_segment(v0, v1):
            x0, y0 = v0[0], v0[1]
            x1, y1 = v1[0], v1[1]
            x, y = x1, y1

            cut = None

            if x1 < self.x0:
                x = self.x0
                cut = 'X'
            elif x1 > self.x1:
                x = self.x1
                cut = 'X'

            if y1 < self.y0:
                y = self.y0
                cut = 'Y'
            elif y1 > self.y1:
                y = self.y1
                cut = 'Y'

            if len(v0) == 3:
                z0, z1 = v0[2], v1[2]

                if cut == 'X':
                    z = z0 + (z1 - z0)*(x - x0)/(x1 - x0)
                else:
                    z = z0 + (z1 - z0)*(y - y0)/(y1 - y0)

                return x, y, z

            else:
                return x, y

        # ----- Body

        if len(vecs) < 2:
            return vecs


        cut = np.empty_like(vecs)
        index = 0
        last_v = None
        last_out = True
        for i, v  in enumerate(vecs):
            if is_outside(v):
                if not last_out:
                    cut[index] = cut_segment(last_v, v)
                    index += 1
                last_out = True
            else:
                if last_out and last_v is not None:
                    cut[index] = cut_segment(v, last_v)
                    index += 1

                cut[index] = v
                index += 1
                last_out = False
                
            last_v = v

        return np.array(cut[:index])
        
    # ====================================================================================================
    # Tiling

    def get_tiles(self, tile_size, origin=None):
        """ Split the area in tiles """

        # ----------------------------------------------------------------------------------------------------
        # Local utility

        def get_tile_range(O, v0, v1, size):
            # Convert to integer grid coordinates
            t0, t1 = v0 - O, v1 - O

            # Index of tile that contains the LEFT edge (included)
            i0 = t0 // size

            # Index of tile whose LEFT edge is just before or equal to x1
            i1 = (t1 - 1) // size

            # Number of tiles
            n = i1 - i0 + 1

            return i0, n
        
        # ----------------------------------------------------------------------------------------------------
        # Body

        if origin is None:
            _origin = (self._x0, self._y0)
        else:
            _origin = round(origin[0]*self.unit_scale), round(origin[1]*self.unit_scale)

        # Tile size to inertnal int values
        if not hasattr(tile_size, '__len__'):
            tile_size = (tile_size, tile_size)

        sizex, sizey = round(tile_size[0]*self.unit_scale), round(tile_size[1]*self.unit_scale)

        i0, nx = get_tile_range(_origin[0], self._x0, self._x1, sizex)
        j0, ny = get_tile_range(_origin[1], self._y0, self._y1, sizey)

        # Build the tiles
        tiles = []
        for i in range(nx):
            row = []
            x0 = (_origin[0] + (i0 + i)*sizex)/self.unit_scale
            for j in range(ny):
                y0 = (_origin[1] + (j0 + j)*sizey)/self.unit_scale
                row.append({
                    "x0":  x0,
                    "y0":  y0,
                    "x1":  x0 + tile_size[0],
                    "y1":  y0 + tile_size[1],
                })
            tiles.append(row)

        return tiles
    
# ====================================================================================================
# Blender Area

class BlenderArea(Area):
    def __init__(self, area, x0=-10, y0=-10, x1=10, y1=10, alt_offset=0., alt_scale=None):
        """ Manages a Blender area on a geographical area

        Arguments
        ---------
        - x0, y0, x1, y1 : area rectangle
        - area (Area) : geographical area
        """

        super().__init__(x0, y0, x1, y1, projection="BLENDER")
        self.area = area

        self.alt_offset = alt_offset
        self.alt_scale  = self.scale[0] if alt_scale is None else alt_scale

    @classmethod
    def Center(cls, area, x=0, y=0, size=10):

        if not hasattr(size, '__len__'):
            size = (size, size)

        return cls(x - size[0]/2, y - size[1]/2, x + size[0]/2, y + size[1]/2, area=area)
    
    def __str__(self):
        return f"<Blender Area: ({self.x0}, {self.y0}, {self.x1}, {self.y1}) on : {str(self.area)}>"
    
    @property
    def scale(self):
        return self.width/self.area.width, self.height/self.area.height
    
    def xy_to_area(self, x, y):
        scale = self.scale
        return self.area.x0 + (x - self.x0)/scale[0], self.area.y0 + (y - self.y0)/scale[1]
    
    def area_to_xy(self, x, y):
        scale = self.scale
        return self.x0 + (x - self.area.x0)*scale[0], self.y0 + (y - self.area.y0)*scale[1]

    def alt_to_z(self, alt):
        return (alt - self.alt_offset)*self.alt_scale
    
    def z_to_alt(self, z):
        return self.alt_offset + z/self.alt_scale

    def grid_to_mesh(self, grid):
        nx, ny, _ = grid.shape
        if False:
            x, y = np.meshgrid(np.linspace(self.x0, self.x1, nx), np.linspace(self.y0, self.y1, ny))
        else:
            x, y = self.area_to_xy(grid[..., 0], grid[..., 1])
            
        if grid.shape[-1] == 2:
            return np.stack((x, y), axis=-1)
        else:
            z = self.alt_to_z(grid[..., -1])
            return np.stack((x, y, z), axis=-1)

