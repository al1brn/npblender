""" Download IGS data from IGN API
"""

import math
import requests
from pyproj import Geod, Transformer
from PIL import Image
from io import BytesIO
import numpy as np
import asyncio
from .areas import Area,BlenderArea
from .asyncrequest import load_batch, ApiCall
from .giscache import GisFiles, AltitudeCache
from .rgealti import rgealti_get, get_tile_name
import os

# ====================================================================================================
# Root class for GIS service API

class GisService(list):
    def __init__(self, provider, data_type, url, resource=None, projection="GPS", max_rate=10, method="POST", cache_dir=None):
        """ A GIS Service groups all calls to a same API.

        Once all the desired requests stacked in a service instance, they can be added as a
        single batch in a instance a ApiCall.

        Several services can add their batch of requests to the ApiCall.

        Once the calls performeds, the responses can be read using get_response()

        ```python
        # Stack the requests for a given API
        index0 = service.add_request(...)
        index1 = service.add_request(...)
        index2 = service.add_request(...)

        # Stack the batches from services
        service.to_api_call(...)
        other_service.to_api_call(...)

        # Parallel requests for different APIs
        api_call.calls()

        # Get the results
        data0 = service.get_data(index0)
        data1 = service.get_data(index1)
        data2 = service.get_data(index2)
        ```
        """

        self.provider   = provider
        self.data_type  = data_type
        self.url        = url
        self.resource   = resource
        self.projection = projection
        self.max_rate   = max_rate
        self.method     = method

        self.cache = None if cache_dir is None else GisFiles(provider=provider, root=cache_dir)

        self.attempts = 3
        self.api_call_index = None
        self.api_call = None

    # ====================================================================================================
    # Clone

    def clone(self, **kwargs):
        cls = self.__class__
        clone = cls.__new__(cls)

        clone.provider   = self.provider
        clone.data_type  = self.data_type
        clone.url        = self.url
        clone.resource   = self.resource
        clone.projection = self.projection
        clone.max_rate   = self.max_rate
        clone.method     = self.method

        clone.cache      = self.cache
        clone.attempts   = self.attempts

        clone.api_call = None
        clone.api_call_index = None

        for k, v in kwargs.items():
            setattr(clone, k, v)

        return clone
    
    # ====================================================================================================
    # Response check function

    #@staticmethod
    #def check_function(response):
    #   return True
    
    @staticmethod
    def get_data_from_response(response):
        return None, f"{type(self).__name__}.get_data_from_response not implemented"
    
    # ====================================================================================================
    # Binary / json

    @property
    def binary(self):
        if self.data_type in ['images']:
            return True
        elif self.data_type in ['altitudes']:
            return False
        else:
            raise ValueError(f"GisService: Unknown data type {self.data_type}")

    @property
    def extension(self):
        if self.data_type in ['images']:
            return "png"
        elif self.data_type in ['altitudes']:
            return "npy"
        else:
            raise ValueError(f"GisService: Unknown data type {self.data_type}")
        
    @property
    def content_type(self):
        if self.data_type in ['images']:
            return "image/png"
        else:
            return "application/json"

    # ====================================================================================================
    # Add a request

    def add_request(self, **kwargs):
        """ Add an empty request in the list

        If cache is managed for this kind of data, cached_data can be read from cache.
        If cached_data is not None, the payload won't be add in the api call and data
        will be copied from cached_data.

        If cache_path is not None, data will be saved in cache if no error occured.

        ret_index is the index to use to get the response after the api call.
        """
        req = {
            "kwargs"      : dict(kwargs), # User arguments
            "payload"     : None,         # Request payload
            "ret_index"   : None,         # Returned data index
            "data"        : None,         # Returned data
            "error"       : None,         # Error
            "cached_data" : None,         # Data in cache
            "cache_path"  : None,         # Cache path
        }
        index = len(self)
        self.append(req)
        return index
    
    # ====================================================================================================
    # Add the batch of requests to an instance of ApiCall

    def to_api_call(self, api_call):

        self.api_call = api_call

        payloads = []
        for i, req in enumerate(self):
            if req["cached_data"] is None:
                req["ret_index"] = len(payloads)
                payloads.append(req["payload"])
            else:
                req["data"] = req["cached_data"]

        self.api_call_index = api_call.add({
            "url"       : self.url,
            "payloads"  : payloads,
            "max_rate"  : self.max_rate,
            "get_data"  : self.get_data_from_response,
            "attempts"  : self.attempts,
            "method"    : self.method,
            "binary"    : self.binary,
        })
        self.responses_are_read = False

    def do_calls(self):
        api_call = ApiCall()
        self.to_api_call(api_call)
        api_call.calls()
        api_call.summary()

    # ====================================================================================================
    # Responses

    def check_call(self):
        if self.api_call is None:
            raise Exception(f"{type(self).__name__} error: please call 'to_api_call()' before trying to get the responses")
        
    def load_responses(self):

        self.check_call()
        if self.responses_are_read:
            return
        
        responses = self.api_call.get_response(self.api_call_index)
        for i, req in enumerate(self):

            if req["ret_index"] is None:
                continue

            response = responses[req["ret_index"]]
            req["data"]  = response["data"]
            req["error"] = response["error"]

            if req["error"] is None and req["cache_path"] is not None:
                self.cache.save_cache_with_path(req["data"], req["cache_path"], self.extension)

        self.responses_are_read = True
        
    def is_error(self, index):
        self.load_responses()
        return self[index]["error"] is not None

    def get_error(self, index):
        self.load_responses()
        return self[index]["error"]

    def get_data(self, index):
        self.load_responses()
        if isinstance(index, int):
            return self[index]["data"]
        elif isinstance(index, (list, np.ndarray)):
            return [self[i]["data"] for i in index]
        else:
            raise Exception(f"{type(self).__name__} error: Unsupported index type: {type(index)}, {str(index):100s}")
    
# ====================================================================================================
# IGN images

class IgnImages(GisService):
    def __init__(self, layers="ORTHOIMAGERY.ORTHOPHOTOS", projection="LAMB93", max_rate=30, cache_dir=None):
        super().__init__(
            provider        = "IGN",
            data_type       = "images",
            url             = "https://data.geopf.fr/wms-r",
            resource        = layers, 
            projection      = projection, 
            max_rate        = max_rate, 
            method          = "GET", 
            cache_dir       = cache_dir)
        
    @staticmethod
    def get_data_from_response(response):
        try:
            data = Image.open(BytesIO(response))
            #data = data.transpose(Image.ROTATE_270)
            #data = data.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)

        except Exception as e:
            return None, f"IgnImage error loading image request {i:3d}: {str(e)}"
        
        return data, None

    # ====================================================================================================
    # Add an image

    def add(self, x0, y0, x1, y1, image_size):

        if not hasattr(image_size, '__len__'):
            if x1-x0 > y1 - y0:
                image_size = (image_size, int(round(image_size*(y1 - y0)/(x1 - x0))))
            else:
                image_size = (int(round(image_size*(x1 - x0)/(y1 - y0))), image_size)

        index = self.add_request(x0=x0, y0=y0, x1=x1, y1=y1, image_size=image_size)
        req = self[index]

        payload = {
            "SERVICE"   : "WMS",
            "VERSION"   : "1.3.0",
            "REQUEST"   : "GetMap",
            "LAYERS"    : self.resource, 
            "CRS"       : "EPSG:2154" if self.projection == "LAMB93" else "EPSG:4326",
            "BBOX"      : f"{x0},{y0},{x1},{y1}" if self.projection == "LAMB93" else f"{y0},{x0},{y1},{x1}",
            "WIDTH"     : image_size[0],
            "HEIGHT"    : image_size[1],
            "FORMAT"    : "image/jpeg",
            "STYLES"    : "",
        }

        print("PAYLOAD:", payload)
        req["payload"] = payload

        if self.cache is not None:
            cache_args = {
                "data_type"     : self.data_type,
                "resource"      : self.resource,
                "rect"          : (x0, y0, x1, y1),
                "resolution"    : image_size,
                "projection"    : self.projection,
                "extension"     : self.extension,
            }

            req["cached_data"] = self.cache.load_cache(**cache_args)
            req["cache_path"]  = self.cache.get_path(**cache_args, create_dirs=True)

        return index    
    
    # ====================================================================================================
    # Add tiles

    def add_tiles(self, x0, y0, x1, y1, tile_size, image_size=1024):
        
        area = Area(x0, y0, x1, y1, projection=self.projection)
        tiles = area.get_tiles(tile_size)

        for row in tiles:
            for tile in row:
                index = self.add(x0=tile["x0"], y0=tile["y0"], x1=tile["x1"], y1=tile["y1"], image_size=image_size)
                tile["ret_index"] = index

        return {"type": "TILES", "nx": len(tiles), "ny": len(tiles[0]), "tiles": tiles}
                   
    # ====================================================================================================
    # Get data from 

    def get_data(self, index):
        self.load_responses()

        if isinstance(index, dict):
            d_type = index.get("type", None)
            if d_type == "TILES":
                for row in index["tiles"]:
                    for tile in row:
                        img_index = tile["ret_index"]
                        err = self.get_error(img_index)
                        if err is None:
                            tile["error"] = None
                            tile["image"] = self[img_index]["data"]
                        else:
                            tile["error"] = err
                            tile["image"] = None

                return index["tiles"]
            else:
                raise TypeError(f"{type(self).__name__}.get_data: Index is not a valid dict {list(index.keys())}")

        else:
            return super().get_data(index)


    
    # ====================================================================================================
    # Load images

    def load_images(self, areas, image_size=512):
        """ Load an satellite image from IGN

        Arguments
        ---------
        - areas (list of dicts) :
            - x0, y0, x1, y1 : lambert93 coordinates of the rectangle to download
            - image_size (optinal tuple) : size of the image in pixels
        - image_size (tuple of ints) : default image size in pixels

        Returns
        -------
        - images (list of PIL.Images) : Images of the satellite image
        """

        # Call with a clone not to interfer with current payloads
        service = self.clone()

        # Loop on the areas
        for area in areas:

            x0, y0, x1, y1 = area["x0"], area["y0"], area["x1"], area["y1"]
            _image_size = area.get("image_size", image_size)

            service.add(x0, y0, x1, y1, image_size=_image_size)

        # Load the paylaods
        service.do_calls()

        return [service.get_data(i) for i in range(len(service))]
    
    # ====================================================================================================
    # Load a single image

    def load_image(self, x0, y0, x1, y1, image_size=512):
        """ Load an satellite image from IGN

        Arguments
        ---------
        - x0, y0, x1, y1 : lambert93 coordinates of the rectangle to download
        - image_size (tuple = 512) : size of the image in pixels

        Returns
        -------
        - image (PIL.Image) : Image of the satellite image
        """

        return self.load_images([{"x0": x0, "y0": y0, "x1": x1, "y1": y1}], image_size=image_size)[0]
    
    # ====================================================================================================
    # Load tiles

    def load_tiles(self, x0, y0, x1, y1, tile_size, image_size=1024):

        # Call with a clone not to interfer with current payloads
        service = self.clone()

        # Build the tiles
        tiles = service.add_tiles(x0, y0, x1, y1, tile_size, image_size=image_size)

        # Call
        service.do_calls()

        return service.get_data(tiles)

# ====================================================================================================
# IGN altitudes

class IgnAltitudes(GisService):

    BATCH_SIZE = 5000 # Max number of points accepted by IGN

    def __init__(self, resource="ign_rge_alti_wld", max_rate=5, cache_dir=None):
        super().__init__(
            provider        = "IGN",
            data_type       = "altitudes",
            url             = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json",
            resource        = resource, 
            projection      = "GPS", 
            max_rate        = max_rate, 
            method          = "POST", 
            cache_dir       = cache_dir)

    @property
    def cache_path(self):
        if self.cache is None is None:
            return None
        else:
            return self.cache.get_specific_path(self.data_type, self.resource, "xycache.lmdb")
        
    # ====================================================================================================
    # Response check function

    @staticmethod
    def get_data_from_response(response):
        if not "elevations" in response:
            if "error" in response:
                s = str(response)[:100]
            else:
                s = response["error"]
            return None, f"IgnAltitudes> error when loading altitudes: {s}"
        
        return np.array([pt["z"] for pt in response["elevations"]], dtype=np.float32), None

    # ====================================================================================================
    # Add altitudes request

    def add(self, lon, lat):

        assert(len(lon) == len(lat))

        alts = {
            "lon"       : np.array(lon),
            "lat"       : np.array(lat),
            "indices"   : [],
            "offsets"   : [],
            }

        file_name = self.cache_path
        if file_name is not None:
            unit_scale = 1 if self.projection == 'LAMB93' else 1_000_000
            alts["unit_scale"] = unit_scale

            with AltitudeCache(file_name, unit_scale) as ac:
                info = ac.get_altitudes(lon, lat)

            for k, v in info.items():
                alts[k] = v

            lon = alts["lon"][alts["missing_indices"]]
            lat = alts["lat"][alts["missing_indices"]]

        alts["z"] = np.zeros(len(lon), np.float32)

        if len(lon):
            for offset in range(0, len(lon), self.BATCH_SIZE):

                ln = lon[offset:offset + self.BATCH_SIZE]
                lt = lat[offset:offset + self.BATCH_SIZE]

                index = self.add_request()
                alts["indices"].append(index)
                alts["offsets"].append(offset)

                req = self[index]
                payload = {
                    "lon"       : "|".join(map(str, ln)),
                    "lat"       : "|".join(map(str, lt)),
                    "resource"  : self.resource,
                    "delimiter" : "|",            
                    "indent"    : "false",
                    "measures"  : "false",
                    "zonly"     : "false"             
                    }
                req["payload"] = payload

        return alts
    
    # ====================================================================================================
    # get data

    def get_data(self, index):
        self.load_responses()

        if isinstance(index, dict):
            alts = index
            for i, offset in zip(alts["indices"], alts["offsets"]):
                if self.get_error(i) is None:
                    z = self[i]["data"]
                    alts["z"][offset:offset + len(z)] = z

            z = alts["z"]

            file_name = self.cache_path
            if file_name is not None:

                from pprint import pprint
                with AltitudeCache(file_name, alts["unit_scale"]) as ac:
                    ac.update_cache(alts["missing_keys"], z)

                alts["altitudes"][alts["missing_indices"]] = z
                z = alts["altitudes"]


            # Area
            grid = np.stack((alts["lon"], alts["lat"], z), axis=-1)

            shape = index.get("area_shape")
            if shape is not None:
                grid = np.reshape(grid, (shape[0] + 1, shape[1] + 1, 3))

            return grid

        else:
            return super().get_data(index)

    
    # ====================================================================================================
    # Add an area

    def add_area(self, area, shape=100):

        if not hasattr(shape, '__len__'):
            r = area.m_rect_size
            if r[0] > r[1]:
                shape = (shape, round(shape*r[1]/r[0]))
            else:
                shape = (round(shape*r[0]/r[1]), shape)

        lon0, lat0, lon1, lat1 = area.rect_gps
        lon, lat = np.meshgrid(
            np.linspace(lon0, lon1, shape[0] + 1),
            np.linspace(lat0, lat1, shape[1] + 1),
            indexing=Area.GRID_INDEXING
        )

        alts = self.add(lon.ravel(), lat.ravel())

        alts["area_shape"] = shape

        return alts


    # ====================================================================================================
    # Load altitudes

    def load_altitudes(self, lon, lat):

        service = self.clone(cache_name=self.cache_name)
        alts = service.add(np.asarray(lon), np.asarray(lat))

        service.do_calls()

        return service.get_data(alts)

    # ====================================================================================================
    # Load an area

    def load_area(self, area, shape=100):

        service = self.clone()
        alts = service.add_area(area, shape=shape)

        service.do_calls()

        return service.get_data(alts)
    
    # ====================================================================================================
    # Test resolution

    def test_resolution(self, lon, lat, size=10, shape=100):

        x, y = Area.gps_to_lamb93(lon, lat)
        (lon0, lon1), (lat0, lat1) = Area.lamb93_to_gps((x-size/2, x+size/2), (y-size/2, y+size/2))

        lon_space = np.linspace(lon0, lon1, shape + 1)
        lat_space = np.linspace(lat0, lat1, shape + 1)

        print("Resolution test with values:")
        print(" - lon:", type(lon_space), lon_space)
        print(" - lat:", type(lat_space), lat_space)

        xs, ys = Area.gps_to_lamb93(lon_space, lat_space)
        print(" - x  :", xs)
        print(" - y  :", ys)

        lons, lats = np.meshgrid(lon_space, lat_space, indexing=Area.GRID_INDEXING)
        print(" - lons:", lons)
        print(" - lats:", lats)
        
        alts = self.add(lons.ravel(), lats.ravel())
        alts["area_shape"] = (shape, shape)

        self.do_calls()

        grid = self.get_data(alts)

        t = np.linspace(0, size, shape + 1)
        xs, ys = np.meshgrid(t, t, indexing=Area.GRID_INDEXING)
        
        grid[..., 0] = xs
        grid[..., 1] = ys

        return grid

class RgeAlti:
    def __init__(self, rge_dir, step=1, for_mesh=True):
        self.rge_dir = rge_dir
        self.step = step
        self.for_mesh = for_mesh

    def load_tile(self, x, y):
        path = os.path.join(self.rge_dir, get_tile_name(x, y, step=self.step))

        try:
            data = np.load(path)
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
            return None

        t = np.linspace(-5, 5, 1000, endpoint=False)
        xs, ys = np.meshgrid(t, t, indexing=Area.GRID_INDEXING)
        grid = np.stack((xs, ys, data), axis=-1)

        if self.for_mesh:
            pass
            #grid = np.rot90(grid, k=3, axes=(0, 1))

        return grid

    def load_altitudes(self, x, y):
        z = rgealti_get(x, y, self.rge_dir, step=self.step)
        grid = np.stack((x, y, z), axis=-1)
        if self.for_mesh:
            pass
            #grid = np.rot90(grid, k=3, axes=(0, 1))
        return grid

    def load_area(self, area, shape=100):
        x, y = area.grid_lamb93(shape=shape, extend=True)
        return self.load_altitudes(x, y).reshape(np.shape(x) + (3,))
    




