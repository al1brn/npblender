import os
import json
import numpy as np
from PIL import Image

import lmdb
import struct

# ====================================================================================================
# Gis cache files

class GisFiles:
    def __init__(self, provider, root="./cache"):
        self.provider = provider
        self.root = root

    def _format_float(self, val):
        return f"{val:.8f}".rstrip("0").rstrip(".")

    def _build_filename(self, data_type, resource, rect, resolution, projection, extension):
        rect_str = "norect"
        if rect:
            rect_str = "_".join(self._format_float(v) for v in rect)

        res_str = "nores"
        if resolution:
            res_str = f"{resolution[0]}x{resolution[1]}"

        filename = f"{projection}_{rect_str}_{res_str}.{extension}"
        return filename
    
    def get_subdir(self, data_type, resource=None):
        if resource is None:
            return os.path.join(self.root, self.provider, data_type)
        else:
            return os.path.join(self.root, self.provider, data_type, resource)
    
    def get_path(self, data_type, resource=None, rect=None, resolution=None,
                 projection="LAMB93", extension="json", create_dirs=True):
        """
        Public method to retrieve the full cache path.

        Parameters
        ----------
        create_dirs : bool
            If True, create directories if they don't exist.

        Returns
        -------
        str
            Full path to the cache file.
        """
        filename = self._build_filename(data_type, resource, rect, resolution, projection, extension)
        if True:
            subdir = self.get_subdir(data_type, resource)
        else:
            subdir = os.path.join(self.root, self.provider, data_type, resource or "default")
        if create_dirs:
            os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, filename)
    
    def get_specific_path(self, data_type, resource=None, file_name=None, create_dirs=True):
        subdir = self.get_subdir(data_type, resource)

        if create_dirs:
            os.makedirs(subdir, exist_ok=True)

        if file_name is None:
            return subdir
        else:
            return os.path.join(subdir, file_name)

    def cache_exists(self, data_type, resource="default", rect=None, resolution=None,
                     projection="LAMB93", extension="json"):
        path = self.get_path(data_type, resource, rect, resolution, projection, extension, create_dirs=False)
        return os.path.exists(path)

    def load_cache(self, data_type, resource="default", rect=None, resolution=None,
                   projection="LAMB93", extension="json", force_reload=False):
        path = self.get_path(data_type, resource, rect, resolution, projection, extension)

        if not os.path.exists(path) or force_reload:
            return None

        if extension == "json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif extension == "npy":
            return np.load(path, allow_pickle=True)
        elif extension == "npz":
            return np.load(path)
        elif extension in ("png", "jpg", "jpeg", "tif", "tiff"):
            return Image.open(path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        
    def save_cache_with_path(self, data, path, extension):

        if extension == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        elif extension == "npy":
            np.save(path, data)
        elif extension == "npz":
            if isinstance(data, dict):
                np.savez(path, **data)
            else:
                raise ValueError("NPZ saving requires a dict of arrays.")
        elif extension in ("png", "jpg", "jpeg", "tif", "tiff"):
            if isinstance(data, np.ndarray):
                img = Image.fromarray(data)
            else:
                img = data
            img.save(path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

    def save_cache(self, data, data_type, resource="default", rect=None, resolution=None,
                   projection="LAMB93", extension="json"):
        path = self.get_path(data_type, resource, rect, resolution, projection, extension)

        self.save_cache_with_path(data, path, extension)


        
# ====================================================================================================
# Cache for altitudes

class AltitudeCache:
    def __init__(self, path, unit_scale, map_size=10_000_000_000):
        """
        Cache altitudes using LMDB.
        os.makedirs(path, exist_ok=True)
        """
        self.env = lmdb.open(path, map_size=map_size)
        self.unit_scale = unit_scale

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.env.close()

    def _keys(self, x, y):
        x = np.round(x * self.unit_scale).astype(np.int64)
        y = np.round(y * self.unit_scale).astype(np.int64)
        return [struct.pack("qq", xi, yi) for xi, yi in zip(x, y)]

    def get_altitudes(self, x, y):
        keys = self._keys(x, y)
        altitudes = np.full(len(keys), np.nan, dtype=np.float32)
        in_cache = np.zeros(len(keys), dtype=bool)
        missing_keys = []
        missing_indices = []

        with self.env.begin() as txn:
            for i, key in enumerate(keys):
                val = txn.get(key)
                if val is not None:
                    altitudes[i] = struct.unpack("f", val)[0]
                    in_cache[i] = True
                else:
                    missing_keys.append(key)
                    missing_indices.append(i)

        return {
            "altitudes"       : altitudes,
            "in_cache"        : in_cache,
            "missing_keys"    : missing_keys,
            "missing_indices" : missing_indices
        }

    def update_cache(self, keys, values):
        with self.env.begin(write=True) as txn:
            for key, value in zip(keys, values):
                txn.put(key, struct.pack("f", value))
