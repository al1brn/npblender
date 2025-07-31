""" RGEALTI tiles

A livraison constains 3 folders:
- MNT : asc files with tiles altitudes
- SRC : source tif files
- DST : distance tif files

MNT is the only resource which is used, other folders can be deleted
MNT asc files are converted into npy files which are more compact and quicket to read
"""


import os
import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor

def extract_coord_from_name(filename: str) -> str:
    """
    Extract tile coordinates from file name
    """
    parts = filename.split('_')
    for i in range(len(parts) - 1):
        if parts[i].isdigit() and parts[i+1].isdigit():
            return f"{int(parts[i]):04d}_{int(parts[i+1]):04d}"
    raise ValueError(f"File name is not valid : {filename}")

def convert_all_asc_to_npy(input_dir: str, output_dir: str):
    """
    Convert all ASC files in NPY files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    count=0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".asc"):
            asc_path = os.path.join(input_dir, filename)
            try:
                tile_code = extract_coord_from_name(filename)
            except ValueError as e:
                print(e)
                continue

            npy_path = os.path.join(output_dir, f"{tile_code}.npy")
            
            with rasterio.open(asc_path) as src:
                data = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
            
            np.save(npy_path, data)
            print(f"✅ {filename} → {tile_code}.npy")
            count += 1

    print(f"🎉 Conversion completed, {count} files converted.")

# ====================================================================================================
# Tile coordinates from Lambert93 to IGN (ex: '0882_6814')

def get_tile_corner(x, y, step=1):
    """ Return the tile north-west corner from coordinates x, y

    e.g.:
    - y = 8001 -> tile 9000
    - y = 8000 -> tile 8000
    - y = 7999 -> tile 8000
    

    """
    tile_size = step*1000
    return np.floor(x/tile_size).astype(int), np.floor((y - 1)/tile_size).astype(int) + 1

def get_tile_code(xcorner, ycorner, step=1):
    """
    Return tile code string (e.g., '0882_6814') for Lambert93 coordinates,
    based on tile size = tile_resolution * 1000 meters.

    Tile name is based on North-West corner
    """
    return f"{xcorner:04d}_{ycorner:04d}"

def get_tile_name(x, y, step=1):
    return f"{get_tile_code(*get_tile_corner(x, y, step=step))}.npy"

def get_asc_tile_name(x: float, y: float, tile=1) -> str:
    """
    Retourne le nom de fichier .asc complet correspondant à la dalle contenant (x, y).
    """
    prefix = "RGEALTI_FXX"
    suffix = "MNT_LAMB93_IGN69"
    tile_code = get_tile_code(*get_tile_corner(x, y), step=step)
    return f"{prefix}_{tile_code}_{suffix}.asc"

# ====================================================================================================
# Tile coordinates from Lambert93 to IGN (ex: '0882_6814')

def process_batch(x_batch, y_batch, rge_dir, step=1):
    """
    Vectorized processing of a batch of coordinates.
    Returns altitude values (with np.nan for missing tiles or out-of-bounds).
    """
    tile_size = step*1000

    n = x_batch.size
    alt_batch = np.full(n, np.nan, dtype=np.float32)

    coords = np.stack([x_batch, y_batch], axis=1)
    x_corners, y_corners = get_tile_corner(x_batch, y_batch, step=step)
    tile_corners = np.stack([x_corners, y_corners], axis=1)
    unique_tiles, indices = np.unique(tile_corners, axis=0, return_inverse=True)

    if False:
        print("CHECK Corners")
        a = x_batch - x_corners*tile_size
        print("   X", np.min(a), np.max(a))
        a = y_batch - y_corners*tile_size
        print("   Y", np.min(a), np.max(a))
        print()

        print("CHECK Unique")
        print("corners", np.shape(tile_corners), "idxs", np.shape(indices))
        print(indices)
        for i_batch, (xc, yc) in enumerate(unique_tiles):
            sel = indices == i_batch
            print("- tile", xc, yc)
            a = x_batch[sel] - xc*tile_size
            print("   X", np.min(a), np.max(a))
            a = y_batch[sel] - yc*tile_size
            print("   Y", np.min(a), np.max(a))
        print()

    for i_unq, (x_corner, y_corner) in enumerate(unique_tiles):

        sel_unq = indices == i_unq

        x0 = x_corner * tile_size
        y0 = y_corner * tile_size

        npy_path = os.path.join(rge_dir, f"{get_tile_code(x_corner, y_corner)}.npy")
        if not os.path.exists(npy_path):
            print(f"⚠️ Missing tile: {npy_path}")
            continue

        try:
            data = np.load(npy_path)
        except Exception as e:
            print(f"❌ Failed to load {tile}: {e}")
            continue

        # CAUTION:
        # - we must transpose tiles
        # - y is oriented towards bottom
        iy = np.clip(np.round(x_batch[sel_unq] - x0).astype(int), 0, 999)
        ix = np.clip(y0 - np.round(y_batch[sel_unq]).astype(int), 0, 999)

        alt_batch[sel_unq] = data[ix, iy]

    return alt_batch

def rgealti_get(x, y, rge_dir, step=1, batch_size=100_000, max_workers=4):
    """
    Full pipeline: parallel extraction of altitudes for arrays x, y of Lambert93 coordinates.
    """
    assert x.shape == y.shape, "x and y must have the same shape"

    x_flat = x.ravel()
    y_flat = y.ravel()
    n_total = x_flat.size

    results = np.full(n_total, np.nan, dtype=np.float32)

    if False:
        print(f"🧮 Processing {n_total} points in batches of {batch_size} (step={step} m)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, n_total, batch_size):
            x_batch = x_flat[i:i+batch_size]
            y_batch = y_flat[i:i+batch_size]
            futures.append((i, executor.submit(process_batch, x_batch, y_batch, rge_dir, step)))

        for i, future in futures:
            result = future.result()
            results[i:i+len(result)] = result

    return results.reshape(x.shape)
