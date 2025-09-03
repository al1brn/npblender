import os
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import box, Polygon
import os
import numpy as np

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union, linemerge


from .areas import Area, BlenderArea

class BDTopo():

    def __init__(self, area, topo_dir, rge_alti=None):

        self.area       = area
        self.topo_dir   = topo_dir
        self.layers     = self.get_all_shapefiles(self.topo_dir)
        self.rge_alti   = rge_alti

    def __str__(self):
        return f"<BDTopo, area: {self.area}\n{list(self.layers.keys())}, {len(self.layers)} layers:\n>"

    # ====================================================================================================
    # List all the shape files

    @staticmethod
    def get_all_shapefiles(topo_dir):

        shapefiles = {}

        for dirpath, _, filenames in os.walk(topo_dir):
            for file in filenames:
                if file.lower().endswith(".shp"):
                    full_path = os.path.join(dirpath, file)
                    theme = os.path.basename(os.path.dirname(full_path))  # dossier parent
                    layer_name = os.path.splitext(file)[0]
                    shapefiles[layer_name] = {
                        "theme": theme,
                        "layer": layer_name,
                        "path": full_path
                    }

        return shapefiles

    # ====================================================================================================
    # Open a shape file

    def open(self, layer_name):

        layer = self.layers.get(layer_name)
        if layer is None:
            raise ValueError(f"BDTopo.select: layer '{layer_name}' not found in {list(self.layers.keys())}.")

        print(f"📂 Opening {layer_name}...")
        shapefile_path = layer["path"]

        gdf = gpd.read_file(shapefile_path)

        if gdf.crs is None:
            raise ValueError("❌ Le shapefile n'a pas de système de coordonnées défini.")

        if gdf.crs.to_epsg() != 2154:
            print(f"🔄 Reprojection to EPSG:2154 (Lambert 93)...")
            gdf = gdf.to_crs(epsg=2154)

        return gdf

    # ====================================================================================================
    # Get a selection

    def select(self, layer_name):

        # Already a selection
        if isinstance(layer_name, GeoDataFrame):
            return layer_name

        gdf = self.open(layer_name)

        if self.area is None:
            return gdf

        # === Select area

        x0, y0, x1, y1 = self.area.rect_lamb93

        zone_geom = box(x0, y0, x1, y1)
        zone_gdf = gpd.GeoSeries([zone_geom], crs="EPSG:2154")

        # === Spatial filter

        print("🔍 Building selection...")
        selection = gdf[gdf.intersects(zone_geom)]

        print(f"✅ {len(selection)} items found.")

        return selection

    # ====================================================================================================
    # Dump the content of a layer

    def dump_layer(self, layer_name, dump_count=5):
        
        selection = self.select(layer_name)
    
        print(f"Layer name: {layer_name}")
        print(f"\titems : {len(selection)}")
        print(f"\tcolumns : {len(selection.columns)}")

        geos = self.get_geometries(selection)
        print(f"\tgeometries : {np.unique([g[0] for g in geos])}")

        for idx, row in selection.iterrows():
            if not dump_count:
                break

            for col in selection.columns:
                if col == 'geometry':
                    s = row.geometry.geom_type
                else:
                    s = row.get(col, 'inconnue')
                print(f"  - {col:20s}    : {s}")
                
            print()
            dump_count -= 1

    # ====================================================================================================
    # Dump layer columns

    def dump_layer_columns(self, layer_name, columns):

        if isinstance(columns, str):
            columns = [columns]

        selection = self.select(layer_name)

        res = {key: [] for key in columns}
        
        for idx, row in selection.iterrows():
            for col in selection.columns:
                if col == 'geometry':
                    s = row.geometry.geom_type
                else:
                    s = row.get(col, 'inconnue')
                if col in columns:
                    res[col].append(s)
    
        print(f"Layer name: {layer_name}")
        print(f"\titems : {len(selection)}")
        print(f"\tcolumns : {len(selection.columns)}")
        for k, v in res.items():
            print(f"- column {k}:")
            print(v)
            print()      

    # ====================================================================================================
    # Geometry to array

    @staticmethod
    def _geometry_to_array(geometry):
        """ Transform geometry into numpy array

        Arguments
        ---------
        - geometry (shapely.geometry) : geometry to transform

        Returns
        -------
        - geometry, array or list of arrays
        """

        gtype = geometry.geom_type
        a = None

        if gtype == "Polygon":
            a = np.array(geometry.exterior.coords)

        elif gtype == "MultiPolygon":
            a = []
            for part in geometry.geoms:
                a.append(np.array(part.exterior.coords))

        elif gtype in ["Points", "LineString"]:
            a = np.array(geometry.coords)

        elif gtype in ["MultiPoint", "MultiLineString"]:
            a = []
            for part in geometry.geoms:
                a.append(np.array(part.coords))

        else:
            raise Exception(f"Unknown geometry type : {gtype}")
        
        return gtype, a

    # ====================================================================================================
    # Get geometries

    def get_geometries(self, layer_name):

        selection = self.select(layer_name)

        return [self._geometry_to_array(row.geometry) for _, row in selection.iterrows()]


    # ====================================================================================================
    # Get geometries merged by column

    def get_geometries_by_column(self, layer_name, column, merge=True):
        """ Group the geometries by a shared name

        Arguments
        ---------
        - layer_name (str) : which shape file to open
        - colume (str) : column to use to group geometries

        Returns
        -------
        - fusion_dict (dict) : dictionary of geometries
        """

        selection = self.select(layer_name)

        # Remplace missing values by a placeholder
        placeholder = "__null__"
        #selection[column] = selection[column].apply(lambda x: x if isinstance(x, str) and x.strip() else placeholder)
        selection[column] = selection[column].fillna(placeholder)

        # Merge dictionary
        fusion_dict = {}

        # Grouping and merging
        for name, group in selection.groupby(column):
            union = unary_union(group.geometry)
            if merge and union.geom_type == "MultiLineString":
                union = linemerge(union)
            fusion_dict[name] = self._geometry_to_array(union)

        return fusion_dict


    # ====================================================================================================
    # Random points in a polygon

    @staticmethod
    def poisson_points_in_polygon(polygon: Polygon, density: float):
        """
        Génère des points dans un polygone selon un processus de Poisson homogène.
        - polygon : shapely Polygon
        - density : densité (points par m²)

        Retourne : np.ndarray de shape (n, 2)
        """
        if polygon.is_empty:
            return np.empty((0, 2))

        area = polygon.area
        expected_num_points = np.random.poisson(density * area)

        # Bounding box
        minx, miny, maxx, maxy = polygon.bounds

        points = []
        while len(points) < expected_num_points:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            p = Point(x, y)
            if polygon.contains(p):
                points.append([x, y])

        return np.array(points)
    
    # ====================================================================================================
    # Create a V roof

    @staticmethod
    def intersect_segments(p1, p2, q1, q2):
        """
        Computes the intersection point between two 2D segments (assumed to intersect).

        Args:
            p1, p2: endpoints of first segment (ridge0)
            q1, q2: endpoints of second segment (ridge1)

        Returns:
            point (2,) ndarray: intersection point in 2D
        """
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        x3, y3 = q1[:2]
        x4, y4 = q2[:2]

        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom == 0:
            raise ValueError("Segments are parallel or colinear")

        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

        return np.array([px, py])

    @staticmethod
    def get_house_ridges(vectors, angle_step=1, angle_thresh_deg=10):
        """
        Extracts two ridge lines:
        - one along the longest axis (faîtage)
        - one along the shortest axis (perpendicular traverse)

        Args:
            vectors: (n, 3) closed polygon (first == last)
            angle_step: angle sweep resolution
            angle_thresh_deg: tolerance to detect axis-aligned segments

        Returns:
            bary (3,)
            angle_deg (float)           : orientation of main axis (faîtage)
            major ridge (2, 3)          : line in longest direction
            minor ridge (2, 3)          : line in shortest direction
        """
        bary = vectors.mean(axis=0)
        centered = vectors - bary

        # Step 1: find best orientation (min width along u)
        angles = np.deg2rad(np.arange(0, 180, angle_step))
        min_span = float("inf")
        best_angle = 0

        for angle in angles:
            u = np.array([np.cos(angle), np.sin(angle), 0])
            proj = centered @ u
            span = proj.max() - proj.min()
            if span < min_span:
                min_span = span
                best_angle = angle

        # Step 2: local frame (u, v)
        u = np.array([np.cos(best_angle), np.sin(best_angle), 0])  # major
        v = np.array([-u[1], u[0], 0])                              # minor
        basis = np.stack([u, v], axis=1)
        coords = centered @ basis  # shape (n, 2)

        angle_thresh_rad = np.deg2rad(angle_thresh_deg)

        top_y, bot_y = [], []
        right_x, left_x = [], []

        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i + 1]
            seg = p1 - p0
            norm = np.linalg.norm(seg)
            if norm == 0:
                continue
            dir_unit = seg / norm
            angle_with_u = np.arccos(np.clip(np.abs(dir_unit @ np.array([1, 0])), 0, 1))
            angle_with_v = np.arccos(np.clip(np.abs(dir_unit @ np.array([0, 1])), 0, 1))

            if angle_with_u < angle_thresh_rad:
                y_avg = 0.5 * (p0[1] + p1[1])
                if dir_unit[0] > 0:
                    top_y.append(y_avg)
                elif dir_unit[0] < 0:
                    bot_y.append(y_avg)

            elif angle_with_v < angle_thresh_rad:
                x_avg = 0.5 * (p0[0] + p1[0])
                if dir_unit[1] > 0:
                    right_x.append(x_avg)
                elif dir_unit[1] < 0:
                    left_x.append(x_avg)

        if not top_y or not bot_y or not right_x or not left_x:
            return None, None, None, None
            raise ValueError("Not enough segment candidates in all directions.")

        y_top = max(top_y)
        y_bot = min(bot_y)
        x_right = max(right_x)
        x_left = min(left_x)
        
        ridge0 = np.zeros((2, 2), np.float32)
        ridge1 = np.zeros_like(ridge0)

        if True:
            center = np.array([(x_left + x_right)/2, (y_bot + y_top)/2])
            ridge0[0] = (center[0], y_bot)
            ridge0[1] = (center[0], y_top)
            ridge1[0] = (x_left, center[1])
            ridge1[1] = (x_right, center[1])

        else:
            ridge1[0][0] = x_left
            ridge1[1][0] = x_right
            ridge0[0][1] = y_bot
            ridge0[1][1] = y_top
            center = BDTopo.intersect_segments(ridge0[0], ridge0[1], ridge1[0], ridge1[1])
        
        ridge0 = ridge0 @ basis.T + bary
        ridge1 = ridge1 @ basis.T + bary
        center = center @ basis.T + bary
        
        return center, best_angle, ridge0, ridge1
    
    @staticmethod
    def compute_roof_from_base_and_ridge(base, ridge_line):
        """
        Computes a pitched V-shaped roof from base polygon and 3D ridge line.

        Parameters:
        - base (ndarray shape (N, 3)) : base polygon as 3D points (same Z)
        - ridge_line (ndarray shape (2, 3)) : two ridge apex points (with apex Z)

        Returns:
        - vertices (ndarray shape (N, 3)) : 3D vertices with updated Z values
        - faces : list of lists of ints : ordered indices for one roof faces
        """
        base = np.asarray(base)
        ridge_line = np.asarray(ridge_line)

        n_points = len(base)
        base_z = base[0, 2]
        ridge_height = ridge_line[0, 2] - base_z

        ridge_start, ridge_end = ridge_line[0, :2], ridge_line[1, :2]
        ridge_vec = ridge_end - ridge_start
        ridge_unit = ridge_vec / np.linalg.norm(ridge_vec)

        # ----------------------------------------------------------------------------------------------
        # Horizontal distances to ridge

        def point_proj_distance(pt2d):
            v = pt2d - ridge_start
            proj_len = np.dot(v, ridge_unit)
            proj = ridge_start + proj_len * ridge_unit
            return np.linalg.norm(pt2d - proj)

        # Precompute distances for elevation interpolation
        dists = [point_proj_distance(p[:2]) for p in base]
        #max_dist = max(dists)
        max_dist = (min(dists) + max(dists))/2

        # ----------------------------------------------------------------------------------------------
        # Loop on the vertices of the base

        vertices = []
        face1 = []
        face2 = []

        current_side = None
        ext1, ext2 = False, False

        # Ridge indices
        g0, g1 = n_points, n_points + 1
        # Faces first en last indices but ridge
        i0, i1, j0, j1 = None, None, None, None

        for i, pt in enumerate(base):
            dist = dists[i]
            #factor = max(0.0, 1 - dist / max_dist)
            factor = 1 - dist / max_dist
            z = base_z + ridge_height * factor
            vertex = np.array([pt[0], pt[1], z])
            vertices.append(vertex)

            side = np.cross(ridge_vec, pt[:2] - ridge_start)
            if side >= 0:
                if current_side == 2 and not ext1:
                    i_ridge = len(face1)
                    face1.extend([g1, g0])
                    ext1 = True
                face1.append(i)
                current_side = 1
            else:
                if current_side == 1 and not ext2:
                    j_ridge = len(face2)
                    face2.extend([g0, g1])
                    ext2 = True
                
                face2.append(i)
                current_side = 2

        if not ext1:
            i_ridge = len(face1)
            face1.extend([g1, g0])
        if not ext2:
            J_ridge = len(face2)
            face2.extend([g0, g1])

        vertices = np.append(vertices, ridge_line, axis=0)

        # ----- Gables
        i0 = face1[(i_ridge + 2)%len(face1)]
        i1 = face1[(i_ridge - 1)%len(face1)]
        j0 = face2[(j_ridge - 1)%len(face2)]
        j1 = face2[(j_ridge + 2)%len(face2)]

        faces = [list(reversed(face1)), list(reversed(face2)), [g0, i0, j0], [g1, j1, i1]]

        return vertices, faces    
        
    # ====================================================================================================
    # Elevate buildings

    def get_buildings(self, layer_name, roof_incr=1., seed=0):

        rng = np.random.default_rng(seed)

        selection = self.select(layer_name)

        all_verts = np.zeros((0, 3), np.float32)
        all_faces = []

        for idx, row in selection.iterrows():

            gtype = row.geometry.geom_type
            if gtype != 'Polygon':
                raise Exception(f"Unsupported geometry type: {gtype}")
            
            # --------------------------------------------------
            # Create the ridge line
            # --------------------------------------------------   

            # To numpy arrays
            base = np.array(row.geometry.exterior.coords)
            center, angle, ridge0, ridge1 = self.get_house_ridges(base)
            ridge = ridge0

            flat_roof = ridge is None


            # Walls height
            wall_height = 3. if np.isnan(row['HAUTEUR']) else row['HAUTEUR']

            # Roof height
            if np.isnan(row['Z_MIN_TOIT']) or np.isnan(row['Z_MAX_TOIT']):
                flat_roof=True
            else:
                roof_height = row['Z_MAX_TOIT'] - row['Z_MIN_TOIT']
                if roof_height < .5:
                    flat_roof = True
                else:
                    roof_height += roof_incr

            # Adjust z
            if self.rge_alti is None:
                z_base = np.min(base[:, 2])
            else:
                true_alts = self.rge_alti.load_altitudes(base[:, 0], base[:, 1])
                z_base = np.min(true_alts[:, 2])

            z_base += wall_height

            base[:, 2] = z_base
            if not flat_roof:
                ridge[:, 2] = z_base + roof_height

            # --------------------------------------------------
            # Flat roof or V Roof
            # --------------------------------------------------

            if flat_roof:
                roof_verts = np.array(base)
                faces = [[i for i in reversed(range(len(base)))]]

            else:
                roof_verts, faces = self.compute_roof_from_base_and_ridge(base, ridge)

            # --------------------------------------------------
            # Extrude walls
            # --------------------------------------------------

            n = len(base)
            ceil_verts = np.array(roof_verts[:n])
            ceil_verts[:, 2] = z_base - wall_height - 10

            verts = np.append(roof_verts, ceil_verts, axis=0)

            for i in range(n):
                faces.append([i, (i+1)%n, len(roof_verts) + ((i+1)%n), len(roof_verts) + i])

            faces.append(np.arange(n) + len(roof_verts))

            # --------------------------------------------------
            # Add to total
            # --------------------------------------------------

            offset = len(all_verts)
            all_verts = np.append(all_verts, verts, axis=0)
            for face in faces:
                all_faces.append([offset + i for i in face])
            
        return all_verts, all_faces
    
    # ====================================================================================================
    # Woods

    def get_woods(self, layer_name, density=.1, seed=0):
    
        selection = self.select(layer_name)

        points = None
        
        for idx, row in selection.iterrows():
                
            pts = self.poisson_points_in_polygon(row.geometry, density=density)

            if points is None:
                points = pts
            else:
                points = np.append(points, pts, axis=0)

        return points


if __name__ == "__main__":
    TOPO_DIR = "/Users/alain/cache/BDTOPO/D 088"
    bx, by = 989752, 6791820
    area = Area.Center(bx, by, 700, projection="LAMB93")


    bdtopo = BDTopo(area, TOPO_DIR)

    #print(bdtopo)

    #bdtopo.dump_layer("TRONCON_DE_ROUTE")
    ms = bdtopo.get_by_column("TRONCON_DE_ROUTE", "NOM_COLL_G")
    for k, v in ms.items():
        print(k, type(v))
        if isinstance(v, LineString):
            print(np.array(v.coords).shape)
        else:
            print([np.array(line.coords).shape for line in v.geoms])



