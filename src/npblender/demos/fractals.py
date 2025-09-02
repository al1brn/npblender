import numpy as np

from npblender import (blender, Mesh, Curve, Animation, Instances, Camera, FieldArray, Simulation, Point)
from npblender.blender import bfloat, bint, bbool
from npblender.maths import distribs, get_angled


def demo(name="Splitter", depth=3, max_depth=10):

    if name.lower() == "sierpinski":
        f = Sierpinski(max_depth=max_depth, max_size=.01, thread_factor=.7)
        f.go()

    elif name.lower() == 'splitter':
        Fractal.Demo()

    elif name.lower() == 'tree':

        tree = Branches()
        tree.go()


    else:
        raise Exception(f"Unknown fractal demo name: '{name}'")

# ====================================================================================================
# Step on a selection fo faces
#
#         x v0
#        /  \
#       n0   n2
#      /      \
# v1 x----x----x v2
#         n1
#
# v0 : index
# v1 : index + 1
# v2 : index + 2
# n0 : nverts:nverts + nfaces
# n1 : n0 + nfaces
# n2 : n1 + nfaces
#
# Stretch
# - v1 -> n0 
# - v2 -> n2 
# New
# - n0, v1, n1
# - n2, n1, v2


class Sierpinski(Animation):
    def __init__(self, max_depth=12, max_size=.01, thread_factor=.7):
        self.max_depth = max_depth
        self.max_size  = max_size
        self.thread_factor = thread_factor

    def sierpinski_step(self, triangle, selection=True, depth=0):
        
        # Number of vertices
        nverts = len(triangle.points)
        
        # Number of corners
        ncorners = len(triangle.corners)
        
        # Number of faces
        nfaces = len(triangle.faces)
        assert(ncorners==nfaces*3)
        
        selection = np.resize(selection, nfaces)
        
        # ----------------------------------------------------------------------------------------------------
        # Maximum number of vertices and corners to be created
        
        new_verts   = np.empty((nfaces*3, 3), float)
        new_corners = np.empty((nfaces*2, 3), int)

        # ----------------------------------------------------------------------------------------------------
        # The couple (Depth, Order) is unique and always the same whatever the camera culling
        # At a given depth, the Order is in [0, 3^depth - 1]
        # The 3 children of a given order are order*3 + (0, 1, 2)
        
        new_order   = np.empty(nfaces*2, int)
        new_thread  = np.empty(nfaces*2, float)
        
        # ----------------------------------------------------------------------------------------------------
        # Loop on the faces
        
        # Shortcut
        vertices = triangle.points.position 
        
        v_count = 0 # triplets counter (+3 per step)
        c_count = 0 # new corners triplets counter (+2 per step)
        for face, sel in zip(triangle.faces, selection):
            
            if not sel:
                continue

            # ----------------------------------------------------------------------------------------------------
            # Face order and random 
            
            face_order  = int(face.Order)
            h = hash((depth, face_order))
            rng = np.random.default_rng(abs(hash((depth, face_order))))
            threads = face.Thread + rng.uniform(-1, 1, 3)*(self.thread_factor**(depth+1))
            
            # ----------------------------------------------------------------------------------------------------
            # Current face vertices
            
            index = face.loop_start
            v0 = triangle.corners.vertex_index[index] 
            v1 = triangle.corners.vertex_index[index + 1] 
            v2 = triangle.corners.vertex_index[index + 2] 
            
            # ----------------------------------------------------------------------------------------------------
            # Create the new vertices : 3 new vertices per face

            new_verts[v_count]     = (vertices[v0] + vertices[v1])/2
            new_verts[v_count + 1] = (vertices[v1] + vertices[v2])/2
            new_verts[v_count + 2] = (vertices[v2] + vertices[v0])/2
        
            # ----------------------------------------------------------------------------------------------------
            # New triangles corners
            
            n0 = nverts + v_count
            n1 = n0 + 1
            n2 = n0 + 2
            
            new_corners[c_count:c_count + 2] = [[n0, v1, n1], [n2, n1, v2]]
            
            face.Depth = depth + 1
            face.Order *= 3

            new_order[c_count]    = face_order*3 + 1
            new_order[c_count+1]  = face_order*3 + 2
            new_thread[c_count]   = threads[1]
            new_thread[c_count+1] = threads[2]
            
            # ----------------------------------------------------------------------------------------------------
            # Stretch current faces
            
            triangle.corners.vertex_index[index + 1] = n0
            triangle.corners.vertex_index[index + 2] = n2
            face.Depth += 1
            face.Thread = threads[0]
            
            # Increment counters
            
            v_count += 3
            c_count += 2
        
        # ----------------------------------------------------------------------------------------------------
        # Add vertices and faces

        if True:
            triangle.add_geometry(
                points = new_verts[:v_count],
                corners = new_corners[:c_count].ravel(),
                faces = 3,
                Depth=depth + 1,
                Thread=new_thread[:c_count],
            )
        else:
            triangle.points.append(position=new_verts[:v_count])
            triangle.add_faces(corners=new_corners[:c_count].ravel(), faces=3, Depth=depth + 1, Thread=new_thread[:c_count])

    def compute(self):

        # ----------------------------------------------------------------------------------------------------
        # Create the initial triangle

        triangle = Mesh.disk(segments=3, materials=["Fractal"])
        
        # The couple (Depth, Order) is unique and always the same    
        triangle.faces.new_int("Depth",    default=0)  # Fractal depth
        triangle.faces.new_int("Order",    default=0)  # Generation order within the depth
        triangle.faces.new_float("Thread", default=1.) # Thread value

        # ----------------------------------------------------------------------------------------------------
        # Fractal loop
    
        cam = Camera()
        for i in range(self.max_depth):
            vis, sizes, _ = cam.visible_faces(triangle)
            vis &= sizes > self.max_size
            if not np.any(vis):
                break
            self.sierpinski_step(triangle, vis, depth=i)

        self.triangle = triangle

    def view(self):            
        self.triangle.to_object("Sierpinski")

    def get_frame_data(self):
        return self.triangle.as_dict()
    
    def set_frame_data(self, data):
        self.triangle = Mesh.FromDict(data)


# ====================================================================================================
# ====================================================================================================
# Fractal
# ====================================================================================================
# ====================================================================================================

# ====================================================================================================
# Splitter

class Splitter:

    def __init__(self, master, division, vertices=None, base=[0, 1, 2, None], faces=None):
        """ Split a "master" geometry into "split" geometries

        To divide a master geometry made of n corners:
        - new vertices are created
        - a list of count*n corners is created with corners from master and corners pointing
          to new vertices

        Example with a Sierpinski triangle:

                  0
                /   \
               1     5
              /       \
             2 -- 3 -- 4

        - master   = [0, 2, 4]
        - division = [0, 1, 5, 1, 2, 3, 5, 3, 4] # 3 * 3

        The division corners are ordered such as to be reshaped in (count, n) corners, each
        line providing the master for further division:
        - division = [
            [0, 1, 5], # sub master 0
            [1, 2, 3], # sub master 1
            [5, 3, 4], # sub master 2
            ]

        The division has two sets of indices:
        - master indices : indices belonging to master, e.g. : [0, 2, 4]
        - split indices : indices not belonging to master, e.g : [1, 3, 5]

        Master corners
        --------------

        Unique corners are extracted from master in their order of appearance. To be reproductible, 
        the extraction is made with the array master_i_unique:

        master            master_i_unique   master_unique = master[master_i_unique]
        [0, 1, 2]         [0, 1, 2]         [0, 1, 2]
        [2, 1, 0]         [0, 1, 2]         [2, 1, 0]
        [7, 8, 9]         [0, 1, 2]         [7, 8, 9]
        [7, 8, 8, 9]      [0, 1, 3]         [7, 8, 9]

        The values of master corners appearing in division, are replaced by their index in master_i_unique:
        - master :            [2, 4, 4, 1]
        - master_i_unique :   [0, 1, 3]
        - division :          [. . 1 . . 2 . . 2 . . 4]
        - division_template : [. . 3 . . 0 . . 0 . . 1]

        Actual new_division is obtained with:
        - new_divison[master_sel] = master[division_template[master_sel]]

        Creating new vertices
        ---------------------
        
        To create new vertices, we need the array of vertices. The master vertices are extracted from
        vertices with master_unique:
        - master_vertices = vertices[master_unique]

        The master vertices are used to create a local base. At initialization time, the new vertices
        are expressed in local base (O, M). At division time:
        - local base (O', M') is computed from master and vertices
        - new vertices are created from (O', M')
        
        New indices
        -----------
        Split indices are ordered using np.unique. New vertices are created in this order. For instance,
        if split corners in division are [3 0 . 5 3 . 6 0 . 0 5 .] the new vertices will be created in the order
        [0, 3, 5, 6].

        In the division template, split indices are replaced by their index in the unique list:
        - split indices :     [3 0 . 5 3 . 6 0 . 0 5 .]
        - division_template : [1 0 . 2 1 . 3 0 . 0 2 .]

        The new vertices being appended to the current list of vertices, the actual indices are computed
        with a simple shift:
        - new_division[split_sel] += offset

        Summary
        -------
        - master            : initial list of master corners
        - division          : initial list of division corners
        - master_sel        : selection of master corners in division
        - master_i_unique   : indices of unique master corners within master
        - division_template : template to compute a division for a given master
        - master_map        : precomputation of division_template[master_i_unique]
        - new_division      : created with : division_template + offset
        - master indices    : new_division[master_sel] = master[master_map]

        Arguments
        ---------
        - master (array of ints) : the corners forming the master
        - division (array of ints) : corners of divided geometry
        - vertices (array of vectors = None) : geometry vertices
        - base (list[4] of ints = [0, 1, 2, None]) : the vertices to use to compute the local base
        - faces (list of ints = None) : master geometry face sizes
        """

        # ----------------------------------------------------------------------------------------------------
        # Dimenions

        if len(division) % len(master) != 0:
            raise ValueError(f"Splitter() error: the length of division ({len(division)}) is not a multiple of the length of master ({len(master)})")

        self.master = np.array(master)
        self.division = np.array(division).flatten()
        self.count = len(self.division) // len(self.master)

        # Number of points
        self.master_npoints = len(np.unique(self.master))
        self.split_npoints  = len(np.unique(self.division)) - self.master_npoints

        # ----------------------------------------------------------------------------------------------------
        # Master corners in division

        self.master_sel = np.isin(self.division, self.master)
        self.split_sel  = np.logical_not(self.master_sel)

        # ----------------------------------------------------------------------------------------------------
        # Master unique corners
        # master_unique = master[master_i_unique]

        idx = []
        crn = []
        for i, c in enumerate(self.master):
            if not c in crn:
                idx.append(i)
                crn.append(c)
        self.master_i_unique = np.array(idx)

        # ----------------------------------------------------------------------------------------------------
        # Master corner indices in division template
        # To be used in:
        # - new_division[master_sel] = master[master_map]

        conv = {self.master[index_in_master]: index_in_master for index_in_master in self.master_i_unique}
        self.master_map = [conv[c] for c in self.division[self.master_sel]]

        # ----------------------------------------------------------------------------------------------------
        # Division template

        self.division_template = np.zeros_like(self.division)
        usplits = np.unique(self.division[self.split_sel])
        self.division_template[self.split_sel] = [list(usplits).index(c) for c in self.division[self.split_sel]]

        # ----------------------------------------------------------------------------------------------------
        # Split vertices in local base
        
        self.vertices   = None  # Reference vertices
        self.base       = None  # Local base indices
        self.faces      = faces # Face sizes for master geometry
        self.center     = None  # Geometry center
        self.size       = None  # Geometry size
        self.size_ratio = 1.    # Ratio between size and length of first vector in local base
        self.size_scale = None  # Division scale

        # Initialization if vertices are provided
        if vertices is not None:
            self.compute_new_points(vertices, base=base)

    # ====================================================================================================

    def __str__(self):
        return f"<Splitter {len(self.master)} split {self.count} times -> {len(self.division)} corners, {self.master_npoints} -> {self.split_npoints} new points>"

    @staticmethod
    def sarray(arr, sel=None):
        if sel is None:
            sel = np.ones(len(arr), bool)
        a = [f"{i:2d}" if ok else " ."  for i, ok in zip(arr, sel)]
        return "[" + f"{' '.join(a)}" + "]"

    def __repr__(self):

        s = "<Splitter {len(self.master)} split {self.count} times -> {len(self.division)} corners, {self.master_npoints} -> {self.split_npoints} new points"

        s += "\n"
        s += f"\n master :   {self.sarray(self.master)}"
        s += f"\n division : {self.sarray(self.division)}"
        s += f"\n    master: {self.sarray(self.division, self.master_sel)}"
        s += f"\n    split : {self.sarray(self.division, self.split_sel)}"

        s += f"\n template : {self.sarray(self.division_template, self.split_sel)}"

        check = self.division_template + self.master_npoints
        check[self.master_sel] = self.master[self.master_map]

        s += "\n"
        s += f"\n division : {self.sarray(self.division)}"
        s += f"\n CHECK :    {self.sarray(check)}"
        s += f"\n    master: {self.sarray(check, self.master_sel)}"
        s += f"\n    split : {self.sarray(check, self.split_sel)}"

        s += "\n>"

        return s

    # ==================================================================================================
    # Get the local base

    # -------------------------------------------------------------------------------------------------
    # Single base

    def get_local_base(self, vertices):
        """ Compute the local base from the given vertices.

        The vertices are the master vertices. They are provided in the order of
        corner appearance in master:
        - vertices = all_vertices[master[master_i_unique]]

        Arguments
        ---------
        - vertices (array of vectors) : the master vertices
        """

        O = vertices[self.base[0]]

        vecs = vertices - O
        ux = None if self.base[1] is None else vecs[self.base[1]]
        uy = None if self.base[2] is None else vecs[self.base[2]]
        uz = None if self.base[3] is None else vecs[self.base[3]]

        if ux is None:
            ux = np.cross(uy, uz)/np.linalg.norm(uz)
        elif uy is None:
            uy = np.cross(uz, ux)/np.linalg.norm(ux)
        elif uz is None:
            uz = np.cross(ux, uy)/np.linalg.norm(uy)

        if False:
            print("----- GLB")
            print(ux)
            print(uy)
            print(uz)
            print()

        return O, np.vstack([ux, uy, uz])
    
    # ----------------------------------------------------------------------------------------------------
    # Several bases at once
    
    def get_local_bases(self, vertices):
        """ Compute the local bases from the given vertices.

        The vertices are the master vertices. They are provided in the order of
        corner appearance in master:
        - vertices = all_vertices[master[master_i_unique]]

        Arguments
        ---------
        - vertices (array of array of vectors) : the master vertices
        """

        ndivs = vertices.shape[0] # Number of divisions

        O = vertices[:, self.base[0]]

        vecs = vertices - O[:, None]
        ux = None if self.base[1] is None else vecs[:, self.base[1]]
        uy = None if self.base[2] is None else vecs[:, self.base[2]]
        uz = None if self.base[3] is None else vecs[:, self.base[3]]

        if ux is None:
            ux = np.cross(uy, uz)/np.linalg.norm(uz, axis=1, keepdims=True)
        elif uy is None:
            uy = np.cross(uz, ux)/np.linalg.norm(ux, axis=1, keepdims=True)
        elif uz is None:
            uz = np.cross(ux, uy)/np.linalg.norm(uy, axis=1, keepdims=True)

        return O, np.stack((ux, uy, uz), axis=1)    
    
    # ==================================================================================================
    # Set the vertices to compute the new points coordinates

    def compute_new_points(self, vertices, base=[0, 1, 2, None]):
        """ Compute the new points coordinates

        The base arguments provides indices to compute the local base:
        - index 0 : index of the point to use as origin
        - indices 1 to 3 : index of the points to use as vectors

        One of the index can be None. In that case the missing axis is computed using
        cross product of the other two.

        Arguments
        ---------
        - vertices (array of vectors) : the reference vertices
        - base (list(4) of ints) : Origin plus 3 vectors
        """

        n = len(np.unique(self.division))
        if len(vertices) < n:
            raise ValueError(f"Splitter.set_vertices error: the number of vertices {len(vertices)} is less than the required number of points {n}")

        self.vertices = np.array(vertices)
        self.base = list(base)

        umaster = self.master[self.master_i_unique]
        O, M = self.get_local_base(vertices[umaster])

        sel = np.unique(self.division[self.split_sel])
        self.new_points = np.linalg.solve(M.T, (vertices[sel] - O).T).T

        # ----------------------------------------------------------------------------------------------------
        # Center, size and size_scale

        # Center and sizes are computed on the divided geometry
        pmin, pmax = np.min(vertices, axis=0), np.max(vertices, axis=0)
        center = (pmin + pmax) / 2
        self.center = np.linalg.solve(M.T, (center - O).T).T
        self.size    = max(np.linalg.norm(pmax - center), np.linalg.norm(pmax - center))

        # Centers and sizes are then computed for each division
        self.centers     = np.empty((self.count, 3), float)
        self.sizes       = np.ones(self.count, float)
        self.size_ratio  = self.size/np.linalg.norm(M[0])

        for i, sub_master in enumerate(self.division.reshape(self.count, len(self.master))):
            sub_O, sub_M = self.get_local_base(vertices[sub_master[self.master_i_unique]])
            sub_center = sub_O + self.center @ sub_M
            self.centers[i] = np.linalg.solve(M.T, (sub_center - O).T).T
            self.sizes[i] = np.linalg.norm(sub_M[0])*self.size_ratio

        self.size_scale = self.sizes[0]/self.size

        # ----------------------------------------------------------------------------------------------------
        # Debug visualization

        if False:
            insts = Instances(
                points = O + self.centers @ M,
                models = Mesh.UVSphere(radius=.05))
            insts.realize()['mesh'].to_object("NEW CENTERS")

            insts = Instances(
                points = O + self.centers @ M,
                models = Mesh.UVSphere(radius=1),
                Scale  = self.sizes[:, None])
            insts.realize()['mesh'].to_object("NEW SIZES")

    # ==================================================================================================
    # Get the master and split meshes

    def get_master_mesh(self):
        """ Master mesh from vertices if exist
        """

        if self.vertices is None:
            raise Exception("Fractal.get_master_mesh error: the vertices are not set")
        
        umaster = self.master[self.master_i_unique]
        verts = self.vertices[umaster]
        
        if self.faces is None:
            mesh = Mesh(points=verts)
        else:
            master_map = [list(umaster).index(c) for c in self.master]
            mesh = Mesh(points=verts, corners=master_map, faces=self.faces)

        return mesh

    def get_split_mesh(self):
        """ Recompute split mesh from vertices if exist

        For debug purpose
        """

        if self.vertices is None:
            raise Exception("Fractal.get_split_mesh error: the vertices are not set")
        
        # ----- Master vertices

        umaster = self.master[self.master_i_unique]
        verts = self.vertices[umaster]

        # ----- Split vertices

        O, M = self.get_local_base(verts)
        new_vertices = O + self.new_points @ M

        verts = np.append(verts, new_vertices, axis=0)

        # ----- Split corners

        new_division = self.division_template + self.master_npoints
        new_division[self.master_sel] = [list(umaster).index(c) for c in self.master] #self.master_map

        if self.faces is None:
            return Mesh(verts=verts)
        else:
            return Mesh(verts=verts, corners=new_division, sizes=np.tile(self.faces, self.count))
        
    def test_split(self):

        vertices = FieldArray()
        vertices.new_field("position", float, 3)

        umaster = self.master[self.master_i_unique]
        vertices.append(position=self.vertices[umaster])

        master = np.array([list(umaster).index(c) for c in self.master])

        corners = self.split(master, vertices)

        if self.faces is None:
            return Mesh(points=vertices.position)
        else:
            return Mesh(points=vertices.position, corners=corners, sizes=np.tile(self.faces, self.count))
        
    # =============================================================================================================================
    # Split
        
    def split(self, master, vertices):
        """ Split step

        Arguments
        ---------
        - master (array of ints) : the master to divide
        - vertices (DynamicRecArray with position field) : all the vertices

        Note : new vertices are added to vertices dynamic array

        Returns
        -------
        - array of ints : the division corners
        """

        # Unique master vertices from the array of vertices
        master_verts = vertices.position[master[self.master_i_unique]]

        # Create new vertices using local base
        O, M = self.get_local_base(master_verts)
        new_vertices = O + self.new_points @ M

        # Append the new vertices
        offset = len(vertices)
        vertices.append_attributes(position=new_vertices)

        # New indices for split geometry
        new_division = self.division_template + offset
        new_division[self.master_sel] = master[self.master_map]

        return new_division
    
    # =============================================================================================================================
    # Split
        
    def splits(self, masters, vertices, sizes=None):
        """ Split several masters at once

        Arguments
        ---------
        - masters (array (ndivs, n) of ints) : the list of ndivs masters to divide
        - vertices (array of vectors) : all the vertices
        - centers (array of vectors) : geometry position
        - sizes (array of floats) : geometry size

        Returns
        -------
        - dictionary : "masters", "vertices", "centers", "sizes"
        """

        ndivs = masters.shape[0]
        nvecs = self.split_npoints # create (ndivs, nvecs, 3) vectors

        # Unique master vertices from the array of vertices
        master_verts = vertices[masters[:, self.master_i_unique]]

        # Create new vertices using local base
        O, M = self.get_local_bases(master_verts)
        new_vertices = O[:, None, :] + np.einsum('ij,njk->nik', self.new_points, M)

        # Append the new vertices
        offset = len(vertices)

        # New indices for split geometry
        offsets = offset + np.arange(ndivs)[:, None] * nvecs
        new_divisions = self.division_template[None, :] + offsets
        new_divisions[:, self.master_sel] = masters[:, self.master_map]

        # Centers and sizes
        centers = O[:, None, :] + np.einsum('ij,njk->nik', self.centers, M)
        new_sizes = sizes[:, None] * self.size_scale * np.ones((1, self.count)) 

        total = ndivs*self.count

        return {"vertices" : new_vertices.reshape(ndivs*nvecs, 3),
                "masters"  : new_divisions.reshape(total, len(self.master)),
                "centers"  : centers.reshape(total, 3), 
                "sizes"    : new_sizes.reshape(total)}


    # =============================================================================================================================
    # Get corners and faces
        
    def get_corners_and_faces(self, divisions):

        n = len(divisions)
        if self.faces is None:
            faces = [len(self.master)]*n
        else:
            faces = np.tile(self.faces, n)

        return divisions.flatten(), faces
    
    # =============================================================================================================================
    # Setup for a start mesh

    def setup_start_mesh(self, mesh):

        if len(mesh.corners) % len(self.master):
            raise Exception(f"Splitter.setup_start_mesh error : the number of corners {len(mesh.corners)} is not a multtiple of the master length ({len(self.master)})")
        
        n = len(mesh.corners) // len(self.master)
        masters = np.array(mesh.corners.vertex_index.reshape(n, len(self.master)))

        vertices = np.array(mesh.points.position)

        centers = np.zeros((n, 3), float)
        sizes = np.ones((n,))

        if self.centers is not None and self.sizes is not None:
            for i in range(n):
                master = masters[i]
                master_verts = vertices[master[self.master_i_unique]]
                O, M = self.get_local_base(master_verts)

                centers[i] = O + self.center @ M
                sizes[i] = self.size_ratio * np.linalg.norm(M[0])

        return {"masters"  : masters,
                "vertices" : vertices,
                "centers"  : centers, 
                "sizes"    : sizes}


    # =============================================================================================================================
    # =============================================================================================================================
    # Constructors
    # =============================================================================================================================
    # =============================================================================================================================
    
    # =============================================================================================================================
    # Sierpinski triangle
    
    @classmethod
    def Sierpinski(cls):
        
        # ----------------------------------------------------------------------------------------------------
        # Splitter for Sierpinski triangle
        
        s3 = np.sqrt(3.)
        A = np.array((0, s3 / 3, 0))
        B = np.array((-0.5, -s3 / 6, 0))
        C = np.array(( 0.5, -s3 / 6, 0))
        verts = np.vstack([A, (A + B)/2, B, (B + C)/2, C, (C + A)/2])

        return cls(master=[0, 2, 4], division=[0, 1, 5, 1, 2, 3, 5, 3, 4], 
                vertices=verts, base=[0, 1, 2, None], faces=[3])
    
    # =============================================================================================================================
    # Square

    @classmethod
    def Square(cls):
        
        t = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(t, t)

        verts = np.stack([xx.ravel(), yy.ravel(), np.zeros(16)], axis=1)

        splitter = cls(
            master   = [0, 3, 15, 12],
            division = [0, 1, 5, 4, 2, 3, 7, 6, 5, 6, 10, 9, 8, 9, 13, 12, 10, 11, 15, 14],
            vertices = verts,
            faces    = [4],
        )

        return splitter
    
    # =============================================================================================================================
    # Cube

    @classmethod
    def Cube(cls):

        t = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(t, t)

        verts = np.stack([xx.ravel(), yy.ravel(), np.zeros(16)], axis=1)

        center = verts[[5, 6, 10, 9]] + (0, 0, 2/3)
        verts = np.append(verts, center, axis=0)

        splitter = cls(
            master   = [0, 3, 15, 12],
            division = [ 0,  1,  5,  4,  1,  2,  6,  5,  2,  3,  7,  6, 
                         4,  5,  9,  8,                  6,  7, 11, 10,
                         8,  9, 13, 12,  9, 10, 14, 13, 10, 11, 15, 14,
                         5,  6, 17, 16,  6, 10, 18, 17, 10,  9, 19, 18,  9,  5, 16, 19,
                         16,17, 18, 19
                       ],
            vertices = verts,
            faces    = [4],
        )

        return splitter

    # =============================================================================================================================
    # Tetrahedron

    @classmethod
    def Tetrahedron(cls):

        def pyr(corners):
            s, a, b, c = corners
            return [s, a, b,   s, b, c,  s, c, a,  c, b, a]

        S = np.array([ 1,  1,  1])
        A = np.array([-1, -1,  1])
        B = np.array([-1,  1, -1])
        C = np.array([ 1, -1, -1])

        verts = np.vstack([S, A, B, C, (S + A)/2, (S + B)/2, (S + C)/2, (A + B)/2, (B + C)/2, (C + A)/2])

        splitter = cls(
            master   = pyr([0, 3, 2, 1]),
            division = pyr([0, 6, 5, 4]) + pyr([5, 8, 2, 7]) + pyr([4, 9, 7, 1]) + pyr([6, 3, 8, 9]),
            base     = [0, 1, 2, 3],
            vertices = verts,
            faces    = [3, 3, 3, 3],
        )

        return splitter
    
    # =============================================================================================================================
    # Pyramid

    @classmethod
    def Pyramid(cls):

        c30 = np.sqrt(3)/2
        s30 = 1/2

        A = np.array([-c30, -s30,  0])
        B = np.array([ c30, -s30,  0])
        C = np.array([   0,   1.,  0])
        D, E, F = (A + B)/2, (B + C)/2, (C + A)/2
        S = (0, 0, np.sqrt(.5)) 

        #                  0  1  2  3  4  5  6
        verts = np.vstack([S, A, B, C, D, E, F])

        splitter = cls(
            master   = [1, 2, 3],
            division = [1, 4, 6,   4, 2, 5,  6, 5, 3,  4, 5, 0,    5, 6, 0,   6, 4, 0],
            base     = [0, 1, 2, None],
            vertices = verts,
            faces    = [3],
        )

        return splitter
    
    
# ----------------------------------------------------------------------------------------------------
# Tetrahedron splitter

class TetrahedronSplitter(Splitter):

    def __init__(self):
        """ Tetrahedron constructor divides a fully defined tetrahedron.

        This version devides only the 4 corners of a tetrahedron. The faces
        are buit once all divisions are completed.
        """

        sp = Splitter.Tetrahedron()

        super().__init__(
            master   = [0, 1, 2, 3],
            division = [0, 6, 5, 4, 5, 8, 2, 7, 4, 9, 7, 1, 6, 3, 8, 9],
            base     = [0, 1, 2, 3],
            vertices = sp.vertices,
            faces    = [4], # Dummy, to have a consistent starting mesh
        )

    # =============================================================================================================================
    # Get corners and faces
        
    def get_corners_and_faces(self, divisions):

        n = len(divisions)
        corners = np.empty((n, 4, 3), int)
        corners[:, 0] = divisions[:, [0, 2, 1]]
        corners[:, 1] = divisions[:, [0, 3, 2]]
        corners[:, 2] = divisions[:, [0, 1, 3]]
        corners[:, 3] = divisions[:, [1, 2, 3]]

        return corners.flatten(), [3]*(4*n)


# ====================================================================================================
# Fractal using a splitter

class Fractal():
    def __init__(self, splitter, mesh=None):

        # ----- Splitter

        self.splitter = splitter

        # ----- Vertices

        self.vertices = FieldArray()
        self.vertices.new_field("position", float, 3)

        # ----- Indices

        self.items = FieldArray()

        self.items.new_field("corners", bint, len(self.splitter.master))
        self.items.new_field("center",  bfloat, 3)
        self.items.new_field("size",    bfloat)

        self.items.new_field("depth",   bint)
        self.items.new_field("uid",     bint)

        self.items.new_field("thread",  bfloat)
        self.items.new_field("random",  bfloat)

        # ----- Pseudo hash is computed for each item

        base_multiplier = np.uint32(1103515245)
        base_increment  = np.uint32(1013904223)
    
        # Create count multipliers and offset
        i = np.arange(self.splitter.count, dtype=np.uint32)
        self.uid_mult = (base_multiplier * (i + 1)) ^ 0x5bd1e995
        self.uid_incr = base_increment * i

        # ----- Start mesh

        self.start_mesh = None
        if mesh is None:
            self.start_mesh = self.splitter.get_master_mesh()

        else:
            self.start_mesh = mesh

        self.split_reset()

    # ====================================================================================================
    # Split

    def split_reset(self):

        mesh_setup = self.splitter.setup_start_mesh(self.start_mesh)

        self.vertices.clear()
        self.vertices.append(position=mesh_setup['vertices'])

        self.items.clear()
        self.items.append(
            corners = mesh_setup['masters'],
            center  = mesh_setup['centers'],
            size    = mesh_setup['sizes'],
            depth   = 0,
            uid     = np.arange(len(mesh_setup['masters'])),
            thread  = 0., 
        )
        #self.final = FieldArray.from_other(self.items, copy=False)
        self.final = FieldArray(self.items, mode='EMPTY')
        if False:
            n = len(self.start_mesh.corners) // len(self.splitter.master)

            self.items.clear()
            self.items.append_attributes(corners=self.start_mesh.corners.vertex_index.reshape(n, len(self.splitter.master)))

    def split(self, depth=3, camera=None, thread=.3, max_vertices=10000000, max_size=.05):

        self.split_reset()

        # ----------------------------------------------------------------------------------------------------
        # Loop on depth

        thread_factor = thread
        for idepth in range(depth):

            # ----------------------------------------------------------------------------------------------------
            # Visible items

            if camera is not None:
                vis, size = camera.visible_points(self.items.center, radius=self.items.size)
                sel = vis[:, camera.VISIBLE]
                sel &= size[:, camera.SIZE] >= max_size

                total = len(self.final) + len(self.items)

                self.final.append_other(self.items[np.logical_not(sel)])
                self.items = self.items.extract(sel)

                assert(total == len(self.final) + len(self.items))

            # ----------------------------------------------------------------------------------------------------
            # Split the visible items

            d = self.splitter.splits(
                masters  = self.items.corners,
                vertices = self.vertices.position,
                sizes    = self.items.size)
            
            # ----------------------------------------------------------------------------------------------------
            # Update the array

            if len(d['vertices']):
                self.vertices.append(position=d['vertices'])

            n = len(d['masters'])
            if n:
                new_items = FieldArray(self.items, mode='EMPTY')

                # ----- Items uid
                hashes = (self.items.uid[:, None] * self.uid_mult[None, :] + self.uid_incr[None, :]) & 0x7FFFFFFF  # force 31 bits for positive values
                uid = hashes.flatten().astype(np.int32)
                del hashes
                
                # ----- Items random value
                temp = (uid * 2654435761) & 0xFFFFFFFF
                rands = (temp / 2**32).astype(np.float32)
                del temp

                # ----- Thread
                temp = (uid * 2246822519) & 0xFFFFFFFF
                th = (temp / 2**32).astype(np.float32)*2 - 1
                threads = self.items.thread[:, None] + thread_factor*np.reshape(th, (len(self.items), self.splitter.count))
                del temp, th

                new_items.append(
                    corners = d['masters'],
                    center  = d['centers'],
                    size    = d['sizes'],
                    
                    depth   = idepth,
                    uid     = uid,
                    random  = rands,
                    thread  = threads.ravel(),
                    )
                
                del self.items
                self.items = new_items

            if len(self.vertices) > max_vertices:
                print(f"Fractal.split stop at {len(self.vertices)} vertices, max is {max_vertices}.")
                break

            thread_factor *= thread


    # =============================================================================================================================
    # Get the mesh

    def get_mesh(self, materials="Fractal"):

        #items = self.final.clone()
        #items.append_other(self.items)
        items = FieldArray(self.final)
        items.extend(self.items)

        nfaces = len(self.splitter.faces)

        corners, faces = self.splitter.get_corners_and_faces(items.corners)

        mesh = Mesh(points=self.vertices.position, corners=corners, faces=faces, materials=materials)

        mesh.faces.new_int("depth", 0)
        mesh.faces.new_int("uid", 0)
        mesh.faces.new_float("random", 0)
        mesh.faces.new_float("thread", 0)

        mesh.faces.depth  = np.tile(items.depth,  nfaces)
        mesh.faces.uid    = np.tile(items.uid,    nfaces)
        mesh.faces.random = np.tile(items.random, nfaces)
        mesh.faces.thread = np.tile(items.thread, nfaces)

        return mesh
    
    # =============================================================================================================================
    # Demo

    @classmethod
    def Demo(cls, depth=3):

        for i_splitter, name in enumerate(("Sierpinski", "Square", "Tetrahedron", "Cube", "Pyramid")):
            spl = getattr(Splitter, name)
            fractal = Fractal(spl())
            fractal.split(depth=depth)
            mesh = fractal.get_mesh()
            obj = mesh.to_object(name, shade_smooth=False)
            obj.location = (i_splitter*3 - 6, 0, 0)


# ====================================================================================================
# Organic fractal

class Branches(Simulation):

    def __init__(self, growing_speed=1, edge_steps=10, max_depth=8, seed=0, **kwargs):

        self.growing_speed = growing_speed
        self.edge_steps = edge_steps
        self.max_depth = max_depth
        self.tree_seed = seed

        # ----- Branch growing

        # Radius from one node to the other
        self.node_radius_factor = .9
        self.node_radius_scale = .1

        # Node angle noise
        self.node_angle = np.radians(10)

        # ----- Children

        self.child_count = 2
        self.child_prob = .3

        # Length from branch to child
        self.child_length_factor = .7
        self.child_length_scale  = .1

        # Radius from branch to child
        self.child_radius_factor = .7
        self.child_radius_scale  = .1

        # Angle from one branch to his child
        self.child_angle = np.radians(30)

        # ----- User choices

        for k, v in kwargs.items():
            setattr(self, k, v)

        # ----- Tips

        self.branch_id = 0

        # Branches tips
        self.tips = Point()

        self.tips.new_vector("position")  # Current position
        self.tips.new_vector("direction") # Current direction
        self.tips.new_float("length")     # Branch current length
        self.tips.new_float("max_length") # Branch length
        self.tips.new_float("radius")     # Branch current radius
        self.tips.new_int("cur_step")     # Current edge step
        self.tips.new_int("depth")        # Current step
        self.tips.new_int("child_count")  # Number of children
        self.tips.new_int("branch_id")    # Current branch id
        self.tips.new_int("last_vertex")  # Last vertex index

        # Resulting Mesh
        # The branches are stored into a mesh because points are inserted and curve
        # is not efficient to insert points into existing splines
        self.mesh = Mesh()
        self.mesh.points.new_float("radius")
        self.mesh.points.new_int("branch_id")

    # ====================================================================================================
    # Add a branch
    # ====================================================================================================

    def add_branch(self, position, direction, max_length, radius, depth):

        # Create the first vertex
        index = len(self.mesh.points)
        self.mesh.points.append(position=position, radius=radius, branch_id=self.branch_id)

        # Add an active branch tip
        self.tips.append(
            position = position,
            direction = direction,
            length = 0,
            max_length = max_length,
            radius = radius,
            cur_step = 0,
            depth = depth,
            branch_id = self.branch_id,
            last_vertex = index,
        )
        self.branch_id += 1

        return self.branch_id - 1
    
    # ====================================================================================================
    # Add a child
    # ====================================================================================================

    def add_child(self, index):

        tip = self.tips[index]
        if tip.depth >= self.max_depth:
            return None
        
        # rng depending on branch id, global seed and child order
        order = int(tip.child_count)
        child_seed = np.random.SeedSequence([int(tip.branch_id), self.tree_seed, order])
        rng = np.random.default_rng(child_seed)
        self.tips[index].child_count += 1

        if True:
            new_dir = get_angled(
                vectors = tip.direction, 
                angle = rng.normal(self.child_angle, self.child_angle*.3), 
                twist = rng.uniform(0, 2*np.pi),
            )
        else:
            # New direction
            new_dir = distribs.dome_dist(
                    radius=1.0,
                    scale=None,
                    axis=self.tips.direction[index],
                    angle=self.child_angle,
                    use_vonmises=False,
                    center=(0, 0, 0),
                    count=1,
                    seed=rng.integers(1<<32),
                )["points"]

        # Child lengths
        new_len = tip.length*rng.normal(self.child_length_factor, self.child_length_scale, 1)

        # Child radius
        new_rad = tip.radius*rng.normal(self.child_radius_factor, self.child_radius_scale, 1)

        # Create the child
        return self.add_branch(
            position=tip.position, 
            direction=new_dir, 
            max_length=new_len, 
            radius=new_rad, 
            depth=tip.depth + 1,
        )


    
    # ====================================================================================================
    # Reset
    # ====================================================================================================

    def reset(self):
        self.mesh.clear_geometry()
        self.tips.clear()
        self.branch_id = 0
        self.add_branch(position=(0, 0, 0), direction=(0, 0, 1), max_length=2, radius=.3, depth=0)

    # ====================================================================================================
    # Grow the active branches
    # ====================================================================================================

    def compute(self):

        rng = np.random.default_rng(self.engine.seed)

        # Delta time and delta length

        dt = self.delta_time
        dl = self.growing_speed*dt

        # Next position
        step = np.minimum(dl, self.tips.max_length - self.tips.length)
        next_pos = self.tips.position + self.tips.direction * step[:, None]

        # Create a new vertex for tips with cur_step == 0
        to_create = []
        mask = self.tips.cur_step == 0
        nbranches = np.sum(mask)
        if nbranches:
            geom = self.mesh.add_geometry(
                points = next_pos[mask],
                radius = self.tips.radius[mask],
                branch_id = self.tips.branch_id[mask],
            )
            self.tips.last_vertex[mask] = geom["points"]

            # Some noise in the directions
            new_dir = distribs.dome_dist(
                radius=1.0,
                scale=None,
                axis=self.tips.direction[mask],
                angle=self.node_angle,
                use_vonmises=False,
                center=(0, 0, 0),
                count=nbranches,
                seed=rng.integers(1<<32),
            )["points"]
            self.tips.direction[mask] = new_dir

            # Randomly create a child
            indices = np.arange(len(self.tips))[mask]
            create = rng.uniform(0, 1, nbranches) < self.child_prob
            to_create = indices[create]

        # Move the last index to next pos
        self.mesh.points.position[self.tips.last_vertex] = next_pos
        self.tips.position = next_pos

        # Update current step
        self.tips.cur_step = (self.tips.cur_step + 1) % self.edge_steps

        # Update current length
        self.tips.length += step

        # Create child branches
        mask = self.tips.length >= self.tips.max_length
        for index in np.arange(len(self.tips))[mask]:
            for _ in range(self.child_count):
                self.add_child(index)

        # Create node children
        for index in to_create:
            self.add_child(index)

        # Kill edges whose length is reached
        mask = self.tips.length >= self.tips.max_length
        self.tips.delete(mask)



    # ====================================================================================================
    # View (mesh to curve)
    # ====================================================================================================

    def view(self):
        curve = Curve()

        for eid in range(self.branch_id):
            mask = self.mesh.points.branch_id == eid
            curve.add_poly(
                points = self.mesh.points.position[mask],
                radius = self.mesh.points.radius[mask],
            )

        curve.to_object("Tree")

        self.mesh.to_object("Tree (M)")

        








