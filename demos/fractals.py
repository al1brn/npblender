import npblender as gp
import numpy as np

from npblender.core.blender import bfloat, bint, bbool

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


class Sierpinski(gp.Animation):
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
            
            face_order  = face.Order
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

        triangle.points.add_points(new_verts[:v_count])
        
        triangle.add_faces(new_corners[:c_count], Depth=depth + 1, Thread=new_thread[:c_count])

    def compute(self):

        # ----------------------------------------------------------------------------------------------------
        # Create the initial triangle

        triangle = gp.Mesh.Circle(segments=3, materials=["Fractal"])
        
        # The couple (Depth, Order) is unique and always the same    
        triangle.faces.new_int("Depth",    default=0)  # Fractal depth
        triangle.faces.new_int("Order",    default=0)  # Generation order within the depth
        triangle.faces.new_float("Thread", default=1.) # Thread value

        # ----------------------------------------------------------------------------------------------------
        # Fractal loop
    
        cam = gp.Camera()
        for i in range(self.max_depth):
            vis, sizes = cam.visible_faces(triangle)
            vis &= sizes > self.max_size
            if not np.any(vis):
                break
            self.sierpinski_step(triangle, vis, depth=i)

        self.triangle = triangle

    def view(self):            
        self.triangle.to_object("Sierpinski")

    def get_animation(self):
        return self.triangle.as_dict()
    
    def set_animation(self, data):
        self.triangle = gp.Mesh.FromDict(data)


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
        - a list of count x n corners is created with corners from master and corners pointing
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
            insts = gp.Instances(
                points = O + self.centers @ M,
                models = gp.Mesh.UVSphere(radius=.05))
            insts.realize()['mesh'].to_object("NEW CENTERS")

            insts = gp.Instances(
                points = O + self.centers @ M,
                models = gp.Mesh.UVSphere(radius=1),
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
            mesh = gp.Mesh(verts=verts)
        else:
            master_map = [list(umaster).index(c) for c in self.master]
            mesh = gp.Mesh(verts=verts, corners=master_map, sizes=self.faces)

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
            return gp.Mesh(verts=verts)
        else:
            return gp.Mesh(verts=verts, corners=new_division, sizes=np.tile(self.faces, self.count))
        
    def test_split(self):

        vertices = gp.DynamicRecArray()
        vertices.new_field("position", float, 3)

        umaster = self.master[self.master_i_unique]
        vertices.append_attributes(position=self.vertices[umaster])

        master = np.array([list(umaster).index(c) for c in self.master])

        corners = self.split(master, vertices)

        if self.faces is None:
            return gp.Mesh(verts=vertices.position)
        else:
            return gp.Mesh(verts=vertices.position, corners=corners, sizes=np.tile(self.faces, self.count))
        
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
        sizes = np.ones((n, 1))

        if self.centers is not None and self.sizes is not None:
            for i in range(n):
                master = masters[i]
                master_verts = vertices[master[self.master_i_unique]]
                O, M = self.get_local_base(master_verts)

                centers[i] = O + self.center @ M
                sizes[i] = self.size_ratio * np.linalg.norm(M[0])

        if False:
            insts = gp.Instances(
                points = centers,
                models = gp.Mesh.UVSphere(radius=.05))
            insts.realize()['mesh'].to_object("SETUP CENTERS")

            insts = gp.Instances(
                points = centers,
                models = gp.Mesh.UVSphere(radius=1),
                Scale  = sizes)
            insts.realize()['mesh'].to_object("SETUP SIZES")
            

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

        if False:
            mesh = gp.Mesh(verts=verts)
            mesh.join(mesh.topology())
            mesh.to_object("SQUARE POINTS")

        splitter = cls(
            master   = [0, 3, 15, 12],
            division = [0, 1, 5, 4, 2, 3, 7, 6, 5, 6, 10, 9, 8, 9, 13, 12, 10, 11, 15, 14],
            vertices = verts,
            faces    = [4],
        )

        if False:
            mesh = gp.Mesh(verts=verts, corners=splitter.corners, sizes=[4, 4, 4, 4, 4])
            mesh.to_object("SQUARE SPLITTER")

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

        if False:
            mesh = gp.Mesh(verts=verts)
            mesh.join(mesh.topology())
            mesh.to_object("CUBE POINTS")

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

        if False:
            mesh = gp.Mesh(verts=verts, corners=splitter.corners, sizes=[4]*13)
            mesh.to_object("CUBE SPLITTER", shade_smooth=False)

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

        if False:
            mesh = gp.Mesh(verts=verts)
            mesh.join(mesh.topology())
            mesh.to_object("TETRAHEDRON POINTS")

        splitter = cls(
            master   = pyr([0, 3, 2, 1]),
            division = pyr([0, 6, 5, 4]) + pyr([5, 8, 2, 7]) + pyr([4, 9, 7, 1]) + pyr([6, 3, 8, 9]),
            base     = [0, 1, 2, 3],
            vertices = verts,
            faces    = [3, 3, 3, 3],
        )

        if False:
            mesh = gp.Mesh(verts=verts, corners=splitter.corners, sizes=splitter.faces*4)
            mesh.to_object("TETRAHEDRON SPLITTER", shade_smooth=False)

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

        if False:
            mesh = gp.Mesh(verts=verts)
            mesh.join(mesh.topology())
            mesh.to_object("PYRAMID POINTS")


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

        self.vertices = gp.DynamicRecArray()
        self.vertices.new_field("position", float, 3)

        # ----- Indices

        self.items = gp.DynamicRecArray()

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
        self.vertices.append_attributes(position=mesh_setup['vertices'])

        self.items.clear()
        self.items.append_attributes(
            corners = mesh_setup['masters'],
            center  = mesh_setup['centers'],
            size    = mesh_setup['sizes'],
            depth   = 0,
            uid     = np.arange(len(mesh_setup['masters'])),
            thread  = 0., 
        )
        self.final = gp.DynamicRecArray.FromOther(self.items, copy=False)
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

            if False and len(self.items) and idepth==depth-1:
                insts = gp.Instances(
                    points = self.items.center,
                    models = gp.Mesh.UVSphere(radius=.1))
                insts.realize()['mesh'].to_object("CENTERS")

                insts = gp.Instances(
                    points = self.items.center,
                    models = gp.Mesh.UVSphere(),
                    Scale  = self.items.size[:, None])
                insts.realize()['mesh'].to_object("SIZES")

            # ----------------------------------------------------------------------------------------------------
            # Split the visible items

            d = self.splitter.splits(
                masters  = self.items.corners,
                vertices = self.vertices.position,
                sizes    = self.items.size)
            
            # ----------------------------------------------------------------------------------------------------
            # Update the array

            if len(d['vertices']):
                self.vertices.append_attributes(position=d['vertices'])

            n = len(d['masters'])
            if n:
                new_items = self.items.clone(empty=True)

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


                new_items.append_attributes(
                    corners = d['masters'],
                    center  = d['centers'],
                    size    = d['sizes'],
                    
                    depth   = idepth,
                    uid     = uid,
                    random  = rands,
                    thread  = threads.flatten(),
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

        items = self.final.clone()
        items.append_other(self.items)

        nfaces = len(self.splitter.faces)

        corners, faces = self.splitter.get_corners_and_faces(items.corners)

        mesh = gp.Mesh(verts=self.vertices.position, corners=corners, sizes=faces, materials=materials)

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
# Fractal class

class Fractal_OLD:
    
    def __init__(self, fractal_type='Mesh'): #, points_count, divisions, fractal_type='Mesh', **kwargs):
        """ Fractal
        
        A fractal consists in replacing an item by divisions similar items.
        
        Dimensions   Master          Divided        
        - vertices : any             more (hopefully)
        - corners  : points_count    divisions * points_count
        - faces    : faces_count     divisions * faces_count
        
        For mesh, a subidivision is defined by the mesh 'model'.
        The master mesh is defined by three arrays:
        - master_points  : index of the vertices belonging to the master
        - master_corners : master corners
        - master_faces   : master face sizes
        

        Examples:
        - a quad defined by 4 indices
        - a cube defined by 8 indices
        - a spline defined by 3 vertices
        
        Division algorithm:
        - loop on the items to divide
        - create new vertices
        - replace the points_count indices by divisions*points_count indices
        
        Conversion to mesh consists in:
        - creating the vertices in the final mesh
        - using the indices as corners
        - tiling the model faces on the items_count items

        Conversion to curve consists in:
        - creating one spline per item
        """
        
        # ----------------------------------------------------------------------------------------------------
        # Initialize the final geometry
        
        self.use_mesh = fractal_type.lower() == 'mesh'

        self.vertices = gp.DynamicRecArray()
        self.vertices.new_field("position", float, 3)
        
        return
        
        # ----------------------------------------------------------------------------------------------------
        # The array of items
        
        self.items = DynamicRecArray()
        self.items.new_field("indices", int, (0, points_count))
        
        self.final_items = DynamicRecArray.FromOther(self.items, copy=False)
        
        # ----------------------------------------------------------------------------------------------------
        # Division parameters
        
        self.points_count = points_count
        self.divisions    = divisions
        self.faces_cuont  = None
        
        self.start        = None
        self.model        = None
        self.origin, self.x_index, self.y_index, self.z_index = 0, 1, 2, None
        self.local_coords = None
        
        self.fractal_mesh = None
        
        # ----------------------------------------------------------------------------------------------------
        # Custom arguments
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    # =============================================================================================================================
    # Build the local base from an item
    
    def build_base(self, indices, vertices=None):

        if vertices is None:
            vertices = self.vertices
            
        # Origin
        O = vertices.position[indices[self.origin]]
            
        # Vertices indexed by indices 
        vecs = vertices.position[indices] - O
        
        # Base
        vx, vy = vecs[self.x_index], vecs[self.y_index]
    
        return O, np.column_stack((vx, vy, np.cross(vx, vy) if self.z_index is None else vecs[self.z_index]))
    
    # =============================================================================================================================
    # Get master mesh from model
    
    def master_mesh(self):
        mesh = gp.Mesh.Points(position=self.model.points.position[self.master_points])
        mesh.corners.add_corners(self.master_corners)
        mesh.faces.add_faces(self.master_faces)
        return mesh
    
    def to_mesh(self):
        mesh = gp.Mesh.Points(position=self.vertices.position)
        mesh.corners.add_corners(self.items.indices.flatten())
        mesh.faces.add_faces(np.tile(self.faces, len(self.items.indices)).flatten())
        return mesh
        
    
    # =============================================================================================================================
    # Initialize the items array
    
    def init_items(self):
        
        self.items = gp.DynamicRecArray()
        self.items.new_field("indices", int, self.m_corners_count)
        
        self.final_items = gp.DynamicRecArray.FromOther(self.items)
    
    # =============================================================================================================================
    # Demo (Sierpinski triangle)
    
    @classmethod
    def Demo(cls):
        
        # ----------------------------------------------------------------------------------------------------
        # model : 3 triangles ih one master triangle
        
        s3 = np.sqrt(3.)
        A = np.array((0, s3 / 3, 0))
        B = np.array((-0.5, -s3 / 6, 0))
        C = np.array(( 0.5, -s3 / 6, 0))
        verts = np.array((A, (A+B)/2, B, (B+C)/2, C, (C+A)/2))
        
        model = gp.Mesh.Points(position=verts)
        model.corners.add_corners([1, 5, 0, 2, 3, 1, 3, 4, 5])
        model.faces.add_faces([3, 3, 3])
        
        return cls.FromModel(model, master_points=[0, 2, 4], master_corners=[0, 1, 2], master_faces=[3])
        

    # =============================================================================================================================
    # Initialize from model
        
    @classmethod
    def FromModel(cls, model, master_points, master_corners, master_faces, base_indices=[0, 1, 2, None]):
        
        fractal = cls()
        
        # Model
        
        fractal.model = model
        fractal.master_points  = master_points
        fractal.master_corners = master_corners
        fractal.master_faces   = master_faces
        
        fractal.m_points_count  = len(fractal.master_points)
        fractal.m_corners_count = len(fractal.master_corners)
        fractal.s_points_count  = len(fractal.model.points)
        fractal.s_corners_count = len(fractal.model.corners)
        fractal.divisions = fractal.s_corners_count // fractal.m_corners_count
        
        
        # Division parameters
        
        fractal.origin  = base_indices[0]
        fractal.x_index = base_indices[1]
        fractal.y_index = base_indices[2]
        fractal.z_index = base_indices[3]
        
        # Compute the coordinates of the fractal vertices in the base
        
        O, base = fractal.build_base(fractal.master_points, vertices=model.points)
        
        fractal.local_coords = np.linalg.solve(base, (model.points.position - O).T).T
        
        # Initialize the indices
        
        fractal.init_items()
        
        # Default start mesh
        
        fractal.start_mesh = fractal.master_mesh()
        
        # Done
        
        return fractal
    
    # =============================================================================================================================
    # Debug : view subdivisions
    
    def view_division(self, name="FRACTAL Division"):
        if self.model is None:
            return
        
        topo = self.model.topology(vertices=True, edges=False, corners=False, faces=True)
        
        mesh = self.master_mesh()
        
        #mesh = gp.Mesh.Points(position=self.model.points.position[self.master_points])
        #mesh.corners.add_corners(self.master_corners)
        #mesh.faces.add_faces(self.master_faces)
        
        mesh.points.position -= (0, 0, 1)
        mesh.join(self.model)
        mesh.join(topo)
        
        mesh.to_object(name, shade_smooth=False)
        
    # =============================================================================================================================
    # =============================================================================================================================
    # Fractal algorithm
    
    def div_reset(self, start_mesh=None):
        
        if start_mesh is not None:
            self.start_mesh = start_mesh
            
        # Vertices
        
        self.vertices.clear()
        self.vertices.append_attributes(position=self.start_mesh.points.position)
        
        # Items
        
        self.items.clear()
        n = len(self.start_mesh.corners)//len(self.master_corners)
        if n*len(self.master_corners) != len(self.start_mesh.corners):
            raise Exception(f"Fractal.div_reset error: the number of corners of 'start_mesh' ({len(self.start_mesh.corners)}) must be a multiple of {len(self.master_corners)=}") 
            
        self.items.append_attributes(indices=self.start_mesh.corners.vertex_index.reshape(n, len(self.master_corners)))
        
    
    def fractal_step(self):
        
        new_items = gp.DynamicRecArray.FromOther(self.items, copy=False)
        
        for i, master_indices in enumerate(self.items.indices):

            # ----------------------------------------------------------------------------------------------------
            # Build the base
            
            O, base = self.build_base(master_indices)
            
            # ----------------------------------------------------------------------------------------------------
            # Compute all the vertices

            vertices = O + self.local_coords @ base
            n = len(vertices)

            # Remove common ones
            
            sel = np.ones(n, bool)
            sel[self.master_points] = False
            offset = len(self.vertices)
            self.vertices.append_attributes(position=vertices[sel])
            
            indices = np.full(self.s_corners_count, -1, int)
            indices[self.master_points] = master_indices
            index = offset
            for i in range(len(indices)):
                if indices[i] < 0:
                    indices[i] = index
                    index += 1
            
            new_items.append_attributes(indices=indices.reshape(self.divisions, self.m_corners_count))
            
        del self.items
        self.items = new_items

  
























class Fractal_OLD(gp.Animation):
    """ A fractal mesh holds a list of vertices belonging to the fractal item
    """

    def __init__(self, model, name="Fractal", thread_factor=.7, seed=0):
        """ The fractal structure stores one record per item

        The basic attributes are:
        - indices
        - position
        - radius
        - depth
        - uid
        - thread

        Arguments
        - model (Mesh model or int): the model or the number of vertices per item
        - thread_factor : a factor to increase the thread value of the fractal
        - seed : a seed value for the fractal

        """

        # ----- Model or int

        if isinstance(model, int):
            length = model
            model  = None
        elif isinstance(model, gp.Mesh):
            length = len(model.points)
        else:
            obj = gp.blender.get_object(model, halt=False)
            if obj is None:
                raise AttributeError(f"Model must be an int, an object or a Mesh")
            model = gp.Mesh.FromObject(obj)
            length = len(model.points)

        # ----- Live items

        self.fractal = gp.DynamicRecArray()

        self.fractal.new_field("indices",  int, length)
        self.fractal.new_field("position", float, 3)
        self.fractal.new_field("radius",   float)

        self.fractal.new_field("depth",    int)
        self.fractal.new_field("order",    int)
        self.fractal.new_field("uid",      int)

        self.fractal.new_field("thread",   float)
        self.fractal.new_field("seed",     int)
        self.fractal.new_field("random",   float)

        # ----- Final items

        self.final = gp.DynamicRecArray.FromOther(self.fractal, copy=False)

        # ----- Vertices

        self.vertices = gp.DynamicRecArray()
        self.vertices.new_field("position", float, 3)

        # ----- Global param

        self.keep_divided  = False
        self.fractal_id    = 0
        self.thread_factor = thread_factor
        self.fractal_seed  = seed
        self.model         = model

        # ----- Animation param

        self.name       = name
        self.camera     = None
        self.max_depth  = 3
        self.min_radius = .01
        self.omit       = 0.

        # ----- Reset

        self.reset()

    # ----------------------------------------------------------------------------------------------------
    # From template

    @classmethod
    def FromTemplate():
        pass



    # ----------------------------------------------------------------------------------------------------
    # str

    def __str__(self):
        return f"<FractalMesh: to split {len(self.fractal)}, final: {len(self.final)}>"
    
    # ----------------------------------------------------------------------------------------------------
    # To Mesh

    def to_mesh(self):

        # ----- Merge the two arrays

        final = gp.DynamicRecArray.FromOther(self.final, copy=True)
        final.append_other(self.fractal)

        # ----- Initialize the mesh with the vertices

        mesh = gp.Mesh.Points(self.vertices.position, materials=self.model.materials)

        mesh.faces.new_int("Depth")
        mesh.faces.new_int("Order")
        mesh.faces.new_int("Uid")

        mesh.faces.new_float("Thread")
        mesh.faces.new_int("Seed")
        mesh.faces.new_float("Random")

        mesh.faces.new_bool("Visible")

        # ----- Corners and faces

        nitems   = len(final)
        nfaces   = len(self.model.faces)
        ncorners = len(self.model.corners)

        if nfaces > 0:

            # ----- Algorithm:
            # Model has nv vertices and nc corners
            # The model 'corners' points to vertices in [0, nv[
            # Each sub item refers to nv vertices created anywhere in the array
            # and which are stored in 'indices'
            # The corners are indices[model.corners]

            indices = final.indices  # shape (nitems, nvertices)
            corners = self.model.corners.vertex_index # shape (ncorners,)

            # ----- UV Map

            attrs = {}
            if "UVMap" in self.model.corners.attributes.names:
                attrs["UVMap"] = np.resize(self.model.corners.UVMap, (nitems*ncorners, 2))

            # ----- Create the corners

            mesh.corners.add_corners(
                corners = np.reshape(indices[np.arange(nitems)[:, None], corners], nitems*ncorners),
                **attrs,
                )

            if self.camera is None:
                vis_debug = True
            else:
                vis_debug, _ = self.camera.visible_points(final.position, radius=final.radius)
                vis_debug = np.repeat(vis_debug[:, 0], nfaces)
            
            # ----- Faces

            attrs = {
                "material_index" : np.resize(self.model.faces.material_index, (nitems*nfaces,)),
                "Depth"          : np.repeat(final.depth, nfaces),
                "Order"          : np.repeat(final.order, nfaces),
                "Uid"            : np.repeat(final.uid, nfaces),
                "Thread"         : np.repeat(final.thread, nfaces) % 1,
                "Seed"           : np.repeat(final.seed, nfaces),
                "Random"         : np.repeat(final.random, nfaces),
                "Visible"        : vis_debug,
            }

            mesh.faces.add_faces(
                faces = np.tile(self.model.faces.loop_total, nitems),
                **attrs,
                )
            
        return mesh

    # ----------------------------------------------------------------------------------------------------
    # Fractal step

    def division_step(self):

        from time import time

        if not len(self.fractal) :
            return
        
        depth = max(self.fractal.depth) + 1

        if self.keep_divided:
            self.final.append_other(self.fractal)

        new_items = gp.DynamicRecArray.FromOther(self.fractal)
        fac = self.thread_factor**depth

        # ----- Camera

        if self.camera is None:
            ok_split = [True] * len(self.fractal)
        else:
            vis, size = self.camera.visible_points(self.fractal.position, radius=self.fractal.radius)
            ok_split = vis[:, 0]
            ok_split &= size[:, self.camera.SIZE] > self.min_radius

        for index, ok in enumerate(ok_split):

            if not ok:
                new_items.append_other(self.fractal[index])

                continue

            attrs = self.split(index)
            n = len(attrs["indices"])
            thread = self.fractal.thread[index]

            order = self.fractal.order[index]*n
            rng = np.random.default_rng(abs(hash((self.fractal_seed, depth, order))))

            to_add = gp.DynamicRecArray.FromOther(new_items, copy=False)

            keep = rng.random(n) >= self.omit
            if False:
                keep = [True]*n

            nkeep = np.sum(keep)
            if nkeep == 0:
                keep[rng.integers(n)] = True
                nkeep = 1

            if nkeep < n:
                keep = np.zeros(n, dtype=bool)
                keep[:nkeep] = True
                keep[nkeep:] = False

            to_add.append_attributes(
                depth  = depth,
                order  = order + np.arange(n),
                uid    = self.fractal_id + np.arange(n),
                thread = thread + rng.uniform(-fac, fac, n),
                seed   = rng.integers(0, 1 << 31, n),
                random = rng.uniform(0, 1, n),
                **attrs)
            
            add_attrs = {key: to_add[key][keep] for key in to_add.names}
            
            new_items.append_attributes(**add_attrs)

            self.fractal_id += nkeep

        del self.fractal
        self.fractal = new_items

    # ====================================================================================================
    # Animation interface

    def reset(self):

        self.fractal.clear()
        self.final.clear()
        self.vertices.clear()

        self.fractal_id = 0

        if self.model is not None:
            self.vertices.append_attributes(
                position = self.model.points.position,
            )

            vmin, vmax = np.min(self.model.points.position, axis=0), np.max(self.model.points.position, axis=0)
            radius = max(np.linalg.norm(vmin), np.linalg.norm(vmin))

            rng = np.random.default_rng(abs(hash((self.fractal_seed, self.fractal_id))))

            self.fractal.append_attributes(
                indices     = np.arange(np.shape(self.fractal.indices)[1]),
                position    = (0, 0, 0),
                radius      = radius,
                depth       = 0,
                uid         = self.fractal_id,
                thread      = rng.uniform(0, 1),
                seed        = rng.integers(0, 1<<63),
                random      = rng.uniform(0, 1),
            )
            self.fractal_id += 1

    def compute(self):
        self.reset()
        for i in range(self.max_depth):
            self.division_step()
        self.view_mesh = self.to_mesh()

    def view(self):
        self.view_mesh.to_object(self.name)

    def get_animation(self):
        return self.view_mesh.as_dict()

    def set_animation(self, data):
        self.view_mesh = gp.Mesh.FromDict(data)




# ====================================================================================================
# ====================================================================================================
# Inner Cube
# ====================================================================================================
# ====================================================================================================

    
class InnerCubeFractal(Fractal):
    """ Cube is repeated 9 times : 8 corners plus centers
    
    Each new cube is one third smaller than the original

    Cube indices are stored in a (2, 2, 2) array. This array is obtained by reshaping
    a range(1, 8). Hence the vertices are :

        3 -------- 7
      /          /
     /          /
    1 -------- 5

       2 -------- 6
      /          /
     /          /
    0 -------- 4
    

    Splitting consists in creating 56 new vertices to get 64=56+8 vertices indexed by a (4, 4, 4) array
    
    """

    def __init__(self, model=None, thread_factor=0.7, seed=0):
        if model is None:
            model = gp.Mesh.Cube()
        super().__init__(model, thread_factor=thread_factor, seed=seed)
        self.fractal_id = 0
    
    # -------------------------------------------------------------------------------
    # Split the cube into 9 sub cubes

    def split(self, item_index, thread_factor=.7):

        from itertools import product    

        depth = self.fractal.depth[item_index]

        base_inds  = np.reshape(self.fractal.indices[item_index], (2, 2, 2)) # (2, 2, 2)
        base_verts = self.vertices.position[base_inds] # (2, 2, 2, 3)

        # ----- All vertices

        offset = len(self.vertices)
        vertices = np.zeros((56, 3), float)
        indices  = np.zeros((4, 4, 4), int)

        t = np.linspace(0, 1, 4)
        grid = np.zeros((4, 4, 4, 3))

        count = 0
        for i, xi in enumerate(t):
            ibound = i in (0, 3)

            for j, yj in enumerate(t):
                jbound = j in (0, 3)

                for k, zk in enumerate(t):
                    kbound = k in (0, 3)

                    point = np.zeros(3)
                    for dx, dy, dz in product([0, 1], repeat=3):
                        w = ((1 - xi) if dx == 0 else xi) * \
                            ((1 - yj) if dy == 0 else yj) * \
                            ((1 - zk) if dz == 0 else zk)
                        point += w * base_verts[dx, dy, dz]
                    
                    if ibound and jbound and kbound:
                        indices[i, j, k] = base_inds[i//3, j//3, k//3]

                    else:
                        vertices[count] = point
                        indices[i, j, k] = offset + count
                        count += 1  

        assert(count == 56)

        # ----- Update the vertices

        self.vertices.append_attributes(position = vertices)

        # ----- The 9 new cubes

        # Center cube replaces the initial one

        cubes = [
            [indices[1, 1, 1], indices[1, 1, 2], indices[1, 2, 1], indices[1, 2, 2],
             indices[2, 1, 1], indices[2, 1, 2], indices[2, 2, 1], indices[2, 2, 2]]
        ]

        # 8 other cubes

        for i in [0, 2]:
            for j in [0, 2]:
                for k in [0, 2]:
                    cubes.append([
                        indices[i,   j, k], indices[i,   j, k+1], indices[i,   j+1, k], indices[i,   j+1, k+1], 
                        indices[i+1, j, k], indices[i+1, j, k+1], indices[i+1, j+1, k], indices[i+1, j+1, k+1]])
                    

        # ----- Create the sub items

        pos = self.fractal.position[item_index]
        radius = self.fractal.radius[item_index]
        r3 = radius/3
        position = [pos, 
                    pos + (-r3, -r3, -r3), pos + (-r3, -r3,  r3), pos + (-r3,  r3, -r3), pos + (-r3,  r3,  r3), 
                    pos + ( r3, -r3, -r3), pos + ( r3, -r3,  r3), pos + ( r3,  r3, -r3), pos + ( r3,  r3,  r3), 
                    ] 
        return {
            "indices"  : cubes,
            "position" : position, 
            "radius"   : r3,
            }
    
# ====================================================================================================
# ====================================================================================================
# Outer Cube
# ====================================================================================================
# ====================================================================================================

class OuterCubeFractal(Fractal):
    """ 6 cubes are added on the faces
    
    Each new cube is one third smaller than the original

    Splitting consists in creating 6*8=48 new vertices
    
    """

    def __init__(self, model=None, thread_factor=0.7, seed=0):
        if model is None:
            model = gp.Mesh.Cube()
        super().__init__(model, thread_factor=thread_factor, seed=seed)

        # Faces corners
        #self.faces = np.reshape(self.model.corners, (6, 4))

        self.fractal_id = 0
        self.keep_divided = True


    
    # -------------------------------------------------------------------------------
    # Add 6 new cubes

    def split(self, item_index, thread_factor=.7):

        from itertools import product

        base_inds  = self.fractal.indices[item_index] # 8
        base_verts = self.vertices.position[base_inds] # (8, 3)

        # ----- All vertices

        offset = len(self.vertices)
        vertices = np.empty((6, 8, 3), float)
        #indices  = np.empty((6, 8), int)

        # ----- Central cube

        center = base_verts.mean(axis=0)
        cube = (base_verts - center)/3
        c7 = base_verts[7] - center
        dx, dy, dz = c7[0], c7[1], c7[2]

        # ----- Creation of 6 smaller cubes

        pos = np.empty((6, 3), float)
        fac = 4/3
        pos[0] = center - (dx*fac, 0, 0)
        pos[1] = center + (dx*fac, 0, 0)
        pos[2] = center - (0, dy*fac, 0)
        pos[3] = center + (0, dy*fac, 0)
        pos[4] = center - (0, 0, dz*fac)
        pos[5] = center + (0, 0, dz*fac)

        vertices[0] = cube + pos[0]
        vertices[1] = cube + pos[1]
        vertices[2] = cube + pos[2]
        vertices[3] = cube + pos[3]
        vertices[4] = cube + pos[4]
        vertices[5] = cube + pos[5]

        # ----- Update the vertices

        self.vertices.append_attributes(position=vertices.reshape(48, 3))

        # ----- Create the sub items

        radius = self.fractal.radius[item_index]

        return {
            "indices"  : (offset + np.arange(48)).reshape(6, 8),
            "position" : pos, 
            "radius"   : radius/3,
            }







# ====================================================================================================
# ====================================================================================================
# Splitter fractal
# ====================================================================================================
# ====================================================================================================

# ====================================================================================================
# Splitter

class Splitter_OLD:

    def __init__(self):
        # Item position
        self.position = np.zeros((1, 3), float)
        # Item radius
        self.radius   = np.ones(1, float)

        # Fractal depth
        self.depth    = np.zeros(1, int)
        # Fractal order within depth
        self.order    = np.zeros(1, int)
        # Fractal thread
        self.thread   = np.ones(1, float)

    def __str__(self):
        depths = list(np.bincount(self.depth))
        return f"<{type(self).__name__}: {depths=}, indices: {np.shape(self.indices)}, vertices: {np.shape(self.vertices)}>"

    def __repr__(self):
        return f"{type(self)}\nVertices:\n" + str(self.vertices) + "\nIndices:\n" +str(self.indices)
    
    def __len__(self):
        return len(self.position)

    def to_mesh(self, materials="Fractal"):
        return None
    
    # ----------------------------------------------------------------------------------------------------
    # Compute fractal attributes after split is done
    
    def after_split(self, item_index, new_items, rng, thread_factor):
        """ Compute fractal attributes after split is done.

        Item whose index is passed has been split into
          sub items. The indices
        of the sub items are passed in 'new_items' argument.

        Args:
            volume_index (int): index of the volume to split
            new_items (list of int): list of indices of new volumes after split

        Returns:
            list of ints : concatenation of volume_index and new_items
        """

        n = len(new_items)

        depth = self.depth[item_index]
        order = self.order[item_index]

        self.depth    = np.append(self.depth,    np.zeros(n, int), axis=0)
        self.order    = np.append(self.order,    np.zeros(n, int), axis=0)
        self.thread   = np.append(self.thread,   np.ones(n, float), axis=0)
        self.position = np.append(self.position, np.zeros((n, 3), float), axis=0)
        self.radius   = np.append(self.radius,   np.ones(n, float), axis=0)

        base_order = order*(1 + n)**(depth+1)

        thread = self.thread[item_index]
        fac = thread_factor**(depth+1)

        all_items = [item_index] + new_items
        for i, ivol in enumerate(all_items):

            self.depth[ivol]  = depth + 1
            self.order[ivol]  = base_order + i
            self.thread[ivol] = 1

            vs = self.vertices[self.indices[ivol]]
            vs = np.reshape(vs, (np.size(vs)//3, 3))

            center = np.average(vs, axis=0)
            self.position[ivol] = center

            dists = np.linalg.norm(vs - center, axis=1)            
            self.radius[ivol] = np.max(dists)

            #vmin, vmax = np.min(vs, axis=0), np.max(vs, axis=0)
            #self.radius[ivol] = np.linalg.norm(vmax - vmin)/2

            # ----- Thread

            self.thread[ivol] = thread + rng.uniform(-1, 1)*fac

        return all_items

    def split(self, item_index, rng, thread_factor):
        pass


# ====================================================================================================
# Splitter Fractal : replace an item by sub items using a Splitter

class SplitterFractal(gp.Animation):

    def __init__(self, splitter_class, name="Splitter Fractal", max_depth=5, max_size=.1, thread_factor=.7, seed=0):
        
        self.splitter_class = splitter_class
        self.splitter = None
        
        self.name = name
        self.max_depth = max_depth
        self.max_size = max_size
        self.thread_factor = thread_factor
        self.mesh = None
        self.thread_seed = 0

        self.camera = gp.Camera()

    def view(self):
        if self.mesh is None:
            return
        self.mesh.to_object(self.name, shade_smooth=False)

    def get_animation(self):
        return self.splitter.to_mesh()
    
    def set_animation(self, data):
        self.mesh = gp.Mesh.FromDict(data)

    def fractal_step(self, depth=0):

        item_vis, item_size = self.camera.visible_points(self.splitter.position, radius=self.splitter.radius)
        count = len(self.splitter)
        for index, (vis, size, dp) in enumerate(zip(item_vis[:, self.camera.VISIBLE], item_size[:, self.camera.SIZE], self.splitter.depth)):

            if not (vis and (size > self.max_size) and (dp == depth)):
                continue

            # Split the item
            h = abs(hash((depth, self.splitter.order[index], self.thread_seed)))

            rng = np.random.default_rng(abs(hash((depth, self.splitter.order[index], self.thread_seed))))
            self.splitter.split(index, rng, self.thread_factor)

    def compute(self):

        # Initialize the splitter
        self.splitter = self.splitter_class()

        # Fractal loop
        for i in range(self.max_depth):
            self.fractal_step(depth=i)

        self.mesh = self.splitter.to_mesh()

# ====================================================================================================
# Cube fractal : replace a cube by 9 smaller cubes

class CubeSplitter_OLD(Splitter):
    """ Vertices are managed in a dedicated array

    Cube indices are stored in a (2, 2, 2) array. This array is obtained by reshaping
    a range(1, 8). Hence the vertices are :

        3 -------- 7
      /          /
     /          /
    1 -------- 5

       2 -------- 6
      /          /
     /          /
    0 -------- 4
    

    Splitting consists in creating 56 new vertices to get 64=56+8 vertices indexed by a (4, 4, 4) array
    
    """

    def __init__(self):
        super().__init__()

        indices  = np.reshape(np.arange(8), (1, 2, 2, 2))
        vertices = np.empty((2, 2, 2, 3), float)
        for i0 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    vertices[i0, i1, i2] = (-1 + 2*i0, -1 + 2*i1, -1 + 2*i2)

        self.vertices = np.reshape(vertices, (8, 3))
        self.indices  = np.reshape(indices, (1, 2, 2, 2))
    
    # -------------------------------------------------------------------------------
    # Convert to a mesh
    
    def to_mesh(self, materials="Fractal"):
        # Vertices
        mesh = gp.Mesh.Points(self.vertices, materials=materials)
        #return mesh

        # Corners
        corners = np.zeros((len(self.indices)*6, 4), int)
        for i in range(len(self.indices)):
            i6 = i*6
            corners[i6+0] = self.indices[i, 0, 1, 0], self.indices[i, 1, 1, 0], self.indices[i, 1, 0, 0], self.indices[i, 0, 0, 0]
            corners[i6+1] = self.indices[i, 0, 0, 1], self.indices[i, 1, 0, 1], self.indices[i, 1, 1, 1], self.indices[i, 0, 1, 1]
            corners[i6+2] = self.indices[i, 0, 0, 0], self.indices[i, 1, 0, 0], self.indices[i, 1, 0, 1], self.indices[i, 0, 0, 1]
            corners[i6+3] = self.indices[i, 0, 1, 0], self.indices[i, 0, 1, 1], self.indices[i, 1, 1, 1], self.indices[i, 1, 1, 0]
            corners[i6+4] = self.indices[i, 0, 0, 0], self.indices[i, 0, 0, 1], self.indices[i, 0, 1, 1], self.indices[i, 0, 1, 0]
            corners[i6+5] = self.indices[i, 1, 0, 0], self.indices[i, 1, 1, 0], self.indices[i, 1, 1, 1], self.indices[i, 1, 0, 1]
        
        mesh.faces.new_int("Depth", 0)
        mesh.faces.new_int("Order", 0)
        mesh.faces.new_float("Thread", 0)

        mesh.add_faces(corners,
                Depth  = np.tile(self.depth [:, np.newaxis], (1, 6)).reshape(-1),
                Order  = np.tile(self.order [:, np.newaxis], (1, 6)).reshape(-1),
                Thread = np.tile(self.thread[:, np.newaxis], (1, 6)).reshape(-1),
        )
        return mesh
    
    # -------------------------------------------------------------------------------
    # Split the cube into 9 sub cubes

    def split(self, item_index, rng, thread_factor):

        from itertools import product    

        depth = self.depth[item_index]    

        base_inds  = self.indices[item_index] # (2, 2, 2)
        base_verts = self.vertices[base_inds] # (2, 2, 2, 3)

        # ----- All vertices

        offset = len(self.vertices)
        vertices = np.zeros((56, 3), float)
        indices  = np.zeros((4, 4, 4), int)

        t = np.linspace(0, 1, 4)
        grid = np.zeros((4, 4, 4, 3))

        count = 0
        for i, xi in enumerate(t):
            ibound = i in (0, 3)

            for j, yj in enumerate(t):
                jbound = j in (0, 3)

                for k, zk in enumerate(t):
                    kbound = k in (0, 3)

                    point = np.zeros(3)
                    for dx, dy, dz in product([0, 1], repeat=3):
                        w = ((1 - xi) if dx == 0 else xi) * \
                            ((1 - yj) if dy == 0 else yj) * \
                            ((1 - zk) if dz == 0 else zk)
                        point += w * base_verts[dx, dy, dz]
                    
                    if ibound and jbound and kbound:
                        indices[i, j, k] = base_inds[i//3, j//3, k//3]

                    else:
                        vertices[count] = point
                        indices[i, j, k] = offset + count
                        count += 1  

        assert(count == 56)

        # ----- Update the vertices

        self.vertices = np.append(self.vertices, vertices, axis=0)

        # ----- The 9 new cubes

        # Replace the splitten cube
        self.indices[item_index] = np.reshape([
            [indices[1, 1, 1], indices[1, 1, 2], indices[1, 2, 1], indices[1, 2, 2],
             indices[2, 1, 1], indices[2, 1, 2], indices[2, 2, 1], indices[2, 2, 2]]
        ], (2, 2, 2))
        # 8 new cubes

        cubes = []
        for i in [0, 2]:
            for j in [0, 2]:
                for k in [0, 2]:
                    cubes.append([
                        indices[i,   j, k], indices[i,   j, k+1], indices[i,   j+1, k], indices[i,   j+1, k+1], 
                        indices[i+1, j, k], indices[i+1, j, k+1], indices[i+1, j+1, k], indices[i+1, j+1, k+1]])

        count = len(self.indices)
        self.indices = np.append(self.indices, np.reshape(cubes, (8, 2, 2, 2)), axis=0)

        # ----- After split

        return self.after_split(item_index, [count + i for i in range(8)], rng, thread_factor)


class CubeFractal(SplitterFractal):
    def __init__(self, name="Cube Fractal", max_depth=5, max_size=.1, thread_factor=.7):
        super().__init__(CubeSplitter, name=name, max_depth=max_depth, max_size=max_size, thread_factor=thread_factor)






        
        


        

        






        





    

    








