import numpy as np
import pyopenvdb as vdb
import bpy


# ====================================================================================================
# Volume


class Volume:
    def __init__(self, name="vdb"):
        """ > Voume class

        Volume vdb file helper:
        - load / save vdb files
        - build vdb grids from points

        Properties
        ----------
        - name (str) : volume name, used as default objet name and in file names
        - grids (list) : list of volume grids
        - scale (tuple) : volume scale

        Arguments
        ---------
        - name (str = "vdb") : volume name
        """
        self.name  = name
        self.grids = []
        self.scale = (1, 1, 1)

    # ====================================================================================================
    # Get a grid by its index or name

    def __getitem__(self, index):
        """ > Get a grid by its name or index

        Arguments
        ---------
        - index (int or str) : index or name of the grid to get

        Returns
        -------
        - pyopenvdb.Grid
        """
        if isinstance(index, str):
            for grid in self.grids:
                if grid.name == index:
                    return grid
            raise IndexError(f"Volume: grid named '{index}' not found in {[g.name for g in self.grids]}.")
        else:
            return self.grids[index]

    # ====================================================================================================
    # Load / save vdb files

    # ----------------------------------------------------------------------------------------------------
    # Volume frame name

    def get_vol_name(self, frame):
        """ > Volume name

        Volume name is based on the <#name> property and the frame number.

        Arguments
        ---------
        - frame (int) : frame number

        Returns
        -------
        - str : volume name
        """
        return f"{self.name}_{frame:03d}"

    # ----------------------------------------------------------------------------------------------------
    # File path

    def get_file_name(self, frame):
        """ > Build file name

        File name is based on the frame. All the created files can be used in
        a volume animation.

        Arguments
        ---------
        - frame (int) : frame number

        Returns
        -------
        - str : full file path
        """
        from npblender.core import blender
        return str(blender.get_temp_folder() / f"{self.get_vol_name(frame)}.vdb")

    # ----------------------------------------------------------------------------------------------------
    # Load a vdb file

    def load_file(self, file_name):
        """ > Load a vdb file

        Arguments
        ---------
        - file_name (str) : vdb file path

        Returns
        -------
        - self
        """
        grids = vdb.readAllGridMetadata(file_name)
        for grid in grids:
            data = vdb.read(file_name, grid.name)
            self.grids.append(data)
        return self

    # ----------------------------------------------------------------------------------------------------
    # Save a vdb file

    def save_file(self, file_name):
        """ > Save volume into a vdb file

        Arguments
        ---------
        - file_name (str) : vdb file path
        """
        vdb.write(file_name, self.grids)

    # ----------------------------------------------------------------------------------------------------
    # Load a vdb file from its frame number

    def load_frame(self, frame):
        """ > Load a vdb file by its frame number

        Arguments
        ---------
        - frame (int) : frame number

        Returns
        -------
        - self
        """
        return self.load_file(self.get_file_name(frame))

    # ----------------------------------------------------------------------------------------------------
    # Save a vdb file from its frame number

    def save_frame(self, frame):
        """ > Save a vdb file by its frame number

        Arguments
        ---------
        - frame (int) : frame number
        """
        self.save_file(self.get_file_name(frame))

    # ====================================================================================================
    # Volume to object

    def to_object(self, name=None):
        """ > Volume to object

        Create a Blender Volume object from the grids.
        In this version, the grids are save in a vdb file based on the current
        scene frame number, then the file is loaded using "Import VDB file".

        Arguments
        ---------
        - name (str = None) : name of the object to create. Used <#name> property if None.

        Returns
        -------
        - Object : the created volume object
        """

        from npblender.core import blender

        if name is None:
            name = self.name

        # ----- Save the volume in a vdb file

        frame = bpy.context.scene.frame_current
        self.save_frame(frame)

        # ----- Current object

        current_obj = blender.get_object(name, halt=False)

        # ----- Current volume

        vol_name = self.get_vol_name(frame)
        current_vol = bpy.data.volumes.get(vol_name)

        if current_vol is not None:
            if current_obj is None or current_obj.data.name != current_vol.name:
                bpy.data.volumes.remove(current_vol)

        # ----- Import the vdb file

        bpy.ops.object.volume_import(filepath=self.get_file_name(frame), align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        vdb_obj = bpy.context.active_object

        # ----- Creating / replacing

        if current_obj is None:
            vdb_obj.name = name
            obj = vdb_obj

        else:
            old_data = current_obj.data
            current_obj.data.user_remap(vdb_obj.data)
            for mat in old_data.materials:
                current_obj.data.materials.append(mat)

            if old_data is not None:
                if old_data.id_type == 'MESH':
                    bpy.data.meshes.remove(old_data)
                elif old_data.id_type == 'CURVE':
                    bpy.data.curves.remove(old_data)
                elif old_data.id_type == 'VOLUME':
                    bpy.data.volumes.remove(old_data)
                else:
                    raise Exception(f"Unsupported type '{old_data.id_type}'.")

            current_obj.data.name = vol_name
            obj = current_obj

            blender.delete_object(vdb_obj)

        obj.scale = self.scale

    # ====================================================================================================
    # Numpy arrays

    # ----------------------------------------------------------------------------------------------------
    # Create an array from points

    def get_array(self, grid_name, shape=(40, 40, 40)):
        """ > Transform a grid into an array

        Arguments
        ---------
        - grid_name (int or str) : grid index or name
        - shape (tuple = (40, 40, 40)) : shape of the array to create

        Returns
        -------
        - numpy.ndarray : the volume array
        """
        if self.grid_is_vector(grid_name):
            a_shape = shape + (3,)
        else:
            a_shape = shape

        a = np.zeros(a_shape, self.grid_type(grid_name))

        # Copy values from a grid of floats into
        # a three-dimensional array of ints.
        self[grid_name].copyToArray(a) #, ijk=(-15, -20, -35))

        return a

    # ----------------------------------------------------------------------------------------------------
    # Create grids from a cloud of points and named attributes

    def new_grids(self, grid_name, points, resolution=1, **grids):
        """ > Create grids from points and attributes

        The number of grids is the length of the grids argument.

        Arguments
        ---------
        - grid_name (int or str) : grid index or name
        - points (array of vectors) : points locations
        - resolution (float = 1) : grid resolution
        - grids (dict) : grid names and values
        """

        # ----- Array shape

        points = np.asarray(points)

        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])
        minz, maxz = np.min(points[:, 2]), np.max(points[:, 2])

        delta = np.maximum((resolution, resolution, resolution), np.array((maxx - minx, maxy - miny, maxz - minz)))
        shape = tuple((delta/resolution).astype(int)  + (1, 1, 1))

        shape  = np.minimum(2048, shape)
        resols = delta/shape
        ratio  = (shape - (1, 1, 1))/delta
        shape  = tuple(shape)
        self.scale = tuple(resols)

        ijks = ((points - (minx, miny, minz))*ratio).astype(int)

        # ----- Loop on the grids

        for grid_name, grid_values in grids.items():

            # ----- Initialize the grid

            values = np.asarray(grid_values)
            is_vector = len(values.shape) == 2

            if is_vector:
                single_value = len(values) == 1

                if values.dtype == np.float32:
                    grid = vdb.Vec3SGrid()
                elif values.dtype == np.float64:
                    grid = vdb.Vec3DGrid()
                elif values.dtype in (np.int32, np.int64):
                    grid = vdb.Vec3IGrid()
                else:
                    raise ValueError(f"Volume> values of type '{values.dtype}' not supported for vectors grids")

                array_shape = shape + (3,)

            else:
                single_value = values.shape == ()

                if values.dtype == bool:
                    grid = vdb.BoolGrid()
                elif values.dtype == np.int32:
                    grid = vdb.Int32Grid()
                elif values.dtype == np.int64:
                    grid = vdb.Int64Grid()
                elif values.dtype == np.float32:
                    grid = vdb.FloatGrid()
                elif values.dtype == np.float64:
                    grid = vdb.DoubleGrid()
                else:
                    raise ValueError(f"Volume> values of type '{values.dtype}' not supported for grids")

                array_shape = shape

            grid.name = grid_name

            # ----- Copy values to the array

            array = np.zeros(array_shape, values.dtype)
            if single_value:
                for ijk in ijks:
                    array[tuple(ijk)] += values
            else:
                for ijk, value in zip(ijks, values):
                    array[tuple(ijk)] += value

            # ----- Copy array in the grid

            grid.copyFromArray(array, ijk=((minx - resolution/2)/resolution, (miny - resolution/2)/resolution, (minz - resolution/2)/resolution))
            self.grids.append(grid)

            del array

        return self

    # ====================================================================================================
    # Uilities

    # ----------------------------------------------------------------------------------------------------
    # Bounding box

    def bounding_box(self, grid_name):
        """ > Bounding box

        Arguments
        ---------
        - grid_name (int or str) : grid index or name

        Returns
        -------
        - two tuples : bounding box
        """
        return self[grid_name].evalActiveVoxelBoundingBox()

    # ----------------------------------------------------------------------------------------------------
    # Grid type

    def grid_type(self, grid_name):
        """ > Grid type

        Arguments
        ---------
        - grid_name (int or str) : grid index or name

        Returns
        -------
        - type : python type of the array to create
        """

        grid = self[grid_name]
        if isinstance(grid, vdb.BoolGrid):
            return bool
        elif isinstance(grid, vdb.DoubleGrid):
            return np.float64
        elif isinstance(grid, vdb.FloatGrid):
            return np.float32
        elif isinstance(grid, vdb.Int32Grid):
            return np.int32
        elif isinstance(grid, vdb.Int64Grid):
            return np.int64
        elif isinstance(grid, vdb.PointDataGrid):
            raise Exception(f"Volume grid of type '{type(grid).__name__}' is not supported")
            return int
        elif isinstance(grid, vdb.Vec3DGrid):
            return np.float64
        elif isinstance(grid, vdb.Vec3IGrid):
            raise Exception(f"Volume grid of type '{type(grid).__name__}' is not supported")
            return int
        elif isinstance(grid, vdb.Vec3SGrid):
            return np.float32
        else:
            raise Exception(f"Volume grid of type '{type(grid).__name__}' is not supported")

    # ----------------------------------------------------------------------------------------------------
    # Grid is made of vectors

    def grid_is_vector(self, grid_name):
        """ > The grid is maed of vectors

        Arguments
        ---------
        - grid_name (int or str) : grid index or name

        Returns
        -------
        - bool : True if grid is vector, False otherwise
        """
        return isinstance(self[grid_name], (vdb.Vec3DGrid, vdb.Vec3IGrid, vdb.Vec3SGrid))

    # ----------------------------------------------------------------------------------------------------
    # Get an accessor

    def get_accessor(self, grid_name, readonly=False):
        """ > Get an accessor

        ``` python
        with volume.get_accessor(grid_name, readonly=True) as acc:
            ...

        Arguments
        ---------
        - grid_name (int or str) : grid index or name
        - readonly (bool = True) : readonly or read/write accessor
        """
        if readonly:
            accessor = self.grids[grid_name].getConstAccessor()
        else:
            accessor = self.grids[grid_name].getAccessor()

        yield accessor

        del accessor














"""
>>> import pyopenvdb as vdb
    2
    3 # Read two grids from a file.
    4 >>> grids, metadata = vdb.readAll('smoke2.vdb')
    5 >>> [grid.name for grid in grids]
    6 ['density', 'v']
    7
    8 # Get read/write accessors to the two grids.
    9 >>> dAccessor = grids[0].getAccessor()
   10 >>> vAccessor = grids[1].getAccessor()
   11
   12 >>> ijk = (100, 103, 101)
   13
   14 >>> dAccessor.probeValue(ijk)
   15 (0.17614534497261047, True)
   16 # Change the value of a voxel.
   17 >>> dAccessor.setValueOn(ijk, 0.125)
   18 >>> dAccessor.probeValue(ijk)
   19 (0.125, True)
   20
   21 >>> vAccessor.probeValue(ijk)
   22 ((-2.90625, 9.84375, 0.84228515625), True)
   23 # Change the active state of a voxel.
   24 >>> vAccessor.setActiveState(ijk, False)
   25 >>> vAccessor.probeValue(ijk)
   26 ((-2.90625, 9.84375, 0.84228515625), False)
   27
   28 # Get a read-only accessor to one of the grids.
   29 >>> dAccessor = grids[0].getConstAccessor()
   30 >>> dAccessor.setActiveState(ijk, False)
   31   File "<stdin>", line 1, in <module>
   32 TypeError: accessor is read-only
   33
   34 # Delete the accessors once they are no longer needed,
   35 # so that the grids can be garbage-collected.
   36 >>> del dAccessor, vAccessor

"""
