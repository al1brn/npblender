#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module

Created on Fri Nov 10 11:15:36 2023

@author: alain.bernard
@email: alain@ligloo.net

-----

Geometry domains are the core of geometry behavior. A domain manages dynamic attributes suc as position or radius.
Mesh, Curve, Instances and Cloud then manage the relationshp between the domains.

"""

import numpy as np

import importlib.util
import subprocess
import sys

def ensure_modules_installed():
    if importlib.util.find_spec("numba") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numba"])
    if importlib.util.find_spec("scipy") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])

ensure_modules_installed()

from numba import njit

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from constants import SPLINE_TYPES, BEZIER, POLY, NURBS
    from maths.splinesmaths import BSplines, Bezier, Poly, Nurbs

else:
    from . constants import SPLINE_TYPES, BEZIER, POLY, NURBS
    from . maths.splinesmaths import BSplines, Bezier, Poly, Nurbs

from scipy.interpolate import BSpline, make_interp_spline, CubicSpline, splder



# ====================================================================================================
# numba optimized calls
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Face Domain
#
# Build the corner indices array from loop_start and loop_total arrays
# If there is not selection, return np.arange(len(corners))
# It is te be used when working on a selection of faces
# ----------------------------------------------------------------------------------------------------

@njit    
def get_corner_indices(loop_start, loop_total):

    total_len = 0
    n = len(loop_start)

    for i in range(n):
        total_len += loop_total[i]

    result = np.empty(total_len, dtype=loop_start.dtype)

    idx = 0
    for i in range(n):
        start = loop_start[i]
        count = loop_total[i]
        for j in range(count):
            result[idx] = start + j
            idx += 1

    return result

# ----------------------------------------------------------------------------------------------------
# Face Domain
#
# Build the edges indices
# ----------------------------------------------------------------------------------------------------

@njit    
def get_face_edges(loop_start, loop_total):

    edges_count = np.sum(loop_total)
    edges = np.empty((edges_count, 2), dtype=bint)
    
    n = len(loop_start)
    idx = 0
    for i in range(n):
        start = loop_start[i]
        count = loop_total[i]
        for j in range(count):
            edges[idx, 0] = start + j
            edges[idx, 1] = start + ((j + 1) % count)
            idx += 1

    return edges

@njit    
def get_face_per_size(loop_start, loop_total):

    n = len(loop_start)
    size_max = np.max(loop_total) + 1

    # ----- On array per possible size
    arrays  = np.empty((size_max, n), dtype=bint)
    counts  = np.zeros(size_max, dtype=bint)

    # ----- Reverse indices
    indices  = np.empty((size_max, n), dtype=bint)

    for i in range(n):
        start = loop_start[i]
        size = loop_total[i]

        # Array
        idx = counts[size]
        arrays[size, idx] = start

        # Reverse index
        indices[size, idx] = i

        # Next
        counts[size] += 1

    return arrays, counts, indices

@njit    
def get_face_reverse_indices(loop_start, loop_total):

    total = np.sum(loop_total)
    indices = np.empty(total, dtype=bint)

    n = len(loop_start)
    index = 0
    for i in range(n):
        start = loop_start[i]
        count = loop_total[i]
        for j in range(count):
            indices[index] = i
            index += 1

    return indices

@njit
def get_face_position(loop_start, loop_total, vertex_index, position):

    n = len(loop_start)
    pos = np.zeros((n, 3), dtype=bfloat)
    for i in range(n):
        start = loop_start[i]
        count = loop_total[i]
        for j in range(count):
            pos[i] += position[vertex_index[start + j]]
        pos[i] /= count

    return pos


if __name__ == "__main__":
    #import blender
    from fieldarray import FieldArray
    from constants import TYPES, DOMAINS, BL_TYPES, bfloat, bint, bbool
else:
    #from . import blender
    from . fieldarray import FieldArray
    from . constants import TYPES, DOMAINS, BL_TYPES, bfloat, bint, bbool

#from numba import njit, prange

USE_JIT = True

# =============================================================================================================================
# reduce indices
#
# selection is a list of indices to be renumeroted from 0 to max
# [7, 9, 9, 2, 5, 7] -> [2, 3, 3, 0, 1, 2]

def reduce_indices(indices, return_map=False):

    uniq = sorted(list(set(indices)))

    new_indices = np.zeros(np.max(uniq) + 1, int)
    new_indices[uniq] = np.arange(len(uniq))

    if return_map:
        return new_indices
    else:
        return np.array(new_indices[indices])
    
# =============================================================================================================================
# jit compute of area vectors 

#@njit(parallel=True)
def area_vectors_jit(count, vertices, corners, starts, sizes):
    
    vectors = np.empty((count, 3), dtype=np.float32)
    
    for i in prange(count):
        
        vec = np.zeros(3, dtype=np.float32)
        
        start = starts[i]
        size  = sizes[i]
            
        if size == 3:
            v0 = vertices[corners[start]]
            v1 = vertices[corners[start + 1]]
            v2 = vertices[corners[start + 2]]
            
            vec = np.cross(v1 - v0, v2 - v0).astype(np.float32)
            
        elif size == 4:
            v0 = vertices[corners[start]]
            v1 = vertices[corners[start + 1]]
            v2 = vertices[corners[start + 2]]
            v3 = vertices[corners[start + 3]]
            
            vec = np.cross(v1 - v0, v3 - v0).astype(np.float32)
            vec += np.cross(v3 - v2, v1 - v2)
            
        else:
            v0 = vertices[corners[start]]
            for j in range(1, size - 1):
                v1 = vertices[corners[start + j]]
                v2 = vertices[corners[start + j + 1]]
                vec += np.cross(v1 - v0, v2 - v0)
                
        vectors[i] = vec
            
    return vectors

# ====================================================================================================
# Domain root class
# ====================================================================================================

class Domain(FieldArray):

    domain_name = None

    # ----------------------------------------------------------------------------------------------------
    # Dunder
    # ----------------------------------------------------------------------------------------------------

    def __init__(self, a=None, mode='COPY', selector=None):
        """ Initialize the array with another array

        Arguments
        ---------
        - a (array or FieldArray) : the array to initialize from
        - mode (str in 'COPY', 'CAPTURE', 'EMPTY') : copy the content
        - selector (Any = None) : a selector on data 
        """
        super().__init__(a, mode=mode, selector=selector)
        if a is None:
            self.declare_attributes()

    def declare_attributes(self):
        pass

    # ----------------------------------------------------------------------------------------------------
    # Dunder
    # ----------------------------------------------------------------------------------------------------

    def __str__(self):
        len_str = "scalar" if self.is_scalar else f"{len(self):5d}"
        opt = [name for name in self.all_names if name not in self.actual_names]
        return f"<Domain {self.domain_name:8s} ({len_str}), attributes: {self.actual_names} (optional: {opt})>"
    
    # ====================================================================================================
    # Add attributes            
    # ====================================================================================================

    def new_attribute(self, name, data_type, default, optional=False, transfer=True):
        """ Add a new domain attribute.

        Use preferrably user friendly methods 'new_float', 'new_vector', ...

        Arguments
        ----------
            - name (str) : attribute name
            - data_type (str) : attribute type
            - default (any) : default value
            - transfer (bool=True) : transfer as geometry attribute into Blender

        """

        # ----- Validate data_type

        if data_type not in TYPES:
            raise ValueError(f"new_attribute > unknown data type '{data_type}'")

        # ----- Create field with associated metadata

        self.new_field(
            name,
            dtype       = TYPES[data_type]['dtype'],
            shape       = TYPES[data_type]['shape'],
            default     = default,
            optional    = optional,
            data_type   = data_type,
            transfer    = transfer
        )

        # ----- Special case: auto-fill ID if non-empty
        if name == 'ID' and len(self) > 0:
            self['ID'] = np.arange(len(self))

    # ----------------------------------------------------------------------------------------------------

    def new_float(self, name, default=0., optional=False, transfer=True):
        """ Create a new attribute of type FLOAT -> float.

        Arguments
        ---------
            - name (str) : attribute name
            - default (float=0) : default value
            - transfer (bool=True) : transfer the attribute to the Blender mesh
        """
        self.new_attribute(name, 'FLOAT', default, optional=optional, transfer=transfer)

    # ----------------------------------------------------------------------------------------------------

    def new_vector(self, name, default=(0., 0., 0.), optional=False, transfer=True):
        """ Create a new attribute of type VECTOR -> array of 3 floats.

        Arguments
        ---------
            - name (str) : attribute name
            - default (tuple=(0, 0, 0)) : default value
            - transfer (bool=True) : transfer the attribute to the Blender mesh
        """

        self.new_attribute(name, 'VECTOR', default, optional=optional, transfer=transfer)

    # ----------------------------------------------------------------------------------------------------

    def new_int(self, name, default=0, optional=False, transfer=True):
        """ Create a new attribute of type INT -> int.

        Arguments
        ---------
            - name (str) : attribute name
            - default (int=0) : default value
            - transfer (bool=True) : transfer the attribute to the Blender mesh
        """

        self.new_attribute(name, 'INT', default, optional=optional, transfer=transfer)

    # ----------------------------------------------------------------------------------------------------

    def new_bool(self, name, default=False, optional=False, transfer=True):
        """ Create a new attribute of type BOOLEAN -> bool.

        Arguments
        ---------
            - name (str) : attribute name
            - default (bool=False) : default value
            - transfer (bool=True) : transfer the attribute to the Blender mesh
        """

        self.new_attribute(name, 'BOOLEAN', default, optional=optional, transfer=transfer)

    # ----------------------------------------------------------------------------------------------------

    def new_color(self, name, default=(0.5, 0.5, 0.5, 1.), optional=False, transfer=True):
        """ Create a new attribute of type FLOAT_COLOR -> array of 4 floats.

        Arguments
        ---------
            - name (str) : attribute name
            - default (tuple=(0, 0, 0, 1)) : default value
            - transfer (bool=True) : transfer the attribute to the Blender mesh
        """

        #self.new_attribute(name, 'FLOAT_COLOR', default, transfer=transfer)
        self.new_attribute(name, 'COLOR', default, optional=optional, transfer=transfer)

    # ----------------------------------------------------------------------------------------------------

    def new_vector2(self, name, default=(0., 0.), optional=False, transfer=True):
        """ Create a new attribute of type FLOAT2 -> array of 2 floats.

        Arguments
        ---------
            - name (str) : attribute name
            - default tuple=(0, 0)) : default value
            - transfer (bool=True) : transfer the attribute to the Blender mesh
        """

        self.new_attribute(name, 'FLOAT2', default, optional=optional, transfer=transfer)

    # ----------------------------------------------------------------------------------------------------

    def new_quaternion(self, name, default=(1, 0, 0, 0), optional=False, transfer=True):
        """ Create a new attribute of type QUATERNION -> array (4) of floats.

        Arguments
        ---------
            - name (str) : attribute name
            - default tuple=(0, 0)) : default value
            - transfer (bool=True) : transfer the attribute to the Blender mesh
        """
        self.new_attribute(name, 'QUATERNION', default, optional=optional, transfer=transfer)

    # ----------------------------------------------------------------------------------------------------

    def new_matrix(self, name, default=np.eye(4), optional=False, transfer=True):
        """ Create a new attribute of type MATRIX -> array (4x4) of floats.

        Arguments
        ---------
            - name (str) : attribute name
            - default tuple=(0, 0)) : default value
            - transfer (bool=True) : transfer the attribute to the Blender mesh
        """
        self.new_attribute(name, 'MATRIX', default, optional=optional, transfer=transfer)


    # ====================================================================================================
    # Dump

    def dump(self, title="Dump", attributes=None, target='SCREEN'):
        """ Dump the content for an Excel sheet or to be displayed at the screen.
        """

        # ---------------------------------------------------------------------------
        # Formatting a value

        def sv(v):
            if target == 'SCREEN':
                if isinstance(v, (float, np.float64)):
                    return f"{v:.1f}"
                else:
                    return str(v)
            else:
                return str(v)

        # ---------------------------------------------------------------------------
        # Colum width

        def build_col(values):
            col = [sv(v) for v in values]
            col_len = max([len(s) for s in col])
            return col, col_len

        # ---------------------------------------------------------------------------
        # Main

        # ----- Selected attributes

        if attributes is None:
            attributes = list(self.actual_names)

        # ---- Columns

        sizes   = []
        cols    = []
        for name in attributes:
            size = self.attribute_size(name)
            sizes.append(size)
            cols.append([] for _ in range(size))

        # ----- Data lines

        cols     = []
        col_lens = []
        for i_attr, name in enumerate(attributes):

            values = self.attributes[name]

            size = sizes[i_attr]
            if size == 1:
                new_col, l = build_col(values)
                cols.append(new_col)
                col_lens.append(l)

            else:
                cols.append([])

                new_col, l0 = build_col(values[:, 0])
                cols[-1].append(new_col)

                new_col, l1 = build_col(values[:, 1])
                cols[-1].append(new_col)

                new_col, l2 = build_col(values[:, 2])
                cols[-1].append(new_col)

                col_lens.append(max(l0, l1, l2))

        for i, name in enumerate(attributes):
            if sizes[i] == 3:
                continue

            col_lens[i] = max(col_lens[i], len(name))

        # ----- Excel formatting

        data  = '-'*50 + f"\n{title}\nDOMAIN {self.domain_name} DUMP: {len(self)} items\n"
        if target == 'EXCEL':

            # Header

            for name, size in zip(attributes, sizes):
                if size == 1:
                    data += f"{name}; "
                else:
                    data += f"{name}_x; {name}_y; {name}_z; "

            data += "\n"

            # Data lines

            for i in range(len(selection)):
                for size, col in zip(sizes, cols):
                    if size == 1:
                        data += f"{col[i]}; "
                    else:
                        data += f"{col[0][i]}; {col[1][i]}; {col[2][i]}; "
                data += "\n"

        # ----- Screen formatting

        else:

            # Header line

            header_line = "num  | "
            for name, size, col_len in zip(attributes, sizes, col_lens):
                n = col_len if size == 1 else col_len*3 + 4
                header_line += f"{name:{n}s} | "
            header_line += "\n"

            # Data lines

            for i in range(len(self)):
                if i % 50 == 0:
                    data += header_line

                data += f"{i:4d} | "
                for size, col_len, col in zip(sizes, col_lens, cols):
                    if size == 1:
                        data += f"{col[i]:>{col_len}s} | "
                    else:
                        data += f"{col[0][i]:>{col_len}s}  {col[1][i]:>{col_len}s}  {col[2][i]:>{col_len}s} | "
                data += "\n"

        return data
    
    # ====================================================================================================
    # Interface with Blender
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Read data attributes
    # ----------------------------------------------------------------------------------------------------

    def from_bl_attributes(self, bl_attributes):
        """ Read the object attributes.

        Arguments
        ---------
            - spec (str or data) : the data to set the attributes to
        """

        from . import blender

        size = None
        for name, binfo in bl_attributes.items():

            # Must be external and in the domain
            if binfo.domain != self.domain_name or binfo.is_internal:
                continue

            # Create if it doesn't exist
            if not self.has_field(name):
                self.new_attribute(name, binfo.data_type, 0, transfer=True)

            # Adjust size
            if size is None:
                size = len(binfo.data)
                self.resize(size)
            else:
                # Should never append
                assert size == len(binfo.data), "Blender read algorithm is incorrect"

            try:
                self[name] = blender.get_attribute(bl_attributes, name)
            except Exception as e:
                raise Exception(f"Failed to read attribute '{name}' from Blender: {e}")
            
        return self

    # ----------------------------------------------------------------------------------------------------
    # Write data attributes
    # ----------------------------------------------------------------------------------------------------

    def to_bl_attributes(self, attributes, update=False):
        """ Transfer the attributes to a blender mesh object.

        Arguments
        ---------
            - attributes (blender attributes collection) : blender attributes
            - update (bool=False) : update the attributes values without trying to create them
        """

        from . import blender

        for name, info in self._infos.items():

            if not info['transfer']:
                continue

            if info['data_type'] == 'STRING':
                pass

            if update:
                blender.set_attribute(attributes, name, self[name])
            else:
                blender.create_attribute(attributes, name, info['data_type'], domain=self.domain_name, value=self[name])
    
# ====================================================================================================
# Point Domain
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Mesh Point
# ----------------------------------------------------------------------------------------------------

class PointDomain(Domain):
    """ Point domain.

    This domain is the root class for geometries with Point:
        - Mesh : vertices
        - Curve : control points
        - Cloud : points
        - Instances : instance locations

    Attributes
    ----------
        - position (vector) : point position
        - radius (float, optional) : point radius
    """

    domain_name = 'POINT'

    def declare_attributes(self):
        self.new_vector('position', transfer=True)

    # ====================================================================================================
    # Utilities
    # ====================================================================================================

    @property
    def x(self):
        """ x coordinate.

        Shortcut for position[:, 0]
        """
        return self.position[..., 0]

    @x.setter
    def x(self, value):
        self.position[..., 0] = value

    @property
    def y(self):
        """ y coordinate.

        Shortcut for position[:, 1]
        """
        return self.position[..., 1]

    @y.setter
    def y(self, value):
        self.position[..., 1] = value

    @property
    def z(self):
        """ x coordinate.

        Shortcut for position[:, 2]
        """
        return self.position[..., 2]

    @z.setter
    def z(self, value):
        self.position[..., 2] = value

# ----------------------------------------------------------------------------------------------------
# Cloud Point
# ----------------------------------------------------------------------------------------------------

class CloudPointDomain(PointDomain):

    def declare_attributes(self):
        super().declare_attributes()
        self.new_float('radius', default=.05, optional=True, transfer=True)

# ----------------------------------------------------------------------------------------------------
# Spline Point
# ----------------------------------------------------------------------------------------------------

class SplinePointDomain(PointDomain):

    def declare_attributes(self):
        super().declare_attributes()

        self.new_float( 'w',                 optional=True, default=1.)

        self.new_vector('handle_left',       optional=True)
        self.new_vector('handle_right',      optional=True)
        self.new_int(   'handle_type_left',  optional=True)
        self.new_int(   'handle_type_right', optional=True)
        self.new_float( 'tilt',              optional=True)
        self.new_float( 'radius',            optional=True, default=1)
        self.new_float( 'weight',            optional=True, default=1.)

# ----------------------------------------------------------------------------------------------------
# Instance Domain
# ----------------------------------------------------------------------------------------------------

class InstanceDomain(PointDomain):
    """ Instance Domain.

    Instance domain directly inherits from Point domain.
    In addition to position attribute, it managed two more transformations : Scale and Rotation to
    be applied to the instances.

    Instances are randomly chosen in a list of models. The index is stored in the model_index attribute.

    The instances capture attributes from other domains.

    Note that Instances Geometry inherits from Instance domain, contrary to the other geometries which
    store domains as attributes:

    ``` python
    class Mesh(Geometry):
        def __init__(self, ...):
            self.points  = PointDomain.New(...)
            self.corners = CornerDomain.New(...)
            self.faces   = FaceDomain.New(...)

    class Instances(InstanceDomain, Geometry):
        def __init__(self, ...):
            super().__init__(...) # Instances domain initialization

    v = Mesh().position        # Invalid : raises an error
    v = Mesh().points.position # Valid
    v = Instances().position   # Valid
    '''

    Attributes
    ----------
        - position (vector) : instance position
        - model_index (int) : index in the list of models
        - Scale (vector, optional) : instance scale
        - Rotation (vector, optional) : instance rotation

    Arguments
    ---------
        - domain_name (str = None) : 'INSTANCE' or None
        - owner (Instance domain = None) : the selection owner if not None
        - selector (selection = None) : selection if initialized as domain selection
        - points (array of vectors = None) : a point domain
        - models (model spec of list of model specs) : the model to pick into
        - indices (array of ints = None) : model_index initialization
        - seed (int = None) : random seed if indices is None
    """

    domain_name = 'INSTANCE'

    def declare_attributes(self):
        super().declare_attributes()

        self.new_int("model_index", transfer=False)

        self.new_float("animation",         optional=True)
        self.new_float("deformation",       optional=True)

        self.new_vector("scale",            optional=True, default=1)
        self.new_vector("euler",            optional=True)
        self.new_matrix("transformation",   optional=True, default=np.eye(4))
        self.new_vector("quaternion",       optional=True, default=(0, 0, 0, 1))


# ====================================================================================================
# Corner Domain
# ====================================================================================================

class CornerDomain(Domain):
    """ Corner domain stores a vertex index for face descriptions.

    This domain is specific to Mesh geometry.
    It keeps a pointer to the Mesh POINT domain.

    Attributes
    ----------
        - vertex_index (int) : vertex index in the points array
        - UVMap (float2, optional) : UV Map coordinates
    """

    domain_name = 'CORNER'

    def declare_attributes(self):
        self.new_int('vertex_index', transfer=False)
        self.new_vector2('UVMap', (0, 0), optional=True, transfer=True)

    # ----------------------------------------------------------------------------------------------------
    # Check
    # ----------------------------------------------------------------------------------------------------

    def check(self, count, halt=True):
        if not len(self):
            return True
        
        if np.max(self.vertex_index) > count:
            if halt:
                raise RuntimeError(f"CornerDomain check fail: {np.max(self.vertex_index)=}, {count=}")
            else:
                return False
        return True

    # ====================================================================================================
    # Methods
    # ====================================================================================================

    def new_uvmap(self, name, value=None):
        self.new_vector2(name)
        if value is not None:
            self[name] = value

# ====================================================================================================
# Root for Face and Spline
# ====================================================================================================

class FaceSplineDomain(Domain):

    def declare_attributes(self):
        self.new_int('loop_start', transfer=False)
        self.new_int('loop_total', transfer=False)

        self.new_int('material_index', optional=True)

    # ----------------------------------------------------------------------------------------------------
    # Check
    # ----------------------------------------------------------------------------------------------------

    def check(self, count, halt=True):
        if not len(self):
            return True
        
        if np.sum(self.loop_total) != count:
            if halt:
                raise RuntimeError(f"FaceDomain check fail: {np.sum(self.loop_total)=}, {count=}")
            else:
                return False
        return True
    
    def dump_face_spline(self, title="Dump face spline"):
        print(f"\n{title}")
        index = 0
        for i in range(min(20, len(self))):
            str_ok = 'ok' if index == self.loop_start[i] else f"KO: {index} expected"
            print(f"{i:2d}> {self.loop_start[i]:3d} -> {self.loop_total[i]:3d} : {str_ok} ")
            index += self.loop_total[i]
    
    # ====================================================================================================
    # Methods
    # ====================================================================================================

    def update_loop_start(self):
        if len(self):
            a = np.roll(np.cumsum(self.loop_total), 1)
            a[0] = 0
            self.loop_start = a

    @property
    def next_loop_start(self):
        """ Return the value of the loop_start offset for adding new items

        Returns
        -------
            - int : sum of the last loop_start and the last loop_total
        """
        if len(self):
            return self.loop_start[-1] + self.loop_total[-1]
        else:
            return 0

    def compute_loop_start(self, loop_total=None):
        """ Compute the loop_start value of new faces.

        Arguments
        ---------
            - loop_total (array of ints) : the sizes of the items to add

        Returns
            - array of ints : one loop_start value per loop_total starting from next_loop_start
        """

        if loop_total is None:
            return None
        
        elif np.shape(loop_total) == ():
            return self.next_loop_start

        elif len(loop_total):
            a = np.roll(np.cumsum(loop_total), 1)
            a[0] = 0
            return a + self.next_loop_start
        
        else:
            return np.zeros(0, int)
        
    def append_sizes(self, sizes, **fields):

        if sizes is None:
            return []
        
        if 'loop_start' in fields:
            res = self.append(loop_total=sizes, *fields)
        else:
            loop_start = self.compute_loop_start(sizes)
            res = self.append(loop_start=loop_start, loop_total=sizes, **fields)

        return res

    def get_corner_indices(self):
        if self.is_scalar:
            return np.arange(self.loop_total) + self.loop_start
        
        # njit function
        return get_corner_indices(self.loop_start, self.loop_total)
    
    # ----------------------------------------------------------------------------------------------------
    # Delete faces
    # ----------------------------------------------------------------------------------------------------

    def delete(self, selection):
        super().delete(selection)
        self.update_loop_start()

    # ====================================================================================================
    # Sort the item per size
    # ====================================================================================================

    def per_size(self):

        if self.is_scalar:
            return {self.loop_total: np.resize(self.loop_start, (1,))}

        if len(self) == 0:
            return {}
        
        arrays, counts, rev_indices = get_face_per_size(self.loop_start, self.loop_total)
        per_size = {}
        for size, (array, count, rev_inds) in enumerate(zip(arrays, counts, rev_indices)):
            if count:
                per_size[size] = {
                    'start'  : np.array(array[:count]),
                    'indices': np.array(rev_inds[:count])}

        return per_size
    
    # ====================================================================================================
    # Reversed indices
    # ====================================================================================================

    @property
    def reversed_indices(self):
        """ Face index from corner index
        """

        return get_face_reverse_indices(self.loop_start, self.loop_total)    


# ====================================================================================================
# Face Domain
# ====================================================================================================

class FaceDomain(FaceSplineDomain):

    domain_name = 'FACE'

    def delete_loops(self, selection, corners):
        corner_indices = self[selection].get_corner_indices()
        self.delete(selection)
        vert_indices = np.array(corners.vertex_index[corner_indices])
        corners.delete(corner_indices)

        return vert_indices

    # ====================================================================================================
    # Get edges
    # ====================================================================================================

    def get_edges(self, corners):
        if self.is_scalar:
            edge0 = self.loop_start + np.arange(self.loop_total, dtype=bint)
            edge1 = self.loop_start + np.roll(np.arange(self.loop_total, dtype=bint), shift=-1)
            return np.array([edge0, edge1], dtype=bint)

        edges = get_face_edges(self.loop_start, self.loop_total)
        edges = corners.vertex_index[edges]

        edges = np.sort(edges, axis=-1)
        edges = np.unique(edges, axis=0)

        return edges
    
    # ====================================================================================================
    # Get position
    # ====================================================================================================
    
    def get_position(self, corners, points):
        return get_face_position(self.loop_start, self.loop_total, corners.vertex_index, points.position)
    
    # ====================================================================================================
    # Surfaces and normals
    # ====================================================================================================

    # ---------------------------------------------------------------------------
    # Surface is computed by cross products of triangle. This also gives
    # the normal to the face. The vector normal the length of which is the surface
    # is called the "surface vector".

    @staticmethod
    def area_vect(vs, size, return_vector=True):
        if size == 3:
            sv = np.cross(
                    vs[..., 1, :] - vs[..., 0, :],
                    vs[..., 2, :] - vs[..., 0, :])

        elif size == 4:
            sv = (np.cross(
                        vs[..., 1, :] - vs[..., 0, :],
                        vs[..., 3, :] - vs[..., 0, :]
                    ) +
                    np.cross(
                        vs[..., 3, :] - vs[..., 2, :],
                        vs[..., 1, :] - vs[..., 2, :]
                    ))

        else:
            sv = np.zeros((len(vs), 3), float)
            for i in range(size-2):
                sv += FaceDomain.surf_vect(vs[..., [0, i+1, i+2], :], 3)

        if return_vector == 'AREA':
            return np.linalg.norm(sv, axis=-1)/2

        elif return_vector == 'NORMAL':
            return sv / np.linalg.norm(sv, axis=-1)[:, None]

        else:
            return sv

    def area_vectors(self, corners, points):
        """ Compute the surfaces vectors

        The surfaces are computed by cross products of triangles.
        This also gives the normal to the face.
        The normal vector normal the length of which is the surface
        is called the *surface vector*.

        Arguments
        ---------
            - faces (int or array of ints = None) : the faces

        Returns
        -------
            - array of vectors of floats: The surfaces normals
        """

        # Less efficient
        if False and USE_JIT:
            return area_vectors_jit(len(self), self.corners.points.position, self.corners.vertex_index, self.loop_start, self.loop_total)

        # ---------------------------------------------------------------------------
        # Compute the surface for faces of the same size

        def surf_vect(vs, size):

            if size == 3:
                return np.cross(
                        vs[...,1,:] - vs[..., 0,:],
                        vs[...,2,:] - vs[..., 0,:])

            elif size == 4:
                return (np.cross(
                            vs[...,1,:] - vs[..., 0, :],
                            vs[...,3,:] - vs[..., 0, :]
                        ) +
                        np.cross(
                            vs[...,3,:] - vs[..., 2, :],
                            vs[...,1,:] - vs[..., 2, :]
                        ))

            else:
                sv = np.zeros((len(vs), 3), float)
                for i in range(size-2):
                    sv += surf_vect(vs[..., [0, i+1, i+2], :], 3)
                return sv

        # ---------------------------------------------------------------------------
        # The surfaces

        area_vectors = np.zeros((len(self), 3), float)
        for size, fcs in self.per_size().items():

            f_ind = fcs['index']
            c_ind = fcs['loop_index']

            verts = points.position[corners.vertex_index[c_ind]]

            area_vectors[f_ind] = surf_vect(verts, size)

        return area_vectors

    # ---------------------------------------------------------------------------
    # Surfaces : norm of the perpendicular vectors

    def area(self, corners, points):
        """ Faces areas

        Args:
            verts (array (:, 3) of floats): The vertices

        Returns:
            array (len(self)) of floats: The surfaces
        """

        return np.linalg.norm(self.area_vectors(corners, points), axis=-1)/2

    # ---------------------------------------------------------------------------
    # Normals : normalized surface vectors

    def normal(self, corners, points):
        """ Compute the normals

        Args:
            verts (array (:, 3) of floats): The vertices

        Returns:
            array (len(self), 3) of floats: The normals
        """

        sv = self.area_vectors(corners, points)
        return sv / np.linalg.norm(sv, axis=-1)[:, None]

    # ---------------------------------------------------------------------------
    # Centers of the faces

    def position(self, corners, points):
        """ Centers of the faces

        Args:
            verts (array (:, 3) of floats): The vertices

        Returns:
            array (len(self), 3) of floats: The centers
        """

        centers = np.zeros((self.size, 3), float)
        for size, fcs in self.sized_items().items():

            f_ind = fcs['index']
            c_ind = fcs['loop_index']

            verts = self.corners.position[c_ind]

            centers[f_ind] = np.average(verts, axis=1)

        return centers

    # ====================================================================================================
    # Get surface as a dict

    def get_surface(self, corners, points):

        svs = self.area_vectors(corners, points)
        assert(len(svs) == len(self))

        areas2 = np.linalg.norm(svs, axis=-1)

        surf  = {'normals': svs/areas2[:, None], 'areas': areas2/2}
        surf['sizes'] = self.loop_total
        surf['verts'] = points.position

        return surf

    # ====================================================================================================
    # As a list of sequences

    def sequences(self, corners):
        inds = list(corners.vertex_index)
        return [inds[lstart:lstart+ltotal] for (lstart, ltotal) in zip(self.loop_start, self.loop_total)]            

# ====================================================================================================
# Edge Domain
# ====================================================================================================

class EdgeDomain(Domain):
    """ Edge domain.

    Attributes
    ---------
        - vertex0 (int) : index of the first vertex
        - vertex1 (int) : index of the second vertex
    """

    domain_name = 'EDGE'

    @property
    def vertices(self):
        return np.stack((self.vertex0, self.vertex1), axis=-1)
    
    @vertices.setter
    def vertices(self, value):
        if np.shape(value) == (2,):
            self.vertex0 = value[0]
            self.vertex1 = value[1]
        else:
            self.vertex0 = value[:, 0]
            self.vertex1 = value[:, 1]

    def declare_attributes(self):
        self.new_int('vertex0', transfer=False)
        self.new_int('vertex1', transfer=False)

    def check(self, count, halt=True):
        if not len(self):
            return True
        if np.max(self.vertex0) >= count or np.max(self.vertex1) >= count:
            if halt:
                raise Exception(f"Edge domain contains vertex index out of range: {np.max(self.vertex0)=}, {np.max(self.vertex0)=}, {count=}")
            else:
                return False
            
        if np.any(self.vertex0 == self.vertex1):
            if halt:
                raise Exception(f"Some edges use the same vertex ! {np.sum(self.vertex0 == self.vertex1)}")
            else:
                return False

        return True


    # ====================================================================================================
    # Remove edges belonging to faces

    def remove_face_edges(self, face_edges):
    
        if not len(face_edges):
            return
        
        vertices = np.sort(self.vertices, axis=1)
        face_edges = np.sort(face_edges, axis=1)

        u_verts = vertices.view([('', vertices.dtype)] * 2)
        u_faces = face_edges.view([('', face_edges.dtype)] * 2)

        mask = np.isin(u_verts, u_faces).ravel()

        self.delete(mask)


# ====================================================================================================
# Spline Domain

class SplineDomain(FaceSplineDomain):
    """ Spline domain.

    Spline domain is specific to Curve geometry. A spline is an array of control points.
    A Spline is similare to a Face but points directly to the control points and not indirectly
    as for the faces.

    Attributes
    ----------
        - loop_start (int) : first index in control points array
        - loop_total (int) : number of control points
        - material_index (int, optional) : material index
        - resolution (int, optional) : spline resolution
        - cyclic (bool, optional) : spline is cyclic or not
        - order (int, optional) : Nurbs spline order
        - bezierpoint (bool, optional) : Nurbs spline bezierpoint flag
        - endpoint (bool, optional) : Nurbs spline endpoint flag
    """

    domain_name = 'SPLINE'

    def declare_attributes(self):
        super().declare_attributes()

        self.new_int('curve_type')

        self.new_int('resolution',   optional=True, default=16)
        self.new_bool('cyclic',      optional=True, default=False)

        # Nurbs
        self.new_int('order',        optional=True, default=4)
        self.new_bool('bezierpoint', optional=True)
        self.new_bool('endpoint',    optional=True)

    # ====================================================================================================
    # Properties
    # ====================================================================================================

    @property
    def has_bezier(self):
        return np.any(self.curve_type == BEZIER)

    # ----------------------------------------------------------------------------------------------------
    # Delete splines
    # ----------------------------------------------------------------------------------------------------

    def delete_splines(self, selection, cpoints):
        corner_indices = self[selection].get_corner_indices()
        self.delete(selection)
        cpoints.delete(corner_indices)

    # ----------------------------------------------------------------------------------------------------
    # Sample attribute
    # ----------------------------------------------------------------------------------------------------

    def sample_attribute(self, value):
        npoints = len(self.bspline.c)
        count = npoints*self.resolution if self.cyclic else (npoints - 1)*self.resolution + 1


    # ====================================================================================================
    # Adding splines
    # ====================================================================================================

    def add_splines(self, splines, **attributes):
        """ Add splines.

        Arguments
        ---------
            - splines (array of ints) : the number of control points per spline
            - attributes (attribute names, attribute values) : value of the corner attributes
        """
        return self.add_loops(splines, **attributes)

    # ====================================================================================================
    # Parameter
    # ====================================================================================================

    @property
    def functions(self):
        """ Return the functions representing the splines.

        The functions are scipy BSplines initialized with the splines parameters.

        Returns
        -------
            - list of BSpline functions
        """

        funcs = BSplines()

        for i, (curve_type, loop_start, loop_total, cyclic, resolution) in enumerate(zip(self.curve_type, self.loop_start, self.loop_total, self.cyclic, self.resolution)):

            if curve_type == BEZIER:
                funcs.append(Bezier(self.points.position[loop_start:loop_start + loop_total], cyclic=cyclic, resolution=resolution,
                                lefts  = self.points.handle_left[loop_start:loop_start + loop_total],
                                rights = self.points.handle_right[loop_start:loop_start + loop_total]
                                ))

            elif curve_type == POLY:
                funcs.append(Poly(self.points.position[loop_start:loop_start + loop_total], cyclic=cyclic))

            elif curve_type == NURBS:
                funcs.append(Nurbs(self.points.position[loop_start:loop_start + loop_total], cyclic=cyclic, resolution=resolution,
                                    w     = self.w[loop_start:loop_start + loop_total],
                                    order = self.order[loop_start:loop_start + loop_total],
                                    ))

            else:
                assert(False)

        return funcs

    @property
    def length(self):
        """ Length of the splines.

        Returns
        -------
            - List of spline lengths
        """
        return self.functions.length

    def tangent(self, t):
        """ Tangents of the splines at a given time.

        Arguments
        ---------
            - t (float) : spline parameter between 0 and 1

        Returns
        -------
            - list of spline tangents evaluated at time t.
        """
        return self.functions.tangent(t)
    
    # ====================================================================================================
    # Interface with Blender
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Read data attributes
    # ----------------------------------------------------------------------------------------------------

    def load_bl_attributes(self, data):
        """ Read the object attributes.

        Arguments
        ---------
            - spec (str or data) : the data to set the attributes to
        """

        from . import blender

        size = None
        if not hasattr(data, 'attributes'):
            return

        for name, binfo in data.attributes.items():

            # Must be external and in the domain
            if binfo.domain != self.domain or binfo.is_internal:
                continue

            # Create if it doesn't exist
            if not self.has_field(name):
                self.new_attribute(name, binfo.data_type, 0, transfer=True)

            # Adjust size
            if size is None:
                size = len(binfo.data)
                self.resize(size)
            else:
                # Should never append
                assert size == len(binfo.data), "Blender read algorithm is incorrect"

            try:
                self[name] = blender.get_attribute(data, name)
            except Exception as e:
                raise Exception(f"Failed to read attribute '{name}' from Blender: {e}")

    # ----------------------------------------------------------------------------------------------------
    # Write data attributes
    # ----------------------------------------------------------------------------------------------------

    def to_object(self, data, update=False):
        """ Transfer the attributes to a blender mesh object.

        Arguments
        ---------
            - spec (str or data) : the mesh to set the attributes to
            - attributes (array of st = None) : the attributes to transfer (all if None)
            - update (bool=False) : update the attributes values without trying to create them
        """

        from . import blender

        if not hasattr(data, 'attributes'):
            return

        for name, info in self._infos.items():

            if not info['transfer']:
                continue

            if info['data_type'] == 'STRING':
                pass

            if update:
                blender.set_attribute(data, name, self[name])
            else:
                blender.create_attribute(data, name, info['data_type'], domain=self.domain, value=self[name])    
    










from pprint import pprint
from time import time

rng = np.random.default_rng(0)


# ====================================================================================================
# Tests
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Face
# ----------------------------------------------------------------------------------------------------

def test_face_delete(n=10):

    face = FaceDomain()
    loops = rng.integers(3, 7, n)
    face.append_loops(loops)

    ok_check = n < 20

    corners = CornerDomain()
    corners.append(vertex_index=np.arange(np.sum(face.loop_total)))

    if ok_check:
        face.check(corners=corners)
        face.dump_face()

    t0 = time()
    face.delete_loops(0, corners)
    if ok_check:
        face.check(corners=corners)
        face.dump_face()

    face.delete_loops([1, 3], corners)
    if ok_check:
        face.check(corners=corners)
        face.dump_face()

    face.delete_loops(face.loop_total==3, corners)
    if ok_check:
        face.check(corners=corners)
        face.dump_face()
    t1 = time()-t0

    print(f"Perf: {t1:.2f} s")

    print("corners", corners.vertex_index[:10])

def test_face_edges(n=10):

    def dump_edges(e0, e1):
        print("Total edges", len(e0))
        for i in range(min(10, len(e0))):
            print(f"- {i:2d}> ({e0[i]:2d}, {e1[i]:2d})")

    face = FaceDomain()
    loops = rng.integers(3, 7, n)
    face.append_loops(loops)

    face.dump_face()

    t0 = time()
    e0, e1 = face.get_edges()
    t1 = time() - t0
    print(f"Perf: {t1:.2f} s")

    dump_edges(e0, e1)

def test_per_size(n=10):

    face = FaceDomain()
    loops = rng.integers(3, 7, n)
    face.append_loops(loops)

    face.dump_face()

    t0 = time()
    d1 = face.per_size(False)
    t1 = time() - t0

    d = face.per_size(True)

    t0 = time()
    d2 = face.per_size(True)
    t2 = time() - t0


    print(f"Perf: t1={t1:.2f} s, t2={t2:.2f} s")

    if n < 20:
        print("-"*10, "d1")
        pprint(d1)
        print("-"*10, "d2")
        pprint(d2)

    _ = face.reversed_indices

    t0 = time()
    rev = face.reversed_indices
    t1 = time() - t0

    print(f"Reversed indices: t1={t1:.2f} s")

    print(face[:20].reversed_indices)

# ----------------------------------------------------------------------------------------------------
# Control point
# ----------------------------------------------------------------------------------------------------

def test_control_point(n=10):

    cp = ControlPointDomain()
    cp.append(points4=np.arange(16).reshape((4, 4)))

    print(cp.position)
    print(repr(cp))

    cp.append(points4=0)
    print(cp.position)
    print(cp.x)
    print(cp.y)
    print(cp.z)
    print(repr(cp))

    print(cp.points4)
    cp[1].points4 = (9, 8, 7, 6)
    cp.points4[[0, 3]] = [(7, 7, 7, 7), (8, 8, 8, 8)]
    print(cp.points4)

    print(cp.points4)
    cp[1].position = 1
    cp.position[[0, 3]] = [[3.14]]
    print(cp.points4)
    
    print(cp[0].x, cp[0].y, cp[0].z, cp[0].w)
    cp[0].x, cp[0].y, cp[0].z, cp[0].w = 1.0, 1.1, 1.2, 1.3
    print(cp[0].x, cp[0].y, cp[0].z, cp[0].w)




    












class DomainDeprecated:

    # =============================================================================================================================
    # Transformations

    # -----------------------------------------------------------------------------------------------------------------------------
    # Utility for transformations

    def block_size(self, other_shape):
        size = np.prod(other_shape, dtype=int)
        rem = self.size % size
        if rem != 0:
            raise RuntimeError(f"Domain error: impossible to combine domain of size {self.size} {self.shape} with array of size {size} {other_shape}.")
        return self.size // size

    # -----------------------------------------------------------------------------------------------------------------------------
    # Transform : locations, scales and eulers

    def transform(self, transf, pivot=None):
        """ Apply a transformation to the position.

        Note that if the size of the transformation doesn't match the size of the domain, the method trys to apply
        the transformation on blocks. This allow to operate transformation on arrays of geometries.

        If it is not possible to have blocks of the same size, an error is raised.

        ``` python
        # ----- An array of count cubes

        count = 10
        mesh = Mesh.Cube()*10

        # ----- Prepare the transformations

        ags = np.linspace(0, 2*np.pi, count, endpoint=False)
        locs = 10*np.stack((np.cos(ags), np.sin(ags), np.zeros(count, float)), axis=-1)

        transf = Transformations(position=locs)
        transf.rz = ags
        transf.sy = 2

        # ----- Apply to the mesh

        mesh.points.transform(transf)

        # ----- Transformations can be applied directly on the mesh

        mesh.points.scale((.5, 1, 1), pivot=locs)
        mesh.points.translate((0, 4, 0))
        mesh.points.rotate_x(np.linspace(0, 2, count))

        # ----- The domain can be shaped as an array of 8 points

        points = mesh.points.shaped((10, 8))
        for i in range(len(points)):
            points[i].translate((0, 0, i*3))
            points[i].translate((0, 0, 10))

        # ----- Let's view the result

        mesh.to_object("Cubes")
        ```

        Arguments
        ---------
            - transf (Transformations) : the transformation to apply
            - pivot (vector or array of vectors = None) : pivot around which the transformation must be performed

        Returns
        -------
            - self
        """

        # ----------------------------------------------------------------------------------------------------
        # Pivot

        if pivot is not None:
            translations = Transformations(position=pivot)
            translations.position *= -1
            self.transform(translations)

        # ----------------------------------------------------------------------------------------------------
        # Position

        if self.size == transf.size or transf.size == 1:
            self.position = transf @ self.position

        else:
            block_size = self.block_size(transf.shape)
            self.position = np.reshape(
                    transf[..., None] @ np.reshape(self.position, transf.shape + (block_size, 3)),
                    (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Bezier specific

        if self.attribute_exists('handle_left'):
            if self.size == transf.size or transf.size == 1:
                self.handle_left = transf @ self.handle_left
            else:
                self.handle_left = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_left, transf.shape + (block_size, 3)),
                        (len(self), 3))

        if self.attribute_exists('handle_right'):
            if self.size == transf.size or transf.size == 1:
                self.handle_right = transf @ self.handle_right
            else:
                self.handle_right = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_right, transf.shape + (block_size, 3)),
                        (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Position

        if pivot is not None:
            translations.position *= -1
            self.transform(translations)

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Translate

    def translate(self, vectors):
        """ Apply a translation on the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : translation
        Returns
        -------
            - self
        """

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        return self.transform(Transformations(position=vectors))

    # -----------------------------------------------------------------------------------------------------------------------------
    # Change the position

    def locate(self, vectors):
        """ Change the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
        Returns
        -------
            - self
        """

        raise Exception(f"Not yet implemented")

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        if self.size == np.size(vectors)//3 or np.size(vectors) == 1:
            self.position = vectors

        else:
            vshape = np.shape(vectors)[:-1]
            block_size = self.block_size(vshape)

            self.position = np.reshape(np.reshape(vectors, vshape + (1, 3)), (len(self), 3))

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Apply a scale factor

    def scale(self, scales, pivot=None):
        """ Apply a scale.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        if np.shape(scales) == ():
            scales = (scales, scales, scales)

        return self.transform(Transformations(scale=scales), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate the positions

    def rotate(self, rotations, pivot=None):
        """ Apply a rotation.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        return self.transform(Transformations(rotation=rotations), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the x axis

    def rotate_x(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the x axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 0] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_y(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the y axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 1] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_z(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the z axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 2] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around

    def rotate_around(self, axis='Z', angle=0., pivot=None):
        """ Apply a rotation around an axis.

        Arguments
        ---------
            - axis (vectors=(0, 0, 1)) : the axis
            - angle (float=0) : the rotation angle
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        axis = axis_vector(axis)
        if np.shape(angle) == ():
            axis *= angle
        else:
            axis = axis * angle[:, None]

        return self.transform(Transformations(rotation=Rotation.from_rotvec(axis)), pivot=pivot)

    # =============================================================================================================================
    # KDTree

    def kd_tree(self):
        return KDTree(self.position)

    def nearest(self, count=1, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        dist, inds = kdt.query(kdt.data, count+1)
        if count == 1:
            return dist[:, 1], inds[:, 1]
        else:
            return dist[:, 1:], inds[:, 1:]

    def ball_point(self, r=1., remove_self=False, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        a = kdt.query_ball_point(self.position, r=r, return_sorted=False)
        if remove_self:
            for i, l in enumerate(a):
                l.remove(i)

        return a





class DomainDeprecated:

    # =============================================================================================================================
    # Transformations

    # -----------------------------------------------------------------------------------------------------------------------------
    # Utility for transformations

    def block_size(self, other_shape):
        size = np.prod(other_shape, dtype=int)
        rem = self.size % size
        if rem != 0:
            raise RuntimeError(f"Domain error: impossible to combine domain of size {self.size} {self.shape} with array of size {size} {other_shape}.")
        return self.size // size

    # -----------------------------------------------------------------------------------------------------------------------------
    # Transform : locations, scales and eulers

    def transform(self, transf, pivot=None):
        """ Apply a transformation to the position.

        Note that if the size of the transformation doesn't match the size of the domain, the method trys to apply
        the transformation on blocks. This allow to operate transformation on arrays of geometries.

        If it is not possible to have blocks of the same size, an error is raised.

        ``` python
        # ----- An array of count cubes

        count = 10
        mesh = Mesh.Cube()*10

        # ----- Prepare the transformations

        ags = np.linspace(0, 2*np.pi, count, endpoint=False)
        locs = 10*np.stack((np.cos(ags), np.sin(ags), np.zeros(count, float)), axis=-1)

        transf = Transformations(position=locs)
        transf.rz = ags
        transf.sy = 2

        # ----- Apply to the mesh

        mesh.points.transform(transf)

        # ----- Transformations can be applied directly on the mesh

        mesh.points.scale((.5, 1, 1), pivot=locs)
        mesh.points.translate((0, 4, 0))
        mesh.points.rotate_x(np.linspace(0, 2, count))

        # ----- The domain can be shaped as an array of 8 points

        points = mesh.points.shaped((10, 8))
        for i in range(len(points)):
            points[i].translate((0, 0, i*3))
            points[i].translate((0, 0, 10))

        # ----- Let's view the result

        mesh.to_object("Cubes")
        ```

        Arguments
        ---------
            - transf (Transformations) : the transformation to apply
            - pivot (vector or array of vectors = None) : pivot around which the transformation must be performed

        Returns
        -------
            - self
        """

        # ----------------------------------------------------------------------------------------------------
        # Pivot

        if pivot is not None:
            translations = Transformations(position=pivot)
            translations.position *= -1
            self.transform(translations)

        # ----------------------------------------------------------------------------------------------------
        # Position

        if self.size == transf.size or transf.size == 1:
            self.position = transf @ self.position

        else:
            block_size = self.block_size(transf.shape)
            self.position = np.reshape(
                    transf[..., None] @ np.reshape(self.position, transf.shape + (block_size, 3)),
                    (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Bezier specific

        if self.attribute_exists('handle_left'):
            if self.size == transf.size or transf.size == 1:
                self.handle_left = transf @ self.handle_left
            else:
                self.handle_left = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_left, transf.shape + (block_size, 3)),
                        (len(self), 3))

        if self.attribute_exists('handle_right'):
            if self.size == transf.size or transf.size == 1:
                self.handle_right = transf @ self.handle_right
            else:
                self.handle_right = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_right, transf.shape + (block_size, 3)),
                        (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Position

        if pivot is not None:
            translations.position *= -1
            self.transform(translations)

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Translate

    def translate(self, vectors):
        """ Apply a translation on the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : translation
        Returns
        -------
            - self
        """

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        return self.transform(Transformations(position=vectors))

    # -----------------------------------------------------------------------------------------------------------------------------
    # Change the position

    def locate(self, vectors):
        """ Change the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
        Returns
        -------
            - self
        """

        raise Exception(f"Not yet implemented")

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        if self.size == np.size(vectors)//3 or np.size(vectors) == 1:
            self.position = vectors

        else:
            vshape = np.shape(vectors)[:-1]
            block_size = self.block_size(vshape)

            self.position = np.reshape(np.reshape(vectors, vshape + (1, 3)), (len(self), 3))

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Apply a scale factor

    def scale(self, scales, pivot=None):
        """ Apply a scale.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        if np.shape(scales) == ():
            scales = (scales, scales, scales)

        return self.transform(Transformations(scale=scales), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate the positions

    def rotate(self, rotations, pivot=None):
        """ Apply a rotation.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        return self.transform(Transformations(rotation=rotations), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the x axis

    def rotate_x(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the x axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 0] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_y(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the y axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 1] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_z(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the z axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 2] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around

    def rotate_around(self, axis='Z', angle=0., pivot=None):
        """ Apply a rotation around an axis.

        Arguments
        ---------
            - axis (vectors=(0, 0, 1)) : the axis
            - angle (float=0) : the rotation angle
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        axis = axis_vector(axis)
        if np.shape(angle) == ():
            axis *= angle
        else:
            axis = axis * angle[:, None]

        return self.transform(Transformations(rotation=Rotation.from_rotvec(axis)), pivot=pivot)

    # =============================================================================================================================
    # KDTree

    def kd_tree(self):
        return KDTree(self.position)

    def nearest(self, count=1, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        dist, inds = kdt.query(kdt.data, count+1)
        if count == 1:
            return dist[:, 1], inds[:, 1]
        else:
            return dist[:, 1:], inds[:, 1:]

    def ball_point(self, r=1., remove_self=False, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        a = kdt.query_ball_point(self.position, r=r, return_sorted=False)
        if remove_self:
            for i, l in enumerate(a):
                l.remove(i)

        return a


class DomainDeprecated:

    # =============================================================================================================================
    # Transformations

    # -----------------------------------------------------------------------------------------------------------------------------
    # Utility for transformations

    def block_size(self, other_shape):
        size = np.prod(other_shape, dtype=int)
        rem = self.size % size
        if rem != 0:
            raise RuntimeError(f"Domain error: impossible to combine domain of size {self.size} {self.shape} with array of size {size} {other_shape}.")
        return self.size // size

    # -----------------------------------------------------------------------------------------------------------------------------
    # Transform : locations, scales and eulers

    def transform(self, transf, pivot=None):
        """ Apply a transformation to the position.

        Note that if the size of the transformation doesn't match the size of the domain, the method trys to apply
        the transformation on blocks. This allow to operate transformation on arrays of geometries.

        If it is not possible to have blocks of the same size, an error is raised.

        ``` python
        # ----- An array of count cubes

        count = 10
        mesh = Mesh.Cube()*10

        # ----- Prepare the transformations

        ags = np.linspace(0, 2*np.pi, count, endpoint=False)
        locs = 10*np.stack((np.cos(ags), np.sin(ags), np.zeros(count, float)), axis=-1)

        transf = Transformations(position=locs)
        transf.rz = ags
        transf.sy = 2

        # ----- Apply to the mesh

        mesh.points.transform(transf)

        # ----- Transformations can be applied directly on the mesh

        mesh.points.scale((.5, 1, 1), pivot=locs)
        mesh.points.translate((0, 4, 0))
        mesh.points.rotate_x(np.linspace(0, 2, count))

        # ----- The domain can be shaped as an array of 8 points

        points = mesh.points.shaped((10, 8))
        for i in range(len(points)):
            points[i].translate((0, 0, i*3))
            points[i].translate((0, 0, 10))

        # ----- Let's view the result

        mesh.to_object("Cubes")
        ```

        Arguments
        ---------
            - transf (Transformations) : the transformation to apply
            - pivot (vector or array of vectors = None) : pivot around which the transformation must be performed

        Returns
        -------
            - self
        """

        # ----------------------------------------------------------------------------------------------------
        # Pivot

        if pivot is not None:
            translations = Transformations(position=pivot)
            translations.position *= -1
            self.transform(translations)

        # ----------------------------------------------------------------------------------------------------
        # Position

        if self.size == transf.size or transf.size == 1:
            self.position = transf @ self.position

        else:
            block_size = self.block_size(transf.shape)
            self.position = np.reshape(
                    transf[..., None] @ np.reshape(self.position, transf.shape + (block_size, 3)),
                    (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Bezier specific

        if self.attribute_exists('handle_left'):
            if self.size == transf.size or transf.size == 1:
                self.handle_left = transf @ self.handle_left
            else:
                self.handle_left = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_left, transf.shape + (block_size, 3)),
                        (len(self), 3))

        if self.attribute_exists('handle_right'):
            if self.size == transf.size or transf.size == 1:
                self.handle_right = transf @ self.handle_right
            else:
                self.handle_right = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_right, transf.shape + (block_size, 3)),
                        (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Position

        if pivot is not None:
            translations.position *= -1
            self.transform(translations)

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Translate

    def translate(self, vectors):
        """ Apply a translation on the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : translation
        Returns
        -------
            - self
        """

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        return self.transform(Transformations(position=vectors))

    # -----------------------------------------------------------------------------------------------------------------------------
    # Change the position

    def locate(self, vectors):
        """ Change the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
        Returns
        -------
            - self
        """

        raise Exception(f"Not yet implemented")

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        if self.size == np.size(vectors)//3 or np.size(vectors) == 1:
            self.position = vectors

        else:
            vshape = np.shape(vectors)[:-1]
            block_size = self.block_size(vshape)

            self.position = np.reshape(np.reshape(vectors, vshape + (1, 3)), (len(self), 3))

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Apply a scale factor

    def scale(self, scales, pivot=None):
        """ Apply a scale.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        if np.shape(scales) == ():
            scales = (scales, scales, scales)

        return self.transform(Transformations(scale=scales), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate the positions

    def rotate(self, rotations, pivot=None):
        """ Apply a rotation.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        return self.transform(Transformations(rotation=rotations), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the x axis

    def rotate_x(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the x axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 0] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_y(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the y axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 1] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_z(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the z axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 2] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around

    def rotate_around(self, axis='Z', angle=0., pivot=None):
        """ Apply a rotation around an axis.

        Arguments
        ---------
            - axis (vectors=(0, 0, 1)) : the axis
            - angle (float=0) : the rotation angle
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        axis = axis_vector(axis)
        if np.shape(angle) == ():
            axis *= angle
        else:
            axis = axis * angle[:, None]

        return self.transform(Transformations(rotation=Rotation.from_rotvec(axis)), pivot=pivot)

    # =============================================================================================================================
    # KDTree

    def kd_tree(self):
        return KDTree(self.position)

    def nearest(self, count=1, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        dist, inds = kdt.query(kdt.data, count+1)
        if count == 1:
            return dist[:, 1], inds[:, 1]
        else:
            return dist[:, 1:], inds[:, 1:]

    def ball_point(self, r=1., remove_self=False, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        a = kdt.query_ball_point(self.position, r=r, return_sorted=False)
        if remove_self:
            for i, l in enumerate(a):
                l.remove(i)

        return a


class DomainDeprecated:

    # =============================================================================================================================
    # Transformations

    # -----------------------------------------------------------------------------------------------------------------------------
    # Utility for transformations

    def block_size(self, other_shape):
        size = np.prod(other_shape, dtype=int)
        rem = self.size % size
        if rem != 0:
            raise RuntimeError(f"Domain error: impossible to combine domain of size {self.size} {self.shape} with array of size {size} {other_shape}.")
        return self.size // size

    # -----------------------------------------------------------------------------------------------------------------------------
    # Transform : locations, scales and eulers

    def transform(self, transf, pivot=None):
        """ Apply a transformation to the position.

        Note that if the size of the transformation doesn't match the size of the domain, the method trys to apply
        the transformation on blocks. This allow to operate transformation on arrays of geometries.

        If it is not possible to have blocks of the same size, an error is raised.

        ``` python
        # ----- An array of count cubes

        count = 10
        mesh = Mesh.Cube()*10

        # ----- Prepare the transformations

        ags = np.linspace(0, 2*np.pi, count, endpoint=False)
        locs = 10*np.stack((np.cos(ags), np.sin(ags), np.zeros(count, float)), axis=-1)

        transf = Transformations(position=locs)
        transf.rz = ags
        transf.sy = 2

        # ----- Apply to the mesh

        mesh.points.transform(transf)

        # ----- Transformations can be applied directly on the mesh

        mesh.points.scale((.5, 1, 1), pivot=locs)
        mesh.points.translate((0, 4, 0))
        mesh.points.rotate_x(np.linspace(0, 2, count))

        # ----- The domain can be shaped as an array of 8 points

        points = mesh.points.shaped((10, 8))
        for i in range(len(points)):
            points[i].translate((0, 0, i*3))
            points[i].translate((0, 0, 10))

        # ----- Let's view the result

        mesh.to_object("Cubes")
        ```

        Arguments
        ---------
            - transf (Transformations) : the transformation to apply
            - pivot (vector or array of vectors = None) : pivot around which the transformation must be performed

        Returns
        -------
            - self
        """

        # ----------------------------------------------------------------------------------------------------
        # Pivot

        if pivot is not None:
            translations = Transformations(position=pivot)
            translations.position *= -1
            self.transform(translations)

        # ----------------------------------------------------------------------------------------------------
        # Position

        if self.size == transf.size or transf.size == 1:
            self.position = transf @ self.position

        else:
            block_size = self.block_size(transf.shape)
            self.position = np.reshape(
                    transf[..., None] @ np.reshape(self.position, transf.shape + (block_size, 3)),
                    (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Bezier specific

        if self.attribute_exists('handle_left'):
            if self.size == transf.size or transf.size == 1:
                self.handle_left = transf @ self.handle_left
            else:
                self.handle_left = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_left, transf.shape + (block_size, 3)),
                        (len(self), 3))

        if self.attribute_exists('handle_right'):
            if self.size == transf.size or transf.size == 1:
                self.handle_right = transf @ self.handle_right
            else:
                self.handle_right = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_right, transf.shape + (block_size, 3)),
                        (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Position

        if pivot is not None:
            translations.position *= -1
            self.transform(translations)

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Translate

    def translate(self, vectors):
        """ Apply a translation on the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : translation
        Returns
        -------
            - self
        """

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        return self.transform(Transformations(position=vectors))

    # -----------------------------------------------------------------------------------------------------------------------------
    # Change the position

    def locate(self, vectors):
        """ Change the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
        Returns
        -------
            - self
        """

        raise Exception(f"Not yet implemented")

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        if self.size == np.size(vectors)//3 or np.size(vectors) == 1:
            self.position = vectors

        else:
            vshape = np.shape(vectors)[:-1]
            block_size = self.block_size(vshape)

            self.position = np.reshape(np.reshape(vectors, vshape + (1, 3)), (len(self), 3))

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Apply a scale factor

    def scale(self, scales, pivot=None):
        """ Apply a scale.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        if np.shape(scales) == ():
            scales = (scales, scales, scales)

        return self.transform(Transformations(scale=scales), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate the positions

    def rotate(self, rotations, pivot=None):
        """ Apply a rotation.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        return self.transform(Transformations(rotation=rotations), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the x axis

    def rotate_x(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the x axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 0] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_y(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the y axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 1] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_z(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the z axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 2] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around

    def rotate_around(self, axis='Z', angle=0., pivot=None):
        """ Apply a rotation around an axis.

        Arguments
        ---------
            - axis (vectors=(0, 0, 1)) : the axis
            - angle (float=0) : the rotation angle
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        axis = axis_vector(axis)
        if np.shape(angle) == ():
            axis *= angle
        else:
            axis = axis * angle[:, None]

        return self.transform(Transformations(rotation=Rotation.from_rotvec(axis)), pivot=pivot)

    # =============================================================================================================================
    # KDTree

    def kd_tree(self):
        return KDTree(self.position)

    def nearest(self, count=1, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        dist, inds = kdt.query(kdt.data, count+1)
        if count == 1:
            return dist[:, 1], inds[:, 1]
        else:
            return dist[:, 1:], inds[:, 1:]

    def ball_point(self, r=1., remove_self=False, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        a = kdt.query_ball_point(self.position, r=r, return_sorted=False)
        if remove_self:
            for i, l in enumerate(a):
                l.remove(i)

        return a


class DomainDeprecated:

    # =============================================================================================================================
    # Transformations

    # -----------------------------------------------------------------------------------------------------------------------------
    # Utility for transformations

    def block_size(self, other_shape):
        size = np.prod(other_shape, dtype=int)
        rem = self.size % size
        if rem != 0:
            raise RuntimeError(f"Domain error: impossible to combine domain of size {self.size} {self.shape} with array of size {size} {other_shape}.")
        return self.size // size

    # -----------------------------------------------------------------------------------------------------------------------------
    # Transform : locations, scales and eulers

    def transform(self, transf, pivot=None):
        """ Apply a transformation to the position.

        Note that if the size of the transformation doesn't match the size of the domain, the method trys to apply
        the transformation on blocks. This allow to operate transformation on arrays of geometries.

        If it is not possible to have blocks of the same size, an error is raised.

        ``` python
        # ----- An array of count cubes

        count = 10
        mesh = Mesh.Cube()*10

        # ----- Prepare the transformations

        ags = np.linspace(0, 2*np.pi, count, endpoint=False)
        locs = 10*np.stack((np.cos(ags), np.sin(ags), np.zeros(count, float)), axis=-1)

        transf = Transformations(position=locs)
        transf.rz = ags
        transf.sy = 2

        # ----- Apply to the mesh

        mesh.points.transform(transf)

        # ----- Transformations can be applied directly on the mesh

        mesh.points.scale((.5, 1, 1), pivot=locs)
        mesh.points.translate((0, 4, 0))
        mesh.points.rotate_x(np.linspace(0, 2, count))

        # ----- The domain can be shaped as an array of 8 points

        points = mesh.points.shaped((10, 8))
        for i in range(len(points)):
            points[i].translate((0, 0, i*3))
            points[i].translate((0, 0, 10))

        # ----- Let's view the result

        mesh.to_object("Cubes")
        ```

        Arguments
        ---------
            - transf (Transformations) : the transformation to apply
            - pivot (vector or array of vectors = None) : pivot around which the transformation must be performed

        Returns
        -------
            - self
        """

        # ----------------------------------------------------------------------------------------------------
        # Pivot

        if pivot is not None:
            translations = Transformations(position=pivot)
            translations.position *= -1
            self.transform(translations)

        # ----------------------------------------------------------------------------------------------------
        # Position

        if self.size == transf.size or transf.size == 1:
            self.position = transf @ self.position

        else:
            block_size = self.block_size(transf.shape)
            self.position = np.reshape(
                    transf[..., None] @ np.reshape(self.position, transf.shape + (block_size, 3)),
                    (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Bezier specific

        if self.attribute_exists('handle_left'):
            if self.size == transf.size or transf.size == 1:
                self.handle_left = transf @ self.handle_left
            else:
                self.handle_left = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_left, transf.shape + (block_size, 3)),
                        (len(self), 3))

        if self.attribute_exists('handle_right'):
            if self.size == transf.size or transf.size == 1:
                self.handle_right = transf @ self.handle_right
            else:
                self.handle_right = np.reshape(
                        transf[..., None] @ np.reshape(self.handle_right, transf.shape + (block_size, 3)),
                        (len(self), 3))

        # ----------------------------------------------------------------------------------------------------
        # Position

        if pivot is not None:
            translations.position *= -1
            self.transform(translations)

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Translate

    def translate(self, vectors):
        """ Apply a translation on the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : translation
        Returns
        -------
            - self
        """

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        return self.transform(Transformations(position=vectors))

    # -----------------------------------------------------------------------------------------------------------------------------
    # Change the position

    def locate(self, vectors):
        """ Change the positions.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
        Returns
        -------
            - self
        """

        raise Exception(f"Not yet implemented")

        if np.shape(vectors) == ():
            vectors = (vectors, vectors, vectors)

        if self.size == np.size(vectors)//3 or np.size(vectors) == 1:
            self.position = vectors

        else:
            vshape = np.shape(vectors)[:-1]
            block_size = self.block_size(vshape)

            self.position = np.reshape(np.reshape(vectors, vshape + (1, 3)), (len(self), 3))

        return self

    # -----------------------------------------------------------------------------------------------------------------------------
    # Apply a scale factor

    def scale(self, scales, pivot=None):
        """ Apply a scale.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        if np.shape(scales) == ():
            scales = (scales, scales, scales)

        return self.transform(Transformations(scale=scales), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate the positions

    def rotate(self, rotations, pivot=None):
        """ Apply a rotation.

        See Domain.Transform

        Arguments
        ---------
            - vectors (vectors) : the locations
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        return self.transform(Transformations(rotation=rotations), pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the x axis

    def rotate_x(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the x axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 0] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_y(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the y axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 1] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around the y axis

    def rotate_z(self, angle=0, pivot=(0, 0, 0)):
        """ Rotate the vertices around the z axis.

        Arguments
        ---------
            - angle (float=0) : rotation angle
            - pivot (array[3] of floats = (0, 0, 0)) : rotation pivot

        Returns
        -------
            - self
        """

        eulers = np.zeros(np.shape(angle) + (3,), float)
        eulers[..., 2] = angle
        return self.rotate(eulers, pivot=pivot)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Rotate around

    def rotate_around(self, axis='Z', angle=0., pivot=None):
        """ Apply a rotation around an axis.

        Arguments
        ---------
            - axis (vectors=(0, 0, 1)) : the axis
            - angle (float=0) : the rotation angle
            - pivot (vector = None) : scale pivot
        Returns
        -------
            - self
        """

        axis = axis_vector(axis)
        if np.shape(angle) == ():
            axis *= angle
        else:
            axis = axis * angle[:, None]

        return self.transform(Transformations(rotation=Rotation.from_rotvec(axis)), pivot=pivot)

    # =============================================================================================================================
    # KDTree

    def kd_tree(self):
        return KDTree(self.position)

    def nearest(self, count=1, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        dist, inds = kdt.query(kdt.data, count+1)
        if count == 1:
            return dist[:, 1], inds[:, 1]
        else:
            return dist[:, 1:], inds[:, 1:]

    def ball_point(self, r=1., remove_self=False, kd_tree=None):

        kdt = self.kd_tree() if kd_tree is None else kd_tree

        a = kdt.query_ball_point(self.position, r=r, return_sorted=False)
        if remove_self:
            for i, l in enumerate(a):
                l.remove(i)

        return a


class Point_OLD:


    @property
    def bounding_box(self):
        if len(self):
            return np.min(self.position, axis=0), np.max(self.position, axis=0)
        else:
            return np.zeros(3, float), np.zeros(3, float)

    # ====================================================================================================
    # Simple transformations
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Twist
    # ----------------------------------------------------------------------------------------------------

    def twist(self, angle=np.pi/4, origin=(0, 0, 0), direction='X', angle_per_unit=False):
        """ Twist around a line.

        ``` python
        # Build a squared cylinder along x
        cyl = Mesh.Cylinder(vertices=4, depth=10, side_segments=100, transformation=Transformations(rotation=(0, np.pi/2, 0)))

        # Twist along x axis
        cyl.points.twist(2*np.pi, direction='X', angle_per_unit=False)

        # To object
        cyl.to_object("Twist", shade_smooth=False)
        ```

        Arguments
        ---------
            - angle (float=pi/4) : rotation par unit
            - origin (array[3] of floats) : a point on the line to twist around
            - direction (str = 'X') : axis name
            - angle_per_unit (bool=False) : angle is interpretad as a rotation per unit rather than the total twist angle

        Returns
        -------
            - self
        """

        # Unary vector for the direction

        u = axis_vector(direction)

        # Project the vertices on the line
        # The projected points are the centers of the rotations
        # The distance gives the angle to rotate

        d = np.dot(self.position, u)
        cs = u*d[:, None] + origin

        if angle_per_unit:
            ags = d*angle

        else:
            d0  = np.min(d)
            d1  = np.max(d)
            ags = (d - d0)/(d1 - d0)*angle - angle/2

        u_i = axis_index(u, False)

        eulers = np.zeros((len(ags), 3), float)
        eulers[:, u_i] = ags

        transf = Transformation.from_euler(eulers)
        self.points.position = transf @ self.points.position

        return self

    # ----------------------------------------------------------------------------------------------------
    # Bend
    # ----------------------------------------------------------------------------------------------------

    def bend(self, angle=np.pi/2, axis='Z', direction='X', pivot=(0, 0, 0)):
        """ Bend.

        ``` python
        # Build a squared cylinder along x
        cyl = Mesh.Cylinder(vertices=4, depth=10, side_segments=100, transformation=Transformations(rotation=(0, np.pi/2, 0)))
        cyl.to_object("Base", shade_smooth=False)

        # Twist along x axis
        cyl.points.bend(angle=np.pi, axis='Z', direction='X')

        # To object
        cyl.to_object("Bend", shade_smooth=False)
        ```

        Arguments
        ---------
            - angle (float=pi/4) : bend angle
            - axis (axis = 'Z') : rotation axis
            - direction (axis='X') : to direction of the line to bend
            - pivot (axis='X') : The invariant point in the line

        Returns
        -------
            - self
        """
        angle = np.clip(angle, -np.pi*2, np.pi*2)
        if abs(angle) < 0.0001:
            return self

        # ----------------------------------------------------------------------------------------------------
        # Rotate such as
        # - the rotation is around the z axis
        # - the bent line is along the x axis
        # - crossing the y axis at location r

        k = get_axis(axis)
        i = get_axis(direction) # Not necessarily normal to k
        j = np.cross(k, i)
        j /= np.linalg.norm(j)

        M = np.array((np.cross(j, k), j, k))

        verts = np.einsum('...ij, ...j', M, self.position - pivot)

        # ----------------------------------------------------------------------------------------------------
        # We take a radius such as 2pi makes a circle

        x0 = np.min(verts[:, 0])
        x1 = np.max(verts[:, 0])
        length = x1 - x0

        radius = length/angle

        # ----------------------------------------------------------------------------------------------------
        # x gives the angles

        ags  = -verts[:, 0]*(angle/length)

        # ----------------------------------------------------------------------------------------------------
        # y is the "altitude" from the radius

        rs = verts[:, 1] - radius
        verts[:, 0] = rs*np.sin(ags)
        verts[:, 1] = rs*np.cos(ags) + radius

        # ----------------------------------------------------------------------------------------------------
        # Back to initial space

        self.position = np.einsum('...ij, ...j', M.T, verts) + pivot

        return self

    # ----------------------------------------------------------------------------------------------------
    # Shear
    # ----------------------------------------------------------------------------------------------------

    def shear(self, ratio=1., axis='X', plane='Y', pivot=(0, 0, 0), selection=None):
        """ Shear the selection.

        ``` python
        # ----- Build a frame

        cube = MeshBuilder.Cube()
        cube.scale((2, .3, .3))

        # ----- Select left and right faces in two ways

        left_face = cube.sel_faces_from_verts(np.argwhere(cube.verts[:, 0] < -.5).flatten())[0]
        right_face = cube.sel_faces_where(lambda faces: cube.normals(faces)[:, 0] > .5)[0]

        # ----- Left and right shear

        cube.shear(1, axis='X', plane='Y', selection=cube.sel_faces_verts(left_face))
        cube.shear(-1, axis='X', plane='Y', selection=cube.sel_faces_verts(right_face))

        # ----- Extrusion downwards

        cube.extrude([left_face, right_face], offset=3, direction='-Z')

        # ----- Inverse the shear

        cube.shear(2, axis='Z', plane='Y', pivot=cube.centers(left_face), selection=cube.sel_faces_verts(left_face))
        cube.shear(-2, axis='Z', plane='Y', pivot=cube.centers(right_face), selection=cube.sel_faces_verts(right_face))

        # ----- Bridge the two last faces

        cube.bridge_faces(left_face, right_face)

        cube.to_object("Shear", shade_smooth=False)
        ```

        Arguments
        ---------
            - ratio (float=1.) : multiply the distance to the axis to get the translation
            - axis (axis='X') : shear direction
            - plane (axis='Z') : shear plane defined by a perpendicular vector
            - pivot (vector=(0, 0, 0)) : pivot
            - selection (vertice selection=None) : vertex indices which must be sheared

        Returns
        -------
            - self
        """

        if True:
            verts = self.position - pivot

            # ----- Axis and plane

            axis  = get_axis(axis)
            plane = get_axis(plane)
            perp = np.cross(plane, axis)

            # ----- Signed distance to the shear axis

            d = np.einsum('...i, ...i', verts, perp)

            self.position += (d[:, None]*ratio)*axis


        else:
            selection = self.get_verts_selector(selection)
            sel_verts = self._verts[selection] - pivot

            # ----- Axis and plane

            axis  = axis_vector(axis)
            plane = axis_vector(plane)
            perp = np.cross(plane, axis)

            # ----- Signed distance to the shear axis

            d = np.einsum('...i, ...i', sel_verts, perp)

            self._verts[selection] += (d[:, None]*ratio)*axis

        return self