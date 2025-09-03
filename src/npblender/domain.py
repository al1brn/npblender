# MIT License
#
# Copyright (c) 2025 Alain Bernard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the \"Software\"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module Name: domain
Author: Alain Bernard
Version: 0.1.0
Created: 2023-11-10
Last updated: 2025-08-31

Summary:
    Geometry domains are the core of geometry behavior. A domain manages dynamic attributes suc as position or radius.
    Mesh, Curve, Instances and Cloud then manage the relationshp between the domains.

Usage example:
    >>> from domain import MyClass
    >>> obj = MyClass()
    >>> obj.do_something()

Notes:
    - Add any relevant implementation details or dependencies.
    - Update version and dates when modifying this file.
"""

__all__ = [
    "Vertex",
    "ControlPoint",
    "Point",
    "Corner",
    "Face",
    "Edge",
    "Spline",
]

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

from . constants import SPLINE_TYPES, BEZIER, POLY, NURBS
from . maths import Rotation, Quaternion, Transformation
from . fieldarray import FieldArray
from . constants import TYPES, DOMAINS, BL_TYPES, bfloat, bint, bbool
from . maths import distribs


USE_JIT = True

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

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
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


# =============================================================================================================================
# Reduce indices
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
    # Init
    # ----------------------------------------------------------------------------------------------------

    def __init__(self, a=None, mode='COPY', selector=None, attr_from=None, **attrs):
        """Initialize a domain array and its attribute schema.

        Initializes the storage from an existing array/domain or creates an empty
        structure. Optionally merges attribute definitions/values from another
        domain and keyword-provided attributes.

        Domains are never instancied directly but by owning geometries.

        Actual Domains are [`Vertex`][npblender.Vertex], [`Faces`][npblender.Faces],
        [`Corner`][npblender.Corner], [`Edge`][npblender.Edge],
        [`ControlPoint`][npblender.ControlPoint], [`Spline`][npblender.Spline]
        and [`Point`][npblender.Point].

        Domains are initialized with their defaults attributes, for instance `position` for
        point domaines.

        Use attributes can be freely added. 

        > ***Note***:
        > user attributes are saved in Blender Mesh objects only, Blender Curve objects
        > don't store user attributes.

        Parameters
        ----------
        a : array-like or FieldArray or None, optional
            Source data used to initialize the domain. If `None`, an empty domain is
            created and `_declare_attributes()` is called to register defaults.
        mode : {'COPY', 'CAPTURE', 'EMPTY'}, default='COPY'
            Initialization mode. `'COPY'` duplicates the input data, `'CAPTURE'`
            references the input buffer when supported, `'EMPTY'` creates the
            structure without copying values.
        selector : Any, optional
            Optional row/element selector applied to `a` prior to initialization.
        attr_from : Domain or Mapping or None, optional
            Attribute schema (and possibly values) to merge into this domain.
        **attrs
            Additional attribute name/value pairs to inject or override.

        Notes
        -----
        The effective attribute list results from `_declare_attributes()`, then
        `attr_from`, then `**attrs` (later entries take precedence).

        Examples
        --------
        ```python
        cube = Mesh.cube() # points, corners, faces and edges domains are created
        # Adding named attribute to point domain
        cube.points.new_float('age')
        # Setting the age
        cube.points.age = np.random.uniforme(10, 10, len(cube.points))
        ```
        """
        super().__init__(a, mode=mode, selector=selector)

        if a is None:
            self._declare_attributes()
        self.join_attributes(attr_from)

        if len(attrs):
            self.append(**attrs)


    def _declare_attributes(self):
        """Declare built-in attributes for the domain.

        Defines the default attribute schema for the current domain type (names,
        data types, defaults, flags). Called automatically when the domain is
        created without a source.

        Notes
        -----
        Override in subclasses to register domain-specific attributes. Typical
        implementations call helpers like `new_float`, `new_vector`, etc.
        """
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

    def new_attribute(self, name, data_type, default, optional=False, transfer=True, transdom=True):
        """Register a new attribute in the domain schema.

        Creates (or ensures) an attribute with a given name, logical data type,
        default value, and flags controlling Blender transfer and cross-domain
        propagation.

        > ***Note:***
        > `data_type` argument is a Blender data type not a python data type. The
        > data type name is compatible with Blender internal storage. `FLOAT`data type is
        > implemented as `np.float32` and  'INT' as `np.int32`.

        Parameters
        ----------
        name : str
            Attribute name (Python identifier recommended).
        data_type : {'FLOAT', 'INT', 'BOOL', 'VECTOR', 'VECTOR2', 'COLOR', 'QUATERNION', 'MATRIX', 'STRING', ...}
            Logical data type used by the domain.
        default : Any
            Default value for newly allocated elements.
        optional : bool, default=False
            If `True`, the attribute may be absent on some elements.
        transfer : bool, default=True
            If `True`, eligible to be transferred to Blender as a geometry attribute.
        transdom : bool, default=True
            If `True`, considered a *trans-domain* attribute that can be copied
            across compatible domains.

        See Also
        --------
        `new_float`, `new_vector`, `new_int`, `new_bool`, `new_color`, `new_vector2`,
        `new_quaternion`, `new_matrix`
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
            transfer    = transfer,
            transdom    = transdom,
        )

        # ----- Special case: auto-fill ID if non-empty
        if name == 'ID' and len(self) > 0:
            self['ID'] = np.arange(len(self))

    # ----------------------------------------------------------------------------------------------------

    def new_float(self, name, default=0., optional=False, transfer=True, transdom=True):
        """Create or ensure a scalar float attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        default : float, default=0.0
            Default value.
        optional : bool, default=False
        transfer : bool, default=True
        transdom : bool, default=True
        """
        self.new_attribute(name, 'FLOAT', default, optional=optional, transfer=transfer, transdom=transdom)

    # ----------------------------------------------------------------------------------------------------

    def new_vector(self, name, default=(0.0, 0.0, 0.0), optional=False, transfer=True, transdom=True):
        """Create or ensure a 3D vector attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        default : array-like of shape (3,), default=(0.0, 0.0, 0.0)
            Default XYZ vector.
        optional : bool, default=False
        transfer : bool, default=True
        transdom : bool, default=True

        > ***Note:*** The expected per-element shape is `(3,)`.
        """
        self.new_attribute(name, 'VECTOR', default, optional=optional, transfer=transfer, transdom=transdom)

    # ----------------------------------------------------------------------------------------------------

    def new_int(self, name, default=0, optional=False, transfer=True, transdom=True):
        """Create or ensure an integer attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        default : int, default=0
            Default value.
        optional : bool, default=False
        transfer : bool, default=True
        transdom : bool, default=True
        """
        self.new_attribute(name, 'INT', default, optional=optional, transfer=transfer, transdom=transdom)

    # ----------------------------------------------------------------------------------------------------

    def new_bool(self, name, default=False, optional=False, transfer=True, transdom=True):
        """Create or ensure a boolean attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        default : bool, default=False
            Default value.
        optional : bool, default=False
        transfer : bool, default=True
        transdom : bool, default=True
        """
        self.new_attribute(name, 'BOOLEAN', default, optional=optional, transfer=transfer, transdom=transdom)

    # ----------------------------------------------------------------------------------------------------

    def new_color(self, name, default=(0.5, 0.5, 0.5, 1.0), optional=False, transfer=True, transdom=True):
        """Create or ensure a color attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        default : array-like of shape (3,) or (4,), default=(0.5, 0.5, 0.5, 1.0)
            Default color as RGB or RGBA.
        optional : bool, default=False
        transfer : bool, default=True
        transdom : bool, default=True

        > ***Note:*** If alpha is omitted, it is assumed to be 1.0.
        """
        self.new_attribute(name, 'COLOR', default, optional=optional, transfer=transfer, transdom=transdom)

    # ----------------------------------------------------------------------------------------------------

    def new_vector2(self, name, default=(0.0, 0.0), optional=False, transfer=True, transdom=True):
        """Create or ensure a 2D vector attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        default : array-like of shape (2,), default=(0.0, 0.0)
            Default XY vector.
        optional : bool, default=False
        transfer : bool, default=True
        transdom : bool, default=True
        """
        self.new_attribute(name, 'FLOAT2', default, optional=optional, transfer=transfer, transdom=transdom)

    # ----------------------------------------------------------------------------------------------------

    def new_quaternion(self, name, default=(0.0, 0.0, 0.0, 1.0), optional=False, transfer=True, transdom=True):
        """Create or ensure a quaternion attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        default : array-like of shape (4,), default=(0.0, 0.0, 0.0, 1.0)
            Default quaternion in `(x, y, z, w)` convention.
        optional : bool, default=False
        transfer : bool, default=True
        transdom : bool, default=True

        > ***Note:*** Quaternions use `(x, y, z, w)` convention. They are loaded and saved using the
        > Blender `(w, x, y, z)` convention.
        """
        self.new_attribute(name, 'QUATERNION', default, optional=optional, transfer=transfer, transdom=transdom)

    # ----------------------------------------------------------------------------------------------------

    def new_matrix(self, name, default=np.eye(4), optional=False, transfer=True, transdom=True):
        """Create or ensure a matrix attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        default : array-like, default=np.eye(4)
            Default matrix. By convention this is a square matrix per element
            (e.g., `(3, 3)` rotation or `(4, 4)` transform).
        optional : bool, default=False
        transfer : bool, default=True
        transdom : bool, default=True

        > ***Caution:*** Ensure downstream code expects the same matrix storage
        order (row-major vs column-major) as this attribute.
        """
        self.new_attribute(name, 'MATRIX', default, optional=optional, transfer=transfer, transdom=transdom)

    # ----------------------------------------------------------------------------------------------------
    # Trans domain attribute names
    # ----------------------------------------------------------------------------------------------------

    @property
    def transdom_names(self):
        """List the names of trans-domain attributes.

        Returns
        -------
        list of str
            Names of attributes flagged with `transdom=True`.

        Examples
        --------
        ```python
        names = D.transdom_names()
        ```
        """
        return [name for name, info in self._infos.items() if info['transdom']]

    # ----------------------------------------------------------------------------------------------------
    # Join attributes definitions from another domain
    # ----------------------------------------------------------------------------------------------------

    def join_attributes(self, other):
        """Merge trans-domain attributes from another domain.

        Copies or aligns attributes from `other` into the current domain, excluding
        any attributes not flagged as trans-domain in `other`.

        Parameters
        ----------
        other : Domain or None
            Source domain. If `None`, the call is a no-op.

        Returns
        -------
        Domain
            The domain itself (for chaining).
        """
        if other is None:
            return self
        
        exclude = [name for name, info in other._infos.items() if not info['transdom']]
        self.join_fields(other, exclude=exclude)
        return self

    # ----------------------------------------------------------------------------------------------------
    # Transfer trans domain attributes
    # ----------------------------------------------------------------------------------------------------

    def transfer_attributes(self, other, shape=None, other_shape=None):
        """Transfer values of trans-domain attributes from another domain.

        Copies values for each trans-domain attribute present in `other` into the
        corresponding attributes of `self`, with optional reshaping for batched
        assignments.

        Parameters
        ----------
        other : Domain
            Source domain providing attribute values.
        shape : tuple of int or None, optional
            Target reshape for `self` before assignment. If `None`, uses
            `(self._length,)`.
        other_shape : tuple of int or None, optional
            Source reshape for `other` before assignment. If `None`, uses
            `(other._length,)`.

        Returns
        -------
        Domain
            The domain itself (for chaining).

        > ***Note:*** Each attribute is reshaped as `shape + item_shape` on `self`
        and `other_shape + item_shape` on `other` prior to assignment.
        """
        self.join_attributes(other)
        if shape is None:
            shape = (self._length,)
        if other_shape is None:
            other_shape = (other._length,)

        for name in other.transdom_names:
            item_shape = self._infos[name]['shape']
            self[name].reshape(shape + item_shape)[:] = other[name].reshape(other_shape + item_shape)
        return self

    # ====================================================================================================
    # Dump

    def dump(self, title='Dump', attributes=None, target='SCREEN'):
        """Pretty-print or export domain content.

        Formats attribute values and prints to screen or builds a tabular dump
        suitable for spreadsheets.

        Parameters
        ----------
        title : str, default='Dump'
            Title displayed in the report.
        attributes : Sequence[str] or None, optional
            Subset of attribute names to include. If `None`, all attributes are shown.
        target : {'SCREEN', ...}, default='SCREEN'
            Output target. `'SCREEN'` prints to stdout; other targets may trigger
            file creation depending on the implementation.

        Returns
        -------
        None

        Examples
        --------
        ```python
        Domain(points).dump(title="Vertices")
        ```

        > ***Note:*** Formatting adapts to the chosen `target`.
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
        """Import attributes from a Blender attribute collection.

        Reads geometry attributes from a Blender data-block and creates/updates the
        corresponding domain attributes, resizing the domain if needed.

        Parameters
        ----------
        bl_attributes : Mapping[str, Any]
            Blender attributes collection (name → attribute descriptor) providing
            at least `.domain`, `.is_internal`, `.data_type`, and `.data`.

        Returns
        -------
        None

        > ***Note:*** Only external (non-internal) Blender attributes matching this
        domain are imported. Missing attributes are created with `transfer=True`.
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
                arr = blender.get_attribute(bl_attributes, name)
            except Exception as e:
                raise Exception(f"Failed to read attribute '{name}' from Blender: {e}")
            
            # Quaternion convention
            if binfo.data_type == "QUATERNION":
                self[name] = Quaternion.wxyz_to_xyzw(arr)
            else:
                self[name] = arr
            
        return self

    # ----------------------------------------------------------------------------------------------------
    # Write data attributes
    # ----------------------------------------------------------------------------------------------------

    def to_bl_attributes(self, attributes, update=False):
        """Export attributes to a Blender attribute collection.

        Writes eligible domain attributes to a Blender data-block, creating missing
        attributes and adjusting sizes as needed.

        Parameters
        ----------
        attributes : Any
            Blender attributes collection receiving the values.
        update : bool, default=False
            If `True`, update existing attributes in-place; otherwise create them
            when missing.

        Returns
        -------
        None

        > ***Caution:*** Only attributes with `transfer=True` are exported. Optional
        attributes are skipped.

        > ***Caution:*** Curve domains user attributes are not saved.        
        """

        from . import blender

        for name, info in self._infos.items():

            if not info['transfer'] or info['optional']:
                continue

            if info['data_type'] == 'STRING':
                pass

            arr = self[name]
            if info['data_type'] == 'QUATERNION':
                arr = Quaternion.xyzw_to_wxyz(arr)

            if update:
                blender.set_attribute(attributes, name, arr)
            else:
                blender.create_attribute(attributes, name, info['data_type'], domain=self.domain_name, value=arr)

    # ====================================================================================================
    # Create buckets from an attribute
    # ====================================================================================================

    def make_buckets(self, attr):
        """Group items into buckets by attribute value.

        When a domain is to be considered as a collection of packets of various sizes,
        buckets mechanism groups pakets by size, allowing further operation with
        numpy vectorization.

        Parameters
        ----------
        attr : array-like or str
            Either an integer of shape `(N,)` or the name of an existing
            integer attribute in the domain.

        Returns
        -------
        list[np.ndarray (count, n)]
            A list of int arrays (count, n): count is the number of buckets of length n.

        Examples
        --------
        ```python
        buckets = mesh.make_buckets('material')
        for bucket in buckets:
            print(bucket.shape)
        ```

        > ***Note:*** The bucket attribute can be read with `attr[bucket[:, 0]]`.
        """
        # ---------------------------------------------------------------------------
        # Check attribute
        # ---------------------------------------------------------------------------

        if isinstance(attr, str):
            groups = self[attr]
        else:
            groups = np.asarray(attr)

        if groups.shape != (len(self),) or (groups.dtype not in (int, np.int32, np.int64)):
            raise ValueError(f"Attribute must be an array({len(self)},) of ints, not array{attr.shape} of {attr.dtype}.")

        # ---------------------------------------------------------------------------
        # Make the buckets
        # ---------------------------------------------------------------------------

        idx = np.arange(groups.size)
        counts = np.bincount(groups)

        # Sort by contigous blocks
        gcounts = counts[groups]
        order = np.lexsort((idx, groups, gcounts))
        
        gc_ordered = gcounts[order]
        buckets = []        
        for count in np.unique(gcounts):
            sel = order[gc_ordered == count]
            buckets.append(sel.reshape(-1, count))        

        return buckets
    
    # ====================================================================================================
    # Check attribute to compute on another domain
    # ====================================================================================================

    def _check_attribute_to_compute(self, attr):
        """Validate that an attribute is available and aligned with the domain.

        Parameters
        ----------
        attr : str or ndarray
            Attribute name or array to check.

        Raises
        ------
        AttributeError
            If the attribute is scalar or its length does not match `len(self)`.

        > ***Note:*** If `attr` is a string, the corresponding field is looked up
        in `self` before validation.
        """
        if isinstance(attr, str):
            attr = self[attr]
        
        if attr.shape == () or len(attr) != len(self):
            raise AttributeError(f"Domain '{type(self).__name__}': attribute (shape {attr.shape}) should have a length of {len(self)}.")
        
        return attr, attr.shape[1:]
    
    # ====================================================================================================
    # shape for operation
    # ====================================================================================================

    def _get_shape_for_operation(self, op_shape, title="Operation"):
        """Infer operand and output shapes for vectorized operations.

        Given an operand shape, returns a pair `(out_shape, op__shape)` that
        aligns with the domain size and a trailing vector dimension of 3 when relevant.

        Parameters
        ----------
        op_shape : tuple of int
            Shape of the input operand (without trailing vector dims).
        title : str, default='Operation'
            Operation title used in error messages.

        Returns
        -------
        out_shape : tuple of int
            Shape for the vectorized result, typically ending with `(…, 3)`.
        op_shape : tuple of int
            Shape for the operand to align the operand with the domain size.

        Raises
        ------
        ValueError
            If the operand cannot be aligned with the domain size.

        Examples
        --------
        ```python
        out_shape, bshape = D._get_shape_for_operation(op_shape=(N,), title="Translate")
        ```
        """
        shape = self.shape
        size = int(np.prod(shape))

        op_size = int(np.prod(op_shape))
        if op_size == size:
            return op_shape + (3,), op_shape
        
        elif size > op_size:
            if size % op_size == 0:
                n = size // op_size
                return op_shape + (n, 3), op_shape + (1,)

        raise ValueError(
            f"Size of domain '{type(self).__name__}' is {size}. "
            f"{title} is not possible with operand of shape {op_shape}.")
    
# ====================================================================================================
# Point Domain
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Root class for points
# ----------------------------------------------------------------------------------------------------

class PointDomain(Domain):
    """ Point domain.

    This domain is the root class for geometries with Point:
        - Mesh : vertices
        - Curve : control points
        - Cloud : points
        - Instances/Meshes : instance locations

    Attributes
    ----------
        - position (vector) : point position
    """

    domain_name = 'POINT'

    def _declare_attributes(self):
        """
        Declare core point-domain attributes.

        Initializes the mandatory ``position`` vector field for the domain.
        """
        self.new_vector('position', transfer=True, transdom=False)

    # ====================================================================================================
    # Properties
    # ====================================================================================================

    @property
    def x(self):
        """
        X coordinate accessor.

        Shorthand for ``position[..., 0]``. Reading returns a view; assigning to
        this property writes into the underlying ``position`` field.

        Returns
        -------
        ndarray
            View of shape ``(...,)`` selecting the x component of ``position``.

        Examples
        --------
        Read and write x in place:

        ```python
        xs = points.x              # view on position[..., 0]
        points.x = xs + 1.0        # shift x by +1
        ```

        > ***Note:*** This is equivalent to ``points.position[..., 0]``.
        """
        return self.position[..., 0]

    @x.setter
    def x(self, value):
        self.position[..., 0] = value

    @property
    def y(self):
        """
        Y coordinate accessor.

        Shorthand for ``position[..., 1]``. Reading returns a view; assigning to
        this property writes into the underlying ``position`` field.

        Returns
        -------
        ndarray
            View of shape ``(...,)`` selecting the y component of ``position``.

        Examples
        --------
        ```python
        points.y = 0.0             # flatten all y to 0
        ```
        """
        return self.position[..., 1]

    @y.setter
    def y(self, value):
        self.position[..., 1] = value

    @property
    def z(self):
        """
        Z coordinate accessor.

        Shorthand for ``position[..., 2]``. Reading returns a view; assigning to
        this property writes into the underlying ``position`` field.

        Returns
        -------
        ndarray
            View of shape ``(...,)`` selecting the z component of ``position``.

        Examples
        --------
        ```python
        points.z += 2.5
        ```
        """
        return self.position[..., 2]

    @z.setter
    def z(self, value):
        self.position[..., 2] = value

    # ====================================================================================================
    # Transformations
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Internal methods using attribute name, 'position' is default
    # ----------------------------------------------------------------------------------------------------

    def _pivot(self, pivot, shape_pv=None):
        """Temporarily shift points to/from a local pivot frame.

        When a pivot is provided, this helper subtracts the pivot from
        `position` (global → local) before an operation, and when called again
        with the returned tuple, it restores the original frame (local → global).

        Parameters
        ----------
        pivot : array-like of shape ``(..., 3)`` or None
            Pivot location(s). If `None`, no-op and returns `None`.
        shape_pv : tuple or None, optional
            Tuple returned by a previous call, used to restore the frame.

        Returns
        -------
        tuple or None
            When entering local space: ``(shape_out, pivot_broadcasted)`` where
            ``shape_out`` is the output shape from
            [`_get_shape_for_operation`](npblender._get_shape_for_operation)
            and ``pivot_broadcasted`` has shape ``(..., 3)``.
            When restoring: returns `None`.
        """
        if pivot is None:
            return None

        # Global -> local
        if shape_pv is None:
            pv = np.asarray(pivot)
            shape, op_shape = self._get_shape_for_operation(pv.shape[:-1], title="Scale pivot")
            pos = self.position.reshape(shape)
            pv = pv.reshape(op_shape + (3,))
            pos -= pv
            return shape, pv
        
        else:
            pos = self.position.reshape(shape_pv[0] + (3,))
            pos += shape_pv[1]

    def _translate(self, translation, attr_name='position'):
        tr = np.asarray(translation)
        shape, op_shape = self._get_shape_for_operation(tr.shape[:-1], title="Translation")
        pos = self[attr_name].reshape(shape)
        pos += tr.reshape(op_shape + (3,))

        return self
    
    def _apply_scale(self, scale, pivot=None, attr_name='position'):
        shape_pv = self._pivot(pivot)

        sc = np.asarray(scale)
        shape, op_shape = self._get_shape_for_operation(sc.shape[:-1], title="Scale")
        pos = self[attr_name].reshape(shape)
        pos *= sc.reshape(op_shape + (3,))

        self._pivot(pivot, shape_pv)

        return self
    
    def _transform(self, transfo, pivot=None, attr_name='position'):
        shape_pv = self._pivot(pivot)

        shape, op_shape = self._get_shape_for_operation(transfo.shape, title="Scale")
        pos = self[attr_name].reshape(shape)
        pos[:] = transfo.reshape(op_shape) @ pos

        self._pivot(pivot, shape_pv)

        return self
    
    # ----------------------------------------------------------------------------------------------------
    # Exposed methods
    # ----------------------------------------------------------------------------------------------------

    def translate(self, translation):
        """Translate points position by a vector or a batch of vectors.

        Supports per-domain translation (single vector), or grouped/batched
        translations that broadcast over buckets of equal size.

        Parameters
        ----------
        translation : array-like of shape ``(..., 3)``
            Translation vectors broadcastable to the domain size.

        Returns
        -------
        PointDomain
            Self (for chaining).

        Examples
        --------
        ```python
        # Per-point random translation
        D.translate(np.random.uniform(-0.1, 0.1, (len(D), 3)))

        ```python
        # A mesh made of 8 cubes
        cubes = Mesh.cube(size=.2)*8
        tr = np.random.uniform(-1, 1, (8, 3))
        # Translate each cube individually
        cubes.points.translate(tr)
        ```

        > ***Caution:*** If a provided batch cannot be aligned with the domain,
        a `ValueError` is raised by
        [`_get_shape_for_operation`](npblender._get_shape_for_operation).
        """
        return self._translate(translation)

        tr = np.asarray(translation)
        shape, op_shape = self._get_shape_for_operation(tr.shape[:-1], title="Translation")
        pos = self.position.reshape(shape)
        pos += tr.reshape(op_shape + (3,))

        return self
    

    
    def apply_scale(self, scale, pivot=None):
        """Apply per-axis scales to points, optionally about a pivot.

        The scaling is broadcast across the domain using
        [`_get_shape_for_operation`](npblender._get_shape_for_operation).
        If a pivot is given, points are moved to the local frame, scaled, then moved back.

        Parameters
        ----------
        scale : array-like of shape ``(..., 3)``
            Per-axis scale factors broadcastable to the domain size.
        pivot : array-like of shape ``(..., 3)`` or None, optional
            Pivot location(s). If `None`, scales are applied about the origin.

        Returns
        -------
        PointDomain
            Self (for chaining).

        Examples
        --------
        ``` python
        # A mesh made of 8 cubes
        cubes = Mesh.cube(size=.2)*8
        pv = np.random.uniform(-1, 1, (8, 3))
        sc = np.random.uniform(.1, 1, (8, 3))
        # Scale each cube individually
        cubes.points.apply_scale(sc, pivot=pv)
        ```

        > ***Note:*** If broadcasting fails, a `ValueError` is raised by
        [`_get_shape_for_operation`](npblender._get_shape_for_operation).
        """
        return self._apply_scale(scale, pivot=pivot)
    

        shape_pv = self._pivot(pivot)

        sc = np.asarray(scale)
        shape, op_shape = self._get_shape_for_operation(sc.shape[:-1], title="Scale")
        pos = self.position.reshape(shape)
        pos *= sc.reshape(op_shape + (3,))

        self._pivot(pivot, shape_pv)

        return self

    def transform(self, transfo, pivot=None):
        """Apply a linear transform (e.g., rotation or transformation) to points.

        If the transformation or pivot size is less than the domain size,
        the scale / pivot is applied on buckets of the same size
        if possible, otherwise an exception is raised.

        Parameters
        ----------
        transfo : Transformation, Rotation or Quaternion
            Transform(s) broadcastable to the domain size. Typical shapes include
            ``(..., 3, 3)``; project-specific transform types are also supported if
            they define the ``@`` operator with vectors.
        pivot : array-like of shape ``(..., 3)`` or None, optional
            Pivot location(s). If `None`, transforms are applied about the origin.

        Returns
        -------
        PointDomain
            Self (for chaining).

        Examples
        --------
        ``` python
        # A mesh made of 8 cubes
        cubes = Mesh.cube(size=1)*8
        pv = np.random.uniform(-1, 1, (8, 3))
        rot = Rotation.from_euler(np.random.uniform(0, 2*np.pi, (8, 3)))
        # Transform the 8 cubes indivicually
        cubes.points.transform(rot, pivot=pv)
        ```

        Examples:
        ---------
        ``` python
        # A mesh made of 8 cubes
        cubes = Mesh.cube(size=1)*8
        pv = np.random.uniform(-1, 1, (8, 3))
        rot = Rotation.from_euler(np.random.uniform(0, 2*np.pi, (8, 3)))
        # Transform the 8 cubes indivicually
        cubes.points.transform(rot, pivot=pv)
        ```
        """
        return self._transform(transfo, pivot=pivot)


        shape_pv = self._pivot(pivot)

        shape, op_shape = self._get_shape_for_operation(transfo.shape, title="Scale")
        pos = self.position.reshape(shape)
        pos[:] = transfo.reshape(op_shape) @ pos

        #self.position = transfo @ self.position
        
        self._pivot(pivot, shape_pv)

        return self
    
# ----------------------------------------------------------------------------------------------------
# Vertex (Mesh)
# ----------------------------------------------------------------------------------------------------

class Vertex(PointDomain):
    """
    Vertex (mesh point) domain.

    This domain represents mesh vertices. It provides fast utilities to
    transfer (map/average) any per-vertex attribute onto other mesh domains
     (faces, corners, edges).

    Attributes
    ----------
    position : (N, 3) float
        World-space vertex positions inherited from :class:`PointDomain`.

    Notes
    -----
    - Attribute transfers are implemented with Numba-jitted kernels for
      performance.
    - When mapping to faces, values are averaged over all corners of each face.

    Examples
    --------
    Map a per-vertex scalar to faces:

    ```python
    face_attr = vertices.compute_attribute_on_faces("mass", corners, faces)
    ```

    Map a per-vertex vector to edges:

    ```python
    edge_attr = vertices.compute_attribute_on_edges("normal", edges)
    ```
    """

    # ----------------------------------------------------------------------------------------------------
    # Compute attribute on faces
    # ----------------------------------------------------------------------------------------------------
    
    def compute_attribute_on_faces(self, attr, corners, faces):
        """Average a per-vertex attribute over each face.

        For every face, this computes the mean of the source vertex attribute over
        all its incident corners (i.e., the face-wise average). The input attribute
        can be given by name or as an array aligned with the vertex domain. The
        attribute is validated with
        [`_check_attribute_to_compute`](npblender._check_attribute_to_compute).

        Parameters
        ----------
        attr : str or ndarray, shape ``(N, ...)``
            Vertex attribute to aggregate. If a string, the corresponding field is
            looked up on this domain; if an array, it must have length ``N == len(self)``.
            Trailing item shape (``...``) is preserved.
        corners : Corner
            Corner domain providing the ``vertex_index`` mapping for the mesh.
        faces : Face
            Face domain providing ``loop_start`` and ``loop_total`` (polygon topology).

        Returns
        -------
        ndarray, shape ``(len(faces), ...)``
            Face-wise averaged attribute. The trailing item shape is preserved and
            the dtype follows the input attribute.

        Raises
        ------
        AttributeError
            If `attr` is scalar or its first dimension does not match ``len(self)`` (raised by
            [`_check_attribute_to_compute`](npblender._check_attribute_to_compute)).
        IndexError
            If `corners.vertex_index` contains indices outside ``[0, len(self))``.
        TypeError
            If the attribute dtype is non-numeric or cannot be averaged (e.g., integer types
            with in-place division).

        Notes
        -----
        This routine computes an **unweighted arithmetic mean** over each face's corners.

        See Also
        --------
        [`compute_attribute_on_corners`](npblender.Vertex.compute_attribute_on_corners) :
            Scatter vertex attributes to corners.
        [`compute_attribute_on_edges`](npblender.Vertex.compute_attribute_on_edges) :
            Average vertex attributes on edges.

        Examples
        --------
        ```python
        # Face centroids (average of vertex positions)
        face_pos = V.compute_attribute_on_faces("position", corners, faces)

        # Average any custom per-vertex float attribute
        face_weight = V.compute_attribute_on_faces(weights, corners, faces)
        ```
        """

        @njit(cache=True)
        def _to_faces(loop_start, loop_total, vertex_index, source, res):

            nfaces = loop_start.shape[0]
            for iface in range(nfaces):
                start = loop_start[iface]
                total = loop_total[iface]
                for icorner in range(start, start + total):
                    vi = vertex_index[icorner]
                    res[iface] += source[vi]
                inv = 1.0 / total
                res[iface] *= inv

            return res        

        attr, item_shape = self._check_attribute_to_compute(attr)
        res = np.zeros((len(faces),) + item_shape, dtype=attr.dtype)
        return _to_faces(faces.loop_start, faces.loop_total, corners.vertex_index, attr, res)
    
    # ----------------------------------------------------------------------------------------------------
    # Compute attribute on corners
    # ----------------------------------------------------------------------------------------------------
    
    def compute_attribute_on_corners(self, attr, corners):
        """Scatter a per-vertex attribute to corners.

        For each corner, copies the attribute of its associated vertex (via
        ``corners.vertex_index``). The attribute is validated with
        [`_check_attribute_to_compute`](npblender._check_attribute_to_compute).

        Parameters
        ----------
        attr : str or ndarray, shape ``(N, ...)``
            Vertex attribute to scatter. If a string, the field is looked up on this
            domain; if an array, it must have length ``N == len(self)``.
        corners : Corner
            Corner domain providing the ``vertex_index`` mapping.

        Returns
        -------
        ndarray, shape ``(len(corners), ...)``
            Corner attribute array (one value per corner), preserving the trailing
            item shape and dtype.

        Raises
        ------
        AttributeError
            If `attr` is scalar or its first dimension does not match ``len(self)`` (raised by
            [`_check_attribute_to_compute`](npblender._check_attribute_to_compute)).
        IndexError
            If `corners.vertex_index` contains indices outside ``[0, len(self))``.

        Examples
        --------
        ```python
        # Duplicate per-vertex colors to corners
        corner_col = V.compute_attribute_on_corners("color", corners)
        ```
        """

        @njit(cache=True)
        def _to_corners(vertex_index, source, res):
            ncorners = vertex_index.shape[0]
            for icorner in range(ncorners):
                vi = vertex_index[icorner]
                res[icorner] += source[vi]
            return res        

        attr, item_shape = self._check_attribute_to_compute(attr)
        res = np.zeros((len(corners),) + item_shape, dtype=attr.dtype)
        return _to_corners(corners.vertex_index, attr, res)    

    # ----------------------------------------------------------------------------------------------------
    # Compute attribute on edges
    # ----------------------------------------------------------------------------------------------------
    
    def compute_attribute_on_edges(self, attr, edges):
        """Average a per-vertex attribute over each edge.

        For every edge, returns the mean of the attribute at its two endpoint
        vertices (``(v0, v1)``). The attribute is validated with
        [`_check_attribute_to_compute`](npblender._check_attribute_to_compute).

        Parameters
        ----------
        attr : str or ndarray, shape ``(N, ...)``
            Vertex attribute to average. If a string, the field is looked up on this
            domain; if an array, it must have length ``N == len(self)``.
        edges : Edge
            Edge domain providing ``vertex0`` and ``vertex1`` index arrays.

        Returns
        -------
        ndarray, shape ``(len(edges), ...)``
            Edge-wise averaged attribute. The trailing item shape is preserved and
            the dtype follows the input attribute.

        Raises
        ------
        AttributeError
            If `attr` is scalar or its first dimension does not match ``len(self)`` (raised by
            [`_check_attribute_to_compute`](npblender._check_attribute_to_compute)).
        IndexError
            If `edges.vertex0` or `edges.vertex1` contain indices outside ``[0, len(self))``.
        TypeError
            If the attribute dtype is non-numeric or cannot be averaged (e.g., integer types
            with in-place division).

        Examples
        --------
        ```python
        # Edge midpoints from vertex positions
        edge_pos = V.compute_attribute_on_edges("position", edges)

        # Average a custom per-vertex scalar on edges
        edge_w = V.compute_attribute_on_edges(weights, edges)
        ```

        > ***Note:*** The average is unweighted: ``0.5 * (attr[v0] + attr[v1])``.
        """

        @njit(cache=True)
        def _to_edges(vertex0, vertex1, source, res):
            nedges = vertex0.shape[0]
            for iedge in range(nedges):
                v0 = vertex0[iedge]
                v1 = vertex1[iedge]
                res[iedge] = (source[v0] + source[v1])* 0.5                
            return res        

        attr, item_shape = self._check_attribute_to_compute(attr)
        res = np.zeros((len(edges),) + item_shape, dtype=attr.dtype)
        return _to_edges(edges.vertex0, edges.vertex1, attr, res)    

# ----------------------------------------------------------------------------------------------------
# Control Point (Curve)
# ----------------------------------------------------------------------------------------------------

class ControlPoint(PointDomain):
    """
    Curve control point domain.

    Attributes
    ----------
    position : (N, 3) float
        Control point positions.
    w : (N,) float, optional
        Rational weight (NURBS).
    handle_left, handle_right : (N, 3) float, optional
        Bezier handles in object space.
    handle_type_left, handle_type_right : (N,) int, optional
        Bezier handle types.
    tilt : (N,) float, optional
        Tilt (radians).
    radius : (N,) float, optional
        Display radius.
    weight : (N,) float, optional
        Generic user-defined weight.

    See Also
    --------
    `Spline` : Spline domain grouping control points.

    Notes
    -----
    - Scaling also applies to handles.
    - Rotational transforms rotate handles using the transform's rotation
      only (scales are not applied to handles).

    Examples
    --------
    Average a control-point attribute on splines:

    ```python
    mean_tilt = cpoints.compute_attribute_on_splines("tilt", splines)
    ```

    Apply a transform that rotates handles consistently:

    ```python
    cpoints.transform(T)  # where T is a Transformation or Rotation
    ```
    """

    def _declare_attributes(self):
        super()._declare_attributes()

        self.new_float( 'w',                 optional=True, default=1., transdom=False)

        self.new_vector('handle_left',       optional=True, transdom=False)
        self.new_vector('handle_right',      optional=True, transdom=False)
        self.new_int(   'handle_type_left',  optional=True, transdom=False)
        self.new_int(   'handle_type_right', optional=True, transdom=False)
        self.new_float( 'tilt',              optional=True, transdom=False)
        self.new_float( 'radius',            optional=True, default=1)
        self.new_float( 'weight',            optional=True, default=1.)

    # ----------------------------------------------------------------------------------------------------
    # Compute attribute on splines
    # ----------------------------------------------------------------------------------------------------
    
    def compute_attribute_on_splines(self, attr, splines):
        """Average a per-control-point attribute over each spline.

        For every spline, computes the mean of the source control-point attribute
        over its range ``[loop_start, loop_start + loop_total)``.

        Parameters
        ----------
        attr : str or ndarray, shape ``(N, ...)``
            Control-point attribute to aggregate. If a string, the field is looked
            up on this domain; if an array, it must have length
            ``N == len(self)``. The trailing item shape (``...``) is preserved.
        splines : Spline
            Spline domain providing ``loop_start`` and ``loop_total`` arrays.

        Returns
        -------
        ndarray, shape ``(len(splines), ...)``
            Spline-wise averaged attribute. The trailing item shape is preserved and
            the dtype follows the input attribute.

        Raises
        ------
        AttributeError
            If `attr` is scalar or its first dimension does not match ``len(self)``
            (raised by
            [`_check_attribute_to_compute`](npblender._check_attribute_to_compute)).
        IndexError
            If ``splines.loop_start``/``loop_total`` describe ranges outside
            ``[0, len(self))``.
        TypeError
            If the attribute dtype cannot be averaged (e.g., integer arrays will
            fail on in-place division).

        Notes
        -----
        This routine computes an **unweighted arithmetic mean** of the attribute
        over each spline's control points.

        Examples
        --------
        ```python
        # Average tilt per spline from control points
        mean_tilt = cpoints.compute_attribute_on_splines("tilt", splines)

        # Average a custom per-control-point float attribute
        mean_w = cpoints.compute_attribute_on_splines(weights, splines)
        ```
        """

        @njit(cache=True)
        def _to_splines(loop_start, loop_total, source, res):

            nsplines = loop_start.shape[0]
            for ispline in range(nsplines):
                start = loop_start[ispline]
                total = loop_total[ispline]
                for ipoint in range(start, start + total):
                    res[ispline] += source[ipoint]
                inv = 1.0 / total
                res[ispline] *= inv

            return res        

        attr, item_shape = self._check_attribute_to_compute(attr)
        res = np.zeros((len(splines),) + item_shape, dtype=attr.dtype)
        return _to_splines(splines.loop_start, splines.loop_total, attr, res)
    
    # ----------------------------------------------------------------------------------------------------
    # Transformation
    # ----------------------------------------------------------------------------------------------------

    def apply_scale(self, scale, pivot=None):
        """Apply per-axis scales to control points, and scale handles as well.

        First applies per-axis scaling to ``position`` via
        [`apply_scale`](npblender.PointDomain.apply_scale), then, if Bezier
        handles are present, scales ``handle_left`` and ``handle_right`` by the
        same `scale`.

        Parameters
        ----------
        scale : array-like of shape ``(..., 3)``
            Per-axis scale factors broadcastable to the domain size.
        pivot : array-like of shape ``(..., 3)`` or None, optional
            Pivot location(s) for point scaling. If `None`, scales are applied
            about the origin.

        Returns
        -------
        ControlPoint
            Self (for chaining).

        Raises
        ------
        ValueError
            If the operand cannot be aligned with the domain size (raised by
            [`_get_shape_for_operation`](npblender._get_shape_for_operation)).

        Examples
        --------
        ```python
        # Uniform scale per control point
        s = np.full((len(CP), 3), 0.5)
        CP.apply_scale(s)

        # Per-spline scale about per-spline pivot
        CP.apply_scale(scales, pivot=pivots)
        ```
        """
        super().apply_scale(scale, pivot=pivot)

        if "handle_left" in self.actual_names:
            if True:
                self._apply_scale(scale, attr_name='handle_left')
                self._apply_scale(scale, attr_name='handle_right')
            else:
                self.handle_left  *= scale
                self.handle_right *= scale

        return self

    def transform(self, transfo, pivot=None):
        """Apply a linear transform to control points; rotate handles consistently.

        Applies the transform to ``position`` via
        [`transform`](npblender.PointDomain.transform). If Bezier handles
        are present, applies **only the rotation part** to ``handle_left`` and
        ``handle_right`` (i.e., scales are not applied to handles). Broadcasting is
        resolved with

        Parameters
        ----------
        transfo : Transformation, Quaternion or Rotation
            The transformtion to apply
        pivot : array-like of shape ``(..., 3)`` or None, optional
            Pivot location(s). If `None`, transforms are applied about the origin.

        Returns
        -------
        ControlPoint
            Self (for chaining).

        Raises
        ------
        ValueError
            If the operand cannot be aligned with the domain size (raised by
            [`_get_shape_for_operation`](npblender._get_shape_for_operation)).
        TypeError
            If `transfo` does not support the ``@`` operator with vectors or lacks
            a usable ``rotation`` component for handle updates.

        Examples
        --------
        ```python
        # Per-point rotations (as 3x3 matrices)
        R = npblender.Rotation.from_euler("XYZ", np.random.uniform(0, 1, (len(CP), 3))).as_matrix()
        CP.transform(R)

        # Rotate per-spline around per-spline pivot
        CP.transform(R_chunks, pivot=pivots)
        ```
        """
        super().transform(transfo, pivot=pivot)

        if "handle_left" in self.actual_names:
            rot = transfo.rotation if isinstance(transfo, Transformation) else transfo
            if True:
                self._transform(rot, attr_name='handle_left')
                self._transform(rot, attr_name='handle_right')
            else:
                self.handle_left  = rot @ self.handle_left
                self.handle_right = rot @ self.handle_right

        return self
    
# ----------------------------------------------------------------------------------------------------
# Point (Geometries with Point as single domain : Cloud, Instances, Meshes)
# ----------------------------------------------------------------------------------------------------

class Point(PointDomain):
    """
    Point domain for cloud and instances.

    Adds optional rotation/scale fields and a convenient kinematics
    initializer, along with several spatial distribution helpers.

    Attributes
    ----------
    position : (N, 3) float
        Point positions.
    radius : (N,) float, optional
        Point radius (e.g., for particle/cloud visualization).
    model_index : (N,) int, optional
        Instance model index.
    euler : (N, 3) float, optional
        Euler angles (radians).
    quat : (N, 4) float, optional
        Quaternion rotation in (x, y, z, w) convention.
    scale : (N, 3) float, optional
        Per-point scale.

    Kinematics (created by ``init_kinematics()``)
    ---------------------------------------------
    speed, accel, force : (N, 3) float
        Linear kinematics fields.
    mass : (N,) float
        Mass per point.
    moment : (N,) float
        Moment of inertia proxy (placeholder).
    omega, torque : (N, 3) float
        Angular velocity and torque.
    age : (N,) float
        Age (seconds or user-defined units).
    locked : (N,) bool
        Lock flag.
    last_pos : (N, 3) float
        Previous position.
    viscosity : (N,) float
        Drag coefficient.

    Properties
    ----------
    has_rotation : bool
        True if either ``euler`` or ``quat`` is present.
    rotation : Rotation or Quaternion
        Combined rotation (quaternion ⊗ euler if both present).

    Methods (selection)
    -------------------
    init_kinematics()
        Add standard kinematics fields.
    line_dist(), arc_dist(), circle_dist(), rect_dist(), pie_dist(),
    disk_dist(), cylinder_dist(), sphere_dist(), dome_dist(),
    cube_dist(), ball_dist()
        Populate positions (and related attributes like tangent/normal)
        from common geometric distributions.
    speed_along(), disk_speed(), shake_speed()
        Generate velocity vectors according to distributions.

    Examples
    --------
    Create a ring of instances with tangents:

    ```python
    pts = Point().circle_dist(radius=2.0, count=128, seed=0)
    # tangents and angles are appended automatically
    ```

    Compose rotations and apply scale:

    ```python
    R = pts.rotation              # auto-composed from euler/quat
    pts.apply_scale(0.5)
    pts.transform(R)              # re-apply or chain with other transforms
    ```
    """

    def _declare_attributes(self):
        super()._declare_attributes()

        # Cloud
        self.new_float('radius', default=.05, optional=True, transfer=True)

        # Instance
        self.new_int('model_index', default=0, optional=True, transfer=False)

        # All
        self.new_vector('euler',       default = (0, 0, 0), optional=True)
        self.new_quaternion("quat",    optional=True)
        self.new_vector("scale",       optional=True, default=1, transfer=False)

    # ====================================================================================================
    # Kinematics attributes
    # ====================================================================================================

    def init_kinematics(self):
        """Create standard kinematics fields (linear & angular).

        Declares common per-point kinematics attributes for simulation or
        procedural animation. This method does **not** modify positions or
        velocities; it only ensures attributes exist with sensible defaults.

        Declared attributes
        -------------------
        speed : (N, 3) float
            Linear velocity.
        accel : (N, 3) float
            Linear acceleration.
        mass : (N,) float, optional
            Point mass (default 1.0).
        force : (N, 3) float
            Accumulated external force.
        moment : (N,) float, optional
            Rotational inertia proxy (scalar; simplification).
        omega : (N, 3) float, optional
            Angular velocity (radians/second).
        torque : (N, 3) float, optional
            Accumulated external torque.
        age : (N,) float, optional
            Age/time since spawn (seconds).
        locked : (N,) bool, optional
            Freeze flag for constraints/solvers.
        last_pos : (N, 3) float, optional
            Previous position (for finite differences).
        viscosity : (N,) float, optional
            Linear damping factor.

        Returns
        -------
        None
        """
        # Speed
        self.new_vector('speed',       default=(0, 0, 0))
        self.new_vector('accel',       default=(0, 0, 0))
        self.new_float('mass',         default = 1., optional=True)
        self.new_vector('force',       default=(0, 0, 0))

        # Rotation
        self.new_float('moment',       default = 1., optional=True) # Should be a 3x3 tensor !
        self.new_vector('omega',       default = (0, 0, 0), optional=True) # Angular velocity
        self.new_vector('torque',      default = (0, 0, 0), optional=True)

        # Miscellaneous
        self.new_float('age',          default = 0, optional=True)
        self.new_bool('locked',        default = False, optional=True)
        self.new_vector('last_pos',    default = (0, 0, 0), optional=True)
        self.new_float('viscosity',    default = .01, optional=True)

    # ====================================================================================================
    # Transformation
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Rotation
    # ----------------------------------------------------------------------------------------------------

    @property
    def has_rotation(self):
        """Whether this domain has an orientation field.

        Returns
        -------
        bool
            `True` if either ``euler`` or ``quat`` exists in the attribute set,
            `False` otherwise.

        See Also
        --------
        [`rotation`](npblender.Point.rotation) :
            Access the per-point rotation object.
        [`get_rotation`](npblender.Point.get_rotation) :
            Safe accessor that can return a default.
        """
        return "quat" in self.actual_names or "euler" in self.actual_names

    @property
    def rotation(self):
        """Per-point rotation object from Euler and/or quaternion.

        If both ``euler`` and ``quat`` exist, returns their composition
        (quaternion composed with Euler). If neither exists, returns an
        identity rotation with shape ``(len(position),)``.

        Returns
        -------
        Rotation
            The point rotation

        See Also
        --------
        [`has_rotation`](npblender.Point.has_rotation) :
            Check if a rotation is available.
        [`get_rotation`](npblender.Point.get_rotation) :
            Return a default when no rotation is stored.

        Examples
        --------
        ```python
        R = P.rotation           # rotation object (vectorized)
        eul = R.as_euler()       # (N, 3)
        quat = R.as_quaternion() # (N, 4) in (x, y, z, w)
        ```
        """
        if "euler" in self.actual_names:
            rot = Rotation.from_euler(self.euler)
        else:
            rot = None

        if "quat" in self.actual_names:
            quat = Quaternion(self.quat)
        else:
            quat = None

        if rot is None:
            if quat is None:
                return Quaternion.identity(shape=(len(self.position),))
            else:
                return quat
        else:
            if quat is None:
                return rot
            else:
                return quat @ rot
            
    def get_rotation(self, default=None):
        """Return the per-point rotation or a default.

        Parameters
        ----------
        default : Any or None, optional
            Value to return if no rotation field is present. If `None`, the method
            returns `None` when no rotation is available.

        Returns
        -------
        Any or None
            [`rotation`](npblender.Point.rotation) if available; otherwise
            `default`.

        See Also
        --------
        [`has_rotation`](npblender.Point.has_rotation)
        [`rotation`](npblender.Point.rotation)
        """
        if self.has_rotation:
            return self.rotation
        else:
            return default

    # ----------------------------------------------------------------------------------------------------
    # Scale
    # ----------------------------------------------------------------------------------------------------
            
    def apply_scale(self, scale, pivot=None):
        """Apply per-axis scales to positions and multiply the `scale` attribute.

        Calls the base point scaling
        ([`apply_scale`](npblender.PointDomain.apply_scale)) to update
        ``position`` (optionally about `pivot`), then multiplies the per-point
        ``scale`` attribute when present.

        Parameters
        ----------
        scale : array-like of shape ``(..., 3)``
            Per-axis scale factors broadcastable to the domain size.
        pivot : array-like of shape ``(..., 3)`` or None, optional
            Pivot location(s). If `None`, scaling is applied about the origin.

        Returns
        -------
        Point
            Self (for chaining).

        Raises
        ------
        ValueError
            If broadcasting cannot align inputs with the domain (raised by
            [`_get_shape_for_operation`](npblender._get_shape_for_operation)).
        """
        super().apply_scale(scale, pivot=pivot)

        if "scale" in self.actual_names:
            if True:
                super()._apply_scale(scale, attr_name='scale')
            else:
                self.scale  *= scale

        return self

    # ----------------------------------------------------------------------------------------------------
    # Transformation
    # ----------------------------------------------------------------------------------------------------
            
    def transform(self, transfo, pivot=None):
        """Transform positions and compose stored orientation.

        First applies the linear transform to ``position`` via the base implementation
        ([`transform`](npblender.PointDomain.transform)). Then, if a rotation
        field exists (``euler`` or ``quat``), composes it with `transfo`’s rotation
        component and writes back to the same representation.

        Parameters
        ----------
        transfo : array-like or object
            Transform(s) broadcastable to the domain size. Typical shapes include
            ``(..., 3, 3)``; project transform types are also supported if they
            implement ``@`` with vectors and expose a `.rotation` component.
        pivot : array-like of shape ``(..., 3)`` or None, optional
            Pivot location(s). If `None`, transforms are applied about the origin.

        Returns
        -------
        Point
            Self (for chaining).

        Raises
        ------
        ValueError
            If broadcasting cannot align inputs with the domain (raised by
            [`_get_shape_for_operation`](npblender._get_shape_for_operation)).
        TypeError
            If `transfo` does not support the ``@`` operator with vectors or lacks
            a usable rotation component.
        """
        super().transform(transfo, pivot=pivot)

        if self.has_rotation:
            use_euler = "euler" in self.actual_names
            if use_euler:
                r = Rotation.from_euler(self.euler)
            else:
                r = Quaternion(self.quat)

            if True:
                shape, op_shape = self._get_shape_for_operation(transfo.shape, title="Scale")
                r = self[attr_name].reshape(shape)
                new_r = transfo.reshape(op_shape) @ r.reshape(shape[:-1])

                r = new_r.reshape(r.shape)
            else:
                rot = transfo.rotation if isinstance(transfo, Transformation) else transfo
                r = rot @ r

            if use_euler:
                self.euler = r.as_euler()
            else:
                self.quat = r.as_quaternion()


        return self

    # ====================================================================================================
    # Position distributions
    # ====================================================================================================

    def line_dist(self, point0=(-1, -1, -1), point1=(1, 1, 1), count=10, density=None, seed=None):
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.new_vector('tangent')
        d = distribs.line_dist(point0, point1, count, density, seed)
        self.append(position=d['points'], tangent=d['tangents'])
        return self

    def arc_dist(
        self,
        radius=1.0,
        scale=None,
        center=(0, 0, 0),
        arc_center=0.0,
        arc_angle=np.pi/2,
        use_vonmises=False,
        count=10,
        density=None,
        seed=None,
    ):
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.new_vector('tangent')
        self.new_float('angle')
        d = distribs.arc_dist(
            radius, scale, center, arc_center, arc_angle, use_vonmises, count, density, seed
        )
        self.append(position=d['points'], tangent=d['tangents'], angle=d['angles'])
        return self

    def circle_dist(
        self,
        radius=1.0,
        scale=None,
        center=(0, 0, 0),
        count=10,
        density=None,
        seed=None,
    ):
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.new_vector('tangent')
        self.new_float('angle')
        d = distribs.circle_dist(radius, scale, center, count, density, seed)
        self.append(position=d['points'], tangent=d['tangents'], angle=d['angles'])
        return self

    def rect_dist(self, a=1, b=1, center=(0, 0, 0), count=10, density=None, seed=None):
        d = distribs.rect_dist(a, b, center, count, density, seed)
        self.append(position=d['points'])
        return self


    def pie_dist(
        self,
        radius=1,
        outer_radius=None,
        center=None,
        normal=None,
        pie_center=0.,
        pie_angle=np.pi/2,
        use_vonmises=False,
        count=10,
        density=None,
        seed=None
    ):
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.new_vector('tangent')
        self.new_float('angle')
        d = distribs.pie_dist(
            radius, outer_radius, center, normal, pie_center, pie_angle, use_vonmises, count, density, seed
        )
        self.append(position=d['points'], tangent=d['tangents'], angle=d['angles'])
        return self

    def disk_dist(self, radius=1, outer_radius=None, center=None, normal=None, count=10, density=None, seed=None):
        self.new_vector('tangent')
        self.new_float('angle')
        d = distribs.disk_dist(
            radius, outer_radius, center, normal, count, density, seed
        )
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.append(position=d['points'], tangent=d['tangents'], angle=d['angles'])
        return self

    def cylinder_dist(
        self,
        radius=1.0,
        scale=None,
        height=1.0,
        center=(0, 0, 0),
        arc_center=0.0,
        arc_angle=2*np.pi,
        use_vonmises=False,
        count=10,
        density=None,
        seed=None,
    ):
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.new_vector('normal')
        self.new_vector('tangent')
        self.new_float('angle')
        d = distribs.cylinder_dist(
            radius, scale, height, center, arc_center, arc_angle, use_vonmises, count, density, seed
        )
        self.append(position=d['points'], normal=d['normals'], tangent=d['tangents'], angle=d['angles'])
        return self


    def sphere_dist(
        self,
        radius=1.0,
        scale=None,
        center=(0, 0, 0),
        count=10,
        density=None,
        seed=None
    ):
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.new_vector('normal')
        self.new_float('theta')
        self.new_float('phi')
        d = distribs.sphere_dist(radius, scale, center, count, density, seed)
        self.append(position=d['points'], normal=d['normals'], theta=d['thetas'], phi=d['phis'])
        return self

    def dome_dist(
        self,
        radius=1.0,
        scale=None,
        axis=(0, 0, 1),
        angle=np.pi / 2,
        use_vonmises=False,
        center=(0, 0, 0),
        count=10,
        density=None,
        seed=None
    ):
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.new_vector('normal')
        d = distribs.dome_dist(
            radius, scale, axis, angle, use_vonmises, center, count, density, seed
        )
        self.append(position=d['points'], normal=d['normals'])
        return self

    def cube_dist(self, size=1, center=(0, 0, 0), count=10, density=None, seed=None):
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.append(position = distribs.cube_dist(size, center, count, density, seed)['points'])
        return self

    def ball_dist(
        self,
        radius=1.0,
        axis=(0, 0, 1),
        angle=np.pi,
        use_vonmises=False,
        center=(0, 0, 0),
        count=10,
        density=None,
        scale=None,
        seed=None,
        **kwargs
    ):
        """ see distributions in [maths.distribs][npblender.maths.distribs]"""
        self.new_vector('normal')        
        d = distribs.ball_dist(
            radius, axis, angle, use_vonmises, center, count, density, scale, seed, **kwargs)
        self.append(position=d['points'], normal=d['normals'])
        return self
    
    # ====================================================================================================
    # Speed distributions
    # ====================================================================================================

    def speed_along(self, speed=1, direction=(0, 0, 1), angle=np.pi/2, scale=None, use_vonmises=False, seed=None):
        """Sample velocity vectors within a cone around a direction.

        Returns random 3D vectors with magnitudes given by `speed` and directions
        sampled within a cone of half-angle `angle` around `direction`. Useful to
        initialize per-point velocities.

        Parameters
        ----------
        speed : float or (N,) or (N, 3)
            Target magnitudes (broadcastable).
        direction : (3,) or (N, 3), default=(0, 0, 1)
            Cone axis per point (broadcastable).
        angle : float, default=pi/2
            Cone half-angle (radians).
        scale : (3,) float or None, optional
            Optional anisotropic scaling of the cone.
        use_vonmises : bool, default=False
            If `True`, directions follow a Von Mises–Fisher-like distribution.
        seed : int or None, optional

        Returns
        -------
        ndarray of shape ``(N, 3)``
            Sampled velocity vectors.

        Raises
        ------
        ValueError
            If inputs cannot be broadcast/aligned (propagated from the backend).
        """
        return distribs.dome_dist(
            radius = self.get(speed),
            scale = scale,
            axis = self.get(direction),
            angle = angle,
            use_vonmises = use_vonmises,
            count = len(self),
            seed = seed,
            )['points']
    
    def disk_speed(self, speed=1, max_speed=None, normal=None, seed=None):
        """Sample velocity vectors in a disk (2D radial distribution).

        Samples vectors lying in a plane (given by `normal`) with lengths in
        ``[0, max_speed]`` (or fixed `speed` if `max_speed` is `None`).

        Parameters
        ----------
        speed : float
            Base radius or fixed magnitude when `max_speed` is `None`.
        max_speed : float or None, optional
            Maximum radius; if provided, lengths are sampled in ``[0, max_speed]``.
        normal : (3,) float or None, optional
            Plane normal. If `None`, uses +Z.
        seed : int or None, optional

        Returns
        -------
        ndarray of shape ``(N, 3)``
            Sampled velocity vectors.

        Raises
        ------
        ValueError
            If inputs are invalid (propagated from the backend).
        """
        return distribs.disk_dist(
            radius = speed,
            outer_radius = max_speed,
            normal=normal,
            count=len(self),
            seed=seed)['points']

    def shake_speed(self, speed, scale=None, length_only=False, seed=None):
        """Add random jitter to existing velocity vectors.

        Parameters
        ----------
        speed : str or ndarray
            A field name or an array of velocity vectors to perturb.
        scale : float or (3,) or None, optional
            Jitter amount (broadcastable).
        length_only : bool, default=False
            If `True`, only magnitudes are perturbed; directions are preserved.
        seed : int or None, optional

        Returns
        -------
        ndarray of shape ``(N, 3)``
            Jittered velocity vectors.

        Raises
        ------
        ValueError
            If inputs are invalid (propagated from the backend).

        See Also
        --------
        [`speed_along`](npblender.Point.speed_along)
        [`disk_speed`](npblender.Point.disk_speed)
        """
        speed = self.get(speed)
        return distribs.shake_vectors(
            speed = self.get(speed),
            scale = scale, 
            length_only = length_only,
            seed = seed)


# ====================================================================================================
# Corner Domain
# ====================================================================================================

class Corner(Domain):
    """
    Mesh corner (loop) domain.

    Stores, for each corner of each face, the index of the associated
    vertex and optional per-corner attributes (e.g., UVs).

    Attributes
    ----------
    vertex_index : (M,) int
        Index of the referenced vertex in the point (vertex) array.
    UVMap : (M, 2) float, optional
        UV coordinates (you may create additional maps via :meth:`new_uvmap`).

    Methods
    -------
    check(count, halt=True)
        Validate that all vertex indices are < ``count``.
    new_uvmap(name, value=None)
        Create and optionally initialize a new UV map attribute.
    compute_attribute_on_points(attr, points)
        Average a per-corner attribute back to points (vertices).

    Examples
    --------
    Create a second UV map:

    ```python
    corners.new_uvmap("UV2", value=(0.0, 0.0))
    ```

    Accumulate per-corner shading to vertices:

    ```python
    v_attr = corners.compute_attribute_on_points("illum", points)
    ```
    """

    domain_name = 'CORNER'

    def _declare_attributes(self):
        self.new_int('vertex_index', transfer=False, transdom=False)
        self.new_vector2('UVMap', (0, 0), optional=True, transfer=True, transdom=False)

    # ----------------------------------------------------------------------------------------------------
    # Check
    # ----------------------------------------------------------------------------------------------------

    def check(self, count, halt=True):
        """Validate corner indices against a vertex count.

        Verifies that all entries of ``vertex_index`` reference valid vertices
        (i.e., are **strictly less than** ``count``). When invalid indices are
        detected, either raises or logs an error depending on ``halt``.

        Parameters
        ----------
        count : int
            Number of vertices in the referenced point/vertex domain.
        halt : bool, default=True
            If `True`, raise on failure; otherwise print a message and return `False`.

        Returns
        -------
        bool
            `True` if the check passes or the domain is empty; `False` only when
            invalid and ``halt`` is `False`.

        Raises
        ------
        RuntimeError
            If invalid indices are found and ``halt`` is `True`.

        Examples
        --------
        ```python
        ok = corners.check(count=len(vertices), halt=False)
        if not ok:
            # fix topology or filter invalid corners
            ...
        ```
        """
        if not len(self):
            return True
        
        if np.max(self.vertex_index) > count:
            msg = f"Corner check fail: {np.max(self.vertex_index)=}, {count=}"
            if halt:
                raise RuntimeError(msg)
            else:
                print(msg)
                return False
        return True

    # ====================================================================================================
    # Methods
    # ====================================================================================================

    def new_uvmap(self, name, value=None):
        """Create (and optionally initialize) a per-corner UV map.

        Declares a new 2D UV attribute with the given ``name``. If ``value`` is
        provided, assigns it to the whole array (broadcast rules apply).

        Parameters
        ----------
        name : str
            Attribute name for the UV map.
        value : array-like of shape ``(2,)`` or ``(M, 2)`` or None, optional
            Initial UV values; broadcastable to the corner count.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``value`` cannot be broadcast to shape ``(M, 2)``.
        TypeError
            If ``value`` has an incompatible dtype.

        See Also
        --------
        [`new_vector2`](npblender.new_vector2) :
            Registers a 2D vector attribute.
        """
        self.new_vector2(name)
        if value is not None:
            self[name] = value

    # ====================================================================================================
    # Compute attribute on points
    # ====================================================================================================

    def compute_attribute_on_points(self, attr, points):
        """Average a per-corner attribute back to points (vertices).

        For each vertex, computes the mean of the source attribute over all
        incident corners (as defined by ``vertex_index``). The attribute is
        validated with
        [`_check_attribute_to_compute`](npblender._check_attribute_to_compute).

        Parameters
        ----------
        attr : str or ndarray, shape ``(M, ...)``
            Corner attribute to average. If a string, the field is looked up on this
            domain; if an array, it must have length ``M == len(self)``. The trailing
            item shape (``...``) is preserved.
        points : Vertex
            Vertex domain used to size the result (``len(points)``).

        Returns
        -------
        ndarray, shape ``(len(points), ...)``
            Per-vertex averaged attribute. The trailing item shape is preserved and
            the dtype follows the input attribute.

        Raises
        ------
        AttributeError
            If `attr` is scalar or its first dimension does not match ``len(self)``
            (raised by
            [`_check_attribute_to_compute`](npblender._check_attribute_to_compute)).
        IndexError
            If ``vertex_index`` contains indices outside ``[0, len(points))``.
        TypeError
            If the attribute dtype cannot be averaged (e.g., non-numeric types).

        Notes
        -----
        This routine computes an **unweighted arithmetic mean** per vertex over all
        incident corners.

        Examples
        --------
        ```python
        # Accumulate per-corner illumination to vertices
        v_illum = corners.compute_attribute_on_points("illum", vertices)

        # Average custom corner vectors per vertex
        v_vec = corners.compute_attribute_on_points(corner_vecs, vertices)
        ```
        """

        @njit(cache=True)
        def _to_points(vertex_index, source, res):
            npoints = res.shape[0]
            ncorners = vertex_index.shape[0]

            count = np.zeros(npoints, dtype=np.int32)
            for icorner in range(ncorners):
                s = source[icorner]

                ipoint = vertex_index[icorner]
                res[ipoint] += s
                count[ipoint] += 1

            trailing = 1
            for d in range(1, res.ndim):
                trailing *= res.shape[d]

            R2 = res.reshape((npoints, trailing))
            for ipoint in range(npoints):
                c = count[ipoint]
                if c > 0:
                    inv = 1.0 / c
                    for j in range(trailing):
                        R2[ipoint, j] *= inv

            return res  
        
        attr, item_shape = self._check_attribute_to_compute(attr)
        res = np.zeros((len(points),) + item_shape, dtype=attr.dtype)
        return _to_points(self.vertex_index, attr, res)                


# ====================================================================================================
# Root for Face and Spline
# ====================================================================================================

class FaceSplineDomain(Domain):

    def _declare_attributes(self):
        self.new_int('loop_start', transfer=False, transdom=False)
        self.new_int('loop_total', transfer=False, transdom=False)

        self.new_int('material_index', optional=True, transdom=False)

    # ----------------------------------------------------------------------------------------------------
    # Check
    # ----------------------------------------------------------------------------------------------------

    def check(self, count, halt=True):
        """Validate loop counters against a reference element count.

        Verifies that the sum of ``loop_total`` equals ``count`` (i.e., the total
        number of referenced elements such as corners or control points).

        Parameters
        ----------
        count : int
            Expected total number of elements referenced by all items.
        halt : bool, default=True
            If `True`, raise on failure; otherwise print a message and return `False`.

        Returns
        -------
        bool
            `True` if the check passes or the domain is empty; `False` only when
            invalid and ``halt`` is `False`.

        Raises
        ------
        RuntimeError
            If ``sum(loop_total) != count`` and ``halt`` is `True`.

        Examples
        --------
        ```python
        ok = fs.check(count=len(corners), halt=False)
        if not ok:
            # fix topology or sizes, then recompute
            fs.update_loop_start()
        ```
        """
        if not len(self):
            return True
        
        if np.sum(self.loop_total) != count:
            msg = f"Face check fail: {np.sum(self.loop_total)=}, {count=}"
            if halt:
                raise RuntimeError(msg)
            else:
                print(msg)
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
        """Recompute ``loop_start`` from ``loop_total`` (cumulative layout).

        Sets ``loop_start`` to a left-rolled cumulative sum of ``loop_total``,
        so item *i* starts right after the end of item *i-1*.

        Returns
        -------
        FaceSplineDomain
            Self (for chaining).

        Notes
        -----
        Call this after deletions / resizes to keep indices consistent.

        See Also
        --------
        [`compute_loop_start`](npblender.FaceSplineDomain.compute_loop_start) :
            Compute offsets for *new* items to be appended.
        """
        if len(self):
            a = np.roll(np.cumsum(self.loop_total), 1)
            a[0] = 0
            self.loop_start = a
        return self

    @property
    def next_loop_start(self):
        """Offset to use for the next appended item.

        Returns
        -------
        int
            ``loop_start[-1] + loop_total[-1]`` if the domain is non-empty,
            otherwise ``0``.

        See Also
        --------
        [`compute_loop_start`](npblender.FaceSplineDomain.compute_loop_start)
        """
        if len(self):
            return self.loop_start[-1] + self.loop_total[-1]
        else:
            return 0

    def compute_loop_start(self, loop_total=None):
        """Compute offsets for one or many new items to append.

        Parameters
        ----------
        loop_total : int or array-like of int or None, optional
            Sizes of the items to add. If `None`, returns `None`. If a scalar,
            returns the single offset. If 1D array-like, returns one offset per size.

        Returns
        -------
        int or ndarray or None
            Offsets starting from
            [`next_loop_start`](npblender.FaceSplineDomain.next_loop_start),
            shaped like `loop_total`.

        Examples
        --------
        ```python
        # Prepare offsets for three faces of sizes 4, 5, 4
        starts = fs.compute_loop_start([4, 5, 4])
        fs.append(loop_start=starts, loop_total=[4, 5, 4])
        ```
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
        """Append new items given their sizes.

        If ``loop_start`` is not provided in ``fields``, computes it from `sizes`
        using [`compute_loop_start`](npblender.FaceSplineDomain.compute_loop_start).

        Parameters
        ----------
        sizes : array-like of int or None
            Number of corners/control-points for each new item.
        **fields
            Additional per-item attributes to append (e.g., ``material_index``).

        Returns
        -------
        Any
            The value returned by ``append(...)`` (implementation-defined, often
            the indices/slice of appended items).

        Notes
        -----
        Passing ``sizes=None`` is a no-op and returns an empty list.
        """
        if sizes is None:
            return []
        
        if 'loop_start' in fields:
            res = self.append(loop_total=sizes, **fields)
        else:
            loop_start = self.compute_loop_start(sizes)
            res = self.append(loop_start=loop_start, loop_total=sizes, **fields)

        return res

    def get_corner_indices(self):
        """Return the contiguous range of corner/control-point indices.

        For each item, expands its ``[loop_start, loop_start + loop_total)`` range
        and concatenates the result for all items.

        Returns
        -------
        ndarray of shape ``(sum(loop_total),)``
            Absolute indices into the corner/control-point array.

        Notes
        -----
        A fast Numba kernel is used for vectorized cases; the scalar case is
        handled directly.
        """
        if self.is_scalar:
            return np.arange(self.loop_total) + self.loop_start
        
        # njit function
        return get_corner_indices(self.loop_start, self.loop_total)
    
    # ----------------------------------------------------------------------------------------------------
    # Delete faces
    # ----------------------------------------------------------------------------------------------------

    def delete(self, selection):
        """Delete selected items and maintain consistent offsets.

        After deleting items via ``super().delete(selection)``, recomputes
        ``loop_start`` with
        [`update_loop_start`](npblender.FaceSplineDomain.update_loop_start).

        Parameters
        ----------
        selection : Any
            Boolean mask, integer index, slice, or array of indices.

        Returns
        -------
        None

        See Also
        --------
        [`update_loop_start`](npblender.FaceSplineDomain.update_loop_start)
        """
        super().delete(selection)
        self.update_loop_start()

    # ====================================================================================================
    # Sort the item per size
    # ====================================================================================================

    def per_size(self):
        """Group items by their ``loop_total`` (polygon/control-point count).

        Returns a dictionary keyed by size (``3``, ``4``, …) where each entry
        contains:
        - ``'start'`` : array of ``loop_start`` values for items of that size.
        - ``'indices'`` : array mapping entry order back to item indices.

        Returns
        -------
        dict[int, dict[str, ndarray]]
            Grouped start offsets and reverse indices for each size present.

        Notes
        -----
        Uses a Numba kernel to bucketize items efficiently.
        """
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
        """Map each corner/control-point index back to its owning item.

        Returns
        -------
        ndarray of shape ``(sum(loop_total),)``
            For index ``k`` in the flattened corner/control-point array, the
            value is the face/spline index that owns ``k``.
        """
        return get_face_reverse_indices(self.loop_start, self.loop_total)    


# ====================================================================================================
# Face Domain
# ====================================================================================================

class Face(FaceSplineDomain):
    """
    Mesh face (polygon) domain.

    Inherits loop bookkeeping from :class:`FaceSplineDomain` and provides
    face-level operations: area/normal/centroid computation, edge
    extraction, attribute transfers, and topology-aware deletions.

    Attributes
    ----------
    loop_start : (F,) int
        Starting corner index of each face.
    loop_total : (F,) int
        Number of corners per face.
    material_index : (F,) int, optional
        Material index per face.
    sharp_face : (F,) bool, optional
        Sharp shading flag.

    Methods
    -------
    delete_loops(selection, corners)
        Delete faces and their incident corners; returns removed vertex indices.
    get_face_edges(corners)
        Edge list per face as ordered vertex-index pairs.
    get_edges(corners)
        Unique undirected edges present in the mesh faces.
    get_position(corners, points)
        Face centroids by averaging incident vertex positions.
    area_vectors(corners, points)
        Area-weighted normal vectors (unnormalized).
    area(corners, points)
        Face areas.
    normal(corners, points)
        Unit normals.
    sequences(corners)
        Per-face sequences of vertex indices.
    compute_attribute_on_points(attr, corners, points)
        Average a face attribute back to points.

    Examples
    --------
    Compute normals and areas:

    ```python
    n = faces.normal(corners, points)
    A = faces.area(corners, points)
    ```

    Extract unique edges:

    ```python
    edges = faces.get_edges(corners)
    ```

    > ***Note:*** Area vectors are computed by triangulating polygons and
    summing triangle cross products.

    > ***Warning:*** After deleting faces via :meth:`delete_loops`, update
    dependent domains accordingly (e.g., rebuild edges if needed).
    """

    domain_name = 'FACE'

    def _declare_attributes(self):
        super()._declare_attributes()

        self.new_bool('sharp_face', optional=True, transdom=False)

    # ====================================================================================================
    # Delete loops
    # ====================================================================================================

    def delete_loops(self, selection, corners):
        """
        Delete faces and their incident corners.

        Removes the selected faces and deletes the corresponding corner loops
        from ``corners``. Returns the vertex indices (as stored on corners)
        that were removed with those loops.

        Parameters
        ----------
        selection : Any
            Face selection accepted by the domain (boolean mask, indices,
            slice, etc.).
        corners : Corner
            Corner domain holding at least the per-corner ``vertex_index`` array.

        Returns
        -------
        numpy.ndarray of int
            The (possibly non-unique) vertex indices referenced by the
            deleted corners. If you need the unique set of affected vertices,
            apply ``np.unique`` on the result.

        Notes
        -----
        - Internally calls [`delete`](npblender.delete) on the face
        domain, which updates loop bookkeeping (see
        [`update_loop_start`](npblender.update_loop_start)).
        - Corners corresponding to the deleted faces are also removed via
        ``corners.delete(...)``.

        Raises
        ------
        IndexError
            If ``selection`` is out of bounds or if corner indices are inconsistent
            with the provided ``corners`` domain.
        """
        corner_indices = self[selection].get_corner_indices()
        self.delete(selection)
        vert_indices = np.array(corners.vertex_index[corner_indices])
        corners.delete(corner_indices)

        return vert_indices

    # ====================================================================================================
    # Get edges
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Get all the edges as ordered couples
    # ----------------------------------------------------------------------------------------------------

    def get_edges(self, corners):
        """
        Return the unique undirected edges present in the faces.

        Parameters
        ----------
        corners : Corner
            Corner domain providing the ``vertex_index`` array.

        Returns
        -------
        numpy.ndarray of shape (E, 2), dtype=int
            Unique vertex-index pairs for all edges.
        """
        edges = np.zeros((len(corners), 2), bint)
        edges[:, 0] = corners.vertex_index
        edges[:-1, 1] = corners.vertex_index[1:]
        edges[self.loop_start + self.loop_total - 1, 1] = corners.vertex_index[self.loop_start]
        
        edges = np.sort(edges, axis=-1)
        edges = np.unique(edges, axis=0)
        return edges
    
    # ====================================================================================================
    # Get position
    # ====================================================================================================
    
    def position(self, corners, points):
        """
        Face centroids (mean of corner positions).

        Computes the arithmetic mean of corner positions per face.

        Parameters
        ----------
        corners : Corner
        points : Point

        Returns
        -------
        numpy.ndarray of shape (F, 3), dtype=float
            Centroid of each face.
        """
        return points.compute_attribute_on_faces("position", corners, self)
    
    # ====================================================================================================
    # Surfaces and normals
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Surface is computed by cross products of triangle. This also gives
    # the normal to the face. The vector normal the length of which is the surface
    # is called the "area vector".
    # ----------------------------------------------------------------------------------------------------

    def _area_vectors(self, corners, points):
        """
        Compute area-weighted normal vectors for each face.

        Each polygonal face is triangulated fan-wise around its first corner, and the
        cross-product of each triangle’s edges is accumulated. The resulting vector has
        a magnitude equal to twice the polygon area and a direction given by the
        face’s winding order.

        Parameters
        ----------
        corners : Corner
            Corner domain providing the per-corner ``vertex_index``.
        points : Point
            Point (vertex) domain providing ``position`` of shape ``(P, 3)``.

        Returns
        -------
        numpy.ndarray of shape ``(F, 3)`` and dtype float
            Area-weighted normal vector for each face.

        Notes
        -----
        - The vector magnitude is proportional to the face area, and its direction
        matches the orientation of the polygon.
        - Faces with fewer than 3 vertices yield a zero vector.

        See Also
        --------
        [`area`](npblender.Face.area)
        [`normal`](npblender.Face.normal)
        [`get_surface`](npblender.Face.get_surface)
        """

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

    # ----------------------------------------------------------------------------------------------------
    # area : norm of the area vectors
    # ----------------------------------------------------------------------------------------------------

    def area(self, corners, points):
        """
        Compute face areas.

        Returns the scalar area of each polygonal face by taking half the Euclidean
        norm of its area-weighted normal vector (see [`area_vectors`](npblender.Face.area_vectors)).

        Parameters
        ----------
        corners : Corner
            Corner domain providing the per-corner ``vertex_index``.
        points : Point
            Point (vertex) domain providing ``position`` of shape ``(P, 3)``.

        Returns
        -------
        numpy.ndarray of shape ``(F,)`` and dtype float
            Area per face.

        Notes
        -----
        - Internally, faces are triangulated and triangle cross-products are summed
        to form area vectors; the area is half the vector norm. See
        [`area_vectors`](npblender.Face.area_vectors) for details.
        - Degenerate faces (zero area) produce zeros here.

        See Also
        --------
        [`area_vectors`](npblender.Face.area_vectors)
        [`normal`](npblender.Face.normal)
        [`get_surface`](npblender.Face.get_surface)
        """
        return np.linalg.norm(self.area_vectors(corners, points), axis=-1)/2

    # ----------------------------------------------------------------------------------------------------
    # Normals : normalized surface vectors
    # ----------------------------------------------------------------------------------------------------

    def normal(self, corners, points):
        """
        Compute per-face unit normals.

        Returns a normalized area vector for each face (see
        [`area_vectors`](npblender.Face.area_vectors)). The direction follows the
        winding of the face’s corners.

        Parameters
        ----------
        corners : Corner
            Corner domain providing the per-corner ``vertex_index``.
        points : Point
            Point (vertex) domain providing ``position`` of shape ``(P, 3)``.

        Returns
        -------
        numpy.ndarray of shape ``(F, 3)`` and dtype float
            Unit normal per face.

        Notes
        -----
        - Area vectors are obtained by triangulating polygons and summing triangle
        cross-products before normalization.
        - Degenerate faces (zero area) yield undefined normals (NaNs or inf). You may
        sanitize with ``np.nan_to_num`` or mask faces where the area is zero.

        See Also
        --------
        [`area_vectors`](npblender.Face.area_vectors)
        [`area`](npblender.Face.area)
        [`get_surface`](npblender.Face.get_surface)
        """
        sv = self.area_vectors(corners, points)
        return sv / np.linalg.norm(sv, axis=-1)[:, None]

    # ---------------------------------------------------------------------------
    # Centers of the faces

    def position_DEPRECATED(self, corners, points):
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
        """
        Convenience bundle of per-face surface data.

        Returns a dictionary containing:
        - ``normals`` : (F, 3) float — unit normals,
        - ``areas``   : (F,) float — face areas,
        - ``sizes``   : (F,) int — number of corners per face (``loop_total``),
        - ``verts``   : (P, 3) float — reference to point positions array.

        Parameters
        ----------
        corners : Corner
        points : Point

        Returns
        -------
        dict
            Mapping with keys ``normals``, ``areas``, ``sizes``, ``verts``.
        """
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
        """
        Vertex-index sequences per face.

        Returns a Python list where each item is the ordered sequence of
        vertex indices for the corresponding face.

        Parameters
        ----------
        corners : Corner

        Returns
        -------
        list[list[int]]
            Vertex index sequence for each face.
        """
        inds = list(corners.vertex_index)
        return [inds[lstart:lstart+ltotal] for (lstart, ltotal) in zip(self.loop_start, self.loop_total)]     

    # ====================================================================================================
    # Compute attribute on points
    # ====================================================================================================

    def compute_attribute_on_points(self, attr, corners, points):
        """
        Average a per-face attribute back to points (vertices).

        For each face attribute value, accumulates it to all incident points
        (via corners) and divides by the number of incident contributions per
        point.

        Parameters
        ----------
        attr : str or numpy.ndarray
            Name of the face attribute to transfer, or an explicit array of
            shape ``(F, ...)``.
        corners : Corner
            Corner domain (provides ``vertex_index``).
        points : Point
            Target point domain (length defines the number of output points).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(points), ...)`` with the averaged values.

        Raises
        ------
        AttributeError
            If ``attr`` is a string and no such face attribute exists.
        IndexError
            If corner vertex indices fall outside ``[0, len(points))``.
        """

        @njit(cache=True)
        def _to_points(loop_start, loop_total, vertex_index, source, res):
            npoints = res.shape[0]
            nfaces = loop_start.shape[0]

            count = np.zeros(npoints, dtype=np.int32)
            for iface in range(nfaces):
                start = loop_start[iface]
                total = loop_total[iface]
                s = source[iface]
                for icorner in range(start, start + total):
                    ipoint = vertex_index[icorner]
                    res[ipoint] += s
                    count[ipoint] += 1

            trailing = 1
            for d in range(1, res.ndim):
                trailing *= res.shape[d]

            R2 = res.reshape((npoints, trailing))
            for ipoint in range(npoints):
                c = count[ipoint]
                if c > 0:
                    inv = 1.0 / c
                    for j in range(trailing):
                        R2[ipoint, j] *= inv

            return res  
        
        attr, item_shape = self._check_attribute_to_compute(attr)
        res = np.zeros((len(points),) + item_shape, dtype=attr.dtype)
        return _to_points(self.loop_start, self.loop_total, corners.vertex_index, attr, res)                

# ====================================================================================================
# Edge Domain
# ====================================================================================================

class Edge(Domain):
    """
    Mesh edge domain.

    Represents undirected edges as pairs of vertex indices and provides
    utilities to validate, filter, and transfer attributes.

    Attributes
    ----------
    vertex0, vertex1 : (E,) int
        Endpoints of each edge.

    Properties
    ----------
    vertices : (E, 2) int
        View/setter exposing concatenated ``(vertex0, vertex1)``.

    Methods
    -------
    check(count, halt=True)
        Validate indices are < ``count`` and no degenerate edges exist.
    remove_face_edges(face_edges)
        Remove edges that belong to a set of face edges.
    compute_attribute_on_points(attr, points)
        Average edge attributes back to points (each endpoint contributes).

    Examples
    --------
    Remove edges already represented by faces:

    ```python
    edges.remove_face_edges(face_edges)
    ```

    Average per-edge scalar to points:

    ```python
    p_attr = edges.compute_attribute_on_points("heat", points)
    ```

    > ***Caution:*** ``check()`` raises if an endpoint index is out of range
    or if an edge uses the same vertex twice.
    """

    domain_name = 'EDGE'

    def _declare_attributes(self):
        self.new_int('vertex0', transfer=False, transdom=False)
        self.new_int('vertex1', transfer=False, transdom=False)

    @property
    def vertices(self):
        """
        Return per-edge vertex indices.

        Provides the connectivity of the edge domain as integer pairs of vertex
        indices. Each row corresponds to one edge, with the two endpoint indices
        given as an unordered pair.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray of shape ``(E, 2)`` and dtype int
            Array of vertex index pairs, one row per edge.

        Notes
        -----
        - The order of indices in each pair is not significant: ``(i, j)`` and
        ``(j, i)`` denote the same undirected edge.
        - This array can be used to look up endpoint positions from a point/vertex
        domain.

        Examples
        --------
        ```python
        # Get vertex index pairs for all edges
        edge_pairs = edges.vertices()

        # Use them to fetch endpoint coordinates
        coords = points.position[edge_pairs]
        ```
        """
        return np.stack((self.vertex0, self.vertex1), axis=-1)
    
    @vertices.setter
    def vertices(self, value):
        if np.shape(value) == (2,):
            self.vertex0 = value[0]
            self.vertex1 = value[1]
        else:
            self.vertex0 = value[:, 0]
            self.vertex1 = value[:, 1]

    def check(self, count, halt=True):
        """
        Validate edge endpoint indices and detect degeneracies.

        Verifies that all edge endpoints are strictly less than ``count`` and that
        no edge uses the same vertex twice.

        Parameters
        ----------
        count : int
            Number of vertices in the referenced point/vertex domain.
        halt : bool, default=True
            If `True`, raise on failure; otherwise print a message and return `False`.

        Returns
        -------
        bool
            `True` if the check passes or the domain is empty; `False` only when
            invalid and ``halt`` is `False`.

        Raises
        ------
        Exception
            If an endpoint index is out of range or if a degenerate edge is found
            and ``halt`` is `True`.

        Examples
        --------
        ```python
        ok = edges.check(count=len(points), halt=False)
        if not ok:
            # inspect or fix edges, then retry
            ...
        ```
        """
        if not len(self):
            return True
        
        if np.max(self.vertex0) >= count or np.max(self.vertex1) >= count:
            msg = f"Edge domain contains vertex index out of range: {np.max(self.vertex0)=}, {np.max(self.vertex0)=}, {count=}"
            if halt:
                raise Exception(msg)
            else:
                print(msg)
                return False
            
        if np.any(self.vertex0 == self.vertex1):
            msg = f"Some edges use the same vertex ! {np.sum(self.vertex0 == self.vertex1)}"
            if halt:
                raise Exception(msg)
            else:
                print(msg)
                return False

        return True


    # ====================================================================================================
    # Remove edges belonging to faces

    def remove_face_edges(self, face_edges):
        """
        Remove edges that belong to a given face-edge set.

        Compares the (undirected) edge list of this domain to ``face_edges`` and
        deletes any matching edges. Both inputs are normalized internally so that
        edge order does not matter.

        This method is typically used when loading a mesh from an object to keep only
        free edges.

        Parameters
        ----------
        face_edges : ndarray of shape ``(M, 2)`` and dtype int
            Vertex-index pairs representing edges built from faces (e.g., via
            [`get_face_edges`](npblender.Face.get_face_edges)).

        Returns
        -------
        None

        Notes
        -----
        - If ``face_edges`` is empty, the method returns immediately.
        - Matching is performed by viewing each pair as a structured item to allow
        fast set membership tests.
        """    

        if not len(face_edges):
            return
        
        vertices = np.sort(self.vertices, axis=1)
        face_edges = np.sort(face_edges, axis=1)

        u_verts = vertices.view([('', vertices.dtype)] * 2)
        u_faces = face_edges.view([('', face_edges.dtype)] * 2)

        mask = np.isin(u_verts, u_faces).ravel()

        self.delete(mask)

    # ====================================================================================================
    # Compute attribute on points
    # ====================================================================================================

    def compute_attribute_on_points(self, attr, points):
        """
        Average an edge attribute back to points (endpoints).

        For each edge attribute value, accumulates it to both endpoints and divides
        by the number of incident contributions per point.

        Parameters
        ----------
        attr : str or numpy.ndarray
            Name of the edge attribute to transfer, or an explicit array with
            shape ``(E, ...)``.
        points : Point
            Target point/vertex domain (``len(points)`` sets the output length).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(points), ...)`` containing the averaged values.

        Raises
        ------
        AttributeError
            If ``attr`` is a string and no such edge attribute exists, or if the
            provided array length does not match ``len(self)`` (validated by
            [`_check_attribute_to_compute`](npblender._check_attribute_to_compute)).
        IndexError
            If an endpoint index falls outside ``[0, len(points))``.
        TypeError
            If the attribute dtype cannot be averaged (e.g., non-numeric types).

        Examples
        --------
        ```python
        # Average per-edge scalar "heat" to vertices
        v_heat = edges.compute_attribute_on_points("heat", points)
        ```
        """

        @njit(cache=True)
        def _to_points(vertex0, vertex1, source, res):
            npoints = res.shape[0]
            nedges = vertex0.shape[0]

            count = np.zeros(npoints, dtype=np.int32)
            for iedge in range(nedges):
                s = source[iedge]

                ipoint = vertex0[iedge]
                res[ipoint] += s
                count[ipoint] += 1

                ipoint = vertex1[iedge]
                res[ipoint] += s
                count[ipoint] += 1

            trailing = 1
            for d in range(1, res.ndim):
                trailing *= res.shape[d]

            R2 = res.reshape((npoints, trailing))
            for ipoint in range(npoints):
                c = count[ipoint]
                if c > 0:
                    inv = 1.0 / c
                    for j in range(trailing):
                        R2[ipoint, j] *= inv

            return res  
        
        attr, item_shape = self._check_attribute_to_compute(attr)
        res = np.zeros((len(points),) + item_shape, dtype=attr.dtype)
        return _to_points(self.vertex0, self.vertex1, attr, res)                

# ====================================================================================================
# Spline Domain
# ====================================================================================================

class Spline(FaceSplineDomain):
    """
    Curve spline domain.

    Groups control points into splines and stores per-spline parameters
    (type, resolution, cyclic flags, and NURBS settings). Provides
    accessors to parametric functions and common evaluations.

    Attributes
    ----------
    loop_start : (S,) int
        Starting control-point index of each spline.
    loop_total : (S,) int
        Control-point count per spline.
    material_index : (S,) int, optional
        Material index per spline.
    curve_type : (S,) int
        One of constants ``BEZIER``, ``POLY``, ``NURBS``.
    resolution : (S,) int, optional
        Evaluation resolution (samples per segment).
    cyclic : (S,) bool, optional
        Whether each spline is closed.
    order : (S,) int, optional
        NURBS order.
    bezierpoint, endpoint : (S,) bool, optional
        NURBS flags.

    Properties
    ----------
    has_bezier : bool
        True if any spline is Bezier.
    functions : list
        List-like container of parametric spline functions (Bezier/Poly/NURBS).
    length : list of float
        Length of each spline (delegates to ``functions``).

    Methods
    -------
    delete_splines(selection, cpoints)
        Delete splines and their control points.
    add_splines(splines, **attributes)
        Append new splines given control-point counts.
    tangent(t)
        Evaluate tangents at parameter ``t`` in [0, 1].
    compute_attribute_on_points(attr, points)
        Broadcast per-spline attributes to control points.

    Examples
    --------
    Build functions and sample tangents:

    ```python
    funcs = splines.functions
    tan = splines.tangent(0.25)
    ```

    Append three splines with different sizes:

    ```python
    splines.add_splines([4, 6, 8], material_index=[0, 1, 1])
    ```

    > ***Note:*** ``functions`` relies on project spline implementations.
    Ensure control-point attributes (e.g., handles for Bezier, weights/order
    for NURBS) are present when required.

    > ***Caution:*** ``resolution`` semantics differ for cyclic vs. non-cyclic
    splines (endpoints handling).
    """
    domain_name = 'SPLINE'

    def _declare_attributes(self):
        super()._declare_attributes()

        self.new_int('curve_type', transdom=False)

        self.new_int('resolution',   optional=True, default=16, transdom=False)
        self.new_bool('cyclic',      optional=True, default=False, transdom=False)

        # Nurbs
        self.new_int('order',        optional=True, default=4, transdom=False)
        self.new_bool('bezierpoint', optional=True, transdom=False)
        self.new_bool('endpoint',    optional=True, transdom=False)

    # ====================================================================================================
    # Properties
    # ====================================================================================================

    @property
    def has_bezier(self):
        """
        Check if the domain contains any Bézier splines.

        Evaluates the ``curve_type`` attribute and returns `True` if at least one
        spline in the domain is of type ``BEZIER``.

        Returns
        -------
        bool
            `True` if the domain has at least one Bézier spline, `False` otherwise.

        Notes
        -----
        - Useful for quickly testing whether Bézier-specific logic (e.g., handle
        attributes) must be considered.
        - For mixed domains, the result is `True` as soon as one Bézier is present.

        Examples
        --------
        ```python
        if splines.has_bezier:
            print("Domain contains Bézier curves.")
        ```
        """
        return np.any(self.curve_type == BEZIER)

    # ----------------------------------------------------------------------------------------------------
    # Delete splines
    # ----------------------------------------------------------------------------------------------------

    def delete_splines(self, selection, cpoints):
        """
        Delete splines and their associated control points.

        Removes the splines selected in this domain, and deletes the corresponding
        control points from the given control-point domain. Corner indices are
        retrieved first to ensure consistent cleanup.

        Parameters
        ----------
        selection : array-like of bool or int
            Indices or boolean mask selecting which splines to delete.
        cpoints : ControlPoint
            Control-point domain from which the corresponding points are removed.

        Returns
        -------
        None

        Raises
        ------
        IndexError
            If any index in ``selection`` is out of bounds for this domain.
        ValueError
            If ``selection`` has an incompatible shape or type.

        Notes
        -----
        - This operation modifies both the spline domain and the control-point
        domain in place.
        - Deletion preserves consistency between splines and their control points.

        Examples
        --------
        ```python
        # Delete the first spline and its control points
        splines.delete_splines([0], cpoints)
        ```
        """
        corner_indices = self[selection].get_corner_indices()
        self.delete(selection)
        cpoints.delete(corner_indices)
    
    # ====================================================================================================
    # Interface with Blender
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Read data attributes
    # ----------------------------------------------------------------------------------------------------

    def load_bl_attributes(self, data):
        """
        Load spline attributes from a Blender object.

        Reads attributes stored in a Blender data block and transfers those that
        belong to the spline domain into this domain. New attributes are created if
        they do not already exist.

        Parameters
        ----------
        data : bpy.types.ID or similar
            Blender object or data block exposing an ``attributes`` mapping.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If reading or assignment of an attribute fails.
        AssertionError
            If Blender provides inconsistent attribute sizes.

        Notes
        -----
        - Only attributes with a matching domain and not marked as internal are
        imported.
        - If an attribute does not exist yet, it is created with the appropriate
        data type and marked as transferable.
        - The domain is resized once to match the size of Blender attributes.

        Examples
        --------
        ```python
        import bpy
        from npblender import Spline

        curve = bpy.data.curves["MyCurve"]
        splines = Spline()
        splines.load_bl_attributes(curve)
        ```
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
        """
        Transfer spline attributes to a Blender object.

        Writes this domain’s Blender attributes into the target Blender object or data
        block.

        Parameters
        ----------
        data : bpy.types.ID or similar
            Blender object or data block exposing an ``attributes`` mapping.
        update : bool, default=False
            If `True`, update existing attributes only. If `False`, create new
            attributes when they do not exist.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If attribute creation or assignment fails.

        Notes
        -----
        - Only attributes flagged with ``transfer=True`` are exported.
        - String attributes are currently skipped.

        Examples
        --------
        ```python
        import bpy
        from npblender import Spline

        curve = bpy.data.curves["MyCurve"]
        splines = Spline()
        splines.to_object(curve, update=False)
        ```

        > ***Caution:*** Blender does not save user-defined attributes inside
        > curve objects. Attributes written here may be lost when saving and
        > reopening the file.
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

    # ====================================================================================================
    # Compute attribute on points
    # ====================================================================================================
    
    def compute_attribute_on_points(self, attr, points):
        """
        Expand spline attributes to control points.

        Broadcasts attributes defined per-spline into per-point arrays by repeating
        each spline’s value over all its control points. This ensures compatibility
        when working in the control-point domain.

        Parameters
        ----------
        attr : str or numpy.ndarray
            Attribute to expand. If a string, it is looked up in this domain. If an
            array, it must have length equal to the number of splines.
        points : ControlPoint
            Control-point domain that receives the expanded attributes. The result
            has length equal to ``len(points)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(points), *attr_shape)`` containing the broadcasted
            attribute values.

        Raises
        ------
        AttributeError
            If the attribute is not found in this domain.
        ValueError
            If the provided attribute array shape does not match the number of
            splines.

        Notes
        -----
        - Expansion uses the ``loop_start`` and ``loop_total`` arrays to map splines
        to their corresponding ranges of control points.
        - This is the inverse of aggregating per-point attributes back to splines.

        Examples
        --------
        ```python
        # Broadcast per-spline weights to all control points
        w_points = splines.compute_attribute_on_points("weight", cpoints)
        ```
        """

        @njit(cache=True)
        def _to_points(loop_start, loop_total, source, res): 
            nsplines = loop_start.shape[0]
            for ispline in range(nsplines):
                start = loop_start[ispline]
                total = loop_total[ispline]
                for ipoint in range(start, start + total):
                    res[ipoint] += source[ispline]

            return res        

        attr, item_shape = self._check_attribute_to_compute(attr)
        res = np.zeros((len(points),) + item_shape, dtype=attr.dtype)
        return _to_points(self.loop_start, self.loop_total, attr, res)                


