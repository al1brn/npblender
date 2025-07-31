# ====================================================================================================
# File        : fieldarray.py
# Package     : npblender (Blender with NumPy)
# Author      : Alain Bernard <lesideesfroides@gmail.com>
# Created     : 2022-11-11
# Updated     : 2025-07-21
# License     : MIT License
#
# Description :
# -------------
# This module defines the `FieldArray` class, a dynamic and extensible wrapper for NumPy structured arrays.
# 
# FieldArray provides:
#   - Dynamic addition and removal of fields with metadata support
#   - Efficient buffer management for appending, extending, and deleting records
#   - Attribute-style access to fields using Python-safe names
#   - Serialization and deserialization to/from dictionary format
#   - Optional field metadata (default values, names, types, etc.)
#
# Designed for use within the `npblender` ecosystem to facilitate geometry attribute manipulation
# in a NumPy-compatible and Blender-aware way.
#
# Typical usage:
# --------------
# from npblender.fieldarray import FieldArray
#
# __all__ = ['FieldArray']
# ====================================================================================================

import numpy as np


class FieldArray(object):

    """ Dynamic structured array

    A structured array with cache management to optimize record appending.
    """

    __slots__ = ('_data', '_length', '_infos')

    # Used by __setattr__
    # Be sure to update this class property in child classes
    # _slots = FieldArray._slots + (x, y, ...)
    _slots = ('_data', '_length', '_infos')

    # ====================================================================================================
    # Initialization
    # ====================================================================================================

    def __init__(self, a=None, mode='COPY', selector=None):
        """ Initialize the array with another array

        Arguments
        ---------
        - a (array or FieldArray) : the array to initialize from
        - mode (str in 'COPY', 'CAPTURE', 'EMPTY') : copy the content
        - selector (Any = None) : a selector on data 
        """

        object.__setattr__(self, '_infos',  {})
        object.__setattr__(self, '_length', 0)
        object.__setattr__(self, '_data',   None)

        if a is not None:

            # Make sure it is an array
            data = np.asarray(a)

            # If we have a selector, we apply it in case names would be removed
            if selector is not None:
                try:
                    data = data[selector]
                except Exception as e:
                    raise ValueError(f"{type(self).__name__} init> invalid selector({type(selector)}), {str(e)}")
                
                if isinstance(data, np.void):
                    data = np.reshape(data, 1)

            # Make sure we have names
            if data.dtype.names is None:
                raise ValueError(f"{type(self).__name__} init> input must be a structured array (selector={selector})")
            
            # Mode
            if mode == 'COPY':
                self._data = data.copy()
            elif mode == 'CAPTURE':
                self._data = data
            elif mode == 'EMPTY':
                self._data = np.zeros(0, dtype=data.dtype)
            else:
                raise ValueError(f"{type(self).__name__} init> mode must be in ('COPY', 'CAPTURE', 'EMPTY'), not '{mode}'")
            
            # Data can be a scalar with shape ()
            if data.shape == ():
                self._length = 1
            else:
                self._length = len(data)
            
            # Build infos
            if isinstance(a, FieldArray):
                self._infos = {name: {**info} for name, info in a._infos.items()}

            else:
                self._infos = {}
                for name in data.dtype.names:
                    shape = data[name].shape
                    if shape is not None:
                        shape = shape[1:]
                    self._infos[name] = {'dtype': data[name].dtype, 'shape': shape, 'default': 0, 'optional': False}


    @property
    def is_scalar(self):
        return self._data.shape == ()

    def __len__(self):
        if self._data is None:
            raise ValueError(f"{type(self).__name__} > data is None.")
        
        if self._data.shape == ():
            raise ValueError(f"{type(self).__name__} > data is scalar.")

        return self._length
    
    def __iter__(self):
        return (self[index] for index in range(self._length))
        #return iter(self._data[:self._length])
    
    def __contains__(self, item):
        return item in self._data[:self._length]
    
    def __array__(self, dtype=None):
        """Allow implicit conversion to np.ndarray"""
        if self._data.shape == ():
            arr = self._data
        else:
            arr = self._data[:self._length]
        return arr.astype(dtype) if dtype else arr

    def __str__(self):
        if self._data is None:
            return f"<{type(self).__name__} None>"
        len_str = "scalar" if self.is_scalar else len(self)
        return f"<{type(self).__name__}: names: {self.all_names}, len: {len_str} >"
    
    def __repr__(self):
        if self._data is None:
            return "<{type(self).__name__}: empty>"

        # Compute max length of original field names for alignment
        max_len = max((len(name) for name in self._infos), default=0)

        lines = [f"<{type(self).__name__}: len={self._length}, fields=["]
        for name, info in self._infos.items():
            #dtype = self._data.dtype[name]
            #shape = dtype.shape if dtype.shape else ()
            dtype, shape = info['dtype'], info['shape']
            #type_str = getattr(dtype.base, '__name__', str(dtype))
            #type_str = dtype.__name__
            type_str = FieldArray._dtype_name(dtype)

            # Shape display
            if shape:
                type_str += f"[{', '.join(map(str, shape))}]"

            # Extra info (e.g., default value)
            default = info.get('default', None)
            default_str = f", default={default}" if default is not None else ""

            opt = info.get('optional', False)
            opt_str = f"O " if opt else "  "

            lines.append(f"  '{name:<{max_len}}' {opt_str}: {type_str}{default_str}")

        lines.append("]>")
        return "\n".join(lines)
    
    # ====================================================================================================
    # Fields
    # ====================================================================================================

    @staticmethod
    def to_py_name_deprecated(name: str) -> str:
        """
        Convert any string to a valid Python identifier, preserving case.

        Examples:
        ---------
        >>> FieldArray.to_py_name("My Field!")
        'My_Field_'
        """
        import re
        return re.sub(r'\W|^(?=\d)', '_', name)
    
    @staticmethod
    def _dtype_name(dtype):
        try:
            dtype = np.dtype(dtype)
            return dtype.name  # 'float32', 'int64', etc.
        except TypeError:
            return getattr(dtype, '__name__', str(dtype))
    
    @staticmethod
    def _normalize_shape(shape):
        if shape is None:
            return ()
        if isinstance(shape, int):
            return (shape,)
        if isinstance(shape, (list, tuple)):
            return tuple(int(d) for d in shape)
        raise TypeError(f"Shape must be int, tuple, list or None, got {type(shape)}")
    
    @property
    def dtype(self):
        """ Array structured dtype

        Returns the dtype property of the structured array.
        """
        if self._data is None:
            return None
        else:
            return self._data.dtype

    @property
    def all_names(self):
        """ Column names.

        Returns all the field names, including optional ones.
        """
        return list(self._infos.keys())

    @property
    def actual_names(self):
        """ Column names.

        Returns the actual field names, excluding optional fields.
        """
        if self._data is None:
            return []
        else:
            return self._data.dtype.names
        #return s[name for name, infos in self._infos.items() if not infos.get('optional', False)]

    def field_exists(self, name):
        return name in self.all_names
        
    def get_field_shape(self, name):
        if name in self.all_names:
            return self._infos[name]['shape']
        else:
            raise KeyError(f"Attribute name '{name}' not found in {self}")

    def get_field_size(self, name):
        return int(np.prod(self.get_field_shape(name)))
    
    # ====================================================================================================
    # Buffer management
    # ====================================================================================================

    def set_buffer_size(self, size):
        if len(self._data) >= size:
            return
        
        new_data = np.empty(size, self._data.dtype)
        new_data[:self._length] = self._data[:self._length]
        self._data = new_data
    
    def _data_check(self, new_length):
        size = len(self._data)
        if new_length > size:
            self.set_buffer_size(size=int(1.5*new_length))

    def clear(self):
        self._length = 0

    def resize(self, length):
        self._data_check(length)
        if length > self._length:

            # Fill with default is current length is null
            if self._length == 0:
                for name, info in self._infos.items():
                    if not info.get('optional', False):
                        self._data[name][self._length:length] = info.get('default', 0)

            # Duplicate current content
            else:
                reps = (length - 1) // self._length
                assert((reps+1)*self._length >= length)
                self._data[self._length:length] = np.tile(self._data[:self._length], reps)[:length - self._length]

        self._length = length

    # ====================================================================================================
    # Items
    # ====================================================================================================

    def __getitem__(self, index):

        # ---------------------------------------------------------------------------
        # Index is a string, we return the array of the full content
        # ---------------------------------------------------------------------------

        if isinstance(index, str):
            self._ensure_optional_field(index)
            if self._data.shape == ():
                return self._data[index]
            else:
                return self._data[index][:self._length]
        
        # ---------------------------------------------------------------------------
        # A single integer, we return a recarray
        # ---------------------------------------------------------------------------

        elif isinstance(index, (int, np.int32, np.int64)):
            if index >= self._length:
                raise IndexError(f"FieldArray has only {self._length} items, {index} index is not valid.")
            
            #return self._data[index].view(np.recarray)
            return type(self)(self._data[index], mode='CAPTURE')

        # ---------------------------------------------------------------------------
        # Other case : we return a FieldArray on the selection
        # ---------------------------------------------------------------------------

        else:
            return type(self)(self._data[:self._length], mode='CAPTURE', selector=index)

    def __setitem__(self, index, value):

        self._ensure_optional_field(index)

        self._data[:self._length][index] = np.asarray(value)

    # ----------------------------------------------------------------------------------------------------
    # Names as attributes
    # ----------------------------------------------------------------------------------------------------

    def __getattr__(self, name):

        if name in self._infos:

            self._ensure_optional_field(name)

            if self._data.shape == ():
                return self._data[name]
            else:
                return self._data[name][:self._length]

        raise AttributeError(f"{type(self).__name__}> no field named '{name}'")
    
    def __setattr__(self, name, value):

        if name in self._slots or name in type(self).__dict__:
            super().__setattr__(name, value)

        elif name in self._infos:

            self._ensure_optional_field(name)

            if self.is_scalar:
                self._data[name] = value
            else:
                self._data[name][:self._length] = value

        else:
            raise AttributeError(f"{type(self).__name__}> no field named '{name}'. Valid names are {self.all_names}")

    # ----------------------------------------------------------------------------------------------------
    # Sub‑selection
    # ----------------------------------------------------------------------------------------------------

    def filtered(self, selector, *, copy=False):
        """
        Return a FieldArray containing only the selected records.

        Parameters
        ----------
        selector : array‑like, slice or int
            Any valid NumPy 1‑D index: boolean mask, integer index/array, or slice.
            It is applied to the current valid part of the buffer
            (``self._data[:self._length]``).

        copy : bool, optional
            - False    (default) => the new array shares the same memory
                                     (changes propagate both ways).
            - True           => the data are physically copied.

        Returns
        -------
        FieldArray
            A new instance holding exactly ``len(selector)`` records and
            inheriting the current field‑infos.
        """
        if selector is None:
            return self

        if self._data is None or self._length == 0:
            # Nothing to select – return an empty compatible array
            return type(self)(self, copy='EMPTY')

        # Apply the selector on the valid slice
        #selected = self._data[:self._length][selector]

        # Choose copy mode
        mode = 'COPY' if copy else 'CAPTURE'

        # Build the new instance
        return type(self)(self, mode=mode, selector=selector)
    
    # ====================================================================================================
    # Create fields
    # ====================================================================================================

    def is_reserved_name(self, name):
        return name in self._slots or hasattr(type(self), name)
    
    def has_field(self, name: str) -> bool:
        return name in self._infos
    
    # ----------------------------------------------------------------------------------------------------    
    # Create a new field in _data
    # ----------------------------------------------------------------------------------------------------

    def _create_field_in_data(self, name):
        """ Add actueallu a field to the structured array.

        The name argument must be a valid key in _infos

        Arguments
        ---------
            - name (str) : field name
        """
        infos = self._infos.get(name)
        if infos is None:
            raise ValueError(f"new_field > '{name}' is not in _infos.")
        
        # Field dtype
        dtype, shape, default = infos['dtype'], infos['shape'], infos['default']
        field_type = [(name, dtype) if shape is None else (name, dtype, shape)]

        # Array new dtype
        if self.dtype is None:
            new_dtype = field_type
        else:
            new_dtype = self.dtype.descr + field_type

        # New data
        length = 10 if self._data is None else len(self._data)

        new_data = np.zeros(length, new_dtype)
        if self._length > 0:
            try:
                new_data[name][:self._length] = default
            except:
                raise ValueError(f"new_field > impossible to set default value {default} to field '{name}'")
            
        # Copy existing data in new_data
        if self._data is not None:
            for fname in self._data.dtype.names:
                new_data[fname] = self._data[fname]
            del self._data

        # Switch
        self._data = new_data

        # Update optional
        self._infos[name]['optional'] = False

    # ----------------------------------------------------------------------------------------------------    
    # Declare a new field and create it if not optional
    # ----------------------------------------------------------------------------------------------------

    def new_field(self, name, dtype, shape=None, default=0, optional=False, **infos):
        """ Add a field to the structured array.

        Arguments
        ---------
            - name (str) : field name
            - dtype (type) : a valid numpy dtype
            - shape (tuple = None) : the shape of the field
            - default (any = 0) : default value
            - optional (bool = False) : the field is created only when accessed
            - infos (dict) : field infos
        """
        # Reserved name
        if self.is_reserved_name(name):
            raise ValueError(f"new_field > '{name}' is a reserved name.")
        
        # Already exists
        if name in self._infos:
            raise ValueError(f"new_field > '{name}' is already defined.")
        
        # Declare in infos
        self._infos[name] = {
            'dtype'    : dtype,
            'shape'    : FieldArray._normalize_shape(shape),
            'default'  : default,
            'optional' : optional,
            **infos,
            }
        
        # Create if not optional
        if not optional:
            self._create_field_in_data(name)

    # ----------------------------------------------------------------------------------------------------    
    # Dynamically create optional field
    # ----------------------------------------------------------------------------------------------------

    def _ensure_optional_field(self, index):
        if isinstance(index, str):
            if index not in self._infos:
                raise IndexError(f"FieldArray has no field named {index}")
            
            if self._infos[index]['optional']:
                self._create_field_in_data(index)

    # ----------------------------------------------------------------------------------------------------    
    # Copy an existing field
    # ----------------------------------------------------------------------------------------------------    

    def copy_field(self, field_name: str, new_name: str, **infos):
        """
        Duplicate an existing field under a new name, with optional metadata overrides.

        Parameters
        ----------
        field_name : str
            The name of the existing field to copy.

        new_name : str
            The name of the new field to create.

        infos : keyword arguments
            Optional metadata (e.g. default, unit, description...) to override
            or supplement the original field's metadata.

        Raises
        ------
        KeyError
            If the source field does not exist.

        ValueError
            If the target name already exists or is reserved.
        """
        # --- Validation -----------------------------------------------------
        if field_name not in self.all_names:
            raise KeyError(f"copy_field > unknown field '{field_name}'")

        if new_name in self.all_names:
            raise ValueError(f"copy_field > '{new_name}' already exists")

        if self.is_reserved_name(new_name):
            raise ValueError(f"copy_field > '{new_name}' is a reserved name")
        
        # --- Duplicate infos and create -------------------------------------
        self._infos[new_name] = {**self._infos[field_name]}

        if not self._infos[new_name]['optional']:
            self._create_field_in_data(new_name)
    
    # ----------------------------------------------------------------------------------------------------    
    # Join fields from another FielArray
    # ----------------------------------------------------------------------------------------------------    

    def join_fields(self, other):
        """
        Add all missing fields from another FieldArray.

        For every field in `other` that is not present in `self`, a new field is created
        with the same dtype and shape, and initialized with its default value across all existing records.

        Parameters
        ----------
        other : FieldArray
            Another FieldArray instance whose fields will be checked for missing fields.
        """
        if isinstance(other, FieldArray):
            for name in other._infos:
                if name in self._infos:
                    continue

                self._infos[name] = {**other._infos[name]}
                if not self._infos[name]['optional']:
                    self._create_field_in_data(name)

        else:
            for name in other.dtype.names:
                if name in self._infos:
                    continue

                self.new_field(name, dtype=other[name].dtype, shape=other[name].shape)

    # ====================================================================================================
    # add & append
    # ====================================================================================================

    def add(self, count, **fields):
        """ Add count records

        New records are set with default values or values provided by the user in fields dict.

        Parameters
        ----------
        count : int
            Number of records to add.
        fields : dict
            Keyword arguments mapping field names to values.

        """
        if count <= 0:
            return
        
        old_length = self._length
        self.resize(self._length + count)
        sl = slice(old_length, self._length)

        for name, value in fields.items():
            if name in self._infos:
                self._ensure_optional_field(name)
                self._data[name][sl] = value
            else:
                raise KeyError(f"Attribute name '{name}' not found in {list(self._infos.keys())}")
            
        for name in self._data.dtype.names:
            if name in fields:
                continue
            self._data[name][sl] = self._infos[name].get('default', 0)



    def append(self, **fields):
        """
        Append values to the structured array.

        The number of records to append is determined by the number of fields provided in the fields
        dictionary. The values of the fields are copied to the new records.

        Parameters
        ----------
        fields : dict
            Keyword arguments mapping field names to values.
        """

        # Determine the number of items to append
        count  = 0
        arrays = {}
        for name, value in fields.items():
            
            if name not in self._infos:
                raise ValueError(f"{type(self).__name__}.append > Invalid field name: '{name}' not in {list(self._infos.keys())}")
            
            self._ensure_optional_field(name)
            
            info = self._infos[name]
            a = np.asarray(value)

            if np.shape(a) == ():
                n = 1

            else:
                shape = self._data.dtype[name].shape
                if shape == a.shape:
                    n = 1
                else:
                    bshape = np.broadcast_shapes(a.shape, shape)
                    n = len(a)

            arrays[name] = a
            if n > count:
                count = n

        if count == 0:
            return []

        # Ensure the internal buffer is large enough
        new_length = self._length + count
        self._data_check(new_length)

        # Assign values to the provided fields
        for name, a in arrays.items():
            self._data[name][self._length:new_length] = a

        # Fill missing fields with default value
        for name in self._data.dtype.names:
            if name not in arrays:
                self._data[name][self._length:new_length] = self._infos[name].get('default', 0)      

        # To be returned
        new_indices = np.arange(self._length, new_length)

        # Update current length
        self._length = new_length

        # Return the indices of records appended
        return new_indices

    # ====================================================================================================
    # Extend
    # ====================================================================================================

    def extend(self, other, join_fields=True):
        """
        Append multiple records from another array or FieldArray.

        Parameters
        ----------
        other : FieldArray or structured np.ndarray
            The array of records to append. Must have named fields matching
            a subset of the current array's fields.
        """
        if other is None:
            return

        # Accept both FieldArray and structured np.ndarray
        data = np.asarray(other)

        count = len(data)
        if count == 0:
            return
        
        # No names
        if data.dtype.names is None:
            raise ValueError(f"{type(self).__name__}.extend> input must have named fields")
        
        # Join fields if required
        if join_fields:
            self.join_fields(other)
            
        # Ensure all fields in 'data' are valid for this array
        invalid = [name for name in data.dtype.names if name not in self.all_names]
        if invalid:
            raise ValueError(f"extend > input contains unknown field(s): {invalid}")

        # Resize buffer
        new_length = self._length + count
        self._data_check(new_length)

        # Copy data field by field
        for name in data.dtype.names:
            self._ensure_optional_field(name)
            self._data[name][self._length:new_length] = data[name]

        # Fill missing fields with default value
        for name in self.actual_names:
            if name not in data.dtype.names:
                self._data[name][self._length:new_length] = self._infos[name].get('default', 0)

        self._length = new_length
            
    # ====================================================================================================
    # Multiply
    # ====================================================================================================

    def multiply(self, count: int):
        """
        Duplicate the current records `count` times.

        Parameters
        ----------
        count : int
            Number of times to repeat the current records.

        Notes
        -----
        This duplicates the current valid records (up to self._length).
        If the array is empty or count <= 1, nothing happens.

        Example:
        --------
        If the array has 3 records and count == 4, the result will be:

        [rec0, rec1, rec2, rec0, rec1, rec2, rec0, rec1, rec2, rec0, rec1, rec2]
        """
        if count <= 1 or self._length == 0:
            return

        self.resize(self._length*count)
        return


        original = self._data[:self._length]
        new_length = self._length * count

        # Use NumPy's resize — safe because new_length is a strict multiple
        self._data = np.resize(original, new_length)
        self._length = new_length


    # ====================================================================================================
    # Delete items
    # ====================================================================================================

    def delete(self, index):
        """
        Delete a selection of items from the array.

        Parameters
        ----------
        index : int, slice, or array-like
            The indices of the elements to delete from the current data.

        Notes
        -----
        This operates only on the valid range `[0:self._length]`.
        The internal buffer is preserved (no reallocation).
        """

        # Normalize the index to a boolean mask
        mask = np.ones(self._length, dtype=bool)
        mask[index] = False  # mark items to keep

        kept = np.count_nonzero(mask)
        if kept == self._length:
            return  # nothing deleted
        elif kept == 0:
            self._length = 0
            return

        # Compact the buffer in-place
        for name in self._data.dtype.names:
            self._data[name][:kept] = self._data[name][:self._length][mask]

        self._length = kept

    # ====================================================================================================
    # Serialization
    # ====================================================================================================

    def to_dict(self, *, copy: bool = True, with_infos: bool = True) -> dict:
        """
        Convert the array to a dictionary of fields or (field, infos) pairs.

        Parameters
        ----------
        copy : bool, default=True
            Whether to copy the arrays.

        with_infos : bool, default=True
            If True, return (array, infos) for each field.

        Returns
        -------
        dict[str, array or (array, dict)]
        """
        result = {}
        valid_slice = slice(0, self._length)
        for name, info in self._infos.items():
            if info.get('optional', False):
                arr = None
            else:
                arr = self._data[name][valid_slice]
                arr = arr.copy() if copy else arr

            result[name] = (arr, dict(info)) if with_infos else arr
        return result
    
    @classmethod
    def from_dict(cls, data: dict):
        """
        Build a FieldArray from a dictionary with field data and optional metadata.

        Parameters
        ----------
        data : dict[str, array-like or (array, dict)]
            Mapping field names to arrays or (array, infos). Infos must include NAME.

        copy : bool, optional
            Whether to copy the data. Default: True.

        Returns
        -------
        FieldArray
        """
        if not data:
            raise ValueError("from_dict > input dictionary is empty")

        arrays = {}
        infos = {}
        lengths = set()

        for name, value in data.items():
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
                arr, arr_infos = value
            else:
                arr = np.asarray(value)
                arr_infos = None

            if arr is None:
                if arr_infos is None:
                    raise ValueError(f"from_dict > field '{name}' has no array, nor infos")
                
                arr_infos['optional'] = True

            else:
                arr = np.asarray(arr)
                if arr_infos is None:
                    arr_infos = {'dtype': arr.dtype, 'shape': arr.shape, 'default': 0, 'optional': False}
                arrays[name] = arr
                lengths.add(arr.shape[0])

            infos[name] = arr_infos

        if len(lengths) != 1:
            raise ValueError("from_dict > all fields must have the same number of records")
        
        n = lengths.pop()
        descr = [
            (name, arr.dtype.str, arr.shape[1:]) if arr.ndim > 1 else (name, arr.dtype.str)
            for name, arr in arrays.items()
        ]

        structured = np.empty(n, dtype=descr)
        for name, arr in arrays.items():
            structured[name] = arr

        fa = cls(structured, mode='CAPTURE')
        fa._infos.update(infos)

        return fa
    
    # ====================================================================================================
    # Utility : as  kwargs arguments
    # ====================================================================================================

    def as_kwargs(self, selector=None, include=None, exclude=None):
        """
        Return a dictionary of field values formatted as kwargs.

        Parameters
        ----------
        selector : slice, int, list, or mask
            Optional selection of elements to extract.

        include : list[str], optional
            List of field names (original or python names) to include.

        exclude : list[str], optional
            List of field names (original or python names) to exclude.

        Returns
        -------
        dict
            Mapping from python-safe field names to array values.
        """
        exclude = set(exclude or [])
        include = set(include) if include is not None else None

        base = FieldArray(self[selector], mode='CAPTURE').to_dict(with_infos=False)
        result = {}

        for name, value in base.items():
            if include and (name not in include):
                continue
            if name in exclude:
                continue
            result[name] = value

        return result
    

if __name__ == "__main__":

    from pprint import pprint

    # ------------------------------------------------------------------------------------------
    # Dump utility
    # ------------------------------------------------------------------------------------------

    def dump(fa, title, what='BOTH'):
        
        is_array = not isinstance(fa, FieldArray)
        str_fa = f"Array {type(fa)}" if is_array else f"FieldArray {type(fa._data)}"
        print("\n", '-'*5, title, f"({str_fa})", '-'*60, "\n")

        if what.upper() in ['BOTH', 'REPR']:
            print(repr(fa))
            print()

        if what.upper() in ['BOTH', 'CONTENT']:
            if is_array:
                for name in fa.dtype.names:
                    print(f" - {name:<15s}", fa[name])
            else:
                for name in fa.actual_names:
                    print(f" - {name:<15s}", fa[name])

    # ------------------------------------------------------------------------------------------
    # Base
    # ------------------------------------------------------------------------------------------

    fa = FieldArray()
    dump(fa, "No Field")

    fa.new_field("an_int", int, default=9)
    fa.new_field("a_vector", float, 3, default=9.1)
    fa.new_field("the_opt", float, default=7, optional=True)
    dump(fa, "Empty", 'REPR')

    fa.append(an_int=np.arange(5), a_vector=[[0], [1], [2], [3], [4]])
    dump(fa, "Append", 'BOTH')

    fa.add(2)
    dump(fa, "Add", 'CONTENT')

    dump(fa[[1, 3]], "fa[[1, 3]]", 'BOTH')
    dump(fa[1], "fa[1]", 'BOTH')

    print(f"fa[1, 3].an_int: {fa[[1, 3]].an_int}")
    print(f"fa[1].an_int: {fa[1].an_int}")

    # ------------------------------------------------------------------------------------------
    # To / from dict
    # ------------------------------------------------------------------------------------------

    d = fa.to_dict()
    fa = FieldArray.from_dict(d)

    dump(fa, "from_dict(d)", 'BOTH')

    print("KWARGS")
    pprint(fa[[1, 3]].as_kwargs())

    # ------------------------------------------------------------------------------------------
    # Add / delete / clear
    # ------------------------------------------------------------------------------------------

    fa.clear()
    fa.add(10)
    fa.an_int = np.arange(10)
    fa.a_vector = np.arange(10)[:, None]
    dump(fa, "Clear + add(10)")

    fa.delete([1, 3])
    dump(fa, "Delete 1, 3", 'CONTENT')

    # ------------------------------------------------------------------------------------------
    # New field on non empty array
    # ------------------------------------------------------------------------------------------

    fa.new_field("index", int, default=7)
    dump(fa, "index Field")

    fa.append(index=[-1]*3)
    dump(fa, "Indices")

    # ------------------------------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------------------------------

    print("ITERATION")

    for i, a in enumerate(fa):
        print("-", i, "->", a.an_int, a.a_vector, a.index, np.shape(a.index))

    # ------------------------------------------------------------------------------------------
    # Get / Set
    # ------------------------------------------------------------------------------------------

    fa.clear()
    fa.append(an_int=np.arange(5), a_vector=np.arange(5)[:, None], index=range(10, 15))
    dump(fa, "Clear + an_int", 'CONTENT')

    fa[3] = fa[1]
    dump(fa, "3 <- 1", 'CONTENT')

    fa.clear()
    fa.append(an_int=np.arange(5), a_vector=np.arange(5)[:, None], index=range(10, 15))

    fa[2:] = fa[:3]
    dump(fa, "2: <- :3", 'CONTENT')

    # ------------------------------------------------------------------------------------------
    # Join fields
    # ------------------------------------------------------------------------------------------

    fb = FieldArray()
    fb.new_field("an_int",  dtype=int, default=321)
    fb.new_field("new_int", dtype=int, default=123)
    fb.new_field("new_opt", dtype=int, default=777, optional=True)

    fa.join_fields(fb)
    dump(fa, "Join Fields")

    # ------------------------------------------------------------------------------------------
    # Extend
    # ------------------------------------------------------------------------------------------

    fb = FieldArray()
    fb.new_field("another_int", int, default=113)
    fb.new_field("another_opt", int, default=113, optional=True)
    fb.add(7, another_int=range(7))
    dump(fb, "fb")

    fa.extend(fb)
    dump(fa, "Extend")

    # ------------------------------------------------------------------------------------------
    # Multiply
    # ------------------------------------------------------------------------------------------

    fa.clear()
    fa.add(2)
    fa.multiply(10)
    dump(fa, "multiply 5", 'REPR')

    # ------------------------------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------------------------------

    print("Get Optional: the_opt")
    print(fa.the_opt)

    dump(fa, 'Get the_opt', 'REPR')

    print("Set Optional: new_opt")
    fa.new_opt = 100 + np.arange(len(fa))
    print(fa.new_opt)

    dump(fa, 'Set the_opt', 'REPR')

    # ------------------------------------------------------------------------------------------
    # Scalars
    # ------------------------------------------------------------------------------------------

    print("\nScalars")

    fa = FieldArray()
    fa.new_field("i", dtype=int)
    fa.new_field("j", dtype=int)
    fa.new_field("v", dtype=int, shape=3)

    fa.append(i=np.arange(3), j=np.arange(3)+5)
    print("fa.is_scalar:", fa.is_scalar)
    print("i:", len(fa), fa.i)
    print("j:", len(fa), fa.j)

    try:
        print("Ooops", len(fa[1]))
    except Exception as e:
        print("Scalar exception: ", str(e))

    print("i[1]:", fa[1].i, fa[1].is_scalar)
    print("j[1]:", fa[1].j, fa[1].is_scalar)

    print(type(fa), type(fa[1]))

    fa[1] = fa[0]
    print("i:", len(fa), fa.i)
    print("j:", len(fa), fa.j)

    fa.clear()
    print("Empty:", len(fa))
    fa.append(i=1, j=2)
    print("Length = 1 ?", len(fa))
    print("i:", len(fa), fa.i)
    print("j:", len(fa), fa.j)

    fa.clear()
    fa.append(v=(1, 2, 3))
    dump(fa, "Set a vector scalar")

    print("Tests completed")








    

    



