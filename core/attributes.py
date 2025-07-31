# ====================================================================================================
# File        : attributes.py
# Package     : npblender (Blender with NumPy)
# Author      : Alain Bernard <lesideesfroides@gmail.com>
# Created     : 2022-11-11
# Updated     : 2025-07-21
# License     : MIT License
#
# Description :
# -------------
# This module defines the `Attributes` class, an extension of `FieldArray` tailored for geometry
# attribute management in Blender-compatible data structures.
#
# Attributes provides:
#   - Domain-aware attribute grouping (e.g., POINT, EDGE, FACE)
#   - User-friendly creation of typed attributes (float, vector, color, bool, etc.)
#   - Default value and metadata management for each field
#   - Integration with Blender's Geometry Nodes attribute system
#   - Serialization, deserialization, and interoperation with structured NumPy arrays
#
# Designed to work seamlessly with Blender's geometry system and the `npblender` data pipeline.
#
# Typical usage:
# --------------
# from npblender.attributes import Attributes
#
# __all__ = ['Attributes']
# ====================================================================================================

import numpy as np

if __name__ == "__main__":
    from fieldarray import FieldArray
    from constants import TYPES, DOMAINS, BL_TYPES, bfloat, bint, bbool
else:
    from . fieldarray import FieldArray
    from . constants import TYPES, DOMAINS, BL_TYPES, bfloat, bint, bbool


# =============================================================================================================================
# Geometry attributes

class Attributes(FieldArray):

    __slots__ = ('domain',)
    _slots = FieldArray._slots + ('domain',)

    def __init__(self, data=None, mode='COPY', domain=None, selector=None):
        """
        Initialize an Attributes object to manage geometric attributes for a given domain.

        Parameters
        ----------
        data : FieldArray, structured array, or Attributes, optional
            The array to initialize from.

        mode : str, default='COPY'
            - 'COPY': make a copy of the data.
            - 'CAPTURE': take ownership of the passed array.
            - 'EMPTY': create an empty buffer with same structure.

        domain : str, optional
            The domain this attribute set belongs to (e.g., 'POINT', 'EDGE', 'FACE', etc.).
            Required if `data` is not an Attributes instance.
        """
        object.__setattr__(self, 'domain', None)

        if isinstance(data, Attributes):
            self.domain = data.domain
        else:
            self.domain = domain
        super().__init__(a=data, mode=mode, selector=selector)

        if isinstance(data, Attributes):
            self.domain = data.domain
        elif domain in DOMAINS:
            self.domain = domain
        else:
            raise ValueError(f"Attributes > invalid or missing domain: {domain}. Valid domains are {DOMAINS}")

    # ====================================================================================
    # Serialization
    # ====================================================================================

    def to_dict(self, *, copy=True) -> dict:
        """
        Serialize the Attributes object to a dictionary.

        Parameters
        ----------
        copy : bool, default=True
            Whether to copy the data arrays (True) or keep views (False).

        Returns
        -------
        dict
            A dictionary with domain, fields and attribute metadata.
        """
        return {
            'domain': self.domain,
            'fields': super().to_dict(copy=copy, with_infos=True),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Attributes":
        """
        Restore an Attributes object from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing 'domain' and 'fields', as returned by `to_dict()`.

        Returns
        -------
        Attributes
            A new instance restored from the serialized dictionary.
        """
        # Reconstruct the FieldArray base
        base = FieldArray.from_dict(data['fields'], copy=False)

        # Create new instance and attach the captured data
        obj = cls(domain=data['domain'], a=base, mode='CAPTURE')
        return obj

    # ====================================================================================================
    # Create a new attribute
    # ====================================================================================================

    def new_attribute(self, name, data_type, default=0, transfer=True):
        """
        Create a new attribute for the given name and data type.

        This method is used internally by user-friendly methods like `new_float`, `new_bool`, etc.

        Parameters
        ----------
        name : str
            The attribute name.

        data_type : str
            Geometry Nodes data type, one of:
            ('FLOAT', 'INT', 'FLOAT_VECTOR', 'FLOAT_COLOR', 'BYTE_COLOR',
            'STRING', 'BOOLEAN', 'FLOAT2', 'INT8', 'INT32_2D', 'QUATERNION', 'MATRIX').

        default : any, optional (default=0)
            The default value to use when new items are created.

        transfer : bool, optional (default=True)
            Whether this attribute should be transferred back to the Blender object.
        """

        # ----- Validate data_type
        if data_type not in TYPES:
            raise ValueError(f"new_attribute > unknown data type '{data_type}'")

        # ----- Create field with associated metadata
        self.new_field(
            name,
            dtype=TYPES[data_type]['dtype'],
            shape=TYPES[data_type]['shape'],
            default=default,
            data_type=data_type,
            transfer=transfer
        )

        # ----- Special case: auto-fill ID if non-empty
        if name == 'ID' and len(self) > 0:
            self['ID'] = np.arange(len(self))

    # ----------------------------------------------------------------------------------------------------
    # User friendly versions

    def new_float(self, name, default=0., transfer=True):
        """Create a new attribute of type FLOAT (float)."""
        self.new_attribute(name, 'FLOAT', default, transfer=transfer)

    def new_vector(self, name, default=(0., 0., 0.), transfer=True):
        """Create a new attribute of type FLOAT_VECTOR (3D float vector)."""
        self.new_attribute(name, 'VECTOR', default, transfer=transfer)

    def new_int(self, name, default=0, transfer=True):
        """Create a new attribute of type INT (int)."""
        self.new_attribute(name, 'INT', default, transfer=transfer)

    def new_bool(self, name, default=False, transfer=True):
        """Create a new attribute of type BOOLEAN (bool)."""
        self.new_attribute(name, 'BOOLEAN', default, transfer=transfer)

    def new_color(self, name, default=(0.5, 0.5, 0.5, 1.), transfer=True):
        """Create a new attribute of type FLOAT_COLOR (RGBA)."""
        self.new_attribute(name, 'COLOR', default, transfer=transfer)

    def new_vector2(self, name, default=(0., 0.), transfer=True):
        """Create a new attribute of type FLOAT2 (2D float vector)."""
        self.new_attribute(name, 'FLOAT2', default, transfer=transfer)

    def new_string(self, name, default="", transfer=True):
        """Create a new attribute of type STRING (text)."""
        self.new_attribute(name, 'STRING', default, transfer=transfer)

    def new_byte_color(self, name, default=(0, 0, 0, 255), transfer=True):
        """Create a new attribute of type BYTE_COLOR (RGBA, 8-bit)."""
        self.new_attribute(name, 'BYTE_COLOR', default, transfer=transfer)

    def new_int8(self, name, default=0, transfer=True):
        """Create a new attribute of type INT8 (small integer)."""
        self.new_attribute(name, 'INT8', default, transfer=transfer)

    def new_int32_2d(self, name, default=(0, 0), transfer=True):
        """Create a new attribute of type INT32_2D (2D integer vector)."""
        self.new_attribute(name, 'INT32_2D', default, transfer=transfer)

    def new_quaternion(self, name, default=(1., 0., 0., 0.), transfer=True):
        """Create a new attribute of type QUATERNION (4D float tuple)."""
        self.new_attribute(name, 'QUATERNION', default, transfer=transfer)

    def new_matrix(self, name, default=((1, 0, 0, 0),
                                        (0, 1, 0, 0),
                                        (0, 0, 1, 0),
                                        (0, 0, 0, 1)), transfer=True):
        """Create a new attribute of type MATRIX (4x4 transformation matrix)."""
        self.new_attribute(name, 'MATRIX', default, transfer=transfer)

    # ====================================================================================================
    # Reset to default values
    # ====================================================================================================

    def reset(self):
        """Reset all attribute values to their default.

        Special case:
        - If the attribute is named 'ID', it is filled with a range from 0 to len(self).
        """
        n = len(self)
        if n == 0:
            return

        for name, info in self._infos.items():
            if name == 'ID':
                self[name] = np.arange(n)
            else:
                default = info.get('default', 0)
                self[name] = default

    # ====================================================================================================
    # Interface with Blender
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Read data attributes
    # ----------------------------------------------------------------------------------------------------

    def from_object(self, spec):
        """ Read the object attributes.

        Arguments
        ---------
            - spec (str or data) : the data to set the attributes to
        """

        from . import blender

        data = blender.get_data(spec)
        size = None

        if not hasattr(data, 'attributes'):
            return

        for name, binfo in data.attributes.items():

            # Must be external in the domain
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

    def to_object(self, spec, update=False):
        """ Transfer the attributes to a blender mesh object.

        Arguments
        ---------
            - spec (str or data) : the mesh to set the attributes to
            - attributes (array of st = None) : the attributes to transfer (all if None)
            - update (bool=False) : update the attributes values without trying to create them
        """

        from . import blender

        data = blender.get_data(spec)

        for name, info in self._infos.items():

            if not info['transfer']:
                continue

            if info['data_type'] == 'STRING':
                pass

            if update:
                blender.set_attribute(data, name, self[name])
            else:
                blender.create_attribute(data, name, info['data_type'], domain=self.domain, value=self[name])

