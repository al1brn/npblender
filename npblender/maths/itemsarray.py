# npblender/itemsarray.py
# MIT License
# Created on 2022-11-11
# Last update: 2025-07-21
# Author: Alain Bernard

"""
ItemsArray
==========

Base class representing a batch of structured array items (e.g., 3D vectors, 4×4 matrices),
backed by a NumPy array. All instances share a fixed `_item_shape`, defined by subclasses.

This class provides:

- Memory-efficient storage of large arrays of structured items
- Broadcasting support: input arrays are validated but not strictly enforced to match shape
- Reshaping, resizing, and filtering operations
- Full NumPy interoperability (`__array__`, slicing, iteration)
- Safe subtyping: `Transformation`, `Rotation`, etc. can inherit and define their own item shape

Typical usage (with `_item_shape = (3,)` for vectors):

    >>> V = ItemsArray([[1, 2, 3], [4, 5, 6]])
    >>> V.shape
    (2,)
    >>> V.matrix
    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)

The base class can be used directly, or extended to represent more structured data
with operations like `rotate()`, `transform()`, etc.
"""

import numpy as np

# ====================================================================================================
# ItemsArray
# ====================================================================================================

class ItemsArray:

    FLOAT = np.float32
    __array_priority__ = 10.0  # Ensures NumPy defers to ItemsArray in mixed operations
    __slots__ = ("_mat",)
    _item_shape = (3,) # Array of vectors


    def __init__(self, mat: np.ndarray | list | tuple, *, copy: bool = True):
        """
        Initialize an ItemsArray from any array-like input.

        The input must be broadcastable to the expected item shape defined
        by the subclass via `_item_shape`. Typical examples include arrays
        of vectors, matrices, or quaternions.

        Parameters
        ----------
        mat : array_like
            Input data to be wrapped. Must be broadcastable to shape (..., *_item_shape).
            For example, if `_item_shape = (3,)`, acceptable inputs include:
            - [1, 2, 3]                         → broadcasted to shape (1, 3)
            - [[1, 2, 3], [4, 5, 6]]            → shape (2, 3)
            - np.ones((10, 1, 3))               → shape (10, 1, 3)

        copy : bool, default True
            If True, the input is copied. If False, a view is kept (when safe),
            which avoids unnecessary allocations.

        Raises
        ------
        ValueError
            If the input is not broadcastable to the required item shape.
        """
        dtype = self.FLOAT
        mat = np.asarray(mat, dtype=dtype)

        # Vérification : les dernières dimensions doivent pouvoir être broadcastées vers _item_shape
        item_ndim = len(self._item_shape)
        try:
            # On essaie de forcer le broadcast pour vérification
            mat = np.broadcast_to(mat, mat.shape[:-item_ndim] + self._item_shape)
        except ValueError:
            raise ValueError(f"Input not broadcastable to item shape {self._item_shape}")

        # Affectation (optionnellement copie)
        self._mat = mat.copy() if copy else mat


    # ------------------------------------------------------------------
    # Dunder helpers for convenience (optional but nice to have)
    # ------------------------------------------------------------------

    def __len__(self):
        return self._mat.shape[0]

    def __repr__(self):
        return f"<{type(self).__name__}(shape={self.shape}, dtype={self._mat.dtype})>"

    # Allow NumPy to treat us like an array when needed.
    def __array__(self, dtype=None):
        return np.asarray(self._mat, dtype=dtype)

    # ------------------------------------------------------------------
    # Shaping helpers
    # ------------------------------------------------------------------

    @property
    def is_scalar(self):
        return self._mat.shape == self._item_shape
    
    @property
    def _non_scalar_mat(self):
        if self.is_scalar:
            return self._mat[None, :]
        else:
            return self._mat
    
    def as_array(self, dtype=None) -> np.ndarray:
        """**View** on the internal array (no copy)."""
        return np.asarray(self._mat, dtype=dtype)

    @property
    def shape(self) -> tuple:
        """Batch shape (everything *except* the final item_shape)."""
        return self._mat.shape[:-len(self._item_shape)]

    @property
    def size(self) -> int:
        """Number of individual items."""
        # E.g. shape (A, B, 4, 4) → A×B matrices.
        return int(np.prod(self.shape))

    def reshape(self, *new_shape: int) -> "ItemsArray":
        """
        Reshape the batch dimensions of the array, preserving the item shape.

        The total number of items must remain unchanged. This is equivalent
        to `np.reshape()` on the batch dimensions only, i.e., the trailing
        `_item_shape` is preserved.

        Parameters
        ----------
        *new_shape : int
            New shape for the batch. Must satisfy `np.prod(new_shape) == self.size`.

        Returns
        -------
        ItemsArray
            A new reshaped view of the same data (no copy).

        Raises
        ------
        ValueError
            If the total number of items does not match.
        """
        # both tuple and ints are accepted
        if len(new_shape) == 1 and isinstance(new_shape[0], tuple):
            new_shape = new_shape[0]

        if np.prod(new_shape) != self.size:
            raise ValueError(f"{type(self).__name__}> Total size mismatch in reshape")
        
        new_mat = self._mat.reshape(*new_shape, *self._item_shape)
        return type(self)(new_mat, copy=False)

    def resize(self, *new_shape: int, fill: float = 0.) -> "ItemsArray":
        """
        Resize the batch to a new shape, preserving item shape.

        If the total number of items remains unchanged, the underlying array
        is simply reshaped (no allocation). Otherwise, a new array is created
        and filled with the given value. Existing data is preserved as much
        as possible (copied in row-major order).

        Parameters
        ----------
        *new_shape : int
            New shape for the batch (excluding the item shape). Can increase or decrease
            the number of items.

        fill : float, default 0.0
            Value used to initialize new items when the array is enlarged.

        Returns
        -------
        ItemsArray
            The current instance (resized in place).
        """
        # both tuple and ints are accepted
        if len(new_shape) == 1 and isinstance(new_shape[0], tuple):
            new_shape = new_shape[0]

        total_new = int(np.prod(new_shape))
        total_old = self.size

        if total_new == total_old:
            # Simple reshape view.
            self._mat = self._mat.reshape(*new_shape, *self._item_shape)
            return self

        # Allocate new array.
        new_arr = np.empty((*new_shape, *self._item_shape), dtype=self._mat.dtype)
        new_arr[...] = fill

        # Copy existing data as much as fits.
        flat_new = new_arr.reshape(-1, *self._item_shape)
        flat_old = self._mat.reshape(-1, *self._item_shape)
        flat_new[: min(total_old, total_new)] = flat_old[: min(total_old, total_new)]

        self._mat = new_arr
        return self
    
    def broadcast_to(self, shape):
        """
        Broadcast the array to a new batch shape.

        This is equivalent to `np.broadcast_to(...)`, preserving the item shape.
        No copy is made.

        Parameters
        ----------
        shape : tuple of int
            New shape for the batch dimensions.

        Returns
        -------
        ItemsArray
            A new view with broadcasted shape.
        """
        out = np.broadcast_to(self._mat, shape + self._item_shape)
        return type(self)(out, copy=False)

    # ------------------------------------------------------------------
    # Global
    # ------------------------------------------------------------------

    def copy(self):
        return type(self)(self._mat.copy(), copy=False)

    def astype(self, dtype):
        return type(self)(self._mat.astype(dtype), copy=False)

    # ------------------------------------------------------------------
    # Access to items
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        return type(self)(self._mat[key], copy=False)

    def __setitem__(self, key, value):
        self._mat[key] = np.asarray(value, dtype=self._mat.dtype)

    def __iter__(self):
        return (type(self)(x, copy=False) for x in self._mat)

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter(self, mask: np.ndarray, in_place: bool = False) -> "ItemsArray":
        """Filter transformations using a boolean mask.

        Parameters
        ----------
        mask : array_like (...,)
            Boolean mask matching the shape of the batch.
        in_place : bool, default False
            If True, modifies the current instance. Otherwise, returns a new one.

        Returns
        -------
        ItemsArray
            Filtered array (or self if in_place is True).
        """
        mask = np.asarray(mask)
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match batch shape {self.shape}")

        filtered = self._mat[mask]
        if in_place:
            self._mat = filtered
            return self
        else:
            return type(self)(filtered, copy=False)
        


