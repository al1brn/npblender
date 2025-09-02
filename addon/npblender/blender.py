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
Module Name: blender
Author: Alain Bernard
Version: 0.1.0
Created: 2022-06-29
Last updated: 2025-08-29

Summary:
This module provides an interface between `npblender` and Blender's internal API for Geometry Nodes.

It includes utilities to:
  - Access and manipulate mesh data blocks (e.g., Mesh, Curve, PointCloud)
  - Read and write geometry attributes from Blender objects
  - Create and update attribute layers compatible with Geometry Nodes
  - Detect attribute domains and data types

Usage example:
    >>> import blender
    >>> obj = blender.get_pbject("Cube")
    >>> me = blender.get_mesh("Cube")
"""

__all__ = [
    "get_collection", "create_collection", "get_temp_collection",
    "get_object", "get_evaluated", "delete_object", "get_empty",
    "create_mesh_object", "create_point_cloud_object", "create_curve_object", "create_text_object", 
    "duplicate", "object_type",
    "get_select_snapshot", "set_select_snapshot", 
    "mesh_to_curve", "curve_to_mesh", "mesh_to_point_cloud", "point_cloud_to_mesh",
    "int_array", "float_array", 
    "get_mesh", "get_point_cloud", "get_curve", "get_data",
    "clear_geometry", "get_mesh_vertices", "set_mesh_vertices", "temp_mesh_object",
    "BMesh", 
    "merge_by_distance", "merge_by_distance_2D", "remove_doubles", "shade_smooth",
    "get_point_cloud_points", "set_point_cloud_points",
    "create_material", "create_materials", "choose_material_type", "set_material", "change_material_image",
    "get_attributes", "attribute_exists", "get_attribute_info", "get_attribute_names",
    "get_attribute", "set_attribute", "delete_attribute", "create_attribute", 
    "create_float_attribute", "create_int_attribute", "create_bool_attribute", "create_vector_attribute",
    "create_vector2_attribute", "create_color_attribute", "create_byte_color_attribute",
    "create_quaternion_attribute", "create_matrix_attribute", 
    "get_temp_folder", "pil_to_bl_image", "pil_to_image", "pil_array_to_image", "get_image_node", 
    "markers", "fps", "marker_frame", "frtime", "frdur", "frame_at",
    "data_path_index", "get_fcurve", "fc_clear", "fc_set_kfs", "fc_set_keyframe", "kf_clear", 
    "set_key_frame", "get_value_at_frame",
    "KeyFrame", "FCurve",
    "shape_key_name", "has_shape_keys", "shape_keys_count", "shape_keys_clear", "get_key_block", 
    "is_viewport", "lock_interface", "render_engine"
    ]


import numpy as np
import bpy
import bmesh
import idprop
from mathutils import Vector

from pathlib import Path

from . constants import bfloat, bint, bbool, TYPES, BL_TYPES

# ====================================================================================================
# ====================================================================================================
# Get Blender things from spec. The spec can be the name of the thing itself
# ====================================================================================================
# ====================================================================================================

# ====================================================================================================
# Collection
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Get an existing collection
# ----------------------------------------------------------------------------------------------------

def get_collection(spec, halt=True):
    """Get a collection by its name or object.

    Args:
        spec (str or bpy.types.Collection): The name of the collection or the collection object itself.
        halt (bool, optional): If True, raise an exception if the collection is not found when `spec` is a string. Defaults to True.

    Returns:
        bpy.types.Collection: The requested collection.

    Raises:
        Exception: If `spec` is a string and the collection is not found and `halt` is True.
        Exception: If `spec` is not a string or a bpy.types.Collection object.
    """

    if isinstance(spec, str):
        coll = bpy.data.collections.get(spec)
        if coll is None:
            if halt:
                raise Exception(f"Collection '{spec}' not found")
        return coll

    elif isinstance(spec, bpy.types.Collection):
        return spec

    else:
        if halt:
            raise Exception(f"Collection name expected, not '{spec}'")
        return None

# ----------------------------------------------------------------------------------------------------
# Create a collection
# ----------------------------------------------------------------------------------------------------

def create_collection(spec, parent=None):
    """Creates a new collection or returns an existing one.

        If a string is provided for `spec`, the function checks if a collection with that name
        already exists. If it does, the existing collection is returned. If not, a new
        collection is created with the given name. The new collection is linked to the
        scene's master collection by default, or to a specified parent collection if provided.
        If a `bpy.types.Collection` object is provided for `spec`, the object is returned directly.

        Args:
            spec (str or bpy.types.Collection): The name of the collection to create or retrieve,
                or a `bpy.types.Collection` object.
            parent (str or bpy.types.Collection, optional): The parent collection to link the
                newly created collection under. Can be a collection name (str) or a
                `bpy.types.Collection` object. Defaults to None, which links to the scene's
                master collection.

        Returns:
            bpy.types.Collection: The created or retrieved collection.

        Raises:
            Exception: If `spec` is not a string or a `bpy.types.Collection` object.
            Exception: If `parent` is a string and the parent collection is not found (raised by `get_collection`).
    """

    if isinstance(spec, str):
        coll = bpy.data.collections.get(spec)
        if coll is None:
            coll = bpy.data.collections.new(spec)
            if parent is None:
                bpy.context.scene.collection.children.link(coll)
            else:
                get_collection(parent).children.link(coll)
        return coll

    elif isinstance(spec, bpy.types.Collection):
        return spec

    else:
        raise Exception(f"Collection name expected, not '{spec}'")

# ----------------------------------------------------------------------------------------------------
# Create a temporary collection
# ----------------------------------------------------------------------------------------------------

def get_temp_collection(name="npblender Temp"):
    """Get or create a temporary collection.

    Checks if a collection with the given name exists. If it does, returns the existing
    collection. If not, creates a new collection with the specified name, links it to
    the scene's master collection, and sets its `hide_render` property to True.

    Args:
        name (str, optional): The name of the temporary collection to get or create.
            Defaults to "npblender Temp".

    Returns:
        bpy.types.Collection: The existing or newly created temporary collection.
    """
    coll = bpy.data.collections.get(name, None)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
        coll.hide_render   = True
        #coll.hide_viewport = True
        #coll.hide_select   = True
    return coll

# ====================================================================================================
# Object
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Get an existing object
# ----------------------------------------------------------------------------------------------------

def get_object(spec, halt=True):
    """Get an object by its name or object reference.

    This function retrieves a Blender object based on either its name (string)
    or a direct reference to a `bpy.types.Object`. If a name is provided and
    the object is not found, an exception is raised depending on the `halt`
    parameter.

    Args:
        spec (str or bpy.types.Object): The name of the object to retrieve,
            or the object itself.
        halt (bool, optional): If True, raise an exception if the object is
            not found when `spec` is a string. Defaults to True.

    Returns:
        bpy.types.Object: The requested Blender object.

    Raises:
        Exception: If `spec` is a string and the object is not found and `halt` is True.
        Exception: If `spec` is neither a string nor a `bpy.types.Object`.
    """

    obj = None
    if isinstance(spec, str):
        obj = bpy.data.objects.get(spec)
        if obj is None:
            if halt:
                raise Exception(f"Object '{spec}' not found")

    elif isinstance(spec, bpy.types.Object):
        obj = spec

    else:
        if halt:
            raise Exception(f"Object or object name expected, not '{spec}'")
        return None

    return obj

# ----------------------------------------------------------------------------------------------------
# Get an evaluated object
# ----------------------------------------------------------------------------------------------------

def get_evaluated(spec):
    """Get the evaluated state of a Blender object.

    Retrieves a Blender object using `get_object` and returns its evaluated
    state based on the current dependency graph.

    Args:
        spec (str or bpy.types.Object): The name or reference of the object
            to retrieve and evaluate.

    Returns:
        bpy.types.Object: The evaluated Blender object.

    Raises:
        Exception: If the object specified by `spec` is not found when `halt`
            is True in the `get_object` call.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()

    return get_object(spec, halt=True).evaluated_get(depsgraph)

# ----------------------------------------------------------------------------------------------------
# Delete object
# ----------------------------------------------------------------------------------------------------

def delete_object(spec):
    """Deletes a Blender object by its name or reference.

    Retrieves the object using `get_object`. If the object is found, it is
    removed from Blender's data and unlinked from any collections. If the
    object is not found (and `get_object` is called with `halt=False`),
    the function returns silently.

    Args:
        spec (str or bpy.types.Object): The name or reference of the object
            to delete.
    """

    obj = get_object(spec, halt=False)
    if obj is None:
        return

    bpy.data.objects.remove(obj, do_unlink=True)

# ----------------------------------------------------------------------------------------------------
# Create an object

def create_object(name, collection=None, type='MESH', halt=True, **kwargs):
    """Create a new Blender object or return an existing one.

    If an object with the given `name` already exists, the function checks if its
    type matches the requested `type`. If they match, the existing object is
    returned. If they don't match and `halt` is True, an exception is raised.
    If `halt` is False, the existing object (of the different type) is returned.

    If no object with the given `name` exists, a new object of the specified `type`
    is created. The new object is linked to the specified `collection`. If no
    collection is specified, it is linked to the currently active collection.

    Supported object types: 'EMPTY', 'MESH', 'CURVE', 'FONT'.
    For 'FONT' type, the `text` keyword argument can be used to set the initial text.

    Args:
        name (str): The name for the object.
        collection (str or bpy.types.Collection, optional): The collection where the
            object should be created or linked. Can be a collection name (str)
            or a `bpy.types.Collection` object. Defaults to None, which uses the
            active collection.
        type (str, optional): The type of object to create ('MESH', 'EMPTY', 'CURVE', 'FONT').
            Case-insensitive. Defaults to 'MESH'.
        halt (bool, optional): If True, raise an exception if an object with the
            same name exists but is of a different type. Defaults to True.
        **kwargs: Additional keyword arguments. Currently used for 'FONT' type
            to set the initial `text`.

    Returns:
        bpy.types.Object: The existing or newly created Blender object.

    Raises:
        Exception: If an object with the same name exists but is of a different
            type and `halt` is True.
        Exception: If the specified `type` is not supported.
        Exception: If `collection` is a string and the collection is not found
            (raised by `get_collection`).
        Exception: If `collection` is not a string, `bpy.types.Collection`, or None
            (raised by `get_collection`).
    """    
    utype = type.upper()

    obj = get_object(name, halt=False)

    # Object already exists

    if obj is not None:
        if obj.type == utype:
            return obj
        
        if halt:
            raise Exception(f"Object named '{name}' already exists and is not a '{type}'")
        else:
            delete_object(obj.name)
            obj = None
        
    # Create the object
        
    if utype == 'EMPTY':
        obj = bpy.data.objects.new(name=name, object_data=None)

    elif utype == 'MESH':
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)

    elif utype in ['POINTCLOUD', 'CLOUD']:
        cloud = bpy.data.pointclouds.new(name)
        obj = bpy.data.objects.new(name, cloud)

    elif utype == 'CURVE':
        curve = bpy.data.curves.new("Curve", type='CURVE')
        curve.dimensions = '3D'
        obj  = bpy.data.objects.new(name, curve)

    elif utype == 'FONT':
        curve = bpy.data.curves.new("Curve", type='FONT')
        curve.body = kwargs.get('text', 'Text')
        obj   = bpy.data.objects.new(name, curve)

    else:
        raise Exception(f"Object type '{type}' not supported")
    
    # Link the object to the collection

    if collection is None:
        coll = bpy.context.collection
    else:
        coll = get_collection(collection)

    coll.objects.link(obj)

    return obj

# ----------------------------------------------------------------------------------------------------
# Create an empty object
# ----------------------------------------------------------------------------------------------------

def get_empty(name, collection=None, halt=False):
    """Get or create an empty object.

    This function is a convenience wrapper around `create_object` specifically
    for creating or retrieving objects of type 'EMPTY'.

    If an object with the given name already exists, it is returned. If no
    object with the name exists, a new empty object is created. The object
    is linked to the specified collection. If no collection is specified,
    it is linked to the currently active collection.

    Args:
        name (str): The name for the empty object.
        collection (str or bpy.types.Collection, optional): The collection where the
            empty object should be created or linked. Can be a collection name (str)
            or a `bpy.types.Collection` object. Defaults to None, which uses the
            active collection.
        halt (bool, optional): If True, raise an exception if an object with the
            same name exists but is of a different type (as handled by `create_object`).
            Defaults to False.

    Returns:
        bpy.types.Object: The existing or newly created empty object.

    Raises:
        Exception: Propagates exceptions from `create_object`, such as if `collection`
            is invalid or not found (when specified) or if `halt` is True and an
            object of a different type exists with the same name.
    """
    return create_object(name, collection=collection, type='EMPTY', halt=halt)

# ----------------------------------------------------------------------------------------------------
# Create a mesh object
# ----------------------------------------------------------------------------------------------------

def create_mesh_object(name, collection=None, halt=False):
    """Create a new mesh object or return an existing one.

    This function is a convenience wrapper around `create_object` specifically
    for creating or retrieving objects of type 'MESH'.

    If an object with the given name already exists, it is returned. If no
    object with the name exists, a new mesh object is created. The object
    is linked to the specified collection. If no collection is specified,
    it is linked to the currently active collection.

    Args:
        name (str): The name for the mesh object.
        collection (str or bpy.types.Collection, optional): The collection where the
            mesh object should be created or linked. Can be a collection name (str)
            or a `bpy.types.Collection` object. Defaults to None, which uses the
            active collection. Handled by `create_object`.
        halt (bool, optional): If True, raise an exception if an object with the
            same name exists but is of a different type (as handled by `create_object`).
            Defaults to False.

    Returns:
        bpy.types.Object: The existing or newly created mesh object.

    Raises:
        Exception: Propagates exceptions from `create_object`, such as if `collection`
            is invalid or not found (when specified) or if `halt` is True and an
            object of a different type exists with the same name.
    """
    return create_object(name, collection=collection, type='MESH', halt=halt)

# ----------------------------------------------------------------------------------------------------
# Create a point cloud object
# ----------------------------------------------------------------------------------------------------

def create_point_cloud_object(name, collection=None, halt=False):
    """Create a new point cloud object or return an existing one.

    This function is a convenience wrapper around `create_object` specifically
    for creating or retrieving objects of type 'MESH'.

    If an object with the given name already exists, it is returned. If no
    object with the name exists, a new mesh object is created. The object
    is linked to the specified collection. If no collection is specified,
    it is linked to the currently active collection.

    Args:
        name (str): The name for the mesh object.
        collection (str or bpy.types.Collection, optional): The collection where the
            mesh object should be created or linked. Can be a collection name (str)
            or a `bpy.types.Collection` object. Defaults to None, which uses the
            active collection. Handled by `create_object`.
        halt (bool, optional): If True, raise an exception if an object with the
            same name exists but is of a different type (as handled by `create_object`).
            Defaults to False.

    Returns:
        bpy.types.Object: The existing or newly created mesh object.

    Raises:
        Exception: Propagates exceptions from `create_object`, such as if `collection`
            is invalid or not found (when specified) or if `halt` is True and an
            object of a different type exists with the same name.
    """
    return create_object(name, collection=collection, type='POINTCLOUD', halt=halt)


# ----------------------------------------------------------------------------------------------------
# Create a curve object
# ----------------------------------------------------------------------------------------------------

def create_curve_object(name, collection=None, halt=False):
    """Create a new curve object or return an existing one.

    This function is a convenience wrapper around `create_object` specifically
    for creating or retrieving objects of type 'CURVE'.

    If an object with the given name already exists, it is returned. If no
    object with the name exists, a new curve object is created. The object
    is linked to the specified collection. If no collection is specified,
    it is linked to the currently active collection.

    Args:
        name (str): The name for the curve object.
        collection (str or bpy.types.Collection, optional): The collection where the
            curve object should be created or linked. Can be a collection name (str)
            or a `bpy.types.Collection` object. Defaults to None, which uses the
            active collection. Handled by `create_object`.
        halt (bool, optional): If True, raise an exception if an object with the
            same name exists but is of a different type (as handled by `create_object`).
            Defaults to False.

    Returns:
        bpy.types.Object: The existing or newly created curve object.

    Raises:
        Exception: Propagates exceptions from `create_object`, such as if `collection`
            is invalid or not found (when specified) or if `halt` is True and an
            object of a different type exists with the same name.
    """
    return create_object(name, collection=collection, type='CURVE', halt=halt)


# ----------------------------------------------------------------------------------------------------
# Create a text object
# ----------------------------------------------------------------------------------------------------

def create_text_object(name, text="Text", collection=None, halt=False):
    """Create a new text object or return an existing one.

    This function is a convenience wrapper around `create_object` specifically
    for creating or retrieving objects of type 'FONT'.

    If an object with the given name already exists, it is returned. If no
    object with the name exists, a new text object is created with the specified
    initial text. The object is linked to the specified collection. If no
    collection is specified, it is linked to the currently active collection.

    Args:
        name (str): The name for the text object.
        text (str, optional): The initial text content for a new text object.
            Defaults to "Text".
        collection (str or bpy.types.Collection, optional): The collection where the
            text object should be created or linked. Can be a collection name (str)
            or a `bpy.types.Collection` object. Defaults to None, which uses the
            active collection. Handled by `create_object`.
        halt (bool, optional): If True, raise an exception if an object with the
            same name exists but is of a different type (as handled by `create_object`).
            Defaults to False.

    Returns:
        bpy.types.Object: The existing or newly created text object.

    Raises:
        Exception: Propagates exceptions from `create_object`, such as if `collection`
            is invalid or not found (when specified) or if `halt` is True and an
            object of a different type exists with the same name.
    """
    return create_object(name, collection=collection, type='FONT', halt=halt, text=text)

# ----------------------------------------------------------------------------------------------------
# Duplicate an object
# ----------------------------------------------------------------------------------------------------

def duplicate(spec, data=True, actions=True, collection=None):
    """Duplicate a Blender object.

    Creates a copy of the specified object. The duplication behavior for
    object data and actions can be controlled via arguments. The duplicate
    can optionally be linked to a specific collection.

    If no collection is specified, the function attempts to find the original
    object's collection and link the duplicate there. If the original object
    is in multiple collections, the duplicate will be linked to the first one found.

    Args:
        spec (str or bpy.types.Object): The object to duplicate, specified by name
            or object reference.
        data (bool, optional): If True, duplicate the object's data (mesh, curve, etc.).
            If False, the duplicate will share data with the original. Defaults to True.
        actions (bool, optional): If True, duplicate the object's action (animation data).
            If False, the duplicate will share action data with the original. Defaults to True.
        collection (str or bpy.types.Collection, optional): The collection where the
            duplicate object should be linked. Can be a collection name (str) or a
            `bpy.types.Collection` object. If None, the duplicate is linked to the
            original object's collection (if found) or the scene's master collection
            implicitly via `get_collection`'s default behavior if the original object
            is not found in any collection. Defaults to None.

    Returns:
        bpy.types.Object: The newly created duplicate object.

    Raises:
        Exception: If the object specified by `spec` is not found (propagated from `get_object`).
        Exception: If a specified `collection` is not found or invalid (propagated from `get_collection`).
    """

    obj = get_object(spec)

    obj_copy = obj.copy()

    if data:
        obj_copy.data = obj_copy.data.copy()

    if actions and obj_copy.animation_data is not None and obj_copy.animation_data.action is not None:
        obj_copy.animation_data.action = obj_copy.animation_data.action.copy()

    if collection is None:
        for coll in bpy.data.collections:
            if obj.name in coll.objects:
                collection = coll

    if collection is not None:
        get_collection(collection, halt=True).objects.link(obj_copy)

    return obj_copy

# ----------------------------------------------------------------------------------------------------
# Object type: Empty of name to data type
# ----------------------------------------------------------------------------------------------------

def object_type(spec):
    """Get the type of the specified Blender object's data.

    If the object has no data (e.g., an Empty object), returns 'Empty'.
    Otherwise, returns the name of the object's data type (e.g., 'Mesh', 'Curve', 'Font').

    Args:
        spec (str or bpy.types.Object): The object to get the data type from,
            specified by name or object reference.

    Returns:
        str: The name of the object's data type, or 'Empty' if the object has no data.

    Raises:
        Exception: If the object specified by `spec` is not found or invalid
            (propagated from `get_object`).
    """


    obj = get_object(spec)
    if obj.data is None:
        return 'Empty'
    else:
        return type(obj.data).__name__

# ----------------------------------------------------------------------------------------------------
# Snapshot
# ----------------------------------------------------------------------------------------------------

def get_select_snapshot():
    """Get a snapshot of the current object selection state.

    Creates a dictionary where keys are object names and values are their
    selection status (True if selected, False otherwise). It also includes
    the currently active object under the special key "$$$ACTIVE OBJECT$$$".

    Returns:
        dict: A dictionary containing the selection status of all objects
              and the active object.
    """
    d = {obj.name: obj.select_get() for obj in bpy.data.objects}
    d["$$$ACTIVE OBJECT$$$"] = bpy.context.active_object

    return d

def set_select_snapshot(d):
    """Set the object selection state based on a snapshot dictionary.

    Restores the selection state of objects and sets the active object
    based on the provided dictionary, which is expected to be in the
    format generated by `get_select_snapshot`.

    Args:
        d (dict): A dictionary containing object names as keys and their
                  selection status (bool) as values. The dictionary should
                  also contain the key "$$$ACTIVE OBJECT$$$" with the
                  bpy.types.Object to be set as the active object.
    """
    for name, value in d.items():
        if name == "$$$ACTIVE OBJECT$$$":
            bpy.context.view_layer.objects.active = value
        else:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.select_set(value)

# ====================================================================================================
# Conversions
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Mesh to Curve
# ----------------------------------------------------------------------------------------------------

def mesh_to_curve(spec):
    """Convert a mesh object to a curve object.

    Retrieves the specified object using `get_object`. If the object is a mesh,
    it converts it to a curve object using Blender's built-in operator
    (`bpy.ops.object.convert`). The original mesh object is replaced in the
    scene by the new curve object. The function temporarily modifies the
    selection state and active object during the conversion process, restoring
    them afterwards using `get_select_snapshot` and `set_select_snapshot`.

    If the specified object is not a mesh, the function does nothing and
    returns the object as is.

    Args:
        spec (str or bpy.types.Object): The mesh object to convert, specified
            by name or object reference. Handled by `get_object`.

    Returns:
        bpy.types.Object: The converted curve object if the original was a mesh,
            otherwise the original object unchanged. Note that if conversion occurs,
            this returned object reference will point to the newly created curve
            object that replaced the original mesh in the scene.

    Raises:
        Exception: If the object specified by `spec` is not found or invalid
            (propagated from `get_object`).
    """    
    obj = get_object(spec)
    if object_type(obj) == 'Mesh':
        snapshot = get_select_snapshot()

        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='CURVE')

        set_select_snapshot(snapshot)

    return obj

# ----------------------------------------------------------------------------------------------------
# Curve to Mesh
# ----------------------------------------------------------------------------------------------------

def curve_to_mesh(spec):
    """Convert a curve object to a mesh object.

    Retrieves the specified object using `get_object`. If the object is a curve,
    it converts it to a mesh object using Blender's built-in operator
    (`bpy.ops.object.convert`). The original curve object is replaced in the
    scene by the new mesh object. The function temporarily modifies the
    selection state and active object during the conversion process, restoring
    them afterwards using `get_select_snapshot` and `set_select_snapshot`.

    If the specified object is not a curve, the function does nothing and
    returns the object as is.

    Args:
        spec (str or bpy.types.Object): The curve object to convert, specified
            by name or object reference. Handled by `get_object`.

    Returns:
        bpy.types.Object: The converted mesh object if the original was a curve,
            otherwise the original object unchanged. Note that if conversion occurs,
            this returned object reference will point to the newly created mesh
            object that replaced the original curve in the scene.

    Raises:
        Exception: If the object specified by `spec` is not found or invalid
            (propagated from `get_object`).
    """
    obj = get_object(spec)
    if object_type(obj) == 'Curve':
        snapshot = get_select_snapshot()

        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='MESH')

        set_select_snapshot(snapshot)

    return obj

# ----------------------------------------------------------------------------------------------------
# Mesh Point Cloud conversion
# ----------------------------------------------------------------------------------------------------

def mesh_to_point_cloud(spec):
    """Convert a mesh object to a point cloud object.

    Retrieves the specified object using `get_object`. If the object is a curve,
    it converts it to a mesh object using Blender's built-in operator
    (`bpy.ops.object.convert`). The original curve object is replaced in the
    scene by the new mesh object. The function temporarily modifies the
    selection state and active object during the conversion process, restoring
    them afterwards using `get_select_snapshot` and `set_select_snapshot`.

    If the specified object is not a curve, the function does nothing and
    returns the object as is.

    Args:
        spec (str or bpy.types.Object): The curve object to convert, specified
            by name or object reference. Handled by `get_object`.

    Returns:
        bpy.types.Object: The converted mesh object if the original was a curve,
            otherwise the original object unchanged. Note that if conversion occurs,
            this returned object reference will point to the newly created mesh
            object that replaced the original curve in the scene.

    Raises:
        Exception: If the object specified by `spec` is not found or invalid
            (propagated from `get_object`).
    """
    obj = get_object(spec)
    if object_type(obj) == 'Mesh':
        snapshot = get_select_snapshot()

        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='POINTCLOUD')

        set_select_snapshot(snapshot)

    return obj

# ----------------------------------------------------------------------------------------------------
# Point Cloud to mesh conversion
# ----------------------------------------------------------------------------------------------------

def point_cloud_to_mesh(spec):
    """Convert a point cloud object to a mesh object.

    Retrieves the specified object using `get_object`. If the object is a curve,
    it converts it to a mesh object using Blender's built-in operator
    (`bpy.ops.object.convert`). The original curve object is replaced in the
    scene by the new mesh object. The function temporarily modifies the
    selection state and active object during the conversion process, restoring
    them afterwards using `get_select_snapshot` and `set_select_snapshot`.

    If the specified object is not a curve, the function does nothing and
    returns the object as is.

    Args:
        spec (str or bpy.types.Object): The curve object to convert, specified
            by name or object reference. Handled by `get_object`.

    Returns:
        bpy.types.Object: The converted mesh object if the original was a curve,
            otherwise the original object unchanged. Note that if conversion occurs,
            this returned object reference will point to the newly created mesh
            object that replaced the original curve in the scene.

    Raises:
        Exception: If the object specified by `spec` is not found or invalid
            (propagated from `get_object`).
    """
    obj = get_object(spec)
    if object_type(obj) == 'PointCloud':
        snapshot = get_select_snapshot()

        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='MESH')

        set_select_snapshot(snapshot)

    return obj

# ====================================================================================================
# Access to typed data
# ====================================================================================================

def int_array(a):
    a = np.asarray(a, dtype=np.int32)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    assert(a.flags['C_CONTIGUOUS'])
    return a

def float_array(a):
    a = np.asarray(a, dtype=np.float32)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    assert(a.flags['C_CONTIGUOUS'])
    return a

# ----------------------------------------------------------------------------------------------------
# Mesh data
# ----------------------------------------------------------------------------------------------------

def get_mesh(spec, halt=True):
    """Get the Mesh data of an object.

    Retrieves the mesh data block associated with a Blender object.
    The input `spec` can be a string (object name), a `bpy.types.Object`,
    or a `bpy.types.Mesh` directly.

    Args:
        spec (str or bpy.types.Object or bpy.types.Mesh): The data specifier.
            Can be the name of an object, an object instance, or a mesh instance.
        halt (bool, optional): If True, raise an exception if the object is not
            found (when `spec` is a string) or if the object found does not
            have mesh data. If False, return None in these cases. Defaults to True.

    Returns:
        bpy.types.Mesh or None: The Mesh data block if found and the object has
            mesh data. Returns None if the object is not found or does not have
            mesh data and `halt` is False.

    Raises:
        Exception: If `spec` is a string and the object is not found and `halt` is True.
        Exception: If the object found does not have mesh data and `halt` is True.
        Exception: If `spec` is not a valid type (string, Object, or Mesh).
    """

    if isinstance(spec, bpy.types.Mesh):
        return spec

    obj = get_object(spec, halt=halt)
    if obj is None:
        return None

    if not isinstance(obj.data, bpy.types.Mesh):
        raise Exception(f"Object '{obj.name}' is not a Mesh")

    return obj.data

# ----------------------------------------------------------------------------------------------------
# Point Cloud data
# ----------------------------------------------------------------------------------------------------

def get_point_cloud(spec, halt=True):
    """Get the PointCloud data of an object.

    Retrieves the mesh data block associated with a Blender object.
    The input `spec` can be a string (object name), a `bpy.types.Object`,
    or a `bpy.types.PointCloud` directly.

    Args:
        spec (str or bpy.types.Object or bpy.types.PointCloud): The data specifier.
            Can be the name of an object, an object instance, or a mesh instance.
        halt (bool, optional): If True, raise an exception if the object is not
            found (when `spec` is a string) or if the object found does not
            have mesh data. If False, return None in these cases. Defaults to True.

    Returns:
        bpy.types.PointCloud or None: The PointCloud data block if found and the object has
            cloud data. Returns None if the object is not found or does not have
            cloud data and `halt` is False.

    Raises:
        Exception: If `spec` is a string and the object is not found and `halt` is True.
        Exception: If the object found does not have cloud data and `halt` is True.
        Exception: If `spec` is not a valid type (string, Object, or PointCloud).
    """

    if isinstance(spec, bpy.types.PointCloud):
        return spec

    obj = get_object(spec, halt=halt)
    if obj is None:
        return None

    if not isinstance(obj.data, bpy.types.PointCloud):
        raise Exception(f"Object '{obj.name}' is not a Point Cloud")

    return obj.data

# ----------------------------------------------------------------------------------------------------
# Curve data
# ----------------------------------------------------------------------------------------------------

def get_curve(spec, halt=True):
    """Get the Curve data of an object.

    Retrieves the curve data block associated with a Blender object.
    The input `spec` can be a string (object name), a `bpy.types.Object`,
    or a `bpy.types.Curve` or `bpy.types.SurfaceCurve` directly.

    Args:
        spec (str | bpy.types.Object | bpy.types.Curve | bpy.types.SurfaceCurve): The data specifier.
            Can be the name of an object, an object instance, or a curve instance.
        halt (bool, optional): If True, raise an exception if the object is not
            found (when `spec` is a string) or if the object found does not
            have curve data. If False, return None in these cases. Defaults to True.

    Returns:
        bpy.types.Curve | bpy.types.SurfaceCurve | None: The Curve data block if found and the object has
            curve data. Returns None if the object is not found or does not have
            curve data and `halt` is False.

    Raises:
        Exception: If `spec` is a string and the object is not found and `halt` is True.
        Exception: If the object found does not have curve data and `halt` is True.
        Exception: If `spec` is not a valid type (string, Object, Curve, or SurfaceCurve).
    """
    if isinstance(spec, (bpy.types.Curve, bpy.types.SurfaceCurve)):
        return spec

    obj = get_object(spec, halt=halt)
    if obj is None:
        return None

    if not isinstance(obj.data, (bpy.types.Curve, bpy.types.SurfaceCurve)):
        raise Exception(f"Object '{obj.name}' is not a Curve")

    return obj.data

# ----------------------------------------------------------------------------------------------------
# Object data
# ----------------------------------------------------------------------------------------------------

def get_data(spec, halt=True):
    """Get the data block of a Blender object.

    Retrieves the data block (like Mesh or Curve) associated with a Blender object.
    The input `spec` can be a string (object name), a `bpy.types.Object`,
    or a `bpy.types.Mesh` or `bpy.types.Curve` directly.

    Args:
        spec (str | bpy.types.Object | bpy.types.Mesh | bpy.types.Curve): The data specifier.
            Can be the name of an object, an object instance, or a data block instance.
        halt (bool, optional): If True, raise an exception if the object is not
            found (when `spec` is a string). Defaults to True.

    Returns:
        bpy.types.Mesh | bpy.types.Curve: The data block (Mesh or Curve) if found.

    Raises:
        Exception: If `spec` is a string and the object is not found and `halt` is True
                   (propagated from `get_object`).
        AttributeError: If the object found does not have a `.data` attribute.
    """

    if isinstance(spec, (bpy.types.Mesh, bpy.types.Curve, bpy.types.PointCloud)):
        return spec

    else:
        return get_object(spec).data

# ====================================================================================================
# Clear Mesh or Curve geometry
# ====================================================================================================

def clear_geometry(spec):
    """Clear the geometry data of a Blender object if it is a Mesh or Curve.

    If the object specified by `spec` is found and its data is a Mesh,
    its geometry is cleared. If its data is a Curve, its splines are cleared.
    If the object is not found or is of a different type, the function does nothing.

    Args:
        spec (str or bpy.types.Object): The object to clear geometry from,
            specified by name or object reference. Handled by `get_object`.
    """

    obj = get_object(spec, halt=False)
    if obj is None:
        return

    data = obj.data
    if isinstance(data, bpy.types.Mesh):
        data.clear_geometry()

    elif isinstance(data, bpy.types.PointCloud):
        raise Exception("Clearing geometry for a PointCloud is not supported.")

    elif isinstance(data, bpy.types.Curve):
        data.splines.clear()


# ====================================================================================================
# Mesh
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Get the vertices
# ----------------------------------------------------------------------------------------------------

def get_mesh_vertices(spec):
    """Get the vertices of a Mesh object.

    Retrieves the mesh data block from the specified object or data specifier
    and extracts its vertex coordinates into a NumPy array.

    Args:
        spec (str | bpy.types.Object | bpy.types.Mesh): The data specifier.
            Can be the name of an object, an object instance, or a mesh instance.
            Handled by the internal `get_mesh` function.

    Returns:
        np.ndarray: A NumPy array of shape (n, 3) where n is the number of
            vertices, containing the (x, y, z) coordinates of each vertex.

    Raises:
        Exception: Propagates exceptions from `get_mesh`, such as if the object
                   is not found (when `spec` is a string) or if the object found
                   does not have mesh data.
    """

    mesh = get_mesh(spec)

    n = len(mesh.vertices)
    a = np.empty(n * 3, bfloat)
    mesh.vertices.foreach_get('co', a)
    return np.reshape(a, (n, 3))

# ----------------------------------------------------------------------------------------------------
# Set the vertices
# ----------------------------------------------------------------------------------------------------

def set_mesh_vertices(spec, verts):
    """Set the vertex coordinates of a Mesh object.

    This function updates the coordinates of existing vertices in a mesh.
    It does not change the number of vertices, edges, or faces.
    The provided `verts` array must contain `n * 3` float values, where `n`
    is the current number of vertices in the mesh. The values should represent
    the (x, y, z) coordinates for each vertex in order.

    Args:
        spec (str | bpy.types.Object | bpy.types.Mesh): The data specifier for the mesh.
            Can be the name of an object, an object instance, or a mesh instance.
            Handled by the internal `get_mesh` function.
        verts (Iterable[float]): An iterable (like a list or NumPy array)
            containing the new vertex coordinates as a flat sequence of floats
            (x1, y1, z1, x2, y2, z2, ...). The total number of elements must be
            three times the number of vertices in the mesh.

    Raises:
        Exception: If the number of elements in `verts` is not exactly three
                   times the number of vertices in the mesh.
        Exception: Propagates exceptions from `get_mesh`, such as if the object
                   is not found (when `spec` is a string) or if the object found
                   does not have mesh data.
    """

    mesh = get_mesh(spec)

    n = len(mesh.vertices)
    if n != np.size(verts)//3:
        raise Exception(f"set_mesh_vertices error: the number of vertices to write ({np.size(verts)//3}) is different" +
                         f" from the number of vertices in mesh '{spec}' which is ({n}).")

    a = np.reshape(verts, n*3).astype(bfloat)
    mesh.vertices.foreach_set('co', a)

    mesh.update()

# ----------------------------------------------------------------------------------------------------
# Temp object
# ----------------------------------------------------------------------------------------------------

def temp_mesh_object(name="Mesh"):
    """Create a temporary mesh object.

    This function creates a new mesh object with a name prefixed by "npblender Temp - "
    and links it to the temporary collection obtained from `get_temp_collection()`.
    This is useful for creating intermediate geometry that might be cleaned up later.

    Args:
        name (str, optional): The base name for the temporary object. The final
            object name will be "npblender Temp - " followed by this name.
            Defaults to "Mesh".

    Returns:
        bpy.types.Object: The newly created temporary mesh object.
    """
    return create_mesh_object(f"npblender Temp - {name}", collection=get_temp_collection())

# ====================================================================================================
# BMesh class
# ====================================================================================================

class BMesh:
    """Context manager for accessing and modifying an object's mesh data using bmesh.

    This class provides a convenient way to work with bmesh, ensuring that
    the bmesh data is properly created from the object's mesh upon entering
    the context and written back to the object's mesh and freed upon exiting.

    Example:
        obj = get_object("MyMeshObject")
        with BMesh(obj) as bm:
            # Modify the mesh using bmesh operations
            bm.verts.new((0, 0, 0))
            # ... other bmesh operations ...
    """    
    def __init__(self, obj):
        """Initialize the BMesh context manager.

        Retrieves the specified Blender object and stores it.

        Args:
            obj (str or bpy.types.Object): The Blender object whose mesh data
                will be accessed and modified. Handled by `get_object`.

        Raises:
            Exception: If the object specified by `obj` is not found or
                       is invalid (propagated from `get_object`).
        """
        self.obj = get_object(obj, halt=True)

    def __enter__(self):
        self.bm = bmesh.new()
        self.bm.from_mesh(self.obj.data)

        return self.bm

    def __exit__(self, *args):
        self.bm.to_mesh(self.obj.data)
        self.bm.free()
        del self.bm

# ====================================================================================================
# Mesh operations
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Merge by distance
# ----------------------------------------------------------------------------------------------------

def merge_by_distance(vertices, precision=.0001):
    """
    Remove near-duplicate 3D points within a tolerance dx by keeping the first one.

    Parameters:
        vertices (np.ndarray): Array of shape (n, 3), float32.
        precision (float = .0001): Distance tolerance for merging points.

    Returns:
        new_vertices (np.ndarray): Unique points (first in each cell).
        inverse (np.ndarray): Mapping from original points to merged ones.
    """
    # Quantize points into grid cells of size precision
    #quant = np.floor(vertices / precision).astype(np.int32)
    quant = np.round(vertices / precision).astype(np.int32)
    

    # Encode 3D grid indices into unique 1D keys
    max_coord = np.max(quant, axis=0) + 1
    keys = (
        quant[:, 0].astype(np.int64) * max_coord[1] * max_coord[2] +
        quant[:, 1].astype(np.int64) * max_coord[2] +
        quant[:, 2].astype(np.int64)
    )

    # Get the first index of each unique key (i.e., first point in each cell)
    _, unique_indices, inverse = np.unique(keys, return_index=True, return_inverse=True)

    # Extract fused points (no averaging, just first occurrence)
    new_vertices = vertices[unique_indices]

    return new_vertices, inverse

# ----------------------------------------------------------------------------------------------------
# Merge by distance, 2D version
# ----------------------------------------------------------------------------------------------------

def merge_by_distance_2D(vertices, precision=.0001):
    """
    Group 2D points that are within 'precision' distance from each other,
    keeping only the first point per group (no averaging).

    Parameters:
        vertices (np.ndarray): Array of shape (n, 2), coordinates of points.
        precision (float): Distance threshold for merging nearby points.

    Returns:
        new_vertices (np.ndarray): Deduplicated points (first per cell).
        inv (np.ndarray): Mapping from original points to new_vertices indices.
    """

    # Step 1: Quantize the coordinates to a grid of size 'precision'
    # Each point is mapped to a grid cell (i, j) = floor(x / dx, y / dx)
    quant = np.round(vertices / precision).astype(np.int32)

    # Step 2: Encode each grid cell as a single unique integer key
    # This creates a 1D "hash" of the 2D grid cell (i, j)
    base = np.max(quant) + 1  # base large enough to avoid collisions
    keys = quant[:, 0] * base + quant[:, 1]

    # Step 3: Use np.unique to find:
    # - the unique grid cells
    # - the first occurrence index (idx) for each unique key
    # - the inverse map (inv): for each original point, index into new_vertices
    _, idx, inv = np.unique(keys, return_index=True, return_inverse=True)

    # Step 4: Select the first point from each unique cell
    new_vertices = vertices[idx]

    return new_vertices, inv

# ----------------------------------------------------------------------------------------------------
# Remove doubles
# ----------------------------------------------------------------------------------------------------

def remove_doubles(spec, threshold=0.0001):
    """Remove duplicate vertices from a mesh object.

    Retrieves the specified object using `get_object`. If the object has mesh
    data, it uses a BMesh context manager to access the mesh and applies the
    `bmesh.ops.remove_doubles` operator to merge vertices within the specified
    distance threshold.

    Args:
        spec (str or bpy.types.Object): The object to remove doubles from,
            specified by name or object reference. Handled by `get_object`.
        threshold (float, optional): The maximum distance between vertices
            to be merged. Defaults to 0.0001.

    Returns:
        bpy.types.Object: The object after the remove doubles operation has
            been applied.

    Raises:
        Exception: If the object specified by `spec` is not found or invalid
            (propagated from `get_object`).
        AttributeError: If the object found does not have mesh data (as BMesh
            context manager expects a mesh object).
    """
    obj = get_object(spec, halt=True)
    with BMesh(obj) as bm:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=threshold)
    return obj

# ----------------------------------------------------------------------------------------------------
# Faces
# ----------------------------------------------------------------------------------------------------

def shade_smooth(obj, smooth=True):
    """Sets smooth shading on the polygons (faces) of a mesh object.

    This function enables or disables smooth shading by modifying the
    'use_smooth' property of all polygons in the mesh.

    args:
        obj (str or bpy.types.Object):
            The target object or its name. Must be a mesh.
        smooth (bool, optional):
            If True (default), enables smooth shading; if False, sets flat shading.

    returns:
        None

    raises:
        TypeError:
            If the resolved object is not of type 'MESH'.
        ValueError:
            If the object could not be found.
    """
    obj = get_object(obj)
    a = [smooth] * len(obj.data.polygons)
    obj.data.polygons.foreach_set('use_smooth', a)


# ====================================================================================================
# Point Cloud
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Get the points
# ----------------------------------------------------------------------------------------------------

def get_point_cloud_points(spec):
    """Get the points of a PointCloud object.

    Retrieves the point cloud data block from the specified object or data specifier
    and extracts its points coordinates into a NumPy array.

    Args:
        spec (str | bpy.types.Object | bpy.types.PointCloud): The data specifier.
            Can be the name of an object, an object instance, or a mesh instance.
            Handled by the internal `get_mesh` function.

    Returns:
        np.ndarray: A NumPy array of shape (n, 3) where n is the number of
            vertices, containing the (x, y, z) coordinates of each point.

    Raises:
        Exception: Propagates exceptions from `get_point_cloud`, such as if the object
                   is not found (when `spec` is a string) or if the object found
                   does not have point cloud data.
    """

    cloud = get_point_cloud(spec)

    n = len(cloud.points)
    a = np.empty(n * 3, bfloat)
    cloud.points.foreach_get('co', a)
    return np.reshape(a, (n, 3))

# ----------------------------------------------------------------------------------------------------
# Set the points
# ----------------------------------------------------------------------------------------------------

def set_point_cloud_points(spec, points):
    """Set the points coordinates of a Point Cloud object.

    This function updates the coordinates of existing points in a point cloud.
    It does not change the number of points.
    The provided `points` array must contain `n * 3` float values, where `n`
    is the current number of points in the point cloud. The values should represent
    the (x, y, z) coordinates for each point in order.

    Args:
        spec (str | bpy.types.Object | bpy.types.PointCloud): The data specifier for the point cloud.
            Can be the name of an object, an object instance, or a pointcloud instance.
            Handled by the internal `get_point_cloud` function.
        verts (Iterable[float]): An iterable (like a list or NumPy array)
            containing the new vertex coordinates as a flat sequence of floats
            (x1, y1, z1, x2, y2, z2, ...). The total number of elements must be
            three times the number of vertices in the mesh.

    Raises:
        Exception: If the number of elements in `points` is not exactly three
                   times the number of points in the point cloud.
        Exception: Propagates exceptions from `get_point_cloud`, such as if the object
                   is not found (when `spec` is a string) or if the object found
                   does not have point cloud data.
    """

    cloud = get_point_cloud(spec)

    n = len(cloud.points)
    if n != np.size(points)//3:
        raise Exception(f"set_point_cloud_points error: the number of points to write ({np.size(points)//3}) is different" +
                         f" from the number of points in the point cloud '{spec}' which is ({n}).")

    a = np.reshape(points, n*3).astype(bfloat)
    cloud.points.foreach_set('co', a)

# ====================================================================================================
# Materials
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Create a new material
# ----------------------------------------------------------------------------------------------------

def create_material(name, reset=False, **kwargs):
    """ Create a material with parameters passed as keyword arguments.

    This simple method allows to specified the values of the input sockets of the Principled BSDF.
    The following example create a blue material.

    ``` python
    mat = create_material("Blue Material", color=mathutils.Color((0, 0, 1)))
    ```
    Args:
    - name (str) : name of the material to create
    - reset (bool=False) : reset the material if already exists (True) or let it unchanged (False).
    - **kwargs : key word arguments to specify the values of the entries in the BSDF Node

    Returns:
    - Material
    """

    if isinstance(name, bpy.types.Material):
        mat = name
        if not reset:
            return mat

    else:
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)

        else:
            if not reset:
                return mat

    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    bsdf = None
    for node in nodes:
        if isinstance(node, bpy.types.ShaderNodeBsdfPrincipled):
            bsdf = node
            break
    if bsdf is None:
        print(f"Warning: ShaderNodeBsdfPrincipled node not found for material '{name}'")
        return mat

    for sock_name, sock_value in kwargs.items():
        found = False
        for socket in bsdf.inputs:
            if sock_name.lower() in [socket.name.lower().replace(' ', '_'), socket.name.lower()]:
                if isinstance(socket.default_value, bpy.types.bpy_prop_array):
                    a = tuple(sock_value)
                    if len(socket.default_value) == 4 and len(a) == 3:
                        socket.default_value = a + (1,)
                    elif len(socket.default_value) == 3 and len(a) == 4:
                        socket.default_value = a[:-1]
                    elif len(socket.default_value) == len(a):
                        socket.default_value = a
                    else:
                        raise RuntimeError(f"create_material error: socket '{sock_name}' takes a vector of len {len(socket.default_value)}, but the parameter len is {len(a)}!")
                else:
                    socket.default_value = sock_value

                found = True
                break

        if not found:
            print([sock.name for sock in bsdf.inputs])
            raise RuntimeError(f"create_material error: Socket '{sock_name}' not found")

    return mat

# ----------------------------------------------------------------------------------------------------
# Create a series of materials
# ----------------------------------------------------------------------------------------------------

def create_materials(mat_specs, reset=False, return_names=True):
    """ Create new materials as specified.

    Call the function **create_material** as specified in the **mat_specs** argument.

    Args:
    - mat_specs (dict {mat_name: mat_spec}) where mat_spec is a dictionary passe to **create_material**
    - reset (bool=False) : reset the material if already exists (True) or let it unchanged (False).
    - return_names (bool=True) : return the material names (True) or the materials (False)

    Returns:
    - list of str or list of Materials : the created materials
    """

    materials = []
    for name, spec in mat_specs.items():
        materials.append(create_material(name, reset=reset, **spec))

    if return_names:
        return [mat.name for mat in materials]
    else:
        return materials

# ----------------------------------------------------------------------------------------------------
# Utility to get the materials prefixed
# ----------------------------------------------------------------------------------------------------

def choose_material_type(mat_type, rng, materials=None):
    """ Return the materials the name of which is prefixed by a given string.

    Args:
    - mat_type (str) : the prefix to use to filter the materials
    - rng (random generator) : use to shuffle the list
    - materials (list of Materials=None) : list to work. Use the whole Materials list if None.

    Returns:
    - list of Materials: the filtered materials randomly ordered
    """

    mats = []
    if materials is None:
        mats = [mat.name for mat in bpy.data.materials if mat.name[:len(mat_type)] == mat_type]
    else:
        mats = [name for name in materials if name[:len(mat_type)] == mat_type]

    if len(mats):
        return mats[rng.choice(np.arange(len(mats)))]
    else:
        return None
    
# ----------------------------------------------------------------------------------------------------
# Set material
# ----------------------------------------------------------------------------------------------------

def set_material(spec, material, faces=None):

    if material is None:
        return

    obj = get_object(spec)
    mat = create_material(material, reset=False)
    mat_index = obj.data.materials.get(mat.name)
    if mat_index is None:
        mat_index = len(obj.data.materials)
        obj.data.materials.append(mat)

    if isinstance(obj.data, bpy.types.Mesh):
        if faces is None:
            a = [mat_index]*len(obj.data.polygons)
            obj.data.polygons.foreach_set('material_index', a.astype(bint))
        else:
            a = np.zeros(len(obj.data.polygons))
            obj.data.polygons.foreach_get('material_index', a.astype(bint))
            a[faces] = mat_index
            obj.data.polygons.foreach_set('material_index', a.astype(bint))

# ----------------------------------------------------------------------------------------------------
# Change material image
# ----------------------------------------------------------------------------------------------------

def change_material_image(model, new_name, image, image_nodes=None):

    # ----- Source material

    mat0 = bpy.data.materials[model] if isinstance(model, str) else model

    # ----- Target material

    mat1 = bpy.data.materials.get(new_name)
    if mat1 is not None:
        return mat1

    mat1 = mat0.copy()
    mat1.name = new_name

    assert(mat1.use_nodes)

    # ----- Change the image

    nodes = mat1.node_tree.nodes
    images = []
    for node in nodes:
        if isinstance(node, bpy.types.ShaderNodeTexImage):

            if image_nodes is None:
                images.append(node)

            elif isinstance(image_nodes, str):
                if node.name == image_nodes:
                    images.append(node)

            elif node.name in image_nodes:
                images.append(node)

    if len(images) == 0:
        raise Exception(f"The material '{mat0.name}' doesn't have a image texture node with name matching: {image_nodes}")

    for node in images:
        node.image = image

    return mat1

# ====================================================================================================
# Data Attributes
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Get attributes property of a data block
# ----------------------------------------------------------------------------------------------------

def get_attributes(spec, halt=False):
    data = get_data(spec)
    if hasattr(data, 'attributes'):
        return data.attributes
    elif halt:
        raise Exception(f"Data {spec} has no attributes property")
    else:
        return None

# ----------------------------------------------------------------------------------------------------
# Attribute exists
# ----------------------------------------------------------------------------------------------------

def attribute_exists(spec, name):
    """ Test if an attribute exists.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name

    Returns
    -------
        - bool : True if exists, False otherwise
    """
    attributes = get_attributes(spec)
    if attributes is None:
        return False
    else:
        return name in attributes

# ----------------------------------------------------------------------------------------------------
# Attribute info
# ----------------------------------------------------------------------------------------------------

def get_attribute_info(spec, name):
    """ Get the attribute info.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name

    Returns
    -------
        - dict with attribute informations, None if the attribute doesn't exist
    """
    return get_attributes(spec, halt=True)[name]

# ----------------------------------------------------------------------------------------------------
# List of attributes
# ----------------------------------------------------------------------------------------------------

def get_attribute_names(spec):
    """ Get the name of the attributes of an object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data

    Returns
    -------
        - list of strs : attribute names
    """
    attributes = get_attributes(spec)
    if attributes is None:
        return []
    else:
        return list(attributes.keys())
    
# ----------------------------------------------------------------------------------------------------
# Get the attribute value
# ----------------------------------------------------------------------------------------------------

def get_attribute(attributes, name):

    battr = attributes[name]
    n     = len(battr.data)
    TYPE = TYPES[BL_TYPES[battr.data_type]]

    value_size = TYPE['size']
    value_name = TYPE['name']
    value_type = TYPE['dtype']

    # ----- String

    if battr.data_type == 'STRING':
        return [str(item.value) for item in battr.data]

    # ----- Byte color = read a float and store u8

    elif battr.data_type == 'BYTE_COLOR':
        a = np.empty(n*4, dtype=float)
        battr.data.foreach_get('color_srgb', a)
        return np.reshape(np.clip(a*255, 0, 255).astype(value_type), (n, 4))

    # ------ Other types

    else:
        a = np.empty(n*value_size, dtype=value_type)
        battr.data.foreach_get(value_name, a)

        return np.reshape(a, (n,) + TYPE['shape'])

# ----------------------------------------------------------------------------------------------------
# Set attribute value
# ----------------------------------------------------------------------------------------------------

def set_attribute(attributes, name, value):

    battr = attributes[name]
    n     = len(battr.data)

    TYPE = TYPES[BL_TYPES[battr.data_type]]

    value_size = TYPE['size']
    value_name = TYPE['name']
    value_type = TYPE['dtype']

    # ----- String

    if battr.data_type == 'STRING':
        if isinstance(value, str):
            for i in range(n):
                battr.data[i].value = value

        elif len(value) == 1:
            for i in range(n):
                battr.data[i].value = value[0]

        else:
            for i in range(n):
                battr.data[i].value = value[i]

        return

    # ----- Byte color type : internal is u8 -> need conversion to float

    if battr.data_type == 'BYTE_COLOR':
        if isinstance(value, np.ndarray):
            value = value/255
        else:
            value = np.array(value)/255
        value_type = bfloat

    # ----- Set the array to the layer

    a = np.asarray(value, dtype=value_type).flatten()
    if n*value_size != len(a):
        raise Exception(f"Set Attribute Error: Object attribute '{name}' len is {n}. Impossible to set with value of shape {np.shape(value)}.")

    if not n:
        return

    battr.data.foreach_set(value_name, a)
    return

    if n*value_size == np.size(value):
        battr.data.foreach_set(value_name, np.reshape(value, np.size(value)).astype(value_type))

    else:
        nvalues = np.size(value)//value_size
        if n % nvalues != 0:
            raise Exception(f"Set Attribute Error: Object attribute '{name}' len is {n} (size={n*value_size}). Impossible to set with value of shape {np.shape(value)} (size={np.size(value)}).")

        item_size = n//nvalues
        a = np.empty((nvalues, item_size, value_size), dtype=value_type)
        a[:] = np.reshape(value, (nvalues, 1, value_size))

        battr.data.foreach_set(value_name, np.reshape(a, np.size(a)))

# ----------------------------------------------------------------------------------------------------
# Delete an attribute
# ----------------------------------------------------------------------------------------------------

def delete_attribute(attributes, name):
    """ Delete an attribute.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
    """
    battr = attributes.get(name)
    if battr is not None:
        attributes.remove(battr)

# ----------------------------------------------------------------------------------------------------
# Create an attribute
# ----------------------------------------------------------------------------------------------------

def create_attribute(attributes, name, data_type, domain='POINT', value=None):
    """ Create an attribute into a Blender object.

    Note that if the attribute already exists, it is deleted.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - data_type (str) : attribute data type
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    data_type = TYPES[data_type]['bl_type']
    battr = attributes.get(name)
    if battr is None:
        attributes.new(name, type=data_type, domain=domain)

    if value is not None:
        set_attribute(attributes, name, value)

# ----------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------

def create_float_attribute(spec, name, domain='POINT', value=None):
    """ Create a float attribute into a Blender object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    create_attribute(spec, name, data_type='FLOAT', domain=domain, value=value)

def create_int_attribute(spec, name, domain='POINT', value=None):
    """ Create an int attribute into a Blender object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    create_attribute(spec, name, data_type='INT', domain=domain, value=value)

def create_bool_attribute(spec, name, domain='POINT', value=None):
    """ Create a bool attribute into a Blender object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    create_attribute(spec, name, data_type='BOOLEAN', domain=domain, value=value)

def create_vector_attribute(spec, name, domain='POINT', value=None):
    """ Create a vector attribute into a Blender object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    create_attribute(spec, name, data_type='VECTOR', domain=domain, value=value)

def create_vector2_attribute(spec, name, domain='CORNER', value=None):
    """ Create a vector2 attribute into a Blender object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    create_attribute(spec, name, data_type='FLOAT2', domain=domain, value=value)

def create_color_attribute(spec, name, domain='POINT', value=None):
    """ Create a float color attribute into a Blender object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    create_attribute(spec, name, data_type='COLOR', domain=domain, value=value)

def create_byte_color_attribute(spec, name, domain='POINT', value=None):
    """ Create a byte color attribute into a Blender object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    create_attribute(spec, name, data_type='BYTE_COLOR', domain=domain, value=value)

def create_quaternion_attribute(spec, name, domain='POINT', value=None):
    """ Create a quaternion attribute into a Blender object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    create_attribute(spec, name, data_type='QUATERNION', domain=domain, value=value)

def create_matrix_attribute(spec, name, domain='POINT', value=None):
    """ Create a matrix attribute into a Blender object.

    Arguments
    ---------
        - spec (str or data) : valid spec for object or data
        - name (str) : attribute name
        - domain (str='POINT') : domain of creation
        - value (any=None) : default value
    """
    create_attribute(spec, name, data_type='MATRIX', domain=domain, value=value)


# ====================================================================================================
# Temp folder
# ====================================================================================================

def get_temp_folder():

    try:
        folder_name = bpy.context.scene.npblender_temp_folder
    except:
        folder_name = bpy.context.preferences.filepaths.temporary_directory

    if folder_name == "":
        return Path.cwd()
    else:
        return Path(folder_name)

# ====================================================================================================
# Pillow image
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Pillow image to Blender image :full
# ----------------------------------------------------------------------------------------------------

def pil_to_bl_image(pil_img, name="FromPIL", colorspace='sRGB'):
    if pil_img.mode == "RGBA":
        img_rgba = pil_img
    elif pil_img.mode == "RGB":
        img_rgba = pil_img.convert("RGBA")
    elif pil_img.mode in ("L", "LA"):
        img_rgba = pil_img.convert("RGBA")
    else:
        img_rgba = pil_img.convert("RGBA")

    w, h = img_rgba.size

    arr = np.array(img_rgba, dtype=np.float32) / 255.0   # (h, w, 4)
    arr = np.flipud(arr)
    flat = np.ascontiguousarray(arr).ravel() # size = w*h*4

    bl_img = bpy.data.images.new(name, width=w, height=h, alpha=True, float_buffer=False)
    bl_img.colorspace_settings.name = colorspace
    bl_img.pixels.foreach_set(flat)
    bl_img.update()
    return bl_img


# ----------------------------------------------------------------------------------------------------
# Convert a pillow image into a blender image OLD
# ----------------------------------------------------------------------------------------------------

def pil_to_image(pil_image, name="Pillow"):
    """ Convert a Pillow image into a Blender Image.

    Args:
    - pil_image : image in the pillow format
    - name (str='Pillow') : named to give to the Blender new image.

    Returns:
    - Image : the converted image
    """

    image = bpy.data.images.new(name, pil_image.width, pil_image.height)
    a = np.insert(np.array(pil_image)/255., 3, 1, -1)
    image.pixels[:] = np.reshape(np.flip(a, axis=0), np.size(a))

    return image

# ----------------------------------------------------------------------------------------------------
# Convert a pillow array into a blender image
# ----------------------------------------------------------------------------------------------------

def pil_array_to_image(pil_array, name="Pillow"):
    """ Convert a Pillow arrayinto a Blender Image.

    Args:
    - pil_array : pillow array
    - name (str='Pillow') : named to give to the Blender new image.

    Returns:
    - Image : the converted image
    """

    image = bpy.data.images.new(name, np.shape(pil_array)[1], np.shape(pil_array)[0])
    a = np.insert(pil_array/255, 3, 1, -1)
    image.pixels[:] = np.reshape(np.flip(a, axis=0), np.size(a))

    return image

# ----------------------------------------------------------------------------------------------------
# Get the texture image node of a material
# ----------------------------------------------------------------------------------------------------

def get_image_node(mat, label='Generated'):
    """ Get the node image of a Material shader.

    The name of the node image is used in case several Image Nodes exist.

    Args:
    - mat (Material) : the material to get the node Image from
    - label (str='Generated') : named of the node image

    Returns:
    - None or Image Node
    """

    if isinstance(mat, str):
        mat = bpy.data.materials[mat]

    nodes = [node for node in mat.node_tree.nodes if node.bl_idname == "ShaderNodeTexImage" ]
    if len(nodes) == 1:
        return nodes[0]

    for node in nodes:
        if node.label == label:
            return node

    raise Exception(f"Material '{mat.name}' has several 'Image Texture' nodes. One must be labeled '{label}'")


# ====================================================================================================
# Markers
# ====================================================================================================

def markers(text, clear=True, start=0):
    """Create markers from a text made of lines.

    Each line contains a frame number and a name. These two tokens can be separated
    by a comma, a semi-column or a tab.

    Args:
    - text (string) : lines separated by \n.
    - clear (bool=True) : delete the existing markers if True
    - start (int=0) : frame number to start from
    """

    print('-'*10)
    print("Markers...")
    fails = 0
    total = 0

    scene = bpy.context.scene

    if clear:
        scene.timeline_markers.clear()

    frame_min = 1000000
    frame_max = 0

    # ----- Split the lines
    lines = text.split('\n')

    mks = []

    # ----- First pass to compute min an max
    for line in lines:

        total += 1

        # Split with possible separators
        ko = True
        for sep in [',', '\t', ';']:
            # Empty line !
            if (line == "") or (line.strip() == sep):
                total -= 1
                ko = False
                break

            # Not empty line
            else:
                fr_na = line.split(sep)
                if len(fr_na) == 2:
                    try:
                        frame = int(fr_na[0])
                        mks.append((frame, fr_na[1].strip()))
                        frame_min = min(frame_min, frame)
                        frame_max = max(frame_max, frame)

                        ko = False
                        break
                    except:
                        pass

        if ko:
            fails += 1
            print(f"Markers: unable to handle the line: '{line}'")

    # ----- Loop on the markers to set
    for fr_na in mks:
        scene.timeline_markers.new(fr_na[1], frame=start + fr_na[0] - frame_min)

    # ----- Start and end frames
    scene.frame_start = start
    scene.frame_end   = start + frame_max - frame_min

    # ----- Synthesis
    print(f"Markers from {start} to {start + frame_max - frame_min}: {total} line(s), {fails} fail(s)")

# ====================================================================================================
# Infos
# ====================================================================================================


def fps():
    return bpy.context.scene.render.fps

def marker_frame(name):
    if isinstance(name, str):
        return bpy.context.scene.timeline_markers[name].frame
    else:
        return name

def frtime(frame):
    scene = bpy.context.scene
    return (marker_frame(frame) - scene.frame_start)/scene.render.fps

def frdur(frame0, frame1):
    return (marker_frame(frame1) - marker_frame(frame0))/bpy.context.scene.render.fps

def frame_at(t):
    return bpy.context.scene.frame_start + round(t*fps())

# ====================================================================================================
# FCurves (Not compatible with layerss yet)
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Return the data_path and in the index of a name path
# ----------------------------------------------------------------------------------------------------

def data_path_index(data, name, index=-1):
    """Key frame utility: normalize the syntax name, index of a key frame.

    Used for instance for location. The syntax "location.x" can be used rather
    than the Blender syntax: "location", index=0.

    Transforms: location.y, -1 --> location, 1

    If index is not -1 and the dotted syntaxe is used, an error is raised.

    Args:
    - name (str) : the name of the attribute with possible the syntax attr.x.
    - index (int=-1) : for array attribute, the index of the entry. The default is -1.


    Returns:
    - couple (str, int) : The not dotted attribute name and in the index in the array
    """

    # ----- Custom property

    if '[' in name:
        return data, name, index

    # ----- Items

    attrs = name.split('.')
    if len(attrs) == 1:
        return data, name, index

    idx = index
    if idx == -1 and attrs[-1] in ["x", "y", "z", "w"]:
        idx = ["x", "y", "z", "w"].index(attrs[-1])
        del attrs[-1]

    if len(attrs) == 1:
        return data, attrs[0], idx

    for attr in attrs[:-1]:
        data = getattr(data, attr)

    return data, attrs[-1], idx

# ----------------------------------------------------------------------------------------------------
# Get a blender fcurve
# ----------------------------------------------------------------------------------------------------

def get_fcurve(spec, name, index=-1, create=True):
    """ Get a Blender fcurve.

    Args:
    - spec (str or Object) : the Object
    - name (str) : name of the attribute
    - index (int=-1) : index in the array
    - create (bool=True) : create the fcurve if it doesn't exist

    Returns:
    - fcurve or None
    """

    obj = get_object(spec)

    data, data_path, index = data_path_index(obj, name, index)


    # ----- animation

    animation = data.animation_data
    if animation is None:
        if create:
            animation = data.animation_data_create()
        else:
            return None

    # ----- action

    if animation.action is None:
        if create:
            animation.action = bpy.data.actions.new(name=f"{obj.name} action")
        else:
            return None

    # ----- Look in fcurves

    fcurves = animation.action.fcurves
    for fc in fcurves:
        if fc.data_path == data_path and (fc.array_index == index or index==-1):
            return fc

    if not create:
        return None

    # --- Create

    return fcurves.new(data_path=data_path, index=index)

# ----------------------------------------------------------------------------------------------------
# reset keyframes
# ----------------------------------------------------------------------------------------------------

def fc_clear(fc, frame0=None, frame1=None):
    """ Clear the points of a fcurve.

    Args:
    - fc (fcurve) : the fcurve to clear
    - frame0 (int=None) : clear from that frame
    - frame1 (int=None) : clear to that frame

    Returns:
    - None
    """

    kfs = []
    for i_kf, kf in enumerate(fc.keyframe_points):

        if frame0 is None:
            suppr = True
        else:
            suppr = kf.co[0] >= frame0

        if suppr and frame1 is not None:
            suppr = kf.co[0] <= frame1

        if suppr:
            kfs.append(i_kf)

    for i_kf in reversed(kfs):
        fc.keyframe_points.remove(fc.keyframe_points[i_kf])

# ----------------------------------------------------------------------------------------------------
# Set a list of keyframes
# ----------------------------------------------------------------------------------------------------

def fc_set_kfs(fc, kfs):

    from mathutils import Vector

    fc.keyframe_points.clear()
    fc.keyframe_points.add(len(kfs))

    for kf, in_kf in zip(fc.keyframe_points, kfs):
        kf.amplitude         = in_kf.amplitude
        kf.back              = in_kf.back
        kf.easing            = in_kf.easing
        kf.co                = Vector(in_kf.co)
        kf.handle_left       = Vector(in_kf.handle_left)
        kf.handle_right      = Vector(in_kf.handle_right)
        #kf.handle_left_type  = in_kf.handle_left_type
        #kf.handle_right_type = in_kf.handle_right_type
        kf.interpolation     = in_kf.interpolation
        kf.period            = in_kf.period
        #kf.type              = in_kf.type

    return fc

# ----------------------------------------------------------------------------------------------------
# Set a new keyframe in a fcurve
# ----------------------------------------------------------------------------------------------------

def fc_set_keyframe(fc, frame, value, interpolation=None):
    """ Set a new keyframe.

    Args:
    - fc (fcurve) : the fcurve to create a keyframe into
    - frame(int or float) : the frame
    - value (float) : the value at the frame
    - interpolation (str=None) : a valid interpolation mode

    Returns:
    - KeyFrame : the created keyframe
    """

    for kf in fc.keyframe_points:
        if abs(kf.co[0] - frame) < 0.1:
            kf.co[1] = value
            if interpolation is not None:
                kf.interpolation = interpolation
                return kf

    kf = fc.keyframe_points.insert(frame, value)
    if interpolation is None:
        kf.interpolation = 'BEZIER'
    else:
        kf.interpolation = interpolation

    fc.update()

    return kf

# ----------------------------------------------------------------------------------------------------
# Keyframe by path
# ----------------------------------------------------------------------------------------------------

def kf_clear(spec, name, frame0=None, frame1=None):
    """ Clear a keyframe.

    Args:
    - spec (str or Object) : the Object
    - name (str) : name of the fcurve
    - frame0 (int=None) : clear from that frame
    - frame1 (int=None) : clear to that frame

    Returns:
    - None
    """

    fc = get_fcurve(spec, name, create=None)
    if fc is not None:
        fc_clear(fc, frame0=frame0, frame1=frame1)

def set_key_frame(spec, name, frame, value, interpolation=None):
    """ Set a keyframe.

    Args:
    - spec (str or Object) : the Object
    - name (str) : name of the fcurve
    - frame(int or float) : the frame
    - value (float) : the value at the frame
    - interpolation (str=None) : a valid interpolation mode

    Returns:
    - KeyFrame : the created keyframe
    """

    return fc_set_keyframe(get_fcurve(spec, name), frame, value, interpolation=interpolation)


def get_value_at_frame(frame, spec, name, index=-1):
    """ Get the value of an attribute possibly keyed.

    Arguments
    ---------
        - frame (int) : frame at which the value is needed
        - spec (data spec) : data owning the property
        - name (str) : attribute name
        - index (int = -1) : index for vector attributes

    Returns
    -------
        - value
    """

    obj = get_object(spec)
    data, data_path, index = data_path_index(obj, name, index)

    # ----- Ref value

    if '[' in data_path:
        ref_value = eval(f"data{data_path}")
    else:
        ref_value = getattr(data, data_path)

    # ----- Target is a Vector or a list and we need all components

    if hasattr(ref_value, '__len__') and index == -1:
        v = []
        for i in range(len(ref_value)):
            v.append(get_value_at_frame(frame, data, data_path, index=i))

        if isinstance(ref_value, idprop.types.IDPropertyArray):
            return list(v)
        elif isinstance(ref_value, Vector):
            return Vector(v)
        else:
            return v

    # ----- Only a single value is required

    fcurve = get_fcurve(data, data_path, index, create=False)

    if fcurve is None:
        if index >= 0:
            return ref_value[index]
        else:
            return ref_value
    else:
        return fcurve.evaluate(frame)

# ====================================================================================================
# A Key frame
# ====================================================================================================

class KeyFrame:

    def __init__(self, frame, value):
        """ KeyFrame class takes same arguments as Blender KeyFrame.

        This class is used for operations on fcurves.

        Args:
        - frame (int or float) : the frame
        - value (float) : the keyframe value
        """

        self.amplitude         = 0.
        self.back              = 0.
        self.co                = Vector((frame, value))
        self.easing            = 'AUTO'
        self.handle_left       = Vector((0., 0.))
        self.handle_right      = Vector((0., 0.))
        self.handle_left_type  = 'FREE'
        self.handle_right_type = 'FREE'
        self.interpolation     = 'CONSTANT'
        self.period            = 0.
        self.type              = 'KEYFRAME'

    @classmethod
    def FromKeyFrame(cls, kf):
        """ Initialize from a Blender keyframe.

        Args:
        - kf (Blender KeyFrame) : the keyframe to copy
        """

        new_kf = cls(kf.co[0], kf.co[1])

        new_kf.amplitude         = kf.amplitude
        new_kf.back              = kf.back
        new_kf.easing            = kf.easing
        new_kf.handle_left       = Vector(kf.handle_left)
        new_kf.handle_right      = Vector(kf.handle_right)
        new_kf.handle_left_type  = kf.handle_left_type
        new_kf.handle_right_type = kf.handle_right_type
        new_kf.interpolation     = kf.interpolation
        new_kf.period            = kf.perdio
        new_kf.type              = kf.type

    def to_keyframe(self, kf):
        """ Copy attributes to a Blender KeyFrame

        Args:
        - kf (Blender KeyFrame) : the keyframe to setup
        """

        kf.amplitude         = self.amplitude
        kf.back              = self.back
        kf.easing            = self.easing
        kf.handle_left       = Vector(self.handle_left)
        kf.handle_right      = Vector(self.handle_right)
        kf.handle_left_type  = self.handle_left_type
        kf.handle_right_type = self.handle_right_type
        kf.interpolation     = self.interpolation
        kf.period            = self.perdio
        kf.type              = self.type

# ====================================================================================================
# A Function curve
# ====================================================================================================

class FCurve(list):

    def __init__(self, spec, name):

        self.fc = get_fcurve(spec, name)
        for kf in fc.keyframe_points:
            self.append(KeyFrame.FromKeyFrame(kf))

    def to_fcurve(self, fc):

        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(self))

        for kf0, kf1 in zip(self, fc.keyframe_points):
            kf0.to_keyframe(kf1)

# ====================================================================================================
# Shape Keys
# ====================================================================================================

def shape_key_name(name="Key", index=0):
    if name == 'Key' and index == 0:
        return 'Basis'
    else:
        return f"{name} {index}"

def has_shape_keys(spec):

    obj = get_object(spec, halt=False)

    if obj is None:
        return False

    return obj.data.shape_keys is not None

def shape_keys_count(spec):

    obj = get_object(spec, halt=False)

    if obj is None or obj.data.shape_keys is None:
        return 0

    return len(obj.data.shape_keys.key_blocks)

def shape_keys_clear(spec):

    obj = get_object(spec, halt=False)

    if obj is None or obj.data.shape_keys is None:
        return

    for shapekey in obj.data.shape_keys.key_blocks:
        obj.shape_key_remove(shapekey)

def get_key_block(spec, index, create=False, name=None):

    obj = get_object(spec)

    # ----- No block at all, let's create the first one if requested

    if obj.data.shape_keys is None:
        if not create:
            return None
        kb = obj.shape_key_add()
        kb.name = 'Basis'

    kbs = obj.data.shape_keys.key_blocks

    # ----- Only integer index, no key name

    if name is None:
        if create:
            for _ in range(len(kbs)-1, index):
                obj.shape_key_add()
        if index >= len(kbs):
            return None
        else:
            return kbs[index]

    # ----- Key block by name

    key_name = shape_key_name(name, index)
    kb = kbs.get(key_name)
    if kb is None:
        if create:
            kb = obj.shape_key_add()
            kb.name = key_name
    return kb

# ====================================================================================================
# Rendering
# ====================================================================================================

def is_viewport():
    return bpy.context.workspace is not None

def lock_interface(value):
    bpy.context.scene.render.use_lock_interface = value

def render_engine():
    return bpy.context.scene.render.engine
