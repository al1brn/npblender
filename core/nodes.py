#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blender Python Geometry module


@author: alain.bernard
@email: alain@ligloo.net

-----

Call Geometry Nodes modifier

"""

import bpy
from npblender.core import blender

# ====================================================================================================
# Apply a Geometry Nodes modifier

def add_gn_modifier(obj, tree_name):
    """ > Apply a Geometry Nodes modifier

    Arguments
    ---------
    - obj (Blender Object) : the object to apply a modifier on
    - tree_name (str or GeometryNodeTree or tuple of) : the Geometry Nodes to Apply

    Returns
    -------
    - Blender Object : the modified object
    """

    if isinstance(tree_name, tuple):
        for name in tree_name:
            add_gn_modifier(obj, name)

    elif isinstance(tree_name, str):
        mod = obj.modifiers.new(name=tree_name, type='NODES')
        mod.node_group = bpy.data.node_groups[tree_name]

    else:
        mod = obj.modifiers.new(name='Geometry Nodes', type='NODES')
        mod.node_group = tree_name

    return obj

# ====================================================================================================
# Get modified object

def get_evaluated(obj):

    depsgraph = bpy.context.evaluated_depsgraph_get()
    object_eval = obj.evaluated_get(depsgraph)

    return object_eval

# ====================================================================================================
# Useful modifiers

# ----------------------------------------------------------------------------------------------------
# Boolean mesh

def mesh_boolean(*meshes, exact=False, operation='DIFFERENCE'):

    import geonodes as gn
    from npblender import Mesh

    object = None
    with gn.GeoNodes("npblender Mesh Boolean"):
        sockets = [gn.Geometry()]
        for i, mesh in enumerate(meshes):
            name = f"npblender temp {i}"
            obj = mesh.to_object(name, collection=blender.get_temp_collection())
            if object is None:
                object = obj
            else:
                sockets.append(gn.nd.object_info(name).geometry)

        node = gn.Mesh.Boolean(*sockets, solver='EXACT' if exact else 'FLOAT', operation=operation)

        node.mesh.out()

    add_gn_modifier(object, "npblender Mesh Boolean")
    evaluated = get_evaluated(object)

    mesh = Mesh.FromObject(evaluated)

    if True:
        for i in range(len(meshes)):
            blender.delete_object(f"npblender temp {i}")

    return mesh


def curve_to_mesh_modifier():

    import geonodes as gn

    tree_name = "npblender Curve to Mesh"
    ctm = bpy.data.node_groups.get(tree_name)
    if True or ctm is None:
        with gn.GeoNodes(tree_name):
            gn.Curve().to_mesh().out()

        ctm = bpy.data.node_groups.get(tree_name)
    return ctm
