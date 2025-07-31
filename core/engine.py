#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:33:23 2022

@author: alain.bernard
"""

import logging
import numpy as np

import bpy
from bpy.types import Depsgraph

from npblender.core.bakefile import BakeFile

# ====================================================================================================
# Animation Engine
#
# Manages
# - list of animations
# - frame and time
# - render / viewport
# - baking

# ====================================================================================================
# Animation parameters

verbose = True

class Parameters:
    rendering   = False
    depsgraph   = None

    scene       = None
    frame       = None
    time_offset = 0.
    time_scale  = 1.

    subframes   = 0
    subframe    = 0

    rng_seed    = 0
    rng         = np.random.default_rng(0)

# List of Animations
animations = []

# Baking
bake_file = None

# ====================================================================================================
# Property functions

# ----------------------------------------------------------------------------------------------------
# Scene

def get_scene():
    if Parameters.scene is None:
        return bpy.context.scene
    else:
        return Parameters.scene

# ----------------------------------------------------------------------------------------------------
# Frame and Time

def get_fps():
    return get_scene().render.fps

def get_frame():
    if Parameters.frame is None:
        return get_scene().frame_current
    else:
        return Parameters.frame

def is_first_frame():
    return get_frame() == get_scene().frame_start

def time():
    # subframes are enumerated in the order : 1, 2, ..., n-1, 0
    if Parameters.subframe == 0:
        frame = get_frame()
    else:
        frame = get_frame() - 1 + Parameters.subframe/(Parameters.subframes + 1)

    return Parameters.time_offset + Parameters.time_scale*frame/get_fps()

def get_duration():
    return Parameters.time_scale*(1 + get_scene().frame_end - get_scene().frame_start)/get_fps()

def delta_time():
    return Parameters.time_scale/get_fps()/(Parameters.subframes + 1)

# ----------------------------------------------------------------------------------------------------
# Context

def is_viewport():
    return not Parameters.rendering

# ----------------------------------------------------------------------------------------------------
# Motion blur class update several times

def use_motion_blur():
    return get_scene().render.use_motion_blur

# ----------------------------------------------------------------------------------------------------
# Random seed

def get_rng():
    return Parameters.rng

def seed():
    return get_rng().integers(1<<63)

# ====================================================================================================
# Utilities

def get_evaluated(spec):
    obj = bpy.data.objects[spec] if isinstance(spec, str) else spec
    if Parameters.depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        return obj.evaluated_get(depsgraph)
    else:
        return obj.evaluated_get(Parameters.depsgraph)

def lock_interface(value):
    bpy.context.scene.render.use_lock_interface = value

# ====================================================================================================
# Add an Animation

def add(simul):
    animations.append(simul)

# ====================================================================================================
# Baking

def init_bake():

    global bake_file

    name = get_scene().npblender_bake_name
    if get_scene().npblender_use_bake and name != "":
        print(f"Engine> init bake '{name}'")
        bake_file = BakeFile(name)
    else:
        bake_file = None

def load_frame():

    # ----- No baking

    if bake_file is None:
        return False

    # ----- Animation data : a list of animation data, one per Animation

    data = bake_file.read(get_frame())
    if data is None:
        logging.debug(f"Engine> {get_frame()} not baked, return None")
        return False

    logging.debug(f"Engine> frame {get_frame():3d} loaded, setting animation to frames.")

    for i, simul in enumerate(animations):
        simul.set_animation(data[i])

    # ----- Frame state

    if bake_file.state_frame == get_frame():
        logging.debug(f"Engine> loading state at frame {get_frame()}.")
        state_data = bake_file.read_state()
        for i, simul in enumerate(animations):
            simul.set_state(state_data[i])

    return True

def save_frame():

    # ----- No baking

    if bake_file is None:
        return

    # ----- Don't save if not following the last saved frame

    skip = None
    if bake_file.state_frame is None:
        if get_frame() != get_scene().frame_start:
            skip = "Not the first frame"
    elif get_frame() == bake_file.state_frame:
        # Already saved : returns silently
        return
    else:
        if get_frame() != bake_file.state_frame + 1:
            skip = f"Not the frame following last saved state {bake_file.state_frame}"

    if skip is not None:
        print(f"Engine> Skip saving frame {get_frame()}: {skip}")
        return

    # ----- Collect data and state

    anim  = []
    state = []
    for anm in animations:
        anim.append(anm.get_animation())
        state.append(anm.get_state())

    # ----- Write in bake file

    bake_file.write(get_frame(), anim, state)

# ====================================================================================================
# Frames computation

# ----------------------------------------------------------------------------------------------------
# Reset
# Called when current frame is the first one

def reset():
    print("Engine> Reset")

    for anm in animations:
        anm.reset()

# ----------------------------------------------------------------------------------------------------
# One step

def step():

    from time import time

    # ----------------------------------------------------------------------------------------------------
    # Preparation

    # FIRST FRAME : initialize baking

    if is_first_frame():
        init_bake()

    # New random generator for this frame

    Parameters.rng = np.random.default_rng(Parameters.rng_seed + get_frame())

    # ----------------------------------------------------------------------------------------------------
    # Compute / load frame

    # ----- Frame is baked

    loaded = False

    load_dur = 0.
    comp_dur = 0.
    save_dur = 0.
    view_dur = 0.

    t0 = time()
    if load_frame():

        load_dur = time() - t0
        if False and verbose:
            print(f"Engine> Frame {get_frame()} loaded in {load_dur:.2f} s")

    # ----- Frame is not baked

    else:
        # Reset for first one

        if is_first_frame():
            reset()
            subframes = 0
        else:
            subframes = Parameters.subframes

        # ----------------------------------------------------------------------------------------------------
        # Sub frames
        # subframes can be 0. The number of computations is subframes + 1
        #
        # order: 1, 2, 3, ..., n-1, 0
        # subframe > 1: computing a subframe
        # subframe == 0 means : we compute current frame after all subframes

        for i in range(1, subframes + 2):
            subframe = 0 if i == subframes + 1 else i
            for anm in animations:
                anm.before_compute()
                anm.compute()
                anm.after_compute()
        comp_dur = time() - t0

        # Bake
        t1 = time()
        save_frame()
        save_dur = time() - t1
        if False and verbose:
            print(f"Engine> Frame {get_frame()} saved in {save_dur:6.2f} s")

    # ---------------------------------------------------------------------------
    # View the result

    t1 = time()
    for anm in animations:
        anm.view()

    view_dur = time() - t1
    duration = time() - t0

    if verbose:
        print(f"Engine> Frame {get_frame()} in {duration:6.2f} s (load: {load_dur:.2f}, comp: {comp_dur:.2f}, save: {save_dur:.2f}, view: {view_dur:.2f})")



# ====================================================================================================
# Viewport animation
# Frame change - handler: frame_change_pre

def update(scene, depsgraph):

    # ----------------------------------------------------------------------------------------------------
    # Rendering : update is done by before_render_image

    if Parameters.rendering:
        Parameters.depsgraph = depsgraph
        before_render_image(scene, Parameters.depsgraph)
        Parameters.depsgraph = None
        return

    # ----------------------------------------------------------------------------------------------------
    # Viewport update

    Parameters.scene = scene
    step()

# ====================================================================================================
# Render animation
# - Capture start and end rendering
# - Call step at each new frame

# ----------------------------------------------------------------------------------------------------
# Render init - handler: render_init

def before_render(scene):

    print("Engine> Start rendering")

    # To be sure
    lock_interface(True)

    Parameters.rendering = True

# ----------------------------------------------------------------------------------------------------
# Render done - handler: render_complete and render_cancel

def after_render(scene):
    print("Engine> Render completed")
    Parameters.rendering = False
    Parameters.depsgraph = None

# ----------------------------------------------------------------------------------------------------
# Render image - handler: render_pre

def before_render_image(scene, depsgraph):

    # ----- Points to the right scene
    # depsgraph has been set by update

    Parameters.scene = scene
    print(f"Engine> Frame {Parameters.frame}")

    step()

# ====================================================================================================
# Engine initialization

def init(subframes=0):

    Parameters.subframes = subframes
    Parameters.subframe  = 0

    animations.clear()

# ====================================================================================================
# Quick launch

def animation(animation, subframes=0):
    """ > Single animation

    Arguments
    ---------
    - animation (Animation)
    - subframes (int = 0) : number of subframes
    """

    # Init the engine
    init(subframes=subframes)

    # Add the animation
    add(animation)

    # Current frames
    step()

# ====================================================================================================
# Go

def go(compute, reset=None, view=None, subframes=0):
    """ > Legacy behavior : animates with functions

    Arguments
    ---------
    - compute (function) : computation function
    - reset (function = None) : reset function
    - view (function = None) : view function
    - subframes (int = 0) : subframes
    """
    animation(SimpleAnimation(compute, reset=reset, view=view), subframes=subframes)

    #Engine.go(compute, reset=reset, view=view, subframes=subframes)

# ====================================================================================================
# Animation

# ----------------------------------------------------------------------------------------------------
# Base Animation

class Animation:

    # ---------------------------------------------------------------------------
    # Exposes animation parameters

    @property
    def scene(self):
        return get_scene()

    @property
    def fps(self):
        return get_fps()

    @property
    def frame(self):
        return get_frame()

    @property
    def subframes(self):
        return Parameters.subframes

    @property
    def subframe(self):
        return Parameters.subframe

    @property
    def time(self):
        return time()

    @property
    def delta_time(self):
        return delta_time()

    @property
    def is_first_frame(self):
        return is_first_frame()

    @property
    def rendering(self):
        return Parameters.rendering

    @property
    def is_viewport(self):
        return Parameters.is_viewport

    @property
    def is_baked(self):
        return bake_file is not None

    @property
    def rng(self):
        return Parameters.rng

    @property
    def seed(self):
        return seed()

    # ---------------------------------------------------------------------------
    # 3 base animation methods

    def reset(self):
        pass

    def compute(self):
        pass

    def view(self):
        pass

    # ---------------------------------------------------------------------------
    # Complementary

    def before_compute(self):
        pass

    def after_compute(self):
        pass

    # ---------------------------------------------------------------------------
    # Baking

    # ----- Animation data

    def get_animation(self):
        return None

    def set_animation(self, data):
        pass

    # ----- State data

    def get_state(self):
        return None

    def set_state(self, data):
        pass

    # ---------------------------------------------------------------------------
    # Run

    def go(self, subframes=0):
        animation(self, subframes=subframes)

# ----------------------------------------------------------------------------------------------------
# Simple Animation

class SimpleAnimation(Animation):
    def __init__(self, compute, reset=None, view=None):
        self._compute = compute
        self._reset   = reset
        self._view    = view

    def reset(self):
        if self._reset is not None:
            self._reset()

    def compute(self):
        self._compute()

    def view(self):
        if self._view is not None:
            self._view()

# ----------------------------------------------------------------------------------------------------
# Demo Animation

class DemoAnimation(Animation):
    def __init__(self):
        self.reset()

    def reset(self):
        from . import blender, mesh
        self.cube = mesh.Mesh.Cube()
    
    def compute(self):
        self.cube.points.position += self.rng.uniform(-.1, .1, (len(self.cube.points), 3))

    def view(self):
        self.cube.to_object("Cube")

    def get_animation(self):
        return self.cube.points.position
    
    def set_animation(self, data):
        self.cube.points.position = data

# ====================================================================================================
# Automation in scene

# ----------------------------------------------------------------------------------------------------
# Create properties in scene

def create_scene_properties():

    Scene = bpy.types.Scene

    Scene.npblender_use_bake            = bpy.props.BoolProperty(   name="Bake animation")
    Scene.npblender_temp_folder         = bpy.props.StringProperty( name="Temp folder")
    Scene.npblender_bake_name           = bpy.props.StringProperty( name="Name")

    scene = bpy.context.scene
    scene.npblender_use_bake     = False
    scene.npblender_temp_folder  = bpy.context.preferences.filepaths.temporary_directory
    scene.npblender_bake_name    = "bake"

# ----------------------------------------------------------------------------------------------------
# Bake file operator

class npblenderBake(bpy.types.Operator):
    """Bake npblender animation"""
    bl_idname = "scene.npblender_bake"
    bl_label = "Bake npblender Animation"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        scene = context.scene
        if scene.npblender_use_bake:
            scene.npblender_use_bake = False
        else:
            scene.npblender_use_bake = True
            if scene.npblender_bake_name == "":
                scene.npblender_bake_name = "bake"

        init_bake()
        if is_first_frame():
            step()

        return {'FINISHED'}



# ----------------------------------------------------------------------------------------------------
# Bake files deletion operator

class npblenderDelBakeFiles(bpy.types.Operator):
    """Delete npblender bake files"""
    bl_idname = "scene.npblender_del_bake_files"
    bl_label = "Delete npblender bake files"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        context.scene.npblender_use_bake = False
        init_bake()
        BakeFile.delete_bake_files(context.scene.npblender_bake_name)
        return {'FINISHED'}

# ----------------------------------------------------------------------------------------------------
# UI Panel

class npblenderBakePanel(bpy.types.Panel):
    """npblender animation bake control"""
    bl_label        = "npblender animation bake"
    bl_idname       = "SCENE_PT_layout"
    bl_space_type   = 'PROPERTIES'
    bl_region_type  = 'WINDOW'
    bl_context      = "scene"

    def draw(self, context):

        layout = self.layout
        scene = context.scene

        layout.prop(scene, "npblender_temp_folder")
        layout.prop(scene, "npblender_bake_name")

        row = layout.row()
        row.scale_x = 1.5
        row.operator("scene.npblender_bake", depress=scene.npblender_use_bake)

        row = layout.row()
        row.scale_x = 1.0
        row.operator("scene.npblender_del_bake_files")


# ====================================================================================================
# Register / unregister

def register():

    create_scene_properties()

    # Handle on change post to get the depsgraph

    # ----- Frame change
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(update)

    # ----- Render init
    bpy.app.handlers.render_init.clear()
    bpy.app.handlers.render_init.append(before_render)

    # ----- Render done
    bpy.app.handlers.render_complete.clear()
    bpy.app.handlers.render_complete.append(after_render)

    bpy.app.handlers.render_cancel.clear()
    bpy.app.handlers.render_cancel.append(after_render)

    # ----- UI

    bpy.utils.register_class(npblenderBake)
    bpy.utils.register_class(npblenderDelBakeFiles)
    bpy.utils.register_class(npblenderBakePanel)


def unregister():
    bpy.app.handlers.frame_change_post.remove(update)
    bpy.app.handlers.render_init.remove(before_render)
    bpy.app.handlers.render_complete.remove(after_render)
    bpy.app.handlers.render_cancel.remove(after_render)

    bpy.utils.unregister_class(npblenderBakePanel)
    bpy.utils.unregister_class(npblenderDelBakeFiles)
    bpy.utils.unregister_class(npblenderBake)

if __name__ == "__main__":
    register()
