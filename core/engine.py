# =============================================================================
#  npblender.engine
# -----------------------------------------------------------------------------
#  Part of the npblender package
#
#  License: MIT
#  Created: 11/11/2022
#  Last updated: 12/08/2025
#  Author: Alain Bernard
# =============================================================================

"""
Engine
======

Main animation engine for npblender, managing animations, frame stepping,
baking, and interaction with Blender's scene and rendering pipeline.

Features:
- Global animation management (add, run, reset)
- Frame stepping with optional subframes
- Baking system for saving and loading animation states
- Integration with Blender's depsgraph and render handlers
- Random number generation per frame for procedural animation
- Support for viewport and render-time updates

This module also provides a base `Animation` class, a `SimpleAnimation`
wrapper for function-based animations, and Blender operators/panels for
managing baked animations.
"""

import logging
import numpy as np

import bpy

from . bakefile import BakeFile

# ====================================================================================================
# Animation Engine
#
# Manages
# - list of animations
# - frame and time
# - render / viewport
# - baking


# ====================================================================================================
# Engine class
# ====================================================================================================

class Engine:

    VERBOSE = True
    STEP_TIMES = False

    # Global list of animations
    animations = [] 

    # Baking
    bake_file = None

    def __init__(self):
        self._scene = None
        self._frame = None

        self.time_offset = 0.
        self.time_scale  = 1.

        self.subframes   = 0
        self.subframe    = 0

        self._rendering  = False
        self._depsgraph  = None

        self._seed       = 0
        self._rng        = np.random.default_rng(self._seed)

    # ====================================================================================================
    # Engine reset
    # ====================================================================================================

    def reset(self, subframes=0):

        self.subframes = subframes
        self.subframe  = 0

        self.animations.clear()

    # ====================================================================================================
    # Properties
    # ====================================================================================================

    @property
    def scene(self):
        if self._scene is None:
            return bpy.context.scene
        else:
            return self._scene
        
    @property
    def fps(self):
        return self.scene.render.fps
    
    @property
    def frame(self):
        if self._frame is None:
            return self.scene.frame_current
        else:
            return self._frame
        
    @property
    def is_first_frame(self):
        return self.frame == self.scene.frame_start
    
    @property
    def time(self):
        # subframes are enumerated in the order : 1, 2, ..., n-1, 0
        if self.subframe == 0:
            frame = self.frame
        else:
            frame = self.frame - 1 + self.subframe/(self.subframes + 1)

        return self.time_offset + self.time_scale*frame/self.fps
    
    @property
    def duration(self):
        return self.time_scale*(1 + self.scene.frame_end - self.scene.frame_start)/self.fps

    @property
    def delta_time(self):
        return self.time_scale/self.fps/(self.subframes + 1)
    
    @property
    def rendering(self):
        return self._rendering

    @property
    def is_viewport(self):
        return not self._rendering

    @property
    def use_motion_blur(self):
        return self.scene.render.use_motion_blur
    
    @property
    def is_baked(self):
        return self.bake_file is not None

    @property
    def rng(self):
        return self._rng

    @property
    def seed(self):
        return self.rng.integers(1<<63)
    
    # ====================================================================================================
    # Utilities
    # ====================================================================================================

    @property
    def depsgraph(self):
        if self._depsgraph is None:
            return bpy.context.evaluated_depsgraph_get()
        else:
            return self._depsgraph

    def get_evaluated(self, spec):
        obj = bpy.data.objects[spec] if isinstance(spec, str) else spec
        return obj.evaluated_get(self.depsgraph)

    # ====================================================================================================
    # Baking
    # ====================================================================================================

    def init_bake(self):

        name = self.scene.npblender_bake_name
        if self.scene.npblender_use_bake and name != "":
            print(f"Engine> init bake '{name}'")
            self.bake_file = BakeFile(name)
        else:
            self.bake_file = None

    def load_frame(self):

        # ----- No baking

        if self.bake_file is None:
            return False

        # ----- Animation data : a list of animation data, one per Animation

        data = self.bake_file.read(self.frame)
        if data is None:
            logging.debug(f"Engine> {self.frame} not baked, return None")
            return False

        logging.debug(f"Engine> frame {self.frame:3d} loaded, setting animation to frames.")

        for i, anim in enumerate(self.animations):
            anim.set_frame_data(data[i])

        # ----- Frame state

        if self.bake_file.state_frame == self.frame:
            logging.debug(f"Engine> loading state at frame {self.frame}.")
            state_data = self.bake_file.read_state()
            for i, anim in enumerate(self.animations):
                anim.set_frame_state(state_data[i])

        return True

    def save_frame(self):

        # ----- No baking

        if self.bake_file is None:
            return

        # ----- Don't save if not following the last saved frame

        skip = None
        if self.bake_file.state_frame is None:
            if self.frame != self.scene.frame_start:
                skip = "Not the first frame"

        elif self.frame ==self.bake_file.state_frame:
            # Already saved : returns silently
            return
        
        else:
            if self.frame != self.bake_file.state_frame + 1:
                skip = f"Not the frame following last saved state {self.bake_file.state_frame}"

        if skip is not None:
            print(f"Engine> Skip saving frame {self.frame}: {skip}")
            return

        # ----- Collect data and state

        animations = []
        states     = []
        for anim in self.animations:
            animations.append(anim.get_frame_data())
            states.append(anim.get_frame_state())

        # ----- Write in bake file

        self.bake_file.write(self.frame, animations, states)

    # ====================================================================================================
    # Computation
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Reset at first frame
    # ----------------------------------------------------------------------------------------------------

    def first_frame_reset(self):
        """ Called when current frame is the first one
        """
        print("Engine> Reset")

        for anim in self.animations:
            anim.reset()

    # ----------------------------------------------------------------------------------------------------
    # Compute one step
    # ----------------------------------------------------------------------------------------------------

    def step(self):
        """ One step
        """

        from time import time

        # ===== Preparation

        # FIRST FRAME : initialize baking
        if self.is_first_frame:
            self.init_bake()

        # New random generator for this frame
        self._rng = np.random.default_rng(self._seed + self.frame)

        # ===== Compute / load frame

        # ----- Frame is baked

        loaded = False

        load_dur = 0.
        comp_dur = 0.
        save_dur = 0.
        view_dur = 0.

        t0 = time()
        if self.load_frame():

            load_dur = time() - t0
            if self.STEP_TIMES:
                print(f"Engine> Frame {self.frame} loaded in {load_dur:.2f} s")

        # ----- Frame is not baked

        else:
            # Reset for first one

            if self.is_first_frame:
                self.first_frame_reset()
                subframes = 0
            else:
                subframes = self.subframes

            # ----------------------------------------------------------------------------------------------------
            # Sub frames
            # subframes can be 0. The number of computations is subframes + 1
            #
            # order: 1, 2, 3, ..., n-1, 0
            # subframe > 1: computing a subframe
            # subframe == 0 means : we compute current frame after all subframes

            for i in range(1, subframes + 2):
                subframe = 0 if i == subframes + 1 else i
                for anim in self.animations:
                    anim.before_compute()
                    anim.compute()
                    anim.after_compute()
            comp_dur = time() - t0

            # Bake
            t1 = time()
            self.save_frame()
            save_dur = time() - t1
            if self.STEP_TIMES:
                print(f"Engine> Frame {self.frame} saved in {save_dur:6.2f} s")

        # ===== View the result

        t1 = time()
        for anim in self.animations:
            anim.view()

        view_dur = time() - t1
        duration = time() - t0

        if self.VERBOSE:
            print(f"Engine> Frame {self.frame} in {duration:6.2f} s (load: {load_dur:.2f}, comp: {comp_dur:.2f}, save: {save_dur:.2f}, view: {view_dur:.2f})")

    # ====================================================================================================
    # Animation API
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Add an animation
    # ----------------------------------------------------------------------------------------------------

    def add(self, animation):
        self.animations.append(animation)

    # ----------------------------------------------------------------------------------------------------
    # Single animation quick launch
    # ----------------------------------------------------------------------------------------------------

    def animation(self, animation, subframes=0):
        """ > Single animation

        Arguments
        ---------
        - animation (Animation)
        - subframes (int = 0) : number of subframes
        """

        # Init the engine
        self.reset(subframes=subframes)

        # Add the animation
        self.add(animation)

        # Current frames
        self.step()

    # ----------------------------------------------------------------------------------------------------
    # Animation functions launch
    # ----------------------------------------------------------------------------------------------------

    def go(self, compute, reset=None, view=None, subframes=0):
        """ > Legacy behavior : animates with functions

        Arguments
        ---------
        - compute (function) : computation function
        - reset (function = None) : reset function
        - view (function = None) : view function
        - subframes (int = 0) : subframes
        """
        self.animation(SimpleAnimation(compute, reset=reset, view=view), subframes=subframes)


# ====================================================================================================
# One single instance of Engine class
# ====================================================================================================

engine = Engine()

# ====================================================================================================
# Lock interface to avoid crashes
# ====================================================================================================

def lock_interface(value):
    bpy.context.scene.render.use_lock_interface = value

# ====================================================================================================
# Viewport animation
# Frame change - handler: frame_change_pre
# ====================================================================================================

def update(scene, depsgraph):

    # Rendering : update is done by before_render_image
    if engine.rendering:
        engine._depsgraph = depsgraph
        before_render_image(scene, engine.depsgraph)
        engine._depsgraph = None
    
    # View port update
    else:
        engine._scene = scene
        engine.step()

# ====================================================================================================
# Render animation
# - Capture start and end rendering
# - Call step at each new frame
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Render init - handler: render_init
# ----------------------------------------------------------------------------------------------------

def before_render(scene):

    print("Engine> Start rendering")

    # To be sure
    lock_interface(True)

    engine._rendering = True

# ----------------------------------------------------------------------------------------------------
# Render done - handler: render_complete and render_cancel
# ----------------------------------------------------------------------------------------------------

def after_render(scene):
    print("Engine> Render completed")
    engine._rendering = False
    engine._depsgraph = None

# ----------------------------------------------------------------------------------------------------
# Render image - handler: render_pre
# ----------------------------------------------------------------------------------------------------

def before_render_image(scene, depsgraph):

    # Points to the right scene (depsgraph has been set by update)
    engine._scene = scene
    print(f"Engine> Frame {engine.frame}")

    # one step

    engine.step()

# ====================================================================================================
# Animation root class
# ====================================================================================================

class Animation:

    # ----------------------------------------------------------------------------------------------------
    # Access to engine
    # ----------------------------------------------------------------------------------------------------

    @property
    def engine(self):
        return engine
    
    @property
    def time(self):
        return engine.time
    
    @property
    def delta_time(self):
        return engine.delta_time

    # ----------------------------------------------------------------------------------------------------
    # Three animation methods
    # - reset : called at first frame
    # - called ateach step and substep
    # - called to put the result into Blender
    # ----------------------------------------------------------------------------------------------------

    def reset(self):
        pass

    def compute(self):
        pass
        #raise NotImplementedError(f"{type(self).__name__} does not implement compute()")

    def view(self):
        pass

    # ----------------------------------------------------------------------------------------------------
    # Complementary
    # ----------------------------------------------------------------------------------------------------

    def before_compute(self):
        pass

    def after_compute(self):
        pass

    # ----------------------------------------------------------------------------------------------------
    # Baking
    # ----------------------------------------------------------------------------------------------------

    # ----- Animation data

    def get_frame_data(self):
        return None

    def set_frame_data(self, data):
        pass

    # ----- State data

    def get_frame_state(self):
        return None

    def set_frame_state(self, data):
        pass

    # ----------------------------------------------------------------------------------------------------
    # Single run
    # ----------------------------------------------------------------------------------------------------

    def go(self, subframes=0):
        engine.animation(self, subframes=subframes)

# ====================================================================================================
# Simple Animation
# ====================================================================================================

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

# ====================================================================================================
# Demo Animation
# ====================================================================================================

class DemoAnimation(Animation):
    def __init__(self):
        self.reset()

    def reset(self):
        from . mesh import Mesh
        self.cube = Mesh.cube()
    
    def compute(self):
        self.cube.points.position += self.engine.rng.uniform(-.1, .1, (len(self.cube.points), 3))

    def view(self):
        self.cube.to_object("Cube", shade_smooth=False)

    # Baking

    def get_frame_data(self):
        return self.cube.points.position
    
    def set_frame_data(self, data):
        self.cube.points.position = data

# ====================================================================================================
# Automation in scene
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Create properties in scene
# ----------------------------------------------------------------------------------------------------

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
# ----------------------------------------------------------------------------------------------------

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

        engine.init_bake()
        if engine.is_first_frame:
            engine.step()

        return {'FINISHED'}

# ----------------------------------------------------------------------------------------------------
# Bake files deletion operator
# ----------------------------------------------------------------------------------------------------

class npblenderDelBakeFiles(bpy.types.Operator):
    """Delete npblender bake files"""
    bl_idname = "scene.npblender_del_bake_files"
    bl_label = "Delete npblender bake files"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        context.scene.npblender_use_bake = False
        engine.init_bake()
        BakeFile.delete_bake_files(context.scene.npblender_bake_name)
        return {'FINISHED'}

# ----------------------------------------------------------------------------------------------------
# UI Panel
# ----------------------------------------------------------------------------------------------------

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
# ====================================================================================================

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

# ----------------------------------------------------------------------------------------------------
# Register
# ----------------------------------------------------------------------------------------------------

register()

