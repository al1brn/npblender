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

This module also provides a base `Animation` class which can be directly used
with function-based animations, and Blender operators/panels for
managing baked animations.
"""

import logging
import numpy as np

import bpy

from .bakefile import BakeFile

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
    """
    Animation engine for **npblender**.

    > ***Note:*** do not instantiate `Engine` class directly but rather use the single
    > and only instance `engine` avaiable as global variable in the module.

    `Engine` manages a list of [`Animations`][npblender.Animation]. At each frame change,
    the `compute` and `view` methods of `Animations` are called.

    If `subframes` is not zero, the `compute` is called as many times ad defined by
    this property. This allows more precision in simulations.

    Advanced animations can create classes based on [`Animation`][npblender.animation] or
    [`Simulation`][npblender.Simulation].

    Simple animations can rely on a single function which is passed as argument of [engine.go][npblender.Engine.go]
    method as shown below:

    ``` python
    from npblender import Mesh, engine

    # Move a cube of .01 per frame along x
    cube = Mesh.cube()
    def update():
        cube.points.x += .01
        cube.to_object("Cube")

    engine.go(update)
    ``` 

    The `engine` instance of `Engine` class exposes useful animation properties and methods such as
    `time` for current time, `delta_time` for simulations, `is_viewport` and `rendering` to adapt
    parameters to the context.

    Key features
    ------------
    - Global animation management (add, run, reset)
    - Frame stepping with optional **subframes**
    - **Baking** to disk via methods [`Animation.get_frame_data`][npblender.Animation.get_frame_data] and
        [`Animation.set_frame_data`][npblender.Animation.set_frame_data.
    - Integration with Blender render/viewport handlers and depsgraph
    - Support for both **viewport** updates and **render-time** updates

    Attributes
    ----------
    animations : list[npblender.Animation]
        Global list of registered animations driven by the engine.
    bake_file : npblender.bakefile.BakeFile or None
        Active bake file when baking is enabled in the scene, else `None`.
    time_offset : float
        Seconds added to the computed time (shifts the timeline origin).
    time_scale : float
        Global time scale multiplier (e.g., slow-motion).
    subframes : int
        Number of subframes per frame (the engine computes `subframes + 1` steps).
    subframe : int
        Current subframe index in `{0, …, subframes}` (`0` means the “main” frame).
    SEED : int
        Base RNG seed used to derive per-(frame, subframe) seeds.
    VERBOSE : bool
        If `True`, prints per-frame timings to the console.
    STEP_TIMES : bool
        If `True`, prints detailed I/O/compute times (load/compute/save/view).

    Properties
    ----------
    scene : bpy.types.Scene
        Active scene (or an overridden scene during handlers).
    fps : int
        Frames per second from `scene.render.fps`.
    frame : int
        Current frame number (overridable internally during handlers).
    is_first_frame : bool
        `True` when `frame == scene.frame_start`.
    time : float
        Current time in seconds, including `time_offset`, `time_scale`, and subframe.
    duration : float
        Timeline duration (seconds) given the current `time_scale`.
    delta_time : float
        Time step between substeps: `time_scale / fps / (subframes + 1)`.
    rendering : bool
        `True` when called from render handlers; `False` in viewport updates.
    is_viewport : bool
        Negation of `rendering`.
    use_motion_blur : bool
        Reflects `scene.render.use_motion_blur`.
    is_baked : bool
        `True` when a bake file is active (`bake_file is not None`).
    frame_seeds : numpy.ndarray
        Table of per-(frame, subframe) seeds derived from `SEED`.
    seed : int
        Seed for the current `(frame, subframe)`.
    rng : numpy.random.Generator
        RNG initialized from `seed`.
    depsgraph : bpy.types.Depsgraph
        Evaluated depsgraph for the current context/update.
    """

    VERBOSE = True
    STEP_TIMES = False

    # Global list of animations
    animations = [] 

    # Baking
    bake_file = None

    # random seed
    SEED = 8694853

    def __init__(self):
        """
        Initialize the animation engine.

        Sets default scene/frame references, time scaling/offset, subframe
        parameters, rendering flags, depsgraph cache, and RNG seed table cache.
        """
        self._scene = None
        self._frame = None

        self.time_offset = 0.
        self.time_scale  = 1.

        self.subframes   = 0
        self.subframe    = 0

        self._rendering  = False
        self._depsgraph  = None

        self._frame_seeds = None

    # ====================================================================================================
    # Engine reset
    # ====================================================================================================

    def reset(self, subframes=0):
        """
        Reset the engine state for a new run.

        Clears the animation list, sets the number of subframes, resets the
        subframe index to 0, and invalidates the cached RNG seed table.

        Parameters
        ----------
        subframes : int, default=0
            Number of subframes per frame. The engine will compute
            `subframes + 1` steps per frame.
        """

        self.subframes = subframes
        self.subframe  = 0
        self._frame_seeds = None

        self.animations.clear()

    # ====================================================================================================
    # Properties
    # ====================================================================================================

    @property
    def scene(self):
        """
        Active Blender scene.

        Returns
        -------
        bpy.types.Scene
            The cached scene during handlers if set, otherwise `bpy.context.scene`.
        """
        if self._scene is None:
            return bpy.context.scene
        else:
            return self._scene
        
    @property
    def fps(self):
        """
        Frames-per-second of the active scene.

        Returns
        -------
        int
            `scene.render.fps`.
        """
        return self.scene.render.fps
    
    @property
    def frame(self):
        """
        Current frame number.

        Returns
        -------
        int
            Cached frame override when set, otherwise `scene.frame_current`.
        """
        if self._frame is None:
            return self.scene.frame_current
        else:
            return self._frame
        
    @property
    def is_first_frame(self):
        """
        Whether the current frame is the timeline start.

        Returns
        -------
        bool
            `True` if `frame == scene.frame_start`, else `False`.
        """
        return self.frame == self.scene.frame_start
    
    @property
    def time(self):
        """
        Current time in seconds.

        Subframes are enumerated in the order `1, 2, ..., n-1, 0`. The `0`
        subframe corresponds to the “main” frame computed after substeps.

        > ***Note:*** Time can be controlled with `time_offset` and `time_scale`
        > properties.

        Returns
        -------
        float
            `time_offset + time_scale * frame / fps`, with `frame` adjusted
            for subframes when `subframe != 0`.
        """
        # subframes are enumerated in the order : 1, 2, ..., n-1, 0
        if self.subframe == 0:
            frame = self.frame
        else:
            frame = self.frame - 1 + self.subframe/(self.subframes + 1)

        return self.time_offset + self.time_scale*frame/self.fps
    
    @property
    def duration(self):
        """
        Timeline duration in seconds.

        Returns
        -------
        float
            `time_scale * (1 + frame_end - frame_start) / fps`.
        """
        return self.time_scale*(1 + self.scene.frame_end - self.scene.frame_start)/self.fps

    @property
    def delta_time(self):
        """
        Time step between steps or substeps.

        Returns
        -------
        float
            `time_scale / fps / (subframes + 1)`.
        """
        return self.time_scale/self.fps/(self.subframes + 1)
    
    @property
    def rendering(self):
        """
        Render-time flag.

        Returns
        -------
        bool
            `True` when called from render handlers, else `False`.
        """
        return self._rendering

    @property
    def is_viewport(self):
        """
        Viewport flag (negation of `rendering`).

        Returns
        -------
        bool
            `True` when updating in the viewport.
        """
        return not self._rendering

    @property
    def use_motion_blur(self):
        """
        Whether motion blur is enabled in the renderer.

        Returns
        -------
        bool
            `scene.render.use_motion_blur`.
        """
        return self.scene.render.use_motion_blur
    
    @property
    def is_baked(self):
        """
        Whether baking is active for the current scene.

        Baking is active as soon as it has been defined in the UI.

        Returns
        -------
        bool
            `True` if a bake file is active (`bake_file is not None`).
        """
        return self.bake_file is not None
    
    # ====================================================================================================
    # Randomness
    # ====================================================================================================

    @property
    def frame_seeds(self):
        """
        Per-(frame, subframe) RNG seeds table.

        Returns
        -------
        numpy.ndarray of dtype uint32, shape (frame_end + 1, subframes + 1)
            A table of deterministic seeds derived from `SEED`. Recomputed
            when dimensions change or if the cache is invalidated.
        """
        shape = (self.scene.frame_end + 1, self.subframes + 1)
        if self._frame_seeds is None or self._frame_seeds.shape != shape:
            rng = np.random.default_rng(self.SEED)
            self._frame_seeds = rng.integers(0, 1<<32, shape)

        return self._frame_seeds

    @property
    def seed(self):
        """
        Seed for the current (frame, subframe).

        Returns
        -------
        int
            `frame_seeds[frame, subframe]`.
        """
        return self.frame_seeds[self.frame, self.subframe]

    @property
    def rng(self):
        """
        Random number generator initialized from `seed`.

        Returns
        -------
        numpy.random.Generator
            Default PCG64 generator seeded with `seed`.
        """
        return np.random.default_rng(self.seed)
    
    # ====================================================================================================
    # Utilities
    # ====================================================================================================

    @property
    def depsgraph(self):
        """
        Evaluated dependency graph.

        Returns
        -------
        bpy.types.Depsgraph
            Cached depsgraph during handlers when available,
            otherwise `bpy.context.evaluated_depsgraph_get()`.
        """
        if self._depsgraph is None:
            return bpy.context.evaluated_depsgraph_get()
        else:
            return self._depsgraph

    def get_evaluated(self, spec):
        """
        Get an evaluated object from the depsgraph.

        Parameters
        ----------
        spec : str or bpy.types.Object
            Object name or object instance.

        Returns
        -------
        bpy.types.Object
            The evaluated object (`obj.evaluated_get(depsgraph)`).
        """
        obj = bpy.data.objects[spec] if isinstance(spec, str) else spec
        return obj.evaluated_get(self.depsgraph)

    # ====================================================================================================
    # Baking
    # ====================================================================================================

    def init_bake(self):
        """
        Initialize or disable baking based on scene properties.

        Creates a new [`BakeFile`][npblender.bakefile.BakeFile] when
        `scene.npblender_use_bake` is enabled and a name is set; otherwise
        clears `bake_file`.
        """
        name = self.scene.npblender_bake_name
        if self.scene.npblender_use_bake and name != "":
            print(f"Engine> init bake '{name}'")
            self.bake_file = BakeFile(name)
        else:
            self.bake_file = None

    def load_frame(self):
        """
        Try to load baked data/state for the current frame.

        Returns
        -------
        bool
            `True` if baked animation data (and optional state) were read and
            applied to all animations; `False` if no bake file or no entry.
        """

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
            if not anim.set_frame_data(data[i]):
                return False

        # ----- Frame state

        if self.bake_file.state_frame == self.frame:
            logging.debug(f"Engine> loading state at frame {self.frame}.")
            state_data = self.bake_file.read_state()
            for i, anim in enumerate(self.animations):
                anim.set_frame_state(state_data[i])

        return True

    def save_frame(self):
        """
        Save current animations' data and state to the bake file.

        Skips saving unless on the first frame or immediately after the last
        saved state frame; otherwise writes `animations` and `states` for
        the current `frame` into `bake_file`.
        """

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
        """
        Reset all animations at the first frame.

        Calls `reset()` on each registered animation and logs the reset.
        """
        # Track
        print("Engine> Reset")

        for anim in self.animations:
            anim.reset()

    # ----------------------------------------------------------------------------------------------------
    # Compute one step
    # ----------------------------------------------------------------------------------------------------

    def step(self):
        """
        Advance one frame step (with subframes) and update the view.

        Workflow:
        1) If first frame, initialize baking.
        2) If the frame is baked, load it; otherwise:
        - On the first frame, call `first_frame_reset()` and use 0 subframes.
        - For `i` in `1..subframes` then `0`, set `subframe` and call
            `compute()` on each animation.
        - Save the frame if baking is active.
        3) Call `view()` on each animation.

        Notes
        -----
        Subframe order is `1, 2, …, n-1, 0`, where `0` denotes the “main”
        frame computed after all substeps.
        """

        from time import time

        # ===== Preparation

        # FIRST FRAME : initialize baking
        if self.is_first_frame:
            self.init_bake()

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
                    anim.compute()
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
        """
        Register an animation to be driven by the engine.

        Parameters
        ----------
        animation : Animation
            An object exposing `reset()`, `compute()`, `view()` and optional
            baking hooks (`get_frame_data`, `set_frame_data`, `get_frame_state`,
            `set_frame_state`).
        """
        self.animations.append(animation)

    # ----------------------------------------------------------------------------------------------------
    # Single animation quick launch
    # ----------------------------------------------------------------------------------------------------

    def animation(self, animation, subframes=0):
        """
        Register and run a single animation (convenience).

        Resets the engine with the desired `subframes`, registers `animation`,
        then steps the current frame once.

        Parameters
        ----------
        animation : Animation
            The animation to run.
        subframes : int, default=0
            Number of subframes per frame.
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
        """
        Run a single animation (convenience).

        The animation is defined by the `compute` function and
        optionally by `reset` and `view` functions.

        Parameters
        ----------
        compute : callable
            Frame/subframe compute function.
        reset : callable, optional
            Optional first-frame reset function.
        view : callable, optional
            Optional view/update function.
        subframes : int, default=0
            Number of subframes per frame.
        """

        self.animation(Animation(compute, reset=reset, view=view), subframes=subframes)


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
    """
    Minimal base class for time-stepped animations.

    `Animation` defines the three lifecycle hooks that the [`Engine`][npblender.engine.Engine]
    calls on every frame/subframe: `reset()`, `compute()`, and `view()`. You can either
    subclass and override these methods, or pass callables to the constructor to set
    `_reset`, `_compute`, and `_view` dynamically. It also exposes lightweight accessors
    to the global engine (`engine`, `time`, `delta_time`) and optional baking hooks.

    Notes
    -----
    - `reset()` is called **once** on the first frame of a run.
    - `compute()` is called at each substep; the engine enumerates subframes as
    `1, 2, …, n-1, 0`, where `0` is the main frame after substeps.
    - `view()` is called after compute/load to push results into Blender.
    """

    def __init__(self, compute=None, reset=None, view=None):
        """
        Optionally provide function-based hooks instead of subclassing.

        If a callable is provided, it is bound to the corresponding private
        attribute (`_compute`, `_reset`, `_view`) that `compute()`, `reset()`,
        and `view()` will delegate to.

        Parameters
        ----------
        compute : callable or None, optional
            Step function called at each subframe.
        reset : callable or None, optional
            Initialization function called at the first frame.
        view : callable or None, optional
            Function that writes results back to Blender.
        """

        if compute is not None:
            self._compute = compute
        if reset is not None:
            self._reset = reset
        if view is not None:
            self._view = view

    # ----------------------------------------------------------------------------------------------------
    # Access to engine
    # ----------------------------------------------------------------------------------------------------

    @property
    def engine(self):
        """
        Global engine singleton.

        Returns
        -------
        Engine
            The module-level engine instance that drives the animation.
        """
        return engine
    
    @property
    def time(self):
        """
        Current time in seconds (proxy to `engine.time`).

        Returns
        -------
        float
            The timeline time including subframes, offset, and scale.
        """
        return engine.time
    
    @property
    def delta_time(self):
        """
        Time step between substeps (proxy to `engine.delta_time`).

        Returns
        -------
        float
            `time_scale / fps / (subframes + 1)`.
        """
        return engine.delta_time

    # ----------------------------------------------------------------------------------------------------
    # Three animation methods
    # - reset : called at first frame
    # - called ateach step and substep
    # - called to put the result into Blender
    # ----------------------------------------------------------------------------------------------------

    def reset(self):
        """
        First-frame initialization hook.

        > ***To be overridden***

        Delegates to `_reset()` if defined. Called once by the engine at the
        first frame of the run.
        """
        if hasattr(self, '_reset'):
            self._reset()

    def compute(self):
        """
        Per-substep compute hook.

        > ***To be overridden***

        Delegates to `_compute()` if defined. Called by the engine for each
        subframe and the main frame (`… , n-1, 0`). 
        """
        if hasattr(self, '_compute'):
            self._compute()

    def view(self):
        """
        Push results into Blender.

        > ***To be overridden***

        Delegates to `_view()` if defined. Called by the engine after compute or
        after loading a baked frame, to update Blender data/objects.
        """
        if hasattr(self, '_view'):
            self._view()

    # ----------------------------------------------------------------------------------------------------
    # Baking
    # ----------------------------------------------------------------------------------------------------

    # ----- Animation data

    def get_frame_data(self):
        """
        Return animation data to be baked for the current frame.

        Returns
        -------
        Any or None
            Per-animation payload to store in the bake file, or `None` to skip.
        """
        return None

    def set_frame_data(self, data):
        """
        Load animation data from the bake file for the current frame.

        Parameters
        ----------
        data : Any
            Payload previously produced by `get_frame_data()`.

        Returns
        -------
        bool
            `True` if the data was applied successfully, else `False`.
        """
        return False

    # ----- State data

    def get_frame_state(self):
        """
        Return state to be saved to resume simulation from the current frame..

        Returns
        -------
        Any or None
            Optional state payload 
        """
        return None

    def set_frame_state(self, data):
        """
        Restore persistent state saved with the current frame.

        The simulation can be resume for next frames.

        Parameters
        ----------
        data : Any
            State payload previously produced by `get_frame_state()`. :contentReference[oaicite:13]{index=13}
        """
        pass

    # ----------------------------------------------------------------------------------------------------
    # Single run
    # ----------------------------------------------------------------------------------------------------

    def go(self, subframes=0):
        """
        Run this animation once through the engine.

        Convenience wrapper for `engine.animation(self, subframes=subframes)`.

        Parameters
        ----------
        subframes : int, default=0
            Number of subframes per frame.

        Examples
        --------
        ```python
        anim = Animation(compute=lambda: ..., reset=lambda: ..., view=lambda: ...)
        anim.go(subframes=4)
        ```
        """
        engine.animation(self, subframes=subframes)

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
        return True

# ====================================================================================================
# Automation in scene
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# Create properties in scene
# ----------------------------------------------------------------------------------------------------

def create_scene_properties():

    #raise Exception("Oups")
    Scene = bpy.types.Scene

    if not hasattr(Scene, "npblender_use_bake"):
        Scene.npblender_use_bake = bpy.props.BoolProperty(
            name="Bake animation",
            default=False,
        )
    if not hasattr(Scene, "npblender_temp_folder"):
        Scene.npblender_temp_folder = bpy.props.StringProperty(
            name="Temp folder",
            default="",  # pas de bpy.context ici !
        )
    if not hasattr(Scene, "npblender_bake_name"):
        Scene.npblender_bake_name = bpy.props.StringProperty(
            name="Name",
            default="bake",
        )

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

    print("Registring npblender engine")

    create_scene_properties()

    # Clean and append
    bpy.app.handlers.frame_change_post[:] = [
        h for h in bpy.app.handlers.frame_change_post if h.__name__ != update.__name__
    ]
    bpy.app.handlers.frame_change_post.append(update)

    bpy.app.handlers.render_init[:] = [h for h in bpy.app.handlers.render_init if h.__name__ != before_render.__name__]
    bpy.app.handlers.render_init.append(before_render)

    bpy.app.handlers.render_complete[:] = [h for h in bpy.app.handlers.render_complete if h.__name__ != after_render.__name__]
    bpy.app.handlers.render_complete.append(after_render)

    bpy.app.handlers.render_cancel[:] = [h for h in bpy.app.handlers.render_cancel if h.__name__ != after_render.__name__]
    bpy.app.handlers.render_cancel.append(after_render)

    bpy.utils.register_class(npblenderBake)
    bpy.utils.register_class(npblenderDelBakeFiles)
    bpy.utils.register_class(npblenderBakePanel)

def unregister():
    for L, f in [
        (bpy.app.handlers.frame_change_post, update),
        (bpy.app.handlers.render_init,       before_render),
        (bpy.app.handlers.render_complete,   after_render),
        (bpy.app.handlers.render_cancel,     after_render),
    ]:
        try:
            L.remove(f)
        except ValueError:
            pass

    for cls in (npblenderBakePanel, npblenderDelBakeFiles, npblenderBake):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    # No engine
    global engine
    engine = None
