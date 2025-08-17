#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:27:01 2023

@author: alain
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 07:45:22 2023

@author: alain
"""

import inspect

import numpy as np

from .engine import engine, Animation
from . import blender

PI  = np.pi
TAU = np.pi*2

# =============================================================================================================================
# A simulation is basically a loop calling actions
#
# The simulation maintains:
# - the current time
# - dt of each loop
#
# Sub steps can be defined if more loops are required between two frames
#
# Functions called at each step are stored in a list
# The call to the function can be disabled
# Enabling / disabling functions can be driven by time (ie event)

# =============================================================================================================================
# Action / Event
#
# Actions are called at each loop.
# An Action maintains a enabled flag which can incremented / decremented to disable or reenabled the action
# Action have a start time and a duration. By default, start time is 0 and duration is None for actions always active.
#
# Main loop is made of
#
# def step():
#   actions : after = False
#   exec_step()
#   actions : after = True

class Action:

    NOT_STARTED = 0
    ACTIVE      = 1
    DONE        = 2

    def __init__(self, func, *args, start=0, duration=None, after=False, **kwargs):
        """ An action is a function called at each loop.

        The function must accept simulation as first argument.
        It can be a method of a Simulation class.

        Hereafter is an example of function simulating gravity;

        ``` python
        def gravity(simulation, g=-9.86):
            simulation.points.speed += (0, 0, g*simulation.dt)
        ```

        The argument <#start> specifies when to start the action. The actual time at which the
        action will start can differ linked to discontinuity of time. The <#started_at> property
        is initialized with the actual time at which the actions starts. The function can access
        the exact elapsed time from the actual start by accepting an 'elapsed' argument in kwargs:

        ``` python
        def func(simulation, elapsed=None):
            # elapsed (float) : elapsed time since the action started
            pass

        action = Action(func, ..., elapsed=None)
        ```

        Properties
        ----------
        - func (function) : function to call
        - args (*args) : arguments to pass the the function
        - kwargs (**kwargs) : keyword arguments to pass to function
        - status (enum in (NOT_STARTED, ACTIVE, DONE)) : action status
        - start (float = 0) : start time
        - duration (float = None) : duration, (once if numm, never stops if None)
        - started_at (float) : actual start time
        - after (bool = False) : action is triggred after exec

        Arguments
        ---------
            - func (function (simulation, *args, **kwargs)) : the function to call at each loop
            - *args : args to pass to the function
            - start (float = 0) : start time
            - duration (float = None) : duration. Never interrupted if None, is called once if equal to 0.
            - after (bool = False) : action is triggred after exec
            - **kwargs : keyword arguments to pass to the function
        """

        self.func    = func
        self.args    = list(args)
        self.kwargs  = {**kwargs}

        self.status     = Action.NOT_STARTED
        self.start      = start
        self.duration   = duration
        self.after      = after

        self.started_at = None

    # ====================================================================================================
    # Representation

    def __str__(self):
        if self.start == 0 and self.duration is None:
            stime = "No timing"

        else:
            if self.duration == 0:
                stime = f"Event  at {self.start:.2f}"
            else:
                stime = f"start: {self.start:.2f} during {self.duration:.2f}"
            stime += f" {['Not started', 'Active', 'Done'].index(self.status)}"

        return f"<Action {self.func.__name__}, {stime}, {'enabled' if self.is_enabled else 'disabled'}>"

    # ----------------------------------------------------------------------------------------------------
    # Reset

    def reset(self):
        """ Reset the action
        """
        self.status     = Action.NOT_STARTED
        self.started_at = None

    # ====================================================================================================
    # Call the action from the simulation

    def call(self, simulation):

        if self.status == Action.NOT_STARTED:
            # Start time before simulation start time
            if (simulation.time - self.start) > simulation.delta_time:
                self.started_at = self.start
            else:
                self.started_at = simulation.time

        elapsed = simulation.time - self.started_at

        if 'elapsed' in self.kwargs.keys():
            self.kwargs['elapsed'] = elapsed

        return self.func(simulation, *self.args, **self.kwargs)

    def __call__(self, simulation):

        # Not yet
        if simulation.time < self.start:
            return None

        # Already done
        if self.status == Action.DONE:
            return None

        # Let's go
        res = self.call(simulation)
        self.status = Action.ACTIVE

        # No more
        if self.duration is not None and simulation.time >= self.start + self.duration:
            self.status = Action.DONE

        # Return the result
        return res

# =============================================================================================================================
# Simulation
#
# A simulation is a list of actions

class Simulation(Animation):
    def __init__(self, object=None):
        """ > Simulation

        This class enhances the basic Simulation class defined in Engine.
        It manages:
        - a geometry
        - a target object name
        - a list of <!Action"Actions>

        By implementing <#get_animation> and <#set_animation>, the simulation can be baked if it
        changes only the <#geometry>, otherwise these methods must be overloaded.

        It implements the following animation methods:
        - reset : reset the actions
        - before_compute : run the actions before (default)
        - after_compute : run the actions after (special)
        - view : call geometry.to_object(target_object)
        - get_animation : save the geometry
        - set_animation : restore the geometry

        Properties
        ----------
        - actions (list) : list of actions
        - geometry (Geometry) : animated geometry
        - object (str or Blender object) : the object to animate

        Arguments
        ---------
        - target_object (str or Blender object) : the object to animate
        """
        super().__init__()

        self.actions = []
        self.geometry = None
        self.object = object


    # ====================================================================================================
    # Actions and event

    @classmethod
    def str_to_func(cls, func):
        if isinstance(func, str):
            return getattr(cls, func)

        elif hasattr(func, '__func__'):
            return func.__func__

        else:
            return func

    def add_action(self, func, *args, start=0, duration=None, after=False, **kwargs):
        """ Add an action to the simulation

        To add an event, you can use <#add_event>.

        Arguments
        ---------
        - func (function) : function of template f(simulation, *args, **kwargs)
        - top (float = 0) : start time for the actiuon
        - duration (float = None) : duration of the action (0: call once, None: never stops)
        - after (bool = False) : exec the action after the exec_loop

        Returns
        -------
        - Action : the action added to the simulation
        """
        action = Action(self.str_to_func(func), *args, start=start, duration=duration, after=after, **kwargs)
        self.actions.append(action)
        return action

    def add_event(self, func, *args, start=0, after=False, **kwargs):
        """ Add an event to the simulation

        The event is executed once. To add an action called at each step, use <#add_action>.

        Arguments
        ---------
        - func (function) : function of template f(simulation, *args, **kwargs)
        - top (float = 0) : start time for the actiuon
        - after (bool = False) : exec the action after the exec_loop

        Returns
        -------
        - Action : the event added to the simulation
        """

        action = Action(self.str_to_func(func), *args, start=start, duration=0., after=after, **kwargs)
        self.actions.append(action)
        return action

    # ====================================================================================================
    # Simulation

    # ----------------------------------------------------------------------------------------------------
    # Reset

    def reset(self):
        """ Reset the simulation
        """
        super().reset()

        for action in self.actions:
            action.reset()

    # ----------------------------------------------------------------------------------------------------
    # Before compute

    def before_compute(self):

        super().before_compute()

        for action in self.actions:
            if action.after:
                continue
            action(self)

    # ----------------------------------------------------------------------------------------------------
    # After compute

    def after_compute(self):

        super().after_compute()

        for action in self.actions:
            if not action.after:
                continue
            action(self)

    # ----------------------------------------------------------------------------------------------------
    # View

    def view(self):
        if self.geometry is not None and self.object is not None:
            self.geometry.to_object(self.object)

    # ----------------------------------------------------------------------------------------------------
    # Get animation

    def get_animation(self):
        if self.geometry is None:
            return {}
        else:
            return self.geometry.save()

    def set_animation(self, data):
        if self.geometry is None:
            return
        self.geometry.restore(data)
