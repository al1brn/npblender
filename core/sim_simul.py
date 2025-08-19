
import numpy as np
from .engine import Animation
from .sim_action import Action

# ====================================================================================================
# Simulation skeleton
# ====================================================================================================

class Simulation(Animation):

    BEFORE = 1
    AFTER  = 2

    def __init__(self, geometry=None, object=None, reset=None, view=None):
        
        super().__init__(reset=reset, view=view)

        self.actions = []

        self.geometry = geometry
        self.object = object

    # ----------------------------------------------------------------------------------------------------
    # Dump
    # ----------------------------------------------------------------------------------------------------

    def __str__(self):
        return f"Simulation: {len(self.actions)} actions>"
    
    # ====================================================================================================
    # Points
    # ====================================================================================================

    @property
    def points(self):
        if self.geometry is None:
            return None
        else:
            return self.geometry.points
    
    # ====================================================================================================
    # Actions management
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # An action can be specified by its name
    # ----------------------------------------------------------------------------------------------------

    def str_to_func(self, func):
        if isinstance(func, str):
            return getattr(self, func)

        elif hasattr(func, '__func__'):
            return func.__func__

        else:
            return func
        
    # ----------------------------------------------------------------------------------------------------
    # Add an action
    # ----------------------------------------------------------------------------------------------------

    def add_action(self, func, *args, start=0, duration=None, flags=0, **kwargs):
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
        action = Action(self.str_to_func(func), *args, start=start, duration=duration, flags=flags, **kwargs)
        self.actions.append(action)
        return action

    # ----------------------------------------------------------------------------------------------------
    # Add an event
    # ----------------------------------------------------------------------------------------------------

    def add_event(self, func, *args, start=0, flags=0, **kwargs):
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

        action = Action(self.str_to_func(func), *args, start=start, duration=0., flags=flags, **kwargs)
        self.actions.append(action)
        return action
    
    # ====================================================================================================
    # Simulation
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------------------------------------

    def reset(self):
        """ Reset the simulation
        """
        super().reset()
        for action in self.actions:
            action.reset()

    # ----------------------------------------------------------------------------------------------------
    # Before compute
    # ----------------------------------------------------------------------------------------------------

    def before_compute(self):
        super().before_compute()
        for action in self.actions:
            if action.flags & self.BEFORE:
                action(self)

    # ----------------------------------------------------------------------------------------------------
    # Compute
    # ----------------------------------------------------------------------------------------------------

    def compute(self):
        super().before_compute()
        for action in self.actions:
            if action.flags & (self.BEFORE | self.AFTER):
                continue
            action(self)

    # ----------------------------------------------------------------------------------------------------
    # After compute
    # ----------------------------------------------------------------------------------------------------

    def after_compute(self):
        super().after_compute()
        for action in self.actions:
            if action.flags and self.AFTER :
                action(self)

    # ----------------------------------------------------------------------------------------------------
    # View
    # ----------------------------------------------------------------------------------------------------

    def view(self):
        if self._view is None:
            if self.geometry is None or self.object is None:
                return
            self.geometry.to_object(self.object)
        else:
            super().view()

    # ----------------------------------------------------------------------------------------------------
    # Baking : save the geometry
    # ----------------------------------------------------------------------------------------------------

    def get_frame_data(self):
        if self.geometry is None:
            return {}
        else:
            return self.geometry.to_dict()

    def set_frame_data(self, data):
        if self.geometry is None:
            return
        self.geometry = self.geometry.from_dict(data)

    # ====================================================================================================
    # Standard actions
    # ====================================================================================================

    def gravity(self, g=[0, 0, -9.81]):
        self.points.acceleration += g

    






    

