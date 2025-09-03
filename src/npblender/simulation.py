
import inspect

import numpy as np
from .enginemod import Animation
from .maths.easings import maprange
from .maths import Quaternion

# ====================================================================================================
# Action
# ====================================================================================================

class Action:

    NOT_STARTED = 0
    ACTIVE      = 1
    DONE        = 2

    def __init__(self, func, *args, start=0, duration=None, flags=0, **kwargs):
        """ Simulation action : calling a function when triggered by the simulation.

        An action has basically a start and end time. It can also be an event when it
        is called only once.

        The function template is:
        ``` python
        def func(*args, **kwargs):
        ``` 

        The func signature is analyzed and some arguments can be added:
        - simulation : the global simulation
        - t : actual time

        Arguments
        ---------
        - func (function): the function to call
        - args : func arguments
        - start (float = 0) : start time
        - duration (float = None) : action duration (0 = event : call once, None: call forever)
        - flags (int = 0) : user flags
        - kwargs (dict) : function keyword arguments
        """
        # ----------------------------------------------------------------------------------------------------
        # Function arguments
        # ----------------------------------------------------------------------------------------------------

        self.sim_arg_index = None
        self.kwargs = {}
        nargs = 0
        sig = inspect.signature(func)

        for name, param in sig.parameters.items():
            if isinstance(param.default, type) and (param.default == param.empty):
                # Note that self is exposed when method is passed as class property
                # not as instance property
                if name in ['self', 'simulation']:
                    self.sim_arg_index = nargs
                nargs += 1
            else:
                self.kwargs[name] = param.default
                if name == 'simulation':
                    self.sim_arg_index = None

        # ----- Replace by passed arguments
        # The number of function positional arguments must match the number of provided arguments
        # except for self or simulation

        if nargs == len(args):
            self.args = list(args)

        elif (len(args) == nargs - 1) and (self.sim_arg_index is not None):
            self.args = [None]*nargs
            index = 0
            for v in args:
                if index == self.sim_arg_index:
                    index += 1
                self.args[index] = v
                index += 1

        else:
            raise AttributeError(
                f"Function '{func.__name__}({sig})' takes {nargs} positional arguments, "
                f"but only {len(args)} have been provided: {args}"
                )

        # ----- kwargs
        for k, v in kwargs.items():
            if not k in self.kwargs:
                raise AttributeError(
                    f"Function '{func.__name__}{sig}' doesn't take '{k}' keyword argument")

            self.kwargs[k] = v

        # ----------------------------------------------------------------------------------------------------
        # Initialize
        # ----------------------------------------------------------------------------------------------------

        self.func       = func
        self.status     = Action.NOT_STARTED
        self.start      = start
        self.duration   = duration
        self.flags      = flags

        self.started_at = None

    # ----------------------------------------------------------------------------------------------------
    # dump
    # ----------------------------------------------------------------------------------------------------

    def __str__(self):
        if self.duration is None:
            stime = f"start: {self.start:.2f}, forever"

        else:
            if self.duration == 0:
                stime = f"event at {self.start:.2f}"
            else:
                stime = f"start: {self.start:.2f} during {self.duration:.2f}"
            stime += f" {['Not started', 'Active', 'Done'][self.status]}"

        return f"<Action '{self.func.__name__}', {stime}>" 

    # ----------------------------------------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------------------------------------

    def reset(self):
        """ Reset the action
        """
        self.status     = Action.NOT_STARTED
        self.started_at = None

    # ====================================================================================================
    # Call the action from the simulation
    # ====================================================================================================

    def call_function(self, simulation):

        # --------------------------------------------------
        # Elapsed time
        # --------------------------------------------------

        if self.status == Action.NOT_STARTED:
            # Start time before simulation start time
            if (simulation.time - self.start) > simulation.delta_time:
                self.started_at = self.start
            else:
                self.started_at = simulation.time

        elapsed = simulation.time - self.started_at

        # --------------------------------------------------
        # Simulation and time arguments
        # --------------------------------------------------

        for arg_name, value in {
            'simulation' : simulation,
            'elapsed': elapsed, 
            't': simulation.time, 
            'dt': simulation.delta_time,
            }.items():
            if arg_name in self.kwargs:
                self.kwargs[arg_name] = value

        # Simulation can be self or positional argument
        # rather than keyword argument
        if self.sim_arg_index is not None:
            self.args[self.sim_arg_index] = simulation

        # --------------------------------------------------
        # Arguments can be functions of time
        # --------------------------------------------------

        args = list(self.args)
        for i, v in enumerate(self.args):
            v = args[i]
            if hasattr(v, '__call__'):
                args[i] = v(elapsed)

        kwargs = {}
        for k, v in self.kwargs.items():
            if hasattr(v, '__call__'):
                kwargs[k] = v(elapsed)
            else:
                kwargs[k] = v

        # --------------------------------------------------
        # Trying to call the function
        # --------------------------------------------------

        try:
            return self.func(*args, **kwargs)
        
        except Exception as e:
            raise Exception(
                f"Error while calling {self.func.__name__}{inspect.signature(self.func)}.\n{e}\n"
                f"args: {args}\n"
                f"kwargs: {kwargs}\n"
                f"sim_arg_index: {self.sim_arg_index}")
    
    # ====================================================================================================
    # Loop call : call the function if time conditions are fulfilled
    # ====================================================================================================

    def __call__(self, simulation):

        # Not yet
        if simulation.time < self.start:
            return None

        # Already done
        if self.status == Action.DONE:
            return None
        
        # Let's go
        res = self.call_function(simulation)
        self.status = Action.ACTIVE

        # No more
        if self.duration is not None and simulation.time >= self.start + self.duration:
            self.status = Action.DONE

        # Return the result
        return res

# ====================================================================================================
# Simulation skeleton
# ====================================================================================================

class Simulation(Animation):

    def __init__(self, compute=None, reset=None, view=None):
        
        super().__init__(compute=compute, reset=reset, view=view)

        self._actions = []

    # ----------------------------------------------------------------------------------------------------
    # Dump
    # ----------------------------------------------------------------------------------------------------

    def __str__(self):
        return f"Simulation: {len(self.actions)} actions>"
    
    # ====================================================================================================
    # Property
    # ====================================================================================================

    @property
    def actions(self):
        if not hasattr(self, '_actions'):
            self._actions = []
        return self._actions

    @property
    def points(self):
        if hasattr(self, "geometry"):
            return self.geometry.points
        raise Exception(f"'points' property not defined for the class '{type(self).__name__}'")
    
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
    
    # ----------------------------------------------------------------------------------------------------
    # Run the actions
    # ----------------------------------------------------------------------------------------------------

    def run_actions(self, flags=None):
        for action in self.actions:
            if flags is None or (flags & actions.flags):
                action(self)
    
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
    # Compute
    # ----------------------------------------------------------------------------------------------------

    def compute(self):
        if hasattr(self, '_compute'):
            self._compute()
        else:
            self.run_actions()

    # ----------------------------------------------------------------------------------------------------
    # View
    # ----------------------------------------------------------------------------------------------------

    def view(self):
        geo = getattr(self, "geometry", None)
        if (not hasattr(self, '_view')) and (geo is not None):
            geo.to_object("Simulation")
        else:
            super().view()

    # ----------------------------------------------------------------------------------------------------
    # Baking
    # ----------------------------------------------------------------------------------------------------

    def get_frame_data(self):
        geo = getattr(self, "geometry", None)
        if geo is not None:
            return geo.to_dict()
    
    def set_frame_data(self, data):
        geo = getattr(self, "geometry", None)
        if geo is not None:
            self.geometry = geo.from_dict(data)
            print("LOADED", self.geometry)
            return True
        return False

    # ====================================================================================================
    # Useful actions
    # ====================================================================================================

    def change_attribute(self, attribute, value, incr=None, factor=None):
        """ Modify a points attribute

        ``` python
        # gravity
        self.add_action("change_attribute", "accel", value=(0, 0, -9.81))
        ```
        """
        if value is None:
            return

        value = np.asarray(value)
        if factor is not None:
            shape = self.points[attribute].shape

            factor = np.reshape(factor, (-1,) + (1,)*(len(shape) - 1))
            value = np.broadcast_to(value, shape)*factor

            print(f"DEBUG -->: {value.shape=}")
            print(value[:5])

        if incr == '+':
            self.points[attribute] += value
        elif incr == '-':
            self.points[attribute] -= value
        elif incr == '*':
            self.points[attribute] *= value
        elif incr == '/':
            self.points[attribute] /= value
        else:
            self.points[attribute] = value

    # ----------------------------------------------------------------------------------------------------
    # Gravity
    # ----------------------------------------------------------------------------------------------------

    def gravity(self, g=np.asarray([0, 0, -9.81]), factor=None):
        self.change_attribute("accel", value=g, factor=factor)

    # ----------------------------------------------------------------------------------------------------
    # A force
    # ----------------------------------------------------------------------------------------------------

    def force(self, force, factor=None):
        self.change_attribute("force", value=force, factor=factor)

    # ----------------------------------------------------------------------------------------------------
    # A torque
    # ----------------------------------------------------------------------------------------------------

    def torque(self, torque, factor=None):
        self.change_attribute("torque", value=force, factor=factor)

    # ----------------------------------------------------------------------------------------------------
    # Newton's Law
    # ----------------------------------------------------------------------------------------------------

    def newton_law(self, G=1, power=2, min_distance=.001):
        """ Newton's law between points
        
        The force between two points is given by:
        
        F = G.m1.m2 / dist**p
        """

        pts = self.points.ravel()
        if len(pts) < 2:
            return
        
        # Vectors between couples of points
        v = pts.position - pts.position[:, None]

        # Distance
        d = np.linalg.norm(v, axis=-1)
        close = d < min_distance

        d[close] = 1
        F = (G*v)*d[..., None]**(-power - 1)
        F *= pts.mass[:, None, None]
        F *= pts.mass[None, :, None]
        F[close] = 0

        pts.force += np.sum(F, axis=0)

    # ----------------------------------------------------------------------------------------------------
    # Central force
    # ----------------------------------------------------------------------------------------------------

    def central_force(self, location=(0, 0, 0), force_factor=1., power=-2, min_distance=.001):

        pts = self.points.ravel()

        v = pts.position - location
        d = np.linalg.norm(v, axis=-1)
        close = d < min_distance

        d[close] = 1
        F = (force_factor*v)*(d[:, None]**(power - 1))
        F[close] = 0

        pts.force += F

    # ----------------------------------------------------------------------------------------------------
    # Centrifugal and Coriolis acceleration
    # ----------------------------------------------------------------------------------------------------

    def centrifugal(self, location=(0, 0, 0), omega=1, axis=(0, 0, 1), coriolis_factor=1.):

        pts = self.points.ravel()

        axis = np.asarray(axis)

        # Position relatively to the axis of rotation
        v = pts.position - location

        # Decompose height / vector to axis then distance
        z  = np.einsum('...i, ...i', v, axis)
        xy = v - axis*z[..., None]

        d = np.linalg.norm(xy, axis=-1)

        # Centrifugal
        acc = xy*(omega*omega)

        # Coriolis
        # Factor articially controls the intensity of the Coriolis force

        acc += np.cross((-2*omega)*axis, pts.speed)*coriolis_factor

        pts.accel += acc

    # ----------------------------------------------------------------------------------------------------
    # Viscosity :  slow down speed according a power law
    # ----------------------------------------------------------------------------------------------------

    def viscosity(self, viscosity_factor=1., power=2, max_force=None, fluid_speed=None):

        from collections.abc import Callable

        pts = self.points.ravel()
        speed = pts.speed

        # Speed relative to fluid speed
        if fluid_speed is None:
            pass

        elif callable(fluid_speed):
            spped -= fluid_speed(pts.position)

        else:
            speed -= fluid_speed

        # Direction and norm
        nrm = np.linalg.norm(speed, axis=-1)
        nrm[nrm < .001] = 1
        u = speed/nrm[:, None]

        # Raw force
        if 'viscosity' in pts.actual_names:
            viscosity_factor = viscosity_factor*pts.viscosity
        
        F = viscosity_factor*(nrm**power)
        
        # Make sure the force doesn't invese de speed
        max_F = pts.mass*nrm/self.delta_time
        F = np.minimum(F, max_F)

        pts.force -= u*F[:, None]

    # ====================================================================================================
    # Simulation
    # ====================================================================================================

    # ----------------------------------------------------------------------------------------------------
    # Dynamics
    # ----------------------------------------------------------------------------------------------------

    def compute_motion(self, translation=True, rotation=True, torque=False, flags=None):

        pts = self.points.ravel()

        # ---------------------------------------------------------------------------
        # Prepare
        # ---------------------------------------------------------------------------

        # Reset the acceleration and force
        if translation:
            pts.accel = 0
            pts.force = 0

        # Reset angular torque
        if torque:
            pts.torque = 0

        # ---------------------------------------------------------------------------
        # Run the actions
        # ---------------------------------------------------------------------------

        self.run_actions(flags=flags)

        # ---------------------------------------------------------------------------
        # Translation
        # ---------------------------------------------------------------------------

        if translation:
            F = pts.force
            mask = pts.mass != 0
            F[mask] /= pts.mass[mask, None]
            F[~mask] = 0

            # Full acceleration
            accel = pts.accel + F

            # Move the particles
            new_speed = pts.speed + accel*self.delta_time
            pts.position += (pts.speed + new_speed)*(self.delta_time/2)
            pts.speed = new_speed

        # ---------------------------------------------------------------------------
        # Rotation
        # ---------------------------------------------------------------------------

        if torque:
            torque = pts.torque
            mask = pts.moment != 0
            torque[mask] /= pts.moment[mask, None]
            torque[~mask] = 0

            omega = pts.omega
            new_omega = omega + torque*self.delta_time
            pts.omega = (omega + new_omega)*(self.delta_time/2)

        if torque or rotation:
            domg = pts.omega*self.delta_time
            ag = np.linalg.norm(domg, axis=-1)
            mask = ag != 0
            domg[mask] = domg[mask]/ag[mask, None]
            domg[~mask] = (1, 0, 0)

            quat = Quaternion.from_axis_angle(domg, ag)

            print("DEBUG", type(quat), type(pts.rotation))


            new_rot = quat @ pts.rotation
            if "euler" in pts.euler:
                pts.euler = new_rot.as_euler()
            else:
                pts.quat = new_rot.as_quaternion()



    






    

