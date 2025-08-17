
import inspect
import numpy as np

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
        self.func   = func
        self.args   = list(args)
        self.kwargs = dict(kwargs)

        # ----- Expected arguments

        self.arg_sim_index = None
        self.func_kwargs = []

        sig = inspect.signature(func)
        for index, (pname, p) in enumerate(sig.parameters.items()):
            if pname == 'self':
                self.arg_sim_index = 0

            if pname == 'simulation':
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    self.arg_sim_index = index

        self.sig_kwargs = [pname for pname, p in sig.parameters.items()
                           if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        if 'simulation' in self.sig_kwargs and self.arg_sim_index is not None:
            self.sig_kwargs.remove('simulation')

        self.status     = Action.NOT_STARTED
        self.start      = start
        self.duration   = duration
        self.flags      = flags

        self.started_at = None

    # ----------------------------------------------------------------------------------------------------
    # dump
    # ----------------------------------------------------------------------------------------------------

    def __str__(self):
        #return f"<Action: {self.args=}, {self.kwargs=}, {self.arg_sim_index=}, {self.sig_kwargs=}>"                
        if self.start == 0 and self.duration is None:
            stime = "No timing"

        else:
            if self.duration == 0:
                stime = f"Event  at {self.start:.2f}"
            else:
                stime = f"start: {self.start:.2f} during {self.duration:.2f}"
            stime += f" {['Not started', 'Active', 'Done'].index(self.status)}"

        return f"<Action '{self.func.__name__}', {stime}>" #, {'enabled' if self.is_enabled else 'disabled'}>"

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

        if self.status == Action.NOT_STARTED:
            # Start time before simulation start time
            if (simulation.time - self.start) > simulation.delta_time:
                self.started_at = self.start
            else:
                self.started_at = simulation.time

        elapsed = simulation.time - self.started_at

        for arg_name, value in {
            'simulation' : simulation,
            'elapsed': elapsed, 
            't': simulation.time, 
            'dt': simulation.delta_time,
            }.items():
            if arg_name in self.sig_kwargs:
                self.kwargs[arg_name] = value

        if self.arg_sim_index is not None:
            self.args[self.arg_sim_index] = simulation

        try:
            return self.func(*self.args, **self.kwargs)
        except Exception as e:
            raise Exception(
                f"Error while calling {self.func.__name__}.\n{e}\n"
                f"args: {self.args}\n"
                f"kwargs: {self.kwargs}\n"
                f"arg_sim_index: {self.arg_sim_index}")
    
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
    

if __name__ == '__main__':

    class Simu:
        def __init__(self, time=0., delta_time=.1):
            self.time = time
            self.delta_time = .1

        def __str__(self):
            return f"<Simu {self.time}>"

        def meth(self, a, b=17, t=None):
            print(f"Method: {str(self)=}, {a=}, {b=}, {t=}")

    def simple(a, b, c=10, t=None, dt=None, elapsed=None):
        print(f"f: {a=}, {b=}, {c=} / {t=}, {dt=}, {elapsed=}")

    def with_simu(simulation, a, b=1, c=10):
        print(f"f: {simulation}, {a=}, {b=}, {c=}")

    for func in [simple, with_simu, Simu.meth]:
        print('\ntest', func.__name__)
        action = Action(func, 10, 11)
        simu = Simu()
        simu.time = 3.14

        action(simu)

        print("ARGS  ", action.args)
        print("KWARGS", action.kwargs)
        print("SIMUL", action.arg_sim_index)
        print()

    # Check standard method

    print("sim with time = 1")
    sim1 = Simu(time=1)
    sim1.meth(123)

    print("sim with time = 2")
    sim2 = Simu(time=2)
    sim2.meth(123)

    print("\nAction using method")
    action = Action(sim1.meth, 1, 2)
    print(action)

    print("\nCalling with sim 1")
    action(sim1)
    print("\nCalling with sim 2")
    action(sim2)
    print()

    from collections.abc import Callable

    print(callable(action.func))
    print(callable(action))
    print(callable(np.asarray(123)))

    





