# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module defines the class :class:`CpoSolverListener` that allows to be warned about different steps of the solve.

Any number of listeners can be added to a solver using the method :meth:`docplex.cp.solver.solver.CpoSolver.add_listener`.
Listeners can also be added on the model object using :meth:`docplex.cp.model.CpoModel.add_listener`

This module also defines some default useful listeners:

 * :class:`AutoStopListener`: Listener that stops the solve if configurable conditions are reached.
 * :class:`DelayListener`: Utility listener that waits some time at each solution found.
 * :class:`SolverProgressPanelListener`: implements a progress panel that appears when the solve is started.
   This panel is based on the package *Tkinter* that is available only in Python 2.7.14 and Python 3.X.

All these listeners are effective if the model provides multiple solutions. They are then more adapted to
optimization problems.

To be able to process multiple solutions, the model should be solved:

 * using the solution iterator given by method :meth:`docplex.cp.model.CpoModel.start_search`,
 * or using the method :meth:`docplex.cp.model.CpoModel.solve`
   but setting the parameter *context.solver.solve_with_start_next* to True.

*New in version 2.8.*

Detailed description
--------------------
"""


###############################################################################
## Listener class
###############################################################################

class CpoSolverListener(object):
    """ Solve listener allows to be warned about different solving steps.

    This class is an 'abstract' class that must be extended by actual listener implementation.
    All method of this class are empty.

    *New in version 2.8.*
    """

    def solver_created(self, solver):
        """ Notify the listener that the solver object has been created.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        pass


    def start_solve(self, solver):
        """ Notify that the solve is started.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        pass


    def end_solve(self, solver):
        """ Notify that the solve is ended.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        pass


    def result_found(self, solver, sres):
        """ Signal that a result has been found.

        This method is called every time a result is provided by the solver. The result, in particular the last one,
        may not contain any solution. This should be checked calling method sres.is_solution().

        This method replaces deprecated method solution_found() that is confusing as result may possibly
        not contain a solution to the model.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
            sres:   Solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`
        """
        pass


    def start_refine_conflict(self, solver):
        """ Notify that the refine conflict is started.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        pass


    def end_refine_conflict(self, solver):
        """ Notify that the refine conflict is ended.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        pass


    def conflict_found(self, solver, cflct):
        """ Signal that a conflict has been found.

        This method is called when a conflict result is found by the solver when method refine_conflict() is called.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
            cflct:  Conflict descriptor, object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`
        """
        pass


    def new_log_data(self, solver, data):
        """ Signal a new piece of log data.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
            data:   New log data as a string
        """
        pass


###############################################################################
## Solver listenet that just log events.
###############################################################################

class LogSolverListener(CpoSolverListener):
    """ Solve listener that just log listener events.

    *New in version 2.8.*
    """

    def __init__(self, prefix=""):
        super(LogSolverListener, self).__init__()
        self.prefix = prefix


    def solver_created(self, solver):
        """ Notify the listener that the solver object has been created.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        print(str(self.prefix) + "Solver created")


    def start_solve(self, solver):
        """ Notify that the solve is started.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        print(str(self.prefix) + "Solve started")


    def end_solve(self, solver):
        """ Notify that the solve is ended.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
        """
        print(str(self.prefix) + "Solve ended")


    def result_found(self, solver, sres):
        """ Signal that a result has been found.

        This method is called every time a result is provided by the solver. The result, in particular the last one,
        may not contain any solution. This should be checked calling method sres.is_solution().

        This method replaces deprecated method solution_found() that is confusing as result may possibly
        not contain a solution to the model.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
            sres:   Solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`
        """
        print(str(self.prefix) + "Result found")


###############################################################################
## Implementation of a solve listener that stops search when no new solution
## is given during a given amount of time
###############################################################################

from threading import Condition

class AutoStopListener(CpoSolverListener):
    """ Solver listener that aborts a search when a predefined criteria is reached.

    *New in version 2.8.*
    """
    def __init__(self, qsc_time=None, qsc_sols=None, min_sols=0, max_sols=None):
        """ Create a new solver listener that aborts the solve if defined criteria are reached.

        Args:
            min_sols: Minimun number of solutions to be found before stopping the solve
            max_sols: Maximum number of solutions after which solve is stopped
            qsc_time: Quiesce time limit. Max time, in seconds, after which solver is stopped if no new solution is found.
            qsc_sols: Quiesce time limit expressed as a number of solutions. Quiesce time limit is computed as this
                      value multiplied by the average time between solutions.
        """
        # Initialize next abort time
        self.min_sols = min_sols
        self.max_sols = max_sols
        self.qsc_time = qsc_time
        self.qsc_sols = qsc_sols
        self.nb_sols = 0                                     # Current number of solutions found
        self.start_time = time.time()
        self.last_sol_time = None                            # Time of the last solution
        # Set time checking active indicator
        self.time_active = self.qsc_time is not None or self.qsc_sols is not None
        self.condition = Condition()

    def start_solve(self, solver):
        # Store solver
        self.solver = solver
        # Check if a time limit is defined
        if self.time_active:
            self.abort_time = self._compute_next_abort_time()
            # Start time waiting thread
            self.time_active = True
            thread = Thread(target=self._waiting_loop)
            thread.start()

    def end_solve(self, solver):
        # Stop time control thread if any
        self._stop_waiting_loop()

    def result_found(self, solver, msol):
        self.nb_sols += 1
        self.last_sol_time = time.time()
        # Check if minimal number of solutions not reached
        if self.nb_sols < self.min_sols:
            return
        # Check if max number of solutions has been reached
        if self.max_sols and self.nb_sols >= self.max_sols:
            # Abort search
            self._stop_waiting_loop()
            self.solver.abort_search()
            return
        # Update next abort time
        if self.time_active:
            self.abort_time = self._compute_next_abort_time()
            with self.condition:
                self.condition.notify()

    def _compute_next_abort_time(self):
        if self.nb_sols < self.min_sols:
            return None
        if self.qsc_sols is not None:
            if self.nb_sols > 0:
                delay = self.qsc_sols * ((time.time() - self.start_time) / self.nb_sols)
                if self.qsc_time is not None:
                    delay = min(delay, self.qsc_time)
                return time.time() + delay
            else:
                if self.qsc_time is not None:
                    return time.time() + self.qsc_time
                return None
        if self.qsc_time is not None:
            return time.time() + self.qsc_time
        return None

    def _waiting_loop(self):
        """ Timer thread body """
        abort_search = False
        self.condition.acquire()
        while self.time_active:
            if self.abort_time is None:
                self.condition.wait()
            else:
                ctime = time.time()
                if self.abort_time <= ctime:
                    self.time_active = False
                    abort_search = True
                else:
                    self.condition.wait(self.abort_time - ctime)
        self.condition.release()
        if abort_search:
            self.solver.abort_search()

    def _stop_waiting_loop(self):
        # Stop time control thread if any
        self.condition.acquire()
        self.time_active = False
        self.condition.notify()
        self.condition.release()


###############################################################################
## Implementation of a solve listener that slow down solutions flow by waiting
## a given delay.
###############################################################################


class DelayListener(CpoSolverListener):
    """ Solver listener that waits a given delay after each solution.

    *New in version 2.8.*
    """
    def __init__(self, delay):
        """ Create a new solver listener that waits a given delay after each solution.

        Args:
            delay: Wait delay in seconds
        """
        self.delay = delay

    def result_found(self, solver, msol):
        time.sleep(self.delay)


###############################################################################
## Implementation of a solve progress panel based on Tkinter
###############################################################################

from threading import Thread
import time
from docplex.cp.utils import CpoNotSupportedException

_UNIT_MULTIPLIER = {'k': 1000, 'K': 1000, 'M': 1000000, 'm': 1000000, 'G': 1000000000, 'g': 1000000000,
                    'b': 1, 'B': 1}

TK_INTER_AVAILABLE = True
try:
    # Python 2 version
    import Tkinter as TK
    import tkMessageBox as messagebox
except:
    try:
        # Python 3 version
        import tkinter as TK
        from tkinter import messagebox as messagebox
    except:
        TK_INTER_AVAILABLE = False

if not TK_INTER_AVAILABLE:

    class SolverProgressPanelListener(CpoSolverListener):
        def __init__(self, parse_log=False):
            raise CpoNotSupportedException("Tkinter is not available in this Python environment")

else:

    class _SolverInfoPanel:
        """ Solver info panel that display objective value(s) and KPIs.

        *New in version 2.8.*
        """
        def __init__(self, master, model, stopfun):
            """
            Create a new solver info panel.

            Args:
                master:  Master TK component
                model:   Model to be solved
                stopfun: Function to be called if stop is requested by user
            """
            # Store attributes
            self.master = master
            self.model = model
            self.stopfun = stopfun
            self.start_time = time.time()

            # Set window title
            master.title("CPO solver infos")

            # Add name of the model
            self.model_label = TK.Label(master, text="Solving '" + self.model.get_name() + "'", font="Helvetica 10 bold")
            self.model_label.grid(row=0, column=0, columnspan=2)
            row = 1

            # Add solve time
            TK.Label(master, text="Run time:").grid(row=row, column=0, sticky=TK.E)
            self.time_text = TK.StringVar()
            self.time_text.set("00:00:00")
            TK.Label(master, textvariable=self.time_text, bg='white', width=10, anchor=TK.E)\
                .grid(row=row, column=1, sticky=TK.E)
            row += 1
            master.after(1000, self._update_time)

            # Add objective value if any
            if model.get_optimization_expression() is None:
                self.objective_value_text = None
            else:
                TK.Label(master, text="Objective value:").grid(row=row, column=0, sticky=TK.E)
                self.objective_value_text = TK.StringVar()
                self.objective_value_text.set("unknown")
                TK.Label(master, textvariable=self.objective_value_text, bg='white', width=10, anchor=TK.E)\
                    .grid(row=row, column=1, sticky=TK.E)
                row += 1

            # Add KPIs if any
            kpis = model.get_kpis()
            self.kpi_value_texts = []
            for k in sorted(kpis.keys()):
                TK.Label(master, text=k + ":").grid(row=row, column=0, sticky=TK.E)
                txt = TK.StringVar()
                txt.set("unknown")
                self.kpi_value_texts.append(txt)
                TK.Label(master, textvariable=txt, bg='white', width=10, anchor=TK.E)\
                    .grid(row=row, column=1, sticky=TK.E)
                row += 1

            # Add stop button
            self.stop_button = TK.Button(master, text="Stop solve", command=self.stopfun)
            self.stop_button.grid(row=row, column=0, columnspan=2)

            # Initialize processing of possible additional infos
            self.infos_texts = {}  # Dictionary of info texts per name
            self.last_row = row

        def notify_solution(self, msol):
            """ Notify progress panel about a new solution
            Args:
                msol: New model solution
            """
            if msol:
                # Update objective value
                if self.objective_value_text:
                    self.objective_value_text.set(', '.join(str(v) for v in msol.get_objective_values()))
                # Update KPIs
                kpis = msol.get_kpis()
                for i, k in enumerate(sorted(kpis.keys())):
                    self.kpi_value_texts[i].set(str(kpis[k]))


        def notify_infos(self, infos):
            """ Notify progress panel about last infos values
            Args:
                infos: Information dictionary
            """
            #print("Current infos: {}".format(infos))
            for k, v in infos.items():
                if k in self.infos_texts:
                    txt = self.infos_texts[k]
                else:
                    # Create a new label for info
                    TK.Label(self.master, text=k + ":").grid(row=self.last_row, column=0, sticky=TK.E)
                    txt = TK.StringVar()
                    self.infos_texts[k] = txt
                    TK.Label(self.master, textvariable=txt, bg='white', width=10, anchor=TK.E)\
                        .grid(row=self.last_row, column=1, sticky=TK.E)
                    self.last_row += 1
                    self.stop_button.grid(row=self.last_row, column=0, columnspan=2)
                txt.set(str(v))


        def _update_time(self):
            """ Update value of time field """
            dur = int(time.time() - self.start_time)
            hours = dur // 3600
            mins  = (dur //60) % 60
            secs  =  dur % 60
            self.time_text.set("{:02d}:{:02d}:{:02d}".format(hours, mins, secs))
            self.master.after(1000, self._update_time)


    class _CpoSolverProgressPanel(Thread):
        """ Main class of the solve progress panel.

        It is implemented in a separate thread to be able to run in parallel from the model solving.
        """
        def __init__(self, solver):
            """ Create a new solver progress panel
            Args:
                solver:  Solver to track
                stopfun: Function to be called if stop is requested by user
            """
            super(_CpoSolverProgressPanel, self).__init__()
            self.solver = solver
            self.model = solver.get_model()
            self.active = True
            self.display_panel = None
            self.initdone = False
            self.last_solution = None
            self.last_infos = {}
            self.info_updates = {}

        def run(self):
            """ Thread main loop
            """
            # Create GUI components
            self.root = TK.Tk()
            self.display_panel = _SolverInfoPanel(self.root, self.model, self._stop_requested_by_user)
            self.root.after_idle(self._poll_updates)
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.initdone = True

            # Start mainloop
            self.root.mainloop()

        def notify_solution(self, msol):
            """ Notify progress panel about a new solution
            Args:
                msol: New model solution
            """
            self.last_solution = msol

        def notify_infos(self, infos):
            """ Notify miscellaneous information given in a dictionary
            Args:
                infos: Information dictionary
            """
            for k, v in infos.items():
                ov = self.last_infos.get(k)
                if ov != v:
                    self.last_infos[k] = v
                    self.info_updates[k] = v

        def _stop_requested_by_user(self):
            """ Notify a stop requested by end user through GUI """
            self.active = False
            self.solver.abort_search()

        def start(self):
            """ Start the thread and wait for init completed """
            super(_CpoSolverProgressPanel, self).start()
            while not self.initdone:
                time.sleep(0.01)

        def stop(self):
            """ Request progress panel to stop and wait for thread termination """
            self.active = False
            self.join()

        def on_closing(self):
            """ Procedure called when closing the window """
            if messagebox.askokcancel("Quit", "Do you really want to terminate solving of model '{}' ?".format(self.model.get_name())):
                self._stop_requested_by_user()

        def _poll_updates(self):
            """ Polling of external changes by the TK mainloop """
            # Check stop requested
            if not self.active:
                self.root.quit()
                return
            # Check new solution
            sol = self.last_solution
            if sol is not None:
                self.last_solution = None
                self.display_panel.notify_solution(sol)
            ifoupd = self.info_updates
            if ifoupd:
                self.info_updates = {}
                self.display_panel.notify_infos(ifoupd)
            # Rearm callback
            self.root.after(1000, self._poll_updates)


    class SolverProgressPanelListener(CpoSolverListener):
        """ Solver listener that display a solving progress panel.

        The solver progress panel displays the following information:

         * the time elapsed since the beginning of the solve,
         * the last known objective (if any)
         * the last known values of the KPIs (if any)
         * a *Stop solve* button allowing to stop the solve and keep the last known solution as model solution.
         * if parse_log indicator is set, information taken from the log: memory usage, bounds, etc

        *New in version 2.8.*

        Args:
            parse_log(optional): Enable log parsing to retrieve additional information such as memory, bounds, etc
        """
        def __init__(self, parse_log=False):
            super(SolverProgressPanelListener, self).__init__()
            self.parse_log = parse_log

        def solver_created(self, solver):
            # Force solve with start/next
            solver.context.solver.solve_with_start_next = True


        def start_solve(self, solver):
            """ Notify that the solve is started. """
            # Create progress panel
            self.progress_panel = _CpoSolverProgressPanel(solver)
            self.progress_panel.start()

        def end_solve(self, solver):
            """ Notify that the solve is ended. """
            self.progress_panel.stop()

        def result_found(self, solver, msol):
            """ Signal that a solution has been found. """
            self.progress_panel.notify_solution(msol)

        def new_log_data(self, solver, data):
            """ Signal a new piece of log data. """
            if not self.parse_log:
                return
            # Search for solve infos
            for line in data.splitlines():
                if line.startswith(" ! Time = "):
                    infos = {}
                    ex = line.find("s,", 10)
                    if ex > 0:
                        infos['SolveTime'] = float(line[10:ex])
                    sx = line.find("Average fail depth = ", ex)
                    if sx >= 0:
                        sx += 21
                        ex = line.find(", ", sx)
                        if ex > 0:
                            infos['Average fail depth'] = int(line[sx:ex])
                    sx = line.find("Memory usage = ", ex)
                    if sx >= 0:
                        sx += 15
                        ex = line.find(" ", sx)
                        if ex > 0:
                           #infos['Memory usage'] = int(float(line[sx:ex]) * _UNIT_MULTIPLIER[line[ex+1]])
                           infos['Memory usage'] = line[sx:]
                    self.progress_panel.notify_infos(infos)
                elif line.startswith(" ! Current bound is "):
                    infos = {}
                    ex = line.find(" (", 20)
                    if ex > 0:
                        #infos['Current bound'] = [int(x) for x in line[20:ex].split(";")]
                        infos['Current bound'] = line[20:ex]
                    self.progress_panel.notify_infos(infos)
                elif line.startswith(" ! Current objective is "):
                    infos = {'Current objective': line[24:]}
                    #infos['Current objective'] = [int(x) for x in line[24:].split(";")]
                    self.progress_panel.notify_infos(infos)

