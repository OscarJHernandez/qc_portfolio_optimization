# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2019
# --------------------------------------------------------------------------
'''This package contains classes to monitor the progress of a MIP solve.

This abstract class defines protocol methods, which are called from within CPLEX's MIP search algorithm.

Progress listeners are based on CPLEX's MIPInfoCallback, and work only with a
local installation of CPLEX, not on Cplexcloud.

At each node, all progress listeners attached to the model receive progress data
through a `notify_progress` method, with an instance of :class:`~ProgressData`.
This named tuple class contains information about solve time, objective value,
best bound, gap and nodes, but no variable values.

The base class for progress listeners is :class:`~ProgressListener`.
DOcplex provides concrete sub-classes to handle basic cases, but
you may also create your own sub-class of :class:`~ProgressListener` to suit your needs.
Each instance of :class:`~ProgressListener` must be attached to a model to receive events during
a MIP solve., using `Model.add_progress_listener`.

An example of such a progress listener is :class:`TextProgressListener`,
which prints a message to stdout in a format similar to CPLEX log, each time it receives a progress data.

If you need to get the values of variables in an intermediate solution,
you should sub-class from class :class:`~SolutionListener`.
In this case, the :func:`~ProgressListener.notify_solution` method will be called
each time CPLEX finds a valid intermediate solution, with an instance of :class:`docplex.mp.solution.SolveSolution`.

Listeners are called from within CPLEX code, with its own frequency. It may occur that listeners are
called with no real progress in either the objective or the best bound. To help you chose what kind
of calls you are really interested in, DOcplex uses the enumerated class :class:`~ProgressClock'.
The baseline clock is the CPLEX clock, but you can refine this clock: for example,
`ProgressClock.Objective` filters all calls but those where the objective has improved.

To summarize, listeners are called from within CPLEX's MIP search algorithm, when the clock
you have chosen accepts the call. All listeners are created with a clock argument that controls which
calls they accept.

When a listener accepts a call, its method :func:`~ProgressListener.notify_progress` is called with the current progress data.
See :class:`~TextProgressListener` as an example of a simple progress listener.

In addition, if the listener derives from :class:`~SolutionListener`, its method :func:`~ProgressListener.notify_solution` is also called
with the intermediate solution, passed as an instance of :class:`docplex.mp.solution.SolveSolution`.
See :class:`~SolutionRecorder` as an example of solution listener

'''
from six import iteritems
from enum import Enum
from collections import namedtuple


from docplex.mp.utils import is_string, is_number
from docplex.mp.error_handler import docplex_fatal

_TProgressData_ = namedtuple('_TProgressData',
                             ['id', 'has_incumbent',
                              'current_objective', 'best_bound', 'mip_gap',
                              'current_nb_iterations', 'current_nb_nodes', 'remaining_nb_nodes',
                              'time', 'det_time'])


# noinspection PyUnresolvedReferences
class ProgressData(_TProgressData_):
    """ A named tuple class to hold progress data, as reeived from CPLEX.

    Attributes:
        has_incumbent: a boolean, indicating whether an incumbent solution is available (or not),
            at the moment the listener is called.
        current_objective: contains the current objective, if an incumbent is available, else None.
        best_bound: The current best bound as reported by the solver.
        mip_gap: the gap between the best integer objective and the objective of the best node remaining;
            available if an incumbent is available.
        current_nb_nodes: the current number of nodes.
        current_nb_iterations: the current number of iterations.
        remaining_nb_nodes: the remaining number of nodes.
        time: the elapsed time since solve started.
        det_time: the deterministic time since solve started.
    """

    def __str__(self):  # pragma: no cover
        fmt = 'ProgressData({0}, {1}, obj={2}, bbound={3}, #nodes={4})'. \
            format(self.id, self.has_incumbent, self.current_objective, self.best_bound, self.current_nb_nodes)
        return fmt


class ProgressClock(Enum):
    """
    This enumerated class controls the type of events a listener listens to.

    The possible values are (in order of decreasing call frequency).

        - All: the listener listens to all calls from CPLEX.
        - BestBound: listen to changes in best bound, not necessarily with a solution present.
        - Solutions: listen to all intermediate solutions, not necessarily impriving.
            Nothing prevents being called several times with an identical solution.

        - Gap: listen to intermediate solutions, where either objective or best bound has improved.
        - Objective: listen to intermediate solutions, where the solution objective has improved.

    To determine whether objective or best bound have improved (or not), see the numerical parameters
    `absdiff` and `reldiff` in `:class:ProgressListener` and all its descendant classes.

    *New in version 2.10*
    """
    #
    # An enumeration of filtering levels for listeners.
    #
    All = 0
    Solutions = 1
    BestBound = 2  # best bound clock is indepedent from solution clock
    Objective = 5  # b101
    Gap = 7  # the gap clock changes when there is a solution, and either bound or objective has changed

    @property
    def listens_to_solution(self):
        # returns true if the enum listens to solutions
        return 1 == self.value & 1

    @classmethod
    def parse(cls, arg):
        if isinstance(arg, ProgressClock):
            return arg
        else:
            # int value
            for fl in cls:
                if arg == fl.value:
                    return fl
                elif is_string(arg) and arg.lower() == fl.name.lower():
                    return fl
            else:
                # pragma: no cover
                raise ValueError('Expecting filter level, {0!r} was passed'.format(arg))


default_absdiff = 1e-1
default_reldiff = 1e-2


class ProgressListener(object):
    '''  The base class for progress listeners.
    '''

    def __init__(self, clock_arg=ProgressClock.All, absdiff=None, reldiff=None):
        self._cb = None
        clock = ProgressClock.parse(clock_arg)
        self._clock = clock
        self._current_progress_data = None
        self._absdiff = absdiff if absdiff is not None else default_absdiff
        self._reldiff = reldiff if reldiff is not None else default_reldiff
        self._filter = make_clock_filter(clock, absdiff, reldiff)

    def _get_model(self):
        ccb = self._cb
        return ccb._model if ccb else None

    @property
    def clock(self):
        """ Returns the clock of the listener.

        :return:
            an instance of :class:`~ProgressClock`

        """
        return self._clock

    @property
    def abs_diff(self):
        return self._absdiff

    @property
    def relative_diff(self):
        return self._reldiff

    @property
    def current_progress_data(self):
        """ This property return the current progress data, if any.

        Returns the latest progress data, if at least one has been received.

        :return: an instance of :class:~ProgressData` or None.
        """
        return self._current_progress_data

    def _set_current_progress_data(self, pdata):
        # INTERNAL
        self._current_progress_data = pdata

    def accept(self, pdata):
        return self._filter.accept(pdata)

    def _disconnect(self):
        # INTERNAL
        self._cb = None

    def _connect_cb(self, cb):
        # INTERNAL
        self._cb = cb

    def requires_solution(self):
        return False

    def notify_solution(self, s):
        """ Redefine this method to handle an intermediate solution from the callback.

        Args:
            s: solution

        Note:
            if you need to access runtime information (time, gap, number of nodes), use
            :function:`SolutionListener.current_progress_data`, which contains the latest
            priogress information as a tuple.
        """
        pass  # pragma: no cover

    def notify_start(self):
        """ This methods  is called on all attached listeners at the beginning of a solve().

        When writing a custom listener, this method is used to restore
        internal data which can be modified during solve.
        Always call the superclass `notify_start` before adding specific code in a sub-class.
        """
        self._current_progress_data = None
        self._filter.reset()

    def notify_jobid(self, jobid):
        pass  # pragma: no cover

    def notify_end(self, status, objective):
        """The method called when solve is finished on a model.

        The status is the solve status from the
        solve() method
        """
        pass  # pragma: no cover

    def notify_progress(self, progress_data):
        """ This method is called from within the solve with a ProgressData instance.

        :param progress_data: an instance of :class:`ProgressData` containing data about the
            current point in the search tree.

        """
        pass  # pragma: no cover

    def abort(self):
        ''' Aborts the CPLEX search.

        This method tells CPLEX to stop the MIP search.
        You may use this method in a custom progress listener to stop the search based on your own
        criteria (for example, when improvements in gap is consistently below a minimum threshold).

        Note:
            This method should be called from within a :func:`notify_progress` method call, otherwise it will have
            no effect.

        '''
        ccb = self._cb
        if ccb is not None:
            ccb.abort()
        else:
            print('!!! callback is not connected - abort() ignored')


# noinspection PyUnusedLocal,PyMethodMayBeStatic
class FilterAcceptAll(object):

    def accept(self, pdata):
        return True

    def reset(self):
        pass


# noinspection PyMethodMayBeStatic
class FilterAcceptAllSolutions(object):

    def accept(self, pdata):
        return pdata.has_incumbent

    def reset(self):
        pass


class Watcher(object):

    def __init__(self, name, absdiff, reldiff, update_fn):
        assert absdiff >= 0
        assert reldiff >= 0

        self.name = name
        self._watched = None
        self._old = None
        self._absdiff = absdiff
        self._reldiff = reldiff
        self._update_fn = update_fn

    def reset(self):
        self._watched = None
        self._old = None

    def accept(self, progress_data):
        accepted = False

        old = self._watched
        new_watched = self._update_fn(progress_data)
        # if new_watched is None, then it is not available (e.g. objective)
        if new_watched is not None:
            if old is None:
                accepted = True
            else:
                # assert is a number
                delta = abs(new_watched - old)
                reldiff = self._reldiff
                absdiff = self._absdiff
                if 0 < absdiff <= abs(new_watched - old):
                    accepted = True
                elif reldiff and delta / (1 + abs(old)) >= reldiff:
                    accepted = True
        return accepted

    def sync(self, pdata):
        cur = self._watched
        self._watched = self._update_fn(pdata)
        self._old = cur

    def __str__(self):  # pragma: no cover
        ws = '--' if self._watched is None else self._watched
        return 'W_{0}[{1}]'.format(self.name, ws)


def clock_filter_accept_stop_here(watcher, pdata):
    pass


class ClockFilter(object):

    def __init__(self, level, obj_absdiff, bbound_absdiff, obj_reldiff=0, bbound_reldiff=0, node_delta=-1):
        watchers = []
        if obj_absdiff > 0 or obj_reldiff > 0:
            def update_obj(pdata):
                return pdata.current_objective if pdata.has_incumbent else None

            obj_watcher = Watcher(name='obj', absdiff=obj_absdiff, reldiff=obj_reldiff, update_fn=update_obj)
            watchers.append(obj_watcher)
        if bbound_absdiff > 0 or bbound_reldiff > 0:
            def update_bbound(pdata):
                return pdata.best_bound

            watchers.append(Watcher(name='gap_bound', absdiff=bbound_absdiff, reldiff=bbound_reldiff,
                                    update_fn=update_bbound))
        if node_delta > 0:
            used_node_delta = max(node_delta, 1)

            def update_nodes(pdata):
                return pdata.current_nb_nodes

            watchers.append(Watcher(name='nodes', absdiff=used_node_delta, reldiff=0, update_fn=update_nodes))
        self._watchers = watchers
        self._clock = level
        self._listens_to_solution = level.listens_to_solution

    def accept(self, progress_data):
        if not progress_data.has_incumbent and self._listens_to_solution:
            return False
        # the filter accepts data as soon as one of its cells accepts it.
        poke = self.peek(progress_data)
        if poke is None:
            return False
        else:
            clock_filter_accept_stop_here(poke, progress_data)
            # print("- accepting event #{0}, reason: {1}".format(progress_data.id, poke.name))
            for w in self._watchers:
                w.sync(progress_data)
            return True

    def peek(self, progress_data):
        for w in self._watchers:
            if w.accept(progress_data):
                return w
        else:
            return None

    def reset(self):
        for w in self._watchers:
            w.reset()


# noinspection PyArgumentEqualDefault
def make_clock_filter(level, absdiff, reldiff, nodediff=0):
    absdiff_ = 1e-1 if absdiff is None else absdiff
    reldiff_ = 1e-2 if reldiff is None else reldiff
    if level == ProgressClock.All:
        return FilterAcceptAll()
    elif level == ProgressClock.Solutions:
        return FilterAcceptAllSolutions()
    elif level == ProgressClock.Gap:
        return ClockFilter(level, obj_absdiff=absdiff_, obj_reldiff=reldiff_,
                           bbound_absdiff=absdiff_, bbound_reldiff=reldiff_, node_delta=nodediff)
    elif level == ProgressClock.Objective:
        return ClockFilter(level, obj_absdiff=absdiff_, obj_reldiff=reldiff_,
                           bbound_absdiff=0, bbound_reldiff=0, node_delta=nodediff)
    elif level == ProgressClock.BestBound:
        return ClockFilter(level, obj_absdiff=0, obj_reldiff=0,
                           bbound_absdiff=absdiff_, bbound_reldiff=reldiff_)

    else:
        # pragma: no cover
        raise ValueError('unexpected level: {0!r}'.format(level))


class TextProgressListener(ProgressListener):
    """ A simple implementation of Progress Listener, which prints messages to stdout,
        in the manner of the CPLEX log.

    :param clock: an enumerated value of type `:class:ProgressClock` which defines the frequency of the listener.
    :param absdiff: a float value which controls the minimum absolute change for objective or best bound values
    :param reldiff: a float value which controls the minimum absolute change for objective or best bound values.

    Note:
        The default behavior is to listen to an improvement in either the objective or best bound,
        each being improved by either an absolute value change of at least `absdiff`,
        or a relative change of at least `reldiff`.

    """

    def __init__(self, clock=ProgressClock.Gap, gap_fmt=None, obj_fmt=None,
                 absdiff=None, reldiff=None):
        ProgressListener.__init__(self, clock, absdiff, reldiff)
        self._gap_fmt = gap_fmt or "{:.2%}"
        self._obj_fmt = obj_fmt or "{:.4f}"
        self._count = 0

    def notify_start(self):
        super(TextProgressListener, self).notify_start()
        self._count = 0

    def notify_progress(self, progress_data):
        self._count += 1
        pdata_has_incumbent = progress_data.has_incumbent
        incumbent_symbol = '+' if pdata_has_incumbent else ' '
        # if pdata_has_incumbent:
        #     self._incumbent_count += 1
        current_obj = progress_data.current_objective
        if pdata_has_incumbent:
            objs = self._obj_fmt.format(current_obj)
        else:
            objs = "N/A"  # pragma: no cover
        best_bound = progress_data.best_bound
        nb_nodes = progress_data.current_nb_nodes
        remaining_nodes = progress_data.remaining_nb_nodes
        if pdata_has_incumbent:
            gap = self._gap_fmt.format(progress_data.mip_gap)
        else:
            gap = "N/A"  # pragma: no cover
        raw_time = progress_data.time
        rounded_time = round(raw_time, 1)

        print("{0:>3}{7}: Node={4} Left={5} Best Integer={1}, Best Bound={2:.4f}, gap={3}, ItCnt={8} [{6}s]"
              .format(self._count, objs, best_bound, gap, nb_nodes, remaining_nodes, rounded_time,
                      incumbent_symbol, progress_data.current_nb_iterations))


class ProgressDataRecorder(ProgressListener):
    """ A specialized class of ProgressListener, which collects all ProgressData it receives.

    """

    def __init__(self, clock=ProgressClock.Gap, absdiff=None, reldiff=None):
        super(ProgressDataRecorder, self).__init__(clock, absdiff, reldiff)
        self._recorded = []

    def notify_start(self):
        super(ProgressDataRecorder, self).notify_start()
        # clear recorded data
        self._recorded = []

    def notify_progress(self, progress_data):
        self._recorded.append(progress_data)

    @property
    def number_of_records(self):
        return len(self._recorded)

    @property
    def iter_recorded(self):
        """ Returns an iterator on stored progress data

        :return: an iterator.
        """
        return iter(self._recorded)

    @property
    def recorded(self):
        """ Returns a copy of the recorded data.

        :return:
        """
        return self._recorded[:]


class SolutionListener(ProgressListener):
    """ The base class for listeners that work on intermediate solutions.

    To define a custom behavior for a subclass of this class, you need to redefine `notify_solution`.
    The current progress data is available from the `current_progress_data` property.
    """

    # noinspection PyMethodMayBeStatic
    def check_solution_clock(self):
        if not self.clock.listens_to_solution:
            docplex_fatal('Solution listener requires a solution clock among (Solutions,Objective|Gap), {0} was passed',
                          self.clock)

    def __init__(self, clock=ProgressClock.Solutions, absdiff=None, reldiff=None):
        super(SolutionListener, self).__init__(clock, absdiff, reldiff)
        self.check_solution_clock()

    def requires_solution(self):
        return True

    def notify_solution(self, sol):
        """ Generic method to be redefined by custom listeners to
        handle intermediate solutions. This method is called by the
        CPLEX search with an intermediate solution.

        :param sol: an instance of :class:`docplex.mp.solution.SolveSolution`

        Note: as this method is called at each node of the MIP search, it may happen
        that several calls are made with an identical solution, that is, different object
        instances, but sharing the same variable values.
        """
        pass  # pragma: no cover

    def notify_start(self):
        super(SolutionListener, self).notify_start()

    def accept(self, pdata):
        return pdata.has_incumbent and super(SolutionListener, self).accept(pdata)


class SolutionRecorder(SolutionListener):
    """ A specialized implementation of :class:`SolutionListener`,
    which stores --all-- intermediate solutions.

    As  the listener might be called at different times with identical incumbent
    values, thus the list of solutions list might well contain identical solutions.
    """

    def __init__(self, clock=ProgressClock.Gap, absdiff=None, reldiff=None):
        super(SolutionRecorder, self).__init__(clock, absdiff, reldiff)
        self._solutions = []

    def notify_start(self):
        """ Redefinition of the generic notify_start() method to clear all data modified by solve().
        In this case, clears the list of solutions.
        """
        super(SolutionListener, self).notify_start()
        self._solutions = []

    def notify_solution(self, sol):
        """ Redefintion of the generic `notify_solution` method, called by CPLEX.
         For this class, appends the intermediate solution to the list of stored solutions.

        """
        self._solutions.append(sol)

    def iter_solutions(self):
        """ Returns an iterator on the stored solutions"""
        return iter(self._solutions)

    @property
    def number_of_solutions(self):
        """ Returns the number of stored solutions. """
        return len(self._solutions)

    @property
    def current_solution(self):
        # redefinition of generic method `notify_solution`, called by CPLEX.
        sols = self._solutions
        return sols[-1] if sols else None


class FunctionalSolutionListener(SolutionListener):
    """ A subclass of SolutionListener, which calls a function at each intermediate solution.

        No exception is caught.
    """

    def __init__(self, solution_fn, clock=ProgressClock.Gap, absdiff=None, reldiff=None):
        SolutionListener.__init__(self, clock, absdiff, reldiff)
        self._sol_fn = solution_fn

    def notify_solution(self, sol):
        self._sol_fn(sol)


class KpiListener(SolutionListener):
    """ A refinement of SolutionListener, which computes KPIs at each intermediate solution.

    Calls the `publish` method with a dicitonary of KPIs. Defaul tis to do nothing.

    This listener listens to the `Gap` clock.

    """

    objective_kpi_name = '_current_objective'
    time_kpi_name = '_current_time'

    def __init__(self, model, clock=ProgressClock.Gap, absdiff=None, reldiff=None):
        super(KpiListener, self).__init__(clock, absdiff, reldiff)
        self.model = model

    def publish(self, kpi_dict):
        """ This method is called at each improving solution, with a dictionay of name, values.

        :param kpi_dict: a dicitonary of names and KPi values, computed on the intermediate solution.

        """
        pass

    def notify_solution(self, sol):
        pdata = self.current_progress_data

        # 1. build a dict from formatted names to kpi values.
        kpis_as_dict = {kp.name: kp.compute(sol) for kp in self.model.iter_kpis()}
        # 2. add predefined keys for obj, time.
        kpis_as_dict[self.objective_kpi_name] = sol.objective_value
        kpis_as_dict[self.time_kpi_name] = pdata.time
        self.publish(kpis_as_dict)


class KpiPrinter(KpiListener):

    def __init__(self, model, clock=ProgressClock.Gap, absdiff=None, reldiff=None,
                 kpi_format='* ItCnt={3:d}  KPI: {1:<{0}} = '):
        super(KpiPrinter, self).__init__(model, clock, absdiff, reldiff)
        self.kpi_format = kpi_format

    def publish(self, kpi_dict):
        try:
            max_kpi_name_len = max(len(kn) for kn in kpi_dict)  # max() raises ValueError on empty
        except ValueError:
            max_kpi_name_len = 0
        kpi_num_format = self.kpi_format + '{2:.3f}'
        kpi_str_format = self.kpi_format + '{2!s}'
        print('-' * (max_kpi_name_len + 15))
        itcnt = self.current_progress_data.current_nb_iterations
        for kn, kv in iteritems(kpi_dict):
            if is_number(kv):
                k_format = kpi_num_format
            else:
                k_format = kpi_str_format
            kps = k_format.format(max_kpi_name_len, kn, kv, itcnt)
            print(kps)

