# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains the different elements representing the result to a solver request.
The main classes are:

 * :class:`CpoRunResult`: root class for all results returned by the solver.
   It contains general purpose information such as the model that has been solved, solver parameters, solver info and log.
 * :class:`CpoSolveResult`: result of a solve request, including a solution to the model if any.
 * :class:`CpoRefineConflictResult`: result of a refine conflict request.

These classes are using the following classes to store utility information:

 * :class:`CpoSolverInfos`: miscellaneous information coming from the solver.
 * :class:`CpoProcessInfos`: miscellaneous information about the processing and solving of the model by the Python API.

When a solution is found by the solver, it is represented by the following classes:

 * :class:`CpoModelSolution`: aggregation of all individual model element solutions,
 * :class:`CpoIntVarSolution`: solution of an integer variable,
 * :class:`CpoIntervalVarSolution`: solution of an interval variable,
 * :class:`CpoSequenceVarSolution`: solution of a sequence variable,
 * :class:`CpoStateFunctionSolution`: solution of a state function, and

These solution objects can be used in multiple ways:

 * To represent a *complete* (fully instantiated) solution, where each model has a unique fixed value, as returned
   by a successful model solve.
 * To represent a *partial* model solution, that is proposed as a solve starting point
   (see :meth:`docplex.cp.model.CpoModel.set_starting_point`)
   In this case, not all variables are present in the solution, and some of them may be partially instantiated.
 * To represent a *partial* model solution that is returned by the solver as result of calling method
   :meth:`docplex.cp.solver.solver.CpoSolver.propagate`.


Detailed description
--------------------
"""

import docplex.cp.utils as utils
from docplex.cp.utils import *
from docplex.cp.expression import CpoVariable, CpoIntVar, CpoFloatVar, CpoIntervalVar, CpoSequenceVar, CpoStateFunction, \
    INT_MIN, INT_MAX, INTERVAL_MIN, INTERVAL_MAX, POSITIVE_INFINITY, NEGATIVE_INFINITY, \
    _domain_iterator, _domain_min, _domain_max, _domain_contains, \
    compare_expressions
from docplex.cp.parameters import CpoParameters
import types
from collections import OrderedDict
import functools


###############################################################################
##  Constants
###############################################################################

# Solve status: Unknown
SOLVE_STATUS_UNKNOWN = "Unknown"

# Solve status: Infeasible
SOLVE_STATUS_INFEASIBLE = "Infeasible"

# Solve status: Feasible
SOLVE_STATUS_FEASIBLE = "Feasible"

# Solve status: Optimal
SOLVE_STATUS_OPTIMAL = "Optimal"

# Solve status: Job aborted
SOLVE_STATUS_JOB_ABORTED = "JobAborted"

# Solve status: Job failed
SOLVE_STATUS_JOB_FAILED = "JobFailed"

# List of all possible solve statuses
ALL_SOLVE_STATUSES = (SOLVE_STATUS_UNKNOWN,
                      SOLVE_STATUS_INFEASIBLE, SOLVE_STATUS_FEASIBLE, SOLVE_STATUS_OPTIMAL,
                      SOLVE_STATUS_JOB_ABORTED, SOLVE_STATUS_JOB_FAILED)


# Fail status: Unknown
FAIL_STATUS_UNKNOWN = "Unknown"

# Fail status: Failed normally
FAIL_STATUS_FAILED_NORMALLY = "SearchHasFailedNormally"

# Fail status: Not failed (success)
FAIL_STATUS_HAS_NOT_FAILED = "SearchHasNotFailed"

# Fail status: Stopped by abort
FAIL_STATUS_ABORT = "SearchStoppedByAbort"

# Fail status: Stopped by exception
FAIL_STATUS_EXCEPTION = "SearchStoppedByException"

# Fail status: Stopped by exit
FAIL_STATUS_EXIT = "SearchStoppedByExit"

# Fail status: Stopped by label
FAIL_STATUS_LABEL = "SearchStoppedByLabel"

# Fail status: Stopped by time limit
FAIL_STATUS_TIME_LIMIT = "SearchStoppedByLimit"

# Fail status: Search completed
FAIL_STATUS_SEARCH_COMPLETED = "SearchCompleted"

# List of all possible search statuses
ALL_FAIL_STATUSES = (FAIL_STATUS_UNKNOWN,
                     FAIL_STATUS_FAILED_NORMALLY, FAIL_STATUS_HAS_NOT_FAILED,
                     FAIL_STATUS_ABORT, FAIL_STATUS_EXCEPTION, FAIL_STATUS_EXIT, FAIL_STATUS_LABEL,
                     FAIL_STATUS_TIME_LIMIT, FAIL_STATUS_SEARCH_COMPLETED)


# Search status: Not started
SEARCH_STATUS_NOT_STARTED = "SearchNotStarted"

# Search status: Ongoing
SEARCH_STATUS_ONGOING = "SearchOngoing"

# Search status: Completed
SEARCH_STATUS_COMPLETED = "SearchCompleted"

# Search status: Stopped. Cause given in SearchStopCause.
SEARCH_STATUS_STOPPED = "SearchStopped"

# List of all possible search statuses
ALL_SEARCH_STATUSES = (SEARCH_STATUS_NOT_STARTED, SEARCH_STATUS_ONGOING, SEARCH_STATUS_COMPLETED, SEARCH_STATUS_STOPPED)


# Stop cause: Not stopped
STOP_CAUSE_NOT_STOPPED = "SearchHasNotBeenStopped"

# Stop cause: Search terminated on limit (time limit, fail limit, etc)
STOP_CAUSE_LIMIT = "SearchStoppedByLimit"

# Stop cause: Exit called while solving, for example by a callback
STOP_CAUSE_EXIT = "SearchStoppedByExit"

# Stop cause: Search aborted externally
STOP_CAUSE_ABORT = "SearchStoppedByAbort"

# Stop cause: Unknown cause
STOP_CAUSE_UNKNOWN = "SearchStoppedByUnknownCause"

# List of all possible stop causes
ALL_STOP_CAUSES = (STOP_CAUSE_NOT_STOPPED, STOP_CAUSE_LIMIT, STOP_CAUSE_EXIT, STOP_CAUSE_ABORT, STOP_CAUSE_UNKNOWN)


###############################################################################
##  Public classes
###############################################################################

class CpoVarSolution(object):
    """ This class is a super class of all classes representing a solution to a variable.
    """
    __slots__ = ('expr',  # Variable expression
                 )
    
    def __init__(self, expr):
        """ Constructor:

        Args:
            expr: Variable expression, object of class :class:`~docplex.cp.expression.CpoVariable` or extending class.
        """
        # Checking already done by extending classes
        # assert isinstance(expr, CpoVariable), "Expression 'expr' should be a CPO variable expression"
        self.expr = expr


    def get_expr(self):
        """ Gets the expression of the variable.

        Returns:
            Model expression of the variable.
        """
        return self.expr


    def get_name(self):
        """ Gets the name of the variable.

        Returns:
            Name of the variable, None if anonymous.
        """
        return self.expr.get_name()


    def get_value(self):
        """ Gets the variable value.
        This method is overloaded by each class extending this class.

        Returns:
            Value of the variable, represented according to its semantic (see specific variable documentation).
        """
        return None


    def __eq__(self, other):
        """ Overwrite equality comparison

        Args:
            other: Other object to compare with
        Returns:
            True if this object is equal to the other, False otherwise
        """
        return utils.equals(self, other)


    def __ne__(self, other):
        """ Overwrite inequality comparison """
        return not self.__eq__(other)


    def __str__(self):
        """ String representing this object """
        return "{}={}".format(self.expr.get_name(), self.get_value())


    def __hash__(self):
        return id(self)


class CpoIntVarSolution(CpoVarSolution):
    """ This class represents a solution to an integer variable.

    The solution can be:
     * *complete* when the value is a single integer,
     * *partial* when the value is a domain, set of multiple values.

    A domain is a list of discrete integer values and/or intervals of values represented by a tuple containing
    interval min and max values (included).

    For example, following are valid domains for an integer variable:
     * 7 (complete solution)
     * (1, 2, 4, 9)
     * (2, 3, (5, 7), 9, (11, 13))
    """
    __slots__ = ('value',  # Variable value / domain
                 )

    def __init__(self, expr, value):
        """ Constructor:

        Args:
            expr:  Variable expression, object of class :class:`~docplex.cp.expression.CpoIntVar`.
            value: Variable value, or domain if not completely instantiated
        """
        assert isinstance(expr, CpoIntVar), "Expression 'expr' should be a CpoIntVar expression"
        super(CpoIntVarSolution, self).__init__(expr)
        self.value = _check_arg_domain(value, 'value')

    def get_value(self):
        """ Gets the value of the variable.

        Returns:
            Variable value (integer), or domain (list of integers or intervals)
        """
        return self.value

    def get_domain_min(self):
        """ Gets the domain lower bound.

        Returns:
            Domain lower bound.
        """
        return _domain_min(self.value)

    def get_domain_max(self):
        """ Gets the domain upper bound.

        Returns:
            Domain upper bound.
        """
        return _domain_max(self.value)

    def domain_iterator(self):
        """ Iterator on the individual values of an integer variable domain.

        Returns:
            Value iterator on the domain of this variable.
        """
        return _domain_iterator(self.value)

    def domain_contains(self, value):
        """ Check whether a given value is in the domain of the variable

        Args:
            val: Value to check
        Returns:
            True if the value is in the domain, False otherwise
        """
        return _domain_contains(self.value, value)


class CpoFloatVarSolution(CpoVarSolution):
    # """ This class represents a solution to a float variable.
    #
    # The solution can be:
    #  * *complete* when the value is a single value,
    #  * *partial* when the value is an interval.
    # """
    __slots__ = ('value',  # Variable value or tuple (interval)
                 )

    def __init__(self, expr, value):
        """ Constructor:

        Args:
            expr:  Variable expression, object of class :class:`~docplex.cp.expression.CpoFloatVar`.
            value: Variable value, or domain if not completely instantiated
        """
        assert isinstance(expr, CpoFloatVar), "Expression 'expr' should be a CpoIntVar expression"
        super(CpoFloatVarSolution, self).__init__(expr)
        self.value = value

    def get_value(self):
        """ Gets the value of the variable.

        Returns:
            Variable value (integer), or domain (list of integers or intervals)
        """
        return self.value

    def get_domain_min(self):
        """ Gets the domain lower bound.

        Returns:
            Domain lower bound.
        """
        return self.value if is_number(self.value) else self.value[0]

    def get_domain_max(self):
        """ Gets the domain upper bound.

        Returns:
            Domain upper bound.
        """
        return self.value if is_number(self.value) else self.value[-1]

    def __str__(self):
        """ Convert this expression into a string """
        return str(self.get_name()) + ": " + str(self.get_value())


class CpoIntervalVarSolution(CpoVarSolution):
    """ This class represents a solution to an interval variable.

    The solution can be complete if all attribute values are integers, or partial if at least one
    of them is an interval expressed as a tuple.
    """
    __slots__ = ('start',    # Interval start
                 'end',      # Interval end
                 'size',     # Interval size
                 'length',   # Interval length
                 'presence', # Presence indicator
                )
    
    def __init__(self, expr, presence=None, start=None, end=None, size=None, length=None):
        """ Constructor:

        Args:
            expr:     Variable expression, object of class :class:`~docplex.cp.expression.CpoIntervalVar`.
            presence: Presence indicator (True for present, False for absent, None for undetermined). Default is None.
            start:    Value of start, or tuple representing the start range. Default is None.
            end:      Value of end, or tuple representing the end range. Default is None.
            size:     Value of size, or tuple representing the size range. Default is None.
            length:   Value of the length, or tuple representing the length range. Default is None.
                      Not to be used if other values are integers.
        """
        assert isinstance(expr, CpoIntervalVar), "Expression 'expr' should be a CpoIntervalVar expression"
        super(CpoIntervalVarSolution, self).__init__(expr)
        self.presence = presence
        self.start    = start
        self.end      = end
        self.size     = size
        self.length   = length


    def is_present(self):
        """ Check if the interval is present.

        Returns:
            True if interval is present, False otherwise.
        """
        return self.presence is True


    def is_absent(self):
        """ Check if the interval is absent.

        Returns:
            True if interval is absent, False otherwise.
        """
        return self.presence is False


    def is_optional(self):
        """ Check if the interval is optional.
        Calling this function returns always False for a complete solution where
        Valid only for a partial solution, meaning that the status of the variable is not yet fixed.

        Returns:
            True if interval is optional (undetermined), False otherwise.
        """
        return self.presence is None


    def get_start(self):
        """ Gets the interval start.

        Returns:
            Interval start value, or domain (tuple (min, max)) if not fully instantiated.
            None if interval is absent.
        """
        return self.start


    def get_end(self):
        """ Gets the interval end.

        Returns:
            Interval end value, or domain (tuple (min, max)) if not fully instantiated.
            None if interval is absent.
        """
        return self.end


    def get_size(self):
        """ Gets the size of the interval.

        The size of the interval is the amount of work done in the interval,
        that depends on the intensity function that has been associated to the interval.

        Returns:
            Interval size value, or domain (tuple (min, max)) if not fully instantiated.
            None if interval is absent.
        """
        return self.size


    def get_length(self):
        """ Gets the length of the interval.

        Length of the interval is the difference between end and start.

        Returns:
            Interval length value, or domain (tuple (min, max)) if not fully instantiated.
            None if interval is absent.
        """
        if self.length is None:
            return None if self.end is None else self.end - self.start
        return self.length


    def get_value(self):
        """ Gets the interval variable value as a tuple (start, end, size), or () if absent.

        If the variable is absent, then the result is an empty tuple.

        If the variable is fully instantiated, the result is a tuple of 3 integers (start, end, size).
        The variable length, easy to compute as end - start, can also be retrieved by calling :meth:`get_length`.

        If the variable is partially instantiated, the result is a tuple (start, end, size, length) where each
        individual value can be an integer or an interval expressed as a tuple.

        Returns:
            Interval variable value as a tuple.
        """
        if self.is_present():
            if self.length is None:
                return (self.start, self.end, self.size, )
            else:
                return (self.start, self.end, self.size, self.length, )
        return ()


    def __str__(self):
        """ Convert this expression into a string """
        res = [str(self.get_name()), ': ']
        if self.is_absent():
            res.append("absent")
        else:
            if self.is_optional():
                res.append("optional")
            res.append("(start=" + str(self.get_start()))
            res.append(", end=" + str(self.get_end()))
            res.append(", size=" + str(self.get_size()))
            res.append(", length=" + str(self.get_length()))
            res.append(")")
        return ''.join(res)

     
class CpoSequenceVarSolution(CpoVarSolution):
    """ This class represents a solution to a sequence variable.
    """
    __slots__ = ('lvars',  # List of interval variable solutions
                )
    
    def __init__(self, expr, lvars):
        """ Constructor:

        Args:
            expr:  Variable expression, object of class :class:`~docplex.cp.expression.CpoSequenceVar`.
            lvars: Ordered list of interval variable solutions that are in this sequence
                   (objects of class :class:`CpoIntervalVarSolution`),
                   or list of interval variables (object of class :class:`~docplex.cp.expression.CpoIntervalVar`).
        """
        assert isinstance(expr, CpoSequenceVar), "Expression 'expr' should be a CpoSequenceVar expression"
        super(CpoSequenceVarSolution, self).__init__(expr)
        self.lvars = lvars


    def get_interval_variables(self):
        """ Gets the list of CpoIntervalVarSolution in this sequence.

        Returns:
            List of CpoIntervalVarSolution in this sequence.
        """
        return self.lvars


    def get_value(self):
        """ Gets the list of CpoIntervalVarSolution in this sequence.

        Returns:
            List of CpoIntervalVarSolution in this sequence.
        """
        return self.lvars


    def __str__(self):
        """ Convert this expression into a string """
        return str(self.get_name()) + ": (" + ", ".join([str(v.get_name()) for v in self.lvars]) + ")"

     
class CpoStateFunctionSolution(CpoVarSolution):
    """ This class represents a solution to a step function.

    A solution to a step function is represented by a list of steps.
    A step is a triplet (start, end, value) that gives the value of the function on the interval [start, end).
    """
    __slots__ = ('steps',  # List of function steps
                )
    
    def __init__(self, expr, steps):
        """ Constructor:

        Args:
            expr:  Variable expression, object of class :class:`~docplex.cp.expression.CpoStateFunction`.
            steps: List of function steps represented as tuples (start, end, value).
        """
        assert isinstance(expr, CpoStateFunction), "Expression 'expr' should be a CpoStateFunction expression"
        super(CpoStateFunctionSolution, self).__init__(expr)
        self.steps = steps


    def get_function_steps(self):
        """ Gets the list of function steps.

        Returns:
            List of function steps. Each step is a tuple (start, end, value).
        """
        return self.steps


    def get_value(self):
        """ Gets the list of function steps. Identical to `get_function_steps()`.

        Returns:
            List of function steps.
        """
        return self.steps


    def __str__(self):
        """ Convert this expression into a string """
        return str(self.get_name()) + ": (" + ", ".join([str(s) for s in self.steps]) + ")"
        
     
class CpoModelSolution(object):
    """ This class represents a solution to the problem represented by the model.
    It contains the solutions for the model variables plus the value of the objective(s), if any.

    Each variable solution can be accessed using its name, or the variable object of the model.
    The solution is either :class:`CpoIntVarSolution`, :class:`CpoIntervalVarSolution`,
    :class:`CpoSequenceVarSolution` or :class:`CpoStateFunctionSolution` depending on the type of the variable.

    A variable solution can be accessed in two ways:

     * using the method :meth:`CpoModelSolution.get_var_solution`, that returns an object representing
       the solution to the variable, or None if the variable is not in the solution.
     * using the standard Python expression `sol[<var>]` that does the same but raises a `KeyError` exception
       if the variable is not in the solution.

    Depending if the request to solver was a solve or a propagate, the solution can be:
      * *complete*, if each variable is assigned to a single value,
      * *partial* if not all variables are defined, or if some variables are defined with domains that are not
        restricted to a single value.

    An instance of this class may also be created explicitly by the programmer of the model to express a *starting point*
    that can be passed to the model to optimize its solve
    (see :meth:`docplex.cp.model.CpoModel.set_starting_point` for details).
    """
    __slots__ = ('var_solutions_dict',  # Map of variable solutions. Key is expression id or variable name, value depends on variable
                 'var_solutions_list',  # List of variable solutions. Value depends on variable
                 'objective_values',    # Objective values
                 'objective_bounds',    # Objective bound values
                 'objective_gaps',      # Objective gap values
                 'kpi_values',          # Values of the KPIs
                 )

    def __init__(self):
        super(CpoModelSolution, self).__init__()
        self.var_solutions_dict = {}
        self.var_solutions_list = []
        self.objective_values = None
        self.objective_bounds = None
        self.objective_gaps = None
        self.kpi_values = OrderedDict()


    def get_objective_values(self):
        """ Gets the numeric values of all objectives.

        If the solution is partial, each objective value may be an interval expressed as a tuple (min, max)

        Returns:
            Array of objective values, None if none.
        """
        return self.objective_values


    def get_objective_bounds(self):
        """ Gets the numeric values of all objectives bound.

        Note that when :meth:`~docplex.cp.modeler.minimize_static_lex` or :meth:`~docplex.cp.modeler.maximize_static_lex` is used,
        the bound values must be taken as a whole, as are the values delivered by :meth:`get_objective_values`.
        One cannot interpret bound values on each criterion independently.
        For example, suppose, we have a problem with two criteria specified to minimize_static_lex,
        a number of workers, and a number of days to complete a job.
        That is, we always prefer to use less workers, but for equal numbers of workers, we prefer to take less days.
        Then a solution with 3 workers and 10 days is perfectly compatible with a lower bound of 2 workers and 13 days,
        even though the lower bound on the number of days is higher than the value in the solution.

        Returns:
            Array of all objective bound values, None if none.
        """
        return self.objective_bounds


    def get_objective_gaps(self):
        """ Gets the numeric values of the gap between objective value and objective bound.

        For a single objective, gap is calculated as *gap = abs(value - bound) / max(1e-10, abs(value))*

        For multiple objectives, each gap is the gap between corresponding value and bound.
        However, after the first gap whose value is not within optimality tolerance specified by
        :attr:`~docplex.cp.CpoParameters.OptimalityTolerance` and :attr:`~docplex.cp.CpoParameters.RelativeOptimalityTolerance`,
        all returned gap values are positive infinity.

        Returns:
            Array of all objective gap values, None if none.
        """
        return self.objective_gaps


    def add_var_solution(self, vsol):
        """ Add a solution to a variable to this model solution.

        Args:
            vsol: Variable solution (object of a class extending :class:`CpoVarSolution`)
        """
        assert isinstance(vsol, CpoVarSolution), "Parameter 'vsol' should be an instance of CpoVarSolution"
        self.var_solutions_list.append(vsol)

        # Add to the dictionary with 2 keys
        var = vsol.expr
        self.var_solutions_dict[id(var)] = vsol
        vname = var.get_name()
        if vname:
            self.var_solutions_dict[vname] = vsol


    def add_var(self, var, value=None, presence=None, start=None, end=None, size=None):
        """ Add a solution to a integer or interval variable.

        Args:
            var:                 CPO variable (object of a class extending :class:`~docplex.cp.expression.CpoVariable`)
            value (Optional):    Value of the variable if the variable is a integer variable.
                                 Can be a domain if variable is not completely instantiated.
            presence (Optional): Presence indicator (true for present, false for absent, None for undetermined),
                                 if the variable is an interval variable.
            start (Optional):    Value of start, or tuple representing the start range,
                                 if the variable is an interval variable.
            end (Optional):      Value of end, or tuple representing the end range,
                                 if the variable is an interval variable.
            size (Optional):     Value of size, or tuple representing the size range,
                                 if the variable is an interval variable.
        """
        if isinstance(var, CpoIntVar):
            self.add_var_solution(CpoIntVarSolution(var, value))
        elif isinstance(var, CpoIntervalVar):
            self.add_var_solution(CpoIntervalVarSolution(var, presence, start, end, size))
        else:
            raise AssertionError("Argument 'var' should be an instance of CpoIntVar or CpoIntervalVar")


    def add_integer_var_solution(self, var, value):
        """ Add a new integer variable solution.

        The solution can be complete if the value is a single integer, or partial if the value
        is a domain, given as a list of integers or intervals expressed as tuples.

        Args:
            var:   Variable expression, object of class :class:`~docplex.cp.expression.CpoIntVar`.
            value: Variable value, or domain if not completely instantiated
        """
        self.add_var_solution(CpoIntVarSolution(var, value))


    def add_interval_var_solution(self, var, presence=None, start=None, end=None, size=None, length=None):
        """ Add a new interval variable solution.

        The solution can be complete if all attribute values are integers, or partial if at least one
        of them is an interval expressed as a tuple.

        Args:
            var:      Variable expression, object of class :class:`~docplex.cp.expression.CpoIntervalVar`.
            presence: Presence indicator (true for present, false for absent, None for undetermined). Default is None.
            start:    Value of start, or tuple representing the start range
            end:      Value of end, or tuple representing the end range
            size:     Value of size, or tuple representing the size range
            length:   Value of the length, or tuple representing the length range. Default is None.
                      Not to be used if other values are integers.
        """
        self.add_var_solution(CpoIntervalVarSolution(var, presence, start, end, size, length))


    def get_var_solution(self, expr):
        """ Gets a variable solution from this model solution.

        Args:
            expr: Variable expression or variable name if any
        Returns:
            Variable solution (class extending :class:`CpoVarSolution`),
            None if variable is not found
        """
        return self.var_solutions_dict.get(expr) if is_string(expr) else self.var_solutions_dict.get(id(expr))


    def get_all_var_solutions(self):
        """ Gets the list of all variable solutions from this model solution.

        Returns:
            List of all variable solutions (class extending :class:`CpoVarSolution`).
        """
        return self.var_solutions_list


    def get_value(self, expr):
        """ Gets the value of a variable or a KPI.

        This method first find the variable with :meth:`get_var_solution` and, if exists,
        returns the result of a call to the method get_value() on this variable.

        The result depends on the type of the variable. For details, please consult documentation of methods:

        The expression can also be the name of a KPI.

         * :meth:`CpoIntVarSolution.get_value`
         * :meth:`CpoIntervalVarSolution.get_value`
         * :meth:`CpoSequenceVarSolution.get_value`
         * :meth:`CpoStateFunctionSolution.get_value`

        Note that the builtin method *__getitem__()* is overwritten to call this method.
        Writing *sol.get_value(x)* is then equivalent to write *sol[x]*.

        Args:
            expr: Variable expression, variable name or KPI name.
        Returns:
            Variable value, None if variable is not found.
        Raises:
            KeyError if expression is not in the solution.
        """
        var = self.get_var_solution(expr)
        if var is not None:
            return var.get_value()
        return self.get_kpi_value(expr)


    def set_value(self, var, value):
        """ Sets the value of a variable.

        This method allows to set an integer variable or an interval variable with the short representation
        used to represent it, as returned by :meth:`CpoIntVarSolution.get_value`
        or :meth:`CpoIntervalVarSolution.get_value`.

        For an integer variable, value can be:

         * If the variable is fully instantiated, a single integer.
         * If the variable is partially instantiated, a domain expressed as a list of integers or intervals.

        For an interval variable, value can be:

         * If the variable is absent, an empty tuple.
         * If the variable is fully instantiated, a tuple of 3 integers (start, end, size).
         * If the variable is partially instantiated, a tuple (start, end, size, length) where each
           individual value can be an integer or an interval expressed as a tuple.

        Note that the builtin method *__setitem__()* is overwritten to call this method.
        Writing *sol.set_value(x, y)* is then equivalent to write *sol[x] = y*.

        *New in version 2.9.*

        Args:
            var: Model variable
            value: short representation of the variable value
        """
        if isinstance(var, CpoIntVar):
            self.add_integer_var_solution(var, value)
        elif isinstance(var, CpoIntervalVar):
            if not value:
                self.add_interval_var_solution(var, presence=False)
            elif len(value) == 3:
                start, end, size = value
                self.add_interval_var_solution(var, presence=True, start=start, end=end, size=size)
            elif len(value) == 4:
                start, end, size, length = value
                self.add_interval_var_solution(var, presence=True, start=start, end=end, size=size, length=length)
            else:
                raise AssertionError("Invalid value format for an interval variable")
        else:
            raise AssertionError("Variable that can be set directly are restricted to integer and interval variables")


    def add_kpi_value(self, name, value):
        """ Add a KPI value to this solution

        Args:
            name:    Name of the KPI
            value:   Model variable representing this KPI
        """
        self.kpi_values[name] = value


    def get_kpis(self):
        """ Get the solution KPIs.

        Returns:
            Ordered dictionary containing value of the KPIs that have been defined in the model.
            Key is KPI publish name, value is expression value.
            Keys are sorted in the order the KPIs have been defined.
        """
        return self.kpi_values


    def get_kpi_value(self, name):
        """ Get the value of a KPI

        Args:
            name: Name of the KPI
        Returns:
            Value of the KPI
        Raises:
            KeyError if KPI is not in the solution.
        """
        return self.kpi_values[name]


    def is_empty(self):
        """ Check whether this solution contains any information

        Returns:
            True if there is no objective value and no variable
        """
        return (self.objective_values is None) and (not self.var_solutions_dict)


    def map_solution(self, sobj):
        """ Map a python object on this solution.

        This method builds a copy of the source object and replace in its attributes all occurrences of
        model expressions by their value in this solution.
        This method is called recursively on all child objects.

        Args:
            sobj:  Source object
        Returns:
            Copy of the source object where model expressions are replaced by their values
        """
        return replace(sobj, self.get_value)


    def _add_json_solution(self, jsol, expr_map, model, prms):
        """ Add a json solution to this solution descriptor

        Args:
            jsol:     JSON document representing solution.
            expr_map: Map of model expressions. Key is name in JSON document, value is corresponding model expression.
            model:    Source model
            prms:     Solving parameters
        """
        # Add objectives
        ovals = jsol.get('objectives')
        if ovals:
            self.objective_values = tuple([_get_interval(v) for v in ovals])

        # Add objectives bounds
        bvals = jsol.get('bounds')
        if bvals:
            self.objective_bounds = tuple([_get_num_value(x) for x in bvals])

        # Add objectives gaps
        gvals = jsol.get('gaps')
        if gvals:
            self.objective_gaps = tuple([_get_num_value(x) for x in gvals])
        elif ovals and bvals and not any(is_array(v) for v in ovals):
            # Gaps not given but bounds present. Recompute gaps
            gvals = []
            rt = prms.RelativeOptimalityTolerance
            at = prms.OptimalityTolerance
            intol = True
            for v, b in zip(self.objective_values, self.objective_bounds):
                if intol:
                    gap = _compute_gap(v, b)
                    intol = _is_below_tolerance(v, b, rt, at)
                else:
                    gap = POSITIVE_INFINITY
                gvals.append(gap)
            self.objective_gaps = tuple(gvals)
        else:
            self.objective_gaps = None

        # Add integer variables
        vars = jsol.get('intVars', ())
        for vname in vars:
            var = _get_expr_from_map(expr_map, vname)
            self.add_var_solution(CpoIntVarSolution(var, _get_domain(vars[vname])))

        # Add integer variables
        vars = jsol.get('floatVars', ())
        for vname in vars:
            var = _get_expr_from_map(expr_map, vname)
            self.add_var_solution(CpoFloatVarSolution(var, _get_domain(vars[vname])))

        # Add interval variables
        vars = jsol.get('intervalVars', ())
        for vname in vars:
            var = _get_expr_from_map(expr_map, vname)
            v = vars[vname]
            if 'start' in v:
                # Check partially instantiated
                if 'presence' in v:
                    vsol = CpoIntervalVarSolution(var,  True if v['presence'] == 1 else None,
                                                  _get_domain(v['start']), _get_domain(v['end']), _get_domain(v['size']))
                    vsol.length = _get_domain(v['length'])
                else:
                    vsol = CpoIntervalVarSolution(var, True, _get_num_value(v['start']), _get_num_value(v['end']), _get_num_value(v['size']))
            else:
                vsol = CpoIntervalVarSolution(var, False)
            self.add_var_solution(vsol)

        # Add sequence variables (MUST be done after single variables)
        vars = jsol.get('sequenceVars', ())
        for vname in vars:
            var = _get_expr_from_map(expr_map, vname)
            vnlist = [v for v in vars[vname]]
            ivres = [self.get_var_solution(vn) for vn in vnlist]
            #ivres = [_get_expr_from_map(expr_map, vn) for vn in vnlist]  Should have been this instead of previous line
            self.add_var_solution(CpoSequenceVarSolution(var, ivres))

        # Add state functions
        funs = jsol.get('stateFunctions', ())
        for fname in funs:
            fun = _get_expr_from_map(expr_map, fname)
            lpts = [( _get_num_value(v['start']), _get_num_value(v['end']), _get_num_value(v['value'])) for v in funs[fname]]
            self.add_var_solution(CpoStateFunctionSolution(fun, lpts))

        # Set kpis
        kpi_values = jsol.get('KPIs', {})
        kpis = model.get_kpis()
        try:
            for name, (expr, loc) in kpis.items():
                if isinstance(expr, types.FunctionType):
                    # KPI is a lambda expression
                    value = expr(self)
                elif name in kpi_values:
                    # KPI is a solver KPI
                    value = kpi_values[name]
                else:
                    # KPI is a model variable
                    value = self.get_value(expr)
                self.add_kpi_value(name, value)
        except:
            # Solution has no values
            pass


    def __getitem__(self, expr):
        """ Overloading of [] to get a variable solution from this model solution

        Args:
            expr: Variable expression or variable name if any
        Returns:
            Variable solution (class CpoVarSolution)
        """
        return self.get_value(expr)


    def __setitem__(self, var, value):
        """ Overloading of [] to set a variable solution in this model solution

        Args:
            var: Variable expression
            value:  Variable value
        Returns:
            Variable solution (class CpoVarSolution)
        """
        return self.set_value(var, value)


    def __contains__(self, expr):
        """ Overloading of 'in' to check that a variable solution is in this model solution

        Args:
            expr: Variable expression or variable name if any
        Returns:
            True if this model solution contains a solution for this variable.
        """
        return self.get_var_solution(expr) is not None


    def print_solution(self, out=None):
        """ Prints the solution on a given output.

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        DEPRECATED. Use :meth:`write` instead.

        Args:
            out: Target output stream or output file, standard output if not given.
        """
        self.write(out)


    def write(self, out=None):
        """ Write the solution.

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out (Optional): Target output stream or file name. If not given, default value is sys.stdout.
        """
        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self.write(f)
                return
        # Check default output
        if out is None:
            out = sys.stdout

        # Print objective value, bounds and gaps
        ovals = self.get_objective_values()
        if ovals:
            out.write(u"Objective values: {}".format(ovals))
        bvals = self.get_objective_bounds()
        if bvals:
            if ovals:
                out.write(u", bounds: {}".format(bvals))
            else:
                out.write(u"Bounds: {}".format(bvals))
        gvals = self.get_objective_gaps()
        if gvals:
            out.write(u", gaps: {}".format(gvals))
        out.write(u"\n")

        # Print all variables in natural name order
        lvars = [v for v in self.get_all_var_solutions() if v.get_name()]
        lvars = sorted(lvars, key=functools.cmp_to_key(lambda v1, v2: compare_expressions(v1.expr, v2.expr)))
        for v in lvars:
            out.write(str(v))
            out.write(u'\n')

        # Print all KPIs in declaration order
        kpis = self.get_kpis()
        for k in kpis.keys():
            out.write(u'{}: {}\n'.format(k, kpis[k]))


    def __str__(self):
        """ Build a short string representation of this object.
        Returns:
            String representation of this object.
        """
        return "(objs: {}, bnds: {}, gaps: {}".format(self.get_objective_values(), self.get_objective_bounds(), self.get_objective_gaps())


    def __eq__(self, other):
        """ Overwrite equality comparison

        Args:
            other: Other object to compare with
        Returns:
            True if this object is equal to the other, False otherwise
        """
        return utils.equals(self, other)


    def __ne__(self, other):
        """ Overwrite inequality comparison """
        return not self.__eq__(other)


class CpoRunResult(object):
    """ This class is an abstract class extended by classes representing the result of a call to the solver.

    It contains the following elements:
       * model that has been solved,
       * solver parameters,
       * solver information,
       * solver output log, if configuration has been set to store it (default).
    """
    def __init__(self, model):
        super(CpoRunResult, self).__init__()
        self.model = model                      # Source model
        self.solver_log = None                  # Solver log
        self.process_infos = CpoProcessInfos()  # Process information
        self.parameters = CpoParameters()       # Solving parameters
        self.solver_infos = CpoSolverInfos()    # Solving information


    def get_model(self):
        """ Gets the source model

        Returns:
            Source model, object of class :class:`~docplex.cp.model.CpoModel`
        """
        return self.model


    def _set_solver_log(self, log):
        """ Set the solver log as a string.

        Args:
            log (str): Log of the solver
        """
        self.solver_log = log


    def get_solver_log(self):
        """ Gets the log of the solver.

        Returns:
            Solver log as a string, None if unknown.
        """
        return self.solver_log


    def get_process_infos(self):
        """ Gets the set of informations provided by the Python API concerning the solving of the model.

        Returns:
            Object of class :class:`CpoProcessInfos` that contains general information on model processing.
        """
        return self.process_infos


    def get_process_info(self, name, default=None):
        """ Get a particular process information.

        Args:
            name:    Name of the process info to get
            default: (optional) Default value if not found. None by default.
        Returns:
            Value of the process info, default value if not found.
        """
        if self.process_infos is None:
            return default
        return self.process_infos.get(name, default)


    def get_parameters(self):
        """ Gets the complete dictionary of solving parameters.

        Returns:
            Solving parameters, object of class :class:`~docplex.cp.parameters.CpoParameters`.
        """
        return self.parameters


    def get_parameter(self, name, default=None):
        """ Get a particular solving parameter.

        Args:
            name:    Name of the parameter to get
            default: (optional) Default value if not found. None by default.
        Returns:
            Parameter value, default value if not found.
        """
        if self.parameters is None:
            return default
        return self.parameters.get(name, default)


    def get_infos(self):
        """ Gets the complete dictionary of solver information attributes.

        Deprecated. use :meth:`get_solver_infos` instead.

        Returns:
            Solver information, object of class :class:`CpoSolverInfos`.
        """
        return self.solver_infos


    def get_solver_infos(self):
        """ Gets the set of information provided by the solver concerning to the solving of the model.

        Returns:
            Solver information, object of class :class:`CpoSolverInfos`.
        """
        return self.solver_infos


    def get_info(self, name, default=None):
        """ Gets a particular solver information attribute.

        Deprecated. use :meth:`get_solver_info` instead.

        Args:
            name:    Name of the information to get
            default: (optional) Default value if not found. None by default.
        Returns:
            Information attribute value, None if not found.
        """
        return self.solver_infos.get(name, default)


    def get_solver_info(self, name, default=None):
        """ Gets a particular solver information attribute.

        Args:
            name:    Name of the information to get
            default: (optional) Default value if not found. None by default.
        Returns:
            Information attribute value, None if not found.
        """
        return self.solver_infos.get(name, default)


    def _set_json_doc(self, jdoc):
        """ Set the JSON document used to build this result.

        Args:
            jdoc:  JSON object
        """
        # Add json format version in process infos
        jver = jdoc.get('cpSerializationFormatVersion')
        if jver is not None:
            self.process_infos['JsonFormatVersion'] = jver

        # Add parameters
        prms = jdoc.get('parameters', None)
        if prms is not None:
            self.parameters.update(prms)

        # Add information attributes
        cpinf = jdoc.get('cpInfo', None)
        if cpinf is not None:
            self.solver_infos.update(cpinf)


    def _is_json_format_version(self, xver):
        """ Check whether the source JSON format version is greater or equal to the argument.

        Args:
            xver:  Expected json format version
        Returns:
            True if json format version is defined and greater or equal to the required value
        """
        jver = self.process_infos.get('JsonFormatVersion')
        return (jver is not None) and (jver >= xver)


    def __str__(self):
        """ Convert this object into representative string.
        Returns:
            String representing this object
        """
        return "(model: {}, log: {})".format(self.model.get_name(), "yes" if self.solver_log else "no")


class CpoSolveResult(CpoRunResult):
    """ This class represents the result of a call to the solve of a model.

    On top of those already stored in :class:`CpoRunResult`, it contains the following elements:
       * solve status,
       * output log
       * solution, if any, object of class :class:`CpoModelSolution`.

    If this result contains a solution, the methods implemented in the class :class:`CpoModelSolution`
    to access solution elements are available directly from this class.
    """
    def __init__(self, model):
        """ Constructor:

        Args:
           model: Related model
        """
        super(CpoSolveResult, self).__init__(model)
        self.solve_status = SOLVE_STATUS_UNKNOWN   # Solve status, with value in SOLVE_STATUS_*
        self.fail_status = FAIL_STATUS_UNKNOWN     # Fail status, with values in FAIL_STATUS_*
        self.search_status = None                  # Search status, with value in SEARCH_STATUS_*
        self.stop_cause = None                     # Stop cause, with values in STOP_CAUSE_*
        self.solveTime = 0                         # Solve time
        self.is_a_solution = False                 # Solution indicator
        self.solution = CpoModelSolution()         # Solution

        self.process_infos[CpoProcessInfos.MODEL_BUILD_TIME] = model.get_modeling_duration()


    def _set_solve_status(self, ssts):
        """ Set the solve status

        Args:
            ssts: Solve status
        """
        self.solve_status = ssts


    def get_solve_status(self):
        """ Gets the solve status.

        Returns:
            Solve status, element of the global list :const:`ALL_SOLVE_STATUSES`.
        """
        return self.solve_status


    def get_fail_status(self):
        """ Gets the solving fail status.

        This method is deprecated since release 12.8.
        Use :meth:`~CpoSolveResult.get_search_status` and :meth:`~CpoSolveResult.get_stop_cause` instead.

        Returns:
            Fail status, element of the global list :const:`ALL_FAIL_STATUSES`.
        """
        return self.fail_status


    def get_search_status(self):
        """ Gets the search status.

        This solver information is provided by the COS 12.8 CP solver in addition/replacement to solve_status.
        Value is None if the solver is earlier than this version.

        Returns:
            Search status, element of the global list :const:`ALL_SEARCH_STATUSES`.
            None if not defined.
        """
        return self.search_status


    def get_stop_cause(self):
        """ Gets the stop cause.

        This solver information is provided by the COS 12.8 CP solver in addition/replacement to fail_status.
        Value is None if the solver is earlier than this version.

        Returns:
            Stop cause, element of the global list :const:`ALL_STOP_CAUSES`.
            None if not defined.
        """
        return self.stop_cause


    def is_solution(self):
        """ Checks if this descriptor is a valid solution to the problem.

        A solution is present if the solve status is 'Feasible' or 'Optimal'.
        Optimality of the solution should be tested using method :meth:`is_solution_optimal()`.

        Returns:
            True if this descriptor is a valid solution to the problem.
        """
        return self.is_a_solution
        #return ((self.solve_status in (SOLVE_STATUS_FEASIBLE, SOLVE_STATUS_OPTIMAL)) and (self.fail_status != FAIL_STATUS_FAILED_NORMALLY)) \
        #        or ((self.solve_status == SOLVE_STATUS_UNKNOWN) and (self.fail_status == FAIL_STATUS_HAS_NOT_FAILED))
        #return ((self.solve_status in (SOLVE_STATUS_FEASIBLE, SOLVE_STATUS_OPTIMAL)) and (self.fail_status != FAIL_STATUS_SEARCH_COMPLETED)) \
        #        or ((self.solve_status == SOLVE_STATUS_UNKNOWN) and (self.fail_status == FAIL_STATUS_HAS_NOT_FAILED))


    def is_solution_optimal(self):
        """ Checks if this descriptor contains an optimal solution to the problem.

        Returns:
            True if there is a solution that is optimal.
        """
        return self.solve_status == SOLVE_STATUS_OPTIMAL


    def map_solution(self, sobj):
        """ Map a python object on this solution.

        This method builds a copy of the source object and replace in its attributes all occurrences of
        model expressions by their value in this solution.
        This method is called recursively on all child objects.

        Args:
            sobj:  Source object
        Returns:
            Copy of the source object where model expressions are replaced by their values
        """
        return self.solution.map_solution(sobj)


    def __nonzero__(self):
        """ Check if this descriptor contains a solution to the problem.
        Equivalent to is_solution()

        Returns:
            True if a solution is available (Search status is 'Feasible' or 'Optimal')
        """
        return self.is_solution()


    def __bool__(self):
        """ Check if this descriptor contains a solution to the problem.
        Equivalent to is_solution()

        Equivalent to __nonzero__ for Python 3

        Returns:
            True if a solution is available (Search status is 'Feasible' or 'Optimal')
        """
        return self.is_solution()


    def get_solution(self):
        """ Get the model solution

        Returns:
            Model solution, object of class :class:`CpoModelSolution`.
        """
        return self.solution


    def get_objective_values(self):
        """ Gets the numeric values of all objectives.

        Returns:
            Array of all objective values, None if none.
        """
        return self.solution.get_objective_values()


    def get_objective_bounds(self):
        """ Gets the numeric values of all objectives bound.

        Note that when :meth:`~docplex.cp.modeler.minimize_static_lex` or :meth:`~docplex.cp.modeler.maximize_static_lex` is used,
        the bound values must be taken as a whole, as are the values delivered by :meth:`get_objective_values`.
        One cannot interpret bound values on each criterion independently.
        For example, suppose, we have a problem with two criteria specified to minimize_static_lex,
        a number of workers, and a number of days to complete a job.
        That is, we always prefer to use less workers, but for equal numbers of workers, we prefer to take less days.
        Then a solution with 3 workers and 10 days is perfectly compatible with a lower bound of 2 workers and 13 days,
        even though the lower bound on the number of days is higher than the value in the solution.

        Returns:
            Array of all objective bound values, None if none.
        """
        return self.solution.get_objective_bounds()


    def get_objective_gaps(self):
        """ Gets the numeric values of the gap between objective value and objective bound.

        For a single objective, gap is calculated as gap = \|value - bound\| / max(1e-10, \|value\|)

        For multiple objectives, each gap is the gap between corresponding value and bound.
        However, after the first gap whose value is not within optimality tolerance specified by
        :attr:`~docplex.cp.CpoParameters.OptimalityTolerance` and :attr:`~docplex.cp.CpoParameters.RelativeOptimalityTolerance`,
        all returned gap values are positive infinity.

        Returns:
            Array of all objective gap values, None if not defined.
        """
        return self.solution.objective_gaps


    def get_kpis(self):
        """ Get the solution kpis

        Returns:
            Dictionary containing value of the KPIs that have been defined in the model.
        """
        return self.solution.get_kpis()


    def _set_model_attributes(self, nbintvars=0, nbitvvars=0, nbseqvars=0, nbctrs=0):
        """ Set the general model attributes.

        This method is called when solve is done on the cloud, when not all information is available from the solver.

        Args:
            nbintvars: Number of integer variables
            nbitvvars: Number of interval variables
            nbseqvars: Number of sequence variables
            nbctrs:    Number of constraints
        """
        self.solver_infos[CpoSolverInfos.NUMBER_OF_INTEGER_VARIABLES] = nbintvars
        self.solver_infos[CpoSolverInfos.NUMBER_OF_INTERVAL_VARIABLES] = nbitvvars
        self.solver_infos[CpoSolverInfos.NUMBER_OF_SEQUENCE_VARIABLES] = nbseqvars
        self.solver_infos[CpoSolverInfos.NUMBER_OF_CONSTRAINTS] = nbctrs


    def _set_solve_time(self, time):
        """ Set the solve time required for this solution.

        Args:
            time (float): Solve time in seconds
        """
        self.solveTime = time


    def get_solve_time(self):
        """ Gets the solve time required for this solution.

        Returns:
            (float) Solve time in seconds.
        """
        return self.solveTime


    def get_var_solution(self, name):
        """ Gets a variable solution from this model solution.

        Args:
            name: Variable name or variable expression.
        Returns:
            Variable solution, object of class :class:`CpoVarSolution`, None if not found.
        """
        return self.solution.get_var_solution(name)


    def get_all_var_solutions(self):
        """ Gets the list of all variable solutions from this model solution.

        Returns:
            List of all variable solutions (class :class:`CpoVarSolution`).
        """
        return self.solution.get_all_var_solutions()


    def get_value(self, name):
        """ Gets the value of a variable.

        For IntVar, value is an integer.
        For IntervalVar, value is a tuple (start, end, size), () if absent.
        For SequenceVar, value is list of interval variable solutions.
        For StateFunction, value is list of steps.

        Args:
            name: Variable name, or model variable descriptor.
        Returns:
            Variable value, None if variable is not found.
        """
        return self.solution.get_value(name)


    def _add_json_solution(self, jsol, expr_map):
        """ Add a json solution to this result descriptor

        Args:
            jsol:     JSON document representing solution.
            expr_map: Map of model expressions. Key is name in JSON document, value is corresponding model expression.
        """
        # Notify run result about JSON document
        self._set_json_doc(jsol)

        # Add solution
        self.solution._add_json_solution(jsol, expr_map, self.model, self.parameters)

        # Add solver status
        status = jsol.get('solutionStatus', None)
        if status:
            self.solve_status  = status.get('solveStatus', self.solve_status)
            self.fail_status   = status.get('failStatus', self.fail_status)
            self.search_status = status.get('SearchStatus')
            self.stop_cause    = status.get('SearchStopCause')

            nsts = status.get('nextStatus')
            if nsts in ('NextFalse', 'NextTerminated'):
                # Only for end of search_next
                self.fail_status = FAIL_STATUS_SEARCH_COMPLETED
                self.is_a_solution = (self.solve_status == SOLVE_STATUS_OPTIMAL) \
                                     and (self.solution.get_objective_values() is not None)
            else:
                rto = jsol.get('responseTo', None)
                if rto == 'Propagate':
                    self.is_a_solution = (self.solve_status != SOLVE_STATUS_INFEASIBLE) and (self.search_status == SEARCH_STATUS_COMPLETED)
                else:
                    self.is_a_solution = self.solve_status in (SOLVE_STATUS_FEASIBLE, SOLVE_STATUS_OPTIMAL)


    def __getitem__(self, name):
        """ Overloading of [] to get a variable solution from this model solution

        Args:
            name: Variable name or CPO variable expression
        Returns:
            Variable solution (class CpoVarSolution)
        """
        return self.get_value(name)


    def print_solution(self, out=None):
        """ Prints the solution on a given output.

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        DEPRECATED. Use write() instead.

        Args:
            out: Target output stream or output file, standard output if not given.
        """
        self.write(out)


    def write(self, out=None):
        """ Write the solve result

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out (Optional): Target output stream or file name. If not given, default value is sys.stdout.
        """
        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self.write(f)
                return
        # Check default output
        if out is None:
            out = sys.stdout

        # Print model attributes
        sinfos = self.get_solver_infos()
        out.write(u"-------------------------------------------------------------------------------\n")
        out.write(u"Model constraints: " + str(sinfos.get_number_of_constraints()))
        out.write(u", variables: integer: " + str(sinfos.get_number_of_integer_vars()))
        out.write(u", interval: " + str(sinfos.get_number_of_interval_vars()))
        out.write(u", sequence: " + str(sinfos.get_number_of_sequence_vars()))
        out.write(u'\n')

        # Print search/solve status
        s = self.get_search_status()
        if s:
            out.write(u"Solve status: " + str(self.get_solve_status()) + "\n")
            out.write(u"Search status: " + str(s))
            s = self.get_stop_cause()
            if s:
                out.write(u", stop cause: " + str(s))
            out.write(u"\n")
        else:
            # Old fashion
            out.write(u"Solve status: " + str(self.get_solve_status()) + ", Fail status: " + str(self.get_fail_status()) + "\n")
        # Print solve time
        out.write(u"Solve time: " + str(round(self.get_solve_time(), 2)) + " sec\n")
        out.write(u"-------------------------------------------------------------------------------\n")

        self.solution.write(out)


    def write_in_string(self):
        """ Build a string representation of this object.

        The string that is returned is the same than what is printed by calling :meth:`write`.

        Returns:
            String representation of this object.
        """
        out = StringIO()
        self.write(out)
        res = out.getvalue()
        out.close()
        return res


    def __str__(self):
        """ Build a short string representation of this object.
        Returns:
            String representation of this object.
        """
        res = "(model: {}, solve: {}, search: {}".format(self.model.get_name(), self.get_solve_status(), self.get_search_status())
        if self.is_solution():
            res += ", solution: {}".format(self.get_solution())
        res += ")"
        return res


    def __eq__(self, other):
        """ Overwrite equality comparison

        Args:
            other: Other object to compare with
        Returns:
            True if this object is equal to the other, False otherwise
        """
        return utils.equals(self, other)

    def __ne__(self, other):
        """ Overwrite inequality comparison """
        return not self.__eq__(other)


class CpoRefineConflictResult(CpoRunResult):
    """ This class represents the result of a call to the conflict refiner.

    A conflict is a subset of the constraints and/or variables of the model which are
    mutually contradictory.

    The conflict refiner first examines the full infeasible model to identify portions of the conflict
    that it can remove. By this process of refinement, the conflict refiner arrives at a minimal conflict.
    A minimal conflict is usually smaller than the full infeasible model and thus makes infeasibility analysis easier.
    Since the conflict is minimal, removal of any one of these constraints will remove that particular cause
    for infeasibility.
    There may be other conflicts in the model; consequently, repair of a given conflict does not guarantee
    feasibility of the remaining model.
    If a model happens to include multiple independent causes of infeasibility,
    then it may be necessary for the user to repair one such cause and then repeat the diagnosis with further
    conflict analysis.
    """
    def __init__(self, model):
        # """ Creates a new empty conflict refiner result.
        #
        # Args:
        #    model: Related model
        # """
        super(CpoRefineConflictResult, self).__init__(model)
        self.member_constraints = []         # List of member constraints
        self.possible_constraints = []       # List of possible member constraints
        self.member_variables = []           # List of member variables
        self.possible_variables = []         # List of possible member variables
        self.solver_infos = CpoSolverInfos() # Solving information
        self.cpo_conflict = None             # Conflict in CPO format


    def get_all_member_constraints(self):
        """ Returns the list of all constraints that are certainly member of the conflict.

        Returns:
            List of model constraints (class CpoExpr) certainly member of the conflict.
        """
        return self.member_constraints


    def get_all_possible_constraints(self):
        """ Returns the list of all constraints that are possibly member of the conflict.

        Returns:
            List of model constraints (class CpoExpr) possibly member of the conflict.
        """
        return self.possible_constraints


    def get_all_member_variables(self):
        """ Returns the list of all variables that are certainly member of the conflict.

        Returns:
            List of model variables (class CpoIntVar or CpoIntervalVar) certainly member of the conflict.
        """
        return self.member_variables


    def get_all_possible_variables(self):
        """ Returns the list of all variables that are possibly member of the conflict.

        Returns:
            List of model variables (class CpoIntVar or CpoIntervalVar) possibly member of the conflict.
        """
        return self.possible_variables


    def get_cpo(self):
        """ Returns the conflict represented in CPO format.

        Returns:
            String containing the conflict in CPO format, None if not given.
        """
        return self.cpo_conflict


    def is_conflict(self):
        """ Checks if this descriptor contains a valid conflict.

        Returns:
            True if there is a conflict, False otherwise.
        """
        return len(self.member_constraints) != 0 or len(self.possible_constraints) != 0 \
               or len(self.member_variables) != 0 or len(self.possible_variables) != 0


    def __nonzero__(self):
        """ Check if this descriptor contains a conflict.
        Equivalent to is_conflict()

        Returns:
            True if there is a conflict, False otherwise.
        """
        return self.is_conflict()


    def __bool__(self):
        """ Check if this descriptor contains a conflict.
        Equivalent to is_conflict()

        Equivalent to __nonzero__ for Python 3

        Returns:
            True if there is a conflict, False otherwise.
        """
        return self.is_conflict()


    def _add_json_solution(self, jsol, expr_map):
        """ Add a json solution to this result descriptor

        Args:
            jsol:     JSON document representing solution.
            expr_map: Map of model expressions. Key is name in JSON document, value is corresponding model expression.
        """
        # Notify run result about JSON document
        self._set_json_doc(jsol)

        # Get conflict data
        conflict = jsol.get('conflict')
        if conflict is None:
            return

        # Add constraints
        for name, status in conflict.get('constraints', {}).items():
            expr = _get_expr_from_map(expr_map, name)
            if status == 'ConflictMember':
                self.member_constraints.append(expr)
            else:
                self.possible_constraints.append(expr)

        # Add variables
        vars = conflict.get('intVars', {}).copy()
        vars.update(conflict.get('intervalVars', {}))
        for name, status in vars.items():
            expr = _get_expr_from_map(expr_map, name)
            if status == 'ConflictMember':
                self.member_variables.append(expr)
            else:
                self.possible_variables.append(expr)


    def print_conflict(self, out=None):
        """ Prints this conflict on a given output.

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        DEPRECATED. Use :meth:`write` instead.

        Args:
            out: Target output stream or output file, standard output if not given.
        """
        self.write(out)


    def write(self, out=None):
        """ Write the conflict

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out (Optional): Target output stream or file name. If not given, default value is sys.stdout.
        """
        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self.write(f)
                return
        # Check default output
        if out is None:
            out = sys.stdout

        out.write(u"Conflict refiner result:\n")
        if not self.is_conflict():
            out.write(u"   No conflict\n")
            return

        # Print constraints in the conflict
        lc = self.get_all_member_constraints()
        if lc:
            out.write(u"Member constraints:\n")
            for c in lc:
                out.write(u"   {}\n".format(_build_conflict_constraint_string(c)))
        lc = self.get_all_possible_constraints()
        if lc:
            out.write(u"Possible member constraints:\n")
            for c in lc:
                out.write(u"   {}\n".format(_build_conflict_constraint_string(c)))

        # Print variables in the conflict
        lc = self.get_all_member_variables()
        if lc:
            out.write(u"Member variables:\n")
            for c in lc:
                out.write(u"   {}\n".format(c))
        lc = self.get_all_possible_variables()
        if lc:
            out.write(u"Possible member variables:\n")
            for c in lc:
                out.write(u"   {}\n".format(c))

        # Print cpo format if any
        cpo = self.get_cpo()
        if cpo:
            out.write(u"Conflict in CPO format:\n")
            for line in cpo.splitlines():
                out.write(u"   " + line + "\n")


    def __str__(self):
        """ Build a string representation of this object.

        The string that is returned is the same than what is printed by calling :meth:`write`.

        Returns:
            String representation of this object.
        """
        out = StringIO()
        self.write(out)
        res = out.getvalue()
        out.close()
        return res


    def __eq__(self, other):
        """ Overwrite equality comparison

        Args:
            other: Other object to compare with
        Returns:
            True if this object is equal to the other, False otherwise
        """
        return utils.equals(self, other)


class CpoSolverInfos(InfoDict):
    """ Dictionary of various solver informations.

    This class groups various information returned by the solver at the end of the solve.
    It is implemented as an extension of the class :class:`docplex.cp.utils.InfoDict` and takes profit of
    the methods such as :meth:`~docplex.cp.utils.InfoDict.write` that allows to easily print
    the full content of the information structure.
    """

    # Total number of constraints
    NUMBER_OF_CONSTRAINTS        = 'NumberOfConstraints'

    # Total number of integer variables
    NUMBER_OF_INTEGER_VARIABLES  = 'NumberOfIntegerVariables'

    # Total number of interval variables
    NUMBER_OF_INTERVAL_VARIABLES = 'NumberOfIntervalVariables'

    # Total number of sequence variables
    NUMBER_OF_SEQUENCE_VARIABLES = 'NumberOfSequenceVariables'

    # Total solve time
    SOLVE_TIME = 'SolveTime'

    def __init__(self):
        super(InfoDict, self).__init__()

    def get_number_of_integer_vars(self):
        """ Gets the number of integer variables in the model.

        Returns:
            Number of integer variables.
        """
        return self.get(CpoSolverInfos.NUMBER_OF_INTEGER_VARIABLES, 0)


    def get_number_of_interval_vars(self):
        """ Gets the number of interval variables in the model.

        Returns:
            Number of interval variables.
        """
        return self.get(CpoSolverInfos.NUMBER_OF_INTERVAL_VARIABLES, 0)


    def get_number_of_sequence_vars(self):
        """ Gets the number of sequence variables in the model.

        Returns:
            Number of sequence variables.
        """
        return self.get(CpoSolverInfos.NUMBER_OF_SEQUENCE_VARIABLES, 0)


    def get_number_of_constraints(self):
        """ Gets the number of constraints in the model.

        Returns:
            Number of constraints.
        """
        return self.get(CpoSolverInfos.NUMBER_OF_CONSTRAINTS, 0)


    def get_solve_time(self):
        """ Gets the total solve time.

        Returns:
            Total solve time in seconds, -1 if unknown
        """
        return self.get(CpoSolverInfos.SOLVE_TIME, -1)


class CpoProcessInfos(InfoDict):
    """ Dictionary of various process information.

    This class groups various information related to the processing of the model by the Python API.
    It is implemented as an extension of the class :class:`~docplex.cp.utils.InfoDict` and takes profit of
    the methods such as :meth:`~docplex.cp.utils.InfoDict.write` that allows to easily print
    the full content of the information structure.

    Note that the content is purely informative. Information names and values depends on the implementation
    of the solver agent that has been used to solve the model.

    This class provides few methods to access the most important information stored in it.
    All information is available using regular dictionary access expression.
    """

    # Name of the agent used to solve the model
    SOLVER_AGENT = "SolverAgent"

    # Model build time (time between model creation and last addition of an expression)
    MODEL_BUILD_TIME = "ModelBuildTime"

    # Attribute name for time needed to transform model into CPO format
    MODEL_COMPILE_TIME = "ModelCompileTime"

    # Attribute name for time needed to dump model in file and/or on log
    MODEL_DUMP_TIME = "ModelDumpTime"

    # Attribute name for time needed to submit the model to solver
    MODEL_SUBMIT_TIME = "ModelSubmitTime"

    # Attribute name for the size of the generated CPO model
    MODEL_DATA_SIZE = "ModelDataSize"

    # Attribute name for total solve time (including model send and response receive)
    SOLVE_TOTAL_TIME = "TotalSolveTime"

    # Attribute name for time needed to retrieve result
    RESULT_RECEIVE_TIME = "ResultReceiveTime"

    # Attribute name for size of the result string
    RESULT_DATA_SIZE = "ResultDataSize"

    # Attribute name for total time needed to parse JSON result
    TOTAL_JSON_PARSE_TIME = "TotalJsonParseTime"

    # Attribute name for size of the log data
    TOTAL_LOG_DATA_SIZE = "TotalLogDataSize"

    # Attribute name for total time needed to encode strings in UTF8
    TOTAL_UTF8_ENCODE_TIME = "TotalUtf8EncodeTime"

    # Attribute name for total time needed to decode strings from UTF-8
    TOTAL_UTF8_DECODE_TIME = "TotalUtf8DecodeTime"

    # Time needed to send model to the solver
    TOTAL_DATA_SEND_TIME = "TotalDataSendTime"

    # Total size of data sent to solver
    TOTAL_DATA_SEND_SIZE = "TotalDataSendSize"

    # Total size of data received from solver
    TOTAL_DATA_RECEIVE_SIZE = "TotalDataReceiveSize"


    def __init__(self):
        super(InfoDict, self).__init__()


    def get_model_build_time(self):
        """ Get the time spent to build the model.

        Modeling time is computed as the time spent between model creation and last addition of a model expression.

        Returns:
            Total modeling time in seconds.
        """
        return self.get(CpoProcessInfos.MODEL_BUILD_TIME)


    def get_total_solve_time(self):
        """ Get the total solve time, including time to send model and retrieve result.

        Returns:
            Total solve time in seconds.
        """
        return self.get(CpoProcessInfos.SOLVE_TOTAL_TIME)


###############################################################################
##  Private functions
###############################################################################

# Constants conversion
_NUMERIC_VALUES = {# Numeric value generated by CPO
                     'intmin': INT_MIN, 'intmax': INT_MAX,
                     'intervalmin': INTERVAL_MIN, 'intervalmax': INTERVAL_MAX,
                     'infinity': POSITIVE_INFINITY, '-infinity': NEGATIVE_INFINITY,
                     # Numeric values generated by JSON
                     'NaN': float('nan'),
                     'Infinity': POSITIVE_INFINITY, '-Infinity': NEGATIVE_INFINITY}


# Marker of interval with holes
_HOLE_MARKER = "holes"

def _get_domain(val):
    """ Convert a solution value into domain.

    Args:
        val: Value to convert
    Returns:
        Variable domain
    """
    if is_array(val):
        res = []
        for v in val:
            if is_array(v):
                vl = len(v)
                if vl == 2:
                    res.append((_get_num_value(v[0]), _get_num_value(v[1])))
                elif vl == 3:
                    res.append((_get_num_value(v[0]), _get_num_value(v[1]), _HOLE_MARKER))
                    assert v[2] == _HOLE_MARKER, "Domain interval with 3 elements must contains '{}' as last one".format(_HOLE_MARKER)
                else:
                    assert False, "Domain interval should contain only 2 elements"
            else:
                res.append(_get_num_value(v))
        return tuple(res)
    else:
        return _get_num_value(val)


def _get_interval(val):
    """ Convert a solution value given in JSON.

    Args:
        val: JSON value to convert
    Returns:
        Converted value
    """
    if isinstance(val, list):
        lb, ub = val
        return (lb, ub) if lb != ub else _get_num_value(lb)
    return _get_num_value(val)


def _get_num_value(val):
    """ Convert a solution value into number.
    Interpret intmin, intmax, intervalmin, intervalmax, NaN, Infinity if any.

    Args:
        val: Value to convert
    Returns:
        Converted value, itself if not found
    """
    return _NUMERIC_VALUES.get(val, val)


def _check_arg_domain(val, name):
    """ Check that an argument is a correct domain and raise error if wrong

    Domain is:
       * a single integer for a fixed domain
       * a list of integers or intervals expressed as tuples.

    Args:
        val:  Argument value
        name: Argument name
    Returns:
        Domain to be set
    Raises:
        Exception if argument has the wrong format
    """
    # Check single integer
    if is_int(val):
        return val
    # Check list of integers or tuples
    assert is_array(val), "Argument '" + name + "' should be a list of integers and/or intervals"
    for v in val:
        if not is_int(v):
            assert _is_domain_interval(v), "Argument '" + name + "' should be a list of integers and/or intervals (tuples of 2 integers)"
    return val


def _is_domain_interval(val):
    """ Check if a value is representing a valid domain interval
    Args:
        val:  Value to check
    Returns:
        True if value is a tuple representing an interval
    """
    if not isinstance(val, tuple):
        return False
    if not (is_int(val[0]) and is_int(val[1]) and (val[1] >= val[0])):
        return False
    vl = len(val)
    if vl == 2:
        return True
    if vl == 3:
        return val[2] == _HOLE_MARKER
    return False


def _build_conflict_constraint_string(ctr):
    """ Build the string used to represent a constraint in conflict refiner
    Args:
        ctr:  Constraint to print
    Returns:
        Constraint string
    """
    return str(ctr)


def _compute_gap(val, bnd):
    """ Compute the gap of a value
    Args:
        val:  Objective value
        bnd:  Objective bound
    Returns:
        Objective gap
    """
    if val in (POSITIVE_INFINITY, NEGATIVE_INFINITY) or bnd in (POSITIVE_INFINITY, NEGATIVE_INFINITY):
        return POSITIVE_INFINITY
    if not is_number(val) or not is_number(bnd):
        return POSITIVE_INFINITY
    return float(abs(val - bnd)) / max(1e-10, abs(val))


def _is_below_tolerance(val, bnd, rt, at):
    """ Check if an objective value is in the tolerance with given bound.
    Args:
        val:  Value to check
        bnd:  Objective bound
        rt:   Relative tolerance
        at:   Absolute tolerance
    Returns:
        True if value is below the tolerance, false otherwise
    """
    if not is_number(val) or not is_number(bnd):
        return False
    if val in (POSITIVE_INFINITY, NEGATIVE_INFINITY) or bnd in (POSITIVE_INFINITY, NEGATIVE_INFINITY):
        return False
    if val == bnd:
        return True
    if val < bnd: # Maximization
       val = -val
       bnd = -bnd
    return (val - at < bnd) or (val * (1 - rt) < bnd)


def _get_expr_from_map(expr_map, id):
    """ Retrieve a model expression from the map of CPO ids for expressions
    Args:
        expr_map: Map of model expressions. Key is name in JSON document, value is corresponding model expression.
    Returns:
        Model expression
    Raises:
        CpoException if expression is not found (should not happen)
    """
    expr = expr_map.get(id)
    if expr is None:
        raise CpoException("INTERNAL ERROR: Solve result refers to '{}' that is not found in the map of expressions".format(id))
    return expr


