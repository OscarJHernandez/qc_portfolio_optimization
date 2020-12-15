# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains principally the class :class:`CpoModel` that handles all the elements that compose a CPO model:

 * the variables of the domain (integer variables, interval variables, sequence variables and state functions),
 * the constraints of the model,
 * optional objective value(s),
 * optional search phases,
 * optional starting point (available for CPO solver release greater or equal to 12.7.0).

The model expressions, constraints, objective and search phases can be added using method :meth:`~CpoModel.add`.
Variables that appear in the expressions are automatically added to the model.

A starting point can be added to the model using method :meth:`~CpoModel.set_starting_point`

The different model expressions and elements are created using services provided by modules:

 * :mod:`docplex.cp.expression` for the simple expression elements,
 * :mod:`docplex.cp.modeler` for complex expressions and constraints using the specialized CP Optimizer functions.

The solving of the model is handled by an object of class :class:`~docplex.cp.solver.solver.CpoSolver` that takes
this model as parameter.
However, most important solving functions are callable directly from this model to avoid explicit
creation of the *CpoSolver* object:

 * :meth:`~CpoModel.solve` solves the model and returns an object of class :class:`~docplex.cp.solution.CpoSolveResult`.
 * :meth:`~CpoModel.start_search` creates a solver that can iterate over multiple solutions of the model.
 * :meth:`~CpoModel.refine_conflict` identifies a minimal conflict for the infeasibility and return it as an object
   of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
 * :meth:`~CpoModel.propagate` invokes the propagation on the current model and returns a partial solution in an object
   of class :class:`~docplex.cp.solution.CpoSolveResult`.

All these methods are taking a variable number of optional parameters that allow to modify the solving context.
The list of arguments is not limited. Each named argument is used to replace the leaf attribute that has
the same name in the global *context* structure initialized in the module :mod:`docplex.cp.config` and its
customizations.

The most important of these parameters are:

 * **context** sets a complete customized context to be used instead of the default one defined in the module :mod:`docplex.cp.config`,
 * **params** overwrites the solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
   that are defined in the *context* object,
 * **agent** forces the selection of a particular solving agent,
 * **trace_cpo** activates the printing of the model in CPO format before its solve,
 * any CP Optimizer solving parameter, as defined in module :mod:`docplex.cp.parameters`, such as:

    * **TimeLimit** indicates a limit in seconds in the time spent in the solve,
      or **ConflictRefinerTimeLimit** that does the same for conflict refiner,
    * **LogVerbosity**, with values in ['Quiet', 'Terse', 'Normal', 'Verbose'],
    * **Workers** specifies the number of threads assigned to solve the model (default value is the number of cores),
    * **SearchType**, with value in ['DepthFirst', 'Restart', 'MultiPoint', 'Auto'], to select a particular solving algorithm,
    * **RandomSeed** changes the seed of the random generator,
    * and so on.

Detailed description
--------------------
"""

# Following imports required to allow modeling just importing this module
from docplex.cp.modeler import *
from docplex.cp.solution import *
from docplex.cp.expression import *
from docplex.cp.function import *
from docplex.cp.solver.solver_listener import CpoSolverListener
from docplex.cp.solver.cpo_callback import CpoCallback

# Imports required locally
import docplex.cp.config as config
import docplex.cp.expression as expression
import docplex.cp.modeler as modeler
from docplex.cp.solver.solver import CpoSolver
from docplex.cp.cpo.cpo_compiler import CpoCompiler
import docplex.cp.utils as utils
import inspect
import sys
import time
import copy
import types
from collections import OrderedDict



###############################################################################
##  Constants
###############################################################################

# Marker of this file to remove it from source location
_THIS_FILE_MARKER = "docplex" + os.path.sep + "cp" + os.path.sep + "model."


###############################################################################
##  Public classes
###############################################################################

# Model statistics
class CpoModelStatistics(object):
    """ This class represents model statistics information.
    """

    def __init__(self, model=None, json=None):
        """ Create new model statistics

        Can be created either by giving source model, or json object.

        Args:
            model: (Optional) Source model
            json:  (Optional) Json representation of this object
        """
        if json is not None:
            self.nb_root_exprs = json.get('nb_root_exprs', 0)
            self.nb_integer_var = json.get('nb_integer_var', 0)
            self.nb_interval_var = json.get('nb_interval_var', 0)
            self.nb_expr_nodes = json.get('nb_expr_nodes', 0)
            self.operation_usage = json.get('operation_usage')
        else:
            if model is None:
                self.nb_root_exprs = 0
            else:
                self.nb_root_exprs = len(model.expr_list) + (0 if model.objective is None else 1)
            self.nb_integer_var   = 0     # Number of integer variables
            self.nb_interval_var  = 0     # Number of interval variables
            self.nb_expr_nodes    = 0     # Number of expression nodes
            self.operation_usage  = {}    # Map of operation usage count.
                                          # Key is the CPO name of the operation, value is the number of times it is used.

    def _add_expression(self, expr):
        """ Update statistics with an expression node.

        Args:
            expr:  Expression
        """
        self.nb_expr_nodes += 1
        if isinstance(expr, CpoIntVar):
            self.nb_integer_var += 1
        elif isinstance(expr, CpoIntervalVar):
            self.nb_interval_var += 1
        elif isinstance(expr, CpoFunctionCall):
            opname = expr.operation.cpo_name
            self.operation_usage[opname] = self.operation_usage.get(opname, 0) + 1

    def add(self, other):
        """ Add other model statistics to this one

        Args:
            other:  Other model statistics, object of class CpoModelStatistics
        """
        self.nb_root_exprs   += other.nb_root_exprs
        self.nb_integer_var  += other.nb_integer_var
        self.nb_interval_var += other.nb_interval_var
        self.nb_expr_nodes   += other.nb_expr_nodes
        for k, v in other.operation_usage.items():
            self.operation_usage[k] = self.operation_usage.get(k, 0) + other.operation_usage.get(k, 0)


    def write(self, out=None, prefix=""):
        """ Write the statistics

        Args:
            out (Optional):    Target output, as stream or file name. sys.stdout if not given
            prefix (Optional): Prefix added at the beginning of each line
        """
        # Check file
        if is_string(out):
            with open(os.path.abspath(out), mode='w') as f:
                self.write(f)
            return

        if out is None:
            out = sys.stdout

        # Write normal attributes
        out.write("{}number of integer variables:  {}\n".format(prefix, self.nb_integer_var))
        out.write("{}number of interval variables: {}\n".format(prefix, self.nb_interval_var))
        out.write("{}number of expressions:        {}\n".format(prefix, self.nb_root_exprs))
        out.write("{}number of expression nodes:   {}\n".format(prefix, self.nb_expr_nodes))
        out.write("{}operations:                   ".format(prefix))
        if self.operation_usage:
            for i, k in enumerate(sorted(self.operation_usage.keys())):
                if i > 0:
                    out.write(", ")
                out.write("{}: {}".format(k, self.operation_usage[k]))
        else:
            out.write("None")
        out.write("\n")

    def to_json(self):
        """ Build a json object from this statistics

        Returns:
            JSON object
        """
        # Build json object for stats
        return {
            'nb_root_exprs':   self.nb_root_exprs,
            'nb_integer_var':  self.nb_integer_var,
            'nb_interval_var': self.nb_interval_var,
            'nb_expr_nodes':   self.nb_expr_nodes,
            'operation_usage': self.operation_usage
        }

    def __str__(self):
        """ Build a string representing this object

        Returns:
            String representing this object
        """
        return "IntegerVars: {}, IntervalVars: {}, Exprs: {}, Nodes: {}, Ops: {}"\
            .format(self.nb_integer_var, self.nb_interval_var, self.nb_root_exprs, self.nb_expr_nodes, len(self.operation_usage))

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


class CpoModel(object):
    """ This class is the Python container of a CPO model.
    """

    def __init__(self, name=None, sfile=None, version=None):
        """ Constructor.

        Args:
            name:    (Optional) Model name (source file name).
            sfile:   (Optional) Source file.
            version: (Optional) Format version
        """
        ctx = config.get_default()
        super(CpoModel, self).__init__()
        self.expr_list        = []            # List of model root expressions as tuples (expression, location)
        self.parameters       = None          # Solving parameters
        self.starting_point   = None          # Starting point
        self.objective        = None          # Objective function
        self.kpis             = OrderedDict() # Dictionary of KPIs. Key is publish name, value is (expr, loc)
        self.listeners        = []            # Solver listeners
        self.callbacks        = []            # Solver callbacks

        # Set version of the CPO format (None = not given)
        self.format_version   = version

        # Indicate to set source location in the model information
        self.source_loc       = ctx.get_by_path("model.add_source_location", True)

        # Initialize times to compute modeling time
        self.create_time      = time.time()        # Model creation absolute time
        self.last_add_time    = self.create_time   # Last time something has been added to the model

        # Store filename of the calling Python source
        if sfile is None:
            loc = self._get_calling_location()
            if loc is not None:
                sfile = loc[0]
        self.source_file = sfile.replace('\\', '/')

        # Store model name
        self.name = name

        # Duplicate model expressions constructor functions to make them callable from the model
        _set_all_modeling_functions(self)


    def __enter__(self):
        # Implemented for compatibility with cplex
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        # Implemented for compatibility with cplex
        return False  # No exception handling


    def add(self, expr):
        """ Adds an expression to the model.

        This method adds one or more CPO expression to the model.
        A CPO expression is an object of class :class:`~docplex.cp.expression.CpoExpr` or derived, obtained by:

         * calling one of the factory method available in module :mod:`docplex.cp.expression`,
         * calling one of the modeling function available in module :mod:`docplex.cp.modeler`,
         * using an overloaded operator with at least one argument that is a :class:`~docplex.cp.expression.CpoExpr` or derived.

        The argument *expr* can be:

         * a constraint,
         * a boolean expression, possibly constant,
         * an objective,
         * a search phase,
         * a variable (but variables that appear in expressions are automatically added to the model),
         * an iterable of expressions to add.

        The order of the expressions that are added to the model is preserved when it is submitted for solving.

        Args:
            expr: CPO expression (constraint, boolean, objective, etc) to add to the model,
                  or iterable of expressions to add to the model.
        Raises:
            CpoException in case of error.
        """
        # Determine calling location
        loc = self._get_calling_location() if self.source_loc else None

        # Check simple expression
        if isinstance(expr, CpoExpr) or is_bool(expr):
            self._add_with_loc(expr, loc)
        else:
            # Argument may be an iterable of expressions
            if is_string(expr):
                raise CpoException("Argument 'expr' should be a CpoExpr or an iterable of CpoExpr")
            # Try as iterable
            try:
                for x in expr:
                    self._add_with_loc(x, loc)
            except:
                raise CpoException("Argument 'expr' should be a CpoExpr or an iterable of CpoExpr")


    def add_constraint(self, expr):
        """ Adds a constraint to the model.

        This method has been added for compatibility with docplex.mp.
        It is equivalent to method :meth:`~CpoModel.add`

        Args:
            expr: Constraint expression to add to the model,
        Raises:
            CpoException in case of error.
        """
        self.add(expr)


    def _add_with_loc(self, expr, loc):
        """ Adds an expression with its location to the model.

        Args:
            expr: CPO expression (constraint, boolean, objective, etc) to add to the model
            loc:  Expression location
        Raises:
            CpoException in case of error.
        """
        #print("Add expression {} at loc {}".format(expr, loc))
        # Update last add time
        self.last_add_time = time.time()

        # Check simple boolean expressions for case where expression built a constant
        if is_bool(expr):
            # Add expression even if false
            self.expr_list.append((CpoValue(expr, Type_Bool), loc))
            return

        # Check type of expression
        etyp = expr.type
        if etyp.is_kind_of(Type_Constraint):
            self.expr_list.append((expr, loc))
        elif etyp is Type_Objective:
            # Check if already added with model function
            if self.objective is not expr:
                # Remove previous objective from the model
                if self.objective is not None:
                    raise CpoException("Only one objective function can be added to the model.")
                self.objective = expr
                self.expr_list.append((expr, loc))
        elif etyp is Type_SearchPhase:
            self.expr_list.append((expr, loc))
        elif isinstance(expr, (CpoVariable, CpoAlias)):
            # Not really useful, just to force variable to be in the model
            self.expr_list.append((expr, loc))
        else:
            raise CpoException("Expression added to the model should be a boolean, constraint, objective or search_phase, not an object of type {}.".format(type(expr)))


    def remove(self, expr):
        """ Remove a single expression from the model.

        This method removes from the model the first occurrence of the expression given as parameter.
        It removes only expressions at the top-level, those added in the model using the method :meth:`~CpoModel.add`,
        it does not remove the expression if it used as sub-expression of another expression.

        If you have multiple expressions to remove, use method :meth:`~CpoModel.remove_expressions` instead.

        Args:
            expr: Expression to remove.
        Returns:
            True if expression has been removed, False if not found
        """
        # Check if it is current objective expression
        if expr is self.objective:
            self.objective = None
        # Remove from list of expressions
        for ix, (x, l) in enumerate(self.expr_list):
            if x is expr:
                del self.expr_list[ix]
                return True
        return False


    def remove_expressions(self, lexpr):
        """ Remove a list of expressions from the model.

        This method removes from the model all occurrences of the expressions given in the list.
        It removes only expressions at the top-level, those added in the model using the method :meth:`~CpoModel.add`,
        it does not remove the expressions that are used as sub-expression of another expression.

        This method is more efficient than calling :meth:`~CpoModel.remove` multiple times.

        Args:
            lexpr: List of expressions to remove from the model.
        Returns:
            Number of expressions actually removed from the model
        """
        # Build a set of ids of expressions to remove
        idset = set(id(x) for x in lexpr)
        # Check if objective is in the expressions to remove
        if id(self.objective) in idset:
            self.objective = None
        # Build a new list of expressions, removing all that are in the list
        nbrem = 0
        nlist = []
        for x in self.expr_list:
            if id(x[0]) in idset:
                nbrem += 1
            else:
                nlist.append(x)
        self.expr_list = nlist
        # Return
        return nbrem


    def minimize(self, expr):
        """ Add an objective expression to minimize.

        DEPRECATED: use add(minimize()) instead.

        Args:
            expr: Expression to minimize.
        Returns:
            Minimization expression that has been added
        """
        # Add new minimization expression
        res = minimize(expr)
        self.add(res)
        return res


    def maximize(self, expr):
        """ Add an objective expression to maximize.

        DEPRECATED: use add(maximize()) instead.

        Args:
            expr: Expression to maximize.
        Returns:
            Maximization expression that has been added
        """
        # Add new maximization expression
        res = maximize(expr)
        self.add(res)
        return res


    def set_parameters(self, params=None, **kwargs):
        """ Set the solving parameters associated to this model.

        The argument *params* can be either:

         * An object of the class :class:`~docplex.cp.parameters.CpoParameters`.
           The parameters object is cloned and replaces the existing parameters associated to the model, if any.

         * A standard Python dictionary where keys are parameter names, and values parameter values.
           In this case, a CpoParameters object is created from the dictionary and then associated to the model.

         * None, to release the parameters of this model.

        If optional named arguments are added to this method, they are considered as additions to the parameters
        given in *params*, that is cloned prior to be modified. If *params* is None, a new
        :class:`~docplex.cp.parameters.CpoParameters` is created.

        Args:
            params (Optional) : Solving parameters, object of class :class:`~docplex.cp.parameters.CpoParameters`,
                                or a dictionary of parameters, or None to remove all parameters.
            **kwargs (Optional): Optional changes to the parameters.
        Returns:
            The new CpoParameters object associated to this model.
        """
        # Check parameters given in params
        if params is None:
            self.parameters = None
            if kwargs:
                self.parameters = CpoParameters(**kwargs)
        elif isinstance(params, CpoParameters):
            self.parameters = params.clone()
            if kwargs:
                self.parameters.add(kwargs)
        elif isinstance(params, dict):
            self.parameters = CpoParameters(**params)
            if kwargs:
                self.parameters.add(**kwargs)
        else:
            raise AssertionError("Argument 'params' should be an object of class CpoParameters, a dictionary, or None.")

        return self.parameters


    def add_parameters(self, **kwargs):
        """ Add parameters to this model.

        This method adds parameters to the :class:`~docplex.cp.parameters.CpoParameters` object currently
        associated to the model.
        If there is no such parameters object yet, a new :class:`~docplex.cp.parameters.CpoParameters` is created.

        Args:
            **kwargs (Optional): List of parameters assignments.
        """
        if kwargs:
            if self.parameters is None:
                self.parameters = CpoParameters()
            for k, v in kwargs.items():
                self.parameters.__setattr__(k, v)


    def get_parameters(self):
        """ Get the solving parameters associated to this model.

        Returns:
            Solving parameters, object of class :class:`~docplex.cp.parameters.CpoParameters`, or None if not defined.
        """
        return self.parameters


    def set_search_phases(self, phases):
        """ Set a list of search phases

        Args:
            phases: Array of search phases, or single phase
        """
        # Check arguments
        if not is_array(phases):
            phases = [phases]

        # Reset list of phases
        self.search_phases = []

        # Add all new phases
        for p in phases:
            assert isinstance(p, CpoExpr) and p.is_type(Type_SearchPhase), "Argument 'phases' should be an array of SearchPhases"
            self.add(p)


    def add_search_phase(self, phase):
        """ Add a search phase to the list of search phases

        This method is deprecated since release 2.3. Use :meth:`~CpoModel.set_search_phases` or
        :meth:`~CpoModel.add` instead.

        Args:
            phase: Phase to add to the list
        """
        warnings.warn("Method 'add_search_phase' is deprecated since release 2.4. Use add() instead.", DeprecationWarning)

        # Check arguments
        assert isinstance(phase, CpoExpr) and phase.is_type(Type_SearchPhase), "Argument 'phase' should be a SearchPhase"

        # Add to model
        self.add(phase)


    def set_starting_point(self, stpoint):
        """ Set a model starting point.

        A starting point specifies a (possibly partial) solution that could be used by CP Optimizer
        to start the search.
        This starting point is represented by an object of class :class:`~docplex.cp.solution.CpoModelSolution`,
        with the following restrictions:

         * Only integer and interval variables are taken into account.
           If present, all other elements are simply ignored.
         * In integer variable, if the domain is not fixed to a single value, only a single range of values is allowed.
           If the variable domain is sparse, the range domain_min..domain_max is used.

        An empty starting point can be created using method :meth:`~CpoModel.create_empty_solution`, and then filled
        using dedicated methods :meth:`~docplex.cp.solution.CpoModelSolution.add_integer_var_solution` and
        :meth:`~docplex.cp.solution.CpoModelSolution.add_interval_var_solution`, or using indexed assignment
        as in the following example:
        ::

            mdl = CpoModel()
            a = integer_var(0, 3)
            b = interval_var(length=(1, 4))
            . . .
            stp = mdl.create_empty_solution()
            stp[a] = 2
            stp[b] = (2, 3, 4)
            mdl.set_starting_point(stp)

        Starting point is available for CPO solver release greater or equal to 12.7.0.

        Args:
            stpoint: Starting point, object of class :class:`~docplex.cp.solution.CpoModelSolution`
        """
        assert (stpoint is None) or isinstance(stpoint, CpoModelSolution), \
            "Argument 'stpoint' should be None or an object of class CpoModelSolution"
        self.starting_point = stpoint


    def get_starting_point(self):
        """ Get the model starting point

        Returns:
            Model starting point, None if none
        """
        return self.starting_point


    def create_empty_solution(self):
        """ Create an empty model solution that can be filled to be used as a starting point.

        *New in version 2.9*

        Returns:
            New empty model solution, object of class :class:`~docplex.cp.solution.CpoModelSolution`
        """
        return CpoModelSolution()


    def add_kpi(self, expr, name=None):
        """ Add a Key Performance Indicator to the model.

        A Key Performance Indicators (KPI) is an expression whose value is considered as representative of the
        model solution and its quality.

        For example, in a scheduling problem one may wish to minimize the makespan
        (date at which all tasks are completed), but other values may be of interest, like the average job
        completion time, or the maximum number of tasks executing in parallel over the horizon.
        One can identify such expressions in the model by marking them as KPIs.

        For CPO solver version lower than 12.9, KPI expressions are limited to:

         * an integer variable,
         * a Python lambda expression that computes the value of the KPI from the solve result given as parameter.

        Example of lambda expression used as KPI:
        ::

            mdl = CpoModel()
            a = integer_var(0, 3)
            b = integer_var(0, 3)
            mdl.add(a < b)
            mdl.add_kpi(lambda res: (res[a] + res[b]) / 2, "Average")

        For CPO solver version greater or equal to 12.9, KPI expressions can be any model expression.
        KPI values are automatically displayed in the log, can be queried after the solve or for each solution,
        and are exported to a CPO file when the model is exported.

        If the model is solved in a cloud context, the KPIs are associated to the objective value in the
        solve details that are sent periodically to the client.

        Args:
            expr:             Model variable to be used as KPI(s).
            name (optional):  Name used to publish this KPI.
                              If absent the expression name is used.
                              If the expression has no name, an exception is raised.
        """
        if isinstance(expr, CpoExpr):
            assert expr.type != Type_Constraint, "KPI expression can not be a top-level constraint"
            # If format version is < 12.9, check integer var only
            if self.format_version is not None and compare_natural(self.format_version, '12.9') < 0:
                assert expr.type == Type_IntVar, "KPI expression can only be an integer variable."
        else:
            assert isinstance(expr, types.FunctionType), "Argument 'expr' should be a model expression or a lambda expression"
        if name is None:
            if isinstance(expr, CpoExpr):
                name = expr.get_name()
        assert name, "A KPI name is mandatory, either as expression name, or as a name given explicitly"
        assert not name in self.kpis, "Name '{}' is already used for another KPI.".format(name)

        # Get expression location
        loc = self._get_calling_location() if self.source_loc else None

        self.kpis[name] = (expr, loc)


    def remove_kpi(self, kpi):
        """ Remove a Key Performance Indicator from the model.

        Args:
            kpi:  KPI expression, or KPI name
        """
        # Check if name given
        if kpi in self.kpis:
            del self.kpis[kpi]
        else:
            # Consider expression has been given
            for k, xl in self.kpis.items():
                if xl[0] is kpi:
                    del self.kpis[k]
                    break


    def remove_all_kpis(self):
        """ Remove all KPIs from this model.
        """
        self.kpis.clear()


    def get_kpis(self):
        """ Returns the dictionary of this model KPIs.

        Returns:
            Ordered dictionary of KPIs.
            Key is publish name, value is kpi as a tuple (expr, loc) where loc is a tuple (source_file, line).
            Keys are sorted in the order the KPIs have been defined.
        """
        return self.kpis


    def _get_kpi_expressions(self):
        """ Returns the list of model expressions used in the kpis

        This method returns all the KPI expressions that are model expressions (lambda expressions are ignored).
        Name of the KPI is absent.

        Returns:
            List of model expressions used as KPIs.
            Each expression is a tuple (expr, loc) where loc is a tuple (source_file, line).
        """
        return [xl for xl in self.kpis.values() if isinstance(xl[0], CpoExpr)]


    def get_all_expressions(self):
        """ Gets the list of all model expressions

        Returns:
            List of model expressions including there location (if any).
            Each expression is a tuple (expr, loc) where loc is a tuple (source_file, line), or None if not set.
        """
        return self.expr_list


    def get_all_variables(self):
        """ Gets the list of all model variables.

        This method goes across all model expressions to identify all variables that are pointed by them.
        Calling this method on a big model may be slow.

        Returns:
            List of model variables.
        """
        # Initialize stack of expressions to parse
        estack = [x for x, l in self.expr_list]
        if self.objective is not None:
            estack.append(self.objective)

        # Loop while expression stack is not empty
        varlist = []     # Result list
        doneset = set()  # Set of expressions already processed
        while estack:
            e = estack.pop()
            eid = id(e)
            if not eid in doneset:
                doneset.add(eid)
                if e.type.is_variable:
                    varlist.append(e)
                # Stack children expressions
                estack.extend(e.children)

        return varlist


    def get_named_expressions_dict(self):
        """ Gets a dictionary of all named expressions.

        This method goes across all model expressions to identify all named expressions.
        Calling this method on a big model may be slow.

        Returns:
            Dictionary of all named expressions. Key is expression name, value is expression.
        """
        # Initialize stack of expressions to parse
        estack = [x for x, l in self.expr_list]
        if self.objective is not None:
            estack.append(self.objective)
        # Loop while expression stack is not empty
        result = {}      # Result dictionary
        doneset = set()  # Set of expressions already processed
        while estack:
            e = estack.pop()
            eid = id(e)
            if not eid in doneset:
                doneset.add(eid)
                if e.name:
                    result[e.name] = e
                # Stack children expressions
                estack.extend(e.children)
        return result


    def get_objective(self):
        """ Gets the objective expression (maximization or minimization).

        Returns:
            Objective expression, None if satisfaction problem.
        """
        return self.objective


    def get_objective_expression(self):
        """ Gets the objective expression (maximization or minimization).

        Returns:
            Objective expression, None if satisfaction problem.
        """
        return self.objective


    def get_optimization_expression(self):
        """ Gets the optimization expression (maximization or minimization).

        DEPRECATED. Use :meth:`~CpoModel.get_objective` instead.

        Returns:
            Optimization expression, None if satisfaction problem.
        """
        return self.get_objective()


    def is_minimization(self):
        """ Check if this model represents a minimization problem.

        Returns:
            True if this model represents a minimization problem.
        """
        return self.objective is not None and "min" in self.objective.operation.cpo_name


    def is_maximization(self):
        """ Check if this model represents a maximization problem.

        Returns:
            True if this model represents a maximization problem.
        """
        return self.objective is not None and "max" in self.objective.operation.cpo_name


    def is_satisfaction(self):
        """ Check if this model represents a satisfaction problem.

        Returns:
            True if this model represents a satisfaction problem.
        """
        return self.objective is None


    def replace_expression(self, oexpr, nexpr):
        """ In all model expressions, replace an expression by another.

        This method goes across all model expressions tree and replace each occurrence of the expression to
        replace by the new expression.
        The comparison of the expression to replace is done by reference (it must be the same object)

        Args:
            oexpr: Expression to replace
            nexpr: Expression to put instead
        Returns:
            Number of replacements done in the model
        """
        # Scan all expressions
        doneset = set()  # Set of expressions already processed
        nbrepl = 0
        for i, (x, l) in enumerate(self.expr_list):
            if x is oexpr:
                self.expr_list[i] = (nexpr, l)
                nbrepl += 1
            elif id(x) not in doneset:
                estack = [x]
                while estack:
                    e = estack.pop()
                    eid = id(e)
                    if eid not in doneset:
                        doneset.add(eid)
                        for cx, c in enumerate(e.children):
                            if c is oexpr:
                                e.children = replace_in_tuple(e.children, cx, nexpr)
                                nbrepl += 1
                            else:
                                estack.append(c)
        return nbrepl


    def get_name(self):
        """ Gets the name of the model.

        If the name is not explicitly defined, the name is the source file name without its extension.
        If source file name is also undefined, name is None.

        Returns:
            Name of the model, None if undefined.
        """
        if self.name is None and self.source_file:
            return utils.get_file_name_only(self.source_file)
        return self.name


    def set_format_version(self, ver):
        """ Set the expected version of the CPO format.

        If the version is None (default), the model is generated with the most recent version.
        If the solver is not the most recent, the model may be rejected at solve time if a recent feature has been used.
        If the version is set, available features are checket at modeling time.

        Args:
            ver:  CPO format version
        """
        self.format_version = str(ver)


    def get_format_version(self):
        """ Gets the version of the CPO format.

        This information is set only when parsing an existing CPO model that contains explicitly a version of the format.
        It is usually not set when creating a new model.
        It can be set explicitly using :meth:`set_format_version` if a specific CPO format is expected.

        Returns:
            String containing the version of the CPO format. None for default.
        """
        return self.format_version


    def get_source_file(self):
        """ Gets the name of the source file from which model has been created.

        Returns:
            Python source file name. None if undefined.
        """
        return self.source_file


    def get_modeling_duration(self):
        """ Get the time spent in modeling.

        The time is computes as difference between the last time an expression has been added
        and the model object creation time.

        Returns:
            Modeling duration in seconds
        """
        return self.last_add_time - self.create_time


    def get_statistics(self):
        """ Get statistics on the model

        This methods compute statistics on the model.

        Returns:
            Model statistics, object of class class :class:`CpoModelStatistics`.
        """
        # Initialize stack of expressions to parse
        estack = [x for x, l in self.expr_list]
        result = CpoModelStatistics(self)
        doneset = set()  # Set of ids of expressions already processed

        # Loop while expression stack is not empty
        while estack:
            e = estack.pop()
            eid = id(e)
            if not eid in doneset:
                doneset.add(eid)
                result._add_expression(e)
                # Stack children expressions
                estack.extend(e.children)

        return result


    def print_information(self, out=None):
        """ Prints model information.

        DEPRECATED. Use :meth:`write_information` instead.

        Args:
            out: Output stream or file name, default is sys.stdout.
        """
        self.write_information(out)


    def write_information(self, out=None):
        """ Write various information about the model.

        This method calls the method :meth:`get_statistics` to retrieve information on the model, and then
        print it with source file name and modeling time.

        Args:
            out: Output stream or file name, default is sys.stdout.
        """
        # Check output
        if is_string(out):
            with open(os.path.abspath(out), mode='w') as f:
                self.write_information(f)
            return

        if out is None:
            out = sys.stdout

        # Print information
        name = self.get_name()
        out.write("Model: {}\n".format(name if name else "Anonymous"))
        sfile = self.get_source_file()
        if sfile:
            out.write(" - source file: {}\n".format(sfile))
        out.write(" - modeling time: {0:.2f} sec\n".format(self.get_modeling_duration()))
        stats = self.get_statistics()
        stats.write(out, " - ")


    def create_solver(self, **kwargs):
        """ Create a new solver instance attached to this model

        All necessary solving parameters are taken from the solving context that is constructed from the following list
        of sources, each one overwriting the previous:

           - the parameters that are set in the model itself,
           - the default solving context that is defined in the module :mod:`~docplex.cp.config`
           - the user-specific customizations of the context that may be defined (see :mod:`~docplex.cp.config` for details),
           - the optional arguments of this method.

        Args:
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            New solver properly initialized.
        """
        slvr = CpoSolver(self, **kwargs)
        # Add solvers listeners
        for l in self.listeners:
            slvr.add_listener(l)
        # Add solvers callbacks
        for l in self.callbacks:
            slvr.add_callback(l)
        return slvr


    def solve(self, **kwargs):
        """ Solves the model.

        This method solves the model using the appropriate :class:`~docplex.cp.solver.solver.CpoSolver`
        created according to default solving context, possibly modified by the parameters of this method.

        The class :class:`~docplex.cp.solver.solver.CpoSolver` contains the actual implementation of this method,
        but also some others functions allowing to invoke more specialized functions.
        An advanced programming may require to explicitly create a CpoSolver instead of calling function at model level.
        Please refer to this class for more details.

        All necessary solving parameters are taken from the solving context that is constructed from the following list
        of sources, each one overwriting the previous:

           - the parameters that are set in the model itself,
           - the default solving context that is defined in the module :mod:`~docplex.cp.config`
           - the user-specific customizations of the context that may be defined (see :mod:`~docplex.cp.config` for details),
           - the optional arguments of this method.

        If an optional argument other than `context` or `params` is given to this method, it is searched in the
        context where its value is replaced by the new one.
        If not found, it is then considered as a solver parameter.
        In this case, only public parameters are allowed, except if the context attribute `solver.enable_undocumented_params`
        is set to True. This can be done directly when calling this method, as for example:
        ::

            mdl.solve(enable_undocumented_params=True, MyPrivateParam=MyValue)

        Args:
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            Model solve result (object of class :class:`~docplex.cp.solution.CpoSolveResult`).
        Raises:
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        solver = self.create_solver(**kwargs)
        msol = solver.solve()
        solver.end()
        return msol


    def start_search(self, **kwargs):
        """ Start a new search sequence to retrieve multiple solutions of the model.

        This method returns a new :class:`~docplex.cp.solver.solver.CpoSolver` object
        that acts as an iterator of the different solutions of the model.
        All solutions can be retrieved using a loop like:
        ::

           lsols = mdl.start_search()
           for sol in lsols:
               sol.write()

        A such solution iteration can be interrupted at any time by calling :meth:`~docplex.cp.solver.solver.CpoSolver.end_search`
        that returns a fail solution including the last solve status.

        Note that, to be sure to retrieve all solutions and only once each,
        recommended parameters are *start_search(SearchType='DepthFirst', Workers=1)*

        Optional arguments are the same than those available in the method :meth:`solve`

        Args:
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            Object of class :class:`~docplex.cp.solver.solver.CpoSolver` allowing to iterate over the different solutions.
        Raises:
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        solver = self.create_solver(**kwargs)
        return solver


    def refine_conflict(self, **kwargs):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        Given an infeasible model, the conflict refiner can identify conflicting constraints and variable domains
        within the model to help you identify the causes of the infeasibility.
        In this context, a conflict is a subset of the constraints and/or variable domains of the model
        which are mutually contradictory.
        Since the conflict is minimal, removal of any one of these constraints will remove that
        particular cause for infeasibility.
        There may be other conflicts in the model; consequently, repair of a given conflict
        does not guarantee feasibility of the remaining model.

        Conflict refiner is controlled by the following parameters, that can be set as parameters of this method:

         * ConflictRefinerBranchLimit
         * ConflictRefinerFailLimit
         * ConflictRefinerIterationLimit
         * ConflictRefinerOnVariables
         * ConflictRefinerTimeLimit

        that are described in module :mod:`docplex.cp.parameters`.

        Note that the general *TimeLimit* parameter is used as a limiter for each conflict refiner iteration, but the
        global limitation in time must be set using *ConflictRefinerTimeLimit* that is infinite by default.

        This method creates a new :class:`~docplex.cp.solver.solver.CpoSolver` with given arguments, and then call
        its method :meth:`~docplex.cp.solver.solver.CpoSolver.refine_conflict`.
        The class :class:`~docplex.cp.solver.solver.CpoSolver` contains the actual implementation of this method,
        but also some others functions allowing to invoke more specialized functions. An advanced programming may
        require to explicitly create a CpoSolver instead of calling function at model level.
        Please refer to this class for more details.

        This function is available on DOcplexcloud and with local CPO solver with release number greater or equal to 12.7.0.

        Args:
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            List of constraints that cause the conflict (object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`)
        Raises:
            :class:`~docplex.cp.utils.CpoNotSupportedException`: if method not available in the solver agent.
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        solver = self.create_solver(**kwargs)
        rsol = solver.refine_conflict()
        solver.end()
        return rsol


    def propagate(self, cnstr=None, **kwargs):
        """ This method invokes the propagation on the current model.

        Constraint propagation is the process of communicating the domain reduction of a decision variable to
        all of the constraints that are stated over this variable.
        This process can result in more domain reductions.
        These domain reductions, in turn, are communicated to the appropriate constraints.
        This process continues until no more variable domains can be reduced or when a domain becomes empty
        and a failure occurs.
        An empty domain during the initial constraint propagation means that the model has no solution.

        The result is a object of class :class:`~docplex.cp.solution.CpoSolveResult`, the same than the one
        returned by the method :meth:`solve`.
        However, variable domains may not be completely defined.

        This method creates a new :class:`~docplex.cp.solver.solver.CpoSolver` with given arguments, and then call
        its method :meth:`~docplex.cp.solver.solver.CpoSolver.propagate`.
        The class :class:`~docplex.cp.solver.solver.CpoSolver` contains the actual implementation of this method,
        but also some others functions allowing to invoke more specialized functions. An advanced programming may
        require to explicitly create a CpoSolver instead of calling function at model level.
        Please refer to this class for more details.

        This function is available on DOcplexcloud and with local CPO solver with release number greater or equal to 12.7.0.

        Args:
            cnstr (Optional):   Optional constraint to be added to the model before invoking propagation.
                                If not given, solving context is the default one that is defined in the module
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            Propagation result (object of class :class:`~docplex.cp.solution.CpoSolveResult`)
        Raises:
            :class:`~docplex.cp.utils.CpoNotSupportedException`: if method not available in the solver agent.
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        # Check if an optional constraint has been given
        if cnstr is None:
            mdl = self
        else:
            # Clone the model and add constraint
            mdl = self.clone()
            mdl.add(cnstr)
        # Call propagation
        solver = mdl.create_solver(**kwargs)
        psol = solver.propagate()
        solver.end()
        return psol


    def run_seeds(self, nbrun, **kwargs):
        """ This method runs *nbrun* times the CP optimizer search with different random seeds
        and computes statistics from the result of these runs.

        Result statistics are displayed on the log output that should be activated.
        If the appropriate configuration variable *context.solver.add_log_to_solution* is set to True (default),
        log is also available in the *CpoRunResult* result object, accessible as a string using the method
        :meth:`~docplex.cp.solution.CpoRunResult.get_solver_log`

        Each run of the solver is stopped according to single solve conditions (TimeLimit for example).
        Total run time is then expected to take *nbruns* times the duration of a single run.

        This function is available only with local CPO solver with release number greater or equal to 12.8.

        Args:
            nbrun:              Number of runs with different seeds.
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            (param) (Optional): Any individual solving parameter as defined in class :class:`~docplex.cp.parameters.CpoParameters`
                               (for example *TimeLimit*, *Workers*, *SearchType*, etc).
            (others) (Optional): Any leaf attribute with the same name in the solving context
                                (for example *agent*, *trace_log*, *trace_cpo*, etc).
        Returns:
            Run result, object of class :class:`~docplex.cp.solution.CpoRunResult`.
        Raises:
            :class:`~docplex.cp.utils.CpoNotSupportedException`: if method not available in the solver agent.
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        solver = self.create_solver(**kwargs)
        rsol = solver.run_seeds(nbrun)
        solver.end()
        return rsol


    def explain_failure(self, ltags=None, **kwargs):
        """ This method allows to explain solve failures.

        If called with no arguments, this method invokes a solve of the model with appropriate parameters
        that enable, in the log, the print of a number tag for each solve failure.

        If called with a list of failure tag to explain, the solver is invoked again in a way that it explains,
        in the log, the reason for the failure of the required failure tags.

        This method sets the following solve parameters before calling the solver:

         * :attr:`~docplex.cp.CpoParameters.LogSearchTags` = 'On'
         * :attr:`~docplex.cp.CpoParameters.Workers` = 1
         * :attr:`~docplex.cp.CpoParameters.LogPeriod` = 1
         * :attr:`~docplex.cp.CpoParameters.SearchType` = 'DepthFirst'

        Args:
            ltags (Optional) :   List of tag ids to explain. If empty or None, the solver is just invoked with
                                 appropriate solve parameters to make failure tags displayed in the log.
            (others) (Optional): Any other solve attribute as it can be passed to method :meth:`~CpoModel.solve`.
        Returns:
            Solve result, object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            :class:`~docplex.cp.utils.CpoNotSupportedException`: if method not available in the solver agent.
            :class:`~docplex.cp.utils.CpoException`: (or derived) if error.
        """
        solver = self.create_solver(**kwargs)
        msol = solver.explain_failure(ltags)
        solver.end()
        return msol


    def add_solver_listener(self, lstnr):
        """ Add a solver listener.

        A solver listener is an object extending the class :class:`~docplex.cp.solver.solver_listener.CpoSolverListener`
        which provides multiple functions that are called to notify about the different solving steps.

        Args:
            lstnr:  Solver listener
        """
        assert isinstance(lstnr, CpoSolverListener), \
            "Listener should be an object of class docplex.cp.solver.solver_listener.CpoSolverListener"
        self.listeners.append(lstnr)


    def remove_solver_listener(self, lstnr):
        """ Remove a solver listener previously added with :meth:`~docplex.cp.model.CpoModel.add_listener`.

        Args:
            lstnr:  Listener to remove.
        """
        self.listeners.remove(lstnr)


    def add_solver_callback(self, cback):
        """ Add a CPO solver callback.

        A solver callback is an object extending the class :class:`~docplex.cp.solver.cpo_callback.CpoCallback`
        which provides multiple functions that are called to notify about the different solving steps.

        Args:
            cback:  Solver callback, object extending :class:`~docplex.cp.solver.cpo_callback.CpoCallback`
        """
        assert isinstance(cback, CpoCallback), \
            "CPO callback should be an object of class docplex.cp.solver.cpo_callback.CpoCallback"
        self.callbacks.append(cback)


    def remove_solver_callback(self, cback):
        """ Remove a CPO solver callback. previously added with :meth:`~docplex.cp.solver.solver.CpoSolver.add_callback`.

        Args:
            cback:  Callback to remove.
        """
        self.callbacks.remove(cback)


    def export_model(self, out=None, **kwargs):
        """ Exports/prints the model in the standard CPO file format.

        Note that calling this method disables automatically all the settings that are set in the default configuration
        to change the format of the model:

         * *context.model.length_for_alias* that rename variables if name is too long,
         * *context.model.name_all_constraints* that force a name for each constraint.

        These options are however possible if explicitly given as parameter of this method, as in:
        ::

           mdl.export_model(length_for_alias=10)

        Args:
            out (Optional):     Target output, stream or file name. Default is sys.stdout.
            context (Optional): Complete solving context.
                                If not given, solving context is the default one that is defined in the module
                                :mod:`~docplex.cp.config`.
            params (Optional):  Solving parameters (object of class :class:`~docplex.cp.parameters.CpoParameters`)
                                that overwrite those in the solving context.
            add_source_location (Optional): Add source location into generated text
            length_for_alias (Optional): Minimum name length to use shorter alias instead
            (others) (Optional): Any leaf attribute with the same name in the solving context
        """
        # Remove all code transformations but respect those provided explicitly
        kwargs.setdefault('length_for_alias', None)
        kwargs.setdefault('name_all_constraints', False)

        CpoCompiler(self, **kwargs).write(out)


    def import_model(self, file):
        """ Import a model from a file containing a model expressed in CPO, FZN or LP format.

        FZN and LP formats are supported with restrictions to integer variables.
        The full list of supported FZN predicates is given in the documentation of module
        :mod:`~docplex.cp.fzn.fzn_parser`.

        Source files can also be provided compressed, in zip or gzip format.

        Args:
            file: Name of the input file, with extension ".cpo", ".fzn" or ".lp", optionally followed by ".gz" or ".zip"
        """
        felems = os.path.splitext(file)
        ext = felems[1].lower()
        if ext in (".gz", ".zip",):
            ext = os.path.splitext(felems[0])[1].lower()

        if ext == ".cpo":
            import docplex.cp.cpo.cpo_parser as cpo_parser
            prs = cpo_parser.CpoParser(self)
            prs.parse(file)

        elif ext == ".fzn":
            import docplex.cp.fzn.fzn_parser as fzn_parser
            prs = fzn_parser.FznParser(self)
            prs.parse(file)
            # Get model to force compilation
            prs.get_model()

        elif ext == ".lp":
            import docplex.cp.lp.lp_parser as lp_parser
            prs = lp_parser.LpParser(self)
            prs.parse(file)

        else:
            raise CpoException("Unknown '{}' file format. Only .cpo, .fzn and .lp are supported.".format(ext))


    def export_as_cpo(self, out=None, **kwargs):
        """ Deprecated form of method :meth:`export_model`.
        """
        self.export_model(out, **kwargs)


    def get_cpo_string(self, **kwargs):
        """ Compiles the model in CPO file format into a string.

        Note that calling this method disables automatically all the settings that are set in the default configuration
        to change the format of the model:

         * *context.model.length_for_alias* that rename variables if name is too long,
         * *context.model.name_all_constraints* that force a name for each constraint.

        These options are however possible if explicitly given as parameter of this method, as in:
        ::

           mstr = mdl.get_cpo_string(length_for_alias=10)

        Args:
            context:             Global solving context. If not given, context is the default context that is set in config.py.
            params:              Solving parameters (CpoParameters) that overwrites those in solving context
            add_source_location: Add source location into generated text
            length_for_alias:    Minimum name length to use shorter alias instead
            (others):            All other context parameters that can be changed
        Returns:
            String containing the model.
        """
        # Remove all code transformations but respect those provided explicitly
        kwargs.setdefault('length_for_alias', None)
        kwargs.setdefault('name_all_constraints', False)

        return CpoCompiler(self, **kwargs).get_as_string()


    def check_equivalence(self, other):
        """ Checks that this model is equivalent to another.

        Variables and expressions are compared, but not names that may differ because of automatic naming.

        Args:
            other:  Other model to compare with.
        Raises:
            Exception if models are not equivalent
        """

        # Check object types
        if not isinstance(other, CpoModel):
            raise CpoException("Other model is not an object of class CpoModel")

        # Compare expressions that are not variables
        lx1 = [x for x, l in self.expr_list if not isinstance(x, (CpoVariable, CpoValue, CpoAlias, CpoFunction))]
        lx2 = [x for x, l in other.expr_list if not isinstance(x, (CpoVariable, CpoValue, CpoAlias, CpoFunction))]
        if len(lx1) != len(lx2):
            raise CpoException("Different number of expressions, {} vs {}.".format(len(lx1), len(lx2)))
        for i, (x1, x2) in enumerate(zip(lx1, lx2)):
            #print("Check expression {}\n   and\n{}".format(lx1[i], lx2[i]))
            if not x1.equals(x2):
                print("X1 = {}".format(x1))
                print("X2 = {}".format(x2))
                raise CpoException("The expression {} differs: '{}' vs '{}'".format(i, x1, x2))


    def equals(self, other):
        """ Checks if this model is equal to another.

        Args:
            other:  Other model to compare with.
        Returns:
            True if models are identical, False otherwise.
        """
        # Check object types
        if not isinstance(other, CpoModel):
            return False
        # Do not compare variables as there may me more with Python as all are named (for example SequenceVar)
        # Check list of expressions (will also compare variables)
        if len(self.expr_list) != len(other.expr_list):
            return False
        for x1, x2 in zip(self.expr_list, other.expr_list):
            if not x1[0].equals(x2[0]):
                # print("different expressions: \n1: {}\n2: {}".format(x1[0], x2[0]))
                return False
        return True


    def clone(self):
        """ Create a copy of this model.

        Result copy duplicates only the attributes of the model and the list of expressions.
        It does not create a deep copy of the expressions.
        """
        res = copy.copy(self)
        res.expr_list = list(self.expr_list)
        if self.parameters is not None:
            res.parameters = self.parameters.copy()
        return res


    def __eq__(self, other):
        """ Check if this model is equal to another

        Args:
            other:  Other model to compare with
        Returns:
            True if models are identical, False otherwise
        """
        return self.equals(other)


    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


    def __str__(self):
        """ Convert the model into string (returns model name) """
        return self.get_name()


    def _search_named_expression(self, name):
        """ Search in the model the first expression whose name is the given one.

        This method goes across all model expressions to search for named expression.
        Calling this method on a big model may be slow.

        Args:
            name:  Name of the expression to search
        Returns:
            Expression, None if not found
        """
        # Initialize stack of expressions to parse
        estack = [x for x, l in self.expr_list]
        if self.objective is not None:
            estack.append(self.objective)
        # Loop while expression stack is not empty
        doneset = set()  # Set of expressions already processed
        while estack:
            e = estack.pop()
            eid = id(e)
            if not eid in doneset:
                if e.name == name:
                    return e
                # Stack children expressions
                doneset.add(eid)
                estack.extend(e.children)
        return None


    def _get_calling_location(self):
        """ Determine the calling location, outside docplex.cp
        Returns:
             Couple (file, line), or None if impossible to determine
        """
        frm = inspect.currentframe()
        # Skip at least 2 frames (first called method + this one)
        if frm is None:
            return None
        frm = frm.f_back
        if frm is None:
            return None
        frm = frm.f_back
        # Loop while still in the docplex.cp package
        while frm:
            fname = frm.f_code.co_filename
            if _THIS_FILE_MARKER not in fname:
                return (fname, frm.f_lineno)
            frm = frm.f_back
        return None


###############################################################################
##  Private Functions
###############################################################################

def _set_all_modeling_functions(trgt):
    """ Copy modeling function in the given target object

    Args:
        trgt: Target object
    """

    # Duplicate constructor functions to make them callable from the model
    trgt.integer_var       = expression.integer_var
    trgt.integer_var_list  = expression.integer_var_list
    trgt.integer_var_dict  = expression.integer_var_dict
    trgt.binary_var        = expression.binary_var
    trgt.binary_var_list   = expression.binary_var_list
    trgt.binary_var_dict   = expression.binary_var_dict
    trgt.interval_var      = expression.interval_var
    trgt.interval_var_list = expression.interval_var_list
    trgt.interval_var_dict = expression.interval_var_dict
    trgt.sequence_var      = expression.sequence_var
    trgt.transition_matrix = expression.transition_matrix
    trgt.tuple_set         = expression.tuple_set
    trgt.state_function    = expression.state_function
    trgt.float_var         = expression.float_var

    # Copy all modeler public functions in the model object
    for f in list_module_public_functions(modeler, ('maximize', 'minimize')):
        setattr(trgt, f.__name__, f)

    # Special case for builtin functions
    trgt.min = modeler.min_of
    trgt.max = modeler.max_of
    trgt.sum = modeler.sum_of
    trgt.abs = modeler.abs_of
    trgt.range = modeler.in_range
    trgt.all = modeler.all_of
    trgt.any = modeler.any_of


# Set all modeling functions to the CpoModel class
#_set_all_modeling_functions(CpoModel)
