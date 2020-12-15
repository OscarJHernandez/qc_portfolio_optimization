# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from __future__ import print_function

import sys
import copy

import six

try:  # pragma: no cover
    from itertools import zip_longest as izip_longest
except ImportError:  # pragma: no cover
    # noinspection PyUnresolvedReferences
    from itertools import izip_longest

from six import iteritems, iterkeys

from docplex.mp.compat23 import StringIO, izip
from docplex.mp.constants import CplexScope, BasisStatus, WriteLevel
from docplex.mp.utils import is_iterable, is_number, is_string, is_indexable, str_maxed, normalize_basename
from docplex.mp.utils import make_output_path2
from docplex.mp.linear import Var
from docplex.mp.error_handler import docplex_fatal

from docplex.mp.solmst import SolutionMSTPrinter
from docplex.mp.soljson import SolutionJSONPrinter

from collections import defaultdict


# noinspection PyAttributeOutsideInit
class SolveSolution(object):
    """
    The :class:`SolveSolution` class holds the result of a solve.
    """

    # a symbolic value for no objective ?
    NO_OBJECTIVE_VALUE = -1e+75

    @staticmethod
    def _is_discrete_value(v):
        return v == int(v)

    def __init__(self, model, var_value_map=None, obj=None, blended_obj_by_priority=None, name=None, solved_by=None,
                 keep_zeros=True):
        """ SolveSolution(model, var_value_map, obj, name)

        Creates a new solution object, associated to a a model.

        Args:
            model: The model to which the solution is associated. This model cannot be changed.

            obj: The value of the objective in the solution. A value of None means the objective is not defined at the
                time the solution is created, and will be set later.

            blended_obj_by_priority: For multi-objective models: the value of sub-problems' objectives (each sub-problem
                groups objectives having same priority).

            var_value_map: a Python dictionary containing associations of variables to values.

            name: a name for the solution. The default is None, in which case the solution is named after the
                model name.

        :return: A solution object.
        """
        assert model is not None
        assert solved_by is None or is_string(solved_by)
        assert obj is None or is_number(obj) or is_indexable(obj)
        assert blended_obj_by_priority is None or is_indexable(blended_obj_by_priority)

        self._model = model
        self._name = name
        self._problem_objective_expr = model.objective_expr if model.has_objective() else None
        self._objective = self.NO_OBJECTIVE_VALUE if obj is None else obj
        self._blended_objective_by_priority = [self.NO_OBJECTIVE_VALUE] if blended_obj_by_priority is None else \
            blended_obj_by_priority
        self._solved_by = solved_by
        self._var_value_map = {}

        # attributes
        self._reduced_costs = None
        self._dual_values = None
        self._slack_values = None
        self._infeasibilities = {}
        self._basis_statuses = None

        self._solve_status = None
        self._keep_zeros = keep_zeros
        self._solve_details = None

        if var_value_map is not None:
            self._store_var_value_map(var_value_map, keep_zeros=keep_zeros)

    @property
    def _checker(self):
        return self._model._checker

    @staticmethod
    def make_engine_solution(model, var_value_map, obj, blended_obj_by_priority, solved_by, solve_details,
                             job_solve_status=None):
        # INTERNAL
        # noinspection PyArgumentEqualDefault
        sol = SolveSolution(model,
                            var_value_map=None,
                            obj=obj,
                            blended_obj_by_priority=blended_obj_by_priority,
                            solved_by=solved_by,
                            keep_zeros=False)
        if solve_details is not None:
            sol._solve_details = copy.copy(solve_details)

        if model.round_solution:
            # only for models which specify round_solution
            roundfn = sol._model._round_function
            for dvar, value in iteritems(var_value_map):
                if value and dvar.is_discrete() and value != int(value):
                    var_value_map[dvar] = roundfn(value)
        # do trust engines...
        sol._var_value_map = var_value_map

        if job_solve_status is not None:
            sol._set_solve_status(job_solve_status)
        return sol

    def _get_var_by_name(self, varname):
        return self._model.get_var_by_name(varname)

    def as_mip_start(self):
        # INTERNAL
        mipstart = SolveSolution(self.model, name=self.name,
                                 var_value_map=None,
                                 obj=self.objective_value,
                                 solved_by=self.solved_by,
                                 keep_zeros=self._keep_zeros)
        # copy value dict
        mipstart._var_value_map = self._var_value_map.copy()
        if self.solved_by and not self._keep_zeros:
            # completion: add all discrete, non-generated:
            for dvi in self.model.generate_user_variables():
                if dvi.is_discrete() and dvi not in self._var_value_map:
                    mipstart._set_var_value_internal(dvi, 0)
        return mipstart

    def clear(self):
        """ Clears all solve result data.

        All data related to the model are left unchanged.
        """
        self._var_value_map = {}
        self._objective = self.NO_OBJECTIVE_VALUE
        self._reduced_costs = None
        self._dual_values = None
        self._slack_values = None
        self._infeasibilities = {}
        self._solve_status = None

    def is_empty(self):
        """
        Checks whether the solution is empty.

        Returns:
            Boolean: True if the solution is empty; in other words, the solution has no defined objective and no variable value.
        """
        return not self.has_objective() and not self._var_value_map

    @property
    def problem_name(self):
        return self._model.name

    @property
    def solved_by(self):
        '''
        Returns a string indicating how the solution was produced.

        - If the solution was created by a program, this field returns None.
        - If the solution originated from a local CPLEX solve, this method returns the string 'cplex_local'.
        - If the solution originated from a DOcplexcloud solve, this method returns 'cplex_cloud'.

        Returns:
            A string, or None.

        '''
        return self._solved_by

    def get_name(self):
        return self._name

    def set_name(self, solution_name):
        self._checker.typecheck_string(solution_name, accept_empty=False, accept_none=True,
                                       caller='SolveSolution.set_name(): ')
        self._name = solution_name

    @property
    def name(self):
        """ This property allows to get/set a name on the solution.

        In some cases , it might be interesting to build different solutions for the same model,
        in this case, use the name property to distinguish them.

        """
        return self._name

    @name.setter
    def name(self, sol_name):
        self.set_name(sol_name)

    def _resolve_var(self, var_key, do_raise):
        # INTERNAL: accepts either strings or variable objects
        # returns a variable or None
        if isinstance(var_key, Var):
            return var_key
        elif is_string(var_key):
            var = self._get_var_by_name(var_key)
            # var might be None here if the name is unknown
            if var is not None:
                return var
            # var is None hereafter
            elif do_raise:
                self.model.fatal("No variable with named {0}", var_key)
            else:
                self.model.warning("No variable with named {0}", var_key)
                return None

        else:  # pragma: no cover
            self.model.fatal("Expecting variable or name, got: {0!r}", var_key)

    def _typecheck_var_key_value(self, var_key, value, caller):
        # INTERNAL
        self._checker.typecheck_num(value, caller=caller)
        if not is_string(var_key) and not isinstance(var_key, Var):
            self.model.fatal("{0} expects either Var or string, got: {1!r}", caller, var_key)

    def add_var_value(self, var_key, value):
        """ Adds a new (variable, value) pair to this solution.

        Args:
            var_key: A decision variable (:class:`docplex.mp.linear.Var`) or a variable name (string).
            value (number): The value of the variable in the solution.
        """
        self._typecheck_var_key_value(var_key, value, caller="Solution.add_var_value")
        self._set_var_key_value(var_key, value, keep_zero=self._keep_zeros)

    def __setitem__(self, var_key, value):
        # always keep zero, no warnings, no checks
        self._set_var_key_value(var_key, value, keep_zero=self._keep_zeros)

    def set_var_key_value(self, var_key, value, keep_zero):
        # INTERNAL
        self._typecheck_var_key_value(var_key, value, caller="Solution.add_var_value")
        self._set_var_key_value(var_key, value, keep_zero)

    def _set_var_key_value(self, var_key, value, keep_zero):
        # INTERNAL: no checks done.
        dvar = self._resolve_var(var_key, do_raise=False)
        if dvar is not None:
            if value or keep_zero:
                # either value is nonzero or we keep all, store.
                self._set_var_value_internal(dvar, value)
            elif self.contains(dvar):
                # value is 0 and we dont keep zeros: zap the variable, if
                del self._var_value_map[dvar]

    def _set_var_value_internal(self, var, value):
        self._var_value_map[var] = value

    def _set_var_value(self, var, value):
        # INTERNAL
        self._set_var_value_internal(var, value)

    def update(self, var_values_iterable):
        """
        Updates the solution from a dictionary. Keys can be either strings, interpreted as variable names,
        or variables; values are the new values for the variable.

        This method returns nothing, only performs a side effect on the solution object.

        :param var_values_iterable: a dictionary of keys, values.

        """
        keep_zeros = self._keep_zeros
        for k, v in iteritems(var_values_iterable):
            self._set_var_key_value(k, v, keep_zeros)

    @property
    def model(self):
        """
        This property returns the model associated with the solution.
        """
        return self._model

    @property
    def solve_details(self):
        """ This property returns the solve_details associated with the solution,if any.

        Note:
            This property returns an instance of solve details if the solution is the result
            of a solve operation. If the solution has been created by API, this property returns None

        See Also:
            :class:`docplex.mp.sdetails.SolveDetails`

        Returns:
            an instance of SolveDetails, or None.

        """
        return self._solve_details

    # @property
    # def error_handler(self):
    #     return self.__model.error_handler

    def get_objective_value(self):
        """
        Gets the objective value (or list of objectives value) as defined in the solution.
        When the objective value has not been defined, a special value `NO_SOLUTION` is returned.
        To check whether the objective has been set, use :func:`has_objective`.

        Returns:
            float or list(float): The value of the objective (or list of values for multi-objective) as defined by
            the solution.
        """
        return self._objective

    def set_objective_value(self, obj):
        """
        Sets the objective value (or list of values for multi-objective) of the solution.
        
        Args:
            obj (float or list(float)): The value of the objective (or list of values for multi-objective) in
            the solution.
        """
        self._objective = obj

    def get_blended_objective_value_by_priority(self):
        """
        Gets the blended objective value (or list of blended objectives value) by priority level as defined in
        the solution.
        When the objective value has not been defined, a special value `NO_SOLUTION` is returned.
        To check whether the objective has been set, use :func:`has_objective`.

        Returns:
            float or list(float): The value of the objective (or list of values for multi-objective) as defined by
            the solution.
        """
        return self._blended_objective_by_priority

    @property
    def blended_objective_values(self):
        return self._blended_objective_by_priority

    def has_objective(self):
        """
        Checks whether or not the objective has been set.

        Returns:
            Boolean: True if the solution defines an objective value.
        """
        return self._objective != self.NO_OBJECTIVE_VALUE

    @property
    def objective_value(self):
        """ This property is used to get the objective value of the solution.
        In case of multi-objective this property returns the value for the first objective

        When the objective value has not been defined, a special value `NO_SOLUTION` is returned.
        To check whether the objective has been set, use :func:`has_objective`.

        """
        try:
            return self._objective[0]
        except TypeError:
            return self._objective

    @objective_value.setter
    def objective_value(self, new_objvalue):
        self.set_objective_value(new_objvalue)

    @property
    def multi_objective_values(self):
        """ This property is used to get the list of objective values of the solution.
        In case of single objective this property returns the value for the objective as a singleton list

        When the objective value has not been defined, a special value `NO_SOLUTION` is returned.
        To check whether the objective has been set, use :func:`has_objective`.

        """
        self_obj = self._objective
        return self_obj if is_indexable(self_obj) else [self_obj]

    @property
    def solve_status(self):
        return self._solve_status

    def _set_solve_status(self, new_status):
        # INTERNAL
        self._solve_status = new_status

    def _store_var_value_map(self, key_value_map, keep_zeros=False):
        # INTERNAL
        for e, val in iteritems(key_value_map):
            # need to check var_keys and values
            self.set_var_key_value(var_key=e, value=val, keep_zero=keep_zeros)

    def store_infeasibilities(self, infeasibilities):
        assert isinstance(infeasibilities, dict)
        self._infeasibilities = infeasibilities

    @staticmethod
    def _resolve_attribute_index_map(attr_idx_map, obj_mapper):
        return {obj_mapper(idx): attr_val
                for idx, attr_val in iteritems(attr_idx_map)
                if attr_val and obj_mapper(idx) is not None}

    @classmethod
    def _resolve_attribute_list(cls, attr_list, obj_mapper):
        # attr list is a list of length N and obj_mapper maps indices to objs
        return {obj_mapper(idx): attr_val for idx, attr_val in enumerate(attr_list)}

    def store_reduced_costs(self, rcs, mapper):
        self._reduced_costs = self._resolve_attribute_index_map(rcs, mapper)

    def store_dual_values(self, duals, mapper):
        self._dual_values = self._resolve_attribute_index_map(duals, mapper)

    def store_slack_values(self, slacks, mapper):
        resolved_linear_slacks = self._resolve_attribute_index_map(slacks, mapper)
        self._slack_values = defaultdict(dict)
        self._slack_values[CplexScope.LINEAR_CT_SCOPE] = resolved_linear_slacks

    def store_attribute_lists(self, mdl, slacks):
        def linct_mapper(idx):
            return mdl.get_constraint_by_index(idx)
        resolved_linear_slacks = self._resolve_attribute_list(slacks, linct_mapper)
        self._slack_values = defaultdict(dict)
        self._slack_values[CplexScope.LINEAR_CT_SCOPE] = resolved_linear_slacks

    def iter_var_values(self):
        """Iterates over the (variable, value) pairs in the solution.

        Returns:
            iterator: A dict-style iterator which returns a two-component tuple (variable, value)
            for all variables mentioned in the solution.
        """
        return iteritems(self._var_value_map)

    iteritems = iter_var_values

    def iter_variables(self):
        """Iterates over all variables mentioned in the solution.

        Returns:
           iterator: An iterator object over all variables mentioned in the solution.
        """
        return iterkeys(self._var_value_map)

    def contains(self, dvar):
        """
        Checks whether or not a decision variable is mentioned in the solution.

        This predicate can also be used in the form `var in solution`, because the
        :func:`__contains_` method has been redefined for this purpose.

        Args:
            dvar (:class:`docplex.mp.linear.Var`): The variable to check.

        Returns:
            Boolean: True if the variable is mentioned in the solution.
        """
        return dvar in self._var_value_map

    def __contains__(self, dvar):
        return self.contains(dvar)

    def get_value(self, arg):
        """
        Gets the value of a variable or an expression in a solution.
        If the variable is not mentioned in the solution,
        the method returns 0 and does not raise an exception.
        Note that this method can also be used as :func:`solution[arg]`
        because the :func:`__getitem__` method has been overloaded.

        Args:
            arg: A decision variable (:class:`docplex.mp.linear.Var`),
                 a variable name (a string), or an expression.

        Returns:
            float: The value of the variable in the solution.
        """
        if is_string(arg):
            var = self._get_var_by_name(arg)
            if var is None:
                self.model.fatal("No variable with name: {0}", arg)
            else:
                return self._get_var_value(var)
        elif isinstance(arg, Var):
            return self._get_var_value(arg)
        else:
            try:
                v = arg._get_solution_value(self)
                return v
            except AttributeError:
                self._model.fatal("Expecting variable, variable name or expression, {0!r} was passed", arg)

    def get_var_value(self, dvar):
        self._checker.typecheck_var(dvar)
        return self._get_var_value(dvar)

    def _get_var_value(self, dvar):
        # INTERNAL
        return self._var_value_map.get(dvar, 0)

    def get_value_list(self, dvars):
        """
        Gets the value of a sequence of variables in a solution.
        If a variable is not mentioned in the solution,
        the method assumes 0 and does not raise an exception.

        Args:
            dvars: an ordered sequence of decision variables.

        Returns:
            list: A list of float values, in the same order as the variable sequence.

        """
        checker = self._checker
        checker.check_ordered_sequence(arg=dvars,
                                       caller='SolveSolution.get_values() expects ordered sequence of variables')
        dvar_seq = checker.typecheck_var_seq(dvars)
        return self._get_values(dvar_seq)

    def get_values(self, var_seq):
        """ Same as get_value_list
        """
        return self.get_value_list(var_seq)

    def _get_values(self, dvars):
        # internal: no checks are done.
        self_value_map = self._var_value_map
        return [self_value_map.get(dv, 0) for dv in dvars]

    def get_all_values(self):
        # internal: no checks are done.
        self_value_map = self._var_value_map
        m = self._model
        return [self_value_map.get(dv, 0) for dv in m.iter_variables()]

    def get_value_dict(self, var_dict, keep_zeros=True, precision=0):
        """ Converts a dictionary of variables to a dictionary of solutions

        Assuming `var_dict` is a dictionary of variables
        (for example, as returned by `Model.integer_var_dict()`,
        returns a dictionary with the same keys and as values the solution values of the
        variables.

        :param var_dict: a dictionary of decision variables.
        :param keep_zeros: an optional flag to keep zero values (default is True)
        :param precision: an optional precision, used when filtering out zero values. The default is 1e-6.
            When keep_zeros is False, all values smaller than this value are left out.

        :return: A dictionary from variable keys to solution values (floats).
        """
        # assume var_dict is a key-> variable dictionary
        if keep_zeros:
            return {k: self._get_var_value(v) for k, v in iteritems(var_dict)}
        else:
            value_dict = {}
            for key, dvar in iteritems(var_dict):
                dvar_value = self._get_var_value(dvar)
                if (precision and abs(dvar_value) >= precision) or dvar_value:
                    value_dict[key] = dvar_value
            return value_dict

    def get_value_df(self, var_dict, value_column_name=None, key_column_names=None):
        """ Returns values of a dicitonary of variables, as a pandas dataframe.

        If pandas is not present, returns a dicitonary of columns.

        :param var_dict: the dicitonary of variables, as created by Model.xx_var_dict
        :param value_column_name: an optional string to name the value column. Default is 'value'
        :param key_column_names: an optional list of strings to name th ekeys of the dicitonary.
            If not present, keys are named 'k1', 'k2', ...

        :return: a pandas DataFrame, if pandas is present.
        """
        keys = list(six.iterkeys(var_dict))
        values = self.get_values(six.itervalues(var_dict))
        if isinstance(keys[0], tuple):
            keys = list(zip(*keys))
            knames = None
            if key_column_names:
                if len(key_column_names) == len(keys):
                    knames = key_column_names
            if not knames:
                knames = ['key_%d' % k for k in range(1, len(keys)+1)]
            kd = {kn: ks for kn, ks in zip(knames, keys)}
        else:
            kn = key_column_names or 'key'
            kd = {kn: keys}
        value_col_name = value_column_name or 'value'
        kd[value_col_name] = values
        try:
            import pandas as pd
            return pd.DataFrame(kd)
        except ImportError:
            self.model.warning("pandas module not found, returning a dict instead of DataFrame")
            return kd

    # def __len__(self):
    #     return len(self.__var_value_map)

    @property
    def number_of_var_values(self):
        """ This property returns the number of variable values stored in this solution.

        """
        return len(self._var_value_map)

    def __getitem__(self, arg):
        return self.get_value(arg)

    def get_status(self, ct):
        """ Returns the status of a linear constraint in the solution.

        Returns 1 if the constraint is satisfied, else returns 0. This is particularly useful when using
        the status variable of constraints.

        :param ct: A linear constraint
        :return: a number (1 or 0)
        """
        self._checker.typecheck_linear_constraint(ct)
        return self._get_status(ct)

    def _get_status(self, ct):
        # INTERNAL
        ct_status_var = ct._get_status_var()
        if ct_status_var:
            return self._var_value_map.get(ct_status_var, 0)
        elif ct.is_added():
            # a posted constraint is true if there is a solution
            return 1
        else:
            return 1 if ct.is_satisfied(self) else 0

    def find_unsatisfied_constraints(self, m, tolerance=1e-6):
        unsats = []
        for ct in m.iter_constraints():
            if not ct.is_satisfied(self, tolerance):
                unsats.append(ct)
        return unsats

    def _var_match_function(self, mdl, match="auto"):
        if mdl is self._model:
            def find_matching_var(dvar_): return dvar_
        elif match == "index" or match == "auto" and mdl.statistics == self._model.statistics:
            def find_matching_var(dvar_):
                return mdl.get_var_by_index(dvar_.index)
        else:
            def find_matching_var(dvar1):
                return mdl.get_var_by_name(dvar1.name)
        return find_matching_var

    def number_of_var_diffs(self, other_sol, precision=1e-6, match="auto"):
        var_match_fn = self._var_match_function(other_sol.model, match)
        nb_diffs = 0
        for dv, dvv in self.iter_var_values():
            other_dv = var_match_fn(dv)
            if other_dv:
                other_dvv = other_sol[other_dv]
                if abs(dvv - other_dvv) >= precision:
                    nb_diffs += 1
        return nb_diffs

    def restore(self, mdl, abs_tolerance=1e-6, rel_tolerance=1e-4, restore_all=False, match="auto"):
        # restores the solution in its model, adding ranges.
        find_matching_var = self._var_match_function(mdl, match)
        lfactory = mdl._lfactory
        restore_ranges = []
        for dvar, val in self.iter_var_values():
            if not dvar.is_generated() or restore_all:
                dvar2 = find_matching_var(dvar)
                if dvar2 is not None:
                    rel_prec = abs(val) * rel_tolerance
                    used_prec = max(abs_tolerance, rel_prec)
                    rlb = max(dvar2.lb, val - used_prec)
                    rub = min(dvar2.ub, val + used_prec)
                    restore_ranges.append(lfactory.new_range_constraint(rlb, dvar2, rub))
                else:
                    print("could not find matching var for {0}".format(dvar))
        mdl.info("restored {0} variable values using range constraints".format(len(restore_ranges)))
        return mdl.add(restore_ranges)

    def find_invalid_domain_variables(self, m, tolerance=1e-6):
        invalid_domain_vars = []
        for dv in m.iter_variables():
            dvv = self.get_var_value(dv)
            if not dv.accepts_value(dvv, tolerance=tolerance):
                invalid_domain_vars.append(dv)
        return invalid_domain_vars

    def is_valid_solution(self, tolerance=1e-6, silent=True):
        """ Returns True if the solution is feasible.

        This method checks that solution values for variables are compatible for their types
        and bounds. It also checks that all constraints are satisfied, within the tolerance.

        :param tolerance: a float number used to check satisfaction; default is 1e-6.
        :param silent: optional flag. If False, prints which variable (or constraint)
          causes the solution to be invalid. default is False(prints nothing.

        :return: True if the solution is valid, within the tolerance value.

        *New in version 2.13*
        """
        m = self.model
        verbose = not silent
        invalid_domain_vars = self.find_invalid_domain_variables(m, tolerance)
        if verbose and invalid_domain_vars:
            m.warning("invalid domain vars: {0}".format(len(invalid_domain_vars)))
            for v, ivd in enumerate(invalid_domain_vars, start=1):
                dvv = self.get_var_value(ivd)
                m.warning("{0} - invalid value {1} for variable {2!s}".format(v, dvv, ivd))

        unsat_cts = self.find_unsatisfied_constraints(m, tolerance)
        if verbose and unsat_cts:
            m.info("unsatisfied constraints[{0}]".format(len(unsat_cts)))
            for u, uct in enumerate(unsat_cts, start=1):
                m.warning("{0} - unsatisfied constraint: {1!s}".format(u, uct))
        return not (invalid_domain_vars or unsat_cts)

    is_feasible_solution = is_valid_solution

    def equals(self, other, check_models=False, obj_precision=1e-3, var_precision=1e-6):
        from itertools import dropwhile
        if check_models and (self.model != other.model):
            return False

        if is_iterable(self.objective_value) and is_iterable(other.objective_value):
            if len(self.objective_value) == len(other.objective_value):
                for self_obj_val, other_obj_val in zip(self.objective_value, other.objective_value):
                    if abs(self_obj_val - other_obj_val) >= obj_precision:
                        return False
            else:  # Different number of objectives
                return False
        elif not is_iterable(self.objective_value) and not is_iterable(other.objective_value):
            if abs(self.objective_value - other.objective_value) >= obj_precision:
                return False
        else:  # One solution is for multi-objective, and not the other
            return False

        # noinspection PyPep8
        this_triplets  = [(dv.index, dv.name, svalue) for dv, svalue in dropwhile(lambda dvv: not dvv[1],
                                                                                  self.iter_var_values())]
        other_triplets = [(dv.index, dv.name, svalue) for dv, svalue in dropwhile(lambda dvv: not dvv[1],
                                                                                  other.iter_var_values())]
        # noinspection PyArgumentList
        res = True
        for this_triple, other_triple in izip(this_triplets, other_triplets):
            this_index, this_name, this_val = this_triple
            other_index, other_name, other_val = other_triple
            if other_index != this_index or this_name != other_name or \
                    abs(this_val - other_val) >= var_precision:
                res = False
                break
        return res

    def ensure_reduced_costs(self, model, engine):
        if self._reduced_costs is None:
            self._reduced_costs = engine.get_all_reduced_costs(model)

    def ensure_dual_values(self, model, engine):
        if self._dual_values is None:
            self._dual_values = engine.get_all_dual_values(model)

    def ensure_slack_values(self, model, engine):
        if self._slack_values is None:
            self._slack_values = engine.get_all_slack_values(model)

    def ensure_basis_statuses(self, model, engine):
        if self._basis_statuses is None:
            #  returns a tuple of two lists
            self._basis_statuses = engine.get_basis(model)

    def has_basis(self):
        m = self.model
        self.ensure_basis_statuses(m, m.get_engine())
        return self._has_basis()

    def _has_basis(self):
        try:
            return len(self._basis_statuses[0]) > 0
        except TypeError:
            return False

    def get_reduced_costs(self, dvars):
        m = self.model
        self.ensure_reduced_costs(m, m.get_engine())
        rcs = self._reduced_costs
        assert rcs is not None
        return [rcs.get(dv, 0) for dv in dvars]

    def get_dual_values(self, lcts):
        duals = self._dual_values
        assert duals is not None
        return [duals.get(lc, 0) for lc in lcts]

    def get_slacks(self, cts):
        all_slacks = self._slack_values
        assert all_slacks is not None
        # first get cplex_scope, then fetch the slack: two indirections
        return [all_slacks[ct.cplex_scope].get(ct, 0) for ct in cts]

    def slack_value(self, ct, handle_error='raise'):
        all_slacks = self._slack_values
        slack = 0
        if all_slacks is None:
            if handle_error == 'raise':
                self.model.fatal("Solution contains no slack data")
            elif handle_error == 'ignore':
                pass
            else:
                raise ValueError("handle_error expects 'raise|ignore|None, {0} was passed".format(handle_error))
        else:
            slack = all_slacks[ct.cplex_scope].get(ct, 0)
        return slack

    def get_var_basis_statuses(self, dvars):
        assert self._basis_statuses is not None
        all_var_basis_statuses = self._basis_statuses[0]
        return [BasisStatus.parse(all_var_basis_statuses.get(dv, -1)) for dv in dvars]

    def get_linearct_basis_statuses(self, linear_cts):
        assert self._basis_statuses is not None
        all_linearct_basis_statuses = self._basis_statuses[1]
        return [BasisStatus.parse(all_linearct_basis_statuses.get(lct, -1)) for lct in linear_cts]

    def get_infeasibility(self, ct):
        return self._infeasibilities.get(ct, 0)

    def display_attributes(self):
        pass

    def display(self,
                print_zeros=True,
                header_fmt="solution for: {0:s}",
                objective_fmt="{0}: {1:.{prec}f}",
                value_fmt="{varname:s} = {value:.{prec}f}",
                iter_vars=None,
                **kwargs):
        print_generated = kwargs.get("print_generated", False)
        problem_name = self.problem_name
        if header_fmt and problem_name:
            print(header_fmt.format(problem_name))
        if self._problem_objective_expr is not None and objective_fmt and self.has_objective():
            obj_prec = self.model.objective_expr.float_precision
            obj_name = self._problem_objective_name()
            print(objective_fmt.format(obj_name, self._objective, prec=obj_prec))
        if iter_vars is None:
            iter_vars = self.iter_variables()
        print_counter = 0
        for dvar in iter_vars:
            if print_generated or not dvar.is_generated():
                var_value = self._get_var_value(dvar)
                if print_zeros or var_value:
                    print_counter += 1
                    varname = dvar.to_string()
                    if type(value_fmt) != type(varname):
                        # infamous mix of str and unicode. Should happen only
                        # in py2. Let's convert things
                        if isinstance(value_fmt, str):
                            # noinspection PyUnresolvedReferences
                            value_fmt = value_fmt.decode('utf-8')
                        else:
                            value_fmt = value_fmt.encode('utf-8')
                    output = value_fmt.format(varname=varname,
                                              value=var_value,
                                              prec=dvar.float_precision,
                                              counter=print_counter)
                    try:
                        print(output)
                    except UnicodeEncodeError:
                        encoding = 'ascii'
                        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
                            encoding = sys.stdout.encoding
                        print(output.encode(encoding,
                                            errors='backslashreplace'))

    def to_string(self, print_zeros=True):
        oss = StringIO()
        self.to_stringio(oss, print_zeros=print_zeros)
        return oss.getvalue()

    def _problem_objective_name(self, default_obj_name="objective"):
        # INTERNAL
        # returns the string used for displaying the objective
        # if the problem has an objective with a name, use it
        # else return the default (typically "objective"
        self_objective_expr = self._problem_objective_expr
        if self_objective_expr is not None and self_objective_expr.has_name():
            return self_objective_expr.name
        else:
            return default_obj_name

    def to_stringio(self, oss, print_zeros=True):
        problem_name = self.problem_name
        if problem_name:
            oss.write("solution for: %s\n" % problem_name)
        if self._problem_objective_expr is not None and self.has_objective():
            obj_name = self._problem_objective_name()
            oss.write("%s: %g\n" % (obj_name, self._objective))

        value_fmt = "{var:s}={value:.{prec}f}"
        for dvar, val in self.iter_var_values():
            if not dvar.is_generated():
                var_value = self._get_var_value(dvar)
                if print_zeros or var_value != 0:
                    oss.write(value_fmt.format(var=str(dvar), value=var_value, prec=dvar.float_precision))
                    oss.write("\n")

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        if self.has_objective():
            s_obj = "obj={0:g}".format(self.objective_value)
        else:
            s_obj = "obj=N/A"
        s_values = ",".join(["{0!s}:{1:g}".format(var, val) for var, val in iteritems(self._var_value_map)])
        r = "docplex.mp.solution.SolveSolution({0},values={{{1}}})".format(s_obj, s_values)
        return str_maxed(r, maxlen=72)

    def __iter__(self):
        # INTERNAL: this is necessary to prevent solution from being an iterable.
        # as it follows getitem protocol, it can mistakenly be interpreted as an iterable
        raise TypeError

    def __as_df__(self, name_key='name', value_key='value'):
        return self.as_df(name_key, value_key)

    def as_df(self, name_key='name', value_key='value'):
        """ Converts the solution to a pandas dataframe with two columns: variable name and values

        :param name_key: column name for variable names. Default is 'name'
        :param value_key: cilumn name for values., Default is 'value'.

        :return: a pandas dataframe, if pandas is present.

        *New in version 2.15*
        """
        assert name_key
        assert value_key
        assert name_key != value_key
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('Cannot convert solution to pandas.DataFrame if pandas is not available')

        names = []
        values = []
        for dv, dvv in self.iter_var_values():
            names.append(dv.to_string())
            values.append(dvv)
        name_value_dict = {name_key: names, value_key: values}
        return pd.DataFrame(name_value_dict)

    def print_mst(self, outs=None, **kwargs):
        """ Writes the solution in MST format in an output stream (default is sys.out)
        """
        if outs is None:
            outs = sys.stdout
        self.export(outs, format='mst', **kwargs)

    def _export_as_string(self, fmt, **kwargs):
        oss = StringIO()
        printer = self.get_printer(fmt)
        printer.print_to_stream(self, oss, **kwargs)
        return oss.getvalue()

    def export_as_mst_string(self, write_level=WriteLevel.Auto, **kwargs):
        kwargs['write_level'] = WriteLevel.parse(write_level)
        return self._export_as_string(fmt='mst', **kwargs)

    def export_as_mst(self, path=None, basename=None, write_level=WriteLevel.Auto, **kwargs):
        """ Exports a solution to a file in CPLEX mst format.

        Args:
            basename: Controls the basename with which the solution is printed.
                Accepts None, a plain string, or a string format.
                If None, the model's name is used.
                If passed a plain string, the string is used in place of the model's name.
                If passed a string format (either with %s or {0}), this format is used to format the
                model name to produce the basename of the written file.
            path: A path to write the file, expects a string path or None.
                Can be a directory, in which case the basename
                that was computed with the basename argument is appended to the directory to produce
                the file.
                If given a full path, the path is directly used to write the file, and
                the basename argument is not used.
                If passed None, the output directory will be ``tempfile.gettempdir()``.
            write_level: an enumerated value which controls which variables are printed.
                The default is WriteLevel.Auto, which prints the values of all discrete variables.
                This parameter also accepts the number values of the corresponding CPLEX parameters
                (1 for AllVars, 2 for DiscreteVars, 3 for NonZeroVars, 4 for NonZeroDiscreteVars)

        Returns:
            The full path of the file, when successful, else None

        Examples:
            Assuming the solution has the name "prob":

            ``sol.export_as_mst()`` will write file prob.mst in a temporary directory.

            ``sol.export_as_mst(write_level=WriteLevel.ALlvars)`` will write file prob.mst in a temporary directory,
                and will print all variables in the problem.

            ``sol.export_as_mst(path="c:/temp/myprob1.mst")`` will write file "c:/temp/myprob1.mst".

            ``sol.export_as_mst(basename="my_%s_mipstart", path ="z:/home/")`` will write "z:/home/my_prob_mipstart.mst".

        See Also:
            :class:`docplex.mp.constants.WriteLevel`
        """
        sol_basename = normalize_basename(self.problem_name, force_lowercase=True)
        mst_path = make_output_path2(actual_name=sol_basename,
                                     extension=SolutionMSTPrinter.mst_extension,
                                     path=path,
                                     basename_fmt=basename)
        if mst_path:
            kwargs2 = kwargs.copy()
            kwargs2['write_level'] = WriteLevel.parse(write_level)
            SolutionMSTPrinter.print_to_stream(self, mst_path, **kwargs2)
            return mst_path

    @classmethod
    def get_printer(cls, key):
        # INTERNAL
        printers = {'json': SolutionJSONPrinter,
                    'xml': SolutionMSTPrinter,
                    'mst': SolutionMSTPrinter
                    }

        printer = printers.get(key.lower())
        if not printer:
            raise ValueError("format key must be one of {}".format(printers.keys()))
        return printer

    def export(self, file_or_filename, format="json", **kwargs):
        """ Export this solution.
        
        Args:
            file_or_filename: If ``file_or_filename`` is a string, this argument contains the filename to
                write to. If this is a file object, this argument contains the file object to write to.
            format: Name of format to use. Possible values are:
                - "json"
                - "mst": the MST cplex format for MIP starts
                - "xml": same as MST
            kwargs: The kwargs passed to the actual exporter
        """

        printer = self.get_printer(format)

        if isinstance(file_or_filename, six.string_types):
            fp = open(file_or_filename, "w")
            close_fp = True
        else:
            fp = file_or_filename
            close_fp = False
        try:
            printer.print_to_stream(self, fp, **kwargs)
        finally:
            if close_fp:
                fp.close()

    def export_as_json_string(self, **kwargs):
        """ Returns the solution as a string in JSON format.

        :return: a string.

        *New in version 2.10*
        """
        return self._export_as_string(fmt='json', **kwargs)

    def check_as_mip_start(self, strong_check=False):
        """Checks that this solution is a valid MIP start.

        To be valid, it must have:

            * at least one discrete variable (integer or binary), and
            * the values for decision variables should be consistent with the type.

        Returns:
            Boolean: True if this solution is a valid MIP start.
        """
        count_values = 0
        count_errors = 0
        m = self.model
        for dv, dvv in self.iter_var_values():
            if dv.is_discrete() and not dv.is_generated():
                count_values += 1
                if not dv.accepts_value(dvv):  # pragma: no cover
                    count_errors += 1
                    m.warning("Solution value {1} is outside the domain of variable {0!r}: {1}, type: {2!s}",
                              dv, dvv, dv.vartype.short_name)
        if count_values == 0:
            docplex_fatal("MIP start contains no discrete variable")  # pragma: no cover
        return not count_errors if strong_check else True

    def as_dict(self, keep_zeros=False):
        var_value_dict = {}
        # INTERNAL: return a dictionary of variable: variable_value
        for dvar, dval in self.iter_var_values():
            if keep_zeros or dval:
                var_value_dict[dvar] = dval
        return var_value_dict

    def as_name_dict(self, keep_zeros=False):
        # INTERNAL: return a dictionary of variable_name: variable_value
        var_value_dict = {}
        for dvar, dval in self.iter_var_values():
            dvar_name = dvar.get_name()
            if keep_zeros or dval:
                if dvar_name:
                    var_value_dict[dvar_name] = dval
            else:
                var_value_dict[dvar.lp_name] = dval
        return var_value_dict

    def kpi_value_by_name(self, name, match_case=False):
        ''' Returns the solution value of a KPI from its name.

        Args:
            name (string): The string to be matched.

            match_case (boolean): If True, looks for a case-exact match, else
               ignores case. Default is False.

        Returns:
            The value of the KPI, evaluated in the solution.

        Note:
            This method raises an error when the string does not match any KPI in the model.

        See:
            :func: `docplex.mp.model.kpi_by_name`
        '''
        kpi = self.model.kpi_by_name(name, try_match=True, match_case=match_case)
        return kpi._get_solution_value(self)


class SolutionPool(object):
    """SolutionPool()

     Solutions pools as returned by `Model.populate()`

    This class is not to be instantiated by users, only used after returned by Model.populate.

    Instances of this class can be used like lists. They are fully iterable,
    and accessible by index.

    See Also:
        :func:`docplex.mp.model.Model.populate`

    """

    def __init__(self, sols, num_replaced=0):
        self._solutions = tuple(sols)
        self._num_replaced = num_replaced

    def __iter__(self):
        """ Returns an iterator on pool solutions.
        """
        return iter(self._solutions)

    def __len__(self):
        """ Returns the number of solutions in the pool.

        """
        return self.size

    @property
    def size(self):
        """ Returns the number of solutions in the pool.

        :return:
        """
        return len(self._solutions)

    def __getitem__(self, item):
        return self._solutions[item]

    def __str__(self):
        return 'SolutionPool[{0}](mean={1:.3f})'.format(len(self), self.mean_objective_value)

    def __repr__(self):
        return "docplex.mp.SolutionPool[{0}]".format(len(self))

    @property
    def num_replaced(self):
        return self._num_replaced

    @property
    def mean_objective_value(self):
        """ This property returns the mean objective value in the pool.

        """
        return self.stats[1]

    def describe_objectives(self):
        """ Prints statistical information about poolobjective values.

        Relies on the `stats` property.

        """
        nb_solutions, obj_mean, obj_sd, obj_min, obj_med, obj_max = self.stats
        print("count  = {0}".format(nb_solutions))
        print("mean   = {0}".format(obj_mean))
        print("std    = {0}".format(obj_sd))
        print("min    = {0}".format(obj_min))
        print("med    = {0}".format(obj_med))
        print("max    = {0}".format(obj_max))

    @property
    def stats(self):
        """ Returns statistics about pool objective values.

        :return: a tuple of floats containing (in this order:
            - number of solutions (same as len()
            - mean objective value
            - standard deviation
            - minimum objective value
            - median objective value
            - maximum objective value

        Note:
            if pool is empty returns dummy values, only the first value (len of 0) is valid.
        """
        from math import sqrt
        nb_solutions = len(self)
        obj_min = 1e+75
        obj_max = -1e+75
        if not nb_solutions:
            # dummy values
            return 0, 0, 0, obj_min, obj_min, obj_max

        objs = []
        obj_sum1 = 0
        obj_sum2 = 0
        for ps in self._solutions:
            obj = ps.objective_value
            objs.append(obj)
            if obj < obj_min:
                obj_min = obj
            if obj > obj_max:
                obj_max = obj
            obj_sum1 += obj
            obj_sum2 += obj * obj
        obj_med = sorted(objs)[nb_solutions//2]

        obj_mean = obj_sum1 / nb_solutions
        variance = (obj_sum2 / nb_solutions) - (obj_mean ** 2)
        obj_sd = sqrt(variance)
        return nb_solutions, obj_mean, obj_sd, obj_min, obj_med, obj_max
