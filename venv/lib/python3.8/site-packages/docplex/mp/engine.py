# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


from docplex.mp.solution import SolveSolution
from docplex.mp.sdetails import SolveDetails
from docplex.mp.utils import DOcplexException
from docplex.mp.error_handler import docplex_fatal
from docplex.util.status import JobSolveStatus
from docplex.mp.constants import CplexScope


# gendoc: ignore


class IEngine(object):

    def only_cplex(self, mname):
        raise TypeError("{0} requires CPLEX - not available.".format(mname))  # pragma: no cover

    def register_callback(self, cb):
        raise NotImplementedError  # pragma: no cover

    def connect_progress_listeners(self, listeners, model):
        raise NotImplementedError  # pragma: no cover

    def solve(self, mdl, parameters, **kwargs):
        ''' Redefine this method for the real solve.
            Returns a solution object or None.
        '''
        raise NotImplementedError  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        """
        Runs feasopt-like algorithm with a set of relaxable cts with preferences

        :param mdl: The model for which relaxation is performed.
        :param relaxable_groups:
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        """
        Runs conflict-refiner algorithm with an optional set constraints groups with preferences

        :param mdl: The model for which conflict refinement is performed.
        :param preferences: an optional dictionary defining constraints preferences.
        :param groups: an optional list of 'docplex.mp.conflict_refiner.ConstraintsGroup'.
        :param parameters:
        :return: A list of "TConflictConstraint" namedtuples, each tuple corresponding to a constraint that is
            involved in the conflict.
            The fields of the "TConflictConstraint" namedtuple are:
                - the name of the constraint or None if the constraint corresponds to a variable lower or upper bound
                - a reference to the constraint or to a wrapper representing a Var upper or lower bound
                - an :enum:'docplex.mp.constants.ConflictStatus' object that indicates the
                conflict status type (Excluded, Possible_member, Member...)
            This list is empty if no conflict is found by the conflict refiner.
        """
        raise NotImplementedError  # pragma: no cover

    def populate(self, **kwargs):
        raise NotImplementedError  # pragma: no cover

    def get_solve_status(self):
        """  Return a DOcplexcloud-style solve status.

        Possible enums are in docloud/status.py
        Default is UNKNOWN at this stage. Redefined for CPLEX and DOcplexcloud engines.
        """
        return JobSolveStatus.UNKNOWN  # pragma: no cover

    def get_cplex(self):
        """
        Returns the underlying CPLEX, if any. May raise an exception if not applicable.
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def has_cplex(self):  # pragma: no cover
        try:
            return self.get_cplex() is not None
        except DOcplexException:
            # some engine may raise an exception when accessing a cplex
            return False

    def set_parameter(self, parameter, value):
        """ Changes the parameter value.
        :param parameter:
        :param value:
        """
        raise NotImplementedError  # pragma: no cover

    def get_parameter(self, parameter):
        raise NotImplementedError  # pragma: no cover

    def get_solve_details(self):
        raise NotImplementedError  # pragma: no cover

    def supports_logical_constraints(self):
        raise NotImplementedError  # pragma: no cover

    def solved_as_mip(self):
        return False

    @property
    def name(self):
        raise NotImplementedError  # pragma: no cover

    def get_infinity(self):
        raise NotImplementedError  # pragma: no cover

    def create_one_variable(self, vartype, lb, ub, name):
        raise NotImplementedError  # pragma: no cover

    def create_variables(self, nb_vars, vartype, lb, ub, name):
        raise NotImplementedError  # pragma: no cover

    def create_multitype_variables(self, keys, vartypes, lbs, ubs, names):
        raise NotImplementedError  # pragma: no cover

    def create_linear_constraint(self, binaryct):
        raise NotImplementedError  # pragma: no cover

    def create_block_linear_constraints(self, ct_seq):
        raise NotImplementedError  # pragma: no cover

    def create_range_constraint(self, rangect):
        raise NotImplementedError  # pragma: no cover

    def create_logical_constraint(self, logct, is_equivalence):
        raise NotImplementedError  # pragma: no cover

    def create_batch_logical_constraints(self, logcts, is_equivalence):
        # the default is to iterate and append.
        return [self.create_logical_constraint(logc, is_equivalence) for logc in logcts]

    def create_quadratic_constraint(self, qct):
        raise NotImplementedError  # pragma: no cover

    def create_pwl_constraint(self, pwl_ct):
        raise NotImplementedError  # pragma: no cover

    def remove_constraint(self, ct):
        raise NotImplementedError  # pragma: no cover

    def remove_constraints(self, cts):
        raise NotImplementedError  # pragma: no cover

    def set_objective_sense(self, sense):
        raise NotImplementedError  # pragma: no cover

    def set_objective_expr(self, new_objexpr, old_objexpr):
        raise NotImplementedError  # pragma: no cover

    def set_multi_objective_exprs(self, new_multiobjexprs, old_multiobjexprs,
                                  priorities=None, weights=None, abstols=None, reltols=None, objnames=None):
        raise NotImplementedError  # pragma: no cover

    def set_multi_objective_tolerances(self, abstols, reltols):
        raise NotImplementedError  # pragma: no cover

    def end(self):
        raise NotImplementedError  # pragma: no cover

    def set_streams(self, out):
        raise NotImplementedError  # pragma: no cover

    def set_var_lb(self, var, lb):
        raise NotImplementedError  # pragma: no cover

    def set_var_ub(self, var, ub):
        raise NotImplementedError  # pragma: no cover

    def rename_var(self, var, new_name):
        raise NotImplementedError  # pragma: no cover

    def change_var_types(self, dvars, new_types):
        raise NotImplementedError  # pragma: no cover

    def update_objective_epxr(self, expr, event, *args):
        raise NotImplemented  # pragma: no cover

    def update_constraint(self, ct, event, *args):
        raise NotImplementedError  # pragma: no cover

    def check_var_indices(self, dvars):
        raise NotImplementedError  # pragma: no cover

    def check_constraint_indices(self, cts, ctscope):
        raise NotImplementedError  # pragma: no cover

    def create_sos(self, sos):
        raise NotImplementedError  # pragma: no cover

    def clear_all_sos(self):
        raise NotImplementedError  # pragma: no cover

    def get_basis(self, mdl):
        raise NotImplementedError  # pragma: no cover

    def set_lp_start(self, var_stats, ct_stats):  # pragma: no cover
        raise NotImplementedError  # pragma: no cover

    def export(self, out, fmt):  # pragma: no cover
        raise NotImplementedError  # pragma: no cover

    def resync(self):
        raise NotImplementedError  # pragma: no cover


# noinspection PyAbstractClass
class MinimalEngine(IEngine):
    # define as most methods as possible with reasonable defaults.

    def export(self, out, fmt):
        self.only_cplex("export to %s" % fmt.name)

    def end(self):
        pass  # pragma: no cover

    def set_streams(self, out):
        pass  # pragma: no cover

    def register_callback(self, cb):
        # no callbacks
        pass  # pragma: no cover

    def connect_progress_listeners(self, listeners, model):
        # no listeners
        if listeners:
            self.only_cplex(mname="connect_progress_listeners")  # pragma: no cover

    def set_parameter(self, parameter, value):
        #
        pass  # pragma: no cover

    def get_parameter(self, parameter):
        # engine value of a parameter is its own
        return parameter.get()  # pragma: no cover

    def get_infinity(self):
        return 1e+20  # pragma: no cover

    def create_one_variable(self, vartype, lb, ub, name):
        # define create one var in terms of create batch vars
        xs = self.create_variables(1, vartype, [lb], [ub], [name])
        return xs[0]  # pragma: no cover

    def set_var_lb(self, var, lb):
        pass  # pragma: no cover

    def set_var_ub(self, var, ub):
        pass  # pragma: no cover

    def rename_var(self, var, new_name):
        pass  # pragma: no cover

    def rename_linear_constraint(self, linct, new_name):
        pass  # pragma: no cover

    def check_var_indices(self, dvars):
        pass  # pragma: no cover

    def check_constraint_indices(self, cts, ctscope):
        pass  # pragma: no cover

    def remove_constraints(self, cts):
        pass  # pragma: no cover

    def update_constraint(self, ct, event, *args):
        self.only_cplex(mname="update_constraint")

    def update_extra_constraint(self, lct, qualifier, *args):
        self.only_cplex(mname="update_extra_constraint")

    def remove_constraint(self, ct):
        pass  # pragma: no cover

    def create_sos(self, sos):  # pragma: no cover
        self.only_cplex("create SOS set")

    def clear_all_sos(self):
        pass  # pragma: no cover

    def create_linear_constraint(self, ct):
        return self.create_block_linear_constraints([ct])[0]  # pragma: no cover

    def get_solve_details(self):
        from docplex.mp.sdetails import SolveDetails
        return SolveDetails()  # pragma: no cover

    def get_basis(self, mdl):
        self.only_cplex("get_basis")  # pragma: no cover

    def set_lp_start(self, var_stats, ct_stats):
        self.only_cplex("set_lp_start")  # pragma: no cover

    def supports_logical_constraints(self):
        return False  # pragma: no cover

    def change_var_types(self, dvars, new_types):
        self.only_cplex(mname="change_var_type")  # pragma: no cover

    def get_cplex(self):
        docplex_fatal("No cplex is available.")  # pragma: no cover

    def create_batch_logical_constraints(self, logcts, is_equivalence):
        self.only_cplex(mname="create_batch_logical_constraints")  # pragma: no cover

    def create_logical_constraint(self, logct, is_equivalence):
        self.only_cplex(mname="create_logical_constraint")  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        self.only_cplex(mname="refine_conflict")  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        self.only_cplex(mname="solve_relaxed")  # pragma: no cover

    def populate(self, **kwargs):
        self.only_cplex(mname="populate")  # pragma: no cover

    def create_pwl_constraint(self, pwl_ct):
        self.only_cplex(mname="create_quadratic_constraint")  # pragma: no cover

    def create_quadratic_constraint(self, qct):
        self.only_cplex(mname="create_quadratic_constraint")  # pragma: no cover

    def set_multi_objective_exprs(self, new_multiobjexprs, old_multiobjexprs,
                                  priorities=None, weights=None, abstols=None, reltols=None, objnames=None):
        self.only_cplex(mname="set_multi_objective_exprs")  # pragma: no cover

    def set_multi_objective_tolerances(self, abstols, reltols):
        self.only_cplex(mname="set_multi_objective_tolerances")  # pragma: no cover

    def resync(self):
        pass


# noinspection PyAbstractClass,PyMethodMayBeStatic
class DummyEngine(IEngine):
    def export(self, out, fmt):
        self.only_cplex("export to %s" % fmt.name)

    def create_range_constraint(self, rangect):
        return -1  # pragma: no cover

    def create_logical_constraint(self, logct, is_equivalence):
        return -1  # pragma: no cover

    def create_quadratic_constraint(self, qct):
        return -1  # pragma: no cover

    def create_pwl_constraint(self, pwl_ct):
        return -1  # pragma: no cover

    def set_streams(self, out):
        pass  # pragma: no cover

    def get_infinity(self):
        return 1e+20  # pragma: no cover

    def create_one_variable(self, vartype, lb, ub, name):
        return -1  # pragma: no cover

    def create_variables(self, nb_vars, vartype, lb, ub, name):
        return [-1] * nb_vars  # pragma: no cover

    def create_multitype_variables(self, size, vartypes, lbs, ubs, names):
        return [-1] * size

    def set_var_lb(self, var, lb):
        pass

    def set_var_ub(self, var, ub):
        pass

    def rename_var(self, var, new_name):
        pass  # nothing to do, except in cplex...

    def rename_linear_constraint(self, linct, new_name):
        pass  # nothing to do, except in cplex...

    def change_var_types(self, dvars, new_types):
        pass  # nothing to do, except in cplex...

    def create_linear_constraint(self, binaryct):
        return -1  # pragma: no cover

    def create_block_linear_constraints(self, ct_seq):
        return [-1] * len(ct_seq)  # pragma: no cover

    def create_batch_logical_constraints(self, logcts, is_equivalence):
        return [-1] * len(logcts)  # pragma: no cover

    def remove_constraint(self, ct):
        pass  # pragma: no cover

    def remove_constraints(self, cts):
        pass  # pragma: no cover

    def set_objective_sense(self, sense):
        pass  # pragma: no cover

    def set_objective_expr(self, new_objexpr, old_objexpr):
        pass  # pragma: no cover

    def set_multi_objective_exprs(self, new_multiobjexprs, old_multiobjexprs,
                                  priorities=None, weights=None, abstols=None, reltols=None, objnames=None):
        pass  # pragma: no cover

    def set_multi_objective_tolerances(self, abstols, reltols):
        pass

    def end(self):
        pass  # pragma: no cover

    def register_callback(self, cb):
        pass  # pragma: no cover

    def unregister_callback(self, cb):
        pass  # pragma: no cover

    def connect_progress_listeners(self, listeners, model):
        if listeners:
            model.warning("Progress listeners require CPLEX, not supported on engine {0}.".format(self.name))

    def disconnect_progress_listeners(self, listeners):
        pass  # pragma: no cover

    def solve(self, mdl, parameters, **kwargs):
        return None  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.UNKNOWN  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        raise None  # pragma: no cover

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        raise None  # pragma: no cover

    def get_cplex(self):
        raise DOcplexException("No CPLEX is available.")  # pragma: no cover

    def update_objective(self, expr, event, *args):
        # nothing to do except for cplex
        pass  # pragma: no cover

    def update_constraint(self, ct, event, *args):
        pass  # pragma: no cover

    def supports_logical_constraints(self):
        return True, None

    def supports_multi_objective(self):
        return True, None  # pragma: no cover

    def check_var_indices(self, dvars):
        pass  # pragma: no cover

    def check_constraint_indices(self, cts, ctscope):
        pass  # pragma: no cover

    def create_sos(self, sos):
        pass  # pragma: no cover

    def clear_all_sos(self):
        pass  # pragma: no cover

    def get_basis(self, mdl):
        return None, None  # pragma: no cover

    def set_lp_start(self, var_stats, ct_stats):  # pragma: no cover
        raise DOcplexException('set_lp_start() requires CPLEX, not available for {0}'.format(self.name))

    def add_lazy_constraints(self, lazies):
        pass

    def clear_lazy_constraints(self):
        pass

    def add_user_cuts(self, lazies):
        pass

    def clear_user_cuts(self):
        pass

    def update_extra_constraint(self, lct, qualifier, *args):
        pass

    def resync(self):
        pass


# noinspection PyAbstractClass,PyUnusedLocal,PyMethodMayBeStatic
class IndexerEngine(DummyEngine):
    """
    An abstract engine facade which generates unique indices for variables, constraints
    """

    def __init__(self, initial_index=0):
        DummyEngine.__init__(self)
        self._initial_index = initial_index  # CPLEX indices start at 0, not 1
        self.__var_counter = self._initial_index
        self._ct_counter = self._initial_index

    def _increment_vars(self, size):
        self.__var_counter += size
        return self.__var_counter

    def _increment_cts(self, size):
        self._ct_counter += size
        return self._ct_counter

    def create_one_variable(self, vartype, lb, ub, name):
        old_count = self.__var_counter
        self._increment_vars(1)
        return old_count

    def create_variables(self, nb_vars, vartype, lb, ub, name):
        old_count = self.__var_counter
        new_count = self._increment_vars(nb_vars)
        return list(range(old_count, new_count))

    def create_multitype_variables(self, keys, vartypes, lbs, ubs, names):
        old_count = self.__var_counter
        new_count = self._increment_vars(len(keys))
        return list(range(old_count, new_count))

    def _create_one_ct(self):
        old_ct_count = self._ct_counter
        self._increment_cts(1)
        return old_ct_count

    def create_linear_constraint(self, binaryct):
        return self._create_one_ct()

    def create_batch_cts(self, ct_seq):
        old_ct_count = self._ct_counter
        size = sum(1 for _ in ct_seq)  # iterator is consumed
        self._increment_cts(size)
        return range(old_ct_count, self._ct_counter)

    def create_block_linear_constraints(self, ct_seq):
        return self.create_batch_cts(ct_seq)

    def create_range_constraint(self, rangect):
        return self._create_one_ct()

    def create_logical_constraint(self, logct, is_equivalence):
        return self._create_one_ct()

    def create_batch_logical_constraints(self, logcts, is_equivalence):
        return self.create_batch_cts(logcts)

    def create_quadratic_constraint(self, ind):
        return self._create_one_ct()

    def create_pwl_constraint(self, pwl_ct):
        return self._create_one_ct()

    def get_all_reduced_costs(self, mdl):
        return {}

    def get_all_dual_values(self, mdl):
        return {}

    def get_all_slack_values(self, mdl):
        return {CplexScope.LINEAR_CT_SCOPE: {},
                CplexScope.QUAD_CT_SCOPE: {},
                CplexScope.IND_CT_SCOPE: {}}

    def set_objective_sense(self, sense):
        pass

    def set_objective_expr(self, new_objexpr, old_objexpr):
        pass

    def set_parameter(self, parameter, value):
        """ Changes the parameter value in the engine.

        For this limited type of engine, nothing to do.

        """
        pass

    def get_parameter(self, parameter):
        """ Gets the current value of a parameter.

        Params:
         parameter: the parameter for which we query the value.

        """
        return parameter.get()


class NoSolveEngine(IndexerEngine):

    def populate(self, **kwargs):
        return None

    def get_solve_details(self):
        SolveDetails.make_fake_details(time=0, feasible=False)

    # INTERNAL: a dummy engine that cannot solve.

    # noinspection PyUnusedLocal
    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)

    @property
    def name(self):
        return "nosolve"

    @staticmethod
    def _no_cplex_error(mdl, method_name):  # pragma: no cover
        mdl.fatal("No CPLEX DLL and no DOcplexcloud credentials: {0} is not available".format(method_name))

    def solve(self, mdl, parameters, **kwargs):  # pragma: no cover
        """
        This solver cannot solve. never ever.
        """
        self._no_cplex_error(mdl, method_name="solve")
        return None

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):  # pragma: no cover
        self._no_cplex_error(mdl, method_name="solve_relaxed")
        return None

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):  # pragma: no cover
        self._no_cplex_error(mdl, method_name="refine_conflict")
        return None

    @staticmethod
    def make_from_model(mdl):
        # used in pickle
        eng = NoSolveEngine(mdl)
        eng._increment_vars(mdl.number_of_variables)
        eng._increment_cts(mdl.number_of_constraints)
        return eng


# noinspection PyUnusedLocal
class ZeroSolveEngine(IndexerEngine):
    def populate(self, **kwargs):
        return []

    # INTERNAL: a dummy engine that says it can solve
    # but returns an all-zero solution.
    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover
        self._last_solved_parameters = None

    def show_parameters(self, params):
        if params is None:
            print("DEBUG> parameters: None")
        else:
            if params.has_nondefaults():
                print("DEBUG> parameters:")
                params.print_information(indent_level=8)  #
            else:
                print("DEBUG> parameters: defaults")

    @property
    def last_solved_parameters(self):
        return self._last_solved_parameters

    @property
    def name(self):
        return "zero_solve"

    @staticmethod
    def get_var_zero_solution(dvar):
        return max(0, dvar.lb)

    def solve(self, mdl, parameters, **kwargs):
        # remember last solved params
        self._last_solved_parameters = parameters.clone() if parameters is not None else None
        self.show_parameters(parameters)
        return self.make_zero_solution(mdl)

    def make_zero_solution(self, mdl):
        # return a feasible value: max of zero and the lower bound
        zlb_map = {v: self.get_var_zero_solution(v) for v in mdl.iter_variables() if v.lb}
        obj = mdl.objective_expr.constant
        return SolveSolution(mdl, obj=obj, var_value_map=zlb_map, solved_by=self.name)  # pragma: no cover

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        params = parameters or mdl.parameters
        self._last_solved_parameters = params
        self.show_parameters(params)
        return self.make_zero_solution(mdl)

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        return None  # pragma: no cover

    def get_solve_details(self):
        return SolveDetails.make_fake_details(time=0, feasible=True)
