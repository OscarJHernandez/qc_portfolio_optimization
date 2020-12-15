# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2020
# --------------------------------------------------------------------------

from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
import re
import numbers
import six
import sys

from docplex.mp.engine import IEngine
from docplex.mp.utils import DOcplexException, str_maxed, is_string
from docplex.mp.compat23 import izip
from docplex.mp.error_handler import docplex_debug_msg

from docplex.mp.constants import ConflictStatus
from docplex.mp.constr import IndicatorConstraint, RangeConstraint, BinaryConstraint, \
    EquivalenceConstraint
from docplex.mp.progress import ProgressData
from docplex.mp.solution import SolveSolution, SolutionPool
from docplex.mp.sdetails import SolveDetails
from docplex.mp.conflict_refiner import TConflictConstraint, VarLbConstraintWrapper, VarUbConstraintWrapper, ConflictRefinerResult
from docplex.mp.cplex_adapter import CplexAdapter


from docplex.mp.compat23 import fast_range, copyreg
# noinspection PyPep8Namingc
from docplex.mp.constants import QualityMetric, UpdateEvent as upd

from docplex.mp.environment import Environment
from docplex.mp.engine import NoSolveEngine


from docplex.mp.constants import CplexScope


def _compute_lp_pass_iterations(cpxs_multiobj, s):
    itcnt = cpxs_multiobj.get_info(s, cpxs_multiobj.long_info.num_iterations)
    if itcnt:
        return itcnt
    siftitcnt = cpxs_multiobj.get_info(s, cpxs_multiobj.long_info.num_sifting_iterations)
    if siftitcnt:
        return siftitcnt
    return cpxs_multiobj.get_info(s, cpxs_multiobj.long_info.num_barrier_iterations)


def _compute_mip_pass_iterations(cpxs_multiobj, s):
    itcnt = cpxs_multiobj.get_info(s, cpxs_multiobj.long_info.num_iterations)
    return itcnt


def get_progress_details(cpx):
    is_mip = cpx._is_MIP()
    if CplexEngine._has_multi_objective(cpx):
        cpxs_multiobj = cpx.solution.multiobj
        compute_pass_iterations = _compute_mip_pass_iterations if is_mip\
            else _compute_lp_pass_iterations
        n_solves = cpxs_multiobj.get_num_solves()
        n_iters = tuple((compute_pass_iterations(cpxs_multiobj, s) for s in range(n_solves)))
        info_nodes = cpxs_multiobj.long_info.num_nodes
        if is_mip:
            n_nodes = (cpxs_multiobj.get_info(s, info_nodes) for s in range(n_solves))
        else:
            n_nodes = tuple((0 for _ in range(n_solves)))
        return n_iters, n_nodes

    else:
        n_iters = cpx.solution.progress.get_num_iterations()
        n_nodes = cpx.solution.progress.get_num_nodes_processed() if is_mip else 0
        return n_iters, n_nodes

# gendoc: ignore


# gendoc: ignore



def _get_connect_listeners_callback(cplex_module):
    class ConnectListenersCallback(cplex_module.callbacks.MIPInfoCallback):
        RELATIVE_EPS = 1e-5
        ABS_EPS = 1e-4

        def make_solution_from_incumbents(self, obj, incumbents):
            # INTERNAL
            mdl = self._model
            assert mdl is not None

            sol = SolveSolution(mdl, obj=obj, solved_by='cplex')
            for dv in mdl.iter_variables():
                # incumbent values are provided as a list with indices as positions.
                incumbent_value = incumbents[dv._index]
                dvv = mdl._round_element_value_if_necessary(dv, incumbent_value)
                sol._set_var_value_internal(dv, dvv)
            return sol

        # noinspection PyAttributeOutsideInit
        def initialize(self, listeners, model):
            self._count = 0
            self._model = model
            assert model is not None
            self._listeners = listeners
            self._start_time = -1
            self._start_dettime = -1
            for l in listeners:
                l._connect_cb(self)
                # precompute the set of those listeners which listen to solutions.
            self._solution_listeners = set(l for l in listeners if l.requires_solution() and hasattr(l, 'notify_solution'))

        def __call__(self):
            self._count += 1

            has_incumbent = self.has_incumbent()

            if self._start_time < 0:
                self._start_time = self.get_start_time()
            if self._start_dettime < 0:
                self._start_dettime = self.get_start_dettime()

            obj = self.get_incumbent_objective_value() if has_incumbent else None
            time = self.get_time() - self._start_time
            det_time = self.get_dettime() - self._start_dettime

            pdata = ProgressData(self._count, has_incumbent,
                                 obj, self.get_best_objective_value(),
                                 self.get_MIP_relative_gap(),
                                 self.get_num_iterations(),
                                 self.get_num_nodes(),
                                 self.get_num_remaining_nodes(),
                                 time, det_time
                                 )

            solution_listeners = self._solution_listeners
            for l in self._listeners:
                try:
                    if l.accept(pdata):
                        # listener has accepted event
                        l.notify_progress(pdata)
                        l._set_current_progress_data(pdata)
                        if has_incumbent and l in solution_listeners:
                            # build a solution from incumbent values as a list of values (value[v] at position index[v])
                            cpx_incumbent_sol = self.make_solution_from_incumbents\
                                (pdata.current_objective, self.get_incumbent_values())
                            l.notify_solution(cpx_incumbent_sol)
                    # else:
                    #     print("-- rejected: #{0}, listener={1!r}".format(pdata.id, l))
                except Exception as e:
                    print('Exception raised in listener {0!r}: {1}, id={2}'.format(type(l).__name__, str(e), pdata.id))

    return ConnectListenersCallback

# internal
class _CplexSyncMode(Enum):
    InSync, InResync, OutOfSync = [1, 2, 3]



class _CplexOverwriteParametersCtx(object):
    # internal context manager to handle forcing parameters during relaxation.

    def __init__(self, cplex_to_overwrite, overwrite_param_dict):
        assert isinstance(overwrite_param_dict, dict)
        self._cplex = cplex_to_overwrite
        self._overwrite_param_dict = overwrite_param_dict
        # store current values
        cplex_params = self._cplex._env.parameters
        self._saved_param_values = {p.cpx_id: cplex_params._get(p.cpx_id) for p in overwrite_param_dict}

    def __enter__(self):
        # force overwrite values.
        cplex_params = self._cplex._env.parameters
        for p, v in six.iteritems(self._overwrite_param_dict):
            cplex_params._set(p.cpx_id, v)
        # return the Cplex instance with the overwritten parameters.
        return self._cplex

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        # whatever happened, restore saved parameter values.
        cplex_params = self._cplex._env.parameters
        for pid, saved_v in six.iteritems(self._saved_param_values):
            cplex_params._set(pid, saved_v)


class SilencedCplexContext(object):
    def __init__(self, log_out, cplex_instance, error_handler=None):
        self.cplex = cplex_instance
        self.saved_streams = None
        self.error_handler = error_handler
        self.is_silent = log_out is None

    def __enter__(self):
        self.saved_streams = CplexEngine.cpx_get_all_streams(self.cplex)
        if not self.is_silent:
            CplexEngine._cpx_set_all_streams(self.cplex, None)
        return self.cplex

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_silent:
            CplexEngine.cpx_set_all_streams(self.cplex,
                                            self.saved_streams,
                                            self.error_handler)


class IndexScope(object):  # pragma: no cover
    def __init__(self, name):
        self._name = name
        self._index = -1

    def clear(self):
        self._index = -1

    def new_index(self):
        self._index += 1
        return self._index

    def new_index_range(self, size):
        first = self._index + 1
        last = first + size
        self._index += size
        return fast_range(first, last)

    def notify_deleted(self, deleted_index):
        if deleted_index >= 0:
            self._index -= 1

    def notify_deleted_block(self, deleted_indices):
        self._index -= len(deleted_indices)

    def __str__(self):  # pragma: no cover
        return 'IndexScope({0})[{1}]'.format(self._name, self._index)


# noinspection PyProtectedMember
class CplexEngine(IEngine):
    """
        CPLEX engine wrapper.
    """
    CPX_RANGE_SYMBOL = 'R'

    fix_multiobj_error_1300 = True

    procedural = True # not static_is_post1210

    cplex_error_re = re.compile(r'CPLEX Error\s+(\d+)')


    @classmethod
    def _has_multi_objective(cls, cpx):
        # returns True if two conditions are satisfied:
        # 1. this version of CPLEX does support multi-objective
        # 2. this instance has several objectives.
        return hasattr(cpx, 'multiobj') and cpx.multiobj.get_num() > 1

    def supports_logical_constraints(self):
        ok = self._cpx_version_as_tuple >= (12, 8, 0) and self.cpx_adapter.supports_typed_indicators
        msg = 'Logical constraints require CPLEX version 12.8 or above, this is CPLEX version: {0}'.format(
            self._cplex.get_version()) if not ok else None
        return ok, msg

    def supports_multi_objective(self):
        ok = (self.cpx_adapter.multiobjsetobj is not None)
        msg = 'Multi-objectives require CPLEX version 12.9 or above, this is CPLEX version: {0}'.format(
            self._cplex.get_version()) if not ok else None
        return ok, msg

    def solved_as_mip(self):
        # INTERNAL
        cpx_cst = self.cpx_adapter.cpx_cst
        return self._cplex.get_problem_type() not in\
               {cpx_cst.CPXPROB_LP, cpx_cst.CPXPROB_QP, cpx_cst.CPXPROB_QCP}

    def solved_as_lp(self):
        # INTERNAL
        cpx = self._cplex
        cpx_cst = self.cpx_adapter.cpx_cst
        return self.cpx_adapter.getprobtype(cpx._env._e, cpx._lp) == cpx_cst.CPXPROB_LP


    @staticmethod
    def allocate_one_index_return(ret_value, scope, expect_range):
        return ret_value[-1] if expect_range else ret_value

    @staticmethod
    def allocate_one_index_guess(ret_value, scope, expect_range):
        return scope.new_index()  # pragma: no cover

    @staticmethod
    def allocate_range_index_return(size, ret_value, scope):
        return ret_value

    # noinspection PyUnusedLocal
    @staticmethod
    def allocate_range_value_guess(size, ret_value, scope):  # pragma: no cover
        return scope.new_index_range(size)

    def fatal(self, *args):
        self._model.fatal(*args)

    def _initialize_constants_from_cplex(self):
        cpx_adapter = self.cpx_adapter
        cplex_module = cpx_adapter.cplex_module
        cpx_cst = cpx_adapter.cpx_cst

        try:
            self.setlonganno = cplex_module._internal._procedural.setlonganno
            self.annotation_map = {CplexScope.VAR_SCOPE: cpx_cst.CPX_ANNOTATIONOBJ_COL,
                                   CplexScope.LINEAR_CT_SCOPE: cpx_cst.CPX_ANNOTATIONOBJ_ROW,
                                   CplexScope.IND_CT_SCOPE: cpx_cst.CPX_ANNOTATIONOBJ_IND,
                                   CplexScope.QUAD_CT_SCOPE: cpx_cst.CPX_ANNOTATIONOBJ_QC,
                                   CplexScope.SOS_SCOPE: cpx_cst.CPX_ANNOTATIONOBJ_SOS}
        except AttributeError:  # pragma: no cover
            self.setlonganno = None
            self.annotation_map = {}

        # initialize default constants for multiobj
        try: 
            # from 12.9 up
            self.DEFAULT_CPX_NO_WEIGHT_CHANGE = cpx_cst.CPX_NO_WEIGHT_CHANGE
            self.DEFAULT_CPX_NO_PRIORITY_CHANGE = cpx_cst.CPX_NO_PRIORITY_CHANGE
            self.DEFAULT_CPX_NO_ABSTOL_CHANGE = cpx_cst.CPX_NO_ABSTOL_CHANGE
            self.DEFAULT_CPX_NO_RELTOL_CHANGE = cpx_cst.CPX_NO_RELTOL_CHANGE

            self.DEFAULT_CPX_STAT_MULTIOBJ_OPTIMAL = cpx_cst.CPX_STAT_MULTIOBJ_OPTIMAL
            self.DEFAULT_CPX_STAT_MULTIOBJ_INForUNBD = cpx_cst.CPX_STAT_MULTIOBJ_INForUNBD
            self.DEFAULT_CPX_STAT_MULTIOBJ_UNBOUNDED = cpx_cst.CPX_STAT_MULTIOBJ_UNBOUNDED
            self.DEFAULT_CPX_STAT_MULTIOBJ_INFEASIBLE = cpx_cst.CPX_STAT_MULTIOBJ_INFEASIBLE
            self.DEFAULT_CPX_STAT_MULTIOBJ_NON_OPTIMAL = cpx_cst.CPX_STAT_MULTIOBJ_NON_OPTIMAL
            self.DEFAULT_CPX_STAT_MULTIOBJ_STOPPED = cpx_cst.CPX_STAT_MULTIOBJ_STOPPED

        except AttributeError:  # pragma: no cover
            self.DEFAULT_CPX_NO_WEIGHT_CHANGE = None
            self.DEFAULT_CPX_NO_PRIORITY_CHANGE = None
            self.DEFAULT_CPX_NO_ABSTOL_CHANGE = None
            self.DEFAULT_CPX_NO_RELTOL_CHANGE = None

            self.DEFAULT_CPX_STAT_MULTIOBJ_OPTIMAL = None
            self.DEFAULT_CPX_STAT_MULTIOBJ_INForUNBD = None
            self.DEFAULT_CPX_STAT_MULTIOBJ_UNBOUNDED = None
            self.DEFAULT_CPX_STAT_MULTIOBJ_INFEASIBLE = None
            self.DEFAULT_CPX_STAT_MULTIOBJ_NON_OPTIMAL = None
            self.DEFAULT_CPX_STAT_MULTIOBJ_STOPPED = None

        # solve status
        self._CPLEX_SOLVE_OK_STATUSES = {cpx_cst.CPX_STAT_OPTIMAL,
                                         cpx_cst.CPX_STAT_NUM_BEST,  # solution exists but numerical issues
                                         cpx_cst.CPX_STAT_FIRSTORDER,  # stting optimlaitytarget to 2
                                         cpx_cst.CPXMIP_OPTIMAL,
                                         cpx_cst.CPXMIP_OPTIMAL_TOL,
                                         cpx_cst.CPXMIP_SOL_LIM,
                                         cpx_cst.CPXMIP_NODE_LIM_FEAS,
                                         cpx_cst.CPXMIP_TIME_LIM_FEAS,
                                         cpx_cst.CPXMIP_DETTIME_LIM_FEAS,  # cf issue #61
                                         cpx_cst.CPXMIP_MEM_LIM_FEAS,
                                         cpx_cst.CPXMIP_FAIL_FEAS,
                                         cpx_cst.CPXMIP_ABORT_FEAS,
                                         cpx_cst.CPXMIP_FAIL_FEAS_NO_TREE,  # integer sol exists (????)
                                         cpx_cst.CPXMIP_OPTIMAL_POPULATED,
                                         cpx_cst.CPXMIP_OPTIMAL_POPULATED_TOL,
                                         cpx_cst.CPXMIP_POPULATESOL_LIM,
                                         301,  # cpx_cst.CPX_STAT_MULTIOBJ_OPTIMAL,
                                         305  # cpx_cst.CPX_STAT_MULTIOBJ_NON_OPTIMAL
                                         }
        # relaxer status
        self._CPLEX_RELAX_OK_STATUSES = frozenset({cpx_cst.CPX_STAT_FEASIBLE,
                                                   cpx_cst.CPXMIP_OPTIMAL_RELAXED_INF,
                                                   cpx_cst.CPXMIP_OPTIMAL_RELAXED_SUM,
                                                   cpx_cst.CPXMIP_OPTIMAL_RELAXED_QUAD,
                                                   cpx_cst.CPXMIP_FEASIBLE_RELAXED_INF,
                                                   cpx_cst.CPXMIP_FEASIBLE_RELAXED_QUAD,
                                                   cpx_cst.CPXMIP_FEASIBLE_RELAXED_SUM,
                                                   cpx_cst.CPX_STAT_FEASIBLE_RELAXED_SUM,
                                                   cpx_cst.CPX_STAT_FEASIBLE_RELAXED_INF,
                                                   cpx_cst.CPX_STAT_FEASIBLE_RELAXED_QUAD,
                                                   cpx_cst.CPX_STAT_OPTIMAL_RELAXED_INF,
                                                   cpx_cst.CPX_STAT_OPTIMAL_RELAXED_SUM,
                                                   cpx_cst.CPXMIP_ABORT_RELAXED
                                                   }).union(self._CPLEX_SOLVE_OK_STATUSES)

        self.fast_add_logicals = self.fast_add_logicals12100 if cpx_adapter.is_post1210 else self.fast_add_logicals1290
        self._fast_add_piecewise_constraint = self._fast_add_piecewise_constraint12100 if cpx_adapter.is_post1210 else self._fast_add_piecewise_constraint1290


    def __init__(self, mdl, **kwargs):
        super(CplexEngine, self).__init__()

        # now we load cplex either from default 'import cplex'
        # (coslocation==None)
        # or from a COS location set in context
        coslocation = None  # default: look in system
        context = kwargs.get('context', None)
        if context is not None:
            coslocation = context.cos.location

        cpx_adapter = CplexAdapter(coslocation, procedural=self.procedural)
        self.cpx_adapter = cpx_adapter

        self._initialize_constants_from_cplex()

        cpx = cpx_adapter.cpx
        self._cplex_location = cpx_adapter.cplex_location
        cpxv = cpx.get_version()

        # resetting DATACHECK to 0 has no measurable effect
        # cpx.parameters._set(1056, 0)

        self._model = mdl
        self._saved_log_output = True  # initialization from model is deferred (pickle)

        self._cpx_version_as_tuple = tuple(float(x) for x in cpx.get_version().split("."))

        if self._cpx_version_as_tuple >= (12, 7):
            # use returned values from Python Cplex from 12.7 onwards
            self._allocate_one_index = self.allocate_one_index_return
            self._allocate_range_index = self.allocate_range_index_return
        else:  # pragma: no cover
            # 12.6 did not return anything, so we had to guess.
            self._allocate_one_index = self.allocate_one_index_return
            self._allocate_range_index = self.allocate_range_index_return

        # deferred bounds changes, as dicts {var: num}
        self._var_lb_changed = {}
        self._var_ub_changed = {}

        self._cplex = cpx

        #self._solve_count = 0
        self._last_solve_status = None
        self._last_solve_details = None

        # for unpickling, remember to resync with model
        self._sync_mode = _CplexSyncMode.InSync

        # remember truly allocated indices
        self._lincts_scope = IndexScope(name='lincts')
        self._indcst_scope = IndexScope(name='indicators')
        self._quadcst_scope = IndexScope(name='quadcts')
        self._pwlcst_scope = IndexScope(name='piecewises')
        self._vars_scope = IndexScope(name='vars')

        # index of benders long annotation
        self._benders_anno_idx = -1

        # callback connector
        self._ccb = None

    def _mark_as_out_of_sync(self):
        self._sync_mode = _CplexSyncMode.OutOfSync

    @classmethod
    def _cpx_set_all_streams(cls, cpx, ofs):
        cpx.set_log_stream(ofs)
        cpx.set_results_stream(ofs)
        cpx.set_error_stream(ofs)
        cpx.set_warning_stream(ofs)

    @classmethod
    def cpx_get_all_streams(cls, cpx):
        # returns an array of streams in the order: log, result, error, warning
        streams = [cpx._env._get_log_stream(),
                   cpx._env._get_results_stream(),
                   cpx._env._get_error_stream(),
                   cpx._env._get_warning_stream()]
        return [x._file if hasattr(x, '_file') else None for x in streams]

    @classmethod
    def cpx_set_all_streams(cls, cpx, streams, error_handler):
        if len(streams) != 4:
            error_handler.fatal("Wrong number of streams, should be 4: {0!s}", len(streams))
        else:
            cpx.set_log_stream(streams[0])
            cpx.set_results_stream(streams[1])
            cpx.set_error_stream(streams[2])
            cpx.set_warning_stream(streams[3])

    def set_streams(self, outs):
        self_log_output = self._saved_log_output
        if self_log_output != outs:
            self._cpx_set_all_streams(self._cplex, outs)
            self._saved_log_output = outs

    def get_var_index(self, dvar):  # pragma: no cover
        self._resync_if_needed()
        dvar_name = dvar.name
        if not dvar_name:
            self.error_handler.fatal("cannot query index for anonymous object: {0!s}", (dvar,))
        else:
            return self._cplex.variables.get_indices(dvar_name)

    def get_ct_index(self, ct):  # pragma: no cover
        self._resync_if_needed()
        ctname = ct.name
        if not ctname:
            self.error_handler.fatal("cannot query index for anonymous constraint: {0!s}", (ct,))
        self_cplex = self._cplex
        ctscope = ct.cplex_scope
        if ctscope is CplexScope.LINEAR_CT_SCOPE:
            return self_cplex.linear_constraints.get_indices(ctname)
        elif ctscope is CplexScope.IND_CT_SCOPE:
            return self_cplex.indicator_constraints.get_indices(ctname)
        elif ctscope is CplexScope.QUAD_CT_SCOPE:
            return self_cplex.quadratic_constraints.get_indices(ctname)
        elif ctscope is CplexScope.PWL_CT_SCOPE:
            return self_cplex.pwl_constraints.get_indices(ctname)

        else:
            self.error_handler.fatal("unrecognized constraint to query index: {0!s}", ct)

    @classmethod
    def sync_data_differ_stop_here(cls, cpx_data, mdl_data):
        # put breakpoint here
        pass

    def _check_one_constraint_index(self, cpx_linear, ct, prec=1e-6):
        def sparse_to_terms(indices_, koefs_):
            terms = [(ix, k) for ix, k in izip(indices_, koefs_)]
            terms.sort(key=lambda  t: t[0])
            return terms

        # assert idx > 0
        cpx_row = cpx_linear.get_rows(ct.index)
        cpx_terms = sparse_to_terms(cpx_row.ind, cpx_row.val)
        mdl_idxs, mdl_coefs = self.linear_ct_to_cplex(ct)
        mdl_terms = sparse_to_terms(mdl_idxs, mdl_coefs)
        assert len(cpx_terms) == len(mdl_terms)
        for cpxt, mdlt in izip(cpx_terms, mdl_terms):
            assert cpxt[0] == mdlt[0]
            assert abs(cpxt[1] - mdlt[1]) <= prec

    def check_constraint_indices(self, cts, ctscope):
        mdl = self._model
        interface = self._scope_interface(ctscope)
        cpx_num = interface.get_num()
        l_cts = list(cts)
        if len(l_cts) != cpx_num:
            mdl.error("Sizes differ: cplex: {0}, docplex: {1}".format(cpx_num, len(l_cts)))
        for c in range(cpx_num):
            mdl_ct = l_cts[c]
            try:
                cpx_name = interface.get_names(c)
            except self.cpx_adapter.CplexSolverError:
                cpx_name = None
            mdl_name = mdl_ct.name
            if mdl_name and cpx_name != mdl_name:
                self.sync_data_differ_stop_here(cpx_name, mdl_name)
                mdl.error("Names differ: index: {0}, cplex: {1}, docplex: {2}".format(c, cpx_name, mdl_name))

            if hasattr(interface, "get_rhs"):
                cpx_rhs = interface.get_rhs(c)
                mdl_rhs = mdl_ct.cplex_num_rhs()
                if abs(cpx_rhs - mdl_rhs) >= 1e-6 :
                    self.sync_data_differ_stop_here(cpx_rhs, mdl_rhs)
                    mdl.error("RHS differ: index: {0}, cplex: {1}, docplex: {2}".format(c, cpx_rhs, mdl_rhs))
        if cpx_num and ctscope == CplexScope.LINEAR_CT_SCOPE:
            full_check_indices = set()
            full_check_indices.add(0)
            if cpx_num > 1:
                full_check_indices.add(cpx_num-1)
            if cpx_num >= 4:
                full_check_indices.add(int(cpx_num/2))
            cpxlinear = interface
            for j in full_check_indices:
                ct = mdl.get_constraint_by_index(j)
                self._check_one_constraint_index(cpxlinear, ct)


    def check_var_indices(self, dvars):  # pragma: no cover
        for dvar in dvars:
            # assuming dvar has a name
            model_index = dvar.index
            cpx_index = self.get_var_index(dvar)
            if model_index != cpx_index:  # pragma: nocover
                self._model.error("indices differ, obj: {0!s}, docplex={1}, CPLEX={2}", dvar, model_index,
                                  cpx_index)

    @property
    def error_handler(self):
        return self._model.error_handler

    def get_cplex(self):
        """
        Returns the underlying CPLEX object
        :return:
        """
        return self._cplex

    def get_cplex_location(self):
        return self._cplex_location

    def get_infinity(self):
        return self.cpx_adapter.cplex_module.infinity

    def _create_cpx_multitype_vartype_list(self, vartypes):
        # vartypes is a list of model variable types
        # if all continuous return []
        if all(mvt.cplex_typecode == 'C' for mvt in vartypes):
            return ""
        else:
            # return a list of 'B', 'C', 'I' symbols
            return "".join(mvt.cplex_typecode for mvt in vartypes)

    @classmethod
    def compute_cpx_vartype(cls, vartype, size):
        if vartype == 'C':
            return ''
        else:
            return vartype * size

    def create_one_variable(self, vartype, lb, ub, name):
        lbs = [float(lb)]
        ubs = [float(ub)]
        names = [name]
        indices = self.create_variables(1, vartype, lbs, ubs, names)
        assert 1 == len(indices)
        return indices[0]

    def create_variables(self, nb_vars, vartype, lbs, ubs, names):
        self._resync_if_needed()
        cpx_types = self.compute_cpx_vartype(vartype.cplex_typecode, nb_vars)
        if not cpx_types:
            if not (lbs or ubs):
                # force at least one list with correct size.
                lbs = [0] * nb_vars
        return self._create_cpx_variables(nb_vars, cpx_types, lbs, ubs, names)

    def create_multitype_variables(self, nb_vars, vartypes, lbs, ubs, names):
        self._resync_if_needed()
        cpx_types = self._create_cpx_multitype_vartype_list(vartypes)
        return self._create_cpx_variables(nb_vars, cpx_types, lbs, ubs, names)

    def _create_cpx_variables(self, nb_vars, cpx_vartypes, lbs, ubs, names):
        ret_add = self.fast_add_cols(cpx_vartypes, lbs, ubs, names)
        return self._allocate_range_index(size=nb_vars, ret_value=ret_add, scope=self._vars_scope)

    def _apply_var_fn(self, dvars, args, setter_fn, getter_fn=None):
        cpxvars = self._cplex.variables

        indices = [_v.index for _v in dvars]
        # noinspection PyArgumentList
        setter_fn(cpxvars, izip(indices, args))
        if getter_fn:
            return getter_fn(cpxvars, indices)
        else:
            return None

    # TODO: to be removed, does not seem to be used ?
    # _getset_map = {"lb": (cplex._internal._subinterfaces.VariablesInterface.set_lower_bounds,
    #                      cplex._internal._subinterfaces.VariablesInterface.get_lower_bounds),
    #               "ub": (cplex._internal._subinterfaces.VariablesInterface.set_upper_bounds,
    #                      cplex._internal._subinterfaces.VariablesInterface.get_upper_bounds),
    #               "name": (cplex._internal._subinterfaces.VariablesInterface.set_names,
    #                        cplex._internal._subinterfaces.VariablesInterface.get_names)}

    def rename_var(self, dvar, new_name):
        #self._cplex.variables.set_names([(dvar.index, new_name or "")])
        self._fast_set_col_name(dvar._index, new_name)

    def fast_set_var_types(self, dvars, vartypes):
        cpx = self._cplex
        cpx_adapter = self.cpx_adapter
        cpx_adapter.chgctype(cpx._env._e, cpx._lp,
                             [dv.index for dv in dvars],
                             "".join(vt.cplex_typecode for vt in vartypes)
                             )

    def change_var_types(self, dvars, newtypes):  # pragma: no cover
        if self.procedural:
            self.fast_set_var_types(dvars, newtypes)
        else:
            # noinspection PyArgumentList
            sparses = [(dv.index, vt.cplex_typecode) for (dv, vt) in izip(dvars, newtypes)]
            self._cplex.variables.set_types(sparses)

    def set_var_lb(self, dvar, lb):
        self._resync_if_needed()
        self_var_lbs = self._var_lb_changed
        self_var_lbs[dvar] = float(lb)

    def set_var_ub(self, dvar, ub):
        self._resync_if_needed()
        self_var_ubs = self._var_ub_changed
        self_var_ubs[dvar] = float(ub)  # force float here: numpy types will crash

    def make_attribute_map_from_scope_fn(self, mdl, cplexfn, scope):
        # transforms an array of cplex values into a map
        # using the scope object as a mapper
        all_values = cplexfn()
        return self.make_attribute_map_from_scope_list(mdl, all_values, scope)

    @classmethod
    def make_attribute_map_from_scope_list(cls, mdl, values, scope, keep_zeros=False):
        value_map = {}
        for ix, cplex_value in enumerate(values):
            mobj = scope(ix)
            if mobj is None:
                mdl.error("No {0} with index: {1} - caution".format(scope.qualifier, ix))
            elif keep_zeros or cplex_value:
                value_map[mobj] = cplex_value

        return value_map

    def get_all_reduced_costs(self, mdl):
        return self.make_attribute_map_from_scope_fn(mdl, self._cplex.solution.get_reduced_costs, mdl._var_scope)

    def get_all_dual_values(self, mdl):
        return self.make_attribute_map_from_scope_fn(mdl, self._cplex.solution.get_dual_values, mdl._linct_scope)

    def get_all_slack_values(self, mdl):
        lin_slacks  = self.make_attribute_map_from_scope_fn(mdl, self._cplex.solution.get_linear_slacks,    mdl._linct_scope)
        quad_slacks = self.make_attribute_map_from_scope_fn(mdl, self._cplex.solution.get_quadratic_slacks, mdl._quadct_scope)
        ind_slacks  = self.make_attribute_map_from_scope_fn(mdl, self._cplex.solution.get_indicator_slacks, mdl._logical_scope)
        # dict : cplex_scope -> dict from obj to slack
        return {CplexScope.LINEAR_CT_SCOPE: lin_slacks,
                CplexScope.QUAD_CT_SCOPE: quad_slacks,
                CplexScope.IND_CT_SCOPE: ind_slacks}

    def get_basis(self, mdl):
        try:
            status_vars, status_licnts = self._cplex.solution.basis.get_basis()
            var_statuses_map = self.make_attribute_map_from_scope_list(mdl, status_vars, mdl._var_scope, keep_zeros=True)
            status_linearct_map = self.make_attribute_map_from_scope_list(mdl, status_licnts, mdl._linct_scope, keep_zeros=True)
            return var_statuses_map, status_linearct_map
        except self.cpx_adapter.CplexError as cpxe:
            if cpxe.args[2] == 1262:  # code 1262 is "no basis exists"
                return {}, {}
            else:  # pragma: no cover
                raise

    # the returned list MUST be of size 2 otherwise the wrapper will crash.
    _trivial_linexpr = [[], []]

    @classmethod
    def linear_ct_to_cplex(cls, linear_ct):
        # INTERNAL
        return cls.make_cpx_linear_from_exprs(linear_ct.get_left_expr(), linear_ct.get_right_expr())

    def make_cpx_linear_from_one_expr(self, expr):
        return self.make_cpx_linear_from_exprs(left_expr=expr, right_expr=None)

    @classmethod
    def make_cpx_linear_from_exprs(cls, left_expr, right_expr):
        indices = []
        coefs = []
        if right_expr is None or right_expr.is_constant():
            nb_terms = left_expr.number_of_terms()
            if nb_terms:
                indices = [-1] * nb_terms
                coefs = [0.0] * nb_terms
                for i, (dv, k) in enumerate(left_expr.iter_terms()):
                    indices[i] = dv._index
                    coefs[i] = float(k)

        elif left_expr.is_constant():
            nb_terms = right_expr.number_of_terms()
            if nb_terms:
                indices = [-1] * nb_terms
                coefs = [0] * nb_terms
                for i, (dv, k) in enumerate(right_expr.iter_terms()):
                    indices[i] = dv._index
                    coefs[i] = -float(k)

        else:
            # hard to guess array size here:
            # we could allocate size(left) + size(right) and truncate, but??
            for dv, k in BinaryConstraint._generate_net_linear_coefs2_unsorted(left_expr, right_expr):
                indices.append(dv._index)
                coefs.append(float(k))

        # all_indices_coefs is a list of  (index, coef) 2-tuples
        if indices:
            # CPLEX requires two lists: one for indices, one for coefs
            # we use zip to unzip the tuples
            return [indices, coefs]
        else:
            # the returned list MUST be of size 2 otherwise the wrapper will crash.
            return cls._trivial_linexpr

    @staticmethod
    def make_cpx_ct_rhs_from_exprs(left_expr, right_expr):
        return right_expr.get_constant() - left_expr.get_constant()

    def __index_problem_stop_here(self):
        #  put a breakpoint here if index problems occur
        pass  # pragma: no cover

    def _make_cplex_linear_ct(self, cpx_lin_expr, cpx_sense, rhs, name):
        # INTERNAL
        cpx_rhs = [float(rhs)]  # if not a float, cplex crashes baaaadly
        cpxnames = [name] if name else []
        cpx_adapter = self.cpx_adapter
        if self.procedural and cpx_adapter.chbmatrix:
            ret_add = self.cpx_adapter.fast_add_linear(self._cplex, cpx_lin_expr, cpx_sense, cpx_rhs, cpxnames)
        else:
            ret_add = self._cplex.linear_constraints.add(cpx_lin_expr, cpx_sense, cpx_rhs, names=cpxnames)
        return self._allocate_one_index(ret_value=ret_add, scope=self._lincts_scope, expect_range=True)

    def create_linear_constraint(self, binaryct):
        self._resync_if_needed()
        cpx_linexp1 = self.linear_ct_to_cplex(binaryct)
        # wrap one more time
        cpx_linexp = [cpx_linexp1] if cpx_linexp1 else []
        # returns a number
        num_rhs = binaryct.cplex_num_rhs()
        return self._make_cplex_linear_ct(cpx_lin_expr=cpx_linexp,
                                          cpx_sense=binaryct.cplex_code(),
                                          rhs=num_rhs, name=binaryct.name)

    def create_block_linear_constraints(self, linct_seq):
        cpx_adapter = self.cpx_adapter
        self._resync_if_needed()
        block_size = len(linct_seq)
        # noinspection PyPep8
        cpx_rhss = [ct.cplex_num_rhs() for ct in linct_seq]
        cpx_sense_string = "".join(ct.cplex_code() for ct in linct_seq)
        if self._model.ignore_names:
            cpx_names = []
        else:
            cpx_names = [ct.safe_name for ct in linct_seq]
        cpx_convert_fn = self.linear_ct_to_cplex
        cpx_linexprs = [cpx_convert_fn(ct) for ct in linct_seq]
        # peek for ranges?
        has_ranges = any(ct.cplex_range_value() for ct in linct_seq)
        cpx_range_values = [ct.cplex_range_value() for ct in linct_seq] if has_ranges else None
        if self.procedural:
            ret_add = cpx_adapter.fast_add_linear(self._cplex, cpx_linexprs, cpx_sense_string, cpx_rhss, cpx_names, ranges=cpx_range_values)
        else:
            ret_add = self._cplex.linear_constraints.add(cpx_linexprs, cpx_sense_string, cpx_rhss,
                                                         range_values=cpx_range_values, names=cpx_names)
        return self._allocate_range_index(size=block_size, ret_value=ret_add, scope=self._lincts_scope)

    def create_range_constraint(self, range_ct):
        self._resync_if_needed()

        expr = range_ct.expr

        cpx_linexpr = self.make_cpx_linear_from_one_expr(expr)
        cpx_linexpr2 = [cpx_linexpr] if cpx_linexpr else []

        cpx_rhs = [float(range_ct.cplex_num_rhs())]
        cpx_range_values = [float(range_ct.cplex_range_value())]
        cpx_names = [range_ct.safe_name]

        if self.procedural:
            cpx_adapter = self.cpx_adapter
            ret_add = cpx_adapter.fast_add_linear(self._cplex, cpx_linexpr2,
                                                  range_ct.cplex_code(),
                                                  cpx_rhs, cpx_names,
                                                  ranges=cpx_range_values)
        else:
            linearcts = self._cplex.linear_constraints
            ret_add = linearcts.add(lin_expr=cpx_linexpr2,
                                    senses=range_ct.cplex_code(),
                                    rhs=cpx_rhs,
                                    range_values=cpx_range_values,
                                    names=cpx_names)
        return self._allocate_one_index(ret_value=ret_add, scope=self._lincts_scope, expect_range=True)

    def rename_linear_constraint(self, linct, new_name):
        safe_new_name = new_name or ""
        linct_index = linct.index
        cpxlinears = self._cplex.linear_constraints
        cpxlinears.set_names([(linct_index, safe_new_name)])

    def create_logical_constraint(self, logct, is_equivalence):
        ret = self.create_batch_logical_constraints([logct], is_equivalence=is_equivalence)
        return ret[0]

    # constant sused for logical cts
    CPX_IF_TYPE = 1
    CPX_IF_ONLYIF_TYPE = 3

    def create_batch_logical_constraints(self, logcts, is_equivalence):
        cpx_adapter = self.cpx_adapter
        self._resync_if_needed()
        if self.procedural and cpx_adapter.chbmatrix:
            return self.fast_add_logicals(logcts, is_equivalence=is_equivalence)
        else:
            cpx_indicators = self._cplex.indicator_constraints
            if hasattr(cpx_indicators, "add_batch"):
                # use non-procedural batch api
                nb_logicals = len(logcts)
                cpx_linexprs = [self.linear_ct_to_cplex(logct.linear_constraint) for logct in logcts]
                cpx_sense = ''.join(logct.linear_constraint.cplex_code() for logct in logcts)

                cpx_rhss = [logct.linear_constraint.cplex_num_rhs() for logct in logcts]
                cpx_binvar_indices = [logct.binary_var.safe_index for logct in logcts]
                cpx_names = [lct.safe_name for lct in logcts]
                cpx_complemented = [lct.cpx_complemented for lct in logcts]
                cpx_indtypes = [self.CPX_IF_ONLYIF_TYPE] * nb_logicals if is_equivalence else [self.CPX_IF_TYPE] * nb_logicals
                r = cpx_indicators.add_batch(cpx_linexprs, cpx_sense, cpx_rhss, cpx_binvar_indices, cpx_complemented,
                                         cpx_names, cpx_indtypes)
                return self._allocate_range_index(nb_logicals, r, self._indcst_scope)

            else:
                # use non-batch, non-procedural
                return [self._create_typed_indicator_internal(cpx_indicators, log_ct.binary_var,
                                                              log_ct.get_linear_constraint(),
                                                              equivalence=is_equivalence,
                                                              cpx_complemented=log_ct.cpx_complemented,
                                                              name=log_ct.safe_name)
                        for log_ct in logcts]


    def create_batch_equivalence_constraints(self, eqcts):
        return self.create_batch_logical_constraints(eqcts, is_equivalence=True)

    def create_batch_indicator_constraints(self, inds):
        return self.create_batch_logical_constraints(inds, is_equivalence=False)

    def fast_add_logicals1290(self, logcts, is_equivalence):
        cpx_adapter = self.cpx_adapter
        cpx = self._cplex
        old_nb_indicators = cpx.indicator_constraints.get_num()
        cpxenv = cpx._env
        cpx_indvars = []
        cpx_linexprs =[]
        cpx_senses = []
        cpx_rhss = []
        cpx_complemented = []
        cpx_indtype = cpx_adapter.cpx_indicator_type_equiv if is_equivalence else cpx_adapter.cpx_indicator_type_ifthen

        cpx_names =[]
        nb_logicals = 0
        for logct in logcts:
            cpx_indvars.append(logct.binary_var.safe_index)
            lct = logct.linear_constraint
            cpx_linexprs.append(self.linear_ct_to_cplex(lct))
            cpx_senses.append(lct.cplex_code())
            cpx_rhss.append(lct.cplex_num_rhs())
            cpx_complemented.append(logct.cpx_complemented)
            cpx_names.append(logct.name) #  or '')
            nb_logicals += 1

        cpx_indtypes = [cpx_indtype] * nb_logicals

        with cpx_adapter.chbmatrix(cpx_linexprs, cpx._env_lp_ptr, 0,
                                   cpxenv._apienc) as (linmat, nnz):
            cpx_adapter.addindconstr(cpxenv._e, cpx._lp,
                                     nb_logicals, cpx_indvars,
                                     cpx_complemented, cpx_rhss, ''.join(cpx_senses),
                                     linmat, cpx_indtypes,
                                     cpx_names, nnz,
                                     cpxenv._apienc)
        return fast_range(old_nb_indicators, cpx.indicator_constraints.get_num())

    def fast_add_logicals12100(self, logcts, is_equivalence):
        cpx_adapter = self.cpx_adapter
        cpx = self._cplex
        old_nb_indicators = cpx.indicator_constraints.get_num()
        cpxenv = cpx._env
        cpx_indvars = []
        cpx_linexprs =[]
        cpx_senses = []
        cpx_rhss = []
        cpx_complemented = []
        cpx_indtype = cpx_adapter.cpx_indicator_type_equiv if is_equivalence else cpx_adapter.cpx_indicator_type_ifthen

        cpx_names =[]
        nb_logicals = 0
        for logct in logcts:
            cpx_indvars.append(logct.binary_var.safe_index)
            lct = logct.linear_constraint
            cpx_linexprs.append(self.linear_ct_to_cplex(lct))
            cpx_senses.append(lct.cplex_code())
            cpx_rhss.append(lct.cplex_num_rhs())
            cpx_complemented.append(logct.cpx_complemented)
            cpx_names.append(logct.name) #  or '')
            nb_logicals += 1

        cpx_indtypes = [cpx_indtype] * nb_logicals

        # noinspection PyArgumentList
        with cpx_adapter.chbmatrix(cpx_linexprs, cpx._env_lp_ptr, 0) as (linmat, nnz):
            cpx_adapter.addindconstr(cpxenv._e, cpx._lp,
                                     nb_logicals, cpx_indvars,
                                     cpx_complemented, cpx_rhss, ''.join(cpx_senses),
                                     linmat, cpx_indtypes,
                                     cpx_names, nnz)
        return fast_range(old_nb_indicators, cpx.indicator_constraints.get_num())


    def _create_typed_indicator_internal(self, cpx_ind, indvar, linct,
                                         equivalence,
                                         cpx_complemented,
                                         name):
        assert name is not None
        cpx_linexpr = self.linear_ct_to_cplex(linct)
        cpx_sense = linct.cplex_code()
        indvar_index = indvar.safe_index
        cpx_rhs = linct.cplex_num_rhs()
        if equivalence:
            return cpx_ind.add(cpx_linexpr, cpx_sense, cpx_rhs, indvar_index, cpx_complemented, name, indtype=3)
        else:
            return cpx_ind.add(cpx_linexpr, cpx_sense, cpx_rhs, indvar_index, cpx_complemented, name)


    def create_quadratic_constraint(self, qct):
        self._resync_if_needed()
        # ---
        self_cplex = self._cplex
        float_rhs = qct.cplex_num_rhs()
        cpx_sense = qct.cplex_code()
        # see RTC-31772, None is accepted from 12.6.3.R0 onward. use get_name() when dropping compat for 12.6.2
        qctname = qct.safe_name

        # linear part
        net_linears = [(lv._index, float(lk)) for lv, lk in qct.iter_net_linear_coefs()]
        list_linears = list(izip(*net_linears))
        if not list_linears:
            list_linears = [(0,), (0.0,)]  # always non empty
        # build a list of three lists: [qv1.index], [qv2.index], [qk..]

        net_quad_triplets = [(qvp[0]._index, qvp[1]._index, float(qk)) for qvp, qk in qct.iter_net_quads()]
        if net_quad_triplets:
            list_quad_triplets = list(izip(*net_quad_triplets))
            ret_add = self_cplex.quadratic_constraints.add(lin_expr=list_linears,
                                                           quad_expr=list_quad_triplets,
                                                           sense=cpx_sense,
                                                           rhs=float_rhs,
                                                           name=qctname)
            return self._allocate_one_index(ret_value=ret_add, scope=self._quadcst_scope, expect_range=False)
        else:
            # actually a linear constraint
            return self._make_cplex_linear_ct(cpx_lin_expr=[list_linears],
                                              cpx_sense=qct.cplex_code(),
                                              rhs=float_rhs, name=qctname)

    def create_pwl_constraint(self, pwl_ct):
        """
        Post a Piecewise Linear Constraint to CPLEX
        :param pwl_ct: the PWL constraint
        :return:
        """
        self._resync_if_needed()
        try:

            pwl_func = pwl_ct.pwl_func
            pwl_def = pwl_func.pwl_def_as_breaks
            pwlctname = pwl_ct.safe_name
            x_var = pwl_ct.expr
            f_var = pwl_ct.y
            cpx_breaksx = [float(breakx) for breakx, _ in pwl_def.breaksxy]
            cpx_breaksy = [float(breaky) for _, breaky in pwl_def.breaksxy]
            n_breaks = len(cpx_breaksx)
            assert n_breaks == len(cpx_breaksy)
            if self.procedural:
                ret_add = self._fast_add_piecewise_constraint(f_var._index, x_var._index,
                                                              float(pwl_def.preslope),
                                                              cpx_breaksx, cpx_breaksy,
                                                              float(pwl_def.postslope),
                                                              name=pwlctname)
            else:
                ret_add = self._cplex.pwl_constraints.add(f_var._index, x_var._index,
                                                          float(pwl_def.preslope),
                                                          float(pwl_def.postslope),
                                                          cpx_breaksx, cpx_breaksy,
                                                          name=pwlctname)

            return self._allocate_one_index(ret_value=ret_add, scope=self._pwlcst_scope, expect_range=False)

        except AttributeError:  # pragma: no cover
            self._model.fatal("Please update Cplex to version 12.7+ to benefit from Piecewise Linear constraints.")

    def remove_constraint(self, ct):
        self._resync_if_needed()
        doomed_index = ct.safe_index
        # we have a safe index
        ctscope = ct.cplex_scope
        if ctscope is CplexScope.QUAD_CT_SCOPE:
            self._cplex.quadratic_constraints.delete(doomed_index)
            self._quadcst_scope.notify_deleted(doomed_index)
        elif ctscope is CplexScope.IND_CT_SCOPE:
            self._cplex.indicator_constraints.delete(doomed_index)
            self._indcst_scope.notify_deleted(doomed_index)
        elif ctscope is CplexScope.LINEAR_CT_SCOPE:
            self._cplex.linear_constraints.delete(doomed_index)
            self._lincts_scope.notify_deleted(doomed_index)
        elif ctscope is CplexScope.PWL_CT_SCOPE:
            self._cplex.pwl_constraints.delete(doomed_index)
            self._pwlcst_scope.notify_deleted(doomed_index)
        else:  # pragma: no cover
            raise TypeError

    def _scope_interface(self, ctscope):
        if ctscope is CplexScope.QUAD_CT_SCOPE:
            return self._cplex.quadratic_constraints
        elif ctscope is CplexScope.IND_CT_SCOPE:
            return self._cplex.indicator_constraints
        elif ctscope is CplexScope.LINEAR_CT_SCOPE:
            return self._cplex.linear_constraints
        elif ctscope is CplexScope.PWL_CT_SCOPE:
            return self._cplex.pwl_constraints
        else:
            self._model.fatal("Unexpected scope: {0}", ctscope)

    @classmethod
    def _target_from_scope(cls, scope):
        target_dict = {CplexScope.LINEAR_CT_SCOPE: lambda cpx: cpx.linear_constraints,
                       CplexScope.QUAD_CT_SCOPE: lambda cpx: cpx.quadratic_constraints,
                       CplexScope.IND_CT_SCOPE: lambda cpx: cpx.indicator_constraints,
                       CplexScope.PWL_CT_SCOPE: lambda cpx: cpx.pwl_constraints}
        return target_dict.get(scope)

    def _scope_2_index_scope(self, scope):
        if scope == CplexScope.LINEAR_CT_SCOPE:
            return self._lincts_scope
        elif scope == CplexScope.QUAD_CT_SCOPE:
            return self._quadcst_scope
        elif scope == CplexScope.IND_CT_SCOPE:
            return self._indcst_scope
        elif scope == CplexScope.PWL_CT_SCOPE:
            return self._pwlcst_scope
        else:
            raise ValueError(scope)

    def remove_constraints(self, cts):
        self._resync_if_needed()
        if cts is None:
            self._cplex.linear_constraints.delete()
            self._lincts_scope.clear()
            self._cplex.quadratic_constraints.delete()
            self._quadcst_scope.clear()
            self._cplex.indicator_constraints.delete()
            self._indcst_scope.clear()
            self._pwlcst_scope.clear()
            self._cplex.pwl_constraints.delete()
        else:
            doomed_by_scope = defaultdict(list)
            for c in cts:
                doomed_by_scope[c.cplex_scope].append(c._index)
            for scope, doomed in six.iteritems(doomed_by_scope):
                if doomed:
                    targetfn = self._target_from_scope(scope)
                    if not targetfn:
                        raise ValueError("unexpected scope value: {0}".format(scope))
                    target = targetfn(self._cplex)
                    target.delete(doomed)
                    index_scope = self._scope_2_index_scope(scope)
                    index_scope.notify_deleted_block(doomed)

            # doomed_linears = [c.safe_index for c in cts if c.cplex_scope is CplexScope.LINEAR_CT_SCOPE]
            # doomed_quadcts = [c.safe_index for c in cts if c.cplex_scope is CplexScope.QUAD_CT_SCOPE]
            # dooomed_indcts = [c.safe_index for c in cts if c.cplex_scope is CplexScope.IND_CT_SCOPE]
            # doomed_pwlcts = [c.safe_index for c in cts if c.cplex_scope is CplexScope.PWL_CT_SCOPE]
            # if doomed_linears:
            #     self._cplex.linear_constraints.delete(doomed_linears)
            #     self._lincts_scope.notify_deleted_block(doomed_linears)
            # if doomed_quadcts:
            #     self._cplex.quadratic_constraints.delete(doomed_quadcts)
            #     self._quadcst_scope.notify_deleted_block(doomed_quadcts)
            # if dooomed_indcts:
            #     self._cplex.indicator_constraints.delete(dooomed_indcts)
            #     self._indcst_scope.notify_deleted_block(dooomed_indcts)
            # if doomed_pwlcts:
            #     self._cplex.pwl_constraints.delete(doomed_pwlcts)
            #     self._pwlcst_scope.notify_deleted_block(doomed_pwlcts)

    # update
    def _unexpected_event(self, event, msg=''):  # pragma: no cover
        self.error_handler.warning('{0} Unexpected event: {1} - ignored', msg, event.name)

    # -- fast 'procedural' api
    @classmethod
    def safe_len(cls, x):
        return 0 if x is None else len(x)

    def fast_add_cols(self, cpx_vartype, lbs, ubs, names):
        # assume lbs, ubs, names are list that are either [] or have len size
        cpx = self._cplex
        cpx_e = cpx._env._e
        cpx_lp = cpx._lp
        old = cpx.variables.get_num()
        size = max(len(cpx_vartype), self.safe_len(lbs), self.safe_len(ubs), self.safe_len(names))
        self.cpx_adapter.newcols(cpx_e, cpx_lp, obj=[], lb=lbs, ub=ubs, xctype=cpx_vartype, colname=names)
        return fast_range(old, old + size)

    def _fast_set_longanno(self, anno_idx, anno_objtype, indices, groups):
        if self.setlonganno is not None:
            cpx = self._cplex
            cpx_e = cpx._env._e
            cpx_lp = cpx._lp
            self.setlonganno(cpx_e, cpx_lp, anno_idx, anno_objtype, indices, groups)

    def _fast_set_quadratic_objective(self, quad_expr):
        cpx = self._cplex
        cpx_e = cpx._env._e
        cpx_lp = cpx._lp
        for qvp, qk in quad_expr.iter_quads():
            qv1x = qvp[0]._index
            qv2x = qvp[1]._index
            obj_qk = 2 * qk if qvp.is_square() else qk
            self.cpx_adapter.chgqpcoef(cpx_e, cpx_lp, qv1x, qv2x, float(obj_qk))

    def _fast_set_linear_objective(self, linexpr):
        indices = []
        koefs = []

        for dv, k in linexpr.iter_terms():
            indices.append(dv._index)
            koefs.append(float(k))
        if indices:
            self.cpx_adapter.static_fast_set_linear_obj(self._cplex, indices, koefs)

    def _fast_set_linear_objective2(self, linexpr):
        nterms = linexpr.number_of_terms()
        if nterms:
            indices = [-1] * nterms
            koefs = [0.0] * nterms
            i = 0
            for dv, k in linexpr.iter_terms():
                indices[i] = dv._index
                koefs[i] = float(k)
                i += 1

            cpx = self._cplex
            self.cpx_adapter.chgobj(cpx._env._e, cpx._lp, indices, koefs)

    def _fast_set_linear_multiobj(self, objidx, linexpr,
                                  weight=None,
                                  priority=None,
                                  abstol=None,
                                  reltol=None,
                                  objname=None):
        # now default value for weight, priority, abstol, reltol
        # depend on cplex module
        weight = self.DEFAULT_CPX_NO_WEIGHT_CHANGE if weight is None else weight
        priority = self.DEFAULT_CPX_NO_PRIORITY_CHANGE if priority is None else priority
        abstol = self.DEFAULT_CPX_NO_ABSTOL_CHANGE if abstol is None else abstol
        reltol = self.DEFAULT_CPX_NO_RELTOL_CHANGE if reltol is None else reltol

        nterms = linexpr.number_of_terms()
        indices = [-1] * nterms
        koefs = [0] * nterms
        if nterms:
            i = 0
            for dv, k in linexpr.iter_terms():
                indices[i] = dv._index
                koefs[i] = float(k)
                i += 1

        cpx = self._cplex
        self.cpx_adapter.multiobjsetobj(cpx._env._e, cpx._lp, objidx, objind=indices, objval=koefs,
                            priority=priority, weight=weight, abstol=abstol, reltol=reltol, objname=objname)

    def _fast_update_linearct_coefs(self, ct_index, var_indices, coefs):
        assert len(var_indices) == len(coefs)

        num_coefs = len(coefs)
        cpx = self._cplex
        self.cpx_adapter.chgcoeflist(cpx._env._e, cpx._lp, [ct_index] * num_coefs, var_indices, coefs)

    def _fast_set_rhs(self, ct_index, new_rhs):
        cpx = self._cplex
        self.cpx_adapter.chgrhs(cpx._env._e, cpx._lp, (ct_index,), (new_rhs,))

    def _fast_set_col_name(self, col_index, new_name):
        cpx = self._cplex
        self.cpx_adapter.chgcolname(cpx._env._e, cpx._lp, [col_index], [new_name or ""])

    def _fast_add_piecewise_constraint1290(self, vary, varx, preslope, breaksx, breaksy, postslope, name):
        cpx = self._cplex
        cpx_env = cpx._env
        self.cpx_adapter.addpwl(cpx_env._e, cpx._lp,
                             vary, varx,
                             preslope, postslope,
                             len(breaksx), breaksx, breaksy,
                             name, cpx_env._apienc)
        # fetch number of pwls, return it -1
        pw_index = self.cpx_adapter.getnumpwl(cpx_env._e, cpx._lp) - 1
        return pw_index

    def _fast_add_piecewise_constraint12100(self, vary, varx, preslope, breaksx, breaksy, postslope, name):
        cpx = self._cplex
        cpx_env = cpx._env
        self.cpx_adapter.addpwl(cpx_env._e, cpx._lp,
                             vary, varx,
                             preslope, postslope,
                             len(breaksx), breaksx, breaksy,
                             name)
        # fetch number of pwls, return it -1
        pw_index = self.cpx_adapter.getnumpwl(cpx_env._e, cpx._lp) - 1
        return pw_index

    # ---

    def _switch_linear_expr(self, index, old_expr, new_expr):
        # INTERNAL
        # clears all linear coefs from an old expr, then set the new coefs
        old_indices = [dv._index for dv in old_expr.iter_variables()]
        old_zeros = [0] * len(old_indices)
        self._fast_update_linearct_coefs(index, old_indices, old_zeros)
        # now set new expr coefs
        cpxlin = self.make_cpx_linear_from_one_expr(expr=new_expr)
        self._fast_update_linearct_coefs(index, cpxlin[0], cpxlin[1])

    def _switch_linear_exprs2(self, ct_index, old_left, old_right, new_left, new_right):
        if old_left and old_right:
            # step 1 zap old coefs to zero (with repeats?)
            zap_indices = [dv._index for dv in old_left.iter_variables()]
            zap_indices += [dv._index for dv in old_right.iter_variables()]
        else:
            zap_indices = None  # self._fast_get_row_vars(ct_index)
        if zap_indices:
            self._fast_update_linearct_coefs(ct_index, zap_indices, [0] * len(zap_indices))
        # step 2: install new coefs
        new_ct_lin = self.make_cpx_linear_from_exprs(new_left, new_right)
        self._fast_update_linearct_coefs(ct_index, new_ct_lin[0], new_ct_lin[1])

    def update_linear_constraint(self, ct, event, *args):
        ct_index = ct.index
        assert ct_index >= 0
        assert event
        updated = False

        if event is upd.ConstraintSense:
            new_ct_cpxtype = args[0]._cplex_code
            self._cplex.linear_constraints.set_senses(ct_index, new_ct_cpxtype)
            updated = True
        else:
            if event in (upd.LinearConstraintCoef, upd.LinearConstraintGlobal):
                if args:
                    self._switch_linear_exprs2(ct_index,
                                               old_left=ct._left_expr, old_right=ct._right_expr,
                                               new_left=args[0], new_right=args[1])
                else:
                    self._switch_linear_exprs2(ct_index,
                                               old_left=None, old_right=None,
                                               new_left=ct._left_expr, new_right=ct._right_expr)
                updated = True
            if event in (upd.LinearConstraintRhs, upd.LinearConstraintGlobal):
                if args:
                    new_ct_rhs = self.make_cpx_ct_rhs_from_exprs(left_expr=args[0], right_expr=args[1])
                else:
                    new_ct_rhs = ct.cplex_num_rhs()
                self._fast_set_rhs(ct_index, new_rhs=new_ct_rhs)
                updated = True

        if not updated:  # pragma: no cover
            self._unexpected_event(event, msg='update_linear-constraint')

    def update_range_constraint(self, rngct, event, *args):
        self._resync_if_needed()
        rng_index = rngct.index
        assert rng_index >= 0
        cpx_linear = self._cplex.linear_constraints
        if event == upd.RangeConstraintBounds:
            new_lb, new_ub = args
            offset = rngct.expr.get_constant()
            cpx_rhs_value = new_ub - offset
            msg = lambda: "Cannot update range values of {0!s} to [{1}..{2}]: domain is infeasible".format(rngct, new_lb, new_ub)
            cpx_range_value = RangeConstraint.static_cplex_range_value(self._model, new_lb, new_ub, msg)
            cpx_linear.set_rhs(rng_index, cpx_rhs_value)
            cpx_linear.set_range_values(rng_index, cpx_range_value)

        elif event == upd.RangeConstraintExpr:
            old_expr = rngct.expr
            new_expr = args[0]
            if old_expr.get_constant() != new_expr.get_constant():
                # need to change rhs but *not* range (ub-lb)
                cpx_rhs_value = rngct.ub - new_expr.get_constant()
                # TODO: check positive??
                cpx_linear.set_rhs(rng_index, cpx_rhs_value)
            # change expr linear components anyway
            self._switch_linear_expr(rng_index, old_expr, new_expr)

        else:  # pragma: no cover
            self._unexpected_event(event, msg='update_range_constraints')

    def update_quadratic_constraint(self, qct, event, *args):
        self._model.fatal('CPLEX cannot modify quadratic constraint: {0!s}', qct)

    def update_extra_constraint(self, lct, qualifier, *args):
        self._model.fatal('CPLEX cannot modify {1}: {0!s}', lct, qualifier)

    def update_logical_constraint(self, lgct, event, *args):
        self._resync_if_needed()
        if isinstance(lgct, IndicatorConstraint):
            self._model.fatal('CPLEX cannot modify a linear constraint used in an indicator: ({0!s})', lgct)
        elif isinstance(lgct, EquivalenceConstraint):
            if not lgct.is_generated():
                self._model.fatal('CPLEX cannot modify a linear constraint used in an equivalence: ({0!s})', lgct)
            else:
                self._model.fatal('Using the truth value of the constraint: ({0!s}) makes the constraint immutable',
                                  lgct.linear_constraint)
        else:  # pragma: no cover
            self._model.fatal('Unexpected type for logical constraint: {0!r}', lgct)

    def update_constraint(self, ct, event, *args):
        self._resync_if_needed()
        if event and ct.index >= 0:
            scope = ct.cplex_scope
            if scope == CplexScope.LINEAR_CT_SCOPE:
                self.update_linear_constraint(ct, event, *args)
            elif scope == CplexScope.IND_CT_SCOPE:
                self.update_logical_constraint(ct, event, *args)
            elif scope == CplexScope.QUAD_CT_SCOPE:
                self.update_quadratic_constraint(ct, event, *args)
            else:
                self._model.fatal('Unexpected scope in update_constraint: {0!r}', scope)

    def set_objective_sense(self, sense):
        self._resync_if_needed()
        # --- set sense
        self._cplex.objective.set_sense(sense.cplex_coef)

    def _clear_objective_from_cplex(self, cpxobj):
        self._clear_linear_objective_from_cplex(cpxobj)
        self._clear_quad_objective_from_cplex(cpxobj)

    def _clear_multiobj_from_cplex(self, cpx_multiobj):
        numobj = cpx_multiobj.get_num()
        for objidx in range(numobj):
            cpx_linear = cpx_multiobj.get_linear(objidx)
            zap_linear = [(idx, 0) for idx, k in enumerate(cpx_linear) if k]
            if zap_linear:
                cpx_multiobj.set_linear(objidx, zap_linear)

    def _clear_linear_objective_from_cplex(self, cpxobj):
        # clear linear part
        cpx_linear = cpxobj.get_linear()
        zap_linear = [(idx, 0) for idx, k in enumerate(cpx_linear) if k]
        if zap_linear:
            cpxobj.set_linear(zap_linear)

    def _clear_quad_objective_from_cplex(self, cpxobj):
        # quadratic
        if cpxobj.get_num_quadratic_variables():
            # need to check before calling get_quadratic() on non-qp -> crash
            cpx_quads = cpxobj.get_quadratic()
            if cpx_quads:
                # reset all vars to zero
                nb_vars = self._model.number_of_variables
                zap_quads = [0.0] * nb_vars  # needs a 0.0 as cplex explicitly tests for double...
                cpxobj.set_quadratic(zap_quads)

    def update_objective(self, expr, event, *args):
        self._resync_if_needed()
        cpxobj = self._cplex.objective
        if event is upd.ExprConstant:
            # update the constant
            self._cplex.objective.set_offset(expr.constant)
        elif event in frozenset([upd.LinExprCoef, upd.LinExprGlobal]):
            self._clear_linear_objective_from_cplex(cpxobj)
            self._set_linear_objective_coefs(cpxobj, linexpr=expr.get_linear_part())
            if event is upd.LinExprGlobal:
                cpxobj.set_offset(expr.constant)
        elif event is upd.QuadExprQuadCoef:
            # clear quad, set quad
            self._clear_quad_objective_from_cplex(cpxobj)
            self._set_quadratic_objective_coefs(cpxobj, expr)
        elif event is upd.QuadExprGlobal:
            # clear all
            self._clear_linear_objective_from_cplex(cpxobj)
            # set all
            self._set_linear_objective_coefs(cpxobj, linexpr=expr.get_linear_part())
            self._clear_quad_objective_from_cplex(cpxobj)
            self._set_quadratic_objective_coefs(cpxobj, expr)
            cpxobj.set_offset(expr.constant)

        else:  # pragma: no cover
            self._unexpected_event(event, msg='update_objective')

    def set_objective_expr(self, new_objexpr, old_objexpr):
        self._resync_if_needed()
        cpx_objective = self._cplex.objective
        # old objective
        if old_objexpr is new_objexpr:
            # cannot use the old expression for clearing, it has been modified
            self._clear_objective_from_cplex(cpxobj=cpx_objective)
        elif old_objexpr is not None:
            self._clear_objective(old_objexpr)
        else:
            # no clearing
            pass
        # # if a multi-objective has been defined, clear it
        # cpx_multiobj = self._cplex.multiobj
        # if cpx_multiobj is not None:
        #     self._clear_multiobj_from_cplex(cpx_multiobj=cpx_multiobj)

        # --- set offset
        cpx_objective.set_offset(float(new_objexpr.get_constant()))
        # --- set coefficients
        if new_objexpr.is_quad_expr():
            self._fast_set_quadratic_objective(quad_expr=new_objexpr)
            self._fast_set_linear_objective(new_objexpr.linear_part)
        else:
            self._fast_set_linear_objective2(linexpr=new_objexpr)

    def set_multi_objective_tolerances(self, abstols, reltols):
        self._check_multi_objective_support()
        cpx = self._cplex
        if abstols is not None:
            for obj_idx, abstol in enumerate(abstols):
                assert abstol >= 0
                cpx.multiobj.set_abstol(obj_idx, abstol=abstol)
        if reltols is not None:
            for obj_idx, reltol in enumerate(reltols):
                assert reltol >= 0
                cpx.multiobj.set_reltol(obj_idx, reltol=reltol)




    def set_multi_objective_exprs(self, new_multiobjexprs, old_multiobjexprs, multiobj_params=None,
                                  priorities=None, weights=None, abstols=None, reltols=None, objnames=None):
        self._check_multi_objective_support()

        cpx_multiobj = self._cplex.multiobj
        if  old_multiobjexprs:
            self._clear_multiobj_from_cplex(cpx_multiobj=cpx_multiobj)

        # --- set number of objectives
        cpx_multiobj.set_num(len(new_multiobjexprs))
        for objidx, new_objexpr in enumerate(new_multiobjexprs):
            # --- set offset
            cpx_multiobj.set_offset(objidx, float(new_objexpr.get_constant()))
            # --- set coefficients
            weight = self.DEFAULT_CPX_NO_WEIGHT_CHANGE
            priority = self.DEFAULT_CPX_NO_PRIORITY_CHANGE
            abstol = self.DEFAULT_CPX_NO_ABSTOL_CHANGE
            reltol = self.DEFAULT_CPX_NO_RELTOL_CHANGE
            objname = None
            if priorities is not None and len(priorities) >= objidx + 1:
                priority = priorities[objidx]
            if weights is not None and len(weights) >= objidx + 1:
                weight = weights[objidx]
            if abstols is not None and len(abstols) >= objidx + 1:
                abstol = abstols[objidx]
            if reltols is not None and len(reltols) >= objidx + 1:
                reltol = reltols[objidx]
            if objnames is not None and len(objnames) >= objidx + 1:
                objname = objnames[objidx]
            self._fast_set_linear_multiobj(objidx, linexpr=new_objexpr,
                                           weight=weight, priority=priority,
                                           abstol=abstol, reltol=reltol,
                                           objname=objname)

    def _set_linear_objective_coefs(self, cpx_objective, linexpr):
        # NOTE: convert to float as numpy doubles will crash cplex....
        #     index_coef_seq = [(dv._index, float(k)) for dv, k in linexpr.iter_terms()]
        #     if index_coef_seq:
        #         # if list is empty, cplex will crash.
        #         cpx_objective.set_linear(index_coef_seq)
        self._fast_set_linear_objective(linexpr)

    def _set_quadratic_objective_coefs(self, cpx_objective, quad_expr):
        quad_obj_triplets = [(qv1._index, qv2._index, 2 * qk if qv1 is qv2 else qk) for qv1, qv2, qk in
                             quad_expr.iter_quad_triplets()]
        if quad_obj_triplets:
            # if list is empty, cplex will crash.
            cpx_objective.set_quadratic_coefficients(quad_obj_triplets)

    def _clear_objective(self, expr):
        # INTERNAL
        self._resync_if_needed()
        if expr.is_constant():
            pass  # resetting offset will do.
        elif expr.is_quad_expr():
            # 1. reset quad part
            cpx_objective = self._cplex.objective
            # -- set quad coeff to 0 for all quad variable pairs
            quad_reset_triplets = [(qvp.first._index, qvp.second._index, 0) for qvp, qk in expr.iter_quads()]
            if quad_reset_triplets:
                cpx_objective.set_quadratic_coefficients(quad_reset_triplets)
            # 2. reset linear part
            self._clear_linear_objective(expr.linear_part)
        else:
            self._clear_linear_objective(expr)

    def _clear_multiobj(self, exprs):
        # INTERNAL
        self._resync_if_needed()
        for objidx, expr in enumerate(exprs):
            if expr.is_constant():
                pass  # resetting offset will do.
            else:
                self._clear_linear_multiobj(objidx, expr)

    def _clear_linear_objective(self, linexpr):
        # compute the sequence of  var indices, then an array of zeroes
        size = linexpr.number_of_terms()
        if size:
            indices = [-1] * size
            i = 0
            for dv, _ in linexpr.iter_terms():
                indices[i] = dv._index
                i += 1
            zeros = [0] * size
            cpx = self._cplex
            self.cpx_adapter.chgobj(cpx._env._e, cpx._lp, indices, zeros)

    def _clear_linear_multiobj(self, objidx, linexpr):
        # compute the sequence of var indices, then an array of zeroes
        size = linexpr.number_of_terms()
        if size:
            indices = [-1] * size
            i = 0
            for dv, _ in linexpr.iter_terms():
                indices[i] = dv._index
                i += 1
            zeros = [0] * size
            cpx = self._cplex
            self.cpx_adapter.multiobjsetobj(cpx._env._e, cpx._lp, objidx, objind=indices, objval=zeros)

    @staticmethod
    def status2string(cplex_module, cpx_status):  # pragma: no cover
        ''' Converts a CPLEX integer status value to a string'''
        return cplex_module._internal._subinterfaces.SolutionInterface.status.__getitem__(cpx_status)

    # Moved to member method, but does not seem to be used ?
    # TODO: Remove as cleanup
    # @classmethod
    # def add_ok_status(cls, ok_status):
    #     # INTERNAL
    #     cls._CPLEX_SOLVE_OK_STATUSES.add(ok_status)

    def _is_solve_status_ok(self, status):
        # Converts a raw CPLEX status to a boolean
        return status in self._CPLEX_SOLVE_OK_STATUSES

    def _is_relaxed_status_ok(self, status):
        # list all status values for which there is a relaxed solution.
        # also consider solve statuses in case  the model is indeed feasible
        # TODO: as cleanup, does not seem to be used ?
        return status in self._CPLEX_RELAX_OK_STATUSES

    @property
    def name(self):
        return 'cplex_local'

    @staticmethod
    def sol_to_cpx_mipstart(model, mipstart, completion=False):
        if completion:
            tl = [(dv.index, mipstart[dv]) for dv in model.generate_user_variables()]
        else:
            tl = [(dv.index, dvv) for dv, dvv in mipstart.iter_var_values()]
        ul = zip(*tl)
        # py3 zip() returns a generator, not a list, and CPLEX needs a list!
        return list(ul)

    def _sync_var_bounds(self, verbose=False):
        self_var_lbs = self._var_lb_changed
        if self_var_lbs:
            lb_vars, lb_values = zip(*six.iteritems(self_var_lbs))
            self._apply_var_fn(dvars=lb_vars, args=lb_values,
                               setter_fn=self.cpx_adapter.cplex_module._internal._subinterfaces.VariablesInterface.set_lower_bounds)
            if verbose:  # pragma: no cover
                print("* synced {} var lower bounds".format(len(self._var_lb_changed)))

        self_var_ubs = self._var_ub_changed
        if self_var_ubs:
            ub_vars, ub_values = zip(*six.iteritems(self_var_ubs))
            self._apply_var_fn(dvars=ub_vars, args=ub_values,
                               setter_fn=self.cpx_adapter.cplex_module._internal._subinterfaces.VariablesInterface.set_upper_bounds)
            if verbose:  # pragma: no cover
                print("* synced {} var upper bounds".format(len(self._var_ub_changed)))

    def _sync_annotations(self, model):
        cpx = self._cplex
        annotated_by_scope = []
        try:
            # separate go to the model and get all annotations
            annotated_by_scope = model.get_annotations_by_scope()

            cpx_anno = cpx.long_annotations
            # du passe faison table rase....
            cpx_anno.delete()
            if annotated_by_scope:
                benders_idx = self._benders_anno_idx
                if benders_idx < 0:
                    # create benders annotation
                    benders_idx = cpx_anno.add(self.cpx_adapter.cpx_cst.CPX_BENDERS_ANNOTATION, 0)  # quid of defval?
                    assert benders_idx >= 0
                    self._benders_anno_idx = benders_idx
                # ---

                # at this stage we have a valid annotation index
                # and a dict of scope -> list of (idx, anno) tuples
                for cpx_scope, annotated in six.iteritems(annotated_by_scope):
                    cpx_anno_objtype = self.annotation_map.get(cpx_scope)
                    if cpx_anno_objtype:
                        annotated_indices = []
                        long_annotations = []
                        for obj, group in annotated:
                            annotated_indices.append(obj.index)
                            long_annotations.append(group)
                        self._fast_set_longanno(benders_idx, cpx_anno_objtype, annotated_indices, long_annotations)
                    else:
                        self._model.error('Cannot map to CPLEX annotation type: {0!r}. ignored', cpx_scope)
            elif cpx_anno.get_num():
                cpx_anno.delete()

        except AttributeError:  # pragma: no cover
            if annotated_by_scope:
                self._model.fatal('Annotations require CPLEX 12.7.1 or higher')

    def create_sos(self, sos_set):
        cpx_sos_type = sos_set.sos_type._cpx_sos_type()
        indices = [dv.index for dv in sos_set.iter_variables()]
        weights = sos_set.weights
        # do NOT pass None to cplex/swig here --> crash
        cpx_sos_name = sos_set.safe_name
        # call cplex...
        sos_index = self._cplex.SOS.add(type=cpx_sos_type,
                                        SOS=self.cpx_adapter.cplex_module.SparsePair(ind=indices, val=weights),
                                        name=cpx_sos_name)
        return sos_index

    def clear_all_sos(self):
        self._cplex.SOS.delete()

    def add_lazy_constraints(self, lazy_cts):
        if lazy_cts:
            # lazy_cts is a sequence of linear constraints
            cpx_rhss = [ct.cplex_num_rhs() for ct in lazy_cts]
            cpx_senses = "".join(ct.cplex_code() for ct in lazy_cts)
            if self._model.ignore_names:
                cpx_names = []
            else:
                cpx_names = [ct.safe_name for ct in lazy_cts]
            cpx_convert_fn = self.linear_ct_to_cplex
            cpx_linexprs = [cpx_convert_fn(ct) for ct in lazy_cts]
            # TODO: switch to procedural API at some point...
            self._cplex.linear_constraints.advanced.add_lazy_constraints(cpx_linexprs, cpx_senses, cpx_rhss, cpx_names)

    def clear_lazy_constraints(self):
        self._cplex.linear_constraints.advanced.free_lazy_constraints()

    def add_user_cuts(self, cut_cts):
        if cut_cts:
            # cut_cts is a sequence of linear constraints
            cpx_rhss = [ct.cplex_num_rhs() for ct in cut_cts]
            cpx_senses = "".join(ct.cplex_code() for ct in cut_cts)
            if self._model.ignore_names:
                cpx_names = []
            else:
                cpx_names = [ct.safe_name for ct in cut_cts]
            cpx_convert_fn = self.linear_ct_to_cplex
            cpx_linexprs = [cpx_convert_fn(ct) for ct in cut_cts]
            # TODO: switch to procedural API at some point...
            self._cplex.linear_constraints.advanced.add_user_cuts(cpx_linexprs, cpx_senses, cpx_rhss, cpx_names)

    def clear_user_cuts(self):
        self._cplex.linear_constraints.advanced.free_user_cuts()

    def _format_cplex_message(self, cpx_msg):
        if 'CPLEX' not in cpx_msg:
            cpx_msg = 'CPLEX: %s' % cpx_msg
        return cpx_msg.rstrip(' .\n')

    def _parse_cplex_exception_as_status(self, cpx_ex):
        cpx_ex_s = str(cpx_ex)
        msg = cpx_ex_s
        for extype in ['CplexSolverError', 'CplexError']:
            prefix = 'cplex.exceptions.errors.{0}: '.format(extype)
            if cpx_ex_s.startswith(prefix):
                msg = cpx_ex_s[len(prefix):]

        cpx_code_match = self.cplex_error_re.match(msg)
        code = -1
        if cpx_code_match:
            try:
                code = int(cpx_code_match.group(1))
            except ValueError:
                pass

        return code, msg

    def clean_before_solve(self):
        # INTERNAL
        # delete all infos that were left by the previous solve
        # from DJ- RTC34054
        #docplex_debug_msg("cleaning solver ----------------")
        cpx = self._cplex
        if cpx._is_MIP():
            cpx.MIP_starts.delete()
        cpx.presolve.free()
        # clear the pool
        cpx.solution.pool.delete()
        # dummy change
        if cpx.variables.get_num() > 0:
            cpx.variables.set_lower_bounds(0, cpx.variables.get_lower_bounds(0))
        elif cpx.linear_constraints.get_num() > 0:
            cpx.linear_constraints.set_senses(0, cpx.linear_constraints.get_senses(0))
        else:
            pass

    def _get_priorities_list_in_decreasing_order(self):
        # Compute list of priorities in decreasing order
        prio_list = [self._cplex.multiobj.get_priority(oidx) for oidx in range(self._cplex.multiobj.get_num())]
        prio_set = set(prio_list)
        inversed_ordered_prio = sorted(list(prio_set))
        inversed_ordered_prio.reverse()
        return inversed_ordered_prio

    def _check_multi_objective_support(self):
        ok, msg = self.supports_multi_objective()
        if not ok:
            self._model.fatal(msg)

    def set_lp_start(self, dvar_stats, lct_stats):
        cpx = self._cplex
        dvar_int_stats = [stat.value for stat in dvar_stats]
        lct_int_stats = [stat.value for stat in lct_stats]
        cpx.start.set_start(dvar_int_stats, lct_int_stats, col_primal=[], col_dual=[], row_primal=[], row_dual=[])


    def build_multiobj_paramsets(self, mdl, lex_timelimits, lex_mipgaps):
        cpx = self._cplex
        paramsets = None
        if lex_timelimits is not None or lex_mipgaps is not None:
            self._check_multi_objective_support()

            # Get list of priorities in decreasing order
            decreasing_ordered_prio = self._get_priorities_list_in_decreasing_order()

            if lex_timelimits is not None and len(lex_timelimits) != len(decreasing_ordered_prio):
                mdl.fatal("lex_timelimits list length does not match number of priorities for multiobjective solve")
            if lex_mipgaps is not None and len(lex_mipgaps) != len(decreasing_ordered_prio):
                mdl.fatal("lex_mipgaps list length does not match number of priorities for multiobjective solve")
            paramsets = []
            for _ in decreasing_ordered_prio:
                paramset = cpx.create_parameter_set()
                paramsets.append(paramset)
            if lex_timelimits is not None:
                for paramIdx, timelimit in enumerate(lex_timelimits):
                    paramsets[paramIdx].add(self.cpx_adapter.cpx_cst.CPX_PARAM_TILIM, timelimit)
            if lex_mipgaps is not None:
                for paramIdx, mipgap in enumerate(lex_mipgaps):
                    paramsets[paramIdx].add(self.cpx_adapter.cpx_cst.CPX_PARAM_EPGAP, mipgap)
            # If there is a single priority level, timelimit and mipgaps are handled at the cplex_parameters level
            paramsets = paramsets if len(paramsets) > 1 else None
        return paramsets

    def sync_cplex(self, mdl):
        # INTERNAL: wrap all syncs
        self._sync_var_bounds()
        self._sync_annotations(mdl)

    def solve(self, mdl, parameters=None, **kwargs):
        self._resync_if_needed()

        cpx = self._cplex
        # keep this line until RTC28217 is solved and closed !!! ----------------
        # see RTC 28217 item #18 for details
        cpx.get_problem_name()  # workaround from Ryan

        # ------ !!! see RTC34123 ---
        self.cpx_adapter.setintparam(cpx._env._e, 1047, -1)
        # ---------------------------
        #self._solve_count += 1
        solve_time_start = cpx.get_time()
        cpx_status = -1
        cpx_miprelgap = None
        cpx_bestbound = None
        linear_nonzeros = -1
        nb_columns = 0
        cpx_probtype = None

        lex_mipstart = kwargs.pop('_lex_mipstart', None)
        lex_mipgaps = kwargs.pop('lex_mipgaps', None)
        lex_timelimits = kwargs.pop('lex_timelimits', None)
        clean_before_solve = kwargs.pop('clean_before_solve', False)

        # print("--> starting CPLEX solve #", self.__solveCount)
        cpx_status_string = None
        nb_iterations, nb_nodes_processed = 0, 0
        try:
            # keep this in the protected  block...
            self.sync_cplex(mdl)
            if clean_before_solve:
                self.clean_before_solve()

            # --- mipstart block ---
            cpx_mip_starts = cpx.MIP_starts
            if not lex_mipstart: # ignore mip starts if within lexicographic
                # do -not- delete mip starts here,
                # as each solve creates one.
                # set clean_before_solve=True to enforce clean start
                # TODO: avoid adding model mip starts twice.
                for (mps, effort_level) in mdl.iter_mip_starts():
                    if not mps.number_of_var_values:
                        self._model.warning("Empty MIP start solution ignored")
                    else:
                        cpx_sol = self.sol_to_cpx_mipstart(mdl, mps)
                        cpx_mip_starts.add(cpx_sol, effort_level.value)
            elif mdl.number_of_mip_starts:
                self._model.warning("Lexicographic solve ignored {0} mipstarts".format(mdl.number_of_mip_starts))

            # --- end of mipstart block ---

            linear_nonzeros = cpx.linear_constraints.get_num_nonzeros()
            nb_columns = cpx.variables.get_num()
            cpx_probtype = cpx.problem_type[cpx.get_problem_type()]
            nb_iterations, nb_nodes_processed = 0, 0

            #Handle lex_timelimits list
            paramsets = self.build_multiobj_paramsets(mdl, lex_timelimits, lex_mipgaps)
            if paramsets:
                cpx.solve(paramsets=paramsets)
            else:
                cpx.solve()

            cpx_status = cpx.solution.get_status()
            cpx_status_string = self._cplex.solution.get_status_string(cpx_status)
            is_mip = cpx._is_MIP()

            solve_ok = self._is_solve_status_ok(cpx_status)
            if solve_ok:
                nb_iterations, nb_nodes_processed = get_progress_details(cpx)
                if is_mip:
                    cpx_miprelgap = cpx.solution.MIP.get_mip_relative_gap()
                    cpx_bestbound = cpx.solution.MIP.get_best_objective()


        except self.cpx_adapter.CplexSolverError as cpx_se:  # pragma: no cover
            cpx_code = cpx_se.args[2]
            solve_ok = False
            if 5002 == cpx_code:
                # we are in the notorious "non convex" case.
                # provide a meaningful status string for the solve details
                cpx_status = 5002  # famous error code...

                if self._model.has_quadratic_constraint():
                    cpx_status_string = "Non-convex QCP"
                    self._model.error('Model is non-convex')
                else:
                    cpx_status_string = "QP with non-convex objective"
                    self._model.error('Model has non-convex objective: {0!s}', str_maxed(self._model.objective_expr, 60))
            elif 1016 == cpx_code:
                # this is the: CPXERR_RESTRICTED_VERSION - " Promotional version. Problem size limits exceeded." case
                cpx_status = 1016
                cpx_status_string = "Promotional version. Problem size limits exceeded., CPLEX code=1016."
                self._model.fatal_ce_limits()

            elif self.fix_multiobj_error_1300 and 1300 == cpx_code:
                # multiobjective error but there IS a solution (?)
                cpx_status = 1300
                cpx_status_string = "multiobjective non-optimal with error"
                # force to build a solution
                solve_ok = True
            else:
                cpx_status, cpx_status_string = self._parse_cplex_exception_as_status(cpx_se)

        except self.cpx_adapter.CplexError as cpx_e:  # pragma: no cover
            s_cpx_e = str(cpx_e)
            self.error_handler.error("CPLEX error: {0!s}", self._format_cplex_message(s_cpx_e))
            cpx_status, cpx_status_string = self._parse_cplex_exception_as_status(cpx_e)
            solve_ok = False

        except Exception as pe:  # pragma: no cover
            solve_ok = False
            self.error_handler.error('Internal error in CPLEX solve: {0}: {1!s}'.format(type(pe).__name__, pe))
            cpx_status_string = 'Internal error: {0!s}'.format(pe)
            cpx_status = -2

        finally:

            solve_time = cpx.get_time() - solve_time_start

            details = SolveDetails(solve_time,
                                   cpx_status, cpx_status_string,
                                   cpx_probtype,
                                   nb_columns, linear_nonzeros,
                                   cpx_miprelgap, cpx_bestbound,
                                   nb_iterations,
                                   nb_nodes_processed)
            if self._model.quality_metrics:
                details._quality_metrics = self._compute_quality_metrics()
            self._last_solve_details = details

        # clear bound change requests
        self._var_lb_changed = {}
        self._var_ub_changed = {}

        self._last_solve_status = solve_ok
        new_solution = None
        if solve_ok:
            new_solution = self._make_solution(mdl, self.get_solve_status())
        else:
            mdl.notify_solve_failed()
        if cpx_status_string:
            mdl.error_handler.trace("CPLEX solve returns with status: {0}", (cpx_status_string,))
        return new_solution

    def _make_solution(self, mdl, job_solve_status):
        cpx_adapter = self.cpx_adapter
        cpx = self._cplex
        full_obj = cpx.solution.get_objective_value()
        if self._has_multi_objective(cpx):
            full_obj = [cpx.solution.multiobj.get_objective_value(objidx) for objidx in range(cpx.multiobj.get_num())]

        # Build list of objectives value by priority level (ie: each priority level corresponds to blended objectives
        # with same priority)
        full_obj_by_prio = [full_obj]
        if self._has_multi_objective(cpx):
            decreasing_ordered_prio = self._get_priorities_list_in_decreasing_order()
            full_obj_by_prio = [cpx.solution.multiobj.get_objval_by_priority(prio) for prio in decreasing_ordered_prio]

        nb_vars = mdl.number_of_variables
        if nb_vars > 0:
            if self.procedural:
                all_var_values = cpx_adapter.fast_get_solution(cpx, nb_vars)
            else:
                all_var_values = cpx.solution.get_values()
                #all_var_values = fast_get_solution(cpx, mdl.number_of_variables)
            vmap = {}
            for dv in mdl.iter_variables():
                dvv = all_var_values[dv._index]
                if dvv:
                    vmap[dv] = dvv
            var_value_map = vmap
            #var_value_map = dict(izip(mdl.iter_variables(), all_var_values))
        else:
            var_value_map = {}

        solve_details = self._last_solve_details
        assert solve_details is not None
        solution = SolveSolution.make_engine_solution(model=mdl,
                                                      var_value_map=var_value_map,
                                                      obj=full_obj,
                                                      blended_obj_by_priority=full_obj_by_prio,
                                                      solved_by=self.name,
                                                      solve_details=solve_details,
                                                      job_solve_status=job_solve_status)
        return solution

    @classmethod
    def handle_cplex_solver_error(cls, logger, mdl, cpxse, initial_status, initial_status_string):
        status, status_string = initial_status, initial_status_string
        cpx_code = cpxse.args[2]
        if 5002 == cpx_code:
            # we are in the notorious "non convex" case.
            # provide a meaningful status string for the solve details
            status = 5002  # famous error code...

            if mdl.has_quadratic_constraint():
                status_string = "Non-convex QCP"
                logger.error('Model is non-convex')
            else:
                status_string = "QP with non-convex objective"
                logger.error('Model has non-convex objective: {0!s}', str_maxed(mdl.objective_expr, 60))
        elif 1016 == cpx_code:
            # this is the: CPXERR_RESTRICTED_VERSION - " Promotional version. Problem size limits exceeded." case
            status = 1016
            status_string = "Promotional version. Problem size limits exceeded., CPLEX code=1016."
            logger.fatal(status_string)
        else:
            logger.error("CPLEX Solver Error: {0!s}", cpxse)

        return status, status_string

    def _run_cpx_solve_fn(self, cpx_fn, ok_statuses, *args):
        cpx = self._cplex
        cpx_time_start = cpx.get_time()
        cpx_status = -1
        cpx_status_string = "*unknown*"
        cpx_miprelgap = None
        cpx_bestbound = None
        linear_nonzeros = -1
        nb_columns = 0
        cpx_probtype = None
        solve_ok = False
        logger = self.error_handler
        # noinspection PyPep8
        try:
            linear_nonzeros = cpx.linear_constraints.get_num_nonzeros()
            nb_columns = cpx.variables.get_num()
            cpx_fn(*args)
            cpx_status = cpx.solution.get_status()
            cpx_probtype = cpx.problem_type[cpx.get_problem_type()]
            cpx_status_string = self._cplex.solution.get_status_string(cpx_status)
            solve_ok = cpx_status in ok_statuses
            if solve_ok:
                if cpx._is_MIP():
                    cpx_miprelgap = cpx.solution.MIP.get_mip_relative_gap()
                    cpx_bestbound = cpx.solution.MIP.get_best_objective()

        except self.cpx_adapter.CplexSolverError as cpx_s:  # pragma: no cover
            new_status, new_s_status = self.handle_cplex_solver_error(logger, self._model, cpx_s, cpx_status, cpx_status_string)
            cpx_status, cpx_status_string = new_status, new_s_status
            # cpx_code = cpx_s.args[2]
            # if 5002 == cpx_code:
            #     # we are in the notorious "non convex" case.
            #     # provide a meaningful status string for the solve details
            #     cpx_status = 5002  # famous error code...
            #
            #     if self._model.has_quadratic_constraint():
            #         cpx_status_string = "Non-convex QCP"
            #         logger.error('Model is non-convex')
            #     else:
            #         cpx_status_string = "QP with non-convex objective"
            #         logger.error('Model has non-convex objective: {0!s}', str_maxed(self._model.objective_expr, 60))
            # elif 1016 == cpx_code:
            #     # this is the: CPXERR_RESTRICTED_VERSION - " Promotional version. Problem size limits exceeded." case
            #     cpx_status = 1016
            #     cpx_status_string = "Promotional version. Problem size limits exceeded., CPLEX code=1016."
            #     logger.fatal(cpx_status_string)
            # else:
            #     logger.error("CPLEX Solver Error: {0!s}", cpx_s)

        except self.cpx_adapter.exceptions.CplexError as cpx_e:  # pragma: no cover
            logger.error("CPLEX Error: {0!s}", cpx_e)


        finally:
            cpx_time = cpx.get_time() - cpx_time_start

        details = SolveDetails(cpx_time,
                               cpx_status, cpx_status_string,
                               cpx_probtype,
                               nb_columns, linear_nonzeros,
                               cpx_miprelgap,
                               cpx_bestbound)
        self._last_solve_details = details
        if solve_ok:
            sol = self._make_solution(self._model, self.get_solve_status())
        else:
            sol = None
        return sol

    def get_solve_details(self):
        # must be solved but not necessarily ok
        return self._last_solve_details

    def _make_groups(self, relaxable_groups):
        cpx_feasopt = self._cplex.feasopt
        all_groups = []
        for (pref, group_cts) in relaxable_groups:
            if pref > 0 and group_cts:
                linears = []
                quads = []
                inds = []
                for ct in group_cts:
                    ctindex = ct.index
                    cpx_scope = ct.cplex_scope
                    if cpx_scope is CplexScope.LINEAR_CT_SCOPE:
                        linears.append(ctindex)
                    elif cpx_scope is CplexScope.IND_CT_SCOPE:
                        inds.append(ctindex)
                    elif cpx_scope is CplexScope.QUAD_CT_SCOPE:
                        quads.append(ctindex)
                    else:
                        self.error_handler.error('cannot relax this: {0!s}'.format(ct))

                if linears:
                    all_groups.append(cpx_feasopt.linear_constraints(pref, linears))
                if quads:
                    all_groups.append(cpx_feasopt.quadratic_constraints(pref, quads))
                if inds:
                    all_groups.append(cpx_feasopt.indicator_constraints(pref, inds))
        return all_groups

    def _decode_infeasibilities(self, cpx, model, cpx_relax_groups, model_scope_resolver=None):
        cpx_adapter = self.cpx_adapter
        if model_scope_resolver is None:
            # set default value for resolver
            model_scope_resolver = {cpx_adapter.ct_linear: lambda m_: m_._linct_scope,
                                    cpx_adapter.ct_quadratic: lambda m_: m_._quadct_scope,
                                    cpx_adapter.ct_indicator: lambda m_: m_._logical_scope
                                    }

        resolver_map = {cpx_adapter.ct_linear: cpx.solution.infeasibility.linear_constraints,
                        cpx_adapter.ct_quadratic: cpx.solution.infeasibility.quadratic_constraints,
                        cpx_adapter.ct_indicator: cpx.solution.infeasibility.indicator_constraints
                        }
        cpx_sol_values = cpx.solution.get_values()
        cts_by_type = defaultdict(list)
        # split and group indices by sense
        for g in cpx_relax_groups:
            # gp is a list of tuples (pref, ctsense, index)
            for t in g._gp:
                ct_sense, ct_index = t[1][0]
                cts_by_type[ct_sense].append(ct_index)

        infeas_map = {}
        for ct_sense, indices in six.iteritems(cts_by_type):
            if indices:
                resolver_fn = resolver_map[ct_sense]
                ctype_infeas = resolver_fn(cpx_sol_values, indices)
                mscope = model_scope_resolver[ct_sense](model)
                assert mscope is not None
                # noinspection PyArgumentList
                for ct_index, ct_infeas in izip(indices, ctype_infeas):
                    ct = mscope.get_object_by_index(ct_index)
                    if ct is not None:
                        infeas_map[ct] = ct_infeas
        return infeas_map

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        # INTERNAL
        self._resync_if_needed()
        self.sync_cplex(mdl)
        if mdl.clean_before_solve:
            self.clean_before_solve()

        self_cplex = self._cplex
        cpx_relax_groups = self._make_groups(relaxable_groups)

        feasopt_parameters = parameters or mdl.parameters
        feasopt_override_params = {feasopt_parameters.feasopt.mode: relax_mode.value}

        with _CplexOverwriteParametersCtx(self_cplex, feasopt_override_params) as cpx:
            # at this stage, we have a list of groups
            # each group is itself a list
            # the first item is a number, the preference
            # the second item is a list of constraint indices.
            relaxed_sol = self._run_cpx_solve_fn(cpx.feasopt,
                                                 self._CPLEX_RELAX_OK_STATUSES,
                                                 *cpx_relax_groups)


        if relaxed_sol is not None:
            infeas_map = self._decode_infeasibilities(self_cplex, mdl, cpx_relax_groups)
            relaxed_sol.store_infeasibilities(infeas_map)
        return relaxed_sol

    def _sync_parameter_defaults_from_cplex(self, parameters):
        # used when a more recent CPLEX DLL is present
        resets = []
        for param in parameters:
            cpx_value = self.get_parameter(param)
            if cpx_value != param.default_value:
                resets.append((param, param.default_value, cpx_value))
                param.reset_default_value(cpx_value)
        return resets

    def _make_cplex_default_groups(self, mdl):
        cpx_cst = self.cpx_adapter.cpx_cst

        def make_atom_group(pref, obj, con):
            return pref, ((con, obj.index),)

        grs = []
        # add all linear constraints with pref=2.0
        lc_pref = 2.0
        for lc in mdl.iter_linear_constraints():
            grs.append( make_atom_group(lc_pref, lc, cpx_cst.CPX_CON_LINEAR))

        # add quadratic w 2
        quad_pref = 2.0
        for qc in mdl.iter_quadratic_constraints():
            grs.append( (make_atom_group(quad_pref, qc, cpx_cst.CPX_CON_QUADRATIC)))

        ind_pref = 1.0
        for lc in mdl.iter_logical_constraints():
            grs.append( (make_atom_group(ind_pref, lc, cpx_cst.CPX_CON_INDICATOR)))

        var_bounds_pref = 4.0
        inf = mdl.infinity
        for dv in mdl.iter_variables():
            if not dv.is_binary():
                if dv.lb != 0:
                    grs.append( make_atom_group(var_bounds_pref, dv, cpx_cst.CPX_CON_LOWER_BOUND))
                if dv.ub < inf:
                    grs.append( make_atom_group(var_bounds_pref, dv, cpx_cst.CPX_CON_UPPER_BOUND))
            else:
                # do not include free binary variables.
                if dv.lb >= 0.5:
                    grs.append( make_atom_group(var_bounds_pref, dv, cpx_cst.CPX_CON_LOWER_BOUND))
                if dv.ub <= 0.5:
                    grs.append( make_atom_group(var_bounds_pref, dv, cpx_cst.CPX_CON_UPPER_BOUND))

        # add pwl with 1

        # add sos with 1
        sos_pref = 1.0
        for sos in mdl.iter_sos():
            grs.append(make_atom_group(sos_pref, sos, cpx_cst.CPX_CON_SOS))

        return grs

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        try:
            # sync parameters
            mdl._apply_parameters_to_engine(parameters)
            self.sync_cplex(mdl)
            cpx = self._cplex
            use_all = False
            if not groups:
                # no groups are specified.
                # emulate cplex interactive here:
                # linear constraints
                if not use_all:
                    cplex_def_grs = self._make_cplex_default_groups(mdl)
                    #print("--- initial #groups = {0}".format(len(cplex_def_grs)))
                    cpx.conflict.refine(*cplex_def_grs)
                else:
                    all_constraints = cpx.conflict.all_constraints()
                    if preferences:
                        grps = self._build_weighted_constraints(mdl, all_constraints._gp, preferences)
                        cpx.conflict.refine(grps)
                    else:
                        cpx.conflict.refine(all_constraints)
            else:
                groups_def = [self._build_group_definition_with_index(grp) for grp in groups]
                cpx.conflict.refine(*groups_def)

            return self._get_conflicts_local(mdl, cpx)

        except DOcplexException as docpx_e:  # pragma: no cover
            mdl._set_solution(None)
            raise docpx_e

    def _build_group_definition_with_index(self, cts_group):
        return cts_group.preference, tuple([(self._get_refiner_type(ct), ct.index)
                                            for ct in cts_group.iter_constraints()])

    def _get_refiner_type(self, conflict_arg):
        cpx_adapter = self.cpx_adapter
        if isinstance(conflict_arg, VarLbConstraintWrapper):
            return cpx_adapter.cpx_cst.CPX_CON_LOWER_BOUND
        elif isinstance(conflict_arg, VarUbConstraintWrapper):
            return cpx_adapter.cpx_cst.CPX_CON_UPPER_BOUND
        elif conflict_arg.is_linear():
            return cpx_adapter.cpx_cst.CPX_CON_LINEAR
        elif conflict_arg.is_logical():
            return cpx_adapter.cpx_cst.CPX_CON_INDICATOR
        elif conflict_arg.is_quadratic:
            return cpx_adapter.cpx_cst.CPX_CON_QUADRATIC
        else:
            conflict_arg.model.fatal("Type unknown (or not supported yet) for constraint: " + repr(conflict_arg))

    def _build_weighted_constraints(self, mdl, groups, preferences=None):
        cpx_cst = self.cpx_adapter.cpx_cst
        weighted_groups = []
        for (pref, seq) in groups:
            for (_type, _id) in seq:
                if _type == cpx_cst.CPX_CON_LOWER_BOUND or _type == cpx_cst.CPX_CON_UPPER_BOUND:
                    # Keep default preference
                    weighted_groups.append((pref, ((_type, _id),)))
                else:
                    if preferences is not None:
                        ct = mdl.get_constraint_by_index(_id)
                        new_pref = preferences.get(ct, None)
                        if new_pref is not None and isinstance(new_pref, numbers.Number):
                            pref = new_pref

                    weighted_groups.append((pref, ((_type, _id),)))
        return weighted_groups

    def _get_conflicts_local(self, mdl, cpx):
        def var_by_index(idx):
            return mdl.get_var_by_index(idx)

        try:
            cpx_conflicts = cpx.conflict.get()
            groups = cpx.conflict.get_groups()
        except self.cpx_adapter.CplexSolverError:
            # Return an empty list if no conflict is available
            return ConflictRefinerResult(conflicts=[], refined_by="cplex")

        cpx_cst = self.cpx_adapter.cpx_cst

        conflicts = []
        for (pref, seq), status in zip(groups, cpx_conflicts):
            if status == cpx_cst.CPX_CONFLICT_EXCLUDED:
                continue
            c_status = ConflictStatus(status)
            for (_type, _id) in seq:
                """
                Possible values for elements of grptype:
                    CPX_CON_LOWER_BOUND 	1 	variable lower bound
                    CPX_CON_UPPER_BOUND 	2 	variable upper bound
                    CPX_CON_LINEAR 	        3 	linear constraint
                    CPX_CON_QUADRATIC 	    4 	quadratic constraint
                    CPX_CON_SOS 	        5 	special ordered set
                    CPX_CON_INDICATOR 	    6 	indicator constraint
                """
                if _type == cpx_cst.CPX_CON_LOWER_BOUND:
                    dv = var_by_index(_id)
                    conflicts.append(TConflictConstraint(dv.name, VarLbConstraintWrapper(dv), c_status))

                if _type == cpx_cst.CPX_CON_UPPER_BOUND:
                    dv = var_by_index(_id)
                    conflicts.append(TConflictConstraint(dv.name, VarUbConstraintWrapper(dv), c_status))

                if _type == cpx_cst.CPX_CON_LINEAR:
                    ct = mdl.get_constraint_by_index(_id)
                    conflicts.append(TConflictConstraint(ct.name, ct, c_status))

                if _type == cpx_cst.CPX_CON_QUADRATIC:
                    ct = mdl.get_quadratic_constraint_by_index(_id)
                    conflicts.append(TConflictConstraint(ct.name, ct, c_status))

                if _type == cpx_cst.CPX_CON_SOS:
                    # TODO: return the SOS
                    sos = mdl._get_sos_by_index(_id)
                    if sos:
                        conflicts.append(TConflictConstraint(sos.name, sos.as_constraint(), c_status))

                if _type == cpx_cst.CPX_CON_INDICATOR:
                    ct = mdl.get_logical_constraint_by_index(_id)
                    conflicts.append(TConflictConstraint(ct.name, ct, c_status))

        return ConflictRefinerResult(conflicts, refined_by=self.name)

    def export(self, out, exchange_format):
        if is_string(out):
            path = out
            self._resync_if_needed()
            try:
                if '.' in path:
                    self._cplex.write(path)
                else:
                    self._cplex.write(path, filetype="lp")

            except self.cpx_adapter.CplexSolverError as cpx_se:  # pragma: no cover
                if cpx_se.args[2] == 1422:
                    raise IOError("SAV export cannot open file: {}".format(path))
                else:
                    raise DOcplexException("CPLEX error in SAV export: {0!s}", cpx_se)
        else:
            # assume a file-like object
            filetype = exchange_format.filetype
            return self._cplex.write_to_stream(out, filetype)

    def end(self):
        """ terminate the engine, cannot find this in the doc.
        """
        # del _cplex is asynchronous, so might be executed later,
        # prefer this synchronous variant.
        #del self._cplex
        self._cplex.end()

        self._cplex = None

    _all_mip_problems = frozenset({'MIP', 'MILP', 'fixedMILP', 'MIQP', 'fixedMIQP'})
    # noinspection PyProtectedMember
    def is_mip(self):
        cpx = self._cplex
        cpx_problem_type = cpx.problem_type[cpx.get_problem_type()]
        return cpx_problem_type in self._all_mip_problems

    def register_callback(self, cb):
        return self._cplex.register_callback(cb)

    def unregister_callback(self, cb):
        return self._cplex.unregister_callback(cb)

    def connect_progress_listeners(self, progress_listeners, model):
        if self.is_mip():
            self_ccb = self._ccb
            if progress_listeners or self_ccb:
                if self_ccb is None:
                    self._ccb = self.register_callback(_get_connect_listeners_callback(self.cpx_adapter.cplex_module))
                    self_ccb = self._ccb
                self_ccb.initialize(progress_listeners, model)
        elif progress_listeners:
            m = self._model
            m.warning("Model: \"{}\" is not a MIP problem, progress listeners are disabled", m.name)

    def _compute_quality_metrics(self):
        qm_dict = {}

#       if self._last_solve_status is not None:
        with SilencedCplexContext(self._saved_log_output, self._cplex, self.error_handler, ) as silent_cplex:
            cpxs = silent_cplex.solution
            for qm in QualityMetric:
                qmcode = qm.code
                try:
                    qmf = cpxs.get_float_quality(qmcode)
                    qm_dict[qm.key] = qmf

                except (self.cpx_adapter.CplexError, self.cpx_adapter.CplexSolverError):
                    pass
                if qm.has_int:
                    try:
                        qmi = cpxs.get_integer_quality(qmcode)
                        if qmi >= 0:
                            # do something with int parameter?
                            qm_dict[qm.int_key] = qmi
                    except (self.cpx_adapter.CplexError, self.cpx_adapter.CplexSolverError):
                        pass
        # else:
        #     self._model.error('Model has not been solved: no quality metrics available.')
        return qm_dict

    def get_parameter_from_id(self, param_id):
        try:
            return self._cplex._env.parameters._get(param_id)
        except self.cpx_adapter.CplexError as cpx_e:
            self.error_handler.warning("Error getting value for parameter from  id \"{0}\": {1!s}", (param_id, cpx_e))
            return None

    def set_parameter_from_id(self, param_id, value, param_short_name=''):
        # sets a parameter from its internal ID
        # BEWARE: no value check is performed.
        try:
            self._cplex._env.parameters._set(param_id, value)
        except self.cpx_adapter.CplexError as cpx_e:
            cpx_msg = str(cpx_e)
            if cpx_msg.startswith("Bad parameter identifier"):
                self.error_handler.warning("Parameter id \"{0}\" is not recognized", (param_id,))
            else:  # pragma: no cover
                self.error_handler.error("Error setting parameter {0} (id: {1}) to value {2} - ignored"
                                         .format(param_short_name,
                                                 param_id,
                                                 value))

    def set_parameter(self, parameter, value):
        # no value check is up to the caller.
        # parameter is a DOcplex parameter object
        self.set_parameter_from_id(parameter.cpx_id, value, param_short_name=parameter.name)


    def get_parameter(self, parameter):
        try:
            return self._cplex._env.parameters._get(parameter.cpx_id)
        except self.cpx_adapter.CplexError:  # pragma: no cover
            return parameter.default_value

    def get_solve_status(self):  # pragma: no cover
        from docplex.util.status import JobSolveStatus
        # In this function we try to do the exact same mappings as the IloCplex C++ and Java classes.
        # However, this is not always possible since the C++ and Java implementations are not consistent
        # and sometimes they are even in error (see RTC-21923).

        cpx_status = self._cplex.solution.get_status()
        # what status for relaxed solutions??
        relaxed_solution_status = JobSolveStatus.INFEASIBLE_SOLUTION

        cpx_cst = self.cpx_adapter.cpx_cst
        #
        if cpx_status in {cpx_cst.CPXMIP_ABORT_FEAS,
                          cpx_cst.CPXMIP_DETTIME_LIM_FEAS,
                          cpx_cst.CPXMIP_FAIL_FEAS,
                          cpx_cst.CPXMIP_FAIL_FEAS_NO_TREE,
                          cpx_cst.CPXMIP_MEM_LIM_FEAS,
                          cpx_cst.CPXMIP_NODE_LIM_FEAS,
                          cpx_cst.CPXMIP_TIME_LIM_FEAS
                          }:
            return JobSolveStatus.FEASIBLE_SOLUTION

        elif cpx_status in {cpx_cst.CPXMIP_ABORT_INFEAS,
                            cpx_cst.CPXMIP_DETTIME_LIM_INFEAS,
                            cpx_cst.CPXMIP_FAIL_INFEAS,
                            cpx_cst.CPXMIP_FAIL_INFEAS_NO_TREE,
                            cpx_cst.CPXMIP_MEM_LIM_INFEAS,
                            cpx_cst.CPXMIP_NODE_LIM_INFEAS,
                            cpx_cst.CPXMIP_TIME_LIM_INFEAS
                            }:
            # Hit a limit without a feasible solution: We don't know anything about the solution.
            return JobSolveStatus.UNKNOWN
        elif cpx_status in {cpx_cst.CPXMIP_OPTIMAL,
                            cpx_cst.CPXMIP_OPTIMAL_TOL,
                            self.DEFAULT_CPX_STAT_MULTIOBJ_OPTIMAL}:
            return JobSolveStatus.OPTIMAL_SOLUTION
        elif cpx_status is cpx_cst.CPXMIP_SOL_LIM:
            #  return hasSolution(env, lp) ? JobSolveStatus.FEASIBLE_SOLUTION : JobSolveStatus.UNKNOWN;
            return JobSolveStatus.FEASIBLE_SOLUTION
        elif cpx_status in {cpx_cst.CPXMIP_INForUNBD,
                            self.DEFAULT_CPX_STAT_MULTIOBJ_INForUNBD}:
            return JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION
        elif cpx_status in {cpx_cst.CPXMIP_UNBOUNDED,
                            cpx_cst.CPXMIP_ABORT_RELAXATION_UNBOUNDED,
                            self.DEFAULT_CPX_STAT_MULTIOBJ_UNBOUNDED}:
            return JobSolveStatus.UNBOUNDED_SOLUTION
        elif cpx_status in {cpx_cst.CPXMIP_INFEASIBLE,
                            self.DEFAULT_CPX_STAT_MULTIOBJ_INFEASIBLE}:  # proven infeasible
            return JobSolveStatus.INFEASIBLE_SOLUTION
        elif cpx_status == cpx_cst.CPXMIP_OPTIMAL_INFEAS:  # optimal with unscaled infeasibilities
            # DANIEL: What exactly do we return here? There is an optimal solution but that solution is
            # infeasible after unscaling.
            return JobSolveStatus.OPTIMAL_SOLUTION

        # feasopt status values
        elif cpx_status in frozenset({
            cpx_cst.CPXMIP_ABORT_RELAXED,  # relaxed solution is available and can be queried
            cpx_cst.CPXMIP_FEASIBLE  # problem feasible after phase I and solution available
        }):
            return JobSolveStatus.FEASIBLE_SOLUTION
        ## -----------------
        ## relaxation (feasopt) there is a relaxed (but infeasible) solution
        ## for now we choose to return INFEASIBLE_SOLUTION
        ##
        elif cpx_status in {cpx_cst.CPXMIP_FEASIBLE_RELAXED_INF,
                            cpx_cst.CPXMIP_FEASIBLE_RELAXED_QUAD,
                            cpx_cst.CPXMIP_FEASIBLE_RELAXED_SUM
                            }:
            return relaxed_solution_status
        elif cpx_status in {cpx_cst.CPXMIP_OPTIMAL_RELAXED_INF,
                            cpx_cst.CPXMIP_OPTIMAL_RELAXED_QUAD,
                            cpx_cst.CPXMIP_OPTIMAL_RELAXED_SUM
                            }:
            return relaxed_solution_status
        elif cpx_status in {cpx_cst.CPX_STAT_FEASIBLE_RELAXED_INF,
                            cpx_cst.CPX_STAT_FEASIBLE_RELAXED_QUAD,
                            cpx_cst.CPX_STAT_FEASIBLE_RELAXED_SUM,
                            }:
            return relaxed_solution_status
        elif cpx_status in {cpx_cst.CPX_STAT_OPTIMAL_RELAXED_INF,
                            cpx_cst.CPX_STAT_OPTIMAL_RELAXED_QUAD,
                            cpx_cst.CPX_STAT_OPTIMAL_RELAXED_SUM}:
            return relaxed_solution_status

        ## -------------------

        # populate status values
        elif cpx_status in {cpx_cst.CPXMIP_OPTIMAL_POPULATED
                            # ,cpx_cst.CPXMIP_OPTIMAL_POPULATED_TO
                            }:
            return JobSolveStatus.OPTIMAL_SOLUTION
        elif cpx_status == cpx_cst.CPXMIP_POPULATESOL_LIM:
            # minimal value for CPX_PARAM_POPULATE_LIM is 1! So there must be a solution
            return JobSolveStatus.FEASIBLE_SOLUTION

        elif cpx_status == cpx_cst.CPX_STAT_OPTIMAL:
            return JobSolveStatus.OPTIMAL_SOLUTION

        elif cpx_status == cpx_cst.CPX_STAT_INFEASIBLE:
            return JobSolveStatus.INFEASIBLE_SOLUTION

        # cpx_cst.CPX_STAT_ABORT_USER:
        # cpx_cst.CPX_STAT_ABORT_DETTIME_LIM:
        # cpx_cst.CPX_STAT_ABORT_DUAL_OBJ_LIM:
        # cpx_cst.CPX_STAT_ABORT_IT_LIM:
        # cpx_cst.CPX_STAT_ABORT_PRIM_OBJ_LIM:
        # cpx_cst.CPX_STAT_ABORT_TIME_LIM:
        #   switch (primalDualFeasible(env, lp)) {
        #   case PRIMAL_FEASIBLE: return JobSolveStatus.FEASIBLE_SOLUTION
        #   case PRIMAL_DUAL_FEASIBLE: return JobSolveStatus.OPTIMAL_SOLUTION
        #   case DUAL_FEASIBLE: return JobSolveStatus.UNKNOWN;
        #   default: return JobSolveStatus.UNKNOWN;
        #   }
        #
        # cpx_cst.CPX_STAT_ABORT_OBJ_LIM:
        #   /** DANIEL: Our Java API returns ERROR here while the C++ API returns Feasible if primal feasible
        #    *         and Unknown otherwise. Since we don't have ERROR in IloSolveStatus we emulate the
        #    *         C++ behavior (this is more meaningful anyway). In the long run we should make sure
        #    *         all the APIs behave in the same way.
        #    */
        #   switch (primalDualFeasible(env, lp)) {
        #   case PRIMAL_FEASIBLE: return JobSolveStatus.FEASIBLE_SOLUTION
        #   case PRIMAL_DUAL_FEASIBLE: return JobSolveStatus.FEASIBLE_SOLUTION
        #   default: return JobSolveStatus.UNKNOWN;
        #   }
        #
        # cpx_cst.CPX_STAT_FIRSTORDER:
        #   // See IloCplexI::CplexToAlgorithmStatus()
        #   return primalFeasible(env, lp) ? JobSolveStatus.FEASIBLE_SOLUTION : JobSolveStatus.UNKNOWN;

        elif cpx_status == cpx_cst.CPX_STAT_CONFLICT_ABORT_CONTRADICTION:
            # Numerical trouble in conflict refiner.
            #  DANIEL: C++ and Java both return Error here although a conflict is
            #          available (but nor proven to be minimal). This looks like a bug
            #          since no exception is thrown there. In IloSolveStatus we don't
            #          have ERROR, so we return UNKNOWN instead. This is fine for now
            #          since we do not support the conflict refiner anyway.
            #
            return JobSolveStatus.UNKNOWN

        elif cpx_status in {

            cpx_cst.CPX_STAT_CONFLICT_ABORT_DETTIME_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_IT_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_MEM_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_NODE_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_OBJ_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_TIME_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_USER
        }:
            # /** DANIEL: C++ and Java return Error here. This is almost certainly wrong.
            # *         Docs say "a conflict is available but not minimal".
            #  *         This is particularly erroneous if no exception gets thrown.
            #  *         See RTC-21923.
            #  *         In IloSolveStatus we don't have ERROR, so we return UNKNOWN instead.
            #  *         This should not be a problem since right now we don't support the
            #  *         conflict refiner anyway.
            #  */
            return JobSolveStatus.UNKNOWN
        elif cpx_status == cpx_cst.CPX_STAT_CONFLICT_FEASIBLE:
            return JobSolveStatus.FEASIBLE_SOLUTION
        elif cpx_status == cpx_cst.CPX_STAT_CONFLICT_MINIMAL:
            return JobSolveStatus.INFEASIBLE_SOLUTION

        elif cpx_status == cpx_cst.CPX_STAT_FEASIBLE:
            return JobSolveStatus.FEASIBLE_SOLUTION

        elif cpx_status == cpx_cst.CPX_STAT_NUM_BEST:
            #  Solution available but not proved optimal (due to numeric difficulties)
            # assert(hasSolution(env, lp));
            return JobSolveStatus.UNKNOWN

        elif cpx_status == cpx_cst.CPX_STAT_OPTIMAL_INFEAS:  # infeasibilities after unscaling
            # assert(hasSolution(env, lp));
            return JobSolveStatus.OPTIMAL_SOLUTION

        elif cpx_status == cpx_cst.CPX_STAT_INForUNBD:  # Infeasible or unbounded in presolve.
            return JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION
        elif cpx_status == cpx_cst.CPX_STAT_OPTIMAL_FACE_UNBOUNDED:
            #    unbounded optimal face (barrier only)
            # // CPX_STAT_OPTIMAL_FACE_UNBOUNDED is explicitly an error in Java and implicitly (fallthrough)
            # // an error in C++. So it should be fine to produce an error here as well.
            # // In IloSolveStatus we don't have ERROR, so we return UNKNOWN instead.
            # // In case of ERROR we should have seen a non-zero status anyway and the
            # // user should not care too much about the returned status.
            return JobSolveStatus.UNKNOWN
        elif cpx_status == cpx_cst.CPX_STAT_UNBOUNDED:
            # definitely unbounded
            return JobSolveStatus.UNBOUNDED_SOLUTION
        elif cpx_status == self.DEFAULT_CPX_STAT_MULTIOBJ_NON_OPTIMAL:
            # Solution available but not proved optimal
            return JobSolveStatus.FEASIBLE_SOLUTION
        elif cpx_status == self.DEFAULT_CPX_STAT_MULTIOBJ_STOPPED:
            # The solve of a multi-objective problem was interrupted (a global work limit was hit or solution process
            # was aborted)
            return JobSolveStatus.UNKNOWN
        else:
            return JobSolveStatus.UNBOUNDED_SOLUTION

    def populate(self, **kwargs):
        mdl = self._model
        cpx = self._cplex
        self.sync_cplex(mdl)
        if kwargs.get('clean_before_solve'):
            self.clean_before_solve()

        populate_sol = self._run_cpx_solve_fn(cpx.populate_solution_pool,
                                             self._CPLEX_SOLVE_OK_STATUSES)
        if not populate_sol:
            return None, None
        else:
            cpx_solp = cpx.solution.pool
            numsol = cpx_solp.get_num()
            num_replaced = cpx_solp.get_num_replaced()

            nb_vars = mdl.number_of_variables
            pool_sols = []
            basename = mdl.name # normalize?

            for p in range(numsol):
                solpvals = cpx_solp.get_values(p)
                solpobj = cpx_solp.get_objective_value(p)
                solpslacks = cpx_solp.get_linear_slacks(p)
                assert len(solpvals) == nb_vars
                sol_name = "{0}_pool_#{1}".format(basename, p)
                sol = mdl.new_solution(var_value_dict={dv: solpvals[dv.index] for dv in mdl.iter_variables()}, name=sol_name,
                                       objective_value=solpobj)
                sol.store_attribute_lists(mdl, solpslacks)

                pool_sols.append(sol)

            solnpool = SolutionPool(pool_sols, num_replaced)
            return populate_sol, solnpool

    def resync(self):
        # life buoy
        self._resync_if_needed()

    def _resync_if_needed(self):
        if self._sync_mode is _CplexSyncMode.OutOfSync:
            # print("-- resync cplex from model...")
            try:
                self._sync_mode = _CplexSyncMode.InResync
                self._model._resync()
            finally:
                self._sync_mode = _CplexSyncMode.InSync


@contextmanager
def overload_cplex_parameter_values(cpx_engine, overload_dict):
    old_values = {p: p.get() for p in overload_dict}
    try:
        yield cpx_engine
    finally:
        # restore params
        for p, saved_value in six.iteritems(old_values):
            p.set(saved_value)


def unpickle_cplex_engine(mdl, is_traced):
    #  INTERNAL
    unpicking_env = Environment()
    if unpicking_env.has_cplex:
        cplex_engine = CplexEngine(mdl)
        cplex_engine.set_streams(sys.stdout if is_traced else None)  # what to do if file??
        # mark to be resync'ed
        cplex_engine._mark_as_out_of_sync()
        return cplex_engine
    else:
        return NoSolveEngine.make_from_model(mdl)


unpickle_cplex_engine.__safe_for_unpickling__ = True


def pickle_cplex_engine(cplex_engine):
    model = cplex_engine._model
    return unpickle_cplex_engine, (model, model.is_logged())


copyreg.pickle(CplexEngine, pickle_cplex_engine)
