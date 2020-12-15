# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2019, 2020
# --------------------------------------------------------------------------

# gendoc: ignore

from six import iteritems

from docplex.mp.cloudutils import is_in_docplex_worker
from docplex.mp.cloudutils import make_new_kpis_dict
from docplex.mp.publish import auto_publishing_kpis_table_names, \
    auto_publishing_result_output_names, _KpiRecorder
from docplex.util.environment import get_environment
from docplex.mp.context import is_auto_publishing_solve_details

from docplex.mp.publish import write_kpis_table, write_result_output

from docplex.mp.utils import apply_thread_limitations

try:
    import numpy
except ImportError:
    numpy = None

class SolveEnv(object):

    def __init__(self, mdl):
        self._model = mdl

    def before_solve(self, context):
        pass

    def after_solve(self, context, solve_res, engine):
        mdl = self._model
        mdl._set_solution(solve_res)

        solve_details = engine.get_solve_details()
        mdl._notify_solve_hit_limit(solve_details)
        mdl._solve_details = solve_details


class DocloudSolveEnv(SolveEnv):

    def __init__(self, m):
        super(DocloudSolveEnv, self).__init__(m)

    def before_solve(self, context):
        mdl = self._model
        mdl.notify_start_solve()
        mdl._fire_start_solve_listeners()
        return mdl.parameters


class CplexLocalSolveEnv(SolveEnv):

    def __init__(self, m):
        super(CplexLocalSolveEnv, self).__init__(m)
        # cached number of threads, if overwritten by solve
        self._saved_params = {}

    def before_solve(self, context):
        mdl = self._model
        auto_publish_details = is_auto_publishing_solve_details(context)

        # step 1 : notify start solve
        mdl.notify_start_solve()

        the_env = get_environment()
        if is_in_docplex_worker() and auto_publish_details:
            # do not use lambda here
            def env_kpi_hookfn(kpd):
                the_env.update_solve_details(kpd)

            mdl.kpi_recorder = _KpiRecorder(mdl,
                                            clock=context.solver.kpi_reporting.filter_level,
                                            publish_hook=env_kpi_hookfn)
            mdl.add_progress_listener(mdl.kpi_recorder)

        # connect progress listeners (if any) if problem is mip
        mdl._connect_progress_listeners()

        # call notifyStart on progress listeners
        mdl._fire_start_solve_listeners()

        # The following block used to be published only if auto_publish_details.
        # It is now modified so that notify start is always performed,
        # then we update solve details only if they need to be published
        # [[[
        self_stats = mdl.statistics
        kpis = make_new_kpis_dict(allkpis=mdl._allkpis[:],
                                  int_vars=self_stats.number_of_integer_variables,
                                  continuous_vars=self_stats.number_of_continuous_variables,
                                  linear_constraints=self_stats.number_of_linear_constraints,
                                  bin_vars=self_stats.number_of_binary_variables,
                                  quadratic_constraints=self_stats.number_of_quadratic_constraints,
                                  total_constraints=self_stats.number_of_constraints,
                                  total_variables=self_stats.number_of_variables)
        # implementation for https://github.ibm.com/IBMDecisionOptimization/dd-planning/issues/2491
        problem_type = mdl._get_cplex_problem_type()
        kpis['STAT.cplex.modelType'] = problem_type
        kpis['MODEL_DETAIL_OBJECTIVE_SENSE'] = mdl._objective_sense.verb

        the_env.notify_start_solve(kpis)
        if auto_publish_details:
            the_env.update_solve_details(kpis)
        # ]]]

        # ---
        #  parameters override if necessary...
        # --
        self_params = mdl.context._get_raw_cplex_parameters()
        parameters = apply_thread_limitations(context)
        if self_params and parameters is not self_params:
            self._saved_params = {p: p.get() for p in self_params}
        else:
            self._saved_params = {}
        # ctx_params = context.cplex_parameters
        # mdl_params = mdl.parameters
        # corrected_threads = compute_overwrite_nb_threads(ctx_params, context.solver)
        # params_to_use = ctx_params or mdl_params
        # if corrected_threads is not None:
        #     params_to_use = params_to_use.copy()
        #     params_to_use.threads = corrected_threads

        # if parameters are overridden, save original values for restoration at solve end
        # if params_to_use is not mdl_params:
        #     self._saved_params = {p: p.get() for p in mdl_params}
        # else:
        #     self._saved_params = {}

        # returns the parameters group to use.
        return parameters

    def after_solve(self, context, solve_res, engine):
        mdl = self._model
        super(CplexLocalSolveEnv, self).after_solve(context, solve_res, engine)

        mdl._disconnect_progress_listeners()

        # --- specific to local
        the_env = get_environment()
        the_env.notify_end_solve(mdl.solve_status)

        if is_auto_publishing_solve_details(context):
            details = mdl.solve_details.as_worker_dict()
            if solve_res:
                new_solution = solve_res
                kpis = mdl.kpis_as_dict(new_solution, use_names=True)

                def publish_name_fn(kn):
                    return 'KPI.%s' % kn
                
                def convert_from_numpy(o):
                    return o.item() if isinstance(o, numpy.generic) else o

                def identity(o):
                    return o
                
                publish_value = convert_from_numpy if numpy else identity
                # build a dict of kpi names (formatted) -> kpi values
                # kpi values are converted to python types if they are numpy types
                kpi_details_dict = {publish_name_fn(kn): publish_value(kv) for kn, kv in iteritems(kpis)}
                details.update(kpi_details_dict)
                # add objective with predefined key name
                details['PROGRESS_CURRENT_OBJECTIVE'] = new_solution.objective_value
            the_env.update_solve_details(details)

        if solve_res:
            auto_publish_solution = auto_publishing_result_output_names(context) is not None
            auto_publish_kpis_table = auto_publishing_kpis_table_names(context) is not None
            if auto_publish_solution:
                write_result_output(env=the_env,
                                    context=context,
                                    model=mdl,
                                    solution=solve_res)

            # save kpi
            if auto_publish_kpis_table:
                write_kpis_table(env=the_env,
                                 context=context,
                                 model=mdl,
                                 solution=solve_res)

        # restore tcached params
        saved_params = self._saved_params
        if saved_params:
            self_engine = mdl.get_engine()
            for p, v in iteritems(saved_params):
                self_engine.set_parameter(p, v)
                # clear saved
                self._saved_params = {}
