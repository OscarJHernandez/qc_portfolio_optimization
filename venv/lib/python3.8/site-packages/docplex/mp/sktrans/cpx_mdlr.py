# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017-2020
# --------------------------------------------------------------------------
# gendoc: ignore


from docplex.mp.utils import is_ordered_sequence, is_number, is_string
from docplex.mp.compat23 import izip
from docplex.mp.constants import ComparisonType
from docplex.mp.aggregator import ModelAggregator
from docplex.mp.sktrans.pd_utils import make_solution
from docplex.mp.cplex_adapter import CplexAdapter


class CpxModeler(object):

    def __init__(self):
        pass

    @classmethod
    def create_cplex_adapter(cls, verbose=False):
        adapter = CplexAdapter()
        if not verbose:
            adapter.cpx.set_log_stream(None)
            adapter.cpx.set_results_stream(None)
            adapter.cpx.set_error_stream(None)
            adapter.cpx.set_warning_stream(None)

        cpx_datacheck_id = 1056
        adapter.setintparam(adapter.cpx._env._e, cpx_datacheck_id, 0)
        return adapter

    @classmethod
    def create_cpx_bound_list(cls, bounds, size):
        if bounds is None:
            return []
        elif is_number(bounds):
            return [bounds] * size
        elif is_ordered_sequence(bounds):
            assert size == len(bounds)
            return [float(bb) for bb in bounds]
        else:
            raise ValueError("Expecting number or sequence of numbers, {0!r} was passed".format(bounds))

    @classmethod
    def create_cpx_vartype_string(cls, vartypes, size):
        if vartypes is None:
            return ""
        else:
            if not is_string(vartypes):
                raise ValueError(
                    "Expecting a string for variable types with [BICSN]*, {0!r} was passed".format(vartypes))
            else:
                vtl = len(vartypes)
                if vtl > 1 and vtl != size:
                    raise ValueError(
                        "Expecting a string for variable types with len 0, 1 or #vars={0}, size: {1} was passed"
                            .format(size, vtl))
                # check each char
                alltypes = "BICNS"
                for i, c in enumerate(vartypes):
                    if c not in alltypes:
                        raise ValueError("Incorrect character in types: {0}, pos: {1}".format(c, i))
                return vartypes * size if vtl == 1 else vartypes

    @classmethod
    def create_column_vars(cls, cpx, nb_vars, var_lbs, var_ubs, var_types, colnames):
        cpx_lbs = cls.create_cpx_bound_list(var_lbs, nb_vars)
        cpx_ubs = cls.create_cpx_bound_list(var_ubs, nb_vars)
        cpx_types = cls.create_cpx_vartype_string(var_types, nb_vars)
        # create N continuous variables.
        # Cplex requires at least one list with size N

        if not (cpx_lbs or cpx_ubs or ((colnames is not None) and len(colnames))):
            # force at least one list with correct size.
            cpx_lbs = [0] * nb_vars
        cpxnames = [] if colnames is None else colnames
        cpx.variables.add(types=cpx_types, names=cpxnames, lb=cpx_lbs, ub=cpx_ubs)

    @classmethod
    def build_matrix_linear_model_and_solve(cls, var_count, var_lbs, var_ubs,
                                            var_types, var_names,
                                            cts_mat, rhs,
                                            objsense, costs,
                                            cast_to_float,
                                            solution_maker=make_solution,
                                            **transform_params):
        adapter = cls.create_cplex_adapter()
        cpx = adapter.cpx
        if cast_to_float:
            print("-- all numbers will be cast to float")
        else:
            print("-- no cast to float is performed")
        cls.create_column_vars(cpx, var_count, var_lbs, var_ubs, var_types, var_names)
        var_indices = list(range(var_count))
        gen_rows = ModelAggregator.generate_rows(cts_mat)
        cpx_rows = []
        if cast_to_float:
            for row in gen_rows:
                # need this step as cplex may crash with np types.
                frow = [float(k) for k in row]
                cpx_rows.append([var_indices, frow])
        else:
            cpx_rows = [[var_indices, row] for row in gen_rows]

        nb_rows = len(cpx_rows)
        if nb_rows:
            ctsense = ComparisonType.parse(transform_params.get('sense', 'le'))
            cpx_senses = ctsense.cplex_code * nb_rows
            cpx_rhss = [float(r) for r in rhs] if cast_to_float else rhs
            adapter.add_linear(cpx, cpx_rows, cpx_senses, cpx_rhss, names=[])
        if costs is not None:
            # set linear objective for all variables.
            fcosts = [float(k) for k in costs]
            adapter.static_fast_set_linear_obj(cpx, var_indices, fcosts)
            cpx.objective.set_sense(objsense.cplex_coef)
        # here we go to solve...
        return cls._solve(cpx, var_names, solution_maker=solution_maker, **transform_params)

    @classmethod
    def _solve(cls, cpx, colnames, solution_maker=make_solution, **params):
        # --- lp export
        lp_export = params.pop('lp_export', False)
        lp_base = params.pop('lp_basename', None)
        lp_path = params.pop('lp_path', None)

        # ---
        keep_zeros = params.pop('keep_zeros', True)

        # -- solving
        do_solve = params.pop("solve", True)
        all_values = None
        if do_solve:
            cpx.solve()
            try:
                all_values = cpx.solution.get_values()
            except:
                pass
        return solution_maker(all_values, colnames, keep_zeros)

    @classmethod
    def build_sparse_linear_model_and_solve(cls, nb_vars, var_lbs, var_ubs,
                                            var_types, var_names,
                                            nb_rows, cts_sparse_coefs,
                                            objsense, costs,
                                            solution_maker=make_solution,
                                            **transform_params):
        adapter = cls.create_cplex_adapter()
        cpx = adapter.cpx
        # varlist = mdl.continuous_var_list(var_count, lb=var_lbs, ub=var_ubs, name=var_names)
        cls.create_column_vars(cpx, nb_vars, var_lbs, var_ubs, var_types, var_names)
        var_indices = list(range(nb_vars))
        cpx_linexprs = [([], []) for _ in range(nb_rows)]
        cpx_rhss = [0.] * nb_rows
        for coef, row, col in cts_sparse_coefs:
            if col >= nb_vars:
                cpx_rhss[row] = float(coef)
            elif coef:
                cpx_row = cpx_linexprs[row]
                #  int() conversio nis mandatory here
                # as sparse matrices contain numpy int types -> cause cplex to crash
                cpx_row[0].append(int(col))
                cpx_row[1].append(float(coef))
        ctsense = ComparisonType.parse(transform_params.get('sense', 'le'))
        cpx_senses = ctsense.cplex_code * nb_rows
        #fast_add_linear(cpx, cpx_linexprs, cpx_senses, cpx_rhss, names=[])
        cpx.linear_constraints.add(cpx_linexprs, cpx_senses, cpx_rhss, names=[])

        if costs is not None:
            # set linear objective for all variables.
            fcosts = [float(k) for k in costs]
            adapter.static_fast_set_linear_obj(cpx, var_indices, fcosts)
            cpx.objective.set_sense(objsense.cplex_coef)
        # here we go to solve...
        return cls._solve(cpx, var_names, solution_maker=solution_maker, **transform_params)

    @classmethod
    def build_matrix_range_model_and_solve(cls, var_count, var_lbs, var_ubs,
                                           var_types, var_names,
                                           cts_mat, range_mins, range_maxs,
                                           objsense, costs,
                                           cast_to_float,
                                           solution_maker=make_solution,
                                           **transform_params):
        adapter = cls.create_cplex_adapter()
        cpx = adapter.cpx
        # varlist = mdl.continuous_var_list(var_count, lb=var_lbs, ub=var_ubs, name=var_names)
        cls.create_column_vars(cpx, var_count, var_lbs, var_ubs, var_types, var_names)
        var_indices = list(range(var_count))
        gen_rows = ModelAggregator.generate_rows(cts_mat)
        cpx_rows = []
        for row in gen_rows:
            # need this step as cplex may crash with np types.
            frow = [float(k) for k in row] if cast_to_float else row
            cpx_rows.append([var_indices, frow])
        nb_rows = len(cpx_rows)
        if nb_rows:
            #ctsense = ComparisonType.parse(transform_params.get('sense', 'le'))
            cpx_senses = 'R' * nb_rows
            cpx_ranges = [float(rmin - rmax) for rmin, rmax in izip(range_mins, range_maxs)]
            # rhs is UB
            cpx_rhss = [float(rmax) for rmax in range_maxs]
            adapter.add_linear(cpx, cpx_rows, cpx_senses, rhs=cpx_rhss, names=[], ranges=cpx_ranges)
        if costs is not None:
            # set linear objective for all variables.
            fcosts = [float(k) for k in costs]
            adapter.static_fast_set_linear_obj(cpx, var_indices, fcosts)
            cpx.objective.set_sense(objsense.cplex_coef)
        # here we go to solve...
        return cls._solve(cpx, var_names, solution_maker=solution_maker, **transform_params)
