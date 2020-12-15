# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017
# --------------------------------------------------------------------------
# gendoc: ignore

from docplex.mp.advmodel import AdvModel
from docplex.mp.sktrans.pd_utils import make_solution


class DOcplexModeler(object):

    def __init__(self):
        pass

    @classmethod
    def _solve_model(cls, mdl, cols, colnames, objsense, costs, solution_maker=make_solution, **params):
        if costs is not None:
            mdl.set_objective(sense=objsense, expr=mdl.scal_prod_vars_all_different(cols, costs))


        # --- lp export
        lp_export = params.pop('lp_export', False)
        lp_base = params.pop('lp_basename', None)
        lp_path = params.pop('lp_path', None)
        if lp_export:
            mdl.export_as_lp(basename=lp_base, path=lp_path)
        # ---
        keep_zeros = params.pop('keep_zeros', False)
        #mdl.prettyprint()


        # -- solving
        s = mdl.solve()
        all_values = s.get_all_values() if s else None
        return solution_maker(all_values, colnames, keep_zeros)


    @classmethod
    def build_matrix_linear_model_and_solve(cls, var_count, var_lbs, var_ubs,
                                            var_types, var_names,
                                            cts_mat, rhs,
                                            objsense, costs,
                                            cast_to_float,
                                            solution_maker=make_solution,
                                            **transform_params):
        with AdvModel(name='lp_transformer') as mdl:
            varlist = mdl.continuous_var_list(var_count, lb=var_lbs, ub=var_ubs, name=var_names)
            sense = transform_params.get('sense', 'le')
            mdl.add(mdl.matrix_constraints(cts_mat, varlist, rhs, sense=sense))
            return cls._solve_model(mdl, varlist, var_names, objsense, costs=costs, solution_maker=solution_maker,
                                    **transform_params)

    @classmethod
    def build_matrix_range_model_and_solve(cls, var_count, var_lbs, var_ubs,
                                           var_types, var_names,
                                           cts_mat, range_mins, range_maxs,
                                           objsense, costs,
                                           cast_to_float,
                                           solution_maker=make_solution,
                                           **transform_params):
        with AdvModel(name='lpr_transformer') as mdl:
            varlist = mdl.continuous_var_list(var_count, lb=var_lbs, ub=var_ubs, name=var_names)
            mdl.add(mdl.matrix_ranges(cts_mat, varlist, range_mins, range_maxs))
            return cls._solve_model(mdl, varlist, var_names, objsense, costs=costs, solution_maker=solution_maker,
                                    **transform_params)

    @classmethod
    def build_sparse_linear_model_and_solve(cls, nb_vars, var_lbs, var_ubs,
                                            var_types, var_names,
                                            nb_rows, cts_sparse_coefs,
                                            objsense, costs,
                                            solution_maker=make_solution,
                                            **transform_params):
        with AdvModel(name='lp_transformer') as mdl:
            varlist = mdl.continuous_var_list(nb_vars, lb=var_lbs, ub=var_ubs)
            lfactory = mdl._lfactory
            r_rows = range(nb_rows)
            exprs = [lfactory.linear_expr() for _ in r_rows]
            rhss = [0] * nb_rows
            for coef, row, col in cts_sparse_coefs:
                if col < nb_vars:
                    exprs[row]._add_term(varlist[col], coef)
                else:
                    assert col == nb_vars
                    rhss[row] = coef
            sense = transform_params.get('sense', 'le')
            cts = [lfactory.new_binary_constraint(exprs[r], rhs=rhss[r], sense=sense) for r in r_rows]
            lfactory._post_constraint_block(cts)
            return cls._solve_model(mdl, varlist, var_names, objsense=objsense, costs=costs,
                                    solution_maker=solution_maker, **transform_params)
