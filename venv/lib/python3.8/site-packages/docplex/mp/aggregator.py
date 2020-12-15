# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from six import itervalues, iteritems
from docplex.mp.compat23 import izip

from docplex.mp.xcounter import update_dict_from_item_value

from docplex.mp.utils import is_number, is_iterable, is_iterator, is_pandas_series, \
    is_numpy_ndarray, is_pandas_dataframe, is_numpy_matrix, is_ordered_sequence
from docplex.mp.linear import MonomialExpr, AbstractLinearExpr, LinearExpr, ZeroExpr
from docplex.mp.dvar import Var
from docplex.mp.functional import _FunctionalExpr
from docplex.mp.operand import LinearOperand
from docplex.mp.quad import QuadExpr, VarPair


class ModelAggregator(object):
    # what type to use for merging dicts

    def __init__(self, linear_factory, quad_factory):
        self._linear_factory = linear_factory
        self._checker = linear_factory._checker
        self._quad_factory = quad_factory
        self._model = linear_factory._model
        self._generate_transients = True

    @property
    def counter_type(self):
        return self._linear_factory.term_dict_type

    def new_zero_expr(self):
        return ZeroExpr(model=self._model)

    def _to_expr(self, qcc, lcc=None, constant=0):
        # no need to sort here, sort is done by str() on the fly.
        if qcc:
            linear_expr = LinearExpr(self._model, e=lcc, constant=constant, safe=True)
            quad = self._quad_factory.new_quad(quads=qcc, linexpr=linear_expr, safe=True)
            quad._transient = self._generate_transients
            return quad
        elif lcc or constant:
            linear_expr = LinearExpr(self._model, e=lcc, constant=constant, safe=True)
            linear_expr._transient = self._generate_transients
            return linear_expr
        else:
            return self.new_zero_expr()

    def scal_prod(self, terms, coefs=1.0):
        # Testing anumpy array for its logical value will not work:
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # we would have to trap the test for ValueError then call any()
        #
        if is_iterable(coefs):
            pass  # ok
        elif is_number(coefs):
            if 0 == coefs:
                return self.new_zero_expr()
            else:
                sum_expr = self.sum(terms)
                return sum_expr * coefs
        else:
            self._model.fatal("scal_prod expects iterable or number, {0!s} was passed", coefs)

        # model has checked terms is an ordered sequence
        return self._scal_prod(terms, coefs)

    def _scal_prod(self, terms, coefs):
        # INTERNAL
        checker = self._checker
        total_num = 0
        lcc = self.counter_type()
        qcc = None

        number_validation_fn = checker.get_number_validation_fn()

        for item, coef in izip(terms, coefs):
            if not coef:
                continue

            safe_coef = number_validation_fn(coef) if number_validation_fn else coef
            if isinstance(item, Var):
                update_dict_from_item_value(lcc, item, safe_coef)

            elif isinstance(item, AbstractLinearExpr):
                total_num += safe_coef * item.get_constant()
                for lv, lk in item.iter_terms():
                    update_dict_from_item_value(lcc, lv, lk * safe_coef)

            elif isinstance(item, QuadExpr):
                if qcc is None:
                    qcc = self.counter_type()
                for qv, qk in item.iter_quads():
                    update_dict_from_item_value(qcc, qv, qk * safe_coef)
                qlin = item.get_linear_part()
                for v, k in qlin.iter_terms():
                    update_dict_from_item_value(lcc, v, k * safe_coef)

                total_num += safe_coef * qlin.constant

            # --- try conversion ---
            else:
                try:
                    e = item.to_linear_expr()
                    total_num += e.get_constant()
                    for dv, k, in e.iter_terms():
                        update_dict_from_item_value(lcc, dv, k * safe_coef)
                except AttributeError:
                    self._model.fatal("scal_prod accepts variables, expressions, numbers, not: {0!s}", item)

        return self._to_expr(qcc, lcc, total_num)

    def _scal_prod_f(self, dvars, coef_fn, assume_alldifferent):
        if isinstance(dvars, dict) and hasattr(dvars, 'items'):
            var_key_iter = iteritems
        elif is_ordered_sequence(dvars):
            var_key_iter = enumerate
        else:
            var_key_iter = None
            self._model.fatal('Model.dotf expects either a dictionary or an ordered sequence of variables, an instance of {0} was passed',
                              type(dvars))

        if assume_alldifferent:
            return self._scal_prod_f_alldifferent(dvars, coef_fn, var_key_iter)
        else:
            return self._scal_prod_f_gen(dvars, coef_fn, var_key_iter=var_key_iter)

    def _scal_prod_f_gen(self, dvars, coef_fn, var_key_iter):
        # var_map is a dictionary of variables.
        # coef_fn is a function accepting dictionary keys
        lcc_type = self.counter_type
        lcc = lcc_type()
        number_validation_fn = self._checker.get_number_validation_fn()
        if number_validation_fn:
            for k, dvar in var_key_iter(dvars):
                fcoeff = coef_fn(k)
                safe_coeff = number_validation_fn(fcoeff)
                if safe_coeff:
                    update_dict_from_item_value(lcc, dvar, safe_coeff)
        else:
            for k, dvar in var_key_iter(dvars):
                fcoeff = coef_fn(k)
                if fcoeff:
                    update_dict_from_item_value(lcc, dvar, fcoeff)

        return self._to_expr(qcc=None, lcc=lcc)

    def _scal_prod_f_alldifferent(self, dvars, coef_fn, var_key_iter):
        # var_map is a dictionary of variables.
        # coef_fn is a function accepting dictionary keys
        lcc_type = self.counter_type
        lcc = lcc_type()
        lcc_setitem = lcc_type.__setitem__
        number_validation_fn = self._checker.get_number_validation_fn()
        if number_validation_fn:
            for k, dvar in var_key_iter(dvars):
                fcoeff = coef_fn(k)
                safe_coeff = number_validation_fn(fcoeff)
                if safe_coeff:
                    lcc_setitem(lcc, dvar, safe_coeff)
        else:
            for k, dvar in var_key_iter(dvars):
                fcoeff = coef_fn(k)
                if fcoeff:
                    lcc_setitem(lcc, dvar, fcoeff)

        return self._to_expr(qcc=None, lcc=lcc)

    def sum(self, sum_args):
        if is_iterator(sum_args):
            sum_res = self._sum_with_iter(sum_args)
        elif is_numpy_ndarray(sum_args):
            sum_res = self._sum_with_iter(sum_args.flat)
        elif is_pandas_series(sum_args):
            sum_res = self.sum(sum_args.values)
        elif isinstance(sum_args, dict):
            # handle dict: sum all values
            sum_res = self._sum_with_iter(itervalues(sum_args))
        elif is_iterable(sum_args):
            sum_res = self._sum_with_seq(sum_args)

        elif is_number(sum_args):
            sum_res = sum_args
        else:
            sum_res = self._linear_factory._to_linear_operand(sum_args)
        return sum_res

    def _sum_with_iter(self, args):
        sum_of_nums = 0
        lcc = self.counter_type()
        checker = self._checker
        qcc = None
        number_validation_fn = checker.get_number_validation_fn()
        for item in args:
            if isinstance(item, LinearOperand):
                for lv, lk in item.iter_terms():
                    update_dict_from_item_value(lcc, lv, lk)
                itc = item.get_constant()
                if itc:
                    sum_of_nums += itc
            elif is_number(item):
                sum_of_nums += number_validation_fn(item) if number_validation_fn else item
            elif isinstance(item, QuadExpr):
                for lv, lk in item.linear_part.iter_terms():
                    update_dict_from_item_value(lcc, lv, lk)
                if qcc is None:
                    qcc = self.counter_type()
                for qvp, qk in item.iter_quads():
                    update_dict_from_item_value(qcc, qvp, qk)
                sum_of_nums += item.get_constant()

            else:
                try:
                    expr = item.to_linear_expr()
                    sum_of_nums += expr.get_constant()
                    for dv, k in expr.iter_terms():
                        update_dict_from_item_value(lcc, dv, k)
                except AttributeError:
                    self._model.fatal("Model.sum() expects numbers/variables/expressions, got: {0!s}", item)

        return self._to_expr(qcc, lcc, sum_of_nums)

    def _sum_vars(self, dvars):
        if is_numpy_ndarray(dvars):
            return self._sum_vars(dvars.flat)
        elif is_pandas_series(dvars):
            return self.sum(dvars.values)
        elif isinstance(dvars, dict):
            # handle dict: sum all values
            return self._sum_vars(itervalues(dvars))
        elif is_iterable(dvars):
            checked_dvars = self._checker.typecheck_var_seq(dvars, caller='Model.sumvars()')
            sumvars_terms = self._varlist_to_terms(checked_dvars)
            return self._to_expr(qcc=None, lcc=sumvars_terms)
        else:
            self._model.fatal('Model.sumvars() expects an iterable returning variables, {0!r} was passed',
                              dvars)

    def _sum_vars_all_different(self, dvars):
        lcc = self._linear_factory._new_term_dict()
        setitem_fn = lcc.__setitem__

        for v in dvars:
            setitem_fn(v, 1)
        return self._to_expr(qcc=None, lcc=lcc)

    def _varlist_to_terms(self, var_list):
        # INTERNAL: converts a sum of vars to a dict, sorting if needed.
        linear_term_dict_type = self._linear_factory.term_dict_type
        try:
            assume_no_dups = len(var_list) == len(set(var_list))
        except TypeError:
            assume_no_dups = False

        if assume_no_dups:
            varsum_terms = linear_term_dict_type()
            linear_terms_setitem = linear_term_dict_type.__setitem__
            for v in var_list:
                linear_terms_setitem(varsum_terms, v, 1)
        else:
            # there might be repeated variables.
            varsum_terms = linear_term_dict_type()
            for v in var_list:
                update_dict_from_item_value(varsum_terms, v, 1)
        return varsum_terms

    def _sum_with_seq(self, sum_args):
        for z in sum_args:
            if not isinstance(z, Var):
                x_seq_all_variables = False
                break
        else:
            x_seq_all_variables = True

        if x_seq_all_variables:
            return self._sum_vars(sum_args)
        else:
            return self._sum_with_iter(args=sum_args)

    def _sumsq(self, args):
        accumulated_ct = 0
        number_validation_fn = self._checker.get_number_validation_fn()
        qcc = self._quad_factory.term_dict_type()
        lcc = self._linear_factory.term_dict_type()

        for item in args:
            if isinstance(item, Var):
                update_dict_from_item_value(qcc, VarPair(item, item), 1)
            elif isinstance(item, MonomialExpr):
                mcoef = item._coef
                # noinspection PyPep8
                mvar = item._dvar
                update_dict_from_item_value(qcc, VarPair(mvar, mvar), mcoef ** 2)

            elif isinstance(item, LinearExpr):
                cst = item.get_constant()
                accumulated_ct += cst ** 2
                for lv1, lk1 in item.iter_terms():
                    for lv2, lk2 in item.iter_terms():
                        if lv1 is lv2:
                            update_dict_from_item_value(qcc, VarPair(lv1, lv1), lk1 * lk1)
                        elif lv1._index < lv2._index:
                            update_dict_from_item_value(qcc, VarPair(lv1, lv2), 2 * lk1 * lk2)
                        else:
                            pass

                    if cst:
                        update_dict_from_item_value(lcc, lv1, 2 * cst * lk1)
            elif isinstance(item, _FunctionalExpr):
                fvar = item.functional_var
                update_dict_from_item_value(qcc, VarPair(fvar), 1)

            elif isinstance(item, ZeroExpr):
                pass

            elif is_number(item):
                safe_item = number_validation_fn(item) if number_validation_fn else item
                accumulated_ct += safe_item ** 2

            else:
                self._model.fatal("Model.sumsq() expects numbers/variables/linear expressions, got: {0!s}", item)

        return self._to_expr(qcc, lcc, constant=accumulated_ct)

    def sumsq(self, sum_args):
        if is_iterable(sum_args):
            if is_iterator(sum_args):
                return self._sumsq(sum_args)
            elif isinstance(sum_args, dict):
                return self._sumsq(sum_args.values())
            elif is_numpy_ndarray(sum_args):
                return self._sumsq(sum_args.flat)
            elif is_pandas_series(sum_args):
                return self._sumsq(sum_args.values)

            else:
                return self._sumsq(sum_args)
        elif is_number(sum_args):
            return sum_args ** 2
        else:
            self._model.fatal("Model.sumsq() expects number/iterable/expression, got: {0!s}", sum_args)

    # --- matrix constraint
    @staticmethod
    def generate_df_rows(df):
        for row in df.itertuples(index=False):
            yield row

    @staticmethod
    def generate_np_matrix_rows(npm):
        for r in npm:
            yield r.tolist()[0]

    def _sparse_make_exprs(self, sp_mat, dvars, nb_exprs):
        lfactory = self._linear_factory
        exprs = [lfactory.linear_expr() for _ in range(nb_exprs)]
        coo_mat = sp_mat.tocoo()
        for coef, row, col in izip(coo_mat.data, coo_mat.row, coo_mat.col):
            exprs[row]._add_term(dvars[col], coef)
        return exprs

    def _sparse_matrix_constraints(self, sp_coef_mat, svars, srhs, op):
        range_cts = range(len(srhs))
        lfactory = self._linear_factory
        exprs = self._sparse_make_exprs(sp_coef_mat, svars, len(srhs))
        cts = [lfactory._new_binary_constraint(exprs[r], sense=op, rhs=srhs[r]) for r in range_cts]
        return cts

    @classmethod
    def generate_rows(cls, coef_mat):
        if is_pandas_dataframe(coef_mat):
            row_gen = cls.generate_df_rows(coef_mat)
        elif is_numpy_matrix(coef_mat):
            row_gen = cls.generate_np_matrix_rows(coef_mat)
        else:
            row_gen = iter(coef_mat)
        return row_gen

    def _matrix_constraints(self, coef_mat, svars, srhs, op):
        row_gen = self.generate_rows(coef_mat)
        lfactory = self._linear_factory

        return [lfactory._new_binary_constraint(lhs=self._scal_prod(svars, row), sense=op, rhs=rhs)
                for row, rhs in izip(row_gen, srhs)]

    def _matrix_ranges(self, coef_mat, svars, lbs, ubs):
        row_gen = self.generate_rows(coef_mat)
        lfactory = self._linear_factory

        return [lfactory.new_range_constraint(expr=self._scal_prod(svars, row), lb=lb, ub=ub)
                for row, lb, ub in izip(row_gen, lbs, ubs)]

    def _sparse_matrix_ranges(self, sp_coef_mat, svars, lbs, ubs):
        assert len(lbs) == len(ubs)
        range_ranges = range(len(lbs))
        lfactory = self._linear_factory
        exprs = self._sparse_make_exprs(sp_coef_mat, svars, nb_exprs=len(lbs))
        rgs = [lfactory.new_range_constraint(lbs[r], exprs[r], ubs[r], check_feasible=False) for r in range_ranges]
        return rgs
