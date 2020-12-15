# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2016
# --------------------------------------------------------------------------

from six import iteritems

from docplex.mp.model import Model
from docplex.mp.aggregator import ModelAggregator
from docplex.mp.quad import VarPair
from docplex.mp.utils import is_number, is_iterable, generate_constant, \
    is_pandas_dataframe, is_pandas_series, is_numpy_matrix, is_scipy_sparse, is_ordered_sequence
from docplex.mp.constants import ComparisonType

from docplex.mp.compat23 import izip
from docplex.mp.error_handler import docplex_fatal
from docplex.mp.xcounter import update_dict_from_item_value

class AdvAggregator(ModelAggregator):
    def __init__(self, linear_factory, quad_factory):
        ModelAggregator.__init__(self, linear_factory, quad_factory)

    def _scal_prod_vars_all_different(self, terms, coefs):
        checker = self._checker
        if not is_iterable(coefs, accept_string=False):
            checker.typecheck_num(coefs)
            return coefs * self._sum_vars_all_different(terms)
        else:
            # coefs is iterable
            lcc_type = self.counter_type
            lcc = lcc_type()
            lcc_setitem = lcc_type.__setitem__
            number_validation_fn = checker.get_number_validation_fn()
            if number_validation_fn:
                for dvar, coef in izip(terms, coefs):
                    safe_coef = number_validation_fn(coef)
                    if safe_coef:
                        lcc_setitem(lcc, dvar, safe_coef)
            else:
                for dvar, coef in izip(terms, coefs):
                    if coef:  # zero test is much cheaper than a setitem
                        lcc_setitem(lcc, dvar, coef)

            return self._to_expr(qcc=None, lcc=lcc)

    def scal_prod_triple(self, left_terms, right_terms, coefs):
        used_coefs = None
        checker = self._model._checker

        if is_iterable(coefs, accept_string=False):
            used_coefs = coefs
        elif is_number(coefs):
            if coefs:
                used_coefs = generate_constant(coefs, count_max=None)
            else:
                return self.new_zero_expr()
        else:
            self._model.fatal("scal_prod_triple expects iterable or number as coefficients, got: {0!r}", coefs)

        if is_iterable(left_terms):
            used_left = checker.typecheck_var_seq(left_terms)
        else:
            checker.typecheck_var(left_terms)
            used_left = generate_constant(left_terms, count_max=None)

        if is_iterable(right_terms):
            used_right = checker.typecheck_var_seq(right_terms)
        else:
            checker.typecheck_var(right_terms)
            used_right = generate_constant(right_terms, count_max=None)

        if used_coefs is not coefs and used_left is not left_terms and used_right is not right_terms:
            # LOOK
            return left_terms * right_terms * coefs

        return self._scal_prod_triple(coefs=used_coefs, left_terms=used_left, right_terms=used_right)

    def _scal_prod_triple(self, coefs, left_terms, right_terms):
        # INTERNAL
        accumulated_ct = 0
        qcc = self.counter_type()
        lcc = self.counter_type()
        number_validation_fn = self._checker.get_number_validation_fn()
        for coef, lterm, rterm in izip(coefs, left_terms, right_terms):
            if coef:
                safe_coef = number_validation_fn(coef) if number_validation_fn else coef
                lcst = lterm.get_constant()
                rcst = rterm.get_constant()
                accumulated_ct += safe_coef * lcst * rcst
                for lv, lk in lterm.iter_terms():
                    for rv, rk in rterm.iter_terms():
                        coef3 = safe_coef * lk * rk
                        update_dict_from_item_value(qcc, VarPair(lv, rv), coef3)
                if rcst:
                    for lv, lk in lterm.iter_terms():
                        update_dict_from_item_value(lcc, lv, safe_coef * lk * rcst)
                if lcst:
                    for rv, rk in rterm.iter_terms():
                        update_dict_from_item_value(lcc, rv, safe_coef * rk * lcst)

        return self._to_expr(qcc, lcc, constant=accumulated_ct)

    def _scal_prod_triple_vars(self, coefs, left_terms, right_terms):
        # INTERNAL
        # assuming all arguments are iterable.
        dcc = self.counter_type
        qcc = dcc()
        number_validation_fn = self._checker.get_number_validation_fn()
        if number_validation_fn:
            for coef, lterm, rterm in izip(coefs, left_terms, right_terms):
                safe_coef = number_validation_fn(coef) if number_validation_fn else coef
                update_dict_from_item_value(qcc, VarPair(lterm, rterm), safe_coef)
        else:
            for coef, lterm, rterm in izip(coefs, left_terms, right_terms):
                update_dict_from_item_value(qcc, VarPair(lterm, rterm), coef)
        return self._to_expr(qcc=qcc)

    def _sumsq_vars_all_different(self, dvars):
        dcc = self._quad_factory.term_dict_type
        qcc = dcc()
        qcc_setitem = dcc.__setitem__
        for t in dvars:
            qcc_setitem(qcc, VarPair(t), 1)
        return self._to_expr(qcc=qcc)

    def _sumsq_vars(self, dvars):
        qcc = self._quad_factory.term_dict_type()
        for v in dvars:
            update_dict_from_item_value(qcc, VarPair(v), 1)
        return self._to_expr(qcc=qcc)

    def quad_matrix_sum(self, matrix, lvars, symmetric=False):
        # assume matrix is a NxN matrix
        # vars is a N-vector of variables
        dcc = self._quad_factory.term_dict_type
        qterms = dcc()

        gen_rows = self.generate_rows(matrix)

        for i, mrow in enumerate(gen_rows):
            vi = lvars[i]
            for j, k in enumerate(mrow):
                if k:
                    vj = lvars[j]
                    if i == j:
                        qterms[VarPair(vi)] = k
                    elif symmetric:
                        if i < j:
                            update_dict_from_item_value(qterms, VarPair(vi, vj), 2 * k)
                        elif i > j:
                            continue
                    else:
                        update_dict_from_item_value(qterms, VarPair(vi, vj), k)

        return self._to_expr(qcc=qterms)

    # noinspection PyUnusedLocal
    def _sparse_quad_matrix_sum(self, sp_coef_mat, lvars, symmetric=False):
        # assume matrix is a NxN matrix
        # vars is a N-vector of variables
        dcc = self._quad_factory.term_dict_type
        qterms = dcc()

        for e in range(sp_coef_mat.nnz):
            k = sp_coef_mat.data[e]
            if k:
                row = sp_coef_mat.row[e]
                col = sp_coef_mat.col[e]
                vi = lvars[row]
                vj = lvars[col]
                update_dict_from_item_value(qterms, VarPair(vi, vj), k)

        return self._to_expr(qcc=qterms)

    def vector_compare(self, left_exprs, right_exprs, sense):
        lfactory = self._linear_factory
        assert len(left_exprs) == len(right_exprs)
        cts = [lfactory._new_binary_constraint(left, sense, right) for left, right in izip(left_exprs, right_exprs)]
        return cts


# noinspection PyProtectedMember
class AdvModel(Model):
    """
    This class is a specialized version of the :class:`docplex.mp.model.Model` class with useful non-standard modeling
    functions.
    """
    _fast_settings = {'keep_ordering': False, 'checker': 'off', 'keep_all_exprs': False}

    def __init__(self, name=None, context=None, **kwargs):
        for k, v in iteritems(self._fast_settings):
            if k not in kwargs:
                # force fast settings if not present
                kwargs[k] = v

        Model.__init__(self, name=name, context=context, **kwargs)
        self._aggregator = AdvAggregator(self._lfactory, self._qfactory)

    def _prepare_constraint(self, ct, ctname, check_for_trivial_ct, arg_checker=None):
        # INTERNAL
        if ct is False:
            # happens with sum([]) and constant e.g. sum([]) == 2
            msg = "Adding a trivially infeasible constraint"
            if ctname:
                msg += ' with name: {0}'.format(ctname)
            # analogous to 0 == 1, model is sure to fail
            self.fatal(msg)

        if ct is True:
            return False


        # --- name management ---
        if ctname:
            ct_name_map = self._cts_by_name
            if ct_name_map is not None:
                ct_name_map[ctname] = ct
            ct.name = ctname
        # ---
        return True

    def sumsq_vars(self, terms):
        return self._aggregator._sumsq_vars(terms)

    def sumsq_vars_all_different(self, terms):
        """
        Creates a quadratic expression by summing squares over a sequence.

        The variable sequence is a list or an iterator of variables.

        This method is faster than the standard summation of squares method due to the fact that it takes only variables and does not take expressions as arguments.

        :param terms: A list or an iterator on variables only, with no duplicates.

        :return: A quadratic expression or 0.

        Note:
           If the list or iterator is empty, this method returns zero.

        Note:
            To improve performance, the check for duplicates can be turned off by setting
            `checker='none'` in the `kwargs` of the :class:`docplex.mp.model.Model` object. As this argument
            turns off checking everywhere, it should be used with extreme caution.
        """
        var_seq = self._checker.typecheck_var_seq_all_different(terms)
        return self._aggregator._sumsq_vars_all_different(var_seq)

    def scal_prod_vars_all_different(self, terms, coefs):
        """
        Creates a linear expression equal to the scalar product of a list of decision variables and a sequence of coefficients.

        The variable sequence is a list or an iterator of variables.
        The coefficients can be either a list of numbers, an iterator over numbers, or a number.

        This method is faster than the standard generic scalar product method due to the fact that it takes only variables and does not take expressions as arguments.

        :param terms: A list or an iterator on variables only, with no duplicates.
        :param coefs: A list or an iterator on numbers, or a number.

        :return: A linear expression or 0.

        Note:
           If either list or iterator is empty, this method returns zero.

        Note:
            To improve performance, the check for duplicates can be turned off by setting
            `checker='none'` in the `kwargs` of the :class:`docplex.mp.model.Model` object. As this argument
            turns off checking everywhere, it should be used with extreme caution.
        """
        self._checker.check_ordered_sequence(arg=terms,
                                             caller='Model.scal_prod() requires a list of expressions/variables')
        var_seq = self._checker.typecheck_var_seq_all_different(terms)
        return self._aggregator._scal_prod_vars_all_different(var_seq, coefs)


    def quad_matrix_sum(self, matrix, dvars, symmetric=False):
        """
        Creates a quadratic expression equal to the quadratic form of a list of decision variables and
        a matrix of coefficients.

        This method sums all quadratic terms built by multiplying the [i,j]th coefficient in the matrix
        by the product of the i_th and j_th variables in `dvars`; in mathematical terms, the expression formed
        by x'Qx.

        :param matrix: A accepts either a list of lists of numbers, a numpy array, a pandas dataframe, or
            a scipy sparse matrix in COO format. T
            The resulting matrix must be square with size (N,N) where N is the number of variables.
        :param dvars: A list or an iterator on variables.
        :param symmetric: A boolean indicating whether the matrix is symmetric or not (default is False).
            No check is done.

        :return: An instance of :class:`docplex.mp.quad.QuadExpr` or 0.

        Note:
           The matrix must be square but not necessarily symmetric. The number of rows of the matrix must be equal
           to the size of the variable sequence.

           The symmetric flag only explores half of the matrix and doubles non-diagonal factors. No actual check is done.
           This flag has no effect on scipy sparse matrix.

        Example:
            `Model.quad_matrix_sum([[[1, 2], [3, 4]], [x, y])` returns the expression `x^2+4y^2+5x*yt`.
        """
        lvars = AdvModel._to_list(dvars, caller='Model.quad_matrix_sum')
        self._checker.typecheck_var_seq(lvars)
        if is_scipy_sparse(matrix):
            return self._aggregator._sparse_quad_matrix_sum(matrix, lvars, symmetric=symmetric)
        else:
            return self._aggregator.quad_matrix_sum(matrix, lvars, symmetric=symmetric)

    def scal_prod_triple(self, left_terms, right_terms, coefs):
        """
        Creates a quadratic expression from two lists of linear expressions and a sequence of coefficients.

        This method sums all quadratic terms built by multiplying the i_th coefficient by the product of the i_th
        expression in `left_terms` and the i_th expression in `right_terms`

        This method accepts different types of input for its arguments. The expression sequences can be either lists
        or iterators of objects that can be converted to linear expressions, that is, variables or linear expressions
        (but no quadratic expressions).
        The most usual case is variables.
        The coefficients can be either a list of numbers, an iterator over numbers, or a number.

        Example:
            `Model.scal_prod_triple([x, y], [z, t], [2, 3])` returns the expression `2xz + 3yt`.

        :param left_terms: A list or an iterator on variables or expressions.
        :param right_terms: A list or an iterator on variables or expressions.
        :param coefs: A list or an iterator on numbers or a number.

        :returns: An instance of :class:`docplex.mp.quad.QuadExpr` or 0.

        Note:
           If either list or iterator is empty, this method returns zero.
        """
        return self._aggregator.scal_prod_triple(left_terms=left_terms, right_terms=right_terms, coefs=coefs)

    def scal_prod_triple_vars(self, left_terms, right_terms, coefs):
        """
        Creates a quadratic expression from two lists of variables and a sequence of coefficients.

        This method sums all quadratic terms built by multiplying the i_th coefficient by the product of the i_th
        expression in `left_terms` and the i_th expression in `right_terms`

        This method is faster than the standard generic scalar quadratic product method due to the fact that it takes only variables and does not take expressions as arguments.

        Example:
            `Model.scal_prod_vars_triple([x, y], [z, t], [2, 3])` returns the expression `2xz + 3yt`.

        :param left_terms: A list or an iterator on variables.
        :param right_terms: A list or an iterator on variables.
        :param coefs: A list or an iterator on numbers or a number.

        :returns: An instance of :class:`docplex.mp.quad.QuadExpr` or 0.

        Note:
           If either list or iterator is empty, this method returns zero.
        """
        used_coefs = None
        checker = self._checker
        nb_non_iterables = 0

        if is_iterable(coefs, accept_string=False):
            used_coefs = coefs
        elif is_number(coefs):
            if coefs:
                used_coefs = generate_constant(coefs, count_max=None)
                nb_non_iterables += 1
            else:
                return self._aggregator.new_zero_expr()
        else:
            self.fatal("scal_prod_triple expects iterable or number as coefficients, got: {0!r}", coefs)

        if is_iterable(left_terms):
            used_left = checker.typecheck_var_seq(left_terms)
        else:
            nb_non_iterables += 1
            checker.typecheck_var(left_terms)
            used_left = generate_constant(left_terms, count_max=None)

        if is_iterable(right_terms):
            used_right = checker.typecheck_var_seq(right_terms)
        else:
            nb_non_iterables += 1
            checker.typecheck_var(right_terms)
            used_right = generate_constant(right_terms, count_max=None)

        if nb_non_iterables >= 3:
            return left_terms * right_terms * coefs
        else:
            return self._aggregator._scal_prod_triple_vars(left_terms=used_left,
                                                           right_terms=used_right, coefs=used_coefs)

    @classmethod
    def _to_list(cls, s, caller):
        if is_pandas_series(s):
            return s.tolist()
        elif is_ordered_sequence(s):
            return s
        else:
            docplex_fatal('{0} requires ordered sequences: lists, numpy array or Series, got: {1}', caller, type(s))
            return list(s)

    def matrix_constraints(self, coef_mat, dvars, rhs, sense='le'):
        """
        Creates a list of linear constraints
        from a matrix of coefficients, a sequence of variables, and a sequence of numbers.

        This method returns the list of constraints built from

            A.X <op> B

        where A is the coefficient matrix (of size (M,N)), X is the variable sequence (size N),
        and B is the sequence of right-hand side values (of size M).

        <op> is the comparison operator that defines the sense of the constraint. By default, this generates
        a 'less-than-or-equal' constraint.

        Example:
            `Model.scal_prod_vars_triple([x, y], [z, t], [2, 3])` returns the expression `2xz + 3yt`.

        :param coef_mat: A matrix of coefficients with M rows and N columns. This argument accepts
            either a list of lists of numbers, a `numpy` array with size (M,N), or a `scipy` sparse matrix.
        :param dvars: An ordered sequence of decision variables: accepts a Python list, `numpy` array,
            or a `pandas` series. The size of the sequence must match the number of columns in the matrix.
        :param rhs: A sequence of numbers: accepts a Python list, a `numpy` array,
            or a `pandas` series. The size of the sequence must match the number of rows in the matrix.
        :param sense: A constraint sense, accepts either a
            value of type `ComparisonType` or a string (e.g 'le', 'eq', 'ge').

        :returns: A list of linear constraints.

        Example:

            If A is a matrix of coefficients with 2 rows and 3 columns::

                    A = [[1, 2, 3],
                         [4, 5, 6]],
                    X = [x, y, z] where x, y, and z are decision variables (size 3), and

                    B = [100, 200], a sequence of numbers (size 2),

            then::

                `mdl.matrix_constraint(A, X, B, 'GE')` returns a list of two constraints
                [(x + 2y+3z <= 100), (4x + 5y +6z <= 200)].

        Note:
            If the dimensions of the matrix and variables or of the matrix and number sequence do not match,
            an error is raised.

        """
        checker = self._checker
        if is_pandas_dataframe(coef_mat) or is_numpy_matrix(coef_mat) or is_scipy_sparse(coef_mat):
            nb_rows, nb_cols = coef_mat.shape
        else:
            # a sequence of sequences
            a_mat = list(coef_mat)
            nb_rows = len(a_mat)
            nb_cols = None
            try:
                shared_len = None
                for r in a_mat:
                    checker.check_ordered_sequence(r, 'matrix_constraints')
                    r_len = len(r)
                    if shared_len is None:
                        shared_len = r_len
                    elif r_len != shared_len:
                        self.fatal('All columns should have same length found  {0} != {1}'.format(shared_len, r_len))
                nb_cols = shared_len if shared_len is not None else 0
            except AttributeError:
                self.fatal('All columns should have a len()')

        s_dvars = self._to_list(dvars, caller='Model.matrix-constraints()')
        s_rhs = self._to_list(rhs, caller='Model.matrix-constraints()')
        # check

        checker.typecheck_var_seq(s_dvars)
        for k in s_rhs:
            checker.typecheck_num(k)

        op = ComparisonType.parse(sense)
        # ---
        # check dimensions and whether to transpose or not.
        # ---
        nb_rhs = len(s_rhs)
        nb_vars = len(s_dvars)
        if (nb_rows, nb_cols) != (nb_rhs, nb_vars):
            self.fatal(
                'Dimension error, matrix is ({0},{1}), expecting ({3}, {2})'.format(nb_rows, nb_cols, nb_vars, nb_rhs))

        if is_scipy_sparse(coef_mat):
            return self._aggregator._sparse_matrix_constraints(coef_mat, s_dvars, s_rhs, op)
        else:
            return self._aggregator._matrix_constraints(coef_mat, s_dvars, s_rhs, op)

    def matrix_ranges(self, coef_mat, dvars, lbs, ubs):
        """
        Creates a list of range constraints
        from a matrix of coefficients, a sequence of variables, and two sequence of numbers.

        This method returns the list of range constraints built from

            L <= Ax <= U

        where A is the coefficient matrix (of size (M,N)), X is the variable sequence (size N),
        L and B are sequence of numbers (resp. the lower and upper bounds of the ranges) both with size M.


        :param coef_mat: A matrix of coefficients with M rows and N columns. This argument accepts
            either a list of lists of numbers, a `numpy` array with size (M,N), or a `scipy` sparse matrix.
        :param dvars: An ordered sequence of decision variables: accepts a Python list, `numpy` array,
            or a `pandas` series. The size of the sequence must match the number of columns in the matrix.
        :param lbs: A sequence of numbers: accepts a Python list, a `numpy` array,
            or a `pandas` series. The size of the sequence must match the number of rows in the matrix.
        :param ubs: A sequence of numbers: accepts a Python list, a `numpy` array,
            or a `pandas` series. The size of the sequence must match the number of rows in the matrix.

        :returns: A list of range constraints.

        Example::

            If A is a matrix of coefficients with 2 rows and 3 columns:

                    A = [[1, 2, 3],
                         [4, 5, 6]],
                    X = [x, y, z] where x, y, and z are decision variables (size 3), and

                    L = [101. 102], a sequence of numbers (size 2),
                    U = [201, 202]

            then::

                `mdl.range_constraints(A, X, L, U)` returns a list of two ranges
                [(101 <= x + 2y+3z <= 102), (201 <= 4x + 5y +6z <= 202)].

        Note:
            If the dimensions of the matrix and variables or of the matrix and number sequence do not match,
            an error is raised.

        """
        checker = self._checker
        if is_pandas_dataframe(coef_mat) or is_numpy_matrix(coef_mat) or is_scipy_sparse(coef_mat):
            nb_rows, nb_cols = coef_mat.shape
        else:
            # a sequence of sequences
            a_mat = list(coef_mat)
            nb_rows = len(a_mat)
            nb_cols = None
            try:
                shared_len = None
                for r in a_mat:
                    checker.check_ordered_sequence(r, 'matrix_constraints')
                    r_len = len(r)
                    if shared_len is None:
                        shared_len = r_len
                    elif r_len != shared_len:
                        self.fatal('All columns should have same length found  {0} != {1}'.format(shared_len, r_len))
                nb_cols = shared_len if shared_len is not None else 0
            except AttributeError:
                self.fatal('All columns should have a len()')

        s_dvars = self._to_list(dvars, caller='Model.range_constraints()')
        s_lbs = self._to_list(lbs, caller='Model.range_constraints()')
        s_ubs = self._to_list(ubs, caller='Model.range_constraints()')
        # check

        checker.typecheck_var_seq(s_dvars)
        checker.typecheck_num_seq(s_lbs, caller="AdvModel.matrix_ranges.lbs")
        checker.typecheck_num_seq(s_ubs, caller="AdvModel.matrix_ranges.ubs")

        # ---
        # check dimensions and whether to transpose or not.
        # ---
        nb_vars = len(s_dvars)
        nb_lbs = len(s_lbs)
        nb_ubs = len(s_ubs)
        if nb_lbs != nb_rows:
            self.fatal('Incorrect size for range lower bounds, expecting: {1}, got: {0},'.format(nb_lbs, nb_rows))
        if nb_ubs != nb_rows:
            self.fatal('Incorrect size for range upper bounds, expecting: {1}, got: {0}'.format(nb_ubs, nb_rows))
        if nb_cols != nb_vars:
            self.fatal(
                'Incorrect number of variables, expecting: {1}, got: {0},  matrix is ({0},{1})'.format(nb_vars, nb_cols))

        if is_scipy_sparse(coef_mat):
            return self._aggregator._sparse_matrix_ranges(coef_mat, s_dvars, s_lbs, s_ubs)
        else:
            return self._aggregator._matrix_ranges(coef_mat, s_dvars, s_lbs, s_ubs)

    def vector_compare(self, lhss, rhss, sense):
        l_lhs = self._to_list(lhss, caller='Model.vector.compare')
        l_rhs = self._to_list(rhss, caller='Model.vector.compare')
        if len(l_lhs) != len(l_rhs):
            self.fatal('Model.vector_compare() got sequences with different length, left: {0}, right: {1}'.
                       format(len(l_lhs), len(l_rhs)))
        ctsense = ComparisonType.parse(sense)
        return self._aggregator.vector_compare(l_lhs, l_rhs, ctsense)

    def vector_compare_le(self, lhss, rhss):
        return self.vector_compare(lhss, rhss, 'le')

    def vector_compare_ge(self, lhss, rhss):
        return self.vector_compare(lhss, rhss, 'ge')

    def vector_compare_eq(self, lhss, rhss):
        return self.vector_compare(lhss, rhss, 'eq')
