# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017
# --------------------------------------------------------------------------

try:
    from sklearn.base import TransformerMixin, BaseEstimator
except ImportError:
    # if sklearn is not available, just use some mock classes
    class TransformerMixin(object):
        pass
    class BaseEstimator(object):
        pass
    
from docplex.mp.constants import ObjectiveSense
from docplex.mp.sktrans.modeler import make_modeler
from docplex.mp.utils import *

try:
    import numpy as np
except ImportError:
    np = None

class CplexTransformerBase(BaseEstimator, TransformerMixin):
    """ Root class for CPLEX transformers
    """

    def __init__(self, sense="min", modeler="cplex"):
        self.objsense = ObjectiveSense.parse(sense)  # fail if error
        self.modeler = make_modeler(modeler)

    def fit(self, *_):
        return self

    def transform(self, X, y=None, **params):
        """ Main method to solve Linear Programming problemss.

        :param X: the matrix describing the  constraints of the problem.
            Accepts numpy matrices, pandas dataframes, or sciPy sparse matrices
        :param y: an optional sequence of scalars descrining the cost vector
        :param params: optional keyword arguments to pass additional parameters.

        :return: a pandas dataframe with two columns: name and value containing the values
            of the columns.
        """
        # look for upper, lower bound columns in keyword args
        var_lbs = params.get("lbs", None)
        var_ubs = params.get("ubs", None)
        var_types = params.get("types", None)
        if is_pandas_dataframe(X):
            return self._transform_from_pandas(X, y, var_lbs, var_ubs, var_types, **params)
        elif is_numpy_matrix(X):
            return self._transform_from_numpy(X, y, var_lbs, var_ubs, var_types, **params)
        elif is_scipy_sparse(X):
            return self._transform_from_sparse(X, y, var_lbs, var_ubs, var_types, **params)
        elif isinstance(X, list):
            return self._transform_from_sequence(X, y, var_lbs, var_ubs, var_types, **params)
        else:
            raise ValueError(
                'transformer expects pandas dataframe, numpy matrix or python list, {0} was passed'.format(X))

    def _transform_from_sequence(self, X, y, var_lbs, var_ubs, var_types, **params):
        raise NotImplemented

    def _transform_from_pandas(self, X, y, var_lbs, var_ubs, var_types, **params):
        raise NotImplemented

    def _transform_from_numpy(self, X, y, var_lbs, var_ubs, var_types, **params):
        raise NotImplemented

    def _transform_from_sparse(self, X, y, var_lbs, var_ubs, var_types, **params):
        raise NotImplemented

    def build_matrix_linear_model_and_solve(self, var_count, var_lbs, var_ubs, var_types, var_names,
                                            cts_mat, rhs,
                                            objsense, costs, cast_to_float,
                                            **params):
        return self.modeler.build_matrix_linear_model_and_solve(var_count, var_lbs, var_ubs,
                                                                var_types, var_names,
                                                                cts_mat, rhs,
                                                                objsense, costs,
                                                                cast_to_float, **params)

    def build_matrix_range_model_and_solve(self, var_count, var_lbs, var_ubs,
                                           var_types, var_names,
                                           cts_mat, range_mins, range_maxs,
                                           objsense, costs, cast_to_float,
                                           **params):
        return self.modeler.build_matrix_range_model_and_solve(var_count, var_lbs, var_ubs,
                                                               var_types, var_names,
                                                               cts_mat, range_mins, range_maxs,
                                                               objsense, costs,
                                                               cast_to_float,
                                                               **params)

    def build_sparse_linear_model_and_solve(self, nb_vars, var_lbs, var_ubs, var_types, var_names,
                                            nb_rows, cts_sparse_coefs,
                                            objsense, costs,
                                            **params):
        return self.modeler.build_sparse_linear_model_and_solve(nb_vars, var_lbs, var_ubs,
                                                                var_types, var_names,
                                                                nb_rows, cts_sparse_coefs,
                                                                objsense, costs,
                                                                **params)


class CplexTransformer(CplexTransformerBase):
    """ A Scikit-learn transformer class to solve linear problems.

    This transformer class solves LP problems of
    type::

            Ax <= B

    """

    def __init__(self, sense="min", modeler="cplex"):
        """
        Creates an instance of LPTransformer to solve linear problems.

        :param sense: defines the objective sense. Accepts 'min" or "max" (not case-sensitive),
            or an instance of docplex.mp.ObjectiveSense

        Note:
            The matrix X is supposed to have shape (M,N+1) where M is the number of rows
            and N the number of variables. The last column contains the right hand sides of the problem
            (the B in Ax <= B)
            The optional vector Y contains the N cost coefficients for each column variables.

        Example:
            Passing X = [[1,2,3], [4,5,6]], Y= [11,12,13] means solving the linear problem:

                minimize 11x + 12y + 13z
                s.t.
                        1x + 2y <= 3
                        4x + 5y <= 6
        """
        super(CplexTransformer, self).__init__(sense, modeler)

    def _transform_from_pandas(self, X, y, var_lbs, var_ubs, var_types, **params):
        assert is_pandas_dataframe(X)

        X_new = X.copy()
        # save min, max per nutrients in lists, drop them
        rhs = X["rhs"].tolist()
        X_new.drop(labels=["rhs"], inplace=True, axis=1)
        _, x_cols = X.shape
        nb_vars = x_cols - 1
        return self.build_matrix_linear_model_and_solve(nb_vars, var_lbs, var_ubs, var_types,
                                                        var_names=X_new.columns,
                                                        cts_mat=X_new, rhs=rhs,
                                                        objsense=self.objsense, costs=y,
                                                        **params)

    def _transform_from_sequence(self, X, y, var_lbs, var_ubs, var_types, **params):
        # matrix is a list of lists nr x [nc+1]
        # last two columns are lbs, ubs in that order
        assert X
        xc = len(X[0])
        for r in X:
            assert xc == len(r)

        colnames = params.pop("colnames", None)

        nb_vars = xc - 1
        X_cts = [r[:-1] for r in X]
        rhs = [r[-1] for r in X]
        # no cast to float except if requested
        cast_to_float = params.pop("cast_to_float", False)
        return self.build_matrix_linear_model_and_solve(nb_vars, var_lbs, var_ubs, var_types,
                                                        colnames,
                                                        X_cts, rhs,
                                                        objsense=self.objsense, costs=y,
                                                        cast_to_float=cast_to_float,
                                                        **params)

    def _transform_from_numpy(self, X, y, var_lbs, var_ubs, var_types, **params):
        # matrix is nrows x (ncols + 2)
        # last two columns are lbs, ubs in that order
        assert is_numpy_matrix(X)

        colnames = params.pop("colnames", None)
        mshape = X.shape
        nr, nc = mshape

        assert nc >= 2
        nb_vars = nc - 1
        X_cts = X[:, :-1]
        rhs = X[:, -1].A1
        # to cast or not to cast?
        cast_to_float = (X.dtype == np.int64) or params.pop("cast-to_float", False)
        return self.build_matrix_linear_model_and_solve(nb_vars, var_lbs, var_ubs, var_types,colnames,
                                                        X_cts, rhs,
                                                        objsense=self.objsense, costs=y,
                                                        cast_to_float=cast_to_float, **params)

    def _transform_from_sparse(self, X, y, var_lbs, var_ubs, var_types, **params):
        assert is_scipy_sparse(X)

        colnames = params.pop("colnames", None)
        mshape = X.shape
        nr, nc = mshape
        nb_vars = nc - 1
        #  convert to coo before iterate()
        x_coo = X.tocoo()
        cts_sparse_coefs = izip(x_coo.data, x_coo.row, x_coo.col)
        return self.build_sparse_linear_model_and_solve(nb_vars, var_lbs, var_ubs, var_types,colnames,
                                                        nr, cts_sparse_coefs,
                                                        objsense=self.objsense, costs=y, **params)


class CplexRangeTransformer(CplexTransformerBase):
    def __init__(self, sense="min", modeler="cplex"):
        """
        Creates an instance of LPRangeTransformer to solve range-based linear problems.

        :param sense: defines the objective sense. Accepts 'min" or "max" (not case-sensitive),
            or an instance of docplex.mp.ObjectiveSense

        Note:
            The matrix X is supposed to have shape (M,N+2) where M is the number of rows
            and N the number of variables.
            The last two columns are assumed to contain the minimum (resp.maximum) values for the
            row ranges, that m and M in:
                    m <= Ax <= M

            The optional vector Y contains the N cost coefficients for each column variables.

        Example:
            Passing X = [[1,2,3,30], [4,5,6,60]], Y= [11,12,13] means solving the linear problem:

                minimize 11x + 12y + 13z
                s.t.
                        3 <= 1x + 2y <= 30
                        6 <= 4x + 5y <= 60
        """
        super(CplexRangeTransformer, self).__init__(sense, modeler)

    def _transform_from_sequence(self, X, y, var_lbs, var_ubs, **params):
        # matrix is a list of lists nr x [nc+1]
        # last two columns are lbs, ubs in that order
        assert X
        xc = len(X[0])
        assert xc >= 3  # one var plus min max
        for r in X:
            assert xc == len(r)

        colnames = params.get("colnames", None)

        nb_vars = xc - 2
        X_cts = [r[:-2] for r in X]
        row_maxs = [r[-1] for r in X]
        row_mins = [r[-2] for r in X]
        return self.build_matrix_range_model_and_solve(nb_vars, var_lbs, var_ubs, colnames,
                                                       cts_mat=X_cts,
                                                       range_mins=row_mins, range_maxs=row_maxs,
                                                       objsense=self.objsense, costs=y,
                                                       **params)

    def _transform_from_pandas(self, X, y, var_lbs, var_ubs, var_types, **params):
        assert is_pandas_dataframe(X)

        x_rows, x_cols = X.shape
        X_new = X.copy()
        # extract columns with name 'min' and 'max' as series then drop
        row_mins = X["min"].tolist()
        row_maxs = X["max"].tolist()
        X_new.drop(labels=["min", "max"], inplace=True, axis=1)
        nb_vars = x_cols - 2
        varnames = X_new.columns.values.tolist()
        return self.build_matrix_range_model_and_solve(nb_vars, var_lbs, var_ubs, var_types,
                                                       varnames,
                                                       cts_mat=X_new,
                                                       range_mins=row_mins, range_maxs=row_maxs,
                                                       objsense=self.objsense, costs=y,
                                                       cast_to_float=True,
                                                       **params)

    def _transform_from_numpy(self, X, y, var_lbs, var_ubs, var_types, **params):
        # matrix is nrows x (ncols + 2)
        # last two columns are lbs, ubs in that order
        assert is_numpy_matrix(X)

        colnames = params.pop("colnames", None)
        mshape = X.shape
        xr, xc = mshape
        assert xc >= 3
        nb_vars = xc - 2
        X_cts = X[:, :-2]
        row_mins = X[:, -2]
        row_maxs = X[:, -1]
        cast_to_float = (X.dtype == np.int64) or params.pop("cast_to_float", False)
        return self.build_matrix_range_model_and_solve(nb_vars, var_lbs, var_ubs,
                                                       var_types, colnames,
                                                       cts_mat=X_cts,
                                                       range_mins=row_mins, range_maxs=row_maxs,
                                                       objsense=self.objsense, costs=y,
                                                       cast_to_float=cast_to_float,
                                                       **params)
