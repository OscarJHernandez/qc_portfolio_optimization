# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2018
# --------------------------------------------------------------------------

try:
    from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
    from pyspark.ml import Transformer
    from pyspark.ml.param.shared import Param

    from pyspark.sql import SparkSession
    spark_version = SparkSession.builder.getOrCreate().version
except ImportError:
    spark_version = None
    def keyword_only(x):
        return x 
    Transformer = object
    Param = SparkSession = None

from docplex.mp.constants import ObjectiveSense
from docplex.mp.sktrans.modeler import make_modeler
from docplex.mp.utils import *
from docplex.mp.sparktrans.spark_utils import make_solution


def convert_to_list_or_value(value):
    if is_spark_dataframe(value):
        return value.rdd.flatMap(lambda x: x).collect()
    elif is_pandas_series(value):
        return value.tolist()
    elif is_numpy_ndarray(value):
        return value.tolist()
    return value


class CplexTransformerBase(Transformer):
    """ Root class for CPLEX transformers for PySpark
    """

    @keyword_only
    def __init__(self, rhsCol=None, minCol=None, maxCol=None, y=None, lbs=None, ubs=None, types=None, sense="min", modeler="cplex", solveParams=None):
        super(CplexTransformerBase, self).__init__()
        self.rhsCol = Param(self, "rhsCol", "")
        self.minCol = Param(self, "minCol", "")
        self.maxCol = Param(self, "maxCol", "")
        self.y = Param(self, "y", "")
        self.lbs = Param(self, "lbs", "")
        self.ubs = Param(self, "ubs", "")
        self.types = Param(self, "types", "")
        self.sense = Param(self, "sense", "")
        self.modeler = Param(self, "modeler", "")
        self.solveParams = Param(self, "solveParams", "")
        self._setDefault(rhsCol=None)
        self._setDefault(minCol=None)
        self._setDefault(maxCol=None)
        self._setDefault(y=None)
        self._setDefault(lbs=None)
        self._setDefault(ubs=None)
        self._setDefault(types=None)
        self._setDefault(sense=sense)
        self._setDefault(modeler=modeler)
        self._setDefault(solveParams={})

        if spark_version < '2.2':
            kwargs = self.__init__._input_kwargs
        else:
            kwargs = self._input_kwargs
        self.setParams(**kwargs)

        self.docplex_modeler = make_modeler(self.getModeler())

    @keyword_only
    def setParams(self, rhsCol=None, minCol=None, maxCol=None, y=None, lbs=None, ubs=None, types=None, solveParams=None, sense=None, modeler=None):
        if spark_version < '2.2':
            kwargs = self.__init__._input_kwargs
        else:
            kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setRhsCol(self, value):
        self._paramMap[self.rhsCol] = value
        return self

    def getRhsCol(self):
        return self.getOrDefault(self.rhsCol)

    def setMinCol(self, value):
        self._paramMap[self.minCol] = value
        return self

    def getMinCol(self):
        return self.getOrDefault(self.minCol)

    def setMaxCol(self, value):
        self._paramMap[self.maxCol] = value
        return self

    def getMaxCol(self):
        return self.getOrDefault(self.maxCol)

    def setY(self, value):
        self._paramMap[self.y] = value
        return self

    def getY(self):
        return self.getOrDefault(self.y)

    def setLbs(self, value):
        self._paramMap[self.lbs] = value
        return self

    def getLbs(self):
        return self.getOrDefault(self.lbs)

    def setUbs(self, value):
        self._paramMap[self.ubs] = value
        return self

    def getUbs(self):
        return self.getOrDefault(self.ubs)

    def setTypes(self, value):
        self._paramMap[self.types] = value
        return self

    def getTypes(self):
        return self.getOrDefault(self.types)

    def setSense(self, value):
        self._paramMap[self.sense] = value
        return self

    def getSense(self):
        return self.getOrDefault(self.sense)

    def setModeler(self, value):
        self._paramMap[self.modeler] = value
        return self

    def getModeler(self):
        return self.getOrDefault(self.modeler)

    def setSolveParams(self, value):
        self._paramMap[self.solveParams] = value
        return self

    def getSolveParams(self):
        return self.getOrDefault(self.solveParams)

    def _transform(self, dataset):
        """ Main method to solve Linear Programming problems.
        Transforms the input dataset.

        :param dataset: the matrix describing the  constraints of the problem, which is an instance
                of :py:class:`pyspark.sql.DataFrame`
        :returns: transformed dataset
        """
        objSense = ObjectiveSense.parse(self.getSense())

        var_lbs = convert_to_list_or_value(self.getLbs())
        var_ubs = convert_to_list_or_value(self.getUbs())
        var_types = self.getTypes()

        min_max_cols = [self.getMinCol(), self.getMaxCol()] if self.getMinCol() else [self.getRhsCol()]
        coef_colnames = [item for item in dataset.columns if item not in min_max_cols]
        nb_vars = len(coef_colnames)

        # Get min and max values as lists
        minVals, maxVals = self._get_min_max_lists(dataset)

        # Get coefficients for objective evaluation
        costs = None if self.getY() is None else convert_to_list_or_value(self.getY())
        assert (costs is None) or (len(costs) == nb_vars)

        # Build matrix of coefficients
        cts_mat = dataset.rdd.map(lambda x: [x[col] for col in coef_colnames]).collect()

        # Call build matrix and solve
        params = self.getSolveParams()  # User-defined engine parameters
        if minVals is None:
            result = self.docplex_modeler.build_matrix_linear_model_and_solve(nb_vars, var_lbs, var_ubs, var_types,
                                                                              coef_colnames,
                                                                              cts_mat, maxVals,
                                                                              objsense=objSense, costs=costs,
                                                                              cast_to_float=True,
                                                                              solution_maker=make_solution,
                                                                              **params)
        else:
            result = self.docplex_modeler.build_matrix_range_model_and_solve(nb_vars, var_lbs, var_ubs, var_types, coef_colnames,
                                                                             cts_mat=cts_mat,
                                                                             range_mins=minVals, range_maxs=maxVals,
                                                                             objsense=objSense, costs=costs,
                                                                             cast_to_float=True,
                                                                             solution_maker=make_solution,
                                                                             **params)
        # 'result' is a list: create a Spark DataFrame from it
        sparkCtx = dataset.sql_ctx
        return sparkCtx.createDataFrame(zip(coef_colnames, result), schema=['name', 'value'])


class CplexTransformer(CplexTransformerBase):
    """ A PySpark transformer class to solve linear problems.

    This transformer class solves LP problems of
    type::

            Ax <= B

    """

    @keyword_only
    def __init__(self, rhsCol=None, y=None, lbs=None, ubs=None, types=None, sense="min", modeler="cplex", solveParams=None):
        """
        Creates an instance of LPTransformer to solve linear problems.

        :param rhsCol: the name of the column in the input Spark DataFrame containing the upper bounds for the constraints.
        :param y: an optional sequence of scalars describing the cost vector
        :param lbs: an optional sequence of scalars describing the lower bounds for decision variables
        :param ubs: an optional sequence of scalars describing the upper bounds for decision variables
        :param types: a string for variable types within [BICSN]*
        :param sense: defines the objective sense. Accepts 'min" or "max" (not case-sensitive),
            or an instance of docplex.mp.ObjectiveSense
        :param solveParams: optional keyword arguments to pass additional parameters for Cplex engine.

        Note:
            The Spark dataframe representing the matrix is supposed to have shape (M,N+1) where M is the number of rows
            and N the number of variables. The column named by the 'rhsCol' parameter contains the right hand sides of the problem
            (the B in Ax <= B)
            The optional vector y contains the N cost coefficients for each column variables.

        Example:
            Passing the Spark DataFrame = ([[1,2,3], [4,5,6]], ['x', 'y', 'max']), rhsCol = 'max', y= [11,12] means
            solving the linear problem:

                minimize 11x + 12y
                s.t.
                        1x + 2y <= 3
                        4x + 5y <= 6
        """
        if spark_version < '2.2':
            super(CplexTransformer, self).__init__(**self.__init__._input_kwargs)
        else:
            super(CplexTransformer, self).__init__(**self._input_kwargs)

    def _get_min_max_lists(self, dataset):
        assert self.getRhsCol() and self.getRhsCol() in dataset.columns, '"rhsCol" parameter is not specified'

        maxVals = dataset.select(self.getRhsCol()).rdd.flatMap(lambda x: x).collect()
        return None, maxVals


class CplexRangeTransformer(CplexTransformerBase):
    """ A PySpark transformer class to solve range-based linear problems.

    This transformer class solves LP problems of
    type::

            B <= Ax <= C

    """

    @keyword_only
    def __init__(self, minCol=None, maxCol=None, y=None, lbs=None, ubs=None, types=None, sense="min", modeler="cplex", solveParams=None):
        """
        Creates an instance of LPRangeTransformer to solve range-based linear problems.

        :param minCol: the name of the column in the input Spark DataFrame containing the lower bounds for the constraints.
        :param maxCol: the name of the column in the input Spark DataFrame containing the upper bounds for the constraints.
        :param y: an optional sequence of scalars describing the cost vector
        :param lbs: an optional sequence of scalars describing the lower bounds for decision variables
        :param ubs: an optional sequence of scalars describing the upper bounds for decision variables
        :param types: a string for variable types within [BICSN]*
        :param sense: defines the objective sense. Accepts 'min" or "max" (not case-sensitive),
            or an instance of docplex.mp.ObjectiveSense
        :param solveParams: optional keyword arguments to pass additional parameters for Cplex engine.

        Note:
            The matrix X is supposed to have shape (M,N+2) where M is the number of rows
            and N the number of variables.
            The column named by the 'minCol' and 'maxCol' parameters contain the minimum (resp.maximum) values for the
            row ranges, that m and M in:
                    m <= Ax <= M
            The optional vector y contains the N cost coefficients for each column variables.

        Example:
            Passing the Spark DataFrame = ([[1,2,3,30], [4,5,6,60]], ['x', 'y', 'min', 'max']), minCol = 'min',
            maxCol = 'max', y= [11,12] means solving the linear problem:

                minimize 11x + 12y
                s.t.
                        3 <= 1x + 2y <= 30
                        6 <= 4x + 5y <= 60
        """
        if spark_version < '2.2':
            super(CplexRangeTransformer, self).__init__(**self.__init__._input_kwargs)
        else:
            super(CplexRangeTransformer, self).__init__(**self._input_kwargs)

    def _get_min_max_lists(self, dataset):
        assert self.getMinCol() and self.getMinCol() in dataset.columns, '"minCol" parameter is not specified'
        assert self.getMaxCol() and self.getMaxCol() in dataset.columns, '"maxCol" parameter is not specified'

        minVals = dataset.select(self.getMinCol()).rdd.flatMap(lambda x: x).collect()
        maxVals = dataset.select(self.getMaxCol()).rdd.flatMap(lambda x: x).collect()
        return minVals, maxVals
