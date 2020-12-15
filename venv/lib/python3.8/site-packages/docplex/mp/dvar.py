from six import PY3
from docplex.mp.constants import CplexScope

from docplex.mp.basic import IndexableObject, ModelingObjectBase, _BendersAnnotatedMixin
from docplex.mp.operand import LinearOperand
from docplex.mp.vartype import BinaryVarType, IntegerVarType, ContinuousVarType
from docplex.mp.utils import is_number, is_quad_expr

from docplex.mp.sttck import StaticTypeChecker


class Var(IndexableObject, LinearOperand, _BendersAnnotatedMixin):
    """Var()

    This class models decision variables.
    Decision variables are instantiated by :class:`docplex.mp.model.Model` methods such as :func:`docplex.mp.model.Model.var`.

    """

    __slots__ = ('_vartype', '_lb', '_ub')

    def __init__(self, model, vartype, name,
                 lb=None, ub=None,
                 _safe_lb=False, _safe_ub=False):
        IndexableObject.__init__(self, model, name)
        self._vartype = vartype

        if _safe_lb:
            self._lb = lb
        else:
            self._lb = vartype._compute_lb(lb, model)
        if _safe_ub:
            self._ub = ub
        else:
            self._ub = vartype._compute_ub(ub, model)

    @property
    def cplex_scope(self):
        return CplexScope.VAR_SCOPE

    # noinspection PyUnusedLocal
    def copy(self, new_model, var_mapping):
        return var_mapping[self]

    relaxed_copy = copy

    # linear operand api

    def as_variable(self):
        return self

    def iter_terms(self):
        yield self, 1

    def clone(self):
        return self

    def negate(self):
        return self.lfactory._new_monomial_expr(self, -1, safe=True)

    iter_sorted_terms = iter_terms

    def number_of_terms(self):
        return 1

    def unchecked_get_coef(self, dvar):
        return 1 if dvar is self else 0

    def contains_var(self, dvar):
        return self is dvar

    def accepts_value(self, candidate_value, tolerance=1e-6):
        # INTERNAL
        return self.vartype.accept_domain_value(candidate_value, lb=self._lb, ub=self._ub, tolerance=tolerance)

    def check_name(self, new_name):
        self.check_lp_name(qualifier='variable', new_name=new_name, accept_empty=False, accept_none=False)

    def __hash__(self):
        return self._index

    def set_name(self, new_name):
        # INTERNAL
        self.check_name(new_name)
        self.model.set_var_name(self, new_name)

    name = property(ModelingObjectBase.get_name, set_name)

    @property
    def lb(self):
        """ This property is used to get or set the lower bound of the variable.

        Possible values for the lower bound depend on the variable type. Binary variables
        accept only 0 or 1 as bounds. An integer variable will convert the lower bound value to the
        ceiling integer value of the argument.
        """
        return self._lb

    @lb.setter
    def lb(self, new_lb):
        self.set_lb(new_lb)

    def set_lb(self, lb):
        if lb != self._lb:
            self._model.set_var_lb(self, lb)
            return self._lb

    def _internal_set_lb(self, lb):
        # Internal, used only by the model
        self._lb = lb

    def _internal_set_ub(self, ub):
        # INTERNAL
        self._ub = ub

    @property
    def ub(self):
        """ This property is used to get or set the upper bound of the variable.

        Possible values for the upper bound depend on the variable type. Binary variables
        accept only 0 or 1 as bounds. An integer variable will convert the upper bound value to the
        floor integer value of the argument.

        To reset the upper bound to its default infinity value, use :func:`docplex.mp.model.Model.infinity`.
        """
        return self._ub

    @ub.setter
    def ub(self, new_ub):
        self.set_ub(new_ub)

    def set_ub(self, ub):
        if ub != self._ub:
            self._model.set_var_ub(self, ub)
            return self._ub

    def has_free_lb(self):
        return self.lb <= - self._model.infinity

    def has_free_ub(self):
        return self.ub >= self._model.infinity

    def is_free(self):
        return self.has_free_lb() and self.has_free_ub()

    def _reset_bounds(self):
        vartype = self._vartype
        vtype_lb, vtype_ub = vartype.default_lb, vartype.default_ub
        self.set_lb(vtype_lb)
        self.set_ub(vtype_ub)

    @property
    def vartype(self):
        """ This property returns the variable type, an instance of :class:`VarType`.

        """
        return self._vartype

    def get_vartype(self):
        return self._vartype

    def set_vartype(self, new_vartype):
        # INTERNAL
        self._model._set_var_type(self, new_vartype)

    def _set_vartype_internal(self, new_vartype):
        # INTERNAL
        self._vartype = new_vartype

    def has_type(self, vartype):
        # internal
        return type(self._vartype) == vartype

    def is_binary(self):
        """ Checks if the variable is binary.

        Returns:
            Boolean: True if the variable is of type Binary.
        """
        return self.has_type(BinaryVarType)

    def is_integer(self):
        """ Checks if the variable is integer.

        Returns:
            Boolean: True if the variable is of type Integer.
        """
        return self.has_type(IntegerVarType)

    def is_continuous(self):
        """ Checks if the variable is continuous.

        Returns:
            Boolean: True if the variable is of type Continuous.
        """
        return self.has_type(ContinuousVarType)

    def is_discrete(self):
        """  Checks if the variable is discrete.

        Returns:
            Boolean: True if the variable is of  type Binary or Integer.
        """
        return self._vartype.is_discrete()

    @property
    def float_precision(self):
        return 0 if self.is_discrete() else self._model.float_precision

    def get_value(self):
        # for compatibility only: use solution_value instead
        print("* get_value() is deprecated, use property solution_value instead")  # pragma: no cover
        return self.solution_value  # pragma: no cover

    @property
    def solution_value(self):
        """ This property returns the solution value of the variable.

        Raises:
            DOCplexException
                if the model has not been solved succesfully.

        """
        self._check_model_has_solution()
        return self._get_solution_value()

    @property
    def unchecked_solution_value(self):
        # INTERNAL
        return self._get_solution_value()

    def _get_solution_value(self, s=None):
        sol = s or self.model._get_solution()
        return sol._get_var_value(self)

    def get_container_index(self):
        ctn = self.container
        return ctn.index if ctn else -1

    def get_key(self):
        """ Returns the key used to create the variable, or None.

        When the variable is part of a list or dictionary of variables created from a sequence of keys,
        returns the key associated with the variable.

        Example:
            xs = m.continuous_var_dict(keys=['a', 'b', 'c'])
            xa = xs['a']
            assert xa.get_key() == 'a'

        :return:
            a Python object, possibly None.
        """
        ctn = self.container
        return ctn.get_var_key(self) if ctn else None

    def __mul__(self, e):
        return self.times(e)

    @classmethod
    def is_zero_op(cls, other):
        try:
            return other.is_zero()
        except AttributeError:
            return False

    def times(self, e):
        if is_number(e):
            return self.lfactory._new_monomial_expr(dvar=self, coeff=e, safe=False)

        elif self.is_zero_op(e):
            return self.lfactory.new_zero_expr()
        elif isinstance(e, LinearOperand):
            return self._model._qfactory.new_var_product(self, e)
        else:
            return self.to_linear_expr().multiply(e)

    def __rmul__(self, e):
        return self.times(e)

    def __add__(self, e):
        return self.plus(e)

    @staticmethod
    def _extract_calling_ct_xhs():
        _searched_patterns = [("lhs", 0), ("left_expr", 0), ("rhs", 1), ("right_expr", 1)]
        import inspect
        # need to get 2 steps higher to find caller to add/sub
        frame = inspect.stack()[2]
        code_context = frame.code_context if PY3 else frame[4]

        def find_in_line(line_):
            for xhs_s, xhs_p in _searched_patterns:
                if xhs_s in line_:
                    return line_.find(xhs_s), xhs_p
            else:
                return -1, -1

        if code_context:
            line = code_context[0]
            if line:
                spos, lr = find_in_line(line)

                if spos > 1:
                    assert lr >= 0
                    # strip whitespace before code...
                    ct_varname = line[:spos - 1].lstrip()
                    # evaluate ct in caller locals dict
                    subframe = frame.frame if PY3 else frame[0]
                    ct_object = subframe.f_locals.get(ct_varname)
                    # returns a constraint (or None if that fails), plus 0-1 (0 for lhs, 1 for rhs)
                    return ct_object, lr
        return None, -1

    def _perform_arithmetic_to_self(self, self_arithmetic_method, e):
        # INTERNAL
        res = self_arithmetic_method(e)
        ct, xhs_pos = self._extract_calling_ct_xhs()
        if ct is not None:
            self.get_linear_factory().set_linear_constraint_expr_from_pos(ct, xhs_pos, res)
        return res

    def add(self, e):
        res = self.plus(e)
        ct, xhs_pos = self._extract_calling_ct_xhs()
        if ct is not None:
            self.get_linear_factory().set_linear_constraint_expr_from_pos(ct, xhs_pos, res)
        return res

    def subtract(self, e):
        res = self.minus(e)
        ct, xhs_pos = self._extract_calling_ct_xhs()
        if ct is not None:
            self.get_linear_factory().set_linear_constraint_expr_from_pos(ct, xhs_pos, res)
        return res

    def plus(self, e):
        if isinstance(e, Var):
            expr = self._make_linear_expr()
            expr._add_term(e)
            return expr

        elif is_number(e):
            return self._make_linear_expr(constant=e, safe=False)
        elif is_quad_expr(e):
            return e.plus(self)
        else:
            return self.to_linear_expr().add(e)

    def to_linear_expr(self):
        # INTERNAL
        return self._make_linear_expr()

    def _make_linear_expr(self, constant=0, safe=True):
        return self.lfactory.linear_expr(self, constant, name=None, safe=safe, transient=True)

    def __radd__(self, e):
        return self.plus(e)

    def __sub__(self, e):
        return self.minus(e)

    def minus(self, e):
        if isinstance(e, LinearOperand):
            return self.to_linear_expr().subtract(e)

        elif is_number(e):
            # v -k -> expression(v,-1) -k
            return self._make_linear_expr(constant=-e, safe=False)

        elif is_quad_expr(e):
            return e.rminus(self)
        else:
            return self.to_linear_expr().subtract(e)

    def __rsub__(self, e):

        expr = self.get_linear_factory()._to_linear_operand(e, force_clone=True)  # makes a clone.
        return expr.subtract(self)

    def divide(self, e):
        return self.to_linear_expr().divide(e)

    def __div__(self, e):
        return self.divide(e)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.divide(e)  # pragma: no cover

    def __rtruediv__(self, e):
        # for py3
        self.fatal("Variable {0!s} cannot be used as denominator of {1!s}", self, e)  # pragma: no cover

    def __rdiv__(self, e):
        self.fatal("Variable {0!s} cannot be used as denominator of {1!s}", self, e)

    def __pos__(self):
        # the "+e" unary plus is syntactic sugar
        return self

    def __neg__(self):
        # the "-e" unary minus returns a linear expression
        return self.negate()

    def __pow__(self, power):
        # INTERNAL
        if 0 == power:
            return 1
        elif 1 == power:
            return self
        elif 2 == power:
            return self.square()
        else:
            self.model.unsupported_power_error(self, power)

    def __rshift__(self, other):
        """ Redefines the right-shift operator to create indicators.

        This operator allows to create indicators with the `>>` operator.
        It expects a linear constraint as second argument.

        :param other: a linear constraint used to create the indicator

        :return:
            an instance of IndicatorConstraint, that is not added to the model.
            Use `Model.add()` to add it to the model.

        Note:
            The variable must be binary, otherwise an exception is raised.

        Example:

            >>> m.add(b >> (x >=3)

            creates an indicator which links the satisfaction of the constraint (x >= 3)
            to the value of binary variable b.
        """
        return self._model.indicator_constraint(self, other)

    def square(self):
        return self._model._qfactory.new_var_square(self)

    def __int__(self):
        """ Converts a decision variable to a integer number.

        This is only possible for discrete variables,
        and when the model has been solved successfully.
        If the model has been solved, returns the variable's solution value.

        Returns:
            int: The variable's solution value.

        Raises:
            DOCplexException
                if the model has not been solved successfully.
            DOCplexException
                if the variable is not discrete.
        """

        if self.is_continuous():
            self.fatal("Cannot convert continuous variable value to int: {0!s}", self)
        return int(self.solution_value)

    def __float__(self):
        """ Converts a decision variable to a floating-point number.

        This is only possible when the model has been solved successfully,
        otherwise an exception is raised.
        If the model has been solved, it returns the variable's solution value.

        Returns:
            float: The variable's solution value.
        Raises:
            DOCplexException
                if the model has not been solved successfully.
        """
        return float(self.solution_value)

    def to_bool(self, precision=1e-6):
        """ Converts a variable value to True or False.

        This is only possible for discrete variables and assumes there is a solution.

        Raises:
            DOCplexException
                if the model has not been solved successfully.
            DOCplexException
                if the variable is not discrete.

        Returns:
            Boolean: True if the variable value is nonzero, else False.
        """
        if not self.is_discrete():
            self.fatal("boolean conversion only for discrete variables, type is {0!s}", self.vartype)
        value = self.solution_value  # this property checks for a solution.
        return abs(value) >= precision

    def __str__(self):
        """
        Returns:
            string: A string representation of the variable.

        """
        return self.to_string()

    def to_string(self):
        return self.lp_name

    def print_name(self):
        # INTERNAL
        return self.lp_name

    @property
    def lp_name(self):
        return self._name or "x%s" % self.index1

    @property
    def lpt_name(self):
        # 'c1', 'b2', 'i3' ... first letter of cplex code + index
        cpx_typecode = self.cplex_typecode.lower()
        return "%s%d" % (cpx_typecode, self.index1)

    @property
    def cplex_typecode(self):
        return  self._vartype.cplex_typecode

    def _must_print_lb(self):
        return self.cplex_typecode not in 'SN' and self.lb == self._vartype.default_lb

    def __repr__(self):
        self_vartype, self_lb, self_ub = self._vartype, self.lb, self.ub
        # print lb for semi-xx
        if self._must_print_lb():
            repr_lb = ''
        else:
            repr_lb = ',lb={0:g}'.format(self_lb)
        if self_vartype.default_ub == self_ub:
            repr_ub = ''
        else:
            repr_ub = ',ub={0:g}'.format(self_ub)
        if self.has_name():
            repr_name = ",name='{0}'".format(self.name)
        else:
            repr_name = ''
        cpxc = self.cplex_typecode
        return "docplex.mp.Var(type={0}{1}{2}{3})". \
            format(cpxc, repr_name, repr_lb, repr_ub)

    @property
    def reduced_cost(self):
        """ Returns the reduced cost of the variable.

       This method will raise an exception if the model has not been solved as a LP.

        Note:
            For a large number of variables (> 100), using the `Model.reduced_costs()` method can be much faster.

        Returns:
            The reduced cost of the variable (a float value).

        See Also:

            :func:`docplex.mp.model.Model.reduced_costs`
        """
        return self._model._reduced_cost1(self)

    @property
    def basis_status(self):
        """ This property returns the basis status of the variable, if any.
        The variable must be continuous, otherwise an exception is raised.

        Returns:
            An enumerated value from the enumerated type `docplex.constants.BasisStatus`.

        Note:
            for the model to hold basis information, the model must have been solved as a LP problem.
            In some cases, a model which failed to solve may still have a basis available. Use
            `Model.has_basis()` to check whether the model has basis information or not.

        See Also:
            :func:`docplex.mp.model.Model.has_basis`
            :class:`docplex.mp.constants.BasisStatus`

        """
        if not self.is_continuous():
            self.fatal("Basis status is for continuous variables, {0!s} has type {1!s}", self, self.vartype.short_name)
        return self._model._var_basis_status1(self)

    @property
    def benders_annotation(self):
        """
        This property is used to get or set the Benders annotation of a variable.
        The value of the annotation must be a positive integer

        """
        return self.get_benders_annotation()

    @benders_annotation.setter
    def benders_annotation(self, new_anno):
        self.set_benders_annotation(new_anno)

    def iter_constraints(self):
        """ Returns an iterator traversing all constraints in which the variable is used.

        :return:
            An iterator.
        """
        for ct in self._model.iter_constraints():
            for ctv in ct.iter_variables():
                if ctv is self:
                    yield ct
                    break

    def equals(self, other):
        """
        This method is used to test equality to an expression.
        Because of the overloading of operator `==` through the redefinition of
        the `__eq__` method, you cannot use `==` to test for equality.
        In order to test that two decision variables ar ethe same, use th` Python `is` operator;
        use the `equals` method to test whether a given expression is equivalent to a variable:
        for example, calling `equals` with a linear expression which consists of this variable only,
        with a coefficient of 1, returns True.

        Args:
            other: an expression or a variable.

        :return:
            A boolean value, True if the passed variable is this very variable, or
            if the passed expression is equivalent to the variable, else False.

        """
        # noinspection PyPep8
        return self is other or \
               (isinstance(other, LinearOperand) and
                other.get_constant() == 0 and
                other.number_of_terms() == 1 and
                other.unchecked_get_coef(self) == 1)

    def as_logical_operand(self):
        # INTERNAL
        return self if self.is_binary() else None

    def _check_binary_variable_for_logical_op(self, op_name):
        if not self.is_binary():
            self.fatal("Logical {0} is available only for binary variables, {1} has type {2}",
                       op_name, self, self.vartype.short_name)

    def logical_and(self, other):
        self._check_binary_variable_for_logical_op(op_name="and")
        StaticTypeChecker.typecheck_logical_op(self, other, caller="Var.logical_and")
        return self.get_linear_factory().new_logical_and_expr([self, other])

    def logical_or(self, other):
        self._check_binary_variable_for_logical_op(op_name="or")
        StaticTypeChecker.typecheck_logical_op(self, other, caller="Var.logical_or")
        return self.get_linear_factory().new_logical_or_expr([self, other])

    def logical_not(self):
        self._check_binary_variable_for_logical_op(op_name="not")
        return self.get_linear_factory().new_logical_not_expr(self)

    def __and__(self, other):
        return self.logical_and(other)

    def __or__(self, other):
        return self.logical_or(other)

    # no unary not in magic methods...
