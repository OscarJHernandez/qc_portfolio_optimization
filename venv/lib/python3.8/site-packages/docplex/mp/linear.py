# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# pylint: disable=too-many-lines
from __future__ import print_function

from six import iteritems

from docplex.mp.constants import ComparisonType, UpdateEvent
from docplex.mp.compat23 import unitext
from docplex.mp.basic import Expr, ModelingObjectBase, _SubscriptionMixin
from docplex.mp.operand import LinearOperand
from docplex.mp.utils import is_int, is_string, is_number, iter_emptyset, is_quad_expr
from docplex.mp.dvar import Var
from docplex.mp.sttck import StaticTypeChecker


class DOCplexQuadraticArithException(Exception):
    # INTERNAL
    pass


# noinspection PyAbstractClass
class AbstractLinearExpr(LinearOperand, Expr):
    __slots__ = ('_discrete_locked',)

    def get_coef(self, dvar):
        """ Returns the coefficient of a variable in the expression.

        Note:
            If the variable is not present in the expression, the function returns 0.

        :param dvar: The variable for which the coefficient is being queried.

        :return: A floating-point number.
        """
        self.model._typecheck_var(dvar)
        return self.unchecked_get_coef(dvar)

    def __getitem__(self, dvar):
        # direct access to a variable coef x[var]
        return self.unchecked_get_coef(dvar)

    def __iter__(self):
        # INTERNAL: this is necessary to prevent expr from being an iterable.
        # as it follows getitem protocol, it can mistakenly be interpreted as an iterable
        # but this would make sum loop forever.
        raise TypeError

    def lock_discrete(self):
        # intern al: used for any expression used in linear constraints inside equivalences
        self._discrete_locked = True

    def is_discrete_locked(self):
        return getattr(self, '_discrete_locked', False)

    def check_discrete_lock_frozen(self, item=None):
        self.get_linear_factory().check_expr_discrete_lock(self, item)

    def relaxed_copy(self, relaxed_model, var_map):
        return self.copy(relaxed_model, var_map)


class MonomialExpr(_SubscriptionMixin, AbstractLinearExpr):
    # INTERNAL

    def _get_solution_value(self, s=None):
        raw = self.coef * self._dvar._get_solution_value(s)
        return self._round_if_discrete(raw)

    # INTERNAL class
    __slots__ = ('_dvar', '_coef', '_subscribers')

    # noinspection PyMissingConstructor
    def __init__(self, model, dvar, coeff, safe=False):
        self._model = model  # faster than to call recursively init methods...
        self._name = None
        self._dvar = dvar
        self._subscribers = []
        if safe:
            self._coef = coeff
        else:
            validfn = model._checker.get_number_validation_fn()
            self._coef = validfn(coeff) if validfn else coeff

    def number_of_variables(self):
        return 1

    def __hash__(self):
        # py3 requires this function
        return id(self)

    @property
    def var(self):
        return self._dvar

    @property
    def coef(self):
        return self._coef

    @property
    def constant(self):
        # for compatibility
        return 0

    def as_variable(self):
        # INTERNAL
        return self._dvar if 1 == self._coef else None

    def clone(self):
        return self.__class__(self.model, self._dvar, self._coef, safe=True)

    def copy(self, target_model, var_mapping):
        copy_var = var_mapping[self._dvar]
        return MonomialExpr(target_model, dvar=copy_var, coeff=self._coef, safe=True)

    def iter_terms(self):
        yield self._dvar, self._coef

    iter_sorted_terms = iter_terms

    def unchecked_get_coef(self, dvar):
        return self._coef if dvar is self._dvar else 0

    def contains_var(self, dvar):
        return self._dvar is dvar

    def is_normalized(self):
        # INTERNAL
        return self._coef != 0  # pragma: no cover

    def is_discrete(self):
        return self._dvar.is_discrete() and is_int(self._coef)

    # arithmetics
    def negate(self):
        self._coef = - self._coef
        self.notify_modified(event=UpdateEvent.LinExprCoef)
        return self

    def plus(self, e):
        if isinstance(e, LinearOperand) or is_number(e):
            return self.to_linear_expr().add(e)
        else:
            return e.plus(self)

    def minus(self, e):
        if isinstance(e, LinearOperand) or is_number(e):
            expr = self.to_linear_expr()
            expr.subtract(e)
            return expr
        else:
            return e.rminus(self)

    def times(self, e):
        if is_number(e):
            if 0 == e:
                return self.get_linear_factory().new_zero_expr()
            else:
                # return a fresh instance
                return MonomialExpr(self._model, self._dvar, self._coef * e, safe=True)
        elif isinstance(e, LinearExpr):
            return e.times(self)
        elif isinstance(e, Var):
            return self.model._qfactory.new_var_product(e, self)
        elif isinstance(e, MonomialExpr):
            return self.model._qfactory.new_monomial_product(self, e)
        else:
            expr = self.to_linear_expr()
            return expr.multiply(e)

    def square(self):
        return self.model._qfactory.new_monomial_product(self, self)

    def quotient(self, e):
        # returns a new instance
        self._model._typecheck_as_denominator(e, self)
        inverse = 1.0 / float(e)
        return MonomialExpr(self._model, self._dvar, self._coef * inverse, safe=True)

    def __add__(self, e):
        return self.plus(e)

    def __radd__(self, e):
        return self.__add__(e)

    def __sub__(self, e):
        return self.minus(e)

    def __rsub__(self, e):
        return self.get_linear_factory()._to_linear_operand(e, force_clone=True).minus(self)

    def __neg__(self):
        opposite = self.clone()
        return opposite.negate()

    def __mul__(self, e):
        return self.times(e)

    def __rmul__(self, e):
        return self.times(e)

    def __div__(self, e):
        return self.quotient(e)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.__div__(e)  # pragma: no cover

    def __rtruediv__(self, e):
        # for py3
        self.model.cannot_be_used_as_denominator_error(self, e)  # pragma: no cover

    def __rdiv__(self, e):
        self.model.cannot_be_used_as_denominator_error(self, e)

    # changing a coef
    def set_coefficient(self, dvar, coef):
        m = self._model
        m._typecheck_var(dvar)
        m._typecheck_num(coef, 'Expr.set_coefficient()')
        return self._set_coefficient(dvar, coef)

    set_coef = set_coefficient

    def _set_coefficient(self, dvar, coef):
        self.check_discrete_lock_frozen(item=coef)
        if dvar is self._dvar:
            self._coef = coef
            self.notify_modified(event=UpdateEvent.LinExprCoef)
        elif coef:
            # monomail is extended to a linear expr
            new_self = self.to_linear_expr()
            new_self._add_term(dvar, coef)
            # beware self is modified here
            self.notify_replaced(new_self)
            # noinspection PyMethodFirstArgAssignment
            self = new_self
        return self

    # -- arithmetic to self
    def __iadd__(self, other):
        return self._add_to_self(other)

    def _add_to_self(self, other):
        self.check_discrete_lock_frozen(item=other)
        if isinstance(other, LinearOperand) or is_number(other):
            added = self.to_linear_expr().add(other)
        else:
            added = other.plus(self)
        self.notify_replaced(added)
        return added

    def add(self, other):
        return self._add_to_self(other)

    def __isub__(self, other):
        return self._sub_to_self(other)

    def _sub_to_self(self, other):
        # INTERNAL
        self.check_discrete_lock_frozen(item=other)
        if isinstance(other, LinearOperand) or is_number(other):
            expr = self.to_linear_expr()
            expr.subtract(other)
            subtracted = expr
        else:
            subtracted = other.rminus(self)
        self.notify_replaced(subtracted)
        return subtracted

    def subtract(self, other):
        return self._sub_to_self(other)

    def __imul__(self, e):
        return self.multiply(e)

    def multiply(self, e):
        self.check_discrete_lock_frozen(e)
        if is_number(e):
            if 0 == e:
                product = self.get_linear_factory().new_zero_expr()
            else:
                self._coef *= e
                self.notify_modified(event=UpdateEvent.LinExprCoef)
                product = self
        elif isinstance(e, LinearExpr):
            product = e.times(self)
        elif isinstance(e, Var):
            product = self.model._qfactory.new_var_product(e, self)
        elif isinstance(e, MonomialExpr):
            product = self.model._qfactory.new_monomial_product(self, e)
        elif is_quad_expr(e):
            if e.has_quadratic_term():
                StaticTypeChecker.mul_quad_lin_error(self._model, self, e)
            else:
                product = self.model._qfactory.new_monomial_product(self, e.linear_part)
        else:
            product = self.to_linear_expr().multiply(e)
            self.notify_replaced(product)
        return product

    mul = multiply

    def __idiv__(self, other):
        return self.divide(other)

    def __itruediv__(self, other):  # pragma: no cover
        # for py3
        return self.divide(other)

    def divide(self, other):
        self._model._typecheck_as_denominator(other, self)
        inverse = 1.0 / float(other)
        self.check_discrete_lock_frozen(inverse)
        self._coef *= inverse
        self.notify_modified(event=UpdateEvent.LinExprCoef)
        return self

    def equals(self, other):
        return isinstance(other, LinearOperand) and \
               other.get_constant() == 0 and \
               other.number_of_terms() == 1 and \
               other.unchecked_get_coef(self._dvar) == self._coef

    # conversion
    def to_linear_expr(self):
        e = LinearExpr(self._model, e=(self._dvar, self._coef), safe=True, transient=True)
        return e

    def to_stringio(self, oss, nb_digits, use_space, var_namer=lambda v: v.lp_name):
        self_coef = self._coef
        if self_coef != 1:
            if self_coef < 0:
                oss.write(u'-')
                self_coef = - self_coef
            if self_coef != 1:
                self._num_to_stringio(oss, num=self_coef, ndigits=nb_digits)
            if use_space:
                oss.write(u' ')
        oss.write(unitext(var_namer(self._dvar)))

    def __repr__(self):
        return "docplex.mp.MonomialExpr(%s)" % self.to_string()


# from private.debug_deco import count_instances
#
# @count_instances
class LinearExpr(_SubscriptionMixin, AbstractLinearExpr):
    """LinearExpr()

    This class models linear expressions.
    This class is not intended to be instantiated. Expressions are built
    either using operators or using `Model.linear_expr()`.
    """

    @staticmethod
    def _new_terms_dict(model, *args, **kwargs):
        return model._lfactory.term_dict_type(*args, **kwargs)

    @staticmethod
    def _new_empty_terms_dict(model):
        return model._lfactory.term_dict_type()

    def to_linear_expr(self):
        return self

    def _get_terms_dict(self):
        # INTERNAL
        return self.__terms

    def __typecheck_terms_dict(self, terms):  # pragma: no cover
        if not isinstance(terms, dict):
            self.fatal("expecting expression terms as python dict, got: {0!s}", terms)
        self_model = self.model
        for (v, k) in iteritems(terms):
            self_model._typecheck_var(v)
            self_model._typecheck_num(k, 'LinearExpr:importTerms')

    def _assign_terms(self, terms, is_safe=False, assume_normalized=False):  # pragma: no cover
        if not is_safe:
            self.__typecheck_terms_dict(terms)
        if assume_normalized:
            self.__terms = terms
        else:
            self.__terms = self._model._lfactory.term_dict_type([(k, v) for k, v in iteritems(terms)])
        return self

    __slots__ = ('_constant', '__terms', '_transient', '_subscribers')

    def __hash__(self):
        # py3 requires this function
        return id(self)

    def __init__(self, model, e=None, constant=0, name=None, safe=False, transient=False):
        ModelingObjectBase.__init__(self, model, name)
        if not safe and constant:
            model._typecheck_num(constant, 'LinearExpr()')
        self._constant = constant
        self._transient = transient
        self._subscribers = []

        if isinstance(e, dict):
            if safe:
                self.__terms = e
            else:
                self_terms = model._lfactory.term_dict_type()
                for (v, k) in iteritems(e):
                    model._typecheck_var(v)
                    model._typecheck_num(k, 'LinearExpr')
                    if k != 0:
                        self_terms[v] = k
                self.__terms = self_terms
            return
        else:
            self.__terms = model._lfactory._new_term_dict()

        if e is None:
            pass

        elif isinstance(e, Var):
            self.__terms[e] = 1

        elif is_number(e):
            self._constant += e

        elif isinstance(e, MonomialExpr):
            # TODO: simplify by self_terms[e.var] = e.coef
            self._add_term(e.var, e.coef)

        elif isinstance(e, LinearExpr):
            # note that transient is not kept.
            self._constant = e.get_constant()
            self.__terms = self._new_terms_dict(model, e._get_terms_dict())  # make a copy

        elif isinstance(e, tuple):
            v, k = e
            self.__terms[v] = k

        else:
            self.fatal("Cannot convert {0!r} to docplex.mp.LinearExpr, type is {1}", e, type(e))

    def keep(self):
        self._transient = False
        return self

    def is_kept(self):
        # INTERNAL
        return not self._transient

    def is_transient(self):  # pragma: no cover
        # INTERNAL
        return self._transient

    def clone_if_necessary(self):
        #  INTERNAL
        if self._transient and not self._model._keep_all_exprs and not self.is_in_use():
            return self
        else:
            return self.clone()

    def set_name(self, name):
        Expr.set_name(self, name)
        # an expression with a name is not transient any more
        if name:
            self.keep()

    def _get_name(self):
        return self._name

    name = property(_get_name, set_name)

    # from private.debug_deco import count_calls
    # @count_calls
    def clone(self):
        """
        Returns:
            A copy of the expression on the same model.
        """
        cloned_terms = self._new_terms_dict(self._model, self.__terms)  # faster than copy() on OrderedDict()
        cloned = LinearExpr(model=self._model, e=cloned_terms, constant=self._constant, safe=True)
        return cloned

    def copy(self, target_model, var_mapping):
        # INTERNAL
        copied_terms = self._new_terms_dict(target_model)
        for v, k in self.iter_sorted_terms():
            copied_terms[var_mapping[v]] = k
        copied_expr = LinearExpr(model=target_model, e=copied_terms, constant=self.constant, safe=True)
        return copied_expr

    def negate(self):
        """ Takes the negation of an expression.

        Changes the expression by replacing each variable coefficient and the constant term
        by its opposite.

        Note:
            This method does not create any new expression but modifies the `self` instance.

        Returns:
            The modified self.

        """
        self._constant = - self._constant
        self_terms = self.__terms
        for v, k in iteritems(self_terms):
            self_terms[v] = -k
        self.notify_modified(event=UpdateEvent.LinExprGlobal)
        return self

    def _clear(self):
        """ Clears the expression.

        All variables and coefficients are removed and the constant term is set to zero.
        """
        self._constant = 0
        self.__terms.clear()

    def equals_constant(self, scalar):
        """ Checks if the expression equals a constant term.

        Args:
            scalar (float): A floating-point number.
        Returns:
            Boolean: True if the expression equals this constant term.
        """
        return self.is_constant() and (scalar == self._constant)

    def is_zero(self):
        return self.equals_constant(0)

    def is_constant(self):
        """
        Checks if the expression is a constant.

        Returns:
            Boolean: True if the expression consists of only a constant term.
        """
        return not self.__terms

    def _has_nonzero_var_term(self):
        for dv, k in self.iter_terms():
            if k:
                return True
        else:
            return False

    def as_variable(self):
        # INTERNAL: returns True if expression is in fact a variable (1*x)
        if 0 == self.constant and 1 == len(self.__terms):
            for v, k in self.iter_terms():
                if k == 1:
                    return v
        return None

    def is_normalized(self):
        # INTERNAL
        for _, k in self.iter_terms():
            if not k:
                return False  # pragma: no cover
        return True

    def normalize(self):
        # modifies self
        doomed = [dv for dv, k in self.iter_terms() if not k]
        lterms = self.__terms
        for d in doomed:
            del lterms[d]

    def normalized(self):
        if self.is_normalized():
            return self
        else:
            cloned = self.clone()
            cloned.normalize()
            return cloned

    def number_of_variables(self):
        return len(self.__terms)

    def unchecked_get_coef(self, dvar):
        # INTERNAL
        return self.__terms.get(dvar, 0)

    def add_term(self, dvar, coeff):
        """
        Adds a term (variable and coefficient) to the expression.

        Args:
            dvar (:class:`Var`): A decision variable.
            coeff (float): A floating-point number.

        Returns:
            The modified expression itself.
        """
        if coeff:
            self._model._typecheck_var(dvar)
            self._model._typecheck_num(coeff)
            self._add_term(dvar, coeff)
            self.notify_modified(event=UpdateEvent.LinExprCoef)
        return self

    def _add_term(self, dvar, coef=1):
        # INTERNAL
        self_terms = self.__terms
        self_terms[dvar] = self_terms.get(dvar, 0) + coef

    def set_coefficient(self, dvar, coeff):
        self._model._typecheck_var(dvar)
        self._model._typecheck_num(coeff)
        self._set_coefficient(dvar, coeff)

    set_coef = set_coefficient

    def _set_coefficient_internal(self, dvar, coeff):
        self_terms = self.__terms
        if coeff or dvar in self_terms:
            self_terms[dvar] = coeff
            return True
        else:
            return False

    def _set_coefficient(self, dvar, coeff):
        self.check_discrete_lock_frozen(coeff)
        if self._set_coefficient_internal(dvar, coeff):
            self.notify_modified(event=UpdateEvent.LinExprCoef)

    def set_coefficients(self, var_coef_seq):
        # TODO: typecheck
        self._set_coefficients(var_coef_seq)

    set_coefs = set_coefficients

    def _set_coefficients(self, var_coef_seq):
        self.check_discrete_lock_frozen()
        nb_changes = 0
        for dv, k in var_coef_seq:
            if self._set_coefficient_internal(dv, k):
                nb_changes += 1
        if nb_changes:
            self.notify_modified(event=UpdateEvent.LinExprCoef)

    def remove_term(self, dvar):
        """ Removes a term associated with a variable from the expression.

        Args:
            dvar (:class:`Var`): A decision variable.

        Returns:
            The modified expression.

        """
        self.set_coefficient(dvar, 0)

    @property
    def constant(self):
        """
        This property is used to get or set the constant term of the expression.
        """
        return self._constant

    @constant.setter
    def constant(self, new_constant):
        self._set_constant(new_constant)

    def get_constant(self):
        return self._constant

    def set_constant(self, new_constant):
        self._model._checker._typecheck_num(new_constant)
        self._set_constant(new_constant)

    def _set_constant(self, new_constant):
        if new_constant != self._constant:
            self.check_discrete_lock_frozen(new_constant)
            self._constant = new_constant
            self.notify_modified(event=UpdateEvent.ExprConstant)

    def contains_var(self, dvar):
        """ Checks whether a decision variable is part of an expression.

        Args:
            dvar (:class:`Var`): A decision variable.

        Returns:
            Boolean: True if `dvar` is mentioned in the expression with a nonzero coefficient.
        """
        return dvar in self.__terms

    def equals(self, other):
        """
         This method is used to test equality between expressions.
         Because of the overloading of operator `==` through the redefinition of
         the `__eq__` method, you cannot use `==` to test for equality.
         The `equals` method to test whether a given expression is equivalent to a variable.
         Two linear expressions are equivalent if they have the same coefficient for all
         variables.

         Args:
            other: a number or any expression.



         Returns:
             A boolean value, True if the passed expression is equivalent, else False.


         Note:
            A constant expression is considered equivalent to its constant number.

                m.linear_expression(3).equals(3) returns True
        """
        if is_number(other):
            return self.is_constant() and other == self.constant
        else:
            if not isinstance(other, LinearOperand):
                return False
            if self.constant != other.get_constant():
                return False
            if self.number_of_terms() != other.number_of_terms():
                return False
            for dv, k in self.iter_terms():
                if k != other.unchecked_get_coef(dv):
                    return False
            else:
                return True

    # noinspection PyPep8
    def to_stringio(self, oss, nb_digits, use_space, var_namer=lambda v: v.lp_name):
        # INTERNAL
        # Writes unicode representation of self
        c = 0
        # noinspection PyPep8Naming
        SP = u' '

        for v, coeff in self.iter_sorted_terms():
            if not coeff:
                continue  # pragma: no cover

            # 1 separator
            if use_space and c > 0:
                oss.write(SP)

            # ---
            # sign is printed if  non-first OR negative
            # at the end of this block coeff is positive
            if coeff < 0 or c > 0:
                oss.write(u'-' if coeff < 0 else u'+')
                if coeff < 0:
                    coeff = -coeff
                if use_space and c > 0:
                    oss.write(SP)
            # ---

            if 1 != coeff:
                self._num_to_stringio(oss, coeff, nb_digits)
                if use_space:
                    oss.write(SP)

            varname = var_namer(v)
            oss.write(unitext(varname))
            c += 1

        k = self.constant
        if c == 0:
            self._num_to_stringio(oss, k, nb_digits)
        elif k != 0:
            if k < 0:
                sign = u'-'
                k = -k
            else:
                sign = u'+'
            if use_space:
                oss.write(SP)
            oss.write(sign)
            if use_space:
                oss.write(SP)
            self._num_to_stringio(oss, k, nb_digits)

    def _add_expr(self, other_expr):
        # INTERNAL
        self._constant += other_expr._constant
        # merge term dictionaries
        for v, k in other_expr.iter_terms():
            # use unchecked version
            self._add_term(v, k)

    def _add_expr_scaled(self, expr, factor):
        # INTERNAL: used by quadratic
        if factor:
            self._constant += expr.get_constant() * factor
            for v, k in expr.iter_terms():
                # use unchecked version
                self._add_term(v, k * factor)

    # --- algebra methods always modify self.
    def add(self, e):
        """ Adds an expression to self.

        Note:
            This method does not create an new expression but modifies the `self` instance.

        Args:
            e: The expression to be added. Can be a variable, an expression, or a number.

        Returns:
            The modified self.

        See Also:
            The method :func:`plus` to compute a sum without modifying the self instance.
        """
        event = UpdateEvent.LinExprGlobal
        if isinstance(e, Var):
            self._add_term(e, coef=1)
        elif isinstance(e, LinearExpr):
            self._add_expr(e)
        elif isinstance(e, MonomialExpr):
            self._add_term(e._dvar, e._coef)
        elif isinstance(e, ZeroExpr):
            event = None
        elif is_number(e):
            validfn = self._model._checker.get_number_validation_fn()
            valid_e = validfn(e) if validfn else e
            self._constant += valid_e
            event = UpdateEvent.ExprConstant
        elif is_quad_expr(e):
            raise DOCplexQuadraticArithException
        else:
            try:
                self.add(e.to_linear_expr())
            except AttributeError:
                self._unsupported_binary_operation(self, "+", e)

        self.notify_modified(event=event)
        return self

    def iter_terms(self):
        """ Iterates over the terms in the expression.

        Returns:
            An iterator over the (variable, coefficient) pairs in the expression.
        """
        return iteritems(self.__terms)

    def number_of_terms(self):
        return len(self.__terms)

    @property
    def size(self):
        return len(self.__terms) + bool(self._constant)

    def subtract(self, e):
        """ Subtracts an expression from this expression.
        Note:
            This method does not create a new expression but modifies the `self` instance.

        Args:
            e: The expression to be subtracted. Can be either a variable, an expression, or a number.

        Returns:
            The modified self.

        See Also:
            The method :func:`minus` to compute a difference without modifying the `self` instance.
        """
        event = UpdateEvent.LinExprCoef
        if isinstance(e, Var):
            self._add_term(e, -1)
        elif is_number(e):
            self._constant -= e
            event = UpdateEvent.ExprConstant
        elif isinstance(e, LinearExpr):
            if e.is_constant() and 0 == e.get_constant():
                return self
            else:
                # 1. decr constant
                self.constant -= e.constant
                # merge term dictionaries 
                for v, k in e.iter_terms():
                    self._add_term(v, -k)
        elif isinstance(e, MonomialExpr):
            self._add_term(e.var, -e.coef)
        elif isinstance(e, ZeroExpr):
            event = None
        elif is_quad_expr(e):
            #
            raise DOCplexQuadraticArithException
        else:
            try:
                self.subtract(e.to_linear_expr())
            except AttributeError:
                self._unsupported_binary_operation(self, "-", e)
        self.notify_modified(event)
        return self

    def _scale(self, factor):
        # INTERNAL: used my multiply
        # this method modifies self.
        if 0 == factor:
            self._clear()
        elif factor != 1:
            self._constant *= factor
            self_terms = self.__terms
            for v, k in iteritems(self_terms):
                self_terms[v] = k * factor

    def multiply(self, e):
        """ Multiplies this expression by an expression.

        Note:
            This method does not create a new expression but modifies the `self` instance.

        Args:
            e: The expression that is used to multiply `self`.

        Returns:
            The modified `self`.

        See Also:
            The method :func:`times` to compute a multiplication without modifying the `self` instance.
        """
        mul_res = self
        event = UpdateEvent.LinExprGlobal
        self_constant = self.get_constant()
        if is_number(e):
            self._scale(factor=e)

        elif isinstance(e, LinearOperand):
            if e.is_constant():
                # simple scaling
                self._scale(factor=e.get_constant())
            elif self.is_constant():
                # self is constant: import other terms , scaled.
                # set constant to zero.
                if self_constant:
                    for lv, lk in e.iter_terms():
                        self.set_coefficient(dvar=lv, coeff=lk * self_constant)
                    self._constant *= e.get_constant()
            else:
                # yields a quadratic
                mul_res = self.model._qfactory.new_linexpr_product(self, e)
                event = UpdateEvent.LinExprPromotedToQuad

        elif isinstance(e, ZeroExpr):
            self._scale(factor=0)

        elif is_quad_expr(e):
            if not e.number_of_quadratic_terms:
                return self.multiply(e.linear_part)
            elif self.is_constant():
                return e.multiply(self.get_constant())
            else:
                StaticTypeChecker.mul_quad_lin_error(self._model, self, e)

        else:
            self.fatal("Multiply expects variable, expr or number, {0!r} was passed (type is {1})", e, type(e))

        self.notify_modified(event=event)

        return mul_res

    def square(self):
        return self.model._qfactory.new_linexpr_product(self, self)

    def divide(self, e):
        """ Divides this expression by an operand.

        Args:
            e: The operand by which the self expression is divided. Only nonzero numbers are permitted.

        Note:
            This method does not create a new expression but modifies the `self` instance.

        Returns:
            The modified `self`.
        """
        self.model._typecheck_as_denominator(e, numerator=self)
        inverse = 1.0 / float(e)
        return self.multiply(inverse)

    # operator-based API
    def opposite(self):
        cloned = self.clone_if_necessary()
        cloned.negate()
        return cloned

    def plus(self, e):
        """ Computes the sum of the expression and some operand.

        Args:
            e: the expression to add to self. Can be either a variable, an expression or a number.

        Returns:
            a new expression equal to the sum of the self expression and `e`

        Note:
            This method doe snot modify self.
        """
        cloned = self.clone_if_necessary()
        try:
            return cloned.add(e)
        except DOCplexQuadraticArithException:
            return e.plus(self)

    def minus(self, e):
        cloned = self.clone_if_necessary()
        try:
            return cloned.subtract(e)
        except DOCplexQuadraticArithException:
            return e.rminus(self)

    def times(self, e):
        """ Computes the multiplication of this expression with an operand.

        Note:
            This method does not modify the `self` instance but returns a new expression instance.

        Args:
            e: The expression that is used to multiply `self`.

        Returns:
            A new instance of expression.
        """
        cloned = self.clone_if_necessary()
        return cloned.multiply(e)

    def quotient(self, e):
        """ Computes the division of this expression with an operand.

        Note:
            This method does not modify the `self` instance but returns a new expression instance.

        Args:
            e: The expression that is used to modify `self`. Only nonzero numbers are permitted.

        Returns:
            A new instance of expression.
        """
        cloned = self.clone_if_necessary()
        cloned.divide(e)
        return cloned

    def __add__(self, e):
        return self.plus(e)

    def __radd__(self, e):
        return self.plus(e)

    def __iadd__(self, e):
        try:
            self.add(e)
            return self
        except DOCplexQuadraticArithException:
            r = e + self
            self.notify_replaced(new_expr=r)
            return r

    def __sub__(self, e):
        return self.minus(e)

    def __rsub__(self, e):
        cloned = self.clone_if_necessary()
        cloned.subtract(e)
        cloned.negate()
        return cloned

    def __isub__(self, e):
        try:
            return self.subtract(e)
        except DOCplexQuadraticArithException:
            r = -e + self
            return r

    def __neg__(self):
        return self.opposite()

    def __mul__(self, e):
        return self.times(e)

    def __rmul__(self, e):
        return self.times(e)

    def __imul__(self, e):
        return self.multiply(e)

    def __div__(self, e):
        return self.quotient(e)

    def __idiv__(self, other):
        return self.divide(other)

    def __itruediv__(self, other):
        # this is for Python 3.z
        return self.divide(other)  # pragma: no cover

    def __truediv__(self, e):
        return self.__div__(e)  # pragma: no cover

    def __rtruediv__(self, e):
        self.fatal("Expression {0!s} cannot be used as divider of {1!s}", self, e)  # pragma: no cover

    @property
    def solution_value(self):
        """ This property returns the solution value of the variable.

        Raises:
            DOCplexException
                if the model has not been solved.
        """
        self._check_model_has_solution()
        return self._get_solution_value()

    def _get_solution_value(self, s=None):
        # INTERNAL: no checks
        val = self._constant
        sol = s or self._model.solution
        for var, koef in self.iter_terms():
            val += koef * sol._get_var_value(var)
        return self._round_if_discrete(val)

    def is_discrete(self):
        """ Checks if the expression contains only discrete variables and coefficients.

        Example:
            If X is an integer variable, X, X+1, 2X+3 are discrete
            but X+0.3, 1.5X, 2X + 0.7 are not.

        Returns:
            Boolean: True if the expression contains only discrete variables and coefficients.
        """
        self_cst = self._constant
        if self_cst != int(self_cst):
            # a float constant with integer value is OK
            return False

        for v, k in self.iter_terms():
            if not v.is_discrete() or not is_int(k):
                return False
        else:
            return True

    def __repr__(self):
        return "docplex.mp.LinearExpr({0})".format(self.truncated_str())

    def _iter_sorted_terms(self):
        # internal
        self_terms = self.__terms
        for dv in sorted(self_terms.keys(), key=lambda v: v._index):
            yield dv, self_terms[dv]

    def iter_sorted_terms(self):
        if self.is_model_ordered():
            return self.iter_terms()
        else:
            return self._iter_sorted_terms()


LinearConstraintType = ComparisonType


class ZeroExpr(_SubscriptionMixin, AbstractLinearExpr):
    def _get_solution_value(self, s=None):
        return 0

    def is_zero(self):
        return True

    # INTERNAL
    __slots__ = ('_subscribers',)

    def __hash__(self):
        return id(self)

    def __init__(self, model):
        ModelingObjectBase.__init__(self, model)
        self._subscribers = []

    def clone(self):
        return self  # this is not cloned.

    def copy(self, target_model, var_map):
        return ZeroExpr(target_model)

    def to_linear_expr(self):
        return self  # this is a linear expr.

    def number_of_variables(self):
        return 0

    def number_of_terms(self):
        return 0

    def iter_terms(self):
        return iter_emptyset()

    def is_constant(self):
        return True

    def is_discrete(self):
        return True

    def unchecked_get_coef(self, dvar):
        return 0

    def contains_var(self, dvar):
        return False

    @property
    def constant(self):
        # for compatibility
        return 0

    @constant.setter
    def constant(self, newk):
        if newk:
            cexpr = self.get_linear_factory().constant_expr(newk, safe_number=False)
            self.notify_replaced(cexpr)

    def negate(self):
        return self

    # noinspection PyMethodMayBeStatic
    def plus(self, e):
        return e

    def times(self, _):
        return self

    # noinspection PyMethodMayBeStatic
    def minus(self, e):
        return -e

    def to_string(self, nb_digits=None, use_space=False):
        return '0'

    def to_stringio(self, oss, nb_digits, use_space, var_namer=lambda v: v.name):
        oss.write(self.to_string())

    # arithmetic
    def __sub__(self, e):
        return self.minus(e)

    def __rsub__(self, e):
        # e - 0 = e !
        return e

    def __neg__(self):
        return self

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __div__(self, other):
        return self._divide(other)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.__div__(e)  # pragma: no cover

    def _divide(self, other):
        self.model._typecheck_as_denominator(numerator=self, denominator=other)
        return self

    def __repr__(self):
        return "docplex.mp.ZeroExpr()"

    def equals(self, other):
        return (isinstance(other, LinearOperand) and (
                0 == other.get_constant() and (0 == other.number_of_terms()))) or \
               (is_number(other) and other == 0)

    def square(self):
        return self

    # arithmetic to self
    add = plus
    subtract = minus
    multiply = times

    def __iadd__(self, other):
        linear_other = self.get_linear_factory()._to_linear_operand(other, force_clone=False)
        self.notify_replaced(linear_other)
        return linear_other

    def __isub__(self, other):
        linear_other = self.get_linear_factory()._to_linear_operand(other, force_clone=True)
        negated = linear_other.negate()
        self.notify_replaced(negated)
        return negated


class ConstantExpr(_SubscriptionMixin, AbstractLinearExpr):
    __slots__ = ('_constant', '_subscribers')

    def __init__(self, model, cst):
        ModelingObjectBase.__init__(self, model=model, name=None)
        # assume constant is a number (to be checked upfront)
        self._constant = cst
        self._subscribers = []

    @property
    def size(self):
        return 1 if self._constant else 0

    # INTERNAL
    def _make_new_constant(self, new_value):
        return ConstantExpr(self._model, new_value)

    def _get_solution_value(self, s=None):
        return self._constant

    def is_zero(self):
        return 0 == self._constant

    def clone(self):
        return self.__class__(self._model, self._constant)

    def copy(self, target_model, var_map):
        return self.__class__(target_model, self._constant)

    def to_linear_expr(self):
        return self  # this is a linear expr.

    def number_of_variables(self):
        return 0

    def iter_variables(self):
        return iter_emptyset()

    def iter_terms(self):
        return iter_emptyset()

    def is_constant(self):
        return True

    def is_discrete(self):
        return is_int(self._constant)

    def unchecked_get_coef(self, dvar):
        return 0

    def contains_var(self, dvar):
        return False

    @property
    def constant(self):
        return self._constant

    @constant.setter
    def constant(self, new_constant):
        self._set_constant(new_constant)

    def get_constant(self):
        return self._constant

    def _set_constant(self, new_constant):
        if new_constant != self._constant:
            self.check_discrete_lock_frozen(new_constant)
            self._constant = new_constant
            self.notify_modified(event=UpdateEvent.ExprConstant)

    def negate(self):
        return self._make_new_constant(- self._constant)

    def _apply_op(self, pyop, arg):
        if is_number(arg):
            return self._make_new_constant(pyop(self.constant, arg))
        else:
            return pyop(arg, self._constant)

    # noinspection PyMethodMayBeStatic
    def plus(self, e):
        import operator
        return self._apply_op(operator.add, e)

    def times(self, e):
        return e * self._constant

    # noinspection PyMethodMayBeStatic
    def minus(self, e):
        return self + (-e)

    def to_string(self, nb_digits=None, prod_symbol='', use_space=False):
        return '{0}'.format(self._constant)

    def to_stringio(self, oss, nb_digits, use_space, var_namer=lambda v: v.name):
        self._num_to_stringio(oss, self._constant, nb_digits)

    # arithmetic
    def __sub__(self, e):
        return self.minus(e)

    def __rsub__(self, e):
        # e - k = e !
        return e - self._constant

    def __neg__(self):
        return self._make_new_constant(- self._constant)

    def __add__(self, other):
        return self.plus(other)

    def __radd__(self, other):
        return self.plus(other)

    def __mul__(self, other):
        return self.times(other)

    def __rmul__(self, other):
        return self.times(other)

    def __div__(self, other):
        return self._divide(other)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.__div__(e)  # pragma: no cover

    def _divide(self, other):
        self.model._typecheck_as_denominator(numerator=self, denominator=other)
        return self._make_new_constant(self._constant / other)

    def __repr__(self):
        return 'docplex.mp.linear.ConstantExpr({0})'.format(self._constant)

    def equals_expr(self, other):
        return isinstance(other, ConstantExpr) and self._constant == other.constant

    def square(self):
        return self._make_new_constant(self._constant ** 2)

    # arithmetci to self

    def _scale(self, factor):
        return self._make_new_constant(self._constant * factor)

    def equals(self, other):
        if is_number(other):
            return self._constant == other
        else:
            return isinstance(other, LinearOperand) \
                   and other.is_constant() and \
                   self._constant == other.get_constant()

    # arithmetic to self
    def __iadd__(self, other):
        return self.add(other)

    def add(self, other):
        if is_number(other):
            self._constant += other
            self.notify_modified(UpdateEvent.ExprConstant)
            return self
        elif isinstance(other, LinearOperand) and other.is_constant():
            self._constant += other.get_constant()
            self.notify_modified(UpdateEvent.ExprConstant)
            return self
        else:
            # replace self by other + self.
            added = other.plus(self._constant)
            self.notify_replaced(added)
            return added

    def subtract(self, other):
        if is_number(other):
            self._constant -= other
            self.notify_modified(UpdateEvent.ExprConstant)
            return self
        elif isinstance(other, LinearOperand) and other.is_constant():
            self._constant -= other.get_constant()
            self.notify_modified(UpdateEvent.ExprConstant)
            return self
        else:
            # replace self by (-other) + self.K
            subtracted = other.negate().plus(self._constant)
            self.notify_replaced(subtracted)
            return subtracted

    def __isub__(self, other):
        return self.subtract(other)

    def multiply(self, other):
        if is_number(other):
            self._constant *= other
            self.notify_modified(UpdateEvent.ExprConstant)
            return self
        elif isinstance(other, LinearOperand) and other.is_constant():
            self._constant *= other.get_constant()
            self.notify_modified(UpdateEvent.ExprConstant)
            return self
        else:
            # replace self by (-other) + self.K
            multiplied = other * self._constant
            self.notify_replaced(multiplied)
            return multiplied

    def __imul__(self, other):
        return self.multiply(other)
