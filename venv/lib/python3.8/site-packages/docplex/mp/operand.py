# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.utils import iter_emptyset
from docplex.mp.constants import ComparisonType


class Operand(object):
    __slots__ = ()

    def get_constant(self):
        return 0

    def is_constant(self):
        return False

    # --- basic subscription api
    def notify_used(self, user):
        pass

    def notify_unsubscribed(self, subscriber):
        pass

    # noinspection PyMethodMayBeStatic
    def is_in_use(self):
        return False

    # noinspection PyMethodMayBeStatic
    def is_shared(self):
        return False

    def notify_modified(self, event):
        pass

    def keep(self):
        return self
    # ---

    def resolve(self):
        # used for lazy expansions
        pass


    def __le__(self, rhs):
        return self._model._new_xconstraint(lhs=self, rhs=rhs, comparaison_type=ComparisonType.LE)

    def __eq__(self, rhs):
        return self._model._new_xconstraint(lhs=self, rhs=rhs, comparaison_type=ComparisonType.EQ)

    def __ge__(self, rhs):
        return self._model._new_xconstraint(lhs=self, rhs=rhs, comparaison_type=ComparisonType.GE)


    le = __le__
    eq = __eq__
    ge = __ge__


class LinearOperand(Operand):
    # no ctor as used in multiple inheritance

    def unchecked_get_coef(self, dvar):
        raise NotImplementedError('unchecked_get_coef missing for class: {0}'.format(self.__class__))  # pragma: no cover

    def iter_variables(self):
        """
        Iterates over all variables in the expression.

        Returns:
            iterator: An iterator over all variables present in the operand.
        """
        for v, k in self.iter_terms():
            yield v

    def get_linear_part(self):
        """ Returns the linear part of the expression: for a linear expression,
        returns the expression itself.

        Defined for compatibility with quadratic expressions.

        :return: a linear expression
        """
        return self

    def iter_terms(self):
        # iterates over all linear terms, if any
        return iter_emptyset()

    iter_sorted_terms = iter_terms

    def number_of_terms(self):
        return sum(1 for _ in self.iter_terms())

    @property
    def size(self):
        return self.number_of_terms()

    # noinspection PyMethodMayBeStatic
    def iter_quads(self):
        return iter_emptyset()

    def is_constant(self):
        # redefine this for subclasses.
        return False  # pragma: no cover

    def is_quad_expr(self):
        return False

    def as_variable(self):
        # return a variable if the expression is actually one variable, else None
        return None

    def is_zero(self):
        return False

    def get_constant(self):
        return 0

    # no strict comparisons
    def __lt__(self, e):
        self.model.unsupported_relational_operator_error(self, "<", e)

    def __gt__(self, e):
        self.model.unsupported_relational_operator_error(self, ">", e)

    def __contains__(self, dvar):
        """Overloads operator `in` for an expression and a variable.

        :param: dvar (:class:`docplex.mp.linear.Var`): A decision variable.

        Returns:
            Boolean: True if the variable is present in the expression, else False.
        """
        return self.contains_var(dvar)

    def __ne__(self, rhs):
        return self._model._lfactory.new_neq_constraint(lhs=self, rhs=rhs)

    def contains_var(self, dvar):
        raise NotImplementedError  # pragma: no cover

    def lock_discrete(self):
        pass

    def is_discrete_locked(self):
        return False
