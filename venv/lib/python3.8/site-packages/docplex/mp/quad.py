# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
from six import iteritems as six_iteritems

from docplex.mp.compat23 import unitext
from docplex.mp.constants import UpdateEvent
from docplex.mp.basic import _SubscriptionMixin
from docplex.mp.linear import Expr, AbstractLinearExpr, Var, ZeroExpr
from docplex.mp.utils import *
from docplex.mp.xcounter import update_dict_from_item_value
from docplex.mp.sttck import StaticTypeChecker


def _compare_vars(v1, v2):
    v1i = v1._index
    v2i = v2._index
    return (v1i > v2i) - (v2i > v1i)


class VarPair(object):
    __slots__ = ("first", "second", "_cached_hash")

    def __init__(self, v1, v2=None):
        if v2 is None:
            self.first = v1
            self.second = v1
        else:
            if _compare_vars(v1, v2) <= 0:
                self.first = v1
                self.second = v2
            else:
                self.first = v2
                self.second = v1
        self._cached_hash = self._hash_pair()

    def is_square(self):
        return self.first is self.second

    def __eq__(self, other):
        # INTERNAL: necessary for use as dict keys
        # VarPair ensures variables are sorted by indices
        return isinstance(other, VarPair) and (self.first is other.first) and (self.second is other.second)

    def _hash_pair(self):
        f = hash(self.first)
        s = hash(self.second)
        # cantor encoding. must cast to int() for py3
        self_hash = int(((f + s) * (s + f + 1) / 2) + s)
        if self_hash == -1:
            # value -1 is reserved for errors
            self_hash = -2  # pragma: no cover
        return self_hash

    def __hash__(self):
        return self._cached_hash

    def __repr__(self):
        return "docplex.mp.quad.VarPair(first={0!s},second={1!s})".format(self.first, self.second)

    def __str__(self):
        return "VarPair({0!s}, {1!s})".format(self.first, self.second)

    def __getitem__(self, item):
        if 0 == item:
            return self.first
        elif 1 == item:
            return self.second
        else:
            raise StopIteration

    def index_tuple(self):
        return self.first._index, self.second._index


class QuadExpr(_SubscriptionMixin, Expr):
    """QuadExpr()

    This class models quadratic expressions.
    This class is not intended to be instantiated. Quadratic expressions are built
    either by using operators or by using :func:`docplex.mp.model.Model.quad_expr`.

    """
    def _new_term_dict(self):
        return self._model._lfactory.term_dict_type()

    def copy(self, target_model, var_mapping):
        copied_quads = self._quadterms.__class__()
        for qv1, qv2, qk in self.iter_quad_triplets():
            new_v1 = var_mapping[qv1]
            new_v2 = var_mapping[qv2]
            copied_quads[VarPair(new_v1, new_v2)] = qk

        copied_linear = self._linexpr.copy(target_model, var_mapping)
        return QuadExpr(model=target_model,
                        quads=copied_quads,
                        linexpr=copied_linear,
                        name=self.name,
                        safe=True)

    def relaxed_copy(self, relaxed_model, var_map):
        raise DocplexLinearRelaxationError(self, cause='quadratic')

    def is_quad_expr(self):
        return True

    def has_quadratic_term(self):
        """ Returns true if there is at least one quadratic term in the expression.
        """
        return any(qk for _, qk in self.iter_quads())

    def square(self):
        if self.has_quadratic_term():
            self.fatal("Cannot take the square of a quadratic term: {0!s}".format(self))
        else:
            return self._linexpr.square()

    def _get_solution_value(self, s=None):
        # INTERNAL
        quad_value = 0
        for qv0, qv1, qk in self.iter_quad_triplets():
            quad_value += qk * (qv0._get_solution_value(s) * qv1._get_solution_value(s))
        lin_value = self._linexpr._get_solution_value(s)
        return quad_value + lin_value

    __slots__ = ('_quadterms', '_linexpr', '_transient', '_subscribers')

    def __init__(self, model, quads=None, linexpr=None, name=None, safe=False):
        Expr.__init__(self, model, name)
        self._transient = False
        self._subscribers = []  # used by subscription mixin
        if quads is None:
            self._quadterms = self._new_term_dict()
        elif isinstance(quads, dict):
            if safe:
                self._quadterms = quads
            else:
                # check
                safe_quads = self._new_term_dict()
                for qvp, qk in six_iteritems(quads):
                    model._typecheck_num(qk)
                    if not isinstance(qvp, VarPair):
                        self.fatal("Expecting variable-pair, got: {0!r}", qvp)
                    else:
                        safe_quads[qvp] = qk
                self._quadterms = safe_quads

        elif isinstance(quads, tuple):
            try:
                v1, v2, qk = quads
                if not safe:
                    model._typecheck_var(v1)
                    model._typecheck_var(v2)
                    model._typecheck_num(qk, 'QuadExpr')
                self._quadterms = model._lfactory._new_term_dict()
                self._quadterms[VarPair(v1, v2)] = qk

            except ValueError:  # pragma: no cover
                self.fatal("QuadExpr accepts tuples of len: 3, got: {0!r}", quads)

        elif is_iterable(quads):
            qterms = model._lfactory.term_dict_type()
            for qv1, qv2, qk in quads:
                qterms[VarPair(qv1, qv2)] = qk
            self._quadterms = qterms
        else:
            self.fatal("unexpected argument for QuadExpr: {0!r}", quads)  # pragma: no cover

        if quads is not None:
            self._model._quad_count += 1

        if linexpr is None:
            self._linexpr = model._lfactory.linear_expr()
        else:
            self._linexpr = model._lfactory._to_linear_expr(linexpr)

    def clone_if_necessary(self):
        #  INTERNAL
        if self._transient and not self._model._keep_all_exprs and not self.is_in_use():
            return self
        else:
            return self.clone()

    def keep(self):
        self._transient = False

    def clone(self):
        """ Makes a copy of the quadratic expression and returns it.

        Returns:
            A quadratic expression.
        """
        cloned_linear = self._linexpr.clone()
        self_name = self.name
        cloned_name = self_name if self_name is None else self_name[:]
        new_quad = QuadExpr(self.model, quads=self._quadterms.copy(),
                            linexpr=cloned_linear,
                            name=cloned_name, safe=True)
        return new_quad

    def is_discrete(self):
        for qv0, qv1, qk in self.iter_quad_triplets():
            if not qv0.is_discrete or not qv1.is_discrete() or not is_int(qk):
                return False
        return self._linexpr.is_discrete()

    def generate_quad_triplets(self):
        # INTERNAL
        # a generator that returns triplets (i.e. tuples of len 3)
        # with the variable pair and the coefficient
        for qvp, qk in six_iteritems(self._quadterms):
            yield qvp[0], qvp[1], qk

    def iter_quads(self):
        return six_iteritems(self._quadterms)

    def iter_sorted_quads(self):
        if self.is_model_ordered():
            return six_iteritems(self._quadterms)
        else:
            return self._iter_sorted_quads()

    def _iter_sorted_quads(self):
        quadterms = self._quadterms
        for vp in sorted(quadterms.keys(), key=lambda vvp: vvp.index_tuple()):
            yield vp, quadterms[vp]

    def iter_opposite_ordered_quads(self):
        # INTERNAL
        for qv, qk in self.iter_sorted_quads():
            yield qv, -qk

    def iter_quad_triplets(self):
        """ Iterates over quadratic terms.

        This iterator returns triplets of the form `v1,v2,k`, where `v1` and `v2` are decision
        variables and `k` is a number.

        Returns:
            An iterator object.
        """
        return self.generate_quad_triplets()

    def iter_terms(self):
        """ Iterates over the linear terms in the quadratic expression.

        Equivalent to self.linear_part.iter_terms()

        Returns:
            An iterator over the (variable, coefficient) pairs in the linear part of the expression.

        Example:
            Calling this method on (x^2 +2x+1) will return one pair (x, 2).

        See Also:
            :function:`docplex.mp.linear.LinearExpr.iter_terms`
        """
        return self._linexpr.iter_terms()

    @property
    def number_of_quadratic_terms(self):
        """ This property returns the number of quadratic terms.

        Counts both the square and product terms.

        Examples:
        
        .. code-block:: python

           q1 = x**2
           q1.number_of_quadratic_terms
           >>> 1
           q2 = (x+y+1)**2
           q2.number_of_quadratic_terms
           >>> 3
        """
        return len(self._quadterms)

    @property
    def size(self):
        return self.number_of_quadratic_terms + self._linexpr.size

    def is_separable(self):
        """ Checks if all quadratic terms are separable.

        Returns:
            True if all quadratic terms are separable.
        """
        for qv, _ in self.iter_quads():
            if not qv.is_square():
                return False
        else:
            return True

    def compute_separable_convexity(self, sense=1):
        # INTERNAL
        # returns 1 if separable, convex
        # returns -1 if separable non convex
        # return s0 if non separable

        justifier = None
        for qv, qk in self.iter_quads():
            if not qv.is_square():
                return 0, None  # non separable: fast exit
            elif qk * sense < 0:
                if justifier is None:
                    justifier = (qk, qv[0])  # separable, non convex, kept
        else:
            return justifier or (1, None)  # (1, None) is for separable, convex

    def get_quadratic_coefficient(self, var1, var2=None):
        ''' Returns the coefficient of a quadratic term in the expression.

        Returns the coefficient of the quadratic term `var1*var2` in the expression, if any.
        If the product is not present in the expression, returns 0.

        Args:
            var1: The first variable of the product (an instance of class Var)
            var2: the second variable of the product. If passed None, returns the coefficient
                of the square of `var1` in the expression.

        Example:
            Assuming `x` and `y` are decision variables and `q` is the expression `2*x**2 + 3*x*y + 5*y**2`, then

            `q.get_quadratic_coefficient(x)` returns 2

            `q.get_quadratic_coefficient(x, y)` returns 3

            `q.get_quadratic_coefficient(y)` returns 5

        Returns:
            The coefficient of one quadratic product term in the expression.
        '''
        self.model._typecheck_var(var1)
        if var2 is None:
            var2 = var1
        else:
            self.model._typecheck_var(var2)
        return self._get_quadratic_coefficient(var1, var2)

    def _get_quadratic_coefficient(self, var1, var2):
        # INTERNAL, no checks
        vp = VarPair(var1, var2 or var1)
        return self._get_quadratic_coefficient_from_var_pair(vp)

    def _get_quadratic_coefficient_from_var_pair(self, vp):
        # INTERNAL
        return self._quadterms.get(vp, 0)

    def set_quadratic_coefficient(self, var1, var2, k):
        self.model._typecheck_var(var1)
        if var2 is None:
            var2 = var1
        else:
            self.model._typecheck_var(var2)
        self._set_quadratic_coefficient(var1, var2, k)

    def _set_quadratic_coefficient(self, var1, var2, k):
        vp = VarPair(var1, var2 or var1)
        event = UpdateEvent.QuadExprGlobal
        self_quadterms = self._quadterms
        if not k:
            if vp in self_quadterms:
                del self_quadterms[vp]
            else:
                event = UpdateEvent.NoOp
        else:
            self_quadterms[vp] = k
        self.notify_modified(event)

    # ---
    def equals(self, other):
        if not isinstance(other, QuadExpr):
            return False
        if self.number_of_quadratic_terms != other.number_of_quadratic_terms:
            return False

        for qvp, qk in self.iter_quads():
            if other._get_quadratic_coefficient_from_var_pair(qvp) != qk:
                return False
        return self._linexpr.equals(other._linexpr)

    def is_constant(self):
        return not self.has_quadratic_term() and self._linexpr.is_constant()

    def get_constant(self):
        return self._linexpr.constant

    def set_constant(self, num):
        linexpr = self._linexpr
        event = None
        if num != linexpr.get_constant():
            linexpr._constant = num
            event = UpdateEvent.ExprConstant
        self.notify_modified(event)

    @property
    def constant(self):
        """This property is used to get or set the constant part of a quadratic expression
        """
        return self.get_constant()

    @constant.setter
    def constant(self, new_cst):
        self.set_constant(new_cst)

    @property
    def linear_part(self):
        """ This property returns the linear part of a quadratic expression.

        For example, the linear part of x^2 +2x+1 is (2x+1)

        :return: an instance of :class:`docplex.mp.LinearExpr`
        """
        return self.get_linear_part()

    def get_linear_part(self):
        linexpr = self._linexpr
        return linexpr if linexpr else 0

    def iter_variables(self):
        for qvp, _ in self.iter_quads():
            if qvp.is_square():
                yield qvp[0]
            else:
                yield qvp[0]
                yield qvp[1]
        linexpr = self._linexpr
        for lv in linexpr.iter_variables():
            yield lv

    def __contains__(self, dvar):
        return self.contains_var(dvar)

    def contains_var(self, dvar):
        # required by tests...
        for qv in self.iter_variables():
            if qv is dvar:
                return True
        else:
            return False

    def __iter__(self):
        # INTERNAL: this is necessary to prevent expr from being an iterable.
        # as it follows getitem protocol, it can mistakenly be interpreted as an iterable
        # but this would make sum loop forever.
        raise TypeError  # pragma: no cover

    def contains_quad(self, qv):
        # INTERNAL
        return qv in self._quadterms

    def __repr__(self):
        return "docplex.mp.quad.QuadExpr(%s)" % self.truncated_str()

    def to_stringio(self, oss, nb_digits, use_space, var_namer=lambda v: v.lp_name):
        q = 0
        # noinspection PyPep8Naming
        SP = u' '
        for qvp, qk in self.iter_sorted_quads():
            if not qk:
                continue
            qv1 = qvp.first
            qv2 = qvp.second
            # ---
            # sign is printed if  non-first OR negative
            # at the end of this block coeff is positive
            if qk < 0 or q > 0:
                oss.write(u'-' if qk < 0 else u'+')
                if qk < 0:
                    qk = -qk
                if use_space and q > 0:
                    oss.write(SP)

            # write coeff if <> 1
            varname1 = var_namer(qv1)
            if 1 != qk:
                self._num_to_stringio(oss, num=qk, ndigits=nb_digits)
                if use_space:
                    oss.write(SP)

            oss.write(unitext(varname1))
            if qv1 is qv2:
                oss.write(u"^2")
            else:
                if use_space:
                    oss.write(SP)
                    oss.write(u'*')
                    oss.write(SP)
                else:
                    oss.write(u'*')
                oss.write(unitext(var_namer(qv2)))
            q += 1
        # problem for linexpr: force '+' ssi c>0
        linexpr = self._linexpr
        lin_constant = linexpr.get_constant()
        if linexpr:
            first_lk = 0
            for lv, lk in linexpr.iter_terms():
                if lk:
                    first_lk = lk
                    break
            if q > 0 and first_lk > 0:
                if use_space:
                    oss.write(u' ')
                oss.write(u"+")

            if first_lk:
                if use_space:
                    oss.write(SP)
                self._linexpr.to_stringio(oss, nb_digits, use_space, var_namer)

            elif lin_constant:
                self._num_to_stringio(oss, lin_constant, nb_digits, print_sign=True, force_plus=q > 0,
                                      use_space=use_space)
            elif not q:
                oss.write(u'0')

    def plus(self, other):
        cloned = self.clone_if_necessary()
        cloned.add(other)
        return cloned

    def minus(self, other):
        cloned = self.clone_if_necessary()
        cloned.subtract(other)
        return cloned

    def rminus(self, other):
        # other - self
        cloned = self.clone()
        cloned.negate()
        cloned.add(other)
        return cloned

    def times(self, other):
        if is_number(other) and 0 == other:
            return self.zero_expr()

        elif isinstance(other, ZeroExpr):
            return other

        elif self.is_constant():
            k = self.constant
            if not k:
                return self.zero_expr()
            elif 1 == k:
                return self._model._lfactory._to_linear_expr(other)
            else:
                return other * k
        else:
            cloned = self.clone()
            cloned.multiply(other)
            return cloned

    def __add__(self, other):
        return self.plus(other)

    def __iadd__(self, other):
        self.add(other)
        return self

    def __radd__(self, other):
        return self.plus(other)

    def __sub__(self, other):
        return self.minus(other)

    def __rsub__(self, other):
        # e - self
        return self.rsubtract(other)

    def __isub__(self, other):
        self.subtract(other)
        return self

    def __mul__(self, other):
        return self.times(other)

    def __rmul__(self, other):
        return self.times(other)

    def __imul__(self, other):
        # self is modified
        return self.multiply(other)

    def __div__(self, e):
        return self.quotient(e)

    def __idiv__(self, other):
        self.divide(other, check=True)
        return self

    def __truediv__(self, e):
        return self.quotient(e)  # pragma: no cover

    def __itruediv__(self, other):
        # this is for Python 3.z
        return self.divide(other)  # pragma: no cover

    def __neg__(self):
        cloned = self.clone()
        cloned.negate()
        return cloned

    def add(self, other):
        # increment the QuadExpr with some other argument
        if isinstance(other, QuadExpr):
            event = self._add_quad(other)
        else:
            self._linexpr.add(other)
            event = UpdateEvent.LinExprGlobal
        self.notify_modified(event=event)

    def rsubtract(self, other):
        # to compute (other - self) we copy self, negate the copy and add other
        # result is always cloned even if other is zero (optimization possible here)
        cloned = self.clone()
        cloned.negate()
        cloned.add(other)
        return cloned

    def subtract(self, other):
        if isinstance(other, QuadExpr):
            self._subtract_quad(other)
            event = UpdateEvent.QuadExprGlobal
        else:
            self._linexpr.subtract(other)
            event = UpdateEvent.LinExprGlobal
        self.notify_modified(event=event)
        return self

    def negate(self):
        # INTERNAL: negate sall coefficients, modify self
        qterms = self._quadterms
        for qvp, qk in six_iteritems(qterms):
            qterms[qvp] = -qk
        self._linexpr.negate()
        self.notify_modified(event=UpdateEvent.QuadExprGlobal)
        return self

    def multiply(self, other):
        event = UpdateEvent.QuadExprGlobal
        if is_number(other):
            self._scale(other)

        elif self.is_constant():
            this_constant = self._linexpr.get_constant()
            if 0 == this_constant:
                # do nothing
                event = None
            else:
                self._assign_scaled(other, this_constant)

        elif self.has_quadratic_term():
            if other.is_constant():
                return self.multiply(other.get_constant())
            else:
                StaticTypeChecker.mul_quad_lin_error(self.model, self, other)

        else:
            # self is actually a linear expression
            if is_quad_expr(other):
                if other.has_quadratic_term():
                    StaticTypeChecker.mul_quad_lin_error(self.model, self, other)
                else:
                    return self.multiply(other._linexpr)
            else:
                other_linexpr = other.to_linear_expr()
                self_linexpr = self._linexpr
                for v1, k1 in self_linexpr.iter_terms():
                    for v2, k2 in other_linexpr.iter_terms():
                        self._add_one_quad_term(VarPair(v1, v2), k1 * k2)
                other_cst = other.get_constant()
                self_cst = self_linexpr.get_constant()
                if other_cst:
                    self_linexpr._scale(other_cst)
                if self_cst:
                    for ov, ok in other.iter_terms():
                        self_linexpr._add_term(ov, ok * self_cst)
        self.notify_modified(event)
        return self

    def quotient(self, e):
        self.model._typecheck_as_denominator(e, self)
        cloned = self.clone()
        cloned.divide(e, check=False)
        return cloned

    def divide(self, other, check=True):
        if check:
            self.model._typecheck_as_denominator(other, self)  # only a nonzero number is allowed...
        inverse = 1.0 / other
        self._scale(inverse)
        self.notify_modified(event=UpdateEvent.QuadExprGlobal)
        return self

    def _scale(self, factor):
        # INTERNAL: scales a quad expr from a numeric constant.
        # no checks done!
        # this method modifies self.
        if 0 == factor:
            self.clear()
        elif 1 == factor:
            # nothing to do
            pass
        else:
            # scale quads
            self_quadterms = self._quadterms
            for qv, qk in self.iter_quads():
                self_quadterms[qv] = factor * qk
            # scale linear part
            if self._linexpr is not None:
                self._linexpr._scale(factor)

    def _assign_scaled(self, other, factor):
        # INTERNAL
        if isinstance(other, (AbstractLinearExpr, Var)):
            scaled = self._model._lfactory._to_linear_expr(other, force_clone=True)
            scaled *= factor
            self._linexpr = scaled
        elif isinstance(other, QuadExpr):
            for qv, qk in other.iter_quads():
                self._add_one_quad_term(qv, qk * factor)
            self._assign_scaled(other.linear_part, factor)
        else:
            pass

    def _add_one_quad_term(self, qv, qk):
        qterms = self._quadterms
        if qk or qv in qterms:
            qterms[qv] = qterms.get(qv, 0) + qk

    def normalize(self):  # pragma: no cover
        # INTERNAL
        quadterms = self._quadterms
        if quadterms:
            to_remove = [qvp for qvp, qk in self.iter_quads() if not qk]
            for rvp in to_remove:
                del quadterms[rvp]
        self._linexpr.normalize()

    def clear(self):
        self._quadterms.clear()
        self._linexpr._clear()

    # quad-specific
    def _add_quad(self, other_quad):
        # add quad part
        for oqv, oqk in other_quad.iter_quads():
            self._add_one_quad_term(oqv, oqk)
        # add linear part
        if other_quad._linexpr.is_zero():
            return UpdateEvent.QuadExprQuadCoef
        else:
            self._linexpr.add(other_quad._linexpr)
            return UpdateEvent.QuadExprGlobal

    def _subtract_quad(self, other_quad):
        # subtract quad
        quadterms = self._quadterms
        for oqv, oqk in other_quad.iter_quads():
            update_dict_from_item_value(quadterms, oqv, -oqk)
        # subtract linear part
        self._linexpr.subtract(other_quad._linexpr)

    def to_linear_expr(self, msg=None):  # pragma: no cover
        # used_msg = msg or "Quadratic expression [{0!s}] cannot be converted to a linear expression"
        # self.fatal(used_msg, self)
        raise DocplexQuadToLinearException(self)

    def is_normalized(self):  # pragma: no cover
        # INTERNAL
        for _, qk in self.iter_quads():
            if not qk:
                return False  # pragma: no cover
        else:
            return True

    # --- relational operators
    def __eq__(self, other):
        return self._model._qfactory.new_eq_constraint(self, other)

    def __le__(self, other):
        return self._model._qfactory.new_le_constraint(self, other)

    def __ge__(self, other):
        return self._model._qfactory.new_ge_constraint(self, other)

    def __ne__(self, other):
        self.model.fatal("Operator `!=` is not supported for quadratic expressions, {0!s} was passed", self)

    def notify_expr_modified(self, expr, event):
        if expr is self._linexpr:
            # something to do..
            self.notify_modified(event, )
