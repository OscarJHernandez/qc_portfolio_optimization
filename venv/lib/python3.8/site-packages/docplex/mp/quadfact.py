# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore
from docplex.mp.mfactory import _AbstractModelFactory

from docplex.mp.constants import ComparisonType, UpdateEvent
from docplex.mp.utils import is_number
from docplex.mp.xcounter import update_dict_from_item_value
from docplex.mp.quad import QuadExpr, VarPair
from docplex.mp.operand import LinearOperand
from docplex.mp.linear import Var, MonomialExpr, ZeroExpr, AbstractLinearExpr
from docplex.mp.constr import QuadraticConstraint


class IQuadFactory(_AbstractModelFactory):
    # INTERNAL

    def new_var_product(self, var, factor):
        raise NotImplementedError  # pragma: no cover

    def new_monomial_product(self, monexpr, factor):
        raise NotImplementedError  # pragma: no cover

    def new_linexpr_product(self, linexpr, factor):
        raise NotImplementedError  # pragma: no cover

    def new_var_square(self, var):
        raise NotImplementedError  # pragma: no cover


class QuadFactory(IQuadFactory):
    def __init__(self, model, engine):
        _AbstractModelFactory.__init__(self, model, engine)
        self._lfactory = model._lfactory

    def new_zero_expr(self):
        return ZeroExpr(self._model)

    def _unexpected_product_error(self, factor1, factor2):
        # INTERNAL
        self._model.fatal("cannot multiply {0!s} by {1!s}", factor1, factor2)

    def new_quad(self, quads=None, linexpr=None, name=None, safe=False):
        return QuadExpr(self._model, quads=quads, linexpr=linexpr, name=name, safe=safe)

    def new_linear_expr(self, e=0, cst=0):
        return self._lfactory.linear_expr(e, cst)

    def new_var_square(self, var):
        return self.new_quad(quads=(var, var, 1), safe=True)

    def new_var_product(self, var, other):
        # computes and returns the product var * other
        if isinstance(other, Var):
            return self.new_quad(quads=(var, other, 1), safe=True)
        elif isinstance(other, MonomialExpr):
            mnm_dvar = other._dvar
            mnm_coef = other.coef
            return self.new_quad(quads=(var, mnm_dvar, mnm_coef), safe=True)
        elif isinstance(other, ZeroExpr):
            return other
        elif isinstance(other, LinearOperand):
            linear_op = other
            quad_args = [(var, dv, k) for dv, k in linear_op.iter_terms()]
            linexpr_k = linear_op.get_constant()
            quad_linexpr = linexpr_k * var if linexpr_k else None
            return self.new_quad(quad_args, quad_linexpr, safe=True)

        # elif isinstance(other, FunctionalExpr):
        #     return self.new_quad(quads=(var, other.as_var, 1), safe=True)

        else:
            self._unexpected_product_error(var, other)

    def new_monomial_product(self, mnm, other):
        mnmk = mnm.coef
        if not mnmk:  # pragma: no cover
            return self.new_zero_expr()
        else:
            var_quad = self.new_var_product(mnm.var, other)
            var_quad._scale(mnmk)
            return var_quad

    def new_linexpr_product(self, linexpr, other):
        if isinstance(other, Var):
            return self.new_var_product(other, linexpr)

        elif isinstance(other, MonomialExpr):
            return self.new_monomial_product(other, linexpr)

        elif isinstance(other, LinearOperand):
            cst1 = linexpr.constant
            cst2 = other.get_constant()

            fcc = self.term_dict_type()
            for lv1, lk1 in linexpr.iter_terms():
                for lv2, lk2 in other.iter_terms():
                    update_dict_from_item_value(fcc, VarPair(lv1, lv2), lk1 * lk2)
            # this is quad
            qlinexpr = self.new_linear_expr()
            # add cst2 * linexp1
            qlinexpr._add_expr_scaled(expr=linexpr, factor=cst2)
            # add cst1 * linexpr2
            qlinexpr._add_expr_scaled(expr=other, factor=cst1)

            # and that's it
            # fix the constant
            qlinexpr.constant = cst1 * cst2
            quad = QuadExpr(self._model, quads=fcc, linexpr=qlinexpr, safe=True)
            return quad

        else:
            self._unexpected_product_error(linexpr, other)

    def new_le_constraint(self, e, rhs, ctname=None):
        return self._new_qconstraint(e, ComparisonType.LE, rhs, name=ctname)

    def new_eq_constraint(self, e, rhs, ctname=None):
        return self._new_qconstraint(e, ComparisonType.EQ, rhs, name=ctname)

    def new_ge_constraint(self, e, rhs, ctname=None):
        return self._new_qconstraint(e, ComparisonType.GE, rhs, name=ctname)

    def _new_qconstraint(self, lhs, ct_sense, rhs, name=None):
        # noinspection PyPep8
        left_expr = self._to_expr(lhs, context="QuadraticConstraint.left_expr")
        right_expr = self._to_expr(rhs, context="QuadraticConstraint.right_expr")
        self._model._checker.typecheck_two_in_model(self._model, left_expr, right_expr, "new_binary_constraint")
        ct = QuadraticConstraint(self._model, left_expr, ct_sense, right_expr, name)
        left_expr.notify_used(ct)
        right_expr.notify_used(ct)
        return ct

    def _to_expr(self, e, context=None):
        # INTERNAL
        if isinstance(e, (AbstractLinearExpr, QuadExpr)):
            return e
        elif is_number(e):
            return self._lfactory.constant_expr(cst=e)
        else:
            try:
                return e.to_linear_expr()

            except AttributeError:  # pragma: no cover
                self._model.fatal("cannot convert to expression: {0!r}", e)

    def set_quadratic_constraint_expr_from_pos(self, qct, pos, new_expr, update_subscribers=True):
        old_expr = qct.get_expr_from_pos(pos)
        new_operand = self._to_expr(e=new_expr)
        exprs = [qct._left_expr, qct._right_expr]
        exprs[pos] = new_operand
        self._engine.update_constraint(qct, UpdateEvent.LinearConstraintGlobal, *exprs)
        # discard old_expr
        qct.set_expr_from_pos(pos, new_expr=new_operand)
        old_expr.notify_unsubscribed(qct)
        if update_subscribers:
            # -- update  subscribers
            old_expr.notify_unsubscribed(qct)
            new_operand.notify_used(qct)

    def set_quadratic_constraint_sense(self, qct, arg_newsense):
        new_sense = ComparisonType.parse(arg_newsense)
        if new_sense != qct.sense:
            self._engine.update_constraint(qct, UpdateEvent.ConstraintSense, new_sense)
            qct._internal_set_sense(new_sense)

    def update_quadratic_constraint(self, qct, expr, event):
        self._engine.update_constraint(qct, event, expr)

