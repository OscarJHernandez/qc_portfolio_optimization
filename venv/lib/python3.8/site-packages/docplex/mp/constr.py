# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
import warnings

from docplex.mp.basic import IndexableObject, ModelingObjectBase, Priority, _BendersAnnotatedMixin
from docplex.mp.constants import ComparisonType, UpdateEvent
from docplex.mp.operand import LinearOperand
from docplex.mp.sttck import StaticTypeChecker
from docplex.mp.utils import DocplexLinearRelaxationError


class _ExtraConstraintUsage(object):
    # INTERNAL
    def __init__(self, tag):
        self.tag = tag

    def __str__(self):
        return "UsedAs<%s>" % self.tag

    def notify_expr_modified(self, linct, new_expr, engine):
        tagval = self.tag
        engine.update_extra_constraint(linct, tagval, new_expr)


class _ConstraintLogicalUsage(object):
    def __init__(self, logct):
        self._log_ct = logct

    @property
    def tag(self):
        return "logical"

    _cannot_modify_linearct_non_discrete_msg = 'Linear constraint: {0} is used in equivalence, cannot be modified with non-discrete expr: {1}'

    def notify_expr_modified(self, linct, new_expr, engine):
        log_ct = self._log_ct
        if log_ct.is_equivalence() and not new_expr.is_discrete():
            linct.fatal(self._cannot_modify_linearct_non_discrete_msg, linct, new_expr)
        else:
            engine.update_constraint(log_ct, event=UpdateEvent.IndicatorLinearConstraint)


class AbstractConstraint(IndexableObject, _BendersAnnotatedMixin):
    __slots__ = ()

    def __init__(self, model, name=None):
        IndexableObject.__init__(self, model, name)

    @property
    def priority(self):
        return self._model.get_constraint_priority(self)

    @priority.setter
    def priority(self, newprio):
        self.set_priority(newprio)

    def set_priority(self, newprio):
        self._model.set_constraint_priority(self, Priority.parse(newprio, logger=self.error_handler, accept_none=True))

    def set_mandatory(self):
        ''' Sets the constraint as mandatory.

        This prevents relaxation from relaxing this constraint.
        To revert this, set the priority to any non-mandatory priority, or None.
        '''
        self.priority = Priority.MANDATORY

    def is_mandatory(self):
        return Priority.MANDATORY == self.priority

    # noinspection PyUnusedLocal
    def _unsupported_relational_op(self, op_string, other):
        self.fatal("Relational operator: {1} is unavailable with constraint: {0!s}", self, op_string)

    def __le__(self, e):
        self._unsupported_relational_op("<=", e)

    def __ge__(self, e):
        self._unsupported_relational_op(">=", e)

    def __lt__(self, e):
        self._unsupported_relational_op("<", e)

    def __gt__(self, e):
        self._unsupported_relational_op(">", e)

    # py3 needs an eq for hashing
    # def __eq__(self, e):
    #     self._unsupported_relational_op("==", e)

    def _no_linear_ct_in_logical_test_error(self):
        raise TypeError("cannot convert a constraint to boolean: {0!s}".format(self))

    def __nonzero__(self):
        self._no_linear_ct_in_logical_test_error()

    def __bool__(self):
        # python 3 version of nonzero
        self._no_linear_ct_in_logical_test_error()  # pragma: no cover

    def iter_variables(self):
        raise NotImplementedError  # pragma: no cover

    def iter_exprs(self):
        raise NotImplementedError  # pragma: no cover

    def get_var_coef(self, dvar):  # pragma: no cover
        raise NotImplementedError

    @property
    def size(self):
        return sum(x.size for x in self.iter_exprs())

    def copy(self, target_model, var_map):
        raise NotImplementedError  # pragma: no cover

    def relaxed_copy(self, relaxed_model, var_map):
        return self.copy(relaxed_model, var_map)

    def compute_infeasibility(self, slack):  # pragma: no cover
        # INTERNAL: only used when json has no infeasibility info.
        return slack

    # noinspection PyMethodMayBeStatic
    def notify_deleted(self):
        # INTERNAL
        self._set_invalid_index()

    @property
    def short_typename(self):
        return "constraint"

    @property
    def lp_name(self):
        return self._name or "c%s" % (self.index + 1)

    def is_trivial(self):
        return False

    def is_linear(self):
        return False

    def is_quadratic(self):
        return False

    def is_added(self):
        """ Returns True if the constraint has been added to its model.

        Example:
            c = (x+y == 1)
            m.add(c)
            c.is_added()
            >>> True
            c2 = (x + 2*y) >= 3
            c2.is_added()
            >>> False

        """
        return self.index >= 0

    @property
    def cplex_scope(self):
        return self._get_index_scope().cplex_scope

    def _get_index_scope(self):
        raise NotImplementedError

    def is_logical(self):
        return False

    def _get_dual_value(self):
        # INTERNAL
        # Note that dual values are only available for LP problems,
        # so can be calle donly on linear or range constraints.
        return self._model._dual_value1(self)

    def notify_expr_modified(self, expr, event):
        # INTERNAL
        pass  # pragma: no cover

    def notify_expr_replaced(self, old_expr, new_expr):
        # INTERNAL
        pass  # pragma: no cover

    def resolve(self):
        raise NotImplementedError  # pragma: no cover

    def is_satisfied(self, solution, tolerance):
        raise NotImplementedError  # pragma: no cover

    def lock_discrete(self):
        # lock sub expressions
        for expr in self.iter_exprs():
            expr.lock_discrete()


# noinspection PyAbstractClass
class BinaryConstraint(AbstractConstraint):
    __slots__ = ("_ctsense", "_left_expr", "_right_expr")

    def _internal_set_sense(self, new_sense):
        self._ctsense = new_sense

    def __init__(self, model, left_expr, ctsense, right_expr, name=None):
        IndexableObject.__init__(self, model, name)
        self._ctsense = ctsense
        # noinspection PyPep8
        self._left_expr = left_expr
        self._right_expr = right_expr

    def _iter_usages(self):
        return iter([])

    @property
    def type(self):
        """ This property returns the type of the constraint; type is an enumerated value
        of type :class:`ComparisonType`, with three possible values:

        - LE for e1 <= e2 constraints

        - EQ for e1 == e2 constraints

        - GE for e1 >= e2 constraints

        where e1 and e2 denote linear expressions.

        """
        return self._ctsense

    def cplex_code(self):
        return self._ctsense._cplex_code

    def get_left_expr(self):
        """ This property returns the left expression in the constraint.

        Example:
            (X+Y <= Z+1) has left expression (X+Y).
        """
        return self._left_expr

    def get_right_expr(self):
        """ This property returns the right expression in the constraint.

        Example:
            (X+Y <= Z+1) has right expression (Z+1).
        """
        return self._right_expr

    def get_var_coef(self, dvar):
        return self._left_expr.unchecked_get_coef(dvar) - self._right_expr.unchecked_get_coef(dvar)

    def to_string(self):
        """ Returns a string representation of the constraint.

        The operators in this representation are the usual operators <=, ==, and >=.

        Example:
            The constraint (X+Y <= Z+1) is represented as "X+Y <= Z+1".

        Returns:
            A string.

        """
        left_string = self._left_expr.to_string()
        right_string = self._right_expr.to_string()
        self_name = self.name
        if self_name:
            return u"%s: %s %s %s" % (self_name,
                                      left_string,
                                      self._ctsense.operator_symbol,
                                      right_string)
        else:
            return u"%s %s %s" % (left_string,
                                  self._ctsense.operator_symbol,
                                  right_string)

    def cplex_num_rhs(self):
        # INTERNAL
        right_cst = self._right_expr.get_constant()
        left_cst = self._left_expr.get_constant()
        return float(right_cst - left_cst)

    def __repr__(self):
        classname = self.__class__.__name__
        user_name = self.safe_name
        typename = self._ctsense.short_name
        sleft = self._left_expr.truncated_str()
        sright = self._right_expr.truncated_str()
        return "docplex.mp.{0}[{1}]({2!s},{3},{4!s})". \
            format(classname, user_name, sleft, typename, sright)

    def _is_trivially_feasible(self):
        # INTERNAL : assume self is trivial()
        op_func = self._ctsense.python_operator
        return op_func(self._left_expr.get_constant(), self._right_expr.get_constant()) if op_func else False

    def _is_trivially_infeasible(self):
        # INTERNAL: assume self is trivial .
        op_func = self._ctsense.python_operator
        return not op_func(self._left_expr.get_constant(), self._right_expr.get_constant()) if op_func else False

    def is_trivial_feasible(self):
        return self.is_trivial() and self._is_trivially_feasible()

    def is_trivial_infeasible(self):
        return self.is_trivial() and self._is_trivially_infeasible()

    @staticmethod
    def _generate_expr_opposite_linear_coefs(expr):
        for v, k in expr.iter_sorted_terms():
            yield v, -k

    def _iter_net_linear_coefs_sorted(self, left_expr, right_expr):
        # INTERNAL
        if right_expr.is_constant():
            return left_expr.iter_sorted_terms()
        elif left_expr.is_constant():
            return self._generate_expr_opposite_linear_coefs(right_expr)
        else:
            return self._generate_net_linear_coefs2_sorted(left_expr, right_expr)

    def iter_variables(self):
        """  Iterates over all variables mentioned in the constraint.

        *Note:* This includes variables that are mentioned with a zero coefficient. For example,
        the iterator on the following constraint:

         X <= X+Y + 1

        will return X and Y, although X is mentioned with a zero coefficient.

        Returns:
            An iterator object.
        """
        if self._right_expr.is_constant():
            return self._left_expr.iter_variables()
        elif self._left_expr.is_constant():
            return self._right_expr.iter_variables()
        else:
            return self.generate_ordered_vars()

    def generate_ordered_vars(self):
        left_expr = self._left_expr
        for lv in left_expr.iter_variables():
            yield lv
        for rv in self._right_expr.iter_variables():
            if not left_expr.contains_var(rv):
                yield rv

    @staticmethod
    def _generate_net_linear_coefs2_sorted(left_expr, right_expr):
        # INTERNAL
        for lv, lk in left_expr.iter_sorted_terms():
            net_k = lk - right_expr.unchecked_get_coef(lv)
            if net_k:
                yield lv, net_k
        for rv, rk in right_expr.iter_sorted_terms():
            if not left_expr.contains_var(rv) and rk:
                yield rv, -rk

    @staticmethod
    def _generate_net_linear_coefs2_unsorted(left_expr, right_expr):
        # INTERNAL
        for lv, lk in left_expr.iter_terms():
            net_k = lk - right_expr.unchecked_get_coef(lv)
            yield lv, net_k
        for rv, rk in right_expr.iter_terms():
            if not left_expr.contains_var(rv):
                yield rv, -rk

    def notify_deleted(self):
        # INTERNAL
        super(BinaryConstraint, self).notify_deleted()
        self._left_expr.notify_unsubscribed(self)
        self._right_expr.notify_unsubscribed(self)

    def iter_exprs(self):
        return iter([self._left_expr, self._right_expr])

    def get_expr_from_pos(self, pos):
        if 0 == pos:
            return self._left_expr
        elif 1 == pos:
            return self._right_expr
        else:  # pragma: no cover
            self.fatal('Unexpected expression position: {0!r}, expecting 0 or 1', pos)

    def set_expr_from_pos(self, pos, new_expr):
        if 0 == pos:
            self._left_expr = new_expr
        elif 1 == pos:
            self._right_expr = new_expr
        else:  # pragma: no cover
            self.fatal('Unexpected expression position: {0!r}, expecting 0 or 1', pos)

    def is_satisfied(self, solution, tolerance=1e-6):
        left_value = self._left_expr._get_solution_value(solution)
        right_value = self._right_expr._get_solution_value(solution)
        return ComparisonType.almost_compare(left_value, self._ctsense, right_value, eps=tolerance)

    def resolve(self):
        self._left_expr.resolve()
        self._right_expr.resolve()

    def is_discrete(self):
        return self.get_left_expr().is_discrete() and self.get_right_expr().is_discrete()


class LinearConstraint(BinaryConstraint, LinearOperand):
    """ The class that models all constraints of the form `<expr1> <OP> <expr2>`,
            where <expr1> and <expr2> are linear expressions.
    """
    __slots__ = ('_status_var', '_usages')

    def __init__(self, model, left_expr, ctsense, right_expr, name=None):
        BinaryConstraint.__init__(self, model, left_expr, ctsense, right_expr, name)
        left_expr.notify_used(self)
        right_expr.notify_used(self)

    def check_name(self, new_name):
        self.check_lp_name('constraint', new_name, accept_empty=True, accept_none=True)

    def set_name(self, new_name):
        # INTERNAL
        self.check_name(new_name)
        if self.is_added():
            self._model.set_linear_constraint_name(self, new_name)
        else:
            self._set_name(new_name)

    name = property(ModelingObjectBase.get_name, set_name)

    def notify_used_in_logical_ct(self, lct):
        self._add_usage(_ConstraintLogicalUsage(lct))

    def _add_usage(self, usage):
        if hasattr(self, '_usages'):
            self._usages.append(usage)
        else:
            self._usages = [usage]

    def _get_usages(self):
        return getattr(self, '_usages', [])

    def _iter_usages(self):
        # INTERNAL
        return iter(self._get_usages())

    def is_linear(self):
        return True

    @property
    def short_typename(self):
        return "linear constraint"

    # noinspection PyMethodMayBeStatic
    def cplex_range_value(self):
        return 0.0

    def is_lazy_constraint(self):
        return self.model._is_lazy_constraint(self)

    def is_user_cut_constraint(self):
        return self.model._is_user_cut_constraint(self)

    def copy(self, target_model, var_map):
        copied_left = self.left_expr.copy(target_model, var_map)
        copied_right = self.right_expr.copy(target_model, var_map)
        copy_name = None if target_model.ignore_names else self.name
        return self.__class__(target_model, copied_left, self.sense, copied_right, copy_name)

    def relaxed_copy(self, relaxed_model, var_map):
        copied_left = self.left_expr.relaxed_copy(relaxed_model, var_map)
        copied_right = self.right_expr.relaxed_copy(relaxed_model, var_map)
        copy_name = self.name
        return self.__class__(relaxed_model, copied_left, self.sense, copied_right, copy_name)

    @property
    def sense(self):
        """ This property is used to get or set the sense of the constraint; sense is an enumerated value
        of type :class:`ComparisonType`, with three possible values:

        - LE for e1 <= e2 constraints

        - EQ for e1 == e2 constraints

        - GE for e1 >= e2 constraints

        where e1 and e2 denote linear expressions.

        """
        return self._ctsense

    @sense.setter
    def sense(self, new_sense):
        self.set_sense(new_sense)

    def set_sense(self, new_sense):
        self.get_linear_factory().set_linear_constraint_sense(self, new_sense)

    @property
    def sense_string(self):
        return self.sense.name

    # compatibility
    @property
    def type(self):  # pragma: no cover
        warnings.warn(
            "ct.type is deprecated, use ct.sense instead.",
            DeprecationWarning, stacklevel=2)
        return self._ctsense

    @property
    def left_expr(self):
        """ This property returns the left expression in the constraint.

        Example:
            (X+Y <= Z+1) has left expression (X+Y).
        """
        return self._left_expr

    @property
    def right_expr(self):
        """ This property returns the right expression in the constraint.

        Example:
            (X+Y <= Z+1) has right expression (Z+1).
        """
        return self._right_expr

    @right_expr.setter
    def right_expr(self, new_rexpr):
        self.set_right_expr(new_rexpr)

    def set_right_expr(self, new_rexpr):
        self.get_linear_factory().set_linear_constraint_right_expr(ct=self, new_rexpr=new_rexpr)

    @left_expr.setter
    def left_expr(self, new_lexpr):
        self.set_left_expr(new_lexpr)

    def set_left_expr(self, new_lexpr):
        self.get_linear_factory().set_linear_constraint_left_expr(ct=self, new_lexpr=new_lexpr)

    # aliases
    lhs = left_expr
    rhs = right_expr

    def _no_linear_ct_in_logical_test_error(self):
        # if self.sense == ComparisonType.EQ:
        #     # for equality testing there -is- a workaround
        #     msg = "Cannot use == to test expression equality, try using Python is operator or method equals: {0!s}".format(self)
        # else:
        msg = "Cannot convert linear constraint to a boolean value: {0!s}".format(self)
        if self.sense == ComparisonType.EQ:
            # for equality testing there -is- a workaround
            msg += "\n  try using Python 'is' operator or method 'equals' for expression equality"

        raise TypeError(msg)

    def _cannot_promote_from_linear_to_quadratic(self, old_expr, new_expr):
        msg = 'Cannot change linear constraint expr from linear to quadratic'
        if new_expr is None:
            self.fatal('{0}: {1}', msg, old_expr)
        else:
            self.fatal('{0}: was: {1!s}, new: {2!s}', msg, old_expr, new_expr)

    def _check_editable(self, new_expr, engine):
        for usage in self._iter_usages():
            usage.notify_expr_modified(self, new_expr, engine)

    def notify_expr_modified(self, expr, event):
        # INTERNAL
        if event:
            if event is UpdateEvent.LinExprPromotedToQuad:
                self._cannot_promote_from_linear_to_quadratic(old_expr=expr, new_expr=None)
            else:
                self.get_linear_factory().update_linear_constraint_exprs(ct=self, expr_event=event)

    def notify_expr_replaced(self, old_expr, new_expr):
        # INTERNAL
        if new_expr.is_quad_expr() and not old_expr.is_quad_expr():
            self._cannot_promote_from_linear_to_quadratic(old_expr, new_expr)
        if old_expr is self._left_expr:
            self.get_linear_factory().set_linear_constraint_expr_from_pos(lct=self, pos=0, new_expr=new_expr,
                                                                          update_subscribers=False)
        elif old_expr is self._right_expr:
            self.get_linear_factory().set_linear_constraint_expr_from_pos(lct=self, pos=1, new_expr=new_expr,
                                                                          update_subscribers=False)
        else:
            # should not happen
            pass
        # new expr takes al subscribers from old expr
        new_expr.grab_subscribers(old_expr)

    def to_string(self):
        """ Returns a string representation of the constraint.

        The operators in this representation are the usual operators <=, ==, and >=.

        Example:
            The constraint (X+Y <= Z+1) is represented as "X+Y <= Z+1".

        Returns:
            A string.

        """
        return BinaryConstraint.to_string(self)

    def compute_infeasibility(self, slack):  # pragma: no cover
        ctsense = self._ctsense
        if ctsense == ComparisonType.EQ:
            infeas = slack
        elif ComparisonType.LE == ctsense:
            infeas = slack if slack <= 0 else 0
        elif ComparisonType.GE == ctsense:
            infeas = slack if slack >= 0 else 0
        else:
            infeas = 0
        return infeas

    def _get_index_scope(self):
        return self._model._linct_scope

    def is_trivial(self):
        # Checks whether the constraint is equivalent to a comparison between numbers.
        # For example, x <= x+1 is trivial, but 1.5 X <= X + 1 is not.
        def has_nonzero_coef(term_iter):
            return any(tk for _, tk in term_iter)

        self_left_expr = self._left_expr
        self_right_expr = self._right_expr
        if self_right_expr.is_constant():
            return not has_nonzero_coef(self.left_expr.iter_terms())
        elif self_left_expr.is_constant():
            return not has_nonzero_coef(self.right_expr.iter_terms())
        else:
            return not has_nonzero_coef(
                BinaryConstraint._generate_net_linear_coefs2_unsorted(self_left_expr, self_right_expr))

    def _post_meta_constraint(self, rhs, ctsense):
        status_var = self.get_resolved_status_var()
        return self._model._new_xconstraint(lhs=status_var, rhs=rhs, comparaison_type=ctsense)

    def le(self, rhs):
        return self._post_meta_constraint(rhs, ComparisonType.LE)

    def eq(self, rhs):
        return self._post_meta_constraint(rhs, ComparisonType.EQ)

    def ge(self, rhs):
        return self._post_meta_constraint(rhs, ComparisonType.GE)

    def __le__(self, rhs):
        return self.le(rhs)

    def __ge__(self, rhs):
        return self.ge(rhs)

    def __hash__(self):
        return id(self)

    def unchecked_get_coef(self, dvar):
        return 1 if dvar is self._get_status_var() else 0

    def contains_var(self, dvar):
        # only user vars
        return dvar is self._get_status_var()

    def iter_terms(self):
        yield self.get_resolved_status_var(), 1

    iter_sorted_terms = iter_terms

    @property
    def dual_value(self):
        """ This property returns the dual value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
            This method is OK with small numbers of constraints. For large numbers of constraints (>100),
            consider using Model.dual_values() with a sequence of constraints.
            
        See Also:
            `func:docplex.mp.model.Model.dual_values()`
        """
        return self._get_dual_value()

    @property
    def slack_value(self):
        """ This property returns the slack value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
            
                        
            This method is OK with small numbers of constraints. For large numbers of constraints (>100),
            consider using Model.slack_values() with a sequence of constraints.
            
        See Also:
            `func:docplex.mp.model.Model.slack_values()`
        """
        return self._model._slack_value1(self)

    @property
    def basis_status(self):
        """ This property returns the basis status of the slack variable of the constraint, if any.

        Returns:
            An enumerated value from the enumerated type `docplex.constants.BasisStatus`.

        Note:
            for the model to hold basis information, the model must have been solved as a LP problem.
            In some cases, a model which failed to solve may still have a basis available. Use
            `Model.has_basis()` to check whether the model has basis information or not.

        See Also:
            :func:`docplex.mp.model.Model.has_basis`
            :class:`docplex.mp.constants.BasisStatus`

        *New in version 2.10*

        """
        return self._model._linearct_basis_status([self])[0]

    def iter_net_linear_coefs(self):
        # INTERNAL
        left_expr = self._left_expr
        right_expr = self._right_expr
        if right_expr.is_constant():
            return left_expr.iter_sorted_terms()
        elif left_expr.is_constant():
            return self._generate_expr_opposite_linear_coefs(right_expr)
        else:
            return self._generate_net_linear_coefs2_sorted(left_expr, right_expr)

    def _get_status_var(self):
        # this call does -not- create the variable. Returns None if not present.
        return getattr(self, '_status_var', None)

    @property
    def status_var(self):
        return self.get_resolved_status_var()

    as_var = status_var

    def as_logical_operand(self):
        if not self.is_discrete():
            return None
        else:
            return self.get_resolved_status_var(caller_msg=
                                                'Conversion to logical operand is available only for discrete constraints')

    def _check_is_discrete(self, ct, msg=None):
        err_msg = msg or "Conversion from constraint to expression is available only for discrete constraints"
        StaticTypeChecker.typecheck_discrete_constraint(self, ct, msg=err_msg)

    def get_resolved_status_var(self, caller_msg=None):
        status_var = self._get_status_var()  # always use the getter!
        if status_var is not None:
            return status_var

        self._model._check_logical_constraint_support()

        # TODO: issue a meaningful message on why the ct is not discrete
        self._check_is_discrete(self, msg=caller_msg)
        # lock it discrete
        self.lock_discrete()

        # lazy allocation of a new status variable...:
        lfactory = self.get_linear_factory()
        status_var = lfactory.new_constraint_status_var(self)
        if self.is_added():
            # status variable is bound
            status_var.lb = 1
        self._status_var = status_var
        # store ct in model
        eqct = lfactory.new_equivalence_constraint(status_var, linear_ct=self)
        eqct.origin = self
        engine = self._model.get_engine()
        eqx = engine.create_logical_constraint(eqct, is_equivalence=True)
        self._model._register_implicit_equivalence_ct(eqct, eqx)
        return status_var

    def to_linear_expr(self):
        return self.status_var

    def notify_deleted(self):
        super(LinearConstraint, self).notify_deleted()
        svar = self._get_status_var()  # possibly not resolved
        if svar:
            svar.lb = 0

    @property
    def benders_annotation(self):
        """
        This property is used to get or set the Benders annotation of a constraint.
        The value of the annotation must be a positive integer

        """
        return self.get_benders_annotation()

    @benders_annotation.setter
    def benders_annotation(self, new_anno):
        self.set_benders_annotation(new_anno)

    # -- arithmetic operators
    def times(self, e):
        # TODO: dow e limit numbers here, otherwise non-convex errors may creep in
        return self.to_linear_expr().__mul__(e)

    def __mul__(self, e):
        return self.times(e)

    def __rmul__(self, e):
        return self.times(e)

    def __div__(self, e):
        return self.quotient(e)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.quotient(e)  # pragma: no cover

    def quotient(self, e):
        svar = self.get_resolved_status_var()
        self._model._typecheck_as_denominator(e, svar)
        inverse = 1.0 / float(e)
        return self.times(inverse)

    def __add__(self, e):
        return self.to_linear_expr().__add__(e)

    def __radd__(self, e):
        return self.to_linear_expr().__add__(e)

    def __sub__(self, e):
        return self.to_linear_expr().__sub__(e)

    def __rsub__(self, e):
        return self.to_linear_expr().__rsub__(e)

    def __or__(self, other):
        return self.logical_or(other)

    def __and__(self, other):
        return self.logical_and(other)

    def _check_logical_operator(self, other, caller=None):
        self._check_is_discrete(self, msg="Logical operators require discrete constraints")
        StaticTypeChecker.typecheck_logical_op(self, other, caller=caller)

    def logical_or(self, other):
        self._check_logical_operator(other, "LinearConstraint.or")
        return self.get_linear_factory().new_binary_constraint_or(self, other)

    def logical_and(self, other):
        self._check_logical_operator(other, "LinearConstraint.and")
        return self.get_linear_factory().new_binary_constraint_and(self, other)

    def __rshift__(self, other):
        """ Redefines the right-shift operator to define if-then constraints.

            This operator allows to create if-then constraint with the `>>` operator.
            It expects a linear constraint as second argument.

            :param other: the linear constraint used to build a new if-then constraint.

            :return:
                an instance of IfThenConstraint, that is not added to the model.
                Use `Model.add()` to add it to the model.

            Note:
                The constraint must be discrete, otherwise an exception is raised.

            Example:

                m.add((x >= 3) >> (y == 5))

                creates  an if-then constraint which links the satisfaction of constraint (x >= 3) to the satisfaction
                of (y ==5).

            """
        return self._model.if_then(if_ct=self, then_ct=other)

    def _tag_as_extra_ct(self, qualifier):
        # INTERNAL
        self._add_usage(_ExtraConstraintUsage(qualifier))

    def _untag_as_extra_ct(self, qualifier):
        usages = self._get_usages()
        upos = -1
        for u, used in enumerate(usages):
            if used.tag == qualifier:
                upos = u
                break
        if upos >= 0:
            del usages[upos]

    _user_cut_tag = "user-cut constraint"
    _lazy_constraint_tag = "lazy constraint"

    def notify_used_as_user_cut(self):
        # INTERNAL
        self._tag_as_extra_ct(self._user_cut_tag)

    def notify_used_as_lazy_constraint(self):
        # INTERNAL
        self._tag_as_extra_ct(self._lazy_constraint_tag)

    def notify_unused_as_user_cut(self):
        # INTERNAL
        self._untag_as_extra_ct(self._user_cut_tag)

    def notify_unused_as_lazy_constraint(self):
        # INTERNAL
        self._untag_as_extra_ct(self._lazy_constraint_tag)


class RangeConstraint(AbstractConstraint):
    """ This class models range constraints.

    A range constraint states that an expression must stay between two
    values, `lb` and `ub`.

    This class is not meant to be instantiated by the user.
    To create a range constraint, use the factory method :func:`docplex.mp.model.Model.add_range`
    defined on :class:`docplex.mp.model.Model`.

    """

    def __init__(self, model, expr, lb, ub, name=None):
        AbstractConstraint.__init__(self, model, name)
        self._ub = ub
        self._lb = lb
        self._expr = expr

    def is_linear(self):
        return True

    # noinspection PyMethodMayBeStatic
    def cplex_code(self):
        return 'R'

    def _get_index_scope(self):
        return self._model._linct_scope

    @property
    def short_typename(self):
        return "range constraint"

    def is_trivial(self):
        return self._expr.is_constant()

    def _is_trivially_feasible(self):
        # INTERNAL : assume self is trivial()
        expr_num = self._expr.constant
        return self._lb <= expr_num <= self._ub

    def _is_trivially_infeasible(self):
        # INTERNAL : assume self is trivial()
        expr_num = self._expr.constant
        return expr_num < self._lb or expr_num > self._ub

    def compute_infeasibility(self, slack):  # pragma: no cover
        # compatible with cplex...
        return -slack

    def get_var_coef(self, dvar):
        return self._expr.unchecked_get_coef(dvar)

    def is_satisfied(self, solution, tolerance=1e-6):
        expr_value = self._expr._get_solution_value(solution)
        return self._lb - tolerance <= expr_value <= self._ub + tolerance

    @property
    def expr(self):
        """ This property returns the linear expression of the range constraint.
        """
        return self._expr

    @expr.setter
    def expr(self, new_expr):
        self.get_linear_factory().set_range_constraint_expr(self, new_expr)

    @property
    def lb(self):
        """ This property is used to get or set the lower bound of the range constraint.

        """
        return self._lb

    @lb.setter
    def lb(self, new_lb):
        self._model._typecheck_num(new_lb)
        self.get_linear_factory().set_range_constraint_lb(self, new_lb)

    @property
    def ub(self):
        """ This property is used to get or set the upper bound of the range constraint.

        """
        return self._ub

    @ub.setter
    def ub(self, new_ub):
        self._model._typecheck_num(new_ub)
        self.get_linear_factory().set_range_constraint_ub(self, new_ub)

    @property
    def bounds(self):
        """ This property is used to get or set the (lower, upper) bounds of a range constraint.

        """
        return self._lb, self._ub

    @bounds.setter
    def bounds(self, new_bounds):
        try:
            new_lb, new_ub = new_bounds
            self._model._typecheck_num(new_lb)
            self._model._typecheck_num(new_ub)
            self.get_linear_factory().set_range_constraint_bounds(self, new_lb, new_ub)
        except ValueError:
            self.fatal('RangeConstraint.bounds expects a 2-tuple of numbers, {0!r} was passed', new_bounds)

    def _internal_set_lb(self, new_lb):
        self._lb = new_lb

    def _internal_set_ub(self, new_ub):
        self._ub = new_ub

    def is_feasible(self):
        return self._ub >= self._lb

    @property
    def dual_value(self):
        """ This property returns the dual value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
        """
        return self._get_dual_value()

    @property
    def slack_value(self):
        """ This property returns the slack value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
        """
        return self._model._slack_value1(self)

    @property
    def basis_status(self):
        """ This property returns the basis status of the slack variable of the constraint, if any.

        Returns:
            An enumerated value from the enumerated type `docplex.constants.BasisStatus`.

        Note:
            for the model to hold basis information, the model must have been solved as a LP problem.
            In some cases, a model which failed to solve may still have a basis available. Use
            `Model.has_basis()` to check whether the model has basis information or not.

        See Also:
            :func:`docplex.mp.model.Model.has_basis`
            :class:`docplex.mp.constants.BasisStatus`

        *New in version 2.10*
        """
        return self._model._linearct_basis_status([self])[0]

    def iter_variables(self):
        """Iterates over all the variables of the range constraint.

        Returns:
           An iterator object.
        """
        return self._expr.iter_variables()

    def iter_exprs(self):
        yield self._expr

    def cplex_range_value(self, do_raise=True):
        return self.static_cplex_range_value(self, self._lb, self._ub,
                                             lambda: "Range has infeasible domain: {0!s}".format(self),
                                             do_raise=do_raise)

    @classmethod
    def static_cplex_range_value(cls, logger, lbval, ubval, msg_fun, do_raise=True):
        rangeval = float(lbval - ubval)
        # this should be negative, otherwise fails....
        # no way to model infeasible ranges with cplex rngval.
        if rangeval >= 1e-6 and do_raise:
            logger.fatal(msg_fun())
        return rangeval

    def cplex_num_rhs(self):
        # force conversion to float for numpy, etc...
        return float(self._ub - self._expr.get_constant())

    def get_left_expr(self):
        return self._expr

    # noinspection PyMethodMayBeStatic
    def get_right_expr(self):
        return None

    def copy(self, target_model, var_map):
        copied_expr = self.expr.copy(target_model, var_map)
        copy_name = None if target_model.ignore_names else self.name
        copied_range = RangeConstraint(target_model, copied_expr, self.lb, self.ub, copy_name)
        return copied_range

    def to_string(self):
        np = self.model._num_printer
        return "{0} <= {1!s} <= {2}".format(np.to_string(self._lb), self._expr, np.to_string(self._ub))

    def __str__(self):
        """ Returns a string representation of the range constraint.

        Example:
            1 <= x+y+z <= 3 represents the range constraint where the expression (x+y+z) is
            constrained to stay between 1 and 3.

        Returns:
            A string.
        """
        return self.to_string()

    def __repr__(self):
        printable_name = self.safe_name
        return "docplex.mp.RangeConstraint[{0}]({1},{2!s},{3})". \
            format(printable_name, self.lb, self._expr, self.ub)

    def resolve(self):
        self._expr.resolve()

    @property
    def benders_annotation(self):
        """
        This property is used to get or set the Benders annotation of a constraint.
        The value of the annotation must be a positive integer

        """
        return self.get_benders_annotation()

    @benders_annotation.setter
    def benders_annotation(self, new_anno):
        self.set_benders_annotation(new_anno)


class NotEqualConstraint(LinearConstraint):

    def __init__(self, model, negated_eqct, name=None):
        # assume negated_eqct is the equality constraint we want to negate
        self._negated_ct = negated_eqct
        aux_eq_ct_status = self._negated_ct.get_resolved_status_var()
        zero = model._lfactory.new_zero_expr()
        LinearConstraint.__init__(self, model, aux_eq_ct_status, ComparisonType.EQ, zero, name)
        self.lock_discrete()

    def to_string(self):
        eqct = self._negated_ct
        return "{0} != {1}".format(eqct._left_expr, eqct._right_expr)

    def set_sense(self, new_sense):
        self.fatal("cannot modify sense of a not_equal constraint: {0!s}", self)

    @property
    def negated_constraint(self):
        return self._negated_ct

    def __str__(self):
        """ Returns a string representation of the not equals constraint.

        Returns:
            A string.
        """
        return self.to_string()

    def __repr__(self):
        return "docplex.mp.NotEquals({0}, {1})". \
            format(self._left_expr, self._right_expr)


class LogicalConstraint(AbstractConstraint):
    """ This class models logical constraints.

    An equivalence constraint links (both ways) the value of a binary variable
    to the satisfaction of a linear constraint.
    
    If the binary variable equals the truth value (default is 1),
    then the constraint is satisfied, conversely if the constraint
    is satisfied, the value of the variable is set to the truth value.

    This class is not meant to be instantiated by the user.

    """
    __slots__ = ('_binary_var', '_linear_ct', '_active_value')

    def __init__(self, model, binary_var, linear_ct, active_value=1, name=None):
        AbstractConstraint.__init__(self, model, name)
        self._binary_var = binary_var
        self._linear_ct = linear_ct
        self._active_value = active_value
        # connect exprs
        for expr in linear_ct.iter_exprs():
            expr.notify_used(self)

        linear_ct.notify_used_in_logical_ct(self)

    def resolve(self):
        self._linear_ct.resolve()

    def _get_index_scope(self):
        return self._model._logical_scope

    def iter_exprs(self):
        return self._linear_ct.iter_exprs()

    def is_equivalence(self):
        raise NotImplementedError  # pragma: no cover

    def cplex_num_rhs(self):
        return self._linear_ct.cplex_num_rhs()

    @property
    def active_value(self):
        return self._active_value

    @property
    def binary_var(self):
        return self._binary_var

    @property
    def linear_constraint(self):
        return self._linear_ct

    @property
    def benders_annotation(self):
        """
        This property is used to get or set the Benders annotation of a constraint.
        The value of the annotation must be a positive integer

        """
        return self.get_benders_annotation()

    @benders_annotation.setter
    def benders_annotation(self, new_anno):
        self.set_benders_annotation(new_anno)

    def get_linear_constraint(self):
        return self._linear_ct

    @property
    def cpx_complemented(self):
        return 1 - self._active_value

    def copy(self, target_model, var_map):
        copied_binary = var_map[self.binary_var]
        copied_linear_ct = self.linear_constraint.copy(target_model, var_map)
        copy_name = None if target_model.ignore_names else self.name
        copied_equiv = self.__class__(target_model,
                                      copied_binary,
                                      copied_linear_ct,
                                      self._active_value,
                                      copy_name)
        return copied_equiv

    def relaxed_copy(self, relaxed_model, var_map):
        raise DocplexLinearRelaxationError(self, cause='logical')

    def is_logical(self):
        return True

    def iter_variables(self):
        yield self._binary_var
        for v in self._linear_ct.iter_variables():
            yield v

    def get_var_coef(self, dvar):
        if dvar is self._binary_var:
            return 1
        else:
            return self._linear_ct.get_var_coef(dvar)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        printable_name = self.safe_name
        clazzname = self.__class__.__name__
        return "docplex.mp.constr.{0:s}[{1}]({2!s},{3!s},true={4})" \
            .format(clazzname, printable_name, self._binary_var, self._linear_ct, self._active_value)

    def notify_expr_modified(self, expr, event):
        # INTERNAL
        self.get_linear_factory().update_indicator_constraint_expr(self, event, expr)

    @property
    def slack_value(self):
        return self._model._slack_value1(self)

    def _symbol(self):
        return '<->' if self.is_equivalence() else '->'

    def to_string(self):
        """
        Displays the equivalence constraint in a (shortened) LP style:
        z <-> x+y+z == 2

        Returns:
            A string.
        """
        eqname = self.get_name()
        name_part = '{0}: '.format(eqname) if eqname else ''
        eq_var = self._binary_var
        if eq_var.is_generated():
            varname = 'x%d' % eq_var._index
        else:
            varname = str(eq_var)
        s_active_value = '' if self._active_value else '=0'
        s_symbol = self._symbol()
        return "{0}{1}{2} {4} [{3!s}]".format(name_part, varname, s_active_value, self.linear_constraint, s_symbol)


class IndicatorConstraint(LogicalConstraint):
    """ This class models indicator constraints.

    An indicator constraint links (one-way) the value of a binary variable to the satisfaction of a linear constraint.
    If the binary variable equals the active value, then the constraint is satisfied, but otherwise the constraint
    may or may not be satisfied.

    This class is not meant to be instantiated by the user.

    To create an indicator constraint, use the factory method :func:`docplex.mp.model.Model.add_indicator`
    defined on :class:`docplex.mp.model.Model`.

    """
    __slots__ = ()

    def __init__(self, model, binary_var, linear_ct, active_value=1, name=None):
        LogicalConstraint.__init__(self, model, binary_var, linear_ct, active_value, name)

    @property
    def short_typename(self):
        return "indicator constraint"

    def is_equivalence(self):
        return False

    def is_logical(self):
        return True

    def invalidate(self):
        """
        Sets the binary variable to the opposite of its active value.
        Typically used by indicator constraints with a trivial infeasible linear part.
        For example, z=1 -> 4 <= 3 sets z to 0 and
        z=0 -> 4 <= 3 sets z to 1.
        This is equivalent to if z=a => False, then z *cannot* be equal to a.
        """
        if 0 == self.active_value:
            # set to 1 : lb = 1
            self.binary_var.lb = 1
        elif 1 == self.active_value:
            # set to 0 ub = 0
            self.binary_var.ub = 0
        else:
            self.fatal("Unexpected active value for indicator constraint: {0!s}, value is: {1!s}, expecting 0 or 1",
                       # pragma: no cover
                       self, self.active_value)  # pragma: no cover

    def is_satisfied(self, solution, tolerance=1e-6):
        binary_value = solution.get_value(self._binary_var)
        if abs(binary_value - self._active_value) <= tolerance:
            is_ct_satisfied = self._linear_ct.is_satisfied(solution, tolerance)
            # only active if binary is active value:
            return abs(1 - is_ct_satisfied) <= tolerance
        else:
            # when binary var is not equal to active_value, indicator has no effect
            return True


class EquivalenceConstraint(LogicalConstraint):
    """ This class models equivalence constraints.

    An equivalence constraint links (both ways) the value of a binary variable
    to the satisfaction of a linear constraint.

    If the binary variable equals the truth value (default is 1),
    then the constraint is satisfied, conversely if the constraint
    is satisfied, the value of the variable is set to the truth value.

    This class is not meant to be instantiated by the user.

    """
    __slots__ = ()

    def __init__(self, model, binary_var, linear_ct, truth_value=1, name=None):
        LogicalConstraint.__init__(self, model, binary_var, linear_ct, truth_value, name)
        # connect exprs
        linear_ct.lock_discrete()

    @property
    def short_typename(self):
        return "equivalence constraint"

    def is_equivalence(self):
        return True

    def is_logical(self):
        return True

    def is_satisfied(self, solution, tolerance=1e-6):
        is_ct_satisfied = self._linear_ct.is_satisfied(solution, tolerance)
        bvar = self._binary_var
        if bvar.is_generated() and not bvar in solution:
            # bvar is not mentioned, ok
            return True
        binary_value = solution.get_value(bvar)
        expected_value = self._active_value if is_ct_satisfied else 1 - self._active_value
        return ComparisonType.almost_equal(binary_value, expected_value, tolerance)


class IfThenConstraint(IndicatorConstraint):

    def __init__(self, model, if_ct, then_ct, negate=False):
        if_ct_status_var = if_ct.status_var
        self._if_ct = if_ct
        # if negated, then the then constraint is satisfied if not(if_ct) is satisfied
        # this is actually an if-else ...
        true_value = 0 if negate else 1
        IndicatorConstraint.__init__(self, model, binary_var=if_ct_status_var,
                                     linear_ct=then_ct, active_value=true_value, name=None)

    def to_string(self):
        return "{0!s} -> {1!s}".format(self._if_ct, self.linear_constraint)


class QuadraticConstraint(BinaryConstraint):
    """ The class models quadratic constraints.

        Quadratic constraints are of the form `<qexpr1> <OP> <qexpr2>`,
        where at least one of <qexpr1> or <qexpr2> is a quadratic expression.

    """

    def copy(self, target_model, var_map):
        # noinspection PyPep8
        copied_left_expr = self.left_expr.copy(target_model, var_map)
        copied_right_expr = self.right_expr.copy(target_model, var_map)
        copy_name = None if target_model.ignore_names else self.name
        return QuadraticConstraint(target_model, copied_left_expr, self.type, copied_right_expr, copy_name)

    def relaxed_copy(self, relaxed_model, var_map):
        raise DocplexLinearRelaxationError(self, cause='quadratic')

    def is_quadratic(self):
        return True

    @property
    def short_typename(self):
        return "quadratic constraint"

    def _get_index_scope(self):
        return self._model._quadct_scope

    __slots__ = ()

    def is_trivial(self):
        for _, nqk in self.iter_net_quads():
            if nqk:
                return False
        # now check linear parts

        for _, lk in self.iter_net_linear_coefs():
            if lk:
                return False
        return True

    def iter_net_linear_coefs(self):
        linear_left = self._left_expr.get_linear_part()
        linear_right = self._right_expr.get_linear_part()
        return self._iter_net_linear_coefs_sorted(linear_left, linear_right)

    def iter_net_quads(self):
        # INTERNAL
        left_expr = self._left_expr
        right_expr = self._right_expr
        if not right_expr.is_quad_expr():
            return left_expr.iter_sorted_quads()
        elif not left_expr.is_quad_expr():
            return right_expr.iter_opposite_ordered_quads()
        else:
            return self.generate_ordered_net_quads(left_expr, right_expr)

    @classmethod
    def generate_ordered_net_quads(cls, qleft, qright):
        # left first, then right
        for lqv, lqk in qleft.iter_sorted_quads():
            net_k = lqk - qright._get_quadratic_coefficient_from_var_pair(lqv)
            if 0 != net_k:
                yield lqv, net_k
        for rqv, rqk in qright.iter_sorted_quads():
            if not qleft.contains_quad(rqv) and 0 != rqk:
                yield rqv, -rqk

    def _set_left_expr(self, new_left_expr):
        self.qfactory.set_quadratic_constraint_expr_from_pos(self, pos=0, new_expr=new_left_expr)

    @property
    def left_expr(self):
        return BinaryConstraint.get_left_expr(self)

    @left_expr.setter
    def left_expr(self, new_left_expr):
        self._set_left_expr(new_left_expr)

    def _set_right_expr(self, new_right_expr):
        self.qfactory.set_quadratic_constraint_expr_from_pos(self, pos=1, new_expr=new_right_expr)

    @property
    def right_expr(self):
        return BinaryConstraint.get_right_expr(self)

    @right_expr.setter
    def right_expr(self, new_right_expr):
        self._set_right_expr(new_right_expr)

    # aliases
    rhs = right_expr
    lhs = left_expr

    @property
    def benders_annotation(self):
        """
        This property is used to get or set the Benders annotation of a constraint.
        The value of the annotation must be a positive integer

        """
        return self.get_benders_annotation()

    @benders_annotation.setter
    def benders_annotation(self, new_anno):
        self.set_benders_annotation(new_anno)

    @property
    def slack_value(self):
        """ This property returns the slack value of the constraint.

        Note:
            This method will raise an exception if the model has not been solved successfully.
        """
        return self._model._slack_value1(self)

    @property
    def sense(self):
        """ This property is used to get or set the sense of the constraint; sense is an enumerated value
        of type :class:`ComparisonType`, with three possible values:

        - LE for e1 <= e2 constraints

        - EQ for e1 == e2 constraints

        - GE for e1 >= e2 constraints

        where e1 and e2 denote quadratic expressions.

        """
        return self._ctsense

    @sense.setter
    def sense(self, new_sense):
        self.set_sense(new_sense)

    def set_sense(self, new_sense):
        self.qfactory.set_quadratic_constraint_sense(self, new_sense)

    # compat
    @property
    def type(self):
        return self._ctsense

    def has_net_quadratic_term(self):
        # INTERNAL
        return any(nk for _, nk in self.iter_net_quads())

    def notify_expr_modified(self, expr, event):
        # INTERNAL
        self.qfactory.update_quadratic_constraint(self, expr, event)

    def notify_expr_replaced(self, old_expr, new_expr):
        qfact = self.qfactory
        if old_expr is self._left_expr:
            qfact.set_quadratic_constraint_expr_from_pos(qct=self, pos=0, new_expr=new_expr,
                                                         supdate_subscribers=False)
        elif old_expr is self._right_expr:
            qfact.set_quadratic_constraint_expr_from_pos(qct=self, pos=1, new_expr=new_expr,
                                                         update_subscribers=False)
        else:
            # should not happen
            pass
        # new expr takes al subscribers from old expr
        new_expr.grab_subscribers(old_expr)


class PwlConstraint(AbstractConstraint):
    """ This class models piecewise linear constraints.

    This class is not meant to be instantiated by the user.
    To create a piecewise constraint, use the factory method :func:`docplex.mp.model.Model.piecewise`
    defined on :class:`docplex.mp.model.Model`.

    """

    __slots__ = ('_pwl_expr', '_input_var', '_y')

    def __init__(self, model, pwl_expr, name=None):
        AbstractConstraint.__init__(self, model, name)
        self._pwl_expr = pwl_expr
        self._input_var = pwl_expr._x_var
        self._y = None

    def resolve(self):
        self._pwl_expr.resolve()

    def is_satisfied(self, solution, tolerance):
        expr_value = self._input_var._get_solution_value(solution)
        y_value = solution._get_var_value(self._y)
        computed_f_expr_value = self.pwl_func.evaluate(expr_value)
        return ComparisonType.almost_equal(y_value, computed_f_expr_value, tolerance)

    @property
    def expr(self):
        """ This property returns the linear expression of the piecewise linear constraint.
        """
        return self._input_var

    @property
    def pwl_func(self):
        """ This property returns the piecewise linear function of the piecewise linear constraint.
        """
        return self._pwl_expr.pwl_func

    @property
    def y(self):
        """ This property returns the output variable associated with the piecewise linear constraint.
        """
        if self._y is None:
            self._y = self._pwl_expr._get_allocated_f_var()
        return self._y

    @property
    def usage_counter(self):
        return self._pwl_expr.usage_counter

    def _get_index_scope(self):
        return self._model._pwl_scope

    def iter_exprs(self):
        yield self._pwl_expr

    def iter_variables(self):
        """Iterates over all the variables of the piecewise linear constraint.

        Returns:
           An iterator object.
        """
        yield self.y
        yield self._input_var

    def iter_extended_variables(self):
        # iterates on all extended variables involved in the computation
        # yvar, xavr, plus all argument rexpr vars
        # if argument var is identical to the input var, it is returned twice...?
        yield self.y
        yield self._input_var
        for v in self._pwl_expr._argument_expr.iter_variables():
            yield v

    def get_var_coef(self, dvar):
        if dvar is self.y:
            return 1
        else:
            return self.expr.unchecked_get_coef(dvar)

    def copy(self, target_model, var_map):
        # Internal: copy must not be invoked on PwlConstraint.
        raise NotImplementedError  # pragma: no cover

    def relaxed_copy(self, relaxed_model, var_map):
        raise DocplexLinearRelaxationError(self, cause='pwl')

    def to_string(self):
        pwlf = self.pwl_func
        pwlf_s = pwlf.name or 'pwl?'
        return "{0} == {1!s}({2!s})".format(self.y, pwlf_s, self.expr)

    def __str__(self):
        """ Returns a string representation of the piecewise linear constraint.

        Example:
            `y == pwl_name(x + z)` represents the piecewise linear constraint where the variable `y` is
            constrained to be equal to the value of the piecewise linear function whose name is 'pwl_name'
            applied to the expression (x + z).

        Returns:
            A string.
        """
        return self.to_string()

    def __repr__(self):
        printable_name = self.safe_name
        return "docplex.mp.PwlConstraint[{0}]({1},{2!s},{3})". \
            format(printable_name, self.y, self.pwl_func, self.expr)
