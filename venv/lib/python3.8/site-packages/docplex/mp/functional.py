# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore

from docplex.mp.basic import Expr
from docplex.mp.constants import SOSType
from docplex.mp.operand import LinearOperand

from docplex.mp.utils import is_iterable, is_iterator, DocplexLinearRelaxationError

# do NOT import Model -> circular

# change this flag to generate named objects
# by default all generated objects will have no name
use_debug_names = False


def get_name_if_debug(name):
    return name if use_debug_names else None


# noinspection PyAbstractClass
class _FunctionalExpr(Expr, LinearOperand):
    # INTERNAL class
    # parent class for all nonlinear expressions.
    __slots__ = ('_f_var', '_resolved')

    def __init__(self, model, name=None):
        Expr.__init__(self, model, name)
        self._f_var = None
        self._resolved = False

    def to_linear_expr(self):
        return self._get_resolved_f_var()

    def iter_terms(self):
        yield self._get_resolved_f_var(), 1

    iter_sorted_terms = iter_terms

    def iter_variables(self):
        # do we need to create it here?
        yield self._get_resolved_f_var()

    def unchecked_get_coef(self, dvar):
        return 1 if dvar is self._f_var else 0

    def _new_generated_free_continuous_var(self, artefact_pos, name=None):
        # INTERNAL
        inf = self._model.infinity
        return self._new_generated_continuous_var(artefact_pos, lb=-inf, ub=inf, name=name)

    def _new_generated_continuous_var(self, artefact_pos, lb=None, ub=None, name=None):
        return self._new_generated_var(artefact_pos, vartype=self._model.continuous_vartype, lb=lb, ub=ub, name=name)

    def _new_generated_binary_var(self, artefact_pos, name=None):
        return self._new_generated_var(artefact_pos, self._model.binary_vartype, name=name)

    def _new_generated_var(self, artefact_pos, vartype, lb=None, ub=None, name=None):
        # INTERNAL
        assert artefact_pos >= 0
        m = self._model
        gvar = m._lfactory.new_var(vartype, lb=lb, ub=ub, varname=name, safe=True)
        gvar.origin = (self, artefact_pos)
        return gvar

    def _new_generated_binary_varlist(self, keys, offset=0, name=None):
        bvars = self.model.binary_var_list(keys, name)
        for b, bv in enumerate(bvars, start=offset):
            bv.origin = (self, b)
        return bvars

    def new_generated_sos1(self, dvars):
        sos1 = self.model._add_sos(dvars, SOSType.SOS1)
        sos1.origin = self
        return sos1

    def _new_generated_indicator(self, binary_var, linear_ct, active_value=1, name=None):
        ind = self._model._lfactory.new_indicator_constraint(binary_var, linear_ct, active_value, name)
        ind.origin = self
        self._model.add(ind)
        return ind

    def _new_generated_binary_ct(self, lhs, rhs, sense='EQ'):
        # posts a constraint and marks it as generated.
        m = self._model
        ct = m._lfactory.new_binary_constraint(lhs=lhs, sense=sense, rhs=rhs)
        m._post_constraint(ct)
        ct.origin = self
        return ct

    def _post_generated_cts(self, cts):
        # takes a sequence of constraints
        # posts a constraint and marks it as generated.
        self._model._lfactory._post_constraint_block(cts)
        for c in cts:
            c.origin =self
        return cts

    def _get_resolved_f_var(self):
        self._ensure_resolved()
        return self._f_var

    def _get_allocated_f_var(self):
        if self._f_var is None:
            self._f_var = self._create_functional_var()
        return self._f_var

    def resolve(self):
        self._ensure_resolved()

    def _ensure_resolved(self):
        if self._f_var is None:
            # 1. create the var (once!)
            self._f_var = self._create_functional_var()
            # 2. post the link between the fvar and the argument expr
        if not self._resolved:
            self._resolve()
            self._resolved = True

    def _is_resolved(self):
        return self._resolved and self._f_var is not None

    def _name_functional_var_name(self, fvar, fvar_meta_format="_%s%d"):
        fname = fvar_meta_format % (self.function_symbol, fvar._index)
        fvar.set_name(fname)

    def _create_functional_var(self, named=True):
        fvar = self._new_generated_free_continuous_var(artefact_pos=0, name=None)
        if named:
            self._name_functional_var_name(fvar)
        return fvar

    @property
    def functional_var(self):
        return self._get_resolved_f_var()

    as_var = functional_var

    def get_artefact(self, pos):
        assert pos == 0
        return self.as_var

    def square(self):
        return self.functional_var.square()

    def _resolve(self):
        raise NotImplementedError  # pragma: no cover

    def _get_function_symbol(self):
        # redefine this to get the function symbol
        raise NotImplementedError  # pragma: no cover

    @property
    def function_symbol(self):
        return self._get_function_symbol()

    def relaxed_copy(self, relaxed_model, var_map):
        raise DocplexLinearRelaxationError(self, cause=self.function_symbol)

    def __str__(self):
        return self.to_string()

    def to_string(self, **kwargs):
        raise NotImplementedError  # pragma: no cover

    # -- arithmetic operators
    def __mul__(self, e):
        return self.functional_var.__mul__(e)

    def __rmul__(self, e):
        return self.functional_var.__mul__(e)

    def __div__(self, e):
        return self.divide(e)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.divide(e)  # pragma: no cover

    def divide(self, e):
        return self.functional_var.divide(e)

    def __add__(self, e):
        return self.functional_var.__add__(e)

    def __radd__(self, e):
        return self.functional_var.__add__(e)

    def __sub__(self, e):
        return self.functional_var.__sub__(e)

    def __rsub__(self, e):
        return self.functional_var.__rsub__(e)

    def __neg__(self):
        # the "-e" unary minus returns a linear expression
        return self.functional_var.__neg__()

    def _allocate_arg_var_if_necessary(self, arg_expr, pos):
        # INTERNAL
        # allocates a new variables if only the argument expr is not a variable
        # and returns it
        try:
            arg_var = arg_expr.as_variable()
        except AttributeError:
            arg_var = None

        if arg_var is None:
            arg_var = self._new_generated_free_continuous_var(artefact_pos=pos)
            self._new_generated_binary_ct(arg_var, arg_expr)
        return arg_var


# noinspection PyAbstractClass
class UnaryFunctionalExpr(_FunctionalExpr):
    def __init__(self, model, argument_expr, name=None):
        _FunctionalExpr.__init__(self, model, name)
        self._argument_expr = model._lfactory._to_linear_operand(argument_expr)
        self._x_var = self._allocate_arg_var_if_necessary(argument_expr, pos=1)

    def get_artefact(self, pos):
        if pos == 0:
            return self.as_var
        elif pos == 1:
            return self._x_var

    @property
    def argument_expr(self):
        return self._argument_expr

    def is_discrete(self):
        return self._argument_expr.is_discrete()

    def to_string(self):
        return "{0:s}({1!s})".format(self.function_symbol, self._argument_expr)

    def copy(self, target_model, memo):
        copy_key = id(self)
        cloned_expr = memo.get(copy_key)
        if cloned_expr is None:
            copied_arg_expr = self._argument_expr.copy(target_model, memo)
            cloned_expr = self.__class__(model=target_model, argument_expr=copied_arg_expr)
            memo[copy_key] = cloned_expr
        return cloned_expr


class AbsExpr(UnaryFunctionalExpr):

    def relaxed_copy(self, relaxed_model, var_map):
        raise DocplexLinearRelaxationError(self, cause='abs')

    def __init__(self, model, argument_expr):
        UnaryFunctionalExpr.__init__(self, model, argument_expr)

    def _get_function_symbol(self):
        return "abs"

    def clone(self):
        return AbsExpr(self.model, self._argument_expr)

    # noinspection PyArgumentEqualDefault,PyArgumentEqualDefault
    def _resolve(self):
        self_f_var = self._f_var
        assert self_f_var
        abs_index = self_f_var.index
        abs_names = ["_abs_pp_%d" % abs_index, "_abs_np_%d" % abs_index] if use_debug_names else [None, None]
        # 1. allocate two variables in one pass.
        positive_var = self._new_generated_continuous_var(artefact_pos=2, lb=0, name=abs_names[0])
        negative_var = self._new_generated_continuous_var(artefact_pos=3, lb=0, name=abs_names[1])

        # F(x) = p + n
        ct1 = (self_f_var == positive_var + negative_var)
        # sos
        self.sos = self.new_generated_sos1(dvars=[positive_var, negative_var])
        # # x = p-n
        ct2 = (self._argument_expr == positive_var - negative_var)

        self._post_generated_cts([ct1, ct2])
        # store
        self._artefact_vars = (positive_var, negative_var)

    def get_artefact(self, pos):
        if pos <= 1:
            return super(AbsExpr, self).get_artefact(pos)
        else:
            # offset is 2
            assert 2 <= pos <= 3
            return self._artefact_vars[pos - 2]

    def _get_solution_value(self, s=None):
        raw = abs(self._argument_expr._get_solution_value(s))
        return self._round_if_discrete(raw)

    def __repr__(self):
        return "docplex.mp.AbsExpr({0:s})".format(self._argument_expr.truncated_str())


# noinspection PyAbstractClass
class _SequenceExpr(_FunctionalExpr):
    # INTERNAL: base class for functional exprs with a sequence argument (e.g. min/max)

    def __init__(self, model, exprs, name=None):
        _FunctionalExpr.__init__(self, model, name)
        if is_iterable(exprs) or is_iterator(exprs):
            self._exprs = exprs
        else:
            self._exprs = [model._lfactory._to_linear_operand(exprs)]
        # allocate xvars iff necessary
        self._xvars = [self._allocate_arg_var_if_necessary(expr, pos=e) for e, expr in enumerate(self._exprs, start=1)]

    @property
    def nb_args(self):
        return len(self._exprs)

    def is_discrete(self):
        return all(map(lambda ex: ex.is_discrete(), self._exprs))

    def _get_args_string(self, sep=","):
        return sep.join(e.truncated_str() for e in self._exprs)

    def to_string(self):
        # generic: format expression arguments with holophraste
        str_args = self._get_args_string()
        return "{0}({1!s})".format(self.function_symbol, str_args)

    def iter_exprs(self):
        return iter(self._exprs)

    def _generate_variables(self):
        # INTERNAL: variable generator scanning all expressions
        # may return the same variable twice (or more)
        # use varset() if you need the set.
        for e in self._exprs:
            for v in e.iter_variables():
                yield v
        yield self._get_resolved_f_var()

    def iter_variables(self):
        return self._generate_variables()

    def contains_var(self, dvar):
        return dvar is self._f_var

    def _get_solution_value(self, s=None):
        fvar = self._f_var
        if self._is_resolved() and (not s or fvar in s):
            raw = fvar._get_solution_value(s)
        else:
            raw = self.compute_solution_value(s)
        return self._round_if_discrete(raw_value=raw)

    def compute_solution_value(self, s):
        raise NotImplementedError  # pragma: no cover

    def copy(self, target_model, memo):
        copy_key = id(self)
        cloned_expr = memo.get(copy_key)
        if cloned_expr is None:
            copied_exprs = [expr.copy(target_model, memo) for expr in self._exprs]
            cloned_expr = self.__class__(target_model, copied_exprs, self.name)
            # add in mapping
            memo[copy_key] = cloned_expr
        return cloned_expr

    def clone(self):
        # generic clone
        return self.__class__(self.model, self._exprs, self.name)

    def get_logical_seq_artefact(self, zvars, pos):
        # 0 -> fvar
        # 1 .. N -> xargs
        # N+1 .. 2N -> zvars
        if pos == 0:
            return self.as_var
        else:
            nb_args = self.nb_args
            if 1 <= pos <= nb_args:
                return self._xvars[pos - 1]
            else:
                assert nb_args + 1 <= pos <= 2 * nb_args
                zvar_pos = pos - (nb_args + 1)
                return zvars[zvar_pos]


class MinimumExpr(_SequenceExpr):
    """ An expression that represents the minimum of a sequence of expressions.

    This expression can be used in all arithmetic operations.
    After a solve, the value of this expression is equal to the minimum of the values
    of its argument expressions.
    """

    def __init__(self, model, exprs, name=None):
        _SequenceExpr.__init__(self, model, exprs, name)

    def _get_function_symbol(self):
        return "min"

    def __repr__(self):
        str_args = self._get_args_string()
        return "docplex.mp.MinExpr({0!s})".format(str_args)

    def _resolve(self):
        self_min_var = self._f_var
        assert self_min_var
        self_x_vars = self._xvars
        nb_args = len(self_x_vars)
        if 0 == nb_args:
            self._f_var.set_bounds(0, 0)
        elif 1 == nb_args:
            self._new_generated_binary_ct(self_min_var, self._xvars[0])
        else:
            cts = []
            for xv in self_x_vars:
                cts.append(self_min_var <= xv)
            # allocate N _generated_ binaries
            # reserve 1 + nb_args slots for artefacts
            z_vars = self._new_generated_binary_varlist(offset=nb_args + 1, keys=nb_args)
            self.z_vars = z_vars
            # sos?
            cts.append(self.model.sum(z_vars) == 1)
            self._post_generated_cts(cts)
            # indicators
            for i in range(nb_args):
                z = z_vars[i]
                x = self_x_vars[i]
                # need a block generation of indicators
                self._new_generated_indicator(binary_var=z, linear_ct=(self_min_var >= x))

    def compute_solution_value(self, s):
        return min(expr._get_solution_value(s) for expr in self._exprs)

    def get_artefact(self, pos):
        return self.get_logical_seq_artefact(self.z_vars, pos)


class MaximumExpr(_SequenceExpr):
    """ An expression that represents the maximum of a sequence of expressions.

    This expression can be used in all arithmetic operations.
    After a solve, the value of this expression is equal to the minimum of the values
    of its argument expressions.
    """

    def __init__(self, model, exprs, name=None):
        _SequenceExpr.__init__(self, model, exprs, name)

    def _get_function_symbol(self):
        return "max"

    def __repr__(self):
        str_args = self._get_args_string()
        return "docplex.mp.MaxExpr({0!s})".format(str_args)

    def _resolve(self):
        self_max_var = self._f_var
        self_x_vars = self._xvars
        nb_args = len(self_x_vars)
        if 0 == nb_args:
            self._f_var.set_bounds(0, 0)  # what else ??
        elif 1 == nb_args:
            self._new_generated_binary_ct(self_max_var, self._xvars[0])
        else:
            for xv in self_x_vars:
                self._new_generated_binary_ct(self_max_var, xv, 'GE')
            # allocate N binaries
            z_vars = self._new_generated_binary_varlist(keys=nb_args, offset=nb_args + 1)
            self.z_vars = z_vars
            # sos?
            self._new_generated_binary_ct(self.model.sum(z_vars), 1)
            # indicators
            for i in range(nb_args):
                z = z_vars[i]
                x = self_x_vars[i]
                self._new_generated_indicator(binary_var=z, linear_ct=(self_max_var <= x))

    def compute_solution_value(self, s):
        return max(expr._get_solution_value(s) for expr in self._exprs)

    def get_artefact(self, pos):
        return self.get_logical_seq_artefact(self.z_vars, pos)


class LogicalNotExpr(UnaryFunctionalExpr):
    def _create_functional_var(self, named=True):
        # the resulting variable is a binary variable...
        bvar = self._new_generated_binary_var(artefact_pos=0, name=None)
        self._name_functional_var_name(bvar)
        return bvar

    def is_discrete(self):
        return True

    def _get_function_symbol(self):
        return "not"

    def as_logical_operand(self):
        return self._get_resolved_f_var()

    def __init__(self, model, argument_expr):
        UnaryFunctionalExpr.__init__(self, model, argument_expr)
        self._logical_op_arg = argument_expr.as_logical_operand()
        assert self._logical_op_arg is not None
        self._actual_arg_s = str(argument_expr)

    def to_string(self):
        return "{0:s}({1!s})".format(self.function_symbol, self._actual_arg_s)

    def clone(self):
        return LogicalNotExpr(self.model, self._argument_expr)

    # noinspection PyArgumentEqualDefault,PyArgumentEqualDefault
    def _resolve(self):
        not_var = self._f_var
        assert not_var

        # not_x + x == 1
        ct1 = (not_var + self._logical_op_arg == 1)
        self._post_generated_cts([ct1])
        # store
        self.not_ct = ct1

    def _get_solution_value(self, s=None):
        arg_val = self._argument_expr._get_solution_value(s)
        return 0 if arg_val else 1

    def __repr__(self):
        return "docplex.mp.NotExpr({0:s})".format(self._argument_expr.truncated_str())


class _LogicalSequenceExpr(_SequenceExpr):

    def as_logical_operand(self):
        return self._get_resolved_f_var()

    def _create_functional_var(self, named=True):
        # the resulting variable is a binary variable...
        bvar = self._new_generated_binary_var(artefact_pos=0, name=None)
        self._name_functional_var_name(bvar)
        return bvar

    def __init__(self, model, exprs, name=None):
        _FunctionalExpr.__init__(self, model, name)
        assert is_iterable(exprs) or is_iterator(exprs)
        self._exprs = exprs
        # never allocate vars: arguments --are-- binary variables.
        self._xvars = exprs

    def _get_args_string(self, sep=","):
        def first_or_id(x):
            try:
                r = x[0]
            except TypeError:
                r = x
            return r

        s = sep.join(str(first_or_id(b.origin)) if b.is_generated() else str(b) for b in self._xvars)
        return s

    def is_discrete(self):
        return True

    precision = 1e-5


class LogicalAndExpr(_LogicalSequenceExpr):

    def _get_function_symbol(self):
        return "and"

    def __repr__(self):
        str_args = self._get_args_string()
        return "docplex.mp.LogicalAndExpr({0!s})".format(str_args)

    def compute_solution_value(self, s):
        # return 1/0 not True/False
        threshold = 1 - self.precision
        return 1 if all(ex._get_solution_value(s) >= threshold for ex in self._exprs) else 0

    def _resolve(self):
        self_and_var = self._f_var
        self_x_vars = self._xvars

        if self_x_vars:
            cts = [(self_and_var <= xv) for xv in self_x_vars]
            m = self._model
            nb_vars = len(self_x_vars)
            # rtc-39600: subtract n-1 from the sum.
            # the -and- var is propagated to 1 if all sum vars are 1.
            cts.append(self_and_var >= m._aggregator._sum_with_seq(self._xvars) - (nb_vars - 1))
            self._post_generated_cts(cts)


class LogicalOrExpr(_LogicalSequenceExpr):

    def _get_function_symbol(self):
        return "or"

    def __repr__(self):
        str_args = self._get_args_string()
        return "docplex.mp.LogicalOrExpr({0!s})".format(str_args)

    def compute_solution_value(self, s):
        # return 1/0 not True/False
        threshold = 1 - self.precision
        return 1 if any(ex._get_solution_value(s) >= threshold for ex in self._exprs) else 0

    def _resolve(self):
        self_or_var = self._f_var
        self_x_vars = self._xvars

        if self_x_vars:
            cts = [(xv <= self_or_var) for xv in self_x_vars]
            m = self._model
            cts.append(self_or_var <= m._aggregator._sum_with_seq(self._xvars))
            self._post_generated_cts(cts)
        self._resolved = True


class PwlExpr(UnaryFunctionalExpr):

    def __init__(self, model,
                 pwl_func, argument_expr,
                 usage_counter,
                 y_var=None,
                 add_counter_suffix=True,
                 resolve=True):
        UnaryFunctionalExpr.__init__(self, model, argument_expr)
        self._pwl_func = pwl_func
        self._usage_counter = usage_counter
        self._f_var = y_var
        if pwl_func.name:
            # ?
            if add_counter_suffix:
                self.name = '{0}_{1!s}'.format(self._pwl_func.name, self._usage_counter)
            else:
                self.name = self._pwl_func.name
        if resolve:
            self._ensure_resolved()

    def _get_function_symbol(self):
        # this method determines the name of the generated variable
        # as usual it starts with "_" to mark this is a generated variable.
        pwl_name = self._pwl_func.get_name()
        # TODO: what if pwl_name is not LP-compliant??
        return "pwl" if not pwl_name else "pwl_%s#" % pwl_name

    def _get_solution_value(self, s=None):
        raw = self._f_var._get_solution_value(s)
        return self._round_if_discrete(raw)

    def iter_variables(self):
        for v in self._argument_expr.iter_variables():
            yield v
        yield self._get_resolved_f_var()

    def _resolve(self):
        mdl = self._model
        pwl_constraint = mdl._lfactory.new_pwl_constraint(self, self.get_name())
        mdl._add_pwl_constraint_internal(pwl_constraint)

    @property
    def pwl_func(self):
        return self._pwl_func

    @property
    def usage_counter(self):
        return self._usage_counter

    def __repr__(self):
        return "docplex.mp.PwlExpr({0:s}, {1:s})".format(self._get_function_symbol(),
                                                         self._argument_expr.truncated_str())

    def copy(self, target_model, memo):
        copy_key = id(self)
        cloned_expr = memo.get(copy_key)
        if cloned_expr is None:
            copied_pwl_func = memo[self.pwl_func]
            copied_x_var = memo[self._x_var]
            cloned_expr = PwlExpr(target_model, copied_pwl_func, copied_x_var, self.usage_counter)
            copied_pwl_expr_f_var = memo.get(self._f_var)
            if copied_pwl_expr_f_var:
                cloned_expr._f_var = copied_pwl_expr_f_var
                # Need to set the _origin attribute of the copied var
                copied_pwl_expr_f_var._origin = cloned_expr

            memo[copy_key] = cloned_expr

        return cloned_expr

    def relaxed_copy(self, relaxed_model, var_map):
        raise DocplexLinearRelaxationError(self, cause='pwl')
