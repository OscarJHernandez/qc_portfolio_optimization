# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from itertools import product, islice

from docplex.mp.sosvarset import SOSVariableSet
from docplex.mp.operand import LinearOperand
from docplex.mp.dvar import Var
from docplex.mp.linear import MonomialExpr, LinearExpr, AbstractLinearExpr, ZeroExpr, ConstantExpr
from docplex.mp.operand import Operand
from docplex.mp.constants import ComparisonType, UpdateEvent, ObjectiveSense
from docplex.mp.constr import LinearConstraint, RangeConstraint, \
    IndicatorConstraint, PwlConstraint, EquivalenceConstraint, IfThenConstraint, NotEqualConstraint, QuadraticConstraint
from docplex.mp.functional import MaximumExpr, MinimumExpr, AbsExpr, PwlExpr, LogicalAndExpr, LogicalOrExpr, \
    LogicalNotExpr
from docplex.mp.pwl import PwlFunction
from docplex.mp.compat23 import fast_range, izip_longest
from docplex.mp.environment import Environment
from docplex.mp.utils import DOcplexException, DocplexQuadToLinearException
from docplex.mp.utils import MultiObjective
from docplex.mp.utils import is_string, is_pandas_dataframe, is_function, is_iterable, is_int, is_number, has_len,\
                              is_iterator, str_maxed, generate_constant, is_ordered_sequence


from docplex.mp.compat23 import izip
from docplex.mp.kpi import KPI
from docplex.mp.solution import SolveSolution
from docplex.mp.sttck import StaticTypeChecker


class ModificationBlocker(object):
    fmt = "Cplex cannot modify {0} in-place"

    all_blockers = {}

    def __init__(self, cause):
        self._msg = self.fmt.format(cause)

    def cannot_modify(self, expr):
        raise DOcplexException("{0}: {1}".format(self._msg, expr))

    def notify_expr_modified(self, expr, event):  # event is ignored
        self.cannot_modify(expr)

    @classmethod
    def get_blocker(cls, name):
        cached_blocker = cls.all_blockers.get(name)
        if cached_blocker is None:
            blocker = ModificationBlocker(name)
            cls.all_blockers[name] = blocker
            cached_blocker = blocker
        return cached_blocker


def fix_format_string(fmt, dimen=1, key_format='_%s'):
    ''' Fixes a format string so that it contains dimen slots with %s inside
        arguments are:
         --- dimen is the number of slots we need
         --- key_format is the format in which the %s is embedded. By default '_%s'
             for example if each item has to be surrounded by {} set key_format to _{%s}
    '''
    assert dimen >= 1
    actual_nb_slots = 0
    curpos = 0
    str_size = len(fmt)
    while curpos < str_size and actual_nb_slots < dimen:
        new_pos = fmt.find('%', curpos)
        if new_pos < 0:
            break
        actual_nb_slots += 1
        if actual_nb_slots >= dimen:
            break
        curpos = new_pos + 2
    # how much slots do we need to add to the end of the string??
    nb_missing = max(0, dimen - actual_nb_slots)
    return fmt + nb_missing * (key_format % '%s')


def is_tuple_w_standard_str(z):
    if isinstance(z, tuple):
        zclass = z.__class__
        return zclass is tuple or not ("__str__" in zclass.__dict__)
    else:
        return False


def str_flatten_tuple(key, sep="_"):
    if is_tuple_w_standard_str(key):
        return sep.join(str(f) for f in key)
    else:
        return str(key)


def compile_naming_function(keys, user_name, dimension=1, key_format=None,
                            _default_key_format='_%s', stringifier=str_flatten_tuple):
    # INTERNAL
    # builds a naming rule from an input , a dimension, and an optional meta-format
    # Makes sure the format string does contain the right number of format slots
    assert user_name is not None

    if is_string(user_name):
        if key_format is None:
            used_key_format = _default_key_format
        elif is_string(key_format):
            # -- make sure some %s is inside, otherwise add it
            if '%s' in key_format:
                used_key_format = key_format
            else:
                used_key_format = key_format + '%s'
        else:  # pragma: no cover
            raise DOcplexException("key format expects string format or None, got: {0!r}".format(key_format))

        fixed_format_string = fix_format_string(user_name, dimension, used_key_format)
        if 1 == dimension:
            return lambda k: fixed_format_string % stringifier(k)
        else:
            # here keys are tuples of size >= 2
            return lambda key_tuple: fixed_format_string % key_tuple

    elif is_function(user_name):
        return user_name

    elif is_iterable(user_name):
        # check that the iterable has same len as keys,
        # otherwise thereis no more default naming and None cannot appear in CPLEX name arrays
        list_names = list(user_name)
        if len(list_names) < len(keys):
            raise DOcplexException("An array of names should have same len as keys, expecting: {0}, go: {1}"
                                   .format(len(keys), len(list_names)))
        key_to_names_dict = {k: nm for k, nm in izip(keys, list_names)}
        # use a closure
        return lambda k: key_to_names_dict[k]  # if k in key_to_names_dict else default_fn()

    else:
        raise DOcplexException('Cannot use this for naming variables: {0!r} - expecting string, function or iterable'
                               .format(user_name))


class _AbstractModelFactory(object):

    def __init__(self, model, engine):
        self._model = model
        self._engine = engine
        self._checker = model._checker
        ordered = model.keep_ordering
        self.term_dict_type = self._term_dict_type_from_ordering(model.keep_ordering)
        self._var_dict_type = dict
        if ordered and not Environment.env_is_python36:
            # FastOrderedDict does not accept zip argument...
            from collections import OrderedDict
            self._var_dict_type = OrderedDict

    def update_engine(self, engine):
        # the model has already disposed the old engine, if any
        self._engine = engine

    def var_dict_type(self, ordered):
        return self._var_dict_type if ordered else dict

    @classmethod
    def _term_dict_type_from_ordering(cls, ordered):
        if not ordered or Environment.env_is_python36:
            return dict
        else:
            from docplex.mp.xcounter import FastOrderedDict
            return FastOrderedDict

    @property
    def ordered(self):
        return self._model._keep_ordering

    def update_ordering(self, ordered):
        self.term_dict_type = self._term_dict_type_from_ordering(ordered)


class ModelFactory(_AbstractModelFactory):
    status_var_fmt = '_bool{{{0:s}}}'

    @staticmethod
    def float_or_default(bound, default_bound):
        return default_bound if bound is None else float(bound)

    def __init__(self, model, engine):
        _AbstractModelFactory.__init__(self, model, engine)

        self._var_container_counter = 0
        self.number_validation_fn = model._checker.get_number_validation_fn()
        self.stringifier = str_flatten_tuple

    @property
    def infinity(self):
        return self._engine.get_infinity()

    def _new_term_dict(self):
        # INTERNAL
        return self.term_dict_type()

    def new_zero_expr(self):
        return ZeroExpr(self._model)

    def fatal(self, msg, *args):
        self._model.fatal(msg, *args)

    def warning(self, msg, *args):
        self._model.warning(msg, args)

    def _make_new_var(self, vartype, lb, ub, varname, origin=None):
        self_model = self._model
        idx = self._engine.create_one_variable(vartype, lb, ub, varname)
        var = Var(self_model, vartype, varname, lb, ub, _safe_lb=True, _safe_ub=True)
        self_model._register_one_var(var, idx, varname)
        var.origin = origin
        return var

    def new_var(self, vartype, lb=None, ub=None, varname=None, safe=False):
        self_model = self._model
        if not safe:
            self._checker.check_var_domain(lb, ub, varname)
            logger = self_model.logger
            rlb = vartype.resolve_lb(lb, logger)
            rub = vartype.resolve_ub(ub, logger)
        else:
            rlb = vartype.default_lb if lb is None else lb
            rub = vartype.default_ub if ub is None else ub
        used_varname = None if self_model.ignore_names else varname
        return self._make_new_var(vartype, rlb, rub, used_varname, origin=None)

    def new_constraint_status_var(self, ct):
        # INTERNAL
        model = self._model
        binary_vartype = model.binary_vartype
        status_var_fmt = self.status_var_fmt
        if model.ignore_names or status_var_fmt is None:
            varname = None
        else:
            # use name if any else use truncated ct string representation
            base_varname = self.status_var_fmt.format(ct.name or str_maxed(ct, maxlen=20))
            # if name is already taken, use unique index at end to disambiguate
            varname = model._get_non_ambiguous_varname(base_varname)

        return self._make_new_var(binary_vartype, 0, 1, varname, origin=ct)

    # --- sequences
    def make_key_seq(self, keys, name):
        # INTERNAL Takes as input a candidate keys input and returns a valid key sequence
        used_name = name
        check_keys = True
        used_keys = []
        if is_iterable(keys):
            if is_pandas_dataframe(keys):
                used_keys = keys.index.values
            elif has_len(keys):
                used_keys = keys
            elif is_iterator(keys):
                used_keys = list(keys)
            else:
                # TODO: make a test for this case.
                self.fatal("Cannot handle iterable var keys: {0!s} : no len() and not an iterator",
                           keys)  # pragma: no cover

        elif is_int(keys) and keys >= 0:
            # if name is str and we have a size, disable automatic names
            used_name = None if name is str else name
            used_keys = range(keys)
            check_keys = False
        else:
            self.fatal("Unexpected var keys: {0!s}, expecting iterable or integer", keys)  # pragma: no cover

        if check_keys and len(used_keys):  # do not check truth value of used_keys: can be a Series!
            self._checker.typecheck_key_seq(used_keys)
        return used_name, used_keys

    def _get_stringifier(self, dimension, keys):
        if dimension > 1:
            stringifier = str_flatten_tuple
        elif len(keys) <= 100:
            stringifier = str_flatten_tuple
        else:
            is_tuple = isinstance(keys[0], tuple)
            stringifier = str_flatten_tuple if is_tuple else str
        return stringifier


    def _expand_names(self, keys, user_name, dimension, key_format):
        if user_name is None or self._model.ignore_names:
            # no automatic names, ever
            return []
        else:
            stringifier_ = self._get_stringifier(dimension, keys)
            actual_naming_fn = compile_naming_function(keys, user_name, dimension, key_format,
                                                       stringifier=stringifier_)
            computed_names = [actual_naming_fn(key) for key in keys]
            return computed_names

    def _check_bounds(self, nb_vars, bounds, default_bound, true_if_lb):
        nb_bounds = len(bounds)
        bound_name = 'lb' if true_if_lb else 'ub'
        for b, b_value in enumerate(bounds):
            if b_value is not None and not is_number(b_value):
                self.fatal("Variable {2} expect numbers, {0!r} was passed at pos: #{1}",
                           b_value, b, bound_name)
        float_bounds = [self.float_or_default(bv, default_bound) for bv in bounds]

        if nb_bounds > nb_vars:
            self.warning(
                "Variable bounds list too large, required: %d, got: %d." % (nb_vars, nb_bounds))
            return float_bounds[:nb_vars]
        else:
            return float_bounds

    def _expand_bounds(self, keys, var_bound, default_bound, size, true_if_lb):
        ''' Converts raw bounds data (either LB or UB) to CPLEX-compatible bounds list.
            If lbs is None, this is the default, return [].
            If lbs is [] take the default again.
            If it is a number, build a list of size <size> with this number.
            If it is a list, use it if size ok (check numbers??),
            else try it as a function over keys.
        '''
        if var_bound is None:
            # default lb is zero, default ub is infinity
            return []

        elif is_number(var_bound):
            self._checker.typecheck_num(var_bound, caller='in variable bound')
            if true_if_lb:
                if var_bound == default_bound:
                    return []
                else:
                    return [float(var_bound)] * size
            else:
                # ub
                if var_bound >= default_bound:
                    return []
                else:
                    return [float(var_bound)] * size

        elif is_ordered_sequence(var_bound):
            nb_bounds = len(var_bound)
            if nb_bounds < size:
                # see how we can use defaults for those missing bounds
                self.fatal("Variable bounds list is too small, expecting: %d, got: %d" % (size, nb_bounds))
            else:
                return self._check_bounds(size, var_bound, default_bound, true_if_lb)

        elif is_iterator(var_bound):
            # unfold the iterator, as CPLEX needs a list
            return list(var_bound)

        elif isinstance(var_bound, dict):
            dict_bounds = [var_bound.get(k, default_bound) for k in keys]
            return self._check_bounds(size, dict_bounds, default_bound, true_if_lb)
        else:
            # try a function?
            try:
                fn_bounds = [var_bound(k) for k in keys]
                return self._check_bounds(size, fn_bounds, default_bound, true_if_lb)

            except TypeError:
                self._bad_bounds_fatal(var_bound)

            except Exception as e:  # pragma: no cover
                self.fatal("error calling function model bounds: {0!s}, error: {1!s}", var_bound, e)

    def _bad_bounds_fatal(self, bad_bound):
        self.fatal("unexpected variable bound: {0!s}, expecting: None|number|function|iterable", bad_bound)

    @staticmethod
    def safe_kth(array_or_empty, k, fallback=None):
        if array_or_empty:
            return array_or_empty[k]
        else:
            return fallback

    def new_multitype_var_list(self, size, vartypes, lbs=None, ubs=None, names=None):
        if not size:
            return []
        mdl = self._model

        # -------------------------
        assert size == len(vartypes)
        assert not lbs or size == len(lbs)
        assert not ubs or size == len(ubs)
        assert not names or size == len(names)
        # -------------------------

        if names:
            allvars = [Var(mdl, vartypes[k],
                           names[k],
                           self.safe_kth(lbs, k, vartypes[k].default_lb),
                           self.safe_kth(ubs, k, vartypes[k].default_ub),
                           _safe_lb=True,
                           _safe_ub=True) for k in fast_range(size)]
        else:
            allvars = [Var(mdl, vartypes[k],
                           None,
                           self.safe_kth(lbs, k),
                           self.safe_kth(ubs, k),
                           _safe_lb=True,
                           _safe_ub=True) for k in fast_range(size)]

        cpxnames = names or []  # no None
        indices = self._engine.create_multitype_variables(size, vartypes, lbs, ubs, cpxnames)
        mdl._register_block_vars(allvars, indices, names)
        return allvars

    def var_list(self, keys, vartype, lb, ub, name=None, key_format=None):
        actual_name, fixed_keys = self.make_key_seq(keys, name)
        ctn = self._new_var_container(vartype, key_list=[fixed_keys], lb=lb, ub=ub, name=name)
        return self.new_var_list(ctn, fixed_keys, vartype, lb, ub, actual_name, 1, key_format)

    def new_var_list(self, var_container,
                     key_seq, vartype,
                     lb=None, ub=None,
                     name=str,
                     dimension=1, key_format=None,
                     _safe_bounds=False,
                     _safe_names=False):
        number_of_vars = len(key_seq)
        if 0 == number_of_vars:
            return []

        # compute defaults once
        default_lb = vartype.default_lb
        default_ub = vartype.default_ub

        if _safe_bounds:
            assert not lb or len(lb) == number_of_vars
            xlbs = lb or []
            assert not ub or len(ub) == number_of_vars
            xubs = ub or []
        else:
            xlbs = self._expand_bounds(key_seq, lb, default_lb, number_of_vars, true_if_lb=True)
            xubs = self._expand_bounds(key_seq, ub, default_ub, number_of_vars, true_if_lb=False)
        # at this point both list are either [] or have size numberOfVars

        if _safe_names:
            xnames = name or []
        else:
            xnames = self._expand_names(key_seq, name, dimension, key_format)

        safe_lbs = _safe_bounds or not xlbs
        safe_ubs = _safe_bounds or not xubs
        if xlbs and xubs and not _safe_bounds:
            # no domain check from reader
            self._checker.check_vars_domain(xlbs, xubs, xnames)

        mdl = self._model
        allvars = [Var(mdl, vartype,
                       self.safe_kth(xnames, k, None),
                       self.safe_kth(xlbs, k, default_lb),
                       self.safe_kth(xubs, k, default_ub),
                       _safe_lb=safe_lbs,
                       _safe_ub=safe_ubs) for k in fast_range(number_of_vars)]

        # query the engine for a list of indices.
        indices = self._engine.create_variables(len(key_seq), vartype, xlbs, xubs, xnames)
        mdl._register_block_vars(allvars, indices, xnames)
        if var_container:
            for dv in allvars:
                mdl.set_var_container(dv, var_container)
        return allvars

    def _make_var_dict(self, keys, var_list, ordered):
        self._checker.check_for_duplicate_keys(keys)
        _dict_type = self.var_dict_type(ordered)
        vdict = _dict_type(izip(keys, var_list))
        return vdict

    def new_var_dict(self, keys, vartype, lb, ub, name, key_format, ordered=False):
        actual_name, key_seq = self.make_key_seq(keys, name)
        ctn = self._new_var_container(vartype, key_list=[key_seq], lb=lb, ub=ub, name=name)
        var_list = self.new_var_list(ctn, key_seq, vartype, lb, ub, actual_name, 1, key_format)
        return self._make_var_dict(key_seq, var_list, ordered)

    def new_var_multidict(self, seq_of_key_seqs, vartype, lb, ub, name, key_format=None, ordered=False):
        # ---
        fixed_keys = [self.make_key_seq(ks, name)[1] for ks in seq_of_key_seqs]
        # the sequence of keysets should answer to len(no generators here)
        dimension = len(fixed_keys)
        if dimension < 1:
            self.fatal("len of key sequence must be >= 1, got: {0}", dimension)  # pragma: no cover

        # create cartesian product of keys...
        all_key_tuples = list(product(*fixed_keys))
        # check empty list
        if not all_key_tuples:
            self.fatal('multidict has no keys to index the variables')

        # always pass a sequence of sequences
        ctn = self._new_var_container(vartype, key_list=fixed_keys, lb=lb, ub=ub, name=name)
        cube_vars = self.new_var_list(ctn, all_key_tuples, vartype, lb, ub, name, dimension, key_format)

        return self._make_var_dict(keys=all_key_tuples, var_list=cube_vars, ordered=ordered)

    def new_var_df(self, keys1, keys2, vartype, lb=None, ub=None, name=None):  # pragma: no cover
        try:
            from pandas import DataFrame
        except ImportError:
            DataFrame = None
            self.fatal('make_var_df() requires pandas module - not found')

        _, row_keys = self.make_key_seq(keys1, name)
        _, col_keys = self.make_key_seq(keys2, name)
        matrix_keys = [(k1, k2) for k1 in row_keys for k2 in col_keys]
        ctn = self._new_var_container(vartype, key_list=matrix_keys, lb=lb, ub=ub, name=name)
        lvars = self.new_var_list(ctn, matrix_keys, vartype, lb, ub, name, dimension=2, key_format=None)
        # TODO: see how to do without this temp dict
        dd = dict(izip(matrix_keys, lvars))
        # row-oriented dict
        rowd = {row_k: [dd[row_k, col_k] for col_k in col_keys] for row_k in row_keys}
        vdtf = DataFrame.from_dict(rowd, orient='index', dtype='object')
        # convert to string or not?
        vdtf.columns = col_keys
        return vdtf

    def _new_constant_expr(self, cst, safe_number=True):
        k = cst
        if not safe_number:
            self_number_validation_fn = self.number_validation_fn
            if self_number_validation_fn:
                k = self_number_validation_fn(cst)
        return ConstantExpr(self._model, k)

    use_constant_expr = True

    def constant_expr(self, cst, safe_number=False):
        if not cst:
            return self.new_zero_expr()
        else:
            return self._new_constant_expr(cst, safe_number=safe_number)

    def linear_expr(self, arg=None, constant=0, name=None, safe=False, transient=False):
        return LinearExpr(self._model, arg, constant, name, safe=safe, transient=transient)

    def _new_monomial_expr(self, dvar, coeff, safe=True):
        if coeff:
            return MonomialExpr(self._model, dvar, coeff, safe)
        else:
            return self.new_zero_expr()

    _operand_types = (AbstractLinearExpr, Var, ZeroExpr)

    @staticmethod
    def _is_operand(arg, accept_numbers=False):
        return isinstance(arg, Operand) or (accept_numbers and is_number(arg))

    def _to_linear_operand(self, e, force_clone=False, msg=None):
        if isinstance(e, LinearOperand):
            if force_clone:
                return e.clone()
            else:
                return e
        elif is_number(e):
            return self.constant_expr(cst=e, safe_number=False)
        else:
            try:
                return e.to_linear_expr()
            except AttributeError:
                # delegate to the factory
                return self.linear_expr(e)
            except DocplexQuadToLinearException as qe:
                used_msg = msg.format(e) if msg else qe.message
                raise DOcplexException(used_msg)

    def _to_linear_expr(self, e, linexpr_class=LinearExpr, force_clone=False):
        # TODO: replace by to_linear_operand
        if isinstance(e, linexpr_class):
            if force_clone:
                return e.clone()
            else:
                return e
        elif isinstance(e, self._operand_types):
            return e.to_linear_expr()
        elif is_number(e):
            return self.constant_expr(cst=e, safe_number=False)
        else:
            try:
                return e.to_linear_expr()
            except AttributeError:
                # delegate to the factory
                return self.linear_expr(e)

    def _to_expr(self, e):
        # INTERNAL
        if hasattr(e, "iter_terms"):
            return e
        elif is_number(e):
            return self.constant_expr(cst=e, safe_number=True)
        else:
            try:
                return e.to_expr()
            except AttributeError:
                self.fatal("cannot convert to expression: {0!r}", e)

    def new_binary_constraint(self, lhs, sense, rhs, name=None):
        ctsense = ComparisonType.parse(sense)
        return self._new_binary_constraint(lhs, ctsense, rhs, name)

    def _new_binary_constraint(self, lhs, sense, rhs, name=None):
        # noinspection PyPep8
        left_expr = self._to_linear_operand(lhs, msg="LinearConstraint. expects linear expressions, {0} was passed")
        right_expr = self._to_linear_operand(rhs, msg="LinearConstraint. expects linear expressions, {0} was passed")
        self._checker.typecheck_two_in_model(self._model, left_expr, right_expr, "new_binary_constraint")
        ct = LinearConstraint(self._model, left_expr, sense, right_expr, name)
        return ct

    def new_le_constraint(self, e, rhs, ctname=None):
        return self._new_binary_constraint(e, ComparisonType.LE, rhs, name=ctname)

    def new_eq_constraint(self, e, rhs, ctname=None):
        return self._new_binary_constraint(e, ComparisonType.EQ, rhs, name=ctname)

    def new_ge_constraint(self, e, rhs, ctname=None):
        return self._new_binary_constraint(e, ComparisonType.GE, rhs, name=ctname)

    def new_neq_constraint(self, lhs, rhs, ctname=None):
        m = self._model
        left_expr = self._to_linear_operand(lhs,
                                            msg="The `!=` operator requires two linear expressions, {0} was passed (left)")
        right_expr = self._to_linear_operand(rhs,
                                             msg="The `!=` operator requires two linear expressions, {0} was passed (right)")
        StaticTypeChecker.typecheck_discrete_expression(m, msg="NotEqualConstraint", expr=left_expr)
        StaticTypeChecker.typecheck_discrete_expression(m, msg="NotEqualConstraint", expr=right_expr)
        self._checker.typecheck_two_in_model(m, left_expr, right_expr, "new_binary_constraint")
        negated_ct = self.new_eq_constraint(lhs, rhs)
        ct = NotEqualConstraint(self._model, negated_ct, ctname)
        return ct

    def _check_range_feasibility(self, lb, ub, expr):
        # INTERNAL
        if not lb <= ub:
            self._model.error("infeasible range constraint, lb={0}, ub={1}, expr={2}", lb, ub, expr)

    def new_range_constraint(self, lb, expr, ub, name=None, check_feasible=True):
        self._check_range_feasibility(lb, ub, expr)
        linexpr = self._to_linear_operand(expr)
        rng = RangeConstraint(self._model, linexpr, lb, ub, name)
        linexpr.notify_used(rng)
        return rng

    def new_indicator_constraint(self, binary_var, linear_ct, true_value=1, name=None):
        # INTERNAL
        indicator_ct = IndicatorConstraint(self._model, binary_var, linear_ct, true_value, name)
        return indicator_ct

    def new_equivalence_constraint(self, binary_var, linear_ct, true_value=1, name=None):
        # INTERNAL
        equiv_ct = EquivalenceConstraint(self._model, binary_var, linear_ct, true_value, name)
        return equiv_ct

    def new_if_then_constraint(self, if_ct, then_ct, negate=False):
        def check_bvar_eq_10(lhs, rhs):
            return isinstance(lhs, Var) and\
                   lhs.is_binary() and\
                   rhs.is_constant() and\
                   (rhs.get_constant() == 0 or rhs.get_constant() == 1)

        # INTERNAL
        m = self._model
        indicator_ct = None
        if if_ct.sense == ComparisonType.EQ:
            if_ct_lhs = if_ct.left_expr
            if_ct_rhs = if_ct.right_expr
            if check_bvar_eq_10(if_ct_lhs, if_ct_rhs):
                bvar = if_ct_lhs
                true_value = if_ct_rhs.get_constant()
            elif check_bvar_eq_10(if_ct_rhs, if_ct_lhs):
                bvar = if_ct_rhs
                true_value = if_ct_lhs.get_constant()
            else:
                bvar = true_value = None
            if bvar is not None:
                assert true_value in {0, 1}
                if negate:
                    true_value = 1 - true_value
                m.info("If_then constraint has been simplified to an indicator constraint, binary variable: '{0}', true_value={1}"
                       .format(if_ct_lhs, true_value))
                indicator_ct = IndicatorConstraint(self._model, if_ct_lhs, linear_ct=then_ct, active_value=true_value)
        if indicator_ct is None:
            indicator_ct = IfThenConstraint(self._model, if_ct, then_ct, negate=negate)
        return indicator_ct

    def new_batch_equivalence_constraints(self, bvars, linear_cts, active_values, names):
        return [self.new_equivalence_constraint(bv, lct, active, name)
                for bv, lct, active, name in izip(bvars, linear_cts, active_values, names)]

    def new_batch_indicator_constraints(self, bvars, linear_cts, active_values, names):
        return [self.new_indicator_constraint(bv, lct, active, name)
                for bv, lct, active, name in izip(bvars, linear_cts, active_values, names)]

    def new_binary_constraint_or(self, ct1, arg2):
        orexpr = self.new_logical_or_expr([ct1, arg2])
        return orexpr

    def new_binary_constraint_and(self, ct, other):
        and_expr = self.new_logical_and_expr([ct, other])
        return and_expr

    # updates

    def update_linear_constraint_exprs(self, ct, expr_event):
        ct_event = UpdateEvent.LinearConstraintRhs if expr_event is UpdateEvent.ExprConstant \
            else UpdateEvent.LinearConstraintGlobal
        self._engine.update_constraint(ct, ct_event)

    def update_indicator_constraint_expr(self, ind, expr, event):
        self._engine.update_constraint(ind, UpdateEvent.IndicatorLinearConstraint, expr)

    def check_expr_discrete_lock(self, expr, arg):
        # INTERNAL
        if expr.is_discrete_locked():
            self.fatal('Expression: {0} is used in equivalence, cannot be modified', expr)

    def set_linear_constraint_expr_from_pos(self, lct, pos, new_expr, update_subscribers=True):
        # INTERNAL
        # pos is 0 for left, 1 for right
        new_operand = self._to_linear_operand(e=new_expr, force_clone=False)
        lct._check_editable(new_operand, self._engine)

        old_expr = lct.get_expr_from_pos(pos)
        exprs = [lct._left_expr, lct._right_expr]
        exprs[pos] = new_operand
        # -- event
        if old_expr.is_constant() and new_operand.is_constant():
            event = UpdateEvent.LinearConstraintRhs
        else:
            event = UpdateEvent.LinearConstraintGlobal
        # ---
        self._engine.update_constraint(lct, event, *exprs)
        lct.set_expr_from_pos(pos, new_operand)
        if update_subscribers:
            # -- update  subscribers
            old_expr.notify_unsubscribed(lct)
            new_operand.notify_used(lct)

    def set_linear_constraint_right_expr(self, ct, new_rexpr):
        self.set_linear_constraint_expr_from_pos(ct, pos=1, new_expr=new_rexpr)

    def set_linear_constraint_left_expr(self, ct, new_lexpr):
        self.set_linear_constraint_expr_from_pos(ct, pos=0, new_expr=new_lexpr)

    def set_linear_constraint_sense(self, ct, arg_newsense):
        new_sense = ComparisonType.parse(arg_newsense)
        if new_sense != ct.sense:
            self._engine.update_constraint(ct, UpdateEvent.ConstraintSense, new_sense)
            ct._internal_set_sense(new_sense)

    def set_range_constraint_lb(self, rngct, new_lb):
        self.set_range_constraint_bounds(rngct, new_lb, None)

    def set_range_constraint_ub(self, rngct, new_ub):
        self.set_range_constraint_bounds(rngct, None, new_ub)

    def set_range_constraint_bounds(self, rngct, new_lb, new_ub):
        lb_to_use = rngct.lb if new_lb is None else new_lb
        ub_to_use = rngct.ub if new_ub is None else new_ub
        # assuming the new bound has been typechecked already..
        self._engine.update_range_constraint(rngct, UpdateEvent.RangeConstraintBounds, lb_to_use, ub_to_use)
        if new_lb is not None:
            rngct._internal_set_lb(new_lb)
        if new_ub is not None:
            rngct._internal_set_ub(new_ub)
        self._check_range_feasibility(rngct.lb, rngct.ub, rngct.expr)

    def set_range_constraint_expr(self, rngct, new_expr):
        new_op = self._to_linear_operand(new_expr)
        self._engine.update_range_constraint(rngct, UpdateEvent.RangeConstraintExpr, new_op)
        old_expr = rngct.expr
        rngct._expr = new_expr
        old_expr.notify_unsubscribed(rngct)

    # ---------------------
    def _new_logical_expr(self, args, ctor, empty_value):
        bvars = [arg.as_logical_operand() for arg in args]
        # assume bvars is a sequence of binary vars
        nb_args = len(bvars)
        if not nb_args:
            return self._new_constant_expr(cst=empty_value, safe_number=True)
        elif 1 == nb_args:
            return bvars[0]
        else:
            return ctor(self._model, bvars)

    def new_logical_and_expr(self, args):
        return self._new_logical_expr(args, ctor=LogicalAndExpr, empty_value=1)

    def new_logical_or_expr(self, args):
        return self._new_logical_expr(args, ctor=LogicalOrExpr, empty_value=0)

    def new_logical_not_expr(self, arg):
        return LogicalNotExpr(self._model, arg)

    def logical_expr_to_constraint(self, logical_expr, name=None):
        # generate a constraint expr == 1
        ct1 = self._new_binary_constraint(lhs=logical_expr, rhs=1.0, sense=ComparisonType.EQ, name=name)
        return ct1

    def _new_min_max_expr(self, expr_class, builtin_fn, empty_value, *args):
        nb_args = len(args)
        if 0 == nb_args:
            return empty_value
        elif 1 == nb_args:
            return args[0]
        elif all(is_number(a) for a in args):
            # if all args are numbers, simply compute a number
            return builtin_fn(args)  # a number
        else:
            return expr_class(self._model, [self._to_linear_operand(a) for a in args])

    def new_max_expr(self, *args):
        return self._new_min_max_expr(MaximumExpr, max, -self.infinity, *args)

    def new_min_expr(self, *args):
        return self._new_min_max_expr(MinimumExpr, min, self.infinity, *args)

    def new_abs_expr(self, e):
        if is_number(e):
            return abs(e)
        else:
            self_model = self._model
            return AbsExpr(self_model, self._to_linear_operand(e))

    def resync_whole_model(self):
        self_model = self._model
        self_engine = self._engine

        for var in self_model.iter_variables():
            # do not call create_one_var public API
            # or resync would loop
            idx = self_engine.create_one_variable(var.vartype, var.lb, var.ub, var.name)
            if idx != var.index:  # pragma: no cover
                print("index discrepancy: {0!s}, new index= {1}, old index={2}"
                      .format(var, idx, var.index))

        for ct in self_model.iter_constraints():
            if isinstance(ct, LinearConstraint):
                self_engine.create_linear_constraint(ct)
            elif isinstance(ct, RangeConstraint):
                self_engine.create_range_constraint(ct)
            elif isinstance(ct, IndicatorConstraint):
                self_engine.create_logical_constraint(ct, is_equivalence=False)
            elif isinstance(ct, EquivalenceConstraint):
                self_engine.create_logical_constraint(ct, is_equivalence=True)
            elif isinstance(ct, QuadraticConstraint):
                self_engine.create_quadratic_constraint(ct)
            elif isinstance(ct, PwlConstraint):
                self_engine.create_pwl_constraint(ct)
            else:
                self_model.fatal("Unexpected constraint type: {0!s} - ignored", type(ct))  # pragma: no cover

        # send objective
        self_engine.set_objective_sense(self_model.objective_sense)
        if self_model.has_multi_objective():
            multi_objective = self_model._multi_objective
            exprs = multi_objective.exprs
            nb_exprs = len(exprs)
            self_engine.set_multi_objective_exprs(new_multiobjexprs=exprs,
                                                  old_multiobjexprs=[],
                                                  priorities=multi_objective.priorities,
                                                  weights=multi_objective.weights,
                                                  abstols=MultiObjective.as_optional_sequence(multi_objective.abstols,
                                                                                              nb_exprs),
                                                  reltols=MultiObjective.as_optional_sequence(multi_objective.reltols,
                                                                                              nb_exprs),
                                                  objnames=multi_objective.names)
        else:
            self_engine.set_objective_expr(self_model.objective_expr, old_objexpr=None)

    def new_sos(self, dvars, sos_type, weights, name):
        # INTERNAL
        new_sos = SOSVariableSet(model=self._model, variable_sequence=dvars, sos_type=sos_type, weights=weights,
                                 name=name)
        return new_sos

    def new_piecewise(self, pwl_def, name):
        self_model = self._model
        # Note that this object is specific only to DOcplex. It does not map to a Cplex object.
        pwl = PwlFunction(self_model, pwl_def=pwl_def, name=name)
        return pwl

    def new_pwl_expr(self, pwl_func, e, usage_counter, y_var=None, add_counter_suffix=True, resolve=True):
        return PwlExpr(self._model, pwl_func, e,
                       usage_counter,
                       add_counter_suffix=add_counter_suffix,
                       y_var=y_var,
                       resolve=resolve)

    def new_pwl_constraint(self, pwl_expr, name=None):
        self_model = self._model
        return PwlConstraint(self_model, pwl_expr, name)

    @staticmethod
    def default_objective_sense():
        return ObjectiveSense.Minimize

    def new_kpi(self, kpi_arg, name_arg):
        # make a name
        if name_arg:
            publish_name = name_arg
        elif hasattr(kpi_arg, 'name') and kpi_arg.name:
            publish_name = kpi_arg.name
        else:
            publish_name = str_maxed(kpi_arg, maxlen=32)
        new_kpi = KPI.new_kpi(self._model, kpi_arg, publish_name)
        return new_kpi

    def _new_constraint_block2(self, cts, ctnames):
        posted_cts = []
        prepfn = self._model._prepare_constraint
        checker = self._checker
        check_trivials = checker.check_trivial_constraints()

        if is_string(ctnames):
            from docplex.mp.utils import _AutomaticSymbolGenerator
            # no separator added, use a terminal "_" if need be
            ctnames = _AutomaticSymbolGenerator(ctnames)

        for ct, ctname in izip_longest(cts, ctnames):  # use izip_longest so as not to forget any ct
            if ct is None:  # izip stops
                break

            checker.typecheck_linear_constraint(ct, accept_range=False)
            checker.typecheck_string(ctname, accept_none=True, accept_empty=False, caller="Model.add_constraints()")
            if prepfn(ct, ctname, check_for_trivial_ct=check_trivials):
                posted_cts.append(ct)
        self._post_constraint_block(posted_cts)
        return posted_cts

    def _new_constraint_block1(self, cts):
        posted_cts = []
        checker = self._checker
        filterfn = self._model._prepare_constraint
        check_trivial = self._checker.check_trivial_constraints()
        # look first
        ctseq = list(cts)
        if not ctseq:
            return []

        try:
            # noinspection PyUnusedLocal
            ct, ctname = ctseq[0]
            tuple_mode = True
        except (TypeError, ValueError):
            # TypeError is for non-tuple
            # ValueError is for nonlength-2 tuples
            # not a tuple: we have only constraints and no names
            tuple_mode = False

        if tuple_mode:
            ctseq2 = self._checker.typecheck_linear_constraint_name_tuple_seq(ctseq)
            for ct, ctname in ctseq2:
                if filterfn(ct, ctname, check_for_trivial_ct=check_trivial, arg_checker=checker):
                    posted_cts.append(ct)
        else:
            checker.typecheck_constraint_seq(ctseq, check_linear=True, accept_range=True)
            for ct in ctseq:
                if filterfn(ct, ctname=None, check_for_trivial_ct=check_trivial, arg_checker=checker):
                    posted_cts.append(ct)
        self._post_constraint_block(posted_cts)
        return posted_cts

    def _post_constraint_block(self, posted_cts):
        if posted_cts:
            ct_indices = self._engine.create_block_linear_constraints(posted_cts)
            self._model._register_block_cts(self._model._linct_scope, posted_cts, ct_indices)

    # --- range block

    def new_range_block(self, lbs, exprs, ubs, names):
        try:
            n_exprs = len(exprs)
            if n_exprs != len(lbs):  # pragma: no cover
                self.fatal('incorrect number of expressions: expecting {0}, got: {1}'.format(len(lbs), n_exprs))

        except TypeError:
            pass  # no len available.
        if not names:
            names = generate_constant(None, None)

        ranges = [self.new_range_constraint(lb, exp, ub, name) for lb, exp, ub, name in
                      izip(lbs, exprs, ubs, names)]
        # else:
        #     ranges = [self.new_range_constraint(lb, exp, ub) for lb, exp, ub in izip(lbs, exprs, ubs)]
        self._post_constraint_block(ranges)
        return ranges

    def new_solution(self, var_value_dict=None, name=None, objective_value=None, **kwargs):
        keep_zeros = kwargs.get('keep_zeros', True)

        return SolveSolution(model=self._model, obj=objective_value,
                             var_value_map=var_value_dict, name=name,
                             keep_zeros=keep_zeros)

    def _new_var_container(self, vartype, key_list, lb, ub, name):
        # INTERNAL
        ctn = _VariableContainer(vartype, key_list, lb, ub, name)
        old_varctn_counter = self._var_container_counter
        ctn._index = old_varctn_counter
        ctn._index_offset = self._model.number_of_variables  # nb of variables before ctn
        self._var_container_counter = old_varctn_counter + 1

        return ctn


class _VariableContainer(object):
    def __init__(self, vartype, keys_seq, lb, ub, name):
        self._index = 0
        self._index_offset = 0
        self._vartype = vartype
        self._keyss = keys_seq
        self._lb = lb
        self._ub = ub
        self._name = name
        self._name_str = None

    @property
    def index(self):
        return self._index

    def copy(self, target_model):
        copied_ctn = self.__class__(self.vartype, self._keyss, self.lb, self.ub, self._name)
        return copied_ctn

    def copy_relaxed(self, target_model):
        copied_ctn = self.__class__(target_model.continuous_vartype, self._keyss, self.lb, self.ub, self._name)
        return copied_ctn

    def keys(self):
        return self._keyss

    @property
    def vartype(self):
        return self._vartype

    @property
    def nb_dimensions(self):
        return len(self._keyss)

    @property
    def namer(self):
        return self._name

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    @property
    def name(self):
        """
        Try to extract a name string from the initial container name.
        handles strings with or without formats, arrays, function.

        :return: A string.
        """
        return self._lazy_compute_name_string()

    def iter_keys(self):
        if 1 == self.nb_dimensions:
            return iter(self._keyss[0])
        else:
            return product(*self._keyss)

    def _lazy_compute_name_string(self):
        if self._name_str is not None:
            return self._name_str
        else:
            raw_name = self._name
            if is_string(raw_name):
                # drop opl-style formats
                s_name = raw_name.replace("({%s})", "")
                # purge fields
                pos_pct = raw_name.find('%')
                if pos_pct >= 0:
                    s_name = raw_name[:pos_pct - 1]
                elif raw_name.find('{') > 0:
                    pos = raw_name.find('{')
                    s_name = raw_name[:pos - 1]
            elif is_iterable(raw_name):
                from os.path import commonprefix
                s_name = commonprefix(raw_name)
            else:
                # try a function
                from os.path import commonprefix
                namefn = raw_name
                try:
                    all_names = [namefn(k) for k in self.iter_keys()]
                    s_name = commonprefix(all_names)

                except TypeError:
                    s_name = ''

            self._name_str = s_name
            return s_name

    def get_var_key(self, dvar):
        # INTERNAL
        # containers store expanded keys (as tuples).
        dvar_index = dvar.index
        relative_offset = dvar_index - self._index_offset
        if self.nb_dimensions == 1:
            try:
                return self._keyss[0][relative_offset]
            except IndexError:  # pragma: no cover
                return None
        else:
            return next(islice(product(*self._keyss), relative_offset, None), None)

    def shape(self):
        return tuple(len(k) for k in self._keyss)

    @property
    def dimension_string(self):
        dim_string = "".join(["[%d]" % s for s in self.shape()])
        return dim_string

    def to_string(self):
        # dvar xxx
        dim_string = self.dimension_string
        ctname = self._name or 'x'
        return "dvar {0} {1} {2}".format(self.vartype.short_name, ctname, dim_string)

    def __str__(self):
        return self.to_string()
