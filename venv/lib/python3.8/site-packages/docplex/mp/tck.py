# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore


import math

from docplex.mp.compat23 import izip
from docplex.mp.constr import AbstractConstraint, LinearConstraint,\
    LogicalConstraint, EquivalenceConstraint, IndicatorConstraint, QuadraticConstraint
from docplex.mp.error_handler import docplex_fatal
from docplex.mp.linear import Expr
from docplex.mp.dvar import Var
from docplex.mp.pwl import PwlFunction
from docplex.mp.progress import ProgressListener
from docplex.mp.utils import is_int, is_number, is_iterable, is_string, generate_constant, \
    is_ordered_sequence, is_iterator, resolve_caller_as_string
from docplex.mp.vartype import VarType
import six

_vartype_code_map = {sc().cplex_typecode: sc().short_name for sc in VarType.__subclasses__()}


def vartype_code_to_string(vartype_code):
    return _vartype_code_map.get(vartype_code, '????')


class DocplexNumericCheckerMixin(object):

    @staticmethod
    def static_validate_num1(e, checked_num=False, infinity=1e+20):
        # checks for number and truncates to 1e=20
        if not checked_num and not is_number(e):
            docplex_fatal("Expecting number, got: {0!r}".format(e))
        elif -infinity <= e <= infinity:
            return e
        elif e >= infinity:
            return infinity
        else:
            return -infinity

    @staticmethod
    def static_validate_num2(e, infinity=1e+20, context_msg=None):
        # checks for number, nans, nath.inf, and truncates to 1e+20
        if not is_number(e):
            docplex_fatal("Not a number: {}".format(e))
        elif math.isnan(e):
            msg = "NaN value found in expression"
            if context_msg is not None:
                try:
                    msg = "{0}: {1}".format(context_msg(), msg)
                except TypeError:
                    msg = "{0}: {1}".format(context_msg, msg)
            docplex_fatal(msg)
        elif math.isinf(e):
            msg = "Infinite value detected"
            if context_msg is not None:
                try:
                    msg = "{0}: {1}".format(context_msg(), msg)
                except TypeError:
                    msg = "{0}: {1}".format(context_msg, msg)
            docplex_fatal(msg)
        elif -infinity <= e <= infinity:
            return e
        elif e >= infinity:
            return infinity
        else:
            return -infinity

    @classmethod
    def typecheck_num_seq(cls, logger, seq, check_math, caller=None):
        # build a list to avoid consuming an iterator
        checked_num_list = list(seq)
        for i, x in enumerate(checked_num_list):
            def loop_caller():
                return "%s, pos %d" % (caller, i) if caller else ""

            cls.typecheck_num(logger, x, check_math, loop_caller)
        return checked_num_list

    @classmethod
    def typecheck_num(cls, logger, arg, check_math, caller=None):
        if not is_number(arg):
            caller_string = resolve_caller_as_string(caller)
            logger.fatal("{0}Expecting number, got: {1!r}", (caller_string, arg))
        elif check_math:
            if math.isnan(arg):
                caller_string = resolve_caller_as_string(caller)
                logger.fatal("{0}NaN value detected", (caller_string,))
            elif math.isinf(arg):
                caller_string = resolve_caller_as_string(caller)
                logger.fatal("{0}Infinite value detected", (caller_string,))

    @classmethod
    def typecheck_int(cls, logger, arg, check_math, accept_negative=True, caller=None):
        if not is_number(arg):
            caller_string = resolve_caller_as_string(caller)
            logger.fatal("{0}Expecting number, got: {1!r}", (caller_string, arg))
        if check_math:
            if math.isnan(arg):
                caller_string = resolve_caller_as_string(caller)
                logger.fatal("{0}NaN value detected", (caller_string,))
            elif math.isinf(arg):
                caller_string = resolve_caller_as_string(caller)
                logger.fatal("{0}Infinite value detected", (caller_string,))
        if not is_int(arg):
            caller_string = resolve_caller_as_string(caller)
            logger.fatal("{0}Expecting integer, got: {1!r}", (caller_string, arg))
        elif not accept_negative and arg < 0:
            caller_string = resolve_caller_as_string(caller)
            logger.fatal("{0}Expecting positive integer, got: {1!r}", (caller_string, arg))


class DocplexTypeCheckerI(object):

    def typecheck_iterable(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_valid_index(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_vartype(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_var(self, obj, vartype=None):
        raise NotImplementedError  # pragma: no cover

    def typecheck_binary_var(self, obj):
        return self.typecheck_var(obj, vartype='B')

    def typecheck_continuous_var(self, obj):
        return self.typecheck_var(obj, vartype='C')

    def typecheck_var_seq(self, seq, vtype=None, caller=None):
        return seq  # pragma: no cover

    def typecheck_logical_op_seq(self, seq):
        return seq  # pragma: no cover

    def typecheck_logical_op(self, arg, caller):
        raise NotImplementedError  # pragma: no cover

    def typecheck_var_seq_all_different(self, seq):
        raise NotImplementedError  # pragma: no cover

    def typecheck_num_seq(self, seq, caller=None):
        raise NotImplementedError  # pragma: no cover

    def typecheck_operand(self, obj, accept_numbers=True, caller=None):
        raise NotImplementedError  # pragma: no cover

    def typecheck_constraint(self, obj):
        raise NotImplementedError  # pragma: no cover

    def typecheck_ct_to_add(self, ct, mdl, caller):
        raise NotImplementedError  # pragma: no cover

    def typecheck_ct_not_added(self, ct, do_raise=False, caller=None):
        raise NotImplementedError  # pragma: no cover

    def typecheck_cts_added_to_model(self, mdl, cts, caller=None):
        return cts  # pragma: no cover

    def typecheck_linear_constraint(self, obj, accept_ranges=True):
        raise NotImplementedError  # pragma: no cover

    def typecheck_constraint_seq(self, cts, check_linear=False, accept_range=True):
        # must return sequence unchanged
        return cts  # pragma: no cover

    def typecheck_logical_constraint_seq(self, cts, true_if_equivalence):
        # must return sequence unchanged
        return cts  # pragma: no cover

    def typecheck_quadratic_constraint_seq(self, cts):
        # must return sequence unchanged
        return cts  # pragma: no cover

    def typecheck_linear_constraint_name_tuple_seq(self, ct_ctname_seq, accept_range=True):
        # must return sequence unchanged
        return ct_ctname_seq  # pragma: no cover

    def typecheck_zero_or_one(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_num(self, arg, caller=None):
        raise NotImplementedError  # pragma: no cover

    def typecheck_int(self, arg, accept_negative=False, caller=None):
        raise NotImplementedError  # pragma: no cover

    def check_vars_domain(self, lbs, ubs, names):
        raise NotImplementedError  # pragma: no cover

    def check_var_domain(self, lbs, ubs, names):
        raise NotImplementedError  # pragma: no cover

    def typecheck_string(self, arg, accept_empty=False, accept_none=False, caller=''):
        raise NotImplementedError  # pragma: no cover

    def typecheck_string_seq(self, arg, accept_empty=False, accept_none=False, caller=''):
        raise NotImplementedError  # pragma: no cover

    def typecheck_in_model(self, model, mobj, caller=''):
        raise NotImplementedError  # pragma: no cover

    def typecheck_key_seq(self, keys, accept_empty_seq=False):
        raise NotImplementedError  # pragma: no cover

    def get_number_validation_fn(self):
        raise NotImplementedError  # pragma: no cover

    def typecheck_progress_listener(self, arg):
        raise NotImplementedError  # pragma: no cover

    def typecheck_two_in_model(self, model, obj1, obj2, ctx_msg):
        raise NotImplementedError  # pragma: no cover

    def check_ordered_sequence(self, arg, caller, accept_iterator=True):
        raise NotImplementedError  # pragma: no cover

    def check_trivial_constraints(self):
        raise NotImplementedError  # pragma: no cover

    def check_solution_hook(self, mdl, sol_hook_fn):
        raise NotImplementedError

    def typecheck_pwl_function(self, pwl):
        raise NotImplementedError

    def check_duplicate_name(self, name, name_table, qualifier):
        raise NotImplementedError

    def check_for_duplicate_keys(self, keys, caller=None):
        # default is no-op
        pass


# noinspection PyAbstractClass
class DOcplexLoggerTypeChecker(DocplexTypeCheckerI):
    def __init__(self, logger):
        self._logger = logger

    def fatal(self, msg, *args):
        self._logger.fatal(msg, args)

    def error(self, msg, *args):  # pragma: no cover
        self._logger.error(msg, args)

    def warning(self, msg, *args):  # pragma: no cover
        self._logger.warning(msg, args)


class StandardTypeChecker(DOcplexLoggerTypeChecker):

    def __init__(self, logger):
        DOcplexLoggerTypeChecker.__init__(self, logger)

    @property
    def name(self):
        return "std"

    def typecheck_iterable(self, arg):
        # INTERNAL: checks for an iterable
        if not is_iterable(arg):
            self.fatal("Expecting iterable, got: {0!s}", arg)

    # safe checks.
    def typecheck_valid_index(self, arg):
        if arg < 0:
            self.fatal("Invalid index: {0!s}", arg)

    def typecheck_vartype(self, arg):
        # INTERNAL: check for a valid vartype
        if not isinstance(arg, VarType):
            self.fatal("Not a variable type: {0!s}, type: {1!s}", arg, type(arg))
        return True

    def typecheck_var(self, obj, vartype=None):
        # INTERNAL: check for Var instance
        if not isinstance(obj, Var):
            self.fatal("Expecting decision variable, got: {0!s} type: {1!s}", obj, type(obj))
        if vartype and obj.cplex_typecode != vartype:
            self.fatal("Expecting {0} variable, got: {1!s} type: {2!s}",
                       vartype_code_to_string(vartype), obj, obj.vartype)

    def typecheck_var_seq(self, seq, vtype=None, caller=None):
        # build a list to avoid consuming an iterator
        checked_var_list = list(seq)
        for i, x in enumerate(checked_var_list):
            if not isinstance(x, Var):
                caller_s = resolve_caller_as_string(caller)
                self.fatal("{2}Expecting an iterable returning variables, {0!r} was passed at position {1}", x, i, caller_s)
            if vtype and x.cplex_typecode != vtype:
                caller_s = resolve_caller_as_string(caller)
                self.fatal("{3}Expecting an iterable returning variables of type {0}, {1!r} was passed at position {2}",
                           vtype.short_name, x, i, caller_s)

        return checked_var_list

    def typecheck_logical_op_seq(self, seq, caller=None):
        checked_args = list(seq)
        for i, x in enumerate(checked_args):
            if caller is None:
                loop_caller = None
            else:
                def loop_caller():
                    return '%s, arg#%d' % (resolve_caller_as_string(caller, sep=''), i)
            self.typecheck_logical_op(x, caller=loop_caller)
        return checked_args

    def typecheck_logical_op(self, arg, caller):
        if not hasattr(arg, 'as_logical_operand') or arg.as_logical_operand() is None:
            caller_s = resolve_caller_as_string(caller)
            self.fatal('{1}Not a logical operand: {0!r}. Expecting binary variable, logical expression', arg, caller_s)

    def typecheck_num_seq(self, seq, caller=None):
        return DocplexNumericCheckerMixin.typecheck_num_seq(self._logger, seq, check_math=False, caller=caller)

    def typecheck_var_seq_all_different(self, seq):
        # return the checked sequence, so take the list
        seq_as_list = list(seq)
        for v in seq_as_list:
            self.typecheck_var(v)
        # check for all differemt and output a justifier variable apperaing twice.
        inc_set = set([])
        for v in seq_as_list:
            if v.index in inc_set:
                self.fatal('Variable: {0} appears twice in sequence', v)

            else:
                inc_set.add(v.index)
        return seq_as_list

    def typecheck_constraint(self, obj):
        if not isinstance(obj, AbstractConstraint):
            self.fatal("Expecting constraint, got: {0!s} with type: {1!s}", obj, type(obj))

    def typecheck_ct_to_add(self, ct, mdl, caller):
        if not isinstance(ct, AbstractConstraint):
            self.fatal("Expecting constraint, got: {0!r} with type: {1!s}", ct, type(ct))
        self.typecheck_in_model(mdl, ct, caller)

    def typecheck_ct_not_added(self, ct, do_raise=False, caller=None):
        if ct.is_added():
            s_caller = resolve_caller_as_string(caller, sep=' ')
            if do_raise:
                self.fatal('{0}expects a non-added constraint, {1} is added (index={2})',
                           s_caller, ct, ct.index
                           )
            else:
                self.warning('{0}expects a non-added constraint, {1} is added (index={2})',
                             s_caller, ct, ct.index
                             )

    def typecheck_cts_added_to_model(self, mdl, cts, caller=None):
        lcts = list(cts)
        for ct in lcts:
            if not ct.is_added():
                s_caller = resolve_caller_as_string(caller, sep=' ')
                mdl.fatal("{0}Constraint: {1!s} has not been added to any model".format(s_caller, ct))
            elif mdl is not ct.model:
                s_caller = resolve_caller_as_string(caller, sep=' ')
                mdl.fatal("{0}Constraint: {1!s} belongs to a different model".format(s_caller, ct))
        return lcts

    def typecheck_ct_added_to_model(self, mdl, ct, caller=None):
        if not ct.is_added():
            s_caller = resolve_caller_as_string(caller, sep=' ')
            mdl.fatal("{0}Constraint: {1!s} has not been added to any model".format(s_caller, ct))
        elif mdl is not ct.model:
            s_caller = resolve_caller_as_string(caller, sep=' ')
            mdl.fatal("{0}Constraint: {1!s} belongs to a different model".format(s_caller, ct))

    def typecheck_linear_constraint(self, obj, accept_range=True):
        if accept_range:
            if not isinstance(obj, AbstractConstraint):
                self.fatal("Expecting linear constraint, got: {0!r}", obj)
            if not obj.is_linear():
                self.fatal("Expecting linear constraint, got: {0!s} with type: {1!s}", obj, type(obj))
        else:
            if not isinstance(obj, LinearConstraint):
                self.fatal("Expecting linear constraint, got: {0!s} with type: {1!s}", obj, type(obj))

    def typecheck_constraint_seq(self, cts, check_linear=False, accept_range=True):
        checked_cts_list = list(cts)
        for i, ct in enumerate(checked_cts_list):
            if not isinstance(ct, AbstractConstraint):
                self.fatal("Expecting sequence of constraints, got: {0!r} at position {1}", ct, i)
            if check_linear:
                if not ct.is_linear():
                    self.fatal("Expecting sequence of linear constraints, got: {0!r} at position {1}", ct, i)
                elif not accept_range and not isinstance(ct, LinearConstraint):
                    self.fatal("Expecting sequence of linear constraints (not ranges), got: {0!r} at position {1}", ct,
                               i)
        return checked_cts_list

    def typecheck_logical_constraint_seq(self, cts, true_if_equivalence):
        checked_cts_list = list(cts)  # listify to avoid consuming iterators....
        if true_if_equivalence:
            checked_type = EquivalenceConstraint
            typename = "equivalence"
        elif true_if_equivalence is False:
            checked_type = IndicatorConstraint
            typename = "indicator"
        else:
            checked_type = LogicalConstraint
            typename = "equivalence or indicator"
        for i, ct in enumerate(checked_cts_list):
            if not isinstance(ct, checked_type):
                self.fatal("Expecting sequence of {0} constraints, got: {1!r} at position {2}", typename, ct, i)

        return checked_cts_list

    def typecheck_quadratic_constraint_seq(self, cts):
        checked_qcts_list = list(cts)  # listify to avoid consuming iterators....
        for i, ct in enumerate(checked_qcts_list):
            if not isinstance(ct, QuadraticConstraint):
                self.fatal("Expecting sequence of quadratic constraints, got: {0!r} at position {1}", ct, i)

        return checked_qcts_list

    def typecheck_linear_constraint_name_tuple_seq(self, ct_ctname_seq, accept_range=True):
        # must return sequence unchanged
        checked_list = list(ct_ctname_seq)
        for c, (ct, ctname) in enumerate(ct_ctname_seq):
            self.typecheck_linear_constraint(ct, accept_range=accept_range)
            # noinspection PyArgumentEqualDefault
            self.typecheck_string(ctname, accept_empty=True, accept_none=False)

        return checked_list

    def typecheck_zero_or_one(self, arg):
        if arg != 0 and arg != 1:
            self.fatal("expecting 0 or 1, got: {0!s}", arg)

    def typecheck_num(self, arg, caller=None):
        if not is_number(arg):
            caller_string = "{0}: ".format(caller) if caller is not None else ""
            self.fatal("{0}Expecting number, got: {1!r}", caller_string, arg)

    def typecheck_int(self, arg, accept_negative=True, caller=None):
        caller_string = "{0}: ".format(caller) if caller is not None else ""
        if not is_number(arg):
            self.fatal("{0}Expecting number, got: {1!r}", caller_string, arg)
        elif not is_int(arg):
            self.fatal("{0}Expecting integer, got: {1!r}", caller_string, arg)
        elif not accept_negative and arg < 0:
            self.fatal("{0}Expecting positive integer, got: {1!r}", caller_string, arg)

    def check_vars_domain(self, lbs, ubs, names):
        l_ubs = len(ubs)
        l_lbs = len(lbs)
        if l_lbs and l_ubs:
            names = names or generate_constant(None, max(l_lbs, l_ubs))
            # noinspection PyArgumentList,PyArgumentList
            for lb, ub, varname in izip(lbs, ubs, names):
                self.check_var_domain(lb, ub, varname)

    def check_var_domain(self, lb, ub, varname):
        if lb is not None and ub is not None and lb > ub:
            self.fatal('Empty variable domain, name={0}, lb={1}, ub={2}'.format(varname, lb, ub))

    def typecheck_string(self, arg, accept_empty=False, accept_none=False, caller=''):
        if is_string(arg):
            if not accept_empty and 0 == len(arg):
                s_caller = resolve_caller_as_string(caller)
                self.fatal("{0}Expecting a non-empty string", s_caller)
        elif not (arg is None and accept_none):
            s_caller = resolve_caller_as_string(caller)
            self.fatal("{0}Expecting string, got: {1!r}", s_caller, arg)

    def typecheck_string_seq(self, arg, accept_empty=False, accept_none=False, caller=''):
        checked_strings = list(arg)
        # do not accept a string
        if is_string(arg):
            s_caller = resolve_caller_as_string(caller)
            self.fatal("{0}Expecting list of strings, a string was passed: '{1}'", s_caller, arg)
        for s in checked_strings:
            self.typecheck_string(s, accept_empty=accept_empty, accept_none=accept_none, caller=caller)
        return checked_strings

    def typecheck_in_model(self, model, mobj, caller=''):
        # produces message of the type: "constraint ... does not belong to model
        if mobj.model != model:
            self.fatal("{0} ({2!s}) is not in model '{1:s}'".format(caller, model.name, mobj))

    def typecheck_key_seq(self, keys, accept_empty_seq=False):
        if any(k is None for k in keys):
            self.fatal("Variable keys cannot be None, got: {0!r}", keys)

    def get_number_validation_fn(self):
        return DocplexNumericCheckerMixin.static_validate_num1

    @staticmethod
    def _is_operand(arg, accept_numbers=True):
        return isinstance(arg, (Expr, Var)) or (accept_numbers and is_number(arg))

    def typecheck_operand(self, arg, accept_numbers=True, caller=None):
        if not self._is_operand(arg, accept_numbers=accept_numbers):
            caller_str = "{0}: ".format(caller) if caller else ""
            accept_str = "Expecting expr/var"
            if accept_numbers:
                accept_str += "/number"
            self.fatal("{0}{1}, got: {2!r}", caller_str, accept_str, arg)

    def typecheck_progress_listener(self, arg):
        if not isinstance(arg, ProgressListener):
            self.fatal('Expecting ProgressListener instance, got: {0!r}', arg)

    def typecheck_two_in_model(self, model, mobj1, mobj2, ctx_msg):
        mobj1_model = mobj1.model
        mobj2_model = mobj2.model
        if mobj1_model != mobj2_model:
            self.fatal("Cannot mix objects from different models in {0}. obj1={1!s}, obj2={2!s}"
                       .format(ctx_msg, mobj1, mobj2))
        elif mobj1_model != model:
            self.fatal("Objects do not belong to model {0}. obj1={1!s}, obj2={2!s}"
                       .format(self, mobj1, mobj2))

    def check_trivial_constraints(self):
        return True

    def check_ordered_sequence(self, arg, caller, accept_iterator=True):
        # in some cases, we need an ordered sequence, if not the code won't crash
        # but may do unexpected things
        if not(is_ordered_sequence(arg) or (accept_iterator and is_iterator(arg))):
            self.fatal("{0}, got: {1!s}", caller, type(arg).__name__)

    def check_solution_hook(self, mdl, sol_hook_fn):
        if not callable(sol_hook_fn):
            self.fatal('Solution hook requires a function taking a solution as argument, a non-callable was passed')
        if six.PY3:
            try:
                from inspect import signature
                hook_signature = signature(sol_hook_fn)
                nb_params = len(hook_signature.parameters)
                if nb_params != 1:
                    self.fatal(
                        'Solution hook requires a function taking a solution as argument, wrong number of arguments: {0}'
                        .format(nb_params))
            except (ImportError, TypeError):  # not a callable object or no signature
                pass

    def typecheck_pwl_function(self, pwl):
        if not isinstance(pwl, PwlFunction):
            self.fatal('Expecting piecewise-linear function, {0!r} was passed', pwl)

    def check_duplicate_name(self, name, name_table, qualifier):
        if name_table is not None:
            if name in name_table:
                self.warning("Duplicate {2} name: {0!s}, used for: {1}", name, name_table[name], qualifier)


class DummyTypeChecker(DOcplexLoggerTypeChecker):

    # noinspection PyUnusedLocal
    def __init__(self, logger):
        super(DummyTypeChecker, self).__init__(logger)

    @property
    def name(self):
        return "off"

    def typecheck_iterable(self, arg):
        pass  # pragma: no cover

    def typecheck_valid_index(self, arg):
        pass  # pragma: no cover

    def typecheck_vartype(self, arg):
        pass  # pragma: no cover

    def typecheck_var(self, obj, vartype=None):
        pass  # pragma: no cover

    def typecheck_var_seq(self, seq, vtype=None, caller=None):
        return seq  # pragma: no cover

    def typecheck_num_seq(self, seq, caller=None):
        return seq  # pragma: no cover

    def typecheck_var_seq_all_different(self, seq):
        return seq

    def typecheck_operand(self, obj, accept_numbers=True, caller=None):
        pass  # pragma: no cover

    def typecheck_constraint(self, obj):
        pass  # pragma: no cover

    def typecheck_ct_to_add(self, ct, mdl, caller):
        pass  # pragma: no cover

    def typecheck_ct_not_added(self, ct, do_raise=False, caller=None):
        pass

    def typecheck_cts_added_to_model(self, mdl, cts, caller=None):
        return cts

    def typecheck_ct_added_to_model(self, mdl, ct, caller=None):
        pass

    def typecheck_linear_constraint(self, obj, accept_range=True):
        pass  # pragma: no cover

    def typecheck_constraint_seq(self, cts, check_linear=False, accept_range=True):
        # must return sequence unchanged
        return cts  # pragma: no cover

    def typecheck_logical_constraint_seq(self, cts, true_if_equivalence):
        # must return sequence unchanged
        return cts  # pragma: no cover

    def typecheck_quadratic_constraint_seq(self, cts):
        # must return sequence unchanged
        return cts  # pragma: no cover

    def typecheck_linear_constraint_name_tuple_seq(self, ct_ctname_seq, accept_range=True):
        # must return sequence unchanged
        return ct_ctname_seq  # pragma: no cover

    def typecheck_zero_or_one(self, arg):
        pass  # pragma: no cover

    def typecheck_num(self, arg, caller=None):
        pass  # pragma: no cover

    def typecheck_int(self, arg, accept_negative=True, caller=None):
        pass  # pragma: no cover

    def check_vars_domain(self, lbs, ubs, names):
        # do nothing on variable bounds
        pass

    def check_var_domain(self, lb, ub, varname):
        pass

    def typecheck_string(self, arg, accept_empty=False, accept_none=False, caller=''):
        pass  # pragma: no cover

    def typecheck_string_seq(self, arg, accept_empty=False, accept_none=False, caller=''):
        return arg

    def typecheck_in_model(self, model, mobj, caller=''):
        pass  # pragma: no cover

    def typecheck_key_seq(self, keys, accept_empty_seq=False):
        pass  # pragma: no cover

    def typecheck_progress_listener(self, arg):
        pass  # pragma: no cover

    def typecheck_two_in_model(self, model, obj1, obj2, ctx_msg):
        pass  # pragma: no cover

    def check_ordered_sequence(self, arg, caller, accept_iterator=True):
        pass  # pragma: no cover

    def check_trivial_constraints(self):
        return False

    def get_number_validation_fn(self):
        return None

    def check_solution_hook(self, mdl, sol_hook_fn):
        pass

    def typecheck_pwl_function(self, pwl):
        pass

    def check_duplicate_name(self, name, name_table, qualifier):
        pass

    def typecheck_logical_op(self, arg, caller):
        pass


class NumericTypeChecker(DummyTypeChecker):

    def __init__(self, logger):
        super(NumericTypeChecker, self).__init__(logger)

    @property
    def name(self):
        return "numeric"

    def get_number_validation_fn(self):
        return DocplexNumericCheckerMixin.static_validate_num2

    def typecheck_num(self, arg, caller=None):
        DocplexNumericCheckerMixin.typecheck_num(self._logger, arg, check_math=True, caller=caller)

    def typecheck_int(self, arg, accept_negative=True, caller=None):
        DocplexNumericCheckerMixin.typecheck_int(self._logger, arg, check_math=True, accept_negative=accept_negative,
                                                 caller=caller)

    def typecheck_num_seq(self, seq, caller=None):
        return DocplexNumericCheckerMixin.typecheck_num_seq(self._logger, seq, check_math=True, caller=caller)


class FullTypeChecker(StandardTypeChecker):

    def __init__(self, logger):
        super(FullTypeChecker, self).__init__(logger)

    @property
    def name(self):
        return "full"

    def get_number_validation_fn(self):
        return DocplexNumericCheckerMixin.static_validate_num2

    def typecheck_num(self, arg, caller=None):
        DocplexNumericCheckerMixin.typecheck_num(self._logger, arg, check_math=True, caller=caller)

    def typecheck_int(self, arg, accept_negative=True, caller=None):
        DocplexNumericCheckerMixin.typecheck_int(self._logger, arg, check_math=True, accept_negative=accept_negative,
                                                 caller=caller)

    def typecheck_num_seq(self, seq, caller=None):
        return DocplexNumericCheckerMixin.typecheck_num_seq(self._logger, seq, check_math=True, caller=caller)

    def check_for_duplicate_keys(self, keys, caller=None):
        key_set = set(keys)
        if len(key_set) < len(keys):
            # some key is duplicated:
            inc_set = set()
            for k in keys:
                if k in inc_set:
                    s_caller = resolve_caller_as_string(caller, sep=' ')
                    self.fatal("{0}Duplicated key: {1!s}".format(s_caller, k))
                else:
                    inc_set.add(k)


#  ------------------------------
# noinspection PyPep8
_tck_map = {'default': StandardTypeChecker,
            'standard': StandardTypeChecker,
            'std': StandardTypeChecker,
            'on': StandardTypeChecker,
            # --
            'numeric': NumericTypeChecker,
            'full': FullTypeChecker,
            # --
            'off': DummyTypeChecker,
            'deploy': DummyTypeChecker,
            'no_checks': DummyTypeChecker}


def get_typechecker(arg, logger):
    if arg:
        key = arg.lower()
        if key in _tck_map:
            checker_type = _tck_map[key]
        else:
            msg = 'Unexpected typechecker key: {0} - expecting on|off|std|default|numeric|full. Using default'.format(
                key)
            if logger:
                logger.error(msg)
            else:
                print('*Warning: {0}'.format(msg))
            checker_type = StandardTypeChecker

    else:
        checker_type = StandardTypeChecker
    return checker_type(logger)
