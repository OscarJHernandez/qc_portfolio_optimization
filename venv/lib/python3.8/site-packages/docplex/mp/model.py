# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2018
# --------------------------------------------------------------------------

# pylint: disable=too-many-lines
from __future__ import print_function

import os
import sys
import warnings
import six
from six import itervalues, iteritems

from docplex.mp.aggregator import ModelAggregator
from docplex.mp.compat23 import StringIO, izip
from docplex.mp.constants import SOSType, CplexScope, ObjectiveSense, BasisStatus, EffortLevel,\
    int_probtype_to_string
from docplex.mp.constr import AbstractConstraint, LinearConstraint, RangeConstraint, \
    IndicatorConstraint, QuadraticConstraint, PwlConstraint, EquivalenceConstraint
from docplex.mp.context import Context, has_credentials, OverridenOutputContext
from docplex.mp.cloudutils import is_url_valid

# from docplex.mp.docloud_engine import DOcloudEngine
from docplex.mp.engine_factory import EngineFactory
from docplex.mp.environment import Environment
from docplex.mp.error_handler import DefaultErrorHandler, \
    docplex_add_trivial_infeasible_ct
from docplex.mp.format import parse_format
from docplex.mp.lp_printer import LPModelPrinter
from docplex.mp.mfactory import ModelFactory
from docplex.mp.model_stats import ModelStatistics
from docplex.mp.numutils import round_nearest_towards_infinity1, _NumPrinter

from docplex.mp.pwl import PwlFunction
from docplex.mp.tck import get_typechecker
from docplex.mp.sttck import StaticTypeChecker
from docplex.mp.utils import DOcplexException, MultiObjective,\
    DOcplexLimitsExceeded
from docplex.mp.utils import is_indexable, is_iterable, is_int, is_string, \
    make_output_path2, generate_constant, _AutomaticSymbolGenerator, _IndexScope, _to_list, \
    compute_is_index, is_number, str_maxed, normalize_basename, izip2_filled
from docplex.mp.utils import apply_thread_limitations
from docplex.mp.vartype import VarType, BinaryVarType, IntegerVarType, \
    ContinuousVarType, SemiContinuousVarType, SemiIntegerVarType
from docplex.util.environment import get_environment

from docplex.mp.cloudutils import context_must_use_docloud, context_has_docloud_credentials



from docplex.mp.solve_env import CplexLocalSolveEnv, DocloudSolveEnv


# noinspection PyProtectedMember
class Model(object):
    """ This is the main class to embed modeling objects.

    The :class:`Model` class acts as a factory to create optimization objects,
    decision variables, and constraints.
    It provides various accessors and iterators to the modeling objects.
    It also manages solving operations and solution management.

    The Model class is a context manager and can be used with the Python `with` statement:

    .. code-block:: python

       with Model() as mdl:
         # start modeling...

    When the `with` block is finished, the :func:`end` method is called automatically, and all resources
    allocated by the model are destroyed.

    When a model is created without a specified ``context``, a default
    ``Context`` is created and initialized as described in :func:`docplex.mp.context.Context.read_settings`.

    Example::

        # Creates a model named 'my model' with default context
        model = Model('my model')

    In this example, we create a model to solve with just 2 threads::

        context = Context.make_default_context()
        context.cplex_parameters.threads = 2
        model = Model(context=context)

    Alternatively, this can be coded as::

        model = Model()
        model.context.cplex_parameters.threads = 2

    Args:
        name (optional): The name of the model.
        context (optional): The solve context to be used. If no ``context`` is
            passed, a default context is created.
        log_output (optional): If ``True``, solver logs are output to
            stdout. If this is a stream, solver logs are output to that
            stream object.
        checker (optional): If ``off``, then checking is disabled everywhere. Turning off checking
            may improve performance but should be done only with extreme caution.
            Possible values for the `checker` keyword argument are:

                - `default` (or `std`, or `on`): detects modeling errors, but doe snot check
                  numerical values for infinities or NaNs. This is the default value
                - `numerical`: checks modeling errors and also checks all number values for  infinities
                  or NaNs. This option should be used when data are not trusted.
                - `off`: no typechecking is performed. This options must be used when the model has been
                  thorougly tested and numerical data are trusted.
        cts_by_name (optional): a flag which control whether the constraint name dictionary is enabled.
            Default is False.
    """

    _name_generator = _AutomaticSymbolGenerator(pattern="docplex_model", offset=1)

    _default_effort_level = EffortLevel.Repair

    @property
    def binary_vartype(self):
        """ This property returns an instance of :class:`docplex.mp.vartype.BinaryVarType`.

        This type instance is used to build all binary decision variable collections of the model.
        """
        return self._binary_vartype

    @property
    def integer_vartype(self):
        """ This property returns an instance of :class:`docplex.mp.vartype.IntegerVarType`.

        This type instance is used to build all integer variable collections of the model.
        """
        return self._integer_vartype

    @property
    def continuous_vartype(self):
        """ This property returns an instance of :class:`docplex.mp.vartype.ContinuousVarType`.

        This type instance is used to build all continuous variable collections of the model.
        """
        return self._continuous_vartype

    @property
    def semicontinuous_vartype(self):
        """ This property returns an instance of :class:`docplex.mp.vartype.SemiContinuousVarType`.

        This type instance is used to build all semi-continuous variable collections of the model.
        """
        return self._semicontinuous_vartype

    @property
    def semiinteger_vartype(self):
        """ This property returns an instance of :class:`docplex.mp.vartype.SemiIntegerType`.

        This type instance is used to build all semi-integer variable collections of the model.
        """
        return self._semiinteger_vartype

    def _vartypes(self):
        return [self._binary_vartype, self._integer_vartype, self._continuous_vartype,
                self._semicontinuous_vartype, self._semiinteger_vartype]

    def _parse_vartype(self, arg):
        if isinstance(arg, VarType):
            return arg
        else:
            self._checker.typecheck_string(arg, accept_empty=False, accept_none=False)
            argl = arg.lower()
            for vt in iter(self._vartypes()):
                if argl == vt.short_name.lower() or argl == vt.cplex_typecode.lower():
                    return vt
            self.fatal("Cannot convert as a variable type: {0!r}", arg)

    def _make_environment(self):
        env = Environment.get_default_env()
        # rtc-28869
        env.numpy_hook = Model.init_numpy
        return env

    def _lazy_get_environment(self):
        if self._environment is None:
            self._environment = self._make_environment()  # pragma: no cover
        return self._environment

    _saved_numpy_options = None

    _unknown_status = None

    @staticmethod
    def init_numpy():
        """ Static method to customize `numpy` for DOcplex.

        This method makes `numpy` aware of DOcplex.
        All `numpy` arrays with DOcplex objects will be printed by their string representations
        as returned by `str(`) instead of `repr()` as with standard numpy behavior.

        All customizations can be removed by calling the :func:`restore_numpy` method.

        Note:
            This method does nothing if `numpy` is not present.

        See Also:
            :func:`restore_numpy`
        """
        try:
            # noinspection PyUnresolvedReferences
            import numpy as np

            Model._saved_numpy_options = np.get_printoptions()
            np.set_printoptions(formatter={'numpystr': Model._numpy_print_str, 'object': Model._numpy_print_str})
        except ImportError:  # pragma: no cover
            pass  # pragma: no cover

    @staticmethod
    def _numpy_print_str(arg):
        return str(arg) if ModelFactory._is_operand(arg) else repr(arg)

    @staticmethod
    def restore_numpy():  # pragma: no cover
        """ Static method to restore `numpy` to its default state.

        This method is a companion method to :func:`init_numpy`. It restores `numpy` to its original state,
        undoing all customizations that were done for DOcplex.

        Note:
            This method does nothing if numpy is not present.

        See Also:
            :func:`init_numpy`
        """
        try:
            # noinspection PyUnresolvedReferences
            import numpy as np

            if Model._saved_numpy_options is not None:
                np.set_printoptions(Model._saved_numpy_options)
        except ImportError:  # pragma: no cover
            pass  # pragma: no cover

    @property
    def environment(self):
        # for a closed model with no CPLEX, numpy, etc return ClosedEnvironment
        # return get_no_graphics_env()
        # from docplex.environment import ClosedEnvironment
        # return ClosedEnvironment
        return self._lazy_get_environment()

    # ---- type checking

    def _typecheck_var(self, obj):
        self._checker.typecheck_var(obj)

    def _typecheck_num(self, arg, caller=None):
        self._checker.typecheck_num(arg, caller)

    def _typecheck_as_denominator(self, denominator, numerator):
        StaticTypeChecker.typecheck_as_denominator(self, denominator, numerator)

    def _typecheck_optional_num_seq(self, nums, accept_none=True, expected_size=None, caller=None):
        return StaticTypeChecker.typecheck_optional_num_seq(self, nums, accept_none, expected_size, caller)


    # ---
    def unsupported_relational_operator_error(self, left_arg, op, right_arg):
        # INTERNAL
        self.fatal("Unsupported relational operator:  {0!s} {1!s} {2!s}, only <=, ==, >= are allowed", left_arg, op,
                   right_arg)

    def cannot_be_used_as_denominator_error(self, denominator, numerator):
        StaticTypeChecker.cannot_be_used_as_denominator_error(self, denominator, numerator)

    def unsupported_power_error(self, e, power):
        self.fatal("Cannot raise {0!s} to the power {1}. A variable's exponent must be 0, 1 or 2.", e, power)

    def _parse_kwargs(self, kwargs):
        # parse some arguments from kwargs
        for arg_name, arg_val in six.iteritems(kwargs):
            if arg_name == "float_precision":
                self.float_precision = arg_val
            elif arg_name in frozenset({'keep_ordering', 'ordering'}):
                self._keep_ordering = bool(arg_val)
            elif arg_name in frozenset({"info_level", "output_level"}):
                self.output_level = arg_val
            elif arg_name in {"agent", "solver_agent"}:
                self.context.solver.agent = arg_val
            elif arg_name == "log_output":
                self.context.solver.log_output = arg_val
            elif arg_name == "warn_trivial":
                self._trivial_cts_message_level = arg_val
            elif arg_name == "max_repr_len":
                self._max_repr_len = int(arg_val)
            elif arg_name == "keep_all_exprs":
                self._keep_all_exprs = bool(arg_val)
            elif arg_name == 'checker':
                self._checker_key = arg_val.lower() if is_string(arg_val) else 'default'
            elif arg_name == 'full_obj':
                self._print_full_obj = bool(arg_val)
            elif arg_name == 'lp_line_size':
                self._lp_line_length = int(arg_val)
            elif arg_name == 'ignore_names':
                self._ignore_names = bool(arg_val)
            elif arg_name == 'clean_before_solve':
                self.clean_before_solve = arg_val
            elif arg_name == 'quality_metrics':
                self._quality_metrics = bool(arg_val)
            elif arg_name in frozenset({'url', 'key'}):
                # these two are known, no need to rant
                pass
            elif arg_name in frozenset({"parameters", 'cplex_parameters'}):
                # update parameters either from a params object or a dict
                self.context.update_cplex_parameters(arg_val)
            elif arg_name == 'cts_by_name':
                # safe
                pass
            else:
                self.warning("keyword argument: {0:s}={1!s} - is not recognized (ignored)", arg_name, arg_val)

    def _get_kwargs(self):
        kwargs_map = {'float_precision': self.float_precision,
                      'keep_ordering': self.keep_ordering,
                      'output_level': self.output_level,
                      'solver_agent': self.solver_agent,
                      'log_output': self.log_output,
                      'warn_trivial': self._trivial_cts_message_level,
                      'max_repr_len': self._max_repr_len,
                      'keep_all_exprs': self._keep_all_exprs,
                      'checker': self._checker_key,
                      'full_obj': self._print_full_obj,
                      'lp_line_size': self._lp_line_length,
                      'ignore_names': self._ignore_names,
                      'clean_before_solve': self._clean_before_solve
                      }
        return kwargs_map

    warn_trivial_feasible = 0
    warn_trivial_infeasible = 1
    warn_trivial_none = 2

    def __init__(self, name=None, context=None, **kwargs):
        """Init a new Model.

        Args:
            name (optional): The name of the model
            context (optional): The solve context to be used. If no ``context`` is
                passed, a default context is created.
            log_output (optional): if ``True``, solver logs are output to
                stdout. If this is a stream, solver logs are output to that
                stream object.
        """
        if name is None:
            name = Model._name_generator.new_symbol()
        self._name = name
        self._provenance = None

        self._error_handler = DefaultErrorHandler(output_level='warning')

        # type instances
        self._binary_vartype = BinaryVarType()
        self._integer_vartype = IntegerVarType()
        self._continuous_vartype = ContinuousVarType()
        self._semicontinuous_vartype = SemiContinuousVarType()
        self._semiinteger_vartype = SemiIntegerVarType()

        #
        self._container_map = {}
        self._origin_map = {}
        self.__vars_by_name = {}
        self._cts_by_name = None
        self.__allpwlfuncs = []
        self._benders_annotations = None
        self._constraint_priority_dict = {}

        self._allsos = []

        self._lazy_constraints = []
        self._user_cuts = []

        self._pwl_counter = {}

        # -- kpis --
        self._allkpis = []

        self._progress_listeners = []
        self._mipstarts = []

        # by default, ignore_names is off
        self._ignore_names = False

        # clean engine before solve (mip starts)
        self._clean_before_solve = False  # default is False: faster

        # expression ordering
        self._keep_ordering = False

        # -- float formats
        self._float_precision = 3
        self._float_meta_format = '{%d:.3f}'
        self._num_printer = _NumPrinter(self._float_precision)

        self._environment = self._make_environment()
        self_env = self._environment

        # init context
        if context is None:
            self.context = Context.make_default_context(_env=self_env)
        else:
            self.context = context
            # a flag to indicate whether ot not parameters have been version-checked.
        self._synced_params = False

        self._engine_factory = EngineFactory(env=self_env)

        if 'docloud_context' in kwargs:
            warnings.warn(
                "Model construction with DOcloudContext is deprecated, use initializer with docplex.mp.context.Context instead.",
                DeprecationWarning, stacklevel=2)

        # maximum length for expression in repr strings...
        self._max_repr_len = 1e+10

        # control whether to warn about trivial constraints
        self._trivial_cts_message_level = self.warn_trivial_infeasible

        # internal
        self._keep_all_exprs = True  # use False to get fast clone...with the risk of side effects...

        # full objective lp
        self._print_full_obj = False

        # lp line size
        self._lp_line_length = 80

        # checker key
        self._checker_key = 'default'

        # quality_metrics
        self._quality_metrics = False

        # rond solution or not
        self._round_solution = True
        self._round_function = round_nearest_towards_infinity1

        # update from kwargs, before the actual inits.
        # pop cts_by name before parse kwargs
        _enable_cts_by_name = kwargs.pop('cts_by_name', False)
        # =======================================================
        # parse without cts_by_name
        self._parse_kwargs(kwargs)
        self._cts_by_name = {} if _enable_cts_by_name else None
        self._check_mip_for_mipstarts = True

        self._checker = get_typechecker(arg=self._checker_key, logger=self.logger)

        # -- scopes
        self._var_scope = _IndexScope("var", cplex_scope=CplexScope.VAR_SCOPE)
        self._linct_scope = _IndexScope("linear constraint",
                                        cplex_scope=CplexScope.LINEAR_CT_SCOPE)
        self._logical_scope = _IndexScope("logical constraint",
                                          cplex_scope=CplexScope.IND_CT_SCOPE
                                          )
        self._quadct_scope = _IndexScope("quadratic constraint",
                                         cplex_scope=CplexScope.QUAD_CT_SCOPE)
        self._pwl_scope = _IndexScope("piecewise constraint",
                                      cplex_scope=CplexScope.PWL_CT_SCOPE)

        self._scopes = [self._var_scope, self._linct_scope, self._logical_scope, self._quadct_scope, self._pwl_scope]

        self._scope_dict = {sc.cplex_scope: sc for sc in self._scopes}

        # init engine
        engine = self._make_new_engine_from_agent(self.solver_agent)
        self.__engine = engine

        self._lfactory = ModelFactory(self, engine)
        from docplex.mp.quadfact import QuadFactory
        self._qfactory = QuadFactory(self, engine)
        self._quad_count = 0
        # after parse kwargs
        self._aggregator = ModelAggregator(self._lfactory, self._qfactory)

        self._solution = None
        self._solve_details = None
        self._last_solve_status = self._unknown_status

        # all the following must be placed after an engine has been set.
        self._objective_expr = None
        self._multi_objective = MultiObjective.new_empty()

        # engine log level
        engineLogLevel = get_environment().get_parameter("oaas.engineLogLevel")
        if engineLogLevel is not None and engineLogLevel in {"FINE", "FINER", "FINEST"}:
            self.parameters.read.datacheck.set(2)

        self.set_objective(sense=self._lfactory.default_objective_sense(),
                           expr=self._new_default_objective_expr())

        model_hook_fn = self.context.model_build_hook
        if model_hook_fn:
            try:
                model_hook_fn(self)
            except Exception as me:
                print("* Error in model_build_hook: {0!s}".format(me))

    def get_name(self):
        return self._name

    @property
    def name(self):
        """ This property is used to get or set the model name.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._check_name(name)
        self._name = name

    def _check_name(self, new_name):
        self._checker.typecheck_string(arg=new_name, accept_empty=False, accept_none=False)
        if ' ' in new_name:
            self.warning("Model name contains whitespaces: |{0:s}|", new_name)

    @property
    def provenance(self):
        return self._provenance

    def _constraint_scopes(self):
        return self._scopes[1:]

    def get_ct_scope(self, cplex_ct_scope, error='raise'):
        ct_scope = self._scope_dict.get(cplex_ct_scope)
        if not ct_scope and error == 'raise':
            raise ValueError("Unexpected scope code: {0}".format(cplex_ct_scope))
        return ct_scope


    def _sync_params(self, params):
        # INTERNAL: execute only once
        if self.has_cplex():
            params.connect_model(self)
            self_env = self._environment
            self_cplex_parameters_version = self.context.cplex_parameters.cplex_version
            self_engine = self.__engine
            installed_cplex_version = self_env.cplex_version
            # installed version is different from parameters: reset all defaults
            if installed_cplex_version != self_cplex_parameters_version:  # pragma: no cover
                # cplex is more recent than parameters. must update defaults.
                self.info(
                    "reset parameter defaults, from parameter version: {0} to installed version: {1}"  # pragma: no cover
                    .format(self_cplex_parameters_version, installed_cplex_version))  # pragma: no cover
                resets = self_engine._sync_parameter_defaults_from_cplex(params)  # pragma: no cover
                if resets:
                    for p, old, new in resets:
                        if p.name != 'randomseed':  # usual practice to change randomseed at each version
                            self.info('parameter changed, name: {0}, old default: {1}, new default: {2}',
                                      p.name, old, new)

    @property
    def infinity(self):
        """ This property returns the numerical value used as the upper bound for continuous decision variables.

        Note:
            CPLEX usually sets this limit to 1e+20.
        """
        return self.__engine.get_infinity()

    def get_cplex(self, do_raise=True):
        """ Returns the instance of Cplex used by the model, if any.

        In case no local installation of CPLEX can be found, this method either raises an exception,
        if parameter `do_raise` is True, or else returns None.

        :param do_raise: An optional flag: if True, raise an exception when no Cplex instance
            is available, otherwise return None.

        :return: an instance of Cplex, or None.
        """
        return self._get_cplex(do_raise=do_raise)

    def _get_cplex(self, do_raise=True, msgfn=None):
        try:
            cpx = self.__engine.get_cplex()
            if cpx:
                return cpx

        except DOcplexException:
            pass
        if do_raise:
            if msgfn:
                raise_msg = msgfn()
            else:
                raise_msg = "CPLEX library not found - No instance of Cplex is available."
            self.fatal(raise_msg)
        else:
            return None

    @property
    def cplex(self):
        """ Returns the instance of Cplex used by the model, if any.

        In case no local installation of CPLEX can be found, this method raises an exception.,

        :return: a Cplex instance.

        *New in version 2.15*
        """
        return self.get_cplex(do_raise=True)

    def has_cplex(self):
        return self.get_cplex(do_raise=False) is not None

    def _read_cplex_file(self, name, path, extension, cpx_read_fn):
        # INTERNAL
        cpx = self._get_cplex(do_raise=True, msgfn=lambda: "CPLEX library not found, cannot read CPLEX {0} file: {1}".format(name, path))
        StaticTypeChecker.check_file(self, name=name, path=path, expected_extensions=(extension,))
        cpx_read_fn(cpx, path)

    @property
    def cplex_matrix_stats(self):
        cpx = self._get_cplex(do_raise=True, msgfn=lambda: "Model.cplex_matrix_stats requires cplex")
        return cpx.get_stats()

    def read_basis_file(self, bas_path):
        """ Read a CPLEX basis status file.

        This method requires that the CPLEX library is installed.

        :param bas_path: the path of a basis file (extension is '.bas')

        *New in version 2.10*
        """
        self._read_cplex_file(name='basis', path=bas_path,
                              extension='.bas',
                              cpx_read_fn=lambda cpx_, path_: cpx_.start.read_basis(path_))


    def read_priority_order_file(self, ord_path):
        """ Read a CPLEX priority order file.

        This method requires that the CPLEX library is installed.

        :param ord_path: the path of a priority order file (extension is '.ord')

        *New in version 2.10*
        """
        self._read_cplex_file(name='priority order', path=ord_path,
                              extension='.ord',
                              cpx_read_fn=lambda cpx_, path_: cpx_.order.read(path_))

    def export_priority_order_file(self, ord_path):
        """ Exports priority order fi a CPLEX priority order file.

        This method requires that the CPLEX library is installed.

        :param ord_path: the path of a priority order file (extension is '.ord')

        *New in version 2.10*
        """
        self._read_cplex_file(name='priority order', path=ord_path,
                              extension='.ord',
                              cpx_read_fn=lambda cpx_, path_: cpx_.order.read(path_))

    # adjust the maximum length of repr.. strings
    @property
    def max_repr_len(self):
        return self._max_repr_len

    @max_repr_len.setter
    def max_repr_len(self, max_repr):
        self._max_repr_len = max_repr

    @property
    def keep_ordering(self):
        return self._keep_ordering

    @keep_ordering.setter
    def keep_ordering(self, ordered):  # pragma: no cover
        # INTERNAL
        b_ordered = bool(ordered)
        self._keep_ordering = b_ordered
        self._lfactory.update_ordering(b_ordered)
        self._qfactory.update_ordering(b_ordered)

    @property
    def ignore_names(self):
        """ This property is used to ignore all names in the model.

         This flag indicates whether names are used or not.
         When set to True, all names are ignored. This could lead to performance 
         improvements when building large models.
         The default value of this flag is False. To change its value, add it as
         keyword argument when creating the Model instance as in:

            >>> m = Model(name="my_model", ignore_names=True)

        Note:
            Once a model instance has been created with `ignore_names=True`, there is no way to restore its names.
            This flag only allows to enable or disable name generation while building the model.
         """
        return self._ignore_names

    @property
    def float_precision(self):
        """ This property is used to get or set the float precision of the model.

        The float precision is an integer number of digits, used
        in printing the solution and objective.
        This number of digits is used for variables and expressions which are not discrete.
        Discrete variables and objectives are printed with no decimal digits.

        """
        return self._float_precision

    @float_precision.setter
    def float_precision(self, nb_digits):
        used_digits = nb_digits
        if nb_digits < 0:
            self.warning("Negative float precision given: {0}, using 0 instead", nb_digits)
            used_digits = 0
        else:
            max_digits = self.environment.max_nb_digits
            bitness = self.environment.bitness
            if nb_digits > max_digits:
                self.warning("Given precision of {0:d} goes beyond {1:d}-bit capability, using maximum: {2:d}".
                             format(nb_digits, bitness, max_digits))
                used_digits = max_digits
        self._float_precision = used_digits
        # recompute float format
        self._float_meta_format = '{%%d:.%df}' % nb_digits
        self._num_printer.precision = nb_digits

    @property
    def quality_metrics(self):
        """ This flag controls whether CPLEX quality metrics are stored into the solve details.
         The default is not to store quality metrics.

         *New in version 2.10*
        """
        return self._quality_metrics

    @quality_metrics.setter
    def quality_metrics(self, use_metrics):
        self._quality_metrics = use_metrics

    @property
    def clean_before_solve(self):
        return self._clean_before_solve

    @clean_before_solve.setter
    def clean_before_solve(self, must_clean):
        self._clean_before_solve = bool(must_clean)

    @property
    def round_solution(self):
        """ This flag controls whether integer and discrete variable values are rounded in solutions, or not.
            If not rounded, it may happen that solution value for a binary variable returns 0.99999.
         The default is to round discrete values.

         *New in version 2.15*
        """
        return self._round_solution

    @round_solution.setter
    def round_solution(self, do_round):
        self._round_solution = bool(do_round)

    def _round_element_value_if_necessary(self, elt, elt_value):
        # INTERNAL
        if self.round_solution and elt_value and elt.is_discrete() and elt_value != int(elt_value):
            return self._round_function(elt_value)
        else:
            return elt_value

    def has_cts_by_name_dict(self):
        return self._cts_by_name is not None

    def enable_cts_by_name_dict(self):
        self._ensure_cts_name_dir()

    def solved_stopped_by_limit(self):
        sd = self.solve_details
        return sd and sd.has_hit_limit()

    @property
    def time_limit(self):
        """ This property is used to get/set the time limit for this model.
        """
        return self.time_limit_parameter.get()

    @time_limit.setter
    def time_limit(self, new_time_limit):
        self.set_time_limit(new_time_limit)

    @property
    def time_limit_parameter(self):
        # INTERNAL
        return self.parameters.timelimit

    def get_time_limit(self):
        """
        Returns:
            The time limit for the model.

        """
        return self.time_limit_parameter.get()

    def set_time_limit(self, time_limit):
        """ Set a time limit for solve operations.

        Args:
            time_limit: The new time limit; must be a positive number.

        """
        self._checker.typecheck_num(time_limit)
        if time_limit < 0:
            self.fatal("Negative time limit: {0}", time_limit)
        elif time_limit < 1:
            self.warning("Time limit too small: {0} - using 1 instead", time_limit)
            time_limit = 1
        else:
            pass

        self.time_limit_parameter.set(time_limit)

    @property
    def lp_line_length(self):
        """ This property lets you get or set the maximum line length of LP files generated by DOcplex.
        The default is 80.

         *New in version 2.11*

        """
        return self._lp_line_length

    @lp_line_length.setter
    def lp_line_length(self, new_length):
        if 70 <= new_length <= 512:
            self._lp_line_length = new_length
        else:
            lpz = min(128, max(new_length, 70))
            print(" LP line size set to: {0}, should be in [70..512], {1} was passed".format(lpz, new_length))
            self._lp_line_length = lpz

    @property
    def solver_agent(self):
        return self.context.solver.agent

    @property
    def error_handler(self):
        return self._error_handler

    @property
    def logger(self):
        return self._error_handler

    @property
    def solution(self):
        """ This property returns the current solution of the model or None if the model has not yet been solved
        or if the last solve has failed.
        """
        return self._solution

    def _get_solution(self):
        # INTERNAL
        return self._solution

    def new_solution(self, var_value_dict=None, objective_value=None, name=None, **kwargs):
        return self._lfactory.new_solution(var_value_dict=var_value_dict,
                                           objective_value=objective_value, name=name, **kwargs)

    def populate_solution_pool(self, **kwargs):
        """ Populates and return a solution pool.

        returns either a solutiion pool object, or None if the model solve fails.

        This method accepts the same keyword arguments as :function:`Model.solve`.
        See the documentation of :function:`Model.solve` for more details.

        :return: an instance of :class:`docplex.mp.solution.SolutionPool`, or None.

        See Also:
            :class:`docplex.mp.solution.SolutionPool`.

        *New in version 2.16*
        """
        self_mname = 'Model.populate_solution_pool'
        if not self.has_cplex():
            self.fatal("{0} requires CPLEX, but a local CPLEX installation could not be found"
                       .format(self_mname))
        pb_type = self._get_cplex_problem_type()
        if pb_type not in {'MILP'}:
            self.fatal("{0} only for MILP problems, model '{1}' is a {2}",
                       self_mname, self.name, pb_type)
        elif self.has_multi_objective():
            self.fatal("M{0}} is not available for multi-objective problems, model '{1}' has {2} objectives",
                       self_mname, self.name, self.number_of_multi_objective_exprs)

        context = self.prepare_actual_context(**kwargs)
        parameters = apply_thread_limitations(context)
        raw_params = self.context._get_raw_cplex_parameters()

        self_engine = self.__engine
        sol = None
        if raw_params and parameters is not raw_params:
            saved_params = {p: p.get() for p in raw_params}
        else:
            saved_params = {}

        log_stream = context.solver.log_output_as_stream
        with OverridenOutputContext(self, log_stream):

            used_clean_before_solve = kwargs.get('clean_before_solve', self.clean_before_solve)
            try:
                used_parameters = parameters or raw_params
                # assert used_parameters is not None
                self._apply_parameters_to_engine(used_parameters)
                sol, solnpool = self.__engine.populate(clean_before_solve=used_clean_before_solve)
                assert (sol is not None) == bool(solnpool)

            finally:
                solve_details = self_engine.get_solve_details()
                self._notify_solve_hit_limit(solve_details)
                self._solve_details = solve_details
                self._set_solution(sol)
                if saved_params:
                    for p, v in six.iteritems(saved_params):
                        self_engine.set_parameter(p, v)


        return solnpool

    populate = populate_solution_pool

    def restore_solution(self, sol, restore_all=True):
        try:
            if self != sol.model:
                self.fatal("Model.restore_solution(): Expecting solution attached to model {0}, but attached to {1}"
                           .format(self.name, sol.model.name))
            # check solution is linked to this model
            sol.restore(self, restore_all=restore_all)
        except AttributeError:
            self.fatal("Model.restore_solution(): Expecting solution, {0!r} was passed", sol)

    def fatal(self, msg, *args):
        self._error_handler.fatal(msg, args)

    def fatal_ce_limits(self, *args):
        self._error_handler.fatal_limits_exceeded()

    def error(self, msg, *args):
        self._error_handler.error(msg, args)

    def warning(self, msg, *args):
        self._error_handler.warning(msg, args)

    def info(self, msg, *args):
        self._error_handler.info(msg, args)

    @property
    def number_of_warnings(self):
        return self._error_handler.number_of_warnings

    @property
    def number_of_errors(self):
        return self._error_handler._number_of_errors


    def trace(self, msg, *args):
        self.logger.trace(msg, args)

    @property
    def output_level(self):
        return self._error_handler.get_output_level()

    @output_level.setter
    def output_level(self, new_output_level):
        self._error_handler.set_output_level(new_output_level)

    def set_quiet(self):
        self.logger.set_quiet()

    def set_log_output(self, out=None):
        self.context.solver.log_output = out
        outs = self.context.solver.log_output_as_stream
        self.__engine.set_streams(outs)

    @property
    def log_output(self):
        return self.context.solver.log_output_as_stream

    @log_output.setter
    def log_output(self, out):
        self.set_log_output(out)

    def set_log_output_as_stream(self, outs):
        self.__engine.set_streams(outs)

    def is_logged(self):
        return self.context.solver.log_output_as_stream is not None

    def clear_engine_before_solve(self):
        # INTERNAL
        try:
            self.__engine.clean_before_solve()
        except AttributeError:
            pass

    def clear(self):
        """ Clears the model of all modeling objects.
        """
        self._clear_internal()

    def _clear_internal(self, terminate=False):
        self._container_map = {}
        self._origin_map = {}
        self.__vars_by_name = {}
        self._cts_by_name = None
        self.__allpwlfuncs = []
        self._benders_annotations = None
        self._allkpis = []
        self.clear_kpis()
        self._last_solve_status = self._unknown_status
        self._solution = None
        self._mipstarts = []
        self._clear_scopes()
        self._allsos = []
        self._pwl_counter = {}
        self._lazy_constraints = []
        self._user_cuts = []
        self._quad_count = 0
        if not terminate:
            self.set_objective(sense=self._lfactory.default_objective_sense(),
                               expr=self._new_default_objective_expr())
            self._set_engine(self._make_new_engine_from_agent(self.solver_agent, self.context))
        else:
            self._terminate_engine()

    def _clear_scopes(self):
        for a_scope in self._scopes:
            a_scope.reset()

    def set_checker(self, checker_key):
        # internal
        if checker_key != self._checker_key:
            new_checker = get_typechecker(arg=checker_key, logger=self.logger)
            self._checker_key = checker_key
            self._checker = new_checker
            self._aggregator._checker = new_checker
            self._lfactory._checker = new_checker
            self._qfactory._checker = new_checker

    def _make_new_engine_from_agent(self, solver_agent, context=None):
        ctx = context or self.context
        new_engine = self._engine_factory.new_engine(solver_agent, self.environment, model=self, context=ctx)
        new_engine.set_streams(self.context.solver.log_output_as_stream)
        return new_engine

    def _set_engine(self, e2):
        # INTERNAL
        old_engine = self.get_engine()
        self.__engine = e2
        self._lfactory.update_engine(e2)
        self._qfactory.update_engine(e2)
        try:
            self._static_terminate_engine(old_engine)
        finally:
            pass

    def _terminate_engine(self):
        # INTERNAL
        old_engine = self.__engine
        self._static_terminate_engine(old_engine)
        self.__engine = None

    @classmethod
    def _static_terminate_engine(cls, engine_to_terminate):
        if engine_to_terminate is not None:
            # dispose of old engine.
            engine_to_terminate.end()
            # from Ryan
            del engine_to_terminate

    # def set_new_engine_from_agent(self, new_agent):
    #     self_context = self.context
    #     # set new agent
    #     if new_agent is None:
    #         new_agent = self_context.solver.agent
    #     elif is_string(new_agent):
    #         self_context.solver.agent = new_agent
    #     else:
    #         self.fatal('unexpected value for agent: {0!r}, expecting string or None', new_agent)
    #     new_engine = self._make_new_engine_from_agent(new_agent, self_context)
    #     self._set_engine(new_engine)

    @property
    def solves_with(self):
        return self.__engine.name


    def get_engine(self):
        # INTERNAL for testing
        return self.__engine

    def print_information(self):
        """ Prints general informational statistics on the model.

        Prints the number of variables and their breakdown by type.
        Prints the number of constraints and their breakdown by type.

        """
        print("Model: %s" % self.name)
        self.get_statistics().print_information()

        # --- annotations
        self_anno = self._benders_annotations
        if self_anno:
            anno_stats = self.get_annotation_stats()
            print(" - annotations: {0}".format(len(self_anno)))
            print("   - {0}".format(', '.join('{0}: {1}'.format(cpx.descr, anno_stats[cpx]) for cpx in CplexScope if cpx in anno_stats)))


        # --- parameters
        self_params = self.context._get_raw_cplex_parameters()
        if self_params and self_params.has_nondefaults():
            print(" - parameters:")
            self_params.print_information(indent_level=5)  # 3 for " - " + 2 = 5
        else:
            print(" - parameters: defaults")

        # ------------------- objective
        minmax = "minimize" if self.is_minimized() else "maximize"
        obj_s = minmax
        if self.has_multi_objective():
            n_objs = self.number_of_multi_objective_exprs
            obj_s = "{0} multiple[{1}]".format(minmax, n_objs)
        elif not self.is_optimized():
            obj_s = "none"
        else:
            try:
                if self.objective_expr.is_quad_expr():
                    obj_s = "{0} quadratic".format(minmax)
            except AttributeError:
                pass
        print(" - objective: {0}".format(obj_s))

        # ------------------- problem type
        cpx = self.get_cplex(do_raise=False)
        if cpx:
            cpx_probtype = cpx.get_problem_type()
            print(" - problem type is: {0}".format(int_probtype_to_string(cpx_probtype)))


    def _get_cplex_problem_type(self, fallback="unknown"):
        # INTERNAL
        cpx = self.get_cplex(do_raise=False)
        if cpx:
            cpx_probtype_code = cpx.get_problem_type()  # this is an int
            return int_probtype_to_string(cpx_probtype_code)
        else:
            return fallback

    def __notify_new_model_object(self, descr,
                                  mobj, mindex, mobj_name,
                                  name_dir, idx_scope,
                                  is_name_safe=False):
        """
        Notifies the return af an object being create on the engine.
        :param descr: A string describing the type of the object being created (e.g. Constraint, Variable).
        :param mobj:  The newly created modeling object.
        :param mindex: The index as returned by the engine.
        :param name_dir: The directory of objects by name (e.g. name -> constraint directory).
        :param idx_scope:  The index scope
        """
        mobj.index = mindex

        if name_dir is not None:
            mobj_name = mobj_name or mobj.get_name()
            if mobj_name:
                # in some cases, names are checked before register
                if not is_name_safe:
                    if mobj_name in name_dir:
                        old_name_value = name_dir[mobj_name]
                        # Duplicate constraint name: foo
                        self.warning("Duplicate {0} name: {1} already used for {2!r}", descr, mobj_name, old_name_value)

                name_dir[mobj_name] = mobj

        # store in idx dir if any
        if idx_scope:
            idx_scope.notify_obj_index(mobj, mindex)

    def _register_one_var(self, var, var_index, var_name):
        self.__notify_new_model_object("variable", var, var_index, var_name, self.__vars_by_name, self._var_scope)

    # @profile
    def _register_block_vars(self, allvars, indices, allnames):
        if allnames:
            varname_dict = self.__vars_by_name
            for var, var_index, var_name in izip(allvars, indices, allnames):
                var._index = var_index
                if var_name:
                    if var_name in varname_dict:
                        old_name_value = varname_dict[var_name]
                        # Duplicate constraint name: foo
                        self.warning("Duplicate variable name: {0} already used for {1!s}", var_name, old_name_value)
                    varname_dict[var_name] = var
        else:
            for var, var_index in izip(allvars, indices):
                var._index = var_index
        self._var_scope.notify_obj_indices(objs=allvars, indices=indices)

    def _register_one_constraint(self, ct, ct_index, is_ctname_safe=False):
        """
        INTERNAL
        :param ct: The new constraint to register.
        :param ct_index: The index as returned by the engine.
        :param is_ctname_safe: True if ct name has been checked for duplicates already.
        :return:
        """
        self.__notify_new_model_object(
            "constraint", ct, ct_index, None,
            self._cts_by_name, ct._get_index_scope(),
            is_name_safe=is_ctname_safe)

    def _ensure_cts_name_dir(self):
        # INTERNAL: make sure the constraint name dir is present.
        if self._cts_by_name is None:
            self._cts_by_name = {ct.get_name(): ct for ct in self.iter_constraints() if ct.has_user_name()}
        return self._cts_by_name

    def _register_block_cts(self, scope, cts, indices):
        # INTERNAL: assert len(cts) == len(indices)
        ct_name_map = self._cts_by_name
        # --
        if ct_name_map:
            for ct, ct_index in izip(cts, indices):
                ct.index = ct_index
                ct_name = ct.get_name()
                if ct_name:
                    ct_name_map[ct_name] = ct
        else:
            for ct, ct_index in izip(cts, indices):
                ct._index = ct_index

        scope.notify_obj_indices(cts, indices)

    def _register_implicit_equivalence_ct(self, eqct, eqctx):
        self._register_one_constraint(eqct, eqctx, is_ctname_safe=True)

    # iterators
    def iter_var_containers(self):
        # INTERNAL
        setof_ctns = set(self._container_map.values())
        sorted_ctns = sorted(setof_ctns, key=lambda ctn_: ctn_._index)
        return iter(sorted_ctns)

    def get_var_container(self, dvar):
        # INTERNAL
        return self._container_map.get(dvar)

    def set_var_container(self, dvar, ctn):
        # INTERNAL
        assert ctn is not None
        self._container_map[dvar] = ctn

    @staticmethod
    def origin_key(obj):
        # use id()here to allow case where index is not valid
        return id(obj)

    def get_obj_origin(self, obj):
        # INTERNAL: retrieve origin of object
        objkey = self.origin_key(obj)
        return self._origin_map.get(objkey)

    def set_obj_origin(self, obj, new_origin):
        # INTERNAL: set origin of object
        okey = self.origin_key(obj)
        if new_origin is not None:
            self._origin_map[okey] = new_origin
        elif obj in self._origin_map:
            del self._origin_map[okey]

    def _is_binary_var(self, dvar):
        return dvar.cplex_typecode == 'B'

    def _is_integer_var(self, dvar):
        return dvar.cplex_typecode == 'I'

    def _is_continuous_var(self, dvar):
        return dvar.cplex_typecode == 'C'

    def _is_semicontinuous_var(self, dvar):
        return dvar.cplex_typecode == 'S'

    def _is_semiinteger_var(self, dvar):
        return dvar.cplex_typecode == 'N'

    @property
    def number_of_generated_variables(self):
        return sum(1 for v in self.iter_variables() if v.is_generated())

    def _count_variables_w_code(self, cpxcode):
        cnt = 0
        for v in self.iter_variables():
            if v.cplex_typecode == cpxcode:
                cnt += 1
        return cnt

    def _iter_variables_w_code(self, cpxcode):
        for v in self.iter_variables():
            if v.cplex_typecode == cpxcode:
                yield v

    @property
    def number_of_variables(self):
        """ This property returns the total number of decision variables, all types combined.

        """
        return self._var_scope.size

    @property
    def number_of_binary_variables(self):
        """ This property returns the total number of binary decision variables added to the model.
        """
        return self._count_variables_w_code('B')

    @property
    def number_of_integer_variables(self):
        """ This property returns the total number of integer decision variables added to the model.
        """
        return self._count_variables_w_code('I')

    @property
    def number_of_continuous_variables(self):
        """ This property returns the total number of continuous decision variables added to the model.
        """
        return self._count_variables_w_code('C')

    @property
    def number_of_semicontinuous_variables(self):
        """ This property returns the total number of semi-continuous decision variables added to the model.
        """
        return self._count_variables_w_code('S')

    @property
    def number_of_semiinteger_variables(self):
        """ This property returns the total number of semi-integer decision variables added to the model.
        """
        return self._count_variables_w_code('N')

    @property
    def number_of_user_variables(self):
        return sum(1 for _ in self.generate_user_variables())

    # def _has_discrete_var(self):
    #     # INTERNAL
    #     return any(v.is_discrete for v in self.iter_variables())

    def _contains_discrete_artefacts(self):
        if hasattr(self._lfactory, "_cached_justifier_discrete_var"):
            return self._lfactory._cached_justifier_discrete_var is not None
        elif self._allsos or self._pwl_counter:
            return True
        for v in self.iter_variables():
            if v.cplex_typecode in 'IBNS':
                self._lfactory._cached_justifier_discrete_var = v
                return True
        return False

    def _clear_cached_discrete_var(self):
        lfactory = self._lfactory
        if hasattr(lfactory, "_cached_justifier_discrete_var"):
            lfactory._cached_justifier_discrete_var = None

    def _has_piecewise(self):
        return len(self._pwl_counter) > 0

    def _solved_as_mip(self):
        # INTERNAL: is the model solved as mip (incl. engine status)
        return self._contains_discrete_artefacts() or self.__engine.solved_as_mip()

    def _solved_as_lp(self):
        # INTERNAL: is the model solved as mip (incl. engine status)
        if self.has_cplex():
            return self.get_engine().solved_as_lp()
        else:
            return not self._contains_discrete_artefacts()


    def is_quadratic(self):
        # returns true if model is quadratic, that is
        # either has atleast one quadratic constraint, or has a quadrtic objective.
        if self.has_quadratic_constraint():
            return True
        elif self.has_multi_objective():
            return any(ex.is_quad_expr() for ex in self.iter_multi_objective_exprs())
        else:
            return self._objective_expr.is_quad_expr()

    def _make_new_stats(self):
        # INTERNAL
        from collections import Counter

        vartype_count = Counter(type(dv.vartype) for dv in self.iter_variables())
        nbbvs = vartype_count[BinaryVarType]
        nbivs = vartype_count[IntegerVarType]
        nbcvs = vartype_count[ContinuousVarType]
        nbscvs = vartype_count[SemiContinuousVarType]
        nbsivs = vartype_count[SemiIntegerVarType]

        linct_count = Counter(ct.sense.cplex_code for ct in self.iter_binary_constraints())
        nb_le_cts = linct_count['L']
        nb_eq_cts = linct_count['E']
        nb_ge_cts = linct_count['G']
        nb_rng_cts= self.number_of_range_constraints
        nb_ind_cts = self.number_of_indicator_constraints
        nb_equiv_cts = self.number_of_equivalence_constraints
        nb_quad_cts = self.number_of_quadratic_constraints
        stats = ModelStatistics(nbbvs, nbivs, nbcvs, nbscvs, nbsivs,
                                nb_le_cts, nb_ge_cts, nb_eq_cts, nb_rng_cts,
                                nb_ind_cts, nb_equiv_cts, nb_quad_cts)
        return stats

    @property
    def statistics(self):
        """ Returns statistics on the model.

        :returns: A new instance of :class:`docplex.mp.model_stats.ModelStatistics`.
        """
        return self._make_new_stats()

    def get_statistics(self):
        return self.statistics

    def iter_pwl_functions(self):
        """ Iterates over all the piecewise linear functions in the model.

        Returns the PWL functions in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return iter(self.__allpwlfuncs)

    def iter_variables(self):
        """ Iterates over all the variables in the model.

        Returns the variables in the order they were added to the model,
        regardless of their type.

        Returns:
            An iterator object.
        """
        return self._var_scope.iter_objects()

    def iter_binary_vars(self):
        """ Iterates over all binary decision variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_w_code('B')

    def iter_integer_vars(self):
        """ Iterates over all integer decision variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_w_code('I')

    def iter_continuous_vars(self):
        """ Iterates over all continuous decision variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_w_code('C')

    def iter_semicontinuous_vars(self):
        """ Iterates over all semi-continuous decision variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_w_code('S')

    def iter_semiinteger_vars(self):
        """ Iterates over all semi-integer decision variables in the model.

        Returns the variables in the order they were added to the model.

        Returns:
            An iterator object.
        """
        return self._iter_variables_w_code('N')


    def get_var_by_name(self, name):
        """ Searches for a variable from a name.

        Returns a variable if it finds one with exactly this name, or None.

        Args:
            name (str): The name of the variable being searched for.

        :returns: A variable (instance of :class:`docplex.mp.linear.Var`) or None.
        """
        return self.__vars_by_name.get(name, None)

    def _get_non_ambiguous_varname(self, base_name):
        # INTERNAL
        varname_map = self.__vars_by_name
        if varname_map is not None and base_name in varname_map:
            return '{0}#{1}'.format(base_name, self.number_of_variables)
        else:
            return base_name

    def generate_user_variables(self):
        # internal
        for dv in self.iter_variables():
            if not dv.is_generated():
                yield dv

    def generate_user_linear_constraints(self):
        # internal
        for lct in self.iter_linear_constraints():
            if not lct.is_generated():
                yield lct

    def find_matching_vars(self, pattern, match_case=False):
        """ Finds all variables whose name contain a given string

        This method searches for all variables whose name
        is not null and contains the passed ``pattern`` string. Anonymous variables
        are not considered.

        :param pattern: a non-empty string.
        :param match_case: optional flag to match case (or not). Default is to not match case.

        :return: A list of decision variables.
        """
        return self._find_matching_objs(self.generate_user_variables, pattern, match_case)

    def find_re_matching_vars(self, regexpr):
        """ Finds all variables whose name match a regular expression.

        This method searches for all variables with a name that
        matches the given regular expression. Anonymous variables
        are not counted as matching.

        :param regexpr: a regular expression, as define din module ``re``

        :return: A list of decision variables.

        *New in version 2.9*
        """
        matches = []
        for dv in self.generate_user_variables():
            dvname = dv.name
            if dvname and regexpr.match(dvname):
                matches.append(dv)
        return matches

    def _find_matching_objs(self, obj_iter, pattern, match_case=False):
        # internal
        self._checker.typecheck_string(pattern, accept_empty=False, accept_none=False, caller="Model.find_matching_vars")
        key_pattern = pattern if match_case else pattern.lower()
        matches = []
        for obj in obj_iter():
            obj_name = obj.name
            if obj_name:
                matched = obj_name if match_case else obj_name.lower()
                if key_pattern in matched:
                    matches.append(obj)
        return matches

    def find_matching_linear_constraints(self, pattern, match_case=False):
        """ Finds all linear constraints whose name contain a given string

        This method searches for all linear constraints whose name
        is not empty and contains the passed ``pattern`` string. Anonymous linear constraints
        are not considered.

        :param pattern: a non-empty string.
        :param match_case: optional flag to match case (or not). Default is to not match case.

        :return: A list of linear constraints.

        *New in version 2.9*
        """
        return self._find_matching_objs(self.generate_user_linear_constraints, pattern, match_case)

    def get_var_by_index(self, idx):
        # INTERNAL
        return self._var_by_index(idx)

    def _var_by_index(self, idx):
        # INTERNAL
        return self._var_scope.get_object_by_index(idx)

    def _set_var_type(self, dvar, new_vartype_):
        # INTERNAL
        new_vartype = self._parse_vartype(new_vartype_)
        self._checker.typecheck_vartype(new_vartype)
        if new_vartype != dvar.vartype:
            self.__engine.change_var_types([dvar], [new_vartype])
            # change type in the Var object.
            dvar._set_vartype_internal(new_vartype)
            self._update_var_bounds_from_type(dvar, new_vartype)
            self._clear_cached_discrete_var()
        return dvar

    def set_var_name(self, dvar, new_name):
        # INTERNAL: use var.name to set variable names
        if new_name != dvar.name:
            self.__engine.rename_var(dvar, new_name)
            dvar._set_name(new_name)

    def set_linear_constraint_name(self, linct, new_name):
        # INTERNAL: use lct.name to set a linear constraint's name
        if new_name != linct.get_name():
            self.__engine.rename_linear_constraint(linct, new_name)
            linct._set_name(new_name)

    def set_var_lb(self, var, candidate_lb):
        # INTERNAL: use var.lb to set lb
        new_lb = var.vartype.resolve_lb(candidate_lb, self)
        self._set_var_lb(var, new_lb)
        return new_lb

    def _set_var_lb(self, var, new_lb):
        # INTERNAL
        self.__engine.set_var_lb(var, new_lb)
        var._internal_set_lb(new_lb)

    def set_var_ub(self, var, candidate_ub):
        # INTERNAL: use var.ub to set ub
        new_ub = var.vartype.resolve_ub(candidate_ub, self)
        self._set_var_ub(var, new_ub)
        return new_ub

    def _set_var_ub(self, var, new_ub):
        # INTERNAL
        self.__engine.set_var_ub(var, new_ub)
        var._internal_set_ub(new_ub)


    def _update_var_bounds_from_type(self, dvar, new_vartype, force_binary01=False):
        # INTERNAL
        old_lb, old_ub = dvar.lb, dvar.ub
        if new_vartype == self.binary_vartype and force_binary01:
            new_lb, new_ub = 0, 1
        else:
            new_lb = new_vartype.resolve_lb(old_lb, logger=self)
            new_ub = new_vartype.resolve_ub(old_ub, logger=self)
        if new_lb != old_lb:
            self._set_var_lb(dvar, new_lb)
        if new_ub != old_ub:
            self._set_var_ub(dvar, new_ub)

    def get_constraint_by_name(self, name):
        """ Searches for a constraint from a name.

        Returns the constraint if it finds a constraint with exactly this name, or None
        if no constraint has this name.

        This function will not raise an exception if the named constraint is not found.

        Note:
            The constraint name dicitonary in class Model is disabled by default. However,
            calling `get_constraint_by_name` will compute one dicitonary on the fly,
            but without warning for duplicate names. To enable the constraint
            name dicitonary from the start (and get duplicate constraint messages),
            add the `cts_by_name` keyword argument when creating the model, as in

            >>> m = Model(name='my_model', cts_by_name=True)

            This enables the constraint name dicitonary, and checks for duplicates when a named
            constraint is added.

        Args:
            name (str): The name of the constraint being searched for.

        Returns:
            A constraint or None.
        """
        return self._ensure_cts_name_dir().get(name)

    def get_constraint_by_index(self, idx):
        """ Searches for a linear constraint from an index.

        Returns the linear constraint with `idx` as index, or None.
        This function will not raise an exception if no constraint with this index is found.

        Note: remember that linear constraints, logical constraints, and quadratic constraints
        each have separate index spaces.

        :param idx: a valid index (greater than 0).

        :return: A linear constraint, or None.
        """
        return self._linct_scope.get_object_by_index(idx, self._checker)

    def get_logical_constraint_by_index(self, idx):
        return self._logical_scope.get_object_by_index(idx, self._checker)

    def get_quadratic_constraint_by_index(self, idx):
        # INTERNAL
        return self._quadct_scope.get_object_by_index(idx, self._checker)

    @property
    def number_of_constraints(self):
        """ This property returns the total number of constraints that were added to the model.

        The number includes linear constraints, range constraints, and indicator constraints.
        """
        return sum(scope.size for scope in self._constraint_scopes())

    @property
    def number_of_user_constraints(self):
        """ This property returns the total number of constraints that were
        explicitly added tothe model, not including generated constraints.

        The number includes all types of constraints.
        """
        return sum(1 for ct in self.iter_constraints() if not ct.is_generated())

    def iter_constraints(self):
        """ Iterates over all constraints (linear, ranges, indicators).

        Returns:
          An iterator object over all constraints in the model.
        """
        for sc in self._constraint_scopes():
            for obj in sc.iter_objects():
                yield obj

    def _count_constraints_with_type(self, scope, cttype):
        return scope.count_filtered(filter=lambda ct: isinstance(ct, cttype))


    @property
    def number_of_range_constraints(self):
        """ This property returns the total number of range constraints added to the model.

        """
        return self._count_constraints_with_type(self._linct_scope, RangeConstraint)

    @property
    def number_of_linear_constraints(self):
        """ This property returns the total number of linear constraints added to the model.

        This counts binary linear constraints (<=, >=, or ==) and
        range constraints.

        See Also:
            :func:`number_of_range_constraints`

        """
        return self._linct_scope.size

    def iter_range_constraints(self):
        """
        Returns an iterator on the range constraints of the model.

        Returns:
            An iterator object.
        """
        return self._linct_scope.generate_objects_with_type(RangeConstraint)

    def iter_binary_constraints(self):
        """
        Returns an iterator on the binary constraints (expr1 <op> expr2) of the model.
        This does not include range constraints.

        Returns:
            An iterator object.
        """
        return self._linct_scope.generate_objects_with_type(LinearConstraint)

    def iter_linear_constraints(self):
        """
        Returns an iterator on the linear constraints of the model.
        This includes binary linear constraints and ranges but not indicators.

        Returns:
            An iterator object.
        """
        for c in self.iter_constraints():
            if c.is_linear():
                yield c


    @property
    def number_of_nonzeros(self):
        return sum(lct.size for lct in self.iter_linear_constraints())

    def iter_indicator_constraints(self):
        """ Returns an iterator on indicator constraints in the model.

        Returns:
            An iterator object.
        """
        return self._logical_scope.generate_objects_with_type(IndicatorConstraint)

    def iter_equivalence_constraints(self):
        """ Returns an iterator on equivalence constraints in the model.

        Returns:
            An iterator object.
        """
        return self._logical_scope.generate_objects_with_type(EquivalenceConstraint)

    @property
    def number_of_indicator_constraints(self):
        """ This property returns the number of indicator constraints in the model.
        """
        return self._count_constraints_with_type(self._logical_scope, IndicatorConstraint)

    @property
    def number_of_equivalence_constraints(self):
        """ This property returns the number of equivalence constraints in the model.
        """
        return self._count_constraints_with_type(self._logical_scope, EquivalenceConstraint)

    @property
    def number_of_implicit_equivalences(self):
        return sum(1 for lct in self.iter_equivalence_constraints() if lct.is_generated())

    def iter_quadratic_constraints(self):
        """
        Returns an iterator on the quadratic constraints of the model.

        Returns:
            An iterator object.
        """
        #return self.gen_constraints_with_type(QuadraticConstraint)
        return self._quadct_scope.iter_objects()

    @property
    def number_of_quadratic_constraints(self):
        """ This property returns the number of quadratic constraints in the model.
        """
        return self._quadct_scope.size

    def has_quadratic_constraint(self):
        return self._quadct_scope.size > 0

    def iter_logical_constraints(self):
        for ct in self.iter_constraints():
            if ct.is_logical():
                yield ct

    def var(self, vartype, lb=None, ub=None, name=None):
        """ Creates a decision variable and stores it in the model.

        Args:
            vartype: The type of the decision variable;
                This field expects a concrete instance of the abstract class
                :class:`docplex.mp.vartype.VarType`.

            lb: The lower bound of the variable; either a number or None, to use the default.
                 The default lower bound for all three variable types is 0.

            ub: The upper bound of the variable domain; expects either a number or None to use the type's default.
                The default upper bound for Binary is 1, otherwise positive infinity.

            name: An optional string to name the variable.

        :returns: The newly created decision variable.
        :rtype: :class:`docplex.mp.linear.Var`

        Note:
            The model holds local instances of BinaryVarType, IntegerVarType, ContinuousVarType which
            are accessible by properties (resp. binary_vartype, integer_vartype, continuous_vartype).

        See Also:
            :attr:`infinity`,
            :attr:`binary_vartype`,
            :attr:`integer_vartype`,
            :attr:`continuous_vartype`

        """
        self._checker.typecheck_vartype(vartype)
        return self._var(vartype, lb, ub, name)

    def _var(self, vartype, lb=None, ub=None, name=None):
        # INTERNAL
        if lb is not None:
            self._checker.typecheck_num(lb, caller='Var.lb')
        if ub is not None:
            self._checker.typecheck_num(ub, caller='Var.ub')
        return self._lfactory.new_var(vartype, lb, ub, name)

    def continuous_var(self, lb=None, ub=None, name=None):
        """ Creates a new continuous decision variable and stores it in the model.

        Args:
            lb: The lower bound of the variable, or None. The default is 0.
            ub: The upper bound of the variable, or None, to use the default. The default is model infinity.
            name (string): An optional name for the variable.

        :returns: A decision variable with type :class:`docplex.mp.vartype.ContinuousVarType`.
        :rtype: :class:`docplex.mp.linear.Var`
        """
        return self._var(self.continuous_vartype, lb, ub, name)

    def integer_var(self, lb=None, ub=None, name=None):
        """ Creates a new integer variable and stores it in the model.

        Args:
            lb: The lower bound of the variable, or None. The default is 0.
            ub: The upper bound of the variable, or None, to use the default. The default is model infinity.
            name: An optional name for the variable.

        :returns: An instance of the :class:`docplex.mp.linear.Var` class with type `IntegerVarType`.
        :rtype: :class:`docplex.mp.linear.Var`
        """
        return self._var(self.integer_vartype, lb, ub, name)

    def binary_var(self, name=None):
        """ Creates a new binary decision variable and stores it in the model.

        Args:
            name (string): An optional name for the variable.

        :returns: A decision variable with type :class:`docplex.mp.vartype.BinaryVarType`.
        :rtype: :class:`docplex.mp.linear.Var`
        """
        return self._var(self.binary_vartype, name=name)

    def semicontinuous_var(self, lb, ub=None, name=None):
        """ Creates a new semi-continuous decision variable and stores it in the model.

        Args:
            lb: The lower bound of the variable  (which must be strictly positive).
            ub: The upper bound of the variable, or None, to use the default. The default is model infinity.
            name (string): An optional name for the variable.

        :returns: A decision variable with type :class:`docplex.mp.vartype.SemiContinuousVarType`.
        :rtype: :class:`docplex.mp.linear.Var`
        """
        self._checker.typecheck_num(lb)  # lb cannot be None
        return self._var(self.semicontinuous_vartype, lb, ub, name)

    def semiinteger_var(self, lb, ub=None, name=None):
        """ Creates a new semi-integer decision variable and stores it in the model.

        Args:
            lb: The lower bound of the variable (which must be strictly positive).
            ub: The upper bound of the variable, or None, to use the default. The default is model infinity.
            name (string): An optional name for the variable.

        :returns: A decision variable with type :class:`docplex.mp.vartype.SemiIntegerVarType`.
        :rtype: :class:`docplex.mp.linear.Var`
        """
        self._checker.typecheck_num(lb)  # lb cannot be None
        return self._var(self.semiinteger_vartype, lb, ub, name)

    def var_list(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        self._checker.typecheck_vartype(vartype)
        return self._var_list(keys, vartype, lb, ub, name, key_format)

    def _var_list(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        return self._lfactory.var_list(keys, vartype, lb, ub, name, key_format)

    def var_dict(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        self._checker.typecheck_vartype(vartype)
        return self._var_dict(keys, vartype, lb, ub, name, key_format)

    def _var_dict(self, keys, vartype, lb=None, ub=None, name=str, key_format=None):
        return self._lfactory.new_var_dict(keys, vartype, lb, ub, name, key_format, ordered=self._keep_ordering)

    def binary_var_list(self, keys, lb=None, ub=None, name=str, key_format=None):
        """ Creates a list of binary decision variables and stores them in the model.

        Args:
            keys: Either a sequence of objects or an integer.
            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if keys is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...
                        
        Example:
            If you want each key string to be surrounded by {}, use a special key_format: "_{%s}",
            the %s denotes where the key string will be formatted and appended to `name`.

        :returns: A list of :class:`docplex.mp.linear.Var` objects with type :class:`doc.mp.vartype.BinaryVarType`.

        Example:
            `mdl.binary_var_list(3, "z")` returns a list of size 3
            containing three binary decision variables with names `z_0`, `z_1`, `z_2`.

        """
        return self._var_list(keys, self.binary_vartype, name=name, lb=lb, ub=ub, key_format=key_format)

    def integer_var_list(self, keys, lb=None, ub=None, name=str, key_format=None):
        """ Creates a list of integer decision variables with type `IntegerVarType`, stores them in the model,
        and returns the list.

        Args:
            keys: Either a sequence of objects or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.
            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers with the same size as keys,
                a function (which will be called on each key argument), or None.
            ub: Upper bounds of the variables.  Accepts either a floating-point number,
                a list of numbers with the same size as keys,
                a function (which will  be called on each key argument), or None.
            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
            key_format: A format string or None. This format string describes how keys contribute to variable names.
                The default is "_%s". For example if name is "x" and each key object is represented by a string
                like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        Note:
            Using None as the lower bound means the default lower bound (0) is used.
            Using None as the upper bound means the default upper bound (the model's positive infinity)
            is used.

        :returns: A list of :class:`doc.mp.linear.Var` objects with type `IntegerVarType`.

        """
        return self._var_list(keys, self.integer_vartype, lb, ub, name, key_format)

    def continuous_var_list(self, keys, lb=None, ub=None, name=str, key_format=None):
        """
        Creates a list of continuous decision variables with type :class:`docplex.mp.vartype.ContinuousVarType`,
        stores them in the model, and returns the list.

        Args:
            keys: Either a sequence of objects or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers
                or use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means using the default lower bound (0) is used.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers
                or use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str()` on each key object.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...
        Note:
            When `keys` is either an empty list or the integer 0, an empty list is returned.


        :returns: A list of :class:`docplex.mp.linear.Var` objects with type :class:`docplex.mp.vartype.ContinuousVarType`.

        See Also:
            :attr:`infinity`

        """
        return self._var_list(keys, self.continuous_vartype, lb, ub, name, key_format)

    def semicontinuous_var_list(self, keys, lb, ub=None, name=str, key_format=None):
        """
        Creates a list of semi-continuous decision variables with type :class:`docplex.mp.vartype.SemiContinuousVarType`,
        stores them in the model, and returns the list.

        Args:
            keys: Either a sequence of objects or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, or a function.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers or
                use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                Note that the lower bound of a semi-continuous variable must be strictly positive.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers or
                use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str()` on each key object.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...
        Note:
            When `keys` is either an empty list or the integer 0, an empty list is returned.


        :returns: A list of :class:`docplex.mp.linear.Var` objects with type :class:`docplex.mp.vartype.SemiContinuousVarType`.

        See Also:
            :attr:`infinity`

        """
        return self._var_list(keys, self.semicontinuous_vartype, lb, ub, name, key_format)

    def semiinteger_var_list(self, keys, lb, ub=None, name=str, key_format=None):
        """
        Creates a list of semi-integer decision variables with type :class:`docplex.mp.vartype.SemiIntegerVarType`,
        stores them in the model, and returns the list.

        Args:
            keys: Either a sequence of objects or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, or a function.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers or
                use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                Note that the lower bound of a semi-integer variable must be strictly positive.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers or
                use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str()` on each key object.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...
        Note:
            When `keys` is either an empty list or the integer 0, an empty list is returned.


        :returns: A list of :class:`docplex.mp.linear.Var` objects with type :class:`docplex.mp.vartype.SemiIntegerVarType`.

        See Also:
            :attr:`infinity`

        """
        return self._var_list(keys, self.semiinteger_vartype, lb, ub, name, key_format)

    def continuous_var_dict(self, keys, lb=None, ub=None, name=None, key_format=None):
        """ Creates a dictionary of continuous decision variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If `keys` is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, or a function.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers or
                use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default lower bound (0) is used.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers or
                use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str` on each key object.

            key_format: A format string or None. This format string describes how `keys` contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects (with type `ContinuousVarType`) indexed by
                  the objects in `keys`.

        See Also:
            :class:`docplex.mp.linear.Var`,
            :attr:`infinity`
        """
        return self._var_dict(keys, self.continuous_vartype, lb=lb, ub=ub, name=name, key_format=key_format)

    def integer_var_dict(self, keys, lb=None, ub=None, name=None, key_format=None):
        """  Creates a dictionary of integer decision variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If `keys` is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, or a function.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers or
                use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default lower bound (0) is used.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers or
                use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.


            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str` on each key object.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns:  A dictionary of :class:`docplex.mp.linear.Var` objects (with type `IntegerVarType`) indexed by the
                   objects in `keys`.

        See Also:
            :attr:`infinity`
        """
        return self._var_dict(keys, self.integer_vartype, lb=lb, ub=ub, name=name, key_format=key_format)

    def binary_var_dict(self, keys, lb=None, ub=None, name=None, key_format=None):
        """ Creates a dictionary of binary decision variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If `keys` is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            name (string): Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects with type
                  :class:`docplex.mp.vartype.BinaryVarType` indexed by the objects in `keys`.
        """
        return self._var_dict(keys, self.binary_vartype, lb=lb, ub=ub, name=name, key_format=key_format)

    def semiinteger_var_dict(self, keys, lb, ub=None, name=str, key_format=None):
        """  Creates a dictionary of semi-integer decision variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If `keys` is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, or a function.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers or
                use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers or
                use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.


            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str` on each key object.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns:  A dictionary of :class:`docplex.mp.linear.Var` objects (with type `SemiIntegerVarType`) indexed by the
                   objects in `keys`.

        See Also:
            :attr:`infinity`
        """
        return self._var_dict(keys, self.semiinteger_vartype, lb, ub, name, key_format)

    def semiinteger_var_matrix(self, keys1, keys2, lb, ub=None, name=None, key_format=None):
        """ Creates a dictionary of semiinteger decision variables, indexed by pairs of key objects.

        Creates a dictionary that allows the retrieval of variables from  a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.

        Arguments `lb`, `ub`, `name`, and `key_format` are interpreted as in :func:`semiinteger_var_dict`.

        *New in version 2.9*
        """
        return self._var_multidict(self.semiinteger_vartype, [keys1, keys2], lb, ub, name, key_format)

    def semicontinuous_var_dict(self, keys, lb, ub=None, name=str, key_format=None):
        """  Creates a dictionary of semi-continuous decision variables, indexed by key objects.

        Creates a dictionary that allows retrieval of variables from business
        model objects. Keys can be either a Python collection, an iterator, or a generator.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If `keys` is empty, this method returns an empty dictionary.

        Args:
            keys: Either a sequence of objects, an iterator, or a positive integer. If passed an integer,
                it is interpreted as the number of variables to create.

            lb: Lower bounds of the variables. Accepts either a floating-point number,
                a list of numbers, or a function.
                Use a number if all variables share the same lower bound.
                Otherwise either use an explicit list of numbers or
                use a function if lower bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.

            ub: Upper bounds of the variables. Accepts either a floating-point number,
                a list of numbers, a function, or None.
                Use a number if all variables share the same upper bound.
                Otherwise either use an explicit list of numbers or
                use a function if upper bounds vary depending on the key, in which case,
                the function will be called on each `key` in `keys`.
                None means the default upper bound (model infinity) is used.


            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if `keys` is a sequence) or the
                index of the variable within the range, if an integer argument is passed.
                If passed a function, this function is called on each key object to generate a name.
                The default behavior is to call :func:`str` on each key object.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns:  A dictionary of :class:`docplex.mp.linear.Var` objects (with type `SemiIntegerVarType`) indexed by the
                   objects in `keys`.

        See Also:
            :attr:`infinity`
        """
        return self._var_dict(keys, self.semicontinuous_vartype, lb, ub, name, key_format)

    def semicontinuous_var_matrix(self, keys1, keys2, lb, ub=None, name=None, key_format=None):
        """ Creates a dictionary of semicontinuous decision variables, indexed by pairs of key objects.

        Creates a dictionary that allows the retrieval of variables from  a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.

        Arguments `lb`, `ub`, `name`, and `key_format` are interpreted as in :func:`semiinteger_var_dict`.

        *New in version 2.9*
        """
        return self._var_multidict(self.semicontinuous_vartype, [keys1, keys2], lb, ub, name, key_format)

    def var_hypercube(self, vartype_spec, seq_of_keys, lb=None, ub=None, name=None, key_format=None):
        """ Creates a dictionary of  decision variables, indexed by tuples
            of arbitrary size.

            Arguments are analogous to methods of the type xxx_var_matrix,
            except a type argument has to be passed.

        Args:
            vartype_spec: type specificsation: accepts either an instance of class
                `docplex.mp.VarType`, or a string that can be translated into a vartype.
                Possible strings are:
                    cplex type codes, e.g. B,I,C,N,S or type short names
                    (e.g.: binary, integer, continuous, semicontinuous, semiinteger)

            seq_of_keys: a sequence of sequence of keys. Typically of length >= 4,
                as other dimensions are handled by the 'list', 'matrix' and 'cube'
                series of methods.
                Variables are indexed by tuples formed by the cartesian product of elements
                form the sequences; all sequences of keys must be non-empty.

        All other arguments have the same meaning as for all the "xx_var_matrix" family of methods.

        Example:
            >>> hc = Model().var_hypercube(vartype_spec='B', seq_of_keys=[[1,2], [3], ['a','b'], [1,2,3,4]]
            >>> len(hc)
            16
            returns a dict of 2x2x4 = 16 variables indexed by tuples formed by the cartesian product
            of the four lists, for example (1,3,'a',4)is a valid key for the hypercube.

        *New in 2.19*

        See Also:
            :class:`docplex.mp.vartype.VarType`

        """
        vartype = self._parse_vartype(vartype_spec)
        self._checker.typecheck_vartype(vartype)
        self._checker.typecheck_iterable(seq_of_keys)
        lkeys = list(seq_of_keys)
        arity = len(lkeys)
        if arity == 0:
            self.fatal("Variable hypercube with zero dimension")

        return self._var_multidict(vartype, lkeys, lb, ub, name, key_format)

    def _var_multidict(self, vartype, keys, lb=None, ub=None, name=None, key_format=None):
        assert isinstance(vartype, VarType)
        return self._lfactory.new_var_multidict(keys, vartype, lb, ub, name, key_format, ordered=self._keep_ordering)

    def var_matrix(self, vartype, keys1, keys2, lb=None, ub=None, name=None, key_format=None):
        return self._var_multidict(vartype, keys=[keys1, keys2],
                                  lb=lb, ub=ub, name=name, key_format=key_format)

    def binary_var_matrix(self, keys1, keys2, name=None, key_format=None):
        """ Creates a dictionary of binary decision variables, indexed by pairs of key objects.

        Creates a dictionary that allows the retrieval of variables from  a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.
        Keys are used in the naming of variables.

        Note:
            If either of `keys1` or `keys2` is empty, this method returns an empty dictionary.

        Args:
            keys1: Either a sequence of objects, an iterator, or a positive integer. If passed an integer N,
                    it is interpreted as a range from 0 to N-1.

            keys2: Same as `keys1`.

            name: Used to name variables. Accepts either a string or
                a function. If given a string, the variable name is formed by appending the string
                to the string representation of the key object (if keys is a sequence) or the
                index of the variable within the range, if an integer argument is passed.

            key_format: A format string or None. This format string describes how keys contribute to variable names.
                        The default is "_%s". For example if name is "x" and each key object is represented by a string
                        like "k1", "k2", ... then variables will be named "x_k1", "x_k2",...

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects with type
                  :class:`docplex.mp.vartype.BinaryVarType` indexed by
                  all couples `(k1, k2)` with `k1` in `keys1` and `k2` in `keys2`.
        """
        return self._var_multidict(self.binary_vartype, [keys1, keys2], 0, 1, name=name, key_format=key_format)

    def integer_var_matrix(self, keys1, keys2, lb=None, ub=None, name=None, key_format=None):
        """ Creates a dictionary of integer decision variables, indexed by pairs of key objects.

        Creates a dictionary that allows the retrieval of variables from  a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.

        Arguments `lb`, `ub`, `name`, and `key_format` are interpreted as in :func:`integer_var_dict`.
        """
        return self._var_multidict(self.integer_vartype, [keys1, keys2], lb, ub, name, key_format)

    def continuous_var_matrix(self, keys1, keys2, lb=None, ub=None, name=None, key_format=None):
        """ Creates a dictionary of continuous decision variables, indexed by pairs of key objects.

        Creates a dictionary that allows retrieval of variables from a tuple
        of two keys, the first one from `keys1`, the second one from `keys2`.
        In short, variables are indexed by the Cartesian product of the two key sets.

        A key can be any Python object, with the exception of None.

        Arguments `lb`, `ub`, `name`, and `key_format` are interpreted the same as in :func:`integer_var_dict`.

        """
        return self._var_multidict(self.continuous_vartype, [keys1, keys2], lb, ub, name, key_format)

    def continuous_var_cube(self, keys1, keys2, keys3, lb=None, ub=None, name=None, key_format=None):
        """ Creates a dictionary of continuous decision variables, indexed by triplets of key objects.

        Same as :func:`continuous_var_matrix`, except that variables are indexed by triplets of
        the form `(k1, k2, k3)` with `k1` in `keys1`, `k2` in `keys2`, `k3` in `keys3`.
        """
        return self._var_multidict(self.continuous_vartype, [keys1, keys2, keys3], lb, ub, name, key_format)

    def integer_var_cube(self, keys1, keys2, keys3, lb=None, ub=None, name=str):
        """ Creates a dictionary of integer decision variables, indexed by triplets.

        Same as :func:`integer_var_matrix`, except that variables are indexed by triplets of
        the form `(k1, k2, k3)` with `k1` in `keys1`, `k2` in `keys2`, `k3` in `keys3`.

        See Also:
            :func:`integer_var_matrix`
        """
        return self._var_multidict(self.integer_vartype, [keys1, keys2, keys3], lb, ub, name)

    def binary_var_cube(self, keys1, keys2, keys3, name=None, key_format=None):
        """Creates a dictionary of binary decision variables, indexed by triplets.

        Same as :func:`binary_var_matrix`, except that variables are indexed by triplets of
        the form `(k1, k2, k3)` with `k1` in `keys1`, `k2` in `keys2`, `k3` in `keys3`.

        :returns: A dictionary of :class:`docplex.mp.linear.Var` objects (with type :class:`docplex.mp.vartype.BinaryVarType`) indexed by
            triplets.

        """
        return self._var_multidict(self.binary_vartype, [keys1, keys2, keys3], name=name, key_format=key_format)

    def linear_expr(self, arg=None, constant=0, name=None):
        ''' Returns a new empty linear expression.

        Args:
            arg: an optional argument to convert to a linear expression. Detailt is None,
                in which case, an empty expression is returned.
            name: An optional string to name the expression.

        :returns: An instance of :class:`docplex.mp.linear.LinearExpr`.
        '''
        self._checker.typecheck_string(arg=name, accept_none=True)
        self._checker.typecheck_num(arg=constant, caller='Model.linear_expr()')
        return self._lfactory.linear_expr(arg=arg, constant=constant, name=name)

    def quad_expr(self, name=None):
        ''' Returns a new empty quadratic expression.

        Args:
            name: An optional string to name the expression.

        :returns: An instance of :class:`docplex.mp.quad.QuadExpr`.
        '''
        return self._qfactory.new_quad(name=name)


    def abs(self, e):
        """ Builds an expression equal to the absolute value of its argument.

        Args:
            e: Accepts any object that can be transformed into an expression:
                decision variables, expressions, or numbers.

        Returns:
            An expression that can be used in arithmetic operators and constraints.

        Note:
            Building the expression generates auxiliary decision variables, including binary decision variables,
            and this may change the nature of the problem from a LP to a MIP.

        """
        self._checker.typecheck_operand(e, caller="Model.abs", accept_numbers=True)
        return self._lfactory.new_abs_expr(e)

    def min(self, *args):
        """ Builds an expression equal to the minimum value of its arguments.

        This method accepts a variable number of arguments.

        If no arguments are provided, returns positive infinity (see :attr:`infinity`).

        Args:
            args: A variable list of arguments, each of which is either an expression, a variable,
                or a container.

        If passed a container or an iterator, this container or iterator must be the unique argument.

        If passed one dictionary, returns the minimum of the dictionary values.


        Returns:
            An expression that can be used in arithmetic operators and constraints.

        Example:
            `model.min()` -> returns `model.infinity`.

            `model.min(e)` -> returns `e`.

            `model.min(e1, e2, e3)` -> returns a new expression equal to the minimum of the values of `e1`, `e2`, `e3`.

            `model.min([x1,x2,x3])` where `x1`, `x2` .. are variables or expressions -> returns the minimum of these expressions.

            `model.min([])` -> returns `model.infinity`.

        Note:
            Building the expression generates auxiliary variables, including binary decision variables,
            and this may change the nature of the problem from a LP to a MIP.
        """
        min_args = args
        nb_args = len(args)
        if 0 == nb_args:
            pass
        elif 1 == nb_args:
            unique_arg = args[0]
            if is_iterable(unique_arg):
                if isinstance(unique_arg, dict):
                    min_args = unique_arg.values()
                else:
                    min_args = _to_list(unique_arg)
                for a in min_args:
                    self._checker.typecheck_operand(a, caller="Model.min()")
            else:
                self._checker.typecheck_operand(unique_arg, caller="Model.min()")

        else:
            for arg in args:
                self._checker.typecheck_operand(arg, caller="Model.min")

        return self._lfactory.new_min_expr(*min_args)

    def max(self, *args):
        """ Builds an expression equal to the maximum value of its arguments.

        This method accepts a variable number of arguments.

        Args:
            args: A variable list of arguments, each of which is either an expression, a variable,
                or a container.

        If passed a container or an iterator, this container or iterator must be the unique argument.

        If passed one dictionary, returns the maximum of the dictionary values.

        If no arguments are provided, returns negative infinity (see :attr:`infinity`).


        Example:
            `model.max()` -> returns `-model.infinity`.

            `model.max(e)` -> returns `e`.

            `model.max(e1, e2, e3)` -> returns a new expression equal to the maximum of the values of `e1`, `e2`, `e3`.

            `model.max([x1,x2,x3])` where `x1`, `x2` .. are variables or expressions -> returns the maximum of these expressions.

            `model.max([])` -> returns `-model.infinity`.


        Note:
            Building the expression generates auxiliary variables, including binary decision variables,
            and this may change the nature of the problem from a LP to a MIP.
        """
        max_args = args
        nb_args = len(args)
        if 0 == nb_args:
            pass
        elif 1 == nb_args:
            unique_arg = args[0]
            if is_iterable(unique_arg):
                if isinstance(unique_arg, dict):
                    max_args = unique_arg.values()
                else:
                    max_args = _to_list(unique_arg)
                for a in max_args:
                    self._checker.typecheck_operand(a, caller="Model.max")
            else:
                self._checker.typecheck_operand(unique_arg, caller="Model.max")
        else:
            for arg in args:
                self._checker.typecheck_operand(arg, caller="Model.max")

        return self._lfactory.new_max_expr(*max_args)


    def logical_and(self, *args):
        """ Builds an expression equal to the logical AND value of its arguments.

        This method takes a variable number of arguments, and accepts
        binary variables, other logical expressions, or discrete constraints.

        Args:
            args: A variable list of logical operands.

        Note:
            If passed an empty number of arguments, this method an expression equal to 1.

        Returns:
            An expression, equal to 1 if and only if all of its
            arguments are equal to 1, else equal to 0.

        See Also:
            :func:`logical_or`
            :func:`logical_not`

        Example:
            # return logical XOR or two binary variables.
            def logxor(m, b1, b2):
                return m.logical_and(m.logical_or(b1, b2), m.logical_not(m.logical_and(b1, b2)))

        """
        bvars = self._checker.typecheck_logical_op_seq(args, caller='Model.logical_and')
        return self._lfactory.new_logical_and_expr(bvars)

    def logical_or(self, *args):
        """ Builds an expression equal to the logical OR value of its arguments.

        This method takes a variable number of arguments, and accepts
            binary variables, other logical expressions, or discrete constraints.

        Args:
            args: A variable list of logical operands.

        Note:
            If passed an empty number of arguments, this method a zero expression.

        Returns:
            An expression, equal to 1 if and only if at least one of its
             arguments is equal to 1, else equal to 0.

        See Also:
            :func:`logical_and`
            :func:`logical_not`

        *New in version 2.11*

        """
        bvars = self._checker.typecheck_logical_op_seq(args, caller='Model.logical_or')
        return self._lfactory.new_logical_or_expr(bvars)

    def logical_not(self, arg):
        """ Builds an expression equal to the logical negation of its argument.

        This method accepts either a binary variable, or another logical expression.

        Args:
            arg: A binary variable, or a logical expression,
                e.g. an expression built by logical_and, logical_or, logical_not

        Returns:
            An expression, equal to 1 if its argument is 0, else 0.

        See Also:
            :func:`logical_and`
            :func:`logical_or`

        """
        StaticTypeChecker.typecheck_logical_op(self, arg, 'Model.logical_not')
        return self._lfactory.new_logical_not_expr(arg)

    def scal_prod(self, terms, coefs):
        """
        Creates a linear expression equal to the scalar product of a sequence of decision variables
        and a sequence of coefficients.

        This method accepts different types of input for both arguments. `terms` can be any
        iterable returning expressions or variables, and `coefs` is usually
        an iterable returning numbers.
        `cal_prod` also accept one number as `coefs`, in which case
        the scalar product reduces to a sum times this coefficient.

        :param terms: An iterable returning variables or expressions.
        :param coefs: An iterable returning numbers, or a number.

        Note:
            - both iterables are iterated at the same time, so the order in which terms and numbers
              are returned must be consistent: using unordered collections (e.g. sets) could lead to unexpected results.

           - Iteration stops as soon as one iterable stops. If both iterables are empty, the method returns 0.

        :return: A linear expression or 0.
        """
        self._checker.check_ordered_sequence(arg=terms, caller='Model.scal_prod() requires a list of expressions/variables')
        return self._aggregator.scal_prod(terms, coefs)

    def dot(self, terms, coefs):
        """ Synonym for :func:`scal_prod`.

        """
        return self.scal_prod(terms, coefs)

    def dotf(self, var_dict, coef_fn, assume_alldifferent=True):
        """ Creates a scalar product from a dictionary of variables and a function.

        This method is a functional variant of `dot`. I takes as asrgument a dictionary of variables,
        as returned by xxx_var_dict or xx_var_var_matrix (where xxx is a type), and a function.

        :param var_dict: a dictionary of variables, as returned by all the xxx_var_dict methods (e.g. integer_var_dict),
            but also multi-dimensional dictionaries, such as xxx_var_matrix (or var_cube).
        :param coef_fn: A function that takes one dictionary key and returns anumber. One-dimension dictionaries (such
            as integer_var_dict) have plain object as keys, but multi-dimensional dictionaries have tuples keys.
            For example, a binary_var_matrix returns a dictionary, the keys of which are 2-tuples.
        :param assume_alldifferent: an optional flag whichi ndicates whether variables values in the dictionary
            can be assumed to be all different. This is true when the dicitonary has been built by Docplex's
            Model.xxx_var-dict(), and thi sis the default behavior.
            For a custom-built dictionary, set the flag to False. A wrong flag value may yield incorrect results.

        :return: an expression, built as a scalar product of all variables in the dictionay, multiplied by the result of the function.

        Examples:

                >>> m1 = m.binary_var_matrix(keys1=range(1,3), keys2=range(1,3), name='b')
                >>> s = m.dotf(m1, lambda keys: keys[0] + keys[1])

                returns 2 b_1_1 + 3 b_1_2 +3 b_2_1 + 4 b_2_2

        """
        StaticTypeChecker.typecheck_callable\
            (self, coef_fn,
             "Functional scalar product requires a function taking variable keys as argument. A non-callable was passed: {0!r}".format(
                 coef_fn))
        return self._aggregator._scal_prod_f(var_dict, coef_fn, assume_alldifferent)

    scal_prod_f = dotf

    def sum(self, args):
        """ Creates a linear expression summing over an iterable over expressions or variables.

        Note:
           This method returns 0 if the argument is empty.
        
        :param args: An iterable returning linear expressions, variables or numbers.

        :return: A linear expression or 0.
        """
        return self._aggregator.sum(args)

    def sumsq(self, args):
        """ Creates a quadratic expression summing squares of expressions.

        Each element of the list is squared and added to the result. Quadratic expressions
        are not accepted, as they cannot be squared.

        Note:
           This method returns 0 if the argument is an empty list or iterator.

        :param args: An iterable returning linear expressions, variables or numbers.

        :return: A quadratic expression (possibly constant).
        """
        return self._aggregator.sumsq(args)

    sum_squares = sumsq

    def sum_vars(self, dvars):
        """ Creates a linear expression that sums variables.

        This method is faster than `Model.sum()` but accepts only variables.

        :param dvars: an iterable returning variables.

        :return: a linear expression equal to the sum of the variables.

        *New in version 2.10*
        """
        return self._aggregator._sum_vars(dvars)


    def sum_vars_all_different(self, terms):
        """
        Creates a linear expression equal to sum of a list of decision variables.
        The variable sequence is a list or an iterator of variables.

        This method is faster than the standard generic summation method due to the fact that it takes only
        variables and does not take expressions as arguments.

        :param terms: A list or an iterator on variables only, with no duplicates.

        :return: a linear expression equal to the sum of the variables.

        Note:
           If the variable iteration contains duplicates, this function returns an incorrect result.

        """
        if isinstance(terms, dict):
            return self.sum_vars_all_different(itervalues(terms))
        else:
            var_seq = self._checker.typecheck_var_seq_all_different(terms)
            return self._aggregator._sum_vars_all_different(var_seq)

    def le_constraint(self, lhs, rhs, name=None):
        """ Creates a "less than or equal to" linear constraint.

        Note:
            This method returns a constraint object, that is not added to the model.
            To add it to the model, use the :func:`add_constraint` method.

        Args:
            lhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
            rhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
            name (string): An optional string to name the constraint.

        :returns: An instance of :class:`docplex.mp.linear.LinearConstraint`.
        """
        return self._lfactory.new_le_constraint(lhs, rhs, name)

    def ge_constraint(self, lhs, rhs, name=None):
        """ Creates a "greater than or equal to" linear constraint.

        Note:
            This method returns a constraint object that is not added to the model.
            To add it to the model, use the :func:`add_constraint` method.

        Args:
            lhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
            rhs: An object that can be converted to a linear expression, typically a variable,
                    a number of an expression.
            name (string): An optional string to name the constraint.

        :returns: An instance of :class:`docplex.mp.linear.LinearConstraint`.
        """
        return self._lfactory.new_ge_constraint(lhs, rhs, name)

    def eq_constraint(self, lhs, rhs, name=None):
        """ Creates an equality constraint.

        Note:
            This method returns a constraint object that is not added to the model.
            To add it to the model, use the :func:`add_constraint` method.

        :param lhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
        :param rhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
        :param name: An optional string to name the constraint.

        :returns: An instance of :class:`docplex.mp.linear.LinearConstraint`.
        """
        return self._lfactory.new_eq_constraint(lhs, rhs, name)

    def linear_constraint(self, lhs, rhs, ctsense, name=None):
        """ Creates a linear constraint.

        Note:
            This method returns a constraint object that is not added to the model.
            To add it to the model, use the :func:`add_constraint` method.

        Args:
            lhs: An object that can be converted to a linear expression, typically a variable,
                    a member of an expression.
            rhs: An object that can be converted to a linear expression, typically a variable,
                    a number of an expression.
            ctsense: A constraint sense; accepts either a
                    value of type `ComparisonType` or a string (e.g 'le', 'eq', 'ge').

            name (string): An optional string to name the constraint.

        :returns: An instance of :class:`docplex.mp.linear.LinearConstraint`.
        """
        return self._lfactory.new_binary_constraint(lhs, ctsense, rhs, name)


    def not_equal_constraint(self, lhs, rhs, name=None):
        return self._lfactory.new_neq_constraint(lhs, rhs, name)

    def _create_engine_constraint(self, ct):
        # INTERNAL
        eng = self.__engine
        if isinstance(ct, LinearConstraint):
            return eng.create_linear_constraint(ct)
        elif isinstance(ct, RangeConstraint):
            return eng.create_range_constraint(ct)
        elif isinstance(ct, IndicatorConstraint):
            # here check whether linear ct is trivial. if yes do not send to CPLEX
            indicator = ct
            linct = indicator.linear_constraint
            if linct.is_trivial():
                is_feasible = linct._is_trivially_feasible()
                if is_feasible:
                    self.warning("Indicator constraint {0!s} has a trivially feasible constraint (no effect)",
                                 indicator)
                    return -2
                else:
                    self.warning(
                        "indicator constraint {0!s} has a trivially infeasible constraint; variable invalidated",
                        indicator)
                    indicator.invalidate()
                    return -4
            return eng.create_logical_constraint(ct, is_equivalence=False)
        elif isinstance(ct, EquivalenceConstraint):
            return eng.create_logical_constraint(ct, is_equivalence=True)

        elif isinstance(ct, QuadraticConstraint):
            return eng.create_quadratic_constraint(ct)
        elif isinstance(ct, PwlConstraint):
            return eng.create_pwl_constraint(ct)
        else:
            self.fatal("Expecting binary constraint, indicator or range, got: {0!s}", ct)  # pragma: no cover

    def _notify_trivial_constraint(self, ct, ctname, is_feasible):
        self_trivial_warn_level = self._trivial_cts_message_level
        if is_feasible and self_trivial_warn_level > self.warn_trivial_feasible:
            return
        elif self_trivial_warn_level > self.warn_trivial_infeasible:
            return
        # ---
        # hereafter we are sure to warn
        if ct is None:
            arg = None
        # elif ctname:
        #     arg = ctname
        # elif ct.has_name():
        #     arg = ct.name
        else:
            arg = str_maxed(ct, maxlen=24)
        # ---
        ct_typename = ct.short_typename if ct is not None else "constraint"
        ct_rank = self.number_of_constraints + 1
        # BEWARE: do not use if arg here
        # because if arg is a constraint, boolean conversion won't work.
        trivial_msg = "Adding trivially {3}feasible {2}: '{0!s}', pos: {1}"
        if arg is not None:
            if is_feasible:
                self.info(trivial_msg, arg, ct_rank, ct_typename, '')
            else:
                self.error(trivial_msg, arg, ct_rank, ct_typename, 'in')
                docplex_add_trivial_infeasible_ct(ct=arg)
        else:
            if is_feasible:
                self.info(trivial_msg, '', ct_rank, ct_typename, '')
            else:
                self.error(trivial_msg, '', ct_rank, ct_typename, 'in')
                docplex_add_trivial_infeasible_ct(ct=None)

    def _register_ct_name(self, ct, ctname, arg_checker):
        checker = arg_checker or self._checker
        if ctname:
            ct_name_map = self._cts_by_name
            if ct_name_map is not None:
                checker.check_duplicate_name(ctname, ct_name_map, "constraint")
                ct_name_map[ctname] = ct
            ct._set_name(ctname)

    def _prepare_constraint(self, ct, ctname, check_for_trivial_ct, arg_checker=None):
        # INTERNAL
        checker = arg_checker or self._checker
        if ct is True:
            # sum([]) == 0
            self._notify_trivial_constraint(ct=None, ctname=ctname, is_feasible=True)
            return False

        elif ct is False:
            # happens with sum([]) and constant e.g. sum([]) == 2
            self._notify_trivial_constraint(ct=None, ctname=ctname, is_feasible=False)
            msg = "Adding a trivially infeasible constraint"
            if ctname:
                msg += ' with name: {0}'.format(ctname)
            # analogous to 0 == 1, model is sure to fail
            self.fatal(msg)

        else:
            checker.typecheck_ct_to_add(ct, self, 'add_constraint')
            # -- watch for trivial cts e.g. linexpr(0) <= linexpr(1)
            if check_for_trivial_ct and ct.is_trivial():
                if ct._is_trivially_feasible():
                    self._notify_trivial_constraint(ct, ctname, is_feasible=True)
                elif ct._is_trivially_infeasible():
                    self._notify_trivial_constraint(ct, ctname, is_feasible=False)

        # check for already posted cts.
        if ct._index >= 0:
            self.warning("constraint has already been posted: {0!s}, index is: {1}", ct, ct.index)  # pragma: no cover
            return False  # pragma: no cover
        # --- name management ---
        self._register_ct_name(ct, ctname, checker)
        # ---
        return True

    def _add_constraint_internal(self, ct, ctname=None):
        used_ct_name = None if self._ignore_names else ctname
        if not isinstance(ct, AbstractConstraint) and hasattr(ct, 'as_logical_operand'):
            ct1 = self._lfactory.logical_expr_to_constraint(ct, ctname)
            return self._post_constraint(ct1)

        check_trivial = self._checker.check_trivial_constraints()
        if self._prepare_constraint(ct, used_ct_name, check_for_trivial_ct=check_trivial):
            self._post_constraint(ct)
            return ct
        elif ct is True or ct is False:
            return None
        else:
            return ct

    def _post_constraint(self, ct):
        ct_engine_index = self._create_engine_constraint(ct)
        self._register_one_constraint(ct, ct_engine_index, is_ctname_safe=True)
        return ct

    def pop_constraint(self):
        linear_scope = self._linct_scope
        if linear_scope.size:
            self._remove_constraint_internal(linear_scope.last)

    def _remove_constraint_internal(self, ct):
        self._remove_constraints_internal(cts_to_remove=(ct,))


    def _resolve_ct(self, ct_arg, silent=False, context=None):
        verbose = not silent
        if context:
            printed_context = context + ": "
        else:
            printed_context = ""
        if isinstance(ct_arg, AbstractConstraint):
            return ct_arg
        elif is_string(ct_arg):
            ct = self.get_constraint_by_name(ct_arg)
            if ct is None and verbose:
                self.error("{0}no constraint with name: \"{1}\" - ignored", printed_context, ct_arg)
            return ct
        elif is_int(ct_arg):
            if ct_arg >= 0:
                ct_index = ct_arg
                ct = self.get_constraint_by_index(ct_index)
                if ct is None and verbose:
                    self.error("{0}no constraint with index: \"{1}\" - ignored", printed_context, ct_arg)
                return ct
            else:
                self.error("{0}not a valid index: \"{1}\" - ignored", printed_context, ct_arg)
                return None

        else:
            if verbose:
                self.error("{0}unexpected argument {1!s}, expecting string or constraint", printed_context, ct_arg)
            return None

    def remove_constraint(self, ct_arg):
        """ Removes a constraint from the model.

        Args:
            ct_arg: The constraint to remove. Accepts either a constraint object or a string.
                If passed a string, looks for a constraint with that name.

        """
        ct = self._resolve_ct(ct_arg, silent=False, context="remove_constraint")
        if ct is not None:
            self._checker.typecheck_in_model(self, ct, caller="constraint")
            self._remove_constraints_internal(cts_to_remove=(ct,))

    def clear_constraints(self):
        """
        This method removes all constraints from the model.
        """
        self.__engine.remove_constraints(cts=None)  # special case to denote all
        # clear containers
        self._cts_by_name = None
        # clear constraint index scopes.
        for ctscope in self._constraint_scopes():
            ctscope.reset()

    def remove_constraints(self, cts=None):
        """
        This method removes a batch of constraints from the model.

        :param cts: an iterable of constraints (linear, range, quadratic, indicators)
        """
        if cts is not None:
            lcts = self._checker.typecheck_constraint_seq(cts)
            self._remove_constraints_internal(lcts)

    def _remove_constraints_internal(self, cts_to_remove):
        removed_cts = [c for c in cts_to_remove if c.has_valid_index()]
        if removed_cts:
            self.__engine.remove_constraints(removed_cts)
            # INTERNAL
            self_cts_by_name = self._cts_by_name

            if self_cts_by_name:
                for d in removed_cts:
                    dname = d.get_name()
                    if dname:
                        try:
                            del self_cts_by_name[dname]
                        except KeyError:
                            pass

            actual_touched_scopes = set()
            removed_ids = set()
            from collections import defaultdict
            idxs_by_scope = defaultdict(set)
            for d in removed_cts:
                removed_ids.add(id(d))
                idxs_by_scope[d.cplex_scope].add(d.index)
                actual_touched_scopes.add(d._get_index_scope())

            for sc, delset in iteritems(idxs_by_scope):
                scope = self.get_ct_scope(sc)
                scope.notify_delete_set(delset)

            for d in removed_cts:
                d.notify_deleted()

        if self.is_docplex_debug():
            for sc in self._constraint_scopes():
                sc.check_indices()

    def remove(self, removed):
        """
        This method removes a constraint or a collection of constraints from the model.

        :param removed: accapts either a constraint or an iterable on constraints (linear, range, quadratic, indicators)
        """
        if is_iterable(removed):
            self.remove_constraints(removed)
        else:
            self.remove_constraint(removed)

    def add_range(self, lb, expr, ub, rng_name=None):
        """ Adds a new range constraint to the model.

        A range constraint states that a linear
        expression has to stay within an interval `[lb..ub]`.
        Both `lb` and `ub` have to be float numbers with `lb` smaller than `ub`.

        The method creates a new range constraint and adds it to the model.

        Args:
            lb (float): A floating-point number.
            expr: A linear expression, e.g. X+Y+Z.
            ub (float): A floating-point number, which should be greater than `lb`.
            rng_name (string): An optional name for the range constraint.

        :returns: The newly created range constraint.
        :rtype: :class:`docplex.mp.constr.RangeConstraint`

        Raises:
            An exception if `lb` is greater than `ub`.

        """
        rng = self.range_constraint(lb, expr, ub)
        ctname = None if self._ignore_names else rng_name
        ct = self._add_constraint_internal(rng, ctname)
        return ct

    def indicator_constraint(self, binary_var, linear_ct, active_value=1, name=None):
        """ Creates and returns a new indicator constraint.

        The indicator constraint is not added to the model.

        Args:
            binary_var: The binary variable used to control the satisfaction of the linear constraint.
            linear_ct: A linear constraint (EQ, LE, GE).
            active_value: 0 or 1. The value used to trigger the satisfaction of the constraint. The default is 1.
            name (string): An optional name for the indicator constraint.

        :return:
            The newly created indicator constraint.
        """
        self._checker.typecheck_binary_var(binary_var)
        self._checker.typecheck_linear_constraint(linear_ct)
        self._checker.typecheck_zero_or_one(active_value)
        self._checker.typecheck_in_model(self, binary_var, caller="binary variable")
        self._checker.typecheck_in_model(self, linear_ct, caller="linear_constraint")
        return self._lfactory.new_indicator_constraint(binary_var, linear_ct, active_value, name)


    def add_indicator(self, binary_var, linear_ct, active_value=1, name=None):
        """ Adds a new indicator constraint to the model.

        An indicator constraint links (one-way) the value of a binary variable to
        the satisfaction of a linear constraint.
        If the binary variable equals the active value, then the constraint is satisfied, but
        otherwise the constraint may or may not be satisfied.

        Args:
             binary_var: The binary variable used to control the satisfaction of the linear constraint.
             linear_ct: A linear constraint (EQ, LE, GE).
             active_value: 0 or 1. The value used to trigger the satisfaction of the constraint. The default is 1.
             name (string): An optional name for the indicator constraint.

        Returns:
            The newly created indicator constraint.
        """
        self._checker.typecheck_string(name, accept_none=True)
        iname = None if self._ignore_names else name
        indicator = self.indicator_constraint(binary_var, linear_ct, active_value)
        return self._add_indicator(indicator, iname)

    _indicator_trivial_feasible_idx = -2
    _indicator_trivial_infeasible_idx = -4

    def _add_indicator(self, indicator, ind_name, check_trivials=False):
        # INTERNAL
        linear_ct = indicator.linear_constraint
        if check_trivials and self._checker.check_trivial_constraints() and linear_ct.is_trivial():
            is_feasible = linear_ct._is_trivially_feasible()
            if is_feasible:
                self.warning("Indicator constraint {0!s} has a trivial feasible linear constraint (has no effect)",
                             indicator)
                return self._indicator_trivial_feasible_idx
            else:
                self.warning("indicator constraint {0!s} has a trivial infeasible linear constraint - invalidated",
                             indicator)
                indicator.invalidate()
                return self._indicator_trivial_infeasible_idx
        else:
            return self._add_constraint_internal(indicator, ind_name)

    def add_equivalence(self, binary_var, linear_ct, true_value=1, name=None):
        """ Adds a new equivalence constraint to the model.

        An equivalence constraints links two-way the value of a binary variable to
        the satisfaction of a discrete linear constraint.
        If the binary variable equals the true value, then the constraint is satisfied,
        conversely if the constraint is satisfied, the binary variable is equal to the true value.

        Args:
             binary_var: The binary variable used to control the satisfaction of the linear constraint.
             linear_ct: A linear constraint (EQ, LE, GE).
             true_value: 0 or 1. The value used to trigger the satisfaction of the constraint. The default is 1.
             name (string): An optional name for the equivalence constraint.

        Returns:
            The newly created equivalence constraint.
        """
        equiv = self.equivalence_constraint(binary_var, linear_ct, true_value, name=None)
        eq_name = None if self.ignore_names else name
        eqct = self._add_constraint_internal(equiv, eq_name)
        return eqct

    def equivalence_constraint(self, binary_var, linear_ct, true_value=1, name=None):
        """ Creates and returns a new equivalence constraint.

        The newly created equivalence constraint is not added to the model.

        Args:
             binary_var: The binary variable used to control the satisfaction of the linear constraint.
             linear_ct: A linear constraint (EQ, LE, GE).
             true_value: 0 or 1. The value used to mark the satisfaction of the constraint. The default is 1.
             name (string): An optional name for the equivalence constraint.

        Returns:
            The newly created equivalence constraint.
        """
        checker = self._checker
        checker.typecheck_binary_var(binary_var)
        checker.typecheck_linear_constraint(linear_ct)
        checker.typecheck_zero_or_one(true_value)
        checker.typecheck_in_model(self, binary_var, caller="binary variable")
        checker.typecheck_in_model(self, linear_ct, caller="linear_constraint")
        checker.typecheck_string(name, accept_empty=True, accept_none=True)
        StaticTypeChecker.typecheck_discrete_constraint(self, linear_ct,
                                                        msg='Model.add_equivalence() requires a discrete constraint')
        used_name = None if self.ignore_names else name
        equiv = self._lfactory.new_equivalence_constraint(binary_var, linear_ct, true_value, used_name)
        return equiv

    def add_equivalences(self, binary_vars, cts, true_values=1, names=None):
        """ Adds a batch of equivalence constraints to the model.

        This method adds a batch of equivalence constraints to the model.

        :param binary_vars: a sequence of binary variables.
        :param cts: a sequence of discrete linear constraints
        :param true_values: the true values to use. Accepts either 1, 0 or a sequence of {0, 1} values.
        :param names: an optional sequence of names

        All sequences must have the same length.

        :return: a list of equivalence constraints.
        """
        return self._add_batch_logical_cts(binary_vars, cts, names, true_values, is_equivalence=True,
                                           caller='Model.add_equivalences')

    def add_indicators(self, binary_vars, cts, true_values=1, names=None):
        """ Adds a batch of indicator constraints to the model.

        This method adds a batch of indicator constraints to the model.

        :param binary_vars: a sequence of binary variables.
        :param cts: a sequence of discrete linear constraints
        :param true_values: the true values to use. Accepts either 1, 0 or a sequence of {0, 1} values.
        :param names: an optional sequence of names

        All sequences must have the same length.

        :return: a list of indicator constraints.
        """
        return self._add_batch_logical_cts(binary_vars, cts, names, true_values, is_equivalence=False,
                                           caller='Model.add_indicators')

    def _add_batch_logical_cts(self, binary_vars, cts, names, true_values, is_equivalence, caller=''):
        # internal
        checker = self._checker
        bvars = checker.typecheck_var_seq(binary_vars, vtype='B', caller=caller)
        ctseq = checker.typecheck_constraint_seq(cts, check_linear=True)
        try:
            n_vars = len(bvars)
            n_cts = len(cts)
        except TypeError:
            # if passed iterators, no len()
            bvars = list(bvars)
            ctseq = list(ctseq)
            n_vars = len(bvars)
            n_cts = len(ctseq)

        if n_vars != n_cts:
            self.fatal('Model.add_equivalences(): binary_vars and linear cts must have same size.')

        if true_values == 0 or true_values == 1:
            actual_true_values = generate_constant(true_values, n_vars)

        elif is_iterable(true_values):
            actual_true_values = list(true_values)
            if len(actual_true_values) != n_vars:
                self.fatal('Model.add_equivalences(): true_values has wrong size. expecting: {0}, got: {1}'
                           .format(n_vars, len(actual_true_values)))
            for a in actual_true_values:
                checker.typecheck_zero_or_one(a)
        else:
            self.fatal('Model.add_equivalence(): true_values expects 0|1 or sequence of {{0, 1}}, got: {0!r}'.format(true_values))

        if names is not None and not self._ignore_names:
            c_names = checker.typecheck_string_seq(names, accept_none=True, accept_empty=True, caller=caller)
            used_names = [n or '' for n in c_names]
            #     checker.typecheck_string(n, accept_empty=False, accept_none=True)
            #     used_names.append(n or '')
        else:
            used_names = generate_constant(None, n_vars)

        if is_equivalence:
            # check discrete
            lcts = list(ctseq)
            caller = "Model.add_equivalences" if is_equivalence else "Model.add_indicators"
            caller += " requires an iterable of discrete constraints"
            for ct in lcts:
                StaticTypeChecker.typecheck_discrete_constraint(self, ct, caller)
            eqcts = self._lfactory.new_batch_equivalence_constraints(bvars, lcts, actual_true_values, used_names)
            self.add_equivalence_constraints_(eqcts)
            return eqcts
        else:
            indcts = self._lfactory.new_batch_indicator_constraints(bvars, ctseq, actual_true_values, used_names)
            self.add_indicator_constraints_(indcts)
            return indcts

    def add_indicator_constraints(self, indcts):
        """ Adds a batch of indicator constraints to the model

        :param indcts: an iterable returning indicator constraints.

        See Also:
            :func:`indicator_constraint`
        """
        ind_cts_list_ = list(indcts)
        self._checker.typecheck_logical_constraint_seq(ind_cts_list_, true_if_equivalence=False)
        ind_indices = self.__engine.create_batch_logical_constraints(ind_cts_list_, is_equivalence=False)
        self._register_block_cts(self._logical_scope, ind_cts_list_, ind_indices)
        return ind_cts_list_

    add_indicator_constraints_ = add_indicator_constraints

    def add_equivalence_constraints(self, eqcts):
        """ Adds a batch of equivalence constraints to the model

        :param eqcts: an iterable returning equivalence constraints.

        See Also:
            :func:`equivalence_constraint`
        """
        eqcts_list_ = list(eqcts)  # the list is traversed twice
        self._checker.typecheck_logical_constraint_seq(eqcts_list_, true_if_equivalence=True)
        eq_indices = self.__engine.create_batch_logical_constraints(eqcts_list_, is_equivalence=True)
        self._register_block_cts(self._logical_scope, eqcts_list_, eq_indices)
        return eqcts_list_

    add_equivalence_constraints_ = add_equivalence_constraints

    def if_then(self, if_ct, then_ct, negate=False):
        """ Creates and returns an if-then constraint.

        An if-then constraint links two constraints ct1, ct2 such that when ct1 is satisfied then ct2 is also satisfied.

        :param if_ct: a linear constraint, the satisfaction of which governs the satisfaction of the `then_ct`
        :param then_ct: a linear constraint, which becomes satisfied as soon as `if_ct` is satisfied
            (or when it is not, depending on the `negate` flag).
        :param negate: an optional boolean flag (default is False). If True, `then_ct` is satisfied when `if_ct` is *not* satisfied.

        :return:
            an instance of IfThenConstraint, that is not added to the model.
            Use Model.add_constraint() or Model.add() to add it to the model.

        Note:
            This constraint relies on the status of the `if_ct` constraint, so this constraint must be discrete,
            otherwise an exception will be raised.
        """
        checker = self._checker
        checker.typecheck_linear_constraint(if_ct)
        checker.typecheck_linear_constraint(then_ct)
        StaticTypeChecker.typecheck_discrete_constraint(logger=self, ct=if_ct, msg='Model.if_then()')
        return self._lfactory.new_if_then_constraint(if_ct, then_ct, bool(negate))

    def add_if_then(self, if_ct, then_ct, negate=False):
        """ Creates a new if-then constraint and adds it to the model

        :param if_ct: a linear constraint, the satisfaction of which governs the satisfaction of the `then_ct`
        :param then_ct: a linear constraint, which becomes satisfied as soon as `if_ct` is satisfied
            (or when it is not, depending on the `negate` flag).
        :param negate: an optional boolean flag (default is False). If True, `then_ct` is satisfied when `if_ct` is *not* satisfied.

        :return:
            an instance of IfThenConstraint.

        Note:
            This constraint relies on the status of the `if_ct` constraint, so this constraint must be discrete,
            otherwise an exception will be raised. On the opposite, the `then_ct` constraint may be non-discrete.

            Also note that this construct relies on the status variable of the `if_ct`, so one extra binary variable is generated.

            *New in 2.16*: when `if_ct` is of the form `bvar == 1` or `bvar ==0`, where `bvar` is a binary variable,
             , no extra variable is generated, and a plain indicator constraint is generated.

            An alternative syntax is to use the `>>` operator on linear constraints:

                >>> m.add(c1 >> c2)

            is exactly equivalent to:

                >>> m.add_if_then(c1, c2)


        """
        ifthen_ct = self.if_then(if_ct, then_ct, negate=negate)
        return self._post_constraint(ifthen_ct)

    def range_constraint(self, lb, expr, ub, rng_name=None):
        """ Creates a new range constraint but does not add it to the model.

        A range constraint states that a linear
        expression has to stay within an interval `[lb..ub]`.
        Both `lb` and `ub` have to be floating-point numbers with `lb` smaller than `ub`.

        The method creates a new range constraint but does not add it to the model.

        Args:
            lb: A floating-point number.
            expr: A linear expression, e.g. X+Y+Z.
            ub: A floating-point number, which should be greater than `lb`.
            rng_name: An optional name for the range constraint.

        Returns:
            The newly created range constraint.
        Raises:
             An exception if `lb` is greater than `ub`.

        """
        self._checker.typecheck_num(lb, 'Model.range_constraint')
        self._checker.typecheck_num(ub, 'Model.range_constraint')
        self._checker.typecheck_string(rng_name, accept_empty=False, accept_none=True)
        rng = self._lfactory.new_range_constraint(lb, expr, ub, rng_name)
        return rng

    def add_constraint(self, ct, ctname=None):
        """ Adds a new linear constraint to the model.

        Args:
            ct: A linear constraint of the form <expr1> <op> <expr2>, where both expr1 and expr2 are
                 linear expressions built from variables in the model, and <op> is a relational operator
                 among <= (less than or equal), == (equal), and >= (greater than or equal).
            ctname (string): An optional string used to name the constraint.

        Returns:
            The newly added constraint.

        See Also:
            :func:`add_constraint_`
        """
        ct = self._add_constraint_internal(ct, ctname)
        return ct

    def add_constraint_(self, ct, ctname=None):
        """ Adds a new linear constraint to the model.

        Args:
            ct: A linear constraint of the form <expr1> <op> <expr2>, where both expr1 and expr2 are
                 linear expressions built from variables in the model, and <op> is a relational operator
                 among <= (less than or equal), == (equal), and >= (greater than or equal).
            ctname (string): An optional string used to name the constraint.

        Note:
            This method does the same as `docplex.mp.model.Model.add_constraint()` except that it has no return value.

        See Also:
            :func:`add_constraint`
        """
        self._add_constraint_internal(ct, ctname)

    def add(self, ct, name=None):
        if is_iterable(ct):
            return self.add_constraints(ct, name)
        else:
            return self.add_constraint(ct, name)

    def add_(self, ct, name=None):
        if is_iterable(ct):
            self.add_constraints_(ct, name)
        else:
            self.add_constraint_(ct, name)

    def add_constraints(self, cts, names=None):
        """ Adds a batch of linear constraints to the model in one operation.

        Each constraint from the `cts` iterable is added to the model.
        If present, the `names` iterable is used to set names to the constraints.

        Example:
                # list
                >>> m.add_constraints([x >= 1, y<= 3], ["c1", "c2"])
                # comprehension
                >>> m.add_constraints((xs[i] >= i for i in range(N)))

        Args:
            cts: An iterable of linear constraints; can be a list, a set or a comprehensions.
                Any Python object, which can be iterated on and yield consttraint objects.
            names: An optional iterable on strings. ANy Python object which can be iterated on
                and yield strings. The default value is None, meaning no names are set.

        Returns:
            A list of the newly added constraints.

        Note:
            This method handles only linear constraints (including range constraints). To add
        multiple quadratic constraints, see :func:`add_quadratic_constraints`

        See Also:
            :func:`add_constraints_`

        """
        self._checker.typecheck_iterable(cts)
        if names is not None and not self.ignore_names:
            if not is_iterable(names) or (is_string(names) and not names):
                self.fatal("Model.add_constraints() expects a sequence of strings or a non-empty string")
            return self._lfactory._new_constraint_block2(cts, names)
        else:
            return self._lfactory._new_constraint_block1(cts)

    def _new_xconstraint(self, lhs, rhs, comparaison_type):
        isquad = False
        if self._quad_count:
            try:
                isquad = rhs.is_quad_expr()
            except AttributeError:
                pass
        if isquad:
            return self._qfactory._new_qconstraint(lhs, comparaison_type, rhs)
        else:
            return self._lfactory._new_binary_constraint(lhs, comparaison_type, rhs)

    def add_constraints_(self, cts, names=None):
        """ Adds a batch of linear constraints to the model in one operation.

        Same as `docplex.model.Model.add_constraints()` except that is does not return anything.
        """
        self._checker.typecheck_iterable(cts)
        if names is not None and not self.ignore_names:
            self._lfactory._new_constraint_block2(cts, names)
        else:
            self._lfactory._new_constraint_block1(cts)


    def add_ranges(self, lbs, exprs, ubs, names=None):
        checker = self._checker
        lbsl = checker.typecheck_num_seq(lbs, caller="Model.add_ranges.lbs")
        ubsl = checker.typecheck_num_seq(ubs, caller="Model.add_ranges.ubs")
        checker.typecheck_iterable(exprs)
        range_names = names if names and not self.ignore_names else None
        return self._lfactory.new_range_block(lbsl, exprs, ubsl, range_names)

    def _post_quadratic_constraint(self, qct):
        qcx = self.__engine.create_quadratic_constraint(qct)
        self._register_one_constraint(qct, qcx, is_ctname_safe=True)
        return qct

    def add_quadratic_constraints(self, qcs):
        """ Adds a batch of quadratic contraints in one call.

        :param qcs: an iterable on a quadratic constraints.

        Note:
            The `Model.add_constraints` method handles only linear constraints.

        New in version 2.16*
        """
        lqcs = self._checker.typecheck_quadratic_constraint_seq(qcs)
        for qc in lqcs:
            self._post_quadratic_constraint(qc)

    # ----------------------------------------------------
    # objective
    # ----------------------------------------------------

    def minimize(self, expr):
        """ Sets an expression as the expression to be minimized.

        The argument is converted to a linear expression. Accepted types are variables (instances of
        :class:`docplex.mp.linear.Var` class), linear expressions (instances of
        :class:`docplex.mp.linear.LinearExpr`), or numbers.

        :param expr: A linear expression or a variable.
        """
        self.set_objective(ObjectiveSense.Minimize, expr)

    def maximize(self, expr):
        """
        Sets an expression as the expression to be maximized.

        The argument is converted to a linear expression. Accepted types are variables (instances of
        :class:`docplex.mp.linear.Var` class), linear expressions (instances of
        :class:`docplex.mp.linear.LinearExpr`), or numbers.

        :param expr: A linear expression or a variable.
        """
        self.set_objective(ObjectiveSense.Maximize, expr)

    def _make_lex_priorities(self, nb_objectives):
        # INTERNAL
        return list(range(nb_objectives-1, -1, -1))

    def _compile_multiobj_expr_list(self, exprs, caller, accept_empty=False):
        # INTERNAL
        # converts an exprs argument to a (possibly empty) list of expressions.
        # no side-effect is performed here, the caller has to take action.
        caller_string = "{0} ".format(caller) if caller is not None else ""
        if not exprs:
            if accept_empty:
                self.warning("{0}requires a non-empty list of linear expressions, got: {1!r}",
                             caller_string, exprs)
                return []
            else:
                self.fatal("{0}requires a non-empty list of linear expressions, got: {1!r}",
                             caller_string, exprs)
        if is_indexable(exprs):
            try:
                exprs = [self._lfactory._to_linear_operand(x) for x in exprs]
                return exprs

            except (TypeError, DOcplexException):
                pass
        self.fatal(
            "{0}requires an indexable sequence of linear expressions, {1!r} was passed",
            caller_string, exprs)


    @classmethod
    def supports_multi_objective(cls):
        return cls()._supports_multi_objective()

    def _supports_multi_objective(self):
        # INTERNAL
        ok, _ = self.__engine.supports_multi_objective()
        return ok

    def _check_multi_objective_support(self):
        # INTERNAL
        ok, why = self.__engine.supports_multi_objective()
        if not ok:
            assert why
            self.fatal(msg=why)

    def minimize_static_lex(self, exprs, abstols=None, reltols=None, objnames=None):
        """ Sets a list of expressions to be minimized in a lexicographic solve.
        exprs must be an ordered sequence of objective functions, that are minimized.

        The argument is converted to a list of linear expressions. Accepted types for the list elements are variables
        (instances of :class:`docplex.mp.linear.Var` class), linear expressions (instances of
        :class:`docplex.mp.linear.LinearExpr`), or numbers.

        Warning:
           This method requires CPLEX 12.9 or higher

        Args:
            exprs: a list of linear expressions or variables
            abstols: if defined, a list of absolute tolerances having the same size as the `exprs` argument.
            reltols: if defined, a list of relative tolerances having the same size as the `exprs` argument.
            objnames:  if defined, a list of names for objectives having the same size as the `exprs` argument.


        *New in version 2.9*
        """
        self._set_lex_multi_objective(ObjectiveSense.Minimize, exprs, abstols=abstols, reltols=reltols, names=objnames,
                                     caller='Model.minimize_static_lex()')


    def maximize_static_lex(self, exprs, abstols=None, reltols=None, objnames=None):
        """ Sets a list of expressions to be maximized in a lexicographic solve.
        exprs defines an ordered sequence of objective functions that are maximized.

        The argument is converted to a list of linear expressions. Accepted types for the list elements are variables
        (instances of :class:`docplex.mp.linear.Var` class), linear expressions (instances of
        :class:`docplex.mp.linear.LinearExpr`), or numbers.

        Warning:
            This method requires CPLEX 12.9 or higher

        Args:
            exprs: a list of linear expressions or variables
            abstols: if defined, a list of absolute tolerances having the same size as the `exprs` argument.
            reltols: if defined, a list of relative tolerances having the same size as the `exprs` argument.
            objnames:  if defined, a list of names for objectives having the same size as the `exprs` argument.


        *New in version 2.9*
        """
        self._set_lex_multi_objective(ObjectiveSense.Maximize, exprs, abstols=abstols, reltols=reltols, names=objnames,
                                     caller='Model.maximize_static_lex()')


    def is_minimized(self):
        """ Checks whether the model is a minimization model.

        Note:
            This returns True even if the expression to minimize is a constant.
            To check whether the model has a non-constant objective, use :func:`is_optimized`.

        Returns:
           Boolean: True if the model is a minimization model.
        """
        return self._objective_sense is ObjectiveSense.Minimize

    def is_maximized(self):
        """ Checks whether the model is a maximization model.

        Note:
           This returns True even if the expression to maximize is a constant.
           To check whether the model has a non-constant objective, use :func:`is_optimized`.

        Returns:
            Boolean: True if the model is a maximization model.
        """
        return self._objective_sense is ObjectiveSense.Maximize

    def objective_coef(self, dvar):
        """ Returns the objective coefficient of a variable.

        The objective coefficient is the coefficient of the given variable in
        the model's objective expression. If the variable is not explicitly
        mentioned in the objective, it returns 0.

        :param dvar: The decision variable for which to compute the objective coefficient.

        Returns:
            float: The objective coefficient of the variable.
        """
        self._checker.typecheck_var(dvar)
        return self._objective_coef(dvar)

    def _objective_coef(self, dvar):
        return self._objective_expr.unchecked_get_coef(dvar)

    def remove_objective(self):
        """ Clears the current objective.

        This is equivalent to setting "minimize 0".
        Any subsequent solve will look only for a feasible solution.
        You can detect this state by calling :func:`has_objective` on the model.

        """
        self.set_objective(self._lfactory.default_objective_sense(), self._new_default_objective_expr())

    def is_optimized(self):
        """ Checks whether the model has a non-constant objective expression.

        A model with a constant objective will only search for a feasible solution when solved.
        This happens either if no objective has been assigned to the model,
        or if the objective has been removed with :func:`remove_objective`.

        Returns:
            Boolean: True, if the model has a non-constant objective expression.

        """
        return self.has_multi_objective() or not self._objective_expr.is_constant()

    def set_multi_objective(self, sense, exprs, priorities=None, weights=None,
                            abstols=None, reltols=None,
                            names=None):
        """ Sets a list of objectives.

        Warning:
            This method requires CPLEX 12.9 or higher

        Args:
            sense: Either an instance of :class:`docplex.mp.basic.ObjectiveSense` (Minimize or Maximize),
                or a string: "min" or "max".
            exprs: Is converted to a list of expressions. Accepted types for this list items are variables,
                linear expressions or numbers.
            priorities: a list of priorities having the same size as the `exprs` argument. Priorities
                define how objectives are grouped together into sub-problems, and in which order these sub-problems
                are solved (in decreasing order of priorities). If not defined, allexpressions are assumed to share the same priority,
                and are combined with `weights`.
            weights: if defined, a list of weights having the same size as the `exprs` argument. Weights define
                how objectives with same priority are blended together to define the associated sub-problem's
                objective that is optimized. If not defined, weights are assumed to be all equal to 1.
            abstols: if defined, a list of absolute tolerances having the same size as the `exprs` argument.
            reltols: if defined, a list of relative tolerances having the same size as the `exprs` argument.
            names:  if defined, a list of names for objectives having the same size as the `exprs` argument.

        Note:
            When using a number for an objective, the search will not optimize but only look for a feasible solution.

        *New in version 2.9.*
        """
        self.set_objective_sense(sense)
        self._set_multi_objective_exprs(exprs, priorities=priorities, weights=weights,
                                        abstols=abstols, reltols=reltols, names=names,
                                        caller='Model.set_multi_objective()')

    def set_lex_multi_objective(self, sense, exprs, abstols=None, reltols=None, names=None):
        """ Sets a list of objectives to be solved in a lexicographic fashion.

        Objective expressions are listed in decreasing priority.

        Warning:
            This method requires CPLEX 12.9 or higher

        Args:
            sense: Either an instance of :class:`docplex.mp.basic.ObjectiveSense` (Minimize or Maximize),
                or a string: "min" or "max".
            exprs: Is converted to a list of expressions. Accepted types for this list items are variables,
                linear expressions or numbers.

            abstols: if defined, a list of absolute tolerances having the same size as the `exprs` argument.
            reltols: if defined, a list of relative tolerances having the same size as the `exprs` argument.
            names:  if defined, a list of names for objectives having the same size as the `exprs` argument.

        Note:
            When using a number for an objective, the search will not optimize but only look for a feasible solution.

        *New in version 2.9.*
        """
        self._set_lex_multi_objective(sense, exprs, abstols=abstols, reltols=reltols, names=names, caller='Model.set_lex_multi_objective')

    def _set_lex_multi_objective(self, sense, exprs, abstols, reltols, names, caller):
        # INTERNAL
        self.set_objective_sense(sense)
        lex_exprs = self._compile_multiobj_expr_list(exprs, caller=caller, accept_empty=False)
        lex_priorities = self._make_lex_priorities(len(exprs))
        self._set_multi_objective_internal(lex_exprs, priorities=lex_priorities,
                                           weights=None,
                                           abstols=abstols, reltols=reltols, names=names,
                                           caller=caller)

    def set_multi_objective_exprs(self, exprs, priorities=None, weights=None,
                                  abstols=None, reltols=None, names=None):
        """ Defines a list of blended objectives.

        Objectives with the same priority are combined using weights. Then, objectives are optimized in a
        lexicographic fashion by decreasing priority.

        Args:
            exprs: Is converted to a list of linear expressions. Accepted types for this list items are variables,
                linear expressions or numbers.
            priorities: if defined, a list of priorities having the same size as the `exprs` argument. Priorities
                define how objectives are grouped together into sub-problems, and in which order these sub-problems
                are solved (in decreasing order of priorities).
            weights: if defined, a list of weights having the same size as the `exprs` argument. Weights define
                how objectives with same priority are blended together to define the associated sub-problem
                objective that is optimized.
            abstols: if defined, a list of absolute tolerances having the same size as the `exprs` argument.
            reltols:if defined, a list of relative tolerances having the same size as the `exprs` argument.
            names: if defined, a list of names for objectives having the same size as the `exprs` argument.

        Note:
            When using a number for an objective, the search will not optimize but only look for a feasible solution.

        *New in version 2.9.*
        """
        self._set_multi_objective_exprs(exprs, priorities, weights, abstols, reltols, names, caller='Model.set_multi_objective()')

    def _set_multi_objective_exprs(self, exprs, priorities=None, weights=None,
                                  abstols=None, reltols=None, names=None, caller=None):
        exprs_ = self._compile_multiobj_expr_list(exprs, accept_empty=False, caller=caller)
        self._set_multi_objective_internal(exprs_, priorities=priorities, weights=weights,
                                           abstols=abstols, reltols=reltols, names=names, clear_objective=True,
                                           caller=caller)

    def _set_multi_objective_internal(self, exprs, priorities, weights,
                                      abstols, reltols,
                                      names, clear_objective=True, caller=None):
        # INTERNAL: assumes exprs is a valid list of linear expressions
        if 1 == len(exprs):
            expr0 = exprs[0]
            self.warning('Multi-objective has been converted to single objective: {0}', str_maxed(expr0, maxlen=16))
            self.set_objective_expr(expr0)
        else:
            for x in exprs:
                x.notify_used(self)
            if self.has_objective() and clear_objective:
                self._clear_objective_expr()

            def refine_caller(caller_, qualifier):
                if not caller_:
                    return caller_
                elif caller_[-1] == ')':
                    return '%s.%s' % (caller_[:-2], qualifier)
                else:
                    return '%s.%s' % (caller_, qualifier)

            abstols_ = self._typecheck_optional_num_seq(abstols, accept_none=True, caller=refine_caller(caller, 'abstols'))
            reltols_ = self._typecheck_optional_num_seq(reltols, accept_none=True, caller=refine_caller(caller, 'reltols'))
            self._set_engine_multi_objective_exprs(exprs, priorities, weights, abstols_, reltols_, names)
            self._multi_objective.update(exprs, priorities, weights, abstols, reltols, names)

    def _clear_multi_objective(self):
        # INTERNAL
        zero_exprs = [self._new_default_objective_expr()]
        self._set_engine_multi_objective_exprs(zero_exprs, priorities=[0], weights=[1],
                                               abstols=[0], reltols=[0], names=[None])
        self._multi_objective.clear()

    def _nth_multi_objective(self, multiobj_index):
        return self._multi_objective[multiobj_index]

    def _check_has_multi_objective(self, caller):
        if not self.has_multi_objective():
            self.fatal("{0} requires model with multi-objective models", caller)

    def set_multi_objective_abstols(self, abstols):
        """ Changes absolute tolerances for multiple objectives.

        Args:
            abstols: new absolute tolerances. Can be either a number (applies to all objectives),
                or a sequence of numbers.
                A sequence must have the same length as the number of objectives.
        *New in version 2.16*
        """
        self._check_has_multi_objective(caller='Model.set_multi_objective_abstols')
        nb_objectives = self.number_of_multi_objective_exprs
        abstols_ = self._typecheck_optional_num_seq(abstols, accept_none=False, expected_size=nb_objectives)
        self.get_engine().set_multi_objective_tolerances(abstols_, reltols=None)
        # update tolerances in multi obj
        self._multi_objective.update(new_abstols=abstols_, new_reltols=None)

    def set_multi_objective_reltols(self, reltols):
        """ Changes relative tolerances for multiple objectives.

        Args:
            reltols: new relative tolerances. Can be either a number (applies to all objectives),
                or a sequence of numbers.
                A sequence must have the same length as the number of objectives.

        *New in version 2.16*
        """
        self._check_has_multi_objective(caller='Model.set_multi_objective_reltols')
        nb_objectives = self.number_of_multi_objective_exprs
        reltols_ = self._typecheck_optional_num_seq(reltols, accept_none=False, expected_size=nb_objectives)
        self.get_engine().set_multi_objective_tolerances(abstols=None, reltols=reltols_)
        # update tolerances in multi obj
        self._multi_objective.update(new_abstols=None, new_reltols=reltols_)

    def _set_engine_multi_objective_exprs(self, exprs, priorities, weights, abstols, reltols, names):
        # INTERNAL
        old_multi_objective_exprs = self._multi_objective.exprs
        eng = self.__engine
        if eng:
            nb_exprs = len(exprs)
            eng.set_multi_objective_exprs(new_multiobjexprs=exprs,
                                          old_multiobjexprs=old_multi_objective_exprs,
                                          priorities=priorities, weights=weights,
                                          abstols=MultiObjective.as_optional_sequence(abstols, nb_exprs),
                                          reltols=MultiObjective.as_optional_sequence(reltols, nb_exprs),
                                          objnames=names)
        if old_multi_objective_exprs is not None:
            for expr in old_multi_objective_exprs:
                expr.notify_unsubscribed(subscriber=self)

    def clear_multi_objective(self):
        """ Clears everything related to multi-objective, if any.

        If the model had previously defined
        multi-objectives, resets the model with an objective of zero.
        If the model had not defined multi-objectives, this method does nothing.

        *New in version 2.10*
        """
        self._clear_multi_objective()

    def has_objective(self):
        # INTERNAL
        return not self._objective_expr.is_zero()

    def has_multi_objective(self):
        """ Returns True if the model has multi objectives defined

        *New in version 2.10*
        """
        return not self._multi_objective.empty()

    @property
    def number_of_multi_objective_exprs(self):
        return self._multi_objective.number_of_objectives

    def iter_multi_objective_tuples(self):
        return self._multi_objective.itertuples()

    def iter_multi_objective_exprs(self):
        return self._multi_objective.iter_exprs()

    def set_objective(self, sense, expr):
        """ Sets a new objective.

        Args:
            sense: Either an instance of :class:`docplex.mp.basic.ObjectiveSense` (Minimize or Maximize),
                or a string: "min" or "max".
            expr: Is converted to an expression. Accepted types are variables,
                linear expressions, quadratic expressions or numbers.

        Note:
            When using a number, the search will not optimize but only look for a feasible solution.

        """
        self.set_objective_sense(sense)
        self.set_objective_expr(expr)

    def set_objective_sense(self, sense):
        actual_sense = self._resolve_sense(sense)
        self._objective_sense = actual_sense
        eng = self.__engine
        if eng:
            # when ending the model, the engine is None here
            eng.set_objective_sense(actual_sense)

    @property
    def objective_sense(self):
        """ This property is used to get or set the direction of the optimization as an instance of
        :class:`docplex.mp.basic.ObjectiveSense`, either Minimize or Maximize.

        This property also accepts strings as arguments: 'min' for minimize and 'max' for maximize.

        """
        return self._objective_sense

    @objective_sense.setter
    def objective_sense(self, new_sense):
        self.set_objective_sense(new_sense)

    def set_objective_expr(self, new_objexpr, clear_multiobj=True):
        # INTERNAL
        if self.has_multi_objective() and clear_multiobj:
            # Need also to set all attributes to default values so that the model won't be treated as multi-objective
            self._clear_multi_objective()

        if new_objexpr is None:
            expr = self._new_default_objective_expr()
        else:
            expr = self._lfactory._to_expr(new_objexpr)
            #expr.keep()
            expr.notify_used(self)

        eng = self.__engine
        current_objective_expr = self._objective_expr
        if eng:
            # when ending the model, the engine is None here
            eng.set_objective_expr(expr, current_objective_expr)

        if current_objective_expr is not None:
            current_objective_expr.notify_unsubscribed(subscriber=self)
        self._objective_expr = expr

    def _clear_objective_expr(self):
        # INTERNAL
        current_objective_expr = self._objective_expr
        if not current_objective_expr.is_zero():
            eng = self.__engine
            current_objective_expr = self._objective_expr
            zero_expr = self._new_default_objective_expr()
            if eng:
                # when ending the model, the engine is None here
                eng.set_objective_expr(new_objexpr=zero_expr, old_objexpr=current_objective_expr)
            if current_objective_expr is not None:
                current_objective_expr.notify_unsubscribed(subscriber=self)
            self._objective_expr = zero_expr

    def get_objective_expr(self):
        """ This method returns the expression used as the model objective.

        Note:
            The default objective is a constant zero expression.

        Returns:
            an expression.
        """
        return self._objective_expr

    @property
    def objective_expr(self):
        """ This property is used to get or set the expression used as the model objective.
        """
        return self._objective_expr

    @objective_expr.setter
    def objective_expr(self, new_expr):
        self.set_objective_expr(new_expr)


    def notify_expr_modified(self, expr, event):
        # INTERNAL
        objexpr = self._objective_expr
        if event and expr is objexpr or expr is objexpr.linear_part:
            # old and new are the same
            self.__engine.update_objective(expr=expr, event=event)

    def notify_expr_replaced(self, old_expr, new_expr):
        if old_expr is self._objective_expr:
            self.__engine.set_objective_expr(new_objexpr=new_expr, old_objexpr=old_expr)
            new_expr.grab_subscribers(old_expr)

    def _new_default_objective_expr(self):
        # INTERNAL
        return self._lfactory.linear_expr(arg=None, constant=0, safe=True)


    def _can_solve(self):
        return self.has_cplex()


    def _make_end_infodict(self):  # pragma: no cover
        return self.solution.as_name_dict() if self.solution is not None else dict()

    def prepare_actual_context(self, **kwargs):
        # prepares the actual context that will be used for a solve

        # use the provided context if any, or the self.context otherwise
        if not kwargs:
            return self.context

        arg_context = kwargs.get('context') or self.context
        if not isinstance(arg_context, Context):
            self.fatal('Expecting instance of docplex.mp.Context, {0!r} was passed', arg_context)
        cloned = False
        context = arg_context

        # update the context with provided kwargs
        for argname, argval in six.iteritems(kwargs):
            # skip context argname if any
            if argname == "url" and (not is_url_valid(argval)) and context.solver.docloud.url:
                pass
            elif argname == 'clean_before_solve':
                pass
            elif argname != "context" and argval is not None:
                if not cloned:
                    context = context.override()
                    cloned = True
                context.update_key_value(argname, argval)

        return context

    def solve(self, **kwargs):
        """ Starts a solve operation on the model.

        Args:
            context (optional): An instance of context to be used in instead of
                the context this model was built with.
            cplex_parameters (optional): A set of CPLEX parameters to use
                instead of the parameters defined as
                ``context.cplex_parameters``.
                Accepts either a RootParameterGroup object (obtained by cloning the model's
                parameters), or a dict of path-like names and values.
            checker (optional): a string which controls which type of checking is performed.
                Possible values are:
                - 'std' (the default) performs type checks on arguments to methods; checks that numerical
                arguments are numbers, but will not check for NaN or infinity.
                - 'numeric' checks that numerical arguments are valid numbers, neither NaN nor
                math.infinity
                - 'full' performs all possible checks, the union of 'std' and 'numeric' checks.
                - 'off' performs no checking at all. Disabling all checks might improve performance, but only when
                it is safe to do so.
            log_output (optional): if ``True``, solver logs are output to
                stdout. If this is a stream, solver logs are output to that
                stream object. Overwrites the ``context.solver.log_output``
                parameter.
            clean_before_solve (optional): a boolean (default is False).
                Solve normally picks up where the previous solve left, but if this flag is set to ``True``,
                a fresh solve is started, with no memory of the previous solutions.

        Returns:
            A :class:`docplex.mp.solution.SolveSolution` object if the solve operation managed to create
            a feasible solution, else None.
            The reason why solve returned None includes not only errors, but also proper cases of infeasibilties
            or unboundedness. When solve returns None, use Model.solve_details to check the status
            of the latest solve operation: Model.solve_details always returns
            a :class:`docplex.mp.sdetails.SolveDetails` object, whether or not
            a solution has been found. This object contains detailed information about the latest solve operation.

        See Also:
            :func:`solve_details`
            :class:`docplex.mp.sdetails.SolveDetails`
        """
        if not self.is_optimized():
            self.info("No objective to optimize - searching for a feasible solution")

        lex_mipstart = kwargs.pop('_lex_mipstart', None)
        lex_timelimits = kwargs.pop('lex_timelimits', None)
        lex_mipgaps = kwargs.pop('lex_mipgaps', None)
        context = self.prepare_actual_context(**kwargs)

        if lex_timelimits is not None and len(lex_timelimits) == 1:
            # This will be handled as a single-objective ==> set timelimit parameter
            context.cplex_parameters.timelimit = lex_timelimits[0]
        if lex_mipgaps is not None and len(lex_mipgaps) == 1:
            # This will be handled as a single-objective ==> set mipgap parameter
            context.cplex_parameters.mip.tolerances.mipgap = lex_mipgaps[0]


        # log stuff
        a_stream = context.solver.log_output_as_stream
        with OverridenOutputContext(self, a_stream):
        #try:
            forced_docloud = context_must_use_docloud(context, **kwargs)
            have_credentials = context_has_docloud_credentials(context, do_warn=True)

            if forced_docloud:
                if have_credentials:
                    return self._solve_cloud(context, lex_mipstart=lex_mipstart)
                else:
                    self.fatal("DOcplexcloud context has no valid credentials: {0!s}",
                               context.solver.docloud)

            # from now on docloud_context is None
            elif self.environment.has_cplex:
                # take arg clean flag or this model's
                used_clean_before_solve = kwargs.get('clean_before_solve', self.clean_before_solve)
                return self._solve_local(context, used_clean_before_solve, lex_timelimits, lex_mipgaps)
            elif have_credentials:
                # no context passed as argument, no Cplex installed, try model's own context
                return self._solve_cloud(context, lex_mipstart=lex_mipstart)
            else:
                # no way to solve.. really
                return self.fatal("Cannot solve model: no CPLEX runtime found.")


    def _connect_progress_listeners(self):
        self.__engine.connect_progress_listeners(self._progress_listeners, self)

    def _disconnect_progress_listeners(self):
        for pl in self._progress_listeners:
            pl._disconnect()

    def _notify_solve_hit_limit(self, solve_details):
        # INTERNAL
        if solve_details and solve_details.has_hit_limit():
            self.info("solve: {0}".format(solve_details.status))

    def _solve_local(self, context, clean_before_solve=None, lex_timelimits=None, lex_mipgaps=None):
        """ Starts a solve operation on the local machine.

        Note:
        This method will never try to solve on DOcplexcloud, regardless of whether the model
        has an attached DOcplexcloud context.
        If CPLEX is not available, an error is raised.

        Args:
            context: a (possibly new) context whose parameters override those of the modle
                during this solve.

        Returns:
            A Solution object if the solve operation succeeded, None otherwise.

        """
        local_solve_env = CplexLocalSolveEnv(self)
        params_to_use = local_solve_env.before_solve(context)

        # auto_publish_details = is_auto_publishing_solve_details(context)
        # auto_publish_solution = (auto_publishing_result_output_names(context) is not None)
        # auto_publish_kpis_table = (auto_publishing_kpis_table_names(context) is not None)
        #
        # self.notify_start_solve()
        #
        # the_env = get_environment()
        # if is_in_docplex_worker() and auto_publish_details:
        #     # publish kpi automatically
        #     env_kpi_hook = lambda kpd: the_env.update_solve_details(kpd)
        #     self.kpi_recorder = _KpiRecorder(self,
        #                                     clock=context.solver.kpi_reporting.filter_level,
        #                                     publish_hook=env_kpi_hook)
        #     self.add_progress_listener(self.kpi_recorder)
        #
        #
        # # connect progress listeners (if any) if problem is mip
        # self._connect_progress_listeners()
        #
        # # call notifyStart on progress listeners
        # self._fire_start_solve_listeners()
        #
        # # The following block used to be published only if auto_publish_details.
        # # It is now modified so that notify start is always performed,
        # # then we update solve details only if they need to be published
        # # [[[
        # self_stats = self.get_statistics()
        # kpis = make_new_kpis_dict(allkpis=self._allkpis,
        #                           int_vars=self_stats.number_of_integer_variables,
        #                           continuous_vars=self_stats.number_of_continuous_variables,
        #                           linear_constraints=self_stats.number_of_linear_constraints,
        #                           bin_vars=self_stats.number_of_binary_variables,
        #                           quadratic_constraints=self_stats.number_of_quadratic_constraints,
        #                           total_constraints=self_stats.number_of_constraints,
        #                           total_variables=self_stats.number_of_variables)
        # # implementation for https://github.ibm.com/IBMDecisionOptimization/dd-planning/issues/2491
        # problem_type = self._get_cplex_problem_type()
        # kpis['STAT.cplex.modelType'] = problem_type
        # kpis['MODEL_DETAIL_OBJECTIVE_SENSE'] = self._objective_sense.verb
        # # self_solve_hooks if for backward compatibility only. Should
        # # really use env instead
        # for h in self._solve_hooks:
        #     h.notify_start_solve(self, kpis)  # pragma: no cover
        # the_env.notify_start_solve(kpis)
        # if auto_publish_details:
        #     get_environment().update_solve_details(kpis)
        # # ]]]
        # # --- solve is protected in try/except block
        # has_solution = False
        # reported_obj = 0
        # engine_status = self._unknown_status
        #
        #
        # self_params = self.context._get_raw_cplex_parameters()
        # parameters = apply_thread_limitations(context)
        # if self_params and parameters is not self_params:
        #     saved_params = {p: p.get() for p in self_params}
        # else:
        #     saved_params = {}
        #
        # new_solution = None
        #
        self_engine = self.__engine
        new_solution = None
        try:
            used_parameters = params_to_use or self.context._get_raw_cplex_parameters()
            # assert used_parameters is not None
            self._apply_parameters_to_engine(used_parameters)

            new_solution = self_engine.solve(self,
                                             parameters=used_parameters,
                                             clean_before_solve=clean_before_solve,
                                             lex_timelimits=lex_timelimits,
                                             lex_mipgaps=lex_mipgaps)
            self._set_solution(new_solution)

            # store solve status as returned by the engine.
            engine_status = self_engine.get_solve_status()
            self._last_solve_status = engine_status

        except DOcplexException as docpx_e:  # pragma: no cover
            self._set_solution(None)
            raise docpx_e

        except Exception as e:
            self._set_solution(None)
            print("----------------- Python exception: {}".format(str(e)))
            raise e

        finally:
            self._set_solution(new_solution)
            local_solve_env.after_solve(context, new_solution, self_engine)
            # solve_details = self_engine.get_solve_details()
            # self._notify_solve_hit_limit(solve_details)
            # self._solve_details = solve_details
            # self._fire_end_solve_listeners(has_solution, reported_obj)
            # self._disconnect_progress_listeners()
            #
            # # call hooks
            # the_env = get_environment()
            # the_env.notify_end_solve(engine_status)
            # if auto_publish_details:
            #     # Only kept for backward compatibility
            #     for h in self._solve_hooks:
            #         h.notify_end_solve(self, has_solution, engine_status, reported_obj,
            #                            self._make_end_infodict())  # pragma: no cover
            #         h.update_solve_details(solve_details.as_worker_dict())  # pragma: no cover
            #     # actual right mean to do it
            #     details = {}
            #     details.update(solve_details.as_worker_dict())
            #     if new_solution:
            #         kpis = self.kpis_as_dict(new_solution, use_names=True)
            #
            #         def publish_name_fn(kn):
            #             return 'KPI.%s' % kn
            #
            #         name_values = {publish_name_fn(kn): kv for kn, kv in iteritems(kpis)}
            #         details.update(name_values)
            #         details['PROGRESS_CURRENT_OBJECTIVE'] = new_solution.objective_value
            #     the_env.update_solve_details(details)
            #
            # # save solution
            # if new_solution:
            #     if auto_publish_solution:
            #         write_result_output(env=the_env,
            #                             context=context,
            #                             model=self,
            #                             solution=new_solution)
            #
            #     # save kpi
            #     if auto_publish_kpis_table:
            #         write_kpis_table(env=the_env,
            #                          context=context,
            #                          model=self,
            #                          solution=new_solution)
            #
            # # restore parameters in sync with model, if necessary
            # if saved_params:
            #     for p, v in six.iteritems(saved_params):
            #         self_engine.set_parameter(p, v)


        return new_solution

    def get_solve_status(self):
        """ Returns the solve status of the last successful solve.

        If the model has been solved successfully, returns the status stored in the
        model solution. Otherwise returns None.

        :returns: The solve status of the last successful solve, a enumerated value of type
            `docplex.utils.JobSolveStatus`

        Note: The status returned by Cplex is stored as `status` in the solve_details of the model.

        >>> m.solve_details.status

        See Also:
            :func:`docplex.mp.SolveDetails.status` to get the Cplex status as a string (eg. "optimal")
            :func:`docplex.mp.SolveDetails.status_code` to get the Cplex status as an integer code..
        """
        warnings.warn("Model.get_solve_status() is deprecated with cloud solve, use Model.solve_details instead", DeprecationWarning)
        return self._last_solve_status

    @property
    def solve_status(self):
        """ Returns the solve status of the last successful solve.

        If the model has been solved successfully, returns the status stored in the
        model solution. Otherwise returns None`.

        :returns: The solve status of the last successful solve, a string, or None.
        """
        warnings.warn("Model.solve_status is deprecated with cloud solve, use Model.solve_details instead", DeprecationWarning)
        return self._last_solve_status

    def _new_docloud_engine(self, ctx):
        return self._engine_factory.new_docloud_engine(model=self,
                                                       docloud_context=ctx.solver.docloud,
                                                       log_output=ctx.solver.log_output_as_stream)

    def _solve_cloud(self, context, **kwargs):
        warnings.warn("Model solving on Docplexcloud is deprecated", DeprecationWarning)
        lex_mipstart = kwargs.get('lex_mipstart')

        cloud_solve_env = DocloudSolveEnv(self)
        parameters_to_use = cloud_solve_env.before_solve(context)

        # see if we can reuse the local docloud engine if any?
        docloud_engine = self._new_docloud_engine(context)
        new_solution = docloud_engine.solve(self, parameters=parameters_to_use, lex_mipstart=lex_mipstart)
        self._set_solution(new_solution)
        cloud_solve_env.after_solve(context, new_solution, docloud_engine)
        return new_solution

    def solve_cloud(self, context=None):
        # Starts execution of the model on the cloud.
        #
        # This method accepts a context (an instance of Context) to be used when
        # solving on the cloud. If the context argument is None or invalid, then it will
        # use the model's own instance of Context, set at model creation time.
        #
        # Note:
        #    This method will always solve the model on the cloud, whether or not CPLEX
        #    is available on the local machine.
        #
        # Args:
        #    context: An optional context to use on the cloud. If None, uses the model's Context instance, if any.
        #
        # :returns: A :class:`docplex.mp.solution.SolveSolution` object if the solve operation succeeded, else None.
        if not context:
            if self.context.solver.docloud:
                if self.__engine.name == 'docloud':
                    return self.solve()
                else:
                    return self._solve_cloud(self.context)
            else:
                self.fatal("context is None: cannot solve on the cloud")
        elif has_credentials(context.solver.docloud):
            return self._solve_cloud(context)
        else:
            self.fatal("DOcplexcloud context has no valid credentials: {0!s}", context.solver.docloud)

    def notify_start_solve(self):
        # INTERNAL
        pass

    def notify_solve_failed(self):
        pass

    @property
    def solve_details(self):
        """
        This property returns detailed information about the latest solve,
        an instance of :class:`docplex.mp.solution.SolveDetails`.

        When the latest solve did return a Solution instance, this property
        returns the solve details corresponding to the solution; when no
        solution has been found (in other terms, the latest solve operation
        returned None), it still returns a SolveDetails object, containing a
        CPLEX code identifying the reason why no solution could be found
        (for example, infeasibility or unboundedness).

        See Also:
            :class:`docplex.mp.sdetails.SolveDetails`

        """
        from copy import copy as shallow_copy

        return shallow_copy(self._solve_details)

    def get_solve_details(self):
        return self.solve_details

    def notify_solve_relaxed(self, relaxed_solution, solve_details):
        # INTERNAL: used by relaxer
        self._solve_details = solve_details
        self._set_solution(relaxed_solution)
        if relaxed_solution is not None:
            self.notify_start_solve()
        else:
            self.notify_solve_failed()

    def _resolve_sense(self, sense_arg):
        """
        INTERNAL
        :param sense_arg:
        :return:
        """
        return ObjectiveSense.parse(sense_arg, self.logger, default_sense=None)  # raise if invalid

    def solve_with_goals(self, goals,
                         senses='min',
                         abstols=None,
                         reltols=None,
                         write_pass_files=False,
                         solution_callbackfn=None,
                         **kwargs):
        """ Performs a solve from an ordered collection of goals.

        :param goals: An ordered collection of linear expressions.

        :param senses: Accepts ither an ordered sequence of senses, one sense, or
           None. The default is None, in which case the solve uses a Minimize
           sense. Each sense can be either a sense object, that is either
           `ObjectiveSense.Minimize` or `Maximize`, or a string "min" or "max".

        :param abstols: if defined, accepts either a number or a list of numbers having the same size as the `exprs` argument,
            interpreted as absolute tolerances. If passed asingle number, this tolerance number will be used for all
            passes.

        :param reltols: if defined, accepts either a number or a list of numbers having the same size as the `exprs` argument,
            interpreted as absolute tolerances. If passed asingle number, this tolerance number will be used for all
            passes.

        Note:
            tolerances are used at each step to constraint the previous
            objective value to be be 'no worse' than the value found in the
            last pass. For example, if relative tolerance is 2% and pass #1 has
            found an objective of 100, then pass #2 will constraint the first
            goal to be no greater than 102 if minimizing, or
            no less than 98, if maximizing.

            If one pass fails, return the previous pass' solution. If the solve fails at the first
            goal, then return None.

        Return:
            If successful, returns a tuple with all pass solutions, reversed else None.
            The current solution of the model is the first solution in the tuple.
        """
        if not goals:
            self.error("solve_with_goals requires a non-empty list of goals, got: {0!r}".format(goals))
            return None
        if not is_indexable(goals):
            self.fatal("solve_with_goals requires an indexable sequence of goals, got: {0!s}", goals)

        actual_goals = [(g.name or "goal%d" % (gi + 1), self._lfactory._to_expr(g))  for gi, g in enumerate(goals)]
        nb_goals = len(actual_goals)
        # --- senses ---
        abstols_ = []
        reltols_ = []

        # compile tolerances to abstols, reltols

        if abstols:
            abstols_ = self._typecheck_optional_num_seq(abstols, expected_size=nb_goals, caller='Model.solve_with_goals')
        if reltols:
            reltols_ = self._typecheck_optional_num_seq(reltols, expected_size=nb_goals, caller='Model.solve_with_goals')

        if not abstols_:
            abstols_ = [1e-6] * nb_goals
        if not reltols_:
            reltols_ = [1e-4] * nb_goals

        old_objective_expr = self._objective_expr
        old_objective_sense = self._objective_sense

        pass_count = 0
        m = self
        results = []

        if not is_iterable(senses, accept_string=False):
            senses = generate_constant(ObjectiveSense.parse(senses), count_max=nb_goals)

        def lex_info(msg):
            self.info("lexicographic: {0}".format(msg))

        # keep extra constraints, in order to remove them at the end.
        extra_cts = []
        cplex_param_key = Context.cplex_parameters_key
        ctx_params = kwargs.get(cplex_param_key)
        iter_pass_params = generate_constant(None, nb_goals)
        baseline_params = None  # parameters to restore at each iteration, default is to do nothing
        if ctx_params and is_iterable(ctx_params):
            # must pop out the list as normal solve won't have it.
            pass_params = list(kwargs.pop(cplex_param_key))
            if len(pass_params) != nb_goals:
                self.fatal("List of parameters should have same length as goals, expecting: {0} but a list of size {1} was passed",
                           nb_goals, pass_params)
            else:
                iter_pass_params = iter(pass_params)
                baseline_params = m.context.parameters  # need to clear/reset parameters at each pass

        # --- main loop ---
        prev_step = (None, None, None)
        all_solutions = []
        solve_kwargs = kwargs.copy()
        current_sol = None
        try:
            for (goal_name, goal_expr), next_sense, abstol, reltol in izip(actual_goals, senses, abstols_, reltols_):
                if goal_expr.is_constant() and pass_count > 1:
                    self.warning("Constant expression in lexicographic solve: {0!s}, skipped", goal_expr)
                    continue
                pass_count += 1

                if pass_count > 1:
                    prev_goal, prev_obj, prev_sense = prev_step
                    tolerance = max(abstol, reltol * abs(prev_obj))
                    if prev_sense.is_minimize():
                        pass_ct = m._post_constraint(prev_goal <= prev_obj + tolerance)
                    else:
                        pass_ct = m._post_constraint(prev_goal >= prev_obj - tolerance)
                    pass_ct.name = "lex_{0}_ct".format(pass_count)
                    lex_info("pass #{0} generated constraint with rhs: {1}, tolerance={2:.3f}"
                             .format(pass_count, str(pass_ct.rhs), tolerance))
                    extra_cts.append(pass_ct)


                sense = self._resolve_sense(next_sense)

                lex_info("starting pass %d, %s: %s" % (pass_count, sense.verb, str_maxed(goal_expr, 64)))
                m.set_objective(sense, goal_expr)

                if write_pass_files:  # pragma: no cover
                    pass_basename = 'lex_%s_%s#%d' % (self.name, goal_name, pass_count)
                    dump_path = self.export_as_sav(basename=pass_basename)
                    lex_info("saved pass file: {0}".format(dump_path))

                # --- update pass parameters, if any
                pass_param = next(iter_pass_params)
                if pass_param:
                    # print('applying custom parameters')
                    # pass_param.print_information()
                    m.context.update_cplex_parameters(pass_param)
                    m.context.cplex_parameters.print_information()
                # ---
                if current_sol and pass_count > 1:
                    solve_kwargs['_lex_mipstart'] = current_sol

                current_sol = m.solve(**solve_kwargs)
                # restore params if need be
                if baseline_params:
                    m.context.cplex_parameters = baseline_params

                if current_sol is not None:
                    current_sol.set_name("lex_{0}_{1}_{2}".format(self.name, goal_name, pass_count))
                    current_obj = current_sol.objective_value
                    results.append(current_obj)
                    prev_step = (goal_expr, current_obj, sense)
                    all_solutions.append(current_sol)
                    lex_info("objective value for pass #{0} is: {1}".format(pass_count, current_sol.objective_value))
                    if solution_callbackfn:
                        solution_callbackfn(current_sol)


                else:  # pragma: no cover
                    sd = m.solve_details
                    status = sd.status
                    self.error("lexicographic: pass {0} fails, status={1} ({2}), stopping",
                               pass_count, status, sd.status_code)
                    break
        finally:
            # print("-> start restoring model at end of lexicographic")
            while extra_cts:
                # using LIFO logic to avoid holes in indices.
                ct_to_remove = extra_cts.pop()
                # print("* removing constraint: name: {0}, idx: {1}".format(ct_to_remove.name, ct_to_remove.index))
                self.pop_constraint() # self._remove_constraint_internal(ct_to_remove)
            # restore objective whatsove
            self.set_objective(old_objective_sense, old_objective_expr)
            # print("<- end restoring model at end of lexicographic")

        # return a solution or None
        return tuple(reversed(all_solutions))


    def _has_solution(self):
        # INTERNAL
        return self._solution is not None

    def _set_solution(self, new_solution):
        """
        INTERNAL: Sets this solution as the model's current solution.
        Copies values to variables (for now, let's think more about this)
        :param new_solution:
        :return:
        """
        self._solution = new_solution

    def _check_has_solution(self):
        # see if we can refine messages here...
        if self._solution is None:
            if self._solve_details is None:
                self.fatal("Model<{0}> has not been solved yet", self.name)
            else:
                self.fatal("Model<{0}> did not solve successfully", self.name)

    def _check_solved_as_mip(self, caller, do_raise):
        if self._check_mip_for_mipstarts:
            if not self._solved_as_mip():
                msg = "{0}  is only available for MIP problems".format(caller)
                if do_raise:
                    self.fatal(msg)
                else:
                    self.error(msg)
                return False
        # either a MIP or we don't care...
        return True

    def add_mip_start(self, mip_start_sol, effort_level=None):
        """  Adds a (possibly partial) solution to use as a starting point for a MIP.

        This is valid only for models with binary or integer decision variables.
        The given solution must contain the value for at least one binary or integer variable.

        This feature is also known as 'warm start'.

        Args:
            mip_start_sol (:class:`docplex.mp.solution.SolveSolution`): The solution object to use as a starting point.
            effort_level: an optional enumerated value of class :class:`docplex.mp.constants.EffortLevel`, or None.

        """
        if self._check_solved_as_mip(caller="Model.add_mip_start", do_raise=False):
            try:
                mip_start_ = mip_start_sol.as_mip_start()
                mip_start_.check_as_mip_start()
                effort = EffortLevel.parse(effort_level)
                self._mipstarts.append((mip_start_, effort))
            except AttributeError:
                self.fatal("add_mip_starts expects solution, {0!r} was passed", mip_start_sol)

    @property
    def mip_starts(self):
        """ This property returns the list of MIP start solutions (a list of instances of :class:`docplex.mp.solution.SolveSolution`)
        attached to the model if MIP starts have been defined, possibly an empty list.
        """
        warnings.warn("Model.mip_starts is deprecated. Use Model.iter_mip_starts instead"
                      , DeprecationWarning, stacklevel=2)
        return [s for (s, _) in self.iter_mip_starts()]

    @property
    def number_of_mip_starts(self):
        """ This property returns the number of MIP start associated with the model.

        *New in version 2.10*
         """
        return len(self._mipstarts)

    def iter_mip_starts(self):
        """ This property returns an iterator on the MIP starts associated with
         the model.

         It returns tuples of size 2:
            - first element is a solution (an instance of :class:`docplex.mp.solution.SolveSolution`)
            - second is an enumerated value of type :class:`docplex.mp.constants.EffortLevel`

        *New in version 2.10*

        """
        return iter(self._mipstarts)

    def clear_mip_starts(self):
        """  Clears all MIP starts associated with the model.
        """
        self._mipstarts = []

    def read_mip_starts(self, mst_path):
        """ Read MIP starts from a file.

        Reads the file and returns a list of (solution, effort_level) tuples.

        :param mst_path: the path to mip start file (in CPLEX MST file format)

        :return: a list of tuples of size 2; the first element is an instance of `SolveSolution`
            and the second element is an enumerated value of type `EffortLevel`

        See Also:
            :class: `docplex.mp.constants.EffortLevel`
            :class: `docplex.mp.solution.SolveSolution`

        * New in version 2.10*

        """
        self._check_solved_as_mip(caller="Model.read_mip_starts", do_raise=True)

        from docplex.mp.mst_xml_reader import read_mst_file
        StaticTypeChecker.check_file(self, mst_path, "MST", expected_extensions=(".mst", ".xml"),
                                     caller='Model.read_mip_starts')

        self.info("Reading mip starts from file: {0}".format(mst_path))
        mip_starts = read_mst_file(mst_path, self)
        if mip_starts is not None:
            if not mip_starts:
                self.warning("Found no MIP starts in file: {0}", mst_path)
            else:
                self.info("Read {0} MIP starts in file: {1}".format(len(mip_starts), mst_path))
            self._mipstarts = mip_starts
            return mip_starts
        else:
            # do not overwrite current mip starts (IMHO)
            return None

    def set_lp_start_basis(self, dvar_stats, lct_stats):
        """ Provides an initial basis for a LP problem.

        :param dvar_stats: an ordered sequence (list) of basis status objects, one for
            each decision variable in the model.
        :param lct_stats: an ordered sequence (list) of basis status objects, one for each linear constraint
            in the model

        Note:
            Basis status are values of the enumerated type :class:`docplex.mp.constants.BasisStatus`.

        See Also:
             :class:`docplex.mp.constants.BasisStatus`.

        * New in version 2.10*

        """
        l_dvar_stats = StaticTypeChecker.typecheck_initial_lp_stats\
            (logger=self, stats=dvar_stats, stat_type='variable', caller='Model.set_lp_start_basis')
        l_lct_stats = StaticTypeChecker.typecheck_initial_lp_stats\
            (logger=self, stats=lct_stats, stat_type='constraint', caller='Model.set_lp_start_basis')
        self.__engine.set_lp_start(l_dvar_stats, l_lct_stats)

    @property
    def objective_value(self):
        """ This property returns the value of the objective expression in the solution of the last solve.
        In case of a multi-objective, only the value of the first objective is returned

        Raises an exception if the model has not been solved successfully.

        """
        self._check_has_solution()
        return self._objective_value()

    def _objective_value(self):
        return self.solution.objective_value

    @property
    def multi_objective_values(self):
        """ This property returns the list of values of the objective expressions in the solution of the last solve.

        Raises an exception if the model has not been solved successfully.

        *New in version 2.9*
        """
        self._check_has_solution()
        return self._multi_objective_values()

    def _multi_objective_values(self):
        # INTERNAL
        return self.solution.multi_objective_values

    @property
    def blended_objective_values(self):
        """ This property returns the list of values of the blended objective expressions based on the decreasing
        order of priorities in the solution of the last solve.

        Raises an exception if the model has not been solved successfully.

        *New in version 2.9.*
        """
        self._check_has_solution()
        blended_obj_values = self.solution.get_blended_objective_value_by_priority()
        return blended_obj_values

    def _reported_objective_value(self, failure_obj=0):
        return self.solution.objective_value if self.solution else failure_obj

    def _resolve_path(self, path_arg, basename_arg, extension):
        # INTERNAL
        if is_string(path_arg):
            if os.path.isdir(path_arg):
                if path_arg == ".":
                    path_arg = os.getcwd()
                return self._make_output_path(extension, basename_arg, path_arg)
            else:
                # add extension if not present (but not twice!)
                return path_arg if path_arg.endswith(extension) else path_arg + extension
        else:
            assert path_arg is None
            return self._make_output_path(extension, basename_arg, path_arg)

    def _make_output_path(self, extension, basename, path=None):
        return make_output_path2(self.name, extension, basename, path)

    def _get_printer(self, format_spec, do_raise=False, silent=False):
        # INTERNAL
        printer_kwargs = {'full_obj': self._print_full_obj}
        format_ = parse_format(format_spec)
        printer = None
        if format_.name == 'LP':
            printer = LPModelPrinter(**printer_kwargs)
        else:
            if do_raise:
                self.fatal("Unsupported output format: {0!s}", format_spec)
            elif not silent:
                self.error("Unsupported output format: {0!s}", format_spec)
        return printer

    def dump_as_lp(self, path=None, basename=None):
        return self._export_from_cplex(path, basename, format_spec="lp")

    def export_as_lp(self, path=None, basename=None, hide_user_names=False):
        """ Exports a model in LP format.

        Args:
            basename: Controls the basename with which the model is printed.
                Accepts None, a plain string, or a string format.
                if None, uses the model's name;
                if passed a plain string, the string is used in place of the model's name;
                if passed a string format (either with %s or {0}, it is used to format the
                model name to produce the basename of the written file.

            path: A path to write file, expects a string path or None.
                can be either a directory, in which case the basename
                that was computed with the basename argument, is appended to the directory to produce
                the file.
                If given a full path, the path is directly used to write the file, and
                the basename argument is not used.
                If passed None, the output directory will be ``tempfile.gettempdir()``.

            hide_user_names: A Boolean indicating whether or not to keep user names for
                variables and constraints. If True, all names are replaced by `x1`, `x2`, ... for variables,
                and `c1`, `c2`, ... for constraints.

        Returns:
            The full path of the generated file, or None if an error occured.

        Examples:
            Assuming the model's name is `mymodel`:
            
            >>> m.export_as_lp()
            
            will write ``mymodel.lp`` in ``gettempdir()``.
            
            >>> m.export_as_lp(basename="foo")
            
            will write ``foo.lp`` in ``gettempdir()``.
            
            >>> m.export_as_lp(basename="foo", path="e:/home/docplex")

            will write file ``e:/home/docplex/foo.lp``.
            
            >>> m.export_as_lp("e/home/docplex/bar.lp")
            
            will write file ``e:/home/docplex/bar.lp``.
            
            >>> m.export_as_lp(basename="docplex_%s", path="e/home/") 
            
            will write file ``e:/home/docplex/docplex_mymodel.lp``.
        """
        return self.export(path, basename, hide_user_names=hide_user_names, format_spec='lp')

    def export_as_sav(self, path=None, basename=None):
        """ Exports a model in CPLEX SAV format.

        Exporting to SAV format requires that CPLEX is installed and
        available in PYTHONPATH. If the CPLEX DLL cannot be found, an exception is raised.

        Args:
            basename: Controls the basename with which the model is printed.
                Accepts None, a plain string, or a string format.
                If None, the model's name is used.
                If passed a plain string, the string is used in place of the model's name.
                If passed a string format (either with %s or {0}), it is used to format the
                model name to produce the basename of the written file.

            path: A path to write the file, expects a string path or None.
                Can be a directory, in which case the basename
                that was computed with the basename argument, is appended to the directory to produce
                the file.
                If given a full path, the path is directly used to write the file, and
                the basename argument is not used.
                If passed None, the output directory will be ``tempfile.gettempdir()``.

        Returns:
            The full path of the generated file, or None if an error occured.

        Examples:
            See the documentation of  :func:`export_as_lp` for examples of pathname generation.
            The logic is identical for both methods.

        """
        return self._export_from_cplex(path, basename, format_spec="sav")

    dump_as_sav = export_as_sav

    def export_as_mps(self, path=None, basename=None):
        """ Exports a model in MPS format.

        Exporting to MPS format requires that CPLEX is installed and
        available in PYTHONPATH. If the CPLEX DLL cannot be found, an exception is raised.

        Args:
            basename: Controls the basename with which the model is printed.
                Accepts None, a plain string, or a string format.
                If None, the model's name is used.
                If passed a plain string, the string is used in place of the model's name.
                If passed a string format (either with %s or {0}), it is used to format the
                model name to produce the basename of the written file.

            path: A path to write the file, expects a string path or None.
                Can be a directory, in which case the basename
                that was computed with the basename argument, is appended to the directory to produce
                the file.
                If given a full path, the path is directly used to write the file, and
                the basename argument is not used.
                If passed None, the output directory will be ``tempfile.gettempdir()``.

        Returns:
            The full path of the generated file, or None if an error occured.

        Examples:
            See the documentation of  :func:`export_as_lp` for examples of pathname generation.
            The logic is identical for both methods.

        """
        return self._export_from_cplex(path, basename, format_spec="mps")

    def export_as_savgz(self, path=None, basename=None):
        """ Exports a model in compressed SAV format.

        Exporting to SAV compressed format requires that CPLEX is installed and
        available in PYTHONPATH. If the CPLEX DLL cannot be found, an exception is raised.

        Arguments 'path' and 'basename' have similar usage as for :func:`export_as_lp`.

        Returns:
            The full path of the generated file, or None if an error occured.

        Examples:
            See the documentation of  :func:`export_as_lp` for examples of pathname generation.
            The logic is identical for both methods.

        *New In 2.19*

        """
        return self._export_from_cplex(path, basename, format_spec="sav.gz")


    def _export_from_cplex(self, path=None, basename=None, hide_user_names=False,
                           format_spec="lp"):
        return self._export(path, basename,
                            use_engine=True,
                            hide_user_names=hide_user_names,
                            format_spec=format_spec)

    def export(self, path=None, basename=None,
               hide_user_names=False, format_spec="lp"):
        # INTERNAL
        return self._export(path, basename,
                            use_engine=False,
                            hide_user_names=hide_user_names,
                            format_spec=format_spec)

    def _export(self, path=None, basename=None,
                use_engine=False, hide_user_names=False,
                format_spec="lp"):
        # INTERNAL
        # path is either a nonempty path string or None
        self._checker.typecheck_string(path, accept_none=True, accept_empty=False)
        self._checker.typecheck_string(basename, accept_none=True, accept_empty=False)
        # INTERNAL
        _format = parse_format(format_spec)
        if not _format:
            self.fatal("Not a supported exchange format: {0!s}", format_spec)
        extension = _format.extension

        # combination of path/directory and basename resolution are done in resolve_path
        path = self._resolve_path(path, basename, extension)
        ret = self._export_to_path(path, hide_user_names, use_engine, _format)
        if ret:
            self.trace("model file: {0} overwritten", path)
        return ret

    def _export_to_path(self, path, hide_user_names=False, use_engine=False, format_spec="lp"):
        # INTERNAL
        format_ = parse_format(format_spec)
        try:
            if use_engine:
                # rely on engine for the dump
                if self.has_cplex():
                    self.__engine.export(path, format_)
                else:  # pragma: no cover
                    self.fatal(
                        "Exporting to {0} requires CPLEX, but a local CPLEX installation could not be found, file: {1} could not be written",
                        format_.name, path)
                    return None
            else:
                # a path is not a stream but anyway it will work
                self._export_to_stream(stream=path, hide_user_names=hide_user_names, format_spec=format_)
            return path

        except IOError:
            self.error("Cannot open file: \"{0}\", model: {1} not exported".format(path, self.name))
            raise

    def _export_to_stream(self, stream, hide_user_names=False, format_spec="lp"):
        format_ = parse_format(format_spec)
        printer = self._get_printer(format_, do_raise=False, silent=True)
        if printer:
            printer.set_mangle_names(hide_user_names)
            printer.printModel(self, stream)
        else:
            self.__engine.export(stream, format_spec)

    def export_to_stream(self, stream, hide_user_names=False, format_spec="lp"):
        """ Export the model to an output stream in LP format.

        A stream can be one of:
            - a string, interpreted as a system path,
            - None, interpreted as `stdout`, or
            - a Python file-type object (e.g. a StringIO() instance).
                
        Args:
            stream: An object defining where the output will be sent.
            
            hide_user_names: An optional Boolean indicating whether or not to keep user names for
                variables and constraints. If True, all names are replaced by `x1`, `x2`, ... for variables,
                and `c1`, `c2`, ,... for constraints. Default is to keep user names.

        """
        self._export_to_stream(stream, hide_user_names, format_spec)

    def export_as_lp_string(self, hide_user_names=False):
        """ Exports the model to a string in LP format.

        The output string contains the model in LP format.

        Args:
            hide_user_names: An optional Boolean indicating whether or not to keep user names for
                variables and constraints. If True, all names are replaced by `x1`, `x2`, ... for variables,
                and `c1`, `c2`, ... for constraints. Default is to keep user names.

        Returns:
            A string, containing the model exported in LP format.
        """
        return self.export_to_string(hide_user_names, "lp")

    @property
    def lp_string(self):
        """ This property returns a string encoding the model in LP format.

            *New in version 2.16*
        """
        return self.export_as_lp_string()

    def export_as_mps_string(self):
        """ Exports the model to a string in MPS format.

        Returns:
            A string, containing the model exported in MPS format.

        *New in version 2.13*
        """
        return self._export_as_cplex_string("mps")

    def export_as_sav_string(self):
        """ Exports the model to a string of bytes in SAV format.

        Returns:
            A string of bytes..

        *New in version 2.13*
        """
        return self._export_as_cplex_string("sav")

    def _export_as_cplex_string(self, format_spec):
        # INTERNAL
        _format = parse_format(format_spec)
        if not self.has_cplex():
            self.fatal("Exporting to {0} requires CPLEX, but a local CPLEX installation could not be found",
                       _format.name)

        from io import BytesIO
        bs = BytesIO()
        self.__engine.export(bs, _format)
        raw_res = bs.getvalue()
        if _format.is_binary:
            # for b, by in enumerate(raw_res):
            #     nl = (b % 21 == 20)
            #     print(f" {by}", end='\n' if nl else '')
            # print()
            return raw_res
        else:
            return raw_res.decode(self.parameters.read.fileencoding.get())

    def export_to_string(self, hide_user_names=False, format_spec="lp"):
        # INTERNAL
        oss = StringIO()
        self._export_to_stream(oss, hide_user_names, format_spec)
        return oss.getvalue()

    def export_parameters_as_prm(self, path=None, basename=None):
        # path is either a nonempty path string or None
        self._checker.typecheck_string(path, accept_none=True, accept_empty=False)
        self._checker.typecheck_string(basename, accept_none=True, accept_empty=False)

        # combination of path/directory and basename resolution are done in resolve_path
        prm_path = self._resolve_path(path, basename, extension='.prm')
        self.parameters.export_prm_to_path(path=prm_path)
        return prm_path

    def export_annotations(self, path=None, basename=None):
        from docplex.mp.anno import ModelAnnotationPrinter

        self._checker.typecheck_string(path, accept_none=True, accept_empty=False)
        self._checker.typecheck_string(basename, accept_none=True, accept_empty=False)

        # combination of path/directory and basename resolution are done in resolve_path
        anno_path = self._resolve_path(path, basename, extension='.ann')
        ap = ModelAnnotationPrinter()
        ap.print_to_stream(self, anno_path)

        return anno_path

    def _check_problem_type(self, feature, requires_solution=True, accept_qxp=True):
        if self._solve_details is None:
            self.fatal('{0} are not available, model is not solved yet'.format(feature))
        elif requires_solution and self._solution is None:
            self.fatal('{0} require a solution, but model is not solved with a solution'.format(feature))
        elif self._solved_as_lp():
            pass
        elif not accept_qxp and self.is_quadratic():
            self.fatal('{0} are not available for QP/QCP problems'.format(feature))
        elif self._solved_as_mip():
            self.fatal('{0} are not available for integer problems'.format(feature))

    def _dual_value1(self, linear_ct):
        # PRIVATE
        self._check_problem_type(feature='dual values')
        self._checker.typecheck_ct_added_to_model(self, linear_ct)
        dvs = self._dual_values([linear_ct])
        return dvs[0]

    def dual_values(self, cts):
        """ Returns the dual values of a sequence of linear constraints.

        Note: the model must a pure LP: no integer or binary variable, no piecewise, no SOS.
        The model must also be solved successfully before calling this method.

        :param cts: a sequence of linear constraints.

        :return: a sequence of float numbers
        """
        self._check_problem_type(feature='dual values')
        checked_lcts = self._checker.typecheck_constraint_seq(cts, check_linear=True)
        return self._dual_values(checked_lcts)

    def _dual_values(self, cts):
        # PRIVATE
        checked_lcts = self._checker.typecheck_cts_added_to_model(mdl=self, cts=cts)
        sol = self.solution
        sol.ensure_dual_values(self, self.get_engine())
        return sol.get_dual_values(checked_lcts)

    def _slack_value1(self, ct):
        # private
        self._checker.typecheck_ct_added_to_model(mdl=self, ct=ct)
        self._check_has_solution()
        return self._slack_values([ct])[0]

    def slack_values(self, cts):
        """ Return the slack values for a sequence of constraints.

        Slack values are available for linear, quadratic and indicator constraints.
        The model must be solved successfully before calling this method.

        :param cts: a sequence of constraints.
        :return: a list of float values, in the same order as the constraints.
        """
        self._check_has_solution()
        ckr = self._checker
        cts1 = ckr.typecheck_constraint_seq(cts)
        cts2 = ckr.typecheck_cts_added_to_model(self, cts1)
        return self._slack_values(cts2)

    def _slack_values(self, cts):
        checked_cts = self._checker.typecheck_constraint_seq(cts)
        # ---
        sol = self.solution
        sol.ensure_slack_values(self, self.get_engine())
        return sol.get_slacks(checked_cts)

    def _reduced_cost1(self, dvar):
        # PRIVATE
        self._check_problem_type(feature='reduced costs')
        #self._checker.typecheck_var(dvar)
        rcs = self._reduced_costs([dvar])
        return rcs[0]

    def reduced_costs(self, dvars):
        """ Returns the reduced costs for a variable iterable.

         Note: the model must a pure LP: no integer or binary variable, no piecewise, no SOS.
         The model must also be solved successfully before calling this method.

        :param dvars: a sequence of variables.
        :return: a list of float numbers, in the same order as the variable sequence.
        """
        self._check_problem_type(feature='reduced costs')
        checked_vars = self._checker.typecheck_var_seq(dvars, caller='Model.reduced_costs')
        return self._reduced_costs(checked_vars)

    def _reduced_costs(self, dvars):
        sol = self.solution
        assert sol is not None
        sol.ensure_reduced_costs(model=self, engine= self.get_engine())
        return sol.get_reduced_costs(dvars)

    def quadratic_dual_slacks(self, *args):
        """ Returns quadratic dual slacks as a dict of dicts.

        Can be called in two forms: either with no arguments, in which case it returns
        quadratic dual slacks for all quadratic constraints in the model, or with
        a list of quadratic constraints. In this case it returns only quadratic dual slacks for those constraints

        :param args: accepts either no arguments,or a list of quadratic constraints.

        :return: a Python dictionary, whose keys are quadratic constraints, and
            values are dictionaries from variables to quadratic dual slacks.

        *New in version 2.15*
        """
        nb_args = len(args)
        if 0 == nb_args:
            qcts = self.iter_quadratic_constraints()
        elif 1 == nb_args:
            qcts = args[0]
        else:
            qcts = None
            self.fatal("Model.quadratic_dual_slacks expects either an iteratble on quadratic constraints, or no args.")

        self._check_problem_type('quadratic_dual_slacks', requires_solution=True, accept_qxp=True)
        cpx = self._get_cplex(do_raise=True, msgfn=lambda: "Quadratic dual slacks require CPLEX library")
        checked_qcts = self._checker.typecheck_quadratic_constraint_seq(qcts)
        if checked_qcts:
            qixs = [qc.index for qc in checked_qcts]
            cpx_qdss = cpx.solution.get_quadratic_dualslack(qixs)
            qds_as_dict = {checked_qcts[q]: {self._var_by_index(idx): qds for idx, qds in zip(cpx_sp.ind, cpx_sp.val)} \
                        for q, cpx_sp in enumerate(cpx_qdss) }
            return qds_as_dict
        else:
            return {}

    def _var_basis_status1(self, dvar):
        # internal
        return self.var_basis_statuses([dvar])[0]

    def var_basis_statuses(self, dvars):
        """
        Returns basis status for a batch of variables.

        :param dvars: an iterable returning variables.
        :return: a list of basis status, of type :class:`docplex.mp.constants.BasisStatus`.
            The order of the list is the order in which variables were returned by the iterable.

        *New in version 2.10*
        """
        self._check_problem_type(feature='basis status', requires_solution=False, accept_qxp=False)
        checked_vars = self._checker.typecheck_var_seq(dvars)
        return self._var_basis_status(checked_vars)

    def _var_basis_status(self, dvars):
        return self._generic_get_basis_status(dvars, pos=0,
                                               sol_getter=lambda s_, dvs_: s_.get_var_basis_statuses(dvs_))

    def linear_constraint_basis_statuses(self, lcts):
        """
        Returns basis status for a batch of linear constraints.

        :param lcts: an iterable returning linear constraints.
        :return: a list of basis status, of type :class:`docplex.mp.constants.BasisStatus`.
            The order of the list is the order in which constraints were returned by the iterable.

        *New in version 2.10*
        """
        self._check_problem_type(feature='basis status', requires_solution=False, accept_qxp=False)
        checked_lincts = self._checker.typecheck_constraint_seq(lcts, check_linear=True, accept_range=True)
        return self._linearct_basis_status(checked_lincts)

    def _linearct_basis_status(self, lcts):
        return self._generic_get_basis_status\
            (lcts, pos=1, sol_getter=lambda s_, cts_: s_.get_linearct_basis_statuses(cts_))

    def _generic_get_basis_status(self, objs, pos, sol_getter):
        assert pos in {0, 1}
        sol = self.solution
        if sol:
            sol.ensure_basis_statuses(model=self, engine= self.get_engine())
            return sol_getter(sol, objs) #sol.get_linearct_basis_statuses(lcts)
        else:
            basis_tuple = self.__engine.get_basis(self)
            basis = basis_tuple[pos]
            if not len(basis):
                self.error("No basis is available")
                return [BasisStatus.NotABasisStatus] * len(objs)
            else:
                return [ BasisStatus.parse(basis.get(obj, -1)) for obj in objs]

    def has_basis(self):
        """ returns True if the model contains basis information.

        *New in version 2.9*
        """
        sol = self.solution
        if sol:
            sol.ensure_basis_statuses(model=self, engine= self.get_engine())
            return sol._has_basis()
        else:
            var_basis, linct_basis = self.__engine.get_basis(self)
            return len(var_basis) > 0

    def _write_cplex_file(self, name, path, basename, extension, cpx_write_fn, check_fn=lambda m_: 0):
        check_fn(self)
        export_basename = normalize_basename(self.name, force_lowercase=True)
        export_path = make_output_path2(actual_name=export_basename,
                                        extension=extension,
                                        path=path,
                                        basename_fmt=basename)
        if export_path:
            msg = "CPLEX library is required for {0} export - file {1} not written".format(name, export_path)
            cpx = self._get_cplex(do_raise=True, msgfn=lambda: msg)
            cpx_write_fn(cpx, export_path)
            return export_path

    def _check_basis(self):
        if not self.has_basis():
            self.fatal("No basis data is available for model '{0}'- cannot write basis file",
                    self.name)

    def export_basis(self, path=None, basename=None):
        return self._write_cplex_file(name='basis', path=path, basename=basename,
                                      extension='.bas',
                                      cpx_write_fn=lambda cpx_, path_: cpx_.solution.basis.write(path_),
                                      check_fn=lambda m_: m_._check_basis())


    DEFAULT_VAR_VALUE_QUOTED_SOLUTION_FMT = '  \"{varname}\"={value:.{prec}f}'
    DEFAULT_VAR_VALUE_UNQUOTED_SOLUTION_FMT = '  {varname}={value:.{prec}f}'
    DEFAULT_OBJECTIVE_FMT = "{0}: {1:.{prec}f}"

    @classmethod
    def supports_logical_constraints(cls):
        return cls()._supports_logical_constraints()

    def _supports_logical_constraints(self):
        # INTERNAL
        ok, _ = self.__engine.supports_logical_constraints()
        return ok
    
    _is_cplex_ce = None
    
    @classmethod
    def is_cplex_ce(cls):
        if cls._is_cplex_ce is None:
            m = Model()
            if not m.has_cplex():
                _is_cplex_ce = False
            else:
                try:
                    for i in range(1001):
                        v = m.integer_var()
                        m.add_constraint(v <= i)
                    m.solve()
                    cls._is_cplex_ce = False
                except DOcplexLimitsExceeded as e:
                    cls._is_cplex_ce = True
        return cls._is_cplex_ce

    def _check_logical_constraint_support(self):
        ok, why = self.__engine.supports_logical_constraints()
        if not ok:
            assert why
            self.fatal(msg=why)

    @classmethod
    def is_docplex_debug(cls):
        return not not os.environ.get("DOCPLEX_DEBUG")

    def _has_username_with_spaces(self):
        for v in self.iter_variables():
            if v.has_user_name() and ' ' in v.name:
                return True
        return False

    def print_solution(self, print_zeros=False,
                       solution_header_fmt=None,
                       var_value_fmt=None,
                       **kwargs):
        """  Prints the values of the model variables after a solve.

        Only valid after a successful solve. If the model has not been solved successfully, an
        exception is raised.

        Args:
            print_zeros (Boolean): If False, only non-zero values are printed. Default is False.
            solution_header_fmt: a solution header string in format syntax, or None.
                This format will be passed to  :function:`SolveSolution.display`.
            var_value_fmt : A format string to format the variable name and value. Again, the default uses the automatically computed precision.

        See Also:
            :func: `docplex.mp.solution.SolveSolution.display`

        """
        if self._solution is None:
            return
        self._check_has_solution()
        if var_value_fmt is None:
            if self._has_username_with_spaces():
                var_value_fmt = self.DEFAULT_VAR_VALUE_QUOTED_SOLUTION_FMT
            else:
                var_value_fmt = self.DEFAULT_VAR_VALUE_UNQUOTED_SOLUTION_FMT
        if not self.has_objective():
            var_value_fmt = var_value_fmt[2:]
        # scope of variables.
        iter_vars = self.iter_variables() if print_zeros else None
        # if some username has a whitespace, use quoted format
        self.solution.display(print_zeros=print_zeros,
                              header_fmt=solution_header_fmt,
                              value_fmt=var_value_fmt,
                              iter_vars=iter_vars, **kwargs)

    def report(self):
        """  Prints the value of the objective and the KPIs.
        Only valid after a successful solve, otherwise states that the model is not solved.
        """
        if self._has_solution():
            if self.has_multi_objective():
                mobj_values = self._multi_objective_values()
                prec = self._float_precision
                s_mobjs = ", ".join("{0:.{prec}f}".format(mo, prec=prec) for mo in mobj_values)
                print("* model {0} solved with objectives = [{1}]".format(self.name, s_mobjs))
            else:
                used_prec = self._float_precision
                print("* model {0} solved with objective = {1:.{prec}f}".format(self.name,
                                                                                self._objective_value(), prec=used_prec))
            self.report_kpis()
        else:
            self.info("Model {0} has not been solved successfully, no reporting done.".format(self.name))

    def report_kpis(self, solution=None, selected_kpis=None, kpi_format='*  KPI: {1:<{0}} = '):
        """  Prints the values of the KPIs.

        KPIs require a solution to be evaluated. This solution can be passed explicitly as a parameter,
        or the model is assumed to be solved with a valid solution.

        :param solution: an instance of `SolveSolution`. If not passed, the model solution is
            queried. If the model has no solution, an exception is raised.
        :param selected_kpis: an optional iterable returning the KPIs to print.
            The default behavior is to print all kpis.
        :param kpi_format: an optional format to print the KPi name and its value.

        See Also:
            :class: `docplex.mp.solution.SolveSolution`
            :func: `new_solution`
        """
        kpi_num_format = kpi_format + self._float_meta_format % (2,)
        kpi_str_format = kpi_format + '{2!s}'
        printed_kpis = list(selected_kpis if is_iterable(selected_kpis) else self.iter_kpis())
        try:
            max_kpi_name_len = max(len(k.name) for k in printed_kpis) # max() raises ValueError on empty
        except ValueError:
            max_kpi_name_len = 0
        for kpi in printed_kpis:
            kpi_value = kpi.compute(solution)
            if is_number(kpi_value):
                k_format = kpi_num_format
            else:
                k_format = kpi_str_format

            if type(k_format) != type(kpi.name):
                # infamous mix of str and unicode. Should happen only
                # in py2. Let's convert things
                if isinstance(k_format, str):
                    k_format = k_format.decode('utf-8')
                else:
                    k_format = k_format.encode('utf-8')


            output = k_format.format(max_kpi_name_len, kpi.name, kpi_value)
            try:
                print(output)
            except UnicodeEncodeError:
                encoding = sys.stdout.encoding if sys.stdout.encoding else 'ascii'
                print(output.encode(encoding,
                                    errors='backslashreplace'))

    def kpis_as_dict(self, solution=None, kpi_filter=None, objective_key=None, use_names=True):
        """ Returns KPI values in a solution as a dictionary.

        Each KPI has a value in the solution. This method returns a dictionary of KPI values,
        indexed by KPI objects.

        :param solution: an instance of solution, as returned by solve(). If not passed, will
            use the model's solution. If no solution is present, an exception is raised.
        :param kpi_filter: an optional predicate to filter some kpis.
            If provided, accepts a function taking one KPI as argument and
            returning a boolean. By default, all KPIs are returned.
        :param objective_key: an optional string key for th eobjective value. If present, the
            value of the objective is added to the dictionary, with this key. By default, this parameter
            is None and the objective is *not* appended to the dictionary.
        :param use_names: a flag which determines whether keys in the resulting dict
            are KPI objects or kpi names. Default is to use KPI names.

        :return:
            A dictionary mapping KPIs, or KPI names to values.

        See Also:
            :class:`docplex.mp.solution.SolveSolution`
        """
        if kpi_filter is None:
            kpi_filter = lambda _: True

        if use_names:
            kpi_dict = {kpi.name: kpi.compute(solution) for kpi in self.iter_kpis() if kpi_filter(kpi)}
        else:
            kpi_dict = {kpi: kpi.compute(solution) for kpi in self.iter_kpis() if kpi_filter(kpi)}
        if objective_key:
            kpi_dict[objective_key] = solution.objective_value
        return kpi_dict

    def _report_lexicographic_goals(self, goal_name_values, kpi_header_format):  # pragma: no cover
        kpi_format = kpi_header_format + self._float_meta_format % (1,) # be safe even integer KPIs might yield floats
        printed_kpis = goal_name_values if is_iterable(goal_name_values) else self.iter_kpis()
        for goal_name, goal_expr in printed_kpis:
            goal_value = goal_expr.solution_value
            print(kpi_format.format(goal_name, goal_value))

    def iter_kpis(self):
        """ Returns an iterator over all KPIs in the model.

        Returns:
           An iterator object.
        """
        return iter(self._allkpis)

    def kpi_by_name(self, name, try_match=True, match_case=False, do_raise=True):
        """ Fetches a KPI from a string.

        This method fetches a KPI from a string, using either exact naming or trying
        to match a substring of the KPI name.

        Args:
            name (string): The string to be matched.
            try_match (Boolean): If True, returns KPI whose name is not equal to the
                argument, but contains it. Default is True.
            match_case: If True, looks for a case-exact match, else ignores case. Default is False.
            do_raise: If True, raise an exception when no KPI is found.

        Example:
            If the KPI name is "Total CO2 Cost" then fetching with argument `co2` and `match_case` to False
            will succeed. If `match_case` is True, then no KPI will be returned.

        Returns:
            The KPI expression if found. If the search fails, either raises an exception or returns a dummy
            constant expression with 0.
        """
        for kpi in iter(reversed(self._allkpis)):
            kpi_name = kpi.name
            ok = False
            if kpi_name == name:
                ok = True
            elif try_match:
                if match_case:
                    ok = kpi_name.find(name) >= 0
                else:
                    ok = kpi_name.lower().find(name.lower()) >= 0
            if ok:
                return kpi
        if do_raise:
            self.fatal('Cannot find any KPI matching: "{0:s}"', name)
        else:
            return self._lfactory.new_zero_expr()

    def kpi_value_by_name(self, name, solution=None, try_match=True, match_case=False, do_raise=True):
        """ Returns a KPI value from a KPI name.

        This method fetches a KPI value from a string, using either exact naming or trying
        to match a substring of the KPI name.

        Args:
            name (str): The string to be matched.
            solution: an optional solution. If not present, assume the model is solved
                and use the model solution.
            try_match (Bool): If True, returns KPI whose name is not equal to the
                argument, but contains it. Default is True.
            match_case: If True, looks for a case-exact match, else ignores case. Default is False.
            do_raise: If True, raise an exception when no matching KPI is found.

        Example:
            If the KPI name is "Total CO2 Cost" then fetching with argument `co2` and `match_case` to False
            will succeed. If `match_case` is True, then no KPI will be returned.

        Note:
            KPIs require a solution to be evaluated. This solution can be passed explicitly as a parameter,
            or the model is assumed to be solved with a valid solution.

        Returns:
            float: The KPI value.

        See Also:
            :class: `docplex.mp.solution.SolveSolution`
            :func: `new_solution`
        """
        kpi = self.kpi_by_name(name, try_match, match_case=match_case, do_raise=do_raise)
        return kpi.compute(solution)

    def add_kpi(self, kpi_arg, publish_name=None):
        """ Adds a Key Performance Indicator to the model.

        Key Performance Indicators (KPIs) are objects that can be evaluated after a solve().
        Typical use is with decision expressions, the evaluation of which return the expression's solution value.

        KPI values are displayed with the method :func:`report_kpis`.

        Args:
            kpi_arg:  Accepted arguments are either an expression,
                        a lambda function with two arguments (model + solution)
                        or an instance of a subclass of abstract class KPI.

            publish_name (string, optional): The published name of the KPI.

        Note:
            - If no publish_name is provided, DOcplex will try to access a
              'name' attribute of the argument; if none exists, it will use the
              string representation of the argument , as returned by `str()`.
            - expression KPIs are seperate from the model. In other terms,
              adding KPIs does not change the model (and matrix) being solved.

        Examples:
            `model.add_kpi(x+y+z, "Total Profit")` adds the expression `(x+y+z)` as a KPI with the name "Total Profit".

            `model.add_kpi(x+y+z)` adds the expression `(x+y+z)` as a KPI with
            the name "x+y+z", assuming variables x,y,z have names 'x', 'y', 'z' (resp.)

        Returns:
            The newly added KPI instance.

        See Also:
            :class:`docplex.mp.kpi.KPI`,
            :class:`docplex.mp.kpi.DecisionKPI`
        """
        self._checker.typecheck_string(publish_name, accept_empty=False, accept_none=True, caller="Model.add_kpi(): ")
        new_kpi = self._lfactory.new_kpi(kpi_arg, publish_name)
        new_kpi_name = new_kpi.get_name()
        if new_kpi_name in set(kp.name for kp  in self._allkpis):
            self.fatal("Duplicate KPI name: \"{0!s}\" ", new_kpi_name)
        self._allkpis.append(new_kpi)
        return new_kpi

    def remove_kpi(self, kpi_arg):
        """ Removes a Key Performance Indicator from the model.

        Args:
            kpi_arg:  A KPI instance that was previously added to the model. Accepts either a KPI object or a string.
                If passed a string, looks for a KPI with that name.

        See Also:
            :func:`add_kpi`
            :class:`docplex.mp.kpi.KPI`,
            :class:`docplex.mp.kpi.DecisionKPI`
        """
        if is_string(kpi_arg):
            kpi = self.kpi_by_name(kpi_arg)
            if kpi:
                self._allkpis.remove(kpi)
                kpi.notify_removed()
        else:
            for k, kp in enumerate(self._allkpis):
                if kp is kpi_arg:
                    kx = k
                    break
            else:
                kx = -1
            if kx >= 0:
                removed_kpi = self._allkpis.pop(kx)
                removed_kpi.notify_removed()

            else:
                self.warning('Model.remove_kpi(): cannot interpret this either as a string or as a KPI: {0!r} - ignored', kpi_arg)

    def clear_kpis(self):
        ''' Clears all KPIs defined in the model.

        
        '''
        self._allkpis = []

    @property
    def number_of_kpis(self):
        return len(self._allkpis)


    def add_progress_listener(self, listener):
        """ Adds a progress listener to the model.

        A progress listener is a subclass of :class:~docplex.mp.ProgressListener:

        :param listener:
        """
        self._checker.typecheck_progress_listener(listener)
        self._add_progress_listener(listener)

    def _add_progress_listener(self, listener):
        # INTERNAL
        self._progress_listeners.append(listener)

    def remove_progress_listener(self, listener):
        """ Remove a progress listener from the model.

        :param listener:
        """
        self._progress_listeners.remove(listener)

    def iter_progress_listeners(self):
        """ Returns an iterator on the progress listeners attached to the model.

        :return: an iterator.

        """
        return iter(self._progress_listeners)

    @property
    def number_of_progress_listeners(self):
        """ Returns the number of progress listeners attached to the model.

        :return: an integer
        """
        return len(self._progress_listeners)

    def clear_progress_listeners(self):
        """ Remove all progress listeners from the model."""
        self._progress_listeners = []

    def _fire_start_solve_listeners(self):
        for l in self._progress_listeners:
            l.notify_start()

    def _fire_end_solve_listeners(self, has_solution, objective_value):
        for l in self._progress_listeners:
            l.notify_end(has_solution, objective_value)

    def fire_jobid(self, jobid):  # pragma: no cover
        # INTERNAL
        for l in self._progress_listeners:
            l.notify_jobid(jobid)

    def fire_progress(self, progress_data):  # pragma: no cover
        for l in self._progress_listeners:
            l.notify_progress(progress_data)



    def prettyprint(self, out=None):
        from docplex.mp.ppretty import ModelPrettyPrinter
        ppr = ModelPrettyPrinter()
        ppr.printModel(self, out=out)

    pprint = prettyprint

    def pprint_as_string(self):
        from docplex.mp.ppretty import ModelPrettyPrinter
        with StringIO() as oss:
            ppr = ModelPrettyPrinter()
            ppr.printModel(self, out=oss)
            return oss.getvalue()

    def clone(self, new_name=None, **clone_kwargs):
        """ Makes a deep copy of the model, possibly with a new name.
        Decision variables, constraints, and objective are copied.

        Args:
            new_name (string): The new name to use. If None is provided, returns a "Copy of xxx" where xxx is the original model name.

        :returns: A new model.

        :rtype: :class:`docplex.mp.model.Model`
        """
        return self.copy(new_name=new_name, **clone_kwargs)

    def copy(self, new_name=None, removed_cts=None, **new_kwargs):
        # INTERNAL
        actual_copy_name = new_name or "Copy of %s" % self.name
        # copy kwargs
        copy_kwargs = self._get_kwargs().copy()
        copy_kwargs.update(**new_kwargs)

        copy_context = self.context.copy()
        # pass copy of initial context
        # plus override kwargs (e.g. log_output)
        copy_model = Model(name=actual_copy_name, context=copy_context, **copy_kwargs)

        # clone variable containers
        ctn_map = {}
        for ctn in self.iter_var_containers():
            copied_ctn = ctn.copy(copy_model)
            ctn_map[ctn] = copied_ctn

        # clone variables
        memo = {}
        generated_vars = []

        # clone PWL functions and add them to var_mapping
        for pwl_func in self.iter_pwl_functions():
            copied_pwl_func = pwl_func.copy(copy_model, memo)
            memo[pwl_func] = copied_pwl_func
        # copy 'primary' variables
        for v in self.iter_variables():
            if v.is_generated():
                generated_vars.append(v)
            else:
                copied_var = copy_model._var(v.vartype, v.lb, v.ub, v.name)
                var_ctn = v.container
                if var_ctn:
                    copied_ctn = ctn_map.get(var_ctn)
                    assert copied_ctn is not None
                    copied_var.container = copied_ctn
                memo[v] = copied_var

        for gv in generated_vars:
            gvoo = gv.origin
            try:
                gvo, gvx = gvoo
            except TypeError:
                gvo = gvoo
                gvx = 0
            assert gvo is not None
            gvk = id(gvo)
            cloned_origin = memo.get(gvk)
            if cloned_origin is None:
                cloned_origin = gvo.copy(copy_model, memo)
                cloned_origin.resolve()
                memo[gvk] = cloned_origin
            cloned_gv = cloned_origin.get_artefact(gvx)
            assert cloned_gv

            memo[gv] = cloned_gv

        # copy constraints
        setof_removed_cts = set(removed_cts) if removed_cts else {}
        linear_cts = []
        for ct in self.iter_constraints():
            if not ct.is_generated() and ct not in setof_removed_cts:
                if isinstance(ct, PwlConstraint):
                    continue
                if ct.is_linear():
                    linear_cts.append(ct.copy(copy_model, memo))
                elif ct.is_logical:
                    if linear_cts:
                        # add stored linear cts
                        copy_model.add_constraints(linear_cts)
                        linear_cts = []
                    copy_model.add(ct.copy(copy_model, memo))

        if linear_cts:
            copy_model.add_constraints(linear_cts)

        # clone objective
        copy_model.set_objective_sense(self.objective_sense)
        if self.has_multi_objective():
            multi_objective = self._multi_objective
            exprs = multi_objective.exprs
            nb_exprs = len(exprs)
            copied_exprs = [expr.copy(copy_model, memo) for expr in exprs]
            copy_model.set_multi_objective(self.objective_sense,
                                           exprs=copied_exprs,
                                           priorities=multi_objective.priorities,
                                           weights=multi_objective.weights,
                                           abstols=MultiObjective.as_optional_sequence(multi_objective.abstols,
                                                                                       nb_exprs),
                                           reltols=MultiObjective.as_optional_sequence(multi_objective.reltols,
                                                                                       nb_exprs),
                                           names=multi_objective.names)

        else:
            copy_model.set_objective(self.objective_sense, self.objective_expr.copy(copy_model, memo))

        # clone kpis
        for kpi in self.iter_kpis():
            copy_model.add_kpi(kpi.copy(copy_model, memo))

        # clone sos
        for sos in self.iter_sos():
            if not sos.is_generated():
                copy_model._create_engine_sos(sos.copy(copy_model, memo))

        # parameters
        for p in self.parameters.iter_params():
            if p.is_nondefault():
                # copy value to new parames
                newp = copy_model.get_parameter_from_id (p.cpx_id)
                if newp:
                    newp.set(p.value)

        return copy_model

    def _sync_constraint_indices(self, cscope):
        self.__engine.check_constraint_indices(cscope.iter, cscope.cplex_scope)

    def _sync_var_indices(self):
        self.__engine.check_var_indices(self.iter_variables())

    def end(self):
        """ Terminates a model instance.

        Since this method destroys the objects associated with the model, you must not use the model
        after you call this member function.

        """
        self._clear_internal(terminate=True)

    @property
    def parameters(self):
        """ This property returns the root parameter group of the model.

        The root parameter group models the parameter hierarchy.
        It is the way to access any CPLEX parameter and get or set its value.

        Examples:

            .. code-block:: python

               model.parameters.mip.tolerances.mipgap

            Returns the parameter itself, an instance of the `Parameter` class.

            To get the value of the parameter, use the `get()` method, as in:

            .. code-block:: python

               model.parameters.mip.tolerances.mipgap.get()
               >>> 0.0001

            To change the value of the parameter, use a standard Python assignment:

            .. code-block:: python

               model.parameters.mip.tolerances.mipgap = 0.05
               model.parameters.mip.tolerances.mipgap.get()
              >>> 0.05

            Assignment is equivalent to the `set()` method:

            .. code-block:: python

               model.parameters.mip.tolerances.mipgap.set(0.02)
               model.parameters.mip.tolerances.mipgap.get()
               >>> 0.02

        Returns:
            The root parameter group, an instance of the `ParameterGroup` class.

        """
        context_params = self.context.cplex_parameters
        if not self._synced_params:
            self._sync_params(context_params)
            self._synced_params = True
        return context_params

    def get_parameter_from_id(self, parameter_cpx_id):
        """ Finds a parameter from a CPLEX id code.

        Args:
            parameter_cpx_id: A CPLEX parameter id (positive integer, for example, 2009 is mipgap).

        :returns: An instance of :class:`docplex.mp.params.parameters.Parameter` if found, else None.
        """
        assert parameter_cpx_id >= 0
        for p in self.parameters.generate_params():
            if p.cpx_id == parameter_cpx_id:
                return p
        return None

    def get_engine_parameter_value(self, param):
        return self.__engine.get_parameter(param)

    def apply_parameters(self):
        self._apply_parameters_to_engine(self.parameters)

    def apply_one_parameter(self, param):
        # internal
        self.__engine.set_parameter(param, param.value)

    def _apply_parameters_to_engine(self, parameters_to_use):
        # internal
        if parameters_to_use is not None:
            self_engine = self.__engine
            for param in parameters_to_use:
                self_engine.set_parameter(param, param.value)

    def _get_cplex_engine(self, caller):
        # INTERNAL
        self_engine = self.get_engine()
        if self_engine.name != 'cplex_local':
            self.fatal("{1} is only for Cplex, engine is '{0}'", self_engine.name, str(caller))
        else:
            return self_engine

    def set_hidden_parameter(self, parameter_id, param_value):
        self_engine = self._get_cplex_engine(caller="Model.set_hidden_parameter")
        self_engine.set_parameter_from_id(parameter_id, param_value)

    def get_hidden_parameter(self, parameter_id):
        self_engine = self._get_cplex_engine(caller="Model.get_hidden_parameter")
        return self_engine.get_parameter_from_id(parameter_id)


    # with protocol
    def __enter__(self):
        return self

    def __exit__(self, atype, avalue, atraceback):
        # terminate the model upon exiting a 'with' block.
        self.end()

    def __iadd__(self, e):
        # implements the "+=" dialect a la PulP
        self.add(e)
        return self

    def _resync(self):
        # INTERNAL
        self._lfactory.resync_whole_model()

    def resync_engine(self):
        # INTERNAL: resync after pickle
        self.__engine.resync()

    def add_sos1(self, dvars, name=None):
        ''' Adds  an SOS of type 1 to the model.

        Args:
            dvars: The variables in the special ordered set.
                This method only accepts ordered sequences of variables or iterators.
                Unordered iterables (e.g. dictionaries or sets) are not accepted.

            name: An optional name.

        Returns:
            The newly added SOS.
        '''
        return self.add_sos(dvars, sos_arg=SOSType.SOS1, name=name)

    def add_sos2(self, dvars, name=None):
        ''' Adds  an SOS of type 2 to the model.

        Args:
           dvars: The variables in the specially ordered set.
                This method only accepts ordered sequences of variables or iterators.
                Unordered iterables (e.g. dictionaries or sets) are not accepted.
           name: An optional name.

        Returns:
            The newly added SOS.
        '''
        return self.add_sos(dvars, sos_arg=SOSType.SOS2, name=name)

    def add_sos(self, dvars, sos_arg, weights=None, name=None):
        ''' Adds  an SOS to the model.

        Args:
           sos_arg: The SOS type. Valid values are numerical (1 and 2) or enumerated (`SOSType.SOS1` and
              `SOSType.SOS2`).
           dvars: The variables in the special ordered set.
                This method only accepts ordered sequences of variables or iterators,
                e.g. lists, numpy arrays, pandas Series.
                Unordered iterables (e.g. dictionaries or sets) are not accepted.
           weights: optional weights. Accepts None (no weights) or a list of numbers, with the same size as
                number of variables.
           name: An optional name.

        Returns:
            The newly added SOS.
        '''
        sos_type = SOSType.parse(sos_arg)
        msg = 'Model.add_%s() expects an ordered sequence (or iterator) of variables' % sos_type.lower()
        self._checker.check_ordered_sequence(arg=dvars, caller=msg)
        var_seq = self._checker.typecheck_var_seq(dvars, caller="Model.add_sos")

        var_list = list(var_seq)  # we need len here.
        nb_vars = len(var_list)
        if nb_vars < sos_type.size:
            self.fatal("A {0:s} variable set must contain at least {1:d} variables, got: {2:d}",
                       sos_type.name, sos_type.size, nb_vars)
        elif nb_vars == sos_type.size:
            self.warning("{0:s} variable is trivial, conrtains {1} variable(s): all variables set to 1",
                       sos_type.name, sos_type.size)
        lweights = StaticTypeChecker.typecheck_optional_num_seq(self, weights, accept_none=True, expected_size=nb_vars,
                                                                caller='Model.add_sos')
        return self._add_sos(dvars, sos_type, weights=lweights, name=name)

    def _add_sos(self, dvars, sos_type, weights=None, name=None):
        # INTERNAL
        new_sos = self._lfactory.new_sos(dvars, sos_type=sos_type, weights=weights, name=name)
        sos_index = self.__engine.create_sos(new_sos)
        self._register_sos(new_sos)
        new_sos.index = sos_index
        return new_sos

    def _create_engine_sos(self, new_sos):
        # internal
        sos_index = self.__engine.create_sos(new_sos)
        self._register_sos(new_sos)
        new_sos.index = sos_index

    def _register_sos(self, new_sos):
        self._allsos.append(new_sos)

    def _get_sos_by_index(self, sos_idx):
        # INTERNAL
        try:
            return self._allsos[sos_idx]
        except IndexError:
            return None

    def iter_sos(self):
        ''' Iterates over all SOS sets in the model.

        Returns:
            An iterator object.
        '''
        return iter(self._allsos)

    @property
    def number_of_sos(self):
        ''' This property returns the total number of SOS sets in the model.

        '''
        return len(self._allsos)

    def clear_sos(self):
        ''' Clears all SOS sets in the model.
        '''
        self._allsos = []
        self.__engine.clear_all_sos()

    def _generate_sos(self, sos_type):
        # INTERNAL
        for sos_set in self.iter_sos():
            if sos_set.sos_type == sos_type:
                yield sos_set

    def iter_sos1(self):
        ''' Iterates over all SOS1 sets in the model.

        Returns:
            An iterator object.
        '''
        return self._generate_sos(SOSType.SOS1)

    def iter_sos2(self):
        ''' Iterates over all SOS2 sets in the model.

        Returns:
            An iterator object.
        '''
        return self._generate_sos(SOSType.SOS2)

    @property
    def number_of_sos1(self):
        ''' This property returns the total number of SOS1 sets in the model.

        '''
        return sum(1 for _ in self.iter_sos1())

    @property
    def number_of_sos2(self):
        ''' This property returns the total number of SOS2 sets in the model.

        '''
        return sum(1 for _ in self.iter_sos2())

    def piecewise(self, preslope, breaksxy, postslope, name=None):
        """  Adds a piecewise linear function (PWL) to the model, using breakpoints to specify the function.

        Args:
            preslope: Before the first segment of the PWL function there is a half-line; its slope is specified by
                        this argument.
            breaksxy: A list `(x[i], y[i])` of coordinate pairs defining segments of the PWL function.
            postslope: After the last segment of the the PWL function there is a half-line; its slope is specified by
                        this argument.
            name: An optional name.

        Example::

            # Creates a piecewise linear function whose value if '0' if the `x_value` is `0`, with a slope
            # of -1 for negative values and +1 for positive value
            model = Model('my model')
            model.piecewise(-1, [(0, 0)], 1)

            # Note that a PWL function may be discontinuous. Here is an example of a step function:
            model.piecewise(0, [(0, 0), (0, 1)], 0)

        Returns:
            The newly added piecewise linear function.
        """

        if breaksxy is None:
            self._checker.fatal("argument 'breaksxy' must be defined")

        StaticTypeChecker.typecheck_num_nan_inf(self, preslope, caller='Model.piecewise.preslope')
        StaticTypeChecker.typecheck_num_nan_inf(self, postslope, caller='Model.piecewise.postslope')
        PwlFunction.check_list_pair_breaksxy(self._checker, breaksxy)
        return self._piecewise(PwlFunction._PwlAsBreaks(preslope, breaksxy, postslope), name)

    def piecewise_as_slopes(self, slopebreaksx, lastslope, anchor=(0, 0), name=None):
        """  Adds a piecewise linear function (PWL) to the model, using a list of slopes and x-coordinates.

        Args:
            slopebreaksx: A list of tuple pairs `(slope[i], breakx[i])` of slopes and x-coordinates defining the slope of
                        the piecewise function between the previous breakpoint (or minus infinity if there is none)
                        and the breakpoint with x-coordinate `breakx[i]`.
                        For representing a discontinuity, two consecutive pairs with the same value for `breakx[i]`
                        are used. The value of `slope[i]` in the second pair is the discontinuity gap.
            lastslope: The slope after the last specified breakpoint.
            anchor: The coordinates of the 'anchor point'. The purpose of the anchor point is to ground the piecewise
                        linear function specified by the list of slopes and breakpoints.
            name: An optional name.
        Example::

            # Creates a piecewise linear function whose value if '0' if the `x_value` is `0`, with a slope
            # of -1 for negative values and +1 for positive value
            model = Model('my model')
            model.piecewise_as_slopes([(-1, 0)], 1, (0, 0))

            # Here is the definition of a step function to illustrate the case of a discontinuous PWL function:
            model.piecewise_as_slopes([(0, 0), (0, 1)], 0, (0, 0))

        Returns:
            The newly added piecewise linear function.
        """
        StaticTypeChecker.typecheck_num_nan_inf(self, lastslope, caller="Model.piecewise_as_slopes.lastslope")
        StaticTypeChecker.check_number_pair(self, anchor, caller="Model.piecewise_as_slopes.anchor")
        PwlFunction.check_list_pair_slope_breakx(self, slopebreaksx, anchor)
        return self._piecewise(PwlFunction._PwlAsSlopes(slopebreaksx, lastslope, anchor), name)

    def add_piecewise_constraint(self, y, pwlf, x, name=None):
        checker = self._checker
        checker.typecheck_continuous_var(x)
        checker.typecheck_continuous_var(y)
        checker.typecheck_pwl_function(pwlf)
        if x is y:
            self.fatal('Piecewise-linear constraint requires two different variables, only one wa passed: {0}', x)

        pwl_expr = self._add_pwl_expr(pwlf, arg=x, yvar=y, resolve=False) # will be resolved later
        pwl_ct = self._lfactory.new_pwl_constraint(pwl_expr, name)
        return self._add_pwl_constraint_internal(pwl_ct)

    def _piecewise(self, pwl_def, name=None):
        pwl_func = self._lfactory.new_piecewise(pwl_def, name)
        self.__allpwlfuncs.append(pwl_func)
        return pwl_func

    def _add_pwl_expr(self, pwl_func, arg, yvar=None, resolve=True):
        pwl_func_usage_counter = self._pwl_counter.get(pwl_func, 0) + 1
        pwl_expr = self._lfactory.new_pwl_expr(pwl_func, arg, pwl_func_usage_counter, y_var=yvar, resolve=resolve)
        return pwl_expr

    def _add_pwl_constraint_internal(self, pwlct):
        ct_engine_index = self.__engine.create_pwl_constraint(pwlct)
        self._register_one_pwl_constraint(pwlct, ct_engine_index)
        return pwlct

    def _register_one_pwl_constraint(self, new_pwl_ct, ct_index):
        self.__notify_new_model_object(
            "pwl", new_pwl_ct, ct_index, mobj_name=None, name_dir=None, idx_scope=self._pwl_scope, is_name_safe=True)

        # Maintain the number of constraints associated to each piecewise function definition.
        # This counter is used when naming PWL constraints.
        self._pwl_counter[new_pwl_ct.pwl_func] = new_pwl_ct.usage_counter


    def iter_pwl_constraints(self):
        """ Iterates over all PWL constraints in the model.

        Returns:
            An iterator object.
        """
        return self._pwl_scope.iter_objects()

    @property
    def number_of_pwl_constraints(self):
        """ This property returns the total number of PWL constraints in the model.
        """
        return self._pwl_scope.size

    def _ensure_benders_annotations(self):
        if self._benders_annotations is None:
            self._benders_annotations = {}
        return self._benders_annotations

    def set_benders_annotation(self, obj, group):
        if group is None:
            self_benders = self._benders_annotations
            if self_benders is not None and obj in self_benders:
                del self_benders[obj]
        else:
            self._checker.typecheck_int(group, accept_negative=False, caller='Model.set_benders_annotation')
            self._ensure_benders_annotations()[obj] = group

    def remove_benders_annotation(self, obj):
        self_benders = self._benders_annotations
        if self_benders:
            del self_benders[obj]

    def get_benders_annotation(self, obj):
        self_benders = self._benders_annotations
        return self_benders.get(obj) if self_benders is not None else None

    def iter_benders_annotations(self):
        self_benders = self._benders_annotations
        return six.iteritems(self_benders) if self_benders is not None else iter([])

    def clear_benders_annotations(self):
        self._benders_annotations = None

    def get_annotations_by_scope(self):
        # INTERNAL
        from collections import defaultdict
        annotated_by_scope = defaultdict(list)
        for obj, group in self.iter_benders_annotations():
            annotated_by_scope[obj.cplex_scope].append((obj, group))
        return annotated_by_scope

    def has_benders_annotations(self):
        self_benders = self._benders_annotations
        return bool(self_benders)

    def get_annotation_stats(self):
        from collections import Counter
        annotated_by_scope = Counter()
        for obj, group in self.iter_benders_annotations():
            annotated_by_scope[obj.cplex_scope] += 1
        return annotated_by_scope

    def register_callback(self, cb_type):
        # Registers a callback with the model.
        #
        # Assumes the type has a `model` setter property. Use a subclass of `ModelCallbackMixin` mixin class
        # as a parent class to ensure this.
        #
        # :param cb_type: a callback type; the type must be a subtype of some cplex callback type
        #
        # :return: an instance of the callback
        #
        # See Also:
        #     :class: `docplex.mp.callbacks.ModelcallbackMixin`
        cplex_cb = self.__engine.register_callback(cb_type)
        if cplex_cb:
            cplex_cb._model = self
        return cplex_cb

    def _resolve_pwls(self):
        # INTERNAL
        self._objective_expr.resolve()
        no_pwl_scopes = [self._linct_scope, self._logical_scope, self._quadct_scope]
        for sc in no_pwl_scopes:
            for x in sc.iter_objects():
                x.resolve()
        # this call updates the dict so we must iterate on somethibg else.
        pwls = [pw for pw in self._pwl_scope.iter_objects()]
        for pw in pwls:
            pw.resolve()




    def get_constraint_priority(self, ct):
        # INTERNAL
        return self._constraint_priority_dict.get(ct)  # return None if not found

    def set_constraint_priority(self, ct, prio):
        # INTERNAL
        self._constraint_priority_dict[ct] = prio

    def _extend_constraint_section(self, collector, extra_cts, ctnames, caller):
        # INTERNAL
        checker = self._checker
        new_cts = checker.typecheck_constraint_seq(extra_cts, check_linear=True, accept_range=False)

        if ctnames is not None:
            checker.typecheck_iterable(ctnames)
            for ct, ctn in izip2_filled(new_cts, ctnames):
                checker.typecheck_string(ctn, accept_none=True)
                if ctn:
                    self._register_ct_name(ct, ctn, checker)

        # extend
        lncts = list(new_cts)
        for nc in lncts:
            checker.typecheck_ct_not_added(nc, do_raise=False, caller=caller)

        collector.extend(new_cts)
        return new_cts

    def add_lazy_constraints(self, lazy_cts, names=None):
        """Adds lazy constraints to the problem.

        This method expects an iterable returning linear constraints (ranges are not accepted).

        :param lazy_cts: an iterable returning linear constraints (not ranges)
        :param names: an optional iterable returning strings, used to set names for lazy constraints.

        *New in version 2.10*
        """
        new_lazy_cts = self._extend_constraint_section(self._lazy_constraints, lazy_cts, names, caller='Model.add_lazy_constraints')
        for nc in new_lazy_cts:
            nc.notify_used_as_lazy_constraint()
        self.__engine.add_lazy_constraints(new_lazy_cts)

    def add_lazy_constraint(self, lazy_ct, name=None):
        """Adds one lazy constraint to the problem.

        This method expects a linear constraint.

        :param lazy_ct: a linear constraints (ranges are not accepted)
        :param name: an optional string, used to set the name of the lazy constraint.

        *New in version 2.10*

        """
        self.add_lazy_constraints((lazy_ct,), names=(name,))

    def clear_lazy_constraints(self):
        """
        Clears all lazy constraints from the model.

        *New in version 2.10*
        """
        old_lazy_cts = self._lazy_constraints
        for lz in old_lazy_cts:
            lz.notify_unused_as_lazy_constraint()
        self._lazy_constraints = []
        self.__engine.clear_lazy_constraints()

    def iter_lazy_constraints(self):
        """ Returns an iterator on the model's lazy constraints

        :return: an iterator on lazy constraints.

        *New in version 2.10*
        """
        return iter(self._lazy_constraints)

    @property
    def number_of_lazy_constraints(self):
        """Returns the number of lazy constraints present in the model
        """
        return len(self._lazy_constraints)

    def _is_lazy_constraint(self, lineart_ct):
        # INTERNAL
        return any(lc is lineart_ct for lc in self.iter_lazy_constraints())

    def add_user_cut_constraints(self, cut_cts, names=None):
        """Adds user cut constraints to the problem.

        This method expects an iterable returning linear constraints (ranges are not accepted).

        :param cut_cts: an iterable returning linear constraints (not ranges)
        :param names: an optional iterable returning strings, used to set names for user cut constraints.

        *New in version 2.10*
        """
        new_user_cuts = self._extend_constraint_section(self._user_cuts, cut_cts, names, caller='Model.add_user_cut_constraints')
        for nc in new_user_cuts:
            nc.notify_used_as_user_cut()
        self.__engine.add_user_cuts(new_user_cuts)

    def add_user_cut_constraint(self, cut_ct, name=None):
        """Adds one user cut constraint to the problem.

        This method expects a linear constraint.

        :param cut_ct: a linear constraints (ranges are not accepted)
        :param name: an optional string, used to set the name for the cut constraint.

        *New in version 2.10*
        """
        self.add_user_cut_constraints((cut_ct,), names=(name,))

    def clear_user_cut_constraints(self):
        """
        Clears all user cut constraints from the model.

        *New in version 2.10*
        """
        old_user_cuts = self._user_cuts
        for uc in old_user_cuts:
            uc.notify_unused_as_user_cut()

        self._user_cuts = []
        self.__engine.clear_user_cuts()

    def iter_user_cut_constraints(self):
        """ Returns an iterator on the model's user cut constraints

        :return: an iterator on user cut constraints.

        *New in version 2.10*
        """
        return iter(self._user_cuts)

    @property
    def number_of_user_cut_constraints(self):
        """Returns the number of user cut constraints present in the model

        *New in version 2.10*

        """
        return len(self._user_cuts)

    def _is_user_cut_constraint(self, lineart_ct):
        # INTERNAL
        return any(lc is lineart_ct for lc in self.iter_user_cut_constraints())
