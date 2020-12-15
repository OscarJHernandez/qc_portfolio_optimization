# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import operator
from enum import Enum

from docplex.mp.error_handler import docplex_fatal
from docplex.mp.utils import is_string, is_int


class VarBoundType(Enum):
    """This enumerated class describes the two types of variable bounds:
        - LB is for lower bound
        - UB is for uupper bound

    This enumerated type is used in conflict refiner.
    """
    LB = 0
    UB = 1


class ComparisonType(Enum):
    """This enumerated class defines the various types of linear constraints:

        - LE for e1 <= e2 constraints

        - EQ for e1 == e2 constraints

        - GE for e1 >= e2 constraints

        where e1 and e2 denote linear expressions.
    """
    LE = 1, '<=', 'L', operator.le
    EQ = 2, '==', 'E', operator.eq
    GE = 3, '>=', 'G', operator.ge

    def __new__(cls, code, operator_symbol, cplex_kw, python_op):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = code
        obj._cplex_code = cplex_kw
        obj._op_symbol = operator_symbol
        obj._pyop = python_op
        return obj

    # NOTE: Never add a static field in an enum class: it would be interpreted as an other enum

    @property
    def short_name(self):
        return self.name

    @property
    def cplex_code(self):
        return self._cplex_code

    @property
    def operator_symbol(self):
        """ Returns a string operator for the constraint.

        Example:
            Returns string "<=" for a e1 <= e2 constraint.

        Returns:
            string: A string describing the logical operator used in the constraint.
        """
        return self._op_symbol

    @property
    def python_operator(self):
        return self._pyop

    @classmethod
    def parse(cls, arg, do_raise=True):
        # INTERNAL
        # noinspection PyTypeChecker
        for op in cls:
            if arg in (op, op.value):
                return op
            elif is_string(arg):
                if arg == op._cplex_code \
                        or arg == str(op.value) \
                        or arg.lower() == op.name.lower():
                    return op
        # not found
        if do_raise:
            docplex_fatal('cannot convert this to a comparison type: {0!r}'.format(arg))
        else:
            return None

    @classmethod
    def cplex_ctsense_to_python_op(cls, cpx_sense):
        return cls.parse(cpx_sense).python_operator

    @classmethod
    def almost_compare(cls, lval, op, rval, eps):
        if op is cls.LE:
            # lval <= rval with eps tolerance means lval-rval <= e
            return lval - rval <= eps
        elif op is cls.GE:
            # lval >= rval with eps tolerance means lval-rval >= -eps
            return lval - rval >= -eps
        elif op is cls.EQ:
            return abs(lval - rval) <= eps
        else:
            raise TypeError

    @classmethod
    def almost_equal(cls, lval, rval, eps):
        return cls.almost_compare(lval, cls.EQ, rval, eps)


class RelaxationMode(Enum):
    """ This enumerated type describes the different strategies for model relaxation:
      - MinSum, OptSum,
      - MinInf, OptInf,
      - MinQuad, OptQuad.

    A relaxation algorithms works in two phases: In the first phase, it finds a
    feasible solution while making minimal changes to the model (according to a metric).
    In the second phase, it
    searches for an optimal solution while keeping the relaxation at the minimal value found in phase 1.

    Enumerated values work in pairs: MinXXX, OptXXX
        - MinXXX values stop at phase 1, they look for a feasible soluion, they do not
            optimize the objective.
        - OptXXX run the two phases, looking for an optimal relaxed solution. They take longer.

    The metric used to evaluate the quality of the relaxation is determined by the XXX part of
    the name. There are three metrics:
      - Inf (MinInf, OptInf) minimizes the number of relaxed constraints.  This metric
        will prefer to relax one constraint, even with a huge slack, instead of two.
      - Sum (MinSum, OptSum): minimizes the sum of relaxations.
      - Quand (MinQuad, OptQuad): minimizes the sum of squares of relaxations.
        This metric is the most expensive in computation time,
        but avoids huge discrepancies between relaxations:
        two constraints with relaxations of 2,2 will have a better quality (2^2 + 2^2 = 8)
        than relaxations of 3,1 (3^2 +1 = 10).
    """

    MinSum, OptSum, MinInf, OptInf, MinQuad, OptQuad = range(6)

    @staticmethod
    def parse(arg):
        # INTERNAL
        # noinspection PyTypeChecker
        for m in RelaxationMode:
            if arg == m or arg == m.value:
                return m
            elif is_string(arg):
                if arg == str(m.value) or arg.lower() == m.name.lower():
                    return m

        docplex_fatal('cannot parse this as a relaxation mode: {0!r}'.format(arg))

    @staticmethod
    def get_no_optimization_mode(mode):
        assert isinstance(mode, RelaxationMode)
        # even values are MinXXX modes
        relax_code = mode.value
        if 0 == relax_code % 2:
            return mode
        else:
            # OptXXX is 2k+1 when MinXXX is 2k
            return RelaxationMode(relax_code - 1)

    def __repr__(self):
        return 'docplex.mp.RelaxationMode.{0}'.format(self.name)


class ConflictStatus(Enum):
    """
    This enumerated class defines the conflict status types.
    """
    Excluded, Possible_member, Possible_member_lower_bound, Possible_member_upper_bound, \
      Member, Member_lower_bound, Member_upper_bound = -1, 0, 1, 2, 3, 4, 5


class SOSType(Enum):
    """This enumerated class defines the SOS types:

        - SOS1 for SOS type 1

        - SOS1 for SOS type 2.
    """
    SOS1, SOS2 = 1, 2

    def lower(self):
        return self.name.lower()

    @staticmethod
    def parse(arg,
              sos1_tokens=frozenset(['1', 'sos1']),
              sos2_tokens=frozenset(['2', 'sos2'])):
        if isinstance(arg, SOSType):
            return arg
        elif 1 == arg:
            return SOSType.SOS1
        elif 2 == arg:
            return SOSType.SOS2
        elif is_string(arg):
            arg_lower = arg.lower()
            if arg_lower in sos1_tokens:
                return SOSType.SOS1
            elif arg_lower in sos2_tokens:
                return SOSType.SOS2

        docplex_fatal("Cannot convert to SOS type: {0!s} - expecting 1|2|'sos1'|'sos2'", arg)

    def _cpx_sos_type(self):
        # INTERNAL
        return str(self.value)

    @property
    def size(self):
        return self.value

    def __repr__(self):
        return 'docplex.mp.SOSType.{0}'.format(self.name)


class SolveAttribute(Enum):
    duals = 1, False, True
    slacks = 2, False, True
    reduced_costs = 3, True, True

    def __new__(cls, code, is_for_vars, requires_solve):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = code
        obj.is_var_attribute = is_for_vars
        obj.requires_solve = requires_solve
        return obj

    @classmethod
    def parse(cls, arg, do_raise=True):
        # INTERNAL
        # noinspection PyTypeChecker
        for m in cls:
            if arg == m or arg == m.value:
                return m
            elif is_string(arg):
                if arg == str(m.value) or arg.lower() == m.name.lower():
                    return m

        if do_raise:
            docplex_fatal('cannot convert this to a solve attribute: {0!r}'.format(arg))
        else:
            return None


class UpdateEvent(Enum):
    # INTERNAL
    NoOp = 0
    #
    # Linear constraint events
    LinearConstraintCoef = 1
    LinearConstraintRhs = 2
    LinearConstraintGlobal = 3  # logical and of Coef + Rhs
    ConstraintSense = 4

    # Range constraint events
    RangeConstraintBounds = 5
    RangeConstraintExpr = 6

    # Expression events
    ExprConstant = 8
    LinExprCoef = 16
    LinExprGlobal = 24
    LinExprPromotedToQuad = 25  # objective is ok, constraint wont support this.

    # Quad
    QuadExprQuadCoef = 32
    QuadExprGlobal = 64

    # Ind
    IndicatorLinearConstraint = 128

    # Quadct
    QuadraticConstraintGlobal = 256

    def __bool__(self):
        return bool(self.value)


class ObjectiveSense(Enum):
    """
    This enumerated class defines the two types of objectives, `Minimize` and `Maximize`.
    """
    Minimize, Maximize = 1, 2

    def is_minimize(self):
        """ Returns True if objective is a minimizing objective.
        """
        return self is ObjectiveSense.Minimize

    def is_maximize(self):
        """ Returns True if objective is a maximizing objective.
        """
        return self is ObjectiveSense.Maximize

    @property
    def cplex_coef(self):
        return 1 if self.is_minimize() else -1

    @property
    def verb(self):
        """ Returns a string describing the objective (in lowercase)

            - 'minimize' for the Minimize objective
            - 'maximize' for the Maximize objective

        """
        return self.name.lower()

    @property
    def short_name(self):
        """ Returns a short (three letters) string describing the objective: min or max

        """
        return self.verb[:3]

    @classmethod
    def from_cplex(cls, cpx_sense):
        if cpx_sense == 1:
            return cls.Minimize
        elif cpx_sense == -1:
            return cls.Maximize
        else:
            raise ValueError("expecting +1 or -1, {0} was passed".format(cpx_sense))

    @staticmethod
    def parse(arg, logger=None, default_sense=None):
        if isinstance(arg, ObjectiveSense):
            return arg

        elif is_string(arg):
            lower_text = arg.lower()
            if lower_text in {"minimize", "min"}:
                return ObjectiveSense.Minimize
            elif lower_text in {"maximize", "max"}:
                return ObjectiveSense.Maximize
            elif default_sense:
                logger.error(
                    "Text is not recognized as objective sense: {0}, expecting \"min\" or \"max\" - using default {1:s}",
                    (arg, default_sense))
                return default_sense
            elif logger:
                logger.fatal("Text is not recognized as objective sense: {0}, expecting ""min"" or ""max", (arg,))
            else:
                docplex_fatal("Text is not recognized as objective sense: {0}, expecting ""min"" or ""max".format(arg))
        elif is_int(arg):
            if arg == 1:
                return ObjectiveSense.Minimize
            elif -1 == arg:
                return ObjectiveSense.Maximize
            else:
                logger.fatal("cannot convert: <{}> to objective sense", (arg,))
        elif arg is None:
            return default_sense
        elif logger:
            logger.fatal("cannot convert: <{}> to objective sense", (arg,))
        else:
            docplex_fatal("cannot convert: <{}> to objective sense".format(arg))


# noinspection PyPep8
class CplexScope(Enum):
    def __new__(cls, code, prefix, descr):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = code
        obj.prefix = prefix
        obj.descr = descr
        return obj

    # INTERNAL
    VAR_SCOPE = 0, 'x', 'variables'
    LINEAR_CT_SCOPE = 1, 'c', 'linear constraints'
    IND_CT_SCOPE = 2, 'ic', 'indicators'
    QUAD_CT_SCOPE = 3, 'qc', 'quadratic constraints'
    PWL_CT_SCOPE = 4, 'pwl', 'piecewise constraints'
    SOS_SCOPE = 5, 'sos', 'SOS'


class QualityMetric(Enum):
    def __new__(cls, code, has_int, cpx_codename):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = code
        obj.has_int = has_int
        obj.codename = cpx_codename
        return obj

    @property
    def cpx_codename(self):
        return 'CPX_' + self.codename

    @property
    def code(self):
        return self._value_

    @property
    def key(self):
        return self.codename.lower()

    @property
    def int_key(self):
        return '%s.int' % self.codename.lower()

    max_primal_infeasibility = 1, 1, 'MAX_PRIMAL_INFEAS'
    max_scaled_primal_infeasibility = 2, 1, 'MAX_SCALED_PRIMAL_INFEAS'
    sum_primal_infeasibilities = 3, 0, 'SUM_PRIMAL_INFEAS'
    sum_scaled_primal_infeasibilities = 4, 0, 'SUM_SCALED_PRIMAL_INFEAS'

    max_dual_infeasibility = 5, 1, 'MAX_DUAL_INFEAS'
    max_scaled_dual_infeasibility = 6, 1, 'MAX_SCALED_DUAL_INFEAS'
    sum_dual_infeasibilities = 7, 0, 'SUM_DUAL_INFEAS'
    sum_scaled_dual_infeasibilities = 8, 0, 'SUM_SCALED_DUAL_INFEAS'

    max_int_infeasibility = 9, 1, 'MAX_INT_INFEAS'
    sum_integer_infeasibilities = 10, 0, 'SUM_INT_INFEAS'

    max_primal_residual = 11, 1, 'MAX_PRIMAL_RESIDUAL'
    max_scaled_primal_residual = 12, 1, 'MAX_SCALED_PRIMAL_RESIDUAL'
    sum_primal_residual = 13, 0, 'SUM_PRIMAL_RESIDUAL'
    sum_scaled_primal_residual = 14, 0, 'SUM_SCALED_PRIMAL_RESIDUAL'

    max_dual_residual = 15, 1, 'MAX_DUAL_RESIDUAL'
    max_scaled_dual_residual = 16, 1, 'MAX_SCALED_DUAL_RESIDUAL'
    sum_dual_residual = 17, 0, 'SUM_DUAL_RESIDUAL'
    sum_scaled_dual_residual = 18, 0, 'SUM_SCALED_DUAL_RESIDUAL'

    max_comp_slack = 19, 1, 'MAX_COMP_SLACK'  # gap here
    sum_comp_slack = 21, 0, 'SUM_COMP_SLACK'  # gap here

    max_x = 23, 1, 'MAX_X'
    max_scaled_x = 24, 1, 'MAX_SCALED_X'
    max_pi = 25, 1, 'MAX_PI'
    max_scaled_pi = 26, 1, 'MAX_SCALED_PI'
    max_slack = 27, 1, 'MAX_SLACK'
    max_scaled_slack = 28, 1, 'MAX_SCALED_SLACK'

    max_reduced_cost = 29, 1, 'MAX_RED_COST'
    max_scaled_reduced_cost = 30, 1, 'MAX_SCALED_RED_COST'

    sum_x = 31, 0, 'SUM_X'
    sum_scaled_x = 32, 0, 'SUM_SCALED_X'
    sum_pi = 33, 0, 'SUM_PI'
    sum_scaled_pi = 34, 0, 'SUM_SCALED_PI'
    sum_slack = 35, 0, 'SUM_SLACK'
    sum_scaled_slack = 36, 0, 'SUM_SCALED_SLACK'

    sum_reduced_cost = 37, 0, 'SUM_RED_COST'
    sum_scaled_reduced_cost = 38, 0, 'SUM_SCALED_RED_COST'

    kappa = 39, 0, 'KAPPA'
    objective_gap = 40, 0, 'OBJ_GAP'
    dual_objective = 41, 0, 'DUAL_OBJ'
    primal_objective = 42, 0, 'PRIMAL_OBJ'

    max_quadratic_primal_residual = 43, 1, 'MAX_QCPRIMAL_RESIDUAL'
    sum_quadratic_primal_residual = 44, 0, 'SUM_QCPRIMAL_RESIDUAL'
    max_quadratic_slack_infeasibility = 45, 1, 'MAX_QCSLACK_INFEAS'
    sum_quadratic_slack_infeasibility = 46, 0, 'SUM_QCSLACK_INFEAS'
    max_quadratic_slack = 47, 1, 'MAX_QCSLACK'
    sum_quadratic_slack = 48, 0, 'SUM_QCSLACK'
    max_indicator_slack_infeasibility = 49, 1, 'MAX_INDSLACK_INFEAS'
    sum_indicator_slack_infeasibility = 50, 0, 'SUM_INDSLACK_INFEAS'

    exact_kappa = 51, 0, 'EXACT_KAPPA'
    kappa_stable = 52, 0, 'KAPPA_STABLE'
    kappa_suspicious = 53, 0, 'KAPPA_SUSPICIOUS'
    kappa_unstable = 54, 0, 'KAPPA_UNSTABLE'
    kappa_illposed = 55, 0, 'KAPPA_ILLPOSED'
    kappa_max = 56, 0, 'KAPPA_MAX'
    kappa_attention = 57, 0, 'KAPPA_ATTENTION'

    @classmethod
    def parse(cls, txt, raise_on_error=True):
        for qm in cls:
            if txt == qm.name:
                return qm
            elif txt == qm.value:
                return qm
            elif txt == qm.cpx_codename:
                return qm

        fmt = '* cannot interpret this as a QualityMetric enum: {0!r}'
        if raise_on_error:
            docplex_fatal(fmt, txt)
        else:
            print(fmt.format(txt))
            return None


class BasisStatus(Enum):
    """ This enumerated type describes the different values for basis status.

    Basis status can be queried for variables and linear constraints in LP problems.

    Possible values are:

    - NotABasisStatus: invalid or unknown status,
    - Basic, means the variable belongs to the base,
    - AtLower, means the variable is non-basic, at its lower bound,
    - AtUpper, means the variable is non-basic, at its upper bound,
    - FreeNonBasic, means the variable is nonbasic and is not at a bound.

    See Also:
        The list of possible values for basis status can be found in the CPLEX documentation:

        https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.cplex.help/refcallablelibrary/cpxapi/getbase.html

    """

    def __new__(cls, code, cpx_codename):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = code
        obj.codename = cpx_codename
        return obj

    NotABasisStatus = -1, "NotBasisStatus"
    AtLowerBound = 0, "CPX_AT_LOWER"
    Basic = 1, "CPX_BASIC"
    AtUpperBound = 2, "CPX_AT_UPPER"
    FreeNonBasic = 3, "CPX_FREE_SUPER"

    @classmethod
    def parse(cls, code):
        for bs in cls:
            if bs.value == code:
                return bs

        return cls.NotABasisStatus


class WriteLevel(Enum):
    """
    This enumerated class controls what is written in MST mip start files.
    The numeric value is identical to the CPLEX WriteLevel parameter values.

    The possible values are (in order of decreasing quantity of information written).

        - AllVars: all variables are written
        - DiscreteVars: all discrete variables are  written (binary, integer, semi-integer)
        - NonZeroVars: all non-zero vars are written, regardless of their type.
        - DiscreteNonZeroVars: all discrete non-zero vars are written.
        - Auto: automatic value, same as DiscreteVars.

    *New in version 2.10*
    """

    def __new__(cls, code, short_name):
        obj = object.__new__(cls)
        # predefined
        obj._value_ = code
        obj.short_name = short_name
        return obj

    Auto = 0, "auto"  # same as DiscreteVars: filter discrete, keep zeros
    AllVars = 1, "all"  # write all variables and their value, zero or nonzero
    DiscreteVars = 2, "discrete"  # write all discrete variables and their value
    NonZeroVars = 3, "nonzero"  # write only nonzero variables
    NonZeroDiscreteVars = 4, "nonzero_discrete"  # write nonzero discrete variables

    def filter_zeros(self):
        return self in {WriteLevel.NonZeroVars, WriteLevel.NonZeroDiscreteVars}

    def filter_discrete(self):
        return self in {WriteLevel.Auto, WriteLevel.DiscreteVars, WriteLevel.NonZeroDiscreteVars}

    @classmethod
    def parse(cls, level):
        if level is None:
            return cls.Auto
        elif isinstance(level, cls):
            return level
        else:
            for wl in cls:
                if wl.value == level:
                    return wl
                elif is_string(level):
                    llevel = level.lower()
                    if llevel == wl.name.lower() or llevel == wl.short_name:
                        return wl
            return cls.Auto


class EffortLevel(Enum):
    """
    This enumerated class controls the effort level used for a MIP start.
    The numeric value is identical to the CPLEX EffortLevel parameter values.

    See Also:
        The list of possible values for effort level status can be found in the CPLEX documentation:

https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.cplex.help/refcppcplex/html/enumerations/IloCplex_MIPStartEffort.html


    """
    Auto = 0
    CheckFeas = 1
    SolveFixed = 2
    SolveMIP = 3
    Repair = 4
    NoCheck = 5

    @classmethod
    def parse(cls, arg):
        fallback = cls.Auto
        if arg is None:
            return fallback
        elif isinstance(arg, EffortLevel):
            return arg
        else:
            for eff in EffortLevel:
                if eff.value == arg:
                    return eff
                elif is_string(arg) and arg.lower() == eff.name.lower():
                    return eff

            return fallback


# problem type conversion
_problemtype_map = {0: "LP",
                    1: "MILP",
                    3: "fixed_MILP",
                    4: "nodeLP",
                    5: "QP",
                    7: "MIQP",
                    8: "fixed_MIQP",
                    9: "node_QP",
                    10: "QCP",
                    11: "MIQCP",
                    12: "node_QCP"}


def int_probtype_to_string(probtype, fallback_probtype="unknown"):
    try:
        iprobe_type = int(probtype)
        return _problemtype_map.get(iprobe_type, fallback_probtype)
    except ValueError:
        return fallback_probtype
