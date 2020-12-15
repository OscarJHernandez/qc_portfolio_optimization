# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


class VarType(object):
    """VarType()

    This abstract class is the parent class for all types of decision variables.

    This class must never be instantiated.
    Specialized sub-classes are defined for each type of decision variable.

    """
    def __init__(self, short_name, lb, ub, cplex_typecode):
        self._short_name = short_name
        self._lb = lb
        self._ub = ub
        self._cpx_typecode = cplex_typecode

    @property
    def cplex_typecode(self):
        """ This property returns the CPLEX type code for this type.
        Possible values are:
            'B' for binary type
            'I' for integer type
            'C' for continuous type
            'S' for semicontinuous type
            'N' for semiinteger type

        :return: a one-letter string.
        """
        return self._cpx_typecode

    @property
    def short_name(self):
        """ This property returns a short name string for the type.
        """
        return self._short_name

    @property
    def default_lb(self):
        """  This property returns the default lower bound for the type.
        """
        return self._lb

    @property
    def default_ub(self):
        """  This property returns the default upper bound for the type.
        """
        return self._ub

    def resolve_lb(self, candidate_lb, logger):
        if candidate_lb is None:
            resolved_lb = self._lb
        else:
            resolved_lb = self._compute_lb(candidate_lb, logger)
        return resolved_lb

    def resolve_ub(self, candidate_ub, logger):
        if candidate_ub is None:
            resolved_ub = self._ub
        else:
            resolved_ub = self._compute_ub(candidate_ub, logger)
        return resolved_ub

    def _compute_lb(self, candidate_lb, logger):  # pragma: no cover
        # INTERNAL
        raise NotImplementedError

    def _compute_ub(self, candidate_ub, logger):  # pragma: no cover
        # INTERNAL
        raise NotImplementedError

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: True if the type is a discrete type.
        """
        raise NotImplementedError  # pragma: no cover

    def accept_value(self, numeric_value, tolerance=1e-6):
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def _is_within_bounds_and_tolerance(cls, candidate_value, lb, ub, tolerance):
        assert tolerance >= 0
        if candidate_value < lb - tolerance:
            res = False
        elif candidate_value > ub + tolerance:
            res = False
        else:
            res = True
        return res

    @classmethod
    def _is_int_within_tolerance(cls, candidate_value, tolerance):
        assert tolerance >= 0
        return abs(candidate_value - round(candidate_value)) <= tolerance

    def accept_domain_value(self, candidate_value, lb, ub, tolerance):
        return self.accept_value(candidate_value, tolerance) and\
               self._is_within_bounds_and_tolerance(candidate_value, lb, ub, tolerance)

    def to_string(self):
        """
        Returns:
            string: A string representation of the type.
        """
        return "VarType_%s" % self.short_name

    def __str__(self):
        return self.to_string()

    def __eq__(self, other):
        return type(other) == type(self)

    def __ne__(self, other):
        return type(other) != type(self)

    def _hash_vartype(self):  # pragma: no cover
        return hash(self.cplex_typecode)


class BinaryVarType(VarType):
    """BinaryVarType()

        This class models the binary variable type and
        is not meant to be instantiated. Each model contains one instance of
        this type.
    """

    def __init__(self):
        VarType.__init__(self, short_name="binary", lb=0, ub=1, cplex_typecode='B')

    def _compute_lb(self, candidate_lb, logger):
        # INTERNAL
        if candidate_lb >= 1 + 1e-6:
            logger.fatal('Lower bound for binary variable should be less than 1, {0} was passed '.format(candidate_lb))
        # return the user bound anyway
        return candidate_lb

    def _compute_ub(self, candidate_ub, logger):
        # INTERNAL
        if candidate_ub <= -1e-6:
            logger.fatal('Upper bound for binary variable should be greater than 0, {0} was passed'.format(candidate_ub))
        # return the user bound anyway
        return candidate_ub

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: True as this is a discrete type.
        """
        return True

    def accept_value(self, numeric_value, tolerance=1e-6):
        return -tolerance <= numeric_value <= tolerance or\
               (1-tolerance <= numeric_value <= 1 + tolerance)

    def __hash__(self):  # pragma: no cover
        return VarType._hash_vartype(self)


class ContinuousVarType(VarType):
    """ContinuousVarType()

        This class models the continuous variable type and
        is not meant to be instantiated. Each model contains one instance of this type.
    """

    def __init__(self, plus_infinity=1e+20):
        VarType.__init__(self, short_name="continuous", lb=0, ub=plus_infinity, cplex_typecode='C')
        self._plus_infinity = plus_infinity
        self._minus_infinity = - plus_infinity

    def _compute_ub(self, candidate_ub, logger):
        return min(candidate_ub, self._plus_infinity)

    def _compute_lb(self, candidate_lb, logger):
        return max(candidate_lb, self._minus_infinity)

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: False because this type is not a discrete type.
        """
        return False

    def accept_value(self, numeric_value, tolerance=1e-6):
        return self._minus_infinity <= numeric_value <= self._plus_infinity

    def __hash__(self):  # pragma: no cover
        return VarType._hash_vartype(self)


class IntegerVarType(VarType):
    """IntegerVarType()
    This class models the integer variable type and
    is not meant to be instantiated. Each models contains one instance
    of this type.

    """

    def __init__(self, plus_infinity=1e+20):
        VarType.__init__(self, short_name="integer", lb=0, ub=plus_infinity, cplex_typecode='I')
        self._plus_infinity = plus_infinity
        self._minus_infinity = -plus_infinity

    def _compute_ub(self, candidate_ub, logger):
        return min(candidate_ub, self._plus_infinity)

    def _compute_lb(self, candidate_lb, logger):
        return max(candidate_lb, self._minus_infinity)

    def is_discrete(self):
        """  Checks if this is a discrete type.

        Returns:
            Boolean: True as this is a discrete type.
        """
        return True

    def accept_value(self, numeric_value, tolerance=1e-6):
        return self._is_int_within_tolerance(numeric_value, tolerance)

    def __hash__(self):  # pragma: no cover
        return VarType._hash_vartype(self)


class SemiContinuousVarType(VarType):
    """SemiContinuousVarType()

            This class models the :index:`semi-continuous` variable type and
            is not meant to be instantiated.
    """

    def __init__(self, plus_infinity=1e+20):
        VarType.__init__(self, short_name="semi-continuous", lb=1e-6, ub=plus_infinity, cplex_typecode='S')
        self._plus_infinity = plus_infinity

    def _compute_ub(self, candidate_ub, logger):
        return self._plus_infinity if candidate_ub >= self._plus_infinity else float(candidate_ub)

    def _compute_lb(self, candidate_lb, logger):
        if candidate_lb <= 0:
            logger.fatal(
                'semi-continuous variable expects strict positive lower bound, not: {0}'.format(candidate_lb))
        return candidate_lb

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: False because this type is not a discrete type.
        """
        return False

    def accept_value(self, numeric_value, tolerance=1e-6):
        return 0 <= numeric_value <= self._plus_infinity

    def accept_domain_value(self, candidate_value, lb, ub, tolerance):
        return 0 == candidate_value or self._is_within_bounds_and_tolerance(candidate_value, lb, ub, tolerance)

    def __hash__(self):  # pragma: no cover
        return VarType._hash_vartype(self)


class SemiIntegerVarType(VarType):
    """SemiIntegerVarType()

            This class models the :index:`semi-integer` variable type and
            is not meant to be instantiated.
    """

    def __init__(self, plus_infinity=1e+20):
        VarType.__init__(self, short_name="semi-integer", lb=1e-6, ub=plus_infinity, cplex_typecode='N')
        self._plus_infinity = plus_infinity

    def _compute_ub(self, candidate_ub, logger):
        return min(candidate_ub, self._plus_infinity)

    def _compute_lb(self, candidate_lb, logger):
        if candidate_lb <= 0:
            logger.fatal('semi-integer variable expects strict positive lower bound, not: {0}'.format(candidate_lb))
        return candidate_lb

    def is_discrete(self):
        """ Checks if this is a discrete type.

        Returns:
            Boolean: True because this type is an integer type.
        """
        return True

    def accept_value(self, numeric_value, tolerance=1e-6):
        if 0 == numeric_value:
            return True
        return numeric_value >= 0 and self._is_int_within_tolerance(numeric_value, tolerance)

    def accept_domain_value(self, candidate_value, lb, ub, tolerance):
        return 0 == candidate_value or self._is_within_bounds_and_tolerance(candidate_value, lb, ub, tolerance)

    def __hash__(self):  # pragma: no cover
        return VarType._hash_vartype(self)




