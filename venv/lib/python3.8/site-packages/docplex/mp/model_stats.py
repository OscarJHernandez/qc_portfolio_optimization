# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from docplex.mp.compat23 import StringIO


class ModelStatistics(object):
    """ModelStatistics()

    This class gathers statistics from the model.

    Instances of this class are returned by the method :func:`docplex.mp.model.Model.get_statistics`.

    The class contains counters on the various types of variables and constraints
    in the model.

    """

    def __init__(self, nb_bvs, nb_ivs, nb_cvs,
                 nb_scvs, nb_sivs,
                 nb_le_cts, nb_ge_cts, nb_eq_cts,
                 nb_rng_cts,
                 nb_ind_cts, nb_equiv_cts, nb_quad_cts):
        self._number_of_binary_variables = nb_bvs
        self._number_of_integer_variables = nb_ivs
        self._number_of_continuous_variables = nb_cvs
        self._number_of_semicontinuous_variables = nb_scvs
        self._number_of_semiinteger_variables = nb_sivs
        self._number_of_le_constraints = nb_le_cts
        self._number_of_ge_constraints = nb_ge_cts
        self._number_of_eq_constraints = nb_eq_cts
        self._number_of_range_constraints = nb_rng_cts
        self._number_of_indicator_constraints = nb_ind_cts
        self._number_of_equivalence_constraints = nb_equiv_cts
        self._number_of_quadratic_constraints = nb_quad_cts

    def as_tuple(self):
        return (self._number_of_binary_variables,
                self._number_of_integer_variables,
                self._number_of_continuous_variables,
                self._number_of_semicontinuous_variables,
                self._number_of_semiinteger_variables,
                self._number_of_le_constraints,
                self._number_of_ge_constraints,
                self._number_of_eq_constraints,
                self._number_of_range_constraints,
                self._number_of_indicator_constraints,
                self._number_of_equivalence_constraints,
                self._number_of_quadratic_constraints)

    def equal_stats(self, other):
        return isinstance(other, ModelStatistics) and (self.as_tuple() == other.as_tuple())

    def __eq__(self, other):
        return self.equal_stats(other)

    @property
    def number_of_variables(self):
        """ This property returns the total number of variables in the model.

        """
        return self._number_of_binary_variables \
               + self._number_of_integer_variables \
               + self._number_of_continuous_variables \
               + self._number_of_semicontinuous_variables

    @property
    def number_of_binary_variables(self):
        """ This property returns the number of binary decision variables in the model.

        """
        return self._number_of_binary_variables

    @property
    def number_of_integer_variables(self):
        """ This property returns the number of integer decision variables in the model.

        """
        return self._number_of_integer_variables

    @property
    def number_of_continuous_variables(self):
        """ This property returns the number of continuous decision variables in the model.

        """
        return self._number_of_continuous_variables

    @property
    def number_of_semicontinuous_variables(self):
        """ This property returns the number of semicontinuous decision variables in the model.

        """
        return self._number_of_semicontinuous_variables

    @property
    def number_of_semiinteger_variables(self):
        """ This property returns the number of semi-integer decision variables in the model.

        """
        return self._number_of_semiinteger_variables

    @property
    def number_of_linear_constraints(self):
        """ This property returns the total number of linear constraints in the model.

        This number comprises all relational constraints: <=, ==, and >=
        and also range constraints.

        """
        return self._number_of_eq_constraints + \
               self._number_of_le_constraints + \
               self._number_of_ge_constraints + \
               self._number_of_range_constraints

    @property
    def number_of_le_constraints(self):
        """ This property returns the number of <= constraints

        """
        return self._number_of_le_constraints

    @property
    def number_of_eq_constraints(self):
        """ This property returns the number of == constraints

        """
        return self._number_of_eq_constraints

    @property
    def number_of_ge_constraints(self):
        """ This property returns the number of >= constraints

        """
        return self._number_of_ge_constraints

    @property
    def number_of_range_constraints(self):
        """ This property returns the number of range constraints.

        Range constraints are of the form L <= expression <= U.

        See Also:
            :class:`docplex.mp.constr.RangeConstraint`

        """
        return self._number_of_range_constraints

    @property
    def number_of_indicator_constraints(self):
        """ This property returns the number of indicator constraints.

        See Also:
            :class:`docplex.mp.constr.IndicatorConstraint`

        """
        return self._number_of_indicator_constraints

    @property
    def number_of_equivalence_constraints(self):
        """ This property returns the number of equivalence constraints.

        See Also:
            :class:`docplex.mp.constr.EquivalenceConstraint`

        """
        return self._number_of_equivalence_constraints

    @property
    def number_of_quadratic_constraints(self):
        """ This property returns the number of quadratic constraints.

        See Also:
            :class:`docplex.mp.constr.QuadraticConstraint`

        """
        return self._number_of_quadratic_constraints

    @property
    def number_of_constraints(self):
        return self.number_of_linear_constraints + \
               self.number_of_quadratic_constraints + \
               self.number_of_indicator_constraints + \
               self._number_of_equivalence_constraints

    def print_information(self):
        """ Prints model statistics in readable format.

        """
        print(' - number of variables: {0}'.format(self.number_of_variables))
        var_fmt = '   - binary={0}, integer={1}, continuous={2}'
        if self._number_of_semicontinuous_variables:
            var_fmt += ', semi-continuous={3}'
        print(var_fmt.format(self.number_of_binary_variables,
                             self.number_of_integer_variables,
                             self.number_of_continuous_variables,
                             self._number_of_semicontinuous_variables
                             ))

        print(' - number of constraints: {0}'.format(self.number_of_constraints))
        ct_fmt = '   - linear={0}'
        if self._number_of_indicator_constraints:
            ct_fmt += ', indicator={1}'
        if self._number_of_equivalence_constraints:
            ct_fmt += ', equiv={2}'
        if self._number_of_quadratic_constraints:
            ct_fmt += ', quadratic={3}'
        print(ct_fmt.format(self.number_of_linear_constraints,
                            self.number_of_indicator_constraints,
                            self.number_of_equivalence_constraints,
                            self.number_of_quadratic_constraints))

    def to_string(self):
        oss = StringIO()
        oss.write(" - number of variables: %d\n" % self.number_of_variables)
        var_fmt = '   - binary={0}, integer={1}, continuous={2}'
        if self._number_of_semicontinuous_variables:
            var_fmt += ', semi-continuous={3}'
        oss.write(var_fmt.format(self.number_of_binary_variables,
                                 self.number_of_integer_variables,
                                 self.number_of_continuous_variables,
                                 self._number_of_semicontinuous_variables
                                 ))
        oss.write('\n')
        nb_constraints = self.number_of_constraints
        oss.write(' - number of constraints: {0}\n'.format(nb_constraints))
        if nb_constraints:
            ct_fmt = '   - linear={0}'
            if self._number_of_indicator_constraints:
                ct_fmt += ', indicator={1}'
            if self._number_of_equivalence_constraints:
                ct_fmt += ', equiv={2}'
            if self._number_of_quadratic_constraints:
                ct_fmt += ', quadratic={3}'
            oss.write(ct_fmt.format(self.number_of_linear_constraints,
                                    self.number_of_indicator_constraints,
                                    self.number_of_equivalence_constraints,
                                    self.number_of_quadratic_constraints))
        return oss.getvalue()

    def __str__(self):
        return self.to_string()

    def __repr__(self):  # pragma: no cover
        return "docplex.mp.Model.ModelStatistics()"
