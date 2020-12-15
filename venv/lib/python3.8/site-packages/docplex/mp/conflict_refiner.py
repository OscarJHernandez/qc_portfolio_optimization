# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
import warnings
from collections import namedtuple, defaultdict

from docplex.mp.utils import is_string
from docplex.mp.constants import ComparisonType, VarBoundType
from docplex.mp.context import check_credentials
from docplex.mp.cloudutils import context_must_use_docloud

from docplex.mp.utils import str_maxed
from docplex.mp.publish import PublishResultAsDf



TConflictConstraint = namedtuple("_TConflictConstraint", ["name", "element", "status"])


def trim_field(element):
    return str_maxed(element, maxlen=50)


def to_output_table(conflicts, use_df=True):
    # Returns the output tables, as df if pandas is available or as a list
    # of named tuple ['Type', 'Status', 'Name', 'Expression']
    columns = ['Type', 'Status', 'Name', 'Expression']
    TOutputTables = namedtuple('TOutputTables', columns)

    def convert_to_pandas_df(c):
        return {'Type': 'Constraint',
                'Status': c.status.name if c.status is not None else '',
                'Name': c.name or '',
                'Expression': trim_field(c.element)}

    def convert_to_namedtuples(c):
        return TOutputTables('Constraint',
                             c.status,
                             c.name or '',
                             trim_field(c.element))

    pandas = None
    if use_df:
        try:
            import pandas
        except ImportError:  # pragma: no cover
            print("pandas module not found...")
            pandas = None

    data_converter = convert_to_pandas_df if pandas and use_df else convert_to_namedtuples
    output_data = list(map(data_converter, conflicts))

    if use_df:
        return pandas.DataFrame(columns=columns, data=output_data)
    else:
        return output_data


class ConflictRefinerResult(object):
    """ This class contains all conflicts as returned by the conflict refiner.

    A conflict refiner result contains a list of named tuples of type ``TConflictConstraint``,
    the fields of which are:

        - an enumerated value of type  ``docplex.mp.constants.ConflictStatus`` that indicates the
                  conflict status type (Excluded, Possible_member, Member...).
        - the name of the constraint or None if the constraint corresponds to a variable lower or upper bound.
        - a modeling object involved in the conflict:
            can be either a constraint or a wrapper representing a variable upper or lower bound.


        *New in version 2.11*
    """

    def __init__(self, conflicts, refined_by=None):
        self._conflicts = conflicts
        assert refined_by is None or is_string(refined_by)
        self._refined_by = refined_by

    @property
    def refined_by(self):
        '''
        Returns a string indicating how the conflicts were produced.

        - If the conflicts are created by a program, this field returns None.
        - If the conflicts originated from a local CPLEX run, this method returns 'cplex_local'.
        - If the conflicts originated from a DOcplexcloud run, this method returns 'cplex_cloud'.

        Returns:
            A string, or None.

        '''
        return self._refined_by

    def __iter__(self):
        return self.iter_conflicts()

    def __len__(self):
        """ Redefintion of maguic method __len__.

        Allows calling len() on an instance of ConflictRefinerResult
        to get the number of conflicts

        :return: the number of conflicts.
        """
        return len(self._conflicts)

    def iter_conflicts(self):
        """ Returns an iterator on conflicts (named tuples)

        :return: an iterator
        """
        return iter(self._conflicts)

    @property
    def number_of_conflicts(self):
        """ This property returns the number of conflicts. """
        return len(self._conflicts)

    def display(self):
        """ Displays all conflicts.

        """
        print('conflict(s): {0}'.format(self.number_of_conflicts))
        for conflict in self.iter_conflicts():
            st = conflict.status
            elt = conflict.element
            if hasattr(conflict.element, 'as_constraint'):
                ct = conflict.element.as_constraint()
                label = elt.short_typename
            else:
                ct = elt
                label = ct.__class__.__name__
            print("  - status: {1}, {0}: {2!s}".format(label, st.name, str_maxed(ct, maxlen=40)))

    def display_stats(self):
        """ Displays statistics on conflicts.

        Display show many conflict elements per type.
        """
        def elt_typename(elt):
            try:
                return elt.short_typename.lower()
            except AttributeError:  # pragma: no cover
                return elt.__class__.__name__.lower()

        ncf = self.number_of_conflicts
        print('conflict{1}: {0}'.format(ncf, "s" if ncf > 1 else ""))
        cf_stats = defaultdict(lambda: 0)
        for conflict in self.iter_conflicts():
            elt_type = elt_typename(conflict.element)
            cf_stats[elt_type] += 1
        for eltt, count in cf_stats.items():
            if count:
                print("  - {0}{2}: {1}".format(eltt, count, ("s" if count > 1 else "")))

    def as_output_table(self, use_df=True):
        return to_output_table(self, use_df)


    def print_information(self):
        """ Similar as `display_stats`
        """
        self.display_stats()


class VarBoundWrapper(object):
    # INTERNAL

    def __init__(self, dvar):
        self._var = dvar

    @property
    def var(self):
        return self._var

    @property
    def index(self):
        return self._var.index

    @property
    def short_typename(self):  # pragma: no cover
        return "Variable Bound"

    def as_constraint(self):  # pragma: no cover
        raise NotImplementedError

    def as_constraint_from_symbol(self, op_symbol):
        self_var = self.var
        var_lb = self.var.lb
        op = ComparisonType.cplex_ctsense_to_python_op(op_symbol)
        ct = op(self_var, var_lb)
        return ct

    @classmethod
    def make_wrapper(cls, var, bound_type):
        if bound_type == VarBoundType.LB:
            return VarLbConstraintWrapper(var)
        elif bound_type == VarBoundType.UB:
            return VarUbConstraintWrapper(var)
        else:
            return None


class VarLbConstraintWrapper(VarBoundWrapper):
    """
    This class is a wrapper for a model variable and its associated lower bound.

    Instances of this class are created by the ``refine_conflict`` method when the conflict involves
    a variable lower bound. Each of these instances is then referenced by a ``TConflictConstraint`` namedtuple
    in the conflict list returned by ``refine_conflict``.

    To check whether the lower bound of a variable causes a conflict, wrap the variable and
    include the resulting constraint in a ConstraintsGroup.
    """
    @property
    def short_typename(self):
        return "Lower Bound"

    def as_constraint(self):
        return self.as_constraint_from_symbol('G')


class VarUbConstraintWrapper(VarBoundWrapper):
    """
    This class is a wrapper for a model variable and its associated upper bound.

    Instances of this class are created by the ``refine_conflict`` method when the conflict involves
    a variable upper bound. Each of these instances is then referenced by a ``TConflictConstraint`` namedtuple
    in the conflict list returned by ``refine_conflict``.

    To check whether the upper bound of a variable causes a conflict, wrap the variable and
    include the resulting constraint in a ConstraintsGroup.
    """

    @property
    def short_typename(self):
        return "Upper Bound"

    def as_constraint(self):
        return self.as_constraint_from_symbol('L')


class ConstraintsGroup(object):
    """
    This class is a container for the definition of a group of constraints.
    A preference for conflict refinement is associated to the group.

    Groups may be assigned preference. A group with a higher preference is more likely to be included in the conflict.
    A negative value specifies that the corresponding group should not be considered in the computation
    of a conflict. In other words, such groups are not considered part of the model. Groups with a preference of 0 (zero)
    are always considered to be part of the conflict.

    Args:
        preference: A floating-point number that specifies the preference for the group. The higher the number, the
                    higher the preference.
    """

    __slots__ = ('_preference', '_cts')

    def __init__(self, preference=1.0, cts=None):
        self._preference = preference
        self._cts = []
        if cts is not None:
            self.add_constraints(cts)

    @classmethod
    def from_var(cls, dvar, bound_type, pref):
        """ A class method to build a group fromone variable.

        :param dvar: The variable whose bound is part of the conflict.
        :param bound_type: An enumerated value of type `VarBoundType`
        :param pref: a numerical preference.

        :return: an instance of ConstraintsGroup.

        See Also:
            :class:`docplex.mp.constants.VarBoundType`
        """
        cgg = cls(preference=pref, cts=VarBoundWrapper.make_wrapper(dvar, bound_type))
        return cgg

    @property
    def preference(self):
        return self._preference

    def add_one(self, x):
        if x is not None:
            self._cts.append(x)

    def add_constraint(self, ct):
        self._cts.append(ct)

    def add_constraints(self, cts):
        try:
            for ct in cts:
                self.add_one(ct)
        except TypeError:  # not iterable.
            self.add_one(cts)

    def iter_constraints(self):
        return iter(self._cts)


class ConflictRefiner(PublishResultAsDf, object):
    ''' This class is an abstract algorithm; it operates on interfaces.

    A conflict is a set of mutually contradictory constraints and bounds within a model.
    Given an infeasible model, the conflict refiner can identify conflicting constraints and bounds
    within it. CPLEX refines an infeasible model by examining elements that can be removed from the
    conflict to arrive at a minimal conflict.
    '''

    # static variables for output
    output_table_property_name = 'conflicts_output'
    default_output_table_name = 'conflicts.csv'
    output_table_using_df = True

    def __init__(self, output_processing=None):
        self.output_table_customizer = output_processing

    @classmethod
    def _make_atomic_ct_groups(cls, mdl_iter, pref):
        # returns a list of singleton groups from a model iterator and a numerical preference.
        lcgrps = [ConstraintsGroup(pref, ct) for ct in mdl_iter]
        return lcgrps

    @classmethod
    def var_bounds(cls, mdl, pref=4.0, include_infinity_bounds=True):
        """ Returns a list of singleton groups with variable bounds.

        This method a list of ConstraintGroup objects, each of which contains a variabel bound.
        It replicate sthe behavior of the CPLEX interactive optimizer, that is, it returns

        - lower bounds for non-binary variables if different from 0
        - upper bound for non-binary-variables if non-default

        For binary variables, bounds are not considered, unless the variable is bound; more precisely:
            - lower bound is included if >= 0.5
            - upper bound is included if <= 0.5

        :param mdl: The model being analyzed for conflicts,
        :param pref: the preference for variable bounds, the defaut is 4.0
        :param include_infinity_bounds: a flag indicating whether infi

        :return: a list of `ConstraintsGroup` objects.
        """
        grps = []
        mdl_inf = mdl.infinity
        for dv in mdl.iter_variables():
            lb, ub = dv.lb, dv.ub
            if not dv.is_binary():
                if lb != 0:
                    if include_infinity_bounds or lb > - mdl_inf:
                        grps.append(ConstraintsGroup.from_var(dv, VarBoundType.LB, pref))
                if include_infinity_bounds or ub < mdl_inf:
                    grps.append(ConstraintsGroup.from_var(dv, VarBoundType.UB, pref))
            else:
                if lb >= 0.5:
                    grps.append(ConstraintsGroup.from_var(dv, VarBoundType.LB, pref))
                if ub <= 0.5:
                    grps.append(ConstraintsGroup.from_var(dv, VarBoundType.UB, pref))
        return grps

    @classmethod
    def linear_constraints(cls, mdl, pref=2.0):
        return cls._make_atomic_ct_groups(mdl.iter_linear_constraints(), pref)

    @classmethod
    def logical_constraints(cls, mdl, pref=1.0):
        return cls._make_atomic_ct_groups(mdl.iter_logical_constraints(), pref)

    @classmethod
    def quadratic_constraints(cls, mdl, pref=1.0):
        return cls._make_atomic_ct_groups(mdl.iter_quadratic_constraints(), pref)

    def refine_conflict(self, mdl, preferences=None, groups=None, display=False, **kwargs):
        """ Starts the conflict refiner on the model.

        Args:
            mdl: The model to be relaxed.
            preferences: A dictionary defining constraint preferences.
            groups: A list of ConstraintsGroups.
            display: a boolean flag (default is True); if True, displays the result at the end.
            kwargs: Accepts named arguments similar to solve.

        Returns:
            An object of type `ConflictRefinerResut` which holds all information about
            the minimal conflict.

        See Also:
            :class:`ConflictRefinerResult`

        """

        if mdl.has_multi_objective():
            mdl.fatal("Conflict refiner is not supported for multi-objective")

        # take into account local argument overrides
        context = mdl.prepare_actual_context(**kwargs)

        # log stuff
        saved_context_log_output = mdl.context.solver.log_output
        saved_log_output_stream = mdl.log_output

        try:
            mdl.set_log_output(context.solver.log_output)

            forced_docloud = context_must_use_docloud(context, **kwargs)

            results = None

            have_credentials = False
            if context.solver.docloud:
                have_credentials, error_message = check_credentials(context.solver.docloud)
                if error_message is not None:
                    warnings.warn(error_message, stacklevel=2)
            if forced_docloud:
                if have_credentials:
                    results = self._refine_conflict_cloud(mdl, context, preferences, groups)
                else:
                    mdl.fatal("DOcplexcloud context has no valid credentials: {0!s}", context.solver.docloud)
            # from now on docloud_context is None
            elif mdl.environment.has_cplex:
                # if CPLEX is installed go for it
                results = self._refine_conflict_local(mdl, context, preferences, groups)
            elif have_credentials:
                # no context passed as argument, no Cplex installed, try model's own context
                results = self._refine_conflict_cloud(mdl, context, preferences, groups)
            else:
                # no way to solve.. really
                return mdl.fatal("CPLEX DLL not found: please provide DOcplexcloud credentials")

            # write conflicts table.write_output_table() handles everything related to
            # whether the table should be published etc...
            if self.is_publishing_output_table(mdl.context):
                self.write_output_table(results.as_output_table(self.output_table_using_df), mdl.context)
            if display:
                results.display_stats()
            return results
        finally:
            if saved_log_output_stream != mdl.log_output:
                mdl.set_log_output_as_stream(saved_log_output_stream)
            if saved_context_log_output != mdl.context.solver.log_output:
                mdl.context.solver.log_output = saved_context_log_output

    # noinspection PyMethodMayBeStatic
    def _refine_conflict_cloud(self, mdl, context, preferences=None, groups=None):
        # INTERNAL
        docloud_context = context.solver.docloud
        parameters = context.cplex_parameters
        # see if we can reuse the local docloud engine if any?
        docloud_engine = mdl._engine_factory.new_docloud_engine(model=mdl,
                                                                docloud_context=docloud_context,
                                                                log_output=context.solver.log_output_as_stream)

        mdl.notify_start_solve()
        mdl._fire_start_solve_listeners()
        conflict = docloud_engine.refine_conflict(mdl, preferences=preferences, groups=groups, parameters=parameters)

        mdl._fire_end_solve_listeners(conflict is not None, None)
        #
        return conflict

    # noinspection PyMethodMayBeStatic
    def _refine_conflict_local(self, mdl, context, preferences=None, groups=None):
        parameters = context.cplex_parameters
        self_engine = mdl.get_engine()
        return self_engine.refine_conflict(mdl, preferences, groups, parameters)

    @staticmethod
    def display_conflicts(conflicts):
        """
        This method displays a formatted representation of the conflicts that are provided.

        Args:
           conflicts: An instance of ``ConflictRefinerResult``
        """
        warnings.warn("deprecated: use ConflictRefinerresult.display", DeprecationWarning)
        conflicts.display()
