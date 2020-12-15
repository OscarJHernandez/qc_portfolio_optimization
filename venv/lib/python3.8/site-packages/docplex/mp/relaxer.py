# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from six import iteritems
from collections import defaultdict, namedtuple
from docplex.mp.constants import RelaxationMode

from docplex.mp.utils import is_function, apply_thread_limitations, _to_list
from docplex.mp.basic import Priority
from docplex.mp.constr import AbstractConstraint
from docplex.mp.error_handler import docplex_fatal
from docplex.mp.sdetails import SolveDetails

from docplex.mp.cloudutils import context_must_use_docloud, context_has_docloud_credentials
from docplex.util import as_df
from docplex.mp.publish import PublishResultAsDf

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

TOutputTables = namedtuple('TOutputTables', ['Constraint', 'Priority', 'Amount'])


def to_output_table(relaxer, use_pd=True):
    if pd and use_pd:
        return as_df(relaxer)
    else:
        result = []
        prioritizer = relaxer.prioritizer
        for ct, relaxed in relaxer.iter_relaxations():
            result.append(TOutputTables(ct.name,
                                        prioritizer.get_priority_internal(ct).name,
                                        relaxed))
        return result


# noinspection PyAbstractClass
class Prioritizer(object):
    ''' Abstract base class for prioritizers.
        This class acts like a functor to be called on each model constraint.
    '''

    def __init__(self, override=False):
        self._override = override

    def get_priority(self, ct):
        raise NotImplementedError("base class")  # pragma: no cover

    def get_priority_internal(self, ct):
        ct_prio = ct.priority
        if not self._override and ct_prio:
            return ct_prio
        else:
            return self.get_priority(ct)


class IRelaxationListener(object):
    # INTERNAL
    # ''' Base class for relaxation listeners.'''

    def notify_start_relaxation(self, priority, relaxables):
        ''' This method is called at each step of the relaxation loop.'''
        pass  # pragma: no cover

    def notify_failed_relaxation(self, priority, relaxables):
        ''' This method is called when a relaxation attempt fails.'''
        pass  # pragma: no cover

    def notify_successful_relaxation(self, priority, relaxables, relaxed_obj_value, violations):
        ''' This method is called when a relaxation succeeds.'''
        pass  # pragma: no cover


class VerboseRelaxationListener(IRelaxationListener):
    # INTERNAL
    # ''' A default implementation of the listener, which prints messages.'''

    def __init__(self):
        IRelaxationListener.__init__(self)
        self.relaxation_count = 0

    def notify_start_relaxation(self, priority, relaxables):
        self.relaxation_count += 1
        print("-> relaxation #{0} starts with priority: {1!s}, #relaxables={2:d}"
              .format(self.relaxation_count, priority.name, len(relaxables)))

    def notify_failed_relaxation(self, priority, relaxables):
        print("<- relaxation #{0} fails, priority: {1!s}, #relaxables={2:d}"
              .format(self.relaxation_count, priority.name, len(relaxables)))

    def notify_successful_relaxation(self, priority, relaxables, obj, violations):
        print("<- relaxation #{0} succeeds: priority: {1!s}, #relaxables={2:d}, obj={3}, #relaxations={4}".
              format(self.relaxation_count, priority.name, len(relaxables), obj, len(violations)))


class NamedPrioritizer(Prioritizer):
    # INTERNAL
    # """ Basic prioritizer that relaxes any constraint with a name.
    #
    #     More precisely, the prioritizer logic works as follows:
    #
    #         - If the constraint has a ``priority`` attribute, then it is assumed
    #           to hold a priority instance and use it.
    #         - If the constraint has a user-defined name, relax it with MEDIUM priority.
    #         - Otherwise, the constraint is not to be relaxed (that is, it is assigned MANDATORY priority).
    #
    # """

    def __init__(self, priority=Priority.MEDIUM, override=False):
        Prioritizer.__init__(self, override)
        self._priority = priority

    def get_priority(self, ct):
        return self._priority if ct.has_user_name() else Priority.MANDATORY


class UniformPrioritizer(Prioritizer):
    # INTERNAL
    # """ Constraint prioritizer that relaxes all constraints.
    #
    #     This prioritizer assigns MEDIUM priority to all constraints
    #     unless an explicit priority has been set by the user.
    #
    # """

    def __init__(self, priority=Priority.MEDIUM, override=False):
        Prioritizer.__init__(self, override)
        self._priority = priority

    def get_priority(self, ct):
        return self._priority


class MatchNamePrioritizer(Prioritizer):
    # INTERNAL
    # """ Constraint prioritizer based on constraint names.
    #
    #     This prioritizer analyzes constraint names for strings that match priority names.
    #     If a constraint contains a string which matches a priority name,
    #     then it is assigned this priority.
    #
    #     If a constraint has a priority explicitly set by the ``priority`` attribute,
    #     this user priority is returned.
    #
    #     Note:
    #         1. Unnamed constraints are considered as non-matches.
    #         2. String matching is not case sensitive.
    #
    #     For example: a constraint named "ct_salary_low" will be considered as having the priority LOW.
    # """

    def __init__(self,
                 priority_for_unnamed=Priority.MANDATORY,
                 priority_for_non_matches=Priority.MANDATORY,
                 case_sensitive=False,
                 override=False):
        Prioritizer.__init__(self, override)
        assert isinstance(priority_for_unnamed, Priority)
        assert isinstance(priority_for_non_matches, Priority)

        self.priority_for_unnamed_cts = priority_for_unnamed
        self.priority_for_non_matching_cts = priority_for_non_matches
        self.priority_by_symbol = {prio.name.lower(): prio for prio in Priority}
        self._is_case_sensitive = bool(case_sensitive)

    def get_priority(self, ct):
        ''' Looks for known priority names inside constraint names.
        '''

        if not ct.has_user_name():
            return self.priority_for_unnamed_cts
        else:
            ctname = ct.name
            ctname_to_match = ctname if self._is_case_sensitive else ctname.lower()
            best_matched = 0
            best_matching_priority = self.priority_for_non_matching_cts
            for (prio_symbol, prio) in iteritems(self.priority_by_symbol):
                if ctname_to_match.find(prio_symbol) >= 0:
                    matched = len(prio_symbol)
                    # longer matches are preferred
                    # e.g. very_low and low both match in very_low_ctxxx
                    # but the prioritizer will return very_low as the match is longer.
                    if matched > best_matched:
                        best_matched = matched
                        best_matching_priority = prio
            return best_matching_priority


class MappingPrioritizer(Prioritizer):
    # INTERNAL
    # """
    # Constraint prioritizer based on a dictionary of constraints and priorities.
    #
    # Initialized from a dictionary and an optional default priority.
    #
    # Args:
    #     priority_mapping: A dictionary with constraints as keys and priorities as values.
    #
    #     default_priority: An optional priority, used when a constraint is not explicitly mentioned
    #         in the mapping. The default value is MANDATORY, meaning that any constraint not mentioned
    #         in the mapping will not be relaxed.
    # """

    def __init__(self, priority_mapping, default_priority=Priority.MANDATORY, override=False):
        Prioritizer.__init__(self, override)
        # --- typecheck that this dict is a a {ct: prio} mapping.
        if not isinstance(priority_mapping, dict):
            raise TypeError
        for k, v in iteritems(priority_mapping):
            if not isinstance(k, AbstractConstraint):
                raise TypeError

            if not hasattr(v, 'cplex_preference'):
                raise TypeError
        # ---
        self._mapping = priority_mapping
        self._default_priority = default_priority

    def get_priority(self, ct):
        # return the dict's value for ct if nay, else its own priority or the default.
        return self._mapping.get(ct, ct.priority or self._default_priority)


class FunctionalPrioritizer(Prioritizer):
    def __init__(self, fn, override=False):
        Prioritizer.__init__(self, override)
        self._prioritize_fn = fn

    def get_priority(self, ct):
        return self._prioritize_fn(ct)


# internal named tuples
_TRelaxableGroup = namedtuple("_TRelaxableGroup", ["preference", "relaxables"])
_TParamData = namedtuple('_TParamInfo', ['short_name', 'default_value', 'accessor'])


class Relaxer(PublishResultAsDf, object):
    ''' This class is an abstract algorithm, in the sense that it operates on interfaces.

        It takes a prioritizer, which an implementation of ``ConstraintPrioritizer``.
        For convenience, predefined prioritizer types are accessible  through names:

            - `all` relaxes all constraints using a MEDIUM priority; this is the default.
            - `named` relaxes all constraints with a user name but not the others.
            - `match` looks for priority names within constraint names;
              unnamed constraints are not relaxed.


        Note:
            All predefined prioritizers apply various forms of logic, but, when a constraint has been assigned
            a priority by the user, this priority is always used. For example, the `named` prioritizer relaxes
            all named constraints with MEDIUM, but if an unnamed constraint was assigned a HIGH priority,
            then HIGH will be used.

        See Also:
           :class:`docplex.mp.basic.Priority`

    '''
    default_precision = 1e-5

    _default_mode = RelaxationMode.OptSum

    def __init__(self, prioritizer='all', verbose=False, precision=default_precision,
                 override=False, **kwargs):
        self.output_table_customizer = kwargs.get('output_processing') # wml tables, internal
        self.output_table_property_name = 'relaxations_output'
        self.default_output_table_name = 'relaxations.csv'
        self.output_table_using_df = True  # if pandas is available of course

        self._precision = precision
        # ---
        if hasattr(prioritizer, 'get_priority'):
            self._prioritizer = prioritizer
        elif prioritizer == 'match':
            self._prioritizer = MatchNamePrioritizer(override=override)
        elif isinstance(prioritizer, dict):
            self._prioritizer = MappingPrioritizer(priority_mapping=prioritizer, override=override)
        elif prioritizer == 'named':
            self._prioritizer = NamedPrioritizer()
        elif prioritizer is None or prioritizer == 'all':
            self._prioritizer = UniformPrioritizer(override=override)
        elif is_function(prioritizer):
            self._prioritizer = FunctionalPrioritizer(prioritizer, override=override)
        else:
            print("Cannot deduce a prioritizer from: {0!r} - expecting \"name\"|\"default\"| dict", prioritizer)
            raise TypeError

        self._cumulative = True
        self._listeners = []

        # result data
        self._last_relaxation_status = False
        self._last_relaxation_objective = -1e+75
        self._last_successful_relaxed_priority = Priority.MANDATORY
        self._last_relaxation_details = SolveDetails.make_dummy()
        self._relaxations = {}
        self._verbose = verbose
        self._verbose_listener = VerboseRelaxationListener()

        if self._verbose:
            self.add_listener(self._verbose_listener)

    @property
    def prioritizer(self):
        return self._prioritizer

    def set_verbose(self, is_verbose):
        if is_verbose != self._verbose:
            self._verbose = is_verbose
            self.set_verbose_listener_from_flag(is_verbose)

    def set_verbose_listener_from_flag(self, is_verbose):
        if is_verbose:
            self.add_listener(self._verbose_listener)
        else:
            self.remove_listener(self._verbose_listener)

    def get_verbose(self):
        return self._verbose

    verbose = property(get_verbose, set_verbose)

    def _check_successful_relaxation(self):
        if not self._last_relaxation_status:
            docplex_fatal("No relaxed solution is present")

    def _reset(self):
        # INTERNAL
        self._last_relaxation_status = False
        self._last_relaxation_objective = -1e+75
        self._last_successful_relaxed_priority = Priority.MANDATORY
        self._relaxations = {}

    def _accept_violation(self, violation):
        ''' The filter method which accepts or rejects a violation.'''
        return 0 == self._precision or abs(violation) >= self._precision

    def add_listener(self, listener):
        # INTERNAL
        # """ Adds a relaxation listener.
        #
        # Args:
        #     listener: The new listener to add. If ``listener`` is not an
        #        instance of ``IRelaxationListener``, it is ignored.
        #
        # See Also:
        #     :class:`IRelaxationListener`
        # """
        if isinstance(listener, IRelaxationListener):
            self._listeners.append(listener)

    def remove_listener(self, listener):
        # INTERNAL
        # """ Removes a relaxation listener.
        #
        # Args:
        #     listener: The listener to remove.
        # """
        if listener in self._listeners:
            self._listeners.remove(listener)

    def clear_listeners(self):
        # INTERNAL
        # """ Removes all relaxation listeners.
        # """
        self._listeners = []

    _param_data = {}

    def relax(self, mdl, relax_mode=None, **kwargs):
        """ Runs the relaxation loop.

        Args:
            mdl: The model to be relaxed.
            relax_mode: the relaxation mode. Accept either None (in which case the default mode is
                used, or an instance of ``RelaxationMode`` enumerated type, or a string
                that can be translated to a relaxation mode.
            kwargs: Accepts named arguments similar to ``solve``.

        Returns:
            If the relaxation succeeds, the method returns a solution object, an instance of ``SolveSolution``; otherwise returns None.

        See Also:
            :func:`docplex.mp.model.Model.solve`,
            :class:`docplex.mp.solution.SolveSolution`,
            :class:`docplex.mp.constants.RelaxationMode`

        """
        self._reset()

        # 1. build a dir {priority : cts}
        priority_map = defaultdict(list)
        nb_prioritized_cts = 0
        mdl_priorities = set()
        mandatory_justifier = None
        nb_mandatories = 0
        for ct in mdl.iter_constraints():
            prio = self._prioritizer.get_priority_internal(ct)
            if prio.is_mandatory():
                nb_mandatories += 1
                if mandatory_justifier is None:
                    mandatory_justifier = ct
            else:
                priority_map[prio].append(ct)
                nb_prioritized_cts += 1
                mdl_priorities.add(prio)

        sorted_priorities = sorted(list(mdl_priorities), key=lambda p: p.value)

        if 0 == nb_prioritized_cts:
            mdl.error("Relaxation algorithm found no relaxable constraints - exiting")
            return None
        if nb_mandatories:
            assert mandatory_justifier is not None
            mdl.warning('{0} constraint(s) will not be relaxed (e.g.: {1!s})', nb_mandatories, mandatory_justifier)

        temp_relax_verbose = kwargs.pop('verbose', False)
        if temp_relax_verbose != self._verbose:
            # install/deinstall listener for this relaxation only
            self.set_verbose_listener_from_flag(temp_relax_verbose)

        # relaxation loop
        all_groups = []
        all_relaxable_cts = []
        is_cumulative = self._cumulative
        # -- relaxation mode
        if relax_mode is None:
            used_relax_mode = self._default_mode
        else:
            used_relax_mode = RelaxationMode.parse(relax_mode)
        if not mdl.is_optimized():
            used_relax_mode = RelaxationMode.get_no_optimization_mode(used_relax_mode)

        # save this for restore later
        saved_context_log_output = mdl.context.solver.log_output
        saved_log_output_stream = mdl.log_output
        saved_context = mdl.context

        # take into account local argument overrides
        relax_context = mdl.prepare_actual_context(**kwargs)

        forced_docloud = context_must_use_docloud(relax_context, **kwargs)
        have_credentials = context_has_docloud_credentials(relax_context)
        transient_engine = False
        relax_engine = mdl.get_engine()

        if forced_docloud:  # pragma: no cover
            if have_credentials:
                # create new docloud engine on the fly
                relax_engine = mdl._new_docloud_engine(relax_context)
                transient_engine = True
            else:
                mdl.fatal("DOcplexcloud context has no valid credentials: {0!s}",
                          relax_context.solver.docloud)

        if temp_relax_verbose:
            print("-- starting relaxation. mode: {0!s}, precision={1}".format(used_relax_mode.name, self._precision))

        try:
            # mdl.context has been saved in saved_context above
            mdl.context = relax_context
            mdl.set_log_output(mdl.context.solver.log_output)

            # engine parameters, if needed to
            parameters = apply_thread_limitations(relax_context)

            mdl._apply_parameters_to_engine(parameters)

            relaxed_sol = None
            for prio in sorted_priorities:
                if prio in priority_map:
                    cts = priority_map[prio]
                    if not cts:
                        # this should not happen...
                        continue  # pragma: no cover

                    pref = prio.cplex_preference
                    # build a new group
                    relax_group = _TRelaxableGroup(pref, cts)

                    # relaxing new batch of cts:
                    if not is_cumulative:  #  pragma: no cover
                        # if not cumulative reset the groupset
                        all_groups = [relax_group]
                        all_relaxable_cts = cts
                    else:
                        all_groups.append(relax_group)
                        all_relaxable_cts += cts

                    # at this stage we have a sequence of groups
                    # a group is itself a sequence of two components
                    # - a preference factor
                    # - a sequence of constraints
                    for l in self._listeners:
                        l.notify_start_relaxation(prio, all_relaxable_cts)

                    # ----
                    # call the engine.
                    # ---

                    try:
                        relaxed_sol = relax_engine.solve_relaxed(mdl, prio.name, all_groups, used_relax_mode)
                    finally:
                        self._last_relaxation_details = relax_engine.get_solve_details()
                    # ---

                    if relaxed_sol is not None:
                        relax_obj = relaxed_sol.objective_value
                        self._last_successful_relaxed_priority = prio
                        self._last_relaxation_status = True
                        self._last_relaxation_objective = relaxed_sol.objective_value

                        # filter irrelevant relaxations below some threshold
                        for ct in all_relaxable_cts:
                            raw_infeas = relaxed_sol.get_infeasibility(ct)
                            if self._accept_violation(raw_infeas):
                                self._relaxations[ct] = raw_infeas
                        if not self._relaxations:
                            mdl.warning(
                                "Relaxation of model `{0}` found one relaxed solution, but no relaxed constraints - check".format(
                                    mdl.name))

                        for l in self._listeners:
                            l.notify_successful_relaxation(prio, all_relaxable_cts, relax_obj, self._relaxations)
                        # now get out
                        break
                    else:
                        # TODO: maybe issue a warning that relaxation has failed?
                        # relaxation has failed, notify the listeners
                        for l in self._listeners:
                            l.notify_failed_relaxation(prio, all_relaxable_cts)

            mdl.notify_solve_relaxed(relaxed_sol, relax_engine.get_solve_details())

            # write relaxation table.write_output_table() handles everything related to
            # whether the table should be published etc...
            if self.is_publishing_output_table(mdl.context):
                output_table = to_output_table(self, self.output_table_using_df)
                self.write_output_table(output_table, mdl.context)
        finally:
            # --- restore context, log_output if set.
            if saved_log_output_stream != mdl.log_output:
                mdl.set_log_output_as_stream(saved_log_output_stream)
            if saved_context_log_output != mdl.context.solver.log_output:
                mdl.context.solver.log_output = saved_context_log_output
            mdl.context = saved_context
            if transient_engine:  # pragma: no cover
                del relax_engine
            if temp_relax_verbose != self._verbose:
                # realign listener with flag
                self.set_verbose_listener_from_flag(self._verbose)

        return relaxed_sol

    def iter_relaxations(self):
        """ Iterates on relaxations.

        Relaxations are built as a dictionary with constraints as keys and numeric violations as values,
        so this iterator returns ``(ct, violation)`` pairs.
        """
        self._check_successful_relaxation()
        return iteritems(self._relaxations)

    def relaxations(self):
        """  Returns a dictionary with all relaxed constraints.

        Returns:
           A dictionary where the keys are the relaxed constraints,
          and the values are the numerical slacks.

        """
        return self._relaxations.copy()

    def __as_df__(self):
        ''' Returns a pandas.DataFrame with all relaxed constraint.

        Returns:
           A pandas.DataFrame which columns are:

              - Constraint: the constraint name
              - Priority: The priority of the constraint
              - Amount: The amount of relaxation
        '''
        if not pd:  # pragma: no cover
            raise NotImplementedError('Cannot convert results to DataFrame, pandas is not available')
        columns = ['Constraint', 'Priority', 'Amount']
        results_list = []
        prioritizer = self.prioritizer
        for ct, relaxed in self.iter_relaxations():
            results_list.append({'Constraint': ct.name,
                                 'Priority': prioritizer.get_priority_internal(ct).name,
                                 'Amount': relaxed})
        df = pd.DataFrame(results_list, columns=columns)
        return df

    def get_total_relaxation(self):
        self._check_successful_relaxation()
        return sum(abs(v) for v in self._relaxations.values())

    @property
    def total_relaxation(self):
        return self.get_total_relaxation()

    def print_information(self):
        self._check_successful_relaxation()
        print("* number of relaxations: {0}".format(len(self._relaxations)))
        for rct, relaxation in self.iter_relaxations():
            arg = rct.name if rct.has_user_name() else str(rct)
            print(" - relaxed: {0}, with relaxation: {1}".format(arg, relaxation))
        print("* total absolute relaxation: {0}".format(self.get_total_relaxation()))

    def as_dict(self):
        rxd = {rct.name or str(rct): relaxed for rct, relaxed in self.iter_relaxations()}
        return rxd

    @property
    def relaxed_objective_value(self):
        """  Returns the objective value of the relaxed solution.

        Raises:
            DOCplexException
                If the relaxation has not been successful.
        """
        self._check_successful_relaxation()
        return self._last_relaxation_objective

    @property
    def number_of_relaxations(self):
        """ This property returns the number of relaxations found.
        """
        return len(self._relaxations)

    def get_relaxation(self, ct):
        """ Returns the infeasibility computed for this constraint.

        Args:
            ct: A constraint.

        Returns:
            The amount by which the constraint has been relaxed by the relaxer.
            The method returns 0 if the constraint has not been relaxed.
        """
        self._check_successful_relaxation()
        return self._relaxations.get(ct, 0)

    def is_relaxed(self, ct):
        ''' Returns true if the constraint ``ct`` has been relaxed

        Args:
            ct: The constraint to check.

        Returns:
            True if the constraint has been relaxed, else False.
        '''
        self._check_successful_relaxation()
        return ct in self._relaxations

    @classmethod
    def run_feasopt(cls, model, relaxables, relax_mode):
        relaxable_list = _to_list(relaxables)
        groups = []
        checker = model._checker
        try:
            for pref, ctseq in relaxable_list:
                checker.typecheck_num(pref)
                cts = _to_list(ctseq)
                for ct in cts:
                    checker.typecheck_constraint(ct)
                groups.append((pref, cts))
        except ValueError:
            model.fatal("expecting container with (preference, constraints), got: {0!s}", relaxable_list)

        feasible = model.get_engine().solve_relaxed(mdl=model, relaxable_groups=groups,
                                                    prio_name='feasopt',
                                                    relax_mode=relax_mode)
        return feasible

