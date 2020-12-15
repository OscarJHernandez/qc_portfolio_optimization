# returns a relaxed model from a given model

from docplex.mp.utils import DocplexLinearRelaxationError

from collections import defaultdict
import six


class LinearRelaxer(object):
    """ This class returns a linear relaxation for a MIP model.

    """

    def __init__(self):
        self._unrelaxables = defaultdict(list)

    def iter_unrelaxables(self):
        return six.iteritems(self._unrelaxables)

    def main_cause(self):
        unrelaxables = self._unrelaxables
        max_urx = -1
        justifier = None
        main_cause = None
        for cause, urxs in six.iteritems(unrelaxables):
            if len(urxs) > max_urx:
                max_urx = len(urxs)
                main_cause = cause
                justifier = urxs[0]
        return main_cause, justifier

    @staticmethod
    def make_relaxed_model(mdl, **kwargs):
        lrx = LinearRelaxer()
        return lrx.linear_relaxation(mdl, **kwargs)

    def linear_relaxation(self, mdl, **kwargs):
        """ Returns a continuous relaxation of the model.

        Variable types are set to continuous (note that semi-xxx variables have their LB set to zero)

        Some constructs are not relaxable, for example, piecewise-linear expressions, SOS sets,
        logical constraints...
        When a model contains at least one of these non-relaxable constructs, a message is printed
        and this method returns None.

        By default, model parameters are not copied. If you want to copy them, pass the
        keyword argument `copy_parameters=True`

        :param mdl: the initial model

        :return: a new model with continuous relaxation, if possible, else None.
        """

        relax_name = kwargs.get('relaxed_name', None)
        verbose = kwargs.get('verbose', True)
        copy_parameters = kwargs.get('copy_parameters', False)

        model_name = mdl.name

        def info(msg):
            if verbose:
                print("* relaxation of model {0}: {1}".format(model_name, msg))

        mdl_class = mdl.__class__
        unrelaxables = defaultdict(list)

        def process_unrelaxable(urx_, reason):
            unrelaxables[reason or 'unknown'].append(urx_)

        relax_model_name = relax_name or "lp_%s" % mdl.name
        relaxed_model = mdl_class(name=relax_model_name)

        # transfer kwargs
        relaxed_model._parse_kwargs(mdl._get_kwargs())

        # transfer variable containers
        ctn_map = {}
        for ctn in mdl.iter_var_containers():
            copied_ctn = ctn.copy_relaxed(relaxed_model)
            ctn_map[ctn] = copied_ctn

        # transfer variables
        var_mapping = {}
        continuous = relaxed_model.continuous_vartype
        for v in mdl.iter_variables():
            cpx_code = v.cplex_typecode
            if not v.is_generated() or cpx_code == 'C':
                # if v has type semixxx, set lB to 0
                if cpx_code in {'N', 'S'}:
                    rx_lb = 0
                else:
                    rx_lb = v.lb
                copied_var = relaxed_model._var(continuous, rx_lb, v.ub, v.name)
                var_ctn = v.container
                if var_ctn:
                    copied_ctn = ctn_map.get(var_ctn)
                    assert copied_ctn is not None
                    copied_var.container = copied_ctn
                var_mapping[v] = copied_var

        # transfer all non-logical cts
        for ct in mdl.iter_constraints():
            if not ct.is_generated():
                if ct.is_logical():
                    process_unrelaxable(ct, 'logical')
                try:
                    copied_ct = ct.relaxed_copy(relaxed_model, var_mapping)
                    relaxed_model.add(copied_ct)
                except DocplexLinearRelaxationError as xe:
                    process_unrelaxable(xe.object, xe.cause)
                except KeyError as ke:
                    info('failed to relax constraint: {0}'.format(ct))
                    process_unrelaxable(ct, 'key')

        # clone objective
        relaxed_model.objective_sense = mdl.objective_sense
        try:
            relaxed_model.objective_expr = mdl.objective_expr.relaxed_copy(relaxed_model, var_mapping)
        except DocplexLinearRelaxationError as xe:
            process_unrelaxable(urx_=xe.object, reason=xe.cause)
        except KeyError:
            process_unrelaxable(urx_=mdl.objective_expr, reason='objective')

        # clone kpis
        for kpi in mdl.iter_kpis():
            relaxed_model.add_kpi(kpi.copy(relaxed_model, var_mapping))

        if mdl.context:
            relaxed_model.context = mdl.context.copy()

        if copy_parameters:
            # copy parameters is not the default behavior
            # by default, the relaxed copy has a clean, default, parameter set.
            # if verbose:
            #     info("copying initial model parameters to relaxed model")
            nb_copied = 0
            for p1, p2 in zip(mdl.parameters.generate_params(), relaxed_model.parameters.generate_params()):
                if p1.is_nondefault():
                    p2.set(p1.get())
                    nb_copied += 1
            if verbose:
                info("copied {0} initial model parameters to relaxed model".format(nb_copied))

        #
        for sos in mdl.iter_sos():
            unrelaxables['sos'].append(sos)

        self._unrelaxables = unrelaxables
        if unrelaxables:
            nb_unrelaxables = len(unrelaxables)
            main_cause, justifier = self.main_cause()

            print("* model {0}: found {1} un-relaxable elements, main cause is {2} (e.g. {3})"
                  .format(mdl.name, nb_unrelaxables, main_cause, justifier))
            if verbose:
                for cause, urxs in six.iteritems(unrelaxables):
                    print('* reason: {0}: {1} unrelaxables'.format(cause, len(urxs)))
                    for u, urx in enumerate(urxs):
                        if hasattr(urx, "is_generated") and urx.is_generated():
                            s_gen = " [generated]"
                        else:
                            s_gen = ""
                        print('--  {0}: cannot be relaxed: {1!s}{2}'.format(u + 1, urx, s_gen))

            return None
        else:
            # force cplex if any, on docloud nothing to do...
            cpx = relaxed_model.get_cplex(do_raise=False)
            if cpx:
                # force type to LP
                cpx.set_problem_type(0)  # 0 is code for LP.
            # sanity  check...
            assert not relaxed_model._contains_discrete_artefacts()
            assert not relaxed_model._solved_as_mip()
            # ---
            return relaxed_model
