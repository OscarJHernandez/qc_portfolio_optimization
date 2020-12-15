# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.engine import NoSolveEngine, ZeroSolveEngine
from docplex.mp.docloud_engine import DOcloudEngine
from docplex.mp.utils import is_string
from docplex.mp.context import has_credentials
from docplex.mp.error_handler import docplex_fatal


class EngineFactory(object):
    """ A factory class that manages creation of solver instances.
    """
    _default_engine_map = {"nosolve": NoSolveEngine,
                           "zero": ZeroSolveEngine,
                           "docloud": DOcloudEngine,
                           "cplexcloud": DOcloudEngine}

    def __init__(self, env=None):
        self._engine_types_by_agent = self._default_engine_map.copy()
        # no cplex engine type yet?
        if env is not None:
            self._resolve_cplex(env)

    def _get_engine_type_from_agent(self, agent, default_engine, default_engine_name):
        if agent is None:
            return default_engine
        elif is_string(agent):
            agent_key = agent.lower()
            engine_type = self._engine_types_by_agent.get(agent_key)
            if engine_type:
                return engine_type
            elif 'cplex' == agent_key:
                print('* warning: CPLEX DLL not found in path, using {0} instead'.format(default_engine_name))
                return self._engine_types_by_agent.get(default_engine_name)
            elif '.' in agent:
                # assuming a qualified name, e.g. com.ibm.docplex.quantum.QuantumEngine
                from docplex.mp.internal.mloader import import_class
                try:
                    agent_class = import_class(agent)
                    return agent_class
                except ValueError as ve:
                    print(
                        "Cannot load agent class {0}, expecting 'cplex', 'docloud' or valid class path, error: {1}".format(
                            agent, str(ve)))
                    raise ve
            else:
                docplex_fatal("Unexpected agent name: {0}, expecting 'cplex', 'docloud' or valid class path", agent)

        else:
            # try a class type
            try:
                # noinspection PyUnresolvedReferences
                from inspect import isclass
                if isclass(agent):
                    return agent
            except ImportError:
                if type(agent) == type:
                    return agent

            # agent cannot be mapped to any class.
            docplex_fatal("* unexpected agent: {0!r} -expecting 'cplex', 'docloud', class or class name", agent)

    def _is_cplex_resolved(self):
        return hasattr(self, "_cplex_engine_type")

    def _resolve_cplex(self, env):
        # INTERNAL
        if env is None:
            docplex_fatal("need an environment to resolve cplex, got None")
        if not self._is_cplex_resolved():
            if env.has_cplex:
                env.check_cplex_version()

                from docplex.mp.cplex_engine import CplexEngine

                self._cplex_engine_type = CplexEngine
                # noinspection PyTypeChecker
                self._engine_types_by_agent["cplex"] = CplexEngine
            else:
                self._cplex_engine_type = None

    def _ensure_cplex_resolved(self, env):
        if not self._is_cplex_resolved():
            self._resolve_cplex(env)
        assert self._is_cplex_resolved()

    def new_engine(self, agent, env, model, context=None):
        self._ensure_cplex_resolved(env)

        # compute a default engine and kwargs to use..
        kwargs = {}
        if self._cplex_engine_type:
            # default is CPLEX if we have it
            default_engine_type = self._cplex_engine_type
            default_engine_name = 'cplex'

        else:
            # no CPLEX, no credentials
            default_engine_type = NoSolveEngine
            default_engine_name = 'nosolve'

        if has_credentials(context.solver.docloud):
            kwargs['docloud_context'] = context.solver.docloud
        if context is not None:
            kwargs['context'] = context

        engine_type = self._get_engine_type_from_agent(agent=agent,
                                                       default_engine=default_engine_type,
                                                       default_engine_name=default_engine_name)
        assert engine_type is not None
        try:
            return engine_type(model, **kwargs)
        except TypeError:
            docplex_fatal("agent: {0!s} failed to create instance from model, kwargs.", agent)

    # noinspection PyMethodMayBeStatic
    def new_docloud_engine(self, model, **kwargs):
        # noinspection PyDeprecation
        return DOcloudEngine(model, **kwargs)

    def extend(self, new_agent, new_engine):
        # INTERNAL
        assert new_engine is not None
        self._engine_types_by_agent[new_agent] = new_engine
