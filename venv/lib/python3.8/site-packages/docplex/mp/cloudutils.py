# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore

import six
import warnings
import json


from docplex.mp.context import check_credentials


def is_in_docplex_worker():
    try:
            import docplex.util.environment as runenv
            is_in_worker = isinstance(runenv.get_environment(), runenv.WorkerEnvironment)
    except:
        is_in_worker = False
    return is_in_worker


def context_must_use_docloud(__context, **kwargs):
    # NOTE: the argument CANNOT be named 'context' here as kwargs may well
    # contain a 'context' key
    #
    # returns True if context + kwargs require an execution on cloud
    # this happens in the following cases:
    # (i)  kwargs contains a "docloud_context" key (compat??)
    # (ii) both an explicit url and api_key appear in kwargs
    # (iv) the context's "solver.agent" is "docloud"
    # (v)  kwargs override agent to be "docloud"
    #
    # Always return false when in docplex worker to ignore url/keys/docloud
    # agent override in the worker
    if is_in_docplex_worker():
        return False
    docloud_agent_name = "docloud"  # this might change
    have_docloud_context = kwargs.get('docloud_context') is not None
    if have_docloud_context:
        warnings.warn(
            "Model construction with DOcloudContext is deprecated, use initializer with docplex.mp.context.Context instead.",
            DeprecationWarning, stacklevel=2)
    # TODO: remove have_api_key = get_key_in_kwargs(__context, kwargs)
    # TODO: remove have_url = get_url_in_kwargs(__context, kwargs)

    has_url_key_in_kwargs = False
    if 'url' in kwargs and 'key' in kwargs:
        has_url_key_in_kwargs = is_url_valid(kwargs['url'])

    context_agent_is_docloud = __context.solver.get('agent') == docloud_agent_name
    kwargs_agent_is_docloud = kwargs.get('agent') == docloud_agent_name
    return have_docloud_context \
           or has_url_key_in_kwargs \
           or context_agent_is_docloud \
           or kwargs_agent_is_docloud


def is_url_valid(url):
    return url is not None and isinstance(url, six.string_types) and \
        url.strip().lower().startswith('http')


def context_has_docloud_credentials(context, do_warn=True):
    have_credentials = False
    if context.solver.docloud:
        have_credentials, error_message = check_credentials(context.solver.docloud)
        if error_message is not None and do_warn:
            warnings.warn(error_message, stacklevel=2)
    return have_credentials


def make_new_kpis_dict(allkpis=None, int_vars=None, continuous_vars=None,
                       linear_constraints=None, bin_vars=None,
                       quadratic_constraints=None, total_constraints=None,
                       total_variables=None):
    # This is normally called once at the beginning of a solve
    # those are the details required for docplexcloud and DODS legacy
    kpis_name= [ kpi.name for kpi in allkpis ]
    kpis = {'MODEL_DETAIL_INTEGER_VARS': int_vars,
            'MODEL_DETAIL_CONTINUOUS_VARS': continuous_vars,
            'MODEL_DETAIL_CONSTRAINTS': linear_constraints,
            'MODEL_DETAIL_BOOLEAN_VARS': bin_vars,
            'MODEL_DETAIL_KPIS': json.dumps(kpis_name)}
    # those are the ones required per https://github.ibm.com/IBMDecisionOptimization/dd-planning/issues/2491
    new_details = {'STAT.cplex.size.integerVariables': int_vars,
                   'STAT.cplex.size.continousVariables': continuous_vars,
                   'STAT.cplex.size.linearConstraints': linear_constraints,
                   'STAT.cplex.size.booleanVariables': bin_vars,
                   'STAT.cplex.size.constraints': total_constraints,
                   'STAT.cplex.size.quadraticConstraints': quadratic_constraints,
                   'STAT.cplex.size.variables': total_variables,
                   }
    kpis.update(new_details)
    return kpis
