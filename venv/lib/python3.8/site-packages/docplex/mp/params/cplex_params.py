# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2018
# --------------------------------------------------------------------------

# gendoc: ignore
from docplex.mp.params.parameter_hierarchy_12620 import make_root_params_12620
from docplex.mp.params.parameter_hierarchy_12630 import make_root_params_12630
from docplex.mp.params.parameter_hierarchy_12700 import make_root_params_12700
from docplex.mp.params.parameter_hierarchy_12710 import make_root_params_12710
from docplex.mp.params.parameter_hierarchy_12800 import make_root_params_12800
from docplex.mp.params.parameter_hierarchy_12900 import make_root_params_12900
from docplex.mp.params.parameter_hierarchy_121000 import make_root_params_121000
from docplex.mp.params.parameter_hierarchy_20100 import make_root_params_20100


def _make_default_parameters():
    params = make_root_params_121000()
    print("-- no cplex version found, using default parameter version: {0}".format(params.cplex_version))
    return params


def get_params_from_cplex_version(cpx_version):
    # INTERNAL
    # returns a parameter tree depending on the cplex version, if any.
    # if none is found, returns a default version.
    if cpx_version is None:
        # this can happen, protect from startswith failure
        return _make_default_parameters()
    if cpx_version.startswith("12.8.0"):
        return make_root_params_12800()
    elif cpx_version.startswith("12.9.0"):
        return make_root_params_12900()
    elif cpx_version.startswith("12.10.0"):
        return make_root_params_121000()
    elif cpx_version.startswith("20.1.0"):
        return make_root_params_20100()
    else:
        return _make_default_parameters()
