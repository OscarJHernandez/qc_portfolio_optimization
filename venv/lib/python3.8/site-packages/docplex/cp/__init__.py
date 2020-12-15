# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

"""
IBM Decision Optimization CPLEX Modeling for Python - Constraint Programming

This package contains a Python API allowing to build Constraint Programming
models and their solving using the Decision Optimization cloud services.
"""

import docplex.version as dcpv
import warnings
import platform
import sys


def check_platform_system():
    """ Check if platform system is compatible with docplex.cp

    Returns:
        Error string if platform is not supported, None if OK.
    """
    psyst = platform.system()
    if psyst.lower() not in ('darwin', 'linux', 'windows', 'microsoft', 'aix'):
        return "docplex.cp is supported on Linux, Windows, Darwin and AIX, not on '{}'. Use it at your own risk.".format(psyst)
    return None


def check_python_version():
    """ Check if python version is compatible with docplex.cp

    Returns:
        Error string if this python version is not supported, None if OK.
    """
    pv = sys.version_info
    if (pv < (2, 7)) or ((pv[0] == 3) and (pv < (3, 4) or pv >= (3, 9))):
        return "docplex.cp is supported by Python versions 2.7.9+, 3.4.x, to 3.8.x, not '{}'. Use it at your own risk."\
               .format('.'.join(str(x) for x in pv))
    return None


# Set version information
__version_info__ = (dcpv.docplex_version_major, dcpv.docplex_version_minor, dcpv.docplex_version_micro)

# Check platform system
msg = check_platform_system()
if msg is not None:
    warnings.warn(msg, RuntimeWarning)

# Check version of Python
msg = check_python_version()
if msg is not None:
    warnings.warn(msg, RuntimeWarning)

