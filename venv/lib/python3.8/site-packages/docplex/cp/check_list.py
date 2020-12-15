# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2020
# --------------------------------------------------------------------------

"""
This module prints a textual report on environment where docplex.cp is used.
"""

import platform
import sys
import os
import docplex.version as version
import docplex.cp.config as config
import docplex.cp.utils as utils
import docplex.cp.solver.solver as solver
import docplex.cp.__init__ as init


def _check_exec_file(name, file):
    """ Check an executable file and returns warning message if any.
    """
    if utils.is_exe_file(file):
        return None
    if os.path.isfile(file):
        return "{} file '{}' exists but is not executable.".format(name, file)
    return "{} file '{}' does not exists.".format(name, file)


def print_environment_report(out=None):
    """ Print a report on the docplex.cp execution environment

    Args:
        out:  (optional) Output stream, default is stdout
    """
    # Check output
    if out is None:
        out = sys.stdout
    out.write("Execution environment:\n")

    # Initialize warning list
    warnings = []
    msg = init.check_platform_system()
    if msg:
        warnings.append(msg)
    msg = init.check_python_version()
    if msg:
        warnings.append(msg)

    # Print context info
    out.write(" * System: {}, {}\n".format(platform.system(), platform.architecture()[0]))
    is_64bits = sys.maxsize > 2 ** 32
    out.write(" * Python version: {}, {} ({})\n".format(platform.python_version(), "64 bits" if is_64bits else "32 bits", sys.executable))
    out.write(" * Docplex version: {}\n".format(version.docplex_version_string))

    # Print package info
    lpacks = ('numpy', 'panda', 'matplotlib')
    out.write(" * Optional packages: {}\n".format(", ".join("{}: {}".format(p, utils.get_module_version(p)) for p in lpacks)))

    # Print solver info
    ctx = config._get_effective_context()
    agt = ctx.solver.agent
    out.write(" * Solver agent: {}".format(agt))
    if not agt:
        warnings.append("No solver agent is defined.")

    if agt == 'local':
        file = ctx.solver.local.execfile
        out.write(", executable file: '{}'".format(file))
        msg = _check_exec_file("Solver executable", file)
        if msg:
            warnings.append(msg)
    elif agt == 'lib':
        file = ctx.solver.lib.libfile
        out.write(", library file: {}".format(file))
        msg = _check_exec_file("Solver library", file)
        if msg:
            warnings.append(msg)
    out.write('\n')

    sver = solver.get_solver_version()
    out.write(" * Solver version: {}\n".format(sver))
    if sver is None and agt is not None:
        warnings.append("Solver version is not accessible with agent '{}'.".format(agt))

    # Print warnings if any
    if warnings:
        out.write("Warnings:\n")
        for w in warnings:
            out.write(" * {}\n".format(w))
    else:
        out.write("No problem found.\n")


if __name__ == "__main__":
    print_environment_report()
