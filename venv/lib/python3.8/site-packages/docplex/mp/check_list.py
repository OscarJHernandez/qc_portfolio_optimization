# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2020
# --------------------------------------------------------------------------

# gendoc: ignore

import importlib
import platform
import sys
import warnings

from sys import version_info


def check_import(mname):
    try:
        importlib.import_module(mname)
        return True
    except ImportError:
        return False


def check_platform():
    platform_error_msg = "docplex is not compatible with this version of Python: only 64 bits on Windows, Linux, Darwin and AIX, with Python 2.7.9+, 3.4+ are supported."

    platform_system = platform.system()
    if platform_system in ('Darwin', 'Linux', 'Windows', 'Microsoft', 'AIX'):
        if version_info[0] == 3:
            if version_info < (3, 4, 0):
                warnings.warn(platform_error_msg)
        elif version_info[0] == 2:
            if version_info[1] != 7:
                warnings.warn(platform_error_msg)
        else:
            warnings.warn(platform_error_msg)
    else:
        print("docplex is not officially supported on this platform. Use it at your own risk.", RuntimeWarning)

    is_64bits = sys.maxsize > 2 ** 32
    if is_64bits is False:
        warnings.warn("docplex is not officially supported on 32 bits. Use it at your own risk.", RuntimeWarning)


def run_docplex_check_list():
    check_platform()
    from docplex.version import latest_cplex_major, latest_cplex_minor
    cplex_latest_version_as_tuple = (latest_cplex_major, latest_cplex_minor)

    diagnostics = []

    # check requirements
    for rm in ["six", "enum", "cloudpickle"]:
        if not check_import(rm):
            diagnostics.append("Module {0} is missing, run: pip install {0}".format(rm))

    # check pandas
    try:
        import pandas as pd  # @UnusedImport
        # noinspection PyUnresolvedReferences
        from pandas import DataFrame
        DataFrame({})
    except ImportError:
        print("-- pandas is not present, some features might be unavailable.")

    from docplex.mp.environment import Environment
    Environment().print_information()

    # check cplex
    try:
        # noinspection PyUnresolvedReferences
        from cplex import Cplex

        cpx = Cplex()
        cpxv = cpx.get_version()
        cpxvt = tuple(float(x) for x in cpx.get_version().split("."))[:2]
        lcpxv = ".".join(str(z) for z in cplex_latest_version_as_tuple)
        if cpxvt < cplex_latest_version_as_tuple:
            diagnostics.append("Your cplex version is '{0}', a newer version '{1}' is available".format(cpxv, lcpxv))
        elif cpxvt > cplex_latest_version_as_tuple:
            print("* Your cplex version {0} is ahead of the latest DOcplex-compatible version {1}, this might not be compatible.".format(cpxv, lcpxv))
        else:
            print("* Your cplex version {0} is the latest available".format(cpxv))
        cpx.end()

    except ImportError as ie:
        Cplex = None
        diagnostics.append("No local installation of CPLEX has been found.")
        print("Cplex DLL not found, error importing cplex: {0!s}".format(ie))
        check_python_path(diagnostics)
    # check creation of an empty model...

    try:
        if Cplex:
            # noinspection PyUnresolvedReferences
            from docplex.mp.model import Model
            Model()
            # promotional?
            if Model.is_cplex_ce():
                print("! Cplex promotional version, limited to 1000 variables, 1000 constraints")
                diagnostics.append("Your local CPLEX edition is limited. Consider purchasing a full license.")

    except ImportError:
        print("Docplex is not present: cannot import class docplex.mp.model")
        diagnostics.append("Your installation of DOcplex may be corrupted.")
    except Exception as e:
        print("Exception raised when creating one model instance: {0!s}".format(e))
        diagnostics.append("Your installation of DOcplex may be corrupted.")

    if diagnostics:
        print("\n* diagnostics: {0}".format(len(diagnostics)))
        for s in diagnostics:
            print("  -- {0}".format(s))
    else:
        print("> No problem found: you're all set!")


def cplex_system_dir():
    platform_system = platform.system().lower()
    if 'windows' in platform_system:
        return 'x64_win64'
    elif 'linux' in platform_system:
        return 'x86-64_linux'
    elif 'darwin' in platform_system:
        return ''
    else:
        return None


def check_python_path(diagnostics):
    import os
    pypaths = os.environ.get('PYTHONPATH')
    if not pypaths:
        print("No PYTHONPATH is set, you must add cplex Python to the PYTHONPATH to solve")
    else:
        expected_sysname = cplex_system_dir()
        if expected_sysname:
            platform_version = platform.python_version_tuple()
            py_version2 = "%s.%s" % platform_version[:2]
            for ppath in pypaths.split(';'):
                base, last = os.path.split(ppath)
                if last == expected_sysname and py_version2 not in base:
                    diagnostics.append("Possible version mismatch in PYTHONPATH: {0}".format(ppath))


if __name__ == "__main__":
    run_docplex_check_list()
