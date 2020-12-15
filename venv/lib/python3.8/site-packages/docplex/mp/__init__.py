# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore
"""
This is the MP package.
"""

import platform
import warnings
import sys

from sys import version_info

ERROR_STRING = "docplex is not compatible with this version of Python: only 64 bits on Windows, Linux, Darwin and AIX, with Python 2.7.9+, 3.4+ are supported."

platform_system = platform.system()
if platform_system in ('Darwin', 'Linux', 'Windows', 'Microsoft', 'AIX'):
    if version_info[0] == 3:
        if version_info < (3, 4, 0):
            raise Exception(ERROR_STRING)
    elif version_info[0] == 2:
        if version_info[1] != 7:
            raise Exception(ERROR_STRING)
    else:
        raise Exception(ERROR_STRING)
else:
    warnings.warn("docplex is not officially supported on this platform. Use it at your own risk.", RuntimeWarning)

is_64bits = sys.maxsize > 2**32
if is_64bits is False:
    warnings.warn("docplex is not officially supported on 32 bits. Use it at your own risk.", RuntimeWarning)

from docplex.version import docplex_version_major, docplex_version_minor, docplex_version_micro
__version_info__ = (docplex_version_major, docplex_version_minor, docplex_version_micro)
