# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

# This file contains all compatibility stuff between py2/py3
import sys

# py2/py3 compatibility
try:
    from Queue import Queue
except ImportError:
    # noinspection PyUnresolvedReferences
    from queue import Queue

# copy_reg is copyreg in Py3
try:
    import copy_reg as copyreg
except ImportError:  # pragma: no cover
    import copyreg   # pragma: no cover

# we want StringIO to process strings in Py2 and Py3
try:
    from cStringIO import StringIO
except ImportError:  # pragma: no cover
    from io import StringIO  # pragma: no cover

try:
    from string import maketrans as mktrans  # Python 2
except ImportError:                 # pragma: no cover
    def mktrans(a, b):              # pragma: no cover
        return str.maketrans(a, b)  # pragma: no cover


try:
    # noinspection PyCompatibility
    xrange(2)
    fast_range = xrange
except NameError:       # pragma: no cover
    fast_range = range  # pragma: no cover

try:
    from itertools import izip
except ImportError:  # pragma: no cover
    izip = zip       # pragma: no cover

# we want unicode in py2, str otherwise
# unitext() returns a unicode representation of its parameter
if sys.version_info[0] == 3:  # pragma: no cover
    unitext = str
else:
    unitext = unicode

try:  # pragma: no cover
    from itertools import zip_longest as izip_longest
except ImportError:  # pragma: no cover
    from itertools import izip_longest