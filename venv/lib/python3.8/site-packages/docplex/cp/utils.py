# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Miscellaneous utility functions. Some of theme are here to prevent possible
port problems between the different versions of Python.

This module is theoretically not public, but documented because some public classes are extending
classes defined here.
"""

import os
import time
import logging
import sys
import threading
import io
import inspect
import json
import platform
import copy
import gzip
import zipfile
import importlib
from collections import deque

try:
    from collections.abc import Iterable  # Python >= 3.7
except:
    from collections import Iterable      # Python < 3.7

try:
    from StringIO import StringIO  # Python 2
except ImportError:
    from io import StringIO        # Python 3


###############################################################################
## Constants
###############################################################################

# Python 2 indicator
IS_PYTHON_2 = (sys.version_info[0] == 2)

# Check if numpy is available
try:
    import numpy
    IS_NUMPY_AVAILABLE = True
    NUMPY_NDARRAY = numpy.ndarray
except:
    IS_NUMPY_AVAILABLE = False
    NUMPY_NDARRAY = False

# Check if panda is available
try:
    import pandas
    from pandas import Series as PandaSeries
    IS_PANDA_AVAILABLE = True
except:
    IS_PANDA_AVAILABLE = False

# Constant used to indicate to set a parameter to its default value
# Useful if default value is not static
DEFAULT = "default"

# Platform type
IS_WINDOWS = platform.system() == 'Windows'


###############################################################################
## Public classes
###############################################################################

class CpoException(Exception):
    """ Exception thrown in case of CPO errors.
    """
    def __init__(self, msg):
        """ Create a new exception

        Args:
            msg: Error message
        """
        super(CpoException, self).__init__(msg)


class CpoNotSupportedException(CpoException):
    """ Exception thrown when a CPO function is not supported.
    """
    def __init__(self, msg):
        """ Create a new exception

        Args:
            msg: Error message
        """
        super(CpoNotSupportedException, self).__init__(msg)


class Context(dict):
    """ Class handling miscellaneous list of parameters. """
    def __init__(self, **kwargs):
        """ Create a new context

        Args:
            **kwargs: List of key=value to initialize context with.
        """
        super(Context, self).__init__()
        vars(self)['parent'] = None
        for k, v in kwargs.items():
            self.set_attribute(k, v)


    def __setattr__(self, name, value):
        """ Set a context parameter.

        Args:
            name:  Parameter name
            value: Parameter value
        """
        self.set_attribute(name, value)


    def __getattr__(self, name):
        """ Get a context parameter.

        Args:
            name:  Parameter name
        Return:
            Parameter value, None if not set
        """
        return self.get_attribute(name)


    def set_attribute(self, name, value):
        """ Set a context attribute.

        Args:
            name:  Attribute name
            value: Attribute value
        """
        self[name] = value
        if isinstance(value, Context):
            vars(value)['parent'] = self


    def get_attribute(self, name, default=None):
        """ Get a context attribute.

        This method search first attribute in this context. If not found, it moves up to
        parent context, and continues as long as not found or root is reached.

        Args:
            name:    Attribute name
            default: Optional, default value if attribute is absent
        Return:
            Attribute value, default value if not found
        """
        if name.startswith('__'):
            raise AttributeError
        ctx = self
        while True:
            if name in ctx:
               return ctx.get(name)
            ctx = ctx.get_parent()
            if ctx is None:
                return default


    def del_attribute(self, name):
        """ Remove a context attribute.

        Args:
            name:    Attribute name
        Return:
            True if attribute has been removed, False if it was not present
        """
        if name in self:
            value = self.get(name)
            if isinstance(value, Context):
                vars(value)['parent'] = None
            del(self[name])
            return True
        else:
            return False


    def set_by_path(self, path, value):
        """ Set a context attribute using its path.

        Attribute path is a sequence of attribute names separated by dots.

        Args:
            path:   Attribute path
            value:  Attribute value
        """
        path = path.split('.')
        trgt = self
        # Go down in sub-contexts
        for k in path[:-1]:
            sc = trgt.get(k)
            if sc is None:
                sc = Context()
                trgt.set_attribute(k, sc)
            elif not isinstance(sc, Context):
                raise Exception("Attribute '" + k + "' should be a Context")
            trgt = sc
        # Set final value
        trgt.set_attribute(path[-1], value)


    def get_by_path(self, path, default=None):
        """ Get a context attribute using its path.

        Attribute path is a sequence of attribute names separated by dots.

        Args:
            path:    Attribute path
            default: Optional, default value if attribute is not found
        Return:
            Attribute value, default value if not found
        """
        res = self
        for k in path.split('.'):
            if k:
                res = res.get_attribute(k)
                if res is None:
                    return default
        return res


    def search_and_replace_attribute(self, name, value, path=""):
        """ Replace an existing attribute.

        The attribute is searched first as a value in this context node.
        If not found, it is searched recursively in children contexts, in alphabetical order.

        Args:
            name:  Attribute name
            value: Attribute value, None to remove attribute
            path:  Path of the current node. default is ''
        Return:
            Full path of the attribute that has been found and replaced, None if not found
        """
        sctxs = []  # List of subcontexts
        # Search first in atomic values
        for k in sorted(self.keys()):
            v = self[k]
            if isinstance(v, Context):
                sctxs.append((k, v))
            else:
                if k == name:
                    ov = self.get_attribute(name)
                    npath = path + "." + k if path else k
                    if ov is not None:
                        if isinstance(value, Context):
                            if not isinstance(ov, Context):
                                raise Exception("Attribute '" + npath + "' is a Context and can only be replaced by a Context")
                    self.__setattr__(name, value)
                    return npath
        # Search then in sub-contexts
        for (k, v) in sctxs:
            npath = path + "." + k if path else k
            apth = v.search_and_replace_attribute(name, value, path=npath)
            if apth:
                return apth
        return None


    def get_parent(self):
        """ Get the parent context.

        Each time a context attribute is set to a context, its parent is assigned to the context where it is stored.

        Return:
            Parent context, None if this context is root
        """
        return vars(self)['parent']


    def get_root(self):
        """ Get the root context (last parent with no parent).

        Return:
            Root context
        """
        res = self
        pp = res.get_parent()
        while pp is not None:
            res = pp
            pp = pp.get_parent()
        return res


    def clone(self):
        """ Clone this context and all sub-contexts recursively.

        Return:
            Cloned copy of this context.
        """
        res = type(self)()
        for k, v in self.items():
            if isinstance(v, Context):
                v = v.clone()
            res.set_attribute(k, v)
        return res


    def add(self, ctx):
        """ Add another context to this one.

        All attributes of given context are set in this one, replacing previous value if any.
        If one value is another context, it is cloned before being set.

        Args:
            ctx:  Other context to add to this one.
        """
        for k, v in ctx.items():
            if isinstance(v, Context):
                v = v.clone()
            self.set_attribute(k, v)


    def is_log_enabled(self, vrb):
        """ Check if log is enabled for a given verbosity

        This method get this context 'log_output' attribute to retrieve the log output, and the
        attribute 'verbose' to retrieve the current verbosity level.

        Args:
            vrb:  Required verbosity level, None for always
        """
        return self.log_output and ((vrb is None) or (self.verbose and (self.verbose >= vrb)))


    def get_log_output(self):
        """ Get this context log output

        This method returns the log_output defined in attribute 'log_output' and convert
        it to sys.stdout if its value is True

        Returns:
            Log output stream, None if none
        """
        out = self.log_output
        return sys.stdout if out == True else out


    def log(self, vrb, *msg):
        """ Log a message if log is enabled with enough verbosity

        This method get this context 'log_output' attribute to retrieve the log output, and the
        attribute 'verbose' to retrieve the current verbosity level.

        Args:
            vrb:  Required verbosity level, None for always
            msg:  Message elements to log (concatenated on one line)
        """
        if self.is_log_enabled(vrb):
            out = self.get_log_output()
            prfx = self.log_prefix
            if prfx:
                out.write(str(prfx))
            out.write(''.join([str(m) for m in msg]) + "\n")
            out.flush()


    def print_context(self, out=None, indent=""):
        """ Print this context.

        At each level, atomic values are printed first, then sub-contexts, in alphabetical order.

        DEPRECATED. Use :meth:`write` instead.

        Args:
            out (Optional):    Target output stream or file name. If not given, default value is sys.stdout.
            indent (Optional): Start line indentation. Default is empty
        """
        self.write(out, indent)


    def write(self, out=None, indent=""):
        """ Write this context.

        At each level, atomic values are printed first, then sub-contexts, in alphabetical order.

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out (Optional):    Target output stream or file name. If not given, default value is sys.stdout.
            indent (Optional): Start line indentation. Default is empty
        """
        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self.write(f)
                return
        # Check default output
        if out is None:
            out = sys.stdout

        sctxs = []  # List of subcontexts
        # Print atomic values
        for k in sorted(self.keys()):
            v = self[k]
            if isinstance(v, Context):
                sctxs.append((k, v))
            else:
                if is_string(v):
                    # Check if value must be masked
                    if k in ("key", "secret"):
                        v = "**********" + v[-4:]
                    vstr = '"' + v + '"'
                else:
                    vstr = str(v)
                out.write(indent + str(k) + " = " + vstr + "\n")
        # Print sub-contexts
        for (k, v) in sctxs:
            out.write(indent + str(k) + ' =\n')
            v.write(out, indent + " : ")
        out.flush()


    def export_flat(self, out=None):
        """ Export this context in flat format

        Each context attribute is written on a single line <path>=<value> with UTF-8 encoding.

        Args:
            out (Optional):  Target output stream or file name. If not given, default value is sys.stdout.
        """
        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self.export(f)
                return
        # Check default output
        if out is None:
            out = sys.stdout

        # Build dictionary of all attributes by pathes
        adict = {}
        self._build_path_dict('', adict)

        # Print values
        for k in sorted(adict.keys()):
            out.write(k)
            out.write(" = ")
            v = adict[k]
            if is_string(v):
                out.write(to_printable_string(v))
            else:
                out.write(str(v))
            out.write('\n')


    def export_flat_as_string(self):
        """ Export this context in flat format as a string.

        Each context attribute is written on a single line <path>=<value>

        Args:
            out (Optional):  Target output stream or file name. If not given, default value is sys.stdout.
        """
        out = StringIO()
        self.export_flat(out)
        res = out.getvalue()
        out.close()
        return res


    def import_flat(self, inp=None):
        """ Import a flat file in this context

        Each context attribute is added in this context.
        Each value is converted in the most appropriate type: string, integer, float, boolean or None.
        Only atomic values are allowed, not lists or other Python complex objects.

        Args:
            inp:  Input stream or file
        """
        # Check file
        if is_string(inp):
            with open_utf8(os.path.abspath(inp), mode='r') as f:
                self.import_flat(f)
                return

        # Read all lines
        for line in inp.readlines():
            # Check empty and comment lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Get attribute path
            cx = line.find('=')
            if cx < 0:
                raise Exception("No equal sign found in line '{}'".format(line))
            k = line[:cx].strip()
            v = line[cx+1:].strip()
            v = string_to_value(v)
            self.set_by_path(k, v)


    def _build_path_dict(self, path, result):
        """ Build dictionary of all context values with pathes as keys

        Args:
            path:  Path of this context
            result: Result dictionary to fill
        """
        # Build path prefix
        if path:
            path = path + "."
        for k in sorted(self.keys()):
            v = self[k]
            if isinstance(v, Context):
                v._build_path_dict(path + k, result)
            else:
                result[path + k] = v


class IdAllocator(object):
    """ Allocator of identifiers.

    This implementation is not thread-safe.
    Use SafeIdAllocator for a usage in a multi-thread environment.
    """
    __slots__ = ('prefix',  # Id prefix
                 'count',   # Allocated id count
                 'bdgts',   # Count printing base digits
                 )
    DIGITS = "0123456789"
    LETTERS_AND_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    def __init__(self, prefix, bdgts="0123456789"):
        """ Create a new id allocator.

        Args:
            prefix:  Prefix of all ids
            bdgts:   List of digit characters to be use for counter conversion
        """
        super(IdAllocator, self).__init__()
        self.prefix = prefix
        self.count = 0
        self.bdgts = bdgts

    def get_prefix(self):
        """ Get the name prefix used by this allocator.

        Returns:
            Name prefix
        """
        return self.prefix

    def get_count(self):
        """ Get the number of id that has been allocated by this allocator.

        Returns:
            Number of id that has been allocated by this allocator.
        """
        return self.count

    def allocate(self):
        """ Allocate a new id.

        Returns:
            Next id for this allocator
        """
        self.count += 1
        cnt = self.count
        res = []
        bdgts = self.bdgts
        blen = len(bdgts)
        while cnt > 0:
           res.append(bdgts[cnt % blen])
           cnt //= blen
        res.reverse()
        return self.prefix + ''.join(res)


class SafeIdAllocator(object):
    """ Allocator of identifiers.

    This implementation uses a lock to protect the increment of the counter,
    allowing to use it by multiple threads.
    """
    __slots__ = ('prefix',  # Id prefix
                 'count',   # Allocated id count
                 'bdgts',   # Count printing base digits
                 'lock',    # Lock to protect counter
                 )
    def __init__(self, prefix, bdgts="0123456789"):
        """ Create a new id allocator.

        Args:
            prefix:  Prefix of all ids
            bdgts:   (Optional) List of digit characters to be use for counter conversion
        """
        super(SafeIdAllocator, self).__init__()
        self.prefix = prefix
        self.count = 0
        self.bdgts = bdgts
        self.lock = threading.Lock()

    def get_prefix(self):
        """ Get the name prefix used by this allocator.

        Returns:
            Name prefix
        """
        return self.prefix

    def get_count(self):
        """ Get the number of id that has been allocated by this allocator.

        Returns:
            Number of id that has been allocated by this allocator.
        """
        return self.count

    def allocate(self):
        """ Allocate a new id

        Returns:
            Next id for this allocator
        """
        with self.lock:
            self.count += 1
            cnt = self.count
        res = []
        bdgts = self.bdgts
        blen = len(bdgts)
        while cnt > 0:
           res.append(bdgts[cnt % blen])
           cnt //= blen
        res.reverse()
        return self.prefix + ''.join(res)

    def reset(self):
        """ Reset the allocator
        """
        self.count = 0


class KeyIdDict(object):
    """ Dictionary using id of the key objects as key.

    This object allows to use any Python object as key (with no __hash__() function),
    and to map a value on the physical instance of the key.
    """
    __slots__ = ('kdict',  # Dictionary of objects
                 )

    def __init__(self):
        super(KeyIdDict, self).__init__()
        self.kdict = {}

    def set(self, key, value):
        """ Set a value in the dictionary.

        Args:
            key:   Key
            value: Value
        """
        kid = id(key)
        # Store value and original key, to not garbage it and preserve its id
        self.kdict[kid] = (key, value)

    def get(self, key, default=None):
        """ Get a value from the dictionary.

        Args:
            key:     Key
            default: Default value if not found. Default is None.
        Returns:
            Value corresponding to the key, default value (None) if not found
        """
        kid = id(key)
        v = self.kdict.get(kid, None)
        return default if v is None else v[1]

    def keys(self):
        """ Get the list of all keys """
        return [k for (k, v) in self.kdict.values()]

    def values(self):
        """ Get the list of all values """
        return [v for (k, v) in self.kdict.values()]

    def clear(self):
        """ Clear all dictionary content """
        self.kdict.clear()

    def __len__(self):
        """ Returns the number of elements in this dictionary """
        return len(self.kdict)


class ObjectCache(object):
    """ Limited size object cache.

    This object allows to associate an object to a key.
    This cache is limited in size. This means that, if the max size is reached, adding a new
    object removes the oldest.
    """
    __slots__ = ('obj_dict',  # Dictionary of objects
                 'max_size',  # Max cache size
                 'key_list',  # Ordered list of objects keys in the cache
                 )

    def __init__(self, maxsize):
        super(ObjectCache, self).__init__()
        self.obj_dict = {}
        self.max_size = maxsize
        self.key_list = deque()

    def set(self, key, value):
        """ Set a value in the cache

        Args:
            key:   Key
            value: Value
        """
        # Check if key already in cache
        if key in self.obj_dict:
            # Just replace the value
            self.obj_dict[key] = value
        else:
            # Remove older object if max size is reached
            if len(self.key_list) >= self.max_size:
                self.obj_dict.pop(self.key_list.popleft())
            # Store new object
            self.obj_dict[key] = value
            self.key_list.append(key)

    def get(self, key):
        """ Get a value from the cache

        Args:
            key:  Key
        Returns:
            Value corresponding to the key, None if not found
        """
        return self.obj_dict.get(key)

    def keys(self):
        """ Get the list of all keys """
        return list(self.key_list)

    def values(self):
        """ Get the list of all values """
        return list(self.obj_dict.values())

    def size(self):
        """ Get the current size of the cache """
        return len(self.key_list)

    def clear(self):
        """ Clear all dictionary content """
        self.obj_dict.clear()
        self.key_list.clear()

    def __len__(self):
        """ Returns the number of elements in this dictionary """
        return len(self.obj_dict)


class ObjectCacheById(object):
    """ Object cache that uses object id as key.

    This object allows to associate an object to a key.
    It is implemented using a dict that uses the id of the key as dict key.
    This allows to:
     * Use any Python object as key, even if it does not implement __hash__() function,
     * Use different key objects that are logically equal

    This cache is limited in size. This means that, if the max size is reached, adding a new
    object will remove the oldest.
    """
    __slots__ = ('obj_dict',  # Dictionary of objects
                 'max_size',  # Max cache size
                 'key_list',  # Ordered list of objects keys in the cache
                 )

    def __init__(self, maxsize):
        super(ObjectCacheById, self).__init__()
        self.obj_dict = {}
        self.max_size = maxsize
        self.key_list = deque()

    def set(self, key, value):
        """ Set a value in the cache.

        Args:
            key:   Key
            value: Value
        """
        kid = id(key)
        # Check if key already in cache
        if kid in self.obj_dict:
            # Just replace the value
            self.obj_dict[kid] = value
        else:
            # Remove older object if max size is reached
            if len(self.key_list) == self.max_size:
                self.obj_dict.pop(id(self.key_list.popleft()))
            # Store new object
            self.obj_dict[kid] = value
            # Append key in deque (side effect is that key is preserved to preserve its id)
            self.key_list.append(key)

    def get(self, key):
        """ Get a value from the cache.

        Args:
            key:  Key
        Returns:
            Value corresponding to the key, None if not found
        """
        return self.obj_dict.get(id(key))

    def keys(self):
        """ Get the list of all keys """
        return list(self.key_list)

    def values(self):
        """ Get the list of all values """
        return list(self.obj_dict.values())

    def clear(self):
        """ Clear all dictionary content """
        self.obj_dict.clear()
        self.key_list.clear()

    def __len__(self):
        """ Returns the number of elements in this dictionary """
        return len(self.obj_dict)


class IdentityAccessor(object):
    """ Object implementing a __getitem__ that returns the key as value. """
    def __getitem__(self, key):
        return key


class TextFileLineReader(object):
    """ Reader for text files, possibly compressed using gzip """
    __slots__ = ('input',     # Input stream
                 'encoding',  # Character encoding
                 )

    def __init__(self, file, encoding='utf-8-sig'):
        """ Create a text file line reader.

        Args:
            file:      File name
            encoding:  (Optional) Character encoding, UTF-8 by default
        """
        ext = os.path.splitext(file)[1].lower()
        if ext == '.gz':
            self.input = gzip.open(file, 'r')
        elif ext == ".zip":
            zfile = zipfile.ZipFile(file)
            lfiles = zfile.infolist()
            if len(lfiles) != 1:
                raise IOError("Zip file does not contains a single entry")
            self.input = zfile.open(lfiles[0])
        else:
            self.input = io.open(file, mode='r', encoding=encoding, errors='ignore')
        self.encoding = encoding

    def __del__(self):
        """ Release resources. """
        self.close()

    def close(self):
        """ Close this reader. """
        if self.input is not None:
            self.input.close()
            self.input = None

    def readline(self):
        """ Read next line of text.

        Returns:
            Next line of text, including ending end of line character, empty line if end of file
        """
        # Check input already closed
        if self.input is None:
            return ''
        line = self.input.readline()
        # Process case where read is done in a compressed file
        if isinstance(line, (bytes, bytearray)):
            line = line.decode(self.encoding)
        if not line:
            self.input.close()
            self.input = None
        return line


class Chrono(object):
    """ Chronometer """
    __slots__ = ('startTime',  # Chrono start time
                 )
    def __init__(self):
        """ Create a new chronometer initialized with current time.
        """
        super(Chrono, self).__init__()
        self.restart()

    def get_start(self):
        """ Get the chrono start time.

        Returns:
            Time when chronometer has been started
        """
        return self.startTime

    def get_elapsed(self):
        """ Get the chrono elapsed time.

        Returns:
            Time spent from chronometer start time (float), in seconds
        """
        return time.time() - self.startTime

    def restart(self):
        """ Restart chrono to current time
        """
        self.startTime = time.time()

    def __str__(self):
        """ Convert this chronometer into a string.

        Returns:
            String of the chrono elapsed time
        """
        return str(self.get_elapsed())


class Barrier(object):
    """ Barrier blocking multiple threads.

    This class implements a simple barrier with no timeout.
    Implemented here because not available in Python 2
    """
    __slots__ = ('parties',  # Number of parties required before unlocking the barrier
                 'count',    # Number of waiting parties
                 'lock',     # Counters protection lock
                 'barrier'   # Threads blocking lock
                 )
    def __init__(self, parties):
        """ Create a new barrier

        Args:
            parties:  Number of parties required before unlocking the barrier
        """
        self.parties = parties
        self.count = 0
        self.lock = threading.Lock()
        self.barrier = threading.Lock()
        self.barrier.acquire()

    def wait(self):
        """ Wait for the barrier
        This method blocks the calling thread until required number of threads has called this method.
        """
        with self.lock:
            self.count += 1
        if self.count < self.parties:
           self.barrier.acquire()
        self.barrier.release()


class FunctionCache(object):
    """ Object caching the result of a function.
    Future calls with same parameters returns a result stored in a cache dictionary.

    This object is not thread-safe.
    """
    __slots__ = ('values',  # Function returned values. Key is the parameters passed to the function,
                            # value is function result if function has already been called.
                 'lock',    # Lock protecting values dictionary
                 'funct',   # Function whose results is cached.
                 )
    def __init__(self, f):
        """
        Args:
            f:  Function whose results should be cached.
        """
        self.values = {}
        self.funct = f
        self.lock = threading.Lock()

    def get(self, *param):
        """ Get the function result for a given parameter
        Args:
            param:  Function parameters
        Returns:
            Function result
        """
        res = self.values.get(param)
        if res is None:
            # First check if None is the actual result
            if param in self.values:
                return None
            # Call function to get actual result
            res = self.funct(*param)
            # Add result to the cache
            self.values[param] = res
        return res

    def clear(self):
        """ Clear all dictionary content """
        self.values.clear()

    def __len__(self):
        """ Returns the number of elements in this cache """
        return len(self.values)


class InfoDict(dict):
    """ Dictionary used to store information of various types.
    """

    def incr(self, key, val):
        """ Increment the value of an attribute considered as a number.

        If the attribute is not already in the dictionary, it is set to the given increment value.

        Args:
            key:  Attribute key
            val:  Value to add
        """
        self[key] = self.get(key, 0) + val


    def clone(self):
        """ Clone this information dictionary

        Returns:
            New information dictionary, cloned copy of this one.
        """
        res = InfoDict()
        res.update(self)
        return res


    def print_infos(self, out=None, indent=""):
        """ Print this information structure.

        Attributes are sorted in alphabetical order

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        DEPRECATED. Use :meth:`write` instead.

        Args:
            out (Optional):    Target output stream or file name. If not given, default value is sys.stdout.
            indent (Optional): Start line indentation. Default is empty
        """
        self.write(out, indent)


    def write(self, out=None, indent=""):
        """ Write this information structure.

        Attributes are sorted in alphabetical order

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out (Optional):    Target output stream or file name. If not given, default value is sys.stdout.
            indent (Optional): Start line indentation. Default is empty
        """
        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self.write(f)
                return
        # Check default output
        if out is None:
            out = sys.stdout

        # Get sorted list of keys
        keys = sorted(list(self.keys()))
        if keys:
            # Print attributes
            mxlen = max(len(k) for k in keys)
            for k in keys:
                out.write(indent + k + (" " * (mxlen - len(k))) + " : " + str((self.get(k))) + "\n")
        else:
            out.write(indent + "none\n")
        out.flush()


class ListDict(dict):
    """ Dictionary specialized to associate lists to the keys.

    This class is a dictionary that enables to associate to each key a list of values.
    When adding a value, key and list are created if not present, otherwise element is added to the existing list.
    """

    def append(self, key, val):
        """ append a value to the list associated to a key.

        Args:
            key:  Attribute key
            val:  Value to add to this key list
        """
        l = self.get(key)
        if l is None:
            self[key] = [val]
        else:
            l.append(val)



###############################################################################
## Public functions
###############################################################################

def check_default(val, default):
    """ Check that an argument value is DEFAULT and returns the default value if so.

    This method has to be used in conjunction with usage of the DEFAULT constant as
    default value of a parameter. It allows to assign a parameter to a default value
    that can be computed dynamically.

    Args:
        val:     Value to check
        default: Default value to return if val is DEFAULT
    Returns:
        val if val is different from DEFAULT, default otherwise
    """
    return default if (val is DEFAULT) else val


def replace_in_tuple(tpl, ndx, val):
    """ Replace a value in a tuple.

    A new tuple is created with one value changed by another.

    Args:
        tpl:     Source tuple
        ndx:  Index of the tuple value to change
        val:  new value
    Returns:
        New tuple with value at position 'ndx' replaced by 'val'
    """
    lst = list(tpl)
    lst[ndx] = val
    return tuple(lst)


def replace(obj, rfun):
    """ Build a copy of a source object and apply a replacement function on all its attributes.

    If an object is not changed, its original version is returned.

    Args:
       obj:  Source object
       rfun: Value replacement function
    Returns:
        Copy of source object with all values replaced, or same object if no value has been replaced.
    """
    # Check if object is immediately replaceable
    try:
        nobj = rfun(obj)
        # Check case where replace function force to keep same object
        if nobj is obj:
            return obj
    except:
        nobj = obj

    # Check basic types (also eliminates strings)
    ot = type(nobj)
    if ot in BASIC_TYPES:
        return nobj

    # Duplicate object a priori
    changed = False
    nobj = copy.copy(obj)

    # Check if object is a tuple (particular case, can not process case of general object extending tuple)
    if isinstance(obj, tuple):
        nvals = tuple(replace(x, rfun) for x in obj)
        changed = any(nx is not x for x, nx in zip(obj, nvals))
        if changed:
            # Particular case of named tuples
            if hasattr(obj, '_fields'):
                nobj = ot(*nvals)
            else:
                nobj = ot(nvals)

    # Check if object is dictionary
    elif isinstance(obj, dict):
        nobj.clear()
        for k, v in obj.items():
            nk = replace(k, rfun)
            nv = replace(v, rfun)
            nobj[nk] = nv
            changed |= (nk is not k) or (nv is not v)

    # Check if object is list
    elif isinstance(obj, list):
        for i in range(len(obj)):
            v = obj[i]
            nv = replace(v, rfun)
            nobj[i] = nv
            changed |= (nv is not v)

    # Replace in attributes if any (even in objects extending basic composed types)
    if hasattr(obj, '__dict__'):
        for atr in obj.__dict__:
            if not atr.startswith('_'):
                try:
                    v = getattr(obj, atr)
                    nv = replace(v, rfun)
                    setattr(nobj, atr, nv)
                    changed |= (nv is not v)
                except:
                    pass

    # Replace in slots attributes if any (even in objects extending basic composed types)
    if hasattr(obj, '__slots__'):
        for atr in obj.__slots__:
            if not atr.startswith('_'):
                v = getattr(obj, atr)
                nv = replace(v, rfun)
                setattr(nobj, atr, nv)
                changed |= (nv is not v)

    # Return object
    return nobj if changed else obj


def assert_arg_int_interval(val, mn, mx, name=None):
    """ Check that an argument is an integer in a given interval.

    Args:
        val:  Argument value
        mn:   Minimal possible value (included)
        mx:   Maximal possible value (excluded)
        name: Name of the parameter (optional), used in raised exception.
    Raises:
      TypeError exception if wrong argument type
    """
    assert is_int(val) and (val >= mn) and (val < mx), \
           "Argument '" + name + "' should be an integer in [" + str(mn) + ".." + str(mx) + ")"


def to_string(val):
    """ Convert a value into a string, recursively for lists and tuples.

    Args:
        val: Value to convert value
    Returns:
        String representation of the value
    """
    # Check tuple
    if isinstance(val, tuple):
        if len(val) == 1:
            return "(" + to_string(val[0]) + ",)"
        return "(" + ", ".join(map(to_string, val)) + ")"

    # Check list
    if isinstance(val, list):
        return "[" + ", ".join(map(to_string, val)) + "]"

    # Default
    return str(val)


_DEFAULT_STRING_VALUES = {"none": None, "true": True, "false": False}
def string_to_value(s):
    """ Convert a string in its most representative python value

    If the string is encapsulated in double_quotes, the internal string is returned.
    If string in lower case is "none", None is returned.
    If string in lower case is "true" or "false", True or False is returned.
    If string matches an integer, the integer is returned.
    If the string matches a float, the float is returned
    Otherwise, string is returned as it is

    Args:
        s: String to convert
    Returns:
        Value corresponding to the string
    """
    # Check first character
    c = s[0]
    ls = len(s)
    if c == '"' and ls > 1 and s[-1] == '"':
        return s[1:ls-1]
    if c.isdigit() or (c == '-' and ls >= 2 and s[1].isdigit()):
        # Try to convert as number
        try:
            # Check if possible float
            if s.find('.') >= 0:
                s = float(s)
            else:
                s = int(s)
        except:
            pass
    else:
        ns = s.lower()
        if ns in _DEFAULT_STRING_VALUES:
            return _DEFAULT_STRING_VALUES[ns]
    return s


def _get_vars(obj):
    """ Get the list variable names of an object.
    """
    # Check if a dictionary is present
    if hasattr(obj, '__dict__'):
        res = getattr(obj, '__dict__').keys()
    # Check if slot is defined
    elif hasattr(obj, '__slots__'):
        slts = getattr(obj, '__slots__')
        if is_array(slts):
            res = list(slts)
        else:
            res = [slts]
        # Go upper in the class hierarchy
        obj = super(obj.__class__, obj)
        while hasattr(obj, '__slots__'):
            slts = getattr(obj, '__slots__')
            if is_array(slts):
                res.extend(slts)
            else:
                res.append(slts)
            obj = super(obj.__class__, obj)
        return res
    # No attributes
    else:
        res = ()
    return sorted(res)


def _equals_lists(l1, l2):
    """ Utility function for equals() to check two lists.
    """
    return (len(l1) == len(l2)) and all(equals(v1, v2) for v1, v2 in zip(l1, l2))


def equals(v1, v2):
    """ Check that two values are logically equal, i.e. with the same attributes with the same values, recursively.

    This method does not call __eq__ except on basic types, and is then proof to possible overloads of '=='

    Args:
       v1: First value
       v2: Second value
    Returns:
        True if both values are identical, false otherwise
    """
    # Check same object (also covers some primitive types as int, float and strings, but not guarantee)
    if v1 is v2:
        return True

    # Check same type
    t = type(v1)
    if not (t is type(v2)):
        if is_string(v1) and is_string(v2):
            return v1 == v2
        return False

    # Check basic types
    if t in BASIC_TYPES:
        return v1 == v2

    # Check list or tuple
    if isinstance(v1, (list, tuple, bytes, bytearray)):
        return _equals_lists(v1, v2)

    # Check dictionary
    if isinstance(v1, dict):
        # Compare keys
        k1 = sorted(tuple(v1.keys()), key=lambda x: str(x))
        k2 = sorted(tuple(v2.keys()), key=lambda x: str(x))
        if not _equals_lists(k1, k2):
            return False
        # Compare values
        for k in k1:
            if not equals(v1.get(k), v2.get(k)):
                return False
        return True

    # Check sets
    if isinstance(v1, (set, frozenset)):
        # Compare values
        return _equals_lists(sorted(tuple(v1)), sorted(tuple(v2)))

    # Compare object attributes
    dv1 = _get_vars(v1)
    if not _equals_lists(dv1, _get_vars(v2)):
        return False
    for k in dv1:
        if not equals(getattr(v1, k, None), getattr(v2, k, None)):
            return False
    return True


def make_directories(path):
    """ Ensure a directory path exists.

    Args:
        path: Directory path to check or create
    Raises:
        Any IO exception if directory creation is not possible
    """
    if (path != "") and (not os.path.isdir(path)):
        os.makedirs(path)


def create_stdout_logger(name):
    """ Create a default logger on stdout with default formatter printing time at the beginning.
        of the line.

    Args:
        name:  Name of the logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s  %(message)s'))
    logger.addHandler(ch)
    return logger


def get_file_name_only(file):
    """ Get the name of a file, without directory or extension.
    Args:
        file:  File name
    Returns:
        Name of the file, without directory or extension
    """
    return os.path.splitext(os.path.basename(file))[0]


def read_string_file(file):
    """ Read a file as a string.
    Args:
        file:  File name
    Returns:
        File content as a string
    """
    with open(file, "r") as f:
        str = f.read()
    return str


def write_string_file(file, str):
    """ Write a string into a file.
    Args:
        file:  File name
        str:   String to write
    """
    with open(file, "w") as f:
        f.write(str)


def format_text(txt, size):
    """ Format a given text in multiple lines.
    Args:
        txt:  Text to format
        size: Line size
    Returns:
        List of lines.
    """
    res = []
    sepchars = ' \t\n\r'
    txt = txt.strip(sepchars)
    while len(txt) > size:
        # Search end of line
        enx = size
        while (enx > 0) and (txt[enx] not in sepchars):
            enx -= 1
        # Check no separator in the line
        if enx == 0:
            enx = size
        # Check for a end of line in the line
        x = txt.find('\n', 0, enx)
        if x > 0:
            enx = x
        # Append line
        res.append(txt[:enx])
        # Remove line from source
        txt = txt[enx:].strip(sepchars)
    # Add last line
    if txt != "":
        res.append(txt)
        return res


def encode_integer_big_endian_4(value, frame, start):
    """ Encode an integer in a byte array using big endian on 4 bytes.
    Args:
        value:  Integer value to encode
        frame:   Target byte array
        start:  Write start index
    """
    frame[start]     = (value >> 24) & 0xFF
    frame[start + 1] = (value >> 16) & 0xFF
    frame[start + 2] = (value >> 8)  & 0xFF
    frame[start + 3] = value         & 0xFF


def decode_integer_big_endian_4(frame, start):
    """ Encode an integer in a byte array using big endian on 4 bytes.
    Args:
        frame:  Source byte array
        start:  Read start index
    Returns:
        Decoded integer value
    """
    return (frame[start] << 24) | (frame[start + 1] << 16) | (frame[start + 2] << 8) | frame[start + 3]


def open_utf8(file, mode='r'):
    """ Open a stream for read or write with UTF-8 encoding.

    Args:
        file:  File to open
        mode:  Open mode
    """
    encd = 'utf-8-sig' if mode.startswith('r') else 'utf-8'
    return io.open(file, mode=mode, encoding=encd, errors='ignore')


def list_module_public_functions(mod, excepted=()):
    """ Build the list of all public functions of a module.

    Args:
        mod:       Module to parse
        excepted:  List of function names to not include. Default is none.
    Returns:
        List of public functions declared in this module
    """
    return [t[1] for t in inspect.getmembers(mod, inspect.isfunction) if not t[0].startswith('_')
            and inspect.getmodule(t[1]) == mod
            and not t[0] in excepted]


def get_module_element_from_path(opath):
    """ Retrieve an element from a python module using its path <package>.<module>.<object>.

    This can be for example a global function or a class.

    Args:
        opath:  Object path
    Returns:
        Module object
    """
    module, oname = opath.rsplit('.', 1)
    mod = importlib.import_module(module)
    return getattr(mod, oname)


#-----------------------------------------------------------------------------
# Checking of object types
#-----------------------------------------------------------------------------

# Determine list of types representing the different python scalar types
BOOL_TYPES    = {bool}
INTEGER_TYPES = {int}
FLOAT_TYPES   = {float}
STRING_TYPES  = {str}

# Add Python2 specific unicode if any
if IS_PYTHON_2:
    INTEGER_TYPES.add(long)
    STRING_TYPES.add(unicode)

# Add numpy types if any
if IS_NUMPY_AVAILABLE:
    BOOL_TYPES.add(numpy.bool_)
    BOOL_TYPES.add(numpy.bool)

    INTEGER_TYPES.add(numpy.int_)
    INTEGER_TYPES.add(numpy.intc)
    INTEGER_TYPES.add(numpy.intp)

    INTEGER_TYPES.add(numpy.int8)
    INTEGER_TYPES.add(numpy.int16)
    INTEGER_TYPES.add(numpy.int32)
    INTEGER_TYPES.add(numpy.int64)

    INTEGER_TYPES.add(numpy.uint8)
    INTEGER_TYPES.add(numpy.uint16)
    INTEGER_TYPES.add(numpy.uint32)
    INTEGER_TYPES.add(numpy.uint64)

    FLOAT_TYPES.add(numpy.float_)
    FLOAT_TYPES.add(numpy.float16)
    FLOAT_TYPES.add(numpy.float32)
    FLOAT_TYPES.add(numpy.float64)

# Build all number type sets
INTEGER_TYPES = frozenset(INTEGER_TYPES)
FLOAT_TYPES   = frozenset(FLOAT_TYPES)
BOOL_TYPES    = frozenset(BOOL_TYPES)
NUMBER_TYPES  = frozenset(INTEGER_TYPES.union(FLOAT_TYPES))
STRING_TYPES  = frozenset(STRING_TYPES)
BASIC_TYPES   = frozenset(NUMBER_TYPES.union(BOOL_TYPES).union(STRING_TYPES))


def is_bool(val):
    """ Check if a value is a boolean, including numpy variants if any.

    Args:
        val: Value to check
    Returns:
        True if value is a boolean.
    """
    return type(val) in BOOL_TYPES

if IS_NUMPY_AVAILABLE:

    def is_int(val):
        """ Check if a value is an integer, including numpy variants if any.

        Args:
            val: Value to check
        Returns:
            True if value is an integer.
        """
        return (type(val) in INTEGER_TYPES) or numpy.issubdtype(type(val), numpy.integer)


    def is_float(val):
        """ Check if a value is a float, including numpy variants if any.

        Args:
            val: Value to check
        Returns:
            True if value is a float
        """
        return (type(val) in FLOAT_TYPES) or numpy.issubdtype(type(val), numpy.float)


    def is_number(val):
        """ Check if a value is a number, including numpy variants if any.

        Args:
            val: Value to check
        Returns:
            True if value is a number
        """
        return (type(val) in NUMBER_TYPES) or numpy.issubdtype(type(val), numpy.number)


    def is_array(val):
        """ Check if a value is an array (list or tuple).

        Args:
            val: Value to check
        Returns:
            True if value is an array (list or tuple)
        """
        return isinstance(val, (list, tuple)) or \
               (isinstance(val, numpy.ndarray) and val.shape)

else:

    def is_int(val):
        """ Check if a value is an integer, including numpy variants if any.

        Args:
            val: Value to check
        Returns:
            True if value is an integer.
        """
        return type(val) in INTEGER_TYPES


    def is_float(val):
        """ Check if a value is a float, including numpy variants if any.

        Args:
            val: Value to check
        Returns:
            True if value is a float
        """
        return type(val) in FLOAT_TYPES


    def is_number(val):
        """ Check if a value is a number, including numpy variants if any.

        Args:
            val: Value to check
        Returns:
            True if value is a number
        """
        return type(val) in NUMBER_TYPES


    def is_array(val):
        """ Check if a value is an array (list or tuple).

        Args:
            val: Value to check
        Returns:
            True if value is an array (list or tuple)
        """
        return isinstance(val, (list, tuple))


if IS_PANDA_AVAILABLE:

    def is_panda_series(val):
        """ Check if a value is a panda serie.

        Args:
            val: Value to check
        Returns:
            True if value is an panda serie
        """
        return isinstance(val, PandaSeries)

else:

    def is_panda_series(val):
        """ Check if a value is a panda serie.

        Args:
            val: Value to check
        Returns:
            False
        """
        return False


def is_int_value(val, xv):
    """ Check if a value is an integer equal to an expected value.
    Allows to check integer value equality even if == operator is overloaded

    Args:
        val:  Value to check
        xv:   Expected integer value
    Returns:
        True if value is an integer.
    """
    return (type(val) is int) and (val == xv)


def is_string(val):
    """ Check if a value is a string or a variant.

    Args:
        val: Value to check
    Returns:
        True if value is a string
    """
    return type(val) in STRING_TYPES


def is_iterable(val):
    """ Check if a value is iterable, but not a string.

    Args:
        val: Value to check
    Returns:
        True if value is iterable
    """
    return isinstance(val, Iterable) and not is_string(val)


def is_int_array(val):
    """ Check that a value is an array of integers.

    Args:
        val: Value to check
    Returns:
        True if value is an array where all elements are integers
    """
    return is_array(val) and (all(is_int(x) for x in val))


def is_array_of_type(val, typ):
    """ Check that a value is an array with all elements instances of a given type.

    Args:
        val: Value to check
        typ: Expected element type
    Returns:
        True if value is an array with all elements with expected type
    """
    return is_array(val) and (all(isinstance(x, typ) for x in val))


def is_in(val, lvals):
    """ Replace the standard 'in' operator but uses 'is' to check membership.

    This method is mandatory if element to check overloads operator '=='.

    Args:
        val:   Value to check
        lvals: List of candidate values
    Returns:
        True if value is in the list of values (presence checked with 'is')
    """
    return any(val is v for v in lvals)


#-----------------------------------------------------------------------------
# String conversion functions
#-----------------------------------------------------------------------------

# Dictionary of special characters conversion
_FROM_SPECIAL_CHARS = {'n': "\n", 't': "\t", 'r': "\r", 'f': "\f", 'b': "\b", '\\': "\\", '"': "\""}

# Dictionary of special characters conversion
_TO_SPECIAL_CHARS = {'\n': "\\n", '\t': "\\t", '\r': "\\r", '\f': "\\f", '\b': "\\b", '\\': "\\\\", '\"': "\\\""}

# Set of symbol characters
_SYMBOL_CHARS = frozenset(x for x in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

# Set of digit characters
_DIGIT_CHARS = frozenset(x for x in "0123456789")


def is_symbol_char(c):
    """ Check whether a character can be used in a symbol.

    Args:
        c: Character
    Returns:
        True if character in 0..9, a..z, A..Z, _ or .
    """
    # return ((c >= 'a') and (c <= 'z')) or ((c >= 'A') and (c <= 'Z')) or ((c >= '0') and (c <= '9')) or (c == '_')
    # Following is 25% faster
    return c in _SYMBOL_CHARS


def to_printable_id(s):
    """ Build a CPO printable identifier from its raw string (add escape sequences and quotes if necessary).

    Args:
        s: Identifier string
    Returns:
        Unicode CPO identifier string, including double quotes and escape sequences if needed if not only chars and integers
    """
    # Check empty string
    if len(s) == 0:
        return u'""'
    # Check if string can be used as it is
    if (all((c in _SYMBOL_CHARS) for c in s)) and not s[0] in _DIGIT_CHARS:
        return make_unicode(s)
    # Build result string
    return u'"' + ''.join(_TO_SPECIAL_CHARS.get(c, c) for c in s) + u'"'


def to_printable_string(s):
    """ Build a printable string from raw string (add escape sequences and quotes if necessary).

    Args:
        str: String to convert
    Returns:
        Unicode string, including always double quotes and escape sequences if needed
    """
    # Check empty string
    if len(s) == 0:
        return u'""'
    # Check if string can be used as it is
    if (all((c in _SYMBOL_CHARS) for c in s)) and not s[0] in _DIGIT_CHARS:
        return u'"' + make_unicode(s) + u'"'
    # Build result string
    return u'"' + ''.join(_TO_SPECIAL_CHARS.get(c, c) for c in s) + u'"'


def to_internal_string(strg):
    """ Convert a string (without enclosing quotes) into internal string (interpret escape sequences).

    Args:
        strg: String to convert
    Returns:
        Raw string corresponding to source
    """
    res = []
    beg = 0,
    end = len(strg)
    while beg < end:
        c = strg[beg]
        if c == '\\':
            beg += 1
            c = _FROM_SPECIAL_CHARS.get(strg[beg], None)
            if c is None:
                raise SyntaxError("Unknown special character '\\" + strg[beg] + "'")
        res.append(c)
        beg += 1
    return u''.join(res)


def to_internal_string(strg):
    """ Convert a string (without enclosing quotes) into internal string (interpret escape sequences).

    Args:
        strg: String to convert
    Returns:
        Raw string corresponding to source
    """
    res = []
    i = 0
    slen = len(strg)
    while i < slen:
        c = strg[i]
        if c == '\\':
            i += 1
            c = _FROM_SPECIAL_CHARS.get(strg[i], None)
            if c is None:
                raise SyntaxError("Unknown special character '\\" + strg[i] + "'")
        res.append(c)
        i += 1
    return u''.join(res)


if IS_PYTHON_2:
    def make_unicode(s):
        """ Convert a string in unicode.

        Args:
            s: String to convert
        Returns:
            String in unicode
        """
        return s if type(s) is unicode else unicode(s)
else:
    def make_unicode(s):
        """ Convert a string in unicode.

        Args:
            s: String to convert
        Returns:
            String in unicode
        """
        return s


def int_to_base(val, bdgts):
    """ Convert an integer into a string with a given base.

    Args:
        val:   Integer value to convert
        bdgts: List of base digits
    Returns:
        String corresponding to the integer
    """
    # Check zero
    if val == 0:
        return bdgts[0]
    # Check negative number
    if val < 0:
        isneg = True
        val = -val
    else:
        isneg = False
    # Fill list of digits
    res = []
    blen = len(bdgts)
    while val > 0:
        res.append(bdgts[val % blen])
        val //= blen
    # Add negative sign if necessary
    if isneg:
        res.append('-')
    # Return
    res.reverse()
    return ''.join(res)


def is_exe_file(f):
    """ Check that a file exists and is executable.

    Args:
        f:  File name
    Returns:
        True if file exists and is executable
    """
    return os.path.isfile(f) and os.access(f, os.X_OK)


def get_system_path():
    """ Get the system path as a list of files

    Returns:
        List of names in the system path
    """
    path = os.getenv('PATH')
    if path:
        return path.split(os.pathsep)
    return []



def search_exec_file(f, path):
    """ Search the first occurrence of an executable file in a given path.

    Args:
        f:    Executable file name
        path: Path where search the file
    Returns:
        Full path of the first executable file found, None if not found
    """
    # Check if given file is directly executable
    if is_exe_file(f):
        return f

    # Search first executable in the path
    for d in path:
        nf = os.path.join(d, f)
        if is_exe_file(nf):
            return nf
    return None


def parse_json_string(jstr):
    """ Parse a JSON string.

    Args:
        jstr: String containing JSON document
    Returns:
        Python representation of JSON document
    """
    return json.loads(jstr, parse_constant=True)


def encode_csv_string(str):
    """ Encode a string to be used in CSV file.

    Args:
        str:  String to encode
    Returns:
        Encoded string, including starting and ending double quote
    """
    res = ['"']
    for c in str:
        res.append(c)
        if c == '"':
            res.append('"')
    res.append('"')
    return ''.join(res)


def compare_natural(s1, s2):
    """ Compare two strings in natural order (for numbers that are inside).

    Args:
        s1:  First string
        s2:  Second string
    Returns:
        integer <0 if s1 < s2, 0 if s1 == s2, >0 if s1 > s2
    """
    # Check null strings
    if s1 is None:
        return 0 if s2 is None else -1
    if s2 is None:
        return 1

    # Skip all identical characters
    len1 = len(s1)
    len2 = len(s2)
    i = 0
    while i < len1 and i < len2 and s1[i] == s2[i]:
        i += 1

    # Check end of string
    if i == len1 or i == len2:
        return len1 - len2

    # Check digit in first string
    c1 = s1[i]
    c2 = s2[i]
    if c1.isdigit():
        # Check digit in first string only
        if not c2.isdigit():
            return 1 if i > 0 and s1[i - 1].isdigit() else ord(c1) - ord(c2)

        # Scan all integer digits
        x1 = i + 1
        while x1 < len1 and s1[x1].isdigit():
            x1 += 1
        x2 = i + 1
        while x2 < len2 and s2[x2].isdigit():
            x2 += 1

        # Longer integer wins, first digit otherwise
        return ord(c1) - ord(c2) if x1 == x2 else x1 - x2

    # Check digit in second string only
    if c2.isdigit():
        return -1 if i > 0 and s2[i - 1].isdigit() else ord(c1) - ord(c2)

    # No digits
    return ord(c1) - ord(c2)


def get_module_version(mname):
    """ Get the version of a Python module

    Args:
        mname:  Module name
    Returns:
        Version of the module, None if not installed, "Unknown" if not set in the module
    """
    try:
        m = importlib.import_module(mname)
        try:
            return m.__version__
        except AttributeError:
            return "Unknown"
    except ImportError:
        return None


#-----------------------------------------------------------------------------
# Zip iterator functions to scan lists simultaneously
#-----------------------------------------------------------------------------

import itertools
if IS_PYTHON_2:
    zip = itertools.izip
    zip_longest = itertools.izip_longest
else:
    # For Python 3.
    zip = zip
    zip_longest = itertools.zip_longest


#-----------------------------------------------------------------------------
# Reload function
#-----------------------------------------------------------------------------

if IS_PYTHON_2:
    pass # reload is builtin
else:
    if sys.version_info  < (3, 4):
        import imp
        reload = imp.reload
    else:
        import importlib
        reload = importlib.reload


#-----------------------------------------------------------------------------
# Retrieve builtin functions that are overwritten
#-----------------------------------------------------------------------------

try:
    import __builtin__ as builtin  # Python 2
except ImportError:
    import builtins as builtin     # Python 3

builtin_min   = builtin.min
builtin_max   = builtin.max
builtin_sum   = builtin.sum
builtin_abs   = builtin.abs
builtin_range = builtin.range
builtin_all   = builtin.all
builtin_any   = builtin.any


#-----------------------------------------------------------------------------
# Range iterator
#-----------------------------------------------------------------------------

if IS_PYTHON_2:
    xrange = builtin.xrange
else:
    xrange = builtin.range


#-----------------------------------------------------------------------------
# Set warning filter to default (print warnings)
#-----------------------------------------------------------------------------

# import warnings
# warnings.simplefilter("default", DeprecationWarning)
# warnings.simplefilter("default", PendingDeprecationWarning)
