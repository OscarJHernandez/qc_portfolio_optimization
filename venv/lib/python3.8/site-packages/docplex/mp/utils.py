# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore
import logging
import os
import sched
import tempfile
import threading
import time
import sys

from itertools import chain, repeat
from six import PY2 as SIX_PY2
from six import itervalues, iteritems


from docplex.mp.compat23 import Queue, izip


__int_types = {int}
__float_types = {float}
__numpy_ndslot_type = None
__numpy_matrix_type = None
__pandas_series_type = None
__pandas_dataframe_type = None
__spark_dataframe_type = None

try:
    # noinspection PyUnresolvedReferences
    type(long)  # @UndefinedVariable
    # long is indeed a type we are in Python2,
    __int_types.add(long)  # @UndefinedVariable
except NameError:  # pragma: no cover
    # long is not a type, do nothing
    pass  # pragma: no cover

try:
    import numpy

    _numpy_is_available = True

    __int_types.add(numpy.bool_)
    __int_types.add(numpy.bool)

    __int_types.add(numpy.int_)
    __int_types.add(numpy.intc)
    __int_types.add(numpy.intp)

    __int_types.add(numpy.int8)
    __int_types.add(numpy.int16)
    __int_types.add(numpy.int32)
    __int_types.add(numpy.int64)

    __int_types.add(numpy.uint8)
    __int_types.add(numpy.uint16)
    __int_types.add(numpy.uint32)
    __float_types.add(numpy.uint64)

    __float_types.add(numpy.float_)
    __float_types.add(numpy.float16)
    __float_types.add(numpy.float32)
    __float_types.add(numpy.float64)

    from numpy import ndarray, matrix

    __numpy_ndslot_type = ndarray
    __numpy_matrix_type = matrix
except ImportError:  # pragma: no cover
    _numpy_is_available = False  # pragma: no cover
    numpy_is_numeric = None
    numpy_is_integer = None

try:
    from pandas import Series, DataFrame

    __pandas_series_type = Series
    __pandas_dataframe_type = DataFrame
except ImportError:
    __pandas_series_type = None
    __pandas_dataframe_type = None

# 'findspark' must be executed if running in Windows environment, before importing Spark
try:
    import findspark  # @UnresolvedImport
    findspark.init()
except (ImportError, IndexError, ValueError):
    pass

try:
    import pyspark

    __spark_dataframe_type = pyspark.sql.dataframe.DataFrame
except ImportError:
    __spark_dataframe_type = None

__int_types = frozenset(__int_types)


def is_int(s):
    type_of_s = type(s)
    return type_of_s in __int_types or (_numpy_is_available and numpy_is_integer(type(s)))


__all_python_num_types = frozenset(__float_types.union(__int_types))

if _numpy_is_available:
    def numpy_is_numeric(t):
        # returns True if the specified type is numeric
        try:
            flag = numpy.issubdtype(t, numpy.number)
            global __all_python_num_types
            if flag:
                tmp = set(__all_python_num_types)
                tmp.add(type(t))
                __all_python_num_types = frozenset(tmp)
            return flag
        except TypeError:
            return False


    def numpy_is_integer(t):
        # returns True if the specified type is integer
        try:
            flag = numpy.issubdtype(t, numpy.integer)
            if flag is True:
                global __all_python_num_types
                global __int_types
                tmp = set(__all_python_num_types)
                tmp.add(type(t))
                __all_python_num_types = frozenset(tmp)

                tmp = set(__int_types)
                tmp.add(type(t))
                __int_types = frozenset(tmp)
            return flag
        except TypeError:
            return False


    def _is_numpy_ndslot(s):
        # returns True if the argument is a numpy number
        # wrapped in a fake ndarray
        # all the following conditions must be satisfied:
        # 1. numpy is present
        # 2. type is ndarray
        # 3. shape is () empty tuple
        # 4. wrapped type in ndarray is numeric.
        try:
            retval = is_numpy_ndarray(s) and s.shape == () and \
                     (s.dtype.type in __all_python_num_types or numpy_is_numeric(s.dtype))
            return retval
        except AttributeError:  # if s is not a numpy type, s.dtype triggers this
            return False
else:
    def numpy_is_numeric(t):
        return False


    def numpy_is_integer(t):
        return False


    def _is_numpy_ndslot(s):
        return False


def is_number(s):
    type_of_s = type(s)
    return type_of_s in __all_python_num_types or numpy_is_numeric(type_of_s) or _is_numpy_ndslot(s)


# def is_number2(s):
#     if isinstance(s, __all_num_types_tuple):
#         return True
#     elif _numpy_is_available:
#         type_of_s = type(s)
#         if (numpy_is_numeric(type_of_s) or _is_numpy_ndslot(s)):
#             return True
#     else:
#         return False

def is_pandas_series(s):
    return __pandas_series_type is not None and type(s) is __pandas_series_type


def is_pandas_dataframe(s):
    return __pandas_dataframe_type and isinstance(s, __pandas_dataframe_type)


def is_numpy_ndarray(s):
    return __numpy_ndslot_type and type(s) is __numpy_ndslot_type


def is_numpy_matrix(s):
    return __numpy_matrix_type and type(s) is __numpy_matrix_type


def is_spark_dataframe(s):
    return __spark_dataframe_type and isinstance(s, __spark_dataframe_type)


string_types = {str}
if SIX_PY2:
    string_types.add(unicode)  # @UndefinedVariable
string_types = frozenset(string_types)


def is_string(e):
    return type(e) in string_types

def has_len(e):
    try:
        len(e)
        return True
    except TypeError:
        return False


def is_indexable(e):
    ''' Returns true if it is indexable
    '''
    return hasattr(e, "__getitem__")


def is_iterable(e, accept_string=True):
    ''' Returns true if we can extract an iterator from it
    '''
    try:
        iter(e)
        return accept_string or not is_string(e)
    except TypeError:
        return False


def is_iterator(e):
    ''' Returns true if e is its own iterator.
    '''
    try:
        # some numpy iterators just fail on == but are ok with "is"
        return e is iter(e)
    except TypeError:
        return False


def is_function(e):
    import platform
    if platform.python_version() >= '3.7':
        from collections.abc import Callable
    else:
        from collections import Callable  # @UnresolvedImport
    return isinstance(e, Callable)


def _to_list(arg):
    # INTERNAL:
    # 1. checks the argument is either a sequence or iterator,;
    # if sequence, returns the sequence, else converts to a list by exhsuating the iterator
    # BEWARE of the infinite generator!
    if is_iterator(arg):
        return list(arg)
    elif is_iterable(arg):
        return arg
    else:
        # an atom: wrap it into a list
        return list(arg)


def _build_ordered_sequence_types():
    if __pandas_series_type and __numpy_ndslot_type:
        return (list, tuple, __pandas_series_type, __numpy_ndslot_type)
    elif __pandas_series_type:
        return (list, tuple, __pandas_series_type)
    elif __numpy_ndslot_type:
        return (list, tuple, __numpy_ndslot_type)
    else:
        return (list, tuple)

def is_ordered_sequence1(arg, type_tuple=_build_ordered_sequence_types()):
    return isinstance(arg, type_tuple)

# def is_ordered_sequence2(arg):
#     if isinstance(arg, (dict, str)):
#         return False
#
#     try:
#         # try it
#         l = len(arg)
#         if l:
#             arg[0]
#         return True
#     except:
#         return False

is_ordered_sequence = is_ordered_sequence1


class DOcplexException(Exception):
    """ Base class for modeling exceptions
    """
    DEFAULT_MSG = 'CplexPythonModeling exception raised'

    def __init__(self, msg, *args):
        Exception.__init__(self, msg)
        self.__msg = msg or self.__class__.DEFAULT_MSG
        self.__edited_message = None
        self.__args = args
        self._resolve_message()

    def _resolve_message(self):
        self.__edited_message = None
        msg = self.__msg
        if self.__args:
            if '%' in msg:
                self.__edited_message = msg % self.__args
            elif '{' in msg:
                self.__edited_message = msg.format(*self.__args)

    @property
    def message(self):
        return self.__edited_message or self.__msg


class DOcplexLimitsExceeded(DOcplexException):
    cplexce_limits_exceeded_msg = "**** Promotional version. Problem size limits exceeded, CPLEX code=1016"

    def __init__(self):
        DOcplexException.__init__(self, self.cplexce_limits_exceeded_msg)


class DocplexQuadToLinearException(DOcplexException):

    def __init__(self, q):
        msg = "Quadratic expression [{0!s}] cannot be converted to a linear expression".format(q)
        DOcplexException.__init__(self, msg)


class DocplexLinearRelaxationError(DOcplexException):

    def __init__(self, obj, cause=None):
        self.object = obj
        self.cause = cause
        msg = "Modeling object: {0!s} cannot be relaxed".format(obj)
        DOcplexException.__init__(self, msg)

def normalize_basename(s, force_lowercase=True, maxlen=255):
    """Replaces some characters from s with a translation table:
    
     trans_table = {" ": "_",
                   "/": "_slash_",
                   "\\": "_backslash_",
                   "?": "_question_",
                   "%": "_percent_",
                   "*": "_asterisk_",
                   ":": "_colon_",
                   "|": "_bar_",
                   '"': "_quote_",
                   "<": "_lt_",
                   ">": "_gt_",
                   "&": "_amp_"}
    
    then if the generated name is longer than maxlen, the name is truncated
    to maxlen and the hash of the name modulo 0xffffffff is appended.
    """
    # replace all whietspaces by _
    l = s.lower() if force_lowercase else s
    # table = mktrans(" ", "_")
    # return l.translate(table)
    trans_table = {" ": "_",
                   "/": "_slash_",
                   "\\": "_backslash_",
                   "?": "_question_",
                   "%": "_percent_",
                   "*": "_asterisk_",
                   ":": "_colon_",
                   "|": "_bar_",
                   '"': "_quote_",
                   "<": "_lt_",
                   ">": "_gt_",
                   "&": "_amp_"}

    n = ("".join([trans_table.get(x, x) for x in l]))

    if len(n) > maxlen - 8:
        h = format(hash(n) & 0xffffffff, "08x")
        n = n[:maxlen-8] + "_"+ h
    return n

def fix_whitespace(s, fix='_'):
    # replaces whitespaces by a character, default is '_'
    return s.replace(' ', fix)


def make_output_path2(actual_name, extension, basename_fmt, path=None):
    # INTERNAL
    raw_basename = resolve_pattern(basename_fmt, actual_name) if basename_fmt else actual_name
    # fix whitespaces
    if raw_basename.find(" ") > 0:
        actual_basename = raw_basename.replace(" ", "_")
    else:
        actual_basename = raw_basename

    output_dir = path or tempfile.gettempdir()
    # fix extension
    if not actual_basename.endswith(extension):
        actual_basename = actual_basename + extension
    return os.path.join(output_dir, actual_basename)


def make_path(error_handler, basename, extension, output_dir=None, name_transformer=None):
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    elif not os.path.exists(output_dir):
        if not os.makedirs(output_dir):
            error_handler.error("directory not found and not created: {0:s}", output_dir)
            return None

    norm_name = normalize_basename(basename)
    basename = norm_name if not name_transformer else name_transformer % norm_name
    filename = basename + extension
    full_path = '/'.join([output_dir, filename])
    return full_path


def generate_constant(the_constant, count_max):
    if count_max is None:
        count_max = sys.maxsize
    loop_counter = 0
    while loop_counter <= count_max:
        yield the_constant
        loop_counter += 1


def iter_emptyset():
    return iter([])

def iter_one(obj):
    yield obj



def resolve_pattern(pattern, args):
    """
    returns a string in which slots have been resolved with args, if the string has slots anyway,
    else returns the strng itself (no copy, should we??)
    :param pattern:
    :param args:
    :return:
    """
    if args is None or len(args) == 0:
        return pattern
    elif pattern.find('%') >= 0:
        return pattern % args
    elif pattern.find("{") >= 0:
        # star magic does not work for single args
        return pattern.format(*args)
    else:
        # fixed pattern, no placeholders
        return pattern


def str_maxed(arg, maxlen):
    """ Returns a (possibly) truncated string representation of arg

    If maxlen is positive (or null), returns str(arg) up to maxlen chars.

    :param arg:
    :param maxlen:
    :return:
    """
    s = str(arg)
    if maxlen <= 0 or len(s) <= maxlen:
        return s
    else:
        return "%s.." % s[:maxlen]


try:
    import scipy.sparse as sp
except ImportError:
    sp = None

def is_scipy_sparse(m):
    return sp and sp.issparse(m)


def compute_is_index(seq, obj):
    # assume obj is iterable multiple times
    for i, elt in enumerate(seq):
        if elt is obj:  # use 'is' to identify obj
            return i

    return None


DOCPLEX_CONSOLE_HANDLER = None


def get_logger(name, verbose=False):
    logging_level = logging.WARNING
    if verbose:
        logging_level = logging.DEBUG

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

    global DOCPLEX_CONSOLE_HANDLER

    if DOCPLEX_CONSOLE_HANDLER is None:
        DOCPLEX_CONSOLE_HANDLER = logging.StreamHandler()
        DOCPLEX_CONSOLE_HANDLER.setLevel(logging_level)

    if DOCPLEX_CONSOLE_HANDLER not in logger.handlers:
        logger.addHandler(DOCPLEX_CONSOLE_HANDLER)

    return logger


def open_universal_newline(filename, mode):
    """Opens a file in universal new line mode, in a python 2 and python 3
    compatible way.
    """
    try:
        # try the python 3 syntax
        return open(filename, mode=mode, newline=None)
    except TypeError as te:
        if "'newline'" in str(te):
            # so open does not have a newline parameter -> python 2, use "U"
            # mode
            return open(filename, mode=mode + "U")
        else:
            # for other errors, just raise them
            raise


class CyclicLoop(object):
    """ A cyclic loop executes actions at specified intervals, until
    ``stop()`` is called.

    This loop is based on sched.scheduler

    Attributes:
        stopped: True if the loop is stopped.
    """

    class Task(object):
        """This class stores information needed to manage tasks.

        Attributes:
            id: The id of the task (automatically generated)
            interval: The interval on which that task is called
            action: The action function to call at ``interval``
            argument: The arguments for the action function
        """
        id = 0
        idgen_lock = threading.Lock()

        def __init__(self, interval, priority, action, argument=()):
            self.interval = interval
            self.priority = priority
            self.action = action
            self.argument = argument
            with self.idgen_lock:
                self.id = CyclicLoop.Task.id
                CyclicLoop.Task.id += 1

    def __init__(self):
        """Initialize a new empty CyclicLoop
        """
        self.stop_lock = threading.Lock()
        self.stopped = False
        self.scheduler = sched.scheduler(time.time, time.sleep)
        # maps task id -> ev
        self.events_by_id = {}
        self.tasks_by_id = {}  # task id -> task

    def enter(self, interval, priority, action, argument=()):
        """Schedule a new event.

        Works like sched.scheduler.enter(), but instead of a ``delay``, the
        first argument is the ``interval`` the action must be performed.
        """
        with self.stop_lock:
            if not self.stopped:
                task = CyclicLoop.Task(interval, priority, action, argument)
                self.tasks_by_id[task.id] = task
                self._queue(task)

    def _queue(self, task):
        def process_task(task_id):
            self._process_task(task_id)

        ev = self.scheduler.enter(task.interval, task.priority, process_task, (task.id,))
        self.events_by_id[task.id] = ev

    def _process_task(self, task_id):
        task = self.tasks_by_id[task_id]
        task.action(*task.argument)
        del self.events_by_id[task_id]
        # do not reschedule if we are shutting down
        with self.stop_lock:
            if not self.stopped:
                self._queue(task)

    def start(self):
        """Starts the loop. The loop stops only when ``stop()`` is called.
        """
        self.scheduler.run()

    def stop(self):
        """Stops the Loop.

        When the loop is stopped, its ``stopped`` attribute is set immediately,
        then all tasks in the scheduler are canceled.
        """
        with self.stop_lock:
            self.stopped = True
            for ev in itervalues(self.events_by_id):
                try:
                    self.scheduler.cancel(ev)
                except ValueError:
                    # if stop() is called from an event, the event has already
                    # been triggered and poped'
                    pass


class ClosableQueue(Queue):
    LAST = object()

    def close(self):
        self.put(self.LAST)

    def __iter__(self):
        while True:
            item = self.get()
            try:
                if item is self.LAST:
                    return
                yield item
            finally:
                self.task_done()


class ThreadedCyclicLoop(object):
    """ A cyclic loop executes actions at specified intervals, until
    ``stop()`` is called.

    This loop is based on threads.

    Attributes:
        stopped: True if the loop is stopped.
    """

    class Task(threading.Thread):
        """
        Attributes:
            id: The id of the task (automatically generated)
            interval: The interval on which that task is called
            action: The action function to call at ``interval``
            argument: The arguments for the action function
        """

        def __init__(self, loop, interval, priority, action, argument=()):
            super(ThreadedCyclicLoop.Task, self).__init__()
            self.loop = loop
            self.interval = interval
            self.priority = priority
            self.action = action
            self.argument = argument
            self.stopped = False

        def run(self):
            while not self.stopped:
                # instead of one big sleep, do some smaller sleeps so that
                # we can stop the thread with smaller granularity
                for _ in range(self.interval):
                    time.sleep(1)
                    if self.stopped:
                        break
                if not self.stopped:
                    self.perform()

        def stop(self):
            self.stopped = True

        def perform(self):
            self.action(*self.argument)

    def __init__(self):
        """Initialize a new empty ThreadedCyclicLoop
        """
        self.stop_lock = threading.Lock()
        self.stopped = False
        self.threads = set()
        self.event_queue = ClosableQueue()

    def enter(self, interval, priority, action, argument=()):
        """Schedule a new event.

        Works like sched.scheduler.enter(), but instead of a ``delay``, the
        first argument is the ``interval`` the action must be performed.
        """
        with self.stop_lock:
            if not self.stopped:
                task = ThreadedCyclicLoop.Task(self, interval, priority,
                                               action, argument)
                self.threads.add(task)

    def start(self, mt_worker=None, mt_arg=()):
        """Starts the loop. The loop stops only when ``stop()`` is called.
        """
        for t in self.threads:
            t.start()
        if mt_worker:
            while not self.stopped:
                for task in self.event_queue:
                    mt_worker(*((task,) + mt_arg))
        for t in self.threads:
            t.join()

    def stop(self):
        """Stops the Loop.

        When the loop is stopped, its ``stopped`` attribute is set immediately,
        then all tasks in the scheduler are canceled.
        """
        with self.stop_lock:
            self.stopped = True
        self.event_queue.close()
        for t in self.threads:
            t.stop()


class _SymbolGenerator(object):
    def __init__(self, pattern, offset=1):
        ''' Initialize the counter and the pattern.
            Fixes the pattern by suffixing '%d' if necessary.
        '''
        self._pattern = pattern
        self._offset = offset
        self._set_pattern(pattern)

    def _set_pattern(self, pattern):
        if pattern.endswith('%d'):
            self._pattern = pattern
        else:
            self._pattern = pattern + '%d'

    def new_index_symbol(self, index):
        return self._pattern % (index + self._offset)

    def new_obj_symbol(self, obj):
        return self.new_index_symbol(obj.index)


class _AutomaticSymbolGenerator(_SymbolGenerator):

    def __init__(self, pattern, offset=1):
        ''' Initialize the counter and the pattern.
            Fixes the pattern by suffixing '%d' if necessary.
        '''
        _SymbolGenerator.__init__(self, pattern, offset)
        self._last_index = -1

    def reset(self):
        self._last_index = -1

    def notify_new_index(self, new_index):
        # INTERNAL
        if new_index > self._last_index:
            self._last_index = new_index

    def new_symbol(self):
        """
        Generates and returns a new symbol.
        Guess a new (yet) unallocated index, then use the pattern.
        Note that we use the offset of 1 to generate the name so x1 has index 0, x3 has index 2, etc.
        :return: A symbol string, suposedly not yet allocated.
        """
        guessed_index = self._last_index + 1
        coined_symbol = _SymbolGenerator.new_index_symbol(self, guessed_index)
        self._last_index = guessed_index
        return coined_symbol

    def __iter__(self):
        while True:
            yield self.new_symbol()

class _IndexScope(object):
    # INTERNAL: full scope of indices.

    def __init__(self, qualifier, cplex_scope):
        self._index_map = {}
        self.cplex_scope = cplex_scope
        self.qualifier = qualifier

    def __repr__(self):
        return "IndexScope<{0}>[{1}]".format(self.qualifier, self.size)

    def __getitem__(self, item):
        return self._index_map[item]

    def count_filtered(self, filter):
        return sum(1 for obj in self.iter_objects() if filter(obj))

    @property
    def last(self):
        size = self.size
        if not size:
            raise ValueError("empty scope")
        else:
            return self._index_map[size-1]

    def iter_objects(self):
        if SIX_PY2:
            sorted_by_index = sorted((o for o in itervalues(self._index_map)), key=lambda o: o.index)
            return iter(sorted_by_index)
        else:
            return iter(self._index_map.values())

    def generate_objects_filtered(self, filter):
        for obj in self.iter_objects():
            if filter(obj):
                yield obj

    def generate_objects_with_type(self, obj_type):
        for obj in self.iter_objects():
            if isinstance(obj, obj_type):
                yield obj

    @property
    def size(self):
        idxmap = self._index_map
        return 0 if idxmap is None else len(idxmap)

    def __call__(self, idx):
        return self._object_by_index(idx)

    def get_object_by_index(self, idx, checker=None):
        if checker:
            checker.typecheck_valid_index(idx)
        return self._object_by_index(idx)

    def _object_by_index(self, idx, do_raise=False):
        # do not raise when not found, return None.
        if do_raise:
            return self._index_map[idx]
        else:
            # returns None if idx not in map
            return self._index_map.get(idx)

    def reset(self):
        self._index_map = {}

    def notify_obj_index(self, obj, index):
        if self._index_map is not None:
            self._index_map[index] = obj

    def notify_obj_indices(self, objs, indices):
        if indices:
            idxmap = self._index_map
            if idxmap is not None:
                for obj, idx in izip(objs, indices):
                    idxmap[idx] = obj

    # def update_indices(self):
    #     if self._index_map is not None:
    #         self._index_map = self._make_index_map()
    #
    # def reindex_all(self, indexer):
    #     for ix, obj in enumerate(self._obj_iter()):
    #         obj.index = ix

    def notify_deleted(self, oldidx):
        if self._index_map:
            try:
                del self._index_map[oldidx]
            except KeyError:
                pass

    def notify_delete_set(self, delset):
        nb_deleted = len(delset)
        idxmap = self._index_map
        if nb_deleted and idxmap is not None:
            kept = [o for o in itervalues(idxmap) if o.index not in delset]
            new_map = {}
            for nx, obj in enumerate(kept):
                obj.index = nx
                new_map[nx] = obj
            self._index_map = new_map

    def dump(self, max_lines=100):
        if max_lines is None or max_lines < 0:
            max_lines = 1e+20
        print("-- scope: {0}[{1}]".format(self.qualifier, self.size))
        for l, (k, o) in iteritems(self._index_map):
            if l > max_lines:
                break
            print("  {0}> index={1}: {2:s}".format(l, k, o))

    def check_indices(self):
        # internal, checks that indices and keys are in sync
        for k, o in iteritems(self._index_map):
            if k != o.index:
                raise ValueError("key-index mismatch, key: {0}, index: {1}"
                                 .format(k, o.index))


def apply_thread_limitations(context, solver_context=None):
    if solver_context is None:
        solver_context = context.solver
    # --- limit threads if needed
    parameters = context._get_raw_cplex_parameters()
    if getattr(solver_context, 'max_threads', None) is not None:
        parameters = context.cplex_parameters
        if parameters.threads.get() == 0:
            max_threads = solver_context.max_threads
        else:
            max_threads = min(solver_context.max_threads, parameters.threads.get())
        # we don't want to duplicate parameters unnecessary
        if max_threads != parameters.threads.get():
            new_parameters = parameters.copy()
            # change actual #threads
            new_parameters.threads = max_threads
            out_stream = solver_context.log_output_as_stream
            if out_stream:
                out_stream.write(
                    "WARNING: Number of workers has been reduced to %s to comply with platform limitations.\n" % max_threads)
            # --- here we copy the initial parameters ---
            return new_parameters

    return parameters


def compute_overwrite_nb_threads(parameters, solver_context):
    # returns a corrected value for threads, if necessary, else None
    if getattr(solver_context, 'max_threads', None) is not None:
        solve_threads = solver_context.max_threads
        param_threads = parameters.threads.get()
        if 0 == param_threads:
            # 0 for cplex meeans any number
            max_threads = solve_threads
        else:
            max_threads = min(solve_threads, param_threads)
        if max_threads != param_threads:
            return max_threads
    # None means no overwrite
    return None


def is_almost_equal(x1, x2, reltol, abstol=0):
    # returns true if x2 equals x1 w.r.t a miax of an absolute tolerance and a relative tolerance
    prec = max(abstol, reltol* abs(x1))
    return abs(x2-x1) <= prec


class OutputStreamAdapter:
    # With this class, we kind of automatically handle binary/non binary output streams
    # it automatically perform encoding of strings when needed,
    # and if the stream is a String stream, strings are just written without conversion
    def __init__(self, stream, encoding='utf-8'):
        self.stream = stream
        self.stream_is_binary = False
        if hasattr(stream, 'mode') and 'b' in stream.mode:
            self.stream_is_binary = True
        from io import TextIOBase
        if not isinstance(stream, TextIOBase):
            self.stream_is_binary = True

        self.encoding = encoding

    def write(self, s):
        # s is supposed to be a string
        output_s = s
        if self.stream_is_binary:
            output_s = s.encode(self.encoding)
        self.stream.write(output_s)

def izip2_filled(it1, it2, **kwds):
    # izip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-
    fillvalue = kwds.get('fillvalue')

    class _ZipExhausted(Exception):
        pass

    class _sentinel():
        def __iter__(self):
            raise _ZipExhausted

    fillers = repeat(fillvalue)
    iterators = [chain(it1, _sentinel()), chain(it2, fillers)]
    try:
        while iterators:
            tpl = tuple(map(next, iterators))
            #print(tpl)
            yield tpl
    except _ZipExhausted:
        pass


class MultiObjective(object):
    # a wrapper class to hold all data for multi-objective

    def __init__(self, exprs, priorities, weights=None, abstols=None, reltols=None, names=None):
        self.exprs = exprs
        self.priorities = priorities
        self.weights = weights
        self.abstols = abstols
        self.reltols = reltols
        self.names = names

    @classmethod
    def new_empty(cls):
        return MultiObjective([], [])

    def empty(self):
        return not self.exprs

    @property
    def number_of_objectives(self):
        nb_exprs = len(self.exprs)
        return 0 if nb_exprs <= 1 else nb_exprs

    def clear(self):
        self.exprs = []
        self.priorities = []
        self.weights = []
        self.abstols = None
        self.reltols = None
        self.names = None

    def convert_pseudo_sequence(self, seq, size, default=None, pred=is_number):
        if pred and pred(seq):
            return [seq] * size
        elif is_indexable(seq):
            return seq
        elif seq is None:
            return [default] * size
        else:
            raise TypeError

    def iter_exprs(self):
        multi_obj_exprs = self.exprs
        if len(multi_obj_exprs) >= 2:
            return iter(multi_obj_exprs)
        else:
            return iter([])

    def __getitem__(self, index):
        return self.exprs[index]

    @staticmethod
    def as_optional_sequence(opt_seq, size):
        # converts the argument to None or a list,
        # special case is plain number, converted to a list of expected size.
        if is_number(opt_seq):
            return [opt_seq] * size
        elif opt_seq is None or is_indexable(opt_seq):
            # expecting either None or a list.
            return opt_seq
        else:
            raise TypeError('unexpected optional number sequence: {0!r}'.format(opt_seq))

    def abstols_as_sequence(self, size):
        return self.as_optional_sequence(self.abstols, size)

    def reltols_as_sequence(self, size):
        return self.as_optional_sequence(self.reltols, size)

    def update(self, new_exprs=None,
               new_prios=None, new_weights=None,
               new_abstols=None, new_reltols=None, new_names=None):
        if new_exprs is not None:
            self.exprs = new_exprs
        if new_prios is not None:
            self.priorities = new_prios
        if new_weights is not None:
            self.weights = new_weights
        if new_abstols is not None:
            self.abstols = new_abstols
        if new_reltols is not None:
            self.reltols = new_reltols
        if new_names is not None:
            self.names = new_names

    def itertuples(self):
        # iterates on all components
        # returns tuples of the form (expr, prio, weight, abstol, reltol, name)
        if self.empty():
            return iter([])

        # not empty
        nb_objs = len(self.exprs)
        assert nb_objs >= 2
        # default weight
        lw = self.convert_pseudo_sequence(self.weights, size=nb_objs, default=1)
        la = self.convert_pseudo_sequence(self.abstols, size=nb_objs, default=0)
        lr = self.convert_pseudo_sequence(self.reltols, size=nb_objs, default=0)
        ln = self.convert_pseudo_sequence(self.names, size=nb_objs, default=None, pred=None)

        return iter(zip(self.exprs, self.priorities, lw, la, lr, ln))


def resolve_caller_as_string(caller, sep=': '):
    # resolve caller as a string
    if caller is None:
        return ""

    try:
        return "%s%s" % (caller(), sep)
    except TypeError:
        return "%s%s" % (caller, sep)


def is_quad_expr(obj):
    try:
        return obj.is_quad_expr()
    except AttributeError:
        return False
