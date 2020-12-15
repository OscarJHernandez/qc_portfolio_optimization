# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2016
# --------------------------------------------------------------------------

'''
Representation of the DOcplex solving environment.

This module handles the various elements that allow an
optimization program to run independently from the solving environment.
This environment may be:

 * on premise, using a local version of CPLEX Optimization Studio to solve MP problems, or
 * on DOcplexcloud, with the Python program running inside the Python Worker.
 * on Decision Optimization in Watson Machine Learning.

As much as possible, the adaptation to the solving environment is
automatic. The functions that are presented here are useful for handling
very specific use cases.

The following code is a program that sums its input (``sum.py``)::

    import json
    import docplex.util.environment as environment

    sum = 0
    # open program input named "data.txt" and sum the contents
    with environment.get_input_stream("data.txt") as input:
        for i in input.read().split():
            sum += int(i)
    # write the result as a simple json in program output "solution.json"
    with environment.get_output_stream("solution.json") as output:
        output.write(json.dumps({'result': sum}))

Let's put some data in a ``data.txt`` file::

    4 7 8
    19

When you run ``sum.py`` with a Python interpreter, it opens the ``data.txt`` file and sums all of the integers
in that file. The result is saved as a JSON fragment in file ``solution.json``::

    $ python sum.py
    $ more solution.json
    {"result": 38}

To submit the program to the DOcplexcloud service, we write a ``submit.py`` program that uses
the `DOcplexcloud Python API <https://developer.ibm.com/docloud/documentation/docloud/python-api/>`_
to create and submit a job. That job has two attachments:

- ``sum.py``, the program to execute and
- ``data.txt``, the data expected by ``sum.py``.

After the solve is completed, the result of the program is downloaded and saved as ``solution.json``::

    from docloud.job import JobClient

    url = "ENTER_YOUR_URL_HERE"
    key = "ENTER_YOUR_KEY_HERE"
    client = JobClient(url, key)
    client.execute(input=["sum.py", "data.txt"], output="solution.json")

Then you run ``submit.py``::

    $ python submit.py
    $ more solution.json
    {"result": 38}

Environment representation can be accessed with different ways:

    * direct object method calls, after retrieving an instance using
      :meth:`docplex.util.environment.get_environment` and using methods of
      :class:`docplex.util.environment.Environment`.
    * using the function in package `docplex.util.environment`. They will call
       the corresponding methods of Environment in the platform
       `default_environment`:

           * :meth:`docplex.util.environment.get_input_stream`
           * :meth:`docplex.util.environment.get_output_stream`
           * :meth:`docplex.util.environment.read_df`
           * :meth:`docplex.util.environment.write_df`
           * :meth:`docplex.util.environment.get_available_core_count`
           * :meth:`docplex.util.environment.get_parameter`
           * :meth:`docplex.util.environment.update_solve_details`
           * :meth:`docplex.util.environment.add_abort_callback`
           * :meth:`docplex.util.environment.remove_abort_callback`

'''
from collections import deque
import json
from functools import partial
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
import warnings

try:
    from string import maketrans, translate
except ImportError:
    maketrans = str.maketrans
    translate = str.translate

try:
    import pandas
except ImportError:
    pandas = None

from six import iteritems

from docplex.util.logging_utils import LoggerToFile, LoggerToDocloud
from docplex.util.csv_utils import write_table_as_csv

in_ws_nb = None

if in_ws_nb is None:
    in_notebook = ('ipykernel' in sys.modules)
    dsx_home_set = 'dsxuser' in os.environ.get('HOME', '').split('/')
    has_hw_spec = 'RUNTIME_HARDWARE_SPEC' in os.environ
    rt_region_set = 'RUNTIME_ENV_REGION' in os.environ
    
    in_ws_nb = in_notebook and dsx_home_set and has_hw_spec and rt_region_set


log_level_mapping = {'OFF': None,
                     'SEVERE': logging.ERROR,
                     'WARNING': logging.WARNING,
                     'INFO': logging.INFO,
                     'CONFIG': logging.INFO,
                     'FINE': logging.DEBUG,
                     'FINER': logging.DEBUG,
                     'FINEST': logging.DEBUG,
                     'ALL': logging.DEBUG}


class NotAvailableError(Exception):
    ''' The exception raised when a feature is not available
    '''
    pass


def default_solution_storage_handler(env, solution):
    ''' The default solution storage handler.

    The storage handler is a function which first argument is the
    :class:`~Environment` on which a solution should be saved. The `solution`
    is a dict containing all the data for an optimization solution.

    The storage handler is responsible for storing the solution in the
    environment.

    For each (key, value) pairs of the solution, the default solution storage
    handler does the following depending of the type of `value`, in the
    following order:

        * If `value` is a `pandas.DataFrame`, then the data frame is saved
          as an output with the specified `name`. Note that `name` must include
          an extension file for the serialization. See
          :meth:`Environment.write_df` for supported formats.
        * If `value` is a `bytes`, it is saved as binary data with `name`.
        * The `value` is saved as an output with the `name`, after it has been
          converted to JSON.

    Args:
        env: The :class:`~Environment`
        solution: a dict containing the solution.
    '''
    for (name, value) in iteritems(solution):
        if pandas and isinstance(value, pandas.DataFrame):
            _, ext = os.path.splitext(name)
            if ext.lower() == '':
                name = '%s.csv' % name  # defaults to csv if no format specified
            env.write_df(value, name)
        elif isinstance(value, bytes):
            with env.get_output_stream(name) as fp:
                fp.write(value)
        else:
            # try jsonify
            with env.get_output_stream(name) as fp:
                json.dump(value, fp)

# The global output lock
global_output_lock = threading.Lock()


class SolveDetailsFilter(object):
    '''Default solve detail filter class.

    This default class filters details so that there are no more than 1 solve
    details per second.
    '''
    def __init__(self, interval=1):
        self.last_accept_time = 0
        self.interval = interval

    def filter(self, details):
        '''Filters the details.

        Returns:
            True if the details are to be published.
        '''
        ret_val = None
        now = time.time()
        if (now - self.last_accept_time > self.interval):
            ret_val = details
            self.last_accept_time = now
        return ret_val


class Environment(object):
    ''' Methods for interacting with the execution environment.

    Internally, the ``docplex`` package provides the appropriate implementation
    according to the actual execution environment.
    The correct instance of this class is returned by the method
    :meth:`docplex.util.environment.get_environment` that is provided in this
    module.

    Attributes:
        abort_callbacks: A list of callbacks that are called when the script is
            run on DOcplexcloud and a job abort operation is requested. You
            add your own callback using::

                env.abort_callbacks += [your_cb]

            or::

                env.abort_callbacks.append(your_cb)

            You remove a callback using::

                env.abort_callbacks.remove(your_cb)

        solution_storage_handler: A function called when a solution is to be
            stored. The storage handler is a function which first argument is
            the :class:`~Environment` on which a solution should be saved. The
            `solution` is a dict containing all the data for an optimization
            solution. The default is :meth:`~default_solution_storage_handler`.
        record_history_fields: Fields which history is to be kept
        record_history_size: maximum number of records in history
        record_interval: min time between to history records
    '''
    def __init__(self):
        self.output_lock = global_output_lock
        self.solution_storage_handler = default_solution_storage_handler
        self.abort_callbacks = []
        self.update_solve_details_dict = True
        self.last_solve_details = {}  # stores the latest published details
        # private behaviour for now: allows to filter details
        # the SolveDetailsFilter.filter() method returns true if the details
        # are to be kept
        self.details_filter = None
        self.unpublished_details = None
        self._record_history_fields = None
        # self.record_history_fields = ['PROGRESS_CURRENT_OBJECTIVE']
        self.record_history = {}  # maps name -> deque
        self.last_history_record = {}  # we keep the last here so that we can publish at end of solve
        self.record_history_time_decimals = 2  # number of decimals for time
        self.record_history_size = 100
        self.record_min_time = 1
        self.recorded_solve_details_count = 0  # number of solve details that have been sent to recording
        self.autoreset = True

    def _reset_record_history(self, force=False):
        if self.autoreset or force:
            self.record_history = {}
            self.unpublished_details = None
            self.last_history_record = {}
            self.recorded_solve_details_count = 0

    def get_record_history_fields(self):
        if self._record_history_fields is None:
            if self.is_dods():
                self._record_history_fields = ['PROGRESS_BEST_OBJECTIVE',
                                               'PROGRESS_CURRENT_OBJECTIVE',
                                               'PROGRESS_GAP']
            else:
                # the default out of dods is to not record any history
                self._record_history_fields = []
        return self._record_history_fields

    def set_record_history_fields(self, value):
        self._record_history_fields = value

    # let record_history_fields be a property that is lazy initialized
    # this gives the opportunity to set is_dods before record history fields are needed
    record_history_fields = property(get_record_history_fields, set_record_history_fields)

    def store_solution(self, solution):
        '''Stores the specified solution.

        This method guarantees that the solution is fully saved if the model
        is running on DOcplexcloud python worker and an abort of the job is
        triggered.

        For each (key, value) pairs of the solution, the default solution
        storage handler does the following depending of the type of `value`, in
        the following order:

        * If `value` is a `pandas.DataFrame`, then the data frame is saved
          as an output with the specified `name`. Note that `name` must include
          an extension file for the serialization. See
          :meth:`Environment.write_df` for supported formats.
        * If `value` is a `bytes`, it is saved as binary data with `name`.
        * The `value` is saved as an output with the `name`, after it has been
          converted to JSON.

        Args:
            solution: a dict containing the solution.
        '''
        with self.output_lock:
            self.solution_storage_handler(self, solution)

    def get_input_stream(self, name):
        ''' Get an input of the program as a stream (file-like object).

        An input of the program is a file that is available in the working directory.

        When run on DOcplexcloud, all input attachments are copied to the working directory before
        the program is run. ``get_input_stream`` lets you open the input attachments of the job.

        Args:
            name: Name of the input object.
        Returns:
            A file object to read the input from.
        '''
        return None

    def read_df(self, name, reader=None, **kwargs):
        ''' Reads an input of the program as a ``pandas.DataFrame``.

        ``pandas`` must be installed.

        ``name`` is the name of the input object, as a filename. If a reader
        is not user provided, the reader used depends on the filename extension.

        The default reader used depending on extension are:

            * ``.csv``: ``pandas.read_csv()``
            * ``.msg``: ``pandas.read_msgpack()``

        Args:
            name: The name of the input object
            reader: an optional reader function
            **kwargs: additional parameters passed to the reader
        Raises:
            NotAvailableError: raises this error when ``pandas`` is not
                available.
        '''
        if pandas is None:
            raise NotAvailableError('read_df() is only available if pandas is installed')
        _, ext = os.path.splitext(name)
        default_kwargs = None
        if reader is None:
            default_readers = {'.csv': (pandas.read_csv, {'index_col': 0}),
                               '.msg': (pandas.read_msgpack, None)}
            reader, default_kwargs = default_readers.get(ext.lower(), None)
        if reader is None:
            raise ValueError('no default reader defined for files with extension: \'%s\'' % ext)
        with self.get_input_stream(name) as ost:
            # allow
            params = {}
            if default_kwargs:
                params.update(default_kwargs)
            if kwargs:
                params.update(kwargs)
            return reader(ost, **params)

    def write_df(self, df, name, writer=None, **kwargs):
        ''' Write a ``pandas.DataFrame`` as an output of the program.

        ``pandas`` must be installed.

        ``name`` is the name of the input object, as a filename. If a writer
        is not user provided, the writer used depends on the filename extension.

        This currently only supports csv output.

        Args:
            name: The name of the input object
            writer: an optional writer function
            **kwargs: additional parameters passed to the writer
        Raises:
            NotAvailableError: raises this error when ``pandas`` is not
                available.
        '''
        if pandas is None:
            raise NotAvailableError('write_df() is only available if pandas is installed')
        _, ext = os.path.splitext(name)
        if writer is None:
            try:
                default_writers = {'.csv': df.to_csv}
                writer = default_writers.get(ext.lower(), None)
            except AttributeError:
                raise NotAvailableError('Could not write writer function for extension: %s' % ext)
        if writer is None:
            raise ValueError('no default writer defined for files with extension: \'%s\'' % ext)
        with self.get_output_stream(name) as ost:
            if sys.version_info[0] < 3:
                ost.write(writer(index=False, encoding='utf8'))
            else:
                ost.write(writer(index=False).encode(encoding='utf8'))

    def set_output_attachment(self, name, filename):
        '''Attach the file which filename is specified as an output of the
        program.

        The file is recorded as being part of the program output.
        This method can be called multiple times if the program contains
        multiple output objects.

        When run on premise, ``filename`` is copied to the the working
        directory (if not already there) under the name ``name``.

        When run on DOcplexcloud, the file is attached as output attachment.

        Args:
            name: Name of the output object.
            filename: The name of the file to attach.
        '''
        pass

    def get_output_stream(self, name):
        ''' Get a file-like object to write the output of the program.

        The file is recorded as being part of the program output.
        This method can be called multiple times if the program contains
        multiple output objects.

        When run on premise, the output of the program is written as files in
        the working directory. When run on DOcplexcloud, the files are attached
        as output attachments.

        The stream is opened in binary mode, and will accept 8 bits data.

        Args:
            name: Name of the output object.
        Returns:
            A file object to write the output to.
        '''
        return None

    def get_available_core_count(self):
        ''' Returns the number of cores available for processing if the environment
        sets a limit.

        This number is used in the solving engine as the number of threads.

        Returns:
            The available number of cores or ``None`` if the environment does not
            limit the number of cores.
        '''
        return None

    def get_parameters(self):
        ''' Returns a dict containing all parameters of the program.

        On DOcplexcloud, this method returns the job parameters.
        On local solver, this method returns ``os.environ``.

        Returns:
            The job parameters
        '''
        return None

    def get_parameter(self, name):
        ''' Returns a parameter of the program.

        On DOcplexcloud, this method returns the job parameter whose name is specified.
        On local solver, this method returns the environment variable whose name is specified.

        Args:
            name: The name of the parameter.
        Returns:
            The parameter whose name is specified or None if the parameter does
            not exists.
        '''
        return None

    def notify_start_solve(self, solve_details, engine_type=None):
        # ===============================================================================
        #         '''Notify the solving environment that a solve is starting.
        #
        #         If ``context.solver.auto_publish.solve_details`` is set, the underlying solver will automatically
        #         send details. If you want to craft and send your own solve details, you can use the following
        #         keys (non exhaustive list):
        #
        #             - MODEL_DETAIL_TYPE : Model type
        #             - MODEL_DETAIL_CONTINUOUS_VARS : Number of continuous variables
        #             - MODEL_DETAIL_INTEGER_VARS : Number of integer variables
        #             - MODEL_DETAIL_BOOLEAN_VARS : Number of boolean variables
        #             - MODEL_DETAIL_INTERVAL_VARS : Number of interval variables
        #             - MODEL_DETAIL_SEQUENCE_VARS : Number of sequence variables
        #             - MODEL_DETAIL_NON_ZEROS : Number of non zero variables
        #             - MODEL_DETAIL_CONSTRAINTS : Number of constraints
        #             - MODEL_DETAIL_LINEAR_CONSTRAINTS : Number of linear constraints
        #             - MODEL_DETAIL_QUADRATIC_CONSTRAINTS : Number of quadratic constraints
        #
        #         Args:
        #             solve_details: A ``dict`` with solve details as key/value pairs
        #         See:
        #             :attr:`.Context.solver.auto_publish.solve_details`
        #         '''
        # ===============================================================================
        self._reset_record_history()

    def update_solve_details(self, details):
        '''Update the solve details.

        You use this method to send solve details to the DOcplexcloud service.
        If ``context.solver.auto_publish`` is set, the underlying
        solver will automatically update solve details once the solve has
        finished.

        Args:
            details: A ``dict`` with solve details as key/value pairs.
        '''
        # publish details
        to_publish = None
        if self.update_solve_details_dict:
            previous = self.last_solve_details
            to_publish = {}
            if details:
                to_publish.update(previous)
                to_publish.update(details)
            self.last_solve_details = to_publish
        else:
            to_publish = details
        # process history
        to_publish = self.record_in_history(to_publish)

        if self.details_filter:
            if self.details_filter.filter(details):
                self.publish_solve_details(to_publish)
            else:
                # just store the details for later use
                self.unpublished_details = to_publish
        else:
            self.publish_solve_details(to_publish)

    def record_in_history(self, details):
        self.recorded_solve_details_count += 1
        for f in self.record_history_fields:
            if f in details:
                current_ts = round(time.time(), self.record_history_time_decimals)
                current_history_element = [current_ts, details[f]]
                l = self.record_history.get(f, deque([], self.record_history_size))
                self.record_history[f] = l
                last_ts = l[-1][0] if len(l) >= 1 else -9999
                if (current_ts - last_ts) >= self.record_min_time:
                    l.append(current_history_element)
                    details['%s.history' % f] = json.dumps(list(l))  # make new copy
                else:
                    self.last_history_record[f] = current_history_element
        return details
    
    def prepare_last_history(self):
        details = {}
        details.update(self.last_solve_details)
        any_added = False
        for k, v in self.last_history_record.items():
            the_list = self.record_history[k]
            do_append = True
            if len(the_list) >= 1:
                last_date_history = the_list[-1][0]
                last_date = v[0]
                do_append = (abs(last_date - last_date_history) >= 0.01)
            if do_append:
                the_list.append(v)
                any_added = True
            details['%s.history' % k] = json.dumps(list(the_list))
        return details if any_added else False

    def publish_solve_details(self, details):
        '''Actually publish the solve specified details.

        Returns:
            The published details
        '''
        pass

    def notify_end_solve(self, status, solve_time=None):
        # ===============================================================================
        #         '''Notify the solving environment that the solve as ended.
        #
        #         The ``status`` can be a docloud.status.JobSolveStatus enum or an integer.
        #
        #         When ``status`` is an integer, it is converted with the following conversion table:
        #
        #             0 - UNKNOWN: The algorithm has no information about the solution.
        #             1 - FEASIBLE_SOLUTION: The algorithm found a feasible solution.
        #             2 - OPTIMAL_SOLUTION: The algorithm found an optimal solution.
        #             3 - INFEASIBLE_SOLUTION: The algorithm proved that the model is infeasible.
        #             4 - UNBOUNDED_SOLUTION: The algorithm proved the model unbounded.
        #             5 - INFEASIBLE_OR_UNBOUNDED_SOLUTION: The model is infeasible or unbounded.
        #
        #         Args:
        #             status: The solve status
        #             solve_time: The solve time
        #         '''
        # ===============================================================================
        if self.unpublished_details:
            self.publish_solve_details(self.unpublished_details)
        if self.recorded_solve_details_count >= 1 and self.last_history_record:
            last_details = self.prepare_last_history()
            if last_details:
                self.publish_solve_details(last_details)

    def set_stop_callback(self, cb):
        '''Sets a callback that is called when the script is run on
        DOcplexcloud and a job abort operation is requested.

        You can also use the ``stop_callback`` property to set the callback.

        Deprecated since 2.4 - Use self.abort_callbacks += [cb] instead'

        Args:
            cb: The callback function
        '''
        warnings.warn('set_stop_callback() is deprecated since 2.4 - Use Environment.abort_callbacks.append(cb) instead')

    def get_stop_callback(self):
        '''Returns the stop callback that is called when the script is run on
        DOcplexcloud and a job abort operation is requested.

        You can also use the ``stop_callback`` property to get the callback.

        Deprecated since 2.4 - Use the abort_callbacks property instead')

        '''
        warnings.warn('get_stop_callback() is deprecated since 2.4 - Use the abort_callbacks property instead')
        return None

    stop_callback = property(get_stop_callback, set_stop_callback)

    def get_engine_log_level(self):
        '''Returns the engine log level as set by job parameter oaas.engineLogLevel.

        oaas.engineLogLevel values are: OFF, SEVERE, WARNING, INFO, CONFIG, FINE, FINER, FINEST, ALL

        The mapping to logging levels in python are:

           * OFF: None
           * SEVERE: logging.ERROR
           * WARNING: logging.WARNING
           * INFO, CONFIG: logging.INFO
           * FINE, FINER, FINEST: logging.DEBUG
           * ALL: logging.DEBUG


        All other values are considered invalid values and will return None.

        Returns:
            The logging level or None if not set (off)
        '''
        oaas_level = self.get_parameter('oaas.engineLogLevel')
        log_level = log_level_mapping.get(oaas_level.upper(), None) if oaas_level else None
        return log_level

    def is_debug_mode(self):
        '''Returns true if the engine should run in debug mode.

        This is equivalent to ``env.get_engine_log_level() <= logging.DEBUG``
        '''
        lvl = self.get_engine_log_level()
        # logging.NOTSET is zero so will return false
        return (self.get_engine_log_level() <= logging.DEBUG) if lvl is not None else False

    def is_dods(self):
        '''Returns true if this environment in running in DODS.
        '''
        value = os.environ.get("IS_DODS")
        return str(value).lower() == "true"

    def get_logger(self):
        '''Returns a ``docplex.util.logging.DocplexLogger`` that can be use
        for logging purposes.
        '''
        return None

class AbstractLocalEnvironment(Environment):
    # The environment solving environment using all local input and outputs.
    def __init__(self):
        super(AbstractLocalEnvironment, self).__init__()
        self.logger = None

        # init number of cores. Default is no limits (engines will use
        # number of cores reported by system).
        # On Watson studio runtimes, the system reports the total number
        # of physical cores but not the number of cores available to the
        # runtime. The number of cores available to the runtime are
        # specified in an environment variable instead.
        self._available_cores = None
        RUNTIME_HARDWARE_SPEC = os.environ.get('RUNTIME_HARDWARE_SPEC', None)
        if RUNTIME_HARDWARE_SPEC:
            try:
                spec = json.loads(RUNTIME_HARDWARE_SPEC)
                num = int(spec.get('num_cpu')) if ('num_cpu' in spec) else None
                self._available_cores = num
            except:
                pass

    def get_input_stream(self, name):
        return open(name, "rb")

    def get_output_stream(self, name):
        return open(name, "wb")

    def get_parameter(self, name):
        return os.environ.get(name, None)

    def get_parameters(self):
        return os.environ

    def set_output_attachment(self, name, filename):
        # check that name leads to a file in cwd
        attachment_abs_path = os.path.dirname(os.path.abspath(name))
        if attachment_abs_path != os.getcwd():
            raise ValueError('Illegal attachment name')

        if os.path.dirname(os.path.abspath(filename)) != os.getcwd():
            shutil.copyfile(filename, name)  # copy to current

    def get_logger(self):
        if not self.logger:  # lazy init
            self.logger = LoggerToFile(sys.stdout)
        return self.logger

    def get_available_core_count(self):
        return self._available_cores

class LocalEnvironment(AbstractLocalEnvironment):
    def __init__(self):
        super(LocalEnvironment, self).__init__()

from .ws.util import START_SOLVE_EVENT, END_SOLVE_EVENT, Tracker

class WSNotebookEnvironment(AbstractLocalEnvironment):
    def __init__(self, tracker=None):
        super(WSNotebookEnvironment, self).__init__()
        self._start_time = None
        self.solve_id = str(uuid.uuid4())  # generate random uuid for each session
        self.model_type = None # set in start solve
        self.tracker = tracker if tracker else Tracker()

    def notify_start_solve(self, solve_details):
        super(WSNotebookEnvironment, self).notify_start_solve(solve_details)
        # Prepare data for WS
        detail_type = solve_details.get('MODEL_DETAIL_TYPE', None)
        model_type = "cpo" if detail_type and detail_type.startswith('CPO') else "cplex"
        num_constraints = solve_details.get('MODEL_DETAIL_CONSTRAINTS', 0)
        num_variables = solve_details.get('MODEL_DETAIL_CONTINUOUS_VARS', 0) \
            + solve_details.get('MODEL_DETAIL_INTEGER_VARS', 0) \
            + solve_details.get('MODEL_DETAIL_BOOLEAN_VARS', 0) \
            + solve_details.get('MODEL_DETAIL_INTERVAL_VARS', 0) \
            + solve_details.get('MODEL_DETAIL_SEQUENCE_VARS', 0)
        model_statistics = {'numConstraints': num_constraints,
                            'numVariables': num_variables}
        cplex_edition = _get_cplex_edition()
        details = {'modelType': model_type,
                   'modelSize': model_statistics,
                   'solveId': self.solve_id,
                   'edition': cplex_edition}
        self.model_type = model_type
        self.tracker.notify_ws(START_SOLVE_EVENT, details)
        self._start_time = time.time()
        
    def notify_end_solve(self, status, solve_time=None):
        super(WSNotebookEnvironment, self).notify_end_solve(status, solve_time=solve_time)
        # do the watson studio things
        if (self._start_time and solve_time == None):
            solve_time = (time.time() - self._start_time)
        details = {'solveTime': solve_time,
                   'modelType': self.model_type,
                   'solveId': self.solve_id}
        self.tracker.notify_ws(END_SOLVE_EVENT, details)
        self._start_time = None

class OutputFileWrapper(object):
    # Wraps a file object so that on __exit__() and on close(), the wrapped file is closed and
    # the output attachments are actually set in the worker
    def __init__(self, file, solve_hook, attachment_name):
        self.file = file
        self.solve_hook = solve_hook
        self.attachment_name = attachment_name
        self.closed = False

    def __getattr__(self, name):
        if name == 'close':
            return self.my_close
        else:
            return getattr(self.file, name)

    def __enter__(self, *args, **kwargs):
        return self.file.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        self.file.__exit__(*args, **kwargs)
        self.close()

    def close(self):
        # actually close the output then set attachment
        if not self.closed:
            self.file.close()
            self.solve_hook.set_output_attachments({self.attachment_name: self.file.name})
            self.closed = True


def worker_env_stop_callback(env):
    # wait for the output lock to be released to make sure that the latest
    # solution store operation has ended.
    with env.output_lock:
        pass
    # call all abort callbacks
    for cb in env.abort_callbacks:
        cb()


class WorkerEnvironment(Environment):
    # The solving environment when we run in the DOcplexCloud worker.
    def __init__(self, solve_hook):
        super(WorkerEnvironment, self).__init__()
        self.solve_hook = solve_hook
        if solve_hook:
            self.solve_hook.stop_callback = partial(worker_env_stop_callback, self)
        self.logger = None

    def get_available_core_count(self):
        return self.solve_hook.get_available_core_count()

    def get_input_stream(self, name):
        # inputs are in the current working directory
        return open(name, "rb")

    def get_output_stream(self, name):
        # open the output in a place we know we can write
        f = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        return OutputFileWrapper(f, self.solve_hook, name)

    def set_output_attachment(self, name, filename):
        self.solve_hook.set_output_attachments({name: filename})

    def get_parameter(self, name):
        return self.solve_hook.get_parameter_value(name)

    def get_parameters(self, ):
        # This is a typo in _DockerSolveHook, this should be "parameters"
        return self.solve_hook.parameter

    def publish_solve_details(self, details):
        super(WorkerEnvironment, self).publish_solve_details(details)
        self.solve_hook.update_solve_details(details)
        # if on dods, we want to publish stats.csv if any
        if self.is_dods():
            self._publish_stats_csv(details)

    def _publish_stats_csv(self, stats):
        # generate the stats.csv file with the specified stats
        names = ['stats.csv']
        stats_table = []
        for k in stats:
            if k.startswith("STAT."):
                stats_table.append([k, stats[k]])
        if stats_table:
            field_names = ['Name', 'Value']
            for name in names:
                write_table_as_csv(self, stats_table, name, field_names)

    def notify_start_solve(self, solve_details):
        super(WorkerEnvironment, self).notify_start_solve(solve_details)
        self.solve_hook.notify_start_solve(None,  # model
                                           solve_details)

    def notify_end_solve(self, status, solve_time=None):
        super(WorkerEnvironment, self).notify_end_solve(status)
        try:
            from docplex.util.status import JobSolveStatus
            engine_status = JobSolveStatus(status) if status else JobSolveStatus.UNKNOWN
            self.solve_hook.notify_end_solve(None,  # model, unused
                                             None,  # has_solution, unused
                                             engine_status,
                                             None,  # reported_obj, unused
                                             None,  # var_value_dict, unused
                                             )
        except ImportError:
            raise RuntimeError("This should have been called only when in a worker environment")

    def set_stop_callback(self, cb):
        warnings.warn('set_stop_callback() is deprecated since 2.4 - Use Environment.abort_callbacks.append(cb) instead')
        self.abort_callbacks += [cb]

    def get_stop_callback(self):
        warnings.warn('get_stop_callback() is deprecated since 2.4 - Use the abort_callbacks property instead')
        return self.abort_callbacks[1] if self.abort_callbacks else None

    def get_logger(self):
        if not self.logger:
            if hasattr(self.solve_hook, 'logger'):
                self.logger = LoggerToDocloud(self.solve_hook.logger)
            else:
                self.logger = LoggerToFile(sys.stdout)
        return self.logger


class OverrideEnvironment(object):
    '''Allows to temporarily replace the default environment.

    If the override environment is None, nothing happens and the default
    environment is not replaced
    '''
    def __init__(self, new_env=None):
        self.set_env = new_env
        self.saved_env = None

    def __enter__(self):
        if self.set_env:
            global default_environment
            self.saved_env = default_environment
            default_environment = self.set_env
        else:
            self.saved_env = None

    def __exit__(self, type, value, traceback):
        if self.saved_env:
            global default_environment
            default_environment = self.saved_env


def _get_default_environment():
    # creates a new instance of the default environment
    try:
        import docplex.worker.solvehook as worker_env
        hook = worker_env.get_solve_hook()
        if hook:
            return WorkerEnvironment(hook)
    except ImportError:
        pass
    if in_ws_nb:
        return WSNotebookEnvironment()
    return LocalEnvironment()

default_environment = _get_default_environment()

def _get_cplex_edition():
    with OverrideEnvironment(Environment()):
        import docplex.mp.model
        import docplex.mp.environment
        edition = " ce" if docplex.mp.model.Model.is_cplex_ce() else ""
        version = docplex.mp.environment.Environment().cplex_version
        return "%s%s" % (version, edition)


def get_environment():
    ''' Returns the Environment object that represents the actual execution
    environment.

    Note: the default environment is the value of the
    ``docplex.util.environment.default_environment`` property.

    Returns:
        An instance of the :class:`.Environment` class that implements methods
        corresponding to actual execution environment.
    '''
    return default_environment


def get_input_stream(name):
    ''' Get an input of the program as a stream (file-like object),
    with the default environment.

    An input of the program is a file that is available in the working directory.

    When run on DOcplexcloud, all input attachments are copied to the working directory before
    the program is run. ``get_input_stream`` lets you open the input attachments of the job.

    Args:
        name: Name of the input object.
    Returns:
        A file object to read the input from.
    '''
    return default_environment.get_input_stream(name)


def set_output_attachment(name, filename):
    ''' Attach the file which filename is specified as an output of the
    program.

    The file is recorded as being part of the program output.
    This method can be called multiple times if the program contains
    multiple output objects.

    When run on premise, ``filename`` is copied to the the working
    directory (if not already there) under the name ``name``.

    When run on DOcplexcloud, the file is attached as output attachment.

    Args:
        name: Name of the output object.
        filename: The name of the file to attach.
    '''
    return default_environment.set_output_attachment(name, filename)


def get_output_stream(name):
    ''' Get a file-like object to write the output of the program.

    The file is recorded as being part of the program output.
    This method can be called multiple times if the program contains
    multiple output objects.

    When run on premise, the output of the program is written as files in
    the working directory. When run on DOcplexcloud, the files are attached
    as output attachments.

    The stream is opened in binary mode, and will accept 8 bits data.

    Args:
        name: Name of the output object.
    Returns:
        A file object to write the output to.
    '''
    return default_environment.get_output_stream(name)


def read_df(name, reader=None, **kwargs):
    ''' Reads an input of the program as a ``pandas.DataFrame`` with the
    default environment.

    ``pandas`` must be installed.

    ``name`` is the name of the input object, as a filename. If a reader
    is not user provided, the reader used depends on the filename extension.

    The default reader used depending on extension are:

        * ``.csv``: ``pandas.read_csv()``
        * ``.msg``: ``pandas.read_msgpack()``

    Args:
        name: The name of the input object
        reader: an optional reader function
        **kwargs: additional parameters passed to the reader
    Raises:
        NotAvailableError: raises this error when ``pandas`` is not
            available.
    '''
    return default_environment.read_df(name, reader=reader, **kwargs)


def write_df(df, name, writer=None, **kwargs):
    ''' Write a ``pandas.DataFrame`` as an output of the program with the
    default environment.

    ``pandas`` must be installed.

    ``name`` is the name of the input object, as a filename. If a writer
    is not user provided, the writer used depends on the filename extension.

    The default writer used depending on extension are:

        * ``.csv``: ``DataFrame.to_csv()``
        * ``.msg``: ``DataFrame.to_msgpack()``

    Args:
        name: The name of the input object
        writer: an optional writer function
        **kwargs: additional parameters passed to the writer
    Raises:
        NotAvailableError: raises this error when ``pandas`` is not
            available.
    '''
    return default_environment.write_df(df, name, writer=writer, **kwargs)


def get_available_core_count():
    ''' Returns the number of cores available for processing if the environment
    sets a limit, with the default environment.

    This number is used in the solving engine as the number of threads.

    Returns:
        The available number of cores or ``None`` if the environment does not
        limit the number of cores.
    '''
    return default_environment.get_available_core_count()


def get_parameter(name):
    ''' Returns a parameter of the program, with the default environment.

    On DOcplexcloud, this method returns the job parameter whose name is specified.

    Args:
        name: The name of the parameter.
    Returns:
        The parameter whose name is specified.
    '''
    return default_environment.get_parameter(name)


def update_solve_details(details):
    '''Update the solve details, with the default environment

    You use this method to send solve details to the DOcplexcloud service.
    If ``context.solver.auto_publish`` is set, the underlying
    solver will automatically update solve details once the solve has
    finished.

    Args:
        details: A ``dict`` with solve details as key/value pairs.
    '''
    return default_environment.update_solve_details(details)


def add_abort_callback(cb):
    '''Adds the specified callback to the default environment.

    The abort callback is called when the script is run on
    DOcplexcloud and a job abort operation is requested.

    Args:
        cb: The abort callback
    '''
    default_environment.abort_callbacks += [cb]


def remove_abort_callback(cb):
    '''Adds the specified callback to the default environment.

    The abort callback is called when the script is run on
    DOcplexcloud and a job abort operation is requested.

    Args:
        cb: The abort callback
    '''
    default_environment.abort_callbacks.remove(cb)

attachment_invalid_characters = '/\\?%*:|"#<> '
attachment_trans_table = maketrans(attachment_invalid_characters, '_' * len(attachment_invalid_characters))


def make_attachment_name(name):
    '''From `name`, create an attachment name that is correct for DOcplexcloud.

    Attachment filenames in DOcplexcloud has certain restrictions. A file name:

        - is limited to 255 characters;
        - can include only ASCII characters;
        - cannot include the characters `/\?%*:|"<>`, the space character, or the null character; and
        - cannot include _ as the first character.

    This method replace all unauthorized characters with _, then removing leading
    '_'.

    Args:
        name: The original attachment name
    Returns:
        An attachment name that conforms to the restrictions.
    Raises:
        ValueError if the attachment name is more than 255 characters
    '''
    new_name = translate(name, attachment_trans_table)
    while (new_name.startswith('_')):
        new_name = new_name[1:]
    if len(new_name) > 255:
        raise ValueError('Attachment names are limited to 255 characters')
    return new_name
