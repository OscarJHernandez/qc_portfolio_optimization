# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2019
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model expressed as a CPO file using a local solver
accessed through a shared library (.dll on Windows, .so on Linux).
Interface with library is done using ctypes module.
See https://docs.python.org/2/library/ctypes.html for details.
"""

from docplex.cp.solution import *
from docplex.cp.solution import CpoSolveResult
from docplex.cp.utils import compare_natural
from docplex.cp.solver.solver import CpoSolver, CpoSolverAgent, CpoSolverException

import ctypes
from ctypes.util import find_library
import json
import sys
import time
import os


###############################################################################
## Constants
###############################################################################

# Events received from library
_EVENT_SOLVER_INFO     = 1  # Information on solver as JSON document
_EVENT_JSON_RESULT     = 2  # Solve result expressed as a JSON string
_EVENT_LOG_OUTPUT      = 3  # Log data on output stream
_EVENT_LOG_WARNING     = 4  # Log data on warning stream
_EVENT_LOG_ERROR       = 5  # Log data on error stream
_EVENT_SOLVER_ERROR    = 6  # Solver error. Details are in event associated string.
_EVENT_CPO_CONFLICT    = 7  # Conflict in CPO format


# Event notifier callback prototype
_EVENT_NOTIF_PROTOTYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)

# CPO callback prototype (event name, json data)
_CPE_CALLBACK_PROTOTYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

# Function prototypes (return type, args_type)
_LIB_FUNCTION_PROTYTYPES = \
{
    'createSession'        : (ctypes.c_void_p, (ctypes.c_void_p,)),
    'deleteSession'        : (ctypes.c_int,    (ctypes.c_void_p,)),
    'setCpoModel'          : (ctypes.c_int,    (ctypes.c_void_p, ctypes.c_char_p)),
    'solve'                : (ctypes.c_int,    (ctypes.c_void_p,)),
    'startSearch'          : (ctypes.c_int,    (ctypes.c_void_p,)),
    'searchNext'           : (ctypes.c_int,    (ctypes.c_void_p,)),
    'endSearch'            : (ctypes.c_int,    (ctypes.c_void_p,)),
    'abortSearch'          : (ctypes.c_int,    (ctypes.c_void_p,)),
    'propagate'            : (ctypes.c_int,    (ctypes.c_void_p,)),
    'refineConflict'       : (ctypes.c_int,    (ctypes.c_void_p,)),
    'refineConflictWithCpo': (ctypes.c_int,    (ctypes.c_void_p, ctypes.c_bool)),
    'runSeeds'             : (ctypes.c_int,    (ctypes.c_void_p, ctypes.c_int,)),
    'setExplainFailureTags': (ctypes.c_int,    (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int))),
    'setCpoCallback'       : (ctypes.c_int,    (ctypes.c_void_p, ctypes.c_void_p,)),
}

# Optional lib functions
_LIB_FUNCTION_OPTIONAL = set(['refineConflictWithCpo', 'setExplainFailureTags'])


###############################################################################
##  Public classes
###############################################################################

class CpoSolverLib(CpoSolverAgent):
    """ Interface to a local solver through a shared library """
    __slots__ = ('lib_handler',         # Lib handler
                 'session',             # Solve session in the library
                 'notify_event_proto',  # Prototype of the event callback
                 'first_error_line',    # First line of error
                 'callback_proto',      # Prototype of the CPO callback function
                 'absent_funs',         # Set of optional lib functions that are not available
                 'last_conflict_cpo',   # Last conflict in CPO format
                 )

    def __init__(self, solver, params, context):
        """ Create a new solver using shared library.

        Args:
            solver:   Parent solver
            params:   Solving parameters
            context:  Solver agent context
        Raises:
            CpoSolverException if library is not found
        """
        # Call super
        super(CpoSolverLib, self).__init__(solver, params, context)

        # Initialize attributes
        self.first_error_line = None
        self.lib_handler = None # (to not block end() in case of init failure)
        self.last_conflict_cpo = None

        # Connect to library
        self.lib_handler = self._get_lib_handler()
        self.context.log(2, "Solving library: '{}'".format(self.context.lib))

        # Create session
        # CAUTION: storing callback prototype is mandatory. Otherwise, it is garbaged and the callback fails.
        self.notify_event_proto = _EVENT_NOTIF_PROTOTYPE(self._notify_event)
        self.session = self.lib_handler.createSession(self.notify_event_proto)
        self.context.log(5, "Solve session: {}".format(self.session))

        # Transfer infos in process info
        for x in ('ProxyVersion', 'LibVersion', 'SourceDate', 'SolverVersion'):
            self.process_infos[x] = self.version_info.get(x)

        # Check solver version if any
        sver = self.version_info.get('SolverVersion')
        mver = solver.get_model_format_version()
        if sver and mver and compare_natural(mver, sver) > 0:
            raise CpoSolverException("Solver version {} is lower than model format version {}.".format(sver, mver))

        # Initialize settings indicators
        self.callback_proto = None


    def solve(self):
        """ Solve the model

        According to the value of the context parameter 'verbose', the following information is logged
        if the log output is set:
         * 1: Total time spent to solve the model
         * 2: shared library file
         * 3: Main solving steps
         * 5: Every individual event got from library

        Returns:
            Model solve result (object of class CpoSolveResult)
        Raises:
            CpoException if error occurs
        """
        # Initialize model if needed
        self._init_model_in_solver()

        # Solve the model
        self._call_lib_function('solve', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def start_search(self):
        """ Start a new search. Solutions are retrieved using method search_next().
        """
        # Initialize model if needed
        self._init_model_in_solver()

        self._call_lib_function('startSearch', False)


    def search_next(self):
        """ Get the next available solution.

        Returns:
            Next model result (type CpoSolveResult)
        """
        # Call library function
        self._call_lib_function('searchNext', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def end_search(self):
        """ End current search.

        Returns:
            Last (fail) model solution with last solve information (type CpoSolveResult)
        """
        # Call library function
        self._call_lib_function('endSearch', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def abort_search(self):
        """ Abort current search.
        This method is designed to be called by a different thread than the one currently solving.
        """
        self._call_lib_function('abortSearch', False)


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        See documentation of CpoSolver.refine_conflict() for details.

        Returns:
            Conflict result,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        """
        # Initialize model if needed
        self._init_model_in_solver()

        # Check if cpo format required
        self.last_conflict_cpo = None
        if self.context.add_conflict_as_cpo and ('refineConflictWithCpo' not in self.absent_funs):
            # Request refine conflict with CPO format
            self._call_lib_function('refineConflictWithCpo', True, True)
        else:
            # Call library function
            self._call_lib_function('refineConflict', True)

        # Build result object
        result = self._create_result_object(CpoRefineConflictResult, self.last_json_result)
        result.cpo_conflict = self.last_conflict_cpo
        self.last_conflict_cpo = None
        return result


    def propagate(self):
        """ This method invokes the propagation on the current model.

        See documentation of CpoSolver.propagate() for details.

        Returns:
            Propagation result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """
        # Initialize model if needed
        self._init_model_in_solver()

        # Call library function
        self._call_lib_function('propagate', True)

        # Build result object
        return self._create_result_object(CpoSolveResult, self.last_json_result)


    def run_seeds(self, nbrun):
        """ This method runs *nbrun* times the CP optimizer search with different random seeds
        and computes statistics from the result of these runs.

        This method does not return anything. Result statistics are displayed on the log output
        that should be activated.

        Each run of the solver is stopped according to single solve conditions (TimeLimit for example).
        Total run time is then expected to take *nbruns* times the duration of a single run.

        Args:
            nbrun: Number of runs with different seeds.
        Returns:
            Run result, object of class :class:`~docplex.cp.solution.CpoRunResult`.
        """
        # Initialize model if needed
        self._init_model_in_solver()

        # Call library function
        self._call_lib_function('runSeeds', False, nbrun)

        # Build result object
        self.last_json_result = None
        return self._create_result_object(CpoRunResult)


    def set_explain_failure_tags(self, ltags=None):
        """ This method allows to set the list of failure tags to explain in the next solve.

        The failure tags are displayed in the log when the parameter :attr:`~docplex.cp.CpoParameters.LogSearchTags`
        is set to 'On'.
        All existing failure tags previously set are cleared prior to set the new ones.
        Calling this method with an empty list is then equivalent to just clear tags.

        Args:
            ltags:  List of tag ids to explain
        """
        # Initialize model if needed
        self._init_model_in_solver()

        # Build list of tags
        if ltags is None:
            ltags = []
        elif not is_array(ltags):
            ltags = (ltags,)
        nbtags = len(ltags)

        # Call the function
        self._call_lib_function('setExplainFailureTags', False, nbtags, (ctypes.c_int * nbtags)(*ltags))


    def end(self):
        """ End solver and release all resources.
        """
        if self.lib_handler is not None:
            self.lib_handler.deleteSession(self.session)
            self.lib_handler = None
            self.session = None
            super(CpoSolverLib, self).end()


    def _call_lib_function(self, dfname, json, *args):
        """ Call a library function
        Args:
            dfname:  Name of the function to be called
            json:    Indicate if a JSON result is expected
            *args:   Optional arguments (after session)
        Raises:
            CpoDllException if function call fails or if expected JSON is absent
        """
        # Log call
        if self.context.is_log_enabled(5):
            self.context.log(5, "Call library function: '", dfname, "' with arguments:")
            if args:
                for a in args:
                    self.context.log(5, "   ", a)
            else:
                self.context.log(5, "   None")

        # Reset JSON result if JSON required
        if json:
            self.last_json_result = None

        # Call library function
        try:
            rc = getattr(self.lib_handler, dfname)(self.session, *args)
        except:
            raise CpoNotSupportedException("The function '{}' is not found in the library. Try with a most recent version.".format(dfname))
        if rc != 0:
            errmsg = "Call to '{}' failure (rc={})".format(dfname, rc)
            if self.first_error_line:
               errmsg += ": {}".format(self.first_error_line)
            raise CpoSolverException(errmsg)

        # Check if JSON result is present
        if json:
            if self.last_json_result is None:
               raise CpoSolverException("No JSON result provided by function '{}'".format(dfname))
            self.context.log(5, "JSON result: ", self.last_json_result)

    def _notify_event(self, event, data):
        """ Callback called by the library to notify Python of an event (log, error, etc)
        Args:
            event:  Event id (integer)
            data:   Event data string
        """
        # Process event
        if event == _EVENT_LOG_OUTPUT or event == _EVENT_LOG_WARNING:
            # Store log if required
            if self.log_enabled:
                self._add_log_data(data.decode('utf-8'))

        elif event == _EVENT_JSON_RESULT:
            self.last_json_result = data.decode('utf-8')

        elif event == _EVENT_SOLVER_INFO:
            self.version_info = verinf = json.loads(data.decode('utf-8'))
            # Update information
            verinf['AgentModule'] = __name__
            self.context.log(3, "Local solver info: '", verinf, "'")

        elif event == _EVENT_LOG_ERROR:
            ldata = data.decode('utf-8')
            if self.first_error_line is None:
                self.first_error_line = ldata.replace('\n', '')
            out = self.log_output if self.log_output is not None else sys.stdout
            out.write("ERROR: {}\n".format(ldata))
            out.flush()

        elif event == _EVENT_SOLVER_ERROR:
            errmsg = data.decode('utf-8')
            if self.first_error_line is not None:
                errmsg += " (" + self.first_error_line + ")"
            out = self.log_output if self.log_output is not None else sys.stdout
            out.write("SOLVER ERROR: {}\n".format(errmsg))
            out.flush()

        elif event == _EVENT_CPO_CONFLICT:
            self.last_conflict_cpo = data.decode('utf-8')


    def _cpo_callback(self, event, data):
        """ Callback called by the library to notify Python of an event (log, error, etc)
        Args:
            event:  Event name (string)
            data:   JSON data (string)
        """
        # Decode all data
        stime = time.time()
        event = event.decode('utf-8')
        data = data.decode('utf-8')
        self.process_infos.incr(CpoProcessInfos.TOTAL_UTF8_DECODE_TIME, time.time() - stime)

        # Build result data and notify solver
        res = self._create_result_object(CpoSolveResult, data)
        self.solver._notify_callback_event(event, res)


    def _get_lib_handler(self):
        """ Access the CPO library
        Returns:
            Library handler, with function prototypes defined.
        Raises:
            CpoSolverException if library is not found
        """
        # Access library
        libf = self.context.libfile
        if not libf:
            raise CpoSolverException("CPO library file should be given in 'solver.lib.libfile' context attribute.")

        # Load library
        lib = self._load_lib(libf)

        # Define function prototypes
        self.absent_funs = set()
        for name, proto in _LIB_FUNCTION_PROTYTYPES.items():
            try:
                f = getattr(lib, name)
                f.restype = proto[0]
                f.argtypes = proto[1]
            except:
                if not name in _LIB_FUNCTION_OPTIONAL:
                    raise CpoSolverException("Function '{}' not found in the library {}".format(name, lib))
                else:
                    self.absent_funs.add(name)

        # Return
        return lib


    def _load_lib(self, libf):
        """ Attempt to load a particular library
        Returns:
            Library handler
        Raises:
            CpoSolverException or other Exception if library is not found
        """
        # Search for library file
        if not os.path.isfile(libf):
            lf = find_library(libf)
            if lf is None:
                raise CpoSolverException("Can not find library '{}'".format(libf))
            libf = lf
        # Check library is executable
        if not is_exe_file(libf):
            raise CpoSolverException("Library file '{}' is not executable".format(libf))
        # Load library
        try:
            return ctypes.CDLL(libf)
        except Exception as e:
            raise CpoSolverException("Can not load library '{}': {}".format(libf, e))


    def _send_model_to_solver(self, cpostr):
        """ Send the model to the solver.
        This method must be extended by agent implementations to actually do the operation.
        Args:
            copstr:  String containing the model in CPO format
        """
        # Encode model
        stime = time.time()
        cpostr = cpostr.encode('utf-8')
        self.process_infos.incr(CpoProcessInfos.TOTAL_UTF8_ENCODE_TIME, time.time() - stime)

        # Send CPO model to process
        stime = time.time()
        self._call_lib_function('setCpoModel', True, cpostr)
        self.process_infos.incr(CpoProcessInfos.TOTAL_DATA_SEND_TIME, time.time() - stime)


    def _add_callback_processing(self):
        """ Add the processing of solver callback.
        This method must be extended by agent implementations to actually do the operation.
        """
        # CAUTION: storing callback prototype is mandatory. Otherwise, it is garbaged and the callback fails.
        self.callback_proto = _CPE_CALLBACK_PROTOTYPE(self._cpo_callback)
        self._call_lib_function('setCpoCallback', False, self.callback_proto)


