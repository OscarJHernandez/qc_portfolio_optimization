# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model expressed as a CPO file using DOcplexcloud services.
"""

import docplex.cp.solution as solution
from docplex.cp.solution import CpoProcessInfos, CpoSolverInfos, CpoSolveResult, CpoRefineConflictResult
from docplex.cp.solver.docloud_client import JobClient
from docplex.cp.utils import CpoException, is_number
import docplex.cp.solver.solver as solver
import time
import warnings


###############################################################################
##  Constants
###############################################################################

# Solution status indication that a json result is expected
_STATUS_WITH_RESULT = (solution.SOLVE_STATUS_FEASIBLE,
                       solution.SOLVE_STATUS_OPTIMAL)


###############################################################################
##  Public classes
###############################################################################

class CpoSolverDocloud(solver.CpoSolverAgent):
    """ Solver of CPO model using DOcplexcloud services. """
    
    def __init__(self, solver, params, context):
        """ Create a new solver using DOcplexcloud web service

        Args:
            solver:   Parent solver
            params:   Solving parameters
            context:  DOcplexcloud Solver context
        Raises:
            CpoException if jar file does not exists
        """
        warnings.warn(
            'Solve using \'docloud\' agent is deprecated. Consider submitting your model to DOcplexcloud. See https://ibm.biz/BdYhhK',
            DeprecationWarning)
        if (context.key is None) or (' ' in context.key):
            raise CpoException("Your DOcplexcloud key has not been set")
        super(CpoSolverDocloud, self).__init__(solver, params, context)


    def solve(self):
        """ Solve the model

        According to the value of the context parameter 'verbose', the following information is logged
        if the log output is set:
         * 1: Total time spent to solve the model
         * 2: Calls to DOcplexcloud solving
         * 3: Detailed DOcplexcloud job information
         * 4: REST requests and response codes

        Returns:
            Model solve result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        Raises:
            CpoException if error occurs
        """
        return self._submit_job('Solve', CpoSolveResult)


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        See documentation of CpoSolver.refine_conflict() for details.

        Returns:
            Conflict result,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        """
        # Ensure cpo model is generated with all constraints named
        self.context.model.name_all_constraints = True

        return self._submit_job('RefineConflict', CpoRefineConflictResult)


    def propagate(self):
        """ This method invokes the propagation on the current model.

        See documentation of CpoSolver.propagate() for details.

        Returns:
            Propagation result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """
        return self._submit_job('Propagate', CpoSolveResult)


    def _submit_job(self, cmd, rclass):
        """ Submit a solving job with given command.

        According to the value of the context parameter 'verbose', the following information is logged
        if the log output is set:
         * 1: Total time spent to solve the model
         * 2: Calls to DOcplexcloud solving
         * 3: Detailed DOcplexcloud job information
         * 4: REST requests and response codes

        Args:
            cmd:    Solving command, in (Solve, RefineConflict, Propagate)
            rclass: Result object class
        Returns:
            Model solve result (object of class CpoSolveResult)
        Raises:
            CpoException if error occurs
        """
        # Create DOcplexcloud client
        ctx = self.context
        client = JobClient(ctx)

        # Convert model into CPO format
        cpostr = self._get_cpo_model_string()

        # Encode model
        stime = time.time()
        cpostr = cpostr.encode('utf-8')
        self.process_infos[CpoProcessInfos.TOTAL_UTF8_ENCODE_TIME] = time.time() - stime

        # Solve model and retrieve solution
        name = self.model.get_name()
        if is_number(ctx.params.TimeLimit):
            maxwait = ctx.params.TimeLimit + ctx.request_timeout + ctx.result_wait_extra_time
        else:
            maxwait = 0
        try:
            # Create job and start execution
            stime = time.time()
            client.create_cpo_job(name, cpostr, cmd)
            self.process_infos[CpoProcessInfos.TOTAL_DATA_SEND_TIME] = time.time() - stime

            # Start execution
            client.execute_job()

            # Wait job termination
            lgntf = None
            if self.log_enabled:
                lgntf = (lambda recs: self._add_log_data(''.join(r['message'] + "\n" for r in recs)))
            client.wait_job_termination(maxwait=maxwait, lognotif=lgntf)
            jinfo = client.get_info()

            # Trace response if required
            if ctx.is_log_enabled(3):
                ctx.log(3, "Job info:")
                for k in jinfo.keys():
                    ctx.log(3, k, " : ", jinfo[k])

            # Check failure
            fail = jinfo.get('failure', None)
            if fail is not None:
                raise CpoException(fail.get('message', "Unknown failure"))

            # Get solve status
            ssts = _get_solve_status(jinfo)
            if (cmd == "Solve") and (ssts not in _STATUS_WITH_RESULT):
                jsol = None
            else:
                # Get solution
                try:
                    stime = time.time()
                    jsol = client.get_attachment("solution.json")
                    self.process_infos[CpoProcessInfos.RESULT_RECEIVE_TIME] = time.time() - stime
                    self.process_infos[CpoProcessInfos.RESULT_DATA_SIZE] = len(jsol)
                    stime = time.time()
                    jsol = jsol.decode('utf-8')
                    self.process_infos[CpoProcessInfos.TOTAL_UTF8_DECODE_TIME] = time.time() - stime
                except Exception as e:
                    raise CpoException("Model solution access error: " + str(e))

            # Create solution structure
            msol = self._create_result_object(rclass, jsol)
            if not jsol:
                msol.solve_status = ssts
            sinfos = msol.solver_infos

            # Add solve details if no json solution
            detls = jinfo.get('details', None)
            if detls is not None:
                # Retrieve common infos
                for k in detls:
                    if not(('.' in k) or ('_' in k)):
                        sinfos[k] = _value_of(detls[k])
                # Retrieve special hard-coded values
                if isinstance(msol, CpoSolveResult) and msol.solution.objective_values is None:
                    oval = detls.get('PROGRESS_CURRENT_OBJECTIVE', None)
                    if oval is not None:
                        msol.solution.objective_values = [float(v) for v in oval.split(";")]
                sinfos.setdefault(CpoSolverInfos.NUMBER_OF_CONSTRAINTS,        int(detls.get('MODEL_DETAIL_CONSTRAINTS', 0)))
                sinfos.setdefault(CpoSolverInfos.NUMBER_OF_INTEGER_VARIABLES,  int(detls.get('MODEL_DETAIL_INTEGER_VARS', 0)))
                sinfos.setdefault(CpoSolverInfos.NUMBER_OF_INTERVAL_VARIABLES, int(detls.get('MODEL_DETAIL_INTERVAL_VARS', 0)))
                sinfos.setdefault(CpoSolverInfos.NUMBER_OF_SEQUENCE_VARIABLES, int(detls.get('MODEL_DETAIL_SEQUENCE_VARS', 0)))

            # Add solve time
            if CpoSolverInfos.SOLVE_TIME not in sinfos:
                sinfos[CpoSolverInfos.SOLVE_TIME] = (float(jinfo.get('endedAt', 0)) - float(jinfo.get('startedAt', 0))) / 1000

        finally:
            # Delete job
            if ctx.clean_job_after_solve:
                client.clean_job()

        # Return
        return msol



###############################################################################
##  Model solution building functions
###############################################################################


def _get_solve_status(jinfo):
    """ Build solve status from docloud job info
    Args:
        jinfo:  DOcloud job info
    """
    sts = jinfo.get('executionStatus', None)
    if sts == "INTERRUPTED":         return solution.SOLVE_STATUS_JOB_ABORTED
    if sts == "FAILED":              return solution.SOLVE_STATUS_JOB_FAILED

    sts = jinfo.get('solveStatus', None)
    if sts == "FEASIBLE_SOLUTION":   return solution.SOLVE_STATUS_FEASIBLE
    if sts == "INFEASIBLE_SOLUTION": return solution.SOLVE_STATUS_INFEASIBLE
    if sts == "OPTIMAL_SOLUTION":    return solution.SOLVE_STATUS_OPTIMAL

    return solution.SOLVE_STATUS_UNKNOWN


def _value_of(vstr):
    """ Build a python value from a string
    Args:
        vstr:  Value as a string
    Returns:
        Integer, float or string value
    """
    try:
        return int(vstr)
    except Exception:
        try:
            return float(vstr)
        except Exception:
            return vstr

