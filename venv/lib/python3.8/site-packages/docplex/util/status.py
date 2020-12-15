# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2018
# --------------------------------------------------------------------------

# gendoc: ignore

'''Since version 2.7, docplex can be used even if docloud is not available.

When docloud is not installed, cloud solving is disabled. Additionnaly,
we redefine our now JobSolveStatus class in this case::

  class JobSolveStatus(Enum):
        """ Job solve status values.

        This `Enum` is used to convert job solve status string values into an
        enumeration::

            >>> job = client.get_job(jobid)
            >>> solveStatus = JobSolveStatus[job['solveStatus']]

        Attributes:
            UNKNOWN: The algorithm has no information about the solution.
            FEASIBLE_SOLUTION: The algorithm found a feasible solution.
            OPTIMAL_SOLUTION: The algorithm found an optimal solution.
            INFEASIBLE_SOLUTION: The algorithm proved that the model is infeasible.
            UNBOUNDED_SOLUTION: The algorithm proved the model unbounded.
            INFEASIBLE_OR_UNBOUNDED_SOLUTION: The model is infeasible or unbounded.
        """
        UNKNOWN = 0
        FEASIBLE_SOLUTION = 1
        OPTIMAL_SOLUTION = 2
        INFEASIBLE_SOLUTION = 3
        UNBOUNDED_SOLUTION = 4
        INFEASIBLE_OR_UNBOUNDED_SOLUTION = 5


If docloud is available, use the `JobSolveStatus <https://api-swagger-oaas.docloud.ibmcloud.com/api_swagger/pythondoc/index.html?cm_mc_uid=82840450017915192306798&cm_mc_sid_50200000=77001241530691867305#docloud.status.JobSolveStatus>`_ from docloud.
'''
try:
    from docloud.status import JobSolveStatus
except ImportError:
    from enum import Enum

    class JobSolveStatus(Enum):
        """ Job solve status values.

        This `Enum` is used to convert job solve status string values into an
        enumeration::

            >>> job = client.get_job(jobid)
            >>> solveStatus = JobSolveStatus[job['solveStatus']]

        Attributes:
            UNKNOWN: The algorithm has no information about the solution.
            FEASIBLE_SOLUTION: The algorithm found a feasible solution.
            OPTIMAL_SOLUTION: The algorithm found an optimal solution.
            INFEASIBLE_SOLUTION: The algorithm proved that the model is infeasible.
            UNBOUNDED_SOLUTION: The algorithm proved the model unbounded.
            INFEASIBLE_OR_UNBOUNDED_SOLUTION: The model is infeasible or unbounded.
        """
        UNKNOWN = 0
        FEASIBLE_SOLUTION = 1
        OPTIMAL_SOLUTION = 2
        INFEASIBLE_SOLUTION = 3
        UNBOUNDED_SOLUTION = 4
        INFEASIBLE_OR_UNBOUNDED_SOLUTION = 5
