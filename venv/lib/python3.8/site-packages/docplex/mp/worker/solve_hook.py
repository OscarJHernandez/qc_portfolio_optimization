# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore
from six import iteritems


class _SolveHook(object):
    # INTERNAL
    def __init__(self):
        pass  # pragma: no cover

    def notify_start_solve(self, mdl, model_statistics):
        """ Notifies the start of a solve.

        Args:
            model_statistics: A dictionary of string->values with various data attributes of the model.

        """
        pass  # pragma: no cover

    def notify_end_solve(self, mdl, has_solution, status, obj, var_value_dict):
        """ Notifies the end of a solve.

        Args:
            has_solution: Boolean, True if solve returned a solution.
            status: An enumerated value of type JobSolveStatus.
            obj: The objective value if solved ok, else irrelevant.
            var_value_dict: A dictionary of variable names to values in the solution.
        """
        pass  # pragma: no cover


class TraceSolveHook(_SolveHook):
    def notify_start_solve(self, mdl, stats):
        print("-> start solve")
        stats.print_information()

    def notify_end_solve(self, mdl, has_solution, status, obj, var_value_dict):
        if has_solution:
            print("<- solve succeeds, status={0}, obj={1}".format(status, obj))
            for vn, vv in iteritems(var_value_dict):
                print("  - var \"{0:s}\" = {1!s}".format(vn, vv))
        else:
            print("<- solve fails, status={0}".format(status))
