# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2019
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module defines the class :class:`CpoCallback` that allows to retrieve events
that are sent by the CP Optimizer solver engine when running.

Any number of callbacks can be added to a solver using the method :meth:`docplex.cp.solver.solver.CpoSolver.add_callback`.
Callbacks can also be added on the model object using :meth:`docplex.cp.model.CpoModel.add_callback`

*New in version 2.10.*

Detailed description
--------------------
"""


#==============================================================================
#  Constants
#==============================================================================

# Call back events
EVENT_START_SOLVE               = "StartSolve"
EVENT_END_SOLVE                 = "EndSolve"
EVENT_START_EXTRACTION          = "StartExtraction"
EVENT_END_EXTRACTION            = "EndExtraction"
EVENT_START_INITIAL_PROPAGATION = "StartInitialPropagation"
EVENT_END_INITIAL_PROPAGATION   = "EndInitialPropagation"
EVENT_START_SEARCH              = "StartSearch"
EVENT_END_SEARCH                = "EndSearch"
EVENT_PERIODIC                  = "Periodic"
EVENT_OBJ_BOUND                 = "ObjBound"
EVENT_SOLUTION                  = "Solution"
EVENT_PROOF                     = "Proof"
EVENT_DESTRUCTION               = "Destruction"

ALL_CALLBACK_EVENTS = (EVENT_START_SOLVE, EVENT_END_SOLVE, EVENT_START_EXTRACTION, EVENT_END_EXTRACTION,
                       EVENT_START_INITIAL_PROPAGATION, EVENT_END_INITIAL_PROPAGATION, EVENT_START_SEARCH, EVENT_END_SEARCH,
                       EVENT_PERIODIC, EVENT_OBJ_BOUND, EVENT_SOLUTION, EVENT_DESTRUCTION,)


#==============================================================================
#  CPO callback class
#==============================================================================

class CpoCallback(object):
    """ CPO callback allows to be warned directly by the solver engine about different solving steps.

    This class is an 'abstract' class that must be extended by actual listener implementation.
    All method of this class are empty.

    *New in version 2.10.*
    """

    def invoke(self, solver, event, sres):
        """ Notify the callback about a solver event.

        This method is called every time an event is notified by the CPO solver.
        Associated to the event, the solver information is provided as a an object of class
        class:`~docplex.cp.solution.CpoSolveResult` that is instantiated with information available at this step.

        Args:
            solver: Originator CPO solver (object of class :class:`~docplex.cp.solver.solver.CpoSolver`)
            event:  Event id, string with value in ALL_CALLBACK_EVENTS
            sres:   Solver data, object of class :class:`~docplex.cp.solution.CpoSolveResult`
        """
        pass


