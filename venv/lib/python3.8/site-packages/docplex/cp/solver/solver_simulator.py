# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module implements a simulator of model solver.
It is mainly used for testing.
"""

from docplex.cp.expression import *
from docplex.cp.solution import *
import docplex.cp.solver.solver as solver
from docplex.cp.utils import *

import random

###############################################################################
##  Constants
###############################################################################

# Max value used by solution simulator
MAX_SIMULATED_VALUE = 10000

# Max number of solutions when using next() iterator
MAX_ITERATED_SOLUTIONS = 6


###############################################################################
##  Public classes
###############################################################################

class CpoSolverSimulatorFail(solver.CpoSolverAgent):
    """ CPO solver simulator that always fail (status unfeasible) """
    
    def __init__(self, solver, params, context):
        """ Create a solver simulator

        Args:
            solver:   Parent solver
            params:   Solving parameters
            context:  Solver context
        """
        super(CpoSolverSimulatorFail, self).__init__(solver, params, context)


    def solve(self):
        """ Solve the model.

        Returns:
            Model solution expressed as CpoModelSolution
        """
        # Warn about simulator
        print("WARNING: Solver simulator always returns fail")

        # Build fake infeasible solution
        msol = CpoSolveResult(self.model)
        msol.solve_status = SOLVE_STATUS_INFEASIBLE
        return msol


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        Given an infeasible model, the conflict refiner can identify conflicting constraints and variable domains
        within the model to help you identify the causes of the infeasibility.
        In this context, a conflict is a subset of the constraints and/or variable domains of the model
        which are mutually contradictory.
        Since the conflict is minimal, removal of any one of these constraints will remove that
        particular cause for infeasibility.
        There may be other conflicts in the model; consequently, repair of a given conflict
        does not guarantee feasibility of the remaining model.

        Returns:
            Conflict result,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        # Warn about simulator
        print("WARNING: Solver simulator always returns fail")
        # Build fake infeasible solution
        msol = CpoRefineConflictResult(self.model)
        msol.solve_status = SOLVE_STATUS_INFEASIBLE
        return msol



class CpoSolverSimulatorRandom(solver.CpoSolverAgent):
    """ CPO solver simulator that generates a random solution """

    def __init__(self, solver, params, context):
        """ Create a solver simulator

        Args:
            solver:   Parent solver
            params:   Solving parameters
            context:  Solver context
        """
        super(CpoSolverSimulatorRandom, self).__init__(solver, params, context)
        self.max_iterator_solutions = random.randint(0, MAX_ITERATED_SOLUTIONS)
        self.solution_count = 0


    def solve(self):
        """ Solve the model.

        Returns:
            Model solve result (object of class CpoSolveResult)
        """
        # Warn about simulator
        print("WARNING: Solver simulator returns a random solution")

        # Force generation of CPO format if required (for testing purpose only)
        if self.context.create_cpo:
            self._get_cpo_model_string()

        # Build fake feasible solution
        return self._generate_solution()


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        Given an infeasible model, the conflict refiner can identify conflicting constraints and variable domains
        within the model to help you identify the causes of the infeasibility.
        In this context, a conflict is a subset of the constraints and/or variable domains of the model
        which are mutually contradictory.
        Since the conflict is minimal, removal of any one of these constraints will remove that
        particular cause for infeasibility.
        There may be other conflicts in the model; consequently, repair of a given conflict
        does not guarantee feasibility of the remaining model.

        Returns:
            Conflict result,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        # Build fake conflict
        csol = CpoRefineConflictResult(self.model)
        vars = self.model.get_all_variables()
        if len(vars) > 0:
            for i in range(1 + (len(vars) // 2)):
                csol.member_variables.append(vars[random.randint(0, len(vars) - 1)])
        ctrs = self.model.get_all_expressions()
        if len(ctrs) > 0:
            for i in range(1 + (len(ctrs) // 2)):
                csol.member_constraints.append(ctrs[random.randint(0, len(ctrs) - 1)][0])

        return csol


    def start_search(self):
        """ Start a new search. Solutions are retrieved using method search_next().

        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        print("WARNING: Solver simulator returns a random list of solutions")
        # Force generation of CPO format if required (for testing purpose only)
        if self.context.create_cpo:
            self._get_cpo_model_string()
        self.solution_count = 0


    def search_next(self):
        """ Search the next available solution.

        Returns:
            Next solution (object of class CpoSolveResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        if self.solution_count >= self.max_iterator_solutions:
            # Generate last solution
            msol = CpoSolveResult(self.model)
            msol.solve_status = SOLVE_STATUS_FEASIBLE
            msol.fail_status = FAIL_STATUS_SEARCH_COMPLETED
            return msol
        else:
           # Generate next solution
           self.solution_count += 1
           return self._generate_solution()


    def end_search(self):
        """ End current search.

        Returns:
            Last (fail) model solution with last solve information (type CpoSolveResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        pass


    def _generate_solution(self):
        """ Generate a random solution complying to the model

        Returns:
            Random model solution expressed as CpoModelSolution
        """
        # Build fake feasible solution
        ssol = CpoSolveResult(self.model)
        ssol.solve_status = SOLVE_STATUS_FEASIBLE
        msol = ssol.solution

        # Generate objective
        x = self.model.get_optimization_expression()
        if x is not None:
            # Determine number of values
            x = x.children[0]
            if x.type.is_array:
                nbval = len(x.value)
            else:
                nbval = 1
            ovals = []
            for i in range(nbval):
                ovals.append(random.randint(0, MAX_SIMULATED_VALUE))
            msol.objective_values = ovals

        # Generate a solution for each variable
        allvars = self.model.get_all_variables()
        seqvars = []
        for var in allvars:
            if isinstance(var, CpoIntVar):
                vsol = CpoIntVarSolution(var, _random_value_in_complex_domain(var.get_domain()))
                msol.add_var_solution(vsol)

            elif isinstance(var, CpoIntervalVar):
                # Generate presence
                if var.is_absent():
                    present = False
                elif var.is_present():
                    present = True
                else:
                    present = (random.random() > 0.3)
                # Generate start and end
                dom = _common_interval_domain(var.get_start(), var.get_end())
                if dom[0] == dom[1]:
                    start = end = dom[0]  # Allow zero-length interval ?
                else:
                    start = end = 0
                    while start >= end:
                        start = _random_value_in_interval_domain(dom)
                        end = _random_value_in_interval_domain(dom)
                # Generate size
                size = _random_value_in_interval_domain(var.get_size())
                # Add variable to solution
                vsol = CpoIntervalVarSolution(var, present, start, end, size)
                msol.add_var_solution(vsol)

            elif isinstance(var, CpoStateFunction):
                # Build list of steps
                lsteps = []
                cstart = 0
                while cstart < MAX_SIMULATED_VALUE:
                    size = random.randint(1, MAX_SIMULATED_VALUE / 10)
                    lsteps.append((cstart, cstart + size, random.randint(0, 10)))
                    cstart += size
                vsol = CpoStateFunctionSolution(var, lsteps)
                msol.add_var_solution(vsol)

            elif isinstance(var, CpoSequenceVar):
                seqvars.append(var)

        # Generate a solution for sequence variables (done after all vars to access their solutions)
        for var in seqvars:
            # Build sequence or results
            lvres = []
            for v in var.get_interval_variables():
                lvres.append(msol.get_var_solution(v))
            random.shuffle(lvres)
            vsol = CpoSequenceVarSolution(var, lvres)
            msol.add_var_solution(vsol)

        # Return
        return ssol


###############################################################################
##  Private methods
###############################################################################

def _random_value_in_interval_domain(dom):
    """ Determine a random integer value in a domain expressed as a single interval

    Args:
        dom:  Interval domain (couple of values)
    Returns:
        Random value in this domain
    """
    return random.randint(dom[0], min(dom[1], dom[0] + MAX_SIMULATED_VALUE))

def _random_value_in_complex_domain(dom):
    """ Determine a random integer value in a domain

    Args:
        dom:  Value domain, list of integers or interval tuples
    Returns:
        Random value in this domain
    """
    # First select a domain element
    dlm = dom[random.randint(0, len(dom) - 1)]
    if is_int(dlm):
        return dlm
    return _random_value_in_interval_domain(dlm)

def _common_interval_domain(dom1, dom2):
    """ Determine the interval domain that is common to two interval domains

    Args:
        dom1:  First interval domain
        dom2:  Second interval domain
    Returns:
        Common domain
    """
    return max(dom1[0], dom2[0]), min(dom1[1], dom2[1])

