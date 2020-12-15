# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# ----------------------------------

# gendoc: ignore

import sys

from docplex.mp.error_handler import DOcplexException
from docplex.mp.utils import is_iterable, OutputStreamAdapter
from docplex.mp.compat23 import izip_longest

import json


class SolutionJSONEncoder(json.JSONEncoder):
    # INTERNAL

    def __init__(self, **kwargs):
        # pop kwargs I know before super()
        self.keep_zeros = kwargs.pop("keep_zeros", False)
        super(SolutionJSONEncoder, self).__init__(**kwargs)

    def default(self, solution):
        n = {'CPLEXSolution': self.encode_solution(solution)}
        return n

    def encode_solution(self, solution):
        n = {}
        n["version"] = "1.0"
        n["header"] = self.encode_header(solution)

        m = solution.model
        was_solved = m._has_solution()
        n["variables"] = self.encode_variables(solution, was_solved)
        lc = self.encode_linear_constraints(solution, was_solved)
        if len(lc) > 0:
            n["linearConstraints"] = lc
        qc = self.encode_quadratic_constraints(solution)
        if len(qc) > 0:
            n["quadraticConstraints"] = qc
        return n

    @staticmethod
    def encode_header(solution):
        n = {}
        n["problemName"] = solution.problem_name
        if solution.has_objective():
            n["objectiveValue"] = "{}".format(solution.objective_value)
        n["solved_by"] = solution.solved_by
        return n

    @classmethod
    def encode_linear_constraints(cls, solution, was_solved):
        n = []
        model = solution.model
        if was_solved:
            duals = []
            try:
                if not model._solved_as_mip():
                    duals = model.dual_values(model.iter_linear_constraints())
            except DOcplexException as dex:
                # in some cases, cplex may change the problem type at solve time.
                if "not available for integer" not in str(dex):
                    # ignore "not available for integer problems error, raise others.
                    raise
            slacks = model.slack_values(model.iter_linear_constraints())
        else:
            duals = []
            slacks = []
            model.warning("Model has not been solved, no dual and slack values are printed")

        for (ct, d, s) in izip_longest(model.iter_linear_constraints(),
                                       duals, slacks,
                                       fillvalue=None):
            c = {"name": ct.name,
                 "index": ct.index}
            if s:
                c["slack"] = s
            if d:
                c["dual"] = d
            n.append(c)
        return n

    @classmethod
    def encode_quadratic_constraints(cls, solution):
        n = []
        model = solution.model
        duals = []
        # RTC#37375
        # if not model._solves_as_mip():
        #     duals = model.dual_values(model.iter_quadratic_constraints())
        slacks = []
        was_solved = model._has_solution()
        if was_solved:
            slacks = model.slack_values(model.iter_quadratic_constraints())
        for (ct, d, s) in izip_longest(model.iter_quadratic_constraints(),
                                       duals, slacks,
                                       fillvalue=None):
            # basis status is not yet supported
            c = {"name": ct.name,
                 "index": ct.index}
            if s:
                c["slack"] = s
            if d:
                c["dual"] = d
            n.append(c)
        return n

    def encode_variables(self, sol, was_solved):
        model = sol.model
        n = []
        if was_solved:
            try:
                reduced_costs = [] if model._solved_as_mip() else model.reduced_costs(model.iter_variables())
            except DOcplexException as dex:
                if "not available for integer" in str(dex):
                    reduced_costs = []
                else:
                    raise
        else:
            reduced_costs = []
            model.warning("Model has not been solved, no reduced costs are printed")

        keep_zeros = sol._keep_zeros
        if self.keep_zeros:
            keep_zeros = keep_zeros or self.keep_zeros

        for (dvar, rc) in izip_longest(model.iter_variables(),
                                       reduced_costs,
                                       fillvalue=None):
            if not dvar.is_generated():  # 38934
                value = sol._get_var_value(dvar)
                if keep_zeros or value:
                    v = {"index": "{}".format(dvar.index),
                         "name": dvar.name,
                         "value": "{}".format(value)}
                    if rc is not None:
                        v["reducedCost"] = rc
                    n.append(v)
        return n


class SolutionJSONPrinter(object):
    json_extension = ".json"

    @classmethod
    def print_to_stream2(cls, out, solutions, indent=None, **kwargs):
        # solutions can be either a plain solution or a sequence or an iterator
        sol_to_print = list(solutions) if is_iterable(solutions) else [solutions]
        # encode all solutions in dict ready for json output
        encoder = SolutionJSONEncoder(**kwargs)
        solutions_as_dict = [encoder.default(sol) for sol in sol_to_print]
        # use an output stream adapter for py2/py3 and str/unicode compatibility
        osa = OutputStreamAdapter(out)
        if len(sol_to_print) == 1:  # if only one solution, use at root node
            osa.write(json.dumps(solutions_as_dict[0], indent=indent))
        else:  # for multiple solutions, we want a "CPLEXSolutions" root
            osa.write(json.dumps({"CPLEXSolutions": solutions_as_dict}, indent=indent))

    @classmethod
    def print_to_stream(cls, solutions, out, extension=json_extension, indent=None, **kwargs):
        if out is None:
            # prints on standard output
            cls.print_to_stream2(sys.stdout, solutions, indent=indent, **kwargs)
        elif isinstance(out, str):
            # a string is interpreted as a path name
            path = out if out.endswith(extension) else out + extension
            with open(path, "w") as of:
                cls.print_to_stream(solutions, of, indent=indent, **kwargs)
                # print("* file: %s overwritten" % path)
        else:
            cls.print_to_stream2(out, solutions, indent=indent, **kwargs)

