# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import json


class JSONSolutionHandler(object):
    '''A class that provides utilities to process the JSON solutions from DOcplexcloud
    '''

    # json keys
    JSON_LINEAR_CTS_KEY = 'linearConstraints'

    def __init__(self, solution_string, has_solution=None):
        '''Initialize a new JSONSolutionHandler

        This handler is initialized with a json fragment with the CPLEXSolution.

        Args:
            solution_string: The json text containing a CPLEXSolution
            has_solution: if set to True or False, forces the has_solution status. If None, this
                is True if json is not None.
        '''
        self.has_solution = bool(solution_string) if has_solution is None else has_solution
        if solution_string is not None:
            self.json = json.loads(solution_string, parse_constant='utf-8')['CPLEXSolution']
        else:
            self.json = None
        # used to store data from the json
        self.__vars = None

    def _getvars(self):
        if self.__vars is None:
            self.__vars = self.json.get('variables', [])
        return self.__vars

    def get_variable_attr_map(self, json_attr_name):
        assert json_attr_name
        if not self.json:
            return {}
        else:
            all_vars = self._getvars()
            attr_map = {int(v['index']): float(v[json_attr_name]) for v in all_vars}
            return attr_map

    def cplex_index_name_map(self):
        json_res = self.json
        if json_res:
            return {int(v['index']): v['name'] for v in self._getvars()}
        else:
            return {}

    def variable_results(self):
        if not self.json:
            return {}, {}
        else:
            all_vars = self._getvars()
            value_map = {int(v['index']): float(v['value']) for v in all_vars}
            if self.is_mip():
                rc_map = {}
            else:
                try:
                    rc_map = {int(v['index']): float(v['reducedCost']) for v in all_vars}
                except KeyError:
                    rc_map = {}
            return value_map, rc_map

    def constraint_results(self):
        if not self.json or self.is_mip():
            return {}, {}
        else:
            lincst_key = self.JSON_LINEAR_CTS_KEY
            if lincst_key in self.json:
                all_linear_cts = self.json[lincst_key]
                try:
                    dual_map = {int(v['index']): float(v['dual']) for v in all_linear_cts}
                except KeyError:
                    dual_map = {}
                try:
                    slack_map = {int(v['index']): float(v['slack']) for v in all_linear_cts}
                except KeyError:
                    slack_map = {}
            else:
                dual_map = slack_map = {}
            return dual_map, slack_map

    def constraint_slacks(self, linkey=JSON_LINEAR_CTS_KEY):
        self_json = self.json
        slack_map = {}
        if self_json and linkey in self_json:
            all_linear_cts = self_json[linkey]
            try:
                slack_map = {int(v['index']): float(v['slack']) for v in all_linear_cts}
            except KeyError:
                pass
        return slack_map

    def is_mip(self):
        return self.json and self.json['header']['solutionMethodString'] == 'mip'

    def _check_nonempty_json(self):
        if not self.json:
            raise ValueError("* empty JSON result!")

    def get_status_id(self):
        self._check_nonempty_json()
        return int(self.json['header']['solutionStatusValue'])

    def get_objective(self):
        self._check_nonempty_json()
        return float(self.json['header']['objectiveValue'])

    def variable_values(self):
        return self.get_variable_attr_map('value')

    def variable_reduced_costs(self):
        return self.get_variable_attr_map('reducedCost')
