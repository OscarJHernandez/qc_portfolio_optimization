# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import json

class JSONInfeasibilityHandler(object):

    def __init__(self, json_as_string):
        if json_as_string is None:
            self._infeasibilities = {}
        else:
            json_infs = json.loads(json_as_string, parse_constant='utf-8')['CPLEXInfeasibilities']
            self._infeasibilities = {int(infeas['index']): float(infeas['value']) for infeas in json_infs['rows']}



    def get_infeasibilities(self):
        return self._infeasibilities
