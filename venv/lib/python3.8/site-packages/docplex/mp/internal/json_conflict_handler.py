# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import json
from docplex.mp.constants import ConflictStatus


class JSONConflictHandler(object):
    '''A class that provides utilities to process the JSON conflict from DOcplexcloud
    '''

    # json keys
    JSON_LINEAR_CTS_KEY = 'linearConstraints'

    row_conflict_by_status = {'excluded': ConflictStatus.Excluded,
                              'possible': ConflictStatus.Possible_member,
                              'member': ConflictStatus.Member}
    grp_conflict_by_status = row_conflict_by_status
    col_conflict_by_status = {'excluded': ConflictStatus.Excluded,
                              'possible': ConflictStatus.Possible_member,
                              'possible_lb': ConflictStatus.Possible_member_lower_bound,
                              'possible_ub': ConflictStatus.Possible_member_upper_bound,
                              'member': ConflictStatus.Member,
                              'member_lb': ConflictStatus.Member_lower_bound,
                              'member_ub': ConflictStatus.Member_upper_bound}

    def __init__(self, conflict_string, grps_dict, has_conflict=None):
        """Initialize a new JSONConflictHandler

        This handler is initialized with a json fragment with the CPLEXConflict.

        Args:
            conflict_string: The json text containing a CPLEXSolution
            grps_dict: dictionary of groups provided to the DOcplexcloud job, with keys = groups index
            has_conflict: if set to True or False, forces the has_conflict status. If None, this
                is True if json is not None.
        """
        self.has_conflict = bool(conflict_string) if has_conflict is None else has_conflict
        if conflict_string is not None:
            self.json = json.loads(conflict_string, parse_constant='utf-8')['CPLEXConflict']
        else:
            self.json = None

        #
        self._grps_dict = grps_dict
        # used to store data from the json
        self.__cols = None
        self.__rows = None
        self.__grps = None

    def _get_grps(self):
        if self.__grps is None:
            self.__grps = self.json.get('grp', [])
        return self.__grps

    def get_conflict_grps_list(self):
        all_grps = self._get_grps()
        conflict_grps_list = [(int(g['index']),
                               self._grps_dict[int(g['index'])],
                               self.grp_conflict_by_status.get(g['status'], None))
                              for g in all_grps]
        return conflict_grps_list

    def _get_cols(self):
        if self.__cols is None:
            self.__cols = self.json.get('col', [])
        return self.__cols

    def _get_rows(self):
        if self.__rows is None:
            self.__rows = self.json.get('row', [])
        return self.__rows
