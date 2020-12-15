# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017
# --------------------------------------------------------------------------

# gendoc: ignore

try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None


def new_empty_dataframe():
    return DataFrame([])


def make_solution(col_values, col_names, keep_zeros):
    if col_values:
        # fetch all values in one call.
        dd = {'value': col_values}
        if col_names is not None:
            assert len(col_names) == len(col_values)
            dd['name'] = col_names
        ret = DataFrame(dd)
        if not keep_zeros:
            ret = ret[ret['value'] != 0]
            ret = ret.reset_index(drop=True)

        return ret
    else:
        return new_empty_dataframe()