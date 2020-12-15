# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2018
# --------------------------------------------------------------------------

# gendoc: ignore


def make_solution(col_values, col_names, keep_zeros):
    # Return values as-is if not None, converting all values to float
    if col_values:
        return [float(i) for i in col_values]
    else:
        return []