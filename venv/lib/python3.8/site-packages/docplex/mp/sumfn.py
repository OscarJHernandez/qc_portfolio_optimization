# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2017
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.utils import is_iterable, is_indexable
from docplex.mp.error_handler import docplex_fatal


def _docplex_extract_model(e, do_raise=True):
    try:
        model = e.model
        return model
    except AttributeError:
        if do_raise:
            raise docplex_fatal("object has no model attribute: {0!s}", e)
        else:
            return None


def docplex_sum(x_seq):
    if is_iterable(x_seq):
        if not x_seq:
            return 0
        elif isinstance(x_seq, dict):
            return _docplex_sum_with_seq(x_seq.values())
        elif is_indexable(x_seq):
            return _docplex_sum_with_seq(x_seq)
        else:
            return _docplex_sum_with_seq(list(x_seq))
    elif x_seq:
        mdl = _docplex_extract_model(x_seq, do_raise=False)
        return mdl.to_linear_expr(x_seq) if mdl else x_seq
    else:
        return 0


def _docplex_sum_with_seq(x_list):
    shared_model = None
    for x in x_list:
        try:
            model = x.model
            if not shared_model:
                shared_model = model
            else:
                if model != shared_model:
                    docplex_fatal("Cannot mix objects belonging to different models")
        except AttributeError:
            pass
    if shared_model:
        return shared_model.sum(x_list)
    else:
        # try a python sum ?
        return sum(x_list)
