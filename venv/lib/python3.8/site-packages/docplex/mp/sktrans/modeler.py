# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017
# --------------------------------------------------------------------------
# gendoc: ignore

from docplex.mp.utils import is_string

from docplex.mp.sktrans.docplex_mdlr import DOcplexModeler
from docplex.mp.sktrans.cpx_mdlr import CpxModeler


def make_modeler(modeler_name):
    if is_string(modeler_name):
        mkey = modeler_name.lower()
        if mkey == "cplex":
            return CpxModeler()
        elif mkey in {"docplex", "model"}:
            return DOcplexModeler()
        else:
            pass

    raise ValueError("expecting cplex|docplex, {0!r} was passed".format(modeler_name))
