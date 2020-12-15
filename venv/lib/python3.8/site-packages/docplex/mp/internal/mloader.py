# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2020
# --------------------------------------------------------------------------

try:
    import inspect
except ImportError:
    inspect = None

def import_class(qname):
    """ Imports a class from a full qualified name, e.g.e com.ibm.docplex.Engine

    Returns:
        a class type.
    """
    import importlib

    # Split class name
    rdotpos = qname.rfind('.')
    if rdotpos < 0:
        raise ValueError("Invalid class name '{0}' - expecting [<package.]+.<module>.<class_name>".format(qname))
    mname = qname[:rdotpos]
    clazzname = qname[rdotpos + 1:]
    if not clazzname:
        raise ValueError("Empty class name in full name: {0}".format(qname))

    # Load module
    try:
        module = importlib.import_module(mname)
    except ImportError as e:
        raise ImportError("Module {0} import error: {1!s}".format(mname, e))

    # Create and check class
    sclass = getattr(module, clazzname, None)
    if sclass is None:
        raise ValueError("Module '" + mname + "' does not contain a class '" + clazzname + "'")
    if inspect and not inspect.isclass(sclass):
        raise ValueError("Symbol {0} does not denote a class, type is {1}"
        .format(clazzname, type(sclass)))

    # this is a class
    return sclass
