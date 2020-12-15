# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2016
# --------------------------------------------------------------------------

# gendoc: ignore


def as_df(what, **kwargs):
    '''
    Returns a `pandas.DataFrame` representation of an object.

    Attributes:
        what: The object to represent as an object.
        **kwargs: Additional parameters for the conversion.
    Returns:
        A `pandas.DataFrame` representation of an object or None if a
        representation could not be found.
    '''
    try:
        return what.__as_df__(**kwargs)
    except AttributeError:
        return None
