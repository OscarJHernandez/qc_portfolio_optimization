# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017-2019
# --------------------------------------------------------------------------

# gendoc: ignore

"""
This package contains one mixin class `ModelCallbackMixin` which is intended to link
CPLEX legacy callbacks and DOcplex models.

In order to define your own custom callbacks, follow these steps:

    - define a new class that derives -both- from `ModelCallbackMixin` and from a CPLEX callback class,
      in this order (mixin class first)
    - the constructor for this new class must take an `env` parameter,
        and call both `ModelCallbackMixin.__init__()` and
        the cplex callback __init__() method, with the `env` parameter.

    Here is an example of a definition of a custom callback class:

        class RoundDown(ModelCallbackMixin, cplex.callbacks.HeuristicCallback):
            def __init__(self, env):
                cplex.callbacks.HeuristicCallback.__init__(self, env)
                ModelCallbackMixin.__init__(self)

    Once the custom callback class is defined, attaching a callback to a Model instance is done
    using `Model.register_callback`, by passing the -type- of the callback.
    For example:

        rd_instance = mdl.register_callback(RoundDown)

    returns an instance of a RoundDown callback, which is linked to the model instance.

    The mixin class provides utility methods to navigate between indices used by CPLEX native
    callbacks and DOcplex modeling objects, for example the `index_to_var` method converts a varfiable index
    to an instance of class `docplex.mp.Var`.



"""