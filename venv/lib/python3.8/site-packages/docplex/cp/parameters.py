# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module handles the public parameters that can be assigned to CP Optimizer
to configure the solving of a model.

The class `CpoParameters` contains the list of modifiable parameters
expressed as properties with getters and setters.
For the parameters that require special values, those values are given as constants.

Changing the value of a parameter can be done in multiple ways.
For example, the `TimeLimit` can be set to 60s with:

   * `params.TimeLimit = 60`
   * `params.set_TimeLimit(60)`
   * `params['TimeLimit'] = 60`
   * `params.set_attribute('TimeLimit', 60)`

Retrieving the value of a parameter can be done in the same way using:

   * `v = params.TimeLimit`
   * `v = params.get_TimeLimit()`
   * `v = params['TimeLimit']`
   * `v = params.get_attribute('TimeLimit')`

If a parameter is not set, the value returned by the first two access forms is None.
The last access form (element of a dictionary) raises an exception.

Setting a parameter value to None is equivalent to force its default value.
This may be for example useful to reset at solve time a parameter that has been set at model level.

Getting the list of all parameters that have been changed can be done by calling the method `keys()`.

Note that the *PEP8* naming convention is not applied here, to keep parameter names as they
are in the solver, so that they can be referenced in solver logs.


Summary of parameters
---------------------

The following list gives the summary of all public parameters.

**Display and output**

 * :attr:`~CpoParameters.LogVerbosity`: Determines the verbosity of the search log.
   The value is a symbol in ['Quiet', 'Terse', 'Normal', 'Verbose']. Default value is 'Normal'.
 * :attr:`~CpoParameters.LogPeriod`: Controls how often the log information is displayed.
   The value is an integer strictly greater than 0. Default value is 1000.
 * :attr:`~CpoParameters.WarningLevel`: Level of warnings issued by CP Optimizer when a solve is launched.
   The value is an integer in [0..3]. Default value is 2.
 * :attr:`~CpoParameters.PrintModelDetailsInMessages`: Controls printing of additional information on error and warning messages.
   The value is a symbol in ['On', 'Off']. Default value is 'On'.
 * :attr:`~CpoParameters.ModelAnonymizer`: Controls anonymization of a model dumped via dumpModel.
   The value is a symbol in ['On', 'Off']. Default value is 'Off'.
 * :attr:`~CpoParameters.UseFileLocations`: Controls whether location information (file, line) is added to the model.
   The value is a symbol in ['On', 'Off']. Default value is 'On'.
 * :attr:`~CpoParameters.LogSearchTags`: Controls the activation of search failure tags.
   The value is a symbol in ['On', 'Off']. Default value is 'Off'.
 * :attr:`~CpoParameters.KPIDisplay`: Controls the display of the KPI values in the log.
   The value is a symbol in ['SingleLine', 'MultipleLines']. Default value is 'SingleLine'.
   
**Presolve**

 * :attr:`~CpoParameters.Presolve`: Controls the presolve of the model to produce more compact formulations and to achieve more domain reduction.

**Optimality tolerances**

 * :attr:`~CpoParameters.OptimalityTolerance`: Absolute tolerance on the objective value for optimization models.
   The value is a positive float. Default value is 1e-09.
 * :attr:`~CpoParameters.RelativeOptimalityTolerance`: Relative tolerance on the objective value for optimization models.
   The value is a non-negative float. Default value is 0.0001.

**Search control**

 * :attr:`~CpoParameters.Workers`: Number of workers to run in parallel to solve the model.
   The value is a positive integer. Default value is Auto.
 * :attr:`~CpoParameters.SearchType`: Type of search that is applied when solving a problem.
   The value is a symbol in ['DepthFirst', 'Restart', 'MultiPoint', 'IterativeDiving', 'Auto']. Default value is 'Auto'.
 * :attr:`~CpoParameters.RandomSeed`: Seed of the random generator used by search strategies.
   The value is a non-negative integer. Default value is 0.
 * :attr:`~CpoParameters.RestartFailLimit`: Controls the number of failures that must occur before restarting search.
   The value is an integer greater than 0. Default value is 100.
 * :attr:`~CpoParameters.RestartGrowthFactor`: Controls the increase of the number of failures between restarts.
   The value is a float greater or equal to 1. Default value is 1.15.
 * :attr:`~CpoParameters.DynamicProbing`: Controls probing carried out during search.
   The value is a symbol in ['On', 'Off', 'Auto']. Default value is 'Auto'.
 * :attr:`~CpoParameters.DynamicProbingStrength`: Controls the effort which is dedicated to dynamic probing.
   The value is a float in [0.001..1000]. Default value is 0.03.
 * :attr:`~CpoParameters.MultiPointNumberOfSearchPoints`: Controls the number of solutions manipulated by the multi-point search algorithm.
   The value is an integer strictly greater than 1. Default value is 30.
 * :attr:`~CpoParameters.TemporalRelaxation`: Advanced parameter can be used to control the usage of a temporal relaxation.
   Possible values are 'On' or 'Off'.
 * :attr:`~CpoParameters.FailureDirectedSearch`: Controls usage of failure-directed search.
   Possible values are 'On' or 'Off'.
 * :attr:`~CpoParameters.FailureDirectedSearchEmphasis`: Controls how much time CP Optimizer invests into failure-directed search once it is started.
   The value is a non-negative integer, or None (default) that does not set any limit.
 * :attr:`~CpoParameters.FailureDirectedSearchMaxMemory`: Controls the maximum amount of memory available to failure-directed search
   The value is a non-negative integer, or None that does not set any limit. Default value is 104857600.
 * :attr:`~CpoParameters.AutomaticReplay`: Low-level control of the behavior of *solve()* and *next()*.
   Possible values are 'On' or 'Off'.

**Search limits**

 * :attr:`~CpoParameters.TimeLimit`: Limits the CPU or elapsed time spent solving before terminating a search.
   The value is a non-negative float, or None (default) that does not set any limit.
 * :attr:`~CpoParameters.TimeMode`: Defines how time is measured in CP Optimizer.
   The value is a symbol in ['CPUTime', 'ElapsedTime']. Default value is 'ElapsedTime'.
 * :attr:`~CpoParameters.FailLimit`: Limits the number of failures that can occur before terminating the search.
   The value is a non-negative integer, or None (default) that does not set any limit.
 * :attr:`~CpoParameters.ChoicePointLimit`: Limits the number of choice points that are created before terminating a search.
   The value is a non-negative integer, or None (default) that does not set any limit.
 * :attr:`~CpoParameters.BranchLimit`: Limits the number of branches that are made before terminating a search.
   The value is a non-negative integer, or None (default) that does not set any limit.
 * :attr:`~CpoParameters.SolutionLimit`: Limits the number of feasible solutions that are found before terminating a search.
   The value is a non-negative integer, or None (default) that does not set any limit.

**Inference levels for constraint propagation**

For all following attributes, possible values are 'Default', 'Low', 'Basic', Medium' or 'Extended'. Default value is 'Default'.

 * :attr:`~CpoParameters.DefaultInferenceLevel`: General inference level for constraints whose particular inference level is Default.
 * :attr:`~CpoParameters.AllDiffInferenceLevel`: Inference level for every constraint *allDiff*.
 * :attr:`~CpoParameters.DistributeInferenceLevel`: Inference level for every constraint *Distribute*.
 * :attr:`~CpoParameters.CountInferenceLevel`: Inference level for every constraint *Count* extracted to the invoked CP instance.
 * :attr:`~CpoParameters.CountDifferentInferenceLevel`: Inference level for every constraint *CountDifferent*.
 * :attr:`~CpoParameters.SequenceInferenceLevel`: Inference level for every constraint *Sequence*.
 * :attr:`~CpoParameters.AllMinDistanceInferenceLevel`: Inference level for every constraint *allMinDistance*.
 * :attr:`~CpoParameters.ElementInferenceLevel`: Inference level for every *element* constraint.
 * :attr:`~CpoParameters.PrecedenceInferenceLevel`: Inference level for precedence constraints between interval variables.
 * :attr:`~CpoParameters.IntervalSequenceInferenceLevel`: Inference level for the maintenance of the domain of every interval sequence variable.
 * :attr:`~CpoParameters.NoOverlapInferenceLevel`: Inference level for every constraint NoOverlap extracted.
 * :attr:`~CpoParameters.CumulFunctionInferenceLevel`: Inference level for constraints on expression *CumulFunctionExpr*.
 * :attr:`~CpoParameters.StateFunctionInferenceLevel`: Inference level for constraints on state functions *StateFunction*.

**Conflict refiner**

 * :attr:`~CpoParameters.ConflictRefinerTimeLimit`: Limits the CPU time spent before terminating the conflict refiner.
   The value is a non-negative float, or None (default) that does not set any limit.
 * :attr:`~CpoParameters.ConflictRefinerIterationLimit`: Limits the number of iterations that are made before terminating the conflict refiner.
   The value is a non-negative integer, or None (default) that does not set any limit.
 * :attr:`~CpoParameters.ConflictRefinerBranchLimit`: Limits the total number of branches that are made before terminating the conflict refiner.
   The value is a non-negative integer, or None (default) that does not set any limit.
 * :attr:`~CpoParameters.ConflictRefinerFailLimit`: Limits the total number of failures that can occur before terminating the conflict refiner.
   The value is a non-negative integer, or None (default) that does not set any limit.
 * :attr:`~CpoParameters.ConflictRefinerOnVariables`: Specifies whether the conflict refiner should refine variables domains.
   The value is a symbol in ['On', 'Off']. Default value is 'Off'.


Private parameters
------------------

A private parameter can only be set using the method :meth:`~CpoParameters.set_attribute`, giving its name as a string and its value.
In this case, there is no local checking of the validity of the parameter or its value at modeling time.
If there is any error, it is detected only at solve time.


Detailed description
--------------------
"""

from docplex.cp.utils import Context, is_int, is_number
import warnings


###############################################################################
## Public constants
###############################################################################

# Set of all public parameter names
PUBLIC_PARAMETER_NAMES = \
    {"AllDiffInferenceLevel", "AllMinDistanceInferenceLevel", "AutomaticReplay", "BranchLimit", "ChoicePointLimit",
     "ConflictRefinerBranchLimit", "ConflictRefinerFailLimit", "ConflictRefinerIterationLimit",
     "ConflictRefinerOnVariables", "ConflictRefinerTimeLimit","CountDifferentInferenceLevel", "CountInferenceLevel",
     "CumulFunctionInferenceLevel", "DefaultInferenceLevel", "DistributeInferenceLevel", "DynamicProbing",
     "DynamicProbingStrength", "ElementInferenceLevel", "FailLimit", "FailureDirectedSearch",
     "FailureDirectedSearchEmphasis", "FailureDirectedSearchMaxMemory", "IntervalSequenceInferenceLevel", "LogPeriod",
     "LogSearchTags", "LogVerbosity", "ModelAnonymizer", "MultiPointNumberOfSearchPoints", "NoOverlapInferenceLevel",
     "OptimalityTolerance", "PrecedenceInferenceLevel", "Presolve", "PrintModelDetailsInMessages", "RandomSeed",
     "RelativeOptimalityTolerance", "RestartFailLimit", "RestartGrowthFactor", "SearchType", "SequenceInferenceLevel",
     "SolutionLimit", "StateFunctionInferenceLevel", "TemporalRelaxation", "TimeLimit", "TimeMode", "UseFileLocations",
     "WarningLevel", "Workers", "KPIDisplay"}

# Set of all private but accepted parameter names
PRIVATE_PARAMETER_NAMES = {"ObjectiveLimit",}

# Set of all authorized parameter names
ALL_PARAMETER_NAMES = PUBLIC_PARAMETER_NAMES | PRIVATE_PARAMETER_NAMES

# Symbolic parameter values
VALUE_AUTO             = 'Auto'
VALUE_OFF              = 'Off'
VALUE_ON               = 'On'
VALUE_DEFAULT          = 'Default'
VALUE_LOW              = 'Low'
VALUE_BASIC            = 'Basic'
VALUE_MEDIUM           = 'Medium'
VALUE_EXTENDED         = 'Extended'
VALUE_QUIET            = 'Quiet'
VALUE_TERSE            = 'Terse'
VALUE_NORMAL           = 'Normal'
VALUE_VERBOSE          = 'Verbose'
VALUE_DEPTH_FIRST      = 'DepthFirst'
VALUE_RESTART          = 'Restart'
VALUE_MULTI_POINT      = 'MultiPoint'
VALUE_ITERATIVE_DIVING = 'IterativeDiving'
VALUE_DIVERSE          = 'Diverse'
VALUE_CPU_TIME         = 'CPUTime'
VALUE_ELAPSED_TIME     = 'ElapsedTime'
VALUE_SINGLE_LINE      = 'SingleLine'
VALUE_MULTIPLE_LINES   = 'MultipleLines'

# Authorized sets of values
_INFERENCE_LEVELS = (VALUE_DEFAULT, VALUE_LOW, VALUE_BASIC, VALUE_MEDIUM, VALUE_EXTENDED)
_ON_OFF_AUTO = (VALUE_ON, VALUE_OFF, VALUE_AUTO)
_ON_OFF = (VALUE_ON, VALUE_OFF)
_KPI_DISPLAY = (VALUE_SINGLE_LINE, VALUE_MULTIPLE_LINES)
_SEARCH_TYPES = (VALUE_DEPTH_FIRST, VALUE_RESTART, VALUE_MULTI_POINT, VALUE_ITERATIVE_DIVING, VALUE_AUTO)


###############################################################################
## Public classes
###############################################################################

class CpoParameters(Context):
    """ Class for handling solving parameters
    """
    def __init__(self, **kwargs):
        """ Creates a new set of solving parameters.

        This constructor takes a variable number of optional arguments that allow to set parameters directly.
        For example:
        ::
           myparams = CpoParameters(TimeLimit=20, LogPeriod=5000))

        Args:
            kwargs: (Optional) Any individual parameter as defined in this class.
        """
        super(CpoParameters, self).__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)


    def __setattr__(self, name, value):
        """ Set a parameter.
        This method calls appropriate setter if exists (property does not work in this context)
        to check the parameter value.
        Args:
            name:  Parameter name
            value: Parameter value
        """
        m = getattr(self, 'set_' + name, None)
        if m and callable(m):
            m(value)
        else:
            self.set_attribute(name, value)


    def reset_to_default(self):
        """ Reset all the parameters to their default value.

        Parameters are reset to their default value by being removed from this object.
        """
        self.clear()


    def add(self, ctx):
        """ Add another parameters to this one.

        All attributes of given parameters are set in this one, except if there value is None.
        If one value is another context, it is cloned before being set.

        Args:
            ctx:  Other context to add to this one.
        """
        for k, v in ctx.items():
            if v is not None:
                if isinstance(v, Context):
                    v = v.clone()
                self.set_attribute(k, v)


    def get_AllDiffInferenceLevel(self):
        """
        This parameter specifies the inference level for every constraint AllDiff extracted to the invoking
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all AllDiff constraints to be
        controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('AllDiffInferenceLevel')

    def set_AllDiffInferenceLevel(self, val):
        self._set_value_enum('AllDiffInferenceLevel', val, _INFERENCE_LEVELS)

    AllDiffInferenceLevel = property(get_AllDiffInferenceLevel, set_AllDiffInferenceLevel)


    def get_AllMinDistanceInferenceLevel(self):
        """
        This parameter specifies the inference level for every constraint AllMinDistance extracted to the
        invoked CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and
        Extended. The default value is Default, which allows the inference strength of all AllMinDistance
        constraints to be controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('AllMinDistanceInferenceLevel')

    def set_AllMinDistanceInferenceLevel(self, val):
        self._set_value_enum('AllMinDistanceInferenceLevel', val, _INFERENCE_LEVELS)

    AllMinDistanceInferenceLevel = property(get_AllMinDistanceInferenceLevel, set_AllMinDistanceInferenceLevel)


    def get_AutomaticReplay(self):
        """
        This parameter is an advanced, low-level one for controlling the behavior of solve() and next().
        When the model being solved has an objective and solve is used, or when startNewSearch and next are
        used to produce multiple solutions, the solver may have a need to replay the last (or best) solution
        found. This can, in some cases, involve re-invoking the strategy which produced the solution.
        Normally this is only necessary if you use low level "Ilc" interfaces to specify problem elements
        not in the model (instance of Model). This parameter can take the values On or Off. The default
        value is On. A typical reason for setting this parameter to Off is, for instance, if you use your
        own custom goal (instance of IlcGoal), and this goal is not deterministic (does not do the same
        thing when executed twice). In this instance, the replay will not work correctly, and you can use
        this parameter to disable replay.

        This parameter is deprecated since release 2.3.

        The value is a symbol in ['On', 'Off']. Default value is 'On'.
        """
        warnings.warn("Parameter 'AutomaticReplay' is deprecated since release 2.3.", DeprecationWarning)
        return self.get_attribute('AutomaticReplay')

    def set_AutomaticReplay(self, val):
        warnings.warn("Parameter 'AutomaticReplay' is deprecated since release 2.3.", DeprecationWarning)
        self._set_value_enum('AutomaticReplay', val, _ON_OFF)

    AutomaticReplay = property(get_AutomaticReplay, set_AutomaticReplay)


    def get_BranchLimit(self):
        """
        This parameter limits the number of branches that are made before terminating a search. A branch is
        a decision made at a choice point in the search, a typical node being made up of two branches, for
        example: x == value and x != value. A branch is only counted at the moment a decision is executed,
        not when the two branches of the choice point are decided. A branch is counted even if the decision
        leads to an inconsistency (failure).

        The value is a non-negative integer, or None (default) that does not set any limit.
        """
        return self.get_attribute('BranchLimit')

    def set_BranchLimit(self, val):
        self._set_value_integer('BranchLimit', val)

    BranchLimit = property(get_BranchLimit, set_BranchLimit)


    def get_ChoicePointLimit(self):
        """
        This parameter limits the number of choice points that are created before terminating a search.

        The value is a non-negative integer, or None (default) that does not set any limit.
        """
        return self.get_attribute('ChoicePointLimit')

    def set_ChoicePointLimit(self, val):
        self._set_value_integer('ChoicePointLimit', val)

    ChoicePointLimit = property(get_ChoicePointLimit, set_ChoicePointLimit)


    def get_ConflictRefinerBranchLimit(self):
        """
        This parameter limits the total number of branches that are made before terminating the conflict
        refiner.

        The value is a non-negative integer, or None (default) that does not set any limit.
        """
        return self.get_attribute('ConflictRefinerBranchLimit')

    def set_ConflictRefinerBranchLimit(self, val):
        self._set_value_integer('ConflictRefinerBranchLimit', val)

    ConflictRefinerBranchLimit = property(get_ConflictRefinerBranchLimit, set_ConflictRefinerBranchLimit)


    def get_ConflictRefinerFailLimit(self):
        """
        This parameter limits the total number of failures that can occur before terminating the conflict
        refiner.

        The value is a non-negative integer, or None (default) that does not set any limit.
        """
        return self.get_attribute('ConflictRefinerFailLimit')

    def set_ConflictRefinerFailLimit(self, val):
        self._set_value_integer('ConflictRefinerFailLimit', val)

    ConflictRefinerFailLimit = property(get_ConflictRefinerFailLimit, set_ConflictRefinerFailLimit)


    def get_ConflictRefinerIterationLimit(self):
        """
        This parameter limits the number of iterations that are made before terminating the conflict
        refiner.

        The value is a non-negative integer, or None (default) that does not set any limit.
        """
        return self.get_attribute('ConflictRefinerIterationLimit')

    def set_ConflictRefinerIterationLimit(self, val):
        self._set_value_integer('ConflictRefinerIterationLimit', val)

    ConflictRefinerIterationLimit = property(get_ConflictRefinerIterationLimit, set_ConflictRefinerIterationLimit)


    def get_ConflictRefinerOnVariables(self):
        """
        This parameter specifies whether the conflict refiner should refine variables domains. Possible
        values for this parameter are On (conflict refiner will refine both constraints and variables
        domains) and Off (conflict refiner will only refine constraints).

        The value is a symbol in ['On', 'Off']. Default value is 'Off'.
        """
        return self.get_attribute('ConflictRefinerOnVariables')

    def set_ConflictRefinerOnVariables(self, val):
        self._set_value_enum('ConflictRefinerOnVariables', val, _ON_OFF)

    ConflictRefinerOnVariables = property(get_ConflictRefinerOnVariables, set_ConflictRefinerOnVariables)


    def get_ConflictRefinerTimeLimit(self):
        """
        This parameter limits the CPU time spent before terminating the conflict refiner.

        The value is a non-negative float, or None (default) that does not set any limit.
        """
        return self.get_attribute('ConflictRefinerTimeLimit')

    def set_ConflictRefinerTimeLimit(self, val):
        self._set_value_float('ConflictRefinerTimeLimit', val)

    ConflictRefinerTimeLimit = property(get_ConflictRefinerTimeLimit, set_ConflictRefinerTimeLimit)


    def get_CountDifferentInferenceLevel(self):
        """
        This parameter specifies the inference level for every constraint CountDifferent extracted to the
        invoking CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and
        Extended. The default value is Default, which allows the inference strength of all CountDifferent
        constraints to be controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('CountDifferentInferenceLevel')

    def set_CountDifferentInferenceLevel(self, val):
        self._set_value_enum('CountDifferentInferenceLevel', val, _INFERENCE_LEVELS)

    CountDifferentInferenceLevel = property(get_CountDifferentInferenceLevel, set_CountDifferentInferenceLevel)


    def get_CountInferenceLevel(self):
        """
        This parameter specifies the inference level for every constraint Count extracted to the invoked CP
        instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all Count constraints to be
        controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('CountInferenceLevel')

    def set_CountInferenceLevel(self, val):
        self._set_value_enum('CountInferenceLevel', val, _INFERENCE_LEVELS)

    CountInferenceLevel = property(get_CountInferenceLevel, set_CountInferenceLevel)


    def get_CumulFunctionInferenceLevel(self):
        """
        This parameter specifies the inference level for constraints on expressions CumulFunctionExpr
        extracted to the invoked CP instance. Possible values for this parameter are Default, Low, Basic,
        Medium, and Extended. The default value is Default, which allows the inference strength of all
        constraints on CumulFunctionExpr to be controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('CumulFunctionInferenceLevel')

    def set_CumulFunctionInferenceLevel(self, val):
        self._set_value_enum('CumulFunctionInferenceLevel', val, _INFERENCE_LEVELS)

    CumulFunctionInferenceLevel = property(get_CumulFunctionInferenceLevel, set_CumulFunctionInferenceLevel)


    def get_DefaultInferenceLevel(self):
        """
        This parameter specifies the general inference level for constraints whose particular inference
        level is Default. Possible values for this parameter (in increasing order of inference strength) are
        Low, Basic, Medium, and Extended.

       The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
         """
        return self.get_attribute('DefaultInferenceLevel')

    def set_DefaultInferenceLevel(self, val):
        self._set_value_enum('DefaultInferenceLevel', val, _INFERENCE_LEVELS)

    DefaultInferenceLevel = property(get_DefaultInferenceLevel, set_DefaultInferenceLevel)


    def get_DistributeInferenceLevel(self):
        """
        This parameter specifies the inference level for every constraint Distribute extracted to the
        invoked CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and
        Extended. The default value is Default, which allows the inference strength of all Distribute
        constraints to be controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('DistributeInferenceLevel')

    def set_DistributeInferenceLevel(self, val):
        self._set_value_enum('DistributeInferenceLevel', val, _INFERENCE_LEVELS)

    DistributeInferenceLevel = property(get_DistributeInferenceLevel, set_DistributeInferenceLevel)


    def get_DynamicProbing(self):
        """
        This parameter controls probing carried out during search. Probing can be useful on some problems as
        it can make stronger inferences on combinations of constraints. Possible values for this parameter
        are On (dynamic probing is activated with a constant strength), Auto (dynamic probing is activated
        and its strength is adjusted adaptively) and Off (dynamic probing is deactivated). The strength of
        probing can be defined by parameter DynamicProbingStrength. Dynamic probing only has an effect when
        using the "Restart" (Restart) search type, on problems without interval variables.

        The value is a symbol in ['On', 'Off', 'Auto']. Default value is 'Auto'.
        """
        return self.get_attribute('DynamicProbing')

    def set_DynamicProbing(self, val):
        self._set_value_enum('DynamicProbing', val, _ON_OFF_AUTO)

    DynamicProbing = property(get_DynamicProbing, set_DynamicProbing)


    def get_DynamicProbingStrength(self):
        """
        This parameter controls the effort which is dedicated to dynamic probing. It is expressed as a
        factor of the total search effort: changing this parameter has no effect unless the DynamicProbing
        parameter is set to Auto or On. When DynamicProbing has value On, the probing strength is held
        constant throughout the search process. When DynamicProbing has value Auto, the probing strength
        starts off at the specified value and is thereafter adjusted automatically. Possible values for this
        parameter range from 0.001 to 1000. A value of 1.0 indicates that dynamic probing will consume a
        roughly equal amount of effort as the rest of the search. The default value of this parameter is
        0.03, meaning that around 3% of total search time is dedicated to dynamic probing.

        The value is a float in [0.001..1000]. Default value is 0.03.
        """
        return self.get_attribute('DynamicProbingStrength')

    def set_DynamicProbingStrength(self, val):
        self._set_value_float('DynamicProbingStrength', val, 0.001, 1000)

    DynamicProbingStrength = property(get_DynamicProbingStrength, set_DynamicProbingStrength)


    def get_ElementInferenceLevel(self):
        """
        This parameter specifies the inference level for every element constraint extracted to the invoked
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all element constraints to be
        controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('ElementInferenceLevel')

    def set_ElementInferenceLevel(self, val):
        self._set_value_enum('ElementInferenceLevel', val, _INFERENCE_LEVELS)

    ElementInferenceLevel = property(get_ElementInferenceLevel, set_ElementInferenceLevel)


    def get_FailLimit(self):
        """
        This parameter limits the number of failures that can occur before terminating the search.

        The value is a non-negative integer, or None (default) that does not set any limit.
        """
        return self.get_attribute('FailLimit')

    def set_FailLimit(self, val):
        self._set_value_integer('FailLimit', val)

    FailLimit = property(get_FailLimit, set_FailLimit)


    def get_FailureDirectedSearch(self):
        """
        This parameter controls usage of failure-directed search. Failure-directed search assumes that there
        is no (better) solution or that such a solution is very hard to find. Therefore it focuses on a
        systematic exploration of search space, first eliminating assignments that are most likely to fail.
        Failure-directed search is used only for scheduling problems (i.e. models containing interval
        variables) and only when the parameter SearchType is set to Restart or Auto. Legal values for the
        FailureDirectedSearch parameter are On (the default) and Off. When the value is On then CP Optimizer
        starts failure-directed search when other search strategies are (no longer) successful and when the
        memory necessary for the search does not exceed the value set by the FailureDirectedSearchMaxMemory
        parameter.

        The value is a symbol in ['On', 'Off']. Default value is 'On'.
        """
        return self.get_attribute('FailureDirectedSearch')

    def set_FailureDirectedSearch(self, val):
        self._set_value_enum('FailureDirectedSearch', val, _ON_OFF)

    FailureDirectedSearch = property(get_FailureDirectedSearch, set_FailureDirectedSearch)


    def get_FailureDirectedSearchEmphasis(self):
        """
        This parameter controls how much time CP Optimizer invests into failure-directed search once it is
        started. The default value Auto means that CP Optimizer observes the actual performance of
        failure-directed search and decides automatically how much time is invested. Any other value means
        that once failure-directed search has started, it is used by given number of workers. The value does
        not have to be integer. For example, value 1.5 means that first worker spends 100% of the time by
        failure-directed search, second worker 50% and remaining workers 0%. See also Workers For more
        information about failure-directed search see parameter FailureDirectedSearch.
        
        The value is a non-negative integer, or None (default) that does not set any limit.
        """
        return self.get_attribute('FailureDirectedSearchEmphasis')

    def set_FailureDirectedSearchEmphasis(self, val):
        self._set_value_float('FailureDirectedSearchEmphasis', val)

    FailureDirectedSearchEmphasis = property(get_FailureDirectedSearchEmphasis, set_FailureDirectedSearchEmphasis)


    def get_FailureDirectedSearchMaxMemory(self):
        """
        This parameter controls the maximum amount of memory (in bytes) available to failure-directed search
        (see FailureDirectedSearchMaxMemory). The default value is 104,857,600 (100MB). Failure-directed
        search can sometimes consume a lot of memory, especially when end times of interval variables are
        not bounded. Therefore it is usually not started immediately, but only when the effective horizon
        (time period over which CP Optimizer must reason) becomes small enough for failure-directed search
        to operate inside the memory limit specified by this parameter. For many types of scheduling
        problems, the effective horizon tends to reduce when CP Optimizer finds a better solution (often
        most significantly when the initial solution is found). Therefore, when each new solution is found,
        CP Optimizer decides whether or not to turn on failure-directed search. Note that this parameter
        does not influence the effectiveness of failure-directed search, once started. Its purpose is only
        to control the point at which failure-directed search will begin to function.

        The value is a non-negative integer, or None that does not set any limit. Default value is 104857600.
        """
        return self.get_attribute('FailureDirectedSearchMaxMemory')

    def set_FailureDirectedSearchMaxMemory(self, val):
        self._set_value_integer('FailureDirectedSearchMaxMemory', val)

    FailureDirectedSearchMaxMemory = property(get_FailureDirectedSearchMaxMemory, set_FailureDirectedSearchMaxMemory)


    def get_IntervalSequenceInferenceLevel(self):
        """
        This parameter specifies the inference level for the maintenance of the domain of every interval
        sequence variable IntervalSequenceVar extracted to the invoking CP instance. Possible values for
        this parameter are Default, Low, Basic, Medium, and Extended. The default value is Default, which
        allows the inference strength of all IntervalSequenceVar to be controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. default value is 'Default'.
        """
        return self.get_attribute('IntervalSequenceInferenceLevel')

    def set_IntervalSequenceInferenceLevel(self, val):
        self._set_value_enum('IntervalSequenceInferenceLevel', val, _INFERENCE_LEVELS)

    IntervalSequenceInferenceLevel = property(get_IntervalSequenceInferenceLevel, set_IntervalSequenceInferenceLevel)


    def get_KPIDisplay(self):
        """
        This parameter determines how KPIs are displayed in the log during the search.

        The value is a symbol in ['SingleLine', 'MultipleLines']. Default value is 'SingleLine'.

        *New in version 2.8, for CPO solver version 12.9.0 and higher.*
        """
        return self.get_attribute('KPIDisplay')

    def set_KPIDisplay(self, val):
        self._set_value_enum('KPIDisplay', val, _KPI_DISPLAY)

    KPIDisplay = property(get_KPIDisplay, set_KPIDisplay)


    def get_LogPeriod(self):
        """
        The CP Optimizer search log includes information that is displayed periodically. This parameter
        controls how often that information is displayed. By setting this parameter to a value of k, the log
        is displayed every k branches (search decisions).

        The value is an integer strictly greater than 0. Default value is 1000.
        """
        return self.get_attribute('LogPeriod')

    def set_LogPeriod(self, val):
        self._set_value_integer('LogPeriod', val, min=1)

    LogPeriod = property(get_LogPeriod, set_LogPeriod)


    # def get_LogSearchTags(self):
    #     """
    #     This parameter controls the log activation. When set to On, the engine will display failure tags
    #     (indices) in the engine log when solving the model. To specify the failures to explain, the member
    #     functions explainFailure(Int failureTag) or explainFailure(IntArray tagArray) needs to be called
    #     with the failure tags as the parameter. Several failures tags can be added. The member function
    #     clearExplanations() is used to clear the set of failure tags to be explained. To be able to see
    #     failure tags and explanations, the parameter SearchType must be set to DepthFirst and the parameter
    #     Workers to 1.
    #
    #     The value is a symbol in ['On', 'Off']. Default value is 'Off'.
    #     """
    #     return self.get_attribute('LogSearchTags')
    #
    # def set_LogSearchTags(self, val):
    #     self._set_value_enum('LogSearchTags', val, _ON_OFF)
    #
    # LogSearchTags = property(get_LogSearchTags, set_LogSearchTags)


    def get_LogVerbosity(self):
        """
        This parameter determines the verbosity of the search log. The possible values are Quiet, Terse,
        Normal, and Verbose. Mode Quiet does not display any information, the other modes display
        progressively more information. The default value is Normal. The CP Optimizer search log is meant
        for visual inspection only, not for mechanized parsing. In particular, the log may change from
        version to version of CP Optimizer in order to improve the quality of information displayed in the
        log. Any code based on the log output for correct functioning may have to be updated when a new
        version of CP Optimizer is released.

        The value is a symbol in ['Quiet', 'Terse', 'Normal', 'Verbose']. Default value is 'Normal'.
        """
        return self.get_attribute('LogVerbosity')

    def set_LogVerbosity(self, val):
        self._set_value_enum('LogVerbosity', val, (VALUE_QUIET, VALUE_TERSE, VALUE_NORMAL, VALUE_VERBOSE))

    LogVerbosity = property(get_LogVerbosity, set_LogVerbosity)


    def get_ModelAnonymizer(self):
        """
        This parameter controls anonymization of a model dumped in CPO file format. The legal values of this
        parameter are Off and On. The default is Off. When the anonymizer is off, then names of variables
        and constraints in the model may be found in the output file. When the anonymizer is on, names given
        to variables or constraints in the model will not be reflected in the output file and standard
        anonymized names will be used.

        The value is a symbol in ['On', 'Off']. Default value is 'Off'.
        """
        return self.get_attribute('ModelAnonymizer')

    def set_ModelAnonymizer(self, val):
        self._set_value_enum('ModelAnonymizer', val, _ON_OFF)

    ModelAnonymizer = property(get_ModelAnonymizer, set_ModelAnonymizer)


    def get_MultiPointNumberOfSearchPoints(self):
        """
        This parameter controls the number of (possibly partial) solutions manipulated by the multi-point
        search algorithm. The default value is 30. A larger value will diversify the search, with possible
        improvement in solution quality at the expense of a longer run time. A smaller value will intensify
        the search, resulting in faster convergence at the expense of solution quality. Note that memory
        consumption increases proportionally to this parameter, for each search point must store each
        decision variable domain.

        The value is an integer strictly greater than 1. Default value is 30.
        """
        return self.get_attribute('MultiPointNumberOfSearchPoints')

    def set_MultiPointNumberOfSearchPoints(self, val):
        self._set_value_integer('MultiPointNumberOfSearchPoints', val, min=2)

    MultiPointNumberOfSearchPoints = property(get_MultiPointNumberOfSearchPoints, set_MultiPointNumberOfSearchPoints)


    def get_NoOverlapInferenceLevel(self):
        """
        This parameter specifies the inference level for every constraint NoOverlap extracted to the invoked
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all NoOverlap constraints to be
        controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('NoOverlapInferenceLevel')

    def set_NoOverlapInferenceLevel(self, val):
        self._set_value_enum('NoOverlapInferenceLevel', val, _INFERENCE_LEVELS)

    NoOverlapInferenceLevel = property(get_NoOverlapInferenceLevel, set_NoOverlapInferenceLevel)


    def get_OptimalityTolerance(self):
        """
        This parameter sets an absolute tolerance on the objective value for optimization models. This means
        that when CP Optimizer reports an optimal solution found, then there is no solution which improves
        the objective by more than the value of this parameter.
        This parameter is used in conjunction with RelativeOptimalityTolerance. The optimality of a solution
        is proven if either of the two parameters' criteria is fulfilled.

        The value is a positive float. Default value is 1e-09.
        """
        return self.get_attribute('OptimalityTolerance', 1e-09)

    def set_OptimalityTolerance(self, val):
        self._set_value_float('OptimalityTolerance', val)

    OptimalityTolerance = property(get_OptimalityTolerance, set_OptimalityTolerance)


    def get_PrecedenceInferenceLevel(self):
        """
        This parameter specifies the inference level for precedence constraints between interval variables
        extracted to the invoking CP instance. Possible values for this parameter are Default, Low, Basic,
        Medium, and Extended. The default value is Default, which allows the inference strength for
        precedence constraints between interval variables to be controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('PrecedenceInferenceLevel')

    def set_PrecedenceInferenceLevel(self, val):
        self._set_value_enum('PrecedenceInferenceLevel', val, _INFERENCE_LEVELS)

    PrecedenceInferenceLevel = property(get_PrecedenceInferenceLevel, set_PrecedenceInferenceLevel)


    def get_Presolve(self):
        """
        This parameter controls the presolve of the model to produce more compact formulations and to
        achieve more domain reduction. Possible values for this parameter are On (presolve is activated) and
        Off (presolve is deactivated).

        The value is a symbol in ['On', 'Off']. Default value is 'On'.
        """
        return self.get_attribute('Presolve')

    def set_Presolve(self, val):
        self._set_value_enum('Presolve', val, _ON_OFF)

    Presolve = property(get_Presolve, set_Presolve)


    def get_PrintModelDetailsInMessages(self):
        """
        Whenever CP Optimizer prints an error or warning message, it can also print concerning part of the
        input model (in cpo file format). This parameter controls printing of this additional information.
        Possible values are On and Off, the Default value is 'On'. See also WarningLevel.

        The value is a symbol in ['On', 'Off']. Default value is 'On'.
        """
        return self.get_attribute('PrintModelDetailsInMessages')

    def set_PrintModelDetailsInMessages(self, val):
        self._set_value_enum('PrintModelDetailsInMessages', val, _ON_OFF)

    PrintModelDetailsInMessages = property(get_PrintModelDetailsInMessages, set_PrintModelDetailsInMessages)


    def get_RandomSeed(self):
        """
        The search uses some randomization in some strategies. This parameter sets the seed of the random
        generator used by these strategies.

        The value is a non-negative integer. Default value is 0.
        """
        return self.get_attribute('RandomSeed')

    def set_RandomSeed(self, val):
        self._set_value_integer('RandomSeed', val)

    RandomSeed = property(get_RandomSeed, set_RandomSeed)


    def get_RelativeOptimalityTolerance(self):
        """
        This parameter sets a relative tolerance on the objective value for optimization models. This means
        that when CP Optimizer reports an optimal solution found, then there is no solution which improves
        the objective by more than the absolute value of the objective times the value of this parameter.
        The default value of this parameter is 1e-4. This parameter is used in conjunction with
        OptimalityTolerance. The optimality of a solution is proven if either of the two parameters'
        criteria are fulfilled.

        The value is a non-negative float. Default value is 0.0001.
        """
        return self.get_attribute('RelativeOptimalityTolerance', 0.0001)

    def set_RelativeOptimalityTolerance(self, val):
        self._set_value_float('RelativeOptimalityTolerance', val)

    RelativeOptimalityTolerance = property(get_RelativeOptimalityTolerance, set_RelativeOptimalityTolerance)


    def get_RestartFailLimit(self):
        """
        When SearchType is set to Restart, a depth-first search is restarted after a certain number of
        failures. This parameter controls the number of failures that must occur before restarting search.
        Possible values range from 0 to Infinity. The default value is 100. This value can increase after
        each restart: see the parameter RestartGrowthFactor.

        The value is an integer greater than 0. Default value is 100.
        """
        return self.get_attribute('RestartFailLimit', 100)

    def set_RestartFailLimit(self, val):
        self._set_value_integer('RestartFailLimit', val, 1)

    RestartFailLimit = property(get_RestartFailLimit, set_RestartFailLimit)


    def get_RestartGrowthFactor(self):
        """
        When SearchType is set to Restart, a depth-first search is restarted after a certain number of
        failures. This parameter controls the increase of this number between restarts. If the last fail
        limit was f after a restart, for next run, the new fail limit will be f times the value of this
        parameter. Possible values of this parameter range from 1.0 to Infinity. The default value is 1.05.
        The initial fail limit can be controlled with the parameter RestartFailLimit.

        The value is a float greater or equal to 1. Default value is 1.15.
        """
        return self.get_attribute('RestartGrowthFactor')

    def set_RestartGrowthFactor(self, val):
        self._set_value_float('RestartGrowthFactor', val, min=1)

    RestartGrowthFactor = property(get_RestartGrowthFactor, set_RestartGrowthFactor)


    def get_SearchType(self):
        """
        This parameter determines the type of search that is applied when solving a problem.

         * When set to *DepthFirst*, a regular depth-first search is applied.
         * When set to *Restart*, a depth-first search that restarts from time to time is applied.
         * When set to *IterativeDiving* on scheduling problems (ones with at least one interval variable),
           a more aggressive diving technique is applied in order to find solutions to large problems more quickly.
         * When set to *MultiPoint*, a method that combines a set of - possibly partial - solutions is applied.
         * When set to *Auto* in sequential mode, this value chooses the appropriate search method to be used.
           In general Auto will be the Restart search. The default value is Auto.

        In parallel mode (i.e, when the number of workers is greater than one - see the Workers
        parameter), the different searches described above are spread over the workers. When the value of
        SearchType is Auto, then the decision of choosing the search type for a worker is automatically
        made; otherwise, all workers execute the same type of search. Note that in the latter case, the
        workers will not do the same exploration due to some randomness introduced to break ties in decision
        making.

        The value is a symbol in ['DepthFirst', 'Restart', 'MultiPoint', 'IterativeDiving', 'Auto']. Default value is 'Auto'.
        """
        return self.get_attribute('SearchType')

    def set_SearchType(self, val):
        self._set_value_enum('SearchType', val, _SEARCH_TYPES)

    SearchType = property(get_SearchType, set_SearchType)


    def get_SequenceInferenceLevel(self):
        """
        This parameter specifies the inference level for every constraint Sequence extracted to the invoked
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all Sequence constraints to be
        controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('SequenceInferenceLevel')

    def set_SequenceInferenceLevel(self, val):
        self._set_value_enum('SequenceInferenceLevel', val, _INFERENCE_LEVELS)

    SequenceInferenceLevel = property(get_SequenceInferenceLevel, set_SequenceInferenceLevel)


    def get_SolutionLimit(self):
        """
        This parameter limits the number of feasible solutions that are found before terminating a search.

        The value is a non-negative integer, or None (default) that does not set any limit.
        """
        return self.get_attribute('SolutionLimit')

    def set_SolutionLimit(self, val):
        self._set_value_integer('SolutionLimit', val)

    SolutionLimit = property(get_SolutionLimit, set_SolutionLimit)


    def get_StateFunctionInferenceLevel(self):
        """
        This parameter specifies the inference level for constraints on state functions StateFunction
        extracted to the invoked CP instance. Possible values for this parameter are Default, Low, Basic,
        Medium, and Extended. The default value is Default, which allows the inference strength of all
        constraints on state functions StateFunction to be controlled via DefaultInferenceLevel.

        The value is a symbol in ['Default', 'Low', 'Basic', 'Medium', 'Extended']. Default value is 'Default'.
        """
        return self.get_attribute('StateFunctionInferenceLevel')

    def set_StateFunctionInferenceLevel(self, val):
        self._set_value_enum('StateFunctionInferenceLevel', val, _INFERENCE_LEVELS)

    StateFunctionInferenceLevel = property(get_StateFunctionInferenceLevel, set_StateFunctionInferenceLevel)


    def get_TemporalRelaxation(self):
        """
        This advanced parameter can be used to control the usage of a temporal relaxation internal to the
        invoking CP engine. This parameter can take values On or Off, with On being the default, meaning the
        relaxation is used in the engine when needed. For some models, using the relaxation becomes
        inefficient, and you may deactivate the use of the temporal relaxation using value Off.

        The value is a symbol in ['On', 'Off']. Default value is 'On'.
        """
        return self.get_attribute('TemporalRelaxation')

    def set_TemporalRelaxation(self, val):
        self._set_value_enum('TemporalRelaxation', val, _ON_OFF)

    TemporalRelaxation = property(get_TemporalRelaxation, set_TemporalRelaxation)


    def get_TimeLimit(self):
        """
        This parameter limits the CPU time spent solving before terminating a search. The time is given in
        seconds.

        The value is a non-negative float, or None (default) that does not set any limit.
        """
        return self.get_attribute('TimeLimit')

    def set_TimeLimit(self, val):
        self._set_value_float('TimeLimit', val)

    TimeLimit = property(get_TimeLimit, set_TimeLimit)


    def get_TimeMode(self):
        """
        This parameter defines how time is measured in CP Optimizer, the two legal values being ElapsedTime
        and CPUTime. CP Optimizer uses time for both display purposes and for limiting the search via
        TimeLimit. Note that when multiple processors are available and the number of workers (Workers) is
        greater than one, then the CPU time can be greater than the elapsed time by a factor up to the
        number of workers.

        The value is a symbol in ['CPUTime', 'ElapsedTime']. Default value is 'ElapsedTime'.
        """
        return self.get_attribute('TimeMode')

    def set_TimeMode(self, val):
        self._set_value_enum('TimeMode', val, (VALUE_CPU_TIME, VALUE_ELAPSED_TIME))

    TimeMode = property(get_TimeMode, set_TimeMode)


    def get_UseFileLocations(self):
        """
        This parameter controls whether CP Optimizer processes file locations. With each constraint,
        variable or expression it is possible to associate a source file location (file name and line
        number). CP Optimizer can use locations later for reporting errors and conflicts. Locations are also
        included in exported/dumped models (#line directives). Legal values for this parameter are On (the
        default) and Off. When the value is Off then CP Optimizer ignores locations in the input model and
        also does not export them in CPO file format (functions dumpModel and exportModel).

        The value is a symbol in ['On', 'Off']. Default value is 'On'.
        """
        return self.get_attribute('UseFileLocations')

    def set_UseFileLocations(self, val):
        self._set_value_enum('UseFileLocations', val, _ON_OFF)

    UseFileLocations = property(get_UseFileLocations, set_UseFileLocations)


    def get_WarningLevel(self):
        """
        This parameter controls the level of warnings issued by CP Optimizer when a solve is launched.
        Specifically, all warnings of level higher than this parameter are masked. Since CP Optimizer
        warning levels run from 1 to 3, setting this parameter to 0 turns off all warnings. Warnings issued
        may indicate potential errors or inefficiencies in your model. The default value of this parameter
        is 2. See also PrintModelDetailsInMessages.

        The value is an integer in [0..3]. Default value is 2.
        """
        return self.get_attribute('WarningLevel')

    def set_WarningLevel(self, val):
        self._set_value_integer('WarningLevel', val, min=0, max=3)

    WarningLevel = property(get_WarningLevel, set_WarningLevel)


    def get_Workers(self):
        """
        This parameter sets the number of workers to run in parallel to solve your model. If the number of
        workers is set to n (with n greater than one), the CP optimizer will create n workers, each in their
        own thread, that will work together to solve the problem. The emphasis of these workers is more to
        find better feasible solutions and then to speed up the proof of optimality. The default value is
        Auto. This amounts to using as many workers as there are CPU cores available on the machine. Note
        that the memory required by CP Optimizer grows roughly linearly as the number of workers is
        increased. If you are solving a very large model on a multi-core processor and memory usage is an
        issue, it is advisable to specify a reduced number of workers, or even one worker, rather than use
        the default value.

        The value is a positive integer. Default value is Auto.
        """
        return self.get_attribute('Workers')

    def set_Workers(self, val):
        assert (val is None) or (is_int(val) and val > 0) or val == "Auto", \
            "Value of parameter 'Workers' should be a positive integer or 'Auto'"
        self.set_attribute('Workers', val)

    Workers = property(get_Workers, set_Workers)


    def _set_value_enum(self, name, val, accepted):
        """ Check if a value is in a given set and set it if ok.
        Args:
            name:     Parameter name
            val:      Parameter value to set
            accepted: Accepted values
        """
        assert (val is None) or (val in accepted), "Value of parameter '{}' should be in {}".format(name, accepted)
        self.set_attribute(name, val)

    def _set_value_integer(self, name, val, min=0, max=None):
        """ Check if a value is a non-negative integer and set it if ok.
        Args:
            name:     Parameter name
            val:      Parameter value to set
            min:      (Optional) Minimun value
            max:      (Optional) Maximum value
        """
        assert (val is None) or \
               (is_int(val) and ((min is None or val >= min) and (max is None or val <= max))), \
                     "Value of parameter '{}' should be an integer in [{}..{}]"\
                         .format(name, "-Infinity" if min is None else min, "Infinity" if max is None else max)
        self.set_attribute(name, val)


    def _set_value_float(self, name, val, min=0, max=None):
        """ Check if a value is a non-negative float and set it if ok.
        Args:
            name:     Parameter name
            val:      Parameter value to set
            min:      (Optional) Minimun value
            max:      (Optional) Maximum value
        """
        assert (val is None) or \
               (is_number(val) and ((min is None or val >= min) and (max is None or val <= max))), \
                      "Value of parameter '{}' should be a float in [{}..{}]"\
                          .format(name, "-Infinity" if min is None else min, "Infinity" if max is None else max)
        self.set_attribute(name, val)

