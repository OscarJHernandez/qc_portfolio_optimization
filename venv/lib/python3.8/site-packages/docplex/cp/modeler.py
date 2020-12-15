# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module contains the functions that allows to construct all operations
and constraints that can be used in a *CP Optimizer* model.

There is one Python function per *CP Optimizer* operation or constraint.
As many operations have multiple combinations of parameters,
the functions of this module are generally declared with a variable number
of arguments.
The valid combinations are detailed in the description of the function.

Following is the list of textual expressions used to identify the different data types that
are used by the modeling methods:

 * *integer expression* represents any integer expression including integer constants, integer variables, and
   modeling operations that take only integer expressions as arguments.
 * *boolean expression* represents a logical value true or false.
   It can be used as integer expressions, with value 1 for true, and 0 for false.
 * *float expression* represents any floating point expression, including float constants, integer expressions, and
   modeling operations that take at least one float expression as argument.
 * *array of xxxx expression* represents a Python list or tuple of *xxxx expression*.
 * *constraint expression* is an expression that can not be used as an argument of another expression.
 * *cumul expression* is an expression that represents the sum of individual contributions of intervals.

In the following, the different modeling functions are briefly presented.

Core CP modeling functions
--------------------------

**Arithmetic expressions**

The following functions are used to construct arithmetic expressions.
When exists, the corresponding operator are also overloaded to make easier the writing of expressions.

 * :meth:`plus`: Addition of two expressions.
 * :meth:`minus`: Difference between two expressions, or unary minus of a single one.
 * :meth:`times`: Multiplication of two expressions.
 * :meth:`int_div`: Integer division of two expressions.
 * :meth:`float_div`: Floating point division of two expressions.
 * :meth:`mod`: Modulo of two expressions.
 * :meth:`abs`:  Absolute value of an expression.
 * :meth:`square`: Square of an expression.
 * :meth:`power`: Power of an expression by another.
 * :meth:`log`: Logarithm of an expression.
 * :meth:`exponent`: Exponentiation of an expression.
 * :meth:`sum`: Sum of multiple expressions.
 * :meth:`min`: Minimum of multiple expressions.
 * :meth:`max`: Maximum of multiple expressions.
 * :meth:`ceil`: Rounds a float expression upward to the nearest integer.
 * :meth:`floor`: Rounds a float expression down to the nearest integer.
 * :meth:`round`: Rounds a float expression to the nearest integer.
 * :meth:`trunc`: Truncated integer parts of a float expression.
 * :meth:`sgn`: Sign of a float expresion.

**Logical expressions**

The following functions are used to construct logical expressions.
As for arithmetic, the corresponding operators are also overloaded.

 * :meth:`logical_and`: Logical AND of two boolean expressions or an array of expressions.
 * :meth:`logical_or`: Logical OR of two boolean expressions or an array of expressions.
 * :meth:`logical_not`: Logical NOT of a boolean expression.
 * :meth:`equal`: Equality between two expressions.
 * :meth:`diff`: Inequality between two expressions.
 * :meth:`greater`: An expression is greater than another.
 * :meth:`greater_or_equal`: An expression is greater or equal to another.
 * :meth:`less`: An expression is less than another.
 * :meth:`less_or_equal`: An expression is less or equal to another.
 * :meth:`true`: Always true boolean expression.
 * :meth:`false`: Always false boolean expression.

**General purpose**

Following functions allow to construct general purpose expressions:

 * :meth:`count`: Counts the occurrences of an expression in an array of integer expressions.
 * :meth:`count_different`: Counts the number of different values in an array of integer expressions.
 * :meth:`scal_prod`: Scalar product of two vectors.
 * :meth:`constant`: Creates an expression from a numeric constant.
 * :meth:`element`: Access to an element of an array using an integer expression.
 * :meth:`range`: Restricts the bounds of an integer or floating-point expression.
 * :meth:`all_min_distance`: Constraint on the minimum absolute distance between a pair of integer expressions in an array.
 * :meth:`if_then`: Creates and returns the new constraint e1 => e2.
 * :meth:`allowed_assignments`: Explicitly defines allowed assignments for one or more integer expressions.
 * :meth:`forbidden_assignments`: Explicitly defines forbidden assignments for one or more integer expressions.
 * :meth:`standard_deviation`: Standard deviation of the values of the variables in an array.
 * :meth:`slope_piecewise_linear`: Evaluates piecewise-linear function given by set of breaking points and slopes.
 * :meth:`coordinate_piecewise_linear`: Evaluates piecewise-linear function given by set of breaking points and values.

and constraints:

 * :meth:`all_diff`: Constrains multiple expressions to be all different.
 * :meth:`bool_abstraction`: Abstracts the values of one array as boolean values in another array.
 * :meth:`pack`: Maintains the load on a set of containers given objects sizes and assignments.
 * :meth:`abstraction`: Abstracts the values of one array as values in another array.
 * :meth:`inverse`: Constrains elements of one array to be inverses of another.
 * :meth:`distribute`: Calculates and/or constrains the distribution of values taken by an array of integer expressions.
 * :meth:`lexicographic`: Constraint which maintains two arrays to be lexicographically ordered.
 * :meth:`strict_lexicographic`: Constraint which maintains two arrays to be strictly lexicographically ordered.
 * :meth:`sequence`:  Constrains the number of occurrences of the values taken by the different subsets of consecutive *k* variables.
 * :meth:`strong`: Encourage CP Optimizer to produce stronger (higher inference) constraints.


**Objective**

Following functions are used to express what expression(s) is to be minimized or maximized.

 * :meth:`minimize`: Specify one expression to minimize.
 * :meth:`minimize_static_lex`: Specify several expressions to minimize.
 * :meth:`maximize`: Specify one expression to maximize.
 * :meth:`maximize_static_lex`: Specify several expressions to maximize.


Scheduling functions
--------------------

**Interval variables**

Following functions allow to construct expressions on interval variables:

 * :meth:`start_of`: Start of an interval variable.
 * :meth:`end_of`: End of an interval variable.
 * :meth:`length_of`: Length of an interval variable.
 * :meth:`size_of`: Size of an interval variable.
 * :meth:`presence_of`: Presence status of an interval variable.

and constraints:

 * :meth:`start_at_start`: Constrains the delay between the starts of two interval variables.
 * :meth:`start_at_end`: Constrains the delay between the start of one interval variable and end of another one.
 * :meth:`start_before_start`: Constrains the minimum delay between starts of two interval variables.
 * :meth:`start_before_end`: Constrains minimum delay between the start of one interval variable and end of another one.
 * :meth:`end_at_start`: Constrains the delay between the end of one interval variable and start of another one.
 * :meth:`end_at_end`:  Constrains the delay between the ends of two interval variables.
 * :meth:`end_before_start`: Constrains minimum delay between the end of one interval variable and start of another one.
 * :meth:`end_before_end`: Constrains the minimum delay between the ends of two interval variables.
 * :meth:`forbid_start`: Forbids an interval variable to start during specified regions.
 * :meth:`forbid_end`: Forbids an interval variable to end during specified regions.
 * :meth:`forbid_extent`: Forbids an interval variable to overlap with specified regions.
 * :meth:`overlap_length`: Length of the overlap of two interval variables.
 * :meth:`start_eval`: Evaluates a segmented function at the start of an interval variable.
 * :meth:`end_eval`: Evaluates a segmented function at the end of an interval variable.
 * :meth:`size_eval`: Evaluates a segmented function on the size of an interval variable.
 * :meth:`length_eval`: Evaluates segmented function on the length of an interval variable.
 * :meth:`span`: Creates a span constraint between interval variables.
 * :meth:`alternative`: Creates an alternative constraint between interval variables.
 * :meth:`synchronize`: Creates a synchronization constraint between interval variables.
 * :meth:`isomorphism`: Creates a isomorphism constraint between two sets of interval variables.

**Sequence variables**

Following functions allow to construct expressions on sequence variables:

 * :meth:`start_of_next`: Start of the interval variable that is next in a sequence.
 * :meth:`start_of_prev`: Start of the interval variable that is previous in a sequence.
 * :meth:`end_of_next`: End of the interval variable that is next in a sequence.
 * :meth:`end_of_prev`: End of the interval variable that is previous in a sequence.
 * :meth:`length_of_next`: Length of the interval variable that is next in a sequence.
 * :meth:`length_of_prev`: Length of the interval variable that is previous in a sequence.
 * :meth:`size_of_next`: Size of the interval variable that is next in a sequence.
 * :meth:`size_of_prev`: Size of the interval variable that is previous in a sequence.
 * :meth:`type_of_next`: Type of the interval variable that is next in a sequence.
 * :meth:`type_of_prev`: Type of the interval variable that is previous in a sequence.

and constraints:

 * :meth:`first`: Constrains an interval variable to be the first in a sequence.
 * :meth:`last`: Constrains an interval variable to be the last in a sequence.
 * :meth:`before`: Constrains an interval variable to be before another interval variable in a sequence.
 * :meth:`previous`: Constrains an interval variable to be previous to another interval variable in a sequence.
 * :meth:`no_overlap`: Constrains a set of interval variables not to overlap each others
 * :meth:`same_sequence`: creates a same-sequence constraint between two sequence variables.
 * :meth:`same_common_subsequence`: Creates a same-common-subsequence constraint between two sequence variables.

**Cumulative expressions**

A cumul function expression is an expression whose value in a solution is a function from the set of integers
to the set of non-negative integers.
A cumul function expression represents the sum of individual contributions of intervals.
A panel of elementary cumul function expressions is available to describe the individual contribution of
an interval variable (or a fixed interval) to a cumul function expression:

 * A pulse function is an elementary function defined by an interval variable (or a fixed interval) whose value is
   equal to 0 outside the interval and equal to a non-negative constant on the interval.
   This value is called the height of the pulse function.
 * A step function is an elementary function defined by one of the end-points of an interval variable
   (its start or end) whose value is equal to 0 before this end-point and equal to a non-negative constant
   after the end-point.
   This value is called the height of the step function.
 * A cumul function expression is defined as the sum of the above elementary functions or their opposite.
   Several constraints over cumul function expressions are provided.
   These constraints allow restricting the possible values of the function over the complete horizon or over
   some fixed or variable interval:

Available methods to build expressions are:

 * :meth:`pulse`: Elementary cumul function of constant value between the start and the end of an interval.
 * :meth:`step_at`: Elementary cumul function of constant value after a given point.
 * :meth:`step_at_start`: Elementary cumul function of constant value after the start of an interval.
 * :meth:`step_at_end`: Elementary cumul function of constant value after the end of an interval.
 * :meth:`height_at_start`: Contribution of an interval variable to a cumul function at its start point.
 * :meth:`height_at_end`: Contribution of an interval variable to a cumul function at its end point.

and constraints:

 * :meth:`always_in`: Restrict the possible values of a cumul expression (or a state function) to a particular range
 * :meth:`cumul_range`: Limits the range of a cumul function expression.

**State functions**

A state function is a decision variable whose value is a set of non-overlapping intervals over which the
function maintains a particular non-negative integer state.
In between those intervals, the state of the function is not defined, typically because of an ongoing
transition between two states.

A set of constraints are available to restrict the evolution of a state function:

 * :meth:`always_in`: Restrict the possible values of a state function (or a cumul expression) to a particular range.
 * :meth:`always_no_state`: Ensures that a state function is undefined on an interval.
 * :meth:`always_constant`: Ensures a constant state for a state function on an interval.
 * :meth:`always_equal`: Fixes a given state for a state function during a variable or fixed interval.

Search phases
-------------

**Variable evaluators**

An evaluator of integer variables is an object that is used by selectors of variables to define instantiation strategies.

 * :meth:`domain_size`: Number of elements in the current domain of the variable chosen by the search.
 * :meth:`domain_max`: Maximum value in the current domain of the variable chosen by the search.
 * :meth:`domain_min`: Minimum value in the current domain of the variable chosen by the search.
 * :meth:`var_impact`: Average reduction of the search space of the variable chosen by the search.
 * :meth:`var_local_impact`: Impact of the variable computed at the current node of the search tree.
 * :meth:`var_index`: Index of the variable in an array of variables.
 * :meth:`var_success_rate`: Success rate of the variable.
 * :meth:`impact_of_last_branch`: Domain reduction that the last instantiation made by search has achieved on the evaluated variable.
 * :meth:`explicit_var_eval`: Variable evaluator that gives an explicit value to variables.

**Value evaluators**

An evaluator of integer values is an object that is used by value selectors to define instantiation strategies.

 * :meth:`value`: Returns as evaluation the value itself.
 * :meth:`value_impact`: Average reduction of the search space observed so far when instantiating the selected variable to the evaluated value.
 * :meth:`value_success_rate`: Success rate of instantiating the selected variable to the evaluated value.
 * :meth:`value_index`: Index of the value in an array of integer values.
 * :meth:`explicit_value_eval`: Gives an explicit evaluation to values.

**Variable selectors**

A selector of integer variables is used by variable choosers to define search strategies.

 * :meth:`select_smallest`: Selector of integer variables having the smallest evaluation according to a given evaluator.
 * :meth:`select_largest`: Selector of integer variables having the largest evaluation according to a given evaluator.
 * :meth:`select_random_var`: Selector of integer variables that selects a variable randomly.

**Value selectors**

A selector of integer values is used by value choosers to define search strategies.

 * :meth:`select_smallest`: Selector of integer values having the smallest evaluation according to a given evaluator.
 * :meth:`select_largest`: Selector of integer values having the smallest evaluation according to a given evaluator.
 * :meth:`select_random_value`: Selector of integer variable value assignments that selects a domain value randomly.

**Search phases**

A search phase is used to define instantiation strategies to help the embedded CP Optimizer search.

 * :meth:`search_phase`: Create a new search phase.


Detailed description
--------------------
"""

from docplex.cp.catalog import *
from docplex.cp.expression import CpoExpr, CpoFunctionCall, CpoValue, build_cpo_expr, build_cpo_tupleset, \
                                  build_cpo_transition_matrix, INTERVAL_MAX, POSITIVE_INFINITY, NEGATIVE_INFINITY
from docplex.cp.utils import *
import warnings

try:
   from collections.abc import Iterator
except:
    from collections import Iterator


###############################################################################
##  Private methods
###############################################################################

def _expand(arg):
    """ Expand an argument if it is an iterator """
    if isinstance(arg, Iterator):
       return [_expand(x) for x in arg]
    return arg


def _is_cpo_expr(x):
    """ Check if x is a CPO expression, or an array of CPO expressions """
    return isinstance(x, CpoExpr) or (isinstance(x, (list, tuple)) and builtin_all(isinstance(v, CpoExpr) for v in x))


def _is_int_couple(x):
    """ Check if x is a tuple or list of 2 integers """
    return is_array(x) and (len(x) == 2) and is_int(x[0]) and is_int(x[1])


def _is_cpo_array(val):
    """ Check if argument could be mapped into CPO array """
    if isinstance(val, CpoExpr):
        return val.type.is_array
    return is_array(val) and builtin_any(isinstance(x, CpoExpr) for x in val)


# Map of type names
TYPE_NAMES = {Type_Bool:                  "boolean",
              Type_BoolExpr:              "boolean expression",
              Type_BoolExprArray:         "array of boolean expression",
              Type_BoolInt:               "boolean integer",
              Type_Constraint:            "constraint expression",
              Type_CumulAtom:             "cumul atom",
              Type_CumulAtomArray:        "array of cumul atoms",
              Type_CumulExpr:             "cumul expression",
              Type_CumulFunction:         "cumul function",
              Type_Float:                 "float value",
              Type_FloatArray:            "array of floats",
              Type_FloatExpr:             "float expression",
              Type_FloatExprArray:        "array of float expression",
              Type_FloatVar:              "float variable",
              Type_Int:                   "integer",
              Type_IntArray:              "array of integers",
              Type_IntExpr:               "integer expression",
              Type_IntExprArray:          "array of integer expression",
              Type_IntValueChooser:       "chooser of integer value",
              Type_IntValueEval:          "evaluator of integer value",
              Type_IntValueSelector:      "selector of integer value",
              Type_IntValueSelectorArray: "array of integer value selectors",
              Type_IntVar:                "integer variable",
              Type_IntVarArray:           "array of integer variables",
              Type_IntVarChooser:         "chooser of integer variable",
              Type_IntVarEval:            "evaluator of integer variable",
              Type_IntVarSelector:        "selector of integer variable",
              Type_IntVarSelectorArray:   "array of interval variable selectors",
              Type_IntervalArray:         "array of intervals",
              Type_IntervalVar:           "interval variable",
              Type_IntervalVarArray:      "array of interval variables",
              Type_Objective:             "objective function",
              Type_PositiveInt:           "positive integer",
              Type_SearchPhase:           "search phase",
              Type_SegmentedFunction:     "segmented function",
              Type_SequenceVar:           "sequence variable",
              Type_SequenceVarArray:      "array of sequence variables",
              Type_StateFunction:         "state function",
              Type_StepFunction:          "step function",
              Type_TimeInt:               "integer representing a time",
              Type_TransitionMatrix:      "transition matrix",
              Type_TupleSet:              "tuple set",
             }


def _convert_arg(val, name, type, errmsg=None):
    """ Convert a Python value in CPO and check its value
    Args:
        val:  Value to convert
        name: Argument name
        type: Expected type
        errmsg: Optional error message
    """
    val = build_cpo_expr(val)
    assert val.is_kind_of(type), errmsg if errmsg is not None else "Argument '{}' should be a {}".format(name, TYPE_NAMES[type])
    return val


def _convert_arg_bool_int(val, name):
    """ Convert a Python value in CPO bool int
    Args:
        val:  Value to convert
        name: Argument name
    """
    if isinstance(val, CpoExpr):
        assert val.is_kind_of(Type_BoolInt), "Argument '{}' should be a {}".format(name, TYPE_NAMES[Type_BoolInt])
        return val
    return CpoValue(1 if val else 0, Type_BoolInt)


def _check_same_size_arrays(x, y):
    """ Check that two arrays have the same size """
    if (type(x) is CpoValue) and (type(y) is CpoValue):
         assert len(x.value) == len(y.value), "Arguments should be arrays of the same size"


#==============================================================================
#  Arithmetic expressions
#==============================================================================


def plus(e1, e2):
    """ Creates an expression that represents the addition of two expressions.

    The python operator '+' is overloaded to implement a call to this modeling method.
    Writing *plus(e1, e2)* is equivalent to write *(e1 + e2)*.

    Args:
        e1:  First expression that can be integer expression, float expression or cumul expression
        e2:  Second expression that can be integer expression, float expression or cumul expression
    Returns:
        An expression representing (e1 + e2).
        The expression is integer expression, float expression or cumul expression depending on the type of the arguments.
    """
    return build_cpo_expr(e1).__add__(e2)


def minus(e1, e2=None):
    """ Creates an expression that represents the difference between two expressions, or the unary minus of a single one.

    This method can be called with one or two arguments.
    With one argument, it returns an expression representing the unary minus of its argument.
    With two arguments, it creates an expression representing the difference between the first and the second argument.
    Possible argument and return type combinations are:

    The python operator '-' is overloaded to implement a call to this modeling method.
    Writing *minus(e1, e2)* is equivalent to write *(e1 - e2)*, and writing minus(e1) is equivalent to write *(-e1)*.

    Args:
        e1:  First expression that can be integer expression, float expression or cumul expression
        e2:  Optional.
             Second expression that can be integer expression, float expression or cumul expression
    Returns:
        An expression representing (e1 - e2), or -e1 if e2 is not given.
        The expression is integer expression, float expression or cumul expression depending on the type of the arguments.
    """
    if e2 is None:
        return build_cpo_expr(e1).__neg__()
    return build_cpo_expr(e1).__sub__(e2)


def times(e1, e2):
    """ Creates an expression that represents the product of two expressions.

    Args:
        e1:  First expression that can be integer or float expression
        e2:  Second expression that can be integer or float expression
    Returns:
        An expression representing (e1 * e2).
        The expression is integer expression or float expression depending on the type of the arguments.
    """
    return build_cpo_expr(e1).__mul__(build_cpo_expr(e2))


def int_div(e1, e2):
    """ Creates an expression representing the integer division of two expressions.

    The python operator '//' is overloaded to implement a call to this modeling method.
    Writing *int_div(e1, e2)* is equivalent to write *(e1 // e2)*.

    Args:
        e1:  First integer expression
        e2:  Second integer expression
    Returns:
        An integer expression representing (e1 // e2).
    """
    return build_cpo_expr(e1).__floordiv__(build_cpo_expr(e2))


def float_div(e1, e2):
    """ Creates an expression representing the float division of two expressions.

    The python operator '/' is overloaded to implement a call to this modeling method.
    Writing *float_div(e1, e2)* is equivalent to write *(e1 / e2)*.

    Args:
        e1:  First integer or float expression
        e2:  Second integer or float expression
    Returns:
        A float expression representing (e1 / e2).
    """
    return build_cpo_expr(e1).__truediv__(build_cpo_expr(e2))


def mod(e1, e2):
    """ Creates an expression representing the modulo of two expressions.

    The python operator '%' is overloaded to implement a call to this modeling method.
    Writing *mod(e1, e2)* is equivalent to write *(e1 % e2)*.

    Args:
        e1:  First integer expression
        e2:  Second integer expression
    Returns:
        An integer expression representing (e1 % e2).
    """
    return build_cpo_expr(e1).__mod__(e2)


def abs_of(x):
    """ Creates an expression that represents the absolute value of an expression.

    Function *abs_of* computes the absolute value of an integer or floating-point expression *x*.
    The type of the function is the same as the type of its argument.

    *abs_of(x)* is a more efficient way of writing *max(x, -x)*.

    Args:
        x: Integer or floating-point expression for which the absolute value is to be computed.
    Returns:
        Model expression of type float or integer (same as argument type)
    """
    x = build_cpo_expr(x)
    if x.is_kind_of(Type_IntExpr):
        return CpoFunctionCall(Oper_abs, Type_IntExpr, (x,))
    assert x.is_kind_of(Type_FloatExpr), "Argument should be a integer or float expression"
    return CpoFunctionCall(Oper_abs, Type_FloatExpr, (x,))


def abs(x):
    """ Creates an expression that represents the absolute value of an expression.

    Function *abs* computes the absolute value of an integer or floating-point expression *x*.
    The type of the function is the same as the type of its argument.

    *abs(x)* is a more efficient way of writing *max(x, -x)*.

    Implementation of this method is proof to import all functions of this module at root level.
    It recalls the builtin abs() function if no model expression is found in the parameters.

    Args:
        x: Integer or floating-point expression for which the absolute value is to be computed.
    Returns:
        Model expression of type float or integer (same as argument type)
    """
    # Check call to builtin function
    if not _is_cpo_expr(x):
        return builtin_abs(x)
    return abs_of(x)


def square(x):
    """ Returns the square of a numeric expression.

    Function *square* computes the square of *x*.
    Depending on the type of *x* the result is an integer or a floating-point expression.

    Args:
        x: Integer or float expression.
    Returns:
        An expression representing (x * x).
        Type is the same than parameter (integer expression or float expression).
    """
    x = build_cpo_expr(x)
    if x.is_kind_of(Type_IntExpr):
        return CpoFunctionCall(Oper_square, Type_IntExpr, (x,))
    assert x.is_kind_of(Type_FloatExpr), "Argument should be a integer or float expression"
    return CpoFunctionCall(Oper_square, Type_FloatExpr, (x,))


def power(v, p):
    """ Creates an expression that represents the power of an expression by another.

    The python operator '**' is overloaded to implement a call to this modeling method.
    Writing *power(e1, e2)* is equivalent to write *(e1 ** e2)*.

    Args:
        v:  Float expression
        p:  Power float expression
    Returns:
        A float expression representing (e1 ** e2).
    """
    return build_cpo_expr(v).__pow__(p)


def log(x):
    """ Returns the logarithm of an expression

    Args:
        x: A float expression.
    Returns:
        A float expression representing log(*x*)
    """
    return CpoFunctionCall(Oper_log, Type_FloatExpr, (_convert_arg(x, "x", Type_FloatExpr),))


def exponent(x):
    """ Returns the exponentiation of an expression.

    Args:
        x: A float expression.
    Returns:
        A float expression representing exp(*x*)
    """
    return CpoFunctionCall(Oper_exponent, Type_FloatExpr, (_convert_arg(x, "x", Type_FloatExpr),))


def sum_of(x):
    """ Returns the sum of all expressions in an array of expressions.

    The function *sum_of* computes the sum of *x*.
    Depending on the type of *x* the result is an integer, floating-point or cumul expression.

    Args:
        x: An array of integer, floating-point or cumul expressions.
    Returns:
        An integer, float or cumul expression depending on the type of argument.
    """
    # Build model expression
    arr = build_cpo_expr(x)

    # Array of integer expressions
    if arr.is_kind_of(Type_IntExprArray):
        # alen = len(arr.value)
        # if alen == 1:
        #     return arr.value[0]
        # if alen == 2:
        #     return arr.value[0] + arr.value[1]
        return CpoFunctionCall(Oper_sum, Type_IntExpr, (arr,))

    # Array of float expressions
    if arr.is_kind_of(Type_FloatExprArray):
        return CpoFunctionCall(Oper_sum, Type_FloatExpr, (arr,))

    # Array of Cumul expr
    assert arr.is_kind_of(Type_CumulExprArray), "Argument should be an array of integer, float or cumul expressions"
    alen = len(arr.value)
    if alen == 1:
        return arr.value[0]
    if alen == 2:
        return arr.value[0] + arr.value[1]
    return CpoFunctionCall(Oper_sum, Type_CumulExpr, (arr,))


def sum(arr, *args):
    """ Returns the sum of all expressions in an array of expressions.

    The function *sum* computes the sum of all expressions in array *x*.
    If all elements of *x* are integer expressions, result is an integer expression.
    If at least one element of x is a floating-point expression, result is a floating-point expression.

    Implementation of this method is proof to import all functions of this module at root level.
    It recalls the builtin sum() function if no model expression is found in the parameters.

    Args:
        arr: An array of integer or floating-point expressions.
    Returns:
        An integer or float expression depending on the type of argument.
    """
    # Check calls to builtin sum()
    if args:
        return builtin_sum(arr, *args)
    # Check array of model expressions
    arr = _expand(arr)
    return sum_of(arr) if _is_cpo_array(arr) else builtin_sum(arr)


def min_of(arr, *args):
    """ Computes the minimum of an array or a list of integer or floating-point expressions.

    The *min_of* function returns an expression which has the same value as the
    minimum of the supplied arguments.
    The return type corresponds to the type of arguments supplied.

    List of expressions can be given extensively, or as a single iterable of expressions.

    Args:
        arr: Array of integer or floating-point expressions, or first expression of the list
        *args: Other expressions if first argument is not already an iterable
    Returns:
        An integer or float expression according to the type of parameters.
    """
    # Build array of arguments
    if args:
        arr = [arr]
        arr.extend(args)
    arr = build_cpo_expr(arr)

    # Check if single argument that is not an array
    if not arr.type.is_array:
        return arr

    # Determine array and element/result type
    assert arr.is_kind_of(Type_FloatExprArray), "Argument should be an array of integer or float expressions"
    rtype = Type_IntExpr if arr.type.is_kind_of(Type_IntExprArray) else Type_FloatExpr

    # Check single and pair of expressions
    vals = arr.value
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return CpoFunctionCall(Oper_min, rtype, (build_cpo_expr(vals[0]), build_cpo_expr(vals[1])))
    return CpoFunctionCall(Oper_min, rtype, (arr,))


def min(arr, *args, **kwargs):
    """ Computes the minimum of an array of integer or floating-point expressions.

    The *min* function returns an expression which has the same value as the
    minimum of the supplied arguments.
    The return type corresponds to the type of arguments supplied.

    List of expressions can be given extensively, or as a single iterable of expressions.

    Implementation of this method is proof to import all functions of this module at root level.
    It recalls the builtin min() function if no model expression is found in the parameters.

    Args:
        arr: Array of integer or floating-point expressions, or first expression of the list
        *args: Other expressions if first argument is not already an iterable
    Returns:
        An integer or float expression according to the type of parameters.
    """
    # Check calls to builtin min()
    if kwargs:
        return builtin_min(arr, *args, **kwargs)
    if args:
        arr = [arr]
        arr.extend(args)
    arr = _expand(arr)
    return min_of(arr) if _is_cpo_array(arr) else builtin_min(arr)


def max_of(arr, *args):
    """ Computes the maximum of an array or a list of integer or floating-point expressions.

    The *max_of* function returns an expression which has the same value as the
    maximum of the supplied arguments.
    The return type corresponds to the type of arguments supplied.

    List of expressions can be given extensively, or as a single iterable of expressions.

    Args:
        arr: Array of integer or floating-point expressions, or first expression of the list
        *args: Other expressions if first argument is not already an iterable
    Returns:
        An integer or float expression according to the type of parameters.
    """
    # Build array of arguments
    if args:
        arr = [arr]
        arr.extend(args)
    arr = build_cpo_expr(arr)

    # Check if single argument that is not an array
    if not arr.type.is_array:
        return arr

    # Determine array and element/result type
    assert arr.is_kind_of(Type_FloatExprArray), "Argument should be an array of integer or float expressions"
    rtype = Type_IntExpr if arr.type.is_kind_of(Type_IntExprArray) else Type_FloatExpr

    # Check single and pair of expressions
    vals = arr.value
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return CpoFunctionCall(Oper_max, rtype, (build_cpo_expr(vals[0]), build_cpo_expr(vals[1])))
    return CpoFunctionCall(Oper_max, rtype, (arr,))


def max(arr, *args, **kwargs):
    """ Computes the maximum of an array of integer or floating-point expressions.

    The *max* function returns an expression which has the same value as the
    maximum of the supplied arguments.
    The return type corresponds to the type of arguments supplied.

    List of expressions can be given extensively, or as a single iterable of expressions.

    Implementation of this method is proof to import all functions of this module at root level.
    It recalls the builtin max() function if no model expression is found in the parameters.

    Args:
        arr: Array of integer or floating-point expressions, or first expression of the list
        *args: Other expressions if first argument is not already an iterable
    Returns:
        An integer or float expression according to the type of parameters.
    """
    # Check calls to builtin max()
    if kwargs:
        return builtin_max(arr, *args, **kwargs)
    if args:
        arr = [arr]
        arr.extend(args)
    arr = _expand(arr)
    return max_of(arr) if _is_cpo_array(arr) else builtin_max(arr)


def ceil(x):
    """ Rounds a float expression upward to the nearest integer.

    Using this modeling expression implicitly constrain the float expression argument to be
    inside the integer value bounds [INT_MIN, INT_MAX].

    Args:
        x: A float expression.
    Returns:
        An integer expression representing ceil(*x*)
    """
    return CpoFunctionCall(Oper__ceil, Type_IntExpr, (_convert_arg(x, "x", Type_FloatExpr),))


def floor(x):
    """ Rounds a float expression down to the nearest integer.

    Using this modeling expression implicitly constrain the float expression argument to be
    inside the integer value bounds [INT_MIN, INT_MAX].

    Args:
        x: A float expression.
    Returns:
        An integer expression representing floor(*x*)
    """
    return CpoFunctionCall(Oper__floor, Type_IntExpr, (_convert_arg(x, "x", Type_FloatExpr),))


def round(x):
    """ Rounds a float expression to the nearest integer.

    Using this modeling expression implicitly constrain the float expression argument to be
    inside the integer value bounds [INT_MIN, INT_MAX].

    Args:
        x: A float expression.
    Returns:
        An integer expression representing round(*x*)
    """
    return CpoFunctionCall(Oper__round, Type_IntExpr, (_convert_arg(x, "x", Type_FloatExpr),))


def trunc(x):
    """ Builds the truncated integer parts of a float expression.

    Using this modeling expression implicitly constrain the float expression argument to be
    inside the integer value bounds [INT_MIN, INT_MAX].

    Args:
        x: A float expression.
    Returns:
        An integer expression representing trunc(*x*)
    """
    return CpoFunctionCall(Oper__trunc, Type_IntExpr, (_convert_arg(x, "x", Type_FloatExpr),))


def sgn(x):
    """ Builds the sign of a float expression.

    The sign of the expression is an integer whose value is -1, 0 or +1 if expression
    *x* is respectively lower than, equal to, or greater than zero.

    Args:
        x: A float expression.
    Returns:
        An integer expression representing sign(*x*) (-1, 0 or +1)
    """
    return CpoFunctionCall(Oper__sgn, Type_IntExpr, (_convert_arg(x, "x", Type_FloatExpr),))


#==============================================================================
#  Logical expressions
#==============================================================================


def all_of(lexpr):
    """ Creates an expression representing the logical AND of an array of boolean expressions.

    Args:
        lexpr:  Array (iterable) of boolean expressions
    Returns:
        A boolean expression representing the logical AND of all expressions in lexpr.
        If the array is empty array of expressions, result is CP constant True.
    """
    # Build and of expressions as list of logical and
    if not lexpr:
        return true()
    assert is_array(lexpr), "Argument should be an array of boolean expressions"
    res = None
    for v in lexpr:
        v = build_cpo_expr(v)
        assert v.is_kind_of(Type_BoolExpr), "Each array element should be a boolean expression"
        res = v if res is None else res & v
    return res


def all(lexpr):
    """ Creates an expression representing the logical AND of an array of expressions.

    Implementation of this method is proof to import all functions of this module at root level.
    It recalls the builtin all() function if no model expression is found in the parameters.

    Args:
        lexpr:  Array (iterable) of expressions
    Returns:
        A boolean expression representing the logical AND of all expressions in lexpr.
        Or the result of the builtin function all() if no model expression is in the list.
    """
    # Build and of expressions as list of logical and
    lexpr = _expand(lexpr)
    return all_of(lexpr) if _is_cpo_array(lexpr) else builtin_all(lexpr)


def any_of(lexpr):
    """ Creates an expression representing the logical OR of an array of boolean expressions.

    Args:
        lexpr:  Array (iterable) of boolean expressions
    Returns:
        A boolean expression representing the logical OR of all expressions in lexpr.
        If the array is empty array of expressions, result is CP constant False.
    """
    # Build and of expressions as list of logical and
    if not lexpr:
        return false()
    assert is_array(lexpr), "Argument should be an array of boolean expressions"
    res = None
    for v in lexpr:
        v = build_cpo_expr(v)
        assert v.is_kind_of(Type_BoolExpr), "Each array element should be a boolean expression"
        res = v if res is None else res | v
    return res


def any(lexpr):
    """ Creates an expression representing the logical OR of an array of expressions.

    Implementation of this method is proof to import all functions of this module at root level.
    It recalls the builtin any() function if no model expression is found in the parameters.

    Args:
        lexpr:  Array (iterable) of expressions
    Returns:
        A boolean expression representing the logical OR of all expressions in lexpr.
        Or the result of the builtin function any() if no model expression is in the list.
    """
    # Build and of expressions as list of logical and
    lexpr = _expand(lexpr)
    return any_of(lexpr) if _is_cpo_array(lexpr) else builtin_any(lexpr)


def logical_and(e1, e2=None):
    """ Creates an expression representing the logical AND of two boolean expressions,
    or of an array of expressions.

    The python operator '&' is overloaded to implement a call to this modeling method.
    Writing *logical_and(e1, e2)* is equivalent to write *(e1 & e2)*.

    Note that python keyword 'and' can not be overloaded. Operator '&' is an operator that is usually
    used for binary operations, with a priority that is different that the logical 'and'.
    We recommend to always fully parenthesise expressions that use such binary operators in place of logical operators.

    Args:
        e1:  First boolean expression, or array of expressions
        e2:  (Optional) Second boolean expression.
    Returns:
        A boolean expression representing (e1 and e2), or logical and of all expressions in array e1.
        If e1 is an empty array of expressions, result is CP constant True.
    """
    if e2 is not None:
        return CpoFunctionCall(Oper_logical_and, Type_BoolExpr, (_convert_arg(e1, "e1", Type_BoolExpr),
                                                                 _convert_arg(e2, "e2", Type_BoolExpr)))
    # Build and of expressions as list of logical and
    return all_of(_expand(e1))


def logical_or(e1, e2=None):
    """ Creates an expression representing the logical OR of two boolean expressions,
    or of an array of expressions.

    The python operator '|' is overloaded to implement a call to this modeling method.
    Writing *logical_or(e1, e2)* is equivalent to write *(e1 | e2)*.

    Note that python keyword 'or' can not be overloaded. Operator '|' is an operator that is usually
    used for binary operations, with a priority that is different that the logical 'or'.
    We recommend to always fully parenthesise expressions that use such binary operators in place of logical operators.

    Args:
        e1:  First boolean expression, or array of expressions
        e2:  (Optional) Second boolean expression.
    Returns:
        A boolean expression representing (e1 or e2), or logical or of all expressions in array e1.
        If e1 is an empty array of expressions, result is CP constant False.
    """
    if e2 is not None:
        return CpoFunctionCall(Oper_logical_or, Type_BoolExpr, (_convert_arg(e1, "e1", Type_BoolExpr),
                                                                _convert_arg(e2, "e2", Type_BoolExpr)))
    # Build and of expressions as list of logical or
    return any_of(_expand(e1))


def logical_not(e):
    """ Creates an expression representing the logical NOT of a boolean expression.

    The python operator '~' is overloaded to implement a call to this modeling method.
    Writing *logical_not(e* is equivalent to write *(~e)*.

    Note that python keyword 'not' can not be overloaded. Operator '~' is an operator that is usually
    used for binary operations, with a priority that is different that the logical 'not'.
    We recommend to always fully parenthesise expressions that use such binary operators in place of logical operators.

    Args:
        e:  Boolean expression
    Returns:
        A boolean expression representing not(e).
    """
    return CpoFunctionCall(Oper_logical_not, Type_BoolExpr, (_convert_arg(e, "e", Type_BoolExpr),))


def equal(e1, e2):
    """ Creates an expression representing the equality between two expressions.

    The python operator '==' is overloaded to implement a call to this modeling method.
    Writing *equal(e1, e2)* is equivalent to write *(e1 == e2)*.

    Args:
        e1:  Integer or float expression
        e2:  Integer or float expression
    Returns:
        A boolean expression representing (e1 == e2)
    """
    return build_cpo_expr(e1).__eq__(e2)


def diff(e1, e2):
    """ Creates an expression representing the inequality of two expressions.

    The python operator '!=' is overloaded to implement a call to this modeling method.
    Writing *diff(e1, e2)* is equivalent to write *(e1 != e2)*.

    Args:
        e1:  Integer or float expression
        e2:  Integer or float expression
    Returns:
        A boolean expression representing (e1 != e2)
    """
    return build_cpo_expr(e1).__ne__(e2)


def greater(e1, e2):
    """ Creates an expression representing that an expression is greater than another.

    The python operator '>' is overloaded to implement a call to this modeling method.
    Writing *greater(e1, e2)* is equivalent to write *(e1 > e2)*.

    Args:
        e1:  Integer or float expression
        e2:  Integer or float expression
    Returns:
        A boolean expression representing (e1 > e2)
    """
    return build_cpo_expr(e1).__gt__(e2)


def greater_or_equal(e1, e2):
    """ Creates an expression representing that an expression is greater or equal to another.

    The python operator '>=' is overloaded to implement a call to this modeling method.
    Writing *greater_or_equal(e1, e2)* is equivalent to write *(e1 >= e2)*.

    Args:
        e1:  Integer, float or cumul expression
        e2:  Integer, float or cumul expression
    Returns:
        A boolean expression representing (e1 >= e2)
        A constraint expression if at least one argument is a cumul expression.
    """
    return build_cpo_expr(e1).__ge__(e2)


def less(e1, e2):
    """ Creates an expression representing that an expression is less than another.

    The python operator '<' is overloaded to implement a call to this modeling method.
    Writing *less(e1, e2)* is equivalent to write *(e1 < e2)*.

    Args:
        e1:  Integer or float expression
        e2:  Integer or float expression
    Returns:
        A boolean expression representing (e1 < e2)
    """
    return build_cpo_expr(e1).__lt__(e2)


def less_or_equal(e1, e2):
    """ Creates an expression for operation *lessOrEqual*.

    The python operator '<=' is overloaded to implement a call to this modeling method.
    Writing *less(e1, e2)* is equivalent to write *(e1 <= e2)*.

    Args:
        e1:  Integer, float or cumul expression
        e2:  Integer, float or cumul expression
    Returns:
        A boolean expression representing (e1 <= e2).
        A constraint expression if at least one argument is a cumul expression.
    """
    return build_cpo_expr(e1).__le__(e2)


def true():
    """ Returns a true boolean expression.

    The function *true()* does not have any particular purpose except for being a filler.

    CP Optimizer usually eliminates *true()* from expressions using partial evaluation.

    Returns:
        A boolean true expression
    """
    return CpoFunctionCall(Oper_true, Type_BoolExpr, ())


def false():
    """ Returns a false boolean expression.

    The function *false()* does not have any particular purpose except for being a filler.

    CP Optimizer usually eliminates *false()* from expressions using partial evaluation.

    Returns:
        An expression of type boolean expression
    """
    return CpoFunctionCall(Oper_false, Type_BoolExpr, ())


#==============================================================================
#  Miscellaneous modeling functions
#==============================================================================


def all_diff(arr, *args):
    """ Returns a constraint stating that multiple expressions must be all different.

    Args:
        arr: Array (list, tuple or iterable) of expressions, or first element of the list
        *args: Other expressions if first argument is a single expression
    Returns:
        New constraint expression.
    """
    if args:
        arr = [arr]
        arr.extend(args)
    return CpoFunctionCall(Oper_all_diff, Type_Constraint, (_convert_arg(arr, "arr", Type_IntExprArray),))


def abstraction(y, x, values, abstractValue):
    """ Returns a constraint that abstracts the values of one array as values in another array.

    For constraint programming: returns a constraint that abstracts the values of expressions contained in one
    array to expressions contained in another array.

    This function returns a constraint that abstracts the values of the elements of one array of expressions
    (called *x*) in a model into the abstract value of another array of expressions (called *y*).
    In other words, for each element *x[i]*, there is an expression *y[i]* corresponding to the abstraction
    of *x[i]* with respect to an array of numeric *values*.
    That is:

     * *x[i] = v* with *v* in *values* if and only if *y[i] = v*
     * *x[i] = v* with *v* not in *values* if and only if *y[i] = abstractValue*

    This constraint maintains a many-to-one mapping that makes it possible to define constraints that impinge
    only on a particular set of values from the domains of expressions.
    The abstract value (specified by *abstractValue*) must not be in the domain of *x[i]*.

    Args:
        y: An array of abstracted integer expressions.
        x: An array of reference integer expressions.
        values: An array of integer values to be abstracted.
        abstractValue: An escape value (integer constant)
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_abstraction, Type_Constraint, (_convert_arg(y, "y", Type_IntExprArray),
                                                               _convert_arg(x, "x", Type_IntExprArray),
                                                               _convert_arg(values, "values", Type_IntArray),
                                                               _convert_arg(abstractValue, "abstractValue", Type_Int), ))


def bool_abstraction(y, x, values):
    """ Creates a constraint that abstracts the values of one array as boolean values in another array.

    This function creates and returns a constraint that abstracts an array of integer expressions in a model.
    It differs from :meth:`abstraction` in that elements each *y[i]* is Boolean.

    Like :meth:`abstraction`, for each element *x[i]* there is an expression *y[i]* corresponding to the
    abstraction of *x[i]* with respect to the *values* array.
    That is:

     * *x[i] = v* with *v* in *values* if and only if *y[i] = true()*
     * *x[i] = v* with *v* not in *values* if and only if *y[i] = false()*

    This constraint maintains a many-to-one mapping that makes it possible to define constraints that impinge
    only on a particular set of values from the domains of constrained variables.

    Args:
        y: An array of abstracted integer expressions.
        x: An array of reference integer expressions.
        values: An array of integer values to abstract.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_bool_abstraction, Type_Constraint, (_convert_arg(x, "x", Type_IntExprArray),
                                                                    _convert_arg(y, "y", Type_IntExprArray),
                                                                    _convert_arg(values, "values", Type_IntArray)
                                                                    ))


def count(exprs, v):
    """ Returns the number of occurrences of a given expression in a given array of integer expressions.

    This expression counts how many of the expressions in *exprs* take the value *v*.

    For convenience reasons, parameters can be expressed in the reverse order.

    Args:
        exprs: An array of integer expressions.
        v:     The value (integer) for which occurrences must be counted.
    Returns:
        An integer expression representing the number of occurrences of *v* in *exprs*
    """
    exprs = build_cpo_expr(exprs)
    v = build_cpo_expr(v)
    if v.is_kind_of(Type_Int):
        assert exprs.is_kind_of(Type_IntExprArray), "Argument 'exprs' should be an array of integer expressions"
    else:
        assert v.is_kind_of(Type_IntExprArray) and exprs.is_kind_of(Type_Int), "Arguments should be an integer and array of integer expressions"
    return CpoFunctionCall(Oper_count, Type_IntExpr, (exprs, v))


def count_different(exprs):
    """ Creates an expression that counts the number of different values in an array of integer expressions.

    Args:
        exprs: An array of integer expressions.
    Returns:
        An integer expression representing the number of values that are different in *exprs*.
    """
    return CpoFunctionCall(Oper_count_different, Type_IntExpr, (_convert_arg(exprs, "exprs", Type_IntExprArray), ))


def scal_prod(x, y):
    """ Returns the scalar product of two vectors.

    The function *scal_prod* returns an integer or floating-point expression that represents the scalar product
    of two vectors *x* and *y*.
    Depending on the type of *x* and *y* the result is either integer or floating-point expression.

    The versions with one array of constants, integer or float, can be slightly faster.

    Each argument can be:

     * an array of integer constants,
     * an array of float constants,
     * an array of integer expressions
     * an array of float expressions.

    Args:
        x: First input array to be multiplied.
        y: Second input array to be multiplied.
    Returns:
        An expression of type float expression or integer expression
    """
    x = build_cpo_expr(x)
    y = build_cpo_expr(y)
    _check_same_size_arrays(x, y)

    if x.is_kind_of(Type_IntExprArray) and y.is_kind_of(Type_IntExprArray):
        return CpoFunctionCall(Oper_scal_prod, Type_IntExpr, (x, y))

    assert x.is_kind_of(Type_FloatExprArray) and y.is_kind_of(Type_FloatExprArray), "Arguments should be arrays of integer or float expressions"
    return CpoFunctionCall(Oper_scal_prod, Type_FloatExpr, (x, y))


def pack(load, where, size, used=None):
    """ Maintains the load on a set of containers given objects sizes and assignments.

    The *pack* constraint is used to represent sub-problems where the requirement is to assign objects to
    containers such that the capacities or minimum fill levels of the containers are respected.

    Let's assume we have *n* objects and *m* containers.
    The sizes of the array arguments of *pack* must correspond to these constants,
    that is *load* must be of size *m*, whereas *where* and *size* must be of size *n*.
    Given assignments to the *where* expressions, the *pack* constraint will calculate the values of
    the *load* and *used* expressions.

    All counting is done from 0, and so the interpretation of 5 being assigned to *where[3]* is that object 3
    (the 4th object) is placed into container 5 (the 6th container).
    This will be reflected by the inclusion of the size of object 3 (*size[3]*) being included in the calculation
    of the value of *load[5]*.

    Naturally, all the arguments (with the exception of *size*) can be constrained by additional constraints.
    The most common form is to limit the capacity of a container.
    For example, to limit container 2 to a capacity of 15, one would write *load[2] <= 15*.
    Minimum fill level requirements can also be specified this way: for example *load[2] >= 12*.
    Other more esoteric constraints are possible, for example to say that only an even number of containers
    can be used: *(used % 2) == 0*.
    If *used* is omitted from the signature, then it will not be possible to specifically
    constrain the number of containers used.

    Args:
        load: An array of integer expressions, each element representing the load (total size of the objects inside)
              of the corresponding container.
        where: An array of integer expressions, each element representing in which container the
               corresponding object is placed.
        size: An array of integers, each element representing the size of the corresponding object.
        used: (Optional) An integer expression indicating the number of used containers.
              That is, the number of containers with at least one object inside.
    Returns:
        Constraint expression
    """
    load  = _convert_arg(load, "load", Type_IntExprArray)
    where = _convert_arg(where, "where", Type_IntExprArray)
    size  = _convert_arg(size, "size", Type_IntArray)
    if used is None:
        return CpoFunctionCall(Oper_pack, Type_Constraint, (load, where, size))
    return CpoFunctionCall(Oper_pack, Type_Constraint, (load, where, size, _convert_arg(used, "used", Type_IntExpr)))


def sequence(min, max, width, vars, values, cards):
    """ Constrains the number of occurrences of the values taken by the different subsets of consecutive *k* variables.

    This constraint ensures:

     * that *cards[i]* will be equal to the number of occurrences of the value *values[i]* in the array *vars*,
     * and that each sequence of *width* consecutive variables (like *vars[j+1]*, *vars[j+2]*, ..., *vars[j+width]*)
       takes at least *min* and at most *max* values of the array *values*.

    Args:
        min:    The minimum number of allowable values (integer)
        max:    The maximum number of allowable values (integer)
        width:  The size of the sequences of consecutive variables (integer)
        vars:   The array of variables (array of integer expresssions)
        values: The array of values (array of integers)
        cards:  The array of cardinality variables (array of integer expressions)
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_sequence, Type_Constraint, (_convert_arg(min, "min", Type_Int),
                                                            _convert_arg(max, "max", Type_Int),
                                                            _convert_arg(width, "width", Type_Int),
                                                            _convert_arg(vars, "vars", Type_IntExprArray),
                                                            _convert_arg(values, "values", Type_IntArray),
                                                            _convert_arg(cards, "cards", Type_IntExprArray),
                                                            ))


def constant(v):
    """ Creates an expression from a numeric constant.

    This method is generally useless in Python as constants are automatically converted in model expressions.
    However, it can be useful if the expression has to be named.

    Args:
        v:  integer or float constant
    Returns:
        A integer expression or float expression depending on the type of *v*.
    """
    v = build_cpo_expr(v)
    vt = v.type
    if vt is Type_Int:
        nt = Type_IntExpr
    elif vt is Type_Float:
        nt = Type_FloatExpr
    else:
        raise AssertionError("Argument should be a integer or float constant")
    return CpoFunctionCall(Oper_constant, nt, (v,))


def element(array, index):
    """ This function returns an element of a given array indexed by an integer expression.

    This function returns an expression for use in a constraint or other expression.
    The semantics of this expression are: when *index* takes the value *i*,
    then the value of the expression is equal to *array[i]*.

    For convenience reasons, parameters can be expressed in any order.

    Possible argument and return type combinations are:

     * (array of integers, integer expression) => integer expression
     * (array of integer expressions, integer expression) => integer expression
     * (array of floats, integer expression) => float expression
     * (integer expression, array of integers) => integer expression
     * (integer expression, array of integer expressions) => integer expression
     * (integer expression, array of floats) => float expression

    Args:
        array:  An array in which an element will be selected using subscript.
        index:  An integer expression used to retrieve array element.
    Returns:
        A float or integer expression depending on the type of the array.
    """
    array = build_cpo_expr(array)
    index = build_cpo_expr(index)
    # Case where 'index' is the index
    if index.is_kind_of(Type_IntExpr):
        if array.is_kind_of(Type_IntExprArray):  # Type_IntArray is included
            return CpoFunctionCall(Oper_element, Type_IntExpr, (array, index))
        assert array.is_kind_of(Type_FloatArray), "Array should be an array of integer, float, or integer expressions"
        return CpoFunctionCall(Oper_element, Type_FloatExpr, (array, index))
    # Case where 'array' is in fact the index
    assert array.is_kind_of(Type_IntExpr), "At least one argument should be an integer expression representing index"
    if index.is_kind_of(Type_IntExprArray):  # Type_IntArray is included
        return CpoFunctionCall(Oper_element, Type_IntExpr, (array, index))
    assert index.is_kind_of(Type_FloatArray), "Array should be an array of integer, float, or integer expressions"
    return CpoFunctionCall(Oper_element, Type_FloatExpr, (array, index))


def member(element, array):
    """ Checks if an integer expression is member of an array of expressions.

    DEPRECATED: use :meth:`allowed_assignments`: instead.

    Args:
        element:  Integer expression whose value is to check in the array.
        array:    Array of integer values
    Returns:
        A boolean expression denoting the presence of the value in the array.
    """
    warnings.warn("Modeling function 'member' is deprecated. Use allowed_assignments instead", DeprecationWarning)
    return CpoFunctionCall(Oper_member, Type_BoolExpr, (_convert_arg(element, "element", Type_IntExpr),
                                                        _convert_arg(array, "array", Type_IntArray),
                                                        ))


def in_range(x, lb, ub):
    """ Restricts the bounds of an integer or floating-point expression.

    This boolean expression (which is interpreted as a constraint outside of an expression)
    determines whether the value of expression *x* is inside the range *[lb, ub]*.
    The returned expression will be true if and only if *x* is no less than *lb* and no greater than *ub*.

    *range(y, a, b)* is also a more efficient form of writing *a <= y && y <= b*.

    Args:
        x: The integer or floating-point expression.
        lb: The lower bound.
        ub: The upper bound.
    Returns:
        An expression of type boolean expression
    """
    return CpoFunctionCall(Oper_range, Type_BoolExpr, (_convert_arg(x, "x", Type_FloatExpr),
                                                       _convert_arg(lb, "lb", Type_Float),
                                                       _convert_arg(ub, "ub", Type_Float)))


def range(x, lb=None, ub=None):
    """ Restricts the bounds of an integer or floating-point expression.

    This boolean expression (which is interpreted as a constraint outside of an expression)
    determines whether the value of expression *x* is inside the range *[lb, ub]*.
    The returned expression will be true if and only if *x* is no less than *lb* and no greater than *ub*.

    *range(y, a, b)* is also a more efficient form of writing *a <= y && y <= b*.

    Implementation of this method is proof to import all functions of this module at root level.
    It recalls the builtin range() function if no model expression is found in the parameters.

    Args:
        x:  The integer or floating-point expression.
        lb: The lower bound.
        ub: The upper bound.
    Returns:
        An expression of type boolean expression
    """
    if lb is None:
        return builtin_range(x)
    if ub is None:
        return builtin_range(x, lb)
    if is_int(x) and is_int(lb) and is_int(ub):
        return builtin_range(x, lb, ub)
    return in_range(x, lb, ub)


def all_min_distance(exprs, distance):
    """ Constraint on the minimum absolute distance between a pair of integer expressions in an array.

    This constraint makes sure that the absolute distance between any pair
    of integer expressions in *exprs* will be greater than or equal to the
    given integer *distance*. In short, for any *i*, *j* distinct indices of *exprs* , it
    enforces *abs(exprs[i] - exprs[j]) >= distance*.

    Args:
        exprs:    Array of integer expressions.
        distance: Integer value used to constrain the distance between two elements of exprs.
    Returns:
        A new boolean expression
    """
    return CpoFunctionCall(Oper_all_min_distance, Type_BoolExpr, (_convert_arg(exprs, "exprs", Type_IntExprArray),
                                                                  _convert_arg(distance, "distance", Type_Int)))


def if_then(e1, e2):
    """ Creates and returns the new constraint e1 => e2.

    Args:
        e1:  First boolean expression
        e2:  Second boolean expression
    Returns:
        A boolean expression stating that e1 => e2
    """
    return CpoFunctionCall(Oper_if_then, Type_BoolExpr, (_convert_arg(e1, "e1", Type_BoolExpr),
                                                         _convert_arg(e2, "e2", Type_BoolExpr)))


def conditional(c, et, ef):
    """ Creates and returns an expression that depends on a condition.

    This expression is equivalent of writing (c ? et : ef) in C++ or Java.

    Args:
        c:   Boolean expression
        et:  Integer expression to return if condition *c* is true.
        ef:  Integer expression to return if condition *c* is false.
    Returns:
        Integer expression *et* or *ef* depending on the value of *c*.
    """
    return element(c, [_convert_arg(ef, "ef", Type_IntExpr), _convert_arg(et, "et", Type_IntExpr)])


def inverse(f, invf):
    """ Constrains elements of one array to be inverses of another.

    This function creates an inverse constraint such that if the length of
    the arrays *f* and *invf* is *n*, then this function returns a
    constraint that ensures that:

     * for all *i* in the interval *[0, n-1]*, *invf[f[i]]* == i*
     * for all *j* in the interval *[0, n-1]*, *f[invf[j]]* == j*

    Args:
        f:    An integer expression array.
        invf: An integer expression array.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_inverse, Type_Constraint, (_convert_arg(f, "f", Type_IntExprArray),
                                                           _convert_arg(invf, "invf", Type_IntExprArray)))


def distribute(counts, exprs, values=None):
    """ Calculates and/or constrains the distribution of values taken by an array
    of integer expressions.

    The *distribute* constraint is used to count the number of occurrences of  several values in
    an array of constrained expressions.
    You can also use *distribute* to force a set of constrained expressions to assume
    values in such a way that only a limited number of the constrained
    expressions can assume each value.

    More precisely, for any index *i* of *counts*, *counts[i]* is equal to the number of
    expressions in *exprs* who have value of *values[i]*.
    When using the signature which has *values* missing, then the values counted are assumed to be
    a set spanning from 0 up to the size of the *counts* array, less one.

    Args:
        counts: An array of integer expressions representing, for each element of values, its cardinality in exprs.
        exprs:  An array of integer expressions for which value occurrences must be counted.
        values: (Optional) An integer array containing values to count.
    Returns:
        Constraint expression
    """
    if values is None:
        return CpoFunctionCall(Oper_distribute, Type_Constraint, (_convert_arg(counts, "counts", Type_IntExprArray),
                                                                  _convert_arg(exprs, "exprs", Type_IntExprArray)))
    return CpoFunctionCall(Oper_distribute, Type_Constraint, (_convert_arg(counts, "counts", Type_IntExprArray),
                                                              _convert_arg(values, "values", Type_IntArray),
                                                              _convert_arg(exprs, "exprs", Type_IntExprArray)))
def allowed_assignments(exprs, values):
    """ Explicitly defines possible assignments on one or more integer expressions.

    This boolean expression (which is interpreted as a constraint outside of an expression)
    determines whether the assignment to a single expression or to an array of expressions
    is contained within a value set or a tuple set respectively.

    The boolean expression will be true if and only if (depending on the signature):

     * the single value of the integer expression *exprs* is present in the array *values*.
     * the values of the integer expressions *exprs* are present in the tuple set *tuples*.

    The order of the constrained variables in the array *exprs* is important because the same order
    is respected in the tuple set *tuples*.

    Args:
        exprs:  A single integer expression, or an array of integer expressions
        values: An array of integer expressions, or a set of tuples,
                that specifies the combinations of allowed values of the expressions exprs.
    Returns:
        A boolean expression
    """
    exprs = build_cpo_expr(exprs)
    if exprs.is_kind_of(Type_IntExpr):
        # Expr is a single integer
        return CpoFunctionCall(Oper_allowed_assignments, Type_BoolExpr, (exprs, _convert_arg(values, values, Type_IntArray)))

    # 'expr' is an array of expressions, and 'values' a tupleset
    assert exprs.is_kind_of(Type_IntExprArray), "Argument 'exprs' should be an array of integer or an array of integer expressions"
    tset = build_cpo_tupleset(values)
    tvals = tset.value
    if tvals:
        assert len(exprs.children) == len(tvals[0]), "Arity of tupleset should match the number of expressions"
    return CpoFunctionCall(Oper_allowed_assignments, Type_BoolExpr, (exprs, tset))


def forbidden_assignments(exprs, values):
    """ Explicitly defines forbidden assignments for one or more integer expressions.

    This boolean expression (which is interpreted as a constraint outside of an expression)
    determines whether the assignment to a single expression or to an array of expressions
    is not contained within a value set or a tuple set respectively.

    The boolean expression will be true if and only if (depending on the signature):

     * the single value of the integer expression *exprs* is not present in the array *values*.
     * the values of the array of integer expressions *exprs* are not present in the tuple set *values*.

    The order of the constrained variables in the array *exprs* is important because the same order
    is respected in the tuple set *tuples*.

    Args:
        exprs:  A single integer expression, or an array of integer expressions
        values: An array of integer expressions, or a set of tuples,
                that specifies the combinations of forbidden values of the expressions exprs.
    Returns:
        A boolean expression
    """
    exprs = build_cpo_expr(exprs)
    if exprs.is_kind_of(Type_IntExpr):
        return CpoFunctionCall(Oper_forbidden_assignments, Type_BoolExpr, (exprs, _convert_arg(values, values, Type_IntArray)))

    # 'expr' is an array of expressions, and 'values' a tupleset
    assert exprs.is_kind_of(Type_IntExprArray), "Argument 'exprs' should be an array of integer or an array of integer expressions"
    tset = build_cpo_tupleset(values)
    tvals = tset.value
    if tvals:
        assert len(exprs.children) == len(tvals[0]), "Arity of tupleset should match the number of expressions"
    return CpoFunctionCall(Oper_forbidden_assignments, Type_BoolExpr, (exprs, tset))


def lexicographic(x, y):
    """ Returns a constraint which maintains two arrays to be lexicographically ordered.

    The *lexicographic* function returns a constraint which maintains two arrays to be
    lexicographically ordered.

    More specifically, *lexicographic(x, y)* maintains that *x* is less than or equal to *y*
    in the lexicographical sense of the term.
    This means that either both arrays are equal or that there exists *i < size(x)* such that
    for all *j < i*, *x[j] = y[j]* and *x[i] < y[i]*.

    Note that the size of the two arrays must be the same.

    Args:
        x: An array of integer expressions.
        y: An array of integer expressions.
    Returns:
        Constraint expression
    """
    x = _convert_arg(x, "x", Type_IntExprArray)
    y = _convert_arg(y, "y", Type_IntExprArray)
    _check_same_size_arrays(x, y)
    return CpoFunctionCall(Oper_lexicographic, Type_Constraint, (x, y))


def strict_lexicographic(x, y):
    """ Returns a constraint which maintains two arrays to be strictly lexicographically ordered.

    The *strict_lexicographic* function returns a constraint which maintains two arrays to be
    strictly lexicographically ordered.

    More specifically, *strict_lexicographic(x, y)* maintains that *x* is strictly less than *y*
    in the lexicographical sense of the term.
    This means that there exists *i < size(x)* such that
    for all *j < i*, *x[j] = y[j]* and *x[i] < y[i]*.

    Note that the size of the two arrays must be the same.

    This function is supported only by CPO solver whose version is greater than 12.10.

    Args:
        x: An array of integer expressions.
        y: An array of integer expressions.
    Returns:
        Constraint expression
    """
    x = _convert_arg(x, "x", Type_IntExprArray)
    y = _convert_arg(y, "y", Type_IntExprArray)
    _check_same_size_arrays(x, y)
    return CpoFunctionCall(Oper_strict_lexicographic, Type_Constraint, (x, y))


def standard_deviation(x, meanLB=NEGATIVE_INFINITY, meanUB=POSITIVE_INFINITY):
    """ Creates a constrained numeric expression equal
    to the standard deviation of the values of the variables in an array.

    This function creates a new constrained numeric expression which is equal to the
    standard deviation of the values of the variables in the array *x*.

    The mean of the values of the variables in the array x is constrained to be in the
    interval [meanLB, meanUB].

    Args:
        x:      An array of integer expressions.
        meanLB (Optional): A float value for lower bound on the mean of the array, -infinity if not given.
        meanUB (Optional): A float value upper bound on the mean of the array, infinity if not given.
    Returns:
        A float expression
    """
    # Check optional bounds
    # if (meanLB is NEGATIVE_INFINITY) and (meanUB is POSITIVE_INFINITY) and (_get_generation_version() > "12.8"):
    #     return CpoFunctionCall(Oper_standard_deviation, Type_FloatExpr, (_convert_arg(x, "x", Type_IntExprArray),))
    return CpoFunctionCall(Oper_standard_deviation, Type_FloatExpr, (_convert_arg(x, "x", Type_IntExprArray),
                                                                     _convert_arg(meanLB, "meanLB", Type_Float),
                                                                     _convert_arg(meanUB, "meanUB", Type_Float),))


def strong(x):
    """ A model annotation to encourage CP Optimizer to produce stronger (higher inference) constraints.

    The *strong* constraint strengthens the model on the expressions *x*.
    This is done by creating an *allowed_assignments* constraint in place of the *strong* constraint
    during presolve.
    Only the assignments to the expressions which do not result in an immediate inconsistency are
    added to the tuple set of the *allowed_assignments* constraint.

    Constraints that can be identified as redundant (when taken together with this new constraint)
    are removed from the model during presolve.
    This is the case for constraints that are only over the variables of the array given as argument.

    Args:
        x: An array of integer expressions over which propagation is to be strengthened.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_strong, Type_Constraint, (_convert_arg(x, "x", Type_IntExprArray),))


def slope_piecewise_linear(x, points, slopes, refX, refY):
    """ Evaluates piecewise-linear function given by set of breaking points and slopes.

    This function evaluates piecewise-linear function at a point *x*.
    The function consists of several segments separated by *points*, within each segment the function is linear.
    The function is defined by slopes of all segments (*slopes*) and by breaking points (*points*) on x-axis.
    Furthermore it is necessary to specify reference value *refX* and corresponding function value *refY*.

    The function is continuous unless some value in *points* is specified twice.
    Specifying the same value in *points* allows to model discontinuous function,
    in this case the corresponding value in *slopes* is not interpreted as a slope but as the height of the jump (delta)
    at that point.

    Assuming that the array *points* has size *n*, the function consists of the following linear segments:

     * the segment 0 is defined on interval (-infinity, *points*[0]) and has a *slope*[0].
     * the segment i, i in 1, 2, .., n-1, is defined on the interval [*points*[i-1], *points*[i]) with a slope *slope*[i].
     * the segment n is defined on the interval [*points*[n-1], infinity) with a slope *slope*[n].

    Args:
        x:      x-value for which the function should be evaluated.
        points: sorted array of n-1 x-values (breaking points) that separate n function segments.
        slopes: array of n slopes, one for each segments.
        refX:   reference x-value.
        refY:   value of the function at refX(reference y-value).
    Returns:
        Value of the function at point x.
    """
    return CpoFunctionCall(Oper_slope_piecewise_linear, Type_FloatExpr,
                           (_convert_arg(x,      "x",      Type_FloatExpr),
                            _convert_arg(points, "points", Type_FloatArray),
                            _convert_arg(slopes, "slopes", Type_FloatArray),
                            _convert_arg(refX,   "refX",   Type_FloatExpr),
                            _convert_arg(refY,   "refY",   Type_FloatExpr),
                            ))


def coordinate_piecewise_linear(x, firstSlope, points, values, lastSlope):
    """ Evaluates piecewise-linear function given by set of breaking points and values.

    This function evaluates piecewise-linear function at point *x*.
    The function consists of several segments separated by *points*, within each segment the function is linear.
    The function is defined by slope of the first segment (*firstSlope*), an array of breaking points (*points*)
    on x-axis, an array of corresponding values on y-axis (*values*) and the slope of the last segment.
    In each segment the function is linear.
    The function may be discontinuous, in this case it is necessary to specify the point of discontinuity
    twice in *points*.

    Assuming that the common length of arrays *points* and *values* is *n*, the function consists of the
    following linear segments:

     * the segment 0 is defined on interval (-infinity, *points*[0]) and is a linear function with slope *firstSlope*
       ending at (*points*[0], *values*[0]).
     * the segment i, i in 1, 2, .., n-1, is defined on the interval [*points*[i-1], *points*[i]) and is a
       linear function from (*points*[-1], *values*[i-1]) to (*points*[i], *values*[i]).
     * the segment n is defined on the interval [*points*[n-1], infinity) and is a linear function
       from (*points*[n-1], *values*[n-1]) with slope *lastSlope*.

    Args:
        x :         x-value for which the function should be evaluated.
        firstSlope: slope of the first function segment (ending at (points[0], values[0])).
        points:     sorted array of x-values that separate function segments (breaking points).
        values:     y-values corresponding to the breaking points (the array must have the same length as points).
        lastSlope:  slope of the last segment beginning at (points[n-1], values[n-1])
                    where n is length of points and  values.
    Returns:
        Value of the function at point x.
    """
    return CpoFunctionCall(Oper_coordinate_piecewise_linear, Type_FloatExpr,
                           (_convert_arg(x,          "x",          Type_FloatExpr),
                            _convert_arg(firstSlope, "firstSlope", Type_Float),
                            _convert_arg(points,     "points",     Type_FloatArray),
                            _convert_arg(values,     "values",     Type_FloatArray),
                            _convert_arg(lastSlope,  "lastSlope",  Type_Float),
                            ))


#==============================================================================
#  Objective functions
#==============================================================================

def maximize(expr):
    """ This function asks CP Optimizer to seek to maximize the value of an expressions.

    The function *maximize* specifies to CP Optimizer a floating-point expression
    whose value is sought to be maximized.
    When this function is used and the problem is feasible, CP Optimizer will generate
    one or more feasible solutions to the problem, with subsequent solutions having
    a larger value of *expr* than preceding ones.
    The search terminates when either the optimality of the last solution is proved,
    a search limit is exhausted, or the search is aborted.

    Args:
        expr: The float expression to be maximized.
    Returns:
        An objective expression
    """
    return CpoFunctionCall(Oper_maximize, Type_Objective, (_convert_arg(expr, "exprs", Type_FloatExpr), ))


def maximize_static_lex(exprs):
    """ A function to specify an optimization problem.  It asks CP Optimizer to
    seek to lexicographically maximize the values of a number of expressions.

    The function *maximize_static_lex* specifies to CP Optimizer a number of
    floating-point expressions whose values are sought to be maximized in a
    lexicographic fashion.  When this function is used and
    the problem is feasible, CP Optimizer will generate one or more
    feasible solutions to the problem, with subsequent solutions having
    a lexicographically larger value of *exprs* than preceding ones.
    This means that a new solution replaces the preceding one as incumbent if
    the value of criterion *exprs[i]* is greater than in the preceding solution,
    so long as the values of criteria *exprs[0..i-1]* are not less than in the
    preceding solution.  In particular, this means that the newer solution is
    preferable even if there are arbitrary reductions in the values of criteria
    after position *i* in *exprs*, as compared with the preceding solution.
    The search terminates when either the optimality of the last solution
    is proved, a search limit is exhausted, or the search is aborted.

    Args:
        exprs: A non-empty array of floating-point expressions whose values are to be
               lexicographically maximized.
    Returns:
        An expression of type objective
    """
    # Check array is not empty
    assert exprs, "Array of expressions should not be empty"
    return CpoFunctionCall(Oper_maximize_static_lex, Type_Objective, (_convert_arg(exprs, "exprs", Type_FloatExprArray), ))



def minimize(expr):
    """ This function asks CP Optimizer to seek to minimize the value of an expressions.

    The function *minimize* specifies to CP Optimizer a floating-point expression
    whose value is sought to be minimized.
    When this function is used and the problem is feasible, CP Optimizer will generate
    one or more feasible solutions to the problem, with subsequent solutions having
    a smaller value of *expr* than preceding ones.
    The search terminates when either the optimality of the last solution is proved,
    a search limit is exhausted, or the search is aborted.

    Args:
        expr: The float expression to be minimized.
    Returns:
        An objective expression
    """
    return CpoFunctionCall(Oper_minimize, Type_Objective, (_convert_arg(expr, "exprs", Type_FloatExpr), ))


def minimize_static_lex(exprs):
    """ A function to specify an optimization problem.  It asks CP Optimizer to
    seek to lexicographically minimize the values of a number of expressions.

    The function *minimize_static_lex* specifies to CP Optimizer a number of
    floating-point expressions whose values are sought to be minimized in a
    lexicographic fashion.  When this function is used and
    the problem is feasible, CP Optimizer will generate
    one or more feasible solutions to the problem, with subsequent solutions having
    a lexicographically smaller value of *exprs* than preceding ones.
    This means that a new solution replaces the preceding one as incumbent if
    the value of criterion *exprs[i]* is less than in the preceding solution,
    so long as the values of criteria *exprs[0..i-1]* are not greater than in the
    preceding solution.  In particular, this means that the newer solution is
    preferable even if there are arbitrary increases in the values of criteria
    after position *i* in *exprs*, as compared with the preceding solution.
    The search terminates when either the optimality of the last solution
    is proved, a search limit is exhausted, or the search is aborted.

    Args:
        exprs: A non-empty array of floating-point expressions whose values are to be
               lexicographically minimized.
    Returns:
        An objective expression
    """
    # Check array is not empty
    assert exprs, "Array of expressions should not be empty"
    return CpoFunctionCall(Oper_minimize_static_lex, Type_Objective, (_convert_arg(exprs, "exprs", Type_FloatExprArray), ))


#==============================================================================
#  Interval variables expressions
#==============================================================================

def start_of(interval, absentValue=None):
    """ Returns the start of a specified interval variable.

    This function returns an integer expression that is equal to start of the interval
    variable *interval* if it is present. If it is absent, then the value of the
    expression is *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        absentValue (Optional): Value to return if the interval variable interval becomes absent.
                      Zero if not given.
    Returns:
        An integer expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    if absentValue is None:
       return CpoFunctionCall(Oper_start_of, Type_IntExpr, (interval,))
    return CpoFunctionCall(Oper_start_of, Type_IntExpr, (interval, _convert_arg(absentValue, "absentValue", Type_Int)))


def end_of(interval, absentValue=None):
    """ Returns the end of specified interval variable.

    This function returns an integer expression that is equal to end of the interval
    variable *interval* if it is present. If it is absent then the value of the
    expression is *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        absentValue (Optional): Value to return if the interval variable interval becomes absent.
                      Zero if not given.
    Returns:
        An integer expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    if absentValue is None:
       return CpoFunctionCall(Oper_end_of, Type_IntExpr, (interval,))
    return CpoFunctionCall(Oper_end_of, Type_IntExpr, (interval, _convert_arg(absentValue, "absentValue", Type_Int)))


def length_of(interval, absentValue=None):
    """ Returns the length of specified interval variable.

    This function returns an integer expression that is equal to the length (*end -
    start*) of the interval variable *interval* if it is present. If it is absent, then
    the value of the expression is *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        absentValue (Optional): Value to return if the interval variable interval becomes absent.
                     Zero if not given.
    Returns:
        An integer expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    if absentValue is None:
       return CpoFunctionCall(Oper_length_of, Type_IntExpr, (interval,))
    return CpoFunctionCall(Oper_length_of, Type_IntExpr, (interval, _convert_arg(absentValue, "absentValue", Type_Int)))


def size_of(interval, absentValue=None):
    """ Returns the size of a specified interval variable.

    This function returns an integer expression that is equal to size of the interval
    variable *interval* if it is present. If it is absent then the value of the
    expression is *absentValue* (zero by default).

    Args:
        interval: Interval variable.
        absentValue (Optional): Value to return if the interval variable interval becomes absent.
                     Zero if not given.
    Returns:
        An integer expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    if absentValue is None:
       return CpoFunctionCall(Oper_size_of, Type_IntExpr, (interval,))

    absentValue = _convert_arg(absentValue, "absentValue", Type_Int)
    return CpoFunctionCall(Oper_size_of, Type_IntExpr, (interval, absentValue))


def presence_of(interval):
    """ Returns the presence status of an interval variable.

    This function returns a boolean expression that represents the presence status of an interval variable.
    If *interval* is present then the value of the expression is 1; if *interval* is absent then the value is 0.

    Use *presence_of* to express logical relationships between interval variables.
    Note that the most effective are binary relations such as *presence_of(x)=>presence_of(y)* because CP Optimizer
    is able to take them into account during propagation of other constraints such as *end_before_start* or *no_overlap*.

    The function *presence_of* can be also used to compute cost associated with
    execution/non-execution of an interval.

    Args:
        interval: Interval variable.
    Returns:
        An expression of type boolean expression
    """
    return CpoFunctionCall(Oper_presence_of, Type_BoolExpr, (_convert_arg(interval, "interval", Type_IntervalVar),))


def start_at_start(a, b, delay=None):
    """ Constrains the delay between the starts of two interval variables.

    The function *start_at_start* constrains interval variables *a* and *b* in the following way.
    If both intervals *a* and *b* are present, then interval *b* must start exactly at *start_of(a) + delay*.
    If *a* or *b* is absent then the constraint is automatically satisfied.

    The default value for *delay* is zero. Note that *delay* can be negative.

    Args:
        a: First interval variables.
        b: Second interval variables.
        delay: Exact delay between starts of *a* and *b*. If not specified then zero is used.
    Returns:
        Constraint expression
    """
    a = _convert_arg(a, "a", Type_IntervalVar)
    b = _convert_arg(b, "b", Type_IntervalVar)
    if delay is None:
       return CpoFunctionCall(Oper_start_at_start, Type_Constraint, (a, b))

    return CpoFunctionCall(Oper_start_at_start, Type_Constraint, (a, b, _convert_arg(delay, "delay", Type_IntExpr)))


def start_at_end(a, b, delay=None):
    """ Constrains the delay between the start of one interval variable and end of another one.

    The function *start_at_end* constrains interval variables *a* and *b* in the following way.
    If both intervals *a* and *b* are present then interval *b* must end exactly at *start_of(a) + delay*.
    If *a* or *b* is absent then the constraint is automatically satisfied.

    The default value for *delay* is zero. Note that *delay* can be negative.

    Args:
        a: First interval variables.
        b: Second interval variables.
        delay: Exact delay between start of *a* and end of *b*. If not specified then zero is used.
    Returns:
        Constraint expression
    """
    a = _convert_arg(a, "a", Type_IntervalVar)
    b = _convert_arg(b, "b", Type_IntervalVar)
    if delay is None:
       return CpoFunctionCall(Oper_start_at_end, Type_Constraint, (a, b))

    return CpoFunctionCall(Oper_start_at_end, Type_Constraint, (a, b, _convert_arg(delay, "delay", Type_IntExpr)))


def start_before_start(a, b, delay=None):
    """ Constrains the minimum delay between starts of two interval variables.

    The function *start_before_start* constrains interval variables *a* and *b* in the following way.
    If both interval variables *a* and *b* are present, then *b* cannot start before *start_of(a) + delay*.
    If *a* or *b* is absent then the constraint is automatically satisfied.

    The default value for *delay* is zero.
    It is possible to specify even negative *delay*, in this case *b* can actually start before the start
    of *a* but still not sooner than *start_of(a) + delay*.

    Args:
        a: Interval variable which starts before.
        b: Interval variable which starts after.
        delay: The minimal delay between start of a and start of b. If not specified then zero is used.
    Returns:
        Constraint expression
    """
    a = _convert_arg(a, "a", Type_IntervalVar)
    b = _convert_arg(b, "b", Type_IntervalVar)
    if delay is None:
       return CpoFunctionCall(Oper_start_before_start, Type_Constraint, (a, b))

    return CpoFunctionCall(Oper_start_before_start, Type_Constraint, (a, b, _convert_arg(delay, "delay", Type_IntExpr)))


def start_before_end(a, b, delay=None):
    """ Constrains minimum delay between the start of one interval variable and end of another one.

    The function *start_before_end* constrains interval variables *a* and *b* in the following way.
    If both interval variables *a* and *b* are present, then *b* cannot end before *start_of(a) + delay*.
    If *a* or *b* is absent then the constraint is automatically satisfied.

    The default value for *delay* is zero.
    It is possible to specify a negative *delay*; in this case *b* can actually end before the start
    of *a* but still not sooner than *start_of(a) + delay*.

    Args:
        a: Interval variable which starts before.
        b: Interval variable which ends after.
        delay: The minimal delay between start of *a* and end of *b*. If not specified then zero is used.
    Returns:
        Constraint expression
    """
    a = _convert_arg(a, "a", Type_IntervalVar)
    b = _convert_arg(b, "b", Type_IntervalVar)
    if delay is None:
       return CpoFunctionCall(Oper_start_before_end, Type_Constraint, (a, b))

    return CpoFunctionCall(Oper_start_before_end, Type_Constraint, (a, b, _convert_arg(delay, "delay", Type_IntExpr)))


def end_at_start(a, b, delay=None):
    """ Constrains the delay between the end of one interval variable and start of another one.

    The function *end_at_start* constrains interval variables *a* and *b* in the following way.
    If both intervals *a* and *b* are present, then interval *b* must start exactly at *end_of(a) + delay*.
    If *a* or *b* is absent then the constraint is automatically satisfied.

    The default value for *delay* is zero. Note that *delay* can be negative.

    Args:
        a: Interval variables.
        b: Interval variables.
        delay: Exact delay between end of *a* and start of *b*. If not specified then zero is used.
    Returns:
        Constraint expression
    """
    a = _convert_arg(a, "a", Type_IntervalVar)
    b = _convert_arg(b, "b", Type_IntervalVar)
    if delay is None:
       return CpoFunctionCall(Oper_end_at_start, Type_Constraint, (a, b))

    return CpoFunctionCall(Oper_end_at_start, Type_Constraint, (a, b, _convert_arg(delay, "delay", Type_IntExpr)))


def end_at_end(a, b, delay=None):
    """ Constrains the delay between the ends of two interval variables.

    The function *end_at_end* constrains interval variables *a* and *b* in the following way.
    If both intervals *a* and *b* are present then interval *b* must end exactly at *end_of(a) + delay*.
    If *a* or *b* is absent then the constraint is automatically satisfied.

    The default value for *delay* is zero. Note that *delay* can be negative.

    Args:
        a: Interval variable.
        b: Interval variable.
        delay: Exact delay between ends of *a* and *b*. If not specified then zero is used.
    Returns:
        Constraint expression
    """
    a = _convert_arg(a, "a", Type_IntervalVar)
    b = _convert_arg(b, "b", Type_IntervalVar)
    if delay is None:
       return CpoFunctionCall(Oper_end_at_end, Type_Constraint, (a, b))

    return CpoFunctionCall(Oper_end_at_end, Type_Constraint, (a, b, _convert_arg(delay, "delay", Type_IntExpr)))


def end_before_start(a, b, delay=None):
    """ Constrains minimum delay between the end of one interval variable and start of another one.

    The function *end_before_start* constrains interval variables *a* and *b* in the following way.
    If both interval variables *a* and *b* are present, then *b* cannot start before *end_of(a) + delay*.
    If *a* or *b* is absent then the constraint is automatically satisfied.

    The default value for *delay* is zero.
    It is possible to specify even negative *delay*, in this case *b* can actually start before the end
    of *a* but still not sooner than *end_of(a) + delay*.

    Args:
        a: Interval variable which ends before.
        b: Interval variable which starts after.
        delay: The minimal delay between end of a and start of b. If not specified then zero is used.
    Returns:
        Constraint expression
    """
    a = _convert_arg(a, "a", Type_IntervalVar)
    b = _convert_arg(b, "b", Type_IntervalVar)
    if delay is None:
       return CpoFunctionCall(Oper_end_before_start, Type_Constraint, (a, b))

    return CpoFunctionCall(Oper_end_before_start, Type_Constraint, (a, b, _convert_arg(delay, "delay", Type_IntExpr)))


def end_before_end(a, b, delay=None):
    """ Constrains the minimum delay between the ends of two interval variables.

    The function *end_before_end* constrains interval variables *a* and *b* in the following way.
    If both interval variables *a* and *b* are present, then *b* cannot end before *end_of(a) + delay*.
    If *a* or *b* is absent then the constraint is automatically satisfied.

    The default value for *delay* is zero.
    It is possible to specify a negative *delay*; in this case *b* can actually end before the end
    of *a* but still not sooner than *end_of(a) + delay*.

    Args:
        a: Interval variable which ends before.
        b: Interval variable which ends after.
        delay: The minimal delay between end of a and end of b. If not specified then zero is used.
    Returns:
        Constraint expression
    """
    a = _convert_arg(a, "a", Type_IntervalVar)
    b = _convert_arg(b, "b", Type_IntervalVar)
    if delay is None:
       return CpoFunctionCall(Oper_end_before_end, Type_Constraint, (a, b))

    return CpoFunctionCall(Oper_end_before_end, Type_Constraint, (a, b, _convert_arg(delay, "delay", Type_IntExpr)))


def forbid_start(interval, function):
    """ Forbids an interval variable to start during specified regions.

    This constraint restricts possible start times of interval variable using a step function.
    The interval variable can start only at points where the function value is not zero.
    When the interval variable is absent then this constraint is automatically satisfied,
    since such interval variable does not have any start at all.

    In declaration of an interval variable it is only possible to specify a range of possible start times.
    This function allows more precise specification of when the interval variable can start.

    Args:
        interval: Interval variable being restricted.
        function: If the function has value 0 at point *t* then the interval variable interval cannot start at *t*.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_forbid_start, Type_Constraint, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                                _convert_arg(function, "function", Type_StepFunction)))

def forbid_end(interval, function):
    """ Forbids an interval variable to end during specified regions.

    In the declaration of an interval variable it is only possible to specify a range of possible end times.
    This function allows the user to specify more precisely when the interval variable can end.
    In particular, the interval variable can end only at point *t* such that the function has non-zero value at
    *t-1*.
    When the interval variable is absent then this constraint is automatically satisfied,
    since such interval variable does not't have any start at all.

    Note the difference between *t* (end time of the interval variable) and *t-1*
    (the point when the function value is checked). It simplifies the sharing of the same function
    in constraints *forbid_start* and *forbid_end*.
    It also allows one to use the same function as *intensity* parameter of interval variable.

    Args:
        interval: Interval variable being restricted.
        function: If the function has value 0 at point *t*-1 then the interval variable interval cannot end at *t*.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_forbid_end, Type_Constraint, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                              _convert_arg(function, "function", Type_StepFunction)))


def forbid_extent(interval, function):
    """ Forbids an interval variable to overlap with specified regions.

    This function allows specification of forbidden regions that the interval variable *interval* cannot overlap with.
    In particular, if interval variable *interval* is present and if *function* has value 0 during
    interval *[a,b)* (i.e. *[a,b)* is a forbidden region) then either *end <= a* (*interval* ends before the
    forbidden region) or *b <= start* (*interval* starts after the forbidden region).

    If the interval variable *interval* is absent then the constraint is automatically satisfied
    (the interval does not exist therefore it cannot overlap with any region).

    Args:
        interval: Interval variable being restricted.
        function: Forbidden regions corresponds to step of the function that have value 0.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_forbid_extent, Type_Constraint, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                                 _convert_arg(function, "function", Type_StepFunction)))


def overlap_length(interval, interval2, absentValue=None):
    """ Returns the length of the overlap of two interval variables.

    This function returns an integer expression that represents the length of the overlap
    of interval variable *interval* and the interval variable *interval2* whenever the interval
    variables *interval* and *interval2* are present.
    When one of the interval variables *interval* or *interval2* is absent, the function returns
    the constant integer value *absentValue* (zero by default).

    Optionally, *interval2* can be a constant interval [*start*, *end*) expressed as a tuple of two
    integers.
    When the interval variable *interval* is absent, the function returns the constant integer value
    *absentValue* (zero by default).

    Args:
        interval: Interval variable
        interval2: Another interval variable, or fixed interval expressed as a tuple of 2 integers.
        absentValue (Optional): Value to return if some interval variable is absent.
    Returns:
        An integer expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)

    # Interval2 is fixed
    if is_array(interval2):
        assert (len(interval2) == 2) and is_int(interval2[0]) and is_int(interval2[1]), \
            "To express a fixed interval, 'interval2' should be a tuple of two integers"
        t1 = _convert_arg(interval2[0], "interval2[0]", Type_TimeInt)
        t2 = _convert_arg(interval2[1], "interval2[1]", Type_TimeInt)
        if absentValue is None:
            return CpoFunctionCall(Oper_overlap_length, Type_IntExpr, (interval, t1, t2))
        return CpoFunctionCall(Oper_overlap_length, Type_IntExpr, (interval, t1, t2,
                                                                   _convert_arg(absentValue, "absentValue", Type_Int)))

    # Interval2 is an interval variable
    interval2 = _convert_arg(interval2, "interval2", Type_IntervalVar)
    if absentValue is None:
        return CpoFunctionCall(Oper_overlap_length, Type_IntExpr, (interval, interval2))
    return CpoFunctionCall(Oper_overlap_length, Type_IntExpr, (interval, interval2,
                                                               _convert_arg(absentValue, "absentValue", Type_Int)))


def start_eval(interval, function, absentValue=None):
    """ Evaluates a segmented function at the start of an interval variable.

    Evaluates *function* at the start of interval variable *interval*.
    If *interval* is absent, it does not have any defined start and *absentValue* is returned.

    Args:
        interval: Interval variable.
        function: Function to evaluate.
        absentValue (Optional): Value to return if interval variable interval is absent.
                     If not given, absent value is zero.
    Returns:
        A float expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    function = _convert_arg(function, "function", Type_SegmentedFunction)
    if absentValue is None:
        return CpoFunctionCall(Oper_start_eval, Type_FloatExpr, (interval, function))
    return CpoFunctionCall(Oper_start_eval, Type_FloatExpr, (interval, function, _convert_arg(absentValue, "absentValue", Type_Float)))


def end_eval(interval, function, absentValue=None):
    """ Evaluates a segmented function at the end of an interval variable.

    Evaluates *function* at the start of interval variable *interval*.
    If *interval* is absent, it does not have any defined end and *absentValue* is returned.

    Args:
        interval: Interval variable.
        function: Function to evaluate.
        absentValue (Optional): Value to return if interval variable interval is absent.
                     If not given, absent value is zero.
    Returns:
        A float expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    function = _convert_arg(function, "function", Type_SegmentedFunction)
    if absentValue is None:
        return CpoFunctionCall(Oper_end_eval, Type_FloatExpr, (interval, function))
    return CpoFunctionCall(Oper_end_eval, Type_FloatExpr, (interval, function, _convert_arg(absentValue, "absentValue", Type_Float)))


def size_eval(interval, function, absentValue=None):
    """ Evaluates a segmented function on the size of an interval variable.

    Evaluate *function* for the x value equal to the size of interval variable *interval*.
    If *interval* is absent then it does not have any defined size and *absentValue* is returned.

    Args:
        interval: Interval variable.
        function: Function to evaluate.
        absentValue (Optional): Value to return if interval variable interval is absent.
                     If not given, absent value is zero.
    Returns:
        A float expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    function = _convert_arg(function, "function", Type_SegmentedFunction)
    if absentValue is None:
        return CpoFunctionCall(Oper_size_eval, Type_FloatExpr, (interval, function))
    return CpoFunctionCall(Oper_size_eval, Type_FloatExpr, (interval, function, _convert_arg(absentValue, "absentValue", Type_Float)))


def length_eval(interval, function, absentValue=None):
    """ Evaluates segmented function on the length of an interval variable.

    Evaluate *function* for the x value equal to the length of interval variable *interval*.
    If *interval* is absent then it does not have any defined length and *absentValue* is returned.

    Args:
        interval: Interval variable.
        function: Function to evaluate.
        absentValue (Optional): Value to return if interval variable interval is absent.
                     If not given, absent value is zero.
    Returns:
        A float expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    function = _convert_arg(function, "function", Type_SegmentedFunction)
    if absentValue is None:
        return CpoFunctionCall(Oper_length_eval, Type_FloatExpr, (interval, function))
    return CpoFunctionCall(Oper_length_eval, Type_FloatExpr, (interval, function, _convert_arg(absentValue, "absentValue", Type_Float)))


def span(interval, array):
    """ Creates a span constraint between interval variables.

    This function creates a span constraint between an interval variable *interval*
    and a set of interval variables in *array*. This constraint states that
    *interval* when it is present spans over all present intervals from the
    *array*. That is: *interval* starts together with the first present
    interval from *array* and ends together with the last one. Interval *interval*
    is absent if and only if all intervals in *array* are absent.

    Args:
        interval: Spanning interval variable.
        array: Array of spanned interval variables.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_span, Type_Constraint, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                        _convert_arg(array, "array", Type_IntervalVarArray)))


def alternative(interval, array, cardinality=None):
    """ Creates an alternative constraint between interval variables.

    This function creates an alternative constraint between interval variable
    *interval* and the set of interval variables in *array*.
    If no *cardinality* expression is specified, if *interval* is present, then one and only
    one of the intervals in *array* will be selected by the alternative constraint
    to be present, and the start and end values of *interval* will be the same as the
    ones of the selected interval.
    If a *cardinality* expression is specified, *cardinality* intervals in *array* will be selected by the
    alternative constraint to be present and the selected intervals will have the
    same start and end value as interval variable *interval*.
    Interval variable *interval* is absent if and only if all interval variables in *array* are absent.

    Args:
        interval:    Interval variable.
        array:       Array of interval variables.
        cardinality (Optional): Cardinality of the alternative constraint.
                     By default, when this optional argument is not specified, a unit cardinality is assumed (cardinality=1).
    Returns:
        Constraint expression
    """
    if cardinality is None:
       return CpoFunctionCall(Oper_alternative, Type_Constraint, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                                  _convert_arg(array, "array", Type_IntervalVarArray)))
    return CpoFunctionCall(Oper_alternative, Type_Constraint, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                               _convert_arg(array, "array", Type_IntervalVarArray),
                                                               _convert_arg(cardinality, "cardinality", Type_IntExpr)))


def synchronize(interval, array):
    """ Creates a synchronization constraint between interval variables.

    This function creates a synchronization constraint between an interval variable *interval*
    and a set of interval variables in *array*.
    This constraint makes all present intervals in *array* start and end together with *interval*,
    if it is present.

    Args:
        interval: Interval variable.
        array:    Array of interval variables synchronized with interval.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_synchronize, Type_Constraint, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                               _convert_arg(array, "array", Type_IntervalVarArray)))


def isomorphism(array1, array2, map=None, absentValue=None):
    """ Returns an isomorphism constraint between two sets of interval variables.

    This function creates an isomorphism constraint between the set of interval variables
    in the array *array1* and the set of interval variables in the array *array2*.
    If an integer expression array *map* is used, it is used to reflect the mapping of the intervals
    of *array1* on the intervals of *array2*, that is, interval variable *array2[i]*, if present,
    is mapped on interval variable *array1[map[i]]**.
    If *array2[i]* is absent, index *map[i]* takes value *absentValue*.

    Args:
        array1: The first array of interval variables.
        array2: The second array of interval variables.
        map:    (Optional) Array of integer expressions mapping intervals of array2 on array1.
        absentValue (Optional): Integer value of map[i] when array2[i] is absent.

    Possible argument and return type combinations are:

     * (array of interval variables, array of interval variables, array of integer expressions [=None], integer constant [=None]) => constraint

    Returns:
        Constraint expression
    """
    array1 = _convert_arg(array1, "array1", Type_IntervalVarArray)
    array2 = _convert_arg(array2, "array2", Type_IntervalVarArray)
    if absentValue is None:
        if map is None:
            return CpoFunctionCall(Oper_isomorphism, Type_Constraint, (array1, array2))
        return CpoFunctionCall(Oper_isomorphism, Type_Constraint, (array1, array2, _convert_arg(map, "map", Type_IntExprArray)))
    return CpoFunctionCall(Oper_isomorphism, Type_Constraint, (array1, array2,
                                                               _convert_arg(map, "map", Type_IntExprArray),
                                                               _convert_arg(absentValue, "absentValue", Type_Int)))


#==============================================================================
#  Sequence variables expressions
#==============================================================================


def first(sequence, interval):
    """ Constrains an interval variable to be the first in a sequence.

    This function returns a constraint that states that whenever interval variable *interval* is present,
    it must be ordered first in the sequence variable *sequence*.

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_first, Type_Constraint, (_convert_arg(sequence, "sequence", Type_SequenceVar),
                                                         _convert_arg(interval, "interval", Type_IntervalVar)))


def last(sequence, interval):
    """ Constrains an interval variable to be the last in a sequence.

    This function returns a constraint that states that whenever interval variable *interval* is present,
    it must be ordered last in the sequence variable *sequence*.

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_last, Type_Constraint, (_convert_arg(sequence, "sequence", Type_SequenceVar),
                                                        _convert_arg(interval, "interval", Type_IntervalVar)))


def before(sequence, interval1, interval2):
    """ Constrains an interval variable to be before another interval variable in a sequence.

    This function returns a constraint that states that whenever both interval variables
    *interval1* and *interval2* are present,
    *interval1* must be ordered before *interval2* in the sequence variable *sequence*.

    Args:
        sequence:  Sequence variable.
        interval1: First interval variables.
        interval2: Second interval variables.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_before, Type_Constraint, (_convert_arg(sequence, "sequence", Type_SequenceVar),
                                                          _convert_arg(interval1, "interval1", Type_IntervalVar),
                                                          _convert_arg(interval2, "interval2", Type_IntervalVar)))


def previous(sequence, interval1, interval2):
    """ Constrains an interval variable to be previous to another interval variable in a sequence.

    This function returns a constraint that states that whenever both interval variables *interval1* and *interval2*
    are present, *interval1* must be the interval variable that is previous to *interval2* in the
    sequence variable *sequence*.

    Args:
        sequence: Sequence variable.
        interval1: Interval variable.
        interval2: Interval variable.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_previous, Type_Constraint, (_convert_arg(sequence, "sequence", Type_SequenceVar),
                                                            _convert_arg(interval1, "interval1", Type_IntervalVar),
                                                            _convert_arg(interval2, "interval2", Type_IntervalVar)))


def _sequence_operation(oper, sequence, interval, value, islast, absentValue):
    """ Returns an integer expression that represents the start of the interval variable that is next.

    This function returns an integer expression that represents the start of the interval variable
    that is next to *interval* in sequence variable *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        oper:        Operation descriptor
        sequence:    Sequence variable.
        interval:    Interval variable.
        value:       Value to return if interval variable interval is the last or first one in sequence.
        islast:      Indicates if value concerns interval variable last or first in the sequence
        absentValue: Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    sequence = _convert_arg(sequence, "sequence", Type_SequenceVar)
    interval = _convert_arg(interval, "interval1", Type_IntervalVar)
    vname = "lastValue" if islast else "firstValue"
    if absentValue is None:
        if value is None:
            return CpoFunctionCall(oper, Type_IntExpr, (sequence, interval))
        return CpoFunctionCall(oper, Type_IntExpr, (sequence, interval, _convert_arg(value, vname, Type_Int)))
    return CpoFunctionCall(oper, Type_IntExpr, (sequence, interval,
                                                _convert_arg(value, vname, Type_Int),
                                                _convert_arg(absentValue, "absentValue", Type_Int)))


def start_of_next(sequence, interval, lastValue=None, absentValue=None):
    """ Returns an integer expression that represents the start of the interval variable that is next.

    This function returns an integer expression that represents the start of the interval variable
    that is next to *interval* in sequence variable *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: (Optional) Value to return if interval variable interval is the last one in sequence.
        absentValue (Optional): Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_start_of_next, sequence, interval, lastValue, true, absentValue)


def start_of_prev(sequence, interval, firstValue=None, absentValue=None):
    """ Returns an integer expression that represents the start of the interval variable that is previous.

    This function returns an integer expression that represents the start of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: (Optional) Value to return if interval variable interval is the first one in sequence.
        absentValue (Optional): Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_start_of_prev, sequence, interval, firstValue, false, absentValue)


def end_of_next(sequence, interval, lastValue=None, absentValue=None):
    """ Returns an integer expression that represents the end of the interval variable that is next.

    This function returns an integer expression that represents the end of the interval variable
    that is next to *interval* in *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: Value to return if interval variable interval is the last one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_end_of_next, sequence, interval, lastValue, true, absentValue)


def end_of_prev(sequence, interval, firstValue=None, absentValue=None):
    """ Returns an integer expression that represents the end of the interval variable that is previous.

    This function returns an integer expression that represents the end of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: Value to return if interval variable interval is the first one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_end_of_prev, sequence, interval, firstValue, false, absentValue)


def length_of_next(sequence, interval, lastValue=None, absentValue=None):
    """ Returns an integer expression that represents the length of the interval variable that is next.

    This function returns an integer expression that represents the length of the interval variable
    that is next to *interval* in sequence variable *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: Value to return if interval variable interval is the last one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_length_of_next, sequence, interval, lastValue, true, absentValue)


def length_of_prev(sequence, interval, firstValue=None, absentValue=None):
    """ Returns an integer expression that represents the length of the interval variable that is previous.

    This function returns an integer expression that represents the length of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: Value to return if interval variable interval is the first one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_length_of_prev, sequence, interval, firstValue, false, absentValue)


def size_of_next(sequence, interval, lastValue=None, absentValue=None):
    """ Returns an integer expression that represents the size of the interval variable that is next.

    This function returns an integer expression that represents the size of the interval variable
    that is next to *interval* in sequence variable *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: Value to return if interval variable interval is the last one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_size_of_next, sequence, interval, lastValue, true, absentValue)


def size_of_prev(sequence, interval, firstValue=None, absentValue=None):
    """ Returns an integer expression that represents the size of the interval variable that is previous.

    This function returns an integer expression that represents the size of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: Value to return if interval variable interval is the first one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_size_of_prev, sequence, interval, firstValue, false, absentValue)


def type_of_next(sequence, interval, lastValue=None, absentValue=None):
    """ Returns an integer expression that represents the type of the interval variable that is next.

    This function returns an integer expression that represents the type of the interval variable
    that is next to *interval* in sequence variable *sequence*. When *interval* is present and is
    the last interval of *sequence*, it returns the constant integer value *lastValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        lastValue: Value to return if interval variable interval is the last one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_type_of_next, sequence, interval, lastValue, true, absentValue)


def type_of_prev(sequence, interval, firstValue=None, absentValue=None):
    """ Returns an integer expression that represents the type of the interval variable that is previous.

    This function returns an integer expression that represents the type of the interval variable
    that is previous to *interval* in sequence variable *sequence*. When *interval* is present and is
    the first interval of *sequence*, it returns the constant integer value *firstValue* (zero by default).
    When *interval* is absent, it returns the constant integer value *absentValue* (zero by default).

    Args:
        sequence: Sequence variable.
        interval: Interval variable.
        firstValue: Value to return if interval variable interval is the first one in sequence.
        absentValue: Value to return if interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    return _sequence_operation(Oper_type_of_prev, sequence, interval, firstValue, false, absentValue)


def no_overlap(sequence, distance_matrix=None, is_direct=None):
    """ Constrains a set of interval variables not to overlap each others.

    This function returns a constraint over a set of interval variables {*a1*, ..., *an*} that states that
    all the present intervals in the set are pairwise non-overlapping.
    It means that whenever both interval variables *ai* and *aj*, i!=j are present, *ai* is constrained to end
    before the start of *aj* or *aj* is constrained to end before the start of *ai*.

    If the no-overlap constraint has been built on an interval sequence variable *sequence*, it means that
    the no-overlap constraint works on the set of interval variables {*a1*, ..., *an*} of the sequence and that
    the order of interval variables of the sequence will describe the order of the non-overlapping intervals.
    That is, if *ai* and *aj*, i!=j are both present and if *ai* appears before *aj* in the sequence value,
    then *ai* is constrained to end before the start of *aj*.
    If a transition matrix *distance_matrix* is specified and if *tpi* and *tpj* respectively denote the types of
    interval variables *ai* and *aj* in the *sequence*, it means that a minimal distance
    *distance_matrix[tpi,tpj]* is to be maintained between the end of *ai* and the start of *aj*.
    If boolean flag *is_direct* is True, the transition distance holds between an interval and its immediate successor
    in the sequence.
    If *is_direct* is False (default), the transition distance holds between an interval and all its successors
     in the sequence.

    If the first argument is an array of interval variables, the two others are ignored.

    Args:
        sequence: A sequence variable, or an array of interval variables.
        distance_matrix (Optional): An optional transition matrix defining the transition distance between consecutive
                         interval variables.
                         Transition matrix is given as an iterable of iterables of positive integers,
                         or as the result of a call to the method :meth:`~docplex.cp.expression.transition_matrix`.
        is_direct (Optional): A boolean flag stating whether the distance specified in the transition matrix
                 *distance_matrix* holds between direct successors (is_direct=True)
                 or also between indirect successors (is_direct=False, default).
    Returns:
        Constraint expression
    """
    # Check if sequence is an array of interval variables
    sequence = build_cpo_expr(sequence)
    if sequence.is_kind_of(Type_IntervalVarArray):
        assert (distance_matrix is None) and (is_direct is None), "As first argument is an array of interval variables, other arguments should be absent"
        return CpoFunctionCall(Oper_no_overlap, Type_Constraint, (sequence,))

    # Sequence is a sequence variable
    assert sequence.is_kind_of(Type_SequenceVar), "First argument should be a sequence variable or an array of interval variables"
    if distance_matrix is None:
        assert is_direct is None, "is_direct should not be given if no distance matrix is given"
        return CpoFunctionCall(Oper_no_overlap, Type_Constraint, (sequence,))
    
    distance_matrix = build_cpo_transition_matrix(distance_matrix)
    if is_direct is None:
        return CpoFunctionCall(Oper_no_overlap, Type_Constraint, (sequence, distance_matrix))
    return CpoFunctionCall(Oper_no_overlap, Type_Constraint, (sequence, distance_matrix, _convert_arg_bool_int(is_direct, "is_direct")))
    


def same_sequence(seq1, seq2, array1=None, array2=None):
    """ This function creates a same-sequence constraint between two sequence variables.

    The constraint states that the two sequences *seq1* and *seq2* are identical modulo a mapping between
    intervals *array1[i]* and *array2[i]*.

    Sequence variables *seq1* and *seq2* should be of the same size *n*.
    If no array of interval variables is specified, the mapping between interval variables of the
    two sequences is given by the order of the interval variables in the arrays *array1* and *array2* used
    in the definition of the sequences.

    If interval variable arrays *array1* and *array2* are used, these arrays define the mapping
    between interval variables of the two sequences.

    Args:
        seq1: First constrained sequence variables.
        seq2: Second constrained sequence variables.
        array1 (Optional): First array of interval variables defining the mapping between the two sequence variables.
        array2 (Optional): Second array of interval variables defining the mapping between the two sequence variables.
    Returns:
        Constraint expression
    """
    seq1 = _convert_arg(seq1, "seq1", Type_SequenceVar)
    seq2 = _convert_arg(seq2, "seq2", Type_SequenceVar)
    assert len(seq1) == len(seq2), "'seq1' and 'seq2' should have the same length"
    if array1 is None:
        assert array2 is None, "Mapping arrays 'array1' and 'array2' should be both given or ignored"
        return CpoFunctionCall(Oper_same_sequence, Type_Constraint, (seq1, seq2))

    return CpoFunctionCall(Oper_same_sequence, Type_Constraint, (seq1, seq2,
                                                                 _convert_arg(array1, "array1", Type_IntervalVarArray),
                                                                 _convert_arg(array2, "array2", Type_IntervalVarArray)))


def same_common_subsequence(seq1, seq2, array1=None, array2=None):
    """ This function creates a same-common-subsequence constraint between two sequence variables.

    If no interval variable array is specified as argument, the sequence variables *seq1* and *seq2*
    should be of the same size and the mapping between interval variables of the two sequences is
    given by the order of the interval variables in the arrays *array1* and *array2* used
    in the definition of the sequences.

    If interval variable arrays *array1* and *array2* are used, these arrays define the mapping
    between interval variables of the two sequences.

    The constraint states that the sub-sequences defined by *seq1* and *seq2* by only considering
    the pairs of present intervals (*array1[i]*, *array2[i]*) are identical modulo the mapping between
    intervals *array1[i]* and *array2[i]*.

    Args:
        seq1:   First constrained sequence variables.
        seq2:   Second constrained sequence variables.
        array1 (Optional): First array of interval variables defining the mapping between the two sequence variables.
        array2 (Optional): Second array of interval variables defining the mapping between the two sequence variables.
    Returns:
        Constraint expression
    """
    seq1 = _convert_arg(seq1, "seq1", Type_SequenceVar)
    seq2 = _convert_arg(seq2, "seq2", Type_SequenceVar)
    assert len(seq1) == len(seq2), "'seq1' and 'seq2' should have the same length"
    if array1 is None:
        assert array2 is None, "Mapping arrays 'array1' and 'array2' should be both given or ignored"
        return CpoFunctionCall(Oper_same_common_subsequence, Type_Constraint, (seq1, seq2))

    return CpoFunctionCall(Oper_same_common_subsequence, Type_Constraint, (seq1, seq2,
                                                                           _convert_arg(array1, "array1", Type_IntervalVarArray),
                                                                           _convert_arg(array2, "array2", Type_IntervalVarArray)))


#==============================================================================
#  Cumulative expressions
#==============================================================================

def pulse(interval, height, _x=None):
    """ Returns an elementary cumul function of constant value between the start and the end of an interval.

    This function returns an elementary cumul function expression that is equal to a value *height*
    everywhere between the start and the end of an interval variable *interval* or a fixed interval
    [*start*, *end*).

    The function is equal to 0 outside of the interval.

    When interval variable `interval` is absent, the function is the constant zero function.

    When a range [`heightMin`, `heightMax`) is specified it means that the height value of the pulse
    is part of the decisions of the problem and will be fixed by the engine within this specified range.

    Args:
        interval: Interval variable contributing to the cumul function,
                  or fixed interval expressed as a tuple of 2 integers.
        height:   Non-negative integer representing the height of the contribution,
                  or tuple of 2 non-negative integers representing the range of possible values
                  for the height of the contribution.
                  This last case is available only if interval is an interval variable.
    Returns:
        A cumul atom expression
    """
    # Case of 3 arguments (backward compatibility with previous implementation)
    if _x is not None:
        msg = "Deprecated calling form, consult documentation for details"
        if is_int(interval):
            return CpoFunctionCall(Oper_pulse, Type_CumulAtom, (_convert_arg(interval, "interval", Type_TimeInt, msg),
                                                                _convert_arg(height, "height", Type_TimeInt, msg),
                                                                _convert_arg(_x, "_x", Type_PositiveInt, msg)))
        return CpoFunctionCall(Oper_pulse, Type_CumulAtom, (_convert_arg(interval, "interval", Type_IntervalVar, msg),
                                                            _convert_arg(height, "height", Type_PositiveInt, msg),
                                                            _convert_arg(_x, "_x", Type_PositiveInt, msg)))

    # Case of fixed interval
    if _is_int_couple(interval):
        return CpoFunctionCall(Oper_pulse, Type_CumulAtom, (_convert_arg(interval[0], "interval[0]", Type_TimeInt),
                                                            _convert_arg(interval[1], "interval[1]", Type_TimeInt),
                                                            _convert_arg(height, "height", Type_PositiveInt)))

    # Case of interval variable
    interval = _convert_arg(interval, "interval", Type_IntervalVar,
                            "Argument 'interval' should be an interval variable or a fixed interval expressed as a tuple of integers")
    if _is_int_couple(height):
        return CpoFunctionCall(Oper_pulse, Type_CumulAtom, (interval,
                                                            _convert_arg(height[0], "height[0]", Type_PositiveInt),
                                                            _convert_arg(height[1], "height[1]", Type_PositiveInt)))
    height = _convert_arg(height, "height", Type_PositiveInt, "Argument 'height' should be an integer or a range expressed as a tuple of integers")
    return CpoFunctionCall(Oper_pulse, Type_CumulAtom, (interval, height))


def step_at(t, h):
    """ Returns an elementary cumul function of constant value after a given point.

    This function returns an elementary cumul function expression that is equal to 0 before
    point *t* and equal to *h* after point *t*.

    Args:
        t: Integer.
        h: Non-negative integer representing the height of the contribution after point t.

    Possible argument and return type combinations are:

     * (integer time, positive integer) => cumul atom

    Returns:
        A cumul atom expression
    """
    return CpoFunctionCall(Oper_step_at, Type_CumulAtom, (_convert_arg(t, "t", Type_TimeInt),
                                                          _convert_arg(h, "h", Type_PositiveInt)))


def step_at_start(interval, height):
    """ Returns an elementary cumul function of constant value after the start of an interval.

    This function returns an elementary cumul function expression that,
    whenever interval variable *interval* is present, is equal to 0 before the start of *interval*
    and equal to *height* after the start of *interval*.

    If *height* is a range specified as a tuple of integers, it means that the height value of the function
    is part of the decisions of the problem and will be fixed by the engine within this specified range.

    When interval variable *interval* is absent, the function is the constant zero function.

    Args:
        interval: Interval variable contributing to the cumul function.
        height:   Non-negative integer representing the height of the contribution,
                  or tuple of 2 non-negative integers representing the range of possible values
                  for the height of the contribution.
    Returns:
        A cumul atom expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    if _is_int_couple(height):
        return CpoFunctionCall(Oper_step_at_start, Type_CumulAtom, (interval,
                                                                    _convert_arg(height[0], "height[0]", Type_PositiveInt),
                                                                    _convert_arg(height[1], "height[1]", Type_PositiveInt)))

    height = _convert_arg(height, "height", Type_PositiveInt, "Argument 'height' should be an integer or a range expressed as a tuple of integers")
    return CpoFunctionCall(Oper_step_at_start, Type_CumulAtom, (interval,height))


def step_at_end(interval, height):
    """ Returns an elementary cumul function of constant value after the end of an interval.

    This function returns an elementary cumul function expression that,
    whenever interval variable *interval* is present, is equal to 0 before the start of *interval*
    and equal to *height* after the end of *interval*.

    If *height* is a range specified as a tuple of integers, it means that the height value of the function
    is part of the decisions of the problem and will be fixed by the engine within this specified range.

    When interval variable *interval* is absent, the function is the constant zero function.

    Args:
        interval: Interval variable contributing to the cumul function.
        height:   Non-negative integer representing the height of the contribution,
                  or tuple of 2 non-negative integers representing the range of possible values
                  for the height of the contribution.
    Returns:
        A cumul atom expression
    """
    interval = _convert_arg(interval, "interval", Type_IntervalVar)
    if _is_int_couple(height):
        return CpoFunctionCall(Oper_step_at_end, Type_CumulAtom, (interval,
                                                                  _convert_arg(height[0], "height[0]", Type_PositiveInt),
                                                                  _convert_arg(height[1], "height[1]", Type_PositiveInt)))

    height = _convert_arg(height, "height", Type_PositiveInt, "Argument 'height' should be an integer or a range expressed as a tuple of integers")
    return CpoFunctionCall(Oper_step_at_end, Type_CumulAtom, (interval,height))


def height_at_start(interval, function, absentValue=None):
    """ Returns the contribution of an interval variable to a cumul function at its start point.

    Whenever interval variable *interval* is present, this function returns an integer expression
    that represents the total contribution of the start of interval variable *interval* to the cumul *function*.

    When interval variable *interval* is absent, this function returns a constant integer expression equal
    to *absentValue* (zero by default).

    Args:
        interval:    Interval variable.
        function:    Cumul function expression.
        absentValue (Optional): Value to return if the interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    if absentValue is None:
        return CpoFunctionCall(Oper_height_at_start, Type_IntExpr, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                                    _convert_arg(function, "function", Type_CumulExpr)))
    return CpoFunctionCall(Oper_height_at_start, Type_IntExpr, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                                _convert_arg(function, "function", Type_CumulExpr),
                                                                _convert_arg(absentValue, "absentValue", Type_Int)))


def height_at_end(interval, function, absentValue=None):
    """ Returns the contribution of an interval variable to a cumul function at its end point.

    Whenever interval variable *interval* is present, this function returns an integer expression
    that represents the total contribution of the end of interval variable *interval* to the cumul *function*.

    When interval variable *interval* is absent, this function returns a constant integer expression equal
    to *absentValue* (zero by default).

    Args:
        interval:    Interval variable.
        function:    Cumul function expression.
        absentValue (Optional): Value to return if the interval variable interval becomes absent.
    Returns:
        An integer expression
    """
    if absentValue is None:
        return CpoFunctionCall(Oper_height_at_end, Type_IntExpr, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                                  _convert_arg(function, "function", Type_CumulExpr)))
    return CpoFunctionCall(Oper_height_at_end, Type_IntExpr, (_convert_arg(interval, "interval", Type_IntervalVar),
                                                              _convert_arg(function, "function", Type_CumulExpr),
                                                              _convert_arg(absentValue, "absentValue", Type_Int)))


def always_in(function, interval, min, max, _x=None):
    """ These constraints restrict the possible values of a *cumulExpr* or *stateFunction*
    to a particular range during a variable or fixed interval.

    These functions return a constraints that restricts the possible values of *function* to a particular range
    [*min*, *max*] during an interval variable *interval* or a fixed interval [*start*, *end*).

    In the case of an interval variable *interval*, this constraint is active only when the interval variable
    is present.
    If the interval is absent, the constraint is always satisfied, regardless of the value of *function*.

    When the constraint is posted on a state function, the range constraint holds only on the segments
    where the state function is defined.

    Args:
        function: Constrained cumul expression or state function.
        interval: Interval variable contributing to the cumul function,
                  or fixed interval expressed as a tuple of 2 integers.
        min: Minimum of the allowed range for values of function during the interval, in [0..intervalmax).
        max: Maximum of the allowed range for values of function during the interval, in [0..intervalmax).
    Returns:
        Constraint expression
    """
    function = build_cpo_expr(function)
    assert function.is_kind_of(Type_CumulExpr) or function.is_kind_of(Type_StateFunction), \
        "Argument 'function' should be a cumul expression or a state function"

    # Case of 5 arguments (backward compatibility with previous implementation)
    if _x is not None:
        msg = "Deprecated calling form, consult documentation for details"
        return CpoFunctionCall(Oper_always_in, Type_Constraint, (function,
                                                                 _convert_arg(interval, "interval", Type_TimeInt, msg),
                                                                 _convert_arg(min, "min", Type_TimeInt, msg),  # Min is end
                                                                 _convert_arg(max, "max", Type_PositiveInt, msg),
                                                                 _convert_arg(_x, "_x", Type_PositiveInt, msg)))

    assert 0 <= min <= INTERVAL_MAX, "Argument 'min' should be in 0..{}".format(INTERVAL_MAX)
    assert 0 <= max <= INTERVAL_MAX, "Argument 'max' should be in 0..{}".format(INTERVAL_MAX)
    min = _convert_arg(min, "min", Type_PositiveInt)
    max = _convert_arg(max, "max", Type_PositiveInt)
    if _is_int_couple(interval):
        return CpoFunctionCall(Oper_always_in, Type_Constraint, (function,
                                                                 _convert_arg(interval[0], "interval[0]", Type_TimeInt),
                                                                 _convert_arg(interval[1], "interval[1]", Type_TimeInt),
                                                                 min, max))

    interval = _convert_arg(interval, "interval", Type_IntervalVar,
                            "Argument 'interval' should be an interval variable or a fixed interval expressed as a tuple of integers")
    return CpoFunctionCall(Oper_always_in, Type_Constraint, (function, interval, min, max))


def cumul_range(function, min, max):
    """ Limits the range of a cumul function expression.

    This function returns a constraint that restricts the possible values of cumul *function*
    to belong to a range [*min*, *max*].

    Args:
        function: Cumul function expression.
        min: Minimum of the range of allowed values for the cumul function.
        max: Maximum of the range of allowed values for the cumul function.
    Returns:
        Constraint expression
    """
    return CpoFunctionCall(Oper_cumul_range, Type_Constraint, (_convert_arg(function, "function", Type_CumulExpr),
                                                               _convert_arg(min, "min", Type_IntExpr),
                                                               _convert_arg(max, "max", Type_IntExpr)))



#==============================================================================
#  State functions
#==============================================================================

def always_no_state(function, interval):
    """ This constraint ensures that a state function is undefined on an interval.

    This function returns a constraint that ensures that *function* is undefined everywhere on
    an interval (interval variable *interval* when it is present or fixed interval [*start*, *end*)).
    This constraint will ensure, in particular, that no interval variable that requires the function
    to be defined (see *always_equal*, *always_constant*) can overlap with interval variable *interval*
    or fixed interval [*start*, *end*)).

    Args:
        function: Constrained state function.
        interval: Interval variable contributing to the cumul function,
                  or fixed interval expressed as a tuple of 2 integers.
    Returns:
        Constraint expression
    """
    function = _convert_arg(function, "function", Type_StateFunction)
    if _is_int_couple(interval):
        return CpoFunctionCall(Oper_always_no_state, Type_Constraint, (function,
                                                                       _convert_arg(interval[0], "interval[0]", Type_TimeInt),
                                                                       _convert_arg(interval[1], "interval[1]", Type_TimeInt)))
    interval = _convert_arg(interval, "interval", Type_IntervalVar,
                            "Argument 'interval' should be an interval variable or a fixed interval expressed as a tuple of integers")
    return CpoFunctionCall(Oper_always_no_state, Type_Constraint, (function, interval))


def always_constant(function, interval, isStartAligned=None, isEndAligned=None):
    """ This constraint ensures a constant state for a state function on an interval.

    This function returns a constraint that ensures that *function* is defined everywhere on an
    interval variable *interval* when it is present or a fixed interval [*start*, *end*))
    and remains constant over this interval.

    Generally speaking, the optional boolean values *isStartAligned* and *isEndAligned* allow
    synchronization of start and end with the intervals of the state function:

     * When *isStartAligned* is true, it means that start must be the start of an interval of the state function.
     * When *isEndAligned* is true, it means that end must be the end of an interval of the state function.

    Args:
        function: Constrained state function.
        interval: Interval variable contributing to the cumul function,
                  or fixed interval expressed as a tuple of 2 integers.
        isStartAligned (Optional): Boolean flag that states whether the interval is start aligned (default: no alignment).
        isEndAligned (Optional): Boolean flag that states whether the interval is end aligned (default: no alignment).
    Returns:
        Constraint expression
    """
    if isStartAligned is None:
        if isEndAligned is not None:
            isStartAligned = 0
    else:
        if isEndAligned is None:
            isEndAligned = 0
    function = _convert_arg(function, "function", Type_StateFunction)

    if _is_int_couple(interval):
        start = _convert_arg(interval[0], "interval[0]", Type_TimeInt)
        end = _convert_arg(interval[1], "interval[1]", Type_TimeInt)
        if isStartAligned is None:
            return CpoFunctionCall(Oper_always_constant, Type_Constraint, (function, start, end))
        return CpoFunctionCall(Oper_always_constant, Type_Constraint, (function, start, end,
                                                                       _convert_arg_bool_int(isStartAligned, "isStartAligned"),
                                                                       _convert_arg_bool_int(isEndAligned, "isEndAligned")))

    interval = _convert_arg(interval, "interval", Type_IntervalVar,
                            "Argument 'interval' should be an interval variable or a fixed interval expressed as a tuple of integers")
    if isStartAligned is None:
        return CpoFunctionCall(Oper_always_constant, Type_Constraint, (function, interval))
    return CpoFunctionCall(Oper_always_constant, Type_Constraint, (function, interval,
                                                                   _convert_arg_bool_int(isStartAligned, "isStartAligned"),
                                                                   _convert_arg_bool_int(isEndAligned, "isEndAligned")))


def always_equal(function, interval, value, isStartAligned=None, isEndAligned=None):
    """ This constraint fixes a given state for a state function during a variable or fixed interval.

    This function returns a constraint that ensures that *function* is defined everywhere on an
    interval *interval* when it is present or a fixed interval [*start*, *end*)), and remains equal
    to value *val* over this interval.

    Generally speaking, the optional boolean values *isStartAligned* and *isEndAligned* allow
    synchronization of start and end with the intervals of the state function:

     * When *isStartAligned* is true, it means that start must be the start of an interval of the state function.
     * When *isEndAligned* is true, it means that end must be the end of an interval of the state function.

    Args:
        function: Constrained state function.
        interval: Interval variable contributing to the cumul function,
                  or fixed interval expressed as a tuple of 2 integers.
        value: Value of the function during the interval.
        isStartAligned (Optional): Boolean flag that states whether the interval is start aligned (default: no alignment).
        isEndAligned (Optional): Boolean flag that states whether the interval is end aligned (default: no alignment).
    Returns:
        Constraint expression
    """
    if isStartAligned is None:
        if isEndAligned is not None:
            isStartAligned = 0
    else:
        if isEndAligned is None:
            isEndAligned = 0
    function = _convert_arg(function, "function", Type_StateFunction)
    value = _convert_arg(value, "value", Type_PositiveInt)

    if _is_int_couple(interval):
        start = _convert_arg(interval[0], "interval[0]", Type_TimeInt)
        end = _convert_arg(interval[1], "interval[1]", Type_TimeInt)
        if isStartAligned is None:
            return CpoFunctionCall(Oper_always_equal, Type_Constraint, (function, start, end, value))
        return CpoFunctionCall(Oper_always_equal, Type_Constraint, (function, start, end, value,
                                                                    _convert_arg_bool_int(isStartAligned, "isStartAligned"),
                                                                    _convert_arg_bool_int(isEndAligned, "isEndAligned")))

    interval = _convert_arg(interval, "interval", Type_IntervalVar,
                            "Argument 'interval' should be an interval variable or a fixed interval expressed as a tuple of integers")
    if isStartAligned is None:
        return CpoFunctionCall(Oper_always_equal, Type_Constraint, (function, interval, value))
    return CpoFunctionCall(Oper_always_equal, Type_Constraint, (function, interval, value,
                                                                _convert_arg_bool_int(isStartAligned, "isStartAligned"),
                                                                _convert_arg_bool_int(isEndAligned, "isEndAligned")))


#==============================================================================
#  Search phases
#==============================================================================

#--- Variable evaluators ------------------------------------------------------

def domain_size():
    """ Variable evaluator that represents the number of elements in the current domain
    of the variable chosen by the search.

    Returns:
        An evaluator of integer variable
    """
    return CpoFunctionCall(Oper_domain_size, Type_IntVarEval, ())


def domain_max():
    """ Variable evaluator that represents the maximum value in the current domain
    of the variable chosen by the search.

    Returns:
        An evaluator of integer variable
    """
    return CpoFunctionCall(Oper_domain_max, Type_IntVarEval, ())


def domain_min():
    """ Variable evaluator that represents the minimum value in the current domain
    of the variable chosen by the search.

    Returns:
        An evaluator of integer variable
    """
    return CpoFunctionCall(Oper_domain_min, Type_IntVarEval, ())


def var_impact():
    """ Variable evaluator that represents the average reduction of the search
    space observed so far for this variable.

    The greater the evaluation, the more space reduction this variable achieves.
    In general, it is a good strategy to start with variable having the greatest impact in order to
    reduce the search space size.

    Returns:
        An evaluator of integer variable
    """
    return CpoFunctionCall(Oper_var_impact, Type_IntVarEval, ())


def var_index(vars, defaultIndex=None):
    """ Variable evaluator that represents the index of the variable in an array of variables.

    The evaluation of vars[i] is i.
    If the variable does not appear in the array, defaultIndex is the evaluation returned.

    Args:
        vars:  Array of integer variables.
        defaultIndex (Optional): Default index that is returned if the variable does not appear in the array.
                      If not given, -1 is used.
    Returns:
        An evaluator of integer variable
    """
    vars = _convert_arg(vars, "vars", Type_IntVarArray)
    if defaultIndex is None:
        return CpoFunctionCall(Oper_var_index, Type_IntVarEval, (vars, ))
    return CpoFunctionCall(Oper_var_index, Type_IntVarEval, (vars, _convert_arg(defaultIndex, "defaultIndex", Type_Float)))


def var_local_impact(effort=None):
    """ Variable evaluator that represents the impact of the variable computed at
    the current node of the search tree.

    This computation is made by probing on values from the domain of the variables.
    The parameter *effort* indicates how much effort should be spent to compute this impact.
    When effort is equal to -1, every value of the domain is probed, otherwise the number of probes effort
    will increase as the effort value increases.

    Args:
        effort (Optional): Integer representing how much effort should be spent to compute this impact.
                By default, -1 is used.
    Returns:
        An evaluator of integer variable
    """
    if effort is None:
        return CpoFunctionCall(Oper_var_local_impact, Type_IntVarEval, ())
    return CpoFunctionCall(Oper_var_local_impact, Type_IntVarEval, (_convert_arg(effort, "effort", Type_Int),))


def var_success_rate():
    """ Variable evaluator that represents the success rate of the variable.

    Assuming the evaluated variable has been instantiated n times so far and this has resulted in f failures,
    the success rate is (n-f)/n.

    Returns:
        An evaluator of integer variable
    """
    return CpoFunctionCall(Oper_var_success_rate, Type_IntVarEval, ())


def impact_of_last_branch():
    """ Variable evaluator that represents the domain reduction that the
    last instantiation made by search has achieved on the evaluated variable.

    Values returned range from 0 to 1.0.
    If the value is close to zero then there wasn't much domain reduction on the evaluated variable
    during the last instantiation.
    If the value is close to one, the domain was reduced considerably.

    Returns:
        An evaluator of integer variable
    """
    return CpoFunctionCall(Oper_impact_of_last_branch, Type_IntVarEval, ())


def explicit_var_eval(vars, vals, defaultEval=None):
    """ Variable evaluator that gives an explicit value to variables.

    The evaluations of variables in the array *vars* are explicitly defined in the array of values *vals*,
    that is the evaluation of *vars[i]* is *vals[i]*.
    The arrays *vars* and *vals* must have the same size.

    The evaluation of a variable that does not appear in the array is given by *defautEval*.

    Args:
        vars: Array of integer variables
        vals: Array of values
        defaultEval (Optional): Default value of a variable that is not in the array
                     If not given, default eval value is zero.
    Returns:
        An evaluator of integer variable
    """
    vars = _convert_arg(vars, "vars", Type_IntVarArray)
    vals = _convert_arg(vals, "vals", Type_FloatArray)
    if defaultEval is None:
        return CpoFunctionCall(Oper_explicit_var_eval, Type_IntVarEval, (vars, vals))
    return CpoFunctionCall(Oper_explicit_var_eval, Type_IntVarEval, (vars, vals,
                                                                     _convert_arg(defaultEval, "defaultEval", Type_Float)))

#--- Value evaluators ---------------------------------------------------------

def value():
    """ Value evaluator that returns as evaluation the value itself.

    This is useful to define instantiation strategies that choose the smallest or the largest
    value in a domain.

    Returns:
        An evaluator of integer value
    """
    return CpoFunctionCall(Oper_value, Type_IntValueEval, ())


def value_impact():
    """ Value evaluator that represents the average reduction of the search space observed
    so far when instantiating the selected variable to the evaluated value.

    The greater the evaluation, the more space reduction this instantiation achieves.
    In general it is a good strategy to prefer a value having the smallest impact.

    Returns:
        An evaluator of integer value
    """
    return CpoFunctionCall(Oper_value_impact, Type_IntValueEval, ())


def value_success_rate():
    """ Value evaluator that represents the success rate of instantiating the selected variable
    to the evaluated value.

    Assuming the selected variable has been instantiated to the evaluated value n times so far
    and this has resulted in f failures, the success rate is (n-f)/n.

    Returns:
        An evaluator of integer value
    """
    return CpoFunctionCall(Oper_value_success_rate, Type_IntValueEval, ())


def value_index(vals, defaultValue=None):
    """  Value evaluator that represents the index of the value in an array of integer values.

    The evaluation of *vals[i]* is *i*.
    If the value does not appear in the array, *defautEval* is the evaluation returned.

    Args:
        vals: Array of integer values
        defaultValue (Optional): Default value that is returned if value is not in the array.
                      By default, this value is -1.
    Returns:
        An evaluator of integer value
    """
    vals = _convert_arg(vals, "vals", Type_IntArray)
    if defaultValue is None:
        return CpoFunctionCall(Oper_value_index, Type_IntValueEval, (vals,))
    return CpoFunctionCall(Oper_value_index, Type_IntValueEval, (vals, _convert_arg(defaultValue, "defaultValue", Type_Float)))


def explicit_value_eval(vals, evals, defaultEval=None):
    """ Value evaluator hat gives an explicit evaluation to values.

    The evaluations of elements of array *vals* are explicitly defined in the array *evals*,
    that is, the evaluation of *vals[i]* is *evals[i]*.
    The arrays *vals* and *evals* must have the same size.

    The evaluation of a value that does not appear in *vals* is given by *defautEval*.

    Args:
        vals: Array of values
        evals: Array of the evaluations of values
        defaultEval (Optional): Evaluation of a value that does not appears in *vals*.
                     By default, this value is zero.
    Returns:
        An evaluator of integer value
    """
    vals = _convert_arg(vals, "vals", Type_IntArray)
    evals = _convert_arg(evals, "evals", Type_FloatArray)
    if defaultEval is None:
        return CpoFunctionCall(Oper_explicit_value_eval, Type_IntValueEval, (vals, evals))
    return CpoFunctionCall(Oper_explicit_value_eval, Type_IntValueEval, (vals, evals,
                                                                         _convert_arg(defaultEval, "defaultEval", Type_Float)))


#--- Variable selectors -------------------------------------------------------

def select_smallest(evaluator, minNumber=None, tolerance=None):
    """ Selector of integer variables or value having the smallest evaluation according to a given evaluator.

    This function returns a selector of value assignments to a variable that selects all values having the
    smallest evaluation according to the evaluator e.

    If *minNumber* is provided, this function returns a selector of integer variable or value assignments that selects
    at least *minNumber* values having the smallest evaluation according to the evaluator.
    The parameter *minNumber* must be at least 1.
    For example, suppose that eight domain values (1-8) have evaluations
    (1) 9, (2) 5, (3) 6, (4) 3, (5) 8, (6) 1, (7) 3, (8) 2.
    When ordered by increasing evaluation, this gives: (6) 1, (8) 2, (4) 3, (7) 3, (2) 5, (3) 6, (5) 8, (1) 9.
    If *minNumber* is 1, then value 6 would be selected, if it is 2, then values 6 and 8 would be selected,
    and if it is 3 then values 6, 8, 4 and 7 would be selected.
    Note that when minNumber is 3, both values 4 and 7 are selected as both are considered equivalent.
    In addition, it is possible to specify a non-integer value of *minNumber*.
    In this case, at least floor(*minNumber*) selections are made, with probability *minNumber* - floor(*minNumber*)
    of selecting an additional value.
    It is still possible that this selector can select less values than *minNumber* if there are less
    than *minNumber* values supplied to it for selection, in which case all supplied values are selected.

    If *tolerance* is provided (exclusively with *minNumber*), this function returns a selector of integer variable
    or value assignments that selects all domain values whose evaluations are in the range [*min*, *min* + *tolerance*],
    where *min* is is the minimum valuation by the evaluator over the domain values to be evaluated.
    The parameter *tolerance* must be non-negative.

    Args:
        evaluator: Evaluator of integer variable or integer value
        minNumber (Optional): Minimum number of values that are selected,
                    with the smallest evaluation according to the evaluator e
        tolerance (Optional): Tolerance of the values to be selected
    Returns:
        An expression of type selector of integer value or selector of integer variable
    """
    evaluator = build_cpo_expr(evaluator)
    if evaluator.is_kind_of(Type_IntValueEval):
        rtype = Type_IntValueSelector
    elif evaluator.is_kind_of(Type_IntVarEval):
        rtype = Type_IntVarSelector
    else:
        assert False, "Argument 'evaluator' should be an evaluator of integer variable or an evaluator of integer value"

    if minNumber is None:
        if tolerance is None:
            return CpoFunctionCall(Oper_select_smallest, rtype, (evaluator,))
        return CpoFunctionCall(Oper_select_smallest, rtype, (evaluator,
                                                             _convert_arg(tolerance, "tolerance", Type_Float)))

    assert tolerance is None, "Arguments 'minNumber' and 'tolerance' can not be set together"
    return CpoFunctionCall(Oper_select_smallest, rtype, (_convert_arg(minNumber, "minNumber", Type_Float),
                                                         evaluator))



def select_largest(evaluator, minNumber=None, tolerance=None):
    """ Selector of integer variables or value having the largest evaluation according to a given evaluator.

    This function returns a selector of value assignments to a variable that selects all values having the
    largest evaluation according to the evaluator e.

    If *minNumber* is provided, this function returns a selector of integer variable or value assignments that selects
    at least *minNumber* values having the largest evaluation according to the evaluator.
    The parameter *minNumber* must be at least 1.
    For example, suppose that eight domain values (1-8) have evaluations
    (1) 5, (2) 8, (3) 3, (4) 9, (5) 2, (6) 8, (7) 1, (8) 7.
    When ordered by decreasing evaluation, this gives: (4) 9, (2) 8, (6) 8, (8) 7, (1) 5, (3) 3, (5) 2, (7) 1.
    If *minNumber* is 1, then value 4 would be selected, if it is 2 or 3, then values 2 and 6 would be selected,
    and if it is 4 then values 4, 2, 6, and 8 would be selected.
    Note that when *minNumber* is 2, both 2 and 6 are selected as both are considered equivalent.
    In addition, it is possible to specify a non-integer value of *minNumber*.
    In this case, at least floor(*minNumber*) selections are made, with probability *minNumber* - floor(*minNumber*)
    of selecting an additional value.
    It is still possible that this selector can select less domain values than *minNumber* if there are
    less than *minNumber* values supplied to it for selection, in which case all supplied values are selected.

    If *tolerance* is provided (exclusively with *minNumber*), this function returns a selector of integer variable
    or value assignments that selects all domain values whose evaluations are in the range [*max* - *tolerance*, *max*],
    where *max* is is the maximum valuation by the evaluator over the domain values to be evaluated.
    The parameter *tolerance* must be non-negative.

    Args:
        evaluator: Evaluator of integer variable or integer value
        minNumber (Optional): Minimum number of values that are selected,
                    with the smallest evaluation according to the evaluator e
        tolerance (Optional): Tolerance of the values to be selected
    Returns:
        An expression of type selector of integer value or selector of integer variable
    """
    evaluator = build_cpo_expr(evaluator)
    if evaluator.is_kind_of(Type_IntValueEval):
        rtype = Type_IntValueSelector
    elif evaluator.is_kind_of(Type_IntVarEval):
        rtype = Type_IntVarSelector
    else:
        assert False, "Argument 'evaluator' should be an evaluator of integer variable or an evaluator of integer value"

    if minNumber is None:
        if tolerance is None:
            return CpoFunctionCall(Oper_select_largest, rtype, (evaluator,))
        return CpoFunctionCall(Oper_select_largest, rtype, (evaluator,
                                                            _convert_arg(tolerance, "tolerance", Type_Float)))

    assert tolerance is None, "Arguments 'minNumber' and 'tolerance' can not be set together"
    return CpoFunctionCall(Oper_select_largest, rtype, (_convert_arg(minNumber, "minNumber", Type_Float),
                                                        evaluator))


def select_random_var():
    """ Selector of integer variables that selects a variable randomly.

    This selector selects only one variable.

    Returns:
        A selector of integer variable
    """
    return CpoFunctionCall(Oper_select_random_var, Type_IntVarSelector, ())


def select_random_value():
    """ Selector of integer variable value assignments that selects a domain value randomly.

    This selector selects only one value.

    Returns:
        A selector of integer value
    """
    return CpoFunctionCall(Oper_select_random_value, Type_IntValueSelector, ())


#--- Search phase -------------------------------------------------------------

def search_phase(vars=None, varchooser=None, valuechooser=None):
    """ Creates a search phase.

    A search phase is composed of:

     * an array of variables to instantiate,
     * a variable chooser that defines how the next variable to instantiate is chosen
     * a value chooser that defines how values are chosen when instantiating variables.

    The embedded search strategy, determined by the value of the solver parameter *SearchType*,
    will use the search phases to instantiate the variables for which a search phase is specified.

    Several search phases can be given to the CP model using method :meth:`docplex.cp.model.CpoModel.set_search_phases`
    or :meth:`docplex.cp.model.CpoModel.add_search_phase`.

    The order of the search phases in the array is important. In the CP search strategy, the variables will be
    instantiated phase by phase starting by the first phase of the array.
    It is not necessary that the variables in the search phases cover all the variables of the problem.
    It can be assumed that a search phase containing all the problem variables is implicitly added to the
    end of the given array of search phases.

    A search phase can be created with an array of variables only.
    The embedded search will then choose an instantiation strategy automatically.
    For instance, assuming that x and y are arrays of integer variables, the following code:
    ::

        mdl.set_search_phases([search_phase(x), search_phase(y)])

    indicates to CP search that variables from the array x must be instantiated before those from the array y.
    The way to instantiate them will be chosen by CP search.

    Similarly, it is not necessary to specify an array of variables to a search phase.
    A search phase defined this way will be applied to every integer variable extracted from the model.

    Args:
        vars (Optional): Array of integer, interval or sequence variables.
        varchooser (Optional): Chooser of integer variable.
                    Used only if *vars* is undefined or contains an array of integer variables.
                    Must be defined if a *valuechooser* is defined.
        valuechooser (Optional): Chooser of integer value
                    Used only if *vars* is undefined or contains an array of integer variables.
                    Must be defined if a *varchooser* is defined.
    Returns:
        A search phase expression
    """
    if vars is None:
        assert (varchooser is not None) and (valuechooser is not None), \
            "If no array of variable is given, then variable and value choosers must be given"
        return CpoFunctionCall(Oper_search_phase, Type_SearchPhase, (_convert_arg(varchooser, "varchooser", Type_IntVarChooser),
                                                                     _convert_arg(valuechooser, "valuechooser", Type_IntValueChooser)))

    vars = build_cpo_expr(vars)
    if vars.is_kind_of(Type_IntVarArray):
        if varchooser is None:
            assert valuechooser is None, "Variable and value chooser should be defined together"
            return CpoFunctionCall(Oper_search_phase, Type_SearchPhase, (vars,))
        assert valuechooser is not None, "Variable and value chooser should be defined together"
        return CpoFunctionCall(Oper_search_phase, Type_SearchPhase, (vars,
                                                                     _convert_arg(varchooser, "varchooser", Type_IntVarChooser),
                                                                     _convert_arg(valuechooser, "valuechooser", Type_IntValueChooser)))

    assert vars.is_kind_of(Type_IntervalVarArray) or vars.is_kind_of(Type_SequenceVarArray), \
        "Argument 'vars' should be an array of integer, interval or sequence variables"
    return CpoFunctionCall(Oper_search_phase, Type_SearchPhase, (vars,))


