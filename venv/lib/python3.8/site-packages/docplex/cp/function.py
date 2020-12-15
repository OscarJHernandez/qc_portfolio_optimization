# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Philippe LABORIE, IBM Analytics, France Lab, Gentilly

"""
This module contains the objects representing CP Optimizer function expressions,
Stepwise and Piecewise Linear functions:

In particular, it defines the following classes:

 * :class:`CpoFunction`: the root class of all function expressions,
 * :class:`CpoSegmentedFunction`: for functions represented as a list of segments,
 * :class:`CpoStepFunction`: for functions represented as a list of steps.


Detailed description
--------------------
"""

from docplex.cp.catalog import Type_SegmentedFunction, Type_StepFunction
from docplex.cp.expression import CpoExpr, INTERVAL_MIN, INTERVAL_MAX
from docplex.cp.utils import *
import bisect


class CpoFunction(CpoExpr):
    """ Root class for step and segmented functions.
    """
    __slots__ = ('_s0',  # Initial slope of the function
                 '_v0',  # Initial value of the function
                 '_x',   # Abscissas
                 '_v',   # Values
                 '_s',   # Slopes
                 )

    def __init__(self, typ, s0=None, v0=None, x=None, v=None, s=None, name=None):
        """
        Args:
            typ: Function type.
            s0:  Initial slope of the function.
            v0:  Initial value of the function.
            x:   Abscissas (assumed to be sorted and all different).
            v:   Values.
            s:   Slopes.
        """
        super(CpoFunction, self).__init__(typ, name)
        if v0 is None:
            self._v0 = 0
        else:
            self._v0 = v0
        if s0 is None:
            self._s0 = 0
        else:
            # Test type is segmented function when some slope is specified
            assert (s0 == 0) or (typ == Type_SegmentedFunction), "Slopes are only supported for segmented functions"
            # Test there exists at least one abscissa
            assert s0 == 0 or (x is not None and 0 < len(x)), "No abscissa for initial slope"
            self._s0 = s0
        if x is None:
            self._x = []
        else:
            # Test abscissa integrality
            assert all(is_int(elt) for elt in x), "Abscissa value is not an integer"
            # Test abscissa list is strictly increasing
            assert all(x[i] < x[i + 1] for i in range(len(x) - 1)), "Abscissa values are not strictly increasing"
            self._x = x
        l = len(self._x)
        if v is None:
            self._v = []
        else:
            # Test value integrality in case of step function
            if typ == Type_StepFunction:
                assert all(is_int(elt) for elt in v), "Value is not an integer"
            self._v = v
        assert len(self._v) == l, "Lists x and v must have the same length"
        if s is None:
            self._s = [0] * len(self._x)
        else:
            # Test type is segmented function when some slope is specified
            assert all(
                v == 0 for v in s) or typ == Type_SegmentedFunction, "Slopes are only supported for segmented functions"
            self._s = s
        assert len(self._s) == l, "Lists x and s must have the same length"

    def _get_expr_string(self):
        """ Get the string representing this expression, name excluded. """
        return "Function(...)"

    @property
    def v0(self):
        return self._v0

    @property
    def s0(self):
        return self._s0

    @property
    def x(self):
        return self._x

    @property
    def v(self):
        return self._v

    @property
    def s(self):
        return self._s

    @property
    def is_step_function(self):
        return self.type == Type_StepFunction

    @property
    def is_segmented_function(self):
        return self.type == Type_SegmentedFunction

    def get_value(self, t):
        """ Gets the value of the function.

        Gets the value of the function for an abscissa `t`.
        Complexity is in `O(log n)` for a function with `n` segments.

        Args:
            t (int or float): Abscissa value.

        Returns:
            Value of the function for abscissa `t`.
        """
        l = len(self._x)
        if l == 0:
            # Constant step function
            return self._v0
        if t < self._x[0]:
            # Abscissa t is on initial step
            return self._v0 - (self._x[0] - t) * self._s0
        # Abscissa is on another step
        i = bisect.bisect_right(self._x, t)
        if self._s is None:
            return self._v[i - 1]
        else:
            return self._v[i - 1] + (t - self._x[i - 1]) * self._s[i - 1]

    def copy(self, type=None):
        """ Creates and returns a copy of the function. """
        xcopy = self._x[:]
        vcopy = self._v[:]
        scopy = self._s[:]
        if type is None:
            tp = self.type
        else:
            tp = type
        return CpoFunction(tp, self._s0, self._v0, xcopy, vcopy, scopy)

    def set_slope(self, x1, x2, v1, s):
        """ Sets the value of the invoking function over an interval.

        Sets the value of the invoking function on interval `[x1,x2)` to be equal
        to equal to `f(x) = v1 + s * (x-x1)`.

        Args:
            x1 (int): Start of the interval.
            x2 (int): End of the interval.
            v1 (int or float): Function value at `x1`.
            s (int or float): Function slope on interval `[x1,x2)`.
        """
        assert s == 0 or not self.is_step_function, "Function not allowed on CpoStepFunction"
        assert is_int(x1) and is_int(x2), "Interval [x1,x2) must be integer"
        if x2 <= x1:
            return
        l = len(self._x)
        if l == 0:
            # Constant step function of value self._v0
            if not (s == 0 and v1 == self._v0):
                self._x.append(x1)
                self._v.append(v1)
                self._s.append(s)
                if x2 < INTERVAL_MAX:
                    self._x.append(x2)
                    self._v.append(self._v0)
                    self._s.append(0)
            return
        if x1 < self._x[0]:
            if self._s0 == s and self._v0 + s * (x1 - self._x[0]) == v1:
                # No change on initial segment
                if x2 <= self._x[0]:
                    # Completely inside initial segment
                    return
                # Make initial segment longer
                self._v0 += s * (x2 - self._x[0])
                i2 = bisect.bisect_right(self._x, x2) - 1
                if 0 < i2:
                    # Remove segments 0..i2-1 and shift arrays
                    del self._x[:i2]
                    del self._v[:i2]
                    del self._s[:i2]
                # Adjust beginning of first segment
                self._v[0] += (x2 - self._x[0]) * self._s[0]
                self._x[0] = x2
                return
            else:
                # Adjust value of initial segment
                v0 = self._v0
                x0 = self._x[0]
                self._v0 += self._s0 * (x1 - self._x[0])
                # Add new segment
                self._x.insert(0, x1)
                self._v.insert(0, v1)
                self._s.insert(0, s)
                if x2 < x0:
                    self._x.insert(1, x2)
                    self._v.insert(1, v0 + (x2 - x0) * self._s0)
                    self._s.insert(1, self._s0)
                    return
                elif x0 <= x2:
                    self.set_slope(x0, x2, v1 + (x0 - x1) * s, s)
                    return
        else:
            # x1 >= self._x[0]:
            i1 = bisect.bisect_right(self._x, x1) - 1
            i2 = bisect.bisect_right(self._x, x2) - 1
            if self._s[i1] == s and self._v[i1] + s * (x1 - self._x[i1]) == v1:
                # Same as current segment
                if i1 == i2:
                    return
                else:
                    self.set_slope(self._x[i1 + 1], x2, v1 + (self._x[i1 + 1] - x1) * s, s)
                    return
            else:
                xcurr = self._x[i1]
                vcurr = self._v[i1]
                scurr = self._s[i1]
                if x1 > self._x[i1]:
                    self._x.insert(i1 + 1, x1)
                    self._v.insert(i1 + 1, v1)
                    self._s.insert(i1 + 1, s)
                    i1 += 1
                    i2 += 1
                else:
                    assert x1 == self._x[i1]
                    self._x[i1] = x1
                    self._v[i1] = v1
                    self._s[i1] = s
                if i1 == i2:
                    self._x.insert(i1 + 1, x2)
                    self._v.insert(i1 + 1, vcurr + (x2 - xcurr) * scurr)
                    self._s.insert(i1 + 1, scurr)
                    return
                else:
                    # Adjust beginning of first segment
                    self._v[i2] += (x2 - self._x[i2]) * self._s[i2]
                    self._x[i2] = x2
                    del self._x[i1 + 1:i2]
                    del self._v[i1 + 1:i2]
                    del self._s[i1 + 1:i2]
                    return

    def add_slope(self, x1, x2, v1, s):
        """ Adds a piecewise linear step on an interval.

        Adds a piecewise linear step `f(x) = v1 + s * (x-x1)` to the invoking
        function on interval `[x1,x2)`.

        Args:
            x1 (int): Start of the interval.
            x2 (int): End of the interval.
            v1 (int or float): Added value at `x1`.
            s (int or float): Added function slope on interval `[x1,x2)`;
        """
        if v1 == 0 and s == 0:
            return
        assert s == 0 or not self.is_step_function, "Function not allowed on CpoStepFunction"
        assert is_int(x1) and is_int(x2), "Interval [x1,x2) must be integer"
        if x2 <= x1:
            return
        l = len(self._x)
        if l == 0:
            # Constant step function of value self._v0
            self._x.append(x1)
            self._v.append(self._v0 + v1)
            self._s.append(s)
            if x2 < INTERVAL_MAX:
                self._x.append(x2)
                self._v.append(self._v0)
                self._s.append(0)
            return
        if x1 < self._x[0]:
            # Adjust value of initial segment
            v0 = self._v0
            x0 = self._x[0]
            self._v0 += self._s0 * (x1 - self._x[0])
            # Add new segment
            self._x.insert(0, x1)
            self._v.insert(0, self._v0 + v1)
            self._s.insert(0, self._s0 + s)
            if x2 < x0:
                self._x.insert(1, x2)
                self._v.insert(1, v0 + (x2 - x0) * self._s0)
                self._s.insert(1, self._s0)
                return
            elif x0 <= x2:
                self.add_slope(x0, x2, v1 + (x0 - x1) * s, s)
                return
        else:
            # x1 >= self._x[0]:
            i1 = bisect.bisect_right(self._x, x1) - 1
            i2 = bisect.bisect_right(self._x, x2) - 1
            xcurr = self._x[i1]
            vcurr = self._v[i1]
            scurr = self._s[i1]
            if x1 > self._x[i1]:
                self._x.insert(i1 + 1, x1)
                self._v.insert(i1 + 1, vcurr + (x1 - self._x[i1]) * scurr + v1)
                self._s.insert(i1 + 1, scurr + s)
                i1 += 1
                i2 += 1
            else:
                assert x1 == self._x[i1]
                self._x[i1] = x1
                self._v[i1] = vcurr + v1
                self._s[i1] = scurr + s
            for i in range(i1 + 1, i2):
                self._v[i] = self._v[i] + v1 + (self._x[i] - x1) * s
                self._s[i] += s
            if i1 == i2:
                self._x.insert(i1 + 1, x2)
                self._v.insert(i1 + 1, vcurr + (x2 - xcurr) * scurr)
                self._s.insert(i1 + 1, scurr)
                return
            elif x2 > self._x[i2]:
                # Adjust beginning of first segment
                xcurr = self._x[i2]
                vcurr = self._v[i2]
                scurr = self._s[i2]
                self._v[i2] = self._v[i2] + v1 + (self._x[i2] - x1) * s
                self._s[i2] += s
                self._x.insert(i2 + 1, x2)
                self._v.insert(i2 + 1, vcurr + (x2 - xcurr) * scurr)
                self._s.insert(i2 + 1, scurr)
                return

    def _add(self, other):
        ov0 = other._v0
        os0 = other._s0
        ol = len(other._x)
        if ol == 0:
            # 'other' is a constant function of value ov0
            self._v0 += ov0
            for i in range(len(self._x)):
                self._v[i] += ov0
        else:
            ox0 = other._x[0]
            if ox0 == self._x[0]:
                self._v0 += ov0
                self._s0 += os0
            elif ox0 < self._x[0]:
                self._v0 = (ox0 - self._x[0]) * self._s0
                self._x.insert(0, ox0)
                self._v.insert(0, self._v0)
                self._s.insert(0, self._s0)
                self._v0 += ov0
                self._s0 += os0
            else:
                # self._x[0] < ox0
                self._s0 += os0
                self._v0 += ov0 + (self._x[0] - ox0) * os0
                self.add_slope(self._x[0], ox0, ov0 + (self._x[0] - ox0) * os0, os0)
            for i in range(ol - 1):
                # This could be optimized to avoid calling add_slope
                self.add_slope(other._x[i], other._x[i + 1], other._v[i], other._s[i])
            self.add_slope(other._x[ol - 1], INTERVAL_MAX, other._v[ol - 1], other._s[ol - 1])
        return self

    def _add_number(self, k):
        self._v0 += k
        for i in range(len(self._x)):
            self._v[i] += k
        return self

    def _mul(self, k):
        self._s0 *= k
        self._v0 *= k
        for i in range(len(self._x)):
            self._v[i] *= k
            self._s[i] *= k
        return self

    def __iadd__(self, other):
        if isinstance(other, CpoFunction):
            assert (self.is_segmented_function or not other.is_segmented_function)
            return self._add(other)
        elif is_number(other) or is_bool(other):
            return self._add_number(other)

    def __isub__(self, other):
        if isinstance(other, CpoFunction):
            assert (self.is_segmented_function or not other.is_segmented_function)
            tmp = other.copy()._mul(-1)
            return self._add(tmp)
        elif is_number(other) or is_bool(other):
            return self._add_number(-other)

    def __add__(self, other):
        if isinstance(other, CpoFunction):
            if self.is_segmented_function:
                return other.copy(Type_SegmentedFunction)._add(self)
            else:
                return other.copy()._add(self)
        elif is_number(other) or is_bool(other):
            return self.copy()._add_number(other)

    def __radd__(self, value):
        return self.__add__(value)

    def __sub__(self, other):
        if isinstance(other, CpoFunction):
            if self.is_segmented_function:
                return other.copy(Type_SegmentedFunction)._mul(-1)._add(self)
            else:
                return other.copy()._mul(-1)._add(self)
        elif is_number(other) or is_bool(other):
            return self.copy()._add_number(-other)

    def __neg__(self):
        return self.copy()._mul(-1)

    def __imul__(self, value):
        return self._mul(value)

    def __mul__(self, value):
        return self.copy()._mul(value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def set_value(self, x1, x2, v):
        """ Sets the value of the function over an interval.

        Sets the value of the invoking function to be constant and equal to `v`
        on the interval `[x1,x2)`.

        Args:
            x1 (int): Start of the interval.
            x2 (int): End of the interval.
            v (int or float): Function value.
        """
        self.set_slope(x1, x2, v, 0)

    def add_value(self, x1, x2, v):
        """ Adds a constant to the value of the function over an interval.

        Adds constant value `v` to the value of the invoking function
        on the interval `[x1,x2)`.

        Args:
            x1 (int): Start of the interval.
            x2 (int): End of the interval.
            v (int or float): Added value.
        """
        self.add_slope(x1, x2, v, 0)

    def __pos__(self):
        return self


class CpoSegmentedFunction(CpoFunction):
    """ Class representing a segmented function.

    In CP Optimizer, piecewise linear functions are typically used in modeling a known function of time,
    for instance the cost that is incurred for completing an activity after a known date.

    A segmented function is a piecewise linear function defined on an interval [xmin, xmax)
    which is partitioned into segments such that over each segment, the function is linear.

    When two consecutive segments of the function are colinear, these segments are merged so that the function
    is always represented with the minimal number of segments.
    """

    def __init__(self, segment0=None, segments=None, name=None):
        """
        Args:
            segment0 (tuple): Initial segment of the function (slope, vright).
            segments (list): Segments of the function represented as a list of
                tuples (xleft, vleft, slope).
        """
        s0 = 0
        v0 = 0
        x = []
        v = []
        s = []
        if segment0 is not None:
            s0 = segment0[0]
            v0 = segment0[1]
        if segments is not None:
            for seg in segments:
                x.append(seg[0])
                v.append(seg[1])
                s.append(seg[2])
        super(CpoSegmentedFunction, self).__init__(Type_SegmentedFunction, s0=s0, v0=v0, x=x, v=v, s=s, name=name)

    def get_segment_list(self):
        """ Returns the list of segments of the function.
        """
        segments = [(self.s0, self.v0)]
        for i in range(len(self.x)):
            segments.append((self.x[i], self.v[i], self.s[i]))
        return segments

    def __str__(self):
        """ Build a string representing this function. """
        res = "SegmentedFunction(" + str(self.get_segment_list()) + ")"
        if self.name:
            res = self.name + "=" + res
        return res


class CpoStepFunction(CpoFunction):
    """ Class representing a step function.

    In CP Optimizer, stepwise functions are typically used to model the efficiency of a resource over time.

    A stepwise function is a special case of piecewise linear function where all slopes are equal to 0 and the
    domain and image of the function are integer.

    When two consecutive steps of the function have the same value, these steps are merged so that the function
    is always represented with the minimal number of steps.

    The function steps are expressed as a list of couples (x, val) specifying that the value of the function is *val*
    after *x*, up to the next step.
    By default, the value of the function is zero up to the first step.
    To change this default value, the first step should be (INTERVAL_MIN, value).
    """
    def __init__(self, steps=None, name=None):
        """ Step function.

        Args:
            steps: (Optional) Function steps, expressed as a list of couples (x, val).
            name:  (Optional) Function name
        """
        if steps is None or len(steps) == 0:
            super(CpoStepFunction, self).__init__(Type_StepFunction, name=name)
        else:
            x = []
            v = []
            if steps[0][0] == INTERVAL_MIN:
                v0 = steps[0][1]
            else:
                v0 = 0
                x.append(steps[0][0])
                v.append(steps[0][1])
            for s in steps[1:]:
                x.append(s[0])
                v.append(s[1])
            super(CpoStepFunction, self).__init__(typ=Type_StepFunction, s0=None, v0=v0, x=x, v=v, s=None, name=name)

    def get_step_list(self):
        """ Returns the list of steps of the function.
        """
        steps = []
        if self.v0 != 0:
            steps.append((INTERVAL_MIN, self.v0))
        for i in range(len(self.x)):
            steps.append((self.x[i], self.v[i]))
        return steps

    def __str__(self):
        """ Build a string representing this function. """
        res = "StepFunction(" + str(self.get_step_list()) + ")"
        if self.name:
            res = self.name + "=" + res
        return res


