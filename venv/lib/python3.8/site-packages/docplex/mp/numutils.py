# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

import math


def round_nearest_halfway_from_zero(x, infinity=1e+20):
    """ Rounds the argument to the nearest integer.

    For values like 1.5 the intetger with greater absolute value is returned.
    This treats positive and negative values in a symmetric manner.
    This is called "round half away from zero"


    Args:
        x: the value to round
        infinity: the model's infinity value. All values above infinity are set to +INF

    Returns:
        an integer value

    Example:
        round_nearest(0) = 0
        round_nearest(1.1) = 1
        round_nearest(1.5) = 2
        round_nearest(1.49) = 1
    """
    if x == 0:
        return 0
    elif x >= infinity:
        return infinity
    elif x <= -infinity:
        return -infinity
    else:
        raw_nearest = my_round_even(x)  # math.floor(x + 0.5)
        return int(raw_nearest)


def my_round_even(number):
    """
    Simplified version from future
    """
    from decimal import Decimal, ROUND_HALF_EVEN

    d = Decimal.from_float(number).quantize(1, rounding=ROUND_HALF_EVEN)
    return int(d)


def round_nearest_towards_infinity(x, infinity=1e+20):
    """ Rounds the argument to the nearest integer.

    For ties like 1.5 the ceiling integer is returned.
    This is called "round towards infinity"

    Args:
        x: the value to round
        infinity: the model's infinity value. All values above infinity are set to +INF

    Returns:
        an integer value

    Example:
        round_nearest(0) = 0
        round_nearest(1.1) = 1
        round_nearest(1.5) = 2
        round_nearest(1.49) = 1
    """
    if x == 0:
        return 0
    elif x >= infinity:
        return infinity
    elif x <= -infinity:
        return -infinity
    else:
        raw_nearest = math.floor(x + 0.5)
        return int(raw_nearest)

def round_nearest_towards_infinity1(x):
    return round_nearest_towards_infinity(x)

class _NumPrinter(object):
    """
    INTERNAL.
    """

    def __init__(self, nb_digits_for_floats, num_infinity=1e+20, pinf="+inf", ninf="-inf"):
        assert (nb_digits_for_floats >= 0)
        assert (isinstance(pinf, str))
        assert (isinstance(ninf, str))
        self.true_infinity = num_infinity
        self.precision = nb_digits_for_floats
        self.__positive_infinity = pinf
        self.__negative_infinity = ninf
        # coin the format from the nb of digits
        # 2 -> %.2f
        self._double_format = "%." + ('%df' % nb_digits_for_floats)

    def to_string(self, num):
        if num >= self.true_infinity:
            return self.__positive_infinity
        elif num <= - self.true_infinity:
            return self.__negative_infinity
        else:
            try:
                if num.is_integer():  # the is_integer() function is faster than testing: num == int(num)
                    return '%d' % num
                else:
                    return self._double_format % num
            except AttributeError:
                return '%d' % num
