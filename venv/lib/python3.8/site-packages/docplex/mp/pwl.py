# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from docplex.mp.utils import is_iterable
from docplex.mp.basic import ModelingObjectBase
from docplex.mp.utils import DOcplexException, is_number
from docplex.mp.compat23 import izip
from docplex.mp.sttck import StaticTypeChecker

import copy


class PwlFunction(ModelingObjectBase):
    """
    This class models piecewise linear (PWL) functions. This class is not intended to be instantiated:
    piecewise linear functions are defined by invoking :func:`docplex.mp.model.Model.piecewise`,
    or :func:`docplex.mp.model.Model.piecewise_as_slopes`.

    Piecewise-linear functions are important in many applications.
    They are often specified either:

    * by giving a set of slopes, a set of breakpoints at which the slopes change, and the value of the functions at
      a given point, or
    * by giving an ordered list of (x,y) points that are linearly connected, along with the slope before the first
      point and the slope after the last point.

    Note that a piecewise-linear function may be discontinuous.
    """

    @staticmethod
    def check_number(logger, arg, caller=None):
        StaticTypeChecker.typecheck_num_nan_inf(logger, arg, caller)


    @staticmethod
    def check_list_pair_breaksxy(logger, arg):
        if not is_iterable(arg):
            logger.fatal("argument 'breaksxy' expects iterable, {0!r} was passed".format(arg))
        if isinstance(arg, tuple):
            # Encapsulate tuple argument into a list: this allows defining a PWL with a tuple if there is only
            #  one element in its definition
            arg = [arg]
        if len(arg) == 0:
            logger.fatal("argument 'breaksxy' must be a non-empty list of (x, y) tuples.")
        prev_pair = None
        pprev_pair = None
        for pair in arg:
            if isinstance(pair, tuple):
                if len(pair) != 2:
                    logger.fatal("invalid tuple in 'breaksxy': {0!s}. Each tuple must have 2 items.".format(pair))
                PwlFunction.check_number(logger, pair[0])
                PwlFunction.check_number(logger, pair[1])
            else:
                logger.fatal("invalid item in 'breaksxy': {0!s}. Each item must be a (x, y) tuple.".format(pair))
            if prev_pair is not None:
                if pair[0] < prev_pair[0]:
                    logger.fatal("X coordinate in: {0!s} cannot be smaller than previous break abscisse: {1!s}.".
                                 format(pair, prev_pair))
                if pprev_pair is not None and pair[0] == prev_pair[0] and prev_pair[0] == pprev_pair[0]:
                    logger.fatal(
                        "invalid break: {0!s}. There cannot be more than 2 consecutive breaks with same abscisse.".
                            format(pair))
            pprev_pair = prev_pair
            prev_pair = pair

    @staticmethod
    def check_number_pair(logger, arg):
        if arg is None:
            logger.fatal("argument 'anchor' must be defined")
        if isinstance(arg, tuple):
            if len(arg) != 2:
                logger.fatal("invalid tuple for 'anchor': {0!s}. Anchor argument must have 2 items.".format(arg))
            PwlFunction.check_number(logger, arg[0])
            PwlFunction.check_number(logger, arg[1])
        else:
            logger.fatal("invalid value for 'anchor': {0!s}. Anchor argument must be a (x, y) tuple.".format(arg))

    @staticmethod
    def check_list_pair_slope_breakx(logger, arg, anchor):
        if arg is None:
            logger.fatal("argument 'slopebreaksx' must be defined")
        if not is_iterable(arg):
            logger.fatal("not an iterable: {0!s}".format(arg))
        if len(arg) == 0:
            return
        if isinstance(arg, tuple):
            # Encapsulate tuple argument into a list: this allows defining a PWL with a tuple if there is only
            #  one element in its definition
            arg = [arg]
        prev_pair = None
        pprev_pair = None
        for pair in arg:
            if isinstance(pair, tuple):
                if len(pair) != 2:
                    logger.fatal("invalid tuple in 'slopebreaksx': {0!s}. Each tuple must have 2 items.".
                                 format(pair))
                PwlFunction.check_number(logger, pair[0])
                PwlFunction.check_number(logger, pair[1])
            else:
                logger.fatal("invalid item in 'slopebreaksx': {0!s}. Each item must be a (x, y) tuple.".format(pair))
            if prev_pair is not None:
                if pair[1] < prev_pair[1]:
                    logger.fatal("X coordinate in: {0!s} cannot be smaller than previous break abscisse: {1!s}.".
                                 format(pair, prev_pair))
                if pprev_pair is not None and pair[1] == prev_pair[1] and prev_pair[1] == pprev_pair[1]:
                    logger.fatal(
                        "invalid break: {0!s}. There cannot be more than 2 consecutive breaks with same abscisse.".
                            format(pair))
                if pair[1] == prev_pair[1] and anchor[0] == pair[1]:
                    logger.fatal("anchor {0!s} cannot be defined at discontinuity point: {1!s}".
                                 format(anchor, pair))
            pprev_pair = prev_pair
            prev_pair = pair

    class _PwlAsBreaks:
        """
        When using this class, the piecewise linear function is specified by:
         - Breakpoints defined as a list of coordinate pairs `(x[i], y[i])` defining the segments of the PWL function.
         - Before the first segment of the PWL function there is a half-line; its slope is specified by `preslope`.
         - After the last segment of the the PWL function there is a half-line; its slope is specified by `postslope`.
        Two consecutive breakpoints may have the same x-coordinate; in such cases there is a discontinuity in the
        PWL function.  Three consecutive breakpoints may not have the same x-coordinate.
        """

        def __init__(self, preslope, breaksxy, postslope):
            self._preslope = preslope
            self._breaksxy = self._reformulate_breaksxy(breaksxy)
            self._postslope = postslope

        @property
        def preslope(self):
            return self._preslope

        @property
        def breaksxy(self):
            return self._breaksxy

        @property
        def postslope(self):
            return self._postslope

        def deepcopy(self):
            breaksxy_copy = copy.deepcopy(self.breaksxy)
            return PwlFunction._PwlAsBreaks(self.preslope, breaksxy_copy, self.postslope)

        @staticmethod
        def _reformulate_breaksxy(breaksxy):
            if isinstance(breaksxy, tuple):
                return [] if len(breaksxy) == 0 else [breaksxy]
            return breaksxy

        @staticmethod
        def _remove_useless_intermediate_breaks(preslope, breaksxy, postslope):
            result_breaksxy = []
            current_slope = preslope
            prev_break = None
            for br in breaksxy:
                if prev_break is None:
                    pass
                else:
                    if br[0] == prev_break[0]:
                        # Check discontinuity
                        if br[1] != prev_break[1]:
                            result_breaksxy.append(prev_break)
                            result_breaksxy.append(br)
                            current_slope = None
                    else:
                        slope = (br[1] - prev_break[1]) / (br[0] - prev_break[0])
                        if current_slope is not None and current_slope != slope:
                            # Add prev_break in list
                            result_breaksxy.append(prev_break)
                        current_slope = slope
                prev_break = br
            # Handle last break
            if not result_breaksxy:
                # Set result breaks = first break
                result_breaksxy = breaksxy[0]
            elif current_slope is not None and current_slope != postslope:
                result_breaksxy.append(prev_break)
            return preslope, result_breaksxy, postslope

        def _get_break_at_index(self, index):
            if len(self.breaksxy) <= index:
                return None, None, index
            break_1 = self.breaksxy[index]
            if len(self.breaksxy) > (index + 1):
                break_2 = self.breaksxy[index + 1]
                if break_1[0] == break_2[0]:
                    # Discontinuity
                    return break_1, break_2, index + 1
            return break_1, None, index

        def _get_y_value(self, x_coord, prev_break_index=-1):
            """
            :param x_coord:
            :param prev_break_index: this parameter is mandatory if a breakxy tuple does exist before x_coord. Otherwise
                an exception is raised.
            :return:
            """
            if prev_break_index < 0:
                break_1, break_2, last_ind = self._get_break_at_index(0)
                if break_1[0] < x_coord:
                    raise DOcplexException("Invalid arguments passed to PwlAsBreaks._get_y_value()")
                if break_1[0] == x_coord:
                    y_coord_1 = break_1[1]
                    y_coord_2 = None if break_2 is None else break_2[1]
                    return y_coord_1, y_coord_2, last_ind
                y_coord_1 = break_1[1] - self.preslope * (break_1[0] - x_coord)
                return y_coord_1, None, -1
            break_1, break_2, last_ind = self._get_break_at_index(prev_break_index)
            next_break_1, next_break_2, next_last_ind = self._get_break_at_index(last_ind + 1)
            if next_break_1 is None:
                # x-coord is after last break
                last_break = break_1 if break_2 is None else break_2
                y_coord_1 = last_break[1] + self.postslope * (x_coord - last_break[0])
                return y_coord_1, None, last_ind
            else:
                if x_coord == break_1[0]:
                    # Here, one must have: x_coord > break_1[0]
                    raise DOcplexException("Invalid arguments passed to PwlAsBreaks._get_y_value()")
                if x_coord == next_break_1[0]:
                    y_coord_1 = next_break_1[1]
                    y_coord_2 = None if next_break_2 is None else next_break_2[1]
                    return y_coord_1, y_coord_2, next_last_ind
                y_coord_prev = break_1[1] if break_2 is None else break_2[1]
                y_coord_next = next_break_1[1]
                slope = (y_coord_next - y_coord_prev) / (next_break_1[0] - break_1[0])
                y_coord_1 = y_coord_prev + slope * (x_coord - break_1[0])
                return y_coord_1, None, last_ind

        def evaluate(self, x_val):
            """ Evaluates the breaks-based PWL function at the point whose x-coordinate is `x_val`.

            Args:
                x_val: The x value for which we want to compute the value of the function.

            Returns:
                The value of the PWL function at point `x_val`
                A DOcplexException exception is raised when evaluating at a discontinuity of the PWL function.
            """
            prev_break_index, index = -1, 0
            while index < len(self.breaksxy):
                break_1, break_2, index = self._get_break_at_index(index)
                if break_1 is None:
                    raise DOcplexException("Invalid PWL definition: no break point is defined")
                if break_1[0] < x_val:
                    prev_break_index = index
                else:
                    if break_1[0] == x_val and break_2 is not None:
                        raise DOcplexException("Cannot evaluate PWL at a discontinuity")
                    break
                index += 1
            y_val, _, _ = self._get_y_value(x_val, prev_break_index)
            return y_val

        def _get_all_breaks(self, all_x_coord):
            all_breaks = []
            prev_break_ind = -1
            for x_coord in all_x_coord:
                y_coord_1, y_coord_2, prev_break_ind = self._get_y_value(x_coord, prev_break_ind)
                all_breaks.append((x_coord, y_coord_1) if y_coord_2 is None else
                                  [(x_coord, y_coord_1), (x_coord, y_coord_2)])
            return all_breaks

        def get_nb_intervals(self):
            nb_discontinuities = 0
            prev_br = None
            for br in iter(self.breaksxy):
                if prev_br is not None and prev_br[0] == br[0]:
                    nb_discontinuities += 1
                prev_br = br
            return len(self.breaksxy) - nb_discontinuities - 1

        def __add__(self, arg):
            if isinstance(arg, PwlFunction._PwlAsBreaks):
                all_x_coord = sorted({br[0] for br in self.breaksxy + arg.breaksxy})
                all_breaks_left = self._get_all_breaks(all_x_coord)
                all_breaks_right = arg._get_all_breaks(all_x_coord)
                result_breaksxy = []
                # Both lists have same size, with same x-coord for breaks ==> perform the addition on each break
                for br_l, br_r in izip(all_breaks_left, all_breaks_right):
                    if isinstance(br_l, tuple) and isinstance(br_r, tuple):
                        result_breaksxy.append((br_l[0], br_l[1] + br_r[1]))
                    else:
                        if isinstance(br_l, tuple):
                            # br_r is a list containing 2 tuple pairs
                            result_breaksxy.append((br_l[0], br_l[1] + br_r[0][1]))
                            result_breaksxy.append((br_l[0], br_l[1] + br_r[1][1]))
                        elif isinstance(br_r, tuple):
                            # br_l is a list containing 2 tuple pairs
                            result_breaksxy.append((br_r[0], br_l[0][1] + br_r[1]))
                            result_breaksxy.append((br_r[0], br_l[1][1] + br_r[1]))
                        else:
                            # br_l and br_r are two lists, each containing 2 tuple pairs
                            result_breaksxy.append((br_l[0][0], br_l[0][1] + br_r[0][1]))
                            result_breaksxy.append((br_l[0][0], br_l[1][1] + br_r[1][1]))
                result_preslope = self.preslope + arg.preslope
                result_postslope = self.postslope + arg.postslope
                return PwlFunction._PwlAsBreaks(*self._remove_useless_intermediate_breaks(
                    result_preslope, result_breaksxy, result_postslope))

            elif is_number(arg):
                return PwlFunction._PwlAsBreaks(
                    self.preslope, [(br[0], br[1] + arg) for br in self.breaksxy], self.postslope)

            else:
                raise DOcplexException("Invalid type for right hand side operand: {0!s}.".format(arg))

        def __sub__(self, arg):
            if isinstance(arg, PwlFunction._PwlAsBreaks):
                return self + arg * (-1)
            elif is_number(arg):
                return PwlFunction._PwlAsBreaks(
                    self.preslope, [(br[0], br[1] - arg) for br in self.breaksxy], self.postslope)
            else:
                raise DOcplexException("Invalid type for right hand side operand: {0!s}.".format(arg))

        def __mul__(self, arg):
            if is_number(arg):
                return PwlFunction._PwlAsBreaks(*self._remove_useless_intermediate_breaks(
                    self.preslope * arg, [(br[0], br[1] * arg) for br in self.breaksxy], self.postslope * arg))
            else:
                raise DOcplexException("Invalid type for right hand side operand: {0!s}.".format(arg))

        def translate(self, arg):
            if is_number(arg):
                return PwlFunction._PwlAsBreaks(
                    self.preslope, [(br[0] + arg, br[1]) for br in self.breaksxy], self.postslope)
            else:
                raise DOcplexException("Invalid type for argument: {0!s}.".format(arg))

        def __str__(self):
            return self.to_string()

        def to_string(self):
            return '({0}, {1}, {2})'.format(self.preslope, self.breaksxy, self.postslope)

        def repr_string(self):
            return 'preslope={0},breaksxy={1},postslope={2}'.format(self.preslope, self.breaksxy, self.postslope)

    class _PwlAsSlopes:
        """
        When using this class, the piecewise linear function is specified by:
         - a list of tuple pairs `(slope[i], breakx[i])` of slopes and x-coordinates defining the slope of the piecewise
           function between the previous breakpoint (or minus infinity if there is none) and the breakpoint with
           x-coordinate `breakx[i]`,
         - the slope after the last specified breakpoint, and
         - the coordinates of the 'anchor point'. The purpose of the anchor point is to ground the piecewise-linear
           function specified by the list of slopes and breakpoints.
        Note that:
         - The `breakx[i]` values must be increasing. If two consecutive `breakx` values have the same value, a
           discontinuity is defined and the value associated with the second argument is considered to be a "step".
         - The list of tuple pairs `(slope[i], breakx[i])` may be empty.
         - The default value for the anchor point is the origin (point with coordinates (0, 0)).
         - If the piecewise linear function defines some discontinuities, the anchor must not reside at one of
           these discontinuities, since the function would not be uniquely defined.
        """

        def __init__(self, slopebreaksx, lastslope, anchor=(0, 0)):
            self._slopebreaksx = self._reformulate_slopebreaksx(slopebreaksx)
            self._lastslope = lastslope
            self._anchor = anchor

        @property
        def slopebreaksx(self):
            return self._slopebreaksx

        @property
        def lastslope(self):
            return self._lastslope

        @property
        def anchor(self):
            return self._anchor

        def deepcopy(self):
            slopebreaksx_copy = copy.deepcopy(self.slopebreaksx)
            anchor_copy = copy.deepcopy(self.anchor)
            return PwlFunction._PwlAsSlopes(slopebreaksx_copy, self.lastslope, anchor_copy)

        @staticmethod
        def _reformulate_slopebreaksx(slopebreaksx):
            if isinstance(slopebreaksx, tuple):
                return [] if len(slopebreaksx) == 0 else [slopebreaksx]
            return slopebreaksx

        @staticmethod
        def _compute_breaksxy_after(slope_breaks, anchor):
            breaks_xy = []
            start_x, start_y = anchor[0], anchor[1]
            for (slope, break_x) in slope_breaks:
                delta_x = break_x - start_x
                start_x = break_x
                if delta_x > 0:
                    start_y = start_y + slope * delta_x
                else:
                    # Discontinuity: slope is considered to be a "step"
                    start_y += slope
                breaks_xy.append((start_x, start_y))
            return breaks_xy

        @staticmethod
        def _compute_breaksxy_before(start_slope, slope_breaks, anchor):
            breaks_xy = []
            start_x, start_y = anchor[0], anchor[1]
            last_slope = start_slope
            for (slope, break_x) in slope_breaks:
                delta_x = break_x - start_x
                start_x = break_x
                if delta_x < 0 or anchor[0] == break_x:
                    start_y = start_y + last_slope * delta_x
                else:
                    # Discontinuity: slope is considered to be a "step"
                    start_y -= last_slope
                last_slope = slope
                breaks_xy.append((start_x, start_y))
            return breaks_xy, last_slope

        def convert_to_pwl_as_breaks(self):
            breaks_before = [(s, b) for (s, b) in self.slopebreaksx if b <= self.anchor[0]]
            breaks_after = [(s, b) for (s, b) in self.slopebreaksx if b > self.anchor[0]]
            # Compute y value at each break point
            anchor_slope = breaks_after[0][0] if len(breaks_after) > 0 else self.lastslope
            breaks_before.reverse()
            breaks_xy_before, preslope = self._compute_breaksxy_before(anchor_slope, breaks_before, self.anchor)
            breaks_xy_before.reverse()
            breaks_xy_after = self._compute_breaksxy_after(breaks_after, self.anchor)
            # Now, we can build the PWL as breaks
            breaksxy = breaks_xy_before + breaks_xy_after
            if len(breaksxy) > 0:
                return PwlFunction._PwlAsBreaks(preslope, breaksxy, self.lastslope)
            else:
                # No breakpoint is defined
                return PwlFunction._PwlAsBreaks(self.lastslope, [self.anchor], self.lastslope)

        def _get_safe_xy_anchor(self):
            """
            Return an anchor point that is on or after (if last break corresponds to a discontinuity) the largest
            x-coord corresponding to a break or the anchor.
            :return:
            """
            breaks_after = [(s, b) for (s, b) in self.slopebreaksx if b > self.anchor[0]]
            breaks_xy_after = self._compute_breaksxy_after(breaks_after, self.anchor)
            if len(breaks_xy_after) > 0:
                # Check if last break corresponds to a discontinuity
                if len(breaks_xy_after) > 1 and breaks_xy_after[-2][0] == breaks_xy_after[-1][0]:
                    # Returns point with x-coord = last_x_coord + 1
                    return breaks_xy_after[-1][0] + 1, breaks_xy_after[-1][1] + self.lastslope
                return breaks_xy_after[-1]
            return self.anchor

        @staticmethod
        def _remove_useless_intermediate_slopes(slopebreaksx, lastslope, anchor):
            result_slopebreaksx = []
            prev_sbr = None
            for sbr in slopebreaksx:
                if prev_sbr is not None:
                    if sbr[0] != prev_sbr[0]:
                        result_slopebreaksx.append(prev_sbr)
                prev_sbr = sbr
            if prev_sbr is not None and prev_sbr[0] != lastslope:
                result_slopebreaksx.append(prev_sbr)
            return result_slopebreaksx, lastslope, anchor

        def _get_all_slopebreaks(self, all_x_coord):
            all_slopebreaks = []
            iter_slopebreakx = iter(self.slopebreaksx)
            current_slopebreakx = next(iter_slopebreakx, None)
            for x_coord in all_x_coord:
                if current_slopebreakx is None:
                    all_slopebreaks.append((self.lastslope, x_coord))
                else:
                    while current_slopebreakx is not None and x_coord > current_slopebreakx[1]:
                        prev_slopebreakx = current_slopebreakx
                        current_slopebreakx = next(iter_slopebreakx, None)
                        if current_slopebreakx is not None and current_slopebreakx[1] == prev_slopebreakx[1]:
                            # Case of a discontinuity ==> update last item in result list to a list containing 2 tuples
                            all_slopebreaks[-1] = [(prev_slopebreakx[0], prev_slopebreakx[1]),
                                                   (current_slopebreakx[0], current_slopebreakx[1])]
                    if current_slopebreakx is None:
                        all_slopebreaks.append((self.lastslope, x_coord))
                    else:
                        all_slopebreaks.append((current_slopebreakx[0], x_coord))
            # Handle case where last break is a discontinuity
            prev_slopebreakx = current_slopebreakx
            current_slopebreakx = next(iter_slopebreakx, None)
            if current_slopebreakx is not None:
                # Case of a discontinuity ==> update last item in result list to a list containing 2 tuples
                all_slopebreaks[-1] = [(prev_slopebreakx[0], prev_slopebreakx[1]),
                                       (current_slopebreakx[0], current_slopebreakx[1])]
            return all_slopebreaks

        def __add__(self, arg):
            if isinstance(arg, PwlFunction._PwlAsSlopes):
                all_x_coord = sorted({sbr[1] for sbr in self.slopebreaksx + arg.slopebreaksx})
                all_slopebreaks_left = self._get_all_slopebreaks(all_x_coord)
                all_slopebreaks_right = arg._get_all_slopebreaks(all_x_coord)
                result_slopebreaksxy = []
                # Both lists have same size, with same x-coord for slopebreaks
                #   ==> perform the addition of slopes on each break
                for sbr_l, sbr_r in izip(all_slopebreaks_left, all_slopebreaks_right):
                    if isinstance(sbr_l, tuple) and isinstance(sbr_r, tuple):
                        result_slopebreaksxy.append((sbr_l[0] + sbr_r[0], sbr_l[1]))
                    else:
                        if isinstance(sbr_l, tuple):
                            # sbr_r is a list containing 2 tuple pairs
                            result_slopebreaksxy.append((sbr_l[0] + sbr_r[0][0], sbr_l[1]))
                            result_slopebreaksxy.append((sbr_r[1][0], sbr_l[1]))
                        elif isinstance(sbr_r, tuple):
                            # sbr_l is a list containing 2 tuple pairs
                            result_slopebreaksxy.append((sbr_l[0][0] + sbr_r[0], sbr_r[1]))
                            result_slopebreaksxy.append((sbr_l[1][0], sbr_r[1]))
                        else:
                            # sbr_l and sbr_r are two lists, each containing 2 tuple pairs
                            result_slopebreaksxy.append((sbr_l[0][0] + sbr_r[0][0], sbr_l[0][1]))
                            result_slopebreaksxy.append((sbr_l[1][0] + sbr_r[1][0], sbr_l[0][1]))
                result_lastslope = self.lastslope + arg.lastslope

                if self.anchor[0] == arg.anchor[0]:
                    result_anchor = (self.anchor[0], self.anchor[1] + arg.anchor[1])
                else:
                    # Compute a new anchor based on the last x-coord in the slopebreakx list + anchor point
                    anchor_l = self._get_safe_xy_anchor()
                    anchor_r = arg._get_safe_xy_anchor()
                    delta = anchor_r[0] - anchor_l[0]
                    if anchor_l[0] < anchor_r[0]:
                        result_anchor = (anchor_r[0], anchor_l[1] + anchor_r[1] + delta * self.lastslope)
                    else:
                        result_anchor = (anchor_l[0], anchor_l[1] + anchor_r[1] - delta * arg.lastslope)

                return PwlFunction._PwlAsSlopes(*self._remove_useless_intermediate_slopes(
                    result_slopebreaksxy, result_lastslope, result_anchor))

            elif is_number(arg):
                return PwlFunction._PwlAsSlopes(copy.deepcopy(self.slopebreaksx),
                                                self.lastslope, (self.anchor[0], self.anchor[1] + arg))
            else:
                raise DOcplexException("Invalid type for right hand side operand: {0!s}.".format(arg))

        def __sub__(self, arg):
            if isinstance(arg, PwlFunction._PwlAsSlopes):
                return self + arg * (-1)
            elif is_number(arg):
                return PwlFunction._PwlAsSlopes(copy.deepcopy(self.slopebreaksx),
                                                self.lastslope, (self.anchor[0], self.anchor[1] - arg))
            else:
                raise DOcplexException("Invalid type for right hand side operand: {0!s}.".format(arg))

        def __mul__(self, arg):
            if is_number(arg):
                return PwlFunction._PwlAsSlopes(*self._remove_useless_intermediate_slopes(
                    [(br[0] * arg, br[1]) for br in self.slopebreaksx],
                    self.lastslope * arg, (self.anchor[0], self.anchor[1] * arg)))
            else:
                raise DOcplexException("Invalid type for right hand side operand: {0!s}.".format(arg))

        def translate(self, arg):
            if is_number(arg):
                return PwlFunction._PwlAsSlopes(
                    [(br[0], br[1] + arg) for br in self.slopebreaksx], self.lastslope,
                    (self.anchor[0] + arg, self.anchor[1]))
            else:
                raise DOcplexException("Invalid type for argument: {0!s}.".format(arg))

        def __str__(self):
            return self.to_string()

        def to_string(self):
            return '{' + ''.join(
                repr(slope) + ' -> ' + repr(break_x) + ';' for (slope, break_x) in self._slopebreaksx) + \
                   repr(self._lastslope) + '}(' + repr(self.anchor[0]) + ', ' + repr(self.anchor[1]) + ')'

    # _name_generator = _AutomaticSymbolGenerator(pattern="pwl", offset=1)

    def __init__(self, model, pwl_def, name=None):
        ModelingObjectBase.__init__(self, model, name=name)
        self._pwl_def = pwl_def
        self._pwl_def_as_breaks = None
        self._set_pwl_definition(pwl_def)

    def _set_pwl_definition(self, pwl_def):
        # INTERNAL
        if isinstance(pwl_def, PwlFunction._PwlAsBreaks):
            # Use the same data structure as input for internal representation (do not duplicate)
            self._pwl_def_as_breaks = pwl_def
        elif isinstance(pwl_def, PwlFunction._PwlAsSlopes):
            pwl_def_as_breaks = pwl_def.convert_to_pwl_as_breaks()
            self._set_pwl_as_breaks(pwl_def_as_breaks.preslope, pwl_def_as_breaks.breaksxy,
                                    pwl_def_as_breaks.postslope)
        else:
            self.model._checker.fatal("Invalid definition for Piecewise Linear Function: {0!s}.".format(pwl_def))

    def _set_pwl_as_breaks(self, preslope, breaksxy=None, postslope=None):
        """Internal format to represent a piecewise linear function is based on the Cplex representation"""
        self._pwl_def_as_breaks = self._PwlAsBreaks(preslope, breaksxy, postslope)

    def copy(self, target_model, _):
        pwl_def_copy = self.pwl_def.deepcopy()
        return target_model._piecewise(pwl_def_copy, self.get_name())

    @property
    def pwl_def(self):
        return self._pwl_def

    @property
    def pwl_def_as_breaks(self):
        return self._pwl_def_as_breaks

    # __call__ builds an expression equal to the piecewise linear value of its argument, based
    # on the definition of the PWL function.
    #
    # Args:
    #     e: Accepts any object that can be transformed into an expression:
    #         decision variables, expressions, or numbers.
    #
    # Returns:
    #     An expression that can be used in arithmetic operators and constraints.
    #
    # Note:
    #     Building the expression generates one auxiliary decision variable.
    def __call__(self, e):
        self.model._checker.typecheck_operand(e, caller="Model.pwl", accept_numbers=True)
        return self.model._add_pwl_expr(self, e)

    def __hash__(self):
        return id(self)

    def __str__(self):
        return self.pwl_def.__str__()

    def __repr__(self):
        return 'docplex.mp.pwl.PwlFunction({0})'.format(self.pwl_def_as_breaks.repr_string())

    def clone(self):
        """ Creates a copy of the PWL function on the same model.

        Returns:
            The copy of the PWL function.
        """
        return self.copy(self.model, None)

    def __add__(self, e):
        return self.plus(e)

    def __radd__(self, e):
        return self.plus(e)

    def __iadd__(self, e):
        self.model.fatal('Cannot modify a PWL function')
        # self.add(e)
        # return self

    def plus(self, e):
        cloned = self.clone()
        return cloned.add(e)

    def add(self, arg):
        """ Adds an expression to self.

        Note:
            This method does not create a new PWL function but modifies the `self` instance.

        Args:
            arg: The expression to be added. Can be a PWL function or a number.

        Returns:
            The modified self.
        """
        if isinstance(arg, PwlFunction):
            if (isinstance(self.pwl_def, PwlFunction._PwlAsBreaks) and
                    isinstance(arg.pwl_def, PwlFunction._PwlAsBreaks)) or \
                    (isinstance(self.pwl_def, PwlFunction._PwlAsSlopes) and
                         isinstance(arg.pwl_def, PwlFunction._PwlAsSlopes)):
                self._pwl_def = self.pwl_def + arg.pwl_def
                self._set_pwl_definition(self._pwl_def)
            else:
                # Use Breaks representation
                self._pwl_def = self.pwl_def_as_breaks + arg.pwl_def_as_breaks
                self._set_pwl_definition(self._pwl_def)
        elif is_number(arg):
            self._pwl_def = self.pwl_def + arg
            self._set_pwl_definition(self._pwl_def)
        else:
            raise DOcplexException("Invalid type for right hand side operand: {0!s}.".format(arg))
        return self

    def __sub__(self, e):
        return self.minus(e)

    def __rsub__(self, e):
        return self * (-1) + e

    def __isub__(self, e):
        self.model.fatal('Cannot modify a PWL function')

    def minus(self, e):
        cloned = self.clone()
        return cloned.subtract(e)

    def subtract(self, arg):
        """ Subtracts an expression from this PWL function.

        Note:
            This method does not create a new function but modifies the `self` instance.

        Args:
            arg: The expression to be subtracted. Can be either a PWL function, or a number.

        Returns:
            The modified self.
        """
        if isinstance(arg, PwlFunction):
            if (isinstance(self.pwl_def, PwlFunction._PwlAsBreaks) and
                    isinstance(arg.pwl_def, PwlFunction._PwlAsBreaks)) or \
                    (isinstance(self.pwl_def, PwlFunction._PwlAsSlopes) and
                         isinstance(arg.pwl_def, PwlFunction._PwlAsSlopes)):
                self._pwl_def = self.pwl_def - arg.pwl_def
                self._set_pwl_definition(self._pwl_def)
            else:
                # Use Breaks representation
                self._pwl_def = self.pwl_def_as_breaks - arg.pwl_def_as_breaks
                self._set_pwl_definition(self._pwl_def)
        elif is_number(arg):
            self._pwl_def = self.pwl_def - arg
            self._set_pwl_definition(self._pwl_def)
        else:
            raise DOcplexException("Invalid type for right hand side operand: {0!s}.".format(arg))
        return self

    def __mul__(self, e):
        return self.times(e)

    def __rmul__(self, e):
        return self.times(e)

    def __imul__(self, e):
        self.model.fatal('Cannot modify a PWL function')
        # return self.multiply(e)

    def __div__(self, e):
        return self.quotient(e)

    def __truediv__(self, e):
        # for py3
        # INTERNAL
        return self.quotient(e)  # pragma: no cover

    def __itruediv__(self, e):
        # for py3
        # INTERNAL
        self.model.fatal('Cannot modify a PWL function')  # pragma: no cover

    def __idiv__(self, other):
        self.model.fatal('Cannot modify a PWL function')

    def __rtruediv__(self, e):
        # for py3
        self.fatal("PWL function {0!s} cannot be used as denominator of {1!s}", self, e)  # pragma: no cover

    def __rdiv__(self, e):
        self.fatal("PWL function {0!s} cannot be used as denominator of {1!s}", self, e)

    def quotient(self, e):
        cloned = self.clone()
        cloned.divide(e)
        return cloned

    def divide(self, arg):
        """ Divides this PWL function by a number.

        Note:
            This method does not create a new function but modifies the `self` instance.

        Args:
            arg: The number that is used to divide `self`.

        Returns:
            The modified `self`.
        """
        self.model._typecheck_as_denominator(arg, numerator=self)
        inverse = 1.0 / float(arg)
        return self.multiply(inverse)

    def times(self, e):
        cloned = self.clone()
        return cloned.multiply(e)

    def multiply(self, arg):
        """ Multiplies this PWL function by a number.

        Note:
            This method does not create a new function but modifies the `self` instance.

        Args:
            arg: The number that is used to multiply `self`.

        Returns:
            The modified `self`.
        """
        if is_number(arg):
            self._pwl_def = self.pwl_def * arg
            self._set_pwl_definition(self._pwl_def)
        else:
            raise DOcplexException("Invalid type for right hand side operand: {0!s}.".format(arg))
        return self

    def translate(self, arg):
        """ Translate this PWL function by a number.
        This method creates a new PWL function instance for which all breakpoints have been moved
        along the horizontal axis by the amount specified by `arg`.

        Args:
            arg: The number that is used to translate all breakpoints.

        Returns:
            The translated PWL function.
        """
        if is_number(arg):
            return PwlFunction(self.model, self.pwl_def.translate(arg))
        else:
            raise DOcplexException("Invalid type for argument: {0!s}.".format(arg))

    def evaluate(self, x_val):
        """ Evaluates the PWL function at the point whose x-coordinate is `x_val`.

        Args:
            x_val: The x value for which we want to compute the value of the function.

        Returns:
            The value of the PWL function at point `x_val`.
            A DOcplexException exception is raised when evaluating at a discontinuity of the PWL function.
        """
        return self._pwl_def_as_breaks.evaluate(x_val)

    def plot(self, lx=None, rx=None, k=1, **kwargs):  # pragma: no cover
        """
        This method displays the piecewise linear function using the matplotlib package, if found.

        :param lx: The value to show the `preslope` (must be before the first breakpoint x value).
        :param rx: The value to show the `postslope` (must be after the last breakpoint x value).
        :param k: Scaling factor to calculate default values for `rx` and/or `lx` if these arguments are not provided,
                based on mean interval length between the `x` values of breakpoints.
        :param kwargs: additional arguments to be passed to matplotlib plot() function
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise DOcplexException('matplotlib is required for plot()')
        bks = self.pwl_def_as_breaks.breaksxy
        xs = [bk[0] for bk in bks]
        ys = [bk[1] for bk in bks]
        # compute mean delta_x
        first_x = xs[0]
        last_x = xs[-1]
        nb_intervals = self._pwl_def_as_breaks.get_nb_intervals()
        # k times the mean interval length is used for left/right extra points
        kdx_m = k * (last_x - first_x) / float(nb_intervals) if nb_intervals > 0 else 1

        if lx is None:
            lx = first_x - kdx_m
        ly = ys[0] - self.pwl_def_as_breaks.preslope * (first_x - lx)
        xs.insert(0, lx)
        ys.insert(0, ly)

        if rx is None or rx <= last_x:
            rx = last_x + kdx_m
        ry = ys[-1] + self.pwl_def_as_breaks.postslope * (rx - last_x)
        xs.append(rx)
        ys.append(ry)

        if plt:
            plt.plot(xs, ys, **kwargs)
            if self.name:
                plt.title('pwl: {0}'.format(self.name))
            plt.show()
