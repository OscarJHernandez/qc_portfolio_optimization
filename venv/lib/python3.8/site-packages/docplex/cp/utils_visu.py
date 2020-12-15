# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Philippe Laborie, IBM Analytics, France Lab Gentilly

"""
An add-on to *matplotlib* for easy display of CP Optimizer scheduling solutions and structures.

It requires to have external libraries *numpy* and *matplotlib* already installed in the Python environment.

Detailed description
--------------------
"""

import heapq
import math

# Import required external libraries (numpy and matplotlib)
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.collections import LineCollection
    LIBRARIES_PRESENT = True
except ImportError:
    LIBRARIES_PRESENT = False

from docplex.cp.function import CpoFunction
from docplex.cp.expression import CpoTransitionMatrix
from docplex.cp.solution import *
import docplex.cp.config as config

def is_visu_enabled():
    """ Check if visu is enabled and print a message if not.

    Returns:
        True if visu is enabled
    """
    if not LIBRARIES_PRESENT:
        print("\nVisu is disabled because packages 'numpy' and/or 'matplotlib' are not available.")
        return False
    if not config.context.visu_enabled:
        print("\nVisu is logically disabled by configuration.")
        return False
    return True


# Timeline objects
class _TLObject(object):
    __slots__ = ('_tl',       # Timeline
                 '_name',     # Name
                 '_origin',   # Origin
                 '_horizon',  # Horizon
                 '_color',    # Color identifier
                 )

    def __init__(self, tl, name=None, origin=None, horizon=None, color=None):
        """ New timeline object
        """
        super(_TLObject, self).__init__()
        self._tl = tl
        self._origin = origin
        self._horizon = horizon
        self._color = color
        self._name = name
        self._tl.record(self)

    @property
    def timeline(self):
        return self._tl

    @property
    def name(self):
        return self._name

    @property
    def origin(self):
        return self._origin

    @property
    def horizon(self):
        return self._horizon

    @property
    def color(self):
        return self._color

    @property
    def tl_origin(self):
        return self._tl.origin

    @property
    def tl_horizon(self):
        return self._tl.horizon

    @property
    def tl_mincolor(self):
        return self._tl.mincolor

    @property
    def tl_maxcolor(self):
        return self._tl.maxcolor


# Internal classes

class _Interval(_TLObject):
    def __init__(self, tl, start, end, color=0, name=None):
        """ Create a new interval
        """
        super(_Interval, self).__init__(tl, name, start, end, color)
        if _visu._naming is not None:
            self._name = _visu._naming(name)
        else:
            self._name = name

    @property
    def start(self):
        return self.origin

    @property
    def end(self):
        return self.horizon


class _Segment(_TLObject):
    def __init__(self, tl, start, end, vstart, vend, name=None):
        """ Create a new interval
        """
        super(_Segment, self).__init__(tl, name=name)
        if _visu._naming is not None:
            self._name = _visu._naming(name)
        else:
            self._name = name
        self._start = start
        self._end = end
        self._vstart = vstart
        self._vend = vend

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def vstart(self):
        return self._vstart

    @property
    def vend(self):
        return self._vend


class _Sequence(_TLObject):
    def __init__(self, tl, name=None):
        """ Create a new empty sequence
        """
        super(_Sequence, self).__init__(tl, name)
        self._intervals = []  # Intervals
        self._segments = []  # Segments
        self._position = -1

    def set_position(self, p):
        self._position = p

    def get_position(self):
        return self._position

    def add_interval(self, itv):
        self._intervals.append(itv)

    def add_segment(self, seg):
        self._segments.append(seg)

    @property
    def intervals(self):
        return self._intervals

    @property
    def segments(self):
        return self._segments


class _Function(_TLObject):
    def __init__(self, tl, name=None, origin=None, horizon=None, style=None, color=None):
        """ Create a new empty sequence
        """
        super(_Function, self).__init__(tl, name, origin, horizon, color)
        self._segments = []  # Segments
        self._style = style
        # Segments are 4-tuples (start, end, vstart, vend), with the exception at the extremities:
        # when start=-infty, and end<+infty, vstart is a slope before end
        # when end  = +infty and start>-infty, vend is a slope after start

    @property
    def segments(self):
        return self._segments

    @property
    def style(self):
        return self._style

    def add_segment(self, seg):
        self._segments.append(seg)


class _Panel(_TLObject):
    def __init__(self, tl, name=None, origin=None, horizon=None, pauses=None):
        """ Create a new panel
        """
        super(_Panel, self).__init__(tl, name, origin, horizon)
        if pauses is None:
            self._pauses = []
        else:
            self._pauses = pauses

    @property
    def pauses(self):
        return self._pauses

    def preshow(self):
        pass

    def get_height(self):
        return 1

    def add_pause(self, start, end):
        self._pauses.append((start, end))

    def display(self, axes):
        """ axes is an instance of Axes
        """

    def show_pauses(self, axes, ymin, ymax):
        all_pauses = self.timeline.pauses + self.pauses
        origin = self.timeline.origin
        horizon = self.timeline.horizon
        for p in all_pauses:
            if origin < p[1] and p[0] < horizon:
                left = max(p[0], origin)
                right = min(p[1], horizon)
                poly = Polygon([(left, ymin), (left, ymax), (right, ymax), (right, ymin)], facecolor='0.94',
                               edgecolor='0.94')
                axes.add_patch(poly)


class _SequencePanel(_Panel):
    def __init__(self, tl, name=None, origin=None, horizon=None, pauses=None):
        """ Create a new empty sequence panel
        """
        super(_SequencePanel, self).__init__(tl, name, origin, horizon, pauses)
        self.sequences = []  # Sequences

    def get_height(self):
        return len(self.sequences) + 1.5

    def add_sequence(self, seq):
        seq.set_position(len(self.sequences))
        self.sequences.append(seq)

    def display(self, axes):
        """ axes is an instance of Axes
        """
        n = len(self.sequences)
        tl = self.timeline
        iw = tl.intervalWidth
        tw = tl.transitionWidth
        origin = tl.origin
        horizon = tl.horizon
        step = tl.timeStep
        cpieces = tl.cpieces
        colormap = self.timeline.seqcmap
        self.show_pauses(axes, 0, n + 1)
        for x in np.arange(origin, horizon, step):
            axes.vlines(x, 0, n + 1, colors="lightgray", linestyle="dotted", lw=1)
        for i in range(0, n):
            axes.hlines(n - i, origin, horizon, colors="lightgray", linestyle="dotted", lw=1)
        axes.set_yticks(list(reversed(range(0, n + 2))))
        names = [""] + [s.name for s in self.sequences] + [""]
        axes.set_yticklabels(names)
        for seq in self.sequences:
            for itv in seq.intervals:
                if is_int(itv.color):
                    if 0 <= itv.color:
                        axes.plot([itv.start, itv.end], [n - seq.get_position(), n - seq.get_position()], marker='|',
                                  markeredgecolor=(0.52, 0.52, 0.52), markersize=iw, linestyle='')
                        axes.hlines(n - seq.get_position(), itv.start, itv.end, colors=tl.get_color(itv.color), lw=iw)
                    else:
                        axes.hlines(n - seq.get_position(), itv.start, itv.end, colors=tl.get_color(itv.color), lw=tw)
                else:
                    axes.plot([itv.start, itv.end], [n - seq.get_position(), n - seq.get_position()], marker='|',
                                  markeredgecolor=(0.52, 0.52, 0.52), markersize=iw, linestyle='')
                    axes.hlines(n - seq.get_position(), itv.start, itv.end, colors=tl.get_color(itv.color), lw=iw)

                if itv.name is not None:
                    axes.text(float(itv.start + itv.end) / 2, n - seq.get_position(), itv.name,
                              horizontalalignment='center', verticalalignment='center')
            fymin = INT_MAX
            fymax = INT_MIN
            for seg in seq.segments:
                start = seg.start
                end = seg.end
                if origin < end and start < horizon:
                    left = max(start, origin)
                    right = min(end, horizon)
                    vleft = _segment_value(seg, left)
                    vright = _segment_value(seg, right)
                    fymin = min(fymin, vleft, vright)
                    fymax = max(fymax, vleft, vright)
            for seg in seq.segments:
                start = seg.start
                end = seg.end
                if origin < end and start < horizon:
                    left = max(start, origin)
                    right = min(end, horizon)
                    vleft = _segment_value(seg, left)
                    vright = _segment_value(seg, right)
                    if vleft == vright:
                        if fymin < fymax:
                            nval = (vleft - fymin) / float(fymax - fymin)
                        else:
                            nval = 0
                        axes.hlines(n - seq.get_position(), left, right, colors=colormap(nval), lw=iw)
                    else:
                        x = np.linspace(left, right, cpieces)
                        y = np.full(cpieces, n - seq.get_position())
                        z = (vleft + (x - left) * (vright - vleft) / (right - left))
                        points = np.array([x, y]).T.reshape(-1, 1, 2)
                        segs = np.concatenate([points[:-1], points[1:]], axis=1)
                        lc = LineCollection(segs, cmap=colormap, norm=plt.Normalize(fymin, fymax))
                        lc.set_array(z)
                        lc.set_linewidth(iw)
                        axes.add_collection(lc)
                    axes.plot([left, right], [n - seq.get_position(), n - seq.get_position()], marker='|',
                              markeredgecolor=(0.52, 0.52, 0.52), markersize=iw, linestyle='')
                    if seg.name is not None:
                        axes.text(float(left + right) / 2, n - seq.get_position(), seg.name,
                                  horizontalalignment='center', verticalalignment='center')


class _IntervalPanel(_Panel):
    def __init__(self, tl, name=None, origin=None, horizon=None, pauses=None):
        """ Create a new empty sequence
        """
        super(_IntervalPanel, self).__init__(tl, name, origin, horizon, pauses)
        self.intervals = []  # Intervals
        self.layered = []  # Intervals augmented with layer index
        self.nblayers = -1

    def add_interval(self, itv):
        self.intervals.append(itv)

    def get_height(self):
        return self.nblayers + 1.5

    def preshow(self):
        events = sorted([(itv.start, +1) for itv in self.intervals] + [(itv.end, -1) for itv in self.intervals])
        n = l = 0
        for e in events:
            l += e[1]
            if l > n:
                n = l
        self.nblayers = n
        intervals = sorted(self.intervals, key=lambda x: x.start)
        smin = intervals[0].start
        h = []
        for lvl in range(0, n):
            heapq.heappush(h, (smin, lvl))
        for itv in intervals:
            lvl = heapq.heappop(h)[1]
            self.layered.append((itv.start, itv.end, itv.color, itv.name, lvl))
            heapq.heappush(h, (itv.end, lvl))

    def display(self, axes):
        """
        axes is an instance of Axes
        """
        tl = self.timeline
        iw = tl.intervalWidth
        origin = tl.origin
        horizon = tl.horizon
        step = tl.timeStep
        n = self.nblayers
        self.show_pauses(axes, 0, n + 1)
        for x in np.arange(origin, horizon, step):
            axes.vlines(x, 0, n + 1, colors="lightgray", linestyle="dotted", lw=1)
        axes.set_yticks([])
        axes.set_yticklabels([])
        # if self.name is not None:
        #    axes.set_ylabel(self.name)
        for itv in self.layered:
            axes.plot([itv[0], itv[1]], [n - itv[4], n - itv[4]], marker='|', markeredgecolor=(0.52, 0.52, 0.52),
                      markersize=iw, linestyle='')
            axes.hlines(n - itv[4], itv[0], itv[1], colors=tl.get_color(itv[2]), lw=iw)
            axes.text(float(itv[0] + itv[1]) / 2, n - itv[4], itv[3], horizontalalignment='center',
                      verticalalignment='center')


def _segment_value(seg, x):
    """
    Returns segment value at a particular x-value
    """
    start = seg.start
    end = seg.end
    vstart = seg.vstart
    vend = seg.vend
    assert start <= x <= end, "Illegal attempt to compute value outside of segment"
    if start == INT_MIN and end < INT_MAX:
        # initial segment
        slope = vstart
        return vend - (end - x) * slope
    elif end == INT_MAX and start > INT_MIN:
        # last segment
        slope = vend
        return vstart + (x - start) * slope
    else:
        if vstart == vend:
            return vstart
        else:
            return vstart + (x - start) * (vend - vstart) / (end - start)


class _FunctionPanel(_Panel):
    def __init__(self, tl, name=None, origin=None, horizon=None, pauses=None):
        """ Create a new empty sequence panel
        """
        super(_FunctionPanel, self).__init__(tl, name, origin, horizon, pauses)
        self.functions = []  # Functions

    def add_function(self, f):
        self.functions.append(f)

    def get_height(self):
        return 4

    def display(self, axes):
        """ axes is an instance of Axes
        """
        tl = self.timeline
        origin = tl.origin
        horizon = tl.horizon
        step = tl.timeStep
        iw = tl.intervalWidth
        # ymin = min(self.y)
        # ymax = max(self.y)
        ymin = INT_MAX
        ymax = INT_MIN
        ticks = set()
        for f in self.functions:
            fymin = INT_MAX
            fymax = INT_MIN
            for s in f.segments:
                start = s.start
                end = s.end
                if origin < end and start < horizon:
                    left = max(start, origin)
                    right = min(end, horizon)
                    vleft = _segment_value(s, left)
                    vright = _segment_value(s, right)
                    fymin = min(fymin, vleft, vright)
                    fymax = max(fymax, vleft, vright)
            ticks.add(fymin)
            ticks.add(fymax)
            ymin = min(ymin, fymin)
            ymax = max(ymax, fymax)
        if ymin < 0 < ymax:
            ticks.add(0)
        rng = ymax - ymin
        delta = rng * 0.20
        self.show_pauses(axes, ymin - delta, ymax + delta)
        axes.set_yticks(list(ticks))
        axes.yaxis.grid(True)
        # For legends
        handles = []
        labels = []
        # Display each function
        single_function = (len(self.functions) == 1)
        for f in self.functions:
            if f.color is None:
                color = 'darkred'
            elif is_int(f.color):
                color = tl.get_color(f.color)
            else:
                color = f.color
            if f.style is None:
                style = 'segment'
            else:
                style = f.style
            prevv = None
            prevx = None

            # Patch for legend
            label = f.name
            if single_function and label == self.name:
                # No need to repeat the panel name in the legend
                label = None
            if label is not None and style == 'area':
                r = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                              alpha=0.9)  # creates rectangle patch for legend use.
                r.set_label(label)
                handles.append(r)
                labels.append(label)
                label = None

            for s in f.segments:
                start = s.start
                end = s.end
                if origin < end and start < horizon:
                    left = max(start, origin)
                    right = min(end, horizon)
                    vleft = _segment_value(s, left)
                    vright = _segment_value(s, right)
                    if style == 'area':
                        axes.fill_between([left, right], ymin, [vleft, vright], facecolor=color, edgecolor=color,
                                          alpha=0.9)
                    elif style == 'interval':
                        axes.plot([left, right], [vleft, vright], marker='|', markeredgecolor=(0.52, 0.52, 0.52),
                                  markersize=iw, linestyle='')
                        h, = axes.plot([left, right], [vleft, vright], solid_capstyle='butt', color=color, lw=iw,
                                       label=label)
                        if s.name is not None:
                            axes.text(float(left + right) / 2, float(vleft + vright) / 2, s.name,
                                      horizontalalignment='center',
                                      verticalalignment='center')
                        if label is not None:
                            handles.append(h)
                            labels.append(label)
                            label = None
                    else:
                        h, = axes.plot([left, right], [vleft, vright], color=color, lw=1.5, label=label)
                        if label is not None:
                            handles.append(h)
                            labels.append(label)
                            label = None
                    if left == prevx and prevv is not None:
                        if style == 'segment':
                            axes.plot([left, left], [prevv, vleft], linestyle=':', color=color, lw=1.5)
                        elif style == 'line':
                            axes.plot([left, left], [prevv, vleft], color=color, lw=1.5)
                    if style == 'segment':
                        if left == start:
                            axes.plot([left], [vleft], marker='o', markerfacecolor=color, markeredgecolor=color,
                                      markersize=5)
                        if right == end:
                            axes.plot([right], [vright], marker='o', fillstyle=u'none', markerfacecolor='white',
                                      markeredgecolor=color, markersize=5)
                    prevv = vright
                    prevx = right

        #
        if self.name is not None:
            axes.set_ylabel(self.name)
        for vx in np.arange(origin, horizon, step):
            axes.vlines(vx, ymin - delta, ymax + delta, colors="lightgray", linestyle="dotted", lw=1)
        axes.set_xlim(xmin=origin, xmax=horizon)
        if 0 < len(handles):
            axes.legend(handles, labels, loc='best', fancybox=True, framealpha=0.5)


class _Figure(object):
    def __init__(self, name=None):
        super(_Figure, self).__init__()
        self.name = name
        self.seqcmap = cm.BuPu  # Sequential colormap
        self.quacmap = cm.Set2  # Qualitative colormap

    def show(self):
        assert False


class _Matrix(_Figure):
    def __init__(self, title=None, matrix=None, tuples=None):
        super(_Matrix, self).__init__(title)
        if matrix is not None:
            assert isinstance(matrix[0], list), "Bad input format for matrix"
            assert len(matrix) == len(matrix[0]), "Input for matrix is not square"
            self._size = len(matrix[0])
            self._values = np.array(matrix, dtype=np.int)
        elif tuples is not None:
            self._size = max(t[i] for t in tuples for i in range(2)) + 1
            self._values = np.zeros((self._size, self._size), dtype=np.int)
            for t in tuples:
                self._values[t[0]][t[1]] = t[2]

    def flush(self):
        pass

    def show(self):
        vmin = INT_MAX
        vmax = INT_MIN
        ivmin = -1
        jvmin = -1
        ivmax = -1
        jvmax = -1
        # for vmin annotation we target the center of the matrix
        ivmintarget = self._size * 0.5
        jvmintarget = self._size * 0.5
        distmin = INT_MAX
        # for vmax annotation we target the center of the matrix
        ivmaxtarget = self._size * 0.75
        jvmaxtarget = self._size * 0.5
        distmax = INT_MAX
        for i in range(0, self._size):
            for j in range(0, self._size):
                val = self._values[i][j]
                if val < INTERVAL_MAX:
                    if val <= vmin:
                        if val < vmin:
                            vmin = val
                            ivmin = i
                            jvmin = j
                            distmin = math.pow(i - ivmintarget, 2) + math.pow(j - jvmintarget, 2)
                        else:
                            dmin = math.pow(i - ivmintarget, 2) + math.pow(j - jvmintarget, 2)
                            if dmin < distmin:
                                distmin = dmin
                                ivmin = i
                                jvmin = j
                    if val >= vmax:
                        if val > vmax:
                            vmax = val
                            ivmax = i
                            jvmax = j
                            distmax = math.pow(i - ivmaxtarget, 2) + math.pow(j - jvmaxtarget, 2)
                        else:
                            dmax = math.pow(i - ivmaxtarget, 2) + math.pow(j - jvmaxtarget, 2)
                            if dmax < distmax:
                                distmax = dmax
                                ivmax = i
                                jvmax = j
        for i in range(0, self._size):
            for j in range(0, self._size):
                val = self._values[i][j]
                if vmax < val:
                    self._values[i][j] = vmax * 1.5
        h = np.array(self._values)  # added some commas and array creation code
        fig = plt.figure(self.name)
        ax = fig.add_subplot(111)
        plt.imshow(h, interpolation="nearest", cmap=self.seqcmap)
        ax.annotate(str(vmin), xy=(jvmin, ivmin), va='center', ha='center')
        ax.annotate(str(vmax), xy=(jvmax, ivmax), va='center', ha='center')


class _TimeLine(_Figure):
    def __init__(self, name=None, origin=None, horizon=None):
        super(_TimeLine, self).__init__(name)
        self._objects = []
        self._pauses = []
        self.next_panel_name = None
        self.next_panel_pauses = None
        self.active_sequencesp = None
        self.active_sequence = None
        self.active_functionsp = None
        self.active_function = None
        self.active_intervalsp = None
        self.intervalWidth = 16
        self.transitionWidth = 8
        self.nbSteps = 20
        self.cpieces = 100
        self.timeStepSync = [1, 2, 5, 10]
        if origin is None:
            self._origin = INT_MAX
        else:
            self._origin = origin
        if horizon is None:
            self._horizon = INT_MIN
        else:
            self._horizon = horizon
        self._mincolor = INT_MAX
        self._maxcolor = INT_MIN
        self.timeStep = 1
        self.panels = []

    def record(self, obj):
        assert isinstance(obj, _TLObject)
        self._objects.append(obj)

    @property
    def origin(self):
        return self._origin

    @property
    def horizon(self):
        return self._horizon

    @property
    def mincolor(self):
        return self._mincolor

    @property
    def maxcolor(self):
        return self._maxcolor

    @property
    def pauses(self):
        return self._pauses

    def get_color(self, c):
        if not is_int(c):
            return c
        if c is None or self._maxcolor == self._mincolor:
            return self.quacmap(0)
        else:
            if c < 0:
                return 'black'
            else:
                return self.quacmap(float(c - self._mincolor) / (self._maxcolor - self._mincolor))

    def update_bounds(self):
        for o in self._objects:
            if o.origin is not None and o.origin > INTERVAL_MIN:
                if o.origin < self._origin:
                    self._origin = o.origin
            if o.horizon is not None and o.horizon < INTERVAL_MAX:
                if o.horizon > self._horizon:
                    self._horizon = o.horizon
            if o.color is not None and is_int(o.color):
                if o.color < self._mincolor:
                    self._mincolor = o.color
                if o.color > self._maxcolor:
                    self._maxcolor = o.color
        assert self.origin < self._horizon and \
               self._origin > INT_MIN and \
               self._horizon < INT_MAX, \
            "Infinite timeline limits, please specify bounded origin and horizon."

    def panel(self, name=None):
        self.next_panel_name = name
        self.next_panel_pauses = []
        # close current sequence if any
        if self.active_sequence is not None:
            self.active_sequencesp.add_sequence(self.active_sequence)
            self.active_sequence = None
        # close current sequence panel if any
        if self.active_sequencesp is not None:
            self.add_panel(self.active_sequencesp)
            self.active_sequencesp = None
        # close current intervals panel if any
        if self.active_intervalsp is not None:
            self.add_panel(self.active_intervalsp)
            self.active_intervalsp = None
        # close current function if any
        if self.active_function is not None:
            self.active_functionsp.add_function(self.active_function)
            self.active_function = None
        # close current function panel if any
        if self.active_functionsp is not None:
            self.add_panel(self.active_functionsp)
            self.active_functionsp = None

    def sequence(self, name=None):
        # close current intervals panel if any
        if self.active_intervalsp is not None:
            self.add_panel(self.active_intervalsp)
            self.active_intervalsp = None
        # close current function if any
        if self.active_function is not None:
            self.active_functionsp.add_function(self.active_function)
            self.active_function = None
        # close current function panel if any
        if self.active_functionsp is not None:
            self.add_panel(self.active_functionsp)
            self.active_functionsp = None
        # create new sequence panel if none exists
        if self.active_sequencesp is None:
            self.active_sequencesp = _SequencePanel(self, self.next_panel_name, pauses=self.next_panel_pauses)
            self.next_panel_name = None
        # create new sequence in the sequence panel
        if self.active_sequence is not None:
            self.active_sequencesp.add_sequence(self.active_sequence)
        self.active_sequence = _Sequence(self, name)

    def interval(self, start, end, color, name):
        # if there is an active sequence, add interval in that sequence
        if self.active_sequence is not None:
            self.active_sequence.add_interval(_Interval(self, start, end, color, name))
        else:
            # close current function if any
            if self.active_function is not None:
                self.active_functionsp.add_function(self.active_function)
                self.active_function = None
            # if there is an active function panel, close it
            if self.active_functionsp is not None:
                self.add_panel(self.active_functionsp)
                self.active_functionsp = None
            # if there is no active intervals panel create one
            if self.active_intervalsp is None:
                self.active_intervalsp = _IntervalPanel(self, self.next_panel_name, pauses=self.next_panel_pauses)
                self.next_panel_name = None
            # add interval in the intervals panel
            self.active_intervalsp.add_interval(_Interval(self, start, end, color, name))

    def transition(self, start, end):
        # if there is no active sequence, create one
        if self.active_sequence is None:
            sequence(self)
        # add transition in the sequence
        if start < end:
            self.active_sequence.add_interval(_Interval(self, start, end, -1, None))

    def function(self, name=None, origin=None, horizon=None, style=None, color=None):
        # close current intervals panel if any
        if self.active_intervalsp is not None:
            self.add_panel(self.active_intervalsp)
            self.active_intervalsp = None
        # close current sequence if any
        if self.active_sequence is not None:
            self.active_sequencesp.add_sequence(self.active_sequence)
            self.active_sequence = None
        # close current sequence panel if any
        if self.active_sequencesp is not None:
            self.add_panel(self.active_sequencesp)
            self.active_sequencesp = None
        # create new function panel if none exists
        if self.active_functionsp is None:
            self.active_functionsp = _FunctionPanel(self, name=self.next_panel_name, pauses=self.next_panel_pauses)
            self.next_panel_name = None
        # create new function in the function panel
        if self.active_function is not None:
            self.active_functionsp.add_function(self.active_function)
        self.active_function = _Function(self, name, origin, horizon, style, color)

    def segment(self, start, end, vstart, vend, name):
        # if there is an active sequence, add segment in that sequence
        if self.active_sequence is not None:
            self.active_sequence.add_segment(_Segment(self, start, end, vstart, vend, name))
        elif self.active_function is not None:
            self.active_function.add_segment(_Segment(self, start, end, vstart, vend, name))
        else:
            self.function()  # Create active function
            self.segment(start, end, vstart, vend, name)

    def pause(self, start, end):
        if self.active_sequencesp is not None:
            self.active_sequencesp.add_pause(start, end)
        elif self.active_functionsp is not None:
            self.active_functionsp.add_pause(start, end)
        elif self.active_intervalsp is not None:
            self.active_intervalsp.add_pause(start, end)
        elif self.next_panel_pauses is not None:
            self.next_panel_pauses.append((start, end))
        else:
            self._pauses.append((start, end))

    def flush(self):
        self.panel()

    def add_panel(self, p):
        self.panels.append(p)

    def compute_time_step(self):
        span = self.horizon - self.origin
        step = span / self.nbSteps
        lg = math.floor(math.log10(step))
        nstep = step / math.pow(10, lg)
        self.timeStep = math.pow(10, lg) * min(self.timeStepSync, key=lambda x: abs(x - nstep))

    def show(self):
        n = len(self.panels)
        self.update_bounds()
        for s in self.panels:
            s.preshow()
        self.compute_time_step()
        heigths = [s.get_height() for s in self.panels]
        f, axarr = plt.subplots(n, sharex=True, num=self.name, gridspec_kw=dict(height_ratios=heigths))
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        if n == 1:
            if self.panels[0].name is not None:
                axarr.set_ylabel(self.panels[0].name)
            self.panels[0].display(axarr)
        else:
            for i in range(0, n):
                # box = dict(facecolor='white', pad=10, alpha=0.8)
                if self.panels[i].name is not None:
                    axarr[i].set_ylabel(self.panels[i].name)  # , bbox=box)
                    axarr[i].yaxis.set_label_coords(-0.05, 0.5)
                self.panels[i].display(axarr[i])
        plt.margins(0.05)
        plt.tight_layout()


class _Visu(object):
    def __init__(self):
        super(_Visu, self).__init__()
        self._active_figure = None
        self._all_figures = []
        self._naming = None

    def timeline(self, title=None, origin=None, horizon=None):
        if self._active_figure is not None:
            self._all_figures.append(self._active_figure)
        self._active_figure = _TimeLine(title, origin, horizon)

    def matrix(self, title=None, matrix=None, tuples=None):
        if self._active_figure is not None:
            self._active_figure.flush()
            self._all_figures.append(self._active_figure)
        self._active_figure = _Matrix(title, matrix, tuples)

    @property
    def has_active_timeline(self):
        return (self._active_figure is not None) and isinstance(self._active_figure, _TimeLine)

    @property
    def active_timeline(self):
        assert self.has_active_timeline, "No active timeline"
        return self._active_figure

    @property
    def has_active_matrix(self):
        return (self._active_figure is not None) and isinstance(self._active_figure, _Matrix)

    @property
    def active_matrix(self):
        assert self.has_active_matrix, "No active matrix"
        return self._active_figure

    def panel(self, name=None):
        if not self.has_active_timeline:
            timeline()
        self.active_timeline.panel(name)

    def sequence(self, name=None):
        if not self.has_active_timeline:
            timeline()
        self.active_timeline.sequence(name)

    def interval(self, start, end, color, name):
        if not self.has_active_timeline:
            timeline()
        self.active_timeline.interval(start, end, color, name)

    def transition(self, start, end):
        if not self.has_active_timeline:
            timeline()
        self.active_timeline.transition(start, end)

    def function(self, name=None, origin=None, horizon=None, style=None, color=None):
        if not self.has_active_timeline:
            timeline()
        self.active_timeline.function(name, origin, horizon, style, color)

    def segment(self, start, end, vstart, vend, name):
        if not self.has_active_timeline:
            timeline()
        self.active_timeline.segment(start, end, vstart, vend, name)

    def pause(self, start, end):
        if not self.has_active_timeline:
            timeline()
        self.active_timeline.pause(start, end)

    def show(self, pngfile=None):
        """ Show the figure
        Args:
            pngfile (optional):  Destination PNG file, None for screen
        """
        if self.has_active_timeline:
            panel()
        for f in self._all_figures:
            f.show()
        if self._active_figure is not None:
            self._active_figure.show()
        if pngfile is None:
            plt.show()
        else:
            plt.savefig(pngfile)
        self._active_figure = None
        self._all_figures = []


_visu = _Visu()


##############################################################################
# This section depends on CPO classes and defines the default display of some
# classes like CpoSolution and CpoTransitionMatrix
##############################################################################

def _canonical_interval(*args):
    """
    Accepted formats for args:
        [int start, int end, int|string color, string name] (canonical form)
        [int start, int end, int|string color]
        [int start, int end]
        [CpoIntervalVarSolution cpointerval, int|string color, string name]
        [CpoIntervalVarSolution cpointerval, int|string color]
        [CpoIntervalVarSolution cpointerval]
    """
    n = len(args)
    color = 0
    name = ''
    assert 1 < n, "Empty argument list for interval"
    if isinstance(args[0], CpoIntervalVarSolution):
        start = args[0].get_start()
        end = args[0].get_end()
        k = 1
    else:
        assert args[0] == 'intervalmin' or is_int(args[0]), "Wrong type for start of interval"
        assert 2 <= n, "Missing end value for interval"
        assert args[1] == 'intervalmax' or is_int(args[1]), "Wrong type for end of interval"
        if args[0] == 'intervalmin':
            start = INTERVAL_MIN
        else:
            start = args[0]
        if args[1] == 'intervalmax':
            end = INTERVAL_MAX
        else:
            end = args[1]
        k = 2
    if k < n:
        color = args[k]
        assert color is None or is_int(color) or is_string(color), "Wrong type for interval color: use 'int' or 'str'"
        k += 1
    if k < n:
        name = args[k]
        assert name is None or is_string(name), "Wrong type for interval name: use 'str'"
        k += 1
    return start, end, color, name


def _canonical_transition(*args):
    """
    Accepted formats for args:
        [int start, int end]
        [CpoIntervalVarSolution]
    """
    n = len(args)
    assert 1 < n, "Empty argument list for transition"
    if isinstance(args[0], CpoIntervalVarSolution):
        start = args[0].get_start()
        end = args[0].get_end()
    else:
        assert args[0] == 'intervalmin' or is_int(args[0]), "Wrong type for start of transition"
        assert 2 <= n, "Missing end value for transition"
        assert args[1] == 'intervalmax' or is_int(args[1]), "Wrong type for end of transition"
        if args[0] == 'intervalmin':
            start = INTERVAL_MIN
        else:
            start = args[0]
        if args[1] == 'intervalmax':
            end = INTERVAL_MAX
        else:
            end = args[1]
    return start, end


def _canonical_pause(*args):
    """
    Accepted formats for args:
        [int start, int end]
        [CpoIntervalVarSolution]
    """
    n = len(args)
    assert 1 < n, "Empty argument list for pause"
    if isinstance(args[0], CpoIntervalVarSolution):
        start = args[0].get_start()
        end = args[0].get_end()
    else:
        assert args[0] == 'intervalmin' or is_int(args[0]), "Wrong type for start of pause"
        assert 2 <= n, "Missing end value for pause"
        assert args[1] == 'intervalmax' or is_int(args[1]), "Wrong type for end of pause"
        if args[0] == 'intervalmin':
            start = INTERVAL_MIN
        else:
            start = args[0]
        if args[1] == 'intervalmax':
            end = INTERVAL_MAX
        else:
            end = args[1]
    return start, end


def _canonical_segment(*args):
    """
    Accepted formats for args:
        [int start, int end, int vstart, int vend, string name] (canonical form)
        [int start, int end, int vstart, int vend]
        [int start, int end, int vstart]
        [CpoIntervalVarSolution cpointerval, int vstart, int vend, string name]
        [CpoIntervalVarSolution cpointerval, int vstart, int vend]
        [CpoIntervalVarSolution cpointerval, int vstart]
    """
    n = len(args)
    name = ''
    assert 1 < n, "Empty argument list for segment"
    if isinstance(args[0], CpoIntervalVarSolution):
        start = args[0].get_start()
        end = args[0].get_end()
        k = 1
    else:
        assert args[0] == 'intervalmin' or is_int(args[0]), "Wrong type for start of segment"
        assert 2 <= n, "Missing end value for segment"
        assert args[1] == 'intervalmax' or is_int(args[1]), "Wrong type for end of segment"
        if args[0] == 'intervalmin':
            start = INTERVAL_MIN
        else:
            start = args[0]
        if args[1] == 'intervalmax':
            end = INTERVAL_MAX
        else:
            end = args[1]
        k = 2
    assert k < n, "Missing start value (or slope) for segment"
    assert is_number(args[k]), \
        "Wrong type for segment start value (or slope): not a number"
    vstart = args[k]
    k += 1
    if k < n:
        if is_string(args[k]):
            name = args[k]
            k = n
            vend = vstart
        else:
            assert is_number(args[k]), \
                "Wrong type for segment end value (or slope): not a number"
            vend = args[k]
            k += 1
    else:
        vend = vstart
    if k < n:
        assert is_string(args[k]), "Wrong type for segment name: use 'str'"
        name = args[k]
        k += 1
    return start, end, vstart, vend, name


def _define_solution(solution, name=None):
    timeline(name)
    vs = solution.get_all_var_solutions()
    itvcolors = dict()
    panel(name="Sequences")
    # Create sequence variables
    has_sequence = False
    for v in vs:
        if isinstance(v, CpoSequenceVarSolution):
            has_sequence = True
            sequence(name=v.get_name())
            for itv in v.get_interval_variables():
                if itv not in itvcolors:
                    color = len(itvcolors)
                    itvcolors[itv] = color
                else:
                    color = itvcolors[itv]
                interval(itv.get_start(), itv.get_end(), color, str(itv.get_name()))
    # Create remaining interval variables
    if has_sequence:
        panel(name="Other intervals")
    else:
        panel(name="Intervals")
    for v in vs:
        if isinstance(v, CpoIntervalVarSolution) and v.is_present():
            if v not in itvcolors:
                color = len(itvcolors)
                itvcolors[v] = color
                interval(v.get_start(), v.get_end(), color, str(v.get_name()))
    # Create state functions
    for v in vs:
        if isinstance(v, CpoStateFunctionSolution):
            sequence(name=v.get_name(), intervals=v)


def _define_matrix(title=None, cpomatrix=None):
    size = cpomatrix.get_size()
    _visu.matrix(title=title,
                 tuples=[(i, j, cpomatrix.get_value(i, j)) for i in range(0, size) for j in range(0, size)])


def _cpofunction_segments(f):
    assert isinstance(f, CpoFunction), "Argument should be an instance of CpoFunction"
    segments = []
    if 0 == len(f.x):
        segments.append((INT_MIN, INT_MAX, f.v0, f.v0))
    else:
        if f.s0 is None:
            segments.append((INT_MIN, f.x[0], 0, f.v0))
        else:
            segments.append((INT_MIN, f.x[0], f.s0, f.v0))
        is_step_function = f.s is None
        for i in range(0, len(f.x) - 1):
            if is_step_function:
                if f.x[i + 1] == INT_MAX:
                    segments.append((f.x[i], f.x[i + 1], f.v[i], 0))
                else:
                    segments.append((f.x[i], f.x[i + 1], f.v[i], f.v[i]))
            else:
                if f.x[i + 1] == INT_MAX:
                    segments.append((f.x[i], f.x[i + 1], f.v[i], f.s[i]))
                else:
                    segments.append((f.x[i], f.x[i + 1], f.v[i], f.v[i] + (f.x[i + 1] - f.x[i]) * f.s[i]))
        i = len(f.x) - 1
        if is_step_function:
            segments.append((f.x[i], INT_MAX, f.v[i], 0))
        else:
            segments.append((f.x[i], INT_MAX, f.v[i], f.s[i]))
    return segments


def _cpofunction_pauses(f):
    assert isinstance(f, CpoFunction), "Argument should be an instance of CpoFunction"
    pauses = []
    if 0 == len(f.x):
        if f.v0 == 0:
            pauses.append((INT_MIN, INT_MAX))
    else:
        if f.s0 is None or f.s0 == 0:
            if f.v0 == 0:
                pauses.append((INT_MIN, f.x[0]))
        is_step_function = f.s is None
        for i in range(0, len(f.x) - 1):
            if is_step_function:
                if f.v[i] == 0:
                    pauses.append((f.x[i], f.x[i + 1]))
            else:
                if f.s[i] == 0 and f.v[i] == 0:
                    pauses.append((f.x[i], f.x[i + 1]))
        i = len(f.x) - 1
        if is_step_function:
            if f.v[i] == 0:
                pauses.append((f.x[i], INT_MAX))
        else:
            if f.s[i] == 0 and f.v[i] == 0:
                pauses.append((f.x[i], INT_MAX))
    return pauses


def _cposequence_intervals(seq):
    assert isinstance(seq, CpoSequenceVarSolution), "Argument should be an instance of CpoSequenceVarSolution"
    itvs = [(s.get_start(), s.get_end(), 0, s.get_name()) for s in seq.get_interval_variables()]
    return itvs


def _cpostatefunction_intervals(f):
    assert isinstance(f, CpoStateFunctionSolution), "Argument should be an instance of CpoStateFunctionSolution"
    # itvs = [(s.get_start(), s.get_end(), s.get_value(), None) for s in f.get_function_steps()]
    itvs = [(s['start'], s['end'], s['value']) for s in f.get_function_steps()]
    return itvs


def _cpostatefunction_segments(f):
    assert isinstance(f, CpoStateFunctionSolution), "Argument should be an instance of CpoStateFunctionSolution"
    # segs = [(s.get_start(), s.get_end(), s.get_value(), s.get_value(), None) for s in f.get_function_steps()]
    # segs = [(s['start'], s['end'], s['value'], s['value']) for s in f.get_function_steps()]
    segs = [(s, e, v, v) for (s, e, v) in f.get_function_steps()]
    return segs


"""
    Here starts the public part of CPOVisu module
"""


def matrix(name=None, matrix=None, tuples=None, cpomatrix=None):
    """ Create a new matrix figure and set it as the current figure.

    One and only one among the arguments 'matrix', 'tuples' or 'cpomatrix'
    should be provided to specify the values of the matrix.

    Args:
        name (str): Name of the figure.
        matrix (list): Values of the matrix specified as a list of lists of integers.
        tuples (list): Values of the matrix specified as a list of tuples (i,j,vij).
            Unspecified matrix cells have value 0 by default.
        cpomatrix (CpoTransitionMatrix): Values of the matrix specified as an
            instance of CpoTransitionMatrix.

    Examples:
    ::

        matrix(name="M1",
               tuples=[(i,j,abs(i-j)) for i in range(50) for j in range(50)])

    """
    n = (matrix is not None) + (tuples is not None) + (cpomatrix is not None)
    assert n == 1, "_Visu.matrix(...): please supply one and only one argument among 'matrix', 'tuples' and 'cpomatrix'"
    if cpomatrix is not None:
        matrix = cpomatrix.value
    if matrix is not None or tuples is not None:
        _visu.matrix(name, matrix, tuples)


def timeline(name=None, origin=None, horizon=None, pauses=None):
    """ Create a new timeline figure and set it as the current figure.

    Args:
        name (str): Name of the figure.
        origin (int): Value of the origin of the x-axis of the timeline.
        horizon (int): Value of the horizon of the x-axis of the timeline.
        pauses (list or CpoFunction): Pause intervals of the timeline. Given as
            and explicit list of pauses (see 'pause') or specified by the
            intervals where the CpoFunction instance has value 0.

    Note that the 'origin' and 'horizon' of the x-axis of the timeline are
    automatically adjusted depending on the content of the timeline.
    In particular, their the scope will be automatically enlarged if the content
    does not fit into the original ['origin','horizon'] interval.

    Pauses specified at the level of the timeline are general pauses that will
    be displayed in all the panels of the timeline.
    """
    _visu.timeline(name, origin, horizon)
    if pauses is not None:
        if isinstance(pauses, CpoFunction):
            pause(pauses)
        else:
            for i in pauses:
                pause(*i)


def panel(name=None):
    """ Creates a new panel in the current timeline figure.

    If no current timeline figure exists, one will be automatically created.
    The type of the created panel will depend on subsequent commands:

        * Command     -> Panel type
        * sequence    -> sequence panel
        * transition  -> sequence panel
        * interval    -> interval panel
        * function    -> function panel
        * segment     -> function panel

    Args:
        name (str): Name of the panel.
    """
    _visu.panel(name)


def sequence(name=None, intervals=None, transitions=None, segments=None):
    """ Creates a new sequence in the current sequence panel.

    If no current panel exists of if the current panel is not a sequence panel,
    a new sequence panel will be automatically created and set as the current panel.

    Args:
        name (str): Name of the sequence.
        intervals (list or CpoSequenceVarSolution or CpoStateFunctionSolution):
            Explicit list of intervals of the sequence (see 'interval') or
            intervals of the specified CpoSequenceVarSolution or
            CpoStateFunctionSolution instance.
        transitions (list): list of transitions of the sequence (see 'transition').
        segments (list or CpoFunction or CpoStateFunctionSolution): Explicit
            list of segments of the sequence (see 'segment') or segments of the
            specified CpoFunction or CpoStateFunctionSolution instance.

    Examples:
    ::

        sequence(name='Machine1',
                 intervals=[(0,10,1,'Job1'),(15,35,2,'Job2')],
                 transitions=[(10,13)])

    """
    _visu.sequence(name)
    if isinstance(intervals, CpoSequenceVarSolution):
        itvs = _cposequence_intervals(intervals)
    elif isinstance(intervals, CpoStateFunctionSolution):
        itvs = _cpostatefunction_intervals(intervals)
    else:
        itvs = intervals
    if itvs is not None:
        for i in itvs:
            interval(*i)

    if transitions is not None:
        for i in transitions:
            transition(*i)

    if isinstance(segments, CpoFunction):
        segs = _cpofunction_segments(segments)
    elif isinstance(segments, CpoStateFunctionSolution):
        segs = _cpostatefunction_segments(segments)
    else:
        segs = segments
    if segs is not None:
        for i in segs:
            segment(*i)


def function(name=None, segments=None, origin=None, horizon=None, style='segment', color=None):
    """ Creates a new function in the current function panel.

    If no current panel exists of if the current panel is not a function panel,
    a new function panel will be automatically created and set as the current
    panel.

    Args:
        name (str): Name of the function.
        segments (list or CpoFunction or CpoStateFunctionSolution): Explicit
            list of segments of the function (see 'segment') or segments of the
            specified CpoFunction or CpoStateFunctionSolution instance.
        origin (int): Value of the origin of function.
        horizon (int): Value of the horizon of function.
        style ('segment' or 'line' or 'area' or 'interval'): Display style of
            the function.
        color (int or str): Color of the function

    Note that the 'origin' and 'horizon' of the x-axis of the timeline are
    automatically adjusted depending on the content of the timeline.
    In particular, their the scope will be automatically enlarged if the content
    does not fit into the original ['origin','horizon'] interval.

    When the color of a function is specified as an integer color index, the
    visualization will automatically allocate a color to this index and will
    ensure that all elements with this color index gets the same color all across
    the timeline.

    Examples:
    ::

        function(name='F1',
                 segments=[(0,10,20),(10,20,0),(20,40,10)],
                 style='area',
                 color=2)
        function(name='F2',
                 segments=[(0,10,20,0),(10,20,0),(20,INT_MAX,7.5,1.0)],
                 style='segment',
                 color='blue')
    """

    if isinstance(segments, CpoFunction):
        segs = _cpofunction_segments(segments)
    elif isinstance(segments, CpoStateFunctionSolution):
        segs = _cpostatefunction_segments(segments)
    else:
        segs = segments
    _visu.function(name, origin, horizon, style, color)
    if segs is not None:
        for i in segs:
            segment(*i)


def interval(*args):
    """ Creates a new interval.

    If the current panel is a sequence panel, the created interval will be added
    to the current sequence in this sequence panel. Otherwise, the interval will
    be added in an interval panel, such an interval panel will be automatically
    created if no current panel exists or the current panel is a function panel.

    Args:
        args : The following combination of arguments is allowed:

            * (int start, int end, int|string color, string name)
            * (int start, int end, int|string color)
            * (int start, int end)
            * (CpoIntervalVarSolution cpointerval, int|string color, string name)
            * (CpoIntervalVarSolution cpointerval, int|string color)
            * (CpoIntervalVarSolution cpointerval)

    When the color of an interval is specified as an integer color index, the
    visualization will automatically allocate a color to this index and will
    ensure that all elements with this color index gets the same color all across
    the timeline.

    Examples:
    ::

        interval(0, 20, 1, 'Job1')
        interval(10, 35, 'darkred')
        interval(itvsol)
    """
    _visu.interval(*_canonical_interval(*args))


def transition(*args):
    """ Creates a new transition interval in the current sequence panel.

    If no current panel exists of if the current panel is not a sequence panel,
    a new sequence panel will be automatically created and set as the current
    panel.

    Args:
        args: The following combination of arguments is allowed:

            * (int start, int end)
            * (CpoIntervalVarSolution cpointerval)

    Examples:
    ::

        transition(0, 20)
        transition(itvsol)
    """
    _visu.transition(*_canonical_transition(*args))


def segment(*args):
    """ Creates a new segment.

    If the current panel is a sequence panel, the created segment will be added
    to the current sequence in this sequence panel. Otherwise, the segment will
    be added in the current function of a function panel. If such a
    function / function panel does not currently exist, it will be automatically
    created.

    Args:
        args: The following combination of arguments is allowed:

            * (int start, int end, int vstart, int vend, string name)
            * (int start, int end, int vstart, int vend)
            * (int start, int end, int vstart)
            * (CpoIntervalVarSolution cpointerval, int vstart, int vend, string name)
            * (CpoIntervalVarSolution cpointerval, int vstart, int vend)
            * (CpoIntervalVarSolution cpointerval, int vstart)

    If start=INT_MIN and end=INT_MAX, then you should have vstart=vend and the
    segment represents the constant function f=vstart on [INT_MIN,INT_MAX).
    Otherwise:

        * When start>INT_MIN, vstart denotes the value of the segment at its start.
        * When start=INT_MIN, vstart denotes the slope of the segment at its start.
        * When end<INT_MAX, vend denotes the value of the segment at its end.
        * When end=INT_MAX, vend denotes the slope of the segment at its end.
        * When vend is not specified, it means the segment has a constant value vstart (step).

    Examples:
    ::

        segment(0, 20, 10, 20)          # value at x=10: 15
        segment(0, 20, 10)              # value at x=10: 10
        segment(INT_MIN,20,-0.5,7.5)    # value at x=10: 12.5
        segment(0, INT_MAX, 0, 0.1)     # value at x=10: 1.0
        segment(INT_MIN,INT_MAX, 3)     # value at x=10: 3
    """
    _visu.segment(*_canonical_segment(*args))


def pause(*args):
    """ Creates a new pause interval.

    If the timeline has a current panel, the pause(s) will be local to this current
    panel, otherwise, if the pause is created before any panel, the pause is a
    general pause of the timeline and will therefore be displayed in all the (future)
    panels of the timeline.

    If the argument is an instance of CpoFunction, pauses will be created for
    all segments of the function with value 0.

    Args:
        args: The following combination of arguments is allowed:

            * (int start, int end)
            * (CpoIntervalVarSolution cpointerval)
            * (CpoFunction)
    """
    if isinstance(args[0], CpoFunction):
        pauses = _cpofunction_pauses(args[0])
        for p in pauses:
            _visu.pause(p[0], p[1])
    else:
        _visu.pause(*_canonical_pause(*args))


def naming(function=None):
    """ Sets a name formatting function.

    The names of the displayed objects (intervals) are sometimes too long or
    ill-formatted for an elegant display. This function allows setting a name
    formatting function that will be used to display all intervals.

    Args:
        function: The name formatter function. If not None, this function
            should take an 'str' as argument and return an 'str'.

    Example:
    ::

        naming(lambda name: name.upper()) # Display all names in upper case
        naming(lambda name: name[0:3])    # Trunk names to first 3 characters

    """
    _visu._naming = function


def show(object=None, name=None, origin=None, horizon=None, pngfile=None):
    """ Shows the active figures.

    Active figures are all the ones that have been created since last call to
    function 'show'. Additionally, this function can use the default display
    to display the CP Optimizer object passed as argument.

    Args:
        object (CpoModelSolution or CpoTransitionMatrix or CpoFunction or
            CpoStateFunctionSolution): object to be displayed using default
            display.
        name (str): Name of the object in the display.
        origin (int): Value of the origin of the x-axis of the timeline.
        horizon (int): Value of the horizon of the x-axis of the timeline.
        file (string): Name of the file where store the figure instead of displaying it on the screen
    """
    # Check if visu enabled
    if not config.context.visu_enabled:
        return

    # Check if particular object has to be displayed
    if object is not None:
        if isinstance(object, CpoSolveResult):
            # use default display for an instance of CPOSolution
            _define_solution(object, name)
        elif isinstance(object, CpoTransitionMatrix):
            # use default display for an instance of CpoTransitionMatrix
            _visu.matrix(title=name, tuples=object.value)
        elif isinstance(object, (CpoFunction, CpoStateFunctionSolution)):
            # use default display for an instance of CpoFunction or
            # CpoStateFunctionSolution
            timeline(name, origin, horizon)
            function(segments=object, name=name, origin=origin, horizon=horizon)
        elif isinstance(object, CpoSequenceVarSolution):
            timeline(name, origin, horizon)
            sequence(intervals=object, name=name)
        elif isinstance(object, CpoStateFunctionSolution):
            timeline(name, origin, horizon)
            function(segments=object, name=name, origin=origin, horizon=horizon)

    # Display active figure(s)
    _visu.show(pngfile=pngfile)

