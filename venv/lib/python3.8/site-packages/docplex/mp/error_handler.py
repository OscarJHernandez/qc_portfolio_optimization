# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from __future__ import print_function
from docplex.mp.utils import DOcplexException, DOcplexLimitsExceeded, resolve_pattern, is_int, is_string

from enum import Enum
import os

##########################
# Error handling


class IErrorHandler(object):
    def __init__(self):
        pass  # pragma: no cover

    def info(self, msg, args=None):
        pass  # pragma: no cover

    def warning(self, msg, args=None):
        pass  # pragma: no cover

    def error(self, msg, args=None):
        pass  # pragma: no cover

    def fatal(self, msg, args=None):
        pass  # pragma: no cover

    def ok(self):
        return False  # pragma: no cover

    def get_output_level(self):
        return 0  # pragma: no cover

    def set_output_level(self, new_level):
        pass  # pragma: no cover

    def ensure(self, condition, msg, *args):
        if not condition:
            self.fatal(msg, args)


class InfoLevel(Enum):
    """ Enumerated type for the possible output levels.

    Info levels are sorted in increasing order of severity: `INFO`, `WARNING`, `ERROR`, `FATAL`.
    Setting a level enables the printing of all messages from that severity and above.

    Example:
        Setting the level to `WARNING` enables the printing of `WARNING`, `ERROR`, and `FATAL` messages, but not
        `INFO` level messages.
        Setting the level to `FATAL` suppresses all messages, except for fatal errors.
    """
    INFO, WARNING, ERROR, FATAL = 1, 10, 100, 9999999

    @classmethod
    def parse(cls, arg, default_level=INFO):
        # INTERNAL
        if not arg:
            return cls.INFO
        elif isinstance(arg, cls):
            return arg
        elif is_string(arg):
            return cls._name2level_map().get(arg.lower(), default_level)
        elif is_int(arg):
            if arg < 10:
                # anything below 10 is INFO
                return cls.INFO
            elif arg < 100:
                return cls.WARNING
            elif arg < 1000:
                return cls.ERROR
            else:
                # level fatal prints nothing except fatal errors
                return cls.FATAL
        else:
            raise DOcplexException("Cannot convert this to InfoLevel: {0!r}".format(arg))

    def __str__(self):
        return self.name

    @staticmethod
    def _headers():
        return {InfoLevel.FATAL: "FATAL",
                InfoLevel.INFO: "*",
                InfoLevel.WARNING: "Warning:",
                InfoLevel.ERROR: "Error:"
                }

    @staticmethod
    def _name2level_map():
        return {"fatal": InfoLevel.FATAL,
                "error": InfoLevel.ERROR,
                "warning": InfoLevel.WARNING,
                "info": InfoLevel.INFO}

    def header(self):
        # cannot put the dict in the class
        # as it willbe interpreted as another enum value.
        return self._headers().get(self, "???")


class AbstractErrorHandler(IErrorHandler):
    TRACE_HEADER = "--"

    def __init__(self, output_level=InfoLevel.INFO):
        IErrorHandler.__init__(self)
        self._trace_enabled = False

        self._number_of_errors = 0
        self._number_of_warnings = 0
        self._number_of_fatals = 0
        self._output_level = InfoLevel.INFO
        self._is_print_suspended = False
        self._postponed = []
        self.set_output_level(output_level)

    @property
    def number_of_warnings(self):
        """ Returns the number of warnings.
        """
        return self._number_of_warnings

    @property
    def number_of_errors(self):
        """ Returns the number of errors.
        """
        return self._number_of_errors

    @property
    def number_of_fatals(self):
        return self._number_of_fatals

    def get_output_level(self):
        return self._output_level

    def set_output_level(self, output_level_arg):
        output_level = InfoLevel.parse(output_level_arg)
        if output_level != self._output_level:
            self._output_level = output_level

    def set_trace_mode(self, trace_mode):
        self._trace_enabled = trace_mode

    def enable_trace(self):
        self.set_trace_mode(True)

    def disable_trace(self):
        self.set_trace_mode(False)

    def is_trace_enabled(self):
        return self._trace_enabled

    def set_quiet(self):
        """ Changes the output level to enable only error messages.
        """
        self.set_output_level(InfoLevel.ERROR)

    def reset(self):
        self._number_of_errors = 0
        self._number_of_warnings = 0
        self._number_of_fatals = 0

    def _internal_is_printed(self, level):
        return self._output_level.value <= level.value

    def _internal_print_if(self, level, msg, args):
        if self._internal_is_printed(level):
            self._internal_print(level, msg, args)

    def _internal_print(self, level, msg, args):
        # resolve message w/ args
        header = level.header()
        self._internal_print_header(header, msg, args)

    def _internal_print_header(self, header, msg, args):
        resolved_message = resolve_pattern(msg, args)
        mline = '%s %s' % (header, resolved_message)
        if self._is_print_suspended:
            self._postponed.append(mline)
        else:
            print(mline)

    def trace_header(self):
        return self.TRACE_HEADER

    def trace(self, msg, args=None):
        if self.is_trace_enabled():
            self._internal_print_header(self.trace_header(), msg, args)

    def info(self, msg, args=None):
        self._internal_print_if(InfoLevel.INFO, msg, args)

    def warning(self, msg, args=None):
        self._number_of_warnings += 1
        self._internal_print_if(InfoLevel.WARNING, msg, args)

    def error(self, msg, args=None):
        docplex_error_stop_here()
        self._number_of_errors += 1
        self._internal_print_if(InfoLevel.ERROR, msg, args)

    def fatal(self, msg, args=None):
        self._number_of_fatals += 1
        resolved_message = resolve_pattern(msg, args)
        docplex_error_stop_here()
        raise DOcplexException(resolved_message)

    def fatal_limits_exceeded(self):
        docplex_error_stop_here()
        raise DOcplexLimitsExceeded()


    def ok(self):
        """ Checks whether the handler has not recorded any error.
        """
        return self._number_of_errors == 0 and self._number_of_fatals == 0

    def prints_trace(self):
        return self.is_trace_enabled()

    def prints_info(self):
        return self._internal_is_printed(InfoLevel.INFO)

    def prints_warning(self):
        return self._internal_is_printed(InfoLevel.WARNING)

    def prints_error(self):
        return self._internal_is_printed(InfoLevel.ERROR)

    def suspend(self):
        self._is_print_suspended = True

    def flush(self):
        self._is_print_suspended = False
        for m in self._postponed:
            print(m)
        self._postponed = []


def docplex_error_stop_here():
    # INTERNAL, use to set breakpoints
    pass


def docplex_add_trivial_infeasible_ct(ct):
    # INTERNAL: set breakpoint here to inspect ct
    pass

def docplex_fatal(msg, *args):
    resolved_message = resolve_pattern(msg, args)
    docplex_error_stop_here()
    raise DOcplexException(resolved_message)

is_debug = os.environ.get('DOCPLEX_DEBUG')

def docplex_debug_msg(*args):
    if is_debug:
        msg = ' '.join(str(x) for x in args)
        #print(f"-- {msg}")
        print("-- {0}".format(msg))


class DefaultErrorHandler(AbstractErrorHandler):
    """ The default error handler class.

    """

    def __init__(self, output_level=InfoLevel):
        AbstractErrorHandler.__init__(self, output_level)


class SilentErrorHandler(AbstractErrorHandler):
    def __init__(self, output_level=InfoLevel):
        AbstractErrorHandler.__init__(self, output_level)

    def _internal_print(self, level, msg, args):
        # nothing out, this is the point!
        pass

    def suspend(self):
        pass

    def flush(self):
        pass


