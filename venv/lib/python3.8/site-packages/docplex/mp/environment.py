# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2019
# --------------------------------------------------------------------------
'''Provides utility functions about the runtime environment.

You can display information about your runtime environment using::

    $ python
    >>> from docplex.mp.environment import Environment
    >>> Environment().print_information()

or by invoking the `docplex.mp.environment` package on your shell command line::

    $ python -m docplex.mp.environment
    * system is: Linux 64bit
    * Python version 3.6.1, located at: /usr/bin/python
    * docplex is present, version is (2, 9, 0)
    * CPLEX library is present, version is 12.9.0.0, located at: /usr/local/CPLEX_Studio129/cplex/python/3.6/x86-64_linux
'''
try:
    import importlib.util as importlib_util
except ImportError:
    importlib_util = None  # Python 2
import platform
import os
import sys
import warnings

from docplex.mp.error_handler import docplex_fatal

min_cplex_major = 12
min_cplex_minor = 8


def env_is_64_bit():
    return sys.maxsize > 2 ** 32


class UnsupportedPlatformError(Exception):
    pass


# maps paths to modules for already loaded cplexes
# for path == None, key is "__NONE__"
_loaded_cplexes = {}


# noinspection PyPep8
class Environment(object):
    """ This class detects and contains information regarding other modules of interest, such as
        whether CPLEX, `numpy`, and `matplotlib` are installed.
    """
    _default_env = None  # The default env singleton

    """ This class detects and contains information regarding other modules of interest, such as
        whether CPLEX, `numpy`, and `matplotlib` are installed.
    """
    def __init__(self, start_auto_configure=True, logger=None):
        """
        __init__(self)
        """
        self._found_cplex = False
        self._cplex_version = ''
        self._cplex_location = None

        self._found_numpy = None
        self._numpy_version = None
        self._numpy_hook = None

        self._found_pandas = None
        self._pandas_version = ''
        self._found_matplotlib = None
        self._matplotlib_version = None

        self._python_version = platform.python_version()
        self._system = platform.system()
        self._machine = platform.machine()
        self._bitness = platform.architecture()[0]
        self._is64bit = sys.maxsize > 2 ** 32

        if start_auto_configure:
            self.auto_configure(logger=logger)

    # class variable
    env_is_python36 = platform.python_version() >= '3.6'

    def _get_numpy_hook(self):
        return self._numpy_hook

    def _set_numpy_hook(self, hook):
        self._numpy_hook = hook
        if hook is not None:
            if self.has_numpy:  # now that we have set a hook, do check for numpy
                hook()          # if numpy is present, do call the hook

    numpy_hook = property(_get_numpy_hook, _set_numpy_hook)

    def equals(self, other):
        if type(other) != Environment:
            return False
        if self.has_cplex != other.has_cplex:
            return False
        if self.cplex_version != other.cplex_version:
            return False

        if self.has_numpy != other.has_numpy:
            return False
        if self.has_matplotlib != other.has_matplotlib:
            return False
        if self.has_pandas != other.has_pandas:
            return False

        return True

    @property
    def has_cplex(self):
        """True if the CPLEX libraries are available.

        The cplex libraries search order is:

            - import the module `import cplex` if the import is sucessful
            - import the module in location $CPLEX_STUDIO_DIR1210/cplex/python/<python.version>/<platform>
            - import the module in location $CPLEX_STUDIO_DIR129/cplex/python/<python.version>/<platform>
            - import the module in location $CPLEX_STUDIO_DIR128/cplex/python/<python.version>/<platform>
        """
        return self._found_cplex

    def hash_cplex_with_version_min(self, min_version):
        return self.has_cplex and self._cplex_version >= min_version

    def check_cplex_version(self):
        cpx_version = self.cplex_version_as_tuple

        if self.has_cplex and cpx_version < (min_cplex_major, min_cplex_minor):
            s_min_version = "{0}.{1}".format(min_cplex_major, min_cplex_minor)
            docplex_fatal("DOcplex supports Cplex from {0} up, unsupported version {1} was found"
                          .format(s_min_version, self.cplex_version))

    @staticmethod
    def cplex_platform():
        sys_platform = platform.system()
        machine = platform.machine()
        if sys_platform == 'Windows':
            return 'x64_win64'
        elif sys_platform == 'Darwin':
            return 'x86-64osx'
        elif sys_platform == 'Linux':
            if machine == 'x86_64':
                return 'x86-64_linux'
            else:
                # different flavors of linux (ppc64le_linux, s390x_linux)
                return machine + "_linux"
        return None

    @staticmethod
    def cplex_distribname():
        distribname = Environment.cplex_platform()
        if distribname == 'x64_win64':
            distribname = 'x64_windows'
        return distribname

    @property
    def cplex_version(self):
        return self._cplex_version

    @property
    def cplex_version_as_tuple(self):
        cpxv = self._cplex_version
        if cpxv:
            return tuple(float(x) for x in cpxv.split('.'))
        else:
            return (0,)

    @property
    def has_matplotlib(self):
        """True if the `matplotlib` libraries are available.
        """
        if self._found_matplotlib is None:
            self.check_matplotlib()
        return self._found_matplotlib

    @property
    def has_pandas(self):
        """True if the `pandas` libraries are available.
        """
        self.check_pandas()
        return self._found_pandas

    @property
    def pandas_version(self):
        self.check_pandas()
        return self._pandas_version

    @property
    def cplex_location(self):
        """The system path where CPLEX is located, if present. Otherwise, returns None.
        """
        return self._cplex_location

    @property
    def has_numpy(self):
        """True if the `numpy` libraries are available.
        """
        self.check_numpy()
        return self._found_numpy

    def is_64bit(self):
        """True if running on a 64-bit platform.
        """
        return self._is64bit

    @property
    def python_version(self):
        """ Returns the Python version as a string"""
        return platform.python_version()

    def auto_configure(self, logger=None):
        self.check_cplex(logger=logger)
        # check for pandas (watson studio)
        self.check_pandas()

    def check_all(self):
        self.check_cplex()
        self.check_pandas()
        self.check_numpy()
        self.check_matplotlib()

    def get_cplex_module(self, default_location=None, logger=None):
        '''Returns the cplex module.

        If `default_location` is None, this method will try to import the cplex module in the following order:

            - by importing the module `import cplex` if the import is sucessful
            - by importing the module in location $CPLEX_STUDIO_DIR201/cplex/python
            - by importing the module in location $CPLEX_STUDIO_DIR1210/cplex/python
            - by importing the module in location $CPLEX_STUDIO_DIR129/cplex/python
            - by importing the module in location $CPLEX_STUDIO_DIR128/cplex/python

        If `default_location` is a valid path and contains a valid python package,
        `cplex` is imported from the specified location.

        If `default_location` is a valid path and
        `<default_location>/cplex/python/<python_version>/<platform>` exists, `cplex`
        is imported from that location

        If `cplex` could not be found, this method returns `None`
        '''
        cplex = None

        def load_cplex(location, version=""):
            # we cache loaded modules as we want to make sure that modules are loaded
            # only once (same behaviour than 'import')
            # example location: C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\python\3.7\x64_win64\
            cplex = _loaded_cplexes.get(location, None)
            if cplex is None:
                absolute_name = "cplex%s" % version
                module_location = os.path.join(location, "cplex", "__init__.py")
                if not os.path.isfile(module_location):
                    raise FileNotFoundError("Could not load module from %s" % module_location)
                if importlib_util:
                    spec = importlib_util.spec_from_file_location(absolute_name,
                                                                  module_location)
                    cplex = importlib_util.module_from_spec(spec)
                    # TODO: Error out if one was already loaded
                    previous = sys.modules.get(absolute_name, None)
                    if previous is not None:
                        lo = location if location else "default sys.path"
                        raise RuntimeError("Cannot load cplex from %s, a previous version has already been loaded from %s" % (lo, previous.__file__))
                    sys.modules[absolute_name] = cplex
                    spec.loader.exec_module(cplex)
                else:
                    import imp
                    cplex = imp.load_source(absolute_name.split('.')[-1], module_location)
                _loaded_cplexes[location] = cplex
            return cplex

        def load_cplex_from_cos_root(cos_root, version=""):
            platform = Environment.cplex_platform()
            if platform is None:
                raise UnsupportedPlatformError("Platform not supported, please install cplex python module")
            python_version = '%s.%s' % (sys.version_info[0],
                                        sys.version_info[1])
            full_path = os.path.join(cos_root, 'cplex', 'python', python_version, platform)
            return load_cplex(full_path, version=version)

        if default_location is None:
            try:
                import cplex  #@UnresolvedImport
                if logger is not None:
                    logger.info("Found cplex with 'import cplex'")
            except (ImportError, ModuleNotFoundError):
                # in py3.7, ModuleNotFoundError is raised
                user_cos_location = os.environ.get('DOCPLEX_COS_LOCATION', None)
                if user_cos_location is not None:
                    cplex = load_cplex_from_cos_root(user_cos_location)
                    if cplex is None:
                        # user provided a cos location that was not right, raise warning
                        warnings.warn("Could not load CPLEX from Location provided by DOCPLEX_COS_LOCATION=%s. Using default locations." % user_cos_location)
                if cplex is None:
                    try_environs = ['CPLEX_STUDIO_DIR201',
                                    'CPLEX_STUDIO_DIR1210',
                                    'CPLEX_STUDIO_DIR129',
                                    'CPLEX_STUDIO_DIR128']
                    for t in try_environs:
                        loc = os.environ.get(t, None)
                        # version = t[len('CPLEX_STUDIO_DIR'):] if loc else ""
                        # currently, there are some import in CPLEX, like:
                        #   File "C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\python\3.7\x64_win64\cplex\_internal\_pycplex_platform.py", line 24, in <module>
                        #     from cplex._internal.py37_cplex12100 import *
                        # that prevent us from loading multiples instances of cplex
                        # so for now, let's just ignore this version
                        cplex = load_cplex_from_cos_root(loc) if loc else None
                        if logger is not None:
                            logger.info("Looking into location %s, found = %s" % (loc, (cplex is not None)))
                        if cplex is not None:
                            return cplex
        else:
            if os.path.isfile(os.path.join(default_location, "__init__.py")):
                cplex = load_cplex(default_location)
            else:
                cplex = load_cplex_from_cos_root(default_location)
        return cplex

    def check_cplex(self, logger=None):
        # detecting CPLEX using default search location
        cplex = self.get_cplex_module(logger=logger)
        self._found_cplex = (cplex is not None)
        if (self.has_cplex):
            cplex_module_file = cplex.__file__
            if cplex_module_file:
                self._cplex_location = os.path.dirname(os.path.dirname(cplex_module_file))
            try:
                self._cplex_version = cplex.__version__
            except AttributeError:
                # older version: use an instance
                cpx = cplex.Cplex()
                # format: MM.mm.rr.ff e.g.e 12.6.2.0
                self._cplex_version = cpx.get_version()
                # terminate the dummy instance...
                del cpx

    def check_numpy(self):
        if self._found_numpy is None:
            try:
                import numpy.version as npv
                self._found_numpy = True
                self._numpy_version = npv.version

                self_numpy_hook = self._numpy_hook
                if self_numpy_hook is not None:
                    # lazy call the hook once at first check time.
                    self_numpy_hook()

            except ImportError:
                self._found_numpy = False
                self._numpy_version = None

        return self._found_numpy

    def check_matplotlib(self):
        try:
            from matplotlib import __version__ as matplotlib_version
            self._found_matplotlib = True
            self._matplotlib_version = matplotlib_version
        except ImportError:
            self._found_matplotlib = False

    def check_pandas(self):
        if self._found_pandas is None:
            try:
                import pandas
                self._found_pandas = True
                self._pandas_version = pandas.__version__
            except ImportError:
                self._found_pandas = False

    @staticmethod
    def _display_feature(is_present, feature_name, feature_version, location=None):
        safe_feature_version = feature_version or "?"
        if is_present is None:
            pass  # we dont know yet
        elif is_present:
            if location:
                print("* {0} is present, version is {1}, located at: {2}".format(feature_name, safe_feature_version,
                                                                                 location))
            else:
                print("* {0} is present, version is {1}".format(feature_name, safe_feature_version))
        else:
            print("* {0} is not available".format(feature_name))

    @property
    def max_nb_digits(self):
        # source: https://en.wikipedia.org/wiki/IEEE_floating_point
        return 17 if self.is_64bit() else 9

    @property
    def bitness(self):
        return 64 if self.is_64bit() else 32

    def print_information(self):
        print("* system is: {0} {1}".format(self._system, self._bitness))
        from sys import version_info
        from docplex.mp import __version_info__

        python_version = '%s.%s.%s' % (version_info[0], version_info[1], version_info[2])
        print("* Python version %s, located at: %s" % (python_version, sys.executable))
        self._display_feature(True, "docplex", "%d.%d.%d" % __version_info__)
        self._display_feature(self.has_cplex, "CPLEX library", self._cplex_version, self._cplex_location)
        self._display_feature(self._found_pandas, "pandas", self._pandas_version)
        self._display_feature(self._found_numpy, "numpy", self._numpy_version)
        self._display_feature(self._found_matplotlib, "matplotlib", self._matplotlib_version)

    @staticmethod
    def closed_env():
        return Environment(start_auto_configure=False)

    @staticmethod
    def make_new_configured_env():
        # returns a fresh new environment
        return Environment(start_auto_configure=True)

    @staticmethod
    def get_default_env():
        if not Environment._default_env:
            Environment._default_env = Environment.make_new_configured_env()
        return Environment._default_env

    # for pickling: recreate a fresh environment at the other end of pickle.
    def __reduce__(self):
        return Environment.make_new_configured_env, ()


def get_closed_environment():
    # This instance assumes nothing is found, CPLEX, numpy, etc, to be used for tests
    env = Environment(start_auto_configure=False)
    # force matplotlib absent
    env._found_matplotlib = False
    env._found_numpy = False
    env._found_pandas = False
    return env


if __name__ == '__main__':
    Environment().print_information()
