# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2018
# --------------------------------------------------------------------------
# gendoc: ignore

'''Runtime Environment
'''
import os
import sys
import traceback
import json

try:
    import numpy
except ImportError:
    numpy = None

from docplex.util.environment import get_environment, WorkerEnvironment, \
    Environment, LocalEnvironment, OverrideEnvironment


from .project_handler import ProjectZipHandler, ProjectDirectoryHandler, WSProjectDirectoryHandler, SimpleCSVProjectDirectoryHandler


def get_line_of_model(n, handler, model, scenario):
    with handler.get_input_stream(model, scenario, 'model.py') as m:
        lines = m.readlines()
        return lines[n - 1]


class InterpreterError(Exception):
    pass


class _WorkerEnvironmentOverride(WorkerEnvironment):
    '''A worker environment override, only useful when the default_env
    is a LocalEnvironment. If the default_env is a WorkerEnvironment,
    just use it.
    '''
    def __init__(self, env, solve_hook=None):
        super(_WorkerEnvironmentOverride, self).__init__(None)
        self.env = env
        self.solveDetailsAsJson = []  # Keep a list of these

    def get_available_core_count(self):
        return self.env.get_available_core_count()

    def get_input_stream(self, name):
        return self.env.get_input_stream(name)

    def get_output_stream(self, name):
        return self.env.get_output_stream(name)

    def set_output_attachment(self, name, filename):
        return self.env.set_output_attachment(name, filename)

    def get_parameter(self, name):
        return self.env.get_parameter(name)

    def update_solve_details(self, details):
        r = self.env.update_solve_details(details)
        self.solveDetailsAsJson.append(json.dumps(details))
        return r

    def notify_start_solve(self, solve_details):
        return self.env.notify_start_solve(solve_details)

    def notify_end_solve(self, status, **kwargs):
        return self.env.notify_end_solve(status, **kwargs)

    def set_stop_callback(self, cb):
        return self.env.set_stop_callback(cb)

    def get_stop_callback(self):
        return self.env.get_stop_callback()


class Executor(object):
    '''The class to execute some code with an execution context that mimics
    that on DODS.

    Note that for a model to be actually run, pandas is necessary.
    '''
    def __init__(self, path):
        self.path = path
        handler = None
        if ProjectZipHandler.accepts(path):
            handler = ProjectZipHandler(path)
        elif ProjectDirectoryHandler.accepts(path):
            handler = ProjectDirectoryHandler(path)
        elif WSProjectDirectoryHandler.accepts(path):
            handler = WSProjectDirectoryHandler(path)
        elif SimpleCSVProjectDirectoryHandler.accepts(path):
            handler = SimpleCSVProjectDirectoryHandler(path)
        else:
            raise ValueError('No handler for %s' % path)
        print("OPENING %s with %s handler" % (path, type(handler)))
        self.handler = handler
        self.override_env = None

    def run_model(self, model, scenario, is_dods=True, override_env=None):
        '''Run the scenario from the specified model.

        Returns:
            a dict with the output tables.
        '''
        variables = {}
        variables['inputs'] = self.handler.get_inputs(model, scenario)
        variables['output_lock'] = get_environment().output_lock
        variables['outputs'] = {}

        self.override_env = None
        if isinstance(get_environment(), LocalEnvironment):
            if override_env is None:
                override_env = _WorkerEnvironmentOverride(get_environment())
            self.override_env = override_env

        with OverrideEnvironment(self.override_env):
            saved_environ = os.environ.copy()
            try:
                if is_dods:
                    os.environ['IS_DODS'] = 'True'
                    os.environ['DOCPLEX_CONTEXT'] = 'solver.auto_publish=True'
                with self.handler.get_input_stream(model, scenario, 'model.py') as m:
                    try:
                        contents = m.read()
                        if not isinstance(contents, str):
                            contents = contents.decode('utf-8')
                        exec(contents, variables)
                    except SyntaxError as err:
                        error_class = err.__class__.__name__
                        detail = err.args[0]
                        line_number = err.lineno
                        imsg = 'File "model.py", line %s\n' % line_number
                        imsg += err.text.rstrip() + '\n'
                        spaces = ' ' * (err.offset - 1) if err.offset > 1 else ''
                        imsg += spaces + "^\n"
                        imsg += '%s: %s\n' % (error_class, detail)
                        raise InterpreterError(imsg)
                    except Exception as err:
                        error_class = err.__class__.__name__
                        detail = err.args[0]
                        _, _, tb = sys.exc_info()
                        ttb = traceback.extract_tb(tb)
                        ttb[1] = ('model.py', ttb[1][1], ttb[1][2],
                                  get_line_of_model(ttb[1][1],
                                                    self.handler, model, scenario))
                        line_number = ttb[1][1]
                        ttb = ttb[1:]
                        s = traceback.format_list(ttb)
                        imsg = (''.join(s))
                        imsg += '%s: %s\n' % (error_class, detail)
                        raise InterpreterError(imsg)
                return variables['outputs']
            finally:
                os.environ = saved_environ
