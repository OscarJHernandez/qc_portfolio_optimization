# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2018
# --------------------------------------------------------------------------
# gendoc: ignore

'''The class to handle zip exports
'''
import glob
import json
import os
from os.path import join, dirname, basename, splitext
from zipfile import ZipFile, BadZipfile

try:
    import pandas
except ImportError:
    pandas = None


def get_project_set(zipfile):
    projectset = set()
    for s in zipfile.namelist():
        r = s.split('/')
        if r:
            projectset.add(r[0])
    return projectset


def get_model_names(root, zipfile):
    modelnames = []
    for n in zipfile.namelist():
        elt = n.split('/')
        if len(elt) == 5 and \
           elt[0] == root and \
           elt[1] == 'decision-optimization' and \
           elt[3] == '.scenarios' and \
           elt[4] == '':  # this means n is a directory
            modelnames.append(elt[2])
    return modelnames


def get_all_inputs(directory=None):
    inputs = {}
    all_csv = "*.csv"
    g = join(directory, all_csv) if directory else all_csv
    for f in glob.glob(g):
        name = splitext(basename(f))[0]
        inputs[name] = pandas.read_csv(f, index_col=None)
    return inputs

class MultipleRootException(Exception):
    '''The exception raised if multiple roots are found in an exported project.
    '''
    pass


def normalized(path):
    '''Returns a normalized path (see os.path.normpath) for a ZipFile.
    Backslashes (if on windows) are converted to slashes
    '''
    p = os.path.normpath(path)
    p = p.replace('\\', '/')
    return p


class WSProjectDirectoryHandler(object):
    @staticmethod
    def accepts(path):
        '''Returns true if the path can be read by this handler.

        This returns true if some .scenarios are found
        '''
        if not os.path.isdir(path):
            return False
        return os.path.isfile(join(path, '.decision.json')) and os.path.isdir(join(path, '.containers'))

    def __init__(self, path):
        '''Creates a new directory projec handler
        '''
        super(WSProjectDirectoryHandler, self).__init__()
        self.path = path

    def get_inputs(self, model, scenario, project=None):
        '''Returns inputs for the model handled by this
        '''
        if not pandas:
            raise RuntimeError('Input can only be read if pandas is installed')
        in_df = {}
        rootpath = join(self.path, '.containers', scenario)
        descriptor_name = '/'.join([rootpath, 'scenario.json'])
        with open(descriptor_name) as f:
            descriptor = json.load(f)
        assets = descriptor['categories']['input']['assets']
        for t in assets:
            if assets[t]["type"] == 'Table':
                table_path = '/'.join([rootpath, t])
                table_name= os.path.splitext(t)[0]
                with open(normalized(table_path)) as t_input:
                    # get columns
                    data = pandas.read_csv(t_input, index_col=None)
                    in_df[table_name] = data
        return in_df

    def get_input_stream(self, model, scenario, path, project=None):
        rootpath = join(self.path, '.containers', scenario)
        return open(join(rootpath, path), 'r')


class SimpleCSVProjectDirectoryHandler(object):
    @staticmethod
    def accepts(path):
        '''Returns true if the path can be read by this handler.

        This returns true if there is a model.py files with some .csv files
        '''
        if not os.path.isdir(path):
            return False
        has_csv = any([os.path.isfile(f) for f in glob.glob(join(path, "*.csv"))])
        return os.path.isfile(join(path, 'model.py')) and has_csv

    def __init__(self, path):
        '''Creates a new directory projec handler
        '''
        super(SimpleCSVProjectDirectoryHandler, self).__init__()
        self.path = path

    def get_inputs(self, model, scenario, project=None):
        '''Returns inputs for the model handled by this
        '''
        if not pandas:
            raise RuntimeError('Input can only be read if pandas is installed')
        in_df = get_all_inputs(self.path)
        return in_df

    def get_input_stream(self, model, scenario, path, project=None):
        rootpath = self.path
        return open(join(rootpath, path), 'r')



class ProjectDirectoryHandler(object):
    @staticmethod
    def accepts(path):
        '''Returns true if the path can be read by this handler.

        This returns true if some .scenarios are found
        '''
        if not os.path.isdir(path):
            return False
        try:
            do = join(path, 'decision-optimization')
            models = glob.glob('%s/*/.scenarios' % do)
            modelnames = [basename(dirname(m)) for m in models]
            return len(modelnames) > 0
        except BadZipfile:
            return False

    def __init__(self, path):
        '''Creates a new directory projec handler
        '''
        super(ProjectDirectoryHandler, self).__init__()
        self.path = path

    def get_inputs(self, model, scenario, project=None):
        '''Returns inputs for the model handled by this
        '''
        if not pandas:
            raise RuntimeError('Input can only be read if pandas is installed')
        in_df = {}
        rootpath = join(self.path, 'decision-optimization', model,
                        '.scenarios', scenario)
        descriptor_name = '/'.join([rootpath, 'scenario.json'])
        with open(descriptor_name) as f:
            descriptor = json.load(f)
        for t in descriptor['tables']:
            if t['category'] == 'input':
                table_path = '/'.join([rootpath, t['path']])
                with open(normalized(table_path)) as t_input:
                    # get columns
                    columns = [c['key'] for c in t['tableType']['columns']]
                    data = pandas.read_csv(t_input, index_col=None, header=None,
                                           names=columns)
                    in_df[t['name']] = data
        return in_df

    def get_input_stream(self, model, scenario, path, project=None):
        rootpath = join(self.path, 'decision-optimization', model,
                        '.scenarios', scenario)
        return open(join(rootpath, path), 'r')


class ProjectZipHandler(object):
    '''This class handles data retrieval from a DODS/DSXL project export as
    a zip file.
    '''

    @staticmethod
    def accepts(path):
        '''Returns true if the path can be read by this handler.

        This returns true if some .scenarios are found
        '''
        if not os.path.isfile(path):
            return False
        try:
            with ZipFile(path, 'r') as zipfile:
                projectset = get_project_set(zipfile)
                if len(projectset) != 1:
                    return False  # currently does not support multi root projects
                root = list(projectset)[0]
                modelnames = get_model_names(root, zipfile)
                return len(modelnames) > 0
        except BadZipfile:
            return False

    def __init__(self, path):
        '''Creates a new content handler
        '''
        super(ProjectZipHandler, self).__init__()
        self.zipfile = ZipFile(path, 'r')
        self.projectset = get_project_set(self.zipfile)

    @property
    def projects(self):
        '''The lists of projects
        '''
        return list(self.projectset)

    @property
    def root_project(self):
        '''The root project
        '''
        if len(self.projectset) == 1:
            return self.projects[0]
        else:
            raise MultipleRootException('Project has more than one root')

    def get_inputs(self, model, scenario, project=None):
        '''Returns inputs for the model handled by this
        '''
        if not pandas:
            raise RuntimeError('Input can only be read if pandas is installed')
        in_df = {}
        try:
            root = project if project else self.root_project
        except MultipleRootException:
            raise ValueError('Multiple projects in archive, please provide the \'project\' parameter')
        rootpath = '/'.join([root, 'decision-optimization', model, '.scenarios', scenario])
        descriptor_name = '/'.join([rootpath, 'scenario.json'])
        with self.zipfile.open(descriptor_name) as f:
            descriptor = json.load(f)
        for t in descriptor['tables']:
            if t['category'] == 'input':
                table_path = '/'.join([rootpath, t['path']])
                npath = normalized(table_path)
                in_scenario = npath.startswith(rootpath)
                with self.zipfile.open(npath) as t_input:
                    # get columns
                    columns = [c['key'] for c in t['tableType']['columns']]
                    types = [c['dataType'] for c in t['tableType']['columns']]
                    if in_scenario:
                        # csvs in scenario dir have no header no index
                        data = pandas.read_csv(t_input, index_col=None, header=None,
                                               names=columns)
                    else:
                        # csvs in datasets have headers and index
                        data = pandas.read_csv(t_input)
                    in_df[t['name']] = data
        return in_df

    def get_input_stream(self, model, scenario, path, project=None):
        try:
            root = project if project else self.root_project
        except MultipleRootException:
            raise ValueError('Multiple projects in archive, please provide the \'project\' parameter')
        rootpath = '/'.join([root, 'decision-optimization', model, '.scenarios', scenario])
        return self.zipfile.open('%s/%s' % (rootpath, path), 'r')
