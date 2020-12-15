# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2018
# --------------------------------------------------------------------------
'''This package provide simple logging facilities.
'''
import logging


class DocplexLogger(object):
    '''Simple logger interface.
    '''
    def log(self, level, msg):
        '''Logs a message with the specified level.
        '''
        raise NotImplementedError('subclasses must override log()!')

    def debug(self, msg):
        '''Logs a message with level DEBUG.
        '''
        self.log(logging.DEBUG, msg)

    def info(self, msg):
        '''Logs a message with level INFO.
        '''
        self.log(logging.INFO, msg)

    def warning(self, msg):
        '''Logs a message with level WARNING.
        '''
        self.log(logging.WARNING, msg)

    def error(self, msg):
        '''Logs a message with level ERROR.
        '''
        self.log(logging.ERROR, msg)

    def critical(self, msg):
        '''Logs a message with level CRITICAL.
        '''
        self.log(logging.CRITICAL, msg)


class LoggerToDocloud(DocplexLogger):
    '''This logger maps logs with python style logging levels to a docplexcloud
    logger expecting java style logging levels.
    '''
    def __init__(self, docloudlogger):
        super(LoggerToDocloud, self).__init__()
        self.docloudlogger = docloudlogger

    def log(self, level, msg):
        if level == logging.DEBUG:
            self.docloudlogger.fine(msg)
        elif level == logging.INFO:
            self.docloudlogger.info(msg)
        elif level == logging.WARNING:
            self.docloudlogger.warning(msg)
        elif level == logging.ERROR or level == logging.CRITICAL:
            self.docloudlogger.error(msg)
        else:
            raise ValueError('Supported logging levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL. Provided = %s' % level)


class LoggerToFile(DocplexLogger):
    '''This logger logs log records with python style logging levels and just print them
    on the specified stream
    '''
    def __init__(self, file):
        super(LoggerToDocloud, self).__init__()
        self.file = file

    def log(self, level, msg):
        self.file.write('[%s] %s\n' % (level, msg))
