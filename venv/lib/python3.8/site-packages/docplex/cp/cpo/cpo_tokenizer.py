# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Tokenizer for reading CPO file format
"""

from docplex.cp.tokenizer import *


#==============================================================================
#  Constants
#==============================================================================

TOKEN_IMPLY     = Token(TOKEN_TYPE_OPERATOR, '=>')
TOKEN_AND       = Token(TOKEN_TYPE_OPERATOR, '&&')
TOKEN_OR        = Token(TOKEN_TYPE_OPERATOR, '||')
TOKEN_NOT       = Token(TOKEN_TYPE_OPERATOR, '!')
TOKEN_INTERVAL  = Token(TOKEN_TYPE_OPERATOR, '..')
TOKEN_DIV       = Token(TOKEN_TYPE_OPERATOR, 'div')
TOKEN_LET       = Token(TOKEN_TYPE_SYMBOL,   'let')
TOKEN_SET       = Token(TOKEN_TYPE_SYMBOL,   'set')


#==============================================================================
#  Public classes
#==============================================================================

class CpoTokenizer(Tokenizer):
    """ Tokenizer for CPO file format """
    __slots__ = ()

    def __init__(self, **args):
        """ Create a new tokenizer
        Args:
            See arguments list of :class:`~docplex.cp.utils.Tokenizer`
        """
        super(CpoTokenizer, self).__init__(**args)

        # Add predefined symbols
        self.symbols = {}
        self.add_predefined_symbols([TOKEN_DIV])
        self.add_predefined_symbols([TOKEN_LET])
        self.add_predefined_symbols([TOKEN_SET])

        # Add character handlers
        self.add_char_handler('.', self._read_dot)
        self.add_char_handler('=', self._read_equal)
        self.add_char_handler('!', self._read_bang)
        self.add_char_handler('&', self._read_and)
        self.add_char_handler('|', self._read_pipe)
        self.add_char_handler('/', self._read_slash)


    def _read_dot(self):
        """ Read token starting by dot """
        if self._next_char() != '.':
            raise SyntaxError(self.build_error_string("Unknown token '.'"))
        return TOKEN_INTERVAL


    def _read_equal(self):
        """ Read token starting by - """
        c = self._peek_char()
        if c == '=':
            self._skip_char()
            return TOKEN_EQUAL
        elif c == '>':
            self._skip_char()
            return TOKEN_IMPLY
        return TOKEN_ASSIGN


    def _read_bang(self):
        """ Read token starting by ! """
        if self._peek_char() == '=':
            self._skip_char()
            return TOKEN_DIFFERENT
        return TOKEN_NOT


    def _read_and(self):
        """ Read token starting by & """
        if self._next_char() != '&':
            raise SyntaxError(self.build_error_string("Unknown token '&'"))
        return TOKEN_AND


    def _read_pipe(self):
        """ Read token starting by | """
        if self._next_char() != '|':
            raise SyntaxError(self.build_error_string("Unknown token '|'"))
        return TOKEN_OR


    def _read_slash(self):
        """ Read token starting by / """
        c = self._peek_char()
        if c == '/':
            self._skip_to_end_of_line()
            return Token(TOKEN_TYPE_COMMENT, None if self.skip_comments else self._get_token()[:-1])
        elif c == '*':
            # Read comment
            cbuff = None if self.skip_comments else ['/', '*']
            self._skip_char()
            pc = ''
            c = self._next_char()
            if cbuff:
                cbuff.append(c)
            while c and ((c != '/') or (pc != '*')):
                pc = c
                c = self._next_char()
                if cbuff:
                    cbuff.append(c)
            return Token(TOKEN_TYPE_COMMENT, ''.join(cbuff) if cbuff else None)
        return TOKEN_SLASH


