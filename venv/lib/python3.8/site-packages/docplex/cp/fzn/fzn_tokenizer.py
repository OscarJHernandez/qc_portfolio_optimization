# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Tokenizer for reading FlatZinc FZN file format
"""

from docplex.cp.tokenizer import *


#==============================================================================
#  Constants
#==============================================================================


TOKEN_INTERVAL = Token(TOKEN_TYPE_OPERATOR, "..")
TOKEN_DIV      = Token(TOKEN_TYPE_OPERATOR, "div")

TOKEN_DOUBLECOLON = Token(TOKEN_TYPE_PUNCTUATION, "::")

TOKEN_KEYWORD_PREDICATE  = Token(TOKEN_TYPE_KEYWORD, "predicate")
TOKEN_KEYWORD_CONSTRAINT = Token(TOKEN_TYPE_KEYWORD, "constraint")
TOKEN_KEYWORD_ARRAY      = Token(TOKEN_TYPE_KEYWORD, "array")
TOKEN_KEYWORD_SET        = Token(TOKEN_TYPE_KEYWORD, "set")
TOKEN_KEYWORD_VAR        = Token(TOKEN_TYPE_KEYWORD, "var")
TOKEN_KEYWORD_OF         = Token(TOKEN_TYPE_KEYWORD, "of")
TOKEN_KEYWORD_BOOL       = Token(TOKEN_TYPE_KEYWORD, "bool")
TOKEN_KEYWORD_FLOAT      = Token(TOKEN_TYPE_KEYWORD, "float")
TOKEN_KEYWORD_INT        = Token(TOKEN_TYPE_KEYWORD, "int")

TOKEN_KEYWORD_SOLVE      = Token(TOKEN_TYPE_KEYWORD, "solve")
TOKEN_KEYWORD_SATISFY    = Token(TOKEN_TYPE_KEYWORD, "satisfy")
TOKEN_KEYWORD_MINIMIZE   = Token(TOKEN_TYPE_KEYWORD, "minimize")
TOKEN_KEYWORD_MAXIMIZE   = Token(TOKEN_TYPE_KEYWORD, "maximize")


# # List of predefined symbols
PREDEFINED_SYMBOL_TOKENS = [TOKEN_KEYWORD_TRUE, TOKEN_KEYWORD_FALSE, TOKEN_DIV,
                            TOKEN_KEYWORD_PREDICATE, TOKEN_KEYWORD_CONSTRAINT, TOKEN_KEYWORD_ARRAY, TOKEN_KEYWORD_SET, TOKEN_KEYWORD_VAR, TOKEN_KEYWORD_OF,
                            TOKEN_KEYWORD_BOOL, TOKEN_KEYWORD_FLOAT, TOKEN_KEYWORD_INT,
                            TOKEN_KEYWORD_SOLVE, TOKEN_KEYWORD_SATISFY, TOKEN_KEYWORD_MINIMIZE, TOKEN_KEYWORD_MAXIMIZE]


#==============================================================================
#  Public classes
#==============================================================================

class FznTokenizer(Tokenizer):
    """ Tokenizer for FZN file format """
    __slots__ = ()

    def __init__(self, **args):
        """ Create a new tokenizer
        Args:
            See arguments list of :class:`~docplex.cp.utils.Tokenizer`
        """
        super(FznTokenizer, self).__init__(**args)

        # Add predefined symbols
        self.add_predefined_symbols(PREDEFINED_SYMBOL_TOKENS)

        # Add character handlers
        self.add_char_handler('%', self._read_percent)
        self.add_char_handler('.', self._read_dot)
        self.add_char_handler(':', self._read_colon)
        self.add_char_handler('+', self._read_number)
        self.add_char_handler('-', self._read_number)


    def _read_percent(self):
        """ Read token starting by % (comment) """
        self._skip_to_end_of_line()
        return Token(TOKEN_TYPE_COMMENT, None if self.skip_comments else self._get_token()[:-1])


    def _read_dot(self):
        """ Read token starting by dot """
        if self._next_char() != '.':
            raise SyntaxError(self.build_error_string("Unknown token '.'"))
        return TOKEN_INTERVAL


    def _read_colon(self):
        """ Read token starting by colon """
        if self._peek_char() == ':':
            self._skip_and_peek_char()
            return TOKEN_DOUBLECOLON
        return TOKEN_COLON


