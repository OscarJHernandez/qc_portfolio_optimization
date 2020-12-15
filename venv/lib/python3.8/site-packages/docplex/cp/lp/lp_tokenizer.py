# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Tokenizer for reading LP file format
"""

from docplex.cp.tokenizer import *


#==============================================================================
#  Constants
#==============================================================================

_VARIABLE_START_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&(),;?@_'{}~")
_VARIABLE_BODY_CHARS  = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&(),;?@_'{}~0123456789.")

# Section start tokens
TOKEN_KEYWORD_MINIMIZE        = Token(TOKEN_TYPE_KEYWORD, "Minimize")
TOKEN_KEYWORD_MAXIMIZE        = Token(TOKEN_TYPE_KEYWORD, "Maximize")
TOKEN_KEYWORD_MINIMIZE_MULTI  = Token(TOKEN_TYPE_KEYWORD, "Minimize multi-objectives")
TOKEN_KEYWORD_MAXIMIZE_MULTI  = Token(TOKEN_TYPE_KEYWORD, "Maximize multi-objectives")
TOKEN_KEYWORD_SUBJECT_TO      = Token(TOKEN_TYPE_KEYWORD, "Subject to")
TOKEN_KEYWORD_BOUNDS          = Token(TOKEN_TYPE_KEYWORD, "Bounds")
TOKEN_KEYWORD_BINARY          = Token(TOKEN_TYPE_KEYWORD, "Binary")
TOKEN_KEYWORD_GENERAL         = Token(TOKEN_TYPE_KEYWORD, "General")
TOKEN_KEYWORD_NAMELIST        = Token(TOKEN_TYPE_KEYWORD, "Namelist")
TOKEN_KEYWORD_SEMI_CONTINUOUS = Token(TOKEN_TYPE_KEYWORD, "Semi-Continuous")
TOKEN_KEYWORD_SOS             = Token(TOKEN_TYPE_KEYWORD, "SOS")
TOKEN_KEYWORD_PWL             = Token(TOKEN_TYPE_KEYWORD, "PWL")
TOKEN_KEYWORD_END             = Token(TOKEN_TYPE_KEYWORD, "End")

# Map of different wording for sections. Key is section name in lower case, value is token.
_SECTION_NAME_MAP = {
    'minimize': TOKEN_KEYWORD_MINIMIZE, 'minimum':TOKEN_KEYWORD_MINIMIZE, 'min':TOKEN_KEYWORD_MINIMIZE,
    'maximize': TOKEN_KEYWORD_MAXIMIZE, 'maximum': TOKEN_KEYWORD_MAXIMIZE, 'max': TOKEN_KEYWORD_MAXIMIZE,
    'minimize multi-objectives': TOKEN_KEYWORD_MINIMIZE_MULTI,
    'maximize multi-objectives': TOKEN_KEYWORD_MAXIMIZE_MULTI,
    'subject to': TOKEN_KEYWORD_SUBJECT_TO, 'such that': TOKEN_KEYWORD_SUBJECT_TO, 'st': TOKEN_KEYWORD_SUBJECT_TO, 's.t.': TOKEN_KEYWORD_SUBJECT_TO, 'st.': TOKEN_KEYWORD_SUBJECT_TO,
    'user cuts': TOKEN_KEYWORD_SUBJECT_TO, 'lazy constraints': TOKEN_KEYWORD_SUBJECT_TO, # Not documented
    'bounds': TOKEN_KEYWORD_BOUNDS, 'bound': TOKEN_KEYWORD_BOUNDS,
    'binary': TOKEN_KEYWORD_BINARY, 'binaries': TOKEN_KEYWORD_BINARY, 'bin': TOKEN_KEYWORD_BINARY,
    'general': TOKEN_KEYWORD_GENERAL, 'generals': TOKEN_KEYWORD_GENERAL, 'gen': TOKEN_KEYWORD_GENERAL,
    'namelist': TOKEN_KEYWORD_NAMELIST,
    'integers': TOKEN_KEYWORD_GENERAL, 'integer': TOKEN_KEYWORD_GENERAL,  # Not documented, but present in examples
    'semi-continuous': TOKEN_KEYWORD_SEMI_CONTINUOUS, 'semi': TOKEN_KEYWORD_SEMI_CONTINUOUS, 'semis': TOKEN_KEYWORD_SEMI_CONTINUOUS,
    'sos': TOKEN_KEYWORD_SOS,
    'pwl': TOKEN_KEYWORD_PWL,
    'end': TOKEN_KEYWORD_END
}

# Specific tokens
TOKEN_IMPLY       = Token(TOKEN_TYPE_OPERATOR, '->')
TOKEN_IMPLY_BACK  = Token(TOKEN_TYPE_OPERATOR, '<-')
TOKEN_EQUIV       = Token(TOKEN_TYPE_OPERATOR, '<->')
TOKEN_DOUBLECOLON = Token(TOKEN_TYPE_PUNCTUATION, "::")


#==============================================================================
#  Public classes
#==============================================================================

class LpTokenizer(Tokenizer):
    """ Tokenizer for LP file format """
    __slots__ = ()

    def __init__(self, **args):
        """ Create a new tokenizer
        Args:
            See arguments list of :class:`~docplex.cp.utils.Tokenizer`
        """
        super(LpTokenizer, self).__init__(**args)

        # Clear all character handlers
        self.clear_all_char_handlers()

        # Add read of comments
        self.add_char_handler('\\', self._read_antislash)

        # Add symbol read handlers
        for c in _VARIABLE_START_CHARS:
            self.add_char_handler(c, self._read_symbol)

        # Add reading of numbers
        for c in "0123456789":
            self.add_char_handler(c, self._read_number)

        # Add character handlers
        self.add_char_handler('+', lambda: TOKEN_PLUS)
        self.add_char_handler('-', self._read_minus)
        self.add_char_handler('*', lambda: TOKEN_STAR)
        self.add_char_handler('/', lambda: TOKEN_SLASH)
        self.add_char_handler('^', lambda: TOKEN_CIRCUMFLEX)
        self.add_char_handler(':', self._read_colon)
        self.add_char_handler('=', self._read_equal)
        self.add_char_handler('[', lambda: TOKEN_HOOK_OPEN)
        self.add_char_handler(']', lambda: TOKEN_HOOK_CLOSE)
        self.add_char_handler('.', self._read_dot)

        # Add comparison handlers
        self.add_char_handler('>', self._read_greater)
        self.add_char_handler('<', self._read_lower)


    def set_pwl_mode(self, pwl):
        """ Set the tokenizer to parse PWL breakpoints

        Args:
            pwl:  Set PWL mode if true, or restore normal mode if false
        """
        if pwl:
            self.add_char_handler('(', lambda: TOKEN_PARENT_OPEN)
            self.add_char_handler(')', lambda: TOKEN_PARENT_CLOSE)
            self.add_char_handler(',', lambda: TOKEN_COMMA)
        else:
            for c in ('(', ')', ','):
                self.add_char_handler(c, self._read_symbol)

    def _read_minus(self):
        """ Read token starting by - """
        c = self._peek_char()
        if c == '>':
            self._skip_char()
            return TOKEN_IMPLY
        return TOKEN_MINUS


    def _read_equal(self):
        """ Read token starting by = """
        c = self._peek_char()
        if c == '>':
            self._skip_char()
            return TOKEN_GREATER_EQUAL
        if c == '<':
            self._skip_char()
            return TOKEN_LOWER_EQUAL
        return TOKEN_ASSIGN


    def _read_greater(self):
        """ Read token starting by > """
        if self._peek_char() == '=':
            self._skip_char()
            return TOKEN_GREATER_EQUAL
        return TOKEN_GREATER_EQUAL


    def _read_lower(self):
        """ Read token starting by < """
        c = self._peek_char()
        if c == '=':
            self._skip_char()
            return TOKEN_LOWER_EQUAL
        if c == '-':
            self._skip_char()
            if self._peek_char() == '>':
                self._skip_char()
                return TOKEN_EQUIV
            return TOKEN_IMPLY_BACK
        return TOKEN_LOWER_EQUAL


    def _read_symbol(self):
        """ Read and return next symbol """
        # Check section names
        if self.first_in_line:
            # Check undocumented comment
            if self.current_char == '%':
                self._skip_to_end_of_line()
                return Token(TOKEN_TYPE_COMMENT, None if self.skip_comments else self._get_token()[:-1])
            else:
                sname = self.line.strip(' \r\n\t').lower()
                tok = _SECTION_NAME_MAP.get(sname)
                if tok is not None:
                    # Skip all line
                    self.read_index = self.line_length
                    return tok
        c = self._current_char()
        while True:
            if c == '(':
                # Skip all characters up to ')'
                while c and (c != ')'):
                    c = self._next_char()
            c = self._peek_char()
            if c and ((c in _VARIABLE_BODY_CHARS) or (ord(c) > 127)):
                self._next_char()
            else:
                break

        # Check again section name for case where it begins a line that also contains an expression
        tval = self._get_token()
        if self.first_in_line and self._peek_char() <= ' ':
            tok = _SECTION_NAME_MAP.get(tval.lower())
            if tok is not None:
                return tok
        return Token(TOKEN_TYPE_SYMBOL, tval)


    def _read_antislash(self):
        """ Read token starting by \ """
        self._skip_to_end_of_line()
        return Token(TOKEN_TYPE_COMMENT, None if self.skip_comments else self._get_token()[:-1])


    def _read_colon(self):
        """ Read token starting by colon """
        if self._peek_char() == ':':
            self._skip_and_peek_char()
            return TOKEN_DOUBLECOLON
        return TOKEN_COLON


    def _read_dot(self):
        """ Read token starting by dot (particular case for floats) """
        self._back_char()
        return self._read_number()


