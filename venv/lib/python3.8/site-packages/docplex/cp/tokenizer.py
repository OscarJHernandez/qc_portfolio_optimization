# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Abstract tokenizer that can be extended to be specialized in different languages.

This tokenizer is already initialized to read integers, floats, strings and basic punctuation.
"""

from docplex.cp.utils import to_internal_string, is_string, StringIO, TextFileLineReader
import os


###############################################################################
## Utility classes
###############################################################################

class Token(object):
    """ Token returned by tokenizer  """
    __slots__ = ('type',   # Token type
                 'value',  # Token string value (with quotes for strings)
                 )

    def __init__(self, type, value):
        """ Create a new token
        Args:
            type:  Token type
            value: Token value
        """
        super(Token, self).__init__()
        self.type = type
        self.value = value

    def get_string(self):
        """ Get the string corresponding to the value, interpreting escape sequences if necessary
        Returns:
            Expanded string value
        """
        return self.value if self.type != TOKEN_TYPE_STRING else to_internal_string(self.value[1:-1])

    def __str__(self):
        """ Build a string representing this token
        Returns:
            String representing this token
        """
        return self.value

    def __eq__(self, other):
        """ Check if this token is equal to another object
        Args:
            other:  Object to compare with
        Returns:
            True if 'other' is a token with the same value then this one, False otherwise
        """
        return (other is self) or (isinstance(other, Token) and (other.type == self.type) and (other.value == self.value))

    def __ne__(self, other):
        """ Check if this token is different than another object
        Args:
            other:  Object to compare with
        Returns:
            True if 'other' is not a token or a different token than this one, False otherwise
        """
        return not self.__eq__(other)


###############################################################################
## Constants
###############################################################################

# Token types
TOKEN_TYPE_NONE        = "None"         # No token (for EOF)
TOKEN_TYPE_INTEGER     = "Integer"      # Integer number
TOKEN_TYPE_FLOAT       = "Float"        # Floating point number
TOKEN_TYPE_PUNCTUATION = "Punctuation"  # Ponctuation, like comma, colon, semicolon, brackets, etc
TOKEN_TYPE_OPERATOR    = "Operator"     # Operator
TOKEN_TYPE_KEYWORD     = "Keyword"      # Keyword (reserved symbol)
TOKEN_TYPE_SYMBOL      = "Symbol"       # Symbol (identifier)
TOKEN_TYPE_STRING      = "String"       # String inside double quotes, with Java/C++/Python escape characters
TOKEN_TYPE_VERSION     = "Version"      # Version number expressed as integers separated by dots
TOKEN_TYPE_COMMENT     = "Comment"      # Comment

# Predefined tokens
TOKEN_EOF = Token(TOKEN_TYPE_NONE, "EOF")

TOKEN_ASSIGN         = Token(TOKEN_TYPE_OPERATOR, "=")
TOKEN_EQUAL          = Token(TOKEN_TYPE_OPERATOR, "==")
TOKEN_DIFFERENT      = Token(TOKEN_TYPE_OPERATOR, "!=")
TOKEN_GREATER        = Token(TOKEN_TYPE_OPERATOR, ">")
TOKEN_GREATER_EQUAL  = Token(TOKEN_TYPE_OPERATOR, ">=")
TOKEN_LOWER          = Token(TOKEN_TYPE_OPERATOR, "<")
TOKEN_LOWER_EQUAL    = Token(TOKEN_TYPE_OPERATOR, "<=")

TOKEN_PLUS           = Token(TOKEN_TYPE_OPERATOR, "+")
TOKEN_MINUS          = Token(TOKEN_TYPE_OPERATOR, "-")
TOKEN_STAR           = Token(TOKEN_TYPE_OPERATOR, "*")
TOKEN_SLASH          = Token(TOKEN_TYPE_OPERATOR, "/")
TOKEN_PERCENT        = Token(TOKEN_TYPE_OPERATOR, "%")
TOKEN_CIRCUMFLEX     = Token(TOKEN_TYPE_OPERATOR, "^")

TOKEN_DOT            = Token(TOKEN_TYPE_PUNCTUATION, ".")
TOKEN_COMMA          = Token(TOKEN_TYPE_PUNCTUATION, ",")
TOKEN_COLON          = Token(TOKEN_TYPE_PUNCTUATION, ":")
TOKEN_SEMICOLON      = Token(TOKEN_TYPE_PUNCTUATION, ";")
TOKEN_HOOK_OPEN      = Token(TOKEN_TYPE_PUNCTUATION, "[")
TOKEN_HOOK_CLOSE     = Token(TOKEN_TYPE_PUNCTUATION, "]")
TOKEN_BRACE_OPEN     = Token(TOKEN_TYPE_PUNCTUATION, "{")
TOKEN_BRACE_CLOSE    = Token(TOKEN_TYPE_PUNCTUATION, "}")
TOKEN_PARENT_OPEN    = Token(TOKEN_TYPE_PUNCTUATION, "(")
TOKEN_PARENT_CLOSE   = Token(TOKEN_TYPE_PUNCTUATION, ")")

TOKEN_PIPE           = Token(TOKEN_TYPE_PUNCTUATION, "|")
TOKEN_HASH           = Token(TOKEN_TYPE_PUNCTUATION, '#')
TOKEN_BANG           = Token(TOKEN_TYPE_PUNCTUATION, '!')

TOKEN_KEYWORD_TRUE   = Token(TOKEN_TYPE_KEYWORD, "true")
TOKEN_KEYWORD_FALSE  = Token(TOKEN_TYPE_KEYWORD, "false")

_SYMBOL_START_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_SYMBOL_BODY_CHARS  = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789")

###############################################################################
## Public classes
###############################################################################

class Tokenizer(object):
    """ Tokenizer for CPO file format """
    __slots__ = ('name',                 # Input name (for error string build)
                 'input',                # Input stream
                 'line',                 # Current input line
                 'current_char',         # Last read character
                 'token_start',          # Index of token start in current line
                 'line_length',          # Current input line length
                 'read_index',           # Current read index in the line
                 'line_number',          # Current line number
                 'skip_comments',        # Skip comments indicator
                 'symbols',              # Map of symbols. Key is name, value is symbol token
                 'char_handlers',        # Map of functions processing characters
                 'default_char_handler', # Default character handler
                 'first_in_line',        # First token in line indicator
                 )

    def __init__(self, name=None, input=None, file=None, encoding='utf-8-sig'):
        """ Create a new tokenizer
        Args:
            name:     (Optional) Name of the stream
            input:    (Optional) Input stream or string
            file:     (Optional) Input file
            encoding: (Optional) Character encoding, utf-8 by default
        """
        super(Tokenizer, self).__init__()

        # Open file if any
        if file is not None:
            self.input = TextFileLineReader(file, encoding)
        # Or get input if any
        elif input is not None:
            if is_string(input):
                self.input = StringIO(input)
            else:
                self.input = input
        # Set name
        self.name = name
        if name is None:
            self.name = file

        # Initialize reading
        self.line = ""
        self.current_char = ''
        self.line_length = 0
        self.read_index = 0
        self.line_number = 0
        self.token_start = 0
        self.skip_comments = True
        self.default_char_handler = None

        # Initialize symbols table (allows use of 'is' instead of '==')
        self.symbols = {}

        # Initialize array of character processors
        self.char_handlers = [None] * 128
        for c in _SYMBOL_START_CHARS:
            self.add_char_handler(c, self._read_symbol)
        for c in "0123456789":
            self.add_char_handler(c, self._read_number)
        self.add_char_handler('"', self._read_string)
        self.add_char_handler('>', self._read_greater)
        self.add_char_handler('<', self._read_lower)
        self.add_char_handler('=', self._read_equal)
        self.add_char_handler('!', self._read_bang)

        self.add_char_handler('+', lambda: TOKEN_PLUS)
        self.add_char_handler('-', lambda: TOKEN_MINUS)
        self.add_char_handler('*', lambda: TOKEN_STAR)
        self.add_char_handler('/', lambda: TOKEN_SLASH)
        self.add_char_handler('%', lambda: TOKEN_PERCENT)
        self.add_char_handler('^', lambda: TOKEN_CIRCUMFLEX)
        self.add_char_handler(',', lambda: TOKEN_COMMA)
        self.add_char_handler('.', lambda: TOKEN_DOT)
        self.add_char_handler(':', lambda: TOKEN_COLON)
        self.add_char_handler(';', lambda: TOKEN_SEMICOLON)
        self.add_char_handler('(', lambda: TOKEN_PARENT_OPEN)
        self.add_char_handler(')', lambda: TOKEN_PARENT_CLOSE)
        self.add_char_handler('[', lambda: TOKEN_HOOK_OPEN)
        self.add_char_handler(']', lambda: TOKEN_HOOK_CLOSE)
        self.add_char_handler('{', lambda: TOKEN_BRACE_OPEN)
        self.add_char_handler('}', lambda: TOKEN_BRACE_CLOSE)
        self.add_char_handler('|', lambda: TOKEN_PIPE)
        self.add_char_handler('#', lambda: TOKEN_HASH)


    def add_predefined_symbols(self, ltoks):
        """ Add a list of predefined symbols.

        Tokenizer ensure that if the symbol is found, always the same token is returned.
        This allows to check a token using 'is' instead of '=='.

        Args:
            ltoks:  List of symbol tokens
        """
        for t in ltoks:
            self.symbols[t.value] = t


    def clear_all_char_handlers(self):
        """ Clear all character handlers.
        """
        self.char_handlers = [None] * 127


    def add_char_handler(self, c, chdl):
        """ Add a character handler.

        Character handlers are used to properly process a character read from the input stream.
        By default, this tokenizer process symbols, strings, integer, floats, version, and basic
        punctuation tokens that are defined.
        Character handlers should be added or overwritten to handle properly specific tokens, including comments.

        A character handler takes this tokenizer as parameters (self) and returns the read token.

        Args:
            c:     Character starting the token
            chdl:  Character handler
        """
        self.char_handlers[ord(c)] = chdl


    def set_default_char_handler(self, c, chdl):
        """ Set the default character handler.

        Default character handler is called if no character handler is found. By default, it raises an exception.

        Args:
            chdl:  Character handler
        """
        self.default_char_handler = chdl


    def set_skip_comment(self, skc):
        """ Set the skip comment indicator

        When skip comment is True (default), comment are ignored.
        When skip comment is False, comments are returned as TOKEN_COMMENT tokens.

        Args:
            skc:  Skip comment indicator
        """
        self.skip_comments = skc


    def next_token(self):
        """ Get the next token

        Returns:
            Next available token, TOKEN_EOF if end of input
        """
        # Set first in line indicator
        self.first_in_line = self.read_index == 0

        # Read token loop
        skc = self.skip_comments
        while True:
            # Skip separators
            while True:
                c = self._next_char()
                if c is None:
                    return TOKEN_EOF
                if c > ' ':
                    break

            # Reset current token
            self.token_start = self.read_index - 1

            # Retrieve character handler
            oc = ord(c)
            if oc < 128:
                chld = self.char_handlers[oc]
                if chld is None:
                    chld = self.default_char_handler
            else:
                chld = self.default_char_handler
            if chld is None:
                raise SyntaxError(self.build_error_string("No possible token starting by character '{}'".format(c)))

            # Call character handler
            try:
                tk = chld()
                if not skc or tk.type != TOKEN_TYPE_COMMENT:
                    return tk
            except Exception as err:
                raise SyntaxError(self.build_error_string("Error parsing token starting by '{}': {}".format(c, err)))


    def close(self):
        """ Close this tokenizer
        """
        self.input.close()
        self.line = None
        self.current_char = None


    def _read_symbol(self):
        """ Read and return next symbol """
        # Read symbol
        c = self._peek_char()
        while c and (c in _SYMBOL_BODY_CHARS):
            c = self._skip_and_peek_char()
        s = self._get_token()
        t = self.symbols.get(s)
        if not t:
            t = Token(TOKEN_TYPE_SYMBOL, s)
            self.symbols[s] = t
        return t


    def _read_number(self):
        """ Read and return next number """
        # Read number
        typ = TOKEN_TYPE_INTEGER
        c = self._peek_char()
        while c and (c >= '0') and (c <= '9'):
            c = self._skip_and_peek_char()
        if c == '.':
            c = self._skip_and_peek_char()
            if c == '.':
                # Case of '..' used to specify intervals
                self.read_index -= 1
                return Token(typ, self._get_token())
            typ = TOKEN_TYPE_FLOAT
            while c and (c >= '0') and (c <= '9'):
                c = self._skip_and_peek_char()
            if (c == 'e') or (c == 'E'):
                c = self._skip_and_peek_char()
                if (c == '-') or (c == '+'):
                    c = self._skip_and_peek_char()
                while c and (c >= '0') and (c <= '9'):
                    c = self._skip_and_peek_char()
            elif c == '.':
                typ = TOKEN_TYPE_VERSION
                while (c == '.') or ((c >= '0') and (c <= '9')):
                    c = self._skip_and_peek_char()
        elif (c == 'e') or (c == 'E'):
            typ = TOKEN_TYPE_FLOAT
            c = self._skip_and_peek_char()
            if (c == '-') or (c == '+'):
                c = self._skip_and_peek_char()
            while c and (c >= '0') and (c <= '9'):
                c = self._skip_and_peek_char()
        return Token(typ, self._get_token())


    def _read_string(self):
        """ Read and return next string """
        # Read character sequence
        res = ['"']
        c = ''
        while (c is not None) and (c != '"'):
            c = self._next_char()
            res.append(c)
            if c == '\\':
                res.append(self._next_char())
                c = ''
        if c is None:
            raise SyntaxError(self.build_error_string("String not ended before end of stream"))
        return Token(TOKEN_TYPE_STRING, ''.join(res))


    def _read_greater(self):
        """ Read token starting by > """
        if self._peek_char() == '=':
            self._skip_char()
            return TOKEN_GREATER_EQUAL
        return TOKEN_GREATER


    def _read_lower(self):
        """ Read token starting by < """
        if self._peek_char() == '=':
            self._skip_char()
            return TOKEN_LOWER_EQUAL
        return TOKEN_LOWER


    def _read_equal(self):
        """ Read token starting by = """
        if self._peek_char() == '=':
            self._skip_char()
            return TOKEN_EQUAL
        return TOKEN_ASSIGN


    def _read_bang(self):
        """ Read token starting by ! """
        if self._peek_char() == '=':
            self._skip_char()
            return TOKEN_DIFFERENT
        return TOKEN_BANG


    def _get_line_reminder(self):
        """ Get reminder of the line
        Returns:
            Line remainder content, without ending \n
        """
        start = self.read_index
        c = self._next_char()
        while c and (c != '\n'):
            c = self._next_char()
        return self.line[start:self.read_index-1]


    def _skip_to_end_of_line(self):
        """ Skip read index to end of line
        """
        self.read_index = self.line_length


    def _get_token(self):
        """ Get the last read token
        """
        return self.line[self.token_start:self.read_index]


    def _peek_char(self):
        """ Peek (but not get) next input character
        Returns:
            Next available character, None if end of input
        """
        # Check end of stream
        if not self.line:
            return None

        # Return current character
        return self.line[self.read_index] if self.read_index < self.line_length else '\n'


    def _skip_char(self):
        """ Skip next input character
        """
        self.read_index += 1


    def _skip_and_peek_char(self):
        """ Skip next input character and peek next one on the same line
        Returns:
            Next available character, \n if end of line
        """
        self.read_index += 1
        return self.line[self.read_index] if self.read_index < self.line_length else '\n'


    def _next_char(self):
        """ Get next input character
        This function sets the variable current_char with the returned character
        Returns:
            Next available character, None if end of input
        """
        # Check end of stream
        line = self.line
        if line is None:
            return None

        # Check end of line
        if self.read_index >= self.line_length:
            # Read next line and check end of file
            line = self.input.readline()
            self.line_number += 1
            self.line = line
            self.line_length = len(line)
            self.read_index = 0
            self.first_in_line = True
            # Check end of input
            if line == '':
                self.close()
                return None
        c = line[self.read_index]
        self.read_index += 1
        self.current_char = c
        return c


    def _back_char(self):
        """ Go back to previous character """
        if self.read_index > 0:
            self.read_index -= 1


    def _current_char(self):
        """ Get last read character
        Returns:
            Last read character, None if end of input
        """
        return self.current_char


    def build_error_string(self, msg):
        """ Build error string for exception
        """
        # return "Error in '{}' at line {} (\"{}\") index {}: {}".format(self.name, self.line_number, self.rline, self.read_index, msg)
        # Insert error token in current line
        rline = self.line[:self.read_index] + " ### " + self.line[self.read_index:].rstrip()
        ermsg = "Error in '{}' at line {}:{} : {} (\"{}\")".format(self.name, self.line_number, self.read_index, msg, rline)
        return ermsg


    def __iter__(self):
        """  Define tokenizer as an iterator """
        return self


    def next(self):
        """ For iteration of tokens, get the next available token.

        Returns:
            Next token, StopIteration if EOF
        """
        tok = self.next_token()
        if tok is TOKEN_EOF:
            raise StopIteration()
        return tok


    def __next__(self):
        """ Get the next available token (same as next() for compatibility with Python 3)

        Returns:
            Next token
        """
        return self.next()


