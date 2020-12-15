# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Parser converting a FZN file to internal model representation.

This parser does not support the complete set of predicates described in the specifications of FlatZinc
that can be found here: http://www.minizinc.org/downloads/doc-1.6/flatzinc-spec.pdf

Basically, it supports essentially integer expressions, some floating point expressions and custom
predicates related to scheduling.

The predicates that are supported are:

 * *array predicates*

   array_bool_and, array_bool_element, array_bool_or, array_bool_xor,
   array_float_element, array_int_element, array_set_element,
   array_var_bool_element, array_var_float_element, array_var_int_element, array_var_set_element.

 * *boolean predicates*

   bool2int, bool_and, bool_clause, bool_eq, bool_eq_reif, bool_le, bool_le_reif,
   bool_lin_eq, bool_lin_le, bool_lt, bool_lt_reif, bool_not, bool_or, bool_xor.

 * *integer predicates*

   int_abs, int_div, int_eq, int_eq_reif, int_le, int_le_reif, int_lin_eq, int_lin_eq_reif,
   int_lin_le, int_lin_le_reif, int_lin_ne, int_lin_ne_reif, int_lt, int_lt_reif, int_max, int_min,
   int_mod, int_ne, int_ne_reif, int_plus, int_times, int2float.

 * *float predicates*

   float_abs, float_exp, float_ln, float_log10, float_log2, float_sqrt, float_eq, float_eq_reif,
   float_le, float_le_reif, float_lin_eq, float_lin_eq_reif, float_lin_le, float_lin_le_reif, float_lin_lt,
   float_lin_lt_reif, float_lin_ne, float_lin_ne_reif, float_lt, float_lt_reif, float_max, float_min,
   float_ne, float_ne_reif, float_plus.

 * *set predicates*

   set_in, set_in_reif.

 * *custom predicates*

   all_different_int, subcircuit, count_eq_const, table_int, inverse,
   lex_lesseq_bool, lex_less_bool, lex_lesseq_int, lex_less_int, int_pow, cumulative


Detailed description
--------------------
"""

from docplex.cp.fzn.fzn_tokenizer import *
from docplex.cp.expression import *
from docplex.cp.solution import *
from docplex.cp.model import CpoModel
import docplex.cp.modeler as modeler
import docplex.cp.config as config
import docplex.cp.expression as expression
import collections
from docplex.cp.utils import xrange, is_int_value
import traceback


###############################################################################
## Constants
###############################################################################

###############################################################################
## Public classes
###############################################################################

class FznParserException(CpoException):
    """ The base class for exceptions raised by the CPO parser
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(FznParserException, self).__init__(msg)

# Parameter descriptor
FznParameter = collections.namedtuple('FznParameter', ('name',   # Variable name
                                                       'type',   # Variable type (string)
                                                       'size',   # Array size (if array), None for single value
                                                       'value',  # Value
                                                       ))

class FznObject(object):
    """ Descriptor of a FZN object
    """
    __slots__ = ()


class FznParameter(FznObject):
    """ Descriptor of a FZN parameter
    """
    __slots__ = ('name',   # Parameter name
                 'type',   # Parameter type
                 'size',   # Array size (if array), None for variable
                 'value',  # Initial value (if any)
                 )
    def __init__(self, name, type, size, value):
        """ Create a new FZN parameter
        Args:
            name:   Name of the parameter
            type:   Type of the parameter
            size:   Array size, None if not array
            value:  Parameter value
        """
        self.name = name
        self.type = type
        self.size = size
        self.value = value

    def __str__(self):
        lstr = [self.name, "(type=", str(self.type)]
        if self.size:
            lstr.append(', size=')
            lstr.append(str(self.size))
            lstr.append(', value=[')
            lstr.append(', '.join(str(x) for x in self.value))
            lstr.append(']')
        else:
            lstr.append(', value=')
            lstr.append(str(self.value))
        lstr.append(')')
        return ''.join(lstr)


class FznVariable(FznObject):
    """ Descriptor of a FZN variable
    """
    __slots__ = ('name',           # Variable name
                 'type',           # Variable type (String)
                 'domain',         # Domain
                 'size',           # Array size (if array), None for variable
                 'value',          # Initial value (if any)
                 'annotations',    # Dictionary of annotations

                 # Attributes needed for model reduction
                 'ref_vars',   # Tuple of variables referenced by this variable
                 )
    def __init__(self, name, type, domain, annotations, size, value):
        """ Create a new FZN variable
        Args:
            name:         Name of the variable
            type:         Variable type
            domain:       Variable domain
            annotations:  Declaration annotations (dictionary)
            size:         Array size, None if not array
            value:        Initial value, None if none
        """
        self.name = name
        self.type = type
        self.domain = domain
        self.annotations = annotations
        self.size = size
        self.value = value

    def is_defined(self):
        """ Check if the variable is introduced
        Return:
            True if variable is introduced, False otherwise
        """
        return 'is_defined_var' in self.annotations

    def is_introduced(self):
        """ Check if the variable is introduced
        Return:
            True if variable is introduced, False otherwise
        """
        return 'var_is_introduced' in self.annotations

    def is_output(self):
        """ Check if the variable is introduced
        Return:
            True if variable is introduced, False otherwise
        """
        return ('output_var' in self.annotations) or ('output_array' in self.annotations)

    def _get_domain_bounds(self):
        """ Get the variable domain bounds
        Return:
            Tuple of values, or single value if identical
        """
        dmin = self.domain[0]
        dmin = dmin[0] if isinstance(dmin, tuple) else dmin
        dmax = self.domain[-1]
        dmax = dmax[-1] if isinstance(dmax, tuple) else dmax
        return (dmin, dmax)

    def __str__(self):
        lstr = [self.name, "(type=", self.type, ", dom=", str(self.domain)]
        if self.is_defined():
            lstr.append(", defined")
        if self.is_introduced():
            lstr.append(", introduced")
        if self.size:
            if self.value:
                lstr.append(', value=[')
                for i, x in enumerate(self.value):
                    if i > 0:
                        lstr.append(', ')
                    if isinstance(x, tuple) and isinstance(x[0], FznVariable):
                        lstr.append("{}[{}]".format(x[0].name, x[1]))
                    elif isinstance(x, FznVariable):
                        lstr.append(x.name)
                    else:
                        lstr.append(str(x))
                lstr.append(']')
            else:
                lstr.append(", size={}".format(self.size))
        elif self.value:
            lstr.append(", value={}".format(self.value))
        lstr.append(')')
        return ''.join(lstr)


class FznConstraint(FznObject):
    """ Descriptor of a FZN constraint
    """
    __slots__ = ('predicate',  # Name of the predicate
                 'args',       # Arguments
                 'defvar',     # Name of the variable defined by this constraint

                 # Attributes needed for model reduction
                 'ref_vars',   # Tuple of variables referenced by this constraint, but not defined
                 )
    def __init__(self, predicate, args, annotations):
        """ Create a new FZN constraint
        Args:
            predicate:    Name of the predicate
            args:         List or arguments
            annotations:  Declaration annotations
        """
        self.predicate = predicate
        self.args = args
        self.defvar = annotations.get('defines_var', (None,))[0]
        self.ref_vars = ()

    def _ref_vars_iterator(self):
        """ Iterator on the variables that are referenced in the arguments of this constraint.
        Returns:
            Iterator on all variables referenced by this constraint
        """
        for a in self.args:
            if is_array(a):
                for v in a:
                    if isinstance(v, FznVariable):
                        yield v
            elif isinstance(a, FznVariable):
                yield a

    def __str__(self):
        lstr = [self.predicate, "("]
        for i, x in enumerate(self.args):
            if i > 0:
                lstr.append(', ')
            if isinstance(x, tuple) and isinstance(x[0], FznVariable):
                lstr.append("{}[{}]".format(x[0].name, x[1]))
            elif isinstance(x, (FznVariable, FznParameter)):
                lstr.append(x.name)
            elif isinstance(x, list):
                lstr.append("[{}]".format(', '.join(str(v) for v in x)))
            else:
                lstr.append(str(x))
        lstr.append(')')
        if self.defvar:
            lstr.append(":")
            lstr.append(self.defvar.name)
        return ''.join(lstr)


class FznObjective(FznObject):
    """ Descriptor of a FZN objective
    """
    __slots__ = ('operation',   # Objective operation in 'satisfy', 'minimize', 'maximize'
                 'expr',        # Target expression
                 'annotations', # Annotations
                )

    def __init__(self, operation, expr, annotations):
        """ Create a new FZN constraint
        Args:
            operation:   Objective operation in 'satisfy', 'minimize', 'maximize'
            expr:        Target expression
            annotations: Annotations
        """
        self.operation   = operation
        self.expr        = expr
        self.annotations = annotations

    def __str__(self):
        return "{} {} ({})".format(self.operation, self.expr, self.annotations)


class FznReader(object):
    """ Reader of FZN file format """
    __slots__ = ('source_file',  # Source file
                 'tokenizer',    # Reading tokenizer
                 'token',        # Last read token
                 'var_map',      # Dictionary of variables.
                                 # Key is variable name, value is variable descriptor
                 'parameters',   # List of parameters
                 'variables',    # List of variables
                 'constraints',  # List of model constraints
                 'objective',    # Model objective
                 )

    def __init__(self, mdl=None):
        """ Create a new FZN reader
        """
        super(FznReader, self).__init__()
        self.source_file = None
        self.tokenizer = None
        self.token = None
        self.var_map = {}
        self.parameters = []
        self.variables = []
        self.constraints = []
        self.objective = None


    def parse(self, cfile):
        """ Parse a FZN file

        Args:
            cfile: FZN file to read
        Raises:
            FznParserException: Parsing exception
        """
        # Store file name if first file
        self.source_file = cfile
        self.tokenizer = FznTokenizer(file=cfile)
        self._read_document()
        self.tokenizer = None


    def parse_string(self, str):
        """ Parse a string

        Result of the parsing is added to the current result model.

        Args:
            str: String to parse
        """
        self.tokenizer = FznTokenizer(input=str)
        self._read_document()
        self.tokenizer = None


    def write(self, out=None):
        """ Write the model.

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        Args:
            out (Optional): Target output stream or file name. If not given, default value is sys.stdout.
        """
        # Check file
        if is_string(out):
            with open_utf8(os.path.abspath(out), mode='w') as f:
                self.write(f)
                return
        # Check default output
        if out is None:
            out = sys.stdout

        # Write model content
        for x in self.parameters:
            print(str(x))
        for x in self.variables:
            print(str(x))
        for x in self.constraints:
            print(str(x))
        out.flush()


    def _read_document(self):
        """ Read all FZN document
        """
        try:
            self._next_token()
            while self._read_predicate():
                pass
            while self._read_parameter_or_variable():
                pass
            while self._read_constraint():
                pass
            self._read_objective()
        except Exception as e:
            if isinstance(e, FznParserException):
                raise e
            if config.context.log_exceptions:
                traceback.print_exc()
            self._raise_exception(str(e))

        if self.token is not TOKEN_EOF:
            self._raise_exception("Unexpected token '{}'".format(self.token))


    def _read_predicate(self):
        """ Read a predicate declaration

        This function is called with first token already read and terminates with next token already read.

        Returns:
            True if a predicate has been read, False if nothing to process
        """
        if self.token is not TOKEN_KEYWORD_PREDICATE:
            return False

        # Read predicate declaration
        while self.token not in (TOKEN_SEMICOLON, TOKEN_EOF):
            self._next_token()
        if self.token is not TOKEN_SEMICOLON:
            self._raise_exception("Semicolon ';' expected at the end of a predicate declaration.")
        self._next_token()
        return True


    def _read_parameter_or_variable(self):
        """ Read a parameter or variable declaration

        This function is called with first token already read and terminates with next token already read.

        Returns:
            True if a parameter has been read, False if nothing to process
        """
        tok = self.token
        if tok.type is not TOKEN_TYPE_KEYWORD:
            return False

        # Read array size if any
        arsize = self._read_array_size()

        # Check if variable declaration
        tok = self.token
        if tok is TOKEN_KEYWORD_VAR:
            self._next_token()
            return self._read_variable(arsize)

        # Check type name
        if tok not in (TOKEN_KEYWORD_BOOL, TOKEN_KEYWORD_FLOAT, TOKEN_KEYWORD_INT, TOKEN_KEYWORD_SET):
            return False
        typ = tok
        if typ is TOKEN_KEYWORD_SET:
            self._check_token(self._next_token(), TOKEN_KEYWORD_OF)
            self._check_token(self._next_token(), TOKEN_KEYWORD_INT)

        # Check separating colon
        self._check_token(self._next_token(), TOKEN_COLON)

        # Check parameter name
        tok = self._next_token()
        if tok.type is not TOKEN_TYPE_SYMBOL:
            self._raise_exception("Symbol expected as parameter name.")
        pid = tok.value
        self._check_token(self._next_token(), TOKEN_ASSIGN)

        # Read expression
        self._next_token()
        expr = self._read_expression()
        if arsize:
            expr = list(expression._domain_iterator(expr))
        if typ is TOKEN_KEYWORD_SET:
            arsize = len(expr)

        self._check_token(self.token, TOKEN_SEMICOLON)
        self._next_token()

        # Build result
        fp = FznParameter(pid, typ.value, arsize, expr)
        self.var_map[pid] = fp
        self.parameters.append(fp)

        return True


    def _read_variable(self, arsize):
        """ Read a variable declaration

        This function is called with first token already read and terminates with next token already read.

        Args:
            arsize:  Array size if any
        Returns:
            True if a variable has been read, False if nothing to process
        """
        # Read type and domain
        typ, dom = self._read_var_domain()
        self._check_token(self.token, TOKEN_COLON)
        tok = self._next_token()
        if tok.type is not TOKEN_TYPE_SYMBOL:
            self._raise_exception("Symbol expected as variable name.")
        vid = tok.value
        tok = self._next_token()

        # Check annotations
        annotations = self._read_annotations()
        # print("Annotations: {}".format(annotations))

        # Check expression
        expr = None
        if self.token is TOKEN_ASSIGN:
            self._next_token()
            expr = self._read_expression()

        # Read ending semicolon
        self._check_token(self.token, TOKEN_SEMICOLON)
        self._next_token()

        # Create variable
        fv = FznVariable(vid, typ.value, dom, annotations, arsize, expr)
        self.var_map[vid] = fv
        self.variables.append(fv)

        return True


    def _read_constraint(self):
        """ Read a constraint

        This function is called with first token already read and terminates with next token already read.

        Returns:
            True if a variable has been read, False if nothing to process
        """
        # Check constraint token
        if self.token is not TOKEN_KEYWORD_CONSTRAINT:
            return False

        # Read constraint name
        tok = self._next_token()
        if tok.type is not TOKEN_TYPE_SYMBOL:
            self._raise_exception("Constraint name '{}' should be a symbol.".format(tok))
        cname = tok.value

        # Read parameters
        args = []
        self._check_token(self._next_token(), TOKEN_PARENT_OPEN)
        self._next_token()
        while self.token is not TOKEN_PARENT_CLOSE:
            args.append(self._read_expression())
            if self.token is TOKEN_COMMA:
                self._next_token()
        self._next_token()

        # Check annotations
        annotations = self._read_annotations()
        defvar = annotations.get('defines_var', (None,))[0]

        # Read ending semicolon
        self._check_token(self.token, TOKEN_SEMICOLON)
        self._next_token()

        # Store constraint
        self.constraints.append(FznConstraint(cname, args, annotations))

        return True


    def _read_objective(self):
        """ Read solve objective

        This function is called with first token already read and terminates with next token already read.

        Returns:
            True if a variable has been read, False if nothing to process
        """
        # Check constraint token
        if self.token is not TOKEN_KEYWORD_SOLVE:
            return False
        self._next_token()

        # Check annotations
        annotations = self._read_annotations()

        # Read solve objective
        tok = self.token
        if (tok not in (TOKEN_KEYWORD_SATISFY, TOKEN_KEYWORD_MINIMIZE, TOKEN_KEYWORD_MAXIMIZE)):
            self._raise_exception(
                "Solve objective '{}' should be a symbol in 'satisfy', 'minimize', 'maximize'.".format(tok))
        obj = tok
        self._next_token()

        # Read expression if any
        expr = None if obj is TOKEN_KEYWORD_SATISFY else self._read_expression()

        # Read ending semicolon
        self._check_token(self.token, TOKEN_SEMICOLON)
        self._next_token()

        # Store objective
        self.objective = FznObjective(obj.value, expr, annotations)

        return True


    def _read_expression(self):
        """ Read an expression

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Expression that has been read
        """
        tok = self.token
        self._next_token()

        # Check int constant
        if tok.type is TOKEN_TYPE_INTEGER:
            v1 = int(tok.value)
            # Check set const
            if self.token is TOKEN_INTERVAL:
                tok2 = self._next_token()
                if tok2.type is not TOKEN_TYPE_INTEGER:
                    self._raise_exception("Set upper bound {} should be an integer constant.".format(tok2))
                self._next_token()
                v2 = int(tok2.value)
                return (v1,) if v1 == v2 else (v1, v2)
            else:
                return v1

        # Check float constant
        if tok.type is TOKEN_TYPE_FLOAT:
            return float(tok.value)

        # Set of integer constant
        if tok is TOKEN_BRACE_OPEN:
            lints = []
            tok = self.token
            while tok is not TOKEN_BRACE_CLOSE:
                if tok.type is not TOKEN_TYPE_INTEGER:
                    self._raise_exception("Set element {} should be an integer constant.".format(tok))
                lints.append(int(tok.value))
                tok = self._next_token()
                if tok is TOKEN_COMMA:
                    tok = self._next_token()
            self._next_token()
            return lints

        # Check symbols
        if tok.type is TOKEN_TYPE_SYMBOL:
            sid = tok.value
            # Check array access
            if self.token is TOKEN_HOOK_OPEN:
                # Access corresponding FZN element
                elem = self.var_map.get(sid)
                if elem is None:
                    self._raise_exception("Unknown symbol '{}'.".format(sid))
                tok2 = self._next_token()
                if tok2.type is not TOKEN_TYPE_INTEGER:
                    self._raise_exception("Array index '{}' should be an integer constant.".format(tok2))
                self._check_token(self._next_token(), TOKEN_HOOK_CLOSE)
                self._next_token()
                # Build array access as a tuple (arr_name, index)
                return (elem, int(tok2.value))
            # Check annotation function call
            elif self.token is TOKEN_PARENT_OPEN:
                lexprs = [sid]
                self._next_token()
                while self.token is not TOKEN_PARENT_CLOSE:
                    lexprs.append(self._read_expression())
                    if self.token is TOKEN_COMMA:
                        self._next_token()
                self._next_token()
                return tuple(lexprs)
            else:
                # Check if corresponds to FZN element
                return self.var_map.get(sid, sid)

        # Array of expressions
        if tok is TOKEN_HOOK_OPEN:
            lexprs = []
            while self.token is not TOKEN_HOOK_CLOSE:
                lexprs.append(self._read_expression())
                if self.token is TOKEN_COMMA:
                    self._next_token()
            self._next_token()
            return lexprs

        # Check boolean constant
        if tok is TOKEN_KEYWORD_TRUE:
            return True
        if tok is TOKEN_KEYWORD_FALSE:
            return False

        # Unknown
        self._raise_exception("Invalid expression start: '{}'.".format(tok))


    def _read_array_size(self):
        """ Read an array size declaration

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Array size as int if given,
            -1 if size is not precised,
            None if no array specified
        """
        # Check array token
        if self.token is not TOKEN_KEYWORD_ARRAY:
            return None

        # Read array specs
        self._check_token(self._next_token(), TOKEN_HOOK_OPEN)
        tok = self._next_token()
        if tok is TOKEN_KEYWORD_INT:
            arsize = -1
        else:
            if (tok.type is not TOKEN_TYPE_INTEGER) and tok.value != '1' :
                self._raise_exception("Array size should start by  '1'")
            self._check_token(self._next_token(), TOKEN_INTERVAL)
            tok = self._next_token()
            if tok.type is not TOKEN_TYPE_INTEGER:
                self._raise_exception("Array size '{}' should be integer.".format(tok))
            arsize = int(tok.value)
        self._check_token(self._next_token(), TOKEN_HOOK_CLOSE)
        self._check_token(self._next_token(), TOKEN_KEYWORD_OF)
        self._next_token()

        return arsize


    def _read_var_domain(self):
        """ Read the domain of a variable.

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Type token and variable domain
        """
        # Get token
        tok = self.token
        typ = tok

        # Check boolean domain
        if typ is TOKEN_KEYWORD_BOOL:
            self._next_token()
            return typ, BINARY_DOMAIN

        # Check undefined domain
        if typ is TOKEN_KEYWORD_INT:
            self._next_token()
            return typ, DEFAULT_INTEGER_VARIABLE_DOMAIN

        # Read set of integers or interval
        if tok is TOKEN_BRACE_OPEN:
            lint = sorted(self._read_expression())
            dom = []
            llen = len(lint)
            i = 0
            while i < llen:
                j = i + 1
                while (j < llen) and (lint[j] == lint[j - 1] + 1):
                    j += 1
                if (j > i + 1):
                    dom.append((lint[i], lint[j - 1]))
                else:
                    dom.append(lint[i])
                i = j
            return TOKEN_KEYWORD_INT, tuple(dom)

        # Check integer domain
        if tok.type is not TOKEN_TYPE_INTEGER:
            self._raise_exception("Variable domain should start by an integer constant.")
        self._next_token()
        if self.token is TOKEN_INTERVAL:
            tok2 = self._next_token()
            if tok2.type is not TOKEN_TYPE_INTEGER:
                self._raise_exception("Domain upper bound {} should be an integer constant.".format(tok2))
            self._next_token()
            v1 = int(tok.value)
            v2 = int(tok2.value)
            if v1 == v2:
                return TOKEN_KEYWORD_INT, (v1,)
            return TOKEN_KEYWORD_INT, ((v1, v2),)
        else:
            return TOKEN_KEYWORD_INT, (int(tok.value),)


    def _read_annotations(self):
        """ Read a list of annotations

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Dictionary of annotations. Key is name, value is tuple or parameters.
        """
        result = {}

        # Check annotation start token
        while self.token is TOKEN_DOUBLECOLON:
            # Read annotation name
            anm = self._next_token()
            if anm.type is not TOKEN_TYPE_SYMBOL:
                self._raise_exception("Annotation name '{}' should be a symbol.".format(anm))
            args = []
            tok = self._next_token()
            if tok is TOKEN_PARENT_OPEN:
                self._next_token()
                while self.token is not TOKEN_PARENT_CLOSE:
                    args.append(self._read_expression())
                    if self.token is TOKEN_COMMA:
                        self._next_token()
                self._next_token()
            result[anm.value] = tuple(args)

        return result


    def _next_token(self):
        """ Read next token
        Returns:
            Next read token, None if end of input
        """
        self.token = self.tokenizer.next_token()
        # print("Line {}, col {}, tok '{}'".format(self.tokenizer.line_number, self.tokenizer.read_index, self.token))
        return self.token


    def _check_token(self, tok, etok):
        """ Check that a read token is a given one an raise an exception if not
        Args:
            tok: Read token
            etok: Expected token
        """
        if tok is not etok:
            self._raise_exception("Read token '{}' instead of expected '{}'".format(tok, etok))


    def _raise_exception(self, msg):
        """ Raise a Parsing exception
        Args:
            msg:  Exception message
        """
        raise FznParserException(self.tokenizer.build_error_string(msg))



class FznParser(object):
    """ Reader of FZN file format """
    __slots__ = ('model',          # Read model
                 'compiled',       # Model compiled indicator
                 'reader',         # FZN reader
                 'cpo_exprs',      # Dictionary of CPO expressions. Key=name, value=CPO expr
                 'reduce',         # Reduce model indicator
                 'interval_gen',   # Name generator for interval var expressions
                 'cumul_gen',      # Name generator for cumul atom expressions

                 'parameters',     # List of parameters
                 'variables',      # List of variables
                 'constraints',    # List of model constraints
                 'objective',      # Model objective

                 'cur_constraint', # Currently compiled constraint descriptor

                 'def_var_exprs',  # List of expressions waiting for defvars to be defined
                 'cpo_variables',  # Set of names of variables that are translated as real CPO variables
                 )

    def __init__(self, mdl=None):
        """ Create a new FZN format parser

        Args:
            mdl:  Model to fill, None (default) to create a new one.
        """
        super(FznParser, self).__init__()
        self.model = mdl if mdl is not None else CpoModel()
        self.compiled = False
        self.reader = FznReader()
        self.interval_gen = IdAllocator("IntervalVar_")
        self.cumul_gen = IdAllocator("VarCumulAtom_")

        # Do not store location information (would store parser instead of real lines)
        self.model.source_loc = False

        # Set model reduction indicator
        self.reduce = config.context.parser.fzn_reduce


    def get_model(self):
        """ Get the model that have been parsed

        Return:
            CpoModel result of the parsing
        """
        if not self.compiled:
            self.compiled = True
            self._compile_to_model()
        return self.model


    def parse(self, cfile):
        """ Parse a FZN file

        Args:
            cfile: FZN file to read
        Raises:
            FznParserException: Parsing exception
        """
        if self.model.source_file is None:
            self.model.source_file = cfile
        self.reader.parse(cfile)


    def parse_string(self, str):
        """ Parse a string

        Result of the parsing is added to the current result model.

        Args:
            str: String to parse
        """
        self.reader.parse_string(str)


    def get_output_variables(self):
        """ Get the list of model output variables

        Returns:
            List of output variables, in declaration order.
        """
        return [v for v in self.variables if v.is_output()]


    def _write_model(self, out=None):
        """ Print read model (short version)
        Args:
            out (optional): Output stream. Default is stdout
        """
        if out is None:
            out = sys.stdout
        out.write(self.get_model().get_cpo_string(short_output=True))
        out.write("\n")


    def _get_cpo_expr_map(self):
        """ For testing, get the map of CPO expressions
        """
        self.get_model()
        return self.cpo_exprs


    def _compile_to_model(self):
        """ Compile FZN model into CPO model
        """
        # Initialize processing
        self.cpo_exprs = {}
        self.parameters  = self.reader.parameters
        self.variables   = self.reader.variables
        self.constraints = self.reader.constraints
        self.objective   = self.reader.objective
        self.def_var_exprs = {}
        self.cpo_variables = set()

        # Reduce model if required
        if self.reduce:
            self._reduce_model()

        # print("=== Variables:")
        # for v in self.variables:
        #     print(" : {}".format(v))
        # print("=== Constraints:")
        # for c in self.constraints:
        #     print(" : {}".format(c))
        # sys.stdout.flush()

        # Compile parameters
        for x in self.parameters:
            self._compile_parameter(x)

        # Compile variables
        for x in self.variables:
            self._compile_variable(x)

        # Compile constraints
        if self.reduce:
            for x in self.constraints:
                if isinstance(x, FznVariable):
                    self._compile_variable(x)
                else:
                    self._compile_constraint(x)
        else:
            for x in self.constraints:
                self._compile_constraint(x)

        # Compile objective
        self._compile_objective(self.objective)


    def _compile_parameter(self, fp):
        """ Compile a FZN parameter into CPO model
        Args:
            fp: Flatzinc parameter, object of class FznParameter
        """
        if fp.type in ('int', 'bool'):
            expr = CpoValue(fp.value, Type_IntArray if fp.size else Type_Int)
        elif fp.type == 'float':
            expr = CpoValue(fp.value, Type_FloatArray if fp.size else Type_Float)
        else:
            expr = build_cpo_expr(fp.value)

        # Add to map
        expr.set_name(fp.name)
        self.cpo_exprs[fp.name] = expr


    def _compile_variable(self, fv):
        """ Compile a FZN variable into CPO model
        Args:
            fv: Flatzinc variable
        """
        # Check if variable is array
        val = fv.value
        if fv.size:
            # Build array of variables
            if val:
                # Check if there is a reference to a not yet defined variable
                if self.reduce:
                    for v in val:
                        if v in self.def_var_exprs:
                            self.def_var_exprs[v].append(fv)
                            return
                arr = [self._get_cpo_expr(e) for e in val]
                expr = CpoValue(arr, Type_IntVarArray if all(x.type == Type_IntVar for x in arr) else Type_IntExprArray)
            else:
                # Build array of variables
                arr = [integer_var(name=fv.name + '[' + str(i + 1) + ']', domain=fv.domain) for i in range(fv.size)]
                expr = CpoValue(arr, Type_IntVarArray)
        else:
            # Build single variable
            if self.reduce and val:
                if is_int(val):
                    expr = CpoValue(val, Type_Int)
                elif is_bool(val):
                    expr = CpoValue(val, Type_Bool)
                else:
                    expr = self._get_cpo_expr(val)
            else:
                # Check if value is another variable
                if isinstance(val, FznVariable):
                    # Retrieve existing variable
                    expr = self.cpo_exprs.get(val.name)
                    assert isinstance(expr, CpoIntVar), "Variable '{}' not found".format(val.name)
                else:
                    # Create new variable
                    dom = _build_domain(val) if val else fv.domain
                    expr = integer_var(domain=dom)
        expr.set_name(fv.name)
        self.cpo_exprs[fv.name] = expr


    def _compile_constraint(self, fc):
        """ Compile a FZN constraint into CPO model
        Args:
            fv: Flatzinc constraint
        """
        # Search in local methods
        cmeth = getattr(self, "_compile_pred_" + fc.predicate, None)
        if not cmeth:
            raise FznParserException("Predicate '{}' is not supported.".format(fc.predicate))

        # Call compile method
        cmeth(fc)


    def _compile_objective(self, fo):
        """ Compile a FZN objective into CPO model
        Args:
            fo: Flatzinc objective
        """
        #print("Compile objective {}".format(fo))
        if fo is None:
            return
        if fo.operation != 'satisfy':
            expr = self._get_cpo_expr(fo.expr)
            oxpr = modeler.maximize(expr) if fo.operation == 'maximize' else modeler.minimize(expr)
            self._add_to_model(oxpr)


    def _reduce_model(self):
        """ Reduce model size by factorizing expressions when possible
        """
        # Access main model elements
        variables = self.variables
        constraints = self.constraints

        # Build reduction data related to variables
        for fv in variables:
            # Build list of variables that are referenced by this one
            fv.ref_vars = tuple(v for v in fv.value if isinstance(v, FznVariable)) if fv.size else ()
            # Set in defined variables if output
            if fv.is_output():
                self.cpo_variables.add(fv.name)

        # In constraints, replace reference to arrays by arrays themselves
        for fc in constraints:
            fc.args = tuple(a.value if isinstance(a, FznVariable) and a.size else a for a in fc.args)

        # Initialize set of variables defined in constraints
        def_var_map = {}      # Key is variable, value is constraint where variable is defined
        for fc in constraints:
            # print("Scan constraint {}".format(fc))
            defvar = fc.defvar
            if defvar is not None:
                def_var_map[defvar] = fc
            # Build list of all variables referenced by this constraint
            res = set()
            nbrefdvar = 0
            for v in fc._ref_vars_iterator():
                if v is defvar:
                    nbrefdvar += 1
                else:
                    res.add(v)
            fc.ref_vars = tuple(res)
            # Special case for cumulative. All variables are supposed defined by the constraint.
            if fc.predicate == 'cumulative':
                for v in fc.ref_vars:
                    def_var_map[v] = fc
            # print("   result list of ref variables: {}".format(fc.ref_vars))
            # If defined variable is referenced twice in the constraint, remove it as defined (keep as declared variable)
            if nbrefdvar > 1:
                fc.defvar = None
                self.cpo_variables.add(defvar.name)

        # Determine connected variable subsets
        #self._determine_connex_variables(def_var_map)

        # Scan variables to move them after definition of their dependencies when needed
        variables = []  # New list of variables
        for fv in self.variables:
            #print("Scan variable {}, Refvars: {}".format(fv, [v.name for v in fv.ref_vars]))
            if any(v in def_var_map for v in fv.ref_vars):
                if self._insert_in_constraints(fv, 0, def_var_map):
                    # Remove from list of variables (moved in constraints)
                    pass
                else:
                    # Keep it as a model variable
                    self.cpo_variables.add(fv.name)
                    variables.append(fv)
            else:
                variables.append(fv)
        self.variables = variables

        # Reorder constraints
        #print("\nScan constraints. Defined vars: {}".format([v.name for v in defined_vars]))
        constraints = self.constraints
        nbct = len(constraints)
        movedcstr = set() # Constraints already moved
        cx = 0
        while cx < nbct:
            fc = constraints[cx]
            #print("Scan constraint {}. Refvars: {}".format(fc, [v.name for v in fc.ref_vars]))
            #print("Defined vars: {}".format([v.name for v in defined_vars]))
            # Process case of variable that has been inserted in constraints
            if isinstance(fc, FznVariable):
                def_var_map.pop(fc, None)
            else:
                # Search if constraint can be moved
                if (fc not in movedcstr) and (fc.predicate != "cumulative") \
                        and any(v in def_var_map for v in fc.ref_vars) \
                        and self._insert_in_constraints(fc, cx + 1, def_var_map):
                    # Move constraint after all is defined
                    del constraints[cx]
                    movedcstr.add(fc)
                    cx -= 1
                else:
                    # Constraint stays where it is
                    if fc.defvar:
                        def_var_map.pop(fc.defvar, None)
            cx += 1

        # print("Reduction ended.")
        # print("   Variables:")
        # for v in self.variables:
        #     print("      {}".format(v))
        # print("   Constraints:")
        # for c in self.constraints:
        #     print("      {}".format(c))


    def _insert_in_constraints(self, fc, cx, varsctsr):
        """ Insert a constraint or a variable in constraints after all its members are defined
        Args:
            fc:         FZN constraint or variable to insert
            cx:         Start insertion index
            varsctsr:   Map of constraints where each variable is defined
        Return:
            True if insertion was successful, False otherwise
        """
        # Build set of constraints to skip to wait for variable definition
        cset = set(varsctsr[v] for v in fc.ref_vars if v in varsctsr)

        # Check no dependency
        if not cset:
            return False

        # Search in next constraints
        constraints = self.constraints
        for ix in xrange(cx, len(constraints)):
            cset.discard(constraints[ix])
            if not cset:
                # Insert after this place
                constraints.insert(ix + 1, fc)
                # print("   inserted at best rank {}".format(cx + 1))
                return True

        # Impossible to insert
        # print("   impossible to insert")
        return False


    def _determine_connex_variables(self, def_var_map):
        """ Search connex sub-graphes in variables dependencies and put all multi-variables connex parts as cpo variables
        Args:
            def_var_map: Map of constraints where variable is defined
        """

        # Create result map. Key is variable descriptor, value is set of connected variable descriptors
        lcomps = {}

        # For each variable, build the list of depending variables
        for fv in self.variables:
            stack = [fv]
            dset = set(stack)
            while stack:
                sv = stack.pop(0)
                for v in sv.ref_vars:
                    if v not in dset:
                        if v in lcomps:
                            dset.update(lcomps[v])
                        else:
                            dset.add(v)
                            stack.append(v)
                fc = def_var_map.get(fv)
                if fc:
                    for v in fc.ref_vars:
                        if v not in dset:
                            #print("   cstr.ref: {}".format(v))
                            if v in lcomps:
                                dset.update(lcomps[v])
                            else:
                                dset.add(v)
                                stack.append(v)
            lcomps[fv] = dset

        # Sort all variables list in ascending cardinality order
        #print("Variables graphs:")
        #for k, v in lcomps.items():
        #    print("   {}: {}".format(k.name, [x.name for x in v]))
        lcomps = sorted(lcomps.values(), key=lambda x: len(x))

        # Remove each set in next ones
        nbcomps = len(lcomps)
        for x1 in range(nbcomps):
            s1 = lcomps[x1]
            if s1:
                for x2 in range(x1 + 1, nbcomps):
                    lcomps[x2] = lcomps[x2].difference(s1)

        # Set as model variables all that are in component with at least two variables
        for c in lcomps:
            #print("   graph component: {}".format([x.name for x in c]))
            if len(c) > 1:
                for v in c:
                    self.cpo_variables.add(v.name)

        #print("CPO model variables identified: {}".format([v for v in self.cpo_variables]))


    def _get_cpo_expr(self, expr):
        """ Retrieve a CPO expression from its FZN representation
        Args:
            expr:  FZN expression
        Returns:
            Corresponding CPO expression
        """

        #print("   _get_cpo_expr({}, type={})".format(expr, type(expr)))

        # Check basic types
        etyp = type(expr)
        if etyp in (FznVariable, FznParameter):
            v = self.cpo_exprs.get(expr.name)
            if v is None:
                raise FznParserException("Can not find element {}".format(expr.name))
            return v

        # Integer constant
        if etyp in INTEGER_TYPES:
            return CpoValue(expr, Type_Int)

        # Array
        if etyp is list:
            return build_cpo_expr([self._get_cpo_expr(x) for x in expr])

        # Check array access
        if etyp is tuple:
            # Check tuple of integers
            if is_int(expr[0]):
                if len(expr) > 1:
                    return [i for i in range(expr[0], expr[1] + 1)]
                return [expr[0]]
            # Access to array element
            arr = self.cpo_exprs.get(expr[0].name)
            if arr is None:
                raise FznParserException("Can not find array {}".format(expr[0]))
            return(arr.value[expr[1]-1])

        # Boolean
        if etyp in BOOL_TYPES:
            return CpoValue(1, Type_Bool) if expr else CpoValue(0, Type_Bool)

        # String
        if etyp in STRING_TYPES:
            v = self.cpo_exprs.get(expr)
            if v is None:
                raise FznParserException("Can not find element {}".format(expr))
            return v

        # Unknown
        raise FznParserException("Can not find element {}".format(expr))

    # Array predicates

    def _compile_pred_array_bool_element(self, fc):
        self._compile_array_xxx_element(fc)

    def _compile_pred_array_int_element(self, fc):
        self._compile_array_xxx_element(fc)

    def _compile_pred_array_float_element(self, fc):
        self._compile_array_xxx_element(fc)

    def _compile_pred_array_var_bool_element(self, fc):
        self._compile_array_xxx_element(fc)

    def _compile_pred_array_var_int_element(self, fc):
        self._compile_array_xxx_element(fc)

    def _compile_pred_array_var_float_element(self, fc):
        self._compile_array_xxx_element(fc)

    def _compile_pred_array_bool_and(self, fc):
        self._compile_op_assign_arg_1(fc, modeler.min_of)

    def _compile_pred_array_bool_or(self, fc):
        self._compile_op_assign_arg_1(fc, modeler.max_of)

    # Bool predicates

    def _compile_pred_bool_and(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.logical_and)

    def _compile_pred_bool_or(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.logical_or)

    def _compile_pred_bool_xor(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.diff)

    def _compile_pred_bool_not(self, fc):
        self._compile_op_assign_arg_1(fc, modeler.logical_not)

    def _compile_pred_bool_eq(self, fc):
        self._compile_xxx_eq(fc)

    def _compile_pred_bool_eq_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.equal)

    def _compile_pred_bool_le(self, fc):
        self._compile_op_arg_2(fc, modeler.less_or_equal)

    def _compile_pred_bool_le_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.less_or_equal)

    def _compile_pred_bool_lin_eq(self, fc):
        self._compile_scal_prod(fc, modeler.equal)

    def _compile_pred_bool_lin_le(self, fc):
        self._compile_scal_prod(fc, modeler.less_or_equal)

    def _compile_pred_bool_lt(self, fc):
        self._compile_op_arg_2(fc, modeler.less)

    def _compile_pred_bool_lt_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.less)

    def _compile_pred_bool2int(self, fc):
        self._compile_xxx_eq(fc)

    # Float predicates

    def _compile_pred_float_abs(self, fc):
        self._compile_op_assign_arg_1(fc, modeler.abs_of)

    def _compile_pred_float_exp(self, fc):
        self._compile_op_assign_arg_1(fc, modeler.exponent)

    def _compile_pred_float_ln(self, fc):
        self._compile_op_assign_arg_1(fc, modeler.log)

    def _compile_pred_float_log10(self, fc):
        a, r = fc.args
        self._make_equal(r, modeler.log(self._get_cpo_expr(a)) / modeler.log(10), fc.defvar)

    def _compile_pred_float_log2(self, fc):
        a, r = fc.args
        self._make_equal(r, modeler.log(self._get_cpo_expr(a)) / modeler.log(2), fc.defvar)

    def _compile_pred_float_sqrt(self, fc):
        a, r = fc.args
        self._make_equal(r, modeler.power(self._get_cpo_expr(a), 0.5), fc.defvar)

    def _compile_pred_float_eq(self, fc):
        self._compile_xxx_eq(fc)

    def _compile_pred_float_eq_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.equal)

    def _compile_pred_float_le(self, fc):
        self._compile_op_arg_2(fc, modeler.less_or_equal)

    def _compile_pred_float_le_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.less_or_equal)

    def _compile_pred_float_lin_eq(self, fc):
        self._compile_scal_prod(fc, modeler.equal)

    def _compile_pred_float_lin_le(self, fc):
        self._compile_scal_prod(fc, modeler.less_or_equal)

    def _compile_pred_float_lin_lt(self, fc):
        self._compile_scal_prod(fc, modeler.lesst)

    def _compile_pred_float_lin_ne(self, fc):
        self._compile_scal_prod(fc, modeler.diff)

    def _compile_pred_float_lin_eq_reif(self, fc):
        self._compile_scal_prod(fc, modeler.equal, True)

    def _compile_pred_float_lin_le_reif(self, fc):
        self._compile_scal_prod(fc, modeler.less_or_equal, True)

    def _compile_pred_float_lin_lt_reif(self, fc):
        self._compile_scal_prod(fc, modeler.less, True)

    def _compile_pred_float_lin_ne_reif(self, fc):
        self._compile_scal_prod(fc, modeler.diff, True)

    def _compile_pred_float_lt(self, fc):
        self._compile_op_arg_2(fc, modeler.less)

    def _compile_pred_float_lt_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.less)

    def _compile_pred_float_max(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.max_of)

    def _compile_pred_float_min(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.min_of)

    def _compile_pred_float_ne(self, fc):
        self._compile_op_arg_2(fc, modeler.diff)

    def _compile_pred_float_ne_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.diff)

    def _compile_pred_float_plus(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.plus)

    # Int predicates

    def _compile_pred_int_abs(self, fc):
        self._compile_op_assign_arg_1(fc, modeler.abs_of)

    def _compile_pred_int_div(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.int_div)

    def _compile_pred_int_eq(self, fc):
        self._compile_xxx_eq(fc)

    def _compile_pred_int_eq_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.equal)

    def _compile_pred_int_le(self, fc):
        self._compile_op_arg_2(fc, modeler.less_or_equal)

    def _compile_pred_int_le_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.less_or_equal)

    def _compile_pred_int_lin_eq(self, fc):
        self._compile_scal_prod(fc, modeler.equal)

    def _compile_pred_int_lin_le(self, fc):
        self._compile_scal_prod(fc, modeler.less_or_equal)

    def _compile_pred_int_lin_ne(self, fc):
        self._compile_scal_prod(fc, modeler.diff)

    def _compile_pred_int_lin_eq_reif(self, fc):
        self._compile_scal_prod(fc, modeler.equal, True)

    def _compile_pred_int_lin_le_reif(self, fc):
        self._compile_scal_prod(fc, modeler.less_or_equal, True)

    def _compile_pred_int_lin_ne_reif(self, fc):
        self._compile_scal_prod(fc, modeler.diff, True)

    def _compile_pred_int_lt(self, fc):
        self._compile_op_arg_2(fc, modeler.less)

    def _compile_pred_int_lt_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.less)

    def _compile_pred_int_max(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.max_of)

    def _compile_pred_int_min(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.min_of)

    def _compile_pred_int_mod(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.mod)

    def _compile_pred_int_ne(self, fc):
        self._compile_op_arg_2(fc, modeler.diff)

    def _compile_pred_int_ne_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.diff)

    def _compile_pred_int_plus(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.plus)

    def _compile_pred_int_times(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.times)

    def _compile_pred_int2float(self, fc):
        self._compile_xxx_eq(fc)


    # Set predicates

    def _compile_pred_set_in(self, fc):
        self._compile_op_arg_2(fc, modeler.allowed_assignments)

    def _compile_pred_set_in_reif(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.allowed_assignments)


    # Custom predicates

    def _compile_pred_all_different_int(self, fc):
        self._compile_op_arg_1(fc, modeler.all_diff)

    def _compile_pred_count_eq_const(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.count)

    def _compile_pred_lex_lesseq_int(self, fc):
        self._compile_op_arg_2(fc, modeler.lexicographic)

    def _compile_pred_lex_lesseq_bool(self, fc):
        self._compile_op_arg_2(fc, modeler.lexicographic)

    def _compile_pred_int_pow(self, fc):
        self._compile_op_assign_arg_2(fc, modeler.power)

    def _compile_pred_cumulative(self, fc):
        """ Requires that a set of tasks given by start times s, durations d, and resource requirements r,
        never require more than a global resource bound b at any one time.
        Args:
            fc:  Constraint descriptor, with arguments
                stime:  Tasks start time
                tdur:   Tasks tasks durations
                rreq:   Task resource requirements
                bnd:    Global resource bound
        """
        #print("Process cumulative constraint {}".format(fc))

        # Access constraint arguments
        stime, tdur, rreq, bnd = fc.args

        # Create interval vars and cumul atoms
        ls = _get_fzn_array(stime)
        ld = _get_fzn_array(tdur)
        lr = _get_fzn_array(rreq)
        #print("ls: {}\nld: {}\nlr: {}".format(ls, [s for s in ld], lr))
        cumul_atoms = []
        for s, d, r in zip(ls, ld, lr):
            vname = None
            # Get start time
            if isinstance(s, FznVariable):
                ds = s._get_domain_bounds()
                vname = s.name
            else:
                ds = (s, s)
            # Get duration
            if isinstance(d, FznVariable):
                dd = d._get_domain_bounds()
                if vname is None:
                    vname = d.name
            else:
                dd = (d, d)
            # Get requirement
            if isinstance(r, FznVariable):
                dr = r._get_domain_bounds()
                if vname is None:
                    vname = r.name
            else:
                dr = (r, r)
            #print("ds: {}, dd: {}, dr: {}".format(ds, dd, dr))

            # Create interval variable
            if vname is None:
                vname = self.interval_gen.allocate()
            else:
                vname = "Itv_" + vname
                if vname in self.cpo_exprs:
                    cnt = 1
                    nname = vname + "_1"
                    while nname in self.cpo_exprs:
                        cnt += 1
                        nname = vname + "_" + str(cnt)
                    vname = nname
            # Create interval variable
            ivar = self.cpo_exprs.get(vname)
            if ivar:
                # Check it is the same
                assert isinstance(ivar, CpoIntervalVar) and ivar.get_start() == ds and ivar.get_size() == dd and ivar.get_end() == (INTERVAL_MIN, INTERVAL_MAX)
            else:
                ivar = interval_var(start=ds, end=(INTERVAL_MIN, INTERVAL_MAX), size=dd, name=vname)
                self.cpo_exprs[vname] = ivar

            # Create pulse
            pulse = modeler.pulse(ivar, dr)
            cumul_atoms.append(pulse)
            # Replace previous variable by access to interval variable
            if isinstance(s, FznVariable):
                self._assign_to_var(s, modeler.start_of(ivar))
            if isinstance(d, FznVariable):
                self._assign_to_var(d, modeler.size_of(ivar))
            if isinstance(r, FznVariable):
                self._assign_to_var(r, modeler.height_at_start(ivar, pulse))

        # Create final constraint
        cumf = CpoFunctionCall(Oper_sum, Type_CumulFunction, (CpoValue(cumul_atoms, Type_CumulAtomArray),))
        self._add_to_model(modeler.greater_or_equal(self._get_cpo_expr(bnd), cumf))


    def _compile_pred_subcircuit(self, fc):
        """ Constrains the elements of x to define a subcircuit where x[i] = j means that j is the successor of i and x[i] = i means that i is not in the circuit.
        """
        x = [0] + list(self._get_cpo_expr(fc.args[0]).value)
        self._add_to_model(CpoFunctionCall(Oper__sub_circuit, Type_Constraint, (build_cpo_expr(x),) ))


    def _compile_pred_inverse(self, fc):
        """ Constrains two arrays of int variables, f and invf, to represent inverse functions. All the values in each array must be within the index set of the other array.
        Args:
            f:     First function as array of int
            invf:  Inverse function
        """
        f, invf = fc.args
        f = [0] + list(self._get_cpo_expr(f).value)
        invf = [0] + list(self._get_cpo_expr(invf).value)
        self._add_to_model(CpoFunctionCall(Oper_inverse, Type_Constraint, (build_cpo_expr(f), build_cpo_expr(invf))))


    def _compile_pred_bool_clause(self, fc):
        """ Implementation of bool_clause predicate
        Args:
            a:  First array of booleans
            b:  Second array of booleans
        """
        a, b = fc.args
        # Default implementation
        exprs = list(self._get_cpo_expr(a).value)
        for x in self._get_cpo_expr(b).value:
            exprs.append(1 - x)
        self._add_to_model(modeler.sum_of(exprs) >= 1)

        # Alternative implementation
        # self._add_to_model( (modeler.max_of(a) > 0) | (modeler.min_of(b) == 0) )

        # Other alternative implementation
        # expr = None
        # for x in _get_value(a):
        #     x = x > 0
        #     expr = x if expr is None else modeler.logical_or(expr, x)
        # for x in _get_value(b):
        #     x = x == 0
        #     expr = x if expr is None else modeler.logical_or(expr, x)
        # self._add_to_model(expr)


    def _compile_pred_table_int(self, fc):
        """ Implement custom predicate table_int
        Args:
            vars:    Array of variables
            values:  List of values
        """
        vars, values = fc.args
        # Split value array in tuples
        vars = self._get_cpo_expr(vars).value
        tsize = len(vars)
        if tsize != 0:
            values = self._get_cpo_expr(values).value
            tuples = [values[i: i + tsize] for i in range(0, len(values), tsize)]
            # Build allowed assignment expression
            self._add_to_model(modeler.allowed_assignments(vars, tuples))


    def _compile_pred_lex_less_bool(self, fc):
        """ Requires that the array vars1 is strictly lexicographically less than array vars2
        Args:
            vars1:  First array of variables
            vars2:  Second array of variables
        """
        vars1, vars2 = fc.args
        # Add 0 and 1 at the end of arrays to force inequality
        vars1 = list(self._get_cpo_expr(vars1).value) + [1]
        vars2 = list(self._get_cpo_expr(vars2).value) + [0]
        self._add_to_model(modeler.lexicographic(vars1, vars2))


    def _compile_pred_lex_less_int(self, fc):
        self._compile_pred_lex_less_bool(fc)


    def _get_domain_bounds(self, x):
        """ Get min and max bounds of an expression, integer variable or integer
        Args:
            expr: CPO integer variable or expression
        Returns:
            Tuple (min, max)
        """
        # Case of variable or variable replaced by an expression
        if isinstance(x, CpoIntVar):
            return (x.get_domain_min(), x.get_domain_max())
        if isinstance(x, CpoFunctionCall):
            if x.operation is Oper_start_of:
                return x.children[0].get_start()
            if x.operation is Oper_size_of:
                return x.children[0].get_size()
            # if x.is_kind_of(Type_BoolExpr):
            #     return (0, 1)
            cpov = self.cpo_exprs.get(x.name)
            if isinstance(cpov, CpoIntVar):
                return (cpov.get_domain_min(), cpov.get_domain_max())
            raise FznParserException("Unknow expression to take bounds from: {}".format(x))
        if isinstance(x, CpoValue):
            x = x.value
        if is_int(x):
            return (x, x)

        return x


    def _assign_to_var(self, var, expr):
        """ Set a identifier with an expression
        Args:
            var:  Target FZN variable
            expr: CPO expression to assign
        Returns:
            None. If needed, changes are done in reader context.
        """
        #print("_assign_to_var {} expression {}".format(var, expr))

        # Retrieve existing expression
        vname = var.name
        vexpr = self.cpo_exprs.get(vname)

        # Check if reduction
        if (not self.reduce) or (vname in self.cpo_variables) or (vexpr is not None and not isinstance(vexpr, CpoIntVar)):
            self._add_to_model(modeler.equal(vexpr, expr))
        else:
            # Assign new name to expression
            self.cpo_exprs[vname] = expr
            expr.set_name(vname)
            # Constrain expression to variable domain
            self._constrain_expr_domain(expr, var)


    def _make_equal(self, var, expr, dvar):
        """ Make equal two FZN expressions
        Args:
            var:   FZN variable or value
            expr:  CPO expression to be equal with
            dvar:  Define variable, None if none
        Returns:
            None. If needed, changes are done in reader context.
        """

        # Check if var is not a variable
        if not isinstance(var, FznVariable):
            self._add_to_model(modeler.equal(expr, self._get_cpo_expr(var)))
            return

        # Check no reduction
        if not self.reduce or (dvar is not var):
            self._add_to_model(modeler.equal(self._get_cpo_expr(var), expr))
            return

        # Assign to variable
        self._assign_to_var(var, expr)


    def _constrain_expr_domain(self, expr, var):
        """ Constrain the domain of an expression to the domain of a variable
        Args:
            expr: CPO expression to constrain
            var:  CPO or FZN integer var to take domain from
        """
        dom = var.domain
        #print("   constrain expression {} to domain {}".format(expr, dom))
        dmin = expression._domain_min(dom)
        dmax = expression._domain_max(dom)

        # Check boolean expression
        if dmin == 0 and dmax == 1:
            return
        if expr.is_kind_of(Type_BoolExpr) and (dmin <= 0) and (dmax >= 1):
            return

        # Add appropriate constraint
        if len(dom) == 1:  # Single segment
            # Use range
            if dmin == dmax:
                self._add_to_model(modeler.equal(expr, dmin))
            else:
                self._add_to_model(modeler.range(expr, dmin, dmax))
        else:
            # Use allowed assignment
            self._add_to_model(modeler.allowed_assignments(expr, expression._domain_iterator(dom)))


    def _compile_op_assign_arg_1(self, fc, op):
        """ Compile operation with single argument equal to a result
        Args:
            fc:  Constraint descriptor
            op:  CPO operation to apply to first argument
        """
        # Access constraint arguments
        a, r = fc.args
        # Assign to expression
        self._make_equal(r, op(self._get_cpo_expr(a)), fc.defvar)


    def _compile_op_assign_arg_2(self, fc, op):
        """ Compile operation with two arguments equal to a result
        Args:
            fc:  Constraint descriptor
            op:  CPO operation to apply to arguments
        """
        # Access constraint arguments
        a, b, r = fc.args
        # Assign to expression
        self._make_equal(r, op(self._get_cpo_expr(a), self._get_cpo_expr(b)), fc.defvar)


    def _compile_op_arg_1(self, fc, op):
        """ Compile operation with one arguments and no result
        Args:
            fc:  Constraint descriptor
            op:  CPO operation to apply to argument
        """
        self._add_to_model(op(self._get_cpo_expr(fc.args[0])))


    def _compile_op_arg_2(self, fc, op):
        """ Compile operation with two arguments and no result
        Args:
            fc:  Constraint descriptor
            op:  CPO operation to apply to arguments
        """
        a, b = fc.args
        self._add_to_model(op(self._get_cpo_expr(a), self._get_cpo_expr(b)))


    def _compile_array_xxx_element(self, fc):
        """ Compile access to array element """
        x, t, r = fc.args
        if not is_int(x):
            x = self._get_cpo_expr(x)
        self._make_equal(r, modeler.element(self._get_cpo_expr(t), x - 1), fc.defvar)


    def _compile_xxx_eq(self, fc):
        """ Compile all equality predicates """

        # Access constraint arguments
        a, b = fc.args

        # Check default case
        if not self.reduce:
            self._add_to_model(modeler.equal(self._get_cpo_expr(a), self._get_cpo_expr(b)))
            return

        # Process trivial cases
        if a is b:
            return None

        # Retrieve defined variable
        defvar = fc.defvar
        if defvar is None:
            self._add_to_model(modeler.equal(self._get_cpo_expr(a), self._get_cpo_expr(b)))
            return

        if defvar is a:
            self._assign_to_var(a, self._get_cpo_expr(b))
        else:
            self._assign_to_var(b, self._get_cpo_expr(a))


    def _compile_scal_prod(self, fc, op, reif=False):
        """ Compile a scalar product
        Args:
            fc:  Constraint
            op:  Comparison operation
        """
        # Access constraint arguments
        if reif:
            coefs, vars, res, reif = fc.args
        else:
            coefs, vars, res = fc.args

        # Check no reduction
        defvar = fc.defvar
        if not self.reduce or not defvar:
            expr = op(modeler.scal_prod(self._get_cpo_expr(coefs), self._get_cpo_expr(vars)), self._get_cpo_expr(res))
            if reif:
                expr = modeler.equal(self._get_cpo_expr(reif), expr)
            self._add_to_model(expr)
            return

        # Get array elements
        coefs = _get_fzn_array(coefs)
        vars = _get_fzn_array(vars)

        # Check if defined variable is the result
        if defvar is res or defvar is reif:
            expr = self._build_scal_prod_expr(coefs, vars, 0)
        else:
            # Arrange expression to have defined variable on the left
            vx = vars.index(defvar)
            vcoef = coefs[vx]
            vars = vars[:vx] + vars[vx + 1:]
            coefs = coefs[:vx] + coefs[vx + 1:]
            expr = res if is_int(res) else self._get_cpo_expr(res)
            if vcoef < 0:
                vcoef = -vcoef
                expr = -expr
            else:
                coefs = list([-c for c in coefs])

            # Build result
            expr = self._build_scal_prod_expr(coefs, vars, expr)
            if vcoef != 1:
                expr = expr / vcoef

        # Check reif
        if reif:
            expr = op(expr, self._get_cpo_expr(res))
            self._assign_to_var(defvar, expr)
        else:
            # Check equality with a variable
            if op is modeler.equal:
                self._assign_to_var(defvar, expr)
            else:
                self._add_to_model(op(expr, self._get_cpo_expr(defvar)))


    def _build_scal_prod_expr(self, coefs, vars, res):
        """ Build a scal prod expression
        Args:
            coefs:  Array of coefficients (integers)
            vars:   Array of FZN variables
            res:    Initial result value (integer)
        Returns:
            New CPO scal_prod expression
        """
        # Build array of CPO variables
        vars = [self._get_cpo_expr(v) for v in vars]

        # Check developed scal_prod
        if len(coefs) <= 2 or (all(c == 1 or c == -1 for c in coefs)):
            for c, v in zip(coefs, vars):
                if c != 0:
                    if is_int_value(res, 0):
                        res = _mutl_by_int(v, c)
                    elif c < 0:
                        res = res - _mutl_by_int(v, -c)
                    else:
                        res = res + _mutl_by_int(v, c)
            return res

        # Build normal scal_prod
        expr = modeler.scal_prod(coefs, vars)
        if not is_int_value(res, 0):
            expr = res + expr
        return expr


    def _add_to_model(self, expr):
        """ Add an expression to the CPO model
        Args:
            expr: CPO expression to add
        """
        #print("_add_to_model({})".format(expr))
        self.model.add(expr)
        
        # Scan expression to identify used variables
        estack = [expr]
        doneset = set()  # Set of expressions already processed
        while estack:
            e = estack.pop()
            eid = id(e)
            if not eid in doneset:
                doneset.add(eid)
                if e.type.is_variable:
                    #print("   add CPO variable {}".format(e))
                    self.cpo_variables.add(e.name)
                # Stack children expressions
                estack.extend(e.children)


    def _write(self, out=None):
        """ Write current parser status
        Args:
            out (optional):  Write output. sys.stdout if not given
        """
        if out is None:
            out = sys.stdout
        out.write("Reader status:\n")
        out.write("   CPO expressions:\n")
        for k in sorted(self.cpo_exprs.keys()):
            v = self.cpo_exprs[k]
            out.write("      {}: {} ({})\n".format(k, v, type(v)))
        out.write("   Model expressions:\n")
        for x in self.model.get_all_expressions():
            out.write("      {}\n".format(x[0]))


###############################################################################
## Utility functions
###############################################################################

def _get_fzn_array(fzo):
    """ Get the list of objects of an FZN object
    Args:
        fzo: FZN object
    Returns:
        Array of FZN objects
    """
    return fzo if isinstance(fzo, list) else fzo.value


def _get_value(expr):
    """ Get the python value of an expression
    Args:
        expr: Expression (python or CPO)
    Returns:
        Python value
    """
    return expr.value if isinstance(expr, CpoValue) else expr


def _build_domain(v):
    """ Build a variable domain from an initial value
    Args:
        v: Variable initial value
    Returns:
        Variable domain
    """
    if v is True:
        return (1,)
    if v is False:
        return (0,)
    return (v,)


# def _build_cpo_value(v):
#     """ Build a value from a single Python value
#     Args:
#         v: Value
#     Returns:
#         Corresponding CPO value
#     """
#     if v is True:
#         v = 1
#     elif v is False:
#         v = 0
#     return build_cpo_expr(v)


def _mutl_by_int(expr, val):
    """ Create an expression that multiply an expression by an integer
    Args:
        expr: Expression to constrain
        val:  Integer value to multiply expression with
    Returns:
        New expression
    """
    if val == 1:
        return expr
    if val == -1:
        return -expr
    if val == 0:
        return 0
    return val * expr


