# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Parser converting a CPO file to internal model representation.
"""

from docplex.cp.cpo.cpo_tokenizer import *
from docplex.cp.expression import *
from docplex.cp.expression import _create_operation
from docplex.cp.function import *
from docplex.cp.catalog import *
from docplex.cp.solution import *
from docplex.cp.model import CpoModel
import docplex.cp.config as config
import math
import traceback


###############################################################################
## Constants
###############################################################################

# Minimum CPO format version number
MIN_CPO_VERSION_NUMBER = "12.6.0.0"

# Maximum CPO format version number
MAX_CPO_VERSION_NUMBER = "12.10.0.0"

# Map of all operators. Key is operator, value is list of corresponding operation descriptors
_ALL_OPERATORS = {}

# Map of all operations. Key is CPO operation name, value is list corresponding operation descriptor
_ALL_OPERATIONS = {}

# Initialization code
for op in ALL_OPERATIONS:
    if op.priority >= 0:
        _ALL_OPERATORS[op.keyword] = op
    _ALL_OPERATIONS[op.cpo_name] = op
_ALL_OPERATIONS['alldiff'] = Oper_all_diff
_ALL_OPERATIONS['allDiff'] = Oper_all_diff

# Known identifiers
_KNOWN_IDENTIFIERS = {"intmax": INT_MAX, "intmin": INT_MIN,
                      "inf": INFINITY,
                      "intervalmax": INTERVAL_MAX, "intervalmin": INTERVAL_MIN,
                      "no": False}

# Map of array types for each CPO name
_ARRAY_TYPES = {'intArray': Type_IntArray, 'floatArray': Type_FloatArray,
                'boolExprArray' : Type_BoolExprArray, 'intExprArray': Type_IntExprArray, 'floatExprArray': Type_FloatExprArray,
                'intervalVarArray': Type_IntervalVarArray, 'sequenceVarArray': Type_SequenceVarArray,
                'intValueSelectorArray': Type_IntValueSelectorArray, 'intVarSelectorArray': Type_IntVarSelectorArray,
                'tupleSet' : Type_TupleSet,
                '_cumulAtomArray' : Type_CumulAtomArray}

# Fake operator '..' to read intervals properly
_OPER_INTERVAL = CpoOperation("_interval", "_interval", "..", 9, (CpoSignature(Type_IntExprArray, (Type_IntExpr, Type_IntExpr)),))
_ALL_OPERATORS[".."] = _OPER_INTERVAL


###############################################################################
## Public classes
###############################################################################

class CpoParserException(CpoException):
    """ The base class for exceptions raised by the CPO parser
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(CpoParserException, self).__init__(msg)


class CpoUnsupportedFormatVersionException(CpoParserException):
    """ Exception raised when a version mismatch is detected
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(CpoUnsupportedFormatVersionException, self).__init__(msg)



class CpoParser(object):
    """ Reader of CPO file format """
    __slots__ = ('model',          # Read model
                 'expr_dict',      # Dictionary of expressions. Key is name, value is CPO expression
                 'source_file',    # Source file
                 'tokenizer',      # Reading tokenizer
                 'token',          # Last read token
                 'pushtoken',      # Pushed token
                 'fun_handlers',   # Special function handlers
                 'current_loc',    # Current source location
                 )
    
    def __init__(self, mdl=None):
        """ Create a new CPO format parser

        Args:
            mdl:  Model to fill, None (default) to create a new one.
        """
        super(CpoParser, self).__init__()
        self.model = mdl if mdl is not None else CpoModel()
        self.expr_dict = {}
        self.source_file = None
        self.tokenizer = None
        self.token = None
        self.pushtoken = None
        self.current_loc = None

        # Do not store location information (would store parser instead of real lines)
        self.model.source_loc = False

        # Initialize special function handlers
        self.fun_handlers = {
            'intVar':            self._read_fun_intVar,
            '_intVar':           self._read_fun_intVar,
            'boolVar':           self._read_fun_boolVar,
            '_boolVar':          self._read_fun_boolVar,
            'floatVar':          self._read_fun_floatVar,
            '_floatVar':         self._read_fun_floatVar,
            'intervalVar':       self._read_fun_intervalVar,
            'sequenceVar':       self._read_fun_sequenceVar,
            'stateFunction':     self._read_fun_stateFunction,
            'stepFunction':      self._read_fun_stepFunction,
            'segmentedFunction': self._read_fun_segmentedFunction,
            'transitionMatrix':  self._read_fun_transitionMatrix,
        }

        # TODO: parse and include source information ?


    def get_model(self):
        """ Get the model that have been parsed

        Return:
            Model result of the parsing, object of class :class:`~docplex.cp.model.CpoModel`
        """
        return self.model


    def parse(self, cfile):
        """ Parse a CPO file.

        Args:
            cfile: CPO file to read
        Return:
            Model result of the parsing, object of class :class:`~docplex.cp.model.CpoModel`
        Raises:
            CpoParserException: Parsing exception
            CpoVersionMismatchException: Read CPO format is not in [MIN_CPO_VERSION_NUMBER .. MAX_CPO_VERSION_NUMBER]
        """
        # Store file name if first file
        self.source_file = cfile
        if self.model.source_file is None:
            self.model.source_file = cfile

        self.tokenizer = CpoTokenizer(file=cfile)
        self._read_statement_list()
        self.tokenizer.close()
        self.tokenizer = None

        return self.model


    def parse_string(self, str):
        """ Parse a string.

        Result of the parsing is added to the current result model.

        Args:
            str: String to parse
        Return:
            Model result of the parsing, object of class :class:`~docplex.cp.model.CpoModel`
        """
        self.tokenizer = CpoTokenizer(input=str)
        self._read_statement_list()
        self.tokenizer.close()
        self.tokenizer = None
        return self.model


    def _read_statement_list(self):
        """ Read a list of statements

        This functions reads all statements up to end of input.
        """
        try:
            while self._read_statement():
                pass
        except Exception as e:
            #if isinstance(e, CpoParserException):
            #    raise e
            if config.context.log_exceptions:
                traceback.print_exc()
            self._raise_exception(str(e))
        return True


    def _read_statement(self):
        """ Read a statement or a section

        This functions reads the first token and exits with current token that is
        the last of the statement.

        Returns:
            True if something has been read, False if end of input
        """
        tok1 = self._next_token()
        if tok1 is TOKEN_EOF:
            return False
        tok2 = self._next_token()

        # Check obsolete let and set forms
        if tok1 in (TOKEN_LET, TOKEN_SET) and tok2.type is TOKEN_TYPE_SYMBOL:
            tok1 = tok2
            tok2 = self._next_token()

        if tok1 is TOKEN_HASH:
            self._read_directive()

        elif tok2 is TOKEN_ASSIGN:
            expr = self._read_assignment(tok1.get_string())
            # Add expression to the model if it is a variable
            if expr.is_variable():
                self.model._add_with_loc(expr, self.current_loc)

        elif tok2 is TOKEN_SEMICOLON:
            # Get existing expression and re-add it to the model
            expr = self.expr_dict.get(tok1.get_string())
            if expr is None:
                self._raise_exception("Expression '{}' not found in the model".format(tok1.get_string()))
            self.model._add_with_loc(expr, self.current_loc)

        elif tok2 is TOKEN_COLON:
            expr = self._read_assignment(tok1.get_string())
            self.model._add_with_loc(expr, self.current_loc)

        elif tok2 is TOKEN_BRACE_OPEN:
            self._read_section(tok1.value)

        else:
            # Read expression
            self._push_token(tok1)
            expr = self._read_expression()
            #print("Expression read in statement: Type: " + str(expr.type) + ", val=" + str(expr))
            self.model._add_with_loc(expr, self.current_loc)
            self._check_token(self.token, TOKEN_SEMICOLON)

        return True


    def _read_directive(self):
        """ Read a directive
        """
        name = self.token.value
        if name == "line":
            self._read_directive_line()

        elif name == "include":
            self._read_directive_include()

        else:
            self._raise_exception("Unknown directive '" + name + "'")


    def _read_directive_line(self):
        """ Read a line directive
        """
        # # Skip line
        # self.tokenizer._skip_to_end_of_line()
        tok = self._next_token()
        # Check line off
        if (tok.type == TOKEN_TYPE_SYMBOL) and tok.value == "off":
            self.current_loc = None
            return
        # Get line number
        if tok.type != TOKEN_TYPE_INTEGER:
            self._raise_exception("Line number should be an integer")
        lnum = int(tok.value)
        # Get optional source file name, string on the same line
        cline = self.tokenizer.line_number
        tok = self._next_token()
        if cline != self.tokenizer.line_number:
            self._push_token(tok)
            fname = None if self.current_loc is None else self.current_loc[0]
        else:
            if tok.type != TOKEN_TYPE_STRING:
               self._raise_exception("File name should be a string")
            fname = tok.get_string()
        self.current_loc = (fname, lnum)


    def _read_directive_include(self):
        """ Read a include directive
        """
        # Get file name
        fname = self._check_token_string(self._next_token())
        if (os.path.dirname(fname) == "") and (self.source_file is not None):
            fname = os.path.dirname(self.source_file) + "/" + fname
        # Push current context
        old_ctx = (self.source_file, self.tokenizer, self.token)
        # Parse file
        self.parse(fname)
        # Restore context
        self.source_file, self.tokenizer, self.token = old_ctx


    def _read_assignment(self, name):
        """ Read an assignment

        Args:
            name:  Assignment name
        Returns:
            Named expression
        """
        # Read expression
        tok = self._next_token()
        expr = self._read_expression()
        # Build CPO expression if needed
        expr = build_cpo_expr(expr)
        # Add name to the expression
        if expr.name:
            # Create an alias if needed
            if name != expr.name:
                expr = CpoAlias(expr, name)
                self.model.add(expr)
        else:
            expr.set_name(name)

        # Add expression to expressions dictionary
        self.expr_dict[name] = expr
        self._check_token(self.token, TOKEN_SEMICOLON)
        return expr


    def _read_interval_var(self):
        """ Read a interval_var declaration
        Returns with token set to the last unexpected token
        Returns:
            CpoIntervalVar variable expression
        """
        res = interval_var()
        while True:
            # Read argument name
            tok = self.token
            if tok.type is not TOKEN_TYPE_SYMBOL:
                return res
            aname = tok.value
            self._next_token()
            if aname == "present":
                res.set_present()
            elif aname == "absent":
                res.set_absent()
            elif aname == "optional":
                res.set_optional()
            else:
                self._check_token(self.token, TOKEN_ASSIGN)
                self._next_token()
                if aname in ("start", "end", "length", "size"):
                    # Read interval
                    intv = self._read_expression()
                    if isinstance(intv, int):
                        intv = (intv, intv)
                    elif not isinstance(intv, (list, tuple)):
                        self._raise_exception("'start', 'end', 'length' or 'size' should be an integer or an interval")
                    setattr(res, aname, intv)
                elif aname == "intensity":
                    res.set_intensity(self._read_expression())
                elif aname == "granularity":
                    self._check_token_integer(self.token)
                    res.set_granularity(int(self.token.value))
                    self._next_token()
                else:
                    self._raise_exception("Unknown IntervalVar attribute argument '" + aname + "'")
            # Read comma
            tok = self.token
            if tok.value == ',':
                tok = self._next_token()


    def _read_expression(self):
        """ Read an expression

        First expression token is already read.
        Function exits with current token following the last expression token

        Returns:
            Expression that has been read
        """
        # Read first sub-expression
        expr = self._read_sub_expression()
        tok = self.token
        if tok.type is not TOKEN_TYPE_OPERATOR:
            return expr

        # Initialize elements stack
        stack = [expr]
        while tok.type is TOKEN_TYPE_OPERATOR:
            op = self._get_and_check_operator(tok)
            self._next_token()
            expr = self._read_sub_expression()
            tok = self.token

            # Reduce stack if possible
            while (len(stack) > 1) and op.priority >= stack[-2].priority:
                oexpr = stack.pop()
                oop = stack.pop()
                stack[-1] = self._create_operation_expression(oop, (stack[-1], oexpr))

            stack.append(op)
            stack.append(expr)

        # Build final expression
        expr = stack.pop()
        while stack:
            op = stack.pop()
            expr = self._create_operation_expression(op, (stack.pop(), expr))
        return expr


    def _read_sub_expression(self):
        """ Read a sub-expression

        First expression token is already read.
        Function exits with current token following the last expression token
        Return:
            Expression that has been read
        """

        tok = self.token
        toktyp = tok.type
        ntok = self._next_token()

        # Check int constant
        if toktyp is TOKEN_TYPE_INTEGER:
            return int(tok.value)
        
        # Check float constant
        if toktyp is TOKEN_TYPE_FLOAT:
            return float(tok.value)

        # Check unary operator
        if toktyp is TOKEN_TYPE_OPERATOR:
            # Retrieve operation descriptor
            op = self._get_and_check_operator(tok)
            # Read next expression
            expr = self._read_sub_expression()
            return self._create_operation_expression(op, (expr,))
        
        # Check symbol
        if toktyp is TOKEN_TYPE_SYMBOL:
            tokval = tok.value
            if ntok is TOKEN_PARENT_OPEN:
                self._next_token()
                # Check special function calls
                fun = self.fun_handlers.get(tokval)
                if fun:
                    return fun()

                # General function call, retrieve operation descriptor
                op = _ALL_OPERATIONS.get(tokval)
                if op is None:
                    self._raise_exception("Unknown operation '" + str(tok.value) + "'")
                args = self._read_expression_list_up_to_parent_close()
                return self._create_operation_expression(op, args)

            if ntok is TOKEN_HOOK_OPEN:
                # Read typed array
                self._next_token()
                expr = self._read_expression_list(TOKEN_HOOK_CLOSE)
                # Search type
                typ = _ARRAY_TYPES.get(tok.value)
                if typ is None:
                    self._raise_exception("Unknown array type '" + str(tok.value) + "'")
                # Check empty array
                if len(expr) == 0:
                    return CpoValue((), typ)
                # Compute best type if array not empty
                res = build_cpo_expr(expr)
                if not res.type.is_kind_of(typ):
                    res.type = typ
                return res

            # Check known identifier
            if tokval in _KNOWN_IDENTIFIERS:
                return _KNOWN_IDENTIFIERS[tok.value]

            # Token is an expression id
            return self._get_identifier_value(tok.get_string())
        
        # Check expression in parenthesis
        if tok is TOKEN_PARENT_OPEN:
            expr = self._read_expression_list_up_to_parent_close()
            if len(expr) == 1:
                return expr[0]
            return expr

        # Check array with no type
        if tok is TOKEN_HOOK_OPEN:
            expr = self._read_expression_list(TOKEN_HOOK_CLOSE)
            return expr
            
        # Check reference to a model expression or variable
        if toktyp is TOKEN_TYPE_STRING:
            return self._get_identifier_value(tok.get_string())
                
        # Unknown expression
        self._raise_exception("Invalid start of expression: '" + str(tok) + "'")


    def _read_fun_intVar(self):
        """ Read a function call to 'intVar'
        Returns: New expression
        """
        args = self._read_expression_list_up_to_parent_close()
        return CpoIntVar(tuple(args))


    def _read_fun_intervalVar(self):
        """ Read a function call to 'intervalVar'
        Returns: New expression
        """
        expr = self._read_interval_var()
        self._check_token(self.token, TOKEN_PARENT_CLOSE)
        self._next_token()
        return expr


    def _read_fun_boolVar(self):
        """ Read a function call to 'boolVar'
        Returns: New expression
        """
        args = self._read_expression_list_up_to_parent_close()
        # Should add some test here to verify the domain
        return CpoBoolVar(tuple(args))


    def _read_fun_floatVar(self):
        """ Read a function call to 'floatVar'
        Returns: New expression
        """
        args = self._read_expression_list_up_to_parent_close()
        if len(args) != 2:
            self._raise_exception("'_floatVar' should have 2 arguments")
        return CpoFloatVar(args[0], args[1])


    def _read_fun_sequenceVar(self):
        """ Read a function call to 'sequenceVar'
        Returns: New expression
        """
        args = self._read_expression_list_up_to_parent_close()
        if len(args) == 1:
            lvars = args[0]
            ltypes = None
        else:
            if len(args) != 2:
                self._raise_exception("'sequenceVar' should have 1 or 2 arguments")
            lvars = args[0]
            ltypes = args[1]
        return CpoSequenceVar(lvars, ltypes)


    def _read_fun_stateFunction(self):
        """ Read a function call to 'stateFunction'
        Returns: New expression
        """
        args = self._read_expression_list_up_to_parent_close()
        nbargs = len(args)
        if nbargs == 0:
            trmx = None
        elif nbargs == 1:
            trmx = args[0]
        else:
            self._raise_exception("'stateFunction' should have 0 or 1 argument")
        return CpoStateFunction(trmx)


    def _read_fun_stepFunction(self):
        """ Read a function call to 'stepFunction'
        Returns: New expression
        """
        args = self._read_expression_list_up_to_parent_close()
        return CpoStepFunction(args)


    def _read_fun_segmentedFunction(self):
        """ Read a function call to 'segmentedFunction'
        Returns: New expression
        """
        args = self._read_expression_list_up_to_parent_close()
        return CpoSegmentedFunction(args[0], args[1:])


    def _read_fun_transitionMatrix(self):
        """ Read a function call to 'transitionMatrix'
        Returns: New expression
        """
        # Check arguments list to support presolved version
        if self.token.value == "matrixSize":
            args = self._read_arguments_list_up_to_parent_close()
            for name, mtrx in args:
                if name == 'matrix':
                    break
        else:
            mtrx = self._read_expression_list_up_to_parent_close()
        slen = len(mtrx)
        size = int(math.sqrt(slen))
        if size * size != slen:
            raise CpoParserException("Length of transition matrix values should be a square")
        return CpoTransitionMatrix(values=(mtrx[i * size : (i+1) * size] for i in range(size)))


    def _read_expression_list_in_parenthesis(self):
        """ Read a list of expressions between parenthesis
        Opening parenthesis is read and checked by this method.
        When returning, current token is token after closing parenthesis.
        Args:
           etok: Expression list ending token string (for example ')' or ']')
        Returns:
            Array of expressions
        """
        self._check_token(self._next_token(), TOKEN_PARENT_OPEN)
        self._next_token()
        lxpr = []
        while self.token is not TOKEN_PARENT_CLOSE:
            lxpr.append(self._read_expression())
            if self.token is TOKEN_COMMA:
                self._next_token()
        self._next_token()
        return tuple(lxpr)


    def _read_expression_list(self, etok):
        """ Read a list of expressions
        This method supposes that the current token is just after list starting character.
        When returning, current token is token after end of list

        Args:
           etok: Expression list ending token string (for example ')' or ']')
        Returns:
            Array of expressions
        """
        lxpr = []
        while self.token is not etok:
            lxpr.append(self._read_expression())
            if self.token is TOKEN_COMMA:
                self._next_token()
        self._next_token()
        return tuple(lxpr)


    def _read_expression_list_up_to_parent_close(self):
        """ Read a list of expressions up to a closing parenthesis
        This method supposes that current token is first after opening parenthesis.
        When returning, current token is token after end of list

        Returns:
            Array of expressions
        """
        lxpr = []
        while self.token is not TOKEN_PARENT_CLOSE:
            lxpr.append(self._read_expression())
            if self.token is TOKEN_COMMA:
                self._next_token()
        self._next_token()
        return tuple(lxpr)


    def _read_arguments_list_up_to_parent_close(self):
        """ Read a list of arguments that are possibly named, up to ending closing parenthesis

        This method supposes that the current token is list start (for example '(' or '[').
        When returning, current token is next to list ending token

        Args:
           etok: Expression list ending token (for example ')' or ']')
        Returns:
            Array of couples (name, expression)
        """
        lxpr = []
        while self.token is not TOKEN_PARENT_CLOSE:
            if self.token.type is TOKEN_TYPE_SYMBOL:
                name = self.token
                if self._next_token() is TOKEN_ASSIGN:
                    self._next_token()
                else:
                    self._push_token(name)
                    name = None
            else:
                name = None
            lxpr.append((name, self._read_expression()))
            if self.token is TOKEN_COMMA:
                self._next_token()
        self._next_token()
        return lxpr


    def _read_section(self, name):
        """ Read a section.
        Current token is the opening brace
        Args:
            name:  Section name
        """
        if name == "parameters":
            self._read_section_parameters()
        elif name == "internals":
            self._read_section_internals()
        elif name == "search":
            self._read_section_search()
        elif name == "startingPoint":
            self._read_section_starting_point()
        elif name == "KPIs":
            self._read_section_kpis()
        elif name == "phases":
            self._read_section_phases()
        else:
            self._raise_exception("Unknown section '" + name + "'")


    def _read_section_parameters(self):
        """ Read a parameters section
        """
        params = CpoParameters()
        tok = self._next_token()
        while (tok is not TOKEN_EOF) and (tok is not TOKEN_BRACE_CLOSE):
            vname = self._check_token_string(tok)
            self._check_token(self._next_token(), TOKEN_ASSIGN)
            value = self._next_token()
            self._check_token(self._next_token(), TOKEN_SEMICOLON)
            params.set_attribute(vname, value.get_string())
            tok = self._next_token()
        if params:
            self.model.set_parameters(params)


    def _read_section_internals(self):
        """ Read a internals section
        """
        # Skip all until section end
        tok = self._next_token()
        while (tok is not TOKEN_EOF) and (tok is not TOKEN_BRACE_CLOSE):
            if self.token.value == "version":
                self._check_token(self._next_token(), TOKEN_PARENT_OPEN)
                vtok = self._next_token()
                if vtok.type is TOKEN_TYPE_VERSION:
                    ver = vtok.get_string()
                    self.model.set_format_version(ver)
                    if compare_natural(ver, MIN_CPO_VERSION_NUMBER) < 0:
                        raise CpoUnsupportedFormatVersionException("Can not parse a CPO file with version {}, lower than {}"
                                                                   .format(ver, MIN_CPO_VERSION_NUMBER))
                    if compare_natural(ver, MAX_CPO_VERSION_NUMBER) > 0:
                        raise CpoUnsupportedFormatVersionException("Can not parse a CPO file with version {}, greater than {}"
                                                                   .format(ver, MAX_CPO_VERSION_NUMBER))
                self._check_token(self._next_token(), TOKEN_PARENT_CLOSE)
            tok = self._next_token()


    def _read_section_phases(self):
        """ Read a phase section (old CPO versions, ignored)
        """
        # Skip all until section end
        tok = self._next_token()
        while (tok is not TOKEN_EOF) and (tok is not TOKEN_BRACE_CLOSE):
            tok = self._next_token()


    def _read_section_search(self):
        """ Read a search section
        """
        # Read statements up to end of section
        tok = self._next_token()
        while (tok is not TOKEN_EOF) and (tok is not TOKEN_BRACE_CLOSE):
            self._push_token(tok)
            self._read_statement()
            tok = self._next_token()


    def _read_section_starting_point(self):
        """ Read a starting point section
        """
        sp = CpoModelSolution()
        # Read statements up to end of section
        tok = self._next_token()
        while (tok is not TOKEN_EOF) and (tok is not TOKEN_BRACE_CLOSE):
            # Check directive (#line)
            if tok is TOKEN_HASH:
                self._next_token()
                self._read_directive()
                tok = self._next_token()
                continue

            # Check token is a string
            vname = self._check_token_string(tok)

            # Check experimental section "expressions"
            if vname == "expressions":
                # Read opening brace
                self._check_token(self._next_token(), TOKEN_BRACE_OPEN)
                # Read and ignore all up to next brace close
                tok = self._next_token()
                while (tok is not TOKEN_EOF) and (tok is not TOKEN_BRACE_CLOSE):
                    tok = self._next_token()
                self._next_token()
                continue

            # Get and check the variable that is concerned
            var = self.expr_dict.get(vname)
            if var is None:
                self._raise_exception("There is no variable named '{}' in this model".format(vname))
            self._check_token(self._next_token(), TOKEN_ASSIGN)
            tok = self._next_token()
            vsol = None

            # Process integer variable
            if var.type is Type_IntVar:
                if tok.value == "intVar":
                    tok = self._next_token()
                if tok is TOKEN_PARENT_OPEN:
                    self._next_token()
                # Read domain
                dom = self._read_expression()
                if self.token is TOKEN_PARENT_CLOSE:
                    self._next_token()
                # Add solution to starting point
                vsol = CpoIntVarSolution(var, (dom, ))

            # Process interval variable
            elif var.type is Type_IntervalVar:
                if tok.value == "absent":
                    vsol = CpoIntervalVarSolution(var, presence=False)
                    self._next_token()
                else:
                    if tok.value == "intervalVar":
                        tok = self._next_token()
                    if tok is TOKEN_PARENT_OPEN:
                        self._next_token()
                        ivar = self._read_interval_var()
                        self._check_token(self.token, TOKEN_PARENT_CLOSE)
                        self._next_token()
                    else:
                        ivar = self._read_interval_var()
                    #ivar.set_name(vname)
                    vsol = CpoIntervalVarSolution(var,
                                                  presence=True if ivar.is_present() else False if ivar.is_absent() else None,
                                                  start=ivar.get_start() if ivar.get_start() != DEFAULT_INTERVAL else None,
                                                  end=ivar.get_end() if ivar.get_end() != DEFAULT_INTERVAL else None,
                                                  size=ivar.get_size() if ivar.get_size() != DEFAULT_INTERVAL else None)

            # Process sequence variable (not public)
            elif var.type is Type_SequenceVar:
                # Read array of variables
                expr = self._read_expression()
                if not is_array(expr):
                    self._raise_exception("In section 'startingPoint', the solution of a sequence variable should be a list of interval variables")
                vsol = CpoSequenceVarSolution(var, expr)

            # Process state function
            elif var.type is Type_StateFunction:
                # Read state function value and ignore
                self._read_expression()

            # Process pulse function
            elif var.type is Type_CumulAtom:
                # Read pulse expression and ignore
                self._read_expression()

            else:
                self._raise_exception("The section 'startingPoint' should contain only integer and interval variables.")

            # Add variable solution to starting point
            if vsol is not None:
                sp.add_var_solution(vsol)

            # Read end of variable starting point
            self._check_token(self.token, TOKEN_SEMICOLON)
            tok = self._next_token()

        # Add starting point to the model
        self.model.set_starting_point(sp)


    def _read_section_kpis(self):
        """ Read a KPI section
        """
        # Check that format version is appropriate
        fver = self.model.get_format_version()
        if fver is not None and compare_natural(fver, '12.9') < 0:
            raise CpoUnsupportedFormatVersionException("Section KPIs is not supported in format version {}".format(fver))

        # Read statements up to end of section
        tok = self._next_token()
        while (tok is not TOKEN_EOF) and (tok is not TOKEN_BRACE_CLOSE):
            # Check directive (line)
            if tok is TOKEN_HASH:
                self._next_token()
                self._read_directive()
                tok = self._next_token()
                continue

            # Get KPI name
            kname = self._check_token_string(tok)
            if self._next_token() is TOKEN_SEMICOLON:
                # Verify KPI name exists as an expression
                expr = self.expr_dict.get(kname)
                if expr is None:
                    self._raise_exception("There is no expression named '{}' in this model".format(kname))
            else:
                # Read expression
                self._check_token(self.token, TOKEN_ASSIGN)
                self._next_token()
                expr = self._read_expression()
                self._check_token(self.token, TOKEN_SEMICOLON)
            tok = self._next_token()
            # Add KPI
            self.model.add_kpi(expr, kname)


    def _check_token(self, tok, etok):
        """ Check that a read token is a given one an raise an exception if not
        Args:
            tok: Read token
            etok: Expected token
        """
        if tok is not etok:
            self._raise_unexpected_token(etok, tok)


    def _get_and_check_operator(self, tok):
        """ Get an operator descriptor and raise an exception if not found
        Args:
            tok:  Operator token
        Returns:
            List of Operation descriptor for this keyword
        Raises:
            CpoException if operator does not exists
        """
        op = _ALL_OPERATORS.get(tok.value)
        if op is None:
            self._raise_exception("Unknown operator '" + str(tok.value) + "'")
        return op


    def _check_token_string(self, tok):
        """ Check that a token is a string and raise an exception if not
        Args:
            tok: Token
        Returns:
            String value of the token            
        """
        if tok.type is TOKEN_TYPE_SYMBOL:
            return tok.value
        if tok.type is TOKEN_TYPE_STRING:
            return tok.get_string()
        self._raise_exception("String expected")


    def _check_token_integer(self, tok):
        """ Check that a token is an integer and raise an exception if not
        Args:
            tok: Token
        Returns:
            integer value of the token
        """
        if tok.type is TOKEN_TYPE_INTEGER:
            return int(tok.value)
        if tok.value in _KNOWN_IDENTIFIERS:
            return _KNOWN_IDENTIFIERS[tok.value]
        self._raise_exception("Integer expected instead of '" + tok.value + "'")


    def _get_identifier_value(self, eid):
        """ Get an expression associated to an identifier
        Args:
            eid:  Expression identifier
        Returns:
            Expression corresponding to this identifier
        """
        try:
            return self.expr_dict[eid]
        except:
            self._raise_exception("Unknown identifier '" + str(eid) + "'")


    def _create_operation_expression(self, op, args):
        """ Create a model operation

        Args:
            op:    Operation descriptor
            args:  Operation arguments
        Returns:
            Model expression
        Raises:
            Cpo exception if error
        """
        # Check interval operator
        if op is _OPER_INTERVAL:
            return tuple(args)
        # Check unary operations on constant value
        if (op is Oper_minus) and (len(args) == 1) and is_number(args[0]):
            return -args[0]
        if (op is Oper_plus) and (len(args) == 1) and is_number(args[0]):
            return args[0]

        try:
            return _create_operation(op, args)
        except Exception as e:
            lastex = Exception("No valid operation found for {}: {}".format(op.cpo_name, e))
            self._raise_exception(str(lastex))


    def _raise_unexpected_token(self, expect=None, tok=None):
        """ Raise a "Unexpected token" exception
        Args:
            tok:  Unexpected token
        """
        if tok is None: 
            tok = self.token
        if expect is None:
            self._raise_exception("Unexpected token '" + str(tok) + "'")
        self._raise_exception("Read '" + str(tok) + "' instead of expected '" + str(expect) + "'")


    def _raise_exception(self, msg):
        """ Raise a Parsing exception
        Args:
            msg:  Exception message
        """
        raise CpoParserException(self.tokenizer.build_error_string(msg))


    def _next_token(self):
        """ Read next token
        Returns:
            Next read token, None if end of input
        """
        # Check if a token has been pushed
        if self.pushtoken is not None:
            tok = self.pushtoken
            self.pushtoken = None
        else:
            tok = self.tokenizer.next_token()
        self.token = tok
        #print("Tok='" + str(tok) + "'")
        return tok


    def _push_token(self, tok):
        """ Push current token
        Args:
            tok: New current token 
        """
        self.pushtoken = self.token
        self.token = tok
        
