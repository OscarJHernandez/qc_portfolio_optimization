# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Parser converting a LP file to internal model representation.
"""

from docplex.cp.lp.lp_tokenizer import *
from docplex.cp.expression import *
from docplex.cp.expression import _create_operation
from docplex.cp.catalog import *
from docplex.cp.solution import *
from docplex.cp.model import CpoModel
import docplex.cp.config as config
import docplex.cp.modeler as modeler
from docplex.cp.parameters import CpoParameters

import traceback
import math
import warnings


#==============================================================================
#  Constants
#==============================================================================

# Known identifiers
_KNOWN_IDENTIFIERS = {"infinity": INT_MAX, "inf": INT_MAX, }

# Special operators
Oper_imply       = CpoOperation("imply", "imply", "->", 7, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr, Type_BoolExpr)),) )
Oper_imply_back  = CpoOperation("implyBack", "implyBack", "<-", 7, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr, Type_BoolExpr)),) )
Oper_equiv       = CpoOperation("equiv", "equiv", "<->", 7, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr, Type_BoolExpr)),) )

# Map of operators
_All_OPERATORS = {'-':   Oper_minus,
                  '+':   Oper_plus,
                  '*':   Oper_times,
                  '/':   Oper_float_div,
                  '^':   Oper_power,
                  '<':   Oper_less,
                  '>':   Oper_greater,
                  '<=':  Oper_less_or_equal,
                  '>=':  Oper_greater_or_equal,
                  '=':   Oper_equal,
                  '->':  Oper_imply,
                  '<-':  Oper_imply_back,
                  '<->': Oper_equiv,
                  }

_COMPARE_OPERATORS = {'<', '>', '<=', '>=', '='}


#==============================================================================
#  Public classes
#==============================================================================

class LpParserException(CpoException):
    """ The base class for exceptions raised by the CPO parser
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(LpParserException, self).__init__(msg)


class LpParser(object):
    """ Reader of CPO file format """
    __slots__ = ('model',        # Read model
                 'expr_dict',    # Dictionary of expressions. Key is name, value is CPO expression
                 'source_file',  # Source file
                 'tokenizer',    # Reading tokenizer
                 'token',        # Last read token
                 'pushtoken',    # Pushed token
                 'allvars',      # Set of all variable names
                 'intvars',      # Set of variable names that are declared as integer
                 )
    
    def __init__(self, mdl=None):
        """ Create a new LP format parser

        Args:
            mdl:  Model to fill, None (default) to create a new one.
        """
        super(LpParser, self).__init__()
        self.model = mdl if mdl is not None else CpoModel()
        self.expr_dict = {}
        self.source_file = None
        self.tokenizer = None
        self.token = None
        self.pushtoken = None
        self.allvars = set()
        self.intvars = set()

        # Do not store location information (would store parser instead of real lines)
        self.model.source_loc = False

        # TODO: parse and include source information ?


    def get_model(self):
        """ Get the model that have been parsed

        Return:
            Model result of the parsing, object of class :class:`~docplex.cp.model.CpoModel`
        """
        return self.model


    def parse(self, file, encoding='utf-8-sig'):
        """ Parse a LP file.

        Args:
            file:     LP file to read
            encoding: (Optional) File encoding
        Return:
            Model result of the parsing, object of class :class:`~docplex.cp.model.CpoModel`
        Raises:
            LpParserException: Parsing exception
        """
        # Store file name in model if first file
        self.source_file = file
        if self.model.source_file is None:
            self.model.source_file = file

        # Read file content
        self.tokenizer = LpTokenizer(file=file, encoding=encoding)
        self._read_section_list()
        self.tokenizer = None

        # Check that all variables are declared as integer variables
        floatvars = self.allvars.difference(self.intvars)
        if floatvars:
            warnings.warn("The LP file '{}' contains {} variable(s) considered as integer but which has not been declared as such.".format(file, len(floatvars)))

        # Return
        return self.model


    def parse_string(self, str):
        """ Parse a string.

        Result of the parsing is added to the current result model.

        Args:
            str: String to parse
        Return:
            Model result of the parsing, object of class :class:`~docplex.cp.model.CpoModel`
        """
        # Read file content
        self.tokenizer = LpTokenizer(input=str)
        self._read_section_list()
        self.tokenizer = None

        # Check that all variables are declared as integer variables
        floatvars = self.allvars.difference(self.intvars)
        if floatvars:
            warnings.warn("The LP string contains {} variable(s) considered as integer but which has not been declared as such.".format(len(floatvars)))

        return self.model


    def _read_section_list(self):
        """ Read a list of sections

        This functions reads all LP document sections up to end of input.
        """
        try:
            self._next_token()
            while self.token != TOKEN_EOF:
                tok = self.token
                self._next_token()
                if tok in (TOKEN_KEYWORD_MINIMIZE, TOKEN_KEYWORD_MAXIMIZE, ):
                    self._read_section_objective(tok)
                elif tok in (TOKEN_KEYWORD_MINIMIZE_MULTI, TOKEN_KEYWORD_MAXIMIZE_MULTI, ):
                    self._read_section_objective_multi(tok)
                elif tok is TOKEN_KEYWORD_SUBJECT_TO:
                    self._read_section_constraints()
                elif tok is TOKEN_KEYWORD_BOUNDS:
                    self._read_section_bounds()
                elif tok is TOKEN_KEYWORD_BINARY:
                    self._read_section_binary()
                elif tok is TOKEN_KEYWORD_GENERAL:
                    self._read_section_general()
                elif tok is TOKEN_KEYWORD_NAMELIST:
                    self._read_section_namelist()
                elif tok is TOKEN_KEYWORD_SEMI_CONTINUOUS:
                    self._read_section_semi_continuous()
                elif tok is TOKEN_KEYWORD_SOS:
                    self._read_section_sos()
                elif tok is TOKEN_KEYWORD_PWL:
                    self._read_section_pwl()
                elif tok is TOKEN_KEYWORD_END:
                    pass
                else:
                    self._raise_exception("Unknown section '{}'".format(tok))

        except Exception as e:
            if config.context.log_exceptions:
                traceback.print_exc()
            self._raise_exception(str(e))
        return True


    def _read_section_objective(self, stok):
        """ Read the objective section

        Args:
            stok:  Section name token
        """
        # Read expression
        expr = self._read_expression()

        # Add expression to the model
        if expr is not None:
           self.model.add(modeler.minimize(expr) if stok is TOKEN_KEYWORD_MINIMIZE else modeler.maximize(expr))


    def _read_section_objective_multi(self, stok):
        """ Read the multi-objective section

        Args:
            stok:  Section name token
        """
        # Initialize list of objectives, tuples (name, dict_of_properties, expr)
        objectives = []

        # Read all objectives
        while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
            exprname = self._read_expression_name()
            props = {}
            expr = None
            while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
                key = self.token.value
                if key in ('Priority', 'Weight', 'AbsTol', 'RelTol'):
                    key = self.token.value
                    self._check_token(self._next_token(), TOKEN_ASSIGN)
                    self._next_token()
                    value = self._check_number_expr(self._read_expression_item())
                    props[key] = value
                else:
                    expr = self._read_expression()
                    break
            objectives.append((exprname, props, expr,))

        # Sort all objectives by priority order
        objectives.sort(key=lambda x: x[1].get('Priority', 1), reverse=True)

        # Build list of objective expressions
        lobjexpr = []
        ox = 0
        while ox < len(objectives):
            # Get all objectives with same priority
            lobjs = [obj for obj in objectives if obj[1].get('Priority') == objectives[ox][1].get('Priority')]
            ox += len(lobjs)
            # Build sum expression
            objexpr = None
            for name, props, expr in lobjs:
                w = props.get('Weight')
                if w is not None and w != 1:
                    expr = modeler.times(expr, w)
                if name is not None:
                    expr.set_name(name)
                objexpr = expr if objexpr is None else modeler.plus(objexpr, expr)
            # Add to list of objectives
            lobjexpr.append(objexpr)

        # Add expression to the model
        if lobjexpr:
           self.model.add(modeler.minimize_static_lex(lobjexpr) if stok is TOKEN_KEYWORD_MINIMIZE_MULTI else modeler.maximize_static_lex(lobjexpr))

        # Determine first tolerance parameters
        abstol = reltol = None
        for obj in objectives:
            props = obj[1]
            at = props.get('AbsTol')
            rt = props.get('RelTol')
            if at is not None or rt is not None:
                if abstol is None and reltol is None:
                    abstol = at
                    reltol = rt
                elif abstol != at or reltol != rt:
                    warnings.warn("Different tolerances found in different objectives. Kept tolerance is the one from the most prioritar objective.")
                    break

        # Set tolerance if any
        if abstol is not None:
            self.model.add_parameters(OptimalityTolerance=abstol)
        if reltol is not None:
            self.model.add_parameters(RelativeOptimalityTolerance=reltol)


    def _read_section_constraints(self):
        """ Read the constraints section
        """
        while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
            expr = self._read_expression()
            self.model.add(expr)


    def _read_section_bounds(self):
        """ Read the bounds section
        """
        while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
            # Check first expression
            var = None
            b1 = self._read_expression_item()
            if isinstance(b1, CpoExpr):
                var = b1
                b1 = None
            elif not is_number(b1):
                self._raise_exception("Invalid bound definition {}".format(b1))

            # Check next
            if self.token.type is TOKEN_TYPE_SYMBOL and self.token.value.lower() == "free":
                # Free variable
                if var is None:
                    self._raise_exception("No variable specified to be free")
                self._set_bounds(var, INT_MIN, INT_MAX)
                self._next_token()
                continue

            # Check assignment
            optok = self.token
            self._next_token()
            if optok is TOKEN_ASSIGN:
                # Read variable value
                val = self._read_expression_item()
                if not is_number(val):
                    self._raise_exception("Variable value should be a number")
                self._set_bounds(var, val, val)
                continue
            elif optok not in (TOKEN_GREATER_EQUAL, TOKEN_LOWER_EQUAL, ):
                self._raise_exception("Operator '{}' not allowed in a bound definition".format(optok))

            # Get next element
            x = self._read_expression_item()
            if var is not None:
                if not is_number(x):
                    self._raise_exception("Bound for variable '{}' should be a number".format(var.get_name()))
                if optok is TOKEN_LOWER_EQUAL:
                    self._set_bounds(var, None, x)
                else:
                    self._set_bounds(var, x, None)
                continue

            if not isinstance(x, CpoIntVar):
                self._raise_exception("Variable expected in bound definition, not {}".format(x))
            var = x

            if self.token is not optok:
                if optok is TOKEN_LOWER_EQUAL:
                    self._set_bounds(var, b1, None)
                else:
                    self._set_bounds(var, None, b1)
                continue

            # Get second bound
            self._next_token()
            b2 = self._read_expression_item()
            if not is_number(b2):
                self._raise_exception("Variable upper bound should be a number")
            if optok is TOKEN_LOWER_EQUAL:
                self._set_bounds(var, b1, b2)
            else:
                self._set_bounds(var, b2, b1)


    def _read_section_general(self):
        """ Read the general section
        """
        while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
            if self.token.type is not TOKEN_TYPE_SYMBOL:
                self._raise_exception("General section should contain only variable names")
            var = self._get_identifier_value(self.token.value)
            self._next_token()
            self.intvars.add(var.get_name())


    def _read_section_namelist(self):
        """ Read the namelist section (not documented, ignored)
        """
        while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
            if self.token.type is not TOKEN_TYPE_SYMBOL:
                self._raise_exception("General section should contain only variable names")
            var = self._get_identifier_value(self.token.value)
            self._next_token()


    def _read_section_binary(self):
        """ Read the binary section
        """
        while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
            if self.token.type is not TOKEN_TYPE_SYMBOL:
                self._raise_exception("Binary section should contain only variable names")
            var = self._get_identifier_value(self.token.value)
            var.set_domain((0, 1,))
            self._next_token()
            self.intvars.add(var.get_name())


    def _read_section_semi_continuous(self):
        """ Read the semi-continuous section
        """
        while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
            if self.token.type is not TOKEN_TYPE_SYMBOL:
                self._raise_exception("Semi-continuous section should contain only variable names")
            var = self._get_identifier_value(self.token.value)
            var.set_domain((0, (var.get_domain_min(), var.get_domain_max(),),))
            self._next_token()
            self.intvars.add(var.get_name())


    def _read_section_sos(self):
        """ Read the SOS section
        """
        while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
            stype = self._check_token_string(self.token)
            self._next_token()
            name = None
            if self.token is TOKEN_COLON:
                name = stype
                stype = self._check_token_string(self._next_token())
                self._next_token()
            self._check_token(self.token, TOKEN_DOUBLECOLON)
            self._next_token()

            # Check set type
            stype = stype.lower()
            if stype not in {'s1', 's2'}:
                self._raise_exception("Set type should be S1 or S2")

            # Read couples variable/weight
            lwvars = []
            while self.token.type is TOKEN_TYPE_SYMBOL:
                var = self.expr_dict.get(self.token.value)
                if not isinstance(var, CpoIntVar):
                    break
                self._check_token(self._next_token(), TOKEN_COLON)
                whg = self._check_token_integer(self._next_token())
                self._next_token()
                lwvars.append((var, whg))

            # Build constraints for this set
            self._add_special_ordered_set(name, stype, lwvars)


    def _read_section_pwl(self):
        """ Read the PWL section
        """
        # Read PWLs
        while self.token is not TOKEN_EOF and self.token.type is not TOKEN_TYPE_KEYWORD:
            exprname = self._read_expression_name()

            # Get variable name
            yvar = self._read_expression_item()
            if not isinstance(yvar, CpoVariable):
                self._raise_exception("Variable expected")

            # Read equal
            self._check_token(self.token, TOKEN_ASSIGN)
            self._next_token()

            # Read x variable
            xvar = self._read_expression_item()
            if not isinstance(xvar, CpoVariable):
                self._raise_exception("Variable expected after '='")

            # Move tokenizer in PWL mode
            self.tokenizer.set_pwl_mode(True)

            # Read initial slope
            preslope = self._read_expression_item()
            if not is_number(preslope):
                self._raise_exception("Preslope should be a number")

            # Read break points
            lpoints = []
            lvalues = []
            while self.token is TOKEN_PARENT_OPEN:
                self._next_token()
                px = self._read_expression_item()
                self._check_number_expr(px)
                self._check_token(self.token, TOKEN_COMMA)
                self._next_token()
                py = self._read_expression_item()
                self._check_number_expr(py)
                self._check_token(self.token, TOKEN_PARENT_CLOSE)
                self._next_token()
                lpoints.append(px)
                lvalues.append(py)

            # Read postslope
            postslope = self._read_expression_item()
            if not is_number(postslope):
                self._raise_exception("Postslope should be a number")

            # Restore tokenizer in normal
            self.tokenizer.set_pwl_mode(False)

            # Build constraints for this piecewise linear function
            pwl = modeler.coordinate_piecewise_linear(xvar, preslope, lpoints, lvalues, postslope)
            expr = modeler.equal(yvar, pwl)
            if exprname is not None:
                expr.set_name(exprname)
            self.model.add(expr)


    def _add_special_ordered_set(self, name, stype, lwvars):
        """ Add constraints related to special ordered set

        Args:
            name:   Set name
            stype:  Type of set, in s1, s2
            lwvars: List of weighted vars (tuples (var, weight))
        Return:
            Expression that has been read
        """
        # Sort variables in order of weights
        lwvars.sort(key=lambda x: x[1])
        lwvars = tuple(x[0] for x in lwvars)

        # Build appropriate constraints
        if stype == "s1":
            lexpr = [x > 0 for x in lwvars]
            ctr = modeler.sum(lexpr) <= 1
        else:
            lexpr = [x > 0 for x in lwvars]
            # Add adjacence constraint
            ladj = []
            for i in range(len(lwvars) - 1):
                ladj.append(modeler.logical_and(lwvars[i] > 0, lwvars[i+1] > 0))
            v = integer_var(min=0, max=len(lwvars), name='NBP')
            self.model.add(v == modeler.sum(lexpr))
            ctr = modeler.logical_and(v <= 2, modeler.if_then(v == 2, modeler.sum(ladj) <= 1))

        # Add constraint
        if name:
            ctr.set_name(name)
        self.model.add(ctr)


    def _read_expression(self):
        """ Read an expression

        Returns:
            Expression that has been read
        """
        # Check named expression
        exprname = self._read_expression_name()

        # Particular case, check no expression (for objective null)
        if self.token.type is TOKEN_TYPE_KEYWORD:
            return None

        # Read first sub-expression
        expr = self._read_expression_item(True)

        # Check special case of products without sign
        enable_blank_product = True
        if self.token.type is TOKEN_TYPE_OPERATOR:
            # Initialize elements stack
            stack = [expr]
            while self.token.type is TOKEN_TYPE_OPERATOR:
                op = self._get_and_check_operator(self.token)
                opprio = op.priority
                if opprio in (5, 6,):
                    enable_blank_product = False
                elif opprio > 6:
                    enable_blank_product = True
                elif opprio < 5 and not enable_blank_product:
                    break

                # Read next expression item
                self._next_token()
                expr = self._read_expression_item(enable_blank_product)

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
                sexpr = stack.pop()
                expr = self._create_operation_expression(op, (sexpr, expr))

        # Set expression name
        expr = build_cpo_expr(expr)
        if exprname is not None:
            expr.set_name(exprname)
            if exprname not in self.expr_dict:
                self.expr_dict[exprname] = expr

        return expr


    def _read_expression_name(self):
        """ Read the optional name of an expression

        Returns:
            Expression name, None if none
        """
        # Check named expression
        exprname = None
        tok = self.token
        if tok.type is TOKEN_TYPE_SYMBOL:
            if self._next_token() is TOKEN_COLON:
                exprname = tok.value
                self._next_token()
            else:
                self._push_token(tok)
        return exprname


    def _read_expression_item(self, isbp=False):
        """ Read an expression item
        Args:
           isbp:  Enable reading of product using a blank
        Return:
            Expression that has been read
        """

        tok = self.token

        # Check int constant
        if tok.type is TOKEN_TYPE_INTEGER:
            ival = int(tok.value)
            tok = self._next_token()
            if isbp and tok.type is TOKEN_TYPE_SYMBOL:
                self._next_token()
                return modeler.times(ival, self._get_identifier_value(tok.value))
            return ival

        # Check float constant
        if tok.type is TOKEN_TYPE_FLOAT:
            try:
                fval = float(tok.value)
            except:
                if tok.value.lower().endswith('e'):
                    fval = float(tok.value[:-1])
            tok = self._next_token()
            if isbp and tok.type is TOKEN_TYPE_SYMBOL:
                self._next_token()
                return modeler.times(fval, self._get_identifier_value(tok.value))
            return fval

        # Check symbol
        if tok.type is TOKEN_TYPE_SYMBOL:
            self._next_token()
            # Check known identifier
            id = tok.value.lower()
            if id in _KNOWN_IDENTIFIERS:
               return _KNOWN_IDENTIFIERS[id]
            # Token is an expression id
            return self._get_identifier_value(tok.value)

        # Check unary operator
        if tok.type is TOKEN_TYPE_OPERATOR:
            # Special case for <=, =, >= without left side
            if tok in (TOKEN_LOWER_EQUAL, TOKEN_ASSIGN, TOKEN_GREATER_EQUAL):
                return 0
            # Read next expression
            self._next_token()
            expr = self._read_expression_item(isbp)
            # Process operator
            if tok is TOKEN_MINUS:
                return -expr if is_number(expr) else modeler.minus(expr)
            if tok is TOKEN_PLUS:
                return expr
            self._raise_exception("Unknown unary operator {}".format(tok))

        # Check expression in hooks (parenthesis)
        if tok is TOKEN_HOOK_OPEN:
            self._next_token()
            expr = self._read_expression()
            self._check_token(self.token, TOKEN_HOOK_CLOSE)
            self._next_token()
            return expr

        # Unknown expression
        self._raise_exception("Invalid start of expression: '" + str(tok) + "'")


    def _set_bounds(self, expr, lb, ub):
        """ Set bounds on a CPO expression
        Args:
            expr:  CPO expression (usually CpoIntVar, but may be objective function
            lb:    Lower bound, None if not explicitly defined
            ub:    Upper bound, None if not explicitly defined
        """
        if isinstance(expr, CpoIntVar):
            # Check type of bounds
            if not(((lb is None) or is_int(lb)) and ((ub is None) or is_int(ub))):
                self._raise_exception("Integer variable {} can have only integer bounds".format(expr.get_name()))

            # Check single value
            if lb == ub:
                #expr.set_domain((int(lb),))
                expr.set_domain((lb,))
            else:
                # Compute appropriate bounds
                if lb is None:
                    lb = expr.get_domain_min()
                # elif not isinstance(lb, int):
                #     lb = int(math.ceil(lb))

                if ub is None:
                    ub = expr.get_domain_max()
                # elif not isinstance(ub, int):
                #     ub = int(math.floor(ub))

                # Set variable domain
                if lb == ub:
                    expr.set_domain((lb,))
                elif lb < ub:
                    expr.set_domain(((lb, ub,),))
                else:
                    # TODO: Should not occur with float variables, but put in evidence by false expressions
                    self.model.add(modeler.greater_or_equal(expr, lb))
                    self.model.add(modeler.less_or_equal(expr, ub))
        else:
            if lb == ub:
                self.model.add(modeler.equal(expr, lb))
            else:
                if lb is not None:
                    self.model.add(modeler.greater_or_equal(expr, lb))
                if ub is not None:
                    self.model.add(modeler.less_or_equal(expr, ub))

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
        op = _All_OPERATORS.get(tok.value)
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
            Integer value of the token
        """
        if tok.type is TOKEN_TYPE_INTEGER:
            return int(tok.value)
        if tok.value in _KNOWN_IDENTIFIERS:
            return _KNOWN_IDENTIFIERS[tok.value]
        self._raise_exception("Integer expected instead of '" + tok.value + "'")


    def _check_number_expr(self, expr):
        """ Check that an expression is a number
        Args:
            expr: Expression to check
        Returns:
            Number value
        """
        if not is_number(expr):
            self._raise_exception("Number expected")
        return expr


    def _get_identifier_value(self, eid):
        """ Get an expression associated to an identifier
        Args:
            eid:  Expression identifier
        Returns:
            Expression corresponding to this identifier
        """
        expr = self.expr_dict.get(eid)
        if expr is None:
            expr = integer_var(name=eid, min=0)
            self.expr_dict[eid] = expr
            self.allvars.add(eid)
        return expr


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
        # # Check unary minus on constant value
        # if (op is Oper_minus) and (len(args) == 1) and is_number(args[0]):
        #     return -args[0]
        # if (op is Oper_plus) and (len(args) == 1) and is_number(args[0]):
        #     return args[0]

        # Check special operators
        if op is Oper_imply:
            return modeler.if_then(args[0], args[1])
        if op is Oper_imply_back:
            return modeler.if_then(args[1], args[0])
        if op is Oper_equiv:
            return modeler.equal(args[0], args[1])

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
        raise LpParserException(self.tokenizer.build_error_string(msg))


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
        return tok


    def _push_token(self, tok):
        """ Push current token
        Args:
            tok: New current token 
        """
        self.pushtoken = self.token
        self.token = tok


