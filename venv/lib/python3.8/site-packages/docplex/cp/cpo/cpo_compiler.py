# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016, 2017, 2018
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
Compiler converting internal model representation to CPO file format.
"""

from docplex.cp.expression import *
from docplex.cp.expression import _domain_min, _domain_max
from docplex.cp.solution import *
from docplex.cp.utils import *
import docplex.cp.config as config
import datetime
import itertools

import sys
import functools


###############################################################################
## Utilities
###############################################################################

# Map of CPO names for each array type
_ARRAY_TYPES = {Type_IntArray: 'intArray', Type_FloatArray: 'floatArray',
                Type_IntExprArray: 'intExprArray', Type_FloatExprArray: 'floatExprArray',
                Type_CumulExprArray: 'cumulExprArray',
                Type_IntervalVarArray: 'intervalVarArray', Type_SequenceVarArray: 'sequenceVarArray',
                Type_IntValueSelectorArray: 'intValueSelectorArray' , Type_IntVarSelectorArray: 'intVarSelectorArray',
                Type_CumulAtomArray: '_cumulAtomArray'}


# Set of CPO types representing an integer
_INTEGER_TYPES = frozenset((Type_Int, Type_PositiveInt, Type_TimeInt))

# Symbol used to reference anonymous variables
_ANONYMOUS = "Anonymous"


###############################################################################
## Public classes
###############################################################################

class CpoCompiler(object):
    """ Compiler to CPO file format """
    __slots__ = ('model',                    # Source model
                 'parameters',               # Solving parameters
                 'add_source_location',      # Indicator to add location traces in generated output
                 'format_version',           # Output format version
                 'is_format_less_than_12_8', # Indicates that format version is less than 12.8
                 'is_format_at_least_12_8',  # Indicates that format version is at least 12.8
                 'is_format_at_least_12_9',  # Indicates that format version is at least 12.9
                 'name_all_constraints',     # Indicator to fore a name on each constraint (for conflict refiner)
                 'min_length_for_alias',     # Minimum variable name length to replace it by an alias
                 'verbose_output',           # Verbose output (not short_output)
                 'id_printable_strings',     # Dictionary of printable string for each identifier
                 'last_location',            # Last source location (file, line)
                 'list_consts',              # List of constants
                 'list_vars',                # List of variables
                 'list_exprs',               # List of model expressions
                 'list_phases',              # List of search phases
                 'expr_infos',               # Map of expression infos.
                                             # Key is expression id, value is list [expr, location, ref_count, cpo_name, is_root, is_compiled]
                 'expr_by_names',            # Map of expressions by CPO name. Used to retrieve expressions from solutions.
                 'alias_name_map',           # Map of variable renamed with a shorter name. key = old name, value = new name
                 'factorize',                # Indicate to factorize expressions used more than ones
                 'sort_names',               # Type of expressions (variables) sorting: O:None, 1:basic, 2:natural
                 )

    def __init__(self, model, **kwargs):
        """ Create a new compiler

        Args:
            model:  Source model
        Optional args:
            context:             Global solving context. If not given, context is the default context that is set in config.py.
            params:              Solving parameters (CpoParameters) that overwrites those in solving context
            add_source_location: Add source location into generated text
            length_for_alias:    Minimum name length to use shorter alias instead
            (others):            All other context parameters that can be changed
        """
        super(CpoCompiler, self).__init__()

        # Build effective context
        if model:
            mparams = model.get_parameters()
            if mparams:
                pparams = kwargs.get('params')
                if pparams:
                    mparams = mparams.clone()
                    mparams.add(pparams)
                kwargs['params'] = mparams
        context = config._get_effective_context(**kwargs)

        # Initialize processing
        self.model = model
        self.parameters = context.params

        self.min_length_for_alias = None
        self.id_printable_strings = {}
        self.name_all_constraints = False
        self.verbose_output = False
        self.factorize = True
        self.sort_names = 'none'

        # Set model parameters
        mctx = context.model
        if mctx is not None:
            self.min_length_for_alias = mctx.length_for_alias
            self.name_all_constraints = mctx.name_all_constraints
            self.verbose_output = not mctx.short_output
            self.factorize = mctx.factorize_expressions
            self.sort_names = str(mctx.sort_names).lower()

        # Determine CPO format version
        self.format_version = None if model is None else model.get_format_version()
        if self.format_version is None:
            self.format_version = mctx.version

        # Determine output variant triggers
        if self.format_version is None:
            # Assume most recent solver
            self.is_format_less_than_12_8 = False
            self.is_format_at_least_12_8 = True
            self.is_format_at_least_12_9 = True
        else:
            self.is_format_less_than_12_8 = self.format_version and compare_natural(self.format_version, '12.8') < 0
            self.is_format_at_least_12_8 = self.format_version and compare_natural(self.format_version, '12.8') >= 0
            self.is_format_at_least_12_9 = self.format_version and compare_natural(self.format_version, '12.9') >= 0

        # Initialize source location
        if self.verbose_output:
            if (self.parameters is not None) and (self.parameters.UseFileLocations is not None):
                self.add_source_location = (self.parameters.UseFileLocations in ('On', True))
            elif (mctx is not None) and (mctx.add_source_location is not None):
                self.add_source_location = mctx.add_source_location
            else:
                self.add_source_location = True
        else:
            self.add_source_location = False
        self.last_location = None

        # Initialize expression dictionaries
        self.expr_by_names = {}
        self.expr_infos = {}
        self.list_consts = []
        self.list_vars = []
        self.list_exprs = []
        self.list_phases = []
        self.alias_name_map = {}

        # Precompile model
        self._pre_compile_model()


    def get_expr_map(self):
        """ Get the map of model expressions.

        Returns:
            Map of model expressions.
            Key is the name that has been used to represent the expression in the CPO file format,
            Value is the corresponding model expression.
        """
        return self.expr_by_names


    def print_model(self, out=None):
        """ Compile the model and print the CPO file format in a given output.

        If the given output is a string, it is considered as a file name that is opened by this method
        using 'utf-8' encoding.

        DEPRECATED. Use :meth:`write` instead.

        Args:
            out: Target output, stream or file name. Default is sys.stdout.
        """
        self.write(out)


    def get_as_string(self):
        """ Compile the model in CPO file format into a string

        Returns:
            String containing the model
        """
        # Print the model into a string
        out = StringIO()
        self.write(out)
        res = out.getvalue()
        out.close()

        # Convert in unicode if required
        if IS_PYTHON_2 and (type(res) is str):
            res = unicode(res)
        return res


    def write(self, out=None):
        """ Write the model in CPO format.

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

        # Initialize processing
        model = self.model
        self.last_location = None

        # Write header
        if self.verbose_output:
            banner = u"/" * 79 + "\n"
            mname = model.get_name()
            sfile = model.get_source_file()
            out.write(banner)
            datestr = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
            if mname:
                out.write(u"// CPO file generated at {} for model: {}\n".format(datestr, mname))
            else:
                out.write(u"// CPO file generated at {} for anonymous model\n".format(datestr))
            if sfile:
                out.write(u"// Source file: {}\n".format(sfile))
            out.write(banner)

            # Write version if any
            if self.format_version is not None:
                out.write(u"\n//--- Internals ---\n")
                out.write(u"internals {\n")
                out.write(u"   version({});\n".format(self.format_version))
                out.write(u"}\n")

        # Print renamed variables as comment
        snm = self.alias_name_map
        if snm and self.verbose_output:
            out.write(u"\n//--- Aliases ---\n")
            out.write(u"// To reduce CPO file size, the following aliases have been used to replace names longer than {}\n".format(self.min_length_for_alias))
            lvars = sorted(snm.keys(), key=functools.cmp_to_key(lambda v1, v2: compare_natural(v1, v2)))
            for v in lvars:
                out.write(u"// {} = {}\n".format(v, snm[v]))

        # Write constants
        if self.verbose_output:
            out.write(u"\n//--- Constants ---\n")
        for lx in self.list_consts:
            self._write_expression(out, lx)

        # Write variables
        if self.verbose_output:
            out.write(u"\n//--- Variables ---\n")
        for lx in self.list_vars:
            self._write_expression(out, lx)

        # Write expressions
        if self.verbose_output:
            out.write(u"\n//--- Expressions ---\n")
        self.last_location = None
        for lx in self.list_exprs:
            self._write_expression(out, lx)

        # Write KPIs if any
        kpis = model.get_kpis()
        if kpis:
            if self.is_format_at_least_12_9:
                if self.verbose_output:
                    out.write(u"\n//--- KPIs ---\n")
                out.write(u"KPIs {\n")
                for k, (x, l) in kpis.items():
                    # Skip lambda expressions
                    if not isinstance(x, CpoExpr):
                        continue
                    # Write KPI name
                    out.write(self._get_id_string(k));
                    # Retrieve expression infos
                    # if x.name != k:
                    #     out.write(u" = " + self._compile_expression(x, False))
                    # xinfo = self.expr_infos.get(id(x))
                    # if xinfo:
                    #     xn = xinfo[0].name
                    #     if ()
                    #     # Check if expression is already compiled
                    #     if xinfo[5] and xn != k:
                    #         self._write_expression(out, [CpoAlias(x, k), l, 1, self._get_id_string(k), False, False])
                    #     else:
                    #         if xn is not None:
                    #             if xinfo[3] is None:
                    #                xinfo[3] = self._get_id_string(k)
                    #             self._write_expression(out, xinfo)
                    #         else:
                    #             self._write_expression(out, [x, l, 1, self._get_id_string(k), True, False])
                    # else:
                    #     self._write_expression(out, [x, l, 1, self._get_id_string(k), True, False])
                    out.write(u";\n")
                out.write(u"}\n")
            else:
                # Check that all KPI model expressions are integer variables
                for (x, l) in kpis.values():
                    if isinstance(x, CpoExpr) and not isinstance(x, CpoIntVar):
                        raise CpoException("With CPO format version {}, KPI expressions must all be integer variables"
                                           .format(self.format_version))

        # Write search phases
        if self.list_phases:
            if self.verbose_output:
                out.write(u"\n//--- Search phases ---\n")
            out.write(u"search {\n")
            for lx in self.list_phases:
                self._write_expression(out, lx)
            out.write(u"}\n")

        # Write starting point
        spoint = model.get_starting_point()
        if spoint is not None:
            if self.verbose_output:
                out.write(u"\n//--- Starting point ---\n")
            if self.last_location is not None:
                out.write(u"#line off\n")
            out.write(u"startingPoint {\n")
            for var in spoint.get_all_var_solutions():
                self._write_starting_point(out, var)
            out.write(u"}\n")

        # Write parameters
        if self.parameters:
            # Build list of valid parameters
            params = []
            for k in sorted(self.parameters.keys()):
                v = self.parameters[k]
                if v is not None:
                    params.append((k, v))
            if len(params) > 0:
                if self.verbose_output:
                    out.write(u"\n//--- Parameters ---\n")
                if self.last_location is not None:
                    out.write(u"#line off\n")
                out.write(u"parameters {\n")
                for k, v in params:
                    out.write(u"   {} = {};\n".format(k, v))
                out.write(u"}\n")

        # Flush stream (required on Linux rhel6.7)
        out.flush()


    def _write_expression(self, out, xinfo):
        """ Write model expression

        Args:
            out:   Target output
            xinfo: Expression info, list (expr, location, ref_count, cpo_name, is_root, is_compiled)
        """
        # Retrieve expression elements
        expr, loc, rcnt, name, isroot, iscpld = xinfo
        #print("Write expr: {}, refcnt: {}, name: {}, root: {}, iscpld: {}".format(expr, rcnt, name, isroot, iscpld))

        # Trace location if required
        self._write_source_location(out, loc)

        # Write expression
        if iscpld and name:
            out.write(name + u";\n")
        else:
            if name:
                if isinstance(expr, CpoAlias):
                    # Simple alias
                    out.write(name + u" = " + self._compile_expression(expr.expr, False) + u";\n")
                elif isroot and expr.type in (Type_Constraint, Type_SearchPhase, Type_BoolExpr):
                    # Named constraint
                    if self.is_format_at_least_12_8:
                        out.write(name + u": " + self._compile_expression(expr, True) + u";\n")
                    else:
                        out.write(name + u" = " + self._compile_expression(expr, True)+ u";\n" + name + u";\n")
                else:
                    out.write(name + u" = " + self._compile_expression(expr, True)+ u";\n")
            else:
                out.write(self._compile_expression(expr, True) + u";\n")
            # Mark as compiled
            xinfo[5] = True


    def _write_starting_point(self, out, var):
        """ Write a starting point variable

        Args:
            out:  Target output
            var:  Variable solution
        """
        # Build starting point declaration
        cout = []
        if isinstance(var, CpoIntVarSolution):
            self._compile_integer_var_starting_point(var, cout)
        elif isinstance(var, CpoIntervalVarSolution):
            self._compile_interval_var_starting_point(var, cout)
        else:
            #raise CpoException("Internal error: unsupported starting point variable: " + str(var))
            pass

        # Write variable starting point
        if cout:
            out.write(self._get_expr_id(var.expr) + u" = " + u''.join(cout) + u";\n")


    def _write_source_location(self, out, loc):
        """ Write a source location

        Args:
            out:  Target output
            loc:  Source location
        """
        if self.add_source_location:
            lloc = self.last_location
            if loc != lloc:
                if loc is None:
                    out.write( u"#line off\n")
                else:
                    (file, line) = loc
                    lline = u"#line " + str(line)
                    if (lloc is None) or (file != lloc[0]):
                        lline += u' "' + file.replace('\\', '/') + '"'
                    out.write(lline + u"\n")
                self.last_location = loc


    def _get_id_string(self, id):
        """ Get the string representing an identifier

        Args:
            id: Identifier name
        Returns:
            Printable identifier string (including double quotes and escape sequences if needed)
        """
        # Check if already converted
        res = self.id_printable_strings.get(id)
        if res is None:
            # Convert id into string and store result for next call
            res = to_printable_id(id)
            self.id_printable_strings[id] = res
        return res


    def _get_expr_id(self, expr):
        """ Get the printable id of an expression

        Args:
            expr: Expression
        Returns:
            Printable identifier string (including double quotes and escape sequences if needed)
        """
        # Retrieve expression info
        xinfo = self.expr_infos.get(id(expr))
        if xinfo:
            # Expression in a compiled model
            return xinfo[3]
        # Expression out of a model
        xname = expr.get_name()
        if xname:
            return self._get_id_string(xname)
        return _ANONYMOUS


    def _compile_expression(self, expr, root=True):
        """ Compile an expression in a string in CPO format

        Args:
            expr: Expression to compile
            root: Root expression indicator
        Returns:
            String representing this expression in CPO format
        """
        # Initialize working variables
        cout = []  # Result list of strings
        estack = [[expr, 0, False]]  # Expression stack [Expression, child index, parenthesis]

        # Loop while expression stack is not empty
        while estack:
            # Get expression to compile
            edscr = estack[-1]
            e = edscr[0]

            # Check if expression is named and not root (named expression and variable)
            xinfo = self.expr_infos.get(id(e))
            if xinfo:
                ename = xinfo[3]
            else:
                ename = e.name
                if ename:
                    ename = self._get_id_string(ename)
            if ename and (not root or (e is not expr) or isinstance(e, CpoAlias)):
                cout.append(ename)
                estack.pop()
                continue

            # Check constant expressions
            t = e.type
            if t.is_constant:
                estack.pop()
                if t.is_array:
                    vals = e.value
                    if len(vals) == 0:
                        cout.append(_ARRAY_TYPES[t])
                        cout.append("[]")
                    else:
                        cout.append('[')
                        _compile_integer_var_domain(vals, cout)
                        cout.append(']')
                elif t is Type_Bool:
                    self._compile_boolean_constant(e, cout)
                elif t is Type_TransitionMatrix:
                    self._compile_transition_matrix(e, cout)
                elif t is Type_TupleSet:
                    self._compile_tuple_set(e, cout)
                elif t is Type_StepFunction:
                    _compile_step_function(e, cout)
                elif t is Type_SegmentedFunction:
                    _compile_segmented_function(e, cout)
                else:
                    cout.append(_number_value_string(e.value))

            # Check variables
            elif t.is_variable:
                estack.pop()
                if t is Type_IntVar:
                    self._compile_integer_var(e, cout)
                elif t is Type_IntervalVar:
                    self._compile_interval_var(e, cout)
                elif t is Type_SequenceVar:
                    self._compile_sequence_var(e, cout)
                elif t is Type_StateFunction:
                    self._compile_state_function(e, cout)
                elif t is Type_FloatVar:
                    self._compile_float_var(e, cout)

            # Check array
            elif t.is_array:
                oprnds = e.children
                alen = len(oprnds)
                if alen == 0:
                    cout.append(_ARRAY_TYPES[t])
                    cout.append("[]")
                    estack.pop()
                else:
                    cnx = edscr[1]
                    if cnx <= 0:
                        cout.append("[")
                    if cnx >= alen:
                        cout.append("]")
                        estack.pop()
                    else:
                        edscr[1] += 1
                        if cnx > 0:
                            cout.append(", ")
                        estack.append([oprnds[cnx], 0, False])

            # General expression
            else:
                # Get operation elements
                oper = e.operation
                prio = oper.priority
                oprnds = e.children
                oplen = len(oprnds)
                cnx = edscr[1]

                # Check if function call
                if prio < 0:
                    # Check first call
                    if cnx <= 0:
                        # Check special case of sum of cumulexpr
                        if self.is_format_less_than_12_8 and oper is Oper_sum and e.type is Type_CumulExpr:
                            # Replace function call by a serie of additions
                            largs = oprnds[0].value
                            res = largs[0]
                            for v in largs[1:]:
                                res = CpoFunctionCall(Oper_plus, Type_CumulExpr, (res, v))
                            estack[-1] = [res, 0, False]
                            continue

                        cout.append(oper.keyword)
                        cout.append("(")
                    if cnx >= oplen:
                        cout.append(")")
                        estack.pop()
                    else:
                        edscr[1] += 1
                        if cnx > 0:
                            cout.append(", ")
                        estack.append([oprnds[cnx], 0, False])

                # Write operation
                else:
                    # Check parenthesis required
                    parents = edscr[2]

                    # Write operation
                    if cnx <= 0:
                        if oplen == 1:
                            cout.append(oper.keyword)
                        if parents:
                            cout.append("(")
                    if cnx >= oplen:
                        # All operands have been processed
                        if parents:
                            cout.append(")")
                        estack.pop()
                    else:
                        # Process operand
                        edscr[1] += 1
                        if cnx > 0:
                            # Add operator
                            cout.append(" " + oper.keyword + " ")
                        # Check if operand will require to have parenthesis
                        arg = oprnds[cnx]
                        nprio = arg.priority
                        # Parenthesis required if priority is greater than parent node, or if this node is not first child
                        chparnts = (nprio > prio) \
                                  or (nprio >= 5) \
                                  or ((nprio == prio) and (cnx > 0)) \
                                  or ((oplen == 1) and not parents and oprnds[0].children)
                        # Put operand on stack
                        estack.append([arg, 0, chparnts])

        # Check output exists
        if not cout:
            # Raise exception without calling str on expr (recursion)
            raise CpoException("Internal error: unable to compile expression of type {}".format(type(expr)))
        return u''.join(cout)


    def _compile_boolean_constant(self, v, cout):
        """ Compile a boolean constant in a string in CPO format
        Args:
            v:    Constant value
            cout: Output string list
        """
        cout.append("true()" if v.value else "false()")


    def _compile_integer_var(self, v, cout):
        """ Compile a integer variable in a string in CPO format
        Args:
            v:    Variable
            cout: Output string list
        """
        cout.append("intVar(")
        _compile_integer_var_domain(v.get_domain(), cout)
        cout.append(")")


    def _compile_float_var(self, v, cout):
        """ Compile a float variable in a string in CPO format
        Args:
            v:    Variable
            cout: Output string list
        """
        cout.append("floatVar(")
        cout.append(_number_value_string(v.get_domain_min()))
        cout.append(", ")
        cout.append(_number_value_string(v.get_domain_max()))
        cout.append(")")


    def _compile_interval_var(self, v, cout):
        """ Compile a IntervalVar in a string in CPO format
        Args:
            v:    Variable
            cout: Output string list
        """
        cout.append("intervalVar(")
        args = []
        if v.is_absent():
            args.append("absent")
        elif v.is_optional():
            args.append("optional")
        if v.start != DEFAULT_INTERVAL:
            args.append("start=" + _interval_var_domain_string(v.start))
        if v.end != DEFAULT_INTERVAL:
            args.append("end=" + _interval_var_domain_string(v.end))
        if v.length != DEFAULT_INTERVAL:
            args.append("length=" + _interval_var_domain_string(v.length))
        if v.size != DEFAULT_INTERVAL:
            args.append("size=" + _interval_var_domain_string(v.size))
        if v.intensity is not None:
            args.append("intensity=" + self._compile_expression(v.intensity, root=False))
        if v.granularity is not None:
            args.append("granularity=" + str(v.granularity))
        cout.append(", ".join(args) + ")")


    def _compile_integer_var_starting_point(self, v, cout):
        """ Compile a integer variable starting point in a string in CPO format
        Args:
            v:    Variable solution (CpoIntVarSolution)
            cout: Output string list
        """
        cout.append("(")
        dom = v.value
        dmin = _domain_min(dom)
        dmax = _domain_max(dom)
        cout.append(_int_var_value_string(dmin))
        if dmin != dmax:
            cout.append('..')
            cout.append(_int_var_value_string(dmax))
        cout.append(")")


    def _compile_interval_var_starting_point(self, v, cout):
        """ Compile a starting IntervalVar in a string in CPO format
        Args:
            v:    Variable solution (CpoIntervalVarSolution)
            cout: Output string list
        """
        if v.is_absent():
            cout.append("absent")
            return
        cout.append("(")
        cout.append("present" if v.is_present() else "optional")
        rng = v.get_start()
        if rng is not None:
            cout.append(", start=" + _interval_var_domain_string(rng))
        rng = v.get_end()
        if rng is not None:
            cout.append(", end=" + _interval_var_domain_string(rng))
        rng = v.get_size()
        if rng is not None:
            cout.append(", size=" + _interval_var_domain_string(rng))
        cout.append(")")


    def _compile_sequence_var(self, sv, cout):
        """ Compile a SequenceVar in a string in CPO format
        Args:
            sv:   Sequence variable
            cout: Output string list
        """
        cout.append("sequenceVar(")
        lvars = sv.get_interval_variables()
        if len(lvars) == 0:
            cout.append("intervalVarArray[]")
        else:
            cout.append("[" + ", ".join(self._get_expr_id(v) for v in lvars) + "]")
        types = sv.get_types()
        if types is not None:
            if len(lvars) == 0:
                cout.append(", intArray[]")
            else:
                cout.append(", [" + ", ".join(str(t) for t in types) + "]")
        cout.append(")")


    def _compile_state_function(self, stfct, cout):
        """ Compile a State in a string in CPO format

        Args:
           stfct: Segmented function
           cout:  Output string list
        """
        cout.append("stateFunction(")
        trmx = stfct.get_transition_matrix()
        if trmx is not None:
            cout.append(self._compile_expression(trmx, root=False))
        cout.append(")")


    @staticmethod
    def _compile_transition_matrix(tm, cout):
        """ Compile a TransitionMatrix in a string in CPO format

        Args:
            tm:   Transition matrix
            cout: Output string list
        """
        cout.append("transitionMatrix(")
        cout.append(", ".join(str(v) for v in tm.get_all_values()))
        cout.append(")")


    def _compile_tuple_set(self, e, cout):
        """ Compile a TupleSet in a string in CPO format

        Args:
           e:    Tuple set expression
           cout: Output string list
        """
        tplset = e.value
        if tplset:
            cout.append("[")
            for i, tpl in enumerate(tplset):
                if i > 0:
                    cout.append(", ")
                cout.append("[")
                _compile_integer_var_domain(tpl, cout)
                cout.append("]")
            cout.append("]")
        else:
            cout.append("tupleSet[]")


    def _pre_compile_model(self):
        """ Pre-compile model

        This method pre-compile model, compute dependencies between expressions and allocate names if required.

        Result is set in compiler attributes.
        """
        # Check null model
        model = self.model
        if model is None:
            return

        # Scan all expressions
        self.expr_infos = expr_infos = {}
        all_exprs = []
        self._scan_expressions(model.get_all_expressions(), expr_infos, all_exprs)

        # Create an alias list for KPIs
        kpiexprs = []
        kpis = model.get_kpis()
        if kpis:
            for k, (x, l) in kpis.items():
                if isinstance(x, CpoExpr) and not isinstance(x, CpoAlias):
                    if x.get_name() != k:
                        kpiexprs.append((CpoAlias(x, k), l))
                    elif not isinstance(x, CpoVariable):
                        xnfo = expr_infos.get(id(x))
                        if (xnfo is None) or (xnfo[2] == 0):
                            kpiexprs.append((x, l))

        # Add KPIs expressions
        self._scan_expressions(kpiexprs, expr_infos, all_exprs)
        # kpexprs = []
        # self._scan_expressions([(x, l) for (x, l) in kpis.values() if isinstance(x, CpoExpr)], expr_infos, kpexprs)
        # for xnfo in kpexprs:
        #     x = xnfo[0]
        #     if x.name and ((not id(x) in expr_infos) or (x.type.is_variable and xnfo[2] == 1)):
        #         all_exprs.append(xnfo)

        # Initialize lists of expressions
        list_consts = []
        list_vars = []
        vars_set = set()
        list_exprs = []
        list_phases = []
        alias_name_map = {}       # List of variable with a name but renamed shortly
        name_constraints = self.name_all_constraints
        min_length_for_alias = self.min_length_for_alias

        # Create name allocators
        id_allocators = [IdAllocator('_EXP_')] * len(ALL_TYPES)
        id_allocators[Type_IntVar.id] = IdAllocator('_INT_')
        id_allocators[Type_IntervalVar.id] = IdAllocator('_ITV_')
        id_allocators[Type_SequenceVar.id] = IdAllocator('_SEQ_')
        id_allocators[Type_StateFunction.id] = IdAllocator('_FUN_')
        id_allocators[Type_Constraint.id] = IdAllocator('_CTR_')

        # Initialize map of names
        expr_by_names = self.expr_by_names = {}

        # Force to keep original name for KPIs
        for x, l in kpiexprs:
            expr_by_names[x.get_name()] = x

        # print ("All expressions infos to process:")
        # for xinfo in all_exprs:
        #     print("   {}".format(xinfo))

        # Allocate names and split variables and constants
        factorize = self.factorize
        for xinfo in all_exprs:
            expr = xinfo[0]
            typ = expr.type
            # Allocate name if needed
            xname = expr.get_name()
            if xname:
                if xinfo[5]:
                    # Already compiled, retrieve stored name for the same expression
                    xinfo[3] = expr_infos[id(expr)][3]
                else:
                    # Check if name too long
                    if min_length_for_alias is not None and len(xname) >= min_length_for_alias and not xname in kpis:
                        xname = _allocate_expr_id(id_allocators[typ.id], expr_by_names)
                        alias_name_map[expr.get_name()] = xname
                        xinfo[3] = xname
                    else:
                        # Check if already used elsewhere
                        # if xname in expr_by_names:
                        ox = expr_by_names.get(xname)
                        if not (ox is None or ox is expr):  # To process the case of pre-allocated names of KPIs
                            # Allocate a next instance
                            ndx = 1
                            xname += "@"
                            nname = xname + "1"
                            while nname in expr_by_names:
                                ndx += 1
                                nname = xname + str(ndx)
                            xname = nname
                        xinfo[3] = to_printable_id(xname)
                    expr_by_names[xname] = expr
            elif (xinfo[3] is None) and ((factorize and (xinfo[2] > 1)) or typ.is_variable):
                # Allocate name
                xname = _allocate_expr_id(id_allocators[typ.id], expr_by_names)
                expr_by_names[xname] = expr
                xinfo[3] = xname
            elif name_constraints and xinfo[4] and (typ in (Type_Constraint, Type_BoolExpr)):
                xname = _allocate_expr_id(id_allocators[typ.id], expr_by_names)
                expr_by_names[xname] = expr
                xinfo[3] = xname

            # Split in different expression lists
            if typ.is_constant and xname:
                list_consts.append(xinfo)
            elif typ.is_variable and not isinstance(expr, CpoAlias):
                if expr not in vars_set:
                    vars_set.add(expr)
                    list_vars.append(xinfo)
            elif typ == Type_SearchPhase:
                list_phases.append(xinfo)
            elif xname or xinfo[4] or (factorize and (xinfo[2] > 1)):
                list_exprs.append(xinfo)

        # Sort list of constants and variables
        if self.sort_names == 'natural':
            sortkey = functools.cmp_to_key(_compare_xinfo_natural)
            self.list_consts = sorted(list_consts, key=sortkey)
            self.list_vars = sorted(list_vars, key=sortkey)
        elif self.sort_names == 'alphabetical':
            sortkey = functools.cmp_to_key(_compare_xinfo_alphabetical)
            self.list_consts = sorted(list_consts, key=sortkey)
            self.list_vars = sorted(list_vars, key=sortkey)
        else:
            self.list_consts = list_consts
            self.list_vars = list_vars


        # Set other attributes
        self.list_exprs = list_exprs
        self.list_phases = list_phases
        self.alias_name_map = alias_name_map


    def _scan_expressions(self, lexpr, expr_infos, all_exprs):
        """ Scan a list of expressions
        Args:
            lexpr:       List of expression to scan (list of tuples (expression, location))
            expr_infos:  Map if expression infos to update
            all_exprs:   List of all root expressions to update
        """
        # Scan all expressions
        for expr, loc in lexpr:
            eid = id(expr)
            xinfo = expr_infos.get(eid)
            if xinfo is not None:
                # Expression already compiled, add it again
                xinfo[2] += 1  # Increment reference count
            else:
                # Create new expression info
                xinfo = [expr, loc, 1, None, True, False]
                expr_infos[eid] = xinfo
                # Process children
                if expr.children:
                    # Initialize expression stack with [expression, child index]
                    stack = [[x, 0] for x in expr.children]
                    while stack:
                        xdscr = stack[-1]
                        e, cx = xdscr
                        # Skip constants
                        if e.type.is_constant_atom and not e.name:
                            stack.pop()
                            continue
                        # Check if expression already processed
                        xid = id(e)
                        xnfo = expr_infos.get(xid)
                        if xnfo:
                            xnfo[2] += 1
                            stack.pop()
                            continue
                        # Check if all children processed
                        if cx >= len(e.children):
                            # Add current children node and remove it from stack
                            xnfo = [e, loc, 1, None, False, False]
                            expr_infos[xid] = xnfo
                            all_exprs.append(xnfo)
                            stack.pop()
                        else:
                            # Push current child on the stack
                            stack.append([e.children[cx], 0])
                            xdscr[1] += 1
            # Append to the list of expressions
            all_exprs.append(xinfo)


###############################################################################
## Public functions
###############################################################################

def get_cpo_model(model, **kwargs):
    """ Convert a model into a string with CPO file format.

    Args:
        model:  Source model
    Optional args:
        context:             Global solving context. If not given, context is the default context that is set in config.py.
        params:              Solving parameters (CpoParameters) that overwrites those in solving context
        add_source_location: Add source location into generated text
        length_for_alias:    Minimum name length to use shorter alias instead
        (others):            All other context parameters that can be changed
    Returns:
        String of the model in CPO file format
    """
    cplr = CpoCompiler(model, **kwargs)
    return cplr.get_as_string()


def expr_to_string(expr):
    """ Convert a single expression into a string using CPO format

    Args:
        expr:  Expression to convert into string
    Returns:
        String of the expression in CPO file format
    """
    return CpoCompiler(None)._compile_expression(expr)


###############################################################################
## Private functions
###############################################################################

_NUMBER_CONSTANTS = {INT_MIN: "intmin", INT_MAX: "intmax",
                     INTERVAL_MIN: "intervalmin", INTERVAL_MAX: "intervalmax",
                     POSITIVE_INFINITY: "inf", NEGATIVE_INFINITY: "-inf", True: "1", False: "0"}

def _number_value_string(val):
    """ Build the string representing a number value

    This methods checks for special values INT_MIN, INT_MAX, INTERVAL_MIN and INTERVAL_MAX.

    Args:
        val: Integer value
    Returns:
        String representation of the value
    """
    try:
        s = _NUMBER_CONSTANTS.get(val)
        return s if s else str(val)
    except:
        # Case where value is not hashable, like numpy.ndarray that can be the value type
        # when numpy operand appears in the left of an overloaded operator.
        return str(val)


def _int_var_value_string(ibv):
    """ Build the string representing an integer variable domain value

    This methods checks for special values INT_MIN and INT_MAX.

    Args:
        ibv: Integer value value
    Returns:
        String representation of the value
    """
    if ibv == INT_MIN:
        return "intmin"
    elif ibv == INT_MAX:
        return "intmax"
    else:
        return str(ibv)


def _allocate_expr_id(allocator, exprmap):
    """ Allocate a new expression id checking it is not already used.

    Args:
        allocator:  Id allocator
        exprmap:    Map of existing expression names
    Returns:
        New id not in exprmap
    """
    id = allocator.allocate()
    while id in exprmap:
        id = allocator.allocate()
    return id


def _interval_var_value_string(ibv):
    """ Build the string representing an interval variable domain value

    This methods checks for special values INTERVAL_MIN and INTERVAL_MAX.

    Args:
        ibv: Interval value
    Returns:
        String representation of the value
    """
    if ibv == INTERVAL_MIN:
        return "intervalmin"
    elif ibv == INTERVAL_MAX:
        return "intervalmax"
    else:
        return str(ibv)


def _interval_var_domain_string(dom):
    """ Build the string representing an interval_var domain

    Args:
        dom: Domain interval
    Returns:
        String representation of the domain
    """
    if isinstance(dom, (list, tuple)):
        smn, smx = dom
        if smn == smx:
            return _interval_var_value_string(smn)
        return _interval_var_value_string(smn) + ".." + _interval_var_value_string(smx)
    else:
        return _interval_var_value_string(dom)


def _compile_integer_var_domain(dom, cout):
    """ Compile a integer variable domain in CPO format

    Args:
        dom:   Variable domain
        cout:  Output string list
    """
    if is_array(dom):
        imin = imax = None
        isfirst = True
        for d in dom:
            if isinstance(d, (list, tuple)):
                if imax is None:
                    imin, imax = d
                else:
                    # Check if current interval extends
                    if d[0] == imax + 1:
                        imax = d[1]
                    else:
                        _write_domain_interval(imin, imax, isfirst, cout)
                        isfirst = False
                        imin, imax = d
            else:
                if imax is None:
                    imin = imax = d
                else:
                    # Check if current interval extends
                    if d == imax + 1:
                        imax = d
                    else:
                        _write_domain_interval(imin, imax, isfirst, cout)
                        isfirst = False
                        imin = imax = d
        # Write last interval if any
        if imin is not None:
            _write_domain_interval(imin, imax, isfirst, cout)
    else:
        # Domain is a single value
        cout.append(_number_value_string(dom))


def _write_domain_interval(imin, imax, isfirst, cout):
    """ Write a domain interval
    Args:
        imin:     Interval lower bound
        imax:     Interval upper bound
        isfirst:  First interval indicator
        cout:     Output string list
    """
    if not isfirst:
        cout.append(", ")
    cout.append(_number_value_string(imin))
    if imin != imax:
        cout.append(", " if imax == imin + 1 else "..")
        cout.append(_number_value_string(imax))


def _compile_list_of_integers(lint, cout):
    """ Compile a list of integers in CPO format

    Args:
        lint:  List of integers
        cout:  Output string list
    """
    llen = len(lint)
    i = 0
    while i < llen:
        if i > 0:
            cout.append(", ")
        j = i + 1
        while (j < llen) and (lint[j] == lint[j - 1] + 1):
            j += 1
        if j > i + 1:
            cout.append(str(lint[i]) + ".." + str(lint[j - 1]))
        else:
            cout.append(str(lint[i]))
        i = j


def _compile_step_function(stfct, cout):
    """ Compile a StepFunction in a string in CPO format

    Args:
       stfct: Step function
       cout:  Output string list
    """
    cout.append("stepFunction(")
    for i, s in enumerate(stfct.get_step_list()):
        if i > 0:
            cout.append(", ")
        cout.append('(' + _number_value_string(s[0]) + ", " + str(s[1]) + ')')
    cout.append(")")


def _compile_segmented_function(sgfct, cout):
    """ Compile a SegmentedFunction in a string in CPO format

    Args:
       sgfct: Segmented function
       cout:  Output string list
    """
    cout.append("segmentedFunction(")
    cout.append(", ".join(map(to_string, sgfct.get_segment_list())))
    cout.append(")")


def _compare_xinfo_natural(x1, x2):
    """ Compare two expressions infos using natural order

    Args:
        x1: Expression info 1
        x2: Expression info 2
    Returns:
        Integer value that is negative if v1 < v2, zero if v1 == v2 and positive if v1 > v2.
    """
    # Retrieve expressions only
    x1 = x1[0]
    x2 = x2[0]
    # First sort by expression type
    if x1.type is not x2.type:
        return x1.type.id - x2.type.id
    # Check object type
    tx1 = type(x1)
    tx2 = type(x2)
    if tx1 is not tx2:
        # Alias always loss
        if tx1 is CpoAlias:
            return 1
        if tx2 is CpoAlias:
            return -1
    # Compare by name in natural order
    return compare_natural(x1.get_name(), x2.get_name())


def _compare_xinfo_alphabetical(x1, x2):
    """ Compare two expressions infos using alphabetical order

    Args:
        x1: Expression info 1
        x2: Expression info 2
    Returns:
        Integer value that is negative if v1 < v2, zero if v1 == v2 and positive if v1 > v2.
    """
    # Retrieve expressions only
    x1 = x1[0]
    x2 = x2[0]
    # First sort by expression type
    if x1.type is not x2.type:
        return x1.type.id - x2.type.id
    # Check object type
    tx1 = type(x1)
    tx2 = type(x2)
    if tx1 is not tx2:
        # Alias always loss
        if tx1 is CpoAlias:
            return 1
        if tx2 is CpoAlias:
            return -1
    # Compare by name in alphabetical order
    xn1 = x1.get_name()
    xn2 = x2.get_name()
    if xn1 is None:
        return 0 if xn2 is None else -1
    if xn2 is None:
        return 1
    return 0 if xn1 == xn2 else -1 if xn1 < xn2 else 1



