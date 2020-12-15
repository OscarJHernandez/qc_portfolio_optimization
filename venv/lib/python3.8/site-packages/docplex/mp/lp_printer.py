# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


from __future__ import print_function


from docplex.mp.constants import ComparisonType
from docplex.mp.constr import LinearConstraint, RangeConstraint, QuadraticConstraint, PwlConstraint
from docplex.mp.environment import env_is_64_bit
from docplex.mp.mprinter import TextModelPrinter, _ExportWrapper, _NumPrinter
from docplex.mp.utils import fix_whitespace

from docplex.mp.format import LP_format
from itertools import chain
from six import iteritems, PY3
from docplex.mp.compat23 import izip


# gendoc: ignore

def _non_compliant_lp_name_stop_here(name):
    pass


class LPModelPrinter(TextModelPrinter):
    #_lp_re = re.compile(r"[a-df-zA-DF-Z!#$%&()/,;?@_`'{}|\"][a-zA-Z0-9!#$%&()/.,;?@_`'{}|\"]*")

    _lp_symbol_map = {ComparisonType.EQ: " = ",  # BEWARE NOT ==
                      ComparisonType.LE: " <= ",
                      ComparisonType.GE: " >= "}

    __new_line_sep = '\n'
    __expr_indent = ' ' * 6

    float_precision_32 = 9
    float_precision_64 = 12  #
    _nb_noncompliant_ids = 0
    _noncompliant_justifier = None

    def __init__(self, hide_user_names=False, **kwargs):
        nb_digits = self.float_precision_64 if env_is_64_bit() else self.float_precision_32
        TextModelPrinter.__init__(self,
                                  indent=1,
                                  comment_start='\\',
                                  hide_user_names=hide_user_names,
                                  nb_digits_for_floats=nb_digits)

        self._noncompliant_varname = None
        # specific printer for lp: do not print +inf/-inf inside constraints!
        self._lp_num_printer = _NumPrinter(nb_digits_for_floats=nb_digits,
                                           num_infinity=1e+20, pinf="1e+20", ninf="-1e+20")
        self._print_full_obj = kwargs.get('full_obj', False)

    def get_format(self):
        return LP_format

    def mangle_names(self):
        return TextModelPrinter.mangle_names(self) or self._noncompliant_varname

    def _print_ct_name(self, ct, name_map):
        lp_ctname = name_map.get(ct._index)
        indented = self._indent_level

        if lp_ctname is not None:
            ct_label = self._indent_space + lp_ctname + ':'
            indented += len(ct_label)
        else:
            ct_label = ''
        ct_indent_space = self._get_indent_from_level(indented)
        return ct_indent_space, ct_label

    def _print_binary_ct(self, wrapper, num_printer, var_name_map, binary_ct, _symbol_map=_lp_symbol_map,
                         allow_empty=False, force_first_sign=False):
        # ensure consistent ordering: left terms then right terms
        iter_diff_coeffs = binary_ct.iter_net_linear_coefs()
        self._print_expr_iter(wrapper, num_printer, var_name_map, iter_diff_coeffs,
                              allow_empty=True,  # when expr is empty print nothing, otherwise CPLEX crashes...!!!
                              force_first_plus=force_first_sign)
        wrapper.write(_symbol_map.get(binary_ct.sense, " ?? "), separator=False)
        wrapper.write(num_printer.to_string(binary_ct.cplex_num_rhs()), separator=False)

    def _print_ranged_ct(self, wrapper, num_printer, var_name_map, ranged_ct):
        exp = ranged_ct.expr
        (varname, rhs, _) = self._rangeData[ranged_ct]
        self._print_lexpr(wrapper, num_printer, var_name_map, exp)
        wrapper.write('-', separator=False)
        wrapper.write(varname)
        wrapper.write('=')
        wrapper.write(self._num_to_string(rhs))

    def _print_logical_ct(self, wrapper, num_printer, var_name_map, logical_ct,
                          logical_symbol):
        wrapper.write(self._var_print_name(logical_ct.binary_var))
        # rtc-39773 : write " = 1 -> " as one atomic symbol
        wrapper.write("= %d %s" % (logical_ct.active_value, logical_symbol))
        # wrapper.write("%d" % logical_ct.active_value)
        # wrapper.write(logical_symbol)
        self._print_binary_ct(wrapper, num_printer, var_name_map, logical_ct.linear_constraint)

    def _print_quadratic_ct(self, wrapper, num_printer, var_name_map, qct):
        q = self._print_qexpr_iter(wrapper, num_printer, var_name_map, qct.iter_net_quads())
        # force a '+' ?
        has_quads = q > 0
        self._print_binary_ct(wrapper, num_printer, var_name_map, qct, allow_empty=has_quads,
                              force_first_sign=has_quads)

    def _print_pwl_ct(self, wrapper, num_printer, var_name_map, pwl):
        """
        Prints a PWL ct in LP
        :param wrapper:
        :param pwl
        :return:
        """
        num2string_fn = num_printer.to_string
        wrapper.write('%s = %s' % (var_name_map[pwl.y._index], var_name_map[pwl.expr._index]))
        pwl_func = pwl.pwl_func
        pwl_def = pwl_func.pwl_def_as_breaks
        wrapper.write('%s' % num2string_fn(pwl_def.preslope))
        for pair in pwl_def.breaksxy:
            wrapper.write('(%s, %s)' % (num2string_fn(pair[0]), num2string_fn(pair[1])))
        wrapper.write('%s' % num2string_fn(pwl_def.postslope))

    def _print_constraint_label(self, wrapper, ct, name_map):
        if self._mangle_names:
            wrapper.set_indent('')
        else:
            indent_str, ct_label = self._print_ct_name(ct, name_map=name_map)
            wrapper.set_indent(indent_str)
            if ct_label is not None:
                wrapper.write(ct_label)

    def _print_constraint(self, wrapper, num_printer, var_name_map, ct):
        if isinstance(ct, PwlConstraint):
            # Pwl constraints are printed in a separate section (names 'PWL')
            return

        wrapper.begin_line()
        if isinstance(ct, LinearConstraint):
            self._print_constraint_label(wrapper, ct, name_map=self._linct_name_map)
            self._print_binary_ct(wrapper, num_printer, var_name_map, ct)
        elif isinstance(ct, RangeConstraint):
            self._print_constraint_label(wrapper, ct, name_map=self._linct_name_map)
            self._print_ranged_ct(wrapper, num_printer, var_name_map, ct)
        elif ct.is_logical():
            is_eq = ct.is_equivalence()
            logical_symbol = '<->' if is_eq else '->'
            self._print_constraint_label(wrapper, ct, name_map=self._lc_name_map)
            self._print_logical_ct(wrapper, num_printer, var_name_map, ct,
                                   logical_symbol=logical_symbol
                                   )
        elif isinstance(ct, QuadraticConstraint):
            self._print_constraint_label(wrapper, ct, name_map=self._qc_name_map)
            self._print_quadratic_ct(wrapper, num_printer, var_name_map, ct)
        else:
            ct.error("ERROR: unexpected constraint not printed: {0!s}".format(ct))  # pragma: no cover

        wrapper.flush(print_newline=True, restart_from_empty_line=True)

    def _print_pwl_constraint(self, wrapper, num_printer, var_name_map, ct):
        wrapper.begin_line()
        self._print_constraint_label(wrapper, ct, name_map=self._pwl_name_map)
        self._print_pwl_ct(wrapper, num_printer, var_name_map, ct)
        wrapper.flush(print_newline=True, restart_from_empty_line=True)

    def _print_var_block(self, wrapper, iter_vars, header):
        wrapper.begin_line()
        printed_header = False
        self_indent = self._indent_space
        for v in iter_vars:
            lp_name = self._var_print_name(v)
            if not printed_header:
                wrapper.newline()
                wrapper.write(header)
                printed_header = True
                wrapper.set_indent(self_indent)
                # Configure indent for next lines
                wrapper.flush(print_newline=True)
            wrapper.write(lp_name)
        if printed_header:
            wrapper.flush(print_newline=True)

    def _print_var_bounds(self, out, num_printer, varname, lb, ub, varname_indent=5 * ' ',
                          le_symbol='<=',
                          free_symbol='Free'):
        if lb is None and ub is None:
            # try to indent with space of '0 <= ', that is 5 space
            out.write(" %s %s %s\n" % (varname_indent, varname, free_symbol))
        elif lb is None:
            out.write(" %s %s %s %s\n" % (varname_indent, varname, le_symbol, num_printer.to_string(ub)))
        elif ub is None:
            out.write(" %s %s %s\n" % (num_printer.to_string(lb), le_symbol, varname))
        elif lb == ub:
            out.write(" %s %s %s %s\n" % (varname_indent, varname, "=", num_printer.to_string(lb)))
        else:
            out.write(" %s %s %s %s %s\n" % (num_printer.to_string(lb),
                                            le_symbol, varname, le_symbol,
                                            num_printer.to_string(ub)))

    TRUNCATE = 200

    def _notify_new_non_compliant_name(self, non_lp_name):
        _non_compliant_lp_name_stop_here(non_lp_name)
        self._nb_noncompliant_ids += 1
        if not self._noncompliant_justifier:
            self._noncompliant_justifier = non_lp_name

    def fix_name(self, mobj, prefix, local_index, hide_names):
        return LP_format.lp_name(mobj.name, prefix, local_index, hide_names,
                                 noncompliant_hook=self._notify_new_non_compliant_name)

    def _print_model_name(self, out, model):
        model_name = None
        if model.name:
            # make sure model name is ascii
            encoded = model.name.encode('ascii', 'backslashreplace')
            if PY3:  # pragma: no cover
                # in python 3, encoded is a bytes at this point. Make it a string again
                encoded = encoded.decode('ascii')
            model_name = encoded.replace('\\\\', '_').replace('\\', '_')
        printed_name = model_name or 'CPLEX'
        out.write("\\Problem name: %s\n" % printed_name)

    @staticmethod
    def is_lp_compliant(name, do_fix_whitespace=True):
        fixed_name = fix_whitespace(name) if do_fix_whitespace else name
        return LP_format.is_lp_compliant(fixed_name)

    @staticmethod
    def _is_injective(name_map):
        nb_keys = len(name_map)
        nb_different_names = len(set(name_map.values()))
        return nb_different_names == nb_keys

    @staticmethod
    def _has_sos_or_pwl_constraints(model):
        return model.number_of_sos > 0 or model.number_of_pwl_constraints > 0

    def _iter_completed_linear_obj_terms(self, model, objlin):
        obj_variables = set(v for v, _ in objlin.iter_terms())  # not sorted
        predeclared_variables = self._get_forced_predeclared_variables(model)
        variables_to_display = predeclared_variables | obj_variables
        # sort by index
        sorted_display_vars = list(variables_to_display)
        sorted_display_vars.sort(key=lambda dv: dv._index)
        for v in sorted_display_vars:
            yield v, objlin.unchecked_get_coef(v)

    def _iter_full_linear_obj_terms(self, model, objlin):
        # INTERNAL: print all variables and their coef in the linear part of the objective
        for v in model.iter_variables():
            yield v, objlin.unchecked_get_coef(v)

    def _get_forced_predeclared_variables(self, model):
        # compute predeclared variables
        predeclared_variables = set()
        for sos in model.iter_sos():
            for sos_var in sos.iter_variables():
                predeclared_variables.add(sos_var)
        for pwlc in model.iter_pwl_constraints():
            for pwv in pwlc.iter_extended_variables():
                predeclared_variables.add(pwv)
        return predeclared_variables

    def post_print_hook(self, model):
        nb_non_compliants = self._nb_noncompliant_ids
        if nb_non_compliants:
            try:
                model.warning('Some identifiers are not valid LP identifiers: %d (e.g.: "%s")',
                               nb_non_compliants, self._noncompliant_justifier)
            except UnicodeEncodeError:  # pragma: no cover
                model.warning('Some identifiers are not valid LP identifiers: %d (e.g.: "%s")',
                              nb_non_compliants, self._noncompliant_justifier.encode('utf-8'))

    def print_single_objective(self, wrapper, model, num_printer, var_name_map):
        wrapper.write(' obj:')
        objexpr = model.objective_expr

        if objexpr.is_quad_expr():
            objlin = objexpr.linear_part
        else:
            objlin = objexpr

        if self._print_full_obj:
            iter_linear_terms = self._iter_full_linear_obj_terms(model, objlin)
        elif self._has_sos_or_pwl_constraints(model):
            iter_linear_terms = self._iter_completed_linear_obj_terms(model, objlin)
        else:
            # write the linear part first
            # prints an expr to a stream
            iter_linear_terms = objlin.iter_sorted_terms()

        printed= self._print_expr_iter(wrapper, num_printer, var_name_map, iter_linear_terms,
                              allow_empty=True, accept_zero=True)

        if objexpr.is_quad_expr() and objexpr.has_quadratic_term():
            self._print_qexpr_obj(wrapper, num_printer, var_name_map,
                                  quad_expr=objexpr,
                                  force_initial_plus=printed)
            printed = True

        obj_offset = objexpr.get_constant()
        if obj_offset:
            if printed and obj_offset > 0:
                wrapper.write(u'+')
            wrapper.write(self._num_to_string(obj_offset))
        # ---
        wrapper.flush(print_newline=True)


    #  @profile
    def print_model_to_stream(self, out, model):
        # reset noncompliant stats
        self._nb_noncompliant_ids = 0
        self._noncompliant_justifier = None

        if not self._is_injective(self._var_name_map):
            # use indices to differentiate names
            import sys
            sys.__stdout__.write("DOcplex: refine variable names\n")
            k = 0
            for dv, lp_varname in iteritems(self._var_name_map):
                refined_name = "%s#%d" % (lp_varname, k)
                self._var_name_map[dv] = refined_name
                k += 1

        TextModelPrinter.prepare(self, model)
        self_num_printer = self._lp_num_printer
        var_name_map = self._var_name_map
        line_size = model.lp_line_length
        wrapper = _ExportWrapper(out, indent_str=self.__expr_indent,line_width=line_size)

        self._print_signature(out)
        self._print_encoding(out)
        self._print_model_name(out, model)
        self._newline(out)

        # ---  print objective
        if model.has_multi_objective():

            out.write(model.objective_sense.name)
            out.write(' multi-objectives')
            self._newline(out)
            env = model.environment
            if env.has_cplex and env.cplex_version.startswith("12.9."):
                mobj_indent = 5
                def make_default_obj_name(o_):
                    return 'obj%d' % o_ if o_ > 0 else 'obj'
            else:
                mobj_indent = 7 # length of 'obj1: '
                def make_default_obj_name(o_):
                    return 'obj%d' % (o_+1)

            wrapper.set_indent(mobj_indent * ' ') # length of 'obj1: '
            for o, ot in enumerate(model.iter_multi_objective_tuples()):
                expr, prio, w, abstol, reltol, oname = ot
                name = oname or make_default_obj_name(o)
                obj_label = '%s%s:' % (' ', name)
                wrapper.write(obj_label)
                wrapper.write('Priority=%g Weight=%g AbsTol=%g RelTol=%g' % (prio, w, abstol, reltol))
                wrapper.flush()
                # print the expression and its constant
                printed = self._print_expr_iter(wrapper, self_num_printer, var_name_map, expr.iter_terms(),
                                   allow_empty=True, accept_zero=True)
                obj_offset = expr.get_constant()
                if obj_offset:
                    if printed and obj_offset > 0:
                        wrapper.write(u'+')
                    wrapper.write(self_num_printer.to_string(obj_offset))

                wrapper.flush(restart_from_empty_line=True)

        else:
            out.write(model.objective_sense.name)
            self._newline(out)
            self.print_single_objective(wrapper, model, self_num_printer, var_name_map)

        # wrapper.write(' obj:')
        # objexpr = model.objective_expr
        #
        # if objexpr.is_quad_expr():
        #     objlin = objexpr.linear_part
        # else:
        #     objlin = objexpr
        #
        # if self._print_full_obj:
        #     iter_linear_terms = self._iter_full_linear_obj_terms(model, objlin)
        # elif self._has_sos_or_pwl_constraints(model):
        #     iter_linear_terms = self._iter_completed_linear_obj_terms(model, objlin)
        # else:
        #     # write the linear part first
        #     # prints an expr to a stream
        #     iter_linear_terms = objlin.iter_sorted_terms()
        #
        # printed= self._print_expr_iter(wrapper, self_num_printer, var_name_map, iter_linear_terms,
        #                       allow_empty=True, accept_zero=True)
        #
        # if objexpr.is_quad_expr() and objexpr.has_quadratic_term():
        #     self._print_qexpr_obj(wrapper, self_num_printer, var_name_map,
        #                           quad_expr=objexpr,
        #                           force_initial_plus=printed)
        #     printed = True
        #
        # obj_offset = objexpr.get_constant()
        # if obj_offset:
        #     if printed and obj_offset > 0:
        #         wrapper.write(u'+')
        #     wrapper.write(self._num_to_string(obj_offset))
        # # ---
        #
        # wrapper.flush(print_newline=True)

        out.write("Subject To\n")

        for ct in model.iter_constraints():
            self._print_constraint(wrapper, self_num_printer, var_name_map, ct)
        # for lct in model._iter_implicit_equivalence_cts():
        #     wrapper.begin_line(True)
        #     self._print_logical_ct(wrapper, self_num_printer, var_name_map, lct, '<->')
        #     wrapper.flush(restart_from_empty_line=True)

        # lazy constraints
        self.print_linear_constraint_section(out, wrapper, model.iter_lazy_constraints(),
                                             "Lazy Constraints", ct_prefix="l", var_name_map=var_name_map,
                                             ctname_map=self._lzc_name_map)
        self.print_linear_constraint_section(out, wrapper, model.iter_user_cut_constraints(),
                                             "User Cuts", ct_prefix="u", var_name_map=var_name_map,
                                             ctname_map=self._ucc_name_map)

        out.write("\nBounds\n")
        symbolic_num_printer = self._num_printer
        print_var_bounds_fn = self._print_var_bounds
        var_print_name_fn = self._var_print_name
        for dvar in model.iter_variables():
            lp_varname = var_print_name_fn(dvar)
            var_lb = dvar.lb
            var_ub = dvar.ub
            if dvar.is_binary():
                print_var_bounds_fn(out, self_num_printer, lp_varname, var_lb, var_ub)
            else:

                free_lb = dvar.has_free_lb()
                free_ub = dvar.has_free_ub()
                if free_lb and free_ub:
                    print_var_bounds_fn(out, self_num_printer, lp_varname, lb=None, ub=None)
                elif free_ub:
                    # print only nonzero lbs
                    if var_lb:
                        print_var_bounds_fn(out, symbolic_num_printer, lp_varname, var_lb, ub=None)

                else:
                    # save the lb if is zero
                    printed_lb = None if 0 == var_lb else var_lb
                    print_var_bounds_fn(out, symbolic_num_printer, lp_varname, lb=printed_lb, ub=var_ub)

        # add ranged cts vars
        for rng in model.iter_range_constraints():
            (varname, _, rngval) = self._rangeData[rng]
            self._print_var_bounds(out, self_num_printer, varname, rngval, 0)

        iter_semis = chain(model.iter_semicontinuous_vars(), model.iter_semiinteger_vars())

        self._print_var_block(wrapper, model.iter_binary_vars(), 'Binaries')
        self._print_var_block(wrapper, chain(model.iter_integer_vars(), model.iter_semiinteger_vars()), 'Generals')
        self._print_var_block(wrapper, iter_semis, 'Semi-continuous')
        self._print_sos_block(wrapper, model)
        self._print_pwl_block(wrapper, model, self_num_printer, var_name_map)
        out.write("End\n")

    def print_linear_constraint_section(self, out, wrapper, ct_iter, section_label, ct_prefix, var_name_map, ctname_map):
        ct_count = 0
        self_indent = self._indent_space
        self_num_printer = self._lp_num_printer
        for lct in ct_iter:
            if not ct_count:
                out.write("\n{0}\n".format(section_label))
            ct_count += 1
            ct_name = ctname_map.get(lct) or '%s%d' % (ct_prefix, ct_count)
            ct_label = '%s%s:' % (self_indent, ct_name)
            wrapper.set_indent(self._get_indent_from_level(len(ct_label)+1))

            # print a normal linear constraint
            wrapper.write(ct_label)
            self._print_binary_ct(wrapper, self_num_printer, var_name_map, lct)
            wrapper.flush(print_newline=True, restart_from_empty_line=True)

    def _print_sos_block(self, wrapper, mdl):
        if mdl.number_of_sos > 0:
            wrapper.write('SOS')
            wrapper.flush(print_newline=True)
            name_fn = self._var_print_name
            for s, sos in enumerate(mdl.iter_sos(), start=1):
                # always print a name string, s1, s2, ... if no name
                sos_name = sos.get_name() or 's%d' % s
                wrapper.write('%s:' % sos_name)
                wrapper.write('S%d ::' % sos.sos_type.value)  # 1 or 2
                ranks = sos.weights
                # noinspection PyArgumentList
                for rank, sos_var in izip(ranks, sos._variables):
                    wrapper.write('%s : %d' % (name_fn(sos_var), rank))
                wrapper.flush(print_newline=True)

    def _print_pwl_block(self, wrapper, mdl, self_num_printer, var_name_map):
        if mdl.number_of_pwl_constraints > 0:
            wrapper.write('Pwl')
            wrapper.flush(print_newline=True)
            # name_fn = self._var_print_name
            for pwl in mdl.iter_pwl_constraints():
                self._print_pwl_constraint(wrapper, self_num_printer, var_name_map, pwl)
