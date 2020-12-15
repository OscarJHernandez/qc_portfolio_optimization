# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# coding=utf-8
# ------------------------------
from __future__ import print_function
import sys

import six
import operator

from docplex.mp.numutils import _NumPrinter


# gendoc: ignore


class ModelPrinter(object):
    ''' Generic Printer code.
    '''

    def __init__(self):
        pass

    def get_format(self):
        """
        returns the Format object
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def extension(self):
        """
        :return: the extension of the format
        """
        return self.get_format().extension

    def printModel(self, mdl, out=None):
        """ Generic method.
            If passed with a string, uses it as a file name
            if None is passed, uses standard output.
            else assume a stream is passed and try it
        """
        mdl._resolve_pwls()
        if out is None:
            # prints on standard output
            self.print_model_to_stream(sys.stdout, mdl)
        elif isinstance(out, six.string_types):
            # a string is interpreted as a path name
            ext = self.extension()
            path = out if out.endswith(ext) else out + ext
            # SAv format requires binary mode!
            write_mode = "wb" if self.get_format().is_binary else "w"
            with open(path, write_mode) as of:
                self.print_model_to_stream(of, mdl)
                # print("* file: %s overwritten" % path)
        else:
            try:
                self.print_model_to_stream(out, mdl)
            except AttributeError:  # pragma: no cover
                pass  # pragma: no cover
                # stringio will raise an attribute error here, due to with
                # print("Cannot use this an output: %s" % str(out))
        self.post_print_hook(mdl)

    def print_model_to_stream(self, out, mdl):
        raise NotImplementedError  # pragma: no cover

    def get_var_name_encoding(self):  # pragma: no cover
        return None  # default is no encoding

    def post_print_hook(self, model):
        # this method is called after printing the model
        # can be redefined for post-print reporting
        pass


class _DisambiguateError(Exception):
    pass


# noinspection PyAbstractClass
class TextModelPrinter(ModelPrinter):
    DEFAULT_ENCODING = "ENCODING=ISO-8859-1"

    def __init__(self, comment_start, indent=1,
                 hide_user_names=False,
                 nb_digits_for_floats=3,
                 encoding=DEFAULT_ENCODING,
                 sort_variable_names=False):
        ModelPrinter.__init__(self)
        # should be elsewhere
        self.true_infinity = float('inf')

        self.line_width = 79
        # noinspection PyArgumentEqualDefault

        self._comment_start = comment_start
        self._mangle_names = hide_user_names
        self._encoding = encoding  # None is a valid value, in which case no encoding is printed
        # -----------------------
        # TODO: refactor these maps as scope objects...
        self._var_name_map = {}
        self._linct_name_map = {}  # linear constraints
        self._lc_name_map = {}  # indicators have a seperate index space.
        self._qc_name_map = {}
        self._pwl_name_map = {}
        self._lzc_name_map = {}  # lazy constraints
        self._ucc_name_map = {}  # user cut constraints
        # ------------------------

        self._rangeData = {}
        self._num_printer = _NumPrinter(nb_digits_for_floats)
        self._indent_level = indent
        self._indent_space = ' ' * indent
        self._indent_map = {1: ' '}

        self.sort_variable_names = sort_variable_names

        # which translate_method to use
        if six.PY2:
            self._translate_chars = self._translate_chars2
        else:  # pragma: no cover
            self._translate_chars = self._translate_chars3

    def _get_indent_from_level(self, level):
        cached_indent = self._indent_map.get(level)
        if cached_indent is None:
            indent = ' ' * level
            self._indent_map[level] = indent
            return indent
        else:
            return cached_indent

    @property
    def nb_digits_for_floats(self):  # pragma: no cover
        return self._num_printer.precision

    def mangle_names(self):
        """
        Actually used to decide whether to encryupt or noyt
        :return:
        """
        return self._mangle_names

    def set_mangle_names(self, mangled):
        self._mangle_names = mangled

    def is_mangling_names(self):
        return self._mangle_names

    def _print_line_comment(self, out, comment_text):
        out.write("%s %s\n" % (self._comment_start, comment_text))

    def _print_encoding(self, out):
        """
        prints the file encoding
        :return:
        """
        if self._encoding:
            self._print_line_comment(out, self._encoding)

    def _print_model_name(self, out, mdl):
        """ Redefine this method to print the model name, if necessary
        :param mdl: the model to be printed
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def _print_signature(self, out):
        """
        Prints a signature message denoting this file comes from Python Modeling Layer
        :return:
        """
        self._print_line_comment(out, "This file has been generated by DOcplex")

    @classmethod
    def _newline(cls, out, nb_lines=1):
        for _ in range(nb_lines):
            out.write("\n")

    disambiguate_try_max = 1000

    def _disambiguate(self, candidate_name, names, mobj, prefix, local_idx, try_max=disambiguate_try_max):
        # candidate_name is already in names
        # we coin successive names with index until the suffixed name is no longer in names
        k = 1
        disambiguate_fmt = '%s##%d'
        cur_name = candidate_name

        while cur_name in names:
            if k >= try_max:
                # giving up
                return self._make_prefix_name(mobj, prefix, local_idx)
            cur_name = disambiguate_fmt % (candidate_name, k)
            k += 1
        # --
        return cur_name

    def _precompute_name_dict(self, mobj_seq, prefix, indexerfn_=None):
        indexerfn = indexerfn_ or operator.attrgetter('index')
        fixed_name_dir = {}
        all_names = set()
        hide_names = self.mangle_names()

        for local_index, mobj in enumerate(mobj_seq):
            fixed_name = self.fix_name(mobj, prefix, local_index, hide_names)
            if fixed_name:
                if fixed_name in all_names:
                    # to disambiguate, start with name#index, then add ##1,2,3,
                    seed_name = '%s#%d' % (fixed_name, mobj._index)
                    fixed_name = self._disambiguate(seed_name, all_names, mobj, prefix, local_index)
                fixed_name_dir[indexerfn(mobj)] = fixed_name
                all_names.add(fixed_name)

        return fixed_name_dir

    def _precompute_name_dict_by_obj(self, mobj_seq, prefix):
        return self._precompute_name_dict(mobj_seq, prefix, indexerfn_=lambda _x: _x)

    def _num_to_string(self, num):
        # INTERNAL
        return self._num_printer.to_string(num)

    def prepare(self, model):
        self._var_name_map = self._precompute_name_dict(model.iter_variables(), prefix='x')
        self._linct_name_map = self._precompute_name_dict(model.iter_linear_constraints(), prefix='c')
        self._lc_name_map = self._precompute_name_dict(model.iter_logical_constraints(), prefix='lc')
        self._qc_name_map = self._precompute_name_dict(model.iter_quadratic_constraints(), prefix='qc')
        self._pwl_name_map = self._precompute_name_dict(model.iter_pwl_constraints(), prefix='pwl')
        self._lzc_name_map = self._precompute_name_dict_by_obj(model.iter_lazy_constraints(), prefix='l')
        self._ucc_name_map = self._precompute_name_dict_by_obj(model.iter_user_cut_constraints(), prefix='u')

        self._rangeData = {}
        for rng in model.iter_range_constraints():
            # precompute data for ranges
            # 1 name ?
            # 2 rhs is lb - constant
            # 3 bounds are (0, ub-lb)
            varname = 'Rg%s' % self.linearct_print_name(rng)
            rhs = rng.cplex_num_rhs()
            rngval = rng.cplex_range_value(do_raise=False)
            self._rangeData[rng] = (varname, rhs, rngval)

    def _var_print_name(self, dvar):
        # INTERNAL
        return self._var_name_map[dvar._index]

    def get_name_to_var_map(self, model):
        # INTERNAL
        name_to_var_map = {}
        for v in model.iter_variables():
            name_to_var_map[self._var_name_map[v._index]] = v
        return name_to_var_map

    def print_name(self, obj, name_dict):
        return name_dict.get(obj._index)  # default is None

    def linearct_print_name(self, ct):
        return self.print_name(ct, self._linct_name_map)

    def logicalct_print_name(self, indicator):
        return self.print_name(indicator, self._lc_name_map)

    def qc_print_name(self, quad_constraint):
        return self.print_name(quad_constraint, self._qc_name_map)

    @staticmethod
    def _make_prefix_name(mobj, prefix, local_index, offset=1):
        prefixed_name = "{0:s}{1:d}".format(prefix, local_index + offset)
        return prefixed_name

    from docplex.mp.compat23 import mktrans

    __raw = " -+/\\<>"
    __cooked = "_mp____"

    _str_translate_table = mktrans(__raw, __cooked)
    _unicode_translate_table = {}
    for c in range(len(__raw)):
        _unicode_translate_table[ord(__raw[c])] = ord(__cooked[c])

    @staticmethod
    def _translate_chars2(raw_name):
        # noinspection PyUnresolvedReferences
        if isinstance(raw_name, unicode):
            char_mapping = TextModelPrinter._unicode_translate_table
        else:
            char_mapping = TextModelPrinter._str_translate_table
        return raw_name.translate(char_mapping)

    @staticmethod
    def _translate_chars3(raw_name):
        return raw_name.translate(TextModelPrinter._unicode_translate_table)

    def fix_name(self, mobj, prefix, local_index, hide_names):
        # INTERNAL
        raw_name = mobj.name
        if hide_names or mobj.is_generated() or not raw_name:
            return self._make_prefix_name(mobj, prefix, local_index, offset=1)
        else:
            return self._translate_chars(raw_name)

    @staticmethod
    def _generate_linear_obj_coefs(model, linear_obj_expr):
        # INTERNAL: print all variables and their coef in the linear part of the objective
        for v in model.iter_variables():
            yield v, linear_obj_expr.unchecked_get_coef(v)

    @staticmethod
    def _generate_linear_obj_coefs_smart(model, linear_obj_expr, selected_variables):
        # INTERNAL: used to print unreferenced variables in sos and pwl constraints and their coef in the linear part
        #  of the objective, in addition to variables in the objective
        for v in selected_variables:
            yield v, linear_obj_expr.unchecked_get_coef(v)

    def _print_lexpr(self, wrapper, num_printer, var_name_map, expr, print_constant=False, allow_empty=False,
                     force_first_plus=False):
        # prints an expr to a stream
        term_iter = expr.iter_sorted_terms()
        k = expr.get_constant() if print_constant else None
        self._print_expr_iter(wrapper, num_printer, var_name_map, term_iter,
                              constant=k, allow_empty=allow_empty,
                              force_first_plus=force_first_plus)

    def _print_expr_iter(self, wrapper, num_printer, var_name_map,
                         expr_iter,
                         allow_empty=False,
                         force_first_plus=False,
                         constant=None,
                         accept_zero=False):
        num2string_fn = num_printer.to_string
        c = 0
        if self.sort_variable_names:
            def varname(x):  # use function, since auto tuple unpacking not supported in py3
                v_, _ = x
                return var_name_map[v_._index]

            expr_iter = sorted(list(expr_iter), key=varname)
        for (v, coeff) in expr_iter:
            curr_token = ''
            if not accept_zero and not coeff:
                continue  # pragma: no cover

            if coeff < 0:
                curr_token += '-'
                wrote_sign = True
                coeff = - coeff
            elif c > 0 or force_first_plus:
                # here coeff is positive, we write the '+' only if term is non-first
                curr_token += '+'
                wrote_sign = True
            else:
                wrote_sign = False

            if 1 != coeff:
                if wrote_sign:
                    curr_token += ' '
                curr_token += num2string_fn(coeff)
            if wrote_sign or 1 != coeff:
                curr_token += ' '
            curr_token += var_name_map[v._index]

            wrapper.write(curr_token)
            c += 1

        printed_k = True
        if constant is not None:
            # here constant is a number
            if constant:
                if constant > 0:
                    if c > 0 or force_first_plus:
                        wrapper.write('+')
                wrapper.write(num2string_fn(constant))
            elif 0 == c and not allow_empty:
                wrapper.write('0')
            else:
                printed_k = False

        else:
            # constant is none here
            if not c and not allow_empty:
                # expr is empty, if we must print something, print 0
                wrapper.write('0')
            else:
                printed_k = False
        return c or printed_k

    def _print_qexpr_obj(self, wrapper, num_printer, var_name_map, quad_expr, force_initial_plus, use_double=True):
        # writes a quadratic expression
        # in the form [ 2a_ij a_i.a_j ] / 2
        # Note that all coefficients must be doubled due to the tQXQ formulation

        if force_initial_plus:
            wrapper.write('+')

        return self._print_qexpr_iter(wrapper, num_printer, var_name_map, quad_expr.iter_sorted_quads(),
                                      use_double=use_double)

    def _print_qexpr_iter(self, wrapper, num_printer, var_name_map, iter_quads, use_double=False):
        q = 0
        varname_getter = self._var_print_name
        for qvp, qk in iter_quads:
            curr_token = ''
            if 0 == qk:
                continue  # pragma: no cover
            if q == 0:
                wrapper.write('[')  # only once
            abs_qk = qk
            if qk < 0:
                curr_token += '-'
                abs_qk = - qk
                wrote_sign = True
            elif q > 0:
                curr_token += '+'
                wrote_sign = True
            else:
                wrote_sign = False
            if wrote_sign:
                curr_token += ' '

            # all coefficients must be doubled because of the []/2 pattern.
            abs_qk2 = 2 * abs_qk if use_double else abs_qk
            if abs_qk2 != 1:
                curr_token += num_printer.to_string(abs_qk2)
                curr_token += ' '

            if qvp.is_square():
                qv_name = varname_getter(qvp[0])
                curr_token += "%s^2" % qv_name
            else:
                qv1 = qvp[0]
                qv2 = qvp[1]
                curr_token += "%s*%s" % (varname_getter(qv1), varname_getter(qv2))

            wrapper.write(curr_token)

            q += 1

        if q:
            closer = ']/2' if use_double else ']'
            wrapper.write(closer)
        return q


class _ExportWrapper(object):
    """
    INTERNAL.
    """
    __new_line_sep = '\n'

    def __init__(self, oss, indent_str, line_width=80):
        self._oss = oss
        self._indent_str = indent_str
        self._line_width = line_width
        self._curr_line = ''
        self._wrote = False

    def wrote(self):  # pragma: no cover
        return not self._wrote

    def set_indent(self, new_indent):
        self._indent_str = new_indent

    def begin_line(self, indented=False):
        # reset dynamic line data
        self._wrote = False
        self._curr_line = self._indent_str if indented else ''

    # The 'write' function is invoked intensively when exporting a model.
    # Any piece of code that can be saved here will improve performance in a visible way.
    def write(self, token, separator=True):
        if token is not None:
            if len(self._curr_line) + len(token) >= self._line_width:
                # faster to write concatenated string, slightly faster to use '\n' instead of ref to static value
                self._oss.write(self._curr_line + '\n')
                # oss.write('\n')
                self._curr_line = self._indent_str + token
            else:
                # print one separator --before-- the token to be printed
                if separator and self._wrote:
                    self._curr_line += (' ' + token)
                else:
                    self._curr_line += token
            self._wrote = True

    def flush(self, print_newline=True, restart_from_empty_line=False):
        self._oss.write(self._curr_line)
        if print_newline:
            self._oss.write('\n')
        # Reset '_wrote' flag so that no separator will be added when writing first token of next line
        self._wrote = False
        # if reset, start a new line.
        self._curr_line = '' if restart_from_empty_line else self._indent_str

    def newline(self):
        self._oss.write('\n')
