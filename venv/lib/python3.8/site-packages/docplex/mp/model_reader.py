# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import os

# docplex
from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException
from docplex.mp.environment import Environment

from docplex.mp.params.cplex_params import get_params_from_cplex_version
from docplex.mp.constants import ComparisonType
from docplex.mp.constr import LinearConstraint

from docplex.mp.cplex_adapter import CplexAdapter

from docplex.mp.compat23 import izip
from docplex.mp.quad import VarPair


class ModelReaderError(DOcplexException):
    pass


class _CplexReaderFileContext(object):
    def __init__(self, filename, read_method=None):
        self._cplex = None
        self._filename = filename
        self._read_method = read_method or ["read"]

    def __enter__(self):
        self.cpx_adapter = CplexAdapter()
        cpx = self.cpx_adapter.cpx
        # no output from CPLEX
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_error_stream(None)
        self_read_fn = cpx
        for m in self._read_method:
            self_read_fn = self_read_fn.__getattribute__(m)

        try:
            self_read_fn(self._filename)
            self._cplex = cpx
            return self.cpx_adapter

        except self.cpx_adapter.CplexError as cpx_e:  # pragma: no cover
            # delete cplex instance
            del cpx
            raise ModelReaderError("*CPLEX error {0!s} reading file {1} - exiting".format(cpx_e, self._filename))

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        cpx = self._cplex
        if cpx is not None:
            del cpx
            self._cplex = None


def compute_full_dotf(mdl, coefs, constant=0):
    """

    :param constant:
    :param mdl:
    :param coefs:
    :return:
    """
    def coef_fn(dvx):
        return coefs[dvx]

    lfactory = mdl._lfactory
    terms_dict = lfactory._new_term_dict()
    for dv in mdl.iter_variables():
        coef = coef_fn(dv.index)
        if coef:
            terms_dict[dv] = coef
    linear_expr = lfactory.linear_expr(terms_dict, constant=constant, safe=True)
    return linear_expr


def compute_full_dot(mdl, coefs, constant=0):
    """

    :param constant:
    :param mdl:
    :param coefs:
    :return:
    """
    lfactory = mdl._lfactory
    terms_dict = lfactory._new_term_dict()
    for dv, k in zip(mdl.iter_variables(), coefs):
        if k:
            terms_dict[dv] = k
    linear_expr = lfactory.linear_expr(terms_dict, constant=constant, safe=True)
    return linear_expr


# noinspection PyArgumentList
class ModelReader(object):
    """ This class is used to read models from CPLEX files (e.g.  SAV, LP, MPS)

    Note:
        This class requires CPLEX to be installed and present in ``PYTHONPATH``. The following file formats are
        accepted: LP, SAV, MPS.

    Example:
        Use the class method ``read`` to read a model file.

        Reads the contents of file ``mymodel.sav`` into an `AdvModel` instance, built with the context `my_ctx`,
        with the parameter ``ignore_names`` set to True::

            m = ModelReader.read(path='mymodel.lp', model_class=AdvModel, context=my_ctx, ignore_names=True)

    """

    @staticmethod
    def _build_linear_expr_from_sparse_pair(lfactory, var_map, cpx_sparsepair):
        expr = lfactory.linear_expr(arg=0, safe=True)
        for ix, k in izip(cpx_sparsepair.ind, cpx_sparsepair.val):
            dv = var_map[ix]
            expr._add_term(dv, k)
        return expr

    _sense2comp_dict = {'L': ComparisonType.LE, 'E': ComparisonType.EQ, 'G': ComparisonType.GE}

    # noinspection PyDefaultArgument
    @classmethod
    def parse_sense(cls, cpx_sense, sense_dict=_sense2comp_dict):
        return sense_dict.get(cpx_sense)

    @classmethod
    def read_prm(cls, filename):
        """ Reads a CPLEX PRM file.

        Reads a CPLEX parameters file and returns a DOcplex parameter group
        instance. This parameter object can be used in a solve().

        Args:
            filename: a path string

        Returns:
            A `RootParameterGroup object`, if the read operation succeeds, else None.
        """
        # TODO: Clean up - now creating an adapter raise importError if CPLEX not found
        # if not Cplex:  # pragma: no cover
        #    raise RuntimeError("ModelReader.read_prm() requires CPLEX runtime.")
        with _CplexReaderFileContext(filename, read_method=["parameters", "read_file"]) as adapter:
            cpx = adapter.cpx
            if cpx:
                # raw parameters
                params = get_params_from_cplex_version(cpx.get_version())
                for param in params:
                    try:
                        cpx_value = cpx._env.parameters._get(param.cpx_id)
                        if cpx_value != param.default_value:
                            param.set(cpx_value)

                    except adapter.CplexError:  # pragma: no cover
                        pass
                return params
            else:  # pragma: no cover
                return None

    @staticmethod
    def _safe_call_get_names(cpx_adapter, get_names_fn, fallback_names=None):
        # cplex crashes when calling get_names on some files (e.g. SAV)
        # in this case filter out error 1219
        # and return a fallback list with None or ""
        try:
            names = get_names_fn()
            return names
        # except TypeError:
        #     print("** type error ignored in call to {0}".format(get_names_fn.__name__))
        #     return fallback_names or []

        except cpx_adapter.CplexSolverError as cpxse:  # pragma: no cover
            errcode = cpxse.args[2]
            # when all indicators have no names, cplex raises this error
            # CPLEX Error  1219: No names exist.
            if errcode == 1219:
                return fallback_names or []
            else:
                # this is something else
                raise

    @classmethod
    def _read_cplex(cls, filename, silent=True):
        cpx_adapter = CplexAdapter()
        cpx = cpx_adapter.cpx
        # no warnings
        if silent:
            cpx.set_results_stream(None)
            cpx.set_log_stream(None)
            cpx.set_warning_stream(None)
            cpx.set_error_stream(None)  # remove messages about names
        try:
            cpx.read(filename)
            return cpx_adapter
        except cpx_adapter.CplexError as cpx_e:
            raise ModelReaderError("*CPLEX error {0!s} reading file {1} - exiting".format(cpx_e, filename))

    @classmethod
    def _make_expr_from_varmap_coefs(cls, lfactory, varmap, var_indices, coefs, offset=0):
        if Environment.env_is_python36:
            terms_dict = {varmap.get(dvx): k for dvx, k in izip(var_indices, coefs)}
        else:
            terms_dict = lfactory._new_term_dict()
            for dvx, k in izip(var_indices, coefs):
                dv = varmap.get(dvx)
                if dv is not None:
                    terms_dict[dv] = k
        return lfactory.linear_expr(arg=terms_dict, constant=offset, safe=True)

    @classmethod
    def read(cls, filename, model_name=None, verbose=False, model_class=None, **kwargs):
        """ Reads a model from a CPLEX export file.

        Accepts all formats exported by CPLEX: LP, SAV, MPS.

        If an error occurs while reading the file, the message of the exception
        is printed and the function returns None.

        Args:
            filename: The file to read.
            model_name: An optional name for the newly created model. If None,
                the model name will be the path basename.
            verbose: An optional flag to print informative messages, default is False.
            model_class: An optional class type; must be a subclass of Model.
                The returned model is built using this model_class and the keyword arguments kwargs, if any.
                By default, the model is class is `Model` (see
            kwargs: A dict of keyword-based arguments that are used when creating the model
                instance.

        Example:
            `m = read_model("c:/temp/foo.mps", model_name="docplex_foo", solver_agent="docloud", output_level=100)`

        Returns:
            An instance of Model, or None if an exception is raised.

        See Also:
            :class:`docplex.mp.model.Model`

        """
        if not os.path.exists(filename):
            raise IOError("* file not found: {0}".format(filename))

        # extract basename
        if model_name:
            name_to_use = model_name
        else:
            basename = os.path.basename(filename)
            if '.' not in filename:
                raise RuntimeError('ModelReader.read_model(): path has no extension: {}'.format(filename))
            dotpos = basename.find(".")
            if dotpos > 0:
                name_to_use = basename[:dotpos]
            else:  # pragma: no cover
                name_to_use = basename

        model_class = model_class or Model

        if 0 == os.stat(filename).st_size:
            print("* file is empty: {0} - exiting".format(filename))
            return model_class(name=name_to_use, **kwargs)

        if verbose:
            print("-> CPLEX starts reading file: {0}".format(filename))
        cpx_adapter = cls._read_cplex(filename)
        cpx = cpx_adapter.cpx
        if verbose:
            print("<- CPLEX finished reading file: {0}".format(filename))

        if not cpx:  # pragma: no cover
            return None

        final_output_level = kwargs.get("output_level", "info")
        debug_read = kwargs.get("debug", False)

        try:
            # force no tck
            if 'checker' in kwargs:
                final_checker = kwargs['checker']
            else:
                final_checker = 'default'
            # build the model with no checker, then restore final_checker in the end.
            kwargs['checker'] = 'off'

            ignore_names = kwargs.get('ignore_names', False)
            # -------------

            mdl = model_class(name=name_to_use, **kwargs)
            mdl._provenance = filename
            lfactory = mdl._lfactory
            qfactory = mdl._qfactory
            mdl.set_quiet()  # output level set to ERROR
            vartype_cont = mdl.continuous_vartype
            vartype_map = {'B': mdl.binary_vartype,
                           'I': mdl.integer_vartype,
                           'C': mdl.continuous_vartype,
                           'S': mdl.semicontinuous_vartype}

            def cpx_type_to_docplex_type(cpxt):
                return vartype_map.get(cpxt, vartype_cont)
            # 1 upload variables
            cpx_nb_vars = cpx.variables.get_num()

            if verbose:
                print("-- uploading {0} variables...".format(cpx_nb_vars))

            cpx_var_names = [] if ignore_names else cls._safe_call_get_names(cpx_adapter,
                                                                             cpx.variables.get_names)

            # check whether all varianbles have same type, it's worth it.
            unique_type = None
            cpx_vartypes = []
            if cpx._is_MIP():
                cpx_types = cpx.variables.get_types()
                first_cpxtype = cpx_types[0]
                if all(cpxt == first_cpxtype for cpxt in cpx_types):
                    unique_type = cpx_type_to_docplex_type(first_cpxtype)
                else:
                    cpx_vartypes = [cpx_type_to_docplex_type(cpxt) for cpxt in cpx_types]
            else:
                unique_type = vartype_cont

            cpx_var_lbs = cpx.variables.get_lower_bounds()
            if not any(cpx_var_lbs):
                cpx_var_lbs = []
            else:
                cpx_var_lbs = list(cpx_var_lbs)

            cpx_var_ubs = cpx.variables.get_upper_bounds()
            if all(cpxu >= 1e+20 for cpxu in cpx_var_ubs):
                cpx_var_ubs = []
            else:
                cpx_var_ubs = list(cpx_var_ubs)

            # if no names, None is fine.
            model_varnames = cpx_var_names or None

            model_lbs = cpx_var_lbs
            model_ubs = cpx_var_ubs
            model_types = cpx_vartypes

            # vars
            if unique_type:
                model_vars = lfactory.new_var_list(var_container=None,
                                                   key_seq=range(cpx_nb_vars),
                                                   vartype=unique_type,
                                                   lb=model_lbs,
                                                   ub=model_ubs,
                                                   name=model_varnames,
                                                   _safe_bounds=True,
                                                   _safe_names=True
                                                   )
            else:
                assert model_types
                model_vars = lfactory.new_multitype_var_list(cpx_nb_vars,
                                                             model_types,
                                                             model_lbs,
                                                             model_ubs,
                                                             model_varnames)

            # inverse map from indices to docplex vars
            cpx_var_index_to_docplex = {v: model_vars[v] for v in range(cpx_nb_vars)}

            # 2. upload linear constraints and ranges (mixed in cplex)
            cpx_linearcts = cpx.linear_constraints
            nb_linear_cts = cpx_linearcts.get_num()
            all_rows = cpx_adapter.fast_get_rows(cpx)
            all_rhs = cpx_linearcts.get_rhs()
            all_senses = cpx_linearcts.get_senses()
            if 'R' in all_senses:
                all_range_values = cpx_linearcts.get_range_values()
            else:
                # do not query ranges if no R in senses...
                all_range_values = []
            cpx_ctnames = [] if ignore_names else cls._safe_call_get_names(cpx_adapter,
                                                                           cpx_linearcts.get_names)

            deferred_cts = []

            def make_constant_expr(k):
                return lfactory.constant_expr(k, safe_number=True)

            if verbose:
                print("-- uploading {0} linear constraints...".format(nb_linear_cts))
            for c in range(nb_linear_cts):
                row = all_rows[c]
                sense = all_senses[c]
                rhs = all_rhs[c]
                ctname = cpx_ctnames[c] if cpx_ctnames else None

                indices, coefs = row
                expr = cls._make_expr_from_varmap_coefs(lfactory, cpx_var_index_to_docplex, indices, coefs)

                if sense == 'R':
                    # no use of querying range vars if no R constraints
                    range_val = all_range_values[c]
                    # rangeval can be negative !!! issue 52
                    if range_val >= 0:
                        range_lb = rhs
                        range_ub = rhs + range_val
                    else:
                        range_ub = rhs
                        range_lb = rhs + range_val

                    rgct = mdl.range_constraint(lb=range_lb, ub=range_ub, expr=expr, rng_name=ctname)
                    deferred_cts.append(rgct)
                else:
                    op = cls.parse_sense(sense)
                    rhs_expr = make_constant_expr(rhs)

                    ct = LinearConstraint(mdl, expr, op, rhs_expr, ctname)
                    deferred_cts.append(ct)
            if deferred_cts:
                # add constraint as a block
                lfactory._post_constraint_block(posted_cts=deferred_cts)

            # 3. upload Quadratic constraints
            cpx_quadraticcts = cpx.quadratic_constraints
            nb_quadratic_cts = cpx_quadraticcts.get_num()
            if nb_quadratic_cts:
                all_rhs = cpx_quadraticcts.get_rhs()
                all_linear_nb_non_zeros = cpx_quadraticcts.get_linear_num_nonzeros()
                all_linear_components = cpx_quadraticcts.get_linear_components()
                all_quadratic_nb_non_zeros = cpx_quadraticcts.get_quad_num_nonzeros()
                all_quadratic_components = cpx_quadraticcts.get_quadratic_components()
                all_senses = cpx_quadraticcts.get_senses()
                cpx_ctnames = [] if ignore_names else cls._safe_call_get_names(cpx_adapter,
                                                                               cpx_quadraticcts.get_names)

                for c in range(nb_quadratic_cts):
                    rhs = all_rhs[c]
                    linear_nb_non_zeros = all_linear_nb_non_zeros[c]
                    linear_component = all_linear_components[c]
                    quadratic_nb_non_zeros = all_quadratic_nb_non_zeros[c]
                    quadratic_component = all_quadratic_components[c]
                    sense = all_senses[c]
                    ctname = cpx_ctnames[c] if cpx_ctnames else None

                    if linear_nb_non_zeros > 0:
                        indices, coefs = linear_component.unpack()
                        # linexpr = mdl._aggregator._scal_prod((cpx_var_index_to_docplex[idx] for idx in indices), coefs)
                        linexpr = cls._make_expr_from_varmap_coefs(lfactory, cpx_var_index_to_docplex, indices, coefs)
                    else:
                        linexpr = None

                    if quadratic_nb_non_zeros > 0:
                        qfactory = mdl._qfactory
                        ind1, ind2, coefs = quadratic_component.unpack()
                        quads = qfactory.term_dict_type()
                        for idx1, idx2, coef in izip(ind1, ind2, coefs):
                            quads[VarPair(cpx_var_index_to_docplex[idx1], cpx_var_index_to_docplex[idx2])] = coef

                    else:  # pragma: no cover
                        # should not happen, but who knows
                        quads = None

                    quad_expr = mdl._aggregator._quad_factory.new_quad(quads=quads, linexpr=linexpr, safe=True)
                    op = ComparisonType.cplex_ctsense_to_python_op(sense)
                    ct = op(quad_expr, rhs)
                    mdl.add_constraint(ct, ctname)

            # 4. upload indicators
            cpx_indicators = cpx.indicator_constraints
            nb_indicators = cpx_indicators.get_num()
            if nb_indicators:
                all_ind_names = [] if ignore_names else cls._safe_call_get_names(cpx_adapter,
                                                                                 cpx_indicators.get_names)

                all_ind_bvars = cpx_indicators.get_indicator_variables()
                all_ind_rhs = cpx_indicators.get_rhs()
                all_ind_linearcts = cpx_indicators.get_linear_components()
                all_ind_senses = cpx_indicators.get_senses()
                all_ind_complemented = cpx_indicators.get_complemented()
                all_ind_types = cpx_indicators.get_types()
                ind_equiv_type = 3

                for i in range(nb_indicators):
                    ind_bvar = all_ind_bvars[i]
                    ind_name = all_ind_names[i] if all_ind_names else None
                    ind_rhs = all_ind_rhs[i]
                    ind_linear = all_ind_linearcts[i]  # SparsePair(ind, val)
                    ind_sense = all_ind_senses[i]
                    ind_complemented = all_ind_complemented[i]
                    ind_type = all_ind_types[i]
                    # 1 . check the bvar is ok
                    ind_bvar = cpx_var_index_to_docplex[ind_bvar]
                    # each var appears once
                    ind_linexpr = cls._build_linear_expr_from_sparse_pair(lfactory, cpx_var_index_to_docplex,
                                                                          ind_linear)
                    op = ComparisonType.cplex_ctsense_to_python_op(ind_sense)
                    ind_lct = op(ind_linexpr, ind_rhs)
                    if ind_type == ind_equiv_type:
                        logct = lfactory.new_equivalence_constraint(
                            ind_bvar, ind_lct, true_value=1 - ind_complemented, name=ind_name)
                    else:
                        logct = lfactory.new_indicator_constraint(
                            ind_bvar, ind_lct, true_value=1 - ind_complemented, name=ind_name)
                    mdl.add(logct)

            # 5. upload Piecewise linear constraints
            try:
                cpx_pwl = cpx.pwl_constraints
                cpx_pwl_defs = cpx_pwl.get_definitions()
                pwl_fallback_names = [""] * cpx_pwl.get_num()
                cpx_pwl_names = pwl_fallback_names if ignore_names else cls._safe_call_get_names(cpx_adapter,
                                                                                                 cpx_pwl.get_names,
                                                                                                 pwl_fallback_names)
                for (vary_idx, varx_idx, preslope, postslope, breakx, breaky), pwl_name in izip(cpx_pwl_defs,
                                                                                                cpx_pwl_names):
                    varx = cpx_var_index_to_docplex.get(varx_idx, None)
                    vary = cpx_var_index_to_docplex.get(vary_idx, None)
                    breakxy = [(brkx, brky) for brkx, brky in zip(breakx, breaky)]
                    pwl_func = mdl.piecewise(preslope, breakxy, postslope, name=pwl_name)
                    pwl_expr = mdl._lfactory.new_pwl_expr(pwl_func, varx, 0, add_counter_suffix=False, resolve=False)
                    pwl_expr._f_var = vary
                    pwl_expr._ensure_resolved()

            except AttributeError:  # pragma: no cover
                pass  # Do not check for PWLs if Cplex version does not support them

            # 6. upload objective

            # noinspection PyPep8
            try:
                cpx_multiobj = cpx.multiobj
            except AttributeError:  # pragma: no cover
                # pre-12.9 version
                cpx_multiobj = None

            if cpx_multiobj is None or cpx_multiobj.get_num() <= 1:
                cpx_obj = cpx.objective
                cpx_sense = cpx_obj.get_sense()

                cpx_all_lin_obj_coeffs = cpx_obj.get_linear()
                obj_expr = compute_full_dot(mdl, cpx_all_lin_obj_coeffs)

                if cpx_obj.get_num_quadratic_variables() > 0:
                    cpx_all_quad_cols_coeffs = cpx_obj.get_quadratic()
                    quads = qfactory.term_dict_type()
                    for v, col_coefs in izip(cpx_var_index_to_docplex, cpx_all_quad_cols_coeffs):
                        var1 = cpx_var_index_to_docplex[v]
                        indices, coefs = col_coefs.unpack()
                        for idx, coef in izip(indices, coefs):
                            vp = VarPair(var1, cpx_var_index_to_docplex[idx])
                            quads[vp] = quads.get(vp, 0) + coef / 2

                    obj_expr += qfactory.new_quad(quads=quads, linexpr=None)

                obj_expr += cpx.objective.get_offset()
                is_maximize = cpx_sense == cpx_adapter.cplex_module._internal._subinterfaces.ObjSense.maximize

                if is_maximize:
                    mdl.maximize(obj_expr)
                else:
                    mdl.minimize(obj_expr)
            else:
                # we have multiple objective
                nb_multiobjs = cpx_multiobj.get_num()
                exprs = [0] * nb_multiobjs
                priorities = [1] * nb_multiobjs
                weights = [1] * nb_multiobjs
                abstols = [0] * nb_multiobjs
                reltols = [0] * nb_multiobjs
                if ignore_names:
                    names = ["Goal_{0}".format(g) for g in range(1, nb_multiobjs+1)]
                else:
                    names = cpx_multiobj.get_names()

                for m in range(nb_multiobjs):
                    (obj_coeffs, obj_offset, weight, prio, abstol, reltol) = cpx_multiobj.get_definition(m)
                    obj_expr = compute_full_dot(mdl, obj_coeffs, obj_offset)
                    exprs[m] = obj_expr
                    priorities[m] = prio
                    weights[m] = weight
                    abstols[m] = abstol
                    reltols[m] = reltol
                sense = cpx_multiobj.get_sense()
                mdl.set_multi_objective(sense, exprs, priorities, weights, abstols, reltols, names)

            # upload sos
            cpx_sos = cpx.SOS
            cpx_sos_num = cpx_sos.get_num()
            if cpx_sos_num > 0:
                cpx_sos_types = cpx_sos.get_types()
                cpx_sos_indices = cpx_sos.get_sets()
                cpx_sos_names = cpx_sos.get_names()
                if not cpx_sos_names:
                    cpx_sos_names = [None] * cpx_sos_num
                for sostype, sos_sparse, sos_name in izip(cpx_sos_types, cpx_sos_indices, cpx_sos_names):
                    sos_var_indices = sos_sparse.ind
                    sos_weights = sos_sparse.val
                    isostype = int(sostype)
                    sos_vars = [cpx_var_index_to_docplex[var_ix] for var_ix in sos_var_indices]
                    mdl.add_sos(dvars=sos_vars, sos_arg=isostype, name=sos_name, weights=sos_weights)

            # upload lazy constraints
            cpx_linear_advanced = cpx.linear_constraints.advanced
            cpx_lazyct_num = cpx_linear_advanced.get_num_lazy_constraints()
            if cpx_lazyct_num:
                print("WARNING: found {0} lazy constraints that cannot be uploaded to DOcplex".format(cpx_lazyct_num))

            mdl.output_level = final_output_level
            if final_checker:
                # need to restore checker
                mdl.set_checker(final_checker)

        except cpx_adapter.CplexError as cpx_e:  # pragma: no cover
            print("* CPLEX error: {0!s} reading file {1}".format(cpx_e, filename))
            mdl = None
            if debug_read:
                raise

        except ModelReaderError as mre:  # pragma: no cover
            print("! Model reader error: {0!s} while reading file {1}".format(mre, filename))
            mdl = None
            if debug_read:
                raise

        except DOcplexException as doe:  # pragma: no cover
            print("! Internal DOcplex error: {0!s} while reading file {1}".format(doe, filename))
            mdl = None
            if debug_read:
                raise

        # except Exception as any_e:  # pragma: no cover
        #     print("Internal exception raised: {0} msg={1!s} while reading file '{2}'".format(type(any_e), any_e, filename))
        #     mdl = None
        #     if debug_read:
        #         raise

        finally:
            # clean up CPLEX instance...
            cpx.end()

        return mdl

    @classmethod
    def read_model(cls, filename, model_name=None, verbose=False, model_class=None, **kwargs):
        """ This method is a synonym of `read` for compatibility.

        """
        import warnings
        warnings.warn("ModelReader.read_model is deprecated, use class method ModelReader.read()", DeprecationWarning)
        return cls.read(filename, model_name, verbose, model_class, **kwargs)


def read_model(filename, model_name=None, verbose=False, **kwargs):
    return ModelReader.read(filename, model_name=model_name, verbose=verbose, **kwargs)