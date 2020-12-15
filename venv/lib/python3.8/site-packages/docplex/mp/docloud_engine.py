# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from io import BytesIO

import os

from six import iteritems
import tempfile

import warnings

from docplex.mp.engine import IndexerEngine

try:
    # import DOcloudConnector only if JobClient is available
    from docloud.job import JobClient
    from docplex.mp.docloud_connector import DOcloudConnector
except ImportError as ie:
    # just ignore if docloud is not available
    pass

from docplex.mp.internal.json_solution_handler import JSONSolutionHandler
from docplex.mp.internal.json_infeasibility_handler import JSONInfeasibilityHandler
from docplex.mp.internal.json_conflict_handler import JSONConflictHandler
from docplex.mp.lp_printer import LPModelPrinter
from docplex.mp.solution import SolveSolution, SolutionMSTPrinter
from docplex.mp.anno import ModelAnnotationPrinter
from docplex.mp.sdetails import SolveDetails
from docplex.mp.utils import DOcplexException, make_path
from docplex.mp.utils import normalize_basename
from docplex.mp.constants import ConflictStatus
from docplex.mp.conflict_refiner import TConflictConstraint, VarLbConstraintWrapper, VarUbConstraintWrapper, ConflictRefinerResult
from docplex.mp.constr import LinearConstraint, QuadraticConstraint, IndicatorConstraint

from docplex.util.environment import make_attachment_name

from docplex.mp.format import LP_format
from docplex.mp.compat23 import StringIO

from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom

from collections import defaultdict
import numbers
import sys

# gendoc: ignore

# this is the default exchange format for docloud
_DEFAULT_EXCHANGE_FORMAT = LP_format

# default attachment name to trigger conflict refiner
feasibility_name = "file.feasibility"


class FeasibilityPrinter(object):
    extension = ".feasibility"

    @classmethod
    def print_to_stream(cls, relaxables, out, extension=extension):
        if out is None:
            # prints on standard output
            cls.print_internal(sys.stdout, relaxables)

        elif isinstance(out, str):
            # a string is interpreted as a path name
            path = out if out.endswith(extension) else out + extension
            with open(path, "w") as of:
                cls.print_to_stream(relaxables, of)
        else:
            try:
                cls.print_internal(out, relaxables)

            except AttributeError:  # pragma: no cover
                pass  # pragma: no cover
                # stringio will raise an attribute error here, due to with
                # print("Cannot use this an output: %s" % str(out))

    @classmethod
    def print_internal(cls, out, relaxables):
        # INTERNAL: out is resolved here to a writeable stream...
        # the infeasibilityFile flag is here to trigger the output of infeasibilities
        # -> not the default.
        out.write("<CPLEXFeasopt infeasibilityFile='true'>\n")
        out.write("  <rhs>\n")
        for relaxable_group in relaxables:
            pref, cts = relaxable_group
            for ct in cts:
                out.write("    <relax index=\"{0}\" preference=\"{1}\"/>".format(ct.index, pref))
                ctname = ct.name
                if ctname:
                    out.write("  <!-- {0} -->".format(ctname))
                out.write("\n")

        out.write("  </rhs>\n")
        out.write("</CPLEXFeasopt>\n")


# constraint_type_by_ct_type = {'lb': VarLbConstraintWrapper,
#                               'ub': VarUbConstraintWrapper,
#                               'lin': LinearConstraint,
#                               'quad': QuadraticConstraint,
#                               'ind': IndicatorConstraint,
#                               'sos': None}


ct_type_by_constraint_type = {VarLbConstraintWrapper: 'lb',
                              VarUbConstraintWrapper: 'ub',
                              LinearConstraint: 'lin',
                              QuadraticConstraint: 'quad',
                              IndicatorConstraint: 'ind'}


class ConflictRefinerPrinter(object):
    extension = ".feasibility"

    @classmethod
    def print_to_stream(cls, artifact_as_xml, out, extension=extension):
        if out is None:
            # prints on standard output
            cls.print_internal(sys.stdout, artifact_as_xml)

        elif isinstance(out, str):
            # a string is interpreted as a path name
            path = out if out.endswith(extension) else out + extension
            with open(path, "w") as of:
                cls.print_to_stream(artifact_as_xml, of)
        else:
            try:
                cls.print_internal(out, artifact_as_xml)

            except AttributeError:  # pragma: no cover
                pass  # pragma: no cover
                # stringio will raise an attribute error here, due to with
                # print("Cannot use this an output: %s" % str(out))

    @classmethod
    def print_internal(cls, out, artifact_as_xml):
        ElementTree.ElementTree(artifact_as_xml.get_root()).write(out, 'utf-8')


class CPLEXRefineConflictExtArtifact(object):
    '''
    XML builder for DOcplexcloud artifact to trigger Conflict Refiner execution
    '''

    def __init__(self):
        self._root = Element('CPLEXRefineconflictext')
        self._grps_index = {}
        self._grps_dict = defaultdict(list)

    def get_root(self):
        return self._root

    def add_constraints(self, ct_type, elements, preference=1.0, preference_dict=None):
        for (elem, name) in elements:
            pref = preference
            if preference_dict is not None:
                new_pref = preference_dict.get(elem, None)
                if new_pref is not None and isinstance(new_pref, numbers.Number):
                    pref = new_pref
            group = self.add_group(pref)
            con = SubElement(group, "con")
            con.set("name", name)
            con.set("type", ct_type)
            self._grps_dict[self._grps_index[group]].append(elem)

    def add_group(self, preference):
        group = SubElement(self._root, "group")
        group.set("preference", repr(preference))
        self._grps_index[group] = len(self._grps_index)
        return group

    def add_constraint_to_group(self, group, ct):
        con = SubElement(group, "con")
        con.set("name", ct.name)
        constraint_type = type(ct)
        con.set("type", ct_type_by_constraint_type.get(constraint_type, ""))
        self._grps_dict[self._grps_index[group]].append(ct)

    def tostring(self, encoding='utf-8'):
        """Return a string for the XML Element hierarchy.
        """
        return ElementTree.tostring(self._root, encoding)

    def pretty_print(self):
        """Return a pretty-printed XML string for the Element.
        """
        as_string = ElementTree.tostring(self._root, 'utf-8')
        dom = minidom.parseString(as_string)
        return dom.toprettyxml(indent="  ")


# noinspection PyProtectedMember
class DOcloudEngine(IndexerEngine):
    """ Engine facade stub to defer solve to drop-solve URL
    """

    def _print_feasibility(self, out, relaxables):
        pass

    def get_cplex(self):
        raise DOcplexException("{0} engine contains no instance of CPLEX".format(self.name))

    def __init__(self, mdl, exchange_format=None, **kwargs):
        IndexerEngine.__init__(self)

        warnings.warn(
            'Solve using \'docloud\' agent is deprecated. Consider submitting your model to DOcplexcloud. See https://ibm.biz/BdYhhK',
            DeprecationWarning)

        docloud_context = kwargs.get('docloud_context')
        # --- log output can be overridden at solve time, so use te one from the context, not the model's
        actual_log_output = kwargs.get('log_output') or mdl.log_output

        self._model = mdl
        self._connector = DOcloudConnector(docloud_context, log_output=actual_log_output)
        self._exchange_format = exchange_format or docloud_context.exchange_format or _DEFAULT_EXCHANGE_FORMAT

        mangle_names = mdl.ignore_names or mdl.context.solver.docloud.mangle_names
        self._printer = LPModelPrinter(full_obj=True)
        if mangle_names:
            self._printer.set_mangle_names(True)

        # -- results.
        self._lpname_to_var_map = {}
        self._solve_details = SolveDetails.make_dummy()
        self._quality_metrics = {}  # latest quality metrics from json

        # noinspection PyPep8
        self.debug_dump = docloud_context.debug_dump
        self.debug_dump_dir = docloud_context.debug_dump_dir

    def _new_printer(self, ctx):
        return self._printer

    def supports_logical_constraints(self):
        # <-> is supposed to be supported in LP?
        return True, None

    docloud_solver_name = 'cplex_cloud'

    @property
    def name(self):
        return self.docloud_solver_name

    def can_solve(self):
        """
        :return: true, as this solver can solve!
        """
        return True

    def connect_progress_listeners(self, progress_listener_list, model):
        if progress_listener_list:
            self._model.warning("Progress listeners are not supported on DOcplexcloud.")

    def register_callback(self, cb):
        self._model.fatal('Callbacks are not available on DOcplexcloud')

    @staticmethod
    def _docloud_cplex_version():
        # INTERNAL: returns the version of CPLEX used in DOcplexcloud
        # for now returns a string. maybe we could ping Docloud and get a dynamic answer.
        return "12.6.3.0"

    def _serialize_parameters(self, parameters, write_level=None, relax_mode=None):
        # return a string in PRM format
        # overloaded params are:
        # - THREADS = 1, if deterministic, else not mentioned.
        # the resulting string will contain all non-default parameters,
        # AND those overloaded.
        # No side effect on actual model parameters
        overloaded_params = dict()
        if relax_mode is not None:
            overloaded_params[parameters.feasopt.mode] = relax_mode.value

        # Do not override write level anymore
        if write_level is not None:
            overloaded_params[parameters.output.writelevel] = write_level

        if self._connector.run_deterministic:
            # overloaded_params[mdl_parameters.threads] = 1 cf RTC28458
            overloaded_params[parameters.parallel] = 1  # 1 is deterministic

        # do we need to limit the version to the one use din docloud
        # i.e. if someone has a *newer* version than docloud??
        prm_data = parameters.export_prm_to_string(overloaded_params)
        return prm_data

    def serialize_model_as_file(self, mdl):
        # step 1 : prints the model in whatever exchange format
        printer = self._new_printer(ctx=mdl.context)

        if self._exchange_format.is_binary:
            filemode = "w+b"
        else:
            filemode = "w+"

        lp_output = tempfile.NamedTemporaryFile(mode=filemode, delete=False)

        printer.printModel(mdl, lp_output)

        # lp name to docplex var
        self._lpname_to_var_map = printer.get_name_to_var_map(mdl)

        # DEBUG: dump request file
        if self.debug_dump:
            dump_path = make_path(error_handler=mdl.error_handler,
                                  basename=mdl.name,
                                  output_dir=self.debug_dump_dir,
                                  extension=printer.extension(),
                                  name_transformer="docloud_%s")
            print("DEBUG DUMP in " + dump_path)
            with open(dump_path, filemode) as out_file:
                lp_output.seek(0)
                out_file.write(lp_output)

        lp_output.close()
        return lp_output.name

    def serialize_model(self, mdl):
        # step 1 : prints the model in whatever exchange format
        printer = self._new_printer(ctx=mdl.context)

        if self._exchange_format.is_binary:
            filemode = "wb"
            oss = BytesIO()
        else:
            filemode = "w"
            oss = StringIO()

        printer.printModel(mdl, oss)

        # lp name to docplex var
        self._lpname_to_var_map = printer.get_name_to_var_map(mdl)

        # DEBUG: dump request file
        if self.debug_dump:
            dump_path = make_path(error_handler=mdl.error_handler,
                                  basename=mdl.name,
                                  output_dir=self.debug_dump_dir,
                                  extension=printer.extension(),
                                  name_transformer="docloud_%s")
            print("DEBUG DUMP in " + dump_path)
            with open(dump_path, filemode) as out_file:
                out_file.write(oss.getvalue())

        if self._exchange_format.is_binary:
            model_data = oss.getvalue()
        else:
            model_data = oss.getvalue().encode('utf-8')
        return model_data

    def _serialize_relaxables(self, relaxables):
        oss = StringIO()
        FeasibilityPrinter.print_to_stream(out=oss, relaxables=relaxables)
        serialized_relaxables = oss.getvalue()
        return serialized_relaxables

    def _dump_if_required(self, data, mdl, basename, extension, is_binary=False, forced=False):
        # INTERNAL
        if self.debug_dump or forced:
            relax_path = make_path(error_handler=mdl.error_handler,
                                   basename=basename,
                                   output_dir=self.debug_dump_dir,
                                   extension=extension,
                                   name_transformer=None)
            if isinstance(data, bytes) or isinstance(data, bytearray):
                is_binary = True  # For binary when data is binary
            fmode = "wb" if is_binary else "w"
            with open(relax_path, fmode) as out_file:
                out_file.write(data)

    def _make_attachment(self, attachment_name, attachment_data):
        # INTERNAL
        return {'name': attachment_name, 'data': attachment_data}

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        # --- 1 serialize
        job_name = normalize_basename(mdl.name, force_lowercase=True)
        model_data = self.serialize_model(mdl)
        docloud_parameters = mdl.parameters
        prm_data = self._serialize_parameters(docloud_parameters, write_level=1, relax_mode=relax_mode)
        prm_name = self._make_attachment_name(job_name, '.prm')
        feasopt_data = self._serialize_relaxables(relaxable_groups)

        # --- dump if need be
        if prio_name:
            prio_name = "_%s" % prio_name
        relax_basename = normalize_basename("%s_feasopt%s" % (mdl.name, prio_name))

        prm_basename = normalize_basename("%s_feasopt" % mdl.name)
        self._dump_if_required(model_data, mdl, basename=job_name, extension=".lp", forced=True)
        self._dump_if_required(feasopt_data, mdl, basename=relax_basename, extension=FeasibilityPrinter.extension,
                               forced=True)
        self._dump_if_required(prm_data, mdl, basename=prm_basename, extension=".prm", forced=True)

        # --- submit job somehow...
        attachments = []
        model_name = normalize_basename(job_name) + self._exchange_format.extension
        attachments.append(self._make_attachment(model_name, model_data))

        attachments.append(self._make_attachment(prm_name, prm_data))
        attachments.append(self._make_attachment(normalize_basename(job_name) + FeasibilityPrinter.extension,
                                                 feasopt_data))

        # here we go...
        def notify_info(info):
            if "jobid" in info:
                mdl.fire_jobid(jobid=info["jobid"])
            if "progress" in info:
                mdl.fire_progress(progress_data=info["progress"])

        connector = self._connector
        mdl.notify_start_solve()
        connector.submit_model_data(attachments,
                                    gzip=not self._exchange_format.is_binary,
                                    info_callback=notify_info,
                                    info_to_monitor={'jobid', 'progress'})

        # --- cplex solve details
        json_details = connector.get_cplex_details()
        self._solve_details = SolveDetails.from_json(json_details)
        # ---

        # --- build a solution object, or None
        solution_handler = JSONSolutionHandler(connector.results.get('solution.json'))
        if not solution_handler.has_solution:
            mdl.notify_solve_failed()
            return None
        else:
            infeas_json = connector.results.get('infeasibilities.json')
            infeas_handler = JSONInfeasibilityHandler(infeas_json) if infeas_json else None
            sol = self._make_relaxed_solution(mdl, solution_handler, infeas_handler)
            return sol

    def _serialize_conflict_refiner(self, artifact_as_xml):
        if sys.version_info[0] < 3:
            oss = StringIO()
        else:
            oss = BytesIO()
        ConflictRefinerPrinter.print_to_stream(artifact_as_xml, out=oss)
        serialized_artifact = oss.getvalue()
        return serialized_artifact

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        """ Starts conflict refiner on the model.

        Args:
            mdl: The model for which conflict refinement is performed.
            preferences: a dictionary defining constraints preferences.
            groups: a list of ConstraintsGroup.
            :parameters: cplex parameters .

        Returns:
            A list of "TConflictConstraint" namedtuples, each tuple corresponding to a constraint that is
            involved in the conflict.
            The fields of the "TConflictConstraint" namedtuple are:
                - the name of the constraint or None if the constraint corresponds to a variable lower or upper bound
                - a reference to the constraint or to a wrapper representing a Var upper or lower bound
                - an :enum:'docplex.mp.constants.ConflictStatus' object that indicates the
                conflict status type (Excluded, Possible_member, Member...)
            This list is empty if no conflict is found by the conflict refiner.
        """
        # Before submitting the job, we will build the list of attachments
        attachments = []

        # make sure model is the first attachment: that will be the name of the job on the console
        job_name = "python_%s" % self._model.name
        model_data = self.serialize_model(self._model)
        mprinter = self._new_printer(ctx=mdl.context)
        model_name = normalize_basename(job_name) + self._exchange_format.extension
        attachments.append({'name': model_name, 'data': model_data})

        # Conflict Refiner Ext
        artifact_as_xml = CPLEXRefineConflictExtArtifact()
        if groups is None or groups == []:
            # Add all constraints
            artifact_as_xml.add_constraints(ct_type_by_constraint_type[VarUbConstraintWrapper],
                                            [(VarUbConstraintWrapper(v), mprinter._var_print_name(v))
                                             for v in self._model.iter_variables()],
                                            preference=1.0)
            artifact_as_xml.add_constraints(ct_type_by_constraint_type[VarLbConstraintWrapper],
                                            [(VarLbConstraintWrapper(v), mprinter._var_print_name(v))
                                             for v in self._model.iter_variables()],
                                            preference=1.0)
            artifact_as_xml.add_constraints(ct_type_by_constraint_type[LinearConstraint],
                                            [(c, mprinter.linearct_print_name(c))
                                             for c in self._model.iter_linear_constraints()],
                                            preference_dict=preferences)
            artifact_as_xml.add_constraints(ct_type_by_constraint_type[QuadraticConstraint],
                                            [(c, mprinter.qc_print_name(c))
                                             for c in self._model.iter_quadratic_constraints()],
                                            preference_dict=preferences)
            artifact_as_xml.add_constraints(ct_type_by_constraint_type[IndicatorConstraint],
                                            [(c, mprinter.logicalct_print_name(c))
                                             for c in self._model.iter_indicator_constraints()],
                                            preference_dict=preferences)
        else:
            for grp in groups:
                group = artifact_as_xml.add_group(grp.preference)
                for ct in grp._cts:
                    artifact_as_xml.add_constraint_to_group(group, ct)

        conflict_refiner_data = self._serialize_conflict_refiner(artifact_as_xml)

        attachments.append({'name': feasibility_name, 'data': conflict_refiner_data})

        def notify_info(info):
            if "jobid" in info:
                self._model.fire_jobid(jobid=info["jobid"])
            if "progress" in info:
                self._model.fire_progress(progress_data=info["progress"])

        # This block used to be try/catched for DOcloudConnector exceptions
        # and DOcloudException, but then infrastructure error were not
        # handled properly. Now we let the exception raise.
        connector = self._connector
        self._model.notify_start_solve()
        connector.submit_model_data(attachments,
                                    gzip=not self._exchange_format.is_binary,
                                    info_callback=notify_info,
                                    info_to_monitor={'jobid', 'progress'})

        # --- build a conflict object, or None
        conflicts_handler = JSONConflictHandler(connector.results.get('conflict.json'), artifact_as_xml._grps_dict)
        if not conflicts_handler.has_conflict:
            return []
        else:
            return self._get_conflicts_cloud(conflicts_handler)
        # ---

    def _get_conflicts_cloud(self, conflicts_handler):
        conflict_grps = conflicts_handler.get_conflict_grps_list()
        conflicts = []

        for index, elems, c_status in conflict_grps:
            if c_status is None:
                self._model.error("Undefined status for constraint conflict for group index: {0}".format(index))
                continue
            if c_status == ConflictStatus.Excluded:
                continue

            for elem in elems:
                if isinstance(elem, VarLbConstraintWrapper):
                    conflicts.append(TConflictConstraint(None, elem, c_status))

                elif isinstance(elem, VarUbConstraintWrapper):
                    conflicts.append(TConflictConstraint(None, elem, c_status))

                else:
                    conflicts.append(TConflictConstraint(elem.name, elem, c_status))

        return ConflictRefinerResult(conflicts, refined_by=self.name)

    def _make_attachment_name(self, basename, extension):
        return make_attachment_name(basename + extension)

    def export_one_mip_start(self, mipstart, job_name, attachments):
        warmstart_data = SolutionMSTPrinter.print_to_string(mipstart).encode('utf-8')
        mipstart_name = mipstart.name.lower() if mipstart.name else job_name
        warmstart_name = self._make_attachment_name(mipstart_name, ".mst")
        attachments.append({'name': warmstart_name, 'data': warmstart_data})

    # noinspection PyProtectedMember
    def solve(self, mdl, parameters=None, **kwargs):
        # Before submitting the job, we will build the list of attachments
        # parameters are CPLEX parameters
        lex_mipstart = kwargs.pop('_lex_mipstart', None)
        attachments = []

        # make sure model is the first attachment: that will be the name of the job on the console
        job_name = normalize_basename("python_%s" % mdl.name)
        model_file = self.serialize_model_as_file(mdl)
        try:
            model_data_name = self._make_attachment_name(job_name, self._exchange_format.extension)
            attachments.append({'name': model_data_name, 'filename': model_file})

            # prm
            docloud_parameters = parameters if parameters is not None else mdl.parameters
            prm_data = self._serialize_parameters(docloud_parameters)
            prm_name = self._make_attachment_name(job_name, '.prm')
            attachments.append({'name': prm_name, 'data': prm_data})

            # warmstart_data
            # export mipstart solution in CPLEX mst format, if any, else None
            # if within a lexicographic solve, th elex_mipstart supersedes allother mipstarts
            if lex_mipstart:
                mipstart_name = lex_mipstart.name.lower() if lex_mipstart.name else job_name
                warmstart_data = SolutionMSTPrinter.print_to_string(lex_mipstart).encode('utf-8')
                warmstart_name = self._make_attachment_name(mipstart_name, ".mst")
                attachments.append({'name': warmstart_name, 'data': warmstart_data})

            elif mdl.number_of_mip_starts:
                mipstart_name = job_name
                warmstart_name = self._make_attachment_name(mipstart_name, ".mst")
                mdl_mipstarts = [s for s, _ in mdl.iter_mip_starts()]
                mdl_efforts = [eff for (_, eff) in mdl.iter_mip_starts()]
                warmstart_data = SolutionMSTPrinter.print_to_string(mdl_mipstarts, effort_level=mdl_efforts,
                                                                    use_lp_names=True).encode('utf-8')
                attachments.append({'name': warmstart_name, 'data': warmstart_data})

            # benders annotation
            if mdl.has_benders_annotations():
                anno_data = ModelAnnotationPrinter.print_to_string(mdl).encode('utf-8')
                anno_name = self._make_attachment_name(job_name, '.ann')
                attachments.append({'name': anno_name, 'data': anno_data})

            # info_to_monitor = {'jobid'}
            # if mdl.progress_listeners:
            # info_to_monitor.add('progress')

            def notify_info(info):
                if "jobid" in info:
                    mdl.fire_jobid(jobid=info["jobid"])
                if "progress" in info:
                    mdl.fire_progress(progress_data=info["progress"])

            # This block used to be try/catched for DOcloudConnector exceptions
            # and DOcloudException, but then infrastructure error were not
            # handled properly. Now we let the exception raise.
            connector = self._connector
            mdl.notify_start_solve()
            connector.submit_model_data(attachments,
                                        gzip=not self._exchange_format.is_binary,
                                        info_callback=notify_info,
                                        info_to_monitor={'jobid', 'progress'})

            # --- cplex solve details
            json_details = connector.get_cplex_details()
            self._solve_details = SolveDetails.from_json(json_details)
            self._solve_details._quality_metrics = self._compute_quality_metrics(json_details)
            # ---

            # --- build a solution object, or None
            solution_handler = JSONSolutionHandler(connector.results.get('solution.json'))
            if not solution_handler.has_solution:
                mdl.notify_solve_failed()
                solution = None
            else:
                solution = self._make_solution(mdl, solution_handler)
            # ---

            return solution
        finally:
            if os.path.isfile(model_file):
                os.remove(model_file)

    def _var_by_cloud_index(self, cloud_index, cloud_index_name_map):
        # index -> cloud_name (lp_name) -> var object
        cloud_name = cloud_index_name_map.get(cloud_index)
        return self._lpname_to_var_map.get(cloud_name) if cloud_name else None

    def _make_solution(self, mdl, solution_handler):
        solver_name = self.docloud_solver_name
        # Store the results of solve in a solution object.
        docloud_obj = solution_handler.get_objective()
        docloud_values_by_idx, docloud_var_rcs = solution_handler.variable_results()
        # CPLEX index to name map
        # for those variables returned by CPLEX.
        # all other are assumed to be zero
        cloud_index_name_map = solution_handler.cplex_index_name_map()
        var_mapper = lambda idx: self._var_by_cloud_index(idx, cloud_index_name_map)
        # send an objective, a var-value dict and a string identifying the engine which solved.
        docloud_values_by_vars = {}
        keep_zeros = False
        count_nonmatching_cloud_vars = 0
        for cpx_idx, val in iteritems(docloud_values_by_idx):
            if keep_zeros or val:
                # first get the name from the cloud idx
                dvar = var_mapper(cpx_idx)
                if dvar:
                    docloud_values_by_vars[dvar] = val
                else:
                    cloud_name = cloud_index_name_map.get(cpx_idx)
                    if cloud_name and cloud_name.startswith("Rgc"):
                        # range variables
                        pass
                    else:
                        # one extra variable from docloud is OK
                        # it represents the constant term in objective
                        # more than one is an issue.
                        if count_nonmatching_cloud_vars:
                            mdl.info("Cannot find matching variable, cloud name is {0!s}", cloud_name)
                        count_nonmatching_cloud_vars += 1

        sol = SolveSolution.make_engine_solution(model=mdl,
                                                 obj=docloud_obj,
                                                 blended_obj_by_priority=[docloud_obj],
                                                 var_value_map=docloud_values_by_vars,
                                                 solved_by=solver_name,
                                                 solve_details=self._solve_details,
                                                 job_solve_status=self.get_solve_status())

        # attributes
        docloud_ct_duals, docloud_ct_slacks = solution_handler.constraint_results()

        ct_mapper = lambda idx: mdl.get_constraint_by_index(idx)
        sol.store_reduced_costs(docloud_var_rcs, mapper=var_mapper)
        sol.store_dual_values(docloud_ct_duals, mapper=ct_mapper)
        sol.store_slack_values(docloud_ct_slacks, mapper=ct_mapper)
        return sol

    def _make_relaxed_solution(self, mdl, solution_handler, infeas_handler):
        if infeas_handler is not None:
            infeasibilities = self._read_infeasibilities(mdl, infeas_handler)
        else:
            infeasibilities = self._compute_infeasibilities_from_slacks(mdl, solution_handler)

        sol = self._make_solution(mdl, solution_handler)
        sol.store_infeasibilities(infeasibilities)
        return sol

    def _read_infeasibilities(self, mdl, infeas_handler):
        # INTERNAL
        raw_infeasibilities = infeas_handler.get_infeasibilities()
        assert len(raw_infeasibilities) > 0
        infeasibilities = {}
        for ct_index, raw_infeas in iteritems(raw_infeasibilities):
            ct = mdl.get_constraint_by_index(ct_index)  # no exception: returns None if not found.
            if ct is not None:
                if raw_infeas != 0:
                    infeasibilities[ct] = raw_infeas
            else:
                self._unexpected_cloud_constraint_index(mdl, ct_index)
        return infeasibilities

    def _compute_infeasibilities_from_slacks(self, mdl, solution_handler):
        # INTERNAL - temporary
        raw_slacks = solution_handler.constraint_slacks()
        # from slack to infeasibility
        infeasibilities = {}
        for ct_index, raw_slack in iteritems(raw_slacks):
            ct = mdl.get_constraint_by_index(ct_index)  # no exception: returns None if not found.
            if ct is not None:
                infeas = ct.compute_infeasibility(raw_slack)
                if 0 != infeas:
                    infeasibilities[ct] = infeas
            else:
                # no constraint with this index, should not happen...
                self._unexpected_cloud_constraint_index(mdl, ct_index)

        return infeasibilities

    def _unexpected_cloud_constraint_index(self, mdl, ct_index):
        mdl.warning('unexpected index from cplex cloud: {}', ct_index)

    def get_solve_attribute(self, attr, index_seq):
        return {}

    def get_all_reduced_costs(self, mdl):
        return {}

    def get_solve_status(self):
        return self._connector.get_solve_status()

    def get_solve_details(self):
        return self._solve_details

    @classmethod
    def demangle_metric_name(cls, mname):
        from docplex.mp.constants import QualityMetric
        # find last occurence of '.' then identify the enum from the last part
        dotpos = mname.rfind('.')
        assert dotpos >= 0
        return QualityMetric.parse(mname[dotpos + 1:], raise_on_error=False)

    def _compute_quality_metrics(self, json_details):
        qms = {}
        if json_details:
            for qk, qv in iteritems(json_details):
                if 'quality.double' in qk:
                    qm = self.demangle_metric_name(qk)
                    if qm:
                        qms[qm.key] = float(qv)
                elif 'quality.int' in qk:
                    qm = self.demangle_metric_name(qk)
                    if qm:
                        iqv = int(qv)
                        if iqv >= 0:
                            qms[qm.int_key] = iqv
        return qms
