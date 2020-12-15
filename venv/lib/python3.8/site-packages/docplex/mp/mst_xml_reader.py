# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2019
# --------------------------------------------------------------------------

# gendoc: ignore

# Reference:
# https://www.ibm.com/support/knowledgecenter/SSSA5P_12.9.0/ilog.odms.cplex.help/CPLEX/FileFormats/topics/MST.html

# noinspection PyPep8Naming
import xml.etree.ElementTree as xml_elt


from docplex.mp.constants import EffortLevel


class MSTReader(object):
    cplex_solution_tag  = "CPLEXSolution"
    cplex_solutions_tag = "CPLEXSolutions"

    @classmethod
    def to_float(cls, s):
        try:
            return float(s)
        except TypeError:
            return None

    @classmethod
    def to_int(cls, s, fallback=None):
        try:
            return int(s)
        except TypeError:
            return fallback

    @classmethod
    def new_empty_mipstart(cls, mdl, mipstart_name=None):
        return mdl.new_solution(name=mipstart_name)

    @classmethod
    def read_one_solution(cls, xml_solution, mdl, mst_path):
        new_mipstart_sol = cls.new_empty_mipstart(mdl)
        nb_missed = 0
        effort_level = EffortLevel.Auto
        for child in xml_solution:
            if child.tag == "header":
                obj_s = child.attrib.get('objectiveValue')
                if obj_s is not None:
                    new_mipstart_sol.set_objective_value(float(obj_s))
                # an optional mip start name
                sol_name = child.attrib.get('solutionName')
                if sol_name:
                    new_mipstart_sol.set_name(sol_name)
                # effort level: 0,1,2,...
                effort_s = child.attrib.get('MIPStartEffortLevel')
                if effort_s is not None:
                    # expect an integer attribute, then convert it to EffortLevel
                    # default is Auto
                    effort_level = EffortLevel.parse(cls.to_int(effort_s))

            elif child.tag == "variables":
                for v, var_elt in enumerate(child, start=1):
                    var_attribs = var_elt.attrib
                    var_name = var_attribs.get('name')
                    var_value_s = var_attribs.get('value')

                    # check the <variable> taag
                    valid = True

                    # we need a value
                    var_value = cls.to_float(var_value_s)
                    if var_value is None:
                        mdl.warning("Variable element has no float value, {0} was found - pos={1}", var_value_s, v)
                        valid = False

                    var_index = cls.to_int(var_attribs.get('index'))
                    if var_index is None:
                        mdl.warning("Variable element has no index tag - pos={0}", v)
                        valid = False
                    elif var_index < 0:
                        mdl.warning("Variable element has invalid index: {1} - pos={0}", v, var_index)
                        valid = False
                    if not valid:
                        nb_missed += 1
                    else:
                        # look for a variable from name, index
                        dv = (var_name and mdl.get_var_by_name(var_name)) or mdl.get_var_by_index(var_index)

                        if dv is None:
                            mdl.warning('Cannot find matching variable: name={0}, index={1} - value={2}',
                                        var_name, var_index, var_value)
                            nb_missed += 1
                        else:
                            new_mipstart_sol._set_var_value(dv, var_value)
        if nb_missed:
            mdl.warning("Found {0} unmatched variable(s) in file {1}", nb_missed, mst_path)
        # return a tiple (solution, effort)
        if not new_mipstart_sol.number_of_var_values:
            mdl.warning("No variable read from MIP start file {0}", mst_path)
            return None
        else:
            return new_mipstart_sol, effort_level

    @classmethod
    def read_many_solutions(cls, xml_solutions, mdl, mst_path):
        mipstarts = []
        for child in xml_solutions:
            if child.tag == "CPLEXSolution":
                mip_start = cls.read_one_solution(child, mdl, mst_path)
                if mip_start is not None:
                    # expecting a tuple (solution, effort:int)
                    mipstarts.append(mip_start)
            else:
                print("Unexpected tag name: {0} ignored, expecting CPLEXSolution".format(child.tag))

        return mipstarts

    @classmethod
    def read_root(cls, root, mdl, mst_path):
        root_name = root.tag
        if root_name == cls.cplex_solution_tag:
            mst1 = MSTReader.read_one_solution(root, mdl, mst_path)
            return None if mst1 is None else [mst1]
        elif root_name == cls.cplex_solutions_tag:
            return MSTReader.read_many_solutions(root, mdl, mst_path)
        else:
            mdl.fatal("Unexpected root element tag, expecting {0}|{1}, found: <{2}>"
                      , cls.cplex_solution_tag, cls.cplex_solutions_tag,
                      root_name)


def read_mst_file(mst_path, mdl):
    try:
        tree = xml_elt.parse(mst_path)
        root = tree.getroot()
        return MSTReader.read_root(root, mdl, mst_path)
    except xml_elt.ParseError as pex:
        mdl.error("XML error: {0!s} in file {1} - read aborted", pex, mst_path)
        # None is for errors
        return None

