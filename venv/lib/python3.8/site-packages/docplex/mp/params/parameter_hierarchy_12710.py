# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# This code is automatically generated
# gendoc: ignore
from docplex.mp.params.parameters import *



# generating code for group: Barrier_Limits
def _group_barrier_limits_params(pgroup):
    return dict(corrections=IntParameter(pgroup, "corrections", "CPX_PARAM_BARMAXCOR", 3013, "maximum correction limit", default_value=-1, min_value=-1.0, max_value=9223372036800000000),
                growth=NumParameter(pgroup, "growth", "CPX_PARAM_BARGROWTH", 3003, "factor used to determine unbounded optimal face", default_value=1e+12, min_value=1.0, max_value=1e+75),
                iteration=IntParameter(pgroup, "iteration", "CPX_PARAM_BARITLIM", 3012, "barrier iteration limit", default_value=9223372036800000000, min_value=0.0, max_value=9223372036800000000),
                objrange=NumParameter(pgroup, "objrange", "CPX_PARAM_BAROBJRNG", 3004, "barrier objective range (above and below zero)", default_value=1e+20, min_value=0.0, max_value=1e+75)
                )


def _group_barrier_limits_make(pgroup):
    return ParameterGroup.make("limits", _group_barrier_limits_params, None, pgroup)


# generating code for group: Barrier
def _group_barrier_params(pgroup):
    return dict(algorithm=IntParameter(pgroup, "algorithm", "CPX_PARAM_BARALG", 3007, "barrier algorithm choice", default_value=0, min_value=0.0, max_value=3.0),
                colnonzeros=IntParameter(pgroup, "colnonzeros", "CPX_PARAM_BARCOLNZ", 3009, "minimum number of entries to consider a column dense", default_value=0, min_value=0.0, max_value=2100000000),
                convergetol=NumParameter(pgroup, "convergetol", "CPX_PARAM_BAREPCOMP", 3002, "tolerance on complementarity for convergence", default_value=1e-08, min_value=1e-12, max_value=1e+75),
                crossover=IntParameter(pgroup, "crossover", "CPX_PARAM_BARCROSSALG", 3018, "barrier crossover choice", default_value=0, min_value=-1.0, max_value=2.0),
                display=IntParameter(pgroup, "display", "CPX_PARAM_BARDISPLAY", 3010, "barrier display level", default_value=1, min_value=0.0, max_value=2.0),
                ordering=IntParameter(pgroup, "ordering", "CPX_PARAM_BARORDER", 3014, "barrier ordering algorithm", default_value=0, min_value=0.0, max_value=3.0),
                qcpconvergetol=NumParameter(pgroup, "qcpconvergetol", "CPX_PARAM_BARQCPEPCOMP", 3020, "tolerance on complementarity for QCP convergence", default_value=1e-07, min_value=1e-12, max_value=1e+75),
                startalg=IntParameter(pgroup, "startalg", "CPX_PARAM_BARSTARTALG", 3017, "barrier starting point algorithm", default_value=1, min_value=1.0, max_value=4.0)
                )


def _group_barrier_subgroups():
    return dict(limits=_group_barrier_limits_make)


def _group_barrier_make(pgroup):
    return ParameterGroup.make("barrier", _group_barrier_params, _group_barrier_subgroups, pgroup)


# generating code for group: Benders_Tolerances
def _group_benders_tolerances_params(pgroup):
    return dict(feasibilitycut=NumParameter(pgroup, "feasibilitycut", "CPX_PARAM_BENDERSFEASCUTTOL", 1509, "tolerance for considering a feasibility cut violated", default_value=1e-06, min_value=1e-09, max_value=0.1),
                optimalitycut=NumParameter(pgroup, "optimalitycut", "CPX_PARAM_BENDERSOPTCUTTOL", 1510, "tolerance for considering an optimality cut violated", default_value=1e-06, min_value=1e-09, max_value=0.1)
                )


def _group_benders_tolerances_make(pgroup):
    return ParameterGroup.make("tolerances", _group_benders_tolerances_params, None, pgroup)


# generating code for group: Benders
def _group_benders_params(pgroup):
    return dict(strategy=IntParameter(pgroup, "strategy", "CPX_PARAM_BENDERSSTRATEGY", 1501, "choice of benders decomposition to use", default_value=0, min_value=-1.0, max_value=3.0),
                workeralgorithm=IntParameter(pgroup, "workeralgorithm", "CPX_PARAM_WORKERALG", 1500, "method for optimizing benders subproblems", default_value=0, min_value=0.0, max_value=5.0)
                )


def _group_benders_subgroups():
    return dict(tolerances=_group_benders_tolerances_make)


def _group_benders_make(pgroup):
    return ParameterGroup.make("benders", _group_benders_params, _group_benders_subgroups, pgroup)


# generating code for group: Conflict
def _group_conflict_params(pgroup):
    return dict(algorithm=IntParameter(pgroup, "algorithm", "CPX_PARAM_CONFLICTALG", 1073, "algorithm used to find minimal conflicts", default_value=0, min_value=0.0, max_value=6.0),
                display=IntParameter(pgroup, "display", "CPX_PARAM_CONFLICTDISPLAY", 1074, "level of the conflict display", default_value=1, min_value=0.0, max_value=2.0)
                )


def _group_conflict_make(pgroup):
    return ParameterGroup.make("conflict", _group_conflict_params, None, pgroup)


# generating code for group: DistMIP_Rampup
def _group_distmip_rampup_params(pgroup):
    return dict(dettimelimit=NumParameter(pgroup, "dettimelimit", "CPX_PARAM_RAMPUPDETTILIM", 2164, "deterministic time limit on rampup", default_value=1e+75, min_value=0.0, max_value=1e+75),
                duration=IntParameter(pgroup, "duration", "CPX_PARAM_RAMPUPDURATION", 2163, "duration of the rampup phase in distributed MIP", default_value=0, min_value=-1.0, max_value=2.0),
                timelimit=NumParameter(pgroup, "timelimit", "CPX_PARAM_RAMPUPTILIM", 2165, "wall-clock time limit on rampup", default_value=1e+75, min_value=0.0, max_value=1e+75)
                )


def _group_distmip_rampup_make(pgroup):
    return ParameterGroup.make("rampup", _group_distmip_rampup_params, None, pgroup)


# generating code for group: DistMIP
def _group_distmip_params(pgroup):
    return dict()


def _group_distmip_subgroups():
    return dict(rampup=_group_distmip_rampup_make)


def _group_distmip_make(pgroup):
    return ParameterGroup.make("distmip", _group_distmip_params, _group_distmip_subgroups, pgroup)


# generating code for group: Emphasis
def _group_emphasis_params(pgroup):
    return dict(memory=BoolParameter(pgroup, "memory", "CPX_PARAM_MEMORYEMPHASIS", 1082, "reduced memory emphasis", default_value=0),
                mip=IntParameter(pgroup, "mip", "CPX_PARAM_MIPEMPHASIS", 2058, "emphasis for MIP optimization", default_value=0, min_value=0.0, max_value=4.0),
                numerical=BoolParameter(pgroup, "numerical", "CPX_PARAM_NUMERICALEMPHASIS", 1083, "extreme numerical caution emphasis", default_value=0)
                )


def _group_emphasis_make(pgroup):
    return ParameterGroup.make("emphasis", _group_emphasis_params, None, pgroup)


# generating code for group: Feasopt
def _group_feasopt_params(pgroup):
    return dict(mode=IntParameter(pgroup, "mode", "CPX_PARAM_FEASOPTMODE", 1084, "relaxation measure", default_value=0, min_value=0.0, max_value=5.0),
                tolerance=NumParameter(pgroup, "tolerance", "CPX_PARAM_EPRELAX", 2073, "minimum amount of accepted relaxation", default_value=1e-06, min_value=0.0)
                )


def _group_feasopt_make(pgroup):
    return ParameterGroup.make("feasopt", _group_feasopt_params, None, pgroup)


# generating code for group: MIP_Cuts
def _group_mip_cuts_params(pgroup):
    return dict(bqp=IntParameter(pgroup, "bqp", "CPX_PARAM_BQPCUTS", 2195, "type of BQP cut generation (only applies to non-convex models solved to global optimality)", default_value=0, min_value=-1.0, max_value=3.0),
                cliques=IntParameter(pgroup, "cliques", "CPX_PARAM_CLIQUES", 2003, "type of clique cut generation", default_value=0, min_value=-1.0, max_value=3.0),
                covers=IntParameter(pgroup, "covers", "CPX_PARAM_COVERS", 2005, "type of cover cut generation", default_value=0, min_value=-1.0, max_value=3.0),
                disjunctive=IntParameter(pgroup, "disjunctive", "CPX_PARAM_DISJCUTS", 2053, "type of disjunctive cut generation", default_value=0, min_value=-1.0, max_value=3.0),
                flowcovers=IntParameter(pgroup, "flowcovers", "CPX_PARAM_FLOWCOVERS", 2040, "type of flow cover cut generation", default_value=0, min_value=-1.0, max_value=2.0),
                gomory=IntParameter(pgroup, "gomory", "CPX_PARAM_FRACCUTS", 2049, "type of Gomory fractional cut generation", default_value=0, min_value=-1.0, max_value=2.0),
                gubcovers=IntParameter(pgroup, "gubcovers", "CPX_PARAM_GUBCOVERS", 2044, "type of GUB cover cut generation", default_value=0, min_value=-1.0, max_value=2.0),
                implied=IntParameter(pgroup, "implied", "CPX_PARAM_IMPLBD", 2041, "type of implied bound cut generation", default_value=0, min_value=-1.0, max_value=2.0),
                liftproj=IntParameter(pgroup, "liftproj", "CPX_PARAM_LANDPCUTS", 2152, "type of Lift and Project cut generation", default_value=0, min_value=-1.0, max_value=3.0),
                localimplied=IntParameter(pgroup, "localimplied", "CPX_PARAM_LOCALIMPLBD", 2181, "type of local implied bound cut generation", default_value=0, min_value=-1.0, max_value=3.0),
                mcfcut=IntParameter(pgroup, "mcfcut", "CPX_PARAM_MCFCUTS", 2134, "type of MCF cut generation", default_value=0, min_value=-1.0, max_value=2.0),
                mircut=IntParameter(pgroup, "mircut", "CPX_PARAM_MIRCUTS", 2052, "type of mixed integer rounding cut generation", default_value=0, min_value=-1.0, max_value=2.0),
                pathcut=IntParameter(pgroup, "pathcut", "CPX_PARAM_FLOWPATHS", 2051, "type of flow path cut generation", default_value=0, min_value=-1.0, max_value=2.0),
                rlt=IntParameter(pgroup, "rlt", "CPX_PARAM_RLTCUTS", 2196, "type of RLT cut generation (only applies to non-convex models solved to global optimality)", default_value=0, min_value=-1.0, max_value=3.0),
                zerohalfcut=IntParameter(pgroup, "zerohalfcut", "CPX_PARAM_ZEROHALFCUTS", 2111, "type of zero-half cut generation", default_value=0, min_value=-1.0, max_value=2.0)
                )


def _group_mip_cuts_make(pgroup):
    return ParameterGroup.make("cuts", _group_mip_cuts_params, None, pgroup)


# generating code for group: MIP_Limits
def _group_mip_limits_params(pgroup):
    return dict(aggforcut=IntParameter(pgroup, "aggforcut", "CPX_PARAM_AGGCUTLIM", 2054, "constraint aggregation limit for cut generation", default_value=3, min_value=0.0, max_value=2100000000),
                auxrootthreads=IntParameter(pgroup, "auxrootthreads", "CPX_PARAM_AUXROOTTHREADS", 2139, "number of threads to use for auxiliary root tasks", default_value=0, min_value=-1.0, max_value=8),
                cutpasses=IntParameter(pgroup, "cutpasses", "CPX_PARAM_CUTPASS", 2056, "number of cutting plane passes", default_value=0, min_value=-1.0, max_value=9223372036800000000),
                cutsfactor=NumParameter(pgroup, "cutsfactor", "CPX_PARAM_CUTSFACTOR", 2033, "rows multiplier factor to limit cuts", default_value=-1.0, min_value=-1.0, max_value=1e+75),
                eachcutlimit=IntParameter(pgroup, "eachcutlimit", "CPX_PARAM_EACHCUTLIM", 2102, "limit on number of cuts for each type per pass", default_value=2100000000, min_value=0.0, max_value=2100000000),
                gomorycand=IntParameter(pgroup, "gomorycand", "CPX_PARAM_FRACCAND", 2048, "candidate limit for generating Gomory fractional cuts", default_value=200, min_value=1.0, max_value=2100000000),
                gomorypass=IntParameter(pgroup, "gomorypass", "CPX_PARAM_FRACPASS", 2050, "pass limit for generating Gomory fractional cuts", default_value=0, min_value=0.0, max_value=9223372036800000000),
                nodes=IntParameter(pgroup, "nodes", "CPX_PARAM_NODELIM", 2017, "branch and cut node limit", default_value=9223372036800000000, min_value=0.0, max_value=9223372036800000000),
                polishtime=NumParameter(pgroup, "polishtime", "CPX_PARAM_POLISHTIME", 2066, "time limit for polishing best solution", default_value=0.0, min_value=0.0, max_value=1e+75),
                populate=IntParameter(pgroup, "populate", "CPX_PARAM_POPULATELIM", 2108, "solutions limit for each populate call", default_value=20, min_value=1.0, max_value=2100000000),
                probedettime=NumParameter(pgroup, "probedettime", "CPX_PARAM_PROBEDETTIME", 2150, "deterministic time limit for probing", default_value=1e+75, min_value=0.0, max_value=1e+75),
                probetime=NumParameter(pgroup, "probetime", "CPX_PARAM_PROBETIME", 2065, "time limit for probing", default_value=1e+75, min_value=0.0, max_value=1e+75),
                repairtries=IntParameter(pgroup, "repairtries", "CPX_PARAM_REPAIRTRIES", 2067, "number of times to try repair heuristic", default_value=0, min_value=-1.0, max_value=9223372036800000000),
                solutions=IntParameter(pgroup, "solutions", "CPX_PARAM_INTSOLLIM", 2015, "mixed integer solutions limit", default_value=9223372036800000000, min_value=1.0, max_value=9223372036800000000),
                strongcand=IntParameter(pgroup, "strongcand", "CPX_PARAM_STRONGCANDLIM", 2045, "strong branching candidate limit", default_value=10, min_value=1.0, max_value=2100000000),
                strongit=IntParameter(pgroup, "strongit", "CPX_PARAM_STRONGITLIM", 2046, "strong branching iteration limit", default_value=0, min_value=0.0, max_value=9223372036800000000),
                submipnodelim=IntParameter(pgroup, "submipnodelim", "CPX_PARAM_SUBMIPNODELIM", 2062, "sub-MIP node limit", default_value=500, min_value=1.0, max_value=9223372036800000000),
                treememory=NumParameter(pgroup, "treememory", "CPX_PARAM_TRELIM", 2027, "upper limit on size of tree in megabytes", default_value=1e+75, min_value=0.0, max_value=1e+75)
                )


def _group_mip_limits_make(pgroup):
    return ParameterGroup.make("limits", _group_mip_limits_params, None, pgroup)


# generating code for group: MIP_PolishAfter
def _group_mip_polishafter_params(pgroup):
    return dict(absmipgap=NumParameter(pgroup, "absmipgap", "CPX_PARAM_POLISHAFTEREPAGAP", 2126, "absolute MIP gap after which to start solution polishing", default_value=0.0, min_value=0.0),
                dettime=NumParameter(pgroup, "dettime", "CPX_PARAM_POLISHAFTERDETTIME", 2151, "deterministic time after which to start solution polishing", default_value=1e+75, min_value=0.0, max_value=1e+75),
                mipgap=NumParameter(pgroup, "mipgap", "CPX_PARAM_POLISHAFTEREPGAP", 2127, "relative MIP gap after which to start solution polishing", default_value=0.0, min_value=0.0, max_value=1.0),
                nodes=IntParameter(pgroup, "nodes", "CPX_PARAM_POLISHAFTERNODE", 2128, "node count after which to start solution polishing", default_value=9223372036800000000, min_value=0.0, max_value=9223372036800000000),
                solutions=IntParameter(pgroup, "solutions", "CPX_PARAM_POLISHAFTERINTSOL", 2129, "solution count after which to start solution polishing", default_value=9223372036800000000, min_value=1.0, max_value=9223372036800000000),
                time=NumParameter(pgroup, "time", "CPX_PARAM_POLISHAFTERTIME", 2130, "time after which to start solution polishing", default_value=1e+75, min_value=0.0, max_value=1e+75)
                )


def _group_mip_polishafter_make(pgroup):
    return ParameterGroup.make("polishafter", _group_mip_polishafter_params, None, pgroup)


# generating code for group: MIP_Pool
def _group_mip_pool_params(pgroup):
    return dict(absgap=NumParameter(pgroup, "absgap", "CPX_PARAM_SOLNPOOLAGAP", 2106, "absolute objective gap", default_value=1e+75, min_value=0.0, max_value=1e+75),
                capacity=IntParameter(pgroup, "capacity", "CPX_PARAM_SOLNPOOLCAPACITY", 2103, "capacity of solution pool", default_value=2100000000, min_value=0.0, max_value=2100000000),
                intensity=IntParameter(pgroup, "intensity", "CPX_PARAM_SOLNPOOLINTENSITY", 2107, "intensity for populating the MIP solution pool", default_value=0, min_value=0.0, max_value=4.0),
                relgap=NumParameter(pgroup, "relgap", "CPX_PARAM_SOLNPOOLGAP", 2105, "relative objective gap", default_value=1e+75, min_value=0.0, max_value=1e+75),
                replace=IntParameter(pgroup, "replace", "CPX_PARAM_SOLNPOOLREPLACE", 2104, "solution pool replacement strategy", default_value=0, min_value=0, max_value=2)
                )


def _group_mip_pool_make(pgroup):
    return ParameterGroup.make("pool", _group_mip_pool_params, None, pgroup)


# generating code for group: MIP_Strategy
def _group_mip_strategy_params(pgroup):
    return dict(backtrack=NumParameter(pgroup, "backtrack", "CPX_PARAM_BTTOL", 2002, "factor for backtracking, lower values give more", default_value=0.9999, min_value=0.0, max_value=1.0),
                bbinterval=IntParameter(pgroup, "bbinterval", "CPX_PARAM_BBINTERVAL", 2039, "interval to select best bound node", default_value=7, min_value=0.0, max_value=9223372036800000000),
                branch=IntParameter(pgroup, "branch", "CPX_PARAM_BRDIR", 2001, "direction of first branch", default_value=0, min_value=-1.0, max_value=1.0),
                dive=IntParameter(pgroup, "dive", "CPX_PARAM_DIVETYPE", 2060, "dive strategy", default_value=0, min_value=0.0, max_value=3.0),
                file=IntParameter(pgroup, "file", "CPX_PARAM_NODEFILEIND", 2016, "file for node storage when tree memory limit is reached", default_value=1, min_value=0.0, max_value=4),
                fpheur=IntParameter(pgroup, "fpheur", "CPX_PARAM_FPHEUR", 2098, "feasibility pump heuristic", default_value=0, min_value=-1.0, max_value=2.0),
                heuristicfreq=IntParameter(pgroup, "heuristicfreq", "CPX_PARAM_HEURFREQ", 2031, "frequency to apply periodic heuristic algorithm", default_value=0, min_value=-1.0, max_value=9223372036800000000),
                kappastats=IntParameter(pgroup, "kappastats", "CPX_PARAM_MIPKAPPASTATS", 2137, "strategy to gather statistics on the kappa of subproblems", default_value=0, min_value=-1.0, max_value=2.0),
                lbheur=BoolParameter(pgroup, "lbheur", "CPX_PARAM_LBHEUR", 2063, "indicator for local branching heuristic", default_value=0),
                miqcpstrat=IntParameter(pgroup, "miqcpstrat", "CPX_PARAM_MIQCPSTRAT", 2110, "MIQCP strategy", default_value=0, min_value=0.0, max_value=2.0),
                nodeselect=IntParameter(pgroup, "nodeselect", "CPX_PARAM_NODESEL", 2018, "node selection strategy", default_value=1, min_value=0.0, max_value=3.0),
                order=BoolParameter(pgroup, "order", "CPX_PARAM_MIPORDIND", 2020, "indicator to use priority orders", default_value=1),
                presolvenode=IntParameter(pgroup, "presolvenode", "CPX_PARAM_PRESLVND", 2037, "node presolve strategy", default_value=0, min_value=-1.0, max_value=3.0),
                probe=IntParameter(pgroup, "probe", "CPX_PARAM_PROBE", 2042, "probing strategy", default_value=0, min_value=-1.0, max_value=3.0),
                rinsheur=IntParameter(pgroup, "rinsheur", "CPX_PARAM_RINSHEUR", 2061, "frequency to apply RINS heuristic", default_value=0, min_value=-2.0, max_value=9223372036800000000),
                search=IntParameter(pgroup, "search", "CPX_PARAM_MIPSEARCH", 2109, "indicator for search method", default_value=0, min_value=0.0, max_value=2.0),
                startalgorithm=IntParameter(pgroup, "startalgorithm", "CPX_PARAM_STARTALG", 2025, "algorithm to solve initial relaxation", default_value=0, min_value=0.0, max_value=6.0),
                subalgorithm=IntParameter(pgroup, "subalgorithm", "CPX_PARAM_SUBALG", 2026, "algorithm to solve subproblems", default_value=0, min_value=0.0, max_value=5.0),
                variableselect=IntParameter(pgroup, "variableselect", "CPX_PARAM_VARSEL", 2028, "variable selection strategy", default_value=0, min_value=-1.0, max_value=4.0)
                )


def _group_mip_strategy_make(pgroup):
    return ParameterGroup.make("strategy", _group_mip_strategy_params, None, pgroup)


# generating code for group: MIP_Tolerances
def _group_mip_tolerances_params(pgroup):
    return dict(absmipgap=NumParameter(pgroup, "absmipgap", "CPX_PARAM_EPAGAP", 2008, "absolute mixed integer optimality gap tolerance", default_value=1e-06, min_value=0.0),
                integrality=NumParameter(pgroup, "integrality", "CPX_PARAM_EPINT", 2010, "integrality tolerance", default_value=1e-05, min_value=0.0, max_value=0.5),
                lowercutoff=NumParameter(pgroup, "lowercutoff", "CPX_PARAM_CUTLO", 2006, "lower objective cutoff", default_value=-1e+75),
                mipgap=NumParameter(pgroup, "mipgap", "CPX_PARAM_EPGAP", 2009, "mixed integer optimality gap tolerance", default_value=0.0001, min_value=0.0, max_value=1.0),
                objdifference=NumParameter(pgroup, "objdifference", "CPX_PARAM_OBJDIF", 2019, "absolute amount successive objective values should differ", default_value=0.0),
                relobjdifference=NumParameter(pgroup, "relobjdifference", "CPX_PARAM_RELOBJDIF", 2022, "relative amount successive objective values should differ", default_value=0.0, min_value=0.0, max_value=1.0),
                uppercutoff=NumParameter(pgroup, "uppercutoff", "CPX_PARAM_CUTUP", 2007, "upper objective cutoff", default_value=1e+75)
                )


def _group_mip_tolerances_make(pgroup):
    return ParameterGroup.make("tolerances", _group_mip_tolerances_params, None, pgroup)


# generating code for group: MIP
def _group_mip_params(pgroup):
    return dict(display=IntParameter(pgroup, "display", "CPX_PARAM_MIPDISPLAY", 2012, "level of mixed integer node display", default_value=2, min_value=0.0, max_value=5.0),
                interval=IntParameter(pgroup, "interval", "CPX_PARAM_MIPINTERVAL", 2013, "interval for printing mixed integer node display", default_value=0, min_value=-9223372036800000000, max_value=9223372036800000000),
                ordertype=IntParameter(pgroup, "ordertype", "CPX_PARAM_MIPORDTYPE", 2032, "type of generated priority order", default_value=0, min_value=0.0, max_value=3.0)
                )


def _group_mip_subgroups():
    return dict(cuts=_group_mip_cuts_make,
                limits=_group_mip_limits_make,
                polishafter=_group_mip_polishafter_make,
                pool=_group_mip_pool_make,
                strategy=_group_mip_strategy_make,
                tolerances=_group_mip_tolerances_make)


def _group_mip_make(pgroup):
    return ParameterGroup.make("mip", _group_mip_params, _group_mip_subgroups, pgroup)


# generating code for group: Output
def _group_output_params(pgroup):
    return dict(intsolfileprefix=StrParameter(pgroup, "intsolfileprefix", "CPX_PARAM_INTSOLFILEPREFIX", 2143, "file name prefix for storing incumbents when they arrive", default_value=""),
                mpslong=BoolParameter(pgroup, "mpslong", "CPX_PARAM_MPSLONGNUM", 1081, "indicator for long numbers in MPS output files", default_value=1),
                writelevel=IntParameter(pgroup, "writelevel", "CPX_PARAM_WRITELEVEL", 1114, "variables to include in .sol and .mst files", default_value=0, min_value=0.0, max_value=4.0)
                )


def _group_output_make(pgroup):
    return ParameterGroup.make("output", _group_output_params, None, pgroup)


# generating code for group: Preprocessing
def _group_preprocessing_params(pgroup):
    return dict(aggregator=IntParameter(pgroup, "aggregator", "CPX_PARAM_AGGIND", 1003, "limit on applications of the aggregator", default_value=-1, min_value=-1.0, max_value=2100000000),
                boundstrength=IntParameter(pgroup, "boundstrength", "CPX_PARAM_BNDSTRENIND", 2029, "type of bound strengthening", default_value=-1, min_value=-1.0, max_value=1.0),
                coeffreduce=IntParameter(pgroup, "coeffreduce", "CPX_PARAM_COEREDIND", 2004, "level of coefficient reduction", default_value=-1, min_value=-1.0, max_value=3.0),
                dependency=IntParameter(pgroup, "dependency", "CPX_PARAM_DEPIND", 1008, "indicator for preprocessing dependency checker", default_value=-1, min_value=-1.0, max_value=3.0),
                dual=IntParameter(pgroup, "dual", "CPX_PARAM_PREDUAL", 1044, "take dual in preprocessing", default_value=0, min_value=-1.0, max_value=1.0),
                fill=PositiveIntParameter(pgroup, "fill", "CPX_PARAM_AGGFILL", 1002, "limit on fill in aggregation", default_value=10, max_value=2100000000),
                linear=IntParameter(pgroup, "linear", "CPX_PARAM_PRELINEAR", 1058, "indicator for linear reductions", default_value=1, min_value=0.0, max_value=1.0),
                numpass=IntParameter(pgroup, "numpass", "CPX_PARAM_PREPASS", 1052, "limit on applications of presolve", default_value=-1, min_value=-1.0, max_value=2100000000),
                presolve=BoolParameter(pgroup, "presolve", "CPX_PARAM_PREIND", 1030, "presolve indicator", default_value=1),
                qcpduals=IntParameter(pgroup, "qcpduals", "CPX_PARAM_CALCQCPDUALS", 4003, "dual calculation for QCPs", default_value=1, min_value=0.0, max_value=2.0),
                qpmakepsd=BoolParameter(pgroup, "qpmakepsd", "CPX_PARAM_QPMAKEPSDIND", 4010, "indicator to make a binary qp psd or tighter", default_value=1),
                qtolin=IntParameter(pgroup, "qtolin", "CPX_PARAM_QTOLININD", 4012, "indicator to linearize products in the objective involving binary variables (for convex MIQP), or all products of bounded variables (for global QP)", default_value=-1, min_value=-1.0, max_value=1.0),
                reduce=IntParameter(pgroup, "reduce", "CPX_PARAM_REDUCE", 1057, "type of primal and dual reductions", default_value=3, min_value=0.0, max_value=3.0),
                relax=IntParameter(pgroup, "relax", "CPX_PARAM_RELAXPREIND", 2034, "indicator for additional presolve of lp relaxation of mip", default_value=-1, min_value=-1.0, max_value=1.0),
                repeatpresolve=IntParameter(pgroup, "repeatpresolve", "CPX_PARAM_REPEATPRESOLVE", 2064, "repeat mip presolve", default_value=-1, min_value=-1.0, max_value=3.0),
                symmetry=IntParameter(pgroup, "symmetry", "CPX_PARAM_SYMMETRY", 2059, "indicator for symmetric reductions", default_value=-1, min_value=-1.0, max_value=5.0)
                )


def _group_preprocessing_make(pgroup):
    return ParameterGroup.make("preprocessing", _group_preprocessing_params, None, pgroup)


# generating code for group: Read
def _group_read_params(pgroup):
    return dict(constraints=IntParameter(pgroup, "constraints", "CPX_PARAM_ROWREADLIM", 1021, "constraint read size", default_value=30000, min_value=0.0, max_value=2100000000),
                datacheck=IntParameter(pgroup, "datacheck", "CPX_PARAM_DATACHECK", 1056, "indicator to check data consistency", default_value=1, min_value=0.0, max_value=2.0),
                fileencoding=StrParameter(pgroup, "fileencoding", "CPX_PARAM_FILEENCODING", 1129, "code page for file reading and writing", default_value="ISO-8859-1"),
                nonzeros=PositiveIntParameter(pgroup, "nonzeros", "CPX_PARAM_NZREADLIM", 1024, "constraint nonzero read size", default_value=250000, max_value=9223372036800000000),
                qpnonzeros=PositiveIntParameter(pgroup, "qpnonzeros", "CPX_PARAM_QPNZREADLIM", 4001, "quadratic nonzero read size", default_value=5000, max_value=9223372036800000000),
                scale=IntParameter(pgroup, "scale", "CPX_PARAM_SCAIND", 1034, "type of scaling used", default_value=0, min_value=-1.0, max_value=1.0),
                variables=IntParameter(pgroup, "variables", "CPX_PARAM_COLREADLIM", 1023, "variable read size", default_value=60000, min_value=0.0, max_value=2100000000)
                )


def _group_read_make(pgroup):
    return ParameterGroup.make("read", _group_read_params, None, pgroup)


# generating code for group: Sifting
def _group_sifting_params(pgroup):
    return dict(algorithm=IntParameter(pgroup, "algorithm", "CPX_PARAM_SIFTALG", 1077, "algorithm used to solve sifting subproblems", default_value=0, min_value=0.0, max_value=4.0),
                simplex=BoolParameter(pgroup, "simplex", "CPX_PARAM_SIFTSIM", 1158, "allow sifting to be called from simplex", default_value=1),
                display=IntParameter(pgroup, "display", "CPX_PARAM_SIFTDISPLAY", 1076, "level of the sifting iteration display", default_value=1, min_value=0.0, max_value=2.0),
                iterations=IntParameter(pgroup, "iterations", "CPX_PARAM_SIFTITLIM", 1078, "sifting iteration limit", default_value=9223372036800000000, min_value=0.0, max_value=9223372036800000000)
                )


def _group_sifting_make(pgroup):
    return ParameterGroup.make("sifting", _group_sifting_params, None, pgroup)


# generating code for group: Simplex_Limits
def _group_simplex_limits_params(pgroup):
    return dict(iterations=IntParameter(pgroup, "iterations", "CPX_PARAM_ITLIM", 1020, "upper limit on primal and dual simplex iterations", default_value=9223372036800000000, min_value=0.0, max_value=9223372036800000000),
                lowerobj=NumParameter(pgroup, "lowerobj", "CPX_PARAM_OBJLLIM", 1025, "lower limit on value of objective", default_value=-1e+75),
                perturbation=IntParameter(pgroup, "perturbation", "CPX_PARAM_PERLIM", 1028, "upper limit on iterations with no progress", default_value=0, min_value=0.0, max_value=2100000000),
                singularity=IntParameter(pgroup, "singularity", "CPX_PARAM_SINGLIM", 1037, "upper limit on repaired singularities", default_value=10, min_value=0.0, max_value=2100000000),
                upperobj=NumParameter(pgroup, "upperobj", "CPX_PARAM_OBJULIM", 1026, "upper limit on value of objective", default_value=1e+75)
                )


def _group_simplex_limits_make(pgroup):
    return ParameterGroup.make("limits", _group_simplex_limits_params, None, pgroup)


# generating code for group: Simplex_Perturbation
def _group_simplex_perturbation_params(pgroup):
    return dict(constant=NumParameter(pgroup, "constant", "CPX_PARAM_EPPER", 1015, "perturbation constant", default_value=1e-06, min_value=1e-08, max_value=1e+75),
                indicator=BoolParameter(pgroup, "indicator", "CPX_PARAM_PERIND", 1027, "perturbation indicator", default_value=0)
                )


def _group_simplex_perturbation_make(pgroup):
    return ParameterGroup.make("perturbation", _group_simplex_perturbation_params, None, pgroup)


# generating code for group: Simplex_Tolerances
def _group_simplex_tolerances_params(pgroup):
    return dict(feasibility=NumParameter(pgroup, "feasibility", "CPX_PARAM_EPRHS", 1016, "feasibility tolerance", default_value=1e-06, min_value=1e-09, max_value=0.1),
                markowitz=NumParameter(pgroup, "markowitz", "CPX_PARAM_EPMRK", 1013, "Markowitz threshold tolerance", default_value=0.01, min_value=0.0001, max_value=0.9999),
                optimality=NumParameter(pgroup, "optimality", "CPX_PARAM_EPOPT", 1014, "reduced cost optimality tolerance", default_value=1e-06, min_value=1e-09, max_value=0.1)
                )


def _group_simplex_tolerances_make(pgroup):
    return ParameterGroup.make("tolerances", _group_simplex_tolerances_params, None, pgroup)


# generating code for group: Simplex
def _group_simplex_params(pgroup):
    return dict(crash=IntParameter(pgroup, "crash", "CPX_PARAM_CRAIND", 1007, "type of crash used", default_value=1, min_value=-1.0, max_value=1),
                dgradient=IntParameter(pgroup, "dgradient", "CPX_PARAM_DPRIIND", 1009, "type of dual gradient used in pricing", default_value=0, min_value=0.0, max_value=5),
                display=IntParameter(pgroup, "display", "CPX_PARAM_SIMDISPLAY", 1019, "level of the iteration display", default_value=1, min_value=0.0, max_value=2),
                pgradient=IntParameter(pgroup, "pgradient", "CPX_PARAM_PPRIIND", 1029, "type of primal gradient used in pricing", default_value=0, min_value=-1.0, max_value=4),
                pricing=IntParameter(pgroup, "pricing", "CPX_PARAM_PRICELIM", 1010, "size of the pricing candidate list", default_value=0, min_value=0.0, max_value=2100000000),
                refactor=IntParameter(pgroup, "refactor", "CPX_PARAM_REINV", 1031, "refactorization interval", default_value=0, min_value=0.0, max_value=10000.0)
                )


def _group_simplex_subgroups():
    return dict(limits=_group_simplex_limits_make,
                perturbation=_group_simplex_perturbation_make,
                tolerances=_group_simplex_tolerances_make)


def _group_simplex_make(pgroup):
    return ParameterGroup.make("simplex", _group_simplex_params, _group_simplex_subgroups, pgroup)


# generating code for group: Tune
def _group_tune_params(pgroup):
    return dict(dettimelimit=NumParameter(pgroup, "dettimelimit", "CPX_PARAM_TUNINGDETTILIM", 1139, "deterministic time limit per model and per test setting", default_value=1e+75, min_value=0.0, max_value=1e+75),
                display=IntParameter(pgroup, "display", "CPX_PARAM_TUNINGDISPLAY", 1113, "level of the tuning display", default_value=1, min_value=0.0, max_value=3.0),
                measure=IntParameter(pgroup, "measure", "CPX_PARAM_TUNINGMEASURE", 1110, "method used to compare across multiple problems", default_value=1, min_value=1.0, max_value=2.0),
                repeat=IntParameter(pgroup, "repeat", "CPX_PARAM_TUNINGREPEAT", 1111, "number of times to permute the model and repeat", default_value=1, min_value=1.0, max_value=2100000000),
                timelimit=NumParameter(pgroup, "timelimit", "CPX_PARAM_TUNINGTILIM", 1112, "time limit per model and per test setting", default_value=1e+75, min_value=0.0, max_value=1e+75)
                )


def _group_tune_make(pgroup):
    return ParameterGroup.make("tune", _group_tune_params, None, pgroup)


# generating code for group: CPXPARAM
def _group_cpxparam_params(pgroup):
    return dict(advance=IntParameter(pgroup, "advance", "CPX_PARAM_ADVIND", 1001, "indicator for advanced starting information", default_value=1, min_value=0.0, max_value=2.0),
                clocktype=IntParameter(pgroup, "clocktype", "CPX_PARAM_CLOCKTYPE", 1006, "type of clock used to measure time", default_value=2, min_value=0.0, max_value=2.0),
                dettimelimit=NumParameter(pgroup, "dettimelimit", "CPX_PARAM_DETTILIM", 1127, "deterministic time limit in ticks", default_value=1e+75, min_value=0.0, max_value=1e+75),
                lpmethod=IntParameter(pgroup, "lpmethod", "CPX_PARAM_LPMETHOD", 1062, "method for linear optimization", default_value=0, min_value=0.0, max_value=6.0),
                optimalitytarget=IntParameter(pgroup, "optimalitytarget", "CPX_PARAM_OPTIMALITYTARGET", 1131, "type of solution CPLEX will attempt to compute", default_value=0, min_value=0.0, max_value=3.0),
                parallel=IntParameter(pgroup, "parallel", "CPX_PARAM_PARALLELMODE", 1109, "parallel optimization mode", default_value=0, min_value=-1, max_value=2),
                qpmethod=IntParameter(pgroup, "qpmethod", "CPX_PARAM_QPMETHOD", 1063, "method for quadratic optimization", default_value=0, min_value=0.0, max_value=6.0),
                randomseed=IntParameter(pgroup, "randomseed", "CPX_PARAM_RANDOMSEED", 1124, "seed to initialize the random number generator", default_value=201610271, min_value=0.0, max_value=2100000000),
                solutiontype=IntParameter(pgroup, "solutiontype", "CPX_PARAM_SOLUTIONTYPE", 1147, "solution information CPLEX will attempt to compute", default_value=0, min_value=0.0, max_value=2.0),
                threads=IntParameter(pgroup, "threads", "CPX_PARAM_THREADS", 1067, "default parallel thread count", default_value=0, min_value=0.0, max_value=2100000000),
                timelimit=NumParameter(pgroup, "timelimit", "CPX_PARAM_TILIM", 1039, "time limit in seconds", default_value=1e+75, min_value=0.0, max_value=1e+75),
                workdir=StrParameter(pgroup, "workdir", "CPX_PARAM_WORKDIR", 1064, "directory for CPLEX working files", default_value="."),
                workmem=NumParameter(pgroup, "workmem", "CPX_PARAM_WORKMEM", 1065, "memory available for working storage (in megabytes)", default_value=2048.0, min_value=0.0, max_value=1e+75)
                )


def _group_cpxparam_subgroups():
    return dict(barrier=_group_barrier_make,
                benders=_group_benders_make,
                conflict=_group_conflict_make,
                distmip=_group_distmip_make,
                emphasis=_group_emphasis_make,
                feasopt=_group_feasopt_make,
                mip=_group_mip_make,
                output=_group_output_make,
                preprocessing=_group_preprocessing_make,
                read=_group_read_make,
                sifting=_group_sifting_make,
                simplex=_group_simplex_make,
                tune=_group_tune_make)


def make_root_params_12710():
    proot = RootParameterGroup.make("parameters", _group_cpxparam_params, _group_cpxparam_subgroups, "12.7.1.0")
    # -- set synchronous params
    proot.read.datacheck._synchronous = True
    return proot

#  --- end of generated code ---
