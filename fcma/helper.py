"""
A bunch of auxiliary methods for the FCMA analysis
"""

import os
import itertools
import copy
from pulp import (
    PulpSolverError,
    log,
    subprocess,
    constants,
    warnings,
    operating_system,
    devnull,
)
from fcma.model import (
    App,
    RequestsPerTime,
    AppFamilyPerf,
    InstanceClass,
    InstanceClassFamily,
    FamilyClassAggPars,
)


def check_inputs(system, workloads):
    """
    Check system correctness, which includes workloads and performance data.
    :raise ValueError: When a check fails.
    """
    # pylint: disable-msg=too-many-branches

    try:
        if not isinstance(workloads, dict):
            raise ValueError
        workload_apps = workloads.keys()
        for app in workload_apps:
            if not isinstance(app, App):
                raise ValueError from None
            workloads[app].to(
                "req/hour"
            )  # It generates an exception when it is not a RequestPerTime
    except Exception as _:
        raise ValueError(
            f"Workloads must be a dict[{App.__name__}, {RequestsPerTime.__name__}]"
        ) from None

    try:
        if not isinstance(system, dict):
            raise ValueError
        perf_apps_fms = system.keys()
        perf_apps = set()
        for perf_app_fm in perf_apps_fms:
            app = perf_app_fm[0]
            perf_apps.add(app)
            if not isinstance(app, App):
                raise ValueError
            fm = perf_app_fm[1]
            if not isinstance(fm, InstanceClassFamily):
                raise ValueError from None
            if not isinstance(system[perf_app_fm], AppFamilyPerf):
                raise ValueError from None
    except Exception as _:
        raise ValueError(
            f"App family performances must be a dict[({App.__name__},"
            f"{InstanceClassFamily.__name__}), {AppFamilyPerf.__name__}]"
        ) from None

    for app in perf_apps:
        if app not in workloads:
            raise ValueError(f"{app.name} has no workload")
    for app in workload_apps:
        if app not in perf_apps:
            raise ValueError(f"{app.name} has no performance parameters")

    # Check that there is at least one application
    if len(workload_apps) == 0:
        raise ValueError("At least one application is required")

    # Check that there is at least one instance class able to allocate any application
    for app_fm in system:
        app = app_fm[0]
        fm = app_fm[1]
        requested_cores = system[app_fm].cores
        requested_mem = system[app_fm].mem
        if not fm.check_fesibility(requested_cores, requested_mem):
            raise ValueError(
                f"There is no instance class in {fm} family wit enough cores "
                f"or memory for {app.name} application"
            )


def remove_ics_same_param_higher_price(
    ics: list[InstanceClass], values: list[float], reverse: bool = False
) -> None:
    """
    Shorten a list of instance classes in the same family, removing those with the same parameter value,
    but a higher price. After the removal operation, instance classes in the list are sorted
    by increasing or decreasing parameter values.
    :param ics: List of instance classes.
    :param values: List with a value for each instance class.
    :param reverse: Instance classes are sorted by decreasing values when it is false.
    """

    # Firstly, check instance classes
    if len(ics) == 0:
        return

    # Sort instance classes by increasing or decreasing parameter values
    ics_value = {ics[i]: values[i] for i in range(len(ics))}
    ics.sort(key=lambda instance_class: ics_value[instance_class], reverse=reverse)
    values.sort(reverse=reverse)

    # After sorting the instance classes, those with the same parameter value are consecutive, so
    # they form groups [first_ic_index, last_ic_index].
    first_ic_index = 0
    while first_ic_index < len(ics):
        # Find the interval [first_ic_index, last_ic_index] of instance classes with the same parameter value
        min_ic = ics[first_ic_index]
        min_ic_index = first_ic_index
        last_ic_index = first_ic_index
        for ic_index in range(first_ic_index + 1, len(ics)):
            ic = ics[ic_index]
            if values[ic_index] == values[min_ic_index]:
                last_ic_index = ic_index
                if ic.price < min_ic.price:
                    min_ic = ic
                    min_ic_index = ic_index
            else:
                break
        # Remove all the instance classes with the same parameter value and insert that with the lowest price.
        # Values are related to instance classes so the same operation must be replicated on values.
        val = values[min_ic_index]
        del values[first_ic_index : last_ic_index + 1]
        values.insert(first_ic_index, val)
        del ics[first_ic_index : last_ic_index + 1]
        ics.insert(first_ic_index, min_ic)
        first_ic_index += 1  # Prepare for the next group of instance classes


def get_fm_aggregation_pars(
    fm: InstanceClassFamily, fm_agg_pars: dict[InstanceClassFamily, FamilyClassAggPars]
) -> FamilyClassAggPars:
    """
    Get aggregation parameters for a given instance class family.
    :param fm: Instance class family.
    :param fm_agg_pars: Cached aggregation parameters calculated for instance class families.
    :return: The parameters to aggregate nodes in the instance class.
    """
    # pylint: disable-msg=too-many-locals

    def _get_max_to_try_for(
        large_ic: InstanceClass,
        small_ic: InstanceClass,
        inter_ic: tuple[InstanceClass, ...],
    ) -> int:
        """
        Get the maximum number of instances of instance class small_ic that can be used to aggregate into an
        instance class of type large_ic. That number is by default the integer division of the number
        of cores of large_ic by the number of cores of small_ic, but if there are intermediate instance types
        in terms of cores between small_ic and large_ic given by inter_ic, the number can be much lower.
        The algorithm checks for each integer between 1 and the aforementioned maximum, if the number of
        cores of small_ic multiplied by that integer is in the set of cores of the intermediate instance
        classes. In that case the maximum is the previous integer.

        :param large_ic: Large instance class in terms of cores.
        :param small_ic: Small instance class that may participte in aggregations giving the large instance class.
        :param inter_ic: Instance classes with cores between the samll and large instance classes.
        :return: The maximum number of small nodes in any aggregation giving one large node.
        """

        for ic in inter_ic:
            cores_relation = (ic.cores // small_ic.cores).magnitude
            if cores_relation * small_ic.cores == ic.cores:
                return int(cores_relation - 1)
        return int((large_ic.cores // small_ic.cores).magnitude)

    def _get_aggregations_for(large_ic: InstanceClass, small_ics: tuple[InstanceClass, ...]):
        """
        For each possible combination of small instance classes, find if that combination can be used to sum
        the number of cores of the large instance class. In that case, yield the number of instances of
        each class that are needed.
        "param large_ic: Large instance class after the aggregation.
        "param small_ics: Small instance classes sorted by increasing number of cores.
        """

        max_to_try = []
        for i_index, ic in enumerate(small_ics):
            max_to_try.append(_get_max_to_try_for(large_ic, ic, small_ics[i_index + 1 :]))
        for val in itertools.product(*[range(n + 1) for n in max_to_try]):
            if sum(q_i * ic.cores for q_i, ic in zip(val, small_ics)) == large_ic.cores:
                yield val

    # Get a tuple with all the instance classes in the family after removing those with the same number of cores,
    # but more expensive. Instance classes in the tuple are sorted by increasing number of cores.
    fm_ics = copy.deepcopy(fm.ics)
    cores = [ic.cores.magnitude for ic in fm_ics]
    remove_ics_same_param_higher_price(fm_ics, cores)
    ics = tuple(fm_ics)
    ic_names = tuple(ic.name for ic in ics)

    # Firstly, search for a family wih the same insatnce class cores, since in that case,
    # aggregation parameters would be the same.
    ic_cores = tuple(cores)
    for fm_agg_par in fm_agg_pars.values():
        if ic_cores == fm_agg_par.ic_cores:
            p_agg = fm_agg_par.p_agg
            v_agg = fm_agg_par.v_agg
            return FamilyClassAggPars(ic_names, ic_cores, p_agg, v_agg)

    # Initialize the solution, which is composed of a dictionary n_aggs with the number of aggregations
    # for each instance type and a dictionary v_agg with the number of instances of each type that are
    # used in each aggregation.
    p_agg_list = [0]  # p_agg[i] = number of aggregations for ics[i]
    v_agg = (
        {}
    )  # v_agg[(i, k, j)] = number of instances of ics[j] in the k-th aggregation to get ics[i]

    for i in range(1, len(ics)):
        target_ic = ics[i]
        smaller_ics = ics[:i]
        k = -1
        for k, q in enumerate(_get_aggregations_for(target_ic, smaller_ics)):
            for j, q_j in enumerate(q):
                if q_j > 0:
                    v_agg[(i, k, j)] = q_j
        p_agg_list.append(k + 1)

    return FamilyClassAggPars(ic_names, tuple(cores), tuple(p_agg_list), v_agg)


# pylint: disable = E, W, R, C
def solve_cbc_patched(self, lp, use_mps=True):      # pragma: no cover
    """
    Solve a MIP problem using CBC patched from original PuLP function
    to save a log with cbc's output and take from it the best bound.
    """

    def take_best_bound_from_log(filename, msg: bool): # pragma: no cover
        """
        Take the lower bound from the log file. If there is a line with "best possible"
        take the minimum between the lower bound and the best possible because the lower
        bound is only printed with three decimal digits.
        """
        lower_bound = None
        best_possible = None
        try:
            with open(filename, "r", encoding="utf8") as f:
                for l in f:
                    if "best possible" in l:
                        # There are lines like this:
                        # Cbc0010I After 155300 nodes, 10526 on tree, 0.0015583333 best solution, best possible 0.0015392781 (59.96 seconds)
                        # or this: 'Cbc0005I Partial search - best objective 0.0015583333 (best possible 0.0015392781), took 5904080 iterations and 121519 nodes (60.69 seconds)\n'
                        try:
                            best_possible = float(
                                l.split("best possible")[-1].strip().split(" ")[0].split(")")[0]
                            )
                        except:
                            pass
                    if l.startswith("Lower bound:"):
                        lower_bound = float(l.split(":")[-1])
        except:
            pass
        if best_possible is not None and lower_bound is not None:
            return min(lower_bound, best_possible)
        return lower_bound

    if not self.executable(self.path):
        raise PulpSolverError("Pulp: cannot execute %s cwd: %s" % (self.path, os.getcwd()))
    tmpLp, tmpMps, tmpSol, tmpMst = self.create_tmp_files(lp.name, "lp", "mps", "sol", "mst")
    if use_mps:
        vs, variablesNames, constraintsNames, _ = lp.writeMPS(tmpMps, rename=1)
        cmds = " " + tmpMps + " "
        if lp.sense == constants.LpMaximize:
            cmds += "max "
    else:
        vs = lp.writeLP(tmpLp)
        # In the Lp we do not create new variable or constraint names:
        variablesNames = dict((v.name, v.name) for v in vs)
        constraintsNames = dict((c, c) for c in lp.constraints)
        cmds = " " + tmpLp + " "
    if self.optionsDict.get("warmStart", False):
        self.writesol(tmpMst, lp, vs, variablesNames, constraintsNames)
        cmds += "mips {} ".format(tmpMst)
    if self.timeLimit is not None:
        cmds += "sec %s " % self.timeLimit
    options = self.options + self.getOptions()
    for option in options:
        cmds += option + " "
    if self.mip:
        cmds += "branch "
    else:
        cmds += "initialSolve "
    cmds += "printingOptions all "
    cmds += "solution " + tmpSol + " "
    if self.msg:
        pipe = subprocess.PIPE  # Modified
    else:
        pipe = open(os.devnull, "w")
    logPath = self.optionsDict.get("logPath")
    if logPath:
        if self.msg:
            warnings.warn(
                "`logPath` argument replaces `msg=1`. The output will be redirected to the log file."
            )
        pipe = open(self.optionsDict["logPath"], "w")
    log.debug(self.path + cmds)
    args = []
    args.append(self.path)
    args.extend(cmds[1:].split())
    with open(tmpLp + ".log", "w", encoding="utf8") as pipe:
        print(f"You can check the CBC log at {tmpLp}.log", flush=True)
        if not self.msg and operating_system == "win":
            # Prevent flashing windows if used from a GUI application
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            cbc = subprocess.Popen(
                args, stdout=pipe, stderr=pipe, stdin=devnull, startupinfo=startupinfo
            )
        else:
            cbc = subprocess.Popen(args, stdout=pipe, stderr=pipe, stdin=devnull)

        # Modified to get the best bound
        # output, _ = cbc.communicate()
        # if pipe:
        #     print("CBC output")
        #     for line in StringIO(output.decode("utf8")):
        #         if line.startswith("Lower bound:"):
        #             lp.bestBound = float(line.split(":")[1].strip())

        #         print(line, end="")

        if cbc.wait() != 0:
            if pipe:
                pipe.close()
            raise PulpSolverError(
                "Pulp: Error while trying to execute, use msg=True for more details" + self.path
            )
        if pipe:
            pipe.close()
    if not os.path.exists(tmpSol):
        raise PulpSolverError("Pulp: Error while executing " + self.path)
    (
        status,
        values,
        reducedCosts,
        shadowPrices,
        slacks,
        sol_status,
    ) = self.readsol_MPS(tmpSol, lp, vs, variablesNames, constraintsNames)
    lp.assignVarsVals(values)
    lp.assignVarsDj(reducedCosts)
    lp.assignConsPi(shadowPrices)
    lp.assignConsSlack(slacks, activity=True)
    lp.assignStatus(status, sol_status)
    lp.bestBound = take_best_bound_from_log(tmpLp + ".log", self.msg)
    self.delete_tmp_files(tmpMps, tmpLp, tmpSol, tmpMst)
    return status
