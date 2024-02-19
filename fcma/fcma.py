"""
Main module of the fcma package. It defines class Fcma for fast container to machine allocation
"""
from math import ceil
import logging
import os
from time import time as current_time
import itertools
from pulp import (
    LpVariable,
    lpSum,
    LpProblem,
    LpMinimize,
    LpMaximize,
    LpAffineExpression,
    PulpSolverError,
    COIN_CMD,
    log,
    subprocess,
    constants,
    warnings,
    operating_system,
    devnull,
    PULP_CBC_CMD,
)
from .model import *


class Fcma:
    """
    This class provide methods to allocate containers to machines using FCMA algorithms
    """
    # A dictionary with instance class aggregation parameters for each family.
    # They are calculated only once for each instance class family and cahed in this variable
    fm_aggregation_pars = {}

    @staticmethod
    def _remove_ics_same_param_higher_price(ics: list[InstanceClass, ...], values: list[float, ...],
                                            reverse: bool = False) -> None:
        """
        Simplify a list of instance classes in the same family, removing those with the same parameter value,
        but a higher price. After the removal operation, instance classes in the list are sorted
        by increasing values if reverse is false or decreasing values otherwise.
        """

        # Firstly, check instance classes
        if len(ics) == 0:
            return
        for ic in ics[1:]:
            if ic.family != ics[0].family:
                raise ValueError(f"Instance classes to simplify must belong to the same family")

        ics_value = {ics[i]: values[i] for i in range(len(ics))}
        ics.sort(key=lambda instance_class: ics_value[instance_class], reverse=reverse)
        values.sort(reverse=reverse)
        next_ic_index = 0
        while next_ic_index < len(ics):
            # Remove instance classes with the same number of cores but higher price
            min_ic = ics[next_ic_index]
            min_ic_index = next_ic_index
            last_ic_index = next_ic_index
            for ic_index in range(next_ic_index+1, len(ics)):
                ic = ics[ic_index]
                if values[ic_index] == values[min_ic_index]:
                    last_ic_index = ic_index
                    if ic.price < min_ic.price:
                        min_ic = ic
                        min_ic_index = ic_index
                else:
                    break
            val = values[min_ic_index]
            del values[next_ic_index:last_ic_index + 1]
            values.insert(next_ic_index, val)
            del ics[next_ic_index:last_ic_index+1]
            ics.insert(next_ic_index, min_ic)
            next_ic_index += 1

    @staticmethod
    def _get_container_classes(app_family_perfs: dict[tuple[App, InstanceClassFamily], AppFamilyPerf]) \
            -> dict[str, list[ContainerClass, ...]]:
        """
        Get container classes from application performance data in instance class families.
        Simplifications are performed to reduce the number of container classes.
        Returns a dictionary with a list of container classes for each application name
        """
        simplified_ics = {}
        result_ccs = {}
        # Get a list of all the instance classes (ics) that must be considered
        for app_fm in app_family_perfs:
            fm = app_fm[1]  # Application family
            # Get valid instance classes in the family (fm) for the application
            valid_ics = []
            for ic in fm.ics:
                # Consider ics with enough cores to allocate the minimum required by the application
                if app_family_perfs[app_fm].cores.magnitude <= ic.cores.magnitude:
                    valid_ics.append(ic)

            # Remove valid instance classes with the same number of cores but higher price.
            # Valid instance classes after the call are ordered by increasing number of cores
            cores = [valid_ic.cores.magnitude for valid_ic in valid_ics]
            Fcma._remove_ics_same_param_higher_price(valid_ics, cores)

            # Add all ics to the simplified list except those that are multiples of smaller ics.
            # Note that the simplified list contains ics that are valid for at least one application
            while len(valid_ics) > 0:
                min_ic = valid_ics[0]
                valid_ics_copy = copy.copy(valid_ics)
                for ic in valid_ics_copy:
                    if min_ic.is_multiple(ic):
                        valid_ics.remove(ic)
                if fm not in simplified_ics:
                    simplified_ics[fm] = []
                if min_ic not in simplified_ics[fm]:
                    simplified_ics[fm].append(min_ic)

        # Get a list of container classes (ccs) for each application (app). They come from ics in the
        # simplified list that have enough cores
        for app_fm in app_family_perfs:
            app = app_fm[0]
            fm = app_fm[1]
            if app.name not in result_ccs:
                result_ccs[app.name] = []
            for ic in simplified_ics[fm]:
                # Consider ics with enough cores to allocate the minimum cores required by the app
                cores = app_family_perfs[app_fm].cores
                mem = app_family_perfs[app_fm].mem[0]
                perf = app_family_perfs[app_fm].perf
                agg = app_family_perfs[app_fm].agg
                if ic.cores >= cores:
                    result_cc = ContainerClass(app=app, ic=ic, fm=ic.family, cores=cores, mem=mem, perf=perf, aggs=agg)
                    result_ccs[app.name].append(result_cc)

        return result_ccs

    @staticmethod
    def _get_fm_aggregation_pars(fm: InstanceClassFamily) -> FamilyClassAggPars:
        """
        Get aggregation parameters for a given instance class family
        """

        def _get_max_to_try_for(tgt_ic: InstanceClass, origin_ic: InstanceClass,
                                inter_ic: tuple[InstanceClass, ...]) -> int:
            """
            Get the maximum number of instances of type origin_ic that can be used to aggregate (with others)
            into an instance of type target_ic. That number is by default the integer division of the number
            of cores of the target_ic by the number of cores of ic, but if there are intermediate instance types
            between origin_ic and target_ic given by inter_ic, the number can be much smaller.

            The algorithm checks for each integer between 1 and the aforementioned maximum, if the number of
            cores of origin_ic multiplied by that integer is in the set of cores of the intermediate instance
            classes. In that case the maximum is the previous integer.
            """
            max_n = (tgt_ic.cores // origin_ic.cores).magnitude
            larger_cores = set(val.cores for val in inter_ic)
            for n in range(max_n):
                if n * origin_ic.cores in larger_cores:
                    break
            else:
                return max_n
            return n - 1

        def _get_aggregations_for(tgt_ic: InstanceClass, small_ics: tuple[InstanceClass, ...]):
            """
            For each possible combination of smaller_ics instance classes, find if that combination can be used to sum
            the number of cores of the target_ic instance class. In that case, yield the number of instances of
            each type that are needed.
            The algorithm limits the number of instance classes to test using function _get_max_to_try_for.
            """
            max_to_try = []
            for i_index, ic in enumerate(small_ics):
                max_to_try.append(_get_max_to_try_for(tgt_ic, ic, small_ics[i_index + 1:]))
            for val in itertools.product(*[range(n + 1) for n in max_to_try]):
                if sum(q_i * ic.cores for q_i, ic in zip(val, small_ics)) == tgt_ic.cores:
                    yield val

        # Get a tuple with all the instance classes in the family after removing those with the same number of cores,
        # but more expensive. Instance classes in the tuple are sorted by increasing number of cores.
        fm_ics = copy.deepcopy(fm.ics)
        cores = [ic.cores.magnitude for ic in fm_ics]
        Fcma._remove_ics_same_param_higher_price(fm_ics, cores)
        ics = tuple(fm_ics)
        ic_names = tuple(ic.name for ic in ics)

        # Initialize the solution, which is composed of a dictionary n_aggs with the number of aggregations
        # for each instance type and a dictionary p_agg with the number of instances of each type that are
        # used in each aggregation.
        n_agg = [0]  # n_agg[i] = number of aggregations for ics[i]
        p_agg = {}  # p_agg[(i, k, j)] = number of instances of ics[j] in the k-th aggregation of ics[i]

        for i in range(1, len(ics)):
            target_ic = ics[i]
            smaller_ics = ics[:i]
            k = -1
            for k, q in enumerate(_get_aggregations_for(target_ic, smaller_ics)):
                for j, q_j in enumerate(q):
                    if q_j > 0:
                        p_agg[(i, k, j)] = q_j
            n_agg.append(k + 1)

        return FamilyClassAggPars(ic_names, tuple(n_agg), p_agg)

    @staticmethod
    def _aggregate_nodes(n_nodes: dict[InstanceClass, int], agg_pars: FamilyClassAggPars) -> FcmaStatus:
        """
        Aggregate the nodes in n_nodes using the aggregation parameters
        """

        # Number of instance classes
        n_ics = len(agg_pars.ic_names)

        # Number of nodes for each instance class name
        ic_name_n_nodes = {ic_name: 0 for ic_name in agg_pars.ic_names}
        for ic, n in n_nodes.items():
            ic_name_n_nodes[ic.name] = n

        # Get indexes of mi,k terms
        m_indexes = []
        for i in range(n_ics):
            if agg_pars.n_agg[i] > 0:
                for k in range(agg_pars.n_agg[i]):
                    m_indexes.append((i, k))

        # Get the agg_path_node_dec(i,k) = summatory of pi,j,k
        agg_path_node_dec = {}
        for i in range(1, n_ics):
            for k in range(agg_pars.n_agg[i]):
                node_dec = 0
                for j in range(i):
                    if (i, k, j) in agg_pars.p_agg:
                        node_dec += agg_pars.p_agg[(i, k, j)]
                agg_path_node_dec[(i, k)] = node_dec

        # Define the ILP problem and variables
        lp_agg_problem = LpProblem("IC_aggregation_problem", LpMaximize)
        m_vars = LpVariable.dicts(name="M", indices=m_indexes, cat=pulp.constants.LpInteger, lowBound=0)

        # Objective
        lp_agg_problem += (
            lpSum((agg_path_node_dec[(i, k)] - 1) * m_vars[(i, k)]
                  for i in range(1, len(agg_pars.n_agg)) for k in range(agg_pars.n_agg[i])),
            "Minimize_the_total_number_of_nodes_in_family"
        )

        # Constraints
        for j in range(n_ics-1):
            if agg_pars.ic_names[j] not in ic_name_n_nodes:
                continue
            # Get node increments after the aggregations
            if j > 0:
                lp_increments = lpSum(m_vars[(j, k)] for k in range(agg_pars.n_agg[j]))
            else:
                lp_increments = LpAffineExpression(0)
            # Get node decrements after the aggregations
            lp_decrements = LpAffineExpression(0)
            for i in range(j + 1, n_ics):
                for k in range(agg_pars.n_agg[i]):
                    if (i, k, j) in agg_pars.p_agg:
                        lp_decrements += m_vars[(i, k)]*agg_pars.p_agg[(i, k, j)]
            # Add constraint
            lp_agg_problem += (
                ic_name_n_nodes[agg_pars.ic_names[j]] + lp_increments - lp_decrements >= 0,
                f"Final_number_of_{agg_pars.ic_names[j]}_nodes>=0",
            )

        # Solve the problem
        try:
            lp_agg_problem.solve(solver=PULP_CBC_CMD(msg=0), use_mps=False)
        except PulpSolverError as _:
            status = FcmaStatus.INVALID
        else:
            # No exceptions
            status = FcmaStatus.pulp_to_fcma_status(lp_agg_problem.status, lp_agg_problem.sol_status)

        # Get the final number of nodes for each instance class in the family after the aggregation
        if FcmaStatus.is_valid(status):
            agg_ic_name_n_nodes = {}
            for i in range(n_ics):
                # For every instance class calculate the node increments
                # and decrements from the ILP problem solution
                ic_name = agg_pars.ic_names[i]
                increments = 0
                for k in range(agg_pars.n_agg[i]):
                    increments += m_vars[(i, k)].value()
                decrements = 0
                for j in range(i + 1, n_ics):
                    for k in range(agg_pars.n_agg[j]):
                        if (j, k, i) in agg_pars.p_agg:
                            decrements += m_vars[(j, k)].value() * agg_pars.p_agg[(j, k, i)]
                # The final number of nodes is the initial plus the increments and minus the decrements
                agg_ic_name_n_nodes[ic_name] = ic_name_n_nodes[ic_name] + int(increments) - int(decrements)
            # Update the number of nodes after the aggregation
            fm = list(n_nodes.keys())[0].family
            n_nodes.clear()
            for ic in fm.ics:
                if ic.name in agg_ic_name_n_nodes and agg_ic_name_n_nodes[ic.name] > 0:
                    n_nodes[ic] = agg_ic_name_n_nodes[ic.name]
        return status

    def _get_fm_cores_apps(self, app_family_perfs: dict[tuple[App, InstanceClassFamily], AppFamilyPerf]) \
            -> dict[InstanceClassFamily, dict]:
        """
        Get the best families to allocate applications in terms of (req/s)/$. For each family it returns
        a dictionary with the container classes to allocate and the total number of requiered cores. For example:
        {
            m5_r5_c5_fm:    {"apps": [appA, appB], "cores": 12.5},
            m6g_r6g_c6g_fm: {"apps": [appC, appD], "cores": 10.3}
        }
        """

        # Firstly, for each application get the best family in terms of (req/s)/$ and  the number of cores
        # to process the application workload
        best_fm_app_cores = {}
        families = {}
        for app_fm in app_family_perfs:
            app = app_fm[0]
            fm = app_fm[1]
            if fm not in families:
                families[fm] = {"apps": [], "cores": 0}
            # Number of required cores to process the application workload in the instance class family
            n_replicas = ceil((self.workloads[app] / app_family_perfs[app_fm].perf).magnitude)
            cores = n_replicas * app_family_perfs[app_fm].cores
            if app not in best_fm_app_cores:
                best_fm_app_cores[app] = {"fm": fm, "cores": cores}
            else:
                min_price_per_req = best_fm_app_cores[app]["fm"].ics[0].price / \
                                    best_fm_app_cores[app]["fm"].ics[0].cores * app_family_perfs[app_fm].cores
                price_per_req = fm.ics[0].price / fm.ics[0].cores * app_family_perfs[app_fm].cores
                if price_per_req < min_price_per_req:
                    best_fm_app_cores[app] = {"fm": fm, "cores": cores}

        # Get the container classes allocated to each instance class family and the total number of cores
        for app in best_fm_app_cores:
            fm = best_fm_app_cores[app]["fm"]
            families[fm]["apps"].append(app)
            families[fm]["cores"] += best_fm_app_cores[app]["cores"]

        return {fm: val for fm, val in families.items() if val["cores"] > 0}

    def __init__(self, app_family_perfs: dict[tuple[App, InstanceClassFamily], AppFamilyPerf],
                 workloads: dict[App, RequestsPerTime] = None) -> None:
        """
        Creator of Fast Container to Machine Allocater (FCMA). It provides several levels of speed in
        the solution of the allocatotion problem:
        - speed_level = 1. The default speed level. Provides the lowest cost solution, but requires
        the hihest computation time.
        - speed_level = 2. Provides an intermediate cost solution and requires also an intermediate
        computation time.
        - speed_level = 3. Provides the fastest computation time, but with the highest cost.
        """

        self.app_family_perfs = app_family_perfs
        if workloads is not None:
            # Get standard workloads in req/hour
            self.workloads = {}
            for app, workload in workloads.items():
                self.workloads[app] = workload.to("req/hour")

        self.vms: dict[InstanceClassFamily, list[Vm]] = {}  # A list of vms for every instance class family

        # -------------  Prepare data for speed_level = 1
        # Get all the container classes
        self.ccs = Fcma._get_container_classes(app_family_perfs)
        # Create instance and container variables from the container classes
        self._create_vars(self.ccs)
        # Create the ILP problem
        self.lp_problem = LpProblem("Container_problem", LpMinimize)
        self.x_vars = LpVariable.dicts(name="X", indices=self.ic_names, cat=pulp.constants.LpInteger, lowBound=0)
        self.y_vars = LpVariable.dicts(name="Y", indices=self.cc_names, cat=pulp.constants.LpInteger, lowBound=0)
        self._create_objective()
        self._create_core_constraints()
        if workloads is not None:
            # Update performance constraints with the new workloads
            self._update_performance_constraints()

        # -------------  Prepare data for speed_level = 2 and speed_level = 3
        # Get the best instance class family for each application in terms of (req/s)/$ and the number of cores
        # required to process the application workload
        self.best_fm_cores_apps = self._get_fm_cores_apps(app_family_perfs)

        self.solving_stats = SolvingStats()

    def _create_vars(self, ccs: dict[str, list[ContainerClass, ...]]) -> None:
        """
        Creates the variables for the partial ILP problem
        """
        self.cc_names = []
        self.ic_names = []
        self.ics = {}
        for app in ccs:
            for cc in ccs[app]:
                cc_name = str(cc)
                ic_name = str(cc.ic)
                self.ics[ic_name] = cc.ic
                self.cc_names.append(cc_name)
                if ic_name not in self.ic_names:
                    self.ic_names.append(ic_name)

        logging.info(
            "There are %d X variables and %d Z variables",
            len(self.ic_names),
            len(self.cc_names),
        )

    def _create_objective(self) -> None:
        """
        Adds the cost function to optimize to the partial ILP problem
        """
        self.lp_problem += lpSum(
            self.x_vars[ic_name] * self.ics[ic_name].price.magnitude
            for ic_name in self.ic_names
        )

    def _create_core_constraints(self) -> None:
        """
        Adds core constraints to the partial ILP problem
        """

        # Get a dictionary with a list of container names for every instance class name
        cc_names_ic_name = {}
        for app in self.ccs:
            for cc in self.ccs[app]:
                ic_name = cc.ic.name
                if ic_name not in cc_names_ic_name:
                    cc_names_ic_name[ic_name] = [str(cc)]
                else:
                    cc_names_ic_name[ic_name].append(str(cc))

        # Get a dictionary with the container associated to every container name
        cc_cc_name = {str(cc): cc for app in self.ccs for cc in self.ccs[app]}

        for ic_name, ic in self.ics.items():
            # Core constraints
            self.lp_problem += (
                lpSum(
                    self.y_vars[cc_name] * cc_cc_name[cc_name].cores.magnitude
                    for cc_name in cc_names_ic_name[ic_name]
                )
                <= self.x_vars[ic_name] * ic.cores.magnitude,
                f"Enough_cores_in_ic_{ic_name}",
            )

    def _update_performance_constraints(self) -> None:
        """
        Updates performance constraints in the ILP problem from the application workloads
        """

        # Get a dictionary with all the applications
        apps = {str(self.ccs[app][0].app): self.ccs[app][0].app for app in self.ccs}

        # Enough performance
        for app_name in self.ccs:
            constraint_name = f"Enough_perf_for_{str(app_name)}"
            # Remove old constraints
            if constraint_name in self.lp_problem.constraints:
                del self.lp_problem.constraints[constraint_name]
            # Add new constraints
            self.lp_problem += (
                lpSum(self.y_vars[str(cc)] * cc.perf.magnitude for cc in self.ccs[app_name])
                >= self.workloads[apps[app_name]].magnitude, constraint_name,
            )

    def _get_fms_sol(self):
        """
        For every node class family in the solution of the partial ILP problem get classes and numbers
        of nodes and containers making up the solution. For example:
           fms_sol = {
            c5_m5_r5: {"ics": {ic_1: 10, ic_2: 5}, "ccs": {cc_a: 1, cc_b: 4, cc_c: 15}},
            c6_m6_r6: {"ics": {ic_3: 6}, "ccs": {cc_d: 7, cc_e: 1, cc_f: 6, cc_g: 1}}
          }
        """
        # Get instance classes in the solution
        ics_sol = {}
        for ic_sol_name in self.x_vars:
            n_nodes = int(self.x_vars[ic_sol_name].value())
            if n_nodes > 0:
                ics_sol[ic_sol_name] = n_nodes

        # Divide instance and containers in the solution into families
        fms_sol = {}
        for ic_sol_name in ics_sol:
            fm = self.ics[ic_sol_name].family
            if fm not in fms_sol:
                fms_sol[fm] = {"n_nodes": {self.ics[ic_sol_name]: ics_sol[ic_sol_name]}, "ccs": {}}
            else:
                fms_sol[fm]["n_nodes"][self.ics[ic_sol_name]] = ics_sol[ic_sol_name]
        for app in self.ccs:
            for cc in self.ccs[app]:
                cc_name = str(cc)
                n_replicas = int(self.y_vars[cc_name].value())
                if n_replicas > 0:
                    cc.ic = False
                    fms_sol[cc.fm]["ccs"][cc] = n_replicas

        return fms_sol

    def _get_fm_nodes_by_division(self) -> dict[InstanceClassFamily, dict[InstanceClass, int]]:
        """
        Returns the minimum number of nodes of each instance class for the best instance class families
        in terms of (req/s)/$. Algorithm is based on a sequence of integer divisions
        """
        n_nodes = {}
        for fm in self.best_fm_cores_apps:
            ics = fm.ics.copy()
            # Remove instance classes with the same number of cores but a higher price
            cores = [ic.cores.magnitude for ic in ics]
            Fcma._remove_ics_same_param_higher_price(ics, cores)
            n_cores = int(ceil(self.best_fm_cores_apps[fm]["cores"].magnitude))
            n_nodes[fm] = {}
            # Sort instance classes by decreasing number of cores
            ics.sort(key=lambda instance_class: instance_class.cores, reverse=True)
            # The number of nodes of each intance class is obtained through a sequence of integer
            # divisions by the number of instance class cores sorted by decreasing order
            next_ic_index = 0
            while n_cores > 0:
                next_ic = ics[next_ic_index]
                n = n_cores // next_ic.cores.magnitude
                if n > 0:
                    n_cores -= n * next_ic.cores.magnitude
                    n_nodes[fm][next_ic] = n
                elif n == 0 and next_ic_index == len(ics) - 1:
                    if next_ic in n_nodes[fm]:
                        n_nodes[fm][next_ic] += 1
                    else:
                        n_nodes[fm][next_ic] = 1
                    n_cores = 0
                next_ic_index += 1
                self.solving_stats.before_allocation_status.append(FcmaStatus.FEASIBLE)

        return n_nodes

    def _get_fm_nodes_by_ilp(self) -> dict[InstanceClassFamily, dict[InstanceClass, int]]:
        """
        Returns the minimum number of nodes of each instance class for the best instance class families
        in terms of (req/s)/$. Algorithm is based on solving two simple ILP problems
        """
        n_nodes = {}
        for fm in self.best_fm_cores_apps:
            ics = fm.ics.copy()
            # Remove instance classes with the same number of cores but a higher price
            cores = [ic.cores.magnitude for ic in ics]
            Fcma._remove_ics_same_param_higher_price(ics, cores)
            n_cores = int(ceil(self.best_fm_cores_apps[fm]["cores"].magnitude))
            n_nodes[fm] = {}
            # ILP problem variables
            n_vars = LpVariable.dicts(name="N", indices=ics, cat=pulp.constants.LpInteger, lowBound=0)

            # First ILP problem
            lp_problem1 = LpProblem(f"{str(fm)} Calculate_minimum_cores", LpMinimize)
            # - Objective: minimize price
            lp_problem1 += (lpSum(n_vars[ic] * ic.price.magnitude for ic in ics),
                            f"Minimize_the_number_of_cores_for_{str(fm)}")
            # - Restrictions; enoughcores
            lp_problem1 += (
                lpSum(n_vars[ic] * int(ic.cores.magnitude) for ic in ics) >= n_cores,
                f"Number_of_cores_in_fm_{str(fm)}",
            )
            # - Solve
            min_cores = n_cores
            try:
                lp_problem1.solve(solver=PULP_CBC_CMD(msg=0), use_mps=False)
            except PulpSolverError as _:
                lp_problem1_status = FcmaStatus.INVALID
            else:
                # No exceptions
                lp_problem1_status = FcmaStatus.pulp_to_fcma_status(lp_problem1.status, lp_problem1.sol_status)
                min_cores = round(sum(n_vars[ic].value() * ic.cores.magnitude for ic in ics))

            # Second ILP problem
            if FcmaStatus.is_valid(lp_problem1_status):
                lp_problem2 = LpProblem(f"{str(fm)} Minimum_number_of_nodes", LpMinimize)
                # - Objective: minimize the number of nodes
                lp_problem2 += (lpSum(n_vars[ic] for ic in ics), f"Minimize_the_number_of_nodes_in_{str(fm)}")
                # - Restrictions: the total number of cores must be the calculted in the previous problem
                lp_problem2 += (
                    lpSum(n_vars[ic] * int(ic.cores.magnitude) for ic in ics) == min_cores,
                    f"Number_of_cores_in_fm_{str(fm)}",
                )
                try:
                    lp_problem2.solve(solver=PULP_CBC_CMD(msg=0), use_mps=False)
                except PulpSolverError as _:
                    lp_problem2_status = FcmaStatus.INVALID
                else:
                    # No exceptions
                    lp_problem2_status = FcmaStatus.pulp_to_fcma_status(lp_problem2.status, lp_problem2.sol_status)
                status = FcmaStatus.get_worst_status([lp_problem1_status, lp_problem2_status])
                self.solving_stats.before_allocation_status.append(status)
                if FcmaStatus.is_valid(status):
                    # Get the number of nodes for the instance classes in the family
                    for ic in ics:
                        n = n_vars[ic].value()
                        if n > 0:
                            n_nodes[fm][ic] = int(n)
                else:
                    n_nodes[fm] = None
            else:
                n_nodes[fm] = None
                self.solving_stats.before_allocation_status.append(lp_problem1_status)

        return n_nodes

    def _add_vms(self, fm: InstanceClassFamily, cc: ContainerClass, n_containers: int) -> None:
        """
        Add the required virtual machines to allocate the containers. Virtual machines must belong to
        the given family
        """
        # Get the number of containers that could be allocated in an empty node of each instance class in the family
        ics_in_fm = []
        n_allocatable = []
        for ic in fm.ics:
            vm = Vm(ic, test=True)
            n = vm.get_allocatable_number(cc)
            ics_in_fm.append(ic)
            n_allocatable.append(n)

        # Sort by decreasing number of allocatable containers. Remove those that can allocate the same
        # number of containers but are more expensive
        Fcma._remove_ics_same_param_higher_price(ics_in_fm, n_allocatable, reverse=True)
        # Allocate using the largest instance classes to avoid a sequence of small virtual machines
        # that would reduce the probability of allocating containers from next applications.
        while n_containers > 0:
            index = 0
            for ic in ics_in_fm:
                n_to_allocate = n_allocatable[index]
                n_vms = n_containers // n_allocatable[ics_in_fm.index(ic)]
                if n_vms >= 1:
                    if index > 0 and n_containers / n_allocatable[ics_in_fm.index(ic)] > 1:
                        # Use the previous instance class. A single virtual machine is enough.
                        # This option may not be the optimal from the point of view of cost, but
                        # it is simple and reduce the number of virtual machines
                        new_vm = Vm(ics_in_fm[index - 1])
                        new_vm.history.append("Added")
                        new_vm.allocate(cc, n_containers)
                        n_containers = 0
                        self.vms[fm].append(new_vm)
                    else:
                        for _ in range(n_vms):
                            new_vm = Vm(ic)
                            new_vm.history.append("Added")
                            new_vm.allocate(cc, n_to_allocate)
                            n_containers -= n_to_allocate
                            self.vms[fm].append(new_vm)
                index += 1
                if n_containers == 0:
                    break

    def _allocation_with_promotion_and_addition(self, fm_sol: dict) -> None:
        """
        Allocates containers to virtual machines in the same family.
        Containers are given by fm_sol["ccs"] while the number of virtual machines for each
        instance class is given by fm_sol["n_nodes"]
        """

        fm = list(fm_sol["n_nodes"].keys())[0].family
        self.vms[fm] = [Vm(ic) for ic in fm_sol["n_nodes"] for _ in range(fm_sol["n_nodes"][ic])]
        vms = self.vms[fm]

        # The number of containers for every container class is converted into a list of tuples sorted by
        # decreasing number of container cores
        ccs = [(cc, n) for cc, n in fm_sol["ccs"].items()]
        ccs.sort(key=lambda cc_n: cc_n[0].cores, reverse=True)

        for cc, n_containers in ccs:
            # Get the maximum number of containers to meet SFMPL application parameter
            max_containers_in_vm = int(floor(cc.app.sfmpl * self.workloads[cc.app] / cc.perf))

            # -------------------- (1) --------------------
            # Allocate the maximum number of containers in each virtual machine. If it is not possible
            # to allocate that maximum, do not allocate any container in the virtual machine.
            # The objective is to reduce load-balancing penalties and increase container aggregation
            # without compromising task SFMPL.
            # At the same time, two lists of tuples with virtual machine and number of allocatable
            # containers are obtained. One for the virtual machines that could allocate the
            # maximum number of containers and another for the rest. Note that at this point the rest
            # of virtual machines do not allocate any container.

            n_allocatable_max = []
            n_allocatable_no_max = []
            for vm in vms:
                # Get how many containers could be allocated, but do not allocate
                allocatable_containers = vm.get_allocatable_number(cc)
                if allocatable_containers >= max_containers_in_vm:
                    # Only allocate if it is possible to allocate the maximum number of containers
                    vm.allocate(cc, max_containers_in_vm)
                    if allocatable_containers >= max_containers_in_vm:
                        n_allocatable_max.append((vm, allocatable_containers - max_containers_in_vm))
                    n_containers -= max_containers_in_vm
                    if n_containers == 0:
                        break
                elif allocatable_containers > 0:
                    n_allocatable_no_max.append((vm, allocatable_containers))
            if n_containers == 0:
                continue  # Allocation of containers in the instance class familiy has ended

            # -------------------- (2) --------------------
            # Allocate as much containers as possible in the virtual machines that appear in
            # n_allocatable_no_max, that is, virtual machines that currently do not allocate
            # containers of the instance class.

            # Firstly, sort the allocatable list of vms by decreasing number of allocatable containers
            n_allocatable_no_max.sort(key=lambda vm_n: vm_n[1], reverse=True)
            # Allocate the highest number of ccs to the allocatable vms
            for allocatable in n_allocatable_no_max:
                vm = allocatable[0]
                n = min(allocatable[1], n_containers)
                vm.allocate(cc, n)
                n_containers -= n
                if n_containers == 0:
                    break
            if n_containers == 0:
                continue  # Allocation of containers in the instance class family has ended

            # -------------------- (3) --------------------
            # We did our best to fulfill the SFMPL of each application without increasing cost, but cost is
            # the most important, so now allocate as many containers as possible in the virtual machines
            # with the maximum number of containers. In any case, distribute containers equitably to reduce
            # the maximum performance loss of a single node failure.
            n_vms = len(n_allocatable_max)
            for i in range(n_vms):
                vm = n_allocatable_max[i][0]
                # Distribute the containers among the current vm and the next vms
                n_containers_per_vm = ceil(n_containers / (n_vms - i))
                n = min(n_allocatable_max[i][1], n_containers_per_vm)
                vm.allocate(cc, n)
                n_containers -= n
                if n_containers == 0:
                    break
            if n_containers == 0:
                continue  # Allocation of containers in the instance class family has ended

            # -------------------- (4) --------------------
            # Try virtual machine promotion to allocate containers
            # Note that promotion is not possible when all the vms are the bigger ones in the family
            while n_containers > 0:
                new_vm = Vm.promote_vm(vms, cc)
                if new_vm is not None:
                    # Allocate as many containers as possible to the new virtual machine
                    allocatable_containers = new_vm.get_allocatable_number(cc)
                    new_vm.allocate(cc, allocatable_containers)
                    n_containers -= allocatable_containers
                else:
                    break

            # -------------------- (5) --------------------
            # At this point there is no choice but to add new nodes.
            self._add_vms(fm, cc, n_containers)

    def container_aggregation(self):
        """
        Updates the container groups in self.vms replacing several container replicas
        with a larger one
        """
        for fm in self.vms:
            for vm in self.vms[fm]:
                new_cgs = []
                for cg in vm.cgs:
                    cc = cg.cc  # Container class
                    aggregations = cc.aggregations(cg.replicas)
                    for replicas, n in aggregations.items():
                        for _ in range(n):
                            new_cgs.append(cc * replicas)
                vm.cgs = new_cgs

    def solve(self, solving_pars=None):
        """
        Solve the containers to nodes allocation problem using FCMA algorithm and the following parameters:
        - speed_level. One of the following speed levels for the pre-allocation phase:
            * speed_level = 1. The default speed level. Provides the lowest cost solution, but requires
            the hihest computation time.
            * speed_level = 2. Provides an intermediate cost solution and requires also an intermediate
            computation time.
            * speed_level = 3. Provides the fastest computation time, but with the highest cost.
        - partial_ilp_max_seconds. Maximum time in seconds to solve the partial ILP problem.
          Ignored for speed_level=2,3
        - max_agg_desagg_seconds. Maximum aggregation time (speed_level=2) o disaggregation time (speed_level=3).
          Ignored for speed_level=3
        """

        start_solving_time = current_time()
        self.solving_stats = SolvingStats()
        if solving_pars is None:
            # Defaut solving parameters
            solving_pars = SolvingPars()
        self.solving_stats.solving_pars = solving_pars
        speed_level = solving_pars.speed_level
        if speed_level not in [1, 2, 3]:
            raise ValueError("Invalid speed_level value")

        # -----------------------------------------------------------
        # Pre-allocation phase with one of the speed levels
        # -----------------------------------------------------------
        fms_sol = {}  # Solution for each family
        self.solving_stats.before_allocation_status = []

        if speed_level == 2 or speed_level == 3:
            # -------- Pre-allocation phase for speed levels 2 and 3

            if speed_level == 2:
                fm_nodes = self._get_fm_nodes_by_ilp()
            else:  # Speed level 3
                fm_nodes = self._get_fm_nodes_by_division()
            for fm in self.best_fm_cores_apps:
                if fm not in fms_sol:
                    fms_sol[fm] = {"n_nodes": fm_nodes[fm], "ccs": {}}
                for app in self.best_fm_cores_apps[fm]["apps"]:
                    cc = ContainerClass(
                        app=app,
                        ic=None,
                        fm=fm,
                        cores=self.app_family_perfs[(app, fm)].cores,
                        mem=self.app_family_perfs[(app, fm)].mem,
                        perf=self.app_family_perfs[(app, fm)].perf,
                        aggs=self.app_family_perfs[(app, fm)].agg
                    )
                    n_replicas = ceil((self.workloads[app] / cc.perf).magnitude)
                    fms_sol[fm]["ccs"][cc] = n_replicas

        else:
            # -------- Pre-allocation phase for speed level 1

            # Solve the partial ILP problem. Nodes in the same instance class are considered as a pool of cores and
            # memory constraints are ignored
            start_ilp_time = current_time()
            self.solving_stats.partial_ilp_status = self._solve_partial_ilp_problem(solving_pars)
            self.solving_stats.partial_ilp_seconds = current_time() - start_ilp_time

            # Only valid solutions are aggregated
            if FcmaStatus.is_valid(self.solving_stats.partial_ilp_status):
                # Get instance classes and container classes in the solution organized by families
                fms_sol = self._get_fms_sol()
                # Aggregate nodes for each instance class in the solution
                for fm in fms_sol:
                    if fm not in Fcma.fm_aggregation_pars:
                        # Get the familiy aggregation parameters
                        Fcma.fm_aggregation_pars[fm] = Fcma._get_fm_aggregation_pars(fm)
                    status = self._aggregate_nodes(fms_sol[fm]["n_nodes"], Fcma.fm_aggregation_pars[fm])
                    worst_status = FcmaStatus.get_worst_status([self.solving_stats.partial_ilp_status, status])
                    self.solving_stats.before_allocation_status.append(worst_status)

        self.solving_stats.before_allocation_seconds = current_time() - start_solving_time
        # Calculate the cost before allocation
        if FcmaStatus.is_valid(self.solving_stats.before_allocation_status):
            self.solving_stats.before_allocation_cost = CurrencyPerTime("0 usd/hour")
            for fm in fms_sol:
                for ic in fms_sol[fm]["n_nodes"]:
                    self.solving_stats.before_allocation_cost += fms_sol[fm]["n_nodes"][ic] * ic.price

        # -----------------------------------------------------------
        # Allocation phase is common to all the speed levels
        # -----------------------------------------------------------
        # Start container to virtual machine allocation only when the previous phase is successful
        if FcmaStatus.is_valid(self.solving_stats.before_allocation_status):
            # Perform the allocation
            start_time = current_time()
            for fm in fms_sol:
                self._allocation_with_promotion_and_addition(fms_sol[fm])
            self.solving_stats.allocation_seconds = current_time() - start_time

        # -----------------------------------------------------------
        # Perform container aggregation
        # -----------------------------------------------------------
        self.container_aggregation()

        # -----------------------------------------------------------
        # Calculate final statistics
        # -----------------------------------------------------------
        self.solving_stats.total_seconds = current_time() - start_solving_time
        # Allocation is always succesfull, so the final status comes from the previous status
        self.solving_stats.final_status = FcmaStatus.get_worst_status(self.solving_stats.before_allocation_status)
        if FcmaStatus.is_valid(self.solving_stats.final_status):
            # Get the final cost, after the allocation
            self.solving_stats.final_cost = CurrencyPerTime("0 usd/hour")
            for fm in fms_sol:
                for ic in fms_sol[fm]["n_nodes"]:
                    self.solving_stats.final_cost += fms_sol[fm]["n_nodes"][ic] * ic.price

    def _solve_partial_ilp_problem(self, solving_pars: SolvingPars) -> SolvingStats:
        """
        Solves the partial ILP problem
        """
        if solving_pars.partial_ilp_max_seconds is None:
            solver = PULP_CBC_CMD(msg=0)
        else:
            solver = PULP_CBC_CMD(msg=0, timeLimit=solving_pars.partial_ilp_max_seconds)
        try:
            self.lp_problem.solve(solver, use_mps=False)
        except PulpSolverError as _:
            status = FcmaStatus.INVALID
        else:
            # No exceptions
            status = FcmaStatus.pulp_to_fcma_status(self.lp_problem.status, self.lp_problem.sol_status)

        return status


def _solve_cbc_patched(self, lp, use_mps=True):
    """Solve a MIP problem using CBC patched from original PuLP function
    to save a log with cbc's output and take from it the best bound."""

    def take_best_bound_from_log(filename, msg: bool):
        ret = None
        try:
            with open(filename, "r", encoding="utf8") as f:
                for l in f:
                    if msg:
                        print(l, end="")
                    if l.startswith("Lower bound:"):
                        ret = float(l.split(":")[-1])
        except:
            pass
        return ret

    if not self.executable(self.path):
        raise PulpSolverError(
            "Pulp: cannot execute %s cwd: %s" % (self.path, os.getcwd())
        )
    tmpLp, tmpMps, tmpSol, tmpMst = self.create_tmp_files(
        lp.name, "lp", "mps", "sol", "mst"
    )
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
                "Pulp: Error while trying to execute, use msg=True for more details"
                + self.path
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


# Monkey patching
COIN_CMD.solve_CBC = _solve_cbc_patched
