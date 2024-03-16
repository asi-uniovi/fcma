"""
Main module of the fcma package. It defines class Fcma for Fast Container to Machine Allocation
"""

from math import ceil, floor
import copy
import logging
from time import time as current_time
import pulp
from pulp import (
    LpVariable,
    lpSum,
    LpProblem,
    LpMinimize,
    LpMaximize,
    LpAffineExpression,
    PulpSolverError,
    COIN_CMD,
)
from fcma.model import (
    System,
    App,
    RequestsPerTime,
    InstanceClassFamily,
    Vm,
    SolvingStats,
    ContainerClass,
    DELTA_CPU,
    FcmaStatus,
    Solution,
    SolvingPars,
    CurrencyPerTime,
    AllocationCheck,
    delta_to_zero,
    InstanceClass,
    ContainerGroup,
    ComputationalUnits,
    FamilyClassAggPars,
    Allocation,
)

from fcma.helper import (
    check_inputs,
    remove_ics_same_param_higher_price,
    get_fm_aggregation_pars,
    solve_cbc_patched,
)


class Fcma:
    """
    This class provide methods to allocate containers to machines using FCMA algorithms.
    """

    # pylint: disable=too-many-instance-attributes

    # A dictionary with instance class aggregation parameters for each family.
    # They are calculated only once for each instance class family and cached in this variable
    fm_agg_pars: dict[InstanceClassFamily, FamilyClassAggPars] = {}

    def __init__(self, system: System, workloads: dict[App, RequestsPerTime]):
        """
        Constructor of Fast Container to Machine Allocation (FCMA).
        :param system: A dictionary of tuples (application, instance class family) with performance data.
        :param workloads: The workload for each application.
        :raises ValueError: When some input check fails.
        """

        self._system = system
        self._workloads = workloads

        # Check system and workloads
        check_inputs(self._system, self._workloads)

        # Workloads to req/hour
        for app, workload in self._workloads.items():
            self._workloads[app] = workload.to("req/hour")

        # A list of virtual machines (nodes) for every instance class family
        self._vms: dict[InstanceClassFamily, Allocation] = {}

        # Container class names, instance class names and instances classes for speed level 1
        self._cc_names: list[str] = []
        self._ic_names: list[str] = []
        self._ics: dict[str, InstanceClass] = {}

        # Solving parameters
        self._solving_pars = None

        # Variables for speed level 1
        self._ccs = None  # Container classes
        self._lp_problem = None  # ILP problem
        self._x_vars = None  # Number of nodes of each instance class
        self._y_vars = None  # Number of containers of each application in each instance class

        # Variables for speed levels 2 and 3
        self._best_fm_cores_apps = None  # Applications and required total cores for each family

        # Statistics of the problem solution
        self._solving_stats = SolvingStats()

    @staticmethod
    def _get_container_classes(system: System) -> dict[str, list[ContainerClass]]:
        """
        Get container classes for each application in the partial ILP problem. Application container classes
        are simplified removing those that will not participate in the solution.
        :param system: A dictionary of tuples (application, instance class family) with performance data.
        :return: A dictionary with a list of container classes for each application name.
        """

        simplified_ics: dict[InstanceClassFamily, list[InstanceClass]] = {}
        result_ccs: dict[str, list[ContainerClass]] = {}
        # Get a list of all the instance classes (ics) that must be considered
        for app_fm in system:
            fm = app_fm[1]  # Application family (fm)
            # Get valid ics in the family for the application
            valid_ics = []
            for ic in fm.ics:
                # Consider ics with enough cores to allocate the cores required by the application
                if ic.cores + DELTA_CPU > system[app_fm].cores:
                    valid_ics.append(ic)

            # Simplification 1: remove valid instance classes with the same number of cores but higher price.
            # Valid instance classes and their cores are sorted by increasing number of cores
            cores = [valid_ic.cores.magnitude for valid_ic in valid_ics]
            remove_ics_same_param_higher_price(valid_ics, cores)

            # Simplification 2: add all instance classes to the simplified list except those that
            # are multiple of smaller instance classes. Note that the simplified list contains instance classes
            # that are valid for at least one application.
            while len(valid_ics) > 0:
                min_ic = valid_ics[0]
                valid_ics_copy = copy.copy(valid_ics)
                for ic in valid_ics_copy:
                    # Remove multiples, including the minimum instance class
                    if min_ic.is_multiple(ic):
                        valid_ics.remove(ic)
                if fm not in simplified_ics:
                    simplified_ics[fm] = []
                if min_ic not in simplified_ics[fm]:
                    # Add the minimum instance class
                    simplified_ics[fm].append(min_ic)

        # Get a list of container classes (ccs) for each application (app). They come from ics in the
        # simplified list that have enough cores.
        for app_fm in system:
            app = app_fm[0]
            fm = app_fm[1]
            if app.name not in result_ccs:
                result_ccs[app.name] = []
            for ic in simplified_ics[fm]:
                # Consider ics with enough cores to allocate the minimum cores required by the app
                cores = system[app_fm].cores
                # Memory for a non-aggregated container is the first value
                mem = system[app_fm].mem[0]
                perf = system[app_fm].perf
                aggs = system[app_fm].aggs
                if ic.cores + DELTA_CPU > cores:
                    result_cc = ContainerClass(
                        app=app,
                        ic=ic,
                        fm=fm,
                        cores=cores,
                        mem=mem,
                        perf=perf,
                        aggs=aggs,
                    )
                    result_ccs[app.name].append(result_cc)

        return result_ccs

    def _aggregate_nodes(
        self, n_nodes: dict[InstanceClass, int], agg_pars: FamilyClassAggPars
    ) -> FcmaStatus:
        """
        Aggregate nodes in the same instance class family using the aggregation parameters of the family.
        The aggregation objective is to reduce the number of nodes, making them larger.
        :param n_nodes: A dictionary with the number of nodes for each instance class.
        :param agg_pars: Aggregation parameters for the instance class family.
        :return: A FCMA pre-allocation status, which can be OPTIMAL, FEASIBLE or INVALID.
        """
        # pylint: disable-msg=too-many-locals
        # pylint: disable-msg=too-many-branches

        # Number of instance classes
        n_ics = len(agg_pars.ic_names)

        # Number of nodes for each instance class name
        ic_name_n_nodes = {ic_name: 0 for ic_name in agg_pars.ic_names}
        for ic, n in n_nodes.items():
            ic_name_n_nodes[ic.name] = n

        # Get indexes of mi,k terms from the aggregation parameters
        m_indexes = []
        for i in range(n_ics):
            for k in range(agg_pars.n_agg[i]):
                m_indexes.append((i, k))

        # Get the decrement of nodes for each aggregation path, agg_path_node_dec(i,k) = -1 + summatory of pi,j,k
        # -1 comes from the target instance class obtained from the aggregation path
        agg_path_node_dec = {}
        # i = 0 is for the smallest ic in terms of cores, which can not the target aggregation paths
        for i in range(1, n_ics):
            for k in range(agg_pars.n_agg[i]):
                node_dec = 0
                for j in range(i):  # For each smaller instance class
                    if (i, k, j) in agg_pars.p_agg:
                        node_dec += agg_pars.p_agg[(i, k, j)]
                agg_path_node_dec[(i, k)] = node_dec - 1

        # Define the ILP problem and variables
        lp_agg_problem = LpProblem("IC_aggregation_problem", LpMaximize)
        m_vars = LpVariable.dicts(
            name="M", indices=m_indexes, cat=pulp.constants.LpInteger, lowBound=0
        )

        # Objective
        lp_agg_problem += (
            lpSum(
                agg_path_node_dec[(i, k)] * m_vars[(i, k)]
                for i in range(1, len(agg_pars.n_agg))
                for k in range(agg_pars.n_agg[i])
            ),
            "The_sum_of_node_decrements",
        )

        # Constraints. The number of nodes of each instance class after the aggregation can not be negative.
        for j in range(n_ics - 1):
            # The largest instance class in terms of cores allways fullfil the constraint
            # Get node increments after the aggregations
            if j > 0:
                lp_increments = lpSum(m_vars[(j, k)] for k in range(agg_pars.n_agg[j]))
            else:  # The smallest instance class can not have node increments
                lp_increments = LpAffineExpression(0)
            # Get node decrements after the aggregations
            lp_decrements = LpAffineExpression(0)
            for i in range(j + 1, n_ics):
                for k in range(agg_pars.n_agg[i]):
                    if (i, k, j) in agg_pars.p_agg:
                        lp_decrements += m_vars[(i, k)] * agg_pars.p_agg[(i, k, j)]
            # Add constraint
            lp_agg_problem += (
                ic_name_n_nodes[agg_pars.ic_names[j]] + lp_increments - lp_decrements >= 0,
                f"Final_number_of_{agg_pars.ic_names[j]}_nodes_hte_0",
            )

        # Solve the problem
        try:
            lp_agg_problem.solve(self._solving_pars.solver, use_mps=False)
        except PulpSolverError as _:
            status = FcmaStatus.INVALID
        else:
            # No exceptions
            status = FcmaStatus.pulp_to_fcma_status(
                lp_agg_problem.status, lp_agg_problem.sol_status
            )

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
                agg_ic_name_n_nodes[ic_name] = (
                    ic_name_n_nodes[ic_name] + int(increments) - int(decrements)
                )
            # Update the number of nodes after the aggregation
            fm = list(n_nodes.keys())[0].family
            n_nodes.clear()
            for ic in fm.ics:
                if ic.name in agg_ic_name_n_nodes and agg_ic_name_n_nodes[ic.name] > 0:
                    n_nodes[ic] = agg_ic_name_n_nodes[ic.name]

        return status

    def _get_best_fm_apps_cores(self, system: System) -> dict[InstanceClassFamily, dict]:
        """
        Get the best families to allocate applications in terms of (req/s)/$. It ignores memory requirements and
        so ignores instance classes with extended memory. In addition, it assumes that cost of instance classes
        in the family is (K · cores / min_price_cores), where K is a constant for the family, cores is the number
        of cores of the instance class and min_price_cores is the number of cores of the cheapest instance class
        in the family.
        For each family it returns a dictionary with the applications to execute and the minimum number of
        required cores. For example: {m5_r5_c5_fm: {"apps": [appA, appB], "cores": 12.5},
        m6g_r6g_c6g_fm: {"apps": [appC, appD], "cores": 10.3}}.
        :param system: A dictionary of tuples (application, instance class family) with performance data.
        :return: A dictionary with applications and minimum number of cores for each instance class family.
        """

        # Get the family price per core from the cheapest instance class
        fm_price_per_core = {}
        for app_fm in system:
            fm = app_fm[1]
            if fm not in fm_price_per_core:
                min_price_per_core = None
                for ic in fm.ics:
                    ic_price_per_core = ic.price / ic.cores
                    if min_price_per_core is None or ic_price_per_core < min_price_per_core:
                        min_price_per_core = ic_price_per_core
                fm_price_per_core[fm] = min_price_per_core

        # For each application get the best family in terms of (req/s)/$ and also the number of cores
        # and price to process the application workload
        best_fm = {}
        for app_fm in system:
            app = app_fm[0]
            fm = app_fm[1]
            # Number of required cores to process the application workload in the instance class family
            n_replicas = ceil((self._workloads[app] / system[app_fm].perf).magnitude)
            fm_app_cores = n_replicas * system[app_fm].cores
            fm_app_price = fm_price_per_core[fm] * fm_app_cores
            if app not in best_fm:
                best_fm[app] = {"fm": fm, "cores": fm_app_cores, "price": fm_app_price}
            elif fm_app_price < best_fm[app]["price"]:
                best_fm[app]["fm"] = fm
                best_fm[app]["cores"] = fm_app_cores
                best_fm[app]["price"] = fm_app_price

        # Get applications that should be allocated to each instance class family and the number of required cores
        family_apps_cores = {}
        for app, best_for_app in best_fm.items():
            fm = best_for_app["fm"]
            if fm not in family_apps_cores:
                family_apps_cores[fm] = {"apps": [], "cores": 0}
            family_apps_cores[fm]["apps"].append(app)
            family_apps_cores[fm]["cores"] += best_fm[app]["cores"]

        return family_apps_cores

    def _create_vars(self, ccs: dict[str, list[ContainerClass, ...]]) -> None:
        """
        Create the variables for the partial ILP problem, which include a list of container class names,
        a list of instance class names and a dictionary with the instance class associated to each
        instance class name.
        :param ccs: Dictionary with a list of container classes for each application name.
        """

        for app in ccs:
            for cc in ccs[app]:
                cc_name = str(cc)
                ic_name = str(cc.ic)
                self._ics[ic_name] = cc.ic
                self._cc_names.append(cc_name)
                if ic_name not in self._ic_names:
                    self._ic_names.append(ic_name)

        logging.info(
            "There are %d node variables and %d container-to-node variables in the partial ILP problem",
            len(self._ic_names),
            len(self._cc_names),
        )

    def _create_objective_and_contraints(self) -> None:
        """
        Add the cost function to optimize to the partial ILP problem.
        Add core constraints to the partial ILP problem from instance class capacity.
        Add performance constraints in the ILP problem from the application workloads.
        """

        # Cost function: price of the solution
        self._lp_problem += lpSum(
            self._x_vars[ic_name] * self._ics[ic_name].price.magnitude for ic_name in self._ic_names
        )

        # Get a dictionary with a list of container class names for every instance class name
        cc_names_ic_name = {}
        for app in self._ccs:
            for cc in self._ccs[app]:
                ic_name = cc.ic.name
                if ic_name not in cc_names_ic_name:
                    cc_names_ic_name[ic_name] = [str(cc)]
                else:
                    cc_names_ic_name[ic_name].append(str(cc))

        # Get a dictionary with the container associated to every container name
        cc_cc_name = {str(cc): cc for app in self._ccs for cc in self._ccs[app]}

        for ic_name, ic in self._ics.items():
            # Core constraints
            self._lp_problem += (
                lpSum(
                    self._y_vars[cc_name] * cc_cc_name[cc_name].cores.magnitude
                    for cc_name in cc_names_ic_name[ic_name]
                )
                <= self._x_vars[ic_name] * ic.cores.magnitude,
                f"Enough_cores_in_ic_{ic_name}",
            )

        # Get a dictionary with all the applications
        apps = {str(self._ccs[app][0].app): self._ccs[app][0].app for app in self._ccs}

        # Enough performance
        for app_name in self._ccs:
            constraint_name = f"Enough_performance_for_{str(app_name)}"
            # Remove old constraints
            if constraint_name in self._lp_problem.constraints:
                del self._lp_problem.constraints[constraint_name]
            # Add new constraints
            self._lp_problem += (
                lpSum(self._y_vars[str(cc)] * cc.perf.magnitude for cc in self._ccs[app_name])
                >= self._workloads[apps[app_name]].magnitude,
                constraint_name,
            )

    def _get_fms_sol(self) -> dict[InstanceClassFamily]:
        """
        For every instance class family in the solution of the partial ILP problem get the number of nodes
        of each instance class and the container classes to allocate in the family.
        """

        # Get instance classes in the solution
        ics_sol = {}
        for ic_sol_name in self._x_vars:
            n_nodes = int(self._x_vars[ic_sol_name].value())
            if n_nodes > 0:
                ics_sol[ic_sol_name] = n_nodes

        # Organize instance and containers in the solution in families
        fms_sol = {}
        # System instance class families. Those that appear in the system definition
        system_fms = tuple(set(app_fm[1] for app_fm in self._system))
        for ic_sol_name, _ in ics_sol.items():
            ic_fm = self._ics[ic_sol_name].family
            # We are interested in a possible parent family in system definition
            fm = ic_fm.get_parent_fm_in(system_fms)
            if fm not in fms_sol:
                fms_sol[fm] = {
                    "n_nodes": {self._ics[ic_sol_name]: ics_sol[ic_sol_name]},
                    "ccs": {},
                }
            else:
                fms_sol[fm]["n_nodes"][self._ics[ic_sol_name]] = ics_sol[ic_sol_name]

        # Set the number of replicas for each pair application and family
        app_fm_ccs = {}
        for app in self._ccs:
            for cc in self._ccs[app]:
                fm = cc.fm
                cc_name = str(cc)
                n_replicas = int(self._y_vars[cc_name].value())
                if n_replicas > 0:
                    if (app, fm) not in app_fm_ccs:
                        # The same container but with None instance class so its name comes from the
                        # application name and family name.
                        new_cc = ContainerClass(
                            app=cc.app,
                            ic=None,
                            fm=cc.fm,
                            cores=cc.cores,
                            mem=cc.mem,
                            perf=cc.perf,
                            aggs=cc.aggs,
                        )
                        app_fm_ccs[(app, fm)] = new_cc
                        fms_sol[fm]["ccs"][new_cc] = n_replicas
                    else:
                        new_cc = app_fm_ccs[(app, fm)]
                        fms_sol[fm]["ccs"][new_cc] += n_replicas

        return fms_sol

    @staticmethod
    def _get_fm_nodes_by_division(
        best_fm_cores_apps: dict[InstanceClassFamily, any]
    ) -> dict[InstanceClassFamily, dict[InstanceClass, int]]:
        """
        Get the minimum number of nodes of each instance class for the best instance class families
        in terms of (req/s)/$. Algorithm is based on a sequence of integer divisions.
        :best_fm_cores_apps: A dictionary with the number of cores and containers for each instance class family.
        :return: The number of nodes of each instance class in each family.
        """

        n_nodes = {}
        for fm in best_fm_cores_apps:
            ics = fm.ics.copy()
            # Remove instance classes with the same number of cores but a higher price.
            # At the same time instance classes are sorted by decreasing number of cores.
            cores = [ic.cores.magnitude for ic in ics]
            remove_ics_same_param_higher_price(ics, cores, reverse=True)
            n_cores = int(ceil(best_fm_cores_apps[fm]["cores"].magnitude))
            n_nodes[fm] = {}
            # The number of nodes of each intance class is obtained through a sequence of integer
            # divisions by the number of instance class cores sorted by decreasing order
            next_ic_index = 0
            next_ic = ics[next_ic_index]
            while n_cores > 0 and next_ic_index < len(ics):
                next_ic = ics[next_ic_index]
                n = n_cores // next_ic.cores.magnitude
                if n > 0:
                    n_cores -= n * next_ic.cores.magnitude
                    n_nodes[fm][next_ic] = n
                next_ic_index += 1
            if n_cores > 0:
                if next_ic in n_nodes[fm]:
                    n_nodes[fm][next_ic] += 1
                else:
                    n_nodes[fm][next_ic] = 1

        return n_nodes

    def _get_fm_nodes_by_ilp(self) -> dict[InstanceClassFamily, dict[InstanceClass, int]]:
        """
        Returns the minimum number of nodes of each instance class for the best instance class families
        in terms of (req/s)/$. Algorithm is based on solving two simple ILP problems.
        :return: The number of nodes of each instance class in each family.
        """
        # pylint: disable-msg=too-many-locals

        n_nodes = {}
        problem_status = [FcmaStatus.FEASIBLE]

        for fm in self._best_fm_cores_apps:
            n_nodes[fm] = {}
            ics = fm.ics.copy()
            # Remove instance classes with the same number of cores but a higher price
            cores = [ic.cores.magnitude for ic in ics]
            remove_ics_same_param_higher_price(ics, cores)
            n_cores = int(ceil(self._best_fm_cores_apps[fm]["cores"].magnitude))
            # ILP problem variables
            n_vars = LpVariable.dicts(
                name="N", indices=ics, cat=pulp.constants.LpInteger, lowBound=0
            )

            # First ILP problem. The number of cores in n_cores may be unfeasible. For example, if all the
            # instance classes hava an even number of cores, it is not possible to get an odd n_cores value.
            # Thus, firstly we need to calculate the minimum number of cores higher than or equal to n_cores
            # that may be obtained from the isntance classes in the family.
            lp_problem1 = LpProblem(f"{str(fm)}_Calculate_minimum_cores", LpMinimize)
            # - Objective: minimize cores, which is equivalent to minimize price in this approach
            lp_problem1 += (
                lpSum(n_vars[ic] * ic.cores.magnitude for ic in ics),
                f"Minimize_the_number_of_cores_for_{str(fm)}",
            )
            # - Restrictions; enough cores
            lp_problem1 += (
                lpSum(n_vars[ic] * int(ic.cores.magnitude) for ic in ics) >= n_cores,
                f"Number_of_cores_in_fm_{str(fm)}",
            )
            # - Solve
            min_cores = n_cores
            try:
                lp_problem1.solve(solver=self._solving_pars.solver, use_mps=False)
            except PulpSolverError as _:
                lp_problem1_status = FcmaStatus.INVALID
            else:
                # No exceptions
                lp_problem1_status = FcmaStatus.pulp_to_fcma_status(
                    lp_problem1.status, lp_problem1.sol_status
                )
                min_cores = round(sum(n_vars[ic].value() * ic.cores.magnitude for ic in ics))

            # Second ILP problem. Now it is time to get the minimum number of nodes in the family adding up
            # min_cores
            if FcmaStatus.is_valid(lp_problem1_status):
                lp_problem2 = LpProblem(f"{str(fm)}_Minimum_number_of_nodes", LpMinimize)
                # - Objective: minimize the number of nodes
                lp_problem2 += (
                    lpSum(n_vars[ic] for ic in ics),
                    f"Minimize_the_number_of_nodes_in_{str(fm)}",
                )
                # - Restrictions: the total number of cores must be that calculated in previously, min_cores
                lp_problem2 += (
                    lpSum(n_vars[ic] * int(ic.cores.magnitude) for ic in ics) == min_cores,
                    f"Number_of_cores_in_fm_{str(fm)}",
                )
                try:
                    lp_problem2.solve(solver=self._solving_pars.solver, use_mps=False)
                except PulpSolverError as _:
                    lp_problem2_status = FcmaStatus.INVALID
                else:
                    # No exceptions
                    lp_problem2_status = FcmaStatus.pulp_to_fcma_status(
                        lp_problem2.status, lp_problem2.sol_status
                    )
                status = FcmaStatus.get_worst_status([lp_problem1_status, lp_problem2_status])
                problem_status.append(status)
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
                problem_status.append(lp_problem1_status)

        # The solution is feasible at most
        self._solving_stats.pre_allocation_status = FcmaStatus.get_worst_status(problem_status)

        return n_nodes

    @staticmethod
    def _add_vms(
        fm: InstanceClassFamily, cc: ContainerClass, replicas: int
    ) -> tuple[list[Vm], int]:
        """
        Add the required virtual machines to allocate the containers. Virtual machines must belong to
        the given family.
        :param fm: Family of the virtual machines to add.
        :param cc: Container class of containers.
        :param replicas: Number of container replicas that must fit in the virtual machines to add.
        :return: The added virtual machines and the number of allocated replicas.
        """

        initial_replicas = replicas
        vms = []

        # Get the number of containers that could be allocated in an empty node of each instance class in the family
        ics_in_fm = []
        n_allocatable = []
        for ic in fm.ics:
            vm = Vm(ic, ignore_ic_index=True)
            n = vm.get_max_allocatable_cc(cc)
            ics_in_fm.append(ic)
            n_allocatable.append(n)

        # Sort by decreasing number of allocatable containers and remove those that can allocate the same
        # number of containers but are more expensive
        remove_ics_same_param_higher_price(ics_in_fm, n_allocatable, reverse=True)
        # Allocate using the largest instance classes to avoid a sequence of small virtual machines
        # that would reduce the probability of allocating containers coming from next applications.
        while replicas > 0:
            index = 0
            for ic in ics_in_fm:
                n_vms = replicas // n_allocatable[index]
                if n_vms >= 1:
                    if index > 0 and replicas / n_allocatable[index] > 1:
                        # Use the previous instance class. A single virtual machine is enough.
                        # This option may not be the optimal from the point of view of cost, but
                        # it is simple and reduce the number of virtual machines.
                        new_vm = Vm(ics_in_fm[index - 1])
                        new_vm.history.append("Added")
                        replicas -= new_vm.allocate(cc, replicas)
                        vms.append(new_vm)
                    else:
                        for _ in range(n_vms):
                            new_vm = Vm(ic)
                            new_vm.history.append("Added")
                            replicas -= new_vm.allocate(cc, n_allocatable[index])
                            vms.append(new_vm)
                index += 1
                if replicas == 0:
                    break
        return vms, initial_replicas - replicas

    def _allocation_with_promotion_and_addition(self, fm_sol: dict) -> list[Vm]:
        """
        Allocates containers to nodes. Container and nodes come from the pre-allocation phase.
        Allocation is performed using a variation of First-Fit- Decreasing (FFD) algoritm.
        Since the given nodes may not be able to allocate the containers, the algorimth can promote
        or add new nodes, increasing the cost.
        :param: fm_sol: The solution for a singleinstance class family coming from the pre-allocation phase.
        :return: A list with the virtual machines after the allocation
        """
        # pylint: disable-msg=too-many-locals
        # pylint: disable-msg=too-many-branches

        # Get the instance class family from the pre-allocation solution. Note that we are interested in
        # the system family, which may be parenet of the instance class family
        ic_fm = list(fm_sol["n_nodes"].keys())[0].family
        system_fms = tuple(set(app_fm[1] for app_fm in self._system))
        fm = ic_fm.get_parent_fm_in(system_fms)

        # The initial list of virtual machines (vms) come from the pre-allocation phase.
        # All the vms are empty, i.e, do not allocate any container at this time.
        vms = [Vm(ic) for ic in fm_sol["n_nodes"] for _ in range(fm_sol["n_nodes"][ic])]

        # The containers that must be allocated to instance classes in the family are previously
        # sorted by decreasing number of container cores, as required by FFD algorithm.
        ccs = list(fm_sol["ccs"].items())

        ccs.sort(key=lambda cc_n: cc_n[0].cores, reverse=True)

        # For every container class in the solution, cc, we must allocate n_containers
        for cc, n_containers in ccs:
            # Get the maximum number of containers in a vm to meet SFMPL application parameter
            max_containers_in_vm = int(floor(cc.app.sfmpl * self._workloads[cc.app] / cc.perf))

            # -------------------- (1) --------------------
            # Allocate the maximum number of containers in each virtual machine. If it is not possible
            # to allocate that maximum, do not allocate any container in the virtual machine.
            # The objective is to reduce load-balancing penalties and increase container aggregation
            # without compromising task SFMPL.
            # At the same time, two lists of tuples are created with the number of allocatable containers.
            # One for the virtual machines that allocate the maximum number of containers and another for the rest.
            # Note that at this point the rest of virtual machines do not allocate any container.

            allocatable_max = []
            allocatable_no_max = []
            for vm in vms:
                # Check if the maximum number of containers could be allocated
                if vm.is_allocatable_cc(cc, max_containers_in_vm):
                    # Only allocate if it is possible to allocate the maximum number of containers
                    vm.allocate(cc, max_containers_in_vm)
                    n_allocatable = vm.get_max_allocatable_cc(cc)
                    allocatable_max.append((vm, n_allocatable))
                    n_containers -= max_containers_in_vm
                    if n_containers == 0:
                        break
                else:
                    n_allocatable = vm.get_max_allocatable_cc(cc)
                    if n_allocatable > 0:
                        allocatable_no_max.append((vm, n_allocatable))
            if n_containers == 0:
                continue  # Allocation of containers in the instance class familiy has ended

            # -------------------- (2) --------------------
            # Allocate as much containers as possible in the virtual machines that appear in
            # n_allocatable_no_max, that is, virtual machines that currently do not allocate
            # containers of the instance class. This way, SFMPL parameter is fulfilled.

            # Firstly, sort the allocatable list of vms by decreasing number of allocatable containers
            allocatable_no_max.sort(key=lambda vm_n: vm_n[1], reverse=True)
            # Allocate the highest number of ccs to the allocatable vms
            for allocatable in allocatable_no_max:
                vm = allocatable[0]  # Virtual machine to try allocation
                # Maximum number of containers that could be allocated to the vm
                n = min(allocatable[1], n_containers)
                n_containers -= vm.allocate(cc, n)
                if n_containers == 0:
                    break
            if n_containers == 0:
                continue  # Allocation of containers in the instance class cc has ended

            # -------------------- (3) --------------------
            # We did our best to fulfill the SFMPL of each application without increasing cost, but cost is
            # the most important, so now allocate as many containers as possible in the virtual machines
            # with the maximum number of containers. In any case, we distribute containers equitably to reduce
            # the maximum performance loss on a single node failure.
            n_vms = len(allocatable_max)
            for i in range(n_vms):
                vm = allocatable_max[i][0]
                # Distribute the containers among the current vm and the next vms
                n_containers_per_vm = ceil(n_containers / (n_vms - i))
                n = min(allocatable_max[i][1], n_containers_per_vm)
                n_containers -= vm.allocate(cc, n)
                if n_containers == 0:
                    break
            if n_containers == 0:
                continue  # Allocation of containers in the instance class cc has ended

            # -------------------- (4) --------------------
            # Try virtual machine promotion to allocate containers.
            # Note that promotion is not possible when all the vms are the largest ones in the family.
            while n_containers > 0:
                new_vm = Vm.promote_vm(vms, cc)
                if new_vm is not None:
                    # Allocate as many containers as possible to the new virtual machine
                    allocatable_containers = min(n_containers, new_vm.get_max_allocatable_cc(cc))
                    n_containers -= new_vm.allocate(cc, allocatable_containers)
                else:
                    break

            # -------------------- (5) --------------------
            if n_containers > 0:
                # At this point there is no choice but to add new nodes
                added_vms, allocated_containers = Fcma._add_vms(fm, cc, n_containers)
                vms.extend(added_vms)
                n_containers -= allocated_containers

        return vms

    def _container_aggregation(self):
        """
        Updates the container groups in the list of virtual machines, self.vms, replacing several container
        replicas with a larger one.
        """

        for _, vms in self._vms.items():
            for vm in vms:
                new_cgs = []
                for cg in vm.cgs:
                    cc = cg.cc  # Container class
                    aggregations = cc.get_aggregations(cg.replicas)
                    for multiplier, replicas in aggregations.items():
                        new_cgs.append(ContainerGroup(cc * multiplier, replicas))
                vm.cgs = new_cgs

    def _get_last_promotion_cost(self, fm: InstanceClassFamily) -> CurrencyPerTime:
        """
        Get the total cost increment of the last promotion performed on each virtual machine of a family.
        :param fm: Instance class family of the virtual machines.
        :return: The total cost of promotions
        """

        cost = CurrencyPerTime("0 usd/hour")
        for vm in self._vms[fm]:
            if vm.vm_before_promotion is not None:
                cost += vm.ic.price - vm.vm_before_promotion.ic.price
        return cost

    def _optimize_vm_addition(self, fm: InstanceClassFamily) -> tuple[CurrencyPerTime, list[Vm]]:
        """
        Re-allocate replicas previously allocated to virtual machines after their last promotion.
        Now they are allocated to added virtual machines.
        :param fm: Instance class family of the virtual machines.
        :return: A tuple with the cost of the new allocation and a list of new virtual machines with
        the allocated replicas.
        """

        # Get replicas to allocate
        replicas = {}
        for vm in self._vms[fm]:
            if vm.cc_after_promotion is not None:
                for cc in vm.cc_after_promotion:
                    if cc not in replicas:
                        replicas[cc] = vm.cc_after_promotion[cc]
                    else:
                        replicas[cc] += vm.cc_after_promotion[cc]

        # Now it is a problem analogous to that solved with speed_level=3
        fm_cores_apps = {fm: {"apps": [], "cores": ComputationalUnits("0 core")}}
        for cc, n_replicas in replicas.items():
            app = cc.app
            if app not in fm_cores_apps[fm]["apps"]:
                fm_cores_apps[fm]["apps"].append(app)
            fm_cores_apps[fm]["cores"] += cc.cores * n_replicas
        fm_nodes = self._get_fm_nodes_by_division(fm_cores_apps)
        n_nodes_ccs = {"n_nodes": fm_nodes[fm], "ccs": replicas}
        vms = self._allocation_with_promotion_and_addition(n_nodes_ccs)

        # Calculate the allocation cost
        cost = CurrencyPerTime("0 usd/hour")
        for vm in vms:
            cost += vm.ic.price

        return cost, vms

    def _preallocation_phase_speed_levels_1_2(self, speed_level) -> dict:
        """
        Carries out the pre-allocation phase for speed levels 1 and 2.
        :return: A dictionary with the pre-allocation solution.
        """

        # Get the best instance class family for each application in terms of (req/s)/$ and the number of cores
        # required to process the application workload
        self._best_fm_cores_apps = self._get_best_fm_apps_cores(self._system)

        fms_sol = {}  # Solution for each family

        # -------- Pre-allocation phase for speed levels 2 and 3
        if speed_level == 2:
            fm_nodes = self._get_fm_nodes_by_ilp()
        else:  # Speed level 3
            fm_nodes = Fcma._get_fm_nodes_by_division(self._best_fm_cores_apps)
            # The solution is always feasible forsince division algorithm may not be optimal
            self._solving_stats.pre_allocation_status = FcmaStatus.FEASIBLE
        for fm in self._best_fm_cores_apps:
            if fm not in fms_sol:
                fms_sol[fm] = {"n_nodes": fm_nodes[fm], "ccs": {}}
            for app in self._best_fm_cores_apps[fm]["apps"]:
                cc = ContainerClass(
                    app=app,
                    ic=None,
                    fm=fm,
                    cores=self._system[(app, fm)].cores,
                    mem=self._system[(app, fm)].mem,
                    perf=self._system[(app, fm)].perf,
                    aggs=self._system[(app, fm)].aggs,
                )
                n_replicas = ceil((self._workloads[app] / cc.perf).magnitude)
                fms_sol[fm]["ccs"][cc] = n_replicas

        return fms_sol

    def _preallocation_phase_speed_level_1(self) -> dict:
        """
        Solve the partial ILP problem, which ignores memory requirements and aggregates
        capacity of nodes in the sam instance class. This operation corresponds to speed level 1 and
        is previous to the allocation phase.
        :return: A dictionary with the pre-allocation solution.
        """

        # Prepare the ILP problem
        # Get all the container classes. This list is reduced to its minimum through simplification processes
        self._ccs = Fcma._get_container_classes(self._system)
        # From the container classes create a list of instance class names, self.ic_names, a list of container
        # class names, self.cc_names, and a dictionary of instance classes, self.ics, indexed by instance
        # class names.
        self._create_vars(self._ccs)
        # Create the partial ILP problem, which ignores memory and aggregate the capacity of all the
        # nodes in the same instance class.
        self._lp_problem = LpProblem("Partial_ILP_problem", LpMinimize)
        self._x_vars = LpVariable.dicts(
            name="X", indices=self._ic_names, cat=pulp.constants.LpInteger, lowBound=0
        )
        self._y_vars = LpVariable.dicts(
            name="Y", indices=self._cc_names, cat=pulp.constants.LpInteger, lowBound=0
        )
        self._create_objective_and_contraints()

        # Solve the partial ILP problem. Nodes in the same instance class are considered as a pool of cores and
        # memory constraints are ignored.
        start_ilp_time = current_time()
        self._solving_stats.partial_ilp_status = self._solve_partial_ilp_problem()
        self._solving_stats.partial_ilp_seconds = current_time() - start_ilp_time

        # Aggregate nodes in the solution to improve the probability of allocation success
        fms_sol = None
        problem_status = []
        if FcmaStatus.is_valid(self._solving_stats.partial_ilp_status):
            # Get instance classes and container classes in the solution organized by families
            fms_sol = self._get_fms_sol()
            # Aggregate nodes for each instance class in the solution
            for fm, _ in fms_sol.items():
                if fm not in Fcma.fm_agg_pars:
                    # Get the family aggregation parameters and update the aggregation parameters in the cache
                    Fcma.fm_agg_pars[fm] = get_fm_aggregation_pars(fm, Fcma.fm_agg_pars)
                agg_status = self._aggregate_nodes(fms_sol[fm]["n_nodes"], Fcma.fm_agg_pars[fm])
                worst_status = FcmaStatus.get_worst_status(
                    [self._solving_stats.partial_ilp_status, agg_status]
                )
                problem_status.append(worst_status)
        self._solving_stats.pre_allocation_status = FcmaStatus.get_worst_status(problem_status)

        return fms_sol

    def _optimize_last_promotion(self, fm: InstanceClassFamily) -> None:
        """
        Try to undo the last promotion in each virtual machine, evicting containers allocated after
        the promotion. These evicted containers are allocated to added virtual machines and the solution
        is compared in terms of cost with promotion. Finally, the best solution in terms of cost is selected.
        :param fm: Instance class family.
        """

        if self._solving_pars.speed_level in [1, 2]:
            # Get the cost coming from the last promotion of each virtual machine
            last_promotion_cost = self._get_last_promotion_cost(fm)
            if last_promotion_cost > CurrencyPerTime("0 usd/hour"):
                # Get virtual machines and cost that would come from virtual machine addition for
                # the same containers allocated after the last promotion in each virtual machines.
                addition_cost, added_vms = self._optimize_vm_addition(fm)
                # If addition gives a lower cost
                if last_promotion_cost > addition_cost:
                    # Undo the last promotion in each virtual machine
                    for i in range(len(self._vms[fm])):
                        if self._vms[fm][i].vm_before_promotion is not None:
                            self._vms[fm][i] = self._vms[fm][i].vm_before_promotion
                    # Append the virtual machines coming from the addition alternative
                    self._vms[fm].extend(added_vms)

    def solve(self, solving_pars: SolvingPars = None) -> Solution:
        """
        Solve the container to node allocation problem using FCMA algorithm.
        :param solving_pars: Parameters of the FCMA algorithm.
        :returns: The solution to the problem.
        """

        # Reset virtual machine indexes
        Vm.reset_ids()

        start_solving_time = current_time()
        self._solving_stats = SolvingStats()
        self._solving_pars = solving_pars
        if solving_pars is None:
            # Default solving parameters
            self._solving_pars = SolvingPars()
        self._solving_stats.solving_pars = solving_pars
        speed_level = self._solving_pars.speed_level
        if speed_level not in [1, 2, 3]:
            raise ValueError("Invalid speed_level value")

        # -----------------------------------------------------------
        # Pre-allocation phase with one of the speed levels
        # -----------------------------------------------------------
        self._solving_stats.pre_allocation_status = FcmaStatus.INVALID
        if speed_level in (2, 3):
            fms_sol = self._preallocation_phase_speed_levels_1_2(speed_level)
        else:  # For speed level 1
            fms_sol = self._preallocation_phase_speed_level_1()

        # Update solving statistics for any speed level
        self._solving_stats.pre_allocation_seconds = current_time() - start_solving_time
        # Calculate the cost before allocation
        if FcmaStatus.is_valid(self._solving_stats.pre_allocation_status):
            self._solving_stats.pre_allocation_cost = CurrencyPerTime("0 usd/hour")
            for fm, sol in fms_sol.items():
                for ic in sol["n_nodes"]:
                    self._solving_stats.pre_allocation_cost += sol["n_nodes"][ic] * ic.price

        # -----------------------------------------------------------
        # Allocation phase is common to all the speed levels
        # -----------------------------------------------------------
        # Start container to virtual machine allocation only when the previous phase is successful
        # pylint: disable=too-many-nested-blocks
        if FcmaStatus.is_valid(self._solving_stats.pre_allocation_status):

            # Perform the allocation
            start_time = current_time()
            for fm, sol in fms_sol.items():

                self._vms[fm] = self._allocation_with_promotion_and_addition(sol)

                # Until now promotion was prefered to node addition, because aggregating CPU and memory
                # capacities makes future allocations easier. However, when the promotion is the last
                # for a given virtual machine, we can try virtual machine addition and compare the costs.
                self._optimize_last_promotion(fm)

            self._solving_stats.allocation_seconds = current_time() - start_time

            # -----------------------------------------------------------
            # Perform container aggregation
            # -----------------------------------------------------------
            self._container_aggregation()

        # -----------------------------------------------------------
        # Calculate final statistics
        # -----------------------------------------------------------
        self._solving_stats.total_seconds = current_time() - start_solving_time
        # Allocation is always succesfull, so the final status comes from the previous status when
        # there is neither virtual machine promotion nor addition.
        self._solving_stats.final_status = self._solving_stats.pre_allocation_status
        if FcmaStatus.is_valid(self._solving_stats.final_status):
            # Get the final cost, after the allocation
            self._solving_stats.final_cost = CurrencyPerTime("0 usd/hour")
            for fm, vms in self._vms.items():
                for vm in vms:
                    self._solving_stats.final_cost += vm.ic.price
            if self._solving_stats.final_cost > self._solving_stats.pre_allocation_cost:
                self._solving_stats.final_status = FcmaStatus.FEASIBLE

        return Solution(self._vms, self._solving_stats)

    def _solve_partial_ilp_problem(self) -> FcmaStatus:
        """
        Solves the partial ILP problem.
        :return: Statistics of the ILP problem solution.
        """

        try:
            self._lp_problem.solve(solver=self._solving_pars.solver, use_mps=False)
        except PulpSolverError as _:
            status = FcmaStatus.INVALID
        else:
            # No exceptions
            status = FcmaStatus.pulp_to_fcma_status(
                self._lp_problem.status, self._lp_problem.sol_status
            )

        return status

    def check_allocation(self) -> AllocationCheck:
        """
        Check the solution to the allocation problem. Assertions are generated when solution is invalid.
        :return: Allocation slacks for CPU, memory and performance.
        """
        # pylint: disable-msg=too-many-locals

        app_perf = {}
        vm_unused_cpu = {}
        vm_unused_mem = {}
        for vms in self._vms.values():
            for vm in vms:
                ic = vm.ic
                vm_unused_cpu[vm] = ic.cores
                vm_unused_mem[vm] = ic.mem
                for cg in vm.cgs:
                    cc = cg.cc
                    replicas = cg.replicas
                    app = cc.app
                    if app not in app_perf:
                        app_perf[app] = cc.perf * replicas
                    else:
                        app_perf[app] += cc.perf * replicas
                    vm_unused_cpu[vm] -= replicas * cc.cores
                    vm_unused_mem[vm] -= cc.get_mem_from_aggregations(replicas)

        min_unused_cpu = delta_to_zero(
            min(cpu / vm.ic.cores for vm, cpu in vm_unused_cpu.items()).magnitude
        )
        max_unused_cpu = delta_to_zero(
            max(cpu / vm.ic.cores for vm, cpu in vm_unused_cpu.items()).magnitude
        )
        global_unused_cpu = delta_to_zero(
            sum(cpu for _, cpu in vm_unused_cpu.items()).magnitude
            / sum(vm.ic.cores for vm in vm_unused_cpu).magnitude
        )
        min_unused_mem = delta_to_zero(
            min(mem / vm.ic.mem for vm, mem in vm_unused_mem.items()).magnitude
        )
        max_unused_mem = delta_to_zero(
            max(mem / vm.ic.mem for vm, mem in vm_unused_mem.items()).magnitude
        )
        global_unused_mem = delta_to_zero(
            sum(mem for _, mem in vm_unused_mem.items()).magnitude
            / sum(vm.ic.mem for vm in vm_unused_mem).magnitude
        )
        min_surplus_perf = delta_to_zero(
            min((perf / self._workloads[app]).magnitude - 1 for app, perf in app_perf.items())
        )
        max_surplus_perf = delta_to_zero(
            max((perf / self._workloads[app]).magnitude - 1 for app, perf in app_perf.items())
        )
        global_surplus_perf = delta_to_zero(
            sum(perf.magnitude for app, perf in app_perf.items())
            / sum(self._workloads[app].magnitude for app in app_perf)
            - 1
        )

        assert (
            min_unused_cpu >= 0
        ), "One virtual machine has not enough cores to allocate its containers"
        assert (
            min_unused_mem >= 0
        ), "One virtual machine has not enough memory to allocate its containers"
        assert (
            min_surplus_perf >= 0
        ), "One application has not enough performance to process its workload"

        return AllocationCheck(
            min_unused_cpu_percentage=min_unused_cpu * 100,
            max_unused_cpu_percentage=max_unused_cpu * 100,
            global_unused_cpu_percentage=global_unused_cpu * 100,
            min_unused_mem_percentage=min_unused_mem * 100,
            max_unused_mem_percentage=max_unused_mem * 100,
            global_unused_mem_percentage=global_unused_mem * 100,
            min_surplus_perf_percentage=min_surplus_perf * 100,
            max_surplus_perf_percentage=max_surplus_perf * 100,
            global_surplus_perf_percentage=global_surplus_perf * 100,
        )


# Monkey patching
COIN_CMD.solve_CBC = solve_cbc_patched
