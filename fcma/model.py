"""
Data classes for containers and nodes of the Fast Container and Machine Allocator (FCMA)
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from enum import Enum
import pulp
from math import floor
from cloudmodel.unified.units import ComputationalUnits, CurrencyPerTime, RequestsPerTime, Storage


class FcmaStatus(Enum):
    """
    Status of FCMA solutions. Theare are two subsets:
    - Pre-allocation: OPTIMAL, FEASIBLE or INVALID.
    - After-allocation: ALLOCATION_OK, ALLOCATION_NOT_ENOUGH_MEM  or ALLOCATION_NOT_ENOUGH_CPU.
    """
    # Pre-allocation status sorted from the best to the worst
    OPTIMAL = 1  # Optimal solution
    FEASIBLE = 2  # Feasible but not optimal. After a timeout
    INVALID = 3  # Invalid result

    # After-allocation status
    ALLOCATION_OK = 10
    ALLOCATION_NOT_ENOUGH_MEM = 11  # Not enough memory for allocation
    ALLOCATION_NOT_ENOUGH_CPU = 12  # Not enough CPU for allocation

    @staticmethod
    def pulp_to_fcma_status(pulp_problem_status: int, pulp_solution_status: int) -> FcmaStatus:
        """
        Calculate a FCMA status from status code for an ILP problem and its solution.
        :param pulp_problem_status: PulP problem status.
        :param pulp_solution_status: PulP solution status.
        :return: A FCMA status.
        """
        if pulp_problem_status == pulp.LpStatusOptimal:
            if pulp_solution_status == pulp.LpSolutionOptimal:
                res = FcmaStatus.OPTIMAL
            else:
                res = FcmaStatus.FEASIBLE
        else:
            res = FcmaStatus.INVALID
        return res

    @staticmethod
    def get_worst_status(status_list: list[FcmaStatus]) -> bool:
        """
        Calculate the worst status in a list of before-allocation status.
        :param status_list: List of pre-allocation status.
        :return: The worst pre-allocation status.
        """
        return FcmaStatus(max(entry.value for entry in status_list))

    @staticmethod
    def is_valid(status: FcmaStatus | list[FcmaStatus]) -> bool:
        """
        Check if the global status is OPTIMAL or FEASIBLE.
        :param status: A FCMA status or list of status.
        :return: True if the worst status is OPTIMAL or FEASIBLE.
        """
        global_status = status
        if isinstance(status, list):
            # The global status is the worst status in the list
            global_status = FcmaStatus.get_worst_status(status)
        return global_status == FcmaStatus.OPTIMAL or global_status == FcmaStatus.FEASIBLE


@dataclass(frozen=True)
class App:
    """
    FCMA application.
    """
    name: str
    sfmpl: [float] = 1.0  # Single failure maximum performance loss in (0, 1]

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class AppFamilyPerf:
    """
    Computational parameters of application containers when it runnings on a given instance class family.
    """
    cores: ComputationalUnits
    perf: RequestsPerTime
    # Memory may be a single value or a list to provide a memory value for each aggregation in [1, maxagg]
    mem: Storage | tuple[Storage, ...]
    # Valid aggregations
    agg: tuple[int] = (1,)
    # Maximum aggregation value that preserves the performance. An n-container aggregation generates 1 container
    # with (n x cores) and at least (n x perf), with n in [1, maxagg]
    maxagg: int = 1

    def __post_init__(self):
        """
        Updates aggregation parameters, checks dimensions are valid and store them in the standard units.
        """
        # Aggregation value 1 is not really an aggregation, but makes program easier
        if 1 not in self.agg:
            new_agg = (1,) + self.agg
            object.__setattr__(self, "agg", new_agg)
        object.__setattr__(self, "maxagg", self.agg[-1])
        object.__setattr__(self, "cores", self.cores.to("cores"))
        if not isinstance(self.mem, tuple):
            mem_value = self.mem.to("gibibytes")
            new_mem = tuple(mem_value for _ in range(self.maxagg))
            object.__setattr__(self, "mem", new_mem)
        else:
            if len(self.mem) != len(self.agg):
                raise ValueError(f"Invalid number of memory items in computational parameters")
            new_mem = tuple(mem.to("gibibytes") for mem in self.mem)
            object.__setattr__(self, "mem", new_mem)
        object.__setattr__(self, "perf", self.perf.to("req/hour"))


@dataclass
class InstanceClass:
    """
    Instance class, i.e., a type of virtual machine in a region.
    """
    name: str
    price: CurrencyPerTime
    cores: ComputationalUnits
    mem: Storage
    family: InstanceClassFamily

    def __post_init__(self):
        """
        Checks dimensions are valid and store them in the standard units.
        """
        object.__setattr__(self, "price", self.price.to("usd/hour"))
        object.__setattr__(self, "cores", self.cores.to("cores"))
        object.__setattr__(self, "mem", self.mem.to("gibibytes"))

        # Add this instance class to its family and parent families
        family = self.family
        family.add_ic_to_family(self)

    def __mul__(self, multiplier: int) -> InstanceClass:
        """
        Multiplies one instance class by a scalar giving an instance class in the same family,
        with price, cores and memory multiplied by that scalar.
        """
        return InstanceClass(f'{multiplier}x{self.name}', self.price * multiplier,
                             self.cores * multiplier, self.mem * multiplier, self.family)

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(repr(self))

    def set_name(self, name: str) -> InstanceClass:
        """
        Set the instance class name.
        :param name: New name for the instance class.
        :return: The new name.
        """
        self.name = name
        return self

    def is_multiple(self, ic: InstanceClass) -> bool:
        """
        Check if the given instance class is in the same family and is multiple in terms of price, CPU and memory.
        :param ic: the instance class to check.
        :return: True when it is in the same family and is multiple in terms of price, CPU and memory.
        """
        if ic.family != self.family:
            return False
        m = ic.cores / self.cores
        if m.is_integer() and abs((m * self.mem - ic.mem).magnitude) < 0.000001 and \
                abs((m * self.price - ic.price).magnitude) < 0.000001:
            return True
        return False

    def _is_cpu_promoted(self, ic: InstanceClass) -> bool:
        """
        Check if the given instance class is in the same family, have the same amount of memory,
        but has more cores (and so it is more expensive).
        :param ic: The instance class to compare with.
        :return: True when the given instance class is CPU promoted.
        """
        if ic.family == self.family and ic.cores > self.cores and abs(1 - self.mem / ic.mem) < 0.000001:
            return True
        return False

    def _is_mem_promoted(self, ic: InstanceClass) -> bool:
        """
        Check if the instance class is in the same family, have the same number of cores,
        but has more memory (and so it is more expensive).
        :param ic: The instance class to compare with.
        :return: True when the given instance class is memory promoted.
        """
        if ic.family == self.family and ic.cores == self.cores and abs(1 - self.mem / ic.mem) > 0.000001:
            return True
        return False

    def is_promoted(self, ic: InstanceClass) -> bool:
        """
        Check if the instance class is in the same family and is CPU or memory promoted
        (and so it is more expensive).
        :param ic: The instance class to compare with.
        :return: True when the given instance class is CPU or memory promoted.
        """
        return self._is_mem_promoted(ic) or self._is_cpu_promoted(ic)


class InstanceClassFamily:
    """
    Family of instance classes. Performance is assumed to be the same for all the instance classes
    in a family, whenever instance classes have enough CPU and memory.
    """

    def __init__(self, name: str,
                 parent_fms: InstanceClassFamily | list[InstanceClassFamily] = None) -> InstanceClassFamily:
        """
        Create an instance class family object. If optional parent families are provided, any
        instance class in the family will be also an instance class of the parent families.
        :param name: Name for the instance class family.
        :parent_fms: One parent family or a list of parent families.
        :return: An instance class family object.
        """
        if parent_fms is None:
            self.parent_fms = []
        elif isinstance(parent_fms, InstanceClassFamily):
            self.parent_fms = [parent_fms]
        else:
            self.parent_fms = parent_fms
        self.name = name
        self.ics = []  # Instance classes in the family

    def __str__(self) -> str:
        return self.name

    def add_ic_to_family(self, ic: InstanceClass) -> None:
        """
        Add the instance class to the family and its parent families.
        :param ic: Instance class to add.
        """
        if ic not in self.ics:
            self.ics.append(ic)
        for parent in self.parent_fms:
            parent.add_ic_to_family(ic)

    def add_parent_families(self, parent_fms: InstanceClassFamily | list[InstanceClassFamily]) -> None:
        """
        Add parent families to the family.
        :param parent_fms: An parent familiy or a list of parent families.
        """
        if parent_fms is None:
            return
        if isinstance(parent_fms, InstanceClassFamily):
            parents = [parent_fms]
        else:
            parents = parent_fms
        self.parent_fms.extend(parents)

        # Add the instance classes of the family to its parents
        for parent in parents:
            for ic in self.ics:
                if ic not in parent.ics:
                    parent.add_ic_to_family(ic)


@dataclass
class ContainerClass:
    """
    Represents a container class, i.e., a type of container running an application with some computational resources.
    """
    app: App
    ic: InstanceClass  # Instance class is None when the container is not allocated
    fm: InstanceClassFamily  # Instance class family
    cores: ComputationalUnits
    mem: Storage | tuple[Storage, ...]  # Memory may be a tuple to provide a value for each aggregation
    perf: RequestsPerTime
    aggs: tuple[int]  # Container valid aggregations
    agg_level: int = 1 # Container current aggregation level

    def __post_init__(self):
        """
        Checks dimensions are valid and store them in the standard units.
        """
        object.__setattr__(self, "cores", self.cores.to("cores"))
        if not isinstance(self.mem, tuple):
            std_mem = tuple(self.mem.to("gibibytes") for _ in range(self.aggs[-1]))
        else:
            if len(self.mem) != self.aggs[-1]:
                if len(self.mem) != 1:
                    raise ValueError(f"Invalid number of memory items in computational parameters")
                # A single memory value is assumed to be independent of aggregation size until maxagg
                else:
                    std_mem = tuple(self.mem[0].to("gibibytes") for _ in range(self.aggs[-1]))
            else:
                std_mem = tuple(mem.to("gibibytes") for mem in self.mem)
        object.__setattr__(self, "mem", std_mem)
        object.__setattr__(self, "perf", self.perf.to("req/hour"))

    def __str__(self) -> str:
        if self.ic is None:
            return f"{self.app.name}-{self.fm.name}"
        else:
            return f"{self.app.name}-{self.ic.name}"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __mul__(self, replicas: int) -> ContainerClass:
        """
        Aggregate the given replicas of the container to get a bigger container.
        :param replicas: A number of replicas in aggs.
        :return: The container obtained from aggregation.
        """
        container = ContainerClass(
            app=self.app,
            ic=self.ic,
            fm=self.fm,
            cores=self.cores * replicas,
            mem=self.mem[self.aggs.index(replicas)],
            perf=self.perf * replicas,
            aggs=(1,),
            agg_level=self.agg_level*replicas
        )
        return container

    def aggregations(self, replicas: int) -> dict[int, int]:
        """
        Calculate container aggregations for a given number of replicas. For example, 9 replicas may be
        aggregated into 2 containers made of 4 replicas each and 1 container made 1 replica.
        (without aggregation), returning {2: 4, 1: 1}.
        :param replicas: The number of container replicas to aggregate.
        :return: A dictionary with the aggregations.
        """
        aggs = list(self.aggs)  # List of aggregations that include the 1
        aggs.sort(reverse=True)  # Aggregations are sorted by increasing values
        res = {}
        while replicas > 0:
            for agg in aggs:
                n = replicas // agg
                if n > 0:
                    res[agg] = n
                    replicas -= n * agg
                    if replicas == 0:  # It will end at some point since aggregations include value 1
                        break
        return res

    def get_mem_from_aggregations(self, replicas: int) -> Storage:
        """
        Get the memory that would require an aggregation made of several replicas.
        :param replicas: The number of container replicas.
        :return: The memory required if the replicas were aggregated.
        """
        # Firsly, get the aggregations for the given number of replicas
        n_aggs = self.aggregations(replicas)
        # Add the memory required by all the aggregations
        mem = Storage("0 gibibytes")
        for agg in n_aggs:
            agg_index = self.aggs.index(agg)
            mem += self.mem[agg_index] * n_aggs[agg]

        return mem


@dataclass
class ContainerGroup:
    """
    Represents a group of container replicas of the same application.
    """
    cc: ContainerClass
    replicas: int


class Vm:
    """
    Represents a virtual machine.
    """

    # Virtual machines in the same instance class get an increasing index
    _last_ic_index = {}

    @staticmethod
    def reset_ids():
        """
        Reset the instance class indexes, so the new virtual machine of each instance class will get index 1.
        """
        Vm._last_ic_index.clear()

    @staticmethod
    def promote_vm(vms: list[Vm], cc: ContainerClass) -> Vm:
        """
        Promote one virtual machine in the list of virtual machines to allocate at least
        one container of the given container class at the lowest cost.
        The promoted vm is replaced by the new one in the list of virtual machines.
        :param vms: List of virtual machines elegible for promotion.
        :param cc: Container class that should be allocated.
        :return: The new virtual machine if promotion is feasible or None otherwise.
        """
        promoted_vm = None  # vm for a promotion
        new_vm_ic = None  # New instance class for a promoted vm
        lowest_price = None  # Lowest price coming from a vm promotion
        for vm in vms:
            # Find the cheapest promotion for the given vm that can allocate the cc.
            # Note that promotion of the largest instance class in the family is not possible, returning None.
            new_ic = vm._cheapest_ic_promotion(cc)
            if new_ic is not None:
                price = new_ic.price - vm.ic.price  # Price of promotion vm.ic -> new_ic
                if lowest_price is None or price  < lowest_price:
                    promoted_vm = vm
                    new_vm_ic = new_ic
                    lowest_price = price
        if new_vm_ic is not None:
            new_vm = Vm(new_vm_ic)
            # Use max to round to zero and avoid values such as -0.00000000001
            new_vm.free_cores = max(ComputationalUnits("0 core"),
                                    promoted_vm.free_cores + new_vm.ic.cores - promoted_vm.ic.cores)
            new_vm.free_mem = max(Storage("0 mebibyte"),
                                  promoted_vm.free_mem + new_vm.ic.mem - promoted_vm.ic.mem)
            new_vm.cgs = promoted_vm.cgs
            new_vm.history = copy.deepcopy(promoted_vm.history)
            new_vm.history.append(promoted_vm.ic.name)
            index_promoted_vm = vms.index(promoted_vm)
            vms[index_promoted_vm] = new_vm
            return new_vm
        else:
            return None

    def __init__(self, ic: InstanceClass, ignore_ic_index: bool = False) -> Vm:
        """
        Create a virtual machine object of the given instance class.
        :param ic: Instance class.
        :param ignore_ic_index: When it is True the virtual machine creation do not affect virtual machine indexing.
        :return: The created virtual machine.
        """
        self.ic = ic
        # vm id is not set when generating a virtual machine for testing
        if not ignore_ic_index:
            if ic not in Vm._last_ic_index:
                Vm._last_ic_index[ic] = 0
            else:
                Vm._last_ic_index[ic] += 1
            self.id = Vm._last_ic_index[ic]  # A number for each virtual machine in the same instance class
        else:
            self.id = None
        self.free_cores = ic.cores  # Free cores
        self.free_mem = ic.mem  # Free memory
        self.cgs: list[ContainerGroup] = []  # The list of container groups allocated is empty
        self.history: list[str] = []  # Virtual machine history filled with vm promotions and additions

    def __str__(self) -> str:
        return f"{self.ic.name}[{self.id}]"

    def get_n_allocatable_cc(self, cc: ContainerClass) -> int:
        """
        Get the number of containers of the instance class that could be allocated.
        :param cc: Container class.
        :return: The number of containers that coul be allocated.
        """
        n_from_cpu = floor(self.free_cores / cc.cores)
        n_from_mem = 0
        mem_usage = 0
        while mem_usage <= self.free_mem:
            if cc.get_mem_from_aggregations(n_from_mem + 1) > self.free_mem:
                break
            n_from_mem += 1
            if n_from_mem >= n_from_cpu:
                break
        return min(n_from_cpu, n_from_mem)

    def get_container_groups(self, cc: ContainerClass) -> list[ContainerGroup]:
        """
        Get the container groups that contain the given instance class.
        :param cc: A container class.
        :return: A list of container groups.
        """
        res = []
        for cg in self.cgs:
            if cg.cc == cc:
                res.append(cg)
        return res

    def allocate(self, cc: ContainerClass, n_replicas: int) -> FcmaStatus:
        """
        Try to allocate the given number of replicas of the container class in the virtual machine.
        :param cc: Container class.
        :param n_replicas: The number of replicas of the container class to allocate.
        :return: The allocation status: ALLOCATION_OK, ALLOCATION_NOT_ENOUGH_MEM, or ALLOCATION_NOT_ENOUGH_CPU.
        """
        # Check if there are enough cores
        if cc.cores * n_replicas > self.free_cores + ComputationalUnits("0.000001 core"):
            return FcmaStatus.ALLOCATION_NOT_ENOUGH_CPU

        # Check if there are enough memory
        cg = self.get_container_groups(cc)
        if len(cg) == 0:
            prev_replicas = 0
        else:
            # Only one container group may contain the containers in the given container class
            prev_replicas = cg[0].replicas
        prev_replicas_mem = cc.get_mem_from_aggregations(prev_replicas)
        all_replicas_mem = cc.get_mem_from_aggregations(n_replicas + prev_replicas)
        mem_increment = all_replicas_mem - prev_replicas_mem
        if mem_increment >= self.free_mem + Storage("0.000001 gibibyte"):
            return FcmaStatus.ALLOCATION_NOT_ENOUGH_MEM

        # At this point allocation is possible, so perform allocation
        self.free_cores = max(ComputationalUnits("0 core"), self.free_cores - cc.cores * n_replicas)
        self.free_mem = max(Storage("0 mebibyte"), self.free_mem - mem_increment)
        if len(cg) > 0:
            # Add the replicas to the first container group including the container class
            cg[0].replicas += n_replicas
        else:
            # Create a new container group with the replicas
            cg = ContainerGroup(cc, n_replicas)
            self.cgs.append(cg)
        return FcmaStatus.ALLOCATION_OK

    def _cheapest_ic_promotion(self, cc: ContainerClass) -> InstanceClass:
        """
        Find the cheapest instance class in the same family with enough number cores and memory to
        allocate the currently allocated containers and one container of the given container class.
        :param cc: Container class that must fit in the instance class.
        :return: The cheapest instance class or None if there is no instance class able to allocate all the containers.
        """
        fm = self.ic.family
        cheapest_ic = None
        min_price = None
        for ic in fm.ics:
            if self.free_cores + ic.cores - self.ic.cores + ComputationalUnits("0.000001 core") >= cc.cores and \
                    self.free_mem + ic.mem - self.ic.mem + Storage("0.000001 gibibyte") >= cc.mem[0]:
                if cheapest_ic is None or ic.price < min_price:
                    cheapest_ic = ic
                    min_price = ic.price
        return cheapest_ic


@dataclass(frozen=True)
class FamilyClassAggPars:
    """
    Represents parameters for instance class aggregation in a family. Each instance class has an index.
    """
    # Instance class names sorted by increasing number of cores.
    ic_names: tuple[str, ...]
    # Instance class cores
    ic_cores: tuple[int]
    # Number of aggregation paths for every instance class name. The same length as ic_names
    n_agg: tuple[int, ...]
    # Number of nodes lost for every tuple (large ic index, aggregation path index, small ic index).
    # Instance class indexes come from the index in ic_names.
    p_agg: dict[tuple[int, int, int], int]


@dataclass(frozen=True)
class SolvingPars:
    """
    FCMA Solving parameters
    """
    # Speed levels are 1, 2 and 3. In general, the lowest speed level gives the lowest cost
    speed_level: int = 1
    #  Maximum number of available seconds to solve the partial ILP problem for
    #  speed level 1. Ignored for the other speed levels
    partial_ilp_max_seconds: [float] = None


@dataclass
class SolvingStats:
    """
    Represents the solving statistics of a solution. Some fields are valid for specific speed levels
    """
    solving_pars: SolvingPars = None  # Parameters of the solving algorithm

    partial_ilp_status: FcmaStatus = None  # Status of the partial ILP solution (speed_level=1)
    partial_ilp_seconds: float = None  # Time spent solving the partial ILP problem (speed_level=1)

    pre_allocation_cost: CurrencyPerTime = None  # Initial cost, before container allocation with promotion
    pre_allocation_seconds: float = None  # Time spent before container allocation with promotion
    pre_allocation_status: FcmaStatus = None  # Status before container allocation with promotion

    allocation_seconds: float = None  # Time spend in the container allocation with promotion

    final_status: FcmaStatus = None  # Final status
    final_cost: CurrencyPerTime = None  # Final cost
    total_seconds: float = None  # Total seconds = pre_allocation_seconds + allocation_seconds
