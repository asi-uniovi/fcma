"""
Data classes for containers and nodes of Fast Container and Machine Allocator (FCMA)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum
import pulp
from math import floor
from cloudmodel.unified.units import (
    ComputationalUnits,
    Currency,
    CurrencyPerTime,
    Time,
    RequestsPerTime,
    Storage,
)


class FcmaStatus(Enum):
    """
    Status of FCMA solutions
    """
    # Pre-allocation status sorted from the best to the worst
    OPTIMAL = 1  # Optimal solution
    FEASIBLE = 2  # Feasible but not optimal. After a timeout
    INVALID = 3  # Invalid result

    # Allocation status
    ALLOCATION_OK = 10
    ALLOCATION_NOT_ENOUGH_MEM = 11  # Not enough memory for allocation
    ALLOCATION_NOT_ENOUGH_CPU = 12  # Not enough CPU for allocation

    @staticmethod
    def pulp_to_fcma_status(pulp_problem_status: int, pulp_solution_status: int) -> FcmaStatus:
        """
        Receives the PuLP status code for an ILP problem and its solution and returns a FCMA status
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
        Return the worst status in the list
        """
        return FcmaStatus(max(entry.value for entry in status_list))

    @staticmethod
    def is_valid(status: FcmaStatus | list[FcmaStatus]) -> bool:
        """
        Return True if the status is OPTIMAL or INTEGER_FEASIBLE
        """
        global_status = status
        if isinstance(status, list):
            # The global status is the worst status in the list
            global_status = FcmaStatus.get_worst_status(status)
        return global_status == FcmaStatus.OPTIMAL or global_status == FcmaStatus.FEASIBLE


@dataclass(frozen=True)
class App:
    """
    Application
    """
    name: str
    sfmpl: Optional[float] = 1.0  # Single failure maximum performance loss in (0, 1]

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class AppFamilyPerf:
    """
    Application computational parameters of the application container when it runs on a given instance class family
    """
    cores: ComputationalUnits
    perf: RequestsPerTime
    # Memory may be a list to provide a memory value for each aggregation in [1, max. agg. value]
    mem: Storage | tuple[Storage, ...]
    # Maximum aggregation value that preserves the performance. Any aggregation that provides one application
    # container with n*cores allocated provides at least a performance n*perf with n <= maxagg
    agg: Optional[tuple[int]] = 1,
    maxagg: Optional[int] = 1

    def __post_init__(self):
        """
        Updates aggreagion parameters, checks dimensions are valid and store them in the standard units
        """
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
            if len(self.mem) != self.maxagg:
                raise ValueError(f"Invalid number of memory items in computational parameters")
            new_mem = tuple(mem.to("gibibytes") for mem in self.mem)
            object.__setattr__(self, "mem", new_mem)
        object.__setattr__(self, "perf", self.perf.to("req/hour"))


@dataclass
class InstanceClass:
    """
    Instance class, i.e., a type of virtual machine in a region
    """

    name: str
    price: CurrencyPerTime
    cores: ComputationalUnits
    mem: Storage
    family: Optional[InstanceClassFamily] = None

    def __post_init__(self):
        """
        Checks dimensions are valid and store them in the standard units
        """
        object.__setattr__(self, "price", self.price.to("usd/hour"))
        object.__setattr__(self, "cores", self.cores.to("cores"))
        object.__setattr__(self, "mem", self.mem.to("gibibytes"))
        if object.__getattribute__(self, "family") is None:
            object.__setattr__(self, "family", self.name)

    def __mul__(self, multiplier: int) -> InstanceClass:
        """
        Multiplies one instance class by a scalar giving an instance class in the same family,
        with price, cores and memory multiplied by that scalar
        """
        return InstanceClass(f'{multiplier}x{self.name}', self.price*multiplier,
                             self.cores*multiplier, self.mem*multiplier, self.family)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(repr(self))

    def set_name(self, name) -> InstanceClass:
        """
        Sets the instance class name and returns it
        """
        self.name = name
        return self

    def is_multiple(self, ic: InstanceClass) -> bool:
        """
        Returns True if the instance class is in the same family and is multiple
        in terms of price, CPU and memory
        """
        if ic.family != self.family:
            return False
        m = ic.cores/self.cores
        if m.is_integer() and abs(1 - m*self.mem/ic.mem) < 0.000001 and abs(1 - m*self.price/ic.price) < 0.000001:
            return True
        return False

    def is_cpu_promoted(self, ic: InstanceClass) -> bool:
        """
        Returns True if the instance class is in the same family, have the same amount of memory,
        but has more cores (and so it is more expensive)
        """
        if ic.family == self.family and ic.cores > self.cores and abs(1 - self.mem/ic.mem) < 0.000001:
            return True
        return False

    def is_mem_promoted(self, ic: InstanceClass) -> bool:
        """
        Returns True if the instance class is in the same family, have the same number of cores,
        but has more memory (and so it is more expensive)
        """
        if ic.family == self.family and ic.cores == self.cores and abs(1 - self.mem/ic.mem) > 0.000001:
            return True
        return False

    def is_promoted(self, ic: InstanceClass) -> bool:
        """
        Returns True if the instance class is in the same family and is CPU promoted or memory promoted
        """
        return self.is_mem_promoted(ic) or self.is_cpu_promoted(ic)


class InstanceClassFamily:
    """
    Family of instance classes
    """
    def __init__(self, name: str, fms: list[InstanceClassFamily] = None):
        """
        Creates an instance class family object. The family may be the parent of one or more families given
        by fms parameter. In that case, it will include the instance classes of these families
        """
        self.parent = None  # Parent families of this family
        self.name = name
        self.ics = [] # Instance classes in the family
        if fms is None or len(fms) == 0:
            self.ics = []
        else:
            for fm in fms:
                fm.parent = self
                self.ics.extend(fm.ics)
                self.ics = list(set(self.ics))

    def __str__(self):
        return self.name

    def add_ics(self, ics: list[InstanceClass]):
        """
        Add instance classes to the family and its parent family
        """
        for ic in ics:
            if ic not in self.ics:
                self.ics.append(ic)
                ic.family = self
            if self.parent is not None:
                self.parent.add_ics([ic])
                ic.family = self.parent


@dataclass
class ContainerClass:
    """
    Represents a container class, i.e., a type of container running an application
    with some computational resources
    """
    app: App
    ic: InstanceClass  # Instance class is None when the container is not allocated
    fm: InstanceClassFamily  # Instance class family
    cores: ComputationalUnits
    # Memory may be a list to provide a memory value for each aggregation in [1, maxagg]
    mem: Storage | tuple[Storage, ...]
    perf: RequestsPerTime
    # Maximum aggregation value that preserves the performance. Any aggregation that provides one application
    # container with n*cores allocated provides at least a performance n*perf with n <= maxagg
    aggs: tuple[int]

    def __post_init__(self):

        """
        Checks dimensions are valid and store them in the standard units
        """
        object.__setattr__(self, "cores", self.cores.to("cores"))
        if not isinstance(self.mem, tuple):
            std_mem = tuple(self.mem.to("gibibytes") for _ in range(self.aggs[-1]))
        else:
            if len(self.mem) != self.aggs[-1]:
                # A number of memory values different to maxagg and higher than 1 is invalid
                if len(self.mem) != 1:
                    raise ValueError(f"Invalid number of memory items in computational parameters")
                # A single memory value is assumed to be independent of aggregation size until maxagg
                else:
                    std_mem = tuple(self.mem[0].to("gibibytes") for _ in range(self.aggs[-1]))
            else:
                std_mem = tuple(mem.to("gibibytes") for mem in self.mem)
        object.__setattr__(self, "mem", std_mem)
        object.__setattr__(self, "perf", self.perf.to("req/hour"))

    def __str__(self):
        if self.ic is None:
            return f"{self.app.name}-{self.fm.name}"
        else:
            return f"{self.app.name}-{self.ic.name}"

    def __hash__(self):
        return hash(repr(self))

    def __mul__(self, replicas: int) -> ContainerClass:
        """
        Returns a new container after aggregating replicas
        """
        container = ContainerClass(
            app=self.app,
            ic=self.ic,
            fm=self.fm,
            cores=self.cores*replicas,
            mem=self.mem[replicas-1],
            perf=self.perf * replicas,
            aggs=(1,)
        )
        return container

    def aggregations(self, replicas: int) -> dict[int, int]:
        """
        For a given number of replicas returns a dictionary with the aggregation sizes and their number.
        For example, 10 replicas may be grouped in 3 groups of 2 containers and 1 group of 4 containers,
        returning {2: 3, 4: 1}
        """
        aggs = reversed(self.aggs)
        res = {}
        while replicas > 0:
            for agg in aggs:
                n = replicas // agg
                if n > 0:
                    res[agg] = n
                replicas -= n * agg
                if replicas == 0:
                    break
        return res

    def get_mem_from_aggregations(self, replicas: int) -> Storage:
        """
        Returns the memory required by the replicas once being aggregated
        """
        n_aggs = self.aggregations(replicas)
        mem = Storage("0 gibibytes")
        for agg in n_aggs:
            agg_index = self.aggs.index(agg)
            mem += self.mem[agg_index] * n_aggs[agg]

        return mem


@dataclass
class ContainerGroup:
    """
    Represents a group of identical containers for the same application
    """
    cc: ContainerClass
    replicas: int


class Vm:
    """
    Represents a virtual machine
    """
    # Last identifier for each instance class virtual machine
    last_ids = {}

    @staticmethod
    def reset_ids():
        Vm.last_ids.clear()

    @staticmethod
    def cheapest_addition(fm: InstanceClassFamily, cc: ContainerClass):
        """
        Returns the cheapest instance class in the familiy that can allocate at least one container of
        the given container class
        """
        cheapest_ic = None
        min_price = None
        for ic in fm.ics:
            if ic.cores >= cc.cores and ic.mem >= cc.mem[0]:
                if cheapest_ic is None or ic.price < min_price:
                    cheapest_ic = ic
                    min_price = ic.price
        return cheapest_ic

    @staticmethod
    def promote_vm(vms: list[Vm], cc: ContainerClass) -> Vm:
        """
        Promote one virtual machine in the list of virtual machines to be able to allocate at least
        one container of the given container class at the lowes cost.
        The promoted vm is replaced by the new one in the list of virtual machines.
        Returns the new virtual machine if promotion is feasible or None otherwise.
        """
        promoted_vm = None  # vm for a promotion
        new_vm_ic = None  # New instance class for a promoted vm
        lowest_price = None  # Lowest price coming from a vm promotion
        for vm in vms:
            # Note that promotion of the biggest ic in the family is not possible, returning None
            new_ic = vm.cheapest_promotion(cc)
            if new_ic is not None:
                if lowest_price is None or new_ic.price < lowest_price:
                    promoted_vm = vm
                    new_vm_ic = new_ic
                    lowest_price = new_ic.price - vm.ic.price
        if new_vm_ic is not None:
            new_vm = Vm(new_vm_ic)
            new_vm.free_cores = promoted_vm.free_cores + new_vm.ic.cores - promoted_vm.ic.cores
            new_vm.free_mem = promoted_vm.free_mem + new_vm.ic.mem - promoted_vm.ic.mem
            new_vm.cgs = promoted_vm.cgs
            new_vm.history = copy.deepcopy(promoted_vm.history)
            new_vm.history.append(promoted_vm.ic.name)
            vms.remove(promoted_vm)
            vms.append(new_vm)
            return new_vm
        else:
            return None

    def __init__(self, ic: InstanceClass, test:bool = False):
        self.ic = ic
        # vm id is not set when generating a virtual machine for testing
        if not test:
            if ic not in Vm.last_ids:
                Vm.last_ids[ic] = 1
            else:
                Vm.last_ids[ic] += 1
            self.id = Vm.last_ids[ic]  # A number for each virtual machine in the same instance class
        else:
            self.id = None
        self.free_cores = ic.cores  # Free cores
        self.free_mem = ic.mem  # Free memory
        self.cgs: list[ContainerGroup] = []  # Container groups allocated
        self.history: list[str] = [] # Virtual machine history in promotions and additions

    def __str__(self) -> str:
        return f"{self.ic.name}-{self.id}"

    def get_allocatable_number(self, cc: ContainerClass) -> int:
        """
        Returns the number of containers of instance class cc than could be allocated
        """
        maximum_from_cpu = floor(self.free_cores / cc.cores)
        maximum_from_mem = 0
        mem_usage = 0
        while mem_usage <= self.free_mem:
            if cc.get_mem_from_aggregations(maximum_from_mem + 1) > self.free_mem:
                break
            maximum_from_mem += 1
            if maximum_from_mem >= maximum_from_cpu:
                break
        return max(maximum_from_cpu, maximum_from_mem)

    def get_container_group(self, cc: ContainerClass) -> ContainerGroup:
        """
        Returns the associated container group
        """
        for cg in self.cgs:
            if cg.cc == cc:
                return cg
        return None

    def allocate(self, cc: ContainerClass, n_replicas: int) -> FcmaStatus:
        """
        Try to allocate n_replicas of the container class
        """
        # Check if there are enough cores
        if cc.cores * n_replicas > self.free_cores + ComputationalUnits("0.000001 core"):
            return FcmaStatus.ALLOCATION_NOT_ENOUGH_CPU
        # Check if there are enough memory
        cg = self.get_container_group(cc)
        if cg is None:
            prev_replicas = 0
        else:
            prev_replicas = cg.replicas
        mem_increment = cc.get_mem_from_aggregations(n_replicas + prev_replicas) - \
                        cc.get_mem_from_aggregations(prev_replicas)
        if mem_increment >= self.free_mem + Storage("0.000001 gibibyte"):
            return FcmaStatus.ALLOCATION_NOT_ENOUGH_MEM

        self.free_cores -= cc.cores * n_replicas
        if self.free_cores.magnitude < 0:
            self.free_cores = ComputationalUnits("0 core")
        self.free_mem -= mem_increment
        if self.free_mem < 0:
            self.free_mem = Storage("0 gibibyte")
        if cg is not None:
            cg.replicas += n_replicas
        else:
            cg = ContainerGroup(cc, n_replicas)
            self.cgs.append(cg)
        return FcmaStatus.ALLOCATION_OK

    def cheapest_promotion(self, cc: ContainerClass) -> InstanceClass:
        """
        Finds the cheapest instance class in the family with enough number cores and memory
        and promote the current virtual machine to that instance class
        """
        fm = self.ic.family
        cheapest_ic = None
        min_price = None
        for ic in fm.ics:
            if self.free_cores + ic.cores - self.ic.cores + ComputationalUnits("0 core") >= cc.cores and \
                    self.free_mem + ic.mem - self.ic.mem + Storage("0 gibibyte") >= cc.mem[0]:
                if cheapest_ic is None or ic.price < min_price:
                    cheapest_ic = ic
                    min_price = ic.price
        return cheapest_ic


@dataclass
class FamilyClassAggPars:
    """
    Represents parameters for instance class aggregation
    """
    # Instance class names
    ic_names: tuple[str, ...]
    # Number of aggregation paths for every instance class in ic_names
    n_agg: tuple[int]
    # Number of nodes lost for every tuple (target ic index, aggregation path index, source ic index)
    p_agg: dict[tuple[int, int, int], int]


@dataclass(frozen=True)
class SolvingPars:
    """
    Solving parameters
    """
    speed_level: int = 1
    partial_ilp_max_seconds: Optional[float] = None


@dataclass
class SolvingStats:
    """
    Represents the solving statistics of a solution. Some fields are valid for specific speed levels
    """
    solving_pars: SolvingPars = None # Parameters of the solving algorithm

    partial_ilp_status: FcmaStatus = None  # Status of the partial ILP solution (speed_level=1)
    partial_ilp_seconds: float = None  # Time spent solving the partial ILP problem (speed_level=1)

    before_allocation_cost: CurrencyPerTime = None  # Cost before container allocation
    before_allocation_seconds: float = None  # Time spent before container allocation
    before_allocation_status: list[FcmaStatus] = None  # Status before container allocation, one per family

    allocation_seconds: float = None # FCMA Container allocation time

    final_status: FcmaStatus = None  # Status after container allocation
    final_cost: CurrencyPerTime = None  # Cost after container allocation
    total_seconds: float = None  # Total time = pre_allocation_time + allocation_time
