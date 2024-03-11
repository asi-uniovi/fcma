"""
Data classes for containers and nodes of the Fast Container and Machine Allocator (FCMA)
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from enum import Enum
from math import floor
import pulp
from pulp import PULP_CBC_CMD
from cloudmodel.unified.units import ComputationalUnits, CurrencyPerTime, RequestsPerTime, Storage, Quantity

DELTA_VAL = 0.000001  # Minimum difference so that two quantities are different
DELTA_CPU = ComputationalUnits(f"{DELTA_VAL} core")  # Maximum CPU cores difference for two "equal" CPU values.
DELTA_MEM = Storage(f"{DELTA_VAL} gibibyte")  # Maximum memory gibybytes for two "equal" memory values


def are_val_equal(val1: Quantity, val2: Quantity) -> bool:
    """
    Compare two quantities for equality.
    :param val1: First value.
    :param val2: Second value.
    :return: True if both quantities are approximately equal and False otherwise.
    """
    return abs((val1 - val2).magnitude) < DELTA_VAL


def delta_to_zero(val: float) -> float:
    """
    Round to zero dimentionless values close to zero.
    :param val: dimentionless value to round.
    :return: The value or zero, depending on its closeness to zero.
    """
    if abs(val) < DELTA_VAL:
        return 0.0
    return val


def delta_cpu_to_zero(val_cpu: ComputationalUnits) -> ComputationalUnits:
    """
    Round to zero computational values close to zero.
    :param val_cpu: CPU value to round.
    :return: The value or zero, depending on its closeness to zero.
    """
    if are_val_equal(val_cpu, DELTA_CPU):
        return ComputationalUnits("0 core")
    return val_cpu


def delta_mem_to_zero(val_mem: Storage) -> Storage:
    """
    Round to zero memory values close to zero.
    :param val_mem: Memory value to round.
    :return: The value or zero, depending on its closeness to zero.
    """
    if are_val_equal(val_mem, DELTA_MEM):
        return ComputationalUnits("0 core")
    return val_mem


class FcmaStatus(Enum):
    """
    Status of FCMA solutions.
    """
    # Pre-allocation status sorted from the best to the worst
    OPTIMAL = 1  # Optimal solution
    FEASIBLE = 2  # Feasible but not optimal. After a timeout
    INVALID = 3  # Invalid result

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
        return global_status in (FcmaStatus.OPTIMAL, FcmaStatus.FEASIBLE)


@dataclass(frozen=True)
class App:
    """
    FCMA application.
    """
    name: str
    sfmpl: [float] = 1.0  # Single failure maximum performance loss in (0, 1]

    def __post_init__(self):
        """
        Check the application parameters.
        :raise ValueError: When parameters are invalid.
        """
        if not isinstance(self.name, str):
            raise ValueError("App name must be a string")
        if self.sfmpl != 1 and not isinstance(self.sfmpl, float) or self.sfmpl < 0 or self.sfmpl > 1.0:
            raise ValueError("App's SFMPL must be a float in range (0, 1]")

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
    aggs: tuple[int] = (1,)
    # Maximum aggregation value that preserves the performance. An n-container aggregation generates 1 container
    # with (n x cores) and at least (n x perf), with n in [1, maxagg]
    maxagg: int = 1

    def __post_init__(self):
        """
        Updates aggregation parameters, checks dimensions are valid and store them in the standard units.
        :raise ValueError: When parameters are invalid.
        """

        self._validate_and_set("cores", "cores", "cores")
        self._validate_and_set("perf", "req/hour", "performance")
        self._validate_and_set_aggs()
        self._validate_and_set_mem()

    def _validate_and_set(self, attr_name, unit, error_name):
        """ Check and set cores and performances """
        try:
            value = Quantity(getattr(self, attr_name)).to(unit)
            object.__setattr__(self, attr_name, value)
        except Exception:
            raise ValueError(f"Invalid value of {error_name}") from None

        if value.magnitude <= 0.0:
            raise ValueError(f"{error_name} values must be positive")

    def _validate_and_set_aggs(self):
        """ Check and set aggregations """
        if not isinstance(self.aggs, tuple):
            raise ValueError("Aggregations must be expressed as a tuple")
        # Aggregation value 1 is not really an aggregation, but helps programming
        if 1 not in self.aggs:
            new_aggs = (1,) + self.aggs
            object.__setattr__(self, "aggs", new_aggs)
        object.__setattr__(self, "maxagg", self.aggs[-1])
        prev_agg = 0
        for agg_value in self.aggs:
            if not isinstance(agg_value, int) or agg_value <= prev_agg:
                raise ValueError("Aggregations must be possitive integers sorted by increasing value")
            prev_agg = agg_value

    def _validate_and_set_mem(self):
        """ Check and set memory """
        if not isinstance(self.mem, tuple):
            new_mem = tuple(self.mem for _ in range(len(self.aggs)))
        else:
            if len(self.mem) != len(self.aggs):
                raise ValueError("Invalid number of memory items")
            new_mem = self.mem
        for mem_value in new_mem:
            try:
                mem_value = mem_value.to("gibibytes")
            except Exception as _:
                raise ValueError("Invalid value of memory") from None
        prev_mem_agg = new_mem[0] / self.aggs[0]
        for index in range(1, len(new_mem)):
            if new_mem[index].magnitude <= 0.0:
                raise ValueError("Memory values must be possitive")
            mem_agg = new_mem[index] / self.aggs[index]
            if mem_agg > prev_mem_agg:
                raise ValueError("Memory requirements must decrease with higher aggregation values")
            prev_mem_agg = mem_agg
        object.__setattr__(self, "mem", new_mem)


@dataclass(frozen=True)
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

    def mul(self, multiplier: int, name: str) -> InstanceClass:
        """
        Multiplies the instance class by a scalar giving an instance class in the same family,
        with price, cores and memory multiplied by that scalar.
        :param multiplier: Instance class mutiplier.
        :param name: Name for the new instance class.
        :return: The new instance class.
        """
        return InstanceClass(name, self.price * multiplier, self.cores * multiplier,
                             self.mem * multiplier, self.family)

    def __str__(self) -> str:
        return self.name

    def is_multiple(self, ic: InstanceClass) -> bool:
        """
        Check if the given instance class is in the same family and is multiple in terms of price, CPU and memory.
        :param ic: the instance class to check.
        :return: True when it is in the same family and is multiple in terms of price, CPU and memory.
        """
        if ic.family != self.family:
            return False
        m = ic.cores / self.cores
        if m.is_integer() and are_val_equal(m * self.mem, ic.mem) and are_val_equal(m * self.price, ic.price):
            return True
        return False

    def _is_cpu_promoted(self, ic: InstanceClass) -> bool:
        """
        Check if the given instance class is in the same family, have the same amount of memory,
        but has more cores (and so it is more expensive).
        :param ic: The instance class to compare with.
        :return: True when the given instance class is CPU promoted.
        """
        if ic.family == self.family and ic.cores > self.cores + DELTA_CPU and are_val_equal(self.mem, ic.mem):
            return True
        return False

    def _is_mem_promoted(self, ic: InstanceClass) -> bool:
        """
        Check if the instance class is in the same family, have the same number of cores,
        but has more memory (and so it is more expensive).
        :param ic: The instance class to compare with.
        :return: True when the given instance class is memory promoted.
        """
        if ic.family == self.family and are_val_equal(ic.cores, self.cores) and ic.mem > self.mem + DELTA_MEM:
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

    def check_fesibility(self, cores: ComputationalUnits, mem: tuple[Storage]) -> bool:
        """
        Check if an instance class exists with enough CPU and memory.
        :param cores: The number of required cores.
        :param mem: The required memory.
        :return: True is an instance class exist with enough CPU and memory
        """
        for ic in self.ics:
            if ic.cores < cores:
                continue
            for mem_val in mem:
                if ic.mem < mem_val:
                    continue
            return True
        return False


@dataclass(frozen=True)
class ContainerClass:
    """
    Represents a container class, i.e., a type of container running an application with some computational resources.
    """
    # pylint: disable=too-many-instance-attributes
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
            std_mem = tuple(self.mem.to("gibibytes") for _ in range(len(self.aggs)))
        else:
            if len(self.mem) != len(self.aggs):
                if len(self.mem) != 1:
                    raise ValueError("Invalid number of memory items in computational parameters")
                # A single memory value is assumed to be independent of the aggregation size
                std_mem = tuple(self.mem[0].to("gibibytes") for _ in range(len(self.aggs)))
            else:
                std_mem = tuple(mem.to("gibibytes") for mem in self.mem)
        object.__setattr__(self, "mem", std_mem)
        object.__setattr__(self, "perf", self.perf.to("req/hour"))

    def __str__(self) -> str:
        if self.ic is None:
            return f"{self.app.name}-{self.fm.name}"
        return f"{self.app.name}-{self.ic.name}"

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

    def get_aggregations(self, replicas: int) -> dict[int, int]:
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
        n_aggs = self.get_aggregations(replicas)
        # Add the memory required by all the aggregations
        mem = Storage("0 gibibytes")
        for agg, n_agg in zip(self.aggs, n_aggs.values()):
            agg_index = self.aggs.index(agg)
            mem += self.mem[agg_index] * n_agg

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
    # pylint: disable=too-many-instance-attributes

    # Virtual machines in the same instance class get an increasing index
    _last_ic_index = {}

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
        self.vm_before_promotion: Vm = None  # This VM before its last promotion
        self.cc_after_promotion: dict[ContainerClass, int] = None  # Containers allocated after the last promotion

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
            new_ic = vm.cheapest_ic_promotion(cc)
            if new_ic is not None:
                price = new_ic.price - vm.ic.price  # Price of promotion vm.ic -> new_ic
                if lowest_price is None or price  < lowest_price:
                    promoted_vm = vm
                    new_vm_ic = new_ic
                    lowest_price = price
        if new_vm_ic is not None:
            new_vm = Vm(new_vm_ic)
            new_vm.free_cores = delta_cpu_to_zero(promoted_vm.free_cores + new_vm.ic.cores - promoted_vm.ic.cores)
            new_vm.free_mem = delta_mem_to_zero(promoted_vm.free_mem + new_vm.ic.mem - promoted_vm.ic.mem)
            new_vm.cgs = promoted_vm.cgs
            promoted_vm.cgs = copy.deepcopy(promoted_vm.cgs)
            new_vm.history = promoted_vm.history
            promoted_vm.history = copy.deepcopy(promoted_vm.history)
            new_vm.history.append(promoted_vm.ic.name)
            promoted_vm.vm_before_promotion = None
            promoted_vm.cc_after_promotion = None
            new_vm.vm_before_promotion = promoted_vm
            new_vm.cc_after_promotion = {}
            index_promoted_vm = vms.index(promoted_vm)
            vms[index_promoted_vm] = new_vm
            return new_vm
        return None

    def __str__(self) -> str:
        return f"{self.ic.name}[{self.id}]"

    def is_allocatable_cc(self, cc: ContainerClass, replicas: int) -> bool:
        """
        Return if the given number of replicas of the container class ar allocatable
        in the viruak machine.
        :param cc: Container class.
        :param replicas: Number of replicas of the container class.
        :return: True if allocation is possible and False otherwise.
        """
        if self.free_cores + DELTA_CPU < replicas * cc.cores:
            return False
        if self.free_mem + DELTA_MEM < cc.get_mem_from_aggregations(replicas):
            return False
        return True

    def get_max_allocatable_cc(self, cc: ContainerClass) -> int:
        """
        Get the maximum number of containers of the instance class that could be allocated
        in the virtual machine.
        :param cc: Container class.
        :return: The number of replicas that could be allocated.
        """

        # Get the number of replicas considering only CPU requirements
        n_from_cpu = floor((self.free_cores + DELTA_CPU) / cc.cores)

        # We assume that memory requirements per core decreases as aggregation increases, so the
        # greater the container aggregation, the lower the memory requested.
        # The maximum number of replicas comes from the maximum number of replicas with the highest
        # aggregation levels.
        # For example, for agg = 1, 2, 4 and memory values 10, 11, 13, respectively, the maximum
        # comes from container with aggregation level 4. If there is more memory free then continue
        # with the previous aggregation level, i.e, 2, and so on
        free_mem = self.free_mem
        n_from_mem = 0
        last_agg_index = len(cc.mem) - 1
        while free_mem + DELTA_MEM >= cc.mem[last_agg_index]:
            n_add_replicas = int(floor((free_mem + DELTA_MEM) / cc.mem[last_agg_index]))
            n_from_mem += n_add_replicas * cc.aggs[last_agg_index]
            if n_from_mem > n_from_cpu:
                return n_from_cpu
            free_mem -= n_add_replicas * cc.mem[last_agg_index]
            last_agg_index -= 1

        return n_from_mem

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

    def allocate(self, cc: ContainerClass, n_replicas: int) -> int:
        """
        Try to allocate the given number of replicas of the container class in the virtual machine.
        :param cc: Container class.
        :param n_replicas: The number of replicas of the container class to allocate.
        :return: The number of replicas that have been allocated.
        """
        # Beceause of container aggregation a lower number of containers may not be allocated while
        # a higher number of containers could be. This paradox comes from the reduced memory requirements
        # of aggregated containers.
        n_allocatable_replicas = n_replicas
        if not self.is_allocatable_cc(cc, n_replicas):
            max_allocatable_replicas = self.get_max_allocatable_cc(cc)
            if max_allocatable_replicas < n_replicas:
                n_allocatable_replicas = max_allocatable_replicas
                if n_allocatable_replicas == 0:
                    return 0
            else:
                n_allocatable_replicas -= 1
                while not self.is_allocatable_cc(cc, n_allocatable_replicas):
                    n_allocatable_replicas -= 1

        # At this point allocate n_allocatable_replicas

        # Update the free cores
        self.free_cores = max(ComputationalUnits("0 core"), self.free_cores - cc.cores * n_allocatable_replicas)

        # Update the free memory.
        # The previous number of total instances of de container class is needed to calculate the new free memory
        cg = self.get_container_groups(cc)
        if len(cg) == 0:
            prev_replicas = 0
        else:
            # Only one container group may contain the containers in the given container class
            prev_replicas = cg[0].replicas
        current_replicas = prev_replicas + n_allocatable_replicas
        prev_replicas_mem = cc.get_mem_from_aggregations(prev_replicas)
        current_replicas_mem = cc.get_mem_from_aggregations(current_replicas)
        mem_increment = current_replicas_mem - prev_replicas_mem
        self.free_mem = max(Storage("0 mebibyte"), self.free_mem - mem_increment)

        #  Update the container group
        if len(cg) > 0:
            # Add the replicas to the container group including the container class
            cg[0].replicas += n_allocatable_replicas
        else:
            # Create a new container group with the replicas
            cg = ContainerGroup(cc, n_allocatable_replicas)
            self.cgs.append(cg)

        # Update the containers after a promotion
        if self.cc_after_promotion is not None:
            if cc not in self.cc_after_promotion:
                self.cc_after_promotion[cc] = n_allocatable_replicas
            else:
                self.cc_after_promotion[cc] += n_allocatable_replicas

        return n_allocatable_replicas

    def cheapest_ic_promotion(self, cc: ContainerClass) -> InstanceClass:
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
            if self.free_cores + ic.cores - self.ic.cores + DELTA_CPU >= cc.cores and \
                    self.free_mem + ic.mem - self.ic.mem + DELTA_MEM >= cc.mem[0]:
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
    # Configure the solver. When it is set to None it uses CBC as solver with default parameters
    solver: any = PULP_CBC_CMD(msg=0)

    def __post_init__(self):
        if self.speed_level == 3:
            object.__setattr__(self, "solver", None)


@dataclass
class SolvingStats:
    """
    Represents the solving statistics of a solution. Some fields are valid for specific speed levels
    """
    # pylint: disable=too-many-instance-attributes
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


@dataclass(frozen=True)
class Solution:
    """
    Solution to the allocation problem
    """
    allocation: Allocation  # List of virtual machines with their container allocations
    statistics: SolvingStats  # Solution statistics


@dataclass(frozen=True)
class AllocationCheck:
    """
    Allocation summary in terms of node unused capacities and application surplus performances
    """
    # pylint: disable=too-many-instance-attributes
    min_unused_cpu_percentage: float  # Minimum unused CPU percentage evaluate among all the VMs
    max_unused_cpu_percentage: float  # Maximum unused CPU percentage evaluate among all the VMs
    global_unused_cpu_percentage: float  # Unused CPU percentage adding the CPU capacity of all the VMs
    min_unused_mem_percentage: float  # Minimum unused memory percentage evaluate among all the VMs
    max_unused_mem_percentage: float  # Maximum unused memory percentage evaluate among all the VMs
    global_unused_mem_percentage: float  # Unused memory percentage adding the CPU capacity of all the VMs
    min_surplus_perf_percentage: float  # Minimum surplus performace percentage evaluate among all the apps
    max_surplus_perf_percentage: float  # Maximum surplus performace percentage evaluate among all the apps
    global_surplus_perf_percentage: float  # Surplus performance adding the performance of all the apps


# One system is defined by application performance parameters for pairs application and family
System = dict[tuple[App, InstanceClassFamily], AppFamilyPerf]

# List of virtual machines with container allocation
Allocation = list[Vm]
