"""
Data classes for containers and nodes of the Fast Container and Machine Allocator (FCMA)
"""

from __future__ import annotations
from collections import defaultdict
import copy
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import chain
from math import floor
from functools import lru_cache
import pulp
from pulp import PULP_CBC_CMD
from cloudmodel.unified.units import (
    ComputationalUnits,
    CurrencyPerTime,
    RequestsPerTime,
    Storage,
    Quantity,
)

# Minimum difference so that two quantities are different
DELTA_VAL = 0.000001
# Maximum CPU cores difference for two "equal" CPU values.
DELTA_CPU = ComputationalUnits(f"{DELTA_VAL} core")
# Maximum memory gibibytes for two "equal" memory values
DELTA_MEM = Storage(f"{DELTA_VAL} gibibyte")


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
        return Storage("0 mebibytes")
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
    def get_worst_status(status_list: list[FcmaStatus]) -> FcmaStatus:
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
    sfmpl: float = 1.0  # Single failure maximum performance loss in (0, 1]

    def __post_init__(self):
        """
        Check the application parameters.
        :raise ValueError: When parameters are invalid.
        """

        if not isinstance(self.name, str):
            raise ValueError("App name must be a string")
        if (
            self.sfmpl != 1
            and not isinstance(self.sfmpl, float)
            or self.sfmpl < 0
            or self.sfmpl > 1.0
        ):
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
    aggs: tuple[int, ...] = (1,)
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
        """
        Check and set cores and performances.
        """

        try:
            value = Quantity(getattr(self, attr_name)).to(unit)
            object.__setattr__(self, attr_name, value)
        except Exception:
            raise ValueError(f"Invalid value of {error_name}") from None

        if value.magnitude <= 0.0:
            raise ValueError(f"{error_name} values must be positive")

    def _validate_and_set_aggs(self):
        """
        Check and set aggregations.
        """
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
                raise ValueError(
                    "Aggregations must be possitive integers sorted by increasing value"
                )
            prev_agg = agg_value

    def _validate_and_set_mem(self):
        """
        Check and set memory.
        """

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

        return InstanceClass(
            name,
            self.price * multiplier,
            self.cores * multiplier,
            self.mem * multiplier,
            self.family,
        )

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
        if (
            m.is_integer()
            and are_val_equal(m * self.mem, ic.mem)
            and are_val_equal(m * self.price, ic.price)
        ):
            return True
        return False

    @staticmethod
    def _get_parent_families(fm: InstanceClassFamily) -> list[InstanceClassFamily]:
        """
        Get a list with all the parent families of a given family.
        :param fm: A family to get parent families.
        :return: A list with all the parent families.
        """
        fms = copy.copy(fm.parent_fms)
        for parent_fm in fm.parent_fms:
            fms.extend(InstanceClass._get_parent_families(parent_fm))
        return list(set(fms))

    @lru_cache
    def get_promotion_ics(self, ic: InstanceClass) -> list[InstanceClass]:
        """
        Get a sorted list by increasing price with all the instance classes in the
        same family or parent families with a higher than or equal number of
        cores and a higher than or equal memory.
        """
        fms = [ic.family]
        fms.extend(InstanceClass._get_parent_families(self.family))
        ics = []
        for fm in fms:
            ics.extend(fm.ics)
        ics = list(set(ics))
        ics.remove(self)

        # Remove instance classes with less memory or cores than the current one
        ics = [ic for ic in ics if ic.cores >= self.cores and ic.mem >= self.mem]

        # Sort instance classes by increasing price
        ics.sort(key=lambda final_ic: final_ic.price)

        return ics


class InstanceClassFamily:
    """
    Family of instance classes. Performance is assumed to be the same for all the instance classes
    in a family, whenever instance classes have enough CPU and memory.
    """

    def __init__(
        self,
        name: str,
        parent_fms: None | InstanceClassFamily | list[InstanceClassFamily] = None,
    ) -> None:
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
        self.ics: list[InstanceClass] = []  # Instance classes in the family

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

    def add_parent_families(
        self, parent_fms: InstanceClassFamily | list[InstanceClassFamily]
    ) -> None:
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

    def get_parent_fm_in(self, fms: tuple[InstanceClassFamily]) -> InstanceClassFamily:
        """
        Get a parent family in the list of families.
        :param fms: Possible parent families.
        :return: The parent family in the list, or itself if there are no parent families
        """
        for fm in fms:
            if fm in self.parent_fms:
                return fm
        return self


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
    # Memory may be a tuple to provide a value for each aggregation
    mem: Storage | tuple[Storage, ...]
    perf: RequestsPerTime
    aggs: tuple[int]  # Container valid aggregations
    agg_level: int = 1  # Container current aggregation level

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
            agg_level=self.agg_level * replicas,
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

        aggs = list(self.aggs)  # List of aggregations, which include the 1
        aggs.sort(reverse=True)  # Aggregations are sorted by increasing values
        res = {}
        while replicas > 0:
            for agg in aggs:
                n = replicas // agg
                if n > 0:
                    res[agg] = n
                    replicas -= n * agg
                    if (
                        replicas == 0
                    ):  # It will end at some point since aggregations include value 1
                        break
        return res

    @lru_cache
    def get_mem_from_aggregations(self, replicas: int) -> Storage:
        """
        Get the memory that would require an aggregation made of several replicas.
        :param replicas: The number of container replicas.
        :return: The memory required if the replicas were aggregated.
        """

        # Firstly, get the aggregations for the given number of replicas
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
    _last_ic_index: dict[InstanceClass, int] = {}

    def __init__(self, ic: InstanceClass, ignore_ic_index: bool = False) -> None:
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
            self.id = Vm._last_ic_index[
                ic
            ]  # A number for each virtual machine in the same instance class
        else:
            self.id = -1
        self.free_cores = ic.cores  # Free cores
        self.free_mem = ic.mem  # Free memory
        self.cgs: list[ContainerGroup] = []  # The list of container groups allocated is empty
        # Virtual machine history filled with vm promotions and additions
        self.history: list[str] = []
        self.vm_before_promotion: Vm | None = None  # This VM before its last promotion
        # Containers allocated after the last promotion
        self.cc_after_promotion: dict[ContainerClass, int] | None = None

    @staticmethod
    def reset_ids():
        """
        Reset the instance class indexes, so the new virtual machine of each instance class will get index 1.
        """

        Vm._last_ic_index.clear()

    @staticmethod
    def promote_vm(vms: list[Vm], cc: ContainerClass) -> None | Vm:
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
                if lowest_price is None or price < lowest_price:
                    promoted_vm = vm
                    new_vm_ic = new_ic
                    lowest_price = price
        if new_vm_ic is not None:
            new_vm = Vm(new_vm_ic)
            new_vm.free_cores = delta_cpu_to_zero(
                promoted_vm.free_cores + new_vm.ic.cores - promoted_vm.ic.cores
            )
            new_vm.free_mem = delta_mem_to_zero(
                promoted_vm.free_mem + new_vm.ic.mem - promoted_vm.ic.mem
            )
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
        Return if the given number of replicas of the container class are allocatable
        in the virtual machine.
        :param cc: Container class.
        :param replicas: Number of replicas of the container class.
        :return: True if allocation is possible and False otherwise.
        """

        if self.free_cores + DELTA_CPU < replicas * cc.cores:
            return False
        cgs = self.get_container_groups(cc)
        if len(cgs) > 0:
            prev_replicas = cgs[0].replicas
        else:
            prev_replicas = 0
        mem_inc = cc.get_mem_from_aggregations(prev_replicas + replicas) - \
            cc.get_mem_from_aggregations(prev_replicas)
        if self.free_mem + DELTA_MEM < mem_inc:
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
        if n_from_cpu == 0:
            return 0

        # We assume that memory requirements per core decreases as aggregation increases, so the
        # greater the container aggregation, the lower the memory per core requested. Thus, the
        # maximum number of replicas comes from the highest aggregation levels.
        # For example, for agg = 1, 2, 4 and memory values 10, 11, 13, respectively, the maximum
        # comes from containers that will aggregate to level 4. If there is more memory free, then continue
        # with the previous aggregation level, i.e, 2, and so on.

        # Firstly, get the number of replicas of the cc previously allocated and the memory they request.
        cgs = self.get_container_groups(cc)
        if len(cgs) == 0:
            initial_replicas = 0
        else:
            initial_replicas = cgs[0].replicas
        free_mem = self.free_mem + cc.get_mem_from_aggregations(initial_replicas)

        agg_index = len(cc.mem) - 1
        total_replicas = 0  # Total number of cc replicas
        while agg_index >= 0:
            n_containers = floor((free_mem + DELTA_MEM) / cc.mem[agg_index])
            total_replicas += n_containers * cc.aggs[agg_index]
            free_mem -= n_containers * cc.mem[agg_index]
            if total_replicas - initial_replicas >= n_from_cpu:
                return n_from_cpu
            agg_index -= 1
        return total_replicas - initial_replicas

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

        # Because of container aggregation a lower number of containers may not be allocated while
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
        self.free_cores = max(
            ComputationalUnits("0 core"),
            self.free_cores - cc.cores * n_allocatable_replicas,
        )

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

    @lru_cache(maxsize=1024)
    def cheapest_ic_promotion(self, cc: ContainerClass) -> InstanceClass:
        """
        Find the cheapest instance class in the same family with enough number cores and memory to
        allocate the currently allocated containers and one container of the given container class.
        :param cc: Container class that must fit in the instance class.
        :return: The cheapest instance class or None if there is no instance class able to allocate all the containers.
        """

        ics = self.ic.get_promotion_ics(self.ic)

        for ic in ics:
            # Check if there are enough cores for 1 container
            free_cores = self.free_cores + ic.cores - self.ic.cores
            if free_cores + DELTA_CPU < cc.cores:
                continue

            # Check if there are enough memory for 1 container
            free_mem = self.free_mem + ic.mem - self.ic.mem
            replicas = 0  # Current replicas of the container in the vm
            for cg in self.cgs:
                if cg.cc == cc:
                    replicas = cg.replicas
                    break
            mem_inc = cc.get_mem_from_aggregations(replicas+1) - cc.get_mem_from_aggregations(replicas)
            if free_mem + DELTA_MEM > mem_inc:
                return ic
        return None


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
    FCMA Solving parameters.
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
    Represents the solving statistics of a solution. Some fields are valid for specific speed levels.
    """

    # pylint: disable=too-many-instance-attributes

    # Parameters of the solving algorithm
    solving_pars: None | SolvingPars = None

    # Status of the partial ILP solution (speed_level=1)
    partial_ilp_status: None | FcmaStatus = None

    # Time spent solving the partial ILP problem (speed_level=1)
    partial_ilp_seconds: None | float = None

    # Initial cost, before container allocation with promotion
    pre_allocation_cost: None | CurrencyPerTime = None

    # Time spent before container allocation with promotion
    pre_allocation_seconds: None | float = None

    # Status before container allocation with promotion
    pre_allocation_status: None | FcmaStatus = None

    # Time spend in the container allocation with promotion
    allocation_seconds: None | float = None

    # Final status
    final_status: None | FcmaStatus = None

    # Final cost
    final_cost: None | CurrencyPerTime = None

    # Total seconds = pre_allocation_seconds + allocation_seconds
    total_seconds: None | float = None


@dataclass(frozen=True)
class Solution:
    """
    Solution to the allocation problem.
    """

    # Virtual machines with their container allocations
    allocation: dict[InstanceClassFamily, Allocation]
    statistics: SolvingStats  # Solution statistics

    def is_infeasible(self) -> bool:
        """
        Return True if the solution is infeasible.
        """

        return self.statistics.final_status not in [
            FcmaStatus.OPTIMAL,
            FcmaStatus.FEASIBLE,
        ]


@dataclass(frozen=True)
class AllocationCheck:
    """
    Allocation summary in terms of node unused capacities and application surplus performances.
    """

    # pylint: disable=too-many-instance-attributes

    # Minimum unused CPU percentage evaluate among all the VMs
    min_unused_cpu_percentage: float

    # Maximum unused CPU percentage evaluate among all the VMs
    max_unused_cpu_percentage: float

    # Unused CPU percentage adding the CPU capacity of all the VMs
    global_unused_cpu_percentage: float

    # Minimum unused memory percentage evaluate among all the VMs
    min_unused_mem_percentage: float

    # Maximum unused memory percentage evaluate among all the VMs
    max_unused_mem_percentage: float

    # Unused memory percentage adding the CPU capacity of all the VMs
    global_unused_mem_percentage: float

    # Minimum surplus performace percentage evaluate among all the apps
    min_surplus_perf_percentage: float

    # Maximum surplus performace percentage evaluate among all the apps
    max_surplus_perf_percentage: float

    # Surplus performance adding the performance of all the apps
    global_surplus_perf_percentage: float


@dataclass(frozen=True)
class SingleVmSummary:
    ic_name: str
    total_num: int
    cost: CurrencyPerTime


@dataclass(frozen=True)
class AllVmSummary:
    vms: tuple[SingleVmSummary, ...]
    total_num: int
    total_cost: CurrencyPerTime


@dataclass(frozen=True)
class ContainerGroupSummary:
    container_name: str
    vm_name: str
    app_name: str
    performance: RequestsPerTime
    replicas: int
    cores: ComputationalUnits


@dataclass(frozen=True)
class AppSummary:
    app_name: str
    container_groups: tuple[ContainerGroupSummary, ...]
    total_replicas: int
    total_perf: RequestsPerTime


class SolutionSummary:
    """
    Allocation summary that can be used to generate printed output
    """

    def __init__(self, solution: Solution) -> None:
        self._solution = solution
        self.app_allocations = None
        self.vm_summary = None

    def get_vm_summary(self) -> AllVmSummary:
        if self.vm_summary is not None:
            return self.vm_summary
        if self.is_infeasible():
            self.vm_summary = AllVmSummary(
                vms=tuple(), total_cost=CurrencyPerTime("0 usd/hour"), total_num=0
            )
            return self.vm_summary
        num_vms = defaultdict(int)
        ic_prices = defaultdict(lambda: CurrencyPerTime("0 usd/hour"))
        total_num = 0
        total_price = CurrencyPerTime("0 usd/hour")
        for family, vm_list in self._solution.allocation.items():
            for vm in vm_list:
                ic_name = vm.ic.name
                num_vms[ic_name] += 1
                ic_prices[ic_name] += vm.ic.price
                total_num += 1
                total_price += vm.ic.price
        self.vm_summary = AllVmSummary(
            vms=tuple(
                SingleVmSummary(
                    ic_name=ic_name, total_num=num_vms[ic_name], cost=ic_prices[ic_name]
                )
                for ic_name in num_vms
            ),
            total_num=total_num,
            total_cost=total_price,
        )
        return self.vm_summary

    def get_app_allocation_summary(self, app_name) -> AppSummary:
        if not self.app_allocations:
            self.get_all_apps_allocations()
        return self.app_allocations[app_name]

    def get_all_apps_allocations(self) -> dict[str, AppSummary]:
        if self.app_allocations is not None:
            return self.app_allocations
        self.app_allocations = {}
        if self.is_infeasible():
            return self.app_allocations

        # First pass, group all container_groups of the same app
        app_info = defaultdict(list)
        alloc = self._solution.allocation.values()
        # Each element in alloc is a list of vms, so we can chain the iterables
        # to write a single loop
        for vm in chain.from_iterable(alloc):
            for container_group in vm.cgs:
                app_name = container_group.cc.app.name
                # Add containergroup info to this app
                app_info[app_name].append(
                    ContainerGroupSummary(
                        container_name=str(container_group.cc),
                        vm_name=str(vm),
                        app_name=container_group.cc.app.name,
                        performance=container_group.cc.perf,
                        replicas=container_group.replicas,
                        cores=container_group.cc.cores,
                    )
                )
        # Second pass, compute totals per app
        for app_name, containers in app_info.items():
            total_replicas = sum(c.replicas for c in containers)
            total_perf = sum(c.performance * c.replicas for c in containers)
            app_summary = AppSummary(
                app_name=app_name,
                container_groups=tuple(containers),
                total_replicas=total_replicas,
                total_perf=total_perf,
            )
            self.app_allocations[app_name] = app_summary
        return self.app_allocations

    def is_infeasible(self):
        return self._solution.is_infeasible()

    def as_dict(self, cost_unit="usd/h", perf_unit="req/s", cores_unit="mcores"):
        """Returns the summary as a dictionary suitable for json serializing

        In order to do that, units have to removed, so some default units are assumed
        for the json. By default, cost will be expressed in usd/h and requests in req/s
        """
        vms = asdict(self.get_vm_summary())
        vms["total_cost"] = vms["total_cost"].m_as(cost_unit)
        apps = {app: asdict(info) for app, info in self.get_all_apps_allocations().items()}

        # Remove units
        for vm in vms["vms"]:
            vm["cost"] = vm["cost"].m_as(cost_unit)
        for app, info in apps.items():
            info["total_perf"] = info["total_perf"].m_as(perf_unit)
            for cg in info["container_groups"]:
                cg["performance"] = cg["performance"].m_as(perf_unit)
                cg["cores"] = cg["cores"].m_as(cores_unit)
        return {
            "vms": vms,
            "apps": apps,
            "units": {"cost": cost_unit, "perf": perf_unit, "cores": cores_unit},
        }

    @classmethod
    def from_dict(cls, summary_dict):
        self = cls(solution=None)
        cost_unit = summary_dict["units"]["cost"]
        perf_unit = summary_dict["units"]["perf"]
        cores_unit = summary_dict["units"]["cores"]

        # Read vms summary
        vms = summary_dict["vms"]
        vms["total_cost"] = CurrencyPerTime(f"{vms['total_cost']} {cost_unit}")
        vms["vms"] = [
            SingleVmSummary(
                ic_name=vm["ic_name"],
                total_num=vm["total_num"],
                cost=CurrencyPerTime(f"{vm['cost']} {cost_unit}"),
            )
            for vm in vms["vms"]
        ]
        self.vm_summary = AllVmSummary(
            vms=tuple(vms["vms"]),
            total_num=vms["total_num"],
            total_cost=vms["total_cost"],
        )

        # Read apps summary
        apps = summary_dict["apps"]
        app_allocations = {}
        for app, info in apps.items():
            info["total_perf"] = RequestsPerTime(f"{info['total_perf']} {perf_unit}")
            info["container_groups"] = [
                ContainerGroupSummary(
                    container_name=cg["container_name"],
                    vm_name=cg["vm_name"],
                    app_name=cg["app_name"],
                    performance=RequestsPerTime(f"{cg['performance']} {perf_unit}"),
                    replicas=cg["replicas"],
                    cores=ComputationalUnits(f"{cg['cores']} {cores_unit}"),
                )
                for cg in info["container_groups"]
            ]
            app_allocations[app] = AppSummary(
                app_name=app,
                container_groups=tuple(info["container_groups"]),
                total_replicas=info["total_replicas"],
                total_perf=info["total_perf"],
            )
        self.app_allocations = app_allocations
        return self


# One system is defined by application performance parameters for pairs application and family
System = dict[tuple[App, InstanceClassFamily], AppFamilyPerf]

# List of virtual machines with container allocation
Allocation = list[Vm]
