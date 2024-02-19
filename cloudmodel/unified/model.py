# coding: utf-8
"""
This module implements the base dataclasses which define a problem to be solved by Malloovia, and
the solution
"""

from dataclasses import dataclass
from typing import Tuple
from ..util import simplified_repr
from .. import __version__
from .units import *


@simplified_repr("name", show_field_names=False)
@dataclass(frozen=True)
class App:
    """App identifier.

    Attributes:
      - name:str: name of the application
    """

    name: str = "unnamed"
    max_resp_time: Time = UNLIMITED_TIME


@simplified_repr("name", show_field_names=False)
@dataclass(frozen=True)
class Region:
    """Region identifier.

    Attributes:
      - name:str: name of the region
    """

    name: str = "unnamed region"


@simplified_repr("description", "time_slot_size")
@dataclass(frozen=True)
class WorkloadSeries:
    """Workload as a sequence for different timeslots

    Attributes:
      - name:str: name of this workload
      - values:tuple[Requests,...]: sequence of workloads for each timeslot as the average number of
            requests arriving globally at the timeslot It can be a tuple with a single element, for
            a single timeslot
      - time_slot_size:Time: duration of the timeslot
      - intra_slot_distribution:str: name of the distribution for the interarrival times of the
            requests ("exponential" by default)
    """

    description: str
    values: Tuple[Requests, ...]
    time_slot_size: Time
    intra_slot_distribution: str = "exponential"

    # def __post_init__(self):
    #     """Checks dimensions of the time_slot_size are valid."""
    #     self.time_slot_size.to("hour")


@simplified_repr("value", "time_slot_size")
@dataclass(frozen=True)
class Workload:
    """
    Workload for a single timeslot (to be deprecated, redundant with WorkloadSeries)

    Attributes:
      - value:Requests: average number of requests arriving globally at the timeslot
      - time_slot_size:Time: duration of the timeslot
      - intra_slot_distribution:str: name of the distribution for the interarrival times of the
            requests ("exponential" by default)
    """

    value: Requests
    time_slot_size: Time
    intra_slot_distribution: str = "exponential"

    # def __post_init__(self):
    #     """Checks dimensions of the time_slot_size are valid."""
    #     self.time_slot_size.to("hour")


@simplified_repr("name", "max_vms", "max_cores")
@dataclass(frozen=True)
class LimitingSet:
    """LimitingSet restrictions.

    Attributes:
      - name:str: name of this limiting set (usually a region name)
      - max_vms:int: limit of the maximum number of VMs that can be running in this limiting set.
            Defaults to 0 which means "no limit"
      - max_cores:ComputationalUnits: limit on the maximum number of vcores that can be running in
            this limiting set. Defaults to 0 which means "no limits"
    """

    name: str
    max_vms: int = 0
    max_cores: ComputationalUnits = ComputationalUnits("0 cores")


@simplified_repr("name", "price", "cores", "mem")
@dataclass(frozen=True)
class InstanceClass:
    """InstanceClass characterization

    Attributes:
      - name:str: name of the instance class, usually built from the name of the VM type and the
            name of the limiting set in which it is deployed.
      - price:CurrencyPerTime: dollar per time unit
      - cores:ComputationalUnits:  millicores available in the VM
      - mem:Storage: GiB available in the VM
      - limit:int: maximum number of VMs (0 means "no limit")
      - limiting_sets:Tuple[LimitingSet]: LimitingSet to which this instance class belongs.
      - is_reserved:bool: True if the instance is reserved
      - is_private:bool: True if this instance class belongs to a private cloud
      - region:Region: Region to which this instance class belongs
    """

    name: str
    price: CurrencyPerTime
    cores: ComputationalUnits
    mem: Storage
    limit: int
    limiting_sets: Tuple[LimitingSet, ...]
    is_reserved: bool = False
    is_private: bool = False
    region: Region = Region("__world__")

    # def __post_init__(self):
    #     """Checks dimensions are valid and store them in the standard units."""
    #     object.__setattr__(self, "price", self.price.to("usd/hour"))
    #     object.__setattr__(self, "cores", self.cores.to("cores"))
    #     object.__setattr__(self, "mem", self.mem.to("gibibytes"))


@simplified_repr("value")
@dataclass(frozen=True)
class Latency:
    """
    Attributes:
        value:Time: latency value
    """

    value: Time = Time("0 s")


@simplified_repr("name", "cores", "mem", "app", "limit")
@dataclass(frozen=True)
class ContainerClass:
    """ContainerClass characterization

    Attributes:
      - name:str: name of the container class
      - cores:ComputationalUnits: number of millicores available in this container
      - mem:Storage: GiB available in this container
      - app:App: application (container image) run in this container
      - limit:int: maximum number of containers of this class
    """

    name: str
    cores: ComputationalUnits
    mem: Storage
    app: App
    limit: int

    # def __post_init__(self):
    #     """Checks dimensions are valid and store them in the standard units."""
    #     object.__setattr__(self, "cores", self.cores.to("millicores"))
    #     object.__setattr__(self, "mem", self.mem.to("gibibytes"))


@dataclass(frozen=True)
class ProcessClass:
    """Running process characterization (TODO: same than ContainerClass?)

    Attributes:
      - name:str: name of the process
      - ecu:float: units of computation that the process can give when running
      - mem:Storage: amount of memory that the process can use
      - app:App: application run by the process
      - limit:int: cpu limit enforced by operating system
    """

    name: str
    ecu: float
    mem: Storage
    app: App
    limit: int


@simplified_repr("value", "slo95")
@dataclass(frozen=True)
class Performance:
    """
    Model for the performance of a given application running on a given infrastructure. The
    performance is expressed as the number of requests that can be served in a unit of time such
    that the 95% of those get response in a time below the SLO.

    This is stored in two attributes:
    - value: the number of requests that can be served in a unit of time (ej: 100 req/s)
    - slo95: the response time below which 95% of the requests are served  (ej: 100 ms)
    """

    value: RequestsPerTime
    slo95: Time

    # Note: Mallovia does not use a explicit slo95. It works under the assumption that all requests
    # arriving at the time slot are served in the time slot. So implicitly the slo95 is the time
    # slot size (or perhaps the time slot divided by the number of requests?) TODO: Think about this


@simplified_repr("name")
@dataclass(frozen=True)
class System:
    """Model for the system, infrastructure and apps

    Attributes:
      - name:str: name of the system
      - ics:list[InstanceClass]: list of instance classes
      - ccs:list[ContainerClass]: list of container classes
      - perfs:dict[Tuple[InstanceClass, ContainerClass | ProcessClass | None, App], Performance]:
            performance of each application running on each instance class or container class
      - latencies:dict[Tuple[Region, Region], Latency]: latency between regions
      - default_latency:Latency: default latency between regions (0 by default) to be used when a
        specific value is not provided for a pair of regions
    """

    name: str
    ics: list[InstanceClass]
    ccs: list[ContainerClass]
    perfs: dict[
        Tuple[InstanceClass, ContainerClass | ProcessClass | None, App],
        Performance,
    ]
    latencies: dict[Tuple[Region, Region], Latency]
    default_latency: Latency = Latency(Time("0 s"))

    # def __post_init__(self):
    #     """Checks dimensions are valid and store them in the standard units."""
    #     new_perfs = {}
    #     for key, value in self.perfs.items():
    #         new_perfs[key] = value.to("req/hour")
    #     object.__setattr__(self, "perfs", new_perfs)


@simplified_repr("name", "system", "version")
@dataclass(frozen=True)
class Problem:
    """Problem description.

    Attributes:
      - name:str: name of the problem
      - system:System: system description
      - workloads:dict[Tuple[App, Region], WorkloadSeries]: workload of each application at each
        region
      - sched_time_size:Time: size of the time slot to schedule
      - max_avg_resp_time:Time: maximum average response time
      - version:str: version of the model
    """

    name: str
    system: System
    workloads: dict[Tuple[App, Region], WorkloadSeries]
    sched_time_size: Time
    max_avg_resp_time: Time = UNLIMITED_TIME
    version: str = __version__

    # Note: In mallovia the workload is not segmented by regions. This can be implemented by
    # defining a region called "world" which is the union of all regions.


def workloadSeries_scale(
    wl_series: WorkloadSeries, to: Time = Time("1 minute")
) -> WorkloadSeries:
    return WorkloadSeries(
        wl_series.description,
        wl_series.values,
        wl_series.time_slot_size.to(to),
        wl_series.intra_slot_distribution,
    )


def normalize_time_units(problem: Problem, units: str = "minute") -> Problem:
    sched_time_size = problem.sched_time_size.to(units)
    workloads = {}
    for (app, region), wl_series in problem.workloads.items():
        workloads[app, region] = workloadSeries_scale(wl_series, to=Time(units))
    ics = []
    for ic in problem.system.ics:
        ics.append(
            InstanceClass(
                name=ic.name,
                price=ic.price.to(f"usd/{units}"),
                cores=ic.cores,
                mem=ic.mem,
                limit=ic.limit,
                limiting_sets=ic.limiting_sets,
                is_reserved=ic.is_reserved,
                is_private=ic.is_private,
            )
        )
    perfs = {}
    for k, v in problem.system.perfs.items():
        ic, cc, app = k
        ic_idx = problem.system.ics.index(ic)
        perfs[ics[ic_idx], cc, app] = Performance(
            value=v.value.to(f"req/{units}"), slo95=v.slo95.to(units)
        )
    lats = {}
    for k_, v_ in problem.system.latencies.items():
        r1, r2 = k_
        lats[r1, r2] = v_.value.to(f"{units}")
    sys = System(
        name=problem.system.name,
        ics=ics,
        ccs=problem.system.ccs,
        perfs=perfs,
        latencies=lats,
    )
    return Problem(
        name=problem.name,
        system=sys,
        workloads=workloads,
        sched_time_size=sched_time_size,
        version=problem.version,
    )


__all__ = [
    "App",
    "ContainerClass",
    "InstanceClass",
    "LimitingSet",
    "Problem",
    "ProcessClass",
    "System",
    "WorkloadSeries",
]
