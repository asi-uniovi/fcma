from dataclasses import asdict, dataclass, field, fields
from cloudmodel.unified.units import CheckedDimensionality
from cloudmodel.unified.units import (
    ComputationalUnits,
    CurrencyPerTime,
    RequestsPerTime,
    Storage,
)
from .fcma import Fcma
from .model import AppFamilyPerf, App, InstanceClass, InstanceClassFamily

DefaultUnitsMap = list[tuple[CheckedDimensionality, str, str]]


@dataclass
class SystemData:
    apps: dict[str, App] = field(default_factory=dict)
    families: list[str] = field(default_factory=list)
    instance_classes: dict[str, dict] = field(default_factory=dict)
    perf: dict[str, AppFamilyPerf] = field(default_factory=dict)


@dataclass
class ProblemData:
    units: dict[str, str]
    system: SystemData
    workloads: dict[str, float]


class ProblemSerializer:
    default_units_map: DefaultUnitsMap = [
        (ComputationalUnits, "cpu", "mcores"),
        (CurrencyPerTime, "cost/t", "usd/h"),
        (RequestsPerTime, "workload", "req/h"),
        (Storage, "mem", "GiB"),
    ]

    def __init__(self, problem: Fcma, default_units: DefaultUnitsMap = None):
        self.problem = problem
        if default_units == None:
            default_units = ProblemSerializer.default_units_map
        self.default_units_map = default_units
        self.name_units_dict = {name: unit for _, name, unit in default_units}
        self.cls_units_dict = {cls: unit for cls, _, unit in default_units}
        self.name_cls_dict = {name: cls for cls, name, _ in default_units}
        self._problem_data: ProblemData = None

    def as_dict(self) -> dict:
        self._prepare_problem()
        return asdict(self._problem_data)

    def _prepare_workloads(self) -> dict:
        if self._workloads is None:
            unit = self.cls_units_dict[RequestsPerTime]
            self._workloads = {
                app.name: wl.m_as(unit) for app, wl in self.problem._workloads.items()
            }
        return self._workloads

    def _prepare_problem(self):
        if self._problem_data is not None:
            return
        wl_unit = self.cls_units_dict[RequestsPerTime]
        system = SystemData()
        workloads = {}
        # Extract app info and workloads
        for app, workload in self.problem._workloads.items():
            workloads[app.name] = workload.m_as(wl_unit)
            system.apps[app.name] = app

        families = {}
        # Extract family names, instance classes
        for (app, fm), perf in self.problem._system.items():
            families[fm.name] = fm
            system.perf[f"({app.name}, {fm.name})"] = self._perf_as_dict(perf)

        system.instance_classes = self._get_instance_classes(families)
        system.families = {fm.name: [fp.name for fp in fm.parent_fms] for fm in families.values()}
        self._problem_data = ProblemData(
            system=system, workloads=workloads, units=self.name_units_dict
        )

    def _perf_as_dict(self, perf: AppFamilyPerf) -> dict:
        return {
            "cores": perf.cores.m_as(self.cls_units_dict[ComputationalUnits]),
            "mem": [m.m_as(self.cls_units_dict[Storage]) for m in perf.mem],
            "aggs": list(perf.aggs),
            "maxagg": perf.maxagg,
            "perf": perf.perf.m_as(self.cls_units_dict[RequestsPerTime]),
        }

    def _get_instance_classes(self, families: dict[str, InstanceClassFamily]) -> dict[str, dict]:
        instance_classes = {}
        new_families = families.copy()
        for family in families.values():
            for ic in family.ics:
                instance_classes[ic.name] = self._ic_as_dict(ic)
                if ic.family.name not in new_families:
                    new_families[ic.family.name] = ic.family
        families.update(new_families)
        return instance_classes

    def _ic_as_dict(self, ic: InstanceClass) -> dict:
        return {
            "cores": ic.cores.m_as(self.cls_units_dict[ComputationalUnits]),
            "mem": ic.mem.m_as(self.cls_units_dict[Storage]),
            "price": ic.price.m_as(self.cls_units_dict[CurrencyPerTime]),
            "family": ic.family.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Fcma:
        problem = cls(problem=None)
        name_units_map = {name: unit for name, unit in data["units"].items()}
        cls_units_map = {
            cls.__name__: (cls, name_units_map[name])
            for cls, name, _ in ProblemSerializer.default_units_map
        }

        apps = {name: App(**app) for name, app in data["system"]["apps"].items()}

        families = {}
        for name, parent in data["system"]["families"].items():
            families[name] = InstanceClassFamily(name, parent_fms=[families[p] for p in parent])

        ics = {}
        for name, ic in data["system"]["instance_classes"].items():
            ic["family"] = families[ic["family"]]
            ic["name"] = name
            ProblemSerializer._add_units(ic, InstanceClass, cls_units_map)
            ics[name] = InstanceClass(**ic)

        perfs = {}
        for name, perf in data["system"]["perf"].items():
            app_name, family_name = name[1:-1].split(", ")
            app = apps[app_name]
            family = families[family_name]
            ProblemSerializer._add_units(perf, AppFamilyPerf, cls_units_map)
            perf["mem"] = tuple([Storage(f"{m} {name_units_map['mem']}") for m in perf["mem"]])
            perf["aggs"] = tuple(perf["aggs"])
            perfs[app, family] = AppFamilyPerf(**perf)
        
        workloads = {}
        for name, wl in data["workloads"].items():
            workloads[apps[name]] = RequestsPerTime(f"{wl} {name_units_map['workload']}")

        return Fcma(system=perfs, workloads=workloads)

    @staticmethod
    def _add_units(data, cls, cls_units_map):
        for field in fields(cls):
            unit_cls, unit = cls_units_map.get(field.type, (None, None))
            if unit_cls:
                data[field.name] = unit_cls(f"{data[field.name]} {unit}")
