"""Classes to serialize Problems to dictionaries and back, which
allows for easy generation of json/yaml as problem interchange
format"""

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
from . import model

DefaultUnitsMap = list[tuple[CheckedDimensionality, str, str]]


@dataclass
class SystemData:
    """Dataclass to store system information for serialization"""
    apps: dict[str, App] = field(default_factory=dict)
    families: list[str] = field(default_factory=list)
    instance_classes: dict[str, dict] = field(default_factory=dict)
    perf: dict[str, AppFamilyPerf] = field(default_factory=dict)


@dataclass
class ProblemData:
    """Dataclass to store problem information for serialization"""
    units: dict[str, str]
    system: SystemData
    workloads: dict[str, float]
    version: str = model.__version__


class ProblemSerializer:
    """Dataclass to serialize and deserialize Fcma problems to dictionaries"""
    default_units_map: DefaultUnitsMap = [
        (ComputationalUnits, "cpu", "mcores"),
        (CurrencyPerTime, "cost/t", "usd/h"),
        (RequestsPerTime, "workload", "req/h"),
        (Storage, "mem", "GiB"),
    ]

    def __init__(self, problem: Fcma, default_units: DefaultUnitsMap = None):
        """Initialize the serializer with a problem and a default units map
        
        default_units is a list of tuples with the following format:
            (unit_cls, name, unit_str), ...

            unit_cls is one of the classes defined in cloudmodel such as Storage
            name is a nickname that this dimensionality will receive in the
              the generated dictionary, in the field "units", for example "mem"
            unit_str is the string expressing the defaultunits that will be used
              for all fields with this dimensionality, for example "GiB"

        This influeces the generated dictinary in two ways:

            1. The dictionary will contain a "units" field containing 
               another dict with pairs {"name": "unit_str"}, for example:
                 "units": {
                    "cpu": "mcores",
                    "cost/t": "usd/h",
                    "workload": "req/h",
                    "mem": "GiB"
                },
            2. All values in the problem will have their units removed, only the
               magnitude will be stored, converted to the specified units, so 
               for example, the memory of a InstanceClass will be a float, for
               example 20.0 that will mean "20 GiB" in this example

        If the paremeter default_units is None, the default units map will be used
        which is the following:

            default_units_map: DefaultUnitsMap = [
                (ComputationalUnits, "cpu", "mcores"),
                (CurrencyPerTime, "cost/t", "usd/h"),
                (RequestsPerTime, "workload", "req/h"),
                (Storage, "mem", "GiB"),
            ]

        See also the documentation of static method ProblemSerializer.from_dict()
        """
        self.problem = problem
        if default_units is None:
            default_units = ProblemSerializer.default_units_map
        self.default_units_map = default_units
        self.name_units_dict = {name: unit for _, name, unit in default_units}
        self.cls_units_dict = {cls.__name__: unit for cls, _, unit in default_units}
        self.name_cls_dict = {name: cls for cls, name, _ in default_units}
        self._problem_data: ProblemData = None
        self._workloads = None

    def as_dict(self) -> dict:
        """Generates a dictionary containing the definition of the problem,
        suitable to serialize as JSON or YAML"""
        self._prepare_problem()
        return asdict(self._problem_data)

    def _prepare_problem(self):
        """Computes all the required data to assemble a ProblemData object"""

        # pylint: disable=protected-access
        if self._problem_data is not None:
            return
        wl_unit = self.cls_units_dict["RequestsPerTime"]
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

    def _prepare_workloads(self) -> dict:
        """Computes the dictionary of workloads"""

        # pylint: disable=protected-access
        if self._workloads is None:
            unit = self.cls_units_dict[RequestsPerTime]
            self._workloads = {
                app.name: wl.m_as(unit) for app, wl in self.problem._workloads.items()
            }
        return self._workloads

    def _perf_as_dict(self, perf: AppFamilyPerf) -> dict:
        """Converts an AppFamilyPerf object to a dictionary, removing units"""
        p = asdict(perf)
        self._remove_units(p, AppFamilyPerf)

        # Nested lists or dicts have to be processed ad-hoc
        # TODO: generalize this so that is not hard-coded?
        if isinstance(p["mem"], (list, tuple)):
            p["mem"] = [m.m_as(self.cls_units_dict["Storage"]) for m in perf.mem]
        return p

    def _get_instance_classes(self, families: dict[str, InstanceClassFamily]) -> dict[str, dict]:
        """Finds all instance classes and families in the problem"""
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
        """Converts an InstanceClass object to a dictionary, removing units"""
        _ic = asdict(ic)
        self._remove_units(_ic, InstanceClass)
        _ic["family"] = ic.family.name
        return _ic

    @classmethod
    def from_dict(cls, data: dict) -> Fcma:
        """Receives a dictionary as the one generated by ProblemSerializer.as_dict()
        and uses it to create a instance of Fcma
        
        During the creation of the Fcma object, the units have to be restored,
        because the data dict contains no units for the magnitudes such as "cores",
        "mem", "perf", etc.

        However, the data dict must contain a "units" field specifying the units
        in which these magnitudes are, depending on their type, for example:

            "units": {
                "cpu": "mcores",
                "cost/t": "usd/h",
                "workload": "req/h",
                "mem": "GiB"
            },

        The Fcma object returned will use appropriate cloudmodel units for
        each required value.
        """

        name_units_map = data["units"]
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
        """Helper function to restore units for the fields that require them"""
        for _field in fields(cls):
            unit_cls, unit = cls_units_map.get(_field.type, (None, None))
            if unit_cls:
                data[_field.name] = unit_cls(f"{data[_field.name]} {unit}")

    def _remove_units(self, data, cls):
        """Helper method to remove units from the fields that have them"""
        for _field in fields(cls):
            unit = self.cls_units_dict.get(_field.type, None)
            if unit:
                data[_field.name] = data[_field.name].m_as(unit)
