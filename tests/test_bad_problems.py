"""Check that appropriate exceptions are raised if we try to define a problem
with invalid input"""

import pytest
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App, AppFamilyPerf, System, Fcma


def simple_example_perfs(aws_eu_west_1, required_cores="100 mcores"):
    """Generate a simple example to use in different tests"""
    apps = {"appA": App(name="appA")}
    system: System = {
        (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits(required_cores),
            mem=Storage("500 mebibytes"),
            perf=RequestsPerTime("0.4 req/s"),
        ),
    }
    return apps, system


@pytest.mark.smoke
def test_not_enough_cores_is_rejected(aws_eu_west_1):
    """A problem that cannot be solved raises an exception"""
    apps, system = simple_example_perfs(aws_eu_west_1, required_cores="400000 mcores")
    workloads = {apps["appA"]: RequestsPerTime("20  req/s")}

    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "enough cores or memory" in str(excinfo.value)


@pytest.mark.smoke
def test_workload_not_dict_is_rejected(aws_eu_west_1):
    """A problem with non-dict workload is rejected"""
    _, system = simple_example_perfs(aws_eu_west_1)
    # Workload is not a dict
    workloads = [20, 30]
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "Workloads must be a dict" in str(excinfo.value)


@pytest.mark.smoke
def test_workload_bad_keys_is_rejected(aws_eu_west_1):
    """A problem with non-apps as keys in workload is rejected"""
    _, system = simple_example_perfs(aws_eu_west_1)
    # Workload keys are not apps
    workloads = {"appA": 20}  # "appA" is not an app, it is a string
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "Workloads must be a dict" in str(excinfo.value)


@pytest.mark.smoke
def test_empty_workload_and_system_is_rejected():
    """A problem with non-apps as keys in workload is rejected"""
    # Workload keys are not apps
    workloads = dict()
    system = dict()
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "At least one application is required" in str(excinfo.value)


@pytest.mark.smoke
def test_workload_bad_units_is_rejected(aws_eu_west_1):
    """A problem with bad units in workload is rejected"""
    apps, system = simple_example_perfs(aws_eu_west_1)
    # Workload values are incorrect
    workloads = {apps["appA"]: 20}  # No units
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "Workloads must be a dict" in str(excinfo.value)

    workloads = {apps["appA"]: Storage("20 GiB")}  # bad dimension
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "Workloads must be a dict" in str(excinfo.value)


@pytest.mark.smoke
def test_non_dict_system_is_rejected(aws_eu_west_1):
    """A problem with non-dict system is rejected"""
    apps, _ = simple_example_perfs(aws_eu_west_1)
    workloads = {apps["appA"]: RequestsPerTime("20  req/s")}
    system = "Foo"  # not a dict
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "App family performances must be a dict" in str(excinfo.value)


@pytest.mark.smoke
def test_incorrect_key_types_in_system_is_rejected(aws_eu_west_1):
    """A problem with incorrect types for the keys of the system dict is rejected"""
    apps, _ = simple_example_perfs(aws_eu_west_1)
    workloads = {apps["appA"]: RequestsPerTime("20  req/s")}

    # Keys are not even tuples
    system = {"foo": "bar"}
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "App family performances must be a dict" in str(excinfo.value)

    # Keys are tuples containing incorrect type for the first element
    system = {("foo", aws_eu_west_1.c5_fm): "bar"}
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "App family performances must be a dict" in str(excinfo.value)

    # Keys are tuples with incorrect type for the second element
    system = {(apps["appA"], aws_eu_west_1.c5_large): "bar"}
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "App family performances must be a dict" in str(excinfo.value)


@pytest.mark.smoke
def test_incorrect_value_types_in_system_is_rejected(aws_eu_west_1):
    """A problem with incorrect types for the keys of the system dict is rejected"""
    apps, _ = simple_example_perfs(aws_eu_west_1)
    workloads = {apps["appA"]: RequestsPerTime("20  req/s")}

    # The value is not of type AppFamilyPerf
    system = {(apps["appA"], aws_eu_west_1.c5_fm): "bar"}
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "App family performances must be a dict" in str(excinfo.value)


@pytest.mark.smoke
def test_inconsistent_apps_in_problem_is_rejected(aws_eu_west_1):
    """A problem with inconsistent apps in the system and workload is rejected"""
    apps, system = simple_example_perfs(aws_eu_west_1)
    workloads = {App("appB"): RequestsPerTime("20  req/s")}

    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "appA has no workload" in str(excinfo.value)

    workloads = {
        apps["appA"]: RequestsPerTime("10 req/s"),
        App("appB"): RequestsPerTime("20  req/s"),
    }

    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "appB has no performance parameters" in str(excinfo.value)
