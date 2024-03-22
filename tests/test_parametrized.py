# pylint: disable=redefined-outer-name
"""
The tests are parametrized, so a few functions perform a lot of tests. The parameters are
the example to run and the speed_level to compute the solution.

Since the parameter value ends up as part of the test name, it is possible to use -k to
run only a specific set of tests. For example:

    # Run all tests for example1, at all speed levels
    pytest -v -k example1

    # Run all tests for speed_level 2, for all examples
    pytest -v -k "-2"

    # Run all tests for example4 at speed level 3
    pytest -v -k example4-3
"""

from itertools import product
import json
from pathlib import Path
import pytest
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App, AppFamilyPerf, System, Fcma
from fcma.visualization import SolutionPrinter
from fcma.model import SolutionSummary
from .examples import example1, example2, example3, example4


# Global variables for the different combinations
examples = [example1, example2, example3, example4]
speeds = [1, 2, 3]
cases_with_solutions = [
    (ex, sp, f"{ex.__name__}_solution_speed_{sp}.json") for ex, sp in product(examples, speeds)
]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def example1_solution_printer_expected_vms(request) -> list:
    """Reads a json file (received as parameter) which contains the expected VMs and their prices.
    Returns a list of "rows", each one containing the name of a instance class with the number
    of instances in parenthesis, and the price of the instances of that class.
    """
    filename = request.param
    path = Path(__file__).parent / Path("expected_sols") / filename
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # Organize by rows instead of columns
    data = [[*row] for row in zip(*data)]
    return data


@pytest.fixture(scope="module")
def example1_expected_solution_printer_expected_allocation(request) -> dict:
    """Reads a json file (received as parameter) which contains the allocation of containers
    for each app. Returns a dictionary in which the keys are the app names and the values
    are lists of "rows", each one containing the VM in which the container is deployed,
    the container name with the number of replicas in parenthesis, the app name,
    and the requests served by all the replicas.
    """
    filename = request.param
    path = Path(__file__).parent / Path("expected_sols") / filename
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # Organize by rows instead of columns
    for app in data:
        data[app] = [[*row] for row in zip(*data[app])]
    return data


# ==============================================================================
# Tests
# ==============================================================================


@pytest.mark.smoke
def test_bad_problem_is_rejected(aws_eu_west_1):
    """A problem that cannot be solved raises an exception"""
    apps = {"appA": App(name="appA")}
    workloads = {apps["appA"]: RequestsPerTime("20  req/s")}
    system: System = {
        (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("400000 mcores"),
            mem=Storage("500 mebibytes"),
            perf=RequestsPerTime("0.4 req/s"),
        ),
    }
    with pytest.raises(ValueError) as excinfo:
        Fcma(system, workloads)
    assert "enough cores or memory" in str(excinfo.value)


@pytest.mark.smoke
@pytest.mark.parametrize("example_data", examples, indirect=["example_data"])
def test_example_data_creation(example_data):
    """Smoke test to check that the generation of the different examples
    does not break"""
    # Check the data
    assert example_data is not None


@pytest.mark.smoke
@pytest.mark.parametrize("example_solving_pars", speeds, indirect=["example_solving_pars"])
def test_example_solving_config(example_solving_pars):
    """Smoke test for the example_solving_pars fixture"""
    # Check the solving configuration
    speed, config = example_solving_pars
    assert config is not None
    assert config.speed_level == speed


@pytest.mark.parametrize(
    "example_data,example_solving_pars",
    list(product(examples, speeds)),
    indirect=["example_data", "example_solving_pars"],
)
def test_example_solution_is_feasible(example_solution):
    """Check that the provided examples are feasible"""
    *_, solution = example_solution

    # Check the solution is feasible
    assert solution.is_infeasible() is False


@pytest.mark.parametrize(
    "example_data,example_solving_pars",
    list(product(examples, speeds)),
    indirect=["example_data", "example_solving_pars"],
)
def test_example_solution_check_allocation_is_valid(example_solution):
    """Check that the solution for the examples has no internal
    contradiction, using the method Fcma.check_allocation()"""
    fcma_problem, *_ = example_solution

    # Check the solution has no contradictions
    fcma_problem.check_allocation()


# ==============================================================================
# Test the solution provided by SolutionPrinter
# ==============================================================================


def test_solution_printer_is_infeasible(aws_eu_west_1, monkeypatch, capsys):
    """Test that the solution printer produces no tables when
    the problem is infeasible"""
    # We solve a perfectly solvable problem, but we using monkeypatching
    # to fake pulp reporting it is infeasible
    apps = {"appX": App(name="appX")}
    workloads = {apps["appX"]: RequestsPerTime("10  req/s")}
    system: System = {
        (apps["appX"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("500 mcores"),
            mem=Storage("300 mebibytes"),
            perf=RequestsPerTime("0.5 req/s"),
        ),
    }

    from fcma import model  # pylint: disable=import-outside-toplevel

    # Fake solving to get an infeasible solution
    monkeypatch.setattr(
        model.FcmaStatus, "pulp_to_fcma_status", lambda *_: model.FcmaStatus.INVALID
    )
    problem = Fcma(system, workloads)
    solution = problem.solve()

    sp = SolutionPrinter(solution)
    sp.print()

    result = capsys.readouterr()
    assert "Non feasible" in result.out
    assert "INVALID" in result.out
    assert result.out.endswith("INVALID\n")  # No tables after this


# The following tests are parametrized only on speed levels 1 and 2 and only for example1


@pytest.mark.parametrize(
    "example_data, example_solving_pars, example1_solution_printer_expected_vms",
    [
        (example1, 1, "solutionprinter_vms_example1_speed_1.json"),
        (example1, 2, "solutionprinter_vms_example1_speed_2.json"),
    ],
    indirect=["example_data", "example_solving_pars", "example1_solution_printer_expected_vms"],
)
def test_example1_solution_printer_vms_and_prices(
    example_solution, example1_solution_printer_expected_vms
):
    """Test that the data in the VM table generated by SolutionPrinter
    matches the expected content for the examples"""

    # pylint: disable=protected-access
    *_, solution = example_solution
    sp = SolutionPrinter(solution)

    # Another way, get the rich table and inspect the contents
    vm_table = sp._get_vm_table()
    # Organize by rows instead of columns
    solution_data = [col._cells for col in vm_table.columns]
    solution_data = [[*row] for row in zip(*solution_data)]
    assert len(solution_data) == len(example1_solution_printer_expected_vms)
    for row in solution_data:
        assert row in example1_solution_printer_expected_vms


@pytest.mark.parametrize(
    "example_data, example_solving_pars, example1_expected_solution_printer_expected_allocation",
    [
        (example1, 1, "solutionprinter_allocation_example1_speed_1.json"),
        (example1, 2, "solutionprinter_allocation_example1_speed_2.json"),
    ],
    indirect=[
        "example_data",
        "example_solving_pars",
        "example1_expected_solution_printer_expected_allocation",
    ],
)
def test_example1_solution_printer_apps_allocations(
    example_solution, example1_expected_solution_printer_expected_allocation
):
    """Test that the data in the Apps table generated by SolutionPrinter
    matches the expected content for the examples"""
    # pylint: disable=protected-access
    fcma_problem, _, solution = example_solution
    sp = SolutionPrinter(solution)
    apps_allocations = sp._get_app_tables()

    # The apps in the solution must match the apps in the problem
    problem_apps = set(app.name for app in fcma_problem._workloads.keys())
    solution_apps = set(apps_allocations.keys())
    assert problem_apps == solution_apps

    def check_app_alloc(app, sol_data):
        expected_alloc = example1_expected_solution_printer_expected_allocation[app]
        assert len(sol_data) == len(expected_alloc)
        for col in sol_data:
            assert col in expected_alloc

    # Check the allocations for each app
    for app, table in apps_allocations.items():
        solution_data = [col._cells for col in table.columns]
        solution_data = [[*row] for row in zip(*solution_data)]
        check_app_alloc(app, solution_data)


# ==============================================================================
# Smoke tests for SolutionSummary class
# ==============================================================================


def test_solution_summary_is_infeasible(aws_eu_west_1, monkeypatch):
    """Test that the SolutionSummary produces empty data structures
    when the solution is not feasible"""
    # We solve a perfectly solvable problem, but we using monkeypatching
    # to fake pulp reporting it is infeasible

    # pylint: disable=protected-access
    apps = {"appZ": App(name="appZ")}
    workloads = {apps["appZ"]: RequestsPerTime("15  req/s")}
    system: System = {
        (apps["appZ"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("600 mcores"),
            mem=Storage("200 mebibytes"),
            perf=RequestsPerTime("0.6 req/s"),
        ),
    }

    from fcma import model  # pylint: disable=import-outside-toplevel

    # Fake solving to get an infeasible solution
    monkeypatch.setattr(
        model.FcmaStatus, "pulp_to_fcma_status", lambda *_: model.FcmaStatus.INVALID
    )
    problem = Fcma(system, workloads)
    solution = problem.solve()

    summary = SolutionSummary(solution)
    assert summary.is_infeasible() is True

    vm_summary = summary.get_vm_summary()
    assert vm_summary.total_num == 0
    assert len(vm_summary.vms) == 0

    apps_summary = summary.get_all_apps_allocations()
    assert len(apps_summary) == 0


# The following test is run for all combinations of examples and speeds
@pytest.mark.parametrize(
    "example_data,example_solving_pars",
    list(product(examples, speeds)),
    indirect=["example_data", "example_solving_pars"],
)
def test_solution_summary_as_dict_and_back(example_solution):
    """Test that the SolutionSummary serialization code works correctly
    by converting it to a dictionary and then back to a SolutionSummary
    and check that the final object is equal to the original"""
    *_, solution = example_solution
    summary = SolutionSummary(solution)
    summary_dict = summary.as_dict()
    summary2 = SolutionSummary.from_dict(summary_dict)
    assert summary.get_vm_summary() == summary2.get_vm_summary()
    assert summary.get_all_apps_allocations() == summary2.get_all_apps_allocations()


# ==============================================================================
# Check solutions of examples, for all examples and speeds
# ==============================================================================


@pytest.mark.parametrize(
    "example_data, example_solving_pars, expected_solution_summary",
    cases_with_solutions,
    indirect=["example_data", "example_solving_pars", "expected_solution_summary"],
)
def test_example_solution_summary_vms(example_solution, expected_solution_summary):
    """Check that the VM summary of the solution matches the expected one
    for the known examples"""
    *_, solution = example_solution
    summary = SolutionSummary(solution)
    vm_alloc = summary.get_vm_summary()
    expected_vm_alloc = expected_solution_summary.get_vm_summary()
    assert vm_alloc == expected_vm_alloc


@pytest.mark.parametrize(
    "example_data, example_solving_pars, expected_solution_summary",
    cases_with_solutions,
    indirect=["example_data", "example_solving_pars", "expected_solution_summary"],
)
def test_example_solution_summary_all_apps(example_solution, expected_solution_summary):
    """Check that the allocation of containers for each app in the solution
    matches the expected one for the known examples"""
    *_, solution = example_solution
    summary = SolutionSummary(solution)
    app_alloc = summary.get_all_apps_allocations()
    expected_app_alloc = expected_solution_summary.get_all_apps_allocations()
    assert app_alloc == expected_app_alloc


# No need to test all cases for this, as the previous tests already check the contents
@pytest.mark.parametrize(
    "example_data, example_solving_pars, expected_solution_summary",
    cases_with_solutions[:2],
    indirect=["example_data", "example_solving_pars", "expected_solution_summary"],
)
def test_example_solution_summary_single_app(example_solution, expected_solution_summary):
    """Test for the method SolutionSummary.get_app_allocation_summary()"""
    *_, solution = example_solution
    summary = SolutionSummary(solution)
    expected_allocation = expected_solution_summary.get_all_apps_allocations()
    for app in expected_allocation:
        info = summary.get_app_allocation_summary(app)
        assert info == expected_allocation[app]


# # # print("\n----------- Solution check --------------")
# # # for attribute in dir(slack):
# # #     if attribute.endswith("percentage"):
# # #         print(f"{attribute}: {getattr(slack, attribute): .2f} %")
# # # print("-----------------------------------------")
