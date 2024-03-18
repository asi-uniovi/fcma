"""
Some tests based on the example provided in examples folder
"""

from itertools import product
import json
from pathlib import Path
import pytest
from pytest import approx
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App, AppFamilyPerf, System, Fcma, SolvingPars, Solution
from fcma.visualization import SolutionPrinter
from fcma.model import Vm, SolutionSummary
import importlib.util
import os
from .examples import example1, example2, example3, example4

from cloudmodel.unified.units import CurrencyPerTime


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
def aws_eu_west_1():
    # This fixture imports the AWS eu-west-1 module from the examples folder, making
    # it accesible to the tests. It cannot be imported directly because the examples folder is
    # not part of fcma and thus is not installed
    path_to_module = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "aws_eu_west_1.py")
    )
    spec = importlib.util.spec_from_file_location("aws_eu_west_1", path_to_module)
    aws_eu_west_1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aws_eu_west_1)
    return aws_eu_west_1


@pytest.fixture(scope="module")
def example_data(aws_eu_west_1, request) -> tuple[System, dict[App, RequestsPerTime]]:
    """Defines a set of parameters for the problem received as parameter"""
    # The parameter is a function whirh returns the data for the example
    example = request.param
    return example(aws_eu_west_1)


@pytest.fixture(scope="module")
def example_solving_pars(request) -> SolvingPars:
    """Defines parameters for solving the example 1, with different speed levels.
    Returns the speed level received as parameter and the SolvingPars object"""

    # Three speed levels are possible: 1, 2 and 3, being speed level 1 the slowest, but the one giving the best
    # cost results. A solver with options can be passed for speed levels 1 and 2, or defaults are used. For instance:
    #             from pulp import PULP_CBC_CMD
    #             solver = PULP_CBC_CMD(timeLimit=10, gapRel=0.01, threads=8)
    #             solving_pars = SolvingPars(speed_level=1, solver=solver)
    # More information can be found on: https://coin-or.github.io/pulp/technical/solvers.html
    speed = request.param
    solving_pars = SolvingPars(speed_level=speed)
    return speed, solving_pars


@pytest.fixture(scope="module")
def example_solution(example_data, example_solving_pars) -> tuple[Fcma, SolvingPars, Solution]:
    """Instantiates a Fcma problem and solves it. Returns the problem, the solving parameters
    and the solution"""
    # Solve the allocation problem
    Vm._last_ic_index = {}
    problem = Fcma(*example_data)
    solution = problem.solve(example_solving_pars[1])
    return problem, example_solving_pars, solution


@pytest.fixture(scope="module")
def example1_expected_vms_SolutionPrinter(request) -> list:
    """Reads a json file (received as parameter) which contains the expected VMs and their prices.
    Returns a list of "rows", each one containing the name of a instance class with the number
    of instances in parenthesis, and the price of the instances of that class.
    """
    filename = request.param
    path = Path(__file__).parent / Path("expected_sols") / filename
    with open(path, "r") as file:
        data = json.load(file)
    # Organize by rows instead of columns
    data = [[*row] for row in zip(*data)]
    return data


@pytest.fixture(scope="module")
def example1_expected_allocation_SolutionPrinter(request) -> dict:
    """Reads a json file (received as parameter) which contains the allocation of containers
    for each app. Returns a dictionary in which the keys are the app names and the values
    are lists of "rows", each one containing the VM in which the container is deployed,
    the container name with the number of replicas in parenthesis, the app name,
    and the requests served by all the replicas.
    """
    filename = request.param
    path = Path(__file__).parent / Path("expected_sols") / filename
    with open(path, "r") as file:
        data = json.load(file)
    # Organize by rows instead of columns
    for app in data:
        data[app] = [[*row] for row in zip(*data[app])]
    return data


@pytest.fixture(scope="module")
def expected_solution_summary(request) -> SolutionSummary:
    filename = request.param
    path = Path(__file__).parent / Path("expected_sols") / filename
    with open(path, "r") as file:
        data = json.load(file)
    ss = SolutionSummary.from_dict(data)
    return ss


# ==============================================================================
# Tests
# ==============================================================================


def test_bad_problem_is_rejected(aws_eu_west_1):
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


@pytest.mark.parametrize("example_data", examples, indirect=["example_data"])
def test_example1_data_creation(example_data):
    # Check the data
    assert example_data is not None


@pytest.mark.parametrize("example_solving_pars", speeds, indirect=["example_solving_pars"])
def test_example1_solving_config(example_solving_pars):
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
    *_, solution = example_solution

    # Check the solution is feasible
    assert solution.is_infeasible() == False


@pytest.mark.parametrize(
    "example_data,example_solving_pars",
    list(product(examples, speeds)),
    indirect=["example_data", "example_solving_pars"],
)
def test_example_solution_is_valid(example_solution):
    fcma_problem, *_ = example_solution

    # Check the solution has no contradictions
    slack = fcma_problem.check_allocation()


# ==============================================================================
# Test the solution provided by SolutionPrinter
# ==============================================================================


def test_SolutionPrinter_is_infeasible(aws_eu_west_1, monkeypatch, capsys):
    # We solve a perfectly solvable problem, but we using monkeypatching
    # to fake pulp reporting it is infeasible
    apps = {"appA": App(name="appA")}
    workloads = {apps["appA"]: RequestsPerTime("20  req/s")}
    system: System = {
        (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("400 mcores"),
            mem=Storage("500 mebibytes"),
            perf=RequestsPerTime("0.4 req/s"),
        ),
    }

    from fcma import model

    # Fake solving to get an infeasible solution
    monkeypatch.setattr(
        model.FcmaStatus, "pulp_to_fcma_status", lambda *_: model.FcmaStatus.INVALID
    )
    # TODO: Fix bug to avoid mokeypatching get_worst_status too
    monkeypatch.setattr(model.FcmaStatus, "get_worst_status", lambda *_: model.FcmaStatus.INVALID)
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
    "example_data, example_solving_pars, example1_expected_vms_SolutionPrinter",
    [
        (example1, 1, "solutionprinter_vms_example1_speed_1.json"),
        (example1, 2, "solutionprinter_vms_example1_speed_2.json"),
    ],
    indirect=["example_data", "example_solving_pars", "example1_expected_vms_SolutionPrinter"],
)
def test_example1_solution_printer_vms_and_prices(
    example_solution, example1_expected_vms_SolutionPrinter
):
    *_, solution = example_solution
    sp = SolutionPrinter(solution)

    # Another way, get the rich table and inspect the contents
    vm_table = sp._get_vm_table()
    # Organize by rows instead of columns
    solution_data = [col._cells for col in vm_table.columns]
    solution_data = [[*row] for row in zip(*solution_data)]
    assert len(solution_data) == len(example1_expected_vms_SolutionPrinter)
    for row in solution_data:
        assert row in example1_expected_vms_SolutionPrinter


@pytest.mark.parametrize(
    "example_data, example_solving_pars, example1_expected_allocation_SolutionPrinter",
    [
        (example1, 1, "solutionprinter_allocation_example1_speed_1.json"),
        (example1, 2, "solutionprinter_allocation_example1_speed_2.json"),
    ],
    indirect=[
        "example_data",
        "example_solving_pars",
        "example1_expected_allocation_SolutionPrinter",
    ],
)
def test_example1_solution_printer_apps_allocations(
    example_solution, example1_expected_allocation_SolutionPrinter
):
    fcma_problem, _, solution = example_solution
    sp = SolutionPrinter(solution)
    apps_allocations = sp._get_app_tables()

    # The apps in the solution must match the apps in the problem
    problem_apps = set(app.name for app in fcma_problem._workloads.keys())
    solution_apps = set(apps_allocations.keys())
    assert problem_apps == solution_apps

    def check_app_alloc(app, sol_data):
        expected_alloc = example1_expected_allocation_SolutionPrinter[app]
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


def test_SolutionSummary_is_infeasible(aws_eu_west_1, monkeypatch):
    # We solve a perfectly solvable problem, but we using monkeypatching
    # to fake pulp reporting it is infeasible
    apps = {"appA": App(name="appA")}
    workloads = {apps["appA"]: RequestsPerTime("20  req/s")}
    system: System = {
        (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("400 mcores"),
            mem=Storage("500 mebibytes"),
            perf=RequestsPerTime("0.4 req/s"),
        ),
    }

    from fcma import model

    # Fake solving to get an infeasible solution
    monkeypatch.setattr(
        model.FcmaStatus, "pulp_to_fcma_status", lambda *_: model.FcmaStatus.INVALID
    )
    # TODO: Fix bug to avoid mokeypatching get_worst_status too
    monkeypatch.setattr(model.FcmaStatus, "get_worst_status", lambda *_: model.FcmaStatus.INVALID)
    problem = Fcma(system, workloads)
    solution = problem.solve()

    summary = SolutionSummary(solution)
    assert summary.is_infeasible() == True

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
def test_SolutionSummary_as_dict_and_back(example_solution):
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
def test_example1_SolutionSummary_vms(example_solution, expected_solution_summary):
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
def test_example1_SolutionSummary_all_apps(example_solution, expected_solution_summary):
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
def test_example1_SolutionSummary_single_app(example_solution, expected_solution_summary):
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
