"""
Some tests based on the example provided in examples folder
"""

import json
from pathlib import Path
import pytest
from pytest import approx
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App, AppFamilyPerf, System, Fcma, SolvingPars, Solution
from fcma.visualization import SolutionPrinter
from fcma.model import AllVmSummary, Vm, SolutionSummary
import importlib.util
import os

from cloudmodel.unified.units import CurrencyPerTime


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
def example1_data(aws_eu_west_1, request) -> Fcma:
    """Defines a set of parameters for the example1 problem"""
    apps = {
        "appA": App(name="appA", sfmpl=0.5),
        "appB": App(name="appB", sfmpl=0.2),
        "appC": App(name="appC"),
        "appD": App(name="appD"),
    }

    workloads = {
        apps["appA"]: RequestsPerTime("6  req/s"),
        apps["appB"]: RequestsPerTime("12 req/s"),
        apps["appC"]: RequestsPerTime("20  req/s"),
        apps["appD"]: RequestsPerTime("15  req/s"),
    }

    system: System = {
        # For family aws_eu_west_1.c5_m5_r5_fm. It includes AWS c5, m5 and r5 instances
        (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("400 mcores"),
            mem=Storage("500 mebibytes"),
            perf=RequestsPerTime("0.4 req/s"),
            aggs=(2,),
        ),
        (apps["appB"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("80 mcores"),
            mem=Storage("200 mebibytes"),
            perf=RequestsPerTime("0.5 req/s"),
            aggs=(2, 4, 8, 12),
        ),
        (apps["appC"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("90 mcores"),
            mem=Storage("350 mebibytes"),
            perf=RequestsPerTime("0.2 req/s"),
            aggs=(2, 4, 10),
        ),
        (apps["appD"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("8500 mcores"),
            mem=Storage("25000 mebibytes"),
            perf=RequestsPerTime("1 req/s"),
        ),
        # For family aws_eu_west_1.c6g_m6g_r6g_fm. It includes AWS c6g, m6g and r6g instances
        (apps["appB"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
            cores=ComputationalUnits("100 mcores"),
            mem=Storage("250 mebibytes"),
            perf=RequestsPerTime("0.35 req/s"),
            aggs=(2, 4, 10),
        ),
        (apps["appC"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
            cores=ComputationalUnits("120 mcores"),
            mem=Storage("450 mebibytes"),
            perf=RequestsPerTime("0.4 req/s"),
            aggs=(2, 4, 8),
        ),
        (apps["appD"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
            cores=ComputationalUnits("6500 mcores"),
            mem=Storage("22000 mebibytes"),
            perf=RequestsPerTime("0.8 req/s"),
        ),
    }

    return system, workloads


@pytest.fixture(scope="module")
def example1_solving_pars(request) -> SolvingPars:
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
def example1_solution(example1_data, example1_solving_pars) -> tuple[Fcma, SolvingPars, Solution]:
    """Instantiates a Fcma problem and solves it. Returns the problem, the solving parameters
    and the solution"""
    # Solve the allocation problem
    Vm._last_ic_index = {}
    problem = Fcma(*example1_data)
    solution = problem.solve(example1_solving_pars[1])
    return problem, example1_solving_pars, solution


@pytest.fixture(scope="module")
def example1_expected_vms_SolutionPrinter(request) -> list:
    """Reads a json file (received as parameter) which contains the expected VMs and their prices.
    Returns a list of "rows", each one containing the name of a instance class with the number
    of instances in parenthesis, and the price of the instances of that class.
    """
    filename = request.param
    path = Path(__file__).parent / filename
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
    path = Path(__file__).parent / filename
    with open(path, "r") as file:
        data = json.load(file)
    # Organize by rows instead of columns
    for app in data:
        data[app] = [[*row] for row in zip(*data[app])]
    return data


@pytest.fixture(scope="module")
def expected_solution_summary(request) -> SolutionSummary:
    filename = request.param
    path = Path(__file__).parent / filename
    with open(path, "r") as file:
        data = json.load(file)
    ss = SolutionSummary.from_dict(data)
    return ss


# ==============================================================================
# Tests
# ==============================================================================


def test_example1_data_creation(example1_data):
    # Check the data
    assert example1_data is not None


# Next test is parametrized, which means that it is run several times with
# different values for the speed_level
@pytest.mark.parametrize("example1_solving_pars", [1, 2, 3], indirect=["example1_solving_pars"])
def test_example1_solving_config(example1_solving_pars):
    # Check the solving configuration
    speed, config = example1_solving_pars
    assert config is not None
    assert config.speed_level == speed


# Next test is parametrized, which means that it is run several times with
# different values for the speed_level
@pytest.mark.parametrize("example1_solving_pars", [1, 2, 3], indirect=["example1_solving_pars"])
def test_example1_solution_is_feasible(example1_solution):
    *_, solution = example1_solution

    # Check the solution is feasible
    assert solution.is_infeasible() == False


# Next test is parametrized, which means that it is run several times with
# different values for the speed_level
@pytest.mark.parametrize("example1_solving_pars", [1, 2, 3], indirect=["example1_solving_pars"])
def test_example1_solution_is_valid(example1_solution):
    fcma_problem, *_ = example1_solution

    # Check the solution has no contradictions
    slack = fcma_problem.check_allocation()


# ==============================================================================
# Test the solution provided by SolutionPrinter
# ==============================================================================


@pytest.mark.parametrize(
    "example1_solving_pars, example1_expected_vms_SolutionPrinter",
    [(1, "example1_expected_vms.json"), (2, "example1_expected_vms_v2.json")],
    indirect=["example1_solving_pars", "example1_expected_vms_SolutionPrinter"],
)
def test_example1_solution_printer_vms_and_prices(
    example1_solution, example1_expected_vms_SolutionPrinter
):
    *_, solution = example1_solution
    sp = SolutionPrinter(solution)

    # Another way, get the rich table and inspect the contents
    vm_table = sp._get_vm_table()
    # Organize by rows instead of columns
    solution_data = [col._cells for col in vm_table.columns]
    solution_data = [[*row] for row in zip(*solution_data)]
    assert len(solution_data) == len(example1_expected_vms_SolutionPrinter)
    for row in solution_data:
        assert row in example1_expected_vms_SolutionPrinter

    # TODO: Make visible the variables used by SolutionPrinter so that the values
    #       can be compared instead of the formatted strings? Or externalize to
    #       other callable methods the computation of the printed values?


@pytest.mark.parametrize(
    "example1_solving_pars, example1_expected_allocation_SolutionPrinter",
    [(1, "example1_expected_allocation.json"), (2, "example1_expected_allocation_v2.json")],
    indirect=["example1_solving_pars", "example1_expected_allocation_SolutionPrinter"],
)
def test_example1_solution_printer_apps_allocations(
    example1_solution, example1_expected_allocation_SolutionPrinter
):
    fcma_problem, _, solution = example1_solution
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


@pytest.mark.parametrize("example1_solving_pars", [1, 2, 3], indirect=["example1_solving_pars"])
def test_SolutionSummary_as_dict_and_back(example1_solution):
    *_, solution = example1_solution
    summary = SolutionSummary(solution)
    summary_dict = summary.as_dict()
    summary2 = SolutionSummary.from_dict(summary_dict)
    assert summary.get_vm_summary() == summary2.get_vm_summary()
    assert summary.get_all_apps_allocations() == summary2.get_all_apps_allocations()


# ==============================================================================
# Check example1 solutions with velocities 1, 2, and 3
# ==============================================================================


@pytest.mark.parametrize(
    "example1_solving_pars, expected_solution_summary",
    [
        (1, "example1_solution_velocity1.json"),
        (2, "example1_solution_velocity2.json"),
        (3, "example1_solution_velocity3.json"),
    ],
    indirect=["example1_solving_pars", "expected_solution_summary"],
)
def test_example1_SolutionSummary_vms(example1_solution, expected_solution_summary):
    *_, solution = example1_solution
    summary = SolutionSummary(solution)
    vm_alloc = summary.get_vm_summary()
    expected_vm_alloc = expected_solution_summary.get_vm_summary()
    assert vm_alloc == expected_vm_alloc


@pytest.mark.parametrize(
    "example1_solving_pars, expected_solution_summary",
    [
        (1, "example1_solution_velocity1.json"),
        (2, "example1_solution_velocity2.json"),
        (3, "example1_solution_velocity3.json"),
    ],
    indirect=["example1_solving_pars", "expected_solution_summary"],
)
def test_example1_SolutionSummary_all_apps(example1_solution, expected_solution_summary):
    *_, solution = example1_solution
    summary = SolutionSummary(solution)
    app_alloc = summary.get_all_apps_allocations()
    expected_app_alloc = expected_solution_summary.get_all_apps_allocations()
    assert app_alloc == expected_app_alloc


@pytest.mark.parametrize(
    "example1_solving_pars, expected_solution_summary",
    [
        (1, "example1_solution_velocity1.json"),
        (2, "example1_solution_velocity2.json"),
        (3, "example1_solution_velocity3.json"),
    ],
    indirect=["example1_solving_pars", "expected_solution_summary"],
)
def test_example1_SolutionSummary_single_app(example1_solution, expected_solution_summary):
    *_, solution = example1_solution
    summary = SolutionSummary(solution)
    expected_allocation = expected_solution_summary.get_all_apps_allocations()
    for app in expected_allocation:
        info = summary.get_app_allocation_summary(app)
        assert info == expected_allocation[app]


# # print("\n----------- Solution check --------------")
# # for attribute in dir(slack):
# #     if attribute.endswith("percentage"):
# #         print(f"{attribute}: {getattr(slack, attribute): .2f} %")
# # print("-----------------------------------------")
