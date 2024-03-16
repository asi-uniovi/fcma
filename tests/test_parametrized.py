"""
Some tests based on the example provided in examples folder
"""

import json
from pathlib import Path
import pytest
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
def example1_expected_vms(request) -> list:
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
def example1_expected_allocation(request) -> dict:
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
def expected_allocation(request) -> dict:
    if request.param == 1:
        expected_allocation = {  # num_cgroups, total_replicas, total_perf
            "appA": (2, 8, 6),
            "appB": (2, 3, 12.5),
            "appC": (6, 9, 20),
            "appD": (5, 19, 15.2),
        }
    elif request.param == 2:
        expected_allocation = {  # num_cgroups, total_replicas, total_perf
            "appA": (2, 8, 6),
            "appB": (1, 2, 12),
            "appC": (6, 9, 20),
            "appD": (5, 19, 15.2),
        }
    else:
        raise NotImplementedError
    return expected_allocation


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
    fcma_problem, _, solution = example1_solution
    # Print results
    sp = SolutionPrinter(solution)

    # Check the solution has no contradictions
    slack = fcma_problem.check_allocation()


# Next test is parametrized, which means that it is run several times with
# different values for the speed_level, and for each speed level it uses
# a different file for expected results
@pytest.mark.parametrize(
    "example1_solving_pars, example1_expected_vms",
    [(1, "example1_expected_vms.json"), (2, "example1_expected_vms_v2.json")],
    indirect=["example1_solving_pars", "example1_expected_vms"],
)
def test_example1_solution_vms_and_prices(example1_solution, example1_expected_vms):
    *_, solution = example1_solution
    sp = SolutionPrinter(solution)

    # Another way, get the rich table and inspect the contents
    vm_table = sp._get_vm_table()
    # Organize by rows instead of columns
    solution_data = [col._cells for col in vm_table.columns]
    solution_data = [[*row] for row in zip(*solution_data)]
    assert len(solution_data) == len(example1_expected_vms)
    for row in solution_data:
        assert row in example1_expected_vms

    # TODO: Make visible the variables used by SolutionPrinter so that the values
    #       can be compared instead of the formatted strings? Or externalize to
    #       other callable methods the computation of the printed values?


@pytest.mark.parametrize(
    "example1_solving_pars, example1_expected_allocation",
    [(1, "example1_expected_allocation.json"), (2, "example1_expected_allocation_v2.json")],
    indirect=["example1_solving_pars", "example1_expected_allocation"],
)
def test_example1_solution_apps_allocations(example1_solution, example1_expected_allocation):
    fcma_problem, _, solution = example1_solution
    sp = SolutionPrinter(solution)
    apps_allocations = sp._get_app_tables()

    # The apps in the solution must match the apps in the problem
    problem_apps = set(app.name for app in fcma_problem._workloads.keys())
    solution_apps = set(apps_allocations.keys())
    assert problem_apps == solution_apps

    def check_app_alloc(app, sol_data):
        expected_alloc = example1_expected_allocation[app]
        assert len(sol_data) == len(expected_alloc)
        for col in sol_data:
            assert col in expected_alloc

    # Check the allocations for each app
    for app, table in apps_allocations.items():
        solution_data = [col._cells for col in table.columns]
        solution_data = [[*row] for row in zip(*solution_data)]
        check_app_alloc(app, solution_data)


@pytest.mark.parametrize("example1_solving_pars", [1, 2], indirect=["example1_solving_pars"])
def test_AllocationSummary_vms(example1_solution):
    *_, solution = example1_solution
    summary = SolutionSummary(solution)
    vm_alloc = summary.get_vm_summary()
    assert vm_alloc.total_num == 6
    assert vm_alloc.cost.magnitude == pytest.approx(10.696, abs=1e-4)


@pytest.mark.parametrize(
    "example1_solving_pars, expected_allocation",
    [(1,1), (2,2)],
    indirect=["example1_solving_pars", "expected_allocation"],
)
def test_AllocationSummary_apps(example1_solution, expected_allocation):
    *_, solution = example1_solution
    summary = SolutionSummary(solution)
    app_alloc = summary.get_all_apps_allocations()
    assert len(app_alloc) == 4
    for app, info in app_alloc.items():
        assert len(info.container_groups) == expected_allocation[app][0]
        assert info.total_replicas == expected_allocation[app][1]
        assert info.total_perf.m_as("req/s") == pytest.approx(expected_allocation[app][2])


# # print("\n----------- Solution check --------------")
# # for attribute in dir(slack):
# #     if attribute.endswith("percentage"):
# #         print(f"{attribute}: {getattr(slack, attribute): .2f} %")
# # print("-----------------------------------------")
