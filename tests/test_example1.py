"""
Some tests based on the example provided in examples folder
"""

import json
from pathlib import Path
import pytest
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App, AppFamilyPerf, System, Fcma, SolvingPars, Solution
from fcma.visualization import SolutionPrinter
import importlib.util
import os


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
def example1_data(aws_eu_west_1) -> Fcma:
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

    # Create an object for the FCMA problem
    fcma_problem = Fcma(system, workloads=workloads)
    return fcma_problem


@pytest.fixture(scope="module")
def example1_solving_pars() -> SolvingPars:
    """Defines parameters for solving the example 1"""

    # Three speed levels are possible: 1, 2 and 3, being speed level 1 the slowest, but the one giving the best
    # cost results. A solver with options can be passed for speed levels 1 and 2, or defaults are used. For instance:
    #             from pulp import PULP_CBC_CMD
    #             solver = PULP_CBC_CMD(timeLimit=10, gapRel=0.01, threads=8)
    #             solving_pars = SolvingPars(speed_level=1, solver=solver)
    # More information can be found on: https://coin-or.github.io/pulp/technical/solvers.html
    solving_pars = SolvingPars(speed_level=1)
    return solving_pars


@pytest.fixture(scope="module")
def example1_solution(example1_data, example1_solving_pars) -> tuple[Fcma, SolvingPars, Solution]:
    # Solve the allocation problem
    solution = example1_data.solve(example1_solving_pars)
    return example1_data, example1_solving_pars, solution


@pytest.fixture(scope="module")
def example1_expected_allocation() -> dict:
    path = Path(__file__).parent / "example1_expected_allocation.json"
    with open(path, "r") as file:
        data = json.load(file)
    # Organize by rows instead of columns
    for app in data:
        data[app] = [[*row] for row in zip(*data[app])]
    return data


@pytest.fixture(scope="module")
def example1_expected_vms() -> dict:
    path = Path(__file__).parent / "example1_expected_vms.json"
    with open(path, "r") as file:
        data = json.load(file)
    # Organize by rows instead of columns
    data = [[*row] for row in zip(*data)]
    return data


# ==============================================================================
# Tests
# ==============================================================================


def test_example1_data_creation(example1_data):
    # Check the data
    assert example1_data is not None


@pytest.mark.velocity1
def test_example1_solving_config(example1_solving_pars):
    # Check the solving configuration
    assert example1_solving_pars is not None
    assert example1_solving_pars.speed_level == 1


@pytest.mark.velocity1
def test_example1_solution_is_feasible(example1_solution, capsys):
    *_, solution = example1_solution
    # Print results
    sp = SolutionPrinter(solution.allocation, solution.statistics)

    # Check the solution is feasible
    assert sp._is_infeasible_sol() == False


@pytest.mark.velocity1
def test_example1_solution_is_valid(example1_solution):
    fcma_problem, _, solution = example1_solution
    # Print results
    sp = SolutionPrinter(solution.allocation, solution.statistics)

    # Check the solution has no contradictions
    slack = fcma_problem.check_allocation()


@pytest.mark.velocity1
@pytest.mark.skip(reason="Checking the printed output is no the best way")
def test_example1_printed_solution_vms_and_prices(example1_solution, capsys):
    *_, solution = example1_solution
    sp = SolutionPrinter(solution.allocation, solution.statistics)

    # Check the vms and prices

    # One way to check that: Capture the output printed by print_vms()
    # and make string comparisons with the expected output
    capsys.readouterr()  # Discard anything printed before
    sp.print_vms()
    output = capsys.readouterr()  # Get what was printed
    for line in output.out.split("\n"):
        if "c5.4xlarge" in line:
            assert "(x1)" in line
            assert "0.768 usd / hour" in line
        if "c6g.4xlarge" in line:
            assert "(x2)" in line
            assert "1.168 usd / hour" in line
        if "c6g.12xlarge" in line:
            assert "(x1)" in line
            assert "1.752 usd / hour" in line
        if "c6g.24xlarge" in line:
            assert "(x2)" in line
            assert "7.008 usd / hour" in line


@pytest.mark.velocity1
def test_example1_solution_vms_and_prices(example1_solution, example1_expected_vms):
    *_, solution = example1_solution
    sp = SolutionPrinter(solution.allocation, solution.statistics)

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


@pytest.mark.velocity1
def test_example1_solution_apps_allocations(example1_solution, example1_expected_allocation):
    fcma_problem, _, solution = example1_solution
    sp = SolutionPrinter(solution.allocation, solution.statistics)
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


# print("\n----------- Solution check --------------")
# for attribute in dir(slack):
#     if attribute.endswith("percentage"):
#         print(f"{attribute}: {getattr(slack, attribute): .2f} %")
# print("-----------------------------------------")
