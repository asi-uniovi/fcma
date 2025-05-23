# pylint: disable=redefined-outer-name
"""Some tests that check that the solution of the example1, as shown
by SolutionPrinter is as expected"""

import json
from pathlib import Path
import pytest
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App, AppFamilyPerf, System, Fcma
from fcma.visualization import SolutionPrinter
from .examples import example1

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


# ==================================================================================
# Test the solution for example1 as provided by SolutionPrinter, for speeds 1 and 2
# ==================================================================================


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
