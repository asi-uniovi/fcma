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
import pytest
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App, AppFamilyPerf, System, Fcma
from fcma.model import SolutionSummary
from .examples import example1, example2, example3, example4


# Global variables for the different combinations
examples = [example1, example2, example3, example4]
speeds = [1, 2, 3]
cases_with_solutions = [
    (ex, sp, f"{ex.__name__}_solution_speed_{sp}.json") for ex, sp in product(examples, speeds)
]


# Fixtures are in conftest.py

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
