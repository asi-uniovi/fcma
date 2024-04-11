"""Some smoke tests to check that SolutionSummary is stable while converting
to dict and back, and also with unfeasible solutions"""

from itertools import product
import pytest
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App, AppFamilyPerf, System, Fcma
from fcma.model import SolutionSummary
from .examples import example1, example2, example3, example4

examples = [example1, example2, example3, example4]
speeds = [1, 2, 3]

# ==============================================================================
# Smoke tests for SolutionSummary class
# ==============================================================================


@pytest.mark.smoke
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
@pytest.mark.smoke
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
