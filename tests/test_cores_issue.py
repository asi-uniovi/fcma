# pylint: disable=redefined-outer-name

"""Contain tests to check that having the cores in instance classes
expressed as floating point round values does not cause issues
in the solver."""

import pytest
from fcma import SolutionSummary, SolvingPars
from .util_asserts import assert_dicts_almost_equal


@pytest.mark.slow
@pytest.mark.parametrize(
    "problem, speed, expected_solution_summary",
    [
        ("big_problem", 2, "big_problem_speed_2.json"),
        ("big_problem", 3, "big_problem_speed_3.json"),
    ],
    indirect=["problem", "expected_solution_summary"],
)
def test_big_problem(problem, speed, expected_solution_summary):
    """Test that the problem is solvable and the solution is the one expected"""
    solver = SolvingPars(speed_level=speed)
    sol = problem.solve(solver)
    assert sol is not None

    summary = SolutionSummary(sol)
    assert_dicts_almost_equal(summary.as_dict(), expected_solution_summary.as_dict())
