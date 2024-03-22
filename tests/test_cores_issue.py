# pylint: disable=redefined-outer-name

"""Contain tests to check that having the cores in instance classes
expressed as floating point round values does not cause issues
in the solver."""

import json
from pathlib import Path
import pytest
from fcma.model import SolutionSummary
from fcma.serialization import ProblemSerializer
from fcma import Fcma, SolvingPars
from .utils_test import assert_dicts_almost_equal

@pytest.fixture(scope="module")
def problem(request) -> Fcma:
    """Reads the problem from json file"""
    filename = request.param
    path = Path(__file__).parent / Path("problems") / f"{filename}.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return ProblemSerializer.from_dict(data)


@pytest.fixture(scope="module")
def expected_solution(request) -> SolutionSummary:
    """Reads the expected solution from json file"""
    filename = request.param
    path = Path(__file__).parent / Path("expected_sols") / f"{filename}.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return SolutionSummary.from_dict(data)


@pytest.mark.parametrize(
    "problem, speed, expected_solution",
    [
        ("big_problem", 2, "big_problem_speed_2"),
    ],
    indirect=["problem", "expected_solution"],
)
def test_big_problem(problem, speed, expected_solution):
    """Test that the problem is solvable and the solution is the one expected"""
    solver = SolvingPars(speed_level=speed)
    sol = problem.solve(solver)
    assert sol is not None

    summary = SolutionSummary(sol)
    assert_dicts_almost_equal(summary.as_dict(), expected_solution.as_dict())
    