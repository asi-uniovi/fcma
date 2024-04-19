# pylint: disable=redefined-outer-name

"""Contain tests to check that having the cores in instance classes
expressed as floating point round values does not cause issues
in the solver."""

from pathlib import Path
import copy
import json
import pytest
from fcma import SolutionSummary, SolvingPars
from fcma.serialization import ProblemSerializer
from .util_asserts import assert_dicts_almost_equal


@pytest.mark.slow
def test_cores_as_integers_or_as_floats_produce_same_solution():
    """Tests that having the cores in instance classes expressed as
    floating point round values does not cause issues in the solver, and
    the solution is the same for both cases."""
    filename = "big_problem"
    path = Path(__file__).parent / Path("problems") / f"{filename}.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    int_data = copy.deepcopy(data)
    for ic in data["system"]["instance_classes"].values():
        ic["cores"] = int(ic["cores"])
    float_problem = ProblemSerializer.from_dict(data)
    int_problem = ProblemSerializer.from_dict(int_data)
    solver = SolvingPars(speed_level=2)
    float_sol = float_problem.solve(solver)
    int_sol = int_problem.solve(solver)

    float_summary = SolutionSummary(float_sol)
    int_summary = SolutionSummary(int_sol)
    assert_dicts_almost_equal(float_summary.as_dict(), int_summary.as_dict())
