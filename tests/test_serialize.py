"""Test for the ProblemSerializer class"""

import pytest
from fcma import SolutionSummary, Fcma, SolvingPars
from fcma.serialization import ProblemSerializer
from .examples import example1, example2, example3, example4
from .util_asserts import assert_dicts_almost_equal


def test_example3_as_dict(aws_eu_west_1):
    """Checks that the conversion to dict of example3 contains
    the expected values in some of the keys."""
    data = example3(aws_eu_west_1)
    problem = Fcma(*data)
    serializer = ProblemSerializer(problem)
    pr_dict = serializer.as_dict()
    assert pr_dict["units"] == {
        "cpu": "mcores",
        "cost/t": "usd/h",
        "workload": "req/h",
        "mem": "GiB",
    }
    wl = pr_dict["workloads"]
    assert len(wl) == 2
    assert all(w == 3600 for w in wl.values())

    sys = pr_dict["system"]
    assert len(sys["apps"]) == 2
    assert set(sys["apps"].keys()) == set(["app_0", "app_1"])

    assert sys["families"] == {
        "c5_m5_r5": [],
        "c5": ["c5_m5_r5"],
        "m5": ["c5_m5_r5"],
        "r5": ["c5_m5_r5"],
    }

    assert len(sys["instance_classes"]) == 24


@pytest.mark.parametrize(
    "example_data", [example1, example2, example3, example4], indirect=["example_data"]
)
def test_example_as_dict_and_back_solution_matches(example_data):
    """Tests that converting a problem to dict and back to a problem
    yields the same solution than the original problem"""
    orig_problem = Fcma(*example_data)
    serializer = ProblemSerializer(orig_problem)
    pr_as_dict = serializer.as_dict()
    new_problem = ProblemSerializer.from_dict(pr_as_dict)

    orig_sol = orig_problem.solve()
    new_sol = new_problem.solve()

    orig_sol_summary = SolutionSummary(orig_sol).as_dict()
    new_sol_summary = SolutionSummary(new_sol).as_dict()

    assert orig_sol_summary == new_sol_summary


@pytest.mark.parametrize(
    "problem, expected_solution_summary",
    [(f"example{i}_problem", f"example{i}_solution_speed_1") for i in range(1, 5)],
    indirect=["problem", "expected_solution_summary"],
)
def test_examples_from_json_solution_as_expected(problem, expected_solution_summary):
    """Gets examples defined from json and solves them, comparing the solution
    with the expected one"""
    solver = SolvingPars(speed_level=1)
    sol = problem.solve(solver)
    assert sol is not None

    summary = SolutionSummary(sol)
    assert_dicts_almost_equal(summary.as_dict(), expected_solution_summary.as_dict())
