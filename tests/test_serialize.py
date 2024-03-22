"""Test for the ProblemSerializer class"""

import pytest
from fcma import SolutionSummary, Fcma
from fcma.serialization import ProblemSerializer
from .examples import example1, example2, example3, example4


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
