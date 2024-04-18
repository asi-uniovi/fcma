# pylint: disable=redefined-outer-name
"""Code for fixtures used among different test files"""

import os
import importlib.util
from pathlib import Path
import json
import pytest
from pulp import PULP_CBC_CMD
from fcma import System, App, RequestsPerTime, SolvingPars, Fcma, Solution, SolutionSummary
from fcma.serialization import ProblemSerializer


@pytest.fixture(scope="module")
def aws_eu_west_1():
    """Imports definitions of instance classes and families for aws. These
    definitions are in ../examples"""
    path_to_module = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "aws_eu_west_1.py")
    )
    spec = importlib.util.spec_from_file_location("aws_eu_west_1", path_to_module)
    aws_eu_west_1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aws_eu_west_1)
    return aws_eu_west_1


@pytest.fixture(scope="module")
def example_data(aws_eu_west_1, request) -> tuple[System, dict[App, RequestsPerTime]]:
    """Defines a set of parameters to create a problem.

    Through request.param it receives the function to run which will
    return the required data"""
    example = request.param
    return example(aws_eu_west_1)


@pytest.fixture(scope="module")
def example_solving_pars(request) -> SolvingPars:
    """Allows the definiton of a SolvinPars object with a given speed (received
    through request.param).

    It returns a tuple with the requested speed and the SolvingPars object"""
    speed = request.param
    gap_rel = 0.05
    solver = PULP_CBC_CMD(msg=0, gapRel=gap_rel)
    solving_pars = SolvingPars(speed_level=speed, solver=solver)
    return speed, solving_pars


@pytest.fixture(scope="module")
def example_solution(example_data, example_solving_pars) -> tuple[Fcma, SolvingPars, Solution]:
    """Instantiates a Fcma problem and solves it.

    Returns a tuple with the problem, the solving parameters and the solution"""
    # Solve the allocation problem
    problem = Fcma(*example_data)
    solution = problem.solve(example_solving_pars[1])
    return problem, example_solving_pars, solution


@pytest.fixture(scope="module")
def expected_solution_summary(request) -> SolutionSummary:
    """Reads a solution summary from the given json file"""
    filename = request.param
    if not filename.endswith(".json"):
        filename += ".json"
    path = Path(__file__).parent / Path("expected_sols") / filename
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    ss = SolutionSummary.from_dict(data)
    return ss


@pytest.fixture(scope="module")
def problem(request) -> Fcma:
    """Reads a problem from the given json file"""
    filename = request.param
    path = Path(__file__).parent / Path("problems") / f"{filename}.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return ProblemSerializer.from_dict(data)
