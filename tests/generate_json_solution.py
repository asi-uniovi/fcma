"""Helper to create the json files for the solutions of the example problems"""

import json
from pathlib import Path
from fcma.model import Solution, SolutionSummary, SolvingPars
from fcma.serialization import ProblemSerializer
from fcma import Fcma
from pulp import PULP_CBC_CMD
import click


def load_problem(problem_name: str):
    """Reads a problem from the given json file"""

    path = Path("problems") / f"{problem_name}_problem.json"
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return ProblemSerializer.from_dict(data)


def solve_problem(problem: Fcma, speed: int, gap_rel=0.05):
    """Solves a problem with the given speed level"""
    solver = PULP_CBC_CMD(msg=0, gapRel=gap_rel)
    solving_pars = SolvingPars(speed_level=speed, solver=solver)
    solution = problem.solve(solving_pars)
    return solution


def serialize_solution(solution: Solution, filename: str):
    """Serializes the solution to a json file"""
    data = SolutionSummary(solution).as_dict()
    with open(f"{filename}.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


# Main function using click
@click.command()
@click.option("--problem", "-p", type=str, required=True, help="Problem to solve")
@click.option("--speed", "-s", type=int, default=1, help="Speed level to solve the problem")
@click.option("--gap", "-g", type=float, default=0.05, help="Gap for the solver")
def main(problem, speed, gap):
    """Main function to solve a problem and save the solution to a json file"""
    fcma = load_problem(problem)
    solution = solve_problem(fcma, speed, gap)
    serialize_solution(solution, f"{problem}_solution_speed_{speed}")

    print(
        f"Solution for {problem} with speed {speed} saved to {problem}_solution_speed_{speed}.json"
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
