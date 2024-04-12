"""Helper to create the json files for the example problems"""

import json
from pathlib import Path
import importlib
from fcma.serialization import ProblemSerializer
from fcma import Fcma
from examples import example1, example2, example3, example4  # pylint: disable=no-name-in-module

example_list = [example1, example2, example3, example4]


def import_aws_definitions(path_to_module: Path):
    """Imports definitions of instance classes and families for aws. These
    definitions are in ../examples"""
    spec = importlib.util.spec_from_file_location("module", path_to_module.absolute())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_example_to_json(
    example_number: int, path: str = "problems", filename: str = "example{}_problem.json"
):
    """Generate a problem file for the example i"""
    aws_eu_west_1 = import_aws_definitions(
        Path(__file__).parent / ".." / "examples" / "aws_eu_west_1.py"
    )
    example = example_list[example_number]
    problem_params = example(aws_eu_west_1)
    problem = Fcma(*problem_params)
    serializer = ProblemSerializer(problem)
    data = serializer.as_dict()
    with open(Path(path) / filename.format(example_number + 1), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


if __name__ == "__main__":
    for i in range(len(example_list)):
        export_example_to_json(i)
