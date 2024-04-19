"""
Get a JOSN file with a system definition and multiplies container cores and performance to fulfill:
- Container cores are the closest submultiple of instance class cores.
- Container cores are multiples each other.
- Multiple/submultiple relation = 2
New json files are stored with the same names in a different directory
"""

import json
import pathlib
import os
from math import ceil, log
from fcma.model import AppFamilyPerf
from fcma.serialization import ProblemSerializer

max_mul_cores = 1
json_dir = "json_data"
new_json_dir = f"new_json_data-{max_mul_cores}"


def create_or_clear_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        # If it exists, delete files within the directory
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print(f"Files within '{directory_path}' deleted.")


create_or_clear_directory(new_json_dir)
file_names = [f.name for f in pathlib.Path(json_dir).glob("*.json")]
for json_file_name in file_names:
    json_path = f"{json_dir}/{json_file_name}"
    new_json_path = f"{new_json_dir}/{json_file_name}"
    with open(json_path, "rb") as f:
        # Read the FCMA problem
        fcma_problem = ProblemSerializer.from_dict(json.load(f))
        new_system = {}
    for app_fm, perf in fcma_problem._system.items():
        app = app_fm[0]
        fm = app_fm[1]
        multiplier = 2 ** ceil(log(perf.cores.magnitude, 2)) / perf.cores.magnitude
        if perf.cores.magnitude * multiplier <= max_mul_cores + 0.000001:
            new_perf = AppFamilyPerf(
                cores=perf.cores*multiplier,
                mem=perf.mem,
                perf=perf.perf*multiplier,
                aggs=perf.aggs
            )
            new_system[(app, fm)] = new_perf
        else:
            new_system[(app, fm)] = perf
    fcma_problem._system = new_system
    serializer = ProblemSerializer(fcma_problem)
    with open(new_json_path, "w") as json_f:
        problem_dict = serializer.as_dict()
        json.dump(problem_dict, json_f)
        print(f"Writting file {new_json_path}")
