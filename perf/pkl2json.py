"""
Get FCMA pickle files from pkl directory and write as JSON files in json directory
"""

import json
import pathlib
import pickle
from fcma.serialization import ProblemSerializer
from pint import set_application_registry
from cloudmodel.unified.units import ureg

set_application_registry(ureg)

pkl_dir = "pkl_data"
json_dir = "json_data"

file_names = [f.name for f in pathlib.Path(pkl_dir).glob("*.pkl")]
for pkl_file_name in file_names:
    pkl_path = f"{pkl_dir}/{pkl_file_name}"
    json_path = f"{json_dir}/{pkl_file_name.split('.')[0]}.json"
    with open(pkl_path, "rb") as f:
        problem = pickle.load(f)
        serializer = ProblemSerializer(problem)
        problem_dict = serializer.as_dict()
        with open(json_path, "w") as json_f:
            json.dump(problem_dict, json_f)
