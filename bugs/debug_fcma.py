import pickle
import json
from pulp import PULP_CBC_CMD  # type: ignore

from pint import set_application_registry
from cloudmodel.unified.units import ureg
from fcma.serialization import ProblemSerializer

set_application_registry(ureg)

import fcma

from fcma.visualization import ProblemPrinter

filename = "bugs/fcma_prob_78_30_4_3_8_0.02.pkl"
print(f"{filename}")
with open(filename, "rb") as f:
    problem = pickle.load(f)

ps = ProblemSerializer(problem)
d = ps.as_dict()

with open("bug.json", "w") as f:
     f.write(json.dumps(d, indent=2))

ProblemPrinter(problem).print()

problem.solve(fcma.SolvingPars(speed_level=2))
