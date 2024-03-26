import pickle
from pulp import PULP_CBC_CMD  # type: ignore

from pint import set_application_registry
from cloudmodel.unified.units import ureg
import json
import fcma
from fcma.visualization import SolutionPrinter
from fcma.serialization import ProblemSerializer
from fcma.visualization import ProblemPrinter


path = "fcma_prob_48_15_1_0.json"
with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

fcma_problem = ProblemSerializer.from_dict(data)

#ProblemPrinter(fcma_problem).print()

solution = fcma_problem.solve(fcma.SolvingPars(speed_level=1))

# Print results
SolutionPrinter(solution).print()

# Check the solution
slack = fcma_problem.check_allocation()
print("\n----------- Solution check --------------")
for attribute in dir(slack):
    if attribute.endswith("percentage"):
        print(f"{attribute}: {getattr(slack, attribute): .2f} %")
print("-----------------------------------------")
