import pickle

from pint import set_application_registry
from cloudmodel.unified.units import ureg
from fcma.visualization import SolutionPrinter

set_application_registry(ureg)

# from fcma.visualization import ProblemPrinter

with open("fcma_bug_2_1_3_8_0.02.pkl", "rb") as f:
    problem = pickle.load(f)

solving_pars = problem._solving_pars
solution = problem.solve()

# Print results
SolutionPrinter(solution).print()

# Check the solution
slack = problem.check_allocation()
print("\n----------- Solution check --------------")
for attribute in dir(slack):
    if attribute.endswith("percentage"):
        print(f"{attribute}: {getattr(slack, attribute): .2f} %")
print("-----------------------------------------")

