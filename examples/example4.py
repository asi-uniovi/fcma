"""
A simple example of how to use the Fcma class
"""

import logging
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
import aws_eu_west_1
from fcma import App, AppFamilyPerf, System, Fcma, SolvingPars
from fcma.visualization import SolutionPrinter

# Set logging level
logging.basicConfig(level=logging.INFO)

# sfmpl is an optional parameter that stands for Single Failure Maximum Performnace Loss.
# For example, with sfml=0.5, FCMA does it best so that a single node failure does not cause an
# application performance loss higher than 50 %. SFMPL is a secondary requirement since cost is
# the most important requirement.
apps = {
    "app_0": App(name="app_0"),
    "app_1": App(name="app_1"),
}

workloads = {
    apps["app_0"]: RequestsPerTime("1  req/s"),
    apps["app_1"]: RequestsPerTime("1  req/s"),
}

system: System = {
    # For family aws_eu_west_1.c5_m5_r5_fm. It includes AWS c5, m5 and r5 instances
    (apps["app_0"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("2.7 cores"),
        mem=Storage("4.05 gibibytes"),
        perf=RequestsPerTime("3.456 req/hour"),
        aggs=(2, ),
    ),
    (apps["app_1"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("2.025 cores"),
        mem=Storage("3.645 gibibytes"),
        perf=RequestsPerTime("173.88 req/hour"),
        aggs=(2, ),
    ),
}

# Create an object for the FCMA problem
fcma_problem = Fcma(system, workloads=workloads)

# Three speed levels are possible: 1, 2 and 3, being speed level 1 the slowest, but the one giving the best
# cost results. A solver with options can be passed for speed levels 1 and 2, or defaults are used. For instance:
#             from pulp import PULP_CBC_CMD
#             solver = PULP_CBC_CMD(timeLimit=10, gapRel=0.01, threads=8)
#             solving_pars = SolvingPars(speed_level=1, solver=solver)
# More information can be found on: https://coin-or.github.io/pulp/technical/solvers.html
solving_pars = SolvingPars(speed_level=1)

# Solve the allocation problem
solution = fcma_problem.solve(solving_pars)

# Print results
SolutionPrinter(solution).print()

# Check the solution
slack = fcma_problem.check_allocation()
print("\n----------- Solution check --------------")
for attribute in dir(slack):
    if attribute.endswith("percentage"):
        print(f"{attribute}: {getattr(slack, attribute): .2f} %")
print("-----------------------------------------")
