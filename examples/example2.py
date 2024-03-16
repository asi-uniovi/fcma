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
    "appA": App(name="appA"),
}

workloads = {
    apps["appA"]: RequestsPerTime("1  req/s"),
}

system: System = {
    # For family aws_eu_west_1.c5_m5_r5_fm. It includes AWS c5, m5 and r5 instances
    (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("120 mcores"),
        mem=Storage("1180 mebibytes"),
        perf=RequestsPerTime("0.0263 req/s"),
        aggs=(2, 4, 8),
    ),
    (apps["appA"], aws_eu_west_1.c6i_m6i_r6i_fm): AppFamilyPerf(
        cores=ComputationalUnits("120 mcores"),
        mem=Storage("1180 mebibytes"),
        perf=RequestsPerTime("0.0250 req/s"),
        aggs=(2, 4, 8),
    ),
    (apps["appA"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
        cores=ComputationalUnits("120 mcores"),
        mem=Storage("737 mebibytes"),
        perf=RequestsPerTime("0.0200 req/s"),
        aggs=(2, 4, 8),
    ),
    (apps["appA"], aws_eu_west_1.c7a_m7a_r7a_fm): AppFamilyPerf(
        cores=ComputationalUnits("120 mcores"),
        mem=Storage("1180 mebibytes"),
        perf=RequestsPerTime("0.0302 req/s"),
        aggs=(2, 4, 8),
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
solving_pars = SolvingPars(speed_level=2)

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
