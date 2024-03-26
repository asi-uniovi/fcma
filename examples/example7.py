"""
A simple example of how to use the Fcma class
"""

import logging
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma.model import SolutionSummary
import aws_eu_west_1
from fcma import App, AppFamilyPerf, System, Fcma, SolvingPars
from fcma.visualization import SolutionPrinter
import sys

# Set logging level
logging.basicConfig(level=logging.INFO)

# sfmpl is an optional parameter that stands for Single Failure Maximum Performnace Loss.
# For example, with sfml=0.5, FCMA does it best so that a single node failure does not cause an
# application performance loss higher than 50 %. SFMPL is a secondary requirement since cost is
# the most important requirement.
apps = {
    "appA": App(name="app_0"),
    "appB": App(name="app_1"),
    "appC": App(name="app_2"),
    "appD": App(name="app_3"),
    "appA": App(name="app_4"),
    "appB": App(name="app_5"),
    "appC": App(name="app_6"),
    "appD": App(name="app_7"),
    "appA": App(name="app_8"),
    "appB": App(name="app_9"),
    "appC": App(name="app_10"),
    "appA": App(name="app_11"),
    "appB": App(name="app_12"),
    "appC": App(name="app_13"),
    "appD": App(name="app_14"),
    "appA": App(name="app_15"),
    "appB": App(name="app_16"),
    "appC": App(name="app_18"),
    "appD": App(name="app_19"),
    "appA": App(name="app_20"),
    "appA": App(name="app_21"),
    "appB": App(name="app_22"),
    "appC": App(name="app_23"),
    "appD": App(name="app_24"),
    "appA": App(name="app_25"),
    "appB": App(name="app_26"),
    "appC": App(name="app_27"),
    "appD": App(name="app_28"),
    "appA": App(name="app_29"),
}

workloads = {
    apps["app_0"]: RequestsPerTime("3600  req/hour"),
    apps["app_1"]: RequestsPerTime("3600  req/hour"),
    apps["app_2"]: RequestsPerTime("3600  req/hour"),
    apps["app_3"]: RequestsPerTime("3600  req/hour"),
    apps["app_4"]: RequestsPerTime("3600  req/hour"),
    apps["app_5"]: RequestsPerTime("3600  req/hour"),
    apps["app_6"]: RequestsPerTime("3600  req/hour"),
    apps["app_7"]: RequestsPerTime("3600  req/hour"),
    apps["app_8"]: RequestsPerTime("3600  req/hour"),
    apps["app_9"]: RequestsPerTime("3600  req/hour"),
    apps["app_10"]: RequestsPerTime("3600  req/hour"),
    apps["app_11"]: RequestsPerTime("3600  req/hour"),
    apps["app_12"]: RequestsPerTime("3600  req/hour"),
    apps["app_13"]: RequestsPerTime("3600  req/hour"),
    apps["app_14"]: RequestsPerTime("3600  req/hour"),
    apps["app_15"]: RequestsPerTime("3600  req/hour"),
    apps["app_16"]: RequestsPerTime("3600  req/hour"),
    apps["app_17"]: RequestsPerTime("3600  req/hour"),
    apps["app_18"]: RequestsPerTime("3600  req/hour"),
    apps["app_19"]: RequestsPerTime("3600  req/hour"),
    apps["app_20"]: RequestsPerTime("3600  req/hour"),
    apps["app_21"]: RequestsPerTime("3600  req/hour"),
    apps["app_22"]: RequestsPerTime("3600  req/hour"),
    apps["app_23"]: RequestsPerTime("3600  req/hour"),
    apps["app_24"]: RequestsPerTime("3600  req/hour"),
    apps["app_25"]: RequestsPerTime("3600  req/hour"),
    apps["app_26"]: RequestsPerTime("3600  req/hour"),
    apps["app_27"]: RequestsPerTime("3600  req/hour"),
    apps["app_28"]: RequestsPerTime("3600  req/hour"),
    apps["app_29"]: RequestsPerTime("3600  req/hour"),
}

# Computational parameters for pairs application and instance class family. Performance is assumed
# the same for all the instance classes in a family, whenever instance classes have enough CPU and memory.
# agg tuple provides valid replicas aggregations, i.e, aggregations that do not reduce
# performance. For example, agg = (2, 4, 10) allows the aggregation of 2, 4 or 10
# replicas to get one bigger aggregated replica with 2x, 4x, or 10x cores and performance.
# Aggregated replicas have the same memory requirement that one replica unless mem parameter
# is set to a tuple. For example, for agg=(2,) and mem=(Storage("500 mebibytes"), Storage("650 mebibytes")),
# a single replica requires 500 Mebibytes, but a 2x aggregated replica would require 650 Mebibytes.
system: System = {
    # For family aws_eu_west_1.c5_m5_r5_fm. It includes AWS c5, m5 and r5 instances
    (apps["app_0"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("400 mcores"),
        mem=Storage("500 mebibytes"),
        perf=RequestsPerTime("0.4 req/s"),
        aggs=(2,),
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


if "json" in sys.argv:
    import json
    ss = SolutionSummary(solution)
    with open("solution.json", "w") as file:
        file.write(json.dumps(ss.as_dict(), indent=2))