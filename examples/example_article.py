"""
A simple example of how to use the Fcma class
"""

import logging
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma.model import SolutionSummary
import ics_article
from fcma import App, AppFamilyPerf, System, Fcma, SolvingPars
from fcma.visualization import SolutionPrinter
from pulp import PULP_CBC_CMD
import sys

# Set logging level
logging.basicConfig(level=logging.INFO)

# sfmpl is an optional parameter that stands for Single Failure Maximum Performnace Loss.
# For example, with sfml=0.5, FCMA does it best so that a single node failure does not cause an
# application performance loss higher than 50 %. SFMPL is a secondary requirement since cost is
# the most important requirement.
apps = {
    "app1": App(name="app1", sfmpl=0.5),
    "app2": App(name="app2", sfmpl=0.4),
    "app3": App(name="app3", sfmpl=1.0),
}

workloads = {
    apps["app1"]: RequestsPerTime("3  req/s"),
    apps["app2"]: RequestsPerTime("60 req/s"),
    apps["app3"]: RequestsPerTime("45  req/s"),
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
    (apps["app1"], ics_article.A_fm): AppFamilyPerf(
        cores=ComputationalUnits("600 mcores"),
        mem=Storage("0.95 gibibytes"),
        perf=RequestsPerTime("0.50 req/s"),
        aggs=(2,4),
    ),
    (apps["app2"], ics_article.A_fm): AppFamilyPerf(
        cores=ComputationalUnits("5000 mcores"),
        mem=Storage("17.45 gibibytes"),
        perf=RequestsPerTime("0.40 req/s"),
        aggs=(1,),
    ),
    (apps["app3"], ics_article.A_fm): AppFamilyPerf(
        cores=ComputationalUnits("1500 mcores"),
        mem=Storage("8.2 gibibytes"),
        perf=RequestsPerTime("2 req/s"),
        aggs=(2,),
    ),
    (apps["app1"], ics_article.B_fm): AppFamilyPerf(
        cores=ComputationalUnits("800 mcores"),
        mem=Storage("0.80 gibibytes"),
        perf=RequestsPerTime("0.40 req/s"),
        aggs=(2,4),
    ),
    (apps["app2"], ics_article.B_fm): AppFamilyPerf(
        cores=ComputationalUnits("7600 mcores"),
        mem=Storage("15.10 gibibytes"),
        perf=RequestsPerTime("3 req/s"),
        aggs=(1,),
    ),
    (apps["app3"], ics_article.B_fm): AppFamilyPerf(
        cores=ComputationalUnits("1200 mcores"),
        mem=Storage("6.40 gibibytes"),
        perf=RequestsPerTime("2.50 req/s"),
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
gap_rel = 0.05
solver = PULP_CBC_CMD(msg=0, gapRel=gap_rel)
solving_pars = SolvingPars(speed_level=1, solver=solver)

# Solve the allocation problem
solution = fcma_problem.solve(solving_pars)

# Update the metrics for secondary optimization objectives
solution.statistics.update_metrics(solution.allocation)

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