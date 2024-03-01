"""
A simple example of how to use the Fcma class
"""
import logging
from cloudmodel.unified.units import (ComputationalUnits, RequestsPerTime, Storage)
from fcma import (App, AppFamilyPerf, Fcma, SolvingPars)
from fcma.visualization import SolutionPrinter
import aws_eu_west_1

# Set logging level
logging.basicConfig(level=logging.INFO)

# sfmpl is an optional parameter that stands for Single Failure Maximum Performnace Loss.
# For example, with sfml=0.5, FCMA does it best so that a single node failure does not cause an
# application performance loss higher than 50 %. SFMPL is a secondary requirement since cost is
# the most important requirement.
apps = {
    "appA": App(name="appA", sfmpl=0.5),
    "appB": App(name="appB", sfmpl=0.2),
    "appC": App(name="appC"),
    "appD": App(name="appD"),
}

workloads = {
    apps["appA"]: RequestsPerTime("6  req/s"),
    apps["appB"]: RequestsPerTime("12 req/s"),
    apps["appC"]: RequestsPerTime("20  req/s"),
    apps["appD"]: RequestsPerTime("15  req/s"),
}

# Computational parameters for pairs application and instance class family. Performance is assumed to be
# the same for all the instance classes in a family, whenever instance classes have enough CPU and memory.
# agg tuple provides valid replicas aggregations, i.e, aggregations that do not reduce
# performance. For example, agg = (2, 4, 10) allows the aggregation of 2, 4 or 10
# replicas to get one bigger aggregated replica with 2x, 4x, or 10x cores and performance.
# Aggregated replicas have the same memory requirement that one replica unless mem parameter
# is set to a tuple. For example, for agg=(2,) and mem=(Storage("500 mebibytes"), Storage("650 mebibytes")),
# a single replica requires 500 Mebibytes, but a 2x aggregated replica would require 650 Mebibytes.
app_family_perfs = {
    # For family aws_eu_west_1.c5_m5_r5_fm. It includes AWS c5, m5 and r5 instances
    (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("400 mcores"),
        mem=Storage("500 mebibytes"),
        perf=RequestsPerTime("0.4 req/s"),
        aggs=(2,)
    ),
    (apps["appB"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("80 mcores"),
        mem=Storage("200 mebibytes"),
        perf=RequestsPerTime("0.5 req/s"),
        aggs=(2, 4, 8, 12)
    ),
    (apps["appC"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("90 mcores"),
        mem=Storage("350 mebibytes"),
        perf=RequestsPerTime("0.2 req/s"),
        aggs=(2, 4, 10)
    ),
    (apps["appD"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("8500 mcores"),
        mem=Storage("25000 mebibytes"),
        perf=RequestsPerTime("1 req/s"),
    ),

    # For family aws_eu_west_1.c6g_m6g_r6g_fm. It includes AWS c6g, m6g and r6g instances
    (apps["appB"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
        cores=ComputationalUnits("100 mcores"),
        mem=Storage("250 mebibytes"),
        perf=RequestsPerTime("0.35 req/s"),
        aggs=(2, 4, 10)
    ),
    (apps["appC"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
        cores=ComputationalUnits("120 mcores"),
        mem=Storage("450 mebibytes"),
        perf=RequestsPerTime("0.4 req/s"),
        aggs=(2, 4, 8)
    ),
    (apps["appD"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
        cores=ComputationalUnits("6500 mcores"),
        mem=Storage("22000 mebibytes"),
        perf=RequestsPerTime("0.8 req/s"),
    ),
}

# Create an object for the FCMA problem
fcma_problem = Fcma(app_family_perfs, workloads=workloads)

# Three speed levels are possible: 1, 2 and 3, being speed level 1 the slowest, but the one
# giving the best cost results. Parameter partial_ilp_max_seconds is applicable only to speed_level=1
# and limits the time spent on solving the partial ILP problem.
solving_pars = SolvingPars(speed_level=3, partial_ilp_max_seconds=10)

# Solve the allocation problem
alloc, statistics = fcma_problem.solve(solving_pars)

# Print results
SolutionPrinter(alloc, statistics).print()

# Check the solution
slack = fcma_problem.check_allocation()
print("\n----------- Solution check --------------")
for attr_name in slack.__annotations__:
    print(f"{attr_name}: {slack.__getattribute__(attr_name): .2f} %")
print("-----------------------------------------")

