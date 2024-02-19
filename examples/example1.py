"""
A simple example of how to use the Fcma class
"""
import logging
from cloudmodel.unified.units import (ComputationalUnits, RequestsPerTime, Storage)
import fcma.AWS.aws_eu_west_1 as aws_eu_west_1
from fcma import (App, AppFamilyPerf, Fcma, SolvingPars)

# Set logging level
logging.basicConfig(level=logging.INFO)

apps = {
    "appA": App(name="appA", sfmpl=0.5),
    "appB": App(name="appB", sfmpl=0.2),
    "appC": App(name="appC"),
    "appD": App(name="appD"),
}

workloads = {
    apps["appA"]: RequestsPerTime("3  req/s"),
    apps["appB"]: RequestsPerTime("60 req/s"),
    apps["appC"]: RequestsPerTime("40  req/s"),
    apps["appD"]: RequestsPerTime("29  req/s"),
}

# Computational parameters for pairs application and instance class family
# maxagg limits the number of aggregated containers
app_family_perfs = {
    # For family aws_eu_west_1.c5_m5_r5_fm
    (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("400 mcores"),
        mem=Storage("500 mebibytes"),
        perf=RequestsPerTime("0.4 req/s"),
        agg=(2,)
    ),
    (apps["appB"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("80 mcores"),
        mem=Storage("200 mebibytes"),
        perf=RequestsPerTime("0.5 req/s"),
        agg=(2, 4, 8, 12)
    ),
    (apps["appC"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("90 mcores"),
        mem=Storage("350 mebibytes"),
        perf=RequestsPerTime("0.2 req/s"),
        agg=(2, 4, 10)
    ),
    (apps["appD"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("8500 mcores"),
        mem=Storage("25000 mebibytes"),
        perf=RequestsPerTime("1 req/s"),
    ),

    # For family aws_eu_west_1.c6g_m6g_r6g_fm
    (apps["appB"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
        cores=ComputationalUnits("100 mcores"),
        mem=Storage("250 mebibytes"),
        perf=RequestsPerTime("0.35 req/s"),
        agg=(2, 4, 10)
    ),
    (apps["appC"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
        cores=ComputationalUnits("120 mcores"),
        mem=Storage("450 mebibytes"),
        perf=RequestsPerTime("0.4 req/s"),
        agg=(2, 4, 8)
    ),
    (apps["appD"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
        cores=ComputationalUnits("6500 mcores"),
        mem=Storage("22000 mebibytes"),
        perf=RequestsPerTime("0.8 req/s"),
    ),
}

# Create an object for the FCMA problem
fcma_problem = Fcma(app_family_perfs, workloads=workloads)
solving_pars = SolvingPars(speed_level=3, partial_ilp_max_seconds=None)
fcma_problem.solve(solving_pars)
