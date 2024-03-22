"""Provides functions to define examples in code"""
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App, AppFamilyPerf, System


def example1(aws_eu_west_1):
    "Example 1"
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

    system: System = {
        # For family aws_eu_west_1.c5_m5_r5_fm. It includes AWS c5, m5 and r5 instances
        (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("400 mcores"),
            mem=Storage("500 mebibytes"),
            perf=RequestsPerTime("0.4 req/s"),
            aggs=(2,),
        ),
        (apps["appB"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("80 mcores"),
            mem=Storage("200 mebibytes"),
            perf=RequestsPerTime("0.5 req/s"),
            aggs=(2, 4, 8, 12),
        ),
        (apps["appC"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("90 mcores"),
            mem=Storage("350 mebibytes"),
            perf=RequestsPerTime("0.2 req/s"),
            aggs=(2, 4, 10),
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
            aggs=(2, 4, 10),
        ),
        (apps["appC"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
            cores=ComputationalUnits("120 mcores"),
            mem=Storage("450 mebibytes"),
            perf=RequestsPerTime("0.4 req/s"),
            aggs=(2, 4, 8),
        ),
        (apps["appD"], aws_eu_west_1.c6g_m6g_r6g_fm): AppFamilyPerf(
            cores=ComputationalUnits("6500 mcores"),
            mem=Storage("22000 mebibytes"),
            perf=RequestsPerTime("0.8 req/s"),
        ),
    }

    return system, workloads


def example2(aws_eu_west_1):
    "Example 2"
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
    return system, workloads


def example3(aws_eu_west_1):
    "Example 3"
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
            cores=ComputationalUnits("3.45 cores"),
            mem=Storage("33.12 gibibytes"),
            perf=RequestsPerTime("13.248 req/hour"),
            aggs=(2,),
        ),
        (apps["app_1"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("2.7 cores"),
            mem=Storage("19.44 gibibytes"),
            perf=RequestsPerTime("216.0 req/hour"),
            aggs=(2,),
        ),
    }
    return system, workloads


def example4(aws_eu_west_1):
    "Example 4"
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
            aggs=(2,),
        ),
        (apps["app_1"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
            cores=ComputationalUnits("2.025 cores"),
            mem=Storage("3.645 gibibytes"),
            perf=RequestsPerTime("173.88 req/hour"),
            aggs=(2,),
        ),
    }
    return system, workloads
